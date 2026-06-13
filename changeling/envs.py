"""Phase 0 environments in pure JAX.

An env is a dict of pure functions sharing the fixed interface protocol
(SPEC section 2): obs padded to OBS_DIM, actions in [0, N_ACT).

  sample_task(key) -> task            (per-lifetime parameters)
  reset(key, task) -> (state, obs)    (per-episode)
  step(state, a, key, task) -> (state, obs, reward, done, metric)

`metric` is the env's gate statistic (bandit: pulled best arm; catch: success,
nonzero only on terminal steps). T is lifetime length in steps.
"""
import jax
import jax.numpy as jnp

from . import OBS_DIM, N_ACT


def bandit_env(n_arms=8, lifetime=200, needle=False, mix=0.0):
    """Bernoulli bandit. Default: p_i ~ U(0,1) per lifetime (forgiving —
    second-best is typically close to best, so satisficing nearly pays).
    needle=True: one arm at 0.9, the rest at 0.1 — finding the needle is
    constitutive of success; settling pays 0.1. mix=p: each lifetime is a
    needle task w.p. p, else uniform (curriculum contrast). One-step episodes."""
    assert 2 <= n_arms <= 8

    def sample_task(key):
        if needle:
            pos = jax.random.randint(key, (), 0, n_arms)
            return jnp.where(jnp.arange(n_arms) == pos, 0.9, 0.1)
        return jax.random.uniform(key, (n_arms,))

    if mix > 0:
        base = sample_task

        def sample_task(key):  # noqa: F811 — mixture wraps the base sampler
            ku, kn, kc = jax.random.split(key, 3)
            pos = jax.random.randint(kn, (), 0, n_arms)
            needle_p = jnp.where(jnp.arange(n_arms) == pos, 0.9, 0.1)
            return jnp.where(jax.random.bernoulli(kc, mix),
                             needle_p, base(ku))

    def reset(key, task):
        return (), jnp.zeros(OBS_DIM)

    def step(state, a, key, task):
        valid = a < n_arms                      # SPEC §2: surplus actions = no-op
        a_eff = jnp.where(valid, a, 0)          # safe gather index when invalid
        pull = jax.random.bernoulli(key, task[a_eff]).astype(jnp.float32)
        r = jnp.where(valid, pull, 0.0)         # a no-op earns nothing
        best = (a_eff == jnp.argmax(task)).astype(jnp.float32)
        metric = jnp.where(valid, best, 0.0)    # and never counts as best-arm
        return (), jnp.zeros(OBS_DIM), r, jnp.bool_(True), metric

    return dict(sample_task=sample_task, reset=reset, step=step,
                T=lifetime, name=f"bandit{n_arms}")


def catch_env(episodes=32):
    """bsuite-style Catch: 10x5 board, ball falls one row/step, paddle on the
    bottom row. +1 catch / -1 miss on the terminal step. 9-step episodes."""
    ROWS, COLS = 10, 5

    def _obs(state):
        ball_row, ball_col, paddle = state
        grid = jnp.zeros((ROWS, COLS))
        grid = grid.at[ball_row, ball_col].set(1.0)
        grid = grid.at[ROWS - 1, paddle].set(1.0)
        flat = grid.reshape(-1)
        return jnp.concatenate([flat, jnp.zeros(OBS_DIM - flat.shape[0])])

    def sample_task(key):
        return jnp.zeros(())  # no task variation in P0

    def reset(key, task):
        col = jax.random.randint(key, (), 0, COLS)
        state = (jnp.int32(0), col, jnp.int32(COLS // 2))
        return state, _obs(state)

    def step(state, a, key, task):
        ball_row, ball_col, paddle = state
        move = jnp.where(a == 0, -1, jnp.where(a == 2, 1, 0))  # 3..7 -> stay
        paddle = jnp.clip(paddle + move, 0, COLS - 1)
        ball_row = ball_row + 1
        done = ball_row >= ROWS - 1
        caught = (ball_col == paddle).astype(jnp.float32)
        r = jnp.where(done, 2.0 * caught - 1.0, 0.0)
        state = (ball_row, ball_col, paddle)
        return state, _obs(state), r, done, jnp.where(done, caught, 0.0)

    return dict(sample_task=sample_task, reset=reset, step=step,
                T=episodes * (ROWS - 1), name="catch")


def cbandit_env(n_arms=8, lifetime=256, C_ctx=5, frozen_rule=True, rule_seed=0,
                mix=0.0, hi_std=0.8, lo_std=0.2, hi_needle=0.95, lo_needle=0.05):
    """Contextual bandit (PREREG_P1 §1, the PRIMARY env). Each one-step trial
    draws a context c ~ U{0..C-1}; obs = onehot(c) padded to OBS_DIM. The correct
    arm is an injective map y(c): C->K (C_ctx<=n_arms). Pulling y(c) pays `hi`,
    else `lo`; the curriculum mixes a sharp "needle" margin (0.95/0.05) w.p. `mix`
    else standard (0.8/0.2). metric = pulled-correct-arm (the gate statistic).

    Two modes (PREREG_P1 B3):
      frozen_rule=True  (cbandit-FR, HEADLINE) — y is FIXED, seeded once from
        `rule_seed`, SHARED across every lifetime and eval. The only per-lifetime
        unknown is the interface (P,π) applied in rollout: the within-lifetime
        slope is pure interface-inference speed.
      frozen_rule=False (cbandit-FG, co-gated) — y resampled per lifetime ⇒
        genuine in-context task-learning under randomization.

    The env is interface-AGNOSTIC: obs-projection P and action-permutation π are
    applied by rollout/collect (B2). step() scores whatever arm it receives, so
    the metric is computed on the post-permutation arm (what was actually pulled).
    """
    assert 2 <= n_arms <= N_ACT
    assert 2 <= C_ctx <= n_arms          # injective map needs C <= K
    assert C_ctx <= OBS_DIM

    def _rule(key):
        # injective C->K: first C entries of a random arm permutation
        return jax.random.permutation(key, n_arms)[:C_ctx]

    Y_FIXED = _rule(jax.random.PRNGKey(rule_seed))   # FR closure constant

    def sample_task(key):
        ky, kn = jax.random.split(key)
        y = Y_FIXED if frozen_rule else _rule(ky)
        needle = jax.random.bernoulli(kn, mix)
        hi = jnp.where(needle, hi_needle, hi_std)
        lo = jnp.where(needle, lo_needle, lo_std)
        return dict(y=y, hi=hi, lo=lo)

    def reset(key, task):
        c = jax.random.randint(key, (), 0, C_ctx)
        obs = jnp.zeros(OBS_DIM).at[c].set(1.0)
        return c, obs

    def step(state, a, key, task):
        c = state
        valid = a < n_arms                       # SPEC §2: surplus actions = no-op
        a_eff = jnp.where(valid, a, 0)
        is_correct = jnp.logical_and(valid, a_eff == task["y"][c])
        p = jnp.where(is_correct, task["hi"], task["lo"])
        r = jnp.where(valid, jax.random.bernoulli(key, p).astype(jnp.float32), 0.0)
        metric = is_correct.astype(jnp.float32)
        # one-step episode (done every trial); rollout auto-resets a new context
        return c, jnp.zeros(OBS_DIM), r, jnp.bool_(True), metric

    return dict(sample_task=sample_task, reset=reset, step=step,
                T=lifetime, name=f"cbandit{'FR' if frozen_rule else 'FG'}")


ENVS = {"bandit": bandit_env, "catch": catch_env, "cbandit": cbandit_env}
