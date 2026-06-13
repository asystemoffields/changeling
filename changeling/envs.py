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

from . import OBS_DIM


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


ENVS = {"bandit": bandit_env, "catch": catch_env}
