"""One lifetime: RL^2 protocol. Memory persists across episodes within a
lifetime; env auto-resets on done (same task). Controls (PREREG_P0):
  c4 — blind the agent's reward *input* with a fair coin (env reward still
       scored); in-context learning should collapse where reward is the only cue.
  c5 — reset hidden state on every episode boundary; memory is the only
       learning mechanism, so this is the no-memory control.
"""
import jax
import jax.numpy as jnp

from . import N_ACT
from .agent import gru_step, hidden_size


def rollout(env, params, task, key, c4=False, c5=False):
    """Returns (rewards, metrics, dones) arrays of shape (T,)."""
    T = env["T"]
    key, k0 = jax.random.split(key)
    state0, obs0 = env["reset"](k0, task)
    h0 = jnp.zeros(hidden_size(params))
    carry0 = (h0, state0, obs0, jnp.zeros(N_ACT), jnp.float32(0.0),
              jnp.float32(1.0), key)

    def step_fn(carry, _):
        h, state, obs, last_a, last_r, boundary, key = carry
        key, ka, kr, kres, kc4 = jax.random.split(key, 5)
        r_in = jax.random.bernoulli(kc4, 0.5).astype(jnp.float32) if c4 else last_r
        x = jnp.concatenate([obs, last_a, jnp.array([r_in, boundary])])
        h, logits = gru_step(params, h, x)
        a = jax.random.categorical(ka, logits)
        state, obs, r, done, metric = env["step"](state, a, kr, task)
        # auto-reset on episode end, same task
        rs_state, rs_obs = env["reset"](kres, task)
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, new, old), rs_state, state)
        obs = jnp.where(done, rs_obs, obs)
        if c5:
            h = h * (1.0 - done.astype(jnp.float32))
        carry = (h, state, obs, jax.nn.one_hot(a, N_ACT), r,
                 done.astype(jnp.float32), key)
        return carry, (r, metric, done.astype(jnp.float32))

    _, (rewards, metrics, dones) = jax.lax.scan(step_fn, carry0, None, length=T)
    return rewards, metrics, dones


def late_weighted_fitness(rewards):
    """Fitness weights ramp 0.5 -> 1.5 across the lifetime (selects for
    improvement, not just performance). PREREG_P0 locked."""
    T = rewards.shape[0]
    w = jnp.linspace(0.5, 1.5, T)
    return jnp.sum(w * rewards) / jnp.sum(w)


def q4_fitness(rewards):
    """Final-quarter reward only: early steps are fitness-free, so
    exploration costs nothing. Anti-settling repricing (exploratory;
    not the locked P0 gate fitness)."""
    T = rewards.shape[0]
    return rewards[-(T // 4):].mean()


FITNESS = {"late": late_weighted_fitness, "q4": q4_fitness}
