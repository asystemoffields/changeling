"""Route R1: OpenES, written out in full (no evosax — PREREG_P0 amendment).

Mirrored sampling, centered-rank fitness shaping, Adam ascent. Common task
seeds AND rollout randomness across the population each generation (full
antithetic CRN: mirror pairs differ only in theta, so the dominant noise term
— the Bernoulli reward draws — cancels in F+ - F-).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .rollout import rollout, late_weighted_fitness


def centered_ranks(f):
    ranks = jnp.argsort(jnp.argsort(f)).astype(jnp.float32)
    return ranks / (f.shape[0] - 1) - 0.5


def adam_init(dim):
    return dict(m=jnp.zeros(dim), v=jnp.zeros(dim), t=jnp.int32(0))


def adam_ascend(theta, grad, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = b1 * st["m"] + (1 - b1) * grad
    v = b2 * st["v"] + (1 - b2) * grad ** 2
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    theta = theta + lr * mhat / (jnp.sqrt(vhat) + eps)
    return theta, dict(m=m, v=v, t=t)


def make_gen_step(env, unravel, pop, n_lifetimes, sigma, lr,
                  fitness_fn=late_weighted_fitness, step=None, ponder_cost=0.0):
    """Returns jitted (theta, adam_state, key) -> (theta', adam_state', stats).

    `step` is a B1 step_fn (looped.make_step); None -> legacy gru. `ponder_cost`
    is the −c·mean_t E[K] compute tax folded into ES fitness (the adaptive-K
    pressure). ponder_cost=0.0 + legacy step keeps member_fitness bitwise-identical
    to the pre-B1 harness (e_k≡1, and f − 0.0·1.0 == f), so G0-A reproduces."""
    assert pop % 2 == 0

    def member_fitness(theta_flat, tasks, roll_keys):
        params = unravel(theta_flat)

        def one(task, k):
            rewards, _, _, e_ks = rollout(env, params, task, k, step=step)
            return fitness_fn(rewards) - ponder_cost * e_ks.mean()

        return jax.vmap(one)(tasks, roll_keys).mean()

    def gen_step(theta, adam_state, key):
        key, ke, kt, kr = jax.random.split(key, 4)
        dim = theta.shape[0]
        eps = jax.random.normal(ke, (pop // 2, dim))
        thetas = jnp.concatenate([theta + sigma * eps, theta - sigma * eps])
        tasks = jax.vmap(env["sample_task"])(jax.random.split(kt, n_lifetimes))
        # shared across the population (hence across each antithetic pair): the
        # only difference between F+ and F- is then the theta perturbation
        roll_keys = jax.random.split(kr, n_lifetimes)
        fits = jax.vmap(member_fitness, in_axes=(0, None, None))(
            thetas, tasks, roll_keys)
        shaped = centered_ranks(fits)
        grad = jnp.concatenate([eps, -eps]).T @ shaped / (pop * sigma)
        theta, adam_state = adam_ascend(theta, grad, adam_state, lr)
        stats = dict(fit_mean=fits.mean(), fit_max=fits.max(), fit_std=fits.std())
        return theta, adam_state, stats

    return jax.jit(gen_step)


def flatten_params(params):
    theta, unravel = ravel_pytree(params)
    return theta, unravel
