"""Gate metrics on held-out lifetimes, plus standing controls C4/C5.

Held-out = eval key space disjoint from training (EVAL_FOLD offset).
Reported per condition:
  reward_q1/q4 — first/final-quarter mean reward
  slope        — q4 - q1 (within-lifetime improvement, the primary endpoint)
  gate_q4      — final-quarter gate metric (bandit: best-arm rate;
                 catch: episode success rate among completed episodes)
"""
import jax
import jax.numpy as jnp

from .rollout import rollout

EVAL_FOLD = 10_000_000


def eval_suite(env, params, n=100, seed=0, c4=False, c5=False):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), EVAL_FOLD)
    kt, kr = jax.random.split(key)
    tasks = jax.vmap(env["sample_task"])(jax.random.split(kt, n))
    keys = jax.random.split(kr, n)

    def one(task, k):
        return rollout(env, params, task, k, c4=c4, c5=c5)

    rewards, metrics, dones = jax.vmap(one)(tasks, keys)  # (n, T)
    T = rewards.shape[1]
    q = T // 4
    q1, q4 = rewards[:, :q], rewards[:, -q:]
    # gate metric averaged over completed episodes in the final quarter
    m4, d4 = metrics[:, -q:], dones[:, -q:]
    gate_q4 = jnp.sum(m4 * d4) / jnp.maximum(jnp.sum(d4), 1.0)
    # per-lifetime slope sign test (Gate 1 locked statistic; one-sided
    # binomial, ties dropped)
    slopes = q4.mean(axis=1) - q1.mean(axis=1)
    n_pos = int((slopes > 0).sum())
    n_eff = int((slopes != 0).sum())
    return dict(
        reward_q1=float(q1.mean()),
        reward_q4=float(q4.mean()),
        slope=float(q4.mean() - q1.mean()),
        slope_pos_frac=n_pos / max(n_eff, 1),
        slope_sign_p=_binom_tail(n_pos, n_eff),
        gate_q4=float(gate_q4),
    )


def _binom_tail(k, n):
    """P(X >= k), X ~ Binomial(n, 0.5)."""
    from math import comb
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def full_eval(env, params, n=100, seed=0):
    """Main condition plus PREREG controls."""
    return dict(
        main=eval_suite(env, params, n, seed),
        c4_coin_reward=eval_suite(env, params, n, seed, c4=True),
        c5_no_memory=eval_suite(env, params, n, seed, c5=True),
    )
