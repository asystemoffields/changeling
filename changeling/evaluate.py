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
from .interface import IFACE_FOLD

EVAL_FOLD = 10_000_000


def eval_suite(env, params, n=100, seed=0, c4=False, c5=False, c6=False, step=None,
               iface_fn=None, c8=False):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), EVAL_FOLD)
    kt, kr = jax.random.split(key)
    tasks = jax.vmap(env["sample_task"])(jax.random.split(kt, n))
    keys = jax.random.split(kr, n)

    if iface_fn is None:
        def one(task, k):
            return rollout(env, params, task, k, c4=c4, c5=c5, c6=c6, step=step)
        rewards, metrics, dones, e_ks = jax.vmap(one)(tasks, keys)
    elif c8:
        # C8 (within-lifetime reshuffle): rollout re-draws the interface EVERY
        # step from the sampler; nothing decodable carries across steps. Predicted
        # to collapse to chance on cbandit-FR (the G1-F falsifier).
        def one(task, k):
            return rollout(env, params, task, k, c4=c4, c5=c5, c6=c6, step=step,
                           c8_iface_fn=iface_fn)
        rewards, metrics, dones, e_ks = jax.vmap(one)(tasks, keys)
    else:
        # NOVEL held-out interfaces: drawn from EVAL_FOLD ∘ IFACE_FOLD, disjoint
        # from training interfaces (train keys ∘ IFACE_FOLD). Does not touch kt/kr.
        ik = jax.random.split(jax.random.fold_in(key, IFACE_FOLD), n)
        ifaces = jax.vmap(iface_fn)(ik)

        def one(task, k, iface):
            return rollout(env, params, task, k, c4=c4, c5=c5, c6=c6, step=step,
                           iface=iface)
        rewards, metrics, dones, e_ks = jax.vmap(one)(tasks, keys, ifaces)
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
        # K-collapse observable (risk #2): mean adaptive-K over the lifetime.
        # ≡1.0 on the legacy substrate; the tripwire wants final-quarter E[K]>1.5.
        e_k_mean=float(e_ks.mean()),
        e_k_q4=float(e_ks[:, -q:].mean()),
    )


def _binom_tail(k, n):
    """P(X >= k), X ~ Binomial(n, 0.5)."""
    from math import comb
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def full_eval(env, params, n=100, seed=0, step=None, iface_fn=None):
    """Main condition plus PREREG controls. `step` is a B1 step_fn (looped path);
    None grades the legacy gru substrate. `iface_fn` (B2) grades under NOVEL
    held-out interfaces; None grades the un-randomized (α=0-equivalent) protocol.
    All conditions share the same held-out interface draw so controls are
    comparable to main."""
    out = dict(
        main=eval_suite(env, params, n, seed, step=step, iface_fn=iface_fn),
        c4_coin_reward=eval_suite(env, params, n, seed, c4=True, step=step,
                                  iface_fn=iface_fn),
        c5_no_memory=eval_suite(env, params, n, seed, c5=True, step=step,
                                iface_fn=iface_fn),
        c6_full_amnesia=eval_suite(env, params, n, seed, c6=True, step=step,
                                   iface_fn=iface_fn),
    )
    # C8 within-lifetime reshuffle only exists under randomization (nothing to
    # reshuffle at α=None); on cbandit-FR it must sit at chance (G1-F).
    if iface_fn is not None:
        out["c8_reshuffle"] = eval_suite(env, params, n, seed, step=step,
                                         iface_fn=iface_fn, c8=True)
    return out
