"""Increment A (SPEC §5b/S3-M.1): the looped core + adaptive-K micro-turns,
PonderNet-MARGINALIZED. Standalone + pure — no store, no slow bank yet.

The per-env-step computation runs K_max weight-tied micro-turns of the core over
the SAME step input x (hourglass: recurrent DEPTH over fixed context), and COMMITS
THE MARGINAL MIXTURE pi = sum_k p_k * softmax(logits_k) — never a discrete halt
sample. Marginalization is the over-determined lynchpin (PPO replay-exactness +
antithetic-CRN + teacher-forcing all force it): `micro_loop` is a deterministic
pure function of (params, h, x), so collect() and loss_fn.forward() will produce
bitwise-identical commit log-probs.

REDUCTION INVARIANT (the regression spine): at K_max=1 this collapses exactly to
`gru_step` -> log_softmax — so the whole substrate is a strict superset of today's
harness. Verified in __main__.
"""
import jax
import jax.numpy as jnp

from . import N_ACT
from .agent import init_gru, gru_step, hidden_size


def init_looped(key, hidden=128):
    """GRU core (policy logits via Wo/bo) + a halting head + a value head."""
    kg, kh, kv = jax.random.split(key, 3)
    p = init_gru(kg, hidden=hidden)  # carries Wxz..Wo/bo
    s_h = 1.0 / jnp.sqrt(hidden)
    p["Wh"] = jax.random.normal(kh, (hidden, 1)) * s_h
    p["bh"] = jnp.zeros(1)
    p["Wv"] = jax.random.normal(kv, (hidden, 1)) * s_h
    p["bv"] = jnp.zeros(1)
    return p


def _ponder_p(halt_k, k_max, k_min):
    """PonderNet halting masses p_k over k=1..K_max (cumulative-product), with a
    K_min floor (no halt before k_min) and forced halt at K_max (residual mass)."""
    lam = jax.nn.sigmoid(halt_k)                       # (K,)
    lam = lam * (jnp.arange(k_max) >= (k_min - 1))     # K_min>=2 floor
    lam = lam.at[-1].set(1.0)                          # must halt by K_max -> sum p = 1
    notbefore = jnp.concatenate(                       # prod_{j<k}(1 - lam_j)
        [jnp.ones(1), jnp.cumprod(1.0 - lam)[:-1]])
    return lam * notbefore                             # (K,)


def micro_loop(p, h, x, k_max=8, k_min=2):
    """One env-step's internal compute. Returns (h_new, commit_logp, commit_v, aux).
    Pure in (p, h, x): no RNG, no state outside the args -> replay-exact."""
    def turn(m, _):
        m, logits = gru_step(p, m, x)                  # weight-tied core re-reads x
        halt = (m @ p["Wh"] + p["bh"])[0]
        v = (m @ p["Wv"] + p["bv"])[0]
        return m, (m, logits, halt, v)

    _, (ms, logits_k, halt_k, v_k) = jax.lax.scan(turn, h, None, length=k_max)
    # ms (K,H)  logits_k (K,A)  halt_k (K,)  v_k (K,)
    pk = _ponder_p(halt_k, k_max, k_min)               # (K,)
    probs_k = jax.nn.softmax(logits_k, axis=-1)        # (K,A)
    pi = (pk[:, None] * probs_k).sum(0)                # (A,) marginal commit dist
    commit_logp = jnp.log(pi + 1e-12)
    commit_v = (pk * v_k).sum()
    h_new = (pk[:, None] * ms).sum(0)                  # marginal committed state
    e_k = (pk * (jnp.arange(k_max) + 1.0)).sum()       # E[K]
    aux = dict(E_K=e_k, p=pk, logits_k=logits_k, v_k=v_k, halt_k=halt_k)
    return h_new, commit_logp, commit_v, aux


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    IN = 74
    key = jax.random.PRNGKey(0)
    kp, kx, kh = jax.random.split(key, 3)
    H = 32
    p = init_looped(kp, hidden=H)
    x = jax.random.normal(kx, (IN,))
    h = jax.random.normal(kh, (H,))
    OK = []
    def chk(name, cond):
        OK.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # (1) REDUCTION: K_max=1 ≡ gru_step -> log_softmax  (the regression spine)
    h1_gru, logits_gru = gru_step(p, h, x)
    hn, clp, cv, aux = micro_loop(p, h, x, k_max=1, k_min=2)
    chk("K_max=1 commit_logp == log_softmax(gru logits)",
        float(jnp.max(jnp.abs(clp - jax.nn.log_softmax(logits_gru)))) < 1e-5)
    chk("K_max=1 h_new == gru hidden",
        float(jnp.max(jnp.abs(hn - h1_gru))) < 1e-6)
    chk("K_max=1 E[K] == 1", abs(float(aux["E_K"]) - 1.0) < 1e-6)

    # (2) commit is a valid distribution at K_max=8
    hn, clp, cv, aux = micro_loop(p, h, x, k_max=8, k_min=2)
    chk("commit exp(logp) sums to 1", abs(float(jnp.exp(clp).sum()) - 1.0) < 1e-4)
    chk("halting masses p sum to 1", abs(float(aux["p"].sum()) - 1.0) < 1e-5)

    # (3) K_min>=2 floor: no halting mass before turn 2
    chk("K_min=2 floor: p[0] == 0", float(aux["p"][0]) < 1e-9)
    chk("E[K] in [k_min, k_max]", 2.0 - 1e-6 <= float(aux["E_K"]) <= 8.0 + 1e-6)

    # (4) PURITY / replay-exactness: identical outputs for identical inputs
    hn2, clp2, cv2, _ = micro_loop(p, h, x, k_max=8, k_min=2)
    chk("pure: commit_logp bitwise-identical on replay",
        float(jnp.max(jnp.abs(clp - clp2))) == 0.0)

    # (5) loop is actually used (depth before any ponder penalty): E[K] > 1
    chk("E[K] > 1 at K_max=8 (loop exercised)", float(aux["E_K"]) > 1.0)

    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    import sys; sys.exit(0 if all(OK) else 1)
