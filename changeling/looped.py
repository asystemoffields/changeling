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

`make_step(cfg)` is the B1 abstract step_fn: it returns a pure
    step(params, h, x) -> (h_new, sample_score, logp_all, value, aux)
that the rollout / collect / loss_fn forward all share. cfg["loop"]=False emits the
EXACT legacy gru path (sample_score = RAW logits, so categorical draws are bitwise
identical to the pre-refactor harness — the G0-A 0.6216 regression spine); loop=True
emits the marginalized micro-loop (sample_score = logp_all = commit_logp).
"""
import jax
import jax.numpy as jnp

from . import N_ACT
from .agent import init_gru, gru_step, hidden_size


def init_looped(key, hidden=128):
    """GRU core (policy logits via Wo/bo) + a halting head + a value head.

    Halt head is named Whalt/bhalt (NOT bh) — the GRU already owns p["bh"] as its
    candidate-state bias; colliding on that key silently ties two unrelated params.
    """
    kg, kh, kv = jax.random.split(key, 3)
    p = init_gru(kg, hidden=hidden)  # carries Wxz..Wo/bo (incl. candidate bias bh)
    s_h = 1.0 / jnp.sqrt(hidden)
    p["Whalt"] = jax.random.normal(kh, (hidden, 1)) * s_h
    p["bhalt"] = jnp.zeros(1)
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


def geometric_prior(lam_p, k_max, k_min):
    """Truncated geometric prior over k=1..K_max (PonderNet KL target), zeroed
    below k_min and renormalized over the realizable support."""
    k = jnp.arange(k_max)                              # 0..K_max-1 -> k=1..K_max
    g = lam_p * (1.0 - lam_p) ** k
    g = g * (k >= (k_min - 1))                         # match the K_min floor
    return g / g.sum()


def kl_to_geometric(p, lam_p, k_max, k_min):
    """KL(p || geometric_prior). p_k=0 terms contribute 0 (PonderNet L_reg)."""
    g = geometric_prior(lam_p, k_max, k_min)
    return jnp.where(p > 0, p * (jnp.log(p + 1e-12) - jnp.log(g + 1e-12)), 0.0).sum()


def micro_loop(p, h, x, k_max=8, k_min=2):
    """One env-step's internal compute. Returns (h_new, commit_logp, commit_v, aux).
    Pure in (p, h, x): no RNG, no state outside the args -> replay-exact."""
    def turn(m, _):
        m, logits = gru_step(p, m, x)                  # weight-tied core re-reads x
        halt = (m @ p["Whalt"] + p["bhalt"])[0]
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


def make_step(cfg):
    """B1 abstract step_fn. Returns a pure
        step(params, h, x) -> (h_new, sample_score, logp_all, value, aux)
    where `sample_score` is what jax.random.categorical consumes, `logp_all` is the
    (A,) action log-prob vector (for stored logp / entropy / the PPO ratio), `value`
    is the committed state-value (commit_v) for the looped path or None for legacy
    (PPO computes the legacy value from its own top-level head), and `aux` always
    carries E_K.

    loop=False is byte-identical to the legacy harness: sample_score = RAW gru
    logits, so categorical(key, sample_score) draws the same action bit-for-bit."""
    if not cfg.get("loop", False):
        def step(params, h, x):
            h2, logits = gru_step(params, h, x)
            return h2, logits, jax.nn.log_softmax(logits), None, {
                "E_K": jnp.float32(1.0)}
        return step

    k_max = int(cfg["k_max"])
    k_min = int(cfg.get("k_min", 2))

    def step(params, h, x):
        h2, clp, cv, aux = micro_loop(params, h, x, k_max=k_max, k_min=k_min)
        return h2, clp, clp, cv, aux
    return step


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

    # (0) NO PARAM COLLISION: gru candidate bias bh kept its (H,) shape, halt head
    # lives under Whalt/bhalt.
    chk("gru candidate bias bh stays shape (H,)", p["bh"].shape == (H,))
    chk("halt head present as Whalt/bhalt", "Whalt" in p and "bhalt" in p)

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

    # (6) make_step legacy path == raw gru logits/log_softmax, sample_score is RAW
    # logits (so categorical draws are bitwise identical to the old harness)
    pg = init_gru(kp, hidden=H)
    leg = make_step({})
    h2, ss, lpa, val, a6 = leg(pg, h, x)
    g_h, g_logits = gru_step(pg, h, x)
    chk("make_step legacy sample_score == raw gru logits",
        float(jnp.max(jnp.abs(ss - g_logits))) == 0.0)
    chk("make_step legacy logp_all == log_softmax(gru logits)",
        float(jnp.max(jnp.abs(lpa - jax.nn.log_softmax(g_logits)))) == 0.0)
    chk("make_step legacy value is None", val is None)
    chk("make_step legacy E_K == 1", abs(float(a6["E_K"]) - 1.0) < 1e-9)

    # (7) make_step looped path == micro_loop
    lp = make_step({"loop": True, "k_max": 8, "k_min": 2})
    h2, ss, lpa, val, a7 = lp(p, h, x)
    mh, mclp, mcv, _ = micro_loop(p, h, x, k_max=8, k_min=2)
    chk("make_step looped sample_score == commit_logp",
        float(jnp.max(jnp.abs(ss - mclp))) == 0.0)
    chk("make_step looped value == commit_v", abs(float(val) - float(mcv)) < 1e-9)

    # (8) KL-to-geometric prior is finite and >=0 at init
    klv = kl_to_geometric(a7["p"], 0.2, 8, 2)
    chk("KL(p||geom) finite and >= -1e-6",
        bool(jnp.isfinite(klv)) and float(klv) >= -1e-6)
    chk("geometric_prior sums to 1",
        abs(float(geometric_prior(0.2, 8, 2).sum()) - 1.0) < 1e-6)

    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    import sys; sys.exit(0 if all(OK) else 1)
