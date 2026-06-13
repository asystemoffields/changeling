"""B2 (PREREG_P1 §1 "Randomization family"): per-lifetime interface randomization.

Each lifetime is handed a FIXED, INVERTIBLE interface (P, perm), resampled every
lifetime, and the agent must re-infer it in-context. The interface is applied by
rollout/collect — this module only SAMPLES it (pure, keyed):

  P    — orthogonal obs projection (OBS_DIM x OBS_DIM), applied as P @ obs.
         expm-orthogonal (train family): P(alpha)=expm(alpha·A), A skew-symmetric
         ⇒ orthogonal for all alpha, EXACTLY I at alpha=0 (reproduces P0), norm-
         preserving (no reward-magnitude leak), kappa=1 always. `PROJ_C` is
         calibrated (calibrate_c) so the mean self-correlation diag(P)≈0 at alpha=1.
         signed-perm (held-out T-family, eval-only): P = D·Pi (Pi permutation,
         D=diag(±1)) — structurally disjoint from the dense-rotation train family,
         proves general interface-inference (G1-G). alpha=0 ⇒ I.
  perm — action permutation in S_{N_ACT}: identity w.p.(1−alpha) else uniform.
         Agent slot a -> env slot perm[a]; alpha=0 ⇒ identity (no-op).

alpha=0 ⇒ (I, identity): a mathematical no-op, so an alpha=0 lifetime is identical
to the un-randomized P0 protocol. Single-axis arms (G1-C/D, C7-P/π) come from
alpha_obs / alpha_act overrides (set the other axis to 0).
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from . import OBS_DIM, N_ACT

# Calibrated 2026-06-13 (calibrate_c, n=64, batch=512): c≈1.94 drives the expected
# mean self-correlation E[mean_i P[i,i]] to ≈0 at alpha=1 (fully decorrelated
# projection). Re-run calibrate_c if OBS_DIM changes. This is the §4 projection-c
# smoke result (un-metered).
PROJ_C = 1.94


def _skew(key, n, c):
    """Skew-symmetric generator A = c·(G−Gᵀ)/√(2n) (PREREG_P1 §1)."""
    g = jax.random.normal(key, (n, n))
    return c * (g - g.T) / jnp.sqrt(2.0 * n)


def sample_obs_proj(key, alpha, n=OBS_DIM, proj_family="expm-orthogonal", c=PROJ_C):
    """Orthogonal obs projection P (n x n). alpha=0 ⇒ I for both families."""
    if proj_family == "expm-orthogonal":
        return expm(alpha * _skew(key, n, c))
    if proj_family == "signed-perm":
        kp, ks = jax.random.split(key)
        perm = jax.random.permutation(kp, n)
        signs = jnp.where(jax.random.bernoulli(ks, 0.5, (n,)), 1.0, -1.0)
        pi = jnp.eye(n)[perm]                      # permutation matrix
        p_full = signs[:, None] * pi               # D·Pi (rows signed)
        # discrete family: used at eval alpha=1; alpha=0 ⇒ exact identity
        return jnp.where(alpha > 0, p_full, jnp.eye(n))
    raise ValueError(f"unknown proj_family {proj_family!r}")


def sample_act_perm(key, alpha, n=N_ACT):
    """Action permutation. identity w.p.(1−alpha) else uniform ∈ S_n. alpha=0 ⇒ id."""
    ku, kb = jax.random.split(key)
    perm = jax.random.permutation(ku, n)
    use = jax.random.bernoulli(kb, alpha)
    return jnp.where(use, perm, jnp.arange(n))


def sample_interface(key, alpha, proj_family="expm-orthogonal", n_obs=OBS_DIM,
                     n_act=N_ACT, c=PROJ_C, alpha_obs=None, alpha_act=None):
    """Per-lifetime interface (P, perm). Pure in key. alpha_obs/alpha_act override
    the shared alpha to isolate a single axis (e.g. alpha_act=0 ⇒ P-only)."""
    a_obs = alpha if alpha_obs is None else alpha_obs
    a_act = alpha if alpha_act is None else alpha_act
    kp, ka = jax.random.split(key)
    P = sample_obs_proj(kp, a_obs, n_obs, proj_family, c)
    perm = sample_act_perm(ka, a_act, n_act)
    return P, perm


def identity_interface(n_obs=OBS_DIM, n_act=N_ACT):
    """The explicit no-op interface (P=I, perm=identity)."""
    return jnp.eye(n_obs), jnp.arange(n_act)


# Disjoint fold for interface keys: keeps interface sampling OFF the existing
# task/rollout split streams, so alpha=None runs reproduce the pre-B2 harness
# bit-for-bit, and eval interfaces stay disjoint from training.
IFACE_FOLD = 0x1FACE


def make_iface_fn(config, for_eval=False):
    """Build a per-lifetime interface sampler iface_fn(key) -> (P, perm) from a
    run config, or None when interface randomization is off (config['alpha'] is
    None) — in which case the whole harness takes the byte-identical no-iface path.
    For eval, `proj_family` may be overridden by `eval_proj_family` (the held-out
    T-family, G1-G). Single-axis arms via `alpha_obs`/`alpha_act`.

    C7 MEMORIZATION control (`fixed_interface: True`): during TRAINING only, every
    lifetime is handed ONE frozen (P*,perm*) seeded from `interface_seed` (matches
    α-strength, removes only the per-lifetime resampling). Eval is untouched — it
    still draws NOVEL held-out interfaces — so a memorizer trained on the single
    fixed interface collapses to chance on eval, the load-bearing falsifier."""
    import functools
    alpha = config.get("alpha")
    if alpha is None:
        return None
    fam = config.get("proj_family", "expm-orthogonal")
    if for_eval:
        fam = config.get("eval_proj_family", fam)
    sampler = functools.partial(
        sample_interface, alpha=float(alpha), proj_family=fam,
        c=float(config.get("projection_c", PROJ_C)),
        alpha_obs=config.get("alpha_obs"), alpha_act=config.get("alpha_act"))
    if config.get("fixed_interface", False) and not for_eval:
        fixed_key = jax.random.PRNGKey(int(config.get("interface_seed", 0)))
        return lambda key: sampler(fixed_key)   # ignore per-lifetime key → one (P*,perm*)
    return sampler


def calibrate_c(key, n=OBS_DIM, batch=512, lo=0.1, hi=12.0, iters=40):
    """Bisection on c so the expected mean self-correlation E[mean_i P[i,i]] ≈ 0 at
    alpha=1 (decorrelated projection). diag(P)=corr(P·eᵢ,eᵢ) since P orthogonal.
    Monotone-decreasing in c over the useful range. Returns the calibrated c."""
    keys = jax.random.split(key, batch)

    def mean_diag(c):
        Ps = jax.vmap(lambda k: expm(_skew(k, n, c)))(keys)
        return jnp.mean(jax.vmap(jnp.diag)(Ps))     # E[mean_i P[i,i]]

    lo, hi = jnp.float32(lo), jnp.float32(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = jax.lax.cond(mean_diag(mid) > 0.0,
                              lambda: (mid, hi), lambda: (lo, mid))
    return float(0.5 * (lo + hi))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    OK = []
    def chk(name, cond, extra=""):
        OK.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))

    # calibrate c and report
    c_star = calibrate_c(jax.random.PRNGKey(1))
    print(f"calibrated PROJ_C ≈ {c_star:.4f} (current default {PROJ_C})")

    # alpha=0 ⇒ exact identity (both families) + identity perm
    P0, pm0 = sample_interface(key, 0.0)
    chk("alpha=0 P == I (expm)", float(jnp.max(jnp.abs(P0 - jnp.eye(OBS_DIM)))) == 0.0)
    chk("alpha=0 perm == identity", bool(jnp.all(pm0 == jnp.arange(N_ACT))))
    Psp, _ = sample_interface(key, 0.0, proj_family="signed-perm")
    chk("alpha=0 signed-perm == I",
        float(jnp.max(jnp.abs(Psp - jnp.eye(OBS_DIM)))) == 0.0)

    # alpha=1 expm: orthogonal (PᵀP=I), norm-preserving, decorrelated diag
    P1, pm1 = sample_interface(key, 1.0, c=c_star)
    ortho = float(jnp.max(jnp.abs(P1.T @ P1 - jnp.eye(OBS_DIM))))
    chk("alpha=1 P orthogonal (PᵀP=I, <1e-4)", ortho < 1e-4, f"max|Δ|={ortho:.2e}")
    v = jax.random.normal(jax.random.PRNGKey(7), (OBS_DIM,))
    chk("P norm-preserving (no reward-magnitude leak)",
        abs(float(jnp.linalg.norm(P1 @ v) - jnp.linalg.norm(v))) < 1e-4)
    md = float(jnp.mean(jnp.diag(P1)))
    chk("alpha=1 mean diag(P) near 0 (decorrelated)", abs(md) < 0.15, f"diaḡ={md:.3f}")

    # signed-perm: orthogonal, exactly one ±1 per row/col
    Psp1, _ = sample_interface(key, 1.0, proj_family="signed-perm")
    chk("signed-perm orthogonal",
        float(jnp.max(jnp.abs(Psp1.T @ Psp1 - jnp.eye(OBS_DIM)))) < 1e-5)
    chk("signed-perm one nonzero/row", bool(jnp.all((Psp1 != 0).sum(1) == 1)))

    # perm is a valid permutation; alpha=1 ⇒ derangement possible but valid perm
    pm = sample_act_perm(jax.random.PRNGKey(3), 1.0)
    chk("alpha=1 perm is a valid permutation",
        bool(jnp.all(jnp.sort(pm) == jnp.arange(N_ACT))))

    # single-axis isolation
    Ppo, pmpo = sample_interface(key, 1.0, alpha_act=0.0)   # P-only
    chk("P-only: perm == identity", bool(jnp.all(pmpo == jnp.arange(N_ACT))))
    chk("P-only: P != I", float(jnp.max(jnp.abs(Ppo - jnp.eye(OBS_DIM)))) > 0.1)
    Pao, pmao = sample_interface(key, 1.0, alpha_obs=0.0)   # perm-only
    chk("perm-only: P == I", float(jnp.max(jnp.abs(Pao - jnp.eye(OBS_DIM)))) == 0.0)

    # purity: same key ⇒ identical interface
    Pa, pa = sample_interface(jax.random.PRNGKey(9), 1.0)
    Pb, pb = sample_interface(jax.random.PRNGKey(9), 1.0)
    chk("pure: identical interface on replay",
        float(jnp.max(jnp.abs(Pa - Pb))) == 0.0 and bool(jnp.all(pa == pb)))

    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    import sys
    sys.exit(0 if all(OK) else 1)
