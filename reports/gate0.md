# Gate 0 — verdict (2026-06-12)

Phase 0 reproduces RL² (Duan et al. 2016) behind the fixed interface protocol
(no randomization yet — that is Phase 1). Gate 0 is a plumbing/feasibility gate,
not a science result. All numbers below are re-scored offline with
`scripts/eval_ckpt.py` on the **pure held-out gate task** (bandit: uniform
U(0,1); catch: catch), n=1000 lifetimes × 2 seeds, eval seed space disjoint from
training.

## Scoreboard

| ID | Criterion | Result | Verdict |
|----|-----------|--------|---------|
| **G0-A** | Bandit best-arm rate ≥ 0.672 (D1) **and** slope > 0 | gate_q4 **0.622** (0.636/0.608); slope +0.080, sign_p ≈ 1e-64 | **PARK** (gate_q4 ≤ 0.63) |
| **G0-B** | Bandit C4 (coin-reward) ≤ 0.225 | 0.12–0.13 | **PASS** |
| **G0-C** | Bandit C5 (no-memory) ≤ 0.225 | 0.14–0.15 (best ckpt); see note | **PASS** |
| **G0-D** | Catch episode success ≥ 0.90 | **1.000** (ES, early-stop gen 199) | **PASS** |
| **G0-E** | ≤ 24 GPU-h for all of Phase 0 | ~3.5 used | **PASS** |

Best agent: R2 PPO-RL², hidden=128 GRU, mixture-curriculum trained (D2, mix=0.5),
graded on pure uniform. Checkpoint `runs/kaggle_r2_s2_mix/.../ckpt_best.npz`.

## G0-A — PARK, not FAIL

Three independent n=1000 measurements cluster tight around **0.62**: r2-mix
predecessor 0.617, s2-mix `ckpt_best` 0.6216, s2-mix final 0.6168. Two stacked
24k-update sessions (48k total) show a flat gate_q4 trajectory — **more PPO does
not move it**. The mixture curriculum broke the earlier 0.52 plateau (D2) and
that win replicated, but a new, harder ceiling sits at ~0.62.

This is the decision tree's branch 3 (`stalls ≤ 0.63 → PARK`). It is honest, not
a failure: the *science* is healthy.

- **Within-lifetime learning is real and robust** — slope +0.07–0.08, sign_p
  ≈ 1e-64. The agent genuinely learns *during* a lifetime.
- **Controls are clean** — C4 (coin-reward) ≈ 0.12 and C6 (full amnesia) ≈ 0.14
  both at chance: the learning is genuinely reward-driven and memory-dependent.
- The only miss is **absolute best-arm precision on a forgiving uniform bandit**
  — the known-hard exploit-precision corner, where second-best ≈ best so reward
  (q4 = 0.85) decouples from best-arm rate (0.62). The 4-param crystal family
  ceiling for this task is ~0.69 (`reports/crystal_bandit.md`), so even a
  near-optimal compact strategy barely clears the 0.672 bar.

**C5/C6 nuance:** on the *final* checkpoint C5 (no-memory) creeps to 0.22–0.24,
above the 0.225 line. This is the known RL²-input-leak artifact (last
action/reward still flows under C5, expressing a memoryless win-stay reflex), not
a gate failure — C6 (full amnesia, the clean control) sits at chance everywhere.
The `ckpt_best` checkpoint scored C5 = 0.14–0.15. Judge memory-dependence by the
gate_q4 − C6 gap (~0.47), not C5 alone.

**Scale-to-signal prediction (PARK record, mg discipline):** the 0.62 plateau is
a capacity/optimization ceiling of the hidden=128 GRU under this curriculum, not
a fundamental barrier. Prediction: a hidden-size sweep {128, 256, 512} shows a
positive scaling slope on gate_q4 and crosses 0.672 at hidden ≈ 256–512. That
sweep is also the first rung of the §1b-1 route-scaling protocol, so it belongs
in **Phase 1**, not a P0 rerun. The criterion is not touched again — D1 was the
one allowed recalibration.

## Route signal (R1 ES vs R2 PPO) — provisional, confounded

On bandit, R2 (0.62) >> R1 (ES plateaued 0.25–0.31). **But this comparison is not
yet clean:** the late-weight w_t enters R1 as an undiscounted whole-lifetime
fitness rank and R2 as a per-step reward reshaped by γ-discounted GAE (effective
weight γ^t·w_t tilts *early*, the reverse of R1's late tilt). See PREREG_P0 **D3**.
The route decision is deferred to the Phase-1 scaling-slope protocol with the
late-weighting reconciled and ES hardened (CRN fix below).

## Harness adversarial review (pre-Phase-1)

Before scaling GPU into Phase 1, a fan-out bug-hunt (6 dimensions × 3-lens
adversarial verification + completeness critic) audited the harness. **13
findings confirmed**; all fixed or documented. Highlights:

- **eval_ckpt.py** imposed slope>0 on *catch*, contradicting G0-D and FAILing every
  converged catcher — fixed (slope required for bandit only).
- **PPO resume** reset `best_gate` to −1 and could clobber `ckpt_best.npz` with a
  lower post-resume value — fixed (persist + restore best_gate; guard overwrite).
- **eval_env_kwargs** silently fell back to the training distribution, which could
  grade the gate on the easier mixture (latent false-PASS) — fixed (assert pure
  gate when training is a curriculum; full-config resume assertions).
- ES route graded its in-loop eval / early-stop on the *training* env (would bite
  the moment ES gets a curriculum in Phase 1) — fixed (eval_env separation, mirrors PPO).
- ES antithetic CRN shared task params but not env stochasticity (the dominant
  Bernoulli-reward noise did not cancel) — fixed (share rollout keys across the
  population; plausibly relevant to the ES plateau).
- Surplus interface actions wrapped onto real arms instead of mapping to no-op
  (SPEC §2) — fixed (no-op; identity at the n_arms=8 gate, verified).
- Latent kernel-build bug: an indented intra-package import survived the
  stripper and would ImportError in any ES kernel — fixed (hoisted + stripper hardened).

**The G0-A verdict was re-scored under the patched code and reproduced 0.6216
exactly** (the no-op change is provably identity at n_arms=8), so the PARK stands.

## Disposition

Phase 0 closes: **G0-A PARK** (with scale prediction), **G0-B/C/D/E PASS**.
Proceed to Phase 1 (interface randomization) with the mixture lesson, the
hardened harness, the C6 control, multi-seed assays, and the D3 route-comparison
reconciliation baked into its prereg.
