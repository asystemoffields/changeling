# HANDOFF — state as of 2026-06-12 (ultracode session 2)

Single source of truth for continuing **changeling**. Read in this order: `SPEC.md`
(design + gates), `PREREG_P0.md` (locked criteria + deviations D1/D2/D3 + Gate 0
outcome), `reports/` (findings — now incl. `gate0.md`), then this file. Everything is
reproducible from this repo + Kaggle account `asystemoffields`. Local box:
`/data/changeling`, venv `.venv` (CPU jax). Memory of record:
`~/.claude/projects/-home-alex/memory/project_changeling.md`.

## Continuity note (model-agnostic by design)

Built under Fable 5; continued under Opus 4.8 after the 2026-06-12 export-control
suspension of Fable/Mythos (anthropic.com/news/fable-mythos-access). Nothing in
changeling depends on the model — carry on without concern for provenance. Git commits
keep the `Co-Authored-By: Claude Fable 5` trailer for repo consistency only.

## Phase 0 is CLOSED (2026-06-12)

**Gate 0 verdict — full writeup in `reports/gate0.md`:**

| ID | Criterion | Result | Verdict |
|----|-----------|--------|---------|
| G0-A | bandit best-arm ≥0.672 (D1) + slope>0 | 0.622 (0.636/0.608), slope +0.08 sign_p~1e-64 | **PARK** (≤0.63) |
| G0-B | C4 coin-reward ≤0.225 | 0.12–0.13 | PASS |
| G0-C | C5 no-memory ≤0.225 | 0.14–0.15 (best ckpt) | PASS |
| G0-D | catch success ≥0.90 | 1.000 (ES, gen 199) | PASS |
| G0-E | ≤24 GPU-h | ~3.5 used | PASS |

G0-A PARK is honest, not failure: within-lifetime slope + controls (C4/C6 at chance) +
catch all green; the only miss is absolute best-arm precision on a forgiving uniform
bandit (the known exploit-precision corner — q4 reward 0.85 decouples from best-arm 0.62;
4-param crystal-family ceiling ~0.69). **Three independent n=1000 scores cluster at 0.62;
+24k more PPO updates moved nothing → a real ceiling for hidden=128.** PARK carries a
falsifiable prediction: hidden-size sweep {128,256,512} crosses 0.672 at ~256–512, run as
the first rung of the Phase-1 route-scaling protocol. Criterion not touched again (D1 was
the one allowed recalibration). Best ckpt: `runs/kaggle_r2_s2_mix/.../ckpt_best.npz`.

Score any ckpt with: `.venv/bin/python scripts/eval_ckpt.py <ckpt.npz> --n 1000 --seeds 1,2`
(auto-detects ES/PPO+hidden, grades on the PURE gate task, prints gate_q4/slope/C4/C5/C6).

## Harness hardened before Phase 1 (adversarial review done)

A fan-out bug-hunt (6 dims × 3-lens verification + completeness critic, 58 agents)
confirmed **13 findings; all fixed or documented** (PREREG_P0 "Harness hardening" entry +
D3). The G0-A verdict was re-scored under the patched code and **reproduced 0.6216 exactly**
(the no-op change is provably identity at n_arms=8). Verified: 14/14 local smoke checks,
both kernels compile with zero leftover intra-imports, backward-compat ckpt load. Key fixes:
`eval_ckpt` no longer fails converged catchers; PPO/ES resume persists `best_gate` + guards
`ckpt_best`; full-config resume assertions; pure-gate `eval_env` assert (closes a latent
false-PASS); ES eval_env separation; ES antithetic CRN shares rollout noise; surplus actions
→ no-op (SPEC §2); kernel-build indented-import bug fixed. **Rebuild kernels after any package
edit** (`scripts/build_kernel.py`); old `kaggle/*/kernel.py` are now stale vs the patched package.

**D3 (declared confounder):** the late-weight w_t is NOT identical selection pressure across
routes — ES ranks undiscounted whole-lifetime fitness (late-tilted w_t); PPO bakes w_t into
per-step reward reshaped by γ-discounted GAE (effective γ^t·w_t, early-tilted). The R1-vs-R2
slope comparison is confounded until reconciled in the Phase-1 route protocol. Does not affect
any single-route eval.

## Standing next actions (priority order)

1. **Phase 1 = interface randomization** (SPEC §7, the load-bearing novelty): per-lifetime
   random obs projection + action permutation. Design via judge-panel → lock PREREG_P1 BEFORE
   the first metered run. Bake in: the mixture/contrast lesson (D2), the hidden-size scaling
   rung (Gate-0 PARK prediction) which doubles as the §1b-1 route protocol, the D3
   late-weighting reconciliation, the C6 control, multi-seed assays, the slope sign test, and
   the no-op interface. Substrate S1 GRU (control) vs S2 small transformer (interp target).
2. **Multi-seed assay hardening** (Kaggle, embarrassingly parallel): the exploration-economics
   2×2 and the mixture result are SINGLE-SEED (caveat in `reports/`). Fan out across seeds +
   cells (mix∈{0.25,0.5,0.75}, needle-gap sweep) before any enters PREREG_P1.
3. **Crystallize the s2-mix agent** (`scripts/crystallize_bandit.py` pattern; needs a
   PPO-ckpt loader — copy `scripts/eval_ckpt.py:load`). Sharp question: did the harder world
   breed the perseveration term DOWN and value-tracking UP vs the ES fossil? Phase-4
   kernel-extraction in miniature.

## Infra truths (carry forward)

- All heavy compute on Kaggle `asystemoffields` per the offload policy; box-independent batch
  kernels with a 7.5h wall guard + clean checkpoint exit; resume across sessions via the
  `changeling-ckpts` DATASET (`kernel_sources` mounts CODE not outputs; recursive `find_resume`
  glob). Verify every push within 90s (silent drops at the 5-slot cap); fresh slug on retry.
- Eval harness verified unbiased (uniform policy → 0.124 vs 0.125 theory). `eval_ckpt.py` is
  the SSOT scorer (pure gate task, auto-detect route/hidden).
- Repo on GitHub: `asystemoffields/changeling` (gh auth setup-git done).
