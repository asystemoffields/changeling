# HANDOFF — state as of 2026-06-12 ~17:30

For any agent or human continuing this work. Read SPEC.md (design), PREREG_P0.md
(locked criteria + deviations D1/D2), reports/ (findings so far). Everything below
is reproducible from this repo + Kaggle account asystemoffields.

## In flight RIGHT NOW

- Kaggle kernel **changeling-p0-r2-s2-mix** (batch, GPU, RUNNING): R2 PPO resuming
  the mixture-trained checkpoint (true gate_q4 0.617 @ n=1000) for +24k updates,
  mix=0.5 training / pure-uniform gate eval, early-stop at 0.672 on n=200 evals,
  saves ckpt_best.npz on new best. Completes within ~80 min of launch regardless
  of any local machine.
- When it lands: `kaggle kernels output asystemoffields/changeling-p0-r2-s2-mix -p runs/...`
  then evaluate BOTH ckpt.npz and ckpt_best.npz at n≥1000 (see the eval snippet in
  git log d7e41ee era, or scripts/crystallize_bandit.py load pattern).

## Gate 0 scoreboard

| criterion | state |
|---|---|
| G0-D catch ≥0.90 | **PASS** (1.000) |
| G0-A bandit ≥0.672 (D1) + slope>0 | best so far 0.617 true / 0.667 touched; slope PASS |
| G0-B / G0-C controls | PASS everywhere |
| G0-E budget ≤24 GPU-h | ~3.5 used after s2-mix |

## Decision tree after s2-mix

1. ckpt_best ≥ 0.672 at n≥1000 → G0-A PASS. Close Gate 0, write the gate report,
   then: R1-vs-R2 route decision needs the §1b-1 scaling-slope protocol (3 compute
   levels per route) — design it into Phase 1 prereg rather than rerunning P0.
2. 0.63–0.67 → one seed-variance check (seed=1,2 at gate scale, mixture config)
   before ANY further deviation. Do NOT touch the criterion again (D1 was enough).
3. Stalls ≤0.63 → PARK G0-A with a written scale/curriculum prediction (mg
   discipline), proceed to Phase 1 prereg with the mixture lesson baked in.

## Key findings to date (full versions in reports/)

- ES fossil = satisficing lock-in; 4-param crystal (softmax-Q + perseveration)
  extracted from its traces OUTPLAYS it 2× — first crystallization ratchet click.
- The world teaches wanting, not the objective: needle contrast tripled ES
  best-arm; q4-only fitness hurt; mixture curriculum broke R2's 0.52 plateau.
- Dispositions transfer as contingencies (needle-bred beats uniform-bred ON uniform).
- kernel_sources mounts CODE not outputs — resume goes via changeling-ckpts dataset.
- Eval harness verified unbiased (uniform policy → 0.1243 vs 0.125 theory).

## Standing next actions (in order)

1. Resolve Gate 0 per decision tree above.
2. Implement Phase 1 (interface randomization) per SPEC §7; lock its prereg
   numbers BEFORE first metered run; include: mixture/contrast lesson, C6 control,
   slope sign test (already in evaluate.py), multi-seed assays.
3. Crystallize the R2/mixture agent (scripts/crystallize_bandit.py pattern) —
   is it the same softmax-Q shape with the habit term bred down?

Local box: /data/changeling, venv at .venv (CPU jax). Memory of record for this
project: ~/.claude/projects/-home-alex/memory/project_changeling.md.
