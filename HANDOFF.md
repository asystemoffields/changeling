# HANDOFF — state as of 2026-06-12 ~21:55

Single source of truth for continuing **changeling**. Read in this order: `SPEC.md`
(design + gates), `PREREG_P0.md` (locked criteria + deviations D1/D2), `reports/`
(findings), then this file. Everything is reproducible from this repo + Kaggle
account `asystemoffields`. Local box: `/data/changeling`, venv `.venv` (CPU jax).
Memory of record: `~/.claude/projects/-home-alex/memory/project_changeling.md`.

## Continuity note (why this handoff exists)

On 2026-06-12 a US govt export-control directive suspended Fable 5 + Mythos 5 for all
customers (other models unaffected; see anthropic.com/news/fable-mythos-access). This
project was built under Fable, transitioned to Opus 4.8 mid-session, and is continuing
under whatever model reads this. **Nothing in changeling depends on the model** — that
was a design goal (SPEC: "the changeling shouldn't care which mind tends the crucible"),
and the Fable→Opus handoff already validated it in production. Carry on without concern
for provenance; git commits keep the `Co-Authored-By: Claude Fable 5` trailer for
repo consistency only.

## In flight RIGHT NOW

- Kaggle kernel **changeling-p0-r2-s2-mix** (batch, GPU, RUNNING as of 21:55): R2 PPO
  resuming the mixture-trained checkpoint (true gate_q4 0.617 @ n=1000) for +24k more
  updates. Trains on mix=0.5 (needle w.p. 0.5 else uniform), **graded on pure uniform**
  (the gate task, unchanged). Early-stops at gate_q4 ≥ 0.672 on n=200 evals; saves
  `ckpt_best.npz` on every new best. Self-contained server-side; finishes regardless of
  the local box. A background watcher is armed but does NOT survive a session change —
  **a fresh session must re-arm it** (see below).
- Re-arm watcher (fresh session, one notification on completion):
  ```bash
  while true; do s=$(/data/kagglecli-venv/bin/kaggle kernels status \
    asystemoffields/changeling-p0-r2-s2-mix 2>&1) || true; \
    echo "$s" | grep -qiE 'complete|error|cancel|404' && { echo "$s"; break; }; \
    sleep 180; done    # run via Bash run_in_background:true
  ```
- When it lands — fetch then SCORE with the new tool (no more scattered snippets):
  ```bash
  cd /data/changeling
  /data/kagglecli-venv/bin/kaggle kernels output \
    asystemoffields/changeling-p0-r2-s2-mix -p runs/kaggle_r2_s2_mix
  .venv/bin/python scripts/eval_ckpt.py runs/kaggle_r2_s2_mix/gate_bandit_ppo/ckpt.npz \
    --n 1000 --seeds 1,2
  # if ckpt_best.npz is in the output, score it too — it is the run's honest max
  ```
  `eval_ckpt.py` auto-detects ES/PPO + hidden from the stored config, grades on the pure
  gate task, prints gate_q4/slope/sign_p/C4/C5/C6 per seed + PASS/FAIL vs the bar.
  Validated: it reproduces the r2-mix predecessor (0.614/0.619, mean 0.617).

## Gate 0 scoreboard

| criterion | state |
|---|---|
| G0-D catch ≥0.90 | **PASS** (1.000, early-stopped gen 199) |
| G0-A bandit ≥0.672 (D1 reference-relative) + slope>0 | best so far 0.617 true / 0.667 touched n=100; slope PASS (sign_p≈0) |
| G0-B coin-reward control ≤0.225 | PASS (~0.12 everywhere) |
| G0-C no-memory(C5) ≤0.225 | PASS but see nuance ↓ |
| G0-E budget ≤24 GPU-h | ~3.5 used; plenty of headroom |

**C5/C6 nuance (important for the verdict):** C5 resets recurrent state but the RL²
input channel still carries last-action/last-reward, so a *memoryless* one-step
win-stay reflex (~0.15–0.24 best-arm) leaks through. On strong agents C5 can creep
**above 0.225** (the r2-mix predecessor hit 0.237 at one seed). That is the known
artifact the PREREG interpretation note + C6 control were added for — do NOT fail the
gate on C5 alone. The clean control is **C6 (full amnesia: C5 + zeroed inputs)**, which
must sit near chance (~0.09–0.15). Report C5 transparently, verify C6 is clean, and
judge memory-dependence by the gate_q4−C6 gap.

## Decision tree after s2-mix

1. **ckpt_best ≥ 0.672 at n≥1000 (slope>0, C6 clean)** → G0-A PASS. Close Gate 0, write
   `reports/gate0.md`. Route decision (R1 ES vs R2 PPO) is NOT a P0 rerun — it needs the
   §1b-1 scaling-slope protocol (3 compute levels/route); design it into the Phase 1
   prereg.
2. **0.63–0.67** → one seed-variance check (seeds 1,2,3 at gate scale) before ANY further
   deviation. **Do NOT touch the criterion again** — D1 was the one allowed recalibration.
3. **Stalls ≤0.63** → PARK G0-A with a written scale/curriculum prediction (mg discipline:
   KILL only at pre-registered scale, else PARK), proceed to Phase 1 with the mixture
   lesson baked in. A PARK here is honest, not a failure — bandit exploit-precision is a
   known-hard corner and the slope/controls/transfer science all already work.

## Key findings to date (full versions in reports/)

- **ES fossil = satisficing lock-in** (explores 2.8/8 arms, repeats after losses, not
  value-tracking). Its MDL knee is **4 params** (softmax-Q α=0.05 β=4 perseveration=2);
  the extracted crystal **outplays its own organism 2×** (0.47 vs 0.22 best-arm) —
  compression-as-regularization, the first crystallization-ratchet click. BC-distilled
  micro-GRUs fit traces better but play worse → **verify-by-playing is load-bearing**.
- **The world teaches wanting, not the objective**: needle contrast alone tripled ES
  best-arm and halved post-loss perseveration; final-quarter-only fitness HURT (honest
  negative); mixture curriculum broke R2's 0.52 plateau (→0.617). Dispositions transfer
  as contingencies (needle-bred beats uniform-bred ON uniform).
- **Infra truths**: `kernel_sources` mounts CODE not outputs → resume via the
  `changeling-ckpts` DATASET (recursive `find_resume` glob). Eval harness verified
  unbiased (uniform policy → 0.124 vs 0.125 theory). Both R2 sessions plateau ~0.52 under
  pure-uniform training regardless of budget (entropy → 0.01: PPO settles in policy space).

## Standing next actions (priority order)

1. **Resolve Gate 0** per the decision tree (sequential, gated on the kernel).
2. **Crystallize the mixture agent** (`scripts/crystallize_bandit.py` pattern; it now
   needs a PPO-checkpoint loader — copy the auto-detect from `scripts/eval_ckpt.py:load`).
   Sharp question: did the harder world breed the **perseveration term DOWN** and
   value-tracking UP vs the fossil? This is the Phase 4 kernel-extraction story in
   miniature.
3. **Phase 1 = interface randomization** (SPEC §7): per-lifetime random obs projection +
   action permutation. Lock its prereg BEFORE the first metered run; bake in the mixture/
   contrast lesson, the C6 control, multi-seed assays, and the slope sign test (already in
   `evaluate.py`). This is the load-bearing novelty of the whole project.

## Ultracode: which work is genuinely workflow-shaped

You're entering an ultracode session — author workflows for the items below; keep #1 of
the decision tree sequential (it's a single gated verdict, not a fan-out).

- **Harness adversarial review BEFORE scaling GPU into Phase 1** (do this early): fan-out
  bug-hunt across `changeling/{envs,agent,rollout,es,ppo,evaluate}.py`, each finding
  adversarially verified. The RL² rollout, the PPO clipped loss + GAE, and the eval/control
  wiring are the high-value targets — a bug there silently corrupts every gate. Classic
  find→verify review workflow.
- **Multi-seed assay hardening**: the exploration-economics 2×2 and the mixture result are
  SINGLE-SEED (flagged as a caveat in reports/). Fan out the matrix across seeds (and a few
  extra cells: mix∈{0.25,0.5,0.75}, needle gap sweep) in parallel before any of it enters a
  locked prereg. Embarrassingly parallel CPU.
- **Crystallization fan-out**: dissect fossil + r2-mix + s2-mix (+ Phase 1 agents) as
  independent parallel agents; synthesize the comparison (do MDL knees stay at 4 params? does
  perseveration shrink as worlds harden?). One agent per specimen.
- **Phase 1 design judge-panel**: N independent Phase 1 designs (projection family, anneal
  schedule, S1/S2/S3 substrate, curriculum), scored by judges, synthesized into the prereg.
  The SPEC sketches Phase 1 but the numbers need careful adversarial design.
- **Literature sweep** (optional): multi-source review of in-context-RL exploration, meta-RL
  curricula, and UED-with-learning-progress (feeds the Phase 2b Weaver design).

All compute on Kaggle (`asystemoffields`) per the offload policy; checkpoint/resume via the
`changeling-ckpts` dataset; verify every push within 90s (silent drops at the slot cap).
