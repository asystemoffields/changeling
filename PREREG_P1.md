# PREREG — Phase 1: interface randomization (DRAFT — NOT LOCKED)

**Status: DRAFT.** Produced by a 5-design judge-panel (transfer-honest spine,
graded 8.0/10; grafts from cold-start-purist, substrate-scaling-first,
reproduction-first, diversity-transition). Locks **only when Alex resolves the
Open Decisions below** and the engineering preconditions B1–B5 pass their
regression gates. No metered (Kaggle) run fires before lock. Builds on the
hardened P0 harness (`reports/gate0.md`) and honors D1 (reference-relative bars),
D2 (curriculum mixture), D3 (route-objective reconciliation).

---

## Open decisions (need Alex before lock)

The design is complete modulo five forks that change *what Phase 1 claims* or
*what it costs* — genuinely yours, not resolvable from SPEC/defaults. My
recommendation is the first listed each time; I've drafted to the recommendation.

1. **Headline env: frozen-rule (cbandit-FR) vs fresh-g (cbandit-FG) as PRIMARY.**
   *Rec: frozen-rule primary, fresh-g secondary.* FR is the only way to make
   action-permutation inference non-vacuous and gives a *pure* interface-inference
   slope; FG is more RL²-faithful but entangles task-learning with interface-inference.
   FR shifts the thesis to "infer the interface for a known rule" vs "infer the
   interface AND learn a new task." **This is the thesis-defining call.**
2. **Spend the KILL-scale S2-C run (≥1M params, ≥2e9 steps, ≥2 seeds, ~11 GPU-h)
   THIS phase, or PARK the KILL to a mini-phase?** *Rec: budget it this phase,
   protected as the last descope item.* All the *science* is on cheap S1; the S2-C
   point is the only thing that converts a PARK into a KILL.
3. **PASS stringency: action-permutation isolation (G1-C) MANDATORY (∧) or
   obs-axis alone (G1-D) sufficient (∨)?** *Rec: ∧ (G1-C mandatory)* — the thesis
   needs BOTH §1 levers, so a pass shouldn't be an obs-only shortcut. Raises failure
   risk if 5-context π-inference is hard at affordable scale.
4. **Route verdict on transfer-per-GPU-h (shared GRU ladder) vs a dedicated
   env-step ladder.** *Rec: shared-ladder GPU-h slope* (budget-efficient, §6-literal).
5. **Reward-affine (3rd SPEC-§2 lever) stays DEFERRED to Phase 3?** *Rec: defer*
   — P1 eval reward magnitude is constant, nothing to infer.

**Curriculum caveat (folded in from `reports/explore_seeds.md`, 2026-06-12):** the
strong single-seed "world teaches wanting" transfer claims did NOT survive 4-seed
replication. Only a modest mix≈0.5 effect is robust (inverted-U; mix>0.5 hurts).
So curriculum here is a **mild prior (mix≈0.5), not a silver bullet**; the
"needle-necessary-for-inference" sub-hypothesis (§1) is **non-gating**, and any
curriculum claim must be re-established at scale, not assumed.

---

## 0. Engineering preconditions (build before any metered run)

- **B1 — substrate-agnostic step interface.** `rollout.py`/`ppo.py` are GRU-shaped
  (`gru_step`, `hidden_size`). Refactor to an abstract `step_fn(params, carry, x) ->
  (carry, logits[, value])` + `init_substrate`; GRU and the new transformer both
  implement it. **Regression gate: the GRU path must reproduce G0-A = 0.6216 exactly**
  before S2 work proceeds.
- **B2 — interface module `interface.py`.** `sample_interface(key, alpha) -> (P, pi)`;
  applied inside `rollout()` and `ppo.collect()` to the padded obs and emitted action.
  Interface is part of the per-lifetime `task` (keyed sampling) so held-out eval draws
  NOVEL interfaces from the disjoint `EVAL_FOLD` automatically.
- **B3 — cbandit env family** (one env, two modes via `frozen_rule: bool`); see §1.
- **B4 — D3 config landmine (DONE 2026-06-12).** `ppo.make_update_step` defaulted
  `reward_scale = 1-gamma`; at γ=1.0 that is **0.0 — it zeros every training reward.**
  A guard now asserts `reward_scale > 0`; every D3-reconciled run passes `reward_scale =
  1/T` explicitly. `collect()` already hardcodes `w = linspace(0.5,1.5,T)`; at γ=1 that
  is exactly the effective per-step weight wanted — no change.
- **B5 — resume asserts** extended to the new objective-determining keys (`alpha`,
  `proj_family`, `perm_scheme`, `frozen_rule`, `rule_seed`, `C_ctx`, `gamma`, `lam`,
  `reward_scale`). (The full-config resume assert from P0 already covers any key present
  in `config`; confirm these are all set.)

## 1. Setup

### Envs and the bandit-null-obs decision
- **bandit (existing) = INVARIANCE / NEGATIVE control.** obs = zeros(64) ⇒ P·0 = 0
  (obs-projection a literal no-op); arms iid U(0,1) per lifetime ⇒ the action permutation
  only relabels already-random labels (a distributional no-op). **Both §1 levers are
  vacuous on bandit by construction** — it tests *equivariance*: the randomized agent
  must equal the un-randomized agent, and the memorization control must NOT collapse
  here. A FAIL flags an env/leak bug, not a thesis failure. (α=0 reproduces Gate-0
  byte-for-byte: 0.622.)
- **cbandit (NEW, PRIMARY) — contextual bandit, two modes.** obs = P·onehot(c), C=5
  discrete contexts; K=8 arms; T=256 one-step trials (context iid each trial, ~51
  visits/context). Reward Bernoulli; correct arm pays 0.8, others 0.2 (gate); curriculum
  mixes a sharp 0.95/0.05 "needle" margin w.p. 0.5 (D2). Action permutation π scrambles
  which slot reaches which arm.
  - **cbandit-FR (frozen-rule) = HEADLINE env.** The context→correct-arm map y(c) is a
    FIXED injective map, seeded once, SHARED across all lifetimes and eval. The only
    per-lifetime unknown is the interface (P,π). Both axes are load-bearing AND in-lifetime
    inferable: obs-projection forces clustering of C=5 recurring projected-context vectors;
    action-permutation forces re-identifying which of 8 slots is correct (non-vacuous
    *because the rule is fixed and shared across contexts*). The within-lifetime slope is
    *pure interface-inference speed*. This is where the memorization control collapses on
    BOTH axes.
  - **cbandit-FG (fresh-g) = SECONDARY confirmation.** y(c) resampled per lifetime ⇒
    genuine in-context *task* learning under randomization (RL²-faithful). Confirms
    interface-inference composes with task-learning; π is absorbed into g here (so it is
    NOT used for the action-axis claim).
- **catch (existing) = PARKED stretch probe, reported-not-gated.** Obs-projection +
  permutation under sparse/spatial reward, BUT its interface is inferable from the
  DYNAMICS channel (paddle moves visibly), so its shuffled-reward control is structurally
  leaky — it cannot carry the reward-driven obs claim. Run only at the S2 top rung as a
  bridge to L2/MinAtar.
- **Env→axis map:** bandit = negative control (both axes vacuous); cbandit-FR =
  obs-projection AND action-permutation inference, isolated by single-axis arms; cbandit-FG
  = obs+task; catch = obs+dynamics, parked.

### Randomization family
- **Obs projection P** (per lifetime, fixed within lifetime, resampled every lifetime):
  orthogonal. At α=1, P = Haar-random orthogonal (QR of N(0,1)^{64×64}, column signs
  fixed). For the anneal, P(α) = expm(α·A), A = c·(G−Gᵀ)/√128 skew-symmetric, c calibrated
  so corr(P·eᵢ, eᵢ)≈0 at α=1; α=0 ⇒ I (reproduces P0). Orthogonal ⇒ norm-preserving (no
  reward-magnitude leak), invertible (κ=1 always). Applied to padded obs; bandit's zeros ⇒
  no-op (by design).
- **Action permutation π** (per lifetime): at α=1 a uniform π∈S₈; for the anneal, π=identity
  w.p.(1−α) else uniform. Agent slot a → env slot π(a); slots ≥ n_live → no-op (SPEC §2,
  already in harness).
- **Reward affine (DEFERRED, open decision 5).** Per-lifetime reward scale/shift on the
  reward INPUT only (raw-reward fitness), α-gated OFF for P1; reactivates Phase 3.
- **T-family (held-out transfer):** train on the dense-rotation family; EVAL also on a
  structurally-disjoint family — sparse signed permutations P = D·Π (Π perm, D=diag(±1)) —
  to prove general interface-inference, not a family-specific trick.

### Substrates (two orthogonal 3-level ladders, §1b-1)
- **S1 GRU (workhorse/control):** hidden {128, 256, 512} = {78 984, 256 264, 905 736}
  params (11.5× span). This IS the Gate-0 PARK ladder.
- **S2 pre-LN causal transformer (interp target, BUILT in P1):** RoPE, 4 heads, MLP×4,
  full-lifetime context (T≤288, no chunking), linear in-embed 74→d, value head train-only.
  Rungs: **S2-A d=96/L=3 ≈0.34M; S2-B d=144/L=4 ≈1.0M; S2-C d=192/L=6 ≈2.67M** (~7.9×
  span; top two ≥1M). Residual stream = Phase-4 SAE substrate. S2-C is the only
  KILL-eligible point.
- **Capacity slope** = transfer vs log-params over each substrate's 3 rungs at a fixed
  env-step budget per rung (plateau detector + ×2 extension if the top rung is still
  rising, flagged). S1-vs-S2 decided by Δslope with seed-bootstrap CI (§1b-1: a 1×-only
  lead counts only if it survives naive 100× extrapolation).

### Routes and D3 reconciliation (mandatory — this design compares routes)
- **R2 PPO-RL² = mainline.** D3 fix: **γ=1.0, λ=0.95, reward_scale=1/T** (set explicitly).
  At γ=1 the start-state objective's effective per-step weight = the hardcoded
  wₜ=linspace(0.5,1.5,T); λ=0.95 only reduces advantage variance.
- **R1 ES (CRN-hardened):** fitness = Σwₜrₜ/Σwₜ ⇒ effective weight = wₜ. **Both routes
  now optimize the identical wₜ profile; log both temporal profiles each run and assert
  they match.**
- **Route decision (open decision 4):** transfer-slope across the shared GRU {128,256,512}
  ladder, both routes, reported vs log-params AND measured log-GPU-h; the §6 verdict uses
  the GPU-h slope.
- **D3 robustness:** at the primary cell run BOTH matching conventions — (i) PPO γ=1 +
  ES-late wₜ, and (ii) PPO γ=0.999 (reward_scale=1e-3) + ES fitness re-weighted by γᵗ·wₜ —
  and confirm the route ordering is invariant. If it flips, the route decision stays
  declared-confounded. γ=0.999 is also the stability fallback if γ=1 PPO destabilizes.

### Cold-start vs anneal (decided once, applied uniformly)
Cold-start (α=1 from step 0) is the MAINLINE claim. The cold/anneal choice is made on ONE
cheap tripwire run (§5) and applied uniformly to all rungs so it cannot confound the
substrate/route slopes. **Pre-committed "cold-start failed" trigger:** at the GRU-128
tripwire (~1e8 steps), cbandit-FR cold-start slope sign-p ≥ 0.05 OR q4 ≤ C6+0.10. If
triggered, ALL runs switch to the anneal (master α on grid {0,0.1,0.2,0.35,0.5,0.7,0.85,1.0},
performance-gated advance, graded always at α=1). The cold-vs-anneal delta is itself a
reported bitter-lesson datapoint.

### Curriculum
Two orthogonal dials sampled independently per lifetime (product distribution), graded on
the pure gate via `assert_pure_gate`: (1) TASK-CONTRAST (D2) — sharp-margin "needle"
lifetime w.p. 0.5 else standard (cbandit: 0.95/0.05 vs 0.8/0.2; bandit: needle 0.9/0.1 vs
uniform), gate on the standard/uniform task; (2) INTERFACE-STRENGTH α (cold=1, or annealed).
**Per `reports/explore_seeds.md`, treat the task-contrast as a mild prior (mix≈0.5), not a
driver.** Secondary falsifiable sub-hypothesis (NON-gating): the needle margin is NECESSARY
to bootstrap interface inference at high α — ablation p_task=0 vs 0.5 at α=1, predicted slope
collapse for soft-only.

## 2. Controls (each with predicted value, cbandit-FR α=1 GRU-256+; chance = 1/8 = 0.125)
- **C7 — MEMORIZATION control (load-bearing falsifier).** Train at α=1 STRENGTH but with a
  SINGLE frozen (P*,π*) reused every lifetime (matches strength, removes only resampling);
  eval on NOVEL interfaces. **Predicted: cbandit-FR q4 ≈ 0.125, slope≈0 (COLLAPSE on BOTH
  axes); bandit q4 ≈ 0.60 (SURVIVES — equivariance).** Single-axis variants C7-P and C7-π
  collapse on their respective isolated axes.
- **C3 — UN-RANDOMIZED REFERENCE (the 70% denominator), per substrate size.** Trained+eval
  at α=0. **Predicted: cbandit-FR ≈ 0.98 → bar 0.70×0.98 ≈ 0.69; bandit 0.622(h128)→~0.69(h512)
  → bar 0.434→0.48; catch ≈1.0 → 0.70.**
- **C6 — full amnesia** (reset h + zero RL² channels). **Predicted: cbandit-FR ≈ 0.125,
  bandit ≈ 0.14.** Memory-dependence judged by gate_q4 − C6.
- **C4 — shuffled-reward:** slope→0, cbandit-FR q4≈0.125. **C5 — no-memory** (one-step
  cbandit resets every step ⇒ clean): ≈0.125. **C1 — random-init:** chance. **C2 — frozen
  non-meta:** novel-task ≈chance. **C8 — within-lifetime interface reshuffle** (resample
  (P,π) every step ⇒ nothing decodable): q4≈0.125, slope≈0 — a high C8 means the "learning"
  is a randomization-invariant heuristic, not inference.

## 3. Gate 1 criteria (held-out NOVEL interfaces, α=1, n≥1000 lifetimes, disjoint EVAL_FOLD)
- **G1-A (PRIMARY, cbandit-FR, both axes):** slope>0 by one-sided binomial sign test p<0.05
  (n=1000) AND final-quarter ≥ 0.70 × C3 (≈0.69). Predicted q4 ≈ 0.85.
- **G1-B (HEADLINE dissociation, cbandit-FR):** C7 FAILS (slope sign-p>0.05 AND q4 ≤ 0.20)
  AND (mainline − C7) gap ≥ 0.40 (predicted ≈ 0.72).
- **G1-C (ACTION-permutation isolation, cbandit-FR π-only):** slope>0 (p<0.05) AND C7-π
  collapses (q4≤0.20). *The genuine action-permutation-inference test (open decision 3).*
- **G1-D (OBS-projection isolation, cbandit-FR P-only):** slope>0 (p<0.05) AND C7-P collapses.
- **G1-E (bandit invariance/negative control):** randomized q4 ≥ 0.90×C3 AND C7 does NOT
  collapse (q4 ≥ 0.90×mainline). FAIL ⇒ env/leak bug, NOT KILL.
- **G1-F (controls):** C4 slope→0; C6, C8 ≤ chance+0.075; C5 collapses.
- **G1-G (T-family):** signed-permutation held-out family — slope>0 (p<0.05); report level.
- **G1-H (budget):** ≤60 GPU-h cumulative.
- **PASS = G1-A ∧ G1-B ∧ G1-C ∧ G1-E at KILL-scale** (open decision 3 sets the ∧/∨).
  G1-D/G/secondary cbandit-FG strengthen.
- **KILL/PARK scale (SPEC §7):** KILL licensed ONLY at S2-C ≥1M params AND ≥2e9 env-steps
  if G1-A fails AND the capacity slope ≤0 AND both cold-start and anneal fail. Every
  lower-scale miss PARKs with the capacity-regression's predicted params-to-cross.

## 4. Compute / multi-seed plan (~58 GPU-h of the ≤60 envelope; all Kaggle, 7.5h wall-guard + clean-exit + full-config-assert resume; verify each push within 90s)
- Calibration smokes (cbandit-FR α=0 ref, α=1 learnability + C7-collapse miniature,
  projection-c) ~2h.
- Cold-start tripwire (first experiment, §5) ~2h.
- cbandit-FR R2 cold-start, GRU {128,256,512} (capacity slope, single-seed interior) +
  GRU-512 ×3 seeds (G1-B headline) ~9h.
- Single-axis arms: P-only + π-only, GRU-256, R2 ~3h; C7-P/C7-π eval-only ~1h.
- C7 fixed-interface (GRU-512 ×3 seeds) + full eval battery (C1–C6, C8, T-family) ~5h.
- bandit invariance + Gate-0 PARK: R2 randomized vs un-rand, GRU {128,256,512}, + C7-bandit ~5h.
- Route R1 ES vs R2 PPO, GRU {128,256,512} × {bandit, cbandit-FR}, D3-reconciled, + D3
  two-convention robustness pair at GRU-256 ~8h.
- S2 build + S2 capacity ladder {A,B,C} on cbandit-FR, R2, cold-start (single-seed) ~12h.
- S2-C KILL point (≥1M, ≥2e9 steps) ×2 seeds (BUDGETED, open decision 2) ~11h.
- cbandit-FG secondary, GRU-256, R2 ~3h. (catch parked stretch ~3h — first to descope.)
- **Multi-seed:** G1-B dissociation pair ≥3 seeds (GRU-512); slope endpoints (top GRU rung,
  S2-C) ≥2 seeds; interior sweep single-seed. Report per-seed + seed-bootstrap CI on every slope.
- **Descope order if budget bites:** catch → cbandit-FG → interior 2nd seeds → S2-A. The
  cheap S1 science + the S2-C KILL point are protected last.

## 5. First experiment (launches immediately after lock)
**Cold-start tripwire on cbandit-FR — GRU-128 + R2 PPO (D3-reconciled), two cells together:
α=0 (reference) and α=1 (cold-start), each ~1e8 env steps (~2 GPU-h total), plus the C7
fixed-interface control eval-only.** Config: env=cbandit, frozen_rule=True, C_ctx=5, K=8,
T=256, task-mixture p=0.5 (gate on standard 0.8/0.2 via `assert_pure_gate`); GRU hidden=128;
R2 with **γ=1.0, λ=0.95, reward_scale=1/256** (explicit — B4); proj_family=expm-orthogonal;
perm_scheme=uniform; eval on disjoint EVAL_FOLD with NOVEL interfaces. Preceded by the
un-metered calibration smoke that fixes projection constant c and confirms α=0 is learnable.
**De-risks four things at once:** (1) the new central env is learnable under full
randomization (α=1 slope>0, beats C6); (2) the cold-vs-anneal fork is decided by the
pre-committed trigger; (3) the inference-not-memorization dissociation appears in miniature
(C7 collapses to chance while α=0 sits near ceiling); (4) the D3-reconciled PPO trains
stably and the reward_scale landmine is cleared.

---

*Provenance: judge-panel design (5 designs × 4-criterion panel + synthesis), workflow
`wf_94bf173a-e07`, 2026-06-12. Spine = transfer-honest; grafts attributed in the workflow
record. DRAFT pending Alex's resolution of the 5 Open Decisions before lock.*
