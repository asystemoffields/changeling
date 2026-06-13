# Memory/compute substrate — risk register + first increment (build-binding)

Companion to SPEC §5b/S3-M. Produced by the hardening workflow `wf_44c62735-db3`
(5 web-grounded red-teams × 3 judged designs + synthesis, 2026-06-13). **Every row's
"build must respect" is a gate/assert/observable the build cannot skip.**

## Risk register (surviving high/blocker after synthesis)

| # | Sev | Risk | Binding mitigation | Build must respect |
|---|---|---|---|---|
| 1 | BLOCKER | Replay non-determinism breaks PPO ratio (the lynchpin) | Marginalize halting (no halt sample); deferred write reconstructable from `last_a`/`last_r`; micro-turns deterministic given (params,carry,x) | **`collect()` & `loss_fn.forward()` produce bitwise-identical commit_logits under a fixed key — hard regression assert BEFORE any metered run** |
| 2 | BLOCKER | Halting collapse to K=1 (loop amputated; false-negative kills the compute axis) | KL-prior WEIGHT warmup-then-anneal (anneal weight not λ_p); λ_p so E[K]≫1; K_min≥2; α=0 loop pre-train; fixed-K fallback | **K-collapse tripwire @ GRU-128 ~1e8 steps: mean E[K]>1.5 AND forcing K=1 drops gate_q4; fixed-K controls mandatory** |
| 3 | BLOCKER | Static-shape break / `while_loop` not reverse-diff, no vmap savings | K as static `lax.scan` of length K_max nested in T-scan; no `while_loop`; dynamic halt deploy-only | Train compute budgeted O(T·K_max); no data-dependent trip count in the train graph |
| 4 | BLOCKER | Deep BPTT (T·K_max + diff reads/writes) explodes/vanishes or OOMs Kaggle T4/P100 | Store BPTT in W-windows (W=1 → depth=K_max); cross-step h full-T (unchanged); nested `jax.checkpoint`; per-turn v_k anchors; γ=0.999 fallback | **Per-route store-path gnorm logging; B sized per rung vs MEASURED peak mem; 8k/16k store rungs OFF Phase-1 critical path** |
| 5 | HIGH | Hard top-k read → zero PPO gradient to selection/non-retrieved keys; ES & PPO optimize different artifacts | Hybrid hard-select + softmax-over-m value-mix; L2 cosine keys; β soft→hard; freeze softness across routes | **Designed ES/PPO bracket on the SAME store; a collapsed-store route is DISQUALIFIED from the slope comparison** |
| 6 | HIGH | stop-grad store across boundaries → PPO ~zero write-policy gradient | W>1 + burn-in on memory-demand env; deferred write short-range-decodable; ES legitimate mainline for writes | **Report ES-vs-PPO & soft-vs-hard write deltas as DECLARED confounders; don't claim PPO trains writes at W=1** |
| 7 | HIGH | Store-size axis has no Phase-1 vehicle (C=5 fits GRU; sweep flat; collapse diagnostic misfires) | Grade N_slots/lifetime slopes on a dedicated memory-demand env; require store-beats-null delta>0 there first | **Usage-collapse diagnostic is a within-env observable, NEVER a kill on cbandit-FR; capacity not asserted on FR** |
| 8 | HIGH | Write/key-utilization collapse to a single averaging bias | PKM BatchNorm-on-query (ES-survives); delta-rule overwrite-nearest; sparse top-m | **Slot-utilization/usage-entropy is a STANDING §4b observable, ALWAYS next to the null-memory ablation** |
| 9 | HIGH | Cross-lifetime store leak silently breaks the memoryless-core claim | Store only in per-lifetime carry, vmapped never scanned, zero-init | **Init assert: trainable leaves disjoint from carry leaves; lifetime-order-invariance bitwise regression; cross-lifetime probe** |
| 10 | HIGH | Halting "learnable but fragile": L_rec trains value-accuracy, a proxy not action-quality | Reward-aligned pathwise PG through the mixture + KL prior; L_rec demoted to value shaping | **Fixed-K control decisive: if adaptive-K doesn't beat fixed-K at matched compute, ship fixed-K** |
| 11 | HIGH/BLK | Consolidation diffused into shared trunk → null-memory control unfalsifiable | FORBID §1d mechanism (a); write ONLY to separable zero-able `params["slow"]`; consolidation dropout; EWC anchor | **Null-memory = one-checkpoint subtraction; "undiminished" = speedup-removed-not-level; named-subtree disjointness assert** |
| 12 | HIGH | Interface×memory chicken-and-egg; canonical-invariance claim outruns the mechanism | Micro-turns establish canonical frame in-activation; bred write-gating on ponder confidence; overwrite-nearest re-keying | **Interface-identity linear-probe falsifier is PHASE-CONDITIONED (at-chance on post-convergence writes)** |
| 13 | HIGH | ES poor grip on consolidation; wrong gate env | Operator low-dim/tied; ES ranks, SGD tunes; mechanism (c) prefers gradient | **Consolidation gate on cbandit-FG/template env, NEVER cbandit-FR; R1 flagged weaker slow-loop route** |

## First build increment (un-metered, CPU — do before any GPU-h)

**Increment A only: the marginalized PonderNet adaptive-K looped core — NO store, NO slow
bank.** Build the B1 abstract `step_fn` refactor first (GRU path preserved), then a `looped`
substrate: K_max weight-tied core steps, PonderNet halting (cumulative-product p_k, K_min≥2,
residual on K_max), commit the marginal mixture π_commit=Σ_k p_k·softmax(logits_k), thread
(commit_logits, commit_v, E[K], aux) through `rollout` / `ppo.collect` / `ppo.loss_fn.forward`.
Wire mixture log-prob into the PPO ratio, KL-to-geometric L_reg + per-turn v_k L_rec into the
loss, `−c·mean_t E[K]` into ES fitness. Tiny CPU config (B=4, T=16, K_max=4, pop=8).

**Pass/fail gates (all un-metered, must pass before touching the store):**
1. **LYNCHPIN — replay exactness:** `max|commit_logits_collect − commit_logits_forward| < 1e-5`
   under fixed PRNG. FAIL ⇒ per-step fn not deterministic-pure (sampled halt / un-replayed RNG)
   ⇒ PPO would silently diverge. Fix before anything else. Becomes the standing regression gate.
2. **Regression safety:** unchanged GRU substrate still reproduces **G0-A = 0.6216 exactly**;
   `looped` at K_max=1 + empty-read ≡ a single weight-tied core step (collect==forward).
3. **Gradient/CRN sanity:** one PPO `epoch_step` + one ES `gen_step` NaN-free, finite gnorm; ES
   `member_fitness` bitwise-deterministic in θ given fixed `roll_keys` (CRN survives marginalization).
4. **Loop is exercised:** under high-warmup prior weight, mean E[K] > 1.5 on the smoke (miniature
   K-collapse tripwire).

If 1–4 pass, Increment B (the store) is unblocked. **Increment A needs none of the Open Decisions
below** (those gate Increments B/C and the prereg lock).

## Open decisions (need Alex before the PREREG_P1 lock — recommendations given)

1. **Where the store-capacity axis is graded.** cbandit-FR can't test N_slots/lifetime
   (flat-by-construction). (a) add a dedicated memory-demand env to Phase-1 now, or (b) ship
   looped-core+K_max in Phase-1 and gate the whole store-capacity + consolidation claim to a
   dedicated S3 mini-phase. *Rec: (b)* — keeps Phase-1 clean within ≤60 GPU-h; store-capacity is
   a separate, properly-resourced phase.
2. **Default store-BPTT window W + write-route strategy.** W=1 (ES mainline for writes; PPO ~no
   write gradient) vs W>1 (PPO trains writes; more memory/instability). *Rec: accept "ES is the
   legitimate mainline for the memory-write axis" as a valid §1b-1 outcome*, escalate to W>1 only
   on the capacity env if the ES/PPO bracket shows PPO write-weakness. (Ratify that "trains under
   both routes" is a measured bracket for writes, not a hard requirement.)
3. **Ratify the declared-harness calibration knobs as pre-reg numbers:** K_max ladder {1,2,4,8};
   λ_p=0.2 (E[K]=5 at K_max=8); β_p constant vs α with warmup-then-anneal on the prior weight;
   K_min≥2; α=0 loop pre-training. *Rec: ratify.*
4. **Narrow SPEC §1d's open consolidation mechanism:** the synthesis forecloses (a) shared-weight
   continual fine-tune (it makes the null-memory control structurally unsatisfiable), committing
   to separable-additive (b)/(c). *Rec: accept the foreclosure* (or keep (a) as a declared-confounded
   control needing a non-subtraction null-memory definition). Confirms a clause Alex authored.
