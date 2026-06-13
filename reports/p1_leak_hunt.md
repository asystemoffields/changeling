# B2/B3 adversarial leak-hunt — verdict + binding follow-ups

Workflow `wf_fd5e9964-1e6` (2026-06-13): 6 leakage lenses × adversarial verify →
synthesis. 17 "confirmed" of 25 candidates, BUT the synthesis (which read the newer
code incl. `fixed_interface` + the p1 cells) corrected four as false/overstated.

## Verdict
- **§5 cold-start tripwire: CLEAN to lock + launch** (the only immediately-scheduled
  metered run). Its dissociation (C7 collapse vs near-ceiling C3 `ref`) is correctly
  wired; resume-safe (kernel rebuilds an identical config each session ⇒ no dropped
  keys). No real finding contaminates it.
- **Full Gate-1 KILL-scale battery: NOT yet sound** — three un-metered fixes required
  BEFORE that later run (none affect the tripwire).

## Corrected (false / overstated — do NOT act)
- "C7 not implemented" — FALSE. `interface.make_iface_fn` `fixed_interface` mode +
  the p1 `c7` cell implement it (train on one frozen (P*,perm*), eval on novel).
- "C3 unclear / missing" — FALSE. C3 = the `ref` cell (`alpha=None`), a separate
  training run by design (PREREG §5), feeds the 0.70× bar.
- "Only C4/C5/C6 ⇒ controls missing" — misleading. C4/C5/C6 are eval toggles (run
  every eval); C1 logged at init; C3/C7 are training runs (correct).
- "Reward magnitude leaks action correctness" — NOT a Phase-1 leak. Reward feedback
  IS the action-inference channel; eval reward magnitude is constant (reward-affine
  deferred to Phase 3, decision 5); P is orthogonal/norm-preserving (no obs magnitude
  cue); C4 is the dedicated reward-dependence falsifier.

## Real findings — BINDING follow-ups before the Gate-1 battery
1. **[BLOCKER for Gate-1, not tripwire] C8 within-lifetime reshuffle absent.** No
   per-step interface resample anywhere; `full_eval` has no `c8`. C8 is the ONLY
   falsifier separating genuine per-lifetime inference from a randomization-invariant
   heuristic (G1-F). For cbandit-FR it must sit at chance by construction.
   *Fix:* `c8` flag resampling the interface each step from `fold_in(key, IFACE_FOLD+t)`;
   add `c8` to `full_eval`. Un-metered.
2. **[BLOCKER for Gate-1] C7-π / C7-P single-axis cells not scheduled.** PASS = G1-A ∧
   G1-B ∧ **G1-C** ∧ G1-E ∧ G1-FG; G1-C needs C7-π collapse, G1-D needs C7-P. Mechanism
   EXISTS (`fixed_interface` + `alpha_act=0` ⇒ C7-P; `alpha_obs=0` ⇒ C7-π); the gap is
   that `build_kernel` only emits the 3-cell tripwire. *Fix:* generate four cells
   (P-only main, π-only main, C7-P, C7-π) from existing flags. No new substrate code.
3. **[HIGH — fix before any single-axis run] `assert_resume_cfg` one-directional.**
   `train.py` loops only over NEW-config keys; a key present in `saved_cfg` but dropped
   on resume is never validated and silently reverts to its default. B5's "covers every
   key automatically" is FALSE. The damaging case: a resume dropping `alpha_obs` flips a
   P-only run into both-axes, contaminating G1-C/G1-D. Tripwire is safe (no dropped keys).
   *Fix:* backward check `for k in saved_cfg: if k not in _RESUME_FREE: assert k in
   config and config[k] == saved_cfg[k]` + a resume-asymmetry unit test that must RAISE.

## Scope clarification (benign, state in the gate writeup)
- Orthogonal P + onehot context ⇒ the C=5 projected contexts stay mutually orthonormal
  every lifetime, so the obs axis is **in-context binding/clustering, not P-recovery**
  (PREREG §1 defines it exactly this way). Non-vacuity rests on (a) C7-P collapsing and
  (b) C8 at chance — i.e. fixes 1 & 2 are what make this benign. Don't oversell the
  obs-axis result as "matrix inversion."

## Verified clean (don't re-litigate)
no-iface path byte-identical (G0-A 18/18); held-out eval interfaces genuinely disjoint
(train `fold_in(gen_key, IFACE_FOLD)` vs eval `fold_in(EVAL_FOLD∘seed, IFACE_FOLD)`,
continuous P ⇒ collision measure-zero — novelty is over P and (P,perm) pairings, not
perms); CRN intact (interface drawn once/gen, shared across the antithetic population);
norm-preservation / κ=1 (asserted).

## Cheap un-metered guards — ALL DONE 2026-06-13
(1) **resume-asymmetry** — `test_interface.py` T8 (drop `alpha_obs` ⇒ RAISES). ✅
(2) **C8-at-chance** — `test_interface.py` T7 (untrained C8 q4=0.107 ≤ chance). ✅
(3) **obs-axis necessity** — context-blind policies on cbandit-FR max at 0.203 (best
    fixed arm = 1/C = 0.2; uniform = 0.124), ≪ the 0.69 G1-A bar ⇒ the bar cannot be
    cleared without in-context obs binding. Matches analytic exactly. ✅
(4) **C7-P/C7-π smoke** — `build_kernel.py --route p1axes` CPU smoke (4 cells train,
    verdict logic correct). ✅

## Follow-up status (2026-06-13)
All 3 binding findings CLOSED: C8 (`5968b34`), bidirectional resume (`5968b34`),
single-axis cells (`589bcee`). Harness is Gate-1-ready for the cold-start mainline.
