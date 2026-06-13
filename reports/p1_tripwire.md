# §5 cold-start tripwire — result (2026-06-13)

Kernel `asystemoffields/changeling-p1-tripwire` (cbandit-FR, GRU-128, R2 PPO
D3-reconciled γ=1.0 reward_scale=1/256, 3 cells, ~1e8 steps/cell, wall 3247s).

| cell | α | gate_q4 | slope | sign_p | reading |
|------|---|---------|-------|--------|---------|
| REF  | 0   | **1.000** | -0.0012 | — | un-randomized cbandit-FR is at CEILING (C3=1.000, > predicted 0.98) |
| COLD | 1   | **0.122** | -0.0021 | 0.985 | cold-start FAILS — at chance, == C6 (0.122); no within-lifetime slope |
| C7   | 1 (fixed-iface, NOVEL eval) | 0.121 | +0.0007 | — | collapses to chance ✓ |

## Verdict (de-risked all 4 §5 targets)
1. **cbandit-FR learnable:** YES, perfectly, at α=0 (REF=1.000). Env + fixed rule +
   reward + D3-PPO all correct.
2. **Cold-vs-anneal:** **cold-start FAILED.** Pre-committed trigger FIRED (slope
   sign-p=0.985 ≥ 0.05 AND q4=0.122 ≤ C6+0.10=0.222) ⇒ **ANNEAL is now the mainline
   schedule for ALL Phase-1 runs** (applied uniformly so it can't confound the
   substrate/route slopes — PREREG §5).
3. **Inference-not-memorization dissociation:** PRESENT (REF ceiling vs C7 chance).
   Note: COLD is also at chance, so at GRU-128 *cold-start* the interface isn't
   inferred at all — exactly what the trigger caught.
4. **D3 PPO stability:** 3 cells trained NaN-free at γ=1.0. reward_scale landmine clear.

## Decision (pre-registered rule applied autonomously)
Switch to the **anneal**: master α on grid {0,0.1,0.2,0.35,0.5,0.7,0.85,1.0},
performance-gated advance, graded always at α=1. Build the α-schedule feature, re-run
an **annealed tripwire** (single cbandit-FR GRU-128 R2 cell) to confirm α=1 becomes
learnable via the curriculum (slope>0, q4 ≫ C6=0.122) BEFORE the capacity/route battery.
If the anneal ALSO fails at GRU-128, the next contingency is the capacity dimension
(GRU {256,512}) — but that is the following gate, not this one.

Artifacts: `runs/p1_tripwire_out/` (ckpts + logs). REF ckpt = a perfect α=0 cbandit-FR
solver (C3 reference, reusable).
