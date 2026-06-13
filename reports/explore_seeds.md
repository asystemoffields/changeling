# Exploration-economics, multi-seed hardening (toy ES h=32, 800 gens, seeds 0–3)

Hardens the **single-seed** `explore_matrix.py` assay. Every cell is trained on
its curriculum and evaluated on the **pure uniform gate** (n=1000 lifetimes) —
the prereg-relevant held-out readout. Toy ES is a sensitive-but-low-fidelity
detector (the optimizer that *settled*); effect *sizes* matter, not absolute
levels. Run on the hardened harness (each cell logged "grades gate on
eval_env=bandit8 kwargs={}", confirming the ES eval_env-separation fix works).

```
cell (eval=uniform)   gate_q4        arms25        p_rep_loss     reward_q4
uniform           0.253±0.034   3.779±1.112   0.655±0.215   0.644±0.046
needle            0.246±0.035   4.257±0.633   0.640±0.161   0.663±0.050
mix0.25           0.270±0.023   3.625±0.460   0.645±0.019   0.681±0.016
mix0.50           0.294±0.024   3.756±0.872   0.770±0.115   0.694±0.023
mix0.75           0.237±0.027   3.264±0.802   0.672±0.113   0.653±0.031
```

## What replicates, what doesn't

**ROBUST — a modest mix≈0.5 curriculum effect with an inverted-U.** mix0.50
(0.294) beats pure uniform (0.253) by ~+0.04 (≈1.2σ), mix0.25 (0.270) is
intermediate, and **mix0.75 (0.237) falls back to ≤ uniform** — over-mixing
hurts uniform-gate transfer. This *validates the D2 mix=0.5 choice as near-optimal*
and is directionally consistent with the gate-scale result (D2: mix=0.5 broke the
0.52 plateau → 0.62 at hidden=128, replicated across 3 independent n=1000 scores).

**DOES NOT REPLICATE — the strong single-seed "world teaches wanting" transfer
claims:**
- *"Needle contrast tripled ES best-arm (0.281→0.690)"* was a **task-difficulty
  artifact**: the 0.690 was needle-bred evaluated on the *needle* task (where the
  best arm pays 0.9 — easy). On the apples-to-apples **uniform** gate, needle-bred
  (0.246) ≈ uniform-bred (0.253). No transfer advantage.
- *"Needle-bred beats uniform-bred ON uniform / dispositions transfer as
  contingencies"* (single-seed 0.321 vs 0.281) **flips to parity** under 4 seeds
  (0.246 vs 0.253). The single-seed gap sat inside the ~0.03 seed σ.
- *"Needle halved P(rep|loss) and tripled exploration"* is **within seed noise**:
  arms25 4.26 vs 3.78 (~0.5σ), and P(rep|loss) shows no clean monotone signal
  (mix0.5 is actually highest at 0.77).

## Bottom line for PREREG_P1

Treat curriculum as a **mild, real prior** (mix≈0.5, not pure needle, never >0.5),
NOT a silver bullet, and **do not** carry the strong needle-transfer /
exploration-economics claims into the locked prereg — they are single-seed
artifacts. The "harder worlds breed better Bayesians, advantage survives in soft
worlds" framing is **not** supported at toy scale on the uniform readout; the only
defensible curriculum claim is the small mix≈0.5 effect (which Phase 1 should
re-test at scale, not assume). An honest negative the single-seed assay could not
see — exactly why multi-seed precedes the lock.
