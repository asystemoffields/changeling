# Exploration-economics assay — what breeds "wanting to explore"?

Question (from the fossil): the ES bandit agent settled (satisficing lock-in).
Which selection-pressure change breaks settling, within the bitter-lesson rule
that exploration must be bred, never injected? 2x2: {fitness late-ramp vs
final-quarter-only} x {arms uniform U(0,1) vs needle 0.9-vs-0.1}, toy ES
(800 gens, h=32, single seed per cell — assay, not gate). Fifth specimen: the
R2 PPO session-1 agent (h=128, uniform world).

Metrics: arms25 = distinct arms tried in first 25 pulls (8 = full sweep);
P(rep|L) = P(repeat | last pull lost); gate_q4 = final-quarter best-arm rate.

```
condition                                           arms25  P(rep|L)  gate_q4  rew_q4
ES uniform fitness=late                               3.55     0.868    0.281   0.687
ES uniform fitness=q4                                 2.22     0.806    0.230   0.662
ES needle fitness=late                                4.09     0.541    0.690   0.653
ES needle fitness=q4                                  3.45     0.440    0.542   0.534
R2 PPO s1 agent (uniform, late, h=128)                3.58     0.881    0.494   0.822
ES fossil h=128 reference                             2.77     0.81     0.248   0.637

transfer probe: needle-bred(late) agent on UNIFORM     2.64     0.847    0.321   0.743
```

## Findings

1. **The world teaches wanting; the fitness schedule didn't.** Needle structure
   alone tripled best-arm rate (0.281 -> 0.690) and halved post-loss
   perseveration (0.87 -> 0.54). In the needle world a loss is strong evidence
   you're on a dead arm, and the 0.8 reward gap is visible through ES's noise.
   The uniform world breeds campers under every schedule tested because its
   statistics make camping nearly rational.
2. **Final-quarter-only fitness HURT in both worlds** (0.281->0.230 uniform,
   0.690->0.542 needle). At matched compute it just quarters the fitness
   sample without changing what pays. Honest negative for the repricing
   hypothesis at this scale; the late-ramp stays.
3. **P(rep|L) is not a pure wanting-metric.** The strong R2 agent shows
   P(rep|L)=0.88 — but staying after an unlucky loss on a good arm is
   *correct*. Read it jointly with gate_q4: the fossil's sin was staying on
   bad arms, not staying per se.
4. **Dispositions transfer as contingencies, not behaviors.** The needle-bred
   agent dropped into the uniform world re-collapses on surface metrics
   (arms25 2.64, P(rep|L) 0.85) yet beats the uniform-bred native on both
   best-arm (0.321 vs 0.281) and reward (0.743 vs 0.687). What transferred is
   a world-reading rule, not an exploration rate — and the harsh-world rule
   remains better even in the soft world.

## Crucible implications

- Tier design needs identification-forcing reward structure (needle-like
  contrast), not just uniform task soup; forgiving distributions breed
  satisficers at every objective we tested.
- "Hard worlds make better Bayesians" — and their advantage survives in soft
  worlds. Supports diversity tiers that *include* harsh contrasts rather than
  interpolating gently.
- Caveats: toy scale, single seed per cell, ES only. Multi-seed + R2 cells
  before any of this enters a locked prereg.
