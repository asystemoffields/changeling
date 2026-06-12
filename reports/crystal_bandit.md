# Crystallization pilot — ES bandit fossil (Gate 0 session 1)

Specimen: GRU-128 (78,984 params) bred by OpenES, plateaued at gate_q4 ≈ 0.25
for 2,800 generations. Question: what strategy did ES actually carve, and does
it survive extraction into algorithmic form?

## Forensics (2,000 trace lifetimes)

- **Barely explores**: 2.77 distinct arms tried in the first 25 steps (of 8).
- **Massively perseverative**: P(repeat|win)=0.89, P(repeat|loss)=0.81 — it
  repeats even after losses; win/loss barely modulates staying.
- **Not value-tracking**: P(action = argmax running mean) = 0.275 in q4.
- Earns 0.637 reward (random = 0.5) at only 0.22 best-arm — it camps on the
  first *decent* arm it stumbles into, not the best one.

**Named strategy: satisficing lock-in.** Drift until an arm pays above
baseline, then camp, with weak value drift underneath. This explains the ES
plateau mechanistically: the organism never explores enough to *find* the best
arm, and ES can't add exploration because exploration costs immediate reward
and the fitness signal at M=8 is too noisy to see past that cost.

## MDL ladder (all candidates played on 4,000 fresh lifetimes, final quarter)

```
family                                                   params  NLL/step  best-arm  reward
fossil GRU-128 (organism)                                 78984         -     0.223   0.637
C1 WSLS fit (stay|w=0.89, stay|l=0.82)                        2         -     0.152   0.535
C2 eps-greedy-ish fit {alpha 0.05, beta 50}                   3    2.9206     0.268   0.705
C3 softmax-Q+stick fit {alpha 0.05, beta 4, stick 2}          4    0.5583     0.473   0.796
C3* tuned-for-reward (ceiling, NOT an extraction)             4         -     0.690   0.863
C4 distilled GRU-4                                          988         -     0.139   0.520
C4 distilled GRU-8                                         2064         -     0.147   0.530
Thompson reference                                            -         -     0.747       -
```

## Findings

1. **The fossil's MDL knee is 4 parameters**: softmax-Q with learning rate
   0.05, inverse temperature 4, perseveration bonus 2 (dominant term), zero
   init. NLL 0.558/step, far below the 3-param family. The fossil *is* a
   weak Q-learner buried under a perseveration habit.
2. **The extracted crystal outplays its organism 2×**: 0.473 best-arm / 0.796
   reward vs the fossil's 0.223 / 0.637. Stripping the body's noise while
   keeping its rule improves it — compression-as-regularization, observed.
   First ratchet click of the Phase 5 crystallization loop, three phases early.
3. **Trace-fit is not the right sole criterion**: the distilled GRUs fit the
   traces *better* than C3 (CE ≈ 0.42–0.46 < 0.558) but play far worse
   (0.14) — behavior cloning latched onto "repeat last action," which
   self-reinforces into camping on the first arm. The verify-by-playing step
   of the crystallization loop is load-bearing, not optional.
4. **The family ceiling is near-Thompson**: 4 params tuned for reward reach
   0.690 / 0.863 (92% of Thompson) via optimistic init — context for R2's
   0.540 and for what the recalibrated gate (0.672) demands.
