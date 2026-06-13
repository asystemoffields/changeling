# Crystallization — uniform-bred fossil vs mixture-bred organism

Companion to `crystal_bandit.md`. Same dissection pipeline
(`scripts/crystallize_bandit.py`, now PPO-aware) applied to the **mixture-bred R2
PPO agent** (s2-mix, `runs/kaggle_r2_s2_mix/.../ckpt_best.npz`, gate_q4 ≈ 0.62),
dissected on the **pure uniform** bandit — directly comparable to the
uniform-bred ES fossil (gate_q4 ≈ 0.25). Sharp question (from the handoff): *did
the harder world breed perseveration DOWN and value-tracking UP?*

## Forensics (uniform task, 2–4k trace lifetimes)

| statistic | fossil (uniform-bred ES) | mixture-bred PPO | direction |
|---|---|---|---|
| distinct arms in first 25 steps | 2.77 / 8 | **3.76 / 8** | explores MORE |
| q4 P(action = argmax running mean) | 0.275 | **0.909** | value-tracks, hugely |
| P(repeat \| loss), early (q1) | ~0.81 | **0.692** (31% switch) | abandons losers MORE |
| P(repeat \| loss), late (q4) | — | 0.964 | correctly holds known-best through stochastic dips |
| organism best-arm / reward (q4) | 0.223 / 0.637 | **0.598 / 0.850** | 2.7× best-arm |

## MDL knee — both families knee at 4 params (softmax-Q + perseveration)

| | fossil crystal | mixture crystal |
|---|---|---|
| learning rate α | 0.05 | **0.10** |
| inverse temp β | 4 | **16** |
| perseveration (stick) | 2 | **2** (unchanged) |
| NLL/step | 0.558 | **0.146** (much tighter fit) |
| crystal best-arm (played) | 0.473 | **0.632** |
| organism best-arm | 0.223 | 0.598 |
| 4-param family ceiling (C3\*) | 0.690 | 0.690 |

## Answer to the sharp question

**Value-tracking: UP, decisively. Perseveration: UNCHANGED.** The curriculum did
*not* breed the stickiness out — the perseveration term is 2 in both crystals.
What it bred is a **sharper, faster value signal**: α doubled (0.05→0.10), β
quadrupled (4→16), and the forensic value-tracking probability went 0.275→0.909.
The fossil was "a weak Q-learner buried under a perseveration habit"; the mixture
organism is a *sharp* Q-learner with the same habit now dominated by the value
term. The early/late P(repeat|loss) split confirms the mechanism is value, not a
lower stick: early it drops losing arms (a loss lowers Q → switch), late it holds
the identified best arm through its 10% stochastic misses (Q stays highest).

So "the world teaches wanting" cashes out, at the mechanism level, as
**sharpening the value-tracking gains, not removing the perseveration prior.**
The compact-strategy family ceiling is identical (0.690 ≈ 92% Thompson) for both
worlds — the curriculum's whole effect is to move the organism *up toward the
fixed ceiling* (0.22 → 0.60; 32% → 87% of ceiling), by tuning α and β, not to
change what a 4-param rule can achieve.

## Replicated secondary findings

- **Compression-as-regularization** clicks again: the extracted 4-param crystal
  outplays its own organism (0.632 vs 0.598). Smaller margin than the fossil's 2×
  (0.473 vs 0.223) — the sharper organism has less body-noise left to strip.
- **Verify-by-playing is load-bearing**: distilled GRU-4/8 fit the traces even
  *better* than the fossil's distillations (CE 0.16–0.29) but play far worse
  (0.15 / 0.23) — BC again latches onto "repeat last action." Trace-fit ≠ play.

## Implication for Phase 4 / Phase 5

The kernel the curriculum breeds is legible and extractable: `softmax-Q(α=0.1,
β=16) + perseveration(2)`. This is the Phase-4 kernel-extraction story in
miniature — and it sets a concrete dissection target for Phase 1, where interface
randomization should force the value-tracking to operate through an *inferred*
action map rather than a fixed one. Prediction to test in Phase 1: under action
permutation, the extractable rule must add an interface-inference component
(arm-identity binding) on top of this softmax-Q core, or it collapses.
