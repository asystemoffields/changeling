# PREREG — Phase 0 (LOCKED 2026-06-12)

Locks Gate 0 only. Later gates remain ⚖ drafts until their phase begins (staged
pre-registration: lock each phase's numbers before that phase's first metered run).

## Setup

- Interface: D=64, K=8. Identity projection and identity action map in P0 —
  randomization begins Phase 1. Input x_t = [obs(64), onehot(a_{t-1})(8), r_{t-1}(1),
  episode-boundary(1)] → 74 dims.
- **Bandit-8**: Bernoulli, p_i ~ U(0,1) iid per lifetime, 8 arms (= full action width,
  the hard edge of the interface); lifetime = 200 one-step trials.
- **Catch**: 10×5 board, 3 effective actions (a∈{0,1,2} = left/stay/right; 3–7 → stay),
  9-step episodes, 32 episodes/lifetime (T=288). No task variation in P0.
- Substrate: S1 GRU, hidden=128 (~79k params), categorical action sampling.
- Route R1: OpenES — mirrored sampling, centered-rank shaping, Adam(lr=0.02, β=.9/.999),
  σ=0.03, pop=256, M=8 lifetimes/member, common task seeds across population per
  generation. R2 (PPO-RL²) must be implemented before the Gate 0 route decision;
  R1 vs R2 judged on scaling slope per SPEC §1b-1.
- Fitness: late-weighted mean reward, w_t = linspace(0.5, 1.5, T), normalized.
- Eval: 100 held-out lifetimes; eval seed space disjoint from training seed space.
- All runs seeded; checkpoint/resume from day one.

## Gate 0 criteria (locked)

| ID | Criterion |
|----|-----------|
| G0-A | Bandit in-context learning: final-quarter best-arm rate ≥ 0.85 AND within-lifetime slope > 0 |
| G0-B | Bandit C4 (reward input → fair coin): final-quarter best-arm rate ≤ 0.225 (chance 1/8 + 10pp) |
| G0-C | Bandit C5 (memory reset every trial): final-quarter best-arm rate ≤ 0.225 |
| G0-D | Catch plumbing: final-quarter episode success ≥ 0.90. *No slope requirement*: P0 catch has no task variation, so its policy is weight-learnable; demanding a slope would be theater. Catch slope requirements begin Phase 1 with interface randomization. |
| G0-E | Budget: cumulative Kaggle GPU ≤ 24 h for all of Phase 0. Local CPU smoke unmetered. |

P0 reproduces known results (RL², Duan et al. 2016): failure here is an engineering
bug, not a science result — no KILL/PARK semantics apply.

## Amendments noted at lock

- SPEC G1 arm count corrected 2–16 → 2–8: interface K=8 cannot express ≥9 arms.
- Phase 0 harness is dependency-free pure JAX (no gymnax/evosax): OpenES is ~20
  transparent lines, API-drift risk removed, runs on Kaggle's preinstalled JAX with no
  pip step. gymnax returns at Phase 3 (MinAtar).

## Addendum 2026-06-12 — R2 config recorded before its first gate run

R2 (PPO-RL²) hypers, fixed before any R2 gate result was seen: B=64 lifetimes/update,
lr=5e-4, γ=0.99, λ=0.95, clip=0.2, 4 epochs/update, vf_coef=0.5, ent_coef=0.01,
max_grad_norm=0.5, training rewards scaled by (1−γ) so returns are O(1). Value head is
training-only scaffolding; the deploy artifact is the identical S1 GRU as R1.
Provenance: R2 v1 had a real bug — unscaled lifetime returns (~40) made the value-loss
gradient drown the policy gradient through the shared trunk (entropy never left
uniform). Caught at toy scale, fixed by reward scaling. v1 results discarded.

Session log: ES gate session 1 (2026-06-12, ~0.12 GPU-h): G0-D PASS (catch 1.000, early-stop
gen 199); bandit G0-A slope/G0-B/G0-C PASS but gate_q4 plateaued ≈0.25-0.31 from gen 200
through gen 2999 — converged local strategy, not budget shortfall. R1 bandit attempts
beyond locked hypers (e.g., M>8) would be deviations and must be logged here first.

**Interpretation note (2026-06-12, no criterion change):** C5 resets recurrent state
but the RL² input channel still carries last-action/last-reward, so a memoryless
policy can express one-step win-stay reflexes (~0.15–0.22 best-arm). C5 therefore
bounds the *non-recurrent* component of performance, not "no memory" absolutely.
R2 session 2 measured C5 = 0.219 vs main 0.515: ≥ 0.30 of its performance is genuine
recurrent in-context learning. A full-amnesia variant (also zeroing the input channel)
joins the control suite from Phase 1.

## Deviations log (append-only)

- **D1 (2026-06-12) — G0-A recalibrated from absolute to reference-relative.**
  The locked bar (final-quarter best-arm rate ≥ 0.85) was set without computing a
  feasibility ceiling. Agent-independent reference simulation
  (`scripts/reference_bandit.py`: Thompson sampling, Beta(1,1) priors, 4000 lifetimes,
  exact gate task distribution and horizon) yields **0.747**; UCB1 yields 0.404. The
  0.85 bar exceeds what near-optimal play achieves — a broken gate, checkable without
  reference to any agent. G0-A becomes: **final-quarter best-arm rate ≥ 0.90 × Thompson
  reference = 0.672** (the RL² Gittins-relative convention). Bias acknowledged: this
  deviation was written after observing agent values (ES 0.248, R2 0.540); mitigations:
  the reference computation never sees the agent, the 90% fraction follows literature
  convention rather than fitting our numbers, and the recalibrated bar does NOT pass the
  current best agent. Slope, C4, C5, G0-D unchanged.

- **D2 (2026-06-12) — R2 training distribution becomes a curriculum mixture.**
  Both routes plateau under pure-uniform training (ES ~0.25; R2 ~0.52 at 8k and at
  fresh 32k updates — more compute does not help; R2's entropy collapses to 0.01,
  i.e. the optimizer settles in policy space). Toy-scale curriculum test
  (`scripts/mixture_test.py`, 4 cells, matched compute, evaluated on the UNIFORM gate
  task): mixture-trained (50% needle lifetimes) 0.333 vs uniform-trained 0.241;
  ent_coef=0.03 alone 0.241; mixture+ent worse than mixture alone. Therefore the next
  R2 gate attempt trains on mix=0.5 (each lifetime: needle task w.p. 0.5, else
  uniform). **The gate evaluation task is unchanged** (pure uniform, same seed
  protocol); early-stop listens to the gate task, not the training distribution.
  Logged before the gate-scale run. Note: this converts G0-A from pure reproduction
  into a miniature of the project thesis — training-distribution breadth buying
  held-out performance.
