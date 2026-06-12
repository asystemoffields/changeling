# CHANGELING — SPEC v0.1 (draft, gates not yet locked)

A changeling is swapped into a foreign world and thrives there. This project breeds one.

**Mission.** Learning here has to MEAN something: the object of this project is to breed
or discover an aspect — a kernel — of intelligence. Not a benchmark number, not a sealed
function: a learning algorithm that demonstrably exists inside the artifact, survives
being taken out of it, and improves with scale because of what it is. Chased because it's
there. (Operationally: the kernel test, §7 Phase 4.)

**North star.** Build a virtual environment (the **Crucible**) that reliably produces
fixed-weight ANNs which, dropped into *literally different* environments (Catch, MinAtar,
bsuite, bandits they have never seen), adapt and succeed within a single lifetime — then
use SAEs to reverse-engineer the learning algorithm those weights implement, and optimize
it (prune, distill, or extract as math). Grow → extract → reverse-engineer, pursued
directly rather than ecologically.

**Status.** v0.1 design. Numbers marked ⚖ are draft pre-registrations — lock before the
first Phase 0 run, then mg scale discipline applies: KILL only at pre-registered scale,
otherwise PARK with a predicted scale-to-signal.

---

## 1. The core bet

A fixed-weight network cannot transfer a *policy* to a different environment — the
observations, actions, and dynamics it was fit to don't exist there. The only thing that
can transfer is a *learning algorithm executed in activations*. So the artifact we are
breeding is an in-context reinforcement learner: weights = the learning rule, activations
= the knowledge acquired this lifetime.

Therefore the Crucible's job is **not to teach skills**. Its job is to make memorization
worthless and learning-to-learn the only winning strategy. Three levers do this:

1. **Interface randomization.** Every lifetime, observations arrive through a fresh
   random projection and actions through a fresh permutation. The agent can never know
   what its sensors and actuators mean a priori — it must infer the interface from reward
   feedback, every lifetime, forever. (Kirsch & Schmidhuber showed random projections
   force general-purpose in-context learning in supervised settings; we apply it to RL.)
2. **Task-distribution diversity.** Generalization to unseen task families emerges from
   breadth of the training distribution, with an observed phase transition from
   memorization to generalization as diversity grows (GPICL; AdA/XLand at scale). We
   control diversity explicitly and measure the transition.
3. **Lifetime structure.** Multi-episode lifetimes with reward and last-action fed back
   as inputs (RL² protocol); memory persists within a lifetime, resets between lifetimes.
   Fitness weights later episodes more than early ones ⚖ (selects for improvement, not
   performance).

"Breeding" is optimizer-agnostic. The selection pressure is the distribution + objective;
whether the outer loop is ES or SGD is an empirical choice made at Gate 0.

## 1b. Bitter-lesson discipline (binding on all gates)

The bitter lesson is applied as mechanism, not sentiment. AdA is the at-scale existence
proof that diversity + memory + curriculum is what compounds with compute; these rules
keep changeling on that curve despite running at 10^-4 of AdA's budget.

1. **Decide by scaling slope, not intercept.** Every head-to-head (S1 vs S2, R1 vs R2,
   hand-built G vs Weaver) runs at ≥ 3 compute levels (1×/3×/10× of a base budget ⚖) and
   the winner is the better *slope* on the transfer endpoint. An option that wins at 1×
   but loses on slope is chosen only if its lead survives naive extrapolation to 100× ⚖.
   We cannot afford AdA's scale, but we can afford to measure which choice would win there.
2. **Hand-built knowledge is scaffolding, declared as such.** Generator tiers G1–G5
   bootstrap training, but the designated *scaling mechanism* for diversity is learned
   generation (the Weaver, §7-2b). 2b remains gate-triggered for budget reasons; it is the
   long-run mainline, not a side quest.
3. **No hand-coded cognition in the agent.** No intrinsic-motivation bonuses, exploration
   heuristics, or auxiliary losses that encode *how* to learn ⚖. Exploration must emerge
   from the meta-objective and the distribution. Hand-shaped variants are permitted only
   as controls/baselines, never as the mainline.
4. **Substrate must have a compute story.** Prefer architectures whose capability grows
   with params and context/memory (attention-based memory — the lever AdA's ablations
   singled out). S1 (GRU) stays as a control even if it wins early.
5. **Scale-first debugging.** When a gate fails below KILL scale, the first response is
   more distribution/compute within budget, not more machinery. Mechanism is added only
   after a scale bump fails to move the metric ⚖.

## 1c. Contrast case: DiscoRL — what "different" means here

DiscoRL (Oh et al., Nature 2025; Alex maintains the PyTorch port, **disco-torch**)
meta-discovers the *outer* update rule: a small frozen meta-network replacing the
hand-crafted RL loss, applied by SGD to agents that still train from scratch in every
new env. It validates two of our premises at scale — learned learning algorithms beat
hand-designed ones, and discovery quality scales with env diversity (Disco103 > Disco57
on held-out benchmarks). But it is an **anti-goal in artifact shape**:

- The artifact is a sealed function at a fixed abstraction. Improving past Disco103
  means re-running DeepMind-scale meta-discovery; no handles for inspection, editing,
  or incremental improvement.
- No intrinsic scaling story: the rule does not get better as the agent grows or
  experiences more. The scale levers act on per-env training runs, which are not the
  artifact.
- Deploy still costs a full training run per environment.

changeling inverts the locus: the learning algorithm lives in the *inner* loop — the
weights of a deployed agent, executed in activations. That choice buys all three missing
properties: **intrinsic scaling** (params, memory/context, lifetime length, and Crucible
diversity each directly enlarge the learner — exactly the AdA ablation axes), **instant
deploy** (adaptation within one lifetime, no optimizer plumbing), and **handles**
(Phase 4 opens the artifact; extraction is the improvement path DiscoRL lacks — and may
yield a DiscoRL-shaped rule-as-math as a byproduct, with full provenance).

North star, Alex 2026-06-12: *"incredibly potent at small scale, which — due to
intrinsic properties — improves with scale."*

## 2. The interface protocol (fixed for the life of the project)

- Observation: vector in R^D, D = 64 ⚖. Native env obs are mapped in by a random sparse
  orthogonal-ish projection sampled per lifetime.
- Action: K = 8 ⚖ discrete; surplus actions map to no-op; assignment permuted per lifetime.
- Input each step: [proj(obs), one-hot(last action), last reward, episode-boundary bit].
- Reward: scalar, randomly scaled/shifted per lifetime within bounds ⚖ (prevents
  reward-magnitude memorization).
- Lifetime: E episodes (family-dependent, ~8–32 ⚖), total ≤ 4k steps ⚖. Hidden state /
  context persists across episodes within a lifetime, hard reset between lifetimes.

This protocol is the whole trick for "literally different envs": at deploy time, MinAtar
through a random projection is, from the agent's point of view, just another world drawn
from a (broader) distribution. Foreignness is reduced to out-of-distribution dynamics,
which is exactly the axis the Crucible trains.

## 3. The Crucible (training world generator G)

Procedurally generated families, all behind the interface protocol. Diversity is a dial:

| Tier | Families |
|------|----------|
| G1 | multi-armed bandits (Bernoulli/Gaussian, 2–8 arms — must fit K=8 interface) |
| G2 | contextual bandits; sequence/pattern prediction with reward |
| G3 | tabular MDPs: gridworld mazes, keys-and-doors, T-maze memory tasks |
| G4 | spatial games: Catch variants, avoidance, simple pursuit; point-mass control |
| G5 | compositional mixes: G3 dynamics × G4 observations, nonstationary worlds (mid-lifetime task switch) |

Controllable axes within every family: state-space size, horizon, stochasticity, reward
sparsity, partial observability. **The transfer battery (§4) is never in G.**

## 4. Transfer battery — the shared yardstick

Held-out forever; also the yardstick for microcosmic-god artifacts (one battery, two
breeding programs).

- **L0** — unseen tasks from training families (new bandit arms, new mazes).
- **L1** — unseen *family*, same tier character (e.g., hold out keys-and-doors entirely).
- **L2** — different genus through the interface: MinAtar (Breakout, Asterix, Freeway,
  Space Invaders, Seaquest), bsuite (deep_sea, umbrella, discounting_chain), CartPole,
  Catch — via random projection, exactly as in training.
- **L3** — native observations (no projection; learned encoder allowed). Stretch goal,
  explicitly out of scope until L2 passes.

Metrics per deployment (each over ≥ 20 lifetimes ⚖):
- **Slope**: within-lifetime improvement (final-quarter vs first-quarter mean return).
  This is the signature of in-context learning and the primary endpoint.
- **Level**: final-quarter return vs (a) random policy, (b) frozen non-meta policy
  trained on G, (c) small PPO trained from scratch on the target with 10× the
  interaction budget (the "is it worth it" bar).
- **Shuffled-reward control**: same deployment with reward signal permuted → slope must
  collapse to ~0. Proves adaptation is reward-driven, not drift.

## 4b. Watchlist: memory and creativity (standing research objects)

**Memory — first-class, because nobody has it right yet.** In-context learning *is* a
memory claim: acquired knowledge lives somewhere — recurrent state, attention KV, or an
explicit store — and each choice has known failure modes (RNN state: capacity + interference;
attention: O(T) reads, no consolidation; external stores: write-policy collapse). Memory
architecture is therefore the one place where substrate search is mainline rather than a
bitter-lesson violation (§1b-3 bars hand-coded *cognition*; a memory substrate is
connectors, not cognition — the kernel/harness language of Phase 4 applies). Standing
observables, every phase: where does acquired knowledge physically live (probe: which
state, when perturbed, deletes what the agent just learned?); capacity scaling with
lifetime length; interference/forgetting within lifetime; whether consolidation emerges
(early-lifetime knowledge migrating to a cheaper representation). Candidate substrates
for S3 (§5) draw on in-house tech: SVA-style content-addressed reads, hourglass-style
PKM stores.

**Creativity — anticipated foundational, watched not injected.** Per §1b-3 we do not
hand-code it; per the airfoil lens (MDL library learning), the operational definition is
**compression-driven recombination**: a creative learner factors its experience into
reusable parts and composes them in configurations it never experienced. Pre-registered
probes ⚖, run at every gate from Phase 1 on:
- *Compositional probe*: lifetimes containing tasks A and B separately, then a test task
  solvable only by composing skills from both. Score = composition success vs. an agent
  that saw A+B but compositionally-blind controls.
- *Structured exploration*: exploration that is hypothesis-shaped rather than noise —
  measured as compressibility of the action/visit sequence relative to reward surprise
  (random exploration is incompressible; systematic search is not).
- *Phase 4 tie-in*: if creativity is real, SAE space should show experience being
  *factored* (part-features that recombine across tasks). A factored memory code found
  under dissection is the creativity result, and links directly back to airfoil's
  library-learning frame.

## 5. Agent substrates

- **S1 (workhorse)**: GRU, 128–256 hidden ⚖, ~100–500k params. Easiest to meta-train;
  Phase 0 vehicle and permanent control.
- **S2 (interp target)**: 2–4 layer transformer, d=128 ⚖, full-lifetime context (chunked
  /sliding if needed), ~1–2M params. Residual stream is the natural SAE substrate.
- **S3 (memory axis, gate-triggered)**: memory-augmented variants — explicit store with
  content-addressed reads (SVA lineage), PKM-style key-value store (hourglass lineage).
  Enter when §4b memory observables show S1/S2 capacity- or interference-bound.
- S1 vs S2 head-to-head at Gate 1, decided by scaling slope per §1b-1; carry both
  forward if affordable, else the winner plus the other as control.

## 6. Outer-loop routes

- **R1 — ES** (evosax: OpenES/SNES, pop 256–1024 ⚖): fitness = late-weighted lifetime
  return. True "breeding"; embarrassingly parallel; no backprop-through-lifetime issues.
- **R2 — gradient meta-RL** (RL² via PPO over lifetimes, PureJaxRL-style).
- **R3 — fallback: Algorithm Distillation** (Laskin et al.): generate per-task learning
  histories with vanilla RL, train S2 to imitate the *history*, yielding an in-context
  learner offline. Known-good at small compute; use if R1 and R2 both stall, at the cost
  of "bred" purity. Source histories may be generated with **disco-torch** rather than
  hand-crafted PPO — distilling a discovered outer-loop rule into an inner-loop learner.

R1 vs R2 decided at Gate 0 by scaling slope on transfer-per-GPU-hour (§1b-1), not ideology.

## 7. Phases and gates

**Phase 0 — harness + reproduction.** JAX end-to-end (gymnax + evosax + PureJaxRL
patterns), all runs on Kaggle, seeded, checkpoint/resume for session limits. Reproduce
known results: RL²-style in-context learning on G1 bandits and Catch with S1, no
interface randomization yet.
*Gate 0* ⚖: held-out 10-arm bandits — final-quarter best-arm rate ≥ 85%; Catch —
final-episode success ≥ 90% from ~chance at lifetime start. Budget ≤ 24 GPU-h. Failure
here is an engineering bug, not a science result.

**Phase 1 — interface randomization (the load-bearing novelty).** Add per-lifetime
projections/permutations, curriculum on randomization strength (anneal from
near-identity ⚖) if cold-start fails. S1 vs S2, R1 vs R2.
*Gate 1* ⚖: on held-out tasks with novel projections: slope > 0 (sign test across ≥ 20
lifetimes, p < 0.05) and final-quarter ≥ 70% of the un-randomized agent's level. KILL
scale: S2 at ≥ 1M params, ≥ 2e9 env steps; below that, PARK with predicted
scale-to-signal.

**Phase 2 — diversity scaling.** Fixed compute, sweep generator richness over ≥ 5 levels
(G1 → G1:5 + axis ranges). Deliverable: the transfer-vs-diversity curve; we are hunting
the memorization→generalization phase transition on L1.
*Gate 2* ⚖: L1 transfer strictly improves with diversity over the sweep. Flat curve at
scale → PARK with prediction.

**Phase 2b — the Weaver (optional, gate-triggered).** Replace/augment the hand-built
generator with a *world-steering NN*: the Weaver parameterizes the generator's dials
(and eventually level layouts) and is rewarded by **agent learning progress** — the
within-lifetime slope of the agents it spawns. The agent's rewards stay vanilla. This is
unsupervised environment design (PAIRED, POET) with learning-progress in place of
regret — a teacher whose objective *is* the thing we're selecting for.
Known failure mode, pre-registered: teachers game progress metrics (e.g., worlds with
artificially low early performance). Controls: slope must be measured against a frozen
reference agent population, not only the current one; Weaver-proposed worlds must keep
the shuffled-reward control negative; cap Weaver control over reward structure.
*Trigger*: enter 2b if the Phase 2 curve is positive but saturates (curriculum bottleneck,
not capacity bottleneck). *Gate 2b* ⚖: Weaver curriculum beats the best hand-built
diversity level on L1 transfer at equal compute — judged on scaling slope per §1b-1.
Per §1b-2, hand-built G is scaffolding; the Weaver is the designated long-run diversity
mechanism.

**Phase 3 — L2 transfer.** Deploy on MinAtar/bsuite/CartPole through the interface.
*Gate 3* ⚖: on ≥ 2 of 5 MinAtar games — slope > 0, final-quarter ≥ 3× random-policy
return within a 10k-step lifetime, shuffled-reward control negative. (Not DQN parity —
DQN gets millions of frames; we get one lifetime. Beating random by 3× zero-shot-ish on
unseen games is the honest, publishable bar.)

**Phase 4 — dissection (the paper's second act).** SAEs on residual stream (S2) /
hidden state (S1) across lifetimes, via interp-lab (dogfood: plan-evidence, causal
criteria). Hunt features tracking offline-computed quantities: TD/reward-prediction
error, action-value, novelty/exploration drive, lifetime-phase, interface-identity
(which permutation am I in?).
*Gate 4* ⚖: ≥ 1 causally validated **learning-relevant** feature: ablation cuts the
within-lifetime slope by ≥ 50% while first-episode (pre-learning) performance drops
≤ 10%. That dissociation — knocking out *learning* while sparing *acting* — is the
headline figure.
Then optimize: prune deploy-time-dead capacity; regress feature dynamics to extract the
update rule as math; distill into a smaller net. "We bred a learner, found the learning
rule inside it, and took it out" is the paper.
**The kernel test** (what "took it out" must mean): the extracted rule, re-instantiated
*outside the bred body* — as explicit math, or re-implemented in a fresh minimal
substrate — must itself produce in-context learning on the transfer battery (positive
slope, C4 shuffled-reward control negative) ⚖. A description of the mechanism is not a
kernel; only a working re-implementation proves we extracted the algorithm rather than
annotated it. This is the bar that separates "we found features correlated with learning"
from "we discovered a kernel of intelligence."
**Kernel vs harness.** The kernel is not required to be self-powering — a battery needs
connectors. The re-instantiation may include a declared **harness**: supporting structure
the kernel needs to apply itself (interface encoder/decoder, a memory substrate to write
to, plumbing for reward feedback). Three rules keep the connectors honest ⚖:
1. *Declared boundary*: the kernel/harness split is written down before the test runs.
2. *Null-kernel control*: the harness with the kernel removed (or replaced by a matched
   null rule) must show NO learning slope. If the connectors light up without the
   battery, the extraction failed — the harness was the learner.
3. *Scope, not residence*: the kernel should function in ≥ 2 materially different
   harnesses (e.g., explicit-math host and fresh-net host), and should not be
   intrinsically pinned to one size — what we claim to have found is the scope and
   scalability of the rule, not its first body or its connectors.

## 8. Controls (standing, every phase)

C1 random-init agent · C2 frozen non-meta policy (same arch, trained on one env) ·
C3 meta-trained *without* interface randomization · C4 shuffled-reward deployments ·
C5 ablated-memory deployment (hidden state reset every episode — in-context learning
should vanish).

## 9. Compute plan

All training on Kaggle (T4/P100/TPU, ~30 GPU-h/wk; box stays free per standing policy).
JAX end-to-end so envs live on-device: gymnax tiers G1–G4 run at ≥ 1e6 steps/s vectorized;
a full Phase 0 meta-train is ~1e9 steps ≈ a session or two. Sessions are 9–12 h →
checkpoint/resume mandatory from day one. Rough envelopes ⚖: P0 ≤ 24 GPU-h, P1 ≤ 60,
P2 ≤ 60, P2b ≤ 40, P3 ≤ 30, P4 mostly CPU/local-light + activation-capture runs.

## 10. Risk register

| Risk | Mitigation / disposition |
|------|--------------------------|
| Meta-RL instability (R2) | R1 ES as co-equal route; R3 AD as known-good fallback |
| Interface randomization too hard cold | curriculum anneal from near-identity |
| Diversity needed exceeds Kaggle compute | Phase 2 curve extrapolation → PARK with scale prediction, not KILL |
| Weaver games the progress metric | frozen reference population, reward-structure caps, C4 must stay negative |
| Transfer confounded (lucky priors, not learning) | C4/C5 controls; slope is primary endpoint, not level |
| SAE features trivial/correlational | interp-lab causal criterion is the gate; correlation alone fails Gate 4 |
| L3 (native obs) doesn't follow from L2 | declared stretch; paper stands on L2 + dissection |

## 11. Relationship to microcosmic-god

Same destination, different route. mg = ecological: evolve conditions under which
perception-coupled learners emerge by selection. changeling = direct: specify the
selection pressure analytically and optimize against it. Shared infrastructure: the
transfer battery (§4) and the Phase 4 dissection pipeline are common property — mg
artifacts get dropped into the same battery, so results compare on one yardstick.
changeling absorbing the transfer goal also serves mg's "cut dilution, review against
one axis" directive: mg keeps the selection axis pure.

## 12. Anchors

RL² (Duan et al. 2016) · Learning to RL (Wang et al. 2016) · GPICL: General-Purpose
In-Context Learning by Meta-Learning Transformers (Kirsch & Schmidhuber 2022) · VSML
(Kirsch & Schmidhuber 2021) · AdA / XLand 2.0 (DeepMind 2023) · Algorithm Distillation
(Laskin et al. 2022) · PAIRED / UED (Dennis et al. 2020) · POET (Wang et al. 2019) ·
PLR (Jiang et al. 2021) · Teacher-student curriculum & learning progress (Matiisen et
al.; Oudeyer) · Differentiable plasticity (Miconi et al. 2018) · MinAtar (Young & Tian
2019) · gymnax / evosax (Lange) · PureJaxRL (Lu et al. 2022).

---
*v0.1 — 2026-06-12. Next action: lock the ⚖ numbers, then Phase 0 harness build.*
