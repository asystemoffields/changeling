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

## 1d. Two timescales: a memoryless core that consolidates (added 2026-06-13, Alex)

The changeling learns on two timescales, strictly ordered.

- **Fast (inner, in activations) — THE CORE.** A fresh instance learns each lifetime from
  scratch, in its recurrent activations, through a randomized interface (the §1 bet). This
  core is *memoryless across lifetimes* — it carries nothing in from past experience — and
  it **must be incredibly potent that way**: it is the cognition, the headline result, and
  the primary optimization target at every gate. ("Memoryless" = no cross-lifetime carry;
  the *within-lifetime* working memory the learning algorithm runs on is essential and is
  exactly what C5/C6 ablate — the core is not amnesiac within a lifetime.)
- **Slow (outer, into weights/store) — consolidation / neuroplasticity.** A *particular
  instance* bakes recurring experience into its slow weights (and/or an explicit store), so
  a more-experienced instance carries the benefit of what it has lived — AlphaGo-shaped:
  fast in-context improvement (≈ MCTS) amortized back into slow weights (≈ self-play),
  round after round. This is **strictly additive**. Its only jobs are to stop hard-won
  in-context experience from being *wasted*, and to carry the intrinsic-scaling story
  (North Star). It is never the cognition.

**Discipline (binding):**
1. Nail the memoryless core first; add consolidation only once the core is excellent
   standalone.
2. **Null-memory control (mandatory).** Ablate consolidation/memory ⇒ the core's competence
   must be *undiminished*. Memory propping up the core is a failure, not a result — the same
   "connectors-not-cognition" / null-kernel rule (§4b, Phase 4) applied to the slow loop.
   Memory's value is reported only as a *separate* delta: experience retained + scaling slope.
3. **Randomization protects plasticity.** Consolidating experience into weights ordinarily
   risks the learning algorithm atrophying into a lookup table (the §1c anti-goal).
   Per-lifetime interface randomization is the regularizer that prevents this: because the
   interface is never stable enough to memorize, the general learner stays load-bearing
   forever, so the slow loop can safely consolidate genuine recurring *task/domain* structure
   without eating the inner-loop learner. The stability–plasticity dilemma is resolved by the
   environment, not a hand-tuned mechanism.

**Eventual gate (triad, scale-gated like the rest):** (a) the memoryless core is excellent
standalone [PRIMARY]; (b) an experienced instance beats a fresh one on *recurring* structure;
(c) the experienced instance still learns *novel* structure from scratch as well as a fresh
one does (plasticity preserved), and (b) vanishes under the null-memory ablation.

**Mechanism — (a) FORECLOSED 2026-06-13 (Alex); decide (b) vs (c) empirically (bitter-lesson).**
(a) continual outer loop / shared-weight continual fine-tune is **foreclosed for the gated
claim**: it diffuses consolidated information into the shared trunk, which makes the mandatory
null-memory ablation (a one-checkpoint subtraction) structurally unsatisfiable and the §1d gate
vacuous (see §5b/S3-M.4, S3-M.6). Consolidation writes ONLY to a separable, zero-able,
stop-gradiented slow bank. Remaining choice: (b) explicit
slow/fast two-weight substrate (fast weights / S3 store do the in-context part, periodic
consolidation into slow weights — CLS / fast-weight-programmer lineage); (c) online
crystallization-into-weights (distill the inner loop's reliably-good behavior back into slow
weights — the most direct AlphaGo analog; reuses the Phase-5 crystallization machinery).
Connects: S3 (§4b) is the fast store; the crystallization ratchet (Phase 5) is the
slow-consolidation operator; the Phase-4 kernel test is "a fresh kernel still learns from
scratch." Phase 1 establishes the memoryless core; this axis enters *after*, never as a
Phase-1 crutch.

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

## §5b / S3-M — Memory + Compute substrate (looped core · adaptive-K micro-turns · selective consolidation)

**Status.** v0.1, committable. **Spine = the *minimal-robust* design** (strict superset of the current harness: K_max=1 + zero-store + zero-slow-bank collapses exactly to today's `gru_step`/`collect`/`loss_fn`). Realized in JAX (port the SVA/hourglass *mechanisms* — content-addressed read/write, looped core — to JAX; they are PyTorch, so we reuse the mechanism not the model). Grafts attributed inline: dual value heads and deferred-write-on-RL²-channel from *maximal-ambitious*; literature scaffolding (PonderNet/PKM/NEC/MERLIN/DNC/Titans/EWC) from *literature-faithful*; the **W-window store-BPTT**, **capacity-env staging**, **phase-conditioned probe**, and **reward-aligned halting supervision** are synthesizer additions. Folds into the PREREG_P1 **B1** abstract-`step_fn` refactor; honors §1d ordering, §1b bitter-lesson, §4b (memory = connectors, not cognition). Build follows `reports/memory_arch_risks.md` (binding risk register + first increment).

### S3-M.0 Sequencing (binding, mirrors §1d)
**Scope ratified 2026-06-13 (Alex, decision 1):** **Phase 1 ships Increment A ONLY** (looped
core + adaptive-K micro-turns, NO store). Increments **B (store) + C (consolidation) are gated
to a dedicated S3 mini-phase** with its own memory-demand env — cbandit-FR cannot test store
capacity (flat-by-construction, S3-M.5). Calibration knobs (S3-M.1) are **ratified pre-reg
numbers**, not ⚖. Write-route: **ES is the legitimate mainline for the memory-write axis** (the
W=1 bracket, decision 2) — "PPO trains writes" is a measured bracket, not a hard requirement.

1. **Increment A — looped core + adaptive-K (no store, no slow bank).** Weight-tied core run K_max micro-turns/env-step, PonderNet-marginalized. Ships into Phase-1 cbandit-FR; the **compute-per-decision (K_max) axis** is the only memory/compute axis Phase-1 can exercise (interface inference is compute-bound, not capacity-bound).
2. **Increment B — fast content-addressed store.** A per-lifetime-carry member. Its **capacity (N_slots) and lifetime-length slopes are NOT graded on cbandit-FR** (S3-M.5) — graded on a dedicated memory-demand env.
3. **Increment C — selective consolidation (slow bank).** Strictly-additive, zero-able, deferred until the memoryless core is excellent standalone at α=1. Never a Phase-1 crutch.

### S3-M.1 Looped core + adaptive-K micro-turns (PonderNet, marginalized)
Replace `gru_step` in the step body with `micro_loop(params, carry, x) -> (carry, commit_logits, commit_v, aux)`, a **static `lax.scan` of length K_max nested inside** the per-env-step scan over T. At the GRU rung the core is `gru_step` applied K_max times to an internal thinking input (hourglass shape; recurrent DEPTH over fixed context, not sequence extension → attention stays O(T²)). Per micro-turn k: query store → read r_k; refine m_{k+1}=core(m_k,[enc(x),r_k]); emit (logits_k, v_k, halt_logit_k). λ_k=σ(halt_logit_k); PonderNet halting mass p_k=λ_k·∏_{j<k}(1−λ_j), residual on K_max so Σp_k=1; **K_min≥2 floor**.

**COMMIT THE MARGINAL MIXTURE, never a discrete halt sample** (over-determined — forced jointly by PPO pathwise halting gradient, exact teacher-forcing replay in `loss_fn`, and antithetic-CRN preservation in `es.gen_step`): π_commit=Σ_k p_k·softmax(logits_k); sample one env action from π_commit. Compute always runs to K_max (training cost O(T·K_max)); dynamic early-halt is **deploy-only**.

**Loss (PPO):** `logp = log(π_commit[a])` (mixture log-prob → `loss_fn` ratio); + `L_reg = β_p·KL(p_k ‖ Geometric(λ_p))`. Halting is supervised by the **reward-aligned pathwise policy gradient through the mixture**, L_reg as the default profile only. Per-turn `v_k` regressed to GAE return (`L_rec`) as a **gradient shortcut to early turns + consolidation surprise signal — NOT the halting supervisor**.
**Loss (ES):** fitness = `late_weighted_fitness − c·mean_t E[K]_t`. Marginalization keeps the rollout deterministic in θ given `roll_keys` → CRN intact.

**Calibration knobs — DECLARED HARNESS** (compute-budget regularizers, §1b-3-clean; ⚖ pre-reg numbers): K_max=8, λ_p=0.2 (E[K]=5 ≫ 1), K_min≥2; β_p **constant vs the α-curriculum**, prior **WEIGHT** runs high-warmup-then-anneal (anneal the weight, never λ_p); α=0 loop pre-training before the α-ramp. Adaptivity is **a measured deliverable, never assumed**: report Var(E[K]) across easy/hard lifetimes and lifetime-phase.

### S3-M.2 Fast content-addressed store (Increment B)
Zero-init member of the **per-lifetime carry** (mirrors `h0`), **vmapped over lifetimes, never scanned, never in θ** → memoryless-across-lifetimes by construction. Leaves `K[N,d_k]`, `V[N,d_v]`, bred usage field `u[N]`.
- **Read (the hinge — trainable under BOTH routes):** L2-normalize keys+query (cosine/modern-Hopfield β). (1) **HARD top-m SELECT** (`lax.top_k`, m≈16, SVA-faithful, ES-clean). (2) **SOFTMAX-over-the-m VALUE mix** (differentiable → PPO gradient to retrieved keys/values + W_q). **β annealed soft→hard**; PKM BatchNorm-on-query (forward-pass, ES-safe, anti-collapse).
- **Write (learned; deferred-by-one-step):** uses (h_canon_t, a_t, r_t) arriving as `last_a`/`last_r` in x_{t+1} → executes at start of t+1, **a deterministic fn of `xs` alone** (bit-exact replay, zero extra stored tensors). **delta-rule overwrite-nearest** (prevents averaging-collapse, evicts pre-inference poison, key→value correction).
- **Eviction:** delta-rule overwrite-nearest (mainline); LRU/DNC-usage as slope-competitors. **§4b observable (always reported next to null-memory ablation):** slot-utilization / usage-entropy.
- **Canonical space (claim softened):** keys on the post-inference state h_canon, not raw obs; invariance is **emergent (asymptotic within a lifetime), not by construction** — early writes pre-canonical, low-gated, overwritten. Falsifier is phase-conditioned (S3-M.6).

### S3-M.3 BPTT cut / route compatibility
- **Cross-step h: keep FULL-T BPTT** (it IS the RL² mechanism; do NOT stop-gradient it — that would regress the Phase-1 memoryless-core result).
- **Store S: truncated-BPTT WINDOWS of length W** (R2D2 burn-in). **W=1** = fast-weight semantics, store-path depth=K_max, regression-safe, ES-trainable, but **PPO gets ~zero write-policy gradient**. **W>1** buys bounded PPO write gradient. Default W=1 on cbandit-FR; W>1 on the memory-demand env. **ES is the legitimate mainline for the memory-write axis** (a valid §1b-1 outcome, not a failure).
- **Memory:** nested `jax.checkpoint` (step fn + micro-turn scan) → O(B·(T+K)·d); global-norm clip + per-route store-path gnorm logging; γ=0.999 stability escape hatch.
- **Dual value heads:** GAE critic off the **pre-micro-turn** state with `stop_gradient` on the store; per-turn v_k anchors the loop. `reward_scale` → running value-normalizer.
- **Honest route comparison:** freeze addressing softness identically across routes; ALSO run PPO with ES's hard reads, report soft-vs-hard delta as a declared confounder; **a route whose store collapsed (usage diagnostic) is DISQUALIFIED, never averaged in.**

### S3-M.4 Selective consolidation — slow bank (Increment C, deferred)
**Separable, zero-able, stop-gradiented, additively-gated** `params["slow"]` (NAMED pytree subtree, asserted disjoint from the per-lifetime carry). **Shared-weight continual fine-tune (§1d mechanism a) is FORBIDDEN for the gated claim** — it makes the null-memory control structurally unsatisfiable. Read combine: `activation += g_slow·slow_read` → **null-memory ablation = literal one-checkpoint subtraction** (zero `params["slow"]`, rerun identical eval).
- **Operator:** surprise-then-confirmed (Titans); form FIXED (connector, §4b-legal), params BRED. **Update path:** a NEW `consolidate.py` between blocks of outer-loop updates (outside `epoch_step`/`gen_step`) — EMA/test-time, stop-gradiented, AlphaGo-shaped distillation of converged fast-store contents (reuses the Phase-5 crystallization operator); EWC anchor to the frozen pre-consolidation (= null-memory) core; snapshot the encoder first (stationary target); generative replay via the Crucible.
- **Discipline:** ONLY at α=1, gated behind interface-inference convergence; **consolidation dropout** (p of lifetimes with `slow` zeroed) makes a store-independent core optimized-for. **"Undiminished" ≝ null-memory removes the experienced-instance speedup (early slope); final-quarter level must match a fresh core.**
- **Routes:** low-dim/tied operator → ES barely grows; ES ranks, SGD/distillation tunes (R1 flagged weaker). **Gate env: cbandit-FG / dedicated recurring-template env, NEVER cbandit-FR.**

### S3-M.5 Scaling axes — and *where each is load-bearing* (the central staging fix)
Each ≥3 levels, winner by **slope not intercept** (§1b-1).

| Axis | Levels | Load-bearing env | Route |
|---|---|---|---|
| Params | GRU {128,256,512}; S2 later | cbandit-FR (existing ladder) | both |
| **Micro-turn K_max** | {1,2,4,8} | **cbandit-FR** (compute-bound) | **PPO primary**; ES K-slope reported, never kills alone |
| **Store size N_slots** | {512,2048,8192} | **DEFERRED off cbandit-FR** → memory-demand env | both, frozen softness |
| **Lifetime length T** | family-dependent | memory-demand env | both |
| Consolidation-lifetimes | AlphaGo amortization curve | cbandit-FG / template env | SGD primary |

**Why the deferral:** cbandit-FR (C=5 contexts, fits the 128-d GRU at 0.6216) makes the {512..8192} sweep **flat-by-construction** and the usage-collapse diagnostic **misfire**. So store-size/lifetime slopes are graded on a dedicated memory-demand env (G3-class many-association/T-maze, N_assoc ≫ GRU-capacity, behind interface randomization); prerequisite = a positive store-vs-null delta there first. K_max is the one new axis Phase-1 can honestly test.

### S3-M.6 Null-memory control · leak falsifiers · control taxonomy
- **Null-memory (§1d-2, mandatory):** zero `params["slow"]`, rerun identical eval; pass = experienced speedup vanishes, level undiminished.
- **Leak falsifiers:** (a) trainable-pytree leaves **asserted disjoint** from per-lifetime-carry leaves at init; (b) **lifetime-order-invariance** — permute vmap order, every per-lifetime metric bitwise identical; (c) **cross-lifetime probe** — lifetime N optimum = N−1's; above-chance ⇒ leak.
- **Phase-conditioned canonical probe:** linear decoder store-keys→interface-identity at chance on post-convergence/late-lifetime writes (fast store; early exempt), unconditionally at chance for the consolidation copy. Re-run C7/C8 with consolidation ACTIVE — both must still collapse.
- **Memory-level taxonomy (extends C5/C6):** micro-turn reset · step(h) reset · episode reset (=C5, + C5-S store-reset) · lifetime reset · consolidation-null. **Fixed-K controls (K=1, K=K_max) mandatory:** if adaptive-K does not beat fixed-K at matched compute, halting is a connector → ship fixed-K (still delivers the compute-per-decision axis).

### S3-M.7 Anchors
PonderNet (2107.05407) · delta-rule fast-weight programmers (2102.11174) · product-key memory + BN-on-query (1907.05242) · NEC (Pritzel 2017) · MERLIN (1803.10760) · DNC (Graves 2016) · Titans (2501.00663) · EWC (Kirkpatrick 2017) · modern Hopfield β (Ramsauer 2020) · R2D2 burn-in (Kapturowski 2019) · in-house SVA + hourglass (port mechanism to JAX). Provenance: hardening workflow `wf_44c62735-db3` (5 web-grounded red-teams × 3 judged designs + synthesis), 2026-06-13.

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
