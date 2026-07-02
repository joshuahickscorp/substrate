# Mixture of Thinking: Thesis and Definition

Sections 1 and 2 of the Mixture of Thinking (MoT) reframing of the Brain program. This document restates
the thesis, defines the object rigorously (what it is and is not, and how it differs from every adjacent
architecture it will be confused with), states the falsifiable core claim, and proposes flagship
experiments. It inherits, and is subordinate to, the doctrine and the corpus state recorded in
`DOCTRINE_SYNTHESIS.md`. Nothing here is a run result; every empirical claim below traces to an experiment
id already on disk or to a proposed experiment with a null and a baseline it must beat.

House style (BLACKHOLE.md): no em dashes or en dashes anywhere, commas and parentheses and colons only.

---

## 0. Why reframe at all, and the one honest fact the reframing must carry

The Brain corpus is, after roughly 119 catalogued experiments run for real on an M3 Pro, a rigorously
honest NEGATIVE map with a very small number of survivors. Almost every biological-plasticity signature
tied or lost its non-biological baseline. Almost every abstraction probe was refuted by frozen-random,
matched-compute, or a tuned baseline, or ceilinged. Two positives survived adversarial review (e7_sparse,
sparse/gated heads halve catastrophic forgetting versus a parameter-matched dense baseline as a
trained-shell dynamics fact, and ex2_latent_planning, short-horizon MPC beats flat reactive and
action-shuffle on true dynamics), and one late substrate result landed positive: real V-JEPA decodes shape
under heavy nuisance at 0.379 while random-pixel untrained features sit at 0.069 (chance 0.167), a delta of
+0.31 off-ceiling (substrate_vs_random.json), the first valid evidence the frozen substrate is special
because pretraining buys nuisance-invariance.

The reframe from "frozen substrate plus tiny trainable shell" to "Mixture of Thinking" is NOT a rebranding
of the same architecture and it is NOT a license to smuggle in the mystical connotations of "thinking". It
is a change of the unit of analysis. The old framing asked: does THIS mechanism (this replay ordering, this
neuromodulator, this refiner) beat its baseline? The corpus answered, mostly, no. The MoT framing asks a
different and (the claim is) more productive question: does a system COMPOSED of several qualitatively
different processing modes, each individually weak or narrow, achieve higher capability DENSITY per unit of
frozen compute than the best single mode plus the best single baseline? This is a compositional hypothesis
about coordination, and it is only worth stating because the single-mechanism hypotheses have been so
cleanly falsified that the residual live question is whether the FAILURE was of the mechanisms or of testing
them in isolation on additive synthetic content.

The reframe carries one non-negotiable honesty constraint inherited from section 3d of the doctrine: the
standing frozen_random_projection control is VACUOUS for any probe-based metric (a full-rank invertible
1024x1024 Gaussian matrix that a linear or MLP probe absorbs, forcing delta to exactly 0.000). Every MoT
claim that reduces to "a probe decodes X" is therefore automatically suspect until it is run against a
genuinely different feature space (random-ENCODER or random-init-ViT features) or against a trained-shell
dynamics metric whose optimization trajectory depends on input conditioning. MoT does not get to reintroduce
vacuous wins by calling them "coordination".

---

## 1. The Nine Thesis Statements

Anchored throughout on capability DENSITY: reasoning per FLOP, learning per example, retention per byte,
adaptation per update, abstraction per parameter, self-correction per compute, plasticity without forgetting.
Density, not peak capability, is the axis on which a frozen-substrate-plus-tiny-shell program can plausibly
win, because it structurally cannot win on peak (it has no frontier compute and a frozen encoder), so every
thesis below is a density thesis or it is not admissible.

### 1.1 One-sentence thesis

Intelligence worth building on a laptop is a coordinated ecology of several cheap, qualitatively distinct
thinking modes over one frozen perceptual substrate, and the win, if any, is measured in capability per unit
of compute, memory, and example, not in peak capability.

### 1.2 One-paragraph thesis

Mixture of Thinking claims that a small, frozen-substrate system can achieve higher capability DENSITY than
any of its parts by hosting several representationally and dynamically distinct thinking modes (a fast
reactive readout, a slow deliberative planner, a sparse memory-protective mode, an uncertainty-gated
exploratory mode, a consolidative offline mode) and routing between them by a cheap, learned or measured
signal. The substrate (frozen V-JEPA 2 ViT-L, inference-only, requires_grad=False) provides a shared
representational geometry that the corpus has now shown carries real learned nuisance-invariance (+0.31
off-ceiling), and the trainable shell (predictor, heads, ensemble, buffer, plasticity, consolidation,
neuromod, modulation, refine) provides the modes and the router. The falsifiable content is that the routed
ecology beats (a) the best single mode at matched compute, (b) a uniform-average ensemble of the same modes,
and (c) a monolithic dense model of equal parameter count, on density metrics, and that at least one mode is
substrate-dependent under a NON-vacuous control. If the routed system merely equals its best single mode, or
if every advantage is a full-rank-projection artifact, MoT is refuted and the program reduces to picking the
one best mode.

### 1.3 Technical thesis

MoT is a router R over a set of experts E_1..E_k, where the experts are NOT (as in Mixture of Experts)
homogeneous FFN sub-blocks selected token-by-token to save FLOPs, but qualitatively different COMPUTATIONS
over the same frozen latent z: E_reactive is a shallow feedforward head (one forward pass), E_plan is a
short-horizon latent MPC over a learned forward model (ex2, iterated rollout), E_sparse is a kWTA/MoE
memory-protective head (e7), E_consolidate is an offline replay+EWC pass, E_explore is an uncertainty-gated
sampler with a noisy-TV guard. The router selects per EPISODE or per TASK (not per token) using a scalar
cost signal (predicted difficulty, uncertainty, novelty) and its selection must beat both the always-best
expert and a compute-matched deeper single expert. The technical crux is that the experts have different
computational depths and different memory footprints, so routing is a way to spend compute where the density
payoff is highest, and the whole thing is trained on CACHED latents so the frozen forward is amortized once.

### 1.4 Philosophical thesis

The philosophical wager is anti-monolithic and anti-mystical: there is no single representation or single
inference procedure that is "thought"; what we call thinking is the coordinated switching among modes that
differ in speed, abstraction level, memory access, and confidence, and the felt unity of thought is a
property of the coordinator, not of any mode. This is deliberately a functional, deflationary claim, and it
is falsifiable: if a single monolithic mode matches the routed ecology on every density metric at matched
compute, then the multiplicity was decorative and the philosophical claim is empty. MoT explicitly refuses
the move where "thinking" is invoked to explain a result that a linear probe on a frozen latent already
explains (the s10_anti_self_deception discipline applies: a mode that adds nothing beyond frozen-random
geometry is a PASS-VACUOUS mode, not a thinking mode).

### 1.5 Architecture thesis

The architecture is a strict two-tier system: an inherited, inference-only substrate (one frozen V-JEPA 2
ViT-L, called only under no_grad, providing pooled and, where available, dense token latents) and a tiny
trainable shell that contains all the modes and the router. No mode may modify substrate weights; all
learning is on cached latents. The router is the only component that sees all modes, and it must be cheaper
than the cheapest mode it can invoke (otherwise routing costs more than it saves and density falls). The
architecture is justified over a single dense head only if (a) at least one mode is substrate-dependent under
a non-vacuous control, and (b) routing beats a uniform ensemble, so the two standing architectural baselines
are the best-single-head and the equal-weight ensemble, both parameter-matched and compute-matched.

### 1.6 Experimental thesis

Every MoT claim decomposes into three pre-registered comparisons that must all pass: BEAT THE BEST SINGLE
MODE at matched compute (else the mixture is redundant), BEAT THE UNIFORM ENSEMBLE (else the ROUTING adds
nothing and only the averaging did), and SHOW SUBSTRATE-DEPENDENCE UNDER A NON-VACUOUS CONTROL (real
V-JEPA versus random-init-ViT or random-encoder features, never versus a full-rank latent projection). A
density metric (reasoning per FLOP, retention per byte, adaptation per update) is the scored quantity, not
raw accuracy, because raw accuracy ceilings on synthetic content and because peak is not the program's axis.
Seed-stability is mandatory (sign-flips published as instability, swept via devsys.harness.sweep.run_sweep
because the modules read cfg.seed), and the test bed must clear difficulty_calibration (D3) before any tie is
trusted, since the binding constraint across the whole corpus has been test difficulty, not the mechanisms.

### 1.7 Why this matters without frontier compute

Frontier labs buy capability with scale: more parameters, more data, more FLOPs, and their peak numbers are
unreachable on an M3 Pro. The only axis on which a laptop program can produce a defensible, non-derivative
result is DENSITY, capability per unit of compute or example or byte, because density is a ratio and a small
system can win a ratio it can never win as an absolute. MoT matters precisely because it is a density
architecture: it amortizes the expensive frozen forward once into a cache, it spends deliberative compute
(planning, iteration) only on the episodes the router judges worth it, and it protects retention with cheap
sparse structure rather than by growing parameters. If MoT works, the transferable finding is not a new
capability but a new EFFICIENCY: how to get more reasoning, retention, and adaptation out of a fixed frozen
substrate than a monolith of equal cost achieves, which is exactly the finding a compute-poor lab is
positioned to discover and a compute-rich lab has no incentive to look for.

### 1.8 Why frozen substrates may not be enough

A frozen substrate has a fixed representational geometry, and the corpus contains the honest limits of that
geometry: pooling is projection-invariant almost everywhere for single factors (dense_vs_pooled.json), the
only surviving substrate signal is a whisper in sparse-head forgetting dynamics, and every attempt to show
developmental moldability (critical windows, path-dependence, U-shaped overgeneralization) came back flat.
The +0.31 nuisance result shows the substrate is special, but special is not the same as sufficient: a frozen
geometry cannot be RESHAPED by experience, so if the two doctrinal questions (moldability, and abstraction
not routed through language) require the representation itself to change with learning, a frozen substrate is
by construction unable to demonstrate it, and MoT would only be moving deck chairs in the shell. The
concrete risk MoT must confront: if compositional_under_nuisance.py returns NON-COMPOSITIONAL / MEMORIZED
(held-out shape collapses to chance because the pooled substrate binds shape and color into unfactorable
conjunctions), then no amount of shell coordination recovers a compositional code the substrate never
carried, and the frozen interface is the bound, not the modes.

### 1.9 When we should build our own

The decision to move off the frozen dense V-JEPA 2 toward a custom or dense-2.1 architecture is NOT
justified today and MoT does not pre-empt it. The pre-registered trigger to build our own is a CONJUNCTION:
(a) the gold-standard substrate control (substrate_vs_random_init_vit.py) confirms the +0.31 is pretraining,
not resolution, so the frozen encoder is worth keeping in principle, AND (b) compositional_under_nuisance.py
returns NON-COMPOSITIONAL / MEMORIZED on the pooled interface specifically (the substrate carries invariance
but cannot factor bound attributes through the pool), AND (c) a coarse-dense-token variant (2x2 or 4x4 grid
instead of full mean-pool, the P7 object-binding-before-pooling probe) recovers the compositional signal the
full pool destroys. That conjunction, and only that conjunction, says the bound is the POOLING INTERFACE and
a denser substrate would help, which is the single defensible reason to build our own. Absent it, building
our own is spending frontier-shaped compute to answer a question the frozen substrate has not yet been shown
to fail, which the doctrine forbids.

---

## 2. Mixture of Thinking, Defined Rigorously

### 2.1 What MoT IS

MoT is a two-tier system in which one inherited frozen perceptual substrate feeds a tiny trainable shell, and
the shell hosts (i) a SET of qualitatively distinct thinking modes that differ along at least one of five
axes, representational geometry (which subspace or dense-token structure the mode reads), learning speed
(one-shot binding versus slow SGD versus offline consolidation), memory system (none, episodic buffer,
consolidated store), uncertainty signal (calibrated confidence, novelty, prediction error), and reasoning
mode (single-pass reactive, iterated deliberative, planned rollout), and (ii) a cheap coordinator (router)
that selects or blends modes per episode or per task using a scalar cost signal, trained on cached latents.
A mode qualifies for the mixture only if it is DISTINCT on at least one axis from every other mode AND passes
a non-vacuous substrate or matched-compute control on its own; a coordinator qualifies only if it beats both
the best single mode and the uniform ensemble at matched compute.

### 2.2 What MoT IS NOT

- It is NOT a peak-capability play. It cannot and does not claim to match a monolithic frontier model on any
  absolute benchmark; the only admissible scoreboard is density.
- It is NOT a new representation learner. The substrate stays frozen; MoT learns coordination and readouts,
  not features.
- It is NOT a claim that any mode "thinks" in a phenomenal sense. Modes are computations with different
  cost/benefit profiles; "thinking" names the coordination, deflationarily.
- It is NOT vindicated by any probe result. A mode whose only evidence is a probe that decodes better under
  the vacuous frozen-random control is PASS-VACUOUS and does not count.
- It is NOT an excuse to revive refuted mechanisms unchanged. e4 neuromod (30/30 runs amplify error on
  noise), b4 homeostatic scaling (harms retention), and staged plasticity (e3, monotone decay) do not
  re-enter as modes unless a NEW control shows they help; their prior negatives stand.

### 2.3 How MoT differs from adjacent architectures

- Mixture of Experts (MoE): MoE experts are HOMOGENEOUS FFN sub-blocks selected per token to save FLOPs
  inside a single trained network; the representation is learned end to end and every expert is the same kind
  of computation. MoT experts are HETEROGENEOUS computations (reactive head, planner, sparse memory head,
  offline consolidator) over a FROZEN shared representation, selected per episode/task, and the point is
  density across qualitatively different modes, not load balancing across identical ones. e7_sparse's kWTA/MoE
  head is ONE mode inside MoT, not the whole thing.
- Ensembles: an ensemble averages or votes homogeneous predictors to reduce variance; there is no router and
  no compute saving (every member runs every time). MoT ROUTES to spend compute selectively and its members
  are heterogeneous, so the uniform ensemble is precisely the baseline MoT must beat to prove routing matters.
- Multimodal fusion: fusion combines DIFFERENT INPUT MODALITIES (vision, audio, text) into one
  representation; MoT combines different COMPUTATIONS over the SAME (visual latent) input. Fusion changes what
  is perceived; MoT changes how the same percept is processed.
- Tool use: tool use calls EXTERNAL deterministic functions (a calculator, a search API) with the model as
  orchestrator; MoT modes are all INTERNAL learned/measured computations over cached latents, with no
  external oracle, and the router is not an LLM planner but a cheap scalar-driven selector.
- Ordinary modular nets: a modular net has fixed hand-wired sub-networks with a fixed dataflow; MoT's
  distinguishing commitments are (a) modes must be distinct on a named axis, (b) routing is learned/measured
  and per-episode, and (c) the whole system is scored on density against best-single-mode and uniform-ensemble
  baselines under non-vacuous controls. A modular net that fails those is just a modular net.

### 2.4 Relations to cognitive and biological frameworks (borrowed as hypotheses, not authority)

- Global Workspace Theory (Baars, Dehaene): the router-plus-shared-substrate resembles a workspace that
  broadcasts a selected mode's output; MoT operationalizes the "ignition/broadcast" as episode-level routing
  and makes it falsifiable (broadcast must beat no-broadcast at matched compute). It borrows the STRUCTURE,
  not the consciousness claim.
- Cognitive architectures (ACT-R, SOAR): those posit distinct memory stores and a production/selection cycle;
  MoT's modes (reactive, episodic-buffer, consolidated) and its router echo declarative/procedural splits and
  conflict resolution, but MoT refuses hand-authored production rules and requires the selection to be learned
  or measured and to beat a baseline.
- Developmental learning (Vygotsky scaffolding, Hensch critical periods): the corpus has REFUTED the direct
  developmental signatures at toy scale (d4 U-shape flat, d6 no substrate-specific window, curriculum no clean
  win); MoT does NOT assume developmental moldability, it treats it as an open question a mode could
  eventually earn, and it inherits the negative results as priors, not as things to re-assert.
- World models (Ha/Schmidhuber, JEPA): the E_plan mode IS a world-model rollout (ex2's latent MPC over a
  learned forward model), and ex2 is a surviving positive, so MoT has one empirically grounded world-model
  mode; note the substrate is NOT a JEPA in the training sense, it is an inference-only inherited encoder.
- Memory systems (complementary learning systems, McClelland): the E_sparse (fast, interference-resistant)
  and E_consolidate (slow, offline replay+EWC) modes instantiate the fast/slow complementary split; the
  corpus supports the STRUCTURAL claim (e7_sparse halves forgetting) but not the biological-schedule claims
  (n1/n4 replay ordering refuted), so MoT keeps the structure, drops the schedule.
- Plasticity / neuromodulation (ACh/NE gating, homeostasis): REFUTED as implemented (e4, n6, b4), so
  neuromodulatory gating is admitted into MoT only as an uncertainty SIGNAL feeding the router (a measured
  scalar), never as a learning-rate gate on the shell, unless a new non-vacuous control resurrects it.

### 2.5 The core claim, stated so it can be falsified

CORE CLAIM. Intelligence, at the density that matters for a compute-poor program, is a COORDINATED ECOLOGY
of representational geometries, learning speeds, memory systems, uncertainty signals, and reasoning modes,
and a learned/measured coordinator over such an ecology achieves strictly higher capability density than the
best single mode, than the uniform ensemble of the same modes, and than a monolithic model of equal
parameters and compute, with at least one mode's advantage surviving a non-vacuous substrate control.

This decomposes into five falsifiable hypotheses, each with a null and a baseline:

- H1 (routing beats best-single-mode). Null: the routed system's density equals or trails its best
  constituent mode at matched compute. Baseline to beat: best single mode, compute-matched. Refuted if delta
  <= 0 or inside the seed spread.
- H2 (routing beats uniform ensemble). Null: an equal-weight blend of the same modes matches the router.
  Baseline: uniform ensemble, same members, same total compute. Refuted if the router adds nothing over
  averaging.
- H3 (at least one mode is substrate-dependent, non-vacuously). Null: every mode's advantage is reproduced
  by random-init-ViT or random-encoder features (or is a full-rank-projection artifact). Baseline:
  random-init-ViT features at matched resolution (substrate_vs_random_init_vit.py logic). Refuted if all
  modes tie the non-vacuous control.
- H4 (heterogeneity is load-bearing). Null: replacing the heterogeneous modes with k copies of the best
  single mode (a homogeneous MoE) matches the heterogeneous mixture. Baseline: homogeneous k-copy MoE, same
  parameter and FLOP budget. Refuted if homogeneous matches heterogeneous.
- H5 (density, not peak, is where the win lives). Null: any routed advantage disappears when scored on raw
  accuracy at UNMATCHED compute (i.e. the win was just spending more FLOPs). Baseline: monolithic dense model
  given the router's total compute budget in one pass. Refuted if the monolith at equal total compute matches
  the mixture, meaning the mixture only helped by unrolling depth (the p9/ex17 gain=0.0 failure mode).

All five must clear difficulty_calibration (D3) on the test bed first, and all five must be run on real
cached latents once the non-additive bound-attribute video cache exists, because every synthetic-content
version will ceiling exactly as the corpus already documents.

### 2.6 What would count as REFUTING MoT wholesale

If, on a difficulty-calibrated, non-ceiling test bed, the best single mode matches the routed ecology (H1
null holds) AND the uniform ensemble matches the router (H2 null holds) AND every mode ties the non-vacuous
substrate control (H3 null holds), then MoT is refuted and the correct conclusion is monolithic: pick the one
best mode and ship it. This is a real, reachable outcome given the corpus's negative prior, and stating it
plainly is the price of the reframe.

---

## 3. Flagship experiments (registry candidates)

Six candidates across the AT (abstraction/thinking), MT (mixture-of-thinking core), and AL
(active-learning/adaptation) families, chosen so each targets exactly one of H1..H5 and each carries a null,
a baseline it must beat, and a non-vacuous control. All are cached-latent-first; the two that need real bound
video are marked studio-tier and blocked on the deferred natural-video cache. See the experiments field of
the structured output for the machine-readable registry entries.

- MT1_routed_vs_best_mode (H1): route between reactive, planner (ex2), sparse (e7) modes per episode by a
  cheap difficulty scalar; beat the best single mode at matched compute. cpu-now on the existing synthetic
  harness for a pilot; the real answer waits on the bound-video cache.
- MT2_routing_vs_ensemble (H2): same modes, router versus equal-weight blend, matched total compute; isolate
  whether ROUTING (selective compute) beats AVERAGING. cpu-now.
- MT3_heterogeneous_vs_homogeneous (H4): heterogeneous modes versus k copies of the best mode (a homogeneous
  MoE) at equal params/FLOPs; isolate whether qualitative diversity is load-bearing versus generic mixture
  capacity. cpu-now.
- AT1_mode_substrate_dependence (H3): for each mode, real V-JEPA versus random-init-ViT features at matched
  256px (reusing substrate_vs_random_init_vit.py logic), on nuisance-heavy content; find which modes' wins
  survive a non-vacuous control. studio-tier (needs the real-encoder pass, must not run concurrently with the
  in-flight ViT job).
- AT2_compositional_routing (H5-adjacent): does a router that dispatches held-out compositional cases to the
  planner/deliberative mode beat a monolith at equal total compute on held-out (shape,color) extrapolation
  under nuisance; directly extends compositional_under_nuisance.py from a single probe to a routed system.
  studio-tier (blocked on the non-additive bound-video cache; synthetic version will ceiling).
- AL1_uncertainty_router_noisytv (H1 with the curiosity guard): route exploration by an uncertainty signal
  that must pass the noisy-TV guard (ignore irreducible aleatoric noise), beating a random-episode-selection
  baseline on adaptation-per-update; resurrects the neuromod SIGNAL as a router input only (never as a shell
  learning-rate gate, since e4 refuted that), and is refuted if the uncertainty router matches random
  selection or chases the noisy TV. cpu-now for the guard pilot.
