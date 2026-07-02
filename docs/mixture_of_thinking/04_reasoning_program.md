# Section 4: The Expanded Reasoning Research Program (DR + MT lines)

## 4.0 What this section is, and the one prior that governs it

This section designs the full reasoning-in-latent-space research line: recurrent latent
refinement, adaptive latent compute, latent scratchpads, latent chain-of-thought without text,
latent self-verification, inter-module debate, tree/beam and Monte-Carlo search over latent
states, cross-substrate reasoning consistency, visual-vs-language-vs-relational reasoning,
memory-first reasoning, counterfactual/causal latent intervention, planning horizon, internal
simulation, disagreement-as-uncertainty, confidence stopping, and reasoning under
compression/quantization/corruption/missing-channel. Each item is one experiment (a DR "deep
reasoning" id or an MT "mixture of thinking" id) with a null, a baseline it must beat, metrics,
controls, diagnostic gates, failure interpretations, required substrate, tier, difficulty,
scientific value, and the three build flags (cached-latent-ok, needs-multiple-models,
needs-custom-model).

The governing prior is the corpus's single strongest reasoning-line result, and it is a
NEGATIVE: p9_thought_without_language and ex17_latent_reasoning both measure matched-compute
gain from iteration at exactly 0.0, n9_attractor_convergence shows no fixed point (residual two
orders of magnitude past threshold), y1 shows unrolling past the training horizon actively
worsens loss, and ex18's trained verifier barely beats a shuffled untrained one. The refiner in
`src/devsys/shell/refine.py` is, by its own docstring, "exactly a residual network unrolled in
place", and the compute-matched control in `src/devsys/diagnostics/compute.py`
(`depth_for_matched_flops` returns exactly `refiner_steps`) has so far erased every gain. So the
default expectation for every experiment below is that iteration ties depth. The design job is
NOT to find a clever way to make iteration look good; it is to build tests sharp enough that if
iteration ever genuinely beats matched-compute depth we would believe it, and honest enough that
when it does not, the null is informative about WHY.

There is a second, subtler prior from the vacuous-control discovery (Section 3d of the doctrine):
the standing `frozen_random_projection` control (a full-rank 1024x1024 Gaussian) is VACUOUS for
any probe-based metric because a linear/MLP readout absorbs the invertible map (measured delta
0.000). Reasoning experiments are mostly TRAINED-SHELL-DYNAMICS metrics (SGD forgetting-like,
convergence trajectories, planning rollouts on true dynamics), so `frozen_random_projection` is
NOT automatically vacuous for them, BUT any reasoning experiment whose final readout is a linear
probe on a refined latent inherits the vacuity. The correct substrate control for those is real
V-JEPA vs random-ENCODER / random-init-ViT features (the `substrate_vs_random_features.py` /
`substrate_vs_random_init_vit.py` family), not the square Gaussian. Every experiment below states
which substrate control it uses and why.

A note on compute honesty that recurs everywhere. `refiner_flops(dim, hidden, steps)` and
`matched_within(a, b, tol=0.10)` in `diagnostics/compute.py` are the arbiter. An adaptive-compute
or search method that "wins" must win at EQUAL total FLOPs summed over all its steps/branches/
rollouts, not at equal per-forward FLOPs. The most common way these experiments will fool
themselves is by counting only the winning branch's compute and not the branches it discarded.
Every search/debate/rollout experiment below budgets total FLOPs including discarded work.

---

## 4.1 The core distinction this program tests

There is exactly one thing that would make a latent reasoning line non-trivial: a computation
whose benefit CANNOT be bought by a single-pass feedforward network of equal FLOPs. Three
mechanisms could in principle produce that, and the program is organized around distinguishing
them:

1. **Input-dependent compute allocation** (adaptive halting, confidence stopping): a variable-
   depth network can, in principle, spend more compute on harder inputs. A fixed-depth matched
   network spends the SAME compute on every input. If hardness is real and heterogeneous, adaptive
   allocation can beat fixed depth at equal AVERAGE FLOPs even though it ties at equal MAX FLOPs.
   This is the one honest place iteration can win, and n10_halting_difficulty already gestures at
   it. MT1 and MT2 below make it the central test.

2. **Search / non-monotone exploration** (tree, beam, MC rollout, debate): a feedforward net
   computes one trajectory; search evaluates many and keeps the best under a scorer. This can beat
   a single forward pass IFF (a) there is a cheap, reliable scorer and (b) the best branch is not
   findable by the forward net directly. Both are strong preconditions and both are separately
   falsifiable. MT3, MT4, DR6, DR7 test them.

3. **Persistent external state** (latent scratchpad, memory-first): writing intermediate results
   to a slot-based memory the reasoner re-reads is strictly more expressive than a fixed-width
   residual stream only if the task needs more working memory than the residual width carries.
   DR3 and DR9 test this and are the experiments most likely to reveal the frozen substrate's
   BOUND (see 4.5).

Everything else in the program is a stress test (corruption, quantization, missing channel,
cross-substrate) or a specificity test (visual vs language vs relational) on these three.

---

## 4.2 The MT line: mixture-of-thinking primitives (adaptive compute, search, debate)

The MT ("mixture of thinking") ids test whether ALLOCATING different amounts or kinds of
computation per input beats spending a fixed budget. The name is literal: a mixture over thinking
strategies, chosen per input, must beat the single best fixed strategy at matched total compute.

### MT1 — Adaptive latent compute beats fixed depth at matched AVERAGE FLOPs
- **Core question.** Does per-sample adaptive halting (the `halt=True` path in `IterativeRefiner`)
  spend less total compute than fixed-N refinement for equal accuracy, i.e. does it allocate
  depth to hard inputs? This is the ONLY framing under which iteration can beat depth, so it is
  the load-bearing MT experiment.
- **Mechanism.** `IterativeRefiner(halt=True, halt_threshold)`; the halt head accumulates a
  per-step stopping probability, freezing samples that cross threshold (refine.py lines 74-87).
  Train with a small ponder-cost so the halt head is pressured to stop early.
- **Baseline it must beat.** A fixed-depth single-pass MLP whose block count equals the ADAPTIVE
  arm's MEAN steps used (not max steps), via `depth_for_matched_flops`. This is the honest match:
  adaptive can only claim a win if it beats a fixed net of its own average cost.
- **Null hypothesis.** At equal average FLOPs, adaptive halting ties fixed-depth: either the task
  has no exploitable hardness heterogeneity, or the halt head learns a constant (halts everyone at
  the same step), which is just fixed depth with a learned N.
- **Metrics.** accuracy at matched mean-FLOPs; halting-step distribution ENTROPY (a constant halt
  head has entropy ~0, the null); correlation of steps-used with an independent difficulty label
  (from `diagnostics/difficulty_calibration.py`); accuracy-vs-compute Pareto frontier area.
- **Controls.** compute-matched fixed MLP at the adaptive mean; shuffled-difficulty control (halt
  head cannot correlate with a permuted difficulty label above chance); the ponder-cost=0 arm (if
  removing the cost kills the effect, the "adaptivity" was a regularization artifact).
- **Diagnostic gates.** `compute_accounting` (mean-FLOPs match within tol 0.10);
  `difficulty_calibration` (the regime must carry real, gradable hardness or halting cannot
  correlate with anything); `determinism`.
- **Failure interpretations.** Constant halt head => the mixture collapsed to a single strategy,
  reasoning-as-allocation is dead on this substrate. Halting correlates with difficulty but does
  NOT beat matched fixed depth => the net can already spend its fixed budget optimally, adaptivity
  buys nothing. Wins only with ponder-cost=0 => the effect is L2-like regularization, not compute
  allocation.
- **Required substrate.** cached latents; needs a difficulty-graded regime.
- **Tier.** cpu-now. **Difficulty.** medium. **Sci value.** high (it is the one honest win path).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### MT2 — Confidence stopping vs a fixed early-exit schedule (is the confidence signal real)
- **Core question.** When the refiner stops "because it is confident", is the confidence a real
  correctness signal, or a proxy for input norm / step count that a fixed schedule captures for
  free?
- **Mechanism.** Compare the learned halt head against a NON-learned early-exit rule that stops
  when the step-to-step update norm `||z_{t+1}-z_t||` (already computed in `refine.unroll`) drops
  below a threshold. Both are "confidence stopping"; only one is trained.
- **Baseline it must beat.** The update-norm early-exit rule (a free, untrained stopping signal),
  at matched mean-FLOPs.
- **Null hypothesis.** The trained halt head ties the free update-norm rule: confidence is just
  "the latent stopped moving", which needs no learned head.
- **Metrics.** accuracy at matched mean-FLOPs; AUROC of the stopping signal against per-sample
  correctness; calibration (ECE) of halt probability vs empirical correctness.
- **Controls.** shuffled-halt-head (permute halt scores across the batch, must destroy any gain);
  update-norm rule; fixed-N at matched mean.
- **Diagnostic gates.** `compute_accounting`; `determinism`.
- **Failure interpretations.** Trained head ties update-norm rule => no learned confidence, stop
  claiming a confidence mechanism. Neither beats fixed-N => stopping signal carries no per-sample
  correctness information at all (this is the ex18 pattern recurring).
- **Required substrate.** cached latents. **Tier.** cpu-now. **Difficulty.** medium. **Sci value.**
  high.
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### MT3 — Beam/tree search over latent states beats greedy single-trajectory refinement
- **Core question.** Does maintaining K candidate latent states, expanding and pruning under a
  scorer, beat greedily refining one state, at matched TOTAL FLOPs (K branches counted)?
- **Mechanism.** From a latent z, generate K perturbed refinement trajectories (stochastic block
  via dropout at inference, or K learned expansion heads), score each with the `Verifier`
  (refine.py lines 108-122), keep top-b, iterate. Total FLOPs = sum over all expanded branches.
- **Baseline it must beat.** Greedy single-trajectory refinement run for enough extra steps to
  match the search arm's total FLOPs (a deeper single chain of thought).
- **Null hypothesis.** At matched total FLOPs, search ties a deeper greedy chain: exploring
  breadth buys nothing a bit more depth does not, because the scorer cannot rank branches better
  than the refiner already ranks its own single next step.
- **Metrics.** accuracy at matched total FLOPs; oracle-beam gap (accuracy if we could pick the
  truly-best branch vs the scorer's pick) which upper-bounds the achievable gain; branch diversity
  (mean pairwise latent distance, a collapsed search has ~0).
- **Controls.** greedy-deeper at matched total FLOPs; RANDOM scorer (pick a branch at random, must
  not tie the learned scorer or the scorer is doing nothing); shuffled-verifier.
- **Diagnostic gates.** `compute_accounting` (total-FLOPs match INCLUDING pruned branches);
  `difficulty_calibration`; `determinism`.
- **Failure interpretations.** Search ties greedy-deeper => breadth is worthless here, the tie
  ex17 predicts. Random scorer ties learned scorer => the verifier carries no ranking signal (the
  ex18/y9 pattern). Oracle-beam also ties greedy => even a perfect scorer cannot help, the answer
  is reachable by depth alone, search is fundamentally the wrong prior for this substrate.
- **Required substrate.** cached latents; a task with multiple plausible latent continuations
  (ambiguous / multi-modal targets), else there is nothing to branch over.
- **Tier.** cpu-now (small K). **Difficulty.** high. **Sci value.** high.
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### MT4 — Latent debate between modules reduces error vs a single module at matched compute
- **Core question.** Do two (or more) independently-trained shell modules that iteratively
  critique/adjust each other's latent proposal reach a better answer than one module given the same
  total compute?
- **Mechanism.** Modules A and B each propose a refined latent; a small referee head reads both,
  and each module revises conditioned on the other's proposal, for R rounds. This is debate as
  latent message-passing (not text). Total FLOPs = both modules x rounds.
- **Baseline it must beat.** A single module of equal total FLOPs (i.e. deeper / wider so its
  compute equals both debaters summed over rounds).
- **Null hypothesis.** Debate ties the single matched module: two heads exchanging latents is just
  an ensemble/MoE unrolled, and a matched single net absorbs it.
- **Metrics.** accuracy at matched total FLOPs; disagreement trajectory (does A-B latent distance
  shrink across rounds, i.e. do they actually converge, or just co-drift); ensemble-baseline gap
  (debate must beat a plain average of A and B's independent answers, else debate is worse than
  a free ensemble).
- **Controls.** single matched module; plain ensemble (average of independent A, B) at matched
  compute; SELF-debate control (A debating a frozen copy of itself, isolating whether diversity of
  the second module matters).
- **Diagnostic gates.** `compute_accounting`; `determinism`; seed-stability (debate outcomes are
  notoriously seed-fragile, publish sign-flips).
- **Failure interpretations.** Debate ties single-module => no benefit from structured exchange.
  Debate ties plain ensemble => the "exchange" adds nothing over independent voting, debate is an
  overcomplicated ensemble. Self-debate matches two-module debate => diversity was irrelevant, it
  is just extra depth.
- **Required substrate.** cached latents; benefits from multiple independently-seeded shells.
- **Tier.** cpu-now. **Difficulty.** high. **Sci value.** medium (high risk of being a dressed-up
  ensemble).
- cached-latent-ok: yes. needs-multiple-models: no (multiple SHELLS, one encoder).
  needs-custom-model: no.

### MT5 — Mixture-of-thinking router: choose a strategy per input beats the single best strategy
- **Core question.** Given a bank of thinking strategies (fixed-N refine, adaptive halt, beam,
  memory-lookup), does a learned per-input router beat always using the single best strategy, at
  matched expected compute?
- **Mechanism.** A tiny router reads the raw latent, picks one strategy (or a soft mixture), pays
  that strategy's compute. This is the literal "mixture of thinking" thesis.
- **Baseline it must beat.** The single best fixed strategy (whichever of the bank wins overall),
  evaluated at the router's expected FLOPs.
- **Null hypothesis.** The router collapses to always picking one strategy (the best fixed one), so
  the mixture buys nothing; OR at matched expected compute the routed mixture ties the best fixed
  strategy.
- **Metrics.** accuracy at matched expected FLOPs; router entropy (collapse detector); per-strategy
  usage vs per-strategy per-input oracle-optimal usage (routing regret).
- **Controls.** each fixed strategy alone; RANDOM router (must not tie the learned router);
  oracle router (upper bound on achievable gain).
- **Diagnostic gates.** `compute_accounting` (expected-FLOPs match); `difficulty_calibration`;
  `determinism`.
- **Failure interpretations.** Router collapses => no input structure distinguishes when each
  strategy helps, the mixture premise is false on this substrate. Learned ties random router =>
  routing signal is absent. Oracle router also ties best-fixed => strategies are redundant, there
  is only one kind of thinking here.
- **Required substrate.** cached latents; depends on MT1-MT3 having produced distinct strategies.
- **Tier.** cpu-now. **Difficulty.** high. **Sci value.** high (it is the synthesis test).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

---

## 4.3 The DR line: deep-reasoning substrate tests (scratchpad, CoT, simulation, causal, memory)

The DR ids test whether the frozen-latent substrate can support reasoning STRUCTURES that a
plain forward pass cannot: persistent scratch state, multi-step chains without text, internal
simulation of dynamics, causal intervention, and memory-first retrieval. These are where the
substrate's ceiling is most likely to show (4.5).

### DR1 — Recurrent latent refinement: contraction to a fixed point vs drift (the N9/Y1 recheck)
- **Core question.** Does the weight-tied recurrent refiner converge to an input-dependent fixed
  point (an attractor, evidence of a genuine iterative computation) or does it drift (unrolled
  depth)? This re-runs the corpus's cleanest reasoning null on the CORRECTED substrate control.
- **Mechanism.** `IterativeRefiner.unroll(z, steps=K)` for K >> trained N, logging the per-step
  update-norm sequence (refine.py lines 89-100). Geometric decay = attractor; flat/growing = drift.
- **Baseline it must beat.** Its own trained horizon: does unrolling PAST N keep improving loss (a
  real fixed point that generalizes) or worsen it (y1's observed failure)?
- **Null hypothesis.** Update norms do not decay geometrically; unrolling past N worsens loss;
  there is no fixed point (the n9/y1 result, expected to recur).
- **Metrics.** update-norm decay ratio per step; loss vs unroll-step curve past the training
  horizon; contraction fraction (fraction of steps with ||u_{t+1}|| < ||u_t||).
- **Controls.** matched-compute single-pass MLP; real-V-JEPA vs random-init-ViT features (the
  contraction property must be substrate-specific, not a property of any high-dim geometry).
- **Diagnostic gates.** `convergence` diagnostics; `determinism`; substrate control uses
  random-INIT-ViT, NOT frozen_random_projection (this metric reads a trained-dynamics trajectory,
  but the safe control is the real feature-space comparison).
- **Failure interpretations.** No decay => confirms unrolled-depth reading, close the "iteration is
  special" hypothesis harder. Decay present on real but ALSO on random-init-ViT features =>
  contraction is a geometry artifact, not a substrate property.
- **Required substrate.** cached latents; ideally both real-V-JEPA and random-init-ViT caches.
- **Tier.** cpu-now. **Difficulty.** medium. **Sci value.** high.
- cached-latent-ok: yes. needs-multiple-models: yes (real vs random-init ViT). needs-custom-model:
  no.

### DR2 — Latent chain-of-thought without text: does an intermediate latent trace help
- **Core question.** If the reasoner emits a SEQUENCE of intermediate latents (a chain of thought
  in latent space, no discretization to tokens) and is trained to make each step predictive of the
  next, does the final answer improve over a net that maps input to answer in one shot at matched
  compute?
- **Mechanism.** An autoregressive latent predictor (`Predictor`, action_dim=0) rolls out T
  intermediate latents; the head reads the final one. Supervise with a next-latent consistency
  loss so the chain is not free-running noise. Text is never in the loop.
- **Baseline it must beat.** A single-pass MLP of equal total FLOPs (T rollout steps folded into
  depth).
- **Null hypothesis.** The latent chain ties one-shot at matched compute: writing intermediate
  latents is just deeper computation, there is no benefit to the sequential factorization (the
  ex17 result generalized to sequences).
- **Metrics.** accuracy at matched total FLOPs; intermediate-latent decodability (do the middle
  latents decode task-relevant sub-quantities a one-shot net's hidden layer does not); gain vs
  chain length.
- **Controls.** matched-compute one-shot MLP; SHUFFLED chain (permute the intermediate steps'
  order, must hurt if the sequence is meaningful); frozen-random substrate ONLY if the final
  readout is a trained-dynamics metric, else random-init-ViT.
- **Diagnostic gates.** `compute_accounting`; `difficulty_calibration` (needs a multi-step task
  whose sub-steps are separately checkable, e.g. a relational chain); `determinism`.
- **Failure interpretations.** Chain ties one-shot => latent CoT is unrolled depth, matches the
  language-CoT-is-just-compute skeptic view. Shuffling the chain does NOT hurt => the steps have no
  sequential structure, the "chain" is a bag of extra layers.
- **Required substrate.** cached latents; a genuinely multi-step / compositional relational task
  (the ceiling problem bites hard here, see 4.5).
- **Tier.** cpu-now (synthetic) / studio (real bound-attribute video). **Difficulty.** high.
  **Sci value.** high.
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR3 — Latent scratchpad: does slot-based external memory beat a fixed residual stream
- **Core question.** Does giving the reasoner a small addressable memory (write/read slots it
  updates across steps) beat a fixed-width residual refiner at matched compute on a task that needs
  more working memory than the residual width carries? This is the experiment most likely to expose
  the frozen substrate's capacity bound (4.5).
- **Mechanism.** A tiny slot memory (K slots, attention read/write) that the refiner reads and
  writes each step; the residual stream stays fixed-width but can offload to slots. Compare to a
  residual-only refiner at matched FLOPs (the slot ops count).
- **Baseline it must beat.** A residual-only `IterativeRefiner` of equal total FLOPs (wider/deeper
  to absorb the slot-attention cost).
- **Null hypothesis.** The scratchpad ties residual-only at matched compute: the task fits in the
  residual width, so external memory is redundant; OR the pooled V-JEPA latent has already
  discarded the fine detail the scratchpad would need to store, so there is nothing to write.
- **Metrics.** accuracy vs required-memory (sweep task working-memory load: how many items must be
  held); the crossover memory-load at which scratchpad beats residual-only; slot-usage entropy
  (unused slots => scratchpad is decorative).
- **Controls.** residual-only matched; scratchpad with slots ZEROED after write (must collapse to
  residual-only performance, proving the slots carry the signal); shuffled-slot-address.
- **Diagnostic gates.** `compute_accounting`; `difficulty_calibration` (the memory-load axis must
  be a real, gradable difficulty); `determinism`.
- **Failure interpretations.** Scratchpad ties residual-only at ALL memory loads => the pooled
  frozen latent cannot supply enough recoverable detail to fill slots (a SUBSTRATE bound, the
  positive-negative in 4.5). Scratchpad wins only at loads no realistic task hits => a curiosity,
  not a capability.
- **Required substrate.** cached latents; ideally DENSE (per-token) V-JEPA latents, not just pooled
  (pooling is the suspected bottleneck).
- **Tier.** studio (needs dense latents). **Difficulty.** high. **Sci value.** high (most
  diagnostic of substrate insufficiency).
- cached-latent-ok: yes (but dense caches, larger). needs-multiple-models: no. needs-custom-model:
  no.

### DR4 — Internal simulation: rollout in latent space beats a reactive head (the ex2 extension)
- **Core question.** Does simulating future latents with the forward `Predictor` and acting on the
  simulated future beat a flat reactive head, at matched compute and against an action-shuffle
  control? This generalizes ex2_latent_planning (a surviving positive) to longer horizons.
- **Mechanism.** Action-conditioned `Predictor` (action_dim>0) rolls the latent forward under
  candidate action sequences; a value/goal head scores the imagined endpoint; pick the best plan.
- **Baseline it must beat.** A flat reactive head (input latent -> action) at matched FLOPs, AND an
  action-shuffle control (planning on shuffled dynamics must not win).
- **Null hypothesis.** Planning ties reactive at matched compute, or ties action-shuffle (the
  forward model's rollouts carry no usable future information beyond one step).
- **Metrics.** task return vs planning horizon; gain over reactive at matched compute; gain over
  action-shuffle; rollout-fidelity (predicted vs true latent divergence over horizon).
- **Controls.** flat reactive matched; action-shuffle; horizon sweep (find where prediction error
  compounds enough to erase the planning benefit, the honest horizon limit).
- **Diagnostic gates.** `compute_accounting`; `rollout_gate` (d6 diagnostic); `determinism`;
  seed-stability (ex2 held on 3 seeds, extend the sweep).
- **Failure interpretations.** Planning ties reactive => no benefit from simulation, contradicts
  ex2 at longer horizon (informative: ex2's win was short-horizon-only). Rollout fidelity collapses
  fast => the frozen latent's forward-predictability horizon is short, a substrate limit on
  internal simulation.
- **Required substrate.** cached latents with actions / temporal structure.
- **Tier.** cpu-now (short horizon) / env-later (real control task). **Difficulty.** high.
  **Sci value.** high (extends a real positive).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR5 — Counterfactual / causal latent intervention: do-operations in latent space
- **Core question.** Can the shell answer counterfactual queries ("what would the next latent be
  if factor X were different") by intervening on a latent direction, and does the intervened
  rollout match the true counterfactual better than a correlational predictor?
- **Mechanism.** Identify a factor direction (e.g. a shape or motion axis) via a probe, perform a
  do-intervention (set/shift that direction), roll forward with `Predictor`, compare to the true
  latent of the counterfactual clip.
- **Baseline it must beat.** A purely correlational predictor (conditions on the observed factor,
  no intervention); and the true counterfactual clip's latent as ground truth.
- **Null hypothesis.** Intervened rollouts do NOT match true counterfactuals better than the
  correlational baseline: the latent factor axes are entangled, so intervening on one drags others,
  and the substrate does not support clean do-operations.
- **Metrics.** counterfactual-match error (intervened vs true-counterfactual latent); leakage
  (how much OTHER factors move under a single-factor intervention); vs correlational baseline.
- **Controls.** correlational predictor; RANDOM-direction intervention (must be worse than the
  identified factor direction); frozen-random substrate is VACUOUS here if the factor probe is
  linear, so use random-init-ViT features as the substrate control.
- **Diagnostic gates.** `compute_accounting`; `difficulty_calibration`; `determinism`; explicit
  entanglement report.
- **Failure interpretations.** High leakage => factors are not linearly separable/independent in
  the frozen latent, causal intervention is not supported (a substrate structure limit). Random
  direction ties the factor direction => the probe found nothing causal, only correlational.
- **Required substrate.** cached latents; REQUIRES factor-annotated video (bound attributes), the
  deferred prerequisite; on synthetic gratings this ceilings.
- **Tier.** studio (needs bound-attribute video). **Difficulty.** high. **Sci value.** high.
- cached-latent-ok: yes (annotated caches). needs-multiple-models: yes (real vs random-init ViT).
  needs-custom-model: no.

### DR6 — Monte-Carlo latent rollouts: does sampling many futures beat one at matched compute
- **Core question.** For a stochastic/ambiguous task, does averaging over many sampled latent
  rollouts (MC) beat a single deterministic rollout at matched total FLOPs (all samples counted)?
- **Mechanism.** Stochastic forward rollouts (dropout or a learned noise head on `Predictor`),
  aggregate the endpoint distribution, act on the aggregate. Total FLOPs = samples x horizon.
- **Baseline it must beat.** A single deterministic rollout given the same total FLOPs (i.e. a
  longer or wider single rollout).
- **Null hypothesis.** MC ties the matched single rollout: for a mostly-deterministic latent
  dynamics the samples collapse to the mean and averaging buys nothing; the compute is better
  spent on one longer rollout.
- **Metrics.** return / accuracy at matched total FLOPs; sample-diversity (collapsed samples => MC
  is pointless); calibration of the endpoint distribution vs true outcome spread.
- **Controls.** single matched rollout; noisy-TV guard (MC must NOT chase irreducible aleatoric
  noise as if it were reducible, `diagnostics/noisy_tv.py`); action-shuffle if actions present.
- **Diagnostic gates.** `compute_accounting`; `noisy_tv`; `determinism`.
- **Failure interpretations.** MC ties single => dynamics too deterministic or samples collapse.
  MC "wins" but fails noisy-TV => it is exploiting aleatoric noise, a false positive.
- **Required substrate.** cached latents; a genuinely stochastic-outcome task.
- **Tier.** cpu-now / env-later. **Difficulty.** high. **Sci value.** medium.
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR7 — Latent self-verification revisited under the corrected controls (ex18/y9 recheck)
- **Core question.** Does a trained verifier that scores a refined latent and triggers a revise
  step beat single-shot AND beat a shuffled/untrained verifier, at matched compute? ex18 found the
  trained verifier "barely beats" a shuffled one; this re-runs it with a difficulty-calibrated
  regime and a real substrate control.
- **Mechanism.** `Verifier.score(z)` triggers an extra `IterativeRefiner` step on low-confidence
  samples (refine.py lines 108-122).
- **Baseline it must beat.** single-shot at matched compute; AND a SHUFFLED verifier (scores
  permuted across batch); AND update-norm confidence (free signal, from MT2).
- **Null hypothesis.** verify-revise ties single-shot at matched compute, and the trained verifier
  ties the shuffled one: the verifier carries no correction-relevant signal (the ex18 result).
- **Metrics.** self-correction gain vs single-shot; trained-minus-shuffled verifier delta; monotone
  descent fraction (does verification monotonically reduce error, y9); AUROC of verifier score vs
  true error.
- **Controls.** single-shot matched; shuffled-verifier; update-norm confidence; regime calibration.
- **Diagnostic gates.** `compute_accounting`; `difficulty_calibration`; `determinism`.
- **Failure interpretations.** Trained ties shuffled => verifier is decorative, the ex18 negative
  confirmed. Gain appears only on an uncalibrated regime => the "correction" was fitting noise.
- **Required substrate.** cached latents; a regime with genuine correctable errors.
- **Tier.** cpu-now. **Difficulty.** high. **Sci value.** medium (rechecks a known near-null with
  better controls).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR8 — Cross-substrate reasoning consistency: does a reasoning gain transfer across encoders
- **Core question.** Any reasoning gain found (MT1-5, DR1-4) must be tested for whether it is a
  property of V-JEPA specifically or of ANY sufficiently rich frozen encoder. This wires the newest
  standing control (cross-substrate convergence) into the reasoning line.
- **Mechanism.** Re-run the winning reasoning primitive on latents from a DIFFERENT frozen encoder
  (a different-objective or different-modality model) and on random-init-ViT features. A universal
  reasoning gain replicates across substrates; a substrate-specific one does not.
- **Baseline it must beat.** The SAME primitive on random-init-ViT features (must beat it) and the
  expectation of replication on a second real encoder.
- **Null hypothesis.** The reasoning gain is identical across all substrates (universal, so it is a
  property of the reasoning shell / task geometry, not the substrate) OR it vanishes on the second
  real encoder (fragile, not a real capability).
- **Metrics.** gain replication delta across substrates; gain on real vs random-init-ViT; rank
  correlation of per-sample benefit across substrates.
- **Controls.** random-init-ViT; second real encoder; the matched-compute baseline carried through.
- **Diagnostic gates.** `compute_accounting`; cross-substrate convergence report; `determinism`.
- **Failure interpretations.** Gain identical on random-init-ViT => not substrate-carried, it is a
  task/shell property (interesting but not a V-JEPA claim). Gain vanishes on encoder 2 => fragile,
  do not generalize.
- **Required substrate.** cached latents from >=2 real encoders + random-init-ViT.
- **Tier.** studio. **Difficulty.** high. **Sci value.** high (gates every reasoning positive).
- cached-latent-ok: yes. needs-multiple-models: yes. needs-custom-model: no.

### DR9 — Memory-first reasoning: retrieve-then-reason beats reason-from-scratch
- **Core question.** Does retrieving similar cached latents from an episodic buffer and conditioning
  the reasoner on them beat reasoning from the query latent alone, at matched compute? Tests whether
  the substrate supports non-parametric memory-augmented reasoning.
- **Mechanism.** kNN over a cached-latent buffer, the retrieved neighbors feed the refiner as extra
  context (concatenate or cross-attend). This is memory-first (retrieve before reason).
- **Baseline it must beat.** Reason-from-scratch (query latent only) at matched compute; AND a
  RANDOM-retrieval control (neighbors picked at random must not match kNN).
- **Null hypothesis.** Retrieval ties from-scratch, or ties random-retrieval: neighbors in frozen-
  latent space carry no task-useful information the query does not, OR the pooled latent's
  neighbor structure is not task-aligned.
- **Metrics.** accuracy vs from-scratch at matched compute; kNN-minus-random-retrieval delta;
  retrieval precision (are neighbors task-relevant); gain vs buffer size.
- **Controls.** from-scratch matched; random-retrieval; shuffled-neighbor-labels; frozen-random is
  VACUOUS if readout is linear, use random-init-ViT for the substrate claim.
- **Diagnostic gates.** `compute_accounting`; `difficulty_calibration`; `determinism`.
- **Failure interpretations.** kNN ties random-retrieval => the frozen latent's metric space is not
  task-aligned, memory-first reasoning is unsupported by the substrate (a substrate-structure
  finding). Retrieval helps but only from same-class neighbors trivially => it is a lookup table,
  not reasoning.
- **Required substrate.** cached latents + episodic buffer (the shell's buffer module).
- **Tier.** cpu-now / studio. **Difficulty.** medium. **Sci value.** high.
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR10 — Reasoning under compression / quantization / corruption / missing channel
- **Core question.** How gracefully does each reasoning primitive degrade as the input latent is
  compressed (VQ / low-rank), quantized (4-bit, the `quantize_dequantize` control), corrupted
  (additive noise, masking), or has a channel dropped (spatial region / temporal segment removed)?
  A reasoning process that recovers information under corruption is doing something a passive
  readout is not.
- **Mechanism.** Apply each degradation to the cached latent, run the reasoning primitive (refine /
  scratchpad / planning) and a single-pass baseline, measure the DEGRADATION SLOPE of each.
- **Baseline it must beat.** A single-pass MLP under the identical degradation: the reasoning
  primitive must degrade MORE GRACEFULLY (flatter slope), not just have higher absolute accuracy.
- **Null hypothesis.** Reasoning and single-pass degrade at the SAME rate under every corruption:
  iteration/search/memory does not recover corrupted information, it just processes what survives.
- **Metrics.** accuracy-vs-corruption-level slope for each primitive vs baseline; recovery gain
  (reasoning-minus-baseline at each corruption level); which corruption (compression / quantization
  / noise / missing-channel) hurts most (a substrate-fragility profile).
- **Controls.** single-pass under identical corruption; `quantize_dequantize` (4-bit) standing
  control; matched compute; noisy-TV guard for the additive-noise arm.
- **Diagnostic gates.** `compute_accounting`; `noisy_tv`; `determinism`; `latent_robustness` (a4
  ablation infra).
- **Failure interpretations.** Equal slopes => reasoning does not recover information, it is passive
  processing (the strong-null, expected). Reasoning recovers under masking specifically =>
  suggestive of a genuine completion/inference capability, one of the few results that would
  actually distinguish reasoning from readout. Missing-channel collapses catastrophically => the
  pooled latent has no redundancy, a substrate fragility.
- **Required substrate.** cached latents; dense latents for the missing-channel arm.
- **Tier.** cpu-now (quantize/noise) / studio (missing-channel needs dense). **Difficulty.** high.
  **Sci value.** high (the graceful-degradation asymmetry is a rare honest reasoning signal).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR11 — Planning-horizon limit: where does rollout error compound past usefulness
- **Core question.** As a standalone axis (pulled out of DR4 so it can be swept cleanly): at what
  horizon does forward-model prediction error compound enough that planning stops beating reactive?
  This bounds internal simulation for THIS substrate.
- **Mechanism.** Sweep planning horizon 1..H with the action-conditioned `Predictor`, logging
  rollout fidelity and planning return at each horizon.
- **Baseline it must beat.** Reactive head, at each horizon, at matched compute.
- **Null hypothesis.** Planning never beats reactive at any horizon (contradicts ex2) OR beats it
  only at horizon 1 (then it is not really planning).
- **Metrics.** the crossover horizon (planning gain -> 0); rollout-fidelity decay curve; return vs
  horizon.
- **Controls.** reactive matched; action-shuffle; rollout-fidelity ground truth from true latents.
- **Diagnostic gates.** `compute_accounting`; `rollout_gate`; `determinism`.
- **Failure interpretations.** Crossover at horizon 1-2 => the frozen latent's forward-
  predictability is too short for meaningful planning, a substrate limit on internal simulation.
- **Required substrate.** cached latents with temporal/action structure.
- **Tier.** cpu-now / env-later. **Difficulty.** medium. **Sci value.** medium (quantifies ex2's
  reach).
- cached-latent-ok: yes. needs-multiple-models: no. needs-custom-model: no.

### DR12 — Disagreement-as-uncertainty: does inter-module disagreement predict error
- **Core question.** Does the disagreement among an ensemble of shell modules (or debate rounds in
  MT4) predict per-sample error better than a single module's own confidence, and does using it to
  gate extra compute beat spending that compute uniformly?
- **Mechanism.** Variance across K independently-seeded shell heads' latent predictions as an
  epistemic-uncertainty signal; gate extra refinement to high-disagreement samples.
- **Baseline it must beat.** Single-head confidence (softmax entropy / update-norm) as the gating
  signal, at matched compute; and uniform extra compute.
- **Null hypothesis.** Disagreement ties single-head confidence as an error predictor, and gating
  on it ties uniform compute: the ensemble variance is redundant with cheaper signals.
- **Metrics.** AUROC of disagreement vs true error; accuracy gain from disagreement-gated compute
  vs uniform at matched total FLOPs; noisy-TV separation (disagreement must track EPISTEMIC not
  ALEATORIC uncertainty).
- **Controls.** single-head confidence; uniform compute; noisy-TV guard (critical: disagreement
  famously chases aleatoric noise); shuffled-head control.
- **Diagnostic gates.** `compute_accounting`; `noisy_tv`; `determinism`; seed-stability.
- **Failure interpretations.** Disagreement ties single-head confidence => no epistemic advantage.
  Disagreement fails noisy-TV => it is tracking irreducible noise, useless as an uncertainty
  signal (the e4 neuromod pattern recurring in a new guise).
- **Required substrate.** cached latents; multiple independently-seeded shells.
- **Tier.** cpu-now. **Difficulty.** medium. **Sci value.** medium.
- cached-latent-ok: yes. needs-multiple-models: no (multiple shells). needs-custom-model: no.

### DR13 — Visual vs language vs relational reasoning: is the gain modality-general
- **Core question.** For any reasoning primitive that survives, is its benefit specific to visual
  (V-JEPA) latents, or does it appear equally on relational-task latents and on language-model
  latents? This separates "reasoning shell that helps on any structured input" from "reasoning that
  exploits visual structure".
- **Mechanism.** Run the identical primitive on three latent sources: V-JEPA video, a
  relational/graph task encoder, and a frozen language-model embedding, holding the shell fixed.
- **Baseline it must beat.** matched-compute single-pass on each modality; the cross-modality
  consistency itself is the observable.
- **Null hypothesis.** The gain is either identical across all three (modality-general, so it is a
  shell/task property, not a visual-reasoning claim) or present only on the easiest modality (an
  artifact of that modality's separability).
- **Metrics.** per-modality gain; cross-modality gain correlation; which modality shows the
  LARGEST substrate-vs-random-init gain (where the substrate matters most).
- **Controls.** per-modality random-init / random-feature control; matched compute.
- **Diagnostic gates.** `compute_accounting`; cross-substrate convergence; `determinism`.
- **Failure interpretations.** Visual-only gain => a genuine visual-reasoning specificity (rare and
  valuable). Uniform gain => reasoning is modality-agnostic, the visual substrate is not special
  for reasoning (even if it is special for perception, per substrate_vs_random).
- **Required substrate.** cached latents from video + relational + language encoders.
- **Tier.** studio. **Difficulty.** high. **Sci value.** high.
- cached-latent-ok: yes. needs-multiple-models: yes. needs-custom-model: no.

---

## 4.4 Coverage map (every requested item to an experiment)

| Requested item | Experiment(s) |
|---|---|
| Recurrent latent refinement | DR1 |
| Adaptive latent compute | MT1 |
| Latent scratchpad | DR3 |
| Latent chain-of-thought without text | DR2 |
| Latent self-verification | DR7 |
| Latent debate between modules | MT4 |
| Tree/beam search over latent states | MT3 |
| MC rollouts in latent space | DR6 |
| Cross-substrate reasoning consistency | DR8 |
| Visual vs language vs relational reasoning | DR13 |
| Memory-first reasoning | DR9 |
| Counterfactual / causal latent intervention | DR5 |
| Planning horizon | DR11 (and DR4) |
| Internal simulation | DR4 |
| Disagreement-as-uncertainty | DR12 |
| Confidence stopping | MT2 |
| Reasoning under compression/quantization/corruption/missing channel | DR10 |
| Mixture-of-thinking router (synthesis) | MT5 |

---

## 4.5 Which reasoning experiments are MOST likely to reveal a frozen V-JEPA-style substrate is insufficient

The doctrine's corrected positive (substrate_vs_random: real V-JEPA carries genuine nuisance-
invariant perceptual structure, delta +0.31 off-ceiling) shows the substrate is SPECIAL for
PERCEPTION. It says nothing about whether it is SUFFICIENT for REASONING, and there are concrete
mechanistic reasons a frozen, mean-pooled video latent could be a bounded substrate for reasoning.
Ranked by how sharply each experiment can expose that bound:

1. **DR3 (latent scratchpad), most diagnostic.** V-JEPA's pooled latent is a fixed-width summary.
   If a task needs to hold more discrete items than the residual width recoverably encodes, a
   scratchpad SHOULD help, and if it does NOT help even at high memory load, that is direct
   evidence the pooled substrate has discarded the fine, separable detail external memory would
   need to store. This is the cleanest "the substrate threw away what reasoning needs" test.
   Requires DENSE (per-token) latents to be fair, which is exactly why the pooled-vs-dense fork
   (doctrine open-fork 1) must resolve first.

2. **DR10 missing-channel arm.** Dropping a spatial region or temporal segment of a DENSE latent
   and asking whether a reasoning process recovers it tests substrate REDUNDANCY. A catastrophic
   collapse means the pooled/entangled latent has no recoverable redundancy, so inference-style
   reasoning (fill in the missing part) is unsupported. Graceful recovery would be one of the few
   positive reasoning signals the corpus could produce.

3. **DR5 (causal intervention) leakage metric.** If single-factor do-operations drag other factors
   (high leakage), the frozen latent does not linearly disentangle the factors reasoning needs to
   intervene on. This is the reasoning-side echo of the corpus's compositional ceiling problem: the
   substrate may bind attributes too entangledly to support clean counterfactual reasoning. Blocked
   on the deferred bound-attribute-video prerequisite (doctrine open-fork 4).

4. **DR2 (latent CoT) shuffle control.** If shuffling the intermediate chain does not hurt, the
   substrate does not support a meaningful sequential decomposition, only more depth, matching the
   ex17 unrolled-depth reading. This is the most likely-to-null of the set (it directly reruns the
   corpus's strongest reasoning negative), which is precisely why a non-null here would be
   important.

5. **DR11 (planning horizon) crossover.** A crossover at horizon 1-2 means the frozen latent's
   forward-predictability is too short to sustain internal simulation, bounding the ex2 positive.

The unifying prediction: the frozen substrate is likely SUFFICIENT for perception-adjacent
reasoning that a single readout could nearly do anyway (where iteration ties depth, the expected
null), and likely INSUFFICIENT precisely where reasoning needs (a) more working memory than the
pooled width carries (DR3), (b) recoverable redundancy (DR10), or (c) disentangled factors to
intervene on (DR5). Those three are the experiments to prioritize once dense and bound-attribute
caches exist, because a clean null in any of them is not "reasoning failed", it is "the substrate
is the binding constraint on reasoning", which is the load-bearing architecture-fork evidence.

A methodological caution that gates all of the above: every experiment whose final answer is read
by a linear/MLP probe on a refined latent inherits the vacuous-control problem, so its
"substrate-specific" claim must use random-init-ViT features, NOT frozen_random_projection. The
experiments that ESCAPE the vacuity (and are therefore the trustworthy substrate-insufficiency
tests) are the trained-shell-DYNAMICS ones: contraction trajectories (DR1), degradation SLOPES
(DR10), planning RETURN on true dynamics (DR4/DR11), and matched-compute Pareto frontiers (MT1),
because those depend on input conditioning, not linear separability. A scratchpad or CoT result
whose readout is a probe must report both the trained-dynamics metric and the acknowledgment that
its probe-based cell is control-blind.
