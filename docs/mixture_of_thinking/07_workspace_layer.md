# Section 7: The Global Workspace / Arbitration Layer

Owner scope: the layer that sits above the shell's parallel components (heads, predictor, ensemble members, and, critically, multiple frozen substrates) and decides what gets combined, what gets broadcast, and what wins. This section designs all fourteen candidate workspace architectures the spec enumerates, gives each its null and matched-capacity control, and then designs in full the one test that motivates the whole layer: does cross-substrate agreement predict correctness better than single-substrate confidence.

## 7.0 Why this layer exists, and why it is dangerous

The doctrine after the vacuous-control discovery is blunt: almost every "combining/routing helps" result in ML is really just extra capacity, and a full-rank invertible projection is enough to erase most probe-level substrate claims. A workspace layer is precisely the kind of mechanism that is trivially confounded by capacity: any fusion network with more parameters than a single-source baseline will usually win, and the win says nothing about workspaces, broadcast, or arbitration. So the entire section is written under one governing rule.

Governing rule (matched-capacity mandatory, no exceptions). Every workspace architecture must beat, at equal trainable-parameter count and equal training FLOPs, a baseline that has the SAME inputs and the SAME capacity but NO workspace structure (no broadcast, no arbitration, no cross-component routing, just a plain MLP over the concatenated inputs). If the structured workspace does not beat the unstructured equal-capacity MLP, the structure is decoration. This is the workspace analogue of frozen-random: a control that most candidates are expected to fail.

Second governing rule (the workspace must earn a SECOND substrate). Nearly all fourteen architectures are only interesting if there is more than one thing to arbitrate between. On a single frozen V-JEPA latent, "fusion" degenerates to "a bigger head", and the corpus already knows a bigger head does not carry special structure a random projection lacks. The layer becomes non-vacuous only when the inputs are genuinely different feature spaces: multiple frozen substrates (V-JEPA plus a second cheap pretrained encoder such as DINOv2 or an audio/text encoder), or a substrate-plus-its-own-forward-model (the predictor as a second, generative view). This is the same doctrinal move as substrate_vs_random_features.py: a genuinely different feature space, not an invertible remap of the same one.

Non-negotiable controls wired into every WS experiment below:
- matched-capacity unstructured MLP (the primary control, replaces frozen-random for this layer)
- single-best-substrate baseline (pick the strongest single source; the workspace must beat the best member, not just the average member, or it is Condorcet bookkeeping not arbitration)
- shuffled-cross-source control (permute which sample's source-B latent pairs with source-A; kills any genuine cross-source binding, leaving only marginal information)
- matched-compute (a workspace that iterates or broadcasts for K rounds must beat a feedforward net unrolled to the same FLOPs, per the p9/ex17 finding that iteration on this substrate is just unrolled depth)
- seed-stability (publish sign-flips; arbitration gains near noise floor are instability)

## 7.1 The fourteen candidate architectures

For each: mechanism; good for; cannot do; baseline it must beat; null that falsifies it; metrics; how to avoid it being just extra capacity; cached-latent-ok; studio-or-not; step-toward-custom-model-or-not.

### A1. Concat baseline (the reference floor, not a real candidate)
- Mechanism: stack the pooled latents from all sources into one vector, feed a single linear/MLP head. No structure.
- Good for: the honest floor. This is the thing every other architecture must beat. It is the maximal-information, minimal-structure combiner.
- Cannot do: any input-dependent routing, any disagreement signal, any broadcast. It cannot down-weight a source that is wrong on this input.
- Baseline it must beat: single-best-substrate (concat must at least beat the best single source, else the sources are redundant and the whole layer is moot).
- Null: concat over both sources ties single-best-source; the second substrate adds no decodable information. (If this null holds, the entire section collapses to "you only ever needed one substrate.")
- Metrics: task accuracy, calibrated NLL, delta over single-best.
- Avoid-capacity: it IS the capacity control at its own capacity; report accuracy-per-parameter.
- Cached-latent-ok: yes.
- Studio: no, cpu-now.
- Toward custom model: no. Pure diagnostic floor.

### A2. Learned linear fusion
- Mechanism: a learned linear map per source into a shared space, then sum: y = W_a z_a + W_b z_b. Global, input-independent weights.
- Good for: discovering a fixed complementary linear combination (source A good on shape, source B good on motion), and being cheap.
- Cannot do: input-dependent trust. The weights are fixed, so it cannot say "on THIS clip trust B".
- Baseline it must beat: concat + linear (A1 at matched capacity). Linear fusion is a strict subspace of concat-linear, so it can only match or lose on training fit; it must win on GENERALIZATION or calibration to justify itself.
- Null: linear fusion equals concat-linear on held-out accuracy (expected, since it is a constrained concat). If it never beats concat on any held-out axis, it is a regularizer, not a workspace.
- Metrics: held-out accuracy, held-out NLL, effective rank of the fused representation.
- Avoid-capacity: constrain to fewer or equal params than concat; the point is it must win DESPITE less capacity (structure-as-regularization is the only honest win here).
- Cached-latent-ok: yes.
- Studio: no, cpu-now.
- Toward custom model: no.

### A3. Attention fusion (cross-source attention, no broadcast)
- Mechanism: source tokens attend to each other via one or more attention layers; output is attention-pooled. Input-dependent mixing weights (softmax over sources/tokens), but no serialized bottleneck.
- Good for: input-dependent, content-based combination; the first architecture that can down-weight a wrong source per-input.
- Cannot do: enforce a low-bandwidth bottleneck (attention can route everything), and it is the archetypal capacity trap (attention layers add many parameters and a strong inductive bias).
- Baseline it must beat: matched-capacity MLP over concat (same params, same FLOPs) AND single-best-source.
- Null: at matched capacity, attention fusion ties the unstructured MLP; the "attention" structure buys nothing beyond parameter count. (This is the expected outcome and the main thing to test.)
- Metrics: accuracy, NLL, plus attention-weight analysis (does it actually route input-dependently, or collapse to a fixed convex combination equal to A2?).
- Avoid-capacity: freeze total params equal to the MLP control; additionally report the degenerate check (if learned attention weights are near-constant across inputs, it has collapsed to A2 and the attention is vacuous).
- Cached-latent-ok: yes.
- Studio: borderline; multi-source attention over dense tokens is studio-scale, pooled-vector attention is cpu-now.
- Toward custom model: partial. Cross-source attention is the reusable core of a custom multi-substrate encoder.

### A4. Global workspace with broadcast (GWT proper, the Baars/Dehaene mechanism)
- Mechanism: a narrow shared latent (the workspace slot, dimension much smaller than any source) that sources COMPETE to write to; the winning content is then BROADCAST back to all sources/heads before the final readout. The bottleneck is the point (corpus vol2 line 965). Implemented per Goyal et al. 2022 shared-workspace coordination.
- Good for: forcing selection and serialization; testing whether a low-bandwidth shared channel improves coordination between otherwise-independent components. The bottleneck should help generalization if there is genuine shared structure to compress into.
- Cannot do: help when the sources are already redundant (nothing to coordinate) or when the task needs full high-bandwidth fusion (the bottleneck then only hurts). Cannot demonstrate "consciousness" or "access" (explicitly out of scope, engineering-only).
- Baseline it must beat: (1) unbottlenecked matched-capacity fusion (A3/MLP), (2) plain regularization at matched capacity (dropout/weight-decay tuned to the same effective bandwidth), per corpus vol2 line 975 which pre-registers exactly this null.
- Null: the bottleneck's benefit is indistinguishable from generic regularization at matched capacity; broadcast adds nothing over write-only. Two separable nulls: (a) narrow-shared vs wide-shared is just capacity, (b) broadcast-back vs write-only is just more layers.
- Metrics: held-out accuracy/NLL vs bandwidth (sweep workspace width), coordination metric (mutual information between sources after broadcast minus before), ablation of the broadcast step alone.
- Avoid-capacity: sweep workspace width as the independent variable at FIXED total params (widen elsewhere to hold capacity); the claim is a U-shaped or plateau benefit of NARROWNESS, which capacity cannot fake because capacity is held constant.
- Cached-latent-ok: yes.
- Studio: cpu-now for pooled; studio for dense-token slots.
- Toward custom model: yes, strongly. A working broadcast bottleneck is a genuine architectural primitive, not a probe.

### A5. Recurrent workspace
- Mechanism: A4's shared slot updated over K recurrent steps; sources read/write the slot each step; readout after K steps.
- Good for: multi-step integration where evidence accumulates (e.g., resolving a source disagreement by iterating). Tests whether iteration over a shared state helps beyond depth.
- Cannot do: escape the p9/ex17/n9/y1 verdict unless it shows a matched-compute gain AND fixed-point convergence, both of which the corpus has repeatedly failed to find for latent iteration on this substrate.
- Baseline it must beat: a feedforward workspace unrolled to K-equal FLOPs (matched-compute), not just K=1.
- Null: gain(K) minus gain(depth at same FLOPs) = 0 (the exact p9/ex17 result); no convergence to a fixed point (n9 residual test); unrolling past training K worsens loss (y1). Any of these firing falsifies "recurrence helps".
- Metrics: accuracy vs K, matched-compute delta, fixed-point residual, past-horizon degradation curve.
- Avoid-capacity: recurrence shares weights across steps, so it is naturally parameter-cheap; the honest control is the FLOP-matched feedforward net, and the honest claim (if any survives) is parameter-efficiency, exactly the narrow residual p9 already found.
- Cached-latent-ok: yes.
- Studio: cpu-now.
- Toward custom model: partial, but the corpus prior is strongly negative; low expected value.

### A6. Graph workspace
- Mechanism: sources/components as nodes, learned or fixed edges, message passing (GNN) to a readout. Structured relational fusion.
- Good for: when there is genuine relational structure among more than two sources (e.g., object-token graphs, or several substrates with a known compatibility topology).
- Cannot do: justify itself with only two sources (a graph over two nodes is just A3), and it is a capacity trap via edge/message MLPs.
- Baseline it must beat: matched-capacity MLP AND a complete-graph-with-uniform-edges ablation (if learned edges do not beat uniform edges, the graph structure is vacuous).
- Null: learned edge structure ties uniform/complete edges at matched capacity; the topology carries no information.
- Metrics: accuracy, NLL, edge-weight informativeness (accuracy drop when edges are shuffled), number-of-sources scaling.
- Avoid-capacity: fix message-MLP capacity equal to control; vary ONLY the edge structure.
- Cached-latent-ok: yes.
- Studio: cpu-now for few nodes.
- Toward custom model: partial; only interesting once there are more than two substrates.

### A7. Object-token workspace
- Mechanism: slot-attention style object slots (a small set of competing slots that bind to spatial regions of the DENSE substrate tokens) feed the workspace; objects, not the pooled global vector, are the units arbitrated.
- Good for: THE deferred prerequisite (P7 object-binding-before-pooling, corpus vol3). This is the one architecture that directly attacks the pooling ceiling: it operates on pre-pool tokens, so it can carry binding structure a pooled vector destroys.
- Cannot do: run on cached POOLED latents (needs dense tokens), so it is blocked until dense-token caching exists; and slot attention is notoriously seed-unstable (publish sign-flips).
- Baseline it must beat: pooled-vector workspace (A4) on a NON-ADDITIVE binding task, and a random-slot-assignment control (slots assigned randomly to regions), and matched-capacity.
- Null: object slots tie pooled global vector on held-out attribute BINDING (not marginal decode); binding is additive so slots buy nothing. Given dense_vs_pooled.json showed single factors decode perfectly from the pooled vector, the null here is strong until a genuinely non-additive test bed exists.
- Metrics: held-out-combination accuracy (bound shape x color unseen pairing), slot-assignment stability across seeds, binding-vs-marginal gap.
- Avoid-capacity: match slot-network params to a pooled MLP; the claim must be a BINDING win, which capacity over a pooled vector provably cannot produce if the pooled vector lacks the spatial structure.
- Cached-latent-ok: NO (needs dense tokens, blocked on P7 prerequisite).
- Studio: yes, dense-token, studio-scale.
- Toward custom model: yes, strongly; slot binding before pooling is a core custom-architecture bet.

### A8. Memory-augmented workspace
- Mechanism: workspace slot plus an external key-value memory (reuse shell/buffer.py) that the workspace reads/writes; broadcast content can be retrieved from past episodes.
- Good for: cross-episode arbitration and few-shot binding (the workspace consults memory of similar past inputs before deciding). Ties to the surviving e7_sparse continual-learning line.
- Cannot do: help on i.i.d. single-shot tasks (nothing to retrieve); risks being a lookup table (memorization, not concept, per p2).
- Baseline it must beat: workspace-without-memory (A4) AND a nearest-neighbor-only baseline (if kNN over memory alone matches, the workspace adds nothing).
- Null: memory read is uninformative (retrieval-shuffled control ties full retrieval); or the win is pure memorization (fails p2 memorize-vs-concept: performance collapses on held-out combinations).
- Metrics: held-out accuracy, retrieval-shuffle delta, memorize-vs-generalize gap, forgetting/BWT (does memory-augmented workspace inherit e7_sparse's retention advantage).
- Avoid-capacity: memory is non-parametric; match the CONTROLLER capacity and report whether the win survives retrieval-shuffle (which holds capacity fixed and destroys only the memory content).
- Cached-latent-ok: yes.
- Studio: cpu-now.
- Toward custom model: partial.

### A9. Uncertainty-weighted fusion
- Mechanism: each source emits a prediction AND a calibrated uncertainty (via shell/heads.py gaussian head or shell/ensemble.py disagreement); fuse by precision-weighting (inverse-variance / Bayesian combination). Input-dependent weights derived from uncertainty, not learned attention.
- Good for: the principled optimal-fusion baseline; down-weights whichever source is uncertain on this input. This is the natural competitor to A3 attention and the mechanistic form of the central test (7.2).
- Cannot do: help if the uncertainties are miscalibrated (garbage precision, garbage fusion), which the corpus warns is common (e4 amplified error on noise, wrong direction); cannot handle correlated errors between sources (precision-weighting assumes independence).
- Baseline it must beat: equal-weight average AND concat-MLP AND single-best-source. Crucially it must beat SINGLE-SOURCE-CONFIDENCE fusion (weight the two views of ONE substrate), which is the null for the whole cross-substrate thesis.
- Null: precision weights are uninformative (calibration is bad), so inverse-variance fusion ties equal-weight; noisy-TV guard must pass (uncertainty must not fire on irreducible aleatoric noise, per diagnostics/noisy_tv.py, the exact e4 failure mode).
- Metrics: calibration (ECE per source), fusion accuracy vs equal-weight, noisy-TV pass/fail, correlation-of-errors between sources.
- Avoid-capacity: uncertainty-weighting adds ~zero parameters (it is a fixed Bayesian rule over source outputs), so it is the ONE architecture that is intrinsically capacity-neutral; a win here is the cleanest possible workspace result.
- Cached-latent-ok: yes.
- Studio: no, cpu-now.
- Toward custom model: no (it is a fusion rule), but it is the load-bearing mechanism for the central test.

### A10. Disagreement-driven workspace
- Mechanism: route/allocate compute based on cross-source (or ensemble) DISAGREEMENT: when sources agree, take the cheap consensus; when they disagree, invoke a more expensive resolver (extra rounds, memory read, or a learned tie-breaker). Disagreement as the control signal.
- Good for: the speed-accuracy frontier (D2/D3 arbitration from corpus vol2), and directly operationalizing "agreement means correct". Ties model-based/model-free arbitration to the workspace layer.
- Cannot do: help if disagreement does not track error (the entire premise, tested in 7.2); risks noisy-TV (disagreement can be high on irreducible noise, wasting compute forever).
- Baseline it must beat: always-cheap and always-expensive (arbitration must dominate BOTH on the speed-accuracy frontier, per corpus vol2 line 561), and disagreement-shuffled (route by a random signal at the same rate).
- Null: disagreement is uncorrelated with correctness (random routing does as well); or disagreement fires on aleatoric noise (noisy-TV fail) and over-invokes the resolver.
- Metrics: correctness-vs-disagreement AUC, compute saved at matched accuracy, frontier dominance vs both pure strategies, noisy-TV pass.
- Avoid-capacity: the router is tiny; the resolver capacity must be matched to an always-on baseline of the same total FLOPs (so the win is ALLOCATION, not extra capacity spent everywhere).
- Cached-latent-ok: yes.
- Studio: no, cpu-now.
- Toward custom model: partial; the arbitration signal is reusable.

### A11. Predictive-coding workspace
- Mechanism: the workspace maintains a top-down prediction of each source's latent; sources send only PREDICTION ERRORS up; the workspace state is updated to minimize total prediction error. Hierarchical predictive coding as the fusion rule (reuse shell/predictor.py, learning/alternatives predictive-coding).
- Good for: principled error-driven fusion, and a mechanism where a source that is well-predicted (redundant) is automatically suppressed; naturally surprise-gated.
- Cannot do: escape the I4/ex17 finding that predictive-coding-style local rules tie backprop at matched compute; cannot demonstrate iterative benefit without failing the matched-compute null (p9).
- Baseline it must beat: backprop-trained fusion at matched compute (I4 already shows PC ties backprop), and a feedforward net at matched FLOPs.
- Null: PC fusion ties backprop fusion at matched compute (the corpus prior, likely to hold); the error-suppression dynamic is just whitening a random projection could also do.
- Metrics: matched-compute accuracy delta, prediction-error reduction curve, convergence (does the error settle, per n9), redundancy-suppression check.
- Avoid-capacity: shared top-down weights make it parameter-cheap; match FLOPs to feedforward; expected honest residual is parameter-efficiency only.
- Cached-latent-ok: yes.
- Studio: cpu-now.
- Toward custom model: partial; low prior given I4/ex17.

### A12. Active-inference workspace
- Mechanism: A11 plus ACTION: the workspace selects actions (or attention shifts, or which source to query next) to minimize expected free energy (predicted future prediction-error + information gain). Perception-action loop over sources.
- Good for: the only architecture that closes the loop (query the source that most reduces expected uncertainty); the deferred P8/A9 closed-loop lane.
- Cannot do: run without an environment or an action-conditioned proxy (A5 action-loop already failed its own control); is the most confounded and least testable offline.
- Baseline it must beat: random source-querying AND greedy-uncertainty (query highest-uncertainty source, i.e., A9 without the expected-free-energy machinery). The full EFE machinery must beat greedy, or it is overhead.
- Null: expected-free-energy querying ties greedy-uncertainty querying; the information-gain term is uninformative; no environment means no real action cost so the whole objective is vacuous.
- Metrics: uncertainty reduction per query, task accuracy at fixed query budget, EFE-vs-greedy delta.
- Avoid-capacity: match the policy-network capacity; the claim is the OBJECTIVE (EFE) beats greedy, capacity held equal.
- Cached-latent-ok: partially (offline query-selection over cached multi-source latents is a valid cpu-now proxy; full action loop is environment-needed).
- Studio: environment-needed for the full loop; cpu-now for the offline query proxy.
- Toward custom model: partial; high risk given A5's prior failure.

### A13. Modular router plus shared workspace (mixture-of-experts arbitration)
- Mechanism: a learned router assigns each input (or token) to one of several expert heads, PLUS a shared workspace slot the experts read/write. Combines sparse routing (the surviving e7_sparse kWTA/MoE line) with a global channel.
- Good for: directly extends the ONE surviving architectural positive (e7_sparse: sparse/gated heads halve forgetting). Tests whether adding a shared workspace on top of proven sparse routing adds arbitration value or just capacity.
- Cannot do: claim the workspace helps if the win is entirely attributable to the sparse routing already validated in e7_sparse (must ablate the shared slot).
- Baseline it must beat: e7_sparse routing WITHOUT the shared workspace (ablate the slot), AND dense matched-capacity. The novel claim is strictly the shared-slot delta over pure sparse routing.
- Null: shared workspace slot adds nothing over sparse routing alone (slot-ablation ties full); the routing does all the work.
- Metrics: forgetting/BWT (inherit e7_sparse's 30-run sweep protocol), slot-ablation delta, expert-utilization balance, matched-capacity accuracy.
- Avoid-capacity: match total params to dense AND to slot-ablated routing; the claim is the slot delta, everything else held fixed.
- Cached-latent-ok: yes (e7_sparse's protocol; real cached latents needed to close its open question 5).
- Studio: cpu-now.
- Toward custom model: yes; builds on the corpus's strongest surviving architectural result.

### A14. Latent-language / shared-code workspace
- Mechanism: sources communicate through a DISCRETE shared code (VQ codebook, reuse i9_vq_rate_distortion) rather than continuous vectors; the workspace is the shared vocabulary. Tests whether a discretized shared channel is a better coordination substrate than continuous fusion.
- Good for: the emergent-communication / shared-language line (p5/s5/y3), and a genuine bottleneck (discrete codes are low-bandwidth by construction).
- Cannot do: escape the corpus verdict that emergent codes are idiolects, not shared languages (p5/s5/y3 fell BELOW the frozen-random floor); discretization usually loses accuracy unless there is genuine shared structure to quantize.
- Baseline it must beat: continuous shared workspace (A4) AND a random-codebook control (assign codes randomly; if learned codes tie random codes, the vocabulary is vacuous, the exact p5/s5 failure).
- Null: discrete codes tie continuous fusion in accuracy while adding no cross-source alignment; learned codebook ties random codebook (idiolect, not language); the code is below the frozen-random floor.
- Metrics: cross-source code-agreement (do both sources map the same concept to the same code, above random), accuracy vs continuous, codebook-utilization, held-out-combination decode from codes.
- Avoid-capacity: match encoder capacity; the claim is the SHARED-DISCRETE structure, tested by random-codebook ablation (capacity-neutral).
- Cached-latent-ok: yes.
- Studio: cpu-now.
- Toward custom model: partial; strong negative prior from p5/s5/y3.

## 7.2 The central test, designed in full

Central question: DOES CROSS-SUBSTRATE AGREEMENT PREDICT CORRECTNESS BETTER THAN SINGLE-SUBSTRATE CONFIDENCE?

This is the load-bearing test of the entire workspace layer and the concrete instantiation of the newest standing control (cross-substrate convergence: universal vs modality/objective/architecture-specific structure). If it fails, most of the fourteen architectures lose their reason to exist, because arbitration between substrates only pays if agreement carries information that a single substrate's own confidence does not.

Setup:
- Substrate A: frozen V-JEPA 2 ViT-L (the incumbent), pooled latents from the existing cache.
- Substrate B: a genuinely DIFFERENT frozen encoder, not an invertible remap of A. Candidates in cost order: (1) a second pretrained vision encoder with a different objective/architecture (DINOv2 ViT, image-SSL contrastive, different training signal than V-JEPA's masked-latent-prediction), cheap enough to run on this laptop's CPU/MPS per the corpus lane-4 note; (2) as a weaker fallback, the V-JEPA forward-model prediction as a second "generative" view of the same clip (same substrate, different readout, a within-substrate agreement control, NOT a true cross-substrate test but a useful lower bound). The primary experiment REQUIRES option 1: a second encoder with a different objective. This is what makes the test non-vacuous, exactly as substrate_vs_random_features.py needed a genuinely different feature space.
- Task: any decodable target where errors actually occur (NOT a ceiling task; run diagnostics/difficulty_calibration.py first, per the corpus ceiling problem). The nuisance-invariant shape-identity task from substrate_vs_random_features.py (0.379 real vs chance 0.167, off-ceiling) is the ready-made non-ceiling regime.

Two competing predictors of correctness, computed per input:
1. Single-substrate confidence: for the best single substrate, its own calibrated confidence in its prediction (max softmax, or ensemble/gaussian-head uncertainty from shell/ensemble.py and shell/heads.py). This is the "you never needed a second substrate" baseline.
2. Cross-substrate agreement: a scalar measuring whether A and B AGREE on this input. Two forms, both reported: (a) prediction agreement (do A's and B's independent heads predict the same label), (b) representational agreement (cosine/CKA-style alignment of A's and B's latents after a fixed linear map, or agreement of their nearest-neighbor sets). Agreement is computed WITHOUT the label.

The measurement: for each input, we have {correct/incorrect}, {single-substrate confidence}, {cross-substrate agreement}. Compute how well each signal predicts correctness:
- Selective-risk / risk-coverage curves: sort inputs by each signal, plot accuracy on the top-k most-confident (or most-agreeing). The signal with the lower area-under-risk-coverage (higher accuracy at every coverage) is the better correctness predictor.
- AUROC of each signal for the binary correct/incorrect outcome.
- The decisive comparison: does agreement's risk-coverage curve dominate confidence's, and by how much, with seed-level confidence intervals.

Nulls and controls (this test lives or dies on them):
- N1 (the primary null): cross-substrate agreement does NOT beat single-substrate confidence at predicting correctness (agreement AUROC minus confidence AUROC less-than-or-equal-to 0 within seed CI). If N1 holds, the workspace layer's central premise is refuted and A9/A10/the whole arbitration family lose their motivation.
- N2 (vacuity guard, the analogue of the invertible-projection discovery): substrate B must NOT be an invertible linear remap of A. Enforce by checking that a linear probe trained on A does NOT recover B's predictions at the same accuracy as B itself (if it does, B is redundant and "agreement" is autocorrelation). This is the single most important control: it prevents the workspace version of the vacuous frozen-random result.
- N3 (shuffled-cross-source): permute which sample's B-latent pairs with each A-latent; agreement should collapse to chance-predictive of correctness. If shuffled agreement still predicts correctness, the signal is a marginal artifact (e.g., agreement is just "easy inputs"), not genuine cross-substrate binding.
- N4 (noisy-TV): on inputs with injected irreducible aleatoric noise, agreement must NOT masquerade as high correctness-signal; disagreement on pure noise must be recognized as aleatoric, not routed for more compute (diagnostics/noisy_tv.py). This is the exact e4 failure mode (amplified error on noise, wrong direction) and must be guarded.
- N5 (correlated-error caveat): report the correlation of A's and B's errors. If A and B fail on the same inputs (correlated errors), agreement is HIGH exactly when both are wrong, and agreement will UNDER-predict error. A genuine positive requires that agreement tracks correctness even where errors are partially decorrelated; publish the error-correlation as a headline number, because it bounds how much any fusion (A9, A10) can ever help.
- Seed-stability: run at 5+ seeds; publish sign-flips of the agreement-minus-confidence delta as instability, per doctrine.

Interpretation matrix (all outcomes pre-registered, no HARKing):
- Agreement beats confidence, survives N2/N3/N4, errors partially decorrelated: FIRST valid evidence that cross-substrate convergence carries correctness information a single substrate's own confidence lacks. This licenses A9 (uncertainty-weighted) and A10 (disagreement-driven) and tilts toward a multi-substrate workspace being real, not decoration. It is the workspace-layer analogue of the +0.31 nuisance result.
- Agreement ties confidence (N1 holds): the second substrate is redundant for correctness estimation; arbitration is bookkeeping; the layer collapses toward single-substrate plus A1 concat. Report as a clean NEGATIVE (a cross-substrate-convergence null).
- Agreement beats confidence but FAILS N2 (B is an invertible remap of A): VACUOUS, the exact trap the corpus already fell into once; discard and find a more different B.
- Agreement beats confidence but FAILS N3/N4 (shuffle or noisy-TV): the signal is a difficulty/noise artifact; report as a refuted positive.
- Errors fully correlated (N5): even a real agreement signal cannot help fusion; report the ceiling on the whole arbitration family.

Cached-latent-ok: yes, once substrate B's latents are cached (one extra encode pass, same doctrine as the existing cache). This is a cached-latent-first experiment.
Studio-or-not: cpu-now for pooled latents from both encoders; the second encoder download is the only new resource and the corpus already judged it laptop-feasible (lane 4).
Step-toward-custom-model: the test itself is a diagnostic, not an architecture. But its outcome is the gate: a positive is the strongest available justification for building a real multi-substrate custom workspace; a negative is a strong reason NOT to, and to stay single-substrate.

## 7.3 Priority and dependency ordering

1. WS1 (the central test) runs FIRST and gates everything. No workspace architecture is worth building if agreement does not beat confidence. It also requires only a second cheap encoder plus existing cache machinery.
2. WS2 (matched-capacity fusion tournament: A1 concat vs A2 linear vs A3 attention vs A4 broadcast) runs second, ONLY at strictly matched capacity, to establish whether ANY structured fusion beats the unstructured MLP floor. Expected doctrinal prior: most tie the MLP (structure is decoration), mirroring the corpus's other negatives.
3. WS3 (uncertainty-weighted vs disagreement-driven, A9 vs A10) runs only if WS1 is positive; it is the mechanistic payoff of a positive central test and is capacity-neutral (the cleanest possible win).
4. WS4 (broadcast-bottleneck bandwidth sweep, A4) tests the one GWT-specific claim (narrowness helps at fixed capacity) against the pre-registered regularization null from corpus vol2 line 975.
5. WS5 (modular-router-plus-shared-slot, A13) extends the surviving e7_sparse positive with a slot-ablation to isolate any workspace contribution from proven sparse routing.
6. Deferred: A7 object-token workspace is blocked on the dense-token / P7 prerequisite (the same ceiling constraint that gates the whole program); A12 active-inference is blocked on an environment; A14 latent-language carries a strong negative prior from p5/s5/y3. These are registry-only until their prerequisites land.

The through-line: the workspace layer is presumed decoration (matched-capacity MLP wins, agreement ties confidence) until WS1 shows cross-substrate agreement carries real correctness information, exactly as the substrate was presumed unspecial until the non-vacuous random-encoder control showed otherwise.
