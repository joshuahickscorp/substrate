# Section 11: The Unified Mixture-of-Thinking Experiment Registry

This is the single deduplicated registry for the Mixture-of-Thinking (MoT) program. It merges every
candidate proposed across sections 01 (thesis), 03 (thinking-mode taxonomy), 04 (reasoning program),
05 (plasticity program), 06 (cognitive-currencies atlas), 07 (workspace layer), 08/09 (custom-model
pathway), and 15 (the brake). Overlapping proposals were merged into one row (see the dedup ledger at
the end), not listed twice.

Prefixes and what they cover:
- **MT** (Mixture of Thinking): routing/arbitration among heterogeneous thinking modes, adaptive
  compute, and the synthesis router.
- **DR** (Deliberative Reasoning): individual latent reasoning primitives (refine, search, plan,
  verify, memory, intervention) and the deferred video-cache prerequisite.
- **PR** (Plasticity/Learning): continual-learning, consolidation, fast/slow weights, retrieval,
  content-gated critical periods, and the substrate-vs-random plasticity test.
- **WS** (Workspace/arbitration): cross-substrate fusion, global-workspace bottleneck, disagreement
  arbitration.
- **AT** (Atlas/substrate typing): the non-vacuous substrate controls, cross-substrate nuisance grid,
  time-axis ablation, programmatic ceiling, probe-class sweep.
- **AL** (Alignment): cross-substrate/cross-modal latent alignment above a random-map floor, plus the
  uncertainty router with the noisy-TV guard.
- **CM** (Custom Model): the compositional/adaptation gates and pilots that would justify (or close)
  the custom-encoder branch of the fork.

All rows carry the program's standing controls (beat-frozen-random via the NON-vacuous random-ENCODER
control, match-compute, tuned-baseline, noisy-TV, seed-stability, cross-substrate convergence). The
column **Custom-model decision** states whether the row moves the frozen-dense-vs-custom fork, since
that decision is the program's live question.

Compute tiers (from section 10): **cpu-now** = Tier 0 (M3 Pro, cached-latent-first);
**studio** = Tier 1 (studio-1tb profile, one extra cached encode pass, no training on pixels beyond
a cheap encode); **wider-box** = Tier 2 (rented CUDA, violates frozen doctrine on purpose under a
pre-registered hypothesis); **moonshot** = full multi-expert capstone. All cached-latent work is
CPU-bound and must NOT run concurrently with the in-flight V-JEPA encode/ViT jobs (OOM risk).

Within each prefix, rows are ordered MOST-DECISIVE FIRST (the gate/precondition experiments and the
central thesis tests before the extensions).

---

## MT: Mixture-of-Thinking routing and adaptive compute

Central thesis of the whole program: routing among heterogeneous modes beats the single best mode at
matched mean compute, and the coordinator (not averaging, not extra depth) carries the gain.

### MT overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| MT1 | Router beats single best mode | A per-episode router over {reactive, planner, sparse} beats the best single mode at matched compute | cpu-now | med | high (core thesis) | Indirect: modes ride the frozen substrate; a win argues the shell (not a new encoder) is the lever |
| MT2 | Routing beats uniform ensemble | Learned/measured routing beats an equal-weight blend of the same modes at matched total compute | cpu-now | med | high | No |
| MT3 | Heterogeneous beats homogeneous MoE | A mixture of qualitatively different modes beats k copies of the best mode at equal params/FLOPs | cpu-now | med | high | No |
| MT4 | Router over reasoning primitives (synthesis) | A learned router over {fixed-N, adaptive-halt, beam, memory, plan, verify} beats the single best strategy at matched expected compute | studio | high | high (synthesis test) | Indirect: consumes DR primitives; a win is a shell result, not a substrate result |
| MT5 | Adaptive halting beats fixed depth | Per-sample adaptive halting beats fixed-N refinement at equal AVERAGE FLOPs by allocating depth to hard inputs | cpu-now | med | high (the only honest iteration-beats-depth framing) | Weakly: exposes whether the frozen latent has exploitable per-sample hardness heterogeneity |
| MT6 | Confidence stopping beats free early-exit | A trained halt head beats a free update-norm early-exit rule at matched mean FLOPs | cpu-now | low | med | No |
| MT7 | Beam/tree search beats greedy refinement | K scored latent trajectories + pruning beat greedy single-chain refinement at matched total FLOPs | cpu-now | med | med | No |
| MT8 | Latent debate beats single module | Two seeded modules critiquing via a referee beat one module AND a plain ensemble at matched total FLOPs | cpu-now | med | med | No |

### MT full schema

| Field | MT1 Router-vs-best | MT2 Routing-vs-ensemble | MT3 Hetero-vs-homo | MT4 Reasoning-router (synthesis) | MT5 Adaptive-halt-vs-depth | MT6 Confidence-stop | MT7 Beam/tree search | MT8 Latent debate |
|-------|---|---|---|---|---|---|---|---|
| **Thesis** | Cheap per-episode router over {reactive readout, ex2 MPC planner, e7 sparse head} beats the single best mode on capability density (accuracy/FLOP, retention/byte) at matched compute | ROUTING (selective per-episode compute) beats a uniform equal-weight ENSEMBLE of the same modes at matched total compute | Mode DIVERSITY is load-bearing: heterogeneous mixture beats k copies of the best mode at equal params/FLOPs | A tiny softmax router over {reactive, iterative-refine, latent-plan, verify-revise} beats the best fixed mode at matched mean FLOPs because modes make different per-sample errors | Per-sample adaptive halting spends less total compute than fixed-N for equal accuracy | A trained halt head carries a real per-sample correctness signal beyond "the latent stopped moving" | Maintaining K scored candidate trajectories and pruning beats greedy | Structured latent exchange between two seeded modules via a referee beats one and beats a plain ensemble |
| **Mechanism** | Scalar-difficulty selector cheaper than the cheapest mode dispatches one mode per episode | Same modes; arms differ only in router vs fixed uniform blend | Two shells at identical budget: heterogeneous vs k-copy homogeneous, routed identically | Softmax router head selects a mode; matched_within enforces equal mean FLOPs | IterativeRefiner halt path allocates depth per input | Learned halt head vs free update-norm threshold | Verifier scores branches; pruned branches counted in FLOPs | Referee arbitrates iterative critiques between modules |
| **Substrates** | Cached V-JEPA 2 ViT-L pooled latents | Same as MT1 | Same as MT1 | Same shell primitives on cached latents | Cached latents, difficulty-graded regime | Cached latents | Cached latents, ambiguous-target task | Cached latents, multiple seeded shells, one frozen encoder |
| **Trainable** | Router selector; reuse existing heads/planner/sparse head | Router only | Two shells' heads/routers | Router head + primitive heads (reused) | Halt head + refiner | Halt head | Verifier scorer + refiner | Two module heads + referee |
| **Frozen** | V-JEPA encoder (no_grad) | Encoder | Encoder | Encoder | Encoder | Encoder | Encoder | Encoder |
| **Data/task** | Synthetic continual-stream pilot; real bound-attribute cache for the real answer (blocked on DR1) | Same set as MT1 | Same set | Difficulty-graded reasoning regime (D3-certified) | Difficulty-graded regime | Same | Multi-modal/ambiguous-target regime | Multi-answer regime with correctable errors |
| **Null hypothesis** | Routed density equals or trails its best constituent mode at matched compute, or the gap sits inside the seed spread | Equal-weight blend matches the routed system's density at matched total compute; the router adds nothing over averaging | Homogeneous k-copy MoE matches the heterogeneous mixture at equal param/FLOP; diversity adds nothing beyond generic mixture capacity | At matched mean FLOPs the router ties or loses to the single best fixed mode; any win is bought compute or vanishes under seed sweep | At equal average FLOPs, adaptive halting ties fixed-depth: no exploitable hardness heterogeneity, or the halt head collapses to a constant | The trained halt head ties the free update-norm rule; confidence is just the latent ceasing to move | At matched total FLOPs, search ties a deeper greedy chain because the scorer cannot outrank the refiner's own step | Debate ties the single matched module AND ties a plain ensemble; structured exchange is an unrolled ensemble/MoE |
| **Falsification** | Density delta < seed spread at matched compute | Router density <= uniform-blend density | Hetero density <= homo density at matched budget | Router mean-FLOP-matched delta <= 0 or sign-flips | Adaptive AUC <= fixed-depth at equal mean FLOPs, or halt collapses to constant N | Trained-halt <= free-rule at matched FLOPs | Search <= matched greedy depth | Debate <= max(single, ensemble) |
| **Positive interp** | Coordination over heterogeneous modes buys density the best single mode cannot; the MoT thesis holds | The coordinator, not averaging, carries the gain | Qualitative diversity is the mechanism, not generic mixture capacity | The literal MoT synthesis works; per-sample error diversity is exploitable | Iteration beats depth ONLY via allocation, the one honest framing | Confidence is a learnable signal, not a free byproduct | A verifier can outrank the refiner's own step | Latent exchange adds over averaging |
| **Negative interp** | Modes are redundant / router overhead exceeds gain; MoT thesis fails on the frozen substrate | Averaging is sufficient; no coordinator needed | Mixture capacity, not diversity, is what helped | Reduces to bought compute (the p9/ex17 unrolled-depth failure) | No hardness heterogeneity in the frozen latent | Confidence == update-norm; no learned head needed | Search == unrolled depth | Debate == ensemble unrolled (extends the ex17 negative) |
| **Metrics** | Capability density (acc/FLOP, retention/byte), oracle-router gap | Density at matched total FLOPs | Density at matched param/FLOP | Accuracy at matched mean FLOPs, per-mode selection entropy | Accuracy-vs-average-FLOP frontier | Accuracy at matched mean FLOPs, halt-vs-correctness AUROC | Accuracy at matched total FLOPs | Error at matched total FLOPs |
| **Controls** | matched-compute, seed-stability, tuned single-mode baseline | matched total FLOPs | matched params AND FLOPs | matched_within (diagnostics/compute), seed sweep | matched average FLOPs, difficulty_calibration | matched mean FLOPs | matched total FLOPs incl. pruned work | matched total FLOPs, plain-ensemble arm |
| **Matched baseline** | Best single mode at same compute | Uniform blend | k-copy homogeneous MoE | Best single fixed strategy | Fixed-N refinement | Free update-norm rule | Deeper greedy chain | Single matched module + ensemble |
| **Random control** | frozen-random is VACUOUS here (probe/trained-mix); rely on shuffle + matched-compute | same | same | shuffled-router selection | shuffled-difficulty | shuffled-halt labels | shuffled branch scores | seed-permuted modules |
| **Dependency risk** | Depends on ex2 planner + e7 head existing (they do); real answer blocked on DR1 video cache | Same as MT1 | Same | HIGH: depends on MT5-MT8 producing genuinely distinct strategies | Needs a genuine hardness gradient (D3) | Low | Needs a verifier that outranks refiner | Needs decorrelated seeded modules (PR1 precondition) |
| **Custom-model decision** | Indirect (shell-lever argument) | No | No | Indirect (shell-lever) | Weak (per-sample hardness in frozen latent) | No | No | No |

---

## DR: Deliberative reasoning primitives (and the deferred prerequisite)

Every DR row's default expectation is that iteration ties depth (the p9/ex17/n9/y1 corpus negatives).
The design job is to find the one framing where it does not. DR9 (the video cache) is a prerequisite,
not a mechanism claim, and unblocks the compositional/binding/permanence modes.

### DR overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| DR1 | Real bound-attribute video cache (prerequisite) | Caching real non-additively bound video through the frozen encoder yields the first non-ceiling, non-additive test bed | studio | med | very high (unblocks everything) | DECISIVE ENABLER: without it no compositional test can bite, so no branch of the fork is justifiable |
| DR2 | Sparse-head forgetting on real latents | e7 sparse/kWTA/MoE halves forgetting vs param-matched dense on REAL latents with a significance test | studio | med | high (moves the surviving positive) | Yes: separates a head-architecture fact from a substrate fact |
| DR3 | Latent scratchpad vs residual stream | An addressable slot memory beats residual-only refinement when WM load exceeds residual width | studio | high | high (most likely to expose a substrate bound) | Yes: a pooled-latent ceiling here is a substrate-insufficiency signal |
| DR4 | Counterfactual / causal latent intervention | Do-interventions on factor directions match true counterfactuals better than a correlational predictor, low leakage | studio | high | high | Yes: entangled factors here bound the frozen encoder |
| DR5 | Cross-substrate reasoning consistency | Any surviving reasoning gain replicates on a second real encoder and beats random-init-ViT | studio | med | high (universality verdict) | Yes: classifies a gain as V-JEPA-specific, universal, or fragile |
| DR6 | Internal simulation / rollout planning | Rollout on the forward model beats reactive + action-shuffle at matched compute (ex2 extension) | cpu-now | med | high (surviving positive) | Weak: a substrate-carried dynamics signal |
| DR7 | Latent chain-of-thought without text | A supervised intermediate-latent trace beats one-shot at matched compute; shuffling hurts | studio | high | med | Weak |
| DR8 | Recurrent refinement: fixed point vs drift | Weight-tied refiner converges to an input-dependent attractor and the property is V-JEPA-specific | cpu-now | med | med | Yes: decay on real AND random-init-ViT = geometry, not substrate |
| DR9 | Verify-revise under corrected controls | A trained verifier beats single-shot, a shuffled verifier, and the free update-norm signal at matched compute | cpu-now | med | med | No |
| DR10 | Memory-first: retrieve-then-reason | kNN-retrieving cached latents and conditioning the refiner beats from-scratch and random retrieval | cpu-now | low | med | Weak: tests whether the frozen neighbor metric is task-aligned |
| DR11 | Monte-Carlo latent rollouts | Averaging many sampled rollouts beats one at matched FLOPs without chasing aleatoric noise | cpu-now | med | med | No |
| DR12 | Disagreement-as-uncertainty | Inter-module disagreement predicts error better than a single head's confidence; gating on it beats uniform, passes noisy-TV | cpu-now | med | med | No |
| DR13 | Planning-horizon limit | Rollout error compounds to a crossover horizon beyond which planning stops beating reactive | cpu-now | low | med | Weak: bounds internal simulation for the frozen substrate |
| DR14 | Reasoning under corruption/compression | Reasoning degrades more gracefully than a single-pass baseline under VQ/4-bit/noise/dropped-channel | studio | med | med | Weak: recovery beyond passive readout is a mild substrate-richness signal |
| DR15 | Modality-general reasoning | A surviving primitive's benefit is tested on video vs relational vs language encoders | studio | med | med | Yes (via convergence): identical everywhere = shell/task, not visual reasoning |

### DR full schema (part 1: DR1-DR8)

| Field | DR1 Video cache (prereq) | DR2 Sparse-real | DR3 Scratchpad | DR4 Causal intervention | DR5 Cross-substrate | DR6 Rollout planning | DR7 Latent CoT | DR8 Fixed-point |
|-------|---|---|---|---|---|---|---|---|
| **Thesis** | Real non-additively bound video cached through the frozen encoder gives the first non-ceiling, non-additive test bed, unblocking dense/slot/relational/permanence modes | e7 sparse-head forgetting advantage persists on real cached latents with a formal significance test, as a head-architecture fact independent of the substrate | Slot memory the refiner reads/writes beats residual-only where WM load exceeds residual width | Do-interventions on factor directions match true counterfactual latents better than correlational, with low cross-factor leakage | Any surviving reasoning gain replicates across a second real encoder and beats random-init-ViT | Simulating futures with the action-conditioned model and acting on the imagined endpoint beats reactive + action-shuffle at matched compute | A supervised sequence of intermediate latents improves the final answer over matched-compute one-shot; shuffling the chain hurts | The weight-tied recurrent refiner converges to an input-dependent fixed point, and the property is V-JEPA-specific not any-high-dim geometry |
| **Mechanism** | Frozen V-JEPA under no_grad emits dense tokens + pooled vectors for real clips | ContextGating/kWTA vs param-matched dense over a domain-incremental real-latent stream | K-slot attention memory over dense per-token latents | Identified factor directions perturbed, rollout compared to true counterfactual clip | Same shell run on two real encoders + random-init-ViT | Predictor action-conditioned rollout, act on endpoint | Predictor autoregressive latent rollout with supervised intermediates | IterativeRefiner.unroll, measure update-norm decay past trained horizon |
| **Substrates** | Frozen V-JEPA 2 ViT-L | Real cached V-JEPA latents | DENSE per-token V-JEPA latents | Factor-annotated bound-attribute video, real vs random-init-ViT | Two real encoders + random-init-ViT | Cached latents with action/temporal structure | Cached latents, multi-step relational task | Real V-JEPA + random-init-ViT latents |
| **Trainable** | None (encode only) | Sparse vs dense heads | Refiner + slot memory | Intervention decoder / rollout head | Shell heads (reused per substrate) | Action-conditioned predictor + planner | Autoregressive predictor | Weight-tied refiner |
| **Frozen** | Encoder | Encoder | Encoder | Encoder(s) | Encoders | Encoder | Encoder | Encoders |
| **Data/task** | Real video clips, bound attributes | Class/domain-incremental real-latent stream | Task whose WM load exceeds residual width | Factor-annotated clips | A regime where a DR gain already survives | Clips with action/temporal structure | Multi-step compositional relational task | Any regime |
| **Null hypothesis** | Even on real video the pooled and dense latents ceiling at 1.0 on held-out-combination decode (D3 says still trivially separable), so no compositional mode scores above the random-encoder floor | On real latents with a D3 certificate and a paired significance test, sparse ties param-matched dense (and ties a dense head with matched activation-sparsity penalty); the synthetic win was cluster separability | Scratchpad ties residual-only at all memory loads: the pooled frozen latent already discarded the detail external memory would store (a substrate bound) | Intervened rollouts do not beat the correlational baseline and single-factor interventions leak into other factors: the frozen latent entangles factors | The reasoning gain is identical across all substrates (a shell/task property, not substrate-carried) OR vanishes on a second real encoder (fragile) | Planning ties reactive at matched compute OR ties action-shuffle: rollouts carry no usable future info beyond one step | The latent chain ties one-shot at matched compute and shuffling does not hurt: the chain is a bag of extra layers (ex17 generalized) | Update norms do not decay geometrically and unrolling past N worsens loss: no fixed point, it is unrolled depth (the n9/y1 result) |
| **Falsification** | Held-out-combo decode ceilings at 1.0 for real AND random-encoder | Delta collapses into the 30-run seed spread or is fully explained by substrate | Scratchpad delta < seed spread at all loads | Intervened <= correlational, or leakage high | Gain identical across substrates, or gone on encoder #2 | Planning <= max(reactive, action-shuffle) | Chain <= one-shot at matched compute, shuffle harmless | No geometric decay; loss rises past N |
| **Positive interp** | The blocking test bed finally exists; every downstream compositional mode becomes scorable | e7 confirmed on real data as a head-architecture fact | External memory recovers what pooling discarded; a real substrate bound with a shell workaround | Frozen latent supports causal factoring; do-ops are meaningful | Gain is substrate-typed (V-JEPA-specific / universal / fragile) | ex2 planning replicates on real dynamics | Latent CoT is real structure, not depth | Attractor dynamics exist and are substrate-specific |
| **Negative interp** | The test bed is still ceilinged; the compositional question stays unanswerable | The synthetic e7 win was separability, not modularity | The pooled substrate is bounded and even memory cannot recover the detail (strong bound) | The frozen latent entangles factors; causal reasoning unsupported | The gain is a shell artifact or a single-encoder fluke | Rollouts are one-step video prediction | Extends the unrolled-depth negative to sequences | It is unrolled depth (confirms n9/y1) |
| **Metrics** | Held-out-combo decode, D3 certificate, ceiling check | BWT, forgetting-area, paired significance | Accuracy vs memory load at matched compute | Counterfactual match error, cross-factor leakage | Per-substrate gain, cross-substrate correlation | Planning success on true dynamics | Final-answer accuracy, shuffle delta | Update-norm decay curve, past-horizon loss |
| **Controls** | D3 difficulty calibration, random-encoder floor | D3 (regime_calibration), matched activation-sparsity dense arm, matched compute | matched compute, dense-token control | random-init-ViT, cross-factor leakage audit | random-init-ViT, cross-substrate convergence | action-shuffle, matched compute, rollout_gate | matched compute, chain-shuffle | random-init-ViT, matched depth |
| **Matched baseline** | n/a (prerequisite) | Param-matched dense (+ sparsity-penalized dense) | Residual-only refiner at matched compute | Correlational predictor | Best single-substrate gain | Flat reactive head + matched depth | Matched-compute one-shot | Deeper unrolled net |
| **Random control** | random-ENCODER (non-vacuous) | frozen_random VALID here (trained-shell dynamics) | dense-token control (frozen_random vacuous for probe) | random-init-ViT | random-init-ViT | action-shuffle | shuffled chain | random-init-ViT |
| **Dependency risk** | HIGH upstream: needs real video acquisition/curation with bound attributes | Needs DR1 for the real-latent stream | Needs DR1 (dense tokens + WM-heavy task) | Needs DR1 factor-annotated clips | Needs a surviving gain to test + a second encoder | Low (ex2 exists) | Needs DR1 multi-step task | Low |
| **Custom-model decision** | DECISIVE enabler | Yes | Yes (substrate-bound probe) | Yes (entanglement probe) | Yes (universality verdict) | Weak | Weak | Yes (geometry vs substrate) |

### DR full schema (part 2: DR9-DR15)

| Field | DR9 Verify-revise | DR10 Memory-first | DR11 MC rollouts | DR12 Disagreement | DR13 Horizon limit | DR14 Corruption | DR15 Modality-general |
|-------|---|---|---|---|---|---|---|
| **Thesis** | A trained verifier that triggers a revise beats single-shot, a shuffled/untrained verifier, and the free update-norm signal, at matched compute on a D3 regime | kNN-retrieving similar cached latents and conditioning the refiner beats from-scratch and random retrieval at matched compute | Averaging many sampled stochastic rollouts beats one deterministic rollout at matched total FLOPs, without chasing aleatoric noise | Inter-module disagreement predicts per-sample error better than a single head's confidence; gating extra compute on it beats uniform while tracking epistemic not aleatoric uncertainty | Rollout error compounds to a crossover horizon beyond which planning no longer beats reactive | Reasoning primitives degrade with a flatter accuracy-vs-corruption slope than a matched single-pass baseline under VQ/4-bit/noise/dropped-channel | A surviving primitive's benefit is tested on video vs relational vs language encoders, isolating visual-reasoning specificity |
| **Mechanism** | Verifier scores refined latent, triggers revise | ReplayBuffer.retrieve feeds IterativeRefiner | Predictor stochastic rollout, average outcomes | Multiple seeded shells; disagreement scalar gates compute | Predictor horizon sweep | quantize_dequantize / latent_robustness / dropped-channel | Same shell on three encoder families |
| **Substrates** | Cached latents, correctable-error regime | Cached latents + episodic buffer | Cached latents, stochastic-outcome task | Cached latents, seeded shells | Cached latents, action/temporal structure | Cached latents (dense for missing-channel) | Video, relational, language encoders |
| **Trainable** | Verifier + refiner | Retrieval-conditioned head | Rollout head | Seeded heads + gate | Planner | Reasoning head | Shell (reused) |
| **Frozen** | Encoder | Encoder | Encoder | Encoder | Encoder | Encoder | Encoders |
| **Data/task** | Regime with genuine correctable errors | New-class few-shot | Stochastic-outcome task | Mixed epistemic/aleatoric | Long-horizon task | Corruption sweep | Cross-modality task |
| **Null hypothesis** | Verify-revise ties single-shot at matched compute and the trained verifier ties the shuffled one: no correction-relevant signal (the ex18 result) | Retrieval ties from-scratch OR ties random retrieval: the frozen neighbor metric is not task-aligned | MC ties the matched single longer rollout (samples collapse to the mean) OR wins only by exploiting irreducible aleatoric noise (fails noisy-TV) | Disagreement ties single-head confidence AND gating on it ties uniform compute, OR it chases irreducible noise (fails noisy-TV, the e4 pattern) | Planning never beats reactive at any horizon (contradicts ex2) OR beats it only at horizon 1 (not really planning) | Reasoning and single-pass degrade at the same rate under every corruption: iteration only processes what survives | The gain is identical across all three modalities (a shell/task property) OR present only on the most separable modality (an artifact) |
| **Falsification** | verify-revise <= single-shot; trained == shuffled | Retrieval <= max(from-scratch, random-retrieval) | MC <= matched single rollout, or fails noisy-TV | Disagreement AUROC <= confidence AUROC, or fails noisy-TV | No horizon where planning > reactive, or only H=1 | Equal slopes | Identical across modalities, or single-modality only |
| **Positive interp** | The verifier carries a real correction signal (overturns ex18) | The frozen neighbor metric is task-aligned; memory-first works | Sampling explores useful stochasticity | Epistemic disagreement is a usable compute-allocation signal (where e4 failed) | Planning has a bounded useful horizon (real internal simulation) | Reasoning recovers corrupted info a readout cannot | Gain is genuinely visual-reasoning-specific |
| **Negative interp** | Confirms ex18 (no correction signal) | Frozen metric not task-aligned; memory-first unsupported | Samples collapse or chase noise | Confirms the e4 conflation | Contradicts or trivializes ex2 | Iteration recovers nothing | Shell artifact, not reasoning |
| **Metrics** | Accuracy at matched compute, trained-vs-shuffled delta | Few-shot accuracy, adaptation speed | Accuracy at matched total FLOPs | Error-prediction AUROC, adaptation-per-update | Accuracy-vs-horizon crossover | Accuracy-vs-corruption slope | Per-modality gain |
| **Controls** | shuffled/untrained verifier, free update-norm, matched compute, D3 | random retrieval, matched compute | matched total FLOPs, noisy_tv | noisy_tv, uniform-compute arm | rollout_gate, matched compute | noisy_tv, quantize/robustness controls | cross-substrate convergence, random-init-ViT |
| **Matched baseline** | Single-shot + free update-norm | From-scratch reasoning | Single longer rollout | Single-head confidence + uniform compute | Reactive at each horizon | Matched single-pass baseline | Best-modality shell |
| **Random control** | shuffled verifier | random retrieval | shuffled outcomes | disagreement-shuffled | shuffled actions | shuffled corruption | random-init-ViT per modality |
| **Dependency risk** | Low | Low | Needs a genuinely stochastic task | Needs decorrelated seeded shells (PR1) | Low | Needs dense latents for missing-channel arm | Needs three encoder families cached |
| **Custom-model decision** | No | Weak (frozen-metric quality) | No | No | Weak (bounds simulation) | Weak (substrate richness) | Yes (convergence verdict) |

---

## PR: Plasticity and learning

Doctrinal question 1 (is the shell developmentally moldable) lives here. Every biological-plasticity
signature in the corpus tied or lost its non-biological baseline, so each PR row must beat a
matched-LR-integral / matched-compute / matched-capacity control, not merely a do-nothing arm.

### PR overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| PR1 | Mode-error disjointness (router precondition) | Per-sample errors across modes are decorrelated enough that an oracle router beats the single best mode | cpu-now | low | high (cheap gate for MT1/MT4) | No (but gates the cheap-vs-studio decision) |
| PR2 | Plasticity advantage: real vs random-encoder | The plastic shell learns faster / forgets less on the special substrate than on random-init-ViT features | cpu-now | med | high | DECISIVE: tests if the +0.31 substrate structure eases LEARNING, not just readout |
| PR3 | Modular plasticity on real latents (= DR2) | Sparse/modular heads halve forgetting vs param-matched dense on real cached latents, significance-tested | cpu-now/studio | med | high (surviving positive) | Yes (head-arch vs substrate) |
| PR4 | Epistemic gate vs noisy-TV | Ensemble-disagreement uncertainty steers plasticity toward reducible structure and ignores aleatoric noise (where e4 failed 30/30) | cpu-now | med | high (retries the strongest negative) | No |
| PR5 | Content-gated critical period | A surprise-triggered plasticity-reopening schedule beats a clock-gated schedule at matched LR-integral (where e3/d6 lost) | cpu-now | med | high (doctrinal Q1) | Weak |
| PR6 | Offline sleep consolidation | Wake/sleep phase separation beats interleaved learning at matched total gradient steps | cpu-now | med | med | No |
| PR7 | Fast/slow two-timescale weights | A Hebbian fast-weight adapter + slow SGD head adapts faster within task at matched capacity and retains after decay | cpu-now | med | med | No (supplies a System-1 mode to MoT) |
| PR8 | Memory-augmented retrieval head | A retrieval-conditioned head over the frozen-latent KV buffer beats plain kNN AND a matched-param parametric head on few-shot | cpu-now | low | med | Weak (frozen keys never go stale = doctrinal free lunch) |

### PR full schema (part 1: PR1-PR4)

| Field | PR1 Error-disjointness | PR2 Real-vs-random-encoder | PR3 Modular-real (=DR2) | PR4 Epistemic-gate |
|-------|---|---|---|---|
| **Thesis** | Per-sample errors of reasoning modes are decorrelated enough that an oracle router would beat the single best mode, the cheap precondition for MT1/MT4 | The plastic shell learns a continual stream faster and forgets less on the special frozen substrate than on random-ENCODER features, testing whether the +0.31 nuisance-invariance makes the LEARNING problem easier, not just the readout | Sparse/modular (kWTA/MoE) heads halve catastrophic forgetting vs param-matched dense on real cached latents via a metric (BWT/forgetting) that survives the vacuous frozen-random projection | Ensemble-disagreement (epistemic) uncertainty steers plasticity and query selection toward reducible structure and IGNORES aleatoric noise, succeeding where e4 point-error gating failed 30/30 |
| **Mechanism** | Run reactive/refine/plan/verify heads, compute per-sample loss vectors, correlation + oracle-selection upper bound; no new module | Identical plastic shell + continual stream on real vs random-init-ViT features at matched 256px | Shell heads swept dense/kWTA/MoE at matched params over a class/domain-incremental stream | Small ensemble sources disagreement feeding Neuromodulation.gate + PlasticityController; scored on reducible-vs-noise LR-integral allocation |
| **Substrates** | Cached latents | Real V-JEPA vs random-init-ViT at 256px | Real cached V-JEPA latents | Cached latents with reducible + irreducible-noise partitions |
| **Trainable** | None (diagnostic) | Plastic shell | Sparse/dense heads | Ensemble + gate |
| **Frozen** | Encoder | Encoder(s) | Encoder | Encoder |
| **Data/task** | Cached-latent regime | Continual stream | Class/domain-incremental stream | Reducible/noise split (noisy-TV) |
| **Null hypothesis** | Per-sample errors are highly correlated (oracle-router gain over single-best is within seed spread), so no router can help and MT1/MT4 are not worth building | Shell adaptation speed and BWT on real V-JEPA are within the seed spread of the same shell on random-init-ViT (matched resolution): the substrate's structure does not ease trained-shell learning dynamics | On real latents with a D3 certificate and a paired significance test, sparse ties param-matched dense (and a matched activation-sparsity dense head): synthetic advantage was cluster separability | On the noisy-TV split the disagreement gate allocates no more LR-integral (or query budget) to the reducible partition than an ungated arm: replicates the e4 epistemic/aleatoric conflation |
| **Falsification** | Oracle gain over best mode < seed spread | Real-vs-random-encoder adaptation/BWT delta < seed spread | Delta collapses into 30-run spread | Reducible-vs-noise LR-integral allocation delta <= 0 |
| **Positive interp** | Modes are complementary; a router can help (green-light MT) | The special substrate genuinely eases learning, not just readout: strongest reason to keep the frozen encoder | e7 confirmed on real data as a head-architecture fact | Epistemic uncertainty is separable and usable (overturns e4) |
| **Negative interp** | Modes redundant; MoT router unlikely to help; stop before studio spend | The +0.31 is readout-only; the shell's learning problem is substrate-agnostic | Synthetic win was separability | Confirms e4: shell cannot separate reducible from irreducible |
| **Metrics** | Per-mode loss correlation, oracle-selection upper bound | Adaptation speed, BWT | BWT, forgetting-area, paired significance | Reducible-vs-noise LR-integral, noisy-TV pass |
| **Controls** | seed spread | matched resolution, matched shell, seed sweep | D3, matched activation-sparsity dense arm, matched compute | noisy_tv, ungated arm, curriculum-permutation |
| **Matched baseline** | Single best mode | Same shell on random-init-ViT | Param-matched dense (+ sparsity-penalized) | Ungated plasticity |
| **Random control** | n/a (diagnostic) | random-ENCODER (non-vacuous) | frozen_random VALID (trained-shell metric) | shuffled-disagreement |
| **Dependency risk** | Low (pure diagnostic); gates MT | Blocked on random-init-ViT features (do NOT run concurrently with in-flight ViT job) | Needs real-latent stream (DR1 or existing cache) | Needs a clean reducible/noise partition |
| **Custom-model decision** | No | DECISIVE (learning-eases test) | Yes | No |

### PR full schema (part 2: PR5-PR8)

| Field | PR5 Content-gated critical period | PR6 Sleep consolidation | PR7 Fast/slow weights | PR8 Memory-augmented retrieval |
|-------|---|---|---|---|
| **Thesis** | A content-gated (surprise-triggered) plasticity-reopening schedule protects early tasks and reopens for novel ones better than a clock-gated schedule at matched LR-integral, beating cosine decay where e3 lost | Separating a wake phase (encode-to-buffer) from an offline sleep phase (replay-only steps + EWC refresh) beats mixing them at matched total gradient steps | A two-timescale system (Hebbian fast-weight adapter + slow SGD head) adapts faster within task at matched capacity and retains after the fast store decays | A retrieval-conditioned head over the frozen-latent KV buffer beats plain kNN AND a matched-param parametric head on few-shot, exploiting that frozen-encoder keys never go stale |
| **Mechanism** | PlasticityController hard/soft/learned + surprise-triggered reopening | ReplayBuffer wake/sleep loop vs interleaved | WorkingMemory-style fast store + slow head | ReplayBuffer.retrieve conditioning a head |
| **Substrates** | Cached latents, incremental stream | Cached latents | Cached latents | Cached latents |
| **Trainable** | Plasticity schedule + heads | Shell + buffer | Fast + slow weights | Retrieval-conditioned head |
| **Frozen** | Encoder | Encoder | Encoder | Encoder |
| **Data/task** | Incremental stream | Continual stream | Within-task adaptation | New-class few-shot |
| **Null hypothesis** | At matched LR-integral and shuffled task order, content-gated shows no retention or reopening advantage over cosine decay / tuned constant LR: pure LR annealing (the e3/d6 negative) | Offline replay-only ties online replay at matched total gradient steps: any sleep benefit is added compute, not phase separation | Fast weights tie a slow-only head (and a matched-size replay buffer) on within-task adaptation: the fast store is redundant capacity or a small cache | Retrieval-conditioned head ties plain kNN (parametric part adds nothing) OR ties the no-memory parametric head (memory adds nothing), at matched capacity |
| **Falsification** | Retention/reopening delta < seed spread at matched LR-integral | Sleep delta <= 0 at matched steps | Fast-slow delta <= 0 at matched capacity | Head ties kNN or ties parametric |
| **Positive interp** | Content-gating is a real critical-period mechanism (doctrinal Q1 lead) | Phase separation genuinely helps consolidation | Two timescales add a real System-1 mode | The frozen KV buffer is a genuine free lunch |
| **Negative interp** | Confirms e3/d6: schedule is cosmetic LR annealing | Sleep is just more compute | Fast store redundant | Memory or parametric part is redundant |
| **Metrics** | BWT (early), FWT (novel), matched-LR-integral | BWT, frontier AUC | Within-task adaptation curve, post-decay retention | Few-shot accuracy, adaptation speed |
| **Controls** | matched LR-integral, shuffled order, cosine/constant baselines, seed sign-flips | matched total gradient steps | matched capacity, matched-size buffer | matched capacity, kNN + parametric arms |
| **Matched baseline** | Cosine decay / tuned constant LR | Interleaved online replay | Slow-only head | kNN + parametric head |
| **Random control** | shuffled task order | n/a | n/a | random-key retrieval |
| **Dependency risk** | Low | Low | Low | Low |
| **Custom-model decision** | Weak (moldability signal) | No | No | Weak |

---

## WS: Global workspace / arbitration layer

Presumed decoration (a matched-capacity MLP wins, agreement ties confidence) until WS1 shows
cross-substrate agreement carries real correctness information. WS1 gates the rest.

### WS overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| WS1 | Agreement vs confidence (central gate) | Cross-substrate agreement (V-JEPA + DINOv2) predicts correctness better than the best single substrate's confidence | cpu-now | med | high (gates all WS) | Yes: a positive argues two frozen encoders beat one, reweighting the atlas branch |
| WS2 | Matched-capacity fusion tournament | At matched params/FLOPs, some structured fusion beats the unstructured concat-MLP floor | cpu-now | med | high | Weak |
| WS3 | Uncertainty vs disagreement arbitration | Precision-weighted fusion and disagreement-driven allocation each beat equal-weight averaging, capacity-neutral | cpu-now | med | high (payoff of positive WS1) | No |
| WS4 | Broadcast-bottleneck bandwidth sweep | A narrow shared GWT slot with broadcast beats matched-capacity fusion AND generic regularization at fixed capacity | cpu-now | med | med | No |
| WS5 | Modular router + shared slot (= extends DR2/PR3) | Adding a shared broadcast slot on top of e7 sparse routing yields a BWT advantage beyond routing alone, slot-ablated | cpu-now | med | med | Weak |

### WS full schema

| Field | WS1 Agreement-vs-confidence | WS2 Fusion tournament | WS3 Uncertainty-vs-disagreement | WS4 Bottleneck sweep | WS5 Router + slot |
|-------|---|---|---|---|---|
| **Thesis** | Cross-substrate agreement between two genuinely different frozen encoders (V-JEPA + DINOv2) predicts per-input correctness better than the best single substrate's calibrated confidence (risk-coverage dominance, higher correct/incorrect AUROC) on a D3 non-ceiling task | At strictly matched trainable-param count and training FLOPs, at least one structured fusion (learned-linear, cross-source attention, GWT broadcast) beats the unstructured concat-MLP floor on held-out accuracy or calibration | Given a positive WS1, precision-weighted (inverse-variance) fusion and disagreement-driven compute allocation each beat equal-weight averaging, concat-MLP, single-best-source, and a disagreement-shuffled router, adding essentially zero params | A narrow shared global-workspace slot with broadcast shows a bandwidth benefit (narrowness helps generalization) at fixed total capacity, beating unbottlenecked matched-capacity fusion and generic regularization tuned to the same effective bandwidth | Adding a shared broadcast slot on top of e7 kWTA/MoE sparse routing yields a forgetting/BWT advantage beyond sparse routing alone, isolated by ablating the slot at fixed routing and capacity, on real latents |
| **Mechanism** | Two encoders' pooled latents; agreement scalar vs single-source calibrated confidence as correctness predictors | All fusion arms at identical param/FLOP budget | Fixed Bayesian inverse-variance rule + disagreement router, no added capacity | Workspace slot width swept while total params held constant by widening elsewhere | e7 30-run protocol + shared slot + slot-ablation |
| **Substrates** | V-JEPA + DINOv2 (one extra cached encode pass) | Same two encoders | Same two encoders | Same two encoders | Real cached V-JEPA latents |
| **Trainable** | Agreement/confidence readouts | All fusion heads | None beyond WS1 (fixed rule) | Fusion + slot | Sparse heads + slot |
| **Frozen** | Both encoders | Both | Both | Both | Encoder |
| **Data/task** | D3 nuisance-invariant shape task | Same | Same | Same | Domain-incremental real-latent stream |
| **Null hypothesis** | Agreement AUROC minus single-substrate-confidence AUROC <= 0 within seed CI; OR the win fails the invertible-remap vacuity guard (B is a linear remap of A), the shuffled-cross-source control, or noisy-TV | At matched capacity every structured fusion ties the concat-MLP on held-out accuracy and NLL: structure buys nothing beyond param count | Precision weights are uninformative so inverse-variance ties equal-weight averaging, and disagreement is uncorrelated with correctness so disagreement routing ties random routing at matched rate; OR disagreement fires on aleatoric noise (fails noisy-TV) | The bottleneck's benefit is indistinguishable from generic regularization at matched capacity, and broadcast-back adds nothing over write-only: narrow-vs-wide is just a capacity effect | The shared slot adds nothing over sparse routing alone: slot-ablation ties the full model on BWT and accuracy at matched capacity; sparse routing does all the work |
| **Falsification** | Agreement AUROC <= confidence AUROC in CI, or fails remap/shuffle/noisy-TV | Every structured arm ties concat-MLP | Inverse-variance ties averaging AND disagreement ties random routing | Bottleneck ties regularization at matched capacity | Slot-ablation ties full model |
| **Positive interp** | Two frozen encoders carry complementary correctness info; the workspace is not decoration; keep multiple frozen encoders | Fusion structure is more than capacity | Arbitration is a capacity-neutral win (the cleanest possible) | GWT narrowness genuinely helps (a specific GWT claim) | The workspace slot adds over proven routing |
| **Negative interp** | One substrate's confidence suffices; workspace is decoration | Structure is decoration (mirrors the corpus negatives) | Precision/disagreement uninformative (extends e4) | It is a capacity effect (pre-registered null, corpus vol2 line 975) | Sparse routing does all the work |
| **Metrics** | Correct/incorrect AUROC, risk-coverage curve | Held-out accuracy, NLL at matched capacity | Speed-accuracy frontier, AUROC | Generalization vs slot width at fixed capacity | BWT, accuracy at matched capacity |
| **Controls** | invertible-remap guard, shuffled-cross-source, noisy-TV, seed CI | matched params AND FLOPs | noisy-TV, disagreement-shuffled router, single-best-source | matched capacity, dropout/weight-decay tuned to same bandwidth | slot-ablation, matched params to dense + slot-ablated |
| **Matched baseline** | Best single-substrate confidence | Concat-MLP floor | Equal-weight averaging + concat-MLP | Unbottlenecked matched-capacity fusion | Slot-ablated sparse routing |
| **Random control** | shuffled-cross-source; invertible-remap guard replaces vacuous frozen-random | n/a (matched-capacity is the control) | disagreement-shuffled router | n/a | frozen_random VALID (trained-shell metric) |
| **Dependency risk** | Low (second encoder is one cached encode pass); gates WS2-WS5 | Depends on WS1 substrate cache | Depends on positive WS1 | Depends on WS1 cache | Depends on real-latent stream + e7 protocol |
| **Custom-model decision** | Yes (multi-encoder value) | Weak | No | No | Weak |

---

## AT: Atlas and substrate typing (the non-vacuous controls)

These rows carry the CRITICAL correction: frozen_random_projection is a full-rank invertible matrix
and VACUOUS for probes. The valid substrate control is real-vs-random-ENCODER (random-init same-arch
ViT-L at matched resolution). AT typing is what actually reweights the frozen-dense-vs-custom fork.

### AT overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| AT1 | Cross-substrate nuisance grid | The 0.379-vs-0.069 nuisance regime across V-JEPA/DINOv2/single-frame-V-JEPA vs each's random-init control classifies the substrate signal as universal / modality-specific / architecture-specific / artifact | studio | med | high | DECISIVE: directly types the substrate signal that reopened the fork |
| AT2 | Mode substrate-dependence | At least one thinking mode's advantage survives the non-vacuous control: real V-JEPA beats random-init-ViT at matched 256px on nuisance content | studio | med | high | Yes: ties a mode's gain to pretraining, not architecture/resolution |
| AT3 | Time-axis ablation | Some factors decode from full-clip but at chance from single-frame V-JEPA (token/frame matched), isolating the temporal currency | cpu-now | med | med | Weak (modality currency, not the fork directly) |
| AT4 | Programmatic ceiling reference | A handcrafted-descriptor substrate on the same atlas factors provides a ceiling so a perceptual tie reads as substrate-bounded, not test-too-easy | cpu-now | med | high | DECISIVE: the only route to reading a tie as a real substrate bound (custom-branch trigger) |
| AT5 | Probe-class sweep | Running every atlas cell under linear/MLP/nonlinear-gain surfaces probe-specific verdicts before any cell is trusted | cpu-now | low | med | Yes (methodological guard on every verdict) |

### AT full schema

| Field | AT1 Nuisance grid | AT2 Mode substrate-dep | AT3 Time-axis | AT4 Programmatic ceiling | AT5 Probe-class sweep |
|-------|---|---|---|---|---|
| **Thesis** | Nuisance-invariant shape identity (0.379-vs-0.069 regime) decoded from V-JEPA, DINOv2, and time-stripped single-frame V-JEPA, each vs its OWN random-init control at matched resolution, classifies as substrate-universal, modality-specific, architecture-specific, or random-control-artifact, reweighting the frozen-encoder fork | At least one thinking mode's advantage is reproduced by real V-JEPA but NOT by random-init same-arch ViT-L (matched 256px) on nuisance-heavy content | Some factors are decodable from full-clip V-JEPA but at chance from single-frame V-JEPA (needs-time factors) while others are static, with token/frame count matched so a time advantage is not a raw-info advantage | Running a programmatic/handcrafted-descriptor substrate on the same atlas factors as a difficulty-calibration upper bound lets a perceptual tie be read as substrate-bounded rather than test-too-easy | Running every atlas cell under >=2 probe classes surfaces probe-specific verdicts (present under one, absent under another), the failure the frozen-random census already caught |
| **Mechanism** | Same nuisance clips through each substrate + its random-init column | Run a mode on real vs random-init-ViT features | Full-clip vs single-frame with token/frame matched | Handcrafted (optical-flow/HOG/color-hist) reference on the same factors | Linear vs MLP vs nonlinear-gain per cell |
| **Substrates** | V-JEPA, DINOv2, single-frame V-JEPA + each's random-init | Real V-JEPA vs random-init-ViT | V-JEPA full-clip vs single-frame | Programmatic descriptors + perceptual substrates | All AT1 substrates + controls |
| **Trainable** | Probes | Mode head + probe | Probes | Probes | Probes |
| **Frozen** | Encoders | Encoders | Encoder | n/a (handcrafted) + encoders | Encoders |
| **Data/task** | 6-class nuisance shape clips (position/scale/rotation/color/clutter/motion) | Nuisance content | Atlas factors | Non-ceiling atlas factors | All cells |
| **Null hypothesis** | Every substrate's decodability delta over its matched-resolution random-init control is within seed spread (random-control-artifact): pretraining bought no nuisance invariance beyond architecture and resolution in any substrate | Every mode's advantage is reproduced by random-init-ViT (or random-encoder), so no mode depends on the pretrained substrate rather than architecture/resolution or generic separability | With token/frame count equalized, full-clip and single-frame decodability are within seed spread for every factor: no factor genuinely requires the temporal axis | The programmatic reference does not exceed the perceptual substrates on any atlas factor (trivially separable everywhere), so no perceptual tie can be attributed to a substrate bound | Every cell's verdict is invariant to probe class, so probe choice never changes decodability and probe-specific is an empty verdict |
| **Falsification** | All deltas < seed spread over random-init | Every mode gain reproduced by random-init-ViT | Full-clip == single-frame for all factors | Programmatic never exceeds perceptual | Verdicts invariant to probe class |
| **Positive interp** | The substrate signal is typed (the whole point); it reweights the fork | A mode is genuinely substrate-carried | A temporal currency exists | A tie now reads as a real substrate bound (justifies custom) | Probe-specific verdicts caught before being trusted |
| **Negative interp** | The +0.31 was architecture/resolution, not pretraining: weakens keep-frozen | Modes are architecture/resolution effects | No factor needs time | No perceptual tie can be blamed on the substrate (blocks custom) | Probe choice never matters |
| **Metrics** | Per-substrate decodability delta over random-init | Mode gain: real vs random-init-ViT | Full-clip minus single-frame decode, token/frame matched | Programmatic vs perceptual per factor | Per-cell verdict across probe classes |
| **Controls** | random-init same-arch per substrate at matched resolution | random-init-ViT at 256px | token/frame count matched | difficulty_calibration, programmatic upper bound | linear/MLP/nonlinear-gain |
| **Matched baseline** | Random-init same-arch per substrate | Random-init-ViT | Single-frame (matched tokens) | Perceptual substrates | Cross-probe |
| **Random control** | random-init same-arch (non-vacuous) | random-init-ViT (non-vacuous) | n/a | n/a | n/a |
| **Dependency risk** | Blocked on random-init-ViT encode; do NOT run concurrently with the in-flight ViT job (OOM) | Same OOM caution; needs a mode with a candidate gain | Low (single-frame is cheap) | Low | Low |
| **Custom-model decision** | DECISIVE | Yes | Weak | DECISIVE (bound-vs-ceiling) | Yes (guard) |

---

## AL: Cross-substrate / cross-modal alignment and the uncertainty router

### AL overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| AL1 | Uncertainty router with noisy-TV guard | An uncertainty signal used ONLY as a router input (never a LR gate) beats random episode selection on adaptation-per-update while passing noisy-TV | cpu-now | med | high | No |
| AL2 | Shared-latent alignment vs random-map floor | A thin linear/low-rank map between substrate pairs predicts one from another above a random-map-of-equal-rank floor only when they share genuine structure | studio | med | med | Yes: separates real cross-currency exchange from map-induced agreement |
| AL3 | Audio/video temporal alignment | Audio-SSL and V-JEPA event boundaries co-locate above a shuffled-time floor, evidence temporal-predictive invariance is substrate-universal across modality | studio | high | med | Yes (universal verdict via cross-modal convergence) |

### AL full schema

| Field | AL1 Uncertainty router | AL2 Shared-latent alignment | AL3 Audio/video alignment |
|-------|---|---|---|
| **Thesis** | An uncertainty signal used ONLY as a router input (never as a shell learning-rate gate) selects which episodes get deliberative compute and beats random episode selection on adaptation-per-update, while passing the noisy-TV guard | A thin linear/low-rank map between each substrate pair on shared clips predicts one substrate's latents from another above a random-map-of-equal-rank floor only when the substrates share genuine structure | Audio-SSL and V-JEPA event boundaries co-locating above a shuffled-time floor on the same clips is evidence temporal-predictive invariance is substrate-universal across modality, not a V-JEPA-visual peculiarity |
| **Mechanism** | Measured uncertainty scalar feeds a router (explicitly NOT an LR gate, since e4 refuted that 30/30) | Learned map vs random-map-of-equal-rank | Event-boundary co-location vs shuffled-time |
| **Substrates** | Cached V-JEPA latents | V-JEPA, DINOv2 (+ VideoMAE if available) | V-JEPA + audio/speech SSL |
| **Trainable** | Router | Linear/low-rank map | Boundary detectors |
| **Frozen** | Encoder | Encoders | Encoders |
| **Data/task** | Cached-latent regime + noisy-TV + curriculum-permutation | Shared clips | Shared audio-video clips |
| **Null hypothesis** | The uncertainty router matches random episode selection on adaptation-per-update, OR it chases irreducible aleatoric noise (fails the noisy-TV guard), replicating the e4 negative in router clothing | A random map of equal rank predicts the target as well as the learned map (alignment-artifact): any apparent cross-substrate agreement is the map's capacity, not shared structure | Audio-SSL and V-JEPA event boundaries co-locate no better than under a shuffled-time permutation: temporal-predictive structure is modality-specific, not universal |
| **Falsification** | Router == random selection, or fails noisy-TV | Learned map == random-map-of-equal-rank | Co-location == shuffled-time |
| **Positive interp** | Uncertainty as a router input works even though as an LR gate it failed (a clean mechanism split) | Substrates share real structure (cross-currency exchange exists) | Temporal invariance is substrate-universal across modality (strongest universal evidence) |
| **Negative interp** | Repeats e4 in router clothing | Agreement is map capacity, not shared structure | Temporal structure is modality-specific |
| **Metrics** | Adaptation-per-update, noisy-TV pass | Prediction R^2 over random-map floor | Boundary co-location over shuffled floor |
| **Controls** | noisy-TV, curriculum-permutation, random-selection arm | random-map-of-equal-rank | shuffled-time |
| **Matched baseline** | Random episode selection | Random map of equal rank | Shuffled-time |
| **Random control** | random selection | random-map-of-equal-rank (non-vacuous alignment control) | shuffled-time |
| **Dependency risk** | Low | Needs a second (and third) encoder cache | Needs an audio SSL encoder + aligned audio-video clips |
| **Custom-model decision** | No | Yes | Yes (universal verdict) |

---

## CM: Custom-model pathway (justify or close the custom branch)

Every CM row exists to decide the fork. CM-gate rows (CM1-CM4) are cached-latent tests that would
JUSTIFY building a custom encoder; the pilots (CM5-CM11) actually build one and are gated behind the
gates plus the in-flight controls. CM7 (the minimum-objective probe) is the ONLY custom-TRAINING
experiment sanctioned to run before a bounding result, because it is a diagnostic of whether objective
is a lever at all, not an attempt to beat frozen V-JEPA.

### CM overview table

| ID | Name | Thesis tested (one line) | Tier | Diff | Sci value | Custom-model decision |
|----|------|--------------------------|------|------|-----------|-----------------------|
| CM1 | Compositional bound on real video (C1 gate) | Frozen V-JEPA factors shape from color on held-out combos off-ceiling, beating random-init-ViT by >+0.05 on a D3 regime | studio | med | very high | DECISIVE: the primary keep-frozen gate |
| CM2 | Multi-substrate atlas gate | A failing gate either clears on a different frozen substrate (bound is objective/arch-specific) or fails on all (universal) | studio | med | high | DECISIVE: swap-encoder vs go-custom |
| CM3 | Dense vs pooled compositional | Dense/coarse-grid latents clear the held-out gate where full mean-pool does not, isolating the bound to the pooling interface | studio | med | high | Yes: pooling-interface fix vs new weights |
| CM4 | Workspace-shell routing | A trained workspace routing shell beats param-matched dense AND matched-compute unrolled depth on BWT and planning, on real latents | studio | high | high | Yes: a shell win argues against needing a custom encoder |
| CM5 | Studio-scale rejuvenation (C3) | At studio scale rejuvenation restores plasticity beyond frozen-random and matched compute; a discriminating arm shows if it needs an adapting substrate | studio | high | med | Yes (C3): if restoration needs an adapting substrate, the frozen encoder is bounded for Q1 |
| CM6 | Compressed capability density | A distilled ViT-S student retains the teacher's nuisance-invariance delta at ~10x fewer params, beating same-size random-init and non-distilled | studio | med | med | Yes: if density transfers, a small student can replace the frozen giant |
| CM7 | Minimum-objective encoder probe | At tiny capacity/matched data/256px, an invariance objective beats a mask objective and both beat random-init | studio | high | high | DECISIVE diagnostic: is the pretraining OBJECTIVE a lever at all before any real-encoder compute |
| CM8 | Custom-JEPA factoring pilot | A small custom JEPA (or factoring adapter) with a bound-factor mask policy beats the best frozen atlas substrate on held-out factoring | wider-box | high | high | DECISIVE (build side): does a custom objective actually beat the best frozen encoder |
| CM9 | Object-centric slot-JEPA binding | Trainable slot-attention over frozen dense tokens beats dense-without-slots and an off-the-shelf slot model on multi-object binding | wider-box | high | med | Yes: slots-as-shell vs custom encoder |
| CM10 | Action-conditioned forward model | A trainable action-conditioned model enables planning beating reactive, action-shuffle, unrolled depth, and frozen V-JEPA 2-AC | wider-box | high | med | Weak (blocked on an env adapter, not compute) |
| CM11 | Developmental scheduled plasticity | A scheduled-plasticity substrate shows critical-window (d6 True), path-dependence (y4>0), U-shape (d4 non-flat), seed-stable, beating flat-schedule | wider-box | very high | med | Yes: the one architecture that genuinely needs an adapting substrate |
| CM12 | Mixture-of-thinking substrate (capstone) | A trained workspace router over 2-4 frozen/pilot experts beats the best single expert+workspace and the strongest open model on a combined battery | moonshot | very high | high | Yes: the full synthesis of the fork |

### CM full schema (part 1: CM1-CM6, the cached-latent gates)

| Field | CM1 C1 gate | CM2 Atlas gate | CM3 Dense-vs-pooled | CM4 Workspace shell | CM5 Rejuvenation C3 | CM6 Distilled density |
|-------|---|---|---|---|---|---|
| **Thesis** | On real non-additively bound video, frozen V-JEPA factors shape from color on held-out combinations off-ceiling, beating random-init same-arch ViT at matched resolution by >+0.05 on a D3-certified regime | A failing compositional/adaptation gate either clears on a different frozen substrate (bound is objective/architecture-specific, adopt a better encoder) or fails on all (bound is universal) | Dense or coarse-grid-pooled (2x2/4x4) latents clear the held-out-combination gate where full mean-pool does not, isolating the bound to the pooling interface not the weights | A trained global-workspace routing shell over frozen features beats a param-matched dense shell AND matched-compute unrolled depth on e7 BWT and ex2 planning, across seeds, on real latents | At studio scale (dim 256 to low thousands, thousands of tasks) rejuvenation restores plasticity beyond frozen-random and matched compute; a discriminating arm shows whether restoration requires an adapting substrate (C3) | A small distilled student retains the teacher's nuisance-invariance delta (+0.21 to +0.31 off-ceiling) at ~10x fewer params, beating same-size random-init and same-size non-distilled |
| **Null hypothesis** | Held-out-combination accuracy for real V-JEPA ties random-init same-arch ViT (delta < +0.05) on a D3-certified separable regime: the substrate memorizes conjunctions rather than factoring them | All frozen substrates tie their own random-init controls on the failing gate (no substrate exceeds +0.15 off-ceiling where V-JEPA-L did not): no substrate swap helps | Dense and coarse-grid tie full mean-pool on held-out combos (delta < +0.1 off-ceiling on a D3 regime): the pooling interface is not the bound (single-factor pooling already refuted, orientation decodes at 1.0 pooled) | The workspace shell ties param-matched dense OR ties matched-compute unrolled depth on BWT and planning, or sign-flips across seeds: the win is capacity/depth, not routing | Plasticity loss never appears at reachable scale, OR rejuvenation ties doing-nothing, OR frozen-substrate restoration ties the adapting-substrate arm (no C3 signal) | A same-size random-init substrate OR a same-size student trained without the invariance-matching loss ties the distilled student on the nuisance-invariance delta: compression bought no capability density |
| **Falsification** | Real vs random-init-ViT delta < +0.05 | No substrate exceeds +0.15 where V-JEPA-L did not | Dense/coarse vs pooled delta < +0.1 | Ties dense OR unrolled depth, or sign-flips | No loss at scale, or rejuv==nothing, or frozen==adapting | Distilled ties random-init or non-distilled |
| **Positive interp** | The frozen substrate factors compositionally: strongest reason to keep it | The bound is fixable by a better frozen encoder (swap, do not build) | The bound is the pooling interface, fixable in the shell (keep the encoder) | A shell beats depth: routing is the lever, not a new encoder | Restoration works; if it needs an adapting substrate, the frozen encoder is Q1-bounded | A small student can replace the frozen giant at 10x density |
| **Negative interp** | Substrate memorizes conjunctions: a real compositional bound (custom-branch trigger) | No swap helps: the bound is universal (custom-branch trigger) | Not the pooling interface: the bound is the weights | Capacity/depth, not routing: workspace is decoration | No moldability signal even at scale (confirms Q1 negatives) | Compression is generic, no density transfer |
| **Metrics** | Held-out-combo accuracy, D3 certificate, off-ceiling delta | Per-substrate off-ceiling delta over random-init | Dense/coarse vs pooled off-ceiling delta | BWT, planning gain, seed sign-flips | Adaptation-speed restoration net of retention cost | Invariance-per-parameter |
| **Controls** | random-init same-arch ViT (NOT the full-rank projection), D3 | per-substrate random-init controls, difficulty_calibration | D3, pooled/dense/coarse arms | param-matched dense, matched-FLOP unrolled depth, seed sweep | frozen-random, matched compute, adapting-vs-frozen arm | same-size random-init, same-size non-distilled |
| **Matched baseline** | Random-init same-arch ViT | Best frozen atlas substrate | Full mean-pool | Param-matched dense + unrolled depth | Do-nothing + matched compute | Same-size non-distilled student |
| **Random control** | random-init same-arch (non-vacuous) | per-substrate random-init | dense-token (frozen_random vacuous for probe) | frozen_random VALID (trained-shell metric) | frozen-random + matched compute | same-size random-init |
| **Tier / Diff** | studio / med | studio / med | studio / med | studio / high | studio / high | studio / med |
| **Dependency risk** | Blocked on DR1 (real bound-attribute video) | Needs multiple frozen substrates + real video | Needs DR1 dense tokens | Needs real-latent stream + e7/ex2 protocols | Extends ex15/b8; needs a wide box at studio scale | Needs the teacher cache + a trainable student |
| **Custom-model decision** | DECISIVE (keep-frozen gate) | DECISIVE (swap vs build) | Yes (interface vs weights) | Yes (shell vs encoder) | Yes (C3, Q1 bound) | Yes (density) |

### CM full schema (part 2: CM7-CM12, the training pilots)

| Field | CM7 Min-objective probe | CM8 Custom-JEPA factoring | CM9 Slot-JEPA binding | CM10 Action forward model | CM11 Developmental plasticity | CM12 MoT substrate (capstone) |
|-------|---|---|---|---|---|---|
| **Thesis** | At tiny capacity (~1-5M), matched nuisance-clip data, matched 256px, the SSL OBJECTIVE (mask vs invariance/contrastive) is a live lever: an invariance objective beats a mask objective and both beat random-init same-arch by > seed spread. A diagnostic of the custom-encoder DECISION, not an attempt to beat frozen V-JEPA | A small custom JEPA (or frozen backbone + trainable factoring adapter) trained with a mask policy targeting the bound factor beats the best frozen atlas substrate on held-out compositional factoring off-ceiling at matched shell compute, seed-stable | A trainable slot-attention module over frozen dense V-JEPA tokens beats dense-tokens-without-slots and an off-the-shelf slot model on multi-object held-out binding and passes a binding-specificity swap test at matched compute | A trainable action-conditioned forward model on frozen latents enables planning beating reactive, action-shuffle, matched-compute unrolled depth, and frozen V-JEPA 2-AC | A small substrate (or adapter) trained with a scheduled-plasticity developmental curriculum shows critical/sensitive-window learning (d6 True), path-dependence (y4>0), U-shaped overgeneralization (d4 non-flat), all seed-stable, beating a flat-schedule baseline at matched compute | A trained workspace router composing 2-4 frozen/pilot substrate experts beats the best single expert+workspace, the single pilot, and the strongest open model on a combined compositional-plus-adaptation battery at matched total compute, seed-stable |
| **Null hypothesis** | At matched tiny capacity, matched data, matched 256px, both custom objectives tie random-init same-arch AND tie each other; objective is not a lever at this scale and the +0.31 was scale/data/architecture/resolution. A tie is a strong negative closing the custom-encoder line | The custom-JEPA pilot ties both a random-init same-arch ViT and the best frozen atlas substrate on held-out combinations at matched compute: the custom objective bought nothing | Dense tokens without slots tie the slotted model on multi-object held-out combinations, OR the binding-specificity swap shows non-target slots change too: slots add nothing over dense | Action-shuffle ties the true-action forward model on planning success: the model ignores actions and is a plain video predictor | A non-developmental substrate trained on identical data with a flat schedule ties the developmental one on every plasticity signature (window, path-dependence, U-shape): the development is cosmetic | A single best-expert-plus-workspace ties the full mixture at matched total FLOPs: the mixture is pure overhead and should be descoped to the single expert |
| **Falsification** | Both objectives tie random-init AND each other | Ties random-init AND best frozen substrate | Ties dense OR swap-test leaks | Action-shuffle ties true-action | Flat-schedule ties developmental on all signatures | Single-expert+workspace ties the mixture |
| **Positive interp** | Objective is a lever; a custom encoder could help; justifies scoping CM8 | A custom objective beats the best frozen encoder: BUILD | Slots add over dense: object-centric shell helps | Real action-conditioned planning (extends ex2 to actions) | Genuine moldability at last (answers Q1 positively) | The full MoT synthesis wins |
| **Negative interp** | Objective is not a lever; the +0.31 is scale/data/arch/resolution: CLOSES the custom-encoder line | Custom objective bought nothing: do not build | Slots are decoration | The model ignores actions | Development is cosmetic (confirms d4/d6/y4 negatives) | Mixture is overhead; descope to one expert |
| **Metrics** | Nuisance-invariance + held-out-combo per objective | Held-out factoring off-ceiling | Multi-object binding, swap-test specificity | Planning success on true dynamics | d6 window, y4 area, d4 U-shape | Combined battery score at matched total FLOPs |
| **Controls** | random-init same-arch, D3, matched capacity/data/resolution | random-init same-arch, best frozen substrate, matched compute | dense-tokens arm, off-the-shelf slot model, swap test | action-shuffle, matched compute, unrolled depth, V-JEPA 2-AC | flat-schedule baseline, matched compute, noisy-TV-guarded ordering, seed sweep | best single expert+workspace, single pilot, strongest open model, matched total FLOPs |
| **Matched baseline** | Random-init same-arch encoder | Best frozen atlas substrate | Dense-without-slots | Reactive + unrolled depth + V-JEPA 2-AC | Flat-schedule substrate | Best single expert + workspace |
| **Random control** | random-init same-arch (non-vacuous) | random-init same-arch | n/a (dense + swap are the controls) | action-shuffle | random-perturbation ordering | random-router |
| **Tier / Diff** | studio / high | wider-box / high | wider-box / high | wider-box / high | wider-box / very high | moonshot / very high |
| **Dependency risk** | Trains an encoder (studio); do NOT run concurrently with the in-flight ViT job (OOM) | Blocked on DR1 + a wider box; gated by CM1/CM2/CM7 | Blocked on DR1 multi-object video + a wider box | Blocked on an ENV ADAPTER (a real implementation gap), not compute | Blocked on curriculum-ordered real video + a wider box; the highest-risk row | Blocked on multiple pilots + open-model access; the full-stack capstone |
| **Custom-model decision** | DECISIVE diagnostic (objective-as-lever) | DECISIVE (build side) | Yes | Weak | Yes (Q1 substrate) | Yes (synthesis) |

---

## Dedup ledger (which proposals merged into which registry id)

Proposals that were merged rather than double-listed:

| Registry id | Merged source proposals | Rationale |
|-------------|-------------------------|-----------|
| MT1 | MT1_routed_vs_best_mode (sec01) | Sole source; the routed-vs-best triad head |
| MT4 | MT5 router (sec04) + DR-ROUTER (sec03) + "Mixture-of-thinking router" (sec04) | All three are the same learned-router-over-reasoning-primitives synthesis test; merged into one central row |
| MT5 | "Adaptive latent compute beats fixed depth" (sec04, sec04-local MT1) | Adaptive-halt-vs-depth |
| MT6 | "Confidence stopping vs fixed early-exit" (sec04-local MT2) | |
| MT7 | "Beam/tree search over latent states" (sec04-local MT3) | |
| MT8 | "Latent debate between modules" (sec04-local MT4) | |
| DR1 | DR-VIDEO-CACHE (sec03) | The deferred prerequisite; kept as DR1 because it gates the DR program |
| DR2 | DR-SPARSE-REAL (sec03) == pr1_modular_plasticity_real_latents (sec05) | Same experiment (sparse heads on real latents); listed as DR2 and cross-referenced as PR3 |
| DR3-DR15 | sec04-local DR3, DR2, DR6, DR12, DR11, DR10, DR5, DR8, DR9, DR1, DR7, DR13 | Renumbered sequentially by decisiveness; sec04 local ids retired |
| PR3 | Duplicate of DR2 (kept a stub row so the plasticity program reads complete) | Cross-reference, one experiment |
| CM5 | cm5_studio_scale_rejuvenation_c3 (sec08/09) overlaps DR-PLASTICITY-SCALE (sec03) | Merged; the studio rejuvenation/C3 test |
| CM1 | cm1_gate_c1_compositional_bound_real_video (sec08/09) overlaps AT2_compositional_routing (sec01) | CM1 is the substrate gate; AT2's routed extension folded into CM4 (workspace routing on the same content) |
| CM7 | CM1_minimum_objective_encoder_probe (sec15) | Renamed CM7 to avoid colliding with the C1-gate CM1 |
| AT1 | AT1_cross_substrate_nuisance_grid (sec06) | The two AT1 proposals (sec01 mode-substrate-dependence, sec06 nuisance grid) were split: sec06's becomes AT1, sec01's becomes AT2 |
| AT2 | AT1_mode_substrate_dependence (sec01) | Renumbered to resolve the AT1 collision |
| AT3-AT5 | AT2_time_axis_ablation, AT3_programmatic_ceiling_reference, AT4_probe_class_sweep (sec06) | Renumbered after the AT1/AT2 split |
| AL1 | AL1_uncertainty_router_noisytv (sec01) | The uncertainty router (sec01) and the alignment AL1 (sec06) were distinct; sec01's becomes AL1, sec06's shared-latent alignment becomes AL2 |
| AL2 | AL1_shared_latent_alignment (sec06) | Renumbered |
| AL3 | AL2_audio_video_temporal_alignment (sec06) | Renumbered |
| PR2 | pr6_plasticity_advantage_real_vs_random_encoder (sec05) | The decisive learning-eases-on-special-substrate test |
| PR4 | pr3_epistemic_gate_vs_noisy_tv (sec05) | |
| PR5 | pr7_content_gated_critical_period (sec05) | |
| PR6 | pr2_offline_sleep_consolidation (sec05) | |
| PR7 | pr4_fast_slow_two_timescale (sec05) | |
| PR8 | pr5_memory_augmented_retrieval (sec05) | |
| PR1 | PR-MODE-ERROR-DISJOINTNESS (sec03) | The cheap router precondition |

Collisions resolved: the two source AT1 labels and the two source AL1 labels (each pair came from
different sections) were split by decisiveness. The section-04-local DR/MT numbering was retired in
favor of a single program-wide sequence. DR-SPARSE-REAL / pr1 is one experiment cross-referenced as
DR2 and PR3.

---

## The fork-deciding shortlist (rows that move frozen-dense-vs-custom, most-decisive first)

1. **DR1** (real bound-attribute video cache) - nothing decides until this lands.
2. **CM1** (C1 compositional bound on real video) - the primary keep-frozen gate.
3. **CM2** (multi-substrate atlas gate) - swap-encoder vs go-custom.
4. **AT1** (cross-substrate nuisance grid) - types the +0.31 signal that reopened the fork.
5. **AT4** (programmatic ceiling) - the only way to read a tie as a real substrate bound.
6. **CM7** (minimum-objective probe) - is the pretraining objective a lever at all.
7. **PR2** (plasticity real vs random-encoder) - does the special substrate ease LEARNING, not just readout.
8. **CM3** (dense vs pooled) / **CM4** (workspace shell) - is the bound the interface/shell, not the weights.
9. **CM8** (custom-JEPA pilot) - the build-side decision, gated behind all of the above.
