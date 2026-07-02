# Section 5: The Learning and Plasticity Program

## 0. Central question and the honest starting position

Can a model learn like a living system: fast when appropriate, stable when necessary, selective under uncertainty and novelty? That is the whole of this section. Every mechanism below is a candidate answer to one of three sub-questions:

1. Speed on demand (fast weights, one-shot binding, few-shot, rapid concept acquisition).
2. Stability under interference (consolidation, EWC/Fisher, replay, plasticity-stability tradeoff, BWT/FWT).
3. Selectivity under uncertainty and novelty (neuromodulated gating, critical periods, active learning, intrinsic motivation).

The honest starting position, stated up front so nothing below reads as a fresh promise: the corpus has already run the biological-plasticity playbook once, hard, and it came back a NEGATIVE map. Fisher-triggered reopening (n5), staged plasticity (e3), ACh/NE gating (n6), tag-and-capture (n4), homeostatic scaling (b4), and neuromod gating (e4) each tie or lose to their non-biological baseline. e4 is one of the strongest confirmed NEGATIVES in the whole program: 30/30 runs amplify error on noise, the wrong direction, because point prediction-error gating cannot separate epistemic from aleatoric uncertainty. The U-shaped overgeneralization signature (d4) is flat zero. No critical/sensitive window is substrate-specific (d6 substrate_specific_window=False). Path-dependence beyond a generic optimizer artifact is ~0 (y4). ex5_local_rules was refuted as an Adam artifact.

So the burden of proof in this section is not "does biology inspire a mechanism." It is "does a biologically-named mechanism beat the boring baseline that does the same computational work without the biological framing." Almost every one so far has not. What DID survive is architectural, not biochemical: e7_sparse (sparse/gated heads halve catastrophic forgetting vs a parameter-matched dense head, 30-run seed/axis sweep) survives BECAUSE forgetting is a trained-shell-dynamics metric, not a probe metric, so the invertible frozen-random projection cannot vacuously match it. That is the single load-bearing plasticity positive, and it is a fact about head architecture, not about V-JEPA geometry.

Two methodological landmines govern everything here:

- The vacuous-control discovery. frozen_random_projection is a full-rank invertible 1024x1024 Gaussian matrix. Any linear or MLP probe absorbs its inverse, so probe deltas are mathematically forced to 0.000. A plasticity claim that reduces to "a probe reads X better" is untestable against this control. Only metrics whose DYNAMICS depend on input conditioning survive it: SGD forgetting, BWT, adaptation speed, path integrals. Every experiment below is scored on a trained-shell-dynamics metric, never a static probe accuracy, or it is scored against a NON-projection control (shuffle, matched-compute, tuned-baseline, random-ENCODER features).
- The ceiling problem. Synthetic gratings and hand-bound objects are trivially linearly separable in 1024-d and additive by construction. A plasticity experiment run on such data will show either 1.0 everywhere (no forgetting to measure) or a forgetting curve that is an artifact of the toy stream, not of the mechanism. The prerequisite that gates the strong versions of most experiments below is the same one that gates the rest of the program: real natural-video cached latents with non-additively bound attributes, plus a difficulty-calibration certificate (D3) that the regime carries real structure.

The one genuinely new empirical hook this section can exploit is the corrected substrate result: real V-JEPA decodes shape under heavy nuisance at 0.379 vs random-pixel untrained features at 0.069 (delta +0.31, off-ceiling, honest ~0.21-0.23 after the resolution discount). This is the first evidence the frozen substrate carries real learned perceptual structure. It matters here because a plastic shell sitting on a substrate that already factors nuisance may have a genuinely easier learning problem than one sitting on random features, and that is itself a testable plasticity claim (PR6 below).

## 1. Mechanism-by-mechanism

Each mechanism gets: MVP impl, stronger impl, best baseline (the thing it must beat), exact null, failure mode, what a positive proves and does NOT prove, metrics, tier, relation to E1-E4, relation to custom-model design. All impls run on the existing shell (`plasticity.py`, `neuromod.py`, `buffer.py`, `consolidation.py`, `modulation.py`) over cached latents; none touch the encoder.

### 1.1 Critical / sensitive periods

- MVP: `PlasticityController` schedule=hard (full plasticity then step-drop to floor at progress 0.5) vs schedule=soft (exponential decay to a positive floor). Train the shell on an early task, measure whether a late task suffers when the window has closed.
- Stronger: signal-triggered reopening (the `reopen_threshold` path in `lr_scale`), where a normalized surprise/novelty z-score above 1.0 REOPENS plasticity. This is the interesting version: a period that is not clock-gated but content-gated.
- Best baseline: a constant learning rate tuned to the same final loss, and a cosine-decay LR schedule (the standard non-biological "plasticity decays" story). The critical-period claim must beat cosine decay, not just a constant.
- Exact null: with the schedule applied to a shuffled task order (so early vs late carries no developmental content), the closed-window penalty on the late task is within the seed spread of the open-window case. If closing the window hurts equally regardless of WHAT is learned first, it is just an LR annealing effect.
- Failure mode: d6 already found no substrate-specific window. The likely failure is that any window effect is fully explained by total effective LR-integral (area under the LR-vs-time curve), i.e. matched-compute kills it.
- Positive proves: that ordering-dependent consolidation exists (learning A early then closing the window protects A better than an LR schedule with the same integral). Does NOT prove: that the substrate has a biological critical period, only that the trained shell exhibits order-dependent stability.
- Metrics: BWT on the early task, FWT on the late task, LR-integral (matched-compute control), seed sign-flip rate.
- Tier: cpu-now (schedule already implemented).
- Relation to E1-E4: this is the mechanism e3_staged tested and lost; the new angle is content-gated reopening vs clock-gated, and matched LR-integral, which e3 did not isolate.
- Relation to custom design: if content-gated reopening wins where clock-gated lost, the custom shell wants a surprise-driven LR gate, cheap to add. If it loses, drop the schedule entirely and use plain SGD.

### 1.2 Staged / neuromodulated / uncertainty-gated plasticity

- MVP: `Neuromodulation.gate` mapping surprise/novelty/uncertainty to an LR multiplier feeding `PlasticityController`.
- Stronger: distributional (epistemic) uncertainty from ensemble disagreement (`disagreement` arg) instead of point prediction-error, gating LR up on high-disagreement (reducible) surprise and NOT on high point-error alone.
- Best baseline: constant LR at matched total compute, and an LR that scales with raw loss magnitude (the naive "learn faster when wrong" rule).
- Exact null: on the noisy-TV split (irreducible aleatoric noise mixed with reducible structure), the gated arm allocates no more learning to the reducible partition than the ungated arm. This is exactly the null e4 FAILED: point-error gating chased the noisy TV 30/30.
- Failure mode: conflating epistemic and aleatoric. The MVP surprise signal cannot do this. Only ensemble-disagreement or a distributional head can, and it may still tie a tuned baseline.
- Positive proves: that a self-generated uncertainty signal can steer plasticity toward reducible structure. Does NOT prove: that the signal is a neuromodulator, only that ensemble disagreement is a usable epistemic proxy.
- Metrics: fraction of LR-integral spent on the reducible vs noise partition, noisy-TV guard pass/fail, calibration (ECE of the uncertainty signal).
- Tier: cpu-now.
- Relation to E1-E4: direct successor to e4_neuromod, whose confirmed negative is the reason the MVP must use disagreement, not point error.
- Relation to custom design: if disagreement-gating beats the noisy-TV where point-error failed, the custom shell keeps a small ensemble purely to source the gate. If not, no gate.

### 1.3 Expected vs unexpected uncertainty (ACh / NE)

- MVP: two channels in `Neuromodulation`, novelty (ACh, expected uncertainty, gates memory write) and disagreement/context-shift (NE, unexpected uncertainty, gates reset/explore).
- Stronger: dissociate them, ACh routes to buffer-write priority, NE routes to LR reopening and a context-switch flag consumed by `ContextGating`.
- Best baseline: a single scalar surprise doing both jobs (the n6 baseline), and a fixed 50/50 write/explore split.
- Exact null: the two-channel dissociation gives no advantage over a single scalar on either retention or adaptation. n6 (ach_ne_dissociation) already found this tie.
- Failure mode: the two signals are correlated in practice (both track surprise), so the dissociation is nominal. n6's negative is the expected outcome unless a task explicitly decouples novelty from disagreement.
- Positive proves: that separating expected from unexpected uncertainty routes learning better. Does NOT prove any neurochemistry.
- Metrics: adaptation speed after a context switch (NE job), one-shot write recall (ACh job), and the cross-condition where only one signal fires.
- Tier: cpu-now.
- Relation to E1-E4: e4 family; n6 is the prior negative.
- Relation to custom design: low priority; only revisit if a task is built that genuinely decouples the two axes.

### 1.4 RPE as dopamine

- MVP: reward-prediction-error as a priority signal into the replay buffer (`priority` arg to `ReplayBuffer.add`) rather than an LR gate.
- Stronger: RPE modulates BOTH replay priority and consolidation-anchor strength (high-RPE experiences get stronger EWC anchors).
- Best baseline: recency priority, uniform priority, and TD-error priority from a plain value head (standard PER).
- Exact null: RPE priority ties uniform/recency at matched buffer size (the e2_replay null). Prioritized replay must beat random replay, which the corpus has NOT cleanly shown to hold generally.
- Failure mode: on a supervised latent stream there is no reward, so RPE collapses to prediction-error, which is just surprise again. Needs an actual reward/return, which means the env-later tier.
- Positive proves: RPE-weighted replay improves retention over recency at matched buffer. Does NOT prove a dopamine analog.
- Metrics: frontier AUC, BWT, buffer hit-rate on high-return states.
- Tier: env-later (needs reward).
- Relation to E1-E4: e2_replay prioritization knob.
- Relation to custom design: the buffer already carries a scalar priority; RPE is just one source. Cheap if reward exists.

### 1.5 Synaptic tagging and capture

- MVP: `PlasticityController` rigidity term is structurally SI; tagging adds a transient per-weight tag set on high-gradient steps that later capture consolidation.
- Stronger: two-timescale tag, a fast tag decays unless a later salience (RPE/novelty) event captures it into the slow rigidity map.
- Best baseline: SI (path integral) and EWC (Fisher) with no tag, at matched penalty strength.
- Exact null: tagging ties plain SI at matched consolidation budget. n4 (tag_and_capture) already found this tie.
- Failure mode: the tag and the SI path integral are computing nearly the same quantity (movement weighted by gradient), so the tag adds nothing SI does not.
- Positive proves: a delayed-capture rule protects late-tagged-then-reinforced weights better than immediate consolidation. Does NOT prove synaptic biology.
- Metrics: BWT on a task whose importance is revealed only AFTER training (delayed reward), sign-flip rate.
- Tier: cpu-now.
- Relation to E1-E4: extends e3/consolidation; n4 is the prior negative.
- Relation to custom design: skip unless delayed-importance tasks appear.

### 1.6 Consolidation (EWC / SI / Fisher)

- MVP: `Consolidation` method=ewc|si|both, penalty added to shell loss (already wired into e1_baseline).
- Stronger: online EWC (running Fisher) plus SI path integral composed, with per-module lambda.
- Best baseline: replay alone at matched buffer, and L2-to-init (naive anchoring). Consolidation must beat L2-to-init, the trivial "stay near start" penalty.
- Exact null: EWC ties L2-to-init, meaning the Fisher weighting adds nothing over uniform anchoring (weights all equally important, i.e. task too easy or too low-dimensional).
- Failure mode: at toy scale the shell has spare capacity, so nothing needs protecting and every method ties at ceiling. This is the ceiling problem again.
- Positive proves: Fisher-weighted anchoring beats uniform anchoring at matched strength. Does NOT prove which weights matter biologically.
- Metrics: BWT, forgetting curve area, Fisher concentration (fraction of penalty mass on top-k weights).
- Tier: cpu-now, but the discriminating regime is studio-scale (needs enough tasks to exhaust capacity).
- Relation to E1-E4: core of e1_baseline protected arm.
- Relation to custom design: EWC/SI are cheap and orthogonal to architecture; keep as a standing option, gate on whether they beat L2-to-init.

### 1.7 Sleep replay / offline consolidation

- MVP: an offline phase between tasks that samples the buffer and does gradient steps with NO new data (pure replay), interleaved with EWC refresh.
- Stronger: generative/interpolative replay (sample buffer keys, interpolate in latent space to synthesize near-manifold pseudo-latents), plus a "sharp-wave-ripple" schedule that prioritizes recently-surprising episodes.
- Best baseline: online replay at matched total gradient steps (the key control, offline replay must beat the same compute spent online), and no-replay.
- Exact null: offline replay ties online replay at matched steps. If sleep helps only by adding compute, it is not a real consolidation phenomenon.
- Failure mode: latent-space interpolation leaves the manifold and injects garbage (the substrate is not a generative model). Mitigated by staying near retrieved keys.
- Positive proves: separating learning into wake (encode-to-buffer) and sleep (replay-only) phases beats mixing them at matched compute. Does NOT prove sleep.
- Metrics: BWT, frontier AUC, matched-step control delta, off-manifold rate of synthesized latents (distance to nearest real key).
- Tier: cpu-now (this is the never-built N2 lane).
- Relation to E1-E4: e2_replay scheduling.
- Relation to custom design: if wake/sleep separation beats matched-online, the custom shell wants an explicit consolidation stage in its loop. This is the strongest candidate to justify a staged training loop.

### 1.8 Plasticity-stability tradeoff

- MVP: sweep the LR floor / rigidity weight and plot the retention-vs-adaptation frontier (BWT vs adaptation speed on a new task).
- Stronger: a controller that moves ALONG the frontier online, driven by the uncertainty gate (high novelty -> more plastic, stable regime -> more rigid).
- Best baseline: fixed points on the frontier (constant LR at several values). The online controller must Pareto-dominate the best fixed point, not just sit on the frontier.
- Exact null: the online controller lands on the same frontier the fixed points trace, i.e. no Pareto improvement, only a different operating point.
- Failure mode: the controller's own hyperparameters are just a reparameterization of the fixed LR, so it cannot dominate.
- Positive proves: online adaptation of plasticity Pareto-dominates any fixed setting. Does NOT prove a homeostatic mechanism.
- Metrics: Pareto frontier area, dominated-point count, seed stability of the dominance.
- Tier: cpu-now.
- Relation to E1-E4: cuts across e1/e3/e4.
- Relation to custom design: this frontier IS the design decision for how much plasticity the shell keeps; the experiment quantifies the cost of the "moldable shell" doctrine.

### 1.9 Meta-plasticity

- MVP: the learned gate in `PlasticityController` (sigmoid with learnable slope/offset) trained across a task distribution so the LR schedule itself is learned.
- Stronger: per-weight learned learning rates (a small hypernetwork emitting LR from weight statistics), the "plasticity of plasticity."
- Best baseline: the best hand-tuned schedule from 1.1, and MAML-style meta-learned init at matched meta-budget.
- Exact null: the learned gate recovers a schedule statistically indistinguishable from a tuned fixed one; meta-learning bought nothing.
- Failure mode: meta-overfitting to the task distribution; fails on held-out task families.
- Positive proves: a learned plasticity schedule generalizes across task families better than any single tuned schedule. Does NOT prove metaplasticity as a biological substrate.
- Metrics: held-out-family adaptation speed, schedule similarity to tuned baseline, meta-compute matched.
- Tier: studio-scale (needs a task distribution).
- Relation to E1-E4: extends e3.
- Relation to custom design: only worth the complexity if it generalizes across families; otherwise ship a tuned schedule.

### 1.10 Drift

- MVP: measure representation drift of the trained shell's readout across a task stream (cosine of head weights over time), separate from encoder (which cannot drift, it is frozen).
- Stronger: distinguish beneficial drift (adaptation) from harmful drift (forgetting) by projecting drift onto the retained-task gradient.
- Best baseline: a static head (no drift by construction) and its retention.
- Exact null: shell drift is uncorrelated with forgetting (drift is benign reparameterization). The frozen encoder makes this cleaner than in a fully-trainable net.
- Failure mode: drift is dominated by the last task and tells us nothing beyond BWT.
- Positive proves: a specific drift direction predicts forgetting and can be penalized. Does NOT prove representational drift as in cortex (the encoder is frozen).
- Metrics: drift-forgetting correlation, projected harmful-drift magnitude.
- Tier: cpu-now.
- Relation to E1-E4: diagnostic on e1.
- Relation to custom design: informs whether a drift penalty is worth adding; likely subsumed by EWC.

### 1.11 LoRA / adapters as controlled plasticity

- MVP: low-rank adapters on the shell predictor as the ONLY trainable path, rank as the plasticity budget knob. (Encoder stays frozen regardless; this is about the shell.)
- Stronger: per-task adapters with a router, so plasticity is modular and task-localized (this connects to e7_sparse and 1.13).
- Best baseline: full fine-tuning of the shell at matched trainable-param count, and a single shared adapter.
- Exact null: rank-r adapters tie full shell fine-tuning at matched params on both retention and adaptation (rank is not a meaningful plasticity budget).
- Failure mode: at toy scale the shell is already tiny, so low-rank vs full is a distinction without a difference.
- Positive proves: constraining plasticity to a low-rank subspace reduces interference at matched capacity. Does NOT prove anything about the frozen substrate.
- Metrics: BWT, adaptation speed, trainable-param-matched delta.
- Tier: cpu-now.
- Relation to E1-E4: architectural sibling of e7_sparse.
- Relation to custom design: adapters are the natural "controlled plasticity" primitive for the custom shell; this experiment sets the rank budget.

### 1.12 Modular plasticity (relation to e7_sparse)

- MVP: reproduce e7_sparse (k-WTA / MoE heads vs parameter-matched dense) and confirm the forgetting halving on REAL cached latents, not just synthetic Gaussian clusters.
- Stronger: route plasticity per-module by the uncertainty gate, so only the responsible expert updates on a given input.
- Best baseline: parameter-matched dense head (the exact e7 baseline), plus a dense head with the same L1 sparsity penalty (to separate sparsity-of-activation from modularity-of-parameters).
- Exact null: sparse/modular ties parameter-matched dense once run on real latents with a formal significance test (the open question #5). Synthetic-only is not enough.
- Failure mode: the synthetic Gaussian-cluster advantage does not transfer to real latents because real classes are not cleanly clustered.
- Positive proves: modular heads reduce catastrophic forgetting on real cached latents, confirming e7 is substrate-independent architecture. Does NOT prove V-JEPA geometry matters (it is a head fact).
- Metrics: mean_gain vs dense, dense_BWT delta, 30-run sign-flip rate, D3 difficulty certificate on the real regime, paired significance test.
- Tier: cpu-now (real latents already cached).
- Relation to E1-E4: e7_sparse is the surviving positive; this is its real-data confirmation, the highest-value single experiment in the section.
- Relation to custom design: if it survives on real latents, modular sparse heads are the FIRST justified custom-shell architectural commitment.

### 1.13 Fast weights / slow weights

- MVP: a fast-weight adapter (Hebbian outer-product store, decays over a short horizon) alongside the slow SGD-trained head, following the `WorkingMemory` slot mechanism in `modulation.py`.
- Stronger: fast weights written by the ACh/NE gate (novelty triggers a fast write), read by attention over recent latents.
- Best baseline: slow weights only, and a plain recurrent working-memory head at matched params.
- Exact null: fast weights tie slow-only on rapid adaptation (the fast store is redundant with the head's own capacity).
- Failure mode: fast weights just cache the last few examples, indistinguishable from a small replay buffer at matched size.
- Positive proves: a two-timescale weight system adapts within a task faster than one-timescale at matched capacity. Does NOT prove a biological fast-weight mechanism.
- Metrics: within-task adaptation curve, retention after the fast store decays, matched-capacity control.
- Tier: cpu-now.
- Relation to E1-E4: extends the modulation modules used around e3/e4.
- Relation to custom design: the two-timescale split is a strong candidate for the custom shell if it beats matched-capacity single-timescale.

### 1.14 Memory-augmented adaptation

- MVP: `ReplayBuffer.retrieve` (KV index over frozen latents) feeding a retrieval-conditioned head (retrieved neighbors' labels as a prior). Because the encoder is frozen, keys never go stale, the one place the frozen constraint actively helps.
- Stronger: retrieval AND fast-weight write on novelty, a full episodic-plus-fast-weight adapter.
- Best baseline: a parametric head at matched params (no memory), and a k-NN classifier on raw latents (the trivial memory baseline). Retrieval-conditioning must beat plain k-NN, or the parametric part is doing nothing.
- Exact null: retrieval-conditioned head ties plain k-NN on latents (the head adds no value over the memory), OR ties the no-memory parametric head (the memory adds no value).
- Failure mode: on ceilinged synthetic data k-NN is already at 1.0, so nothing to beat.
- Positive proves: combining episodic retrieval with a parametric head beats either alone at matched capacity. Does NOT prove anything about the substrate beyond stable keys.
- Metrics: few-shot accuracy vs k-NN and vs parametric, retrieval hit-rate, adaptation speed on a new class.
- Tier: cpu-now (buffer implemented).
- Relation to E1-E4: e2_replay retrieval knob.
- Relation to custom design: stable-key retrieval is a doctrinal FREE LUNCH (frozen encoder), and the natural episodic-memory primitive for Mixture of Perspectives.

### 1.15 Online / continual learning, forgetting, FWT / BWT

- MVP: the e1_baseline harness itself (domain/class-incremental latent stream, protected vs naive arms, BWT/FWT/adaptation-speed metrics).
- Stronger: task-agnostic continual learning (no task boundaries given), boundaries inferred from the NE context-shift signal.
- Best baseline: naive sequential (lower bound), joint offline training (upper bound), replay-only, EWC-only.
- Exact null: with shuffled task labels, protected and naive show no retention gap (no real task structure), OR both fail the last task (nothing learned). This is the e1 null, the gate every downstream result depends on.
- Failure mode: the stream is too short or too easy to induce forgetting, so BWT is ~0 for everyone (ceiling).
- Positive proves: a protected learner retains across a stream while still learning the last task. Does NOT prove any single mechanism; it is the substrate on which the others are measured.
- Metrics: BWT, FWT, adaptation speed, frontier AUC, D3 difficulty certificate.
- Tier: cpu-now, discriminating regime studio-scale.
- Relation to E1-E4: this IS e1.
- Relation to custom design: the continual-learning harness is the evaluation frame for the whole plasticity stack.

### 1.16 Rapid concept acquisition / few-shot / one-shot binding

- MVP: fast-mapping (d1_fast_mapping style), one exposure to a new class, immediate test, via retrieval + fast weights.
- Stronger: one-shot binding of a novel attribute conjunction (this is the P7 object-binding lane, and it needs non-additive bound attributes to be non-trivial).
- Best baseline: k-NN one-shot (memory lower bound), and a linear probe fit on the single shot.
- Exact null: one-shot binding ties k-NN, OR ceilings (the synthetic conjunction is linearly separable, so one shot suffices trivially, the ceiling problem in its sharpest form).
- Failure mode: exactly the ceiling problem, synthetic bound objects are additive and separable, so one-shot binding is trivial and proves nothing. This is why the strong version is gated on real bound-attribute video.
- Positive proves: the shell binds a novel conjunction from one example, off-ceiling, beating k-NN. Does NOT prove compositional abstraction unless the conjunction is held-out and non-additive.
- Metrics: one-shot accuracy on held-out conjunctions, k-NN gap, D3 certificate that the regime is non-ceiling.
- Tier: cpu-now for the MVP, env/data-later for the discriminating (real-video) version.
- Relation to E1-E4: developmental d1/d2 family.
- Relation to custom design: one-shot binding is the sharpest test of whether the special substrate (the +0.31 result) also FACTORS attributes; ties into compositional_under_nuisance.

### 1.17 Child-like learning / curriculum

- MVP: ordered (easy-to-hard) vs shuffled curriculum on the latent stream (d5/d7 style).
- Stronger: self-generated curriculum from learning-progress (LP) signal, the noisy-TV-guarded version.
- Best baseline: random order, anti-curriculum (hard-to-easy), and matched-compute random order. Curriculum must beat matched-compute random, not just random at fixed budget.
- Exact null: ordered ties shuffled at matched compute (d5/b3 already lose; d7/b7 confounded). Curriculum benefit is a compute artifact.
- Failure mode: the LP signal chases the noisy TV (irreducible-noise items look maximally "learnable-next"), the same failure as e4.
- Positive proves: an ordering exists that beats matched-compute random. Does NOT prove developmental staging.
- Metrics: final accuracy at matched compute, LP-signal noisy-TV guard, sample efficiency curve.
- Tier: cpu-now.
- Relation to E1-E4: e3 staging, e4 uncertainty signal.
- Relation to custom design: if no ordering beats matched-compute random, drop curriculum from the stack entirely.

### 1.18 Active learning / intrinsic motivation

- MVP: uncertainty-sampling (query the highest-disagreement latent next) vs random query.
- Stronger: intrinsic-motivation curiosity (novelty/learning-progress reward) driving query selection, noisy-TV guarded.
- Best baseline: random query at matched query budget, and uncertainty-sampling with a distributional (epistemic) signal vs a point signal.
- Exact null: active queries tie random at matched budget (no informative query structure), OR the curiosity signal chases the noisy TV (the e4/curiosity failure).
- Failure mode: point-uncertainty active learning selects irreducible-noise items (aleatoric), wasting the query budget, the confirmed e4 direction.
- Positive proves: epistemic-uncertainty querying beats random at matched budget and ignores aleatoric noise. Does NOT prove intrinsic motivation as a drive.
- Metrics: accuracy vs query count, noisy-TV guard, epistemic-vs-aleatoric query allocation.
- Tier: env-later for true active querying, cpu-now for pool-based simulation.
- Relation to E1-E4: e4 uncertainty, noisy-TV guard.
- Relation to custom design: an epistemic query head is cheap if it beats random; otherwise the shell learns passively.

## 2. The Plasticity Stack

The stack is the concrete architecture that carries the surviving and testable mechanisms. It is deliberately minimal: every component must earn inclusion by beating a matched baseline on a trained-shell-dynamics metric, and everything defaults OFF until it does.

Components, in data-flow order:

1. Frozen substrate (V-JEPA 2 ViT-L, 1024-d, no_grad). Not plastic. Justified by the +0.31 nuisance result: it carries real learned perceptual structure a random encoder does not. Cached-latent-first, so keys never go stale.
2. Plastic shell (predictor + heads). The only always-on trainable path. Small. Trained by SGD on cached latents.
3. Fast-weight adapter (1.13). Two-timescale Hebbian store for within-task rapid adaptation. Default OFF; on only if it beats matched-capacity slow-only.
4. Episodic replay + KV retrieval (1.7, 1.14). The latent hippocampus. Stable keys (frozen encoder free lunch). Feeds both retrieval-conditioning and offline consolidation.
5. Uncertainty gate (1.2, 1.3). Ensemble-disagreement (epistemic) signal, NOT point error. Routes plasticity and buffer-write priority. Default OFF until it beats the noisy-TV guard, because e4 failed here 30/30.
6. Critical-period / plasticity schedule (1.1, 1.8). Content-gated (surprise-triggered) reopening, at matched LR-integral. Sets the operating point on the plasticity-stability frontier.
7. Consolidation stage (1.6, 1.7). EWC/SI plus an explicit offline replay-only "sleep" phase. Must beat L2-to-init and matched-online replay respectively.
8. Compressed long-term memory. The buffer under a rate-distortion budget (ties to the I-series i5/i9), so LTM is bounded, not unbounded storage.
9. Self-eval head. A small verifier/calibration head reading the ensemble disagreement, producing the epistemic signal that gates 5, 6, and the active-learning query in 1.18. This is also the honest-confidence output for Mixture of Perspectives.

Standing controls wired into the stack, not per-experiment: every stack claim is scored on BWT / adaptation-speed / forgetting-area (trained-shell dynamics, survives the vacuous projection), against matched-compute and tuned baselines, with a D3 difficulty certificate on the regime and a seed sweep publishing sign-flips. The random-ENCODER control (not the vacuous frozen-random projection) is the substrate arm.

What the stack is NOT: it is not a claim that any component is biological. Each is included as an engineering primitive that beat a baseline. The biological names are mnemonics, and the confirmed negatives (e4, n4, n5, n6, b4, e3) are the reason most biological framings are demoted to OFF-by-default.

## 3. How the plasticity stack becomes part of Mixture of Perspectives

Mixture of Perspectives is the larger frame: multiple thinking modes routed by a controller. The plasticity stack contributes three things to it.

First, the self-eval head (component 9) is the router's confidence input. The epistemic-uncertainty signal that gates plasticity is the SAME signal a mixture controller needs to decide "think harder / retrieve / defer / commit." One signal, two consumers. This is the tightest integration point: the plasticity gate and the thinking-mode router share the disagreement estimate.

Second, the fast/slow-weight split (component 3) is itself a thinking-mode distinction: fast weights are System-1-like within-context adaptation, slow weights plus consolidation are System-2-like durable learning. Mixture of Perspectives can route between "adapt fast in-context" (fast weights + retrieval) and "commit to memory" (consolidation stage) using the uncertainty gate. The plasticity stack thus supplies two of the mixture's modes.

Third, episodic retrieval (component 4) is the mixture's memory-lookup mode: when disagreement is high and a near neighbor exists, retrieve rather than compute. Because keys are stable (frozen encoder), this mode is cheap and doctrinally clean.

The honest caveat: none of this is justified until the surviving positives replicate on real cached latents. The single most load-bearing dependency is the confirmation of e7_sparse / modular plasticity (1.12, PR1) on real latents, because modular heads are the first architectural commitment the custom shell would make, and the offline-consolidation "sleep" win (1.7, PR2), because it is the only thing that would justify a staged training loop. Everything else in the stack defaults OFF and must earn its place.

## 4. Relation to the frozen-dense-vs-custom fork

The plasticity program bears on THE fork (keep frozen V-JEPA vs build custom) in a specific, limited way. The +0.31 nuisance result tilts toward keeping the frozen encoder because the substrate carries real structure. The plasticity experiments do NOT re-litigate the encoder; they all keep it frozen. What they decide is the SHELL architecture: modular sparse heads (1.12), low-rank adapters (1.11), a two-timescale weight split (1.13), and whether a staged consolidation loop is warranted (1.7). If PR1 and PR2 both land positive on real latents, the custom shell has three justified commitments (modular heads, adapter-rank plasticity budget, staged consolidation). If they tie, the shell stays a plain SGD-trained head and the plasticity doctrine is, for now, unsupported beyond e7_sparse. Neither outcome touches the encoder decision, which the substrate results own.

## 5. PR experiments emitted

The candidate registry rows below use the free PR prefix. Each carries a null and a baseline it must beat (the enforced contract). They are ordered by value: PR1 and PR2 are the load-bearing pair; PR3-PR7 extend the stack. All score on trained-shell-dynamics metrics or non-projection controls, never a static probe against the vacuous frozen-random matrix.
