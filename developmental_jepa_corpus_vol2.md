# Toward Developmental JEPA

## Corpus Volume II: The Remaining Mechanisms

### Memory, Prediction, Uncertainty, Neuromodulation, Curiosity, Maps, and Attention

---

## Framing note, carried from Volume I and sharpened

The object of design in this corpus is a developmental learning system. V-JEPA is not that system. V-JEPA is one frozen module inside it: inherited perception, the eyes and the early visual cortex, trained once and then held fixed. The system being built is not a JEPA and is not trained with a JEPA objective. It does not use masked latent denoising as its learning rule. It is a continual, self-directed agent whose perceptual front end happens to be a frozen V-JEPA encoder, and whose every other part (memory, plasticity control, neuromodulation, curiosity, relational maps, structural adaptation, learning rules, goals) is the actual research surface.

Because of that, the field Volume I called "JEPA mapping" is renamed here to "Substrate attachment." It specifies how each lever bolts onto the frozen perceptual module without becoming part of it. The encoder is borrowed. The mind, such as it is, is the thing around it.

Each dossier keeps the doctrine's shape, compressed to one page: biology, computational abstraction, substrate attachment, ML analogs, the experiment it feeds, failure mode with a detection metric, tractability, developmental role, dependencies, and a closing ladder-and-verdict line. Truth levels are labeled in brackets at the point of claim. Ladder levels refer to the 0-to-6 evidence ladder defined in Volume I, Section 4. Verdicts use the same vocabulary as the Volume I ranking table: build now, prototype, toy test, theory or later, simulation only, lab-scale. No em dashes anywhere, per the standing rule.

A short map of what the trainable shell looks like, so attachments are concrete. The frozen encoder produces latent tokens per clip (pooled in V-JEPA 2, dense per-token in 2.1). On top of that sit the parts this corpus designs: a predictor (forward latent dynamics, optionally action-conditioned), one or more task heads, an episodic buffer, a set of scalar neuromodulatory signals computed from the predictor's own statistics, a plasticity controller that gates learning rates and rigidity, and, where an environment exists, a planner or policy. When a lever says it attaches to "the predictor" or "the head" or "the buffer," it means one of these, never the encoder.

---

## Category A. Developmental timing and plasticity

This category is the rest of Axis A, after the spine levers (staged plasticity 6.1, consolidation 6.3, uncertainty gating 6.4) covered in Volume I. These are the finer distinctions and the harder controllers.

### A1. Sensitive periods, as distinct from critical periods

Biology. [Established neuroscience] A sensitive period is a window of heightened but not all-or-none plasticity. Unlike a critical period (hard open then hard close, the ocular-dominance case), a sensitive period leaves residual adult plasticity, so learning outside the window is still possible at reduced efficiency. Second-language phonology and absolute pitch are the standard human examples: harder after childhood, not impossible. The distinguishing variable is the degree of residual reversibility, not the existence of a window.

Computational abstraction. A soft, graded learning-rate envelope per skill or module, tapering toward a nonzero floor rather than slamming to zero, and partially recoverable.

Substrate attachment. A per-module learning-rate schedule on the predictor and heads that decays to a small positive floor instead of to zero. This is the explicit soft contrast to the hard-gate critical-period schedule of 6.1, and the point of having both is to measure whether hardness buys anything.

ML analogs. [Established ML] Discriminative fine-tuning and layer-wise learning-rate decay (Howard and Ruder, ULMFiT 2018), warm restarts (Loshchilov and Hutter 2017), learning-rate floors in long-run training.

Feeds. Experiment 3, as the soft-schedule arm against the hard critical-period arm.

Failure mode. Indistinguishable from a well-tuned constant small learning rate. Detection: compare directly against a constant-small-LR baseline on the adaptation-retention frontier; if the taper does not beat the floor, the sensitivity framing added nothing.

Tractability. Laptop.

Developmental role. Graded plasticity. Axis A.

Dependencies. A refinement of 6.1, only meaningful when the hard-gate version runs alongside so the contrast is visible.

Ladder: Level 2 to 3. Verdict: build now, as an arm of Experiment 3.

### A2. Piagetian staging

Biology. [Established developmental psychology, contested specifics] Piaget proposed qualitative stage reorganizations (sensorimotor, preoperational, concrete operational, formal operational), each depending on mastery of the last, rather than smooth accumulation. The modern view softens this: stages are more domain-specific and less synchronized than Piaget claimed (neo-Piagetian work by Case; core-knowledge critiques by Spelke showing infants know more, earlier). What survives robustly is the dependency structure: some competences are prerequisites for others.

Computational abstraction. A curriculum with hard prerequisites, where the task class available at stage N+1 is gated on a competence threshold at stage N. Distinct from a smooth curriculum by the discreteness of the unlock.

Substrate attachment. Order the task stream so that, for example, latent object-tracking and permanence-style tasks must clear a competence bar before relational or compositional tasks are introduced; gate progression on measured competence, not on step count.

ML analogs. [Established ML] Curriculum learning (Bengio et al. 2009), competence-based and teacher-student automatic curricula.

Feeds. Experiment 5 (self-curriculum) and the ordering logic of the whole dependency graph in Volume I Section 8.

Failure mode. The stage boundaries are researcher-imposed and arbitrary; the model may learn the "late" task without the "early" one, falsifying the claimed dependency. Detection: ablate the ordering. If order does not matter, there is no real stage structure to claim.

Tractability. Laptop, data ordering only.

Developmental role. Staged competence. Axis A and curriculum.

Dependencies. Pairs with learning-progress curiosity (6.5), which can discover stage boundaries automatically instead of hand-coding them. Prefer the discovered version once the hand-coded version establishes the baseline.

Ladder: Level 1 to 2. Verdict: prototype, as ordering structure for Experiment 5.

### A3. Scaffolding and the zone of proximal development

Biology. [Established developmental psychology] Vygotsky's zone of proximal development is the gap between what a learner does alone and what they do with help. Learning is most efficient at the frontier of assisted competence, and scaffolding is temporary support, faded as competence grows.

Computational abstraction. An adaptive difficulty controller that holds task difficulty just above current solo competence and fades assistance over time. Mechanically close to learning-progress maximization and to frontier-targeting automatic curricula.

Substrate attachment. A difficulty scheduler over latent-prediction tasks (prediction horizon, mask ratio, distractor count) that tracks current error and holds it inside a target band. Assistance is concrete: provide partial ground-truth latents or shorten the horizon early, then withdraw both as error falls.

ML analogs. [Established ML] Self-paced learning (Kumar et al. 2010), prioritized level replay (Jiang et al. 2021), success-rate-band task sampling in automatic curricula.

Feeds. Experiment 5, and it is the difficulty-controller form of curiosity.

Failure mode. Collapses into training on medium-hard examples with no gain over a fixed mixed-difficulty distribution. Detection: compare against a fixed-difficulty-distribution baseline at matched compute.

Tractability. Laptop.

Developmental role. Frontier-targeted curriculum. Axis A and C.

Dependencies. An instance of learning progress (6.5) turned into a controller; requires a difficulty axis to exist in the task design.

Ladder: Level 2. Verdict: prototype.

### A4. Maturational constraints, the "starting small" effect

Biology. [Established, with a famous ML echo] Biological learners begin with low-acuity sensors, small working memory, and restricted motor range, all of which scale up over development. The "less is more" hypothesis (Newport 1990) argues limited early capacity forces extraction of the simplest regularities first.

Computational abstraction. Deliberately limiting capacity (bandwidth, memory, resolution, horizon) early and growing it, used as a curriculum and regularizer that biases toward simple structure first.

Substrate attachment. Start the predictor with a short prediction horizon, a low-rank head, a small context window, or hard-pooled latents, then relax each over training. The frozen encoder's own progressive-resolution pretraining is a static cousin; here the growth is dynamic and on the shell, not the encoder.

ML analogs. [Established ML] Elman's "starting small" (1993), where a recurrent net learned a hard grammar only with initially limited memory that was then grown; progressive growing of GANs (Karras et al. 2018) as an architecture-growth echo; capacity-based curricula.

Feeds. Experiment 3 (capacity-schedule arm), and it interacts with module birth in Volume III.

Failure mode. Starting small merely slows convergence with no gain in final performance or retention. Detection: match final compute and compare both endpoint performance and forgetting.

Tractability. Laptop.

Developmental role. Capacity scheduling. Axis A.

Dependencies. Pairs with staged plasticity (6.1) and with structural growth (Volume III, module birth).

Ladder: Level 2. Verdict: build now, as an arm of Experiment 3.

### A5. Progressive unfreezing

Biology. [Engineering analog, weak direct biology] Loosely echoes the sequential maturation of cortical hierarchies, sensory areas stabilizing before association areas. In practice this is a transfer-learning technique with a developmental gloss.

Computational abstraction. Unfreeze modules in a fixed order over training, typically higher modules first, so lower features are not destroyed before higher ones have adapted to them.

Substrate attachment. The encoder is frozen by design, so this applies to the trainable stack: unfreeze the task head first, then the predictor, then any adapters, in stages. The one case where it touches the encoder is a deliberate ablation, partially unfreezing the top encoder block to measure the cost and benefit of breaking the frozen constraint, which is a clean substrate probe in its own right.

ML analogs. [Established ML] ULMFiT progressive unfreezing (Howard and Ruder 2018), surgical fine-tuning (Lee et al. 2022, where the layer to tune depends on the type of distribution shift).

Feeds. Experiment 1 and 3 as a training-protocol variable; the encoder-block-unfreeze ablation feeds the frozen-ceiling question of Volume I Section 2.

Failure mode. Order does not matter at this scale. Detection: permute the unfreeze order across seeds.

Tractability. Laptop.

Developmental role. Staged plasticity. Axis A.

Dependencies. A refinement of 6.1. The encoder-unfreeze ablation is the bridge to the frozen-latent-ceiling measurement.

Ladder: Level 2. Verdict: build now, as a protocol variable plus one ablation.

### A6. Perineuronal-net analogs

Biology. [Established neuroscience] Perineuronal nets are extracellular-matrix structures that condense around mature neurons, especially parvalbumin interneurons, and physically restrict plasticity, mechanically closing critical periods. Enzymatic digestion with chondroitinase reopens plasticity in adult cortex, which is the cleanest demonstration that closure is an active, reversible lock rather than decay.

Computational abstraction. A per-weight or per-unit rigidity term that accumulates as a unit stabilizes, and that can be selectively dissolved back to plastic on demand. The reopenable lock.

Substrate attachment. A per-parameter rigidity multiplier on the predictor that grows with a stability statistic (low recent gradient variance, high estimated importance) and that a norepinephrine-like surprise signal (6.4) can locally dissolve. This is the concrete implementation that unifies critical-period closure (6.1) and synaptic consolidation (6.3) and makes surprise-triggered reopening real rather than metaphorical.

ML analogs. [Established ML] Synaptic Intelligence importance (Zenke et al. 2017) is the closest; hard attention to the task masks (Serra et al. 2018); metaplasticity models.

Feeds. Experiment 3 (the reopening mechanism) and Experiment 4 (gate-driven dissolution).

Failure mode. Rigidity that never dissolves becomes global freezing; dissolution that fires too readily erases consolidation. Detection: the adaptation-retention frontier, plus a reopening-latency analysis between a surprising event and the rigidity drop.

Tractability. Laptop.

Developmental role. Plasticity lock with reopen. The Axis A and B bridge.

Dependencies. This is the shared implementation substrate for 6.1, 6.3, and 6.4. Build it once, interpret it under all three labels, and state that unification explicitly.

Ladder: Level 2 to 3. Verdict: build now, as the mechanism behind Experiments 3 and 4.

### A7. Inhibitory maturation and shifting excitation-inhibition balance

Biology. [Established neuroscience] The maturation of GABAergic inhibition, especially parvalbumin interneurons, is the trigger that opens and later helps close critical periods. The excitation-to-inhibition ratio shifts over development and sets cortical gain and selectivity. Too little inhibition and no critical period opens; the right level and it opens; net-stabilized inhibition and it closes.

Computational abstraction. A maturing normalization or gain-control term that sharpens representations and gates when learning is most effective. In artificial terms, a schedule on competition strength that tightens over training.

Substrate attachment. Schedule the strength of any competitive or normalizing operation in the head (k-winners-take-all sparsity level, divisive normalization, attention temperature) from loose to tight, and tie critical-period opening to reaching a target selectivity level rather than to a step count.

ML analogs. [Mostly novel as framed] Nearest existing pieces are scheduled sparsity, softmax temperature annealing, and the k-winners-take-all sparsity lever in Volume III.

Feeds. Experiment 3, and it connects to sparse coding in Volume III.

Failure mode. Equivalent to plain temperature or sparsity annealing with no developmental content. Detection: compare against fixed sparsity and fixed temperature.

Tractability. Laptop.

Developmental role. Gain and selectivity maturation. Axis A.

Dependencies. Couples critical periods (6.1) to sparse coding (Volume III). A candidate mechanistic trigger for plasticity opening, worth testing as such.

Ladder: Level 1 to 2. Verdict: prototype, coupled to the sparsity lever.

### A8. Learned-optimizer plasticity controllers

Biology. [Speculative biology, strong ML] Plasticity rules are themselves shaped by evolution and neuromodulation, so how to learn is partly learned. The biological warrant is loose; the engineering case is strong.

Computational abstraction. Replace the hand-designed plasticity schedule with a small learned controller (a meta-learned optimizer or a hypernetwork) that emits per-module learning rates and rigidity from the system's own signals.

Substrate attachment. A tiny controller network reads the neuromodulatory signals of 6.4 (error, ensemble disagreement, task-boundary estimates) and outputs the learning-rate and rigidity gates of 6.1 and A6, meta-trained across a distribution of task streams to optimize the adaptation-retention frontier directly.

ML analogs. [Established ML] Learned optimizers (Andrychowicz et al. 2016; the Metz et al. learned-optimizer line), differentiable plasticity and backpropamine (Miconi et al. 2018, 2019), meta-learned learning rates (Meta-SGD).

Feeds. A stretch arm of Experiment 4, the most advanced plasticity controller in the program.

Failure mode. Meta-learning is unstable and overfits the meta-training task distribution, failing on a held-out stream. Detection: held-out task-stream evaluation. Instability and compute cost are explicit risks.

Tractability. The controller is small and laptop-trainable, but meta-training over many episodes is finicky and benefits from a single GPU. Mid-program.

Developmental role. Meta-control of plasticity. The apex of Axis A.

Dependencies. Needs 6.1, 6.4, and A6 working first, since those define the action space it controls. Do this last in Axis A.

Ladder: Level 2. Verdict: prototype late, after the hand-designed controllers are characterized.

---

## Category B. Memory and consolidation

This category extends Axis B beyond the spine levers (latent replay 6.2, consolidation 6.3). The core attachment is the same buffer of frozen latents; what varies is the read policy, the write policy, and the offline processing.

### B1. Reverse replay

Biology. [Established neuroscience] After an experience, hippocampal place-cell sequences reactivate in reverse temporal order, preferentially at reward sites, and reverse replay is implicated in credit assignment: propagating value backward along the path that led to reward. Forward replay tends to dominate before an action (planning), reverse after (learning).

Computational abstraction. Replaying a stored episode backward to assign credit along the trajectory, distinct from forward replay used for simulation and planning.

Substrate attachment. When the system has reward or a goal-distance signal, replay buffered latent sequences in reverse to update value or goal-distance estimates along the path. In the passive video setting without reward, the closer analog is replaying a sequence backward to train a backward dynamics or inverse model, which has independent value as a representation probe.

ML analogs. [Established ML] Reverse experience replay and its credit-assignment benefits (eligibility-trace and backward-update literature; episodic backward update, Lee et al. 2019). Prioritized replay (Schaul et al. 2016) for the reward-weighting half.

Feeds. Experiment 2 as a replay-ordering variant, and Experiment 10 if a reward exists.

Failure mode. With no reward and no value function, reverse order gives no benefit over forward or shuffled replay. Detection: ablate ordering (forward, reverse, shuffled) and measure backward transfer and value-estimate accuracy.

Tractability. Laptop.

Developmental role. Credit assignment and consolidation. Axis B.

Dependencies. Needs the buffer (6.2). Reward-weighting needs a reward or goal signal; flag that dependency, since most early experiments are reward-free.

Ladder: Level 2. Verdict: prototype, as a replay-ordering ablation.

### B2. Time-compressed replay

Biology. [Established neuroscience] Replay events run faster than the original experience, often by an order of magnitude, compressing seconds of behavior into a sharp-wave ripple of tens of milliseconds. Compression is part of how the hippocampus teaches the cortex efficiently.

Computational abstraction. Replaying subsampled or summarized trajectories rather than full-resolution sequences, trading temporal fidelity for throughput and coverage.

Substrate attachment. Store and replay strided or pooled latent subsequences (every k-th latent, or segment summaries) so the predictor sees more distinct episodes per unit of replay compute. A natural pairing with the maturational "starting small" horizon schedule (A4).

ML analogs. [Established ML] Frame skipping and temporal subsampling in RL; trajectory summarization; hierarchical replay buffers. Less formalized than other replay variants, so partly novel as a developmental knob.

Feeds. Experiment 2, as a replay-efficiency arm.

Failure mode. Compression discards the temporal detail the task actually needs, hurting performance on fine-grained dynamics. Detection: sweep the compression ratio and find the point where performance degrades; report the frontier rather than a single setting.

Tractability. Laptop.

Developmental role. Efficient consolidation. Axis B.

Dependencies. The buffer (6.2). Trades against horizon-dependent tasks; test jointly with prediction horizon.

Ladder: Level 2. Verdict: prototype, as an efficiency sweep within Experiment 2.

### B3. Generative replay

Biology. [Plausible, debated] One reading of dreaming and offline replay is that the brain generates plausible experiences rather than only replaying stored ones, supporting consolidation without storing every episode. The generative interpretation is more contested than the literal-replay one.

Computational abstraction. Replace or augment the episodic buffer with a generative model that produces synthetic past-task samples for interleaving, removing the need to store raw episodes.

Substrate attachment. Train a small generator over the frozen latent space (a latent-sequence VAE, diffusion, or autoregressive prior) and sample synthetic latent sequences to interleave during continual learning. Because the encoder is frozen, the generator only needs to model the latent distribution, which is far easier than modeling pixels and is the reason generative replay is more tractable in this setting than in pixel-space continual learning.

ML analogs. [Established ML] Deep generative replay (Shin et al. 2017), brain-inspired replay (van de Ven et al. 2020), pseudo-rehearsal (Robins 1995).

Feeds. Experiment 2, as the buffer-free or buffer-light variant.

Failure mode. The generator itself forgets or its samples drift from the true past distribution, so replaying them poisons rather than preserves; this is the well-documented generator-degradation problem. Detection: measure distributional distance between generated and held-out true latents over the task stream, and compare retention against a stored-buffer control.

Tractability. Laptop, since the generator works in latent space, not pixels.

Developmental role. Memory without storage. Axis B.

Dependencies. The frozen encoder makes this unusually feasible; flag that as an advantage. Compare directly against the stored-latent buffer (6.2) to quantify the storage-versus-fidelity tradeoff, which is a sharp sub-question.

Ladder: Level 2 to 3. Verdict: prototype, as a high-value comparison against stored replay.

### B4. Memory indexing

Biology. [Established neuroscience] Indexing theory (Teyler and DiScenna 1986; Teyler and Rudy 2007): the hippocampus does not store the full content of an episode but an index, a pointer that reinstates the distributed neocortical pattern that was active during encoding. Memory is content stored in cortex, addressed by a hippocampal key.

Computational abstraction. Separate the index (a compact key) from the content (the full representation), and retrieve content by matching keys. A key-value memory.

Substrate attachment. Store compact keys derived from latents (a low-dimensional projection or hash) paired with values (the full latent sequence or a task target); retrieve by nearest-neighbor or learned attention over keys. This is the architecture that makes large episodic memory cheap and makes retrieval quality an explicit, ablatable variable, which matters because the Volume I failure analysis flags retrieval quality as a likely bottleneck.

ML analogs. [Established ML] Key-value and external memories (Memory Networks, Weston et al. 2015; Neural Turing Machines, Graves et al. 2014; differentiable neural dictionary in NEC, Pritzel et al. 2017), retrieval-augmented models.

Feeds. Experiment 2, as the retrieval-architecture variant, and it underpins any episodic-control experiment.

Failure mode. The index loses the distinctions that matter, so different episodes collide on the same key, exactly the frozen-latent-distinctiveness failure from Volume I Section 2.6. Detection: the linear-probe diagnostic on keys, plus retrieval precision and recall against held-out queries.

Tractability. Laptop.

Developmental role. Scalable episodic memory. Axis B.

Dependencies. The buffer (6.2) becomes a key-value store under this lever. Retrieval quality couples to the frozen-latent geometry; run the distinctiveness probe first.

Ladder: Level 2 to 3. Verdict: build now, as the retrieval backbone for Experiment 2.

### B5. Reconsolidation

Biology. [Established neuroscience] A reactivated memory becomes transiently labile and must be restabilized; during that window it can be updated, strengthened, or weakened (Nader, Schafe, LeDoux 2000). Memory is not write-once; recall reopens it for editing.

Computational abstraction. On retrieval, allow the stored item to be updated by current information before being written back, rather than treating stored memories as immutable.

Substrate attachment. When an episode is retrieved from the buffer and found to mispredict current dynamics, update the stored target (or its priority, or its key) and write it back, so the memory tracks a changing world instead of preserving stale episodes. A controlled lability window prevents runaway rewriting.

ML analogs. [Sparse ML precedent] Editable and updatable memories, online buffer correction, and the broad model-editing literature (knowledge editing) as a loose cousin. Genuinely underexplored as a continual-learning mechanism, which makes it a novelty opportunity and a risk.

Feeds. Experiment 2, as a buffer-update variant, and it interacts with reconsolidation-style updates under non-stationarity.

Failure mode. Uncontrolled rewriting degrades memories toward the present and erases the past the buffer was meant to protect, reintroducing forgetting through the back door. Detection: measure retention of original episodes after many reconsolidation events; compare against a write-once buffer.

Tractability. Laptop.

Developmental role. Adaptive memory under non-stationarity. Axis B.

Dependencies. The buffer (6.2) and a mispredict or surprise signal (6.4) to trigger the update. Pairs naturally with non-stationary task streams.

Ladder: Level 1 to 2. Verdict: prototype, with tight controls; high novelty, real risk.

### B6. Forgetting as pruning

Biology. [Established neuroscience] Forgetting is active and often adaptive: synaptic pruning removes unused connections, sleep-dependent downscaling (the synaptic homeostasis hypothesis, Tononi and Cirelli) renormalizes synaptic strength globally, and motivated forgetting clears interference. Not all forgetting is failure; some is housekeeping.

Computational abstraction. Deliberate, structured removal of stored items or weights to reduce interference and cost, with a policy for what to drop (least useful, least recent, most redundant).

Substrate attachment. A buffer-eviction policy (drop low-priority, redundant, or superseded episodes) and a weight-pruning schedule on the predictor, framed as adaptive forgetting rather than capacity failure. The interesting claim to test: structured forgetting can improve generalization and reduce interference, not merely save space.

ML analogs. [Established ML] Coreset and reservoir-sampling buffer policies (iCaRL herding, Rebuffi et al. 2017; gradient-based sample selection, Aljundi et al. 2019), magnitude and movement pruning (Han et al. 2015; Sanh et al. 2020), the lottery-ticket framing (Frankle and Carbin 2019).

Feeds. Experiment 2 (eviction policy) and the structural-pruning lever in Volume III.

Failure mode. Pruning removes exactly the rare-but-important episodes or weights, causing targeted forgetting of low-frequency tasks. Detection: stratify retention by task frequency; the failure shows as collapse on rare tasks while common tasks survive.

Tractability. Laptop.

Developmental role. Adaptive forgetting and interference control. Axis B, links to structure.

Dependencies. The buffer (6.2) for eviction, the predictor for weight pruning. Connects to Volume III structural sparsity.

Ladder: Level 2. Verdict: build now, as buffer-eviction policy plus a pruning arm.

### B7. Dreaming and sleep-like phases as offline optimization

Biology. [Established phenomenon, debated function] Sleep supports consolidation through replay during slow-wave sleep and possibly schema integration and synaptic renormalization; REM is variously linked to integration and to creativity. The phenomenon is solid; the precise computational function is still argued.

Computational abstraction. Distinct offline phases, interleaved with online experience, during which the system does not take in new data but reprocesses stored or generated experience: consolidation, renormalization, generative recombination, planning rehearsal.

Substrate attachment. A scheduled offline loop that, between task-stream segments, runs replay (6.2), generative recombination (B3), synaptic downscaling (a global weight-norm renormalization), and forward simulation for planning. This is the integrative phase that lets several Axis-B levers run together in a biologically motivated cadence, and it is a clean way to test whether offline processing beats fully online training at matched total compute.

ML analogs. [Established ML] Experience replay and target-network updates in DQN are crude offline-online splits; the broad use of separate consolidation phases in continual learning; offline RL as the limiting case of learning without new interaction.

Feeds. Experiment 2 and Experiment 5, and it is the cadence within which Axis-B levers combine.

Failure mode. The offline phase adds compute without measurable benefit over interleaved online training, or the renormalization step erases useful structure. Detection: match total compute between an offline-phase schedule and a fully online schedule, and compare the adaptation-retention frontier; ablate each offline sub-step.

Tractability. Laptop.

Developmental role. Offline consolidation cadence. Axis B integrator.

Dependencies. Composes 6.2, B3, B6, and the planner. A good place to first combine memory levers, but only after each is characterized alone, per the doctrine.

Ladder: Level 2. Verdict: prototype, as the combination cadence for Axis B.

---

## Category C. Prediction and uncertainty

This category is the rest of Axis C beyond curiosity. Several of these levers determine whether the neuromodulatory signals of 6.4 can be computed cleanly at all, so they are upstream of the whole uncertainty story.

### C1. Hierarchical prediction

Biology. [Established neuroscience, framework-level] Cortex appears to predict at multiple timescales and levels of abstraction simultaneously: fast low-level sensory prediction, slow high-level contextual prediction, with the hierarchy from sensory to association cortex spanning a gradient of temporal receptive windows (Hasson et al. 2008).

Computational abstraction. Predictors operating at multiple temporal and representational scales, with slower or more abstract levels providing context that constrains faster, more concrete ones.

Substrate attachment. Stack predictors over the frozen latent: a fast next-latent predictor, plus slower predictors over pooled or segment-summary latents that forecast coarse future state, with the slow prediction conditioning the fast one. In V-JEPA 2.1 the dense per-token features additionally allow a spatial hierarchy, not only temporal. In plain V-JEPA 2, the pooled latent limits this to the temporal axis, which is a substrate-dependent constraint worth stating.

ML analogs. [Established ML] Hierarchical and multi-timescale recurrent models (Clockwork RNN, Koutnik et al. 2014; HM-RNN, Chung et al. 2017), hierarchical world models, slow-feature analysis (Wiskott and Sejnowski 2002).

Feeds. The predictor design underlying Experiments 2 through 6; directly enables better surprise signals for Experiment 4.

Failure mode. The extra levels do not capture distinct timescales and collapse to redundant copies of the fast predictor. Detection: measure the temporal receptive window and the prediction error at each level; redundancy shows as identical error profiles.

Tractability. Laptop.

Developmental role. Multi-scale world model. Axis C backbone.

Dependencies. Upgrades the basic predictor. A spatial hierarchy depends on dense features, so it is 2.1-gated.

Ladder: Level 2. Verdict: build now, as a predictor-architecture upgrade.

### C2. Counterfactual prediction

Biology. [Plausible, framework-level] Prefrontal and hippocampal circuits support simulation of alternatives and counterfactual reasoning ("what would have happened if"), supporting planning and credit assignment beyond what actually occurred.

Computational abstraction. Predicting outcomes under hypothetical actions or interventions not actually taken, requiring an action-conditioned or intervention-conditioned model rather than a passive forecaster.

Substrate attachment. Use the action-conditioned predictor (the 2-AC-style or 2.1-AC-style model) to roll out latents under counterfactual action sequences and compare to the realized rollout, supporting planning and a controllability-aware curiosity signal. Counterfactuals over short horizons only, given the documented compounding-error limit on long rollouts.

ML analogs. [Established ML] Model-based RL rollouts and Dyna (Sutton 1991), world-model imagination (Dreamer line, Hafner et al.), counterfactual credit assignment (Mesnard et al. 2021).

Feeds. Experiment 6 indirectly and any planning experiment; the controllability filter for curiosity (links to ICM in 6.5).

Failure mode. Counterfactual rollouts diverge from reality due to compounding latent error, so the counterfactuals are fiction beyond a few steps. Detection: measure rollout fidelity versus horizon; cap counterfactual horizon at the point where fidelity degrades.

Tractability. Laptop for short horizons; environment needed for grounding longer ones.

Developmental role. Simulation and planning support. Axis C, links to action.

Dependencies. Needs an action-conditioned predictor. Bounded by the short-horizon rollout reality.

Ladder: Level 2. Verdict: prototype, short-horizon only.

### C3. Efference copy

Biology. [Established neuroscience] When the brain issues a motor command it sends a copy (efference copy or corollary discharge) to sensory areas, predicting the sensory consequences of self-action so they can be cancelled or attributed to self. This is why you cannot tickle yourself and how the visual world stays stable across saccades.

Computational abstraction. Conditioning sensory prediction on a copy of the agent's own action to predict and discount self-generated sensory change, separating self-caused from world-caused variation.

Substrate attachment. Feed the action into the latent predictor as an explicit input (already the design of the action-conditioned model) and use the resulting self-prediction to subtract expected self-caused latent change, leaving a residual that is world-caused. That residual is a cleaner surprise signal and a controllability filter, directly improving the epistemic-versus-aleatoric separation that Experiment 4 depends on.

ML analogs. [Established ML] Inverse-forward models in ICM (Pathak et al. 2017) implement exactly this controllability filter; action-conditioned video prediction; predictive state representations.

Feeds. Experiment 4 (cleaner neuromodulatory signal) and Experiment 5 (controllability-filtered curiosity).

Failure mode. The self-prediction is poor, so the residual mixes self and world and the filter fails. Detection: test on a sequence with known self-caused versus world-caused changes and measure separation accuracy.

Tractability. Laptop, given an action-conditioned predictor.

Developmental role. Self-versus-world separation. Axis C, links to action and curiosity.

Dependencies. Needs action conditioning. A prerequisite for trustworthy controllability-based curiosity.

Ladder: Level 2 to 3. Verdict: build now where actions exist; it directly sharpens Experiment 4.

### C4. Cerebellar forward models

Biology. [Established neuroscience] The cerebellum learns fast, precise forward models for motor control and timing, predicting the sensory consequences of movement on a short timescale and enabling smooth, calibrated action; it learns through climbing-fiber error signals and supervised-style correction.

Computational abstraction. A fast, supervised, low-level forward model specialized for precise short-horizon prediction and error correction, distinct from the slower, more abstract cortical predictor.

Substrate attachment. A small, fast, supervised next-latent predictor trained with a simple regression loss for high-precision short-horizon forecasting, complementing the slower hierarchical predictor (C1). Useful as the fast inner loop of a planner and as a high-resolution surprise detector.

ML analogs. [Established ML] Supervised forward dynamics models in model-based control; fast linear or low-capacity predictors used alongside larger ones; the broad internal-model literature in motor control (Wolpert and Kawato 1998).

Feeds. The fast layer of the predictor stack; planning and short-horizon surprise.

Failure mode. Redundant with the fast level of the hierarchical predictor (C1), adding nothing. Detection: ablate against C1's fast level. If indistinguishable, merge them.

Tractability. Laptop.

Developmental role. Fast precise forward model. Axis C.

Dependencies. Overlaps C1; treat as the fast specialized member of the predictor family, not a separate system, unless it earns separation empirically.

Ladder: Level 2. Verdict: prototype, likely merged into C1.

### C5. Kalman-filter analogs

Biology. [Framework-level, established as a model] Perception and motor control are well modeled as Bayesian filtering: maintaining a belief state, predicting it forward, and correcting it with observations weighted by their reliability. Cue-combination and sensorimotor studies fit Kalman-like optimal integration (Kording and Wolpert 2004).

Computational abstraction. Maintain a latent belief state with an associated uncertainty, predict it forward, and update it with new observations weighted by relative reliability, so the model knows how confident its state estimate is.

Substrate attachment. Wrap the latent predictor in a recursive belief update that carries an uncertainty estimate over the latent state (a learned Kalman-style gain, or a probabilistic latent), so the system maintains calibrated confidence in its current world-state estimate. This is the principled source of the expected-uncertainty (acetylcholine-like) signal in 6.4 and of confidence calibration in C7.

ML analogs. [Established ML] Deep Kalman filters (Krishnan et al. 2015), recurrent state-space models (Hafner et al. PlaNet 2019), Kalman variational autoencoders, deep state-space models.

Feeds. Experiment 4 (the expected-uncertainty signal), C7 (calibration), and any planner needing belief uncertainty.

Failure mode. The uncertainty estimate is uncalibrated or collapses, giving overconfident or useless belief variance. Detection: calibration curves on held-out predictions; collapse shows as constant or vanishing variance.

Tractability. Laptop.

Developmental role. Belief-state estimation with uncertainty. Axis C backbone for neuromodulation.

Dependencies. Upstream of 6.4 and C7. A probabilistic latent head may be the cleanest implementation; flag the choice between learned-gain and fully probabilistic variants.

Ladder: Level 2 to 3. Verdict: build now, as the belief-and-uncertainty layer that the neuromodulation experiment needs.

### C6. Bayesian surprise

Biology. [Established as a model] Surprise can be quantified as the divergence between prior and posterior beliefs after an observation (Itti and Baldi); attention and orienting track Bayesian surprise, which is distinct from raw prediction error because it weights by how much beliefs actually moved.

Computational abstraction. Surprise as the KL divergence between belief before and after an observation, a principled measure of information gained, as opposed to point-prediction error which conflates reducible and irreducible uncertainty.

Substrate attachment. Given the belief state of C5, compute the divergence between prior and posterior latent-belief distributions on each observation and use it as the surprise signal for memory writes, plasticity gating, and curiosity. This is the theoretically cleaner alternative to raw latent prediction error and is the most direct defense against the noisy-TV trap, because pure noise updates beliefs little once its irreducibility is learned.

ML analogs. [Established ML] Bayesian surprise (Itti and Baldi 2009), information gain as intrinsic reward (VIME, Houthooft et al. 2016), expected information gain in active learning.

Feeds. Experiment 4 (the principled surprise signal) and Experiment 5 (information-gain curiosity).

Failure mode. Requires a usable posterior; if the belief model is poor, Bayesian surprise is noise. Detection: validate on the noisy-TV distractor, where Bayesian surprise should fall to near zero for irreducible noise while raw prediction error stays high; that gap is the whole point and is directly measurable.

Tractability. Laptop.

Developmental role. Principled surprise and information gain. Axis C.

Dependencies. Needs C5's belief state. The preferred surprise signal once a belief model exists; raw prediction error is the fallback.

Ladder: Level 2 to 3. Verdict: build now, paired with C5, as the noisy-TV-resistant surprise signal.

### C7. Confidence calibration as a developmental variable

Biology. [Established phenomenon] Metacognition and confidence judgments track accuracy in humans and animals, and calibration of confidence develops and can be trained; well-calibrated uncertainty is what makes uncertainty-gated behavior trustworthy.

Computational abstraction. The match between a model's stated confidence and its actual accuracy, treated here as a quantity to monitor and improve over development rather than a fixed property, because every uncertainty-gated mechanism in this program is only as good as its calibration.

Substrate attachment. Measure and regularize calibration of the predictor's uncertainty estimates (temperature scaling, calibration losses, or proper scoring rules) and track how calibration changes as the system learns, since a gate driven by miscalibrated uncertainty is the silent failure mode behind several Axis-C experiments.

ML analogs. [Established ML] Calibration of neural networks (Guo et al. 2017), proper scoring rules and the Brier score, deep ensembles for calibrated uncertainty (Lakshminarayanan et al. 2017), evidential deep learning.

Feeds. A monitoring layer across Experiments 4 and 5; a precondition for trusting any uncertainty gate.

Failure mode. Calibration is treated as solved when it is not, so a gate fires on overconfident garbage. Detection: this lever is itself the detection; report calibration curves alongside every uncertainty-gated result, and treat poor calibration as an invalidating condition for those results.

Tractability. Laptop.

Developmental role. Trustworthy uncertainty. Axis C quality control.

Dependencies. Sits on top of 6.4, C5, and C6 as a mandatory diagnostic. Not optional if any uncertainty-gated claim is to be believed.

Ladder: Level 2 to 3. Verdict: build now, as a required diagnostic rather than a standalone result.

---

## Category D. Neuromodulation and control

This category extends 6.4 beyond the acetylcholine, norepinephrine, and dopamine signals already specified, into the slower control systems and the arbitration between control modes. The shared substrate is a set of scalar signals computed from the predictor's statistics that gate other processes.

### D1. Serotonin-like patience and temporal discounting

Biology. [Established phenomenon, contested mechanism] Serotonin is associated with behavioral inhibition, patience, and the willingness to wait for delayed reward; tonic serotonin appears to lengthen the effective time horizon and reduce impulsive switching, though the precise computational role remains debated.

Computational abstraction. A slow scalar that modulates the discount factor and the threshold for switching behavior or abandoning a task, controlling how long the system persists before giving up or moving on.

Substrate attachment. A tonic signal that adjusts the planner's discount factor and the give-up threshold for the current task or goal, set from progress statistics (raise patience when learning progress is positive but slow, lower it when progress has stalled). This is the lever that prevents both premature abandonment and pointless persistence on the self-curriculum.

ML analogs. [Sparse ML precedent] Meta-learned or adaptive discount factors, adaptive horizons in RL, persistence and commitment in option frameworks. Patience as an explicit neuromodulatory signal is underexplored, which is novelty and risk.

Feeds. Experiment 5 (curriculum persistence) and any planning experiment with task switching.

Failure mode. No clean operationalization, so the signal is arbitrary and unfalsifiable, which Volume I already flagged by placing serotonin-like patience at Level 0. Detection: only meaningful if tied to a measurable behavior (switch rate, time-to-abandon) and shown to beat a fixed discount; otherwise it stays metaphor.

Tractability. Laptop, conditional on a usable operationalization.

Developmental role. Persistence and horizon control. Axis C, links to curriculum.

Dependencies. Needs a planner or a task-switching decision to modulate, and a progress signal (6.5) to drive it. Lowest-confidence neuromodulator; treat skeptically.

Ladder: Level 0 to 1. Verdict: theory or later; promote only if a clean operationalization is found.

### D2. Model-based and model-free arbitration

Biology. [Established neuroscience] Behavior reflects two systems: a flexible, deliberative model-based controller and a fast, habitual model-free controller, and the brain arbitrates between them based on their relative reliability and the cost of deliberation (Daw, Niv, Dayan 2005; Lee, Shimojo, O'Doherty 2014). Reliance shifts with training, uncertainty, and time pressure.

Computational abstraction. Maintain both a planner (model-based, uses the world model to simulate) and a cached policy or value (model-free, fast), and arbitrate by their relative uncertainty and the available compute or time budget.

Substrate attachment. Run a planner over the latent world model (model-based) and a cached reactive policy or value head (model-free), and arbitrate by comparing their uncertainties (from C5 and 6.4): trust the planner when the world model is reliable and time allows, trust the cached policy when it is confident and fast action is needed. The arbitration signal is itself a neuromodulatory quantity.

ML analogs. [Established ML] Arbitrated and hybrid model-based and model-free RL, the Dyna family (Sutton 1991), value-equivalence and when-to-plan work, successor-feature arbitration.

Feeds. Any planning experiment; connects the world model to action and to D3 habit formation.

Failure mode. The arbitration signal is miscalibrated, so the system over-plans (slow, error-prone rollouts) or over-relies on a stale habit. Detection: measure decision quality and latency under each controller and under arbitration; arbitration should dominate both pure strategies on a speed-accuracy frontier.

Tractability. Laptop for the arbitration logic; environment needed to make the planner versus policy tradeoff real.

Developmental role. Control-mode selection. Axis C and action.

Dependencies. Needs a planner, a cached policy, and uncertainty estimates (C5, 6.4). Foundational for D3 and D5.

Ladder: Level 2. Verdict: prototype where an environment exists.

### D3. Habit formation

Biology. [Established neuroscience] With repetition, behavior shifts from goal-directed (sensitive to outcome value, model-based) to habitual (stimulus-response, outcome-insensitive, model-free), a transfer associated with dorsolateral striatum; habits are cheap and fast but inflexible (Dickinson; Yin and Knowlton 2006).

Computational abstraction. Caching the output of repeated deliberation into a fast reactive policy, trading flexibility for speed and cost as a behavior becomes reliable.

Substrate attachment. Distill repeated, successful planner outputs into the cached reactive policy of D2, so frequently solved situations bypass planning; combine with the arbitration signal so habits are overridden when outcomes change. The developmental claim: habitization should emerge with competence and reverse under surprise, and that dynamic is measurable.

ML analogs. [Established ML] Policy distillation (Rusu et al. 2016), amortized inference, distilling search into a fast policy (the AlphaZero pattern of training a policy on search outputs), behavioral cloning of a planner.

Feeds. Builds on D2; interacts with reconsolidation and with surprise-driven reopening.

Failure mode. Habits persist after the world changes, causing outcome-insensitive errors, which is the defining bug of habitization. Detection: change the outcome contingency and measure how long the cached policy persists in the old behavior; healthy arbitration should break the habit when surprise rises.

Tractability. Laptop for distillation; environment for the contingency-change test.

Developmental role. Automatization of competence. Axis C and action.

Dependencies. Needs D2 and a surprise signal (6.4) to trigger de-habitization. Pairs with D5.

Ladder: Level 2. Verdict: prototype, after D2.

### D4. Goal-directed control

Biology. [Established neuroscience] Goal-directed behavior is sensitive to the current value of outcomes and uses a model of action-outcome contingencies, associated with prefrontal cortex and dorsomedial striatum; it is the flexible counterpart to habit and is what allows rapid adjustment when goals or values change.

Computational abstraction. Action selection by simulating outcomes under the world model and choosing actions that achieve the current goal, fully sensitive to changes in goal or outcome value.

Substrate attachment. The planner of D2 operating in goal-conditioned mode: condition the latent rollout on a goal latent (an image-goal or a goal embedding) and select actions minimizing predicted latent distance to the goal, which is exactly the energy-minimization planning the action-conditioned substrate already supports. Goal-directed control is therefore the most directly inherited control mode, bounded by the same short-horizon rollout limit.

ML analogs. [Established ML] Goal-conditioned RL and hindsight experience replay (Andrychowicz et al. 2017), model-predictive control with learned models, universal value function approximators (Schaul et al. 2015).

Feeds. Any goal-reaching or planning experiment; the substrate for autotelic goal generation (links to 6.10).

Failure mode. Inherits the substrate's image-goal reliance and long-horizon degradation, so goal-directed control works for near goals and fails for distant or abstractly specified ones. Detection: sweep goal distance and abstraction; report the horizon at which success collapses.

Tractability. Laptop for short-horizon latent planning; environment for embodied goals.

Developmental role. Flexible goal pursuit. Axis C and action.

Dependencies. The action-conditioned predictor and a goal representation. Bounded by short-horizon planning; pairs with autotelic goal generation.

Ladder: Level 3. Verdict: build now where an environment and goals exist; it is the inherited control mode.

### D5. Planning-reactive switching

Biology. [Established neuroscience] The brain switches between deliberate planning and fast reactive responses based on time pressure, stakes, and confidence, and this switch is closely tied to the model-based and model-free arbitration of D2 but framed at the level of behavioral mode under real-time constraints.

Computational abstraction. A meta-controller that decides, per decision, whether to spend compute on planning or to act reactively, given a time budget and confidence, optimizing a speed-accuracy-cost tradeoff.

Substrate attachment. A switch that gates whether the planner runs (expensive latent rollouts) or the cached policy fires (cheap), driven by the arbitration signal of D2 and a compute or latency budget; on the action-conditioned substrate this directly governs whether the roughly multi-second planning loop is invoked, which is a real cost given the documented per-action planning time.

ML analogs. [Established ML] Adaptive computation time (Graves 2016), metareasoning and bounded rationality, anytime planning, learning when to plan.

Feeds. The real-time wrapper around D2 and D4; relevant to any wall-clock-sensitive experiment.

Failure mode. The switch adds overhead without improving the speed-accuracy frontier, or always picks one mode. Detection: measure the achieved speed-accuracy-cost frontier against always-plan and always-react baselines; the switch must dominate both.

Tractability. Laptop for the logic; environment and a real latency budget to make it meaningful.

Developmental role. Real-time control allocation. Axis C and action.

Dependencies. Needs D2 and D4 in place. The practical reason it matters here is the substrate's nontrivial planning latency.

Ladder: Level 2. Verdict: prototype, after D2 and D4.

---

## Category E. Curiosity and self-data

This category extends 6.5 beyond prediction-error, RND, and learning-progress curiosity into the data-selection and behavioral forms. The recurring honesty point: most of these need an environment or at least an active sampling choice, which is the main reason the curiosity cluster is staged later and partly flagged lab-scale.

### E1. Novelty search as data selection

Biology. [Established in developmental robotics and behavior] Organisms seek novel states and stimuli independent of reward; novelty drives exploration and is one of the oldest intrinsic motivations. In the developmental-robotics tradition it is formalized as seeking rarely-visited regions of state space.

Computational abstraction. Selecting data or actions to maximize novelty (low visitation density), as a pure diversity drive distinct from competence or information gain.

Substrate attachment. In the passive video setting, novelty becomes a data-selection policy: choose which clips, segments, or masked regions to train on next by latent-space novelty (low density under a visitation estimate over latents), which is testable without any environment and reframes curiosity as active learning. With an environment, novelty enters as a state-visitation bonus.

ML analogs. [Established ML] Novelty search (Lehman and Stanley 2011), count-based and pseudo-count exploration (Bellemare et al. 2016; Ostrovski et al. 2017), RND as a novelty proxy (Burda et al. 2019), core-set active learning.

Feeds. Experiment 5, as the data-selection form of self-curriculum, and it connects curiosity to active learning.

Failure mode. Pure novelty chases irreducible noise and degenerate diversity, the noisy-TV failure again, since novelty alone does not distinguish learnable from unlearnable. Detection: the noisy-TV distractor; a novelty-only policy should be shown to fixate on it, motivating the learning-progress correction.

Tractability. Laptop for data selection; environment for state-visitation novelty.

Developmental role. Diversity-driven exploration. Axis C, links to active learning.

Dependencies. A visitation or density estimate over latents. Best paired with learning progress (6.5) to filter noise.

Ladder: Level 2. Verdict: prototype, as the active-data-selection arm of Experiment 5.

### E2. Empowerment

Biology. [Theoretical, biologically motivated] Empowerment formalizes an agent's intrinsic drive to maximize its control over its environment, measured as the channel capacity between its actions and future states; it captures the intuition that agents seek states from which they have many controllable options.

Computational abstraction. An intrinsic objective to maximize the mutual information between actions and resulting future states, rewarding states of maximal controllability and influence.

Substrate attachment. Estimate the mutual information between action sequences and resulting latent states under the action-conditioned predictor, and reward high-empowerment states; this requires both an action-conditioned model and an environment, and the mutual-information estimate is notoriously hard, so it is a later, careful lever rather than an early one.

ML analogs. [Established ML] Empowerment (Klyubin et al. 2005), variational empowerment (Mohamed and Rezende 2015), and the skill-discovery methods that approximate it (DIAYN, Eysenbach et al. 2019; VIC).

Feeds. Experiment 10 indirectly, and skill discovery; links to options in Volume III.

Failure mode. Mutual-information estimation is high-variance and the objective is expensive, so the signal is unreliable at small scale. Detection: validate the estimator on a toy environment with known empowerment structure before trusting it.

Tractability. Single-GPU and an environment; estimation cost is real. Later.

Developmental role. Control-seeking exploration. Axis C and open-endedness.

Dependencies. Action-conditioned model, environment, a workable MI estimator. Low compute feasibility for a solo run.

Ladder: Level 1 to 2. Verdict: theory or later.

### E3. Active sensing and saccade-like sampling

Biology. [Established neuroscience] Vision is active: saccades, foveation, and attention sample the world non-uniformly to gather the most useful information, and active sensing (sniffing, whisking, eye movements) is the rule, not the exception. Perception is a sampling policy, not passive reception.

Computational abstraction. A policy that chooses where and what to sample next to maximize information, applied to the input itself rather than to locomotion, treating perception as a sequential decision.

Substrate attachment. Choose which spatial regions or which frames to attend to or encode next, for example selecting which dense-token regions (in 2.1) or which temporal windows to process under a compute budget, driven by expected information gain. This makes the system an active rather than passive consumer of video and is one of the few curiosity forms fully testable without locomotion, especially on 2.1's dense tokens.

ML analogs. [Established ML] Recurrent models of visual attention (Mnih et al. 2014), hard-attention and glimpse networks, active perception and next-best-view selection, foveated processing.

Feeds. Experiment 5 (active sampling) and it pairs with attention levers in Category G.

Failure mode. The sampling policy gives no benefit over uniform or random sampling at matched compute. Detection: compare task performance per unit of encoding compute against uniform sampling; the active policy must win on that budget-normalized metric.

Tractability. Laptop, especially on cached dense latents (2.1). One of the more attractive curiosity experiments precisely because it needs no environment.

Developmental role. Active perception. Axis C, links to attention.

Dependencies. Benefits strongly from dense features, so it is 2.1-favored. Pairs with salience and selective attention (G1, G2).

Ladder: Level 2. Verdict: build now on 2.1, as an environment-free active-learning experiment.

### E4. Boredom and habituation

Biology. [Established neuroscience and behavior] Habituation is the decline of response to repeated, unchanging stimuli, the simplest and most universal form of learning; boredom drives disengagement from mastered or unchanging situations and pushes the organism toward novelty. Both regulate where attention and learning go.

Computational abstraction. A decaying interest signal that falls for stimuli that are mastered or unchanging and recovers for novel or newly-changing ones, implementing automatic disengagement and re-engagement.

Substrate attachment. An interest or engagement signal per region or task that decays as prediction error falls and recovers when error rises, gating data selection and curriculum progression so the system stops dwelling on the mastered and the irreducibly noisy alike. Crucially, habituation to noise is the behavioral correlate of the learning-progress fix for noisy-TV: a learning-progress agent habituates to noise because its progress is zero.

ML analogs. [Established ML] Habituation and adaptation in novelty estimation, count decay, the learning-progress derivative (Oudeyer) as a principled boredom signal, recency-weighted novelty.

Feeds. Experiment 5, as the disengagement half of self-curriculum.

Failure mode. Habituation either never disengages (no boredom, fixation) or disengages too fast (no consolidation, premature abandonment). Detection: measure dwell time per task as a function of learning progress; both pathologies show as the wrong dwell-progress relationship.

Tractability. Laptop.

Developmental role. Disengagement and re-engagement. Axis C.

Dependencies. The progress signal (6.5). The behavioral expression of learning-progress curiosity; test them together.

Ladder: Level 2. Verdict: build now, as the disengagement mechanism in Experiment 5.

### E5. Play

Biology. [Established in developmental science] Play is intrinsically motivated, low-stakes, exploratory behavior that builds skills and models without immediate goals; it is widespread in young mammals and birds and is thought to develop motor, social, and cognitive competence through safe practice. Play recombines and stress-tests behaviors outside task demands.

Computational abstraction. Self-directed, goal-free or self-goal-generating behavior in a safe setting that builds reusable skills and improves the world model without external reward, combining curiosity, autotelic goals, and skill practice.

Substrate attachment. The most integrative curiosity lever: a phase in which the system generates its own goals (links to 6.10), pursues them with intrinsic reward (6.5), stores resulting trajectories (6.2), and builds a skill repertoire, all without external task reward, in an environment cheap enough to fail safely. Because it composes several Axis-C and open-endedness levers, it is a combination target, not an early standalone, and it genuinely needs an environment.

ML analogs. [Established ML] Intrinsically motivated skill acquisition, unsupervised RL and skill discovery (DIAYN, Eysenbach et al. 2019), autotelic and goal-generating agents (IMGEP, Forestier et al. 2017; GoExplore as a cousin), open-ended play in procedurally generated environments.

Feeds. Experiment 10 (minimal open-ended) is the closest realization; it sits at the convergence of Axes C and the open-endedness levers.

Failure mode. Without sufficient environmental richness, play produces no useful stepping stones and degenerates into trivial or repetitive behavior, which is the honest hard ceiling identified throughout this program. Detection: measure whether play-acquired skills transfer to held-out tasks; no transfer means the play was empty.

Tractability. Single-GPU and a rich enough environment; the richness requirement is the binding constraint and likely exceeds a solo year for anything ambitious.

Developmental role. Integrative skill building. Axis C and open-endedness apex.

Dependencies. Composes 6.2, 6.5, 6.10, and skill libraries (Volume III). The clearest case where environment richness, not algorithm, is the limit.

Ladder: Level 2, aspiring to Level 6. Verdict: lab-scale; the minimal version is Experiment 10.

---

## Category F. Cognitive maps and relational structure

This category extends 6.6 (successor representations, TEM, object-centric) into the specific cell types, binding mechanisms, and graph forms. The dominant substrate fact governs the whole category: pooled V-JEPA 2 latents resist factorization into objects and relations, so most of these levers are weak on 2 and become real on 2.1's dense per-token features. Where a lever is 2.1-gated, it is marked.

### F1. Head-direction and border-cell analogs

Biology. [Established neuroscience] Beyond place and grid cells, the spatial system includes head-direction cells (tuned to facing direction, an internal compass), border and boundary-vector cells (firing near environmental boundaries), and speed cells, together forming a metric, allocentric spatial code (Taube; Solstad et al. 2008; the broader entorhinal-hippocampal map).

Computational abstraction. Dedicated representations for orientation, boundaries, and self-motion that, combined with position, yield a full metric map supporting path integration and navigation.

Substrate attachment. Train small auxiliary heads on the frozen latent (dense in 2.1) to decode or predict orientation, boundary proximity, and self-motion in a navigation setting, then test whether such structured spatial variables are linearly present in the latent or must be constructed. On the navigation results the substrate already shows (low trajectory error on driving-style data), this is a probe of how much allocentric structure the frozen encoder already carries.

ML analogs. [Established ML] Emergence of grid-like and head-direction units in path-integrating networks (Banino et al. 2018; Cueva and Wei 2018), spatial representation learning, SLAM as the engineering counterpart.

Feeds. Experiment 6 (relational and spatial structure), navigation experiments.

Failure mode. The variables are not linearly decodable and adding heads merely adds capacity, the recurring trap for relational levers on a non-factored latent. Detection: the linear-probe diagnostic for each spatial variable, run before building anything on top.

Tractability. Laptop on cached latents.

Developmental role. Metric spatial map. Axis B and relational.

Dependencies. Strongly favored by dense features (2.1) and by a navigation setting. Run the probe first.

Ladder: Level 2, conditional on probe. Verdict: prototype on 2.1, gated by the decodability probe.

### F2. Object files

Biology. [Established cognitive science] Object files are mid-level visual representations that track individual objects across time and occlusion, maintaining identity and a bound set of features per object; they underlie object permanence and multiple-object tracking (Kahneman, Treisman, Gibbs 1992; Pylyshyn's FINST).

Computational abstraction. Persistent per-object slots that bind features and track identity across frames and occlusions, distinct from a holistic scene representation.

Substrate attachment. Learn a set of object slots over the dense latent tokens of 2.1 (slot attention applied to per-token features) and test whether slots track identity through occlusion in latent space; on pooled V-JEPA 2 this is largely infeasible because the latent does not expose per-object structure, which is the cleanest single illustration of why 2.1 reshapes this category.

ML analogs. [Established ML] Slot Attention (Locatello et al. 2020), object-centric world models (OP3, Veerapaneni et al. 2020; SAVi, Kipf et al. 2022), tracking-by-slots.

Feeds. Experiment 6 (object-centric arm, 2.1 only).

Failure mode. On non-factored latents, slots fail to separate objects and degenerate to spatial partitions or duplicates. Detection: object-tracking accuracy through occlusion and slot-identity consistency; failure is slots that swap or merge identities.

Tractability. Laptop on cached 2.1 latents; effectively blocked on plain 2.

Developmental role. Persistent object tracking. Relational, Axis B.

Dependencies. 2.1-gated. The flagship example of the 2-versus-2.1 distinction.

Ladder: Level 2 on 2.1, Level 0 to 1 on 2. Verdict: prototype on 2.1 only; the 2-versus-2.1 contrast is itself a result.

### F3. Latent binding

Biology. [Established as a problem, contested as a mechanism] The binding problem is how distributed features (color, motion, shape, location) are combined into unified object and event representations; candidate mechanisms include temporal synchrony and attention-mediated binding, none fully settled.

Computational abstraction. A mechanism that combines separately represented features into coherent bound entities, so that "red" and "moving left" and "to the right" attach to the same object rather than floating free.

Substrate attachment. Test and, if needed, add a binding mechanism over dense latent tokens (attention-based grouping, or a learned binding head) so that feature combinations are represented as belonging to entities; closely tied to object files (F2) and to compositional models (F4). Whether binding is needed at all depends on how bound the frozen latent already is, which is an empirical question the probe answers.

ML analogs. [Established ML] Attention as soft binding (Transformers), complex-valued and synchrony-based binding (Reichert and Serre 2014), the broader object-centric and binding literature (Greff, van Steenkiste, Schmidhuber 2020 survey on binding in neural networks).

Feeds. Experiment 6, alongside F2 and F4.

Failure mode. Binding is either already implicit in the latent (so the mechanism is redundant) or unachievable on a pooled latent (so it is impossible). Detection: a binding probe testing whether feature-object assignments can be read out; both redundancy and impossibility are visible in the probe.

Tractability. Laptop on cached latents; 2.1-favored.

Developmental role. Feature-entity coherence. Relational backbone.

Dependencies. 2.1-favored, tied to F2 and F4. Probe before building.

Ladder: Level 1 to 2. Verdict: prototype on 2.1, contingent on the binding probe.

### F4. Compositional world models

Biology. [Established cognitive science, framework-level] Human cognition is compositional: complex situations are understood as structured combinations of objects, relations, and events, supporting systematic generalization to novel combinations (the systematicity arguments of Fodor and Pylyshyn; structured cognition).

Computational abstraction. A world model whose state factorizes into entities and relations, so dynamics can be predicted compositionally and novel entity-relation combinations generalize without retraining.

Substrate attachment. Build the predictor over a factored, object-and-relation state derived from 2.1 slots (F2) plus a relational head (links to the graph head in 6.6), and test systematic generalization to unseen object-relation combinations; this is the strongest form of the relational program and is only meaningful once slots and binding work, so it sits downstream of F2 and F3.

ML analogs. [Established ML] Compositional and object-centric world models (C-SWM, Kipf et al. 2020; structured world models), graph network dynamics (Battaglia et al. 2016, 2018), neuro-symbolic and relational reasoning.

Feeds. Experiment 6 (the relational head and its 2-versus-2.1 comparison).

Failure mode. The factorization is imposed but does not match the latent's actual structure, so compositional generalization fails and a monolithic predictor matches or beats it. Detection: systematic-generalization splits (train on a subset of combinations, test on held-out combinations); failure is no compositional-generalization gap over the monolith.

Tractability. Laptop on cached 2.1 latents.

Developmental role. Systematic generalization. Relational apex.

Dependencies. Downstream of F2 and F3; 2.1-gated. The most ambitious relational claim; build last in the category.

Ladder: Level 2 on 2.1. Verdict: prototype on 2.1, after slots and binding.

### F5. Topological maps

Biology. [Established neuroscience] Not all spatial representation is metric; the hippocampal map also supports topological and graph-like structure (which places connect to which), and cognitive maps generalize beyond physical space to relational and conceptual graphs (the cognitive-map-as-graph view; Behrens et al. 2018).

Computational abstraction. A graph of states or places and their transitions, capturing connectivity without requiring precise metric coordinates, supporting planning by graph search and generalization across structurally similar graphs.

Substrate attachment. Build a transition graph over discretized or clustered latent states (nodes are latent prototypes, edges are observed transitions) and use it for graph-based planning and as a substrate for the successor representation (6.6); this works on pooled V-JEPA 2 latents because it needs only state similarity and transitions, not object factorization, which makes it one of the few relational levers viable on plain 2.

ML analogs. [Established ML] Semi-parametric topological memory (Savinov et al. 2018), graph-based planning over learned states, landmark and waypoint maps, the successor representation as a soft topological code (Stachenfeld et al. 2017).

Feeds. Experiment 6, and it underpins successor-representation experiments.

Failure mode. Latent clustering produces meaningless nodes or a graph too dense to plan over. Detection: planning success over the graph versus a baseline, and node-purity measures; failure is a graph that does not support better-than-baseline planning.

Tractability. Laptop on cached latents; works on plain 2.

Developmental role. Relational and topological map. Axis B, links to planning.

Dependencies. Needs only state similarity and transitions, so it is the most substrate-robust relational lever. Pairs with the successor representation (6.6).

Ladder: Level 2. Verdict: build now; viable on plain V-JEPA 2, unlike most of this category.

### F6. Scene-graph prediction

Biology. [Loose analog] Humans parse scenes into objects and their relations and predict how those relations will change; the relational parse of a scene is part of how events are understood and anticipated.

Computational abstraction. Representing a scene as a graph of objects (nodes) and relations (edges) and predicting the future scene graph, making relational change the prediction target rather than raw future state.

Substrate attachment. On 2.1 slots, attach a relational head that predicts future object relations (contact, support, containment, relative motion) and train it as an auxiliary or primary objective; test whether predicting relations improves dynamics modeling and generalization over predicting holistic latents. This is the explicit relational-prediction form of Experiment 6 and is 2.1-gated for the same reason as F2.

ML analogs. [Established ML] Scene-graph generation and prediction, relational dynamics models, graph network forward models (Battaglia et al. 2018), interaction networks (Battaglia et al. 2016).

Feeds. Experiment 6 directly, as the relational-prediction target.

Failure mode. Relations are not recoverable from the latent or relational prediction does not beat holistic latent prediction, so the relational framing adds nothing. Detection: compare relational-target prediction against holistic-latent prediction on dynamics and generalization; failure is no gain.

Tractability. Laptop on cached 2.1 latents.

Developmental role. Relational dynamics. Relational, Axis C overlap.

Dependencies. 2.1-gated, downstream of F2. The prediction-side complement to compositional world models (F4).

Ladder: Level 2 on 2.1. Verdict: prototype on 2.1, as a target arm of Experiment 6.

---

## Category G. Attention and bottlenecks

This category was placed at Level 0 to 1 in Volume I (attention and global workspace as theory or later) because clean operationalization is the hard part. The dossiers here separate the tractable, testable forms (selective attention, salience, working-memory limits, chunking, context gating, all buildable) from the genuinely theory-stage ones (attention schema, conscious access), and say which is which.

### G1. Selective attention

Biology. [Established neuroscience] Attention selects a subset of available information for enhanced processing, improving signal and suppressing distractors; it is both bottom-up (salience-driven) and top-down (goal-driven), and it gates what enters deeper processing and memory (the broad selective-attention literature; Desimone and Duncan 1995 biased competition).

Computational abstraction. A learned, possibly top-down-modulated weighting over inputs or features that concentrates processing and learning on task-relevant subsets.

Substrate attachment. A goal-conditioned attention mask over dense latent tokens (2.1) or over temporal windows, learned to weight task-relevant content, and used both to focus prediction and to gate what gets written to memory. On pooled 2 the spatial form is unavailable; the temporal form (which frames to attend) still works.

ML analogs. [Established ML] Attention mechanisms broadly (Bahdanau et al. 2015; Transformers), top-down and goal-conditioned attention, hard attention (Mnih et al. 2014).

Feeds. Experiment 6 and the active-sensing lever (E3); a gate on memory writes (B-category).

Failure mode. Learned attention provides no benefit over uniform processing at matched compute, or attends to spurious features. Detection: ablate attention against uniform weighting on task and compute-normalized metrics.

Tractability. Laptop, especially on 2.1.

Developmental role. Information selection. Axis C, links to memory and active sensing.

Dependencies. 2.1-favored for spatial selection; pairs with E3, G2, and memory gating.

Ladder: Level 2. Verdict: build now, especially on 2.1.

### G2. Salience maps

Biology. [Established neuroscience] Bottom-up salience maps highlight conspicuous locations (by contrast, motion, novelty) and guide attention and gaze before any goal is considered; the saliency model (Itti, Koch, Niebur 1998) is a classic operationalization.

Computational abstraction. A scalar field over the input marking conspicuity from low-level features and novelty, used to prioritize processing and sampling bottom-up.

Substrate attachment. Compute a salience field over dense latent tokens from feature contrast, motion, and novelty (RND-style) and use it to drive active sampling (E3) and bottom-up attention (G1); this gives the bottom-up half of attention a concrete, testable form on the frozen latent, complementing the top-down half.

ML analogs. [Established ML] Saliency models (Itti et al. 1998; deep saliency prediction), bottom-up attention in vision, novelty-based saliency.

Feeds. E3 (sampling) and G1 (bottom-up attention).

Failure mode. Salience computed on the latent does not predict task-useful regions, so it misdirects sampling. Detection: compare task gain from salience-driven sampling against random sampling at matched compute.

Tractability. Laptop on cached latents; 2.1-favored.

Developmental role. Bottom-up prioritization. Axis C.

Dependencies. Pairs with E3 and G1. The bottom-up complement to selective attention.

Ladder: Level 2. Verdict: build now on 2.1, as the bottom-up driver for active sensing.

### G3. Attention schema

Biology. [Theory, contested] Attention schema theory (Graziano 2013) proposes the brain builds a simplified internal model of its own attention, used to predict and control attention and possibly underlying subjective reports of awareness. It is a specific, named theory, not consensus neuroscience.

Computational abstraction. A model of the system's own attention state, used to predict and regulate where attention will go, a form of metacognitive self-model restricted to attention.

Substrate attachment. A small module that predicts the system's own attention distribution (from G1) one step ahead and uses that prediction to stabilize or pre-allocate attention; this is implementable in principle but its benefit is speculative and hard to isolate, so it belongs in the theory-or-later tier rather than the build-now tier.

ML analogs. [Sparse] Meta-attention and learning-to-attend, self-modeling networks, predicting one's own activations. Little direct precedent, which is both novelty and risk.

Feeds. A speculative extension of G1; no core experiment depends on it.

Failure mode. No measurable benefit over direct attention control, and the self-model is unfalsifiable as anything beyond a second attention layer. Detection: only meaningful if it beats a non-schema attention controller on a defined task; otherwise it stays theory.

Tractability. Laptop to implement, but low expected payoff.

Developmental role. Attention self-model. Axis C, speculative.

Dependencies. Needs G1. Lowest priority in the category.

Ladder: Level 0 to 1. Verdict: theory or later.

### G4. Conscious access and low-bandwidth bottlenecks

Biology. [Theory, framework-level] Global workspace theory (Baars; Dehaene) proposes a low-bandwidth global broadcast: a small amount of information is selected, made globally available across modules, and this broadcast correlates with conscious access. The bottleneck is the point, not a limitation.

Computational abstraction. A narrow shared channel through which a small set of selected representations is broadcast to all modules, forcing competition for access and serializing certain processing.

Substrate attachment. A narrow shared bottleneck (a small set of slots written by competition and read by all heads) through which only a few latent items pass per step, forcing the system to select what is globally relevant; testable as whether a hard information bottleneck improves generalization or coordination, but the link to anything developmental is loose and the mechanism is theory-stage.

ML analogs. [Established ML] Information bottleneck (Tishby et al.), shared workspace and coordination through bottlenecks (Goyal et al. 2022, "Coordination Among Neural Modules Through a Shared Global Workspace"), slot bottlenecks, the perceiver latent bottleneck (Jaegle et al. 2021).

Feeds. A possible architectural variant for multi-module experiments; no core experiment requires it.

Failure mode. The bottleneck hurts more than it helps, or its benefit is just regularization with no workspace interpretation. Detection: compare against an unbottlenecked architecture and against plain regularization at matched capacity.

Tractability. Laptop to implement; interpretation is the hard part.

Developmental role. Global coordination. Speculative.

Dependencies. Most relevant once there are multiple modules to coordinate (Volume III modularity). Theory-stage here.

Ladder: Level 1. Verdict: theory or later; revisit if modular experiments need coordination.

### G5. Working-memory limits

Biology. [Established cognitive science] Working memory has a small, sharply limited capacity (the classic estimates of a few items; Miller 1956; Cowan's roughly four), and this limit, far from being a pure deficit, may force abstraction, chunking, and efficient coding. The bottleneck shapes representation.

Computational abstraction. A small-capacity, fast-read-write store for currently relevant items, whose limited size pressures the system toward compression and chunking.

Substrate attachment. A small fixed-size recurrent or slot-based working store over latents (a handful of slots) that the predictor and planner read and write, with the capacity limit as a deliberate design variable; test whether a tight limit improves abstraction and generalization, connecting directly to chunking (G6).

ML analogs. [Established ML] Slot-based and external working memories (Neural Turing Machines, Graves et al. 2014; DNC, Graves et al. 2016), fixed-size recurrent state, the working-memory bottleneck as inductive bias.

Feeds. Planning and relational experiments; pairs with chunking (G6).

Failure mode. The capacity limit only hurts (less is just less) with no abstraction benefit. Detection: sweep working-memory capacity and measure both performance and a compression or abstraction metric; the interesting result is a capacity sweet spot, not monotonic improvement with size.

Tractability. Laptop.

Developmental role. Capacity-forced abstraction. Axis C, links to structure.

Dependencies. Pairs with chunking (G6) and with maturational capacity growth (A4). A clean, testable bottleneck.

Ladder: Level 2. Verdict: build now, as a capacity-sweep study.

### G6. Chunking

Biology. [Established cognitive science] Chunking groups individual items into higher-order units, effectively expanding working-memory capacity and enabling expertise (the classic chess-expertise studies of Chase and Simon 1973); it is how limited capacity scales with learning.

Computational abstraction. Learning to group frequently co-occurring elements into reusable units that are then treated as single items, hierarchically compressing sequences and states.

Substrate attachment. Learn recurring latent subsequences or co-occurring token groups as reusable chunks (via a learned codebook, sequence segmentation, or hierarchical prediction in C1) and predict and plan over chunks rather than raw latents, which both relieves the working-memory limit (G5) and builds the hierarchical structure that temporal abstraction needs.

ML analogs. [Established ML] Vector quantization and codebooks (VQ-VAE, van den Oord et al. 2017), byte-pair-style sequence chunking, hierarchical RL option discovery as temporal chunking, neural sequence chunking.

Feeds. Hierarchical prediction (C1), working memory (G5), and temporal abstraction toward options (Volume III).

Failure mode. Discovered chunks are not reused or not predictive, so chunking adds structure without benefit. Detection: measure chunk reuse frequency and downstream prediction or planning gain; failure is low reuse or no gain.

Tractability. Laptop.

Developmental role. Hierarchical compression. Axis C, links to structure and options.

Dependencies. Pairs with G5 and C1; a stepping stone to temporal abstraction and skills.

Ladder: Level 2. Verdict: build now, paired with the working-memory study.

### G7. Context gating

Biology. [Established neuroscience] The brain gates information flow by context: prefrontal cortex maintains task context and gates which inputs and rules are currently relevant, supporting flexible, context-dependent behavior and rapid task switching (prefrontal gating models; the PBWM framework of O'Reilly and Frank 2006).

Computational abstraction. Context-dependent modulation of which inputs, features, or sub-networks are active, allowing the same system to behave differently in different contexts without interference.

Substrate attachment. A context signal (task embedding, inferred task identity, or detected boundary) that gates which heads, adapters, or attention patterns are active, so different tasks use partly different pathways through the shared shell; this directly reduces interference in continual learning and is the mechanistic bridge between task-boundary detection (6.4) and modular routing (Volume III), making it one of the more useful and testable attention-category levers.

ML analogs. [Established ML] Context and task gating (PathNet, Fernando et al. 2017; conditional computation), FiLM modulation (Perez et al. 2018), hypernetwork-generated context-conditioned weights, mixture-of-experts routing by context.

Feeds. Continual-learning experiments (interference reduction) and modular routing (Volume III); pairs with task-boundary detection.

Failure mode. Context inference is wrong, so the gating routes to the wrong pathway and increases interference instead of reducing it. Detection: measure interference and accuracy with oracle context versus inferred context; the gap quantifies how much the gating depends on correct context inference.

Tractability. Laptop.

Developmental role. Context-dependent routing. Axis C, links to structure and consolidation.

Dependencies. Task-boundary detection (6.4); a bridge to modular routing (Volume III). One of the highest-value attention levers because it attacks interference directly.

Ladder: Level 2 to 3. Verdict: build now, as an interference-reduction mechanism in the continual-learning harness.

---

## End of Volume II

Volume II has given one-page treatment to the remaining mechanisms across developmental timing, memory, prediction, neuromodulation, curiosity, relational maps, and attention. The pattern that recurs and is worth stating plainly: a large share of these levers attach cleanly to the frozen perceptual module, run on cached latents on a laptop, and have clean null interpretations, and the ones that do not are almost always the ones that need either an environment (the curiosity, play, and control-mode levers) or dense per-token features (most of the relational and object-centric levers, which is why the 2-versus-2.1 comparison keeps surfacing as a result in its own right). The honest dividing line is not biological plausibility, which is high across the board, but compute feasibility and substrate compatibility, exactly as the ranking doctrine predicts.

Volume III takes the hardware-challenging and open-ended frontier: sparse and structural mechanisms, local learning and dendrites at depth, spiking and neuromorphic computation, and embodiment and open-endedness at depth. These are the levers most likely to be scoped to simulation, toy tests, or a larger program than a solo year, and Volume III says so for each, while still specifying the minimal version that would be informative.
