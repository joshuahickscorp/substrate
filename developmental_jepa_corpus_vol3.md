# Toward Developmental JEPA

## Corpus Volume III: The Hardware-Challenging and Open-Ended Frontier

### Sparsity and Structure, Local Learning and Dendrites, Spiking and Neuromorphic, Embodiment and Open-Endedness

---

## Framing note for Volume III

This volume covers the levers that the Volume I ranking placed lowest on compute feasibility for a solo researcher: structural and sparse mechanisms, biologically faithful local learning rules, spiking and neuromorphic computation, and the open-ended embodiment program. The framing from Volume II holds without change: the object of design is the developmental system, and V-JEPA is one frozen perceptual module inside it, never the system itself. The "Substrate attachment" field describes how each lever bolts onto that frozen module without becoming part of it.

The honesty rule for this volume is sharper than for the others. Several of these levers cannot beat backpropagation at this scale, or need hardware a solo researcher does not have, or need an environment richer than one person can build in a year. The doctrine's instruction is not to reject hard mechanisms but to scope them, so each dossier still specifies the minimal version that would be informative, and then says plainly what compute class it actually needs and what the honest expected outcome is. A lever scored 1 on compute feasibility is not deleted here; it is given a simulation-only or toy-test or lab-scale verdict and a clear statement of why.

Two structural truths organize the volume. First, local-learning and spiking levers are mostly scientific-interest experiments whose realistic outcome is "does not beat backprop, but characterizes the gap," and that is a legitimate result, not a failure. Second, the open-endedness levers are where the program's hardest ceiling lives: development may require an environment rich enough to generate stepping stones, and building that environment, not the algorithm, is the binding constraint. The volume does not pretend otherwise.

Ladder levels refer to the 0-to-6 evidence ladder of Volume I, Section 4. Verdicts use the established vocabulary: build now, prototype, toy test, theory or later, simulation only, lab-scale. No em dashes, per the standing rule.

---

## Category H. Sparse and structural mechanisms

This category extends the structural levers of Volume I (sparse coding and k-winners-take-all, mixture-of-experts, structural pruning, neurogenesis) into the finer mechanisms. The recurring engineering caveat: unstructured sparsity rarely yields wall-clock speedup on a laptop GPU, so most of these are studied for their effect on learning dynamics and interference, not for efficiency, and the corpus accepts a no-speedup result as fine.

### H1. Lateral inhibition and excitation-inhibition balance

Biology. [Established neuroscience] Lateral inhibition sharpens representations by having active units suppress their neighbors, producing contrast enhancement, sparse codes, and competitive selection; balanced excitation and inhibition keeps networks in a stable, high-dynamic-range regime, and the E-I ratio is tightly regulated (the balanced-network literature; center-surround receptive fields).

Computational abstraction. A competitive normalization in which active units suppress others, yielding sparse, decorrelated, high-contrast representations, with a balance term preventing runaway excitation or silence.

Substrate attachment. Add lateral-inhibition or divisive-normalization operations within the predictor and heads (a competition layer over latent features), tuned to a target sparsity and stability, and connect it to the inhibitory-maturation schedule (Volume II A7) so the E-I balance tightens over development. Studied for its effect on representation sparsity, decorrelation, and interference, not for speed.

ML analogs. [Established ML] Local response normalization and divisive normalization, k-winners-take-all layers, lateral-inhibition layers, sparse coding objectives (Olshausen and Field 1996).

Feeds. The sparsity study (H3) and the inhibitory-maturation lever (A7); a representation-quality knob for the continual-learning harness.

Failure mode. The inhibition either silences the network (too strong) or does nothing distinguishable from layernorm (too weak). Detection: sweep inhibition strength and measure sparsity, decorrelation, and downstream interference; report the stable band.

Tractability. Laptop, no speedup expected.

Developmental role. Representation sharpening and stability. Structure, links to Axis A.

Dependencies. Pairs with A7 (maturation schedule) and H3 (sparsity). A mechanism, not a result on its own.

Ladder: Level 2. Verdict: build now, as a representation-quality knob; accept no wall-clock gain.

### H2. Homeostatic plasticity

Biology. [Established neuroscience] Homeostatic plasticity keeps neural activity within a target range despite ongoing Hebbian changes that would otherwise drive runaway potentiation or silence; synaptic scaling (Turrigiano) multiplicatively renormalizes synaptic strengths to stabilize firing rates, operating slowly alongside fast Hebbian learning.

Computational abstraction. A slow stabilizing process that renormalizes weights or activations toward a target activity level, counterbalancing fast learning and preventing drift, distinct from a one-off normalization in that it is continuous and activity-dependent.

Substrate attachment. A slow activity-renormalization term on the predictor and heads (scale weights or gains to hold mean activation near a target), running alongside the fast task learning, and connected to the sleep-like downscaling phase (Volume II B7) as its offline expression. The developmental claim to test: homeostatic renormalization improves long-run stability and reduces forgetting in continual learning over many tasks.

ML analogs. [Established ML] Weight and activation normalization (batch, layer, weight norm), the synaptic-homeostasis hypothesis as an offline downscaling step, adaptive gain control, decorrelation and whitening as stabilizers.

Feeds. The sleep-phase lever (B7) and long-run continual-learning stability.

Failure mode. Renormalization erases the very weight structure that consolidation is trying to protect, trading stability for forgetting. Detection: long-run retention with and without homeostatic scaling, and an interaction test with EWC and replay; failure is improved stability bought with worse retention.

Tractability. Laptop.

Developmental role. Long-run stability. Structure, links to Axis B.

Dependencies. Interacts with consolidation (6.3) and the sleep phase (B7); test the interaction, since stability and retention can trade off.

Ladder: Level 2. Verdict: prototype, with an explicit stability-versus-retention interaction test.

### H3. Structured versus unstructured sparsity, a comparison dossier

Biology. [Established neuroscience] Cortical activity is sparse, and connectivity is sparse and structured (local clustering, specific long-range projections); biological sparsity is both about which units fire (activation sparsity) and which connect (connectivity sparsity), and it is structured, not random.

Computational abstraction. A comparison across sparsity types: unstructured (arbitrary zeros), structured (whole units, channels, or blocks pruned), fine-grained structured (the 2-of-4 pattern hardware can accelerate), and block sparsity, evaluated on learning dynamics, interference, capacity, and, where relevant, wall-clock.

Substrate attachment. Apply each sparsity type to the predictor and heads over the frozen latent and compare on the continual-learning harness: does sparsity reduce interference and improve capacity allocation, and which type does so without destroying performance. The honest framing is that only fine-grained structured (2-of-4) and block sparsity give wall-clock benefit on current hardware, while unstructured sparsity is studied purely for its learning-dynamics effect.

ML analogs. [Established ML] Magnitude and movement pruning (Han et al. 2015; Sanh et al. 2020), the lottery-ticket hypothesis (Frankle and Carbin 2019), 2-of-4 fine-grained structured sparsity on Ampere-class sparse tensor cores, block-sparse kernels, structured pruning of channels and heads.

Feeds. The structural-pruning lever (Volume I) and forgetting-as-pruning (B6); a capacity-allocation study for continual learning.

Failure mode. Sparsity at this scale gives neither interference reduction nor speedup, only accuracy loss. Detection: the comparison itself is the test; report a Pareto front of sparsity type against accuracy, interference, and wall-clock, and accept a null (no benefit) as informative.

Tractability. Laptop, with the explicit caveat that only structured forms accelerate on the M3 GPU.

Developmental role. Capacity allocation and interference control. Structure.

Dependencies. Connects B6 (forgetting as pruning) and the Volume I pruning lever. A comparison study, not a single mechanism.

Ladder: Level 2. Verdict: build now, framed as a comparison with an accepted-null outcome.

### H4. Neural reuse

Biology. [Established cognitive neuroscience] Neural reuse (Anderson 2010) is the principle that the same neural circuits are redeployed across many cognitive functions, so brain regions are typically used by multiple, sometimes unrelated, tasks; cognition is built largely by reusing and recombining existing circuits rather than growing dedicated new ones.

Computational abstraction. Sharing and recombining existing modules or parameters across tasks, maximizing reuse rather than allocating fresh capacity per task, the opposite design pressure from strict modular isolation.

Substrate attachment. Measure and encourage reuse of predictor and head sub-modules across the task stream (shared adapters, parameter sharing with task-specific gating via context gating G7), and study the tradeoff between reuse (efficient, risks interference) and isolation (interference-free, risks capacity blow-up). This is the design tension at the heart of continual learning, framed biologically.

ML analogs. [Established ML] Parameter sharing in multitask learning, modular and reusable sub-networks (PathNet, Fernando et al. 2017), adapters and LoRA-style shared-plus-specific decomposition, soft and hard parameter sharing.

Feeds. The modular-routing lever (this category, H6, and Volume I MoE) and context gating (G7); the central reuse-versus-isolation study.

Failure mode. Reuse causes interference (shared circuits clash across tasks) or isolation causes uncontrolled growth; the lever is the study of that tradeoff, so the failure of one extreme is the expected finding. Detection: sweep the reuse-isolation axis and plot interference against parameter growth; the result is a frontier, not a point.

Tractability. Laptop.

Developmental role. Capacity reuse versus isolation. Structure, links to Axis B.

Dependencies. Pairs with G7 (context gating) and MoE routing; defines the axis that module birth (H6) and pruning sit on.

Ladder: Level 2. Verdict: build now, as the reuse-versus-isolation frontier study.

### H5. Degeneracy

Biology. [Established biology] Degeneracy (Edelman and Gally 2001) is the ability of structurally different elements to perform the same function, pervasive in biological systems and a major source of robustness and evolvability; multiple distinct circuits can produce the same behavior, so damage to one is compensated by others.

Computational abstraction. Maintaining multiple, structurally different solutions to the same function, providing robustness to damage and a substrate for variation, distinct from redundancy (identical copies) in that the solutions differ.

Substrate attachment. Encourage and measure degeneracy in the predictor by training redundant-but-different sub-solutions (an ensemble with diversity pressure, or dropout-induced solution multiplicity) and test robustness to ablation and to distribution shift; degeneracy connects to the ensemble-disagreement uncertainty signal of 6.4, since diverse solutions are what make disagreement a meaningful epistemic measure.

ML analogs. [Established ML] Deep ensembles and their diversity (Lakshminarayanan et al. 2017), dropout as implicit ensembling, diversity-regularized ensembles, the degeneracy-robustness link in network science.

Feeds. The uncertainty signal (6.4) and robustness evaluation; a quality property rather than a standalone experiment.

Failure mode. Enforced degeneracy costs capacity and accuracy for robustness that the task does not need. Detection: measure robustness gain (under ablation and shift) against the capacity and accuracy cost; failure is cost without robustness payoff.

Tractability. Laptop.

Developmental role. Robustness and variation substrate. Structure, links to uncertainty.

Dependencies. Underlies meaningful ensemble disagreement (6.4); pairs with the uncertainty-gating experiments.

Ladder: Level 2. Verdict: prototype, as a robustness property of the uncertainty ensemble.

### H6. Module birth and death dynamics

Biology. [Established neuroscience] Adult neurogenesis (notably in the hippocampal dentate gyrus) adds new neurons throughout life, implicated in pattern separation and the encoding of new memories, while synaptic and cellular pruning removes unused structure; the brain grows and removes structure adaptively, not just reweights fixed structure.

Computational abstraction. Dynamically adding capacity (new units, modules, or experts) when current capacity is insufficient and removing it when redundant, so architecture adapts to the task stream rather than being fixed in advance.

Substrate attachment. A growable predictor or head that adds modules or experts on a trigger (high persistent error, a detected new task via 6.4) and prunes them on redundancy (forgetting-as-pruning B6), realizing structural development on the trainable shell while the encoder stays fixed. The developmental claim: a system that grows and prunes structure adapts to a long task stream better than a fixed-capacity system on the adaptation-retention frontier.

ML analogs. [Established ML] Progressive networks (Rusu et al. 2016), dynamically expandable networks (Yoon et al. 2018), growing and pruning architectures, expert addition in mixtures of experts, net2net-style growth (Chen et al. 2016).

Feeds. The neurogenesis lever (Volume I) and the reuse-versus-isolation study (H4); a structural-development experiment.

Failure mode. Uncontrolled growth (capacity explosion, the trivial way to avoid forgetting) or over-aggressive pruning (forgetting through removal). Detection: track parameter count alongside the adaptation-retention frontier; the honest comparison fixes a parameter budget so growth cannot win by simply adding capacity without bound.

Tractability. Laptop, with growable-network tooling.

Developmental role. Structural development. Structure apex, links to Axis B.

Dependencies. Needs a growth trigger (6.4) and a pruning policy (B6); sits on the reuse-isolation axis of H4. Compare under a fixed parameter budget to keep the result honest.

Ladder: Level 2. Verdict: prototype, under a fixed-budget comparison.

---

## Category I. Local learning and dendrites at depth

This category extends the local-learning levers of Volume I (Hebbian and three-factor rules, dendritic predictors, forward-forward, equilibrium propagation, feedback alignment) into the backpropagation-alternatives program. The shared honest expectation: at this scale, on a frozen encoder with a small trainable head, none of these is likely to beat backpropagation on accuracy, and the legitimate goal is to characterize the gap, the biological plausibility, and any locality or memory advantage. The corpus treats "does not beat backprop, here is exactly how it falls short and what it buys" as a real result.

### I1. Target propagation

Biology. [Biologically motivated alternative] Target propagation addresses the biological implausibility of backpropagation (the weight-transport problem, the need for a separate backward pass with symmetric weights) by propagating target activations backward through learned inverse mappings rather than gradients, so each layer learns to produce a target rather than to follow a transported error signal.

Computational abstraction. Each layer learns a local inverse and is trained toward a layer-local target computed from the layer above, replacing global gradient transport with local target matching.

Substrate attachment. Apply target propagation to train the trainable stack on the frozen latent, on a small supervised head where the comparison to backprop is clean, and measure the accuracy gap, the locality of the learning signal, and stability. The frozen encoder makes this a controlled testbed: only the shallow shell is trained, so the alternative rule is tested in isolation rather than confounded with deep feature learning.

ML analogs. [Established ML] Difference target propagation (Lee et al. 2015), target propagation variants in the broader biologically-plausible-learning literature (Bengio's program).

Feeds. The backprop-alternatives comparison (I4); a biological-plausibility experiment.

Failure mode. Target propagation underperforms backprop and is unstable to train, the common finding for deeper or harder settings. Detection: matched comparison to backprop on the same head and data; report the gap and the stability, not a claimed win.

Tractability. Laptop on a small head.

Developmental role. Biologically plausible credit assignment. Local learning.

Dependencies. Part of the I4 comparison; isolate on a small head first.

Ladder: Level 2. Verdict: toy test, within the I4 comparison.

### I2. Synthetic gradients and decoupled learning

Biology. [Engineering-motivated, loose biological gloss] Synthetic gradients decouple layers by having each predict its own incoming gradient locally, removing the need to wait for a full backward pass; the loose biological gloss is local, asynchronous learning without a global lockstep backward sweep.

Computational abstraction. Local modules predict the gradient (or a learning signal) they would have received from a global backward pass, allowing asynchronous, decoupled updates and removing the backward-pass lock.

Substrate attachment. Use synthetic-gradient modules to train parts of the trainable shell asynchronously, and study whether decoupling buys anything (parallelism, online adaptation) at the cost of the known accuracy penalty; on the frozen-encoder setup the relevant question is whether decoupled learning helps the continual, online setting where waiting for a full backward pass is itself a constraint.

ML analogs. [Established ML] Decoupled neural interfaces and synthetic gradients (Jaderberg et al. 2017), local-learning-signal methods, predicted-gradient approaches.

Feeds. The backprop-alternatives comparison (I4); an online and asynchronous-learning experiment.

Failure mode. Synthetic gradients are inaccurate, so training is slower or worse with no offsetting benefit at this scale. Detection: matched comparison to backprop, plus a specific test of any online or decoupling advantage; failure is a penalty with no compensating gain.

Tractability. Laptop.

Developmental role. Decoupled, asynchronous learning. Local learning.

Dependencies. Part of I4; the online-setting angle is the only place it might earn its keep.

Ladder: Level 2. Verdict: toy test, within I4, with an online-advantage probe.

### I3. Energy-based learning

Biology. [Biologically motivated framework] Energy-based models define learning and inference as the settling of a network toward low-energy states, a framing compatible with attractor dynamics and with local, physically plausible update rules; Hopfield networks and equilibrium-based learning sit here, and the appeal is that inference and learning can both be local relaxation processes.

Computational abstraction. Define an energy over states and parameters; inference minimizes energy over states, learning adjusts parameters to lower energy at desired configurations, with updates that can be local (the contrastive, two-phase form).

Substrate attachment. Build a small energy-based head over the frozen latent (a modern Hopfield-style associative memory, or an equilibrium-propagation-trained head) and use it for associative retrieval (a complement to the episodic buffer) and for prediction, comparing convergence, capacity, and accuracy to a feedforward head. Energy-based associative memory is a natural fit for the memory-indexing lever (B4), since content-addressable recall is exactly what associative attractors provide.

ML analogs. [Established ML] Hopfield networks and modern Hopfield networks (Ramsauer et al. 2020), equilibrium propagation (Scellier and Bengio 2017), energy-based models (LeCun et al. tutorial), the broad associative-memory line.

Feeds. The backprop-alternatives comparison (I4) and the memory-indexing lever (B4); an associative-memory experiment.

Failure mode. Slow convergence, limited capacity, or training instability relative to a feedforward head. Detection: compare retrieval capacity and prediction accuracy and settling cost against a feedforward baseline; report the tradeoff.

Tractability. Laptop on a small head.

Developmental role. Associative memory and energy-based inference. Local learning, links to Axis B.

Dependencies. Connects to B4 (memory indexing) and to equilibrium propagation in I4. The associative-memory angle is its strongest practical use.

Ladder: Level 2. Verdict: toy test, with the associative-memory use as the most promising angle.

### I4. The backprop-alternatives comparison, a unifying experiment

Biology. [Established as a motivation] Backpropagation is biologically implausible for several specific reasons (weight transport, the need for a distinct backward pass, non-local error signals, update locking), and a family of alternatives, feedback alignment, forward-forward, equilibrium propagation, target propagation, predictive-coding approximations, each relaxes a different one of these constraints. The scientific value is in comparing them on equal footing.

Computational abstraction. A single controlled benchmark that trains the same small head on the same frozen-latent task with each learning rule, measuring accuracy gap to backprop, biological plausibility (which constraints each relaxes), locality, memory and compute cost, and stability.

Substrate attachment. The frozen encoder is the ideal controlled setting for this comparison precisely because it removes deep feature learning as a confound: every rule trains the same shallow shell on identical cached latents, so differences are attributable to the rule, not to representation learning. This is the cleanest contribution this category can make and plays directly to a benchmarking strength.

ML analogs. [Established ML] Feedback alignment and direct feedback alignment (Lillicrap et al. 2016; Nokland 2016), forward-forward (Hinton 2022), equilibrium propagation (Scellier and Bengio 2017), difference target propagation (Lee et al. 2015), predictive-coding approximations to backprop (Whittington and Bogacz 2017; Millidge et al.).

Feeds. This is the integrating experiment for I1, I2, I3 and the Volume I local-learning levers; a standalone methods contribution.

Failure mode. The expected and acceptable result is that backprop wins on accuracy and the alternatives differ in plausibility, locality, and cost; the failure would be running it uncontrolled so differences are not attributable. Detection: fix the head, data, seeds, and budget across all rules; the controlled design is what makes the null informative.

Tractability. Laptop on a small head; this is the rare local-learning experiment that is both feasible and genuinely useful as a clean comparison.

Developmental role. Methods map of biologically plausible learning. Local learning capstone.

Dependencies. Subsumes I1, I2, I3 and the Volume I local-learning levers into one benchmark. The highest-value item in the category for a solo researcher.

Ladder: Level 2 to 4 (Level 4 once it is a controlled experiment). Verdict: build now, as a controlled comparison; accept backprop winning as the informative result.

---

## Category J. Spiking and neuromorphic computation

This is the lowest-feasibility category for a solo researcher, scored 1 on compute in Volume I because faithful spiking computation needs either neuromorphic hardware or expensive simulation, and the frozen continuous-valued V-JEPA latent is a poor match for spike-based processing. Every dossier here is simulation-only or hardware-gated, and the corpus says so. The category is included because the doctrine forbids dropping hard mechanisms; it is scoped, not pursued.

### J1. Spike-timing-dependent plasticity

Biology. [Established neuroscience] Spike-timing-dependent plasticity adjusts synaptic strength based on the relative timing of pre- and post-synaptic spikes: pre-before-post strengthens, post-before-pre weakens, on a millisecond timescale (Bi and Poo 1998). It is a leading candidate for a biological local learning rule and is inherently temporal and event-based.

Computational abstraction. A local, timing-dependent Hebbian rule operating on spike events, where the sign and magnitude of the weight change depend on precise relative spike times.

Substrate attachment. Only meaningful if the continuous latent is converted to spikes (rate or temporal coding) and processed by a spiking head, which is a large, somewhat artificial transformation of the frozen latent; the realistic scope is a simulation of a small spiking head trained with STDP on encoded latents, as a curiosity, not a competitive learner. The mismatch between continuous dense latents and spike timing is itself worth stating as a substrate-level limitation.

ML analogs. [Established ML] STDP in spiking neural network simulators (Brian2, BindsNET), surrogate-gradient training of spiking nets (Neftci et al. 2019) as the practical alternative to STDP, conversion of trained ANNs to SNNs.

Feeds. Nothing in the core program; a simulation-only side experiment.

Failure mode. STDP underperforms badly at this scale and the latent-to-spike conversion is lossy and artificial. Detection: if attempted, compare a surrogate-gradient SNN to an STDP-trained one on a small task; the expected result is that neither is competitive and STDP is worse.

Tractability. Simulation only on a laptop (slow), or neuromorphic hardware not available to a solo researcher.

Developmental role. Biological local learning, mostly out of scope here. Spiking.

Dependencies. Requires a spiking head and a latent-to-spike encoding. Lowest priority in the corpus.

Ladder: Level 0 to 1 in this context. Verdict: simulation only; a scoped curiosity, not a program.

### J2. Event-based computation

Biology. [Established neuroscience] Neural computation is event-driven and asynchronous: neurons fire sparsely in time and compute only when inputs arrive, giving energy efficiency and natural temporal sensitivity, unlike the dense, clocked computation of standard hardware.

Computational abstraction. Asynchronous, sparse-in-time computation triggered by events rather than a global clock, processing change rather than re-processing static input every step, with potential efficiency gains on matching hardware.

Substrate attachment. The closest tractable analog on standard hardware is processing only latent changes between frames (skip computation when the scene is static, compute on change), which captures the efficiency intuition without true event-based hardware; full event-based processing needs an event camera and neuromorphic substrate that sit outside the frozen-V-JEPA pipeline entirely.

ML analogs. [Established ML] Event cameras and event-based vision (the DVS line), asynchronous and sparse computation, change-based and delta-network processing (Neil et al. delta networks), temporal sparsity exploitation.

Feeds. A possible efficiency variant of the predictor (compute on change); not required by any core experiment.

Failure mode. On standard hardware the change-based version gives little real efficiency benefit, and true event-based processing is incompatible with the frozen continuous encoder. Detection: measure compute saved by change-gating against accuracy cost; the honest expectation is modest savings at best.

Tractability. The change-gating analog is laptop-feasible; true event-based computation is hardware-gated and out of scope.

Developmental role. Efficiency and temporal sparsity. Spiking, mostly out of scope.

Dependencies. The frozen continuous latent is the obstacle; only the change-gating shadow of this lever is reachable.

Ladder: Level 1. Verdict: simulation only for the change-gating analog; the full lever is out of scope.

### J3. Neuromorphic deployment

Biology. [Hardware-inspired] Neuromorphic chips (Loihi, SpiNNaker, and successors) implement spiking, event-driven, massively parallel computation in hardware, promising large energy-efficiency gains for the right workloads; they are a deployment target, not a learning mechanism per se.

Computational abstraction. Mapping a trained spiking or event-based system onto neuromorphic hardware for efficient inference, accepting the constraints of spike-based, low-precision, event-driven computation in exchange for energy efficiency.

Substrate attachment. This lever has essentially no attachment to the frozen-V-JEPA developmental program for a solo researcher: it would require a spiking version of the whole system, neuromorphic hardware access, and a workload that justifies the conversion, none of which the program has or needs. It is documented as a real frontier and explicitly placed outside the solo scope.

ML analogs. [Established hardware] Loihi and the Intel neuromorphic research community, SpiNNaker, the neuromorphic-deployment literature; ANN-to-SNN conversion as the bridge.

Feeds. Nothing in the core program; documented for completeness.

Failure mode. Not applicable; the lever is out of scope rather than failure-prone. The honest statement is that pursuing it would consume the entire program for a deployment concern that the research questions do not require.

Tractability. Requires neuromorphic hardware; out of reach and out of scope for a solo researcher.

Developmental role. Deployment efficiency, not development. Spiking.

Dependencies. Would require all of Category J first. Lowest feasibility in the corpus.

Ladder: Level 0 in this context. Verdict: out of scope; documented, not pursued.

---

## Category K. Embodiment and open-endedness at depth

This category extends the open-endedness levers of Volume I (open-ended environment generation, quality diversity, autotelic goal generation, language as scaffolding) into the full embodiment and cultural program. It is the category where the program's hardest ceiling lives, and the dossiers are honest about it: development of genuinely new competence may require an environment rich enough to generate an endless supply of stepping stones, and for a solo researcher the binding constraint is building that environment, not designing the algorithm. Most verdicts here are lab-scale or later, with the minimal informative version specified and the environment requirement stated plainly. The action-conditioned predictor and the language-aligned head (the V-JEPA encoder aligned to a language model) are the two substrate features these levers lean on most.

### K1. Affordances

Biology. [Established in ecological psychology] Gibson's affordances (1979) are the action possibilities an environment offers an agent relative to its body and skills: a surface affords walking, a handle affords grasping. Perception is geared to picking up affordances directly, not to building a neutral world model first; what an agent perceives is shaped by what it can do.

Computational abstraction. Representing states in terms of the actions they enable rather than in purely perceptual terms, so the world model predicts not just what will happen but what can be done.

Substrate attachment. Learn an affordance head over the frozen latent (in an environment) that predicts which actions are available or likely to succeed from the current latent state, and condition the planner on affordances rather than on raw latents; on the action-conditioned substrate this is a natural auxiliary that grounds perception in action. Without an environment there are no affordances to learn, so this lever is environment-gated by definition.

ML analogs. [Established ML] Affordance learning in robotics and RL, action-conditioned models that predict feasibility, the grounding of perception in action (the SayCan-style affordance grounding, Ahn et al. 2022, where a value function estimates action feasibility).

Feeds. Goal-directed control (D4), tool use (K2), and the minimal open-ended experiment (Volume I Experiment 10).

Failure mode. Affordances collapse to a relabeling of the value function with no added structure, or cannot be learned because the latent does not expose action-relevant features. Detection: test whether affordance prediction improves planning over a plain value baseline; failure is no improvement.

Tractability. Single-GPU and an environment.

Developmental role. Action-grounded perception. Embodiment, links to action.

Dependencies. Environment-gated; builds on the action-conditioned predictor. A prerequisite framing for tool use.

Ladder: Level 2. Verdict: lab-scale; environment required.

### K2. Tool use

Biology. [Established in comparative cognition] Tool use, extending the body with external objects to achieve goals, appears across species (primates, corvids, otters) and is a marker of flexible problem-solving; it requires representing how an external object changes the agent's action possibilities, an extension of affordance perception to manipulable objects.

Computational abstraction. Incorporating external objects into the agent's action repertoire, modeling how using an object transforms what the agent can do, a compositional extension of affordances.

Substrate attachment. In an environment with manipulable objects, learn that certain objects extend the action space (tool latent plus body latent yields new affordances) and plan over tool-augmented action possibilities; this needs both a rich environment and the affordance head (K1), and is squarely a lab-scale embodied-RL problem rather than anything reachable on cached latents alone.

ML analogs. [Established ML] Tool use in embodied RL and robotics, object-as-action-extension models, compositional skill and affordance learning, emergent tool use in multi-agent settings (the hide-and-seek line, Baker et al. 2020).

Feeds. Skill libraries (K4) and the open-ended experiment; an advanced embodied capability.

Failure mode. The environment is not rich enough to make tools useful, so no tool-use behavior emerges. Detection: whether tool-using solutions appear and transfer; absence indicates an environment too simple to require tools.

Tractability. Single-GPU and a rich environment with manipulable objects.

Developmental role. Compositional action extension. Embodiment.

Dependencies. Needs K1 and a rich environment. Among the more demanding levers in the corpus.

Ladder: Level 2. Verdict: lab-scale; rich environment required.

### K3. Options and hierarchical reinforcement learning

Biology. [Established as a computational framework with biological resonance] Behavior is organized hierarchically into temporally extended sub-behaviors (reach, grasp, walk-to) composed into longer plans; the brain appears to use hierarchical control, and the options framework formalizes temporally extended actions with their own initiation conditions, internal policies, and termination.

Computational abstraction. Temporally extended actions (options or skills) with initiation sets, internal policies, and termination conditions, composed by a higher-level policy, enabling planning and credit assignment over long horizons by operating on skills rather than primitive actions.

Substrate attachment. Discover and learn options over the latent state in an environment (option initiation and termination defined in latent space, internal policies acting through the action-conditioned predictor), composed by a manager policy; temporal abstraction is also the principled answer to the substrate's short-horizon rollout limit, since planning over options covers more time per step than planning over primitive latents. This is the single most strategically important open-endedness lever because it directly addresses the horizon ceiling.

ML analogs. [Established ML] The options framework (Sutton, Precup, Singh 1999), option-critic (Bacon, Harb, Precup 2017), feudal RL (Dayan and Hinton 1993) and FeUdal Networks (Vezhnevets et al. 2017), MAXQ value decomposition (Dietterich 2000), the broad hierarchical-RL literature.

Feeds. Skill libraries (K4), the open-ended experiment, and it relieves the short-horizon planning limit identified throughout the substrate analysis.

Failure mode. Discovered options are degenerate (collapse to primitives or to a single option) or unhelpful, the well-documented option-collapse problem. Detection: measure option diversity, reuse, and the horizon extension they provide; failure is collapse or no horizon benefit.

Tractability. Single-GPU and an environment.

Developmental role. Temporal abstraction. Embodiment and open-endedness, the horizon fix.

Dependencies. Environment-gated; builds on the action-conditioned predictor; pairs with chunking (G6) as its representational precursor. The highest-leverage lever in this category.

Ladder: Level 2 to 3. Verdict: lab-scale, high priority within the open-ended track; environment required.

### K4. Skill libraries

Biology. [Established in developmental science] Competence accumulates as a growing repertoire of reusable skills, each built on earlier ones, so development is partly the construction and indexing of a skill library that later learning draws on; new skills are composed from old.

Computational abstraction. A growing, indexed store of learned skills (options, policies, or programs) that can be retrieved, composed, and refined, so the agent builds on its own prior competence rather than relearning from scratch.

Substrate attachment. Maintain a library of learned options or skills (K3) indexed by the latent states or goals they apply to, retrieved by similarity (the memory-indexing machinery of B4 reused for skills) and composed for new tasks; this turns the episodic and option machinery into a cumulative competence store. The clearest realization in the literature is an agent that writes, stores, and reuses skills in an open-ended environment, and the substrate's language-aligned head makes language-indexed skills feasible.

ML analogs. [Established ML] Voyager (Wang et al. 2023), which builds and reuses a skill library via an LLM in an open-ended environment; skill-discovery and skill-composition methods; library learning in program synthesis (DreamCoder, Ellis et al. 2021, as a non-embodied cousin).

Feeds. The open-ended experiment and play (Volume II E5); the cumulative-competence backbone.

Failure mode. Skills are not reusable or not composable, so the library grows without improving new-task learning. Detection: measure whether library size correlates with faster acquisition of held-out tasks; no correlation means the library is inert.

Tractability. Single-GPU and an environment; the indexing reuses laptop-feasible machinery.

Developmental role. Cumulative competence. Open-endedness backbone.

Dependencies. Needs K3 (skills to store) and B4 (indexing). The point where memory and options and language converge.

Ladder: Level 2. Verdict: lab-scale; environment required, but the indexing design is reachable now.

### K5. Social learning and imitation

Biology. [Established in developmental and comparative science] Much learning is social: imitation, emulation, and observation let an agent acquire behaviors from others far faster than individual trial and error, and social learning is a major accelerant of development and the substrate of culture; infants are powerful imitators.

Computational abstraction. Acquiring behavior by observing and reproducing the behavior of another agent, via imitation (copy the actions), emulation (copy the outcomes), or inferred-goal imitation, short-cutting individual exploration.

Substrate attachment. Use the frozen encoder to perceive a demonstrator's behavior as latent trajectories and train a policy to reproduce them (latent-space imitation, or inverse RL over latents), then refine with the agent's own intrinsic motivation; because perception is shared and frozen, demonstrator and learner latents live in the same space, which is a genuine advantage for imitation. Needs demonstrations, hence an environment with a demonstrator or a dataset of demonstrations.

ML analogs. [Established ML] Imitation learning and behavioral cloning, inverse reinforcement learning, learning from observation (third-person and observational imitation), generative adversarial imitation (Ho and Ermon 2016).

Feeds. Teacher-student learning (K8), cultural accumulation (K7), and the open-ended experiment; an exploration accelerant.

Failure mode. The agent imitates surface behavior without grasping goals, failing to generalize (the correspondence and goal-inference problem). Detection: test transfer of imitated behavior to varied situations; surface imitation fails to transfer.

Tractability. Single-GPU and demonstrations or a demonstrator environment.

Developmental role. Social acquisition. Open-endedness, links to culture.

Dependencies. Shared frozen perception aids it; needs demonstrations. Precursor to teacher-student and cultural levers.

Ladder: Level 2. Verdict: lab-scale; demonstrations required, with the shared-latent advantage worth exploiting.

### K6. Language as developmental instruction

Biology. [Established in developmental science] Language is a primary channel for transmitting knowledge and structuring cognition in human development: instructions, labels, and explanations let children acquire concepts and skills they could not easily discover alone, and language scaffolds thought (the Vygotskian view) as well as transmitting it.

Computational abstraction. Using language as an interface for specifying goals, providing instruction, decomposing tasks, and labeling structure, so a language source can direct and accelerate the agent's learning and behavior.

Substrate attachment. This is the lever the substrate is unusually well-positioned for, because the V-JEPA encoder can be aligned to a language model, giving a shared visual-language space. Use language to specify goals for goal-directed control (D4), to name and index skills in the library (K4), to decompose tasks into option sequences (K3), and to provide a curriculum; an LLM can propose goals and sub-tasks grounded in the visual latent. The language-aligned head turns language from an external add-on into a native interface to the perceptual space.

ML analogs. [Established ML] Language-conditioned policies and instruction following, SayCan (Ahn et al. 2022) grounding language in affordances, LLM-driven agents that plan and propose goals (the Voyager pattern, Wang et al. 2023), language as an abstraction for hierarchical RL.

Feeds. Goal-directed control (D4), skill libraries (K4), options (K3), and the autotelic-goal and language-scaffolding levers of Volume I.

Failure mode. Language is ungrounded, so instructions do not connect to latent states and actions, or the LLM proposes goals the agent cannot perceive or achieve. Detection: measure grounding (do language goals map to achievable latent states) and whether language instruction improves learning over no-language baselines.

Tractability. The alignment and language interface are reachable; full language-directed open-ended learning needs an environment. Mid-program for the interface, lab-scale for the full loop.

Developmental role. Instructed development. Open-endedness, the substrate's strongest open-ended asset.

Dependencies. The language-aligned head; pairs with K3, K4, D4. The most distinctive open-endedness opportunity this substrate offers.

Ladder: Level 2 to 3. Verdict: prototype the language-goal interface now; the full instructed-learning loop is lab-scale.

### K7. Cultural accumulation

Biology. [Established in cultural evolution] Human competence accumulates across generations through cumulative culture: each generation inherits and slightly improves on the previous one's knowledge and tools, the ratchet effect (Tomasello), so capability grows beyond what any individual could invent, driven by high-fidelity social transmission plus innovation.

Computational abstraction. Iterated transmission and improvement of knowledge or skills across successive agents or generations, where each generation learns from the last and adds to it, producing open-ended capability growth that no single learning run achieves.

Substrate attachment. Run successive agent generations that learn from the prior generation's skill library (K4) and demonstrations (K5), each adding new skills, with the frozen shared perception ensuring transmitted skills remain interpretable across generations; this is a multi-agent, multi-generation program that needs a rich environment and substantial compute, and is honestly beyond a solo year except in a minimal toy form. It is included because it is the deepest form of open-endedness and the place where the program's ambition meets its hardest resource ceiling.

ML analogs. [Established ML] Cultural evolution and accumulation in multi-agent learning, iterated learning (the iterated-learning models of language emergence), generational training, the ratchet effect in machine culture (the cultural-accumulation line in deep RL).

Feeds. The far horizon of the open-ended program; nothing in the near-term experiment bank depends on it.

Failure mode. Transmission fidelity is too low (skills degrade across generations) or innovation is absent (no improvement), so no ratchet forms. Detection: measure capability across generations; a flat or declining curve means no cumulative culture.

Tractability. Multi-agent, multi-generation, rich environment: the highest compute demand in the corpus, lab-scale at best.

Developmental role. Cross-generational accumulation. Open-endedness apex.

Dependencies. Needs K4, K5, and a rich persistent environment. The clearest case where the ceiling is resources, not ideas.

Ladder: Level 2, aspiring to Level 6. Verdict: lab-scale; a minimal toy version is the most a solo program could attempt.

### K8. Teacher-student learning

Biology. [Established in developmental science] Active teaching, where a more competent agent shapes a less competent one's learning through scaffolded instruction, demonstration, and feedback, accelerates acquisition beyond passive observation, and the teacher's adaptation to the student's state (staying in the zone of proximal development) is itself a skill.

Computational abstraction. A teacher agent that adapts its instruction, demonstrations, or generated tasks to the student's current competence, optimizing the student's learning, a two-agent loop coupling a curriculum-generating teacher to a learning student.

Substrate attachment. Pair a teacher (which can be an LLM via the language-aligned head, or a more competent prior-generation agent) that generates goals, demonstrations, or difficulty-tuned tasks with a student agent learning over the frozen latent, with the teacher adapting to the student's measured progress (the scaffolding and ZPD levers, Volume II A3, made into an explicit second agent). This connects automatic curricula, social learning, and language instruction into one loop and is partly reachable: the teacher-as-task-generator over cached-latent tasks needs no environment, while teacher-demonstration in an embodied setting does.

ML analogs. [Established ML] Teacher-student curricula and automatic curriculum generation, PAIRED-style adversarial environment design (Dennis et al. 2020) where a teacher generates challenges at the frontier, asymmetric self-play (the AlphaStar-league and the asymmetric self-play of Sukhbaatar et al. 2018), LLM-as-teacher task generation.

Feeds. The self-curriculum (Volume I Experiment 5), scaffolding (A3), and language instruction (K6).

Failure mode. The teacher generates tasks that are too easy, too hard, or off-distribution, so the student does not improve; the teacher fails to track the student's frontier. Detection: measure student learning rate under the adaptive teacher versus a fixed curriculum; failure is no improvement over fixed.

Tractability. The task-generating teacher over cached latents is laptop-feasible; the embodied demonstrating teacher is lab-scale.

Developmental role. Active instruction. Open-endedness and curriculum.

Dependencies. Builds on A3, K5, K6. The task-generation form is one of the more reachable open-ended experiments.

Ladder: Level 2. Verdict: prototype the task-generating teacher now; the embodied teacher is lab-scale.

### K9. POET-style environment generation

Biology. [No direct biological analog; an open-endedness mechanism] There is no biological mechanism here; this lever comes from open-ended machine learning and addresses the ceiling the rest of the category keeps hitting: where do endless stepping-stone challenges come from. The answer is to generate the environment and the agent together, co-evolving problems and solutions so that newly solvable challenges keep appearing.

Computational abstraction. Co-evolving a population of environments and a population of agents, with environments selected for being newly solvable (at the frontier of agent capability) and solutions transferred across environments, producing an open-ended curriculum that the system generates for itself.

Substrate attachment. Generate a space of tasks or environments parameterized so difficulty and structure can be varied, and co-evolve them with the agent learning over the frozen latent, transferring skills (K4) across environments; this is the explicit machinery for open-ended self-curriculum and is the most compute-hungry single lever after cultural accumulation, needing a parameterizable environment and population-scale training. It is the principled answer to the stepping-stone problem and the honest reason genuine open-endedness is lab-scale for a solo researcher.

ML analogs. [Established ML] POET and Enhanced POET (Wang et al. 2019, 2020), PAIRED (Dennis et al. 2020), unsupervised environment design broadly, MAP-Elites for quality-diversity environment and solution archives (Mouret and Clune 2015), novelty search (Lehman and Stanley 2011).

Feeds. The minimal open-ended experiment (Volume I Experiment 10) is the scoped version; the full lever is the open-ended program's engine.

Failure mode. Generated environments are not meaningfully diverse or not at the frontier, so no open-ended progression occurs and the population stagnates. Detection: measure environment diversity and the appearance of newly-solved challenges over time; stagnation means the generator is not producing useful stepping stones.

Tractability. Population-scale training and a parameterizable environment: among the highest compute demands in the corpus, lab-scale.

Developmental role. Self-generated open-ended curriculum. Open-endedness engine.

Dependencies. Needs a parameterizable environment and skill transfer (K4). With cultural accumulation (K7), the deepest and most resource-bound lever in the program.

Ladder: Level 2, aspiring to Level 6. Verdict: lab-scale; Experiment 10 is the minimal informative version a solo researcher can run.

---

## End of Volume III, and the close of the corpus

Volume III has scoped the hardware-challenging and open-ended frontier: sparse and structural mechanisms, biologically faithful local learning, spiking and neuromorphic computation, and the embodiment and open-endedness program. Three honest conclusions close the volume and the corpus.

First, on structure and local learning. The structural levers (sparsity, reuse, degeneracy, module birth and death) are laptop-feasible and worth running for their effect on interference and capacity allocation, with the standing caveat that unstructured sparsity buys learning-dynamics insight rather than speed. The local-learning levers culminate in one genuinely valuable experiment, the controlled backprop-alternatives comparison (I4), which the frozen encoder makes unusually clean by removing deep feature learning as a confound; its expected result, that backpropagation wins on accuracy while the alternatives differ in plausibility and locality and cost, is a real contribution and a natural fit for a benchmarking sensibility. The spiking and neuromorphic category is the one place the corpus says stop: the continuous frozen latent is a poor match for spikes, the hardware is out of reach, and these levers are documented and scoped out rather than pursued.

Second, on open-endedness. This is where the program's ambition meets its hardest ceiling, and the ceiling is not algorithmic. The options and skill-library and language-instruction levers are the high-leverage ones, options especially because temporal abstraction is the principled fix for the substrate's short-horizon planning limit, and language because the V-JEPA-to-language alignment gives this system a native instruction interface that most world-model agents lack. But affordances, tool use, social learning, cultural accumulation, and POET-style environment generation all require an environment rich enough to generate stepping stones, and building that environment, not designing the learning rule, is the binding constraint for a solo researcher. The corpus states this without flinching: development of genuinely new competence may require a world to develop in, and the minimal versions (Experiment 10, the task-generating teacher, the language-goal interface, the skill-indexing design) are the most an individual program can honestly attempt.

Third, the synthesis across all three volumes. The lever list has now been covered end to end: the ten deep dossiers of the developmental spine in Volume I, the forty-five remaining mechanisms in Volume II, and the twenty-eight hardware-hard and open-ended levers in Volume III. The pattern is consistent and is the corpus's central empirical claim about feasibility. A large majority of biologically motivated learning mechanisms attach cleanly to a frozen perceptual module, run on cached latents on a single laptop, and admit clean null interpretations, which is exactly why the developmental spine (staged plasticity, latent replay, consolidation, uncertainty-gated neuromodulation, and their combination) is the right place to start and the right place to expect a publishable result. The levers that resist this fall into two honest groups: those that need dense per-token features, which is most of the object-centric and relational family and which is why the V-JEPA 2 versus 2.1 comparison keeps recurring as a result in its own right, and those that need an environment and rollouts, which is the entire curiosity, control, and open-endedness program. The dividing line that matters is therefore not biological plausibility, which is high almost everywhere, but compute feasibility and substrate compatibility.

The frozen encoder is not a limitation to apologize for. It is the design commitment that makes this program tractable for one person: perception is inherited and fixed, so stored latents never go stale, the entire trainable surface is small enough to run locally, and every alternative learning rule and plasticity schedule and memory policy can be tested in isolation against a clean baseline. The system being built is not a JEPA. It is a developmental learning system that has been given a frozen V-JEPA encoder for eyes, and whose memory, plasticity, motivation, and structure are the actual science. The corpus has now specified that system lever by lever, said which parts to build now and which to scope, and named the one ceiling, environmental richness, that no amount of clever wiring around a frozen encoder can substitute for.
