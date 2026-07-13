# Toward Developmental JEPA

## A Biology-Grounded, Mechanism-Tested Research Corpus for Continual, Self-Directed World Models on a Frozen V-JEPA Substrate

### Corpus Volume I: Substrate, Spine, and Experiment Bank

---

## How to read this volume

This is the first volume of the corpus your prompt asked for, not the whole thing. The source document specs a body of work that, done at full one-page-per-lever depth across all 175 levers plus a complete experiment bank, is a multi-volume effort. Trying to emit all of it in one pass would force padding, and padding is exactly what the source doctrine forbids ("prefer specificity over elegance, informative failure over vague success").

So Volume I does the part that everything else depends on, to real depth:

1. The substrate, grounded in primary sources (V-JEPA, V-JEPA 2, V-JEPA 2-AC, V-JEPA 2.1). This is where current accurate facts matter most and where guessing would poison the whole program.
2. The developmental spine: the small number of levers that carry most of the load (staged plasticity, latent replay and consolidation, uncertainty-gated learning, curiosity, relational maps, structural adaptation, local learning, open-endedness).
3. The full lever ranking table, so the ruthless ranking the doctrine demands exists before any code.
4. The ten named experiments, each fully specified down to seeds, compute, and null-result interpretation.
5. The plans, the integrated architecture, the negative-result taxonomy, the open questions, the final judgment.
6. A roadmap for Volumes II and III (the remaining levers as structured one-pagers).

A note on confidence. Everything in the substrate section is anchored to primary releases and I have read the relevant papers and model cards directly. Everything in the mechanism sections is anchored to established literature; where I give a specific number I am confident of it, and where the source document asked for a precise figure I do not have, I say so rather than inventing one. Truth levels are labeled per the doctrine.

A note on format. No em dashes anywhere in this corpus, per your standing rule. Tables and structure are used because this is a reference artifact, not a chat reply.

---

## 0. Source ledger and confidence

Primary substrate sources, read directly:

- V-JEPA (Bardes et al., 2024). The original action-free video JEPA. Latent mask-denoising, EMA target encoder, multi-block masking.
- V-JEPA 2 (Assran et al., FAIR at Meta, arXiv 2506.09985, Jun 2025). Encoder up to ~1B params, pretrained on 1M+ hours of video plus 1M images, L1 feature-prediction loss, EMA target, progressive resolution training. V-JEPA 2-AC is a 300M block-causal transformer trained on under 62 hours of Droid robot data for action-conditioned latent prediction and planning.
- V-JEPA 2.1 (Mur-Labadia et al., FAIR at Meta and Universidad de Zaragoza, arXiv 2603.14482, Mar 2026, last revised Jun 2026). Dense predictive loss, deep self-supervision, multimodal tokenizers, scaling. Adds ViT-B, L, g, G variants and a new AC model. This release post-dates the standard training cutoff, so all 2.1 claims here come from the paper and its surrounding discussion, not from prior knowledge.
- HuggingFace `facebook/v-jepa-2` collection and the `facebookresearch/vjepa2` repo. Checkpoint names, sizes, and the attentive-probe and AC checkpoints.

Useful recent context, read directly:

- V-JEPA 2-AC planning numbers vs Cosmos latent diffusion (efficiency and manipulation success).
- "Latent Video Prediction Learns Better World Models" (arXiv 2605.15618, May 2026), a controlled frozen-encoder comparison of V-JEPA 2.1, V-JEPA 2, VideoPrism, VideoMAEv2.
- LeJEPA / SIGReg (Balestriero and LeCun, 2025) and LeWM (Maes et al., 2026): isotropic-Gaussian anti-collapse and a tiny end-to-end action-conditioned JEPA.
- SALT (Apple, 2025): teacher-student decoupling that beats V-JEPA 2 under frozen evaluation at matched FLOPs.

Mechanism sources are established literature, cited inline by author and year at the point of use. They are not re-derived here.

---

## 1. Executive thesis

The bet, in one sentence: a frozen V-JEPA encoder gives you an inherited perceptual cortex with genuinely good motion, short-horizon dynamics, and (in 2.1) dense spatial structure, and the open question worth a year of work is how much developmental behavior, meaning continual adaptation without catastrophic forgetting, self-generated curriculum, and consolidation, can be manufactured in the cheap trainable shell around it before the frozen latent geometry becomes the binding constraint.

The realistic contribution is not a developmental mind. It is a controlled frozen-JEPA developmental testbed plus a mechanism map that says which biological levers measurably help, which fail, and under exactly which interpretable conditions they fail. That is publishable, it is honest, and it is the thing the field actually lacks. Current frozen-JEPA world-model work is narrow: an action-conditioned predictor for pick-and-place, or comparative frozen-eval studies. Nobody has built the developmental wrapper and stress-tested it. That gap is the opening.

The sharpest single hypothesis to organize everything around: on a sequential stream of latent-prediction tasks built from a frozen V-JEPA encoder, a developmental schedule (broad early plasticity, then latent replay plus synaptic consolidation, then plasticity closure, then surprise-gated reopening) beats constant-plasticity training on the adaptation-retention frontier, and the size of that win is bounded by how much task-relevant episodic detail the frozen latent preserves. Everything downstream is a test of one half of that sentence.

The single most important early result is a negative-capable baseline: a frozen-encoder continual-learning harness that can demonstrably forget and demonstrably adapt, with reproducible seeds. If the baseline cannot fail in a measurable way, no biological test built on top of it means anything. This is Experiment 1, and it gates the rest.

---

## 2. Substrate analysis: V-JEPA 2, 2-AC, and 2.1

### 2.1 The objective, and why it is the right substrate

[Established ML] V-JEPA learns by predicting the representations of masked spatiotemporal regions of a video in a learned latent space, not in pixels. A clip is patchified into tokens, a subset is dropped (masked), the encoder processes the visible tokens, learnable mask tokens marking the missing positions are concatenated, a predictor fills them in, and the predicted features are regressed against targets produced by an EMA copy of the encoder. V-JEPA 2 uses an L1 feature-prediction loss. Collapse (the trivial solution where everything maps to a constant) is prevented structurally by the asymmetry between the context encoder and the EMA target encoder plus the masking, in the same family as the I-JEPA and BYOL-style stop-gradient-to-an-EMA-target design, rather than by an explicit contrastive term.

Why this matters for a developmental program: the JEPA objective deliberately throws away unpredictable pixel detail (the exact texture of grass, the precise glint on metal) and keeps predictable structure (where the object will be). That is the right inductive bias for a world model you want to plan in, and it is the wrong substrate for anything that needs the discarded detail. Several null results later in this corpus reduce to "the frozen latent already threw away the variable your mechanism needed." Hold that thought; it recurs.

[Speculative mapping] The JEPA prediction-error-in-latent-space framing is the formal-analogy bridge to predictive coding and the free-energy view of cortex. The analogy is real at the level of "both minimize a prediction error over a learned representation," and it is loose everywhere else: JEPA does not do iterative inference to a fixed point, has no explicit precision-weighting, and its hierarchy is a feedforward transformer, not reciprocal cortical columns. Treat predictive coding as Level 1 on the evidence ladder for this substrate, not Level 3, unless you actually implement iterative inference.

### 2.2 V-JEPA 2 architecture, checkpoints, and what runs on 18GB

The released encoders, with the practically relevant facts for a solo researcher on an M3 Pro 18GB:

| Checkpoint | Encoder | Approx params | Res | Frames/clip | Realistic on M3 18GB? |
|---|---|---|---|---|---|
| `facebook/vjepa2-vitl-fpc64-256` | ViT-L/16 | ~304M | 256 | 64 | Yes, comfortably, for inference and latent extraction (MPS, fp16) |
| `facebook/vjepa2-vith-fpc64-256` | ViT-H/16 | ~630M | 256 | 64 | Yes for inference, slower; fine for batch latent extraction overnight |
| `facebook/vjepa2-vitg-fpc64-256` | ViT-g/16 | ~1B | 256 | 64 | Borderline; works for inference but tight, prefer small batches |
| `facebook/vjepa2-vitg-fpc64-384` | ViT-g/16 | ~1B | 384 | 64 | Tightest; 384 res inflates token count, use only if you need it |
| AC predictor (`vjepa2_ac_vit_giant`) | block-causal transformer on ViT-g | ~300M | n/a | n/a | Inference yes; it sits on top of frozen ViT-g |

[Established ML, from model cards and the paper] The encoder is a vanilla ViT with 16x16 spatial patches and a short temporal tubelet, ingesting 64 frames per clip at the "fpc64" checkpoints. The predictor used in pretraining is a narrower transformer than the encoder. The attentive probe used for the understanding and anticipation evals is small (a handful of cross-attention blocks plus a linear head), which is the key fact for you: the things you actually train are tiny.

The decisive practical point: in this whole program the frozen encoder is the only large object, and you never train it. Everything you do train (predictor heads, memory retrieval, uncertainty heads, curiosity modules, plasticity controllers, routers) is in the tens of millions of parameters at most, often single-digit millions. Those train on the Mac. The encoder runs on the Mac for inference. You cache latents to disk once and then iterate against cached latents, which decouples your experiment loop from encoder cost entirely. This is the same move you already make when benchmarking local models: pay the expensive forward pass once, then iterate cheap.

Latent storage, order of magnitude (verify against the actual config before committing disk): a 256-res 64-frame clip yields on the order of a few thousand tokens after patchification and temporal tubelet stride, at 1024 dims for ViT-L. In fp16 that is on the order of 10 to 20 MB of dense per-token features per clip. Pooled (one vector per clip or per frame) it is kilobytes. Decide per experiment: dense features are needed for object-centric and relational work and for 2.1's strengths; pooled features are fine for global classification and most continual-learning streams and are 1000x cheaper to store. A 10k-clip dense latent dataset is on the order of 100 to 200 GB; pooled it is well under a gigabyte. Start pooled, go dense only where a lever demands it.

### 2.3 V-JEPA 2-AC: action conditioning, planning, and rollout horizon

[Established ML] V-JEPA 2-AC freezes the encoder and trains a new 300M-parameter predictor with block-causal attention that autoregressively predicts the next frame's representation conditioned on past frame representations, the action, and the end-effector state. It was trained on under 62 hours of unlabeled Droid trajectories. Actions are injected as additional tokens in the block-causal sequence: at each timestep, patch features attend to patch features, actions, and end-effector states from the current and previous steps.

Planning is energy minimization in latent space inside a model-predictive-control loop. Given an image goal (encoded to a target latent), candidate action sequences are rolled out through the AC predictor, the energy is the distance between the predicted future latent and the goal latent, and the planner (cross-entropy method style sampling) picks the action sequence that minimizes it, executes the first action, then replans. This is receding-horizon control over a learned latent forward model.

The numbers that bound ambition: against a Cosmos latent-diffusion world model, V-JEPA 2-AC reached 100% on single-goal reaching and 60 to 80% on prehensile manipulation, at roughly 16 seconds per action versus minutes for the diffusion baseline, zero-shot in new labs with no reward and no task-specific training. Camera-pose sensitivity is roughly linear in azimuth error and is correctable by a linear calibration. The paper's own limitations section names three: sensitivity to camera positioning, degradation on long-horizon planning, and reliance on image goals (you must be able to express the goal as an image the encoder can embed).

The rollout-horizon fact is the one that constrains every planning and open-endedness experiment: latent rollouts stay useful for short horizons and then compounding error dominates. The paper does not publish a single clean "N steps" number because it is task and setup dependent, but the practical regime is short multi-step plans with frequent replanning, not long open-loop rollouts. Any experiment that assumes long reliable rollouts is mis-specified for this substrate.

### 2.4 V-JEPA 2.1: what actually changed, and why it changes the plan

[Established ML, from the 2.1 paper] V-JEPA 2.1 keeps the JEPA frame and adds four things:

1. Dense predictive loss. The training signal comes from all tokens, both the visible context and the masked positions, with a distance-weighted prediction loss, instead of supervising only masked regions. This is the central change and it is what produces spatially dense, per-token-meaningful features rather than features that are mainly good in aggregate.
2. Deep self-supervision. The SSL objective is applied hierarchically at multiple intermediate encoder layers, not just at the output, improving representation quality through the depth of the network.
3. Multimodal tokenizers. Unified tokenization lets images and video train together in one model.
4. Scaling in capacity and data, including a new ViT-B (smaller than L, good news for your laptop) and a ViT-G (larger than g).

Reported results: 7.71 mAP on Ego4D short-term object-interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS high-level action anticipation, a 20-point improvement in real-robot grasping success over V-JEPA 2-AC, robotic navigation at 5.687 ATE on TartanDrive, depth estimation at 0.307 RMSE on NYUv2 with a linear probe, and 77.7 on Something-Something-v2 for global recognition.

Why this reshapes the program, concretely:

- The relational-maps and object-centric levers (Stage 6) move up the ranking on 2.1 and stay low on 2. Object extraction, slot binding, and what/where factorization all need features that are meaningful per spatial location. V-JEPA 2's features are good in aggregate but not strongly dense; V-JEPA 2.1's are explicitly dense and produce a usable depth probe and navigation signal. On 2.1, an object-centric or graph head has real features to bind to; on 2, the same head is more likely to "add capacity but not structure," which is the canonical Stage 6 null result.
- The embodiment and planning axis gets a better substrate (the 20-point grasping jump and the navigation and depth results), so Stage 9 experiments should default to the 2.1 AC model where available.
- The ViT-B variant lowers the floor: latent extraction and even light experimentation get cheaper, and the determinism-sanity instinct you already use (multiple runs at temperature 0 to measure byte-level reproducibility) transfers directly to checking that your latent extraction is deterministic across runs on MPS before you trust any continual-learning delta.
- Default choice: build the harness on V-JEPA 2 ViT-L first (best-documented, most third-party code, smallest well-supported encoder), then re-run the headline experiments on V-JEPA 2.1 ViT-L or ViT-B as a "does dense help" ablation. That comparison is itself a clean, publishable mini-result and it plays directly to your benchmarking strengths.

### 2.5 Attachment points: what to freeze, what to train cheaply

| Component | Status | Notes |
|---|---|---|
| Encoder | Frozen, always | The whole premise. ~304M (ViT-L) to ~1B (ViT-g). |
| EMA target encoder | Frozen / not used at inference | Only relevant if you ever re-open the SSL objective, which is out of scope. |
| Tokenizer / patch embed | Frozen | Part of the encoder. |
| Latent predictor | Trainable, cheap | Your main object. Tens of millions of params. |
| Action-conditioned predictor | Trainable, cheap | The 300M AC predictor, or your own smaller one. |
| Attentive probe / heads | Trainable, very cheap | Single-digit to low-tens of millions. |
| Latent memory + retrieval head | Trainable, cheap | Buffer is data, retrieval head is small. |
| Uncertainty estimator | Trainable, cheap | Ensemble of small predictors or MC-dropout. |
| Plasticity controller | Trainable or scripted | Can be a schedule, a rule, or a small meta-learned net. |
| Curiosity module | Trainable, cheap | RND target+predictor, or learning-progress tracker. |
| Planner cost function | Mostly scripted | Energy = latent distance; CEM/MPPI sampler around it. |
| Relational / graph head | Trainable, cheap (needs dense features) | Prefer 2.1. |
| Sparse / MoE router | Trainable, cheap (compute benefit needs kernels) | Scientifically testable on GPU without speedup. |
| Local-learning head | Trainable, cheap | Hebbian/forward-forward/EP on a small head. |

The cleanest attachment points, ranked by leverage: (1) after the encoder, on cached latents, for everything continual-learning and memory; (2) inside or replacing the predictor, for plasticity, sparsity, and local-learning experiments; (3) around the MPC loop, for uncertainty-gated planning and curiosity; (4) in the data/action selection step, for self-curriculum.

### 2.6 What V-JEPA encodes well and poorly, and which gaps are bolt-on-fixable

| Capability | V-JEPA 2 | V-JEPA 2.1 | Fixable by bolt-on? |
|---|---|---|---|
| Motion, short-horizon dynamics | Strong | Strong | n/a (already good) |
| Action anticipation | Strong (SoTA EK100) | Stronger | n/a |
| Spatial / dense structure | Moderate, mostly aggregate | Strong, explicitly dense | Use 2.1; do not try to bolt density onto 2 |
| Object persistence in the latent | Weak | Better but not guaranteed | Partially: object-centric head helps on 2.1; full fix needs retraining |
| Long-horizon causal structure | Weak | Weak | No clean bolt-on; this is a representation limit |
| Agent-specific goals | Absent | Absent | Yes: goal head, autotelic module |
| Persistent identity over long times | Absent | Absent | Yes: episodic memory + indexing |
| Fine-grained contact dynamics | Weak | Weak | Mostly no; needs richer/embodied data |
| Counterfactuals | Weak | Weak | Partial: action-conditioned rollouts give "what if I act," not full counterfactuals |
| Tool use | Absent | Absent | Only via embodiment + skill learning |
| Explicit symbolic relations | Absent | Absent | Yes: relational/graph head |
| Continual updates | Absent | Absent | Yes: this entire program |
| Self-directed learning | Absent | Absent | Yes: curiosity + curriculum |

Gaps that need encoder retraining, not bolt-ons: any missing perceptual factor the encoder never learned to represent, poor object permanence baked into the latent geometry, failure to represent a causally relevant variable, and lack of proprioception. You cannot recover information the frozen encoder discarded. The single most important diagnostic in the whole program is therefore a linear-probe sweep: before testing any mechanism that needs variable X, probe whether X is linearly decodable from the frozen latent. If it is not, the mechanism's failure is pre-ordained and uninformative, and you must either switch to 2.1, switch tasks, or label the lever "requires retraining."

### 2.7 Hardware reality for this specific researcher

You are solo on an M3 Pro 18GB. The program is designed around that, not apologizing for it. The pattern:

- Encoder inference and one-time latent extraction: on the Mac, MPS, fp16, ViT-L or 2.1 ViT-B. Overnight batch jobs.
- All trainable modules: on the Mac. They are small. A predictor head plus memory plus uncertainty is well under 100M params and trains fine in 18GB.
- The few things that need rented GPU: (a) any experiment with a live environment and many rollouts (curiosity, open-endedness, planning-in-the-loop), because the bottleneck is environment steps, not model size; (b) MoE or growable-module experiments if you want real wall-clock numbers; (c) any sweep large enough that Mac wall-clock becomes the limiting factor on iteration speed. For those, a single A100 or H100 spot instance for hours, not weeks, is the right spend. Your accounting brain will appreciate that the dominant cost is your time, and caching latents converts most of the work to CPU/MPS-bound iteration that costs nothing.
- MLX vs PyTorch-MPS: PyTorch-MPS is the safe default because the V-JEPA repo and HF integration are PyTorch. MLX is worth a side experiment only if you want to push encoder inference throughput on Apple Silicon, and that is an optimization, not a blocker. Do not let it become a yak-shave.

---

## 3. The developmental hypothesis, sharpened

The central claim, restated with the binding constraint made explicit:

A frozen-encoder JEPA system becomes more developmental to the exact extent that its trainable shell can implement staged plasticity, memory-supported consolidation, surprise-gated reopening, and self-generated curriculum, and that extent is upper-bounded by the information the frozen latent preserves. The encoder is the inherited perceptual cortex. Development happens in the shell, or not at all.

The nine stages, with the failure flag attached to each:

1. Perceptual inheritance. Begin with V-JEPA's pretrained representation. Failure flag: the inherited representation may simply lack a variable you need (Section 2.6).
2. Broad plastic stage. Predictor and heads highly plastic; learn broad latent dynamics. Failure flag: with a frozen encoder there may be little to "stabilize" beyond the head, so stages may not differentiate.
3. Episodic accumulation. Novel, surprising, rewarding, or high-learning-progress latents enter memory. Failure flag: prioritization signal may not track learnability (noisy-TV).
4. Consolidation. Replay moves structure from memory into weights; regularization protects old skills. Failure flag: frozen latent geometry may not preserve replay-worthy episodic detail.
5. Plasticity closure. Learning rates fall, weights stabilize, modules specialize. Failure flag: closure may just be a learning-rate decay with no developmental content.
6. Reopening. Genuine novelty re-raises plasticity; the system must separate learnable novelty from noise. Failure flag: the uncertainty signal may be uncalibrated or react too late.
7. Self-curriculum. The agent chooses actions or data that maximize learning progress. Failure flag: needs an action space and an environment rich enough to have a curriculum.
8. Structural adaptation. Modules grow, prune, specialize, route. Failure flag: routing overhead may dominate; specialization may block transfer.
9. Open-ended loop. New skills create new goals create new data create new abstractions. Failure flag: open-endedness may be a population-level property that a single agent cannot exhibit.

The honest prior: stages 1 through 6 are testable now on cached latents and are where the publishable results live. Stages 7 through 9 need embodiment and environment generation and are where the program shades from "solo year" into "lab-scale."

---

## 4. Evidence ladder, populated

The doctrine's 0-to-6 ladder, with the core levers placed:

- Level 0, metaphor only: oscillations and communication-through-coherence (no concrete routing mechanism on a transformer), serotonin-like "patience" (no clean operationalization yet).
- Level 1, formal analogy: predictive coding and free energy (real prediction-error structure, wrong inference process), global workspace (bottleneck analogy without a mechanism).
- Level 2, ML precedent exists: replay, EWC, Synaptic Intelligence, RND, ICM, learning-progress curiosity, MoE, feedback alignment, forward-forward, equilibrium propagation, successor representations. All demonstrated in artificial systems, none yet on a frozen video JEPA.
- Level 3, JEPA-attachable: a latent replay buffer over V-JEPA features; a plasticity schedule on the predictor; an ensemble-disagreement uncertainty head on latent rollouts; an RND novelty bonus in the planner's reward. These are the things you can wire up in week one.
- Level 4, controlled experiment: each of the ten experiments in Section 7, once it has a baseline, an ablation, a metric, and a null interpretation.
- Level 5, integrated developmental contribution: replay plus staged plasticity plus uncertainty gating beating any single component on the adaptation-retention frontier. This is the headline result the program is aiming at.
- Level 6, open-ended contribution: the system acquires skills or generates goals beyond its task list. Most levers will not reach Level 6 on this substrate, and the corpus says so plainly.

---

## 5. Lever ranking table

Scoring is 1 to 5, calibrated for a solo researcher on the V-JEPA substrate, not for a lab. "JEPA attach" means attachability to a frozen encoder specifically. "Compute" means feasibility for you (5 = laptop, 1 = needs neuromorphic or large cluster). "Null clarity" means how cleanly a failure is interpretable. "Dev" is developmental relevance, "Open" is open-endedness relevance. Verdict is the build decision.

| Lever | Bio | ML prec | JEPA attach | Compute | Null clarity | Dev | Open | Action? | Retrain? | Kernel? | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Staged plasticity / critical periods | 5 | 4 | 5 | 5 | 4 | 5 | 2 | no | no | no | Build now |
| Latent replay (hippocampal) | 5 | 5 | 5 | 5 | 5 | 5 | 2 | no | no | no | Build now |
| Synaptic consolidation (EWC/SI) | 4 | 5 | 5 | 5 | 5 | 5 | 1 | no | no | no | Build now |
| CLS (fast hippocampus + slow cortex split) | 5 | 4 | 4 | 4 | 4 | 5 | 2 | no | no | no | Build now |
| Uncertainty-gated plasticity | 4 | 4 | 4 | 4 | 4 | 5 | 3 | no | no | no | Build now |
| Neuromodulation (DA/ACh/NE analogs) | 5 | 3 | 4 | 4 | 3 | 4 | 3 | partial | no | no | Prototype |
| Prediction-error curiosity / ICM | 4 | 5 | 4 | 3 | 4 | 4 | 4 | yes | no | no | Prototype (needs env) |
| RND novelty | 3 | 5 | 4 | 3 | 5 | 3 | 4 | yes | no | no | Prototype (clean noisy-TV test) |
| Learning-progress curiosity | 5 | 4 | 3 | 3 | 4 | 5 | 5 | yes | no | no | Prototype (needs env) |
| Empowerment | 3 | 3 | 2 | 2 | 3 | 3 | 4 | yes | maybe | no | Theory / later |
| Active inference | 4 | 3 | 3 | 3 | 2 | 4 | 4 | yes | no | no | Prototype carefully |
| Successor representations | 4 | 4 | 4 | 4 | 4 | 4 | 3 | partial | no | no | Build now (mid-program) |
| Place/grid cells, TEM | 5 | 3 | 3 | 3 | 3 | 4 | 3 | partial | maybe | no | Prototype (needs 2.1 dense) |
| Object-centric / slots | 4 | 4 | 3 | 3 | 4 | 4 | 3 | no | maybe | no | Prototype on 2.1 only |
| Relational / graph head | 3 | 4 | 4 | 4 | 4 | 4 | 4 | no | no | no | Prototype on 2.1 |
| Sparse coding / k-WTA | 4 | 4 | 4 | 4 | 4 | 4 | 2 | no | no | maybe | Build now (no-speedup OK) |
| Mixture-of-experts / routing | 2 | 5 | 4 | 3 | 4 | 4 | 3 | no | no | maybe | Prototype |
| Structural pruning | 4 | 5 | 4 | 4 | 4 | 4 | 2 | no | no | no | Build now (mid-program) |
| Neurogenesis / module birth | 4 | 3 | 3 | 3 | 3 | 4 | 4 | no | no | no | Prototype (growable nets) |
| Dendritic predictors | 4 | 3 | 3 | 4 | 3 | 3 | 2 | no | no | maybe | Prototype |
| Local learning (Hebb/Oja/BCM) | 5 | 3 | 3 | 4 | 4 | 3 | 2 | no | no | no | Prototype (head only) |
| Forward-forward | 2 | 3 | 3 | 4 | 4 | 3 | 2 | no | no | no | Toy test |
| Equilibrium propagation | 3 | 3 | 2 | 3 | 4 | 3 | 2 | no | no | maybe | Toy test |
| Feedback alignment | 3 | 4 | 3 | 4 | 4 | 3 | 2 | no | no | no | Toy test |
| STDP / spiking | 5 | 2 | 2 | 1 | 3 | 3 | 2 | no | maybe | yes | Neuromorphic / sim only |
| Three-factor learning rules | 5 | 3 | 3 | 4 | 4 | 4 | 3 | partial | no | no | Prototype (head + modulator) |
| Generative replay | 3 | 4 | 3 | 3 | 4 | 4 | 2 | no | maybe | no | Prototype (latent generator) |
| Metaplasticity | 4 | 3 | 4 | 4 | 4 | 5 | 2 | no | no | no | Build now (as plasticity controller) |
| Attention / global workspace | 4 | 3 | 3 | 4 | 2 | 3 | 3 | no | no | no | Theory / later |
| Open-ended env generation (POET) | 3 | 4 | 3 | 2 | 3 | 4 | 5 | yes | no | no | Lab-scale / later |
| Quality diversity (MAP-Elites) | 2 | 4 | 3 | 3 | 4 | 3 | 5 | yes | no | no | Prototype (skill archive) |
| Autotelic goal generation | 3 | 4 | 3 | 3 | 3 | 4 | 5 | yes | no | no | Prototype (needs env) |
| Language as scaffolding | 4 | 4 | 4 | 4 | 3 | 3 | 4 | no | no | no | Prototype (use the LLM-aligned head) |

Reading the table: the "build now" cluster is the developmental spine, and it is all doable on cached latents on your laptop with clean null interpretations. The curiosity and open-endedness cluster is high on open-endedness relevance and low on compute feasibility for a solo run, because it needs environments and rollouts, which is the honest reason it is staged later and partly flagged lab-scale. The local-learning and dendritic and spiking cluster is scientifically interesting and mostly cannot beat backprop at this scale, so it is toy-test or simulation-only, exactly as the doctrine's tractability labels intend: not rejected, scoped.

---

## 6. Deep lever dossiers (the developmental spine)

Each dossier follows the doctrine's structure compactly: biology, computational abstraction, JEPA mapping, ML analogs, the experiment it feeds, failure mode, tractability, developmental role, dependencies. These are the levers that carry the program. The remaining levers get one-pager treatment in Volumes II and III (Section 15).

### 6.1 Staged plasticity and critical periods

Biology. [Established neuroscience] Critical and sensitive periods are windows in development when specific circuits are maximally plastic and after which plasticity drops sharply. The canonical case is ocular dominance in primary visual cortex (Hubel and Wiesel; Wiesel's monocular-deprivation experiments in cats and monkeys): deprive one eye during the window and the cortical territory rewires permanently; do the same after the window and little changes. Closure is not passive decay. It is actively gated by the maturation of inhibition (parvalbumin interneurons), by the formation of perineuronal nets that physically stabilize synapses, and by molecular brakes. Timescale: developmental, days to months depending on species and circuit. What it regulates: how much experience can reshape a circuit, as a function of age and circuit, not uniformly.

Computational abstraction. A schedule, possibly per-module and possibly experience-gated rather than purely clock-gated, that governs effective learning rate and weight-change magnitude. The deep claim is not "decay the learning rate." It is that the timing and triggering of plasticity has structure: early broad plasticity to find the right gross organization, then closure to protect it, then targeted reopening on genuine novelty.

JEPA mapping. Attaches to the predictor and heads, never the encoder. Implement as a per-parameter or per-module learning-rate gate. The interesting version ties the gate to a signal (task-boundary detection, surprise, ensemble disagreement) rather than to wall-clock step count. The "perineuronal net" analog is a per-weight rigidity term that grows as a weight stabilizes, which is structurally close to Synaptic Intelligence's importance accumulation (6.3), and that overlap is a feature: critical-period closure and synaptic consolidation may be two views of the same mechanism.

ML analogs. [Established ML] Achille, Rovere, and Soatto, "Critical Learning Periods in Deep Networks" (ICLR 2019): deep nets have their own critical periods, a deficit (e.g. blur) imposed early and then removed causes permanent damage, and the Fisher-information trace over training shows a characteristic rise and fall that mirrors a sensitivity window. This is the single most important ML precedent for the whole lever and you should reproduce its Fisher-trace measurement as a sanity check. Progressive freezing and progressive unfreezing in transfer learning are the crude engineering cousins.

Feeds. Experiment 3 (critical-period schedule).

Failure mode. The classic ML failure is that "staged plasticity" collapses into "a learning-rate decay schedule with extra steps," producing no benefit over a well-tuned cosine decay. The detection metric is a direct comparison against a tuned monotonic-decay baseline; if the staged schedule does not beat it on the adaptation-retention frontier, the developmental framing added nothing. The biological-analogy failure is reading too much into "critical period" when the frozen encoder means there is no representation to critically shape, only a head.

Tractability. Laptop feasible. This is the cheapest lever in the program.

Developmental role. Plasticity opening and closing. The spine of Axis A.

Dependencies. Weak alone. The literature and the doctrine both predict it needs memory (replay) and a trigger (uncertainty) to beat the decay baseline. Test in isolation first to establish that it does little alone, then combine. Do not combine before the isolated null is documented.

### 6.2 Latent replay and complementary learning systems

Biology. [Established neuroscience] Hippocampal replay: during rest and sleep, hippocampal place-cell sequences from recent experience reactivate, compressed in time, sometimes reversed. Complementary Learning Systems theory (McClelland, McNaughton, O'Reilly 1995; Kumaran, Hassabis, McClelland 2016): a fast-learning hippocampus stores specific episodes and a slow-learning neocortex extracts statistical structure, and replay is the teaching channel by which the fast system trains the slow one offline, interleaving old and new to avoid catastrophic interference. Reverse replay is preferentially associated with reward; sharp-wave-ripple replay supports consolidation. Timescale: milliseconds (a replay event) to weeks (systems consolidation).

Computational abstraction. Two memory systems with different learning rates plus an interleaving mechanism. The fast store is an explicit buffer of episodes; the slow store is the weights of a network trained by replaying buffered episodes mixed with new data. Prioritization (what to replay) is a separate knob: by recency, by surprise, by reward, by learning progress, or reverse-ordered.

JEPA mapping. The cleanest, highest-leverage attachment in the program. The buffer holds V-JEPA latent sequences (pooled or dense) plus whatever target the task needs. The "cortex" is your trainable predictor. Replay = sampling buffered latent sequences and training the predictor on them interleaved with the current task stream. Because the encoder is frozen, the latents are stable over the whole lifetime, which is the one place the frozen constraint actively helps: stored latents never go stale the way they would if the encoder kept changing under them. This is a genuine architectural advantage worth stating in any paper.

ML analogs. [Established ML] Experience replay is the workhorse of continual learning (Rolnick et al. 2019, "Experience Replay for Continual Learning"; iCaRL, Rebuffi et al. 2017). Prioritized replay (Schaul et al. 2016) for RL. Generative replay (Shin et al. 2017) replaces the buffer with a generator. Latent replay (Pellegrini et al. 2020) specifically stores activations from a frozen lower network and replays those, which is almost exactly your setting and is the closest existing precedent. Your novelty is doing it over a video world-model latent with developmental scheduling and prioritization variants, and measuring against the frozen-latent-geometry ceiling.

Feeds. Experiment 2 (latent hippocampus), and it is a component of Experiments 3, 4, 5, 10.

Failure mode. The buffer overfits: replaying a small set of episodes too often makes the predictor memorize them rather than consolidate structure, detectable as high train-on-buffer performance with no backward-transfer gain. The frozen latent fails to preserve episodic distinctiveness: if two episodes that need different responses map to nearly identical latents, no replay scheme recovers them, detectable by the linear-probe diagnostic of Section 2.6 run on buffer contents. Retrieval too weak: nearest-neighbor in latent space may return unhelpful neighbors; detectable by ablating retrieval quality.

Tractability. Laptop feasible. The buffer is data on disk; the cortex is small.

Developmental role. Memory and consolidation. The spine of Axis B.

Dependencies. Enables consolidation (6.3) and pairs with staged plasticity (6.1). Conflicts with nothing. This and staged plasticity are the two levers to combine first, because the headline Level-5 result is built on their interaction.

### 6.3 Synaptic consolidation (EWC and Synaptic Intelligence)

Biology. [Established neuroscience] Synaptic consolidation stabilizes specific synapses after learning via molecular cascades (protein synthesis, structural changes), making them resistant to overwriting on a timescale of hours, distinct from the systems-level consolidation that replay supports over weeks. The principle: important synapses become harder to change.

Computational abstraction. A per-parameter regularizer that penalizes changes to weights deemed important for previously learned tasks, where importance is estimated by a Fisher-information proxy (EWC) or by the path integral of the gradient over training (Synaptic Intelligence). This is the weight-space dual of replay's data-space approach to forgetting.

JEPA mapping. Add an EWC or SI penalty to the predictor's loss. Pure weight-space, no buffer needed (though it composes with one). Trivial to attach. The conceptual link to 6.1 is exact: a per-weight rigidity that grows with importance is both "synaptic consolidation" and "critical-period closure," and you can implement one mechanism and interpret it under both labels, which is worth flagging as a small unifying observation.

ML analogs. [Established ML] EWC (Kirkpatrick et al., PNAS 2017); Synaptic Intelligence (Zenke, Poole, Ganguli, ICML 2017); Memory Aware Synapses (Aljundi et al. 2018). All well-characterized, all known to help on task-incremental settings and to struggle on class-incremental and on long task sequences where the Fisher estimate saturates.

Feeds. Experiment 2 (as the "replay plus EWC" variant) and Experiment 3.

Failure mode. The Fisher proxy saturates over a long task stream and the model becomes globally rigid, unable to learn task N, detectable as collapsing new-task adaptation speed late in the stream. On a frozen encoder with a small head, the head may lack the capacity for importance-weighting to matter, detectable by comparing against a larger head.

Tractability. Laptop feasible.

Developmental role. Consolidation, plasticity closing. Axis B.

Dependencies. Composes with replay (often additively) and with staged plasticity. The interesting question your program can answer cleanly: does weight-space consolidation (EWC/SI) add anything on top of data-space consolidation (replay) when the encoder is frozen, or does the frozen encoder make one of them redundant? That is a sharp, publishable sub-question.

### 6.4 Uncertainty-gated plasticity and artificial neuromodulation

Biology. [Established neuroscience] Yu and Dayan (Neuron 2005), "Uncertainty, neuromodulation, and attention": acetylcholine signals expected uncertainty (known unreliability of cues within a familiar context) and norepinephrine signals unexpected uncertainty (a change in the context itself), and these modulate the balance between relying on priors versus current sensory evidence, and gate learning. Dopamine carries reward-prediction error (Schultz, Dayan, Montague 1997). Serotonin is associated, more contestedly, with patience, behavioral inhibition, and temporal discounting. Timescale: fast (phasic, sub-second) to slow (tonic, minutes).

Computational abstraction. Scalar or low-dimensional global signals derived from the system's own statistics that gate other processes: learning rate, memory-write probability, replay priority, exploration temperature, planner-versus-reactive arbitration, attention. The unifying idea is "learn when to learn and how much," with the gates driven by surprise and uncertainty rather than fixed schedules.

JEPA mapping. Compute candidate signals from the latent predictor: prediction error (predicted vs target latent), ensemble disagreement across a small ensemble of predictors, MC-dropout variance, calibration error, novelty (RND), reward-prediction error if there is reward, learning progress (rate of error decrease). Wire each to a gate: error or disagreement to the plasticity learning-rate gate (this is the bridge to 6.1, making critical-period reopening surprise-triggered instead of clock-triggered), surprise to replay priority and memory-write, uncertainty to planner reliance. The "norepinephrine reset" is a global gain or a partial reinitialization on a detected context shift (task-boundary detection).

ML analogs. [Established ML] Ensemble disagreement for epistemic uncertainty (Lakshminarayanan et al. 2017; deep ensembles). MC-dropout (Gal and Ghahramani 2016). Plasticity or learning-rate modulation by uncertainty appears in meta-learning and in neuromodulated networks (Miconi et al., differentiable plasticity; backpropamine). Bayesian surprise (Itti and Baldi). The crucial known failure is that raw prediction error conflates epistemic uncertainty (learnable, reducible) with aleatoric uncertainty (irreducible noise), which is the noisy-TV trap.

Feeds. Experiment 4 (artificial neuromodulation) directly; gates the reopening in Experiment 3 and the prioritization in Experiment 2.

Failure mode. The gate fires on irreducible noise because the uncertainty signal is not calibrated and does not separate epistemic from aleatoric, detectable with the mandatory noisy-TV distractor: a region of the input that is high-entropy but unlearnable should not attract plasticity or replay or exploration. If it does, the signal failed. Second failure: the gate reacts too late (after the model has already adapted or already forgotten), detectable by latency analysis between the surprising event and the gate response.

Tractability. Laptop feasible for the signal computation and gating; the ensemble is several small predictors, which fit easily.

Developmental role. Plasticity control, exploration control, memory gating. The connective tissue between Axes A, B, and C.

Dependencies. Needs replay and a predictor in place. Enables the surprise-triggered reopening that makes staged plasticity more than a decay schedule. The epistemic/aleatoric separation may require a probabilistic latent-dynamics head (predicting a distribution over next latents, not a point), which is a dependency to flag: if point-prediction error cannot separate the two, upgrade the predictor to output uncertainty.

### 6.5 Curiosity, intrinsic motivation, and the noisy-TV problem

Biology. [Established neuroscience and developmental psychology] Intrinsic motivation drives exploration and play in the absence of external reward; infants and animals seek novelty and controllable contingencies. Berlyne's work on curiosity and arousal; the developmental-robotics tradition (Oudeyer and Kaplan) formalizes "intrinsic motivation systems" where the reward is internal. Dopamine responds to novelty and to information-predictive cues, not only to primary reward.

Computational abstraction. An internal reward signal that drives action or data selection. Variants differ in what they reward: prediction error (seek states you predict badly), information gain (seek states that most reduce posterior uncertainty), learning progress (seek states where your error is decreasing fastest, the derivative of competence), novelty (seek rarely-visited states), empowerment (seek states from which you have the most control over future states). Learning progress is the theoretically strongest because it is the only one that is intrinsically immune to the noisy-TV: an unlearnable noisy region has high prediction error but zero learning progress, so a learning-progress agent loses interest.

JEPA mapping. The intrinsic reward enters the planner's energy or the policy's reward. Prediction-error curiosity: reward = latent prediction error of your forward model. RND: reward = error of a predictor network trained to match a fixed random target network applied to the latent (novel latents have high RND error). Learning progress: track per-region prediction-error decrease over time and reward regions with high decrease. ICM: the inverse-forward model formulation, predicting action from latent transitions to filter out uncontrollable variation. Attention-as-experiment-selection: in the passive video setting, "exploration" becomes which clips or which masked regions to sample next, which is a data-selection problem rather than an action problem and is testable without an environment.

ML analogs. [Established ML] ICM (Pathak et al., ICML 2017). RND (Burda et al., ICLR 2019) plus the "Large-Scale Study of Curiosity-Driven Learning" (Burda et al. 2019) that documents the noisy-TV failure directly. Learning-progress and IAC (Oudeyer, Kaplan, Hafner). Empowerment (Klyubin et al.; Mohamed and Rezende 2015). VIME (Houthooft et al. 2016) for information gain. All have RL track records; ICM and RND are known to chase noise without the controllability filter or the learning-progress derivative.

Feeds. Experiment 5 (curiosity as self-curriculum), and it is the engine of Experiment 10.

Failure mode. Noisy-TV fixation (the canonical failure, must be tested). Novelty exhaustion: once everything is visited, the signal goes flat and exploration stops, detectable by tracking exploration over long runs. Curiosity without goal structure produces diverse-but-useless behavior, detectable by measuring downstream transfer rather than diversity alone. On the frozen substrate specifically, if the latent does not encode controllability, ICM's filter cannot work; probe for controllability decodability first.

Tractability. The signal is laptop-cheap, but the setting is not: meaningful curiosity needs an environment with actions, and rollouts are the bottleneck, which is why this is rented-GPU-when-scaled and prototype-now-on-a-toy-env. The data-selection variant (which clips to sample) is laptop-feasible and is the right first cut.

Developmental role. Exploration and curriculum. The spine of Axis C.

Dependencies. Needs an action space and an environment for the full version. The data-selection version needs only a pool of clips. Pairs with memory (curiosity decides what enters the buffer) and with uncertainty (learning progress is a derivative of the same error signals as 6.4). Should not be tested in isolation from a learnable-vs-noisy environment design, or the result is uninterpretable.

### 6.6 Cognitive maps: successor representations, TEM, and object-centric structure

Biology. [Established neuroscience] Place cells (O'Keefe), grid cells (the Mosers), head-direction and border cells: the hippocampal-entorhinal system builds a metric and relational map of space and, more generally, of task structure. The Tolman-Eichenbaum Machine (Whittington et al., Cell 2020) formalizes hippocampal-entorhinal function as factorizing structural knowledge (the abstract relational graph, grid-like) from sensory content (what is where), enabling generalization to new environments with the same structure. The successor representation (Dayan 1993; Stachenfeld, Botvinick, Gershman, Nat Neuro 2017) reframes place fields as predictive maps: a state is represented by the discounted future states it leads to, which is a middle ground between model-free and model-based.

Computational abstraction. Explicit relational or predictive structure layered on top of the latent. Successor representation: learn M, the expected discounted future occupancy, which makes planning and reward-transfer cheap. TEM-style: factorize "structure" (a learned relational graph or grid code) from "content" (the V-JEPA latent at each node), and bind them. Object-centric: decompose the scene into slots with persistent identity. What/where factorization: separate object identity from object location.

JEPA mapping. This is the lever where V-JEPA 2.1 versus 2 matters most. Successor representation over a discretized or learned latent state space attaches as a head predicting future-occupancy from current latent. A graph head builds nodes from latent clusters and edges from observed transitions, then plans on the graph. Object-centric slots and what/where factorization need per-location features that are individually meaningful, which is exactly what 2.1's dense predictive loss produces and what 2's aggregate features do not reliably give. On 2.1 these heads have real dense features to bind; on 2 they are likely to "add capacity but not structure," the canonical Stage 6 null.

ML analogs. [Established ML] Successor features (Barreto et al. 2017) for transfer in RL. Slot Attention (Locatello et al. 2020) and object-centric world models (SAVi, etc.) for slots. Neural graph planning and GNN world models. Successor-representation agents in deep RL (Kulkarni et al.). All demonstrated; object-centric methods are known to be finicky and to depend heavily on input representation quality, which is precisely why the dense-feature substrate question is decisive.

Feeds. Experiment 6 (relational map over JEPA latents).

Failure mode. The graph or slot module adds parameters and capacity but no genuine structure, so it matches a plain predictor with equal parameters, detectable by a parameter-matched dense baseline. Object extraction from frozen latents is poor (especially on 2), detectable by visualizing slot assignments and by probing object identity. The task does not actually require relational generalization, so the structure cannot pay off, detectable by including an explicit recombination or detour test that a flat model should fail.

Tractability. Laptop feasible for successor representations and graph heads on cached latents; object-centric is heavier but still single-GPU. Strongly prefer 2.1.

Developmental role. Abstraction and planning. Cross-cuts Axes C and D.

Dependencies. Needs dense features (2.1) for the object-centric variants. Successor representations need a defined state space (discretize the latent, or learn a codebook). Pairs with planning and with relational replay. Conflicts with nothing, but is wasted effort on 2 for the object-centric branch.

### 6.7 Sparse, modular, and structural adaptation

Biology. [Established neuroscience] Cortical activity is sparse; excitation and inhibition are balanced; lateral inhibition and winner-take-all dynamics sharpen representations; homeostatic plasticity keeps average activity in range; synaptic pruning massively reduces connections over development; adult neurogenesis adds new neurons in the dentate gyrus. Structure is not fixed; it is sculpted by experience.

Computational abstraction. Control over which units or modules are active and which exist. Sparse activation (k-winner-take-all): only the top-k units fire, which reduces interference between tasks because different tasks use different units. Mixture-of-experts: route inputs to specialized sub-networks. Pruning: remove low-importance weights. Growth and neurogenesis: add capacity when needed (progressive networks, growable modules). Module birth and death: a population of modules that are created, specialized, and culled.

JEPA mapping. All of these attach to the predictor. A k-WTA predictor, an MoE predictor with learned or uncertainty-driven routing, a prunable predictor, a growable predictor that adds a module per new task or per high-novelty event. The doctrine's key point applies forcefully here: a sparse method does not need to run faster on a GPU to be scientifically interesting. On dense GPU hardware, k-WTA and unstructured sparsity often run no faster (and can run slower) than dense, but they can still reduce catastrophic interference and improve specialization, which is a representational result, not a speed result. The speed result needs custom kernels and is a separate, later, custom-kernel-candidate question.

ML analogs. [Established ML] k-WTA and sparse coding (Olshausen and Field; Makhzani and Frey, k-sparse autoencoders). MoE for capacity and conditional computation (Shazeer et al. 2017; modern MoE LLMs). Progressive Neural Networks (Rusu et al. 2016) and PackNet (Mallya and Lazebnik 2018) for continual learning via structure. Lottery-ticket pruning (Frankle and Carbin 2019). Active dendrites and context-gated sparsity for continual learning (Iyer et al., 2022, Numenta) is a directly relevant precedent that combines dendrite-like gating with k-WTA to reduce forgetting.

Feeds. Experiment 7 (sparse/modular developmental predictor).

Failure mode. Routing overhead dominates and there is no compute or quality benefit, detectable by comparing wall-clock and accuracy against dense. Sparse modules fragment the data so each module sees too little to learn, detectable by per-module data counts and per-module performance. Over-specialization blocks transfer, detectable by a forward-transfer metric. Task boundaries are unclear so routing has nothing to key on, detectable by routing entropy that never drops.

Tractability. Laptop feasible for the representational questions (interference, specialization, transfer) with no speedup. Custom-kernel-candidate for the actual compute-efficiency claim. Do the science on the laptop; defer the kernel work.

Developmental role. Structural growth, plasticity closing (specialization). The spine of Axis D.

Dependencies. Benefits from task-boundary detection (from 6.4). Pairs with consolidation (specialized modules are easier to protect). The growable variant pairs with neurogenesis framing and with curiosity (grow a module when novelty is high).

### 6.8 Local learning and dendritic computation

Biology. [Established neuroscience] Real neurons learn with local information; there is no global backprop in cortex. Dendrites perform nonlinear local computation; pyramidal neurons have multiple compartments with branch-specific plasticity. Hebbian plasticity ("fire together, wire together") and its stabilized forms (Oja's rule, the BCM rule with its sliding threshold) are local. Spike-timing-dependent plasticity (STDP) makes the sign of the change depend on relative spike timing. Three-factor rules add a global neuromodulatory signal (the third factor, e.g. dopamine) to a local Hebbian product, which is the biologically plausible route to reward-modulated and credit-assigned learning.

Computational abstraction. Update rules that use only locally available signals (pre- and post-synaptic activity, plus possibly a global modulator) rather than gradients backpropagated through the whole network. The promise is online, memory-light adaptation (no storing activations for a backward pass) and biological plausibility; the cost is weak credit assignment through depth.

JEPA mapping. Train a small head on the frozen latent with a local rule, or use a local rule as an auxiliary regularizer alongside backprop. Three-factor on the final head, with the third factor being one of the neuromodulatory signals from 6.4, is the most promising variant because it connects local learning to the uncertainty machinery you already built. Forward-forward and equilibrium propagation can train a head without a backward pass. Feedback alignment uses random feedback weights instead of transposed forward weights.

ML analogs. [Established ML] Feedback alignment and direct feedback alignment (Lillicrap et al. 2016; Nokland 2016): work on small nets, degrade with depth and on hard tasks. Forward-Forward (Hinton 2022): two forward passes with positive and negative data, competitive on small benchmarks, not yet at scale. Equilibrium Propagation (Scellier and Bengio 2017): energy-based, biologically motivated, mostly small-scale. Predictive-coding networks can approximate backprop (Millidge, Tschantz, Buckley 2020). The honest summary: backprop dominates at any nontrivial scale, and local rules are interesting for online adaptation, energy, and plausibility, not for beating backprop on accuracy.

Feeds. Experiments 8 (dendritic predictor) and 9 (local learning head).

Failure mode. The local rule cannot assign credit through depth, so it fails on anything but a shallow head, detectable by depth ablation. The dendritic module just adds parameters, matchable by a parameter-equal MLP. The benefit only appears with spiking or precise timing, which this substrate does not have, in which case the lever is correctly labeled neuromorphic-only.

Tractability. Toy-testable on the laptop for a small head. Spiking and STDP are neuromorphic-or-simulation-only and score 1 on compute feasibility; they are mapped and deferred, not rejected.

Developmental role. Online adaptation, action (three-factor reward modulation). Mostly a robustness and plausibility contribution, rarely a headline.

Dependencies. Three-factor depends on the neuromodulatory signals (6.4). The honest expectation is that this cluster produces negative or modest results on accuracy and a possible positive result on online-adaptation memory cost, and that is a legitimate negative-result contribution.

### 6.9 Active inference and predictive coding (cross-cutting)

Biology and theory. [Established neuroscience for predictive coding; contested as a global theory for free energy] Predictive coding (Rao and Ballard, Nat Neuro 1999) posits cortex as a hierarchy that passes predictions down and prediction errors up, with precision (inverse variance) weighting the errors. The free-energy principle and active inference (Friston) generalize this: agents minimize variational free energy, which unifies perception (update beliefs to reduce error) and action (act to make predictions come true), with precision-weighting as attention and as the epistemic/pragmatic value tradeoff.

Computational abstraction. Perception as iterative inference to minimize precision-weighted prediction error; action as selecting policies that minimize expected free energy, which decomposes into an information-gain (epistemic, explore) term and a goal-achieving (pragmatic, exploit) term. This is a principled unification of the curiosity and planning machinery, but it is computationally heavy and notoriously hard to scale.

JEPA mapping. JEPA already does prediction-error-in-latent-space, so the formal analogy is immediate but shallow (Section 2.1): JEPA does not iterate to a fixed point and has no explicit precision. To make it Level 3 rather than Level 1, you would add iterative latent inference (refine the latent estimate by gradient steps on prediction error at inference time) and precision-weighting (an uncertainty head that down-weights noisy dimensions, which is the 6.4 machinery again). Active inference as a planner: the expected-free-energy objective in the MPC loop gives a principled curiosity-plus-goal reward, where the epistemic term is information gain (6.5) and the pragmatic term is the image-goal energy (2.3).

ML analogs. [Established ML] Active inference agents in small discrete and continuous settings (Friston's group; Tschantz et al.; Millidge). Predictive-coding approximations of backprop. The known failure is scaling: expected-free-energy computation over policies is expensive, and clean wins over simpler curiosity-plus-RL baselines are not established at scale.

Feeds. Informs Experiment 4 (precision as the gate) and Experiment 5 (expected free energy as the curiosity objective). Not its own first experiment because the clean-win bar is high and the compute cost is real.

Failure mode. The free-energy framing adds notation without changing behavior relative to prediction-error curiosity plus goal-distance reward, detectable by ablating to that simpler baseline and showing no difference. Iterative inference does not improve over feedforward prediction on this substrate, detectable directly.

Tractability. The iterative-inference and precision pieces are laptop-feasible toy tests; the full active-inference planner is single-GPU and is a prototype-carefully lever, easy to overclaim.

Developmental role. A unifying lens over exploration, attention, and planning, more valuable as organizing theory than as a single module.

Dependencies. Subsumes parts of 6.4 and 6.5. Worth implementing only after those exist, as a principled re-derivation, with the simpler baselines kept as the comparison so that "free energy" never becomes an unfalsifiable explanation.

### 6.10 Open-endedness: quality diversity, autotelic goals, environment generation

Biology and theory. [Established for the behavioral phenomenon, theoretical for the mechanism] Open-ended learning is the hallmark of development and of evolution: an ever-expanding set of skills and goals, where new competencies create the preconditions for newer ones (stepping stones). Autotelic agents set their own goals. Cultural accumulation and social scaffolding (imitation, teaching, language) compress and transmit the search.

Computational abstraction. A population or archive of behaviors or solutions that grows in diversity, a goal-generation mechanism, and ideally a co-evolving environment that keeps producing the right next challenge. Quality diversity (MAP-Elites) maintains an archive of high-performing solutions across a behavior space. Novelty search rewards behavioral novelty rather than objective progress. POET co-evolves agents and environments so that each environment is matched to the frontier of agent capability. Autotelic goal-conditioned agents sample their own goals, often from a learned goal space, and curriculum themselves toward harder ones.

JEPA mapping. The skill archive holds policies or goal-conditioned latents; behavior characterization can use V-JEPA latents (encode the resulting trajectory, characterize by its latent signature). Goals are image-goals or latent-goals (2.3), so the V-JEPA goal space is natural. Environment generation is external to V-JEPA and is the heaviest component. The minimal operational test of open-endedness for a frozen-JEPA agent: over a long run, does the agent generate or select new goals, improve on a growing archive of skills, discover stepping stones (skills that unlock other skills), transfer old skills to newly generated challenges, expand behavioral diversity, and avoid collapsing to a single local optimum. That is a measurable bundle, and the honest expectation is that a single agent on a fixed environment exhibits little of it, while an agent plus environment-generation plus an archive exhibits some.

ML analogs. [Established ML] Novelty search (Lehman and Stanley 2011). MAP-Elites and quality diversity (Mouret and Clune 2015). POET (Wang et al. 2019) and Enhanced POET. Autotelic / goal-generation agents (IMGEP, Forestier et al.; CURIOUS; GoalGAN). Skill discovery (DIAYN, Eysenbach et al. 2019). Open-Ended Learning Team work (XLand, DeepMind). The repeated finding: open-endedness is a system-and-population property, and single-agent fixed-environment setups plateau.

Feeds. Experiment 10 (minimal open-ended JEPA).

Failure mode. The system collapses to a fixed local optimum (no expanding diversity), detectable by archive-diversity-over-time. Curiosity does not create stepping stones (novelty without compounding), detectable by skill-reuse metrics. Open-endedness requires population-level search and a single agent cannot show it, which is the most likely outcome and is itself the contribution: a clear statement of what is missing.

Tractability. Lab-scale for the full version (environment generation plus population plus long runs equals real compute). A reduced single-agent autotelic version on a procedural environment is a rented-GPU prototype. This is the part of the program that honestly exceeds a solo year for the full ambition, and the corpus says so.

Developmental role. Open-endedness. The spine of the open-ended loop, Axis D's most ambitious end.

Dependencies. Requires everything else (memory, plasticity, curiosity, planning) plus an action space, an environment, and ideally environment generation. Should be attempted last, and a negative result here is expected and valuable.

---

## 7. Experiment bank

Ten experiments, each fully specified. Compute estimates assume your M3 Pro 18GB for cached-latent work and a single rented A100/H100 spot instance for environment-rollout work. "Seeds" means independent random seeds for the reproducibility claim; with Apple-Silicon Metal nondeterminism at temperature 0 running near 50% byte-identical in your own benchmarking experience, you should treat seed variance as real and report mean and spread, never a single run. Run the determinism sanity loop (repeat one configuration several times, measure spread) before trusting any cross-condition delta, exactly as you do for benches.

### Experiment 1: Baseline frozen-latent developmental harness (the gate)

Hypothesis. A frozen V-JEPA encoder plus a trainable latent predictor can be turned into a continual-learning system that demonstrably forgets and demonstrably adapts, reproducibly across seeds.

Biological mechanism. None; this is the substrate-readiness test that every biological mechanism depends on.

ML analog. Standard continual-learning task-incremental baseline.

JEPA attachment. After the frozen encoder, on cached latents. Train a small predictor on domain A, continue on B, C, D, evaluate all domains after each stage.

Dataset/environment. Cached latents from three or four distinct video domains. Concretely: a Something-Something-v2 subset (motion-centric), an Ego4D or EPIC-KITCHENS subset (egocentric, different statistics), and a synthetic moving-object simulator you control (so you can dial difficulty). Optionally DROID clips if accessible. Start pooled latents; go dense only if a task needs it.

Baseline. Ordinary sequential training, no replay, fixed learning rate, fixed architecture.

Ablation. Vary predictor capacity (so you can later attribute null results to capacity vs representation) and vary task-stream difficulty.

Controls. A joint-training upper bound (train on all domains at once) and a single-domain reference per domain. Forgetting is measured relative to these.

Metric. Per-domain latent-prediction loss (L1 or cosine), probe accuracy on a downstream label per domain, average accuracy across domains, backward transfer (forgetting), forward transfer, adaptation speed (steps to reach a threshold on a new domain), rollout horizon (steps until rollout error exceeds a threshold), calibration.

Success threshold. Measurable learning on each new domain and measurable forgetting of earlier domains (a backward-transfer gap that is clearly outside seed noise), reproducible across at least 5 seeds.

Failure threshold. No forgetting appears (stream too easy or predictor too large) or no learning appears (frozen latents inadequate for the chosen tasks, or predictor too small).

Null-result interpretation. If no forgetting: increase task dissimilarity or shrink the predictor; without measurable forgetting, no continual-learning mechanism can be evaluated. If no learning: run the Section 2.6 linear-probe diagnostic; if the target is not decodable from the frozen latent, switch to 2.1 dense features or change the task. This experiment exists precisely to make later biology tests meaningful.

Compute. Laptop. Latent extraction overnight; predictor training in hours.

Seeds. 5 minimum.

Expected failure mode. The most likely real outcome is too little forgetting with pooled latents on easy tasks; the fix is harder, more dissimilar tasks and a deliberately under-parameterized predictor.

Next step. Once the harness can fail measurably, proceed to Experiment 2.

### Experiment 2: Latent hippocampus (replay)

Hypothesis. A latent episodic memory buffer with replay improves the retention-adaptation frontier over no replay, without fully blocking new learning.

Biological mechanism. Hippocampal replay and complementary learning systems (6.2).

ML analog. Experience replay, prioritized replay, latent replay, generative replay (6.2).

JEPA attachment. Buffer of latent sequences plus targets; interleave replayed samples with the current domain during sequential training.

Dataset/environment. Same harness as Experiment 1.

Baseline. No replay (the Experiment 1 baseline).

Ablations (the variant grid). (1) no replay, (2) random replay, (3) surprise-prioritized replay, (4) reward-prioritized replay (if reward exists), (5) learning-progress-prioritized replay, (6) reverse replay, (7) time-compressed replay, (8) replay plus EWC, (9) replay plus a consolidation phase (offline replay-only epochs between domains), (10) replay plus memory pruning (cap buffer, evict by a rule).

Controls. Buffer-size sweep (so retention gains are attributed to scheme, not just to storing more data) and a "store-everything joint-training" upper bound.

Metric. Average accuracy across domains, backward transfer, forward transfer, forgetting, memory footprint, replay compute, adaptation speed.

Success threshold. At least one prioritization scheme beats random replay, and replay beats no-replay, on the retention-adaptation frontier (Pareto-dominant or clearly better at matched buffer size), across 5 seeds.

Failure threshold. No replay scheme beats no-replay at any buffer size, or all schemes tie random replay.

Null-result interpretation. Frozen latents may not preserve episodic distinctiveness (run the buffer-contents probe diagnostic). Retrieval may be too weak (ablate retrieval quality). Consolidation may need plasticity scheduling (defer to combined Experiment 3 plus 2). The task stream may not require memory (it should, by Experiment 1's design).

Compute. Laptop.

Seeds. 5.

Expected failure mode. Prioritization schemes tie random replay because the stream is too short for prioritization to matter; fix with longer streams and tighter buffers.

Next step. Carry the best replay scheme into Experiments 3 and 4 as a component.

### Experiment 3: Critical-period schedule (staged plasticity)

Hypothesis. A predictor trained with high early plasticity and later closure beats constant plasticity on sequential latent learning, and beats a tuned monotonic learning-rate decay.

Biological mechanism. Critical and sensitive periods, metaplasticity (6.1).

ML analog. Critical-learning-periods-in-deep-networks (Achille et al.); progressive freezing.

JEPA attachment. A per-module learning-rate gate on the predictor.

Dataset/environment. Same harness.

Baseline. Constant learning rate, and (the harder baseline) a well-tuned cosine or monotonic decay.

Ablations. (1) constant LR, (2) monotonic decay, (3) high-early-then-closure, (4) high-early plus replay, (5) task-boundary-triggered closure, (6) surprise-triggered reopening (uses 6.4 signals), (7) novelty-triggered reopening, (8) learning-progress-triggered reopening, (9) a small meta-learned plasticity controller.

Controls. The tuned-decay baseline is the control that prevents the "it is just a learning-rate trick" overclaim. Also a Fisher-information trace measurement across training to check for a critical-period signature.

Metric. Retention, adaptation, parameter drift, Fisher-information proxy, gradient norm, probe stability, the old/new performance frontier.

Success threshold. A staged or triggered schedule beats both constant LR and tuned decay on the frontier, across 5 seeds. Bonus: a Fisher-trace signature consistent with a critical period.

Failure threshold. No schedule beats tuned decay.

Null-result interpretation. Staged plasticity may require memory (then test variant 4 and combine with Experiment 2). Critical periods may not emerge with a frozen encoder and a small head (there may be nothing to critically shape). The schedule may need to be tied to uncertainty, not time (variants 6 to 8). If even the triggered variants fail, that is a strong, clean negative result: developmental timing per se does not help on a frozen substrate without other levers.

Compute. Laptop.

Seeds. 5.

Expected failure mode. Constant-vs-staged ties tuned decay; the informative comparison is the triggered (surprise/novelty/learning-progress) variants, which is where a real win, if any, lives.

Next step. Combine the winner with replay and uncertainty toward the Level-5 result.

### Experiment 4: Artificial neuromodulation (uncertainty-gated learning)

Hypothesis. A global modulation signal derived from the system's own uncertainty improves learning by gating plasticity, replay, attention, or exploration, and crucially learns from meaningful novelty while ignoring irreducible noise.

Biological mechanism. Dopamine RPE, acetylcholine expected uncertainty, norepinephrine unexpected uncertainty (6.4).

ML analog. Deep ensembles, MC-dropout, Bayesian surprise, neuromodulated/plastic networks.

JEPA attachment. Compute prediction error, ensemble disagreement, MC-dropout variance, calibration error, novelty (RND), learning progress; wire to gates on learning rate, replay priority, memory write, planner reliance, attention mask, module routing.

Dataset/environment. The harness, plus a mandatory noisy-TV construction: a region of the input or the latent that is high-entropy but unlearnable (e.g. injected noise), alongside a learnable-novelty region.

Baseline. Fixed learning rate, ungated.

Ablations. (1) fixed LR, (2) prediction-error-gated, (3) ensemble-disagreement-gated, (4) surprise-gated replay, (5) uncertainty-gated plasticity reopening, (6) uncertainty-gated planner-vs-reactive, (7) noisy-TV stress test across all gates, (8) explicit aleatoric/epistemic separation (point-prediction error vs a distributional predictor).

Controls. The noisy-TV region is the central control: a correct mechanism does not allocate plasticity, replay, or exploration to it. A calibration check (reliability diagram) on the uncertainty signal.

Metric. Learning from the learnable region (adaptation speed there), noise attraction (resources spent on the noisy region, should be near zero), retention of old skills, calibration error.

Success threshold. Gated learning adapts faster on learnable novelty, allocates near-zero resources to the noisy-TV region, and preserves old skills, beating the ungated baseline across 5 seeds.

Failure threshold. The gate fires on the noisy region (cannot separate learnable from noisy) or provides no benefit on learnable novelty.

Null-result interpretation. The signal is uncalibrated (improve calibration). Prediction error conflates noise with novelty (move to the distributional predictor for epistemic/aleatoric separation). The gate reacts too late (latency analysis, faster gate). The neuromodulatory analogy is too loose (then it is a Level-1 metaphor, not a Level-3 mechanism, and you say so). Better uncertainty may require probabilistic latent dynamics, which becomes a dependency.

Compute. Laptop (ensemble of small predictors).

Seeds. 5.

Expected failure mode. Point-prediction-error gating fails the noisy-TV test; ensemble disagreement or the distributional predictor is needed to pass it. Documenting that point error fails and disagreement passes is itself a clean result.

Next step. The passing gate becomes the trigger for Experiment 3's reopening and Experiment 2's prioritization, assembling the Level-5 combination.

### Experiment 5: Curiosity as self-curriculum

Hypothesis. Curiosity based on learning progress or information gain produces better training data than random exploration or passive viewing, and resists noisy-TV better than prediction-error or RND curiosity.

Biological mechanism. Intrinsic motivation, learning progress (6.5).

ML analog. ICM, RND, learning progress, VIME, empowerment (6.5).

JEPA attachment. Intrinsic reward into the planner's energy or the policy reward; or, in the passive setting, into a data-selection policy over clips and masked regions.

Dataset/environment. A controlled environment that explicitly contains learnable structure, irrelevant novelty, uncontrollable noise (noisy-TV), sparse reward, and compositional goals. Start with a toy gridworld or DMControl-pixels variant you can fully instrument; the data-selection variant can run on a fixed clip pool on the laptop first.

Baseline. Random exploration, and passive (uniform) data sampling.

Ablations. (1) random, (2) prediction-error curiosity, (3) ICM, (4) RND, (5) learning progress, (6) expected information gain, (7) empowerment, (8) a hybrid.

Controls. The four novelty types in the environment (learnable, irrelevant, uncontrollable, controllable-but-useless) are the controls that diagnose what each curiosity signal chases.

Metric. Environment steps (or data samples) to competence, fraction of time on learnable states, noise attraction, downstream task transfer, curriculum diversity, skill diversity, representation improvement (probe accuracy gain).

Success threshold. A curiosity variant reaches competence in fewer steps than random, with low noisy-TV fixation, and learning progress specifically shows the least noise attraction, across 5 seeds.

Failure threshold. No curiosity variant beats random, or all fixate on the noisy region.

Null-result interpretation. The curiosity signal is not grounded in learnability (then learning progress should be the only survivor; if even it fails, suspect the signal computation). The action space is insufficient for useful exploration. The frozen latent does not encode controllability (probe for it; if absent, ICM's filter cannot work). The environment is too closed-ended (no curriculum to find). The planner is too weak to exploit intrinsic rewards.

Compute. Data-selection variant: laptop. Environment variant with rollouts: rented single GPU for the rollout-heavy sweeps; the bottleneck is environment steps, not model size.

Seeds. 5, more if variance is high (curiosity is notoriously high-variance).

Expected failure mode. Prediction-error and RND chase the noisy-TV; learning progress and information gain do better. The expected clean result is exactly that ranking.

Next step. Feed the best curiosity signal into Experiment 10's open-ended loop.

### Experiment 6: Relational map over JEPA latents

Hypothesis. Explicit relational or map-like structure improves generalization and planning over novel rearrangements, and this help is much larger on V-JEPA 2.1 dense features than on V-JEPA 2.

Biological mechanism. Place/grid cells, cognitive maps, TEM, successor representations (6.6).

ML analog. Successor features, Slot Attention, GNN world models, neural graph planning (6.6).

JEPA attachment. A graph head over latent-derived nodes; a successor-representation head; object-centric slots; a spatial-map memory; relational replay.

Dataset/environment. Tasks that require relational generalization: object rearrangement, detour navigation, changed layouts, hidden-object persistence, relational transfer. A controllable simulator is best so you can construct held-out recombinations.

Baseline. A flat latent-only predictor, parameter-matched to the structured variants.

Ablations. (1) latent-only, (2) object-slot predictor, (3) graph predictor, (4) successor-representation head, (5) cognitive-map memory, (6) topological planner. Each run on both V-JEPA 2 and V-JEPA 2.1 (the central comparison).

Controls. The parameter-matched flat baseline (so gains are attributed to structure, not capacity). A held-out recombination test that a flat model is expected to fail.

Metric. Generalization to rearranged scenes, planning success, sample efficiency, relational-probe accuracy. Plus the 2-vs-2.1 delta as the headline comparison.

Success threshold. A structured head beats the parameter-matched flat baseline on recombination, and the gain is significantly larger on 2.1 than on 2, across 5 seeds.

Failure threshold. Structured heads tie the parameter-matched baseline on both substrates.

Null-result interpretation. The frozen latent lacks object factorization (especially on 2; this is the expected explanation for a 2-only failure). The graph or slot module adds capacity but not structure (parameter-matched control catches this). The task does not require relational generalization (the recombination control should ensure it does). On 2.1, if object-centric still fails, that bounds how dense the 2.1 features really are, which is a useful finding.

Compute. Laptop for successor representations and graph heads on cached dense latents; single GPU for object-centric. Dense-feature extraction from 2.1 is the storage cost; plan disk accordingly.

Seeds. 5.

Expected failure mode. Object-centric fails on 2 and works partially on 2.1; successor representations help modestly on both. The 2-vs-2.1 contrast is the publishable core.

Next step. Use the successful relational structure in the planner for Experiments 5 and 10.

### Experiment 7: Sparse / modular developmental predictor

Hypothesis. Sparse or modular predictors reduce catastrophic interference and improve continual learning, independent of any GPU speedup.

Biological mechanism. Sparse coding, k-WTA, lateral inhibition, E/I balance, homeostasis, neurogenesis, pruning (6.7).

ML analog. k-sparse autoencoders, MoE, Progressive Networks, PackNet, active-dendrites-plus-k-WTA continual learning (6.7).

JEPA attachment. Replace or augment the predictor with a sparse or modular variant.

Dataset/environment. The continual-learning harness from Experiment 1, with clear task structure to exploit.

Baseline. A dense predictor of equal parameter count.

Ablations. (1) dense, (2) sparse-activation, (3) k-WTA, (4) MoE, (5) uncertainty-routed MoE (uses 6.4), (6) task-routed MoE, (7) pruned, (8) growable (add a module per task or per high-novelty event).

Controls. Parameter-matched dense baseline (interference reduction must not be just more capacity). Wall-clock measurement (to honestly report that there is or is not a speedup, separating the representational claim from the compute claim).

Metric. Forgetting, task interference, modular specialization, routing entropy, compute cost, wall-clock, robustness to unit ablation, transfer, capacity efficiency.

Success threshold. A sparse or modular variant shows lower interference or better specialization than parameter-matched dense, across 5 seeds. Speedup is reported separately and not required for success.

Failure threshold. No variant beats parameter-matched dense on interference, specialization, or transfer.

Null-result interpretation. Routing overhead dominates with no benefit. Sparse modules fragment the data (per-module data counts too low). Over-specialization blocks transfer (forward-transfer metric). Task boundaries unclear so routing cannot key on them (routing entropy never drops). The predictor is too small for sparsity to matter (scale the predictor). Real compute benefit needs custom kernels (then it is a labeled custom-kernel candidate, deferred).

Compute. Laptop for the representational science (no speedup). Custom-kernel-candidate and rented-GPU for the wall-clock-efficiency claim.

Seeds. 5.

Expected failure mode. k-WTA and uncertainty-routed MoE reduce interference; the speedup does not materialize without kernels. The interference result is the keeper.

Next step. Combine the best structural variant with consolidation (the doctrine's prediction is that specialized modules are easier to protect).

### Experiment 8: Dendritic predictor

Hypothesis. Dendrite-like nonlinear subunits increase capacity-per-parameter or improve online adaptation around frozen latents.

Biological mechanism. Dendritic compartments, branch-specific plasticity (6.8).

ML analog. Active-dendrites networks (Iyer et al. 2022), gated nonlinear subunits.

JEPA attachment. Replace the predictor's MLP blocks with dendritic-subunit blocks; or add gated branches with context-dependent gating.

Dataset/environment. The harness, with an online-streaming variant (one pass, no epochs) to expose any online-adaptation advantage.

Baseline. A parameter-matched MLP predictor.

Ablations. (1) MLP, (2) dendritic-subunit, (3) gated-branch, (4) local branch updates, (5) branch-specific plasticity, (6) three-factor branch modulation (uses 6.4).

Controls. Parameter-matched MLP (the dendritic block must beat equal parameters, not just add them).

Metric. Prediction loss, adaptation speed, forgetting, parameter efficiency, stability under online learning, compute cost.

Success threshold. A dendritic variant beats the parameter-matched MLP on capacity-per-parameter or on online adaptation, across 5 seeds.

Failure threshold. No dendritic variant beats the parameter-matched MLP.

Null-result interpretation. The dendritic analogy adds complexity without benefit (most likely outcome). The benefit requires spiking or precise timing (then neuromorphic-only label). Backprop-trained MLP remains superior (expected). The task is too static to reward online adaptation (use the streaming variant).

Compute. Laptop.

Seeds. 5.

Expected failure mode. Modest or no gain on accuracy; possible small gain on context-gated continual learning specifically (the active-dendrites precedent suggests interference reduction is the place to look). A clean modest-or-null result.

Next step. If context-gating helps, fold it into the structural variant of Experiment 7.

### Experiment 9: Local learning head

Hypothesis. A local learning rule can train a small JEPA head competitively enough to support online adaptation under streaming constraints.

Biological mechanism. Hebbian, Oja, BCM, STDP, three-factor rules; local credit assignment (6.8).

ML analog. Feedback alignment, forward-forward, equilibrium propagation, predictive-coding learning (6.8).

JEPA attachment. Train a small head on the frozen latent with a local rule; or use a local rule as an auxiliary regularizer alongside backprop.

Dataset/environment. A simple latent classification or short-horizon prediction task on cached latents, in a streaming (online) regime where backprop's activation storage is a real cost.

Baseline. The same head trained with backprop.

Ablations (the rule grid). Hebbian, Oja, BCM, three-factor, feedback alignment, forward-forward, predictive-coding learning, equilibrium propagation.

Controls. The backprop head (the accuracy ceiling). A memory-cost measurement (local rules should win on activation memory if they win anywhere).

Metric. Accuracy or loss vs backprop, memory use, online stability, adaptation speed, a biological-plausibility descriptor (locality, no weight transport, no separate backward pass).

Success threshold. A local rule reaches within a stated margin of backprop accuracy while using less memory or adapting online more stably, across 5 seeds.

Failure threshold. All local rules fall far short of backprop with no compensating memory or online advantage.

Null-result interpretation. Local rules fail with depth (use a shallower head, or accept that they are auxiliary-only). Credit-assignment bottleneck dominates. Local learning works only as an auxiliary regularizer alongside backprop (a legitimate, modest positive result).

Compute. Laptop. Toy-test tractability.

Seeds. 5.

Expected failure mode. Local rules underperform backprop on accuracy; the honest contribution is on memory cost and online stability, or a negative result. This is explicitly a negative-result-friendly experiment.

Next step. If three-factor works on the head, connect its third factor to the neuromodulatory signal from Experiment 4 and report the small synthesis.

### Experiment 10: Minimal open-ended JEPA

Hypothesis. A JEPA agent combining curiosity, memory, plasticity control, and goal generation expands its own curriculum beyond a fixed task list, discovering and reusing skills over a long run.

Biological mechanism. Autotelic goals, intrinsic motivation, skill archives, open-ended development (6.10).

ML analog. POET, novelty search, MAP-Elites, IMGEP, DIAYN (6.10).

JEPA attachment. Frozen encoder; action-conditioned predictor (the AC model, ideally 2.1); episodic memory; replay; plasticity controller; curiosity (the Experiment 5 winner); a skill archive; a goal generator (sample image-goals or latent-goals); a procedural or generated environment.

Dataset/environment. A procedural environment with compositional structure (so stepping stones exist) and ideally a lightweight environment-generation mechanism.

Baseline. A fixed-task agent (no goal generation), and a random-goal agent (goals sampled uniformly, no autotelic curriculum).

Ablations. Add components one at a time: curiosity only, plus memory, plus plasticity control, plus goal generation, plus environment generation. This ablation ladder is the experiment's spine because it shows which components are load-bearing for open-endedness.

Controls. Skill-reuse and archive-diversity baselines; a collapse detector (does behavior converge to one strategy).

Metric. Number of distinct skills, skill reuse, archive diversity, transfer to generated tasks, stepping-stone discovery (skills that unlock other skills), non-collapse over time.

Success threshold. The full system discovers increasingly diverse behaviors and reuses old skills on new generated challenges, beating both baselines on archive diversity and transfer over a long run.

Failure threshold. The system collapses to a fixed local optimum, or shows no skill reuse, or matches the random-goal baseline.

Null-result interpretation. Open-endedness requires population-level search (single agent insufficient; the most likely outcome). Open-endedness requires richer embodiment than the substrate provides. Latent prediction alone is insufficient without environment generation. The skill archive or curiosity signal is too weak. Individual-agent learning is the wrong scale. Each of these is a clean, publishable statement of what is missing, which is the realistic contribution of this experiment.

Compute. Rented GPU, the most expensive experiment, dominated by long runs and rollouts. Honestly partly lab-scale for the environment-generation variant; the single-agent autotelic version is a feasible rented-GPU prototype.

Seeds. 3 to 5 (expensive; report spread).

Expected failure mode. The single agent plateaus and the environment-generation component is what is missing; the contribution is a precise account of why, plus whatever partial open-endedness the autotelic version does show.

Next step. This is the program's capstone; its negative result feeds the final judgment (Section 14) on whether latent prediction is enough.

---

## 8. Dependency graph

The build order is not arbitrary. Each node lists what must exist before it and what it unlocks.

```text
                          [Frozen V-JEPA encoder]
                                   |
                       [Cached latent datasets]
                                   |
                    [E1: Baseline harness] (GATE)
                    must fail measurably before anything else
                          /        |         \
                         /         |          \
            [E2: Replay]   [E3: Staged plasticity]   [linear-probe diagnostic]
                  \            /        \              (runs throughout)
                   \          /          \
              [E4: Uncertainty gating] <--+  (gating triggers reopening in E3,
                   |        |                 prioritization in E2)
                   |        |
        +----------+        +------------------+
        |                                      |
 [E5: Curiosity]                    [E2+E3+E4 COMBINED]
 (needs env or clip pool)            = LEVEL-5 RESULT (headline)
        |                                      |
        |                          [E7: Sparse/modular]  [E6: Relational maps]
        |                          (pairs w/ consolidation) (needs 2.1 dense)
        |                                      |
        |                          [E8: Dendritic] [E9: Local learning]
        |                          (mostly negative/modest, toy-test)
        |                                      |
        +------------------+-------------------+
                           |
                  [E10: Open-ended JEPA] (CAPSTONE)
                  needs E2,E3,E4,E5 + action + env + archive
                  honestly part lab-scale; negative result expected & valuable
```

Hard ordering rules. E1 gates everything; do not run a single biological experiment until the baseline forgets and adapts reproducibly. E4 must pass its noisy-TV test before its signal is trusted as a trigger in E3 or a prioritizer in E2. The Level-5 combination (E2+E3+E4) is the headline and must be built from individually-understood components, never as a first move. E6's object-centric branch is wasted on V-JEPA 2 and should run on 2.1. E10 should be attempted last.

Conflicts and joint-test requirements. Staged plasticity and tuned learning-rate decay are confounded; always run decay as the control (E3). Structure (E7) and capacity are confounded; always parameter-match. Curiosity is meaningless without a learnable-vs-noisy environment design (E5); never test it on a single homogeneous environment. Local learning (E9) and accuracy are a near-certain loss; test it on memory and online stability instead, where it might win.

---

## 9. Compute tractability map

Per the doctrine, tractability is a scoping label, not a rejection. Every lever is placed; none is dropped for being inconvenient.

| Class | Meaning | Levers / experiments |
|---|---|---|
| Laptop feasible (M3 18GB) | Cached-latent work, small modules | E1, E2, E3, E4, E7 (science only), E8, E9; staged plasticity, replay, EWC/SI, uncertainty gating, sparse/k-WTA interference, dendritic, local-learning heads, successor representations, graph heads on cached dense latents, data-selection curiosity |
| Single rented GPU (hours, spot) | Environment rollouts, object-centric, wall-clock claims | E5 (env variant), E6 (object-centric), E7 (speedup claim), E10 (single-agent autotelic) |
| Custom-kernel candidate | Real compute benefit needs Triton/CUDA/Metal/MLX sparse or routing kernels | Sparse/MoE speedup, block-sparse, event-based batching, k-WTA throughput |
| Neuromorphic / simulation-only | Needs spiking hardware or event cameras, or is sim-only | STDP, spiking networks, event-based computation, full equilibrium propagation at scale |
| Theoretical-for-now | Important, not cleanly implementable on this substrate yet | Oscillations / communication-through-coherence, global workspace as a mechanism, empowerment at scale, full active-inference planner, full POET environment co-evolution |

The encoder is the only large object and you never train it, so the dominant practical cost across the laptop-feasible tier is your time, not compute. Cache latents once (overnight on MPS), then iterate against cached latents for free. Rent a GPU only for the rollout-bound and wall-clock-claim experiments, and only for hours. This maps cleanly onto how you already run benches: pay the expensive pass once, instrument carefully, iterate cheap, and treat Metal nondeterminism at temperature 0 as a measured quantity rather than an assumption.

---

## 10. Plans

These adapt the source document's generic plans to a solo researcher on an M3 Pro 18GB with occasional rented GPU. They front-load the cheap high-leverage spine and defer the expensive open-ended capstone.

### 10.1 Twelve-week plan (solo, laptop-first)

- Weeks 1 to 2, substrate and reading. Read V-JEPA 2 and 2.1 and the 2-AC sections in full. Stand up the repo, load ViT-L, run the demo, confirm deterministic latent extraction on MPS (your determinism sanity loop). Build the lever table into a living spreadsheet. Deliverable: a working encoder, a determinism report, and the ranking table.
- Weeks 3 to 4, infrastructure (E1). Cache pooled latents from three domains. Train the baseline predictor and a probe. Build the eval harness (forgetting, adaptation, rollout, calibration) with seed-averaged reporting. Deliverable: the gating baseline that forgets and adapts reproducibly, or a documented reason it does not (and the fix).
- Weeks 5 to 6, replay (E2). Implement the buffer and the prioritization grid. Run the retention-adaptation frontier across schemes and buffer sizes. Deliverable: replay frontier curves and the best scheme.
- Weeks 7 to 8, staged plasticity (E3). Implement the schedule and triggered variants, with tuned decay as the control and a Fisher-trace measurement. Deliverable: the plasticity frontier and the decay-vs-staged verdict.
- Weeks 9 to 10, uncertainty and neuromodulation (E4). Build the ensemble, the gates, and the mandatory noisy-TV. Deliverable: the learnable-vs-noisy result and a calibrated (or documented-uncalibrated) signal.
- Weeks 11 to 12, the combination and a curiosity prototype. Combine E2+E3+E4 toward the Level-5 result; stand up the data-selection curiosity variant on a clip pool. Deliverable: the first developmental-JEPA prototype report, with the combined-vs-individual comparison as the headline.

### 10.2 Six-month plan

- Month 1: full literature map; E1 baseline; the V-JEPA 2 vs 2.1 latent-quality comparison (plays to your benchmarking strengths and is publishable on its own as a frozen-eval study).
- Month 2: E2 replay and consolidation; buffer-retrieval analysis; the latent-geometry bottleneck analysis (the Section 2.6 diagnostic, run systematically).
- Month 3: E3 staged plasticity; the meta-learned plasticity controller; critical-period ablations and Fisher traces.
- Month 4: E4 uncertainty-gated learning; artificial neuromodulation; noisy-TV battery; epistemic/aleatoric separation via a distributional predictor.
- Month 5: E5 curiosity and self-curriculum (rented GPU for the env variant); E6 relational maps on 2.1 (the 2-vs-2.1 contrast).
- Month 6: integrated system; the E2+E3+E4 Level-5 result written up; a negative-result appendix; an open-endedness pilot. Target paper: "Staged Plasticity and Latent Replay Improve Continual Adaptation in Frozen Video World Models," with the 2-vs-2.1 dense-feature comparison as a strong secondary contribution.

### 10.3 One-year, thesis-scale plan

- Q1: substrate and baseline; the frozen-JEPA continual-development benchmark itself as a contribution (a reusable harness, datasets, metrics, and seed protocol).
- Q2: memory, replay, consolidation; staged plasticity; neuromodulatory gating; the Level-5 combination.
- Q3: curiosity and active curriculum; cognitive maps and relational structure (2-vs-2.1); sparse modular predictors (interference science plus a kernel-deferred speedup note); the local-learning and dendritic negative-result battery.
- Q4: integrated developmental JEPA; the open-endedness pilot (single-agent autotelic, with the honest population-scale caveat); thesis writing; a full negative-result appendix.

Thesis thesis statement to defend: a frozen self-supervised video world model can be moved toward developmental behavior through external mechanisms for memory, plasticity scheduling, uncertainty-gated learning, and active curriculum, and the gains are real but bounded by frozen-latent geometry, while open-ended intelligence requires embodiment, environment generation, and structural adaptation beyond latent prediction alone.

---

## 11. Integrated architecture

The first integrated developmental loop, refined for the 2.1 substrate:

```text
            environment frame / video clip
                        |
              [Frozen V-JEPA 2.1 encoder]   <- inherited perceptual cortex, never trained
                        |
        dense per-token latents + pooled summary
                        |
        +======== Developmental workspace ========+
        |                                          |
        |  [latent predictor] <----+               |
        |        |                 |               |
        |  predicted latent   [uncertainty est.]   |  (ensemble / dropout / distributional)
        |        |                 |               |
        |  compare to target ---> surprise signal  |
        |        |                 |               |
        |        v                 v               |
        |  [episodic latent memory] <- write gate (surprise / reward / learning-progress)
        |        |                 |               |
        |  [replay scheduler] --> [consolidation loss] (EWC/SI)
        |        |                 |               |
        |  [plasticity controller] <- triggered reopening (surprise / novelty / LP)
        |        |                 |               |
        |  [curiosity module] --> intrinsic reward / data-selection
        |        |                 |               |
        |  [noisy-TV detector] (epistemic vs aleatoric guard)
        |        |                 |               |
        |  [optional: relational/graph head]  (2.1 dense features)
        |  [optional: sparse modular router]  (interference reduction)
        |  [optional: action-conditioned predictor + MPC planner] (image goals)
        |                                          |
        +==========================================+
                        |
        action  /  prediction  /  probe output  /  memory write  /  next-data choice
                        |
                     (loop)
```

The loop, in words: observe, encode (frozen), predict the next latent, compare to the EMA-free target latent, estimate uncertainty, decide whether the surprise is learnable or noise, decide whether and how much to update, write to memory if the event is worth keeping, replay old memories interleaved, consolidate with a weight-space penalty, optionally prune or specialize modules, choose the next action or the next data to sample, repeat. This is a growing system with a small number of load-bearing parts, not a bag of tricks, and the program's job is to find out which parts actually bear load on a frozen substrate.

---

## 12. Negative-result taxonomy

Every null result in this program must be attributed to exactly one of these, with the named detection method. This is the doctrine's ten categories, operationalized.

1. The biological mechanism was mapped badly. Detection: a cleaner re-mapping changes the result. Guard: pre-register the mapping and its truth-level label before running.
2. The mechanism requires an unfrozen encoder. Detection: the same mechanism works when a small adapter on the encoder is unfrozen (a deliberate, scoped violation of the freeze, run only as a diagnostic). Guard: this is the most important boundary the program maps.
3. The frozen latent lacks the needed information. Detection: the Section 2.6 linear-probe diagnostic shows the target variable is not decodable. Guard: run the probe before every mechanism that depends on a specific variable.
4. The predictor or head was too weak. Detection: scaling the head changes the result. Guard: always include a capacity ablation.
5. The task was too easy. Detection: no forgetting or trivial adaptation in the baseline. Guard: E1's design plus a difficulty dial.
6. The task was too hard. Detection: no learning even at the joint-training upper bound. Guard: the upper-bound control.
7. The mechanism only works with embodiment or action. Detection: it fails in the passive setting and works in the action setting. Guard: test both where feasible.
8. The mechanism only works combined with another lever. Detection: it fails alone and works in combination. Guard: the dependency graph predicts which combinations to try, but only after the isolated null is documented.
9. The mechanism is computationally incompatible with the tested hardware. Detection: it needs custom kernels or neuromorphic hardware for the claimed benefit. Guard: separate the representational claim (laptop-testable) from the compute claim (kernel-deferred).
10. The mechanism is conceptually irrelevant to JEPA-style latent prediction. Detection: even a clean mapping with a capable head on a suitable task and substrate shows nothing. Guard: this is the strongest negative result and the most valuable; it bounds the substrate.

The rule that makes failure informative: a result that lands in none of these, or in "it just did not work," is not an acceptable stopping point. Re-run until the failure is attributable.

---

## 13. Open theoretical questions

1. Is there any developmental content in "staged plasticity" on a frozen encoder beyond learning-rate scheduling, given that there is no representation to critically shape, only a head? The Fisher-trace measurement (E3) is the empirical handle, but the conceptual question stands.
2. Does the frozen encoder's stability (latents never go stale) make weight-space consolidation (EWC/SI) redundant with data-space consolidation (replay), or are they complementary? E3-plus-E2 answers it empirically; the theory is open.
3. Can point-prediction error ever separate epistemic from aleatoric uncertainty on this substrate, or is a distributional latent-dynamics head strictly necessary? This bears on whether curiosity can be noisy-TV-proof without upgrading the predictor.
4. What is the right state space for a successor representation over a continuous video latent: a learned codebook, a clustering, or a continuous SR, and does the choice determine whether relational structure helps?
5. Is open-endedness ever exhibitable by a single agent on a fixed environment, or is it definitionally a population-and-environment property? If the latter, the frozen-JEPA single-agent program has a hard ceiling that no lever crosses, and that is worth proving cleanly.
6. Does V-JEPA 2.1's dense, hierarchically-supervised representation change the answer to "is latent prediction enough," or only raise the ceiling without changing the kind of thing the substrate can become?
7. Where exactly is the boundary between "fixable by bolt-on" and "requires retraining" for object permanence specifically, given that 2.1 improves but does not guarantee it?

---

## 14. Final judgment: is latent prediction enough?

The honest answer the program is built to defend: latent prediction is necessary but not sufficient.

Necessary, because a compressed predictive model of temporal structure is exactly what a developmental agent needs as its perceptual substrate, and V-JEPA provides a genuinely good one (state-of-the-art motion and anticipation, and in 2.1 dense spatial structure usable for depth and navigation). Inherited perception is real and valuable, and freezing it is a defensible and economical choice.

Not sufficient, for reasons the program will quantify rather than assert. Development also requires persistent memory (which the encoder does not have and which bolt-on episodic memory can supply), active data selection (curiosity, which the substrate does not have and which works only with an environment), action and embodiment (which the AC model partially supplies and the passive encoder does not), plasticity control and consolidation (the spine of this program, and the place where the clearest positive results will live), goal generation, and an environment rich enough to create stepping stones (which is the part that honestly exceeds a solo year and is the most likely hard ceiling).

The frozen V-JEPA encoder is best understood not as the whole mind but as the inherited perceptual cortex of a larger developmental agent. The program's strongest realistic contribution is therefore not "we built a developmental mind" but "we built a controlled frozen-JEPA developmental testbed and showed which biological mechanisms measurably improve continual adaptation, memory, self-curriculum, and robustness on it, and which fail under clear, interpretable conditions, and exactly where the frozen-latent ceiling and the embodiment requirement bind." Development may require more than perception. It may require a life. This program maps how much of a life can be built around an inherited cortex before that becomes the limiting truth.

---

## 15. Corpus roadmap: Volumes II and III

This volume did the substrate, the spine, and the experiment bank to real depth. The remaining levers from the master list get structured one-pager treatment in the next volumes, each following the same dossier shape (biology, computational abstraction, JEPA mapping, ML analogs, minimal experiment, failure mode, tractability, developmental role, dependencies), with truth-level labels and a placement on the evidence ladder and the ranking table.

Volume II, the remaining mechanisms (one page each):

- Developmental timing and plasticity: sensitive periods (distinct from critical), Piagetian staging, scaffolding and zone of proximal development, maturational constraints, progressive unfreezing, perineuronal-net analogs, inhibitory maturation, learned-optimizer plasticity controllers.
- Memory and consolidation: reverse replay and time-compressed replay as standalone dossiers, generative replay, memory indexing, reconsolidation, forgetting-as-pruning, dreaming and sleep-like phases as offline optimization.
- Prediction and uncertainty: hierarchical prediction, counterfactual prediction, efference copy, cerebellar forward models, Kalman-filter analogs, Bayesian surprise, confidence calibration as its own lever.
- Neuromodulation and control: serotonin-like patience and discounting, model-based/model-free arbitration, habit formation, goal-directed control, planning-reactive switching.
- Curiosity and self-data: novelty search as data selection, empowerment, active sensing and saccade-like sampling, boredom and habituation, play.
- Cognitive maps and relational structure: head-direction and border-cell analogs, object files, latent binding, compositional world models, topological maps, scene-graph prediction.
- Attention and bottlenecks: selective attention, salience maps, attention schema, conscious-access and low-bandwidth bottlenecks, working-memory limits, chunking, context gating.

Volume III, the hardware-challenging and open-ended frontier (one page each, with heavier theoretical-and-neuromorphic labeling):

- Sparse and structural: lateral inhibition and E/I balance as standalone, homeostatic plasticity, structured vs unstructured vs 2:4 vs block sparsity as a comparison dossier, neural reuse, degeneracy, module birth and death dynamics.
- Local learning and dendrites at depth: target propagation, synthetic gradients, energy-based learning, the full backprop-alternatives comparison with a shared small-head benchmark.
- Spiking and neuromorphic: STDP, event-based computation, neuromorphic deployment, with simulation-only experiment sketches and the explicit note that these score 1 on solo compute feasibility and belong to a larger program.
- Embodiment and open-endedness at depth: affordances, tool use, options and hierarchical RL, skill libraries, social learning and imitation, language as developmental instruction (using the LLM-aligned V-JEPA head), cultural accumulation, teacher-student learning, and a full POET-style environment-generation design flagged lab-scale.

Each Volume II and III dossier ends with the same question this volume asks of every lever: does it move a frozen-JEPA system toward development, at what compute class, with what null interpretation, and is it worth a solo researcher's week. The ranking table in Section 5 already gives the provisional verdict; the volumes justify each one.

---

### Closing note on method

The doctrine that governs this corpus, restated so it governs the volumes too: survey maximally, rank ruthlessly, test progressively, combine only after isolated mechanisms are understood. Prefer specificity over elegance. Prefer tested mechanisms over beautiful metaphors. Prefer informative failure over vague success. Do not avoid hard mechanisms because they are hard; scope them. Do not worship biology, GPUs, or current frameworks. Map the frontier, then make it testable. This volume is the map of the part of the frontier nearest to the ground, drawn carefully enough to start walking.
