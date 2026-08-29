# Substrate Final Revision Research Survey

Access date: 2026-07-28

Status: pre-architecture research record

## Decision boundary

The historical closure result remains `terminal_closed_null`. The research below
does not alter it. In particular, no paper or repository establishes that
modularity, biological metaphor, a latent state, a world model, or model
plurality is intrinsically cognitive.

The strongest design implication is narrower:

1. owned persistent state should be independent of transient model context;
2. mutations should be recorded as typed, replayable events;
3. materialized cognitive state should be reconstructable and covered by the
   same semantic digest as checkpoints;
4. optional learned perception, prediction, routing, and compression should sit
   behind replaceable contracts;
5. every extra architectural mechanism must survive an equal-resource test
   against the simplest task-independent persistent core.

This favors a small event-sourced persistent kernel as the initial
`simplest_sufficient` candidate. It does not preselect the tournament winner.

## Mechanism survey

| Idea | Core mechanism and problem addressed | Evidence boundary and limitations | Cost and Substrate fit |
| --- | --- | --- | --- |
| Event sourcing | Append-only typed events plus deterministic projections separate mutation history from current state and permit replay, audit, rollback, and crash recovery. | ESAA reports two software-agent case studies, not a cognitive advantage experiment. Event logs can become transcript replay unless projections own semantic state. | Low CPU and disk cost; no tensor or LLM required; directly incremental. Strong fit for identity and checkpoint coverage. |
| Selective state-space models | Input-conditioned state transitions selectively retain or forget information with linear sequence scaling. | Mamba demonstrates sequence-model efficiency and quality, not persistent identity or developmental ownership. Learned latent state is hard to audit and expensive to train. | Tensor training required for full use; optional future compressor or predictor, not the authority plane. |
| Recurrent independent mechanisms | Sparsely communicating recurrent groups encourage specialization and systematic generalization. | RIMs supports a modular inductive bias on bounded tasks. It does not show that modules beat a fair monolith that implements the same organization. | Tensor training required; moderate-to-high research cost. Candidate only. |
| Graph dynamics | Object-and-relation nodes with learned message passing provide structural bias for physical prediction and system identification. | Graph-network results concern physical systems and do not imply general ontology or cognition. Graph construction errors can dominate. | Tensor-backed versions are costly; an explicit graph projection is cheap and interpretable. Candidate world-model projection. |
| Learned world models | Latent dynamics predict future representations and rewards; policies improve through imagined rollouts. | DreamerV3 has strong control evidence across many domains, but training cost and reward specification are substantial. A world model can be independently useful without being an identity substrate. | High tensor and environment cost; optional model adapter after a standalone benchmark. |
| Predictive video representations | Joint-embedding prediction learns temporally useful video representations; action-conditioned variants support planning. | V-JEPA 2 reports video understanding and bounded robot planning. It does not establish persistent cognitive organization, and checkpoints are large. | High model cost; replaceable perception/world-model organ only. |
| Global workspace | Capacity-limited selection and broadcast coordinate otherwise separate processes. | Functional theories motivate an interface, not a performance result for this repository. A workspace can be decorative if no decision changes under ablation. | Cheap explicit prototype; requires activity and ablation evidence. |
| Blackboard systems | Specialists communicate through a shared, structured problem state. | Coordination can reduce duplicated context, but a blackboard can collapse into a monolithic dictionary. | Low implementation cost; scientifically distinct only if scheduling/broadcast changes behavior. |
| Continual replay | Rehearsal or generated replay balances new learning with retention of prior competence. | Continual-learning evidence shows the stability-plasticity problem is real. Replay can leak held-out outcomes or preserve poisoned examples. | Memory and retraining costs vary; requires quarantine, retention tests, and rollback. |
| Task-free continual learning | Distribution discrepancy triggers capacity or memory changes without declared task boundaries. | Evidence remains benchmark-specific and tensor-dependent. Expansion can grow without bound. | Optional future learning policy; not needed for the first persistent kernel. |
| Model/expert routing | Sparse routing selects bounded experts while controlling activation cost; expert-choice routing can improve load balance. | MoE results concern trained neural layers. Named models are not independent organs unless each has standalone value and distinct checkpoints. | Optional model-fabric policy; simple deterministic routing is adequate for structural tests. |
| Unified agent memory policies | Memory operations become explicit actions for storing, retrieving, updating, summarizing, and discarding. | Recent agent-memory results are promising but LLM- and benchmark-dependent. Learned memory policies may optimize shortcuts. | Useful contract shape; training is deferred until a non-saturated benchmark exists. |
| Causal graphical models | Interventions alter a declared mechanism while observations condition on it; counterfactuals preserve unchanged background variables. | Causal conclusions require assumptions and identifiable structure. Language explanations alone are not interventions. | Small symbolic models are cheap and auditable; strong fit for controlled canaries. |
| DINOv2 | Self-supervised visual features transfer to depth and segmentation tasks. | A visual encoder is not grounding by itself, and feature quality depends on domain. | Apache-2.0 weights; tensor runtime required. Candidate only. |
| SAM 2 | Promptable image/video segmentation with memory across frames. | Segmentation is not object permanence, causality, or active perception. | Apache-2.0 code and checkpoints; substantial tensor memory. Candidate only. |
| RAFT and CoTracker | Dense optical flow or point trajectories expose actual temporal pixel structure. | Camera motion and object motion remain confounded without geometry; CoTracker's main license is noncommercial. | RAFT is a possible optical-flow baseline; deterministic array motion remains the cheap structural fixture. |
| Depth Anything V2 | Monocular relative-depth estimation from real images. | Scale is ambiguous and larger checkpoints have noncommercial licenses. A depth map alone is not 3D understanding. | Small checkpoint is Apache-2.0 and plausible for a later bounded tournament. |
| VGGT and DUSt3R | Multi-view inference estimates camera parameters, point/depth maps, and 3D structure. | High memory use and license restrictions require exact checkpoint review. Reconstruction does not supply goals or identity. | Deferred until a 3D component benchmark justifies download. |
| Whisper and wav2vec 2.0 | Speech models operate on acoustic waveforms and learn robust or self-supervised speech representations. | ASR can hallucinate and does not cover general audio events. Transcripts must not replace waveform evidence. | Whisper code/weights are MIT; small models are feasible but not acquired without a task-specific benchmark. |
| BEATs | Acoustic tokenizers and masked prediction learn general audio-event representations. | Audio classification is not cross-modal event grounding. Model provenance and license must be pinned before use. | Tensor runtime and checkpoint required; candidate audio-event organ. |
| Active perception | The agent chooses costly observations or interventions to reduce decision uncertainty. | Free oracle views invalidate the mechanism. Value must be measured net of sensing cost. | Cheap to test with generated arrays and withheld views; included in canaries. |
| Actor systems | Isolated state owners communicate through messages and supervision boundaries. | Operational fault isolation is not cognitive evidence; distributed ordering complicates replay. | Useful only if failure recovery justifies complexity. Candidate rejected when a single process is equivalent. |

## Primary sources

- [ESAA: Event Sourcing for Autonomous Agents](https://arxiv.org/abs/2602.23193)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893)
- [Graph Networks as Learnable Physics Engines](https://arxiv.org/abs/1806.01242)
- [DreamerV3](https://arxiv.org/abs/2301.04104) and its [official implementation](https://github.com/danijar/dreamerv3)
- [Continual-learning replay thesis](https://arxiv.org/abs/2007.00487)
- [Task-Free Continual Learning](https://arxiv.org/abs/2210.06579)
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [Agentic Memory](https://arxiv.org/abs/2601.01885)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [official repository](https://github.com/facebookresearch/vjepa2)
- [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- [SAM 2](https://arxiv.org/abs/2408.00714) and [official repository](https://github.com/facebookresearch/sam2)
- [RAFT](https://arxiv.org/abs/2003.12039)
- [CoTracker](https://github.com/facebookresearch/co-tracker)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) and [official repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [VGGT](https://github.com/facebookresearch/vggt)
- [DUSt3R](https://github.com/naver/dust3r)
- [Whisper](https://arxiv.org/abs/2212.04356) and [official repository](https://github.com/openai/whisper)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [BEATs](https://arxiv.org/abs/2212.09058) and [official implementation](https://github.com/microsoft/unilm/tree/master/beats)

## Acquisition decision

The preflight host is an Apple M3 Ultra with 96 GB unified memory, but only
approximately 68 GiB of free disk was available and the frozen Python
environment had no PyTorch installation. No model or corpus was downloaded.
The first sensorium implementation therefore has to process real array and
waveform structures using pinned NumPy only. Learned components remain
replaceable candidates and must win a bounded component tournament before
acquisition.

## Unresolved research conditions

- No reviewed source demonstrates that a modular architecture is necessary for
  the functions already matched by S2.
- No reviewed source licenses an unqualified Nous claim.
- Learned latent state has not yet shown sufficient incremental value to offset
  its training, reproducibility, and interpretability cost in Substrate.
- Real model acquisition remains contingent on a predeclared task, standalone
  benchmark, checksum, exact license, resource parity, and fallback.
