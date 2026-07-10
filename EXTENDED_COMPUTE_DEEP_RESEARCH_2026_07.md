# Extended-compute deep research: Form Substrate

Snapshot: 2026-07-10, America/Toronto  
Decision: **do not procure or rent extended compute yet**  
Machine-readable scope: `proof/EXTENDED_COMPUTE_REQUIREMENTS.json`

## Executive answer

Nothing in the current Form Substrate program is empirically Studio-scale, accelerator-required, or distributed-compute-required.

The complete current registry has 195 rows. After correcting stale tier labels and the now-published V-JEPA 2.1 checkpoints, their first blockers are:

| Primary category | Current registry rows | Meaning |
|---|---:|---|
| 1 | 167 | Already runnable on the measured M3 Pro envelope |
| 2 | 11 | Local implementation or matched-control work |
| 3 | 17 | Data, cache, rights, or provenance intake |
| 4 | 0 | Runnable only as a slow serial job |
| 5 | 0 | Requires local factorization to fit |
| 6 | 0 | External environment or interaction infrastructure is not the first blocker for a current registry row |
| 7 | 0 | Unpublished or unavailable upstream model |
| 8 | 0 | Measured larger-compute benefit, not necessity |
| 9 | 0 | Measured larger-compute scientific necessity |

The wider 289-row planning matrix, which adds F21-F66, 17 cross-cutting frontiers, W0-W11, and dossier pillars 7-25, also has zero category 8 or 9 rows. That is stricter than saying a GPU would be convenient. No next-rung benchmark has yet measured the benefit, so even category 8 is unearned. It is also stricter than saying every future experiment will remain local: category 9 is available, but only after a named fidelity, power, or real-time target survives the local feasibility attack and a bounded next-rung parity pilot.

The best use of additional compute today is not a larger inherited encoder. It is more independent referents and environments, stronger controls, longer developmental histories, interventions, architecture diversity at matched active FLOPs, and exact failure discovery. The immediate program is therefore a capability-density program, not a parameter-scaling program.

## Evidence grammar and claim boundary

This report uses five evidence tags:

- **[M] Measured:** a current repository receipt or a value directly extracted from one.
- **[P] Published:** a primary paper, official repository, official dataset page, or first-party hardware/runtime documentation.
- **[C] Calculated:** arithmetic from named measured or published inputs.
- **[E] Estimated:** an engineering range that still needs calibration.
- **[H] Hypothesis:** a falsifiable scientific proposal, not evidence.

No execution receipt here is evidence of consciousness, sentience, general intelligence, or intrinsic agency. Workspace, self-model, confidence, report, curiosity, and broadcast are theory-neutral operational variables only.

The snapshot was taken in a dirty working tree with two unrelated long-running local jobs. No large sizing probe, cloud rental, corpus download, dataset-term acceptance, or external mutation was performed for this research. In-progress CM7 and SANPO outputs are not used as durable evidence until their final receipts close.

## Repository audit

### What the current host has actually demonstrated

**[M]** `proof/STUDIO_READINESS_CURRENT_HOST.json` records an Apple M3 Pro with 19.3 GB unified memory, MPS available, a 180-minute bounded wall, one-heavy-process discipline, and a fail-closed 40 GB free-disk floor. Unified memory is not interchangeable with dedicated VRAM. PyTorch exposes current MPS allocation, driver allocation, and Metal's recommended working set separately; allocator watermarks are ratios of that working set, not promises that every installed byte is safe for GPU residency ([MPS memory APIs](https://docs.pytorch.org/docs/main/mps.html), [MPS allocator variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)).

**[M]** ViT-L, ViT-H, and ViT-g all encode locally and serially:

| Frozen encoder | Shared clips | Seconds/clip | Peak RSS | 10,000-clip serial projection |
|---|---:|---:|---:|---:|
| ViT-L | 8 | 22.335 | 2.589 GB | 62.0 h / 2.58 d |
| ViT-H | 8 | 34.728 | 4.000 GB | 96.5 h / 4.02 d |
| ViT-g | 8 | 187.333 | 6.616 GB | 520.4 h / 21.7 d |

The first three columns come from the citable cache run receipts. The last is **[C]** `10,000 * measured seconds/clip / 3,600`. The result is a serious throughput cost, not a resident-memory failure. A learned-plus-random cache at all three scales would take about 1,358 serial hours at those rates if random initialization had identical throughput. That extrapolation is a planning upper bound, not evidence for buying a machine.

**[M]** The eight-referent scale atlas scores ViT-L/H/g identically at the programmatic factor-probe ceiling while ViT-g costs about 8.4 times ViT-L's encode time and 2.6 times its RSS. This is evidence against assuming a scale benefit. It is not a scientific null: the atlas has only eight programmatic referents, no byte-identical cross-resolution stimuli, no matched random-architecture cache, and no seed distribution.

**[M]** `proof/CUSTOM_SUBSTRATE_CALIBRATION.json` demonstrates a teacher-independent CM7 mechanics path with 1,646,080 parameters, 256 dense spatiotemporal tokens, four matched objectives, exact frozen initialization, checkpoints, and finite MPS execution. Its one seed and three updates per arm are calibration only. The historical `cm7_local180` and `cm7_local180_citable` directories are not promotable: one was aborted for source drift, and one stopped after a transient Metal command-buffer recovery. Exact CPU and isolated-MPS replay were finite, so that event is a recovery/software issue, not a memory boundary. The source-snapshotted v3 campaign was still in progress at audit time and is excluded from the durable conclusion.

**[M]** A newer `proof/LOCAL_ACTION_ENVIRONMENT.json` verifies three deterministic programmatic seeds, exact replay, action-blind/shuffled controls, and local E5/F6/F15/CM10 mechanics in 1.79 seconds at about 263 MB peak RSS. Its own scope forbids scientific promotion. E10 still needs population search, environment generation, transfer, and a non-plateau horizon. CM10 still needs rendered observations, a citable substrate cache, the exact frozen V-JEPA 2-AC control, and the matched-compute unrolled-depth control. This receipt demotes the environment adapter from a hardware concern; it does not establish embodiment, open-endedness, or a real-time requirement.

**[M]** The project-exhaustion ledger accounts for all 175 non-F rows exactly once and reports zero measured hardware blockers. Its generation-time `self_verification.verified` must not be read as a live guarantee; current-source auditing finds stale embedded hashes described below.

### Natural intake and F-chain limits

**[M]** The bounded SANPO plan targets 10 session-disjoint sessions, 80 frames, and roughly 289 MB. SANPO provides stereo video, depth, IMU, attributes, real/synthetic splits, and processed privacy-sensitive content under a linked CC BY 4.0 release; it does not list native audio ([official SANPO page](https://google-research-datasets.github.io/sanpo_dataset/), [CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/)). Attribution, change indication, privacy, publicity, and trademark review remain distinct obligations.

The dry-run receipt predates the migrated plan identity. The existing execute receipt is a previous fail-closed plan-conflict result, not the outcome of the concurrently running corrected intake. Even a successful 80-frame smoke intake would establish natural input mechanics only; it would not satisfy F8/F16 trusted-provenance or scientific-promotion gates.

F1-F20 are local synthetic/P0 mechanics. F8 and F16 are first blocked by trusted natural trajectories and provenance. Their configured resident cap is 6 GB and their parameter cap is 50M; neither has a measured local failure. F10's positive gate fails because its verifier is non-independent. F7, F11, F14, F17, F19, and F20 support local nulls. F13, F18, and F20 must still be redesigned: F13 estimates rather than measures energy, F18 is supervised arithmetic rather than causal transport, and F20's avoided-compute ratio is hypothetical rather than measured.

### Stale-artifact corrections

Generated artifacts were treated as stale whenever an embedded hash disagreed with its current source.

| Artifact | Live audit | Consequence |
|---|---|---|
| `proof/PROJECT_EXPERIMENT_EXHAUSTION.json` | Current embedded-hash audit finds changed experiment/config, preflight, generator, and registry paths; repeated references are deduplicated in the matrix | Preserve generation-time context only; authoritative row evidence points to currently hashed direct receipts instead |
| `proof/FRONTIER_LOCALIZATION.json` | Three references retain the old scale-atlas hash | Its fail-closed conclusion survives; its embedded atlas identity does not |
| ViT-H/g standalone CPU-forward receipts | Their encoder config hashes predate random-init/revision fields | The self-contained forward and cache receipts prove local execution; the old config binding is stale |
| `proof/FORM_SUBSTRATE/SCORECARD.json` | F8/F16 still carry old GPU/Studio labels | Current registry, configs, classes, campaign, and refreshed contract audit govern: environment/data, not hardware |
| `proof/ARTIFACT_INDEX/form_substrate.json` | 26 of 66 hashes differ; bundle says `all_ok=false`, durable 0, missing 2 | Do not cite the bundle as a current durable chain |
| Old random-init ViT-L artifact | Quarantined because no `cache_manifest.json` existed | Superseded 2026-07-10: a citable seed-0 matched random-architecture ViT-L cache now exists (`data/cache/vjepa2_vitl_local8_random_s0`, backend `vjepa_hf_random_init`, manifest and state-dict hash recorded) and the refreshed scale atlas consumes it |

The exact current hashes and audit pointers are emitted under `embedded_hash_audits` in the requirements JSON.

### Upstream correction: V-JEPA 2.1 is available

The registry and local `configs/encoder/vjepa21_vit{b,l}.yaml` still describe placeholder or unpublished weights. That is now false. Meta's official repository records a V-JEPA 2.1 release on 2026-03-16 with four dense encoders at 384 px ([official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2), [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482)):

| torch.hub entrypoint | Parameters | Direct checkpoint |
|---|---:|---|
| `vjepa2_1_vit_base_384` | 80M | `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt` |
| `vjepa2_1_vit_large_384` | 300M | `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` |
| `vjepa2_1_vit_giant_384` | 1B | `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt` |
| `vjepa2_1_vit_gigantic_384` | 2B | `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt` |

The code and weights are majority-MIT with Apache-2.0 portions, and the download path needs no manual terms acceptance. Two access cautions: the official Hugging Face release is still an open upstream request ([vjepa2#137](https://github.com/facebookresearch/vjepa2/issues/137), [transformers#45496](https://github.com/huggingface/transformers/issues/45496)), so the repo's `vjepa_hf` backend cannot load 2.1 first-party yet; and the community conversion `Dev-Jahn/vjepa2.1-vitl-fpc64-384` is not first-party and may be cited only after a receipted feature-parity check against the torch.hub checkpoint. The B and L checkpoints are distillations from ViT-G (the `dist_vitG` suffix), which any scale-comparison design must declare: 2.1 B/L are not independent scale points.

Therefore E6 and the dense part of DR14 are category 2, not category 7. **[M]** Measured 2026-07-10: the full 1.664 GB ViT-B checkpoint was acquired against the pinned ETag/version (sha256 `848a77c3...`, `data/models/vjepa21/`), strict `ema_encoder` load passed (`proof/VJEPA21_VITB_LOAD.json`), and finite dense forwards passed at 8 frames ([1, 2304, 768], 0.88 s forward, 0.83 GB peak child RSS, `proof/VJEPA21_VITB_FORWARD.json`) and native 64 frames ([1, 18432, 768], 25.2 s forward, 1.33 GB peak child RSS, `proof/VJEPA21_VITB_FORWARD_64F.json`), all CPU. The 2.1 dense lane is measurably local at B scale; the encoder config's `available` flip and experiment routing stay with the E6 loader wiring, which remains the category-2 blocker. None of this is a reason to rent hardware.

## Complete current-registry blocker reclassification

All current rows not named below are category 1. This defines all 195 rows without hiding a historical tier behind a summary count.

### Category 2: eleven local implementation/control rows

| ID | First blocker | Why not hardware |
|---|---|---|
| `e6_relational` | Integrate and verify an official dense V-JEPA 2.1 checkpoint | Checkpoint is now published; L/H/g already execute serially locally |
| `mop_dr5_cross_substrate_consistency` | Citable same-architecture random controls and compatible-task grid | Independent caches and seeds serialize |
| `mop_dr14_corruption` | Dense-token cache integration for the dropped-channel arm | Released checkpoint plus streaming cache; no measured resident failure |
| `mop_at1_nuisance_grid` | Complete citable encoder × random-init columns | A control-construction problem |
| `mop_cm6_distilled_density` | Trainable student plus same-size non-distilled/random controls | Frozen teacher can be cached; 1.65M-scale students are local |
| `e10_openended` | Integrate a bounded local persistent environment and fixed horizon | MiniGrid/Crafter-class mechanics do not require external hardware |
| `mop_mt4_reasoning_router` | Six compatible verified primitive outputs and an independent verifier | Local components and serial evaluation suffice |
| `mop_pr2_plasticity_substrates` | Missing citable random-init-ViT control | A locally producible control |
| `mop_at2_mode_substrate_dep` | Verified winning-mode input and complete real/random cache condition | Cache/control integration, not memory |
| `mop_cm11_developmental_plasticity` | Calibrated curriculum and independent signature recomputation | Local implementation and verification |
| `mop_cm12_mop_substrate_capstone` | Compatible experts, shared battery, and open-model control | Input/control assembly before any scale question |

### Category 3: seventeen data/cache/rights rows

| IDs | First blocker |
|---|---|
| `mop_dr1`, `mop_dr2`, `mop_dr3`, `mop_dr4`, `mop_dr7`, `mop_dr15` | Rights-cleared natural/causal/compatible modality streams and citable caches |
| `mop_al2`, `mop_al3` | Meaningful shared content and native aligned audiovisual data |
| `mop_cm1`, `mop_cm2`, `mop_cm3`, `mop_cm4`, `mop_cm8`, `mop_cm9`, `mop_cm10` | Natural/action-rendered cache dependency chain, dense tokens, binding annotations, or the exact frozen action-conditioned control |
| `f8`, `f16` | Trusted natural trajectories and provenance |

The table uses shortened prefixes for readability; exact IDs and row-specific reasons are in the JSON matrix.

### Category 6: zero current rows

The currently registered E5 variant is implemented and the three-seed `proof/LOCAL_ACTION_ENVIRONMENT.json` now verifies bounded deterministic action mechanics locally, so E5 remains category 1. E10 first needs population search, environment generation, sustained horizons, and transfer implementation, so it is category 2. CM10's adapter mechanics now run, but its registered claim still needs rendered action-conditioned observations, a citable substrate cache, and the exact frozen V-JEPA 2-AC control, so it is category 3. Later real sensors, participants, robots, or non-replayable environments are separate category-6 promotion stages; no accelerator supplies those intervention surfaces.

### Categories 4, 5, 7, 8, and 9: zero current rows

Some local executions are slow, some future variants will need factorization, and a high-memory GPU would probably reduce wall time. None of those claims currently displaces category 1-3 or 6 as the first blocker, and no cross-machine measurement earns category 8.

## F21-F66 reclassification

Every proposed expansion begins below extended-compute status. Category 2 means the smallest valid simulation/mechanics test comes first; it does not pre-authorize a later physical claim.

| IDs | Category | First honest blocker |
|---|---:|---|
| F21-F64 | 2 | Their first not-yet-completed stage is a controlled local/simulated implementation with family-specific matched controls |
| F65-F66 | 6 | Multiple fabricated specimens and a real digital-to-material portability endpoint are intrinsic to the named claim |

This is complete coverage of 46 rows. `classified_stage` is explicit: the matrix classifies the first not-yet-completed rung in the expansion plan. F21-F64 therefore begin at category 2 even when their later natural-data, live-environment, or bench promotion stage becomes category 3 or 6. F61-F64 specifically begin with the plan's synthetic-device/material simulation stage; later specimen tests are carried by W10. Exact titles, family-specific blockers, and promotion blockers are preserved in the JSON. Participant- or specimen-level conclusions stay category 6 regardless of GPU supply.

## Workstream and pillar coverage

The machine matrix also covers every explicit expansion workstream and the experimental dossier pillars, closing a gap that a registry-only audit would miss.

| Workstream | Category | First blocker |
|---|---:|---|
| W0 Real-evidence completion | 3 | Rights-cleared real and natively aligned data |
| W1 Temporal referents and event identity | 2 | Controlled temporal mechanics |
| W2 Active multimodal perception | 6 | Acquisition-capable environment or sensor interface |
| W3 Boundary, agency, and body model | 6 | Causal action/body/tool intervention environment |
| W4 Memory lifecycle and continuity | 2 | Local lifecycle implementation and controls |
| W5 Multiscale plasticity and morphogenesis | 2 | Local growth/plasticity implementation and capacity matching |
| W6 Workspace and operational self-model | 2 | Broadcast, telemetry, lesion, and report-grounding hooks |
| W7 Social reference and cultural accumulation | 2 | Simulated partner populations first |
| W8 Open-ended developmental ecology | 2 | Bounded local ecology and fixed horizon |
| W9 Material substrate simulation | 2 | Digital simulator and conventional controls |
| W10 Bench material computing | 6 | Bench devices and multiple specimens |
| W11 Ethics, welfare uncertainty, and containment | 2 | Governance, audit, and containment implementation |

Dossier pillars 7-25 are represented as 19 source-derived rows. Vision, audio, and multisensory binding are category 3 because their next decisive stage needs suitable natural/aligned inputs. The other pillars begin at category 2 operationalization or simulation. Bench material work remains separated under W10/category 6; theory-neutral operational probes do not acquire a larger-compute status merely because the surrounding philosophical question is difficult.

## Candidate-frontier falsification

| Frontier | Smallest meaningful attack | Primary category | What could eventually earn a larger rung |
|---|---|---:|---|
| Natural-video objective tournament | Session-disjoint smoke set, one compact substrate, paired objectives/seeds | 3 | A powered surviving effect plus measured serial wall bottleneck |
| Teacher-free scaling | 1.65M baseline and log-spaced 0.5×/1×/2×/4× models | 2 | First implement and measure the sweep; only then can 4/5 or a larger rung be earned |
| Dense high-resolution long context | Masking, tubelets, local windows, recurrence/SSM, cached memory | 2 | First implement all attacks; category 5 requires a measured local factorization need |
| Learned versus random scale controls | Same bytes/referents, architecture, seed, precision, resolution | 2 | Implement citable controls before measuring serial slowness or a next-rung speedup |
| Multi-seed objective/ablation matrix | Paired sequential seeds with anytime-valid or fixed design | 2 | Name endpoint, SESOI, variance, dependence, and harness before calling the matrix slow |
| Million-event continual learning | Stream input, disk replay, bounded state, resumable checkpoints | 2 | Implement and profile the harness; a non-replayable live deadline could later change the rung |
| Action-conditioned world model | MiniGrid-scale adapter, action shuffle, blind model, tiny latent model | 2 | Measured p95 latency misses an environment deadline that cannot be slowed |
| Active perception | Simulated camera/sensor choices and sensor-cost curve | 2 | Real sensor/actuator validation is category 6, not a compute rung |
| Native multimodal binding | Compact bottleneck fusion on aligned A/V; telemetry as a local modality | 3 | Full modality × objective replication only after compatible data exist |
| Natural objects/events/causal state | MOVi/CLEVRER mechanics then natural multi-object sessions and interventions | 3 | Compute cannot resolve observational non-identifiability |
| Population/open-ended/social | Small QD/multi-agent population with fixed horizon | 2 | Measured throughput benefit; human/physical partners remain category 6 |
| Workspace/operational self-model | Logged telemetry, calibrated failure prediction, causal ablations | 2 | Ensembles serialize; a live deadline must be measured |
| Small-substrate architecture search | Query/proxy screen, random search, exact shortlist retraining | 2 | Category 8 only after proxy rank validity and campaign timing |
| Digital material simulation | Small simulator, drift/damage controls, digital reservoir baseline | 2 | Device claims require specimens, not more simulation compute |
| Robustness sweeps | Minimal corruption × severity screen, adaptive expansion | 2 | Implement the adaptive screen; category 4/8 only after measured expansion |
| Full-system density accounting | Ingest/cache/train/eval/retry/storage accounting | 2 | Instrumented wall-power work is category 6 until a meter exists |
| Physical/sensor/participant validation | Named external unit and preregistered protocol | 6 | No compute purchase removes this blocker |

### Why efficient temporal methods are compulsory controls

Published large video runs show that scale can work; they do not prove this program needs that scale. Efficient alternatives are scientifically relevant controls:

- VideoMAE uses extreme masking to reduce redundant video-token work ([paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html)); VideoMAE V2 adds dual masking ([paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)).
- MeMViT adds long-term memory rather than repeating full global context ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.html)).
- Transformer-XL establishes segment recurrence ([paper](https://aclanthology.org/P19-1285/)); Mamba-2 provides a state-space duality route to efficient sequence modeling ([paper](https://proceedings.mlr.press/v235/dao24a.html)).
- FlashAttention demonstrates that naive attention residency can be an I/O/algorithm issue rather than an irreducible hardware boundary ([paper](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)).
- Token Merging is an explicit token-reduction control ([paper](https://openreview.net/pdf?id=JroZRaRw7Eu)).

If a global dense model beats these controls, the result must separate the effect of context fidelity from added parameters, data, precision, and optimizer changes.

### Action, causal, and social frontiers

MiniGrid is a lightweight Apache-2.0 Gymnasium environment suitable for the first local action tests ([official repository](https://github.com/Farama-Foundation/Minigrid)). Procgen's 64×64 environments were designed for procedural generalization and do not themselves require a GPU ([paper](https://cdn.openai.com/procgen.pdf), [official repository](https://github.com/openai/procgen)). TD-MPC2 supplies a compact decoder-free latent world-model control ([paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)). These sources support local feasibility, not guaranteed scientific success.

For object-centric mechanics, Slot Attention is a compact baseline ([paper](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)). Causal representation results make the deeper point: identifiability requires assumptions, interventions, or side information; observational natural video plus more compute cannot manufacture it ([interventional CRL](https://proceedings.mlr.press/v202/ahuja23a.html), [CITRIS](https://proceedings.mlr.press/v162/lippe22a.html), [general identifiability analysis](https://proceedings.mlr.press/v238/varici24a.html)).

MAP-Elites, QDax, and POET motivate population and environment-diversity studies ([MAP-Elites](https://arxiv.org/abs/1504.04909), [QDax](https://arxiv.org/abs/2308.03665), [POET](https://arxiv.org/abs/1901.01753)). Their populations are mostly a throughput multiplier. Melting Pot provides an Apache-2.0 multi-agent substrate for partner generalization ([paper](https://proceedings.mlr.press/v139/leibo21a/leibo21a.pdf), [repository](https://github.com/google-deepmind/meltingpot)); human-partner evidence remains external infrastructure.

### Material and neuromorphic frontier

Growing neural cellular automata provide a compact local morphogenic baseline ([article/code](https://distill.pub/2020/growing-ca/)). Lava permits CPU-side neuromorphic simulation ([official repository](https://github.com/lava-nc/lava)). NeuroBench explicitly separates hardware-independent algorithm evaluation from hardware-dependent system evaluation ([paper](https://www.nature.com/articles/s41467-025-56739-4)). Deep physical neural-network work shows why this boundary matters: even an accurate digital model can miss device noise that is present in the physical forward path ([paper](https://www.nature.com/articles/s41586-021-04223-6)). Therefore simulation is category 2; energy, drift, damage, repair, and specimen-transfer claims are category 6.

## Local feasibility attack

Every candidate must traverse this table before any hardware promotion.

| Attack | Resource removed | Scientific equivalence test | Stop condition |
|---|---|---|---|
| Stream inputs | Dataset RAM residency | Deterministic sharding, split integrity, no duplication, declared shuffle | Continue if I/O rather than memory becomes limiting |
| Cache frozen teachers | Teacher compute and simultaneous residency | Key checkpoint, revision, preprocessing, view/crop, RNG, precision, output shape | Reject cache if stochastic-view equivalence fails |
| Batch 1 / microbatch | Activation residency | Preserve loss normalization, optimizer schedule, clipping, BatchNorm behavior | Stop shrinking at stable numerical parity |
| Gradient accumulation | Effective batch without resident batch | Compare update and RNG order to reference | Do not call a larger batch new science |
| Activation checkpointing | Saved activations | Control RNG/device movement; PyTorch documents determinism caveats ([docs](https://docs.pytorch.org/docs/stable/checkpoint)) | Stop when ≥20% working-set headroom |
| AMP BF16/FP16 | Tensor memory/bandwidth | Objective × precision interaction, over/underflow, held-out geometry ([docs](https://docs.pytorch.org/docs/stable/amp.html)) | Use exact precision if parity fails |
| INT8/INT4 frozen teacher | Frozen-weight residency | Exact vs quantized feature geometry on held-out referents | Training-state savings are not inferred |
| Tubelets/masking/windowing | Token and quadratic attention load | Exact global model as a small control; matched active FLOPs | Promote only if approximation changes decisive result |
| Recurrence/SSM/external memory | Long history without replaying all tokens | Match history access, state budget, update rules | Kill dense scaling if compact control wins per joule |
| Sequential seeds | Parallel residency | Common seeds/referents and unchanged stopping rule | Parallel hardware changes wall time only |
| Sequential power | Unnecessary fixed replication | Preregister confidence sequence or alpha spending | Stop for futility or predeclared success |
| Random/low-rank probes | Expensive representation readout | Exact control on a subset; rank and parameter matching | Reject approximation if ranking reverses |
| Compact derived data | Decode/storage cost | Hash raw source, transform code, parameters, and split; preserve provenance | Never discard reconstructive provenance silently |

PyTorch DDP replicates a model per process; aggregate GPU memory is not automatically pooled. FSDP can shard parameters, gradients, and optimizer state, but adds collectives and transient unsharded states ([FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)). Multi-GPU memory is not earned until DDP is rejected and FSDP/tensor-parallel behavior is measured.

## Ceiling calculations

### Video-token residency

For tubelet length `t`, spatial patch `p`, frames `F`, and resolution `H × W`:

```text
tokens = ceil(F/t) * ceil(H/p) * ceil(W/p)
```

At 64 frames, tubelet 2, 256², patch 16, this is 8,192 tokens. A ViT-L dense cache with width 1,024 uses:

```text
8,192 * 1,024 * 2 bytes = 16.78 MB/clip in fp16
10,000 clips = 167.8 GB before metadata/checksums
```

At 384², the same temporal setting has 18,432 tokens and a 1,024-wide fp16 cache is 37.75 MB/clip or 377.5 GB per 10,000 clips. These are **[C] storage calculations**, not resident-state requirements: shard them, pool them when scientifically valid, or compute on demand.

A naive materialized attention-score tensor is:

```text
batch * heads * tokens^2 * bytes/value
```

It is an upper-bound diagnostic because optimized scaled-dot-product attention may avoid materializing it. A category-9 claim must measure the actual backend peak, not cite this formula as if it were an allocation receipt.

### Trainable state

An Adam-like pre-measurement budget is approximately 16-18 bytes per trainable parameter for weights, master weights, gradients, and two FP32 moments before activations/workspaces:

| Parameters | Trainable state estimate |
|---:|---:|
| 1.646M CM7 | 26.3-29.6 MB |
| 100M | 1.6-1.8 GB |
| 1B | 16-18 GB |

The CM7 state is nowhere near the host ceiling. A 1B trainable model would be a poor local fit before activations, but no current hypothesis requires 1B parameters. “A model we chose would not fit” is not “the scientific question requires that model.”

### Campaign time, storage, and energy

```text
T_serial = units * seeds * updates * measured_step_time
           * (1 + retry_fraction + validation_fraction)
           + preprocessing_time

T_parallel = T_serial / (workers * measured_parallel_efficiency)

cloud_cost = instance_rate * reserved_hours
             + storage + checkpoint/API I/O + egress + retry overhead

IT_kWh = measured_wall_kW * hours
facility_kWh = IT_kWh * PUE
```

TDP is not wall power. MLPerf's power methodology measures the system boundary rather than substituting component TDP ([measurement documentation](https://docs.mlcommons.org/inference/power/)); Green Algorithms supplies a broader computation-impact framework ([paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202100707)). Until this host has a plug-level meter, local joules are an estimate and F13 cannot claim measured energy.

## Statistical power and independent units

The confirmatory default is a paired, two-sided design with common seeds and referents, 80% power, and Holm correction across five named comparisons. A conservative per-comparison design point is `alpha=.01`.

Exact noncentral paired-t calculations:

| Paired standardized effect `d_z` | n at α=.05 | n at α=.01 |
|---:|---:|---:|
| 0.3 | 90 | 134 |
| 0.5 | 34 | 51 |
| 0.6 | 24 | 36 |
| 0.8 | 15 | 22 |
| 1.0 | 10 | 16 |

Five seeds are a variance-estimation and debugging pilot, not a universal powered experiment. A default confirmatory target of 22 paired units is justified only for `d_z=.8`; a moderate `d_z=.5` needs 51. The unit is the training seed paired over the highest independent data unit: session, referent, environment, partner, or specimen. Frames from one session, crops from one clip, and replicas of one referent are nested observations, not independent `n`.

Use cluster/bootstrap intervals at that highest unit, paired differences for common-random-number designs, Holm for named confirmatory families, and Benjamini-Hochberg only for explicitly exploratory families. Repeated peeking requires a preregistered anytime-valid confidence sequence or alpha-spending rule. Seed uncertainty is a known source of misleading algorithm comparisons ([How Many Random Seeds?](https://arxiv.org/abs/1806.08295)); robust aggregate/interval reporting is preferable to a single mean ([RLiable](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)).

## Capability-density response surface

The added-compute program should fit a response surface rather than a single parameter scaling curve:

```text
Y = f(P, T, C, R, E, I, U, M, O, J, S)
```

where:

- `P`: trainable parameters;
- `T`: tokens observed;
- `C`: temporal context or recurrent-state horizon;
- `R`: independent referents/session diversity;
- `E`: independent environments/partners;
- `I`: intervention diversity;
- `U`: lifetime updates;
- `M`: replay/external-memory budget;
- `O`: objective mixture;
- `J`: measured or explicitly estimated joules;
- `S`: storage, including caches/checkpoints/retries.

Start with a fractional-factorial screen at matched active FLOPs and matched storage, then fit main effects and selected interactions. Report elasticities such as `∂log(capability)/∂log(P)` and the corresponding referent-, temporal-, intervention-, and memory-budget elasticities.

Falsifiable hypotheses:

1. **[H] Referent diversity beats width:** at two active-FLOP budgets, added independent sessions improve held-out compositional/causal transfer per joule more than doubling parameters.
2. **[H] Temporal depth beats token density:** recurrence or bounded external memory improves long-horizon prediction per byte more than dense global attention.
3. **[H] Architecture diversity beats one large model:** a matched-budget recurrent/sparse/cellular/reservoir shortlist yields a better capability-density frontier than width/depth scaling.
4. **[H] Intervention diversity beats passive volume:** novel interventions improve causal transfer more than equal-byte passive footage.
5. **[H] Developmental order matters:** ordered curricula improve future plasticity at equal examples, updates, and final capacity.

Demote parameter scaling if its upper confidence bound on improvement per joule is below the lower bound for added referent diversity, temporal memory, interventions, or architecture diversity at two budgets. Promote it only when the parameter effect survives matched data, active FLOPs, optimizer, context, and control capacity.

## Data, rights, environment, and physical separation

| Source/type | Access/right signal | Exact caution | Primary blocker |
|---|---|---|---:|
| SANPO | Official CC BY 4.0-linked release, video/depth/IMU | Attribution/change log; privacy/publicity/trademark review; no native audio listed | 3 |
| AViD | Official repository describes CC source videos and MIT redistribution | Audit sampled audio, source/license manifests, and residual personality rights ([repository](https://github.com/piergiaj/AViD)) | 3 |
| YFCC100M | Per-item Creative Commons metadata ([paper](https://arxiv.org/abs/1503.01817)) | Filter per-item license; preserve attribution/deletion state | 3 |
| V3C | Creative Commons Vimeo corpus ([paper](https://arxiv.org/abs/1810.04401)) | TRECVID access requires an agreement; do not accept it implicitly | 3 |
| Wikimedia Commons | Per-file free-license/public-domain descriptions ([licensing policy](https://commons.wikimedia.org/wiki/Commons:Licensing)) | Review each file and non-copyright rights | 3 |
| MiniGrid/Procgen/Melting Pot | Open software environments | Environment adapter and reproducibility still need implementation | 2, then 6 for real-world claims |
| Sensors/robots/participants | Protocol-specific | Consent, safety, recruitment, calibration, and independent-unit definition | 6 |
| Physical material devices | Specimen-specific | Simulation cannot establish device noise, energy, drift, damage, or transfer | 6 |

AudioSet annotation licensing does not transfer underlying YouTube media rights ([official page](https://research.google.com/audioset/)); URL-based Kinetics/VGGSound/ACAV collections should likewise not be called raw-media-rights-clean. Data availability, terms, and provenance are not compute properties.

## The extended-compute ladder

This is an escalation ladder, not a shopping list. Costs are checked-date examples and must be refreshed before any decision.

| Rung | Envelope | Memory/storage | Wall/power/cost | What it could unlock | What it cannot unlock | Stop rule |
|---|---|---|---|---|---|---|
| L0 bounded local | Current M3 Pro, MPS/CPU, one heavy process | 19.3 GB unified; runtime safe working set measured, ≥40 GB disk free | ≤180 min/shard; wall energy unmeasured | All 167 category-1 rows, local mechanics/integrations | Rights, data, environments, specimens | Stop shard before wall/disk guard; require ≥20% measured memory headroom |
| L1 resumable local | Same host, sequential overnight/multi-day shards | Disk-backed input/replay/cache; atomic checkpoints | Days are allowed, each shard ≤180 min; no capex | Powered seeds, random controls, long streams, cache generation | Real-time interaction or missing external inputs | Stop on null/futility, I/O bound, unstable recovery, or disk floor |
| L2 high-memory Apple | M4 Max up to 128 GB or live-listed M3 Ultra up to 256 GB unified; 546/819 GB/s ([current specs](https://support.apple.com/en-us/122211)) | Up to 16 TB internal; still shared memory, not VRAM | Canadian base prices currently start near CAD 2,699/5,499; final quote dynamic ([store](https://www.apple.com/ca/shop/buy-mac/mac-studio)). Apple wall tests include 145 W base M4 Max and 270 W historical maxed M3 Ultra ([power](https://support.apple.com/en-ca/102027)) | A measured single-state MPS workload between L0 safe memory and L2 safe working set | CUDA-only kernels, data, environments | Do not buy without repeated local OOM/pressure and an MPS parity pilot |
| L3 one 96 GB CUDA | Google G4 one RTX PRO 6000, 96 GB GPU, 180 GiB host, local SSD | Dedicated 96 GB GDDR7; up to 1.5 TiB local SSD ([G4 specs](https://docs.cloud.google.com/compute/docs/gpus)) | Page showed about USD 4.50/h on-demand, 2.25 flex, 1.80 spot when checked ([dynamic pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)); GPU up to 600 W, not wall power ([NVIDIA](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)) | Cheapest bounded CUDA/runtime/throughput validation; a four-hour pilot ≈USD 18 compute | Aggregate multi-GPU memory, missing science inputs | Stop if parity fails, decoder/I/O dominates, or predicted bottleneck does not move |
| L4 small 2-4 GPU | 2-4 G4 GPUs with PCIe P2P | 192-384 GB aggregate, not automatically pooled | Roughly 2×-4× L3 rates; measure efficiency | FSDP/tensor-parallel pilot or parallel powered campaign | Simultaneous state unless sharding proves valid | Require FSDP memory trace; stop below preregistered scaling efficiency |
| L5 eight-GPU/distributed | AWS P5e/P5en 8×H200, 1,128 GB HBM3e total, NVSwitch, 2 TiB host, 30.7 TB NVMe ([AWS P5](https://aws.amazon.com/ec2/instance-types/p5/)); DGX B200 8×180 GB, 1.44 TB GPU memory | High-bandwidth sharded state | AWS's dated update lists USD 5.97-6.865/H200-hour and a one-day minimum, about USD 1,146-1,318/node-day ([price](https://aws.amazon.com/ec2/capacityblocks/pricing/), [duration](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-how.html)). DGX B200 maximum system power is about 14.3 kW, or 343 kWh/day before PUE ([guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)) | Only an irreducible >single-GPU state or synchronized population proven on L4 | Rights, participants, physical specimens; independent seeds do not need NVSwitch | No L5 until L4 fails for measured per-device memory/latency with parity intact |
| L6 external infrastructure | Sensors, actuators, robots, participants, physical specimens, calibrated power meter | Protocol-specific | Cost/safety/ethics-specific, not estimated here | Real intervention, embodiment, device energy/drift/damage evidence | It is not a compute substitute | Acquire only for a named protocol with governance and an independent-unit power plan |

NVIDIA's H200 has 141 GB HBM3e, 4.8 TB/s bandwidth, and 600-700 W accelerator power depending on form factor ([official H200 page](https://www.nvidia.com/en-au/data-center/h200/)). This describes a possible single-device memory rung, not current need or guaranteed single-device cloud availability.

Apple's 2025 launch release mentioned a 512 GB M3 Ultra option, while the live 2026 technical page lists 96 or 256 GB. Use 256 GB as the conservative currently documented ceiling unless a fresh written quote proves otherwise ([launch release](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/), [live specifications](https://support.apple.com/en-us/122211)).

## What would make category 8 or 9 true?

Category 8 requires a measured local calibration, a named powered run count, a named elapsed-time target, and a bounded next-rung benchmark showing useful speed/cost improvement at parity. It still means the experiment can be done locally.

Category 9 requires all of:

1. A preregistered fidelity, statistical-power, or real-time target.
2. End-to-end local profiling, including ingest, decode, teacher, training, evaluation, checkpoint, and retry overhead.
3. Scientifically valid batch-1, streaming, caching, AMP, checkpointing, quantization, windowing/recurrence, sequential-seed, and resumability attacks where applicable.
4. Three repeated peak-memory or p95-latency failures against the runtime safe envelope.
5. A written proof that serialization/factorization changes the estimand.
6. A calculation mapping the measured excess to the smallest sufficient rung.
7. A bounded next-rung pilot with numerical, data-order, and metric parity.
8. A null rule, a next-rung promotion rule, and a permanent kill rule.

The three genuine enablement tests are:

- a non-factorizable resident state exceeds the safe working set;
- closed-loop p95 latency misses a scientifically necessary real-time deadline;
- synchronized state or interaction cannot be serialized/replayed without changing the estimand.

More seeds, frames, referents, or independent jobs alone are throughput, not enablement.

## The Studio-necessity facet: what could actually consume the studio-m1ultra envelope

Added 2026-07-10 as the goal-loop fold-in. Everything above answers "is anything category 8 or 9 today" (no). This facet answers the inverse question the program keeps asking: if a host satisfying the `studio-m1ultra` envelope materializes (128 GB unified memory, 8 TB disk with an 800 GB reserve and 250 GB free floor, tiers C+E, week-scale wall clock, 16-worker CPU default; `src/mop/studio/profiles.py`, `procurement_status: unverified-procurement-scenario`), which workloads would honestly consume it, in what order, and what receipt must exist before each may claim Studio time. Nothing here weakens the executive answer: every row below is category 1-6 today, and the envelope stays hypothetical until a strict doctor measures a real host against its floors.

### Corrected local measurement base

- **[M]** Wave-0 encode microbench (2026-07-03, `STUDIO_RUN_REPORT.md`): V-JEPA 2 ViT-L, 64 frames at 256 px, one CPU worker, 13.69 s/clip stable over 8 real clips. A later auto receipt measured 16.191 s/clip and the earlier citable cache builds measured 21-22.3 s/clip on the same path, so the honest planning band is 13.7-22.3 s/clip.
- **[M]** The same microbench ran MPS to completion at about 821 s/clip on one clip: roughly 60x slower than CPU, memory-pressured at 18 GB. The historical hard `Invalid buffer size` overflow recorded in `STUDIO_HANDOFF.md` did not reproduce on this path. The measured laptop MPS wall is paging pressure, not a proven per-buffer ceiling, so the old claim "more unified memory would not help MPS" is retired. Whether 128 GB makes MPS the encode winner is an open measurement that only the target host can settle (the pending Wave-0 microbench).
- **[M]** ViT-H 34.7 s/clip at 4.0 GB peak RSS; ViT-g 187.3 s/clip at 6.6 GB peak RSS, serial CPU, citable cache receipts.
- **[M]** Host: 19.3 GB unified memory, 460 GB disk whose free space measured between 60 and 97 GB this week against the fail-closed 40 GB floor, a 180-minute shard wall, one-heavy-process discipline, and the host is the operator's daily machine.

### The four measurable Studio quantities

Every honest Studio claim in this program reduces to one of four quantities, each measurable on both hosts:

| Quantity | Laptop measured | studio-m1ultra envelope | Who consumes it |
|---|---|---|---|
| Encode throughput | 13.7-22.3 s/clip serial CPU; MPS 821 s/clip at 18 GB | **[E]** 1-2 s/clip aggregate at 14-16 workers; MPS unknown until measured | S2 natural corpora, S3 control caches, S6 scale atlas |
| Resident working set | 19.3 GB installed; safe MPS working set below that | 128 GB installed (floor 120) | S5 perspective ecology, S6 ViT-G, any future P5 boundary |
| Durable artifact bytes | 460 GB total, 60-97 GB free, 40 GB floor | 8 TB total, 7.2 TB usable, 250 GB floor | S1 dense caches, hosted corpora |
| Dedicated calendar | 180-minute shards on a daily-driver machine | week-scale wall (`max_wall_min` = 7 days), always-on | S4 long streams, S8 seed retrofits, every powered campaign |

Parallel workers on the laptop are bounded by memory as well as discipline: at the measured 2.6-4.0 GB per ViT-L/H worker, 18 GB minus OS headroom supports at most 2-3 concurrent encoders, so the 16-worker projection is a genuine envelope difference, not a scheduling choice.

### Ranked dossier: candidate Studio-consuming workloads

Each row states the arithmetic from measured constants, the local attack that must run first, the named gate that would earn the rung, and the demotion result. Ranks follow the goal loop's standing order, not raw size.

**S2. Natural-corpus encode at scale (P2/P3 at scale, DR1 beyond the smoke pack, facet 14).**
**[C]** At the measured band, 1,000 clips cost 3.8-6.2 serial CPU hours (locally feasible in 2-3 shards); 10,000 cost 38-62 hours (a week-plus of daily shards); 100,000 cost 380-620 hours, which is 16-26 days of continuous single-worker compute and not schedulable on a daily-driver laptop. **[E]** The 16-worker projection puts 100k clips at 2.5-6 days. The scientific need for more than about 1,000 clips is not yet earned: session-level variance from the first natural pilot must set the confirmatory n, and rights intake precedes everything. Gate: measured session variance, a named corpus size from the power calculation, and a measured local calendar that exceeds a preregistered decision deadline. Demotion: if the pilot's endpoints sit at ceiling or the session effect is resolvable at 1k clips, the scale claim dies.

**S4. Long-stream continual learning at loss-inducing length (P6, PR9, facet 16).**
Streams replay from disk and checkpoints resume exactly, so laptop shards remain scientifically valid at any length; the measured blocker is dedicated calendar, an operational-availability boundary rather than a validity one. Gate: the local 100k-event calibration (P6 card) plus a certificate that the loss-inducing stream length exceeds what 180-minute shards can traverse before the preregistered decision date. The one honest exception: a preregistered estimand coupled to wall clock (a live, non-replayable stream) would convert this to enablement; no current card names one. Demotion: if 100k events resolve the horizon question, the million-event rung is never bought.

**S1. The dense-token cache lane (DR3 scratchpad, CM3, DR14 dense arm, ex9 dense arm, facet 8).**
**[C]** Dense V-JEPA tokens at 64 f/256 px are 8,192 x 1,024 x 2 bytes = 16.78 MB/clip fp16; 384 px multiplies by 2.25 (18,432 tokens, 37.7 MB/clip). So 10k clips = 168 GB and 100k = 1.68 TB at 256 px. **[M/C]** Against 60-97 GB free and the 40 GB floor, the honest local dense ceiling is roughly 1,200-3,400 clips; a 10k-clip dense cache is physically impossible on this disk regardless of scheduling. Local attack: pooled screening first (2 KB/clip), dense-on-demand recompute (trades disk for the measured s/clip and multiplies under multi-pass access), sharded rotation under the floor. Gate: a named dense-token experiment whose power analysis needs more than the local dense ceiling and whose access pattern defeats recompute (repeated random access across the whole corpus). Demotion: dense endpoints at ceiling/floor in the small-dense pilot kill the terabyte artifact permanently.

**S3. Learned-versus-random control caches beyond ViT-L (P3).**
**[C]** A 64-clip cache costs about 24 min (L), 37 min (H), 3.3 h (g) serial; a 22-session paired campaign across three scales and two arms lands near 60-90 serial hours, L1-feasible but calendar-hostile. Category 8 candidate at a named calendar target only after the L-scale delta leaves a scale-dependent uncertainty. Demotion: random ties learned at L on natural referents.

**S6. V-JEPA 2.1 ViT-G and the atlas top rung.**
**[P]** ViT-G is 2B parameters at 384 px, roughly 8 GB of fp32 weights. **[E]** Extrapolating measured ViT-g 187 s/clip, a ViT-G 384 px CPU forward plausibly costs 6-12 minutes/clip, and its peak RSS is the one integration where a local memory boundary is plausible but unmeasured. Probe order: ViT-B 2.1 one-clip forward locally (**[M]** passed 2026-07-10: 25.2 s/clip forward, 1.33 GB peak at native 64 f/384 px dense geometry), then ViT-L 2.1 only when an experiment names it, ViT-G only after an off-ceiling endpoint requires a fourth scale point. The current eight-referent atlas ties L/H/g at ceiling, so a fourth point is scientifically unearned today. Note the distillation caveat: 2.1 B/L are ViT-G distillates, not independent scale points.

**S5. The resident perspective ecology (facet 15).**
**[C]** Ten-plus encoders plus a 7B-class captioner are about 30 GB fp16 of weights alone, above the 18 GB host and comfortably inside 128 GB. Local attack: cached features on identical referents make simultaneity unnecessary for every currently registered alignment claim; the ecology evidence object is the `PerspectiveMatrix`, which is storage, not residency. Category 5-then-8. It becomes category 9 only if a closed-loop multi-perspective interaction with a latency deadline is preregistered; none is.

**S7. The MPS-at-128GB encode decision (pending Wave-0 microbench).**
Pure measurement, no science rides on it directly; it selects the encode path for S2/S1/S3 through the existing `encode_scheduler` contract. The laptop's 821 s/clip receipt is the honest prior; measure, never assume, in either direction.

**S8. B5 multi-seed re-encode and 30-seed retrofits (facet 9).**
**[C]** Pure throughput at the measured s/clip and run times; locally shardable without validity loss. Category 8 candidate at a calendar target. The standing order keeps it last: never refine an owned number while an unbuilt instrument blocks an axis.

### First measured P5 trace (2026-07-10)

`proof/P5_MEMORY_BOUNDARY_TRACE.json` (built by `scripts/p5_memory_boundary_probe.py`, 20 cells x 3 cold-process repeats, CPU, batch 1, CM7-class width 192, training step including backward and AdamW):

| Tokens (frames x res) | exact_math peak GB | sdpa peak GB | checkpointed peak GB | window_local peak GB |
|---:|---:|---:|---:|---:|
| 512 (16f 128px) | 0.32 | 0.30 | 0.30 | 0.30 |
| 2,048 (16f 256px) | 0.62 | 0.36 | 0.35 | 0.36 |
| 4,096 (32f 256px) | 1.50 | 0.43 | 0.42 | 0.43 |
| 8,192 (64f 256px) | 4.85 | 0.57 | 0.54 | 0.57 |

**[M]** The strongest plausible local memory boundary, dense 64-frame 256 px training context, is measured at 4.85 GB with naive materialized attention and 0.54-0.57 GB under any standard factorization, an 8.5x reduction, all far inside the 19.3 GB host. **[C]** Extrapolating the measured quadratic, exact-math attention first threatens this host near 16,384 tokens (roughly 15-18 GB, for example 128 f at 256 px or 64 f at 384 px), where SDPA remains under about 1 GB. Consequence: at CM7 width there is no dense-context memory rung on this host unless a preregistered estimand requires literally materialized exact global attention above 16k tokens, which no current card does. The P5 card's remaining work is scientific (does exactness change the decisive result), not residency.

### What even a measured Studio cannot unlock (unchanged)

Rights-clean natural media, aligned audio-video, interactive environments, participants, physical specimens, causal identifiability from observational data, the upstream Hugging Face release, and the frozen-encoder formation wall (Process C stays a doctrine decision, not a hardware one). Any Studio plan that claims these is mislabeled.

### The fold-in rule

The Form Substrate extended-compute lane (P1-P10) and the MoP goal-loop spine (DR1, PR9, dense pair plus atlas, B5; facets 12-17) now share one escalation grammar: every Studio claim must name its row here, its consumed quantity, and its receipt. The goal loop's standing order (DR1 first, PR9 second, dense plus atlas ride the conveyor, B5 last) is exactly S2 then S4 then S1/S3/S6 then S8, and the spine's fail-closed gates (source card, caption gate, certificate, dense pair gate, verdict ledgers) are the execution-time enforcement of this dossier's promotion gates.

## Unresolved questions and uncertainty register

1. The corrected SANPO execute intake was still running at the evidence snapshot; its final receipt may change the immediate data plan, not the hardware decision.
2. CM7 v3 was still running. Completed five-seed results may change objective ranking and variance estimates, not the already measured fit of its 1.65M-parameter mechanics.
3. Resolved 2026-07-10 at B scale: the official V-JEPA 2.1 ViT-B checkpoint is acquired, hash-receipted, strict-loaded, and forward-verified at 8 and 64 frames on CPU (25.2 s/clip forward, 1.33 GB peak at native dense geometry). E6/DR14 remain category 2 on experiment-side loader wiring; ViT-L/g/G stay unprobed until an experiment names them.
4. Partially resolved 2026-07-10: a citable seed-0 architecture-matched random ViT-L control now exists and the refreshed atlas consumes it. **[M]** At the eight-referent programmatic ceiling the random control ties learned on the hue factor (both 1.0) and sits at chance (0.5) on the motion factor while learned L/H/g score 1.0. Direction only: one control seed, four held-out referents, non-promotable by the receipt's own eligibility gate. H/g random controls, more seeds, and byte-identical stimulus hashes remain open.
5. Local wall energy is unmeasured. Any joule or carbon comparison is provisional until plug-level instrumentation defines the system boundary.
6. The paired-effect variance for decisive natural-video outcomes is unknown. Five seeds estimate it; confirmatory `n` must be recalculated from the pilot.
7. Cloud prices, inventory, and Apple configurations are unstable. Refresh official pages and obtain a quote only after the scientific gate passes.
8. MPS fallback and operator coverage can affect throughput. Log backend fallbacks rather than interpreting a slow operation as a memory requirement.
9. Natural-media copyright, privacy, publicity, and trademark status are per-source facts. A dataset-level license alone is insufficient.
10. Exact dense global context may eventually be scientifically necessary. That is a hypothesis to test against masking, recurrence, hierarchy, and bounded memory, not a default architecture.

## Bottom line

The repository has already falsified the easy version of the Studio story: ViT-H and ViT-g run locally, CM7 is tiny and trainable, the ledger has no measured hardware blocker, F8/F16 are data-gated, and the only apparent unpublished dense checkpoint is now officially released. The decisive next work is to repair evidence identity, finish small local campaigns, acquire rights-clean independent data, build matched controls, and measure response surfaces. Hardware escalation remains a gated experimental result, not a planning assumption.
