# Substrate v5 ready-stage architecture

Substrate v5 is an evidence-bound, local multimodal substrate. Its owned identity,
world state, memories, goals, uncertainty, body state, and model registry live in
persistent substrate state rather than in a model call. At this stage every model
is a deterministic local model-equivalent used for bounded scientific fixtures;
v5 does not contain or admit a downloaded foundation-model checkpoint.

```text
seeded local environments
        |
        v
typed sensor events -> tracking, binding, spatial state, active perception
        |                         |
        +-------------------------+
                    |
                    v
event-sourced explicit state <-> bounded latent state
                    |
                    v
model-neutral routing, support, verification, and replacement
                    |
                    v
sealed checkpoints, receipts, controls, and independent recomputation
```

## Selected cognitive kernel

The bounded construction benchmark implemented five candidates: extended v4,
event-sourced graph, recurrent latent, hybrid explicit-latent, and typed actors.
It selected `candidate_d_hybrid_explicit_latent`, an
`event_sourced_explicit_latent_substrate`. Selection used integrated mechanism,
auditability, throughput, and checkpoint-compactness utility; throughput alone
could not select a kernel.

The selected candidate passed 10 of 11 construction checks. It demonstrated
identity and unfinished-goal persistence, object permanence and occlusion state,
model replacement, checkpoint restore, multimodal coverage, explicit provenance,
latent transition, and explicit-latent synchronization. Its only false check was
typed actor messaging, which belongs to the separate actor candidate. These are
construction and moderate-pilot results, not principal or terminal cognition
claims.

The permanent-state implementation is an append-only, hash-chained event
projection. It owns:

- identity, monotonic internal time, active and latent context;
- sensors, bounded sensory buffers, objects, agents, places, events, and spatial
  and structural world state;
- goals, unfinished tasks, unresolved hypotheses, beliefs, and verified
  knowledge;
- episodic, semantic, and procedural memory;
- body and tool state, model contracts and relationships, competence and
  availability state; and
- bounded background, consolidation, and learning queues.

Checkpoints contain the event chain and its exact deterministic projection.
Restore rejects invalid seals, non-contiguous sequences, broken hashes,
non-monotonic time, identity disagreement, state-digest disagreement, or any
`activation` value other than `false`. Schema migrations between implemented
versions 1 and 2 retain a pre-migration checkpoint and support exact rollback.

## Sensorium and bodies

The typed sensorium admits eight local fixture modalities:

```text
text, image, video, motion, audio, speech, depth_3d, body_tool
```

Raw signal, preprocessed signal, perceptual proposal, tracked world, inferred
event, verified relation, structural belief, and knowledge are separate
representation layers. Events bind sensor identity, time, sequence, coordinate
frame, source digest, preprocessing identity, model identity, confidence,
uncertainty, provenance, and quality or missing-data flags. Hidden target or
oracle identifiers are refused.

Implemented mechanisms include checked coordinate transforms, object tracking
through occlusion and viewpoint change, event tracking, audiovisual timing,
cross-modal binding with conflict preservation, explicit 3D scene state, and an
expected-information active-perception policy.

Two deterministic bodies are implemented: a desktop/browser-style sandbox and a
seeded 3D simulator. Physics state is separate from rendered observations.
Actions return receipts and remain inside controlled local environments.

## Model fabric

The registry contains 13 independently callable deterministic Python
model-equivalents:

```text
language_interpreter       image_object_detector
video_event_segmenter      motion_estimator
audio_event_encoder        speech_grounder
depth_estimator            spatial_scene_mapper
body_dynamics_predictor    cross_modal_binder
evidence_verifier          contextual_router
plan_simulator
```

Each declares checkpoint identity, runtime, accepted and produced modalities,
schemas, confidence semantics, cost, latency, memory, provenance, limitations,
and allowed roles. Every contract includes `independent_performer`; support roles
do not remove independent callability. The fabric implements outcome-blind
routing and measured relationships such as draft/verify, simulate, translate,
and route.

These modules are hand-specified deterministic scientific fixtures with no
training data and a bounded-synthetic-operations limitation. They are not claims
of general model competence. External cached models remain inventory only until
license, hash, strict-load, resource, and parity gates pass.

## Evidence and campaign fabric

V5 writes canonical finite JSON beneath `configs/substrate/v5`,
`evidence/substrate/v5`, `runs/substrate/v5`, and
`artifacts/substrate/v5`. Named publications are atomic indexes over immutable,
content-addressed sealed objects. Writers refuse path escape and any non-false
activation field.

The frozen principal DAG has 5,760 units: 80 developmental histories, 18 focused
arms, 20 phases in four ordered shards, and 576,000 generated sensory events or
cognitive episodes. Workers compute independently; one publisher validates and
publishes receipts and checkpoints atomically. Principal, replication,
open-world review, independent verification, and final classification remain
pending.

No external corpus or checkpoint is scientifically admitted. Current admitted
inputs are the deterministic seeded desktop environment, deterministic seeded
3D environment, and frozen synthetic multimodal developmental generator.

## Safety and claim boundary

`activation` is exactly `false`. V5 permits deterministic local simulations,
sandboxed interfaces, local tools, controlled virtual bodies, authorized
read-only data, and explicitly permitted human interactions. It forbids
purchase, account, carrier, financing, checkout, order, credential, activation,
and uncontrolled real-world actions.

No v5 result claims consciousness, phenomenal experience, sentience, feeling,
suffering, desire, personhood, life, or moral status. “Model organism” is only an
engineering metaphor. Unqualified Nous is never an automatic classification.

Authoritative implementation and evidence:

- `src/substrate/v5state.py`, `src/substrate/v5sensorium.py`,
  `src/substrate/v5models.py`, `src/substrate/v5environment.py`
- `src/substrate/v5kernels.py`, `src/substrate/v5principal.py`,
  `src/substrate/v5verify.py`
- `evidence/substrate/v5/SUBSTRATE_V5_KERNEL_SELECTION.json`
- `evidence/substrate/v5/SUBSTRATE_V5_MODEL_REGISTRY.json`
- `evidence/substrate/v5/SUBSTRATE_V5_SENSORIUM_SCHEMA.json`
- `evidence/substrate/v5/SUBSTRATE_V5_PRINCIPAL_DAG.json`
