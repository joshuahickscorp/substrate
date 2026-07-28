# Substrate v5 terminal architecture

Substrate v5 is an evidence-bound local multimodal substrate. Its identity,
world state, memories, goals, uncertainty, body state, and model registry live
in persistent substrate state rather than in a model call. Every admitted model
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

The bounded benchmark executed five candidates: extended v4, event-sourced
graph, recurrent latent, hybrid explicit-latent, and typed actors. It selected
`candidate_d_hybrid_explicit_latent`, an
`event_sourced_explicit_latent_substrate`, using integrated mechanism,
auditability, throughput, and checkpoint-compactness utility. Throughput alone
could not select a kernel.

The selected candidate passed 10 of 11 candidate-specific construction checks.
It demonstrated identity and unfinished-goal persistence, object permanence and
occlusion state, model replacement, checkpoint restore, multimodal coverage,
explicit provenance, latent transition, and explicit-latent synchronization.
Its false typed-actor-messaging check belongs to the separate actor candidate
and was not a selection requirement.

Terminal H_M11 evidence independently confirmed the selected kernel’s
integrated advantage over the extended-v4 reference:

```text
principal mean effect: 0.2601
95% bootstrap CI: [0.2522, 0.2684]
SESOI: 0.05
result: pass
```

The effect also passed in independent-replication and generator-held-out
open-world splits.

## Permanent state

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
`activation` value other than `false`. Implemented schema migrations retain a
pre-migration checkpoint and support exact rollback.

Terminal verification passed structured long-history advantage (H_M1),
continuing-entity advantage (H_M12), model-replacement continuity (H_M14), and
coherence under sensory conflict, body change, model change, interruption, and
developmental time (H_M15).

## Sensorium and bodies

The typed sensorium admits eight local fixture modality classes:

```text
text, image, video, motion, audio, speech, depth_3d, body_tool
```

Raw signal, preprocessed signal, perceptual proposal, tracked world, inferred
event, verified relation, structural belief, and knowledge are separate
representation layers. Events bind sensor identity, time, sequence, coordinate
frame, source digest, preprocessing identity, model identity, confidence,
uncertainty, provenance, and quality or missing-data flags. Hidden target or
oracle identifiers are refused.

Mechanisms include checked coordinate transforms, object tracking through
occlusion and viewpoint change, event tracking, audiovisual timing,
cross-modal binding with conflict preservation, explicit 3D scene state, and an
expected-information active-perception policy. Raw terminal receipts contain
records for each claimed mechanism.

Two deterministic bodies are implemented: a desktop/browser-style sandbox and
a seeded 3D simulator. Physics state is separate from rendered observations.
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
and allowed roles. Every contract includes `independent_performer`; support
roles do not remove independent callability. The fabric implements
outcome-blind routing and measured relationships such as draft/verify, simulate,
translate, and route.

The modules are hand-specified deterministic scientific fixtures with no
training data and bounded synthetic competence. They are not claims of general
model competence. External cached models remain inventory only until license,
hash, strict-load, resource, and parity gates pass; none was admitted in v5.

Terminal verification passed model-fabric routing (H_M6) and model support
(H_M7) across principal, replication, and open-world evaluation.

## Campaign and evidence fabric

V5 writes canonical finite JSON beneath `configs/substrate/v5`,
`evidence/substrate/v5`, `runs/substrate/v5`, and
`artifacts/substrate/v5`. Named publications are atomic indexes over immutable
content-addressed sealed objects. Writers refuse path escape and any non-false
activation field.

The frozen DAG completed all 5,760 units and 576,000 sensory events or cognitive
episodes:

```text
principal                 3,456 / 3,456
independent replication   1,152 / 1,152
open-world review         1,152 / 1,152
```

Workers compute independently; one publisher validates and publishes receipts
and checkpoints atomically. The independent verifier does not trust the
principal summary: it loads sealed units and checkpoints, regenerates every
deterministic work unit, follows checkpoint chains, and rebuilds statistical and
classification inputs from raw phase rows.

The review package publishes a compressed raw-receipt archive and compact
indexes without committing the approximately 1.1 GiB operational run tree. The
completion scorecard separately reports implementation, mechanism activity,
instrument validity, cheap evidence, moderate evidence, principal evidence,
replication, and classification for every required category.

## Safety and claim boundary

`activation` is exactly `false`. V5 permits deterministic local simulations,
sandboxed interfaces, local tools, controlled virtual bodies, authorized
read-only data, and explicitly permitted human interactions. It forbids
purchase, account, carrier, financing, checkout, order, credential, activation,
and uncontrolled real-world actions.

The exact terminal label `multimodal_nous_ready_for_review` means eligible for
external review only. It is never an unqualified Nous declaration. No result
establishes consciousness, phenomenal experience, sentience, feeling,
suffering, desire, personhood, life, or moral status. “Model organism” is only
an engineering metaphor.

Authoritative terminal evidence:

- `evidence/substrate/v5/SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json`
- `evidence/substrate/v5/SUBSTRATE_V5_INDEPENDENT_VERIFICATION.json`
- `evidence/substrate/v5/SUBSTRATE_V5_MUTATION_REPORT.json`
- `evidence/substrate/v5/SUBSTRATE_V5_CLEAN_CLONE.json`
- `evidence/substrate/v5/SUBSTRATE_V5_FINAL_CLASSIFICATION.json`
- `evidence/substrate/v5/SUBSTRATE_V5_COMPLETION_SCORECARD.json`
- `artifacts/substrate/v5/SUBSTRATE_V5_TERMINAL_REPORT.md`
- `artifacts/substrate/v5/review/`
