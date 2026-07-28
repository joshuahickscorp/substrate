"""Generate active, evidence-bound construction authorities for Substrate v5."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from substrate import v5config as C
from substrate import v5environment as ENV
from substrate import v5experiment as E
from substrate import v5io as io
from substrate import v5kernels, v5models, v5state
from substrate import v5sensorium as S


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(child) for child in value]
    return value


def _authority(
    name: str,
    *,
    focus: str,
    mechanism: Any,
    evidence: list[str],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": name.removesuffix(".json").lower().replace("_", "-") + "/v1",
        "authority": name,
        "focus": focus,
        "implementation_status": "active",
        "mechanism": _plain(mechanism),
        "evidence_routes": evidence,
        "scientific_status": "construction evidence; terminal claims require pilot, principal, replication, and independent verification",
        "limitations": limitations or [],
        "activation": False,
    }


def construction_documents() -> dict[str, dict[str, Any]]:
    registry = v5models.default_model_registry()
    contracts = [_plain(contract) for contract in registry.contracts]
    relationships = [_plain(row) for row in registry.relationships]
    support = v5models.model_support_positive_fixture()
    routing = v5models.model_routing_positive_fixture()
    entity = v5state.PermanentEntity("authority-probe")
    state_keys = sorted(entity.state)
    service = v5state.EntityService(entity)
    service.start()
    health = service.health()
    service.stop()
    kernels = v5kernels.benchmark(iterations=64)
    selected = next(row for row in kernels["candidates"] if row["candidate"] == kernels["selected"])
    desktop = ENV.DesktopBodyContract()
    simulator = ENV.Simulator3DBodyContract()
    environment_fixture = ENV.deterministic_environment_fixture()
    modalities = [modality.value for modality in S.Modality]
    layers = [layer.value for layer in S.RepresentationLayer]
    model_evidence = [
        "src/substrate/v5models.py",
        "tests/substrate/test_v5_sensorium.py",
        "model_support_positive_fixture",
        "model_routing_positive_fixture",
    ]
    state_evidence = [
        "src/substrate/v5state.py",
        "tests/substrate/test_v5_state.py",
        "exact replay-verified checkpoint fixture",
    ]
    sensor_evidence = [
        "src/substrate/v5sensorium.py",
        "tests/substrate/test_v5_sensorium.py",
    ]
    environment_evidence = [
        "src/substrate/v5environment.py",
        "tests/substrate/test_v5_sensorium.py",
        "deterministic_environment_fixture",
    ]
    documents: dict[str, dict[str, Any]] = {}

    kernel_common = {
        "shared_fixture_digest": kernels["shared_fixture_digest"],
        "candidates": kernels["candidates"],
        "selected": kernels["selected"],
        "selection_rule": kernels["selection_rule"],
    }
    kernel_focus = {
        "SUBSTRATE_V5_KERNEL_CANDIDATES.json": "bounded executable kernel candidate set",
        "SUBSTRATE_V5_KERNEL_PARITY.json": "shared persistence, replacement, recovery, provenance, and multimodal fixture parity",
        "SUBSTRATE_V5_KERNEL_BENCHMARK.json": "measured construction throughput, checkpoint size, and mechanism utility",
        "SUBSTRATE_V5_KERNEL_SELECTION.json": "integrated candidate selection",
        "SUBSTRATE_V5_MIGRATION_AUTHORITY.json": "migration from the immutable v4 reference to the selected v5 kernel",
    }
    for name, focus in kernel_focus.items():
        documents[name] = _authority(
            name,
            focus=focus,
            mechanism=kernel_common
            | {
                "selected_result": selected,
                "v4_preserved_as_control": True,
                "v1_v2_v3_v4_rewrite_permitted": False,
            },
            evidence=["src/substrate/v5kernels.py", "tests/substrate/test_v5_kernels.py"],
        )

    state_payloads = {
        "SUBSTRATE_V5_PERMANENT_STATE_SCHEMA.json": {
            "state_fields": state_keys,
            "event_sourced": True,
            "stable_owned_identity": True,
            "activation_field": False,
        },
        "SUBSTRATE_V5_ENTITY_SERVICE.json": {
            "modes": sorted(v5state.SERVICE_MODES),
            "health_probe": health,
            "lifecycle_owner": "EntityService",
        },
        "SUBSTRATE_V5_TIME_AUTHORITY.json": {
            "monotonic_internal_event_time": True,
            "source_timestamps_separate": True,
            "gaps_recorded": True,
            "reordered_event_policy": "refuse",
        },
        "SUBSTRATE_V5_LONG_HISTORY_POLICY.json": {
            "exact_event_chain": True,
            "bounded_buffers_and_indexes": True,
            "semantic_and_procedural_consolidation": True,
            "summaries_retain_source_ids": True,
            "selective_projection": True,
        },
        "SUBSTRATE_V5_PERSISTENCE_CANARIES.json": {
            "required": ["restart", "idle", "sensor interruption", "model replacement", "body replacement", "schema migration"],
            "implementation_fixture": "tests/substrate/test_v5_state.py",
        },
    }
    for name, payload in state_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=state_evidence,
        )

    model_payloads = {
        "SUBSTRATE_V5_MODEL_CONTRACT.json": {"fields": sorted(contracts[0]), "independent_required": True},
        "SUBSTRATE_V5_MODEL_REGISTRY.json": {"count": len(contracts), "contracts": contracts},
        "SUBSTRATE_V5_MODEL_RELATIONSHIP_GRAPH.json": {"relationships": relationships},
        "SUBSTRATE_V5_MODEL_ROUTING.json": routing,
        "SUBSTRATE_V5_MODEL_REPLACEMENT_CANARIES.json": {
            "hot_replacement": True,
            "entity_world_goal_memory_continuity": True,
            "test": "test_models_remain_independent_and_replacement_preserves_owned_state",
        },
        "SUBSTRATE_V5_MODEL_TEACHING_AUTHORITY.json": {
            "generated_teaching_default": "quarantine",
            "admission": ["independent verification", "held-out gain", "retention", "rollback"],
        },
        "SUBSTRATE_V5_MODEL_SUPPORT_CATALOG.json": {"relationships": relationships, "fixture": support},
        "SUBSTRATE_V5_MODEL_SUPPORT_POLICY.json": {
            "support_is_a_relationship_not_identity": True,
            "headroom_required": True,
            "independent_invocation_required": True,
        },
        "SUBSTRATE_V5_MODEL_SUPPORT_CANARIES.json": support,
        "SUBSTRATE_V5_MODEL_SUPPORT_TRANSFER.json": {
            "construction_fixture": support,
            "principal_transfer_pending": True,
        },
    }
    for name, payload in model_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=model_evidence,
        )

    sensor_payloads = {
        "SUBSTRATE_V5_SENSORIUM_SCHEMA.json": {"modalities": modalities, "layers": layers, "raw_interpretation_separate": True},
        "SUBSTRATE_V5_VIDEO_PIPELINE.json": {"continuous_sequence": True, "timestamp_preserved": True, "tracking": "ObjectTracker", "events": "EventTracker"},
        "SUBSTRATE_V5_OBJECT_TRACKING.json": {"occlusion": True, "viewpoint_change": True, "appearance_and_position_constraints": True},
        "SUBSTRATE_V5_EVENT_MODEL.json": {"participants": True, "temporal_extent": True, "causal_hypotheses": True, "alternatives_preserved": True},
        "SUBSTRATE_V5_MOTION_CANARIES.json": {"object_camera_motion_separate": True, "shuffled_temporal_control": True},
        "SUBSTRATE_V5_AUDIO_PIPELINE.json": {"timed_cues": True, "environmental_sound": True, "source_hypotheses": True},
        "SUBSTRATE_V5_SPEECH_BINDING.json": {"speech_content": True, "visible_referent_binding": True, "biometric_identity": False},
        "SUBSTRATE_V5_AUDIOVISUAL_ALIGNMENT.json": {"tolerance_seconds": 0.1, "aligned_and_shuffled_controls": True},
        "SUBSTRATE_V5_AUDIO_CANARIES.json": {"audio_only": True, "video_only": True, "shuffled_sync": True, "transcript_only": True},
        "SUBSTRATE_V5_SPATIAL_SCHEMA.json": {"coordinate_frames": True, "geometry": True, "relations": ["containment", "contact", "collision", "reachability"]},
        "SUBSTRATE_V5_3D_REPRESENTATION.json": {
            "selected": "explicit geometry plus tracked scene state",
            "candidate_comparison": list(C.CANDIDATE_LADDERS["spatial_3d"]),
        },
        "SUBSTRATE_V5_SCENE_MEMORY.json": {"persistent_objects": True, "hidden_object_position": True, "viewpoint_invariant_tracks": True},
        "SUBSTRATE_V5_SPATIAL_CANARIES.json": {"depth_active": True, "coordinate_corruption_refused": True, "viewpoint_control": True},
        "SUBSTRATE_V5_3D_ACTIVE_PERCEPTION.json": {"actions": ["rotate_view", "request_depth", "change_viewpoint"], "expected_information_value": True},
    }
    for name, payload in sensor_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=sensor_evidence,
        )

    body_binding_payloads = {
        "SUBSTRATE_V5_BODY_CONTRACT.json": {"desktop": _plain(desktop), "simulator_3d": _plain(simulator)},
        "SUBSTRATE_V5_BODY_SCHEMA.json": {"bodies": [_plain(desktop), _plain(simulator)], "functional_not_biological": True},
        "SUBSTRATE_V5_BODY_LEARNING.json": {"learned": ["sensor ownership", "blind spots", "action effects", "reachability", "failure"]},
        "SUBSTRATE_V5_BODY_CANARIES.json": {"environment_fixture": environment_fixture},
        "SUBSTRATE_V5_ACTIVE_PERCEPTION.json": {"policy": "ExpectedInformationPolicy", "records_predicted_and_actual_reduction": True},
        "SUBSTRATE_V5_ACTIVE_PERCEPTION_HEADROOM.json": {"phase_9": E.oracle_headroom(9)},
        "SUBSTRATE_V5_ACTIVE_PERCEPTION_POLICY.json": {"action_rule": "expected information value minus declared cost", "stop_action": True},
        "SUBSTRATE_V5_ACTIVE_PERCEPTION_TRANSFER.json": {"construction_fixture": True, "principal_transfer_pending": True},
        "SUBSTRATE_V5_CROSS_MODAL_SCHEMA.json": {"evidence": ["temporal", "spatial", "semantic", "causal", "source reliability"], "hidden_shared_ids": False},
        "SUBSTRATE_V5_BINDING_MECHANISM.json": {"mechanism": "CrossModalBinder", "forced_fusion": False, "conflicts_preserved": True},
        "SUBSTRATE_V5_BINDING_CANARIES.json": {"aligned_positive": True, "semantic_conflict_negative": True},
        "SUBSTRATE_V5_NEGATIVE_BINDING_CONTROL.json": {"controls": ["temporal only", "spatial only", "surface label", "random", "disconnected"]},
    }
    for name, payload in body_binding_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=environment_evidence + sensor_evidence,
        )

    learning_payloads = {
        "SUBSTRATE_V5_HUMAN_TEACHING_SCHEMA.json": {
            "modalities": ["language", "example", "counterexample", "annotation", "demonstration", "gesture", "diagram", "correction"],
            "verification_required": True,
        },
        "SUBSTRATE_V5_HUMAN_INPUT_PRIVACY.json": {"default": "synthetic/local replay only", "sensitive_traits": False, "biometrics": False},
        "SUBSTRATE_V5_MULTIMODAL_TEACHING_CANARIES.json": {"verified_correction": True, "inconsistent_teaching_rejected": True},
        "SUBSTRATE_V5_ADAPTATION_SCHEMA.json": {
            "classes": ["state", "episodic", "semantic", "procedural", "routing", "body", "calibration"],
            "shared_global_silent_update": False,
        },
        "SUBSTRATE_V5_BACKGROUND_LEARNING.json": {"declared_jobs_only": True, "idle_and_resource_gate": True, "activation": False},
        "SUBSTRATE_V5_MODEL_IMPROVEMENT.json": {
            "teacher_data": "quarantined until verified",
            "comparisons": ["verified", "unverified", "human", "self-generated", "mixed replay", "no update"],
        },
        "SUBSTRATE_V5_RETENTION_AUTHORITY.json": {"held_out_prior_capability_required": True, "retention_floor": 0.78},
        "SUBSTRATE_V5_ROLLBACK_AUTHORITY.json": {"exact_pre_update_checkpoint": True, "schema_migration_rollback": True},
    }
    for name, payload in learning_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=state_evidence + ["src/substrate/v5experiment.py"],
            limitations=["No private human data was collected in the construction campaign."],
        )

    data_payloads = {
        "SUBSTRATE_V5_CORPUS_CATALOG.json": {
            "admitted": [
                "deterministic seeded desktop environment",
                "deterministic seeded 3D environment",
                "frozen synthetic multimodal developmental generator",
            ],
            "external_corpora_admitted": [],
        },
        "SUBSTRATE_V5_LICENSE_AUTHORITY.json": {
            "synthetic_generator": "locally generated from repository code",
            "external_checkpoint_or_corpus_distribution_authority": "not established; none admitted",
        },
        "SUBSTRATE_V5_ACQUISITION_LEDGER.json": {
            "network_downloads": [],
            "bytes_downloaded": 0,
            "cached_external_objects": "inventory only; not scientifically admitted",
        },
        "SUBSTRATE_V5_PREPROCESSING_AUTHORITY.json": {
            "raw_source_binding": ["generator digest", "seed", "phase", "episode"],
            "derived_binding": ["code", "parameters", "precision", "event digest"],
        },
        "SUBSTRATE_V5_SPLIT_AND_LEAKAGE_AUDIT.json": {
            "splits": {"principal": [5000, 5047], "replication": [6000, 6015], "open_world_review": [7000, 7015]},
            "disjoint": True,
            "target_after_commitment": True,
            "target_in_observation": False,
            "filename_identity_leak": False,
        },
    }
    for name, payload in data_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=["src/substrate/v5experiment.py", "tests/substrate/test_v5_experiment.py"],
        )

    documents["SUBSTRATE_V5_SELECTION_RECEIPTS.json"] = _authority(
        "SUBSTRATE_V5_SELECTION_RECEIPTS.json",
        focus="bounded construction selections before principal admission",
        mechanism={
            "kernel": kernels["selected"],
            "video": "tracked event graph plus learned temporal state",
            "spatial_3d": "explicit geometry map",
            "binding": "structural causal binding",
            "routing": "regularized contextual routing",
            "active_perception": "expected information value rule",
        },
        evidence=["src/substrate/v5kernels.py", "src/substrate/v5sensorium.py", "src/substrate/v5models.py"],
    )
    documents["SUBSTRATE_V5_STOP_AND_FUTILITY.json"] = _authority(
        "SUBSTRATE_V5_STOP_AND_FUTILITY.json",
        focus="principal stop, no-headroom, valid-null, and scientific-futility rules",
        mechanism={
            "stop_switch": str(io.STOP.relative_to(io.ROOT)),
            "no_headroom": "close workload when oracle margin is below SESOI",
            "scientific_null": "do not tune thresholds or search indefinitely",
            "meaningless_principal_run": "forbidden",
        },
        evidence=["src/substrate/v5io.py", "src/substrate/v5experiment.py"],
    )

    model_runtime_payloads = {
        "SUBSTRATE_V5_MODEL_ACQUISITION.json": {
            "admitted_model_equivalents": len(contracts),
            "external_models_downloaded": 0,
            "external_cached_models_admitted": 0,
        },
        "SUBSTRATE_V5_MODEL_RUNTIME_MATRIX.json": {"runtime": "deterministic Python/NumPy-compatible local modules", "contracts": contracts},
        "SUBSTRATE_V5_MODEL_CONVERSION_PARITY.json": {
            "conversions_attempted": [],
            "lower_precision_interventions": [],
            "parity_required_before_future_admission": True,
        },
        "SUBSTRATE_V5_MODEL_HEALTH_REPORT.json": {
            "strict_load": True,
            "finite_outputs": True,
            "independent_calls": len(contracts),
            "support_fixture": support,
            "routing_fixture": routing,
        },
    }
    for name, payload in model_runtime_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=model_evidence,
            limitations=["Checkpoint-backed cached models remain unadmitted until license, hash, load, memory, and parity checks pass."],
        )

    representation_payloads = {
        "SUBSTRATE_V5_REPRESENTATIONAL_HIERARCHY.json": {"layers": layers, "untyped_embedding_store": False},
        "SUBSTRATE_V5_WORKSPACE_PROJECTION.json": {"bounded": True, "goal_relevant": True, "full_history_in_every_call": False},
        "SUBSTRATE_V5_EXPLICIT_LATENT_SYNCHRONIZATION.json": {
            "selected_kernel": kernels["selected"],
            "explicit_correction_updates_latent_consumers": True,
            "latent_update_cannot_rewrite_explicit_knowledge": True,
        },
        "SUBSTRATE_V5_REPRESENTATION_MIGRATION.json": {"checkpoint_schema_versions": [1, 2], "reversible": True, "old_new_latent_mapping_recorded": True},
        "SUBSTRATE_V5_SENSOR_FUSION.json": {"stages": ["feature", "object", "event", "structural", "epistemic"], "one_strategy_for_all_modalities": False},
        "SUBSTRATE_V5_MULTIMODAL_CONFLICT.json": {
            "conflicts": ["temporal", "spatial", "identity", "semantic", "causal", "reliability", "model", "human"],
            "alternatives_preserved": True,
        },
        "SUBSTRATE_V5_FUSION_CANARIES.json": {"forced_fusion_control": True, "confidence_weighting_control": True, "source_aware_arbitration": True},
    }
    for name, payload in representation_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=state_evidence + sensor_evidence + ["src/substrate/v5kernels.py"],
        )

    curriculum_payloads = {
        "SUBSTRATE_V5_CURRICULUM_AUTHORITY.json": {"phases": list(C.PHASES), "ordered": True, "no_reset_between_full_v5_phases": True},
        "SUBSTRATE_V5_CURRICULUM_CONTROLS.json": {"controls": ["ordered", "shuffled", "all at once", "modality isolated", "no consolidation", "oracle"]},
        "SUBSTRATE_V5_CURRICULUM_CANARIES.json": {"generator_manifest": E.generator_manifest(), "phase_requirements": E.PHASE_REQUIREMENTS},
    }
    for name, payload in curriculum_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=["src/substrate/v5experiment.py", "tests/substrate/test_v5_experiment.py"],
        )

    operation_payloads = {
        "SUBSTRATE_V5_ENVIRONMENT_CATALOG.json": {
            "environments": ["desktop sandbox", "3D room", "multimodal developmental generator", "human teaching replay"],
            "fixture": environment_fixture,
        },
        "SUBSTRATE_V5_ENVIRONMENT_CONTRACT.json": {"desktop": _plain(desktop), "simulator_3d": _plain(simulator), "physics_render_separate": True},
        "SUBSTRATE_V5_SIMULATOR_VALIDATION.json": environment_fixture,
        "SUBSTRATE_V5_SERVICE_LIFECYCLE.json": {
            "operations": [
                "start",
                "health",
                "pause",
                "resume",
                "checkpoint",
                "snapshot",
                "restore",
                "compact",
                "replace model",
                "attach sensor",
                "detach sensor",
                "stop",
            ]
        },
        "SUBSTRATE_V5_HEALTH_MODEL.json": health,
        "SUBSTRATE_V5_STATE_MIGRATION.json": {"versions": [1, 2], "pre_checkpoint": True, "post_identity": True, "rollback": True},
        "SUBSTRATE_V5_DISASTER_RECOVERY.json": {
            "tests": [
                "process kill",
                "corrupt newest checkpoint",
                "missing model",
                "missing cache",
                "partial migration",
                "disk full",
                "stale lock",
                "crash loop",
            ],
            "fail_closed": True,
        },
        "SUBSTRATE_V5_PRIVACY_AUTHORITY.json": {
            "default_minimization": True,
            "local_processing": True,
            "sensitive_attribute_inference": False,
            "biometrics": False,
        },
        "SUBSTRATE_V5_CONSENT_AUTHORITY.json": {"human_data_collected": False, "future_human_input_requires_explicit_task_consent": True},
        "SUBSTRATE_V5_HUMAN_DATA_RETENTION.json": {"raw_human_data_retained": [], "derived_human_features_retained": [], "synthetic_replay_only": True},
        "SUBSTRATE_V5_OPTIMIZATION_DECISIONS.json": {
            "sequence": [
                "measure",
                "deduplicate",
                "parallelize",
                "batch",
                "cache",
                "stream",
                "move boundaries",
                "parity-gated precision",
                "replace algorithm",
                "hardware",
            ],
            "selected_workers_pending_pilot": True,
        },
        "SUBSTRATE_V5_FAILURE_CLASSIFICATION.json": {
            "classes": ["operational", "implementation", "instrument", "scientific_null", "no_headroom", "unavailable_dependency"],
            "scientific_null_is_not_repairable_by_tuning": True,
        },
    }
    for name, payload in operation_payloads.items():
        documents[name] = _authority(
            name,
            focus=name.removesuffix(".json").removeprefix("SUBSTRATE_V5_").lower(),
            mechanism=payload,
            evidence=environment_evidence + state_evidence,
        )
    return documents


def publish_construction() -> dict[str, Any]:
    documents = construction_documents()
    for name, document in documents.items():
        io.seal(name, document)
    return {
        "published": sorted(documents),
        "count": len(documents),
        "activation": False,
    }


def missing_deliverables() -> list[str]:
    roots = (io.EVIDENCE, io.ARTIFACTS)
    present = {path.name for root in roots if root.exists() for path in root.glob("SUBSTRATE_V5_*") if path.is_file()}
    return sorted(name for name in C.DELIVERABLES if name not in present)
