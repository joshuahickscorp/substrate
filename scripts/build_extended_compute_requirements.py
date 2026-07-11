#!/usr/bin/env python3
"""Build and verify the extended-compute blocker matrix.

The matrix is deliberately conservative: category 8 or 9 and
``hardware_required: true`` are rejected unless a named, locally measured or
calculated enablement gate has passed.  Run with ``--check`` to verify that the
committed JSON is exactly reproducible from the current registry and evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "proof" / "EXTENDED_COMPUTE_REQUIREMENTS.json"
EXPECTED_REGISTRY_COUNT = 227
EXPECTED_REGISTRY_ID_SHA256 = "53de3591be3ba7e8eeb7bf644e6b57063ee8d3a615898976b007b8fb7404b1c6"
P5_VERIFICATION_PATH = "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
P5_VERIFICATION_SCHEMA = "mop-p5-context-independent-verifier/v1"
P5_SMOKE_RECEIPT_PATH = "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json"
P5_SMOKE_RECEIPT_SCHEMA = "mop-local-throttle-receipt/v1"
P5_SMOKE_TASK_ID = "p5smoke_cpu"
P5_SMOKE_EXPECTED_RUN_ID = "p5smoke_20260711_leg3"
P5_SMOKE_CPU_REASON = "first_lane normalized one-minute load ceiling"
P5_SMOKE_MEMORY_REASON = "measured available unified memory covers candidate peak plus headroom"
P5_SMOKE_COMMAND = [
    ".venv/bin/python",
    "scripts/p5_context_capability.py",
    "--profile",
    "p5smoke",
    "--device",
    "cpu",
    "--run-dir",
    "runs/p5_context/p5smoke",
    "--out",
    "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
]
P5_VERIFICATION_FIELDS = {
    "verification_complete": True,
    "all_ok": True,
    "prerequisite_ready": True,
    "problems": [],
    "all_controls_passed": True,
    "all_mutations_rejected": True,
    "controls.seed_arm_checkpoint_artifacts_exactly_joined": True,
    "controls.nonterminal_outcome_has_off_ceiling_multiunit_support": True,
    "controls.threshold_tie_is_null": True,
    "independence.checkpoint_files_opened_with_weights_only": True,
    "independence.checkpoint_model_and_target_state_hashes_recomputed": True,
    "independence.heldout_metrics_reexecuted_from_checkpoint": False,
    "outcome_contract.tie_is_null": True,
    "promotion.confirmatory_promotable": False,
    "scientific_promotion": False,
}

CATEGORY_LABELS = {
    1: "already runnable locally",
    2: "locally runnable after implementation",
    3: "locally runnable after data or rights intake",
    4: "locally runnable serially but slow",
    5: "locally runnable with memory/runtime factorization",
    6: "external environment, sensor, participant, or physical specimen",
    7: "unpublished or unavailable upstream model",
    8: "empirically extended-compute-beneficial but not required",
    9: "empirically extended-compute-required for a named target",
}

RUNG_DEFINITIONS = {
    "L0": {
        "kind": "current-local",
        "summary": "M3 Pro bounded shard, one heavy process, 300 minutes, 40 GB free-disk floor",
    },
    "L1": {
        "kind": "current-local-resumable",
        "summary": "same host across atomic overnight or multi-day serial shards",
    },
    "L2": {
        "kind": "larger-single-node-uma",
        "summary": "larger-memory Apple Silicon or equivalent, only after a measured working-set failure",
    },
    "L3": {
        "kind": "single-high-memory-accelerator",
        "summary": "one 96-141 GB CUDA-class accelerator for a bounded parity pilot",
    },
    "L4": {
        "kind": "small-multi-accelerator",
        "summary": "2-4 accelerators; aggregate memory requires measured sharding, not DDP",
    },
    "L5": {
        "kind": "distributed-accelerator",
        "summary": "eight-accelerator or multi-node campaign after L4 failure",
    },
    "L6": {
        "kind": "external-infrastructure",
        "summary": "environment, sensor, participant, power meter, robot, or physical specimen; not a compute rung",
    },
}

MEASUREMENT_IDS = {
    "m_host_profile",
    "m_vitl_cache",
    "m_cm7_pilot",
}

CALCULATION_IDS = {
    "c_cache_10000",
    "c_dense_cache_storage",
    "c_adam_trainable_state",
    "c_paired_seed_power",
    "c_energy_scope",
    "c_active_disk_floor",
    "c_peak_memory_spec",
    "c_checkpoint_volume_spec",
    "c_campaign_time_spec",
    "c_cloud_cost_spec",
}

# This catalog is intentionally role-empty at this snapshot.  Existing local
# measurements and sizing formulas are useful evidence, but none is a repeated
# boundary failure, two-rung same-workload comparison, parity pilot, or
# smallest-rung derivation.  Gate validation inspects these typed roles rather
# than accepting an arbitrary known ID or a self-asserted boolean.
GATE_EVIDENCE_SPECS: dict[str, dict[str, Any]] = {
    "m_host_profile": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:proof/STUDIO_READINESS_CURRENT_HOST.json",
    },
    "m_vitl_cache": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
    },
    "m_cm7_pilot": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:proof/CUSTOM_SUBSTRATE_PILOT.json",
    },
    **{
        identifier: {
            "record_type": "calculation",
            "gate_roles": [],
            "source_ref": "local:scripts/build_extended_compute_requirements.py",
        }
        for identifier in CALCULATION_IDS
    },
}

# These are explicit exceptions to evidence-derived registry classification. Registry-only F rows
# are preregistration-only category 2 by default, or category 6 when their declared resource tier
# is environment-needed. The official ViT-B runtime and E6/DR14 cache-control interfaces are
# verified locally, so their first remaining blocker is a citable natural task cohort rather than
# hardware.
REGISTRY_OVERRIDES: dict[str, tuple[int, str, str]] = {
    "e5_curiosity": (
        1,
        "rerun the current bounded curiosity class and retain the existing local action-environment mechanics as supporting evidence",
        "implemented-local-evidence-source-rerun-pending",
    ),
    "mop_p5_context_capability": (
        1,
        "complete the implemented conditional P5 sequence through the local governor after three healthy admission samples",
        "implemented-governor-execution-pending",
    ),
    "e6_relational": (
        3,
        "materialize a rights-clean annotated natural-video cohort with untouched test membership, then encode the declared learned/random cache pair serially",
        "verified-e6-integration-plus-citable-natural-input-gap",
    ),
    "mop_dr5_cross_substrate_consistency": (
        2,
        "implement citable same-architecture random controls and the expanded compatible-task grid",
        "registry-relation-plus-cache-audit",
    ),
    "mop_dr14_corruption": (
        3,
        "materialize the citable dense natural-task cache consumed by the implemented deterministic nested corruption views",
        "verified-dr14-integration-plus-citable-natural-input-gap",
    ),
    "mop_at1_nuisance_grid": (
        2,
        "complete the citable active ViT-B/custom instrument grid and matched random-init columns",
        "registry-relation-plus-random-control-quarantine-audit",
    ),
    "mop_cm6_distilled_density": (
        2,
        "implement the trainable student and matched non-distilled/random controls around an already local teacher cache",
        "registry-relation-plus-local-cache-receipts",
    ),
    "mop_mt4_reasoning_router": (
        2,
        "wire six compatible verified primitive outputs and an independent verifier",
        "registry-relation-plus-ledger-audit",
    ),
    "mop_pr2_plasticity_substrates": (
        2,
        "generate the missing citable random-init-ViT control cache",
        "registry-relation-plus-cache-audit",
    ),
    "mop_at2_mode_substrate_dep": (
        2,
        "complete the citable real/random cache condition and verified winning-mode input",
        "registry-relation-plus-cache-audit",
    ),
    "mop_cm11_developmental_plasticity": (
        2,
        "implement a calibrated curriculum and independent signature recomputation",
        "registry-relation-plus-ledger-audit",
    ),
    "mop_cm12_mop_substrate_capstone": (
        2,
        "assemble compatible experts, a shared battery, and the declared open-model control",
        "registry-relation-plus-ledger-audit",
    ),
    "ex5_local_rules_scale": (
        2,
        "rerun the current changed source under a fresh source-bound receipt before treating the mechanics as current",
        "live-source-drift-audit",
    ),
    "mop_dr1_video_cache": (
        3,
        "rights-cleared, session-diverse natural video and a citable cache",
        "project-exhaustion-ledger",
    ),
    "mop_dr2_sparse_real": (3, "a citable full real-latent stream", "project-exhaustion-ledger"),
    "mop_dr3_latent_scratchpad": (3, "compatible citable real-latent tasks", "project-exhaustion-ledger"),
    "mop_dr4_causal_intervention": (
        3,
        "interventional trajectories with preserved provenance",
        "project-exhaustion-ledger",
    ),
    "mop_dr7_latent_cot": (3, "compatible citable real-latent tasks", "project-exhaustion-ledger"),
    "mop_dr15_modality_general": (
        3,
        "compatible video, relational/language, and audio cache families",
        "project-exhaustion-ledger",
    ),
    "mop_al2_shared_latent_alignment": (
        3,
        "expanded meaningful shared content with split-safe provenance",
        "project-exhaustion-ledger",
    ),
    "mop_al3_audio_video_alignment": (
        3,
        "rights-cleared temporally aligned audio-video clips; the documented SANPO release/schema exposes no audio modality",
        "project-exhaustion-ledger-plus-dataset-audit",
    ),
    "mop_cm1_compositional_gate": (
        3,
        "rights-cleared bound-attribute natural video",
        "project-exhaustion-ledger",
    ),
    "mop_cm2_atlas_gate": (
        3,
        "the upstream CM1 natural-data result before an atlas decision is meaningful",
        "dependency-order-audit",
    ),
    "mop_cm3_dense_vs_pooled": (
        3,
        "dense natural-video tokens on a held-out-combination task",
        "project-exhaustion-ledger",
    ),
    "mop_cm4_workspace_shell": (3, "citable full real-cache inputs", "project-exhaustion-ledger"),
    "mop_cm8_custom_jepa_pilot": (
        3,
        "CM1/DR1 natural data and a same-referent teacher battery",
        "cm8-preflight",
    ),
    "mop_cm9_slot_jepa_binding": (
        3,
        "multi-object natural video and binding annotations",
        "project-exhaustion-ledger",
    ),
    "f8_plastic_substrate_rewrite": (
        3,
        "trusted natural trajectories and provenance; its configured 6 GB cap has not failed",
        "form-contract-audit",
    ),
    "f16_perfect_slate_null": (
        3,
        "trusted natural trajectories and provenance; its configured 6 GB cap has not failed",
        "form-contract-audit",
    ),
    "e10_openended": (
        2,
        "implement bounded population search, environment generation, sustained-horizon, and transfer gates",
        "local-action-environment-plus-registry",
    ),
    "mop_cm10_action_forward_model": (
        3,
        "obtain independently sourced action-conditioned trajectories with an exact-referent action control and predeclared replication, then reuse the verified local P7 harness",
        "verified-p7-programmatic-mechanics-plus-external-validity-gate",
    ),
}

F_PROPOSED_CATEGORY = {
    65: 6,
    66: 6,
}

F_PROPOSED_BLOCKER = {
    65: "multiple fabricated specimens",
    66: "a real material substrate for the cross-substrate transfer endpoint",
}

F_FAMILY_BLOCKERS = {
    range(21, 29): "implement the controlled temporal/acquisition/binding fixture and its matched baseline",
    range(
        29, 36
    ): "implement the action-conditioned simulator, yoked intervention, telemetry, and causal-state hooks",
    range(36, 45): "implement the broadcast, calibration, memory-lifecycle, and delayed-condition harness",
    range(
        45, 53
    ): "implement the lifetime stream, adaptive-capacity controls, lesion/curriculum protocol, and fixed QD horizon",
    range(
        53, 59
    ): "implement simulated held-out partner populations with fixed channel, data, and lineage budgets",
    range(
        59, 61
    ): "implement poisoning fixtures and transactional commit, shadow-evaluation, and rollback mechanics",
    range(
        61, 65
    ): "implement the expansion plan's synthetic-device and simulated material/drift/damage stage",
}

THEMATIC_FRONTIERS = [
    (
        "natural_video_objective_tournament",
        "Natural-video objective tournaments",
        3,
        "independent rights-cleared sessions and exact provenance",
    ),
    (
        "teacher_free_substrate_scaling",
        "Teacher-free custom-substrate scaling",
        2,
        "implement a log-spaced matched-budget sweep, then measure whether it is slow or memory-limited",
    ),
    (
        "dense_long_context_video",
        "Dense high-resolution long-context video",
        2,
        "implement exact/windowed/recurrent/checkpointed attacks before assigning a measured ceiling category",
    ),
    (
        "learned_vs_random_scale_controls",
        "Same-referent learned versus random controls",
        2,
        "implement citable architecture-matched controls",
    ),
    (
        "multi_seed_ablation_matrix",
        "Powered multi-seed objective and ablation matrix",
        2,
        "name endpoint, SESOI, variance source, multiplicity family, and run harness before calling it slow",
    ),
    (
        "continual_million_event_learning",
        "Continual learning over millions of events",
        2,
        "run the implemented disk stream at 10,000 and 100,000 events, then one million only if the shorter rung leaves the horizon decision unresolved",
    ),
    (
        "action_conditioned_world_models",
        "Action-conditioned persistent world models",
        3,
        "obtain independently sourced action-conditioned rendered or natural trajectories with an exact-referent control and predeclared replication, then run the existing local harness before any scale campaign",
    ),
    (
        "active_perception",
        "Active perception under sensor cost",
        2,
        "a local simulator and selective-acquisition baseline before physical sensors",
    ),
    (
        "native_multimodal_binding",
        "Native audio/video/language/action/telemetry binding",
        3,
        "compatible aligned modalities with rights and split discipline",
    ),
    (
        "natural_object_event_causal_state",
        "Natural object, event, relation, and causal-state discovery",
        3,
        "rights/provenance-clean natural multi-object sessions",
    ),
    (
        "population_openended_social",
        "Population, open-ended, and social learning",
        2,
        "bounded local multi-agent implementation and predeclared horizon",
    ),
    (
        "workspace_operational_self_model",
        "Calibrated workspace and operational self-model",
        3,
        "obtain independent natural workload and failure episodes with prospectively registered telemetry, real bounded interventions, and replicated held-out calibration",
    ),
    (
        "small_substrate_architecture_search",
        "Small-substrate architecture search",
        2,
        "proxy-benchmark implementation and exact retraining of a shortlist",
    ),
    (
        "digital_material_simulation",
        "Digital material-substrate simulation",
        2,
        "calibrated simulator plus matched conventional controls",
    ),
    (
        "staged_robustness_sweeps",
        "Corruption/domain/counterfactual robustness sweeps",
        2,
        "implement a minimal adaptive screen before expanding a measured matrix",
    ),
    (
        "full_system_density_accounting",
        "Full-system performance-density accounting",
        3,
        "run the implemented accountant over independent natural end-to-end workloads and failures; attach a meter with an explicit system boundary before any energy claim",
    ),
    (
        "physical_sensor_participant_validation",
        "Physical, sensor, and participant validation",
        6,
        "named devices, specimens, sensors, or participants; compute cannot substitute",
    ),
]

WORKSTREAMS = [
    ("w0", "Real-evidence completion", 3, "rights-cleared real and natively aligned data"),
    (
        "w1",
        "Temporal referents and event identity",
        2,
        "controlled temporal mechanics before natural-data promotion",
    ),
    (
        "w2",
        "Active multimodal perception",
        2,
        "implement the local selective-acquisition simulator and sensor-cost controls",
    ),
    (
        "w3",
        "Boundary, agency, and body model",
        2,
        "extend the verified local action contract with body/tool interventions and yoked controls",
    ),
    ("w4", "Memory lifecycle and continuity", 2, "bounded local lifecycle implementation and controls"),
    (
        "w5",
        "Multiscale plasticity and morphogenesis",
        2,
        "bounded local growth/plasticity implementation and matched final capacity",
    ),
    (
        "w6",
        "Workspace and operational self-model",
        2,
        "causal broadcast, telemetry, and report-grounding hooks",
    ),
    (
        "w7",
        "Social reference and cultural accumulation",
        2,
        "simulated partner populations before participant validation",
    ),
    ("w8", "Open-ended developmental ecology", 2, "bounded local ecology with a fixed evaluation horizon"),
    ("w9", "Material substrate simulation", 2, "digital material simulator and conventional controls"),
    ("w10", "Bench material computing", 6, "bench devices and multiple physical specimens"),
    (
        "w11",
        "Ethics, welfare uncertainty, and containment",
        2,
        "governance, audit, and containment implementation",
    ),
]

PILLARS = [
    (7, "Time, referents, objects, and events", 2),
    (8, "Vision through the computer", 3),
    (9, "Audio through the computer", 3),
    (10, "Multisensory binding and computer interoception", 3),
    (11, "Prediction, world models, and imagination", 2),
    (12, "Action, affordance, agency, and boundaries", 2),
    (13, "Memory as a governed ecology", 2),
    (14, "Plasticity and moldability", 2),
    (15, "Biological motifs beyond neurons", 2),
    (16, "Developmental cognition", 2),
    (17, "Metacognition, self-models, and reports", 2),
    (18, "Workspace, attention, and mode ecology", 2),
    (19, "Social cognition, communication, teaching, and culture", 2),
    (20, "Curiosity, open-endedness, and artificial ecology", 2),
    (21, "Material computation landscape", 2),
    (22, "Morphogenesis, repair, and adaptive matter", 2),
    (23, "Neuromorphic and compositional representation", 2),
    (24, "Performance, density, and hardware reality", 2),
    (25, "Safety, security, and governance", 2),
]

LOCAL_SOURCE_PATHS = [
    "EXTENDED_COMPUTE_RESEARCH_PROMPT.md",
    "FORM_SUBSTRATE_DEEP_RESEARCH_2026_07.md",
    "FORM_SUBSTRATE_DEEP_EXPANSION_PLAN.md",
    "docs/CUSTOM_SUBSTRATE_WORKBENCH.md",
    "docs/LOCAL_CEILING_RESIDUAL_AUDIT_2026_07_10.md",
    "registry/experiments.yaml",
    "proof/PROJECT_EXPERIMENT_EXHAUSTION.json",
    "proof/FRONTIER_LOCALIZATION.json",
    "proof/LOCAL_ACTION_ENVIRONMENT.json",
    "proof/STUDIO_READINESS_CURRENT_HOST.json",
    "proof/CUSTOM_SUBSTRATE_CALIBRATION.json",
    "proof/CUSTOM_SUBSTRATE_PILOT.json",
    "proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json",
    "proof/CUSTOM_SUBSTRATE_ABORTED_MPS_RECOVERY.json",
    "proof/CUSTOM_SUBSTRATE_ABORTED_SOURCE_DRIFT.json",
    "proof/CACHE_QUARANTINE_AUDIT.json",
    "proof/SANPO_REAL_SMOKE_INTAKE.json",
    "proof/SANPO_REAL_SMOKE_INTAKE_DRY_RUN.json",
    "proof/SANPO_REAL_SMOKE_VERIFICATION.json",
    "proof/VJEPA21_VITB_LOAD.json",
    "proof/VJEPA21_VITB_FORWARD.json",
    "proof/VJEPA21_VITB_FORWARD_64F.json",
    "proof/E6_VITB_DENSE_PREFLIGHT.json",
    "proof/EXPANSION_WAVE0.json",
    "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json",
    "proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json",
    "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
    "proof/P9_ACCOUNTING_MECHANICS.json",
    "proof/P9_CAUSAL_MONITORING_PREFLIGHT.json",
    "proof/FORM_SUBSTRATE/CONTRACT_AUDIT.json",
    "proof/FORM_SUBSTRATE/LOCAL_RUN_SUMMARY.json",
    "proof/FORM_SUBSTRATE/SCORECARD.json",
    "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json",
    "proof/ARTIFACT_INDEX/form_substrate.json",
    "configs/encoder/vjepa21_vitb.yaml",
    "configs/experiment/e6_dense_cache.yaml",
    "configs/experiment/continual_million_event_preflight.yaml",
    "configs/experiment/continual_million_event_rungs.yaml",
    "configs/experiment/p7_action_world_model_preflight.yaml",
    "configs/experiment/p9_causal_monitoring_preflight.yaml",
    "configs/local_execution_throttle.yaml",
    "src/mop/substrate/vjepa21_official.py",
    "src/mop/substrate/vjepa21_dense_tasks.py",
    "scripts/vjepa21_official.py",
    "scripts/vjepa21_dense_tasks.py",
    "scripts/continual_million_event_preflight.py",
    "scripts/continual_million_event_rung.py",
    "scripts/p7_action_world_model_preflight.py",
    "scripts/p9_causal_monitoring_preflight.py",
    "docs/VJEPA21_LOCAL_INTEGRATION.md",
    "docs/E6_DENSE_RELATIONAL_CACHE.md",
    "docs/P6_CONTINUAL_MILLION_EVENT_AUDIT_2026_07.md",
    "docs/P7_ACTION_WORLD_MODEL_AUDIT_2026_07.md",
    "docs/P9_CAUSAL_MONITORING_PREFLIGHT.md",
    "src/mop/substrate/continual_stream.py",
    "src/mop/studies/continual_million_event.py",
    "src/mop/studies/action_world_model.py",
    "src/mop/studies/p9_accounting.py",
    "src/mop/studies/p9_causal_monitoring.py",
    "data/cache/vjepa2_vitl_local8_citable/cache_manifest.json",
    "data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
    "scripts/build_extended_compute_requirements.py",
    "EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md",
    "EXTENDED_COMPUTE_EXECUTION_PLAN.md",
]

EXTERNAL_SOURCES = [
    {
        "id": "meta-vjepa21",
        "url": "https://github.com/facebookresearch/vjepa2",
        "fact": "V-JEPA 2.1 release dated 2026-03-16 and the retained official dense ViT-B checkpoint",
    },
    {
        "id": "meta-vjepa2-paper",
        "url": "https://arxiv.org/abs/2506.09985",
        "fact": "V-JEPA 2 and action-conditioned planning evidence",
    },
    {
        "id": "meta-vjepa21-paper",
        "url": "https://arxiv.org/abs/2603.14482",
        "fact": "V-JEPA 2.1 dense representation evidence",
    },
    {
        "id": "sanpo",
        "url": "https://google-research-datasets.github.io/sanpo_dataset/",
        "fact": "video-first stereo/depth/IMU dataset description and CC BY 4.0 link; no native audio listed",
    },
    {
        "id": "cc-by-4",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "fact": "attribution and change-indication obligations; other rights may remain",
    },
    {
        "id": "avid",
        "url": "https://github.com/piergiaj/AViD",
        "fact": "official access, source-license, redistribution, and face-blurring claims; audio still requires audit",
    },
    {
        "id": "yfcc100m",
        "url": "https://arxiv.org/abs/1503.01817",
        "fact": "per-item Creative Commons metadata",
    },
    {
        "id": "wikimedia-licensing",
        "url": "https://commons.wikimedia.org/wiki/Commons:Licensing",
        "fact": "per-file license and source-page review",
    },
    {
        "id": "v3c",
        "url": "https://arxiv.org/abs/1810.04401",
        "fact": "Creative Commons Vimeo collection; access agreement remains separate",
    },
    {
        "id": "droid",
        "url": "https://droid-dataset.github.io/",
        "fact": "action/proprioceptive trajectories, CC BY 4.0 claim, storage and privacy caveats",
    },
    {
        "id": "audioset",
        "url": "https://research.google.com/audioset/",
        "fact": "annotation licensing does not transfer underlying media rights",
    },
    {
        "id": "videomae",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html",
        "fact": "extreme masking as a data/compute-efficient video baseline",
    },
    {
        "id": "memvit",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.html",
        "fact": "memory-augmented long-video baseline",
    },
    {
        "id": "flashattention",
        "url": "https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html",
        "fact": "exact attention can be I/O-aware; naive score residency is not a hardware proof",
    },
    {
        "id": "transformer-xl",
        "url": "https://aclanthology.org/P19-1285/",
        "fact": "segment recurrence baseline",
    },
    {
        "id": "mamba2",
        "url": "https://proceedings.mlr.press/v235/dao24a.html",
        "fact": "state-space/recurrent alternative to quadratic temporal attention",
    },
    {
        "id": "ewc",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101/",
        "fact": "continual-learning retention baseline",
    },
    {
        "id": "clear",
        "url": "https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/2838023a778dfaecdc212708f721b788-Abstract-round2.html",
        "fact": "streaming temporal continual-evaluation benchmark",
    },
    {
        "id": "slot-attention",
        "url": "https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html",
        "fact": "compact object-centric baseline",
    },
    {
        "id": "citris",
        "url": "https://proceedings.mlr.press/v162/lippe22a.html",
        "fact": "causal representation learning with interventions",
    },
    {
        "id": "minigrid",
        "url": "https://github.com/Farama-Foundation/Minigrid",
        "fact": "lightweight Apache-2.0 local interactive environments",
    },
    {
        "id": "map-elites",
        "url": "https://arxiv.org/abs/1504.04909",
        "fact": "quality-diversity population baseline",
    },
    {
        "id": "qdax",
        "url": "https://arxiv.org/abs/2308.03665",
        "fact": "accelerated quality-diversity implementation and benchmark",
    },
    {
        "id": "melting-pot",
        "url": "https://proceedings.mlr.press/v139/leibo21a/leibo21a.pdf",
        "fact": "multi-agent partner/generalization substrate",
    },
    {
        "id": "nas-bench-101",
        "url": "https://proceedings.mlr.press/v97/ying19a.html",
        "fact": "queryable architecture-search benchmark",
    },
    {
        "id": "nats-bench",
        "url": "https://arxiv.org/abs/2009.00437",
        "fact": "topology and size architecture benchmarks",
    },
    {
        "id": "nas-rank-bias",
        "url": "https://proceedings.mlr.press/v162/xu22h.html",
        "fact": "weight-sharing rank-bias evidence requiring exact shortlist retraining",
    },
    {
        "id": "video-robustness",
        "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Schiappa_A_Large-Scale_Robustness_Analysis_of_Video_Action_Recognition_Models_CVPR_2023_paper.html",
        "fact": "video corruption and robustness evaluation",
    },
    {
        "id": "neurobench",
        "url": "https://www.nature.com/articles/s41467-025-56739-4",
        "fact": "separates algorithm evaluation from physical system evaluation",
    },
    {
        "id": "physical-nn",
        "url": "https://www.nature.com/articles/s41586-021-04223-6",
        "fact": "physical-device noise can make simulation insufficient for device claims",
    },
    {
        "id": "pytorch-mps",
        "url": "https://docs.pytorch.org/docs/main/mps.html",
        "fact": "MPS allocated, driver-allocated, and recommended working-set memory APIs",
    },
    {
        "id": "pytorch-mps-env",
        "url": "https://docs.pytorch.org/docs/stable/mps_environment_variables.html",
        "fact": "MPS allocator watermarks and fallback controls",
    },
    {
        "id": "pytorch-checkpoint",
        "url": "https://docs.pytorch.org/docs/stable/checkpoint",
        "fact": "activation checkpointing behavior and RNG caveats",
    },
    {
        "id": "pytorch-amp",
        "url": "https://docs.pytorch.org/docs/stable/amp.html",
        "fact": "mixed-precision APIs and numerical caveats",
    },
    {
        "id": "pytorch-fsdp",
        "url": "https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html",
        "fact": "parameter, gradient, and optimizer-state sharding",
    },
    {
        "id": "apple-studio",
        "url": "https://support.apple.com/en-us/122211",
        "fact": "live M4 Max/M3 Ultra memory, bandwidth, storage, and I/O specifications",
    },
    {
        "id": "apple-power",
        "url": "https://support.apple.com/en-ca/102027",
        "fact": "measured wall-power configurations including PSU loss",
    },
    {
        "id": "nvidia-rtxpro6000",
        "url": "https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/",
        "fact": "96 GB GDDR7 and power envelopes",
    },
    {
        "id": "nvidia-h200",
        "url": "https://www.nvidia.com/en-au/data-center/h200/",
        "fact": "141 GB HBM3e, bandwidth, and power",
    },
    {
        "id": "google-gpu",
        "url": "https://docs.cloud.google.com/compute/docs/gpus",
        "fact": "G4 and A3 machine topology and memory",
    },
    {
        "id": "google-pricing",
        "url": "https://cloud.google.com/products/compute/pricing/accelerator-optimized",
        "fact": "dynamic accelerator-optimized instance prices",
    },
    {
        "id": "aws-p5",
        "url": "https://aws.amazon.com/ec2/instance-types/p5/",
        "fact": "P5/P5e/P5en accelerator, host, NVMe, NVSwitch, and network configurations",
    },
    {
        "id": "aws-capacity-price",
        "url": "https://aws.amazon.com/ec2/capacityblocks/pricing/",
        "fact": "dated Capacity Block accelerator-hour prices",
    },
    {
        "id": "aws-capacity-duration",
        "url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-how.html",
        "fact": "Capacity Block minimum duration",
    },
    {
        "id": "dgx-b200",
        "url": "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html",
        "fact": "DGX B200 memory, storage, interconnect, and maximum power",
    },
    {
        "id": "mlperf-power",
        "url": "https://docs.mlcommons.org/inference/power/",
        "fact": "instrumented system power methodology",
    },
    {
        "id": "green-algorithms",
        "url": "https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202100707",
        "fact": "computation energy/carbon accounting framework",
    },
    {
        "id": "seeds",
        "url": "https://arxiv.org/abs/1806.08295",
        "fact": "random-seed uncertainty and reporting",
    },
    {
        "id": "rliable",
        "url": "https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html",
        "fact": "interval and aggregate evaluation for stochastic algorithms",
    },
]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _p5_smoke_refusal_summary(receipt: dict[str, Any], *, evidence_root: Path = ROOT) -> dict[str, Any]:
    """Parse the current P5 local-admission refusal or fail closed on semantic drift."""
    problems: list[str] = []

    if receipt.get("schema") != P5_SMOKE_RECEIPT_SCHEMA:
        problems.append("schema")
    run_id = receipt.get("run_id")
    if run_id != P5_SMOKE_EXPECTED_RUN_ID:
        problems.append("run_id")
    if receipt.get("mode") != "execute-refused":
        problems.append("mode")
    if receipt.get("status") != "admission-refused":
        problems.append("status")
    if receipt.get("command_executed") is not False:
        problems.append("command_executed")
    if receipt.get("active_lanes") != []:
        problems.append("active_lanes")

    task = receipt.get("task")
    if not isinstance(task, dict):
        problems.append("task")
        task = {}
    if task.get("task_id") != P5_SMOKE_TASK_ID:
        problems.append("task.task_id")
    if task.get("lane") != "heavy" or task.get("accelerator") != "none":
        problems.append("task.resource_class")
    if task.get("command") != P5_SMOKE_COMMAND:
        problems.append("task.command")

    admission = receipt.get("admission")
    if not isinstance(admission, dict):
        problems.append("admission")
        admission = {}
    expected_admission = {
        "allowed": False,
        "consecutive_bad_samples": 3,
        "consecutive_good_samples": 0,
        "required_consecutive_good_samples": 3,
        "samples_observed": 3,
    }
    for key, expected in expected_admission.items():
        if admission.get(key) != expected:
            problems.append(f"admission.{key}")
    if admission.get("reason") != "admission requires the configured consecutive healthy samples":
        problems.append("admission.reason")

    decisions = receipt.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 3:
        problems.append("decisions")
        decisions = []
    memory_observations: list[float] = []
    memory_limits: list[float] = []
    cpu_observations: list[float] = []
    cpu_limits: list[float] = []
    projected_disk: list[float] = []
    expected_gate_names = {
        "required_telemetry",
        "receipt_prerequisites",
        "resource_measurement",
        "lane_count",
        "exclusive_lane",
        "one_heavy",
        "second_lane_kind",
        "unmanaged_heavy_process",
        "foreground_second_lane",
        "cpu_load",
        "cpu_utilization",
        "declared_cpu_cores",
        "memory_pressure",
        "candidate_memory_headroom",
        "declared_memory_budget",
        "swap",
        "thermal",
        "power",
        "forecasted_disk",
    }
    for index, decision in enumerate(decisions):
        prefix = f"decisions[{index}]"
        if not isinstance(decision, dict):
            problems.append(prefix)
            continue
        if decision.get("schema") != "mop-local-throttle-decision/v1":
            problems.append(f"{prefix}.schema")
        if decision.get("task_id") != P5_SMOKE_TASK_ID:
            problems.append(f"{prefix}.task_id")
        if decision.get("allowed") is not False:
            problems.append(f"{prefix}.allowed")
        if decision.get("active_lanes") != []:
            problems.append(f"{prefix}.active_lanes")
        if decision.get("denied_reasons") != [P5_SMOKE_CPU_REASON, P5_SMOKE_MEMORY_REASON]:
            problems.append(f"{prefix}.denied_reasons")

        raw_gates = decision.get("gates")
        if not isinstance(raw_gates, list) or not all(isinstance(gate, dict) for gate in raw_gates):
            problems.append(f"{prefix}.gates")
            continue
        gate_names = [str(gate.get("name")) for gate in raw_gates]
        if len(gate_names) != len(set(gate_names)) or set(gate_names) != expected_gate_names:
            problems.append(f"{prefix}.gate_names")
            continue
        gates = {str(gate["name"]): gate for gate in raw_gates}
        failing = {name for name, gate in gates.items() if gate.get("ok") is not True}
        if failing != {"cpu_load", "candidate_memory_headroom"}:
            problems.append(f"{prefix}.failing_gates")

        cpu = gates["cpu_load"]
        cpu_observed = cpu.get("observed")
        cpu_limit = cpu.get("limit")
        if (
            cpu.get("ok") is not False
            or cpu.get("reason") != P5_SMOKE_CPU_REASON
            or isinstance(cpu_observed, bool)
            or not isinstance(cpu_observed, (int, float))
            or not math.isfinite(float(cpu_observed))
            or isinstance(cpu_limit, bool)
            or not isinstance(cpu_limit, (int, float))
            or not math.isfinite(float(cpu_limit))
            or float(cpu_observed) <= float(cpu_limit)
        ):
            problems.append(f"{prefix}.cpu_load")
        else:
            cpu_observations.append(float(cpu_observed))
            cpu_limits.append(float(cpu_limit))

        memory = gates["candidate_memory_headroom"]
        observed = memory.get("observed")
        limit = memory.get("limit")
        if (
            memory.get("ok") is not False
            or memory.get("reason") != P5_SMOKE_MEMORY_REASON
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(float(limit))
            or float(observed) >= float(limit)
        ):
            problems.append(f"{prefix}.candidate_memory_headroom")
        else:
            memory_observations.append(float(observed))
            memory_limits.append(float(limit))

        power = gates["power"]
        if power.get("ok") is not True or power.get("observed") != "AC Power":
            problems.append(f"{prefix}.power")
        disk = gates["forecasted_disk"]
        disk_observed = disk.get("observed")
        disk_limit = disk.get("limit")
        projected = disk_observed.get("projected_free_gb") if isinstance(disk_observed, dict) else None
        if (
            disk.get("ok") is not True
            or isinstance(disk_limit, bool)
            or not isinstance(disk_limit, (int, float))
            or float(disk_limit) != 40.0
            or isinstance(projected, bool)
            or not isinstance(projected, (int, float))
            or not math.isfinite(float(projected))
            or float(projected) < float(disk_limit)
        ):
            problems.append(f"{prefix}.forecasted_disk")
        else:
            projected_disk.append(float(projected))

    if memory_limits and len(set(memory_limits)) != 1:
        problems.append("memory_limit_consistency")
    if cpu_limits and len(set(cpu_limits)) != 1:
        problems.append("cpu_limit_consistency")

    for field, relative in (
        ("policy", "configs/local_execution_throttle.yaml"),
        ("implementation", "src/mop/studio/local_throttle.py"),
    ):
        record = receipt.get(field)
        live_path = evidence_root / relative
        if not isinstance(record, dict):
            problems.append(field)
            continue
        declared_path = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(declared_path, str) or not declared_path.replace("\\", "/").endswith(relative):
            problems.append(f"{field}.path")
        if (
            not live_path.is_file()
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or sha256_path(live_path) != expected_hash
        ):
            problems.append(f"{field}.sha256")

    try:
        from mop.studio.local_throttle import aggregate_admission, evaluate_task, load_policy

        policy = load_policy(evidence_root / "configs" / "local_execution_throttle.yaml")
        live_task = policy.tasks[P5_SMOKE_TASK_ID]
        canonical_task = json.loads(json.dumps(asdict(live_task)))
        if receipt.get("task") != canonical_task:
            problems.append("task.live_policy_binding")
        telemetry_samples = receipt.get("telemetry_samples")
        if not isinstance(telemetry_samples, list) or len(telemetry_samples) != 3:
            problems.append("telemetry_samples")
        else:
            rebuilt_decisions: list[dict[str, Any]] = []
            for index, telemetry in enumerate(telemetry_samples):
                if not isinstance(telemetry, dict):
                    problems.append(f"telemetry_samples[{index}]")
                    continue
                rebuilt = evaluate_task(
                    live_task,
                    telemetry,
                    policy,
                    active=[],
                    evidence_root=evidence_root,
                )
                rebuilt_decisions.append(rebuilt)
                rebuilt_without_time = dict(rebuilt)
                rebuilt_without_time.pop("created_at", None)
                actual_without_time = dict(decisions[index]) if index < len(decisions) else {}
                actual_without_time.pop("created_at", None)
                if actual_without_time != rebuilt_without_time:
                    problems.append(f"decisions[{index}].canonical_rebuild")
            required_good = int(policy.monitor["admission_good_samples"])
            if receipt.get("admission") != aggregate_admission(rebuilt_decisions, required_good):
                problems.append("admission.canonical_rebuild")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        problems.append(f"canonical_rebuild:{exc}")

    if problems:
        raise ValueError("invalid P5 smoke admission refusal: " + ", ".join(dict.fromkeys(problems)))
    return {
        "state": "cpu-load-and-memory-admission-refusal",
        "run_id": run_id,
        "command_executed": False,
        "decision_count": len(decisions),
        "failed_gates": ["cpu_load", "candidate_memory_headroom"],
        "cpu_load_per_logical_cpu": cpu_observations,
        "maximum_cpu_load_per_logical_cpu": cpu_limits[0],
        "available_memory_gb": memory_observations,
        "required_memory_gb": memory_limits[0],
        "power_source": "AC Power",
        "minimum_projected_disk_gb": min(projected_disk),
    }


def resolved_gate_evidence_catalog() -> dict[str, dict[str, Any]]:
    """Resolve only code-declared gate records against the live local bytes.

    Role-bearing records must pin an expected source hash.  Merely pointing at
    an existing file or asserting ``current_hash_verified`` is insufficient.
    """
    catalog: dict[str, dict[str, Any]] = {}
    for identifier, spec in GATE_EVIDENCE_SPECS.items():
        record = dict(spec)
        source_ref = record.get("source_ref")
        path = (
            ROOT / str(source_ref).removeprefix("local:")
            if isinstance(source_ref, str) and source_ref.startswith("local:")
            else None
        )
        actual = sha256_path(path) if path is not None and path.is_file() else None
        expected = record.get("expected_source_sha256")
        has_roles = bool(record.get("gate_roles"))
        schema_validated_role = not has_roles
        if has_roles and path is not None and path.suffix == ".json" and path.is_file():
            try:
                payload = json.loads(path.read_text())
                embedded = payload.get("extended_compute_gate_record")
                if (
                    isinstance(embedded, dict)
                    and embedded.get("gate_roles") == record.get("gate_roles")
                    and embedded.get("record_type") == record.get("record_type")
                ):
                    for key in ("target_id", "repeat_index", "workload_id", "rung", "result_rung"):
                        if key in embedded:
                            record[key] = embedded[key]
                    schema_validated_role = True
            except Exception:
                schema_validated_role = False
        record["source_sha256"] = actual
        record["schema_validated_role"] = schema_validated_role
        record["current_hash_verified"] = (
            bool(actual)
            and schema_validated_role
            and (not has_roles or (isinstance(expected, str) and expected == actual))
        )
        catalog[identifier] = record
    return catalog


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


def source_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "id": f"local:{relative}",
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_path(path) if path.is_file() else None,
    }


def normalize_evidence_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def embedded_hash_audit(relative: str) -> dict[str, Any]:
    """Audit ordinary file SHA-256 claims embedded in a JSON artifact."""
    artifact = ROOT / relative
    result: dict[str, Any] = {"artifact": relative, "checks": 0, "mismatches": [], "missing": []}
    if not artifact.is_file():
        result["missing"].append({"path": relative, "reason": "audit artifact missing"})
        return result
    try:
        payload = json.loads(artifact.read_text())
    except Exception as exc:  # pragma: no cover - fail-closed diagnostic
        result["missing"].append({"path": relative, "reason": f"JSON parse failed: {exc}"})
        return result

    repository_roots: list[Path] = []
    repository_candidates = (
        ((payload.get("runtime_authority") or {}).get("repository") or {}).get("path"),
        (payload.get("repository_validation") or {}).get("local_path"),
    )
    for candidate in repository_candidates:
        if isinstance(candidate, str):
            candidate_path = Path(candidate)
            if candidate_path.is_dir():
                repository_roots.append(candidate_path)

    def walk(node: Any, pointer: str = "") -> None:
        if isinstance(node, dict):
            candidate = node.get("path")
            expected = node.get("sha256")
            if (
                isinstance(candidate, str)
                and isinstance(expected, str)
                and re.fullmatch(r"[0-9a-f]{64}", expected)
            ):
                raw_path = Path(candidate)
                possible_paths = (
                    [raw_path]
                    if raw_path.is_absolute()
                    else [ROOT / raw_path, *(root / raw_path for root in repository_roots)]
                )
                existing_paths = [path for path in possible_paths if path.is_file()]
                path = next(
                    (path for path in existing_paths if sha256_path(path) == expected),
                    existing_paths[0] if existing_paths else possible_paths[0],
                )
                result["checks"] += 1
                if not path.is_file():
                    result["missing"].append(
                        {"pointer": pointer, "path": candidate, "expected_sha256": expected}
                    )
                else:
                    actual = sha256_path(path)
                    if actual != expected:
                        try:
                            shown = str(path.relative_to(ROOT))
                        except ValueError:
                            shown = str(path)
                        result["mismatches"].append(
                            {
                                "pointer": pointer,
                                "path": shown,
                                "expected_sha256": expected,
                                "actual_sha256": actual,
                            }
                        )
            for key, value in node.items():
                walk(value, f"{pointer}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")

    walk(payload)
    # A stale source can repeat the same old path/hash in many rows.  Preserve
    # one machine-checkable record per distinct claim and expose occurrence
    # count separately rather than inflating the defect count.
    for key in ("mismatches", "missing"):
        unique: dict[str, dict[str, Any]] = {}
        for item in result[key]:
            identity = json.dumps({k: v for k, v in item.items() if k != "pointer"}, sort_keys=True)
            if identity not in unique:
                unique[identity] = {**item, "occurrences": 1}
            else:
                unique[identity]["occurrences"] += 1
        result[key] = list(unique.values())
    result["all_current"] = not result["mismatches"] and not result["missing"]
    return result


def hardware_requirement_gate() -> dict[str, Any]:
    return {
        "status": "not_passed",
        "named_scientific_target": None,
        "evidence_ids": [],
        "local_feasibility_attacks_complete": False,
        "repeated_boundary_failures": 0,
        "safe_headroom_fraction_target": 0.20,
        "boundary": {
            "nonfactorizable_resident_state": False,
            "scientifically_necessary_realtime_latency": False,
            "inseparable_synchronized_state_or_interaction": False,
        },
        "factorization_changes_estimand_proved": False,
        "next_rung_parity_pilot_passed": False,
        "smallest_sufficient_rung": None,
        "null_rule": None,
        "kill_rule": None,
    }


def hardware_benefit_gate() -> dict[str, Any]:
    return {
        "status": "not_passed",
        "named_elapsed_time_or_cost_target": None,
        "evidence_ids": [],
        "same_workload_and_data_order": False,
        "numerical_and_metric_parity_passed": False,
        "measured_comparative_benefit": False,
        "smallest_beneficial_rung": None,
        "stop_rule": None,
    }


def row(
    *,
    identifier: str,
    name: str,
    scope: str,
    category: int,
    blocker: str,
    basis: str,
    status: str,
    series: str | None = None,
    registry_resource_tier: str | None = None,
    registry_runtime_class: str | None = None,
    classified_stage: str = "next unresolved runnable stage at the 2026-07-10 snapshot",
    promotion_blocker: str | None = None,
    evidence_refs: list[str] | None = None,
    measured: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_rung = "L0" if category == 1 else ("L6" if category == 6 else None)
    return {
        "id": identifier,
        "name": name,
        "scope": scope,
        "series": series,
        "status": status,
        "primary_category": category,
        "primary_category_label": CATEGORY_LABELS[category],
        "primary_blocker": blocker,
        "classification_basis": basis,
        "classified_stage": classified_stage,
        "promotion_blocker": promotion_blocker,
        "registry_resource_tier": registry_resource_tier,
        "registry_runtime_class": registry_runtime_class,
        "required_rung": required_rung,
        "post_blocker_local_rung": "L0" if category in (2, 3) else None,
        "hardware_required": False,
        "hardware_requirement_gate": hardware_requirement_gate(),
        "hardware_benefit_gate": hardware_benefit_gate(),
        "evidence_refs": evidence_refs or ["local:registry/experiments.yaml"],
        "calculation_refs": [],
        "measured": measured or {"present": False},
        "estimated": {"present": False},
        "independent_unit": None,
        "dependence_structure": "must be preregistered before promotion testing",
        "minimum_model_data": "smallest registered or planned mechanics rung",
        "controls": "registry controls or matched controls named in the narrative plan",
        "local_feasibility_attack": "implementation/data/environment blocker must clear before any compute escalation",
        "promotion_rule": "no hardware promotion without the global requirement/benefit gate",
        "kill_rule": "kill or redesign on failed validity/control gate before scaling",
        "reusable_artifact": "hashed config, source snapshot, receipt, metrics, and split/provenance manifest",
    }


def registry_rows() -> tuple[list[dict[str, Any]], list[str]]:
    payload = yaml.safe_load((ROOT / "registry" / "experiments.yaml").read_text())
    experiments = payload["experiments"]
    exhaustion = json.loads((ROOT / "proof" / "PROJECT_EXPERIMENT_EXHAUSTION.json").read_text())
    p5_receipt_path = ROOT / P5_SMOKE_RECEIPT_PATH
    p5_refusal = _p5_smoke_refusal_summary(json.loads(p5_receipt_path.read_text()))
    ledger = {item["id"]: item for item in exhaustion["entries"]}
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for item in experiments:
        identifier = item["id"]
        ids.append(identifier)
        override = REGISTRY_OVERRIDES.get(identifier)
        if override is not None:
            category, blocker, basis = override
        elif item.get("series") == "F" and item.get("status") == "registry-only":
            if item.get("resource_tier") == "environment-needed":
                category = 6
                blocker = (
                    "satisfy the preregistered external environment, participant, sensor, or "
                    "physical-specimen gate; compute cannot substitute"
                )
            else:
                category = 2
                blocker = (
                    "implement the preregistered local experiment, difficulty-calibrate it, and "
                    "run its declared controls plus an independent verifier"
                )
            basis = "registry-preregistration-only"
        else:
            category, blocker, basis = (
                1,
                "none beyond executing or extending an already local implementation",
                "registry-plus-project-exhaustion-ledger",
            )
        ledger_entry = ledger.get(identifier)
        if ledger_entry is not None:
            measured = {
                "present": bool(ledger_entry.get("execution_verified")),
                "execution_verified_at_generation": bool(ledger_entry.get("execution_verified")),
                "ledger_classification": ledger_entry.get("classification"),
                "implementation_surface": ledger_entry.get("implementation_surface"),
                "evidence_scope": "generation-time; current-source mismatches are separately audited and stale paths are not treated as closed",
            }
            promotion_blocker = ledger_entry.get("remaining_scientific_blocker")
            refs = ["local:registry/experiments.yaml"]
            for evidence in ledger_entry.get("evidence", []):
                candidate = evidence.get("path")
                expected = evidence.get("sha256")
                if isinstance(candidate, str) and isinstance(expected, str):
                    path = normalize_evidence_path(candidate)
                    if path.is_file() and sha256_path(path) == expected:
                        if path.suffix == ".json":
                            try:
                                relative_for_audit = str(path.relative_to(ROOT))
                            except ValueError:
                                continue
                            if not embedded_hash_audit(relative_for_audit)["all_current"]:
                                continue
                        with contextlib.suppress(ValueError):
                            refs.append(f"local:{path.relative_to(ROOT)}")
            if identifier in {"mop_cm5_studio_rejuvenation", "mop_cm11_developmental_plasticity"}:
                refs.extend(
                    ["local:proof/LOCAL_FRONTIER_PREFLIGHTS.json", "local:scripts/frontier_localization.py"]
                )
            if identifier == "mop_cm7_min_objective_probe":
                refs.append("local:proof/CUSTOM_SUBSTRATE_PILOT.json")
                measured["present"] = True
                measured["evidence_scope"] = (
                    "five-seed, 1,000-update-per-arm teacher-independent programmatic-video pilot; "
                    "the bound independent verifier returned not-promoted after familywise correction"
                )
            if identifier == "mop_p5_context_capability":
                refs.append(f"local:{P5_SMOKE_RECEIPT_PATH}")
                measured["present"] = False
                measured["governor_admission"] = p5_refusal
                measured["evidence_scope"] = (
                    "implementation and governed local-admission refusal evidence; the "
                    "source-current P5 scientific sequence has not executed"
                )
            if identifier == "e5_curiosity":
                refs.append("local:proof/LOCAL_ACTION_ENVIRONMENT.json")
            if identifier == "mop_cm10_action_forward_model":
                refs.extend(
                    [
                        "local:proof/LOCAL_ACTION_ENVIRONMENT.json",
                        "local:proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
                        "local:configs/experiment/p7_action_world_model_preflight.yaml",
                        "local:src/mop/studies/action_world_model.py",
                        "local:scripts/p7_action_world_model_preflight.py",
                        "local:docs/P7_ACTION_WORLD_MODEL_AUDIT_2026_07.md",
                    ]
                )
            if identifier in {"e6_relational", "mop_dr14_corruption"}:
                refs.extend(
                    [
                        "local:proof/VJEPA21_VITB_LOAD.json",
                        "local:proof/VJEPA21_VITB_FORWARD.json",
                        "local:proof/VJEPA21_VITB_FORWARD_64F.json",
                        "local:proof/E6_VITB_DENSE_PREFLIGHT.json",
                        "local:configs/encoder/vjepa21_vitb.yaml",
                        "local:configs/experiment/e6_dense_cache.yaml",
                        "local:src/mop/substrate/vjepa21_official.py",
                        "local:src/mop/substrate/vjepa21_dense_tasks.py",
                        "local:scripts/vjepa21_dense_tasks.py",
                        "local:docs/VJEPA21_LOCAL_INTEGRATION.md",
                        "local:docs/E6_DENSE_RELATIONAL_CACHE.md",
                    ]
                )
            if identifier in {
                "mop_dr5_cross_substrate_consistency",
                "mop_at1_nuisance_grid",
                "mop_at2_mode_substrate_dep",
                "mop_pr2_plasticity_substrates",
            }:
                refs.extend(
                    [
                        "local:proof/CACHE_QUARANTINE_AUDIT.json",
                        "local:data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
                    ]
                )
            source_current = True
            for source in ledger_entry.get("source_evidence", []):
                candidate = source.get("path")
                expected = source.get("sha256")
                if not isinstance(candidate, str) or not isinstance(expected, str):
                    source_current = False
                    break
                path = normalize_evidence_path(candidate)
                if not path.is_file() or sha256_path(path) != expected:
                    source_current = False
                    break
            if identifier not in REGISTRY_OVERRIDES:
                direct_current = len(refs) > 1
                calibrated_exception = (
                    identifier == "mop_cm7_min_objective_probe"
                    and "local:proof/CUSTOM_SUBSTRATE_PILOT.json" in refs
                )
                if not (
                    (bool(ledger_entry.get("execution_verified")) and source_current and direct_current)
                    or calibrated_exception
                ):
                    raise ValueError(
                        f"unclassified registry row lacks a current direct local receipt: {identifier}"
                    )
        else:
            # The exhaustion ledger intentionally covers non-F rows only.
            if item.get("series") == "F" and item.get("status") == "registry-only":
                measured = {
                    "present": False,
                    "execution_verified_at_generation": False,
                    "evidence_scope": (
                        "preregistration contract only; no experiment execution or result receipt"
                    ),
                }
                promotion_blocker = blocker
                refs = ["local:registry/experiments.yaml"]
            else:
                measured = {
                    "present": identifier.startswith("f"),
                    "execution_verified_at_generation": None,
                    "evidence_scope": (
                        "implemented F-chain mechanics are covered by the current contract audit, "
                        "not the non-F exhaustion ledger"
                    ),
                }
                promotion_blocker = None
                refs = [
                    "local:registry/experiments.yaml",
                    "local:proof/FORM_SUBSTRATE/CONTRACT_AUDIT.json",
                    "local:proof/FORM_SUBSTRATE/LOCAL_RUN_SUMMARY.json",
                ]
        refs = list(dict.fromkeys(refs))
        rows.append(
            row(
                identifier=identifier,
                name=item["name"],
                scope="current_registry",
                category=category,
                blocker=blocker,
                basis=basis,
                status=item["status"],
                series=str(item.get("series")) if item.get("series") is not None else None,
                registry_resource_tier=item.get("resource_tier"),
                registry_runtime_class=item.get("runtime_class"),
                classified_stage=(
                    "current registry preregistration; no execution receipt"
                    if item.get("series") == "F" and item.get("status") == "registry-only"
                    else "current registered experiment; category 1 means executable local mechanics, not a promotion-grade scientific verdict"
                ),
                promotion_blocker=promotion_blocker,
                evidence_refs=refs,
                measured=measured,
            )
        )
    return rows, ids


def proposed_f_rows() -> tuple[list[dict[str, Any]], list[int]]:
    text = (ROOT / "FORM_SUBSTRATE_DEEP_EXPANSION_PLAN.md").read_text()
    found: dict[int, str] = {}
    for number, name in re.findall(r"^\| F(\d+) \|\s*([^|]+?)\s*\|", text, flags=re.MULTILINE):
        n = int(number)
        if 21 <= n <= 66:
            found[n] = name.strip()
    expected = set(range(21, 67))
    if set(found) != expected:
        raise ValueError(
            f"F21-F66 extraction mismatch: missing={sorted(expected - set(found))}, extra={sorted(set(found) - expected)}"
        )
    rows: list[dict[str, Any]] = []
    for n in sorted(found):
        name = found[n]
        category = F_PROPOSED_CATEGORY.get(n, 2)
        blocker = F_PROPOSED_BLOCKER.get(n)
        if blocker is None:
            blocker = next(value for family, value in F_FAMILY_BLOCKERS.items() if n in family)
        rows.append(
            row(
                identifier=f"f{n}",
                name=f"F{n} {name}",
                scope="proposed_F21_F66",
                category=category,
                blocker=blocker,
                basis="expansion-plan-plus-independent-repository-and-frontier-audits",
                status="proposed",
                series="F",
                classified_stage="first not-yet-completed rung in the expansion plan; later natural/bench promotion is recorded separately",
                promotion_blocker=(
                    "promotion-grade natural data, environment, or specimen evidence after local mechanics"
                    if category == 2
                    else blocker
                ),
                evidence_refs=["local:FORM_SUBSTRATE_DEEP_EXPANSION_PLAN.md"],
            )
        )
    return rows, sorted(found)


def thematic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, name, category, blocker in THEMATIC_FRONTIERS:
        promotion = None
        if identifier == "natural_object_event_causal_state":
            promotion = "causal identifiability requires declared interventions/side information; a new physical intervention surface is category 6"
        elif identifier == "full_system_density_accounting":
            promotion = "instrumented comparable wall energy requires a plug-level meter or equivalent category-6 measurement boundary"
        elif identifier in {"active_perception", "action_conditioned_world_models"}:
            promotion = "real sensor/robot or non-replayable latency validation is a later category-6 stage"
        evidence_refs = [
            "local:EXTENDED_COMPUTE_RESEARCH_PROMPT.md",
            "local:EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md",
        ]
        basis = "local-feasibility-attack-plus-primary-source-review"
        if identifier == "continual_million_event_learning":
            evidence_refs.extend(
                [
                    "local:proof/EXPANSION_WAVE0.json",
                    "local:proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json",
                    "local:proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json",
                    "local:configs/experiment/continual_million_event_preflight.yaml",
                    "local:configs/experiment/continual_million_event_rungs.yaml",
                    "local:configs/local_execution_throttle.yaml",
                    "local:src/mop/substrate/continual_stream.py",
                    "local:src/mop/studies/continual_million_event.py",
                    "local:scripts/continual_million_event_preflight.py",
                    "local:scripts/continual_million_event_rung.py",
                    "local:docs/P6_CONTINUAL_MILLION_EVENT_AUDIT_2026_07.md",
                ]
            )
            basis = "verified-continual-stream-mechanics-plus-progressive-local-execution-gate"
        elif identifier == "action_conditioned_world_models":
            evidence_refs.extend(
                [
                    "local:proof/EXPANSION_WAVE0.json",
                    "local:proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
                    "local:configs/experiment/p7_action_world_model_preflight.yaml",
                    "local:src/mop/studies/action_world_model.py",
                    "local:scripts/p7_action_world_model_preflight.py",
                    "local:docs/P7_ACTION_WORLD_MODEL_AUDIT_2026_07.md",
                ]
            )
            basis = "verified-programmatic-action-world-model-mechanics-plus-external-validity-gate"
        elif identifier in {"workspace_operational_self_model", "full_system_density_accounting"}:
            evidence_refs.extend(
                [
                    "local:proof/P9_ACCOUNTING_MECHANICS.json",
                    "local:proof/P9_CAUSAL_MONITORING_PREFLIGHT.json",
                    "local:configs/experiment/p9_causal_monitoring_preflight.yaml",
                    "local:src/mop/studies/p9_accounting.py",
                    "local:src/mop/studies/p9_causal_monitoring.py",
                    "local:scripts/p9_causal_monitoring_preflight.py",
                    "local:docs/P9_CAUSAL_MONITORING_PREFLIGHT.md",
                ]
            )
            basis = "verified-causal-monitoring-and-accounting-mechanics-plus-external-workload-gate"
        rows.append(
            row(
                identifier=f"frontier_{identifier}",
                name=name,
                scope="candidate_frontier",
                category=category,
                blocker=blocker,
                basis=basis,
                status="proposed",
                series="frontier",
                classified_stage="smallest scientifically meaningful falsification attack, before a scale campaign",
                promotion_blocker=promotion,
                evidence_refs=evidence_refs,
            )
        )
    return rows


def workstream_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, name, category, blocker in WORKSTREAMS:
        promotion = None
        if identifier in {"w2", "w3"}:
            promotion = "real sensor, robot, participant, or non-replayable environment validation is a later category-6 stage"
        rows.append(
            row(
                identifier=identifier,
                name=f"{identifier.upper()} {name}",
                scope="expansion_workstream",
                category=category,
                blocker=blocker,
                basis="source-derived-expansion-workstream-audit",
                status="proposed",
                series="W",
                classified_stage="first runnable workstream stage, with later promotion dependencies kept secondary",
                promotion_blocker=promotion,
                evidence_refs=["local:FORM_SUBSTRATE_DEEP_EXPANSION_PLAN.md"],
            )
        )
    return rows


def pillar_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, name, category in PILLARS:
        blocker = (
            "rights-cleared, appropriately aligned natural inputs and provenance"
            if category == 3
            else "bounded local operationalization, implementation, and matched controls"
        )
        rows.append(
            row(
                identifier=f"pillar_{section}",
                name=name,
                scope="research_pillar",
                category=category,
                blocker=blocker,
                basis="source-derived-research-pillar-audit",
                status="research_program",
                series="pillar",
                classified_stage="first experimental stage implied by dossier sections 7-25",
                evidence_refs=["local:FORM_SUBSTRATE_DEEP_RESEARCH_2026_07.md"],
            )
        )
    return rows


def load_cache_measurements() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in ("vitl",):
        path = ROOT / "data" / "cache" / f"vjepa2_{label}_local8_citable" / "run_receipt.json"
        receipt = json.loads(path.read_text())
        output[label] = {
            "clips": receipt["samples"],
            "host_device": receipt["host_device"],
            "frames_per_sample": receipt["frames_per_sample"],
            "resolution": receipt["resolution"],
            "claim_scope": receipt["claim_scope"],
            "seconds_per_clip": receipt["seconds_per_clip"],
            "encode_seconds": receipt["encode_seconds"],
            "max_rss_bytes": receipt["max_rss_bytes"],
            "evidence_path": str(path.relative_to(ROOT)),
            "kind": "measured",
        }
    return output


def _p6_dry_prerequisite_state(
    task: dict[str, Any], decisions: list[dict[str, Any]], *, evidence_root: Path = ROOT
) -> tuple[bool, str]:
    prerequisites = task.get("prerequisites")
    if not isinstance(prerequisites, list) or len(prerequisites) != 1:
        return False, "task-prerequisite-contract-drift"
    requirement = prerequisites[0]
    if not isinstance(requirement, dict):
        return False, "task-prerequisite-contract-drift"
    raw_fields = requirement.get("fields")
    try:
        fields = dict(raw_fields) if isinstance(raw_fields, (dict, list)) else {}
    except (TypeError, ValueError):
        fields = {}
    if (
        requirement.get("path") != P5_VERIFICATION_PATH
        or requirement.get("schema") != P5_VERIFICATION_SCHEMA
        or fields != P5_VERIFICATION_FIELDS
    ):
        return False, "task-prerequisite-contract-drift"

    gates: list[dict[str, Any]] = []
    for decision in decisions:
        matches = [
            gate
            for gate in decision.get("gates") or []
            if isinstance(gate, dict) and gate.get("name") == "receipt_prerequisites"
        ]
        if len(matches) != 1:
            return False, "receipt-gate-count-drift"
        gates.append(matches[0])
    if gates and all(gate.get("ok") is True for gate in gates):
        return True, "satisfied"
    if not gates or any(gate.get("ok") is True for gate in gates):
        return False, "mixed-prerequisite-state"
    if (evidence_root / P5_VERIFICATION_PATH).exists():
        return False, "present-p5-verifier-was-rejected"

    expected_reason = "P6 and other dependent tasks fail closed until immutable prior receipts pass"
    for decision, gate in zip(decisions, gates, strict=True):
        observed = gate.get("observed")
        denied_reasons = decision.get("denied_reasons")
        failing_gate_reasons = {
            candidate.get("reason")
            for candidate in decision.get("gates") or []
            if isinstance(candidate, dict)
            and candidate.get("ok") is False
            and isinstance(candidate.get("reason"), str)
        }
        if (
            not isinstance(denied_reasons, list)
            or expected_reason not in denied_reasons
            or len(denied_reasons) != len(set(denied_reasons))
            or set(denied_reasons) != failing_gate_reasons
            or gate.get("reason") != expected_reason
            or not isinstance(observed, list)
            or len(observed) != 1
        ):
            return False, "missing-prerequisite-refusal-drift"
        row = observed[0]
        if not isinstance(row, dict) or (
            row.get("path") != P5_VERIFICATION_PATH
            or row.get("all_ok") is not False
            or row.get("schema") is not None
            or row.get("sha256") is not None
            or row.get("governor_provenance") is not None
            or "receipt is missing" not in (row.get("problems") or [])
        ):
            return False, "missing-prerequisite-refusal-drift"
    return True, "missing-sealed-p5-verifier"


def authoritative_receipt_checks() -> dict[str, Any]:
    """Run schema-aware checks for evidence whose paths have scoped semantics."""
    errors: list[str] = []
    checks: dict[str, Any] = {}

    try:
        from mop.substrate.cache_tools import validate_cache

        cache_results: dict[str, Any] = {}
        for label in ("vitl",):
            relative = f"data/cache/vjepa2_{label}_local8_citable"
            problems = list(validate_cache(ROOT / relative, citable=True))
            cache_results[label] = {
                "path": relative,
                "citable": True,
                "problems": problems,
                "all_ok": not problems,
            }
            errors.extend(f"{relative}: {problem}" for problem in problems)
        checks["citable_caches"] = cache_results
    except Exception as exc:
        errors.append(f"cache schema validation failed to execute: {exc}")
        checks["citable_caches"] = {"all_ok": False, "error": str(exc)}

    for relative in ("proof/CUSTOM_SUBSTRATE_PILOT.json", "proof/LOCAL_ACTION_ENVIRONMENT.json"):
        audit = embedded_hash_audit(relative)
        checks[relative] = audit
        if not audit["all_current"]:
            errors.append(f"{relative}: embedded root-relative evidence is stale")

    try:
        p5_receipt = json.loads((ROOT / P5_SMOKE_RECEIPT_PATH).read_text())
        p5_refusal = _p5_smoke_refusal_summary(p5_receipt)
        checks["p5_smoke_admission_refusal"] = {"all_ok": True, **p5_refusal}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        checks["p5_smoke_admission_refusal"] = {"all_ok": False, "error": str(exc)}
        errors.append(f"P5 smoke admission refusal failed: {exc}")

    form_contract = json.loads((ROOT / "proof/FORM_SUBSTRATE/CONTRACT_AUDIT.json").read_text())
    form_runs = json.loads((ROOT / "proof/FORM_SUBSTRATE/LOCAL_RUN_SUMMARY.json").read_text())
    checks["form_substrate"] = {
        "contract_all_ok": form_contract.get("all_ok") is True,
        "contract_total": form_contract.get("summary", {}).get("total"),
        "local_runs_all_ok": form_runs.get("all_ok") is True,
        "local_runs_completed": form_runs.get("summary", {}).get("completed"),
    }
    if not all(
        (
            checks["form_substrate"]["contract_all_ok"],
            checks["form_substrate"]["local_runs_all_ok"],
            checks["form_substrate"]["local_runs_completed"] == 18,
        )
    ):
        errors.append("Form-substrate current contract/local-run check failed")

    quarantine = json.loads((ROOT / "proof/CACHE_QUARANTINE_AUDIT.json").read_text())
    checks["cache_quarantine"] = {
        "all_ok": quarantine.get("all_ok") is True,
        "examined": quarantine.get("summary", {}).get("examined"),
        "uncitable": quarantine.get("summary", {}).get("uncitable"),
    }
    if not checks["cache_quarantine"]["all_ok"]:
        errors.append("cache quarantine audit is not all_ok")

    runtime_paths = {
        "load": ROOT / "proof/VJEPA21_VITB_LOAD.json",
        "forward_8f": ROOT / "proof/VJEPA21_VITB_FORWARD.json",
        "forward_64f": ROOT / "proof/VJEPA21_VITB_FORWARD_64F.json",
    }
    runtime_receipts = {name: json.loads(path.read_text()) for name, path in runtime_paths.items()}
    runtime_hashes = {
        str((receipt.get("authority") or {}).get("checkpoint_sha256") or "")
        for receipt in runtime_receipts.values()
    }
    load_receipt = runtime_receipts["load"]
    forward_receipts = [runtime_receipts["forward_8f"], runtime_receipts["forward_64f"]]
    runtime_ok = bool(
        len(runtime_hashes) == 1
        and all(len(value) == 64 for value in runtime_hashes)
        and load_receipt.get("status") == "passed"
        and (load_receipt.get("child") or {}).get("strict_load") is True
        and (load_receipt.get("child") or {}).get("parameters") == 86_833_152
        and (load_receipt.get("child") or {}).get("trainable_parameters") == 0
        and all(
            receipt.get("status") == "passed"
            and receipt.get("hardware_limit_reached") is False
            and (receipt.get("child") or {}).get("output_finite") is True
            and (receipt.get("child") or {}).get("shape_matches") is True
            and (receipt.get("claim_boundary") or {}).get("vitb_runtime_evidence_gate_passed") is True
            for receipt in forward_receipts
        )
    )
    checks["vjepa21_vitb_runtime"] = {
        "all_ok": runtime_ok,
        "checkpoint_sha256": next(iter(runtime_hashes), None),
        "parameters": (load_receipt.get("child") or {}).get("parameters"),
        "forward_frames": [(receipt.get("probe") or {}).get("frames") for receipt in forward_receipts],
        "max_process_tree_rss_bytes": max(
            int(receipt.get("max_process_tree_rss_bytes") or 0) for receipt in runtime_receipts.values()
        ),
    }
    if not runtime_ok:
        errors.append("V-JEPA 2.1 ViT-B local load/forward evidence failed")

    e6_preflight = json.loads((ROOT / "proof/E6_VITB_DENSE_PREFLIGHT.json").read_text())
    e6_gates = e6_preflight.get("gates") or {}
    e6_integration_ok = bool(
        e6_preflight.get("schema") == "mop-vjepa21-dense-task-preflight/v1"
        and e6_preflight.get("all_ok") is True
        and e6_preflight.get("mode") == "no-heavy-preflight"
        and e6_preflight.get("model_constructed") is False
        and e6_preflight.get("checkpoint_tensor_bytes_read") is False
        and e6_preflight.get("forward_executed") is False
        and e6_gates.get("implementation_ready") is True
        and e6_gates.get("input_manifest_ready") is False
        and e6_gates.get("encode_allowed_now") is False
        and e6_preflight.get("scientific_promotion") is False
        and (e6_preflight.get("registration") or {}).get("all_ok") is True
    )
    checks["e6_dense_integration"] = {
        "all_ok": e6_integration_ok,
        "mode": e6_preflight.get("mode"),
        "implementation_ready": e6_gates.get("implementation_ready"),
        "input_manifest_ready": e6_gates.get("input_manifest_ready"),
        "model_constructed": e6_preflight.get("model_constructed"),
        "scientific_promotion": e6_preflight.get("scientific_promotion"),
        "remaining_gates": e6_preflight.get("remaining_gates"),
    }
    if not e6_integration_ok:
        errors.append("E6/DR14 dense task integration preflight failed")

    wave0 = json.loads((ROOT / "proof/EXPANSION_WAVE0.json").read_text())
    wave0_verifier = wave0.get("independent_verifier") or {}
    wave0_ok = bool(
        wave0.get("schema") == "mop-expansion-wave0/v1"
        and wave0.get("status") == "mechanics-pass"
        and wave0.get("all_sentinels_pass") is True
        and len(wave0.get("shared_units") or []) == 3
        and wave0_verifier.get("verified") is True
        and wave0_verifier.get("all_mutations_rejected") is True
        and (wave0_verifier.get("checks") or {}).get("metric_count") == 72
        and (wave0_verifier.get("checks") or {}).get("unit_count") == 3
    )
    checks["expansion_wave0"] = {
        "all_ok": wave0_ok,
        "status": wave0.get("status"),
        "shared_units": len(wave0.get("shared_units") or []),
        "metric_count": (wave0_verifier.get("checks") or {}).get("metric_count"),
        "all_mutations_rejected": wave0_verifier.get("all_mutations_rejected"),
        "claim_scope": wave0.get("claim_scope"),
    }
    if not wave0_ok:
        errors.append("Wave E0 shared-mechanics receipt failed")

    p6 = json.loads((ROOT / "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json").read_text())
    p6_checks = p6.get("checks") or {}
    p6_gate = p6.get("remaining_full_run_gate") or {}
    p6_ok = bool(
        p6.get("schema") == "mop-continual-million-event-preflight/v1"
        and p6.get("status") == "mechanics-pass"
        and p6.get("all_mechanics_ok") is True
        and p6.get("no_heavy_preflight") is True
        and all(value is True for value in p6_checks.values())
        and p6_gate.get("status") == "not-run"
        and p6_gate.get("hardware_boundary_earned") is False
        and p6_gate.get("progressive_rungs") == [10_000, 100_000, 1_000_000]
        and p6_gate.get("minimum_independent_seeds") == 5
    )
    checks["continual_million_event_preflight"] = {
        "all_ok": p6_ok,
        "status": p6.get("status"),
        "no_heavy_preflight": p6.get("no_heavy_preflight"),
        "progressive_rungs": p6_gate.get("progressive_rungs"),
        "minimum_independent_seeds": p6_gate.get("minimum_independent_seeds"),
        "hardware_boundary_earned": p6_gate.get("hardware_boundary_earned"),
        "claim_scope": p6.get("claim_scope"),
    }
    if not p6_ok:
        errors.append("P6 continual-stream mechanics receipt failed")

    p7 = json.loads((ROOT / "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json").read_text())
    p7_units = p7.get("units") or []
    p7_resource = p7.get("resource_observation") or {}
    p7_boundary = p7.get("claim_boundary") or {}
    expected_p7_arms = {
        "reactive_rendered",
        "model_free_recurrent",
        "compact_latent_transition",
        "object_centered_transition",
        "oracle_state",
        "action_blind",
        "action_shuffled",
        "matched_depth_reactive",
    }
    p7_ok = bool(
        p7.get("schema") == "mop-p7-action-world-model-preflight/v1"
        and p7.get("status") == "mechanics-pass"
        and p7.get("all_mechanics_ok") is True
        and p7_boundary.get("mechanics_only") is True
        and p7_boundary.get("scientific_promotion_allowed") is False
        and p7_boundary.get("natural_data") is False
        and p7_boundary.get("sentience_or_cognition_claim") is False
        and len(p7_units) == 3
        and len({(unit.get("independent_unit") or {}).get("seed") for unit in p7_units}) == 3
        and all(
            unit.get("all_mechanics_ok") is True
            and set((unit.get("arms") or {}).keys()) == expected_p7_arms
            and (unit.get("mutation_suite") or {}).get("all_rejected") is True
            and (unit.get("dataset_verification") or {}).get("verified") is True
            and (unit.get("equal_core_compute") or {}).get("matched") is True
            for unit in p7_units
        )
        and p7_resource.get("device") == "cpu"
        and p7_resource.get("torch_threads") == 1
        and p7_resource.get("accelerator_required") is False
        and p7_resource.get("model_weights_loaded") is False
        and p7_resource.get("model_downloads_performed") is False
        and p7_resource.get("command_executed_heavy_work") is False
    )
    checks["p7_action_world_model_preflight"] = {
        "all_ok": p7_ok,
        "status": p7.get("status"),
        "independent_units": len(p7_units),
        "arms_per_unit": sorted({len(unit.get("arms") or {}) for unit in p7_units}),
        "all_mutations_rejected": all(
            (unit.get("mutation_suite") or {}).get("all_rejected") is True for unit in p7_units
        ),
        "scientific_promotion_allowed": p7_boundary.get("scientific_promotion_allowed"),
        "remaining_external_validity_gate": p7_boundary.get("remaining_external_validity_gate"),
        "elapsed_seconds": p7_resource.get("elapsed_seconds"),
        "max_rss_bytes": p7_resource.get("max_rss_bytes"),
        "claim_scope": p7.get("claim_scope"),
    }
    if not p7_ok:
        errors.append("P7 action-conditioned world-model mechanics receipt failed")

    p9 = json.loads((ROOT / "proof/P9_CAUSAL_MONITORING_PREFLIGHT.json").read_text())
    p9_units = p9.get("units") or []
    p9_dataset = p9.get("dataset") or {}
    p9_budget = p9_dataset.get("budget_contract") or {}
    p9_mutations = p9.get("mutation_suite") or {}
    p9_resume = p9.get("resume") or {}
    p9_resource = p9.get("resource_observation") or {}
    p9_boundary = p9.get("claim_boundary") or {}
    p9_aggregate = p9.get("causal_vs_correlational_aggregate") or {}
    p9_ok = bool(
        p9.get("schema") == "mop-p9-causal-monitoring-preflight/v1"
        and p9.get("status") == "mechanics-pass"
        and p9.get("all_mechanics_ok") is True
        and p9_boundary.get("mechanics_only") is True
        and p9_boundary.get("scientific_promotion_allowed") is False
        and p9_boundary.get("natural_workloads") is False
        and p9_boundary.get("energy_measured") is False
        and p9_boundary.get("cognition_or_sentience_claim") is False
        and len(p9_units) == 5
        and all(unit.get("all_mechanics_ok") is True for unit in p9_units)
        and p9_budget.get("independent_units") == 5
        and p9_budget.get("total_lineages") == 260
        and p9_budget.get("total_branches") == 1300
        and (p9_dataset.get("verification") or {}).get("verified") is True
        and p9_mutations.get("count") == 8
        and p9_mutations.get("rejected") == 8
        and p9_mutations.get("all_rejected") is True
        and p9_resume.get("interrupted_after_chunks") == 4
        and p9_resume.get("completed_chunks") == 15
        and p9_resume.get("exact") is True
        and p9_resume.get("corrupt_checkpoint_rejected") is True
        and all(
            (p9_aggregate.get(metric) or {}).get("positive_units") == 5
            for metric in (
                "brier_improvement",
                "roc_auc_delta",
                "intervention_sign_agreement_delta",
                "controller_utility_delta",
            )
        )
        and p9_resource.get("device") == "cpu"
        and p9_resource.get("cpu_threads") == 1
        and p9_resource.get("accelerator_required") is False
        and p9_resource.get("model_weights_loaded") is False
        and p9_resource.get("model_downloads_performed") is False
        and p9_resource.get("external_data_loaded") is False
        and p9_resource.get("command_executed_heavy_work") is False
        and ((p9_resource.get("workload_accounting") or {}).get("energy") or {}).get("measured") is False
    )
    checks["p9_causal_monitoring_preflight"] = {
        "all_ok": p9_ok,
        "status": p9.get("status"),
        "independent_units": len(p9_units),
        "total_lineages": p9_budget.get("total_lineages"),
        "total_branches": p9_budget.get("total_branches"),
        "mutations_rejected": p9_mutations.get("rejected"),
        "interrupted_resume_exact": p9_resume.get("exact"),
        "scientific_promotion_allowed": p9_boundary.get("scientific_promotion_allowed"),
        "energy_measured": p9_boundary.get("energy_measured"),
        "remaining_evidence_gate": p9_boundary.get("remaining_evidence_gate"),
        "elapsed_seconds": p9_resource.get("elapsed_seconds"),
        "max_rss_bytes": p9_resource.get("max_rss_bytes"),
        "claim_scope": p9.get("claim_scope"),
    }
    if not p9_ok:
        errors.append("P9 causal-monitoring mechanics receipt failed")

    p6_dry = json.loads((ROOT / "proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json").read_text())
    p6_dry_task = p6_dry.get("task") or {}
    p6_dry_decisions = p6_dry.get("decisions") or []
    p6_dry_allowed = [decision.get("allowed") is True for decision in p6_dry_decisions]
    p6_dry_prerequisite_ok, p6_dry_prerequisite_state = _p6_dry_prerequisite_state(
        p6_dry_task, p6_dry_decisions
    )
    p6_dry_exclusive_consistent = all(
        any(
            gate.get("name") == "exclusive_lane"
            and gate.get("ok") is (len(decision.get("active_lanes") or []) == 0)
            for gate in decision.get("gates") or []
        )
        for decision in p6_dry_decisions
    )
    p6_dry_ok = bool(
        p6_dry.get("schema") == "mop-local-throttle-receipt/v1"
        and p6_dry.get("mode") in {"dry-run", "run-dry-run"}
        and p6_dry.get("command_executed") is False
        and p6_dry_task.get("task_id") == "p6_10k_resource_probe_cpu"
        and p6_dry_task.get("resource_probe") is True
        and p6_dry_task.get("requires_empty_lanes") is True
        and p6_dry_task.get("estimated_unified_memory_gb") is None
        and p6_dry_task.get("wall_minutes") == 300
        and p6_dry_task.get("depends_on") == []
        and len(p6_dry_decisions) == 3
        and (p6_dry.get("admission") or {}).get("allowed") is all(p6_dry_allowed)
        and p6_dry_exclusive_consistent
        and p6_dry_prerequisite_ok
        and all(
            any(
                gate.get("name") == "resource_measurement" and gate.get("ok") is True
                for gate in decision.get("gates") or []
            )
            and any(
                gate.get("name") == "forecasted_disk" and gate.get("ok") is True
                for gate in decision.get("gates") or []
            )
            for decision in p6_dry_decisions
        )
    )
    checks["p6_10k_scheduler_dry_run"] = {
        "all_ok": p6_dry_ok,
        "command_executed": p6_dry.get("command_executed"),
        "admission_allowed": (p6_dry.get("admission") or {}).get("allowed"),
        "task_id": p6_dry_task.get("task_id"),
        "resource_probe": p6_dry_task.get("resource_probe"),
        "estimated_unified_memory_gb": p6_dry_task.get("estimated_unified_memory_gb"),
        "decision_count": len(p6_dry_decisions),
        "exclusive_lane_state_consistent": p6_dry_exclusive_consistent,
        "prerequisite_state": p6_dry_prerequisite_state,
    }
    if not p6_dry_ok:
        errors.append("P6 10k scheduler dry-run receipt failed")

    try:
        from mop.studio.sanpo_real_intake import verify_existing_intake

        sanpo_live = verify_existing_intake(ROOT / "data/raw/sanpo_real_smoke_v0")
        sanpo_receipt = json.loads((ROOT / "proof/SANPO_REAL_SMOKE_VERIFICATION.json").read_text())
        sanpo_ok = all(
            (
                sanpo_live.get("all_ok") is True,
                sanpo_live.get("official_files_verified") == 94,
                sanpo_live.get("content_set_sha256") == sanpo_receipt.get("content_set_sha256"),
                sanpo_receipt.get("all_ok") is True,
                sanpo_receipt.get("claim_boundary", {}).get("scientific_promotion") is False,
            )
        )
        checks["sanpo_real_smoke"] = {
            "all_ok": sanpo_ok,
            "official_files_verified": sanpo_live.get("official_files_verified"),
            "content_set_sha256": sanpo_live.get("content_set_sha256"),
            "claim_status": sanpo_receipt.get("claim_boundary", {}).get("status"),
        }
        if not sanpo_ok:
            errors.append("SANPO schema-aware re-verification failed")
    except Exception as exc:
        errors.append(f"SANPO schema-aware re-verification failed to execute: {exc}")
        checks["sanpo_real_smoke"] = {"all_ok": False, "error": str(exc)}

    return {"all_ok": not errors, "errors": errors, "checks": checks}


def gate_errors(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence_catalog = resolved_gate_evidence_catalog()
    identifier = item["id"]
    category = item["primary_category"]
    requirement = item["hardware_requirement_gate"]
    benefit = item["hardware_benefit_gate"]

    if item["hardware_required"] != (category == 9):
        errors.append(f"{identifier}: hardware_required must be true iff category is 9")
    if category == 8:
        required = [
            benefit["status"] == "passed",
            bool(benefit["named_elapsed_time_or_cost_target"]),
            bool(benefit["evidence_ids"]),
            benefit["same_workload_and_data_order"],
            benefit["numerical_and_metric_parity_passed"],
            benefit["measured_comparative_benefit"],
            benefit["smallest_beneficial_rung"] in {"L2", "L3", "L4", "L5"},
            bool(benefit["stop_rule"]),
        ]
        if not all(required):
            errors.append(f"{identifier}: category 8 comparative-benefit gate incomplete")
        missing = set(benefit["evidence_ids"]) - set(evidence_catalog)
        if missing:
            errors.append(f"{identifier}: category 8 unresolved evidence ids {sorted(missing)}")
        records = [evidence_catalog[key] for key in benefit["evidence_ids"] if key in evidence_catalog]
        comparative = [
            record for record in records if "comparative_workload_measurement" in record.get("gate_roles", [])
        ]
        parity = [record for record in records if "parity_pilot" in record.get("gate_roles", [])]
        benefit_calcs = [
            record for record in records if "comparative_benefit_calculation" in record.get("gate_roles", [])
        ]
        workload_ids = {record.get("workload_id") for record in comparative}
        compared_rungs = {record.get("rung") for record in comparative}
        workload_id = next(iter(workload_ids)) if len(workload_ids) == 1 else None
        typed_gate = (
            len(comparative) >= 2
            and len({record.get("source_ref") for record in comparative}) >= 2
            and len({record.get("source_sha256") for record in comparative}) >= 2
            and len(workload_ids) == 1
            and None not in workload_ids
            and bool(compared_rungs & {"L0", "L1"})
            and bool(compared_rungs & {"L2", "L3", "L4", "L5"})
            and benefit["smallest_beneficial_rung"] in compared_rungs
            and any(
                record.get("workload_id") == workload_id and record.get("current_hash_verified")
                for record in parity
            )
            and any(
                record.get("workload_id") == workload_id
                and record.get("result_rung") == benefit["smallest_beneficial_rung"]
                and record.get("current_hash_verified")
                for record in benefit_calcs
            )
            and all(record.get("current_hash_verified") for record in comparative)
            and all(record.get("schema_validated_role") for record in comparative + parity + benefit_calcs)
        )
        if not typed_gate:
            errors.append(
                f"{identifier}: category 8 lacks typed two-rung same-workload, parity, and comparative-benefit records"
            )
    elif benefit["status"] == "passed":
        errors.append(f"{identifier}: passed benefit gate requires category 8")

    if category == 9:
        boundary = requirement["boundary"]
        required = [
            requirement["status"] == "passed",
            bool(requirement["named_scientific_target"]),
            bool(requirement["evidence_ids"]),
            requirement["local_feasibility_attacks_complete"],
            requirement["repeated_boundary_failures"] >= 3,
            any(boundary.values()),
            requirement["factorization_changes_estimand_proved"],
            requirement["next_rung_parity_pilot_passed"],
            requirement["smallest_sufficient_rung"] in {"L2", "L3", "L4", "L5"},
            bool(requirement["null_rule"]),
            bool(requirement["kill_rule"]),
        ]
        if not all(required):
            errors.append(f"{identifier}: category 9 requirement gate incomplete")
        missing = set(requirement["evidence_ids"]) - set(evidence_catalog)
        if missing:
            errors.append(f"{identifier}: category 9 unresolved evidence ids {sorted(missing)}")
        records = [evidence_catalog[key] for key in requirement["evidence_ids"] if key in evidence_catalog]
        failures = [record for record in records if "boundary_failure" in record.get("gate_roles", [])]
        parity = [record for record in records if "parity_pilot" in record.get("gate_roles", [])]
        rung_calcs = [
            record for record in records if "smallest_sufficient_rung" in record.get("gate_roles", [])
        ]
        target_ids = {record.get("target_id") for record in failures}
        repeat_ids = {record.get("repeat_index") for record in failures}
        failure_sources = {record.get("source_ref") for record in failures}
        parity_sources = {record.get("source_ref") for record in parity}
        named_target = requirement["named_scientific_target"]
        typed_gate = (
            len(failures) >= 3
            and len({record.get("source_ref") for record in failures}) >= 3
            and len({record.get("source_sha256") for record in failures}) >= 3
            and len(target_ids) == 1
            and target_ids == {named_target}
            and len(repeat_ids) >= 3
            and None not in repeat_ids
            and all(record.get("current_hash_verified") for record in failures)
            and any(
                record.get("target_id") == named_target
                and record.get("current_hash_verified")
                and record.get("source_ref") not in failure_sources
                for record in parity
            )
            and any(
                record.get("target_id") == named_target
                and record.get("result_rung") == requirement["smallest_sufficient_rung"]
                and record.get("current_hash_verified")
                and record.get("source_ref") not in failure_sources | parity_sources
                for record in rung_calcs
            )
            and all(record.get("schema_validated_role") for record in failures + parity + rung_calcs)
        )
        if not typed_gate:
            errors.append(
                f"{identifier}: category 9 lacks three typed same-target boundary failures, parity receipt, and rung derivation"
            )
        if item["required_rung"] != requirement["smallest_sufficient_rung"]:
            errors.append(
                f"{identifier}: category 9 required_rung must derive from the passed requirement gate"
            )
    elif requirement["status"] == "passed":
        errors.append(f"{identifier}: passed requirement gate requires category 9")
    return errors


def validation(
    rows: list[dict[str, Any]],
    registry_ids: list[str],
    proposed_f: list[int],
    source_ids: set[str],
) -> dict[str, Any]:
    ids = [item["id"] for item in rows]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("row ids are not unique")
    registry_matrix = [item["id"] for item in rows if item["scope"] == "current_registry"]
    if registry_matrix != registry_ids:
        errors.append("current registry coverage/order mismatch")
    registry_id_sha256 = hashlib.sha256(("\n".join(registry_ids) + "\n").encode()).hexdigest()
    if len(registry_ids) != EXPECTED_REGISTRY_COUNT or registry_id_sha256 != EXPECTED_REGISTRY_ID_SHA256:
        errors.append(
            f"registry snapshot identity drift: count={len(registry_ids)}, id_sha256={registry_id_sha256}"
        )
    if proposed_f != list(range(21, 67)):
        errors.append("F21-F66 coverage mismatch")
    workstream_ids = [item["id"] for item in rows if item["scope"] == "expansion_workstream"]
    if workstream_ids != [item[0] for item in WORKSTREAMS]:
        errors.append("W0-W11 workstream coverage/order mismatch")
    expansion_text = (ROOT / "FORM_SUBSTRATE_DEEP_EXPANSION_PLAN.md").read_text()
    source_workstreams = [
        (match.group(1).lower(), match.group(2).strip())
        for match in re.finditer(r"^### (W\d+)\.\s+(.+)$", expansion_text, flags=re.MULTILINE)
    ]
    declared_workstreams = [(item[0], item[1]) for item in WORKSTREAMS]
    if source_workstreams != declared_workstreams:
        errors.append("source-derived W0-W11 heading inventory drift")
    pillar_ids = [item["id"] for item in rows if item["scope"] == "research_pillar"]
    if pillar_ids != [f"pillar_{item[0]}" for item in PILLARS]:
        errors.append("research pillar coverage/order mismatch")
    dossier_text = (ROOT / "FORM_SUBSTRATE_DEEP_RESEARCH_2026_07.md").read_text()
    source_pillars = [
        (int(match.group(1)), match.group(2).strip())
        for match in re.finditer(r"^## (\d+)\.\s+(.+)$", dossier_text, flags=re.MULTILINE)
        if 7 <= int(match.group(1)) <= 25
    ]
    declared_pillars = [(item[0], item[1]) for item in PILLARS]
    if source_pillars != declared_pillars:
        errors.append("source-derived dossier pillar heading inventory drift")
    prompt_text = (ROOT / "EXTENDED_COMPUTE_RESEARCH_PROMPT.md").read_text()
    expected_prompt_sections = (
        "Read first",
        "Current corrections to preserve",
        "Research question",
        "Required per-survivor audit",
        "Mandatory local feasibility attack",
        "Candidate necessity classes",
        "Rung calculation",
        "Adversarial questions",
        "Required deliverables",
        "Final decision language",
    )
    prompt_sections = tuple(
        match.group(1).strip() for match in re.finditer(r"^## (.+)$", prompt_text, flags=re.MULTILINE)
    )
    if prompt_sections != expected_prompt_sections:
        errors.append("survivor-only extended-compute prompt section inventory drift")
    if not re.search(r"no measured hardware\s+blocker", prompt_text):
        errors.append("survivor-only prompt no longer preserves the measured-boundary correction")
    valid_gate_evidence = resolved_gate_evidence_catalog()
    for evidence_id, record in valid_gate_evidence.items():
        if record.get("source_ref") not in source_ids:
            errors.append(
                f"gate evidence {evidence_id}: unresolved current source ref {record.get('source_ref')}"
            )
        if not isinstance(record.get("gate_roles"), list):
            errors.append(f"gate evidence {evidence_id}: gate_roles must be a list")
    for item in rows:
        category = item["primary_category"]
        if category not in CATEGORY_LABELS:
            errors.append(f"{item['id']}: invalid category {category}")
        missing_sources = set(item["evidence_refs"]) - source_ids
        if missing_sources:
            errors.append(f"{item['id']}: unresolved local evidence refs {sorted(missing_sources)}")
        if item["required_rung"] is not None and item["required_rung"] not in RUNG_DEFINITIONS:
            errors.append(f"{item['id']}: undefined required rung {item['required_rung']}")
        if category == 1 and item["required_rung"] != "L0":
            errors.append(f"{item['id']}: category 1 must use L0")
        if category in (2, 3, 7, 8) and item["required_rung"] is not None:
            errors.append(f"{item['id']}: category {category} cannot have an ungated required rung")
        if category == 6 and item["required_rung"] != "L6":
            errors.append(f"{item['id']}: category 6 must use L6")
        errors.extend(gate_errors(item))

    # Adversarial negative tests prove that fabricated IDs and incomplete
    # category-9/category-8 gates do not pass merely because row() defaults are
    # conservative.
    probe = json.loads(json.dumps(rows[0]))
    probe["primary_category"] = 9
    probe["hardware_required"] = True
    probe["hardware_requirement_gate"]["status"] = "passed"
    probe["hardware_requirement_gate"]["evidence_ids"] = ["does-not-exist"]
    probe_errors = gate_errors(probe)
    negative_bogus_id_rejected = any("unresolved evidence ids" in error for error in probe_errors)
    probe2 = json.loads(json.dumps(rows[0]))
    probe2["primary_category"] = 9
    probe2["hardware_required"] = True
    negative_category9_mismatch_rejected = any(
        "requirement gate incomplete" in error for error in gate_errors(probe2)
    )
    probe3 = json.loads(json.dumps(rows[0]))
    probe3["primary_category"] = 8
    negative_incomplete_category8_rejected = any(
        "comparative-benefit gate incomplete" in error for error in gate_errors(probe3)
    )
    probe4 = json.loads(json.dumps(rows[0]))
    probe4["primary_category"] = 9
    probe4["hardware_required"] = True
    probe4["required_rung"] = "L3"
    requirement4 = probe4["hardware_requirement_gate"]
    requirement4.update(
        {
            "status": "passed",
            "named_scientific_target": "fake-target",
            "evidence_ids": ["c_energy_scope"],
            "local_feasibility_attacks_complete": True,
            "repeated_boundary_failures": 3,
            "factorization_changes_estimand_proved": True,
            "next_rung_parity_pilot_passed": True,
            "smallest_sufficient_rung": "L3",
            "null_rule": "fake-null",
            "kill_rule": "fake-kill",
        }
    )
    requirement4["boundary"]["nonfactorizable_resident_state"] = True
    negative_specification_only_category9_rejected = any(
        "lacks three typed" in error for error in gate_errors(probe4)
    )
    probe5 = json.loads(json.dumps(rows[0]))
    probe5["primary_category"] = 8
    benefit5 = probe5["hardware_benefit_gate"]
    benefit5.update(
        {
            "status": "passed",
            "named_elapsed_time_or_cost_target": "fake-target",
            "evidence_ids": ["c_cloud_cost_spec"],
            "same_workload_and_data_order": True,
            "numerical_and_metric_parity_passed": True,
            "measured_comparative_benefit": True,
            "smallest_beneficial_rung": "L3",
            "stop_rule": "fake-stop",
        }
    )
    negative_specification_only_category8_rejected = any(
        "lacks typed two-rung" in error for error in gate_errors(probe5)
    )
    if not negative_bogus_id_rejected:
        errors.append("negative gate test failed: bogus evidence accepted")
    if not negative_category9_mismatch_rejected:
        errors.append("negative gate test failed: incomplete category 9 accepted")
    if not negative_incomplete_category8_rejected:
        errors.append("negative gate test failed: incomplete category 8 accepted")
    if not negative_specification_only_category9_rejected:
        errors.append("negative gate test failed: calculation specification accepted as category 9 evidence")
    if not negative_specification_only_category8_rejected:
        errors.append("negative gate test failed: calculation specification accepted as category 8 evidence")
    return {
        "passed": not errors,
        "errors": errors,
        "unique_row_ids": len(set(ids)),
        "row_count": len(rows),
        "registry_expected": len(registry_ids),
        "registry_covered": len(registry_matrix),
        "registry_id_sha256": registry_id_sha256,
        "proposed_F21_F66_expected": 46,
        "proposed_F21_F66_covered": len(proposed_f),
        "workstreams_expected": 12,
        "workstreams_covered": len(workstream_ids),
        "research_pillars_expected": len(PILLARS),
        "research_pillars_covered": len(pillar_ids),
        "prompt_required_sections_expected": len(expected_prompt_sections),
        "prompt_required_sections_covered": len(prompt_sections),
        "thematic_frontiers_matrix_count": len(THEMATIC_FRONTIERS),
        "category_8_or_9_rows": sum(item["primary_category"] in (8, 9) for item in rows),
        "hardware_required_rows": sum(item["hardware_required"] for item in rows),
        "negative_gate_tests": {
            "bogus_evidence_id_rejected": negative_bogus_id_rejected,
            "incomplete_category9_rejected": negative_category9_mismatch_rejected,
            "incomplete_category8_rejected": negative_incomplete_category8_rejected,
            "specification_only_category9_rejected": negative_specification_only_category9_rejected,
            "specification_only_category8_rejected": negative_specification_only_category8_rejected,
        },
    }


def build() -> dict[str, Any]:
    current, registry_ids = registry_rows()
    proposed, proposed_f = proposed_f_rows()
    frontiers = thematic_rows()
    workstreams = workstream_rows()
    pillars = pillar_rows()
    rows = current + proposed + frontiers + workstreams + pillars
    counts = Counter(item["primary_category"] for item in rows)
    scope_counts = Counter(item["scope"] for item in rows)

    source_paths = list(LOCAL_SOURCE_PATHS)
    for item in rows:
        for reference in item["evidence_refs"]:
            if reference.startswith("local:"):
                source_paths.append(reference.removeprefix("local:"))
    sources = [source_record(path) for path in dict.fromkeys(source_paths)]
    source_ids = {item["id"] for item in sources if item["exists"]}
    source_errors = [item["path"] for item in sources if not item["exists"]]
    audits = [
        embedded_hash_audit("proof/PROJECT_EXPERIMENT_EXHAUSTION.json"),
        embedded_hash_audit("proof/FRONTIER_LOCALIZATION.json"),
        embedded_hash_audit("proof/ARTIFACT_INDEX/form_substrate.json"),
    ]

    host = json.loads((ROOT / "proof" / "STUDIO_READINESS_CURRENT_HOST.json").read_text())
    cache_measurements = load_cache_measurements()
    validations = validation(rows, registry_ids, proposed_f, source_ids)
    authoritative_checks = authoritative_receipt_checks()
    if not authoritative_checks["all_ok"]:
        validations["errors"].extend(authoritative_checks["errors"])
        validations["passed"] = False

    schema_special = {
        "proof/CACHE_QUARANTINE_AUDIT.json",
        "data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
    }
    row_json_audits: dict[str, Any] = {}
    for item in rows:
        for reference in item["evidence_refs"]:
            if not reference.startswith("local:") or not reference.endswith(".json"):
                continue
            relative = reference.removeprefix("local:")
            if relative in schema_special or relative in row_json_audits:
                continue
            audit = embedded_hash_audit(relative)
            row_json_audits[relative] = audit
            if not audit["all_current"]:
                validations["errors"].append(
                    f"row-referenced JSON has stale root-relative evidence: {relative}"
                )
                validations["passed"] = False
    validations["authoritative_receipt_checks_passed"] = authoritative_checks["all_ok"]
    validations["row_referenced_json_audits_passed"] = all(
        audit["all_current"] for audit in row_json_audits.values()
    )
    if source_errors:
        validations["errors"].append(f"missing local source evidence: {source_errors}")
        validations["passed"] = False

    beneficial_count = counts.get(8, 0)
    required_count = counts.get(9, 0)
    procurement = (
        "no-procurement" if beneficial_count == 0 and required_count == 0 else "gate-dependent-review"
    )
    stale_artifacts = [audit["artifact"] for audit in audits if not audit["all_current"]]
    stale_ids = {f"local:{path}" for path in stale_artifacts}
    stale_row_refs = {
        item["id"]: sorted(set(item["evidence_refs"]) & stale_ids)
        for item in rows
        if set(item["evidence_refs"]) & stale_ids
    }
    if stale_row_refs:
        validations["errors"].append(
            f"authoritative row evidence references stale artifacts: {stale_row_refs}"
        )
        validations["passed"] = False
    validations["stale_artifacts_detected"] = len(stale_artifacts)
    validations["stale_artifacts_excluded_from_row_evidence"] = not stale_row_refs
    validations["decision_counts_derived"] = True
    external_sources = [
        {
            **item,
            "checked": "2026-07-10",
            "content_hash": None,
            "hash_note": "URL/fact ledger; local evidence is byte-hashed",
        }
        for item in EXTERNAL_SOURCES
    ]
    for source in external_sources:
        if source["id"] in {"meta-vjepa21", "meta-vjepa21-paper"}:
            source["pinned_local_evidence_ref"] = "local:proof/VJEPA21_VITB_LOAD.json"
            source["pinned_repository_commit"] = "204698b45b3712590f06245fbfba32d3be539812"
            source["hash_note"] = (
                "classification-critical runtime authority is pinned in the strict-load receipt"
            )
    matrix: dict[str, Any] = {
        "schema": "mop-extended-compute-requirements/v1",
        "snapshot_date": "2026-07-10",
        "scope": "all 227 current registry experiments, F21-F66, 17 candidate frontiers, W0-W11, and dossier pillars 7-25",
        "decision": {
            "procurement": procurement,
            "studio_scale_required_now": required_count > 0,
            "extended_compute_required_count": required_count,
            "extended_compute_beneficial_count": beneficial_count,
            "counts_derived_from_rows": True,
            "evidence_status": "local-byte-and-schema-verified-with-stale-exclusions; external-URL-ledger-manually-reviewed",
            "explanation": "No row passes a named non-factorizable resident-memory, scientifically necessary real-time, or inseparable synchronized-state gate. Larger hardware is presently throughput only and remains unearned even as category 8.",
        },
        "category_definitions": {str(key): value for key, value in CATEGORY_LABELS.items()},
        "classification_counts": {str(key): counts.get(key, 0) for key in CATEGORY_LABELS},
        "scope_counts": dict(sorted(scope_counts.items())),
        "rung_definitions": RUNG_DEFINITIONS,
        "host": {
            "profile": host["profile"]["resolved"],
            "chip": host["host"]["chip"],
            "unified_memory_gb": host["host"]["unified_memory_gb"],
            "memory_semantics": "shared unified memory, not dedicated VRAM; safe MPS working set must be measured at runtime",
            "mps_available": host["host"]["mps_available"],
            "one_heavy_process": True,
            "max_wall_minutes": host["profile"]["envelope"]["max_wall_min"],
            "min_free_disk_gb": host["profile"]["envelope"]["min_free_disk_gb"],
            "kind": "measured-repository-receipt",
            "evidence_path": "proof/STUDIO_READINESS_CURRENT_HOST.json",
        },
        "measured_ceiling_evidence": {
            "vjepa_serial_cache": cache_measurements,
            "cm7_pilot": {
                "parameters": 1_646_080,
                "dense_tokens": 256,
                "matched_objectives": 4,
                "seeds": 5,
                "updates_per_arm": 1000,
                "promotion_ready": False,
                "verdict": "not-promoted",
                "evidence_path": "proof/CUSTOM_SUBSTRATE_PILOT.json",
                "kind": "measured",
            },
        },
        "measurement_catalog": {
            "m_host_profile": {
                "kind": "measured",
                "pointer": "/host",
                "source": "proof/STUDIO_READINESS_CURRENT_HOST.json",
            },
            "m_vitl_cache": {
                "kind": "measured",
                "pointer": "/measured_ceiling_evidence/vjepa_serial_cache/vitl",
            },
            "m_cm7_pilot": {
                "kind": "measured",
                "pointer": "/measured_ceiling_evidence/cm7_pilot",
            },
        },
        "gate_evidence_catalog": resolved_gate_evidence_catalog(),
        "calculations": {
            "c_cache_10000": {
                "formula": "clips * measured_seconds_per_clip / 3600",
                "vitl": 10_000 * cache_measurements["vitl"]["seconds_per_clip"] / 3600,
                "kind": "calculation-from-measurement",
                "limitations": "extrapolated from eight CPU-encoded programmatic clips; no timing variance, cold/warm-cache band, natural-video decode sensitivity, CUDA calibration, or retry overhead",
                "interpretation": "multi-day serial throughput hypothesis, not a resident-memory impossibility",
            },
            "c_dense_cache_storage": {
                "formula": "N_clips * temporal_tokens * spatial_tokens * embedding_dim * bytes_per_value; simple example assumes divisible non-overlapping tubelet/patch convolution and no special tokens",
                "general_tokens_per_axis": "floor((input + 2*padding - dilation*(kernel-1) - 1)/stride + 1)",
                "example_vitl_64f_256_fp16_bytes_per_clip": 8192 * 1024 * 2,
                "example_vitl_10000_clips_fp16_gb_decimal": 10_000 * 8192 * 1024 * 2 / 1e9,
                "kind": "engineering-calculation",
                "actual_cache_note": "the current local ViT-L control artifact is a pooled FP32 vector; this is a hypothetical dense FP16 cache and precision/pooling can change the estimand",
                "interpretation": "storage/fidelity sizing, not attention-activation or simultaneous-residency evidence",
            },
            "c_adam_trainable_state": {
                "formula": "parameters * 16..18 bytes before activations",
                "cm7_low_mb_decimal": 1_646_080 * 16 / 1e6,
                "cm7_high_mb_decimal": 1_646_080 * 18 / 1e6,
                "one_billion_low_gb_decimal": 1_000_000_000 * 16 / 1e9,
                "one_billion_high_gb_decimal": 1_000_000_000 * 18 / 1e9,
                "kind": "engineering-estimate",
                "warning": "not peak memory; excludes activations, attention/workspaces, batch, teacher, decoder/runtime, and fragmentation",
            },
            "c_paired_seed_power": {
                "test": "two-sided paired t test",
                "power": 0.80,
                "independent_unit": "one aggregate paired difference per independent training seed only when seed variability is the target; referents/sessions/environments are nested or crossed repeated measures and require a hierarchical model or multiway bootstrap",
                "alpha_0_05": {"dz_0_3": 90, "dz_0_5": 34, "dz_0_8": 15, "dz_1_0": 10},
                "alpha_0_01": {"dz_0_3": 134, "dz_0_5": 51, "dz_0_8": 22, "dz_1_0": 16},
                "multiplicity": "alpha=.01 is the Bonferroni and first-step Holm worst-case planning threshold for five named comparisons, not Holm's threshold for every ordered test",
                "kind": "exact-noncentral-t-calculation",
                "sequential_note": "table is fixed-sample; repeated looks require preregistered alpha spending or an anytime-valid confidence sequence",
                "interpretation": "five seeds estimate pilot variance; every named endpoint still needs SESOI, variance, paired correlation, and dependence assumptions",
            },
            "c_energy_scope": {
                "formula": "IT_kWh = measured_wall_kW * hours; facility_kWh = IT_kWh * PUE",
                "kind": "calculation-specification",
                "warning": "accelerator TDP is not measured full-system wall power",
            },
            "c_active_disk_floor": {
                "receipt_free_gb_decimal": host["host"]["disk_free_gb"],
                "required_floor_gb_decimal": host["profile"]["envelope"]["min_free_disk_gb"],
                "safe_headroom_gb_decimal": host["host"]["disk_free_gb"]
                - host["profile"]["envelope"]["min_free_disk_gb"],
                "hypothetical_dense_fp16_clips_before_floor": int(
                    (host["host"]["disk_free_gb"] - host["profile"]["envelope"]["min_free_disk_gb"])
                    * 1e9
                    // (8192 * 1024 * 2)
                ),
                "free_needed_for_10000_dense_fp16_plus_floor_gb_decimal": 10_000 * 8192 * 1024 * 2 / 1e9
                + host["profile"]["envelope"]["min_free_disk_gb"],
                "kind": "calculation-from-measured-receipt",
                "interpretation": "active storage floor, not accelerator-memory evidence",
            },
            "c_peak_memory_spec": {
                "formula": "weights + master_weights + gradients + optimizer + activations + attention/workspaces + batch + resident_teacher + decoder/runtime + fragmentation",
                "required_measurements": [
                    "torch.mps.recommended_max_memory",
                    "current_allocated_memory",
                    "driver_allocated_memory",
                    "process-tree RSS",
                    "memory pressure",
                    "three repeated peaks",
                ],
                "kind": "measurement-specification",
            },
            "c_checkpoint_volume_spec": {
                "formula": "checkpoint_bytes * retained_checkpoints * seeds + staging_copy + metadata; distributed runs also count every shard",
                "distinguish": ["model-only", "resumable optimizer/RNG/sampler", "distributed sharded"],
                "kind": "calculation-specification",
            },
            "c_campaign_time_spec": {
                "formula": "independent_pairs * sum(matched_arm_times) * (1 + validation + retry overhead) / measured_parallel_workers_efficiency",
                "kind": "calculation-specification",
            },
            "c_cloud_cost_spec": {
                "formula": "rate * billable_reserved_hours + storage + checkpoint_IO + egress + retry_and_idle_overhead",
                "required_metadata": [
                    "region",
                    "currency",
                    "pricing model",
                    "minimum duration",
                    "retrieval date",
                    "price volatility",
                ],
                "kind": "calculation-specification",
            },
        },
        "hardware_enablement_gate": {
            "all_required": True,
            "conditions": [
                "predeclared scientific fidelity, power, or real-time target",
                "end-to-end local profiling including ingest, decode, cache, train, evaluation, and checkpointing",
                "valid streaming, batch-1, AMP, checkpointing, caching, windowing/recurrence, and resumability attacks attempted",
                "three repeated peak-memory or p95-latency measurements exceed the safe runtime envelope with at least 20% headroom target",
                "proof that factorization or approximation changes the estimand",
                "bounded next-rung pilot passes numerical and data-order parity",
                "named calculation maps the failure to the smallest sufficient rung",
                "null and permanent kill rules are preregistered",
            ],
            "status": "not_passed",
        },
        "rows": rows,
        "local_source_evidence": sources,
        "embedded_hash_audits": audits,
        "authoritative_receipt_checks": authoritative_checks,
        "row_referenced_json_audits": row_json_audits,
        "stale_evidence_policy": {
            "known_stale_artifacts": stale_artifacts,
            "handling": "excluded from authoritative row evidence; retained only as byte-hashed audit inputs and generation-time context",
            "all_stale_artifacts_excluded_from_row_evidence": not stale_row_refs,
            "warning": "a raw hash of a stale generated artifact proves only the bytes audited, not the freshness of its embedded claims",
        },
        "external_primary_sources": external_sources,
        "self_verification": validations,
    }
    if not validations["passed"]:
        raise ValueError("matrix validation failed: " + "; ".join(validations["errors"]))
    matrix["payload_sha256"] = canonical_sha256(matrix)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail unless the committed JSON exactly matches a rebuild"
    )
    args = parser.parse_args()
    payload = build()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file():
            print(f"missing {OUTPUT}", file=sys.stderr)
            return 1
        if OUTPUT.read_text() != rendered:
            print(f"stale {OUTPUT}; rerun {Path(__file__).relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(
            f"verified {OUTPUT.relative_to(ROOT)}: {payload['self_verification']['row_count']} rows, payload {payload['payload_sha256']}"
        )
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: {payload['self_verification']['row_count']} rows, payload {payload['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
