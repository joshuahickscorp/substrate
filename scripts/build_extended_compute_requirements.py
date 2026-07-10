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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "proof" / "EXTENDED_COMPUTE_REQUIREMENTS.json"
EXPECTED_REGISTRY_COUNT = 196
EXPECTED_REGISTRY_ID_SHA256 = "fbf69a660bba4b3cce47b4bd622bf9ba634a13a08dd997dbc3241b11c9d88552"

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
        "summary": "M3 Pro bounded shard, one heavy process, 180 minutes, 40 GB free-disk floor",
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
    "m_vith_cache",
    "m_vitg_cache",
    "m_vjepa_scale_atlas",
    "m_cm7_calibration",
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
    "m_vith_cache": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:data/cache/vjepa2_vith_local8_citable/run_receipt.json",
    },
    "m_vitg_cache": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:data/cache/vjepa2_vitg_local8_citable/run_receipt.json",
    },
    "m_vjepa_scale_atlas": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:proof/VJEPA_SCALE_ATLAS_LOCAL.json",
    },
    "m_cm7_calibration": {
        "record_type": "measurement",
        "gate_roles": [],
        "source_ref": "local:proof/CUSTOM_SUBSTRATE_CALIBRATION.json",
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

# These are the only current registry rows that are not category 1.  The
# V-JEPA 2.1 release on 2026-03-16 changes e6 from historical category 7 to
# category 2: the weights now exist, while the repository integration does not.
REGISTRY_OVERRIDES: dict[str, tuple[int, str, str]] = {
    "e6_relational": (
        2,
        "after the active heavy lane exits, acquire the pinned official ViT-B checkpoint, compute its full SHA-256, strict-load ema_encoder, and execute a supervised 8-frame 384px dense forward",
        "hash-pinned-vjepa21-preflight",
    ),
    "mop_dr5_cross_substrate_consistency": (
        2,
        "implement citable same-architecture random controls and the expanded compatible-task grid",
        "registry-relation-plus-cache-audit",
    ),
    "mop_dr14_corruption": (
        2,
        "complete the pinned ViT-B strict-load/forward gate and generate the registered dropped-channel dense cache locally",
        "hash-pinned-vjepa21-preflight-plus-registry-relation",
    ),
    "mop_at1_nuisance_grid": (
        2,
        "complete the citable multi-encoder grid and matched random-init columns",
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
        2,
        "implement rendered action-conditioned observations, a citable substrate cache, the exact frozen V-JEPA 2-AC control, and matched-depth control",
        "local-action-environment-proof-plus-registry",
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
        "implement streaming, disk replay, resumable state, and bounded-memory controls, then measure",
    ),
    (
        "action_conditioned_world_models",
        "Action-conditioned persistent world models",
        2,
        "implement rendered observations, a citable substrate cache, the exact frozen action-conditioned control, and matched-depth control",
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
        2,
        "telemetry logging, intervention hooks, and calibration implementation",
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
        2,
        "local end-to-end ingest/cache/train/eval/retry/storage accounting implementation, with energy explicitly estimated",
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
    "proof/VJEPA_SCALE_ATLAS_LOCAL.json",
    "proof/CUSTOM_SUBSTRATE_CALIBRATION.json",
    "proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json",
    "proof/CUSTOM_SUBSTRATE_ABORTED_MPS_RECOVERY.json",
    "proof/CUSTOM_SUBSTRATE_ABORTED_SOURCE_DRIFT.json",
    "proof/CACHE_QUARANTINE_AUDIT.json",
    "proof/SANPO_REAL_SMOKE_INTAKE.json",
    "proof/SANPO_REAL_SMOKE_INTAKE_DRY_RUN.json",
    "proof/SANPO_REAL_SMOKE_VERIFICATION.json",
    "proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json",
    "proof/FORM_SUBSTRATE/CONTRACT_AUDIT.json",
    "proof/FORM_SUBSTRATE/LOCAL_RUN_SUMMARY.json",
    "proof/FORM_SUBSTRATE/SCORECARD.json",
    "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json",
    "proof/ARTIFACT_INDEX/form_substrate.json",
    "proof/ENCODER_SCALE_VITH_CPU_FORWARD.json",
    "proof/ENCODER_SCALE_VITG_CPU_FORWARD.json",
    "configs/encoder/vjepa21_vitb.yaml",
    "configs/encoder/vjepa21_vitl.yaml",
    "src/mop/substrate/vjepa21_official.py",
    "scripts/vjepa21_official.py",
    "docs/VJEPA21_LOCAL_INTEGRATION.md",
    "data/cache/vjepa2_vitl_local8_citable/cache_manifest.json",
    "data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
    "data/cache/vjepa2_vith_local8_citable/cache_manifest.json",
    "data/cache/vjepa2_vith_local8_citable/run_receipt.json",
    "data/cache/vjepa2_vitg_local8_citable/cache_manifest.json",
    "data/cache/vjepa2_vitg_local8_citable/run_receipt.json",
    "scripts/build_extended_compute_requirements.py",
    "EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md",
    "EXTENDED_COMPUTE_EXECUTION_PLAN.md",
]

EXTERNAL_SOURCES = [
    {
        "id": "meta-vjepa21",
        "url": "https://github.com/facebookresearch/vjepa2",
        "fact": "V-JEPA 2.1 release dated 2026-03-16 and official dense ViT-B/L/g/G checkpoint links",
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

    def walk(node: Any, pointer: str = "") -> None:
        if isinstance(node, dict):
            candidate = node.get("path")
            expected = node.get("sha256")
            if (
                isinstance(candidate, str)
                and isinstance(expected, str)
                and re.fullmatch(r"[0-9a-f]{64}", expected)
            ):
                path = normalize_evidence_path(candidate)
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
    ledger = {item["id"]: item for item in exhaustion["entries"]}
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for item in experiments:
        identifier = item["id"]
        ids.append(identifier)
        category, blocker, basis = REGISTRY_OVERRIDES.get(
            identifier,
            (
                1,
                "none beyond executing or extending an already local implementation",
                "registry-plus-project-exhaustion-ledger",
            ),
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
                refs.append("local:proof/CUSTOM_SUBSTRATE_CALIBRATION.json")
                measured["present"] = True
                measured["evidence_scope"] = (
                    "current one-seed/three-update calibration mechanics only; active v3 campaign excluded until closure"
                )
            if identifier == "e5_curiosity":
                refs.append("local:proof/LOCAL_ACTION_ENVIRONMENT.json")
            if identifier == "mop_cm10_action_forward_model":
                refs.extend(
                    ["local:proof/LOCAL_ACTION_ENVIRONMENT.json", "local:scripts/local_action_environment.py"]
                )
            if identifier in {"e6_relational", "mop_dr14_corruption"}:
                refs.extend(
                    [
                        "local:proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json",
                        "local:configs/encoder/vjepa21_vitb.yaml",
                        "local:src/mop/substrate/vjepa21_official.py",
                        "local:docs/VJEPA21_LOCAL_INTEGRATION.md",
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
                        "local:data/cache/vjepa2_vith_local8_citable/run_receipt.json",
                        "local:data/cache/vjepa2_vitg_local8_citable/run_receipt.json",
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
                    and "local:proof/CUSTOM_SUBSTRATE_CALIBRATION.json" in refs
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
            measured = {
                "present": identifier.startswith("f")
                and identifier not in {"f8_plastic_substrate_rewrite", "f16_perfect_slate_null"},
                "execution_verified_at_generation": None,
                "evidence_scope": "F-chain mechanics are covered by the current contract audit, not the non-F exhaustion ledger",
            }
            promotion_blocker = None
            refs = [
                "local:registry/experiments.yaml",
                "local:proof/FORM_SUBSTRATE/CONTRACT_AUDIT.json",
                "local:proof/FORM_SUBSTRATE/LOCAL_RUN_SUMMARY.json",
            ]
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
                classified_stage="current registered experiment; category 1 means executable local mechanics, not a promotion-grade scientific verdict",
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
        rows.append(
            row(
                identifier=f"frontier_{identifier}",
                name=name,
                scope="candidate_frontier",
                category=category,
                blocker=blocker,
                basis="local-feasibility-attack-plus-primary-source-review",
                status="proposed",
                series="frontier",
                classified_stage="smallest scientifically meaningful falsification attack, before a scale campaign",
                promotion_blocker=promotion,
                evidence_refs=[
                    "local:EXTENDED_COMPUTE_RESEARCH_PROMPT.md",
                    "local:EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md",
                ],
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
    for label in ("vitl", "vith", "vitg"):
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


def authoritative_receipt_checks() -> dict[str, Any]:
    """Run schema-aware checks for evidence whose paths have scoped semantics."""
    errors: list[str] = []
    checks: dict[str, Any] = {}

    try:
        from mop.substrate.cache_tools import validate_cache

        cache_results: dict[str, Any] = {}
        for label in ("vitl", "vith", "vitg"):
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

    for relative in ("proof/CUSTOM_SUBSTRATE_CALIBRATION.json", "proof/LOCAL_ACTION_ENVIRONMENT.json"):
        audit = embedded_hash_audit(relative)
        checks[relative] = audit
        if not audit["all_current"]:
            errors.append(f"{relative}: embedded root-relative evidence is stale")

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

    vjepa21 = json.loads((ROOT / "proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json").read_text())
    config_path = ROOT / "configs/encoder/vjepa21_vitb.yaml"
    vjepa21_ok = all(
        (
            vjepa21.get("all_ok") is True,
            vjepa21.get("checkpoint_remote_validation", {}).get("all_ok") is True,
            vjepa21.get("config_validation", {}).get("all_ok") is True,
            vjepa21.get("config_validation", {}).get("sha256") == sha256_path(config_path),
            vjepa21.get("repository_validation", {}).get("all_ok") is True,
            all(
                item.get("verified") is True
                for item in vjepa21.get("repository_validation", {}).get("artifacts", [])
            ),
            vjepa21.get("claim_boundary", {}).get("model_loaded") is False,
            vjepa21.get("claim_boundary", {}).get("forward_executed") is False,
        )
    )
    checks["vjepa21_vitb_preflight"] = {
        "all_ok": vjepa21_ok,
        "repository_commit": vjepa21.get("official_release", {}).get("repository_commit"),
        "remote_ranges_verified": vjepa21.get("checkpoint_remote_validation", {})
        .get("ranges", {})
        .get("verified"),
        "local_model_loaded": vjepa21.get("claim_boundary", {}).get("model_loaded"),
        "forward_executed": vjepa21.get("claim_boundary", {}).get("forward_executed"),
    }
    if not vjepa21_ok:
        errors.append("V-JEPA 2.1 pinned preflight validation failed")

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
    prompt_frontier_section = prompt_text.split("## Candidate frontiers to investigate, not assume", 1)[
        1
    ].split("## Required extended-compute ladder", 1)[0]
    prompt_frontier_count = len(re.findall(r"^- ", prompt_frontier_section, flags=re.MULTILINE))
    if prompt_frontier_count != 16 or len(THEMATIC_FRONTIERS) != prompt_frontier_count + 1:
        errors.append("prompt candidate-frontier inventory drift")
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
        "prompt_frontiers_expected": prompt_frontier_count,
        "additional_frontiers": len(THEMATIC_FRONTIERS) - prompt_frontier_count,
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
        embedded_hash_audit("proof/ENCODER_SCALE_VITH_CPU_FORWARD.json"),
        embedded_hash_audit("proof/ENCODER_SCALE_VITG_CPU_FORWARD.json"),
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
        "proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json",
        "proof/CACHE_QUARANTINE_AUDIT.json",
        "data/cache/vjepa2_vitl_local8_citable/run_receipt.json",
        "data/cache/vjepa2_vith_local8_citable/run_receipt.json",
        "data/cache/vjepa2_vitg_local8_citable/run_receipt.json",
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
            source["pinned_local_evidence_ref"] = "local:proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json"
            source["pinned_repository_commit"] = "204698b45b3712590f06245fbfba32d3be539812"
            source["hash_note"] = (
                "classification-critical release facts are pinned and hash-validated in the local preflight"
            )
    matrix: dict[str, Any] = {
        "schema": "mop-extended-compute-requirements/v1",
        "snapshot_date": "2026-07-10",
        "scope": "all 196 current registry experiments, F21-F66, 17 candidate frontiers, W0-W11, and dossier pillars 7-25",
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
            "vjepa_atlas": {
                "models": ["ViT-L", "ViT-H", "ViT-g"],
                "referents": 8,
                "all_factor_probe_scores": 1.0,
                "promotion_ready": False,
                "limitations": [
                    "programmatic referents",
                    "no byte-identical cross-resolution stimuli",
                    "no matched random-architecture caches",
                    "no seed distribution",
                ],
                "evidence_path": "proof/VJEPA_SCALE_ATLAS_LOCAL.json",
                "kind": "measured",
            },
            "cm7_calibration": {
                "parameters": 1_646_080,
                "dense_tokens": 256,
                "matched_objectives": 4,
                "seeds": 1,
                "updates_per_arm": 3,
                "promotion_ready": False,
                "evidence_path": "proof/CUSTOM_SUBSTRATE_CALIBRATION.json",
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
            "m_vith_cache": {
                "kind": "measured",
                "pointer": "/measured_ceiling_evidence/vjepa_serial_cache/vith",
            },
            "m_vitg_cache": {
                "kind": "measured",
                "pointer": "/measured_ceiling_evidence/vjepa_serial_cache/vitg",
            },
            "m_vjepa_scale_atlas": {"kind": "measured", "pointer": "/measured_ceiling_evidence/vjepa_atlas"},
            "m_cm7_calibration": {
                "kind": "measured",
                "pointer": "/measured_ceiling_evidence/cm7_calibration",
            },
        },
        "gate_evidence_catalog": resolved_gate_evidence_catalog(),
        "calculations": {
            "c_cache_10000": {
                "formula": "clips * measured_seconds_per_clip / 3600",
                "vitl": 10_000 * cache_measurements["vitl"]["seconds_per_clip"] / 3600,
                "vith": 10_000 * cache_measurements["vith"]["seconds_per_clip"] / 3600,
                "vitg": 10_000 * cache_measurements["vitg"]["seconds_per_clip"] / 3600,
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
                "actual_cache_note": "current local ViT-L/H/g artifacts are pooled FP32 vectors; this is a hypothetical dense FP16 cache and precision/pooling can change the estimand",
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
