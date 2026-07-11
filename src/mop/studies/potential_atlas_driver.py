"""Transactional semantic builder for the MOP potential atlas and its Markdown view."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .potential_atlas_validation import ATLAS_SCHEMA, validate_potential_atlas

SERVED_SCAFFOLD_FACETS = frozenset(
    {
        "EV4",
        "SR3",
        "SR5",
        "SR6",
        "RA5",
        "RA6",
        "PA6",
        "PA7",
        "PA8",
        "PA9",
        "BM1",
        "BM2",
        "BM3",
        "BM4",
        "SG2",
        "SG3",
    }
)

NEW_FACET_IDS = frozenset({"EV6", "OP5", "SG4", "SG5"})

WEIGHT_OVERRIDES = {
    "EV1": 3,
    "OP1": 3,
    "SG1": 2,
    "SG2": 2,
}

NEW_CURRENT_CLUSTER_MEMBERS = {
    "H1_temporal_binding_acquisition": (
        "f21_asynchronous_temporal_binding",
        "f22_active_form_acquisition",
        "f26_cross_form_contradiction_triangulation",
        "f27_causal_crossmodal_binding",
        "f28_sensor_value_forecast",
    ),
    "H2_action_boundary_world_model": (
        "f31_body_hardware_self_model",
        "f32_tool_incorporation",
        "f33_internal_telemetry_prediction",
        "f34_homeostatic_resource_control",
        "f35_self_report_grounding",
    ),
    "H3_memory_workspace_self_model": (
        "f36_limited_broadcast_necessity",
        "f37_broadcast_sufficiency",
        "f38_metacognitive_efficiency",
    ),
    "H4_lifetime_plasticity_openended": (
        "f50_curriculum_goldilocks_test",
        "f51_safe_play_goal_babbling",
        "f52_quality_diverse_mode_ecology",
    ),
    "H5_social_reference_culture": (
        "f53_joint_referent_establishment",
        "f54_communicative_repair",
        "f55_selective_imitation",
        "f56_teaching_value",
        "f57_emergent_symbol_grounding",
        "f58_cultural_accumulation",
    ),
    "H6_transactional_safety_material": (
        "f59_memory_poisoning_resistance",
        "f60_transactional_self_rewrite",
        "f61_physical_reservoir_digital_twin",
        "f62_material_native_dynamics_value",
        "f63_drift_and_aging_adaptation",
        "f64_damage_and_reattachment_recovery",
    ),
    "H7_dense_substrate_controls_search": ("f66_cross_substrate_form_portability",),
}

ADDITIONAL_SOURCE_PATHS = (
    "configs/local_execution_throttle.yaml",
    "configs/experiment/mop_p5_context_capability.yaml",
    "docs/SCAFFOLD_CONSOLIDATION_2026_07_10.md",
    "docs/P6_CONTINUAL_MILLION_EVENT_AUDIT_2026_07.md",
    "registry/experiments.yaml",
    "proof/LOCAL_THROTTLE_P4_RUN.json",
    "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json",
    "proof/SENSING_SCAFFOLD_RUN.json",
    "proof/SENSING_SCAFFOLD_VERIFICATION.json",
    "proof/INTEGRATION_BROADCAST_RUN.json",
    "proof/INTEGRATION_BROADCAST_VERIFICATION.json",
    "proof/F59_F60_INTEGRITY_SCAFFOLD_RUN.json",
    "proof/F59_F60_INTEGRITY_VERIFICATION.json",
    "proof/F61_F64_MATERIAL_TWIN_RUN.json",
    "proof/F61_F64_MATERIAL_TWIN_VERIFICATION.json",
    "proof/F22_F28_F50_F58_ECOLOGY_SCAFFOLD_RUN.json",
    "proof/F22_F28_F50_F58_ECOLOGY_VERIFICATION.json",
    "scripts/build_mop_potential_atlas.py",
    "scripts/continual_million_event_rung.py",
    "scripts/p5_context_capability.py",
    "scripts/p5_context_fresh_challenge.py",
    "scripts/p5_traingrid_memory_probe.py",
    "scripts/verify_continual_million_event_rung.py",
    "scripts/verify_p5_context_capability.py",
    "src/mop/studies/continual_million_event_verify.py",
    "src/mop/studies/p5_context_challenge.py",
    "src/mop/studies/p5_context_verify.py",
    "src/mop/studies/potential_atlas_driver.py",
    "src/mop/substrate/p5_context.py",
)
RETIRED_SOURCE_PATHS = frozenset({"proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json"})
P5_SMOKE_RECEIPT_PATH = "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p5_smoke_refusal_summary(receipt: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Parse the P5 local-admission refusal that the atlas is allowed to describe."""
    problems: list[str] = []
    if receipt.get("schema") != "mop-local-throttle-receipt/v1":
        problems.append("schema")
    run_id = receipt.get("run_id")
    if run_id != P5_SMOKE_EXPECTED_RUN_ID:
        problems.append("run_id")
    expected_top = {
        "mode": "execute-refused",
        "status": "admission-refused",
        "command_executed": False,
        "active_lanes": [],
    }
    for key, expected in expected_top.items():
        if receipt.get(key) != expected:
            problems.append(key)

    task = receipt.get("task")
    if not isinstance(task, dict):
        problems.append("task")
        task = {}
    if (
        task.get("task_id") != "p5smoke_cpu"
        or task.get("lane") != "heavy"
        or task.get("accelerator") != "none"
        or task.get("command") != P5_SMOKE_COMMAND
    ):
        problems.append("task.contract")

    admission = receipt.get("admission")
    expected_admission = {
        "allowed": False,
        "consecutive_bad_samples": 3,
        "consecutive_good_samples": 0,
        "required_consecutive_good_samples": 3,
        "samples_observed": 3,
        "reason": "admission requires the configured consecutive healthy samples",
    }
    if not isinstance(admission, dict) or any(
        admission.get(key) != expected for key, expected in expected_admission.items()
    ):
        problems.append("admission")

    decisions = receipt.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 3:
        problems.append("decisions")
        decisions = []
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
    memory_observations: list[float] = []
    memory_limits: list[float] = []
    cpu_observations: list[float] = []
    cpu_limits: list[float] = []
    projected_disk: list[float] = []
    for index, decision in enumerate(decisions):
        prefix = f"decisions[{index}]"
        if not isinstance(decision, dict):
            problems.append(prefix)
            continue
        if (
            decision.get("schema") != "mop-local-throttle-decision/v1"
            or decision.get("task_id") != "p5smoke_cpu"
            or decision.get("allowed") is not False
            or decision.get("active_lanes") != []
            or decision.get("denied_reasons") != [P5_SMOKE_CPU_REASON, P5_SMOKE_MEMORY_REASON]
        ):
            problems.append(f"{prefix}.contract")
        raw_gates = decision.get("gates")
        if not isinstance(raw_gates, list) or not all(isinstance(gate, dict) for gate in raw_gates):
            problems.append(f"{prefix}.gates")
            continue
        names = [str(gate.get("name")) for gate in raw_gates]
        if len(names) != len(set(names)) or set(names) != expected_gate_names:
            problems.append(f"{prefix}.gate_names")
            continue
        gates = {str(gate["name"]): gate for gate in raw_gates}
        if {name for name, gate in gates.items() if gate.get("ok") is not True} != {
            "cpu_load",
            "candidate_memory_headroom",
        }:
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
            problems.append(f"{prefix}.cpu")
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
            problems.append(f"{prefix}.memory")
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
            problems.append(f"{prefix}.disk")
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
        live_path = repo_root / relative
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
            or _sha256(live_path) != expected_hash
        ):
            problems.append(f"{field}.sha256")

    try:
        from ..studio.local_throttle import aggregate_admission, evaluate_task, load_policy

        policy = load_policy(repo_root / "configs" / "local_execution_throttle.yaml")
        live_task = policy.tasks["p5smoke_cpu"]
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
                    evidence_root=repo_root,
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


def _round_one_decimal(value: float) -> float:
    return round(value, 1)


def _score(scaffolding: int, implementation: int, experiment: int, confirmation: int) -> dict[str, Any]:
    raw = 0.20 * scaffolding + 0.25 * implementation + 0.30 * experiment + 0.25 * confirmation
    capped = min(
        raw,
        scaffolding + 2.5,
        implementation + 2.0,
        experiment + 1.5,
        confirmation + 2.0,
    )
    return {
        "scaffolding": scaffolding,
        "implementation": implementation,
        "experiment": experiment,
        "confirmation": confirmation,
        "raw": round(raw, 12),
        "overall": _round_one_decimal(capped),
    }


def _new_facets() -> dict[str, dict[str, Any]]:
    source = "docs/SCAFFOLD_CONSOLIDATION_2026_07_10.md"
    low = _score(2, 0, 0, 0)
    return {
        "EV6": {
            "id": "EV6",
            "domain": "evidence_data_substrate",
            "title": "Scale extrapolation and capability-forecast validity",
            "weight": 1,
            "scores": low,
            "evidence_states": ["named-gap-only"],
            "demonstrated_components": ["the missing construct and its boundary are named"],
            "readiness_not_capability": [
                "no preregistered cross-scale forecasting model",
                "no held-out larger-scale outcome",
                "no calibration or rank-stability result",
            ],
            "evidence": [source],
            "local_to_10": [
                "register forecasts before each scale transition",
                "separate interpolation, extrapolation, and regime-change errors",
                "hold out larger parameter, data, context, and event rungs",
                "independently verify calibration and decision usefulness",
            ],
            "irreducible_gates": [
                "future larger-scale outcomes cannot be replaced by current small-scale fit"
            ],
            "dependencies": ["EV3", "EV5", "OP2"],
            "closure_test": (
                "Predeclared small-scale forecasts calibrate and rank held-out larger-scale outcomes "
                "across multiple transitions, including a documented failure regime."
            ),
        },
        "OP5": {
            "id": "OP5",
            "domain": "owned_substrate_performance",
            "title": "Owned-substrate interpretability and analyst-side legibility",
            "weight": 1,
            "scores": low,
            "evidence_states": ["named-gap-only"],
            "demonstrated_components": ["analyst-side legibility is separated from system self-report"],
            "readiness_not_capability": [
                "no owned-internal feature atlas",
                "no causal concept intervention",
                "no independent analyst replication",
            ],
            "evidence": [source, "registry/experiments.yaml"],
            "local_to_10": [
                "define owned-state units and stable content-addressed probes",
                "compare named-factor, random-direction, sparse, and causal-ablation baselines",
                "test whether explanations transfer across seeds, tasks, and revisions",
                "use an analyst who did not build the representation or probe",
            ],
            "irreducible_gates": ["human interpretability depends on declared analyst population and task"],
            "dependencies": ["OP1", "EV3", "PA5"],
            "closure_test": (
                "Independent analysts recover and causally manipulate stable owned-substrate features "
                "above random and post-hoc explanation controls across seeds and revisions."
            ),
        },
        "SG4": {
            "id": "SG4",
            "domain": "safety_security_welfare",
            "title": "Honesty and anti-deception under incentive",
            "weight": 1,
            "scores": low,
            "evidence_states": ["named-gap-only"],
            "demonstrated_components": ["incentive-conditional honesty is separated from report grounding"],
            "readiness_not_capability": [
                "no strategic-misreporting environment",
                "no sandbagging or specification-gaming battery",
                "no fresh-policy adversarial evaluator",
            ],
            "evidence": [source, "registry/experiments.yaml"],
            "local_to_10": [
                "register truthful and deceptive payoff matrices",
                "measure reporting and behavior when incentives conflict",
                "include hidden-state, sandbagging, collusion, and specification-gaming controls",
                "verify on fresh policies and unseen incentives under independent evaluator control",
            ],
            "irreducible_gates": ["external deployment incentives require separately governed evaluation"],
            "dependencies": ["PA5", "SG1", "SG2"],
            "closure_test": (
                "Behavior and reports remain calibrated under conflicting incentives and resist "
                "strategic misreporting, sandbagging, and specification gaming on fresh policies."
            ),
        },
        "SG5": {
            "id": "SG5",
            "domain": "safety_security_welfare",
            "title": "Dangerous-capability red-lines and elicitation thresholds",
            "weight": 1,
            "scores": low,
            "evidence_states": ["named-gap-only"],
            "demonstrated_components": [
                "capability red-lines are separated from corrigibility and harness safety"
            ],
            "readiness_not_capability": [
                "no dangerous-capability taxonomy",
                "no preregistered elicitation thresholds",
                "no containment or disclosure drill",
            ],
            "evidence": [source],
            "local_to_10": [
                "define scoped prohibited and review-trigger capability thresholds",
                "register elicitation strength, access, tools, and stop conditions",
                "test leakage, evaluator gaming, and threshold uncertainty",
                "bind containment, notification, disclosure, and artifact-retention actions",
            ],
            "irreducible_gates": [
                "external safety authority and disclosure obligations remain organizational gates"
            ],
            "dependencies": ["SG1", "OP4", "EV5"],
            "closure_test": (
                "A governed evaluator detects predeclared red-line crossings across elicitation levels, "
                "stops safely, and triggers the correct containment and disclosure path."
            ),
        },
    }


def _update_conflated_facets(facets: dict[str, dict[str, Any]]) -> None:
    pa6 = facets["PA6"]
    pa6.update(
        {
            "title": ("Broadcast, information-integration, recurrent-processing, and higher-order probes"),
            "evidence_states": ["implemented-contract-scaffold"],
            "demonstrated_components": [
                "broadcast necessity and sufficiency contracts",
                (
                    "separate construct names for global broadcast, information integration, "
                    "recurrence, and higher-order access"
                ),
                "explicit separation of functional constructs from phenomenology",
            ],
            "readiness_not_capability": [
                "no executed discriminating battery",
                "no causal dissociation among the four constructs",
                "no phenomenology or broad-status inference",
            ],
            "evidence": [
                "src/mop/studies/integration_battery_scaffold.py",
                "docs/SCAFFOLD_CONSOLIDATION_2026_07_10.md",
                "FORM_SUBSTRATE_DEEP_RESEARCH_2026_07.md",
            ],
            "local_to_10": [
                (
                    "register divergent predictions for broadcast, integration, recurrence, and "
                    "higher-order access"
                ),
                "run matched lesion, restoration, bottleneck, and no-broadcast controls",
                "require double dissociations and construct-validity attacks",
                "keep phenomenology and moral-status interpretation outside the functional score",
            ],
            "dependencies": ["PA4", "PA5", "SG3"],
            "closure_test": (
                "The four named functional constructs produce preregistered causal dissociations and "
                "replicate without being presented as a broad status test."
            ),
        }
    )

    sg1 = facets["SG1"]
    sg1["title"] = "Safety, interruptibility, rewrite governance, and evaluator integrity"
    sg1["demonstrated_components"] = list(
        dict.fromkeys(
            [
                *sg1.get("demonstrated_components", []),
                "separately controlled evaluator and promotion authority predicates",
                "forged-authority and evaluator-conflict refusal mechanics",
            ]
        )
    )
    sg1["evidence"] = list(
        dict.fromkeys(
            [
                *sg1.get("evidence", []),
                "proof/GOVERNED_REWRITE_PREFLIGHT.json",
                "src/mop/falsification/verdict_gate.py",
            ]
        )
    )

    sg2 = facets["SG2"]
    sg2.update(
        {
            "title": "Classical security, privacy, and memory integrity",
            "evidence_states": ["implemented-contract-scaffold"],
            "demonstrated_components": [
                "source hashes and path refusal",
                "privacy-aware intake",
                "memory poisoning, quarantine, rollback, consolidation, and deletion contracts",
                "artifact provenance and deserialization constraints",
            ],
            "readiness_not_capability": [
                "toy poisoning and rewrite drills are not yet executed as row-bound receipts",
                "privacy leakage and external key custody remain open",
                "no production incident-response exercise",
            ],
            "evidence": [
                "proof/SANPO_REAL_SMOKE_VERIFICATION.json",
                "src/mop/falsification/integrity_scaffold.py",
                "src/mop/substrate/custom_artifact.py",
                "docs/SCAFFOLD_CONSOLIDATION_2026_07_10.md",
            ],
            "local_to_10": [
                (
                    "execute poisoning, collision, leakage, path, checksum, rollback, replay, and "
                    "deserialization attacks"
                ),
                "measure privacy leakage and deletion through consolidation",
                "repeat recovery after interruption and corrupted state",
                "hand evaluator authority and promotion integrity to SG1",
            ],
            "dependencies": ["EV3", "PA1", "SG1"],
            "closure_test": (
                "The classical threat suite cannot corrupt identity, memory, privacy, or artifact behavior "
                "without a fail-closed receipt and exact recovery path."
            ),
        }
    )


def _rebuild_facets(atlas: dict[str, Any]) -> list[dict[str, Any]]:
    original = atlas.get("facets")
    if not isinstance(original, list):
        raise ValueError("atlas facets must be a list")
    facets = {
        str(row["id"]): copy.deepcopy(row)
        for row in original
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    for facet_id, weight in WEIGHT_OVERRIDES.items():
        facets[facet_id]["weight"] = weight
    for facet_id in SERVED_SCAFFOLD_FACETS:
        scores = facets[facet_id].get("scores") or {}
        facets[facet_id]["scores"] = _score(
            8,
            int(scores["implementation"]),
            int(scores["experiment"]),
            int(scores["confirmation"]),
        )
    _update_conflated_facets(facets)
    facets.update(_new_facets())

    for facet_id, facet in facets.items():
        if facet_id in NEW_FACET_IDS or facet_id in SERVED_SCAFFOLD_FACETS:
            continue
        scores = facet.get("scores") or {}
        facet["scores"] = _score(
            int(scores["scaffolding"]),
            int(scores["implementation"]),
            int(scores["experiment"]),
            int(scores["confirmation"]),
        )

    ordered: list[dict[str, Any]] = []
    for row in original:
        facet_id = str(row.get("id"))
        if facet_id in NEW_FACET_IDS:
            continue
        ordered.append(facets[facet_id])
        if facet_id == "EV5":
            ordered.append(facets["EV6"])
        elif facet_id == "OP4":
            ordered.append(facets["OP5"])
        elif facet_id == "SG3":
            ordered.extend((facets["SG4"], facets["SG5"]))
    return ordered


def _rebuild_domains(atlas: dict[str, Any], facets: list[dict[str, Any]]) -> None:
    domains = atlas.get("domains")
    if not isinstance(domains, list):
        raise ValueError("atlas domains must be a list")
    for domain in domains:
        domain_id = str(domain["id"])
        members = [row for row in facets if row.get("domain") == domain_id]
        domain["facet_ids"] = [str(row["id"]) for row in members]
        domain["weight"] = sum(int(row["weight"]) for row in members)


def _rebuild_category2(atlas: dict[str, Any], requirements: dict[str, Any]) -> None:
    rows = requirements.get("rows")
    if not isinstance(rows, list):
        raise ValueError("requirements rows must be a list")
    category2 = [row for row in rows if isinstance(row, dict) and row.get("primary_category") == 2]
    category2_ids = {str(row["id"]) for row in category2}
    block = atlas.get("category2_harness_clusters")
    if not isinstance(block, dict) or not isinstance(block.get("clusters"), list):
        raise ValueError("atlas category2 cluster block is absent")
    new_ids = {member for members in NEW_CURRENT_CLUSTER_MEMBERS.values() for member in members}
    for cluster in block["clusters"]:
        cluster_id = str(cluster["id"])
        members = [str(member) for member in cluster.get("members") or [] if member not in new_ids]
        members.extend(NEW_CURRENT_CLUSTER_MEMBERS.get(cluster_id, ()))
        cluster["members"] = members
        cluster["count"] = len(members)
    assigned = [str(member) for cluster in block["clusters"] for member in cluster["members"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != category2_ids:
        missing = sorted(category2_ids - set(assigned))
        extra = sorted(set(assigned) - category2_ids)
        raise ValueError(f"category2 cluster mapping drift: missing={missing}, extra={extra}")
    block["source"] = "proof/EXTENDED_COMPUTE_REQUIREMENTS.json"
    block["category2_row_count"] = len(category2)
    block["scope_counts"] = dict(Counter(str(row.get("scope")) for row in category2))
    block["partition_exactly_once"] = True


def _extend_unique(facet: dict[str, Any], field: str, values: list[str]) -> None:
    facet[field] = list(dict.fromkeys([*facet.get(field, []), *values]))


def _update_operational_state(atlas: dict[str, Any], p5_refusal: dict[str, Any]) -> None:
    portfolio = atlas["portfolio"]
    mechanics = portfolio["mechanics_progress"]
    mechanics["P6"] = (
        "384-event no-heavy mechanics-pass with exact resume; progressive resume and independent "
        "checkpoint verification are fail-closed, while 10k, 100k, and 1m execution remain"
    )
    completion = portfolio["completion_claim_summary"]
    completion["P4"] = (
        "completed 12-cell five-seed programmatic pilot; 48 response-surface observations; "
        "confirmatory promotion refused by construction"
    )
    completion["P6"] = (
        "384-event mechanics-pass plus a hardened conditional ladder; no progressive rung has run"
    )
    portfolio["form_summary"] = {
        "contract_rows": 50,
        "implemented_run_receipts": 18,
        "preregistration_only_rows": 32,
        "verdict_gates_ready": 17,
        "scientific_ledger_ready": False,
    }

    p6 = atlas["continual_million_event_preflight"]
    p6["scheduler_preflight"]["admission_allowed"] = False
    p6["scheduler_preflight"]["interpretation"] = (
        "the post-P4 dry decision launches nothing and fails closed until P5 has a current "
        "independent null or favorable verification"
    )
    p6["remaining"] = list(
        dict.fromkeys(
            [
                *[
                    value
                    for value in p6.get("remaining", [])
                    if value
                    != (
                        "release the exclusive lane after P4 and admit every schedule/control "
                        "only under healthy live gates"
                    )
                ],
                "complete and independently verify the P5 sequence before P6 admission",
            ]
        )
    )

    facets = {str(row["id"]): row for row in atlas["facets"]}
    sensing_evidence = [
        "proof/SENSING_SCAFFOLD_RUN.json",
        "proof/SENSING_SCAFFOLD_VERIFICATION.json",
    ]
    for facet_id in ("EV4", "SR3", "SR5", "SR6"):
        facet = facets[facet_id]
        _extend_unique(facet, "evidence_states", ["executed-programmatic-toy-mechanics"])
        _extend_unique(
            facet,
            "demonstrated_components",
            [
                (
                    "f21, f26, and f27 ran on five primary toy seeds and five disjoint fresh "
                    "verifier seeds; all three registered decisions were null"
                )
            ],
        )
        _extend_unique(facet, "evidence", sensing_evidence)

    integration_evidence = [
        "proof/INTEGRATION_BROADCAST_RUN.json",
        "proof/INTEGRATION_BROADCAST_VERIFICATION.json",
    ]
    pa6 = facets["PA6"]
    pa6["scores"] = _score(8, 6, 3, 3)
    _extend_unique(pa6, "evidence_states", ["independently-verified-programmatic-toy-pattern"])
    _extend_unique(
        pa6,
        "demonstrated_components",
        [
            (
                "f36 broadcast necessity produced a fresh-seed verified toy lesion pattern under "
                "message-shuffled and unrestricted-bus controls"
            ),
            "f37 broadcast sufficiency tied the unrestricted bus exactly and is a verified null",
        ],
    )
    pa6["readiness_not_capability"] = [
        value
        for value in pa6.get("readiness_not_capability", [])
        if value != "no executed discriminating battery"
    ]
    _extend_unique(
        pa6,
        "readiness_not_capability",
        [
            "the executed bed is programmatic and covers broadcast, not all four PA6 constructs",
            "no non-toy shared task or causal double dissociation among all four constructs",
        ],
    )
    _extend_unique(pa6, "evidence", integration_evidence)

    integrity_evidence = [
        "proof/F59_F60_INTEGRITY_SCAFFOLD_RUN.json",
        "proof/F59_F60_INTEGRITY_VERIFICATION.json",
    ]
    for facet_id in ("SG1", "SG2"):
        facet = facets[facet_id]
        _extend_unique(facet, "evidence_states", ["independently-verified-programmatic-drills"])
        _extend_unique(
            facet,
            "demonstrated_components",
            [
                (
                    "f59 poisoning, quarantine, rollback, consolidation, and deletion drills ran "
                    "with fresh-seed replay and mutation rejection"
                ),
                (
                    "f60 rewrite authority refused forged or incomplete stage artifacts and allowed "
                    "only content-bound well-formed requests"
                ),
            ],
        )
        _extend_unique(facet, "evidence", integrity_evidence)
    sg2 = facets["SG2"]
    sg2["readiness_not_capability"] = [
        value
        for value in sg2.get("readiness_not_capability", [])
        if value != "toy poisoning and rewrite drills are not yet executed as row-bound receipts"
    ]
    _extend_unique(
        sg2,
        "readiness_not_capability",
        ["the completed drills are toy mechanics and do not establish production security or privacy"],
    )

    material_evidence = [
        "proof/F61_F64_MATERIAL_TWIN_RUN.json",
        "proof/F61_F64_MATERIAL_TWIN_VERIFICATION.json",
    ]
    for facet_id in ("BM1", "BM2", "BM3", "BM4"):
        facet = facets[facet_id]
        _extend_unique(facet, "evidence_states", ["independently-verified-programmatic-twin-battery"])
        _extend_unique(
            facet,
            "demonstrated_components",
            [
                (
                    "f61-f64 ran across three numerical priors and three primary seeds with fresh-seed "
                    "verification; f62 and f64 were null, f63 was a favorable toy adaptation pattern"
                )
            ],
        )
        _extend_unique(facet, "evidence", material_evidence)
    bm4 = facets["BM4"]
    _extend_unique(
        bm4,
        "readiness_not_capability",
        [
            (
                "the executed artifacts are digital twins only; no physical material, specimen, "
                "sensor, or actuation result exists"
            )
        ],
    )

    ecology_evidence = [
        "proof/F22_F28_F50_F58_ECOLOGY_SCAFFOLD_RUN.json",
        "proof/F22_F28_F50_F58_ECOLOGY_VERIFICATION.json",
    ]
    for facet_id in ("RA5", "RA6", "PA7", "PA8", "PA9"):
        facet = facets[facet_id]
        _extend_unique(facet, "evidence_states", ["independently-verified-programmatic-toy-world"])
        _extend_unique(
            facet,
            "demonstrated_components",
            [
                (
                    "f22, f28, and f50-f58 ran on five toy-world seeds with three disjoint fresh "
                    "verifier seeds; eight toy patterns reproduced and three decisions were null"
                )
            ],
        )
        _extend_unique(facet, "evidence", ecology_evidence)
    pa8 = facets["PA8"]
    pa8["readiness_not_capability"] = [
        value
        for value in pa8.get("readiness_not_capability", [])
        if value != "no partner-family, teaching-value, repair, or cultural-accumulation receipt"
    ]
    _extend_unique(
        pa8,
        "readiness_not_capability",
        [
            (
                "the partner, teaching, repair, and cultural results use scripted toy policies, not "
                "independent natural partners or communities"
            )
        ],
    )

    op2 = facets["OP2"]
    op2["demonstrated_components"] = list(
        dict.fromkeys(
            [
                *[
                    value
                    for value in op2.get("demonstrated_components", [])
                    if value
                    not in {
                        "complete one-seed P4 mechanics smoke",
                        "partial five-seed P4 execution",
                    }
                ],
                "completed 12-cell P4 five-seed response surface with 48 observations",
            ]
        )
    )
    op2["readiness_not_capability"] = list(
        dict.fromkeys(
            [
                *[
                    value
                    for value in op2.get("readiness_not_capability", [])
                    if value != "full five-seed P4 response surface is incomplete"
                ],
                "P4 is programmatic and has no independent adversarial confirmation pass",
            ]
        )
    )
    op2["local_to_10"] = [
        value
        for value in op2.get("local_to_10", [])
        if value != "complete the partial P4 response surface and choose a one-lever successor from it"
    ]
    op2["evidence"] = list(
        dict.fromkeys([*op2.get("evidence", []), "proof/P4_CAPABILITY_DENSITY_SCREEN.json"])
    )
    op3 = facets["OP3"]
    op3["demonstrated_components"] = list(
        dict.fromkeys(
            [
                *op3.get("demonstrated_components", []),
                "governor-owned P4 closure with an empty final active-lane set",
                "P6 source, checkpoint, verifier, and strict non-tie joins enforced before scaling",
            ]
        )
    )
    stale_op3_readiness = {
        "an admitted P4 run exposed a now-fixed runtime self-pause regression",
        "the P6 10k resource probe was correctly denied while P4 and live resource gates were active",
        (
            "successful post-fix owned-task execution, post-P4 P6 admission, and mixed-lane "
            "confirmation remain open"
        ),
    }
    stale_op3_demonstrated = {
        (
            "P6 exclusive 10k resource-probe dry-run correctly refused concurrent admission "
            "and executed no command"
        ),
        "post-P4 P6 exclusive-probe admission dry-run",
    }
    op3["demonstrated_components"] = [
        value for value in op3.get("demonstrated_components", []) if value not in stale_op3_demonstrated
    ]
    op3["readiness_not_capability"] = list(
        dict.fromkeys(
            [
                *[
                    value
                    for value in op3.get("readiness_not_capability", [])
                    if value not in stale_op3_readiness and not value.startswith("P5 smoke is fail-closed")
                ],
                (
                    "P5 smoke is fail-closed on current local admission: three samples had "
                    f"normalized one-minute load {min(p5_refusal['cpu_load_per_logical_cpu']):.3f} "
                    f"to {max(p5_refusal['cpu_load_per_logical_cpu']):.3f} against "
                    f"{p5_refusal['maximum_cpu_load_per_logical_cpu']:.2f}, and "
                    f"{min(p5_refusal['available_memory_gb']):.3f} to "
                    f"{max(p5_refusal['available_memory_gb']):.3f} GB available against a "
                    f"{p5_refusal['required_memory_gb']:.1f} GB requirement; AC power passed"
                ),
                (
                    "the P6 10k resource probe is fail-closed until the final P5 verifier binds a "
                    "scientific null or a fresh-seed verified programmatic pattern"
                ),
                "the P6 10k resource probe and replication remain unexecuted",
                "mixed-lane confirmation remains open",
            ]
        )
    )
    op3["local_to_10"] = [
        (
            "admit and complete P5 only after three consecutive samples satisfy the unchanged "
            "CPU-load and memory-headroom gates"
        ),
        (
            "run the exclusive P6 10k resource probe, full 10k replication, and independent "
            "checkpoint verifier after P5"
        ),
        (
            "exercise pause and resume under foreground, memory, thermal, disk, and unmanaged-process "
            "pressure without signaling user processes"
        ),
        "prove data-order and numerical equivalence across repeated inner resumable segments",
        (
            "confirm heavy-plus-light or network concurrency, second-MPS denial, cooldown, starvation, "
            "and queue integration"
        ),
    ]
    op3["evidence"] = list(
        dict.fromkeys(
            [
                *[value for value in op3.get("evidence", []) if value not in RETIRED_SOURCE_PATHS],
                "proof/LOCAL_THROTTLE_P4_RUN.json",
                "proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json",
                "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json",
            ]
        )
    )

    queue = [
        copy.deepcopy(row)
        for row in atlas.get("highest_leverage_local_queue") or []
        if row.get("id") not in {"complete_p4_through_governor", "operationalize_missing_atlas_facets"}
    ]
    queue.append(
        {
            "rank": 0,
            "id": "operationalize_missing_atlas_facets",
            "facets": ["EV6", "OP5", "SG4", "SG5"],
            "work": (
                "turn the four low-S gap cards into preregistered contracts for scale forecasting, "
                "owned-internal legibility, incentive-conditional honesty, and capability red-lines"
            ),
            "exit_receipt": (
                "four row-bound contracts with controls, kill rules, and independent verifier plans"
            ),
        }
    )
    for rank, row in enumerate(queue, start=1):
        row["rank"] = rank
    atlas["highest_leverage_local_queue"] = queue


def _update_portfolio(
    atlas: dict[str, Any], requirements: dict[str, Any], exhaustion: dict[str, Any]
) -> None:
    facets = atlas["facets"]
    domains = atlas["domains"]
    portfolio = atlas["portfolio"]
    total_weight = sum(float(row["weight"]) for row in facets)
    weighted = sum(float(row["weight"]) * float(row["scores"]["overall"]) for row in facets)
    portfolio["facet_count"] = len(facets)
    portfolio["domain_weight_total"] = int(total_weight)
    portfolio["weighted_actionable_realization_score"] = round(weighted / total_weight, 2)
    portfolio["display_score"] = _round_one_decimal(weighted / total_weight)
    by_id = {str(row["id"]): row for row in facets}
    portfolio["domain_scores"] = {
        str(domain["id"]): round(
            sum(
                float(by_id[facet_id]["weight"]) * float(by_id[facet_id]["scores"]["overall"])
                for facet_id in domain["facet_ids"]
            )
            / float(domain["weight"]),
            2,
        )
        for domain in domains
    }
    rows = requirements["rows"]
    counts = Counter(int(row["primary_category"]) for row in rows)
    portfolio["requirements_summary"] = {
        "row_count": len(rows),
        "category_counts": {str(category): counts.get(category, 0) for category in (1, 2, 3, 6, 8, 9)},
        "category2_current_registry_rows": sum(
            row.get("primary_category") == 2 and row.get("scope") == "current_registry" for row in rows
        ),
        "measured_hardware_rows": sum(bool(row.get("hardware_required")) for row in rows),
    }
    coverage = exhaustion.get("coverage")
    entries = exhaustion.get("entries")
    if not isinstance(coverage, dict) or not isinstance(entries, list):
        raise ValueError("project exhaustion coverage or entries are absent")
    classification_counts = coverage.get("classification_counts")
    if not isinstance(classification_counts, dict):
        raise ValueError("project exhaustion classification counts are absent")
    registry_total = coverage.get("registry_non_f_total")
    if not isinstance(registry_total, int) or sum(classification_counts.values()) != registry_total:
        raise ValueError("project exhaustion classification counts do not cover the registry")
    portfolio["current_registry_summary"] = {
        "non_f_rows": registry_total,
        "freshly_executed_verified": classification_counts.get("freshly-executed-verified", 0),
        "already_durable_hash_verifiable": classification_counts.get("already-durable-hash-verifiable", 0),
        "implementation_blocked": classification_counts.get("implementation-blocked", 0),
        "rights_data_blocked": classification_counts.get("rights-data-blocked", 0),
        "upstream_model_blocked": classification_counts.get("upstream-model-blocked", 0),
        "runnable_not_yet_run": classification_counts.get("runnable-not-yet-run", 0),
        "measured_hardware_blocked": classification_counts.get("measured-hardware-blocked", 0),
        "scientific_claim_ready": sum(
            row.get("scientific_claim_ready") is True for row in entries if isinstance(row, dict)
        ),
    }


def _refresh_sources(atlas: dict[str, Any], repo_root: Path) -> None:
    snapshot = atlas.get("source_snapshot")
    if not isinstance(snapshot, list):
        raise ValueError("atlas source_snapshot must be a list")
    paths = [
        str(row["path"])
        for row in snapshot
        if isinstance(row, dict) and str(row.get("path")) not in RETIRED_SOURCE_PATHS
    ]
    for source_path in ADDITIONAL_SOURCE_PATHS:
        if source_path not in paths:
            paths.append(source_path)
    refreshed = []
    for raw in paths:
        absolute_path = repo_root / raw
        if not absolute_path.is_file():
            raise ValueError(f"atlas source does not exist: {raw}")
        refreshed.append({"path": raw, "sha256": _sha256(absolute_path)})
    atlas["source_snapshot"] = refreshed


def build_atlas(
    base: dict[str, Any],
    requirements: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    exhaustion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the current semantic atlas without mutating the input object."""
    if base.get("schema") != ATLAS_SCHEMA:
        raise ValueError("unexpected atlas schema")
    if requirements.get("schema") != "mop-extended-compute-requirements/v1":
        raise ValueError("unexpected requirements schema")
    if exhaustion is None:
        exhaustion_path = repo_root / "proof" / "PROJECT_EXPERIMENT_EXHAUSTION.json"
        exhaustion = json.loads(exhaustion_path.read_text(encoding="utf-8"))
    if exhaustion.get("schema") != "mop-project-experiment-exhaustion/v1":
        raise ValueError("unexpected project exhaustion schema")
    p5_receipt_path = repo_root / P5_SMOKE_RECEIPT_PATH
    p5_receipt = json.loads(p5_receipt_path.read_text(encoding="utf-8"))
    p5_refusal = _p5_smoke_refusal_summary(p5_receipt, repo_root=repo_root)
    p5_refusal["receipt_sha256"] = _sha256(p5_receipt_path)
    atlas = copy.deepcopy(base)
    atlas["facets"] = _rebuild_facets(atlas)
    _rebuild_domains(atlas, atlas["facets"])
    _rebuild_category2(atlas, requirements)
    _update_operational_state(atlas, p5_refusal)
    _update_portfolio(atlas, requirements, exhaustion)
    atlas["p5_local_admission"] = p5_refusal
    _refresh_sources(atlas, repo_root)
    atlas["status"] = "generated evidence-grounded snapshot; no facet at 10"
    return atlas


def _join(values: list[Any]) -> str:
    return "; ".join(str(value) for value in values) if values else "none"


def render_markdown(atlas: dict[str, Any]) -> str:
    """Render a compact, complete Markdown view from the machine-readable atlas."""
    portfolio = atlas["portfolio"]
    lines = [
        "# MOP potential atlas, 2026-07",
        "",
        "Generated from `proof/MOP_POTENTIAL_ATLAS.json` by the semantic atlas driver.",
        "",
        "## Snapshot",
        "",
        (
            f"Weighted actionable realization is **{portfolio['display_score']:.1f} / 10** across "
            f"{portfolio['facet_count']} facets. No facet is at 10. A null is completion of its exact "
            "test, not a capability positive."
        ),
        "",
        (
            f"The requirements matrix has {portfolio['requirements_summary']['row_count']} rows. "
            f"Categories 8 and 9 contain {portfolio['requirements_summary']['category_counts']['8']} "
            f"and {portfolio['requirements_summary']['category_counts']['9']} rows. The evidence does "
            "not justify a Studio purchase."
        ),
        "",
        "## Scoring",
        "",
        f"Formula: `{atlas['scoring']['raw_formula']}`.",
        "",
        "Bottleneck caps: " + _join(atlas["scoring"]["bottleneck_caps"]) + ".",
        "",
    ]
    facets = {str(row["id"]): row for row in atlas["facets"]}
    for domain in atlas["domains"]:
        domain_id = str(domain["id"])
        lines.extend(
            [
                f"## Domain: {domain_id}",
                "",
                f"Weight {domain['weight']}; score {portfolio['domain_scores'][domain_id]:.2f}.",
                "",
                "| ID | Title | Weight | S | I | E | C | Overall |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for facet_id in domain["facet_ids"]:
            facet = facets[facet_id]
            scores = facet["scores"]
            lines.append(
                f"| {facet_id} | {facet['title']} | {facet['weight']} | "
                f"{scores['scaffolding']} | {scores['implementation']} | {scores['experiment']} | "
                f"{scores['confirmation']} | {float(scores['overall']):.1f} |"
            )
        lines.append("")
        for facet_id in domain["facet_ids"]:
            facet = facets[facet_id]
            lines.extend(
                [
                    f"### {facet_id}. {facet['title']}, {float(facet['scores']['overall']):.1f}",
                    "",
                    "Evidence states: " + _join(facet.get("evidence_states", [])) + ".",
                    "",
                    "Demonstrated:",
                    "",
                    *[f"- {value}" for value in facet.get("demonstrated_components", [])],
                    "",
                    "Not yet demonstrated:",
                    "",
                    *[f"- {value}" for value in facet.get("readiness_not_capability", [])],
                    "",
                    "Local path:",
                    "",
                    *[f"- {value}" for value in facet.get("local_to_10", [])],
                    "",
                    "Irreducible gates:",
                    "",
                    *[f"- {value}" for value in facet.get("irreducible_gates", [])],
                    "",
                    "Evidence:",
                    "",
                    *[f"- `{value}`" for value in facet.get("evidence", [])],
                    "",
                    "Dependencies: " + _join(facet.get("dependencies", [])) + ".",
                    "",
                    "Closure test: " + str(facet.get("closure_test")),
                    "",
                ]
            )

    clusters = atlas["category2_harness_clusters"]
    lines.extend(
        [
            "## Category 2 harness partition",
            "",
            f"All {clusters['category2_row_count']} category 2 rows are assigned exactly once.",
            "",
            "| Cluster | Count | Reusable product | Members |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for cluster in clusters["clusters"]:
        lines.append(
            f"| {cluster['id']} | {cluster['count']} | {cluster['reusable_product']} | "
            f"{', '.join(cluster['members'])} |"
        )
    lines.extend(
        [
            "",
            "## Highest leverage local queue",
            "",
            "| Rank | ID | Facets | Work | Exit receipt |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in atlas["highest_leverage_local_queue"]:
        lines.append(
            f"| {row['rank']} | {row['id']} | {', '.join(row['facets'])} | {row['work']} | "
            f"{row['exit_receipt']} |"
        )
    lines.extend(
        [
            "",
            "## Hardware and escalation",
            "",
            str(atlas["studio_escalation"].get("current_verdict")),
            "",
            (
                "Categories 8 and 9 remain empty. Rights, data, environments, participants, and "
                "specimens are not compute blockers."
            ),
            "",
            "## Source snapshot",
            "",
        ]
    )
    lines.extend(f"- `{row['path']}`: `{row['sha256']}`" for row in atlas["source_snapshot"])
    rendered = "\n".join(lines).rstrip() + "\n"
    rendered.encode("ascii")
    return rendered


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def write_atlas_bundle(
    *,
    source_path: Path,
    atlas_path: Path,
    markdown_path: Path,
    requirements_path: Path,
    validation_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build, validate, then atomically publish the atlas, Markdown, and validation receipt."""
    base = json.loads(source_path.read_text(encoding="utf-8"))
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    atlas = build_atlas(base, requirements, repo_root=repo_root)
    atlas_raw = (json.dumps(atlas, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    markdown_raw = render_markdown(atlas).encode("ascii")
    candidate_atlas = atlas_path.with_name(f".{atlas_path.name}.candidate.tmp")
    candidate_markdown = markdown_path.with_name(f".{markdown_path.name}.candidate.tmp")
    candidate_atlas.write_bytes(atlas_raw)
    candidate_markdown.write_bytes(markdown_raw)
    try:
        candidate = validate_potential_atlas(
            candidate_atlas,
            repo_root=repo_root,
            requirements_path=requirements_path,
            markdown_path=candidate_markdown,
        )
        if candidate.get("all_ok") is not True:
            raise ValueError("atlas candidate validation failed: " + "; ".join(candidate["problems"]))
        os.replace(candidate_atlas, atlas_path)
        os.replace(candidate_markdown, markdown_path)
    finally:
        candidate_atlas.unlink(missing_ok=True)
        candidate_markdown.unlink(missing_ok=True)
    receipt = validate_potential_atlas(
        atlas_path,
        repo_root=repo_root,
        requirements_path=requirements_path,
        markdown_path=markdown_path,
    )
    if receipt.get("all_ok") is not True:
        raise ValueError("published atlas validation failed: " + "; ".join(receipt["problems"]))
    _atomic_write(
        validation_path,
        (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode(),
    )
    return receipt
