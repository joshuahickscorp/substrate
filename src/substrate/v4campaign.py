"""Preflight, v3 root-cause audit, and frozen authorities for Substrate v4."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from substrate import historical, v2io, v3io
from substrate import v4config as C
from substrate import v4io as io

V1_TAG = "substrate-v1-terminal"
V2_TAG = "substrate-v2-terminal"
V3_READY_TAG = "substrate-v3-nous-ready"
V3_TAG = "substrate-v3-terminal"
PRE_TAG = "substrate-v4-pre-structural-understanding"
IMPLEMENTATION_BRANCH = "agent/substrate-v4-structural-understanding"


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=io.ROOT, text=True).strip()


def _json_command(arguments: list[str], fallback):
    result = subprocess.run(arguments, cwd=io.ROOT, capture_output=True, text=True)
    if result.returncode:
        return fallback
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return fallback


def _resource_snapshot() -> dict:
    disk = shutil.disk_usage(io.ROOT)
    memory_result = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True)
    memory_match = re.search(r"System-wide memory free percentage:\s*(\d+)%", memory_result.stdout)
    swap_result = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    swap = {key: float(value) for key, value in re.findall(r"(total|used|free)\s*=\s*([0-9.]+)M", swap_result.stdout)}
    cpu_result = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True)
    cpu = sum(float(value) for value in cpu_result.stdout.split() if value.replace(".", "", 1).isdigit())
    return {
        "disk_available_gib": disk.free / 1024**3,
        "memory_free_percent": int(memory_match.group(1)) if memory_match else None,
        "swap_total_mib": swap.get("total"),
        "swap_used_mib": swap.get("used"),
        "swap_free_mib": swap.get("free"),
        "aggregate_cpu_percent": cpu,
        "machine": {
            "processor": platform.processor() or "Apple silicon",
            "logical_cores": os.cpu_count(),
            "platform": platform.platform(),
        },
    }


def _hawking_snapshot() -> dict:
    result = subprocess.run(
        ["ps", "axo", "pid=,ppid=,%cpu=,%mem=,rss=,etime=,command="],
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "/hawking/" not in lowered or "rg -i" in lowered:
            continue
        fields = line.strip().split(None, 6)
        if len(fields) != 7:
            continue
        pid, ppid, cpu, memory, rss, elapsed, command = fields
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "cpu_percent": float(cpu),
                "memory_percent": float(memory),
                "rss_mib": int(rss) / 1024,
                "elapsed": elapsed,
                "command": command,
            }
        )
    return {
        "processes": rows,
        "active_process_count": len(rows),
        "observation_only": True,
        "signals_sent": 0,
        "processes_modified": 0,
        "controllers_modified": 0,
        "mps_adopted": False,
    }


def _tree_integrity(tag: str, roots: tuple[str, ...]) -> dict:
    lines = _git("ls-tree", "-r", tag, *roots).splitlines()
    parsed = []
    for line in lines:
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        parsed.append((mode, kind, blob, path))
    current_hashes: dict[str, str | None] = {}
    for offset in range(0, len(parsed), 200):
        batch = parsed[offset : offset + 200]
        existing = [path for _, _, _, path in batch if (io.ROOT / path).is_file()]
        if existing:
            hashes = subprocess.check_output(["git", "hash-object", *existing], cwd=io.ROOT, text=True).splitlines()
            current_hashes.update(dict(zip(existing, hashes, strict=True)))
        for _, _, _, path in batch:
            current_hashes.setdefault(path, None)
    objects = {}
    drift = []
    for mode, kind, blob, path in parsed:
        current = io.ROOT / path
        current_blob = current_hashes[path]
        row = {
            "mode": mode,
            "kind": kind,
            "tag_blob": blob,
            "current_blob": current_blob,
            "byte_identical": current_blob == blob,
            "sha256": hashlib.sha256(current.read_bytes()).hexdigest() if current.is_file() else None,
        }
        objects[path] = row
        if not row["byte_identical"]:
            drift.append(path)
    return {
        "tag": tag,
        "commit": _git("rev-parse", f"{tag}^{{}}"),
        "object_count": len(objects),
        "objects": objects,
        "drift": drift,
        "byte_identical": not drift,
    }


def _sealed_tree(root: Path, owner) -> dict:
    rows = {}
    for path in sorted(root.rglob("*.json")):
        document = json.loads(path.read_text())
        declared = document.get("sha256")
        rows[path.relative_to(io.ROOT).as_posix()] = {
            "valid": declared is None or declared == owner.sha_obj({key: value for key, value in document.items() if key != "sha256"}),
            "seal_present": declared is not None,
            "activation_false": document.get("activation") is False,
        }
    return rows


def immutability() -> dict:
    v1 = _tree_integrity(V1_TAG, ("evidence/substrate/v1", "runs/substrate/v1", "artifacts/substrate/v1", "proof"))
    v2 = _tree_integrity(V2_TAG, ("evidence/substrate/v2", "runs/substrate/v2", "artifacts/substrate/v2", "configs/substrate/v2"))
    v3 = _tree_integrity(V3_TAG, ("evidence/substrate/v3", "runs/substrate/v3", "artifacts/substrate/v3", "configs/substrate/v3"))
    historical_check = historical.verify_all()
    v2_seals = {
        **_sealed_tree(v2io.EVIDENCE, v2io),
        **_sealed_tree(v2io.ARTIFACTS, v2io),
        **_sealed_tree(v2io.CONFIGS, v2io),
    }
    v3_seals = {
        **_sealed_tree(v3io.EVIDENCE, v3io),
        **_sealed_tree(v3io.ARTIFACTS, v3io),
        **_sealed_tree(v3io.CONFIGS, v3io),
    }
    v2_final = json.loads((v2io.EVIDENCE / "SUBSTRATE_V2_FINAL_CLASSIFICATION.json").read_text())
    v3_final = json.loads((v3io.EVIDENCE / "SUBSTRATE_V3_FINAL_CLASSIFICATION.json").read_text())
    checks = {
        "v1_byte_identical": v1["byte_identical"],
        "v2_byte_identical": v2["byte_identical"],
        "v3_byte_identical": v3["byte_identical"],
        "historical_authority_intact": historical_check["all_pass"],
        "v2_seals_valid": all(row["valid"] and row["activation_false"] for row in v2_seals.values()),
        "v3_seals_valid": all(row["valid"] and row["activation_false"] for row in v3_seals.values()),
        "v2_classification_preserved": v2_final.get("classification") == "persistent_developmental_cognition",
        "v3_classification_preserved": v3_final.get("classification") == "reflective_cognitive_organization",
    }
    return {
        "schema": "substrate-v4-v1-v2-v3-immutability/v1",
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v3_ready": {"tag": V3_READY_TAG, "commit": _git("rev-parse", f"{V3_READY_TAG}^{{}}")},
        "historical": {
            "object_count": len(historical_check["objects"]),
            "failed": historical_check["failed"],
        },
        "v2_seals": v2_seals,
        "v3_seals": v3_seals,
        "v2_classification": v2_final.get("classification"),
        "v3_classification": v3_final.get("classification"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def preflight() -> dict:
    pre = _git("rev-parse", f"{PRE_TAG}^{{}}")
    main = _git("rev-parse", "main")
    origin_main = _git("rev-parse", "origin/main")
    resources = _resource_snapshot()
    hawking = _hawking_snapshot()
    processes = subprocess.run(
        ["ps", "axo", "pid=,ppid=,stat=,%cpu=,%mem=,rss=,etime=,command="],
        capture_output=True,
        text=True,
    ).stdout
    workers = [
        line.strip() for line in processes.splitlines() if "substrate" in line.lower() and any(command in line.lower() for command in (" v4 run", " v4 resume"))
    ]
    principal_files = (
        sorted(path.relative_to(io.ROOT).as_posix() for path in (io.RUNS / "principal").rglob("*") if path.is_file())
        if (io.RUNS / "principal").exists()
        else []
    )
    integrity = immutability()
    checks = {
        "pre_tag_matches_terminal_main": pre == main == origin_main,
        "main_matches_origin_main": main == origin_main,
        "implementation_branch_active": _git("branch", "--show-current") == IMPLEMENTATION_BRANCH,
        "one_worktree": _git("worktree", "list", "--porcelain").count("worktree ") == 1,
        "no_v4_principal_receipts": not principal_files,
        "no_v4_worker": not workers,
        "v1_v2_v3_immutable": integrity["all_pass"],
        "activation_false": io.ACTIVATION is False,
        "disk_safe_for_development": resources["disk_available_gib"] >= 25,
        "memory_safe_for_development": (resources["memory_free_percent"] or 0) >= 8,
        "swap_observed": resources["swap_free_mib"] is not None,
        "hawking_observation_only": hawking["signals_sent"] == 0 and hawking["processes_modified"] == 0,
    }
    return {
        "schema": "substrate-v4-preflight/v1",
        "repository": str(io.ROOT),
        "remote": _git("remote", "get-url", "origin"),
        "entry": {
            "pre_tag": PRE_TAG,
            "pre_tag_commit": pre,
            "main": main,
            "origin_main": origin_main,
            "current_head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": _git("status", "--porcelain=v1").splitlines(),
            "worktrees": _git("worktree", "list", "--porcelain").splitlines(),
            "open_pull_requests": _json_command(
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    "joshuahickscorp/substrate",
                    "--state",
                    "open",
                    "--json",
                    "number,title,headRefName,baseRefName,isDraft,url",
                ],
                [],
            ),
            "latest_main_ci": _json_command(
                [
                    "gh",
                    "run",
                    "list",
                    "--repo",
                    "joshuahickscorp/substrate",
                    "--branch",
                    "main",
                    "--limit",
                    "3",
                    "--json",
                    "databaseId,headSha,status,conclusion,url,workflowName,event",
                ],
                [],
            ),
        },
        "v4_workers": workers,
        "principal_files": principal_files,
        "resources": resources,
        "hawking": hawking,
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "all_pass": all(checks.values()),
        "activation": False,
    }


def seal_preflight() -> dict:
    integrity = immutability()
    entry = preflight()
    hawking = {
        "schema": "substrate-v4-hawking-coexistence/v1",
        "policy": "observe only; never signal, pause, restart, modify, or adopt Hawking",
        "snapshot": entry["hawking"],
        "resources": entry["resources"],
        "safe_for_cheap_cpu_work": entry["checks"]["disk_safe_for_development"] and entry["checks"]["memory_safe_for_development"],
        "principal_requires_resource_rehearsal": True,
        "activation": False,
    }
    io.seal("SUBSTRATE_V4_PREFLIGHT.json", entry, artifact=True)
    io.seal("SUBSTRATE_V4_V1_V2_V3_IMMUTABILITY.json", integrity, artifact=True)
    io.seal("SUBSTRATE_V4_HAWKING_COEXISTENCE.json", hawking, artifact=True)
    return {"preflight": entry, "integrity": integrity, "hawking": hawking}


def _v3_cycle_comparison(family: str, *, phase_contains: str | None = None) -> dict:
    rows = []
    for seed in range(1000, 1048):
        full_paths = sorted((v3io.RUNS / "principal" / "units").glob(f"principal-{seed}-full_v3-*-shard*.json"))
        control_paths = sorted((v3io.RUNS / "principal" / "units").glob(f"principal-{seed}-more_compute-*-shard*.json"))
        for full_path, control_path in zip(full_paths, control_paths, strict=True):
            full = json.loads(full_path.read_text())
            control = json.loads(control_path.read_text())
            full_cycles = [cycle for cycle in full["cycles"] if cycle["family"] == family and (phase_contains is None or phase_contains in cycle["phase"])]
            control_cycles = [
                cycle for cycle in control["cycles"] if cycle["family"] == family and (phase_contains is None or phase_contains in cycle["phase"])
            ]
            rows.extend(
                {
                    "seed": seed,
                    "task": left["identity"],
                    "same_decision": left["decision"] == right["decision"],
                    "same_correctness": left["outcome"]["correct"] == right["outcome"]["correct"],
                    "full_decision": left["decision"],
                    "control_decision": right["decision"],
                }
                for left, right in zip(full_cycles, control_cycles, strict=True)
            )
    return {
        "comparisons": len(rows),
        "same_decision_rate": sum(row["same_decision"] for row in rows) / len(rows),
        "same_correctness_rate": sum(row["same_correctness"] for row in rows) / len(rows),
        "examples": rows[:12],
    }


def root_cause() -> dict:
    verification = json.loads((v3io.EVIDENCE / "SUBSTRATE_V3_INDEPENDENT_VERIFICATION.json").read_text())
    endpoints = verification["metrics"]["endpoints"]
    comparisons = {
        "cross_representation": _v3_cycle_comparison("cross_representation_systems"),
        "causal": _v3_cycle_comparison("causal_micro_worlds", phase_contains="causal_intervention"),
        "counterfactual": _v3_cycle_comparison("causal_micro_worlds", phase_contains="counterfactual"),
    }
    paths = {
        "cross_representation_mapping": {
            "input": "public surface relations, start node, and query distance",
            "declared_state": "StructuralUnderstanding with representation mappings",
            "actual_state_mutation": "none; the proposal traversed public surface relations directly",
            "checkpoint_ownership": "static structure was checkpointed but no task-specific mapping was learned",
            "downstream_consumer": "IntegratedEntity.propose",
            "scoring_endpoint": "cross_representation_utility",
            "ablation_behavior": "no-understanding was weaker, but more-compute retained the same direct traversal",
            "control_behavior": comparisons["cross_representation"],
        },
        "causal_prediction": {
            "input": "observed cause, confounder, and observational effect",
            "declared_state": "world model",
            "actual_state_mutation": "none; proposal returned observed_cause",
            "checkpoint_ownership": "no learned causal edge entered the v3 checkpoint",
            "downstream_consumer": "IntegratedEntity.propose",
            "scoring_endpoint": "causal_intervention_utility",
            "ablation_behavior": "no-world was weaker, but more-compute used the identical shortcut",
            "control_behavior": comparisons["causal"],
        },
        "counterfactual_evaluation": {
            "input": "public background and one public change",
            "declared_state": "counterfactual structural model",
            "actual_state_mutation": "none; a hard-coded Boolean expression evaluated the public fields",
            "checkpoint_ownership": "no counterfactual history affected structural parameters",
            "downstream_consumer": "IntegratedEntity.propose",
            "scoring_endpoint": "counterfactual_utility",
            "ablation_behavior": "no-world was weaker, but more-compute retained the identical expression",
            "control_behavior": comparisons["counterfactual"],
        },
        "history_divergence": {
            "input": "ontology versus epistemic task histories",
            "declared_state": "history_specialization and static StructuralUnderstanding",
            "actual_state_mutation": "semantic counters changed; causal edges and representation maps did not",
            "checkpoint_ownership": "different hashes reflected different episode summaries rather than useful structural organization",
            "downstream_consumer": "direct task-specific proposal logic",
            "scoring_endpoint": "useful_epistemic_divergence",
            "ablation_behavior": "wrong-history accuracy equaled matched-history accuracy",
            "control_behavior": {
                "effect": endpoints["divergence"]["mean"],
                "confidence_interval": endpoints["divergence"]["bootstrap_95_ci"],
            },
        },
    }
    causes = {
        "no_executable_structural_model": True,
        "structural_state_created_but_unused": True,
        "surface_specific_state_with_no_shared_representation": True,
        "causal_edges_recorded_but_never_executed": True,
        "interventions_treated_as_observation_shortcuts": True,
        "counterfactuals_implemented_as_public_expression_evaluation": True,
        "mappings_supplied_or_bypassed_rather_than_inferred": True,
        "history_updates_not_reaching_structural_parameters": True,
        "scoring_insensitive_to_structure": False,
        "baseline_equally_capable": True,
        "task_lacking_oracle_headroom": False,
    }
    audit = {
        "schema": "substrate-v4-v3-root-cause-audit/v1",
        "v3_terminal_commit": _git("rev-parse", f"{V3_TAG}^{{}}"),
        "v3_classification": "reflective_cognitive_organization",
        "v3_primary_nulls": {
            name: {
                "effect": endpoints[name]["mean"],
                "confidence_interval": endpoints[name]["bootstrap_95_ci"],
            }
            for name in ("cross_representation", "causal", "counterfactual", "divergence")
        },
        "paths": paths,
        "causes": causes,
        "conclusion": (
            "v3 contained a static inspectable graph and correct local operators, but its principal task path "
            "bypassed learned structural state. Full v3 and more-compute therefore committed identical "
            "structural, causal, and counterfactual decisions. History changed summaries rather than executable "
            "causal parameters. V4 must infer one verified intervention model, reuse it across surfaces, and make "
            "all four endpoints consume that same checkpointed state."
        ),
        "scientific_null_preserved": True,
        "activation": False,
    }
    gaps = {
        "schema": "substrate-v4-structural-gap-map/v1",
        "gaps": {
            "executable_state": "learn causal transition laws from verified interventions",
            "shared_consumer": "prediction, intervention, counterfactual, explanation, and alignment consume one model",
            "representation_alignment": "infer surface mappings from structural constraints without shared identifiers",
            "history_individuation": "verified causal histories change held-out specialization",
            "checkpoint": "hash and restore every model, alternative, mapping, revision, and validation receipt",
        },
        "v3_capabilities_to_preserve": [
            "ontology revision",
            "epistemic organization",
            "reasoning selection",
            "inquiry",
            "self model",
            "world model",
            "developmental transfer",
            "identity continuity",
        ],
        "activation": False,
    }
    io.seal("SUBSTRATE_V4_V3_ROOT_CAUSE_AUDIT.json", audit)
    io.seal("SUBSTRATE_V4_STRUCTURAL_GAP_MAP.json", gaps)
    return {"audit": audit, "gaps": gaps}


def freeze() -> dict:
    split_values = [seed for values in C.SPLITS.values() for seed in values]
    constitution = {
        "schema": "substrate-v4-scientific-constitution/v1",
        "central_question": (
            "Can one continuing reflective developmental entity infer, revise, execute, and transfer one history-shaped structural model with causal value?"
        ),
        "hypotheses": C.HYPOTHESES,
        "sesoi": C.SESOI,
        "candidate_ladder": C.CANDIDATE_LADDER,
        "statistics": C.STATISTICS,
        "freeze_rule": "no scientific premise changes after principal launch",
        "claim_boundary": C.CLAIM_BOUNDARY,
        "activation": False,
    }
    classification = {
        "schema": "substrate-v4-classification-authority/v1",
        "demonstrated_structural_understanding": [
            "cross representation structural transfer",
            "causal intervention",
            "counterfactual reasoning",
            "structural explanation",
            "model boundary detection",
            "valid instruments",
            "active mechanism",
            "strongest baselines beaten",
            "independent verification",
        ],
        "functional_proto_nous_candidate": [
            "reflective cognitive organization preserved",
            "demonstrated structural understanding",
            "useful epistemic divergence",
            "structural inquiry",
            "world and self structural utility",
            "developmental transfer",
            "identity and body continuity",
            "interruption recovery",
        ],
        "nous_ready_for_review": [
            "functional proto nous candidate",
            "independent replication",
            "multiple latent and representation families",
            "open world review",
            "zero mutations",
            "complete review package",
        ],
        "unqualified_nous": False,
        "activation": False,
    }
    split = {
        "schema": "substrate-v4-split-authority/v1",
        "splits": {key: list(value) for key, value in C.SPLITS.items()},
        "disjoint": len(split_values) == len(set(split_values)),
        "latent_graph_crossing": "only declared positive and negative transfer relationships",
        "surface_mapping_crossing": False,
        "principal_outcomes_available_to_construction": False,
        "activation": False,
    }
    workload = {
        "schema": "substrate-v4-workload-catalog/v1",
        "families": C.WORKLOADS,
        "representations": C.REPRESENTATIONS,
        "family_count": len(C.WORKLOADS),
        "activation": False,
    }
    generator = {
        "schema": "substrate-v4-generator-authority/v1",
        "owner": "substrate.v4fabric.generate_task",
        "latent_identity_private": True,
        "targets_private_until_commitment": True,
        "verified_intervention_training_only": True,
        "representation_names_randomized": True,
        "ordering_randomized": True,
        "surface_tokens_randomized": True,
        "principal_generators_frozen": True,
        "activation": False,
    }
    transfer = {
        "schema": "substrate-v4-transfer-graph/v1",
        "positive_pairs": [
            ["symbolic_rules", "graph_adjacency"],
            ["event_sequences", "relation_tables"],
            ["structured_language", "tool_state"],
        ],
        "negative_pairs": "same asymmetric topology with a contradictory verified causal orientation",
        "activation": False,
    }
    stop = {
        "schema": "substrate-v4-stop-and-futility/v1",
        "operator_stop": str(io.STOP),
        "futility": "do not launch principal if any required primary mechanism is an active valid admission null",
        "valid_retry": C.STATISTICS["retry_rule"],
        "terminal_failure": "publish exact failed unit and do not silently replace",
        "activation": False,
    }
    io.config_json("frozen_configuration.json", C.configuration())
    io.config_json("split_manifest.json", split)
    io.config_json("candidate_ladder.json", {"candidate_ladder": C.CANDIDATE_LADDER, "activation": False})
    for name, document in (
        ("SUBSTRATE_V4_SCIENTIFIC_CONSTITUTION.json", constitution),
        ("SUBSTRATE_V4_HYPOTHESIS_GRAPH.json", {"schema": "substrate-v4-hypothesis-graph/v1", "hypotheses": C.HYPOTHESES, "activation": False}),
        ("SUBSTRATE_V4_CLASSIFICATION_AUTHORITY.json", classification),
        ("SUBSTRATE_V4_CLAIM_BOUNDARY.json", {"schema": "substrate-v4-claim-boundary/v1", **C.CLAIM_BOUNDARY}),
        ("SUBSTRATE_V4_STATISTICAL_AUTHORITY.json", {"schema": "substrate-v4-statistical-authority/v1", **C.STATISTICS, "activation": False}),
        ("SUBSTRATE_V4_SPLIT_AUTHORITY.json", split),
        ("SUBSTRATE_V4_WORKLOAD_CATALOG.json", workload),
        ("SUBSTRATE_V4_GENERATOR_AUTHORITY.json", generator),
        ("SUBSTRATE_V4_TRANSFER_GRAPH.json", transfer),
        ("SUBSTRATE_V4_CANDIDATE_LADDER.json", {"schema": "substrate-v4-candidate-ladder/v1", "ladders": C.CANDIDATE_LADDER, "activation": False}),
        ("SUBSTRATE_V4_STOP_AND_FUTILITY.json", stop),
    ):
        io.seal(name, document)
    return {
        "configuration": C.configuration(),
        "split_disjoint": split["disjoint"],
        "sealed": list(C.DELIVERABLES[:0])
        + [
            "SUBSTRATE_V4_SCIENTIFIC_CONSTITUTION.json",
            "SUBSTRATE_V4_HYPOTHESIS_GRAPH.json",
            "SUBSTRATE_V4_CLASSIFICATION_AUTHORITY.json",
            "SUBSTRATE_V4_CLAIM_BOUNDARY.json",
            "SUBSTRATE_V4_STATISTICAL_AUTHORITY.json",
            "SUBSTRATE_V4_SPLIT_AUTHORITY.json",
            "SUBSTRATE_V4_WORKLOAD_CATALOG.json",
            "SUBSTRATE_V4_GENERATOR_AUTHORITY.json",
            "SUBSTRATE_V4_TRANSFER_GRAPH.json",
            "SUBSTRATE_V4_CANDIDATE_LADDER.json",
            "SUBSTRATE_V4_STOP_AND_FUTILITY.json",
        ],
        "activation": False,
    }
