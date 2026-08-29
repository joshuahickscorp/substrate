"""Constitutional preflight, retrospective, and frozen authorities for Substrate v3."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess

from substrate import historical, v2io
from substrate import v3config as C
from substrate import v3io as io
from substrate.evidence import canonical_current_path

V1_TAG = "substrate-v1-terminal"
V2_READY_TAG = "substrate-v2-developmental-ready"
V2_TAG = "substrate-v2-terminal"
PRE_TAG = "substrate-v3-pre-constitutional-ascent"


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
    objects = {}
    drift = []
    for line in lines:
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        current = canonical_current_path(io.ROOT, path)
        current_relative = current.relative_to(io.ROOT).as_posix()
        current_blob = _git("hash-object", "--", current_relative) if current.is_file() else None
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


def immutability() -> dict:
    v1 = _tree_integrity(V1_TAG, ("evidence/substrate/v1", "runs/substrate/v1", "artifacts/substrate/v1", "proof"))
    v2 = _tree_integrity(V2_TAG, ("evidence/substrate/v2", "runs/substrate/v2", "artifacts/substrate/v2", "configs/substrate/v2"))
    historical_check = historical.verify_all()
    v2_seals = {}
    for root_name, root in (("evidence", v2io.EVIDENCE), ("artifacts", v2io.ARTIFACTS)):
        for path in sorted(root.glob("*.json")):
            document = json.loads(path.read_text())
            v2_seals[f"{root_name}/{path.name}"] = {
                "valid": document.get("sha256")
                == v2io.sha_obj({key: value for key, value in document.items() if key != "sha256"}),
                "activation_false": document.get("activation") is False,
            }
    v2_final = json.loads((v2io.EVIDENCE / "SUBSTRATE_V2_FINAL_CLASSIFICATION.json").read_text())
    checks = {
        "v1_byte_identical": v1["byte_identical"],
        "v2_byte_identical": v2["byte_identical"],
        "historical_authority_intact": historical_check["all_pass"],
        "v2_seals_valid": all(row["valid"] and row["activation_false"] for row in v2_seals.values()),
        "v2_terminal_classification_preserved": v2_final.get("classification") == "persistent_developmental_cognition",
    }
    return {
        "schema": "substrate-v3-v1-v2-immutability/v1",
        "v1": v1,
        "v2": v2,
        "v2_ready": {
            "tag": V2_READY_TAG,
            "commit": _git("rev-parse", f"{V2_READY_TAG}^{{}}"),
        },
        "historical": {
            "object_count": len(historical_check["objects"]),
            "failed": historical_check["failed"],
        },
        "v2_seals": v2_seals,
        "v2_classification": v2_final.get("classification"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def preflight() -> dict:
    base = _git("rev-parse", f"{PRE_TAG}^{{}}")
    main = _git("rev-parse", "main")
    origin_main = _git("rev-parse", "origin/main")
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=io.ROOT, text=True).rstrip("\n")
    prs = _json_command(
        [
            "gh", "pr", "list", "--repo", "joshuahickscorp/substrate", "--state", "open",
            "--json", "number,title,headRefName,baseRefName,isDraft,url",
        ],
        [],
    )
    ci = _json_command(
        [
            "gh", "run", "list", "--repo", "joshuahickscorp/substrate", "--branch", "main",
            "--limit", "3", "--json", "databaseId,headSha,status,conclusion,url,workflowName,event",
        ],
        [],
    )
    integrity = immutability()
    resources = _resource_snapshot()
    hawking = _hawking_snapshot()
    v3_principal = io.RUNS / "principal"
    principal_files = sorted(path.relative_to(io.ROOT).as_posix() for path in v3_principal.rglob("*") if path.is_file()) if v3_principal.exists() else []
    processes = subprocess.run(
        ["ps", "axo", "pid=,ppid=,stat=,%cpu=,%mem=,rss=,etime=,command="],
        capture_output=True,
        text=True,
    ).stdout
    v3_workers = [
        line.strip()
        for line in processes.splitlines()
        if "substrate" in line.lower()
        and any(command in line.lower() for command in (" v3 run", " v3 resume"))
    ]
    checks = {
        "pre_tag_matches_entry_main": base == main == origin_main,
        "main_matches_origin_main": main == origin_main,
        "implementation_branch_active": _git("branch", "--show-current") == "agent/substrate-v3-nous-constitutional-ascent",
        "one_worktree": _git("worktree", "list", "--porcelain").count("worktree ") == 1,
        "no_v3_principal_receipts": not principal_files,
        "no_v3_worker": not v3_workers,
        "v1_v2_immutable": integrity["all_pass"],
        "activation_false": io.ACTIVATION is False,
        "disk_safe_for_development": resources["disk_available_gib"] >= 25,
        "memory_safe_for_development": (resources["memory_free_percent"] or 0) >= 8,
        "swap_observed": resources["swap_free_mib"] is not None,
        "hawking_observation_only": hawking["signals_sent"] == 0 and hawking["processes_modified"] == 0,
    }
    return {
        "schema": "substrate-v3-preflight/v1",
        "repository": str(io.ROOT),
        "remote": _git("remote", "get-url", "origin"),
        "entry": {
            "pre_tag": PRE_TAG,
            "pre_tag_commit": base,
            "main": main,
            "origin_main": origin_main,
            "current_head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": dirty.splitlines(),
            "worktrees": _git("worktree", "list", "--porcelain").splitlines(),
            "open_pull_requests": prs,
            "latest_main_ci": ci,
        },
        "principal_files": principal_files,
        "v3_workers": v3_workers,
        "resources": resources,
        "hawking": hawking,
        "checks": checks,
        "failed": sorted(name for name, value in checks.items() if not value),
        "all_pass": all(checks.values()),
        "activation": False,
    }


def seal_preflight() -> dict:
    integrity = immutability()
    entry = preflight()
    hawking = {
        "schema": "substrate-v3-hawking-coexistence/v1",
        "policy": "observe only; never signal, pause, restart, modify, or adopt Hawking",
        "snapshot": entry["hawking"],
        "resources": entry["resources"],
        "safe_for_cheap_cpu_work": entry["checks"]["disk_safe_for_development"] and entry["checks"]["memory_safe_for_development"],
        "principal_requires_resource_rehearsal": True,
        "activation": False,
    }
    io.seal("SUBSTRATE_V3_PREFLIGHT.json", entry, artifact=True)
    io.seal("SUBSTRATE_V3_V1_V2_IMMUTABILITY.json", integrity, artifact=True)
    io.seal("SUBSTRATE_V3_HAWKING_COEXISTENCE.json", hawking, artifact=True)
    return {"preflight": entry, "integrity": integrity, "hawking": hawking}


CAPABILITIES = (
    "identity", "continuity", "memory", "procedural learning", "semantic learning", "development",
    "transfer", "goal preservation", "unity", "cognitive integrity", "self model", "world model",
    "metacognition", "allocation", "ontology", "epistemology", "knowledge", "reasoning",
    "understanding", "explanation", "causality", "counterfactuals", "analogy", "planning",
    "diagnosis", "inquiry", "uncertainty", "reflection", "body continuity", "tool competence",
    "open world adaptation",
)

V2_PRINCIPAL = {
    "identity", "continuity", "memory", "procedural learning", "semantic learning", "development",
    "transfer", "goal preservation", "unity", "cognitive integrity", "self model", "world model",
    "body continuity", "tool competence",
}
V2_NULL = {"allocation"}
V3_REQUIRED = {
    "ontology", "epistemology", "knowledge", "reasoning", "understanding", "explanation", "causality",
    "counterfactuals", "analogy", "planning", "diagnosis", "inquiry", "uncertainty", "reflection",
    "open world adaptation", "allocation",
}


def capability_matrix() -> dict:
    rows = {}
    for dimension in CAPABILITIES:
        if dimension in V2_PRINCIPAL:
            status = "principally_demonstrated"
            principal = "substrate-v2-terminal"
            known_null = None
        elif dimension in V2_NULL:
            status = "principally_null"
            principal = "SUBSTRATE_V2_FINAL_CLASSIFICATION.json:H_D3"
            known_null = "no_oracle_headroom"
        else:
            status = "implemented_only" if dimension in {"ontology", "epistemology", "reasoning"} else "absent"
            principal = None
            known_null = "not tested at the v3 standard"
        rows[dimension] = {
            "original_philosophical_intention": f"one continuing entity has causally useful {dimension}",
            "status": status,
            "current_implementation": (
                "v1 or v2 owned runtime path" if dimension in V2_PRINCIPAL
                else "v3 candidate mechanism requires certification" if dimension in V3_REQUIRED
                else "no v3 specific implementation"
            ),
            "current_causal_path": (
                "principal v2 developmental cycle" if dimension in V2_PRINCIPAL
                else "candidate path only; no v3 credit before canaries"
            ),
            "existing_cheap_evidence": "v1 and v2 sealed batteries" if dimension in V2_PRINCIPAL else None,
            "existing_principal_evidence": principal,
            "strongest_control": "v2 focused ablation or fresh control" if dimension in V2_PRINCIPAL else "not yet run",
            "known_null": known_null,
            "known_defect": (
                "v2 inquiry bed lacked oracle residual above SESOI" if dimension == "allocation"
                else "v3 causal value unestablished" if dimension in V3_REQUIRED
                else None
            ),
            "missing_evidence": (
                "v3 principal and independent verification" if dimension in V3_REQUIRED
                else None
            ),
            "proposed_v3_test": f"{dimension} focused canary, integrated pilot, principal ablation, replication",
        }
    counts = {status: sum(row["status"] == status for row in rows.values()) for status in sorted({row["status"] for row in rows.values()})}
    return {
        "schema": "substrate-v3-capability-matrix/v1",
        "dimensions": rows,
        "counts": counts,
        "architecture_is_not_evidence": True,
        "activation": False,
    }


def retrospect() -> dict:
    matrix = capability_matrix()
    gaps = {
        key: {
            "status": row["status"],
            "missing_evidence": row["missing_evidence"],
            "proposed_v3_test": row["proposed_v3_test"],
            "classification_dependency": key in V3_REQUIRED,
        }
        for key, row in matrix["dimensions"].items()
        if row["missing_evidence"]
    }
    retrospective = {
        "schema": "substrate-v3-constitutional-retrospective/v1",
        "v1_terminal": _git("rev-parse", f"{V1_TAG}^{{}}"),
        "v2_ready": _git("rev-parse", f"{V2_READY_TAG}^{{}}"),
        "v2_terminal": _git("rev-parse", f"{V2_TAG}^{{}}"),
        "v2_classification": "persistent_developmental_cognition",
        "v2_established": sorted(V2_PRINCIPAL),
        "v2_nulls": {"allocation": "no_oracle_headroom"},
        "v2_did_not_establish": sorted(V3_REQUIRED),
        "strongest_conclusion": (
            "v2 established persistent developmental cognition with transfer and exact continuity; "
            "it did not establish the epistemic, structural, inquiry, allocation, and reflective criteria required by v3"
        ),
        "capability_count": len(matrix["dimensions"]),
        "gap_count": len(gaps),
        "activation": False,
    }
    constitution = """# Substrate v3 Nous Constitution

Substrate v3 tests a functional and developmental target. It does not claim consciousness, phenomenal
experience, sentience, feeling, suffering, desire, personhood, life, or moral status. External activation
remains false.

V1 remains a certified cognitive scaffold. V2 remains persistent developmental cognition. Neither
classification is weakened or rewritten by this successor constitution.

Functional Nous, for this campaign, means one continuing cognitive organization that develops and repairs
concepts, distinguishes warranted belief from unsupported confidence, executes and selects multiple
inference procedures, models latent structure across representations, explains and tests counterfactuals,
identifies discriminating missing evidence, regulates inquiry under cost, and preserves identity,
developmental history, goals, body and tool continuity throughout those changes.

Architecture and fluent self-description earn no credit. Every promoted capability requires causal value
over its strongest valid control, an effect at or above the frozen SESOI, independent recomputation from
raw receipts, adversarial mutations, and clean reproduction. A tie or favorable sub-SESOI result is a null.

The strongest automated classification is `nous_ready_for_review`. The unqualified word `Nous` requires
external scientific and philosophical review and cannot be assigned by this campaign.
"""
    io.seal("SUBSTRATE_V3_CONSTITUTIONAL_RETROSPECTIVE.json", retrospective)
    io.seal("SUBSTRATE_V3_CAPABILITY_MATRIX.json", matrix)
    io.seal(
        "SUBSTRATE_V3_EVIDENCE_GAP_MAP.json",
        {"schema": "substrate-v3-evidence-gap-map/v1", "gaps": gaps, "activation": False},
    )
    io.seal_markdown("SUBSTRATE_V3_NOUS_CONSTITUTION.md", constitution)
    return {"retrospective": retrospective, "matrix": matrix, "gaps": gaps}


def freeze() -> dict:
    split_values = [seed for values in C.SPLITS.values() for seed in values]
    classification_dependencies = {
        "persistent_developmental_cognition": ["v2 terminal authority"],
        "epistemically_organized_reasoner": [
            "active ontology", "epistemic distinctions", "defeaters", "underdetermination",
            "multiple reasoning modes", "reasoning selection",
        ],
        "demonstrated_structural_understanding": [
            "epistemically organized reasoner", "cross representation transfer", "explanation",
            "causal intervention", "counterfactual prediction", "model boundary detection",
        ],
        "reflective_cognitive_organization": [
            "persistent developmental cognition", "epistemically organized reasoner",
            "self model utility", "world model utility", "endogenous allocation", "held out inquiry transfer",
        ],
        "functional_proto_nous_candidate": [
            "reflective cognitive organization", "demonstrated structural understanding",
            "active ontology revision", "active inquiry", "useful epistemic divergence",
            "developmental preservation", "body continuity", "interruption recovery",
            "cognitive integrity", "principal completion", "independent verification",
        ],
        "nous_ready_for_review": [
            "functional proto Nous candidate", "independent workload replication",
            "independent history replication", "generator held out evaluation",
            "zero claim mutations", "complete raw review package",
        ],
    }
    constitution = {
        "schema": "substrate-v3-scientific-constitution/v1",
        "purpose": "test development of coherent revisable world, self, warrant, reasoning, understanding, inquiry, and reflective control",
        "v1_dependency": {"tag": V1_TAG, "commit": _git("rev-parse", f"{V1_TAG}^{{}}")},
        "v2_dependency": {
            "ready_tag": V2_READY_TAG,
            "ready_commit": _git("rev-parse", f"{V2_READY_TAG}^{{}}"),
            "terminal_tag": V2_TAG,
            "terminal_commit": _git("rev-parse", f"{V2_TAG}^{{}}"),
            "classification": "persistent_developmental_cognition",
        },
        "definitions": {
            "memory": "information remains accessible",
            "knowledge": "traceable usable belief surviving relevant defeaters",
            "reasoning": "declared inference operations produce conclusions",
            "understanding": "organization supports explanation prediction intervention counterfactual compression reconstruction transfer",
            "inquiry": "identify missing evidence that resolves uncertainty",
            "reflection": "self competence and limitation knowledge regulates cognition",
            "Nous": "one continuing cognitive organization integrates these capacities through development",
        },
        "failure_classes": [
            "implemented", "causally_active", "instrument_verified", "invalid_bed",
            "instrumentation_failure", "no_oracle_headroom", "mechanism_null",
            "unverified_candidate", "mechanism_positive", "terminally_gated", "unlicensed",
        ],
        "sesoi": C.SESOI,
        "tie": "mechanism_null",
        "favorable_below_sesoi": "mechanism_null",
        "principal_changes_after_outcome": "forbidden",
        "activation": False,
    }
    workload_catalog = {}
    for name, row in C.WORKLOADS.items():
        workload_catalog[name] = {
            **row,
            "observation_schema": "opaque local structured record without target or target digest",
            "target": "private latent consequence revealed after committed response",
            "outcome_timing": "after proposal and reasoning receipt are committed",
            "tool_actions": ["deterministic local compare", "deterministic sandbox simulation"],
            "costs": {"base": 1.0, "inquiry": 0.70, "simulation": 0.18},
            "oracle": "private latent structure and optimal admissible action",
            "simple_baselines": ["fixed action", "surface rule", "confidence threshold", "contradiction first"],
            "random_baseline": "rate matched deterministic pseudorandom action",
            "maximum_compute_baseline": "all candidate procedures with greater compute charge",
            "answer_leakage_test": "target and target digest absent from observation serialization",
            "floor": "random accuracy above zero and below oracle",
            "ceiling": "oracle exactly reconstructs private latent consequence",
            "independent_unit": C.STATISTICS["independent_unit"],
        }
    documents = {
        "SUBSTRATE_V3_CLASSIFICATION_AUTHORITY.json": {
            "schema": "substrate-v3-classification-authority/v1",
            "ordered_levels": C.CLAIM_BOUNDARY["ordered_levels"],
            "dependencies": classification_dependencies,
            "maximum": C.CLAIM_BOUNDARY["maximum"],
            "unqualified_nous_requires_external_review": True,
            "activation": False,
        },
        "SUBSTRATE_V3_CLAIM_BOUNDARY.json": {"schema": "substrate-v3-claim-boundary/v1", **C.CLAIM_BOUNDARY},
        "SUBSTRATE_V3_HYPOTHESIS_GRAPH.json": {
            "schema": "substrate-v3-hypothesis-graph/v1",
            "hypotheses": C.HYPOTHESES,
            "classification_dependencies": classification_dependencies,
            "activation": False,
        },
        "SUBSTRATE_V3_SCIENTIFIC_CONSTITUTION.json": constitution,
        "SUBSTRATE_V3_STATISTICAL_AUTHORITY.json": {
            "schema": "substrate-v3-statistical-authority/v1", **C.STATISTICS, "activation": False,
        },
        "SUBSTRATE_V3_SPLIT_AUTHORITY.json": {
            "schema": "substrate-v3-split-authority/v1",
            "splits": {key: list(value) for key, value in C.SPLITS.items()},
            "no_seed_crosses_splits": len(set(split_values)) == len(split_values),
            "no_policy_trains_on_principal_replication_review": True,
            "activation": False,
        },
        "SUBSTRATE_V3_GENERATOR_AUTHORITY.json": {
            "schema": "substrate-v3-generator-authority/v1",
            "callable": "substrate.v3fabric.generate_task",
            "target_private_until_commit": True,
            "deterministic": True,
            "configuration_digest": C.configuration()["configuration_digest"],
            "activation": False,
        },
        "SUBSTRATE_V3_WORKLOAD_CATALOG.json": {
            "schema": "substrate-v3-workload-catalog/v1",
            "workloads": workload_catalog,
            "required_family_count": 6,
            "family_count": len(workload_catalog),
            "activation": False,
        },
        "SUBSTRATE_V3_TRANSFER_GRAPH.json": {
            "schema": "substrate-v3-transfer-graph/v1",
            "positive": {
                key: value["positive_transfer"] for key, value in C.WORKLOADS.items() if value["positive_transfer"]
            },
            "negative": {
                key: value["negative_transfer"] for key, value in C.WORKLOADS.items() if value["negative_transfer"]
            },
            "activation": False,
        },
        "SUBSTRATE_V3_CANDIDATE_LADDER.json": {
            "schema": "substrate-v3-candidate-ladder/v1", **C.CANDIDATE_LADDER, "activation": False,
        },
        "SUBSTRATE_V3_STOP_AND_FUTILITY.json": {
            "schema": "substrate-v3-stop-and-futility/v1",
            "stop": [
                "activation becomes true", "identity mismatch", "answer leakage",
                "principal source drift", "resource safety violation", "operator stop switch",
            ],
            "futility": [
                "valid mechanism null below SESOI", "no oracle headroom",
                "confidence interval excludes SESOI in the unfavorable direction",
            ],
            "retry": C.STATISTICS["retry_rule"],
            "activation": False,
        },
    }
    io.config_json("frozen_configuration.json", C.configuration())
    io.config_json(
        "split_manifest.json",
        {
            "splits": {key: list(value) for key, value in C.SPLITS.items()},
            "configuration_digest": C.configuration()["configuration_digest"],
            "activation": False,
        },
    )
    for name, document in documents.items():
        io.seal(name, document)
    return {
        "sealed": sorted(documents),
        "configuration_digest": C.configuration()["configuration_digest"],
        "split_disjoint": len(set(split_values)) == len(split_values),
        "activation": False,
    }
