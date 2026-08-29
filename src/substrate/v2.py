"""Substrate v2 developmental campaign authority and command family.

The developmental mechanisms extend the existing eleven stage runtime.  V1 evidence and receipts are
read only inputs identified by the immutable terminal tag.

"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from substrate import evidence as v1
from substrate import execution as v1_execution
from substrate import historical, v2fabric
from substrate import v2config as C
from substrate import v2io as io

V1_TAG = "substrate-v1-terminal"
PRE_TAG = "substrate-v2-pre-development"
SESOI = 0.05


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
            "chip": platform.processor() or "Apple M3 Ultra",
            "logical_cores": os.cpu_count(),
            "memory_gib": 96,
        },
    }


def _hawking_snapshot() -> dict:
    result = subprocess.run(
        ["ps", "axo", "pid=,ppid=,%cpu=,rss=,etime=,comm=,args="],
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if "glm52_activation_aware_pack.py" not in line:
            continue
        fields = line.strip().split(None, 6)
        if len(fields) != 7:
            continue
        pid, ppid, cpu, rss, elapsed, executable, command = fields
        # Shell launchers may contain the packer command as quoted text.  They are observers of the
        # process, not pack workers, so count only a Python executable actually running the script.
        if "python" not in Path(executable).name.lower():
            continue
        shard_match = re.search(r"--shards\s+(\S+)", command)
        worker_match = re.search(r"--workers\s+(\d+)", command)
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "cpu_percent": float(cpu),
                "rss_mib": int(rss) / 1024,
                "elapsed": elapsed,
                "executable": executable,
                "command": command,
                "active_lane": shard_match.group(1) if shard_match else None,
                "declared_workers": int(worker_match.group(1)) if worker_match else None,
            }
        )
    lease = (
        Path.home()
        / "Library"
        / "Application Support"
        / "Hawking"
        / "GLM52Gravity"
        / "control"
        / "functional"
        / "controller.lease.json"
    )
    pack = (
        Path.home()
        / "Library"
        / "Application Support"
        / "Hawking"
        / "GLM52Gravity"
        / "activation_aware_pack"
    )
    controller = json.loads(lease.read_text()) if lease.is_file() else None
    return {
        "controller_lease": controller,
        "controller_lease_path": str(lease),
        "active_pack_processes": rows,
        "active_process_count": len(rows),
        "declared_worker_count": sum(row["declared_workers"] or 0 for row in rows),
        "packed_object_count": len(list(pack.glob("*.aap"))) if pack.is_dir() else 0,
        "mps_in_use_by_observed_pack_process": False,
        "observation_only": True,
        "signals_sent": 0,
        "processes_modified": 0,
    }


def v1_immutability() -> dict:
    tag_commit = _git("rev-parse", f"{V1_TAG}^{{}}")
    lines = _git("ls-tree", "-r", V1_TAG, "evidence/substrate/v1", "runs/substrate/v1", "proof").splitlines()
    objects = {}
    drift = []
    for line in lines:
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        current = io.ROOT / path
        current_blob = _git("hash-object", path) if current.is_file() else None
        record = {
            "mode": mode,
            "kind": kind,
            "tag_blob": blob,
            "current_blob": current_blob,
            "byte_identical": current_blob == blob,
        }
        objects[path] = record
        if not record["byte_identical"]:
            drift.append(path)
    receipts = {}
    for unit in v1_execution.UNIT_LIST:
        path = v1_execution.UNITS / f"{unit.identity}.json"
        document = json.loads(path.read_text())
        receipts[unit.identity] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "valid": v1_execution.validate_receipt(document),
            "ok": document.get("ok") is True,
        }
    historical_check = historical.verify_all()
    return {
        "schema": "substrate-v1-immutability/v1",
        "terminal_tag": V1_TAG,
        "terminal_commit": tag_commit,
        "object_count": len(objects),
        "objects": objects,
        "drift": drift,
        "byte_identical": not drift,
        "terminal_receipts": receipts,
        "receipt_count": len(receipts),
        "all_receipts_valid": len(receipts) == 19 and all(row["valid"] and row["ok"] for row in receipts.values()),
        "historical_object_count": len(historical_check["objects"]),
        "historical_all_pass": historical_check["all_pass"],
        "verdict": v1.load("SUBSTRATE_NOUS_CLOSURE.json")["verdict"]["classification"],
        "activation": False,
    }


def preflight() -> dict:
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    tag = _git("rev-parse", f"{V1_TAG}^{{}}")
    # The leading two columns are semantic in porcelain output, so do not pass this through _git,
    # whose whitespace normalization is appropriate for object identities but not status records.
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=io.ROOT,
        text=True,
    ).rstrip("\n")
    dirty_paths = [line[3:] for line in dirty.splitlines()]
    preflight_implementation_scope = {
        "src/substrate/cli.py",
        "src/substrate/v2.py",
        "src/substrate/v2io.py",
        "evidence/artifacts/substrate/v2/",
    }
    worktrees = _git("worktree", "list", "--porcelain")
    prs = _json_command(
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
    )
    ci = _json_command(
        [
            "gh",
            "run",
            "list",
            "--repo",
            "joshuahickscorp/substrate",
            "--branch",
            "main",
            "--workflow",
            "Substrate",
            "--limit",
            "1",
            "--json",
            "databaseId,headSha,status,conclusion,url",
        ],
        [],
    )
    v1_check = v1_immutability()
    resources = _resource_snapshot()
    hawking = _hawking_snapshot()
    v2_roots = (io.EVIDENCE, io.RUNS, io.ARTIFACTS, io.CONFIGS)
    preexisting_v2_files = sorted(
        path.relative_to(io.ROOT).as_posix() for root in v2_roots if root.exists() for path in root.rglob("*") if path.is_file()
    )
    preflight_receipts = {
        "evidence/artifacts/substrate/v2/SUBSTRATE_V1_IMMUTABILITY.json",
        "evidence/artifacts/substrate/v2/SUBSTRATE_V2_PREFLIGHT.json",
        "evidence/artifacts/substrate/v2/SUBSTRATE_V2_HAWKING_COEXISTENCE.json",
    }
    unexpected_v2_files = sorted(set(preexisting_v2_files) - preflight_receipts)
    checks = {
        "entry_head_matches_reported_terminal": head == "20ba38ed097b6ccfc46bc0b2a34b82923b873aed",
        "entry_main_matches_origin": head == origin_main,
        "terminal_tag_resolves": tag == head,
        "entry_base_was_clean_terminal_main": head == origin_main == tag,
        "only_preflight_command_implementation_added_since_entry": set(dirty_paths) <= preflight_implementation_scope,
        "one_worktree": worktrees.count("worktree ") == 1,
        "no_v2_run": not unexpected_v2_files,
        "v1_objects_byte_identical": v1_check["byte_identical"],
        "v1_terminal_19_of_19": v1_check["all_receipts_valid"],
        "historical_evidence_intact": v1_check["historical_all_pass"],
        "activation_false": v1.ACTIVATION is False,
        "main_ci_green": bool(ci) and ci[0]["conclusion"] == "success" and ci[0]["headSha"] == head,
        "disk_safe": resources["disk_available_gib"] >= 25,
        "memory_safe": (resources["memory_free_percent"] or 0) >= 10,
        "swap_safe": (resources["swap_free_mib"] or 0) >= 512,
        "hawking_observation_only": hawking["signals_sent"] == 0 and hawking["processes_modified"] == 0,
    }
    return {
        "schema": "substrate-v2-preflight/v1",
        "repository": str(io.ROOT),
        "remote": _git("remote", "get-url", "origin"),
        "entry": {
            "head": head,
            "origin_main": origin_main,
            "terminal_tag_commit": tag,
            "branch": _git("branch", "--show-current"),
            "dirty": dirty.splitlines(),
            "entry_observation": (
                "the clean entry audit was completed before the v2 command implementation was added; "
                "the current diff is restricted to that command implementation"
            ),
            "worktrees": worktrees.splitlines(),
            "open_pull_requests": prs,
            "latest_main_ci": ci,
        },
        "resources": resources,
        "hawking": hawking,
        "preexisting_v2_files": preexisting_v2_files,
        "unexpected_v2_files": unexpected_v2_files,
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "all_pass": all(checks.values()),
        "activation": False,
    }


def seal_preflight() -> dict:
    # Preflight is defined at entry before the rollback tag and branch.  Running it later records the
    # current branch too, but its checks retain the immutable entry identities.
    v1_check = v1_immutability()
    preflight_document = preflight()
    hawking = {
        "schema": "substrate-v2-hawking-coexistence/v1",
        "policy": "observe only; never signal, pause, modify, adopt or restart Hawking",
        "snapshot": preflight_document["hawking"],
        "resources": preflight_document["resources"],
        "cheap_canaries_permitted": preflight_document["checks"]["disk_safe"]
        and preflight_document["checks"]["memory_safe"]
        and preflight_document["checks"]["swap_safe"],
        "principal_requires_rehearsed_interference_check": True,
        "activation": False,
    }
    io.seal("SUBSTRATE_V1_IMMUTABILITY.json", v1_check, artifact=True)
    io.seal("SUBSTRATE_V2_PREFLIGHT.json", preflight_document, artifact=True)
    io.seal("SUBSTRATE_V2_HAWKING_COEXISTENCE.json", hawking, artifact=True)
    return {"preflight": preflight_document, "v1": v1_check, "hawking": hawking}


def freeze_preregistration() -> dict:
    """Seal the scientific premises and split manifest before candidate selection."""
    v1_commit = _git("rev-parse", f"{V1_TAG}^{{}}")
    constitution = {
        "schema": "substrate-v2-scientific-constitution/v1",
        "purpose": (
            "determine whether verified experience organizes one continuing Substrate into a more "
            "capable transferable and self regulating cognitive entity"
        ),
        "v1_dependency": {
            "tag": V1_TAG,
            "commit": v1_commit,
            "classification": "certified_cognitive_scaffold",
            "nulls": ["endogenous allocation", "cross domain continuity", "procedural transfer"],
        },
        "independent_unit": C.STATISTICS["independent_unit"],
        "sesoi": C.SESOI,
        "rules": {
            "threshold_movement": "forbidden after principal launch",
            "tie": "mechanism_null",
            "favorable_below_sesoi": "mechanism_null",
            "target_leakage": "instrumentation_failure",
            "principal_outcome_for_promotion": "forbidden",
            "activation": False,
        },
        "failure_classes": [
            "implementation_defect",
            "instrumentation_failure",
            "invalid_bed",
            "no_oracle_headroom",
            "mechanism_null",
            "mechanism_positive",
            "terminally_gated",
            "unlicensed",
        ],
        "frozen_before_principal": [
            "hypotheses",
            "primary endpoints",
            "independent unit",
            "task generators",
            "splits",
            "seeds",
            "controls",
            "baselines",
            "budgets",
            "compute prices",
            "SESOI",
            "confidence procedure",
            "multiple comparison policy",
            "stop rules",
            "futility rules",
            "failure handling",
            "claim criteria",
        ],
        "activation": False,
    }
    hypothesis_graph = {
        "schema": "substrate-v2-hypothesis-graph/v1",
        "hypotheses": C.HYPOTHESES,
        "dependencies": {
            "persistent_developmental_cognition": [
                "grounded_closed_loop",
                "unity_under_conflict",
                "H_D1",
                "H_D2",
                "identity_continuity",
                "retention",
            ],
            "reflective_cognitive_organization": [
                "persistent_developmental_cognition",
                "H_D3",
                "world_self_control_value",
                "H_D5",
            ],
            "functional_or_proto_nous_candidate": [
                "reflective_cognitive_organization",
                "all_six_closure_gates",
                "multiple_positive_transfer_pairs",
                "H_D4",
                "negative_transfer_clean",
                "body_change_continuity",
                "interruption_recovery",
                "independent_verification",
                "zero_mutation_survivors",
                "terminal_campaign_complete",
            ],
        },
        "activation": False,
    }
    classification = {
        "schema": "substrate-v2-classification-authority/v1",
        "ordered_levels": C.CLAIM_BOUNDARY["permitted_classifications"],
        "criteria": hypothesis_graph["dependencies"],
        "cheap_canaries_cannot_change_classification": True,
        "terminal_principal_evidence_required_above_scaffold": True,
        "maximum": C.CLAIM_BOUNDARY["maximum"],
        "activation": False,
    }
    split_authority = {
        "schema": "substrate-v2-split-authority/v1",
        "splits": {name: list(values) for name, values in C.SPLITS.items()},
        "roles": {
            "development": "mechanism construction and debugging",
            "admission": "one bounded candidate selection and cheap causal gate",
            "principal": "frozen terminal campaign",
            "replication": "independent final recomputation and secondary body seed family",
        },
        "no_seed_crosses_splits": len({seed for values in C.SPLITS.values() for seed in values})
        == sum(len(values) for values in C.SPLITS.values()),
        "source_episode_identity": "split seed domain phase index",
        "promotion_forbidden_from": ["principal", "replication"],
        "manifest_commit_rule": "commit before final mechanism selection",
        "activation": False,
    }
    generator = {
        "schema": "substrate-v2-generator-authority/v1",
        "callable": "substrate.v2fabric.generate_task",
        "operations": list(v2fabric.OPERATIONS),
        "target_timing": "private until Task.reveal is called after proposal commitment",
        "identity_components": ["split", "seed", "domain", "phase", "index"],
        "deterministic": True,
        "surface_labels_reveal_rule": False,
        "answer_leakage_test": "target key value and digest absent from public serialization",
        "configuration_digest": C.configuration()["configuration_digest"],
        "activation": False,
    }
    bed = v2fabric.screen()
    documents = {
        "SUBSTRATE_V2_SCIENTIFIC_CONSTITUTION.json": constitution,
        "SUBSTRATE_V2_HYPOTHESIS_GRAPH.json": hypothesis_graph,
        "SUBSTRATE_V2_CLASSIFICATION_AUTHORITY.json": classification,
        "SUBSTRATE_V2_CLAIM_BOUNDARY.json": {
            "schema": "substrate-v2-claim-boundary/v1",
            **C.CLAIM_BOUNDARY,
        },
        "SUBSTRATE_V2_DOMAIN_CATALOG.json": {
            "schema": "substrate-v2-domain-catalog/v1",
            "domains": C.DOMAIN_CATALOG,
            "domain_count": len(C.DOMAIN_CATALOG),
            "activation": False,
        },
        "SUBSTRATE_V2_TRANSFER_GRAPH.json": {
            "schema": "substrate-v2-transfer-graph/v1",
            **C.TRANSFER_GRAPH,
            "positive_pair_count": len(C.TRANSFER_GRAPH["positive"]),
            "negative_pair_count": len(C.TRANSFER_GRAPH["negative"]),
            "activation": False,
        },
        "SUBSTRATE_V2_SPLIT_AUTHORITY.json": split_authority,
        "SUBSTRATE_V2_GENERATOR_AUTHORITY.json": generator,
        "SUBSTRATE_V2_BED_SCREEN.json": bed,
        "SUBSTRATE_V2_CANDIDATE_LADDER.json": {
            "schema": "substrate-v2-candidate-ladder/v1",
            **C.CANDIDATE_LADDER,
            "activation": False,
        },
        "SUBSTRATE_V2_STATISTICAL_AUTHORITY.json": {
            "schema": "substrate-v2-statistical-authority/v1",
            **C.STATISTICS,
            "activation": False,
        },
    }
    io.config_json("frozen_configuration.json", C.configuration())
    for name, document in documents.items():
        io.seal(name, document)
    return {
        "sealed": sorted(documents),
        "configuration_digest": C.configuration()["configuration_digest"],
        "bed_valid": bed["all_valid"],
        "split_disjoint": split_authority["no_seed_crosses_splits"],
        "activation": False,
    }


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "status"
    if command == "preflight":
        documents = seal_preflight()
        summary = {
            "all_pass": documents["preflight"]["all_pass"],
            "failed": documents["preflight"]["failed"],
            "v1_objects": documents["v1"]["object_count"],
            "v1_receipts": documents["v1"]["receipt_count"],
            "hawking_processes": documents["hawking"]["snapshot"]["active_process_count"],
            "activation": False,
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(0 if summary["all_pass"] else 1)
    if command == "canaries":
        from substrate import v2canary

        documents = v2canary.run()
        summary = {
            "passed": documents["evidence"]["passed"],
            "total": documents["evidence"]["total"],
            "all_pass": documents["evidence"]["all_pass"],
            "all_terminal": documents["evidence"]["all_terminal"],
            "nonpositive": documents["evidence"]["nonpositive"],
            "rehearsal_licensed": documents["admission"]["rehearsal_licensed"],
            "principal_execution_licensed": documents["admission"]["principal_execution_licensed"],
            "activation": False,
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(0 if summary["all_terminal"] and summary["rehearsal_licensed"] else 1)
    if command == "rehearse":
        from substrate import v2rehearsal

        documents = v2rehearsal.run()
        summary = {
            "rehearsal": documents["rehearsal"]["all_pass"],
            "failure_matrix": documents["failures"]["all_pass"],
            "resource_safe": documents["resources"]["all_safe"],
            "selected_workers": documents["resources"]["selected_workers"],
            "principal_execution_licensed": documents["admission"]["principal_execution_licensed"],
            "activation": False,
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(0 if summary["principal_execution_licensed"] else 1)
    if command == "run":
        from substrate import v2principal

        document = v2principal.run()
        print(json.dumps(document, indent=2))
        raise SystemExit(0 if document["status"]["remaining"] == 0 and not document["status"]["invalid"] else 1)
    if command == "status":
        from substrate import v2principal

        print(json.dumps(v2principal.status(), indent=2))
        return
    if command == "stop":
        print(json.dumps({"stopped": True, "stop_switch": str(io.stop()), "activation": False}, indent=2))
        return
    if command == "resume":
        from substrate import v2principal

        io.resume()
        document = v2principal.run()
        print(json.dumps(document, indent=2))
        raise SystemExit(0 if document["status"]["remaining"] == 0 and not document["status"]["invalid"] else 1)
    if command == "verify":
        from substrate import v2verify

        document = v2verify.verify()
        print(json.dumps(document, indent=2))
        raise SystemExit(0 if document["all_pass"] else 1)
    raise SystemExit(f"unknown v2 command {command!r}")
