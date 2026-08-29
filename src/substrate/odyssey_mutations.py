"""Isolated, source-bound control-plane mutation checks for Odyssey G12.

This module intentionally exercises *only* admission, custody, integrity, and
execution-boundary code.  It never invokes a model, real task panel, external
corpus/tool, or evaluator, reads evaluator-owned answers, or produces
scientific evidence.  It may materialize disposable synthetic candidate
fixtures solely to exercise the admission validators.  Each fault is injected
into a fresh copy of an exact clean clone, then evaluated by the clone's real
control-plane code.  A passing report therefore says that the listed synthetic
faults were rejected by this particular frozen source tree; it says nothing
about scientific outcomes, model quality, or task validity.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from substrate import odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("docs/plans/substrate/tangible_next_launch")
SCHEMA = "SUBSTRATE_ODYSSEY_MUTATION_REPORT/v1"
RUNNER_SOURCE_KEY = "odyssey_mutations"
FIXTURE_MARKER = ".odyssey-g12-isolated-fixture.json"


class Refused(RuntimeError):
    """The mutation suite cannot truthfully issue its control-plane receipt."""


@dataclass(frozen=True)
class Mutation:
    """One synthetic control-plane fault and the guard that must reject it."""

    identifier: str
    category: str
    target: str
    description: str
    application: str


MUTATIONS = (
    Mutation(
        "answer_leakage",
        "custody_boundary",
        "candidate-visible task manifest",
        "Inject evaluator-only answer material into a candidate-visible task.",
        "g03_subject",
    ),
    Mutation(
        "shared_mutable_cache",
        "lane_isolation",
        "G06 private-state layout",
        "Alias candidate and control mutable cache roots inside one paired cell.",
        "g06_subject",
    ),
    Mutation(
        "cross_lane_model_context",
        "lane_isolation",
        "G06 model-context layout",
        "Alias candidate and control model-context roots across a paired cell.",
        "g06_subject",
    ),
    Mutation(
        "duplicate_writer",
        "single_writer",
        "Odyssey worker lock",
        "Hold a worker lock while a second worker admission attempts the same run root.",
        "worker_lock",
    ),
    Mutation(
        "forged_receipt", "receipt_integrity", "Odyssey adapter receipt", "Forge the request binding in an otherwise shaped adapter receipt.", "adapter_receipt"
    ),
    Mutation("missing_checkpoint", "durability_boundary", "G09 recovery rehearsal", "Remove a required delta checkpoint from a recovery chain.", "g09_subject"),
    Mutation(
        "wrong_source_digest",
        "source_integrity",
        "src/substrate/odyssey_worker.py",
        "Change a frozen worker source file after its source map was sealed.",
        "source_drift",
    ),
    Mutation(
        "result_dependent_task_selection",
        "task_selection",
        "candidate-visible task manifest",
        "Inject result-dependent task selection metadata before candidate execution.",
        "g03_subject",
    ),
    Mutation(
        "control_under_resourcing",
        "resource_parity",
        "G06 paired-cell resource map",
        "Reduce the control arm's resource allocation below the candidate allocation.",
        "g06_subject",
    ),
    Mutation(
        "pseudo_replication_analysis",
        "statistical_integrity",
        "Odyssey hardened design",
        "Substitute event-level scoring for the paired-frontier primary unit.",
        "pseudoreplication",
    ),
    Mutation(
        "full_program_frontier_omission",
        "schedule_integrity",
        "full-program worker authority",
        "Attempt to admit a seven-frontier full Odyssey schedule.",
        "worker_contract",
    ),
    Mutation(
        "full_program_width_reduction",
        "schedule_integrity",
        "full-program worker authority",
        "Attempt to admit full Odyssey at width seven rather than calibrated width eight.",
        "worker_contract",
    ),
    Mutation(
        "full_program_checkpoint_cadence_drift",
        "durability_boundary",
        "full-program worker authority",
        "Alter the sealed two-hour / twelve-hour checkpoint cadence.",
        "worker_contract",
    ),
    Mutation(
        "candidate_manifest_digest_tamper",
        "manifest_integrity",
        "candidate-visible manifest digest",
        "Supply an incorrect digest for an otherwise valid candidate manifest.",
        "manifest_digest",
    ),
    Mutation(
        "authority_evaluator_input_injection",
        "custody_boundary",
        "test worker authority",
        "Inject evaluator-only authority material before worker admission.",
        "worker_authority",
    ),
    Mutation("authority_self_digest_forgery", "authority_integrity", "test worker authority", "Forge a sealed authority self-digest.", "worker_authority"),
)
MUTATION_BY_ID = {mutation.identifier: mutation for mutation in MUTATIONS}
SOURCE_DRIFT_IDS = frozenset(mutation.identifier for mutation in MUTATIONS if mutation.application == "source_drift")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    unsigned = dict(value)
    claimed = unsigned.pop("sha256", None)
    if require_digest and (not isinstance(claimed, str) or claimed != digest(unsigned)):
        raise Refused(f"{path} has an invalid self-digest")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    if path.exists():
        raise Refused(f"refusing to overwrite existing mutation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise Refused(f"refusing to replace existing temporary mutation report: {temporary}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    if not _inside(root, path):
        raise Refused(f"path escapes repository root: {path}")
    return str(path.resolve().relative_to(root.resolve()))


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=False)


def _git_head(root: Path) -> str:
    completed = _git(root, "rev-parse", "HEAD")
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise Refused("cannot resolve current git HEAD for mutation receipt")
    return head


def _implementation_paths(root: Path) -> dict[str, Path]:
    paths = odyssey_transition.implementation_inputs(root)
    if not isinstance(paths, dict) or not paths:
        raise Refused("Odyssey transition has no implementation source map")
    if RUNNER_SOURCE_KEY not in paths:
        raise Refused("frozen implementation map must bind odyssey_mutations before G12 can run")
    return paths


def _frozen_source_binding(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    frozen_path = root / PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen = _read_json(frozen_path, require_digest=True)
    if frozen.get("schema") != "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1":
        raise Refused("unexpected Odyssey frozen-build schema")
    implementation = frozen.get("implementation_sha256")
    inputs = frozen.get("input_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise Refused("frozen build lacks an implementation source map")
    if not isinstance(inputs, dict) or not inputs:
        raise Refused("frozen build lacks a protocol input source map")
    if RUNNER_SOURCE_KEY not in implementation:
        raise Refused("frozen implementation map must bind odyssey_mutations before G12 can run")
    paths = _implementation_paths(root)
    if set(implementation) != set(paths):
        raise Refused("frozen implementation map does not exactly match the Odyssey source map")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise Refused(f"frozen implementation source is missing: {name}")
        observed[name] = odyssey_transition.canonical_source_digest(path)
    if observed != implementation:
        raise Refused("current Odyssey implementation drifts from the frozen source map")
    input_paths = odyssey_transition.build_inputs(root)
    if not isinstance(input_paths, dict) or set(inputs) != set(input_paths):
        raise Refused("frozen input map does not exactly match the Odyssey protocol input map")
    observed_inputs: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise Refused(f"frozen Odyssey protocol input is missing: {name}")
        observed_inputs[name] = file_digest(path)
    if observed_inputs != inputs:
        raise Refused("current Odyssey protocol inputs drift from the frozen input map")
    runner_path = paths[RUNNER_SOURCE_KEY]
    expected_runner = implementation.get(RUNNER_SOURCE_KEY)
    if (
        runner_path.resolve() != (root / "src/substrate/odyssey_mutations.py").resolve()
        or expected_runner != odyssey_transition.canonical_source_digest(runner_path)
    ):
        raise Refused("mutation runner is not exactly source-bound to this frozen build")
    return frozen, paths


def _assert_scoped_tree_clean(root: Path, *path_maps: dict[str, Path]) -> None:
    scoped = {_relative(root, path) for paths in path_maps for path in paths.values()}
    scoped.add(str(PLAN / "ODYSSEY_FROZEN_BUILD.json"))
    completed = _git(root, "status", "--porcelain", "--", *sorted(scoped))
    if completed.returncode != 0:
        raise Refused("cannot inspect Odyssey mutation source status")
    if completed.stdout.strip():
        raise Refused("Odyssey mutation sources and frozen build must be committed before G12 can run")


def _clone_at_head(root: Path, destination: Path, source_commit: str) -> dict[str, Any]:
    clone = destination / "substrate"
    clone_result = subprocess.run(
        ["git", "clone", "--quiet", "--no-local", "--no-hardlinks", str(root), str(clone)],
        cwd=destination,
        capture_output=True,
        text=True,
        check=False,
    )
    if clone_result.returncode != 0:
        detail = (clone_result.stderr or clone_result.stdout).strip()
        raise Refused(f"isolated git clone failed: {detail or 'unknown git error'}")
    checkout_result = _git(clone, "checkout", "--quiet", "--detach", source_commit)
    if checkout_result.returncode != 0:
        detail = (checkout_result.stderr or checkout_result.stdout).strip()
        raise Refused(f"isolated clone checkout failed: {detail or 'unknown git error'}")
    clone_head = _git_head(clone)
    if clone_head != source_commit:
        raise Refused("isolated clone did not reproduce the requested source commit")
    return {
        "root": clone,
        "receipt": {
            "method": "git_clone_no_local_no_hardlinks",
            "source_commit": source_commit,
            "clone_head": clone_head,
            "exact_commit_checkout": True,
        },
    }


def _append_inert_marker(path: Path, mutation_id: str) -> None:
    if not path.is_file():
        raise Refused(f"mutation target does not exist: {path}")
    marker = f"# isolated-odyssey-mutation: {mutation_id}\n"
    source = path.read_text(encoding="utf-8")
    if marker in source:
        raise Refused(f"mutation marker already present: {mutation_id}")
    path.write_text(source.rstrip() + "\n\n" + marker, encoding="utf-8")


def _apply_source_mutation(root: Path, mutation: Mutation, *, variant: str) -> None:
    if variant != "mutant":
        return
    if mutation.identifier not in SOURCE_DRIFT_IDS:
        return
    target = root / mutation.target
    _append_inert_marker(target, mutation.identifier)


def _mark_isolated_fixture(root: Path, mutation: Mutation, *, variant: str) -> None:
    """Make the child refuse accidental execution in a real worktree."""
    marker = {
        "schema": "SUBSTRATE_ODYSSEY_G12_ISOLATED_FIXTURE/v1",
        "mutation": mutation.identifier,
        "variant": variant,
        "synthetic_control_plane_only": True,
        "activation": False,
    }
    marker["sha256"] = digest(marker)
    path = root / FIXTURE_MARKER
    if path.exists():
        raise Refused("isolated mutation fixture marker already exists")
    path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")


def _require_isolated_fixture(root: Path, mutation_id: str, variant: str) -> None:
    path = root / FIXTURE_MARKER
    if not path.is_file():
        raise Refused("mutation probe requires an isolated fixture marker")
    marker = _read_json(path, require_digest=True)
    if (
        marker.get("schema") != "SUBSTRATE_ODYSSEY_G12_ISOLATED_FIXTURE/v1"
        or marker.get("mutation") != mutation_id
        or marker.get("variant") != variant
        or marker.get("synthetic_control_plane_only") is not True
        or marker.get("activation") is not False
    ):
        raise Refused("mutation probe requires its matching isolated fixture marker")


def _child_result(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _self_sealed(value: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical self-digested control-plane fixture document."""
    body = dict(value)
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def _write_fixture_document(root: Path, relative: str, value: dict[str, Any]) -> dict[str, str]:
    """Write one self-digested fixture and return its content-addressed ref."""
    path = root / relative
    if not _inside(root, path):
        raise Refused(f"fixture document escapes root: {relative}")
    if path.exists():
        raise Refused(f"fixture document already exists: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _self_sealed(value)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return {"path": _relative(root, path), "sha256": file_digest(path)}


def _read_fixture_document(root: Path, reference: dict[str, str]) -> dict[str, Any]:
    return _read_json(root / reference["path"], require_digest=True)


def _frozen_binding_fields(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    return {
        "program": PROGRAM,
        "status": "pass",
        "activation": False,
        "external_activation": False,
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": _git_head(root),
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
    }


def _g03_subject(root: Path, frozen: dict[str, Any], *, mutation_id: str, variant: str) -> dict[str, Any]:
    """Materialize a synthetic, candidate-only G03 subject and validate it."""
    from substrate import odyssey_authority as authority
    from substrate import odyssey_task_bank as task_bank

    manifests: list[dict[str, Any]] = []
    selection_rows: list[dict[str, str]] = []
    task_count = authority.FULL_TASKS_PER_FRONTIER
    # G03 intentionally treats every candidate-visible path as part of the
    # custody boundary.  Do not place a semantic mutation label such as
    # ``answer_leakage`` in those paths: that would make the clean fixture
    # itself look like an answer channel rather than test the intended guard.
    fixture_token = digest({"mutation": mutation_id})[:16]
    for frontier in authority.FRONTIER_IDS:
        base = f"g12-fixtures/g03/case-{fixture_token}/{variant}/{frontier}"
        asset = _write_fixture_document(
            root,
            f"{base}/source-packet.json",
            {
                "schema": "SUBSTRATE_ODYSSEY_G12_SYNTHETIC_SOURCE_PACKET/v1",
                "program": PROGRAM,
                "activation": False,
                "frontier": frontier,
                "control_plane_fixture": True,
                "scientific_evidence": False,
            },
        )
        assets = [{"path": asset["path"], "sha256": asset["sha256"], "role": "synthetic_control_plane_packet", "read_only": True}]
        source_bundle = {
            "assets": assets,
            "selection_sha256": authority.digest({"frontier": frontier, "assets": assets}),
        }
        seed = f"g12-{mutation_id}-{frontier}"
        commitment = task_bank._digest({"seed": seed})
        candidate, _evaluator = task_bank.materialize(commitment, seed, frontier, task_count)
        candidate["source_bundle"] = source_bundle
        if variant == "mutant" and frontier == "A":
            if mutation_id == "answer_leakage":
                candidate["tasks"][0]["expected_answer"] = "synthetic-forbidden-answer"
            elif mutation_id == "result_dependent_task_selection":
                candidate["tasks"][0]["result_dependent_task_selection"] = "synthetic-forbidden-result-channel"
        candidate.pop("sha256", None)
        candidate["sha256"] = task_bank._digest(candidate)
        manifest_path = root / f"{base}/candidate-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
        task_ids = [task["task_id"] for task in candidate["tasks"]]
        row = {
            "id": frontier,
            "frontier": frontier,
            "path": _relative(root, manifest_path),
            "file_sha256": file_digest(manifest_path),
            "schema": "SUBSTRATE_ODYSSEY_CANDIDATE_TASK_MANIFEST/v1",
            "seed_commitment": commitment,
            "task_count": len(task_ids),
            "task_ids_sha256": authority.digest({"task_ids": task_ids}),
            "source_selection_sha256": source_bundle["selection_sha256"],
            "source_bundle_sha256": authority.digest(source_bundle),
        }
        manifests.append(row)
        selection_rows.append({"id": frontier, "source_selection_sha256": source_bundle["selection_sha256"]})
    subject = {
        "schema": "SUBSTRATE_ODYSSEY_FRONTIER_MANIFEST_SET/v1",
        **_frozen_binding_fields(root, frozen),
        "all_pass": True,
        "manifest_count": len(manifests),
        "task_bank_generator_sha256": frozen["implementation_sha256"]["task_bank_generator"],
        "frontier_contract_sha256": frozen["input_sha256"]["frontier_contract"],
        "task_bank_sha256": frozen["input_sha256"]["task_bank"],
        "rendered_build_index_sha256": frozen["input_sha256"]["rendered_build_index"],
        "source_selection_sha256": authority.digest({"source_selections": selection_rows}),
        "manifests": manifests,
        "checks": {
            "frozen_build_bound": True,
            "source_maps_bound": True,
            "candidate_manifests_structurally_safe": True,
            "frontier_set_exact": True,
            "scheduled_task_count_exact": True,
            "source_bundle_bound": True,
        },
    }
    return _self_sealed(subject)


def _g06_cell(root: Path, *, mutation_id: str, variant: str, width: int, repetition: int, frontier: str) -> dict[str, Any]:
    """Build one executable, non-scientific G06 private-layout observation."""
    base = f"g12-fixtures/g06/{mutation_id}/{variant}/{width}x/{repetition}/{frontier}"
    fields = (
        "candidate_root",
        "control_root",
        "candidate_event_ledger",
        "control_event_ledger",
        "candidate_checkpoint_root",
        "control_checkpoint_root",
        "candidate_mutable_state_root",
        "control_mutable_state_root",
        "candidate_model_context_root",
        "control_model_context_root",
    )
    row = {"id": frontier}
    for field in fields:
        path = root / base / field
        path.mkdir(parents=True, exist_ok=True)
        row[field] = _relative(root, path)
    if variant == "mutant" and width == 1 and repetition == 1 and frontier == "A":
        if mutation_id == "shared_mutable_cache":
            row["control_mutable_state_root"] = row["candidate_mutable_state_root"]
        elif mutation_id == "cross_lane_model_context":
            row["control_model_context_root"] = row["candidate_model_context_root"]
        elif mutation_id == "control_under_resourcing":
            pass
    resources = {
        "allowed_observations": ["candidate-visible-synthetic-input"],
        "models": ["synthetic-control-plane-adapter"],
        "tools": ["synthetic-receipt-writer"],
        "token_budget": 64,
        "compute_ceiling": 1,
        "storage_ceiling": 1,
        "wall_time_seconds": 1800,
    }
    control = copy.deepcopy(resources)
    if variant == "mutant" and mutation_id == "control_under_resourcing" and width == 1 and repetition == 1 and frontier == "A":
        control["compute_ceiling"] = 0
    row["resource_parity"] = {"candidate": resources, "control": control}
    row.update(
        {
            "task_binding": {
                "manifest_path": f"{base}/candidate-manifest.json",
                "manifest_sha256": digest({"g06": mutation_id, "frontier": frontier, "kind": "manifest"}),
                "task_index": 0,
                "task_id": f"synthetic-{frontier}-retrieval",
                "task_sha256": digest({"g06": mutation_id, "frontier": frontier, "kind": "retrieval-task"}),
            },
            "model_call_count": 2,
            "source_bundle_guard_calls": 2,
            "active_work_seconds": 1.0,
            "deadline_met": True,
        }
    )
    return row


def _g06_adapter_receipt(
    root: Path,
    *,
    relative_root: str,
    role: str,
    frontier: str,
    task: dict[str, Any],
    manifest_sha256: str,
    authority_sha256: str,
    run_id: str,
    model: str,
    adapter_sha256: str,
) -> dict[str, str]:
    """Write a real-shaped synthetic arm receipt for the G06 validator."""
    request_sha256 = digest({"g12": relative_root, "role": role, "kind": "request"})
    usage = {
        "prompt_eval_count": 1,
        "eval_count": 1,
        "total_duration_ns": 1,
        "load_duration_ns": 0,
        "eval_duration_ns": 1,
    }
    output = _write_fixture_document(
        root,
        f"{relative_root}/{role}-output.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_ARM_OUTPUT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": frontier,
            "role": role,
            "cycle": 0,
            "phase": "retrieval",
            "task_id": task["task_id"],
            "request_sha256": request_sha256,
            "candidate_manifest_sha256": manifest_sha256,
            "adapter_sha256": adapter_sha256,
            "model": model,
            "prompt_sha256": digest({"g12": relative_root, "role": role, "kind": "prompt"}),
            "response": {"synthetic_response": role},
            "resource_usage": usage,
        },
    )
    output_document = _read_fixture_document(root, output)
    return _write_fixture_document(
        root,
        f"{relative_root}/{role}-receipt.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": frontier,
            "role": role,
            "cycle": 0,
            "phase": "retrieval",
            "task_id": task["task_id"],
            "candidate_manifest_sha256": manifest_sha256,
            "request_sha256": request_sha256,
            "elapsed_seconds": 1.0,
            "adapter_sha256": adapter_sha256,
            "model": {"id": model, "endpoint": "http://127.0.0.1:11434"},
            "output_artifacts": [output],
            "response_sha256": output_document["sha256"],
            "state_before_sha256": digest({"g12": relative_root, "role": role, "state": "before"}),
            "state_after_sha256": digest({"g12": relative_root, "role": role, "state": "after"}),
            "state_change": {
                "mode": "flat_exact_associative_monolith" if role == "candidate" else "append_only_history_retrieval"
            },
            "resource_usage": usage,
        },
    )


def _g06_phase_boundary(
    root: Path,
    *,
    relative_root: str,
    authority_sha256: str,
    run_id: str,
    cells: list[dict[str, Any]],
) -> dict[str, str]:
    """Write the G06 parent-side trace/checkpoint/state chain for a fixture."""
    boundary_root = root / relative_root
    trace_path = boundary_root / "EVENTS.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    chain = ""
    trace_rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda item: str(item["id"])):
        event = {
            "schema": "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": cell["id"],
            "cycle": 0,
            "phase": "retrieval",
            "task_id": cell["task_binding"]["task_id"],
            "candidate_receipt_sha256": cell["candidate_receipt"]["sha256"],
            "control_receipt_sha256": cell["control_receipt"]["sha256"],
            "candidate_elapsed_seconds": 1.0,
            "control_elapsed_seconds": 1.0,
            "source_bundle_guard_calls": 2,
            "previous_event_sha256": chain,
        }
        event["event_sha256"] = digest(event)
        chain = event["event_sha256"]
        trace_rows.append(event)
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")
    checkpoint = _write_fixture_document(
        root,
        f"{relative_root}/checkpoints/delta-001.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "kind": "delta",
            "cycle": 0,
            "completed_phase_count": 1,
            "completed_paired_events": len(cells),
            "event_chain_sha256": chain,
            "parent_checkpoint_sha256": "",
        },
    )
    checkpoint_document = _read_fixture_document(root, checkpoint)
    return _write_fixture_document(
        root,
        f"{relative_root}/STATE.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "completed_phase_count": 1,
            "total_phase_count": 1,
            "completed_paired_events": len(cells),
            "event_chain_sha256": chain,
            "checkpoint_sha256": checkpoint_document["sha256"],
            "checkpoint_count": 1,
            "complete": False,
            "elapsed_seconds": 1.0,
            "broker_hold_seconds": 0.0,
        },
    )


def _g06_g02_subject(root: Path, frozen: dict[str, Any], *, base: str) -> dict[str, str]:
    """Materialize a valid synthetic G02 chain for isolated G06 mutations.

    The mutation suite must exercise the real G06 validator rather than rely
    on an unbound stand-in for its G02 selection.  This is still a local
    control-plane fixture: it never contacts a model or claims science.
    """
    from substrate import odyssey_authority as authority

    template_path = root / PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json"
    template = _read_json(template_path)
    aliases = template.get("candidate_aliases")
    cases = template.get("case_set")
    if not isinstance(aliases, list) or base not in aliases or not isinstance(cases, list):
        raise Refused("G12 synthetic G02 cannot read the frozen public model cohort")
    runtime_sha256 = digest({"g12": "synthetic-runtime"})

    def model_pin(name: str) -> dict[str, str]:
        revision_digest = digest({"model": name})
        return {
            "id": name,
            "revision": f"fixture:{revision_digest[:16]}",
            "weight_sha256": digest({"model": name, "kind": "weights"}),
            "tokenizer_sha256": digest({"model": name, "kind": "tokenizer"}),
            "runtime_sha256": runtime_sha256,
            "quantization": "fixture-q4",
        }

    candidates: list[dict[str, Any]] = []
    selected = model_pin(base)
    for name in aliases:
        if not isinstance(name, str):
            raise Refused("G12 synthetic G02 has a non-text public model alias")
        eligible = name == base
        candidate_cases = []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                raise Refused("G12 synthetic G02 has an invalid public case")
            candidate_cases.append(
                {
                    "id": case["id"],
                    "response_sha256": digest({"g12": name, "case": case["id"], "kind": "response"}) if eligible else None,
                    "answer": "fixture-final" if eligible else None,
                    "passed": eligible,
                    "latency_ms": 1.0 if eligible else None,
                }
            )
        candidates.append(
            {
                "base_model": model_pin(name),
                "model_size_bytes": 1,
                "service_peak_bytes": 1,
                "swap_pageout_delta_bytes": 0,
                "width_eight": {"requests": 8, "completed": 8 if eligible else 0, "all_responses_valid": eligible},
                "canary": {
                    "total": len(cases),
                    "passed": len(cases) if eligible else 0,
                    "median_latency_ms": 1.0 if eligible else None,
                    "case_results": candidate_cases,
                },
                "errors": [] if eligible else ["synthetic non-selected public body"],
                "eligible": eligible,
            }
        )
    canary = {
        "schema": "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_CANARY/v1",
        "program": PROGRAM,
        "status": "pass",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "scientific_evidence": False,
        "evidence_scope": "frozen_public_model_selection_canaries_only",
        "completed_at": "2026-08-03T00:00:00Z",
        "frozen_build_sha256": frozen["sha256"],
        "canary_template_sha256": file_digest(template_path),
        "runtime": {"id": "fixture-ollama", "version": "fixture-v1", "sha256": runtime_sha256},
        "model_service_cap_bytes": template["model_service_cap_bytes"],
        "required_concurrent_clients": 8,
        "selection_rule": template["selection_rule"],
        "neutral_organ_prompt_sha256": hashlib.sha256(template["neutral_organ_prompt"].encode("utf-8")).hexdigest(),
        "reasoning_effort_policy": template["reasoning_effort_policy"],
        "conversation_policy": template["conversation_policy"],
        "max_output_tokens": template["max_output_tokens"],
        "hidden_seed_commitments_materialized": False,
        "candidates": candidates,
        "selected_base_model": selected,
        "checks": {
            "frozen_template_bound": True,
            "public_only": True,
            "all_candidates_accounted": True,
            "all_configured_candidates_eligible": False,
            "no_hidden_seed_commitments": True,
            "selection_rule_applied": True,
            "selected_candidate_eligible": True,
            "shared_service_footprint_within_24_gib": True,
            "no_swap": True,
            "width_eight_admitted": True,
        },
        "all_pass": True,
        "non_claims": ["Synthetic isolated G12 control-plane fixture only."],
    }
    selection_token = digest({"mutation-g02": base})[:16]
    base_path = f"g12-fixtures/g06/{selection_token}/selection"
    canary_ref = _write_fixture_document(root, f"{base_path}/public-canary.json", canary)
    adapter_sha256 = frozen["implementation_sha256"]["odyssey_arms"]

    def arm_pin(identifier: str) -> dict[str, str]:
        return {
            "id": identifier,
            "revision": selected["revision"],
            "artifact_sha256": selected["weight_sha256"],
            "adapter_sha256": adapter_sha256,
        }

    g02 = {
        "schema": "SUBSTRATE_ODYSSEY_ARM_SELECTION/v1",
        **_frozen_binding_fields(root, frozen),
        "unqualified_nous": False,
        "selection_id": "g12-synthetic-pre-outcome-selection",
        "public_model_canary": canary_ref,
        "base_model": selected,
        "candidate": {**arm_pin("g12-candidate"), "treatment_id": "g12-synthetic-treatment"},
        "controls_by_frontier": {frontier: arm_pin(f"g12-control-{frontier}") for frontier in authority.FRONTIER_IDS},
        "parity_by_frontier": {
            frontier: {field: True for field in authority.PARITY_FIELDS} for frontier in authority.FRONTIER_IDS
        },
        "selection_checks": {
            "pre_outcome_selection": True,
            "public_canary_receipt_reviewed": True,
            "one_shared_base_body_verified": True,
            "candidate_pin_complete": True,
            "control_pins_complete": True,
            "candidate_control_difference_declared": True,
            "parity_reviewed": True,
        },
    }
    return _write_fixture_document(root, f"{base_path}/G02.subject.json", g02)


def _g06_subject(root: Path, frozen: dict[str, Any], *, mutation_id: str, variant: str) -> dict[str, Any]:
    from substrate import odyssey_authority as authority

    required_checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "receipt_invariant": True,
        "no_memory_threshold_breach": True,
        "no_critical_pressure": True,
        "no_unexpected_swap_or_pageout_increase": True,
        "io_latency_within_sealed_limit": True,
        "slowdown_within_sealed_limit": True,
        "distinct_run_roots": True,
        "no_shared_writable_evaluator_or_data_root": True,
        "record_cpu_memory_io": True,
        "strict_dispatch_deadline_met": True,
        "production_paired_adapters_complete": True,
        "source_bundle_revalidation_complete": True,
        "parent_global_dwell_complete": True,
    }
    base = f"g12-fixtures/g06/{mutation_id}/{variant}"
    g02_ref = _g06_g02_subject(root, frozen, base="gpt-oss:20b")
    g03_document = _g03_subject(root, frozen, mutation_id=mutation_id, variant=variant)
    g03_ref = _write_fixture_document(root, f"{base}/G03.subject.json", g03_document)
    g02_document = _read_fixture_document(root, g02_ref)
    selected_base = g02_document["base_model"]
    adapter_sha256 = g02_document["candidate"]["adapter_sha256"]
    manifest_by_frontier = {row["id"]: row for row in g03_document["manifests"]}
    manifest_bindings = [
        {
            "id": frontier,
            "path": manifest_by_frontier[frontier]["path"],
            "sha256": manifest_by_frontier[frontier]["file_sha256"],
        }
        for frontier in authority.FRONTIER_IDS
    ]
    retrieval_tasks = {
        frontier: _read_json(root / manifest_by_frontier[frontier]["path"], require_digest=True)["tasks"][0]
        for frontier in authority.FRONTIER_IDS
    }
    phase_harness = {
        "schema": "SUBSTRATE_ODYSSEY_G06_REAL_PHASE_HARNESS/v1",
        "measurement_basis": "active_paired_dispatch_wall_with_deadline_guard",
        "full_phase_seconds": 1800,
        "strict_dispatch_budget_seconds": 150,
        "scale_factor": 12,
        "phase_boundary_guard_interval_seconds": 30,
        "paired_adapter_dispatches_per_cell": 2,
        "source_bundle_pre_dispatch_revalidation": True,
        "scheduling_mode": "initial_release_only;per_frontier_candidate_then_control;no_global_role_barrier;parent_global_dwell",
        "worker_sha256": frozen["implementation_sha256"]["odyssey_worker"],
        "adapter_sha256": adapter_sha256,
        "model": selected_base["id"],
        "max_output_tokens": 64,
        "g03_manifest_bindings": manifest_bindings,
    }
    phase_harness["dispatch_contract_sha256"] = digest(phase_harness)
    phase_harness.update(
        {
            "g02_subject": g02_ref,
            "g03_subject": g03_ref,
            "minimum_width_eight_scheduled_seconds": 450,
        }
    )
    observations: list[dict[str, Any]] = []
    for width in authority.CALIBRATION_WIDTHS:
        for repetition in range(1, authority.CALIBRATION_REPETITIONS + 1):
            phase_authority_sha256 = digest(
                {
                    "schema": "SUBSTRATE_ODYSSEY_G06_CALIBRATION_AUTHORITY/v1",
                    "dispatch_contract_sha256": phase_harness["dispatch_contract_sha256"],
                    "width": width,
                    "repetition": repetition,
                }
            )
            phase_run_id = f"g06-{phase_harness['dispatch_contract_sha256'][:16]}-{width}x-{repetition}"
            cells = [
                _g06_cell(root, mutation_id=mutation_id, variant=variant, width=width, repetition=repetition, frontier=frontier)
                for frontier in authority.FRONTIER_IDS[:width]
            ]
            refs: list[dict[str, str]] = []
            for cell in cells:
                frontier = cell["id"]
                task = retrieval_tasks[frontier]
                cell["task_binding"] = {
                    "manifest_path": manifest_by_frontier[frontier]["path"],
                    "manifest_sha256": manifest_by_frontier[frontier]["file_sha256"],
                    "task_index": 0,
                    "task_id": task["task_id"],
                    "task_sha256": digest(task),
                }
                cell["resource_parity"]["candidate"]["models"] = [selected_base["id"]]
                cell["resource_parity"]["control"]["models"] = [selected_base["id"]]
                receipt_base = f"{base}/{width}x/{repetition}/{cell['id']}"
                candidate_receipt = _g06_adapter_receipt(
                    root,
                    relative_root=receipt_base,
                    role="candidate",
                    frontier=cell["id"],
                    task=task,
                    manifest_sha256=manifest_by_frontier[cell["id"]]["file_sha256"],
                    authority_sha256=phase_authority_sha256,
                    run_id=phase_run_id,
                    model=selected_base["id"],
                    adapter_sha256=adapter_sha256,
                )
                control_receipt = _g06_adapter_receipt(
                    root,
                    relative_root=receipt_base,
                    role="control",
                    frontier=cell["id"],
                    task=task,
                    manifest_sha256=manifest_by_frontier[cell["id"]]["file_sha256"],
                    authority_sha256=phase_authority_sha256,
                    run_id=phase_run_id,
                    model=selected_base["id"],
                    adapter_sha256=adapter_sha256,
                )
                cell["candidate_receipt"] = candidate_receipt
                cell["control_receipt"] = control_receipt
                refs.extend((candidate_receipt, control_receipt))
            boundary_receipt = _g06_phase_boundary(
                root,
                relative_root=f"{base}/{width}x/{repetition}/phase-boundary",
                authority_sha256=phase_authority_sha256,
                run_id=phase_run_id,
                cells=cells,
            )
            refs.append(boundary_receipt)
            observations.append(
                {
                    "width": width,
                    "repetition": repetition,
                    "cells": cells,
                    "metrics": {
                        "aggregate_throughput": 1.0,
                        "per_cell_slowdown_ratio": 1.0,
                        "resident_memory_bytes": 1,
                        "swap_pageout_delta_bytes": 0,
                        "disk_latency_ms": 1.0,
                        "checkpoint_latency_ms": 1.0,
                        "model_latency_ms": 1.0,
                        "cpu_time_seconds": 1.0,
                        "io_bytes": 1,
                        "thermal_pressure": "nominal",
                        "critical_pressure": False,
                        "strict_dispatch_budget_seconds": 150,
                        "scheduled_phase_seconds": 150,
                        "global_dwell_seconds": 1.0,
                        "parent_guard_samples": 1,
                        "paired_adapter_dispatches": 2 * width,
                        "phase_boundary_receipt": boundary_receipt,
                        "observation_wall_seconds": 150.0,
                        "active_dispatch_wall_seconds": 1.0,
                        "raw_active_dispatch_slowdown_ratio": 1.0,
                        "e2e_slowdown_ratio": 1.0,
                        "width1_baseline_seconds": 1.0,
                        "slowdown_basis": "active_paired_dispatch_wall_with_deadline_guard",
                    },
                    "checks": dict(required_checks),
                    "receipt_refs": refs,
                }
            )
    return _self_sealed(
        {
            "schema": "SUBSTRATE_ODYSSEY_WIDTH_CALIBRATION/v1",
            **_frozen_binding_fields(root, frozen),
            "all_pass": True,
            "admitted_width": 8,
            "full_program_requires_width": 8,
            "calibration_widths": list(authority.CALIBRATION_WIDTHS),
            "repetitions_per_width": authority.CALIBRATION_REPETITIONS,
            "observations": observations,
            "phase_harness": phase_harness,
            "width_eight_scheduled_seconds": 450.0,
            "checks": required_checks,
        }
    )


def _g09_arm(root: Path, *, mutation_id: str, variant: str, frontier: str, role: str) -> dict[str, Any]:
    """Produce a self-digested synthetic recovery chain for an authority check."""
    base = f"g12-fixtures/g09/{mutation_id}/{variant}/{frontier}/{role}"
    full = _write_fixture_document(
        root,
        f"{base}/full.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
            "program": PROGRAM,
            "activation": False,
            "kind": "full",
            "event_chain_sha256": "a" * 64,
        },
    )
    full_document = _read_fixture_document(root, full)
    delta = _write_fixture_document(
        root,
        f"{base}/delta.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
            "program": PROGRAM,
            "activation": False,
            "kind": "delta",
            "event_chain_sha256": "b" * 64,
            "parent_checkpoint_sha256": full_document["sha256"],
        },
    )
    trace = _write_fixture_document(
        root,
        f"{base}/trace.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_G12_SYNTHETIC_EVENT_TRACE/v1",
            "program": PROGRAM,
            "activation": False,
            "event_chain_sha256": "b" * 64,
        },
    )
    restart = _write_fixture_document(
        root,
        f"{base}/restart.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_G12_SYNTHETIC_RESTART_RECEIPT/v1",
            "program": PROGRAM,
            "activation": False,
            "recovered": True,
            "interactive_shell_independent": True,
        },
    )
    writer = _write_fixture_document(
        root,
        f"{base}/single-writer.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_G12_SYNTHETIC_SINGLE_WRITER_RECEIPT/v1",
            "program": PROGRAM,
            "activation": False,
            "single_writer": True,
        },
    )
    lock = root / f"{base}/writer.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("synthetic exclusive writer lock\n", encoding="utf-8")
    return {
        "pre_interrupt_state_sha256": "c" * 64,
        "restored_state_sha256": "c" * 64,
        "full_checkpoint": full,
        "delta_checkpoints": [] if variant == "mutant" and mutation_id == "missing_checkpoint" and frontier == "A" and role == "candidate" else [delta],
        "event_trace": trace,
        "restart_receipt": restart,
        "writer_lock": _relative(root, lock),
        "single_writer_receipt": writer,
        "recovery_downtime_seconds": 0,
        "resumed_at_sealed_boundary": True,
    }


def _g09_subject(root: Path, frozen: dict[str, Any], *, mutation_id: str, variant: str) -> dict[str, Any]:
    from substrate import odyssey_authority as authority

    design = authority._frozen_design(root, frozen)
    storage = design["storage"]
    rehearsals = [
        {
            "frontier": frontier,
            "arms": {
                "candidate": _g09_arm(root, mutation_id=mutation_id, variant=variant, frontier=frontier, role="candidate"),
                "control": _g09_arm(root, mutation_id=mutation_id, variant=variant, frontier=frontier, role="control"),
            },
            "unplanned_interruptions": 0,
            "max_single_unplanned_downtime_seconds": 0,
            "cumulative_unplanned_downtime_seconds": 0,
        }
        for frontier in authority.FRONTIER_IDS
    ]
    disturbances = {
        name: _write_fixture_document(
            root,
            f"g12-fixtures/g09/{mutation_id}/{variant}/disturbances/{name}.json",
            {
                "schema": "SUBSTRATE_ODYSSEY_G12_SYNTHETIC_DISTURBANCE_RECEIPT/v1",
                "program": PROGRAM,
                "activation": False,
                "disturbance": name,
            },
        )
        for name in ("process_restart", "model_replacement", "tool_or_body_change", "sensor_or_source_interruption")
    }
    return _self_sealed(
        {
            "schema": "SUBSTRATE_ODYSSEY_DURABILITY_REHEARSAL/v1",
            **_frozen_binding_fields(root, frozen),
            "all_pass": True,
            "checkpoint_policy": {
                "delta_interval_seconds": storage["delta_checkpoint_interval_seconds"],
                "full_interval_seconds": storage["full_checkpoint_interval_seconds"],
            },
            "rehearsals": rehearsals,
            "scheduled_disturbance_receipts": disturbances,
            "checks": {
                "frozen_build_bound": True,
                "source_maps_bound": True,
                "checkpoint_round_trip": True,
                "delta_plus_full_restore": True,
                "process_restart": True,
                "model_replacement": True,
                "tool_or_body_change": True,
                "sensor_or_source_interruption": True,
                "single_writer": True,
                "interactive_shell_independent": True,
                "recovery_limits_bound": True,
                "event_chain_valid": True,
            },
        }
    )


def _adapter_script(path: Path, *, mutant_kind: str | None = None) -> None:
    """Write a tiny receipt adapter; it has no model or evaluator access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
receipt = {
    'schema': 'SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1',
    'activation': False,
    'authority_sha256': request['authority_sha256'],
    'run_id': request['run_id'],
    'frontier': request['frontier'],
    'role': request['role'],
    'cycle': request['cycle'],
    'phase': request['phase'],
    'task_id': request['task']['task_id'],
    'candidate_manifest_sha256': request['candidate_manifest_sha256'],
    'request_sha256': request['request_sha256'],
    'elapsed_seconds': 0.001,
}
mutation = __MUTANT_KIND__
if mutation == 'cross_lane_model_context':
    receipt['role'] = 'control'
elif mutation == 'forged_receipt':
    receipt['request_sha256'] = '0' * 64
output = Path(request['receipt_path'])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, sort_keys=True))
""".replace("__MUTANT_KIND__", repr(mutant_kind)),
        encoding="utf-8",
    )


def _adapter_action(root: Path, *, mutation_id: str, variant: str) -> None:
    from substrate import odyssey_worker as worker

    adapter = root / f"g12-fixtures/adapter/{mutation_id}/{variant}/adapter.py"
    _adapter_script(adapter, mutant_kind=mutation_id if variant == "mutant" else None)
    worker._adapter(
        root,
        authority_sha256="a" * 64,
        run_id="g12-synthetic-adapter",
        worker_root=root / f"g12-fixtures/adapter/{mutation_id}/{variant}/run",
        frontier="A",
        role="candidate",
        command=[sys.executable, str(adapter)],
        manifest_sha256="b" * 64,
        task={"task_id": "g12-A-0000"},
        cycle=0,
        phase="retrieval",
    )


def _worker_fixture_authority(root: Path, *, mutation_id: str, variant: str) -> Path:
    """Create a one-phase worker fixture for admission and lock checks only."""
    from substrate import odyssey_task_bank as task_bank
    from substrate import odyssey_worker as worker

    base = f"g12-fixtures/worker/{mutation_id}/{variant}"
    seed = f"g12-worker-{mutation_id}-{variant}"
    candidate, _evaluator = task_bank.materialize(task_bank._digest({"seed": seed}), seed, "A", 1)
    candidate_path = root / f"{base}/candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
    adapter = root / f"{base}/adapter.py"
    _adapter_script(adapter)
    run_root = f"{base}/run"
    worker_fields: dict[str, Any] = {
        "test_mode": True,
        "run_root": run_root,
        "publication_root": f"{base}/publication",
        "frontiers": [
            {
                "id": "A",
                "candidate_manifest": _relative(root, candidate_path),
                "candidate_manifest_sha256": worker.file_digest(candidate_path),
                "candidate_command": [sys.executable, str(adapter)],
                "control_command": [sys.executable, str(adapter)],
            }
        ],
        "phase_names": ["retrieval"],
        "phase_seconds": 0,
        "microcycles_per_frontier": 1,
        "max_parallel_frontiers": 1,
    }
    body = {
        "schema": "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "run_id": f"g12-{mutation_id}-{variant}",
        "program": {"id": worker.TEST_PROGRAM, "duration_seconds": 0, "launch_allowed": True},
        "seal": {"status": "sealed"},
        "launch_gates": [{"id": "G12", "status": "pass"}],
        "worker_source_sha256": worker.file_digest(Path(worker.__file__)),
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
        "worker": worker_fields,
    }
    body["sha256"] = worker._digest(body)
    path = root / f"docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.test.{mutation_id}.{variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return path


def _duplicate_writer_action(root: Path, *, variant: str) -> None:
    """Exercise the real worker's non-blocking exclusive lock admission."""
    from substrate import odyssey_worker as worker

    authority_path = _worker_fixture_authority(root, mutation_id="duplicate_writer", variant=variant)
    if variant == "clean":
        worker.run(root, authority_file=authority_path)
        return
    authority = _read_json(authority_path)
    run_root = root / authority["worker"]["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "worker.lock"
    with lock_path.open("a+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            worker.run(root, authority_file=authority_path)
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)


def _pseudoreplication_action(*, variant: str) -> None:
    """Run the existing hardened-design validator against a complete fixture."""
    from substrate import odyssey7d

    value = {
        "program": {"id": PROGRAM, "activation": False, "launch_allowed": False, "duration_seconds": 7 * 24 * 3600, "duration_hours": 168},
        "independent_units": {
            "type": "paired_frontier_history_block",
            "count": 8,
            "candidate_histories": 8,
            "control_histories": 8,
            "total_continuous_state_histories": 16,
            "pseudoreplication_guard": "microcycles and events are not independent units",
        },
        "timeline": {
            "microcycles_per_frontier": 84,
            "total_paired_microcycles": 672,
            "total_scored_paired_events": 2688,
            "total_scored_dimension_observations": 10752,
        },
        "frontiers": [{"id": frontier} for frontier in "ABCDEFGH"],
        "blindness": {"custodians": 8, "two_custodian_day7_reveal": True, "trace_lock_before_answer_reveal": True},
        "resources": {
            "resident_cap_gib": 85,
            "normal_admission_ceiling_gib": 75,
            "p2_checkpoint_threshold_gib": 80,
            "p1_pause_threshold_gib": 82,
            "global_hold_threshold_gib": 85,
            "widths_to_calibrate": [1, 2, 4, 6, 8],
            "calibration_repetitions": 3,
            "full_program_requires_width": 8,
        },
        "storage": {"delta_checkpoint_interval_seconds": 7200, "full_checkpoint_interval_seconds": 43200},
        "statistics": {"primary_unit": "paired_frontier_history_block"},
        "launch_gates": [{"id": f"G{index:02d}", "status": "pending"} for index in range(1, 16)],
    }
    if variant == "mutant":
        value["statistics"]["primary_unit"] = "event"
    checks = odyssey7d.validate(value)
    if not all(checks.values()):
        raise odyssey7d.Refused(f"hardened design validation failed: {[name for name, passed in checks.items() if not passed]}")


def _fixture_authority(root: Path, *, full: bool, worker_fields: dict[str, Any]) -> Path:
    """Create a non-scientific authority fixture used only by a child probe."""
    from substrate import odyssey_worker as worker

    program = {
        "id": worker.PROGRAM if full else worker.TEST_PROGRAM,
        "duration_seconds": 7 * 24 * 3600 if full else 0,
        "launch_allowed": True,
    }
    frozen_sha256: str | None = None
    if full:
        # Full-program worker admission now revalidates the exact transition
        # freeze at dispatch.  The synthetic clean/mutant fixture must bind
        # that real fixture freeze rather than bypassing the production check.
        frozen = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
        frozen_sha256 = frozen["sha256"]
    body = {
        "schema": "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "run_id": "g12-synthetic-control-plane-fixture",
        "program": program,
        "seal": {"status": "sealed", **({"frozen_build_sha256": frozen_sha256} if frozen_sha256 is not None else {})},
        "launch_gates": [{"id": "G12", "status": "pass"}],
        "worker_source_sha256": worker.file_digest(Path(worker.__file__)),
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
        "worker": worker_fields,
        **({"frozen_build_sha256": frozen_sha256} if frozen_sha256 is not None else {}),
    }
    body["sha256"] = worker._digest(body)
    path = root / PLAN / "ODYSSEY_7D.test.g12-mutation-authority.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return path


def _full_worker_fields(*, frontiers: list[str], width: int = 8, checkpoint: dict[str, int] | None = None) -> dict[str, Any]:
    from substrate import odyssey_worker as worker

    return {
        "test_mode": False,
        "frontiers": [{"id": frontier} for frontier in frontiers],
        "microcycles_per_frontier": 84,
        "phase_names": list(worker.PHASES),
        "phase_seconds": 1800,
        "max_parallel_frontiers": width,
        "checkpoint": checkpoint or {"delta_interval_seconds": 7200, "full_interval_seconds": 43200},
    }


def _observe(callback: Any, *, expected: tuple[type[Exception], ...]) -> dict[str, Any]:
    """Execute a guard once without treating a crash as an observed rejection."""
    try:
        callback()
    except expected as error:
        return {"accepted": False, "rejected": True, "unexpected_error": False, "error_class": type(error).__name__, "error_sha256": digest(str(error))}
    except Exception as error:
        return {"accepted": False, "rejected": False, "unexpected_error": True, "error_class": type(error).__name__, "error_sha256": digest(str(error))}
    return {"accepted": True, "rejected": False, "unexpected_error": False, "error_class": None, "error_sha256": None}


def probe(root: Path, mutation_id: str, variant: str) -> dict[str, Any]:
    """Execute one clean or mutant control-plane probe inside an isolated clone."""
    mutation = MUTATION_BY_ID.get(mutation_id)
    if mutation is None:
        raise Refused(f"unknown Odyssey mutation: {mutation_id}")
    if variant not in {"clean", "mutant"}:
        raise Refused("mutation variant must be clean or mutant")
    root = root.expanduser().resolve()
    _require_isolated_fixture(root, mutation.identifier, variant)
    from substrate import odyssey_authority as authority
    from substrate import odyssey_task_bank as task_bank
    from substrate import odyssey_worker as worker

    frozen = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    if mutation.application == "source_drift":
        result = _observe(lambda: _frozen_source_binding(root), expected=(Refused,))
    elif mutation.application == "g03_subject":
        subject = _g03_subject(root, frozen, mutation_id=mutation.identifier, variant=variant)
        result = _observe(lambda: authority._gate_specific_checks(root, "G03", subject, frozen), expected=(authority.Refused,))
    elif mutation.application == "g06_subject":
        subject = _g06_subject(root, frozen, mutation_id=mutation.identifier, variant=variant)
        result = _observe(lambda: authority._gate_specific_checks(root, "G06", subject, frozen), expected=(authority.Refused,))
    elif mutation.application == "g09_subject":
        subject = _g09_subject(root, frozen, mutation_id=mutation.identifier, variant=variant)
        result = _observe(lambda: authority._gate_specific_checks(root, "G09", subject, frozen), expected=(authority.Refused,))
    elif mutation.application == "worker_lock":
        result = _observe(lambda: _duplicate_writer_action(root, variant=variant), expected=(worker.Refused,))
    elif mutation.application == "adapter_receipt":
        result = _observe(lambda: _adapter_action(root, mutation_id=mutation.identifier, variant=variant), expected=(worker.Refused,))
    elif mutation.application == "pseudoreplication":
        from substrate import odyssey7d

        result = _observe(lambda: _pseudoreplication_action(variant=variant), expected=(odyssey7d.Refused,))
    elif mutation.identifier == "candidate_manifest_digest_tamper":
        candidate, _evaluator = task_bank.materialize(worker._digest({"seed": "g12-control-plane"}), "g12-control-plane", "A", 1)
        path = root / f"g12-fixtures/manifest-digest/{variant}/candidate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
        frontier = {
            "id": "A",
            "candidate_manifest": _relative(root, path),
            "candidate_manifest_sha256": "0" * 64 if variant == "mutant" else worker.file_digest(path),
            "candidate_command": ["/usr/bin/true"],
            "control_command": ["/usr/bin/true"],
        }
        result = _observe(lambda: worker._manifest_for_frontier(root, frontier, full=False, task_count=1), expected=(worker.Refused,))
    elif mutation.identifier == "authority_evaluator_input_injection":
        fields = {
            "test_mode": True,
            "frontiers": [{"id": "A"}],
            "phase_names": ["retrieval"],
            "phase_seconds": 1,
            "microcycles_per_frontier": 1,
        }
        if variant == "mutant":
            fields["evaluator_answer_path"] = "evaluator-only/answers.json"
        path = _fixture_authority(root, full=False, worker_fields=fields)
        result = _observe(lambda: worker.validate_authority(root, path), expected=(worker.Refused,))
    elif mutation.identifier == "full_program_frontier_omission":
        path = _fixture_authority(root, full=True, worker_fields=_full_worker_fields(frontiers=list("ABCDEFGH") if variant == "clean" else list("ABCDEFG")))
        result = _observe(lambda: worker.validate_authority(root, path), expected=(worker.Refused,))
    elif mutation.identifier == "full_program_width_reduction":
        path = _fixture_authority(root, full=True, worker_fields=_full_worker_fields(frontiers=list("ABCDEFGH"), width=8 if variant == "clean" else 7))
        result = _observe(lambda: worker.validate_authority(root, path), expected=(worker.Refused,))
    elif mutation.identifier == "full_program_checkpoint_cadence_drift":
        checkpoint = {"delta_interval_seconds": 7200, "full_interval_seconds": 43200}
        if variant == "mutant":
            checkpoint["delta_interval_seconds"] = 3600
        path = _fixture_authority(root, full=True, worker_fields=_full_worker_fields(frontiers=list("ABCDEFGH"), checkpoint=checkpoint))
        result = _observe(lambda: worker.validate_authority(root, path), expected=(worker.Refused,))
    elif mutation.identifier == "authority_self_digest_forgery":
        path = _fixture_authority(
            root,
            full=False,
            worker_fields={"test_mode": True, "frontiers": [{"id": "A"}], "phase_names": ["retrieval"], "phase_seconds": 1, "microcycles_per_frontier": 1},
        )
        if variant == "mutant":
            forged = json.loads(path.read_text(encoding="utf-8"))
            forged["sha256"] = "0" * 64
            path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
        result = _observe(lambda: worker.validate_authority(root, path), expected=(worker.Refused,))
    else:
        raise Refused(f"mutation has no probe implementation: {mutation.identifier}")
    expected_outcome = "accepted" if variant == "clean" else "rejected"
    observed_outcome = "accepted" if result["accepted"] else "rejected" if result["rejected"] else "error"
    return {
        "schema": "SUBSTRATE_ODYSSEY_MUTATION_PROBE/v1",
        "program": PROGRAM,
        "mutation": mutation.identifier,
        "variant": variant,
        "control_plane_only": True,
        "scientific_evidence": False,
        "activation": False,
        "expected_outcome": expected_outcome,
        "observed_outcome": observed_outcome,
        "passed": observed_outcome == expected_outcome,
        **result,
    }


def _probe_variant(
    baseline: Path,
    mutation: Mutation,
    variant: str,
    *,
    python_executable: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Use a fresh process and a fresh exact-clone copy for every case variant."""
    with tempfile.TemporaryDirectory(prefix=f"substrate-odyssey-g12-{mutation.identifier}-{variant}-") as temporary:
        fixture = Path(temporary) / "fixture"
        # Keep the clone's .git metadata: G03/G06/G09 verify that a receipt is
        # for the exact checked-out commit, and each fixture remains a private
        # copy rather than a link back to the main worktree.
        shutil.copytree(baseline, fixture, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _mark_isolated_fixture(fixture, mutation, variant=variant)
        _apply_source_mutation(fixture, mutation, variant=variant)
        environment = {**os.environ, "PYTHONPATH": str(fixture / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [python_executable, "-m", "substrate.odyssey_mutations", "probe", "--root", str(fixture), "--case", mutation.identifier, "--variant", variant]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=fixture, capture_output=True, text=True, env=environment, check=False, timeout=180)
            timed_out = False
        except subprocess.TimeoutExpired as error:
            completed = None
            timed_out = True
            stdout, stderr = error.stdout or "", error.stderr or ""
        else:
            stdout, stderr = completed.stdout, completed.stderr
        execution = {
            "command": command[:3] + ["<probe>", "--case", mutation.identifier, "--variant", variant],
            "returncode": None if completed is None else completed.returncode,
            "timed_out": timed_out,
            "duration_milliseconds": int((time.monotonic() - started) * 1000),
            "stdout_sha256": digest(stdout),
            "stderr_sha256": digest(stderr),
        }
        child = _child_result(stdout) if completed is not None and completed.returncode == 0 else None
        return child, execution


def _write_execution_receipt(
    root: Path,
    receipt_root: Path,
    *,
    mutation: Mutation,
    variant: str,
    source_commit: str,
    frozen: dict[str, Any],
    probe_result: dict[str, Any] | None,
    execution: dict[str, Any],
) -> dict[str, str]:
    """Persist a self-digested, content-addressed observation for one probe."""
    passed = probe_result is not None and probe_result.get("passed") is True
    document = {
        "schema": "SUBSTRATE_ODYSSEY_MUTATION_EXECUTION_RECEIPT/v1",
        "program": PROGRAM,
        "activation": False,
        "external_activation": False,
        "scientific_evidence": False,
        "evidence_scope": "isolated_synthetic_control_plane_fault_injection_only",
        "mutation": mutation.identifier,
        "variant": variant,
        "source_commit": source_commit,
        "frozen_build_sha256": frozen["sha256"],
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "execution": execution,
        "probe": probe_result,
        "passed": passed,
    }
    document = _self_sealed(document)
    path = receipt_root / f"{mutation.identifier}.{variant}.{document['sha256']}.json"
    _write_json(path, document)
    return {"path": _relative(root, path), "sha256": file_digest(path)}


def _paired_case(
    root: Path,
    baseline: Path,
    receipt_root: Path,
    mutation: Mutation,
    *,
    python_executable: str,
    source_commit: str,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    clean_probe, clean_execution = _probe_variant(baseline, mutation, "clean", python_executable=python_executable)
    mutant_probe, mutant_execution = _probe_variant(baseline, mutation, "mutant", python_executable=python_executable)
    clean_receipt = _write_execution_receipt(
        root,
        receipt_root,
        mutation=mutation,
        variant="clean",
        source_commit=source_commit,
        frozen=frozen,
        probe_result=clean_probe,
        execution=clean_execution,
    )
    mutant_receipt = _write_execution_receipt(
        root,
        receipt_root,
        mutation=mutation,
        variant="mutant",
        source_commit=source_commit,
        frozen=frozen,
        probe_result=mutant_probe,
        execution=mutant_execution,
    )
    clean_passed = clean_probe is not None and clean_probe.get("passed") is True and clean_probe.get("accepted") is True
    detected = mutant_probe is not None and mutant_probe.get("passed") is True and mutant_probe.get("rejected") is True
    return {
        "id": mutation.identifier,
        "category": mutation.category,
        "target": mutation.target,
        "description": mutation.description,
        "injected": True,
        "detected": detected,
        "survived": not detected,
        "clean_case_passed": clean_passed,
        "clean_receipt": clean_receipt,
        "mutant_receipt": mutant_receipt,
    }


def run(root: Path, output_path: Path, *, python_executable: str | None = None) -> dict[str, Any]:
    """Run the whole non-scientific G12 mutation suite and write one receipt."""
    root = root.expanduser().resolve()
    output_path = (root / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if not _inside(root, output_path):
        raise Refused("mutation report output must stay inside the repository root")
    frozen, paths = _frozen_source_binding(root)
    input_paths = odyssey_transition.build_inputs(root)
    _assert_scoped_tree_clean(root, paths, input_paths)
    source_commit = _git_head(root)
    executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="substrate-odyssey-g12-cleanclone-") as temporary:
        clone = _clone_at_head(root, Path(temporary), source_commit)
        baseline = clone["root"]
        receipt_root = output_path.parent / f"{output_path.stem}.receipts"
        cases = [
            _paired_case(root, baseline, receipt_root, mutation, python_executable=executable, source_commit=source_commit, frozen=frozen)
            for mutation in MUTATIONS
        ]
        clone_receipt = clone["receipt"]
    survivors = [row["id"] for row in cases if row["survived"] is True]
    clean_failures = [row["id"] for row in cases if row["clean_case_passed"] is not True]
    required_ids = {
        "answer_leakage",
        "shared_mutable_cache",
        "cross_lane_model_context",
        "duplicate_writer",
        "forged_receipt",
        "missing_checkpoint",
        "wrong_source_digest",
        "result_dependent_task_selection",
        "control_under_resourcing",
        "pseudo_replication_analysis",
    }
    ids = {row["id"] for row in cases}
    uncovered = sorted(required_ids - ids)
    all_pass = not survivors and not clean_failures and not uncovered
    body = {
        "schema": SCHEMA,
        "program": PROGRAM,
        "status": "pass" if all_pass else "fail",
        "activation": False,
        "external_activation": False,
        "scientific_evidence": False,
        "evidence_scope": "isolated_synthetic_control_plane_fault_injection_only",
        "non_claims": [
            "no model, external tool, corpus, or evaluator was invoked",
            "only disposable synthetic candidate fixtures were materialized",
            "no evaluator-owned answer material was persisted or consumed",
            "this receipt is not scientific, model, task, or outcome evidence",
        ],
        "source_commit": source_commit,
        "frozen_build_sha256": frozen["sha256"],
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "runner_source_key": RUNNER_SOURCE_KEY,
        "runner_source_sha256": frozen["implementation_sha256"][RUNNER_SOURCE_KEY],
        "clean_clone": clone_receipt,
        "mutations": cases,
        "checks": {
            "frozen_build_bound": all_pass,
            "source_maps_bound": all_pass,
            "runtime_mutants_injected": all_pass,
            "runtime_mutants_detected": all_pass,
            "clean_baselines_accepted": all_pass,
            "guard_coverage_complete": all_pass,
            "no_pending_mutations": all_pass,
        },
        "declared_mutation_count": len(cases),
        "injected_count": len(cases),
        "detected_count": len(cases) - len(survivors),
        "pending_count": 0,
        "survivor_count": len(survivors),
        "survivors": survivors,
        "uncovered": uncovered,
        "undeclared": [],
        "clean_failures": clean_failures,
        "all_pass": all_pass,
    }
    body["sha256"] = digest(body)
    _write_json(output_path, body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="write one isolated G12 mutation receipt")
    run_parser.add_argument("--root", type=Path, default=Path.cwd())
    run_parser.add_argument("--out", type=Path, required=True)
    probe_parser = commands.add_parser("probe", help="internal: run one mutation against an isolated fixture")
    probe_parser.add_argument("--root", type=Path, required=True)
    probe_parser.add_argument("--case", choices=tuple(MUTATION_BY_ID), required=True)
    probe_parser.add_argument("--variant", choices=("clean", "mutant"), required=True)
    args = parser.parse_args()
    result = run(args.root, args.out) if args.command == "run" else probe(args.root, args.case, args.variant)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
