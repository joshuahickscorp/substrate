"""Append-only second robustness horizon for Generation 1.

The v2 horizon begins only after the complete v1 result, separately authored
verification, generated report receipt, and exact H05 classification all
validate.  It preserves the v1 scientific and safety boundaries while adding
five further fresh cycles, H06 through H10 (absolute cycles 7 through 11).

The proven v1 execution engine is reused under a process-local, re-entrant,
scoped substitution.  Every substituted global is restored before the public
operation returns, including exceptional exits.  No v1 source or artifact is
mutated.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as predecessor
from mop.studies import generation1_successor_horizon_verify as predecessor_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256, sha256_file
from mop.studio import generation1_supervisor as supervisor

PROGRAM_ID = "generation1-successor-horizon-v2"
ADMISSION_SCHEMA = "mop-generation1-successor-horizon-admission/v2"
SHARD_SCHEMA = "mop-generation1-successor-horizon-shard/v2"
CLASSIFICATION_SCHEMA = "mop-generation1-successor-horizon-classification/v2"
RESULT_SCHEMA = "mop-generation1-successor-horizon-result/v2"
VERIFICATION_SCHEMA = "mop-generation1-successor-horizon-verification/v2"
REPORT_RECEIPT_SCHEMA = "mop-generation1-successor-horizon-report-receipt/v2"
CLAIM_SCOPE = (
    "result-gated same-code fresh-seed robustness continuing a verified predecessor horizon; "
    "no independent scientific confirmation or activation"
)
VERIFICATION_CLAIM_SCOPE = (
    "independent predecessor-program, terminal-supervisor, receipt, dependency-closure, "
    "seed-boundary, aggregation, classification, and mutation verification; "
    "no second generator family"
)

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_ADMISSION = DEFAULT_ROOT / "admission.json"
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_HORIZON_V2.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_HORIZON_V2.verification.json"
DEFAULT_REPORT = REPO_ROOT / "runs/generation1/GENERATION1_SUCCESSOR_EVIDENCE_REPORT_V2.md"
DEFAULT_REPORT_RECEIPT = DEFAULT_ROOT / "report_receipt.json"
PROGRAM_MANIFEST = REPO_ROOT / "configs/campaign/generation1_successor_horizon_v2.json"

DEFAULT_PARENT_RESULT = predecessor.DEFAULT_RESULT
DEFAULT_PARENT_VERIFICATION = predecessor.DEFAULT_VERIFICATION
DEFAULT_PARENT_REPORT_RECEIPT = predecessor.DEFAULT_REPORT_RECEIPT

EPOCH_IDS = ("H06", "H07", "H08", "H09", "H10")
EPOCH_CYCLES = (7, 8, 9, 10, 11)
D1_SHARD_COUNT = predecessor.D1_SHARD_COUNT
MECHANICS_SHARD_COUNT = predecessor.MECHANICS_SHARD_COUNT
D1_PARTITIONS = predecessor.D1_PARTITIONS
MECHANICS_PARTITIONS = predecessor.MECHANICS_PARTITIONS
RETRY_LIMIT = predecessor.RETRY_LIMIT
IDLE_WORKERS = predecessor.IDLE_WORKERS
HAWKING_WORKERS = predecessor.HAWKING_WORKERS

_ADMISSION_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "claim_scope",
        "created_at",
        "parent_horizon",
        "epoch_ids",
        "fresh_cycle_indices",
        "d1_predecessor_classification",
        "d1_initially_eligible",
        "mechanics_predecessor_survivors",
        "mechanics_internal_dependencies",
        "mechanics_dependency_pruned_lanes",
        "mechanics_initially_eligible_lanes",
        "boundary_rules",
        "planned_compute",
        "complete",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "independent_scientific_confirmation",
        "admission_sha256",
    }
)
_PARENT_BINDING_FIELDS = frozenset(
    {
        "program_manifest",
        "supervisor_status",
        "result",
        "verification",
        "report_receipt",
        "final_classification",
    }
)

_RUNTIME_LOCK = threading.RLock()
_V1_RUN_SHARD = predecessor.run_shard
_V1_SHARD_WORK_ITEMS = predecessor.shard_work_items
_V1_VALIDATE_SHARD = predecessor.validate_shard
_V1_BUILD_CLASSIFICATION = predecessor.build_classification
_V1_VALIDATE_CLASSIFICATION = predecessor.validate_classification
_V1_CLASSIFY_EPOCH = predecessor.classify_epoch
_V1_BUILD_RESULT = predecessor.build_result
_V1_VALIDATE_RESULT = predecessor.validate_result
_V1_AGGREGATE = predecessor.aggregate
_V1_VALIDATE_REPORT_RECEIPT = predecessor.validate_report_receipt


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path is not a string")
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} path is not repository-relative")
    resolved = (REPO_ROOT / raw).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{label} path escapes the repository")
    return resolved


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(core):
        raise ValueError(f"{label} seal drifted")


def _binding(path: Path, value: Mapping[str, Any], seal_field: str) -> dict[str, Any]:
    seal = value.get(seal_field)
    if not isinstance(seal, str):
        raise ValueError(f"{path} does not contain {seal_field}")
    return {
        "path": _relative(path),
        "file_sha256": sha256_file(path),
        seal_field: seal,
    }


def _classification_path(root: Path, epoch_index: int) -> Path:
    return Path(root).resolve() / "classifications" / f"{EPOCH_IDS[epoch_index].lower()}.json"


def _shard_path(root: Path, epoch_index: int, lane: str, shard_index: int) -> Path:
    return Path(root).resolve() / "shards" / EPOCH_IDS[epoch_index].lower() / f"{lane}_{shard_index:02d}.json"


def planned_epoch_compute_seconds() -> float:
    return predecessor.planned_epoch_compute_seconds()


def planned_horizon_compute_seconds() -> float:
    return len(EPOCH_IDS) * planned_epoch_compute_seconds()


def _mechanics_dependency_map() -> dict[str, list[str]]:
    known = {lane.lane_id for lane in mechanics.LANES}
    return {
        lane.lane_id: [dependency for dependency in lane.dependencies if dependency in known]
        for lane in mechanics.LANES
    }


def _dependency_closed_lanes(predecessor_survivors: list[str]) -> tuple[list[str], list[str]]:
    """Apply the transitive in-mechanics dependency closure in canonical lane order."""

    order = [lane.lane_id for lane in mechanics.LANES]
    known = set(order)
    if len(predecessor_survivors) != len(set(predecessor_survivors)):
        raise ValueError("predecessor mechanics survivor list is duplicated")
    if not set(predecessor_survivors) <= known:
        raise ValueError("predecessor mechanics survivor list contains an unknown lane")
    dependencies = _mechanics_dependency_map()
    retained = set(predecessor_survivors)
    changed = True
    while changed:
        changed = False
        for lane_id in order:
            if lane_id in retained and not set(dependencies[lane_id]) <= retained:
                retained.remove(lane_id)
                changed = True
    eligible = [lane_id for lane_id in order if lane_id in retained]
    dependency_pruned = [
        lane_id for lane_id in order if lane_id in predecessor_survivors and lane_id not in retained
    ]
    return eligible, dependency_pruned


def _parent_root_from_result(value: Mapping[str, Any]) -> Path:
    rows = value.get("classifications")
    if not isinstance(rows, list) or len(rows) != len(predecessor.EPOCH_IDS):
        raise ValueError("predecessor classification inventory is incomplete")
    final = rows[-1]
    if not isinstance(final, Mapping):
        raise ValueError("predecessor final classification binding is invalid")
    path = _repo_path(final.get("path"), "predecessor final classification")
    if path.parent.name != "classifications":
        raise ValueError("predecessor final classification path is outside its canonical root")
    return path.parent.parent


def _dotted(value: object, path: str) -> object:
    current = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _parent_artifact_report(
    expectation: supervisor.ArtifactExpectation,
    *,
    capsule_id: str,
) -> tuple[Path, dict[str, Any]]:
    artifact_path = _repo_path(
        expectation.path,
        f"predecessor supervisor capsule {capsule_id} artifact",
    )
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise ValueError(f"predecessor supervisor capsule {capsule_id} artifact is absent or unsafe")
    payload = _read_object(artifact_path)
    problems: list[str] = []
    if payload.get("schema") != expectation.schema:
        problems.append("artifact schema drifted")
    for dotted, expected in expectation.fields:
        observed = _dotted(payload, dotted)
        if observed != expected:
            problems.append(f"{dotted}={observed!r}, expected {expected!r}")
    if expectation.seal_field is not None:
        try:
            _validate_seal(
                payload,
                expectation.seal_field,
                f"predecessor supervisor capsule {capsule_id} artifact",
            )
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError(
            f"predecessor supervisor capsule {capsule_id} artifact is not clean: " + "; ".join(problems)
        )
    return artifact_path, {
        "path": expectation.path,
        "sha256": sha256_file(artifact_path),
        "schema": payload.get("schema"),
        "problems": [],
        "all_ok": True,
    }


def _validated_parent_program_state(
    *,
    parent_root: Path,
    required_artifacts: tuple[Path, ...],
) -> dict[str, dict[str, Any]]:
    manifest_path = Path(predecessor.PROGRAM_MANIFEST).resolve()
    if not manifest_path.is_relative_to(REPO_ROOT.resolve()) or not manifest_path.is_file():
        raise ValueError("canonical predecessor program manifest is absent")
    try:
        program = supervisor.load_program(manifest_path, repo_root=REPO_ROOT)
    except (supervisor.Generation1Refused, OSError, ValueError) as exc:
        raise ValueError("canonical predecessor program manifest is invalid") from exc
    if (
        program.path != manifest_path
        or program.program_id != predecessor.PROGRAM_ID
        or program.program_root != parent_root
    ):
        raise ValueError("canonical predecessor program identity or root drifted")

    try:
        status = supervisor.read_status(program)
    except (supervisor.Generation1Refused, OSError, ValueError) as exc:
        raise ValueError("predecessor generic supervisor status is absent or invalid") from exc
    status_path = program.status_path.resolve()
    expected_program = {
        "path": str(program.path),
        "file_sha256": program.file_sha256,
        "program_sha256": program.program_sha256,
    }
    expected_queue_head = canonical_sha256(
        {
            "program_sha256": program.program_sha256,
            "base_capsules": [capsule.capsule_sha256 for capsule in program.capsules],
        }
    )
    capsule_rows = status.get("capsules")
    capsules = {capsule.capsule_id: capsule for capsule in program.capsules}
    if (
        status.get("program") != expected_program
        or status.get("execution_enabled") is not True
        or status.get("state") != "complete"
        or status.get("queue_head_sha256") != expected_queue_head
        or status.get("accepted_injection_count") != 0
        or status.get("next_injection_sequence") != 1
        or status.get("current_capsule") is not None
        or status.get("lane_reservation") is not None
        or status.get("problems") != []
        or not isinstance(capsule_rows, Mapping)
        or set(capsule_rows) != set(capsules)
    ):
        raise ValueError("predecessor generic supervisor completion boundary drifted")

    bound_artifacts: set[Path] = set()
    for capsule_id, capsule in capsules.items():
        row = capsule_rows[capsule_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"predecessor supervisor capsule {capsule_id} is invalid")
        attempts = row.get("attempts")
        if (
            row.get("id") != capsule.capsule_id
            or row.get("kind") != capsule.kind
            or row.get("priority") != capsule.priority
            or row.get("depends_on") != list(capsule.depends_on)
            or row.get("capsule_sha256") != capsule.capsule_sha256
            or row.get("source") != "base"
            or row.get("status") != "complete"
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or row.get("returncode") != 0
            or row.get("last_problem") is not None
        ):
            raise ValueError(f"predecessor supervisor capsule {capsule_id} completion drifted")
        reports = row.get("artifacts")
        if not isinstance(reports, list) or len(reports) != len(capsule.artifacts):
            raise ValueError(f"predecessor supervisor capsule {capsule_id} artifact inventory drifted")
        for expectation, report in zip(capsule.artifacts, reports, strict=True):
            artifact_path, expected_report = _parent_artifact_report(
                expectation,
                capsule_id=capsule_id,
            )
            if report != expected_report:
                raise ValueError(f"predecessor supervisor capsule {capsule_id} artifact bytes drifted")
            bound_artifacts.add(artifact_path)

    if not set(required_artifacts) <= bound_artifacts:
        raise ValueError("predecessor supervisor status does not bind every required horizon artifact")
    manifest = _read_object(manifest_path)
    return {
        "program_manifest": _binding(
            manifest_path,
            manifest,
            "program_sha256",
        ),
        "supervisor_status": _binding(
            status_path,
            status,
            "status_sha256",
        ),
    }


def _validated_parent_state(
    *,
    result_path: Path,
    verification_path: Path,
    report_receipt_path: Path,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    verification_path = Path(verification_path).resolve()
    report_receipt_path = Path(report_receipt_path).resolve()
    for path, label in (
        (result_path, "predecessor result"),
        (verification_path, "predecessor verification"),
        (report_receipt_path, "predecessor report receipt"),
    ):
        if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"{label} is absent or outside the repository")

    result = _read_object(result_path)
    parent_root = _parent_root_from_result(result)
    predecessor.validate_result(result, root=parent_root)

    verification = _read_object(verification_path)
    predecessor_verify.validate_verification(verification)
    source = verification.get("source") or {}
    if (
        source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
        or source.get("result_sha256") != result.get("result_sha256")
    ):
        raise ValueError("predecessor verification does not bind the exact result")
    replayed_verification = predecessor_verify.build_verification(result_path)
    if verification != replayed_verification:
        raise ValueError("predecessor verification differs from independent raw-artifact replay")

    report_receipt = _read_object(report_receipt_path)
    predecessor.validate_report_receipt(report_receipt)
    if (
        report_receipt.get("result", {}).get("path") != _relative(result_path)
        or report_receipt.get("result", {}).get("file_sha256") != sha256_file(result_path)
        or report_receipt.get("verification", {}).get("path") != _relative(verification_path)
        or report_receipt.get("verification", {}).get("file_sha256") != sha256_file(verification_path)
    ):
        raise ValueError("predecessor report receipt does not bind the exact result and verification")

    final_binding = result["classifications"][-1]
    final_path = _repo_path(final_binding.get("path"), "predecessor H05 classification")
    final_classification = _read_object(final_path)
    predecessor.validate_classification(
        final_classification,
        len(predecessor.EPOCH_IDS) - 1,
        root=parent_root,
    )
    if (
        final_binding.get("epoch_id") != "H05"
        or final_binding.get("cycle_index") != 6
        or final_binding.get("file_sha256") != sha256_file(final_path)
        or final_binding.get("classification_sha256") != final_classification.get("classification_sha256")
    ):
        raise ValueError("predecessor H05 classification binding drifted")
    program_bindings = _validated_parent_program_state(
        parent_root=parent_root,
        required_artifacts=(
            result_path,
            verification_path,
            report_receipt_path,
            final_path,
        ),
    )

    routing = final_classification.get("routing") or {}
    predecessor_survivors = routing.get("mechanics_lanes_for_next_epoch")
    if not isinstance(predecessor_survivors, list) or any(
        not isinstance(item, str) for item in predecessor_survivors
    ):
        raise ValueError("predecessor H05 mechanics routing is invalid")
    eligible, dependency_pruned = _dependency_closed_lanes(predecessor_survivors)
    d1_row = final_classification.get("d1") or {}
    d1_continue = routing.get("continue_d1")
    if not isinstance(d1_continue, bool) or d1_row.get("continue_d1") is not d1_continue:
        raise ValueError("predecessor H05 D1 routing is invalid")

    return {
        "bindings": {
            **program_bindings,
            "result": _binding(result_path, result, "result_sha256"),
            "verification": _binding(verification_path, verification, "verification_sha256"),
            "report_receipt": _binding(report_receipt_path, report_receipt, "receipt_sha256"),
            "final_classification": _binding(
                final_path,
                final_classification,
                "classification_sha256",
            ),
        },
        "d1_classification": d1_row.get("classification"),
        "d1_initially_eligible": d1_continue,
        "mechanics_predecessor_survivors": list(predecessor_survivors),
        "mechanics_internal_dependencies": _mechanics_dependency_map(),
        "mechanics_dependency_pruned_lanes": dependency_pruned,
        "mechanics_initially_eligible_lanes": eligible,
    }


def _boundary_rules() -> dict[str, bool]:
    return {
        "predecessor_routes_are_immutable": True,
        "dependency_pruned_lanes_cannot_be_resurrected": True,
        "stable_d1_null_prunes_later_d1": True,
        "mixed_or_candidate_d1_continues": True,
        "mechanics_warning_prunes_lane": True,
        "past_or_active_epoch_mutation_allowed": False,
        "future_change_requires_new_sealed_child": True,
    }


def _planned_compute() -> dict[str, float]:
    return {
        "epoch_serial_seconds": planned_epoch_compute_seconds(),
        "horizon_serial_seconds": planned_horizon_compute_seconds(),
        "horizon_idle_eight_worker_hours": planned_horizon_compute_seconds() / IDLE_WORKERS / 3_600,
    }


def build_admission(
    *,
    parent_result_path: Path = DEFAULT_PARENT_RESULT,
    parent_verification_path: Path = DEFAULT_PARENT_VERIFICATION,
    parent_report_receipt_path: Path = DEFAULT_PARENT_REPORT_RECEIPT,
) -> dict[str, Any]:
    with _predecessor_identity_scope():
        parent = _validated_parent_state(
            result_path=parent_result_path,
            verification_path=parent_verification_path,
            report_receipt_path=parent_report_receipt_path,
        )
    core = {
        "schema": ADMISSION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "parent_horizon": parent["bindings"],
        "epoch_ids": list(EPOCH_IDS),
        "fresh_cycle_indices": list(EPOCH_CYCLES),
        "d1_predecessor_classification": parent["d1_classification"],
        "d1_initially_eligible": parent["d1_initially_eligible"],
        "mechanics_predecessor_survivors": parent["mechanics_predecessor_survivors"],
        "mechanics_internal_dependencies": parent["mechanics_internal_dependencies"],
        "mechanics_dependency_pruned_lanes": parent["mechanics_dependency_pruned_lanes"],
        "mechanics_initially_eligible_lanes": parent["mechanics_initially_eligible_lanes"],
        "boundary_rules": _boundary_rules(),
        "planned_compute": _planned_compute(),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "admission_sha256": canonical_sha256(core)}


def validate_admission(value: Mapping[str, Any]) -> None:
    _validate_seal(value, "admission_sha256", "successor horizon v2 admission")
    if set(value) != _ADMISSION_FIELDS:
        raise ValueError("successor horizon v2 admission field inventory drifted")
    created_at = value.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("successor horizon v2 admission timestamp is invalid")
    try:
        parsed = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("successor horizon v2 admission timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("successor horizon v2 admission timestamp is not UTC")

    parent_bindings = value.get("parent_horizon")
    if not isinstance(parent_bindings, Mapping) or set(parent_bindings) != _PARENT_BINDING_FIELDS:
        raise ValueError("successor horizon v2 parent bindings are invalid")
    paths: dict[str, Path] = {}
    for label in (
        "program_manifest",
        "supervisor_status",
        "result",
        "verification",
        "report_receipt",
        "final_classification",
    ):
        binding = parent_bindings.get(label)
        if not isinstance(binding, Mapping):
            raise ValueError(f"successor horizon v2 {label} binding is invalid")
        path = _repo_path(binding.get("path"), f"successor horizon v2 {label}")
        if not path.is_file() or binding.get("file_sha256") != sha256_file(path):
            raise ValueError(f"successor horizon v2 {label} file binding drifted")
        paths[label] = path

    with _predecessor_identity_scope():
        parent = _validated_parent_state(
            result_path=paths["result"],
            verification_path=paths["verification"],
            report_receipt_path=paths["report_receipt"],
        )
    expected_parent = parent["bindings"]
    if dict(parent_bindings) != expected_parent:
        raise ValueError("successor horizon v2 parent authority binding drifted")
    if paths["final_classification"] != _repo_path(
        expected_parent["final_classification"]["path"],
        "expected predecessor H05 classification",
    ):
        raise ValueError("successor horizon v2 final predecessor classification drifted")

    if (
        value.get("schema") != ADMISSION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_ids") != list(EPOCH_IDS)
        or value.get("fresh_cycle_indices") != list(EPOCH_CYCLES)
        or value.get("d1_predecessor_classification") != parent["d1_classification"]
        or value.get("d1_initially_eligible") is not parent["d1_initially_eligible"]
        or value.get("mechanics_predecessor_survivors") != parent["mechanics_predecessor_survivors"]
        or value.get("mechanics_internal_dependencies") != parent["mechanics_internal_dependencies"]
        or value.get("mechanics_dependency_pruned_lanes") != parent["mechanics_dependency_pruned_lanes"]
        or value.get("mechanics_initially_eligible_lanes") != parent["mechanics_initially_eligible_lanes"]
        or value.get("boundary_rules") != _boundary_rules()
        or value.get("planned_compute") != _planned_compute()
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("successor horizon v2 admission identity, routing, or safety drifted")


def admit(
    *,
    output: Path = DEFAULT_ADMISSION,
    parent_result_path: Path = DEFAULT_PARENT_RESULT,
    parent_verification_path: Path = DEFAULT_PARENT_VERIFICATION,
    parent_report_receipt_path: Path = DEFAULT_PARENT_REPORT_RECEIPT,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("successor horizon v2 admission must remain inside the repository")
    if output.is_file():
        existing = _read_object(output)
        validate_admission(existing)
        return existing
    value = build_admission(
        parent_result_path=parent_result_path,
        parent_verification_path=parent_verification_path,
        parent_report_receipt_path=parent_report_receipt_path,
    )
    validate_admission(value)
    atomic_write_json(output, value)
    return value


_V1_PATCH = {
    "PROGRAM_ID": PROGRAM_ID,
    "ADMISSION_SCHEMA": ADMISSION_SCHEMA,
    "SHARD_SCHEMA": SHARD_SCHEMA,
    "CLASSIFICATION_SCHEMA": CLASSIFICATION_SCHEMA,
    "RESULT_SCHEMA": RESULT_SCHEMA,
    "REPORT_RECEIPT_SCHEMA": REPORT_RECEIPT_SCHEMA,
    "CLAIM_SCOPE": CLAIM_SCOPE,
    "DEFAULT_ROOT": DEFAULT_ROOT,
    "DEFAULT_ADMISSION": DEFAULT_ADMISSION,
    "DEFAULT_RESULT": DEFAULT_RESULT,
    "DEFAULT_VERIFICATION": DEFAULT_VERIFICATION,
    "DEFAULT_REPORT": DEFAULT_REPORT,
    "DEFAULT_REPORT_RECEIPT": DEFAULT_REPORT_RECEIPT,
    "PROGRAM_MANIFEST": PROGRAM_MANIFEST,
    "EPOCH_IDS": EPOCH_IDS,
    "EPOCH_CYCLES": EPOCH_CYCLES,
    "validate_admission": validate_admission,
}


@contextmanager
def _v1_runtime_scope() -> Iterator[None]:
    """Temporarily point the immutable v1 engine at the v2 plan."""

    with _RUNTIME_LOCK:
        original = {name: getattr(predecessor, name) for name in _V1_PATCH}
        try:
            for name, replacement in _V1_PATCH.items():
                setattr(predecessor, name, replacement)
            yield
        finally:
            for name, prior in original.items():
                setattr(predecessor, name, prior)


# The pristine v1 engine identity, snapshotted at import before any _v1_runtime_scope runs.
_V1_IDENTITY_ORIGINAL = {name: getattr(predecessor, name) for name in _V1_PATCH}


@contextmanager
def _predecessor_identity_scope() -> Iterator[None]:
    """Restore the pristine v1 engine identity while validating the v1 parent.

    The parent-authority validation runs inside _v1_runtime_scope (the v1 module globals
    are repointed to the v2 plan), but the parent it checks is a genuine v1 result and
    must be validated against v1's own identity contract (RESULT_SCHEMA, PROGRAM_ID,
    CLAIM_SCOPE, ...), not v2's. This reentrant scope (_RUNTIME_LOCK is an RLock) restores
    the original v1 identity for the parent check, then restores whatever was active on
    exit. When no v2 scope is active it is a no-op (restores original to original)."""

    with _RUNTIME_LOCK:
        active = {name: getattr(predecessor, name) for name in _V1_IDENTITY_ORIGINAL}
        try:
            for name, original in _V1_IDENTITY_ORIGINAL.items():
                setattr(predecessor, name, original)
            yield
        finally:
            for name, prior in active.items():
                setattr(predecessor, name, prior)


def _d1_work(epoch_index: int, source_index: int) -> consolidated.WorkItem:
    with _v1_runtime_scope():
        return predecessor._d1_work(epoch_index, source_index)


def _mechanics_work(epoch_index: int, source_index: int) -> consolidated.WorkItem:
    with _v1_runtime_scope():
        return predecessor._mechanics_work(epoch_index, source_index)


def shard_work_items(*, epoch_index: int, lane: str, shard_index: int) -> tuple[consolidated.WorkItem, ...]:
    with _v1_runtime_scope():
        return _V1_SHARD_WORK_ITEMS(
            epoch_index=epoch_index,
            lane=lane,
            shard_index=shard_index,
        )


def run_shard(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
    epoch_index: int,
    lane: str,
    shard_index: int,
    idle_workers: int = IDLE_WORKERS,
    hawking_workers: int = HAWKING_WORKERS,
    retry_limit: int = RETRY_LIMIT,
) -> dict[str, Any]:
    if idle_workers != IDLE_WORKERS or hawking_workers != HAWKING_WORKERS:
        raise ValueError("successor horizon v2 requires exact --idle-workers 20 --hawking-workers 1")
    with _v1_runtime_scope():
        return _V1_RUN_SHARD(
            root=root,
            admission_path=admission_path,
            epoch_index=epoch_index,
            lane=lane,
            shard_index=shard_index,
            idle_workers=idle_workers,
            hawking_workers=hawking_workers,
            retry_limit=retry_limit,
        )


def validate_shard(
    value: Mapping[str, Any],
    *,
    epoch_index: int,
    lane: str,
    shard_index: int,
    root: Path = DEFAULT_ROOT,
) -> None:
    with _v1_runtime_scope():
        _V1_VALIDATE_SHARD(
            value,
            epoch_index=epoch_index,
            lane=lane,
            shard_index=shard_index,
            root=root,
        )


def build_classification(*, root: Path, admission_path: Path, epoch_index: int) -> dict[str, Any]:
    with _v1_runtime_scope():
        return _V1_BUILD_CLASSIFICATION(
            root=root,
            admission_path=admission_path,
            epoch_index=epoch_index,
        )


def validate_classification(
    value: Mapping[str, Any],
    epoch_index: int,
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    with _v1_runtime_scope():
        _V1_VALIDATE_CLASSIFICATION(value, epoch_index, root=root)


def classify_epoch(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
    epoch_index: int,
) -> dict[str, Any]:
    with _v1_runtime_scope():
        return _V1_CLASSIFY_EPOCH(
            root=root,
            admission_path=admission_path,
            epoch_index=epoch_index,
        )


def build_result(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
) -> dict[str, Any]:
    with _v1_runtime_scope():
        return _V1_BUILD_RESULT(root=root, admission_path=admission_path)


def validate_result(
    value: Mapping[str, Any],
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    with _v1_runtime_scope():
        _V1_VALIDATE_RESULT(value, root=root)


def aggregate(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    with _v1_runtime_scope():
        return _V1_AGGREGATE(root=root, admission_path=admission_path, output=output)


def validate_report_receipt(value: Mapping[str, Any]) -> None:
    with _v1_runtime_scope():
        _V1_VALIDATE_REPORT_RECEIPT(value)


def _validate_verification_for_report(
    value: Mapping[str, Any],
    *,
    result_path: Path,
) -> None:
    from mop.studies import generation1_successor_horizon_v2_verify as verifier

    if verifier.CLAIM_SCOPE != VERIFICATION_CLAIM_SCOPE:
        raise ValueError("successor horizon v2 verifier claim-scope authority drifted")
    verifier.validate_verification(value)
    _validate_seal(value, "verification_sha256", "successor horizon v2 verification")
    source = value.get("source") or {}
    result = _read_object(result_path)
    if (
        value.get("schema") != VERIFICATION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != VERIFICATION_CLAIM_SCOPE
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
        or source.get("result_sha256") != result.get("result_sha256")
    ):
        raise ValueError("successor horizon v2 verification is not a clean bound authority")


def render_report(
    *,
    result_path: Path = DEFAULT_RESULT,
    verification_path: Path = DEFAULT_VERIFICATION,
    report_path: Path = DEFAULT_REPORT,
    receipt_path: Path = DEFAULT_REPORT_RECEIPT,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    verification_path = Path(verification_path).resolve()
    report_path = Path(report_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    for path in (result_path, verification_path, report_path, receipt_path):
        if not path.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError("successor horizon v2 report paths must remain inside the repository")

    result = _read_object(result_path)
    validate_result(result)
    verification = _read_object(verification_path)
    _validate_verification_for_report(verification, result_path=result_path)
    if receipt_path.is_file():
        existing = _read_object(receipt_path)
        validate_report_receipt(existing)
        if not report_path.is_file():
            raise ValueError("existing successor horizon v2 report receipt has no current report argument")
        expected_bindings = {
            "result": {
                "path": _relative(result_path),
                "file_sha256": sha256_file(result_path),
            },
            "verification": {
                "path": _relative(verification_path),
                "file_sha256": sha256_file(verification_path),
            },
            "report": {
                "path": _relative(report_path),
                "file_sha256": sha256_file(report_path),
            },
        }
        if any(existing.get(label) != binding for label, binding in expected_bindings.items()):
            raise ValueError("existing successor horizon v2 report receipt binds different arguments")
        return existing

    lines = [
        "# Generation 1 successor evidence report, horizon v2",
        "",
        (
            "This generated view covers the append-only H06-H10 robustness horizon. "
            "Sealed JSON remains authoritative."
        ),
        "",
        "| Epoch | Fresh cycle | D1 classification | Terminal route | Mechanics lanes retained |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in result["classifications"]:
        lines.append(
            f"| {row['epoch_id']} | {row['cycle_index']} | {row['d1_classification']} | "
            f"{row['d1_terminal_route']} | {len(row['mechanics_lanes_for_next_epoch'])} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        (
            "These five epochs are same-code robustness evidence downstream of the verified v1 "
            "artifact chain. Dependency closure and later warning routes may only prune work. "
            "Neither the result nor verifier is independent scientific confirmation, activation "
            "authority, or Stage 3 promotion."
        ),
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
        f"Verification SHA-256: `{verification['verification_sha256']}`",
        "",
    ]
    encoded = ("\n".join(lines)).encode("utf-8")
    predecessor._atomic_write_text(report_path, encoded.decode("utf-8"))
    core = {
        "schema": REPORT_RECEIPT_SCHEMA,
        "program_id": PROGRAM_ID,
        "result": {"path": _relative(result_path), "file_sha256": sha256_file(result_path)},
        "verification": {
            "path": _relative(verification_path),
            "file_sha256": sha256_file(verification_path),
        },
        "report": {
            "path": _relative(report_path),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    validate_report_receipt(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "ADMISSION_SCHEMA",
    "CLASSIFICATION_SCHEMA",
    "CLAIM_SCOPE",
    "DEFAULT_ADMISSION",
    "DEFAULT_PARENT_REPORT_RECEIPT",
    "DEFAULT_PARENT_RESULT",
    "DEFAULT_PARENT_VERIFICATION",
    "DEFAULT_REPORT",
    "DEFAULT_REPORT_RECEIPT",
    "DEFAULT_RESULT",
    "DEFAULT_ROOT",
    "DEFAULT_VERIFICATION",
    "D1_PARTITIONS",
    "D1_SHARD_COUNT",
    "EPOCH_CYCLES",
    "EPOCH_IDS",
    "HAWKING_WORKERS",
    "IDLE_WORKERS",
    "MECHANICS_PARTITIONS",
    "MECHANICS_SHARD_COUNT",
    "PROGRAM_ID",
    "PROGRAM_MANIFEST",
    "REPORT_RECEIPT_SCHEMA",
    "RESULT_SCHEMA",
    "RETRY_LIMIT",
    "SHARD_SCHEMA",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_CLAIM_SCOPE",
    "admit",
    "aggregate",
    "build_admission",
    "build_classification",
    "build_result",
    "classify_epoch",
    "planned_epoch_compute_seconds",
    "planned_horizon_compute_seconds",
    "render_report",
    "run_shard",
    "shard_work_items",
    "validate_admission",
    "validate_classification",
    "validate_report_receipt",
    "validate_result",
    "validate_shard",
]
