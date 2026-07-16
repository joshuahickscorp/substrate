"""Independent verifier for the append-only Generation 1 successor horizon v2.

The verifier independently validates the complete v1 predecessor chain and the
dependency-closed H05 admission into H06.  It then reuses the separately
authored v1 streaming verifier engine inside the v2 runtime's locked, scoped
context to reconstruct H06-H10 shards, raw receipts, classifications, routing,
and seed boundaries.

This remains artifact verification over the same generator family.  It is not
independent scientific confirmation and cannot authorize activation.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as predecessor
from mop.studies import generation1_successor_horizon_v2 as horizon
from mop.studies import generation1_successor_horizon_verify as predecessor_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio import generation1_supervisor

VERIFICATION_SCHEMA = "mop-generation1-successor-horizon-verification/v2"
CLAIM_SCOPE = horizon.VERIFICATION_CLAIM_SCOPE

_PARENT_PROGRAM_ID = predecessor.PROGRAM_ID
_PARENT_PROGRAM_MANIFEST = Path("configs/campaign/generation1_successor_horizon_v1.json")
_PARENT_PROGRAM_ROOT = Path("runs/generation1") / _PARENT_PROGRAM_ID
_PARENT_RESULT_SCHEMA = predecessor.RESULT_SCHEMA
_PARENT_CLASSIFICATION_SCHEMA = predecessor.CLASSIFICATION_SCHEMA
_PARENT_REPORT_RECEIPT_SCHEMA = predecessor.REPORT_RECEIPT_SCHEMA
_PARENT_CLAIM_SCOPE = predecessor.CLAIM_SCOPE
_PARENT_EPOCH_IDS = tuple(predecessor.EPOCH_IDS)
_PARENT_EPOCH_CYCLES = tuple(predecessor.EPOCH_CYCLES)
_PARENT_VERIFICATION_SCHEMA = predecessor_verify.VERIFICATION_SCHEMA

_ADMISSION_FIELDS = {
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

_PARENT_CAPSULE_ARTIFACTS = {
    "result": "g1_horizon_aggregate",
    "verification": "g1_horizon_verify",
    "report_receipt": "g1_horizon_report",
    "final_classification": "g1_h05_classify",
}


canonical_bytes = predecessor_verify.canonical_bytes
canonical_sha256 = predecessor_verify.canonical_sha256
sha256_file = predecessor_verify.sha256_file


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sealed(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == canonical_sha256({key: item for key, item in value.items() if key != field})


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError("artifact path is outside the repository") from exc


def _load_binding(
    binding: Any,
    *,
    label: str,
    seal_field: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding is invalid")
    if set(binding) != {"path", "file_sha256", seal_field}:
        raise ValueError(f"{label} binding fields drifted")
    path = _repo_path(binding.get("path"), label)
    if not path.is_file() or binding.get("file_sha256") != sha256_file(path):
        raise ValueError(f"{label} file binding drifted")
    value = _read_object(path)
    if not _sealed(value, seal_field) or binding.get(seal_field) != value.get(seal_field):
        raise ValueError(f"{label} payload binding drifted")
    return path, value


def _dotted(payload: object, path: str) -> object:
    value = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _artifact_report(
    program: generation1_supervisor.Program,
    expectation: generation1_supervisor.ArtifactExpectation,
) -> dict[str, Any]:
    path = (program.repo_root / expectation.path).resolve()
    if not path.is_relative_to(program.repo_root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError(f"predecessor supervisor artifact is absent or unsafe: {expectation.path}")
    payload = _read_object(path)
    problems: list[str] = []
    if payload.get("schema") != expectation.schema:
        problems.append("artifact schema drifted")
    for dotted, expected in expectation.fields:
        if _dotted(payload, dotted) != expected:
            problems.append(f"{dotted}={_dotted(payload, dotted)!r}, expected {expected!r}")
    if expectation.seal_field is not None and not _sealed(
        payload,
        expectation.seal_field,
    ):
        problems.append("artifact self-seal mismatch")
    return {
        "path": expectation.path,
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "problems": problems,
        "all_ok": not problems,
    }


def _parent_artifact_paths(
    program: generation1_supervisor.Program,
) -> dict[str, Path]:
    capsules = {capsule.capsule_id: capsule for capsule in program.capsules}
    paths: dict[str, Path] = {}
    for label, capsule_id in _PARENT_CAPSULE_ARTIFACTS.items():
        capsule = capsules.get(capsule_id)
        if capsule is None or len(capsule.artifacts) != 1:
            raise ValueError(f"predecessor manifest {capsule_id} artifact inventory drifted")
        paths[label] = (program.repo_root / capsule.artifacts[0].path).resolve()
    return paths


def _validate_parent_supervisor_status(
    program: generation1_supervisor.Program,
    status: Mapping[str, Any],
) -> None:
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
    by_id = {capsule.capsule_id: capsule for capsule in program.capsules}
    if (
        status.get("schema") != generation1_supervisor.STATUS_SCHEMA
        or status.get("program_id") != _PARENT_PROGRAM_ID
        or status.get("program") != expected_program
        or status.get("execution_enabled") is not True
        or status.get("state") != "complete"
        or status.get("queue_head_sha256") != expected_queue_head
        or status.get("next_injection_sequence") != 1
        or status.get("accepted_injection_count") != 0
        or status.get("current_capsule") is not None
        or status.get("lane_reservation") is not None
        or status.get("problems") != []
        or not isinstance(capsule_rows, Mapping)
        or set(capsule_rows) != set(by_id)
    ):
        raise ValueError("predecessor terminal supervisor status drifted")

    for capsule_id, capsule in by_id.items():
        row = capsule_rows[capsule_id]
        expected_reports = [_artifact_report(program, expectation) for expectation in capsule.artifacts]
        if not isinstance(row, Mapping):
            raise ValueError(f"predecessor terminal capsule status drifted: {capsule_id}")
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
            or row.get("artifacts") != expected_reports
        ):
            raise ValueError(f"predecessor terminal capsule status drifted: {capsule_id}")


def _validate_parent_execution_authority(
    program_binding: Any,
    status_binding: Any,
    *,
    expected_parent_root: Path,
) -> dict[str, Path]:
    if not isinstance(program_binding, Mapping) or set(program_binding) != {
        "path",
        "file_sha256",
        "program_sha256",
    }:
        raise ValueError("predecessor program-manifest binding fields drifted")
    manifest_path = _repo_path(
        program_binding.get("path"),
        "predecessor program manifest",
    )
    canonical_manifest = (REPO_ROOT / _PARENT_PROGRAM_MANIFEST).resolve()
    if manifest_path != canonical_manifest:
        raise ValueError("predecessor program manifest is not canonical")
    try:
        program = generation1_supervisor.load_program(
            manifest_path,
            repo_root=REPO_ROOT,
        )
    except (generation1_supervisor.Generation1Refused, OSError, ValueError) as exc:
        raise ValueError(f"predecessor program manifest is invalid: {exc}") from exc
    canonical_root = (REPO_ROOT / _PARENT_PROGRAM_ROOT).resolve()
    if (
        program.path != canonical_manifest
        or program.program_id != _PARENT_PROGRAM_ID
        or program.program_root != canonical_root
        or program.program_root != expected_parent_root.resolve()
        or dict(program_binding)
        != {
            "path": _relative(program.path),
            "file_sha256": program.file_sha256,
            "program_sha256": program.program_sha256,
        }
    ):
        raise ValueError("predecessor program identity or root drifted")

    if not isinstance(status_binding, Mapping) or set(status_binding) != {
        "path",
        "file_sha256",
        "status_sha256",
    }:
        raise ValueError("predecessor supervisor-status binding fields drifted")
    if (
        _repo_path(
            status_binding.get("path"),
            "predecessor supervisor status",
        )
        != program.status_path.resolve()
    ):
        raise ValueError("predecessor supervisor-status path drifted")
    try:
        status = generation1_supervisor.read_status(program)
    except (generation1_supervisor.Generation1Refused, OSError, ValueError) as exc:
        raise ValueError(f"predecessor supervisor status is invalid: {exc}") from exc
    if dict(status_binding) != {
        "path": _relative(program.status_path),
        "file_sha256": sha256_file(program.status_path),
        "status_sha256": status.get("status_sha256"),
    }:
        raise ValueError("predecessor supervisor-status binding drifted")
    _validate_parent_supervisor_status(program, status)
    return _parent_artifact_paths(program)


def _clean_boundary(value: Mapping[str, Any], *, independent_field: bool = True) -> bool:
    return bool(
        value.get("complete") is True
        and value.get("problems") == []
        and value.get("activation_allowed") is False
        and value.get("scientific_promotion") is False
        and (not independent_field or value.get("independent_scientific_confirmation") is False)
    )


def _validate_parent_result(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        not _sealed(value, "result_sha256")
        or value.get("schema") != _PARENT_RESULT_SCHEMA
        or value.get("program_id") != _PARENT_PROGRAM_ID
        or value.get("claim_scope") != _PARENT_CLAIM_SCOPE
        or value.get("grid", {}).get("epoch_count") != len(_PARENT_EPOCH_IDS)
        or value.get("decision", {}).get("independent_scientific_confirmation") is not False
        or not _clean_boundary(value, independent_field=False)
    ):
        raise ValueError("predecessor horizon result boundary drifted")
    rows = value.get("classifications")
    if not isinstance(rows, list) or len(rows) != len(_PARENT_EPOCH_IDS):
        raise ValueError("predecessor horizon classification inventory is incomplete")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("epoch_id") != _PARENT_EPOCH_IDS[index]
            or row.get("cycle_index") != _PARENT_EPOCH_CYCLES[index]
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("file_sha256"), str)
            or not isinstance(row.get("classification_sha256"), str)
        ):
            raise ValueError("predecessor horizon classification binding drifted")
    return rows


def _validate_parent_verification(
    value: Mapping[str, Any],
    *,
    result_path: Path,
    result: Mapping[str, Any],
) -> None:
    checks = value.get("checks")
    expected_checks = {
        "result_seal_valid": True,
        "admission_and_consolidated_authority_valid": True,
        "all_shards_and_raw_artifacts_valid": True,
        "classifications_independently_reproduced": True,
        "all_seed_intervals_disjoint": True,
        "mutation_suite_passed": True,
        "independent_generator_family_present": False,
    }
    source = value.get("source")
    mutations = value.get("mutation_suite")
    recomputation = value.get("recomputation")
    if (
        not _sealed(value, "verification_sha256")
        or value.get("schema") != _PARENT_VERIFICATION_SCHEMA
        or value.get("program_id") != _PARENT_PROGRAM_ID
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or checks != expected_checks
        or mutations != {"count": 9, "rejected": 9, "all_rejected": True}
        or not isinstance(recomputation, Mapping)
        or recomputation.get("all_seed_intervals_disjoint") is not True
        or recomputation.get("bound_shard_count")
        != len(_PARENT_EPOCH_IDS) * (predecessor.D1_SHARD_COUNT + predecessor.MECHANICS_SHARD_COUNT)
        or not _clean_boundary(value)
        or not isinstance(source, Mapping)
        or source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
        or source.get("result_sha256") != result.get("result_sha256")
    ):
        raise ValueError("predecessor horizon verification boundary drifted")
    try:
        expected = predecessor_verify.build_verification(result_path)
        predecessor_verify.validate_verification(expected)
    except (OSError, ValueError) as exc:
        raise ValueError(f"predecessor horizon independent recomputation failed: {exc}") from exc
    if dict(value) != expected:
        raise ValueError("predecessor horizon verification differs from independent recomputation")


def _validate_parent_report_receipt(
    value: Mapping[str, Any],
    *,
    result_path: Path,
    verification_path: Path,
) -> None:
    if (
        not _sealed(value, "receipt_sha256")
        or value.get("schema") != _PARENT_REPORT_RECEIPT_SCHEMA
        or value.get("program_id") != _PARENT_PROGRAM_ID
        or not _clean_boundary(value, independent_field=False)
    ):
        raise ValueError("predecessor horizon report receipt boundary drifted")
    expected = {
        "result": result_path,
        "verification": verification_path,
    }
    for label, expected_path in expected.items():
        binding = value.get(label)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != _relative(expected_path)
            or binding.get("file_sha256") != sha256_file(expected_path)
        ):
            raise ValueError(f"predecessor horizon report {label} binding drifted")
    report = value.get("report")
    if not isinstance(report, Mapping) or set(report) != {"path", "file_sha256"}:
        raise ValueError("predecessor horizon report file binding is invalid")
    report_path = _repo_path(report.get("path"), "predecessor horizon report")
    if not report_path.is_file() or report.get("file_sha256") != sha256_file(report_path):
        raise ValueError("predecessor horizon report file binding drifted")


def _validate_h05(
    value: Mapping[str, Any],
    *,
    final_binding: Mapping[str, Any],
    previous_binding: Mapping[str, Any],
) -> tuple[str, bool, list[str]]:
    routing = value.get("routing")
    d1_row = value.get("d1")
    if (
        not _sealed(value, "classification_sha256")
        or value.get("schema") != _PARENT_CLASSIFICATION_SCHEMA
        or value.get("program_id") != _PARENT_PROGRAM_ID
        or value.get("claim_scope") != _PARENT_CLAIM_SCOPE
        or value.get("epoch_id") != "H05"
        or value.get("epoch_index") != len(_PARENT_EPOCH_IDS) - 1
        or value.get("cycle_index") != 6
        or value.get("parent_classification_sha256") != previous_binding.get("classification_sha256")
        or not _clean_boundary(value)
        or final_binding.get("classification_sha256") != value.get("classification_sha256")
        or not isinstance(routing, Mapping)
        or not isinstance(d1_row, Mapping)
    ):
        raise ValueError("predecessor H05 classification boundary drifted")
    d1_continue = routing.get("continue_d1")
    d1_classification = d1_row.get("classification")
    survivors = routing.get("mechanics_lanes_for_next_epoch")
    if (
        not isinstance(d1_continue, bool)
        or d1_row.get("continue_d1") is not d1_continue
        or not isinstance(d1_classification, str)
        or not d1_classification
        or not isinstance(survivors, list)
        or any(not isinstance(item, str) for item in survivors)
    ):
        raise ValueError("predecessor H05 routing is invalid")
    assert isinstance(d1_continue, bool)
    return d1_classification, d1_continue, list(survivors)


def _mechanics_dependency_map() -> dict[str, list[str]]:
    known = {lane.lane_id for lane in mechanics.LANES}
    return {
        lane.lane_id: [dependency for dependency in lane.dependencies if dependency in known]
        for lane in mechanics.LANES
    }


def _dependency_closed_lanes(
    predecessor_survivors: list[str],
) -> tuple[list[str], list[str]]:
    order = [lane.lane_id for lane in mechanics.LANES]
    known = set(order)
    if (
        len(predecessor_survivors) != len(set(predecessor_survivors))
        or not set(predecessor_survivors) <= known
        or predecessor_survivors != [lane_id for lane_id in order if lane_id in predecessor_survivors]
    ):
        raise ValueError("predecessor mechanics survivor inventory drifted")
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
        "epoch_serial_seconds": horizon.planned_epoch_compute_seconds(),
        "horizon_serial_seconds": horizon.planned_horizon_compute_seconds(),
        "horizon_idle_eight_worker_hours": horizon.planned_horizon_compute_seconds()
        / horizon.IDLE_WORKERS
        / 3_600,
    }


def _validate_admission(
    value: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if not _sealed(value, "admission_sha256") or set(value) != _ADMISSION_FIELDS:
        raise ValueError("successor horizon v2 admission seal or field inventory drifted")
    created_at = value.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("successor horizon v2 admission timestamp is invalid")
    try:
        parsed = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("successor horizon v2 admission timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("successor horizon v2 admission timestamp is not UTC")

    result_binding = result.get("admission")
    if not isinstance(result_binding, Mapping):
        raise ValueError("successor horizon v2 result admission binding is invalid")
    admission_path = _repo_path(
        result_binding.get("path"),
        "successor horizon v2 admission",
    )
    if (
        set(result_binding) != {"path", "file_sha256", "admission_sha256"}
        or not admission_path.is_file()
        or result_binding.get("file_sha256") != sha256_file(admission_path)
        or result_binding.get("admission_sha256") != value.get("admission_sha256")
    ):
        raise ValueError("successor horizon v2 result admission binding drifted")

    parent = value.get("parent_horizon")
    if not isinstance(parent, Mapping) or set(parent) != {
        "program_manifest",
        "supervisor_status",
        "result",
        "verification",
        "report_receipt",
        "final_classification",
    }:
        raise ValueError("successor horizon v2 parent authority inventory drifted")
    result_path, parent_result = _load_binding(
        parent["result"],
        label="predecessor horizon result",
        seal_field="result_sha256",
    )
    parent_rows = _validate_parent_result(parent_result)
    final_result_path = _repo_path(
        parent_rows[-1].get("path"),
        "predecessor H05 result binding",
    )
    if final_result_path.parent.name != "classifications":
        raise ValueError("predecessor H05 result binding is outside a classifications root")
    parent_root = final_result_path.parent.parent
    expected_paths = _validate_parent_execution_authority(
        parent["program_manifest"],
        parent["supervisor_status"],
        expected_parent_root=parent_root,
    )
    if result_path != expected_paths["result"]:
        raise ValueError("predecessor result is not the canonical manifest artifact")
    verification_path, parent_verification = _load_binding(
        parent["verification"],
        label="predecessor horizon verification",
        seal_field="verification_sha256",
    )
    if verification_path != expected_paths["verification"]:
        raise ValueError("predecessor verification is not the canonical manifest artifact")
    _validate_parent_verification(
        parent_verification,
        result_path=result_path,
        result=parent_result,
    )
    report_receipt_path, parent_report_receipt = _load_binding(
        parent["report_receipt"],
        label="predecessor horizon report receipt",
        seal_field="receipt_sha256",
    )
    if report_receipt_path != expected_paths["report_receipt"]:
        raise ValueError("predecessor report receipt is not the canonical manifest artifact")
    _validate_parent_report_receipt(
        parent_report_receipt,
        result_path=result_path,
        verification_path=verification_path,
    )
    h05_path, h05 = _load_binding(
        parent["final_classification"],
        label="predecessor H05 classification",
        seal_field="classification_sha256",
    )
    if h05_path != expected_paths["final_classification"]:
        raise ValueError("predecessor H05 classification is not the canonical manifest artifact")
    final_binding = parent_rows[-1]
    if (
        final_binding.get("path") != _relative(h05_path)
        or final_binding.get("file_sha256") != sha256_file(h05_path)
        or dict(parent["final_classification"])
        != {
            "path": final_binding.get("path"),
            "file_sha256": final_binding.get("file_sha256"),
            "classification_sha256": final_binding.get("classification_sha256"),
        }
    ):
        raise ValueError("predecessor H05 result binding drifted")
    d1_classification, d1_continue, predecessor_survivors = _validate_h05(
        h05,
        final_binding=final_binding,
        previous_binding=parent_rows[-2],
    )
    eligible, dependency_pruned = _dependency_closed_lanes(predecessor_survivors)

    if (
        value.get("schema") != horizon.ADMISSION_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("claim_scope") != horizon.CLAIM_SCOPE
        or value.get("epoch_ids") != list(horizon.EPOCH_IDS)
        or value.get("fresh_cycle_indices") != list(horizon.EPOCH_CYCLES)
        or value.get("d1_predecessor_classification") != d1_classification
        or value.get("d1_initially_eligible") is not d1_continue
        or value.get("mechanics_predecessor_survivors") != predecessor_survivors
        or value.get("mechanics_internal_dependencies") != _mechanics_dependency_map()
        or value.get("mechanics_dependency_pruned_lanes") != dependency_pruned
        or value.get("mechanics_initially_eligible_lanes") != eligible
        or value.get("boundary_rules") != _boundary_rules()
        or value.get("planned_compute") != _planned_compute()
        or not _clean_boundary(value)
    ):
        raise ValueError("successor horizon v2 admission predecessor, dependency, or safety boundary drifted")


@contextmanager
def _v1_verifier_scope() -> Iterator[None]:
    """Run the v1 verifier with the v2 runtime's locked scoped constants."""

    with horizon._v1_runtime_scope():
        yield


def _all_cycle_seed_spaces_disjoint() -> bool:
    """Prove exact fresh-cycle formulas remain disjoint through H10.

    Cycles zero and one are the consolidated campaign, two through six are the
    v1 horizon, and seven through eleven are this horizon.
    """

    d1_intervals: list[tuple[int, int]] = []
    mechanics_intervals: list[tuple[int, int]] = []
    for cycle in range(max(horizon.EPOCH_CYCLES) + 1):
        for source_index in range(predecessor_verify.d1.DEFAULT_RUNG_COUNT):
            config = predecessor_verify._expected_d1_config(source_index, cycle)
            d1_intervals.extend(
                (
                    int(config[f"{kind}_seed_start"]),
                    int(config[f"{kind}_seed_start"]) + predecessor_verify.d1.SEEDS_PER_RUNG,
                )
                for kind in ("train", "heldout")
            )
        for source in mechanics.WORK_ITEMS:
            start = (
                source.seed_start
                + consolidated.MECHANICS_FRESH_BASE
                + cycle * consolidated.MECHANICS_CYCLE_STRIDE
            )
            mechanics_intervals.append((start, start + source.seed_count))
    return bool(
        predecessor_verify._intervals_disjoint(d1_intervals)
        and predecessor_verify._intervals_disjoint(mechanics_intervals)
        and max(end for _, end in d1_intervals) < min(start for start, _ in mechanics_intervals)
    )


def _validate_v2_shard_boundaries(result: Mapping[str, Any]) -> None:
    """Cover v2 fields omitted by the reused v1 streaming shard validator."""

    expected = {
        (epoch_id, lane, shard_index)
        for epoch_id in horizon.EPOCH_IDS
        for lane, count in (
            ("d1", horizon.D1_SHARD_COUNT),
            ("mechanics", horizon.MECHANICS_SHARD_COUNT),
        )
        for shard_index in range(count)
    }
    bindings = result.get("shard_index")
    if not isinstance(bindings, list) or len(bindings) != len(expected):
        raise ValueError("successor horizon v2 shard boundary inventory drifted")
    seen: set[tuple[str, str, int]] = set()
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or isinstance(binding.get("shard_index"), bool)
            or not isinstance(binding.get("shard_index"), int)
        ):
            raise ValueError("successor horizon v2 shard boundary binding is invalid")
        identity = (
            str(binding.get("epoch_id")),
            str(binding.get("lane")),
            int(binding["shard_index"]),
        )
        if identity not in expected or identity in seen:
            raise ValueError("successor horizon v2 shard boundary identity is missing or duplicated")
        epoch_id, lane, shard_index = identity
        epoch_index = horizon.EPOCH_IDS.index(epoch_id)
        path = _repo_path(
            binding.get("path"),
            "successor horizon v2 shard",
        )
        if not path.is_file() or path.is_symlink() or binding.get("file_sha256") != sha256_file(path):
            raise ValueError("successor horizon v2 shard file binding drifted")
        shard = _read_object(path)
        if (
            not _sealed(shard, "shard_sha256")
            or binding.get("shard_sha256") != shard.get("shard_sha256")
            or shard.get("schema") != horizon.SHARD_SCHEMA
            or shard.get("program_id") != horizon.PROGRAM_ID
            or shard.get("claim_scope") != horizon.CLAIM_SCOPE
            or shard.get("epoch_id") != epoch_id
            or shard.get("cycle_index") != horizon.EPOCH_CYCLES[epoch_index]
            or shard.get("lane") != lane
            or shard.get("shard_index") != shard_index
            or not _clean_boundary(shard)
        ):
            raise ValueError("successor horizon v2 shard claim, cycle, identity, or safety drifted")
        seen.add(identity)
    if seen != expected:
        raise ValueError("successor horizon v2 shard boundary inventory is incomplete")


def _validate_result_shell(value: Mapping[str, Any]) -> None:
    with _v1_verifier_scope():
        predecessor_verify._validate_result_shell(value)


def _mutation_suite(result: Mapping[str, Any]) -> dict[str, Any]:
    with _v1_verifier_scope():
        return predecessor_verify._mutation_suite(result)


def build_verification(
    result_path: Path = horizon.DEFAULT_RESULT,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    if not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("successor horizon v2 result is outside the repository")
    result = _read_object(result_path)
    admission_binding = result.get("admission")
    if not isinstance(admission_binding, Mapping):
        raise ValueError("successor horizon v2 result admission binding is invalid")
    admission_path = _repo_path(
        admission_binding.get("path"),
        "successor horizon v2 admission",
    )
    admission = _read_object(admission_path)
    _validate_admission(admission, result)
    _validate_v2_shard_boundaries(result)

    with _v1_verifier_scope():
        predecessor_verify._validate_result_shell(result)
        classifications = predecessor_verify._classification_rows(result)
        recomputation = predecessor_verify._recompute(
            result,
            admission,
            classifications,
        )
        mutations = predecessor_verify._mutation_suite(result)

    grid = result.get("grid") or {}
    if (
        grid.get("d1_shard_count") != len(horizon.EPOCH_IDS) * horizon.D1_SHARD_COUNT
        or grid.get("mechanics_shard_count") != len(horizon.EPOCH_IDS) * horizon.MECHANICS_SHARD_COUNT
        or grid.get("executed_d1_rung_count") != recomputation["executed_d1_rung_count"]
        or grid.get("executed_mechanics_rung_count") != recomputation["executed_mechanics_rung_count"]
    ):
        raise ValueError("successor horizon v2 result grid differs from independent inventory")
    all_cycle_seed_spaces_disjoint = _all_cycle_seed_spaces_disjoint()
    if not all_cycle_seed_spaces_disjoint:
        raise ValueError("successor horizon v2 overlaps a predecessor seed space")

    checks = {
        "result_seal_valid": True,
        "predecessor_authority_chain_valid": True,
        "dependency_closed_admission_valid": True,
        "all_shards_and_raw_artifacts_valid": True,
        "classifications_independently_reproduced": True,
        "all_seed_intervals_disjoint": recomputation["all_seed_intervals_disjoint"],
        "predecessor_and_v2_seed_spaces_disjoint": all_cycle_seed_spaces_disjoint,
        "mutation_suite_passed": mutations["all_rejected"],
        "independent_generator_family_present": False,
    }
    core = {
        "schema": VERIFICATION_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "source": {
            "path": _relative(result_path),
            "file_sha256": sha256_file(result_path),
            "result_sha256": result["result_sha256"],
        },
        "checks": checks,
        "recomputation": recomputation,
        "mutation_suite": mutations,
        "verification_complete": all(
            value is True for key, value in checks.items() if key != "independent_generator_family_present"
        ),
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def _validate_recomputation_shape(value: Any) -> None:
    expected_fields = {
        "d1_classifications",
        "mechanics_lanes_retained",
        "d1_interval_count",
        "mechanics_interval_count",
        "all_seed_intervals_disjoint",
        "bound_shard_count",
        "executed_d1_rung_count",
        "executed_mechanics_rung_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("successor horizon v2 verification recomputation fields drifted")
    d1_rows = value.get("d1_classifications")
    mechanics_rows = value.get("mechanics_lanes_retained")
    allowed_d1 = {
        "stable_candidate_trace",
        "stable_null",
        "mixed_or_seed_sensitive",
        "not_run_pruned",
    }
    lane_order = [lane.lane_id for lane in mechanics.LANES]
    if (
        not isinstance(d1_rows, Mapping)
        or set(d1_rows) != set(horizon.EPOCH_IDS)
        or any(item not in allowed_d1 for item in d1_rows.values())
        or not isinstance(mechanics_rows, Mapping)
        or set(mechanics_rows) != set(horizon.EPOCH_IDS)
    ):
        raise ValueError("successor horizon v2 verification classification recomputation drifted")
    for retained in mechanics_rows.values():
        if (
            not isinstance(retained, list)
            or any(not isinstance(item, str) for item in retained)
            or len(retained) != len(set(retained))
            or retained != [lane_id for lane_id in lane_order if lane_id in retained]
        ):
            raise ValueError("successor horizon v2 verification mechanics recomputation drifted")

    numeric_fields = (
        "d1_interval_count",
        "mechanics_interval_count",
        "bound_shard_count",
        "executed_d1_rung_count",
        "executed_mechanics_rung_count",
    )
    if any(
        isinstance(value.get(field), bool) or not isinstance(value.get(field), int) or int(value[field]) < 0
        for field in numeric_fields
    ):
        raise ValueError("successor horizon v2 verification recomputation counts are invalid")
    if (
        value.get("all_seed_intervals_disjoint") is not True
        or value.get("bound_shard_count")
        != len(horizon.EPOCH_IDS) * (horizon.D1_SHARD_COUNT + horizon.MECHANICS_SHARD_COUNT)
        or value.get("d1_interval_count") != 2 * int(value["executed_d1_rung_count"])
        or value.get("mechanics_interval_count") != int(value["executed_mechanics_rung_count"])
        or int(value["executed_d1_rung_count"])
        > len(horizon.EPOCH_IDS) * predecessor_verify.d1.DEFAULT_RUNG_COUNT
        or int(value["executed_mechanics_rung_count"]) > len(horizon.EPOCH_IDS) * len(mechanics.WORK_ITEMS)
    ):
        raise ValueError("successor horizon v2 verification recomputation boundary drifted")


def validate_verification(value: Mapping[str, Any]) -> None:
    expected_checks = {
        "result_seal_valid",
        "predecessor_authority_chain_valid",
        "dependency_closed_admission_valid",
        "all_shards_and_raw_artifacts_valid",
        "classifications_independently_reproduced",
        "all_seed_intervals_disjoint",
        "predecessor_and_v2_seed_spaces_disjoint",
        "mutation_suite_passed",
        "independent_generator_family_present",
    }
    checks = value.get("checks")
    expected_fields = {
        "schema",
        "program_id",
        "claim_scope",
        "source",
        "checks",
        "recomputation",
        "mutation_suite",
        "verification_complete",
        "independent_scientific_confirmation",
        "complete",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "verification_sha256",
    }
    if (
        not _sealed(value, "verification_sha256")
        or set(value) != expected_fields
        or value.get("schema") != VERIFICATION_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or checks.get("independent_generator_family_present") is not False
        or any(
            checks.get(name) is not True
            for name in expected_checks
            if name != "independent_generator_family_present"
        )
        or value.get("mutation_suite") != {"count": 9, "rejected": 9, "all_rejected": True}
        or not _clean_boundary(value)
    ):
        raise ValueError("successor horizon v2 verification identity or safety drifted")
    _validate_recomputation_shape(value.get("recomputation"))
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "file_sha256",
        "result_sha256",
    }:
        raise ValueError("successor horizon v2 verification source is invalid")
    path = _repo_path(source.get("path"), "successor horizon v2 result")
    result = _read_object(path)
    _validate_result_shell(result)
    if source.get("file_sha256") != sha256_file(path) or source.get("result_sha256") != result.get(
        "result_sha256"
    ):
        raise ValueError("successor horizon v2 verification source binding drifted")
    try:
        expected = build_verification(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"successor horizon v2 verification replay failed: {exc}") from exc
    if dict(value) != expected:
        raise ValueError("successor horizon v2 verification differs from independent replay")


def verify(
    *,
    result_path: Path = horizon.DEFAULT_RESULT,
    output: Path = horizon.DEFAULT_VERIFICATION,
) -> dict[str, Any]:
    value = build_verification(result_path)
    validate_verification(value)
    consolidated.atomic_write_json(Path(output).resolve(), value)
    return value


__all__ = [
    "CLAIM_SCOPE",
    "VERIFICATION_SCHEMA",
    "build_verification",
    "canonical_bytes",
    "canonical_sha256",
    "validate_verification",
    "verify",
]
