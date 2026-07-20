"""Read-only terminal release audit for the Generation-1 empirical campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..studio.generation1_supervisor import MAX_CAPSULE_ATTEMPTS, MAX_RUNTIME_EVENTS
from .generation1_cognitive_corpus import build_corpus
from .generation1_cognitive_corpus_verify import verify_corpus
from .generation1_evidence_synthesis_verify import verify_evidence_synthesis
from .generation1_report import build_report

SCHEMA = "mop-generation1-release-audit/v1"
FAILURE_SCHEMA = "mop-generation1-release-audit-failure/v1"
CLAIM_SCOPE = (
    "terminal publication integrity for the Generation-1 C0 discovery corpus only; "
    "no activation, substrate-formation, natural-world, or scientific-promotion claim"
)

PROGRAM_SCHEMA = "mop-generation1-program/v1"
STATE_SCHEMA = "mop-generation1-state/v1"
STATUS_SCHEMA = "mop-generation1-status/v1"
INJECTION_SCHEMA = "mop-generation1-injection/v1"
INJECTION_RECEIPT_SCHEMA = "mop-generation1-injection-receipt/v1"
CAPSULE_SCHEMA = "mop-generation1-capsule/v1"

PROOF_CONTRACTS = {
    "corpus": ("mop-generation1-cognitive-corpus/v2", "corpus_sha256"),
    "corpus_verification": (
        "mop-generation1-cognitive-corpus-verification/v2",
        "verification_sha256",
    ),
    "empirical_report": ("mop-generation1-empirical-report/v2", "report_sha256"),
    "evidence_synthesis": ("mop-generation1-evidence-synthesis/v1", "synthesis_sha256"),
    "evidence_synthesis_verification": (
        "mop-generation1-evidence-synthesis-verification/v1",
        "verification_sha256",
    ),
}

REQUIRED_SYNTHESIS_BOUNDARIES = frozenset(
    {
        "context_disjoint_actor_niches",
        "nonlinear_perspective_cooperation",
        "integrated_substrate_advantage",
        "natural_world_generality",
    }
)

REQUIRED_CORPUS_CHECKS = frozenset(
    {
        "corpus_schema",
        "corpus_self_hash",
        "config_schema",
        "config_hash_bound",
        "experiment_set_exact",
        "seed_set_exact",
        "all_seed_receipts_valid",
        "all_attempt_receipts_valid",
        "all_cell_authorities_valid",
        "seed_authority_exact",
        "no_pseudoreplication",
        "independent_summary_match",
        "directional_inference_fail_closed",
        "full_regeneration_match",
        "corpus_complete",
        "promotion_blocked",
        "all_mutations_rejected",
    }
)

REQUIRED_SYNTHESIS_VERIFICATION_CHECKS = frozenset(
    {
        "source_corpus_schema",
        "source_corpus_self_seal",
        "source_corpus_verification_schema",
        "source_corpus_verification_self_seal",
        "source_corpus_verification_complete",
        "source_corpus_verification_all_checks",
        "source_corpus_verification_file_binding",
        "source_report_schema",
        "source_report_self_seal",
        "source_report_corpus_file_binding",
        "source_report_verification_file_binding",
        "source_program_state_schema",
        "source_program_state_current_self_seal",
        "source_base_capsules_complete",
        "source_source_receipts_constructible",
        "source_program_state_path_present",
        "synthesis_schema",
        "synthesis_self_seal",
        "required_corpus_verification_checks_present",
        "top_level_contract_exact",
        "claim_scope_exact",
        "source_receipts_exact",
        "trace_index_exact",
        "recurrence_analysis_exact",
        "negative_evidence_register_exact",
        "claim_boundaries_exact",
        "claim_boundaries_valid",
        "mechanism_dimension_matrix_exact",
        "mechanism_dimension_matrix_valid",
        "conditional_priority_queue_exact",
        "conditional_priority_queue_valid",
        "base_runtime_accounting_exact",
        "base_runtime_subset_self_hash",
        "promotion_and_activation_blocked",
        "all_synthesis_mutations_rejected",
    }
)

REQUIRED_CORPUS_MUTATIONS = frozenset(
    {
        "plan_tamper",
        "seed_authority_tamper",
        "evidence_class_tamper",
        "attempt_tamper",
        "summary_tamper",
    }
)

REQUIRED_SYNTHESIS_MUTATIONS = frozenset(
    {
        "trace_membership_removed",
        "dimension_removed",
        "successor_dimension_status_escalated",
        "source_receipt_drifted",
        "activation_enabled",
        "scientific_promotion_enabled",
        "claim_scope_escalated",
        "undeclared_top_level_claim_added",
    }
)

SYNTHESIS_CLAIM_SCOPE = (
    "read-only Generation-1 C0 evidence synthesis and successor-hypothesis selection; "
    "no context-disjoint niche, cooperation, integrated-substrate, activation, or "
    "scientific-promotion claim"
)

SYNTHESIS_VERIFICATION_CLAIM_SCOPE = (
    "independent verification of the derived G1-C0 evidence synthesis only; "
    "no mechanism or substrate promotion"
)

REPORT_CLAIM_SCOPE = (
    "programmatic Generation-1 capability corpus and hypothesis-selection evidence only; "
    "no integrated architecture or natural-world promotion"
)

STATE_KEYS = frozenset(
    {
        "schema",
        "program_id",
        "program",
        "supervisor",
        "execution_enabled",
        "status",
        "queue_head_sha256",
        "next_injection_sequence",
        "accepted_injections",
        "processed_injection_files",
        "capsules",
        "current_capsule",
        "last_admission",
        "lane_reservation",
        "started_at",
        "updated_at",
        "finished_at",
        "problems",
        "state_sha256",
    }
)

STATUS_KEYS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "program",
        "supervisor",
        "execution_enabled",
        "state",
        "queue_head_sha256",
        "next_injection_sequence",
        "accepted_injection_count",
        "current_capsule",
        "capsules",
        "last_admission",
        "lane_reservation",
        "problems",
        "status_sha256",
    }
)

CAPSULE_STATE_KEYS = frozenset(
    {
        "id",
        "kind",
        "priority",
        "depends_on",
        "capsule_sha256",
        "source",
        "status",
        "attempts",
        "child_pid",
        "child_create_time",
        "started_at",
        "finished_at",
        "returncode",
        "artifacts",
        "last_problem",
        "runtime",
    }
)

CAPSULE_KEYS = frozenset(
    {
        "schema",
        "id",
        "kind",
        "priority",
        "depends_on",
        "command",
        "cwd",
        "environment",
        "resources",
        "artifacts",
        "authorities",
        "capsule_sha256",
    }
)

RUNTIME_KEYS = frozenset(
    {
        "sample_count",
        "peak_process_tree_rss_bytes",
        "minimum_memory_available_gb",
        "minimum_memory_available_percent",
        "minimum_memory_pressure_free_percent",
        "maximum_swap_used_gb",
        "minimum_disk_free_gb",
        "thermal_statuses",
        "power_sources",
        "reservation_count",
        "last_reservation",
        "resource_stop_count",
        "retry_count",
        "safety_state",
        "last_sample",
        "event_count",
        "events_dropped",
        "events",
    }
)

FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "activation_allowed",
        "scientific_promotion",
        "substrate_formed",
        "substrate_formation_established",
    }
)

TERMINAL_RUNTIME_EVENT_NAMES = frozenset(
    {
        "lane-reserved",
        "lane-released",
        "wall-boundary-stop",
        "runtime-resource-stop",
        "capsule-retry-scheduled",
        "recovery-child-gone",
        "child-started",
        "capsule-complete",
    }
)


@dataclass(frozen=True, slots=True)
class ReleaseContract:

    program_id: str = "generation1-empirical-cognitive-corpus-v2"
    program_sha256: str = "31bd05924a9ebd9a97f05acc46503bb4ab6f9a37ff4e8c9e3c89ceecfa01bfa8"
    base_capsule_count: int = 43
    total_capsule_count: int = 45
    injection_id: str = "inj-000001-bc8051b78fba"
    injection_sha256: str = "91f6ae230a5b2bf12fb7ff7d67cd89cb1ba04d6ad68b26fdd9ea70097cdddd23"
    injection_file_sha256: str = "ec150f33d86d2d7e6f02e7ece73b1c0a7d66c0e9f965f3f2160c680dee7651b3"
    injection_sequence: int = 1
    next_injection_sequence: int = 2
    previous_queue_head_sha256: str = "bd03c9fcdf89606c1a39c834dc398885bf225a3852e2b903570e847ecbaf7e34"
    final_queue_head_sha256: str = "2dda8a13000eba58c95a5e389f7ff6d50b302a80762408e69e77f6f47a33b25d"
    injection_receipt_sha256: str = "dc4f92384062ae4d2934efeed8c1e2a3a8a865257d74c5d3c3c9f11d853e6e31"
    injected_capsule_ids: tuple[str, ...] = (
        "g1_evidence_synthesis",
        "g1_evidence_synthesis_verify",
    )


@dataclass(frozen=True, slots=True)
class ReleasePaths:

    repo_root: Path
    audit_module: Path
    audit_wrapper: Path
    supervisor_module: Path
    program: Path
    state: Path
    status: Path
    injection: Path
    injection_receipt: Path
    corpus_config: Path
    corpus_run_root: Path
    corpus: Path
    corpus_verification: Path
    empirical_report: Path
    evidence_synthesis: Path
    evidence_synthesis_verification: Path

    @classmethod
    def defaults(cls, repo_root: Path = REPO_ROOT) -> ReleasePaths:
        root = repo_root.resolve()
        campaign = root / "runs/generation1/generation1-empirical-cognitive-corpus-v2"
        return cls(
            repo_root=root,
            audit_module=root / "src/mop/studies/generation1_release_audit.py",
            audit_wrapper=root / "scripts/audit_generation1_release.py",
            supervisor_module=root / "src/mop/studio/generation1_supervisor.py",
            program=root / "configs/campaign/generation1_empirical_program_v2.json",
            state=campaign / "program_state.json",
            status=campaign / "current_status.json",
            injection=campaign / "control/inbox/inj-000001-bc8051b78fba.json",
            injection_receipt=(
                campaign / "control/injection_receipts/accepted/inj-000001-bc8051b78fba-ec150f33d86d.json"
            ),
            corpus_config=root / "configs/experiment/generation1_cognitive_corpus.json",
            corpus_run_root=root / "runs/generation1/cognitive_corpus",
            corpus=root / "proof/GENERATION1_COGNITIVE_CORPUS.json",
            corpus_verification=(root / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"),
            empirical_report=root / "proof/GENERATION1_EMPIRICAL_REPORT.json",
            evidence_synthesis=root / "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            evidence_synthesis_verification=(root / "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json"),
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bytes(path: Path, label: str) -> tuple[Path, bytes]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or not stat.st_size:
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    raw = resolved.read_bytes()
    if len(raw) != stat.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    return resolved, raw


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _, raw = _load_bytes(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, raw


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == _canonical_sha256(core)


def _repo_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"release input escapes repository: {path}")
    return str(resolved.relative_to(root))


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, root: Path, label: str) -> None:
    lexical_root = _lexical_absolute(root)
    lexical_path = _lexical_absolute(path)
    if not lexical_path.is_relative_to(lexical_root):
        raise ValueError(f"{label} escapes the repository: {path}")
    current = lexical_root
    for component in lexical_path.relative_to(lexical_root).parts:
        current /= component
        try:
            current.lstat()
        except OSError as exc:
            raise ValueError(f"{label} has an unreadable path component: {current}") from exc
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink path component: {current}")


def _stable_file(path: Path, raw: bytes, root: Path) -> bool:
    try:
        _assert_no_symlink_components(path, root, "release input")
        return path.is_file() and _sha256_file(path) == _sha256_bytes(raw)
    except (OSError, ValueError):
        return False


def _snapshot_file_entry(path: Path, root: Path, label: str) -> dict[str, Any]:
    _assert_no_symlink_components(path, root, label)
    try:
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if not path.is_file() or identity_before != identity_after:
        raise ValueError(f"{label} changed while being snapshotted: {path}")
    return {
        "path": _repo_path(path, root),
        "sha256": digest,
        "bytes": after.st_size,
    }


def _snapshot_receipt(entries: list[dict[str, Any]], **identity: object) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda row: str(row["path"]))
    return {
        **identity,
        "file_count": len(ordered),
        "total_bytes": sum(int(row["bytes"]) for row in ordered),
        "entries_sha256": _canonical_sha256(ordered),
    }


def _snapshot_paths(paths: tuple[Path, ...], root: Path, label: str) -> dict[str, Any]:
    unique = sorted(set(paths), key=lambda path: _repo_path(path, root))
    entries = [_snapshot_file_entry(path, root, label) for path in unique]
    return _snapshot_receipt(entries)


def _snapshot_tree(path: Path, root: Path, label: str) -> dict[str, Any]:
    _assert_no_symlink_components(path, root, label)
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: str(item)):
        _assert_no_symlink_components(candidate, root, label)
        if candidate.is_file():
            entries.append(_snapshot_file_entry(candidate, root, label))
        elif not candidate.is_dir():
            raise ValueError(f"{label} contains a non-regular entry: {candidate}")
    return _snapshot_receipt(entries, root=_repo_path(path, root))


def _safe_repo_file(path_value: object, root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("repository-relative path is malformed")
    candidate = root / path_value
    _assert_no_symlink_components(candidate, root, "repository authority")
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes repository: {path_value}")
    return path


def _source_receipt(
    path: Path,
    payload: dict[str, Any],
    raw: bytes,
    root: Path,
    seal_field: str,
) -> dict[str, Any]:
    return {
        "path": _repo_path(path, root),
        "sha256": _sha256_bytes(raw),
        "schema": payload.get("schema"),
        "seal_field": seal_field,
        "seal_sha256": payload.get(seal_field),
    }


def _file_receipt(path: Path, raw: bytes, root: Path) -> dict[str, Any]:
    return {
        "path": _repo_path(path, root),
        "sha256": _sha256_bytes(raw),
        "bytes": len(raw),
    }


def _input_paths(paths: ReleasePaths) -> tuple[Path, ...]:
    return (
        paths.audit_module,
        paths.audit_wrapper,
        paths.supervisor_module,
        paths.program,
        paths.state,
        paths.status,
        paths.injection,
        paths.injection_receipt,
        paths.corpus_config,
        paths.corpus,
        paths.corpus_verification,
        paths.empirical_report,
        paths.evidence_synthesis,
        paths.evidence_synthesis_verification,
    )


def _dotted(payload: object, dotted: str) -> object:
    current = payload
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _claim_flags_disabled(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_CLAIM_KEYS and item is not False:
                return False
            if not _claim_flags_disabled(item):
                return False
        return True
    if isinstance(value, list):
        return all(_claim_flags_disabled(item) for item in value)
    return True


def _authority_rows(program: dict[str, Any], injection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy = program.get("policy")
    if isinstance(policy, dict):
        rows.append(policy)
    authorities = program.get("authorities")
    if isinstance(authorities, list):
        rows.extend(item for item in authorities if isinstance(item, dict))
    for capsule in [*program.get("capsules", []), *injection.get("capsules", [])]:
        if not isinstance(capsule, dict):
            continue
        capsule_authorities = capsule.get("authorities")
        if isinstance(capsule_authorities, list):
            rows.extend(item for item in capsule_authorities if isinstance(item, dict))
    return rows


def _snapshot_authorities(
    program: dict[str, Any],
    injection: dict[str, Any],
    root: Path,
) -> tuple[tuple[tuple[Path, bytes], ...], dict[str, Any]]:
    rows = _authority_rows(program, injection)
    if not rows:
        raise ValueError("program and injection declare no file authorities")
    snapshots: dict[Path, bytes] = {}
    for row in rows:
        path = _safe_repo_file(row.get("path"), root)
        declared = row.get("sha256")
        if not isinstance(declared, str):
            raise ValueError(f"file authority has no SHA-256 identity: {row.get('path')}")
        if path not in snapshots:
            _, snapshots[path] = _load_bytes(path, f"file authority {row.get('path')}")
        if _sha256_bytes(snapshots[path]) != declared:
            raise ValueError(f"file authority drifted: {row.get('path')}")
    entries = [
        {
            "path": _repo_path(path, root),
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        }
        for path, raw in sorted(snapshots.items(), key=lambda item: _repo_path(item[0], root))
    ]
    receipt = {
        "declared_row_count": len(rows),
        "unique_file_count": len(entries),
        "entries_sha256": _canonical_sha256(entries),
    }
    return tuple(sorted(snapshots.items(), key=lambda item: str(item[0]))), receipt


def _capsules(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = payload.get("capsules")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} capsules are malformed")
    if any(
        set(row) != CAPSULE_KEYS
        or row.get("schema") != CAPSULE_SCHEMA
        or not _valid_seal(row, "capsule_sha256")
        for row in rows
    ):
        raise ValueError(f"{label} contains a capsule self-seal mismatch")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} capsule ids are malformed or duplicated")
    return rows


def _dependency_graph_valid(capsules: dict[str, tuple[dict[str, Any], str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(capsule_id: str) -> bool:
        if capsule_id in visiting:
            return False
        if capsule_id in visited:
            return True
        raw = capsules[capsule_id][0].get("depends_on")
        if (
            not isinstance(raw, list)
            or any(not isinstance(item, str) for item in raw)
            or len(raw) != len(set(raw))
            or capsule_id in raw
            or any(item not in capsules for item in raw)
        ):
            return False
        visiting.add(capsule_id)
        if any(not visit(item) for item in raw):
            return False
        visiting.remove(capsule_id)
        visited.add(capsule_id)
        return True

    return all(visit(capsule_id) for capsule_id in sorted(capsules))


def _genesis_head(program: dict[str, Any], base_capsules: list[dict[str, Any]]) -> str:
    return _canonical_sha256(
        {
            "program_sha256": program.get("program_sha256"),
            "base_capsules": [row["capsule_sha256"] for row in base_capsules],
        }
    )


def _injected_head(previous: str, injection: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return _canonical_sha256(
        {
            "previous_head_sha256": previous,
            "injection_sha256": injection.get("injection_sha256"),
            "capsules": [row["capsule_sha256"] for row in rows],
        }
    )


def _expected_capsule_map(
    base: list[dict[str, Any]], injected: list[dict[str, Any]], injection_id: str
) -> dict[str, tuple[dict[str, Any], str]]:
    result = {str(row["id"]): (row, "base") for row in base}
    for row in injected:
        capsule_id = str(row["id"])
        if capsule_id in result:
            raise ValueError(f"injected capsule collides with base capsule: {capsule_id}")
        result[capsule_id] = (row, f"injection:{injection_id}")
    return result


def _declared_artifact_paths(
    capsules: dict[str, tuple[dict[str, Any], str]],
    root: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for capsule, _source in capsules.values():
        artifacts = capsule.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"capsule {capsule.get('id')} artifacts are malformed")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError(f"capsule {capsule.get('id')} artifact is malformed")
            paths.append(_safe_repo_file(artifact.get("path"), root))
    if len(paths) != len(set(paths)):
        raise ValueError("capsule artifact paths are duplicated")
    return tuple(paths)


def _artifact_report(expectation: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _safe_repo_file(expectation.get("path"), root)
    payload, raw = _load_object(path, f"capsule artifact {expectation.get('path')}")
    problems: list[str] = []
    if payload.get("schema") != expectation.get("schema"):
        problems.append("artifact schema drifted")
    fields = expectation.get("fields")
    if not isinstance(fields, dict):
        problems.append("artifact fields contract is malformed")
    else:
        for dotted, expected in fields.items():
            if _dotted(payload, str(dotted)) != expected:
                problems.append(f"{dotted} did not match its declared value")
    seal_field = expectation.get("seal_field")
    if seal_field is not None and (not isinstance(seal_field, str) or not _valid_seal(payload, seal_field)):
        problems.append("artifact self-seal mismatch")
    return {
        "path": str(expectation.get("path")),
        "sha256": _sha256_bytes(raw),
        "schema": payload.get("schema"),
        "problems": problems,
        "all_ok": not problems,
    }


def _strict_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: object, *, minimum: float = 0.0, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    number = float(value)
    return math.isfinite(number) and number >= minimum and (maximum is None or number <= maximum)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _runtime_summary_valid(runtime: object, capsule_id: str, attempts: int) -> bool:
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS:
        return False
    integer_fields = (
        "sample_count",
        "peak_process_tree_rss_bytes",
        "reservation_count",
        "resource_stop_count",
        "retry_count",
        "event_count",
        "events_dropped",
    )
    if any(not _strict_nonnegative_int(runtime.get(name)) for name in integer_fields):
        return False
    sample_count = runtime["sample_count"]
    reservation_count = runtime["reservation_count"]
    retry_count = runtime["retry_count"]
    resource_stop_count = runtime["resource_stop_count"]
    event_count = runtime["event_count"]
    events_dropped = runtime["events_dropped"]
    if reservation_count < 1 or retry_count != attempts - 1 or resource_stop_count > retry_count:
        return False
    numeric_fields = (
        "minimum_memory_available_gb",
        "minimum_memory_available_percent",
        "minimum_memory_pressure_free_percent",
        "maximum_swap_used_gb",
        "minimum_disk_free_gb",
    )
    for name in numeric_fields:
        value = runtime.get(name)
        maximum = 100.0 if name.endswith("percent") else None
        if value is not None and not _finite_number(value, maximum=maximum):
            return False
    if sample_count == 0:
        if runtime["peak_process_tree_rss_bytes"] != 0 or runtime.get("last_sample") is not None:
            return False
    elif not isinstance(runtime.get("last_sample"), dict):
        return False
    for name in ("thermal_statuses", "power_sources"):
        values = runtime.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            return False
    reservation = runtime.get("last_reservation")
    if (
        not isinstance(reservation, dict)
        or reservation.get("capsule_id") != capsule_id
        or _timestamp(reservation.get("reserved_at")) is None
        or not isinstance(reservation.get("receipt"), dict)
        or reservation["receipt"].get("reserved") is not True
        or reservation["receipt"].get("capsule_id") != capsule_id
    ):
        return False
    if runtime.get("safety_state") not in {"awaiting-child", "owned-safe"}:
        return False
    events = runtime.get("events")
    if (
        not isinstance(events, list)
        or event_count < 4
        or not events
        or len(events) > MAX_RUNTIME_EVENTS
        or len(events) + events_dropped != event_count
    ):
        return False
    expected_sequences = list(range(events_dropped + 1, event_count + 1))
    observed_sequences: list[int] = []
    for event in events:
        if (
            not isinstance(event, dict)
            or not _strict_nonnegative_int(event.get("sequence"))
            or event["sequence"] == 0
            or _timestamp(event.get("at")) is None
            or not isinstance(event.get("event"), str)
            or event["event"] not in TERMINAL_RUNTIME_EVENT_NAMES
        ):
            return False
        observed_sequences.append(event["sequence"])
    final_event = events[-1]
    return bool(
        observed_sequences == expected_sequences
        and final_event.get("event") == "capsule-complete"
        and type(final_event.get("attempt")) is int
        and final_event["attempt"] == attempts
        and type(final_event.get("returncode")) is int
        and final_event["returncode"] == 0
    )


def _terminal_capsule_row_valid(row: dict[str, Any], capsule_id: str) -> bool:
    attempts = row.get("attempts")
    started = _timestamp(row.get("started_at"))
    finished = _timestamp(row.get("finished_at"))
    return bool(
        set(row) == CAPSULE_STATE_KEYS
        and type(attempts) is int
        and 1 <= attempts <= MAX_CAPSULE_ATTEMPTS
        and type(row.get("child_pid")) is int
        and row["child_pid"] > 0
        and _finite_number(row.get("child_create_time"), minimum=1e-9)
        and started is not None
        and finished is not None
        and finished >= started
        and type(row.get("returncode")) is int
        and row["returncode"] == 0
        and row.get("last_problem") is None
        and _runtime_summary_valid(row.get("runtime"), capsule_id, attempts)
    )


def _capsule_inventory_checks(
    state: dict[str, Any],
    expected: dict[str, tuple[dict[str, Any], str]],
    root: Path,
) -> tuple[bool, bool, dict[str, str]]:
    state_rows = state.get("capsules")
    if not isinstance(state_rows, dict) or set(state_rows) != set(expected):
        return False, False, {}
    metadata_ok = True
    artifacts_ok = True
    capsule_hashes: dict[str, str] = {}
    for capsule_id, (capsule, source) in sorted(expected.items()):
        row = state_rows.get(capsule_id)
        if not isinstance(row, dict):
            return False, False, {}
        if not _terminal_capsule_row_valid(row, capsule_id):
            metadata_ok = False
        capsule_hashes[capsule_id] = str(capsule.get("capsule_sha256"))
        expected_metadata = {
            "id": capsule_id,
            "kind": capsule.get("kind"),
            "priority": capsule.get("priority"),
            "depends_on": capsule.get("depends_on"),
            "capsule_sha256": capsule.get("capsule_sha256"),
            "source": source,
        }
        metadata_ok = metadata_ok and all(row.get(key) == value for key, value in expected_metadata.items())
        metadata_ok = metadata_ok and row.get("status") == "complete"
        declared = capsule.get("artifacts")
        if not isinstance(declared, list) or not declared:
            artifacts_ok = False
            continue
        try:
            reports = [_artifact_report(item, root) for item in declared if isinstance(item, dict)]
        except (OSError, ValueError):
            artifacts_ok = False
            continue
        if len(reports) != len(declared) or row.get("artifacts") != reports:
            artifacts_ok = False
        if any(report["all_ok"] is not True for report in reports):
            artifacts_ok = False
    return metadata_ok, artifacts_ok, capsule_hashes


def _proofs(paths: ReleasePaths) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    path_map = {
        "corpus": paths.corpus,
        "corpus_verification": paths.corpus_verification,
        "empirical_report": paths.empirical_report,
        "evidence_synthesis": paths.evidence_synthesis,
        "evidence_synthesis_verification": paths.evidence_synthesis_verification,
    }
    payloads: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for name, path in path_map.items():
        payloads[name], raw[name] = _load_object(path, name.replace("_", " "))
    return payloads, raw


def _proof_contracts_valid(proofs: dict[str, dict[str, Any]]) -> bool:
    return all(
        proofs[name].get("schema") == schema and _valid_seal(proofs[name], seal)
        for name, (schema, seal) in PROOF_CONTRACTS.items()
    )


def _verification_complete(
    payload: dict[str, Any],
    *,
    required_checks: frozenset[str],
    required_mutations: frozenset[str],
) -> bool:
    checks = payload.get("checks")
    suite = payload.get("mutation_suite")
    if (
        payload.get("verification_complete") is not True
        or payload.get("problems") != []
        or not isinstance(checks, dict)
        or not checks
        or set(checks) != required_checks
        or any(value is not True for value in checks.values())
        or not isinstance(suite, dict)
        or not isinstance(suite.get("count"), int)
        or suite.get("count") != suite.get("rejected")
        or not isinstance(suite.get("results"), dict)
        or set(suite["results"]) != required_mutations
        or len(suite["results"]) != suite.get("count")
        or any(value is not True for value in suite["results"].values())
    ):
        return False
    return suite["count"] == len(required_mutations)


def _proof_bindings_exact(
    proofs: dict[str, dict[str, Any]], raw: dict[str, bytes], paths: ReleasePaths
) -> bool:
    corpus = proofs["corpus"]
    corpus_verification = proofs["corpus_verification"]
    report = proofs["empirical_report"]
    synthesis = proofs["evidence_synthesis"]
    corpus_sha = _sha256_bytes(raw["corpus"])
    verification_sha = _sha256_bytes(raw["corpus_verification"])
    report_sha = _sha256_bytes(raw["empirical_report"])
    verifier_corpus = corpus_verification.get("corpus")
    report_corpus = report.get("corpus")
    sources = synthesis.get("sources")
    if not isinstance(verifier_corpus, dict) or not isinstance(report_corpus, dict):
        return False
    if not isinstance(sources, dict):
        return False
    report_source = report_corpus.get("source")
    report_verification = report_corpus.get("verification")
    if not isinstance(report_source, dict) or not isinstance(report_verification, dict):
        return False
    expected_static = {
        "corpus": (paths.corpus, corpus_sha, "corpus_sha256", proofs["corpus"]),
        "corpus_verification": (
            paths.corpus_verification,
            verification_sha,
            "verification_sha256",
            proofs["corpus_verification"],
        ),
        "empirical_report": (
            paths.empirical_report,
            report_sha,
            "report_sha256",
            proofs["empirical_report"],
        ),
    }
    static_ok = True
    for name, (path, file_sha, seal_field, source_payload) in expected_static.items():
        row = sources.get(name)
        static_ok = static_ok and isinstance(row, dict)
        if isinstance(row, dict):
            static_ok = static_ok and row.get("path") == _repo_path(path, paths.repo_root)
            static_ok = static_ok and row.get("sha256") == file_sha
            static_ok = static_ok and row.get("schema") == source_payload.get("schema")
            static_ok = static_ok and row.get("seal_field") == seal_field
            static_ok = static_ok and row.get("seal_sha256") == source_payload.get(seal_field)
            static_ok = static_ok and row.get("self_seal_valid_at_read") is True
    mutable = sources.get("program_state_snapshot")
    mutable_ok = (
        isinstance(mutable, dict)
        and mutable.get("path") == _repo_path(paths.state, paths.repo_root)
        and mutable.get("schema") == STATE_SCHEMA
        and mutable.get("seal_field") == "state_sha256"
        and isinstance(mutable.get("seal_sha256"), str)
        and mutable.get("self_seal_valid_at_read") is True
        and isinstance(mutable.get("mutable_snapshot_sha256"), str)
        and mutable.get("mutable_after_read") is True
        and mutable.get("authority_scope") == "base_runtime_accounting.base_runtime_sha256"
    )
    return bool(
        verifier_corpus.get("sha256") == corpus_sha
        and verifier_corpus.get("corpus_sha256") == corpus.get("corpus_sha256")
        and report_source.get("sha256") == corpus_sha
        and report_verification.get("sha256") == verification_sha
        and static_ok
        and mutable_ok
    )


def _safe_claims(proofs: dict[str, dict[str, Any]]) -> bool:
    corpus = proofs["corpus"]
    report = proofs["empirical_report"]
    synthesis = proofs["evidence_synthesis"]
    synthesis_verification = proofs["evidence_synthesis_verification"]
    boundaries = synthesis.get("claim_boundaries")
    if not isinstance(boundaries, dict) or set(boundaries) != REQUIRED_SYNTHESIS_BOUNDARIES:
        return False
    boundaries_safe = all(
        isinstance(row, dict) and row.get("status") == "not_tested_by_g1_c0" for row in boundaries.values()
    )
    next_authority = report.get("next_authority")
    return bool(
        boundaries_safe
        and corpus.get("corpus_complete") is True
        and corpus.get("scientific_promotion") is False
        and report.get("claim_scope") == REPORT_CLAIM_SCOPE
        and report.get("scientific_promotion") is False
        and isinstance(next_authority, dict)
        and next_authority.get("ready_to_activate_or_integrate_substrate") is False
        and next_authority.get("automatic_activation_allowed") is False
        and next_authority.get("automatic_scientific_promotion_allowed") is False
        and synthesis.get("claim_scope") == SYNTHESIS_CLAIM_SCOPE
        and synthesis.get("activation_allowed") is False
        and synthesis.get("scientific_promotion") is False
        and synthesis_verification.get("claim_scope") == SYNTHESIS_VERIFICATION_CLAIM_SCOPE
        and synthesis_verification.get("activation_allowed") is False
        and synthesis_verification.get("scientific_promotion") is False
    )


def _reseal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: _canonical_sha256(core)}


def _replay_corpus_verifier(paths: ReleasePaths, recorded: dict[str, Any]) -> bool:
    regenerated = verify_corpus(
        corpus_path=paths.corpus.resolve(),
        config_path=paths.corpus_config.resolve(),
        run_root=paths.corpus_run_root.resolve(),
    )
    regenerated["recorded_at"] = recorded.get("recorded_at")
    return _reseal(regenerated, "verification_sha256") == recorded


def _replay_corpus_producer(paths: ReleasePaths, recorded: dict[str, Any]) -> bool:
    regenerated = build_corpus(paths.corpus_config.resolve(), paths.corpus_run_root.resolve())
    return regenerated == recorded


def _replay_report(paths: ReleasePaths, recorded: dict[str, Any]) -> bool:
    regenerated = build_report(paths.corpus.resolve(), paths.corpus_verification.resolve())
    regenerated["created_at"] = recorded.get("created_at")
    return _reseal(regenerated, "report_sha256") == recorded


def _replay_synthesis_verifier(paths: ReleasePaths, recorded: dict[str, Any]) -> bool:
    regenerated = verify_evidence_synthesis(
        paths.corpus,
        paths.corpus_verification,
        paths.empirical_report,
        paths.state,
        paths.evidence_synthesis,
        recorded_at=recorded.get("recorded_at"),
    )
    return regenerated == recorded


def audit_generation1_release(
    paths: ReleasePaths,
    *,
    contract: ReleaseContract = ReleaseContract(),
) -> dict[str, Any]:

    root = paths.repo_root.resolve()
    for value in _input_paths(paths):
        _assert_no_symlink_components(value, root, "release input")
        _repo_path(value, root)
    run_root = paths.corpus_run_root
    _assert_no_symlink_components(run_root, root, "corpus run root")
    if not run_root.resolve().is_relative_to(root):
        raise ValueError("corpus run root escapes the repository")
    if not run_root.resolve().is_dir():
        raise ValueError("corpus run root is not a directory")
    _, audit_module_raw = _load_bytes(paths.audit_module, "release-audit implementation")
    _, audit_wrapper_raw = _load_bytes(paths.audit_wrapper, "release-audit wrapper")
    _, supervisor_module_raw = _load_bytes(paths.supervisor_module, "Generation-1 supervisor")
    _, corpus_config_raw = _load_bytes(paths.corpus_config, "corpus configuration")
    program, program_raw = _load_object(paths.program, "program manifest")
    state, state_raw = _load_object(paths.state, "program state")
    status, status_raw = _load_object(paths.status, "current status")
    injection, injection_raw = _load_object(paths.injection, "accepted injection")
    receipt, receipt_raw = _load_object(paths.injection_receipt, "accepted injection receipt")
    authority_inputs, authority_receipt = _snapshot_authorities(program, injection, root)
    proofs, proof_raw = _proofs(paths)

    base_capsules = _capsules(program, "program")
    injected_capsules = _capsules(injection, "injection")
    expected_capsules = _expected_capsule_map(
        base_capsules,
        injected_capsules,
        contract.injection_id,
    )
    declared_artifact_paths = _declared_artifact_paths(expected_capsules, root)
    artifact_snapshot = _snapshot_paths(
        declared_artifact_paths,
        root,
        "declared capsule artifact",
    )
    run_tree_snapshot = _snapshot_tree(paths.corpus_run_root, root, "corpus run tree")
    genesis = _genesis_head(program, base_capsules)
    final_head = _injected_head(genesis, injection, injected_capsules)
    metadata_ok, artifacts_ok, capsule_hashes = _capsule_inventory_checks(
        state,
        expected_capsules,
        root,
    )

    state_capsules = state.get("capsules")
    expected_program_reference = {
        "path": str(paths.program.resolve()),
        "file_sha256": _sha256_bytes(program_raw),
        "program_sha256": contract.program_sha256,
    }
    expected_supervisor_reference = {
        "implementation_path": _repo_path(paths.supervisor_module, root),
        "implementation_sha256": _sha256_bytes(supervisor_module_raw),
    }
    state_started = _timestamp(state.get("started_at"))
    state_updated = _timestamp(state.get("updated_at"))
    state_finished = _timestamp(state.get("finished_at"))
    status_created = _timestamp(status.get("created_at"))
    terminal_timestamps_valid = bool(
        state_started is not None
        and state_updated is not None
        and state_finished is not None
        and status_created is not None
        and state_started <= state_finished <= state_updated <= status_created
    )
    mirror_fields = {
        "program": "program",
        "supervisor": "supervisor",
        "execution_enabled": "execution_enabled",
        "queue_head_sha256": "queue_head_sha256",
        "next_injection_sequence": "next_injection_sequence",
        "current_capsule": "current_capsule",
        "capsules": "capsules",
        "last_admission": "last_admission",
        "lane_reservation": "lane_reservation",
        "problems": "problems",
    }
    state_status_exact = all(status.get(left) == state.get(right) for left, right in mirror_fields.items())
    state_status_exact = state_status_exact and status.get("state") == state.get("status")
    state_status_exact = state_status_exact and status.get("accepted_injection_count") == len(
        state.get("accepted_injections", [])
    )

    accepted = state.get("accepted_injections")
    processed = state.get("processed_injection_files")
    expected_injection_path = _repo_path(paths.injection, root)
    accepted_exact = (
        isinstance(accepted, list)
        and len(accepted) == 1
        and accepted[0]
        == {
            "id": contract.injection_id,
            "sequence": contract.injection_sequence,
            "path": expected_injection_path,
            "file_sha256": contract.injection_file_sha256,
            "injection_sha256": contract.injection_sha256,
            "previous_head_sha256": contract.previous_queue_head_sha256,
            "new_head_sha256": contract.final_queue_head_sha256,
        }
    )
    processed_exact = (
        isinstance(processed, list)
        and len(processed) == 1
        and processed[0]
        == {
            "path": expected_injection_path,
            "file_sha256": contract.injection_file_sha256,
            "accepted": True,
            "injection_id": contract.injection_id,
        }
    )

    receipt_exact = bool(
        receipt.get("schema") == INJECTION_RECEIPT_SCHEMA
        and _valid_seal(receipt, "receipt_sha256")
        and receipt.get("receipt_sha256") == contract.injection_receipt_sha256
        and receipt.get("program_id") == contract.program_id
        and receipt.get("injection_id") == contract.injection_id
        and receipt.get("path") == expected_injection_path
        and receipt.get("file_sha256") == contract.injection_file_sha256
        and receipt.get("accepted") is True
        and receipt.get("problems") == []
        and receipt.get("previous_head_sha256") == contract.previous_queue_head_sha256
        and receipt.get("new_head_sha256") == contract.final_queue_head_sha256
        and receipt.get("scientific_promotion") is False
    )

    proof_contracts = _proof_contracts_valid(proofs)
    corpus_verification_complete = _verification_complete(
        proofs["corpus_verification"],
        required_checks=REQUIRED_CORPUS_CHECKS,
        required_mutations=REQUIRED_CORPUS_MUTATIONS,
    )
    synthesis_verification_complete = _verification_complete(
        proofs["evidence_synthesis_verification"],
        required_checks=REQUIRED_SYNTHESIS_VERIFICATION_CHECKS,
        required_mutations=REQUIRED_SYNTHESIS_MUTATIONS,
    )
    loaded_inputs = (
        (paths.audit_module, audit_module_raw),
        (paths.audit_wrapper, audit_wrapper_raw),
        (paths.supervisor_module, supervisor_module_raw),
        (paths.program, program_raw),
        (paths.state, state_raw),
        (paths.status, status_raw),
        (paths.injection, injection_raw),
        (paths.injection_receipt, receipt_raw),
        (paths.corpus_config, corpus_config_raw),
        (paths.corpus, proof_raw["corpus"]),
        (paths.corpus_verification, proof_raw["corpus_verification"]),
        (paths.empirical_report, proof_raw["empirical_report"]),
        (paths.evidence_synthesis, proof_raw["evidence_synthesis"]),
        (
            paths.evidence_synthesis_verification,
            proof_raw["evidence_synthesis_verification"],
        ),
    )

    checks = {
        "program_schema_and_seal": (
            program.get("schema") == PROGRAM_SCHEMA and _valid_seal(program, "program_sha256")
        ),
        "program_identity_exact": (
            program.get("program_id") == contract.program_id
            and program.get("program_sha256") == contract.program_sha256
            and state.get("program") == expected_program_reference
            and status.get("program") == expected_program_reference
        ),
        "supervisor_authority_exact": (
            isinstance(state.get("supervisor"), dict)
            and set(state["supervisor"])
            == {"pid", "create_time", "implementation_path", "implementation_sha256"}
            and state["supervisor"].get("implementation_path")
            == expected_supervisor_reference["implementation_path"]
            and state["supervisor"].get("implementation_sha256")
            == expected_supervisor_reference["implementation_sha256"]
            and type(state["supervisor"].get("pid")) is int
            and state["supervisor"]["pid"] > 0
            and _finite_number(state["supervisor"].get("create_time"), minimum=1e-9)
        ),
        "audit_authorities_stable": (
            _sha256_file(paths.audit_module) == _sha256_bytes(audit_module_raw)
            and _sha256_file(paths.audit_wrapper) == _sha256_bytes(audit_wrapper_raw)
            and _sha256_file(paths.supervisor_module) == _sha256_bytes(supervisor_module_raw)
            and _sha256_file(paths.corpus_config) == _sha256_bytes(corpus_config_raw)
        ),
        "program_and_injection_authorities_current": (
            int(authority_receipt["unique_file_count"]) > 0
            and int(authority_receipt["declared_row_count"]) >= int(authority_receipt["unique_file_count"])
        ),
        "base_capsule_count_exact": len(base_capsules) == contract.base_capsule_count,
        "injection_schema_and_seal": (
            injection.get("schema") == INJECTION_SCHEMA and _valid_seal(injection, "injection_sha256")
        ),
        "injection_identity_exact": (
            injection.get("program_id") == contract.program_id
            and injection.get("injection_id") == contract.injection_id
            and injection.get("sequence") == contract.injection_sequence
            and injection.get("action") == "append-capsules"
            and isinstance(injection.get("created_at"), str)
            and bool(injection.get("created_at"))
            and isinstance(injection.get("reason"), str)
            and bool(injection.get("reason"))
            and injection.get("injection_sha256") == contract.injection_sha256
            and _sha256_bytes(injection_raw) == contract.injection_file_sha256
            and injection.get("expected_queue_head_sha256") == contract.previous_queue_head_sha256
            and all(row.get("kind") == "exploratory" for row in injected_capsules)
            and tuple(sorted(str(row["id"]) for row in injected_capsules))
            == tuple(sorted(contract.injected_capsule_ids))
        ),
        "queue_hash_chain_exact": (
            genesis == contract.previous_queue_head_sha256
            and final_head == contract.final_queue_head_sha256
            and state.get("queue_head_sha256") == final_head
            and status.get("queue_head_sha256") == final_head
        ),
        "accepted_injection_state_exact": accepted_exact and processed_exact,
        "accepted_injection_receipt_exact": receipt_exact,
        "state_schema_and_seal": (
            state.get("schema") == STATE_SCHEMA
            and set(state) == STATE_KEYS
            and _valid_seal(state, "state_sha256")
            and _claim_flags_disabled(state)
        ),
        "status_schema_and_seal": (
            status.get("schema") == STATUS_SCHEMA
            and set(status) == STATUS_KEYS
            and _valid_seal(status, "status_sha256")
            and _claim_flags_disabled(status)
        ),
        "terminal_state_exact": (
            state.get("program_id") == contract.program_id
            and status.get("program_id") == contract.program_id
            and state.get("status") == "complete"
            and status.get("state") == "complete"
            and state.get("current_capsule") is None
            and status.get("current_capsule") is None
            and state.get("lane_reservation") is None
            and status.get("lane_reservation") is None
            and isinstance(state.get("last_admission"), dict)
            and state["last_admission"].get("allowed") is True
            and state.get("problems") == []
            and status.get("problems") == []
            and state.get("execution_enabled") is True
            and status.get("execution_enabled") is True
            and terminal_timestamps_valid
            and state.get("next_injection_sequence") == contract.next_injection_sequence
            and status.get("next_injection_sequence") == contract.next_injection_sequence
        ),
        "state_status_mirror_exact": state_status_exact,
        "capsule_inventory_exact": (
            len(expected_capsules) == contract.total_capsule_count
            and isinstance(state_capsules, dict)
            and len(state_capsules) == contract.total_capsule_count
            and _dependency_graph_valid(expected_capsules)
            and metadata_ok
        ),
        "all_capsule_artifacts_current": artifacts_ok,
        "proof_schemas_and_seals": proof_contracts,
        "proof_bindings_exact": _proof_bindings_exact(proofs, proof_raw, paths),
        "corpus_producer_replay_exact": (
            proof_contracts and _replay_corpus_producer(paths, proofs["corpus"])
        ),
        "corpus_verification_complete": corpus_verification_complete,
        "corpus_verification_replay_exact": (
            proof_contracts
            and corpus_verification_complete
            and _replay_corpus_verifier(paths, proofs["corpus_verification"])
        ),
        "empirical_report_replay_exact": (
            proof_contracts and _replay_report(paths, proofs["empirical_report"])
        ),
        "synthesis_verification_complete": synthesis_verification_complete,
        "synthesis_verification_replay_exact": (
            proof_contracts
            and synthesis_verification_complete
            and _replay_synthesis_verifier(
                paths,
                proofs["evidence_synthesis_verification"],
            )
        ),
        "claim_and_nonauthorization_boundaries_exact": _safe_claims(proofs),
        "input_files_stable_after_replay": all(_stable_file(path, raw, root) for path, raw in loaded_inputs),
        "campaign_authority_files_stable_after_replay": all(
            _stable_file(path, raw, root) for path, raw in authority_inputs
        ),
        "declared_artifacts_stable_after_replay": (
            _snapshot_paths(
                declared_artifact_paths,
                root,
                "declared capsule artifact",
            )
            == artifact_snapshot
        ),
        "corpus_run_tree_stable_after_replay": (
            _snapshot_tree(paths.corpus_run_root, root, "corpus run tree") == run_tree_snapshot
        ),
    }
    problems = sorted(name for name, passed in checks.items() if not passed)
    artifact_bundle_complete = all(
        checks[name]
        for name in (
            "all_capsule_artifacts_current",
            "proof_schemas_and_seals",
            "proof_bindings_exact",
            "corpus_producer_replay_exact",
            "corpus_verification_complete",
            "corpus_verification_replay_exact",
            "empirical_report_replay_exact",
            "synthesis_verification_complete",
            "synthesis_verification_replay_exact",
            "input_files_stable_after_replay",
            "campaign_authority_files_stable_after_replay",
            "declared_artifacts_stable_after_replay",
            "corpus_run_tree_stable_after_replay",
        )
    )
    source_paths = {
        "program": (paths.program, program, program_raw, "program_sha256"),
        "state": (paths.state, state, state_raw, "state_sha256"),
        "status": (paths.status, status, status_raw, "status_sha256"),
        "injection": (
            paths.injection,
            injection,
            injection_raw,
            "injection_sha256",
        ),
        "injection_receipt": (
            paths.injection_receipt,
            receipt,
            receipt_raw,
            "receipt_sha256",
        ),
    }
    sources = {
        name: _source_receipt(path, payload, raw, root, seal)
        for name, (path, payload, raw, seal) in source_paths.items()
    }
    sources["release_audit_implementation"] = _file_receipt(
        paths.audit_module,
        audit_module_raw,
        root,
    )
    sources["release_audit_wrapper"] = _file_receipt(
        paths.audit_wrapper,
        audit_wrapper_raw,
        root,
    )
    sources["generation1_supervisor_implementation"] = _file_receipt(
        paths.supervisor_module,
        supervisor_module_raw,
        root,
    )
    sources["corpus_config"] = _file_receipt(
        paths.corpus_config,
        corpus_config_raw,
        root,
    )
    sources["campaign_authorities"] = authority_receipt
    sources["declared_capsule_artifacts"] = artifact_snapshot
    sources["corpus_run_tree"] = run_tree_snapshot
    proof_path_map = {
        "corpus": paths.corpus,
        "corpus_verification": paths.corpus_verification,
        "empirical_report": paths.empirical_report,
        "evidence_synthesis": paths.evidence_synthesis,
        "evidence_synthesis_verification": paths.evidence_synthesis_verification,
    }
    for name, path in proof_path_map.items():
        sources[name] = _source_receipt(
            path,
            proofs[name],
            proof_raw[name],
            root,
            PROOF_CONTRACTS[name][1],
        )

    core = {
        "schema": SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "program_id": contract.program_id,
        "finished_at": state.get("finished_at"),
        "sources": sources,
        "inventory": {
            "base_capsule_count": len(base_capsules),
            "injected_capsule_count": len(injected_capsules),
            "total_capsule_count": len(expected_capsules),
            "capsule_hashes_sha256": _canonical_sha256(capsule_hashes),
        },
        "injection_chain": {
            "injection_id": contract.injection_id,
            "sequence": contract.injection_sequence,
            "previous_queue_head_sha256": genesis,
            "final_queue_head_sha256": final_head,
        },
        "checks": checks,
        "problems": problems,
        "artifact_bundle_complete": artifact_bundle_complete,
        "release_complete": not problems,
        "activation_allowed": False,
        "substrate_formation_established": False,
        "scientific_promotion": False,
    }
    return {**core, "audit_sha256": _canonical_sha256(core)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_output_path(path: Path, paths: ReleasePaths) -> Path:
    root = paths.repo_root.resolve()
    expected = root / "proof/GENERATION1_RELEASE_AUDIT.json"
    _assert_no_symlink_components(expected.parent, root, "release-audit output parent")
    if _lexical_absolute(path) != _lexical_absolute(expected):
        raise ValueError(f"release-audit output must be exactly {expected}")
    if path.is_symlink():
        raise ValueError("release-audit output must not be a symlink")
    resolved = path.resolve()
    if resolved != expected.resolve() or not resolved.is_relative_to(root):
        raise ValueError(f"release-audit output must be exactly {expected}")
    for source in _input_paths(paths):
        if resolved == source.resolve():
            raise ValueError("release-audit output aliases an input path")
        if resolved.exists() and source.exists() and os.path.samefile(resolved, source):
            raise ValueError("release-audit output hardlinks an input file")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("release-audit output exists but is not a regular file")
    return resolved


def _write_validation_failure(
    canonical_output: Path,
    paths: ReleasePaths,
    exc: Exception,
) -> Path | None:
    root = paths.repo_root.resolve()
    try:
        _assert_no_symlink_components(
            canonical_output.parent,
            root,
            "release-audit output parent",
        )
    except ValueError:
        return None
    canonical_lexical = _lexical_absolute(canonical_output)
    if not canonical_output.is_symlink() and any(
        _lexical_absolute(source) == canonical_lexical for source in _input_paths(paths)
    ):
        return None
    try:
        _atomic_json(canonical_output, _failure_receipt(exc))
    except OSError:
        return None
    return canonical_output


def _failure_receipt(exc: Exception) -> dict[str, Any]:
    core = {
        "schema": FAILURE_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "problems": ["audit_exception"],
        "artifact_bundle_complete": False,
        "release_complete": False,
        "activation_allowed": False,
        "substrate_formation_established": False,
        "scientific_promotion": False,
    }
    return {**core, "audit_sha256": _canonical_sha256(core)}


def _parser() -> argparse.ArgumentParser:
    defaults = ReleasePaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=defaults.repo_root)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--injection", type=Path)
    parser.add_argument("--injection-receipt", type=Path)
    parser.add_argument("--corpus-config", type=Path)
    parser.add_argument("--corpus-run-root", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-verification", type=Path)
    parser.add_argument("--empirical-report", type=Path)
    parser.add_argument("--evidence-synthesis", type=Path)
    parser.add_argument(
        "--evidence-synthesis-verification",
        type=Path,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repo_root.resolve()
    defaults = ReleasePaths.defaults(root)
    paths = ReleasePaths(
        repo_root=root,
        audit_module=(root / "src/mop/studies/generation1_release_audit.py"),
        audit_wrapper=(root / "scripts/audit_generation1_release.py"),
        supervisor_module=(root / "src/mop/studio/generation1_supervisor.py"),
        program=arguments.program or defaults.program,
        state=arguments.state or defaults.state,
        status=arguments.status or defaults.status,
        injection=arguments.injection or defaults.injection,
        injection_receipt=arguments.injection_receipt or defaults.injection_receipt,
        corpus_config=arguments.corpus_config or defaults.corpus_config,
        corpus_run_root=arguments.corpus_run_root or defaults.corpus_run_root,
        corpus=arguments.corpus or defaults.corpus,
        corpus_verification=arguments.corpus_verification or defaults.corpus_verification,
        empirical_report=arguments.empirical_report or defaults.empirical_report,
        evidence_synthesis=arguments.evidence_synthesis or defaults.evidence_synthesis,
        evidence_synthesis_verification=(
            arguments.evidence_synthesis_verification or defaults.evidence_synthesis_verification
        ),
    )
    canonical_output = root / "proof/GENERATION1_RELEASE_AUDIT.json"
    requested_output = arguments.out or canonical_output
    try:
        output = _validate_output_path(requested_output, paths)
    except Exception as exc:
        failure_path = _write_validation_failure(canonical_output, paths, exc)
        print(
            json.dumps(
                {
                    "release_complete": False,
                    "error": str(exc),
                    "failure_receipt": str(failure_path) if failure_path else None,
                },
                indent=2,
            )
        )
        return 2
    try:
        result = audit_generation1_release(paths)
        _atomic_json(output, result)
        print(
            json.dumps(
                {
                    "release_complete": result["release_complete"],
                    "artifact_bundle_complete": result["artifact_bundle_complete"],
                    "problems": result["problems"],
                    "audit_sha256": result["audit_sha256"],
                    "out": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["release_complete"] else 1
    except Exception as exc:
        failure = _failure_receipt(exc)
        try:
            _atomic_json(output, failure)
        except Exception as write_exc:
            print(
                json.dumps(
                    {
                        "release_complete": False,
                        "error": str(exc),
                        "failure_receipt_error": str(write_exc),
                    },
                    indent=2,
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "release_complete": False,
                    "error": str(exc),
                    "failure_receipt": str(output),
                },
                indent=2,
            )
        )
        return 2


__all__ = [
    "CLAIM_SCOPE",
    "ReleaseContract",
    "ReleasePaths",
    "SCHEMA",
    "audit_generation1_release",
    "main",
]
