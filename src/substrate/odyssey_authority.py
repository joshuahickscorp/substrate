"""Fail-closed authority, gate, and storage admission for Substrate Odyssey.

This module is deliberately a control plane, not an experiment runner.  It
turns explicit, sealed human decisions and machine receipts into one immutable
authority only after every named gate passes.  It never chooses a candidate,
control, task, answer, model, custodian, or storage reservation itself.

The final authority is shaped to be consumed by :mod:`substrate.odyssey7d`'s
launchd-owned supervisor.  Until an authority is sealed, no command in this
module can start a worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from substrate import odyssey_task_bank as task_bank
from substrate import odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
MACHINE_GATE_EVIDENCE = Path("evidence/substrate/odyssey/gates")
GIB = 1024**3
# The only fixed disk reservation for a full Odyssey is a device-wide free
# floor. The rest of the admission envelope is measured from the selected
# model, eight-cell rehearsal, and live free space. Keep the old public name
# as a receipt-schema compatibility alias.
DEVICE_FREE_FLOOR_BYTES = 25 * GIB
BASE_PROTECTED_FLOOR_BYTES = DEVICE_FREE_FLOOR_BYTES
FRONTIER_IDS = tuple("ABCDEFGH")
PHASE_NAMES = ("retrieval", "exposure", "transfer", "repair_checkpoint")
FULL_WIDTH_TRANSIENT_SLOTS = len(FRONTIER_IDS)
PHASES_PER_MICROCYCLE = 4
MICROCYCLES_PER_FRONTIER = 84
FULL_TASKS_PER_FRONTIER = PHASES_PER_MICROCYCLE * MICROCYCLES_PER_FRONTIER
CALIBRATION_WIDTHS = (1, 2, 4, 6, 8)
CALIBRATION_REPETITIONS = 3

# Closed subject shapes make machine-measured pins and single-operator custody
# decisions durable and source-bound.  A hash preserves a measurement; it does
# not invent a second human custodian or restore multi-party custody.
ARM_PIN_FIELDS = frozenset({"id", "revision", "artifact_sha256", "adapter_sha256"})
BASE_MODEL_PIN_FIELDS = frozenset(
    {"id", "revision", "weight_sha256", "tokenizer_sha256", "runtime_sha256", "quantization"}
)
PUBLIC_MODEL_CANARY_SCHEMA = "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_CANARY/v1"
PUBLIC_MODEL_CANARY_CHECKS = frozenset(
    {
        "frozen_template_bound",
        "public_only",
        "all_candidates_accounted",
        "all_configured_candidates_eligible",
        "no_hidden_seed_commitments",
        "selection_rule_applied",
        "selected_candidate_eligible",
        "shared_service_footprint_within_24_gib",
        "no_swap",
        "width_eight_admitted",
    }
)
PARITY_FIELDS = frozenset(
    {
        "task_schedule",
        "allowed_observations",
        "models",
        "tools",
        "token_budget",
        "compute_ceiling",
        "storage_ceiling",
        "wall_time",
    }
)
SCORE_DIMENSIONS = (
    "task_utility",
    "developmental_retention_or_transfer",
    "epistemic_integrity",
    "resource_adjusted_continuity",
)

# G10 is machine-verified from live access probes.  Each observation must name
# which account attempted which access against which root, and a denial that was
# never attempted is refused.  Topology is deliberately separate from the four
# required denied accesses so the builder-to-evaluator boundary is not inferred
# from a decorative topology record.
ISOLATION_OBSERVATION_SCHEMA = "SUBSTRATE_ODYSSEY_ISOLATION_OBSERVATION/v1"
ISOLATION_DENIAL_ERRNOS = frozenset({"EACCES", "EPERM"})
ISOLATION_OBSERVATION_EXPECTATIONS: dict[str, dict[str, str | None]] = {
    "candidate_evaluator_read_denied": {
        "actor_role": "candidate",
        "operation": "read",
        "target_root_field": "evaluator_only_root",
        "access_result": "denied",
    },
    "candidate_evaluator_write_denied": {
        "actor_role": "candidate",
        "operation": "write",
        "target_root_field": "evaluator_only_root",
        "access_result": "denied",
    },
    "evaluator_candidate_private_write_denied": {
        "actor_role": "evaluator",
        "operation": "write",
        "target_root_field": "candidate_visible_root",
        "access_result": "denied",
    },
    "builder_evaluator_read_denied": {
        "actor_role": "builder",
        "operation": "read",
        "target_root_field": "evaluator_only_root",
        "access_result": "denied",
    },
    "topology_observed": {
        "actor_role": "builder",
        "operation": "topology_inspection",
        "target_root_field": None,
        "access_result": "observed",
    },
}
# Shared closed envelope for the former human gates now sealed as machine
# subjects (status=pass).  Field name retained for call-site stability.
HUMAN_SUBJECT_BASE_FIELDS = frozenset(
    {
        "schema",
        "program",
        "status",
        "activation",
        "external_activation",
        "unqualified_nous",
        "frozen_build_sha256",
        "source_commit",
        "implementation_sha256",
        "input_sha256",
        "sha256",
    }
)

# G04 single-operator custody: early reveal is detectable, not impossible.
G04_CUSTODY_INDEPENDENCE = "single_operator"
G04_TRACE_LOCK_RECIPE = {
    "algorithm": "sha256_canonical_json",
    "fields": ["candidate_event_chain_sha256", "control_event_chain_sha256"],
    "combine": "digest({candidate_event_chain_sha256, control_event_chain_sha256})",
}
G04_CUSTODY_LIMITATION_STATEMENT = (
    "Early reveal of answer or scorer material is detectable via pre-launch "
    "commitment-digest mismatch or seal break, but is not prevented: a single "
    "operator holds every key, and no independent party holds reveal authority. "
    "This seal does not provide multi-party custody or double-blind reveal impossibility."
)
G04_CUSTODY_CHECKS = frozenset(
    {
        "commitments_sealed_before_launch",
        "commitment_digests_distinct",
        "reveal_gated_on_trace_lock",
        "single_operator_custody_declared",
        "no_task_seed_answer_role_overlap",
        "evaluator_only_answers",
        "trace_lock_before_reveal",
        "daily_scores_hidden",
    }
)

# These are the non-scoring workload dimensions the hardened design requires
# an eight-cell rehearsal to reproduce.  They deliberately describe work, not
# a claimed scientific result.
STORAGE_REHEARSAL_OPERATIONS = (
    "event_rate",
    "checkpoint_rate",
    "log_rate",
    "model_call_ledger_rate",
    "media_access",
    "daily_compaction",
    "restart",
    "restore",
)

# The canary suite must exercise each of these boundary failures against a
# clean baseline.  A short list of decorative booleans is not mutation
# evidence: each row must name an injected mutant and the two durable receipts
# that show the clean and rejected outcomes.
REQUIRED_MUTATION_IDS = frozenset(
    {
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
)

# A gate's kind communicates where judgement originates.  Machine-verified gates
# admit only measured or derived facts.  G04 records single-operator custody
# explicitly; it does not restore multi-party reveal impossibility.
GATE_SPECS: dict[str, dict[str, str]] = {
    "G01": {
        "name": "R2 terminal and independently verified",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_R2_TRANSITION_RECEIPT/v1",
    },
    "G02": {
        "name": "candidate and controls pinned",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_ARM_SELECTION/v1",
    },
    "G03": {
        "name": "all frontier manifests sealed",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_FRONTIER_MANIFEST_SET/v1",
    },
    "G04": {
        "name": "custodian and answer commitments sealed",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_CUSTODY_SEAL/v1",
    },
    "G05": {
        "name": "model and tool panel pinned",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_MODEL_TOOL_PANEL/v1",
    },
    # G06 measured simultaneity: whether eight concurrent cells run as fast as
    # one.  A single shared GPU cannot satisfy that, and the Odyssey schedule
    # never required it -- what is load-bearing is that every cell finishes its
    # work before its deadline.  G06-DC replaces it in the launch set; G06's
    # validator and its 1.35 limit remain intact for the historical receipt.
    "G06-DC": {
        "name": "width eight deadline capacity admitted",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_DEADLINE_CAPACITY_CALIBRATION/v1",
    },
    "G07": {
        "name": "eight-cell storage rehearsal passes",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_STORAGE_REHEARSAL/v1",
    },
    "G08": {
        "name": "memory broker canaries pass",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_MEMORY_BROKER_CANARY/v1",
    },
    "G09": {
        "name": "durability and recovery rehearsals pass",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_DURABILITY_REHEARSAL/v1",
    },
    "G10": {
        "name": "blindness and evaluator isolation pass",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_ISOLATION_ATTESTATION/v1",
    },
    "G11": {
        "name": "statistics and score weights frozen",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_STATISTICS_AUTHORITY/v1",
    },
    "G12": {
        "name": "mutation suite has zero survivors",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_MUTATION_REPORT/v1",
    },
    "G13": {
        "name": "clean clone and CI pass",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_CLEAN_CLONE_CI/v1",
    },
    "G14": {
        "name": "Telegram probe passes",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_TELEGRAM_PROBE/v1",
    },
    "G15": {
        "name": "source and protocol digests frozen",
        "kind": "machine_verified",
        "subject_schema": "SUBSTRATE_ODYSSEY_PROTOCOL_DIGESTS/v1",
    },
}

# G06-DC is the scientific successor to G06's *capacity-admission role*.  It is
# sealable via seal_machine_gate but is intentionally not a member of the frozen
# G01-G15 GATE_SPECS launch set: integrating it into operator inputs / hardened
# launch_gates requires a separate design freeze (see
# evidence/substrate/odyssey/ODYSSEY_G06_DEADLINE_CAPACITY_TRANSITION.json).
# G06 and _validate_g06 remain intact historical evidence, including the 1.35
# simultaneity limit and the measured 4.39x width-8 slowdown.
SIDE_MACHINE_GATE_SPECS: dict[str, dict[str, str]] = {
}

# Deadline-capacity limits frozen from the real Odyssey schedule (phase 1800 s,
# microcycle 7200 s).  These are not back-fit to make a measurement pass.
G06_DC_PHASE_SECONDS = 1800
G06_DC_MICROCYCLE_SECONDS = 7200
G06_DC_P95_DISPATCH_FRACTION = 0.50
G06_DC_WORST_DISPATCH_FRACTION = 0.75
G06_DC_MIN_DEADLINE_HEADROOM = 2.0
G06_DC_RESIDENT_CAP_BYTES = 85 * GIB
# Exact historical width-8 slowdown under unmodified G06; never hide this number.
G06_DC_PRESERVED_WIDTH8_SLOWDOWN = 4.392411013227944
G06_DC_REQUIRED_CHECKS = frozenset(
    {
        "frozen_build_bound",
        "source_maps_bound",
        "width_ladder_complete",
        "tool_bearing_real",
        "transport_and_semantic_valid",
        "zero_pageouts_clean_window",
        "no_sustained_swap_growth",
        "peak_rss_under_ceiling",
        "candidate_control_parity",
        "p95_active_dispatch_within_limit",
        "worst_active_dispatch_within_limit",
        "microcycle_work_complete_before_deadline",
        "minimum_deadline_headroom_met",
        "no_missed_phase_deadline",
        "no_cross_lane_model_context",
        "no_evaluator_leakage",
        "soak_no_memory_creep",
        "soak_no_thermal_collapse",
        "historical_4_39x_preserved",
        "slowdown_reported_not_gated",
        "no_synthetic_workload",
        "no_suppressed_tool_work",
        "no_cached_replay",
        "no_unfair_queues",
        "deadline_denominator_correct",
        "no_missing_frontier_cell",
        "no_dropped_failures",
        "pageout_counter_not_reset",
    }
)


def machine_gate_spec(gate_id: str) -> dict[str, str] | None:
    """Return the sealable machine-gate spec for a launch or side gate."""
    return GATE_SPECS.get(gate_id) or SIDE_MACHINE_GATE_SPECS.get(gate_id)


class Refused(RuntimeError):
    """A required authority input is absent, mutable, or insufficient."""


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


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise Refused("cannot resolve current git HEAD for clean-clone verification")
    return head


def _contains_true_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (key.casefold() in {"activation", "external_activation"} and child is not False) or _contains_true_activation(child) for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_true_activation(child) for child in value)
    return False


def _contains_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        folded = value.casefold()
        return "replace_" in folded or folded.startswith("pending") or folded in {"todo", "tbd", "unknown"}
    if isinstance(value, dict):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(child) for child in value)
    return False


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    if _contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    claimed = value.get("sha256")
    if require_digest and not isinstance(claimed, str):
        raise Refused(f"{path} is missing a self-digest")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} self-digest mismatch")
    return value


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and not overwrite:
        raise Refused(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _sealed(schema: str, payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    body = {
        "schema": schema,
        "program": PROGRAM,
        "status": status,
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_relative(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise Refused(f"{label} must be a non-empty root-relative path")
    path = (root / raw).resolve()
    if not _inside(root, path):
        raise Refused(f"{label} escapes the repository root")
    return path


def _relative(root: Path, path: Path) -> str:
    if not _inside(root, path):
        raise Refused(f"path escapes the repository root: {path}")
    return str(path.resolve().relative_to(root.resolve()))


def _assert_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise Refused(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _assert_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or _contains_placeholder(value):
        raise Refused(f"{label} must be explicit and non-placeholder")
    return value


def _assert_all_true(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or not value or not all(item is True for item in value.values()):
        raise Refused(f"{label} must be a non-empty object whose checks all pass")


def _assert_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Refused(f"{label} must be an integer >= {minimum}")
    return value


def _assert_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < minimum:
        raise Refused(f"{label} must be a number >= {minimum}")
    return float(value)


def _assert_exact_true_checks(value: Any, *, label: str, required: set[str]) -> dict[str, Any]:
    _assert_all_true(value, label=label)
    assert isinstance(value, dict)  # narrowed by _assert_all_true
    missing = sorted(required - set(value))
    if missing:
        raise Refused(f"{label} is missing required checks: {missing}")
    return value


def _assert_exact_keys(value: Any, *, label: str, required: set[str] | frozenset[str]) -> dict[str, Any]:
    """Require a closed object shape, so an attested subject has no hidden branch."""
    if not isinstance(value, dict):
        raise Refused(f"{label} must be an object")
    observed = set(value)
    if observed != set(required):
        missing = sorted(set(required) - observed)
        unexpected = sorted(observed - set(required))
        raise Refused(f"{label} has the wrong fields; missing={missing}, unexpected={unexpected}")
    return value


def _assert_exact_true_check_set(value: Any, *, label: str, required: set[str] | frozenset[str]) -> dict[str, Any]:
    checks = _assert_exact_keys(value, label=label, required=required)
    _assert_all_true(checks, label=label)
    return checks


def _assert_logical_root(value: Any, *, label: str) -> str:
    """Accept a non-sensitive logical root name, never a host-absolute path."""
    root = _assert_nonempty_text(value, label=label)
    path = Path(root)
    if path.is_absolute() or ".." in path.parts:
        raise Refused(f"{label} must be a non-escaping logical root")
    return root


def _assert_arm_pin(value: Any, *, label: str) -> dict[str, str]:
    pin = _assert_exact_keys(value, label=label, required=ARM_PIN_FIELDS)
    return {
        "id": _assert_nonempty_text(pin.get("id"), label=f"{label}.id"),
        "revision": _assert_nonempty_text(pin.get("revision"), label=f"{label}.revision"),
        "artifact_sha256": _assert_sha256(pin.get("artifact_sha256"), label=f"{label}.artifact_sha256"),
        "adapter_sha256": _assert_sha256(pin.get("adapter_sha256"), label=f"{label}.adapter_sha256"),
    }


def _assert_base_model_pin(value: Any, *, label: str) -> dict[str, str]:
    """Normalize the body pin shared by every candidate and control arm."""
    pin = _assert_exact_keys(value, label=label, required=BASE_MODEL_PIN_FIELDS)
    return {
        "id": _assert_nonempty_text(pin.get("id"), label=f"{label}.id"),
        "revision": _assert_nonempty_text(pin.get("revision"), label=f"{label}.revision"),
        "weight_sha256": _assert_sha256(pin.get("weight_sha256"), label=f"{label}.weight_sha256"),
        "tokenizer_sha256": _assert_sha256(pin.get("tokenizer_sha256"), label=f"{label}.tokenizer_sha256"),
        "runtime_sha256": _assert_sha256(pin.get("runtime_sha256"), label=f"{label}.runtime_sha256"),
        "quantization": _assert_nonempty_text(pin.get("quantization"), label=f"{label}.quantization"),
    }


def _validate_converted_machine_subject_binding(
    root: Path, gate_id: str, subject: dict[str, Any], frozen: dict[str, Any]
) -> None:
    """Bind a converted machine subject to the exact frozen protocol and sources.

    Former human gates (G02/G04/G05/G10/G11) keep the same source-map binding and
    placeholder rejection, but seal as status=pass on the machine path.
    """
    if _contains_placeholder(subject):
        raise Refused(f"{gate_id} subject contains a placeholder")
    _require_frozen_subject_binding(root, gate_id, subject, frozen)
    if subject.get("unqualified_nous") is not False:
        raise Refused(f"{gate_id} subject may not make an unqualified-nous claim")


def _validate_human_subject_binding(root: Path, gate_id: str, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Compatibility alias: converted gates now use the machine pass binding."""
    _validate_converted_machine_subject_binding(root, gate_id, subject, frozen)


def _frozen_design(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """Return the exact hardened design already bound into ``frozen``."""
    expected = frozen.get("input_sha256", {}).get("hardened_design")
    path = root / PLAN / "ODYSSEY_7D.hardened.draft.json"
    if not isinstance(expected, str) or not path.is_file() or file_digest(path) != expected:
        raise Refused("hardened design drifted from the frozen build")
    design = _read_json(path)
    if design.get("program", {}).get("id") != PROGRAM:
        raise Refused("frozen hardened design has the wrong Odyssey program")
    return design


def _require_frozen_subject_binding(root: Path, gate_id: str, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Require a measured machine subject to describe the exact frozen build.

    The source maps are intentionally complete rather than a hand-picked set.
    A calibration, rehearsal, or mutation result cannot be carried across a
    source or protocol change merely because the old receipt still says pass.
    """
    if subject.get("program") != PROGRAM or subject.get("status") != "pass":
        raise Refused(f"{gate_id} subject is not a passing Odyssey receipt")
    if subject.get("activation") is not False or subject.get("external_activation") is not False:
        raise Refused(f"{gate_id} subject must remain inactive")
    if subject.get("frozen_build_sha256") != frozen.get("sha256"):
        raise Refused(f"{gate_id} subject is not bound to this frozen build")
    if subject.get("source_commit") != _git_head(root):
        raise Refused(f"{gate_id} subject is not for the current git HEAD")
    if subject.get("implementation_sha256") != frozen.get("implementation_sha256"):
        raise Refused(f"{gate_id} subject implementation source map drifted")
    if subject.get("input_sha256") != frozen.get("input_sha256"):
        raise Refused(f"{gate_id} subject protocol input map drifted")


def _relative_non_sensitive_path(root: Path, value: Any, *, label: str) -> Path:
    """Resolve a candidate-visible path and reject evaluator/answer channels."""
    path = _resolve_relative(root, value, label=label)
    folded = _relative(root, path).casefold()
    if any(token in folded for token in ("evaluator", "answer", "hidden", "scoring", "score-key", "reveal")):
        raise Refused(f"{label} may not name evaluator-only or answer material")
    return path


def _require_file_ref(root: Path, value: Any, *, label: str, candidate_visible: bool = False) -> tuple[Path, dict[str, Any]]:
    """Validate a content-addressed evidence reference and its sealed object."""
    if not isinstance(value, dict):
        raise Refused(f"{label} must be an object with path and sha256")
    path = (
        _relative_non_sensitive_path(root, value.get("path"), label=f"{label}.path")
        if candidate_visible
        else _resolve_relative(root, value.get("path"), label=f"{label}.path")
    )
    expected = _assert_sha256(value.get("sha256"), label=f"{label}.sha256")
    if not path.is_file() or file_digest(path) != expected:
        raise Refused(f"{label} file is missing or drifted")
    document = _read_json(path, require_digest=True)
    return path, document


def _require_file_refs(
    root: Path,
    value: Any,
    *,
    label: str,
    minimum: int = 1,
    candidate_visible: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise Refused(f"{label} must contain at least {minimum} content-addressed receipt references")
    observed: list[tuple[Path, dict[str, Any]]] = []
    paths: set[str] = set()
    for index, row in enumerate(value):
        path, document = _require_file_ref(
            root,
            row,
            label=f"{label}[{index}]",
            candidate_visible=candidate_visible,
        )
        relative = _relative(root, path)
        if relative in paths:
            raise Refused(f"{label} may not repeat one receipt path")
        paths.add(relative)
        observed.append((path, document))
    return observed


def _forbidden_candidate_key(value: Any) -> str | None:
    """Find an obvious answer/evaluator channel in candidate-visible material."""
    forbidden = (
        "answer",
        "evaluator",
        "hidden",
        "scoring",
        "score_key",
        "reveal",
        "result_dependent",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and any(token in key.casefold() for token in forbidden):
                return key
            nested = _forbidden_candidate_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _forbidden_candidate_key(child)
            if nested is not None:
                return nested
    return None


def _candidate_task_ids(manifest: dict[str, Any], *, frontier: str) -> list[str]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != FULL_TASKS_PER_FRONTIER:
        raise Refused(f"G03 candidate manifest for {frontier} must contain exactly {FULL_TASKS_PER_FRONTIER} scheduled tasks")
    observed: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise Refused(f"G03 candidate task {frontier}[{index}] must be an object")
        task_id = _assert_nonempty_text(task.get("task_id"), label=f"G03 candidate task {frontier}[{index}].task_id")
        if task.get("frontier") != frontier or task.get("program") != PROGRAM or task.get("activation") is not False:
            raise Refused(f"G03 candidate task {task_id} has the wrong frontier, program, or activation state")
        schema = _assert_nonempty_text(task.get("schema"), label=f"G03 candidate task {task_id}.schema")
        if not schema.startswith("SUBSTRATE_ODYSSEY_"):
            raise Refused(f"G03 candidate task {task_id} has an unexpected schema")
        _assert_nonempty_text(task.get("family"), label=f"G03 candidate task {task_id}.family")
        _assert_nonempty_text(task.get("request"), label=f"G03 candidate task {task_id}.request")
        receipts = task.get("required_receipt")
        if not isinstance(receipts, list) or not receipts or not all(isinstance(item, str) and item.strip() for item in receipts):
            raise Refused(f"G03 candidate task {task_id} lacks required receipt structure")
        forbidden = _forbidden_candidate_key(task)
        if forbidden is not None:
            raise Refused(f"G03 candidate task {task_id} exposes forbidden candidate-visible key {forbidden!r}")
        observed.append(task_id)
    if len(set(observed)) != len(observed):
        raise Refused(f"G03 candidate manifest for {frontier} repeats task identifiers")
    return observed


def _validate_source_bundle(root: Path, manifest: dict[str, Any], *, frontier: str) -> tuple[str, str]:
    bundle = manifest.get("source_bundle")
    if not isinstance(bundle, dict):
        raise Refused(f"G03 candidate manifest for {frontier} lacks a source_bundle")
    forbidden = _forbidden_candidate_key(bundle)
    if forbidden is not None:
        raise Refused(f"G03 source bundle for {frontier} exposes forbidden candidate-visible key {forbidden!r}")
    assets = bundle.get("assets")
    if not isinstance(assets, list) or not assets:
        raise Refused(f"G03 source bundle for {frontier} must contain selected source assets")
    paths: set[str] = set()
    roles: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise Refused(f"G03 source bundle asset {frontier}[{index}] must be an object")
        path = _relative_non_sensitive_path(root, asset.get("path"), label=f"G03 source bundle asset {frontier}[{index}].path")
        expected = _assert_sha256(asset.get("sha256"), label=f"G03 source bundle asset {frontier}[{index}].sha256")
        if not path.is_file() or file_digest(path) != expected:
            raise Refused(f"G03 source bundle asset {frontier}[{index}] is missing or drifted")
        role = _assert_nonempty_text(asset.get("role"), label=f"G03 source bundle asset {frontier}[{index}].role")
        if asset.get("read_only") is not True:
            raise Refused(f"G03 source bundle asset {frontier}[{index}] must declare read_only true")
        relative = _relative(root, path)
        if relative in paths or role in roles:
            raise Refused(f"G03 source bundle for {frontier} repeats an asset path or role")
        paths.add(relative)
        roles.add(role)
    selection = _assert_sha256(bundle.get("selection_sha256"), label=f"G03 source bundle {frontier}.selection_sha256")
    if selection != digest({"frontier": frontier, "assets": assets}):
        raise Refused(f"G03 source bundle {frontier} selection digest does not bind its assets")
    return selection, digest(bundle)


def source_digest_for_frozen(frozen: dict[str, Any]) -> str:
    """Digest the complete implementation source map without a self-reference."""
    implementation = frozen.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise Refused("frozen build lacks an implementation source map")
    return digest({"implementation_sha256": implementation})


def protocol_digest_for_frozen(frozen: dict[str, Any]) -> str:
    """Digest every frozen protocol component used by a future authority."""
    required = ("schema", "program", "input_sha256", "implementation_sha256", "r2_requirements", "transition")
    if any(field not in frozen for field in required):
        raise Refused("frozen build lacks a protocol component")
    return digest({field: frozen[field] for field in required})


def _input_template() -> dict[str, Any]:
    return {
        "schema": "SUBSTRATE_ODYSSEY_OPERATOR_INPUTS/v1",
        "program": PROGRAM,
        "input_status": "template_unsealed",
        "run_id": "REPLACE_WITH_UNIQUE_ODYSSEY_RUN_ID",
        "operator_approval": {
            "actor": "REPLACE_WITH_OPERATOR_ID",
            "attested_at": "REPLACE_WITH_ISO8601_TIMESTAMP",
            "scope": "I approve sealing only if the referenced evidence remains exact and all gates pass.",
        },
        "frozen_build_sha256": "REPLACE_WITH_ODYSSEY_FROZEN_BUILD_SHA256",
        "gate_evidence": {
            gate_id: {
                "path": f"REPLACE_WITH_{gate_id}_GATE_EVIDENCE_PATH",
                "file_sha256": f"REPLACE_WITH_{gate_id}_GATE_EVIDENCE_FILE_SHA256",
            }
            for gate_id in GATE_SPECS
        },
        "storage_admission": {
            "p95_private_growth_bytes": "REPLACE_WITH_MEASURED_INTEGER",
            "largest_transient_bytes": "REPLACE_WITH_MEASURED_INTEGER",
            "terminal_allowance_bytes": "REPLACE_WITH_MEASURED_INTEGER",
            "explicit_model_reserve_bytes": "REPLACE_WITH_CURRENT_MODEL_RESERVATION_INTEGER",
            "private_write_cap_bytes": "REPLACE_WITH_G07_MEASURED_DYNAMIC_CAP_INTEGER",
        },
        "worker": {
            "argv": ["REPLACE_WITH_ABSOLUTE_WORKER_EXECUTABLE"],
            "source_files": [
                {
                    "path": "src/substrate/odyssey_worker.py",
                    "sha256": "REPLACE_WITH_WORKER_SOURCE_SHA256",
                }
            ],
        },
        "activation": False,
        "external_activation": False,
    }


def _gate_template(gate_id: str) -> dict[str, Any]:
    if gate_id not in GATE_SPECS:
        raise Refused(f"unknown Odyssey gate: {gate_id}")
    spec = GATE_SPECS[gate_id]
    human = spec["kind"] == "human_attested"
    return {
        "schema": "SUBSTRATE_ODYSSEY_GATE_EVIDENCE/v1",
        "program": PROGRAM,
        "gate_id": gate_id,
        "gate_name": spec["name"],
        "evidence_kind": spec["kind"],
        "status": "template_unsealed",
        "frozen_build_sha256": "REPLACE_WITH_ODYSSEY_FROZEN_BUILD_SHA256",
        "subject": {
            "path": "REPLACE_WITH_ROOT_RELATIVE_SEALED_SUBJECT_PATH",
            "file_sha256": "REPLACE_WITH_SUBJECT_FILE_SHA256",
            "schema": spec["subject_schema"],
        },
        "checks": {"REPLACE_WITH_GATE_SPECIFIC_CHECK": False},
        "human_attestation": (
            {
                "actor": "REPLACE_WITH_CUSTODIAN_OR_OPERATOR_ID",
                "attested_at": "REPLACE_WITH_ISO8601_TIMESTAMP",
                "statement": "REPLACE_WITH_SCOPE_BOUND_ATTESTATION",
            }
            if human
            else None
        ),
        "activation": False,
        "external_activation": False,
    }


def _human_subject_template_base(root: Path, frozen: dict[str, Any], schema: str) -> dict[str, Any]:
    """Return a deliberately unsealable subject skeleton bound to this build."""
    return {
        "schema": schema,
        "program": PROGRAM,
        "status": "template_unsealed",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": _git_head(root),
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "sha256": "REPLACE_WITH_CANONICAL_SELF_DIGEST_AFTER_A_REAL_REVIEW",
    }


def _human_gate_wrapper_template(gate_id: str, frozen: dict[str, Any]) -> dict[str, Any]:
    spec = GATE_SPECS[gate_id]
    return {
        "schema": "SUBSTRATE_ODYSSEY_GATE_EVIDENCE/v1",
        "program": PROGRAM,
        "gate_id": gate_id,
        "gate_name": spec["name"],
        "evidence_kind": "human_attested",
        "status": "template_unsealed",
        "frozen_build_sha256": frozen["sha256"],
        "subject": {
            "path": "REPLACE_WITH_ROOT_RELATIVE_SEALED_SUBJECT_PATH",
            "file_sha256": "REPLACE_WITH_SUBJECT_FILE_SHA256",
            "schema": spec["subject_schema"],
        },
        "checks": {
            "subject_schema_valid": False,
            "frozen_build_bound": False,
            "human_review_complete": False,
        },
        "human_attestation": {
            "actor": "REPLACE_WITH_REAL_CUSTODIAN_OR_OPERATOR_ID",
            "attested_at": "REPLACE_WITH_REAL_ISO8601_TIMESTAMP",
            "statement": "REPLACE_WITH_A_FACTUAL_SCOPE_BOUND_ATTESTATION",
        },
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "sha256": "REPLACE_WITH_CANONICAL_SELF_DIGEST_AFTER_A_REAL_REVIEW",
    }


def human_evidence_pack(root: Path) -> dict[str, Any]:
    """Render a read-only declaration that G02/G04/G05/G10/G11 are machine-generated.

    Multi-human custody fields are deliberately absent.  Subjects are produced by
    ``python -m substrate.odyssey_machine_subjects``; this pack never seals them.
    """
    frozen_document = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = _assert_sha256(frozen_document.get("sha256"), label="frozen_build_sha256")
    frozen = _validate_frozen_build(root, frozen_sha256)
    statistics = _frozen_design(root, frozen)["statistics"]
    independent_units = _frozen_design(root, frozen)["independent_units"]
    roots = {
        "builder_visible_root": "builder-visible",
        "candidate_visible_root": "candidate-visible",
        "evaluator_only_root": "evaluator-only",
        "publication_safe_root": "publication-safe",
    }
    return {
        "schema": "SUBSTRATE_ODYSSEY_HUMAN_EVIDENCE_PACK_TEMPLATE/v1",
        "program": PROGRAM,
        "status": "template_unsealed",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "never_a_gate_receipt": True,
        "never_an_attestation": True,
        "custody_independence": G04_CUSTODY_INDEPENDENCE,
        "custody_limitations": [G04_CUSTODY_LIMITATION_STATEMENT],
        "instructions": [
            "G02, G04, G05, G10, and G11 are machine_verified. Do not fill multi-human custody fields.",
            "Generate subjects with: python -m substrate.odyssey_machine_subjects <g02|g04|g05|g10|g11> --root . --out <path>",
            "Seal with: python -m substrate.odyssey_authority seal-machine-gate --gate <ID> --subject <path> --out <path>",
            "G04 records single_operator custody: early reveal is detectable but not prevented.",
            "Do not use this template as a pass receipt or an authority input.",
        ],
        "gate_wrappers": {},
        "machine_subject_generators": {
            gate_id: {
                "gate_id": gate_id,
                "kind": GATE_SPECS[gate_id]["kind"],
                "subject_schema": GATE_SPECS[gate_id]["subject_schema"],
                "generator": f"python -m substrate.odyssey_machine_subjects {gate_id.lower()} --root <root> --out <path>",
            }
            for gate_id in ("G02", "G04", "G05", "G10", "G11")
        },
        "subjects": {
            "G02": {
                "schema": GATE_SPECS["G02"]["subject_schema"],
                "status": "generate_via_odyssey_machine_subjects",
                "frozen_build_sha256": frozen["sha256"],
                "notes": "Derive pins from the sealed public model-canary receipt and operator decision.",
            },
            "G04": {
                "schema": GATE_SPECS["G04"]["subject_schema"],
                "status": "generate_via_odyssey_machine_subjects",
                "frozen_build_sha256": frozen["sha256"],
                "custody_independence": G04_CUSTODY_INDEPENDENCE,
                "custody_limitations": [G04_CUSTODY_LIMITATION_STATEMENT],
                "roots": dict(roots),
                "notes": "Requires a sealed G03 manifest set. Commitments replace multi-human custodians.",
            },
            "G05": {
                "schema": GATE_SPECS["G05"]["subject_schema"],
                "status": "generate_via_odyssey_machine_subjects",
                "frozen_build_sha256": frozen["sha256"],
                "notes": "Pin tools by digesting real tool artifacts declared in the operator decision.",
            },
            "G10": {
                "schema": GATE_SPECS["G10"]["subject_schema"],
                "status": "generate_via_odyssey_machine_subjects",
                "frozen_build_sha256": frozen["sha256"],
                "roots": dict(roots),
                "notes": "Live sudo -u nobody denial probes with recorded EACCES/EPERM; refuse if sudo is unavailable.",
            },
            "G11": {
                "schema": GATE_SPECS["G11"]["subject_schema"],
                "status": "generate_via_odyssey_machine_subjects",
                "frozen_build_sha256": frozen["sha256"],
                "score_weights": dict(statistics["score_weights"]),
                "primary_unit": statistics["primary_unit"],
                "independent_unit_count": independent_units["count"],
                "notes": "Statistics and weights must match the frozen hardened design exactly.",
            },
        },
        "supporting_observation_templates": {
            "G10": {
                kind: {
                    "schema": ISOLATION_OBSERVATION_SCHEMA,
                    "observation_kind": kind,
                    "access_result": ISOLATION_OBSERVATION_EXPECTATIONS[kind]["access_result"],
                    "attempted_required": True,
                    "errno_name_required_for_denial": sorted(ISOLATION_DENIAL_ERRNOS),
                }
                for kind in ISOLATION_OBSERVATION_EXPECTATIONS
            },
        },
    }


def _validate_frozen_build(root: Path, expected_sha256: str) -> dict[str, Any]:
    path = root / PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen = _read_json(path, require_digest=True)
    if frozen.get("schema") != "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1":
        raise Refused("unexpected Odyssey frozen-build schema")
    if frozen.get("sha256") != expected_sha256:
        raise Refused("operator inputs are not bound to the current frozen build")
    if frozen.get("activation") is not False or frozen.get("scientific_status") != "frozen_waiting_for_verified_r2":
        raise Refused("frozen build is not an inactive R2-bound build")
    # The frozen build has to agree with the current static sources before an
    # authority can rely on it.  This catches local drift, including a changed
    # renderer or task-bank generator after the frozen digest was produced.
    for name, expected in frozen.get("input_sha256", {}).items():
        source = odyssey_transition.build_inputs(root).get(name)
        if source is None or not source.is_file() or file_digest(source) != expected:
            raise Refused(f"frozen input drift: {name}")
    implementation_paths = {
        **odyssey_transition.implementation_inputs(root),
        "odyssey_worker": root / "src/substrate/odyssey_worker.py",
        "odyssey_authority": root / "src/substrate/odyssey_authority.py",
    }
    for name, expected in frozen.get("implementation_sha256", {}).items():
        source = implementation_paths.get(name)
        if source is None or not source.is_file() or file_digest(source) != expected:
            raise Refused(f"frozen implementation drift: {name}")
    # Every source the frozen transition declares is required.  This is more
    # robust than a hand-maintained subset: a new execution, materialization,
    # mutation, notification, or verifier source cannot be silently omitted
    # from a later frozen implementation map.
    required_sources = set(implementation_paths)
    if not required_sources.issubset(frozen.get("implementation_sha256", {})):
        missing = sorted(required_sources - set(frozen.get("implementation_sha256", {})))
        raise Refused(f"frozen build must bind every declared Odyssey implementation source: {missing}")
    return frozen


def validate_current_frozen_build(root: Path) -> dict[str, Any]:
    """Read-only public validation of the current Odyssey frozen-build map.

    Monitors can use this without reaching into private helpers.  It neither
    freezes sources nor writes a receipt, and it rejects any stale input or
    implementation map exactly as authority sealing would.
    """
    root = root.expanduser().resolve()
    document = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = _assert_sha256(document.get("sha256"), label="frozen_build_sha256")
    return _validate_frozen_build(root, frozen_sha256)


def _gate_subject(root: Path, gate: dict[str, Any], gate_id: str, frozen_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = GATE_SPECS[gate_id]
    if gate.get("schema") != "SUBSTRATE_ODYSSEY_GATE_EVIDENCE/v1":
        raise Refused(f"{gate_id} gate evidence has the wrong schema")
    if gate.get("program") != PROGRAM or gate.get("gate_id") != gate_id:
        raise Refused(f"{gate_id} gate evidence is not bound to this Odyssey program")
    if gate.get("gate_name") != spec["name"] or gate.get("evidence_kind") != spec["kind"]:
        raise Refused(f"{gate_id} gate identity or evidence kind drifted")
    if gate.get("status") != "pass" or gate.get("frozen_build_sha256") != frozen_sha256:
        raise Refused(f"{gate_id} gate is not an admitted receipt for this frozen build")
    _assert_all_true(gate.get("checks"), label=f"{gate_id} checks")
    human = spec["kind"] == "human_attested"
    attestation = gate.get("human_attestation")
    if human:
        if not isinstance(attestation, dict):
            raise Refused(f"{gate_id} requires an explicit human attestation")
        for field in ("actor", "attested_at", "statement"):
            _assert_nonempty_text(attestation.get(field), label=f"{gate_id} attestation.{field}")
    elif attestation is not None:
        raise Refused(f"{gate_id} is machine-verified and may not be relabeled as human-attested")
    subject_ref = gate.get("subject")
    if not isinstance(subject_ref, dict):
        raise Refused(f"{gate_id} has no sealed subject reference")
    subject_path = _resolve_relative(root, subject_ref.get("path"), label=f"{gate_id} subject.path")
    expected_file = _assert_sha256(subject_ref.get("file_sha256"), label=f"{gate_id} subject.file_sha256")
    if not subject_path.is_file() or file_digest(subject_path) != expected_file:
        raise Refused(f"{gate_id} subject file is missing or drifted")
    subject = _read_json(subject_path, require_digest=True)
    if subject_ref.get("schema") != spec["subject_schema"] or subject.get("schema") != spec["subject_schema"]:
        raise Refused(f"{gate_id} subject schema is not the required sealed schema")
    return subject, subject_ref


def _validate_g03(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G03", subject, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G03 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "candidate_manifests_structurally_safe",
            "frontier_set_exact",
            "scheduled_task_count_exact",
            "source_bundle_bound",
        },
    )
    if subject.get("all_pass") is not True or subject.get("manifest_count") != len(FRONTIER_IDS):
        raise Refused("G03 requires a passing complete eight-manifest subject")
    if subject.get("task_bank_generator_sha256") != frozen.get("implementation_sha256", {}).get("task_bank_generator"):
        raise Refused("G03 task-bank generator is not bound to the frozen implementation map")
    for field, frozen_key in (
        ("frontier_contract_sha256", "frontier_contract"),
        ("task_bank_sha256", "task_bank"),
        ("rendered_build_index_sha256", "rendered_build_index"),
    ):
        if subject.get(field) != frozen.get("input_sha256", {}).get(frozen_key):
            raise Refused(f"G03 {field} is not bound to the frozen input map")
    _assert_sha256(subject.get("source_selection_sha256"), label="G03 source_selection_sha256")
    manifests = subject.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != len(FRONTIER_IDS):
        raise Refused("G03 requires an ordered list of eight frontier manifests")
    if [row.get("id") if isinstance(row, dict) else None for row in manifests] != list(FRONTIER_IDS):
        raise Refused("G03 frontier manifests must be ordered exactly A-H")
    for row in manifests:
        assert isinstance(row, dict)  # narrowed by the ordered-list test above
        frontier = row["id"]
        path = _relative_non_sensitive_path(root, row.get("path"), label=f"G03 {frontier}.path")
        expected = _assert_sha256(row.get("file_sha256"), label=f"G03 {frontier}.file_sha256")
        if not path.is_file() or file_digest(path) != expected:
            raise Refused(f"G03 candidate manifest file is missing or drifted for {frontier}")
        if row.get("schema") != "SUBSTRATE_ODYSSEY_CANDIDATE_TASK_MANIFEST/v1":
            raise Refused(f"G03 candidate manifest schema is wrong for {frontier}")
        manifest = _read_json(path, require_digest=True)
        if manifest.get("schema") != row["schema"] or manifest.get("activation") is not False or manifest.get("frontier") != frontier:
            raise Refused(f"G03 candidate manifest identity is invalid for {frontier}")
        if _forbidden_candidate_key(manifest) is not None or not task_bank.candidate_is_structurally_safe(manifest):
            raise Refused(f"G03 candidate manifest fails structural candidate-safety validation for {frontier}")
        seed = _assert_sha256(manifest.get("seed_commitment"), label=f"G03 {frontier}.seed_commitment")
        if row.get("frontier") != frontier or row.get("seed_commitment") != seed:
            raise Refused(f"G03 candidate manifest row does not bind frontier/seed for {frontier}")
        task_ids = _candidate_task_ids(manifest, frontier=frontier)
        if row.get("task_count") != len(task_ids):
            raise Refused(f"G03 candidate manifest row has the wrong task count for {frontier}")
        task_ids_sha256 = _assert_sha256(row.get("task_ids_sha256"), label=f"G03 {frontier}.task_ids_sha256")
        if task_ids_sha256 != digest({"task_ids": task_ids}):
            raise Refused(f"G03 candidate manifest task-id digest does not bind ordered tasks for {frontier}")
        source_selection_sha256, source_bundle_sha256 = _validate_source_bundle(root, manifest, frontier=frontier)
        if row.get("source_selection_sha256") != source_selection_sha256:
            raise Refused(f"G03 candidate manifest source selection digest drifted for {frontier}")
        if row.get("source_bundle_sha256") != source_bundle_sha256:
            raise Refused(f"G03 candidate manifest source bundle digest drifted for {frontier}")


def _layout_paths(root: Path, row: dict[str, Any], *, label: str, fields: tuple[str, ...]) -> set[str]:
    """Validate a non-evaluator writable layout and return canonical names."""
    observed: set[str] = set()
    for field in fields:
        path = _relative_non_sensitive_path(root, row.get(field), label=f"{label}.{field}")
        relative = _relative(root, path)
        if relative in observed:
            raise Refused(f"{label} reuses writable layout path {relative}")
        observed.add(relative)
    return observed


def _require_resource_parity(
    value: Any,
    *,
    label: str,
    model: str | None = None,
    token_budget: int | None = None,
    wall_time_seconds: int | None = None,
) -> None:
    """Reject a calibration cell whose control is merely declared, not matched."""
    if not isinstance(value, dict) or set(value) != {"candidate", "control"}:
        raise Refused(f"{label} must contain exact candidate/control resource declarations")
    candidate = value["candidate"]
    control = value["control"]
    if not isinstance(candidate, dict) or not isinstance(control, dict):
        raise Refused(f"{label} candidate/control resource declarations must be objects")
    required = {
        "allowed_observations",
        "models",
        "tools",
        "token_budget",
        "compute_ceiling",
        "storage_ceiling",
        "wall_time_seconds",
    }
    if set(candidate) != required or set(control) != required or candidate != control:
        raise Refused(f"{label} does not preserve exact candidate/control resource parity")
    for field in ("allowed_observations", "models", "tools"):
        entries = candidate[field]
        if (
            not isinstance(entries, list)
            or not entries
            or not all(isinstance(item, str) and item.strip() and not _contains_placeholder(item) for item in entries)
        ):
            raise Refused(f"{label}.{field} must be a non-empty explicit list")
    for field in ("token_budget", "compute_ceiling", "storage_ceiling", "wall_time_seconds"):
        _assert_int(candidate[field], label=f"{label}.{field}", minimum=1)
    if model is not None and candidate["models"] != [model]:
        raise Refused(f"{label} is not pinned to the selected G02 base model")
    if token_budget is not None and candidate["token_budget"] != token_budget:
        raise Refused(f"{label} token budget does not match the phase harness")
    if wall_time_seconds is not None and candidate["wall_time_seconds"] != wall_time_seconds:
        raise Refused(f"{label} wall-time declaration does not match the phase harness")


def _validate_g06_resource_usage(value: Any, *, label: str) -> dict[str, int | None]:
    """Validate the fixed local-Ollama telemetry shape emitted by an arm."""
    required = {
        "prompt_eval_count",
        "eval_count",
        "total_duration_ns",
        "load_duration_ns",
        "eval_duration_ns",
    }
    usage = _assert_exact_keys(value, label=label, required=required)
    normalized: dict[str, int | None] = {}
    for field in sorted(required):
        item = usage[field]
        normalized[field] = None if item is None else _assert_int(item, label=f"{label}.{field}")
    return normalized


def _validate_g06_adapter_receipt(
    root: Path,
    reference: Any,
    *,
    label: str,
    role: str,
    frontier: str,
    task: dict[str, Any],
    manifest_sha256: str,
    authority_sha256: str,
    run_id: str,
    model: str,
    adapter_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Verify one real adapter receipt and its output artifact.

    G06 is only meaningful if it observed the actual production arm.  A
    self-digested arbitrary JSON document cannot stand in for that arm, so
    bind the receipt all the way through its output artifact and the exact
    phase identity derived from the signed dispatch contract.
    """
    path, receipt = _require_file_ref(root, reference, label=label)
    required = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "frontier",
        "role",
        "cycle",
        "phase",
        "task_id",
        "candidate_manifest_sha256",
        "request_sha256",
        "elapsed_seconds",
        "adapter_sha256",
        "model",
        "output_artifacts",
        "response_sha256",
        "state_before_sha256",
        "state_after_sha256",
        "state_change",
        "resource_usage",
        "sha256",
    }
    _assert_exact_keys(receipt, label=label, required=required)
    checks = {
        "schema": receipt.get("schema") == "SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1",
        "inactive": receipt.get("activation") is False,
        "authority": receipt.get("authority_sha256") == authority_sha256,
        "run": receipt.get("run_id") == run_id,
        "frontier": receipt.get("frontier") == frontier,
        "role": receipt.get("role") == role,
        "cycle": receipt.get("cycle") == 0,
        "phase": receipt.get("phase") == "retrieval",
        "task": receipt.get("task_id") == task.get("task_id"),
        "manifest": receipt.get("candidate_manifest_sha256") == manifest_sha256,
        "adapter": receipt.get("adapter_sha256") == adapter_sha256,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise Refused(f"{label} does not bind the real G06 {role} dispatch: {failed}")
    _assert_sha256(receipt.get("request_sha256"), label=f"{label}.request_sha256")
    _assert_number(receipt.get("elapsed_seconds"), label=f"{label}.elapsed_seconds")
    _assert_sha256(receipt.get("state_before_sha256"), label=f"{label}.state_before_sha256")
    _assert_sha256(receipt.get("state_after_sha256"), label=f"{label}.state_after_sha256")
    model_pin = _assert_exact_keys(receipt.get("model"), label=f"{label}.model", required={"id", "endpoint"})
    endpoint = _assert_nonempty_text(model_pin.get("endpoint"), label=f"{label}.model.endpoint")
    if model_pin.get("id") != model or not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise Refused(f"{label} model identity or endpoint is not the sealed local arm")
    expected_mode = "flat_exact_associative_monolith" if role == "candidate" else "append_only_history_retrieval"
    state_change = receipt.get("state_change")
    if not isinstance(state_change, dict) or state_change.get("mode") != expected_mode:
        raise Refused(f"{label} state-change mode is not the sealed {role} arm")
    usage = _validate_g06_resource_usage(receipt.get("resource_usage"), label=f"{label}.resource_usage")
    artifacts = receipt.get("output_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise Refused(f"{label} must bind exactly one arm output artifact")
    output_path, output = _require_file_ref(
        root,
        artifacts[0],
        label=f"{label}.output_artifacts[0]",
        candidate_visible=True,
    )
    output_required = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "frontier",
        "role",
        "cycle",
        "phase",
        "task_id",
        "request_sha256",
        "candidate_manifest_sha256",
        "adapter_sha256",
        "model",
        "prompt_sha256",
        "response",
        "resource_usage",
        "sha256",
    }
    _assert_exact_keys(output, label=f"{label}.output", required=output_required)
    output_checks = {
        "schema": output.get("schema") == "SUBSTRATE_ODYSSEY_ARM_OUTPUT/v1",
        "inactive": output.get("activation") is False,
        "authority": output.get("authority_sha256") == authority_sha256,
        "run": output.get("run_id") == run_id,
        "frontier": output.get("frontier") == frontier,
        "role": output.get("role") == role,
        "cycle": output.get("cycle") == 0,
        "phase": output.get("phase") == "retrieval",
        "task": output.get("task_id") == task.get("task_id"),
        "request": output.get("request_sha256") == receipt.get("request_sha256"),
        "manifest": output.get("candidate_manifest_sha256") == manifest_sha256,
        "adapter": output.get("adapter_sha256") == adapter_sha256,
        "model": output.get("model") == model,
        "usage": output.get("resource_usage") == usage,
        "response": isinstance(output.get("response"), dict),
        "response_digest": receipt.get("response_sha256") == output.get("sha256"),
    }
    if not all(output_checks.values()):
        failed = sorted(name for name, passed in output_checks.items() if not passed)
        raise Refused(f"{label} output artifact is not bound to its receipt: {failed}")
    _assert_sha256(output.get("prompt_sha256"), label=f"{label}.output.prompt_sha256")
    _assert_sha256(receipt.get("response_sha256"), label=f"{label}.response_sha256")
    return path, receipt


def _validate_g06_phase_boundary(
    root: Path,
    reference: Any,
    *,
    label: str,
    authority_sha256: str,
    run_id: str,
    events: list[dict[str, str]],
) -> Path:
    """Verify the G06 parent trace, checkpoint, and state durability boundary."""
    state_path, state = _require_file_ref(root, reference, label=label)
    if state_path.name != "STATE.json":
        raise Refused(f"{label} must point to the parent STATE.json durability boundary")
    required_state = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "completed_phase_count",
        "total_phase_count",
        "completed_paired_events",
        "event_chain_sha256",
        "checkpoint_sha256",
        "checkpoint_count",
        "complete",
        "elapsed_seconds",
        "broker_hold_seconds",
        "sha256",
    }
    _assert_exact_keys(state, label=f"{label}.state", required=required_state)
    state_checks = {
        "schema": state.get("schema") == "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
        "inactive": state.get("activation") is False,
        "authority": state.get("authority_sha256") == authority_sha256,
        "run": state.get("run_id") == run_id,
        "completed": state.get("completed_phase_count") == 1,
        "total": state.get("total_phase_count") == 1,
        "paired": state.get("completed_paired_events") == len(events),
        "checkpoint_count": state.get("checkpoint_count") == 1,
        "incomplete": state.get("complete") is False,
    }
    if not all(state_checks.values()):
        failed = sorted(name for name, passed in state_checks.items() if not passed)
        raise Refused(f"{label} state does not describe the completed paired G06 phase: {failed}")
    _assert_number(state.get("elapsed_seconds"), label=f"{label}.state.elapsed_seconds")
    _assert_number(state.get("broker_hold_seconds"), label=f"{label}.state.broker_hold_seconds")
    state_chain = _assert_sha256(state.get("event_chain_sha256"), label=f"{label}.state.event_chain_sha256")
    state_checkpoint = _assert_sha256(state.get("checkpoint_sha256"), label=f"{label}.state.checkpoint_sha256")

    trace_path = state_path.parent / "EVENTS.jsonl"
    if trace_path.is_symlink() or not trace_path.is_file():
        raise Refused(f"{label} has no regular parent event trace")
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Refused(f"{label} parent event trace is unreadable: {error}") from error
    expected_events = sorted(events, key=lambda row: row["frontier"])
    if len(lines) != len(expected_events):
        raise Refused(f"{label} parent event trace does not retain every frontier dispatch")
    chain = ""
    event_required = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "frontier",
        "cycle",
        "phase",
        "task_id",
        "candidate_receipt_sha256",
        "control_receipt_sha256",
        "candidate_elapsed_seconds",
        "control_elapsed_seconds",
        "source_bundle_guard_calls",
        "previous_event_sha256",
        "event_sha256",
    }
    for index, (line, expected) in enumerate(zip(lines, expected_events, strict=True)):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise Refused(f"{label} parent event trace contains invalid JSON") from error
        _assert_exact_keys(event, label=f"{label}.trace[{index}]", required=event_required)
        event_sha256 = event.pop("event_sha256")
        checks = {
            "schema": event.get("schema") == "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
            "inactive": event.get("activation") is False,
            "authority": event.get("authority_sha256") == authority_sha256,
            "run": event.get("run_id") == run_id,
            "frontier": event.get("frontier") == expected["frontier"],
            "cycle": event.get("cycle") == 0,
            "phase": event.get("phase") == "retrieval",
            "task": event.get("task_id") == expected["task_id"],
            "candidate_receipt": event.get("candidate_receipt_sha256") == expected["candidate_receipt_sha256"],
            "control_receipt": event.get("control_receipt_sha256") == expected["control_receipt_sha256"],
            "source_guard": event.get("source_bundle_guard_calls") == 2,
            "parent": event.get("previous_event_sha256") == chain,
            "digest": isinstance(event_sha256, str) and event_sha256 == digest(event),
        }
        _assert_number(event.get("candidate_elapsed_seconds"), label=f"{label}.trace[{index}].candidate_elapsed_seconds")
        _assert_number(event.get("control_elapsed_seconds"), label=f"{label}.trace[{index}].control_elapsed_seconds")
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise Refused(f"{label} parent event trace is not bound to its arm receipts: {failed}")
        assert isinstance(event_sha256, str)
        chain = event_sha256
    if chain != state_chain:
        raise Refused(f"{label} parent state event-chain cursor does not match its trace")

    checkpoint_path = state_path.parent / "checkpoints" / "delta-001.json"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise Refused(f"{label} has no regular delta checkpoint")
    checkpoint = _read_json(checkpoint_path, require_digest=True)
    required_checkpoint = {
        "schema",
        "activation",
        "authority_sha256",
        "kind",
        "cycle",
        "completed_phase_count",
        "completed_paired_events",
        "event_chain_sha256",
        "parent_checkpoint_sha256",
        "sha256",
    }
    _assert_exact_keys(checkpoint, label=f"{label}.checkpoint", required=required_checkpoint)
    checkpoint_checks = {
        "schema": checkpoint.get("schema") == "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
        "inactive": checkpoint.get("activation") is False,
        "authority": checkpoint.get("authority_sha256") == authority_sha256,
        "kind": checkpoint.get("kind") == "delta",
        "cycle": checkpoint.get("cycle") == 0,
        "completed": checkpoint.get("completed_phase_count") == 1,
        "paired": checkpoint.get("completed_paired_events") == len(events),
        "chain": checkpoint.get("event_chain_sha256") == chain,
        "parent": checkpoint.get("parent_checkpoint_sha256") == "",
        "state": checkpoint.get("sha256") == state_checkpoint,
    }
    if not all(checkpoint_checks.values()):
        failed = sorted(name for name, passed in checkpoint_checks.items() if not passed)
        raise Refused(f"{label} checkpoint is not bound to the durable G06 phase: {failed}")
    return state_path


def _frozen_g06_phase_contract(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """Load the exact static phase-harness contract already bound by freeze."""
    expected = frozen.get("input_sha256", {}).get("resource_calibration")
    path = root / PLAN / "RESOURCE_CALIBRATION_SPEC.draft.json"
    if not isinstance(expected, str) or not path.is_file() or file_digest(path) != expected:
        raise Refused("G06 resource calibration specification drifted from the frozen build")
    calibration = _read_json(path)
    expected_fields = {
        "measurement_basis": "active_paired_dispatch_wall_with_deadline_guard",
        "full_phase_seconds": 1800,
        "strict_dispatch_budget_seconds": 150,
        "scale_factor": 12,
        "phase_boundary_guard_interval_seconds": 30,
        "paired_adapter_dispatches_per_cell": 2,
        "scheduling_mode": "initial_release_only;per_frontier_candidate_then_control;no_global_role_barrier;parent_global_dwell",
    }
    for field, value in expected_fields.items():
        if calibration.get(field) != value:
            raise Refused(f"G06 frozen resource calibration has an invalid {field}")
    if calibration["full_phase_seconds"] // calibration["scale_factor"] != calibration["strict_dispatch_budget_seconds"]:
        raise Refused("G06 frozen resource calibration phase scale is invalid")
    if calibration.get("minimum_width_eight_scheduled_seconds") != 450:
        raise Refused("G06 frozen resource calibration width-eight duration is invalid")
    if calibration.get("max_slowdown_ratio") != 1.35:
        raise Refused("G06 frozen resource calibration slowdown limit is invalid")
    requirements = calibration.get("requirements")
    for name in (
        "strict_dispatch_deadline",
        "production_paired_adapters",
        "source_bundle_pre_dispatch_revalidation",
        "parent_global_dwell",
    ):
        if not isinstance(requirements, dict) or requirements.get(name) is not True:
            raise Refused(f"G06 frozen resource calibration does not require {name}")
    return {
        **expected_fields,
        "minimum_width_eight_scheduled_seconds": 450,
    }


def _validate_g06(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G06", subject, frozen)
    if subject.get("launch_subject") is False:
        raise Refused("G06 diagnostic/non-launch subjects cannot be sealed")
    phase_contract = _frozen_g06_phase_contract(root, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G06 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "receipt_invariant",
            "no_memory_threshold_breach",
            "no_critical_pressure",
            "no_unexpected_swap_or_pageout_increase",
            "io_latency_within_sealed_limit",
            "slowdown_within_sealed_limit",
            "distinct_run_roots",
            "no_shared_writable_evaluator_or_data_root",
            "record_cpu_memory_io",
            "strict_dispatch_deadline_met",
            "production_paired_adapters_complete",
            "source_bundle_revalidation_complete",
            "parent_global_dwell_complete",
        },
    )
    if (
        subject.get("all_pass") is not True
        or subject.get("admitted_width") != 8
        or subject.get("full_program_requires_width") != 8
        or subject.get("calibration_widths") != list(CALIBRATION_WIDTHS)
        or subject.get("repetitions_per_width") != CALIBRATION_REPETITIONS
    ):
        raise Refused("G06 requires the exact passing width-eight admission schedule")
    observations = subject.get("observations")
    expected_pairs = [(width, repetition) for width in CALIBRATION_WIDTHS for repetition in range(1, CALIBRATION_REPETITIONS + 1)]
    pairs = (
        [(row.get("width"), row.get("repetition")) if isinstance(row, dict) else (None, None) for row in observations] if isinstance(observations, list) else []
    )
    if pairs != expected_pairs:
        raise Refused("G06 must retain all 1/2/4/6/8 widths with three ordered repetitions each")
    harness = _assert_exact_keys(
        subject.get("phase_harness"),
        label="G06 phase_harness",
        required={
            "schema",
            "measurement_basis",
            "full_phase_seconds",
            "strict_dispatch_budget_seconds",
            "scale_factor",
            "phase_boundary_guard_interval_seconds",
            "paired_adapter_dispatches_per_cell",
            "source_bundle_pre_dispatch_revalidation",
            "scheduling_mode",
            "worker_sha256",
            "adapter_sha256",
            "model",
            "max_output_tokens",
            "g03_manifest_bindings",
            "dispatch_contract_sha256",
            "g02_subject",
            "g03_subject",
            "minimum_width_eight_scheduled_seconds",
        },
    )
    if harness.get("schema") != "SUBSTRATE_ODYSSEY_G06_REAL_PHASE_HARNESS/v1":
        raise Refused("G06 phase harness schema is invalid")
    for field, expected in phase_contract.items():
        if harness.get(field) != expected:
            raise Refused(f"G06 phase harness does not retain frozen {field}")
    if harness.get("source_bundle_pre_dispatch_revalidation") is not True:
        raise Refused("G06 phase harness must revalidate source bundles before both arms")
    if harness.get("worker_sha256") != frozen.get("implementation_sha256", {}).get("odyssey_worker"):
        raise Refused("G06 phase harness worker digest drifted")
    if harness.get("adapter_sha256") != frozen.get("implementation_sha256", {}).get("odyssey_arms"):
        raise Refused("G06 phase harness adapter digest drifted")
    _assert_nonempty_text(harness.get("model"), label="G06 phase harness model")
    if harness.get("max_output_tokens") != 64:
        raise Refused("G06 phase harness output envelope drifted")
    manifest_bindings = harness.get("g03_manifest_bindings")
    if not isinstance(manifest_bindings, list) or [row.get("id") if isinstance(row, dict) else None for row in manifest_bindings] != list(FRONTIER_IDS):
        raise Refused("G06 phase harness lacks ordered G03 manifest bindings")
    for index, binding in enumerate(manifest_bindings):
        assert isinstance(binding, dict)
        _assert_exact_keys(binding, label=f"G06 phase_harness.g03_manifest_bindings[{index}]", required={"id", "path", "sha256"})
        _relative_non_sensitive_path(root, binding["path"], label=f"G06 phase_harness manifest {binding['id']}.path")
        _assert_sha256(binding["sha256"], label=f"G06 phase_harness manifest {binding['id']}.sha256")
    contract_body = dict(harness)
    claimed_contract = contract_body.pop("dispatch_contract_sha256")
    contract_body.pop("g02_subject")
    contract_body.pop("g03_subject")
    contract_body.pop("minimum_width_eight_scheduled_seconds")
    if _assert_sha256(claimed_contract, label="G06 phase harness dispatch_contract_sha256") != digest(contract_body):
        raise Refused("G06 phase harness dispatch contract digest is invalid")
    _g02_path, g02_subject = _require_file_ref(root, harness.get("g02_subject"), label="G06 phase harness G02 subject")
    _g03_path, g03_subject = _require_file_ref(root, harness.get("g03_subject"), label="G06 phase harness G03 subject")
    _validate_g02(root, g02_subject, frozen)
    _validate_g03(root, g03_subject, frozen)
    selected_base = _assert_base_model_pin(g02_subject.get("base_model"), label="G06 bound G02 base model")
    candidate = g02_subject.get("candidate")
    if not isinstance(candidate, dict):
        raise Refused("G06 bound G02 candidate pin is missing")
    selected_adapter = _assert_arm_pin(
        {name: candidate.get(name) for name in ARM_PIN_FIELDS},
        label="G06 bound G02 candidate",
    )
    if harness.get("model") != selected_base["id"] or harness.get("adapter_sha256") != selected_adapter["adapter_sha256"]:
        raise Refused("G06 phase harness model or adapter does not match its validated G02 selection")
    g03_rows = _g03_manifest_rows(g03_subject)
    expected_bindings = [
        {"id": frontier, "path": g03_rows[frontier]["path"], "sha256": g03_rows[frontier]["file_sha256"]}
        for frontier in FRONTIER_IDS
    ]
    if manifest_bindings != expected_bindings:
        raise Refused("G06 phase harness manifest bindings do not match validated G03 manifests")
    g03_first_tasks: dict[str, dict[str, Any]] = {}
    for frontier in FRONTIER_IDS:
        manifest_path = _resolve_relative(root, g03_rows[frontier]["path"], label=f"G06 bound G03 {frontier} manifest")
        manifest = _read_json(manifest_path, require_digest=True)
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
            raise Refused(f"G06 bound G03 {frontier} manifest lacks retrieval task zero")
        g03_first_tasks[frontier] = tasks[0]
    scheduled_width_eight = _assert_number(
        subject.get("width_eight_scheduled_seconds"),
        label="G06 width_eight_scheduled_seconds",
        minimum=float(phase_contract["minimum_width_eight_scheduled_seconds"]),
    )
    if scheduled_width_eight < phase_contract["minimum_width_eight_scheduled_seconds"]:
        raise Refused("G06 width-eight scheduled observations are too short")
    layouts: set[str] = set()
    for index, row in enumerate(observations):
        assert isinstance(row, dict)
        width = row["width"]
        repetition = row["repetition"]
        phase_authority_sha256 = digest(
            {
                "schema": "SUBSTRATE_ODYSSEY_G06_CALIBRATION_AUTHORITY/v1",
                "dispatch_contract_sha256": claimed_contract,
                "width": width,
                "repetition": repetition,
            }
        )
        phase_run_id = f"g06-{claimed_contract[:16]}-{width}x-{repetition}"
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) != width:
            raise Refused(f"G06 observation {index} does not exercise exactly its declared paired-cell width")
        cell_ids = [cell.get("id") if isinstance(cell, dict) else None for cell in cells]
        if len(set(cell_ids)) != len(cell_ids) or any(cell_id not in FRONTIER_IDS for cell_id in cell_ids):
            raise Refused(f"G06 observation {index} has invalid or repeated frontier cells")
        if width == len(FRONTIER_IDS) and cell_ids != list(FRONTIER_IDS):
            raise Refused("G06 width-eight observation must exercise frontier cells A-H in order")
        expected_boundary_events: list[dict[str, str]] = []
        required_receipt_refs: list[Any] = []
        for cell in cells:
            assert isinstance(cell, dict)
            cell_layout = _layout_paths(
                root,
                cell,
                label=f"G06 {width}x cell {cell['id']}",
                fields=(
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
                ),
            )
            overlap = layouts & cell_layout
            if overlap:
                raise Refused(f"G06 reuses a writable path across calibration cells: {sorted(overlap)}")
            layouts.update(cell_layout)
            _require_resource_parity(
                cell.get("resource_parity"),
                label=f"G06 {width}x cell {cell['id']}.resource_parity",
                model=selected_base["id"],
                token_budget=harness["max_output_tokens"],
                wall_time_seconds=harness["full_phase_seconds"],
            )
            task_binding = _assert_exact_keys(
                cell.get("task_binding"),
                label=f"G06 {width}x cell {cell['id']}.task_binding",
                required={"manifest_path", "manifest_sha256", "task_index", "task_id", "task_sha256"},
            )
            _relative_non_sensitive_path(root, task_binding["manifest_path"], label=f"G06 {width}x cell {cell['id']}.manifest_path")
            _assert_sha256(task_binding["manifest_sha256"], label=f"G06 {width}x cell {cell['id']}.manifest_sha256")
            if task_binding.get("task_index") != 0:
                raise Refused("G06 must execute the live retrieval task at index zero")
            _assert_nonempty_text(task_binding.get("task_id"), label=f"G06 {width}x cell {cell['id']}.task_id")
            _assert_sha256(task_binding.get("task_sha256"), label=f"G06 {width}x cell {cell['id']}.task_sha256")
            expected_manifest = g03_rows[cell["id"]]
            expected_task = g03_first_tasks[cell["id"]]
            if (
                task_binding.get("manifest_path") != expected_manifest["path"]
                or task_binding.get("manifest_sha256") != expected_manifest["file_sha256"]
                or task_binding.get("task_id") != expected_task.get("task_id")
                or task_binding.get("task_sha256") != digest(expected_task)
            ):
                raise Refused("G06 cell task binding does not match validated G03 retrieval task zero")
            candidate_path, _candidate_receipt = _validate_g06_adapter_receipt(
                root,
                cell.get("candidate_receipt"),
                label=f"G06 {width}x cell {cell['id']}.candidate_receipt",
                role="candidate",
                frontier=cell["id"],
                task=expected_task,
                manifest_sha256=expected_manifest["file_sha256"],
                authority_sha256=phase_authority_sha256,
                run_id=phase_run_id,
                model=selected_base["id"],
                adapter_sha256=selected_adapter["adapter_sha256"],
            )
            control_path, _control_receipt = _validate_g06_adapter_receipt(
                root,
                cell.get("control_receipt"),
                label=f"G06 {width}x cell {cell['id']}.control_receipt",
                role="control",
                frontier=cell["id"],
                task=expected_task,
                manifest_sha256=expected_manifest["file_sha256"],
                authority_sha256=phase_authority_sha256,
                run_id=phase_run_id,
                model=selected_base["id"],
                adapter_sha256=selected_adapter["adapter_sha256"],
            )
            expected_boundary_events.append(
                {
                    "frontier": cell["id"],
                    "task_id": expected_task["task_id"],
                    "candidate_receipt_sha256": file_digest(candidate_path),
                    "control_receipt_sha256": file_digest(control_path),
                }
            )
            required_receipt_refs.extend((cell.get("candidate_receipt"), cell.get("control_receipt")))
            if cell.get("model_call_count") != 2 or cell.get("source_bundle_guard_calls") != 2:
                raise Refused("G06 cell did not execute exactly one candidate and one control production dispatch")
            _assert_number(cell.get("active_work_seconds"), label=f"G06 {width}x cell {cell['id']}.active_work_seconds", minimum=0.000001)
            if cell.get("deadline_met") is not True:
                raise Refused("G06 cell did not meet the strict dispatch deadline")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise Refused(f"G06 observation {index} lacks measured metrics")
        _assert_number(metrics.get("aggregate_throughput"), label=f"G06 observation {index}.aggregate_throughput", minimum=0.000001)
        slowdown = _assert_number(metrics.get("per_cell_slowdown_ratio"), label=f"G06 observation {index}.per_cell_slowdown_ratio", minimum=0.000001)
        if slowdown > 1.35:
            raise Refused(f"G06 observation {index} exceeds the frozen active-dispatch slowdown limit")
        _assert_int(metrics.get("resident_memory_bytes"), label=f"G06 observation {index}.resident_memory_bytes", minimum=1)
        _assert_int(metrics.get("swap_pageout_delta_bytes"), label=f"G06 observation {index}.swap_pageout_delta_bytes")
        _assert_number(metrics.get("disk_latency_ms"), label=f"G06 observation {index}.disk_latency_ms", minimum=0.000001)
        _assert_number(metrics.get("checkpoint_latency_ms"), label=f"G06 observation {index}.checkpoint_latency_ms", minimum=0.000001)
        _assert_number(metrics.get("model_latency_ms"), label=f"G06 observation {index}.model_latency_ms", minimum=0.000001)
        _assert_number(metrics.get("cpu_time_seconds"), label=f"G06 observation {index}.cpu_time_seconds", minimum=0.000001)
        _assert_int(metrics.get("io_bytes"), label=f"G06 observation {index}.io_bytes", minimum=1)
        _assert_nonempty_text(metrics.get("thermal_pressure"), label=f"G06 observation {index}.thermal_pressure")
        if metrics.get("strict_dispatch_budget_seconds") != phase_contract["strict_dispatch_budget_seconds"]:
            raise Refused(f"G06 observation {index} strict dispatch budget drifted")
        if metrics.get("scheduled_phase_seconds") != phase_contract["strict_dispatch_budget_seconds"]:
            raise Refused(f"G06 observation {index} scheduled phase duration drifted")
        active_dispatch_wall = _assert_number(
            metrics.get("active_dispatch_wall_seconds"),
            label=f"G06 observation {index}.active_dispatch_wall_seconds",
            minimum=0.000001,
        )
        if active_dispatch_wall > phase_contract["strict_dispatch_budget_seconds"]:
            raise Refused(f"G06 observation {index} active paired dispatch exceeded its strict deadline")
        observation_wall = _assert_number(
            metrics.get("observation_wall_seconds"),
            label=f"G06 observation {index}.observation_wall_seconds",
            minimum=float(phase_contract["strict_dispatch_budget_seconds"]),
        )
        if metrics.get("slowdown_basis") != phase_contract["measurement_basis"]:
            raise Refused(f"G06 observation {index} slowdown measurement basis drifted")
        raw_slowdown = _assert_number(
            metrics.get("raw_active_dispatch_slowdown_ratio"),
            label=f"G06 observation {index}.raw_active_dispatch_slowdown_ratio",
            minimum=0.000001,
        )
        e2e_slowdown = _assert_number(
            metrics.get("e2e_slowdown_ratio"),
            label=f"G06 observation {index}.e2e_slowdown_ratio",
            minimum=0.000001,
        )
        _assert_number(metrics.get("width1_baseline_seconds"), label=f"G06 observation {index}.width1_baseline_seconds", minimum=0.000001)
        if raw_slowdown != slowdown or e2e_slowdown > slowdown:
            raise Refused(f"G06 observation {index} active dispatch slowdown summaries are inconsistent")
        if observation_wall < active_dispatch_wall:
            raise Refused(f"G06 observation {index} observation wall precedes active dispatch completion")
        _assert_number(metrics.get("global_dwell_seconds"), label=f"G06 observation {index}.global_dwell_seconds")
        _assert_int(metrics.get("parent_guard_samples"), label=f"G06 observation {index}.parent_guard_samples", minimum=1)
        if metrics.get("paired_adapter_dispatches") != 2 * width:
            raise Refused(f"G06 observation {index} did not execute both arms for every frontier")
        boundary_ref = metrics.get("phase_boundary_receipt")
        _validate_g06_phase_boundary(
            root,
            boundary_ref,
            label=f"G06 observation {index}.phase_boundary_receipt",
            authority_sha256=phase_authority_sha256,
            run_id=phase_run_id,
            events=expected_boundary_events,
        )
        required_receipt_refs.append(boundary_ref)
        if metrics.get("critical_pressure") is not False:
            raise Refused(f"G06 observation {index} must record non-critical memory pressure")
        _assert_exact_true_checks(
            row.get("checks"),
            label=f"G06 observation {index} checks",
            required={
                "receipt_invariant",
                "no_memory_threshold_breach",
                "no_critical_pressure",
                "no_unexpected_swap_or_pageout_increase",
                "io_latency_within_sealed_limit",
                "slowdown_within_sealed_limit",
                "distinct_run_roots",
                "no_shared_writable_evaluator_or_data_root",
                "record_cpu_memory_io",
                "strict_dispatch_deadline_met",
                "production_paired_adapters_complete",
                "source_bundle_revalidation_complete",
                "parent_global_dwell_complete",
            },
        )
        observed_refs = _require_file_refs(
            root,
            row.get("receipt_refs"),
            label=f"G06 observation {index}.receipt_refs",
            minimum=2 * width + 1,
        )
        observed_ref_keys = {(_relative(root, path), file_digest(path)) for path, _document in observed_refs}
        required_ref_keys: set[tuple[str, str]] = set()
        for ref_index, reference in enumerate(required_receipt_refs):
            if not isinstance(reference, dict):
                raise Refused(f"G06 observation {index} required receipt {ref_index} is malformed")
            path = _resolve_relative(
                root,
                reference.get("path"),
                label=f"G06 observation {index} required receipt {ref_index}.path",
            )
            sha256 = _assert_sha256(
                reference.get("sha256"),
                label=f"G06 observation {index} required receipt {ref_index}.sha256",
            )
            required_ref_keys.add((_relative(root, path), sha256))
        if not required_ref_keys.issubset(observed_ref_keys):
            raise Refused(f"G06 observation {index} receipt_refs omit a bound arm or phase-boundary receipt")


def _validate_g06_dc(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate the G06-DC deadline-capacity subject.

    This gate does **not** apply the G06 1.35 simultaneity limit.  It requires
    that the real tool-bearing workload finish inside schedule deadlines, with
    the historical 4.39x slowdown preserved and reported.  G06 remains intact.
    """
    _require_frozen_subject_binding(root, "G06-DC", subject, frozen)
    if subject.get("schema") != GATE_SPECS["G06-DC"]["subject_schema"]:
        raise Refused("G06-DC subject schema is invalid")
    _assert_exact_true_checks(subject.get("checks"), label="G06-DC subject checks", required=set(G06_DC_REQUIRED_CHECKS))
    if subject.get("all_pass") is not True or subject.get("admitted_width") != 8:
        raise Refused("G06-DC requires a passing width-eight deadline-capacity admission")
    if (
        subject.get("calibration_widths") != list(CALIBRATION_WIDTHS)
        or subject.get("repetitions_per_width") != CALIBRATION_REPETITIONS
        or subject.get("full_program_requires_width") != 8
    ):
        raise Refused("G06-DC must retain the exact 1/2/4/6/8 x 3 width ladder")
    if subject.get("workload_class") != "tool_bearing_final":
        raise Refused("G06-DC refuses a non-tool-bearing or synthetic workload class")
    if subject.get("synthetic_workload") is not False:
        raise Refused("G06-DC refuses synthetic workload substituted for real tool-bearing work")
    if subject.get("model_or_tool_work_suppressed") is not False:
        raise Refused("G06-DC refuses suppressed model or tool work")
    if subject.get("cached_outputs_replayed_as_fresh") is not False:
        raise Refused("G06-DC refuses cached outputs replayed as fresh")
    if subject.get("pageout_counter_reset") is not False:
        raise Refused("G06-DC refuses a pageout counter reset")
    if subject.get("failures_dropped") is not False:
        raise Refused("G06-DC refuses dropped failures")
    if subject.get("cross_lane_model_context") is not False:
        raise Refused("G06-DC refuses cross-lane model context")
    if subject.get("evaluator_leakage") is not False:
        raise Refused("G06-DC refuses evaluator leakage")
    if subject.get("candidate_control_queues_equal") is not True:
        raise Refused("G06-DC refuses unfair candidate/control queues")
    if subject.get("deadline_denominator_seconds") != G06_DC_PHASE_SECONDS:
        raise Refused("G06-DC refuses wrong deadline denominator (must be full phase 1800 s)")

    limits = subject.get("deadline_limits")
    if not isinstance(limits, dict):
        raise Refused("G06-DC lacks deadline_limits")
    if (
        limits.get("phase_seconds") != G06_DC_PHASE_SECONDS
        or limits.get("microcycle_seconds") != G06_DC_MICROCYCLE_SECONDS
        or limits.get("p95_active_dispatch_fraction_of_phase") != G06_DC_P95_DISPATCH_FRACTION
        or limits.get("worst_active_dispatch_fraction_of_phase") != G06_DC_WORST_DISPATCH_FRACTION
        or limits.get("minimum_deadline_headroom") != G06_DC_MIN_DEADLINE_HEADROOM
        or limits.get("resident_cap_bytes") != G06_DC_RESIDENT_CAP_BYTES
    ):
        raise Refused("G06-DC deadline_limits drifted from the frozen schedule contract")

    preserved = _assert_number(
        subject.get("preserved_historical_width8_slowdown"),
        label="G06-DC preserved_historical_width8_slowdown",
        minimum=0.000001,
    )
    if abs(preserved - G06_DC_PRESERVED_WIDTH8_SLOWDOWN) > 1e-9:
        raise Refused("G06-DC must preserve the exact historical 4.392411013227944 width-8 slowdown")

    prior = subject.get("prior_model_dispatch")
    if not isinstance(prior, dict):
        raise Refused("G06-DC lacks prior_model_dispatch binding")
    prior_path = _resolve_relative(root, prior.get("path"), label="G06-DC prior_model_dispatch.path")
    prior_sha = _assert_sha256(prior.get("file_sha256"), label="G06-DC prior_model_dispatch.file_sha256")
    if not prior_path.is_file() or file_digest(prior_path) != prior_sha:
        raise Refused("G06-DC prior model-dispatch evidence is missing or drifted")
    if abs(float(prior.get("preserved_width8_max_slowdown") or 0.0) - G06_DC_PRESERVED_WIDTH8_SLOWDOWN) > 1e-9:
        raise Refused("G06-DC prior binding does not preserve the 4.39x result")

    observations = subject.get("observations")
    expected_pairs = [(width, repetition) for width in CALIBRATION_WIDTHS for repetition in range(1, CALIBRATION_REPETITIONS + 1)]
    pairs = (
        [(row.get("width"), row.get("repetition")) if isinstance(row, dict) else (None, None) for row in observations]
        if isinstance(observations, list)
        else []
    )
    if pairs != expected_pairs:
        raise Refused("G06-DC must retain all 1/2/4/6/8 widths with three ordered repetitions each")

    actives: list[float] = []
    for index, row in enumerate(observations):
        assert isinstance(row, dict)
        width = int(row["width"])
        cells = row.get("cell_ids")
        if not isinstance(cells, list) or len(cells) != width:
            raise Refused(f"G06-DC observation {index} does not exercise its declared width")
        if any(cell_id not in FRONTIER_IDS for cell_id in cells) or len(set(cells)) != len(cells):
            raise Refused(f"G06-DC observation {index} has invalid or repeated frontier cells")
        if width == len(FRONTIER_IDS) and cells != list(FRONTIER_IDS):
            raise Refused("G06-DC width-eight observation must exercise frontier cells A-H in order")
        active = _assert_number(
            row.get("active_dispatch_wall_seconds"),
            label=f"G06-DC observation {index}.active_dispatch_wall_seconds",
            minimum=0.000001,
        )
        actives.append(active)
        slowdown = _assert_number(
            row.get("per_cell_slowdown_ratio"),
            label=f"G06-DC observation {index}.per_cell_slowdown_ratio",
            minimum=0.000001,
        )
        # Report slowdown; do not gate on 1.35.  Still require it is a real number.
        _ = slowdown
        deadline = row.get("deadline")
        if not isinstance(deadline, dict):
            raise Refused(f"G06-DC observation {index} lacks deadline metrics")
        utilization = _assert_number(
            deadline.get("deadline_utilization_of_phase"),
            label=f"G06-DC observation {index}.deadline_utilization_of_phase",
            minimum=0.0,
        )
        headroom = _assert_number(
            deadline.get("deadline_headroom"),
            label=f"G06-DC observation {index}.deadline_headroom",
            minimum=0.0,
        )
        if abs(utilization - active / float(G06_DC_PHASE_SECONDS)) > 1e-6:
            raise Refused(f"G06-DC observation {index} deadline utilization is inconsistent with phase denominator")
        if abs(headroom - float(G06_DC_PHASE_SECONDS) / max(active, 1e-9)) > 1e-3:
            raise Refused(f"G06-DC observation {index} deadline headroom is inconsistent")
        if active > G06_DC_P95_DISPATCH_FRACTION * G06_DC_PHASE_SECONDS:
            # Per-observation soft check; aggregate p95 enforced below.  Worst hard cap:
            pass
        if active > G06_DC_WORST_DISPATCH_FRACTION * G06_DC_PHASE_SECONDS:
            raise Refused(f"G06-DC observation {index} exceeds worst-case dispatch fraction of the phase")
        if active > G06_DC_PHASE_SECONDS:
            raise Refused(f"G06-DC observation {index} missed the phase deadline")
        if active > G06_DC_MICROCYCLE_SECONDS:
            raise Refused(f"G06-DC observation {index} missed the microcycle deadline")
        if headroom < G06_DC_MIN_DEADLINE_HEADROOM:
            raise Refused(f"G06-DC observation {index} has insufficient deadline headroom")
        rss = _assert_int(row.get("resident_memory_bytes"), label=f"G06-DC observation {index}.resident_memory_bytes", minimum=1)
        if rss > G06_DC_RESIDENT_CAP_BYTES:
            raise Refused(f"G06-DC observation {index} peak RSS exceeds the 85 GiB ceiling")
        pageout = _assert_int(
            row.get("swap_pageout_delta_bytes"),
            label=f"G06-DC observation {index}.swap_pageout_delta_bytes",
        )
        if pageout > 0:
            raise Refused(f"G06-DC observation {index} recorded new pageouts")
        if row.get("all_objects_valid") is not True:
            raise Refused(f"G06-DC observation {index} objects are not all transport/semantically valid")
        tool_proof = row.get("tool_proof")
        if not isinstance(tool_proof, list) or len(tool_proof) != 2 * width:
            raise Refused(f"G06-DC observation {index} lacks real tool proof for every arm")
        for proof_index, proof in enumerate(tool_proof):
            if not isinstance(proof, dict):
                raise Refused(f"G06-DC observation {index} tool_proof[{proof_index}] malformed")
            _assert_nonempty_text(proof.get("operation"), label=f"G06-DC observation {index} tool_proof[{proof_index}].operation")
            revision = proof.get("tool_revision")
            if not isinstance(revision, dict):
                raise Refused(f"G06-DC observation {index} tool_proof[{proof_index}] lacks tool_revision")
            _assert_nonempty_text(
                revision.get("tool_id") or revision.get("version"),
                label=f"G06-DC observation {index} tool_proof[{proof_index}].tool_revision",
            )
            _assert_sha256(
                proof.get("artifact_digest"),
                label=f"G06-DC observation {index} tool_proof[{proof_index}].artifact_digest",
            )
            if proof.get("fresh") is not True or proof.get("cached_replay") is not False:
                raise Refused(f"G06-DC observation {index} tool_proof[{proof_index}] looks like cached replay")
        checks = row.get("checks")
        if not isinstance(checks, dict) or any(checks.get(name) is not True for name in (
            "tool_bearing_real",
            "zero_pageouts",
            "peak_rss_under_ceiling",
            "candidate_control_parity",
            "no_synthetic_workload",
            "no_suppressed_tool_work",
            "no_cached_replay",
            "deadline_denominator_is_phase_1800",
            "failures_not_dropped",
            "pageout_counter_not_reset",
        )):
            raise Refused(f"G06-DC observation {index} failed a required per-observation check")

    # Aggregate p95 / worst / headroom across the ladder.
    actives_sorted = sorted(actives)
    if len(actives_sorted) >= 2:
        # Nearest-rank 95th percentile.
        rank = max(0, min(len(actives_sorted) - 1, int(math.ceil(0.95 * len(actives_sorted)) - 1)))
        p95_active = actives_sorted[rank]
    else:
        p95_active = actives_sorted[0]
    worst_active = max(actives)
    min_headroom = min(float(G06_DC_PHASE_SECONDS) / max(value, 1e-9) for value in actives)
    if p95_active > G06_DC_P95_DISPATCH_FRACTION * G06_DC_PHASE_SECONDS:
        raise Refused("G06-DC p95 active dispatch exceeds 50% of the 1800 s phase")
    if worst_active > G06_DC_WORST_DISPATCH_FRACTION * G06_DC_PHASE_SECONDS:
        raise Refused("G06-DC worst active dispatch exceeds 75% of the 1800 s phase")
    if worst_active > G06_DC_MICROCYCLE_SECONDS:
        raise Refused("G06-DC microcycle work did not complete before 7200 s")
    if min_headroom < G06_DC_MIN_DEADLINE_HEADROOM:
        raise Refused("G06-DC minimum deadline headroom is below 2x")

    by_width = subject.get("by_width")
    if not isinstance(by_width, dict) or set(by_width) != {str(width) for width in CALIBRATION_WIDTHS}:
        raise Refused("G06-DC by_width summary is incomplete")
    width8 = by_width.get("8")
    if not isinstance(width8, dict):
        raise Refused("G06-DC by_width lacks width 8")
    width8_slowdown = _assert_number(width8.get("max_slowdown"), label="G06-DC by_width.8.max_slowdown", minimum=0.000001)
    if abs(width8_slowdown - G06_DC_PRESERVED_WIDTH8_SLOWDOWN) > 1e-9:
        raise Refused("G06-DC by_width must preserve the exact 4.392411013227944 width-8 slowdown")
    if width8.get("any_pageout") is not False or width8.get("all_objects_valid") is not True:
        raise Refused("G06-DC width-8 summary reports pageouts or invalid objects")

    proof = subject.get("per_frontier_tool_proof")
    if not isinstance(proof, dict) or sorted(proof) != list(FRONTIER_IDS):
        raise Refused("G06-DC requires per-frontier tool proof for A-H")
    for frontier in FRONTIER_IDS:
        entry = proof[frontier]
        if not isinstance(entry, dict):
            raise Refused(f"G06-DC tool proof for {frontier} is malformed")
        arms = entry.get("arms")
        if not isinstance(arms, list) or len(arms) != 2:
            raise Refused(f"G06-DC tool proof for {frontier} must cover candidate and control")
        roles = sorted(arm.get("role") for arm in arms if isinstance(arm, dict))
        if roles != ["candidate", "control"]:
            raise Refused(f"G06-DC tool proof for {frontier} lacks both arms")
        for arm in arms:
            assert isinstance(arm, dict)
            _assert_nonempty_text(arm.get("operation"), label=f"G06-DC {frontier} tool operation")
            revision = arm.get("tool_revision")
            if not isinstance(revision, dict):
                raise Refused(f"G06-DC {frontier} tool_revision missing")
            _assert_sha256(arm.get("artifact_digest"), label=f"G06-DC {frontier} artifact_digest")

    soak = subject.get("soak")
    if not isinstance(soak, dict):
        raise Refused("G06-DC lacks soak evidence")
    if soak.get("memory_creep_ok") is not True or soak.get("thermal_ok") is not True:
        raise Refused("G06-DC soak failed memory-creep or thermal checks")
    if int(soak.get("pageout_window_delta_bytes") or 0) > 0:
        raise Refused("G06-DC soak window recorded pageouts")


def _storage_requirements(
    *,
    p95_total_private_growth: int,
    largest_transient: int,
    terminal_allowance: int,
    explicit_model_reserve: int,
    concurrent_transient_slots: int,
) -> tuple[int, int]:
    """Return the runtime floor and stricter launch-time requirement.

    The runtime floor protects the device while the width-eight worker has
    concurrent adapters in flight. Launch additionally needs room for the
    total projected private growth across all eight lanes; treating the latter
    as a permanent runtime floor would strand the very capacity the rehearsal
    admitted.
    """

    if concurrent_transient_slots != FULL_WIDTH_TRANSIENT_SLOTS:
        raise Refused("storage transient slots must retain full-width Odyssey concurrency")
    runtime_required = (
        BASE_PROTECTED_FLOOR_BYTES
        + explicit_model_reserve
        + concurrent_transient_slots * largest_transient
        + terminal_allowance
    )
    return runtime_required, runtime_required + p95_total_private_growth


def _validate_g07(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G07", subject, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G07 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "eight_cells_exercised",
            "event_rate_reproduced",
            "checkpoint_rate_reproduced",
            "log_rate_reproduced",
            "model_call_ledger_rate_reproduced",
            "media_access_reproduced",
            "daily_compaction_reproduced",
            "restart_reproduced",
            "restore_reproduced",
            "private_roots_distinct",
            "measurements_nonzero",
            "full_width_concurrent_transient_bound",
            "formula_bound",
        },
    )
    if subject.get("all_pass") is not True or subject.get("cells") != len(FRONTIER_IDS):
        raise Refused("G07 requires a passing eight-cell storage rehearsal")
    if subject.get("reproduced_operations") != list(STORAGE_REHEARSAL_OPERATIONS):
        raise Refused("G07 does not reproduce every hardened storage workload operation")
    design = _frozen_design(root, frozen)
    if subject.get("formula") != design.get("storage", {}).get("launch_formula"):
        raise Refused("G07 storage formula is not bound to the frozen hardened design")
    observations = subject.get("cell_observations")
    if not isinstance(observations, list) or [row.get("id") if isinstance(row, dict) else None for row in observations] != list(FRONTIER_IDS):
        raise Refused("G07 requires ordered A-H cell observations")
    layouts: set[str] = set()
    growth: list[int] = []
    transients: list[int] = []
    for row in observations:
        assert isinstance(row, dict)
        cell_layout = _layout_paths(
            root,
            row,
            label=f"G07 cell {row['id']}",
            fields=(
                "candidate_root",
                "control_root",
                "candidate_checkpoint_root",
                "control_checkpoint_root",
                "candidate_mutable_state_root",
                "control_mutable_state_root",
                "candidate_model_context_root",
                "control_model_context_root",
            ),
        )
        overlap = layouts & cell_layout
        if overlap:
            raise Refused(f"G07 reuses a private writable path: {sorted(overlap)}")
        layouts.update(cell_layout)
        for field, minimum in (
            ("event_count", 1),
            ("checkpoint_count", 1),
            ("log_bytes", 1),
            ("model_call_ledger_bytes", 1),
            ("media_access_count", 1),
            ("restart_count", 1),
            ("restore_count", 1),
        ):
            _assert_int(row.get(field), label=f"G07 cell {row['id']}.{field}", minimum=minimum)
        if row.get("daily_compaction") is not True:
            raise Refused(f"G07 cell {row['id']} did not record daily compaction")
        growth.append(_assert_int(row.get("durable_growth_bytes"), label=f"G07 cell {row['id']}.durable_growth_bytes", minimum=1))
        transients.append(_assert_int(row.get("largest_transient_bytes"), label=f"G07 cell {row['id']}.largest_transient_bytes", minimum=1))
        _require_file_refs(root, row.get("receipt_refs"), label=f"G07 cell {row['id']}.receipt_refs")
    p95 = _assert_int(subject.get("p95_private_growth_bytes"), label="G07 p95_private_growth_bytes", minimum=1)
    largest = _assert_int(subject.get("largest_transient_bytes"), label="G07 largest_transient_bytes", minimum=1)
    observed_total_growth = sum(growth)
    if subject.get("observed_total_private_growth_bytes") != observed_total_growth:
        raise Refused("G07 total private-growth observation does not match its eight-cell sum")
    if p95 < observed_total_growth or largest != max(transients):
        raise Refused("G07 total growth/transient summary does not match the eight-cell observations")
    resources = design.get("resources")
    storage = design.get("storage")
    if not isinstance(resources, dict) or not isinstance(storage, dict):
        raise Refused("G07 frozen hardened design lacks resource or storage policy")
    hard_cap_gib = storage.get("private_write_cap_gib")
    if not isinstance(hard_cap_gib, int) or hard_cap_gib != 120:
        raise Refused("G07 frozen hardened design must retain the 120 GiB private-write cap")
    concurrent_slots = _assert_int(
        subject.get("concurrent_transient_slots"), label="G07 concurrent_transient_slots", minimum=1
    )
    if (
        concurrent_slots != FULL_WIDTH_TRANSIENT_SLOTS
        or concurrent_slots != resources.get("full_program_requires_width")
    ):
        raise Refused("G07 must bound simultaneous transients at the full width-eight concurrency")
    terminal = _assert_int(subject.get("terminal_allowance_bytes"), label="G07 terminal_allowance_bytes", minimum=1)
    reserve = _assert_int(subject.get("explicit_model_reserve_bytes"), label="G07 explicit_model_reserve_bytes")
    cap = _assert_int(subject.get("private_write_cap_bytes"), label="G07 private_write_cap_bytes", minimum=1)
    before = _assert_int(subject.get("observed_free_before_bytes"), label="G07 observed_free_before_bytes", minimum=1)
    after = _assert_int(subject.get("observed_free_after_bytes"), label="G07 observed_free_after_bytes", minimum=1)
    minimum = _assert_int(subject.get("minimum_free_bytes_observed"), label="G07 minimum_free_bytes_observed", minimum=1)
    runtime_required, launch_required = _storage_requirements(
        p95_total_private_growth=p95,
        largest_transient=largest,
        terminal_allowance=terminal,
        explicit_model_reserve=reserve,
        concurrent_transient_slots=concurrent_slots,
    )
    if p95 > cap:
        raise Refused("G07 measured private growth exceeds its declared private-write cap")
    if cap > before - runtime_required:
        raise Refused("G07 private-write cap exceeds the observed live dynamic capacity")
    if cap > hard_cap_gib * GIB:
        raise Refused("G07 declared private-write cap exceeds the frozen 120 GiB maximum")
    if minimum < runtime_required:
        raise Refused("G07 rehearsal crossed the runtime device free-space floor")
    if (
        subject.get("base_protected_floor_bytes") != BASE_PROTECTED_FLOOR_BYTES
        or subject.get("runtime_required_free_bytes") != runtime_required
        or subject.get("measured_required_free_bytes") != launch_required
    ):
        raise Refused("G07 measured required free space does not match its exact formula")
    if minimum > before or minimum > after:
        raise Refused("G07 minimum free-space observation is impossible")


def _validate_g08(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G08", subject, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G08 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "threshold_table_bound",
            "all_required_pools_observed",
            "sampling_cadence_bound",
            "critical_pressure_override",
            "decision_receipts_bound",
            "no_semantic_decision",
        },
    )
    design = _frozen_design(root, frozen)
    resources = design.get("resources")
    if not isinstance(resources, dict):
        raise Refused("G08 frozen hardened design lacks resources")
    expected_thresholds = {
        "resident_cap_gib": resources.get("resident_cap_gib"),
        "normal_admission_ceiling_gib": resources.get("normal_admission_ceiling_gib"),
        "p2_checkpoint_threshold_gib": resources.get("p2_checkpoint_threshold_gib"),
        "p1_pause_threshold_gib": resources.get("p1_pause_threshold_gib"),
        "global_hold_threshold_gib": resources.get("global_hold_threshold_gib"),
    }
    if subject.get("all_pass") is not True or any(subject.get(key) != value for key, value in expected_thresholds.items()):
        raise Refused("G08 threshold table is not identical to the frozen hardened design")
    if subject.get("measurement_interval_seconds") != 30 or subject.get("accounting_uncertainty_gib") != 2:
        raise Refused("G08 must retain the hardened 30-second cadence and 2 GiB accounting allowance")
    # The admission action executed at runtime is owned by
    # ``odyssey_worker._broker_action_for_bytes``.  ``odyssey7d.py`` contains
    # a small static mirror used by preflight diagnostics, but binding a live
    # broker-canary receipt to that renderer would leave a worker-only change
    # outside the gate's explicit source identity.
    if subject.get("broker_source_sha256") != frozen.get("implementation_sha256", {}).get("odyssey_worker"):
        raise Refused("G08 broker source is not bound to the frozen implementation map")
    observations = subject.get("observations")
    expected_cases = (
        ("below_normal_admission", 74.9, False, "admit_or_resume"),
        ("normal_admission_boundary", 75.0, False, "deny_new_work"),
        ("p2_checkpoint_boundary", 80.0, False, "checkpoint_reduce_p2"),
        ("p1_pause_boundary", 82.0, False, "pause_p1_checkpoint_p2"),
        ("global_hold_boundary", 85.0, False, "safe_hold_non_p0"),
        ("critical_pressure_override", 74.0, True, "safe_hold_non_p0"),
    )
    if not isinstance(observations, list) or len(observations) != len(expected_cases):
        raise Refused("G08 must record every broker threshold and a critical-pressure override")
    required_pools = {"host", "vm", "container", "model_service", "broker"}
    for row, (case, resident, critical, decision) in zip(observations, expected_cases, strict=True):
        if not isinstance(row, dict) or (row.get("case"), row.get("resident_gib"), row.get("critical_pressure"), row.get("decision")) != (
            case,
            resident,
            critical,
            decision,
        ):
            raise Refused(f"G08 broker observation does not reproduce {case}")
        pools = row.get("memory_pools_gib")
        if not isinstance(pools, dict) or set(pools) != required_pools:
            raise Refused(f"G08 {case} must observe host, VM, container, model-service, and broker pools")
        pool_total = 0.0
        for pool, value in pools.items():
            pool_total += _assert_number(value, label=f"G08 {case}.memory_pools_gib.{pool}")
        accounted = _assert_number(row.get("accounted_total_gib"), label=f"G08 {case}.accounted_total_gib")
        if abs(accounted - (pool_total + 2.0)) > 0.000001 or abs(accounted - resident) > 0.000001:
            raise Refused(f"G08 {case} resident accounting does not include the exact 2 GiB allowance")
        lanes = row.get("lane_resident_gib")
        if not isinstance(lanes, dict) or not lanes:
            raise Refused(f"G08 {case} lacks per-lane resident observations")
        lane_total = sum(_assert_number(value, label=f"G08 {case}.lane_resident_gib.{lane}") for lane, value in lanes.items())
        if lane_total > resident:
            raise Refused(f"G08 {case} lane resident total exceeds measured resident memory")
        _require_file_refs(root, row.get("receipt_refs"), label=f"G08 {case}.receipt_refs")


def _checkpoint_document(root: Path, value: Any, *, label: str, expected_kind: str) -> tuple[Path, dict[str, Any]]:
    path, document = _require_file_ref(root, value, label=label)
    if document.get("schema") != "SUBSTRATE_ODYSSEY_CHECKPOINT/v1" or document.get("kind") != expected_kind:
        raise Refused(f"{label} is not a sealed {expected_kind} Odyssey checkpoint")
    if document.get("activation") is not False:
        raise Refused(f"{label} checkpoint is not inactive")
    _assert_sha256(document.get("event_chain_sha256"), label=f"{label}.event_chain_sha256")
    return path, document


def _validate_recovery_arm(root: Path, value: Any, *, label: str, writer_locks: set[str]) -> None:
    if not isinstance(value, dict):
        raise Refused(f"{label} must be an object")
    before = _assert_sha256(value.get("pre_interrupt_state_sha256"), label=f"{label}.pre_interrupt_state_sha256")
    restored = _assert_sha256(value.get("restored_state_sha256"), label=f"{label}.restored_state_sha256")
    if before != restored:
        raise Refused(f"{label} restore did not reproduce the pre-interrupt state")
    _full_path, full = _checkpoint_document(root, value.get("full_checkpoint"), label=f"{label}.full_checkpoint", expected_kind="full")
    deltas = value.get("delta_checkpoints")
    if not isinstance(deltas, list) or not deltas:
        raise Refused(f"{label} requires a non-empty delta checkpoint chain")
    parent = full.get("sha256")
    final_delta: dict[str, Any] | None = None
    for index, delta_ref in enumerate(deltas):
        _delta_path, delta = _checkpoint_document(root, delta_ref, label=f"{label}.delta_checkpoints[{index}]", expected_kind="delta")
        if delta.get("parent_checkpoint_sha256") != parent:
            raise Refused(f"{label} delta checkpoint chain does not link to its parent")
        parent = delta.get("sha256")
        final_delta = delta
    _trace_path, trace = _require_file_ref(root, value.get("event_trace"), label=f"{label}.event_trace")
    if trace.get("event_chain_sha256") != final_delta.get("event_chain_sha256"):
        raise Refused(f"{label} event trace does not match the restored delta chain")
    _restart_path, restart = _require_file_ref(root, value.get("restart_receipt"), label=f"{label}.restart_receipt")
    if restart.get("recovered") is not True or restart.get("interactive_shell_independent") is not True:
        raise Refused(f"{label} restart receipt does not prove autonomous recovery")
    lock = _relative(root, _relative_non_sensitive_path(root, value.get("writer_lock"), label=f"{label}.writer_lock"))
    if lock in writer_locks:
        raise Refused(f"{label} reuses a writer lock held by another recovery arm")
    writer_locks.add(lock)
    _writer_path, writer = _require_file_ref(root, value.get("single_writer_receipt"), label=f"{label}.single_writer_receipt")
    if writer.get("single_writer") is not True:
        raise Refused(f"{label} single-writer receipt does not prove exclusive ownership")
    _assert_int(value.get("recovery_downtime_seconds"), label=f"{label}.recovery_downtime_seconds")
    if value.get("resumed_at_sealed_boundary") is not True:
        raise Refused(f"{label} did not resume at a sealed schedule boundary")


def _validate_g09(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G09", subject, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G09 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "checkpoint_round_trip",
            "delta_plus_full_restore",
            "process_restart",
            "model_replacement",
            "tool_or_body_change",
            "sensor_or_source_interruption",
            "single_writer",
            "interactive_shell_independent",
            "recovery_limits_bound",
            "event_chain_valid",
        },
    )
    if subject.get("all_pass") is not True:
        raise Refused("G09 durability subject does not pass")
    design = _frozen_design(root, frozen)
    durability = design.get("durability")
    storage = design.get("storage")
    if not isinstance(durability, dict) or not isinstance(storage, dict):
        raise Refused("G09 frozen hardened design lacks durability/storage policy")
    if subject.get("checkpoint_policy") != {
        "delta_interval_seconds": storage.get("delta_checkpoint_interval_seconds"),
        "full_interval_seconds": storage.get("full_checkpoint_interval_seconds"),
    }:
        raise Refused("G09 checkpoint cadence is not bound to the frozen hardened design")
    rehearsals = subject.get("rehearsals")
    if not isinstance(rehearsals, list) or [row.get("frontier") if isinstance(row, dict) else None for row in rehearsals] != list(FRONTIER_IDS):
        raise Refused("G09 requires a durable recovery rehearsal for every frontier A-H")
    writer_locks: set[str] = set()
    for row in rehearsals:
        assert isinstance(row, dict)
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"candidate", "control"}:
            raise Refused(f"G09 {row['frontier']} must rehearse candidate and control recovery")
        _validate_recovery_arm(root, arms["candidate"], label=f"G09 {row['frontier']}.candidate", writer_locks=writer_locks)
        _validate_recovery_arm(root, arms["control"], label=f"G09 {row['frontier']}.control", writer_locks=writer_locks)
        interruptions = _assert_int(row.get("unplanned_interruptions"), label=f"G09 {row['frontier']}.unplanned_interruptions")
        single = _assert_int(row.get("max_single_unplanned_downtime_seconds"), label=f"G09 {row['frontier']}.max_single_unplanned_downtime_seconds")
        cumulative = _assert_int(row.get("cumulative_unplanned_downtime_seconds"), label=f"G09 {row['frontier']}.cumulative_unplanned_downtime_seconds")
        if (
            interruptions > durability.get("max_unplanned_interruptions_per_frontier", -1)
            or single > durability.get("max_single_unplanned_downtime_seconds", -1)
            or cumulative > durability.get("max_cumulative_unplanned_downtime_seconds", -1)
        ):
            raise Refused(f"G09 {row['frontier']} exceeds the frozen recovery allowance")
    disturbances = subject.get("scheduled_disturbance_receipts")
    required_disturbances = {"process_restart", "model_replacement", "tool_or_body_change", "sensor_or_source_interruption"}
    if not isinstance(disturbances, dict) or set(disturbances) != required_disturbances:
        raise Refused("G09 must bind every mandatory recovery disturbance receipt")
    for name in sorted(required_disturbances):
        _require_file_ref(root, disturbances[name], label=f"G09 scheduled_disturbance_receipts.{name}")


def _validate_g12(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require_frozen_subject_binding(root, "G12", subject, frozen)
    _assert_exact_true_checks(
        subject.get("checks"),
        label="G12 subject checks",
        required={
            "frozen_build_bound",
            "source_maps_bound",
            "runtime_mutants_injected",
            "runtime_mutants_detected",
            "clean_baselines_accepted",
            "guard_coverage_complete",
            "no_pending_mutations",
        },
    )
    if subject.get("all_pass") is not True or subject.get("survivors") != []:
        raise Refused("G12 requires a passing mutation report with zero survivors")
    rows = subject.get("mutations")
    if not isinstance(rows, list) or not rows:
        raise Refused("G12 requires durable rows for injected runtime mutations")
    ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if len(set(ids)) != len(ids) or not REQUIRED_MUTATION_IDS.issubset(set(ids)):
        missing = sorted(REQUIRED_MUTATION_IDS - set(ids))
        raise Refused(f"G12 mutation suite lacks required live attack coverage: {missing}")
    for row in rows:
        if not isinstance(row, dict):
            raise Refused("G12 mutation row must be an object")
        _assert_nonempty_text(row.get("target"), label=f"G12 {row.get('id')}.target")
        if row.get("injected") is not True or row.get("detected") is not True or row.get("survived") is not False or row.get("clean_case_passed") is not True:
            raise Refused(f"G12 mutation {row.get('id')} did not prove an injected clean/rejected pair")
        _require_file_ref(root, row.get("clean_receipt"), label=f"G12 {row.get('id')}.clean_receipt")
        _require_file_ref(root, row.get("mutant_receipt"), label=f"G12 {row.get('id')}.mutant_receipt")
    count = len(rows)
    expected_counts = {
        "declared_mutation_count": count,
        "injected_count": count,
        "detected_count": count,
        "pending_count": 0,
        "survivor_count": 0,
    }
    if any(subject.get(name) != expected for name, expected in expected_counts.items()):
        raise Refused("G12 mutation totals are inconsistent with its durable attack rows")
    if subject.get("uncovered") != [] or subject.get("undeclared") != []:
        raise Refused("G12 mutation report has uncovered or undeclared attacks")


def _validate_public_model_canary(root: Path, value: Any, frozen: dict[str, Any]) -> dict[str, str]:
    """Validate the inert technical screen that precedes a human G02 review.

    The receipt is deliberately not an attestation.  It merely preserves the
    exact public prompt cohort, body pins, bounded service observation, and
    deterministic tie-break that a real reviewer must inspect before naming a
    shared base body in G02.
    """
    _path, receipt = _require_file_ref(root, value, label="G02 public_model_canary")
    required = {
        "schema",
        "program",
        "status",
        "activation",
        "external_activation",
        "unqualified_nous",
        "scientific_evidence",
        "evidence_scope",
        "completed_at",
        "frozen_build_sha256",
        "canary_template_sha256",
        "runtime",
        "model_service_cap_bytes",
        "required_concurrent_clients",
        "selection_rule",
        "hidden_seed_commitments_materialized",
        "candidates",
        "selected_base_model",
        "checks",
        "all_pass",
        "non_claims",
        "sha256",
    }
    organ_fields = {
        "neutral_organ_prompt_sha256",
        "reasoning_effort_policy",
        "conversation_policy",
        "max_output_tokens",
    }
    observed_fields = set(receipt) if isinstance(receipt, dict) else set()
    if organ_fields.issubset(observed_fields):
        required |= organ_fields
    elif observed_fields & organ_fields:
        raise Refused("G02 public model-canary organ policy fields are incomplete")
    _assert_exact_keys(receipt, label="G02 public_model_canary receipt", required=required)
    if (
        receipt.get("schema") != PUBLIC_MODEL_CANARY_SCHEMA
        or receipt.get("program") != PROGRAM
        or receipt.get("status") != "pass"
        or receipt.get("activation") is not False
        or receipt.get("external_activation") is not False
        or receipt.get("unqualified_nous") is not False
        or receipt.get("scientific_evidence") is not False
        or receipt.get("evidence_scope") != "frozen_public_model_selection_canaries_only"
        or receipt.get("all_pass") is not True
        or receipt.get("hidden_seed_commitments_materialized") is not False
    ):
        raise Refused("G02 public model-canary receipt is not an inactive passing technical screen")
    _assert_nonempty_text(receipt.get("completed_at"), label="G02 public_model_canary.completed_at")
    if receipt.get("frozen_build_sha256") != frozen.get("sha256"):
        raise Refused("G02 public model-canary receipt is not bound to this frozen build")
    template_path = root / PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json"
    template = _read_json(template_path)
    expected_template_sha256 = frozen.get("input_sha256", {}).get("public_model_canary_template")
    if (
        not isinstance(expected_template_sha256, str)
        or file_digest(template_path) != expected_template_sha256
        or receipt.get("canary_template_sha256") != expected_template_sha256
    ):
        raise Refused("G02 public model-canary receipt is not bound to the frozen public case template")
    runtime = _assert_exact_keys(receipt.get("runtime"), label="G02 public_model_canary.runtime", required={"id", "version", "sha256"})
    _assert_nonempty_text(runtime.get("id"), label="G02 public_model_canary.runtime.id")
    _assert_nonempty_text(runtime.get("version"), label="G02 public_model_canary.runtime.version")
    runtime_sha256 = _assert_sha256(runtime.get("sha256"), label="G02 public_model_canary.runtime.sha256")
    resources = _frozen_design(root, frozen).get("resources")
    expected_cap = resources.get("shared_model_service_cap_gib") if isinstance(resources, dict) else None
    if not isinstance(expected_cap, int) or receipt.get("model_service_cap_bytes") != expected_cap * GIB:
        raise Refused("G02 public model-canary receipt does not retain the hardened model service cap")
    if receipt.get("required_concurrent_clients") != 8:
        raise Refused("G02 public model-canary receipt did not retain width-eight admission")
    if receipt.get("selection_rule") != template.get("selection_rule"):
        raise Refused("G02 public model-canary receipt selection rule drifted from the frozen template")
    if organ_fields.issubset(observed_fields):
        prompt = template.get("neutral_organ_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Refused("G02 public model-canary template lacks the neutral organ prompt")
        if receipt.get("neutral_organ_prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise Refused("G02 public model-canary neutral organ prompt digest drifted")
        if receipt.get("reasoning_effort_policy") != template.get("reasoning_effort_policy"):
            raise Refused("G02 public model-canary reasoning policy drifted")
        if receipt.get("conversation_policy") != template.get("conversation_policy"):
            raise Refused("G02 public model-canary conversation policy drifted")
        if receipt.get("max_output_tokens") != template.get("max_output_tokens"):
            raise Refused("G02 public model-canary output budget drifted")
    checks = _assert_exact_keys(receipt.get("checks"), label="G02 public_model_canary.checks", required=PUBLIC_MODEL_CANARY_CHECKS)
    if not isinstance(checks.get("all_configured_candidates_eligible"), bool):
        raise Refused("G02 public model-canary eligibility summary is malformed")
    if any(value is not True for name, value in checks.items() if name != "all_configured_candidates_eligible"):
        raise Refused("G02 public model-canary checks did not pass")
    non_claims = receipt.get("non_claims")
    if not isinstance(non_claims, list) or not non_claims or not all(isinstance(item, str) and item.strip() for item in non_claims):
        raise Refused("G02 public model-canary receipt must retain explicit non-claims")
    aliases = template.get("candidate_aliases")
    cases = template.get("case_set")
    if not isinstance(aliases, list) or not isinstance(cases, list):
        raise Refused("frozen public model-canary template is malformed")
    expected_case_ids = [row.get("id") if isinstance(row, dict) else None for row in cases]
    rows = receipt.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(aliases):
        raise Refused("G02 public model-canary receipt does not account for every frozen candidate")
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        candidate = _assert_exact_keys(
            row,
            label=f"G02 public_model_canary.candidates[{index}]",
            required={
                "base_model",
                "model_size_bytes",
                "service_peak_bytes",
                "swap_pageout_delta_bytes",
                "width_eight",
                "canary",
                "errors",
                "eligible",
            },
        )
        base = _assert_base_model_pin(candidate.get("base_model"), label=f"G02 public_model_canary.candidates[{index}].base_model")
        if base["id"] != aliases[index] or base["runtime_sha256"] != runtime_sha256:
            raise Refused("G02 public model-canary candidate does not bind the frozen alias and runtime")
        _assert_int(candidate.get("model_size_bytes"), label=f"G02 public_model_canary.candidates[{index}].model_size_bytes", minimum=1)
        service_peak = _assert_int(candidate.get("service_peak_bytes"), label=f"G02 public_model_canary.candidates[{index}].service_peak_bytes", minimum=1)
        eligible = candidate.get("eligible") is True
        if eligible and service_peak > receipt["model_service_cap_bytes"]:
            raise Refused("G02 public model-canary admitted a candidate above the shared service cap")
        if eligible and candidate.get("swap_pageout_delta_bytes") != 0:
            raise Refused("G02 public model-canary admitted a candidate with pageouts")
        width = _assert_exact_keys(
            candidate.get("width_eight"),
            label=f"G02 public_model_canary.candidates[{index}].width_eight",
            required={"requests", "completed", "all_responses_valid"},
        )
        if eligible and width != {"requests": 8, "completed": 8, "all_responses_valid": True}:
            raise Refused("G02 public model-canary candidate did not complete the width-eight probe")
        canary = _assert_exact_keys(
            candidate.get("canary"),
            label=f"G02 public_model_canary.candidates[{index}].canary",
            required={"total", "passed", "median_latency_ms", "case_results"},
        )
        total = _assert_int(canary.get("total"), label=f"G02 public_model_canary.candidates[{index}].canary.total", minimum=1)
        passed = _assert_int(canary.get("passed"), label=f"G02 public_model_canary.candidates[{index}].canary.passed", minimum=0)
        raw_latency = canary.get("median_latency_ms")
        if raw_latency is None and not eligible:
            latency = float("inf")
        else:
            latency = _assert_number(raw_latency, label=f"G02 public_model_canary.candidates[{index}].canary.median_latency_ms")
        if not math.isfinite(latency) and eligible or total != len(expected_case_ids) or passed > total:
            raise Refused("G02 public model-canary candidate score is invalid")
        case_results = canary.get("case_results")
        if (
            not isinstance(case_results, list)
            or not all(isinstance(item, dict) for item in case_results)
            or [item.get("id") for item in case_results] != expected_case_ids
        ):
            raise Refused("G02 public model-canary candidate lacks the exact frozen public cases")
        if sum(item.get("passed") is True for item in case_results) != passed:
            raise Refused("G02 public model-canary candidate score does not match its case rows")
        for case_index, result in enumerate(case_results):
            row_result = _assert_exact_keys(
                result,
                label=f"G02 public_model_canary.candidates[{index}].case_results[{case_index}]",
                required={"id", "response_sha256", "answer", "passed", "latency_ms"},
            )
            if eligible:
                _assert_sha256(
                    row_result.get("response_sha256"),
                    label=(
                        f"G02 public_model_canary.candidates[{index}]"
                        f".case_results[{case_index}].response_sha256"
                    ),
                )
                _assert_nonempty_text(
                    row_result.get("answer"),
                    label=(
                        f"G02 public_model_canary.candidates[{index}]"
                        f".case_results[{case_index}].answer"
                    ),
                )
                case_latency = _assert_number(
                    row_result.get("latency_ms"),
                    label=f"G02 public_model_canary.candidates[{index}].case_results[{case_index}].latency_ms",
                )
                valid_measurement = math.isfinite(case_latency)
            else:
                valid_measurement = (
                    row_result.get("response_sha256") is None or isinstance(row_result.get("response_sha256"), str)
                ) and (row_result.get("answer") is None or isinstance(row_result.get("answer"), str)) and (
                    row_result.get("latency_ms") is None or isinstance(row_result.get("latency_ms"), (int, float))
                )
            if not valid_measurement or not isinstance(row_result.get("passed"), bool):
                raise Refused("G02 public model-canary case row is invalid")
        errors = candidate.get("errors")
        if not isinstance(errors, list) or not all(isinstance(item, str) and item.strip() for item in errors):
            raise Refused("G02 public model-canary candidate errors are malformed")
        if eligible and errors:
            raise Refused("G02 public model-canary marked an errored candidate eligible")
        if eligible:
            normalized_rows.append({"base_model": base, "service_peak_bytes": service_peak, "passed": passed, "latency": latency})
    expected_winner = min(
        normalized_rows,
        key=lambda row: (-row["passed"], row["service_peak_bytes"], row["latency"], row["base_model"]["weight_sha256"]),
    )
    selected = _assert_base_model_pin(receipt.get("selected_base_model"), label="G02 public_model_canary.selected_base_model")
    if selected != expected_winner["base_model"]:
        raise Refused("G02 public model-canary selected base does not follow the frozen tie-break")
    return selected


def _validate_g02(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate the machine-pinned candidate/control pins and parity basis."""
    _assert_exact_keys(
        subject,
        label="G02 arm-selection subject",
        required=HUMAN_SUBJECT_BASE_FIELDS
        | {
            "selection_id",
            "public_model_canary",
            "base_model",
            "candidate",
            "controls_by_frontier",
            "parity_by_frontier",
            "selection_checks",
        },
    )
    _validate_converted_machine_subject_binding(root, "G02", subject, frozen)
    _assert_nonempty_text(subject.get("selection_id"), label="G02 selection_id")
    selected_base = _validate_public_model_canary(root, subject.get("public_model_canary"), frozen)
    base_model = _assert_base_model_pin(subject.get("base_model"), label="G02 base_model")
    if base_model != selected_base:
        raise Refused("G02 base model does not exactly match the reviewed public-canary selection")
    candidate = _assert_exact_keys(
        subject.get("candidate"),
        label="G02 candidate",
        required=ARM_PIN_FIELDS | {"treatment_id"},
    )
    candidate_pin = _assert_arm_pin({name: candidate[name] for name in ARM_PIN_FIELDS}, label="G02 candidate")
    _assert_nonempty_text(candidate.get("treatment_id"), label="G02 candidate.treatment_id")
    adapter_path = root / "src/substrate/odyssey_arms.py"
    if not adapter_path.is_file():
        raise Refused("G02 production arm adapter source is missing")
    adapter_sha256 = file_digest(adapter_path)
    if frozen.get("implementation_sha256", {}).get("odyssey_arms") != adapter_sha256:
        raise Refused("G02 production arm adapter is not bound by this frozen build")
    if candidate_pin["adapter_sha256"] != adapter_sha256:
        raise Refused("G02 candidate adapter does not match the frozen production arm source")
    controls = _assert_exact_keys(
        subject.get("controls_by_frontier"),
        label="G02 controls_by_frontier",
        required=set(FRONTIER_IDS),
    )
    parity = _assert_exact_keys(
        subject.get("parity_by_frontier"),
        label="G02 parity_by_frontier",
        required=set(FRONTIER_IDS),
    )
    for frontier in FRONTIER_IDS:
        control_pin = _assert_arm_pin(controls[frontier], label=f"G02 controls_by_frontier.{frontier}")
        if control_pin["revision"] != base_model["revision"] or control_pin["artifact_sha256"] != base_model["weight_sha256"]:
            raise Refused(f"G02 {frontier} control does not use the one shared base-model body")
        if control_pin["adapter_sha256"] != adapter_sha256:
            raise Refused(f"G02 {frontier} control adapter does not match the frozen production arm source")
        _assert_exact_true_check_set(
            parity[frontier],
            label=f"G02 parity_by_frontier.{frontier}",
            required=PARITY_FIELDS,
        )
    _assert_exact_true_check_set(
        subject.get("selection_checks"),
        label="G02 selection_checks",
        required={
            "pre_outcome_selection",
            "public_canary_receipt_reviewed",
            "one_shared_base_body_verified",
            "candidate_pin_complete",
            "control_pins_complete",
            "candidate_control_difference_declared",
            "parity_reviewed",
        },
    )
    if candidate_pin["revision"] != base_model["revision"] or candidate_pin["artifact_sha256"] != base_model["weight_sha256"]:
        raise Refused("G02 candidate does not use the one shared base-model body")


def _assert_g04_custody_limitations(value: Any) -> list[str]:
    """Refuse a G04 subject that omits the single-operator custody limitation."""
    if not isinstance(value, list) or not value:
        raise Refused("G04 custody_limitations must be a non-empty list")
    texts: list[str] = []
    for index, item in enumerate(value):
        texts.append(_assert_nonempty_text(item, label=f"G04 custody_limitations[{index}]"))
    blob = " ".join(texts).casefold()
    if "detectable" not in blob:
        raise Refused("G04 custody_limitations must state early reveal is detectable")
    if not any(token in blob for token in ("not prevented", "not prevent", "not impossible")):
        raise Refused("G04 custody_limitations must state early reveal is not prevented")
    if "no independent" not in blob and "single operator" not in blob:
        raise Refused("G04 custody_limitations must state no independent reveal authority")
    return texts


def _validate_g04(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate single-operator pre-launch commitment custody without multi-party claims.

    Surviving properties: evaluator-only answers, trace-lock-before-reveal, daily
    scores hidden, four distinct roots, per-frontier distinct task/answer/scorer
    commitments, and G03 manifest binding.  Lost property: independent human
    custody that made early reveal *impossible* — only *detectable* here.  That
    loss is mandatory in ``custody_independence`` and ``custody_limitations``.
    """
    _assert_exact_keys(
        subject,
        label="G04 custody-seal subject",
        required=HUMAN_SUBJECT_BASE_FIELDS
        | {
            "answers_evaluator_only",
            "trace_lock_before_answer_reveal",
            "daily_scores_hidden",
            "custody_independence",
            "custody_limitations",
            "roots",
            "frontiers",
            "pre_launch_commitment_seal",
            "day7_reveal",
            "custody_checks",
        },
    )
    _validate_converted_machine_subject_binding(root, "G04", subject, frozen)
    if subject.get("custody_independence") != G04_CUSTODY_INDEPENDENCE:
        raise Refused("G04 custody_independence must be single_operator")
    _assert_g04_custody_limitations(subject.get("custody_limitations"))
    if (
        subject.get("answers_evaluator_only") is not True
        or subject.get("trace_lock_before_answer_reveal") is not True
        or subject.get("daily_scores_hidden") is not True
    ):
        raise Refused("G04 custody policy must keep answers/scores evaluator-only until trace lock")
    roots = _assert_exact_keys(
        subject.get("roots"),
        label="G04 roots",
        required={"builder_visible_root", "candidate_visible_root", "evaluator_only_root", "publication_safe_root"},
    )
    root_values = {
        name: _assert_logical_root(roots[name], label=f"G04 roots.{name}")
        for name in sorted(roots)
    }
    if len(set(root_values.values())) != len(root_values):
        raise Refused("G04 custody roots must be distinct")
    frontiers = subject.get("frontiers")
    if not isinstance(frontiers, list) or [row.get("id") if isinstance(row, dict) else None for row in frontiers] != list(
        FRONTIER_IDS
    ):
        raise Refused("G04 requires one ordered custody commitment for every frontier A-H")
    commitments: dict[str, set[str]] = {
        "task_seed_commitment_sha256": set(),
        "answer_commitment_sha256": set(),
        "scorer_commitment_sha256": set(),
    }
    all_commitment_digests: set[str] = set()
    ordered_commitment_rows: list[dict[str, str]] = []
    for row in frontiers:
        assert isinstance(row, dict)
        _assert_exact_keys(
            row,
            label=f"G04 frontier {row.get('id')}",
            required={
                "id",
                "task_seed_commitment_sha256",
                "answer_commitment_sha256",
                "scorer_commitment_sha256",
                "candidate_manifest_sha256",
                "candidate_can_read_evaluator_only",
                "trace_lock_required",
                "daily_scores_hidden",
            },
        )
        frontier = row["id"]
        task_seed = _assert_sha256(row.get("task_seed_commitment_sha256"), label=f"G04 {frontier}.task_seed_commitment_sha256")
        answer = _assert_sha256(row.get("answer_commitment_sha256"), label=f"G04 {frontier}.answer_commitment_sha256")
        scorer = _assert_sha256(row.get("scorer_commitment_sha256"), label=f"G04 {frontier}.scorer_commitment_sha256")
        if len({task_seed, answer, scorer}) != 3:
            raise Refused(f"G04 {frontier} task-seed/answer/scorer commitments must be distinct digests")
        for name, digest_value in (
            ("task_seed_commitment_sha256", task_seed),
            ("answer_commitment_sha256", answer),
            ("scorer_commitment_sha256", scorer),
        ):
            if digest_value in commitments[name] or digest_value in all_commitment_digests:
                raise Refused(f"G04 {name} must be distinct for each frontier")
            commitments[name].add(digest_value)
            all_commitment_digests.add(digest_value)
        _assert_sha256(row.get("candidate_manifest_sha256"), label=f"G04 {frontier}.candidate_manifest_sha256")
        if (
            row.get("candidate_can_read_evaluator_only") is not False
            or row.get("trace_lock_required") is not True
            or row.get("daily_scores_hidden") is not True
        ):
            raise Refused(f"G04 {frontier} has a weakened custody policy")
        ordered_commitment_rows.append(
            {
                "id": frontier,
                "task_seed_commitment_sha256": task_seed,
                "answer_commitment_sha256": answer,
                "scorer_commitment_sha256": scorer,
                "candidate_manifest_sha256": row["candidate_manifest_sha256"],
            }
        )
    seal = _assert_exact_keys(
        subject.get("pre_launch_commitment_seal"),
        label="G04 pre_launch_commitment_seal",
        required={"sealed_before_launch", "commitment_set_sha256", "frontiers_commitment_chain_sha256"},
    )
    if seal.get("sealed_before_launch") is not True:
        raise Refused("G04 commitments must be sealed before launch")
    expected_set = digest({"frontiers": ordered_commitment_rows})
    expected_chain = digest(
        {
            "algorithm": "sha256_canonical_json",
            "ordered_frontier_commitments": ordered_commitment_rows,
        }
    )
    if seal.get("commitment_set_sha256") != expected_set:
        raise Refused("G04 pre_launch_commitment_seal.commitment_set_sha256 does not bind frontier commitments")
    if seal.get("frontiers_commitment_chain_sha256") != expected_chain:
        raise Refused("G04 pre_launch_commitment_seal.frontiers_commitment_chain_sha256 does not bind the chain")
    reveal = _assert_exact_keys(
        subject.get("day7_reveal"),
        label="G04 day7_reveal",
        required={
            "gated_on_trace_lock",
            "trace_lock_recipe_sha256",
            "trace_lock_recipe",
            "release_after_candidate_and_control_trace_lock",
        },
    )
    if reveal.get("gated_on_trace_lock") is not True:
        raise Refused("G04 reveal is not chained to trace lock")
    if reveal.get("release_after_candidate_and_control_trace_lock") is not True:
        raise Refused("G04 Day 7 reveal must require candidate and control trace lock")
    recipe = reveal.get("trace_lock_recipe")
    if recipe != G04_TRACE_LOCK_RECIPE:
        raise Refused("G04 day7_reveal.trace_lock_recipe must match the frozen single-operator recipe")
    recipe_digest = _assert_sha256(reveal.get("trace_lock_recipe_sha256"), label="G04 day7_reveal.trace_lock_recipe_sha256")
    if recipe_digest != digest(G04_TRACE_LOCK_RECIPE):
        raise Refused("G04 reveal is not chained to trace lock")
    _assert_exact_true_check_set(
        subject.get("custody_checks"),
        label="G04 custody_checks",
        required=set(G04_CUSTODY_CHECKS),
    )


def _validate_g05(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate complete model/tool configuration pins and per-frontier parity."""
    _assert_exact_keys(
        subject,
        label="G05 model-tool-panel subject",
        required=HUMAN_SUBJECT_BASE_FIELDS
        | {"panel_id", "models", "tools", "gateway", "frontier_assignments", "panel_checks"},
    )
    _validate_converted_machine_subject_binding(root, "G05", subject, frozen)
    _assert_nonempty_text(subject.get("panel_id"), label="G05 panel_id")
    models = subject.get("models")
    if not isinstance(models, list) or not models:
        raise Refused("G05 requires one or more pinned models")
    model_ids: set[str] = set()
    for index, pin in enumerate(models):
        normalized = _assert_arm_pin(pin, label=f"G05 models[{index}]")
        if normalized["id"] in model_ids:
            raise Refused("G05 model pin identifiers must be unique")
        model_ids.add(normalized["id"])
    tools = subject.get("tools")
    if not isinstance(tools, list) or not tools:
        raise Refused("G05 requires a non-empty pinned tool panel")
    tool_ids: set[str] = set()
    for index, tool in enumerate(tools):
        _assert_exact_keys(tool, label=f"G05 tools[{index}]", required={"id", "version", "artifact_sha256"})
        tool_id = _assert_nonempty_text(tool.get("id"), label=f"G05 tools[{index}].id")
        _assert_nonempty_text(tool.get("version"), label=f"G05 tools[{index}].version")
        _assert_sha256(tool.get("artifact_sha256"), label=f"G05 tools[{index}].artifact_sha256")
        if tool_id in tool_ids:
            raise Refused("G05 tool pin identifiers must be unique")
        tool_ids.add(tool_id)
    gateway = _assert_exact_keys(
        subject.get("gateway"),
        label="G05 gateway",
        required={"id", "revision", "artifact_sha256", "stateless"},
    )
    _assert_nonempty_text(gateway.get("id"), label="G05 gateway.id")
    _assert_nonempty_text(gateway.get("revision"), label="G05 gateway.revision")
    _assert_sha256(gateway.get("artifact_sha256"), label="G05 gateway.artifact_sha256")
    if gateway.get("stateless") is not True:
        raise Refused("G05 model gateway must be explicitly stateless")
    assignments = _assert_exact_keys(
        subject.get("frontier_assignments"),
        label="G05 frontier_assignments",
        required=set(FRONTIER_IDS),
    )
    for frontier in FRONTIER_IDS:
        assignment = _assert_exact_keys(
            assignments[frontier],
            label=f"G05 frontier_assignments.{frontier}",
            required={"candidate_model_id", "control_model_id", "candidate_tool_ids", "control_tool_ids"},
        )
        for name in ("candidate_model_id", "control_model_id"):
            model_id = _assert_nonempty_text(assignment.get(name), label=f"G05 {frontier}.{name}")
            if model_id not in model_ids:
                raise Refused(f"G05 {frontier}.{name} is not a pinned model")
        candidate_tools = assignment.get("candidate_tool_ids")
        control_tools = assignment.get("control_tool_ids")
        if not isinstance(candidate_tools, list) or not isinstance(control_tools, list):
            raise Refused(f"G05 {frontier} tool assignments must be lists")
        if candidate_tools != control_tools or len(set(candidate_tools)) != len(candidate_tools):
            raise Refused(f"G05 {frontier} candidate/control tool assignment is not parity-preserving")
        for tool_id in candidate_tools:
            if not isinstance(tool_id, str) or tool_id not in tool_ids:
                raise Refused(f"G05 {frontier} references an unpinned tool")
    _assert_exact_true_check_set(
        subject.get("panel_checks"),
        label="G05 panel_checks",
        required={
            "model_pins_complete",
            "tool_pins_complete",
            "stateless_gateway_pinned",
            "frontier_assignments_complete",
            "candidate_control_tool_parity",
        },
    )


def _validate_isolation_observation(
    root: Path,
    value: Any,
    *,
    label: str,
    kind: str,
    roots: dict[str, str],
    principals: dict[str, dict[str, Any]],
    mounts: dict[str, Any],
    frozen: dict[str, Any],
) -> Path:
    """Validate one factual G10 access/topology observation.

    Records the actual actor, attempted operation, target, errno for denials,
    and observed topology.  A denial recorded without a real attempt is refused.
    """
    expectation = ISOLATION_OBSERVATION_EXPECTATIONS[kind]
    path, observation = _require_file_ref(root, value, label=label)
    _assert_exact_keys(
        observation,
        label=label,
        required={
            "schema",
            "program",
            "status",
            "activation",
            "external_activation",
            "unqualified_nous",
            "frozen_build_sha256",
            "observation_kind",
            "observed_at",
            "command_argv",
            "actor_role",
            "actor_id",
            "actor_uid",
            "attempt",
            "access_result",
            "assertion_passed",
            "process_exit_code",
            "attempted",
            "errno_name",
            "errno",
            "topology",
            "sha256",
        },
    )
    if (
        observation.get("schema") != ISOLATION_OBSERVATION_SCHEMA
        or observation.get("program") != PROGRAM
        or observation.get("status") != "observed"
        or observation.get("activation") is not False
        or observation.get("external_activation") is not False
        or observation.get("unqualified_nous") is not False
        or observation.get("frozen_build_sha256") != frozen.get("sha256")
        or observation.get("observation_kind") != kind
    ):
        raise Refused(f"{label} does not identify the exact inactive G10 observation")
    _assert_nonempty_text(observation.get("observed_at"), label=f"{label}.observed_at")
    command = observation.get("command_argv")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and _assert_nonempty_text(item, label=f"{label}.command_argv") for item in command
    ):
        raise Refused(f"{label}.command_argv must be a non-empty explicit command list")
    role = expectation["actor_role"]
    assert isinstance(role, str)
    principal = principals[role]
    if (
        observation.get("actor_role") != role
        or observation.get("actor_id") != principal["id"]
        or observation.get("actor_uid") != principal["uid"]
    ):
        raise Refused(f"{label} actor does not bind the expected G10 principal")
    target_field = expectation["target_root_field"]
    expected_target = roots[target_field] if isinstance(target_field, str) else None
    attempt = _assert_exact_keys(observation.get("attempt"), label=f"{label}.attempt", required={"operation", "target_root"})
    if attempt != {"operation": expectation["operation"], "target_root": expected_target}:
        raise Refused(f"{label} does not bind the expected operation and target root")
    if observation.get("access_result") != expectation["access_result"] or observation.get("assertion_passed") is not True:
        raise Refused(f"{label} does not record the expected successful isolation assertion")
    if observation.get("attempted") is not True:
        if expectation["access_result"] == "denied":
            raise Refused(f"{label} denial was not attempted")
        raise Refused(f"{label} observation was not attempted")
    if expectation["access_result"] == "denied":
        errno_name = observation.get("errno_name")
        if errno_name not in ISOLATION_DENIAL_ERRNOS:
            raise Refused(f"{label} denial must record EACCES or EPERM errno_name")
        errno_value = observation.get("errno")
        if not isinstance(errno_value, int) or isinstance(errno_value, bool) or errno_value <= 0:
            raise Refused(f"{label} denial must record a positive errno integer")
        exit_code = _assert_int(observation.get("process_exit_code"), label=f"{label}.process_exit_code", minimum=0)
        if exit_code == 0:
            raise Refused(f"{label} denial cannot claim process_exit_code 0")
    else:
        if observation.get("errno_name") is not None or observation.get("errno") is not None:
            raise Refused(f"{label} topology observation must not claim a denial errno")
        _assert_int(observation.get("process_exit_code"), label=f"{label}.process_exit_code", minimum=0)
    topology = _assert_exact_keys(
        observation.get("topology"),
        label=f"{label}.topology",
        required={"roots", "principals", "mounts"},
    )
    if topology != {"roots": roots, "principals": principals, "mounts": mounts}:
        raise Refused(f"{label} topology does not exactly match the G10 topology")
    return path


def _validate_g10(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate machine-probed evaluator boundary isolations with real denials."""
    _assert_exact_keys(
        subject,
        label="G10 isolation-attestation subject",
        required=HUMAN_SUBJECT_BASE_FIELDS
        | {
            "isolation_mode",
            "candidate_can_read_evaluator_only",
            "candidate_can_write_evaluator_only",
            "evaluator_can_write_candidate_private_state",
            "builder_can_read_evaluator_only",
            "roots",
            "principals",
            "mounts",
            "isolation_receipts",
            "isolation_checks",
        },
    )
    _validate_converted_machine_subject_binding(root, "G10", subject, frozen)
    mode = subject.get("isolation_mode")
    if mode not in {"separate_uid", "mount_isolation"}:
        raise Refused("G10 requires separate_uid or mount_isolation")
    if any(
        subject.get(name) is not False
        for name in (
            "candidate_can_read_evaluator_only",
            "candidate_can_write_evaluator_only",
            "evaluator_can_write_candidate_private_state",
            "builder_can_read_evaluator_only",
        )
    ):
        raise Refused("G10 isolation policy permits a forbidden evaluator boundary access")
    roots = _assert_exact_keys(
        subject.get("roots"),
        label="G10 roots",
        required={"builder_visible_root", "candidate_visible_root", "evaluator_only_root", "publication_safe_root"},
    )
    root_values = {
        name: _assert_logical_root(roots[name], label=f"G10 roots.{name}")
        for name in sorted(roots)
    }
    if len(set(root_values.values())) != len(root_values):
        raise Refused("G10 isolation roots must be distinct")
    principals = _assert_exact_keys(subject.get("principals"), label="G10 principals", required={"candidate", "evaluator", "builder"})
    observed_uids: dict[str, int] = {}
    for role in ("candidate", "evaluator", "builder"):
        principal = _assert_exact_keys(principals[role], label=f"G10 principals.{role}", required={"id", "uid"})
        _assert_nonempty_text(principal.get("id"), label=f"G10 principals.{role}.id")
        observed_uids[role] = _assert_int(principal.get("uid"), label=f"G10 principals.{role}.uid")
    mounts = subject.get("mounts")
    if mode == "separate_uid":
        _assert_exact_keys(mounts, label="G10 mounts", required=set())
        if observed_uids["evaluator"] in {observed_uids["candidate"], observed_uids["builder"]}:
            raise Refused("G10 separate_uid requires an evaluator UID distinct from candidate and builder")
    else:
        mount_ids = _assert_exact_keys(
            mounts,
            label="G10 mounts",
            required={"candidate_mount_id", "evaluator_mount_id", "builder_mount_id"},
        )
        candidate_mount = _assert_nonempty_text(mount_ids.get("candidate_mount_id"), label="G10 mounts.candidate_mount_id")
        evaluator_mount = _assert_nonempty_text(mount_ids.get("evaluator_mount_id"), label="G10 mounts.evaluator_mount_id")
        builder_mount = _assert_nonempty_text(mount_ids.get("builder_mount_id"), label="G10 mounts.builder_mount_id")
        if evaluator_mount in {candidate_mount, builder_mount}:
            raise Refused("G10 mount_isolation requires an evaluator mount distinct from candidate and builder")
    receipts = _assert_exact_keys(
        subject.get("isolation_receipts"),
        label="G10 isolation_receipts",
        required=set(ISOLATION_OBSERVATION_EXPECTATIONS),
    )
    receipt_paths: set[str] = set()
    for name in ISOLATION_OBSERVATION_EXPECTATIONS:
        path = _validate_isolation_observation(
            root,
            receipts[name],
            label=f"G10 isolation_receipts.{name}",
            kind=name,
            roots=root_values,
            principals=principals,
            mounts=mounts,
            frozen=frozen,
        )
        relative = _relative(root, path)
        if relative in receipt_paths:
            raise Refused("G10 isolation receipts must be distinct observations")
        receipt_paths.add(relative)
    _assert_exact_true_check_set(
        subject.get("isolation_checks"),
        label="G10 isolation_checks",
        required={
            "candidate_evaluator_read_denied",
            "candidate_evaluator_write_denied",
            "evaluator_candidate_private_write_denied",
            "builder_evaluator_read_denied",
            "topology_observed",
            "no_shared_mutable_roots",
        },
    )


def _validate_g11(root: Path, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate a preregistered statistics/scoring authority against the design."""
    _assert_exact_keys(
        subject,
        label="G11 statistics-authority subject",
        required=HUMAN_SUBJECT_BASE_FIELDS
        | {
            "statistics_authority_id",
            "score_weights_frozen",
            "score_weights",
            "rubric_sha256",
            "analysis_plan_sha256",
            "primary_unit",
            "independent_unit_count",
            "repeated_observations_are_independent_replicates",
            "sesoi",
            "primary_methods",
            "secondary_event_model",
            "outcome_a_requires_all_eight_valid",
            "analysis_checks",
        },
    )
    _validate_converted_machine_subject_binding(root, "G11", subject, frozen)
    _assert_nonempty_text(subject.get("statistics_authority_id"), label="G11 statistics_authority_id")
    if subject.get("score_weights_frozen") is not True:
        raise Refused("G11 score weights are not frozen")
    weights = _assert_exact_keys(subject.get("score_weights"), label="G11 score_weights", required=set(SCORE_DIMENSIONS))
    total_weight = 0.0
    for dimension in SCORE_DIMENSIONS:
        weight = _assert_number(weights[dimension], label=f"G11 score_weights.{dimension}", minimum=0.0)
        if not math.isfinite(weight) or weight <= 0.0:
            raise Refused(f"G11 score_weights.{dimension} must be a finite positive number")
        total_weight += weight
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise Refused("G11 score weights must sum to exactly one")
    rubrics = _assert_exact_keys(subject.get("rubric_sha256"), label="G11 rubric_sha256", required=set(SCORE_DIMENSIONS))
    for dimension in SCORE_DIMENSIONS:
        _assert_sha256(rubrics[dimension], label=f"G11 rubric_sha256.{dimension}")
    _assert_sha256(subject.get("analysis_plan_sha256"), label="G11 analysis_plan_sha256")
    design = _frozen_design(root, frozen)
    statistics = design.get("statistics")
    independent = design.get("independent_units")
    if not isinstance(statistics, dict) or not isinstance(independent, dict):
        raise Refused("G11 frozen hardened design lacks statistics/independent-unit policy")
    if weights != statistics.get("score_weights"):
        raise Refused("G11 score weights differ from the frozen hardened design")
    if subject.get("primary_unit") != statistics.get("primary_unit"):
        raise Refused("G11 primary unit differs from the frozen hardened design")
    if subject.get("independent_unit_count") != independent.get("count"):
        raise Refused("G11 independent-unit count differs from the frozen hardened design")
    if subject.get("repeated_observations_are_independent_replicates") is not False:
        raise Refused("G11 may not relabel repeated observations as independent replicates")
    sesoi = _assert_number(subject.get("sesoi"), label="G11 sesoi", minimum=0.0)
    if not math.isfinite(sesoi) or sesoi != statistics.get("sesoi"):
        raise Refused("G11 SESOI differs from the frozen hardened design")
    if subject.get("primary_methods") != statistics.get("primary_methods"):
        raise Refused("G11 primary methods differ from the frozen hardened design")
    if subject.get("secondary_event_model") != statistics.get("secondary_event_model"):
        raise Refused("G11 secondary event model differs from the frozen hardened design")
    if subject.get("outcome_a_requires_all_eight_valid") != statistics.get("outcome_a_requires_all_eight_valid"):
        raise Refused("G11 outcome rule differs from the frozen hardened design")
    _assert_exact_true_check_set(
        subject.get("analysis_checks"),
        label="G11 analysis_checks",
        required={
            "score_weights_sum_to_one",
            "rubrics_pinned",
            "primary_unit_matches_design",
            "pseudoreplication_guard",
            "primary_methods_frozen",
            "outcome_rule_frozen",
        },
    )


def _validate_human_cross_gate_bindings(subjects: dict[str, dict[str, Any]]) -> None:
    """Ensure separately sealed machine subjects describe one compatible experiment."""
    manifests = _g03_manifest_rows(subjects["G03"])
    g04_frontiers = subjects["G04"]["frontiers"]
    for row in g04_frontiers:
        assert isinstance(row, dict)
        frontier = row["id"]
        if row["candidate_manifest_sha256"] != manifests[frontier]["file_sha256"]:
            raise Refused(f"G04 {frontier} custody commitment is not bound to the validated G03 manifest")

    candidate = subjects["G02"]["candidate"]
    controls = subjects["G02"]["controls_by_frontier"]
    panel_models = {row["id"]: row for row in subjects["G05"]["models"]}
    candidate_pin = {name: candidate[name] for name in ARM_PIN_FIELDS}
    if panel_models.get(candidate_pin["id"]) != candidate_pin:
        raise Refused("G05 model panel does not contain the exact G02 candidate pin")
    assignments = subjects["G05"]["frontier_assignments"]
    for frontier in FRONTIER_IDS:
        control_pin = controls[frontier]
        if panel_models.get(control_pin["id"]) != control_pin:
            raise Refused(f"G05 model panel does not contain the exact G02 {frontier} control pin")
        assignment = assignments[frontier]
        if assignment["candidate_model_id"] != candidate_pin["id"] or assignment["control_model_id"] != control_pin["id"]:
            raise Refused(f"G05 {frontier} panel assignment does not match G02 arm selection")

    if subjects["G10"]["roots"] != subjects["G04"]["roots"]:
        raise Refused("G10 isolation topology is not the exact G04 custody topology")
    if subjects["G04"].get("custody_independence") != G04_CUSTODY_INDEPENDENCE:
        raise Refused("G04 cross-gate binding requires single_operator custody_independence")


def _gate_specific_checks(root: Path, gate_id: str, subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Reject weak generic receipts before they can become a pass gate."""
    frozen_sha256 = frozen["sha256"]
    if gate_id == "G01":
        if not (
            subject.get("state") == "odyssey_preflight_authorized"
            and subject.get("preflight_authorized") is True
            and subject.get("frozen_build_sha256") == frozen_sha256
        ):
            raise Refused("G01 requires the exact authorized R2-to-Odyssey receipt")
    elif gate_id == "G02":
        _validate_g02(root, subject, frozen)
    elif gate_id == "G03":
        _validate_g03(root, subject, frozen)
    elif gate_id == "G04":
        _validate_g04(root, subject, frozen)
    elif gate_id == "G05":
        _validate_g05(root, subject, frozen)
    elif gate_id == "G06-DC":
        _validate_g06_dc(root, subject, frozen)
    elif gate_id == "G06":
        _validate_g06(root, subject, frozen)
    elif gate_id == "G06-DC":
        _validate_g06_dc(root, subject, frozen)
    elif gate_id == "G07":
        _validate_g07(root, subject, frozen)
    elif gate_id == "G08":
        _validate_g08(root, subject, frozen)
    elif gate_id == "G09":
        _validate_g09(root, subject, frozen)
    elif gate_id == "G14":
        required_checks = {
            "frozen_build_bound",
            "notifier_source_bound",
            "telegram_api_acknowledged",
            "probe_message_id_valid",
        }
        checks = subject.get("checks")
        delivery = subject.get("delivery")
        if subject.get("all_pass") is not True or not isinstance(checks, dict):
            raise Refused("G14 requires a passing Telegram probe receipt")
        if not all(checks.get(name) is True for name in required_checks):
            raise Refused("G14 Telegram probe has a failed or missing required check")
        if subject.get("source_commit") != _git_head(root):
            raise Refused("G14 Telegram probe is not for the current git HEAD")
        if subject.get("frozen_build_sha256") != frozen_sha256:
            raise Refused("G14 Telegram probe is not bound to this frozen build")
        if subject.get("notifier_source_sha256") != frozen.get("implementation_sha256", {}).get("telegram_notifier"):
            raise Refused("G14 Telegram notifier source is not bound to the frozen source map")
        if not isinstance(delivery, dict) or delivery.get("acknowledged") is not True:
            raise Refused("G14 Telegram probe lacks an acknowledged delivery")
        if not isinstance(delivery.get("message_id"), int) or delivery["message_id"] <= 0:
            raise Refused("G14 Telegram probe lacks a valid acknowledged message id")
    elif gate_id == "G13":
        required_checks = {
            "exact_commit_checkout",
            "scoped_tests",
            "ruff_check",
            "frozen_build_regeneration",
            "source_map_match",
        }
        checks = subject.get("checks")
        if subject.get("all_pass") is not True or not isinstance(checks, dict):
            raise Refused("G13 requires a passing clean-clone CI receipt")
        if not all(checks.get(name) is True for name in required_checks):
            raise Refused("G13 clean-clone receipt has a failed or missing required check")
        if subject.get("source_commit") != _git_head(root):
            raise Refused("G13 clean-clone receipt is not for the current git HEAD")
        if subject.get("frozen_build_sha256") != frozen_sha256:
            raise Refused("G13 clean-clone receipt is not bound to this frozen build")
        if subject.get("regenerated_frozen_build_sha256") != frozen_sha256:
            raise Refused("G13 clean clone did not reproduce this frozen build")
        if subject.get("implementation_sha256") != frozen.get("implementation_sha256"):
            raise Refused("G13 clean-clone implementation source map drifted")
        if subject.get("input_sha256") != frozen.get("input_sha256"):
            raise Refused("G13 clean-clone input source map drifted")
    elif gate_id == "G10":
        _validate_g10(root, subject, frozen)
    elif gate_id == "G11":
        _validate_g11(root, subject, frozen)
    elif gate_id == "G12":
        _validate_g12(root, subject, frozen)
    elif gate_id == "G15":
        if subject.get("frozen_build_sha256") != frozen_sha256:
            raise Refused("G15 is not bound to this frozen build")
        if subject.get("all_pass") is not True:
            raise Refused("G15 source and protocol receipt does not pass")
        if subject.get("source_digest") != source_digest_for_frozen(frozen):
            raise Refused("G15 source digest does not match the frozen implementation map")
        if subject.get("protocol_digest") != protocol_digest_for_frozen(frozen):
            raise Refused("G15 protocol digest does not match the frozen protocol")


def _validate_storage(root: Path, storage: Any, g07_subject: dict[str, Any]) -> dict[str, int | bool]:
    if not isinstance(storage, dict):
        raise Refused("storage_admission must be an object")
    required_fields = (
        "p95_private_growth_bytes",
        "largest_transient_bytes",
        "terminal_allowance_bytes",
        "explicit_model_reserve_bytes",
        "private_write_cap_bytes",
    )
    values: dict[str, int] = {}
    for field in required_fields:
        value = storage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Refused(f"storage_admission.{field} must be a non-negative integer")
        values[field] = value
    if values["p95_private_growth_bytes"] > values["private_write_cap_bytes"]:
        raise Refused("measured P95 private growth exceeds the declared private-write cap")
    runtime_required, launch_required = _storage_requirements(
        p95_total_private_growth=values["p95_private_growth_bytes"],
        largest_transient=values["largest_transient_bytes"],
        terminal_allowance=values["terminal_allowance_bytes"],
        explicit_model_reserve=values["explicit_model_reserve_bytes"],
        concurrent_transient_slots=_assert_int(
            g07_subject.get("concurrent_transient_slots"),
            label="G07 concurrent_transient_slots",
            minimum=1,
        ),
    )
    expected_subject = {
        "base_protected_floor_bytes": BASE_PROTECTED_FLOOR_BYTES,
        "runtime_required_free_bytes": runtime_required,
        "p95_private_growth_bytes": values["p95_private_growth_bytes"],
        "largest_transient_bytes": values["largest_transient_bytes"],
        "terminal_allowance_bytes": values["terminal_allowance_bytes"],
        "explicit_model_reserve_bytes": values["explicit_model_reserve_bytes"],
        "private_write_cap_bytes": values["private_write_cap_bytes"],
        "measured_required_free_bytes": launch_required,
    }
    for field, expected in expected_subject.items():
        if g07_subject.get(field) != expected:
            raise Refused(f"G07 storage rehearsal does not bind {field}")
    free = shutil.disk_usage(root).free
    return {
        "base_protected_floor_bytes": BASE_PROTECTED_FLOOR_BYTES,
        "runtime_required_free_bytes": runtime_required,
        "launch_required_free_bytes": launch_required,
        "measured_required_free_bytes": launch_required,
        # The worker consumes the runtime floor; launch itself additionally
        # requires the measured P95 growth envelope.
        "required_free_bytes": runtime_required,
        "measured_guard_bytes": runtime_required - BASE_PROTECTED_FLOOR_BYTES,
        "concurrent_transient_slots": FULL_WIDTH_TRANSIENT_SLOTS,
        "free_bytes_now": free,
        "dynamic_private_write_capacity_bytes": max(0, free - runtime_required),
        "headroom_bytes": free - launch_required,
        "admitted": free >= launch_required,
    }


def _g03_manifest_rows(subject: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the already-validated G03 rows, preserving their exact shape."""
    rows = subject.get("manifests")
    if not isinstance(rows, list) or [row.get("id") if isinstance(row, dict) else None for row in rows] != list(FRONTIER_IDS):
        raise Refused("worker binding requires the validated ordered G03 manifest rows A-H")
    return {row["id"]: row for row in rows if isinstance(row, dict)}


def _validate_worker_manifest_binding(
    root: Path,
    row: dict[str, Any],
    *,
    frontier_id: str,
    g03_binding: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Bind one executable worker row to the exact G03 candidate manifest.

    The worker itself reads only a path/hash.  The authority additionally
    persists a closed binding object so a later operator-input draft cannot
    point a valid A--H worker at a different, otherwise well-formed task bank.
    """
    binding = row.get("candidate_manifest_binding")
    if binding != g03_binding:
        raise Refused(f"worker.{frontier_id} candidate manifest binding does not exactly equal validated G03")
    if _relative(root, manifest_path) != g03_binding["path"] or manifest_sha256 != g03_binding["file_sha256"]:
        raise Refused(f"worker.{frontier_id} candidate manifest path/hash does not match validated G03")
    manifest = _read_json(manifest_path, require_digest=True)
    if (
        manifest.get("schema") != g03_binding["schema"]
        or manifest.get("frontier") != frontier_id
        or manifest.get("seed_commitment") != g03_binding["seed_commitment"]
        or _forbidden_candidate_key(manifest) is not None
        or not task_bank.candidate_is_structurally_safe(manifest)
    ):
        raise Refused(f"worker.{frontier_id} candidate manifest no longer matches its validated G03 identity")
    task_ids = _candidate_task_ids(manifest, frontier=frontier_id)
    if g03_binding["task_count"] != len(task_ids) or g03_binding["task_ids_sha256"] != digest({"task_ids": task_ids}):
        raise Refused(f"worker.{frontier_id} candidate task schedule drifted from validated G03")
    selection_sha256, bundle_sha256 = _validate_source_bundle(root, manifest, frontier=frontier_id)
    if g03_binding["source_selection_sha256"] != selection_sha256 or g03_binding["source_bundle_sha256"] != bundle_sha256:
        raise Refused(f"worker.{frontier_id} candidate source bundle drifted from validated G03")
    return binding


def _validate_worker(root: Path, worker: Any, g03_subject: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(worker, dict):
        raise Refused("worker must be an object")
    argv = worker.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item and "\n" not in item for item in argv):
        raise Refused("worker.argv must be a non-empty argv list without newlines")
    executable = Path(argv[0])
    if not executable.is_absolute():
        raise Refused("worker argv[0] must be an absolute executable path")
    source_files = worker.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise Refused("worker.source_files must be a non-empty list")
    observed: list[dict[str, str]] = []
    for row in source_files:
        if not isinstance(row, dict):
            raise Refused("worker source file entry must be an object")
        path = _resolve_relative(root, row.get("path"), label="worker source path")
        expected = _assert_sha256(row.get("sha256"), label="worker source sha256")
        if not path.is_file() or file_digest(path) != expected:
            raise Refused(f"worker source drift or absence: {path}")
        observed.append({"path": _relative(root, path), "sha256": expected})
    required_worker = "src/substrate/odyssey_worker.py"
    if required_worker not in {row["path"] for row in observed}:
        raise Refused("worker source binding must include src/substrate/odyssey_worker.py")
    run_root = _resolve_relative(root, worker.get("run_root"), label="worker.run_root")
    if not _inside(root / "runs", run_root):
        raise Refused("worker.run_root must live under runs/")
    if "evaluator" in str(worker.get("storage", "")).casefold():
        raise Refused("worker storage declaration may not expose evaluator material")
    frontiers = worker.get("frontiers")
    if not isinstance(frontiers, list) or len(frontiers) != len(FRONTIER_IDS):
        raise Refused("worker.frontiers must contain exactly eight candidate/control rows")
    g03_rows = _g03_manifest_rows(g03_subject)
    observed_frontiers: list[dict[str, Any]] = []
    for row in frontiers:
        if not isinstance(row, dict):
            raise Refused("worker frontier entry must be an object")
        frontier_id = row.get("id")
        if frontier_id not in FRONTIER_IDS:
            raise Refused("worker frontier has an unknown identifier")
        manifest = _resolve_relative(root, row.get("candidate_manifest"), label=f"worker.{frontier_id}.candidate_manifest")
        expected_manifest = _assert_sha256(row.get("candidate_manifest_sha256"), label=f"worker.{frontier_id}.candidate_manifest_sha256")
        if not manifest.is_file() or file_digest(manifest) != expected_manifest:
            raise Refused(f"worker candidate manifest drift or absence: {manifest}")
        binding = _validate_worker_manifest_binding(
            root,
            row,
            frontier_id=frontier_id,
            g03_binding=g03_rows[frontier_id],
            manifest_path=manifest,
            manifest_sha256=expected_manifest,
        )
        commands: dict[str, list[str]] = {}
        for command_name in ("candidate_command", "control_command"):
            command = row.get(command_name)
            if not isinstance(command, list) or not command or not all(isinstance(item, str) and item and "\n" not in item for item in command):
                raise Refused(f"worker.{frontier_id}.{command_name} must be a non-empty argv list")
            commands[command_name] = command
        observed_frontiers.append(
            {
                "id": frontier_id,
                "candidate_manifest": _relative(root, manifest),
                "candidate_manifest_sha256": expected_manifest,
                "candidate_manifest_binding": binding,
                **commands,
            }
        )
    if [row["id"] for row in observed_frontiers] != list(FRONTIER_IDS):
        raise Refused("worker frontiers must be ordered exactly A-H")
    phase_names = worker.get("phase_names")
    if phase_names != list(PHASE_NAMES):
        raise Refused("worker must preserve the sealed four-phase Odyssey schedule")
    if worker.get("phase_seconds") != 1800 or worker.get("microcycles_per_frontier") != 84:
        raise Refused("worker must preserve the sealed 30-minute phase and 84-cycle frontier schedule")
    max_parallel_frontiers = worker.get("max_parallel_frontiers")
    if max_parallel_frontiers != len(FRONTIER_IDS):
        raise Refused("worker must preserve the calibrated width-eight admission")
    checkpoint = worker.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("delta_interval_seconds") != 7200 or checkpoint.get("full_interval_seconds") != 43200:
        raise Refused("worker checkpoint cadence must be two-hour delta and twelve-hour full")
    storage = worker.get("storage")
    if not isinstance(storage, dict) or not storage:
        raise Refused("worker.storage must declare the candidate/control writable layout")
    return {
        "argv": argv,
        "source_files": observed,
        "run_root": _relative(root, run_root),
        "frontiers": observed_frontiers,
        "phase_names": list(PHASE_NAMES),
        "phase_seconds": 1800,
        "microcycles_per_frontier": 84,
        "max_parallel_frontiers": len(FRONTIER_IDS),
        "checkpoint": checkpoint,
        "storage": storage,
    }


def _validate_inputs(root: Path, inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    if inputs.get("schema") != "SUBSTRATE_ODYSSEY_OPERATOR_INPUTS/v1" or inputs.get("program") != PROGRAM:
        raise Refused("unexpected Odyssey operator-input schema or program")
    if inputs.get("input_status") != "sealed_for_preflight":
        raise Refused("operator inputs are not sealed for preflight")
    if _contains_placeholder(inputs):
        raise Refused("operator inputs contain a placeholder")
    _assert_nonempty_text(inputs.get("run_id"), label="run_id")
    approval = inputs.get("operator_approval")
    if not isinstance(approval, dict):
        raise Refused("operator_approval must be an explicit object")
    for field in ("actor", "attested_at", "scope"):
        _assert_nonempty_text(approval.get(field), label=f"operator_approval.{field}")
    frozen_sha = _assert_sha256(inputs.get("frozen_build_sha256"), label="frozen_build_sha256")
    frozen = _validate_frozen_build(root, frozen_sha)
    gate_refs = inputs.get("gate_evidence")
    if not isinstance(gate_refs, dict) or set(gate_refs) != set(GATE_SPECS):
        raise Refused("operator inputs must contain exactly G01-G15 gate evidence references")
    subjects: dict[str, dict[str, Any]] = {}
    evidence_summary: dict[str, dict[str, Any]] = {}
    for gate_id, spec in GATE_SPECS.items():
        reference = gate_refs[gate_id]
        if not isinstance(reference, dict):
            raise Refused(f"{gate_id} gate reference must be an object")
        gate_path = _resolve_relative(root, reference.get("path"), label=f"{gate_id} evidence.path")
        expected = _assert_sha256(reference.get("file_sha256"), label=f"{gate_id} evidence.file_sha256")
        if not gate_path.is_file() or file_digest(gate_path) != expected:
            raise Refused(f"{gate_id} evidence file is missing or drifted")
        gate = _read_json(gate_path, require_digest=True)
        subject, subject_ref = _gate_subject(root, gate, gate_id, frozen_sha)
        _gate_specific_checks(root, gate_id, subject, frozen)
        subjects[gate_id] = subject
        evidence_summary[gate_id] = {
            "name": spec["name"],
            "kind": spec["kind"],
            "gate_evidence": _relative(root, gate_path),
            "gate_evidence_file_sha256": expected,
            "subject": subject_ref,
        }
    _validate_human_cross_gate_bindings(subjects)
    storage = _validate_storage(root, inputs.get("storage_admission"), subjects["G07"])
    worker = _validate_worker(root, inputs.get("worker"), subjects["G03"])
    return frozen, evidence_summary, {"storage": storage, "worker": worker}


def seal_inputs(root: Path, draft_path: Path, output_path: Path) -> dict[str, Any]:
    """Self-seal fully explicit operator inputs; never fill a decision field."""
    draft = _read_json(draft_path)
    if draft.get("schema") != "SUBSTRATE_ODYSSEY_OPERATOR_INPUTS/v1" or draft.get("program") != PROGRAM:
        raise Refused("unexpected Odyssey operator-input draft")
    if draft.get("input_status") != "ready_for_preflight":
        raise Refused("input draft must explicitly be marked ready_for_preflight")
    if _contains_placeholder(draft):
        raise Refused("input draft contains a placeholder")
    sealed = dict(draft)
    sealed["input_status"] = "sealed_for_preflight"
    sealed.pop("sha256", None)
    sealed["sha256"] = digest(sealed)
    # Validate the now-sealed inputs before persisting them.  A failure does not
    # produce a partial authority or a misleading "almost ready" receipt.
    _validate_inputs(root, sealed)
    _write_json(output_path, sealed)
    return sealed


def seal_machine_gate(root: Path, gate_id: str, subject_path: Path, output_path: Path) -> dict[str, Any]:
    """Seal one fully validated machine gate without inferring a human decision."""
    spec = machine_gate_spec(gate_id)
    if spec is None or spec["kind"] != "machine_verified":
        raise Refused("only an explicitly machine-verified gate can be sealed here")
    frozen_path = root / PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen_document = _read_json(frozen_path, require_digest=True)
    frozen_sha256 = _assert_sha256(frozen_document.get("sha256"), label="frozen_build_sha256")
    frozen = _validate_frozen_build(root, frozen_sha256)
    subject = _read_json(subject_path, require_digest=True)
    if subject.get("schema") != spec["subject_schema"]:
        raise Refused(f"{gate_id} subject schema is not the required sealed schema")
    _gate_specific_checks(root, gate_id, subject, frozen)
    gate = _sealed(
        "SUBSTRATE_ODYSSEY_GATE_EVIDENCE/v1",
        {
            "gate_id": gate_id,
            "gate_name": spec["name"],
            "evidence_kind": spec["kind"],
            "frozen_build_sha256": frozen_sha256,
            "subject": {
                "path": _relative(root, subject_path),
                "file_sha256": file_digest(subject_path),
                "schema": spec["subject_schema"],
            },
            "checks": {
                "subject_self_digest_valid": True,
                "gate_specific_validation_passed": True,
                "frozen_build_bound": True,
            },
            "human_attestation": None,
        },
        status="pass",
    )
    _write_json(output_path, gate)
    return gate


def emit_protocol_digests(root: Path, output_path: Path) -> dict[str, Any]:
    """Write the one machine-derived G15 subject for the current frozen build."""
    frozen_document = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = _assert_sha256(frozen_document.get("sha256"), label="frozen_build_sha256")
    frozen = _validate_frozen_build(root, frozen_sha256)
    subject = _sealed(
        "SUBSTRATE_ODYSSEY_PROTOCOL_DIGESTS/v1",
        {
            "frozen_build_sha256": frozen_sha256,
            "source_digest": source_digest_for_frozen(frozen),
            "protocol_digest": protocol_digest_for_frozen(frozen),
            "implementation_sha256": frozen["implementation_sha256"],
            "input_sha256": frozen["input_sha256"],
            "all_pass": True,
        },
        status="pass",
    )
    _write_json(output_path, subject)
    return subject


def machine_gate_ids(root: Path) -> frozenset[str]:
    """Return only current, fully validated machine-gate evidence identifiers."""
    frozen_document = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = _assert_sha256(frozen_document.get("sha256"), label="frozen_build_sha256")
    frozen = _validate_frozen_build(root, frozen_sha256)
    admitted: set[str] = set()
    for path in sorted((root / MACHINE_GATE_EVIDENCE).glob("*.json")):
        try:
            gate = _read_json(path, require_digest=True)
            gate_id = gate.get("gate_id")
            if not isinstance(gate_id, str) or GATE_SPECS.get(gate_id, {}).get("kind") != "machine_verified":
                continue
            subject, _ = _gate_subject(root, gate, gate_id, frozen_sha256)
            _gate_specific_checks(root, gate_id, subject, frozen)
        except Refused:
            continue
        admitted.add(gate_id)
    return frozenset(admitted)


def preflight(root: Path, inputs_path: Path, output_path: Path) -> dict[str, Any]:
    """Evaluate all gates and live storage, without granting launch authority."""
    inputs = _read_json(inputs_path, require_digest=True)
    frozen, evidence, derived = _validate_inputs(root, inputs)
    result = _sealed(
        "SUBSTRATE_ODYSSEY_GATE_EVALUATION/v1",
        {
            "input_sha256": inputs["sha256"],
            "frozen_build_sha256": frozen["sha256"],
            "authority_source_sha256": file_digest(Path(__file__)),
            "gates": [{"id": gate_id, "status": "pass", **evidence[gate_id]} for gate_id in GATE_SPECS],
            "storage": derived["storage"],
            "worker": derived["worker"],
            "all_gates_pass": True,
            "preflight_admitted": derived["storage"]["admitted"],
            "launch_allowed": False,
        },
        status="admitted_waiting_for_authority_seal" if derived["storage"]["admitted"] else "refused_storage",
    )
    _write_json(output_path, result)
    return result


def seal(root: Path, inputs_path: Path, preflight_path: Path, output_path: Path) -> dict[str, Any]:
    """Create the one launchable authority only from a fresh admitted preflight."""
    inputs = _read_json(inputs_path, require_digest=True)
    preflight_receipt = _read_json(preflight_path, require_digest=True)
    if preflight_receipt.get("schema") != "SUBSTRATE_ODYSSEY_GATE_EVALUATION/v1":
        raise Refused("preflight receipt has the wrong schema")
    if preflight_receipt.get("input_sha256") != inputs.get("sha256"):
        raise Refused("preflight receipt is not bound to these exact operator inputs")
    if preflight_receipt.get("authority_source_sha256") != file_digest(Path(__file__)):
        raise Refused("authority source drifted since preflight")
    if preflight_receipt.get("all_gates_pass") is not True or preflight_receipt.get("preflight_admitted") is not True:
        raise Refused("all gates and live storage admission must pass before authority seal")
    # Re-evaluate rather than trusting an old pass receipt.  This detects a
    # changed subject, altered source file, or a storage drop between preflight
    # and seal.
    frozen, evidence, derived = _validate_inputs(root, inputs)
    if not derived["storage"]["admitted"]:
        raise Refused("live storage fell below the measured admission requirement")
    if (
        preflight_receipt.get("storage", {}).get("measured_required_free_bytes") != derived["storage"]["measured_required_free_bytes"]
        or preflight_receipt.get("storage", {}).get("runtime_required_free_bytes") != derived["storage"]["runtime_required_free_bytes"]
    ):
        raise Refused("preflight storage calculation does not match the current measured inputs")
    protocol_digest = _gate_subject(
        root,
        _read_json(_resolve_relative(root, inputs["gate_evidence"]["G15"]["path"], label="G15 evidence.path"), require_digest=True),
        "G15",
        frozen["sha256"],
    )[0]["protocol_digest"]
    authority = _sealed(
        "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        {
            "run_id": inputs["run_id"],
            "program_config": {
                "id": PROGRAM,
                "duration_seconds": 7 * 24 * 3600,
                "duration_hours": 168,
                "launch_allowed": True,
            },
            # Keep the compatibility keys consumed by the existing detached
            # supervisor while the worker implementation is being completed.
            "program": {"id": PROGRAM, "launch_allowed": True, "activation": False},
            "seal": {
                "status": "sealed",
                "all_gates_must_pass": True,
                "frozen_build_sha256": frozen["sha256"],
                "protocol_digest": protocol_digest,
                "authority_source_sha256": file_digest(Path(__file__)),
            },
            "launch_gates": [{"id": gate_id, "status": "pass", **evidence[gate_id]} for gate_id in GATE_SPECS],
            "operator_inputs": {"path": _relative(root, inputs_path), "file_sha256": file_digest(inputs_path), "sha256": inputs["sha256"]},
            "preflight": {"path": _relative(root, preflight_path), "file_sha256": file_digest(preflight_path), "sha256": preflight_receipt["sha256"]},
            "storage": derived["storage"],
            "worker": derived["worker"],
            "detached_worker_command": shlex.join(derived["worker"]["argv"]),
            "supervisor_source_sha256": file_digest(root / "src/substrate/odyssey7d.py"),
            "worker_source_sha256": next(row["sha256"] for row in derived["worker"]["source_files"] if row["path"] == "src/substrate/odyssey_worker.py"),
            "frozen_build_sha256": frozen["sha256"],
            "launch_allowed": True,
        },
        status="sealed_admitted",
    )
    _write_json(output_path, authority)
    return authority


def verify(root: Path, authority_path: Path) -> dict[str, Any]:
    """Read-only audit of a sealed authority against live files and storage."""
    authority = _read_json(authority_path, require_digest=True)
    checks: dict[str, bool] = {
        "schema": authority.get("schema") == "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "program": authority.get("program", {}).get("id") == PROGRAM,
        "sealed": authority.get("seal", {}).get("status") == "sealed",
        "launch_allowed": authority.get("program", {}).get("launch_allowed") is True and authority.get("launch_allowed") is True,
        "all_gates": (
            [row.get("id") for row in authority.get("launch_gates", [])] == list(GATE_SPECS)
            and all(row.get("status") == "pass" for row in authority.get("launch_gates", []))
        ),
        "authority_source": authority.get("seal", {}).get("authority_source_sha256") == file_digest(Path(__file__)),
        "supervisor_source": authority.get("supervisor_source_sha256") == file_digest(root / "src/substrate/odyssey7d.py"),
        "worker_source": (
            authority.get("worker_source_sha256") == file_digest(root / "src/substrate/odyssey_worker.py")
            if (root / "src/substrate/odyssey_worker.py").is_file()
            else False
        ),
        "argv_round_trip": shlex.split(authority.get("detached_worker_command", "")) == authority.get("worker", {}).get("argv"),
    }
    storage = authority.get("storage", {})
    if isinstance(storage, dict):
        checks["storage_live"] = shutil.disk_usage(root).free >= storage.get("required_free_bytes", -1)
        checks["base_floor"] = storage.get("base_protected_floor_bytes") == BASE_PROTECTED_FLOOR_BYTES
    else:
        checks["storage_live"] = False
        checks["base_floor"] = False
    return {
        "schema": "SUBSTRATE_ODYSSEY_AUTHORITY_VERIFICATION/v1",
        "program": PROGRAM,
        "authority": _relative(root, authority_path),
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def _path_argument(root: Path, raw: Path) -> Path:
    path = raw.expanduser().resolve()
    if not _inside(root, path):
        raise Refused(f"path must stay inside repository root: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Substrate Odyssey authority control plane")
    parser.add_argument(
        "command",
        choices=(
            "template",
            "gate-template",
            "human-evidence-pack",
            "protocol-digests",
            "seal-machine-gate",
            "seal-inputs",
            "preflight",
            "seal",
            "verify",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gate", choices=tuple(GATE_SPECS) + tuple(SIDE_MACHINE_GATE_SPECS))
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--subject", type=Path)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "template":
            if args.out is None:
                raise Refused("template requires --out")
            result = _input_template()
            _write_json(_path_argument(root, args.out), result)
        elif args.command == "gate-template":
            if args.out is None or args.gate is None:
                raise Refused("gate-template requires --gate and --out")
            result = _gate_template(args.gate)
            _write_json(_path_argument(root, args.out), result)
        elif args.command == "human-evidence-pack":
            if args.out is None:
                raise Refused("human-evidence-pack requires --out")
            result = human_evidence_pack(root)
            _write_json(_path_argument(root, args.out), result)
        elif args.command == "seal-machine-gate":
            if args.out is None or args.gate is None or args.subject is None:
                raise Refused("seal-machine-gate requires --gate, --subject, and --out")
            result = seal_machine_gate(
                root,
                args.gate,
                _path_argument(root, args.subject),
                _path_argument(root, args.out),
            )
        elif args.command == "protocol-digests":
            if args.out is None:
                raise Refused("protocol-digests requires --out")
            result = emit_protocol_digests(root, _path_argument(root, args.out))
        elif args.command == "seal-inputs":
            if args.draft is None or args.out is None:
                raise Refused("seal-inputs requires --draft and --out")
            result = seal_inputs(root, _path_argument(root, args.draft), _path_argument(root, args.out))
        elif args.command == "preflight":
            if args.inputs is None or args.out is None:
                raise Refused("preflight requires --inputs and --out")
            result = preflight(root, _path_argument(root, args.inputs), _path_argument(root, args.out))
        elif args.command == "seal":
            if args.inputs is None or args.preflight is None or args.out is None:
                raise Refused("seal requires --inputs, --preflight, and --out")
            result = seal(root, _path_argument(root, args.inputs), _path_argument(root, args.preflight), _path_argument(root, args.out))
        else:
            if args.authority is None:
                raise Refused("verify requires --authority")
            result = verify(root, _path_argument(root, args.authority))
    except Refused as error:
        print(json.dumps({"refused": str(error), "activation": False}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
