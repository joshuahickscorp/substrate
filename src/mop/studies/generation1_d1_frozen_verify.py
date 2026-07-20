
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT

AGGREGATE_SCHEMA = "mop-generation1-c3-d1-frozen-producer-challenge/v1"
VERIFICATION_SCHEMA = "mop-generation1-d1-frozen-evidence-closure/v1"
BYTE_BOUND_VERIFICATION_SCHEMA = "mop-generation1-d1-frozen-byte-bound-verification/v1"
PROGRAM_ID = "generation1-c3-d1-frozen-producer-challenge-v1"
VERIFIER_ID = "generation1-d1-frozen-structural-verifier-v1"
CLAIM_SCOPE = (
    "independent structural and frozen-gate verification of generated D1 evidence; "
    "not independent scientific generation"
)
AGGREGATE_CLAIM_SCOPE = (
    "frozen generated D1 producer and challenge replication; independent verification pending"
)
INTERPRETATION_LIMIT = (
    "The producer and challenge use disjoint fresh seeds but share one implementation. "
    "They can establish repeatability of the frozen generated pattern, not independent "
    "verification, activation, or scientific promotion."
)

DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.verification.json"
DEFAULT_RUNG_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID

FROZEN_VARIANT_ID = "centroid-h64-e60-lr03"
FROZEN_VARIANT = {
    "variant_id": FROZEN_VARIANT_ID,
    "feature_set": "centroid",
    "hidden": 64,
    "epochs": 60,
    "lr": 0.003,
}
SCREEN_BINDING = {
    "path": "proof/GENERATION1_C3_D1_ROUTER_REDESIGN_SCREEN.json",
    "file_sha256": "edd845aae19d679cbd9eabb4c7fc4d6571c5d9be2a3af7535d653eeb8f2262d9",
    "result_sha256": "457547218ccc134276720acb804ad57c39a53a061f64519c0b8e654a3cb08c92",
    "frozen_variant_id": FROZEN_VARIANT_ID,
}
CRITERIA = {
    "minimum_mean_advantage_over_each_static_control": 0.01,
    "comparison_interval_lower_bound_must_exceed": 0.0,
    "minimum_favorable_seed_fraction": 0.75,
    "maximum_mean_gap_below_fixed_c2_context_route": 0.02,
    "minimum_work_saving_vs_all_five_actors": 0.70,
    "all_conditions_required": True,
}

PHASES = ("producer", "challenge")
STATIC_CONTROLS = ("global_static", "difficulty_static")
DIFFERENCE_CONTROLS = (
    "context_route_nonpromotable",
    "difficulty_static",
    "global_static",
    "random_actor",
)
OVERALL_ARMS = (
    "context_route_nonpromotable",
    "difficulty_static",
    "global_static",
    "learned_dispatch",
    "oracle_nonpromotable",
    "random_actor",
)
CONDITION_FIELDS = (
    "context_route_gap_gate",
    "favorable_seed_fraction_gate",
    "static_margin_gate",
    "work_saving_gate",
)

RUNGS_PER_PHASE = 288
RUNG_COUNT = RUNGS_PER_PHASE * 2
SEEDS_PER_RUNG = 576
DIFFICULTY_COUNT = 5
CELLS_PER_RUNG = SEEDS_PER_RUNG * DIFFICULTY_COUNT
PHASE_SEED_COUNT = RUNGS_PER_PHASE * SEEDS_PER_RUNG
PHASE_CELL_COUNT = RUNGS_PER_PHASE * CELLS_PER_RUNG
TOTAL_SEED_COUNT = PHASE_SEED_COUNT * 2
TOTAL_CELL_COUNT = PHASE_CELL_COUNT * 2

NULL_SAFE_PRUNE = "null_safe_prune"
STRUCTURAL_ONLY = "structural_summary_only_unclosed"
STRUCTURAL_ONLY_INTERPRETATION = "summary_consistent_pending_raw_rung_replay"
NONCONFIRMATORY_GATE_FAILURE = "nonconfirmatory_gate_failure_structurally_verified_no_independent_generator"
NONCONFIRMATORY_MIXED_RESULT = "nonconfirmatory_mixed_gate_result"
NONCONFIRMATORY_CANDIDATE = "nonconfirmatory_candidate_pattern"

_AGGREGATE_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "claim_scope",
        "screen_binding",
        "frozen_variant",
        "criteria",
        "rungs",
        "phases",
        "grid",
        "overall",
        "learned_dispatch_differences",
        "decision",
        "interpretation_limit",
        "complete",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "result_sha256",
    }
)
_PHASE_FIELDS = frozenset(
    {
        "grid",
        "overall",
        "learned_dispatch_differences",
        "favorable_seed_fraction",
        "mean_gap_below_context_route",
        "work_saving_vs_all_five_actors",
        "conditions",
        "all_frozen_criteria_passed",
    }
)
_VERIFICATION_FIELDS = frozenset(
    {
        "schema",
        "verifier_id",
        "claim_scope",
        "source",
        "checks",
        "classification",
        "verification_complete",
        "problems",
        "independent_scientific_generation",
        "ready_for_confirmatory_claim",
        "activation_allowed",
        "scientific_promotion",
        "verification_sha256",
    }
)
_BYTE_BOUND_FIELDS = frozenset(
    {
        "schema",
        "verifier_id",
        "claim_scope",
        "source",
        "rung_replay",
        "verification",
        "classification",
        "verification_complete",
        "problems",
        "independent_scientific_generation",
        "ready_for_confirmatory_claim",
        "activation_allowed",
        "scientific_promotion",
        "artifact_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = {key: item for key, item in value.items() if key != field}
    return {**core, field: canonical_sha256(core)}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _read_regular_json_object(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], bytes, Path]:
    candidate = Path(path)
    resolved = candidate.resolve()
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    try:
        before = resolved.stat()
        raw = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {resolved}") from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(raw) != after.st_size:
        raise ValueError(f"{label} changed while it was being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return value, raw, resolved


def _read_regular_aggregate(path: Path) -> tuple[dict[str, Any], bytes, Path]:
    return _read_regular_json_object(path, "frozen D1 aggregate")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} field inventory drifted")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _close(left: Any, right: Any) -> bool:
    return (
        _is_number(left)
        and _is_number(right)
        and math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


def _canonically_equal(left: Any, right: Any) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def _validate_ci(value: Any, label: str, expected_n: int) -> None:
    row = _require_mapping(value, label)
    _require_exact_fields(row, frozenset({"mean", "lo", "hi", "n"}), label)
    if not _is_integer(row["n"]) or row["n"] != expected_n:
        raise ValueError(f"{label} observation count drifted")
    if not all(_is_number(row[field]) for field in ("mean", "lo", "hi")):
        raise ValueError(f"{label} contains a non-finite estimate")
    if not float(row["lo"]) <= float(row["mean"]) <= float(row["hi"]):
        raise ValueError(f"{label} interval ordering drifted")


def _expected_conditions(phase: Mapping[str, Any]) -> dict[str, bool]:
    differences = _require_mapping(
        phase.get("learned_dispatch_differences"),
        "frozen D1 phase differences",
    )
    favorable = _require_mapping(
        phase.get("favorable_seed_fraction"),
        "frozen D1 favorable fractions",
    )
    static_margin = all(
        float(_require_mapping(differences[control], f"{control} difference")["mean"])
        >= float(CRITERIA["minimum_mean_advantage_over_each_static_control"])
        and float(_require_mapping(differences[control], f"{control} difference")["lo"])
        > float(CRITERIA["comparison_interval_lower_bound_must_exceed"])
        for control in STATIC_CONTROLS
    )
    favorable_gate = all(
        float(favorable[control]) >= float(CRITERIA["minimum_favorable_seed_fraction"])
        for control in STATIC_CONTROLS
    )
    context_gate = float(phase["mean_gap_below_context_route"]) <= float(
        CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"]
    )
    work_gate = float(phase["work_saving_vs_all_five_actors"]) >= float(
        CRITERIA["minimum_work_saving_vs_all_five_actors"]
    )
    return {
        "static_margin_gate": static_margin,
        "favorable_seed_fraction_gate": favorable_gate,
        "context_route_gap_gate": context_gate,
        "work_saving_gate": work_gate,
    }


def _validate_phase(value: Any, phase_name: str) -> bool:
    phase = _require_mapping(value, f"frozen D1 {phase_name} phase")
    _require_exact_fields(phase, _PHASE_FIELDS, f"frozen D1 {phase_name} phase")

    grid = _require_mapping(phase["grid"], f"frozen D1 {phase_name} grid")
    _require_exact_fields(
        grid,
        frozenset(
            {
                "rung_count",
                "train_seed_count",
                "heldout_seed_count",
                "completed_cell_count",
            }
        ),
        f"frozen D1 {phase_name} grid",
    )
    expected_grid = {
        "rung_count": RUNGS_PER_PHASE,
        "train_seed_count": PHASE_SEED_COUNT,
        "heldout_seed_count": PHASE_SEED_COUNT,
        "completed_cell_count": PHASE_CELL_COUNT,
    }
    if any(not _is_integer(grid[field]) for field in expected_grid) or not _canonically_equal(
        grid, expected_grid
    ):
        raise ValueError(f"frozen D1 {phase_name} grid drifted")

    overall = _require_mapping(phase["overall"], f"frozen D1 {phase_name} overall")
    _require_exact_fields(overall, frozenset(OVERALL_ARMS), f"frozen D1 {phase_name} overall")
    for arm in OVERALL_ARMS:
        _validate_ci(overall[arm], f"frozen D1 {phase_name} {arm}", PHASE_CELL_COUNT)

    differences = _require_mapping(
        phase["learned_dispatch_differences"],
        f"frozen D1 {phase_name} differences",
    )
    _require_exact_fields(
        differences,
        frozenset(DIFFERENCE_CONTROLS),
        f"frozen D1 {phase_name} differences",
    )
    for control in DIFFERENCE_CONTROLS:
        _validate_ci(
            differences[control],
            f"frozen D1 {phase_name} difference against {control}",
            PHASE_CELL_COUNT,
        )
        expected_mean = float(_require_mapping(overall["learned_dispatch"], "learned dispatch")["mean"])
        expected_mean -= float(_require_mapping(overall[control], f"{control} overall")["mean"])
        if not _close(
            _require_mapping(differences[control], f"{control} difference")["mean"],
            expected_mean,
        ):
            raise ValueError(f"frozen D1 {phase_name} {control} mean difference drifted")

    favorable = _require_mapping(
        phase["favorable_seed_fraction"],
        f"frozen D1 {phase_name} favorable fractions",
    )
    _require_exact_fields(
        favorable,
        frozenset(STATIC_CONTROLS),
        f"frozen D1 {phase_name} favorable fractions",
    )
    if any(
        not _is_number(favorable[control]) or not 0.0 <= float(favorable[control]) <= 1.0
        for control in STATIC_CONTROLS
    ):
        raise ValueError(f"frozen D1 {phase_name} favorable fraction drifted")

    context_gap = phase["mean_gap_below_context_route"]
    work_saving = phase["work_saving_vs_all_five_actors"]
    if (
        not _is_number(context_gap)
        or float(context_gap) < 0.0
        or not _close(
            context_gap,
            -float(
                _require_mapping(
                    differences["context_route_nonpromotable"],
                    "context-route difference",
                )["mean"]
            ),
        )
    ):
        raise ValueError(f"frozen D1 {phase_name} context-route gap drifted")
    if not _is_number(work_saving) or not 0.0 <= float(work_saving) <= 1.0:
        raise ValueError(f"frozen D1 {phase_name} work saving drifted")

    conditions = _require_mapping(
        phase["conditions"],
        f"frozen D1 {phase_name} conditions",
    )
    _require_exact_fields(
        conditions,
        frozenset(CONDITION_FIELDS),
        f"frozen D1 {phase_name} conditions",
    )
    if any(not isinstance(conditions[field], bool) for field in CONDITION_FIELDS):
        raise ValueError(f"frozen D1 {phase_name} gate type drifted")
    expected_conditions = _expected_conditions(phase)
    if not _canonically_equal(conditions, expected_conditions):
        raise ValueError(f"frozen D1 {phase_name} frozen gate recomputation drifted")
    expected_passed = all(expected_conditions.values())
    if phase["all_frozen_criteria_passed"] is not expected_passed:
        raise ValueError(f"frozen D1 {phase_name} candidate gate drifted")
    return bool(expected_passed)


def validate_frozen_aggregate(value: Mapping[str, Any]) -> None:

    aggregate = _require_mapping(value, "frozen D1 aggregate")
    _require_exact_fields(aggregate, _AGGREGATE_FIELDS, "frozen D1 aggregate")
    core = {key: item for key, item in aggregate.items() if key != "result_sha256"}
    if not _is_sha256(aggregate["result_sha256"]) or aggregate["result_sha256"] != canonical_sha256(core):
        raise ValueError("frozen D1 aggregate self-seal is invalid")
    variant = _require_mapping(aggregate["frozen_variant"], "frozen D1 variant")
    criteria = _require_mapping(aggregate["criteria"], "frozen D1 criteria")
    screen_binding = _require_mapping(
        aggregate["screen_binding"],
        "frozen D1 screen binding",
    )
    if (
        aggregate["schema"] != AGGREGATE_SCHEMA
        or aggregate["program_id"] != PROGRAM_ID
        or aggregate["claim_scope"] != AGGREGATE_CLAIM_SCOPE
        or not _canonically_equal(screen_binding, SCREEN_BINDING)
        or not _canonically_equal(variant, FROZEN_VARIANT)
        or not _canonically_equal(criteria, CRITERIA)
        or aggregate["interpretation_limit"] != INTERPRETATION_LIMIT
        or aggregate["complete"] is not True
        or aggregate["problems"] != []
        or aggregate["activation_allowed"] is not False
        or aggregate["scientific_promotion"] is not False
    ):
        raise ValueError("frozen D1 aggregate identity, authority, or safety drifted")

    rungs = aggregate["rungs"]
    if not isinstance(rungs, list) or len(rungs) != RUNG_COUNT:
        raise ValueError("frozen D1 aggregate rung inventory drifted")
    result_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for index, row_value in enumerate(rungs):
        row = _require_mapping(row_value, f"frozen D1 rung {index}")
        _require_exact_fields(
            row,
            frozenset(
                {
                    "index",
                    "phase",
                    "result_sha256",
                    "config_sha256",
                    "completed_cell_count",
                }
            ),
            f"frozen D1 rung {index}",
        )
        expected_phase = "producer" if index < RUNGS_PER_PHASE else "challenge"
        if (
            not _is_integer(row["index"])
            or row["index"] != index
            or row["phase"] != expected_phase
            or not _is_sha256(row["result_sha256"])
            or not _is_sha256(row["config_sha256"])
            or not _is_integer(row["completed_cell_count"])
            or row["completed_cell_count"] != CELLS_PER_RUNG
        ):
            raise ValueError(f"frozen D1 rung {index} identity or completion drifted")
        result_hash = str(row["result_sha256"])
        config_hash = str(row["config_sha256"])
        if result_hash in result_hashes or config_hash in config_hashes:
            raise ValueError("frozen D1 aggregate contains duplicate rung result or config hashes")
        result_hashes.add(result_hash)
        config_hashes.add(config_hash)

    phases = _require_mapping(aggregate["phases"], "frozen D1 phases")
    _require_exact_fields(phases, frozenset(PHASES), "frozen D1 phases")
    producer_passed = _validate_phase(phases["producer"], "producer")
    challenge_passed = _validate_phase(phases["challenge"], "challenge")

    grid = _require_mapping(aggregate["grid"], "frozen D1 aggregate grid")
    _require_exact_fields(
        grid,
        frozenset(
            {
                "phase_count",
                "rung_count",
                "train_seed_count",
                "heldout_seed_count",
                "completed_cell_count",
            }
        ),
        "frozen D1 aggregate grid",
    )
    expected_grid = {
        "phase_count": 2,
        "rung_count": RUNG_COUNT,
        "train_seed_count": TOTAL_SEED_COUNT,
        "heldout_seed_count": TOTAL_SEED_COUNT,
        "completed_cell_count": TOTAL_CELL_COUNT,
    }
    if any(not _is_integer(grid[field]) for field in expected_grid) or not _canonically_equal(
        grid, expected_grid
    ):
        raise ValueError("frozen D1 aggregate grid drifted")

    challenge = _require_mapping(phases["challenge"], "frozen D1 challenge phase")
    if not _canonically_equal(aggregate["overall"], challenge["overall"]) or not _canonically_equal(
        aggregate["learned_dispatch_differences"],
        challenge["learned_dispatch_differences"],
    ):
        raise ValueError("frozen D1 aggregate challenge projection drifted")

    decision = _require_mapping(aggregate["decision"], "frozen D1 decision")
    _require_exact_fields(
        decision,
        frozenset(
            {
                "producer_all_frozen_criteria_passed",
                "challenge_all_frozen_criteria_passed",
                "frozen_pattern_repeated",
                "independent_verification_complete",
                "ready_for_confirmatory_claim",
                "next_action",
            }
        ),
        "frozen D1 decision",
    )
    expected_decision = {
        "producer_all_frozen_criteria_passed": producer_passed,
        "challenge_all_frozen_criteria_passed": challenge_passed,
        "frozen_pattern_repeated": producer_passed and challenge_passed,
        "independent_verification_complete": False,
        "ready_for_confirmatory_claim": False,
        "next_action": "run_v1_m1_g1_sibling_batch",
    }
    if not _canonically_equal(decision, expected_decision):
        raise ValueError("frozen D1 decision or interpretation boundary drifted")


def _classification_from_gates(
    producer: bool,
    challenge: bool,
    *,
    raw_replay_complete: bool,
) -> dict[str, Any]:
    if not raw_replay_complete:
        return {
            "evidence_classification": STRUCTURAL_ONLY,
            "scientific_interpretation": STRUCTURAL_ONLY_INTERPRETATION,
            "producer_candidate_gate_passed": producer,
            "challenge_candidate_gate_passed": challenge,
            "independent_structural_artifact_verification_complete": False,
            "independent_scientific_generation": False,
            "exact_centroid_design_eligible_for_future_freeze": False,
            "disposition": "replay_all_raw_rungs_before_evidence_closure",
        }
    if not producer and not challenge:
        evidence_classification = NULL_SAFE_PRUNE
        scientific_interpretation = NONCONFIRMATORY_GATE_FAILURE
        disposition = "retire_exact_centroid_candidate_retain_as_noneligible_control"
    elif producer and challenge:
        evidence_classification = "candidate_pattern_nonconfirmatory"
        scientific_interpretation = NONCONFIRMATORY_CANDIDATE
        disposition = "require_separately_authored_independent_verification_before_any_claim"
    else:
        evidence_classification = "mixed_nonconfirmatory"
        scientific_interpretation = NONCONFIRMATORY_MIXED_RESULT
        disposition = "do_not_freeze_candidate_without_new_preregistered_redesign"
    return {
        "evidence_classification": evidence_classification,
        "scientific_interpretation": scientific_interpretation,
        "producer_candidate_gate_passed": producer,
        "challenge_candidate_gate_passed": challenge,
        "independent_structural_artifact_verification_complete": raw_replay_complete,
        "independent_scientific_generation": False,
        "exact_centroid_design_eligible_for_future_freeze": False,
        "disposition": disposition,
    }


def classify_frozen_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:

    validate_frozen_aggregate(value)
    decision = _require_mapping(value["decision"], "frozen D1 decision")
    return _classification_from_gates(
        bool(decision["producer_all_frozen_criteria_passed"]),
        bool(decision["challenge_all_frozen_criteria_passed"]),
        raw_replay_complete=False,
    )


def verify_frozen_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:

    classification = classify_frozen_aggregate(value)
    phases = _require_mapping(value["phases"], "frozen D1 phases")
    source = {
        "schema": value["schema"],
        "program_id": value["program_id"],
        "result_sha256": value["result_sha256"],
        "canonical_payload_sha256": canonical_sha256(value),
        "frozen_variant_id": FROZEN_VARIANT_ID,
        "producer_conditions": dict(
            _require_mapping(
                _require_mapping(phases["producer"], "producer phase")["conditions"],
                "producer conditions",
            )
        ),
        "challenge_conditions": dict(
            _require_mapping(
                _require_mapping(phases["challenge"], "challenge phase")["conditions"],
                "challenge conditions",
            )
        ),
    }
    core = {
        "schema": VERIFICATION_SCHEMA,
        "verifier_id": VERIFIER_ID,
        "claim_scope": CLAIM_SCOPE,
        "source": source,
        "checks": {
            "aggregate_field_inventory_exact": True,
            "aggregate_self_seal_valid": True,
            "frozen_centroid_identity_exact": True,
            "aggregate_rung_index_inventory_exact": True,
            "phase_summary_arithmetic_consistent": True,
            "frozen_gates_independently_recomputed": True,
            "nonpromotion_boundary_preserved": True,
        },
        "classification": classification,
        "verification_complete": True,
        "problems": [],
        "independent_scientific_generation": False,
        "ready_for_confirmatory_claim": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "verification_sha256")


def validate_verification(value: Mapping[str, Any]) -> None:

    verification = _require_mapping(value, "frozen D1 verification")
    _require_exact_fields(verification, _VERIFICATION_FIELDS, "frozen D1 verification")
    core = {key: item for key, item in verification.items() if key != "verification_sha256"}
    if not _is_sha256(verification["verification_sha256"]) or verification[
        "verification_sha256"
    ] != canonical_sha256(core):
        raise ValueError("frozen D1 verification self-seal is invalid")
    source = _require_mapping(verification["source"], "frozen D1 verification source")
    _require_exact_fields(
        source,
        frozenset(
            {
                "schema",
                "program_id",
                "result_sha256",
                "canonical_payload_sha256",
                "frozen_variant_id",
                "producer_conditions",
                "challenge_conditions",
            }
        ),
        "frozen D1 verification source",
    )
    if (
        source["schema"] != AGGREGATE_SCHEMA
        or source["program_id"] != PROGRAM_ID
        or source["frozen_variant_id"] != FROZEN_VARIANT_ID
        or not _is_sha256(source["result_sha256"])
        or not _is_sha256(source["canonical_payload_sha256"])
    ):
        raise ValueError("frozen D1 verification source binding drifted")
    for phase in PHASES:
        conditions = _require_mapping(
            source[f"{phase}_conditions"],
            f"frozen D1 verification {phase} conditions",
        )
        _require_exact_fields(
            conditions,
            frozenset(CONDITION_FIELDS),
            f"frozen D1 verification {phase} conditions",
        )
        if any(not isinstance(conditions[field], bool) for field in CONDITION_FIELDS):
            raise ValueError(f"frozen D1 verification {phase} condition type drifted")

    checks = _require_mapping(verification["checks"], "frozen D1 verification checks")
    expected_checks = {
        "aggregate_field_inventory_exact": True,
        "aggregate_self_seal_valid": True,
        "frozen_centroid_identity_exact": True,
        "aggregate_rung_index_inventory_exact": True,
        "phase_summary_arithmetic_consistent": True,
        "frozen_gates_independently_recomputed": True,
        "nonpromotion_boundary_preserved": True,
    }
    if not _canonically_equal(checks, expected_checks):
        raise ValueError("frozen D1 verification checks drifted")

    producer = all(bool(item) for item in source["producer_conditions"].values())
    challenge = all(bool(item) for item in source["challenge_conditions"].values())
    expected_classification = _classification_from_gates(
        producer,
        challenge,
        raw_replay_complete=False,
    )
    if not _canonically_equal(verification["classification"], expected_classification):
        raise ValueError("frozen D1 verification classification drifted")
    if (
        verification["schema"] != VERIFICATION_SCHEMA
        or verification["verifier_id"] != VERIFIER_ID
        or verification["claim_scope"] != CLAIM_SCOPE
        or verification["verification_complete"] is not True
        or verification["problems"] != []
        or verification["independent_scientific_generation"] is not False
        or verification["ready_for_confirmatory_claim"] is not False
        or verification["activation_allowed"] is not False
        or verification["scientific_promotion"] is not False
    ):
        raise ValueError("frozen D1 verification identity or safety boundary drifted")


class _PhaseAccumulator:
    def __init__(self) -> None:
        self.overall: dict[str, array[float]] = {arm: array("d") for arm in OVERALL_ARMS}
        self.differences: dict[str, array[float]] = {control: array("d") for control in DIFFERENCE_CONTROLS}
        self.favorable_seed_count = {control: 0 for control in STATIC_CONTROLS}
        self.seed_count = 0
        self.rung_count = 0
        self.train_seed_count = 0
        self.heldout_seed_count = 0
        self.cell_count = 0

    def add(
        self,
        receipt: Mapping[str, Any],
        expected_config: Mapping[str, Any],
        *,
        label: str,
        visible_inputs: Sequence[str],
    ) -> None:
        cells = receipt.get("cells")
        if not isinstance(cells, list) or len(cells) != CELLS_PER_RUNG:
            raise ValueError(f"{label} cell inventory drifted")
        expected_pairs = {
            (seed, difficulty)
            for seed in range(
                int(expected_config["heldout_seed_start"]),
                int(expected_config["heldout_seed_start"]) + int(expected_config["heldout_seed_count"]),
            )
            for difficulty in expected_config["difficulty_indices"]
        }
        observed_pairs: set[tuple[int, int]] = set()
        seed_margins: dict[int, dict[str, list[float]]] = {}
        for cell_index, cell_value in enumerate(cells):
            cell = _require_mapping(cell_value, f"{label} cell {cell_index}")
            _require_exact_fields(
                cell,
                frozenset(
                    {
                        "seed",
                        "effective_seed",
                        "difficulty_index",
                        "dataset_sha256",
                        "observation_count",
                        "variant_accuracy",
                        "control_accuracy",
                        "selected_actor_counts",
                        "router_received_fields",
                        "heldout_sensitive_payloads_emitted",
                    }
                ),
                f"{label} cell {cell_index}",
            )
            seed = cell["seed"]
            difficulty = cell["difficulty_index"]
            if (
                not _is_integer(seed)
                or not _is_integer(cell["effective_seed"])
                or not _is_integer(difficulty)
                or cell["effective_seed"] != seed + difficulty * 1_000_003
                or (seed, difficulty) not in expected_pairs
                or (seed, difficulty) in observed_pairs
                or not _is_sha256(cell["dataset_sha256"])
                or not _is_integer(cell["observation_count"])
                or cell["observation_count"]
                != int(_require_mapping(expected_config["dataset"], "D1 dataset")["n_test"])
                or cell["router_received_fields"] != list(visible_inputs)
                or cell["heldout_sensitive_payloads_emitted"] is not False
            ):
                raise ValueError(f"{label} cell {cell_index} identity or contract drifted")
            observed_pairs.add((seed, difficulty))
            variants = _require_mapping(
                cell["variant_accuracy"],
                f"{label} cell {cell_index} variant accuracy",
            )
            controls = _require_mapping(
                cell["control_accuracy"],
                f"{label} cell {cell_index} control accuracy",
            )
            _require_exact_fields(
                variants,
                frozenset({FROZEN_VARIANT_ID}),
                f"{label} cell {cell_index} variant accuracy",
            )
            _require_exact_fields(
                controls,
                frozenset(arm for arm in OVERALL_ARMS if arm != "learned_dispatch"),
                f"{label} cell {cell_index} control accuracy",
            )
            learned = variants[FROZEN_VARIANT_ID]
            if not _is_number(learned) or not 0.0 <= float(learned) <= 1.0:
                raise ValueError(f"{label} cell {cell_index} learned accuracy drifted")
            learned_value = float(learned)
            self.overall["learned_dispatch"].append(learned_value)
            for control in controls:
                control_value = controls[control]
                if not _is_number(control_value) or not 0.0 <= float(control_value) <= 1.0:
                    raise ValueError(f"{label} cell {cell_index} {control} accuracy drifted")
                self.overall[str(control)].append(float(control_value))
            for control in DIFFERENCE_CONTROLS:
                margin = learned_value - float(controls[control])
                self.differences[control].append(margin)
                if control in STATIC_CONTROLS:
                    seed_margins.setdefault(seed, {}).setdefault(control, []).append(margin)
            selected = _require_mapping(
                cell["selected_actor_counts"],
                f"{label} cell {cell_index} selected actor counts",
            )
            _require_exact_fields(
                selected,
                frozenset({FROZEN_VARIANT_ID}),
                f"{label} cell {cell_index} selected actor counts",
            )
            actor_counts = _require_mapping(
                selected[FROZEN_VARIANT_ID],
                f"{label} cell {cell_index} actor counts",
            )
            if (
                not actor_counts
                or any(not _is_integer(count) or count < 0 for count in actor_counts.values())
                or sum(int(count) for count in actor_counts.values()) != cell["observation_count"]
            ):
                raise ValueError(f"{label} cell {cell_index} actor accounting drifted")
        if observed_pairs != expected_pairs:
            raise ValueError(f"{label} seed/difficulty inventory drifted")
        for seed in sorted(seed_margins):
            margins = seed_margins[seed]
            if set(margins) != set(STATIC_CONTROLS) or any(
                len(margins[control]) != DIFFICULTY_COUNT for control in STATIC_CONTROLS
            ):
                raise ValueError(f"{label} favorable-seed inventory drifted")
            for control in STATIC_CONTROLS:
                if statistics.fmean(margins[control]) > 0.0:
                    self.favorable_seed_count[control] += 1
        self.seed_count += len(seed_margins)
        self.rung_count += 1
        self.train_seed_count += int(expected_config["train_seed_count"])
        self.heldout_seed_count += int(expected_config["heldout_seed_count"])
        self.cell_count += len(cells)

    def summary(self) -> dict[str, Any]:
        if (
            self.rung_count != RUNGS_PER_PHASE
            or self.seed_count != PHASE_SEED_COUNT
            or self.cell_count != PHASE_CELL_COUNT
        ):
            raise ValueError("frozen D1 replay phase is incomplete")
        overall = {arm: _mean_ci_from_values(values) for arm, values in self.overall.items()}
        differences = {control: _mean_ci_from_values(values) for control, values in self.differences.items()}
        favorable = {
            control: self.favorable_seed_count[control] / self.seed_count for control in STATIC_CONTROLS
        }
        context_gap = -float(differences["context_route_nonpromotable"]["mean"])
        conditions = {
            "static_margin_gate": all(
                float(differences[control]["mean"])
                >= float(CRITERIA["minimum_mean_advantage_over_each_static_control"])
                and float(differences[control]["lo"])
                > float(CRITERIA["comparison_interval_lower_bound_must_exceed"])
                for control in STATIC_CONTROLS
            ),
            "favorable_seed_fraction_gate": all(
                favorable[control] >= float(CRITERIA["minimum_favorable_seed_fraction"])
                for control in STATIC_CONTROLS
            ),
            "context_route_gap_gate": context_gap
            <= float(CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"]),
            "work_saving_gate": float(CRITERIA["minimum_work_saving_vs_all_five_actors"]) <= 0.8,
        }
        return {
            "grid": {
                "rung_count": self.rung_count,
                "train_seed_count": self.train_seed_count,
                "heldout_seed_count": self.heldout_seed_count,
                "completed_cell_count": self.cell_count,
            },
            "overall": overall,
            "learned_dispatch_differences": differences,
            "favorable_seed_fraction": favorable,
            "mean_gap_below_context_route": context_gap,
            "work_saving_vs_all_five_actors": 0.8,
            "conditions": conditions,
            "all_frozen_criteria_passed": all(conditions.values()),
        }


def _mean_ci_from_values(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def _recomputed_aggregate(
    rung_rows: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> dict[str, Any]:
    repeated = bool(producer["all_frozen_criteria_passed"] and challenge["all_frozen_criteria_passed"])
    core = {
        "schema": AGGREGATE_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": AGGREGATE_CLAIM_SCOPE,
        "screen_binding": dict(SCREEN_BINDING),
        "frozen_variant": dict(FROZEN_VARIANT),
        "criteria": dict(CRITERIA),
        "rungs": [dict(row) for row in rung_rows],
        "phases": {"producer": dict(producer), "challenge": dict(challenge)},
        "grid": {
            "phase_count": 2,
            "rung_count": RUNG_COUNT,
            "train_seed_count": TOTAL_SEED_COUNT,
            "heldout_seed_count": TOTAL_SEED_COUNT,
            "completed_cell_count": TOTAL_CELL_COUNT,
        },
        "overall": challenge["overall"],
        "learned_dispatch_differences": challenge["learned_dispatch_differences"],
        "decision": {
            "producer_all_frozen_criteria_passed": producer["all_frozen_criteria_passed"],
            "challenge_all_frozen_criteria_passed": challenge["all_frozen_criteria_passed"],
            "frozen_pattern_repeated": repeated,
            "independent_verification_complete": False,
            "ready_for_confirmatory_claim": False,
            "next_action": "run_v1_m1_g1_sibling_batch",
        },
        "interpretation_limit": INTERPRETATION_LIMIT,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def _assert_recomputed_aggregate_match(
    aggregate: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    if not _canonically_equal(aggregate, recomputed):
        raise ValueError("frozen D1 aggregate does not exactly replay from its raw rung receipts")


def _replay_raw_rungs(
    aggregate: Mapping[str, Any],
    rung_root: Path,
) -> dict[str, Any]:
    from mop.studies import generation1_c3_d1_frozen_queue as queue
    from mop.studies import generation1_c3_router_redesign as producer_code

    candidate_root = Path(rung_root)
    resolved_root = candidate_root.resolve()
    if candidate_root.is_symlink() or not resolved_root.is_dir():
        raise ValueError(f"frozen D1 rung root is not a regular directory: {resolved_root}")
    aggregate_rows = aggregate["rungs"]
    if not isinstance(aggregate_rows, list):
        raise ValueError("frozen D1 aggregate rung inventory is invalid")
    inventory: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    result_hashes: set[str] = set()
    config_hashes: set[str] = set()
    router_seeds: set[int] = set()
    seed_ranges: list[tuple[int, int, str]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    accumulator = _PhaseAccumulator()
    total_bytes = 0
    for index in range(RUNG_COUNT):
        phase = "producer" if index < RUNGS_PER_PHASE else "challenge"
        local_index = index if phase == "producer" else index - RUNGS_PER_PHASE
        path = resolved_root / phase / f"rung_{local_index:03d}.json"
        receipt, raw, resolved = _read_regular_json_object(
            path,
            f"frozen D1 raw rung {index}",
        )
        expected_config = queue.rung_config(index)
        producer_code.validate_result(
            receipt,
            expected_config,
            repo_root=REPO_ROOT,
        )
        if not _canonically_equal(receipt.get("config"), expected_config):
            raise ValueError(f"frozen D1 raw rung {index} config payload drifted")
        aggregate_row = _require_mapping(
            aggregate_rows[index],
            f"frozen D1 aggregate rung {index}",
        )
        if (
            receipt.get("result_sha256") != aggregate_row["result_sha256"]
            or receipt.get("config_sha256") != aggregate_row["config_sha256"]
        ):
            raise ValueError(f"frozen D1 raw rung {index} aggregate binding drifted")
        result_hash = str(receipt["result_sha256"])
        config_hash = str(receipt["config_sha256"])
        if result_hash in result_hashes or config_hash in config_hashes:
            raise ValueError("frozen D1 raw rung result or config hash is duplicated")
        result_hashes.add(result_hash)
        config_hashes.add(config_hash)
        router_seed = int(expected_config["router_training_seed"])
        if router_seed in router_seeds:
            raise ValueError("frozen D1 raw rung router seed is duplicated")
        router_seeds.add(router_seed)
        for partition in ("train", "heldout"):
            start = int(expected_config[f"{partition}_seed_start"])
            count = int(expected_config[f"{partition}_seed_count"])
            seed_ranges.append((start, start + count, f"{index}:{partition}"))
        expected_grid = {
            "variant_count": 1,
            "train_seed_count": SEEDS_PER_RUNG,
            "heldout_seed_count": SEEDS_PER_RUNG,
            "difficulty_count": DIFFICULTY_COUNT,
            "completed_cell_count": CELLS_PER_RUNG,
            "shared_actor_evaluation": True,
        }
        if not _canonically_equal(receipt.get("grid"), expected_grid):
            raise ValueError(f"frozen D1 raw rung {index} grid drifted")
        accumulator.add(
            receipt,
            expected_config,
            label=f"frozen D1 raw rung {index}",
            visible_inputs=producer_code.VISIBLE_ROUTER_INPUTS,
        )
        replay_row = {
            "index": index,
            "phase": phase,
            "result_sha256": result_hash,
            "config_sha256": config_hash,
            "completed_cell_count": CELLS_PER_RUNG,
        }
        replay_rows.append(replay_row)
        inventory.append(
            {
                **replay_row,
                "path": _path_label(resolved),
                "file_sha256": _sha256_bytes(raw),
            }
        )
        total_bytes += len(raw)
        if index in (RUNGS_PER_PHASE - 1, RUNG_COUNT - 1):
            phase_summaries[phase] = accumulator.summary()
            accumulator = _PhaseAccumulator()
    ranges = sorted(seed_ranges)
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if previous[1] > current[0]:
            raise ValueError(f"frozen D1 raw seed partitions overlap: {previous[2]} and {current[2]}")
    recomputed = _recomputed_aggregate(
        replay_rows,
        phase_summaries["producer"],
        phase_summaries["challenge"],
    )
    _assert_recomputed_aggregate_match(aggregate, recomputed)
    return {
        "root_path": _path_label(resolved_root),
        "rung_count": RUNG_COUNT,
        "raw_file_bytes": total_bytes,
        "rung_inventory": inventory,
        "rung_inventory_sha256": canonical_sha256(inventory),
        "result_hashes_unique": len(result_hashes) == RUNG_COUNT,
        "config_hashes_unique": len(config_hashes) == RUNG_COUNT,
        "seed_partitions_disjoint": True,
        "router_training_seeds_unique": len(router_seeds) == RUNG_COUNT,
        "producer_summary_sha256": canonical_sha256(phase_summaries["producer"]),
        "challenge_summary_sha256": canonical_sha256(phase_summaries["challenge"]),
        "recomputed_result_sha256": recomputed["result_sha256"],
        "aggregate_exactly_recomputed": True,
    }


def build_byte_bound_verification(
    path: Path = DEFAULT_RESULT,
    *,
    rung_root: Path = DEFAULT_RUNG_ROOT,
) -> dict[str, Any]:

    aggregate, raw, resolved = _read_regular_aggregate(path)
    verification = verify_frozen_aggregate(aggregate)
    validate_verification(verification)
    rung_replay = _replay_raw_rungs(aggregate, rung_root)
    decision = _require_mapping(aggregate["decision"], "frozen D1 decision")
    classification = _classification_from_gates(
        bool(decision["producer_all_frozen_criteria_passed"]),
        bool(decision["challenge_all_frozen_criteria_passed"]),
        raw_replay_complete=True,
    )
    core = {
        "schema": BYTE_BOUND_VERIFICATION_SCHEMA,
        "verifier_id": VERIFIER_ID,
        "claim_scope": CLAIM_SCOPE,
        "source": {
            "path": _path_label(resolved),
            "file_sha256": _sha256_bytes(raw),
            "result_sha256": aggregate["result_sha256"],
            "canonical_payload_sha256": canonical_sha256(aggregate),
        },
        "rung_replay": rung_replay,
        "verification": verification,
        "classification": classification,
        "verification_complete": True,
        "problems": [],
        "independent_scientific_generation": False,
        "ready_for_confirmatory_claim": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "artifact_sha256")


def _validate_rung_replay_report(
    value: Any,
    source: Mapping[str, Any],
) -> None:
    report = _require_mapping(value, "byte-bound frozen D1 rung replay")
    _require_exact_fields(
        report,
        frozenset(
            {
                "root_path",
                "rung_count",
                "raw_file_bytes",
                "rung_inventory",
                "rung_inventory_sha256",
                "result_hashes_unique",
                "config_hashes_unique",
                "seed_partitions_disjoint",
                "router_training_seeds_unique",
                "producer_summary_sha256",
                "challenge_summary_sha256",
                "recomputed_result_sha256",
                "aggregate_exactly_recomputed",
            }
        ),
        "byte-bound frozen D1 rung replay",
    )
    inventory = report["rung_inventory"]
    if (
        not isinstance(report["root_path"], str)
        or not report["root_path"]
        or report["rung_count"] != RUNG_COUNT
        or not _is_integer(report["raw_file_bytes"])
        or report["raw_file_bytes"] <= 0
        or not isinstance(inventory, list)
        or len(inventory) != RUNG_COUNT
        or report["rung_inventory_sha256"] != canonical_sha256(inventory)
        or report["result_hashes_unique"] is not True
        or report["config_hashes_unique"] is not True
        or report["seed_partitions_disjoint"] is not True
        or report["router_training_seeds_unique"] is not True
        or not _is_sha256(report["producer_summary_sha256"])
        or not _is_sha256(report["challenge_summary_sha256"])
        or report["recomputed_result_sha256"] != source["result_sha256"]
        or report["aggregate_exactly_recomputed"] is not True
    ):
        raise ValueError("byte-bound frozen D1 rung replay summary drifted")
    result_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for index, row_value in enumerate(inventory):
        row = _require_mapping(
            row_value,
            f"byte-bound frozen D1 rung replay row {index}",
        )
        _require_exact_fields(
            row,
            frozenset(
                {
                    "index",
                    "phase",
                    "result_sha256",
                    "config_sha256",
                    "completed_cell_count",
                    "path",
                    "file_sha256",
                }
            ),
            f"byte-bound frozen D1 rung replay row {index}",
        )
        expected_phase = "producer" if index < RUNGS_PER_PHASE else "challenge"
        if (
            not _is_integer(row["index"])
            or row["index"] != index
            or row["phase"] != expected_phase
            or not _is_sha256(row["result_sha256"])
            or not _is_sha256(row["config_sha256"])
            or not _is_sha256(row["file_sha256"])
            or not isinstance(row["path"], str)
            or not row["path"]
            or row["completed_cell_count"] != CELLS_PER_RUNG
        ):
            raise ValueError(f"byte-bound frozen D1 rung replay row {index} drifted")
        result_hashes.add(str(row["result_sha256"]))
        config_hashes.add(str(row["config_sha256"]))
    if len(result_hashes) != RUNG_COUNT or len(config_hashes) != RUNG_COUNT:
        raise ValueError("byte-bound frozen D1 rung replay hash inventory is duplicated")


def validate_byte_bound_verification(
    value: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    rung_root: Path | None = None,
) -> None:

    artifact = _require_mapping(value, "byte-bound frozen D1 verification")
    _require_exact_fields(
        artifact,
        _BYTE_BOUND_FIELDS,
        "byte-bound frozen D1 verification",
    )
    core = {key: item for key, item in artifact.items() if key != "artifact_sha256"}
    if not _is_sha256(artifact["artifact_sha256"]) or artifact["artifact_sha256"] != canonical_sha256(core):
        raise ValueError("byte-bound frozen D1 verification self-seal is invalid")
    verification = _require_mapping(
        artifact["verification"],
        "byte-bound frozen D1 structural verification",
    )
    validate_verification(verification)
    source = _require_mapping(
        artifact["source"],
        "byte-bound frozen D1 source",
    )
    _require_exact_fields(
        source,
        frozenset(
            {
                "path",
                "file_sha256",
                "result_sha256",
                "canonical_payload_sha256",
            }
        ),
        "byte-bound frozen D1 source",
    )
    verification_source = _require_mapping(
        verification["source"],
        "frozen D1 structural verification source",
    )
    if (
        not isinstance(source["path"], str)
        or not source["path"]
        or not _is_sha256(source["file_sha256"])
        or source["result_sha256"] != verification_source["result_sha256"]
        or source["canonical_payload_sha256"] != verification_source["canonical_payload_sha256"]
    ):
        raise ValueError("byte-bound frozen D1 source binding drifted")
    _validate_rung_replay_report(artifact["rung_replay"], source)
    producer_conditions = _require_mapping(
        verification_source["producer_conditions"],
        "byte-bound frozen D1 producer conditions",
    )
    challenge_conditions = _require_mapping(
        verification_source["challenge_conditions"],
        "byte-bound frozen D1 challenge conditions",
    )
    expected_classification = _classification_from_gates(
        all(bool(value) for value in producer_conditions.values()),
        all(bool(value) for value in challenge_conditions.values()),
        raw_replay_complete=True,
    )
    if not _canonically_equal(artifact["classification"], expected_classification):
        raise ValueError("byte-bound frozen D1 final classification drifted")
    if (
        artifact["schema"] != BYTE_BOUND_VERIFICATION_SCHEMA
        or artifact["verifier_id"] != VERIFIER_ID
        or artifact["claim_scope"] != CLAIM_SCOPE
        or artifact["verification_complete"] is not True
        or artifact["problems"] != []
        or artifact["independent_scientific_generation"] is not False
        or artifact["ready_for_confirmatory_claim"] is not False
        or artifact["activation_allowed"] is not False
        or artifact["scientific_promotion"] is not False
    ):
        raise ValueError("byte-bound frozen D1 verification safety boundary drifted")
    if (source_path is None) is not (rung_root is None):
        raise ValueError("live byte-bound validation requires both aggregate and rung-root paths")
    if source_path is not None and rung_root is not None:
        aggregate, raw, resolved = _read_regular_aggregate(source_path)
        replayed = verify_frozen_aggregate(aggregate)
        replayed_rungs = _replay_raw_rungs(aggregate, rung_root)
        expected_source = {
            "path": _path_label(resolved),
            "file_sha256": _sha256_bytes(raw),
            "result_sha256": aggregate["result_sha256"],
            "canonical_payload_sha256": canonical_sha256(aggregate),
        }
        if (
            not _canonically_equal(source, expected_source)
            or not _canonically_equal(verification, replayed)
            or not _canonically_equal(artifact["rung_replay"], replayed_rungs)
        ):
            raise ValueError("byte-bound frozen D1 source replay drifted")


def materialize_frozen_verification(
    source_path: Path = DEFAULT_RESULT,
    output_path: Path = DEFAULT_VERIFICATION,
    *,
    rung_root: Path = DEFAULT_RUNG_ROOT,
) -> dict[str, Any]:

    artifact = build_byte_bound_verification(source_path, rung_root=rung_root)
    validate_byte_bound_verification(artifact)
    target = Path(output_path).resolve()
    if target == Path(source_path).resolve():
        raise ValueError("frozen D1 verification output cannot replace its source")
    if Path(output_path).is_symlink():
        raise ValueError("frozen D1 verification output cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(artifact) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    written = target.read_bytes()
    if written != canonical_bytes(artifact) + b"\n":
        raise ValueError("frozen D1 verification materialization drifted")
    persisted = json.loads(written)
    if not isinstance(persisted, dict) or not _canonically_equal(persisted, artifact):
        raise ValueError("frozen D1 verification persisted payload drifted")
    validate_byte_bound_verification(persisted)
    aggregate, raw, resolved = _read_regular_aggregate(source_path)
    expected_source = {
        "path": _path_label(resolved),
        "file_sha256": _sha256_bytes(raw),
        "result_sha256": aggregate["result_sha256"],
        "canonical_payload_sha256": canonical_sha256(aggregate),
    }
    if not _canonically_equal(artifact["source"], expected_source):
        raise ValueError("frozen D1 aggregate changed during verification materialization")
    return artifact


def load_and_verify_frozen_aggregate(path: Path = DEFAULT_RESULT) -> dict[str, Any]:

    value, _, _ = _read_regular_aggregate(path)
    verification = verify_frozen_aggregate(value)
    validate_verification(verification)
    return verification


__all__ = [
    "AGGREGATE_SCHEMA",
    "BYTE_BOUND_VERIFICATION_SCHEMA",
    "CLAIM_SCOPE",
    "CRITERIA",
    "DEFAULT_RESULT",
    "DEFAULT_RUNG_ROOT",
    "DEFAULT_VERIFICATION",
    "FROZEN_VARIANT",
    "FROZEN_VARIANT_ID",
    "NONCONFIRMATORY_GATE_FAILURE",
    "NULL_SAFE_PRUNE",
    "PROGRAM_ID",
    "STRUCTURAL_ONLY",
    "STRUCTURAL_ONLY_INTERPRETATION",
    "VERIFICATION_SCHEMA",
    "VERIFIER_ID",
    "build_byte_bound_verification",
    "canonical_bytes",
    "canonical_sha256",
    "classify_frozen_aggregate",
    "load_and_verify_frozen_aggregate",
    "materialize_frozen_verification",
    "validate_byte_bound_verification",
    "validate_frozen_aggregate",
    "validate_verification",
    "verify_frozen_aggregate",
]
