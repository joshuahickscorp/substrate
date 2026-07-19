"""Unit tests for the reusable Mechanism Admission Battery (Bundle A).

These tests exercise each of the eight clauses on a clearly-passing and a clearly-failing synthetic case,
run the whole battery on an all-pass and a one-fail case, run the three read-only audit adapters against the
sealed STARSS23 beds (asserting at least one failed clause per bed and that the sealed verdict bytes are not
mutated), and assert the redesign route forbids the four default routes.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mop.admission import audit_adapters
from mop.admission.audit_adapters import audit_counting, audit_doa, audit_onset
from mop.admission.battery import (
    clause_architecture_independence,
    clause_control_behavior,
    clause_design_adequacy,
    clause_group_disjoint_validity,
    clause_incremental_value,
    clause_oracle_budget_headroom,
    clause_what_absolute_sufficiency,
    clause_when_decodability,
    run_battery,
)
from mop.admission.prereg import (
    MECHANISM_ADMISSION_PREREG,
    MIN_INDEPENDENT_UNITS,
    build_prereg,
    verify_prereg_seal,
)
from mop.admission.redesign_route import (
    FORBIDDEN_DEFAULT_ROUTES,
    relational_temporal_redesign_route,
)

PREREG = MECHANISM_ADMISSION_PREREG


def _passing_inputs(seed: int = 0) -> dict[str, Any]:
    """A synthetic case in which all eight clauses clearly pass."""

    rng = np.random.default_rng(seed)
    n_groups, per_group = 16, 40
    n = n_groups * per_group
    group_ids = np.repeat(np.arange(n_groups), per_group)
    z = rng.random(n)  # latent nonnegative marginal value of recomputation
    when = np.column_stack([z + 0.02 * rng.standard_normal(n), z**2, np.sin(3.0 * z)])
    baseline = rng.standard_normal((n, 3))  # uninformative energy/rate/change proxies
    what_true = 2.0 * z
    what_pred = {
        "candidate": what_true + 0.01 * rng.standard_normal(n),
        "constant": np.full(n, float(what_true.mean())),
        "empirical_prior": np.array([what_true[group_ids == gid].mean() for gid in group_ids]),
        "frozen_random": rng.standard_normal(n) * 2.0,
        "handcrafted_control": rng.random(n) * 2.0,
        "rate_matched_random": rng.random(n) * 2.0,
    }
    return {
        "what_true": what_true,
        "what_pred": what_pred,
        "when_features": when,
        "baseline_heuristics": baseline,
        "recompute_value": z,
        "labels": (z > 0.5).astype(float),
        "group_ids": group_ids,
        "budget": 0.1,
        "design": {
            "sesoi": 0.05,
            "power": 0.9,
            "multiplicity_correction": "holm-bonferroni",
            "stop_rule": "stop for futility when conditional power at the SESOI drops below 0.2",
        },
        "noisy_tv_firing_rate": 0.02,
        "noisy_tv_base_rate": 0.13,
        "shuffled_target_score": 0.5,
        "wrong_time_score": 0.5,
        "chance_level": 0.5,
        "primary_control": "rate_matched_random",
        "architecture_favorable": [True, True],
    }


def _break_what(d: dict[str, Any]) -> None:
    d["what_pred"]["candidate"] = d["what_pred"]["constant"].copy()


def _break_oracle(d: dict[str, Any]) -> None:
    d["recompute_value"] = np.full_like(d["recompute_value"], 0.5)


def _break_decodability(d: dict[str, Any]) -> None:
    d["when_features"] = np.random.default_rng(9).standard_normal(d["when_features"].shape)


def _break_incremental(d: dict[str, Any]) -> None:
    d["when_features"] = d["baseline_heuristics"].copy()


def _break_group_validity(d: dict[str, Any]) -> None:
    d["what_pred"]["candidate"] = d["what_pred"]["rate_matched_random"].copy()


def _break_design(d: dict[str, Any]) -> None:
    d["design"]["power"] = 0.3


def _break_control(d: dict[str, Any]) -> None:
    d["noisy_tv_firing_rate"] = 0.9


def _break_architecture(d: dict[str, Any]) -> None:
    d["architecture_favorable"] = [True, False]


_CLAUSE_CASES: list[tuple[str, Callable[..., dict], Callable[[dict], None]]] = [
    ("what_absolute_sufficiency", clause_what_absolute_sufficiency, _break_what),
    ("oracle_budget_headroom", clause_oracle_budget_headroom, _break_oracle),
    ("when_decodability", clause_when_decodability, _break_decodability),
    ("incremental_value", clause_incremental_value, _break_incremental),
    ("group_disjoint_validity", clause_group_disjoint_validity, _break_group_validity),
    ("design_adequacy", clause_design_adequacy, _break_design),
    ("control_behavior", clause_control_behavior, _break_control),
    ("architecture_independence", clause_architecture_independence, _break_architecture),
]


@pytest.mark.parametrize("clause_id,clause_fn,breaker", _CLAUSE_CASES)
def test_clause_pass_and_fail(clause_id, clause_fn, breaker) -> None:
    passing = _passing_inputs()
    result_pass = clause_fn(passing, PREREG)
    assert result_pass["clause"] == clause_id
    assert result_pass["passed"] is True, result_pass["evidence"]

    failing = copy.deepcopy(passing)
    breaker(failing)
    result_fail = clause_fn(failing, PREREG)
    assert result_fail["passed"] is False, result_fail["evidence"]


def test_clause_missing_input_fails_closed() -> None:
    # A clause given none of its required inputs must fail, never crash.
    for _, clause_fn, _ in _CLAUSE_CASES:
        result = clause_fn({}, PREREG)
        assert result["passed"] is False


def test_run_battery_all_pass() -> None:
    result = run_battery(_passing_inputs(), PREREG)
    assert result["admitted"] is True
    assert result["n_passed"] == result["n_clauses"] == 8
    assert all(entry["passed"] for entry in result["clauses"].values())
    # A sealed-artifact-style result must carry the three hardcoded false flags.
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False
    assert result["independent_scientific_confirmation"] is False


def test_run_battery_one_fail() -> None:
    # architecture_independence has no cross-dependency on the array-based clauses, so breaking it
    # isolates a single clause failure.
    inputs = _passing_inputs()
    _break_architecture(inputs)
    result = run_battery(inputs, PREREG)
    assert result["admitted"] is False
    assert result["n_passed"] == 7
    assert result["clauses"]["architecture_independence"]["passed"] is False
    still_passing = [k for k, v in result["clauses"].items() if v["passed"]]
    assert "architecture_independence" not in still_passing
    assert len(still_passing) == 7


def test_run_battery_defaults_to_module_prereg() -> None:
    result = run_battery(_passing_inputs())
    assert result["prereg_sha256"] == PREREG["seal"]["sha256"]


def test_prereg_is_self_sealed_and_tamper_evident() -> None:
    assert verify_prereg_seal(PREREG) is True
    assert verify_prereg_seal(build_prereg()) is True
    tampered = copy.deepcopy(PREREG)
    tampered["constants"]["sesoi_default"] = 0.5
    assert verify_prereg_seal(tampered) is False
    # the frozen contract has exactly eight clauses and the three hardcoded false flags
    assert len(PREREG["clauses"]) == 8
    assert PREREG["activation_allowed"] is False
    assert PREREG["scientific_promotion"] is False
    assert PREREG["independent_scientific_confirmation"] is False


def _proof_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "adapter,proof_files,expected_clauses",
    [
        (audit_onset, ["STARSS23_ESCS_BED.json"], {"when_decodability", "incremental_value"}),
        (
            audit_counting,
            ["STARSS23_COUNTING_BED.json", "STARSS23_COUNTING_REPRO_gate_arch.json"],
            {"architecture_independence"},
        ),
        (
            audit_doa,
            ["STARSS23_DOA_BED.json"],
            {"what_absolute_sufficiency", "oracle_budget_headroom"},
        ),
    ],
)
def test_audit_adapter_classifies_without_mutating(adapter, proof_files, expected_clauses) -> None:
    proof_root = audit_adapters._DEFAULT_PROOF_ROOT
    before = {f: _proof_sha256(proof_root / f) for f in proof_files}

    result = adapter()

    # the sealed proof bytes are untouched: read-only, no verdict overwrite
    after = {f: _proof_sha256(proof_root / f) for f in proof_files}
    assert before == after

    # the sealed verdict is echoed verbatim, not reinterpreted
    import json

    on_disk_verdict = json.loads((proof_root / proof_files[0]).read_text(encoding="utf-8"))["verdict"]
    assert result["sealed_verdict"] == on_disk_verdict
    assert result["sealed_verdict_preserved"] is True
    assert result["read_only"] is True

    # at least one failed clause is classified, and it includes the expected clauses for this bed
    failed = set(result["design_classification"]["failed_clauses"])
    assert len(failed) >= 1
    assert expected_clauses <= failed
    for clause_id in failed:
        assert result["design_classification"]["per_clause_reason"].get(clause_id)


def test_min_independent_units_gate_is_enforced() -> None:
    # a design with too few grouped units must fail design_adequacy and group_disjoint_validity
    inputs = _passing_inputs()
    keep = inputs["group_ids"] < (MIN_INDEPENDENT_UNITS - 4)
    for key in ("what_true", "when_features", "baseline_heuristics", "recompute_value", "labels"):
        arr = np.asarray(inputs[key])
        inputs[key] = arr[keep]
    inputs["group_ids"] = np.asarray(inputs["group_ids"])[keep]
    inputs["what_pred"] = {k: np.asarray(v)[keep] for k, v in inputs["what_pred"].items()}
    assert clause_design_adequacy(inputs, PREREG)["passed"] is False
    assert clause_group_disjoint_validity(inputs, PREREG)["passed"] is False


def test_redesign_route_forbids_four_default_routes() -> None:
    route = relational_temporal_redesign_route()
    forbidden = route["forbidden_default_routes"]
    for name in ("try another MLP", "more seeds", "weaker thresholds", "another spacing regularizer"):
        assert name in forbidden
    assert set(FORBIDDEN_DEFAULT_ROUTES) == set(forbidden)
    assert route["redesign_target"] == "relational-temporal representation"
    assert route["activation_allowed"] is False


def test_redesign_route_engages_on_failing_battery() -> None:
    inputs = _passing_inputs()
    _break_decodability(inputs)
    result = run_battery(inputs, PREREG)
    route = relational_temporal_redesign_route(result)
    assert route["engaged"] is True
    assert "when_decodability" in route["responding_to_failed_clauses"]
    assert "incremental_value" in route["responding_to_failed_clauses"]

    admitted_route = relational_temporal_redesign_route(run_battery(_passing_inputs(), PREREG))
    assert admitted_route["engaged"] is False
    # even when not engaged, the four forbidden routes are still declared
    for name in FORBIDDEN_DEFAULT_ROUTES:
        assert name in admitted_route["forbidden_default_routes"]
