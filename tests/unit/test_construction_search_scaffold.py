from __future__ import annotations

import pytest

from mop.mechanisms.construction_search_scaffold import (
    ALL_CONTROLS,
    CLAIM_SCOPE,
    CONSTRUCTION_SEARCH_SCHEMA,
    ORACLE_HEADROOM_CONTROL,
    PRIOR_NULL,
    REQUIRED_CONTROLS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    ConstructionControl,
    ConstructionControlSet,
    ConstructionSearchActivationGate,
    ConstructionSearchActivationRefusal,
    ConstructionSearchContract,
    ConstructionSearchRefusal,
    SealedObjective,
    SearchBudget,
    SearchTrace,
    SearchValueVerdict,
    build_default_contract,
    build_default_control_set,
    run_construction_search,
    seal_objective,
    verdict_from_trace,
)


def test_search_budget_must_be_non_vacuous() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="non-vacuous"):
        SearchBudget(candidate_evaluations=0, objective_queries=1, wall_proxy_units=1, memory_bytes=1)


def test_search_budget_digest_is_stable() -> None:
    budget = SearchBudget(
        candidate_evaluations=16, objective_queries=16, wall_proxy_units=8, memory_bytes=1024
    )
    twin = SearchBudget(candidate_evaluations=16, objective_queries=16, wall_proxy_units=8, memory_bytes=1024)
    assert budget.digest() == twin.digest()
    assert len(budget.digest()) == 64


def test_search_budget_rejects_widened_claim_scope() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="claim scope"):
        SearchBudget(
            candidate_evaluations=1,
            objective_queries=1,
            wall_proxy_units=1,
            memory_bytes=1,
            claim_scope="a capability was demonstrated",
        )


def test_seal_objective_roundtrips() -> None:
    objective = seal_objective(
        objective_id="obj.x", task_ids=("task.a", "task.b"), num_members=6, size_penalty=0.05
    )
    assert objective.objective_sha256 == objective._core_digest()
    assert (
        objective.digest()
        == SealedObjective(
            **{
                "objective_id": objective.objective_id,
                "task_ids": objective.task_ids,
                "num_members": objective.num_members,
                "size_penalty": objective.size_penalty,
                "objective_sha256": objective.objective_sha256,
            }
        ).digest()
    )


def test_sealed_objective_detects_tampering() -> None:
    objective = seal_objective(
        objective_id="obj.x", task_ids=("task.a", "task.b"), num_members=6, size_penalty=0.05
    )
    with pytest.raises(ConstructionSearchRefusal, match="seal digest"):
        SealedObjective(
            objective_id="obj.x",
            task_ids=("task.a", "task.b"),
            num_members=6,
            size_penalty=0.99,  # changed after the seal was computed
            objective_sha256=objective.objective_sha256,
        )


def test_sealed_objective_requires_sealed_before_search() -> None:
    objective = seal_objective(
        objective_id="obj.x", task_ids=("task.a", "task.b"), num_members=6, size_penalty=0.05
    )
    with pytest.raises(ConstructionSearchRefusal, match="sealed before search"):
        SealedObjective(
            objective_id="obj.x",
            task_ids=("task.a", "task.b"),
            num_members=6,
            size_penalty=0.05,
            objective_sha256=objective.objective_sha256,
            sealed_before_search=False,
        )


def test_sealed_objective_requires_multiple_tasks() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="at least two tasks"):
        seal_objective(objective_id="obj.x", task_ids=("task.a",), num_members=6, size_penalty=0.05)


def test_default_control_set_covers_every_family_in_order() -> None:
    control_set = build_default_control_set()
    assert tuple(c.family for c in control_set.controls) == ALL_CONTROLS
    assert control_set.cheap_control_families() == REQUIRED_CONTROLS


def test_control_set_fails_closed_on_membership_or_order_drift() -> None:
    controls = build_default_control_set().controls
    with pytest.raises(ConstructionSearchRefusal, match="membership or order drift"):
        ConstructionControlSet(schema=CONSTRUCTION_SEARCH_SCHEMA, controls=tuple(reversed(controls)))


def test_control_set_needs_exactly_one_headroom() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="membership or order drift"):
        ConstructionControlSet(
            schema=CONSTRUCTION_SEARCH_SCHEMA,
            controls=(
                ConstructionControl(id="ctrl.a", family="no-search", rationale="r", is_oracle_headroom=False),
            ),
        )


def test_control_headroom_flag_must_match_family() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="oracle-headroom flag"):
        ConstructionControl(id="ctrl.a", family="no-search", rationale="r", is_oracle_headroom=True)


def test_control_rejects_unknown_family() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="unsupported control family"):
        ConstructionControl(
            id="ctrl.a", family="simulated-annealing", rationale="r", is_oracle_headroom=False
        )


def test_default_contract_digest_is_stable() -> None:
    assert build_default_contract().digest() == build_default_contract().digest()
    assert len(build_default_contract().digest()) == 64


def test_contract_requires_search_budget_charged() -> None:
    contract = build_default_contract()
    with pytest.raises(ConstructionSearchRefusal, match="charge the search cost"):
        ConstructionSearchContract(
            objective=contract.objective,
            budget=contract.budget,
            controls=contract.controls,
            search_budget_charged=False,
        )


def test_contract_requires_matched_cost() -> None:
    contract = build_default_contract()
    with pytest.raises(ConstructionSearchRefusal, match="matched full-system cost"):
        ConstructionSearchContract(
            objective=contract.objective,
            budget=contract.budget,
            controls=contract.controls,
            matched_cost_required=False,
        )


def test_contract_rejects_widened_claim_scope() -> None:
    contract = build_default_contract()
    with pytest.raises(ConstructionSearchRefusal, match="claim scope"):
        ConstructionSearchContract(
            objective=contract.objective,
            budget=contract.budget,
            controls=contract.controls,
            claim_scope="capability shown",
        )


def test_verdict_may_claim_only_when_net_positive() -> None:
    verdict = SearchValueVerdict(
        gross_improvement=0.5,
        charged_search_cost=0.2,
        oracle_headroom_gap=0.1,
        claims_improvement=True,
    )
    assert verdict.net_improvement == pytest.approx(0.3)
    assert verdict.claims_improvement


def test_verdict_refuses_overclaim_when_cost_erases_gain() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="prior null holds"):
        SearchValueVerdict(
            gross_improvement=0.2,
            charged_search_cost=0.2,  # net is zero; the null must stand
            oracle_headroom_gap=0.0,
            claims_improvement=True,
        )


def test_verdict_requires_matched_cost_charged() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="matched search cost is charged"):
        SearchValueVerdict(
            gross_improvement=1.0,
            charged_search_cost=0.0,
            oracle_headroom_gap=0.0,
            claims_improvement=True,
            matched_cost_charged=False,
        )


def test_verdict_rejects_negative_headroom_gap() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="headroom gap"):
        SearchValueVerdict(
            gross_improvement=0.0,
            charged_search_cost=0.0,
            oracle_headroom_gap=-0.1,
            claims_improvement=False,
        )


def test_activation_gate_refuses_by_default() -> None:
    with pytest.raises(ConstructionSearchActivationRefusal, match="not activated"):
        ConstructionSearchActivationGate().authorize_claim()


def test_activation_gate_rejects_inactive_with_credentials() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="neither a receipt nor a license"):
        ConstructionSearchActivationGate(activated=False, license_id="lic.x")


def test_activation_gate_requires_valid_receipt_when_active() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="confirmation_receipt_sha256"):
        ConstructionSearchActivationGate(
            activated=True, confirmation_receipt_sha256="not-a-digest", license_id="lic.x"
        )


def test_activation_gate_authorizes_when_licensed() -> None:
    gate = ConstructionSearchActivationGate(
        activated=True,
        confirmation_receipt_sha256="a" * 64,
        license_id="lic.independent_replication",
    )
    gate.authorize_claim()  # does not raise


def test_run_is_deterministic_under_seed() -> None:
    first = run_construction_search(seed=11)
    second = run_construction_search(seed=11)
    assert first.digest() == second.digest()
    assert first.scores == second.scores


def test_run_covers_every_arm() -> None:
    trace = run_construction_search(seed=3)
    expected = set(REQUIRED_CONTROLS) | {ORACLE_HEADROOM_CONTROL, "construction-search"}
    assert set(trace.scores) == expected
    assert set(trace.evaluations) == expected


def test_search_never_exceeds_oracle_headroom() -> None:
    for seed in (1, 2, 5, 9, 17):
        trace = run_construction_search(seed=seed)
        assert trace.scores["construction-search"] <= trace.scores[ORACLE_HEADROOM_CONTROL] + 1e-12


def test_verdict_from_trace_holds_null_when_search_ties_controls() -> None:
    trace = run_construction_search(seed=7)
    verdict = verdict_from_trace(trace)
    assert not verdict.claims_improvement
    assert verdict.oracle_headroom_gap >= 0.0


def test_verdict_from_trace_charges_cost_and_refuses_when_it_erases_gain() -> None:
    trace = run_construction_search(seed=4)
    scores = dict(trace.scores)
    scores["construction-search"] = scores["no-search"] + 1.0
    scores[ORACLE_HEADROOM_CONTROL] = max(scores.values())
    charged = SearchTrace(
        seed=trace.seed,
        num_members=trace.num_members,
        num_tasks=trace.num_tasks,
        size_penalty=trace.size_penalty,
        per_eval_cost=10.0,
        scores=scores,
        evaluations=trace.evaluations,
    )
    verdict = verdict_from_trace(charged)
    assert not verdict.claims_improvement  # cost per evaluation dwarfs the gross gain


def test_trace_fails_closed_on_incomplete_arms() -> None:
    with pytest.raises(ConstructionSearchRefusal, match="cover every arm"):
        SearchTrace(
            seed=0,
            num_members=4,
            num_tasks=2,
            size_penalty=0.1,
            per_eval_cost=0.0,
            scores={"no-search": 0.1},
            evaluations={"no-search": 1},
        )


def test_module_declares_no_capability_claim() -> None:
    assert SCIENTIFIC_CAPABILITY_CLAIM is False
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"
    assert "G0 formation mechanics" in PRIOR_NULL
