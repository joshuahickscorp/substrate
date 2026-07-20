
from __future__ import annotations

import pytest

from mop.mechanisms.integrated_escs_scaffold import (
    ADVANTAGE_VERDICT,
    CLAIM_SCOPE,
    COST_AXES,
    MECHANISM_LADDER,
    NULL_VERDICT,
    REQUIRED_BASELINES,
    SCIENTIFIC_CAPABILITY_CLAIM,
    AblationLadderContract,
    ActivationReceipt,
    BaselineDeclaration,
    BaselineSet,
    CostVector,
    FrontierPoint,
    FrontierVerdict,
    IntegratedActivationGate,
    IntegratedAdvantageContract,
    IntegratedEscsRefusal,
    MechanismRung,
    build_default_ablation_ladder,
    build_default_baseline_set,
    build_dominating_frontier_verdict,
    build_null_frontier_verdict,
    coverage,
)

INTEGRATED_ESCS_SCHEMA = "mop-integrated-escs/v1"


def _cost() -> CostVector:
    return CostVector(params=10, flops=20, memory_bytes=40, wall_ticks=8, energy_units=4)


def test_module_declares_no_capability_claim() -> None:
    assert SCIENTIFIC_CAPABILITY_CLAIM is False
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_cost_vector_must_be_non_vacuous() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="non-vacuous"):
        CostVector(params=0, flops=1, memory_bytes=1, wall_ticks=1, energy_units=1)


def test_cost_vector_axis_order_is_stable() -> None:
    cost = _cost()
    assert tuple(cost.as_mapping()) == COST_AXES
    assert cost.as_tuple() == (10, 20, 40, 8, 4)


def test_frontier_point_digest_is_stable() -> None:
    a = FrontierPoint(label="p.a", quality=0.5, cost=_cost())
    b = FrontierPoint(label="p.a", quality=0.5, cost=_cost())
    assert a.digest() == b.digest()
    assert len(a.digest()) == 64


def test_frontier_point_rejects_out_of_range_quality() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="lie in"):
        FrontierPoint(label="p.a", quality=1.5, cost=_cost())


def test_frontier_point_rejects_widened_claim_scope() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="claim scope cannot be widened"):
        FrontierPoint(label="p.a", quality=0.5, cost=_cost(), claim_scope="a capability was shown")


def test_dominance_requires_strict_improvement_somewhere() -> None:
    cheap = CostVector(params=5, flops=10, memory_bytes=20, wall_ticks=4, energy_units=2)
    a = FrontierPoint(label="p.a", quality=0.7, cost=cheap)
    b = FrontierPoint(label="p.b", quality=0.6, cost=_cost())
    assert a.dominates(b)
    assert not b.dominates(a)
    tie = FrontierPoint(label="p.tie", quality=0.7, cost=cheap)
    assert not a.dominates(tie)  # identical cost and quality is not strict domination


def test_default_baseline_set_covers_every_family_in_order() -> None:
    baseline_set = build_default_baseline_set()
    assert tuple(row.family for row in baseline_set.declarations) == REQUIRED_BASELINES


def test_baseline_set_fails_closed_on_order_drift() -> None:
    full = build_default_baseline_set()
    reordered = tuple(reversed(full.declarations))
    with pytest.raises(IntegratedEscsRefusal, match="membership or order drift"):
        BaselineSet(schema=INTEGRATED_ESCS_SCHEMA, declarations=reordered)


def test_baseline_must_be_declared_matched() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="cost-matched"):
        BaselineDeclaration(
            family="single-perspective",
            point=FrontierPoint(label="b.sp", quality=0.5, cost=_cost()),
            rationale="r",
            matched=False,
        )


def test_baseline_rejects_unknown_family() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="unsupported baseline family"):
        BaselineDeclaration(
            family="quantum-council",
            point=FrontierPoint(label="b.x", quality=0.5, cost=_cost()),
            rationale="r",
            matched=True,
        )


def test_baseline_cost_match_check_fails_closed() -> None:
    baseline_set = build_default_baseline_set()
    other_cost = CostVector(params=1, flops=1, memory_bytes=1, wall_ticks=1, energy_units=1)
    with pytest.raises(IntegratedEscsRefusal, match="not matched"):
        baseline_set.assert_all_cost_matched(other_cost)


def _advantage_contract(**overrides: object) -> IntegratedAdvantageContract:
    baseline_set = build_default_baseline_set()
    integrated = FrontierPoint(
        label="integrated.escs",
        quality=0.9,
        cost=baseline_set.declarations[0].point.cost,
    )
    kwargs: dict[str, object] = {
        "schema": INTEGRATED_ESCS_SCHEMA,
        "integrated": integrated,
        "baselines": baseline_set,
        "matched_cost_required": True,
        "dominance_required": True,
        "replication_min": 2,
    }
    kwargs.update(overrides)
    return IntegratedAdvantageContract(**kwargs)  # type: ignore[arg-type]


def test_advantage_contract_valid_and_digest_stable() -> None:
    contract = _advantage_contract()
    assert contract.digest() == _advantage_contract().digest()


def test_advantage_contract_requires_matched_cost() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="matched full-system cost"):
        _advantage_contract(matched_cost_required=False)


def test_advantage_contract_requires_dominance() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="Pareto dominance"):
        _advantage_contract(dominance_required=False)


def test_advantage_contract_rejects_single_replication() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="independent replications"):
        _advantage_contract(replication_min=1)


def test_default_ablation_ladder_is_valid_and_ordered() -> None:
    ladder = build_default_ablation_ladder()
    assert tuple(row.mechanism for row in ladder.rungs) == MECHANISM_LADDER
    assert ladder.total_marginal_gain() > 0.0


def test_ablation_ladder_fails_closed_on_unjustified_rung() -> None:
    rungs = list(build_default_ablation_ladder().rungs)
    starved = MechanismRung(
        mechanism=MECHANISM_LADDER[-1],
        marginal_quality_gain=1e-9,
        marginal_cost=CostVector(
            params=10, flops=1_000_000, memory_bytes=10, wall_ticks=1, energy_units=1
        ),
        min_efficiency=1.0,
    )
    rungs[-1] = starved
    with pytest.raises(IntegratedEscsRefusal, match="do not justify their marginal compute"):
        AblationLadderContract(schema=INTEGRATED_ESCS_SCHEMA, rungs=tuple(rungs))


def test_ablation_ladder_fails_closed_on_membership_drift() -> None:
    rungs = build_default_ablation_ladder().rungs[:-1]
    with pytest.raises(IntegratedEscsRefusal, match="membership or order drift"):
        AblationLadderContract(schema=INTEGRATED_ESCS_SCHEMA, rungs=rungs)


def test_mechanism_rung_requires_positive_gain() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="strictly positive marginal quality"):
        MechanismRung(
            mechanism=MECHANISM_LADDER[0],
            marginal_quality_gain=0.0,
            marginal_cost=_cost(),
            min_efficiency=1e-6,
        )


def test_frontier_verdict_default_is_null_and_refuses_to_earn() -> None:
    verdict = build_null_frontier_verdict()
    assert verdict.verdict() == NULL_VERDICT
    with pytest.raises(IntegratedEscsRefusal, match="unearned"):
        verdict.assert_advantage_earned()


def test_frontier_verdict_fails_closed_on_unmatched_cost_flag() -> None:
    baseline_set = build_default_baseline_set()
    integrated = FrontierPoint(
        label="integrated.escs", quality=0.9, cost=baseline_set.declarations[0].point.cost
    )
    with pytest.raises(IntegratedEscsRefusal, match="matched cost was not confirmed"):
        FrontierVerdict(
            schema=INTEGRATED_ESCS_SCHEMA,
            integrated=integrated,
            baselines=baseline_set,
            matched_cost_confirmed=False,
            replications=2,
        )


def test_frontier_verdict_fails_closed_on_actually_unmatched_cost() -> None:
    baseline_set = build_default_baseline_set()
    mismatched = FrontierPoint(
        label="integrated.escs",
        quality=0.9,
        cost=CostVector(params=1, flops=1, memory_bytes=1, wall_ticks=1, energy_units=1),
    )
    with pytest.raises(IntegratedEscsRefusal, match="not matched"):
        FrontierVerdict(
            schema=INTEGRATED_ESCS_SCHEMA,
            integrated=mismatched,
            baselines=baseline_set,
            matched_cost_confirmed=True,
            replications=2,
        )


def test_frontier_verdict_dominating_earns_advantage() -> None:
    verdict = build_dominating_frontier_verdict()
    assert verdict.verdict() == ADVANTAGE_VERDICT
    verdict.assert_advantage_earned()  # does not raise


def test_frontier_verdict_is_deterministic_under_seed() -> None:
    assert build_null_frontier_verdict(seed=7).digest() == build_null_frontier_verdict(seed=7).digest()
    assert build_null_frontier_verdict(seed=1).digest() != build_null_frontier_verdict(seed=2).digest()


def test_activation_gate_refuses_by_default() -> None:
    gate = IntegratedActivationGate()
    verdict = build_dominating_frontier_verdict()
    with pytest.raises(IntegratedEscsRefusal, match="not earned"):
        gate.authorize(None, expected_verdict_digest=verdict.digest())


def test_activation_gate_rejects_permissive_construction() -> None:
    with pytest.raises(IntegratedEscsRefusal, match="never permitted"):
        IntegratedActivationGate(local_activation_permitted=True)


def test_activation_gate_rejects_mismatched_receipt() -> None:
    gate = IntegratedActivationGate()
    verdict = build_dominating_frontier_verdict()
    receipt = ActivationReceipt(
        verdict_digest="0" * 64,
        replications=3,
        independent_auditor="external audit lab",
        preregistration_sha256="a" * 64,
    )
    with pytest.raises(IntegratedEscsRefusal, match="does not bind"):
        gate.authorize(receipt, expected_verdict_digest=verdict.digest())


def test_activation_gate_accepts_valid_receipt() -> None:
    gate = IntegratedActivationGate()
    verdict = build_dominating_frontier_verdict()
    receipt = ActivationReceipt(
        verdict_digest=verdict.digest(),
        replications=3,
        independent_auditor="external audit lab",
        preregistration_sha256="a" * 64,
    )
    gate.authorize(receipt, expected_verdict_digest=verdict.digest())  # does not raise


def test_activation_receipt_rejects_single_replication() -> None:
    verdict = build_dominating_frontier_verdict()
    with pytest.raises(IntegratedEscsRefusal, match="independent replications"):
        ActivationReceipt(
            verdict_digest=verdict.digest(),
            replications=1,
            independent_auditor="external audit lab",
            preregistration_sha256="a" * 64,
        )


def test_coverage_lists_every_sub_question_with_bullets() -> None:
    cov = coverage()
    assert set(cov) == {
        "matched-baseline-frontier",
        "marginal-mechanism-justification",
        "pareto-dominance-or-null",
    }
    for bullets in cov.values():
        assert len(bullets) >= 2
