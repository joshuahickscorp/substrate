"""Unit tests for the niche-dispatch scaffold (epoch G1 sub-questions C1, C2, D1).

These tests exercise the reproduction, disjointness, complementarity, and dispatch-value contracts.
They assert fail-closed refusals, digest stability, determinism under seed, control-set completeness,
the off-by-default activation gate, and claim-scope integrity. No capability is claimed.
"""

from __future__ import annotations

import pytest

from mop.experiments.expansion_harness import CLAIM_SCOPE as HARNESS_CLAIM_SCOPE
from mop.mechanisms.niche_dispatch_scaffold import (
    ACTIVATION_SCOPES,
    CLAIM_SCOPE,
    DISPATCH_CONTROLS,
    NICHE_DISPATCH_SCHEMA,
    SCIENTIFIC_CAPABILITY_CLAIM,
    ComplementarityContract,
    ConfirmationReceipt,
    ContextPartition,
    DisjointNicheContract,
    DispatchActivationGate,
    DispatchValueContract,
    MatchedBudget,
    NicheDeclaration,
    NicheDispatchRefusal,
    PerspectiveAssessment,
    ReproducedNiche,
    assert_valid_edcm_bed,
    build_default_dispatch_value_contract,
    coverage,
    synthesize_disjoint_bed,
    synthesize_partition,
    synthesize_valid_assessments,
)

# ---------------------------------------------------------------------------
# Claim scope and capability flag.
# ---------------------------------------------------------------------------


def test_claim_scope_is_byte_identical_to_harness() -> None:
    assert CLAIM_SCOPE == HARNESS_CLAIM_SCOPE
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_context_partition_rejects_widened_claim_scope() -> None:
    with pytest.raises(NicheDispatchRefusal, match="claim scope cannot be widened"):
        ContextPartition(
            partition_id="p.0",
            cells=("cell.000", "cell.001"),
            claim_scope="a capability was demonstrated",
        )


# ---------------------------------------------------------------------------
# C1. Partition, niche declaration, reproduction, disjointness.
# ---------------------------------------------------------------------------


def test_partition_digest_is_stable() -> None:
    a = ContextPartition(partition_id="p.0", cells=("cell.000", "cell.001"))
    b = ContextPartition(partition_id="p.0", cells=("cell.000", "cell.001"))
    assert a.digest() == b.digest()
    assert len(a.digest()) == 64


def test_partition_rejects_unsorted_or_duplicate_cells() -> None:
    with pytest.raises(NicheDispatchRefusal):
        ContextPartition(partition_id="p.0", cells=("cell.001", "cell.000"))
    with pytest.raises(NicheDispatchRefusal):
        ContextPartition(partition_id="p.0", cells=("cell.000", "cell.000"))


def test_niche_declaration_rejects_bad_schema() -> None:
    with pytest.raises(NicheDispatchRefusal, match="unsupported niche declaration schema"):
        NicheDeclaration(
            perspective_id="perspective.00",
            seed=0,
            cells=("cell.000",),
            schema="mop-niche-dispatch/v2",
        )


def test_reproduced_niche_refuses_when_seeds_disagree() -> None:
    with pytest.raises(NicheDispatchRefusal, match="does not reproduce"):
        ReproducedNiche(
            perspective_id="perspective.00",
            seed_cells={0: ("cell.000",), 1: ("cell.001",)},
        )


def test_reproduced_niche_requires_two_seeds() -> None:
    with pytest.raises(NicheDispatchRefusal, match="at least two independent seeds"):
        ReproducedNiche(perspective_id="perspective.00", seed_cells={0: ("cell.000",)})


def test_disjoint_contract_refuses_overlap() -> None:
    partition = ContextPartition(partition_id="p.0", cells=("cell.000", "cell.001", "cell.002"))
    left = ReproducedNiche(
        perspective_id="perspective.00", seed_cells={0: ("cell.000", "cell.001"), 1: ("cell.000", "cell.001")}
    )
    right = ReproducedNiche(perspective_id="perspective.01", seed_cells={0: ("cell.001",), 1: ("cell.001",)})
    with pytest.raises(NicheDispatchRefusal, match="disjoint"):
        DisjointNicheContract(partition=partition, niches=(left, right))


def test_disjoint_contract_refuses_cells_outside_partition() -> None:
    partition = ContextPartition(partition_id="p.0", cells=("cell.000", "cell.001"))
    stray = ReproducedNiche(perspective_id="perspective.00", seed_cells={0: ("cell.099",), 1: ("cell.099",)})
    other = ReproducedNiche(perspective_id="perspective.01", seed_cells={0: ("cell.000",), 1: ("cell.000",)})
    with pytest.raises(NicheDispatchRefusal, match="outside the declared partition"):
        DisjointNicheContract(partition=partition, niches=(stray, other))


def test_synthesize_disjoint_bed_is_deterministic_and_valid() -> None:
    first = synthesize_disjoint_bed(seeds=(0, 1, 2), n_perspectives=3, cells_each=2)
    second = synthesize_disjoint_bed(seeds=(0, 1, 2), n_perspectives=3, cells_each=2)
    assert first.digest() == second.digest()
    assert first.perspective_ids == ("perspective.00", "perspective.01", "perspective.02")
    assert first.cells_owned_by("perspective.01") == ("cell.002", "cell.003")


# ---------------------------------------------------------------------------
# C2. Complementarity screening and the EDCM invalid-bed null.
# ---------------------------------------------------------------------------


def test_valid_assessments_pass_edcm_bed() -> None:
    assessments = synthesize_valid_assessments(seed=7, n_perspectives=3, cells_each=2)
    assert_valid_edcm_bed(assessments, evenness_tolerance=0.0)
    contract = ComplementarityContract(assessments=assessments, evenness_tolerance=0.0)
    assert contract.retained == ("perspective.00", "perspective.01", "perspective.02")
    assert contract.refused == ()


def test_edcm_null_refuses_net_harmful_perspective() -> None:
    good = PerspectiveAssessment(
        perspective_id="perspective.00", net_effect=1.0, unique_positive_cells=("cell.000",)
    )
    harmful = PerspectiveAssessment(
        perspective_id="perspective.01", net_effect=-0.5, unique_positive_cells=("cell.001",)
    )
    with pytest.raises(NicheDispatchRefusal, match="net-harmful"):
        assert_valid_edcm_bed((good, harmful), evenness_tolerance=0.0)


def test_edcm_null_refuses_when_none_uniquely_positive() -> None:
    flat_a = PerspectiveAssessment(perspective_id="perspective.00", net_effect=0.0, unique_positive_cells=())
    flat_b = PerspectiveAssessment(perspective_id="perspective.01", net_effect=0.0, unique_positive_cells=())
    with pytest.raises(NicheDispatchRefusal, match="no uniquely positive"):
        assert_valid_edcm_bed((flat_a, flat_b), evenness_tolerance=0.0)


def test_edcm_null_refuses_uneven_bed() -> None:
    small = PerspectiveAssessment(
        perspective_id="perspective.00", net_effect=1.0, unique_positive_cells=("cell.000",)
    )
    big = PerspectiveAssessment(
        perspective_id="perspective.01",
        net_effect=1.0,
        unique_positive_cells=("cell.001", "cell.002", "cell.003"),
    )
    with pytest.raises(NicheDispatchRefusal, match="uneven"):
        assert_valid_edcm_bed((small, big), evenness_tolerance=1.0)


def test_complementarity_refuses_universal_assumption() -> None:
    assessments = synthesize_valid_assessments(seed=1)
    with pytest.raises(NicheDispatchRefusal, match="refuses the universal-complementarity assumption"):
        ComplementarityContract(
            assessments=assessments,
            evenness_tolerance=0.0,
            assume_universal_complementarity=True,
        )


def test_complementarity_retention_honesty_check() -> None:
    good = PerspectiveAssessment(
        perspective_id="perspective.00", net_effect=1.0, unique_positive_cells=("cell.000",)
    )
    neutral = PerspectiveAssessment(
        perspective_id="perspective.01", net_effect=0.5, unique_positive_cells=("cell.001",)
    )
    contract = ComplementarityContract(assessments=(good, neutral), evenness_tolerance=0.0)
    contract.assert_retention_honest(("perspective.00", "perspective.01"))
    with pytest.raises(NicheDispatchRefusal, match="without a uniquely positive niche"):
        contract.assert_retention_honest(("perspective.99",))


# ---------------------------------------------------------------------------
# D1. Matched compute and the dispatch value contract.
# ---------------------------------------------------------------------------


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(NicheDispatchRefusal, match="non-vacuous"):
        MatchedBudget(params=0, flops=1, wall_ticks=1, dispatch_calls=1)


def test_default_dispatch_contract_has_full_control_family() -> None:
    contract = build_default_dispatch_value_contract()
    assert contract.controls == DISPATCH_CONTROLS
    assert set(DISPATCH_CONTROLS) == {
        "all-perspectives",
        "random-dispatch",
        "single-best",
        "majority-vote",
    }
    assert contract.digest() == build_default_dispatch_value_contract().digest()


def test_dispatch_contract_refuses_control_drift() -> None:
    with pytest.raises(NicheDispatchRefusal, match="incomplete or out of canonical order"):
        DispatchValueContract(
            controls=("random-dispatch", "all-perspectives", "single-best", "majority-vote"),
            matched=MatchedBudget(params=1, flops=1, wall_ticks=1, dispatch_calls=1),
            matched_cost_required=True,
            replication_min=2,
            metric_name="held_out_task_score",
        )


def test_dispatch_contract_requires_matched_cost() -> None:
    with pytest.raises(NicheDispatchRefusal, match="matched full-system cost"):
        DispatchValueContract(
            controls=DISPATCH_CONTROLS,
            matched=MatchedBudget(params=1, flops=1, wall_ticks=1, dispatch_calls=1),
            matched_cost_required=False,
            replication_min=2,
            metric_name="held_out_task_score",
        )


def test_dispatch_value_only_claimed_when_beating_every_control() -> None:
    contract = build_default_dispatch_value_contract()
    winning = {
        "all-perspectives": 0.80,
        "random-dispatch": 0.60,
        "single-best": 0.79,
        "majority-vote": 0.81,
    }
    # Ties or losses against majority-vote yield no claim.
    assert contract.may_claim_value(0.81, winning) is False
    assert contract.may_claim_value(0.90, winning) is True


def test_dispatch_value_refuses_incomplete_control_scores() -> None:
    contract = build_default_dispatch_value_contract()
    with pytest.raises(NicheDispatchRefusal, match="cover exactly the declared control family"):
        contract.may_claim_value(0.9, {"all-perspectives": 0.5})


# ---------------------------------------------------------------------------
# E. Activation gate.
# ---------------------------------------------------------------------------


def test_activation_gate_refuses_by_default() -> None:
    gate = DispatchActivationGate()
    with pytest.raises(NicheDispatchRefusal, match="not earned yet"):
        gate.authorize()


def test_activation_gate_rejects_permissive_construction() -> None:
    with pytest.raises(NicheDispatchRefusal, match="never permitted"):
        DispatchActivationGate(local_activation_permitted=True)


def test_activation_gate_accepts_valid_receipt() -> None:
    gate = DispatchActivationGate()
    receipt = ConfirmationReceipt(
        issuer="external.authority",
        license_sha256="a" * 64,
        scope="gated-pilot",
    )
    assert receipt.scope in ACTIVATION_SCOPES
    assert gate.authorize(receipt) is receipt


def test_confirmation_receipt_rejects_unknown_scope() -> None:
    with pytest.raises(NicheDispatchRefusal, match="unsupported activation scope"):
        ConfirmationReceipt(issuer="external.authority", license_sha256="a" * 64, scope="production")


# ---------------------------------------------------------------------------
# Coverage record.
# ---------------------------------------------------------------------------


def test_coverage_lists_every_subquestion_with_two_bullets() -> None:
    cov = coverage()
    assert set(cov) == {"C1", "C2", "D1"}
    for bullets in cov.values():
        assert len(bullets) >= 2


def test_partition_schema_constant_matches() -> None:
    assert NICHE_DISPATCH_SCHEMA == "mop-niche-dispatch/v1"
    partition = synthesize_partition(n_perspectives=2, cells_each=2)
    assert partition.schema == NICHE_DISPATCH_SCHEMA
    assert partition.cells == ("cell.000", "cell.001", "cell.002", "cell.003")
