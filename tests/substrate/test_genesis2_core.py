from __future__ import annotations

from typing import cast

from substrate import genesis2_clean_clone as CC2
from substrate import genesis2_config as C2
from substrate import genesis2_harness as H2
from substrate import genesis2_ledger as L2
from substrate import genesis2_material as F2
from substrate import genesis2_microstore as MS2
from substrate import genesis_material as M


def _opportunity(observations: list[M.Observation]) -> M.Opportunity:
    return M.equal_opportunity(
        envelope="1GB",
        observations=observations,
        sensor_channels=tuple(sorted({row.channel for row in observations})),
        operation_budget=100_000,
        durable_write_budget=4_096,
    )


def test_genesis2_constitution_is_closed_and_inactive() -> None:
    assert C2.ACTIVATION is False
    assert C2.CLAIM_BOUNDARY["unqualified_nous"] is False
    assert C2.CLAIM_BOUNDARY["external_activation"] is False
    assert len(C2.CANARIES) == 22
    assert len(C2.REVIEW_CELLS) == 71
    assert len(C2.MUTATIONS) == 17


def test_parameterised_affine_requires_an_observed_affine_form() -> None:
    pairs = [((value,), ((3 * value + 2) % MS2.MODULUS,)) for value in range(6)]
    observed = next(rule for rule in MS2.induce("tool_use@1>2", pairs) if rule.kind == "affine")
    compiled = MS2.parameterise_affine(observed, argument_slot=1, scale_slot=2, offset_slot=3)

    assert compiled.apply((99, 7, 3, 2)) == ((3 * 7 + 2) % MS2.MODULUS,)


def test_conditional_scheduler_admits_slot_derived_consolidation() -> None:
    observations = [
        M.Observation(
            index=index,
            channel="tool_use",
            payload=(1, 99, value, (3 * value + 2) % MS2.MODULUS),
            teaching=True,
        )
        for index, value in enumerate(range(6))
    ]
    material = cast(F2.L9_minimal_sufficient_field, M.build("L9_minimal_sufficient_field", _opportunity(observations)))
    for observation in observations:
        material.observe(observation)

    proposals = material.propose()
    derived = [proposal for proposal in proposals if proposal.kind == "structural_consolidation" and proposal.target == "tool_use@1>2"]
    assert derived, "the slot-derived relation was left inert"
    receipts = material.apply([M.Verdict(derived[0].proposal_id, True, 1.0, 1.0)])
    material.finalize_receipt(receipts[0])

    answer = material.answer(M.Probe(0, "tool_acquisition", "tool", (99, 7, 3, 2), 1))
    assert answer.value == ((3 * 7 + 2) % MS2.MODULUS,)


def test_checkpoint_preserves_scheduler_accounting_and_lineage() -> None:
    observations = [M.Observation(index, "scope", (1, index, (index + 2) % 8), teaching=True) for index in range(4)]
    material = cast(
        F2.L9_minimal_sufficient_field,
        M.build("L9_minimal_sufficient_field", _opportunity(observations)),
    )
    for observation in observations:
        material.observe(observation)
    proposals = list(material.propose())
    receipts = material.apply([M.Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in proposals])
    for receipt in receipts:
        material.finalize_receipt(receipt)
    assert not material.store.undo
    assert not material.structure_undo

    checkpoint = material.checkpoint()
    replica = cast(
        F2.L9_minimal_sufficient_field,
        M.build("L9_minimal_sufficient_field", _opportunity(observations)),
    )
    replica.restore(checkpoint)

    assert replica.durable_state_digest() == material.durable_state_digest()
    assert replica.active_state_digest() == material.active_state_digest()
    assert replica.proposal_epoch == material.proposal_epoch
    assert replica._opportunity.ledger.operations == material._opportunity.ledger.operations
    assert replica._opportunity.ledger.durable_writes == material._opportunity.ledger.durable_writes
    assert replica.receipts == material.receipts


def test_statistics_use_family_history_cells_as_independent_units() -> None:
    from substrate import genesis2_statistics as S2

    rows = [
        {"family": family, "history_id": 0, "arm": arm, "score": score}
        for family, values in {
            "a": {"field": 0.8, "monolith": 0.3},
            "b": {"field": 0.4, "monolith": 0.3},
        }.items()
        for arm, score in values.items()
    ]
    differences = S2.paired_differences(
        rows,
        candidate="field",
        comparator="monolith",
    )
    assert [round(value, 9) for value in differences] == [0.5, 0.1]


def test_microstore_rollback_and_checkpoint_are_exact() -> None:
    store = MS2.Microstore()
    before = store.digest()
    store.write("scope", (1,), (5,), undo_token="write")
    assert store.digest() != before
    assert store.rollback("write")
    assert store.digest() == before

    store.write("scope", (2,), (6,))
    replica = MS2.Microstore()
    replica.restore(store.document())
    assert replica.digest() == store.digest()


def test_update_ledger_reports_real_units() -> None:
    ledger = L2.UpdateLedger("arm")
    ledger.record(
        L2.UpdateRecord(
            "proposal",
            "arm",
            "micro_association",
            bytes_written=16,
            compute=4,
        )
    )
    ledger.settle("proposal", committed=True, future_utility=0.5)
    report = ledger.report()
    assert report["attempt_count"] == 1
    assert report["committed_count"] == 1
    assert report["utility_per_written_byte"] == 0.5 / 16


def test_harness_batches_micro_updates_but_not_structure() -> None:
    assert H2._batched("micro_association")
    assert H2._batched("monolith_association_write")
    assert not H2._batched("structural_consolidation")
    assert not H2._batched("topology_revision")
    proposals = [M.Proposal(str(index), "micro_association", str(index)) for index in range(H2.MICRO_BATCH_LIMIT + 1)]
    batches = H2._micro_batches(proposals)
    assert [len(batch) for batch in batches] == [H2.MICRO_BATCH_LIMIT, 1]


def test_clean_clone_analysis_comparison_is_strict() -> None:
    expected = {
        "candidate": "field",
        "comparator": "monolith",
        "histories": 8,
        "effect": 0.1,
        "confidence_lower": 0.05,
        "confidence_upper": 0.15,
        "p_value": 0.01,
        "oracle_headroom": 0.2,
        "primary_gate_pass": True,
        "robust_gate_pass": False,
    }
    assert CC2._same_analysis(expected, dict(expected))
    changed = dict(expected)
    changed["effect"] = 0.1000001
    assert not CC2._same_analysis(expected, changed)
