from __future__ import annotations

from copy import deepcopy

import pytest

from mop.escs.accounting import LifecycleLedger, WorkVector
from mop.substrate.events import EventRef


def test_work_vector_keeps_operational_work_and_retained_byte_time_separate() -> None:
    active = WorkVector(
        raw_transport_and_adapters=2,
        event_formation=3,
        actor_execution=5,
        idle_floor=7,
    )
    retention = WorkVector.retention(retained_bytes=11, start_tick=2, end_tick=7)

    total = active + retention

    assert total.total_work == 17
    assert total.retained_byte_time == 55
    assert WorkVector.from_payload(total.payload()) == total
    assert total.scale(2).actor_execution == 10


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_work_vector_rejects_inexact_or_negative_counters(value: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        WorkVector(actor_execution=value)  # type: ignore[arg-type]


def test_lifecycle_ledger_charges_idle_retention_and_exact_replay() -> None:
    event_id = EventRef("event:source")
    ledger = LifecycleLedger()
    ledger.charge_idle(
        owner="event-former",
        reason="empty-queue-check",
        idle_work=2,
        start_tick=0,
        end_tick=1,
    )
    ledger.charge(
        owner="actor:one",
        reason="bounded-activation",
        work=WorkVector(actor_execution=5, messages=3),
        start_tick=2,
        end_tick=2,
        causal_event_ids=(event_id,),
    )
    ledger.charge_retention(
        owner="archive",
        reason="hot-journal-retention",
        retained_bytes=10,
        start_tick=2,
        end_tick=6,
        causal_event_ids=(event_id,),
    )

    replay = LifecycleLedger.replay(ledger.payload())

    assert replay.payload() == ledger.payload()
    assert replay.sha256 == ledger.sha256
    assert replay.total.total_work == 10
    assert replay.total.retained_byte_time == 40
    assert replay.verify(event_ids={str(event_id)}) == []


def test_lifecycle_replay_fails_closed_on_unknown_fields_and_tampering() -> None:
    ledger = LifecycleLedger()
    ledger.charge(
        owner="adapter",
        reason="transport",
        work=WorkVector(raw_transport_and_adapters=1),
        start_tick=0,
        end_tick=0,
    )
    extra = deepcopy(ledger.payload())
    extra["entries"][0]["unpriced"] = 9
    with pytest.raises(ValueError, match="fields mismatch"):
        LifecycleLedger.from_payload(extra)

    drift = deepcopy(ledger.payload())
    drift["total"]["raw_transport_and_adapters"] = 0
    with pytest.raises(ValueError, match="total mismatch"):
        LifecycleLedger.from_payload(drift)


def test_lifecycle_event_reference_validation_is_explicit() -> None:
    ledger = LifecycleLedger()
    ledger.charge(
        owner="index",
        reason="append",
        work=WorkVector(indexing_and_graph_maintenance=1),
        start_tick=1,
        end_tick=1,
        causal_event_ids=(EventRef("event:missing"),),
    )
    assert ledger.verify(event_ids=set()) == ["charge 0 references missing event event:missing"]

    with pytest.raises(ValueError, match="must contain EventRef"):
        ledger.charge(
            owner="index",
            reason="untyped-reference",
            work=WorkVector(indexing_and_graph_maintenance=1),
            start_tick=2,
            end_tick=2,
            causal_event_ids=("event:not-typed",),  # type: ignore[arg-type]
        )


def test_idle_charge_requires_nonzero_declared_check_work() -> None:
    ledger = LifecycleLedger()

    with pytest.raises(ValueError, match="positive integer"):
        ledger.charge_idle(
            owner="event-former",
            reason="empty-queue-check",
            idle_work=0,
            start_tick=0,
            end_tick=0,
        )

    assert ledger.entries == ()


def test_lifecycle_entry_count_is_constant_shape_and_tracks_next_sequence() -> None:
    ledger = LifecycleLedger()
    assert ledger.entry_count == 0
    ledger.charge_idle(
        owner="event-former",
        reason="poll",
        idle_work=1,
        start_tick=0,
        end_tick=0,
    )
    assert ledger.entry_count == 1
