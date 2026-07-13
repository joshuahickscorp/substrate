from __future__ import annotations

import hashlib

import pytest

from mop.escs.archive import (
    ArchiveCharge,
    BoundedArchive,
    CorruptSegmentError,
    PayloadErasedError,
    PayloadErasureError,
    ReplayAuthority,
    ReplayAuthorityError,
)


def _envelope(event_id: str, payload: bytes) -> dict[str, object]:
    return {
        "schema": "mop-escs-event-envelope/v1",
        "event_id": event_id,
        "event_kind": "observation",
        "causal_parent_ids": [],
        "payload_digest": hashlib.sha256(payload).hexdigest(),
    }


class _MutableStore:
    def __init__(self) -> None:
        self.segments: dict[str, bytes] = {}

    def put(self, root_sha256: str, segment: bytes) -> None:
        assert hashlib.sha256(segment).hexdigest() == root_sha256
        self.segments[root_sha256] = bytes(segment)

    def read(self, root_sha256: str) -> bytes:
        return self.segments[root_sha256]

    def delete(self, root_sha256: str) -> bool:
        return self.segments.pop(root_sha256, None) is not None

    def contains(self, root_sha256: str) -> bool:
        return root_sha256 in self.segments


class _NonDeletingStore(_MutableStore):
    def delete(self, root_sha256: str) -> bool:
        return False


def test_hot_journal_enforces_byte_and_tick_bounds() -> None:
    archive = BoundedArchive(max_hot_bytes=10_000, max_hot_age_ticks=2)
    archive.append(_envelope("event:a", b"a"), b"a", admitted_tick=0)
    archive.append(_envelope("event:b", b"b"), b"b", admitted_tick=2)

    assert archive.hot_event_ids == ("event:a", "event:b")
    snapshot = archive.advance(3)

    assert snapshot is not None
    assert [row.event_id for row in snapshot.rows] == ["event:a"]
    assert archive.hot_event_ids == ("event:b",)
    assert archive.hot_bytes <= archive.max_hot_bytes
    assert archive.audit() == ()

    byte_bounded = BoundedArchive(max_hot_bytes=1, max_hot_age_ticks=100)
    byte_bounded.append(_envelope("event:large", b"payload"), b"payload", admitted_tick=0)
    assert byte_bounded.hot_event_ids == ()
    assert byte_bounded.hot_bytes == 0
    assert byte_bounded.latest_snapshot is not None


def test_compaction_roots_are_independent_of_compaction_cadence() -> None:
    rows = [("event:a", b"alpha", 0), ("event:b", b"beta", 1), ("event:c", b"gamma", 2)]
    one_shot = BoundedArchive(max_hot_bytes=10_000, max_hot_age_ticks=100)
    incremental = BoundedArchive(max_hot_bytes=10_000, max_hot_age_ticks=100)

    for event_id, payload, tick in rows:
        one_shot.append(_envelope(event_id, payload), payload, admitted_tick=tick)
    one_shot_snapshot = one_shot.compact(2, force=True)

    for event_id, payload, tick in rows:
        incremental.append(_envelope(event_id, payload), payload, admitted_tick=tick)
        incremental.compact(tick, force=True)
    incremental_snapshot = incremental.latest_snapshot

    assert one_shot_snapshot is not None
    assert incremental_snapshot is not None
    assert one_shot_snapshot.snapshot_sha256 == incremental_snapshot.snapshot_sha256
    assert one_shot_snapshot.lineage_root_sha256 == incremental_snapshot.lineage_root_sha256
    assert (
        one_shot_snapshot.segment_commitment_root_sha256
        == incremental_snapshot.segment_commitment_root_sha256
    )
    assert one_shot_snapshot.payload() == incremental_snapshot.payload()


def test_corrupt_cold_segment_is_rejected_and_not_masked_by_cache() -> None:
    store = _MutableStore()
    archive = BoundedArchive(
        max_hot_bytes=1,
        max_hot_age_ticks=10,
        max_cache_bytes=100,
        segment_store=store,
    )
    payload = b"decision-relevant-payload"
    archive.append(_envelope("event:a", payload), payload, admitted_tick=0)
    location = archive.payload_location("event:a")
    assert location is not None and location.segment_root_sha256 is not None
    root = location.segment_root_sha256

    assert archive.retrieve("event:a") == payload
    assert "event:a" in archive.cached_event_ids
    store.segments[root] = store.segments[root][:-1] + bytes([store.segments[root][-1] ^ 1])

    with pytest.raises(CorruptSegmentError, match="content-address mismatch"):
        archive.retrieve("event:a")
    assert "event:a" not in archive.cached_event_ids
    assert any("cold segment rejected" in problem for problem in archive.audit())


def test_erasure_removes_payload_index_cache_and_disables_exact_replay() -> None:
    store = _MutableStore()
    archive = BoundedArchive(
        max_hot_bytes=1,
        max_hot_age_ticks=100,
        max_cache_bytes=100,
        segment_store=store,
    )
    payload = b"erasable"
    lineage = archive.append(_envelope("event:erase", payload), payload, admitted_tick=0)
    location = archive.payload_location("event:erase")
    assert location is not None and location.segment_root_sha256 is not None
    root = location.segment_root_sha256
    archive.retrieve("event:erase")

    marker = archive.erase_payload("event:erase", deletion_tick=1, reason_code="retention-expired")

    assert archive.lineage("event:erase") == lineage
    assert archive.payload_location("event:erase") is None
    assert "event:erase" not in archive.payload_index_event_ids
    assert "event:erase" not in archive.cached_event_ids
    assert not store.contains(root)
    assert marker.logical_non_retrievability is True
    assert marker.physical_media_erasure_claimed is False
    assert not {"payload", "payload_digest", "segment_root", "content"}.intersection(marker.payload())
    assert archive.replay_authority is ReplayAuthority.DISABLED_AFTER_ERASURE
    with pytest.raises(PayloadErasedError, match="logically erased"):
        archive.retrieve("event:erase")
    with pytest.raises(ReplayAuthorityError, match="disabled"):
        archive.require_replay_authority()
    assert archive.audit() == ()

    archive.append(_envelope("event:new", b"new"), b"new", admitted_tick=2)
    assert archive.replay_authority is ReplayAuthority.DISABLED_AFTER_ERASURE


def test_hot_payload_erasure_removes_hot_and_cached_copies() -> None:
    archive = BoundedArchive(max_hot_bytes=10_000, max_hot_age_ticks=100)
    payload = b"hot"
    archive.append(_envelope("event:hot", payload), payload, admitted_tick=0)
    archive.retrieve("event:hot")
    assert archive.hot_event_ids == ("event:hot",)
    assert archive.cached_event_ids == frozenset({"event:hot"})

    archive.erase_payload("event:hot", deletion_tick=0, reason_code="user-request")

    assert archive.hot_event_ids == ()
    assert archive.hot_bytes == 0
    assert archive.cached_event_ids == frozenset()
    assert archive.payload_index_event_ids == frozenset()
    assert archive.audit() == ()


def test_failed_store_deletion_does_not_publish_an_erasure_marker() -> None:
    store = _NonDeletingStore()
    archive = BoundedArchive(max_hot_bytes=1, max_hot_age_ticks=100, segment_store=store)
    payload = b"still-readable"
    archive.append(_envelope("event:retained", payload), payload, admitted_tick=0)

    with pytest.raises(PayloadErasureError, match="logical non-retrievability"):
        archive.erase_payload("event:retained", deletion_tick=1, reason_code="user-request")

    assert archive.deletion_marker("event:retained") is None
    assert archive.replay_authority is ReplayAuthority.ENABLED
    assert archive.retrieve("event:retained") == payload


def test_audit_detects_a_cold_segment_resurrected_after_erasure() -> None:
    store = _MutableStore()
    archive = BoundedArchive(max_hot_bytes=1, max_hot_age_ticks=100, segment_store=store)
    payload = b"must-stay-erased"
    archive.append(_envelope("event:resurrection", payload), payload, admitted_tick=0)
    location = archive.payload_location("event:resurrection")
    assert location is not None and location.segment_root_sha256 is not None
    root = location.segment_root_sha256
    saved_segment = store.read(root)

    archive.erase_payload("event:resurrection", deletion_tick=1, reason_code="user-request")
    assert archive.audit() == ()

    store.segments[root] = saved_segment
    assert any("erased segment resurrected" in problem for problem in archive.audit())


def test_throwing_accounting_observer_cannot_erase_internal_charge_history() -> None:
    def fail_observer(_: ArchiveCharge) -> None:
        raise RuntimeError("observer unavailable")

    archive = BoundedArchive(
        max_hot_bytes=1_000,
        max_hot_age_ticks=10,
        accounting_hook=fail_observer,
    )
    payload = b"charged-despite-observer"

    archive.append(_envelope("event:observer", payload), payload, admitted_tick=0)

    snapshot = archive.accounting_snapshot
    assert snapshot.charge_count == 1
    assert dict((row[0], row[1]) for row in snapshot.by_operation)["append"] == 1
    assert any("accounting observer failed" in problem for problem in archive.audit())


def test_repeated_retrieval_uses_fixed_shape_accounting_not_an_unbounded_journal() -> None:
    archive = BoundedArchive(max_hot_bytes=1_000, max_hot_age_ticks=10)
    payload = b"stable-cache-payload"
    archive.append(_envelope("event:retrieval-scale", payload), payload, admitted_tick=0)
    before = archive.retained_bytes

    for _ in range(200):
        assert archive.retrieve("event:retrieval-scale") == payload

    snapshot = archive.accounting_snapshot
    assert snapshot.charge_count == 201
    assert tuple(row[0] for row in snapshot.by_operation) == (
        "append",
        "compaction",
        "erasure",
        "retention",
        "retrieval",
    )
    assert archive.retained_bytes - before < 64


def test_accounting_hook_charges_compaction_retrieval_and_retained_byte_time() -> None:
    charges: list[ArchiveCharge] = []
    archive = BoundedArchive(
        max_hot_bytes=1,
        max_hot_age_ticks=10,
        accounting_hook=charges.append,
    )
    payload = b"charged"
    archive.append(_envelope("event:charged", payload), payload, admitted_tick=0)
    archive.retrieve("event:charged", current_tick=4)

    by_operation = {charge.operation: charge for charge in charges}
    assert by_operation["append"].work_units == 1
    assert by_operation["append"].bytes_touched > len(payload)
    assert by_operation["compaction"].work_units == 1
    assert by_operation["compaction"].bytes_touched > len(payload)
    assert by_operation["retrieval"].bytes_touched > len(payload)
    assert by_operation["retention"].retained_bytes > 0
    assert by_operation["retention"].retained_byte_ticks > 0


def test_envelope_payload_separation_and_identity_are_fail_closed() -> None:
    archive = BoundedArchive(max_hot_bytes=1_000, max_hot_age_ticks=10)
    payload = b"payload"
    embedded = _envelope("event:bad", payload)
    embedded["payload"] = "not allowed in immutable envelope lineage"
    with pytest.raises(ValueError, match="embeds payload content"):
        archive.append(embedded, payload, admitted_tick=0)

    nested = _envelope("event:nested-bad", payload)
    nested["source_and_provenance"] = {
        "adapter": "fixture-v1",
        "metadata": [{"payload_bytes": payload.decode("ascii")}],
    }
    with pytest.raises(ValueError, match=r"\$\.source_and_provenance\.metadata\[0\]\.payload_bytes"):
        archive.append(nested, payload, admitted_tick=0)

    wrong_digest = _envelope("event:wrong", b"other")
    with pytest.raises(ValueError, match="does not match"):
        archive.append(wrong_digest, payload, admitted_tick=0)

    archive.append(_envelope("event:unique", payload), payload, admitted_tick=0)
    with pytest.raises(ValueError, match="duplicate"):
        archive.append(_envelope("event:unique", payload), payload, admitted_tick=1)
    archive.advance(2)
    with pytest.raises(ValueError, match="cannot move backwards"):
        archive.advance(1)
