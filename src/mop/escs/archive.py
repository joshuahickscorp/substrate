"""Bounded archival mechanics for the Event-Sourced Coalition Substrate.

This module deliberately separates immutable event-envelope lineage from replayable payload bytes.
Payloads begin in a byte- and tick-bounded hot journal and move into independently erasable,
content-addressed segments.  Compaction snapshots commit to envelope and segment roots but contain no
payload bytes.

``erase_payload`` establishes *logical non-retrievability* inside this archive: the payload segment,
payload index, and payload cache are removed, and exact replay authority is irreversibly disabled.  It
does not claim secure deletion from physical media, backups, allocator slack, or storage-controller
caches.  The deletion marker contains no payload, payload digest, segment root, or free-form reason text.

These are mechanics, not a substitute for the ESCS deletion CommitmentEvent and ConsequenceEvent.
Callers remain responsible for recording those lifecycle events in the immutable event plane.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

ARCHIVE_SCHEMA = "mop-escs-bounded-archive/v1"
SEGMENT_SCHEMA = "mop-escs-payload-segment/v1"
SNAPSHOT_SCHEMA = "mop-escs-compaction-snapshot/v1"
DELETION_MARKER_SCHEMA = "mop-escs-deletion-marker/v1"

_SEGMENT_MAGIC = b"MOP-ESCS-SEGMENT-V1\n"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_FORBIDDEN_ENVELOPE_KEYS = frozenset({"content", "payload", "payload_b64", "payload_bytes", "raw_payload"})
_ARCHIVE_OPERATIONS = ("append", "compaction", "erasure", "retention", "retrieval")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _forbidden_envelope_paths(value: Any, *, path: str = "$") -> tuple[str, ...]:
    """Find payload-bearing field names anywhere in retained lineage metadata.

    Checking only the top-level envelope would allow callers to retain an erasable payload under a
    nested provenance or metadata object.  This is deliberately a structural guard, not a semantic
    claim: an arbitrary string under an unrelated key can still encode content and must be controlled
    by the experiment's declared envelope schema.
    """

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_ENVELOPE_KEYS:
                found.append(nested_path)
            found.extend(_forbidden_envelope_paths(nested, path=nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_envelope_paths(nested, path=f"{path}[{index}]"))
    return tuple(found)


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


class ArchiveError(RuntimeError):
    """Base class for bounded-archive operation failures."""


class CorruptSegmentError(ArchiveError):
    """A stored payload segment failed its content or lineage commitments."""


class PayloadErasedError(ArchiveError):
    """A payload was logically erased and cannot be retrieved."""


class ReplayAuthorityError(ArchiveError):
    """Exact replay authority is unavailable after a payload erasure."""


class PayloadErasureError(ArchiveError):
    """The segment store did not establish logical non-retrievability."""


class ReplayAuthority(StrEnum):
    ENABLED = "enabled"
    DISABLED_AFTER_ERASURE = "disabled-after-erasure"


class PayloadTier(StrEnum):
    HOT = "hot"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class ArchiveCharge:
    """One narrow accounting observation emitted by archive mechanics.

    ``retained_byte_ticks`` is nonzero only for retention intervals. ``retained_bytes`` is a
    point-in-time total after ordinary operations and at the start of a retention interval.
    """

    operation: str
    work_units: int
    bytes_touched: int
    retained_bytes: int
    retained_byte_ticks: int = 0

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("archive accounting operation must not be empty")
        if (
            min(
                self.work_units,
                self.bytes_touched,
                self.retained_bytes,
                self.retained_byte_ticks,
            )
            < 0
        ):
            raise ValueError("archive accounting values must be nonnegative")


AccountingHook = Callable[[ArchiveCharge], None]


@dataclass(frozen=True, slots=True)
class ArchiveAccountingSnapshot:
    """Fixed-shape cumulative accounting used for exactly-once chassis reconciliation."""

    charge_count: int
    by_operation: tuple[tuple[str, int, int, int], ...]
    observer_failed: bool

    def __post_init__(self) -> None:
        if self.charge_count < 0:
            raise ValueError("archive charge_count must be nonnegative")
        if tuple(row[0] for row in self.by_operation) != _ARCHIVE_OPERATIONS:
            raise ValueError("archive accounting operations must use canonical order")
        if any(min(row[1:]) < 0 for row in self.by_operation):
            raise ValueError("archive accounting totals must be nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "charge_count": self.charge_count,
            "by_operation": {
                operation: {
                    "work_units": work_units,
                    "bytes_touched": bytes_touched,
                    "retained_byte_ticks": retained_byte_ticks,
                }
                for operation, work_units, bytes_touched, retained_byte_ticks in self.by_operation
            },
            "observer_failed": self.observer_failed,
        }


class SegmentStore(Protocol):
    """Minimal logical store required by the archive.

    ``delete`` means that subsequent ``contains`` and ``read`` calls cannot recover the object. It
    intentionally says nothing about deletion from physical media.
    """

    def put(self, root_sha256: str, segment: bytes) -> None: ...

    def read(self, root_sha256: str) -> bytes: ...

    def delete(self, root_sha256: str) -> bool: ...

    def contains(self, root_sha256: str) -> bool: ...


class InMemorySegmentStore:
    """Small immutable content-addressed store used by the mechanics scaffold."""

    def __init__(self) -> None:
        self._segments: dict[str, bytes] = {}

    def put(self, root_sha256: str, segment: bytes) -> None:
        _require_sha256(root_sha256, "segment root")
        raw = bytes(segment)
        if _sha256(raw) != root_sha256:
            raise ValueError("segment bytes do not match their content address")
        existing = self._segments.get(root_sha256)
        if existing is not None and existing != raw:
            raise ValueError("content-address collision with different segment bytes")
        self._segments[root_sha256] = raw

    def read(self, root_sha256: str) -> bytes:
        try:
            return self._segments[root_sha256]
        except KeyError as exc:
            raise KeyError(f"payload segment {root_sha256} is unavailable") from exc

    def delete(self, root_sha256: str) -> bool:
        return self._segments.pop(root_sha256, None) is not None

    def contains(self, root_sha256: str) -> bool:
        return root_sha256 in self._segments


@dataclass(frozen=True, slots=True)
class EnvelopeLineage:
    """Immutable canonical envelope retained independently of its payload."""

    sequence: int
    admitted_tick: int
    event_id: str
    envelope_canonical: bytes
    envelope_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.admitted_tick < 0:
            raise ValueError("lineage sequence and admitted tick must be nonnegative")
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        _require_sha256(self.envelope_sha256, "envelope_sha256")
        _require_sha256(self.payload_sha256, "payload_sha256")
        if _sha256(self.envelope_canonical) != self.envelope_sha256:
            raise ValueError("immutable envelope digest mismatch")
        try:
            decoded = json.loads(self.envelope_canonical)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("immutable envelope is not valid canonical JSON") from exc
        if _canonical_bytes(decoded) != self.envelope_canonical:
            raise ValueError("immutable envelope bytes are not canonical JSON")
        if not isinstance(decoded, dict):
            raise ValueError("immutable envelope must be a JSON mapping")
        if decoded.get("event_id") != self.event_id:
            raise ValueError("immutable envelope event identity drift")
        if decoded.get("payload_digest") != self.payload_sha256:
            raise ValueError("immutable envelope payload commitment drift")

    def envelope(self) -> dict[str, Any]:
        value = json.loads(self.envelope_canonical)
        if not isinstance(value, dict):  # guarded by construction; keeps the return type exact
            raise AssertionError("validated envelope stopped being a mapping")
        return value

    def commitment_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "envelope_sha256": self.envelope_sha256,
        }


@dataclass(frozen=True, slots=True)
class PayloadLocation:
    tier: PayloadTier
    sequence: int
    segment_root_sha256: str | None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("payload location sequence must be nonnegative")
        if self.tier is PayloadTier.HOT and self.segment_root_sha256 is not None:
            raise ValueError("hot payloads do not yet have a cold segment root")
        if self.tier is PayloadTier.COLD:
            if self.segment_root_sha256 is None:
                raise ValueError("cold payloads require a segment root")
            _require_sha256(self.segment_root_sha256, "segment_root_sha256")


@dataclass(frozen=True, slots=True)
class CompactionRow:
    sequence: int
    event_id: str
    envelope_sha256: str
    segment_root_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0 or not self.event_id:
            raise ValueError("compaction row identity is invalid")
        _require_sha256(self.envelope_sha256, "compaction envelope root")
        _require_sha256(self.segment_root_sha256, "compaction segment root")

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "envelope_sha256": self.envelope_sha256,
            "segment_root_sha256": self.segment_root_sha256,
        }


def _snapshot_core(rows: tuple[CompactionRow, ...]) -> dict[str, Any]:
    lineage_rows = [
        {
            "sequence": row.sequence,
            "event_id": row.event_id,
            "envelope_sha256": row.envelope_sha256,
        }
        for row in rows
    ]
    segment_rows = [
        {
            "sequence": row.sequence,
            "event_id": row.event_id,
            "segment_root_sha256": row.segment_root_sha256,
        }
        for row in rows
    ]
    return {
        "schema": SNAPSHOT_SCHEMA,
        "through_sequence": rows[-1].sequence if rows else None,
        "compacted_event_count": len(rows),
        "lineage_root_sha256": _sha256(_canonical_bytes(lineage_rows)),
        "segment_commitment_root_sha256": _sha256(_canonical_bytes(segment_rows)),
        "rows": [row.payload() for row in rows],
    }


@dataclass(frozen=True, slots=True)
class CompactionSnapshot:
    """Cumulative deterministic commitment to every compacted envelope and segment root."""

    rows: tuple[CompactionRow, ...]
    through_sequence: int | None
    lineage_root_sha256: str
    segment_commitment_root_sha256: str
    snapshot_sha256: str
    schema: str = SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported compaction snapshot schema")
        sequences = [row.sequence for row in self.rows]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("compaction rows must have unique increasing sequences")
        expected = _snapshot_core(self.rows)
        if self.through_sequence != expected["through_sequence"]:
            raise ValueError("compaction snapshot boundary drift")
        if self.lineage_root_sha256 != expected["lineage_root_sha256"]:
            raise ValueError("compaction lineage root drift")
        if self.segment_commitment_root_sha256 != expected["segment_commitment_root_sha256"]:
            raise ValueError("compaction segment commitment root drift")
        if self.snapshot_sha256 != _sha256(_canonical_bytes(expected)):
            raise ValueError("compaction snapshot root drift")

    @classmethod
    def create(cls, rows: tuple[CompactionRow, ...]) -> CompactionSnapshot:
        core = _snapshot_core(rows)
        return cls(
            rows=rows,
            through_sequence=core["through_sequence"],
            lineage_root_sha256=core["lineage_root_sha256"],
            segment_commitment_root_sha256=core["segment_commitment_root_sha256"],
            snapshot_sha256=_sha256(_canonical_bytes(core)),
        )

    def payload(self) -> dict[str, Any]:
        return {**_snapshot_core(self.rows), "snapshot_sha256": self.snapshot_sha256}


@dataclass(frozen=True, slots=True)
class DeletionMarker:
    """Non-content-bearing lineage fact left after logical payload erasure."""

    event_id: str
    sequence: int
    deletion_tick: int
    reason_code: str
    logical_non_retrievability: bool = True
    physical_media_erasure_claimed: bool = False
    schema: str = DELETION_MARKER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DELETION_MARKER_SCHEMA:
            raise ValueError("unsupported deletion marker schema")
        if not self.event_id or self.sequence < 0 or self.deletion_tick < 0:
            raise ValueError("deletion marker identity is invalid")
        if _REASON_CODE_RE.fullmatch(self.reason_code) is None:
            raise ValueError("deletion reason must be a bounded non-content-bearing reason code")
        if not self.logical_non_retrievability or self.physical_media_erasure_claimed:
            raise ValueError("deletion marker semantics cannot claim physical-media erasure")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "deletion_tick": self.deletion_tick,
            "reason_code": self.reason_code,
            "logical_non_retrievability": self.logical_non_retrievability,
            "physical_media_erasure_claimed": self.physical_media_erasure_claimed,
        }


def _encode_segment(lineage: EnvelopeLineage, payload: bytes) -> tuple[str, bytes]:
    header = _canonical_bytes(
        {
            "schema": SEGMENT_SCHEMA,
            "sequence": lineage.sequence,
            "event_id": lineage.event_id,
            "envelope_sha256": lineage.envelope_sha256,
            "payload_sha256": lineage.payload_sha256,
            "payload_bytes": len(payload),
        }
    )
    segment = _SEGMENT_MAGIC + len(header).to_bytes(8, "big") + header + payload
    return _sha256(segment), segment


def _decode_segment(raw: bytes, lineage: EnvelopeLineage, expected_root: str) -> bytes:
    segment = bytes(raw)
    if _sha256(segment) != expected_root:
        raise CorruptSegmentError("payload segment content-address mismatch")
    prefix_bytes = len(_SEGMENT_MAGIC) + 8
    if len(segment) < prefix_bytes or not segment.startswith(_SEGMENT_MAGIC):
        raise CorruptSegmentError("payload segment magic or length is invalid")
    header_bytes = int.from_bytes(segment[len(_SEGMENT_MAGIC) : prefix_bytes], "big")
    header_end = prefix_bytes + header_bytes
    if header_bytes < 2 or header_end > len(segment):
        raise CorruptSegmentError("payload segment header length is invalid")
    try:
        header = json.loads(segment[prefix_bytes:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptSegmentError("payload segment header is invalid JSON") from exc
    try:
        canonical_header = _canonical_bytes(header)
    except (TypeError, ValueError) as exc:
        raise CorruptSegmentError("payload segment header is outside strict JSON") from exc
    if not isinstance(header, dict) or canonical_header != segment[prefix_bytes:header_end]:
        raise CorruptSegmentError("payload segment header is not canonical")
    expected_keys = {
        "schema",
        "sequence",
        "event_id",
        "envelope_sha256",
        "payload_sha256",
        "payload_bytes",
    }
    if set(header) != expected_keys:
        raise CorruptSegmentError("payload segment header fields drifted")
    payload = segment[header_end:]
    expected_header = {
        "schema": SEGMENT_SCHEMA,
        "sequence": lineage.sequence,
        "event_id": lineage.event_id,
        "envelope_sha256": lineage.envelope_sha256,
        "payload_sha256": lineage.payload_sha256,
        "payload_bytes": len(payload),
    }
    if header != expected_header:
        raise CorruptSegmentError("payload segment lineage or byte-count mismatch")
    if _sha256(payload) != lineage.payload_sha256:
        raise CorruptSegmentError("payload bytes do not match the immutable envelope commitment")
    return payload


class BoundedArchive:
    """In-process ESCS archive with bounded hot state and independently erasable cold segments."""

    def __init__(
        self,
        *,
        max_hot_bytes: int,
        max_hot_age_ticks: int,
        max_cache_bytes: int | None = None,
        max_snapshots: int = 8,
        segment_store: SegmentStore | None = None,
        accounting_hook: AccountingHook | None = None,
    ) -> None:
        if max_hot_bytes < 1:
            raise ValueError("max_hot_bytes must be positive")
        if max_hot_age_ticks < 0:
            raise ValueError("max_hot_age_ticks must be nonnegative")
        resolved_cache_bytes = max_hot_bytes if max_cache_bytes is None else max_cache_bytes
        if resolved_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be nonnegative")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        self.max_hot_bytes = max_hot_bytes
        self.max_hot_age_ticks = max_hot_age_ticks
        self.max_cache_bytes = resolved_cache_bytes
        self.max_snapshots = max_snapshots
        self._store = segment_store if segment_store is not None else InMemorySegmentStore()
        self._accounting_hook = accounting_hook

        self._lineage_by_event: dict[str, EnvelopeLineage] = {}
        self._lineage_by_sequence: list[EnvelopeLineage] = []
        self._hot_order: deque[int] = deque()
        self._hot_payloads: dict[int, bytes] = {}
        self._hot_bytes = 0
        self._payload_index: dict[str, PayloadLocation] = {}
        self._payload_cache: dict[str, bytes] = {}
        self._cache_order: deque[str] = deque()
        self._cache_bytes = 0
        self._segment_sizes: dict[str, int] = {}
        self._compacted_rows: list[CompactionRow] = []
        self._snapshots: dict[str, CompactionSnapshot] = {}
        self._deletion_markers: dict[str, DeletionMarker] = {}
        self._erased_segment_roots: dict[str, str] = {}
        self._charge_count = 0
        self._accounting_totals: dict[str, list[int]] = {
            operation: [0, 0, 0] for operation in _ARCHIVE_OPERATIONS
        }
        self._accounting_hook_failed = False
        self._replay_authority = ReplayAuthority.ENABLED
        self._last_tick = 0

    @property
    def replay_authority(self) -> ReplayAuthority:
        return self._replay_authority

    @property
    def hot_bytes(self) -> int:
        return self._hot_bytes

    @property
    def hot_event_ids(self) -> tuple[str, ...]:
        return tuple(self._lineage_by_sequence[sequence].event_id for sequence in self._hot_order)

    @property
    def payload_index_event_ids(self) -> frozenset[str]:
        return frozenset(self._payload_index)

    @property
    def cached_event_ids(self) -> frozenset[str]:
        return frozenset(self._payload_cache)

    @property
    def lineages(self) -> tuple[EnvelopeLineage, ...]:
        return tuple(self._lineage_by_sequence)

    @property
    def snapshots(self) -> tuple[CompactionSnapshot, ...]:
        return tuple(self._snapshots.values())

    @property
    def accounting_snapshot(self) -> ArchiveAccountingSnapshot:
        """Return fixed-shape cumulative totals; callers reconcile monotone deltas exactly once."""

        rows: list[tuple[str, int, int, int]] = []
        for operation in _ARCHIVE_OPERATIONS:
            work_units, bytes_touched, retained_byte_ticks = self._accounting_totals[operation]
            rows.append((operation, work_units, bytes_touched, retained_byte_ticks))
        return ArchiveAccountingSnapshot(
            charge_count=self._charge_count,
            by_operation=tuple(rows),
            observer_failed=self._accounting_hook_failed,
        )

    @property
    def latest_snapshot(self) -> CompactionSnapshot | None:
        return next(reversed(self._snapshots.values()), None) if self._snapshots else None

    @property
    def retained_bytes(self) -> int:
        lineage_bytes = sum(
            len(
                _canonical_bytes(
                    {
                        "schema": ARCHIVE_SCHEMA,
                        "sequence": row.sequence,
                        "admitted_tick": row.admitted_tick,
                        "event_id": row.event_id,
                        "envelope": row.envelope(),
                        "envelope_sha256": row.envelope_sha256,
                    }
                )
            )
            for row in self._lineage_by_sequence
        )
        hot_payload_bytes = sum(len(payload) for payload in self._hot_payloads.values())
        cold_segment_bytes = sum(self._segment_sizes.values())
        cache_bytes = self._cache_bytes
        index_bytes = len(
            _canonical_bytes(
                {
                    event_id: {
                        "tier": str(location.tier),
                        "sequence": location.sequence,
                        "segment_root_sha256": location.segment_root_sha256,
                    }
                    for event_id, location in sorted(self._payload_index.items())
                }
            )
        )
        snapshot_bytes = sum(
            len(_canonical_bytes(snapshot.payload())) for snapshot in self._snapshots.values()
        )
        marker_bytes = sum(
            len(_canonical_bytes(marker.payload())) for marker in self._deletion_markers.values()
        )
        erased_root_bytes = len(_canonical_bytes(sorted(self._erased_segment_roots.items())))
        accounting_bytes = len(_canonical_bytes(self.accounting_snapshot.payload()))
        return (
            lineage_bytes
            + hot_payload_bytes
            + cold_segment_bytes
            + cache_bytes
            + index_bytes
            + snapshot_bytes
            + marker_bytes
            + erased_root_bytes
            + accounting_bytes
        )

    def lineage(self, event_id: str) -> EnvelopeLineage:
        try:
            return self._lineage_by_event[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown archive event {event_id!r}") from exc

    def payload_location(self, event_id: str) -> PayloadLocation | None:
        return self._payload_index.get(event_id)

    def deletion_marker(self, event_id: str) -> DeletionMarker | None:
        return self._deletion_markers.get(event_id)

    def append(
        self,
        envelope: Mapping[str, Any],
        payload: bytes | bytearray | memoryview,
        *,
        admitted_tick: int,
    ) -> EnvelopeLineage:
        """Append one immutable envelope and replayable payload, then enforce both hot bounds."""

        if not isinstance(envelope, Mapping):
            raise TypeError("archive envelope must be a mapping")
        envelope_dict = dict(envelope)
        forbidden = sorted(_forbidden_envelope_paths(envelope_dict))
        if forbidden:
            raise ValueError(f"archive envelope embeds payload content in forbidden fields: {forbidden}")
        event_id = envelope_dict.get("event_id")
        payload_sha256 = envelope_dict.get("payload_digest")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("archive envelope requires a nonempty string event_id")
        if not isinstance(payload_sha256, str):
            raise ValueError("archive envelope requires payload_digest")
        _require_sha256(payload_sha256, "payload_digest")
        if event_id in self._lineage_by_event:
            raise ValueError(f"duplicate archive event_id {event_id!r}")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("archive payload must be bytes-like")
        payload_bytes = bytes(payload)
        if _sha256(payload_bytes) != payload_sha256:
            raise ValueError("payload does not match the immutable envelope payload_digest")

        try:
            envelope_bytes = _canonical_bytes(envelope_dict)
        except (TypeError, ValueError) as exc:
            raise ValueError("archive envelope must contain strict JSON values") from exc
        self._advance_time(admitted_tick)
        sequence = len(self._lineage_by_sequence)
        lineage = EnvelopeLineage(
            sequence=sequence,
            admitted_tick=admitted_tick,
            event_id=event_id,
            envelope_canonical=envelope_bytes,
            envelope_sha256=_sha256(envelope_bytes),
            payload_sha256=payload_sha256,
        )
        self._lineage_by_event[event_id] = lineage
        self._lineage_by_sequence.append(lineage)
        self._hot_order.append(sequence)
        self._hot_payloads[sequence] = payload_bytes
        self._hot_bytes += len(envelope_bytes) + len(payload_bytes)
        self._payload_index[event_id] = PayloadLocation(PayloadTier.HOT, sequence, None)
        self._charge(
            "append",
            work_units=1,
            bytes_touched=len(envelope_bytes) + len(payload_bytes),
        )
        self._compact_eligible(force=False)
        return lineage

    def advance(self, current_tick: int) -> CompactionSnapshot | None:
        """Advance archive time, charge retained byte-time, and compact newly expired entries."""

        self._advance_time(current_tick)
        return self._compact_eligible(force=False)

    def compact(self, current_tick: int, *, force: bool = False) -> CompactionSnapshot | None:
        """Enforce the hot bounds, or deterministically compact the whole hot journal."""

        self._advance_time(current_tick)
        return self._compact_eligible(force=force)

    def retrieve(self, event_id: str, *, current_tick: int | None = None) -> bytes:
        """Retrieve a payload only after checking its immutable digest and cold segment root."""

        if current_tick is not None:
            self.advance(current_tick)
        if event_id in self._deletion_markers:
            raise PayloadErasedError(f"payload for {event_id!r} was logically erased")
        try:
            lineage = self._lineage_by_event[event_id]
            location = self._payload_index[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown or unavailable archive payload {event_id!r}") from exc

        bytes_touched = 0
        try:
            if location.tier is PayloadTier.HOT:
                try:
                    payload = self._hot_payloads[location.sequence]
                except KeyError as exc:
                    raise CorruptSegmentError("hot payload index points to missing content") from exc
                bytes_touched = len(payload)
                if _sha256(payload) != lineage.payload_sha256:
                    raise CorruptSegmentError("hot payload does not match its envelope commitment")
            else:
                root = location.segment_root_sha256
                if root is None:
                    raise CorruptSegmentError("cold payload index is missing its segment root")
                try:
                    raw = bytes(self._store.read(root))
                except (KeyError, OSError) as exc:
                    raise CorruptSegmentError(f"cold payload segment {root} is unavailable") from exc
                bytes_touched = len(raw)
                payload = _decode_segment(raw, lineage, root)
            self._cache_payload(event_id, payload)
        except (CorruptSegmentError, KeyError):
            self._drop_cache(event_id)
            self._charge("retrieval", work_units=1, bytes_touched=bytes_touched)
            raise
        self._charge("retrieval", work_units=1, bytes_touched=bytes_touched)
        return bytes(payload)

    def erase_payload(
        self,
        event_id: str,
        *,
        deletion_tick: int,
        reason_code: str = "retention-expired",
    ) -> DeletionMarker:
        """Remove retrievable payload state without claiming physical-media erasure."""

        self._advance_time(deletion_tick)
        self._compact_eligible(force=False)
        if event_id in self._deletion_markers:
            raise PayloadErasedError(f"payload for {event_id!r} was already logically erased")
        try:
            lineage = self._lineage_by_event[event_id]
            location = self._payload_index[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown or unavailable archive payload {event_id!r}") from exc
        marker = DeletionMarker(
            event_id=event_id,
            sequence=lineage.sequence,
            deletion_tick=deletion_tick,
            reason_code=reason_code,
        )

        bytes_touched = len(self._payload_cache.get(event_id, b""))
        if location.tier is PayloadTier.HOT:
            payload = self._hot_payloads.pop(location.sequence)
            bytes_touched += len(payload)
            self._hot_order.remove(location.sequence)
            self._hot_bytes -= len(lineage.envelope_canonical) + len(payload)
        else:
            root = location.segment_root_sha256
            if root is None:
                raise PayloadErasureError("cold payload index has no segment root")
            bytes_touched += self._segment_sizes.get(root, 0)
            self._store.delete(root)
            residual_readable = False
            try:
                self._store.read(root)
            except (KeyError, OSError):
                pass
            else:
                residual_readable = True
            if self._store.contains(root) or residual_readable:
                raise PayloadErasureError("segment store did not establish logical non-retrievability")
            self._erased_segment_roots[event_id] = root
            self._segment_sizes.pop(root, None)

        self._payload_index.pop(event_id, None)
        self._drop_cache(event_id)
        self._deletion_markers[event_id] = marker
        self._replay_authority = ReplayAuthority.DISABLED_AFTER_ERASURE
        self._charge("erasure", work_units=1, bytes_touched=bytes_touched)
        return marker

    def require_replay_authority(self) -> None:
        if self._replay_authority is not ReplayAuthority.ENABLED:
            raise ReplayAuthorityError(
                "exact archive replay authority was disabled by logical payload erasure"
            )

    def audit(self) -> tuple[str, ...]:
        """Return integrity and boundedness violations without caching retrieved content."""

        problems: list[str] = []
        computed_hot_bytes = 0
        hot_sequences = set(self._hot_order)
        if len(hot_sequences) != len(self._hot_order):
            problems.append("hot journal contains duplicate sequence entries")
        for expected_sequence, lineage in enumerate(self._lineage_by_sequence):
            if lineage.sequence != expected_sequence:
                problems.append(f"lineage sequence drift at {expected_sequence}")
            if self._lineage_by_event.get(lineage.event_id) != lineage:
                problems.append(f"lineage event index drift for {lineage.event_id}")
            marker = self._deletion_markers.get(lineage.event_id)
            location = self._payload_index.get(lineage.event_id)
            if marker is not None:
                if location is not None or lineage.event_id in self._payload_cache:
                    problems.append(f"erased payload state remains indexed for {lineage.event_id}")
                erased_root = self._erased_segment_roots.get(lineage.event_id)
                if erased_root is not None:
                    residual_readable = self._store.contains(erased_root)
                    if not residual_readable:
                        try:
                            self._store.read(erased_root)
                        except (KeyError, OSError):
                            pass
                        else:
                            residual_readable = True
                    if residual_readable:
                        problems.append(f"erased segment resurrected for {lineage.event_id}")
                continue
            if location is None:
                problems.append(f"nonerased payload lacks an index for {lineage.event_id}")
                continue
            if location.sequence != lineage.sequence:
                problems.append(f"payload index sequence drift for {lineage.event_id}")
            if location.tier is PayloadTier.HOT:
                payload = self._hot_payloads.get(lineage.sequence)
                if payload is None or lineage.sequence not in hot_sequences:
                    problems.append(f"hot payload state missing for {lineage.event_id}")
                    continue
                computed_hot_bytes += len(lineage.envelope_canonical) + len(payload)
                if _sha256(payload) != lineage.payload_sha256:
                    problems.append(f"hot payload commitment drift for {lineage.event_id}")
            else:
                root = location.segment_root_sha256
                if root is None:
                    problems.append(f"cold payload root missing for {lineage.event_id}")
                    continue
                try:
                    raw = self._store.read(root)
                    _decode_segment(raw, lineage, root)
                except (KeyError, OSError, CorruptSegmentError) as exc:
                    problems.append(f"cold segment rejected for {lineage.event_id}: {exc}")
        if computed_hot_bytes != self._hot_bytes:
            problems.append("hot journal byte accounting drift")
        if self._hot_bytes > self.max_hot_bytes:
            problems.append("hot journal exceeds its byte bound")
        if self._hot_order:
            oldest = self._lineage_by_sequence[self._hot_order[0]]
            if self._last_tick - oldest.admitted_tick > self.max_hot_age_ticks:
                problems.append("hot journal exceeds its time bound")
        if sum(len(payload) for payload in self._payload_cache.values()) != self._cache_bytes:
            problems.append("payload cache byte accounting drift")
        if set(self._cache_order) != set(self._payload_cache) or len(self._cache_order) != len(
            self._payload_cache
        ):
            problems.append("payload cache eviction order drift")
        if self._cache_bytes > self.max_cache_bytes:
            problems.append("payload cache exceeds its byte bound")
        if self._deletion_markers and self._replay_authority is not ReplayAuthority.DISABLED_AFTER_ERASURE:
            problems.append("payload erasure did not disable exact replay authority")
        if self._accounting_hook_failed:
            problems.append("archive accounting observer failed")
        for snapshot in self._snapshots.values():
            try:
                CompactionSnapshot(
                    rows=snapshot.rows,
                    through_sequence=snapshot.through_sequence,
                    lineage_root_sha256=snapshot.lineage_root_sha256,
                    segment_commitment_root_sha256=snapshot.segment_commitment_root_sha256,
                    snapshot_sha256=snapshot.snapshot_sha256,
                    schema=snapshot.schema,
                )
            except ValueError as exc:
                problems.append(f"compaction snapshot rejected: {exc}")
        return tuple(problems)

    def _drop_cache(self, event_id: str) -> None:
        payload = self._payload_cache.pop(event_id, None)
        if payload is None:
            return
        self._cache_bytes -= len(payload)
        self._cache_order.remove(event_id)

    def _cache_payload(self, event_id: str, payload: bytes) -> None:
        self._drop_cache(event_id)
        if len(payload) > self.max_cache_bytes:
            return
        self._payload_cache[event_id] = payload
        self._cache_order.append(event_id)
        self._cache_bytes += len(payload)
        while self._cache_bytes > self.max_cache_bytes:
            self._drop_cache(self._cache_order[0])

    def _advance_time(self, tick: int) -> None:
        if tick < 0:
            raise ValueError("archive ticks must be nonnegative")
        if tick < self._last_tick:
            raise ValueError("archive time cannot move backwards")
        elapsed = tick - self._last_tick
        if elapsed:
            retained = self.retained_bytes
            self._charge(
                "retention",
                work_units=0,
                bytes_touched=0,
                retained_bytes=retained,
                retained_byte_ticks=retained * elapsed,
            )
        self._last_tick = tick

    def _compact_eligible(self, *, force: bool) -> CompactionSnapshot | None:
        compacted = 0
        bytes_touched = 0
        while self._hot_order:
            sequence = self._hot_order[0]
            lineage = self._lineage_by_sequence[sequence]
            expired = self._last_tick - lineage.admitted_tick > self.max_hot_age_ticks
            over_bytes = self._hot_bytes > self.max_hot_bytes
            if not force and not expired and not over_bytes:
                break
            payload = self._hot_payloads[sequence]
            root, segment = _encode_segment(lineage, payload)
            self._store.put(root, segment)
            try:
                stored_segment = bytes(self._store.read(root))
            except (KeyError, OSError) as exc:
                raise ArchiveError("segment store did not retain a compacted payload") from exc
            if not self._store.contains(root) or stored_segment != segment:
                raise ArchiveError("segment store changed a compacted payload during publication")
            self._hot_order.popleft()
            self._hot_payloads.pop(sequence)
            self._hot_bytes -= len(lineage.envelope_canonical) + len(payload)
            self._payload_index[lineage.event_id] = PayloadLocation(PayloadTier.COLD, sequence, root)
            self._segment_sizes[root] = len(segment)
            self._compacted_rows.append(
                CompactionRow(sequence, lineage.event_id, lineage.envelope_sha256, root)
            )
            compacted += 1
            bytes_touched += len(payload) + 2 * len(segment)
        if not compacted:
            return None
        snapshot = CompactionSnapshot.create(tuple(self._compacted_rows))
        self._snapshots.setdefault(snapshot.snapshot_sha256, snapshot)
        while len(self._snapshots) > self.max_snapshots:
            del self._snapshots[next(iter(self._snapshots))]
        bytes_touched += len(_canonical_bytes(snapshot.payload()))
        self._charge("compaction", work_units=compacted, bytes_touched=bytes_touched)
        return snapshot

    def _charge(
        self,
        operation: str,
        *,
        work_units: int,
        bytes_touched: int,
        retained_bytes: int | None = None,
        retained_byte_ticks: int = 0,
    ) -> None:
        if operation not in self._accounting_totals:
            raise ValueError(f"unsupported archive accounting operation {operation!r}")
        charge = ArchiveCharge(
            operation=operation,
            work_units=work_units,
            bytes_touched=bytes_touched,
            retained_bytes=self.retained_bytes if retained_bytes is None else retained_bytes,
            retained_byte_ticks=retained_byte_ticks,
        )
        self._charge_count += 1
        totals = self._accounting_totals[operation]
        totals[0] += charge.work_units
        totals[1] += charge.bytes_touched
        totals[2] += charge.retained_byte_ticks
        if self._accounting_hook is None:
            return
        try:
            self._accounting_hook(charge)
        except Exception:
            # The cumulative snapshot remains authoritative and the archive audit fails closed.  A
            # notification callback cannot roll back a segment-store mutation that already happened.
            self._accounting_hook_failed = True
