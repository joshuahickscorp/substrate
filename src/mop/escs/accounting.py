
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Self

from mop.substrate.events import BranchRef, EventRef, canonical_sha256

ACCOUNTING_SCHEMA = "mop-escs-accounting/v1"
LIFECYCLE_LEDGER_SCHEMA = "mop-escs-lifecycle-ledger/v1"
FACTUAL_BRANCH = BranchRef("branch:factual")


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class WorkVector:

    raw_transport_and_adapters: int = 0
    event_formation: int = 0
    indexing_and_graph_maintenance: int = 0
    dispatch_and_exploration: int = 0
    actor_execution: int = 0
    messages: int = 0
    counterfactual_credit: int = 0
    learning: int = 0
    archival_and_erasure: int = 0
    retained_byte_time: int = 0
    idle_floor: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_nonnegative_int(getattr(self, field.name), f"WorkVector.{field.name}")

    @classmethod
    def zero(cls) -> Self:
        return cls()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("WorkVector payload must be a mapping")
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, "WorkVector")
        return cls(**{name: _require_nonnegative_int(payload[name], name) for name in expected})

    @classmethod
    def retention(cls, *, retained_bytes: int, start_tick: int, end_tick: int) -> Self:

        retained_bytes = _require_nonnegative_int(retained_bytes, "retained_bytes")
        start_tick = _require_nonnegative_int(start_tick, "start_tick")
        end_tick = _require_nonnegative_int(end_tick, "end_tick")
        if end_tick < start_tick:
            raise ValueError("retention end_tick cannot precede start_tick")
        return cls(retained_byte_time=retained_bytes * (end_tick - start_tick))

    @property
    def total_work(self) -> int:
        return (
            self.raw_transport_and_adapters
            + self.event_formation
            + self.indexing_and_graph_maintenance
            + self.dispatch_and_exploration
            + self.actor_execution
            + self.messages
            + self.counterfactual_credit
            + self.learning
            + self.archival_and_erasure
            + self.idle_floor
        )

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, field.name) == 0 for field in fields(self))

    def __add__(self, other: object) -> Self:
        if not isinstance(other, WorkVector):
            return NotImplemented
        return type(self)(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)}
        )

    def __radd__(self, other: object) -> Self:
        if other == 0:
            return self
        if isinstance(other, WorkVector):
            return other + self
        return NotImplemented

    def scale(self, multiplier: int) -> Self:
        multiplier = _require_nonnegative_int(multiplier, "multiplier")
        return type(self)(**{field.name: getattr(self, field.name) * multiplier for field in fields(self)})

    def payload(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class LifecycleCharge:

    sequence: int
    owner: str
    reason: str
    start_tick: int
    end_tick: int
    branch_id: BranchRef
    causal_event_ids: tuple[EventRef, ...]
    work: WorkVector
    previous_charge_sha256: str | None
    charge_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.sequence, "LifecycleCharge.sequence")
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("LifecycleCharge.owner must not be empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("LifecycleCharge.reason must not be empty")
        _require_nonnegative_int(self.start_tick, "LifecycleCharge.start_tick")
        _require_nonnegative_int(self.end_tick, "LifecycleCharge.end_tick")
        if self.end_tick < self.start_tick:
            raise ValueError("LifecycleCharge end_tick cannot precede start_tick")
        if not isinstance(self.branch_id, BranchRef):
            raise ValueError("LifecycleCharge.branch_id must be a BranchRef")
        if not all(isinstance(event_id, EventRef) for event_id in self.causal_event_ids):
            raise ValueError("LifecycleCharge.causal_event_ids must contain EventRef values")
        if tuple(sorted(self.causal_event_ids, key=str)) != self.causal_event_ids:
            raise ValueError("causal_event_ids must be in canonical sorted order")
        if len(set(self.causal_event_ids)) != len(self.causal_event_ids):
            raise ValueError("causal_event_ids must be unique")
        if self.previous_charge_sha256 is not None:
            _require_digest(self.previous_charge_sha256, "previous_charge_sha256")
        if not isinstance(self.work, WorkVector):
            raise ValueError("LifecycleCharge.work must be a WorkVector")
        _require_digest(self.charge_sha256, "charge_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.charge_sha256:
            raise ValueError("lifecycle charge digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        owner: str,
        reason: str,
        start_tick: int,
        end_tick: int,
        work: WorkVector,
        branch_id: BranchRef = FACTUAL_BRANCH,
        causal_event_ids: Sequence[EventRef] = (),
        previous_charge_sha256: str | None = None,
    ) -> Self:
        canonical_events = tuple(sorted(causal_event_ids, key=str))
        partial = {
            "schema": ACCOUNTING_SCHEMA,
            "sequence": sequence,
            "owner": owner,
            "reason": reason,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "branch_id": str(branch_id),
            "causal_event_ids": [str(event_id) for event_id in canonical_events],
            "work": work.payload(),
            "previous_charge_sha256": previous_charge_sha256,
        }
        return cls(
            sequence=sequence,
            owner=owner,
            reason=reason,
            start_tick=start_tick,
            end_tick=end_tick,
            branch_id=branch_id,
            causal_event_ids=canonical_events,
            work=work,
            previous_charge_sha256=previous_charge_sha256,
            charge_sha256=canonical_sha256(partial),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = {
            "schema",
            "sequence",
            "owner",
            "reason",
            "start_tick",
            "end_tick",
            "branch_id",
            "causal_event_ids",
            "work",
            "previous_charge_sha256",
            "charge_sha256",
        }
        _require_exact_keys(payload, expected, "LifecycleCharge")
        if payload["schema"] != ACCOUNTING_SCHEMA:
            raise ValueError(f"unsupported accounting schema {payload['schema']!r}")
        event_ids = payload["causal_event_ids"]
        if not isinstance(event_ids, list) or not all(isinstance(value, str) for value in event_ids):
            raise ValueError("causal_event_ids must be a list of event reference strings")
        return cls(
            sequence=payload["sequence"],
            owner=payload["owner"],
            reason=payload["reason"],
            start_tick=payload["start_tick"],
            end_tick=payload["end_tick"],
            branch_id=BranchRef(payload["branch_id"]),
            causal_event_ids=tuple(EventRef(value) for value in event_ids),
            work=WorkVector.from_payload(payload["work"]),
            previous_charge_sha256=payload["previous_charge_sha256"],
            charge_sha256=payload["charge_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ACCOUNTING_SCHEMA,
            "sequence": self.sequence,
            "owner": self.owner,
            "reason": self.reason,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "branch_id": str(self.branch_id),
            "causal_event_ids": [str(event_id) for event_id in self.causal_event_ids],
            "work": self.work.payload(),
            "previous_charge_sha256": self.previous_charge_sha256,
        }
        if include_digest:
            result["charge_sha256"] = self.charge_sha256
        return result


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class LifecycleLedger:

    def __init__(self) -> None:
        self._entries: list[LifecycleCharge] = []

    @property
    def entries(self) -> tuple[LifecycleCharge, ...]:
        return tuple(self._entries)

    @property
    def entry_count(self) -> int:

        return len(self._entries)

    @property
    def head_sha256(self) -> str | None:
        return self._entries[-1].charge_sha256 if self._entries else None

    @property
    def total(self) -> WorkVector:
        return sum((entry.work for entry in self._entries), WorkVector.zero())

    def charge(
        self,
        *,
        owner: str,
        reason: str,
        work: WorkVector,
        start_tick: int,
        end_tick: int,
        branch_id: BranchRef = FACTUAL_BRANCH,
        causal_event_ids: Sequence[EventRef] = (),
    ) -> LifecycleCharge:
        entry = LifecycleCharge.create(
            sequence=len(self._entries),
            owner=owner,
            reason=reason,
            start_tick=start_tick,
            end_tick=end_tick,
            branch_id=branch_id,
            causal_event_ids=causal_event_ids,
            work=work,
            previous_charge_sha256=self.head_sha256,
        )
        self._entries.append(entry)
        return entry

    def charge_retention(
        self,
        *,
        owner: str,
        reason: str,
        retained_bytes: int,
        start_tick: int,
        end_tick: int,
        branch_id: BranchRef = FACTUAL_BRANCH,
        causal_event_ids: Sequence[EventRef] = (),
    ) -> LifecycleCharge:
        return self.charge(
            owner=owner,
            reason=reason,
            work=WorkVector.retention(
                retained_bytes=retained_bytes,
                start_tick=start_tick,
                end_tick=end_tick,
            ),
            start_tick=start_tick,
            end_tick=end_tick,
            branch_id=branch_id,
            causal_event_ids=causal_event_ids,
        )

    def charge_idle(
        self,
        *,
        owner: str,
        reason: str,
        idle_work: int,
        start_tick: int,
        end_tick: int,
        branch_id: BranchRef = FACTUAL_BRANCH,
        causal_event_ids: Sequence[EventRef] = (),
    ) -> LifecycleCharge:
        idle_work = _require_nonnegative_int(idle_work, "idle_work")
        if idle_work == 0:
            raise ValueError("idle_work must be a positive integer")
        return self.charge(
            owner=owner,
            reason=reason,
            work=WorkVector(idle_floor=idle_work),
            start_tick=start_tick,
            end_tick=end_tick,
            branch_id=branch_id,
            causal_event_ids=causal_event_ids,
        )

    def verify(self, *, event_ids: set[str] | None = None) -> list[str]:
        problems: list[str] = []
        previous: str | None = None
        for sequence, entry in enumerate(self._entries):
            if entry.sequence != sequence:
                problems.append(f"charge {sequence} sequence drift")
            if entry.previous_charge_sha256 != previous:
                problems.append(f"charge {sequence} previous digest drift")
            if canonical_sha256(entry.payload(include_digest=False)) != entry.charge_sha256:
                problems.append(f"charge {sequence} digest drift")
            if event_ids is not None:
                for event_id in entry.causal_event_ids:
                    if str(event_id) not in event_ids:
                        problems.append(f"charge {sequence} references missing event {event_id}")
            previous = entry.charge_sha256
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_LEDGER_SCHEMA,
            "entries": [entry.payload() for entry in self._entries],
            "head_sha256": self.head_sha256,
            "total": self.total.payload(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = {"schema", "entries", "head_sha256", "total"}
        _require_exact_keys(payload, expected, "LifecycleLedger")
        if payload["schema"] != LIFECYCLE_LEDGER_SCHEMA:
            raise ValueError(f"unsupported lifecycle ledger schema {payload['schema']!r}")
        rows = payload["entries"]
        if not isinstance(rows, list):
            raise ValueError("LifecycleLedger.entries must be a list")
        parsed = [LifecycleCharge.from_payload(row) for row in rows]
        ledger = cls()
        for expected_sequence, row in enumerate(parsed):
            if row.sequence != expected_sequence or row.previous_charge_sha256 != ledger.head_sha256:
                raise ValueError(f"invalid lifecycle hash chain at charge {expected_sequence}")
            ledger._entries.append(row)
        if payload["head_sha256"] != ledger.head_sha256:
            raise ValueError("lifecycle ledger head digest mismatch")
        if WorkVector.from_payload(payload["total"]) != ledger.total:
            raise ValueError("lifecycle ledger total mismatch")
        if problems := ledger.verify():
            raise ValueError("invalid lifecycle ledger: " + "; ".join(problems))
        return ledger

    @classmethod
    def replay(cls, payload: Mapping[str, Any]) -> Self:
        return cls.from_payload(payload)
