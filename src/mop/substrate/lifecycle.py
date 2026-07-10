"""Append-only lifecycle journal with exact replay and availability queries.

Every append is a new revision linked to the prior entry digest. Rollback is also an append: old
entries remain intact, and the new revision restores a prior state. The journal is a bounded local
mechanics instrument, not a claim about memory outside its recorded event stream.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .events import EventRef, FrozenJSON, canonical_sha256

LIFECYCLE_SCHEMA = "mop-lifecycle-journal/v1"
_MEMORY_REF_RE = re.compile(r"^memory:[a-z0-9][a-z0-9._:/-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, order=True)
class MemoryRef:
    value: str

    def __post_init__(self) -> None:
        if _MEMORY_REF_RE.fullmatch(self.value) is None:
            raise ValueError("memory reference must use the memory namespace and stable characters")

    def __str__(self) -> str:
        return self.value


class LifecycleOperation(StrEnum):
    RECORD = "record"
    REVISE = "revise"
    AVAILABILITY = "availability"
    ROLLBACK = "rollback"
    DELETE = "delete"
    CONFLICT = "conflict"
    POISONING = "poisoning"


@dataclass(frozen=True, slots=True)
class MemoryState:
    memory_ref: MemoryRef
    revision: int
    content: FrozenJSON | None
    availability_enabled: bool
    available_from_tick: int | None
    available_until_tick: int | None
    deleted: bool = False
    conflicted: bool = False
    poisoned: bool = False
    rollback_to_revision: int | None = None

    @property
    def exists(self) -> bool:
        return self.content is not None and not self.deleted

    def available_at(self, tick: int) -> bool:
        if tick < 0:
            raise ValueError("availability tick must be nonnegative")
        if not self.exists or not self.availability_enabled or self.conflicted or self.poisoned:
            return False
        if self.available_from_tick is not None and tick < self.available_from_tick:
            return False
        return self.available_until_tick is None or tick <= self.available_until_tick

    def payload(self) -> dict[str, Any]:
        return {
            "memory_ref": str(self.memory_ref),
            "revision": self.revision,
            "content": self.content.payload() if self.content is not None else None,
            "availability_enabled": self.availability_enabled,
            "available_from_tick": self.available_from_tick,
            "available_until_tick": self.available_until_tick,
            "deleted": self.deleted,
            "conflicted": self.conflicted,
            "poisoned": self.poisoned,
            "rollback_to_revision": self.rollback_to_revision,
        }


@dataclass(frozen=True, slots=True)
class LifecycleEntry:
    memory_ref: MemoryRef
    event_ref: EventRef
    sequence: int
    revision: int
    operation: LifecycleOperation
    content: FrozenJSON | None
    availability_enabled: bool | None
    available_from_tick: int | None
    available_until_tick: int | None
    target_revision: int | None
    reason: str
    previous_entry_sha256: str | None
    entry_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.revision != self.sequence + 1:
            raise ValueError("lifecycle revision must equal sequence plus one")
        if not self.reason.strip():
            raise ValueError("lifecycle entry reason must not be empty")
        if (
            self.previous_entry_sha256 is not None
            and _SHA256_RE.fullmatch(self.previous_entry_sha256) is None
        ):
            raise ValueError("previous lifecycle digest is invalid")
        if _SHA256_RE.fullmatch(self.entry_sha256) is None:
            raise ValueError("lifecycle entry digest is invalid")
        _validate_interval(self.available_from_tick, self.available_until_tick)
        self._validate_operation_fields()
        if canonical_sha256(self.payload(include_digest=False)) != self.entry_sha256:
            raise ValueError("lifecycle entry digest mismatch")

    def _validate_operation_fields(self) -> None:
        if self.operation in {LifecycleOperation.RECORD, LifecycleOperation.REVISE}:
            if self.content is None or self.availability_enabled is None:
                raise ValueError(f"{self.operation} requires content and availability")
            if self.target_revision is not None:
                raise ValueError(f"{self.operation} cannot target another revision")
        elif self.operation is LifecycleOperation.AVAILABILITY:
            if self.content is not None or self.availability_enabled is None:
                raise ValueError("availability entries require a boolean and no content")
            if self.target_revision is not None:
                raise ValueError("availability entries cannot target another revision")
        elif self.operation is LifecycleOperation.ROLLBACK:
            if self.target_revision is None or self.target_revision >= self.revision:
                raise ValueError("rollback must target an earlier positive revision")
            if self.target_revision < 1:
                raise ValueError("rollback target must be positive")
            if self.content is not None or self.availability_enabled is not None:
                raise ValueError("rollback state is recovered by replay, not embedded in the entry")
        else:
            if any(
                value is not None
                for value in (
                    self.content,
                    self.availability_enabled,
                    self.available_from_tick,
                    self.available_until_tick,
                    self.target_revision,
                )
            ):
                raise ValueError(f"{self.operation} does not accept content or availability fields")

    @classmethod
    def create(
        cls,
        *,
        memory_ref: MemoryRef,
        event_ref: EventRef,
        sequence: int,
        operation: LifecycleOperation,
        content: FrozenJSON | None,
        availability_enabled: bool | None,
        available_from_tick: int | None,
        available_until_tick: int | None,
        target_revision: int | None,
        reason: str,
        previous_entry_sha256: str | None,
    ) -> LifecycleEntry:
        partial = {
            "memory_ref": str(memory_ref),
            "event_ref": str(event_ref),
            "sequence": sequence,
            "revision": sequence + 1,
            "operation": str(operation),
            "content": content.payload() if content is not None else None,
            "availability_enabled": availability_enabled,
            "available_from_tick": available_from_tick,
            "available_until_tick": available_until_tick,
            "target_revision": target_revision,
            "reason": reason,
            "previous_entry_sha256": previous_entry_sha256,
        }
        return cls(
            memory_ref=memory_ref,
            event_ref=event_ref,
            sequence=sequence,
            revision=sequence + 1,
            operation=operation,
            content=content,
            availability_enabled=availability_enabled,
            available_from_tick=available_from_tick,
            available_until_tick=available_until_tick,
            target_revision=target_revision,
            reason=reason,
            previous_entry_sha256=previous_entry_sha256,
            entry_sha256=canonical_sha256(partial),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        out = {
            "memory_ref": str(self.memory_ref),
            "event_ref": str(self.event_ref),
            "sequence": self.sequence,
            "revision": self.revision,
            "operation": str(self.operation),
            "content": self.content.payload() if self.content is not None else None,
            "availability_enabled": self.availability_enabled,
            "available_from_tick": self.available_from_tick,
            "available_until_tick": self.available_until_tick,
            "target_revision": self.target_revision,
            "reason": self.reason,
            "previous_entry_sha256": self.previous_entry_sha256,
        }
        if include_digest:
            out["entry_sha256"] = self.entry_sha256
        return out


def _validate_interval(start: int | None, end: int | None) -> None:
    if start is not None and start < 0:
        raise ValueError("available_from_tick must be nonnegative")
    if end is not None and end < 0:
        raise ValueError("available_until_tick must be nonnegative")
    if start is not None and end is not None and end < start:
        raise ValueError("availability interval end precedes start")


class LifecycleJournal:
    """In-memory append-only journal with a hash-linked portable payload."""

    def __init__(self, memory_ref: MemoryRef):
        self.memory_ref = memory_ref
        self._entries: list[LifecycleEntry] = []

    @property
    def entries(self) -> tuple[LifecycleEntry, ...]:
        return tuple(self._entries)

    @property
    def head_sha256(self) -> str | None:
        return self._entries[-1].entry_sha256 if self._entries else None

    def _append(
        self,
        *,
        event_ref: EventRef,
        operation: LifecycleOperation,
        content: FrozenJSON | None = None,
        availability_enabled: bool | None = None,
        available_from_tick: int | None = None,
        available_until_tick: int | None = None,
        target_revision: int | None = None,
        reason: str,
    ) -> LifecycleEntry:
        entry = LifecycleEntry.create(
            memory_ref=self.memory_ref,
            event_ref=event_ref,
            sequence=len(self._entries),
            operation=operation,
            content=content,
            availability_enabled=availability_enabled,
            available_from_tick=available_from_tick,
            available_until_tick=available_until_tick,
            target_revision=target_revision,
            reason=reason,
            previous_entry_sha256=self.head_sha256,
        )
        candidate = [*self._entries, entry]
        _replay_entries(self.memory_ref, candidate)
        self._entries.append(entry)
        return entry

    def record(
        self,
        event_ref: EventRef,
        content: Any,
        *,
        available: bool = True,
        available_from_tick: int | None = 0,
        available_until_tick: int | None = None,
        reason: str = "initial record",
    ) -> LifecycleEntry:
        if self._entries:
            raise ValueError("record is valid only for an empty journal")
        return self._append(
            event_ref=event_ref,
            operation=LifecycleOperation.RECORD,
            content=FrozenJSON.from_value(content),
            availability_enabled=available,
            available_from_tick=available_from_tick,
            available_until_tick=available_until_tick,
            reason=reason,
        )

    def revise(
        self,
        event_ref: EventRef,
        content: Any,
        *,
        available: bool = True,
        available_from_tick: int | None = 0,
        available_until_tick: int | None = None,
        reason: str = "content revision",
    ) -> LifecycleEntry:
        return self._append(
            event_ref=event_ref,
            operation=LifecycleOperation.REVISE,
            content=FrozenJSON.from_value(content),
            availability_enabled=available,
            available_from_tick=available_from_tick,
            available_until_tick=available_until_tick,
            reason=reason,
        )

    def set_availability(
        self,
        event_ref: EventRef,
        *,
        available: bool,
        available_from_tick: int | None = None,
        available_until_tick: int | None = None,
        reason: str = "availability revision",
    ) -> LifecycleEntry:
        return self._append(
            event_ref=event_ref,
            operation=LifecycleOperation.AVAILABILITY,
            availability_enabled=available,
            available_from_tick=available_from_tick,
            available_until_tick=available_until_tick,
            reason=reason,
        )

    def rollback(
        self,
        event_ref: EventRef,
        target_revision: int,
        *,
        reason: str = "rollback",
    ) -> LifecycleEntry:
        return self._append(
            event_ref=event_ref,
            operation=LifecycleOperation.ROLLBACK,
            target_revision=target_revision,
            reason=reason,
        )

    def delete(self, event_ref: EventRef, *, reason: str = "deletion") -> LifecycleEntry:
        return self._append(event_ref=event_ref, operation=LifecycleOperation.DELETE, reason=reason)

    def mark_conflict(self, event_ref: EventRef, *, reason: str = "conflict") -> LifecycleEntry:
        return self._append(event_ref=event_ref, operation=LifecycleOperation.CONFLICT, reason=reason)

    def mark_poisoned(self, event_ref: EventRef, *, reason: str = "poisoning quarantine") -> LifecycleEntry:
        return self._append(event_ref=event_ref, operation=LifecycleOperation.POISONING, reason=reason)

    def state_at(self, *, revision: int | None = None) -> MemoryState:
        snapshots = _replay_entries(self.memory_ref, self._entries)
        if not snapshots:
            return MemoryState(
                memory_ref=self.memory_ref,
                revision=0,
                content=None,
                availability_enabled=False,
                available_from_tick=None,
                available_until_tick=None,
            )
        target = revision if revision is not None else len(snapshots)
        if target not in snapshots:
            raise ValueError(f"unknown lifecycle revision {target}")
        return snapshots[target]

    def forecast(self, ticks: Iterable[int], *, revision: int | None = None) -> tuple[bool, ...]:
        state = self.state_at(revision=revision)
        return tuple(state.available_at(int(tick)) for tick in ticks)

    def verify(self, *, event_refs: set[str] | None = None) -> list[str]:
        problems: list[str] = []
        previous: str | None = None
        for index, entry in enumerate(self._entries):
            if entry.memory_ref != self.memory_ref:
                problems.append(f"entry {index} memory reference drift")
            if entry.sequence != index or entry.revision != index + 1:
                problems.append(f"entry {index} sequence or revision drift")
            if entry.previous_entry_sha256 != previous:
                problems.append(f"entry {index} previous digest drift")
            if canonical_sha256(entry.payload(include_digest=False)) != entry.entry_sha256:
                problems.append(f"entry {index} digest drift")
            if event_refs is not None and str(entry.event_ref) not in event_refs:
                problems.append(f"entry {index} references missing event {entry.event_ref}")
            previous = entry.entry_sha256
        try:
            _replay_entries(self.memory_ref, self._entries)
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_SCHEMA,
            "memory_ref": str(self.memory_ref),
            "entries": [entry.payload() for entry in self._entries],
            "head_sha256": self.head_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def _replay_entries(memory_ref: MemoryRef, entries: Iterable[LifecycleEntry]) -> dict[int, MemoryState]:
    snapshots: dict[int, MemoryState] = {}
    current = MemoryState(
        memory_ref=memory_ref,
        revision=0,
        content=None,
        availability_enabled=False,
        available_from_tick=None,
        available_until_tick=None,
    )
    for expected_sequence, entry in enumerate(entries):
        if entry.memory_ref != memory_ref:
            raise ValueError(f"entry {expected_sequence} belongs to another memory")
        if entry.sequence != expected_sequence or entry.revision != expected_sequence + 1:
            raise ValueError(f"entry {expected_sequence} revision is not monotonic")
        if entry.operation is LifecycleOperation.RECORD:
            if current.content is not None:
                raise ValueError("record cannot replace an existing lifecycle")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=entry.content,
                availability_enabled=bool(entry.availability_enabled),
                available_from_tick=entry.available_from_tick,
                available_until_tick=entry.available_until_tick,
            )
        elif entry.operation is LifecycleOperation.REVISE:
            if current.content is None or current.deleted:
                raise ValueError("revise requires an existing nondeleted memory")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=entry.content,
                availability_enabled=bool(entry.availability_enabled),
                available_from_tick=entry.available_from_tick,
                available_until_tick=entry.available_until_tick,
            )
        elif entry.operation is LifecycleOperation.AVAILABILITY:
            if current.content is None or current.deleted:
                raise ValueError("availability revision requires an existing nondeleted memory")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=current.content,
                availability_enabled=bool(entry.availability_enabled),
                available_from_tick=entry.available_from_tick,
                available_until_tick=entry.available_until_tick,
                conflicted=current.conflicted,
                poisoned=current.poisoned,
                rollback_to_revision=current.rollback_to_revision,
            )
        elif entry.operation is LifecycleOperation.ROLLBACK:
            target = snapshots.get(int(entry.target_revision or -1))
            if target is None or target.content is None:
                raise ValueError(f"rollback target {entry.target_revision} is unavailable")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=target.content,
                availability_enabled=target.availability_enabled,
                available_from_tick=target.available_from_tick,
                available_until_tick=target.available_until_tick,
                deleted=target.deleted,
                conflicted=target.conflicted,
                poisoned=target.poisoned,
                rollback_to_revision=entry.target_revision,
            )
        elif entry.operation is LifecycleOperation.DELETE:
            if current.content is None:
                raise ValueError("delete requires an existing memory")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=current.content,
                availability_enabled=False,
                available_from_tick=current.available_from_tick,
                available_until_tick=current.available_until_tick,
                deleted=True,
                conflicted=current.conflicted,
                poisoned=current.poisoned,
                rollback_to_revision=current.rollback_to_revision,
            )
        elif entry.operation in {LifecycleOperation.CONFLICT, LifecycleOperation.POISONING}:
            if current.content is None or current.deleted:
                raise ValueError(f"{entry.operation} requires an existing nondeleted memory")
            current = MemoryState(
                memory_ref=memory_ref,
                revision=entry.revision,
                content=current.content,
                availability_enabled=current.availability_enabled,
                available_from_tick=current.available_from_tick,
                available_until_tick=current.available_until_tick,
                conflicted=current.conflicted or entry.operation is LifecycleOperation.CONFLICT,
                poisoned=current.poisoned or entry.operation is LifecycleOperation.POISONING,
                rollback_to_revision=current.rollback_to_revision,
            )
        else:
            raise ValueError(f"unsupported lifecycle operation {entry.operation}")
        snapshots[entry.revision] = current
    return snapshots
