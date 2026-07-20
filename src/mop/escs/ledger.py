
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Self

from mop.substrate.events import BranchRef, EventRef, canonical_sha256

from .accounting import FACTUAL_BRANCH
from .events import (
    CommitmentEvent,
    ConsequenceEvent,
    EpistemicStatus,
    ESCSEvent,
    HypothesisEvent,
    ObservationEvent,
    event_from_payload,
    state_version_for_parents,
)

EVENT_LEDGER_ENTRY_SCHEMA = "mop-escs-event-ledger-entry/v1"
EVENT_LEDGER_SCHEMA = "mop-escs-event-ledger/v1"
_EVENT_TYPES = (ObservationEvent, HypothesisEvent, CommitmentEvent, ConsequenceEvent)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class EventLedgerEntry:
    sequence: int
    event: ESCSEvent
    previous_entry_sha256: str | None
    entry_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.sequence, "EventLedgerEntry.sequence")
        if type(self.event) not in _EVENT_TYPES:
            raise ValueError("EventLedgerEntry.event must be one of the four ESCS event types")
        if self.previous_entry_sha256 is not None:
            _require_digest(self.previous_entry_sha256, "previous_entry_sha256")
        _require_digest(self.entry_sha256, "entry_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.entry_sha256:
            raise ValueError("event ledger entry digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        event: ESCSEvent,
        previous_entry_sha256: str | None,
    ) -> EventLedgerEntry:
        partial = {
            "schema": EVENT_LEDGER_ENTRY_SCHEMA,
            "sequence": sequence,
            "event": event.payload(),
            "previous_entry_sha256": previous_entry_sha256,
        }
        return cls(
            sequence=sequence,
            event=event,
            previous_entry_sha256=previous_entry_sha256,
            entry_sha256=canonical_sha256(partial),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EventLedgerEntry:
        expected = {"schema", "sequence", "event", "previous_entry_sha256", "entry_sha256"}
        _require_exact_keys(payload, expected, "EventLedgerEntry")
        if payload["schema"] != EVENT_LEDGER_ENTRY_SCHEMA:
            raise ValueError(f"unsupported event ledger entry schema {payload['schema']!r}")
        return cls(
            sequence=payload["sequence"],
            event=event_from_payload(payload["event"]),
            previous_entry_sha256=payload["previous_entry_sha256"],
            entry_sha256=payload["entry_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": EVENT_LEDGER_ENTRY_SCHEMA,
            "sequence": self.sequence,
            "event": self.event.payload(),
            "previous_entry_sha256": self.previous_entry_sha256,
        }
        if include_digest:
            result["entry_sha256"] = self.entry_sha256
        return result


class EventLedger:

    def __init__(self) -> None:
        self._entries: list[EventLedgerEntry] = []
        self._events: dict[EventRef, ESCSEvent] = {}
        self._branch_ids: dict[BranchRef, list[EventRef]] = {}
        self._children_by_parent: dict[EventRef, list[EventRef]] = {}
        self._consequences_by_commitment: dict[EventRef, list[EventRef]] = {}
        self._lock = RLock()

    @property
    def entries(self) -> tuple[EventLedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def events(self) -> tuple[ESCSEvent, ...]:
        return tuple(entry.event for entry in self._entries)

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(str(event_id) for event_id in self._events)

    @property
    def head_sha256(self) -> str | None:
        return self._entries[-1].entry_sha256 if self._entries else None

    def get(self, event_id: EventRef) -> ESCSEvent:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise ValueError(f"unknown event {event_id}") from exc

    def events_on_branch(self, branch_id: BranchRef) -> tuple[ESCSEvent, ...]:
        return tuple(self._events[event_id] for event_id in self._branch_ids.get(branch_id, []))

    def consequences_for(self, commitment_event_id: EventRef) -> tuple[ConsequenceEvent, ...]:
        rows: list[ConsequenceEvent] = []
        for event_id in self._consequences_by_commitment.get(commitment_event_id, ()):
            event = self._events[event_id]
            if not isinstance(event, ConsequenceEvent):
                raise RuntimeError("consequence index points to a non-consequence event")
            rows.append(event)
        return tuple(rows)

    def commitments_for(self, hypothesis_event_id: EventRef) -> tuple[CommitmentEvent, ...]:

        return tuple(
            event
            for event_id in self._children_by_parent.get(hypothesis_event_id, ())
            if isinstance((event := self._events[event_id]), CommitmentEvent)
        )

    def append(self, event: ESCSEvent) -> EventLedgerEntry:
        with self._lock:
            self._validate_new_event(event)
            entry = EventLedgerEntry.create(
                sequence=len(self._entries),
                event=event,
                previous_entry_sha256=self.head_sha256,
            )
            self._entries.append(entry)
            self._events[event.event_id] = event
            self._branch_ids.setdefault(event.branch_id, []).append(event.event_id)
            for parent_id in event.envelope.causal_parent_ids:
                self._children_by_parent.setdefault(parent_id, []).append(event.event_id)
            if isinstance(event, ConsequenceEvent):
                self._consequences_by_commitment.setdefault(event.commitment_event_id, []).append(
                    event.event_id
                )
            return entry

    def append_batch(self, events: tuple[ESCSEvent, ...]) -> tuple[EventLedgerEntry, ...]:

        if not isinstance(events, tuple):
            raise TypeError("event batch must be an immutable tuple")
        if not events:
            return ()
        with self._lock:
            entry_count_before = len(self._entries)
            preexisting_branches = {
                event.branch_id for event in events if event.branch_id in self._branch_ids
            }
            preexisting_parent_indices = {
                parent_id
                for event in events
                for parent_id in event.envelope.causal_parent_ids
                if parent_id in self._children_by_parent
            }
            preexisting_consequence_indices = {
                event.commitment_event_id
                for event in events
                if isinstance(event, ConsequenceEvent)
                and event.commitment_event_id in self._consequences_by_commitment
            }
            published: list[EventLedgerEntry] = []
            try:
                for event in events:
                    published.append(self.append(event))
            except Exception as exc:
                appended_entries = self._entries[entry_count_before:]
                appended_by_branch: dict[BranchRef, list[EventRef]] = {}
                appended_by_parent: dict[EventRef, list[EventRef]] = {}
                appended_by_commitment: dict[EventRef, list[EventRef]] = {}
                for entry in appended_entries:
                    self._events.pop(entry.event.event_id, None)
                    appended_by_branch.setdefault(entry.event.branch_id, []).append(entry.event.event_id)
                    for parent_id in entry.event.envelope.causal_parent_ids:
                        appended_by_parent.setdefault(parent_id, []).append(entry.event.event_id)
                    if isinstance(entry.event, ConsequenceEvent):
                        appended_by_commitment.setdefault(entry.event.commitment_event_id, []).append(
                            entry.event.event_id
                        )
                for branch_id, appended_ids in appended_by_branch.items():
                    branch_ids = self._branch_ids.get(branch_id)
                    if branch_ids is None or branch_ids[-len(appended_ids) :] != appended_ids:
                        raise RuntimeError("event batch rollback detected branch-index drift") from exc
                    del branch_ids[-len(appended_ids) :]
                    if not branch_ids and branch_id not in preexisting_branches:
                        self._branch_ids.pop(branch_id, None)
                for parent_id, appended_ids in appended_by_parent.items():
                    child_ids = self._children_by_parent.get(parent_id)
                    if child_ids is None or child_ids[-len(appended_ids) :] != appended_ids:
                        raise RuntimeError("event batch rollback detected parent-index drift") from exc
                    del child_ids[-len(appended_ids) :]
                    if not child_ids and parent_id not in preexisting_parent_indices:
                        self._children_by_parent.pop(parent_id, None)
                for commitment_id, appended_ids in appended_by_commitment.items():
                    consequence_ids = self._consequences_by_commitment.get(commitment_id)
                    if consequence_ids is None or consequence_ids[-len(appended_ids) :] != appended_ids:
                        raise RuntimeError("event batch rollback detected consequence-index drift") from exc
                    del consequence_ids[-len(appended_ids) :]
                    if not consequence_ids and commitment_id not in preexisting_consequence_indices:
                        self._consequences_by_commitment.pop(commitment_id, None)
                del self._entries[entry_count_before:]
                raise
            return tuple(published)

    def _validate_new_event(self, event: ESCSEvent) -> None:
        if type(event) not in _EVENT_TYPES:
            raise ValueError("ledger accepts only the four ESCS event types")
        if event.event_id in self._events:
            raise ValueError(f"duplicate event identity {event.event_id}")
        expected_state_version = state_version_for_parents(event.envelope.causal_parent_ids)
        if event.envelope.producer_state_version != expected_state_version:
            raise ValueError("event state version omits or changes a causal parent")

        parents: list[ESCSEvent] = []
        for event_id in event.envelope.causal_parent_ids:
            parent = self._events.get(event_id)
            if parent is None:
                raise ValueError(f"event {event.event_id} has missing causal parent {event_id}")
            if parent.envelope.clock_end_tick > event.envelope.clock_start_tick:
                raise ValueError(f"event {event.event_id} begins before causal parent {event_id} ends")
            parents.append(parent)

        if parents and event.evidence_class.taint_rank < max(
            parent.evidence_class.taint_rank for parent in parents
        ):
            raise ValueError("event evidence class cannot downgrade causal-parent taint")

        self._validate_branch_transition(event, parents)
        self._validate_supersession(event)
        if isinstance(event, ObservationEvent):
            self._validate_observation(event, parents)
        elif isinstance(event, HypothesisEvent):
            self._validate_hypothesis(event, parents)
        elif isinstance(event, CommitmentEvent):
            self._validate_commitment(event, parents)
        else:
            self._validate_consequence(event, parents)

    def _validate_branch_transition(self, event: ESCSEvent, parents: list[ESCSEvent]) -> None:
        branch_id = event.branch_id
        if branch_id == FACTUAL_BRANCH:
            if any(parent.branch_id != FACTUAL_BRANCH for parent in parents):
                raise ValueError("counterfactual content cannot authorize a factual event")
            return

        existing = self._branch_ids.get(branch_id, [])
        if not existing:
            if not isinstance(event, HypothesisEvent) or (
                event.epistemic_status is not EpistemicStatus.SIMULATED
            ):
                raise ValueError("a counterfactual branch must begin with a simulated hypothesis")
            if not parents or any(parent.branch_id != FACTUAL_BRANCH for parent in parents):
                raise ValueError("a counterfactual branch root requires factual causal parents")
            return
        if any(parent.branch_id != branch_id for parent in parents):
            raise ValueError("an established counterfactual branch accepts only same-branch parents")

    def _validate_supersession(self, event: ESCSEvent) -> None:
        for superseded_id in event.envelope.supersedes_event_ids:
            superseded = self._events.get(superseded_id)
            if superseded is None:
                raise ValueError(f"event supersedes unknown event {superseded_id}")
            if superseded.kind is not event.kind:
                raise ValueError("an event may supersede only the same event kind")
            if superseded.branch_id != event.branch_id:
                raise ValueError("an event may supersede only within its own branch")

    @staticmethod
    def _validate_observation(event: ObservationEvent, parents: list[ESCSEvent]) -> None:
        if any(not isinstance(parent, ObservationEvent | ConsequenceEvent) for parent in parents):
            raise ValueError("observations may descend only from observations or consequences")

    @staticmethod
    def _validate_hypothesis(event: HypothesisEvent, parents: list[ESCSEvent]) -> None:
        if not parents:
            raise ValueError("a hypothesis requires at least one causal parent")
        if any(
            not isinstance(parent, ObservationEvent | HypothesisEvent | ConsequenceEvent)
            for parent in parents
        ):
            raise ValueError("a hypothesis cannot consume an unobserved commitment")

    @staticmethod
    def _validate_commitment(event: CommitmentEvent, parents: list[ESCSEvent]) -> None:
        if not parents or any(not isinstance(parent, HypothesisEvent) for parent in parents):
            raise ValueError("a commitment must be authorized only by one or more hypotheses")

    def _validate_consequence(self, event: ConsequenceEvent, parents: list[ESCSEvent]) -> None:
        commitment = self._events.get(event.commitment_event_id)
        if not isinstance(commitment, CommitmentEvent):
            raise ValueError("a consequence must bind an earlier CommitmentEvent")
        if commitment.branch_id != event.branch_id:
            raise ValueError("a consequence must remain on its commitment branch")
        if event.envelope.clock_start_tick < commitment.envelope.clock_end_tick:
            raise ValueError("a consequence cannot begin before its commitment completes")
        commitment_parents = [parent for parent in parents if isinstance(parent, CommitmentEvent)]
        if commitment_parents != [commitment]:
            raise ValueError("a consequence must name exactly its bound commitment parent")
        for parent in parents:
            if isinstance(parent, ObservationEvent):
                continue
            if isinstance(parent, ConsequenceEvent):
                if parent.commitment_event_id != event.commitment_event_id:
                    raise ValueError("partial consequences cannot mix commitment identities")
                continue
            if parent is not commitment:
                raise ValueError("a consequence has an invalid causal parent kind")

    def verify(self) -> list[str]:
        problems: list[str] = []
        replay = EventLedger()
        for expected_sequence, entry in enumerate(self._entries):
            if entry.sequence != expected_sequence:
                problems.append(f"entry {expected_sequence} sequence drift")
                continue
            if entry.previous_entry_sha256 != replay.head_sha256:
                problems.append(f"entry {expected_sequence} previous digest drift")
                continue
            try:
                regenerated = replay.append(entry.event)
            except ValueError as exc:
                problems.append(f"entry {expected_sequence} replay failed: {exc}")
                continue
            if regenerated.entry_sha256 != entry.entry_sha256:
                problems.append(f"entry {expected_sequence} digest drift")
        if self._events != replay._events:
            problems.append("event lookup cache drift from deterministic replay")
        if self._branch_ids != replay._branch_ids:
            problems.append("branch index cache drift from deterministic replay")
        if self._children_by_parent != replay._children_by_parent:
            problems.append("parent-child index cache drift from deterministic replay")
        if self._consequences_by_commitment != replay._consequences_by_commitment:
            problems.append("consequence index cache drift from deterministic replay")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "schema": EVENT_LEDGER_SCHEMA,
            "entries": [entry.payload() for entry in self._entries],
            "head_sha256": self.head_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(payload, {"schema", "entries", "head_sha256"}, "EventLedger")
        if payload["schema"] != EVENT_LEDGER_SCHEMA:
            raise ValueError(f"unsupported event ledger schema {payload['schema']!r}")
        rows = payload["entries"]
        if not isinstance(rows, list):
            raise ValueError("EventLedger.entries must be a list")
        ledger = cls()
        for expected_sequence, row in enumerate(rows):
            parsed = EventLedgerEntry.from_payload(row)
            if parsed.sequence != expected_sequence:
                raise ValueError(f"event ledger sequence drift at entry {expected_sequence}")
            if parsed.previous_entry_sha256 != ledger.head_sha256:
                raise ValueError(f"event ledger hash-chain drift at entry {expected_sequence}")
            regenerated = ledger.append(parsed.event)
            if regenerated.payload() != parsed.payload():
                raise ValueError(f"event ledger replay mismatch at entry {expected_sequence}")
        if payload["head_sha256"] != ledger.head_sha256:
            raise ValueError("event ledger head digest mismatch")
        if problems := ledger.verify():
            raise ValueError("invalid event ledger: " + "; ".join(problems))
        return ledger

    @classmethod
    def replay(cls, payload: Mapping[str, Any]) -> Self:
        return cls.from_payload(payload)
