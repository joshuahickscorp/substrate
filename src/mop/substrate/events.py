from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVENT_GRAPH_SCHEMA = "mop-event-graph/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_BODY_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*$")
_EVENT_KINDS = frozenset(
    {
        "birth",
        "transform",
        "occlusion",
        "reveal",
        "split",
        "merge",
        "delay",
        "intervention",
        "consequence",
    }
)
_ENTITY_STATES = frozenset({"active", "occluded", "split", "merged", "retired", "ambiguous"})
_OBSERVATION_ROLES = frozenset({"primary", "control", "counterfactual"})


def canonical_bytes(value: Any) -> bytes:

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_ref(value: str, namespace: str) -> None:
    prefix = f"{namespace}:"
    if not value.startswith(prefix) or _REF_BODY_RE.fullmatch(value[len(prefix) :]) is None:
        raise ValueError(f"{namespace} reference must start with {prefix!r} and use stable characters")


@dataclass(frozen=True, slots=True, order=True)
class EventRef:
    value: str

    def __post_init__(self) -> None:
        _require_ref(self.value, "event")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class EntityRef:
    value: str

    def __post_init__(self) -> None:
        _require_ref(self.value, "entity")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class ObservationRef:
    value: str

    def __post_init__(self) -> None:
        _require_ref(self.value, "observation")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class BranchRef:
    value: str

    def __post_init__(self) -> None:
        _require_ref(self.value, "branch")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class SensorClockRef:
    value: str

    def __post_init__(self) -> None:
        _require_ref(self.value, "clock")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FrozenJSON:
    canonical: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            value = json.loads(self.canonical)
        except json.JSONDecodeError as exc:
            raise ValueError("FrozenJSON contains invalid JSON") from exc
        expected = canonical_bytes(value).decode("utf-8")
        if self.canonical != expected:
            raise ValueError("FrozenJSON text is not canonical")
        _require_sha256(self.sha256, "FrozenJSON.sha256")
        if hashlib.sha256(self.canonical.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("FrozenJSON digest mismatch")

    @classmethod
    def from_value(cls, value: Any) -> FrozenJSON:
        raw = canonical_bytes(value)
        return cls(canonical=raw.decode("utf-8"), sha256=hashlib.sha256(raw).hexdigest())

    def value(self) -> Any:
        return json.loads(self.canonical)

    def payload(self) -> dict[str, Any]:
        return {"value": self.value(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SensorClock:
    ref: SensorClockRef
    sensor_id: str
    capture_start_tick: int
    capture_end_tick: int
    arrival_tick: int
    uncertainty_ticks: int = 0
    tick_period_ns: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.sensor_id.strip():
            raise ValueError("sensor_id must not be empty")
        if self.capture_start_tick < 0 or self.capture_end_tick < self.capture_start_tick:
            raise ValueError("sensor capture interval is invalid")
        if self.arrival_tick < self.capture_end_tick:
            raise ValueError("arrival_tick cannot precede capture_end_tick")
        if self.uncertainty_ticks < 0 or self.tick_period_ns <= 0:
            raise ValueError("sensor uncertainty and tick period must be nonnegative and positive")

    @property
    def expanded_interval(self) -> tuple[int, int]:
        return (
            max(0, self.capture_start_tick - self.uncertainty_ticks),
            self.capture_end_tick + self.uncertainty_ticks,
        )

    def overlaps(self, other: SensorClock) -> bool:
        left_start, left_end = self.expanded_interval
        right_start, right_end = other.expanded_interval
        return max(left_start, right_start) <= min(left_end, right_end)

    def payload(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "sensor_id": self.sensor_id,
            "capture_start_tick": self.capture_start_tick,
            "capture_end_tick": self.capture_end_tick,
            "arrival_tick": self.arrival_tick,
            "uncertainty_ticks": self.uncertainty_ticks,
            "tick_period_ns": self.tick_period_ns,
        }


@dataclass(frozen=True, slots=True)
class EventMeta:
    ref: EventRef
    kind: str
    sequence: int
    parent_event_refs: tuple[EventRef, ...]
    entity_refs: tuple[EntityRef, ...]
    data: FrozenJSON
    branch_ref: BranchRef | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(f"unsupported event kind {self.kind!r}")
        if self.sequence < 0:
            raise ValueError("event sequence must be nonnegative")
        if len(set(self.parent_event_refs)) != len(self.parent_event_refs):
            raise ValueError("event parent references must be unique")
        if len(set(self.entity_refs)) != len(self.entity_refs):
            raise ValueError("event entity references must be unique")

    def payload(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "kind": self.kind,
            "sequence": self.sequence,
            "parent_event_refs": [str(ref) for ref in self.parent_event_refs],
            "entity_refs": [str(ref) for ref in self.entity_refs],
            "data": self.data.payload(),
            "branch_ref": str(self.branch_ref) if self.branch_ref is not None else None,
        }


@dataclass(frozen=True, slots=True)
class EntityMeta:
    ref: EntityRef
    birth_event_ref: EventRef
    parent_refs: tuple[EntityRef, ...]
    state: FrozenJSON
    lifecycle_state: str = "active"

    def __post_init__(self) -> None:
        if self.lifecycle_state not in _ENTITY_STATES:
            raise ValueError(f"unsupported entity lifecycle state {self.lifecycle_state!r}")
        if self.ref in self.parent_refs or len(set(self.parent_refs)) != len(self.parent_refs):
            raise ValueError("entity lineage parents must be unique and cannot contain self")

    def payload(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "birth_event_ref": str(self.birth_event_ref),
            "parent_refs": [str(ref) for ref in self.parent_refs],
            "state": self.state.payload(),
            "lifecycle_state": self.lifecycle_state,
        }


@dataclass(frozen=True, slots=True)
class ObservationMeta:
    ref: ObservationRef
    event_ref: EventRef
    entity_refs: tuple[EntityRef, ...]
    clock: SensorClock
    content: FrozenJSON
    role: str = "primary"
    ambiguous_entity_refs: tuple[EntityRef, ...] = ()
    abstention_required: bool = False

    def __post_init__(self) -> None:
        if self.role not in _OBSERVATION_ROLES:
            raise ValueError(f"unsupported observation role {self.role!r}")
        if not self.entity_refs:
            raise ValueError("observation must reference at least one entity")
        if len(set(self.entity_refs)) != len(self.entity_refs):
            raise ValueError("observation entity references must be unique")
        if not set(self.ambiguous_entity_refs) <= set(self.entity_refs):
            raise ValueError("ambiguous entities must be present in entity_refs")
        if self.ambiguous_entity_refs and not self.abstention_required:
            raise ValueError("ambiguous observations must require abstention")

    def payload(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "event_ref": str(self.event_ref),
            "entity_refs": [str(ref) for ref in self.entity_refs],
            "clock": self.clock.payload(),
            "content": self.content.payload(),
            "role": self.role,
            "ambiguous_entity_refs": [str(ref) for ref in self.ambiguous_entity_refs],
            "abstention_required": self.abstention_required,
        }


def branch_replay_sha256(
    *,
    fork_event_ref: EventRef,
    parent_state: FrozenJSON,
    intervention: FrozenJSON,
    consequence_event_ref: EventRef,
    consequence_state: FrozenJSON,
) -> str:
    return canonical_sha256(
        {
            "fork_event_ref": str(fork_event_ref),
            "parent_state_sha256": parent_state.sha256,
            "intervention_sha256": intervention.sha256,
            "consequence_event_ref": str(consequence_event_ref),
            "consequence_state_sha256": consequence_state.sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class BranchMeta:
    ref: BranchRef
    fork_event_ref: EventRef
    parent_state: FrozenJSON
    intervention: FrozenJSON
    consequence_event_ref: EventRef
    consequence_state: FrozenJSON
    chosen: bool
    exact_replay_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.exact_replay_sha256, "exact_replay_sha256")
        expected = branch_replay_sha256(
            fork_event_ref=self.fork_event_ref,
            parent_state=self.parent_state,
            intervention=self.intervention,
            consequence_event_ref=self.consequence_event_ref,
            consequence_state=self.consequence_state,
        )
        if self.exact_replay_sha256 != expected:
            raise ValueError("branch exact replay digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        ref: BranchRef,
        fork_event_ref: EventRef,
        parent_state: FrozenJSON,
        intervention: FrozenJSON,
        consequence_event_ref: EventRef,
        consequence_state: FrozenJSON,
        chosen: bool,
    ) -> BranchMeta:
        return cls(
            ref=ref,
            fork_event_ref=fork_event_ref,
            parent_state=parent_state,
            intervention=intervention,
            consequence_event_ref=consequence_event_ref,
            consequence_state=consequence_state,
            chosen=chosen,
            exact_replay_sha256=branch_replay_sha256(
                fork_event_ref=fork_event_ref,
                parent_state=parent_state,
                intervention=intervention,
                consequence_event_ref=consequence_event_ref,
                consequence_state=consequence_state,
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "fork_event_ref": str(self.fork_event_ref),
            "parent_state": self.parent_state.payload(),
            "intervention": self.intervention.payload(),
            "consequence_event_ref": str(self.consequence_event_ref),
            "consequence_state": self.consequence_state.payload(),
            "chosen": self.chosen,
            "exact_replay_sha256": self.exact_replay_sha256,
        }


@dataclass(frozen=True, slots=True)
class EventGraph:
    source: FrozenJSON
    events: tuple[EventMeta, ...]
    entities: tuple[EntityMeta, ...]
    observations: tuple[ObservationMeta, ...]
    branches: tuple[BranchMeta, ...]
    schema: str = EVENT_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_GRAPH_SCHEMA:
            raise ValueError(f"unsupported event graph schema {self.schema!r}")
        problems = self.validate()
        if problems:
            raise ValueError("invalid event graph: " + "; ".join(problems))

    def validate(self) -> list[str]:
        problems: list[str] = []
        event_by_ref = {str(row.ref): row for row in self.events}
        entity_by_ref = {str(row.ref): row for row in self.entities}
        observation_refs = [str(row.ref) for row in self.observations]
        branch_by_ref = {str(row.ref): row for row in self.branches}
        clock_refs = [str(row.clock.ref) for row in self.observations]

        if len(event_by_ref) != len(self.events):
            problems.append("event references are not unique")
        if len(entity_by_ref) != len(self.entities):
            problems.append("entity references are not unique")
        if len(set(observation_refs)) != len(observation_refs):
            problems.append("observation references are not unique")
        if len(branch_by_ref) != len(self.branches):
            problems.append("branch references are not unique")
        if len(set(clock_refs)) != len(clock_refs):
            problems.append("sensor clock references are not unique")

        for event in self.events:
            for parent in event.parent_event_refs:
                parent_row = event_by_ref.get(str(parent))
                if parent_row is None:
                    problems.append(f"event {event.ref} has missing parent {parent}")
                elif parent_row.sequence >= event.sequence:
                    problems.append(f"event {event.ref} parent {parent} is not earlier")
            for entity_ref in event.entity_refs:
                if str(entity_ref) not in entity_by_ref:
                    problems.append(f"event {event.ref} has missing entity {entity_ref}")
            if event.branch_ref is not None and str(event.branch_ref) not in branch_by_ref:
                problems.append(f"event {event.ref} has missing branch {event.branch_ref}")

        for entity in self.entities:
            if str(entity.birth_event_ref) not in event_by_ref:
                problems.append(f"entity {entity.ref} has missing birth event {entity.birth_event_ref}")
            for entity_parent in entity.parent_refs:
                if str(entity_parent) not in entity_by_ref:
                    problems.append(f"entity {entity.ref} has missing lineage parent {entity_parent}")

        for observation in self.observations:
            if str(observation.event_ref) not in event_by_ref:
                problems.append(f"observation {observation.ref} has missing event {observation.event_ref}")
            for entity_ref in observation.entity_refs:
                if str(entity_ref) not in entity_by_ref:
                    problems.append(f"observation {observation.ref} has missing entity {entity_ref}")

        by_fork: dict[str, list[BranchMeta]] = {}
        for branch in self.branches:
            fork_event = event_by_ref.get(str(branch.fork_event_ref))
            consequence = event_by_ref.get(str(branch.consequence_event_ref))
            if fork_event is None:
                problems.append(f"branch {branch.ref} has missing fork event {branch.fork_event_ref}")
            if consequence is None:
                problems.append(
                    f"branch {branch.ref} has missing consequence event {branch.consequence_event_ref}"
                )
            elif consequence.branch_ref != branch.ref:
                problems.append(f"branch {branch.ref} does not own its consequence event")
            expected = branch_replay_sha256(
                fork_event_ref=branch.fork_event_ref,
                parent_state=branch.parent_state,
                intervention=branch.intervention,
                consequence_event_ref=branch.consequence_event_ref,
                consequence_state=branch.consequence_state,
            )
            if branch.exact_replay_sha256 != expected:
                problems.append(f"branch {branch.ref} exact replay digest drift")
            by_fork.setdefault(str(branch.fork_event_ref), []).append(branch)

        for fork_ref, rows in by_fork.items():
            if len({row.parent_state.sha256 for row in rows}) != 1:
                problems.append(f"fork {fork_ref} does not use one parent state")
            if sum(row.chosen for row in rows) != 1:
                problems.append(f"fork {fork_ref} must have exactly one chosen branch")
            if len({row.intervention.sha256 for row in rows}) != len(rows):
                problems.append(f"fork {fork_ref} intervention identities are not unique")

        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source.payload(),
            "events": [row.payload() for row in self.events],
            "entities": [row.payload() for row in self.entities],
            "observations": [row.payload() for row in self.observations],
            "branches": [row.payload() for row in self.branches],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(path, raw)


def write_canonical_json(payload: dict[str, Any], out_path: str | Path) -> Path:

    path = Path(out_path)
    atomic_write_bytes(path, canonical_bytes(payload))
    return path
