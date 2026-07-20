from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..substrate.events import (
    BranchMeta,
    BranchRef,
    EntityMeta,
    EntityRef,
    EventGraph,
    EventMeta,
    EventRef,
    FrozenJSON,
    ObservationMeta,
    ObservationRef,
    SensorClock,
    SensorClockRef,
    canonical_bytes,
    canonical_sha256,
)

SCENARIO_SCHEMA = "mop-expansion-scenario/v1"
OPERATIONS = ("occlusion", "transform", "split", "merge", "delay", "intervention")
JOIN_CONTROLS = frozenset({"wrong-time", "wrong-event", "appearance-only"})


def _stable_int(seed: int, label: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(canonical_bytes({"seed": seed, "label": label})).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _token(namespace: str, label: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{label}".encode()).hexdigest()[:20]
    return f"{namespace}/{label}-{digest}"


@dataclass(frozen=True, slots=True)
class JoinProbe:
    control: str
    left_observation_ref: ObservationRef
    right_observation_ref: ObservationRef
    expected_control_accept: bool
    ground_truth_relation: str

    def __post_init__(self) -> None:
        if self.control not in JOIN_CONTROLS:
            raise ValueError(f"unsupported join control {self.control!r}")
        if not self.ground_truth_relation.strip():
            raise ValueError("join probe ground truth must not be empty")

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "left_observation_ref": str(self.left_observation_ref),
            "right_observation_ref": str(self.right_observation_ref),
            "expected_control_accept": self.expected_control_accept,
            "ground_truth_relation": self.ground_truth_relation,
        }


@dataclass(frozen=True, slots=True)
class ScenarioBundle:
    seed: int
    graph: EventGraph
    operation_event_refs: tuple[tuple[str, tuple[EventRef, ...]], ...]
    join_probes: tuple[JoinProbe, ...]
    schema: str = SCENARIO_SCHEMA

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("scenario seed must be nonnegative")
        if self.schema != SCENARIO_SCHEMA:
            raise ValueError(f"unsupported scenario schema {self.schema!r}")
        problems = self.validate()
        if problems:
            raise ValueError("invalid deterministic scenario: " + "; ".join(problems))

    def validate(self) -> list[str]:
        problems = self.graph.validate()
        operation_map = dict(self.operation_event_refs)
        if tuple(operation_map) != OPERATIONS:
            problems.append("scenario operations are missing or out of canonical order")
        if any(not refs for refs in operation_map.values()):
            problems.append("every scenario operation must reference at least one event")
        event_refs = {str(row.ref) for row in self.graph.events}
        if any(str(ref) not in event_refs for refs in operation_map.values() for ref in refs):
            problems.append("scenario operation references an unknown event")

        observations = {str(row.ref): row for row in self.graph.observations}
        for probe in self.join_probes:
            left = observations.get(str(probe.left_observation_ref))
            right = observations.get(str(probe.right_observation_ref))
            if left is None or right is None:
                problems.append(f"{probe.control} probe references a missing observation")
                continue
            if probe.control == "wrong-time" and left.clock.overlaps(right.clock):
                problems.append("wrong-time probe intervals overlap")
            elif probe.control == "wrong-event" and left.event_ref == right.event_ref:
                problems.append("wrong-event probe uses one event")
            elif probe.control == "appearance-only":
                if left.content.sha256 == right.content.sha256:
                    problems.append("appearance-only probe content did not change")
                if not set(left.entity_refs) & set(right.entity_refs):
                    problems.append("appearance-only probe lacks a persistent entity")
            if probe.expected_control_accept:
                problems.append(f"negative control {probe.control} cannot be marked accepted")

        if {row.control for row in self.join_probes} != JOIN_CONTROLS:
            problems.append("join probe control coverage is incomplete")
        return problems

    def payload(self) -> dict[str, Any]:
        graph_payload = self.graph.payload()
        return {
            "schema": self.schema,
            "seed": self.seed,
            "operations": list(OPERATIONS),
            "operation_event_refs": {
                operation: [str(ref) for ref in refs] for operation, refs in self.operation_event_refs
            },
            "graph": graph_payload,
            "graph_sha256": canonical_sha256(graph_payload),
            "join_probes": [probe.payload() for probe in self.join_probes],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def make_scenario(*, seed: int) -> ScenarioBundle:

    if seed < 0:
        raise ValueError("scenario seed must be nonnegative")
    source = FrozenJSON.from_value(
        {
            "factory": "wave-e0-deterministic",
            "schema": SCENARIO_SCHEMA,
            "seed": seed,
            "operations": list(OPERATIONS),
        }
    )
    prefix = source.sha256[:16]

    def event_ref(label: str) -> EventRef:
        return EventRef(_token(f"event:{prefix}", label))

    def entity_ref(label: str) -> EntityRef:
        return EntityRef(_token(f"entity:{prefix}", label))

    def observation_ref(label: str) -> ObservationRef:
        return ObservationRef(_token(f"observation:{prefix}", label))

    def branch_ref(label: str) -> BranchRef:
        return BranchRef(_token(f"branch:{prefix}", label))

    def clock_ref(label: str) -> SensorClockRef:
        return SensorClockRef(_token(f"clock:{prefix}", label))

    event = {
        "birth": event_ref("birth"),
        "transform": event_ref("transform"),
        "occlusion": event_ref("occlusion"),
        "reveal": event_ref("reveal"),
        "split": event_ref("split"),
        "merge": event_ref("merge"),
        "delay": event_ref("delay"),
        "intervention": event_ref("intervention"),
        "hold": event_ref("consequence-hold"),
        "translate": event_ref("consequence-translate"),
    }
    entity = {
        "root": entity_ref("root"),
        "left": entity_ref("split-left"),
        "right": entity_ref("split-right"),
        "merged": entity_ref("merged"),
    }
    branch = {"hold": branch_ref("hold"), "translate": branch_ref("translate")}

    x = _stable_int(seed, "x", 7)
    y = _stable_int(seed, "y", 7)
    root_initial = FrozenJSON.from_value({"position": [x, y], "orientation_quarters": 0, "parts": ["whole"]})
    root_transformed = FrozenJSON.from_value(
        {"position": [x, y], "orientation_quarters": 1, "parts": ["whole"]}
    )
    left_state = FrozenJSON.from_value({"position": [x, y], "orientation_quarters": 1, "parts": ["left"]})
    right_state = FrozenJSON.from_value(
        {"position": [x + 1, y], "orientation_quarters": 1, "parts": ["right"]}
    )
    merged_state = FrozenJSON.from_value(
        {"position": [x, y], "orientation_quarters": 1, "parts": ["left", "right"]}
    )
    translated_state = FrozenJSON.from_value(
        {"position": [x, y + 1], "orientation_quarters": 1, "parts": ["left", "right"]}
    )

    branches = (
        BranchMeta.create(
            ref=branch["hold"],
            fork_event_ref=event["intervention"],
            parent_state=merged_state,
            intervention=FrozenJSON.from_value({"action": "hold", "delta": [0, 0]}),
            consequence_event_ref=event["hold"],
            consequence_state=merged_state,
            chosen=False,
        ),
        BranchMeta.create(
            ref=branch["translate"],
            fork_event_ref=event["intervention"],
            parent_state=merged_state,
            intervention=FrozenJSON.from_value({"action": "translate", "delta": [0, 1]}),
            consequence_event_ref=event["translate"],
            consequence_state=translated_state,
            chosen=True,
        ),
    )

    events = (
        EventMeta(
            ref=event["birth"],
            kind="birth",
            sequence=0,
            parent_event_refs=(),
            entity_refs=(entity["root"],),
            data=FrozenJSON.from_value(
                {"entity_ref": str(entity["root"]), "state_sha256": root_initial.sha256}
            ),
        ),
        EventMeta(
            ref=event["transform"],
            kind="transform",
            sequence=1,
            parent_event_refs=(event["birth"],),
            entity_refs=(entity["root"],),
            data=FrozenJSON.from_value(
                {
                    "operation": "quarter-turn",
                    "before_sha256": root_initial.sha256,
                    "after_sha256": root_transformed.sha256,
                }
            ),
        ),
        EventMeta(
            ref=event["occlusion"],
            kind="occlusion",
            sequence=2,
            parent_event_refs=(event["transform"],),
            entity_refs=(entity["root"],),
            data=FrozenJSON.from_value({"visible": False, "entity_ref": str(entity["root"])}),
        ),
        EventMeta(
            ref=event["reveal"],
            kind="reveal",
            sequence=3,
            parent_event_refs=(event["occlusion"],),
            entity_refs=(entity["root"],),
            data=FrozenJSON.from_value({"visible": True, "entity_ref": str(entity["root"])}),
        ),
        EventMeta(
            ref=event["split"],
            kind="split",
            sequence=4,
            parent_event_refs=(event["reveal"],),
            entity_refs=(entity["root"], entity["left"], entity["right"]),
            data=FrozenJSON.from_value(
                {
                    "parent_ref": str(entity["root"]),
                    "child_refs": [str(entity["left"]), str(entity["right"])],
                }
            ),
        ),
        EventMeta(
            ref=event["merge"],
            kind="merge",
            sequence=5,
            parent_event_refs=(event["split"],),
            entity_refs=(entity["left"], entity["right"], entity["merged"]),
            data=FrozenJSON.from_value(
                {
                    "parent_refs": [str(entity["left"]), str(entity["right"])],
                    "child_ref": str(entity["merged"]),
                }
            ),
        ),
        EventMeta(
            ref=event["delay"],
            kind="delay",
            sequence=6,
            parent_event_refs=(event["merge"],),
            entity_refs=(entity["merged"],),
            data=FrozenJSON.from_value({"delayed_sensor": "audio", "arrival_delay_ticks": 7}),
        ),
        EventMeta(
            ref=event["intervention"],
            kind="intervention",
            sequence=7,
            parent_event_refs=(event["delay"],),
            entity_refs=(entity["merged"],),
            data=FrozenJSON.from_value({"branch_count": 2, "parent_state_sha256": merged_state.sha256}),
        ),
        EventMeta(
            ref=event["hold"],
            kind="consequence",
            sequence=8,
            parent_event_refs=(event["intervention"],),
            entity_refs=(entity["merged"],),
            data=FrozenJSON.from_value({"action": "hold", "consequence_state_sha256": merged_state.sha256}),
            branch_ref=branch["hold"],
        ),
        EventMeta(
            ref=event["translate"],
            kind="consequence",
            sequence=8,
            parent_event_refs=(event["intervention"],),
            entity_refs=(entity["merged"],),
            data=FrozenJSON.from_value(
                {"action": "translate", "consequence_state_sha256": translated_state.sha256}
            ),
            branch_ref=branch["translate"],
        ),
    )

    entities = (
        EntityMeta(
            ref=entity["root"],
            birth_event_ref=event["birth"],
            parent_refs=(),
            state=root_transformed,
            lifecycle_state="split",
        ),
        EntityMeta(
            ref=entity["left"],
            birth_event_ref=event["split"],
            parent_refs=(entity["root"],),
            state=left_state,
            lifecycle_state="merged",
        ),
        EntityMeta(
            ref=entity["right"],
            birth_event_ref=event["split"],
            parent_refs=(entity["root"],),
            state=right_state,
            lifecycle_state="merged",
        ),
        EntityMeta(
            ref=entity["merged"],
            birth_event_ref=event["merge"],
            parent_refs=(entity["left"], entity["right"]),
            state=merged_state,
            lifecycle_state="active",
        ),
    )

    def clock(
        label: str,
        sensor_id: str,
        start: int,
        end: int,
        arrival: int,
        uncertainty: int = 0,
    ) -> SensorClock:
        return SensorClock(
            ref=clock_ref(label),
            sensor_id=sensor_id,
            capture_start_tick=start,
            capture_end_tick=end,
            arrival_tick=arrival,
            uncertainty_ticks=uncertainty,
        )

    observation = {
        label: observation_ref(label)
        for label in (
            "birth",
            "transform",
            "occlusion",
            "reveal",
            "split-ambiguous",
            "merge",
            "delay-visual",
            "delay-audio",
            "delay-wrong-time",
            "hold",
            "translate",
        )
    }
    observations = (
        ObservationMeta(
            ref=observation["birth"],
            event_ref=event["birth"],
            entity_refs=(entity["root"],),
            clock=clock("birth", "vision", 0, 0, 0),
            content=FrozenJSON.from_value({"appearance": "blue-square", "visible": True}),
        ),
        ObservationMeta(
            ref=observation["transform"],
            event_ref=event["transform"],
            entity_refs=(entity["root"],),
            clock=clock("transform", "vision", 10, 10, 10),
            content=FrozenJSON.from_value({"appearance": "blue-diamond", "visible": True}),
        ),
        ObservationMeta(
            ref=observation["occlusion"],
            event_ref=event["occlusion"],
            entity_refs=(entity["root"],),
            clock=clock("occlusion", "vision", 20, 20, 20),
            content=FrozenJSON.from_value({"appearance": None, "visible": False}),
        ),
        ObservationMeta(
            ref=observation["reveal"],
            event_ref=event["reveal"],
            entity_refs=(entity["root"],),
            clock=clock("reveal", "vision", 30, 30, 30),
            content=FrozenJSON.from_value({"appearance": "blue-diamond", "visible": True}),
        ),
        ObservationMeta(
            ref=observation["split-ambiguous"],
            event_ref=event["split"],
            entity_refs=(entity["left"], entity["right"]),
            clock=clock("split-ambiguous", "vision", 40, 40, 40, 1),
            content=FrozenJSON.from_value({"appearance": "overlapping-components", "visible": True}),
            ambiguous_entity_refs=(entity["left"], entity["right"]),
            abstention_required=True,
        ),
        ObservationMeta(
            ref=observation["merge"],
            event_ref=event["merge"],
            entity_refs=(entity["merged"],),
            clock=clock("merge", "vision", 60, 62, 62, 1),
            content=FrozenJSON.from_value({"appearance": "joined-components", "visible": True}),
        ),
        ObservationMeta(
            ref=observation["delay-visual"],
            event_ref=event["delay"],
            entity_refs=(entity["merged"],),
            clock=clock("delay-visual", "vision", 60, 62, 62, 1),
            content=FrozenJSON.from_value({"modality": "vision", "pulse": 3}),
        ),
        ObservationMeta(
            ref=observation["delay-audio"],
            event_ref=event["delay"],
            entity_refs=(entity["merged"],),
            clock=clock("delay-audio", "audio", 59, 61, 68, 1),
            content=FrozenJSON.from_value({"modality": "audio", "pulse": 3}),
        ),
        ObservationMeta(
            ref=observation["delay-wrong-time"],
            event_ref=event["delay"],
            entity_refs=(entity["merged"],),
            clock=clock("delay-wrong-time", "control", 90, 91, 91),
            content=FrozenJSON.from_value({"modality": "control", "pulse": 3}),
            role="control",
        ),
        ObservationMeta(
            ref=observation["hold"],
            event_ref=event["hold"],
            entity_refs=(entity["merged"],),
            clock=clock("hold", "counterfactual", 70, 70, 70),
            content=merged_state,
            role="counterfactual",
        ),
        ObservationMeta(
            ref=observation["translate"],
            event_ref=event["translate"],
            entity_refs=(entity["merged"],),
            clock=clock("translate", "vision", 70, 70, 70),
            content=translated_state,
        ),
    )

    graph = EventGraph(
        source=source,
        events=events,
        entities=entities,
        observations=observations,
        branches=branches,
    )
    join_probes = (
        JoinProbe(
            control="wrong-time",
            left_observation_ref=observation["delay-visual"],
            right_observation_ref=observation["delay-wrong-time"],
            expected_control_accept=False,
            ground_truth_relation="same-event-nonoverlapping-clock",
        ),
        JoinProbe(
            control="wrong-event",
            left_observation_ref=observation["delay-visual"],
            right_observation_ref=observation["merge"],
            expected_control_accept=False,
            ground_truth_relation="different-event-overlapping-clock",
        ),
        JoinProbe(
            control="appearance-only",
            left_observation_ref=observation["birth"],
            right_observation_ref=observation["transform"],
            expected_control_accept=False,
            ground_truth_relation="same-entity-changed-appearance",
        ),
    )
    return ScenarioBundle(
        seed=seed,
        graph=graph,
        operation_event_refs=(
            ("occlusion", (event["occlusion"], event["reveal"])),
            ("transform", (event["transform"],)),
            ("split", (event["split"],)),
            ("merge", (event["merge"],)),
            ("delay", (event["delay"],)),
            ("intervention", (event["intervention"], event["hold"], event["translate"])),
        ),
        join_probes=join_probes,
    )
