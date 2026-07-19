#!/usr/bin/env python
"""Independently replay and mutation-test the Wave E0 mechanics receipt.

This verifier reads raw JSON. It does not import the driver, scenario factory, journal, or harness
metric functions. It recomputes identities, lineage, lifecycle state, sentinel metrics, and semantic
mutation rejection with a separate implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.substrate.events import canonical_bytes

VERIFIER_SCHEMA = "mop-expansion-wave0-independent-verifier/v1"
WAVE_SCHEMA = "mop-expansion-wave0/v1"
UNIT_SCHEMA = "mop-expansion-wave0-unit/v1"
GRAPH_SCHEMA = "mop-event-graph/v1"
SCENARIO_SCHEMA = "mop-expansion-scenario/v1"
LIFECYCLE_SCHEMA = "mop-lifecycle-journal/v1"
HARNESS_SCHEMA = "mop-expansion-harness/v1"
RESULT_SCHEMA = "mop-expansion-sentinel-result/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
OPERATIONS = ("occlusion", "transform", "split", "merge", "delay", "intervention")
CONTROLS = (
    "wrong-time",
    "wrong-event",
    "appearance-only",
    "action-blind",
    "action-shuffled",
    "stale-memory",
    "unavailable-memory",
    "identity-ambiguity-abstain",
    "exact-replay",
)
IMPLEMENTATION_PATHS = (
    "src/mop/substrate/events.py",
    "src/mop/environments/scenario_factory.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/experiments/expansion_harness.py",
    "scripts/mop_expansion_wave0.py",
    "scripts/verify_expansion_wave0.py",
)
METRICS = {
    "f23": (
        "event-bytes-bound",
        "transform-identity-persists",
        "occlusion-identity-persists",
        "split-lineage-exact",
        "merge-lineage-exact",
        "ambiguity-requires-abstention",
        "wrong-time-rejected",
        "wrong-event-rejected",
        "appearance-only-rejected",
    ),
    "f29": (
        "branch-bytes-bound",
        "same-state-branches",
        "one-chosen-branch",
        "distinct-interventions",
        "distinct-consequences",
        "exact-replay",
        "action-blind-exposed",
        "action-shuffled-rejected",
    ),
    "f39": (
        "journal-bytes-bound",
        "revisions-monotonic",
        "event-lineage-shared",
        "availability-forecast-exact",
        "rollback-exact",
        "stale-memory-rejected",
        "unavailable-memory-abstains",
    ),
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")




def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "value" in value
        and isinstance(value.get("sha256"), str)
        and _SHA_RE.fullmatch(value["sha256"]) is not None
        and _sha(value["value"]) == value["sha256"]
    )


def _clock_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_uncertainty = int(left.get("uncertainty_ticks", 0))
    right_uncertainty = int(right.get("uncertainty_ticks", 0))
    left_interval = (
        max(0, int(left["capture_start_tick"]) - left_uncertainty),
        int(left["capture_end_tick"]) + left_uncertainty,
    )
    right_interval = (
        max(0, int(right["capture_start_tick"]) - right_uncertainty),
        int(right["capture_end_tick"]) + right_uncertainty,
    )
    return max(left_interval[0], right_interval[0]) <= min(left_interval[1], right_interval[1])


def _branch_replay_sha(branch: dict[str, Any], consequence: dict[str, Any] | None = None) -> str:
    state = consequence if consequence is not None else branch["consequence_state"]
    return _sha(
        {
            "fork_event_ref": branch["fork_event_ref"],
            "parent_state_sha256": branch["parent_state"]["sha256"],
            "intervention_sha256": branch["intervention"]["sha256"],
            "consequence_event_ref": branch["consequence_event_ref"],
            "consequence_state_sha256": state["sha256"],
        }
    )


def _verify_scenario(scenario: dict[str, Any], seed: int) -> list[str]:
    errors: list[str] = []
    if scenario.get("schema") != SCENARIO_SCHEMA:
        errors.append("scenario schema drift")
    if scenario.get("seed") != seed:
        errors.append("scenario seed drift")
    if tuple(scenario.get("operations", ())) != OPERATIONS:
        errors.append("scenario operation coverage drift")
    graph = scenario.get("graph")
    if not isinstance(graph, dict):
        return [*errors, "scenario graph missing"]
    if scenario.get("graph_sha256") != _sha(graph):
        errors.append("scenario graph digest drift")
    if graph.get("schema") != GRAPH_SCHEMA:
        errors.append("event graph schema drift")
    source = graph.get("source")
    if not _frozen_valid(source):
        errors.append("event graph source digest drift")
    elif isinstance(source, dict):
        source_value = source.get("value")
        if not isinstance(source_value, dict) or source_value.get("seed") != seed:
            errors.append("event graph source seed drift")

    event_rows = graph.get("events", [])
    entity_rows = graph.get("entities", [])
    observation_rows = graph.get("observations", [])
    branch_rows = graph.get("branches", [])
    events = {row.get("ref"): row for row in event_rows if isinstance(row, dict)}
    entities = {row.get("ref"): row for row in entity_rows if isinstance(row, dict)}
    observations = {row.get("ref"): row for row in observation_rows if isinstance(row, dict)}
    branches = {row.get("ref"): row for row in branch_rows if isinstance(row, dict)}
    if len(events) != len(event_rows):
        errors.append("event references are not unique")
    if len(entities) != len(entity_rows):
        errors.append("entity references are not unique")
    if len(observations) != len(observation_rows):
        errors.append("observation references are not unique")
    if len(branches) != len(branch_rows):
        errors.append("branch references are not unique")

    clock_refs: list[str] = []
    for event in event_rows:
        if not _frozen_valid(event.get("data")):
            errors.append(f"event {event.get('ref')} data digest drift")
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            errors.append(f"event {event.get('ref')} sequence invalid")
            continue
        for parent_ref in event.get("parent_event_refs", []):
            parent = events.get(parent_ref)
            if parent is None:
                errors.append(f"event {event.get('ref')} missing parent")
            elif not isinstance(parent.get("sequence"), int) or parent["sequence"] >= sequence:
                errors.append(f"event {event.get('ref')} parent is not earlier")
        for entity_ref in event.get("entity_refs", []):
            if entity_ref not in entities:
                errors.append(f"event {event.get('ref')} missing entity")
        branch_ref = event.get("branch_ref")
        if branch_ref is not None and branch_ref not in branches:
            errors.append(f"event {event.get('ref')} missing branch")

    for entity in entity_rows:
        if not _frozen_valid(entity.get("state")):
            errors.append(f"entity {entity.get('ref')} state digest drift")
        if entity.get("birth_event_ref") not in events:
            errors.append(f"entity {entity.get('ref')} missing birth event")
        for parent_ref in entity.get("parent_refs", []):
            if parent_ref not in entities:
                errors.append(f"entity {entity.get('ref')} missing lineage parent")

    for observation in observation_rows:
        if not _frozen_valid(observation.get("content")):
            errors.append(f"observation {observation.get('ref')} content digest drift")
        if observation.get("event_ref") not in events:
            errors.append(f"observation {observation.get('ref')} missing event")
        for entity_ref in observation.get("entity_refs", []):
            if entity_ref not in entities:
                errors.append(f"observation {observation.get('ref')} missing entity")
        clock = observation.get("clock")
        if not isinstance(clock, dict):
            errors.append(f"observation {observation.get('ref')} clock missing")
            continue
        clock_refs.append(str(clock.get("ref")))
        start = clock.get("capture_start_tick")
        end = clock.get("capture_end_tick")
        arrival = clock.get("arrival_tick")
        uncertainty = clock.get("uncertainty_ticks")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(arrival, int)
            or not isinstance(uncertainty, int)
        ):
            errors.append(f"observation {observation.get('ref')} capture interval invalid")
        else:
            if (
                start < 0
                or end < start
                or arrival < end
                or uncertainty < 0
            ):
                errors.append(f"observation {observation.get('ref')} capture interval invalid")
        ambiguous = set(observation.get("ambiguous_entity_refs", []))
        if not ambiguous <= set(observation.get("entity_refs", [])):
            errors.append(f"observation {observation.get('ref')} ambiguity references drift")
        if ambiguous and observation.get("abstention_required") is not True:
            errors.append(f"observation {observation.get('ref')} ambiguity lacks abstention")
    if len(set(clock_refs)) != len(clock_refs):
        errors.append("sensor clock references are not unique")

    branches_by_fork: dict[str, list[dict[str, Any]]] = {}
    for branch in branch_rows:
        for name in ("parent_state", "intervention", "consequence_state"):
            if not _frozen_valid(branch.get(name)):
                errors.append(f"branch {branch.get('ref')} {name} digest drift")
        if branch.get("fork_event_ref") not in events:
            errors.append(f"branch {branch.get('ref')} missing fork event")
        consequence = events.get(branch.get("consequence_event_ref"))
        if consequence is None:
            errors.append(f"branch {branch.get('ref')} missing consequence event")
        elif consequence.get("branch_ref") != branch.get("ref"):
            errors.append(f"branch {branch.get('ref')} consequence ownership drift")
        try:
            if branch.get("exact_replay_sha256") != _branch_replay_sha(branch):
                errors.append(f"branch {branch.get('ref')} exact replay drift")
        except (KeyError, TypeError):
            errors.append(f"branch {branch.get('ref')} exact replay fields invalid")
        branches_by_fork.setdefault(str(branch.get("fork_event_ref")), []).append(branch)
    for fork_ref, rows in branches_by_fork.items():
        if len({row.get("parent_state", {}).get("sha256") for row in rows}) != 1:
            errors.append(f"fork {fork_ref} parent state drift")
        if sum(row.get("chosen") is True for row in rows) != 1:
            errors.append(f"fork {fork_ref} chosen branch count drift")
        if len({row.get("intervention", {}).get("sha256") for row in rows}) != len(rows):
            errors.append(f"fork {fork_ref} intervention identity drift")

    operation_refs = scenario.get("operation_event_refs", {})
    if set(operation_refs) != set(OPERATIONS):
        errors.append("operation event reference coverage drift")
    if any(ref not in events for refs in operation_refs.values() for ref in refs):
        errors.append("operation references missing event")

    kind_rows: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        kind_rows.setdefault(str(row.get("kind")), []).append(row)
    if kind_rows.get("split") and kind_rows.get("merge"):
        split_data = kind_rows["split"][0]["data"]["value"]
        parent_ref = split_data.get("parent_ref")
        children = split_data.get("child_refs", [])
        if len(children) != 2 or any(
            entities.get(child, {}).get("parent_refs") != [parent_ref] for child in children
        ):
            errors.append("split lineage drift")
        merge_data = kind_rows["merge"][0]["data"]["value"]
        child_ref = merge_data.get("child_ref")
        if set(merge_data.get("parent_refs", [])) != set(children) or set(
            entities.get(child_ref, {}).get("parent_refs", [])
        ) != set(children):
            errors.append("merge lineage drift")

    probes = {row.get("control"): row for row in scenario.get("join_probes", [])}
    if set(probes) != {"wrong-time", "wrong-event", "appearance-only"}:
        errors.append("join control coverage drift")
    for name, probe in probes.items():
        left = observations.get(probe.get("left_observation_ref"))
        right = observations.get(probe.get("right_observation_ref"))
        if left is None or right is None:
            errors.append(f"{name} control observation missing")
            continue
        if probe.get("expected_control_accept") is not False:
            errors.append(f"{name} negative control marked accepted")
        if name == "wrong-time":
            if left.get("event_ref") != right.get("event_ref") or _clock_overlap(
                left["clock"], right["clock"]
            ):
                errors.append("wrong-time control relation drift")
        elif name == "wrong-event":
            if left.get("event_ref") == right.get("event_ref"):
                errors.append("wrong-event control uses one event")
            if not _clock_overlap(left["clock"], right["clock"]):
                errors.append("wrong-event control lacks overlapping clocks")
        elif name == "appearance-only":
            if left.get("content", {}).get("sha256") == right.get("content", {}).get("sha256"):
                errors.append("appearance-only control content did not change")
            if not set(left.get("entity_refs", [])) & set(right.get("entity_refs", [])):
                errors.append("appearance-only control lacks persistent entity")
    return errors


def _entry_core(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "entry_sha256"}


def _verify_lifecycle(
    lifecycle: dict[str, Any], event_refs: set[str]
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if lifecycle.get("schema") != LIFECYCLE_SCHEMA:
        errors.append("lifecycle schema drift")
    memory_ref = lifecycle.get("memory_ref")
    if not isinstance(memory_ref, str) or not memory_ref.startswith("memory:"):
        errors.append("lifecycle memory reference invalid")
    entries = lifecycle.get("entries", [])
    states: dict[int, dict[str, Any]] = {}
    current: dict[str, Any] = {
        "content": None,
        "availability_enabled": False,
        "available_from_tick": None,
        "available_until_tick": None,
        "deleted": False,
        "conflicted": False,
        "poisoned": False,
        "rollback_to_revision": None,
    }
    previous: str | None = None
    for index, entry in enumerate(entries):
        revision = index + 1
        if entry.get("memory_ref") != memory_ref:
            errors.append(f"lifecycle entry {index} memory reference drift")
        if entry.get("sequence") != index or entry.get("revision") != revision:
            errors.append(f"lifecycle entry {index} revision drift")
        if entry.get("event_ref") not in event_refs:
            errors.append(f"lifecycle entry {index} missing event")
        if entry.get("previous_entry_sha256") != previous:
            errors.append(f"lifecycle entry {index} previous digest drift")
        if entry.get("entry_sha256") != _sha(_entry_core(entry)):
            errors.append(f"lifecycle entry {index} digest drift")
        start = entry.get("available_from_tick")
        end = entry.get("available_until_tick")
        if (start is not None and (not isinstance(start, int) or start < 0)) or (
            end is not None and (not isinstance(end, int) or end < 0)
        ):
            errors.append(f"lifecycle entry {index} availability interval invalid")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            errors.append(f"lifecycle entry {index} availability interval invalid")

        operation = entry.get("operation")
        if operation in {"record", "revise"}:
            if not _frozen_valid(entry.get("content")):
                errors.append(f"lifecycle entry {index} content digest drift")
            if operation == "record" and current["content"] is not None:
                errors.append("lifecycle record replaced existing content")
            if operation == "revise" and (current["content"] is None or current["deleted"]):
                errors.append("lifecycle revise lacks active content")
            current = {
                "content": entry.get("content"),
                "availability_enabled": entry.get("availability_enabled") is True,
                "available_from_tick": start,
                "available_until_tick": end,
                "deleted": False,
                "conflicted": False,
                "poisoned": False,
                "rollback_to_revision": None,
            }
        elif operation == "availability":
            if current["content"] is None or current["deleted"]:
                errors.append("lifecycle availability lacks active content")
            current = {
                **current,
                "availability_enabled": entry.get("availability_enabled") is True,
                "available_from_tick": start,
                "available_until_tick": end,
            }
        elif operation == "rollback":
            target_revision = entry.get("target_revision")
            if not isinstance(target_revision, int) or target_revision >= revision:
                errors.append("lifecycle rollback target must be earlier")
                target = None
            else:
                target = states.get(target_revision)
            if target is None:
                errors.append("lifecycle rollback target missing")
            else:
                current = {**target, "rollback_to_revision": target_revision}
        elif operation == "delete":
            if current["content"] is None:
                errors.append("lifecycle delete lacks content")
            current = {**current, "deleted": True, "availability_enabled": False}
        elif operation == "conflict":
            if current["content"] is None or current["deleted"]:
                errors.append("lifecycle conflict lacks active content")
            current = {**current, "conflicted": True}
        elif operation == "poisoning":
            if current["content"] is None or current["deleted"]:
                errors.append("lifecycle poisoning lacks active content")
            current = {**current, "poisoned": True}
        else:
            errors.append(f"lifecycle entry {index} operation invalid")
        states[revision] = dict(current)
        previous = entry.get("entry_sha256")
    if lifecycle.get("head_sha256") != previous:
        errors.append("lifecycle head digest drift")
    return states, errors


def _available(state: dict[str, Any], tick: int) -> bool:
    if (
        state.get("content") is None
        or state.get("availability_enabled") is not True
        or state.get("deleted") is True
        or state.get("conflicted") is True
        or state.get("poisoned") is True
    ):
        return False
    start = state.get("available_from_tick")
    end = state.get("available_until_tick")
    return (start is None or tick >= int(start)) and (end is None or tick <= int(end))


def _verify_unit(record: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    errors: list[str] = []
    unit_meta = record.get("unit", {})
    artifact = record.get("artifact")
    if not isinstance(artifact, dict):
        return {}, ["unit artifact is not a mapping"]
    artifact_sha = _sha(artifact)
    if record.get("artifact_sha256") != artifact_sha:
        errors.append("unit outer artifact digest drift")
    if unit_meta.get("artifact_sha256") != artifact_sha:
        errors.append("unit contract artifact digest drift")
    if artifact.get("schema") != UNIT_SCHEMA:
        errors.append("unit schema drift")
    if artifact.get("seed") != unit_meta.get("seed"):
        errors.append("unit seed drift")
    scenario = artifact.get("scenario")
    if not isinstance(scenario, dict):
        return {}, [*errors, "unit scenario missing"]
    if artifact.get("scenario_sha256") != _sha(scenario):
        errors.append("unit scenario digest drift")
    errors.extend(_verify_scenario(scenario, int(artifact.get("seed", -1))))
    lifecycle = artifact.get("lifecycle")
    if not isinstance(lifecycle, dict):
        return {}, [*errors, "unit lifecycle missing"]
    if artifact.get("lifecycle_sha256") != _sha(lifecycle):
        errors.append("unit lifecycle digest drift")
    event_refs = {row.get("ref") for row in scenario["graph"].get("events", [])}
    states, lifecycle_errors = _verify_lifecycle(lifecycle, event_refs)
    errors.extend(lifecycle_errors)
    return states, errors


def _indexes(unit: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    scenario = unit["scenario"]
    graph = scenario["graph"]
    events = {row["ref"]: row for row in graph["events"]}
    entities = {row["ref"]: row for row in graph["entities"]}
    observations = {row["ref"]: row for row in graph["observations"]}
    branches = {row["ref"]: row for row in graph["branches"]}
    probes = {row["control"]: row for row in scenario["join_probes"]}
    return scenario, graph, events, entities, observations, branches, probes


def _kind(events: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [row for row in events.values() if row.get("kind") == name]


def _metrics_f23(unit: dict[str, Any]) -> dict[str, bool]:
    scenario, graph, events, entities, observations, _, probes = _indexes(unit)
    birth, transform = _kind(events, "birth")[0], _kind(events, "transform")[0]
    occlusion, reveal = _kind(events, "occlusion")[0], _kind(events, "reveal")[0]
    split, merge = _kind(events, "split")[0], _kind(events, "merge")[0]
    root = birth["entity_refs"][0]
    appearance = probes["appearance-only"]
    before = observations[appearance["left_observation_ref"]]
    after = observations[appearance["right_observation_ref"]]
    split_data = split["data"]["value"]
    children = split_data["child_refs"]
    merge_data = merge["data"]["value"]
    merged = merge_data["child_ref"]
    ambiguous = [row for row in observations.values() if row.get("ambiguous_entity_refs")]
    wrong_time = probes["wrong-time"]
    wt_left = observations[wrong_time["left_observation_ref"]]
    wt_right = observations[wrong_time["right_observation_ref"]]
    wrong_event = probes["wrong-event"]
    we_left = observations[wrong_event["left_observation_ref"]]
    we_right = observations[wrong_event["right_observation_ref"]]
    values = (
        scenario.get("graph_sha256") == _sha(graph)
        and all(_frozen_valid(row.get("data")) for row in graph["events"]),
        transform["entity_refs"] == [root]
        and root in before["entity_refs"]
        and root in after["entity_refs"]
        and before["content"]["sha256"] != after["content"]["sha256"],
        occlusion["entity_refs"] == reveal["entity_refs"] == [root]
        and occlusion["data"]["value"].get("visible") is False
        and reveal["data"]["value"].get("visible") is True,
        split_data["parent_ref"] == root
        and len(children) == 2
        and all(entities[child]["parent_refs"] == [root] for child in children),
        set(merge_data["parent_refs"]) == set(children)
        and set(entities[merged]["parent_refs"]) == set(children),
        len(ambiguous) == 1 and ambiguous[0].get("abstention_required") is True,
        wrong_time.get("expected_control_accept") is False
        and wt_left["event_ref"] == wt_right["event_ref"]
        and not _clock_overlap(wt_left["clock"], wt_right["clock"]),
        wrong_event.get("expected_control_accept") is False
        and we_left["event_ref"] != we_right["event_ref"]
        and _clock_overlap(we_left["clock"], we_right["clock"]),
        appearance.get("expected_control_accept") is False
        and before["content"]["sha256"] != after["content"]["sha256"]
        and bool(set(before["entity_refs"]) & set(after["entity_refs"])),
    )
    return dict(zip(METRICS["f23"], values, strict=True))


def _metrics_f29(unit: dict[str, Any]) -> dict[str, bool]:
    scenario, graph, _, _, _, branch_map, _ = _indexes(unit)
    branches = list(branch_map.values())
    parents = {row["parent_state"]["sha256"] for row in branches}
    interventions = {row["intervention"]["sha256"] for row in branches}
    consequences = {row["consequence_state"]["sha256"] for row in branches}
    values = (
        scenario.get("graph_sha256") == _sha(graph)
        and all(
            all(_frozen_valid(row[key]) for key in ("parent_state", "intervention", "consequence_state"))
            for row in branches
        ),
        len(branches) == 2 and len(parents) == 1,
        sum(row.get("chosen") is True for row in branches) == 1,
        len(interventions) == len(branches) == 2,
        len(consequences) == len(branches) == 2,
        all(row.get("exact_replay_sha256") == _branch_replay_sha(row) for row in branches),
        len(parents) == 1 and len(consequences) == 2,
        len(branches) == 2
        and all(
            _branch_replay_sha(row, branches[1 - index]["consequence_state"])
            != row["exact_replay_sha256"]
            for index, row in enumerate(branches)
        ),
    )
    return dict(zip(METRICS["f29"], values, strict=True))


def _metrics_f39(unit: dict[str, Any]) -> dict[str, bool]:
    lifecycle = unit["lifecycle"]
    event_refs = {row["ref"] for row in unit["scenario"]["graph"]["events"]}
    states, errors = _verify_lifecycle(lifecycle, event_refs)
    entries = lifecycle["entries"]
    final = states.get(len(entries), {})
    query = unit["availability_query"]
    observed = [_available(final, int(tick)) for tick in query["ticks"]]
    expected = unit["lifecycle_expectations"]
    target_revision = int(expected["rollback_target_revision"])
    pre_revision = int(expected["pre_rollback_revision"])
    target = states.get(target_revision, {})
    pre = states.get(pre_revision, {})
    values = (
        unit.get("lifecycle_sha256") == _sha(lifecycle) and not errors,
        all(
            row.get("sequence") == index and row.get("revision") == index + 1
            for index, row in enumerate(entries)
        ),
        all(row.get("event_ref") in event_refs for row in entries),
        observed == query["expected_available"],
        final.get("rollback_to_revision") == target_revision
        and final.get("content") == target.get("content"),
        pre.get("content") != final.get("content"),
        not _available(final, int(expected["unavailable_tick"])),
    )
    return dict(zip(METRICS["f39"], values, strict=True))


def _verify_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != HARNESS_SCHEMA:
        errors.append("harness schema drift")
    if tuple(contract.get("controls", ())) != CONTROLS:
        errors.append("harness control set drift")
    roles = {row.get("role") for row in contract.get("arms", [])}
    if roles != {"primary", "baseline", "negative-control"}:
        errors.append("harness arm roles drift")
    resources = contract.get("resources", {})
    if (
        resources.get("device") != "cpu"
        or resources.get("accelerator_required") is not False
        or resources.get("model_weights_loaded") is not False
    ):
        errors.append("harness resource envelope drift")
    stop = contract.get("stop", {})
    if (
        stop.get("minimum_independent_units", 0) < 3
        or stop.get("require_all_controls") is not True
        or stop.get("stop_on_identity_drift") is not True
        or stop.get("claim_scope") != CLAIM_SCOPE
    ):
        errors.append("harness stop contract drift")
    units = contract.get("independent_units", [])
    if len(units) < 3 or len({row.get("ref") for row in units}) != len(units):
        errors.append("harness independent units drift")
    return errors


def _core_hash(receipt: dict[str, Any]) -> tuple[str | None, str]:
    core = {
        key: value
        for key, value in receipt.items()
        if key not in {"core_payload_sha256", "independent_verifier", "payload_sha256"}
    }
    return receipt.get("core_payload_sha256"), _sha(core)


def verify_payload_sha256(receipt: dict[str, Any]) -> bool:
    expected = receipt.get("payload_sha256")
    if not isinstance(expected, str):
        return False
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    return expected == _sha(payload)


def _base_errors(receipt: dict[str, Any], *, check_live_files: bool) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if receipt.get("schema") != WAVE_SCHEMA:
        errors.append("wave schema drift")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        errors.append("wave claim scope drift")
    expected_core, actual_core = _core_hash(receipt)
    if expected_core != actual_core:
        errors.append("core payload digest drift")

    implementation = receipt.get("implementation", [])
    if tuple(row.get("path") for row in implementation) != IMPLEMENTATION_PATHS:
        errors.append("implementation path set drift")
    if check_live_files:
        for row in implementation:
            path = REPO_ROOT / str(row.get("path"))
            if not path.is_file() or _sha_file(path) != row.get("sha256"):
                errors.append(f"implementation digest drift at {row.get('path')}")

    contract = receipt.get("harness_contract")
    if not isinstance(contract, dict):
        return [*errors, "harness contract missing"], {}
    errors.extend(_verify_contract(contract))
    contract_sha = _sha(contract)
    if receipt.get("harness_contract_sha256") != contract_sha:
        errors.append("harness contract digest drift")

    shared_units = receipt.get("shared_units", [])
    unit_records = {row.get("unit", {}).get("ref"): row for row in shared_units}
    if len(unit_records) != len(shared_units):
        errors.append("shared unit references are not unique")
    contract_units = contract.get("independent_units", [])
    if [row.get("ref") for row in contract_units] != list(unit_records):
        errors.append("contract and shared unit order drift")
    unit_errors: dict[str, list[str]] = {}
    states: dict[str, dict[int, dict[str, Any]]] = {}
    for contract_unit in contract_units:
        ref = contract_unit.get("ref")
        record = unit_records.get(ref)
        if record is None:
            unit_errors[str(ref)] = ["shared unit missing"]
            continue
        if record.get("unit") != contract_unit:
            unit_errors[str(ref)] = ["shared unit metadata drift"]
        unit_states, found = _verify_unit(record)
        if found:
            unit_errors.setdefault(str(ref), []).extend(found)
        states[str(ref)] = unit_states
    for ref, found in unit_errors.items():
        errors.extend(f"{ref}: {message}" for message in found)

    evaluators = {"f23": _metrics_f23, "f29": _metrics_f29, "f39": _metrics_f39}
    results = receipt.get("sentinel_results", [])
    if [row.get("sentinel", {}).get("id") for row in results] != ["f23", "f29", "f39"]:
        errors.append("sentinel result order drift")
    recomputed: dict[str, dict[str, dict[str, bool]]] = {}
    for result in results:
        if result.get("schema") != RESULT_SCHEMA or result.get("claim_scope") != CLAIM_SCOPE:
            errors.append("sentinel result schema or scope drift")
            continue
        sentinel = result.get("sentinel", {})
        sentinel_id = sentinel.get("id")
        if sentinel_id not in evaluators:
            errors.append(f"unknown sentinel {sentinel_id}")
            continue
        if tuple(sentinel.get("metric_names", ())) != METRICS[sentinel_id]:
            errors.append(f"{sentinel_id} metric schema drift")
        if sentinel.get("harness_contract_sha256") != contract_sha:
            errors.append(f"{sentinel_id} harness identity drift")
        per_unit = result.get("per_unit", [])
        if [row.get("unit_ref") for row in per_unit] != [row.get("ref") for row in contract_units]:
            errors.append(f"{sentinel_id} independent unit order drift")
        recomputed[sentinel_id] = {}
        for row in per_unit:
            ref = row.get("unit_ref")
            record = unit_records.get(ref)
            if record is None:
                errors.append(f"{sentinel_id} result references missing unit {ref}")
                continue
            try:
                metrics = evaluators[sentinel_id](record["artifact"])
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                errors.append(f"{sentinel_id} independent metric replay failed for {ref}: {exc}")
                continue
            recomputed[sentinel_id][str(ref)] = metrics
            if row.get("metrics") != metrics:
                errors.append(f"{sentinel_id} metric drift for {ref}")
            if row.get("artifact_sha256") != record.get("artifact_sha256"):
                errors.append(f"{sentinel_id} artifact identity drift for {ref}")
            if row.get("all_metrics_pass") is not all(metrics.values()):
                errors.append(f"{sentinel_id} unit aggregate drift for {ref}")
        expected_pass = len(per_unit) >= 3 and all(
            all(metrics.values()) for metrics in recomputed[sentinel_id].values()
        )
        if result.get("all_units_pass") is not expected_pass:
            errors.append(f"{sentinel_id} aggregate drift")
        expected_status = "mechanics-pass" if expected_pass else "mechanics-fail"
        if result.get("status") != expected_status:
            errors.append(f"{sentinel_id} status drift")

    all_pass = len(recomputed) == 3 and all(
        len(rows) >= 3 and all(all(metrics.values()) for metrics in rows.values())
        for rows in recomputed.values()
    )
    if receipt.get("all_sentinels_pass") is not all_pass:
        errors.append("wave sentinel aggregate drift")
    expected_status = "mechanics-pass" if all_pass else "mechanics-fail"
    if receipt.get("status") != expected_status:
        errors.append("wave status drift")
    return errors, {
        "unit_count": len(unit_records),
        "sentinel_count": len(recomputed),
        "metric_count": sum(len(metric) for rows in recomputed.values() for metric in rows.values()),
        "all_recomputed_metrics_pass": all_pass,
        "live_implementation_checked": check_live_files,
    }


def _rehash_frozen(block: dict[str, Any]) -> None:
    block["sha256"] = _sha(block["value"])


def _rehash_graph(scenario: dict[str, Any]) -> None:
    scenario["graph_sha256"] = _sha(scenario["graph"])


def _rehash_journal(lifecycle: dict[str, Any]) -> None:
    previous: str | None = None
    for entry in lifecycle["entries"]:
        entry["previous_entry_sha256"] = previous
        entry["entry_sha256"] = _sha(_entry_core(entry))
        previous = entry["entry_sha256"]
    lifecycle["head_sha256"] = previous


def _mutation_tests(
    receipt: dict[str, Any], *, base_errors: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    first = receipt["shared_units"][0]["artifact"]
    seed = int(first["seed"])
    tests: list[tuple[str, str, list[str]]] = []

    source = copy.deepcopy(first["scenario"])
    source["graph"]["source"]["value"]["seed"] += 1000
    _rehash_frozen(source["graph"]["source"])
    _rehash_graph(source)
    tests.append(("source-bytes", "source seed drift", _verify_scenario(source, seed)))

    event_lineage = copy.deepcopy(first["scenario"])
    event_lineage["graph"]["events"][1]["parent_event_refs"] = ["event:missing/parent"]
    _rehash_graph(event_lineage)
    tests.append(("event-lineage", "missing parent", _verify_scenario(event_lineage, seed)))

    split = copy.deepcopy(first["scenario"])
    split_event = next(row for row in split["graph"]["events"] if row["kind"] == "split")
    child = split_event["data"]["value"]["child_refs"][0]
    next(row for row in split["graph"]["entities"] if row["ref"] == child)["parent_refs"] = []
    _rehash_graph(split)
    tests.append(("split-lineage", "split lineage drift", _verify_scenario(split, seed)))

    clock = copy.deepcopy(first["scenario"])
    clock["graph"]["observations"][0]["clock"]["capture_end_tick"] = -1
    _rehash_graph(clock)
    tests.append(("sensor-clock", "capture interval invalid", _verify_scenario(clock, seed)))

    branch = copy.deepcopy(first["scenario"])
    branch["graph"]["branches"][0]["parent_state"]["value"]["mutation"] = True
    _rehash_frozen(branch["graph"]["branches"][0]["parent_state"])
    _rehash_graph(branch)
    tests.append(("branch-parent-state", "parent state drift", _verify_scenario(branch, seed)))

    join = copy.deepcopy(first["scenario"])
    wrong_event = next(row for row in join["join_probes"] if row["control"] == "wrong-event")
    wrong_event["right_observation_ref"] = wrong_event["left_observation_ref"]
    tests.append(("control-join", "wrong-event control uses one event", _verify_scenario(join, seed)))

    lifecycle = copy.deepcopy(first["lifecycle"])
    rollback = next(row for row in lifecycle["entries"] if row["operation"] == "rollback")
    rollback["target_revision"] = rollback["revision"]
    _rehash_journal(lifecycle)
    event_refs = {row["ref"] for row in first["scenario"]["graph"]["events"]}
    _, lifecycle_errors = _verify_lifecycle(lifecycle, event_refs)
    tests.append(("lifecycle-rollback", "rollback target must be earlier", lifecycle_errors))

    lifecycle_event = copy.deepcopy(first["lifecycle"])
    lifecycle_event["entries"][0]["event_ref"] = "event:missing/source"
    _rehash_journal(lifecycle_event)
    _, lifecycle_event_errors = _verify_lifecycle(lifecycle_event, event_refs)
    tests.append(("lifecycle-event", "missing event", lifecycle_event_errors))

    contract = copy.deepcopy(receipt["harness_contract"])
    contract["controls"].pop()
    tests.append(("control-contract", "control set drift", _verify_contract(contract)))

    inherited = set(base_errors)
    rows: list[dict[str, Any]] = []
    for name, expected, errors in tests:
        mutation_errors = [error for error in errors if error not in inherited]
        rows.append(
            {
                "id": name,
                "expected_rejection": expected,
                "expected_rejection_observed": any(expected in error for error in mutation_errors),
                "rejected": bool(mutation_errors),
                "observed_errors": errors,
                "mutation_specific_errors": mutation_errors,
            }
        )
    return rows


def verify_receipt(
    receipt: dict[str, Any], *, run_mutations: bool = True, check_live_files: bool = True
) -> dict[str, Any]:
    errors, checks = _base_errors(receipt, check_live_files=check_live_files)
    mutations: list[dict[str, Any]] = []
    mutation_runner_error: str | None = None
    if run_mutations:
        try:
            mutations = _mutation_tests(receipt, base_errors=tuple(errors))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            mutation_runner_error = f"semantic mutation battery could not execute: {exc}"
            errors.append(mutation_runner_error)
    all_mutations_rejected = bool(mutations) and all(row["rejected"] is True for row in mutations)
    if run_mutations and not all_mutations_rejected:
        errors.append("semantic mutation rejection incomplete")
    return {
        "schema": VERIFIER_SCHEMA,
        "implementation": "independent raw JSON replay without driver metric imports",
        "claim_scope": CLAIM_SCOPE,
        "core_payload_sha256": receipt.get("core_payload_sha256"),
        "checks": checks,
        "mutation_tests": mutations,
        "mutation_runner_error": mutation_runner_error,
        "all_mutations_rejected": all_mutations_rejected,
        "errors": errors,
        "verified": not errors,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt",
        nargs="?",
        default=str(REPO_ROOT / "proof" / "EXPANSION_WAVE0.json"),
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--skip-live-files", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    report = verify_receipt(
        receipt,
        run_mutations=True,
        check_live_files=not args.skip_live_files,
    )
    envelope_ok = verify_payload_sha256(receipt)
    report["payload_sha256_verified"] = envelope_ok
    if args.report:
        _atomic_write(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["verified"] and envelope_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
