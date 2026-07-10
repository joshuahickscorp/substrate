#!/usr/bin/env python
"""Build the deterministic mechanics-only Wave E0 composite receipt.

The driver creates three programmatic independent units, binds them to one shared harness, runs
F23, F29, and F39 as thin sentinels, and invokes the separately implemented mutation verifier.
It does not load model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.environments.scenario_factory import make_scenario
from mop.experiments.expansion_harness import (
    CLAIM_SCOPE,
    IndependentUnit,
    SentinelSpec,
    UnitArtifact,
    make_contract,
    run_sentinel,
)
from mop.substrate.events import FrozenJSON, canonical_sha256
from mop.substrate.lifecycle import LifecycleJournal, MemoryRef

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WAVE_SCHEMA = "mop-expansion-wave0/v1"
UNIT_SCHEMA = "mop-expansion-wave0-unit/v1"
IMPLEMENTATION_PATHS = (
    "src/mop/substrate/events.py",
    "src/mop/environments/scenario_factory.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/experiments/expansion_harness.py",
    "scripts/mop_expansion_wave0.py",
    "scripts/verify_expansion_wave0.py",
)
F23_METRICS = (
    "event-bytes-bound",
    "transform-identity-persists",
    "occlusion-identity-persists",
    "split-lineage-exact",
    "merge-lineage-exact",
    "ambiguity-requires-abstention",
    "wrong-time-rejected",
    "wrong-event-rejected",
    "appearance-only-rejected",
)
F29_METRICS = (
    "branch-bytes-bound",
    "same-state-branches",
    "one-chosen-branch",
    "distinct-interventions",
    "distinct-consequences",
    "exact-replay",
    "action-blind-exposed",
    "action-shuffled-rejected",
)
F39_METRICS = (
    "journal-bytes-bound",
    "revisions-monotonic",
    "event-lineage-shared",
    "availability-forecast-exact",
    "rollback-exact",
    "stale-memory-rejected",
    "unavailable-memory-abstains",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_ok(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and "value" in payload
        and isinstance(payload.get("sha256"), str)
        and canonical_sha256(payload["value"]) == payload["sha256"]
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


def _scenario_indexes(unit: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    scenario = unit["scenario"]
    graph = scenario["graph"]
    events = {row["ref"]: row for row in graph["events"]}
    entities = {row["ref"]: row for row in graph["entities"]}
    observations = {row["ref"]: row for row in graph["observations"]}
    branches = {row["ref"]: row for row in graph["branches"]}
    probes = {row["control"]: row for row in scenario["join_probes"]}
    return scenario, graph, events, entities, observations, branches, probes


def _events_of_kind(events: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [row for row in events.values() if row.get("kind") == kind]


def _f23_metrics(unit: dict[str, Any]) -> dict[str, bool]:
    scenario, graph, events, entities, observations, _, probes = _scenario_indexes(unit)
    event_bytes_bound = scenario.get("graph_sha256") == canonical_sha256(graph) and all(
        _frozen_ok(row.get("data")) for row in graph["events"]
    )

    birth = _events_of_kind(events, "birth")[0]
    transform = _events_of_kind(events, "transform")[0]
    occlusion = _events_of_kind(events, "occlusion")[0]
    reveal = _events_of_kind(events, "reveal")[0]
    split = _events_of_kind(events, "split")[0]
    merge = _events_of_kind(events, "merge")[0]
    root_ref = birth["entity_refs"][0]

    transform_probe = probes["appearance-only"]
    before = observations[transform_probe["left_observation_ref"]]
    after = observations[transform_probe["right_observation_ref"]]
    transform_identity = (
        transform["entity_refs"] == [root_ref]
        and root_ref in before["entity_refs"]
        and root_ref in after["entity_refs"]
        and before["content"]["sha256"] != after["content"]["sha256"]
    )
    occlusion_identity = (
        occlusion["entity_refs"] == reveal["entity_refs"] == [root_ref]
        and occlusion["data"]["value"].get("visible") is False
        and reveal["data"]["value"].get("visible") is True
    )

    split_data = split["data"]["value"]
    split_children = split_data.get("child_refs", [])
    split_lineage = (
        split_data.get("parent_ref") == root_ref
        and len(split_children) == 2
        and all(entities[child]["parent_refs"] == [root_ref] for child in split_children)
    )
    merge_data = merge["data"]["value"]
    merged_ref = merge_data.get("child_ref")
    merge_lineage = (
        isinstance(merged_ref, str)
        and set(merge_data.get("parent_refs", [])) == set(split_children)
        and set(entities[merged_ref]["parent_refs"]) == set(split_children)
    )

    ambiguous = [row for row in observations.values() if row.get("ambiguous_entity_refs")]
    ambiguity_abstains = len(ambiguous) == 1 and ambiguous[0].get("abstention_required") is True

    wrong_time = probes["wrong-time"]
    wt_left = observations[wrong_time["left_observation_ref"]]
    wt_right = observations[wrong_time["right_observation_ref"]]
    wrong_time_rejected = (
        wrong_time.get("expected_control_accept") is False
        and wt_left["event_ref"] == wt_right["event_ref"]
        and not _clock_overlap(wt_left["clock"], wt_right["clock"])
    )
    wrong_event = probes["wrong-event"]
    we_left = observations[wrong_event["left_observation_ref"]]
    we_right = observations[wrong_event["right_observation_ref"]]
    wrong_event_rejected = (
        wrong_event.get("expected_control_accept") is False
        and we_left["event_ref"] != we_right["event_ref"]
        and _clock_overlap(we_left["clock"], we_right["clock"])
    )
    appearance_only_rejected = (
        transform_probe.get("expected_control_accept") is False
        and before["content"]["sha256"] != after["content"]["sha256"]
        and bool(set(before["entity_refs"]) & set(after["entity_refs"]))
    )
    return dict(
        zip(
            F23_METRICS,
            (
                event_bytes_bound,
                transform_identity,
                occlusion_identity,
                split_lineage,
                merge_lineage,
                ambiguity_abstains,
                wrong_time_rejected,
                wrong_event_rejected,
                appearance_only_rejected,
            ),
            strict=True,
        )
    )


def _branch_replay_sha(branch: dict[str, Any], consequence: dict[str, Any] | None = None) -> str:
    consequence_state = consequence if consequence is not None else branch["consequence_state"]
    return canonical_sha256(
        {
            "fork_event_ref": branch["fork_event_ref"],
            "parent_state_sha256": branch["parent_state"]["sha256"],
            "intervention_sha256": branch["intervention"]["sha256"],
            "consequence_event_ref": branch["consequence_event_ref"],
            "consequence_state_sha256": consequence_state["sha256"],
        }
    )


def _f29_metrics(unit: dict[str, Any]) -> dict[str, bool]:
    scenario, graph, _, _, _, branches_by_ref, _ = _scenario_indexes(unit)
    branches = list(branches_by_ref.values())
    branch_bytes_bound = scenario.get("graph_sha256") == canonical_sha256(graph) and all(
        all(_frozen_ok(row[key]) for key in ("parent_state", "intervention", "consequence_state"))
        for row in branches
    )
    parent_hashes = {row["parent_state"]["sha256"] for row in branches}
    intervention_hashes = {row["intervention"]["sha256"] for row in branches}
    consequence_hashes = {row["consequence_state"]["sha256"] for row in branches}
    exact_replay = all(row["exact_replay_sha256"] == _branch_replay_sha(row) for row in branches)
    shuffled_rejected = len(branches) == 2 and all(
        _branch_replay_sha(row, branches[1 - index]["consequence_state"])
        != row["exact_replay_sha256"]
        for index, row in enumerate(branches)
    )
    values = (
        branch_bytes_bound,
        len(branches) == 2 and len(parent_hashes) == 1,
        sum(row.get("chosen") is True for row in branches) == 1,
        len(intervention_hashes) == len(branches) == 2,
        len(consequence_hashes) == len(branches) == 2,
        exact_replay,
        len(parent_hashes) == 1 and len(consequence_hashes) == 2,
        shuffled_rejected,
    )
    return dict(zip(F29_METRICS, values, strict=True))


def _entry_core(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "entry_sha256"}


def _replay_lifecycle(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    problems: list[str] = []
    entries = payload.get("entries", [])
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
        if entry.get("sequence") != index or entry.get("revision") != revision:
            problems.append(f"entry {index} revision drift")
        if entry.get("previous_entry_sha256") != previous:
            problems.append(f"entry {index} previous digest drift")
        if canonical_sha256(_entry_core(entry)) != entry.get("entry_sha256"):
            problems.append(f"entry {index} digest drift")
        operation = entry.get("operation")
        if operation in {"record", "revise"}:
            if not _frozen_ok(entry.get("content")):
                problems.append(f"entry {index} content digest drift")
            current = {
                "content": entry.get("content"),
                "availability_enabled": entry.get("availability_enabled") is True,
                "available_from_tick": entry.get("available_from_tick"),
                "available_until_tick": entry.get("available_until_tick"),
                "deleted": False,
                "conflicted": False,
                "poisoned": False,
                "rollback_to_revision": None,
            }
        elif operation == "availability":
            current = {
                **current,
                "availability_enabled": entry.get("availability_enabled") is True,
                "available_from_tick": entry.get("available_from_tick"),
                "available_until_tick": entry.get("available_until_tick"),
            }
        elif operation == "rollback":
            target = states.get(int(entry.get("target_revision", -1)))
            if target is None:
                problems.append(f"entry {index} rollback target missing")
            else:
                current = {**target, "rollback_to_revision": entry.get("target_revision")}
        elif operation == "delete":
            current = {**current, "deleted": True, "availability_enabled": False}
        elif operation == "conflict":
            current = {**current, "conflicted": True}
        elif operation == "poisoning":
            current = {**current, "poisoned": True}
        else:
            problems.append(f"entry {index} unsupported operation")
        states[revision] = dict(current)
        previous = entry.get("entry_sha256")
    if payload.get("head_sha256") != previous:
        problems.append("journal head digest drift")
    return states, problems


def _available_at(state: dict[str, Any], tick: int) -> bool:
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


def _f39_metrics(unit: dict[str, Any]) -> dict[str, bool]:
    lifecycle = unit["lifecycle"]
    states, problems = _replay_lifecycle(lifecycle)
    entries = lifecycle["entries"]
    graph_event_refs = {row["ref"] for row in unit["scenario"]["graph"]["events"]}
    final = states.get(len(entries), {})
    query = unit["availability_query"]
    ticks = [int(tick) for tick in query["ticks"]]
    observed = [_available_at(final, tick) for tick in ticks]
    target_revision = int(unit["lifecycle_expectations"]["rollback_target_revision"])
    pre_rollback_revision = int(unit["lifecycle_expectations"]["pre_rollback_revision"])
    target = states.get(target_revision, {})
    pre_rollback = states.get(pre_rollback_revision, {})
    revisions_monotonic = all(
        row.get("sequence") == index and row.get("revision") == index + 1
        for index, row in enumerate(entries)
    )
    values = (
        unit.get("lifecycle_sha256") == canonical_sha256(lifecycle) and not problems,
        revisions_monotonic,
        all(row.get("event_ref") in graph_event_refs for row in entries),
        observed == query["expected_available"],
        final.get("rollback_to_revision") == target_revision
        and final.get("content") == target.get("content"),
        pre_rollback.get("content") != final.get("content"),
        not _available_at(final, int(unit["lifecycle_expectations"]["unavailable_tick"])),
    )
    return dict(zip(F39_METRICS, values, strict=True))


def _build_unit(seed: int) -> UnitArtifact:
    scenario = make_scenario(seed=seed)
    graph = scenario.graph
    events_by_kind = {
        kind: [row for row in graph.events if row.kind == kind] for kind in {row.kind for row in graph.events}
    }
    birth = events_by_kind["birth"][0]
    transform = events_by_kind["transform"][0]
    delay = events_by_kind["delay"][0]
    consequences = {row.data.value()["action"]: row for row in events_by_kind["consequence"]}
    root_ref = str(birth.entity_refs[0])
    chosen_branch = next(row for row in graph.branches if row.chosen)

    journal = LifecycleJournal(MemoryRef(f"memory:{scenario.sha256[:16]}/persistent-referent"))
    journal.record(
        birth.ref,
        {"entity_ref": root_ref, "event_data_sha256": birth.data.sha256, "phase": "birth"},
        available_from_tick=0,
        available_until_tick=4,
        reason="bind the initial persistent referent",
    )
    journal.revise(
        transform.ref,
        {
            "entity_ref": root_ref,
            "event_data_sha256": transform.data.sha256,
            "phase": "transformed",
        },
        available_from_tick=0,
        available_until_tick=4,
        reason="preserve identity through transformation",
    )
    journal.set_availability(
        delay.ref,
        available=True,
        available_from_tick=6,
        available_until_tick=9,
        reason="apply delayed availability window",
    )
    journal.revise(
        consequences["translate"].ref,
        {
            "entity_ref": str(consequences["translate"].entity_refs[0]),
            "state_sha256": chosen_branch.consequence_state.sha256,
            "phase": "intervened",
        },
        available_from_tick=6,
        available_until_tick=12,
        reason="record chosen intervention consequence",
    )
    journal.rollback(
        consequences["hold"].ref,
        2,
        reason="restore the verified pre-intervention revision",
    )
    journal.set_availability(
        consequences["translate"].ref,
        available=True,
        available_from_tick=6,
        available_until_tick=9,
        reason="restore bounded post-rollback availability",
    )
    graph_event_refs = {str(row.ref) for row in graph.events}
    problems = journal.verify(event_refs=graph_event_refs)
    if problems:
        raise RuntimeError("constructed lifecycle failed validation: " + "; ".join(problems))

    unit_payload = {
        "schema": UNIT_SCHEMA,
        "seed": seed,
        "scenario": scenario.payload(),
        "scenario_sha256": scenario.sha256,
        "lifecycle": journal.payload(),
        "lifecycle_sha256": journal.sha256,
        "availability_query": {
            "ticks": [5, 7, 10],
            "expected_available": [False, True, False],
        },
        "lifecycle_expectations": {
            "rollback_target_revision": 2,
            "pre_rollback_revision": 4,
            "unavailable_tick": 10,
        },
    }
    content = FrozenJSON.from_value(unit_payload)
    unit = IndependentUnit(
        ref=f"unit:wave-e0-seed-{seed}",
        seed=seed,
        artifact_sha256=content.sha256,
    )
    return UnitArtifact(unit=unit, content=content)


def build_receipt(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Wave E0 requires at least three unique nonnegative seeds")
    artifacts_in_order = tuple(_build_unit(seed) for seed in seeds)
    artifacts = {row.unit.ref: row for row in artifacts_in_order}
    contract = make_contract([row.unit for row in artifacts_in_order])
    specs = (
        SentinelSpec(
            id="f23",
            title="Persistent Referent Identity",
            metric_names=F23_METRICS,
            harness_contract_sha256=contract.sha256,
        ),
        SentinelSpec(
            id="f29",
            title="Controllability Boundary",
            metric_names=F29_METRICS,
            harness_contract_sha256=contract.sha256,
        ),
        SentinelSpec(
            id="f39",
            title="Memory Availability Forecast",
            metric_names=F39_METRICS,
            harness_contract_sha256=contract.sha256,
        ),
    )
    evaluators = {"f23": _f23_metrics, "f29": _f29_metrics, "f39": _f39_metrics}
    results = tuple(
        run_sentinel(
            spec=spec,
            contract=contract,
            artifacts=artifacts,
            evaluator=evaluators[spec.id],
        )
        for spec in specs
    )
    implementation = [
        {"path": path, "sha256": _sha256_file(REPO_ROOT / path)} for path in IMPLEMENTATION_PATHS
    ]
    core: dict[str, Any] = {
        "schema": WAVE_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if all(row.all_units_pass for row in results) else "mechanics-fail",
        "determinism": {
            "seeds": list(seeds),
            "random_source": "SHA-256 labeled integer derivation",
            "timestamp_omitted": True,
        },
        "implementation": implementation,
        "shared_units": [row.payload() for row in artifacts_in_order],
        "harness_contract": contract.payload(),
        "harness_contract_sha256": contract.sha256,
        "sentinel_results": [row.payload() for row in results],
        "all_sentinels_pass": all(row.all_units_pass for row in results),
    }
    core["core_payload_sha256"] = canonical_sha256(core)

    from scripts.verify_expansion_wave0 import verify_receipt

    verification = verify_receipt(core, run_mutations=True, check_live_files=True)
    if verification["verified"] is not True:
        raise RuntimeError("independent Wave E0 verification failed: " + "; ".join(verification["errors"]))
    receipt = {**core, "independent_verifier": verification}
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "EXPANSION_WAVE0.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = build_receipt(_parse_seeds(args.seeds))
    output = Path(args.out)
    _atomic_write(output, receipt)
    print(
        f"wrote {output}: {len(receipt['shared_units'])} units, "
        f"{len(receipt['sentinel_results'])} mechanics sentinels, {receipt['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
