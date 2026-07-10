from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from scripts.mop_expansion_wave0 import build_receipt, main
from scripts.verify_expansion_wave0 import verify_payload_sha256, verify_receipt

from mop.environments.scenario_factory import OPERATIONS, make_scenario
from mop.experiments.expansion_harness import CLAIM_SCOPE, REQUIRED_CONTROLS
from mop.substrate.events import EntityRef, EventRef, FrozenJSON, SensorClock, SensorClockRef
from mop.substrate.lifecycle import LifecycleJournal, MemoryRef


def test_typed_refs_and_exact_json_are_deeply_immutable():
    ref = EntityRef("entity:test/root")
    with pytest.raises(FrozenInstanceError):
        ref.value = "entity:test/changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="entity reference"):
        EntityRef("event:test/wrong-namespace")

    frozen = FrozenJSON.from_value({"nested": {"value": 1}})
    decoded = frozen.value()
    decoded["nested"]["value"] = 9
    assert frozen.value() == {"nested": {"value": 1}}
    assert FrozenJSON(frozen.canonical, frozen.sha256) == frozen
    with pytest.raises(ValueError, match="digest mismatch"):
        FrozenJSON(frozen.canonical, "0" * 64)


def test_sensor_clocks_express_capture_delay_and_uncertainty():
    visual = SensorClock(SensorClockRef("clock:test/visual"), "vision", 10, 12, 12, 1)
    audio = SensorClock(SensorClockRef("clock:test/audio"), "audio", 9, 11, 18, 1)
    distant = SensorClock(SensorClockRef("clock:test/distant"), "control", 30, 31, 31)
    assert visual.overlaps(audio)
    assert not visual.overlaps(distant)
    with pytest.raises(ValueError, match="capture interval"):
        SensorClock(SensorClockRef("clock:test/invalid"), "vision", 3, 2, 3)


def test_scenario_factory_is_deterministic_complete_and_seed_distinct():
    first = make_scenario(seed=4)
    repeated = make_scenario(seed=4)
    other = make_scenario(seed=5)
    assert first.payload() == repeated.payload()
    assert first.sha256 == repeated.sha256
    assert first.sha256 != other.sha256
    assert tuple(dict(first.operation_event_refs)) == OPERATIONS
    assert first.validate() == []
    assert {row.kind for row in first.graph.events} >= {
        "transform",
        "occlusion",
        "reveal",
        "split",
        "merge",
        "delay",
        "intervention",
        "consequence",
    }
    ambiguous = [row for row in first.graph.observations if row.ambiguous_entity_refs]
    assert len(ambiguous) == 1 and ambiguous[0].abstention_required
    delayed = next(row for row in first.graph.observations if row.clock.sensor_id == "audio")
    assert delayed.clock.arrival_tick > delayed.clock.capture_end_tick


def test_lifecycle_is_append_only_hash_linked_available_and_rollback_safe():
    journal = LifecycleJournal(MemoryRef("memory:test/referent"))
    events = [EventRef(f"event:test/e{index}") for index in range(7)]
    journal.record(events[0], {"version": 1}, available_until_tick=3)
    journal.revise(events[1], {"version": 2}, available_until_tick=4)
    journal.set_availability(events[2], available=True, available_from_tick=6, available_until_tick=9)
    journal.revise(events[3], {"version": 3}, available_from_tick=6, available_until_tick=12)
    journal.rollback(events[4], 2)
    journal.set_availability(events[5], available=True, available_from_tick=6, available_until_tick=9)

    assert journal.verify(event_refs={str(ref) for ref in events}) == []
    assert journal.forecast((5, 7, 10)) == (False, True, False)
    assert journal.state_at().content == journal.state_at(revision=2).content
    assert journal.state_at().rollback_to_revision == 2
    assert isinstance(journal.entries, tuple)
    with pytest.raises(FrozenInstanceError):
        journal.entries[0].reason = "changed"  # type: ignore[misc]

    journal.mark_conflict(events[6])
    assert not journal.state_at().available_at(7)
    journal.rollback(events[6], 6)
    assert journal.state_at().available_at(7)
    journal.mark_poisoned(events[6])
    assert not journal.state_at().available_at(7)
    journal.rollback(events[6], 8)
    journal.delete(events[6])
    assert not journal.state_at().exists


def test_wave_receipt_uses_one_contract_and_shared_unit_identities():
    receipt = build_receipt()
    assert receipt["schema"] == "mop-expansion-wave0/v1"
    assert receipt["claim_scope"] == CLAIM_SCOPE
    assert receipt["status"] == "mechanics-pass"
    assert receipt["all_sentinels_pass"] is True
    assert tuple(receipt["harness_contract"]["controls"]) == REQUIRED_CONTROLS
    resources = receipt["harness_contract"]["resources"]
    assert resources["device"] == "cpu"
    assert resources["accelerator_required"] is False
    assert resources["model_weights_loaded"] is False

    expected = {row["unit"]["ref"]: row["artifact_sha256"] for row in receipt["shared_units"]}
    assert len(expected) == 3
    for result in receipt["sentinel_results"]:
        assert result["sentinel"]["harness_contract_sha256"] == receipt["harness_contract_sha256"]
        assert {row["unit_ref"]: row["artifact_sha256"] for row in result["per_unit"]} == expected
        assert result["all_units_pass"] is True
        assert result["claim_scope"] == CLAIM_SCOPE


def test_independent_verifier_recomputes_metrics_and_rejects_semantic_mutations():
    receipt = build_receipt()
    embedded = receipt["independent_verifier"]
    assert embedded["verified"] is True
    assert embedded["checks"]["metric_count"] == 72
    assert embedded["all_mutations_rejected"] is True
    assert {row["id"] for row in embedded["mutation_tests"]} == {
        "source-bytes",
        "event-lineage",
        "split-lineage",
        "sensor-clock",
        "branch-parent-state",
        "control-join",
        "lifecycle-rollback",
        "lifecycle-event",
        "control-contract",
    }
    assert verify_payload_sha256(receipt)
    replay = verify_receipt(receipt, run_mutations=True, check_live_files=True)
    assert replay["verified"] is True
    assert replay["errors"] == []


def test_independent_verifier_fails_closed_on_result_tampering():
    receipt = build_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["sentinel_results"][0]["per_unit"][0]["metrics"]["event-bytes-bound"] = False
    replay = verify_receipt(tampered, run_mutations=False, check_live_files=False)
    assert replay["verified"] is False
    assert any("metric drift" in error or "core payload digest drift" in error for error in replay["errors"])
    assert not verify_payload_sha256(tampered)


def test_independent_verifier_still_runs_mutations_when_base_receipt_is_invalid():
    receipt = build_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["status"] = "mechanics-fail"
    replay = verify_receipt(tampered, run_mutations=True, check_live_files=False)
    assert replay["verified"] is False
    assert len(replay["mutation_tests"]) == 9
    assert replay["all_mutations_rejected"] is True
    assert replay["mutation_runner_error"] is None


def test_driver_output_is_byte_deterministic(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(["--out", str(first)]) == 0
    assert main(["--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text())
    assert verify_payload_sha256(parsed)
