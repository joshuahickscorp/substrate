from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_full_generations_wave as wave
from mop.studies import generation1_new_mechanisms_queue as newq
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import atomic_write_json


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fake_parent(mechanics_lanes: list[str]) -> dict[str, Any]:
    return {
        "bindings": {
            "program_manifest": {
                "path": "configs/parent.json",
                "file_sha256": "1" * 64,
                "program_sha256": "2" * 64,
            },
            "supervisor_status": {
                "path": "runs/parent/status.json",
                "file_sha256": "3" * 64,
                "status_sha256": "4" * 64,
            },
            "result": {
                "path": "proof/parent.json",
                "file_sha256": "5" * 64,
                "result_sha256": "6" * 64,
            },
            "verification": {
                "path": "proof/parent.verify.json",
                "file_sha256": "7" * 64,
                "verification_sha256": "8" * 64,
            },
            "report_receipt": {
                "path": "runs/parent/report.json",
                "file_sha256": "9" * 64,
                "receipt_sha256": "a" * 64,
            },
        },
        "mechanics_lanes": mechanics_lanes,
    }


def _materialize_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mechanics_lanes: list[str],
) -> Path:
    monkeypatch.setattr(wave, "REPO_ROOT", tmp_path)
    parent = _fake_parent(mechanics_lanes)
    monkeypatch.setattr(wave, "_validated_parent", lambda **_kwargs: parent)
    root = tmp_path / "runs/fullgen"
    for gate_index in range(len(wave.GATE_IDS)):
        wave.materialize_gate(root=root, gate_index=gate_index)
    return root


def test_taxonomy_cycles_and_real_mechanics_envelope_are_exact() -> None:
    assert wave.PROGRAM_ID == "generation1-full-generations-wave-v1"
    assert wave.GATE_IDS == (
        "admit_wave_v1",
        "carry_d1_freeze",
        "rescreen_d1_redesign",
        "admit_new_lanes",
        "freeze_routing",
    )
    assert wave.EPOCH_IDS == (
        "W08", "W09", "W10", "W11", "W12", "W13", "W14",
        "W15", "W16", "W17", "W18", "W19", "W20", "W21",
    )
    assert wave.EPOCH_CYCLES == (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    assert wave.CATEGORY_LANES == {
        "formation_trace": ("G1-C0", "G1-E1"),
        "communication_repair": ("G1-V1", "G1-M1", "G1-K1"),
        "memory_plasticity": ("G1-R1", "G1-P1R"),
        "action_simulation": ("G1-A1", "G1-S1"),
        "construction": ("G1-G1",),
        "dispatch_redesign": ("G1-D1",),
        "uncertainty_curiosity": ("G1-U1", "G1-N1"),
    }
    assert tuple(c.order for c in wave.CATEGORIES) == (0, 1, 2, 3, 4, 5, 6)
    assert wave.NEW_LANE_IDS == ("G1-U1", "G1-N1", "G1-P1R")
    # The old G1-P1 lane is retired and never appears; its redesigned successor G1-P1R replaces it.
    assert "G1-P1" not in wave.I1_DEPENDENCIES
    assert wave.I1_DEPENDENCIES == ("G1-E1", "G1-D1", "G1-M1", "G1-V1", "G1-R1", "G1-P1R")
    all_category_lanes = {lane for cat in wave.CATEGORIES for lane in cat.lane_ids}
    assert "G1-P1" not in all_category_lanes
    assert wave.CAPSULE_COUNT == 123
    # The balanced planning shard count stays eight; the idle-host worker pool is tuned to sixteen.
    assert wave.INTERNAL_SHARD_COUNT == 8
    assert wave.IDLE_WORKERS == 16
    assert wave.MAXIMUM_RAW_RECEIPT_COUNT == 35_255
    assert wave.MAXIMUM_RAW_RECEIPT_COUNT == 14 * 2_509 + 129
    assert wave.planned_serial_hours() == pytest.approx(wave.planned_program_compute_seconds() / 3_600)
    assert wave.planned_ideal_worker_hours() == pytest.approx(
        wave.planned_serial_hours() / wave.IDLE_WORKERS
    )
    assert 380.0 < wave.planned_serial_hours() < 440.0


def test_dual_work_item_tables_use_fresh_cycles_and_disjoint_seed_space() -> None:
    old_only = {"formation_trace", "communication_repair", "action_simulation", "construction", "dispatch_redesign"}
    for epoch_index, cycle in enumerate(wave.EPOCH_CYCLES):
        cycle_keys: list[str] = []
        for category_id in wave.CATEGORY_IDS:
            works = wave.category_work_items(epoch_index, category_id)
            assert {wave._work_lane_id(work) for work in works} == set(wave.CATEGORY_LANES[category_id])
            assert all(work.cycle == cycle for work in works)
            origins = {work.origin for work in works}
            if category_id in old_only:
                assert origins == {wave._OLD_ORIGIN}
            elif category_id == "uncertainty_curiosity":
                assert origins == {wave._NEW_ORIGIN}
            else:  # memory_plasticity is mixed: carried G1-R1 plus redesigned G1-P1R
                assert origins == {wave._OLD_ORIGIN, wave._NEW_ORIGIN}
            cycle_keys.extend(work.key for work in works)
        assert len(cycle_keys) == len(set(cycle_keys)) == 2_509

    i1 = wave.integration_work_items()
    assert len(i1) == 129
    assert {wave._work_lane_id(work) for work in i1} == {"G1-I1"}
    assert all(work.cycle == 32 and work.origin == wave._OLD_ORIGIN for work in i1)

    # Old-lane fresh items shift by the same cycle stride as the sealed consolidated mapping.
    first_old = wave._fresh_item(wave.category_work_items(0, "formation_trace")[0])
    last_old = wave._fresh_item(wave.category_work_items(13, "formation_trace")[0])
    assert last_old.seed_start - first_old.seed_start == 13 * consolidated.MECHANICS_CYCLE_STRIDE
    # New-lane fresh items use the identical offset math over the new-mechanisms table.
    p1r = next(w for w in wave.category_work_items(0, "memory_plasticity") if w.origin == wave._NEW_ORIGIN)
    source = newq.NEW_WORK_ITEMS[p1r.source_index]
    fresh = wave._fresh_item(p1r)
    expected = source.seed_start + consolidated.MECHANICS_FRESH_BASE + 19 * consolidated.MECHANICS_CYCLE_STRIDE
    assert fresh.seed_start == expected
    assert fresh.phase == f"fresh_c19_{source.phase}"


def test_artifact_paths_and_validation_dispatch_by_origin(tmp_path: Path) -> None:
    old_work = wave.category_work_items(0, "formation_trace")[0]
    new_work = next(w for w in wave.category_work_items(0, "uncertainty_curiosity") if w.origin == wave._NEW_ORIGIN)
    old_path = wave._artifact_path(tmp_path, old_work)
    new_path = wave._artifact_path(tmp_path, new_work)
    assert old_path.parent.name == "mechanics_fresh"
    assert new_path.parent.name == wave._NEW_ARTIFACT_KIND
    assert old_path != new_path


def test_gates_round_trip_and_never_synthesize_redesign_efficacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lanes = [lane.lane_id for lane in mechanics.LANES if lane.lane_id != "G1-I1"]
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=lanes)

    admit = wave._read_object(root / "gates/admit_wave_v1.json")
    assert admit["payload"]["operation"] == "bind_clean_categorized_batch_wave_v1"
    assert admit["payload"]["mechanics_lanes"] == lanes

    carry = wave._read_object(root / "gates/carry_d1_freeze.json")
    assert carry["payload"]["carried_forward"] is True
    assert carry["payload"]["old_d1_retired"] is True
    assert carry["payload"]["old_d1_resurrection_allowed"] is False
    assert carry["payload"]["d1_redesign_efficacy_execution_authorized"] is False

    rescreen = wave._read_object(root / "gates/rescreen_d1_redesign.json")
    assert rescreen["payload"]["candidate_evidence_count"] == 0
    assert rescreen["payload"]["eligible_for_future_authority"] is False
    assert rescreen["payload"]["execution_authorized"] is False
    assert rescreen["payload"]["screen_state"] == "no_candidate_execution_evidence"

    admit_new = wave._read_object(root / "gates/admit_new_lanes.json")
    admitted_ids = [row["lane_id"] for row in admit_new["payload"]["new_lanes"]]
    assert admitted_ids == ["G1-U1", "G1-N1", "G1-P1R"]
    authorities = admit_new["payload"]["mechanism_authorities"]
    assert set(authorities) == {"calibrated_uncertainty", "reducible_novelty", "stability_plasticity_r2"}
    for mechanism, rows in authorities.items():
        assert set(rows) == {"bed", "runner"}
        assert rows["bed"]["path"] == f"src/mop/mechanisms/{mechanism}_bed.py"
        assert rows["runner"]["path"] == f"src/mop/mechanisms/{mechanism}_runner.py"
        assert all(len(rows[role]["file_sha256"]) == 64 for role in ("bed", "runner"))
    substitution = admit_new["payload"]["i1_dependency_substitution"]
    assert substitution["old_dependency"] == "G1-P1"
    assert substitution["new_dependency"] == "G1-P1R"
    assert substitution["substituted_dependency_lane_ids"] == list(wave.I1_DEPENDENCIES)
    assert "proof/parent.json" in substitution["justification"]
    assert admit_new["payload"]["execution_authorized"] is False

    freeze = wave._read_object(root / "gates/freeze_routing.json")
    assert freeze["payload"]["admitted_new_lane_ids"] == ["G1-U1", "G1-N1", "G1-P1R"]
    assert freeze["payload"]["i1_dependency_lane_ids"] == list(wave.I1_DEPENDENCIES)
    assert freeze["payload"]["i1_initially_eligible"] is True
    route = freeze["payload"]["route_seed"]
    assert route["memory_plasticity"]["eligible_lane_ids"] == ["G1-R1", "G1-P1R"]
    assert route["uncertainty_curiosity"]["eligible_lane_ids"] == ["G1-U1", "G1-N1"]
    for gate_index in range(len(wave.GATE_IDS)):
        gate = wave._read_object(wave._gate_path(root, wave.GATE_IDS[gate_index]))
        wave.validate_gate(gate, gate_index, root=root)


def test_freeze_gate_seal_and_field_mutations_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=["G1-C0"])
    freeze = wave._read_object(wave._gate_path(root, "freeze_routing"))
    tampered = dict(freeze)
    payload = dict(tampered["payload"])
    payload["i1_initially_eligible"] = not payload["i1_initially_eligible"]
    tampered["payload"] = payload
    with pytest.raises(ValueError):
        wave.validate_gate(tampered, len(wave.GATE_IDS) - 1, root=root)


def test_new_lane_category_runs_through_new_dispatch_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    canaries = {
        lane_id: next(item for item in newq.NEW_WORK_ITEMS if item.lane_id == lane_id and item.phase == "canary")
        for lane_id in ("G1-U1", "G1-N1")
    }
    works = tuple(
        wave.WaveWorkItem(
            key=f"w08_new_{lane_id.lower()}",
            origin=wave._NEW_ORIGIN,
            source_index=canaries[lane_id].index,
            cycle=19,
        )
        for lane_id in ("G1-U1", "G1-N1")
    )
    original_fresh_new = wave._fresh_new_item

    def reduced_fresh_new(work: wave.WaveWorkItem) -> mechanics.WorkItem:
        fresh = original_fresh_new(work)
        return mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=2,
        )

    monkeypatch.setattr(wave, "_fresh_new_item", reduced_fresh_new)
    monkeypatch.setattr(
        wave,
        "category_work_items",
        lambda _epoch, category: works if category == "uncertainty_curiosity" else (),
    )

    def execute(raw_root: Path, pending: tuple[wave.WaveWorkItem, ...]):
        receipts: dict[str, dict[str, Any]] = {}
        durations: dict[str, float] = {}
        for work in pending:
            payload = newq._run_item_payload(wave._fresh_new_item(work))
            atomic_write_json(wave._new_artifact_path(raw_root, work), payload)
            receipts[work.key] = payload
            durations[work.key] = 0.01
        return receipts, durations, len(pending)

    monkeypatch.setattr(wave, "_execute_pending", execute)
    value = wave.run_category(root=root, epoch_index=0, category_id="uncertainty_curiosity")

    assert value["execution"]["executed_item_count"] == 2
    assert value["execution"]["compute_started"] is True
    assert value["execution"]["eligible_lane_ids"] == ["G1-U1", "G1-N1"]
    assert value["lane_results"]["G1-U1"]["continue_lane"] is True
    assert value["lane_results"]["G1-N1"]["continue_lane"] is True
    assert {row["origin"] for row in value["artifact_index"]} == {wave._NEW_ORIGIN}
    assert {row["lane_id"] for row in value["artifact_index"]} == {"G1-U1", "G1-N1"}
    # Re-validation round-trips against the reduced new-lane expectation.
    wave.validate_category(value, epoch_index=0, category_id="uncertainty_curiosity", root=root)


def test_pruned_dispatch_category_executes_nothing_and_cannot_resurrect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    called = False

    def forbidden(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("pruned category executed")

    monkeypatch.setattr(wave, "_execute_pending", forbidden)
    value = wave.run_category(root=root, epoch_index=0, category_id="dispatch_redesign")
    assert called is False
    assert value["execution"]["executed_item_count"] == 0
    assert value["execution"]["skipped_item_count"] == 129
    assert value["execution"]["compute_started"] is False
    assert value["redesign_v2_efficacy"]["execution_authorized"] is False
    assert value["redesign_v2_efficacy"]["carried_forward"] is True
    assert value["redesign_v2_efficacy"]["candidate_evidence_count"] == 0


def test_routing_chain_and_anti_resurrection_over_two_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=["G1-C0", "G1-E1"])
    original_fresh = consolidated.fresh_mechanics_item
    source_by_lane = {
        lane_id: next(item for item in mechanics.WORK_ITEMS if item.lane_id == lane_id)
        for lane_id in ("G1-C0", "G1-E1")
    }

    def tiny_category_work_items(epoch_index: int, category_id: str) -> tuple[wave.WaveWorkItem, ...]:
        if category_id != "formation_trace":
            return ()
        return tuple(
            wave.WaveWorkItem(
                key=f"partial_{wave.EPOCH_IDS[epoch_index].lower()}_{lane_id.lower()}",
                origin=wave._OLD_ORIGIN,
                source_index=source_by_lane[lane_id].index,
                cycle=wave.EPOCH_CYCLES[epoch_index],
            )
            for lane_id in ("G1-C0", "G1-E1")
        )

    def reduced_fresh(work: consolidated.WorkItem) -> mechanics.WorkItem:
        fresh = original_fresh(work)
        if not str(work.key).startswith("partial_"):
            return fresh
        return mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=1,
        )

    executed_lane_batches: list[tuple[str, ...]] = []

    def execute(raw_root: Path, pending: tuple[wave.WaveWorkItem, ...]):
        executed_lane_batches.append(tuple(wave._work_lane_id(work) for work in pending))
        receipts: dict[str, dict[str, Any]] = {}
        durations: dict[str, float] = {}
        for work in pending:
            payload = mechanics._run_item_payload(reduced_fresh(wave._consolidated_work(work)))
            atomic_write_json(wave._artifact_path(raw_root, work), payload)
            receipts[work.key] = payload
            durations[work.key] = 0.01
        return receipts, durations, len(pending)

    original_lane_verdicts = wave._lane_verdicts

    def lane_verdicts(works, receipts):
        rows = original_lane_verdicts(works, receipts)
        if any(str(work.key).startswith("partial_w08_") for work in works) and "G1-C0" in rows:
            rows["G1-C0"] = {
                **rows["G1-C0"],
                "verdict_counts": {"mechanics-warning": 1},
                "continue_lane": False,
            }
        return rows

    monkeypatch.setattr(wave, "category_work_items", tiny_category_work_items)
    monkeypatch.setattr(consolidated, "fresh_mechanics_item", reduced_fresh)
    monkeypatch.setattr(wave, "_execute_pending", execute)
    monkeypatch.setattr(wave, "_lane_verdicts", lane_verdicts)

    for category_id in wave.CATEGORY_IDS:
        wave.run_category(root=root, epoch_index=0, category_id=category_id)
    first = wave.classify_epoch(root=root, epoch_index=0)
    assert first["routing"]["formation_trace"]["eligible_lane_ids"] == ["G1-E1"]
    assert first["routing"]["formation_trace"]["pruned_lane_ids"] == ["G1-C0"]

    second_category = wave.run_category(root=root, epoch_index=1, category_id="formation_trace")
    for category_id in wave.CATEGORY_IDS:
        if category_id != "formation_trace":
            wave.run_category(root=root, epoch_index=1, category_id=category_id)
    second = wave.classify_epoch(root=root, epoch_index=1)

    assert executed_lane_batches == [("G1-C0", "G1-E1"), ("G1-E1",)]
    assert second_category["execution"]["eligible_item_count"] == 1
    assert second_category["execution"]["pruned_item_count"] == 1
    assert second_category["lane_results"]["G1-E1"]["continue_lane"] is True
    assert {row["lane_id"] for row in second_category["artifact_index"]} == {"G1-E1"}
    pruned_c0 = tiny_category_work_items(1, "formation_trace")[0]
    assert not wave._artifact_path(
        wave._category_raw_root(root, 1, "formation_trace"),
        pruned_c0,
    ).exists()
    assert second["routing"]["formation_trace"]["eligible_lane_ids"] == ["G1-E1"]
    assert second["routing"]["formation_trace"]["pruned_lane_ids"] == ["G1-C0"]


def _fake_final_classification(root: Path, retained: dict[str, list[str]]) -> None:
    classification_path = wave._classification_path(root, len(wave.EPOCH_IDS) - 1)
    routing = {
        category_id: {
            "eligible_lane_ids": [
                lane_id for lane_id in wave.CATEGORY_LANES[category_id] if lane_id in set(retained.get(category_id, []))
            ]
        }
        for category_id in wave.CATEGORY_IDS
    }
    _write(classification_path, {"routing": routing, "classification_sha256": "f" * 64})


def test_i1_eligible_iff_substituted_p1r_dependency_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_deps = [dep for dep in wave.I1_DEPENDENCIES if dep in {lane.lane_id for lane in mechanics.LANES}]
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=old_deps)
    monkeypatch.setattr(wave, "validate_classification", lambda *_args, **_kwargs: None)

    retained_full = {category_id: list(lanes) for category_id, lanes in wave.CATEGORY_LANES.items()}
    _fake_final_classification(root, retained_full)
    _binding, initially, eligible = wave._i1_eligible(root)
    assert initially is True
    assert eligible is True

    # Prune the substituted G1-P1R dependency: I1 must no longer be eligible.
    retained_no_p1r = {category_id: list(lanes) for category_id, lanes in wave.CATEGORY_LANES.items()}
    retained_no_p1r["memory_plasticity"] = ["G1-R1"]
    _fake_final_classification(root, retained_no_p1r)
    _binding, initially, eligible = wave._i1_eligible(root)
    assert initially is True
    assert eligible is False


def test_pruned_i1_seals_substituted_dependencies_without_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_deps = [dep for dep in wave.I1_DEPENDENCIES if dep in {lane.lane_id for lane in mechanics.LANES}]
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=old_deps)
    monkeypatch.setattr(wave, "validate_classification", lambda *_args, **_kwargs: None)
    retained_no_p1r = {category_id: list(lanes) for category_id, lanes in wave.CATEGORY_LANES.items()}
    retained_no_p1r["memory_plasticity"] = ["G1-R1"]
    _fake_final_classification(root, retained_no_p1r)

    called = False

    def forbidden(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("pruned I1 executed")

    monkeypatch.setattr(wave, "_execute_pending", forbidden)
    value = wave.run_integration(root=root)

    assert called is False
    assert value["execution"]["eligible"] is False
    assert value["execution"]["executed_item_count"] == 0
    assert value["execution"]["skipped_item_count"] == len(wave.integration_work_items())
    assert value["dependency_lane_ids"] == list(wave.I1_DEPENDENCIES)
    assert "G1-P1R" in value["dependency_lane_ids"]
    assert "G1-P1" not in value["dependency_lane_ids"]
    assert value["i1_initially_eligible"] is True


def test_release_audit_advisory_seals_regardless_of_audit_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mop.studies import generation1_release_audit as release_audit

    synthetic_report = {
        "schema": release_audit.SCHEMA,
        "release_complete": False,
        "artifact_bundle_complete": False,
        "problems": ["synthetic incomplete bundle"],
        "audit_sha256": "b" * 64,
    }
    monkeypatch.setattr(release_audit, "audit_generation1_release", lambda _paths: dict(synthetic_report))

    value = wave.build_release_audit()
    wave.validate_release_audit(value)
    assert value["advisory"] is True
    assert value["audit_exit_code"] == 1
    assert value["audit_report"] == synthetic_report
    assert value["complete"] is True
    assert value["problems"] == []
    assert value["activation_allowed"] is False
    assert value["scientific_promotion"] is False
    assert value["independent_scientific_confirmation"] is False

    # A complete audit derives a zero exit code but still seals a valid advisory artifact.
    monkeypatch.setattr(
        release_audit,
        "audit_generation1_release",
        lambda _paths: {**synthetic_report, "release_complete": True},
    )
    complete_value = wave.build_release_audit()
    wave.validate_release_audit(complete_value)
    assert complete_value["audit_exit_code"] == 0
    assert complete_value["advisory"] is True
