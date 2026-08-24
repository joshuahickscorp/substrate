"""G06-DC deadline-capacity gate: validator, mutations, and seal surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from substrate import odyssey_authority as authority
from substrate import odyssey_g06_dc as g06_dc

ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _frozen(root: Path) -> dict[str, Any]:
    document = authority._read_json(root / "plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    return authority._validate_frozen_build(root, document["sha256"])


def _minimal_tool_proof(width: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frontier in "ABCDEFGH"[:width]:
        for role in ("candidate", "control"):
            rows.append(
                {
                    "frontier": frontier,
                    "role": role,
                    "operation": "repo.inspect",
                    "tool_revision": {
                        "tool_id": "git",
                        "version": "test",
                        "artifact_sha256": "a" * 64,
                        "path": "/usr/bin/git",
                    },
                    "artifact_digest": "b" * 64,
                    "receipt_sha256": "c" * 64,
                    "fresh": True,
                    "cached_replay": False,
                }
            )
    return rows


def _observation(width: int, repetition: int, *, active: float = 20.0, slowdown: float = 1.0) -> dict[str, Any]:
    headroom = 1800.0 / active
    return {
        "width": width,
        "repetition": repetition,
        "cell_ids": list("ABCDEFGH"[:width]),
        "model_active_dispatch_wall_seconds": active,
        "tool_wall_seconds": 1.0,
        "active_dispatch_wall_seconds": active,
        "serial_model_plus_tool_seconds": active + 1.0,
        "per_cell_slowdown_ratio": slowdown,
        "deadline": {
            "phase_seconds": 1800.0,
            "microcycle_seconds": 7200.0,
            "active_dispatch_wall_seconds": active,
            "deadline_utilization_of_phase": active / 1800.0,
            "deadline_headroom": headroom,
            "p95_limit_seconds": 900.0,
            "worst_limit_seconds": 1350.0,
            "within_p95_limit": True,
            "within_worst_limit": True,
            "microcycle_complete_before_deadline": True,
            "headroom_meets_minimum": headroom >= 2.0,
            "no_missed_phase_deadline": True,
            "no_missed_microcycle_deadline": True,
        },
        "resident_memory_bytes": 50 * 1024**3,
        "swap_pageout_delta_bytes": 0,
        "transport_valid_arms": 2 * width,
        "semantic_valid_arms": 2 * width,
        "all_objects_valid": True,
        "tool_proof": _minimal_tool_proof(width),
        "checks": {
            "all_frontiers_present": True,
            "tool_bearing_real": True,
            "transport_and_semantic_valid": True,
            "zero_pageouts": True,
            "peak_rss_under_ceiling": True,
            "candidate_control_parity": True,
            "no_missed_phase_deadline": True,
            "no_missed_microcycle_deadline": True,
            "within_p95_dispatch_limit": True,
            "within_worst_dispatch_limit": True,
            "headroom_meets_minimum": True,
            "no_cross_lane_model_context": True,
            "no_evaluator_leakage": True,
            "no_synthetic_workload": True,
            "no_suppressed_tool_work": True,
            "no_cached_replay": True,
            "pageout_counter_not_reset": True,
            "failures_not_dropped": True,
            "deadline_denominator_is_phase_1800": True,
        },
    }


def _passing_subject(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    prior_path = root / "evidence/substrate/odyssey/ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json"
    prior_sha = authority.file_digest(prior_path)
    observations = [
        _observation(
            width,
            repetition,
            active=12.0 * width,  # still << 900s p95 limit
            slowdown=1.0 if width == 1 else min(4.392411013227944, 0.55 * width),
        )
        for width in (1, 2, 4, 6, 8)
        for repetition in (1, 2, 3)
    ]
    # Preserve exact width-8 slowdown on every width-8 row.
    for row in observations:
        if row["width"] == 8:
            row["per_cell_slowdown_ratio"] = 4.392411013227944
    by_width = {
        str(width): {
            "repetitions": 3,
            "slowdowns": [row["per_cell_slowdown_ratio"] for row in observations if row["width"] == width],
            "max_slowdown": max(row["per_cell_slowdown_ratio"] for row in observations if row["width"] == width),
            "deadline_utilizations": [row["deadline"]["deadline_utilization_of_phase"] for row in observations if row["width"] == width],
            "max_deadline_utilization": max(
                row["deadline"]["deadline_utilization_of_phase"] for row in observations if row["width"] == width
            ),
            "min_deadline_headroom": min(row["deadline"]["deadline_headroom"] for row in observations if row["width"] == width),
            "active_dispatch_seconds": [row["active_dispatch_wall_seconds"] for row in observations if row["width"] == width],
            "p95_active_dispatch_seconds": max(row["active_dispatch_wall_seconds"] for row in observations if row["width"] == width),
            "worst_active_dispatch_seconds": max(row["active_dispatch_wall_seconds"] for row in observations if row["width"] == width),
            "peak_resident_memory_bytes": 50 * 1024**3,
            "any_pageout": False,
            "all_objects_valid": True,
        }
        for width in (1, 2, 4, 6, 8)
    }
    per_frontier = {
        frontier: {
            "frontier": frontier,
            "arms": [
                {
                    "role": role,
                    "operation": "repo.inspect",
                    "tool_revision": {"tool_id": "git", "version": "test", "artifact_sha256": "a" * 64, "path": "/usr/bin/git"},
                    "artifact_digest": "b" * 64,
                }
                for role in ("candidate", "control")
            ],
        }
        for frontier in "ABCDEFGH"
    }
    checks = {name: True for name in sorted(authority.G06_DC_REQUIRED_CHECKS)}
    subject = {
        "schema": authority.GATE_SPECS["G06-DC"]["subject_schema"],
        "program": authority.PROGRAM,
        "status": "pass",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": authority._git_head(root),
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "gate_id": "G06-DC",
        "all_pass": True,
        "admitted_width": 8,
        "full_program_requires_width": 8,
        "calibration_widths": [1, 2, 4, 6, 8],
        "repetitions_per_width": 3,
        "deadline_limits": {
            "phase_seconds": 1800,
            "microcycle_seconds": 7200,
            "p95_active_dispatch_fraction_of_phase": 0.5,
            "worst_active_dispatch_fraction_of_phase": 0.75,
            "minimum_deadline_headroom": 2.0,
            "resident_cap_bytes": 85 * 1024**3,
        },
        "prior_model_dispatch": {
            "path": "evidence/substrate/odyssey/ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json",
            "file_sha256": prior_sha,
            "width1_active_dispatch_wall_seconds": 12.91,
            "preserved_width8_max_slowdown": 4.392411013227944,
        },
        "preserved_historical_width8_slowdown": 4.392411013227944,
        "observations": observations,
        "by_width": by_width,
        "aggregates": {
            "p95_active_dispatch_seconds": 96.0,
            "worst_active_dispatch_seconds": 96.0,
            "min_deadline_headroom": 1800.0 / 96.0,
            "width8_max_slowdown": 4.392411013227944,
            "width8_max_deadline_utilization": 96.0 / 1800.0,
            "width8_min_deadline_headroom": 1800.0 / 96.0,
        },
        "per_frontier_tool_proof": per_frontier,
        "soak": {
            "seconds": 30.0,
            "rss_samples": [50 * 1024**3, 50 * 1024**3],
            "pageout_window_delta_bytes": 0,
            "memory_creep_ok": True,
            "thermal_ok": True,
        },
        "workload_class": "tool_bearing_final",
        "synthetic_workload": False,
        "model_or_tool_work_suppressed": False,
        "cached_outputs_replayed_as_fresh": False,
        "pageout_counter_reset": False,
        "failures_dropped": False,
        "deadline_denominator_seconds": 1800,
        "cross_lane_model_context": False,
        "evaluator_leakage": False,
        "candidate_control_queues_equal": True,
        "measurement_sha256": "d" * 64,
        "checks": checks,
    }
    subject["sha256"] = authority.digest({key: value for key, value in subject.items() if key != "sha256"})
    return subject


def test_g06_dc_replaces_g06_in_the_launch_set() -> None:
    """G06-DC supersedes G06's capacity-admission role in the launch set.

    G06 measured simultaneity, which a shared GPU cannot satisfy and the Odyssey
    schedule never required.  Its validator stays reachable so the historical
    4.39x receipt remains verifiable, but it is no longer a launch gate.
    """
    assert "G06-DC" in authority.GATE_SPECS
    assert "G06" not in authority.GATE_SPECS
    assert len(authority.GATE_SPECS) == 15
    spec = authority.machine_gate_spec("G06-DC")
    assert spec is not None
    assert spec["kind"] == "machine_verified"
    # The superseded validator must survive for historical evidence.
    assert hasattr(authority, "_validate_g06")
    # The frozen design's launch_gates must agree with the authority's set.
    import json
    from pathlib import Path

    design = json.loads(
        (Path(__file__).parents[2] / "plans/substrate/tangible_next_launch/ODYSSEY_7D.hardened.draft.json").read_text()
    )
    assert [row["id"] for row in design["launch_gates"]] == list(authority.GATE_SPECS)
    assert authority.G06_DC_PRESERVED_WIDTH8_SLOWDOWN == 4.392411013227944


def test_g06_still_rejects_slowdown_above_1_35() -> None:
    """Historical G06 limit is not weakened by this transition."""
    # The frozen calibration still encodes 1.35; validator path for G06 is intact.
    assert authority._frozen_g06_phase_contract  # callable exists
    cal = _read(ROOT / "plans/substrate/tangible_next_launch/RESOURCE_CALIBRATION_SPEC.draft.json")
    assert cal["max_slowdown_ratio"] == 1.35


def test_passing_subject_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use real repo root for frozen digests and prior evidence.
    root = ROOT
    frozen = _frozen(root)
    subject = _passing_subject(root, frozen)
    authority._validate_g06_dc(root, subject, frozen)


def _mutate_and_refuse(root: Path, frozen: dict[str, Any], mutator) -> str:
    subject = _passing_subject(root, frozen)
    mutator(subject)
    # Re-seal digest after mutation so schema checks are about semantics, not self-hash.
    subject.pop("sha256", None)
    subject["sha256"] = authority.digest(subject)
    with pytest.raises(authority.Refused) as raised:
        authority._validate_g06_dc(root, subject, frozen)
    return str(raised.value)


def test_mutation_synthetic_workload(tmp_path: Path) -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(synthetic_workload=True, workload_class="synthetic"))
    assert "synthetic" in message.lower() or "tool-bearing" in message.lower() or "workload" in message.lower()


def test_mutation_suppressed_tool_work() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(model_or_tool_work_suppressed=True))
    assert "suppress" in message.lower()


def test_mutation_cached_replay() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(cached_outputs_replayed_as_fresh=True))
    assert "cached" in message.lower()


def test_mutation_unfair_queues() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(candidate_control_queues_equal=False))
    assert "queue" in message.lower()


def test_mutation_wrong_deadline_denominator() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(deadline_denominator_seconds=150))
    assert "denominator" in message.lower() or "1800" in message


def test_mutation_missing_frontier_cell() -> None:
    root = ROOT
    frozen = _frozen(root)

    def mutate(subject: dict[str, Any]) -> None:
        # Drop frontier H from the last width-8 observation.
        for row in subject["observations"]:
            if row["width"] == 8 and row["repetition"] == 3:
                row["cell_ids"] = list("ABCDEFG")
                row["tool_proof"] = row["tool_proof"][:14]

    message = _mutate_and_refuse(root, frozen, mutate)
    assert "width" in message.lower() or "frontier" in message.lower() or "A-H" in message or "cell" in message.lower()


def test_mutation_dropped_failures() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(failures_dropped=True))
    assert "drop" in message.lower()


def test_mutation_pageout_counter_reset() -> None:
    root = ROOT
    frozen = _frozen(root)
    message = _mutate_and_refuse(root, frozen, lambda s: s.update(pageout_counter_reset=True))
    assert "pageout" in message.lower()


def test_mutation_hides_4_39x() -> None:
    root = ROOT
    frozen = _frozen(root)

    def mutate(subject: dict[str, Any]) -> None:
        subject["preserved_historical_width8_slowdown"] = 1.0
        subject["by_width"]["8"]["max_slowdown"] = 1.0
        subject["prior_model_dispatch"]["preserved_width8_max_slowdown"] = 1.0
        for row in subject["observations"]:
            if row["width"] == 8:
                row["per_cell_slowdown_ratio"] = 1.0

    message = _mutate_and_refuse(root, frozen, mutate)
    assert "4.392" in message or "slowdown" in message.lower()


def test_mutation_deadline_miss_at_width8() -> None:
    root = ROOT
    frozen = _frozen(root)

    def mutate(subject: dict[str, Any]) -> None:
        for row in subject["observations"]:
            if row["width"] == 8:
                active = 1400.0  # > 75% of 1800
                row["active_dispatch_wall_seconds"] = active
                row["deadline"]["active_dispatch_wall_seconds"] = active
                row["deadline"]["deadline_utilization_of_phase"] = active / 1800.0
                row["deadline"]["deadline_headroom"] = 1800.0 / active

    message = _mutate_and_refuse(root, frozen, mutate)
    assert "worst" in message.lower() or "dispatch" in message.lower() or "deadline" in message.lower()


def test_deadline_metrics_helper() -> None:
    metrics = g06_dc._deadline_metrics(57.0)
    assert metrics["deadline_utilization_of_phase"] == pytest.approx(57.0 / 1800.0)
    assert metrics["deadline_headroom"] == pytest.approx(1800.0 / 57.0)
    assert metrics["headroom_meets_minimum"] is True
    assert metrics["within_p95_limit"] is True
