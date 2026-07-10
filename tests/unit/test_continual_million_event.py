from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from scripts.continual_million_event_preflight import DEFAULT_CONFIG, _load_config, build_preflight

from mop.studies.continual_million_event import ContinualSmokeProfile, run_smoke_arm
from mop.substrate.continual_stream import (
    ContinualStreamSpec,
    TransitionSchedule,
    event_at,
    iter_stream,
    materialize_stream,
    verify_stream,
)


def _spec(schedule: TransitionSchedule = TransitionSchedule.ABRUPT) -> ContinualStreamSpec:
    return ContinualStreamSpec(
        seed=41,
        total_events=96,
        chunk_events=16,
        n_domains=3,
        n_classes=4,
        transition_schedule=schedule,
        gradual_width_events=8,
        deletion_event=48,
    )


def _profile() -> ContinualSmokeProfile:
    return ContinualSmokeProfile(
        checkpoint_every=11,
        replay_capacity=64,
        future_window_events=16,
        threshold_window_events=8,
        future_accuracy_threshold=0.5,
    )


def test_abrupt_and_gradual_schedules_are_typed_deterministic_and_distinct(tmp_path: Path):
    abrupt, gradual = _spec(), _spec(TransitionSchedule.GRADUAL)
    abrupt_manifest = materialize_stream(tmp_path / "abrupt", abrupt)
    gradual_manifest = materialize_stream(tmp_path / "gradual", gradual)
    abrupt_events = list(iter_stream(tmp_path / "abrupt"))
    gradual_events = list(iter_stream(tmp_path / "gradual"))

    assert abrupt_manifest["stream_sha256"] != gradual_manifest["stream_sha256"]
    assert all(event.blend_milli == 0 for event in abrupt_events)
    assert any(0 < event.blend_milli < 1000 for event in gradual_events)
    assert any(event.transition for event in abrupt_events)
    assert any(event.transition for event in gradual_events)
    assert str(abrupt_events[0].event_ref).startswith("event:")
    assert str(abrupt_events[0].entity_ref).startswith("entity:")
    assert event_at(tmp_path / "abrupt", 95) == abrupt_events[-1]


def test_chunk_materialization_resumes_atomically_and_matches_replica(tmp_path: Path):
    spec = _spec(TransitionSchedule.GRADUAL)
    partial = materialize_stream(tmp_path / "primary", spec, max_new_chunks=1)
    assert partial["generated_events"] == spec.chunk_events
    assert partial["complete"] is False
    assert verify_stream(tmp_path / "primary", expected_spec=spec, require_complete=False)["verified"]

    complete = materialize_stream(tmp_path / "primary", spec)
    replica = materialize_stream(tmp_path / "replica", spec)
    assert complete["complete"] is True
    assert complete["stream_sha256"] == replica["stream_sha256"]
    assert complete["chain_head_sha256"] == replica["chain_head_sha256"]
    assert verify_stream(tmp_path / "primary", expected_spec=spec)["record_count"] == 96


def test_stream_verifier_rejects_chunk_byte_mutation(tmp_path: Path):
    spec = _spec()
    materialize_stream(tmp_path, spec)
    chunk = tmp_path / "chunk_000001.bin"
    raw = bytearray(chunk.read_bytes())
    raw[7] ^= 1
    chunk.write_bytes(raw)
    audit = verify_stream(tmp_path, expected_spec=spec)
    assert audit["verified"] is False
    assert any("digest drift" in error for error in audit["errors"])


def test_arm_cursor_resume_is_exact_and_deletion_is_exercised(tmp_path: Path):
    spec, profile = _spec(), _profile()
    stream = tmp_path / "stream"
    materialize_stream(stream, spec)
    checkpoint = tmp_path / "resume.json"
    partial = run_smoke_arm(
        stream_root=stream,
        spec=spec,
        arm="replay",
        profile=profile,
        checkpoint_path=checkpoint,
        event_budget=27,
    )
    assert partial["complete"] is False and partial["next_sequence"] == 27
    resumed = run_smoke_arm(
        stream_root=stream,
        spec=spec,
        arm="replay",
        profile=profile,
        checkpoint_path=checkpoint,
    )
    fresh = run_smoke_arm(
        stream_root=stream,
        spec=spec,
        arm="replay",
        profile=profile,
        checkpoint_path=tmp_path / "fresh.json",
    )
    assert resumed["resumed_from_atomic_checkpoint"] is True
    assert resumed["metrics"] == fresh["metrics"]
    assert resumed["state_sha256"] == fresh["state_sha256"]
    assert resumed["lifecycle_sha256"] == fresh["lifecycle_sha256"]
    assert resumed["metrics"]["deletion"]["replay_records_removed"] > 0
    assert resumed["metrics"]["deletion"]["complete"] is True


def test_checkpoint_refuses_arm_identity_drift(tmp_path: Path):
    spec, profile = _spec(), _profile()
    stream = tmp_path / "stream"
    materialize_stream(stream, spec)
    checkpoint = tmp_path / "checkpoint.json"
    run_smoke_arm(
        stream_root=stream,
        spec=spec,
        arm="replay",
        profile=profile,
        checkpoint_path=checkpoint,
        event_budget=9,
    )
    with pytest.raises(ValueError, match="identity drift"):
        run_smoke_arm(
            stream_root=stream,
            spec=spec,
            arm="no-replay",
            profile=profile,
            checkpoint_path=checkpoint,
        )


def test_all_controls_emit_preregistered_matched_metrics(tmp_path: Path):
    spec, profile = _spec(TransitionSchedule.GRADUAL), _profile()
    stream = tmp_path / "stream"
    materialize_stream(stream, spec)
    results = {
        arm: run_smoke_arm(
            stream_root=stream,
            spec=spec,
            arm=arm,
            profile=profile,
            checkpoint_path=tmp_path / f"{arm}.json",
        )
        for arm in ("replay", "no-replay", "fresh-init")
    }
    expected_metrics = {
        "retention",
        "acquisition",
        "future_learnability",
        "stale_memory",
        "deletion",
        "resources",
    }
    assert all(set(result["metrics"]) == expected_metrics for result in results.values())
    assert all(result["all_mechanics_ok"] for result in results.values())
    assert all(result["metrics"]["resources"]["updates_per_event"] == 2.0 for result in results.values())
    assert results["replay"]["metrics"]["resources"]["replay_samples"] > 0
    assert results["no-replay"]["metrics"]["resources"]["replay_samples"] == 0
    assert results["fresh-init"]["controls"]["reset_count"] == spec.n_domains - 1


def test_preflight_is_no_heavy_complete_and_repeatable(tmp_path: Path):
    first = build_preflight(DEFAULT_CONFIG, tmp_path / "work")
    second = build_preflight(DEFAULT_CONFIG, tmp_path / "work")
    assert first["config"]["payload"]["null_hypothesis"].startswith(
        "the bounded stream cannot preserve exact identity"
    )
    assert first["status"] == "mechanics-pass"
    assert first["all_mechanics_ok"] is True
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first["checks"] == {
        "both_transition_schedules": True,
        "all_streams_verified": True,
        "all_identity_replicas_match": True,
        "all_arms_resumed_atomically": True,
        "all_metric_families_present": True,
        "all_deletions_complete": True,
        "all_compute_matched": True,
        "no_model_weights_loaded": True,
    }
    assert first["remaining_full_run_gate"]["target_events_per_stream"] == 1_000_000
    assert first["remaining_full_run_gate"]["hardware_boundary_earned"] is False


def test_no_heavy_guard_refuses_million_event_preflight_config(tmp_path: Path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    unsafe = copy.deepcopy(config)
    unsafe["stream"]["total_events"] = 1_000_000
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(unsafe))
    with pytest.raises(ValueError, match="no-heavy guard"):
        _load_config(path)
