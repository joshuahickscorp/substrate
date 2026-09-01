from __future__ import annotations

import json

import pytest

from substrate import v5config as C
from substrate import v5experiment as E
from substrate import v5sensorium as sensors


def test_v5_frozen_generator_covers_curriculum_and_arms() -> None:
    manifest = E.generator_manifest()
    assert manifest["phase_count"] == 20 == len(C.PHASES)
    assert manifest["arm_count"] == 18 == len(C.ARMS)
    assert set(E.ARM_DISABLED) == set(C.ARMS)
    assert manifest["target_leakage"] is False
    assert manifest["activation"] is False


def test_v5_target_is_revealed_after_commitment_and_not_in_observation() -> None:
    row = E.episode(
        split="construction",
        history_seed=101,
        arm="full_v5",
        phase_index=9,
        episode_index=3,
    )
    assert row["commitment"]["step"] < row["outcome"]["revealed_step"]
    assert "target" not in json.dumps(row["observation"], sort_keys=True)
    assert row["activation"] is False


def test_v5_generator_is_deterministic_and_mechanism_ablations_are_active() -> None:
    full = E.phase_result(
        split="construction",
        history_seed=102,
        arm="full_v5",
        phase_index=9,
    )
    ablated = E.phase_result(
        split="construction",
        history_seed=102,
        arm="no_active_perception",
        phase_index=9,
    )
    repeat = E.phase_result(
        split="construction",
        history_seed=102,
        arm="full_v5",
        phase_index=9,
    )
    assert full == repeat
    assert "active_perception" in full["mechanisms_active"]
    assert "active_perception" in ablated["mechanisms_missing"]
    assert full["event_digest"] != ablated["event_digest"]
    assert E.oracle_headroom(9)["has_headroom"]


def test_v5_cached_public_tasks_remain_isolated_between_callers() -> None:
    first_identity, first_observation, first_target = E._public_task(
        "construction", 103, 0, 0
    )
    first_observation["modality_cues"]["text"] = 999.0
    first_observation["mechanism_cues"]["model_fabric"] = 999.0
    first_observation["modalities"].append("mutated")

    second_identity, second_observation, second_target = E._public_task(
        "construction", 103, 0, 0
    )
    assert second_identity == first_identity
    assert second_target == first_target
    assert "mutated" not in second_observation["modalities"]
    assert second_observation["modality_cues"]["text"] != 999.0
    assert second_observation["mechanism_cues"]["model_fabric"] != 999.0


def test_v5_cached_sensor_events_remain_isolated_and_digest_exact() -> None:
    uncached = E._sensor_event_uncached(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    first, first_digest = E._sensor_event_with_digest(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    assert first == uncached
    assert first_digest == sensors.canonical_event_digest(uncached)

    first.observation["observable_cue"] = 999.0
    second, second_digest = E._sensor_event_with_digest(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    assert second is not first
    assert second == uncached
    assert second.observation["observable_cue"] == 0.25
    assert second_digest == sensors.canonical_event_digest(second)


def test_v5_cached_ingest_rechecks_mutable_target_boundary() -> None:
    event, _ = E._sensor_event_with_digest(
        "task:cached-ingest",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    event.observation["target"] = True
    with pytest.raises(sensors.SensoriumError, match="hidden target authority"):
        sensors.Sensorium()._ingest_cached(event)


def test_v5_terminal_retention_is_default_but_explicitly_optional() -> None:
    retained = E.phase_result(
        split="construction",
        history_seed=102,
        arm="full_v5",
        phase_index=len(C.PHASES) - 1,
    )
    summary_only = E.phase_result(
        split="construction",
        history_seed=102,
        arm="full_v5",
        phase_index=len(C.PHASES) - 1,
        include_v4_retention=False,
    )
    assert retained["v4_retention"]["preserved"] is True
    assert summary_only["v4_retention"] is None
    for field in ("phase", "accuracy", "mean_cost", "utility", "event_digest", "development_update"):
        assert summary_only[field] == retained[field]
