from __future__ import annotations

import json

from substrate import v5config as C
from substrate import v5experiment as E


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
