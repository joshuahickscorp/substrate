
from pathlib import Path

import pytest

from mop import config, devices
from mop.experiments import get_experiment
from mop.falsification.experiment_contracts import build_contract_audit


@pytest.fixture(scope="module")
def f14_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    cfg = config.compose(["experiment=f14_lifelong_form_expansion", "device=cpu"])
    dev = devices.resolve("cpu")
    run_dir = Path(tmp_path_factory.mktemp("f14"))
    return get_experiment("f14_lifelong_form_expansion").run(cfg, dev, run_dir)


def test_f14_contract_is_exact() -> None:
    audit = build_contract_audit(series="F")
    record = next(row for row in audit["records"] if row["experiment_id"] == "f14_lifelong_form_expansion")
    assert record["all_ok"], "\n".join(record["problems"])


def test_f14_old_memory_is_bit_exact_immutable(f14_result: dict) -> None:
    assert f14_result["memory_slots"] > 0
    assert f14_result["memory_tensor_bytes"] > 0
    assert f14_result["memory_index_backend"] == "brute"
    assert f14_result["memory_invariants_all_ok"]
    assert f14_result["memory_keys_unchanged"]
    assert f14_result["memory_values_unchanged"]
    assert f14_result["memory_referent_ids_unchanged"]
    assert f14_result["memory_length_unchanged"]
    assert f14_result["memory_seen_unchanged"]
    assert f14_result["old_alignment_unchanged"]
    assert f14_result["old_memory_recall_delta"] == 0.0
    assert set(f14_result["per_seed_old_memory_recall_delta"]) == {0.0}

    for row in f14_result["memory_invariants_by_seed"]:
        assert all(bool(value) for key, value in row.items() if key != "seed")
    for receipt in f14_result["memory_snapshot_receipts"]:
        assert receipt["before"] == receipt["after"]
        assert receipt["length_before"] == receipt["length_after"]
        assert receipt["seen_before"] == receipt["seen_after"]


def test_f14_replay_controls_are_compute_matched(f14_result: dict) -> None:
    accounting = f14_result["matched_accounting"]
    replay = accounting["replay_expansion"]
    no_replay = accounting["no_replay_expansion"]
    scratch = accounting["scratch"]
    assert f14_result["matched_replay_no_replay_compute"]
    assert replay == no_replay
    assert f14_result["matched_scratch_cumulative_compute"]
    assert replay["head_params"] == scratch["head_params"]
    assert replay["total_rows_seen"] == scratch["total_rows_seen"]
    assert replay["optimizer_steps"] == scratch["optimizer_steps"]
    assert replay["phase_two_rows_per_step"] == scratch["phase_two_rows_per_step"]
    assert replay["phase_two_steps"] == scratch["phase_two_steps"]


def test_f14_reports_paired_seed_evidence_and_dynamic_range(f14_result: dict) -> None:
    seeds = f14_result["seeds"]
    assert len(seeds) >= 5
    assert len(set(seeds)) == len(seeds)
    for key in (
        "seed_ci",
        "old_form_bwt_seed_ci",
        "replay_bwt_advantage_seed_ci",
        "old_memory_recall_seed_ci",
        "new_memory_seed_ci",
        "alignment_floor_seed_ci",
    ):
        assert f14_result[key]["n"] == len(seeds)
        assert not f14_result[key]["unstable"]

    paired_gain = sum(f14_result["per_seed_deltas"]) / len(seeds)
    memory_gain = sum(f14_result["per_seed_new_memory_deltas"]) / len(seeds)
    assert f14_result["new_form_gain_over_strongest_control"] == pytest.approx(paired_gain, abs=1.0e-4)
    assert f14_result["new_form_memory_gain_over_floor"] == pytest.approx(memory_gain, abs=1.0e-4)
    assert 0.0 < f14_result["new_form_transfer"] < 0.95
    assert 0.0 < f14_result["new_form_memory_recall"] < 0.95
    assert 0.0 < f14_result["old_memory_recall_before"] < 0.95


def test_f14_null_tracks_the_preregistered_strongest_control(f14_result: dict) -> None:
    assert f14_result["null_supported"] == bool(f14_result["null_reasons"])
    if f14_result["seed_ci"]["lo"] <= 0.01:
        assert "replay_failed_strongest_existing_head_control" in f14_result["null_reasons"]
