"""Semantic gates for the F-series rows whose original R0 implementations overclaimed mechanics.

These tests never require a positive scientific result. They require the declared comparison to be
the comparison actually executed: disjoint evidence splits, matched exposure, preserved token
geometry, full cost axes, Form-B prediction, and a real four-scale indexed memory.
"""

import json
from pathlib import Path

import pytest
import torch

from mop import config, devices
from mop.experiments import get_experiment
from mop.experiments.f_form_substrate import _F4TokenProbe
from mop.falsification.experiment_contracts import build_contract_audit

REPAIRED_IDS = (
    "f1_form_alignment_gate",
    "f2_heldout_form_transfer",
    "f3_form_bottleneck_capacity",
    "f4_raw_payload_vs_form_tokens",
    "f13_form_energy_budget",
    "f18_counterfactual_form_intervention",
    "f19_cross_scale_referent_binding",
    "f20_substrate_crisis_test",
)


@pytest.fixture(scope="module")
def repaired_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("f-repaired")
    out = {}
    for experiment_id in REPAIRED_IDS:
        cfg = config.compose([f"experiment={experiment_id}", "device=cpu"])
        out[experiment_id] = get_experiment(experiment_id).run(
            cfg,
            devices.resolve("cpu"),
            root / experiment_id,
        )
    return out


def test_all_f_contract_surfaces_are_exactly_equal():
    audit = build_contract_audit(series="F", implemented_only=False)
    assert audit["all_ok"], audit["problems"]
    assert audit["summary"] == {
        "total": 50,
        "aligned": 50,
        "misaligned": 0,
        "implemented": 18,
        "preregistration_only": 32,
    }


def test_f1_uses_disjoint_rows_and_non_ceiling_fixture(repaired_results):
    out = repaired_results["f1_form_alignment_gate"]
    assert out["disjoint_splits"]
    assert sum(out["split_rows"].values()) == 420
    assert 0.125 < out["aligned_transfer"] < 0.95
    assert out["seed_ci"]["n"] == len(out["seeds"]) == 5
    assert len(out["per_seed_deltas"]) == 5
    assert out["target_supervised_oracle"] < 0.95


def test_f2_matches_rows_updates_initialization_and_head(repaired_results):
    out = repaired_results["f2_heldout_form_transfer"]
    assert out["disjoint_splits"] and sum(out["split_rows"].values()) == 400
    assert out["matched_rows_updates_head"]
    accounting = list(out["matched_accounting"].values())
    assert accounting and all(record == accounting[0] for record in accounting)
    assert out["heldout_form_acc"] < 0.95
    assert out["seed_ci"]["n"] == 5


def test_f3_uses_nested_bottlenecks_and_one_head_topology(repaired_results):
    out = repaired_results["f3_form_bottleneck_capacity"]
    assert out["nested_projection"] and out["small_zero_padded"]
    assert out["identical_head_topology"]
    accounting = list(out["matched_accounting"].values())
    assert accounting and all(record == accounting[0] for record in accounting)
    assert out["wide_form_acc"] < 0.95
    assert out["seed_ci"]["n"] == 5


def test_f4_preserves_token_axis_and_rejects_flat_inputs(repaired_results):
    out = repaired_results["f4_raw_payload_vs_form_tokens"]
    assert out["token_shape"] == [4, 8]
    assert out["token_axis_preserved"] and out["audit_all_ok"]
    accounting = list(out["matched_accounting"].values())
    assert accounting and all(record == accounting[0] for record in accounting)
    assert out["canonical_cross_form_acc"] < 0.95
    with pytest.raises(ValueError, match=r"\[N,T,D\]"):
        _F4TokenProbe(4, 8, 8, 8)(torch.zeros(2, 32))


def test_f13_prices_the_full_grid_and_labels_energy_as_estimated(repaired_results):
    out = repaired_results["f13_form_energy_budget"]
    assert out["energy_measured"] is False
    assert out["estimated_energy_unit"] == "joules_per_correct_prediction"
    assert out["grid"] == {
        "widths": [4, 8, 16],
        "token_counts": [1, 4],
        "shell_sizes": [0, 16],
        "replay_bytes": [4096, 16384],
        "seeds": [0, 1, 2, 3, 4],
    }
    assert len(out["frontier_points"]) == 72
    assert all(
        point["params"] > 0
        and point["retained_bytes"] > 0
        and point["estimated_flops"] > 0
        and point["estimated_energy_joules"] > 0
        for point in out["frontier_points"]
    )
    receipt = json.loads(Path(out["frontier_receipt"]).read_text())
    assert receipt["schema"] == "mop-f13-density-frontier/v2"
    assert receipt["energy_measured"] is False
    assert out["density"]["primary"] in get_experiment("f13_form_energy_budget").metric


def test_f18_predicts_form_b_with_exactly_matched_training(repaired_results):
    out = repaired_results["f18_counterfactual_form_intervention"]
    assert out["predicted_object"] == "form_b_state"
    assert out["matched_train_rows"] and out["matched_compute"]["matched"]
    assert out["matched_compute"]["ratio"] == 1.0
    assert out["cross_form_readout_acc"] > out["chance"]
    assert out["counterfactual_match_acc"] < 0.95
    assert out["seed_ci"]["n"] == 5


def test_f19_is_one_four_scale_byte_matched_index(repaired_results):
    out = repaired_results["f19_cross_scale_referent_binding"]
    assert out["hierarchy"]["parent_links_explicit"]
    assert out["hierarchy"]["shared_index"]
    assert out["matched_memory_bytes"]
    assert len(set(out["store_bytes_by_arm"].values())) == 1
    assert set(out["relations"]) == {
        "task_to_episode",
        "episode_to_task",
        "episode_to_scene",
        "scene_to_episode",
        "scene_to_object",
        "object_to_scene",
    }
    assert out["cross_scale_recall_at_k"] < 0.95
    assert out["seed_ci"]["n"] == 5


def test_f20_does_not_relabel_hypothetical_savings_as_measured(repaired_results):
    out = repaired_results["f20_substrate_crisis_test"]
    assert out["avoided_compute_measured"] is False
    assert out["avoided_wasted_compute_per_monitor_flop"] == 0.0
    assert out["seed_ci"]["n"] == 5
    assert out["sign_flip_report"]["n"] == 5
