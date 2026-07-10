import copy
import json

from omegaconf import OmegaConf

from mop.config import REPO_ROOT
from mop.falsification.form_verifier import (
    CANDIDATE_POSITIVES,
    ExecutionTrace,
    _assignment_minimum_dp,
    _semantic_checks,
    validate_verifier_receipt,
)
from mop.falsification.null_cards import render_card
from mop.falsification.verdict_gate import build_verdict_gate
from mop.studio.artifact_bundle import preset_paths


def _candidate(eid):
    receipt = json.loads((REPO_ROOT / f"proof/FORM_SUBSTRATE/RECEIPTS/{eid}.json").read_text())
    config = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / f"configs/experiment/{eid}.yaml"), resolve=True
    )
    return copy.deepcopy(receipt["metrics"]), dict(config)


def _trace(eid, metrics):
    effect_keys = (
        "per_seed_deltas",
        "per_seed_transfer_deltas",
        "per_seed_frontier_deltas",
    )
    effects = next(
        (list(metrics[key]) for key in effect_keys if isinstance(metrics.get(key), list)),
        [0.2] * 5,
    )
    trace = ExecutionTrace(ci_inputs=[effects], sign_inputs=[effects])
    if eid == "f12_private_form_language_stability":
        n = len(metrics["seeds"])
        expected = n * (n - 1) + 2 * (n - 1)
        trace.hungarian_calls = [{"n": 16, "exact": True}] * expected
    return trace


def _failed_names(eid, metrics, config):
    checks = []
    _semantic_checks(eid, metrics, config, _trace(eid, metrics), checks)
    return {check["name"] for check in checks if check["passed"] is not True}


def _card():
    return {
        "exp_id": "f1_form_alignment_gate",
        "title": "gate test",
        "hypothesis": "alignment beats controls",
        "null_hypothesis": "alignment ties controls",
        "baseline": "controls",
        "ablation": "remove alignment",
        "metric": "aligned_transfer",
        "probe_dependency": {
            "factor": "identity",
            "encoder": "fixture",
            "atlas_row": "not-applicable",
            "decodable": "yes",
            "acc_above_chance": 0.5,
        },
        "encoder_scale": "not-applicable",
        "seeds": {"n": 5, "sem": 0.01, "sign_stability": "stable"},
        "provenance_tag": "structured-synthetic",
        "result": "candidate",
        "taxonomy_category": 1,
        "verdict": "PUBLISH-POSITIVE",
        "badges": ["preregistered"],
        "raw_run_id": "receipt.json",
        "repro_level": "R1",
    }


def test_every_candidate_has_an_experiment_specific_semantic_verifier():
    assert len(CANDIDATE_POSITIVES) == 12
    for eid in CANDIDATE_POSITIVES:
        metrics, config = _candidate(eid)
        assert _failed_names(eid, metrics, config) == set(), eid


def test_strongest_control_is_recomputed_instead_of_trusting_null_flag():
    metrics, config = _candidate("f1_form_alignment_gate")
    metrics["null_supported"] = False
    metrics["raw_transfer"] = metrics["aligned_transfer"] + 0.01
    assert "f1_paired_alignment_beats_every_unpaired_control" in _failed_names(
        "f1_form_alignment_gate", metrics, config
    )


def test_token_geometry_and_form_b_transport_checks_fail_closed():
    metrics, config = _candidate("f4_raw_payload_vs_form_tokens")
    metrics["token_shape"] = [1, config["token_count"] * config["token_dim"]]
    assert "f4_native_token_geometry_is_preserved_and_enforced" in _failed_names(
        "f4_raw_payload_vs_form_tokens", metrics, config
    )

    metrics, config = _candidate("f18_counterfactual_form_intervention")
    metrics["predicted_object"] = "scalar_factor_label"
    assert "f18_predicts_actual_form_b_geometry_not_a_scalar_surrogate" in _failed_names(
        "f18_counterfactual_form_intervention", metrics, config
    )


def test_full_cost_grid_check_detects_a_silently_dropped_point():
    metrics, config = _candidate("f13_form_energy_budget")
    metrics["frontier_points"].pop()
    assert "f13_preregistered_full_grid_is_complete_without_duplicates" in _failed_names(
        "f13_form_energy_budget", metrics, config
    )


def test_independent_assignment_dp_finds_greedy_trap_optimum():
    import numpy as np

    cost = np.array([[1.0, 2.0, 100.0], [1.1, 100.0, 100.0], [100.0, 1.0, 1.0]])
    assert _assignment_minimum_dp(cost) == 4.1


def test_failed_form_verifier_cannot_pass_gate_via_nested_true_flag(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    verifier = tmp_path / "verifier.json"
    card.write_text(render_card(_card()))
    run.write_text(json.dumps({"claim_id": "f1_form_alignment_gate"}) + "\n")
    verifier.write_text(
        json.dumps(
            {
                "schema": "mop-form-independent-verifier/v1",
                "experiment_id": "f1_form_alignment_gate",
                "passed": False,
                "all_ok": False,
                "independent": True,
                "adversarial": True,
                "checks": [{"name": "one", "passed": True}],
                "problems": ["strongest control won"],
                "fresh_seeds_or_heldout_cases": [101, 211, 307, 401, 503],
                "canonical_seeds": [0, 1, 2, 3, 4],
                "frozen_live_contract": {
                    "contract_fingerprint": "a",
                    "config_sha256": "b",
                    "class_source_sha256": "c",
                    "verifier_sha256": "d",
                },
                "source_receipts": [{"path": str(run), "sha256": "wrong"}],
            }
        )
        + "\n"
    )
    gate = build_verdict_gate(
        null_card_path=card,
        run_receipt_path=run,
        verifier_receipt_path=verifier,
    )
    assert gate["all_ok"] is False
    assert gate["verifier_receipt"]["passed"] is False


def test_strict_receipt_validator_rejects_partial_envelopes():
    receipt = {
        "schema": "mop-form-independent-verifier/v1",
        "experiment_id": "f1_form_alignment_gate",
        "passed": True,
        "all_ok": True,
        "independent": True,
        "adversarial": True,
        "checks": [{"name": "bad", "passed": False}],
        "problems": [],
        "fresh_seeds_or_heldout_cases": [101, 211, 307, 401, 503],
        "canonical_seeds": [0, 1, 2, 3, 4],
        "frozen_live_contract": {
            "contract_fingerprint": "a",
            "config_sha256": "b",
            "class_source_sha256": "c",
            "verifier_sha256": "d",
        },
    }
    assert "one or more verifier checks did not pass" in validate_verifier_receipt(
        receipt, "f1_form_alignment_gate"
    )


def test_form_artifact_bundle_includes_every_candidate_verifier():
    paths = set(preset_paths("form-substrate"))
    assert {f"proof/FORM_SUBSTRATE/VERIFIERS/{eid}.json" for eid in CANDIDATE_POSITIVES} <= paths
