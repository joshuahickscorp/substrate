import copy
import json

import yaml

from mop.config import REPO_ROOT
from mop.devel.registries import load_experiments
from mop.experiments import REGISTRY
from mop.falsification.experiment_contracts import compare_contract_sources
from mop.falsification.form_evidence import (
    CAMPAIGN_SCHEMA,
    FORM_CAMPAIGN_IDS,
    build_density_input,
    build_null_card,
    build_oa_input,
    build_run_receipt,
    load_form_campaign,
    validate_form_campaign,
)
from mop.falsification.null_cards import load_card, validate_card
from mop.studio.artifact_bundle import preset_paths
from mop.studio.form_boundary import (
    BOUNDARY_EVIDENCE_SCHEMA,
    _studio_only_boundary,
    validate_scale_boundary_evidence,
)


class _Contract:
    id = "f_test"
    metric = ("score", "cost")
    null_hypothesis = "the mechanism ties the control"
    tier = "cpu-now"


def test_exact_contract_comparison_detects_one_character_drift():
    row = {
        "id": "f_test",
        "metrics": ["score", "cost"],
        "null_hypothesis": "the mechanism ties the control",
        "exp_tier": "cpu-now",
        "status": "implemented",
    }
    config = {
        "id": "f_test",
        "metric": ["score", "cost"],
        "null_hypothesis": "the mechanism ties the control",
        "tier": "cpu-now",
    }
    assert compare_contract_sources(row, _Contract, config)["all_ok"] is True
    config["null_hypothesis"] += "."
    audit = compare_contract_sources(row, _Contract, config)
    assert audit["all_ok"] is False
    assert audit["comparisons"]["null_hypothesis"]["equal"] is False


def test_live_form_campaign_is_complete_and_acyclic():
    campaign = load_form_campaign()
    assert campaign["schema"] == CAMPAIGN_SCHEMA
    assert validate_form_campaign(campaign) == []
    assert len(campaign["legs"]) == 20


def test_form_campaign_membership_is_frozen_not_numeric_prefix_inferred():
    campaign = load_form_campaign()
    rows = load_experiments()
    assert tuple(leg["id"] for leg in campaign["legs"]) == FORM_CAMPAIGN_IDS

    alias_rows = [*copy.deepcopy(rows), {"id": "f1_variant", "series": "F"}]
    problems = validate_form_campaign(campaign, registry_rows=alias_rows)
    assert any("unfrozen F1-F20 aliases" in problem for problem in problems)

    later_rows = [*copy.deepcopy(rows), {"id": "f67_future_scaffold", "series": "F"}]
    assert validate_form_campaign(campaign, registry_rows=later_rows) == []

    missing_rows = [row for row in copy.deepcopy(rows) if row["id"] != FORM_CAMPAIGN_IDS[0]]
    problems = validate_form_campaign(campaign, registry_rows=missing_rows)
    assert any("exactly once; found 0" in problem for problem in problems)


def test_registry_backed_form_null_card_is_strict():
    row = next(row for row in load_experiments() if row["id"] == "f5_cross_form_memory_binding")
    card = build_null_card(row, intended_seeds=5)
    assert card["null_hypothesis"] == row["null_hypothesis"]
    assert card["verdict"] == "DOWNGRADE-TIE"
    assert "reconstructed-after-audit" in card["badges"]
    assert validate_card(card, strict=True) == []


def test_run_receipt_rejects_stale_run_and_accepts_exact_snapshot(tmp_path):
    eid = "f5_cross_form_memory_binding"
    row = next(row for row in load_experiments() if row["id"] == eid)
    cls = REGISTRY[eid]
    config = {
        "id": eid,
        "name": "cross_form_memory_binding",
        "module": "f_form_substrate",
        "metric": list(cls.metric),
        "null_hypothesis": cls.null_hypothesis,
        "tier": cls.tier,
        "seeds": [0, 1, 2, 3, 4],
    }
    (tmp_path / "registry").mkdir()
    (tmp_path / "configs/experiment").mkdir(parents=True)
    (tmp_path / "campaign").mkdir()
    (tmp_path / "registry/experiments.yaml").write_text(
        yaml.safe_dump({"experiments": [row]}, sort_keys=False)
    )
    (tmp_path / f"configs/experiment/{eid}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "series": "F",
        "minimum_seeds": 5,
        "canonical_output_root": "proof/FORM_SUBSTRATE",
        "legs": [
            {
                "id": eid,
                "phase": "interface",
                "depends_on": [],
                "local_requirement": "canonical-run",
                "scale_boundary": "local",
            }
        ],
    }
    (tmp_path / "campaign/form_substrate_campaign.yaml").write_text(yaml.safe_dump(campaign, sort_keys=False))
    run_dir = tmp_path / "runs" / eid / "000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": config}, sort_keys=False))
    metrics = {name: 0.5 for name in cls.metric}
    metrics.update(
        {
            "seeds": [0, 1, 2, 3, 4],
            "seed_ci": {"n": 5, "mean": 0.2, "lo": 0.1, "hi": 0.3},
            "sign_flip_report": {"n": 5, "n_pos": 5, "n_neg": 0, "n_zero": 0},
            "null_supported": False,
            "density": {
                "schema": "mop-density-block/v1",
                "primary": cls.metric[0],
                "capability": {cls.metric[0]: 0.5},
                "cost": {"seconds": 1.0},
                "density": {f"{cls.metric[0]}_per_seconds": 0.5},
            },
        }
    )
    manifest = {
        "name": eid,
        "seed": 0,
        "device": "cpu",
        "git": "abc1234",
        "platform": "test",
        "started": 1.0,
        "finished": 2.0,
        "status": "ok",
        "result_tag": "provisional",
        "metrics": metrics,
        "extra": {"contract": cls().contract()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    receipt = build_run_receipt(eid, source_run_dir=run_dir, repo_root=tmp_path)
    assert receipt["all_ok"] is True
    assert receipt["receipt_fingerprint"]

    stale = {"experiment": {**config, "null_hypothesis": config["null_hypothesis"] + "."}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(stale, sort_keys=False))
    receipt = build_run_receipt(eid, source_run_dir=run_dir, repo_root=tmp_path)
    assert receipt["all_ok"] is False
    assert any("run config" in problem or "snapshot" in problem for problem in receipt["problems"])


def test_oa_and_density_inputs_preserve_component_provenance():
    receipt = {
        "experiment_id": "f20_substrate_crisis_test",
        "operational_awareness": {
            "components": {"oa6_crisis_detection": {"auroc": 0.8, "raw_error_auroc": 0.6, "chance": 0.5}}
        },
        "contract": {"canonical": {"metric": ["crisis_auroc"]}},
        "density": {
            "schema": "mop-density-block/v1",
            "primary": "crisis_auroc",
            "capability": {"crisis_auroc": 0.8},
            "cost": {"flops": 100.0},
            "density": {"crisis_auroc_per_flops": 0.008},
        },
    }
    oa = build_oa_input([receipt])
    density = build_density_input([receipt])
    assert oa["composite_score"] is None
    assert oa["components"]["oa6_crisis_detection"][0]["experiment_id"] == receipt["experiment_id"]
    assert density["all_ok"] is True


def test_scale_boundary_requires_measurement_and_command():
    receipt = {
        "schema": BOUNDARY_EVIDENCE_SCHEMA,
        "experiment_id": "f7_developmental_form_growth",
        "local_attempted": True,
        "limit_type": "memory",
        "measurement": {
            "local_available": 18.0,
            "required_or_observed": 44.0,
            "unit": "GB",
            "method": "measured peak plus full-grid projection",
        },
        "studio_profile": "studio-m1ultra",
        "source_receipts": ["proof/FORM_SUBSTRATE/PREFLIGHT/f7_developmental_form_growth.json"],
        "full_scale_command": ["python", "scripts/form_substrate_campaign.py", "run-studio-f7"],
    }
    assert validate_scale_boundary_evidence(receipt, receipt["experiment_id"]) == []
    del receipt["measurement"]["method"]
    assert any(
        "method" in problem for problem in validate_scale_boundary_evidence(receipt, receipt["experiment_id"])
    )


def test_studio_only_boundary_cannot_pass_vacuously_or_with_non_hardware_blockers():
    base = {
        "local_exhausted": True,
        "scientific_ledger_ready": True,
        "verified_studio_boundaries": [],
        "unproved_studio_boundaries": [],
        "non_hardware_blockers": [],
        "beyond_studio": [],
    }
    assert _studio_only_boundary(**base) is False
    base["verified_studio_boundaries"] = ["f7_developmental_form_growth"]
    assert _studio_only_boundary(**base) is True
    base["non_hardware_blockers"] = [{"experiment_id": "f8_plastic_substrate_rewrite"}]
    assert _studio_only_boundary(**base) is False


def test_form_artifact_preset_names_durable_campaign_surfaces():
    paths = preset_paths("form-substrate")
    assert "campaign/form_substrate_campaign.yaml" in paths
    assert "proof/FORM_SUBSTRATE/SCORECARD.json" in paths
    assert "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json" in paths
    assert "proof/FORM_SUBSTRATE/NULL_CARDS/f20_substrate_crisis_test.md" in paths
    assert "proof/FORM_SUBSTRATE/RECEIPTS/f5_cross_form_memory_binding.json" in paths
    assert "proof/FORM_SUBSTRATE/PREFLIGHT/f16_perfect_slate_null.json" in paths


def test_durable_form_null_cards_cover_every_campaign_leg_and_parse_strictly():
    campaign = load_form_campaign()
    expected = {leg["id"] for leg in campaign["legs"]}
    card_dir = REPO_ROOT / "proof/FORM_SUBSTRATE/NULL_CARDS"
    actual = {path.stem for path in card_dir.glob("*.md")}
    assert actual == expected
    for eid in sorted(expected):
        assert validate_card(load_card(card_dir / f"{eid}.md"), strict=True) == []


def test_durable_score_and_boundary_receipts_are_machine_readable():
    score = json.loads((REPO_ROOT / "proof/FORM_SUBSTRATE/SCORECARD.json").read_text())
    boundary = json.loads((REPO_ROOT / "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json").read_text())
    assert score["schema"] == "mop-form-campaign-scorecard/v1"
    assert boundary["schema"] == "mop-form-pre-studio-boundary/v1"
    assert {row["experiment_id"] for row in score["legs"]} == {
        row["experiment_id"] for row in boundary["classifications"]
    }
