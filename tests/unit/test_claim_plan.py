import json

import pytest
import scripts.studio.__main__ as studio_cli

from mop.falsification.null_cards import render_card
from mop.studio.claim_plan import build_claim_daemon_plan, write_claim_daemon_plan
from mop.studio.long_run import load_plan, validate_plan_contract


def _card(verdict="PUBLISH-POSITIVE"):
    return {
        "exp_id": "studio_claim",
        "title": "studio claim",
        "hypothesis": "mechanism beats the control",
        "null_hypothesis": "mechanism ties the control",
        "baseline": "matched control",
        "ablation": "remove mechanism",
        "metric": "delta",
        "probe_dependency": {
            "factor": "identity",
            "encoder": "vjepa2_vitl_fpc64_256",
            "atlas_row": "proof/atlas/vjepa2_vitl_fpc64_256/identity.json",
            "decodable": "yes",
            "acc_above_chance": 0.42,
        },
        "encoder_scale": "L",
        "seeds": {"n": 3, "sem": 0.01, "sign_stability": "stable at S>=3"},
        "provenance_tag": "structured-synthetic",
        "result": "mechanism delta clears the seed CI",
        "taxonomy_category": "null rejected" if verdict == "PUBLISH-POSITIVE" else 4,
        "verdict": verdict,
        "badges": ["preregistered"],
        "raw_run_id": "runs/studio_claim.json",
        "repro_level": "R1",
    }


def _write_claim_files(tmp_path, verdict="PUBLISH-POSITIVE"):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    verifier = tmp_path / "verifier.json"
    card.write_text(render_card(_card(verdict)))
    run.write_text(json.dumps({"claim_id": "studio_claim", "verdict": verdict}) + "\n")
    verifier.write_text(json.dumps({"passed": True, "independent": True}) + "\n")
    return card, run, verifier


def test_positive_claim_plan_inserts_verdict_and_artifact_gates_before_ledger(tmp_path):
    card, run, verifier = _write_claim_files(tmp_path)
    plan = build_claim_daemon_plan(
        null_card=str(card),
        run_receipt=str(run),
        verifier_receipt=str(verifier),
        verdict_gate_out=str(tmp_path / "gate.json"),
        artifact_index_out=str(tmp_path / "index.json"),
        copy_dir=str(tmp_path / "bundle"),
        ledger_cmd=["python", "-m", "scripts.studio", "wave0-report", "--apply"],
    )
    assert [job["kind"] for job in plan["jobs"]] == [
        "verdict-gate",
        "artifact-bundle",
        "positive-ledger",
    ]
    assert validate_plan_contract(load_plan_from_object(tmp_path, plan)) == []
    artifact_cmd = plan["jobs"][1]["cmd"]
    assert "--require-durable" in artifact_cmd
    assert str(verifier) in artifact_cmd


def test_null_claim_plan_uses_non_positive_ledger_kind(tmp_path):
    card, run, _verifier = _write_claim_files(tmp_path, verdict="DOWNGRADE-TIE")
    plan = build_claim_daemon_plan(
        null_card=str(card),
        run_receipt=str(run),
        verifier_receipt=None,
        verdict="DOWNGRADE-TIE",
        verdict_gate_out=str(tmp_path / "gate.json"),
        artifact_index_out=str(tmp_path / "index.json"),
        ledger_cmd=["python", "-m", "scripts.studio", "wave0-report", "--apply"],
    )
    assert plan["jobs"][-1]["kind"] == "ledger"


def test_claim_plan_requires_ledger_command(tmp_path):
    card, run, verifier = _write_claim_files(tmp_path)
    with pytest.raises(ValueError, match="ledger_cmd"):
        build_claim_daemon_plan(
            null_card=str(card),
            run_receipt=str(run),
            verifier_receipt=str(verifier),
            verdict_gate_out=str(tmp_path / "gate.json"),
            artifact_index_out=str(tmp_path / "index.json"),
            ledger_cmd=[],
        )


def test_claim_plan_cli_writes_daemon_valid_plan(tmp_path):
    card, run, verifier = _write_claim_files(tmp_path)
    out = tmp_path / "plan.json"
    rc = studio_cli.main(
        [
            "claim-plan",
            "--null-card",
            str(card),
            "--run-receipt",
            str(run),
            "--verifier-receipt",
            str(verifier),
            "--verdict-gate-out",
            str(tmp_path / "gate.json"),
            "--artifact-index-out",
            str(tmp_path / "index.json"),
            "--copy-dir",
            str(tmp_path / "bundle"),
            "--ledger-cmd-json",
            '["python","-m","scripts.studio","wave0-report","--apply"]',
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    jobs = load_plan(out)
    assert [job.kind for job in jobs] == ["verdict-gate", "artifact-bundle", "positive-ledger"]


def load_plan_from_object(tmp_path, plan):
    path = tmp_path / "plan_object.json"
    write_claim_daemon_plan(plan, path)
    return load_plan(path)
