import json

from mop.falsification.null_cards import render_card
from mop.falsification.verdict_gate import build_verdict_gate, write_verdict_gate


def _card(verdict="PUBLISH-POSITIVE"):
    return {
        "exp_id": "gate_claim",
        "title": "receipt gate claim",
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
        "raw_run_id": "runs/gate_claim.json",
        "repro_level": "R1",
    }


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _write_card(path, verdict="PUBLISH-POSITIVE"):
    path.write_text(render_card(_card(verdict)))


def test_positive_verdict_requires_verifier_receipt(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    _write_card(card)
    _write_json(run, {"claim_id": "gate_claim", "verdict": "PUBLISH-POSITIVE"})
    gate = build_verdict_gate(null_card_path=card, run_receipt_path=run)
    assert gate["positive"] is True
    assert gate["all_ok"] is False
    assert any("verifier_receipt missing" in p for p in gate["problems"])


def test_positive_verdict_passes_with_independent_verifier(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    verifier = tmp_path / "verifier.json"
    _write_card(card)
    _write_json(run, {"claim_id": "gate_claim", "verdict": "PUBLISH-POSITIVE"})
    _write_json(verifier, {"claim_id": "gate_claim", "passed": True, "independent": True})
    gate = build_verdict_gate(null_card_path=card, run_receipt_path=run, verifier_receipt_path=verifier)
    assert gate["all_ok"] is True
    assert gate["verifier_receipt"]["sha256"]


def test_positive_verdict_refuses_same_receipt_as_verifier(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    _write_card(card)
    _write_json(run, {"passed": True, "independent": True})
    gate = build_verdict_gate(null_card_path=card, run_receipt_path=run, verifier_receipt_path=run)
    assert gate["all_ok"] is False
    assert any("path equals run_receipt" in p for p in gate["problems"])


def test_null_or_tie_verdict_does_not_need_verifier(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    _write_card(card, verdict="DOWNGRADE-TIE")
    _write_json(run, {"claim_id": "gate_claim", "verdict": "DOWNGRADE-TIE"})
    gate = build_verdict_gate(null_card_path=card, run_receipt_path=run)
    assert gate["positive"] is False
    assert gate["all_ok"] is True


def test_gate_receipt_write_round_trips(tmp_path):
    card = tmp_path / "card.md"
    run = tmp_path / "run.json"
    out = tmp_path / "gate.json"
    _write_card(card, verdict="DOWNGRADE-TIE")
    _write_json(run, {"claim_id": "gate_claim"})
    gate = build_verdict_gate(null_card_path=card, run_receipt_path=run)
    write_verdict_gate(gate, out)
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == "mop-verdict-gate/v1"
    assert loaded["all_ok"] is True
