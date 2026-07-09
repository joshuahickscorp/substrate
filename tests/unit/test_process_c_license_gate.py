import json

import scripts.studio.process_c_license_gate as process_c_cli

from mop.studio.process_c_gate import build_process_c_license_gate


def _pr9(**overrides):
    receipt = {
        "schema": "mop-pr9-verdict-ledger/v1",
        "all_ok": True,
        "status": "null_no_certificate",
        "decision": "NULL-NO-PLASTICITY-LOSS",
        "process_c_licensed": True,
        "cache": "data/cache/vjepa2_vitl_comp_video",
        "certificate": {"fired": False},
        "claim_status": "publish-null-or-wall",
        "problems": [],
    }
    receipt.update(overrides)
    return receipt


def _dr1(**overrides):
    receipt = {
        "schema": "mop-dr1-adversarial-verification/v1",
        "integrity_ok": True,
        "a6_survives": True,
        "passed": True,
        "all_ok": True,
        "independent": True,
        "adversarial": True,
        "cache_dir": "data/cache/vjepa2_vitl_comp_video",
        "summary": {"failed": 0},
        "problems": [],
    }
    receipt.update(overrides)
    return receipt


def test_process_c_gate_licenses_from_pr9_wall(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    receipt = build_process_c_license_gate(
        pr9_verdict=_pr9(),
        dr1_verification=_dr1(),
        null_card_path=card,
    )
    assert receipt["schema"] == "mop-process-c-license-gate/v1"
    assert receipt["all_ok"] is True
    assert receipt["launch_allowed"] is True
    assert receipt["licensing_sources"] == ["pr9"]


def test_process_c_gate_licenses_from_dr1_representational_wall(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    receipt = build_process_c_license_gate(
        pr9_verdict=_pr9(process_c_licensed=False, status="evidence_cbp_win"),
        dr1_verification=_dr1(a6_survives=False, passed=False, all_ok=False),
        null_card_path=card,
    )
    assert receipt["all_ok"] is True
    assert receipt["launch_allowed"] is True
    assert receipt["licensing_sources"] == ["dr1"]
    assert receipt["sources"]["dr1"]["status"] == "licensed_representational_wall"


def test_process_c_gate_can_complete_as_not_licensed(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    receipt = build_process_c_license_gate(
        pr9_verdict=_pr9(process_c_licensed=False, status="evidence_cbp_win"),
        dr1_verification=_dr1(),
        null_card_path=card,
    )
    assert receipt["all_ok"] is True
    assert receipt["status"] == "not_licensed"
    assert receipt["launch_allowed"] is False
    assert receipt["blockers"]


def test_process_c_gate_blocks_without_decisive_receipts(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    receipt = build_process_c_license_gate(
        pr9_verdict=None,
        dr1_verification=None,
        null_card_path=card,
    )
    assert receipt["all_ok"] is False
    assert receipt["status"] == "undecidable"
    assert "no decisive" in receipt["problems"][0]


def test_process_c_gate_cli_writes_receipt(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    pr9 = tmp_path / "pr9.json"
    dr1 = tmp_path / "dr1.json"
    out = tmp_path / "gate.json"
    pr9.write_text(json.dumps(_pr9()))
    dr1.write_text(json.dumps(_dr1()))
    rc = process_c_cli.main(
        [
            "--pr9-verdict",
            str(pr9),
            "--dr1-verification",
            str(dr1),
            "--null-card",
            str(card),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-process-c-license-gate/v1"
    assert data["launch_allowed"] is True
