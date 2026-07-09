import json

import scripts.studio.pr9_verdict_ledger as pr9_cli

from mop.studio.pr9_verdict import build_pr9_verdict_ledger


def _result(**overrides):
    result = {
        "cache": "data/cache/vjepa2_vitl_comp_video",
        "certificate": {"fired": True, "adapt_trends_down": True, "dead_trends_up": True},
        "any_zero_reinit": False,
        "lr_integral_matched_all": True,
        "winning_rates": [],
        "reinit_count_total_all_rates": 10,
        "null_supported": True,
        "verdict": "NULL SUPPORTED: CBP did not restore plasticity beyond seed spread",
    }
    result.update(overrides)
    return result


def _state(**overrides):
    state = {
        "schema": "mop-pr9-run-state/v1",
        "status": "complete",
        "expected_leg_count": 6,
        "completed_leg_count": 6,
        "resume_behavior": "rerun the same command; completed legs are skipped",
    }
    state.update(overrides)
    return state


def test_pr9_verdict_ledger_blocks_missing_result(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_pr9_verdict_ledger(result=None, state=None, null_card_path=card)
    assert ledger["schema"] == "mop-pr9-verdict-ledger/v1"
    assert ledger["all_ok"] is False
    assert ledger["status"] == "missing"


def test_pr9_verdict_ledger_refuses_local_smoke_cache(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_pr9_verdict_ledger(
        result=_result(cache="data/cache/vjepa2_vitl_fpc64_256_real"),
        state=_state(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is False
    assert ledger["status"] == "non_scoring"
    assert "not the DR1 real cache" in ledger["problems"][0]


def test_pr9_verdict_ledger_marks_cbp_win_as_candidate_positive(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_pr9_verdict_ledger(
        result=_result(null_supported=False, winning_rates=[0.001], verdict="NULL REJECTED"),
        state=_state(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is True
    assert ledger["status"] == "evidence_cbp_win"
    assert ledger["decision"] == "CANDIDATE-POSITIVE"
    assert ledger["claim_status"] == "candidate-positive-needs-verdict-gate"


def test_pr9_verdict_ledger_null_can_license_process_c(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_pr9_verdict_ledger(
        result=_result(certificate={"fired": False}, null_supported=True),
        state=_state(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is True
    assert ledger["status"] == "null_no_certificate"
    assert ledger["process_c_licensed"] is True


def test_pr9_verdict_cli_writes_receipt(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    result = tmp_path / "result.json"
    state = tmp_path / "state.json"
    out = tmp_path / "ledger.json"
    result.write_text(json.dumps(_result()))
    state.write_text(json.dumps(_state()))
    rc = pr9_cli.main(
        ["--result", str(result), "--state", str(state), "--null-card", str(card), "--out", str(out)]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-pr9-verdict-ledger/v1"
