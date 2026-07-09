import json

import scripts.studio.atlas_verdict_ledger as atlas_cli

from mop.studio.atlas_verdict import build_atlas_verdict_ledger


def _dense_gate(**overrides):
    gate = {
        "schema": "mop-dense-atlas-cache-gate/v1",
        "all_ok": True,
        "real_cache": {"path": "data/cache/vjepa21_vitl_dense8192_real"},
        "randominit_cache": {"path": "data/cache/vjepa21_vitl_dense8192_randominit"},
        "pair": {"keys_match": True},
        "problems": [],
    }
    gate.update(overrides)
    return gate


def _atlas(**overrides):
    result = {
        "full_registered_grid": True,
        "full_registered_pairs": True,
        "registered_columns_missing": [],
        "registered_arms_missing": [],
        "null_supported": True,
        "verdict": "NULL SUPPORTED: full atlas supports controls",
        "atlas_scope": {"scope": "random-control-artifact"},
    }
    result.update(overrides)
    return result


def test_atlas_verdict_blocks_missing_dense_gate(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_atlas_verdict_ledger(atlas=_atlas(), dense_gate=None, null_card_path=card)
    assert ledger["schema"] == "mop-atlas-verdict-ledger/v1"
    assert ledger["all_ok"] is False
    assert ledger["status"] == "dense_gate_missing"


def test_atlas_verdict_blocks_partial_grid(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_atlas_verdict_ledger(
        atlas=_atlas(full_registered_grid=False, registered_columns_missing=["vjepa21_dense_8192"]),
        dense_gate=_dense_gate(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is False
    assert ledger["status"] == "partial_non_scoring"
    assert "missing registered columns" in ledger["problems"][0]


def test_atlas_verdict_candidate_positive_needs_verdict_gate(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_atlas_verdict_ledger(
        atlas=_atlas(null_supported=False, verdict="NULL REJECTED"),
        dense_gate=_dense_gate(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is True
    assert ledger["status"] == "candidate_positive"
    assert ledger["decision"] == "CANDIDATE-POSITIVE"
    assert ledger["claim_status"] == "candidate-positive-needs-verdict-gate"


def test_atlas_verdict_null_supported_is_scoring_wall(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    ledger = build_atlas_verdict_ledger(
        atlas=_atlas(null_supported=True),
        dense_gate=_dense_gate(),
        null_card_path=card,
    )
    assert ledger["all_ok"] is True
    assert ledger["status"] == "null_supported"
    assert ledger["decision"] == "NULL-SUPPORTED"


def test_atlas_verdict_cli_writes_receipt(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("card")
    dense_gate = tmp_path / "dense_gate.json"
    atlas = tmp_path / "atlas.json"
    out = tmp_path / "ledger.json"
    dense_gate.write_text(json.dumps(_dense_gate()))
    atlas.write_text(json.dumps(_atlas(null_supported=False, verdict="NULL REJECTED")))
    rc = atlas_cli.main(
        ["--atlas", str(atlas), "--dense-gate", str(dense_gate), "--null-card", str(card), "--out", str(out)]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-atlas-verdict-ledger/v1"
