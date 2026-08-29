"""Active replacements for the four predecessor collapse regressions."""

from __future__ import annotations

import json

import pytest

from substrate import audit, evidence, historical, runtime


def test_portability_contract_uses_no_predecessor_checkout_path():
    active = [evidence.ROOT / "src/substrate", evidence.ROOT / "tests/substrate", evidence.ROOT / "docs"]
    forbidden = "/Users/scammermike/Downloads/" + "mop"
    hits = []
    for root in active:
        for path in root.rglob("*"):
            relative = path.relative_to(evidence.ROOT)
            if relative.parts[:2] == ("docs", "archive"):
                continue
            if path.suffix in {".py", ".md"} and forbidden in path.read_text(errors="ignore"):
                hits.append(path.relative_to(evidence.ROOT).as_posix())
    assert hits == []


def test_collapse_invariants_leave_one_product_and_one_writer():
    assert not (evidence.ROOT / "src" / "substrate" / "compat").exists()
    assert (evidence.ROOT / "src/substrate/cli.py").is_file()
    assert historical.verify_all()["all_pass"] is True
    assert audit.run()["results"]["exclusive_producers"]["ok"] is True
    # v5 deliberately owns a separate immutable writer with a stricter path
    # contract; the shared text writer must still have one producer.
    writers = [
        path
        for path in (evidence.ROOT / "src/substrate").glob("*.py")
        if path.name != "v5io.py" and "def atomic_write_bytes(" in path.read_text()
    ]
    assert writers == [evidence.ROOT / "src/substrate/evidence.py"]


def test_custom_substrate_artifact_is_content_bound_and_tamper_evident():
    entity = runtime.Substrate()
    entity.step({"label": "a", "label_confidence": 0.8}, outcome="a")
    checkpoint = entity.checkpoint()
    encoded = json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), default=str)
    assert checkpoint["identity"] == runtime.io.sha_obj(entity._state_for_hash())
    assert "identity" in encoded
    checkpoint["step"] = 999
    with pytest.raises(runtime.Refused):
        runtime.Substrate().restore(checkpoint)


def test_unindexed_proof_contract_has_no_active_orphan_artifacts():
    result = audit.run()["results"]["exclusive_producers"]
    assert result["duplicated"] == {}
    assert result["on_disk_without_a_producer"] == []
