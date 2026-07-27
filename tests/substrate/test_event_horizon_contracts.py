"""Active replacements for the four predecessor collapse regressions."""

from __future__ import annotations

import json

import pytest

from substrate import audit, evidence, runtime


def test_portability_contract_uses_no_predecessor_checkout_path():
    active = [evidence.ROOT / "src/substrate", evidence.ROOT / "tests/substrate", evidence.ROOT / "docs"]
    forbidden = "/Users/scammermike/Downloads/" + "mop"
    hits = []
    for root in active:
        for path in root.rglob("*"):
            if path.suffix in {".py", ".md"} and forbidden in path.read_text(errors="ignore"):
                hits.append(path.relative_to(evidence.ROOT).as_posix())
    assert hits == []


def test_collapse_invariants_leave_one_product_and_one_writer():
    assert not (evidence.ROOT / "src/mop").exists()
    assert (evidence.ROOT / "src/substrate/cli.py").is_file()
    assert audit.run()["results"]["exclusive_producers"]["ok"] is True
    writers = [path for path in (evidence.ROOT / "src/substrate").glob("*.py") if "def _atomic_write(" in path.read_text()]
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
