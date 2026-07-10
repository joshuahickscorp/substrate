from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.run_material_twin_batteries import build_receipt, main
from scripts.verify_material_twin_batteries import verify_payload_sha256, verify_receipt


def test_material_batteries_close_f61_to_f64_at_honest_scope() -> None:
    receipt = build_receipt()
    assert receipt["status"] == "complete"
    assert receipt["f61"]["result"] == "programmatic-mechanics-pass"
    assert receipt["f62"]["result"] == "null"
    assert receipt["f63"]["result"] == "favorable-programmatic-pilot"
    assert receipt["f64"]["result"] == "null"
    assert all(row["tie_with_restart"] for row in receipt["f64"]["units"])
    assert receipt["difficulty_calibration"]["f62_ceilinged"] is False
    assert receipt["difficulty_calibration"]["ceilinged_tie_promoted"] is False
    verifier = receipt["independent_verifier"]
    assert verifier["verified"] is True
    assert verifier["fresh_seed_count"] == 3
    assert verifier["fresh_seeds_disjoint_from_primary"] is True
    assert verifier["all_mutations_rejected"] is True
    assert len(verifier["mutation_tests"]) == 6
    assert verify_payload_sha256(receipt)


def test_material_replay_rejects_false_f64_promotion() -> None:
    receipt = build_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["f64"]["units"][0]["result"] = "favorable"
    replay = verify_receipt(tampered, check_live_files=False, run_mutations=True)
    assert replay["verified"] is False
    assert any("f64" in error for error in replay["errors"])
    assert replay["all_mutations_rejected"] is True


def test_material_driver_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(["--out", str(first)]) == 0
    assert main(["--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert verify_payload_sha256(json.loads(first.read_text(encoding="utf-8")))
