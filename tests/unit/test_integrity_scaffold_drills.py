from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.run_integrity_scaffold_drills import build_receipt, main
from scripts.verify_integrity_scaffold_drills import verify_payload_sha256, verify_receipt


def test_integrity_drills_execute_every_declared_control_and_verify() -> None:
    receipt = build_receipt()
    assert receipt["status"] == "mechanics-pass"
    assert receipt["f59"]["independent_unit_count"] == 6
    assert receipt["f60"]["independent_unit_count"] == 3
    assert receipt["difficulty_calibration"]["ceilinged_tie"] is False
    verifier = receipt["independent_verifier"]
    assert verifier["verified"] is True
    assert verifier["fresh_seed_count"] == 3
    assert verifier["fresh_seeds_disjoint_from_primary"] is True
    assert verifier["all_mutations_rejected"] is True
    assert len(verifier["mutation_tests"]) == 6
    assert verify_payload_sha256(receipt)


def test_integrity_independent_replay_rejects_stage_artifact_tamper() -> None:
    receipt = build_receipt()
    tampered = copy.deepcopy(receipt)
    wellformed = tampered["f60"]["units"][0]["cases"][-1]
    wellformed["request"]["stage_artifacts"]["shadow"]["status"] = "fail"
    replay = verify_receipt(tampered, check_live_files=False, run_mutations=True)
    assert replay["verified"] is False
    assert any("f60" in error for error in replay["errors"])
    assert replay["all_mutations_rejected"] is True


def test_integrity_driver_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(["--out", str(first)]) == 0
    assert main(["--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert verify_payload_sha256(json.loads(first.read_text(encoding="utf-8")))
