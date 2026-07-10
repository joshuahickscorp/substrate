import json

from mop.config import REPO_ROOT
from mop.studio_doctor import CHECK_NAMES, SCHEMA


def test_current_host_readiness_receipt_is_honest_and_complete():
    receipt = json.loads((REPO_ROOT / "proof/STUDIO_READINESS_CURRENT_HOST.json").read_text())
    assert receipt["schema"] == SCHEMA
    assert [check["name"] for check in receipt["checks"]] == list(CHECK_NAMES)
    assert receipt["summary"]["all_ok"] is True
    assert receipt["classification"]["studio_only_boundary_proven"] is False
    assert receipt["classification"]["measured_hardware_limits"] == []
    assert receipt["profile"]["resolved"] == "m3pro-local-max"
    assert receipt["host"]["chip"] == "Apple M3 Pro"
