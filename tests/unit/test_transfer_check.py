import json

from mop.studio.transfer_check import SCHEMA, TransferCheckConfig, run_transfer_check, write_transfer_report


def test_transfer_check_passes_on_current_repo_when_dirty_allowed():
    report = run_transfer_check(TransferCheckConfig(allow_dirty=True, audit_path=None))
    assert report["schema"] == SCHEMA
    assert report["all_ok"] is True
    names = {c["name"] for c in report["checks"]}
    assert "profile" in names
    assert "git_state" in names
    assert "cache_manifests" in names
    assert "path:scripts/studio/__main__.py" in names


def test_transfer_check_requires_studio_m1ultra_profile():
    report = run_transfer_check(
        TransferCheckConfig(profile_name="m3pro-local-max", allow_dirty=True, audit_path=None)
    )
    profile = next(c for c in report["checks"] if c["name"] == "profile")
    assert profile["ok"] is False
    assert report["all_ok"] is False


def test_transfer_check_can_write_report(tmp_path):
    report = run_transfer_check(
        TransferCheckConfig(allow_dirty=True, audit_path=None, require_receipts=False)
    )
    out = tmp_path / "transfer.json"
    write_transfer_report(report, out)
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == SCHEMA
