"""Studio readiness doctor. Pins the check surface (every expected check is present and shaped
{name,ok,detail}), proves the cache-write check actually writes and cleans up, proves the
config-validation check tracks the harness validator, proves render_md is a faithful record
(contains every check name), and proves the non-fatal contract: a missing video backend or an
unreachable HuggingFace are reported, never failures.
"""

import mop.studio_doctor as sd
from mop.harness.validate import check_all
from mop.studio_doctor import CHECK_NAMES, doctor, render_md


def test_doctor_returns_all_expected_checks():
    report = doctor()
    names = [c["name"] for c in report["checks"]]
    assert names == list(CHECK_NAMES)  # every expected check, in order, no extras
    assert set(names) == set(CHECK_NAMES)


def test_each_check_has_ok_and_detail():
    report = doctor()
    for c in report["checks"]:
        assert set(c) == {"name", "ok", "detail"}
        assert isinstance(c["ok"], bool)
        assert isinstance(c["detail"], str) and c["detail"]


def test_report_shape_and_summary_counts():
    report = doctor()
    assert set(report) == {"checks", "all_ok", "summary"}
    s = report["summary"]
    assert s["total"] == len(CHECK_NAMES)
    assert s["passed"] + s["failed"] == s["total"]
    assert report["all_ok"] == (s["failed"] == 0)
    assert report["all_ok"] == all(c["ok"] for c in report["checks"])


def _find(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


def test_cache_write_check_passes_and_cleans_up():
    report = doctor()
    cw = _find(report, "cache_write")
    assert cw["ok"], cw["detail"]
    # the probe must not leave its test file behind
    assert not (sd.REPO_ROOT / "data" / "cache" / ".studio_doctor_write_test").exists()


def test_config_validation_tracks_check_all():
    report = doctor()
    cv = _find(report, "config_validation")
    clean = len(check_all()) == 0
    assert cv["ok"] == clean
    if clean:
        assert "0 problems" in cv["detail"]


def test_encoders_check_lists_configs_with_dim():
    report = doctor()
    enc = _find(report, "encoders")
    assert enc["ok"]
    assert "configs" in enc["detail"] and "d=" in enc["detail"]  # name(d=...) rows


def test_python_and_torch_checks_pass_here():
    report = doctor()
    assert _find(report, "python")["ok"]
    assert _find(report, "torch")["ok"]  # torch is a hard dep of the project


def test_video_and_hf_are_non_fatal():
    # contract: these are REPORTED, never failures, regardless of presence/reachability
    report = doctor()
    assert _find(report, "video_backend")["ok"]
    assert _find(report, "huggingface")["ok"]


def test_video_backend_absent_still_ok(monkeypatch):
    # force every video backend import to fail: the check must still be ok (reported, not error)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("torchvision.io", "decord"):
            raise ImportError("forced absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, detail = sd._check_video_backend()
    assert ok and "none" in detail


def test_huggingface_unreachable_still_ok(monkeypatch):
    # force the metadata ping to blow up: the check must catch it and stay ok
    class BoomApi:
        def model_info(self, *a, **k):
            raise RuntimeError("network down")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", BoomApi)
    ok, detail = sd._check_huggingface()
    assert ok and "unreachable" in detail


def test_probe_exception_becomes_failed_check():
    def boom():
        raise ValueError("kaboom")

    c = sd._check("x", boom)
    assert c == {"name": "x", "ok": False, "detail": "ValueError: kaboom"}


def test_render_md_contains_every_check():
    report = doctor()
    md = render_md(report)
    assert md.startswith("# Studio readiness doctor")
    assert "| check | status | detail |" in md
    for c in report["checks"]:
        assert c["name"] in md  # every check name appears as a row
    verdict = "STUDIO READY" if report["all_ok"] else "NOT READY"
    assert verdict in md
    # one header row + one separator + one row per check, plus title block
    assert md.count("\n|") >= len(report["checks"]) + 1
