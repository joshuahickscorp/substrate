import json
from types import SimpleNamespace

import pytest

import mop.studio_doctor as sd
from mop.harness.validate import check_all


@pytest.fixture(scope="module")
def report():
    return sd.doctor()


def _find(report, name):
    return next(check for check in report["checks"] if check["name"] == name)


def test_report_contract_is_complete_and_honest(report):
    checks = report["checks"]
    assert [check["name"] for check in checks] == list(sd.CHECK_NAMES)
    assert all(set(check) == {"name", "ok", "detail"} for check in checks)
    assert all(isinstance(check["ok"], bool) and check["detail"] for check in checks)
    summary = report["summary"]
    assert summary["total"] == len(sd.CHECK_NAMES)
    assert summary["passed"] + summary["failed"] == summary["total"]
    assert report["all_ok"] == all(check["ok"] for check in checks) == (summary["failed"] == 0)
    assert report["classification"]["studio_only_boundary_proven"] is False
    assert report["classification"]["measured_hardware_limits"] == []


def test_sealed_current_host_receipt_retains_readiness_boundary():
    receipt = json.loads((sd.REPO_ROOT / "proof/STUDIO_READINESS_CURRENT_HOST.json").read_text())
    assert (receipt["schema"], receipt["summary"]["all_ok"]) == (sd.SCHEMA, True)
    assert [check["name"] for check in receipt["checks"]] == list(sd.CHECK_NAMES)
    assert receipt["classification"]["studio_only_boundary_proven"] is False
    assert receipt["classification"]["measured_hardware_limits"] == []
    assert (receipt["profile"]["resolved"], receipt["host"]["chip"]) == ("m3pro-local-max", "Apple M3 Pro")


def test_local_baseline_probes_have_required_semantics(report):
    assert _find(report, "python")["ok"] and _find(report, "torch")["ok"]
    assert _find(report, "huggingface")["ok"]
    assert "network not probed" in _find(report, "huggingface")["detail"]
    assert isinstance(_find(report, "video_backend")["ok"], bool)
    encoders = _find(report, "encoders")
    assert encoders["ok"] and "configs" in encoders["detail"] and "d=" in encoders["detail"]
    config = _find(report, "config_validation")
    assert config["ok"] == (not check_all())


def test_cache_write_probe_passes_and_cleans_up(report):
    probe = _find(report, "cache_write")
    assert probe["ok"], probe["detail"]
    assert not (sd.REPO_ROOT / "data/cache/.studio_doctor_write_test").exists()


def test_profile_floor_reports_block(monkeypatch):
    profile = SimpleNamespace(name="local", min_free_disk_gb=60.0, free_disk_ok=lambda: (False, 13.3))
    monkeypatch.setattr(sd, "get_profile", lambda _name: profile)
    ok, detail = sd._check_profile_floor("local")
    assert not ok and "PROFILE BLOCKED" in detail


def test_explicit_studio_profile_cannot_impersonate_small_host(monkeypatch):
    measured = {"chip": "Laptop", "unified_memory_gb": 18.0, "disk_total_gb": 500.0}
    profile = SimpleNamespace(
        name="studio-example",
        min_host_unified_memory_gb=96.0,
        min_host_disk_gb=1000.0,
        host_compatibility=lambda: (False, ["low memory"], measured),
    )
    monkeypatch.setattr(sd, "get_profile", lambda _name: profile)
    ok, detail = sd._check_profile_host("studio-example")
    assert not ok and "PROFILE/HOST MISMATCH" in detail


def test_video_backend_absence_fails_closed_in_isolation(monkeypatch):
    calls = []
    missing = type("Missing", (), {"returncode": 1, "stdout": "", "stderr": "ModuleNotFoundError"})()

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return missing

    sd._check_video_backend.cache_clear()
    monkeypatch.setattr(sd.subprocess, "run", run)
    try:
        ok, detail = sd._check_video_backend()
    finally:
        sd._check_video_backend.cache_clear()
    assert not ok and ".[video]" in detail and len(calls) == 2
    assert all(command[1:3] == ["-I", "-c"] for command, _ in calls)
    assert all(kwargs["cwd"] != str(sd.REPO_ROOT) for _, kwargs in calls)


def test_huggingface_check_never_calls_network(monkeypatch):
    class BoomApi:
        def model_info(self, *_args, **_kwargs):
            raise RuntimeError("network down")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", BoomApi)
    assert sd._check_huggingface()[0]


def test_package_import_probe_is_isolated():
    ok, detail = sd._check_package_import()
    assert isinstance(ok, bool) and ("PYTHONPATH" in detail or "isolated cwd" in detail)


def test_memory_telemetry_fails_when_required_fields_are_missing(monkeypatch):
    monkeypatch.setattr(sd, "_memory_snapshot", lambda: 1 / 0)
    ok, detail = sd._check_memory_telemetry()
    assert not ok and "missing" in detail


def test_local_weight_detection_uses_cache_only(monkeypatch, tmp_path):
    shard = tmp_path / "hub/models--org--model/snapshots/sha/model.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert sd._local_weight_files("org/model") == [shard]


def test_cache_manifest_check_does_not_pass_vacuously(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "REPO_ROOT", tmp_path)
    ok, detail = sd._check_cache_manifests()
    assert not ok and "0 latent stores" in detail


def test_probe_exception_becomes_failed_check():
    def boom():
        raise ValueError("kaboom")

    assert sd._check("x", boom) == {"name": "x", "ok": False, "detail": "ValueError: kaboom"}


def test_markdown_contains_verdict_and_every_check(report):
    markdown = sd.render_md(report)
    assert markdown.startswith("# Host and Studio-transfer readiness doctor")
    assert "| check | status | detail |" in markdown
    assert all(check["name"] in markdown for check in report["checks"])
    verdict = (
        f"CURRENT HOST READY FOR {report['profile']['resolved']}"
        if report["all_ok"]
        else "CURRENT HOST NOT READY"
    )
    assert verdict in markdown
