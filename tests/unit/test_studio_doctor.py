"""Studio readiness doctor. Pins the check surface (every expected check is present and shaped
{name,ok,detail}), proves the cache-write check actually writes and cleans up, proves the
config-validation check tracks the harness validator, proves render_md is a faithful record
(contains every check name), and proves the non-fatal contract: a missing video backend or an
unreachable HuggingFace are reported, never failures.
"""

import hashlib
import json

from omegaconf import OmegaConf

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
    assert set(report) == {
        "schema",
        "created_at",
        "profile",
        "host",
        "checks",
        "classification",
        "all_ok",
        "summary",
    }
    s = report["summary"]
    assert s["total"] == len(CHECK_NAMES)
    assert s["passed"] + s["failed"] == s["total"]
    assert report["all_ok"] == (s["failed"] == 0)
    assert report["all_ok"] == all(c["ok"] for c in report["checks"])
    assert report["classification"]["studio_only_boundary_proven"] is False
    assert report["classification"]["measured_hardware_limits"] == []


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


def test_profile_floor_reports_blocked_profile(monkeypatch):
    class DummyProfile:
        name = "m3pro-local-max"
        min_free_disk_gb = 60.0

        def free_disk_ok(self):
            return False, 13.3

    monkeypatch.setattr(sd, "get_profile", lambda name: DummyProfile())
    ok, detail = sd._check_profile_floor("m3pro-local-max")
    assert not ok
    assert "m3pro-local-max" in detail
    assert "PROFILE BLOCKED" in detail


def test_python_and_torch_checks_pass_here():
    report = doctor()
    assert _find(report, "python")["ok"]
    assert _find(report, "torch")["ok"]  # torch is a hard dep of the project


def test_video_is_strict_and_hf_never_probes_network():
    report = doctor()
    assert isinstance(_find(report, "video_backend")["ok"], bool)
    assert _find(report, "huggingface")["ok"]
    assert "network not probed" in _find(report, "huggingface")["detail"]


def test_video_backend_absent_fails_closed_in_isolated_process(monkeypatch):
    calls = []

    class MissingBackend:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: forced absent"

    def missing_backend(command, **kwargs):
        calls.append((command, kwargs))
        return MissingBackend()

    sd._check_video_backend.cache_clear()
    monkeypatch.setattr(sd.subprocess, "run", missing_backend)
    try:
        ok, detail = sd._check_video_backend()
    finally:
        sd._check_video_backend.cache_clear()

    assert not ok and ".[video]" in detail
    assert len(calls) == 2
    assert all(command[1:3] == ["-I", "-c"] for command, _kwargs in calls)
    assert all(kwargs["cwd"] != str(sd.REPO_ROOT) for _command, kwargs in calls)


def test_huggingface_check_does_not_call_network(monkeypatch):
    class BoomApi:
        def model_info(self, *a, **k):
            raise RuntimeError("network down")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", BoomApi)
    ok, detail = sd._check_huggingface()
    assert ok and "network not probed" in detail


def test_explicit_studio_profile_cannot_impersonate_small_host(monkeypatch):
    class DummyProfile:
        name = "studio-example"
        min_host_unified_memory_gb = 96.0
        min_host_disk_gb = 1000.0

        def host_compatibility(self):
            return (
                False,
                ["unified memory 18 GB below 96 GB"],
                {
                    "chip": "Laptop",
                    "unified_memory_gb": 18.0,
                    "disk_total_gb": 500.0,
                },
            )

    monkeypatch.setattr(sd, "get_profile", lambda name: DummyProfile())
    ok, detail = sd._check_profile_host("studio-example")
    assert not ok
    assert "PROFILE/HOST MISMATCH" in detail


def test_package_import_probe_is_isolated_from_pythonpath():
    ok, detail = sd._check_package_import()
    # This reflects the invoking environment. A source-only PYTHONPATH run must fail rather than
    # being mistaken for a portable install; an installed environment reports the isolated path.
    assert isinstance(ok, bool)
    assert "PYTHONPATH" in detail or "isolated cwd" in detail


def test_memory_telemetry_fails_when_required_fields_are_missing(monkeypatch):
    monkeypatch.setattr(
        sd,
        "memory_snapshot",
        lambda stage: {
            "process_rss_gb": None,
            "system_total_gb": None,
            "system_available_gb": None,
        },
    )
    ok, detail = sd._check_memory_telemetry()
    assert not ok
    assert "missing" in detail


def test_local_weight_detection_uses_cache_only(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    shard = cache / "models--org--model" / "snapshots" / "sha" / "model.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert sd._local_weight_files("org/model") == [shard]


def _direct_checkpoint_fixture(monkeypatch, tmp_path):
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "config.yaml").write_text("defaults:\n  encoder: fixture\n")
    payload = b"official-direct-checkpoint-fixture"
    checkpoint = tmp_path / "data" / "models" / "vjepa21" / "fixture.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source_url = "https://weights.example/fixture.pt"
    receipt = {
        "schema": "mop-vjepa21-official-checkpoint/v1",
        "all_ok": True,
        "size": len(payload),
        "sha256": digest,
        "source_url": source_url,
        "source_etag": '"fixture-etag"',
        "source_version_id": "fixture-version",
        "repository_commit": "fixture-commit",
    }
    receipt_path = checkpoint.with_name(checkpoint.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt))
    cfg = OmegaConf.create(
        {
            "name": "vjepa21_fixture",
            "available": True,
            "source_kind": "official_pytorch_checkpoint",
            "hf_id": "official-pytorch-only-fixture",
            "checkpoint_url": source_url,
            "checkpoint_content_length": len(payload),
            "checkpoint_sha256": digest,
            "checkpoint_etag": receipt["source_etag"],
            "checkpoint_version_id": receipt["source_version_id"],
            "official_repo_commit": receipt["repository_commit"],
        }
    )
    monkeypatch.setattr(sd, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sd, "_encoder_configs", lambda: [(config_root / "encoder.yaml", cfg)])
    monkeypatch.setattr(
        sd,
        "_local_weight_files",
        lambda _hf_id: (_ for _ in ()).throw(AssertionError("direct checkpoint used HF lookup")),
    )
    monkeypatch.setattr(sd, "_hf_cache_roots", lambda: [])
    sd._sha256_snapshot.cache_clear()
    return checkpoint, receipt_path, receipt


def test_official_pytorch_checkpoint_uses_direct_file_and_receipt(monkeypatch, tmp_path):
    checkpoint, receipt_path, _ = _direct_checkpoint_fixture(monkeypatch, tmp_path)
    ok, detail = sd._check_encoder_weights("studio-1tb")
    assert ok, detail
    assert "vjepa21_fixture" in detail
    assert str(checkpoint) in detail
    assert str(receipt_path) in detail
    assert "official-pytorch-only-fixture" not in detail


def test_official_pytorch_checkpoint_rejects_same_size_hash_drift(monkeypatch, tmp_path):
    checkpoint, _, _ = _direct_checkpoint_fixture(monkeypatch, tmp_path)
    checkpoint.write_bytes(b"x" * checkpoint.stat().st_size)
    sd._sha256_snapshot.cache_clear()
    ok, detail = sd._check_encoder_weights("studio-1tb")
    assert not ok
    assert "file SHA256 does not match configured checkpoint_sha256" in detail
    assert "checkpoint receipt SHA256 does not match config and file" in detail


def test_official_pytorch_checkpoint_rejects_receipt_drift(monkeypatch, tmp_path):
    _, receipt_path, receipt = _direct_checkpoint_fixture(monkeypatch, tmp_path)
    receipt["all_ok"] = False
    receipt_path.write_text(json.dumps(receipt))
    ok, detail = sd._check_encoder_weights("studio-1tb")
    assert not ok
    assert "checkpoint receipt is not green" in detail


def test_cache_manifest_check_does_not_pass_vacuously(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "REPO_ROOT", tmp_path)
    ok, detail = sd._check_cache_manifests()
    assert not ok
    assert "0 latent stores" in detail


def test_probe_exception_becomes_failed_check():
    def boom():
        raise ValueError("kaboom")

    c = sd._check("x", boom)
    assert c == {"name": "x", "ok": False, "detail": "ValueError: kaboom"}


def test_render_md_contains_every_check():
    report = doctor()
    md = render_md(report)
    assert md.startswith("# Host and Studio-transfer readiness doctor")
    assert "| check | status | detail |" in md
    for c in report["checks"]:
        assert c["name"] in md  # every check name appears as a row
    profile_name = report["profile"]["resolved"]
    verdict = f"CURRENT HOST READY FOR {profile_name}" if report["all_ok"] else "CURRENT HOST NOT READY"
    assert verdict in md
    # one header row + one separator + one row per check, plus title block
    assert md.count("\n|") >= len(report["checks"]) + 1
