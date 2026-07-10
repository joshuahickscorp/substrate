import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from mop.substrate import vjepa21_official as vj


def _write_config(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": vj.VITB["slug"],
                "arch": vj.VITB["architecture"],
                "embed_dim": vj.VITB["embed_dim"],
                "patch_size": vj.VITB["patch_size"],
                "tubelet": vj.VITB["tubelet_size"],
                "frames_per_clip": vj.VITB["configured_frames"],
                "resolution": vj.VITB["resolution"],
                "dense": True,
                "pool": "none",
                "source_kind": "official_pytorch_checkpoint",
                "official_repo_commit": vj.OFFICIAL_REPOSITORY_COMMIT,
                "hub_entrypoint": vj.VITB["hub_entrypoint"],
                "checkpoint_url": vj.VITB["checkpoint_url"],
                "available": True,
                "availability_state": "local_hash_strict_load_and_8f_64f_forward_verified",
                "cache_first_only": True,
                "checkpoint_sha256": ("848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"),
                "prefer_real": False,
            }
        )
    )


def test_dense_token_contract_is_spatiotemporal_and_exact():
    assert vj.expected_dense_tokens(8) == 2304
    assert vj.expected_dense_tokens(16) == 4608
    assert vj.expected_dense_tokens(64) == 18432
    with pytest.raises(ValueError, match="tubelet"):
        vj.expected_dense_tokens(7)


def test_config_validation_binds_official_source_and_rejects_unavailable_regression(tmp_path):
    path = tmp_path / "config.yaml"
    _write_config(path)
    assert vj.validate_vitb_config(path)["all_ok"] is True
    raw = yaml.safe_load(path.read_text())
    raw["available"] = False
    path.write_text(yaml.safe_dump(raw))
    report = vj.validate_vitb_config(path)
    assert report["all_ok"] is False
    assert "available" in " ".join(report["problems"])


def test_remote_metadata_and_ranges_must_both_match(monkeypatch):
    observed_version = {"value": vj.VITB["checkpoint_version_id"]}
    monkeypatch.setattr(
        vj,
        "_head",
        lambda url, timeout=30.0: {
            "status": 200,
            "final_url": url,
            "headers": {
                "content-length": str(vj.VITB["checkpoint_content_length"]),
                "etag": vj.VITB["checkpoint_etag"],
                "last-modified": vj.VITB["checkpoint_last_modified"],
                "x-amz-version-id": observed_version["value"],
                "content-type": vj.VITB["checkpoint_content_type"],
                "accept-ranges": "bytes",
            },
        },
    )
    width = vj.VITB["range_bytes"]
    total = vj.VITB["checkpoint_content_length"]
    first = b"a" * width
    last = b"b" * width
    monkeypatch.setitem(vj.VITB, "first_range_sha256", hashlib.sha256(first).hexdigest())
    monkeypatch.setitem(vj.VITB, "last_range_sha256", hashlib.sha256(last).hexdigest())

    def fake_range(url, start, end, timeout=30.0):
        del url, timeout
        payload = first if start == 0 else last
        return payload, {
            "status": 206,
            "headers": {"content-range": f"bytes {start}-{end}/{total}"},
            "final_url": "official",
        }

    monkeypatch.setattr(vj, "_range_bytes", fake_range)
    assert vj.validate_checkpoint_remote()["all_ok"] is True
    observed_version["value"] = "changed"
    assert vj.validate_checkpoint_remote()["all_ok"] is False


def test_disk_projection_preserves_floor_and_working_headroom(monkeypatch, tmp_path):
    required = (
        vj.MIN_FREE_DISK_BYTES + vj.VITB["checkpoint_content_length"] + vj.DOWNLOAD_WORKING_HEADROOM_BYTES
    )
    monkeypatch.setattr(
        vj.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100_000_000_000, used=1, free=required - 1),
    )
    assert vj.disk_report(tmp_path)["download_feasible_now"] is False
    monkeypatch.setattr(
        vj.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100_000_000_000, used=1, free=required),
    )
    assert vj.disk_report(tmp_path)["download_feasible_now"] is True


def test_heavy_lane_blocks_download_readiness(monkeypatch):
    class Process:
        info = {
            "pid": 42,
            "cmdline": ["python", "scripts/custom_substrate_workbench.py", "cm7"],
        }

    import psutil

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [Process()])
    report = vj.active_heavy_lane_report()
    assert report["clear_for_new_heavy_lane"] is False
    assert report["active"] == [{"pid": 42, "matched_patterns": ["custom_substrate_workbench.py cm7"]}]


def test_doctor_must_be_fresh_green_and_local_profile(tmp_path):
    from datetime import UTC, datetime

    path = tmp_path / "doctor.json"
    path.write_text(
        json.dumps(
            {
                "schema": "mop-studio-readiness/v2",
                "created_at": datetime.now(UTC).isoformat(),
                "all_ok": True,
                "profile": {"resolved": "m3pro-local-max"},
                "summary": {"passed": 15, "total": 15},
            }
        )
    )
    assert vj.doctor_receipt_report(path)["fresh_and_green"] is True
    payload = json.loads(path.read_text())
    payload["all_ok"] = False
    path.write_text(json.dumps(payload))
    assert vj.doctor_receipt_report(path)["fresh_and_green"] is False


def test_checkpoint_without_authority_receipt_never_loads(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not official")
    report = vj.validate_checkpoint_receipt(checkpoint)
    assert report["all_ok"] is False
    assert "receipt missing" in " ".join(report["problems"])


def test_checkpoint_receipt_binds_full_local_sha(monkeypatch, tmp_path):
    payload = b"official fixture bytes"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(payload)
    monkeypatch.setitem(vj.VITB, "checkpoint_content_length", len(payload))
    receipt = {
        "schema": vj.CHECKPOINT_RECEIPT_SCHEMA,
        "source_url": vj.VITB["checkpoint_url"],
        "source_etag": vj.VITB["checkpoint_etag"],
        "source_version_id": vj.VITB["checkpoint_version_id"],
        "size": len(payload),
        "repository_commit": vj.OFFICIAL_REPOSITORY_COMMIT,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    vj.checkpoint_receipt_path(checkpoint).write_text(json.dumps(receipt))
    assert vj.validate_checkpoint_receipt(checkpoint)["all_ok"] is True
    checkpoint.write_bytes(payload + b"corrupt")
    assert vj.validate_checkpoint_receipt(checkpoint)["all_ok"] is False


def test_live_official_seam_has_no_larger_variant_catalog():
    assert not hasattr(vj, "LARGER_VARIANTS")
    assert vj.VITB["slug"] == "vjepa21_vitb"
