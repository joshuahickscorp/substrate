import copy
import json
from types import SimpleNamespace

import pytest
import scripts.studio.wikimedia_av_intake as av_cli

from mop.studio import wikimedia_av_intake as intake


def _manifest():
    return intake.load_manifest(intake.DEFAULT_MANIFEST)


def _observed(row):
    return {
        "pageid": row["pageid"],
        "title": row["title"],
        "object_url": row["object_url"],
        "description_url": row["description_url"],
        "size_bytes": row["size_bytes"],
        "duration_seconds": row["duration_seconds"],
        "sha1": row["sha1"],
        "upload_timestamp": row["upload_timestamp"],
        "mime": "video/webm",
        "mediatype": "VIDEO",
        "video": dict(row["video"]),
        "audio": dict(row["audio"]),
        "license_short_name": "CC0",
        "license_code": "cc0",
        "license_url": row["license_url"],
        "attribution_required": "false",
        "restrictions": "",
        "credit": "<span>Own work</span>",
        "artist": row["artist"],
        "categories": ["CC-Zero", "Self-published work"],
    }


class _FakeCommons:
    def __init__(self, manifest, *, drift_pageid=None):
        self.rows = {row["pageid"]: _observed(row) for row in manifest["objects"]}
        if drift_pageid is not None:
            self.rows[drift_pageid]["sha1"] = "0" * 40

    def pages(self, pageids):
        ids = list(pageids)
        return {pageid: copy.deepcopy(self.rows[pageid]) for pageid in ids}


def test_frozen_manifest_is_small_source_disjoint_and_exact():
    manifest = _manifest()
    intake.validate_manifest(manifest)
    assert len(manifest["objects"]) == 12
    assert sum(row["size_bytes"] for row in manifest["objects"]) == 95_791_426
    assert sum(row["size_bytes"] for row in manifest["objects"] if row["role"] != "test") == 60_714_441
    assert {row["role"] for row in manifest["objects"]} == {"train", "validation", "test"}
    assert len({row["capture_family"] for row in manifest["objects"]}) == 12
    assert len({row["artist"] for row in manifest["objects"]}) == 12
    assert all(row["credit"] == "Own work" for row in manifest["objects"])


def test_manifest_refuses_duplicate_authority_relicense_and_weakened_floor():
    manifest = _manifest()
    manifest["objects"][1]["sha1"] = manifest["objects"][0]["sha1"]
    manifest["objects"][2]["license_code"] = "cc-by-nc"
    manifest["disk_policy"]["min_free_disk_bytes"] = 39_999_999_999
    with pytest.raises(intake.WikimediaAVIntakeError) as error:
        intake.validate_manifest(manifest)
    message = str(error.value)
    assert "sha1 values must be present and unique" in message
    assert "per-object license is not CC0" in message
    assert "disk floor was weakened" in message


def test_dry_run_is_metadata_only_and_keeps_test_locked(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=50_000_000_000),
    )
    plan = intake.build_dry_run_plan(manifest, client=_FakeCommons(manifest))
    assert plan["all_ok"] is True
    assert plan["media_requests"] == 0
    assert plan["media_bytes_requested"] == 0
    assert plan["test_media_accessed"] is False
    assert plan["safety"]["test_download_implemented"] is False
    assert plan["projected"]["source_bytes_total"] == 95_791_426
    assert plan["projected"]["train_validation_bytes"] == 60_714_441
    assert plan["projected"]["atomic_train_validation_reserve_bytes"] == 121_428_882
    assert plan["post_cm7"]["test_bytes_remain_locked"] == 35_076_985
    assert plan["claim_boundary"]["scientific_promotion"] is False
    assert plan["claim_boundary"]["al3_ready"] is False
    intake.validate_dry_run_plan(plan)


def test_dry_run_proof_is_self_verifying(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=50_000_000_000),
    )
    plan = intake.build_dry_run_plan(manifest, client=_FakeCommons(manifest))
    plan["manifest_file"] = "/an/untrusted/location.json"
    plan["manifest_file_sha256"] = "f" * 64
    intake.validate_dry_run_plan(plan)
    plan["objects"][0]["sha1"] = "0" * 40
    with pytest.raises(intake.WikimediaAVIntakeError, match="plan SHA-256 mismatch"):
        intake.validate_dry_run_plan(plan)


def test_dry_run_reports_live_authority_drift_without_weakening_claim(monkeypatch):
    manifest = _manifest()
    pageid = manifest["objects"][0]["pageid"]
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=50_000_000_000),
    )
    plan = intake.build_dry_run_plan(
        manifest,
        client=_FakeCommons(manifest, drift_pageid=pageid),
    )
    assert plan["all_ok"] is False
    assert plan["authority"]["objects_matching"] == 11
    assert any("sha1 drifted" in problem for problem in plan["problems"])
    assert plan["claim_boundary"]["status"] == "blocked-fail-closed"
    assert plan["claim_boundary"]["scientific_promotion"] is False


def test_dry_run_disk_guard_reserves_atomic_bytes_above_floor(monkeypatch):
    manifest = _manifest()
    reserve = 2 * manifest["disk_policy"]["projected_train_validation_bytes"]
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=intake.MIN_FREE_DISK_BYTES + reserve - 1),
    )
    plan = intake.build_dry_run_plan(manifest, client=_FakeCommons(manifest))
    assert plan["all_ok"] is False
    assert plan["safety"]["train_validation_download_allowed_by_disk"] is False
    assert any("disk guard" in problem for problem in plan["problems"])


def test_temporal_controls_are_frozen_and_not_a_scientific_result(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=50_000_000_000),
    )
    plan = intake.build_dry_run_plan(manifest, client=_FakeCommons(manifest))
    controls = plan["temporal_controls"]
    assert controls["within_clip_circular_offset_fractions"] == [0.25, 0.5, 0.75]
    assert controls["no_proxy_audio"] is True
    assert controls["no_caption_as_audio"] is True
    assert controls["no_test_control_tuning"] is True
    assert "manual privacy" in " ".join(plan["blockers_after_dry_run"])


def test_mux_validation_requires_one_real_audio_and_video_stream():
    row = _manifest()["objects"][0]
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "vp9",
                "width": 1080,
                "height": 1920,
                "start_time": "0.000",
            },
            {
                "codec_type": "audio",
                "codec_name": "opus",
                "sample_rate": "48000",
                "channels": 2,
                "start_time": "0.040",
            },
        ],
        "format": {"duration": "6.080"},
    }
    verified = intake._validate_ffprobe(row, probe)
    assert verified["same_original_mux_container"] is True
    assert verified["stream_start_delta_seconds"] == pytest.approx(0.04)
    with pytest.raises(intake.WikimediaAVIntakeError, match="one video and one audio"):
        intake._validate_ffprobe(row, {"streams": probe["streams"][:1], "format": probe["format"]})
    probe["streams"][1]["start_time"] = "0.250"
    with pytest.raises(intake.WikimediaAVIntakeError, match="more than 100 ms"):
        intake._validate_ffprobe(row, probe)


def test_api_client_uses_metadata_endpoint_and_normalizes_streams():
    row = _manifest()["objects"][0]
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": row["pageid"],
                    "title": row["title"],
                    "videoinfo": [
                        {
                            "timestamp": row["upload_timestamp"],
                            "size": row["size_bytes"],
                            "duration": row["duration_seconds"],
                            "url": row["object_url"],
                            "descriptionurl": row["description_url"],
                            "sha1": row["sha1"],
                            "mime": "video/webm",
                            "mediatype": "VIDEO",
                            "metadata": [
                                {
                                    "name": "audio",
                                    "value": [
                                        {"name": "dataformat", "value": "A_OPUS"},
                                        {"name": "sample_rate", "value": 48000},
                                        {"name": "channels", "value": 2},
                                        {"name": "language", "value": "und"},
                                    ],
                                },
                                {
                                    "name": "video",
                                    "value": [
                                        {"name": "dataformat", "value": "V_VP9"},
                                        {"name": "resolution_x", "value": 1080},
                                        {"name": "resolution_y", "value": 1920},
                                        {"name": "frame_rate", "value": 50},
                                    ],
                                },
                            ],
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC0"},
                                "License": {"value": "cc0"},
                                "LicenseUrl": {"value": row["license_url"]},
                                "AttributionRequired": {"value": "false"},
                                "Restrictions": {"value": ""},
                                "Credit": {"value": "Own work"},
                                "Artist": {"value": row["artist"]},
                                "Categories": {"value": "CC-Zero|Self-published work"},
                            },
                        }
                    ],
                }
            ]
        }
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            del size
            return json.dumps(payload).encode()

    def opener(request, timeout):
        assert request.full_url.startswith(intake.API_ENDPOINT)
        assert intake.OBJECT_HOST not in request.full_url
        assert timeout == 1.0
        return Response()

    observed = intake.WikimediaCommonsAPI(timeout=1.0, opener=opener).pages([row["pageid"]])
    assert observed[row["pageid"]]["sha1"] == row["sha1"]
    assert observed[row["pageid"]]["audio"]["codec"] == "A_OPUS"
    assert observed[row["pageid"]]["video"]["codec"] == "V_VP9"


def test_cli_refuses_execution_without_explicit_post_cm7_confirmation():
    assert av_cli.main(["--execute-train-validation"]) == 2
    assert av_cli.main(["--confirm-cm7-complete"]) == 2
