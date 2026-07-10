import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import scripts.studio.sanpo_real_intake as sanpo_cli

from mop.studio import sanpo_real_intake as intake


def _meta(name, payload=b"x", *, size=None, generation="1"):
    return {
        "name": name,
        "size": len(payload) if size is None else size,
        "md5Hash": base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(),
        "crc32c": base64.b64encode(intake.crc32c((payload,)).to_bytes(4, "big")).decode(),
        "generation": generation,
        "etag": f"etag-{generation}",
    }


class _FakeGCS:
    def __init__(self):
        self.payloads = {}
        self.metadata_by_name = {}
        self.frame_lists = {}
        train = [f"n{i}" for i in range(4)] + [f"p{i}" for i in range(4)]
        test = ["tn0", "tp0"]
        for split, rows in (("train", train), ("test", test)):
            name = f"{intake.OFFICIAL_PREFIX}/splits/{split}_session_ids.txt"
            payload = ("\n".join(rows) + "\n").encode()
            self.payloads[name] = payload
            self.metadata_by_name[name] = _meta(name, payload)
        for session_id in [*train, *test]:
            is_park = session_id.startswith("p") or session_id.startswith("tp")
            description = json.dumps(
                {
                    "session_type": "real",
                    "session_video_metadata": {
                        "environment_types": [
                            "ENVIRONMENT_TYPE_PARK" if is_park else "ENVIRONMENT_TYPE_URBAN"
                        ],
                        "visibility": "VISIBILITY_HIGH",
                        "ego_motion": "EGO_MOTION_WALKING",
                    },
                }
            ).encode()
            description_name = f"{intake.OFFICIAL_PREFIX}/{session_id}/description.json"
            self.payloads[description_name] = description
            self.metadata_by_name[description_name] = _meta(description_name, description)
            prefix = f"{intake.OFFICIAL_PREFIX}/{session_id}/camera_head/left/video_frames/"
            frames = []
            for frame_index in intake.FRAME_INDICES:
                name = f"{prefix}{frame_index:06d}.png"
                frames.append(_meta(name, b"frame", size=3_500_000, generation=str(frame_index + 1)))
            self.frame_lists[prefix] = frames

    def metadata(self, name):
        return self.metadata_by_name[name]

    def verified_bytes(self, metadata, *, max_bytes=1_000_000):
        del max_bytes
        return self.payloads[metadata["name"]]

    def list_metadata(self, prefix, *, limit=None):
        rows = self.frame_lists[prefix]
        return rows if limit is None else rows[:limit]


def _repo_authority():
    return {
        "repository": intake.OFFICIAL_REPOSITORY,
        "commit": intake.OFFICIAL_REPO_COMMIT,
        "commit_api": "https://api.github.com/pinned",
        "commit_verified": True,
        "artifacts": [dict(item) for item in intake.REPOSITORY_ARTIFACTS],
        "required_statements_verified": {
            "dataset_license": "Creative Commons V4.0",
            "official_split": "mutually exclusive session IDs",
        },
    }


@pytest.fixture
def plan(monkeypatch, tmp_path):
    monkeypatch.setattr(intake, "_repo_authority", lambda opener: _repo_authority())
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(total=500_000_000_000, used=300_000_000_000, free=100_000_000_000),
    )
    return intake.build_intake_plan(client=_FakeGCS(), disk_root=tmp_path)


def test_crc32c_known_vector():
    assert intake.crc32c((b"123456789",)) == 0xE3069283


def test_verify_bytes_checks_official_md5_crc_and_assigns_sha256():
    payload = b"official bytes"
    metadata = _meta("sanpo_dataset/v0/sanpo-real/x", payload)
    hashes = intake.verify_bytes(payload, metadata)
    assert hashes["official_integrity_verified"] is True
    assert hashes["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(intake.SanpoIntakeError, match="integrity mismatch"):
        intake.verify_bytes(payload + b"corrupt", metadata)


def test_plan_is_balanced_and_keeps_official_test_isolated(plan):
    intake.validate_intake_plan(plan, current_free_bytes=100_000_000_000)
    counts = {}
    for session in plan["sessions"]:
        key = (session["role"], session["is_park"])
        counts[key] = counts.get(key, 0) + 1
        if session["role"] == "test":
            assert session["official_split"] == "test"
        else:
            assert session["official_split"] == "train"
    assert counts == {
        ("train", False): 3,
        ("train", True): 3,
        ("validation", False): 1,
        ("validation", True): 1,
        ("test", False): 1,
        ("test", True): 1,
    }
    assert plan["projected"]["sessions"] == 10
    assert plan["projected"]["frames"] == 80
    assert plan["projected"]["frame_bytes"] == 280_000_000
    assert plan["claim_boundary"]["scientific_promotion"] is False
    assert plan["claim_boundary"]["f8_f16_trusted_provenance_satisfied"] is False


def test_plan_refuses_test_leakage_and_weakened_disk_floor(plan):
    plan["sessions"][0]["official_split"] = "test"
    plan["safety"]["min_free_disk_bytes"] = intake.MIN_FREE_DISK_BYTES - 1
    with pytest.raises(intake.SanpoIntakeError) as error:
        intake.validate_intake_plan(plan, current_free_bytes=100_000_000_000)
    assert "train must come from official train" in str(error.value)
    assert "disk floor was weakened" in str(error.value)


def test_plan_refuses_frame_reordering(plan):
    plan["sessions"][0]["frames"] = list(reversed(plan["sessions"][0]["frames"]))
    with pytest.raises(intake.SanpoIntakeError, match="ordered frame indices"):
        intake.validate_intake_plan(plan, current_free_bytes=100_000_000_000)


def test_plan_identity_excludes_live_disk_observations(plan):
    changed = copy.deepcopy(plan)
    changed["safety"]["free_disk_before_plan_bytes"] -= 9_000_000_000
    changed["safety"]["projected_free_after_bytes"] -= 9_000_000_000
    assert intake._sha256_json(intake._plan_identity(changed)) == plan["plan_identity_sha256"]


def test_atomic_download_resumes_and_verifies_before_rename(monkeypatch, tmp_path):
    payload = b"abcdef"
    authority = _meta("sanpo_dataset/v0/sanpo-real/x", payload)
    destination = tmp_path / "frame.png"
    (tmp_path / "frame.png.part").write_bytes(b"abc")

    class Response:
        status = 206

        def __init__(self):
            self._chunks = [b"def", b""]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            del size
            return self._chunks.pop(0)

    def opener(request, timeout):
        assert request.headers["Range"] == "bytes=3-"
        assert timeout == 1.0
        return Response()

    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=100_000_000_000),
    )
    hashes, transfer = intake._download_atomic_resumable(
        url="https://storage.googleapis.com/pinned",
        destination=destination,
        authority=authority,
        disk_floor_bytes=intake.MIN_FREE_DISK_BYTES,
        opener=opener,
        timeout=1.0,
    )
    assert destination.read_bytes() == payload
    assert not (tmp_path / "frame.png.part").exists()
    assert transfer["status"] == "resumed"
    assert transfer["resumed_from_bytes"] == 3
    assert hashes["sha256"] == hashlib.sha256(payload).hexdigest()


def test_disk_guard_aborts_before_transfer(monkeypatch, tmp_path):
    payload = b"abcdef"
    authority = _meta("sanpo_dataset/v0/sanpo-real/x", payload)
    monkeypatch.setattr(
        intake.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=intake.MIN_FREE_DISK_BYTES + len(payload) - 1),
    )
    with pytest.raises(intake.SanpoIntakeError, match="disk guard failed"):
        intake._download_atomic_resumable(
            url="https://storage.googleapis.com/pinned",
            destination=tmp_path / "frame.png",
            authority=authority,
            disk_floor_bytes=intake.MIN_FREE_DISK_BYTES,
            opener=lambda *args, **kwargs: pytest.fail("network must not be touched"),
        )


def test_consumer_manifest_has_loader_handoff_and_content_set_hash(plan):
    records = {
        item["local_path"]: {
            "size": item["size"],
            "md5_base64": item["md5Hash"],
            "crc32c_base64": item["crc32c"],
            "sha256": hashlib.sha256(item["name"].encode()).hexdigest(),
        }
        for item in plan["gcs_objects"]
    }
    manifest = intake._consumer_manifest(plan, records)
    assert manifest["schema"] == intake.CONSUMER_SCHEMA
    assert len(manifest["sessions"]) == 10
    assert len(manifest["content_entries"]) == 90
    assert len(manifest["content_set_sha256"]) == 64
    assert all(len(session["ordered_frames"]) == 8 for session in manifest["sessions"])
    assert all(
        session["test_only_no_tuning"] is (session["role"] == "test") for session in manifest["sessions"]
    )


def test_cli_refuses_to_lower_floor():
    assert sanpo_cli.main(["--min-free-gb", "39.9"]) == 2
