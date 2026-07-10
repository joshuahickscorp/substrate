import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from PIL import __version__ as pillow_version

from mop.substrate import sanpo_bridge as bridge


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fixture(root: Path) -> Path:
    data = root / "data"
    proof = root / "proof"
    config = root / "config"
    roles = {
        "train": [f"train-{index}" for index in range(6)],
        "validation": [f"validation-{index}" for index in range(2)],
        "test": [f"test-{index}" for index in range(2)],
    }
    sessions = []
    entries = []
    referents = []
    download_records = []
    for role, session_ids in roles.items():
        for role_index, session_id in enumerate(session_ids):
            is_park = role_index % 2 == 1
            attributes = {
                "environment_types": ["ENVIRONMENT_TYPE_PARK" if is_park else "ENVIRONMENT_TYPE_URBAN"],
                "ego_motion": "EGO_MOTION_WALKING",
                "visibility": "VISIBILITY_HIGH",
            }
            description_path = f"sessions/{session_id}/description.json"
            _write_json(
                data / description_path,
                {"session_type": "real", "session_video_metadata": attributes},
            )
            description_sha = _sha(data / description_path)
            description_entry = {
                "path": description_path,
                "gcs_object": f"official/{description_path}",
                "generation": "1",
                "size": (data / description_path).stat().st_size,
                "sha256": description_sha,
            }
            entries.append(description_entry)
            download_records.append({"local_path": description_path, "sha256": description_sha})
            ordered = []
            for frame_index in bridge.FRAME_INDICES:
                frame_path = f"sessions/{session_id}/frames/{frame_index:06d}.png"
                path = data / frame_path
                path.parent.mkdir(parents=True, exist_ok=True)
                value = (180 if is_park else 20) + frame_index
                Image.new("RGB", (8, 6), color=(value, 40, 220 - value)).save(path, format="PNG")
                frame_sha = _sha(path)
                frame = {
                    "frame_index": frame_index,
                    "path": frame_path,
                    "gcs_object": f"official/{frame_path}",
                    "generation": "1",
                    "size": path.stat().st_size,
                    "md5_base64": "unused-but-preserved",
                    "crc32c_base64": "unused-but-preserved",
                    "sha256": frame_sha,
                }
                ordered.append(frame)
                entries.append(
                    {key: frame[key] for key in ("path", "gcs_object", "generation", "size", "sha256")}
                )
                download_records.append({"local_path": frame_path, "sha256": frame_sha})
                referents.append(
                    {
                        "referent_id": f"sanpo-real:{session_id}:head:left:{frame_index:06d}",
                        "session_id": session_id,
                        "official_split": "test" if role == "test" else "train",
                        "role": role,
                        "is_park": is_park,
                        "frame_index": frame_index,
                        "path": frame_path,
                        "sha256": frame_sha,
                    }
                )
            sessions.append(
                {
                    "session_id": session_id,
                    "official_split": "test" if role == "test" else "train",
                    "role": role,
                    "test_only_no_tuning": role == "test",
                    "is_park": is_park,
                    "high_level_attributes": attributes,
                    "description_path": description_path,
                    "description_sha256": description_sha,
                    "ordered_frames": ordered,
                }
            )
    entries.sort(key=lambda row: row["path"])
    content_set_sha = bridge.json_sha256(entries)
    boundary = {
        "scientific_promotion": False,
        "official_test_used_for_model_selection_or_tuning": False,
    }
    consumer = {
        "schema": bridge.CONSUMER_SCHEMA,
        "content_entries": entries,
        "content_set_sha256": content_set_sha,
        "sessions": sessions,
        "claim_boundary": boundary,
    }
    splits = {
        "schema": bridge.SPLITS_SCHEMA,
        "official_test_tuning_allowed": False,
        "roles": roles,
    }
    referent_manifest = {"schema": bridge.REFERENTS_SCHEMA, "referents": referents}
    intake = {
        "schema": bridge.INTAKE_SCHEMA,
        "all_ok": True,
        "mode": "executed",
        "destination": str(data.resolve()),
        "integrity": {"content_set_sha256": content_set_sha},
        "claim_boundary": boundary,
        "download_records": download_records,
    }
    verification = {
        "schema": bridge.VERIFICATION_SCHEMA,
        "all_ok": True,
        "mode": "verified-existing",
        "destination": str(data.resolve()),
        "content_set_sha256": content_set_sha,
        "official_files_verified": 94,
        "claim_boundary": boundary,
    }
    files = {
        "consumer_manifest": data / "consumer_manifest.json",
        "splits": data / "splits.json",
        "referents": data / "referents.json",
        "intake_proof": proof / "intake.json",
        "verification_proof": proof / "verification.json",
    }
    for label, payload in (
        ("consumer_manifest", consumer),
        ("splits", splits),
        ("referents", referent_manifest),
        ("intake_proof", intake),
        ("verification_proof", verification),
    ):
        _write_json(files[label], payload)
    source_schemas = {
        "consumer_manifest": bridge.CONSUMER_SCHEMA,
        "splits": bridge.SPLITS_SCHEMA,
        "referents": bridge.REFERENTS_SCHEMA,
        "intake_proof": bridge.INTAKE_SCHEMA,
        "verification_proof": bridge.VERIFICATION_SCHEMA,
    }
    plan = {
        "schema": bridge.BRIDGE_PLAN_SCHEMA,
        "source": {
            "root": "data",
            "content_set_sha256": content_set_sha,
            **{
                label: {
                    "path": str(path.relative_to(root)),
                    "schema": source_schemas[label],
                    "sha256": _sha(path),
                }
                for label, path in files.items()
            },
        },
        "preprocessing": {
            "decoder": {"library": "Pillow", "version": pillow_version, "format": "PNG", "mode": "RGB"},
            "spatial": {
                "operation": "short-side-resize-then-center-crop",
                "interpolation": "bilinear",
                "target_size": 4,
            },
            "temporal_indices": list(bridge.FRAME_INDICES),
            "tensor": {
                "axis_order": ["batch", "channel", "time", "height", "width"],
                "batch_size": 1,
                "channels": 3,
                "dtype": "float32",
                "value_range": [0.0, 1.0],
                "normalization": "uint8-divide-by-255-no-channel-standardization",
            },
        },
        "evaluation_policy": {
            "development_roles": list(bridge.DEVELOPMENT_ROLES),
            "official_test_sealed_by_default": True,
            "official_test_sessions": 2,
            "official_test_tuning_allowed": False,
            "official_test_scientific_promotion": False,
            "one_shot_attempt_receipt": "proof/official_test_one_shot.json",
        },
        "claim_boundary": {"scientific_promotion": False},
    }
    plan["plan_identity_sha256"] = bridge.json_sha256(plan)
    plan_path = config / "bridge.json"
    _write_json(plan_path, plan)
    return plan_path


@pytest.fixture
def fixture_plan(tmp_path):
    return tmp_path, _build_fixture(tmp_path)


def test_verified_loader_exposes_only_development_clips(fixture_plan):
    root, plan = fixture_plan
    dataset = bridge.SanpoCustomSubstrateBridge(plan, repo_root=root)
    clips = list(dataset.iter_development())
    assert len(clips) == 8
    assert {clip.session.role for clip in clips} == {"train", "validation"}
    assert all(clip.tensor.shape == (1, 3, 8, 4, 4) for clip in clips)
    assert all(clip.tensor.dtype == torch.float32 for clip in clips)
    assert all(0.0 <= float(clip.tensor.min()) <= float(clip.tensor.max()) <= 1.0 for clip in clips)
    test_id = next(session.session_id for session in dataset.sessions if session.role == "test")
    with pytest.raises(bridge.SanpoBridgeRefused, match="official-test session is sealed"):
        dataset.load_development_session(test_id)


def test_tampered_frame_fails_before_decode(fixture_plan):
    root, plan = fixture_plan
    frame = next((root / "data").glob("sessions/*/frames/*.png"))
    frame.write_bytes(frame.read_bytes() + b"tamper")
    with pytest.raises(bridge.SanpoBridgeRefused, match="frame (size|bytes) drift"):
        bridge.SanpoCustomSubstrateBridge(plan, repo_root=root)


def test_preflight_decodes_zero_official_test_frames(fixture_plan):
    root, plan = fixture_plan
    proof = root / "proof" / "preflight.json"
    result = bridge.run_preflight(plan, proof, repo_root=root)
    assert result["all_ok"] is True
    assert result["development_decode"]["sessions_decoded"] == 8
    assert result["development_decode"]["frames_decoded"] == 64
    assert result["official_test_seal"]["frames_sha_verified"] == 16
    assert result["official_test_seal"]["frames_decoded"] == 0
    assert result["execution_boundary"]["model_forward"] is False
    assert json.loads(proof.read_text())["plan_identity_sha256"] == result["plan_identity_sha256"]


def test_plan_or_source_drift_fails_closed(fixture_plan):
    root, plan_path = fixture_plan
    plan = json.loads(plan_path.read_text())
    plan["preprocessing"]["spatial"]["target_size"] = 5
    _write_json(plan_path, plan)
    with pytest.raises(bridge.SanpoBridgeRefused, match="plan identity drift"):
        bridge.SanpoCustomSubstrateBridge(plan_path, repo_root=root)


def test_reordered_clip_is_refused_even_when_sidecar_hash_is_updated(fixture_plan):
    root, plan_path = fixture_plan
    plan = json.loads(plan_path.read_text())
    manifest_path = root / plan["source"]["consumer_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["sessions"][0]["ordered_frames"].reverse()
    _write_json(manifest_path, manifest)
    plan["source"]["consumer_manifest"]["sha256"] = _sha(manifest_path)
    plan.pop("plan_identity_sha256")
    plan["plan_identity_sha256"] = bridge.json_sha256(plan)
    _write_json(plan_path, plan)
    with pytest.raises(bridge.SanpoBridgeRefused, match="frame order drift"):
        bridge.SanpoCustomSubstrateBridge(plan_path, repo_root=root)


class _FakeModel:
    def __call__(self, clip):
        mean = clip.mean(dim=(1, 2, 3, 4))
        return SimpleNamespace(pooled_retrieval_key=torch.stack((mean + 0.1, 1.1 - mean), dim=1))


def _fake_loader(artifact_dir, *, device):
    del device
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    return SimpleNamespace(manifest=manifest, model=_FakeModel())


def test_two_stage_interface_is_development_first_and_test_is_one_shot(fixture_plan):
    root, plan = fixture_plan
    artifact = root / "artifact"
    _write_json(
        artifact / "manifest.json",
        {
            "artifact_id": "a" * 64,
            "evidence": {"independent_verifier_verdict": "promote-local-objective-lever"},
        },
    )
    selection_path = root / "proof" / "selection.json"
    selection = bridge.evaluate_development_artifact(
        plan,
        artifact,
        selection_path,
        repo_root=root,
        _loader=_fake_loader,
    )
    assert selection["official_test"]["sessions_decoded"] == 0
    assert selection["validation_diagnostic"]["n"] == 2
    attempt = root / "proof" / "official_test_one_shot.json"
    with pytest.raises(bridge.SanpoBridgeRefused, match="pass --unlock-official-test"):
        bridge.evaluate_official_test_once(
            plan,
            artifact,
            selection_path,
            unlock_official_test=False,
            repo_root=root,
            _loader=_fake_loader,
        )
    assert not attempt.exists()
    tampered_path = root / "proof" / "tampered-selection.json"
    tampered = json.loads(selection_path.read_text())
    tampered["frozen_train_centroids"]["false"]["values"][0] += 0.1
    tampered["frozen_train_centroids"]["false"]["sha256"] = bridge.tensor_sha256(
        torch.tensor(tampered["frozen_train_centroids"]["false"]["values"], dtype=torch.float32)
    )
    _write_json(tampered_path, tampered)
    with pytest.raises(bridge.SanpoBridgeRefused, match="selection identity drift"):
        bridge.evaluate_official_test_once(
            plan,
            artifact,
            tampered_path,
            unlock_official_test=True,
            repo_root=root,
            _loader=_fake_loader,
        )
    assert not attempt.exists()
    result = bridge.evaluate_official_test_once(
        plan,
        artifact,
        selection_path,
        unlock_official_test=True,
        repo_root=root,
        _loader=_fake_loader,
    )
    assert result["official_test_diagnostic"]["n"] == 2
    assert result["claim_boundary"]["scientific_promotion"] is False
    assert attempt.exists()
    with pytest.raises(bridge.SanpoBridgeRefused, match="already claimed"):
        bridge.evaluate_official_test_once(
            plan,
            artifact,
            selection_path,
            unlock_official_test=True,
            repo_root=root,
            _loader=_fake_loader,
        )
