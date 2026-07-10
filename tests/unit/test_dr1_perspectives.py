import json

import numpy as np
import pytest
import scripts.studio.dr1_curate_bound_video as dr1

from mop.studio.dr1_perspectives import build_dr1_perspective_receipt, write_dr1_perspective_receipt
from mop.substrate import LatentStore


def _store_with_sidecars(tmp_path):
    store = LatentStore.create(tmp_path, "merged", (3,), capacity=2, key_dim=3)
    latents = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    store.write_batch(0, latents, latents)
    store.finalize()
    stems = ["dog-running-0", "cat-sitting-0"]
    (store.root / "clip_stems.json").write_text(json.dumps(stems, indent=2))
    (store.root / "clip_cells.json").write_text(
        json.dumps(
            {
                "dog-running-0": "dog-running",
                "cat-sitting-0": "cat-sitting",
            },
            indent=2,
        )
    )
    return store


def test_dr1_perspective_receipt_builds_aligned_vision_caption_matrix(tmp_path):
    store = _store_with_sidecars(tmp_path)
    receipt = build_dr1_perspective_receipt(
        store.root,
        {
            "dog-running-0": "a dog is running",
            "cat-sitting-0": "a cat is sitting",
        },
        factors=("object", "action"),
    )
    assert receipt["ok"] is True
    assert receipt["n_referents"] == 2
    assert receipt["tags"] == ["caption_text", "vision_vjepa2"]
    assert receipt["audit"]["feature_dims"]["vision_vjepa2"] == 3
    assert receipt["audit"]["feature_dims"]["caption_text"] == 256
    assert receipt["audit"]["schema"] == "mop-form-matrix/v1"
    assert receipt["audit"]["kinds"] == {"text": ["caption_text"], "vision": ["vision_vjepa2"]}
    assert receipt["audit"]["modalities"] == {
        "caption_text": "language",
        "vision_vjepa2": "vision",
    }
    assert receipt["arms"]["caption_text"]["kind"] == "text"
    assert receipt["arms"]["caption_text"]["modality"] == "language"
    assert receipt["arms"]["caption_text"]["factors"] == ("action", "object")
    assert receipt["factor_counts"]["object"] == {"cat": 1, "dog": 1}


def test_write_dr1_perspective_receipt_persists_json(tmp_path):
    store = _store_with_sidecars(tmp_path)
    receipt = write_dr1_perspective_receipt(
        store.root,
        {
            "dog-running-0": "a dog is running",
            "cat-sitting-0": "a cat is sitting",
        },
        factors=("object", "action"),
    )
    path = store.root / "perspective_matrix_receipt.json"
    assert receipt["path"] == str(path)
    assert json.loads(path.read_text())["schema"] == "mop-dr1-perspective-matrix-receipt/v1"


def test_dr1_perspective_receipt_refuses_missing_caption(tmp_path):
    store = _store_with_sidecars(tmp_path)
    with pytest.raises(ValueError, match="captions missing"):
        build_dr1_perspective_receipt(
            store.root,
            {"dog-running-0": "a dog is running"},
            factors=("object", "action"),
        )


def test_dr1_merge_writes_blocked_perspective_receipt_without_root_store(tmp_path, monkeypatch):
    monkeypatch.setattr(dr1, "REPO_ROOT", tmp_path)
    root = tmp_path / "data" / "cache" / "dr1_smoke"
    leg = root / "leg_0_2"
    leg.mkdir(parents=True)
    (leg / "cells.json").write_text(
        json.dumps(
            {
                "leg": [0, 2],
                "n_encoded": 2,
                "backend": "vjepa_hf",
                "factors": ["object", "action"],
                "clip_stems": ["dog-running-0", "cat-sitting-0"],
                "clip_cells": ["dog-running", "cat-sitting"],
            }
        )
    )
    manifest = dr1.merge_shards("dr1_smoke", source=str(tmp_path / "source"), factors=("object", "action"))
    receipt_path = root / "perspective_matrix_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert manifest["perspective_receipt"] == str(receipt_path)
    assert receipt["ok"] is False
    assert "merged LatentStore" in receipt["blocked_reason"]


def test_dr1_a6_guard_main_writes_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(dr1, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dr1, "assert_studio_ram", lambda: 128.0)

    source = tmp_path / "source"
    source.mkdir()
    (source / "captions.json").write_text(json.dumps({"clip0": "a dog is running"}))

    def fake_guard(store_dir, captions_map, factors):
        return {
            "guard": "a6_residual_alignment (cross-modal caption<->vision nuisance control)",
            "store": str(store_dir),
            "conditions": {"minus_factors": {"survives": True}},
            "decisive_condition": "minus_factors",
            "verdict": "GENUINE-SHARED-STRUCTURE",
        }

    monkeypatch.setattr(dr1, "a6_residual_guard", fake_guard)

    rc = dr1.main(
        [
            "--a6-guard",
            "--source",
            str(source),
            "--name",
            "dr1_smoke",
            "--factors",
            "object,action",
        ]
    )

    path = tmp_path / "data" / "cache" / "dr1_smoke" / "a6_residual_guard.json"
    assert rc == 0
    assert json.loads(path.read_text())["path"] == str(path)
