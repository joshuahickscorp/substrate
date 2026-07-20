
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mop.experiments.e6_dense_relational import build_mechanics_fixture
from mop.substrate import vjepa21_dense_tasks as dense


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest() -> dict:
    combinations = {
        "train": [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
        "val": [(0, 1, 0), (1, 2, 1), (2, 0, 2)],
        "test": [(0, 2, 0), (1, 0, 1), (2, 1, 2)],
    }
    rows = []
    for split, entries in combinations.items():
        for factor_a, factor_b, label in entries:
            referent = f"{split}:{factor_a}:{factor_b}"
            rows.append(
                {
                    "referent": referent,
                    "tensor_path": f"/absent/{referent}.npy",
                    "tensor_file_sha256": _sha(f"file:{referent}"),
                    "tensor_sha256": _sha(f"tensor:{referent}"),
                    "tensor_shape": [3, 8, 384, 384],
                    "tensor_dtype": "float32",
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "task_label": label,
                    "split": split,
                }
            )
    source = dense._normalize_source(  # exact payload hash contract, no input bytes read
        {
            "source_kind": "natural-video",
            "natural_video": True,
            "rights_clean": True,
            "dataset_license": "fixture-license",
            "test_split_untouched": True,
            "resolution": 384,
            "encoded_frames": 8,
            "source_authority": {"dataset": "fixture", "version": "1"},
            "view_recipe": {"resize": 384, "frames": 8, "normalization": "fixture"},
        },
        rows,
    )
    manifest = {"schema": dense.INPUT_SCHEMA, "source": source, "rows": rows}
    manifest["content_sha256"] = dense._json_sha256({"source": source, "rows": rows})
    return manifest


def test_input_manifest_binds_split_annotation_source_and_view_payloads():
    manifest = _manifest()
    audit = dense.validate_input_manifest(manifest, verify_files=False)
    assert audit["mechanics_ok"] is True
    assert audit["promotion_ready"] is False
    assert audit["promotion_problems"] == ["at least 200 rows required"]

    tampered = json.loads(json.dumps(manifest))
    tampered["rows"][0]["split"] = "test"
    tampered["content_sha256"] = dense._json_sha256({"source": tampered["source"], "rows": tampered["rows"]})
    report = dense.validate_input_manifest(tampered, verify_files=False)
    assert report["mechanics_ok"] is False
    assert "split_authority_sha256" in " ".join(report["problems"])


def test_cache_plan_forces_one_ordered_manifest_for_both_arms(monkeypatch, tmp_path):
    manifest = _manifest()
    audit = dense.validate_input_manifest(manifest, verify_files=False)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(dense, "validate_input_manifest", lambda *args, **kwargs: audit)
    plan = dense.build_cache_plan(
        path,
        learned_cache=tmp_path / "learned",
        random_cache=tmp_path / "random",
        dtype="float16",
    )
    learned, random = plan["arms"]["learned"], plan["arms"]["random"]
    assert plan["serial_only"] is True
    assert plan["same_input_manifest_both_arms"] is True
    for field in (
        "input_manifest_file_sha256",
        "input_content_sha256",
        "ordered_referents",
        "ordered_input_tensor_sha256s",
        "output_shape_per_row",
        "dtype",
    ):
        assert learned[field] == random[field]
    assert learned["cache"] != random["cache"]


def test_runtime_authority_reuses_receipts_without_constructing_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no-heavy authority audit constructed a model")

    monkeypatch.setattr(dense, "load_vitb_encoder", forbidden)
    monkeypatch.setattr(dense, "build_vitb_encoder", forbidden)
    authority = dense.runtime_authority()
    assert authority["all_ok"] is True
    assert authority["checkpoint"]["full_checkpoint_rehashed_this_preflight"] is False
    assert authority["frozen_invariant"] == {
        "strict_load": True,
        "parameters": 86_833_152,
        "trainable_parameters": 0,
    }
    assert authority["verified_shapes"] == {"8": [1, 2304, 768], "64": [1, 18432, 768]}


def test_no_heavy_preflight_never_enters_model_or_forward(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no-heavy preflight entered a model path")

    monkeypatch.setattr(dense, "load_vitb_encoder", forbidden)
    monkeypatch.setattr(dense, "build_vitb_encoder", forbidden)
    receipt = dense.no_heavy_preflight()
    assert receipt["all_ok"] is True
    assert receipt["model_constructed"] is False
    assert receipt["checkpoint_tensor_bytes_read"] is False
    assert receipt["forward_executed"] is False
    assert receipt["gates"]["implementation_ready"] is True
    assert receipt["gates"]["input_manifest_ready"] is False
    assert receipt["scientific_promotion"] is False


def test_dr14_dense_views_are_nested_deterministic_and_fixture_is_explicit(tmp_path):
    fixture = build_mechanics_fixture(tmp_path)
    learned = Path(fixture["stores"]["learned"])
    with pytest.raises(dense.DenseTaskError, match="official dense cache"):
        dense.build_dr14_dense_views(learned, fractions=(0.0, 0.25, 0.5), group_width=2)

    first = dense.build_dr14_dense_views(
        learned,
        fractions=(0.0, 0.25, 0.5),
        group_width=2,
        seed=7,
        strict_run_identity=False,
    )
    second = dense.build_dr14_dense_views(
        learned,
        fractions=(0.0, 0.25, 0.5),
        group_width=2,
        seed=7,
        strict_run_identity=False,
    )
    assert first["shared_corrupted_tensor_for_both_arms"] is True
    assert first["strict_run_identity"] is False
    assert first["masks"] == second["masks"]
    quarter = set(first["masks"]["0.250000"]["dropped_groups"])
    half = set(first["masks"]["0.500000"]["dropped_groups"])
    assert quarter < half
    assert first["masks"]["0.500000"]["view_sha256"] == second["masks"]["0.500000"]["view_sha256"]
