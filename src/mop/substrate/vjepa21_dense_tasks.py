"""Official V-JEPA 2.1 ViT-B dense cache and matched-control task seam.

Cheap preflight and manifest operations never construct a model. ``encode_dense_cache`` is the
only heavy entrypoint: it is explicit, serial, resumable, and consumes one immutable input tensor
manifest for either the learned or exact-architecture random arm. E6 and DR14 therefore join on
ordered referents and tensor hashes instead of filenames or post-encoder feature similarity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import REPO_ROOT
from .cache_manifest import (
    ENCODER_RECEIPT_SCHEMA,
    RANDOM_INIT_RECEIPT_SCHEMA,
    write_cache_manifest,
)
from .cache_tools import validate_cache
from .encoder import module_state_sha256
from .latent_store import LatentStore
from .vjepa21_official import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_REPOSITORY_DIR,
    OFFICIAL_REPOSITORY_COMMIT,
    REPOSITORY_ARTIFACTS,
    VITB,
    build_vitb_encoder,
    checkpoint_receipt_path,
    expected_dense_tokens,
    load_vitb_encoder,
    sha256_file,
    validate_checkpoint_receipt,
    validate_repository,
)

INPUT_SCHEMA = "mop-vjepa21-dense-input-manifest/v1"
PLAN_SCHEMA = "mop-vjepa21-dense-cache-plan/v1"
PREFLIGHT_SCHEMA = "mop-vjepa21-dense-task-preflight/v1"
PROGRESS_SCHEMA = "mop-vjepa21-dense-cache-progress/v1"
RUN_SCHEMA = "mop-vjepa21-dense-cache-run/v1"
DR14_VIEWS_SCHEMA = "mop-dr14-dense-drop-views/v1"
E6_SOURCE_SCHEMA = "mop-e6-dense-source/v1"
MODEL_ID = "official-pytorch-only-vjepa21-vitb"
MIN_FREE_DISK_BYTES = 40_000_000_000
DEFAULT_TASK_CONFIG = REPO_ROOT / "configs/experiment/e6_dense_cache.yaml"
DEFAULT_LOAD_RECEIPT = REPO_ROOT / "proof/VJEPA21_VITB_LOAD.json"
DEFAULT_FORWARD_8F_RECEIPT = REPO_ROOT / "proof/VJEPA21_VITB_FORWARD.json"
DEFAULT_FORWARD_64F_RECEIPT = REPO_ROOT / "proof/VJEPA21_VITB_FORWARD_64F.json"
VISION_SOURCE = next(
    row for row in REPOSITORY_ARTIFACTS if row["path"] == "app/vjepa_2_1/models/vision_transformer.py"
)


class DenseTaskError(RuntimeError):
    """A runtime authority, input identity, cache, or matched-control contract failed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path | str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseTaskError(f"cannot read JSON {path}: {exc}") from exc


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _split_authority_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"referent": row.get("referent"), "split": row.get("split")} for row in rows]


def _annotation_authority_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "referent": row.get("referent"),
            "factor_a": row.get("factor_a"),
            "factor_b": row.get("factor_b"),
            "task_label": row.get("task_label"),
        }
        for row in rows
    ]


def _normalize_source(source: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind source declarations to their canonical payloads and immutable row annotations."""

    normalized = json.loads(json.dumps(source))
    source_authority = normalized.get("source_authority")
    view_recipe = normalized.get("view_recipe")
    if not isinstance(source_authority, dict):
        raise DenseTaskError("source.source_authority must be a mapping")
    if not isinstance(view_recipe, dict):
        raise DenseTaskError("source.view_recipe must be a mapping")
    expected = {
        "source_authority_sha256": _json_sha256(source_authority),
        "view_recipe_sha256": _json_sha256(view_recipe),
        "split_authority_sha256": _json_sha256(_split_authority_payload(rows)),
        "annotation_authority_sha256": _json_sha256(_annotation_authority_payload(rows)),
    }
    for field, digest in expected.items():
        declared = normalized.get(field)
        if declared is not None and declared != digest:
            raise DenseTaskError(f"source {field} does not match its canonical payload")
        normalized[field] = digest
    return normalized


def _config() -> dict[str, Any]:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if not isinstance(raw, dict):
        raise DenseTaskError(f"{DEFAULT_CONFIG} must contain a mapping")
    return raw


def _probe_receipt(path: Path, *, frames: int | None) -> tuple[dict[str, Any], list[str]]:
    receipt = _read_json(path)
    problems: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("schema") != "mop-vjepa21-official-probe/v1":
        return {}, [f"{path}: wrong probe schema"]
    child_value = receipt.get("child")
    authority_value = receipt.get("authority")
    child: dict[str, Any] = child_value if isinstance(child_value, dict) else {}
    authority: dict[str, Any] = authority_value if isinstance(authority_value, dict) else {}
    probe_value = receipt.get("probe")
    probe: dict[str, Any] = probe_value if isinstance(probe_value, dict) else {}
    if receipt.get("status") != "passed" or receipt.get("hardware_limit_reached") is not False:
        problems.append(f"{path}: runtime probe did not pass")
    if child.get("strict_load") is not True or child.get("trainable_parameters") != 0:
        problems.append(f"{path}: frozen strict-load invariant failed")
    if child.get("parameters") != 86_833_152:
        problems.append(f"{path}: ViT-B parameter count mismatch")
    if child.get("checkpoint_key") != VITB["checkpoint_key"]:
        problems.append(f"{path}: checkpoint key mismatch")
    if child.get("repository_commit") != OFFICIAL_REPOSITORY_COMMIT:
        problems.append(f"{path}: repository commit mismatch")
    if authority.get("repository_commit") != OFFICIAL_REPOSITORY_COMMIT:
        problems.append(f"{path}: authority repository commit mismatch")
    if authority.get("repository_validation_ok") is not True:
        problems.append(f"{path}: repository validation was not clean")
    if authority.get("checkpoint_sha256") != child.get("checkpoint_sha256"):
        problems.append(f"{path}: child and authority checkpoint hashes differ")
    if frames is not None:
        expected = [1, expected_dense_tokens(frames), int(VITB["embed_dim"])]
        if child.get("mode") != "forward" or child.get("output_shape") != expected:
            problems.append(f"{path}: expected dense shape {expected}")
        if child.get("output_finite") is not True or child.get("shape_matches") is not True:
            problems.append(f"{path}: forward finiteness/shape gate failed")
        if probe.get("frames") != frames:
            problems.append(f"{path}: probe frame count mismatch")
    elif child.get("mode") != "load" or probe.get("mode") != "load":
        problems.append(f"{path}: expected a load-only receipt")
    if not _valid_sha256(authority.get("checkpoint_sha256")):
        problems.append(f"{path}: checkpoint authority hash missing")
    return receipt, problems


def runtime_authority(
    *,
    repository: Path | str = DEFAULT_REPOSITORY_DIR,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    load_receipt: Path | str = DEFAULT_LOAD_RECEIPT,
    forward_8f_receipt: Path | str = DEFAULT_FORWARD_8F_RECEIPT,
    forward_64f_receipt: Path | str = DEFAULT_FORWARD_64F_RECEIPT,
) -> dict[str, Any]:
    """Validate retained bytes and prior probes without constructing or forwarding a model."""

    repository_path = Path(repository).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    repo = validate_repository(repository_path)
    checkpoint_validation = validate_checkpoint_receipt(checkpoint_path, rehash=False)
    load, load_problems = _probe_receipt(Path(load_receipt), frames=None)
    forward_8f, forward_8f_problems = _probe_receipt(Path(forward_8f_receipt), frames=8)
    forward_64f, forward_64f_problems = _probe_receipt(Path(forward_64f_receipt), frames=64)
    config = _config()
    problems = [
        *repo.get("problems", []),
        *checkpoint_validation.get("problems", []),
        *load_problems,
        *forward_8f_problems,
        *forward_64f_problems,
    ]
    authorities = [receipt.get("authority") or {} for receipt in (load, forward_8f, forward_64f)]
    checkpoint_hashes = {str(value.get("checkpoint_sha256") or "") for value in authorities}
    receipt_hashes = {str(value.get("checkpoint_receipt_sha256") or "") for value in authorities}
    if len(checkpoint_hashes) != 1:
        problems.append("runtime receipts do not bind one checkpoint SHA256")
    if len(receipt_hashes) != 1:
        problems.append("runtime receipts do not bind one checkpoint-receipt SHA256")
    checkpoint_sha = next(iter(checkpoint_hashes), "")
    local_receipt_path = checkpoint_receipt_path(checkpoint_path)
    if not local_receipt_path.is_file():
        problems.append("retained checkpoint receipt is missing")
    elif sha256_file(local_receipt_path) not in receipt_hashes:
        problems.append("retained checkpoint receipt hash differs from runtime authority")
    if checkpoint_validation.get("receipt", {}).get("sha256") != checkpoint_sha:
        problems.append("retained checkpoint metadata differs from runtime checkpoint SHA256")
    expected_config = {
        "available": True,
        "availability_state": "local_hash_strict_load_and_8f_64f_forward_verified",
        "checkpoint_sha256": checkpoint_sha,
        "cache_first_only": True,
        "frozen": True,
        "dense": True,
        "pool": "none",
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            problems.append(f"encoder config {field}={config.get(field)!r}, expected {expected!r}")
    return {
        "schema": "mop-vjepa21-vitb-runtime-authority/v1",
        "created_at": _utc_now(),
        "repository": {
            "path": str(repository_path),
            "commit": repo.get("commit"),
            "all_ok": repo.get("all_ok"),
            "vision_transformer": dict(VISION_SOURCE),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else None,
            "sha256": checkpoint_sha,
            "receipt_path": str(local_receipt_path),
            "receipt_sha256": sha256_file(local_receipt_path) if local_receipt_path.is_file() else None,
            "full_checkpoint_rehashed_this_preflight": False,
            "prior_probe_authority_reused": True,
        },
        "receipts": {
            "load": {"path": str(Path(load_receipt).resolve()), "sha256": sha256_file(Path(load_receipt))},
            "forward_8f": {
                "path": str(Path(forward_8f_receipt).resolve()),
                "sha256": sha256_file(Path(forward_8f_receipt)),
            },
            "forward_64f": {
                "path": str(Path(forward_64f_receipt).resolve()),
                "sha256": sha256_file(Path(forward_64f_receipt)),
            },
        },
        "frozen_invariant": {
            "strict_load": (load.get("child") or {}).get("strict_load"),
            "parameters": (load.get("child") or {}).get("parameters"),
            "trainable_parameters": (load.get("child") or {}).get("trainable_parameters"),
        },
        "verified_shapes": {
            "8": (forward_8f.get("child") or {}).get("output_shape"),
            "64": (forward_64f.get("child") or {}).get("output_shape"),
        },
        "config": config,
        "problems": problems,
        "all_ok": not problems,
    }


def build_input_manifest(
    records: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    output: Path | str | None = None,
) -> dict[str, Any]:
    """Hash exact preprocessed tensors and freeze ordered factors, labels, and splits."""

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        path = Path(str(record.get("tensor_path") or "")).resolve()
        try:
            tensor = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DenseTaskError(f"input row {index} cannot load {path}: {exc}") from exc
        if tensor.dtype != np.float32 or tensor.ndim != 4:
            raise DenseTaskError(
                f"input row {index} must be float32 [C,T,H,W], got {tensor.dtype} {tensor.shape}"
            )
        try:
            task_label = int(record["task_label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DenseTaskError(f"input row {index} needs an integer task_label") from exc
        normalized.append(
            {
                "referent": str(record.get("referent") or ""),
                "tensor_path": str(path),
                "tensor_file_sha256": sha256_file(path),
                "tensor_sha256": tensor_sha256(tensor),
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "factor_a": record.get("factor_a"),
                "factor_b": record.get("factor_b"),
                "task_label": task_label,
                "split": str(record.get("split") or ""),
            }
        )
    normalized_source = _normalize_source(source, normalized)
    manifest = {
        "schema": INPUT_SCHEMA,
        "created_at": _utc_now(),
        "source": normalized_source,
        "rows": normalized,
    }
    manifest["content_sha256"] = _json_sha256({"source": manifest["source"], "rows": normalized})
    audit = validate_input_manifest(manifest, verify_files=True)
    if not audit["mechanics_ok"]:
        raise DenseTaskError("input manifest invalid: " + "; ".join(audit["problems"]))
    if output is not None:
        _atomic_json(Path(output), manifest)
    return manifest


def validate_input_manifest(
    manifest_or_path: dict[str, Any] | Path | str,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    manifest = (
        _read_json(manifest_or_path)
        if isinstance(manifest_or_path, (str, Path))
        else json.loads(json.dumps(manifest_or_path))
    )
    problems: list[str] = []
    if not isinstance(manifest, dict) or manifest.get("schema") != INPUT_SCHEMA:
        return {"mechanics_ok": False, "problems": [f"input schema must be {INPUT_SCHEMA}"]}
    source_value = manifest.get("source")
    rows_value = manifest.get("rows")
    source: dict[str, Any] = source_value if isinstance(source_value, dict) else {}
    rows: list[Any] = rows_value if isinstance(rows_value, list) else []
    if not rows:
        problems.append("input manifest rows must be nonempty")
    row_maps = [row for row in rows if isinstance(row, dict)]
    if len(row_maps) != len(rows):
        problems.append("every input row must be a mapping")
    referents = [str(row.get("referent") or "") for row in row_maps]
    tensor_hashes = [str(row.get("tensor_sha256") or "") for row in row_maps]
    if len(referents) != len(rows) or any(not value for value in referents):
        problems.append("every row needs a nonempty referent")
    if len(set(referents)) != len(referents):
        problems.append("referents must be unique")
    if len(tensor_hashes) != len(rows) or not all(_valid_sha256(value) for value in tensor_hashes):
        problems.append("every row needs a tensor SHA256")
    if len(set(tensor_hashes)) != len(tensor_hashes):
        problems.append("input tensors must be unique across referents")
    split_names = {str(row.get("split") or "") for row in row_maps}
    if split_names != {"train", "val", "test"}:
        problems.append("row splits must contain exactly train, val, and test")
    if any(not any(row.get("split") == split for row in row_maps) for split in ("train", "val", "test")):
        problems.append("train, val, and test must each be nonempty")
    shapes: set[tuple[int, ...]] = set()
    labels: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index} must be a mapping")
            continue
        try:
            shape = tuple(int(value) for value in row.get("tensor_shape", []))
            label_value = row.get("task_label")
            if label_value is None:
                raise ValueError("missing task label")
            label = int(label_value)
        except (TypeError, ValueError):
            problems.append(f"row {index} has invalid shape or task label")
            continue
        shapes.add(shape)
        labels.append(label)
        if row.get("tensor_dtype") != "float32":
            problems.append(f"row {index} tensor dtype must be float32")
        if not str(row.get("tensor_path") or ""):
            problems.append(f"row {index} tensor path must be nonempty")
        if not _valid_sha256(row.get("tensor_file_sha256")):
            problems.append(f"row {index} tensor file hash must be a SHA256")
        if row.get("factor_a") is None or row.get("factor_b") is None:
            problems.append(f"row {index} needs both factors")
        if verify_files:
            path = Path(str(row.get("tensor_path") or ""))
            if not path.is_file():
                problems.append(f"row {index} tensor file missing: {path}")
                continue
            try:
                tensor = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                problems.append(f"row {index} tensor load failed: {exc}")
                continue
            if list(tensor.shape) != list(shape) or str(tensor.dtype) != row.get("tensor_dtype"):
                problems.append(f"row {index} tensor shape/dtype drift")
            if tensor_sha256(tensor) != row.get("tensor_sha256"):
                problems.append(f"row {index} tensor content hash drift")
            if sha256_file(path) != row.get("tensor_file_sha256"):
                problems.append(f"row {index} tensor file hash drift")
    if len(shapes) != 1:
        problems.append("all input tensors must have one shape")
        shape = ()
    else:
        shape = next(iter(shapes))
    if len(shape) != 4 or shape[0] != 3 or shape[2:] != (384, 384):
        problems.append(f"input shape must be [3,T,384,384], got {shape}")
    frames = shape[1] if len(shape) == 4 else 0
    if frames not in (8, 16, 64):
        problems.append("input frames must be 8, 16, or 64")
    combination_sets = {
        split: {
            (str(row.get("factor_a")), str(row.get("factor_b")))
            for row in row_maps
            if row.get("split") == split
        }
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if combination_sets[left] & combination_sets[right]:
            problems.append(f"factor combinations overlap between {left} and {right}")
    try:
        train_labels = {int(row["task_label"]) for row in row_maps if row.get("split") == "train"}
    except (KeyError, TypeError, ValueError):
        train_labels = set()
        problems.append("training rows contain an invalid task label")
    if labels and train_labels != set(labels):
        problems.append("training split must contain every task label")
    if labels and set(labels) != set(range(max(labels) + 1)):
        problems.append("task labels must be contiguous nonnegative integers starting at zero")
    for factor in ("factor_a", "factor_b"):
        all_levels = {json.dumps(row.get(factor), sort_keys=True, default=str) for row in row_maps}
        train_levels = {
            json.dumps(row.get(factor), sort_keys=True, default=str)
            for row in row_maps
            if row.get("split") == "train"
        }
        if all_levels and train_levels != all_levels:
            problems.append(f"training split must contain every {factor} level")
    authority_payloads = {
        "source_authority_sha256": source.get("source_authority"),
        "view_recipe_sha256": source.get("view_recipe"),
        "split_authority_sha256": _split_authority_payload(row_maps),
        "annotation_authority_sha256": _annotation_authority_payload(row_maps),
    }
    for field, payload in authority_payloads.items():
        if field in {"source_authority_sha256", "view_recipe_sha256"} and not isinstance(payload, dict):
            problems.append(f"source {field.removesuffix('_sha256')} must be a mapping")
            continue
        if not _valid_sha256(source.get(field)):
            problems.append(f"source {field} must be a SHA256")
        elif source.get(field) != _json_sha256(payload):
            problems.append(f"source {field} does not match its canonical payload")
    if source.get("resolution") != 384 or source.get("encoded_frames") != frames:
        problems.append("source resolution/frames must match exact input tensors")
    expected_content = _json_sha256({"source": source, "rows": rows})
    if manifest.get("content_sha256") != expected_content:
        problems.append("input manifest content_sha256 mismatch")
    promotion_problems: list[str] = []
    if source.get("source_kind") != "natural-video" or source.get("natural_video") is not True:
        promotion_problems.append("natural-video source required")
    if source.get("rights_clean") is not True or not str(source.get("dataset_license") or ""):
        promotion_problems.append("rights-clean source and license required")
    if source.get("test_split_untouched") is not True:
        promotion_problems.append("test split must remain untouched")
    if len(rows) < 200:
        promotion_problems.append("at least 200 rows required")
    return {
        "schema": "mop-vjepa21-dense-input-audit/v1",
        "count": len(rows),
        "shape": list(shape),
        "frames": frames,
        "referents": referents,
        "tensor_sha256s": tensor_hashes,
        "splits": {
            split: [index for index, row in enumerate(row_maps) if row.get("split") == split]
            for split in ("train", "val", "test")
        },
        "combination_splits": {
            split: [list(pair) for pair in sorted(values)] for split, values in combination_sets.items()
        },
        "factors": {
            "factor_a": [row.get("factor_a") for row in row_maps],
            "factor_b": [row.get("factor_b") for row in row_maps],
        },
        "task_labels": labels,
        "source": source,
        "content_sha256": manifest.get("content_sha256"),
        "problems": problems,
        "mechanics_ok": not problems,
        "promotion_problems": promotion_problems,
        "promotion_ready": not problems and not promotion_problems,
    }


def e6_source_block(audit: dict[str, Any]) -> dict[str, Any]:
    source = audit["source"]
    return {
        "schema": E6_SOURCE_SCHEMA,
        "source_kind": source.get("source_kind"),
        "natural_video": source.get("natural_video"),
        "rights_clean": source.get("rights_clean"),
        "dataset_license": source.get("dataset_license"),
        "source_authority_sha256": source.get("source_authority_sha256"),
        "content_set_sha256": audit.get("content_sha256"),
        "split_authority_sha256": source.get("split_authority_sha256"),
        "annotation_authority_sha256": source.get("annotation_authority_sha256"),
        "view_recipe_sha256": source.get("view_recipe_sha256"),
        "encoded_frames": audit.get("frames"),
        "resolution": source.get("resolution"),
        "byte_identical_inputs_across_arms": True,
        "test_split_untouched": source.get("test_split_untouched"),
        "input_tensor_sha256_by_referent": [
            {"referent": referent, "sha256": digest}
            for referent, digest in zip(audit["referents"], audit["tensor_sha256s"], strict=True)
        ],
    }


def build_cache_plan(
    input_manifest: Path | str,
    *,
    learned_cache: Path | str,
    random_cache: Path | str,
    random_seed: int = 20260710,
    dtype: str = "float32",
) -> dict[str, Any]:
    if dtype not in {"float16", "float32"}:
        raise ValueError("dense cache dtype must be float16 or float32")
    manifest_path = Path(input_manifest).resolve()
    audit = validate_input_manifest(manifest_path, verify_files=True)
    if not audit["mechanics_ok"]:
        raise DenseTaskError("input manifest invalid: " + "; ".join(audit["problems"]))
    learned_path = Path(learned_cache).resolve()
    random_path = Path(random_cache).resolve()
    if learned_path == random_path:
        raise DenseTaskError("learned and random caches must have different destinations")
    tokens = expected_dense_tokens(int(audit["frames"]))
    bytes_per_value = np.dtype(dtype).itemsize
    per_arm = audit["count"] * tokens * int(VITB["embed_dim"]) * bytes_per_value
    common = {
        "input_manifest": str(manifest_path),
        "input_manifest_file_sha256": sha256_file(manifest_path),
        "input_content_sha256": audit["content_sha256"],
        "ordered_referents": audit["referents"],
        "ordered_input_tensor_sha256s": audit["tensor_sha256s"],
        "output_shape_per_row": [tokens, int(VITB["embed_dim"])],
        "dtype": dtype,
        "estimated_latent_bytes": per_arm,
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "serial_only": True,
        "same_input_manifest_both_arms": True,
        "arms": {
            "learned": {**common, "cache": str(learned_path), "weights_real": True},
            "random": {
                **common,
                "cache": str(random_path),
                "weights_real": False,
                "random_seed": int(random_seed),
            },
        },
        "total_estimated_latent_bytes": 2 * per_arm,
        "consumers": {
            "e6": "bounded token bins versus learned-flat, token-shuffle, and random-init controls",
            "dr14": "shared deterministic dropped-channel views versus matched single-pass control",
        },
        "scientific_promotion": False,
        "promotion_problems": audit["promotion_problems"],
    }
    plan["plan_sha256"] = _json_sha256(plan)
    return plan


def encoder_config_for_arm(arm: str, authority: dict[str, Any], random_seed: int) -> dict[str, Any]:
    if arm not in {"learned", "random"}:
        raise ValueError("arm must be learned or random")
    random = arm == "random"
    return {
        "name": f"vjepa21_vitb_{arm}",
        "arch": "vit_base",
        "hf_id": MODEL_ID,
        "revision": OFFICIAL_REPOSITORY_COMMIT if random else authority["checkpoint"]["sha256"],
        "embed_dim": 768,
        "patch_size": 16,
        "tubelet": 2,
        "frames_per_clip": 64,
        "resolution": 384,
        "dense": True,
        "pool": "none",
        "official_repo_commit": OFFICIAL_REPOSITORY_COMMIT,
        "hub_entrypoint": VITB["hub_entrypoint"],
        "actual_backend": "vjepa_official_random_init" if random else "vjepa_official",
        "random_init": random,
        "random_init_seed": int(random_seed) if random else None,
        "prefer_real": not random,
        "require_real": not random,
        "frozen": True,
    }


def learned_encoder_receipt(authority: dict[str, Any]) -> dict[str, Any]:
    checkpoint = authority["checkpoint"]
    return {
        "schema": ENCODER_RECEIPT_SCHEMA,
        "weights_real": True,
        "backend": "vjepa_official",
        "model_id": MODEL_ID,
        "revision": checkpoint["sha256"],
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "checkpoint_etag": VITB["checkpoint_etag"],
        "checkpoint_version_id": VITB["checkpoint_version_id"],
        "checkpoint_bytes": VITB["checkpoint_content_length"],
        "checkpoint_authority_receipt_sha256": checkpoint["receipt_sha256"],
        "frozen": True,
        "trainable_parameters": 0,
        "files": [
            {
                "path": checkpoint["path"],
                "bytes": checkpoint["bytes"],
                "sha256": checkpoint["sha256"],
            }
        ],
    }


def random_initialization_receipt(encoder: Any, repository: Path | str, seed: int) -> dict[str, Any]:
    source = Path(repository).resolve() / str(VISION_SOURCE["path"])
    state = encoder.state_dict()
    return {
        "schema": RANDOM_INIT_RECEIPT_SCHEMA,
        "weights_real": False,
        "backend": "vjepa_official_random_init",
        "model_id": MODEL_ID,
        "revision": OFFICIAL_REPOSITORY_COMMIT,
        "seed": int(seed),
        "parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "state_dict_tensors": len(state),
        "state_dict_sha256": module_state_sha256(encoder),
        "model_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "architecture_files": [
            {
                "path": VISION_SOURCE["path"],
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ],
    }


def _open_writable(root: Path) -> LatentStore:
    opened = LatentStore.open(root)
    return LatentStore(root, opened.meta, mode="r+")


def finalize_dense_cache(
    cache: Path | str,
    *,
    arm: str,
    input_manifest: Path | str,
    authority: dict[str, Any],
    random_seed: int = 20260710,
    initialization_receipt: dict[str, Any] | None = None,
    output_row_sha256s: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(cache).resolve()
    audit = validate_input_manifest(input_manifest, verify_files=True)
    if not audit["mechanics_ok"]:
        raise DenseTaskError("input manifest invalid: " + "; ".join(audit["problems"]))
    if not authority.get("all_ok"):
        raise DenseTaskError("ViT-B runtime authority is not clean")
    store = LatentStore.open(root)
    expected_shape = (expected_dense_tokens(int(audit["frames"])), int(VITB["embed_dim"]))
    if len(store) != audit["count"] or tuple(store.meta.feat_shape) != expected_shape:
        raise DenseTaskError(
            f"encoded store shape/count {(len(store), store.meta.feat_shape)} != "
            f"{(audit['count'], list(expected_shape))}"
        )
    labels = store.labels()
    if labels is None or labels.tolist() != audit["task_labels"]:
        raise DenseTaskError("encoded store labels do not match immutable input manifest")
    actual_row_hashes: list[str] = []
    for index in range(len(store)):
        row = store.latents(index).numpy()
        if not np.isfinite(row).all():
            raise DenseTaskError(f"encoded dense row {index} is non-finite")
        actual_row_hashes.append(tensor_sha256(row))
    if output_row_sha256s is not None and output_row_sha256s != actual_row_hashes:
        raise DenseTaskError("encoded output row hashes differ from progress receipt")
    source = e6_source_block(audit)
    config = encoder_config_for_arm(arm, authority, random_seed)
    factor_metadata = {
        "factor_a_name": "factor_a",
        "factor_b_name": "factor_b",
        "heldout_combination_policy": "immutable disjoint combinations from input manifest",
        "combination_splits": audit["combination_splits"],
        "annotation_authority_sha256": audit["source"].get("annotation_authority_sha256"),
        "input_manifest_content_sha256": audit["content_sha256"],
    }
    run_receipt = {
        "schema": RUN_SCHEMA,
        "created_at": _utc_now(),
        "arm": arm,
        "input_manifest": str(Path(input_manifest).resolve()),
        "input_manifest_file_sha256": sha256_file(Path(input_manifest)),
        "input_manifest_content_sha256": audit["content_sha256"],
        "ordered_input_tensor_sha256s": audit["tensor_sha256s"],
        "ordered_output_tensor_sha256s": actual_row_hashes,
        "e6_source": source,
        "frozen_encoder": True,
        "scientific_promotion": False,
    }
    _atomic_json(root / "run_receipt.json", run_receipt)
    encoder_receipt = None
    if arm == "learned":
        encoder_receipt = learned_encoder_receipt(authority)
    elif arm == "random":
        if initialization_receipt is None:
            raise DenseTaskError("random cache finalization requires initialization receipt")
        _atomic_json(root / "initialization_receipt.json", initialization_receipt)
    else:
        raise ValueError("arm must be learned or random")
    write_cache_manifest(
        root,
        encoder_config=config,
        encoder_receipt=encoder_receipt,
        factors=audit["factors"],
        factor_metadata=factor_metadata,
        splits=audit["splits"],
        referents=audit["referents"],
        form_kind="vision",
        form_objective="inherited-frozen" if arm == "learned" else "random-control",
        referent_scheme=(
            "natural-video-clip-id"
            if audit["source"].get("source_kind") == "natural-video"
            else "programmatic-fixture-id"
        ),
        full_hash_arrays=False,
    )
    problems = validate_cache(root, citable=True)
    return {
        "schema": "mop-vjepa21-dense-cache-finalization/v1",
        "cache": str(root),
        "arm": arm,
        "count": len(store),
        "feat_shape": list(store.meta.feat_shape),
        "input_content_sha256": audit["content_sha256"],
        "output_rows_sha256": _json_sha256(actual_row_hashes),
        "problems": problems,
        "all_ok": not problems,
        "scientific_promotion": False,
    }


def encode_dense_cache(
    input_manifest: Path | str,
    cache: Path | str,
    *,
    arm: str,
    device: str = "cpu",
    dtype: str = "float32",
    random_seed: int = 20260710,
    repository: Path | str = DEFAULT_REPOSITORY_DIR,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    """Explicit heavy encoder path. Preflight and imports never call this function."""

    if arm not in {"learned", "random"}:
        raise ValueError("arm must be learned or random")
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if dtype not in {"float16", "float32"}:
        raise ValueError("dense cache dtype must be float16 or float32")
    manifest_path = Path(input_manifest).resolve()
    audit = validate_input_manifest(manifest_path, verify_files=True)
    if not audit["mechanics_ok"]:
        raise DenseTaskError("input manifest invalid: " + "; ".join(audit["problems"]))
    authority = runtime_authority(repository=repository, checkpoint=checkpoint)
    if not authority["all_ok"]:
        raise DenseTaskError("runtime authority invalid: " + "; ".join(authority["problems"]))
    tokens = expected_dense_tokens(int(audit["frames"]))
    estimated_bytes = audit["count"] * tokens * int(VITB["embed_dim"]) * np.dtype(dtype).itemsize
    root = Path(cache).resolve()
    disk_root = root.parent
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    if shutil.disk_usage(disk_root).free - estimated_bytes < MIN_FREE_DISK_BYTES:
        raise DenseTaskError("dense cache allocation would cross the 40 GB free-disk floor")

    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise DenseTaskError("MPS requested but unavailable")
    initialization_receipt = None
    if arm == "learned":
        encoder = load_vitb_encoder(repository, checkpoint)
    else:
        encoder = build_vitb_encoder(repository, random_seed=random_seed)
        initialization_receipt = random_initialization_receipt(encoder, repository, random_seed)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise DenseTaskError("frozen encoder invariant failed before cache encoding")
    encoder = encoder.to(device)
    identity = {
        "arm": arm,
        "input_manifest_file_sha256": sha256_file(manifest_path),
        "input_content_sha256": audit["content_sha256"],
        "authority_checkpoint_sha256": authority["checkpoint"]["sha256"],
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "random_seed": int(random_seed) if arm == "random" else None,
        "device": device,
        "output_shape": [tokens, int(VITB["embed_dim"])],
        "dtype": dtype,
    }
    identity_sha = _json_sha256(identity)
    progress_path = root / "encoding_progress.json"
    if root.exists():
        if not progress_path.is_file():
            raise DenseTaskError(f"existing cache lacks resumable progress authority: {root}")
        progress = _read_json(progress_path)
        if progress.get("identity_sha256") != identity_sha or progress.get("identity") != identity:
            raise DenseTaskError("existing dense cache progress identity drift")
        if progress.get("schema") != PROGRESS_SCHEMA or not isinstance(progress.get("rows"), dict):
            raise DenseTaskError("existing dense cache progress receipt is malformed")
        store = _open_writable(root)
    else:
        store = LatentStore.create(
            root.parent,
            root.name,
            (tokens, int(VITB["embed_dim"])),
            int(audit["count"]),
            int(VITB["embed_dim"]),
            dtype=dtype,
            has_labels=True,
        )
        progress = {
            "schema": PROGRESS_SCHEMA,
            "created_at": _utc_now(),
            "identity": identity,
            "identity_sha256": identity_sha,
            "rows": {},
            "complete": False,
        }
        _atomic_json(progress_path, progress)
    for index, row in enumerate(_read_json(manifest_path)["rows"]):
        prior = progress["rows"].get(str(index))
        if (
            isinstance(prior, dict)
            and prior.get("referent") == audit["referents"][index]
            and prior.get("input_sha256") == row["tensor_sha256"]
            and index < len(store)
            and tensor_sha256(store.latents(index).numpy()) == prior.get("output_sha256")
        ):
            continue
        tensor = np.load(row["tensor_path"], allow_pickle=False)
        if tensor_sha256(tensor) != row["tensor_sha256"]:
            raise DenseTaskError(f"input tensor drift before row {index}")
        clip = torch.from_numpy(np.array(tensor, copy=True)).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = encoder(clip)
        if device == "mps":
            torch.mps.synchronize()
        if list(output.shape) != [1, tokens, int(VITB["embed_dim"])] or not bool(
            torch.isfinite(output).all()
        ):
            raise DenseTaskError(f"row {index} output shape/finiteness failed")
        array = output[0].float().cpu().numpy().astype(dtype)
        if not np.isfinite(array).all():
            raise DenseTaskError(f"row {index} became non-finite after {dtype} cache conversion")
        output_sha = tensor_sha256(array)
        store.write_batch(
            index,
            array[None],
            array.mean(axis=0, keepdims=True),
            np.asarray([audit["task_labels"][index]], dtype="int64"),
        )
        store.finalize()
        progress["rows"][str(index)] = {
            "referent": audit["referents"][index],
            "input_sha256": row["tensor_sha256"],
            "output_sha256": output_sha,
        }
        progress["updated_at"] = _utc_now()
        _atomic_json(progress_path, progress)
    expected_rows = {str(index) for index in range(audit["count"])}
    progress["complete"] = set(progress["rows"]) == expected_rows
    progress["completed_at"] = _utc_now()
    _atomic_json(progress_path, progress)
    if not progress["complete"]:
        raise DenseTaskError("dense cache encoding ended incomplete")
    final = finalize_dense_cache(
        root,
        arm=arm,
        input_manifest=manifest_path,
        authority=authority,
        random_seed=random_seed,
        initialization_receipt=initialization_receipt,
        output_row_sha256s=[progress["rows"][str(index)]["output_sha256"] for index in range(audit["count"])],
    )
    return {"progress": progress, "finalization": final, "all_ok": final["all_ok"]}


def build_dr14_dense_views(
    cache: Path | str,
    *,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    seed: int = 0,
    group_width: int = 16,
    strict_run_identity: bool = True,
) -> dict[str, Any]:
    """Build shared dropped-channel views from a dense cache without loading an encoder.

    The mask acts on each dense token before pooling. Because mean pooling is linear, the bounded
    implementation applies the same mask to the pooled row, while retaining hashes of the exact
    source row, channel mask, and resulting shared view. ``strict_run_identity=False`` exists only
    for programmatic mechanics fixtures and is recorded as non-promotable.
    """

    root = Path(cache).resolve()
    problems = validate_cache(root, citable=True)
    if problems:
        raise DenseTaskError("DR14 dense cache validation failed: " + "; ".join(problems))
    store = LatentStore.open(root)
    if len(store.meta.feat_shape) < 2 or store.labels() is None:
        raise DenseTaskError("DR14 needs dense token axes and task labels")
    run_receipt_path = root / "run_receipt.json"
    run_receipt = _read_json(run_receipt_path) if run_receipt_path.is_file() else {}
    if strict_run_identity:
        if not isinstance(run_receipt, dict) or run_receipt.get("schema") != RUN_SCHEMA:
            raise DenseTaskError(f"DR14 official dense cache requires {RUN_SCHEMA}")
        if run_receipt.get("frozen_encoder") is not True:
            raise DenseTaskError("DR14 official dense cache must preserve the frozen encoder invariant")
        declared_output_hashes = run_receipt.get("ordered_output_tensor_sha256s")
        if not isinstance(declared_output_hashes, list) or len(declared_output_hashes) != len(store):
            raise DenseTaskError("DR14 official dense cache lacks exact ordered output row hashes")
    else:
        declared_output_hashes = None
    feature_dim = int(store.meta.feat_shape[-1])
    if group_width < 1 or feature_dim % group_width:
        raise ValueError("group_width must divide dense feature dimension")
    values = tuple(float(value) for value in fractions)
    if not values or values[0] != 0.0 or any(value < 0 or value >= 1 for value in values):
        raise ValueError("fractions must start at 0 and remain in [0,1)")
    if tuple(sorted(set(values))) != values:
        raise ValueError("fractions must be unique and increasing")
    groups = feature_dim // group_width
    permutation = np.random.default_rng(seed).permutation(groups)
    clean = np.empty((len(store), feature_dim), dtype="float32")
    source_row_hashes: list[str] = []
    for index in range(len(store)):
        exact_row = store.latents(index).numpy()
        row_hash = tensor_sha256(exact_row)
        source_row_hashes.append(row_hash)
        if declared_output_hashes is not None and row_hash != declared_output_hashes[index]:
            raise DenseTaskError(f"DR14 dense cache output row {index} hash drift")
        row = exact_row.reshape(-1, feature_dim)
        clean[index] = row.mean(axis=0)
    views: dict[str, np.ndarray] = {}
    masks: dict[str, dict[str, Any]] = {}
    for fraction in values:
        dropped_groups = int(math.floor(fraction * groups))
        selected = permutation[:dropped_groups]
        mask = np.ones(feature_dim, dtype="float32")
        for group in selected:
            mask[group * group_width : (group + 1) * group_width] = 0.0
        view = clean * mask[None]
        key = f"{fraction:.6f}"
        views[key] = view
        masks[key] = {
            "fraction": fraction,
            "dropped_groups": selected.tolist(),
            "mask_sha256": tensor_sha256(mask),
            "view_sha256": tensor_sha256(view),
        }
    splits = _read_json(root / "splits.json")
    label_tensor = store.labels()
    if label_tensor is None:  # guarded above, retained for static and runtime fail-closed safety
        raise DenseTaskError("DR14 dense labels disappeared while views were being built")
    labels = label_tensor.numpy().astype("int64")
    return {
        "schema": DR14_VIEWS_SCHEMA,
        "cache": str(root),
        "cache_manifest_sha256": sha256_file(root / "cache_manifest.json"),
        "run_receipt_sha256": sha256_file(run_receipt_path) if run_receipt_path.is_file() else None,
        "source_row_sha256s": source_row_hashes,
        "source_rows_sha256": _json_sha256(source_row_hashes),
        "count": len(store),
        "feature_dim": feature_dim,
        "group_width": group_width,
        "seed": int(seed),
        "fractions": list(values),
        "masks": masks,
        "views": views,
        "labels": labels,
        "splits": splits,
        "shared_corrupted_tensor_for_both_arms": True,
        "strict_run_identity": strict_run_identity,
        "scientific_promotion": False,
    }


def _active_heavy_processes() -> list[dict[str, Any]]:
    markers = (
        "p4_capability_density.py",
        "p5_context_capability.py",
        "vjepa21_dense_tasks.py encode",
        "cache_real_encoder.py",
        "custom_substrate_workbench.py",
    )
    try:
        import psutil

        rows = []
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            if process.pid != os.getpid() and any(marker in command for marker in markers):
                rows.append({"pid": process.pid, "command": command})
        return rows
    except (ImportError, OSError):
        return [{"pid": None, "command": "process telemetry unavailable"}]


def registration_audit(task_config: dict[str, Any]) -> dict[str, Any]:
    """Bind the task seam to the registered E6 config and DR14 script mirror."""

    problems: list[str] = []
    registry_raw = yaml.safe_load((REPO_ROOT / "registry/experiments.yaml").read_text())
    mirror_raw = yaml.safe_load((REPO_ROOT / "configs/experiment/_mot_mirrors.yaml").read_text())
    e6_raw = yaml.safe_load((REPO_ROOT / "configs/experiment/e6_relational.yaml").read_text())
    experiments = registry_raw.get("experiments", []) if isinstance(registry_raw, dict) else []
    mirrors = mirror_raw.get("mirrors", []) if isinstance(mirror_raw, dict) else []
    rows = {str(row.get("id")): row for row in experiments if isinstance(row, dict) and row.get("id")}
    mirror_rows = {str(row.get("id")): row for row in mirrors if isinstance(row, dict) and row.get("id")}
    e6_row = rows.get("e6_relational", {})
    dr14_row = rows.get("mop_dr14_corruption", {})
    dr14_mirror = mirror_rows.get("mop_dr14_corruption", {})
    task_receipts_value = task_config.get("runtime_receipts")
    task_receipts: dict[str, Any] = task_receipts_value if isinstance(task_receipts_value, dict) else {}
    task_consumers_value = task_config.get("consumers")
    task_consumers: dict[str, Any] = task_consumers_value if isinstance(task_consumers_value, dict) else {}
    dr14_consumer_value = task_consumers.get("mop_dr14_corruption")
    dr14_consumer: dict[str, Any] = dr14_consumer_value if isinstance(dr14_consumer_value, dict) else {}
    expectations = {
        "task.schema": (task_config.get("schema"), "mop-vjepa21-dense-task-config/v1"),
        "task.serial_only": (task_config.get("serial_only"), True),
        "task.cache_dtype": (task_config.get("cache_dtype"), "float32"),
        "task.min_free_disk_gb": (task_config.get("min_free_disk_gb"), 40.0),
        "task.runtime.load": (task_receipts.get("load"), "proof/VJEPA21_VITB_LOAD.json"),
        "task.runtime.forward_8f": (
            task_receipts.get("forward_8f"),
            "proof/VJEPA21_VITB_FORWARD.json",
        ),
        "task.runtime.forward_64f": (
            task_receipts.get("forward_64f"),
            "proof/VJEPA21_VITB_FORWARD_64F.json",
        ),
        "e6.registry.status": (e6_row.get("status"), "implemented"),
        "e6.registry.resource_tier": (e6_row.get("resource_tier"), "environment-needed"),
        "e6.registry.exp_tier": (e6_row.get("exp_tier"), "env-later"),
        "e6.config.execution_path": (
            e6_raw.get("execution_path") if isinstance(e6_raw, dict) else None,
            "cache-first",
        ),
        "e6.config.allow_legacy_fixture": (
            e6_raw.get("allow_legacy_fixture") if isinstance(e6_raw, dict) else None,
            False,
        ),
        "e6.config.cache_task_config": (
            e6_raw.get("cache_task_config") if isinstance(e6_raw, dict) else None,
            "configs/experiment/e6_dense_cache.yaml",
        ),
        "e6.config.learned_cache": (
            e6_raw.get("learned_cache") if isinstance(e6_raw, dict) else None,
            task_config.get("learned_cache"),
        ),
        "e6.config.random_cache": (
            e6_raw.get("random_cache") if isinstance(e6_raw, dict) else None,
            task_config.get("random_cache"),
        ),
        "dr14.registry.resource_tier": (dr14_row.get("resource_tier"), "environment-needed"),
        "dr14.registry.exp_tier": (dr14_row.get("exp_tier"), "env-later"),
        "dr14.mirror.tier": (dr14_mirror.get("tier"), "env-later"),
        "dr14.mirror.pilot": (dr14_mirror.get("pilot"), True),
        "dr14.task.execution_path": (
            dr14_consumer.get("execution_path"),
            "dense-cache-optional",
        ),
    }
    for label, (observed, expected) in expectations.items():
        if observed != expected:
            problems.append(f"{label}={observed!r}, expected {expected!r}")
    paths = {
        "task_cli": REPO_ROOT / "scripts/vjepa21_dense_tasks.py",
        "e6_cache_consumer": REPO_ROOT / "src/mop/experiments/e6_dense_relational.py",
        "dr14_runner": REPO_ROOT / "scripts/mop_dr14_corruption.py",
    }
    for label, path in paths.items():
        if not path.is_file():
            problems.append(f"{label} missing: {path}")
    return {
        "schema": "mop-vjepa21-dense-registration-audit/v1",
        "expectations": {
            label: {"observed": observed, "expected": expected}
            for label, (observed, expected) in expectations.items()
        },
        "authorities": {
            "registry": {
                "path": str(REPO_ROOT / "registry/experiments.yaml"),
                "sha256": sha256_file(REPO_ROOT / "registry/experiments.yaml"),
            },
            "mot_mirrors": {
                "path": str(REPO_ROOT / "configs/experiment/_mot_mirrors.yaml"),
                "sha256": sha256_file(REPO_ROOT / "configs/experiment/_mot_mirrors.yaml"),
            },
            "e6_config": {
                "path": str(REPO_ROOT / "configs/experiment/e6_relational.yaml"),
                "sha256": sha256_file(REPO_ROOT / "configs/experiment/e6_relational.yaml"),
            },
        },
        "artifacts": {
            label: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
            for label, path in paths.items()
        },
        "problems": problems,
        "all_ok": not problems,
    }


def no_heavy_preflight(
    *,
    task_config: Path | str = DEFAULT_TASK_CONFIG,
    input_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Audit runtime, wiring, and optional inputs without constructing a model or reading weights."""

    config_path = Path(task_config).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise DenseTaskError("dense task config must be a mapping")
    authority = runtime_authority()
    registration = registration_audit(raw)
    input_path = Path(input_manifest or raw["input_manifest"]).resolve()
    if input_path.is_file():
        input_audit = validate_input_manifest(input_path, verify_files=True)
    else:
        input_audit = {
            "mechanics_ok": False,
            "promotion_ready": False,
            "problems": [f"input manifest missing: {input_path}"],
            "promotion_problems": ["rights-clean natural task cohort not materialized"],
        }
    heavy = _active_heavy_processes()
    module_path = Path(__file__).resolve()
    interfaces = {
        "official_learned_encoder": callable(load_vitb_encoder),
        "official_random_encoder": callable(build_vitb_encoder),
        "same_input_cache_plan": callable(build_cache_plan),
        "strict_cache_finalizer": callable(finalize_dense_cache),
        "e6_pair_gate_consumer": (REPO_ROOT / "src/mop/experiments/e6_dense_relational.py").is_file(),
        "dr14_dense_drop_views": callable(build_dr14_dense_views),
    }
    implementation_ok = authority["all_ok"] and registration["all_ok"] and all(interfaces.values())
    encode_allowed = bool(
        implementation_ok
        and input_audit.get("mechanics_ok")
        and not heavy
        and shutil.disk_usage(REPO_ROOT).free >= MIN_FREE_DISK_BYTES
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": _utc_now(),
        "mode": "no-heavy-preflight",
        "model_constructed": False,
        "checkpoint_tensor_bytes_read": False,
        "forward_executed": False,
        "task_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "implementation": {"path": str(module_path), "sha256": sha256_file(module_path)},
        "runtime_authority": authority,
        "registration": registration,
        "interfaces": interfaces,
        "input_manifest": input_audit,
        "active_heavy_processes": heavy,
        "gates": {
            "implementation_ready": implementation_ok,
            "input_manifest_ready": bool(input_audit.get("mechanics_ok")),
            "heavy_lane_clear": not heavy,
            "encode_allowed_now": encode_allowed,
            "e6_scientific_ready": bool(input_audit.get("promotion_ready")) and False,
            "dr14_scientific_ready": False,
        },
        "remaining_gates": [
            *input_audit.get("problems", []),
            *input_audit.get("promotion_problems", []),
            "encode learned and matched random caches serially from the exact input manifest",
            "run independent E6 and DR14 statistical verifiers on natural task labels",
        ],
        "all_ok": implementation_ok,
        "scientific_promotion": False,
    }
