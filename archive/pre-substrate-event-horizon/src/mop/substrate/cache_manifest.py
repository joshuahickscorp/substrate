from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evidence import canonical_sha256

FORM_KINDS = (
    "vision",
    "audio",
    "text",
    "symbolic",
    "timeseries",
    "control",
    "code",
    "math",
    "latent",
    "mixed",
)
OBJECTIVE_FAMILIES = (
    "unknown",
    "inherited-frozen",
    "random-control",
    "handcrafted",
    "self-supervised",
    "supervised",
    "programmatic",
    "metadata",
    "learned-shell",
    "custom-substrate",
)

SCHEMA = "mop-cache-data-plane/v2"
DEFAULT_MANIFEST = "cache_manifest.json"
DEFAULT_SAMPLE_BYTES = 1024 * 1024
ENCODER_RECEIPT_SCHEMA = "mop-encoder-weight-receipt/v1"
RANDOM_INIT_RECEIPT_SCHEMA = "mop-random-init-encoder-receipt/v1"
RANDOM_INIT_BACKENDS = frozenset({"vjepa_hf_random_init", "vjepa_official_random_init"})
WEIGHTED_OBJECTIVES = frozenset(
    {"inherited-frozen", "self-supervised", "supervised", "learned-shell", "custom-substrate"}
)


def validate_cache_manifest(
    store_dir: Path | str,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
) -> list[str]:
    root = Path(store_dir)
    manifest = _read_json(root / manifest_name)
    if manifest is None:
        return [f"{manifest_name} missing"]
    if not isinstance(manifest, dict):
        return [f"{manifest_name} must contain a JSON mapping"]

    problems: list[str] = []
    form = manifest.get("form")
    if form is not None and not isinstance(form, dict):
        problems.append("form declaration must be a mapping")
    elif form is not None:
        fk, fo = form.get("kind"), form.get("objective")
        if fk is not None and fk not in FORM_KINDS:
            problems.append(f"form.kind {fk!r} not in {FORM_KINDS}")
        if fo is not None and fo not in OBJECTIVE_FAMILIES:
            problems.append(f"form.objective {fo!r} not in {OBJECTIVE_FAMILIES}")
    if manifest.get("schema") != SCHEMA:
        problems.append(f"citable cache requires schema {SCHEMA!r}")
    if not isinstance(form, dict):
        problems.append("citable cache requires a form declaration")
    else:
        for key in ("kind", "objective", "referent_scheme"):
            if not form.get(key):
                problems.append(f"citable cache requires form.{key}")
    encoder_cfg = manifest.get("encoder_config")
    if encoder_cfg is None or not manifest.get("encoder_config_hash"):
        problems.append("citable cache requires encoder_config and encoder_config_hash")
    elif not isinstance(encoder_cfg, dict):
        problems.append("citable cache encoder_config must be a mapping")
        encoder_cfg = None
    sidecar_roles = {
        str(record.get("role")) for record in manifest.get("sidecars", []) if isinstance(record, dict)
    }
    objective = form.get("objective") if isinstance(form, dict) else None
    if objective in WEIGHTED_OBJECTIVES or objective == "random-control":
        random_init = objective == "random-control"
        filename = "initialization_receipt.json" if random_init else "encoder_receipt.json"
        receipt = _read_json(root / filename)
        if receipt is None:
            problems.append(
                "citable random-control cache requires initialization_receipt.json"
                if random_init
                else (f"citable {objective} cache requires encoder_receipt.json with immutable weight hashes")
            )
        elif not isinstance(receipt, dict):
            problems.append(f"{filename} must contain a JSON mapping")
        else:
            validator = _validate_random_init_receipt if random_init else _validate_encoder_receipt
            problems.extend(validator(receipt))
            if isinstance(encoder_cfg, dict):
                problems.extend(_validate_receipt_config_match(receipt, encoder_cfg, random_init=random_init))
        role = "initialization_receipt" if random_init else "encoder_receipt"
        if role not in sidecar_roles:
            kind = "random-control" if random_init else "weighted"
            problems.append(f"citable {kind} cache manifest must fingerprint {filename}")

    current_meta = _read_json(root / "meta.json")
    recorded_meta = manifest.get("store", {}).get("meta")
    if current_meta != recorded_meta:
        problems.append("meta.json differs from manifest store.meta")
    count = int((current_meta or recorded_meta or {}).get("count", 0))

    for rec in manifest.get("arrays", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="array"))
    for rec in manifest.get("sidecars", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="sidecar"))
    if not any(rec.get("role") == "referents" for rec in manifest.get("sidecars", [])):
        problems.append("citable cache manifest does not fingerprint its referent sidecar")

    factors = _read_json(root / "factors.json")
    if factors is not None:
        try:
            _validate_factors(factors, count)
        except ValueError as e:
            problems.append(str(e))
    referent_path = next(
        (root / name for name in ("referents.json", "clip_stems.json") if (root / name).exists()),
        None,
    )
    if referent_path is not None:
        try:
            values = json.loads(referent_path.read_text())
            if isinstance(values, dict):
                values = values.get("referents", values.get("ids"))
            _validate_referents(values, count)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            problems.append(str(e))
    else:
        problems.append("citable cache requires referents.json or clip_stems.json")
    splits = _read_json(root / "splits.json")
    if splits is not None:
        try:
            _validate_splits(splits, count)
        except ValueError as e:
            problems.append(str(e))

    cfg = manifest.get("encoder_config")
    cfg_hash = manifest.get("encoder_config_hash")
    if cfg is not None and cfg_hash != canonical_sha256(cfg):
        problems.append("encoder_config_hash does not match encoder_config")

    return problems


def _validate_encoder_receipt(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != ENCODER_RECEIPT_SCHEMA:
        problems.append(f"encoder receipt schema must be {ENCODER_RECEIPT_SCHEMA!r}")
    if receipt.get("weights_real") is not True:
        problems.append("encoder receipt weights_real must be true")
    for field in ("model_id", "revision"):
        if not str(receipt.get(field) or "").strip():
            problems.append(f"encoder receipt {field} must be nonempty")
    file_problems, paths = _validate_receipt_files(receipt, random_init=False)
    problems.extend(file_problems)
    if paths is None:
        return problems
    weight_suffixes = (".safetensors", ".bin", ".pth", ".pt", ".ckpt")
    if not any(path.lower().endswith(weight_suffixes) for path in paths):
        problems.append("encoder receipt must hash at least one model weight file")
    return problems


def _validate_random_init_receipt(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != RANDOM_INIT_RECEIPT_SCHEMA:
        problems.append(f"initialization receipt schema must be {RANDOM_INIT_RECEIPT_SCHEMA!r}")
    if receipt.get("weights_real") is not False:
        problems.append("random initialization receipt weights_real must be false")
    backend = str(receipt.get("backend") or "")
    if backend not in RANDOM_INIT_BACKENDS:
        problems.append(
            f"random initialization receipt backend must be one of {sorted(RANDOM_INIT_BACKENDS)}"
        )
    for field in ("model_id", "revision", "seed"):
        if receipt.get(field) in (None, ""):
            problems.append(f"random initialization receipt {field} must be present")
    if not isinstance(receipt.get("seed"), int) or isinstance(receipt.get("seed"), bool):
        problems.append("random initialization receipt seed must be an integer")
    if not _positive_int(receipt.get("parameter_count")):
        problems.append("random initialization receipt parameter_count must be positive")
    if not _positive_int(receipt.get("state_dict_tensors")):
        problems.append("random initialization receipt state_dict_tensors must be positive")
    if not _valid_sha256(receipt.get("state_dict_sha256")):
        problems.append("random initialization receipt state_dict_sha256 must be hex")
    if not str(receipt.get("model_class") or "").strip():
        problems.append("random initialization receipt model_class must be nonempty")
    file_problems, paths = _validate_receipt_files(receipt, random_init=True)
    problems.extend(file_problems)
    if paths is None:
        return problems
    if backend == "vjepa_hf_random_init":
        if not any(Path(path).name == "config.json" for path in paths):
            problems.append("Hugging Face random initialization receipt must hash config.json")
    elif backend == "vjepa_official_random_init":
        commit = str(receipt.get("repository_commit") or "")
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
            problems.append("official random initialization receipt repository_commit must be hex")
        required_source = "app/vjepa_2_1/models/vision_transformer.py"
        if required_source not in paths:
            problems.append(
                "official random initialization receipt must hash the V-JEPA 2.1 vision transformer"
            )
    return problems


def _validate_receipt_files(
    receipt: dict[str, Any],
    *,
    random_init: bool,
) -> tuple[list[str], list[str] | None]:
    field = "architecture_files" if random_init else "files"
    record_label = "random initialization architecture file" if random_init else "encoder receipt file"
    files = receipt.get(field)
    if not isinstance(files, list) or not files:
        collection = f"random initialization receipt {field}" if random_init else "encoder receipt files"
        requirement = "non-empty" if random_init else "a non-empty list"
        return [f"{collection} must be {requirement}"], None
    problems: list[str] = []
    paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            problems.append(f"{record_label} {index} must be a mapping")
            continue
        path = str(record.get("path") or "").strip()
        paths.append(path)
        if not path:
            problems.append(f"{record_label} {index} path must be nonempty")
        if not _valid_sha256(record.get("sha256")):
            detail = "hex" if random_init else "a 64-character hex digest"
            problems.append(f"{record_label} {index} sha256 must be {detail}")
        if not _positive_int(record.get("bytes")):
            problems.append(f"{record_label} {index} bytes must be positive")
    if len(paths) != len(set(paths)):
        problems.append(f"{record_label} paths must be unique")
    return problems, paths


def _valid_sha256(value: Any) -> bool:
    digest = str(value or "")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest.lower())


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0 and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


def _validate_receipt_config_match(
    receipt: dict[str, Any], encoder_config: dict[str, Any], *, random_init: bool
) -> list[str]:
    problems: list[str] = []
    for receipt_field, config_field in (("model_id", "hf_id"), ("revision", "revision")):
        left = str(receipt.get(receipt_field) or "").strip()
        right = str(encoder_config.get(config_field) or "").strip()
        if not right:
            problems.append(f"encoder_config {config_field} must be nonempty")
        elif left != right:
            problems.append(f"{receipt_field} mismatch between receipt {left!r} and encoder_config {right!r}")
    receipt_backend = str(receipt.get("backend") or "").strip()
    config_backend = str(encoder_config.get("actual_backend") or "").strip()
    if receipt_backend and config_backend and receipt_backend != config_backend:
        problems.append(
            f"backend mismatch between receipt {receipt_backend!r} and encoder_config {config_backend!r}"
        )
    if random_init:
        if encoder_config.get("random_init") is not True:
            problems.append("random-control encoder_config random_init must be true")
        if encoder_config.get("prefer_real") is not False:
            problems.append("random-control encoder_config prefer_real must be false")
        if encoder_config.get("require_real") is not False:
            problems.append("random-control encoder_config require_real must be false")
        if config_backend not in RANDOM_INIT_BACKENDS:
            problems.append(
                f"random-control encoder_config actual_backend must be one of {sorted(RANDOM_INIT_BACKENDS)}"
            )
        config_seed = encoder_config.get("random_init_seed")
        if receipt.get("seed") != config_seed:
            problems.append(
                f"random seed mismatch between receipt {receipt.get('seed')!r} "
                f"and encoder_config {config_seed!r}"
            )
    return problems


def _file_digest(path: Path, *, full_hash: bool, sample_bytes: int) -> tuple[str, str]:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"{path.name}:{size}:".encode())
    if full_hash or size <= 2 * sample_bytes:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest(), "full"

    with path.open("rb") as f:
        h.update(f.read(sample_bytes))
        f.seek(max(0, size - sample_bytes))
        h.update(f.read(sample_bytes))
    return h.hexdigest(), "sample"


def _compare_fingerprint(root: Path, rec: dict[str, Any], *, prefix: str) -> list[str]:
    rel = str(rec.get("path", ""))
    path = root / rel
    if not path.exists():
        return [f"{prefix} {rel} missing"]
    sample_bytes = int(rec.get("sample_bytes", DEFAULT_SAMPLE_BYTES))
    digest, kind = _file_digest(path, full_hash=rec.get("hash_kind") == "full", sample_bytes=sample_bytes)
    now: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": digest, "hash_kind": kind}
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        now.update(shape=list(array.shape), dtype=str(array.dtype))
    problems = []
    for key in ("bytes", "sha256", "hash_kind", "shape", "dtype"):
        if rec.get(key) != now.get(key):
            problems.append(f"{prefix} {rel} {key} changed: {rec.get(key)!r} != {now.get(key)!r}")
    return problems


def _validate_factors(factors: dict[str, Any], count: int) -> None:
    if not isinstance(factors, dict):
        raise ValueError("factors must be a mapping")
    columns = factors.get("columns", factors)
    if not isinstance(columns, dict):
        raise ValueError("factors.columns must be a mapping")
    for name, values in columns.items():
        if not isinstance(values, list):
            raise ValueError(f"factor {name!r} must be a list")
        if len(values) != count:
            raise ValueError(f"factor {name!r} length {len(values)} != cache count {count}")


def _validate_referents(referents: Any, count: int) -> None:
    if not isinstance(referents, list):
        raise ValueError("referents must be a list")
    if len(referents) != count:
        raise ValueError(f"referent length {len(referents)} != cache count {count}")
    normalized = [str(v) for v in referents]
    if len(set(normalized)) != len(normalized):
        raise ValueError("referents contain duplicate ids")


def _validate_splits(splits: dict[str, list[int]], count: int) -> None:
    if not isinstance(splits, dict):
        raise ValueError("splits must be a dict of split -> list[int]")
    seen: set[int] = set()
    for name, idxs in splits.items():
        if not isinstance(idxs, list):
            raise ValueError(f"split {name!r} must be a list")
        local = [int(i) for i in idxs]
        if len(local) != len(set(local)):
            raise ValueError(f"split {name!r} contains duplicate indices")
        bad = [i for i in local if i < 0 or i >= count]
        if bad:
            raise ValueError(f"split {name!r} has out-of-range indices {bad[:5]} for count {count}")
        overlap = sorted(seen.intersection(local))
        if overlap:
            raise ValueError(f"split {name!r} overlaps earlier splits at indices {overlap[:5]}")
        seen.update(local)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
