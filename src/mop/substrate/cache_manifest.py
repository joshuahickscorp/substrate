"""Latent cache data-plane receipts.

`LatentStore` is the array format. This module is the receipt format that makes a Studio-scale
cache auditable after it has been copied, pruned around, or used by many probes. It records:

- array fingerprints for pooled or dense memmaps;
- optional factor sidecars, stored as columnar JSON lists;
- optional disjoint split membership;
- an encoder config hash, so the exact perspective config is part of the cache identity;
- a small columnar index for quick inspection.

The array fingerprint defaults to sampled hashing because dense caches can be terabytes. Operators
can request full array hashes when the cache is small enough or when a transfer receipt needs that
extra cost. Sidecars are always fully hashed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import git_dirty, git_sha, package_versions

SCHEMA = "mop-cache-data-plane/v1"
DEFAULT_MANIFEST = "cache_manifest.json"
DEFAULT_SAMPLE_BYTES = 1024 * 1024


def write_cache_manifest(
    store_dir: Path | str,
    *,
    encoder_config: dict[str, Any] | None = None,
    factors: dict[str, list[Any]] | None = None,
    splits: dict[str, list[int]] | None = None,
    full_hash_arrays: bool = False,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Write a cache data-plane manifest beside an existing latent store.

    `factors` and `splits`, when supplied, are persisted as `factors.json` and `splits.json` before
    the manifest is written, so the receipt covers the exact sidecars downstream probes will read.
    """
    root = Path(store_dir)
    meta = _read_json(root / "meta.json")
    if meta is None:
        raise FileNotFoundError(f"{root / 'meta.json'} missing")
    count = int(meta.get("count", 0))

    if factors is not None:
        _validate_factors(factors, count)
        _write_json(root / "factors.json", factors)
    if splits is not None:
        _validate_splits(splits, count)
        _write_json(root / "splits.json", splits)

    factor_data = _read_json(root / "factors.json")
    split_data = _read_json(root / "splits.json")
    if factor_data is not None:
        _validate_factors(factor_data, count)
    if split_data is not None:
        _validate_splits(split_data, count)

    arrays = _array_fingerprints(root, full_hash_arrays=full_hash_arrays, sample_bytes=sample_bytes)
    sidecars = _sidecar_fingerprints(root)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "store": {"path": root.name, "meta": meta},
        "encoder_config": encoder_config,
        "encoder_config_hash": json_sha256(encoder_config) if encoder_config is not None else None,
        "arrays": arrays,
        "sidecars": sidecars,
        "index": _columnar_index(meta, arrays, factor_data, split_data),
        "writer": {
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "packages": package_versions(),
        },
    }
    _write_json(root / manifest_name, manifest)
    return manifest


def validate_cache_manifest(
    store_dir: Path | str,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
) -> list[str]:
    """Validate an existing cache data-plane manifest. Empty list means clean."""
    root = Path(store_dir)
    manifest = _read_json(root / manifest_name)
    if manifest is None:
        return [f"{manifest_name} missing"]

    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema {manifest.get('schema')!r} != {SCHEMA!r}")

    current_meta = _read_json(root / "meta.json")
    recorded_meta = manifest.get("store", {}).get("meta")
    if current_meta != recorded_meta:
        problems.append("meta.json differs from manifest store.meta")
    count = int((current_meta or recorded_meta or {}).get("count", 0))

    for rec in manifest.get("arrays", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="array"))
    for rec in manifest.get("sidecars", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="sidecar"))

    factors = _read_json(root / "factors.json")
    if factors is not None:
        try:
            _validate_factors(factors, count)
        except ValueError as e:
            problems.append(str(e))
    splits = _read_json(root / "splits.json")
    if splits is not None:
        try:
            _validate_splits(splits, count)
        except ValueError as e:
            problems.append(str(e))

    cfg = manifest.get("encoder_config")
    cfg_hash = manifest.get("encoder_config_hash")
    if cfg is not None and cfg_hash != json_sha256(cfg):
        problems.append("encoder_config_hash does not match encoder_config")

    return problems


def json_sha256(obj: Any) -> str:
    """Stable SHA256 for JSON-compatible data."""
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def _array_fingerprints(root: Path, *, full_hash_arrays: bool, sample_bytes: int) -> list[dict[str, Any]]:
    roles = (
        ("latents", "latents.npy"),
        ("features", "features.npy"),
        ("keys", "keys.npy"),
        ("labels", "labels.npy"),
        ("labels", "labels_shape.npy"),
    )
    out = []
    for role, rel in roles:
        if (root / rel).exists():
            out.append(_fingerprint(root, rel, role, full_hash=full_hash_arrays, sample_bytes=sample_bytes))
    return out


def _sidecar_fingerprints(root: Path) -> list[dict[str, Any]]:
    out = []
    for role, rel in (("factors", "factors.json"), ("splits", "splits.json")):
        if (root / rel).exists():
            out.append(_fingerprint(root, rel, role, full_hash=True, sample_bytes=DEFAULT_SAMPLE_BYTES))
    return out


def _fingerprint(
    root: Path,
    rel: str,
    role: str,
    *,
    full_hash: bool,
    sample_bytes: int,
) -> dict[str, Any]:
    path = root / rel
    size = path.stat().st_size
    digest, kind = _file_digest(path, full_hash=full_hash, sample_bytes=sample_bytes)
    rec: dict[str, Any] = {
        "role": role,
        "path": rel,
        "bytes": size,
        "sha256": digest,
        "hash_kind": kind,
        "sample_bytes": int(sample_bytes),
    }
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        rec["shape"] = list(arr.shape)
        rec["dtype"] = str(arr.dtype)
    return rec


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
    full = rec.get("hash_kind") == "full"
    sample_bytes = int(rec.get("sample_bytes", DEFAULT_SAMPLE_BYTES))
    now = _fingerprint(root, rel, str(rec.get("role", "")), full_hash=full, sample_bytes=sample_bytes)
    problems = []
    for key in ("bytes", "sha256", "hash_kind", "shape", "dtype"):
        if rec.get(key) != now.get(key):
            problems.append(f"{prefix} {rel} {key} changed: {rec.get(key)!r} != {now.get(key)!r}")
    return problems


def _columnar_index(
    meta: dict[str, Any],
    arrays: list[dict[str, Any]],
    factors: dict[str, list[Any]] | None,
    splits: dict[str, list[int]] | None,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for arr in arrays:
        columns.append(
            {
                "name": arr["role"],
                "kind": "array",
                "path": arr["path"],
                "shape": arr.get("shape"),
                "dtype": arr.get("dtype"),
            }
        )
    if factors:
        for name, values in sorted(factors.items()):
            columns.append(
                {
                    "name": name,
                    "kind": "factor",
                    "path": "factors.json",
                    "count": len(values),
                    "cardinality": len({json.dumps(v, sort_keys=True, default=str) for v in values}),
                }
            )
    if splits:
        for name, idxs in sorted(splits.items()):
            columns.append({"name": name, "kind": "split", "path": "splits.json", "count": len(idxs)})
    return {"count": int(meta.get("count", 0)), "columns": columns}


def _validate_factors(factors: dict[str, list[Any]], count: int) -> None:
    if not isinstance(factors, dict):
        raise ValueError("factors must be a dict of column -> list")
    for name, values in factors.items():
        if not isinstance(values, list):
            raise ValueError(f"factor {name!r} must be a list")
        if len(values) != count:
            raise ValueError(f"factor {name!r} length {len(values)} != cache count {count}")


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_bytes(_canonical_json(obj) + b"\n")


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=2, default=str).encode()
