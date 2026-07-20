from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import git_dirty, git_sha, package_versions
from .form import FORM_KINDS, OBJECTIVE_FAMILIES

SCHEMA = "mop-cache-data-plane/v2"
SCHEMA_V1 = "mop-cache-data-plane/v1"
ACCEPTED_SCHEMAS = (SCHEMA, SCHEMA_V1)
DEFAULT_MANIFEST = "cache_manifest.json"
DEFAULT_SAMPLE_BYTES = 1024 * 1024
ENCODER_RECEIPT_SCHEMA = "mop-encoder-weight-receipt/v1"
RANDOM_INIT_RECEIPT_SCHEMA = "mop-random-init-encoder-receipt/v1"
RANDOM_INIT_BACKENDS = frozenset({"vjepa_hf_random_init", "vjepa_official_random_init"})
WEIGHTED_OBJECTIVES = frozenset(
    {"inherited-frozen", "self-supervised", "supervised", "learned-shell", "custom-substrate"}
)


def write_cache_manifest(
    store_dir: Path | str,
    *,
    encoder_config: dict[str, Any] | None = None,
    encoder_receipt: dict[str, Any] | None = None,
    factors: dict[str, list[Any]] | None = None,
    factor_metadata: dict[str, Any] | None = None,
    splits: dict[str, list[int]] | None = None,
    referents: list[Any] | None = None,
    form_kind: str | None = None,
    form_objective: str | None = None,
    referent_scheme: str | None = None,
    full_hash_arrays: bool = False,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(store_dir)
    meta = _read_json(root / "meta.json")
    if meta is None:
        raise FileNotFoundError(f"{root / 'meta.json'} missing")
    count = int(meta.get("count", 0))
    form = _form_block(form_kind, form_objective, referent_scheme)

    if encoder_receipt is not None:
        receipt_problems = _validate_encoder_receipt(encoder_receipt)
        if receipt_problems:
            raise ValueError("invalid encoder receipt: " + "; ".join(receipt_problems))
        _write_json(root / "encoder_receipt.json", encoder_receipt)

    if factors is not None:
        _validate_factors(factors, count)
        factor_payload: dict[str, Any]
        if factor_metadata is None:
            factor_payload = factors
        else:
            factor_payload = {
                "schema": "mop-factor-sidecar/v2",
                "metadata": factor_metadata,
                "columns": factors,
            }
        _write_json(root / "factors.json", factor_payload)
    elif factor_metadata is not None:
        raise ValueError("factor_metadata requires factors")
    if referents is not None:
        _validate_referents(referents, count)
        _write_json(root / "referents.json", referents)
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
        "form": form,
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


def _form_block(
    form_kind: str | None, form_objective: str | None, referent_scheme: str | None
) -> dict[str, Any] | None:
    if form_kind is None and form_objective is None and referent_scheme is None:
        return None
    if form_kind is not None and form_kind not in FORM_KINDS:
        raise ValueError(f"form_kind {form_kind!r} not in {FORM_KINDS}")
    if form_objective is not None and form_objective not in OBJECTIVE_FAMILIES:
        raise ValueError(f"form_objective {form_objective!r} not in {OBJECTIVE_FAMILIES}")
    return {"kind": form_kind, "objective": form_objective, "referent_scheme": referent_scheme}


def validate_cache_manifest(
    store_dir: Path | str,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    citable: bool = False,
) -> list[str]:
    root = Path(store_dir)
    manifest = _read_json(root / manifest_name)
    if manifest is None:
        return [f"{manifest_name} missing"]
    if not isinstance(manifest, dict):
        return [f"{manifest_name} must contain a JSON mapping"]

    problems: list[str] = []
    if manifest.get("schema") not in ACCEPTED_SCHEMAS:
        problems.append(f"schema {manifest.get('schema')!r} not in {ACCEPTED_SCHEMAS!r}")

    form = manifest.get("form")
    if form is not None and not isinstance(form, dict):
        problems.append("form declaration must be a mapping")
    elif form is not None:
        fk, fo = form.get("kind"), form.get("objective")
        if fk is not None and fk not in FORM_KINDS:
            problems.append(f"form.kind {fk!r} not in {FORM_KINDS}")
        if fo is not None and fo not in OBJECTIVE_FAMILIES:
            problems.append(f"form.objective {fo!r} not in {OBJECTIVE_FAMILIES}")
    if citable:
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
        if objective in WEIGHTED_OBJECTIVES:
            receipt = _read_json(root / "encoder_receipt.json")
            if receipt is None:
                problems.append(
                    f"citable {objective} cache requires encoder_receipt.json with immutable weight hashes"
                )
            elif not isinstance(receipt, dict):
                problems.append("encoder_receipt.json must contain a JSON mapping")
            else:
                problems.extend(_validate_encoder_receipt(receipt))
                if isinstance(encoder_cfg, dict):
                    problems.extend(_validate_receipt_config_match(receipt, encoder_cfg, random_init=False))
            if "encoder_receipt" not in sidecar_roles:
                problems.append("citable weighted cache manifest must fingerprint encoder_receipt.json")
        if objective == "random-control":
            receipt = _read_json(root / "initialization_receipt.json")
            if receipt is None:
                problems.append("citable random-control cache requires initialization_receipt.json")
            elif not isinstance(receipt, dict):
                problems.append("initialization_receipt.json must contain a JSON mapping")
            else:
                problems.extend(_validate_random_init_receipt(receipt))
                if isinstance(encoder_cfg, dict):
                    problems.extend(_validate_receipt_config_match(receipt, encoder_cfg, random_init=True))
            if "initialization_receipt" not in sidecar_roles:
                problems.append(
                    "citable random-control cache manifest must fingerprint initialization_receipt.json"
                )

    current_meta = _read_json(root / "meta.json")
    recorded_meta = manifest.get("store", {}).get("meta")
    if current_meta != recorded_meta:
        problems.append("meta.json differs from manifest store.meta")
    count = int((current_meta or recorded_meta or {}).get("count", 0))

    for rec in manifest.get("arrays", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="array"))
    for rec in manifest.get("sidecars", []):
        problems.extend(_compare_fingerprint(root, rec, prefix="sidecar"))
    if citable and not any(rec.get("role") == "referents" for rec in manifest.get("sidecars", [])):
        problems.append("citable cache manifest does not fingerprint its referent sidecar")

    factors = _read_json(root / "factors.json")
    if factors is not None:
        try:
            _validate_factors(factors, count)
        except ValueError as e:
            problems.append(str(e))
    referent_path = _referent_sidecar(root)
    if referent_path is not None:
        try:
            values = json.loads(referent_path.read_text())
            if isinstance(values, dict):
                values = values.get("referents", values.get("ids"))
            _validate_referents(values, count)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            problems.append(str(e))
    elif citable:
        problems.append("citable cache requires referents.json or clip_stems.json")
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
    for role, rel in (
        ("factors", "factors.json"),
        ("splits", "splits.json"),
        ("referents", "referents.json"),
        ("referents", "clip_stems.json"),
        ("encoder_receipt", "encoder_receipt.json"),
        ("initialization_receipt", "initialization_receipt.json"),
        ("run_receipt", "run_receipt.json"),
    ):
        if (root / rel).exists():
            out.append(_fingerprint(root, rel, role, full_hash=True, sample_bytes=DEFAULT_SAMPLE_BYTES))
    return out


def _validate_encoder_receipt(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != ENCODER_RECEIPT_SCHEMA:
        problems.append(f"encoder receipt schema must be {ENCODER_RECEIPT_SCHEMA!r}")
    if receipt.get("weights_real") is not True:
        problems.append("encoder receipt weights_real must be true")
    for field in ("model_id", "revision"):
        if not str(receipt.get(field) or "").strip():
            problems.append(f"encoder receipt {field} must be nonempty")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        problems.append("encoder receipt files must be a non-empty list")
        return problems
    paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            problems.append(f"encoder receipt file {index} must be a mapping")
            continue
        path = str(record.get("path") or "").strip()
        paths.append(path)
        if not path:
            problems.append(f"encoder receipt file {index} path must be nonempty")
        digest = str(record.get("sha256") or "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            problems.append(f"encoder receipt file {index} sha256 must be a 64-character hex digest")
        try:
            size = int(record.get("bytes", 0))
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            problems.append(f"encoder receipt file {index} bytes must be positive")
    if len(paths) != len(set(paths)):
        problems.append("encoder receipt file paths must be unique")
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
    files = receipt.get("architecture_files")
    if not isinstance(files, list) or not files:
        problems.append("random initialization receipt architecture_files must be non-empty")
        return problems
    paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            problems.append(f"random initialization architecture file {index} must be a mapping")
            continue
        path = str(record.get("path") or "").strip()
        paths.append(path)
        if not path:
            problems.append(f"random initialization architecture file {index} path must be nonempty")
        digest = str(record.get("sha256") or "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            problems.append(f"random initialization architecture file {index} sha256 must be hex")
        if not _positive_int(record.get("bytes")):
            problems.append(f"random initialization architecture file {index} bytes must be positive")
    if len(paths) != len(set(paths)):
        problems.append("random initialization architecture file paths must be unique")
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
    factors: dict[str, Any] | None,
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
    factor_columns = _factor_columns(factors) if factors else {}
    if factor_columns:
        for name, values in sorted(factor_columns.items()):
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


def _factor_columns(factors: dict[str, Any]) -> dict[str, list[Any]]:
    if not isinstance(factors, dict):
        raise ValueError("factors must be a mapping")
    if "columns" in factors:
        columns = factors["columns"]
        if not isinstance(columns, dict):
            raise ValueError("factors.columns must be a mapping")
        return columns
    return {str(name): values for name, values in factors.items() if isinstance(values, list)}


def _validate_factors(factors: dict[str, Any], count: int) -> None:
    if not isinstance(factors, dict):
        raise ValueError("factors must be a mapping")
    for name, values in _factor_columns(factors).items():
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


def _referent_sidecar(root: Path) -> Path | None:
    for name in ("referents.json", "clip_stems.json"):
        path = root / name
        if path.exists():
            return path
    return None


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
