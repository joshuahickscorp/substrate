
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..substrate.cache_manifest import validate_cache_manifest
from ..substrate.cache_tools import validate_cache

SCHEMA = "mop-dense-atlas-cache-gate/v1"
DEFAULT_REAL_CACHE = "data/cache/vjepa21_vitl_dense8192_real"
DEFAULT_RANDOMINIT_CACHE = "data/cache/vjepa21_vitl_dense8192_randominit"
DEFAULT_MIN_TOKENS = 8192
DEFAULT_EXPECTED_DIM = 1024


def build_dense_atlas_cache_gate(
    *,
    real_cache: str | Path = DEFAULT_REAL_CACHE,
    randominit_cache: str | Path = DEFAULT_RANDOMINIT_CACHE,
    min_count: int = 1,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    expected_dim: int | None = DEFAULT_EXPECTED_DIM,
) -> dict[str, Any]:
    real = _cache_summary(
        Path(real_cache),
        min_count=min_count,
        min_tokens=min_tokens,
        expected_dim=expected_dim,
    )
    rand = _cache_summary(
        Path(randominit_cache),
        min_count=min_count,
        min_tokens=min_tokens,
        expected_dim=expected_dim,
    )
    pair_problems = _pair_problems(real, rand)
    problems = [*(f"real: {p}" for p in real["problems"]), *(f"randominit: {p}" for p in rand["problems"])]
    problems.extend(pair_problems)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "requirements": {
            "min_count": int(min_count),
            "min_tokens": int(min_tokens),
            "expected_dim": expected_dim,
            "matched_randominit_required": True,
            "keys_must_match": True,
        },
        "real_cache": real,
        "randominit_cache": rand,
        "pair": {
            "count_match": _eq_known(real.get("count"), rand.get("count")),
            "token_shape_match": _eq_known(real.get("tokens"), rand.get("tokens"))
            and _eq_known(real.get("dim"), rand.get("dim")),
            "keys_match": _eq_known(real.get("keys_fingerprint"), rand.get("keys_fingerprint")),
            "factor_sidecars_match": _sidecar_match(real, rand, "factors"),
            "split_sidecars_match": _sidecar_match(real, rand, "splits"),
            "problems": pair_problems,
        },
        "problems": problems,
    }
    receipt["all_ok"] = not problems
    return receipt


def write_dense_atlas_cache_gate(receipt: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def _cache_summary(
    root: Path,
    *,
    min_count: int,
    min_tokens: int,
    expected_dim: int | None,
) -> dict[str, Any]:
    problems: list[str] = []
    meta = _read_json(root / "meta.json")
    manifest = _read_json(root / "cache_manifest.json")
    if meta is None:
        problems.append("meta.json missing")
    if manifest is None:
        problems.append("cache_manifest.json missing")

    if root.exists():
        problems.extend(f"cache integrity: {p}" for p in validate_cache(root))
        problems.extend(f"manifest integrity: {p}" for p in validate_cache_manifest(root))
    else:
        problems.append("cache directory missing")

    count = _int_or_none((meta or {}).get("count"))
    feat_shape = [int(x) for x in (meta or {}).get("feat_shape", []) if _int_or_none(x) is not None]
    tokens = _dense_token_count(feat_shape)
    dim = feat_shape[-1] if feat_shape else None
    if count is None or count < int(min_count):
        shown = count if count is not None else "missing"
        problems.append(f"cache count {shown} < min_count {int(min_count)}")
    if tokens is None or tokens < int(min_tokens):
        problems.append(
            f"dense token count {tokens if tokens is not None else 'missing'} < min_tokens {int(min_tokens)}"
        )
    if expected_dim is not None and dim != int(expected_dim):
        problems.append(f"dense embedding dim {dim if dim is not None else 'missing'} != {int(expected_dim)}")

    arrays = _records_by_role((manifest or {}).get("arrays", []))
    sidecars = _records_by_role((manifest or {}).get("sidecars", []))
    keys_fingerprint = _fingerprint(arrays.get("keys"))
    if manifest is not None and keys_fingerprint is None:
        problems.append("keys fingerprint missing from manifest")
    return {
        "path": str(root),
        "exists": root.exists(),
        "manifest_path": str(root / "cache_manifest.json"),
        "manifest_schema": (manifest or {}).get("schema"),
        "count": count,
        "feat_shape": feat_shape,
        "tokens": tokens,
        "dim": dim,
        "keys_fingerprint": keys_fingerprint,
        "factor_sidecar": _fingerprint(sidecars.get("factors")),
        "split_sidecar": _fingerprint(sidecars.get("splits")),
        "problems": problems,
        "all_ok": not problems,
    }


def _pair_problems(real: dict[str, Any], rand: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if _both_known(real.get("count"), rand.get("count")) and real.get("count") != rand.get("count"):
        problems.append(f"dense real/control counts differ: {real.get('count')} != {rand.get('count')}")
    if _both_known(real.get("tokens"), rand.get("tokens")) and real.get("tokens") != rand.get("tokens"):
        problems.append(
            f"dense real/control token counts differ: {real.get('tokens')} != {rand.get('tokens')}"
        )
    if _both_known(real.get("dim"), rand.get("dim")) and real.get("dim") != rand.get("dim"):
        problems.append(f"dense real/control dims differ: {real.get('dim')} != {rand.get('dim')}")
    if _both_known(real.get("keys_fingerprint"), rand.get("keys_fingerprint")) and real.get(
        "keys_fingerprint"
    ) != rand.get("keys_fingerprint"):
        problems.append("dense real/control referent keys do not match")
    problems.extend(_sidecar_problems(real, rand, "factors"))
    problems.extend(_sidecar_problems(real, rand, "splits"))
    return problems


def _sidecar_problems(real: dict[str, Any], rand: dict[str, Any], role: str) -> list[str]:
    left = real.get(f"{role[:-1] if role.endswith('s') else role}_sidecar")
    right = rand.get(f"{role[:-1] if role.endswith('s') else role}_sidecar")
    if left is None and right is None:
        return []
    if left is None or right is None:
        return [f"dense real/control {role} sidecar presence differs"]
    if left != right:
        return [f"dense real/control {role} sidecars differ"]
    return []


def _sidecar_match(real: dict[str, Any], rand: dict[str, Any], role: str) -> bool | None:
    left = real.get(f"{role[:-1] if role.endswith('s') else role}_sidecar")
    right = rand.get(f"{role[:-1] if role.endswith('s') else role}_sidecar")
    if left is None and right is None:
        return None
    return left == right


def _dense_token_count(feat_shape: list[int]) -> int | None:
    if len(feat_shape) < 2:
        return None
    tokens = 1
    for size in feat_shape[:-1]:
        tokens *= int(size)
    return tokens


def _records_by_role(records: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(records, list):
        for rec in records:
            if isinstance(rec, dict) and rec.get("role"):
                out[str(rec["role"])] = rec
    return out


def _fingerprint(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    if rec is None:
        return None
    return {
        "path": rec.get("path"),
        "bytes": rec.get("bytes"),
        "sha256": rec.get("sha256"),
        "hash_kind": rec.get("hash_kind"),
        "shape": rec.get("shape"),
        "dtype": rec.get("dtype"),
    }


def _eq_known(left: Any, right: Any) -> bool:
    return left is not None and right is not None and left == right


def _both_known(left: Any, right: Any) -> bool:
    return left is not None and right is not None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None
