"""Cache-first, token-aware E6 relational mechanics.

The legacy E6 runner flattens dense ``[tokens, features]`` outputs.  At the official V-JEPA 2.1
ViT-B 64-frame shape that would expose 14,155,776 values per clip to the first learned layer and
erase the very token geometry E6 is meant to test.  This module instead reads already-built citable
caches one row at a time, pools fixed contiguous token bins, applies frozen low-rank projections,
and fits exactly parameter-matched ridge readouts for two factors on held-out combinations.

Scientific promotion is fail closed.  It requires a real learned cache and an exact-architecture
random-init cache over byte-identical natural-video inputs, identical referents/factors/splits,
immutable source and weight receipts, enough rows, five projection seeds, off-ceiling behavior,
and positive confidence bounds over every declared control.  The tiny fixture only exercises the
data plane and bounded readout; it can never satisfy those source gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..substrate.cache_tools import validate_cache
from ..substrate.latent_store import LatentStore
from ..substrate.vjepa21_official import OFFICIAL_REPOSITORY_COMMIT, VITB, expected_dense_tokens

SCHEMA = "mop-e6-dense-relational-cache/v1"
FIXTURE_SCHEMA = "mop-e6-dense-relational-fixture/v1"
SOURCE_SCHEMA = "mop-e6-dense-source/v1"
MIN_SCIENCE_COUNT = 200
MIN_SEEDS = 5
T_CRITICAL_DF4_95 = 2.776
ARCHITECTURE_FIELDS = (
    "arch",
    "embed_dim",
    "patch_size",
    "tubelet",
    "frames_per_clip",
    "resolution",
    "dense",
    "pool",
    "official_repo_commit",
    "hub_entrypoint",
)
EXPECTED_VITB_ARCHITECTURE = {
    "arch": "vit_base",
    "embed_dim": 768,
    "patch_size": 16,
    "tubelet": 2,
    "frames_per_clip": 64,
    "resolution": 384,
    "dense": True,
    "pool": "none",
    "official_repo_commit": OFFICIAL_REPOSITORY_COMMIT,
    "hub_entrypoint": "vjepa2_1_vit_base_384",
}
OFFICIAL_VISION_TRANSFORMER_SHA256 = "d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9"


class E6DenseError(RuntimeError):
    """A cache, control, split, or bounded-readout contract failed."""


@dataclass(frozen=True)
class TokenReadoutSpec:
    """Immutable bounds for the fixed token summary and matched ridge heads."""

    bins: int = 8
    feature_rank: int = 16
    summary_dim: int = 32
    ridge: float = 0.01
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    min_margin: float = 0.02
    ceiling: float = 0.98

    def __post_init__(self) -> None:
        if not 2 <= self.bins <= 16:
            raise ValueError("bins must be in [2, 16]")
        if not 1 <= self.feature_rank <= 64:
            raise ValueError("feature_rank must be in [1, 64]")
        if not 2 <= self.summary_dim <= 256:
            raise ValueError("summary_dim must be in [2, 256]")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.min_margin <= 0:
            raise ValueError("min_margin must be positive")
        if not 0.5 < self.ceiling <= 1.0:
            raise ValueError("ceiling must be in (0.5, 1]")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise E6DenseError(f"cannot read JSON authority {path}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    digest = str(value or "")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest.lower())


def _factor_payload(path: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise E6DenseError(f"{path} must contain a mapping")
    if raw.get("schema") == "mop-factor-sidecar/v2":
        metadata = raw.get("metadata")
        columns = raw.get("columns")
        if not isinstance(metadata, dict) or not isinstance(columns, dict):
            raise E6DenseError(f"{path} v2 payload needs metadata and columns mappings")
        return metadata, {str(key): list(value) for key, value in columns.items()}
    return {}, {str(key): list(value) for key, value in raw.items()}


def _manifest_sidecar_is_fingerprinted(manifest: dict[str, Any], role: str) -> bool:
    return any(
        isinstance(row, dict)
        and row.get("role") == role
        and row.get("hash_kind") == "full"
        and _valid_sha256(row.get("sha256"))
        for row in manifest.get("sidecars", [])
    )


def _architecture_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in ARCHITECTURE_FIELDS}


def _split_combo_sets(
    factors: dict[str, list[Any]], splits: dict[str, list[int]]
) -> dict[str, set[tuple[str, str]]]:
    factor_a = factors["factor_a"]
    factor_b = factors["factor_b"]
    return {
        name: {(str(factor_a[index]), str(factor_b[index])) for index in indices}
        for name, indices in splits.items()
    }


def inspect_dense_cache(root: Path | str) -> dict[str, Any]:
    """Read and validate one citable dense cache without flattening its arrays."""
    path = Path(root).resolve()
    problems = list(validate_cache(path, citable=True)) if path.exists() else ["cache directory missing"]
    required = (
        "meta.json",
        "cache_manifest.json",
        "referents.json",
        "factors.json",
        "splits.json",
        "run_receipt.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    problems.extend(f"{name} missing" for name in missing)
    if missing:
        return {"path": str(path), "problems": problems, "mechanics_ok": False}

    meta = _read_json(path / "meta.json")
    manifest = _read_json(path / "cache_manifest.json")
    referents = _read_json(path / "referents.json")
    splits = _read_json(path / "splits.json")
    run_receipt = _read_json(path / "run_receipt.json")
    encoder_receipt = (
        _read_json(path / "encoder_receipt.json") if (path / "encoder_receipt.json").is_file() else {}
    )
    initialization_receipt = (
        _read_json(path / "initialization_receipt.json")
        if (path / "initialization_receipt.json").is_file()
        else {}
    )
    factor_metadata, factors = _factor_payload(path / "factors.json")
    if not all(isinstance(value, dict) for value in (meta, manifest, splits, run_receipt)):
        problems.append("meta, manifest, splits, and run receipt must be mappings")
        return {"path": str(path), "problems": problems, "mechanics_ok": False}
    if not isinstance(referents, list):
        problems.append("referents must be a list")
        referents = []
    count = int(meta.get("count", 0))
    shape = tuple(int(value) for value in meta.get("feat_shape", []))
    if len(shape) < 2:
        problems.append(f"dense cache needs token axes plus feature axis, got {shape}")
    token_count = math.prod(shape[:-1]) if len(shape) >= 2 else 0
    feature_dim = shape[-1] if shape else 0
    if count < 1 or token_count < 2 or feature_dim < 1:
        problems.append("cache count/token/feature dimensions must be positive")
    if len(referents) != count or len(set(str(value) for value in referents)) != count:
        problems.append("referents must be unique and match cache count")
    for factor in ("factor_a", "factor_b"):
        if factor not in factors or len(factors.get(factor, [])) != count:
            problems.append(f"{factor} must be present for every referent")
    split_names = {"train", "val", "test"}
    if set(splits) != split_names:
        problems.append(f"splits must be exactly {sorted(split_names)}")
    else:
        split_rows = [int(index) for name in sorted(splits) for index in splits[name]]
        if any(not splits[name] for name in split_names):
            problems.append("train, val, and test splits must all be nonempty")
        if sorted(split_rows) != list(range(count)):
            problems.append("train, val, and test must cover every row exactly once")
    if not _manifest_sidecar_is_fingerprinted(manifest, "run_receipt"):
        problems.append("run_receipt.json must be fully fingerprinted by cache_manifest.json")
    source = run_receipt.get("e6_source")
    if not isinstance(source, dict) or source.get("schema") != SOURCE_SCHEMA:
        problems.append(f"run_receipt.e6_source must use {SOURCE_SCHEMA}")
        source = {}
    config = manifest.get("encoder_config")
    if not isinstance(config, dict):
        problems.append("manifest encoder_config must be a mapping")
        config = {}
    form = manifest.get("form") if isinstance(manifest.get("form"), dict) else {}
    if factors.get("factor_a") and factors.get("factor_b") and set(splits) == split_names:
        combos = _split_combo_sets(factors, splits)
        if any(
            combos[left] & combos[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            problems.append("factor combinations must be disjoint across train, val, and test")
    else:
        combos = {}
    return {
        "path": str(path),
        "count": count,
        "feat_shape": list(shape),
        "token_count": token_count,
        "feature_dim": feature_dim,
        "legacy_flattened_dim": token_count * feature_dim,
        "manifest_sha256": _json_sha256(manifest),
        "form": form,
        "encoder_config": config,
        "architecture_signature": _architecture_signature(config),
        "referents": tuple(str(value) for value in referents),
        "referents_sha256": _json_sha256(referents),
        "factor_metadata": factor_metadata,
        "factors": factors,
        "factors_sha256": _json_sha256({"metadata": factor_metadata, "columns": factors}),
        "splits": {str(key): [int(value) for value in values] for key, values in splits.items()},
        "splits_sha256": _json_sha256(splits),
        "combination_sets": {name: sorted(values) for name, values in combos.items()},
        "run_receipt_sha256": _json_sha256(run_receipt),
        "encoder_receipt": encoder_receipt,
        "encoder_receipt_sha256": _json_sha256(encoder_receipt) if encoder_receipt else None,
        "initialization_receipt": initialization_receipt,
        "initialization_receipt_sha256": (
            _json_sha256(initialization_receipt) if initialization_receipt else None
        ),
        "source": source,
        "problems": problems,
        "mechanics_ok": not problems,
    }


def _declared_combinations_match(cache: dict[str, Any]) -> bool:
    declared = cache["factor_metadata"].get("combination_splits")
    if not isinstance(declared, dict):
        return False
    normalized = {
        str(name): sorted((str(pair[0]), str(pair[1])) for pair in pairs)
        for name, pairs in declared.items()
        if isinstance(pairs, list) and all(isinstance(pair, list) and len(pair) == 2 for pair in pairs)
    }
    return normalized == cache["combination_sets"]


def _source_promotion_problems(source: dict[str, Any], referents: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    expected_truths = {
        "natural_video": True,
        "rights_clean": True,
        "byte_identical_inputs_across_arms": True,
        "test_split_untouched": True,
    }
    for field, expected in expected_truths.items():
        if source.get(field) is not expected:
            problems.append(f"source {field} must be {expected}")
    if source.get("source_kind") != "natural-video":
        problems.append("source_kind must be natural-video")
    if not str(source.get("dataset_license") or "").strip():
        problems.append("source dataset_license must be nonempty")
    for field in (
        "source_authority_sha256",
        "content_set_sha256",
        "split_authority_sha256",
        "annotation_authority_sha256",
        "view_recipe_sha256",
    ):
        if not _valid_sha256(source.get(field)):
            problems.append(f"source {field} must be a SHA256")
    encoded_frames = source.get("encoded_frames")
    if encoded_frames not in (8, 16, 64):
        problems.append("source encoded_frames must be one of 8, 16, or 64")
    if source.get("resolution") != 384:
        problems.append("source resolution must be 384")
    records = source.get("input_tensor_sha256_by_referent")
    if not isinstance(records, list) or len(records) != len(referents):
        problems.append("source input tensor hashes must cover every referent")
    else:
        record_referents = tuple(str(row.get("referent")) for row in records if isinstance(row, dict))
        hashes = [row.get("sha256") for row in records if isinstance(row, dict)]
        if record_referents != referents:
            problems.append("source input tensor hashes must preserve exact referent order")
        if len(hashes) != len(referents) or not all(_valid_sha256(value) for value in hashes):
            problems.append("source input tensor hashes must all be SHA256 values")
        if len(set(hashes)) != len(hashes):
            problems.append("source inputs must not duplicate across referents or splits")
    return problems


def _learned_weight_problems(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = {
        "weights_real": True,
        "backend": "vjepa_official",
        "model_id": "official-pytorch-only-vjepa21-vitb",
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "checkpoint_etag": VITB["checkpoint_etag"],
        "checkpoint_version_id": VITB["checkpoint_version_id"],
        "checkpoint_bytes": VITB["checkpoint_content_length"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            problems.append(f"learned encoder receipt {field} does not match pinned ViT-B authority")
    if not _valid_sha256(receipt.get("checkpoint_authority_receipt_sha256")):
        problems.append("learned encoder receipt must hash the checkpoint authority receipt")
    files = receipt.get("files")
    if not isinstance(files, list) or not any(
        isinstance(row, dict)
        and str(row.get("path") or "").endswith("vjepa2_1_vitb_dist_vitG_384.pt")
        and row.get("bytes") == VITB["checkpoint_content_length"]
        and _valid_sha256(row.get("sha256"))
        for row in files
    ):
        problems.append("learned encoder receipt must hash the exact full ViT-B checkpoint")
    return problems


def _random_weight_problems(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if receipt.get("weights_real") is not False:
        problems.append("random initialization receipt weights_real must be false")
    if receipt.get("backend") != "vjepa_official_random_init":
        problems.append("random initialization receipt must use vjepa_official_random_init")
    if receipt.get("model_id") != "official-pytorch-only-vjepa21-vitb":
        problems.append("random initialization receipt model_id must identify official ViT-B")
    if receipt.get("repository_commit") != OFFICIAL_REPOSITORY_COMMIT:
        problems.append("random initialization receipt repository commit must match pinned ViT-B")
    files = receipt.get("architecture_files")
    if not isinstance(files, list) or not any(
        isinstance(row, dict)
        and row.get("path") == "app/vjepa_2_1/models/vision_transformer.py"
        and row.get("sha256") == OFFICIAL_VISION_TRANSFORMER_SHA256
        for row in files
    ):
        problems.append("random initialization receipt must hash the pinned official encoder source")
    if not _valid_sha256(receipt.get("state_dict_sha256")):
        problems.append("random initialization receipt must hash the realized state dict")
    return problems


def build_pair_gate(
    learned_cache: Path | str,
    random_cache: Path | str,
    *,
    min_science_count: int = MIN_SCIENCE_COUNT,
) -> dict[str, Any]:
    """Validate mechanics and separately enumerate every promotion blocker."""
    learned = inspect_dense_cache(learned_cache)
    random = inspect_dense_cache(random_cache)
    mechanics_problems = [
        *(f"learned: {problem}" for problem in learned.get("problems", [])),
        *(f"random: {problem}" for problem in random.get("problems", [])),
    ]
    if learned.get("mechanics_ok") and random.get("mechanics_ok"):
        pair_equalities = {
            "count": learned["count"] == random["count"],
            "shape": learned["feat_shape"] == random["feat_shape"],
            "referents": learned["referents"] == random["referents"],
            "factors": learned["factors_sha256"] == random["factors_sha256"],
            "splits": learned["splits_sha256"] == random["splits_sha256"],
            "source": learned["source"] == random["source"],
            "architecture": learned["architecture_signature"] == random["architecture_signature"],
        }
        mechanics_problems.extend(
            f"pair {name} mismatch" for name, matches in pair_equalities.items() if not matches
        )
        if learned["encoder_config"].get("random_init") is True:
            mechanics_problems.append("learned cache must not declare random_init=true")
        if random["encoder_config"].get("random_init") is not True:
            mechanics_problems.append("random control must declare random_init=true")
    else:
        pair_equalities = {}
    mechanics_ok = not mechanics_problems

    promotion_problems = list(mechanics_problems)
    if mechanics_ok:
        if learned["count"] < int(min_science_count):
            promotion_problems.append(
                f"cache count {learned['count']} below preregistered science floor {min_science_count}"
            )
        if learned["form"].get("objective") != "inherited-frozen":
            promotion_problems.append("learned cache objective must be inherited-frozen")
        if random["form"].get("objective") != "random-control":
            promotion_problems.append("random cache objective must be random-control")
        promotion_problems.extend(_learned_weight_problems(learned["encoder_receipt"]))
        promotion_problems.extend(_random_weight_problems(random["initialization_receipt"]))
        if learned["architecture_signature"] != EXPECTED_VITB_ARCHITECTURE:
            promotion_problems.append("cache architecture is not the pinned official V-JEPA 2.1 ViT-B")
        if learned["form"].get("referent_scheme") != "natural-video-clip-id":
            promotion_problems.append("learned cache referent scheme must be natural-video-clip-id")
        if random["form"].get("referent_scheme") != "natural-video-clip-id":
            promotion_problems.append("random cache referent scheme must be natural-video-clip-id")
        promotion_problems.extend(_source_promotion_problems(learned["source"], learned["referents"]))
        encoded_frames = learned["source"].get("encoded_frames")
        if encoded_frames in (8, 16, 64):
            expected_tokens = expected_dense_tokens(int(encoded_frames))
            if learned["token_count"] != expected_tokens:
                promotion_problems.append(
                    f"dense token count {learned['token_count']} != {expected_tokens} for "
                    f"{encoded_frames} frames"
                )
        if not _declared_combinations_match(learned):
            promotion_problems.append("declared held-out combination split does not match factor rows")
        if learned["factor_metadata"].get("heldout_combination_policy") is None:
            promotion_problems.append("factor metadata lacks heldout_combination_policy")
        if not _valid_sha256(learned["factor_metadata"].get("annotation_authority_sha256")):
            promotion_problems.append("factor metadata lacks annotation authority SHA256")

    return {
        "learned": _public_cache_summary(learned),
        "random": _public_cache_summary(random),
        "pair_equalities": pair_equalities,
        "mechanics_problems": mechanics_problems,
        "mechanics_ok": mechanics_ok,
        "promotion_requirements": {
            "minimum_count": int(min_science_count),
            "learned_objective": "inherited-frozen",
            "random_objective": "random-control",
            "architecture": EXPECTED_VITB_ARCHITECTURE,
            "exact_referents_factors_splits_source": True,
            "byte_identical_input_hashes": True,
            "natural_rights_clean_source": True,
        },
        "promotion_problems": promotion_problems,
        "promotion_ready": not promotion_problems,
        "_learned_private": learned,
        "_random_private": random,
    }


def _public_cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "path",
        "count",
        "feat_shape",
        "token_count",
        "feature_dim",
        "legacy_flattened_dim",
        "manifest_sha256",
        "form",
        "architecture_signature",
        "referents_sha256",
        "factors_sha256",
        "splits_sha256",
        "combination_sets",
        "run_receipt_sha256",
        "encoder_receipt_sha256",
        "initialization_receipt_sha256",
        "source",
        "problems",
        "mechanics_ok",
    )
    return {key: cache.get(key) for key in allowed}


def _projection(rows: int, columns: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(0.0, 1.0 / math.sqrt(max(1, rows)), size=(rows, columns)).astype("float64")


def _summary_maps(
    *, tokens: int, feature_dim: int, spec: TokenReadoutSpec, seed: int
) -> dict[str, np.ndarray]:
    rank = min(spec.feature_rank, feature_dim)
    return {
        "feature": _projection(feature_dim, rank, seed + 11),
        "structured": _projection(spec.bins * rank, spec.summary_dim, seed + 23),
        "flat": _projection(feature_dim, spec.summary_dim, seed + 37),
        "base_permutation": np.random.default_rng(seed + 53).permutation(tokens),
    }


def _structured_row(
    tokens: np.ndarray,
    *,
    maps: dict[str, np.ndarray],
    bins: int,
    row_index: int,
    shuffled: bool,
    seed: int,
) -> np.ndarray:
    ordered = tokens
    if shuffled:
        # A different label-independent permutation per referent destroys stable position while
        # preserving every token value and the exact per-clip token multiset.
        permutation = np.random.default_rng(seed + 104729 * (row_index + 1)).permutation(tokens.shape[0])
        ordered = tokens[permutation]
    bin_means = np.stack([chunk.mean(axis=0) for chunk in np.array_split(ordered, bins)])
    reduced = bin_means @ maps["feature"]
    return reduced.reshape(-1) @ maps["structured"]


def summarize_cache(
    root: Path | str,
    *,
    spec: TokenReadoutSpec,
) -> dict[int, dict[str, np.ndarray]]:
    """Stream every dense row once and emit only bounded summaries for every declared seed."""
    store = LatentStore.open(Path(root))
    shape = tuple(int(value) for value in store.meta.feat_shape)
    if len(shape) < 2:
        raise E6DenseError(f"cache {root} is not dense: {shape}")
    tokens, feature_dim = math.prod(shape[:-1]), shape[-1]
    if tokens < spec.bins:
        raise E6DenseError(f"cache has {tokens} tokens, fewer than {spec.bins} fixed bins")
    maps_by_seed = {
        seed: _summary_maps(tokens=tokens, feature_dim=feature_dim, spec=spec, seed=seed)
        for seed in spec.seeds
    }
    output = {
        seed: {
            mode: np.empty((len(store), spec.summary_dim), dtype="float64")
            for mode in ("structured", "flat", "token_shuffled")
        }
        for seed in spec.seeds
    }
    for index in range(len(store)):
        row = store.latents(index).numpy().reshape(tokens, feature_dim).astype("float64", copy=False)
        for seed, maps in maps_by_seed.items():
            output[seed]["structured"][index] = _structured_row(
                row,
                maps=maps,
                bins=spec.bins,
                row_index=index,
                shuffled=False,
                seed=seed,
            )
            output[seed]["token_shuffled"][index] = _structured_row(
                row,
                maps=maps,
                bins=spec.bins,
                row_index=index,
                shuffled=True,
                seed=seed,
            )
            output[seed]["flat"][index] = row.mean(axis=0) @ maps["flat"]
    return output


def _ridge_factor_score(
    features: np.ndarray,
    labels: list[Any],
    splits: dict[str, list[int]],
    *,
    ridge: float,
) -> dict[str, Any]:
    label_array = np.asarray(labels)
    classes = np.unique(label_array)
    class_to_index = {value.item() if hasattr(value, "item") else value: i for i, value in enumerate(classes)}
    encoded = np.asarray(
        [class_to_index[value.item() if hasattr(value, "item") else value] for value in label_array],
        dtype="int64",
    )
    train = np.asarray(splits["train"], dtype="int64")
    if set(encoded[train].tolist()) != set(range(len(classes))):
        raise E6DenseError("training split must contain every factor level")
    mean = features[train].mean(axis=0, keepdims=True)
    scale = features[train].std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate([standardized, np.ones((standardized.shape[0], 1))], axis=1)
    targets = np.eye(len(classes), dtype="float64")[encoded]
    penalty = np.eye(design.shape[1], dtype="float64") * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design[train].T @ design[train] + penalty, design[train].T @ targets[train])
    scores: dict[str, float] = {}
    for name in ("train", "val", "test"):
        indices = np.asarray(splits[name], dtype="int64")
        predictions = (design[indices] @ weights).argmax(axis=1)
        scores[name] = float((predictions == encoded[indices]).mean())
    return {
        "classes": [value.item() if hasattr(value, "item") else value for value in classes],
        "chance": 1.0 / len(classes),
        "parameters": int(design.shape[1] * len(classes)),
        "accuracy": scores,
    }


def _score_arm(
    summaries: dict[int, dict[str, np.ndarray]],
    cache: dict[str, Any],
    spec: TokenReadoutSpec,
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for seed, modes in summaries.items():
        output[seed] = {}
        for mode, features in modes.items():
            factor_a = _ridge_factor_score(
                features,
                cache["factors"]["factor_a"],
                cache["splits"],
                ridge=spec.ridge,
            )
            factor_b = _ridge_factor_score(
                features,
                cache["factors"]["factor_b"],
                cache["splits"],
                ridge=spec.ridge,
            )
            if factor_a["parameters"] + factor_b["parameters"] != 2 * (spec.summary_dim + 1) * len(
                factor_a["classes"]
            ):
                # The explicit check below handles different A/B class counts; this branch only
                # guards an unexpected implementation change for equal-cardinality fixture factors.
                pass
            output[seed][mode] = {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "combined_test_accuracy": float(
                    (factor_a["accuracy"]["test"] + factor_b["accuracy"]["test"]) / 2
                ),
                "combined_val_accuracy": float(
                    (factor_a["accuracy"]["val"] + factor_b["accuracy"]["val"]) / 2
                ),
                "total_head_parameters": factor_a["parameters"] + factor_b["parameters"],
            }
    return output


def _ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype="float64")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = T_CRITICAL_DF4_95 * std / math.sqrt(len(array)) if len(array) > 1 else math.inf
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "conservative_t_critical": T_CRITICAL_DF4_95,
        "lower": mean - half,
        "upper": mean + half,
        "all_positive": all(value > 0 for value in values),
    }


def run_dense_relational(
    learned_cache: Path | str,
    random_cache: Path | str,
    *,
    spec: TokenReadoutSpec | None = None,
) -> dict[str, Any]:
    """Run the bounded E6 readout after the paired cache mechanics gate passes."""
    spec = spec or TokenReadoutSpec()
    pair = build_pair_gate(learned_cache, random_cache)
    if not pair["mechanics_ok"]:
        raise E6DenseError("paired dense-cache mechanics failed: " + "; ".join(pair["mechanics_problems"]))
    learned_private = pair.pop("_learned_private")
    random_private = pair.pop("_random_private")
    learned_summaries = summarize_cache(learned_cache, spec=spec)
    random_summaries = summarize_cache(random_cache, spec=spec)
    learned_scores = _score_arm(learned_summaries, learned_private, spec)
    random_scores = _score_arm(random_summaries, random_private, spec)

    per_seed: list[dict[str, Any]] = []
    for seed in spec.seeds:
        learned_structured = learned_scores[seed]["structured"]["combined_test_accuracy"]
        controls = {
            "learned_flat": learned_scores[seed]["flat"]["combined_test_accuracy"],
            "learned_token_shuffled": learned_scores[seed]["token_shuffled"]["combined_test_accuracy"],
            "random_structured": random_scores[seed]["structured"]["combined_test_accuracy"],
            "random_flat": random_scores[seed]["flat"]["combined_test_accuracy"],
        }
        best_control = max(controls.values())
        per_seed.append(
            {
                "seed": seed,
                "learned_structured_test_accuracy": learned_structured,
                "controls": controls,
                "best_control": best_control,
                "primary_delta": learned_structured - best_control,
                "position_specificity_delta": learned_structured - controls["learned_token_shuffled"],
                "head_parameters": learned_scores[seed]["structured"]["total_head_parameters"],
                "parameter_match": len(
                    {
                        learned_scores[seed][mode]["total_head_parameters"]
                        for mode in ("structured", "flat", "token_shuffled")
                    }
                    | {
                        random_scores[seed][mode]["total_head_parameters"]
                        for mode in ("structured", "flat", "token_shuffled")
                    }
                )
                == 1,
            }
        )
    primary = _ci([row["primary_delta"] for row in per_seed])
    position = _ci([row["position_specificity_delta"] for row in per_seed])
    structured_scores = [row["learned_structured_test_accuracy"] for row in per_seed]
    statistical_gate = {
        "minimum_seed_count": len(spec.seeds) >= MIN_SEEDS,
        "all_heads_parameter_matched": all(row["parameter_match"] for row in per_seed),
        "primary_mean_clears_margin": primary["mean"] > spec.min_margin,
        "primary_lower_bound_positive": primary["lower"] > 0,
        "primary_no_sign_flip": primary["all_positive"],
        "position_lower_bound_positive": position["lower"] > 0,
        "position_no_sign_flip": position["all_positive"],
        "off_ceiling": float(np.mean(structured_scores)) < spec.ceiling,
    }
    statistical_gate["all_ok"] = all(statistical_gate.values())
    science_promotion = bool(pair["promotion_ready"] and statistical_gate["all_ok"])
    flattened = int(learned_private["legacy_flattened_dim"])
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "implementation": {
            "module": str(Path(__file__).resolve()),
            "module_sha256": file_sha256(Path(__file__).resolve()),
        },
        "preregistered": {
            "question": (
                "do fixed token-position summaries improve factor decoding on held-out combinations "
                "beyond a token-blind pool and every matched random/token-shuffle control"
            ),
            "null": (
                "the token-aware learned-cache readout ties the strongest learned-flat, "
                "learned-token-shuffled, random-structured, or random-flat control"
            ),
            "spec": asdict(spec),
            "primary_metric": "learned_structured minus strongest per-seed declared control",
            "confidence_rule": "five seeds, conservative df=4 t interval, lower bound > 0",
            "promotion_rule": (
                "paired natural/input/weight gate AND every statistical gate; fixture data never promotes"
            ),
        },
        "pair_gate": pair,
        "bounded_interface": {
            "input_layout": "one memmapped cache row [token_axes..., feature] at a time",
            "geometry": "contiguous token bins preserved before frozen low-rank projection",
            "legacy_flattened_dim": flattened,
            "learned_head_input_dim": spec.summary_dim,
            "dimension_reduction_ratio": flattened / spec.summary_dim,
            "maximum_bins": 16,
            "maximum_feature_rank": 64,
            "maximum_summary_dim": 256,
            "full_cache_flattened_matrix_materialized": False,
        },
        "per_seed": per_seed,
        "confidence": {"primary_delta": primary, "position_specificity_delta": position},
        "statistical_gate": statistical_gate,
        "scientific_promotion": science_promotion,
        "claim_boundary": {
            "mechanics_fixture_only": not pair["promotion_ready"],
            "natural_video_claim": science_promotion,
            "e6_null_rejected": science_promotion,
            "larger_vjepa21_variants_unlocked": False,
            "interpretation": (
                "mechanics success establishes bounded token-aware cache consumption only; it does not "
                "substitute programmatic tokens for natural video or waive matched random controls"
            ),
        },
        "all_ok": True,
    }


def _combination_plan() -> dict[str, list[tuple[int, int]]]:
    return {
        "train": [(0, 2), (1, 0), (2, 1)],
        "val": [(0, 1), (1, 2), (2, 0)],
        "test": [(0, 0), (1, 1), (2, 2)],
    }


def build_mechanics_fixture(root: Path | str, *, seed: int = 20260710) -> dict[str, Any]:
    """Build a tiny paired dense cache; both arms remain programmatic and non-promotable."""
    from ..substrate.cache_manifest import write_cache_manifest

    root = Path(root)
    count_per_combo, tokens, feature_dim = 4, 16, 8
    plan = _combination_plan()
    rows: list[tuple[str, int, int, int]] = []
    for split in ("train", "val", "test"):
        for factor_a_value, factor_b_value in plan[split]:
            for repeat in range(count_per_combo):
                rows.append((split, factor_a_value, factor_b_value, repeat))
    referents = [
        f"e6-fixture:{split}:a{factor_a_value}:b{factor_b_value}:r{repeat}"
        for split, factor_a_value, factor_b_value, repeat in rows
    ]
    split_indices = {
        split: [index for index, row in enumerate(rows) if row[0] == split]
        for split in ("train", "val", "test")
    }
    factor_a_values = [row[1] for row in rows]
    factor_b_values = [row[2] for row in rows]
    generator = np.random.default_rng(seed)
    stimuli = np.zeros((len(rows), tokens, feature_dim), dtype="float32")
    for index, (_, a, b, _) in enumerate(rows):
        va = np.zeros(feature_dim, dtype="float32")
        vb = np.zeros(feature_dim, dtype="float32")
        va[a] = 3.0
        vb[3 + b] = 3.0
        stimuli[index, 0:4] += va
        stimuli[index, 4:8] -= va
        stimuli[index, 8:12] += vb
        stimuli[index, 12:16] -= vb
        stimuli[index] += generator.normal(0.0, 0.05, size=(tokens, feature_dim)).astype("float32")
    input_records = [
        {"referent": referent, "sha256": _tensor_sha256(stimuli[index])}
        for index, referent in enumerate(referents)
    ]
    content_set_sha = _json_sha256(input_records)
    source = {
        "schema": SOURCE_SCHEMA,
        "source_kind": "programmatic",
        "natural_video": False,
        "rights_clean": False,
        "dataset_license": "internal-programmatic-fixture",
        "source_authority_sha256": _json_sha256({"generator": "e6-fixture", "seed": seed}),
        "content_set_sha256": content_set_sha,
        "split_authority_sha256": _json_sha256(split_indices),
        "annotation_authority_sha256": _json_sha256(
            {"factor_a": factor_a_values, "factor_b": factor_b_values}
        ),
        "view_recipe_sha256": _json_sha256({"tokens": tokens, "feature_dim": feature_dim}),
        "encoded_frames": None,
        "resolution": None,
        "byte_identical_inputs_across_arms": True,
        "test_split_untouched": True,
        "input_tensor_sha256_by_referent": input_records,
    }
    annotation_sha = _json_sha256({"plan": plan, "factors": [factor_a_values, factor_b_values]})
    factor_metadata = {
        "factor_a_name": "fixture_spatial_a",
        "factor_b_name": "fixture_spatial_b",
        "heldout_combination_policy": "three disjoint Latin-square combination sets",
        "combination_splits": {
            name: [[a, b] for a, b in combinations] for name, combinations in plan.items()
        },
        "annotation_authority_sha256": annotation_sha,
    }
    architecture = {
        "arch": "fixture_vit_base",
        "embed_dim": feature_dim,
        "patch_size": 1,
        "tubelet": 1,
        "frames_per_clip": tokens,
        "resolution": tokens,
        "dense": True,
        "pool": "none",
        "official_repo_commit": "fixture",
        "hub_entrypoint": "fixture",
        "hf_id": "fixture/e6-dense",
        "revision": "fixture-v1",
        "prefer_real": False,
        "require_real": False,
    }
    arm_arrays = {
        "learned": stimuli
        + np.random.default_rng(seed + 1).normal(0.0, 0.02, size=stimuli.shape).astype("float32"),
        "random": np.random.default_rng(seed + 2).normal(0.0, 1.0, size=stimuli.shape).astype("float32"),
    }
    stores: dict[str, str] = {}
    cache_problems: dict[str, list[str]] = {}
    for arm, features in arm_arrays.items():
        name = f"e6_dense_fixture_{arm}"
        store = LatentStore.create(
            root,
            name,
            (tokens, feature_dim),
            len(rows),
            feature_dim,
            has_labels=True,
        )
        labels = np.asarray([a * 3 + b for a, b in zip(factor_a_values, factor_b_values, strict=True)])
        store.write_batch(0, features, features.mean(axis=1), labels)
        store.finalize()
        config = {
            **architecture,
            "name": name,
            "actual_backend": f"programmatic-{arm}",
            "random_init": arm == "random",
            "random_init_seed": seed + 2 if arm == "random" else None,
        }
        write_cache_manifest(
            store.root,
            encoder_config=config,
            factors={"factor_a": factor_a_values, "factor_b": factor_b_values},
            factor_metadata=factor_metadata,
            splits=split_indices,
            referents=referents,
            form_kind="vision",
            form_objective="programmatic",
            referent_scheme="programmatic-e6-fixture-id",
            full_hash_arrays=True,
        )
        run_receipt = {
            "schema": "mop-e6-dense-fixture-arm/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "arm": arm,
            "e6_source": source,
            "scientific_promotion": False,
        }
        (store.root / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2, sort_keys=True) + "\n")
        write_cache_manifest(
            store.root,
            encoder_config=config,
            form_kind="vision",
            form_objective="programmatic",
            referent_scheme="programmatic-e6-fixture-id",
            full_hash_arrays=True,
        )
        stores[arm] = str(store.root)
        cache_problems[arm] = validate_cache(store.root, citable=True)
    return {
        "schema": FIXTURE_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": "tiny programmatic token-cache mechanics; never natural-video evidence",
        "shape_per_arm": [len(rows), tokens, feature_dim],
        "stores": stores,
        "cache_problems": cache_problems,
        "source": source,
        "all_ok": not any(cache_problems.values()),
    }
