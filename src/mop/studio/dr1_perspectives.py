"""DR1 PerspectiveMatrix receipts.

The real DR1 cache should not hand-wave "vision plus captions" as aligned. This module verifies that the
merged latent store rows and paired captions share the same referent ids, builds the existing
PerspectiveMatrix contract, and writes a compact JSON receipt with the audit surface.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..perspectives import (
    LatentStorePerspectiveAdapter,
    PerspectiveMeta,
    TensorPerspectiveAdapter,
    build_perspective_matrix,
    perspective_audit,
)
from ..substrate import LatentStore

SCHEMA = "mop-dr1-perspective-matrix-receipt/v1"
DEFAULT_OUT_NAME = "perspective_matrix_receipt.json"
CAPTION_DIM = 256


def build_dr1_perspective_receipt(
    store_dir: Path | str,
    captions: Mapping[str, str],
    *,
    factors: Sequence[str],
    cell_delim: str = "-",
) -> dict[str, Any]:
    """Build a receipt proving the merged DR1 store and captions align by referent id."""
    root = Path(store_dir)
    store = LatentStore.open(root)
    stems = _load_stems(root)
    if len(stems) != len(store):
        raise ValueError(f"store has {len(store)} rows but clip_stems.json has {len(stems)} referents")
    missing_caps = [stem for stem in stems if stem not in captions]
    if missing_caps:
        raise ValueError(f"captions missing for {len(missing_caps)} referent(s), first={missing_caps[:5]}")

    stem_to_cell = _load_cells(root, stems)
    factor_values = _factor_values(stem_to_cell, stems, factors, cell_delim)
    caption_texts = [str(captions[stem]) for stem in stems]
    caption_features = _caption_features(caption_texts)
    factor_tensors = {name: torch.tensor(values, dtype=torch.long) for name, values in factor_values.items()}

    vision = LatentStorePerspectiveAdapter(
        store,
        tag="vision_vjepa2",
        modality="vision",
        source=str(root),
        referents=stems,
        factors=factor_tensors,
        license="source-cache",
        notes="DR1 merged V-JEPA latent store",
    )
    caption = TensorPerspectiveAdapter(
        PerspectiveMeta(
            tag="caption_text",
            modality="language",
            feature_dim=int(caption_features.shape[1]),
            source="captions.json",
            derived=True,
            license="source-sidecar",
            factors=tuple(factors),
            notes="paired real caption sidecar, hashed trigram features for receipt alignment",
        ),
        caption_features,
        stems,
        factors=factor_tensors,
    )
    matrix = build_perspective_matrix([vision, caption])
    audit = perspective_audit(matrix)
    return {
        "schema": SCHEMA,
        "ok": True,
        "store": str(root),
        "n_referents": len(stems),
        "referent_sha256": _sha_json(stems),
        "tags": matrix.tags(),
        "audit": audit,
        "arms": {tag: asdict(matrix.metadata[tag]) for tag in matrix.tags()},
        "factor_values": _factor_value_names(stem_to_cell, stems, factors, cell_delim),
        "factor_counts": _factor_counts(stem_to_cell, stems, factors, cell_delim),
        "notes": [
            "receipt verifies referent alignment only; it is not a positive result",
            "missing_controls in the audit must stay visible until matched controls are added",
        ],
    }


def write_dr1_perspective_receipt(
    store_dir: Path | str,
    captions: Mapping[str, str],
    *,
    factors: Sequence[str],
    out_path: Path | str | None = None,
    cell_delim: str = "-",
) -> dict[str, Any]:
    """Build and write the DR1 PerspectiveMatrix receipt."""
    root = Path(store_dir)
    receipt = build_dr1_perspective_receipt(root, captions, factors=factors, cell_delim=cell_delim)
    out = Path(out_path) if out_path is not None else root / DEFAULT_OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt["path"] = str(out)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return receipt


def _load_stems(root: Path) -> tuple[str, ...]:
    path = root / "clip_stems.json"
    if not path.exists():
        raise ValueError(f"missing {path}; DR1 merge must persist store row order")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty list")
    stems = tuple(str(x) for x in data)
    if len(set(stems)) != len(stems):
        raise ValueError("clip_stems.json contains duplicate referents")
    return stems


def _load_cells(root: Path, stems: Sequence[str]) -> dict[str, str]:
    path = root / "clip_cells.json"
    if not path.exists():
        raise ValueError(f"missing {path}; DR1 merge must persist stem to cell mapping")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    out = {str(k): str(v) for k, v in data.items()}
    missing = [stem for stem in stems if stem not in out]
    if missing:
        raise ValueError(f"clip_cells.json missing {len(missing)} referent(s), first={missing[:5]}")
    return out


def _parse_cell(cell: str, factors: Sequence[str], cell_delim: str) -> dict[str, str]:
    parts = cell.split(cell_delim)
    if len(parts) != len(factors) or not all(parts):
        raise ValueError(f"cell {cell!r} does not name factors {tuple(factors)!r}")
    return dict(zip((str(f) for f in factors), parts, strict=True))


def _factor_values(
    stem_to_cell: Mapping[str, str],
    stems: Sequence[str],
    factors: Sequence[str],
    cell_delim: str,
) -> dict[str, list[int]]:
    parsed = [_parse_cell(stem_to_cell[stem], factors, cell_delim) for stem in stems]
    out: dict[str, list[int]] = {}
    for factor in factors:
        vals = sorted({p[str(factor)] for p in parsed})
        idx = {value: i for i, value in enumerate(vals)}
        out[str(factor)] = [idx[p[str(factor)]] for p in parsed]
    return out


def _factor_value_names(
    stem_to_cell: Mapping[str, str],
    stems: Sequence[str],
    factors: Sequence[str],
    cell_delim: str,
) -> dict[str, list[str]]:
    parsed = [_parse_cell(stem_to_cell[stem], factors, cell_delim) for stem in stems]
    return {str(factor): sorted({p[str(factor)] for p in parsed}) for factor in factors}


def _factor_counts(
    stem_to_cell: Mapping[str, str],
    stems: Sequence[str],
    factors: Sequence[str],
    cell_delim: str,
) -> dict[str, dict[str, int]]:
    parsed = [_parse_cell(stem_to_cell[stem], factors, cell_delim) for stem in stems]
    return {str(factor): dict(sorted(Counter(p[str(factor)] for p in parsed).items())) for factor in factors}


def _caption_features(captions: Sequence[str], dim: int = CAPTION_DIM) -> torch.Tensor:
    feats = torch.zeros(len(captions), dim)
    for i, cap in enumerate(captions):
        s = str(cap).lower()
        for j in range(len(s) - 2):
            feats[i, _stable_hash(s[j : j + 3]) % dim] += 1.0
    norms = feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return feats / norms


def _stable_hash(s: str) -> int:
    h = 2166136261
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _sha_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
