#!/usr/bin/env python
"""Build a deterministic dense programmatic Form cache with full citable sidecars.

This is a data-plane acceptance fixture, not natural-form evidence. It exists so fresh-clone and
pre-Studio checks can exercise dense geometry, referents, factors, splits, manifests, and FormMatrix
construction before any licensed video is available.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mop.config import REPO_ROOT
from mop.substrate.cache_manifest import validate_cache_manifest, write_cache_manifest
from mop.substrate.cache_tools import validate_cache
from mop.substrate.latent_store import LatentStore
from mop.substrate.events import sha256_file

SCHEMA = "mop-programmatic-form-cache-receipt/v1"


def build_fixture(root: Path, *, name: str, count: int, seed: int) -> dict:
    if count < 10:
        raise ValueError("fixture count must be at least 10")
    target = root / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing fixture {target}")
    rng = np.random.default_rng(seed)
    classes, tokens, dim = 5, 4, 8
    labels = np.arange(count, dtype="int64") % classes
    centers = rng.normal(0.0, 1.0, size=(classes, tokens, dim)).astype("float32")
    features = centers[labels] + rng.normal(0.0, 0.35, size=(count, tokens, dim)).astype("float32")
    keys = features.mean(axis=1)
    store = LatentStore.create(
        root,
        name,
        (tokens, dim),
        count,
        dim,
        has_labels=True,
    )
    store.write_batch(0, features, keys, labels)
    store.finalize()
    order = np.random.default_rng(seed + 1).permutation(count).tolist()
    train_end, val_end = int(count * 0.6), int(count * 0.8)
    manifest = write_cache_manifest(
        store.root,
        encoder_config={
            "name": "deterministic-programmatic-token-generator",
            "revision": "v1",
            "script": "scripts/build_citable_form_fixture.py",
            "seed": seed,
            "tokens": tokens,
            "dim": dim,
        },
        factors={
            "class": labels.tolist(),
            "phase": (np.arange(count) % tokens).tolist(),
        },
        factor_metadata={"classes": classes, "generator_seed": seed},
        splits={
            "train": order[:train_end],
            "val": order[train_end:val_end],
            "test": order[val_end:],
        },
        referents=[f"programmatic-token-{index:05d}" for index in range(count)],
        form_kind="symbolic",
        form_objective="programmatic",
        referent_scheme="programmatic-token-id",
        full_hash_arrays=True,
    )
    problems = validate_cache(store.root, citable=True)
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": "data-plane acceptance fixture; not natural-form evidence",
        "store": str(store.root),
        "count": count,
        "shape": [count, tokens, dim],
        "manifest_schema": manifest["schema"],
        "manifest_sha256": sha256_file(store.root / "cache_manifest.json"),
        "strict_manifest_problems": validate_cache_manifest(store.root, citable=True),
        "cache_problems": problems,
        "all_ok": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/cache")
    parser.add_argument("--name", default="programmatic_form_fixture_v1")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "proof/PROGRAMMATIC_FORM_CACHE.json")
    args = parser.parse_args(argv)
    receipt = build_fixture(args.root, name=args.name, count=args.count, seed=args.seed)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
