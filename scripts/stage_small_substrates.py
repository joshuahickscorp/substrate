#!/usr/bin/env python
"""WP-01 staging (Q0.1): download the three SMALL substrates to the HF disk cache and write the sha
manifest. Network-only, RAM <200MB, NO model is ever loaded into memory here (downloads are the only
thing licensed to run alongside the in-flight V-JEPA encode). Everything staged here backs up and
transfers to the Studio.

Exact ids and sizes per the manifest: facebook/dinov2-small (~90MB), Qwen/Qwen2.5-0.5B (~1.0GB, fallback
HuggingFaceTB/SmolLM2-360M ~0.7GB), facebook/wav2vec2-base (~380MB). Total ~1.5GB against the 51GB free
floor; nothing larger is permitted (disk is below the 60GB floor).

Usage: python scripts/stage_small_substrates.py   -> runs/mot/staging_manifest.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from devsys.config import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "runs" / "mot" / "staging_manifest.json"
MIN_FREE_GB = 10  # refuse to download into a nearly-full disk
IGNORE_COMMON = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"]

MODELS = [
    # role, primary repo id, fallback repo id, extra ignore patterns (skip duplicate weight formats)
    ("image_encoder", "facebook/dinov2-small", None, ["*.bin"]),
    ("text_encoder", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M", ["*.bin"]),
    ("audio_encoder", "facebook/wav2vec2-base", None, []),  # wav2vec2-base ships .bin only, keep it
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_manifest(repo_id: str, ignore: list[str]) -> dict:
    path = Path(snapshot_download(repo_id, ignore_patterns=IGNORE_COMMON + ignore))
    files = {}
    total = 0
    for p in sorted(path.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            files[str(p.relative_to(path))] = {"bytes": size, "sha256": _sha256(p)}
            total += size
    return {
        "repo_id": repo_id,
        "path": str(path),
        "revision": path.name,
        "n_files": len(files),
        "total_bytes": total,
        "files": files,
    }


def stage_all() -> dict:
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"refusing to download: only {free_gb:.1f}GB free (< {MIN_FREE_GB}GB floor)")
    out: dict = {"staged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}, "fallbacks_used": []}
    for role, repo_id, fallback, ignore in MODELS:
        try:
            entry = _snapshot_manifest(repo_id, ignore)
        except Exception as e:  # network or gating failure: stage the declared fallback
            if fallback is None:
                raise
            print(f"{repo_id} failed ({e}); staging fallback {fallback}", flush=True)
            entry = _snapshot_manifest(fallback, ignore)
            entry["fallback_for"] = repo_id
            out["fallbacks_used"].append(role)
        out["models"][role] = entry
        print(
            f"staged {role}: {entry['repo_id']} rev={entry['revision']} "
            f"{entry['total_bytes'] / 1e6:.0f}MB in {entry['n_files']} files",
            flush=True,
        )
    out["total_bytes"] = sum(m["total_bytes"] for m in out["models"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stage the WP-01 small substrates (network-only, no loads)")
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    a = ap.parse_args(argv)
    manifest = stage_all()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {p} (total {manifest['total_bytes'] / 1e9:.2f}GB staged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
