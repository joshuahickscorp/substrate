"""Disk accounting for the latent substrate: estimate a cache before you compute it, list
caches with their on-disk size, prune old ones SAFELY. The cache is the laptop-feasibility
keystone (encode once, train forever), so knowing its footprint up front, and being able to
reclaim it without ever fat-fingering a delete, is part of the contract.

A cache on disk is <root>/<name>/{latents.npy, keys.npy, meta.json, labels.npy?, provenance.json}.
The bytes are dominated by three memmap arrays: latents [N, *feat], keys [N, key_dim],
labels [N]. Each .npy carries a tiny fixed header (~128 bytes); estimate_cache_bytes returns the
raw array bytes (the load-bearing term) and the header is rounding error at any real N. dtype
widths: float32=4, float16=2, int64=8 (labels are always int64).

Expected disk for natural-video caches (per clip), pooled vs dense
------------------------------------------------------------------
Pooled = one embed_dim vector per clip; the keys array duplicates it (key_dim==embed_dim), so
pooled latents+keys cost 2*embed_dim*4 bytes/clip in float32. Dense keeps every token: a
64-frame ViT-L/16 at 256px tiles to 16x16=256 spatial patches over 64/2=32 tubelets ~= 8192
tokens/clip (DENSE_TOKENS_PER_CLIP default below); keys stay pooled (one vector). So dense
latents dwarf the pooled vector by the token factor (~8192x for ViT-L).

Per-clip cost, float32 latents (keys+labels included), at the documented token estimates:

  encoder       embed_dim  pooled/clip   dense latents/clip   dense total/clip
  ViT-L/16        1024       ~8.0 KB        ~32.0 MB (8192 tok)   ~32.0 MB
  ViT-H/16        1280       ~10.0 KB       ~40.0 MB (8192 tok)   ~40.0 MB
  ViT-g/16 384    1408       ~11.0 KB       ~58.7 MB (10976 tok)  ~58.7 MB

(ViT-g at 384px tiles to 24x24=576 patches * 32 tubelets ~= 18432 tokens; the table uses the
conservative shared 8192/10976 estimates documented at DENSE_TOKENS_PER_CLIP. Adjust per encoder
via estimate_for_encoder(..., dense_tokens=...).) At scale: 10k pooled ViT-L clips ~= 78 MB;
10k dense ViT-L clips ~= 313 GB. Pooled is laptop-cheap; a full 10k dense corpus exceeds the
current local disk envelope, while bounded dense shards remain locally testable. E6 is deferred
because its registered V-JEPA 2.1 dense checkpoint is unavailable, not because ViT-H or ViT-g
cannot execute locally.
"""

from __future__ import annotations

import json
from pathlib import Path

# bytes per element by numpy dtype name (the dtypes a LatentStore can write).
_DTYPE_BYTES = {"float32": 4, "float16": 2, "fp16": 2, "int64": 8, "int32": 4, "uint8": 1}

# Default dense tokens/clip: 64-frame ViT-L/16 at 256px -> 16*16 spatial * (64/2) tubelets = 8192.
# A single, documented stand-in so dense estimates are explicit and tunable, not magic.
DENSE_TOKENS_PER_CLIP = 8192


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[str(dtype)]
    except KeyError as e:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(_DTYPE_BYTES)}") from e


def estimate_cache_bytes(
    count: int,
    feat_shape: list[int],
    dtype: str = "float32",
    has_labels: bool = True,
    dense: bool = False,
) -> int:
    """Raw array bytes for a cache of `count` items with per-item latent shape `feat_shape`.

    Sums the three memmap arrays: latents (count * prod(feat_shape) * dtype_width), keys
    (count * key_dim * 4, key_dim taken as the last feat dim, always float32 like the store),
    labels (count * 8 int64 when has_labels). Ignores the ~128-byte/file npy header (rounding
    error). `dense` is accepted for signature symmetry; the shape already encodes density, so it
    does not change the math here (use estimate_for_encoder to expand pooled vs dense shapes).
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    if not feat_shape:
        raise ValueError("feat_shape must be non-empty")
    per_item = 1
    for d in feat_shape:
        per_item *= int(d)
    latents = count * per_item * _dtype_bytes(dtype)
    key_dim = int(feat_shape[-1])  # keys store the pooled vector; LatentStore writes float32
    keys = count * key_dim * _dtype_bytes("float32")
    labels = count * _dtype_bytes("int64") if has_labels else 0
    return int(latents + keys + labels)


def estimate_for_encoder(
    encoder_cfg_dict: dict,
    n_clips: int,
    dense: bool = False,
    dtype: str = "float32",
    has_labels: bool = True,
    dense_tokens: int | None = None,
) -> dict:
    """Estimate cache bytes for `n_clips` through an encoder config (a dict with `embed_dim`,
    optionally `dense`/`pool`/`name`). Pooled -> feat_shape [embed_dim]; dense -> feat_shape
    [tokens, embed_dim] with tokens = dense_tokens or DENSE_TOKENS_PER_CLIP. The explicit
    `dense` arg overrides the config's dense flag so callers can price either layout for any
    encoder. Returns {bytes, human, n_clips, dense, tokens_per_clip, embed_dim, per_clip_bytes}.
    """
    embed_dim = int(encoder_cfg_dict["embed_dim"])
    use_dense = bool(dense or encoder_cfg_dict.get("dense", False))
    tokens = int(dense_tokens or DENSE_TOKENS_PER_CLIP) if use_dense else 1
    feat_shape = [tokens, embed_dim] if use_dense else [embed_dim]
    total = estimate_cache_bytes(n_clips, feat_shape, dtype=dtype, has_labels=has_labels, dense=use_dense)
    per_clip = estimate_cache_bytes(1, feat_shape, dtype=dtype, has_labels=has_labels, dense=use_dense)
    return {
        "bytes": total,
        "human": human_bytes(total),
        "n_clips": int(n_clips),
        "dense": use_dense,
        "tokens_per_clip": tokens,
        "embed_dim": embed_dim,
        "per_clip_bytes": per_clip,
        "encoder": str(encoder_cfg_dict.get("name", "?")),
    }


def human_bytes(n: int) -> str:
    """Human-readable size, binary units (1 KB == 1024 B). 2 sig figs past KB, integer bytes."""
    x = float(n)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while abs(x) >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} B"
    return f"{x:.2f} {units[i]}"


def dir_size(path: Path) -> int:
    """Total bytes of all files under `path` (recursive). Missing path -> 0."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _is_cache_dir(p: Path) -> bool:
    return p.is_dir() and (p / "meta.json").is_file()


def list_caches_with_size(root: Path) -> list[dict]:
    """Every cache directory directly under `root`, with on-disk size and (when present) the
    count/feat_shape/dtype/backend from meta.json + provenance.json. A cache dir is one holding a
    meta.json. Sorted largest first. Missing root -> []."""
    r = Path(root)
    out: list[dict] = []
    if not r.is_dir():
        return out
    for child in sorted(r.iterdir()):
        if not _is_cache_dir(child):
            continue
        size = dir_size(child)
        rec = {"name": child.name, "path": str(child), "bytes": size, "human": human_bytes(size)}
        try:
            meta = json.loads((child / "meta.json").read_text())
            rec.update(
                count=meta.get("count"),
                feat_shape=meta.get("feat_shape"),
                dtype=meta.get("dtype"),
            )
        except Exception:
            pass
        prov = child / "provenance.json"
        if prov.is_file():
            try:
                pj = json.loads(prov.read_text())
                rec["backend"] = pj.get("encoder_backend")
                rec["result_tag"] = pj.get("result_tag")
            except Exception:
                pass
        out.append(rec)
    out.sort(key=lambda d: d["bytes"], reverse=True)
    return out


def prune_caches(root: Path, keep: list[str] | None = None, dry_run: bool = True) -> list[dict]:
    """Plan (and only with dry_run=False, perform) deletion of caches under `root` whose name is
    NOT in `keep`. SAFE BY DEFAULT: dry_run=True (the default) deletes nothing, it only returns
    the plan. Returns one record per cache: {name, path, bytes, human, kept, would_delete,
    deleted}. Deletion happens only when dry_run is False AND the cache is not kept; everything
    in `keep` is always preserved. `keep=None` keeps nothing (every cache is a deletion
    candidate), so an empty keep-list is an explicit, deliberate choice by the caller.
    """
    keepset = set(keep or [])
    plan: list[dict] = []
    for rec in list_caches_with_size(root):
        kept = rec["name"] in keepset
        would_delete = not kept
        deleted = False
        if would_delete and not dry_run:
            _rmtree(Path(rec["path"]))
            deleted = True
        plan.append(
            {
                "name": rec["name"],
                "path": rec["path"],
                "bytes": rec["bytes"],
                "human": rec["human"],
                "kept": kept,
                "would_delete": would_delete,
                "deleted": deleted,
            }
        )
    return plan


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p)
