
from __future__ import annotations

import json
from pathlib import Path

_DTYPE_BYTES = {"float32": 4, "float16": 2, "fp16": 2, "int64": 8, "int32": 4, "uint8": 1}

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
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _is_cache_dir(p: Path) -> bool:
    return p.is_dir() and (p / "meta.json").is_file()


def list_caches_with_size(root: Path) -> list[dict]:
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
