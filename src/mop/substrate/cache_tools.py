
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..provenance import RESULT_TAGS, cache_id
from .cache_manifest import DEFAULT_MANIFEST, validate_cache_manifest
from .latent_store import LatentStore

DEFAULT_ROOT = Path("data/cache")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _meta_summary(store_dir: Path) -> dict:
    out: dict = {"name": store_dir.name, "count": None, "dim": None, "backend": None, "result_tag": None}
    try:
        meta = LatentStore.open(store_dir).meta
        out["name"] = meta.name
        out["count"] = meta.count
        out["dim"] = int(np.prod(meta.feat_shape)) if meta.feat_shape else None
    except Exception:
        pass
    prov = _read_json(store_dir / "provenance.json")
    if prov:
        out["backend"] = prov.get("encoder_backend")
        out["result_tag"] = prov.get("result_tag")
    return out


def list_caches(root: Path | str = DEFAULT_ROOT) -> list[dict]:
    root = Path(root)
    if not root.exists():
        return []
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())
    return [_meta_summary(d) for d in dirs]


def cache_info(store_dir: Path | str) -> dict:
    store_dir = Path(store_dir)
    meta = _read_json(store_dir / "meta.json")
    prov = _read_json(store_dir / "provenance.json")
    facts: dict = {"store_dir": str(store_dir)}
    if meta is None:
        facts["error"] = "meta.json missing"
        return {"meta": None, "provenance": prov, "facts": facts}

    lat_path = _latent_path(store_dir)
    key_path = store_dir / "keys.npy"
    facts["latents_on_disk"] = _npy_header(lat_path)
    facts["keys_on_disk"] = _npy_header(key_path)
    if meta.get("has_labels"):
        lab = _load_npy(store_dir / "labels.npy")
        if lab is not None and lab.size:
            count = int(meta.get("count", 0)) or lab.shape[0]
            sub = lab[:count]
            facts["label_min"] = int(sub.min())
            facts["label_max"] = int(sub.max())
            facts["n_classes_seen"] = int(np.unique(sub).size)
    facts["recomputed_cache_id"] = _content_cache_id(store_dir, meta)
    if prov is not None:
        facts["cache_id_matches"] = facts["recomputed_cache_id"] == prov.get("cache_id")
    manifest = store_dir / DEFAULT_MANIFEST
    facts["cache_manifest_present"] = manifest.exists()
    if manifest.exists():
        facts["cache_manifest_clean"] = not validate_cache_manifest(store_dir)
        facts["citable_ready"] = not validate_cache_manifest(store_dir, citable=True)
    else:
        facts["citable_ready"] = False
    return {"meta": meta, "provenance": prov, "facts": facts}


def validate_cache(store_dir: Path | str, *, citable: bool = False) -> list[str]:
    store_dir = Path(store_dir)
    problems: list[str] = []

    raw = store_dir / "meta.json"
    if not raw.exists():
        return ["meta.json missing"]
    try:
        meta_dict = json.loads(raw.read_text())
        meta = LatentStore.open(store_dir).meta
    except Exception as e:  # malformed json or unsupported store layout
        return [f"meta.json/store layout cannot open as LatentStore: {e}"]

    count, feat_shape = int(meta.count), tuple(meta.feat_shape)

    lat_hdr = _npy_header(_latent_path(store_dir))
    if lat_hdr is None:
        problems.append("latents.npy/features.npy missing or unreadable")
    else:
        actual_shape = tuple(lat_hdr["shape"])
        if actual_shape[0] < count or actual_shape[1:] != feat_shape:
            problems.append(
                f"latents shape {lat_hdr['shape']} does not cover meta count={count}, feat_shape={feat_shape}"
            )
        if count == 0 and actual_shape[0] > 0:
            problems.append(
                f"meta count is zero but latent array contains {actual_shape[0]} rows; cache is incomplete"
            )
        if lat_hdr["dtype"] != meta.dtype:
            problems.append(f"latents dtype {lat_hdr['dtype']} != meta dtype {meta.dtype}")

    key_hdr = _npy_header(store_dir / "keys.npy")
    if key_hdr is None:
        if int(meta.key_dim) > 0:
            problems.append("keys.npy missing or unreadable")
    else:
        if int(key_hdr["shape"][0]) < count:
            problems.append(f"keys count {key_hdr['shape'][0]} < meta count {count}")
        if len(key_hdr["shape"]) >= 2 and int(key_hdr["shape"][1]) != int(meta.key_dim):
            problems.append(f"keys dim {key_hdr['shape'][1]} != meta key_dim {meta.key_dim}")

    if meta.has_labels:
        label_path = (
            store_dir / "labels.npy"
            if (store_dir / "labels.npy").exists()
            else store_dir / "labels_shape.npy"
        )
        lab = _load_npy(label_path)
        if lab is None:
            problems.append("has_labels true but labels.npy/labels_shape.npy missing or unreadable")
        else:
            if lab.shape[0] < count:
                problems.append(f"labels length {lab.shape[0]} < meta count {count}")
            sub = lab[:count]
            if sub.size and (int(sub.min()) < 0):
                problems.append(f"labels contain negatives (min {int(sub.min())})")
            if sub.size and not np.issubdtype(sub.dtype, np.integer):
                problems.append(f"labels dtype {sub.dtype} is not integer")
            # honest class coverage: a persisted label_map must not claim classes the cache lacks.
            lm = _read_json(store_dir / "label_map.json")
            if lm and sub.size:
                declared = {int(v) for v in lm.values()}
                present = {int(x) for x in np.unique(sub).tolist()}
                missing = sorted(declared - present)
                if missing:
                    problems.append(
                        f"label_map declares classes {sorted(declared)} but cached labels cover only "
                        f"{sorted(present)} (missing {missing}); a capped cache must persist only "
                        f"present classes"
                    )

    prov = _read_json(store_dir / "provenance.json")
    if prov is not None:
        if not prov.get("encoder_backend"):
            problems.append("provenance present but encoder_backend tag missing")
        tag = prov.get("result_tag")
        if tag not in RESULT_TAGS:
            problems.append(f"provenance result_tag {tag!r} not in {RESULT_TAGS}")
        recomputed = _content_cache_id(store_dir, meta_dict)
        if recomputed is not None and recomputed != prov.get("cache_id"):
            problems.append(
                f"cache_id mismatch: provenance {prov.get('cache_id')!r} != recomputed {recomputed!r}"
            )

    if (store_dir / DEFAULT_MANIFEST).exists():
        for problem in validate_cache_manifest(store_dir, citable=citable):
            problems.append(f"{DEFAULT_MANIFEST}: {problem}")
    elif citable:
        problems.append(f"{DEFAULT_MANIFEST}: missing for citable cache")

    if not problems and count > 0:
        try:
            store = LatentStore.open(store_dir)
            rng = np.random.default_rng(0)
            for idx in rng.integers(0, count, size=min(3, count)):
                row = store.latents(int(idx))
                if int(row.shape[0]) != int(np.prod(feat_shape)) and tuple(row.shape) != feat_shape:
                    problems.append(f"row {int(idx)} shape {tuple(row.shape)} unexpected for {feat_shape}")
        except Exception as e:
            problems.append(f"random rows not readable via LatentStore.open: {e}")

    return problems


def _content_cache_id(store_dir: Path, meta: dict) -> str | None:
    lat = _load_npy(_latent_path(store_dir))
    if lat is None:
        return None
    count = int(meta.get("count", 0)) or lat.shape[0]
    name = meta.get("name", store_dir.name)
    sample = lat[: min(4, count)].tobytes()[:4096] if count else b""
    return cache_id(name, count, sample)


def _latent_path(store_dir: Path) -> Path:
    native = store_dir / "latents.npy"
    return native if native.exists() else store_dir / "features.npy"


def _npy_header(path: Path) -> dict | None:
    arr = _load_npy(path)
    if arr is None:
        return None
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        return None
