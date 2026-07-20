from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch


@dataclass
class StoreMeta:
    name: str
    feat_shape: list[int]  # per-item latent shape, e.g. [1024] pooled or [T,P,1024] dense
    dtype: str
    key_dim: int
    count: int
    has_labels: bool


class LatentStore:
    def __init__(self, root: Path, meta: StoreMeta, mode: str = "r"):
        self.root = Path(root)
        self.meta = meta
        self.mode = mode
        mm: Literal["r", "r+"] = "r" if mode == "r" else "r+"
        lat_path = self._p("latents.npy") if self._p("latents.npy").exists() else self._p("features.npy")
        self._lat = np.load(lat_path, mmap_mode=mm)
        self._keys = np.load(self._p("keys.npy"), mmap_mode=mm) if self._p("keys.npy").exists() else None
        if meta.has_labels:
            lab = self._p("labels.npy") if self._p("labels.npy").exists() else self._p("labels_shape.npy")
            self._labels = np.load(lab, mmap_mode=mm)
        else:
            self._labels = None

    def _p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @classmethod
    def create(
        cls,
        root: Path,
        name: str,
        feat_shape: tuple[int, ...],
        capacity: int,
        key_dim: int,
        dtype: str = "float32",
        has_labels: bool = False,
    ) -> LatentStore:
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(d / "latents.npy", mode="w+", dtype=dtype, shape=(capacity, *feat_shape))
        np.lib.format.open_memmap(d / "keys.npy", mode="w+", dtype="float32", shape=(capacity, key_dim))
        if has_labels:
            np.lib.format.open_memmap(d / "labels.npy", mode="w+", dtype="int64", shape=(capacity,))
        meta = StoreMeta(name, list(feat_shape), dtype, key_dim, 0, has_labels)
        (d / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        return cls(d, meta, mode="r+")

    @classmethod
    def open(cls, root: Path) -> LatentStore:
        root = Path(root)
        raw = json.loads((root / "meta.json").read_text())
        allowed = {f.name for f in fields(StoreMeta)}
        if "feat_shape" in raw:
            meta = StoreMeta(**{k: v for k, v in raw.items() if k in allowed})
        else:
            feat_file = "latents.npy" if (root / "latents.npy").exists() else "features.npy"
            arr = np.load(root / feat_file, mmap_mode="r")
            has_labels = (root / "labels.npy").exists() or (root / "labels_shape.npy").exists()
            meta = StoreMeta(
                name=str(raw.get("tag", root.name)),
                feat_shape=list(arr.shape[1:]),
                dtype=str(arr.dtype),
                key_dim=0,
                count=int(arr.shape[0]),
                has_labels=has_labels,
            )
        return cls(root, meta, mode="r")

    def write_batch(
        self, start: int, latents: np.ndarray, keys: np.ndarray, labels: np.ndarray | None = None
    ) -> int:
        n = latents.shape[0]
        self._lat[start : start + n] = latents.astype(self.meta.dtype)
        if self._keys is not None:
            self._keys[start : start + n] = keys.astype("float32")
        if labels is not None and self._labels is not None:
            self._labels[start : start + n] = labels.astype("int64")
        self.meta.count = max(self.meta.count, start + n)
        return start + n

    def finalize(self) -> None:
        self._lat.flush()
        if self._keys is not None:
            self._keys.flush()
        if self._labels is not None:
            self._labels.flush()
        (self.root / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))

    def __len__(self) -> int:
        return self.meta.count

    def latents(self, idx=None) -> torch.Tensor:
        a = self._lat[: self.meta.count] if idx is None else self._lat[idx]
        return torch.from_numpy(np.array(a, copy=True, order="C"))

    def keys(self, idx=None) -> torch.Tensor | None:
        if self._keys is None:
            return None
        a = self._keys[: self.meta.count] if idx is None else self._keys[idx]
        return torch.from_numpy(np.array(a, copy=True, order="C"))

    def labels(self, idx=None) -> torch.Tensor | None:
        if self._labels is None:
            return None
        a = self._labels[: self.meta.count] if idx is None else self._labels[idx]
        return torch.from_numpy(np.array(a, copy=True, order="C"))
