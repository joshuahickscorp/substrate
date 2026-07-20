
from __future__ import annotations

import json
from pathlib import Path

import torch

from .datasets import Task
from .latent_store import LatentStore


def open_real_store(name: str, data_dir: str | Path = "data/cache") -> LatentStore:
    root = Path(data_dir) / name
    if not (root / "meta.json").exists():
        raise FileNotFoundError(f"no LatentStore at {root} (expected meta.json); build the cache first")
    return LatentStore.open(root)


def _as_store(store: LatentStore | str, data_dir: str | Path = "data/cache") -> LatentStore:
    return store if isinstance(store, LatentStore) else open_real_store(store, data_dir)


def factors_meta(store: LatentStore) -> dict | None:
    p = store.root / "factors.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    if isinstance(payload, dict) and payload.get("schema") == "mop-factor-sidecar/v2":
        metadata = payload.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else None
    return payload


def real_task_stream(
    store: LatentStore | str,
    n_tasks: int,
    incremental: str = "class",  # class | task | domain
    samples_per_task: int | None = None,
    forward_dynamics: bool = False,
    seed: int = 0,
    data_dir: str | Path = "data/cache",
) -> list[Task]:
    st = _as_store(store, data_dir)
    x = st.latents().flatten(1).float()
    y = st.labels()
    if y is None:
        y = torch.zeros(len(st), dtype=torch.long)
    y = y.long()
    classes = sorted(set(y.tolist()))
    k = len(classes)
    g = torch.Generator().manual_seed(seed)

    def _cap(xt: torch.Tensor, yt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if samples_per_task is None or xt.shape[0] <= samples_per_task:
            return xt, yt
        idx = torch.randperm(xt.shape[0], generator=g)[:samples_per_task]
        return xt[idx], yt[idx]

    tasks: list[Task] = []
    if incremental == "class":
        per = max(1, k // n_tasks)
        for t in range(n_tasks):
            block = set(classes[t * per : (t + 1) * per]) or {classes[-1]}
            mask = torch.tensor([c in block for c in y.tolist()])
            xt, yt = _cap(x[mask], y[mask])
            tasks.append(
                Task(f"real_task{t}", xt, yt, xt if forward_dynamics else None, n_classes=k, task_id=t)
            )
        return tasks

    remap = {c: i for i, c in enumerate(classes)}
    by_class: dict[int, torch.Tensor] = {}
    for c in classes:
        idx = (y == c).nonzero(as_tuple=True)[0]
        by_class[c] = idx[torch.randperm(idx.shape[0], generator=g)]
    for t in range(n_tasks):
        xs, ys = [], []
        for c in classes:
            idx = by_class[c]
            if incremental == "domain":
                fold = max(1, idx.shape[0] // n_tasks)
                sel = idx[t * fold : (t + 1) * fold] if (t + 1) * fold <= idx.shape[0] else idx[-fold:]
            else:  # task-incremental: same samples reused every task (fixed geometry, known task id)
                sel = idx
            xs.append(x[sel])
            ys.append(torch.full((sel.shape[0],), remap[c], dtype=torch.long))
        xt = torch.cat(xs)
        yt = torch.cat(ys)
        xt, yt = _cap(xt, yt)
        tasks.append(Task(f"real_task{t}", xt, yt, xt if forward_dynamics else None, n_classes=k, task_id=t))
    return tasks


def factorized_arrays(
    store: LatentStore | str, data_dir: str | Path = "data/cache"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    st = _as_store(store, data_dir)
    fm = factors_meta(st)
    if fm is None:
        raise ValueError(
            f"store {st.meta.name} has no factors.json: it is a single-factor cache, not factorized. "
            "Build a factorized cache (scripts/cache_factorized_encoder.py) to use factorized_arrays."
        )
    n_b = int(fm["n_b"])
    x = st.latents().flatten(1).float()
    y = st.labels()
    if y is None:
        raise ValueError(f"store {st.meta.name} has no labels")
    y = y.long()
    y_a = torch.div(y, n_b, rounding_mode="floor")
    y_b = y % n_b
    return x, y_a, y_b
