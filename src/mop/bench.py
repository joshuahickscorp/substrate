
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from .devices import resolve
from .harness.sweep import expand
from .learning.backprop import Learner, TrainConfig
from .logging_utils import RunManifest
from .shell import ClassHead, ReplayBuffer
from .substrate import LatentStore, preprocess_clip

_SEED = 0
_DIM = 32  # latent dim shared by buffer / learner / store benches
_RETRIEVE_N = 128  # vectors in the index for the faiss-vs-brute bench


def _result(name: str, seconds: float, n: int) -> dict:
    seconds = max(float(seconds), 1e-9)
    return {"name": name, "seconds": seconds, "n": int(n), "throughput": n / seconds}


def bench_preprocess_clip(n: int = 8) -> dict:
    g = torch.Generator().manual_seed(_SEED)
    frames = (torch.rand(12, 32, 32, 3, generator=g) * 255).to(torch.uint8)
    t0 = time.perf_counter()
    for _ in range(n):
        preprocess_clip(frames, frames_per_clip=4, res=16)
    return _result("preprocess_clip", time.perf_counter() - t0, n)


def bench_latent_store(n: int = 64) -> dict:
    g = torch.Generator().manual_seed(_SEED)
    x = torch.randn(n, _DIM, generator=g).numpy()
    y = torch.arange(n).numpy()
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        store = LatentStore.create(Path(d), "bench", (_DIM,), capacity=n, key_dim=_DIM, has_labels=True)
        store.write_batch(0, x, x, y)
        store.finalize()
        r = LatentStore.open(Path(d) / "bench")
        _ = r.latents()
        secs = time.perf_counter() - t0
    return _result("latent_store", secs, n)


def bench_buffer_sample(n: int = 256) -> dict:
    g = torch.Generator().manual_seed(_SEED)
    buf = ReplayBuffer(capacity=128, dim=_DIM, prioritized=True, index="brute", seed=_SEED)
    buf.add(torch.randn(128, _DIM, generator=g), torch.randint(0, 4, (128,), generator=g))
    t0 = time.perf_counter()
    drawn = 0
    for _ in range(8):
        drawn += len(buf.sample(n // 8)["idx"])
    return _result("buffer_sample", time.perf_counter() - t0, drawn)


_PROBE = (
    "import os;os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE');import torch;import faiss;"
    f"import numpy as np;i=faiss.IndexFlatL2({_DIM});"
    f"i.add(np.zeros(({_RETRIEVE_N},{_DIM}),'float32'));"
    f"i.search(np.zeros((16,{_DIM}),'float32'),5);print('ok')"
)
_faiss_safe_cache: bool | None = None


def _faiss_search_safe() -> bool:
    global _faiss_safe_cache
    if _faiss_safe_cache is None:
        try:
            r = subprocess.run(
                [sys.executable, "-c", _PROBE],
                capture_output=True,
                timeout=60,
                env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
            )
            _faiss_safe_cache = r.returncode == 0 and b"ok" in r.stdout
        except Exception:
            _faiss_safe_cache = False
    return _faiss_safe_cache


def _retrieve_bench(index: str, name: str, n: int) -> dict:
    g = torch.Generator().manual_seed(_SEED)
    buf = ReplayBuffer(capacity=_RETRIEVE_N, dim=_DIM, key_dim=_DIM, index=index, seed=_SEED)
    buf.add(torch.randn(_RETRIEVE_N, _DIM, generator=g), torch.randint(0, 4, (_RETRIEVE_N,), generator=g))
    q = torch.randn(n, _DIM, generator=g)
    buf.retrieve(q, k=5)  # first call builds the index; exclude from the timed loop
    t0 = time.perf_counter()
    for _ in range(4):
        buf.retrieve(q, k=5)
    return _result(name, time.perf_counter() - t0, n * 4)


def bench_retrieve_faiss(n: int = 16) -> dict:
    if _faiss_search_safe():
        return _retrieve_bench("faiss", "retrieve_faiss", n)
    return _retrieve_bench("brute", "retrieve_faiss_unavailable", n)


def bench_retrieve_brute(n: int = 16) -> dict:
    return _retrieve_bench("brute", "retrieve_brute", n)


def bench_learner_step(n: int = 8) -> dict:
    dev = resolve("cpu")
    g = torch.Generator().manual_seed(_SEED)
    model = ClassHead(_DIM, n_classes=4, hidden=32, depth=1)
    learner = Learner(model, dev, TrainConfig(base_lr=1e-3), seed=_SEED)
    x = torch.randn(16, _DIM, generator=g)
    y = torch.randint(0, 4, (16,), generator=g)
    t0 = time.perf_counter()
    for _ in range(n):
        learner._step(x, y, progress=0.0)
    return _result("learner_step", time.perf_counter() - t0, n)


def bench_queue_expand(n: int = 64) -> dict:
    axes: dict[str, list] = {"a": [1, 2, 3, 4], "b": [True, False]}
    seeds = list(range(8))  # 4 * 2 * 8 = 64 combos
    base = ["experiment=e1_baseline"]
    t0 = time.perf_counter()
    out = expand(axes, seeds, base)
    return _result("queue_expand", time.perf_counter() - t0, len(out))


def bench_manifest_write(n: int = 8) -> dict:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        t0 = time.perf_counter()
        for i in range(n):
            man = RunManifest(name="bench", seed=_SEED, device="cpu")
            man.write(root / f"{i:03d}")
        secs = time.perf_counter() - t0
    return _result("manifest_write", secs, n)


BENCHES = (
    bench_preprocess_clip,
    bench_latent_store,
    bench_buffer_sample,
    bench_retrieve_faiss,
    bench_retrieve_brute,
    bench_learner_step,
    bench_queue_expand,
    bench_manifest_write,
)


def run_benches(reps: int = 3) -> list[dict]:
    reps = max(1, int(reps))
    out = []
    for fn in BENCHES:
        runs = [fn() for _ in range(reps)]
        secs = statistics.median(r["seconds"] for r in runs)
        nn = runs[0]["n"]
        rec = _result(runs[0]["name"], secs, nn)
        rec["reps"] = reps
        out.append(rec)
    return out


def render_md(records: list[dict]) -> str:
    head = "| bench | n | reps | seconds | throughput (items/s) |\n|---|---|---|---|---|\n"
    rows = "".join(
        f"| {r['name']} | {r['n']} | {r.get('reps', 1)} | {r['seconds']:.6f} | {r['throughput']:.1f} |\n"
        for r in records
    )
    return "# Microbenchmarks\n\n" + head + rows
