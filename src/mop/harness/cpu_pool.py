
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger("cpu_pool")
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def physical_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
    except Exception:
        return os.cpu_count() or 4


def plan_pool(
    mode: str = "small", footprint_mb: float = 600.0, mem_budget_mb: float = 13000.0
) -> tuple[int, int]:
    cores = physical_cores()
    threads = 1 if mode == "small" else max(2, cores // 4)
    workers = max(1, cores // threads)
    mem_cap = max(1, int(mem_budget_mb // max(1.0, footprint_mb)))
    workers = min(workers, mem_cap)
    return workers, threads


def _init_worker(threads: int) -> None:
    for v in _THREAD_ENV:
        os.environ[v] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _run_one(key: str, overrides: list[str], ckpt_dir: str) -> dict:
    ckpt = Path(ckpt_dir) / f"{key}.json"
    if ckpt.exists():
        rec = json.loads(ckpt.read_text())
        rec["cached"] = True
        return rec
    from ..config import compose
    from .runner import run_experiment

    t0 = time.perf_counter()
    cfg = compose(list(overrides))
    metrics = run_experiment(cfg)
    rec = {
        "key": key,
        "overrides": list(overrides),
        "metrics": metrics,
        "seconds": time.perf_counter() - t0,
        "cached": False,
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(rec, default=str))
    return rec


@dataclass
class PoolResult:
    results: dict = field(default_factory=dict)
    degraded: list = field(default_factory=list)
    wall_seconds: float = 0.0


def run_pool(units: list[dict], workers: int, threads: int, ckpt_dir: Path, retries: int = 1) -> PoolResult:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = PoolResult()
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    pending = list(units)
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(threads,)
        ) as ex:
            futs = {ex.submit(_run_one, u["key"], u["overrides"], str(ckpt_dir)): u for u in units}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    rec = fut.result()
                    out.results[u["key"]] = rec
                    log.info(
                        "done %s (%.1fs%s)", u["key"], rec["seconds"], " cached" if rec.get("cached") else ""
                    )
                except Exception as e:  # one unit failed, pool intact
                    out.degraded.append({"key": u["key"], "error": repr(e)[:200]})
                    log.warning("unit %s failed: %s", u["key"], repr(e)[:160])
            pending = []
    except Exception as e:  # pool broke (segfault/OOM): finish remaining serially
        log.warning("pool broke (%s); finishing %d units serially", repr(e)[:120], len(pending))
        _init_worker(threads)
        done = set(out.results)
        for u in pending:
            if u["key"] in done:
                continue
            for attempt in range(retries + 1):
                try:
                    out.results[u["key"]] = _run_one(u["key"], u["overrides"], str(ckpt_dir))
                    break
                except Exception as ee:
                    if attempt == retries:
                        out.degraded.append({"key": u["key"], "error": repr(ee)[:200]})
    out.wall_seconds = time.perf_counter() - t0
    return out
