from __future__ import annotations

import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "mop-memory-envelope/v1"


@dataclass
class MemorySampler:
    label: str
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, stage: str) -> dict[str, Any]:
        snap = memory_snapshot(stage)
        self.samples.append(snap)
        return snap

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            self.sample("empty")
        return summarize_samples(self.label, self.samples)


def memory_snapshot(stage: str) -> dict[str, Any]:
    process_rss_gb = _process_rss_gb()
    system = _system_memory()
    mps = _mps_memory()
    return {
        "stage": str(stage),
        "t_wall_s": round(time.time(), 3),
        "process_rss_gb": _round(process_rss_gb),
        "process_maxrss_gb": _round(_ru_maxrss_gb()),
        "system_total_gb": _round(system.get("total_gb")),
        "system_available_gb": _round(system.get("available_gb")),
        "mps_current_allocated_gb": _round(mps.get("current_allocated_gb")),
        "mps_driver_allocated_gb": _round(mps.get("driver_allocated_gb")),
    }


def summarize_samples(label: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "label": str(label),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "n_samples": len(samples),
        "process_rss_gb": _series_summary(samples, "process_rss_gb", peak=max),
        "process_maxrss_gb": _series_summary(samples, "process_maxrss_gb", peak=max),
        "system_available_gb": _series_summary(samples, "system_available_gb", peak=min),
        "mps_current_allocated_gb": _series_summary(samples, "mps_current_allocated_gb", peak=max),
        "mps_driver_allocated_gb": _series_summary(samples, "mps_driver_allocated_gb", peak=max),
        "samples": samples,
    }


def _series_summary(samples: list[dict[str, Any]], key: str, *, peak) -> dict[str, float | None]:
    values = [float(s[key]) for s in samples if isinstance(s.get(key), int | float)]
    if not values:
        return {"start": None, "end": None, "peak": None}
    return {"start": values[0], "end": values[-1], "peak": _round(peak(values))}


def _process_rss_gb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return None


def _system_memory() -> dict[str, float | None]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {"total_gb": mem.total / 1e9, "available_gb": mem.available / 1e9}
    except Exception:
        return {"total_gb": None, "available_gb": None}


def _ru_maxrss_gb() -> float | None:
    try:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return raw / 1e9
    return raw * 1024.0 / 1e9


def _mps_memory() -> dict[str, float | None]:
    try:
        import torch

        if not torch.backends.mps.is_available():
            return {"current_allocated_gb": None, "driver_allocated_gb": None}
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        return {
            "current_allocated_gb": (float(current()) / 1e9) if callable(current) else None,
            "driver_allocated_gb": (float(driver()) / 1e9) if callable(driver) else None,
        }
    except Exception:
        return {"current_allocated_gb": None, "driver_allocated_gb": None}


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
