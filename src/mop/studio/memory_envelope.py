from __future__ import annotations

import resource
import sys
import time
from typing import Any


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
