from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-p9-workload-accounting/v1"


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw if sys.platform == "darwin" else raw * 1024


def _lifetime_max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _accelerator_bytes() -> dict[str, int | None]:
    try:
        import torch

        if torch.backends.mps.is_available():
            return {
                "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
    except Exception:
        pass
    return {"mps_current_allocated_bytes": None, "mps_driver_allocated_bytes": None}


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
    return total


@dataclass
class _Phase:
    name: str
    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    rss_start_bytes: int
    rss_end_bytes: int
    accelerator_start: dict[str, int | None]
    accelerator_end: dict[str, int | None]
    storage_delta_bytes: dict[str, int]
    retries: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_seconds": self.wall_seconds,
            "cpu_user_seconds": self.cpu_user_seconds,
            "cpu_system_seconds": self.cpu_system_seconds,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "accelerator_start": self.accelerator_start,
            "accelerator_end": self.accelerator_end,
            "storage_delta_bytes": self.storage_delta_bytes,
            "retries": self.retries,
        }


@dataclass
class WorkloadAccountant:
    workload: str
    watch_paths: dict[str, Path] = field(default_factory=dict)
    _phases: list[_Phase] = field(default_factory=list, init=False)
    _open_phase: str | None = field(default=None, init=False)
    _started: float = field(default_factory=time.perf_counter, init=False)
    _created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(), init=False)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self._open_phase is not None:
            raise RuntimeError(f"phase {name!r} opened while {self._open_phase!r} is still open")
        if not name or not isinstance(name, str):
            raise ValueError("phase name must be a nonempty string")
        self._open_phase = name
        usage0 = resource.getrusage(resource.RUSAGE_SELF)
        storage0 = {label: _tree_bytes(path) for label, path in self.watch_paths.items()}
        rss0 = _rss_bytes()
        acc0 = _accelerator_bytes()
        wall0 = time.perf_counter()
        try:
            yield
        finally:
            wall1 = time.perf_counter()
            usage1 = resource.getrusage(resource.RUSAGE_SELF)
            self._phases.append(
                _Phase(
                    name=name,
                    wall_seconds=wall1 - wall0,
                    cpu_user_seconds=usage1.ru_utime - usage0.ru_utime,
                    cpu_system_seconds=usage1.ru_stime - usage0.ru_stime,
                    rss_start_bytes=rss0,
                    rss_end_bytes=_rss_bytes(),
                    accelerator_start=acc0,
                    accelerator_end=_accelerator_bytes(),
                    storage_delta_bytes={
                        label: _tree_bytes(path) - storage0[label] for label, path in self.watch_paths.items()
                    },
                )
            )
            self._open_phase = None

    def note_retry(self, phase_name: str) -> None:
        for phase in reversed(self._phases):
            if phase.name == phase_name:
                phase.retries += 1
                return
        raise KeyError(f"no completed phase named {phase_name!r} to attach a retry to")

    def receipt(self) -> dict[str, Any]:
        if self._open_phase is not None:
            raise RuntimeError(f"cannot emit a receipt while phase {self._open_phase!r} is open")
        span = time.perf_counter() - self._started
        phase_wall = sum(phase.wall_seconds for phase in self._phases)
        return {
            "schema": SCHEMA,
            "created_at": self._created_at,
            "workload": self.workload,
            "phases": [phase.as_record() for phase in self._phases],
            "totals": {
                "span_seconds": span,
                "accounted_phase_seconds": phase_wall,
                "idle_seconds": max(0.0, span - phase_wall),
                "cpu_user_seconds": sum(p.cpu_user_seconds for p in self._phases),
                "cpu_system_seconds": sum(p.cpu_system_seconds for p in self._phases),
                "lifetime_max_rss_bytes": _lifetime_max_rss_bytes(),
                "storage_delta_bytes": {
                    label: sum(p.storage_delta_bytes.get(label, 0) for p in self._phases)
                    for label in self.watch_paths
                },
                "retries": sum(p.retries for p in self._phases),
            },
            "energy": {
                "measured": False,
                "statement": (
                    "no wall-power meter with a stated system boundary is attached; energy is "
                    "not reported and must not be cited from this receipt"
                ),
            },
            "honesty": {
                "rss_sampling": "phase boundaries plus process-lifetime maximum only",
                "idle_definition": "accountant lifetime span minus summed phase walls",
                "overlapping_phases": "refused",
            },
        }

    def write(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.receipt(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out
