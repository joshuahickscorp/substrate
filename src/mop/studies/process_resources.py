"""Phase-local process resource sampling for long-lived Python workers."""

from __future__ import annotations

import threading
from typing import Any

import psutil


class PeakRSSMonitor:
    """Sample current process RSS without inheriting an earlier process-lifetime peak."""

    def __init__(self, *, interval_seconds: float = 0.002) -> None:
        if interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.end_rss_bytes = 0
        self.sample_count = 0
        self.sampling_error: str | None = None
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            rss = int(self._process.memory_info().rss)
        except (OSError, psutil.Error) as exc:
            self.sampling_error = f"{type(exc).__name__}: {exc}"
            return
        self.sample_count += 1
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.end_rss_bytes = rss

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> PeakRSSMonitor:
        self._sample()
        self.start_rss_bytes = self.end_rss_bytes
        self._thread = threading.Thread(target=self._run, name="mop-peak-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval_seconds * 10))
        self._sample()

    @property
    def all_ok(self) -> bool:
        return self.sampling_error is None and self.sample_count >= 2 and self.peak_rss_bytes > 0

    @property
    def peak_increment_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)

    def receipt(self) -> dict[str, Any]:
        return {
            "method": "phase-local psutil current-RSS sampler",
            "interval_seconds": self.interval_seconds,
            "start_rss_bytes": self.start_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_increment_bytes": self.peak_increment_bytes,
            "end_rss_bytes": self.end_rss_bytes,
            "sample_count": self.sample_count,
            "sampling_error": self.sampling_error,
            "all_ok": self.all_ok,
        }
