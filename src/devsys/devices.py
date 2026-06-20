"""Device layer. One place that decides cpu/mps/cuda and degrades gracefully.

The whole point: the same code runs toy on an M3 laptop (mps) and full on a Studio
or rented CUDA box by config alone. Unsupported MPS ops fall back to CPU through
`safe_to`/`autofallback`, never crash the run.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # cpu | mps | cuda
    device: torch.device
    supports_amp: bool
    supports_pin: bool
    note: str


def _cuda_ok() -> bool:
    return torch.cuda.is_available()


def _mps_ok() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def resolve(requested: str = "auto") -> DeviceInfo:
    """Pick a real device. `auto` prefers cuda, then mps, then cpu. An explicit
    request that is unavailable degrades to the best present device, logged in note."""
    req = (requested or "auto").lower()
    note = ""
    if req == "cuda" and not _cuda_ok():
        note = "cuda requested but unavailable; "
        req = "auto"
    if req == "mps" and not _mps_ok():
        note = "mps requested but unavailable; "
        req = "auto"
    if req == "auto":
        kind = "cuda" if _cuda_ok() else "mps" if _mps_ok() else "cpu"
    else:
        kind = req
    if kind == "mps":
        # let unsupported ops silently run on cpu instead of throwing
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    info = DeviceInfo(
        kind=kind,
        device=torch.device(kind),
        supports_amp=kind == "cuda",
        supports_pin=kind == "cuda",
        note=note + f"resolved {kind}",
    )
    return info


def safe_to(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to device, falling back to cpu on an MPS dtype/op gap."""
    try:
        return x.to(device)
    except (RuntimeError, NotImplementedError) as e:  # pragma: no cover - hardware dependent
        warnings.warn(f"device move to {device} failed ({e}); staying on cpu", stacklevel=2)
        return x.to("cpu")


@contextmanager
def autofallback(device: torch.device):
    """Run a block on `device`; on an unsupported-op RuntimeError, retry on cpu.

    Used to wrap ops known to be patchy on Metal. The caller gets back whichever
    device actually ran, so downstream stays consistent.
    """
    try:
        yield device
    except (RuntimeError, NotImplementedError) as e:  # pragma: no cover - hardware dependent
        warnings.warn(f"op failed on {device} ({e}); caller should retry on cpu", stacklevel=2)
        raise


def empty_cache(info: DeviceInfo) -> None:
    if info.kind == "cuda":
        torch.cuda.empty_cache()
    elif info.kind == "mps":
        torch.mps.empty_cache()


def memory_mb(info: DeviceInfo) -> float:
    if info.kind == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    if info.kind == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0
