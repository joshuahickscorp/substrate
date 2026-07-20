from __future__ import annotations

import functools
import os
import platform
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # cpu | mps | cuda
    device: torch.device
    supports_amp: bool
    supports_pin: bool
    note: str
    amp_dtype: str = "none"  # float16 on mps/cuda, none on cpu


def _cuda_ok() -> bool:
    return torch.cuda.is_available()


def _mps_ok() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def apple_silicon_info() -> dict:
    is_as = platform.system() == "Darwin" and platform.machine() == "arm64"
    if not is_as:
        return {"is_apple_silicon": False}
    mem = _sysctl("hw.memsize")
    return {
        "is_apple_silicon": True,
        "chip": _sysctl("machdep.cpu.brand_string") or "Apple Silicon",
        "performance_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "unified_memory_gb": round(int(mem) / 1e9, 1) if mem else None,
        "mps_available": _mps_ok(),
    }


def resolve(requested: str = "auto") -> DeviceInfo:
    req = (requested or "auto").lower()
    note = ""
    if req == "cuda" and not _cuda_ok():
        note = "cuda requested but unavailable; "
        req = "auto"
    if req == "mps" and not _mps_ok():
        note = "mps requested but unavailable; "
        req = "auto"
    if req == "auto":
        kind = "mps" if _mps_ok() else "cuda" if _cuda_ok() else "cpu"
    else:
        kind = req
    amp_dtype = "none"
    if kind == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported ops run on cpu
        torch.set_float32_matmul_precision("high")
        amp_dtype = "float16"  # Metal does fp16 well; halves memory + speeds inference
    elif kind == "cuda":
        torch.set_float32_matmul_precision("high")
        amp_dtype = "float16"
    return DeviceInfo(
        kind=kind,
        device=torch.device(kind),
        supports_amp=kind in ("mps", "cuda"),
        supports_pin=kind == "cuda",
        note=note + f"resolved {kind}",
        amp_dtype=amp_dtype,
    )


@contextmanager
def autocast(info: DeviceInfo, enabled: bool = True):
    if enabled and info.supports_amp:
        with torch.autocast(device_type=info.kind, dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield
