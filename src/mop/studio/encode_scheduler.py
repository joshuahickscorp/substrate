"""Profile-aware encode scheduler.

The Wave-0 benchmark answers "how fast is CPU vs MPS on this machine". This module turns that
measurement into a launch plan: device, CPU worker count, cache footprint, disk reserve gates,
wall-clock gate, and checkpoint cadence. It never loads a model and never encodes a clip.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..devices import apple_silicon_info
from ..substrate.storage import estimate_for_encoder, human_bytes
from .profiles import Profile, get_profile


@dataclass(frozen=True)
class EncodeBenchmark:
    """Measured encode speeds from Wave 0. CPU is one worker; MPS is one-device wall s/clip."""

    cpu_s_per_clip: float | None
    mps_s_per_clip: float | None = None
    n_clips: int = 0
    source: str = "manual"


DEFAULT_CPU_WORKERS = {
    "m3pro-local-max": 1,
    "studio-1tb": 8,
    "studio-m1ultra": 16,
}


def plan_encode(
    *,
    profile_name: str,
    benchmark: EncodeBenchmark | dict[str, Any],
    encoder_config: dict[str, Any],
    requested_clips: int,
    dense: bool = False,
    dense_tokens: int | None = None,
    dtype: str = "float32",
    cpu_workers: int | None = None,
    checkpoint_min: int = 30,
    root: Path | None = None,
    free_gb: float | None = None,
) -> dict[str, Any]:
    """Return an encode launch plan with profile kill switches.

    `requested_clips` is clamped to the profile cap. Disk gates use the effective clip count and
    block when the estimated cache would leave less than the profile's free-disk floor.
    """
    memory_envelope = benchmark.get("memory_envelope") if isinstance(benchmark, dict) else None
    profile = get_profile(profile_name)
    bench = _bench(benchmark)
    root = Path(root or REPO_ROOT)
    effective_clips = profile.clamp_clips(requested_clips)
    workers = _cpu_workers(profile, cpu_workers)
    cache = estimate_for_encoder(
        encoder_config,
        n_clips=effective_clips,
        dense=dense,
        dtype=dtype,
        dense_tokens=dense_tokens,
    )
    free = _free_gb(root) if free_gb is None else float(free_gb)
    cache_gb = cache["bytes"] / 1e9
    projected_free = free - cache_gb

    candidates = _candidates(bench, workers)
    winner = min(candidates, key=lambda c: c["wall_s_per_clip"]) if candidates else None
    if winner is None:
        estimated_wall_min = None
        checkpoint_every_clips = None
    else:
        estimated_wall_min = effective_clips * winner["wall_s_per_clip"] / 60.0
        checkpoint_every_clips = max(1, math.ceil((checkpoint_min * 60.0) / winner["wall_s_per_clip"]))

    gates = [
        {
            "name": "profile_exists",
            "ok": True,
            "detail": profile.name,
        },
        {
            "name": "clip_cap",
            "ok": requested_clips == effective_clips,
            "detail": (
                f"requested {requested_clips}, effective {effective_clips}, cap {profile.max_cache_clips}"
            ),
            "warning_only": True,
        },
        {
            "name": "free_disk_start",
            "ok": free >= profile.min_free_disk_gb,
            "detail": f"{free:.1f} GB free, min {profile.min_free_disk_gb:.0f} GB",
        },
        {
            "name": "free_disk_after_cache",
            "ok": projected_free >= profile.min_free_disk_gb,
            "detail": (
                f"{projected_free:.1f} GB projected after {cache['human']} cache, "
                f"min {profile.min_free_disk_gb:.0f} GB"
            ),
        },
        {
            "name": "speed_measurement",
            "ok": winner is not None,
            "detail": "at least one measured path" if winner is not None else "no usable CPU or MPS speed",
        },
        {
            "name": "wall_clock",
            "ok": estimated_wall_min is not None and estimated_wall_min <= profile.max_wall_min,
            "detail": (
                "no speed estimate"
                if estimated_wall_min is None
                else f"{estimated_wall_min:.1f} min vs cap {profile.max_wall_min} min"
            ),
        },
    ]
    hard_ok = all(g["ok"] for g in gates if not g.get("warning_only"))

    command_device = winner["device"] if winner else "blocked"
    command = [
        ".venv/bin/python",
        "scripts/cache_real_encoder.py",
        f"device={command_device}",
        "+classes=auto",
        "+per_class=auto",
        f"+batch={winner['batch_size'] if winner else 1}",
    ]

    return {
        "profile": profile.as_dict(),
        "machine": apple_silicon_info(),
        "benchmark": asdict(bench),
        "memory_envelope": memory_envelope,
        "requested_clips": int(requested_clips),
        "effective_clips": int(effective_clips),
        "encoder": {
            "name": str(encoder_config.get("name", "?")),
            "embed_dim": int(encoder_config["embed_dim"]),
            "dense": bool(dense),
            "dtype": dtype,
            "tokens_per_clip": cache["tokens_per_clip"],
        },
        "cache_estimate": cache,
        "disk": {
            "free_gb": round(free, 3),
            "cache_gb_decimal": round(cache_gb, 3),
            "projected_free_gb": round(projected_free, 3),
            "required_floor_gb": profile.min_free_disk_gb,
        },
        "candidates": candidates,
        "winner": winner,
        "checkpoint": {
            "every_min": int(checkpoint_min),
            "every_clips": checkpoint_every_clips,
            "resume_contract": "write shard receipt after each checkpoint window before advancing",
        },
        "thermal_pacing": {
            "mode": "monitor",
            "heartbeat_every_min": 5,
            "pause_s": 0,
            "note": "scheduler records pacing hooks; measured thermal throttling can raise pause_s",
        },
        "gates": gates,
        "ok_to_launch": hard_ok,
        "blocked_reasons": [g["detail"] for g in gates if not g["ok"] and not g.get("warning_only")],
        "next_command": " ".join(command),
    }


def benchmark_from_autoselect(result: dict[str, Any]) -> EncodeBenchmark:
    """Convert `scripts/mop_encode_autoselect.py` output into an EncodeBenchmark."""
    cpu = _num(result.get("cpu_s_per_clip"))
    mps = _num(result.get("mps"))
    return EncodeBenchmark(cpu_s_per_clip=cpu, mps_s_per_clip=mps, n_clips=int(result.get("n_clips") or 0))


def _bench(raw: EncodeBenchmark | dict[str, Any]) -> EncodeBenchmark:
    if isinstance(raw, EncodeBenchmark):
        return raw
    if "winner" in raw or "cpu_s_per_clip" in raw:
        return benchmark_from_autoselect(raw)
    return EncodeBenchmark(
        cpu_s_per_clip=_num(raw.get("cpu")),
        mps_s_per_clip=_num(raw.get("mps")),
        n_clips=int(raw.get("n_clips") or 0),
        source=str(raw.get("source", "manual")),
    )


def _num(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _cpu_workers(profile: Profile, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    return DEFAULT_CPU_WORKERS.get(profile.name, 1)


def _candidates(bench: EncodeBenchmark, workers: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if bench.cpu_s_per_clip is not None and bench.cpu_s_per_clip > 0:
        out.append(
            {
                "device": "cpu",
                "workers": workers,
                "batch_size": 1,
                "single_worker_s_per_clip": bench.cpu_s_per_clip,
                "wall_s_per_clip": bench.cpu_s_per_clip / workers,
            }
        )
    if bench.mps_s_per_clip is not None and bench.mps_s_per_clip > 0:
        out.append(
            {
                "device": "mps",
                "workers": 1,
                "batch_size": 1,
                "single_worker_s_per_clip": bench.mps_s_per_clip,
                "wall_s_per_clip": bench.mps_s_per_clip,
            }
        )
    return out


def _free_gb(root: Path) -> float:
    return shutil.disk_usage(root).free / 1e9


def format_plan(plan: dict[str, Any]) -> str:
    """Small human-readable summary for logs and reports."""
    winner = plan.get("winner")
    if not winner:
        route = "blocked"
    else:
        route = f"{winner['device']} ({winner['wall_s_per_clip']:.3f} s/clip wall)"
    cache = plan["cache_estimate"]
    status = "OK" if plan["ok_to_launch"] else "BLOCKED"
    return (
        f"{status}: {plan['effective_clips']} clips, cache {cache['human']} "
        f"({human_bytes(cache['per_clip_bytes'])}/clip), route {route}"
    )
