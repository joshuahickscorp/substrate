from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    disk_total_gb: float
    reserve_gb: float
    download_budget_gb: float
    download_hard_cap_gb: float
    fixture_budget_gb: float
    raw_smoke_gb: float
    max_cache_clips: int
    min_free_disk_gb: float
    max_run_count: int
    max_wall_min: int
    max_source_count: int
    max_per_source_gb: float
    allowed_tiers: frozenset[str] = field(default_factory=lambda: frozenset({"C"}))
    allow_manual_auth: bool = False
    dry_run_default: bool = True
    require_apple_silicon: bool = False
    min_host_unified_memory_gb: float = 0.0
    min_host_disk_gb: float = 0.0
    procurement_status: str = "current-host"

    @property
    def usable_gb(self) -> float:
        return max(0.0, self.disk_total_gb - self.reserve_gb)

    def free_disk_ok(self, root: Path | None = None) -> tuple[bool, float]:
        return (free_gb := shutil.disk_usage(root or REPO_ROOT).free / 1e9) >= self.min_free_disk_gb, free_gb

    def host_compatibility(
        self, *, host: dict[str, Any] | None = None, disk_root: Path | None = None
    ) -> tuple[bool, list[str], dict[str, Any]]:
        if host is None:
            from ..devices import apple_silicon_info

            host = apple_silicon_info()
        disk_gb = shutil.disk_usage(disk_root or REPO_ROOT).total / 1e9
        memory_gb = float(host.get("unified_memory_gb") or 0.0)
        problems = []
        if self.require_apple_silicon and not host.get("is_apple_silicon"):
            problems.append("Apple Silicon required")
        if memory_gb < self.min_host_unified_memory_gb:
            problems.append(
                f"unified memory {memory_gb:.1f} GB below {self.min_host_unified_memory_gb:.1f} GB"
            )
        if disk_gb < self.min_host_disk_gb:
            problems.append(f"disk {disk_gb:.1f} GB below {self.min_host_disk_gb:.1f} GB")
        measured = {
            "is_apple_silicon": bool(host.get("is_apple_silicon")),
            "chip": host.get("chip"),
            "unified_memory_gb": host.get("unified_memory_gb"),
            "disk_total_gb": round(disk_gb, 1),
        }
        return not problems, problems, measured

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "disk_total_gb": self.disk_total_gb,
            "reserve_gb": self.reserve_gb,
            "usable_gb": self.usable_gb,
            "download_budget_gb": self.download_budget_gb,
            "download_hard_cap_gb": self.download_hard_cap_gb,
            "fixture_budget_gb": self.fixture_budget_gb,
            "raw_smoke_gb": self.raw_smoke_gb,
            "max_cache_clips": self.max_cache_clips,
            "min_free_disk_gb": self.min_free_disk_gb,
            "max_run_count": self.max_run_count,
            "max_wall_min": self.max_wall_min,
            "max_source_count": self.max_source_count,
            "max_per_source_gb": self.max_per_source_gb,
            "allowed_tiers": sorted(self.allowed_tiers),
            "allow_manual_auth": self.allow_manual_auth,
            "dry_run_default": self.dry_run_default,
            "require_apple_silicon": self.require_apple_silicon,
            "min_host_unified_memory_gb": self.min_host_unified_memory_gb,
            "min_host_disk_gb": self.min_host_disk_gb,
            "procurement_status": self.procurement_status,
        }


def _profile(
    name: str,
    storage: tuple[float, float, float, float, float, float],
    campaign: tuple[int, float, int, int, int, float],
    host: tuple[float, float],
    *,
    tiers: tuple[str, ...] = ("C",),
    manual: bool = False,
    status: str,
) -> Profile:
    return Profile(name, *storage, *campaign, frozenset(tiers), manual, True, True, *host, status)  # type: ignore[arg-type]


_PROFILE_ROWS = (
    _profile(
        "studio-m1ultra",
        (8000.0, 800.0, 4000.0, 6000.0, 200.0, 2000.0),
        (5_000_000, 250.0, 500_000, 10_080, 48, 1500.0),
        (120.0, 7000.0),
        tiers=("C", "E"),
        manual=True,
        status="unverified-procurement-scenario",
    ),
    _profile(
        "studio-1tb",
        (1000.0, 100.0, 900.0, 950.0, 50.0, 900.0),
        (2_000_000, 60.0, 100_000, 2880, 24, 400.0),
        (90.0, 900.0),
        tiers=("C", "E"),
        manual=True,
        status="unverified-procurement-scenario",
    ),
    _profile(
        "m3pro-local-max",
        (80.0, 10.0, 10.0, 25.0, 2.0, 5.0),
        (128, 40.0, 64, 300, 8, 5.0),
        (15.0, 70.0),
        status="measured-current-host-envelope",
    ),
)
PROFILES = {profile.name: profile for profile in _PROFILE_ROWS}
_ALIASES = {
    "m1ultra": "studio-m1ultra",
    "8tb": "studio-m1ultra",
    "studio-max": "studio-m1ultra",
    "studio": "studio-1tb",
    "1tb": "studio-1tb",
    "m3pro": "m3pro-local-max",
    "local": "m3pro-local-max",
    "local-max": "m3pro-local-max",
    "laptop": "m3pro-local-max",
}


def get_profile(name: str) -> Profile:
    key = str(name).strip().lower()
    try:
        return PROFILES[_ALIASES.get(key, key)]
    except KeyError as error:
        choices = f"{sorted(PROFILES)} or aliases {sorted(_ALIASES)}"
        raise ValueError(f"unknown profile {name!r}; choose from {choices}") from error


def list_profiles() -> list[dict[str, Any]]:
    return [profile.as_dict() for profile in PROFILES.values()]
