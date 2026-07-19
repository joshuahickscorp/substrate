
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT


@dataclass(frozen=True)
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

    def within_budget(self, requested_gb: float) -> tuple[bool, str]:
        if requested_gb < 0:
            return False, "requested GB is negative"
        if requested_gb > self.download_hard_cap_gb:
            return (
                False,
                f"{requested_gb:g} GB exceeds hard cap {self.download_hard_cap_gb:g} GB ({self.name})",
            )
        return True, f"{requested_gb:g} GB within hard cap {self.download_hard_cap_gb:g} GB"

    def effective_budget_gb(self, requested_gb: float | None) -> float:
        if requested_gb is None:
            return min(self.download_budget_gb, self.download_hard_cap_gb)
        want = float(requested_gb)
        if want < 0:
            raise ValueError(f"requested budget {want:g} GB is negative ({self.name})")
        return min(want, self.download_hard_cap_gb)

    def free_disk_ok(self, root: Path | None = None) -> tuple[bool, float]:
        du = shutil.disk_usage(root or REPO_ROOT)
        free_gb = du.free / 1e9
        return free_gb >= self.min_free_disk_gb, free_gb

    def clamp_clips(self, requested: int) -> int:
        return max(0, min(int(requested), self.max_cache_clips))

    def clamp_runs(self, requested: int) -> int:
        return max(0, min(int(requested), self.max_run_count))

    def tier_allowed(self, tier: str) -> bool:
        return tier in self.allowed_tiers

    def host_compatibility(
        self,
        *,
        host: dict[str, Any] | None = None,
        disk_root: Path | None = None,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        if host is None:
            from ..devices import apple_silicon_info

            host = apple_silicon_info()
        disk_total_gb = shutil.disk_usage(disk_root or REPO_ROOT).total / 1e9
        memory_gb = float(host.get("unified_memory_gb") or 0.0)
        problems: list[str] = []
        if self.require_apple_silicon and not bool(host.get("is_apple_silicon")):
            problems.append("Apple Silicon required")
        if memory_gb < self.min_host_unified_memory_gb:
            problems.append(
                f"unified memory {memory_gb:.1f} GB below {self.min_host_unified_memory_gb:.1f} GB"
            )
        if disk_total_gb < self.min_host_disk_gb:
            problems.append(f"disk {disk_total_gb:.1f} GB below {self.min_host_disk_gb:.1f} GB")
        measured = {
            "is_apple_silicon": bool(host.get("is_apple_silicon")),
            "chip": host.get("chip"),
            "unified_memory_gb": host.get("unified_memory_gb"),
            "disk_total_gb": round(disk_total_gb, 1),
        }
        return not problems, problems, measured

    def as_dict(self) -> dict:
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


STUDIO = Profile(
    name="studio-1tb",
    disk_total_gb=1000.0,
    reserve_gb=100.0,
    download_budget_gb=900.0,
    download_hard_cap_gb=950.0,
    fixture_budget_gb=50.0,
    raw_smoke_gb=900.0,
    max_cache_clips=2_000_000,
    min_free_disk_gb=60.0,
    max_run_count=100_000,
    max_wall_min=60 * 48,
    max_source_count=24,
    max_per_source_gb=400.0,
    allowed_tiers=frozenset({"C", "E"}),
    allow_manual_auth=True,
    dry_run_default=True,
    require_apple_silicon=True,
    min_host_unified_memory_gb=90.0,
    min_host_disk_gb=900.0,
    procurement_status="unverified-procurement-scenario",
)

M1_ULTRA = Profile(
    name="studio-m1ultra",
    disk_total_gb=8000.0,
    reserve_gb=800.0,
    download_budget_gb=4000.0,
    download_hard_cap_gb=6000.0,
    fixture_budget_gb=200.0,
    raw_smoke_gb=2000.0,
    max_cache_clips=5_000_000,
    min_free_disk_gb=250.0,
    max_run_count=500_000,
    max_wall_min=60 * 24 * 7,
    max_source_count=48,
    max_per_source_gb=1500.0,
    allowed_tiers=frozenset({"C", "E"}),
    allow_manual_auth=True,
    dry_run_default=True,
    require_apple_silicon=True,
    min_host_unified_memory_gb=120.0,
    min_host_disk_gb=7000.0,
    procurement_status="unverified-procurement-scenario",
)

_M3PRO_OS_RESERVE_GB = 10.0
_M3PRO_DOWNLOAD_HARD_CAP_GB = 25.0
_M3PRO_WORKING_HEADROOM_GB = 5.0
M3PRO_LOCAL_MIN_FREE_DISK_GB = _M3PRO_OS_RESERVE_GB + _M3PRO_DOWNLOAD_HARD_CAP_GB + _M3PRO_WORKING_HEADROOM_GB

M3PRO_LOCAL_MAX = Profile(
    name="m3pro-local-max",
    disk_total_gb=80.0,  # the slice of laptop disk this lane is allowed to consider
    reserve_gb=_M3PRO_OS_RESERVE_GB,
    download_budget_gb=10.0,
    download_hard_cap_gb=_M3PRO_DOWNLOAD_HARD_CAP_GB,
    fixture_budget_gb=2.0,
    raw_smoke_gb=5.0,
    max_cache_clips=128,
    min_free_disk_gb=M3PRO_LOCAL_MIN_FREE_DISK_GB,
    max_run_count=64,
    max_wall_min=300,
    max_source_count=8,
    max_per_source_gb=5.0,
    allowed_tiers=frozenset({"C"}),
    allow_manual_auth=False,
    dry_run_default=True,
    require_apple_silicon=True,
    min_host_unified_memory_gb=15.0,
    min_host_disk_gb=70.0,
    procurement_status="measured-current-host-envelope",
)

PROFILES: dict[str, Profile] = {p.name: p for p in (M1_ULTRA, STUDIO, M3PRO_LOCAL_MAX)}
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
    if key in PROFILES:
        return PROFILES[key]
    if key in _ALIASES:
        return PROFILES[_ALIASES[key]]
    raise ValueError(
        f"unknown profile {name!r}; choose from {sorted(PROFILES)} or aliases {sorted(_ALIASES)}"
    )


def list_profiles() -> list[dict]:
    return [p.as_dict() for p in PROFILES.values()]


def is_studio_profile(name: str) -> bool:
    try:
        return get_profile(name).procurement_status == "unverified-procurement-scenario"
    except ValueError:
        return False
