"""Device profiles and kill switches (Frontier 3B + Frontier 15). One profile bundles every
hard limit that keeps an acquisition/cache/run from filling a disk or launching an unbudgeted
sweep, so the SAME safety logic serves the tiny current laptop and the future 1 TB Studio: only
the numbers differ, never the enforcement path.

A profile is a frozen dataclass of caps (disk, download, fixtures, clips, free-disk reserve,
run count, wall time, source counts, allowed tiers, manual-auth allowance, dry-run default). The
kill-switch helpers (within_budget, free_disk_ok, clamp_*) read those caps and answer one
question: is this action inside the envelope. They never act; the caller decides. The two shipped
profiles are STUDIO (1 TB, 900 GB usable) and M3PRO_LOCAL_MAX (the current device, ~18 GB unified
memory, minutes not hours, dry-run by default for anything heavy).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT


@dataclass(frozen=True)
class Profile:
    """Every hard limit for one execution context. Heavy work checks these before it runs.

    disk_total_gb / reserve_gb: the disk the plan may fill and the headroom kept for OS, venv,
      runs, reports, temp decode files. usable = disk_total - reserve is what the planner spends.
    download_budget_gb: the DEFAULT acquisition budget (the planner targets this).
    download_hard_cap_gb: a real download may never exceed this even with an explicit --budget-gb,
      unless the caller raises the cap deliberately (the kill switch the user cannot fat-finger past).
    fixture_budget_gb / raw_smoke_gb: caps on locally GENERATED control video and on tiny real
      raw-video smoke pulls.
    max_cache_clips: clips a single cache build may write here (the laptop clip ceiling).
    min_free_disk_gb: refuse to start if the filesystem has less free than this.
    max_run_count / max_wall_min: campaign run-unit and wall-time ceilings.
    max_source_count / max_per_source_gb: breadth + per-source caps for the planner.
    allowed_tiers: campaign tiers permitted here (C only on the laptop; C/E/R on the Studio).
    allow_manual_auth: may the planner SELECT sources that require signed terms (Studio yes once
      the user has access; laptop no, they stay deferred).
    dry_run_default: heavy actions (real downloads, full runs) default to dry-run under this profile.
    """

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

    @property
    def usable_gb(self) -> float:
        """Disk the planner may actually spend: total minus the reserve, never below zero."""
        return max(0.0, self.disk_total_gb - self.reserve_gb)

    def within_budget(self, requested_gb: float) -> tuple[bool, str]:
        """Is a requested download size inside BOTH the requested budget and the hard cap. The hard
        cap is the floor of safety: even a generous --budget-gb cannot push a real download past it."""
        if requested_gb < 0:
            return False, "requested GB is negative"
        if requested_gb > self.download_hard_cap_gb:
            return (
                False,
                f"{requested_gb:g} GB exceeds hard cap {self.download_hard_cap_gb:g} GB ({self.name})",
            )
        return True, f"{requested_gb:g} GB within hard cap {self.download_hard_cap_gb:g} GB"

    def effective_budget_gb(self, requested_gb: float | None) -> float:
        """The budget to enforce: the requested value clamped to the hard cap, or the profile
        default when none is requested. Never returns more than the hard cap, and rejects a
        NEGATIVE request (mirroring within_budget) so a bad --budget-gb fails fast instead of
        producing a negative-budget, all-skipped run that reads as budget exhaustion."""
        if requested_gb is None:
            return min(self.download_budget_gb, self.download_hard_cap_gb)
        want = float(requested_gb)
        if want < 0:
            raise ValueError(f"requested budget {want:g} GB is negative ({self.name})")
        return min(want, self.download_hard_cap_gb)

    def free_disk_ok(self, root: Path | None = None) -> tuple[bool, float]:
        """(enough free disk, free_gb). Refuses to proceed below min_free_disk_gb so a run can never
        be the thing that fills the disk."""
        du = shutil.disk_usage(root or REPO_ROOT)
        free_gb = du.free / 1e9
        return free_gb >= self.min_free_disk_gb, free_gb

    def clamp_clips(self, requested: int) -> int:
        """Clamp a requested clip count to max_cache_clips (and to >= 0)."""
        return max(0, min(int(requested), self.max_cache_clips))

    def clamp_runs(self, requested: int) -> int:
        """Clamp a requested run count to max_run_count (and to >= 0)."""
        return max(0, min(int(requested), self.max_run_count))

    def tier_allowed(self, tier: str) -> bool:
        return tier in self.allowed_tiers

    def as_dict(self) -> dict:
        """Flat dict for manifests/reports (the enforced envelope, recorded with every artifact)."""
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
        }


# The future 1 TB Mac Studio: breadth at scale, full gated campaign, large acquisitions.
# 900 GB usable of 1000, C+E enabled by default (R stays opt-in: it is rented CUDA, not the Studio).
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
)

# The current device (M3 Pro class, ~18 GB unified memory). Minutes not hours, tiny downloads,
# generated controls, smoke caches. Every heavy thing is dry-run by default. These are the goal's
# stated current-device limits, made enforceable.
M3PRO_LOCAL_MAX = Profile(
    name="m3pro-local-max",
    disk_total_gb=80.0,  # the slice of laptop disk this lane is allowed to consider
    reserve_gb=10.0,
    download_budget_gb=10.0,
    download_hard_cap_gb=25.0,
    fixture_budget_gb=2.0,
    raw_smoke_gb=5.0,
    max_cache_clips=128,
    min_free_disk_gb=60.0,
    max_run_count=64,
    max_wall_min=90,
    max_source_count=8,
    max_per_source_gb=5.0,
    allowed_tiers=frozenset({"C"}),
    allow_manual_auth=False,
    dry_run_default=True,
)

PROFILES: dict[str, Profile] = {p.name: p for p in (STUDIO, M3PRO_LOCAL_MAX)}
# Friendly aliases so a caller need not remember the exact slug.
_ALIASES = {
    "studio": "studio-1tb",
    "1tb": "studio-1tb",
    "m3pro": "m3pro-local-max",
    "local": "m3pro-local-max",
    "local-max": "m3pro-local-max",
    "laptop": "m3pro-local-max",
}


def get_profile(name: str) -> Profile:
    """Resolve a profile by exact name or a friendly alias. Raises with the valid set on a miss."""
    key = str(name).strip().lower()
    if key in PROFILES:
        return PROFILES[key]
    if key in _ALIASES:
        return PROFILES[_ALIASES[key]]
    raise ValueError(
        f"unknown profile {name!r}; choose from {sorted(PROFILES)} or aliases {sorted(_ALIASES)}"
    )


def list_profiles() -> list[dict]:
    """Every shipped profile as a flat dict (for `studio_pipeline.py profiles`)."""
    return [p.as_dict() for p in PROFILES.values()]
