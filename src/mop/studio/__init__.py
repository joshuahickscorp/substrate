"""Studio acquisition layer: the pre-Studio readiness surface. Registries (datasets + models),
device profiles + kill switches, the 1 TB knapsack planner, the dry-run downloader orchestrator,
data cards + license ledger, synthetic control expansion, and the one pipeline that ties them
together (plan/acquire/validate/cache/run/optimize/report + local-max). Nothing here trains or
downloads heavy assets by default; heavy acquisition is gated behind execute + budget + license."""

from __future__ import annotations

from .encode_scheduler import EncodeBenchmark, format_plan, plan_encode
from .long_run import DaemonJob, run_daemon, write_plan_template
from .profiles import M3PRO_LOCAL_MAX, PROFILES, STUDIO, Profile, get_profile, list_profiles
from .transfer_check import TransferCheckConfig, run_transfer_check, write_transfer_report

__all__ = [
    "EncodeBenchmark",
    "plan_encode",
    "format_plan",
    "DaemonJob",
    "run_daemon",
    "write_plan_template",
    "TransferCheckConfig",
    "run_transfer_check",
    "write_transfer_report",
    "Profile",
    "STUDIO",
    "M3PRO_LOCAL_MAX",
    "PROFILES",
    "get_profile",
    "list_profiles",
]
