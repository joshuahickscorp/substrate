"""Studio acquisition layer: the pre-Studio readiness surface. Registries (datasets + models),
device profiles + kill switches, the 1 TB knapsack planner, the dry-run downloader orchestrator,
data cards + license ledger, synthetic control expansion, and the one pipeline that ties them
together (plan/acquire/validate/cache/run/optimize/report + local-max). Nothing here trains or
downloads heavy assets by default; heavy acquisition is gated behind execute + budget + license."""

from __future__ import annotations

from .artifact_bundle import build_artifact_index, preset_paths, write_artifact_index
from .dr1_schedule import build_dr1_schedule_plan, daemon_plan_from_dr1_schedule_plan, load_encode_schedule
from .encode_scheduler import EncodeBenchmark, format_plan, plan_encode
from .long_run import DaemonJob, run_daemon, validate_plan_contract, write_plan_template
from .memory_envelope import MemorySampler, memory_snapshot, summarize_samples
from .native_lanes import build_native_lane_manifest, write_native_daemon_plan, write_native_manifest
from .profiles import M3PRO_LOCAL_MAX, PROFILES, STUDIO, Profile, get_profile, list_profiles
from .transfer_check import TransferCheckConfig, run_transfer_check, write_transfer_report
from .wave0_report import build_wave0_report, render_markdown, upsert_report_block

__all__ = [
    "EncodeBenchmark",
    "plan_encode",
    "format_plan",
    "DaemonJob",
    "run_daemon",
    "validate_plan_contract",
    "write_plan_template",
    "build_artifact_index",
    "write_artifact_index",
    "preset_paths",
    "build_dr1_schedule_plan",
    "daemon_plan_from_dr1_schedule_plan",
    "load_encode_schedule",
    "MemorySampler",
    "memory_snapshot",
    "summarize_samples",
    "build_native_lane_manifest",
    "write_native_manifest",
    "write_native_daemon_plan",
    "TransferCheckConfig",
    "run_transfer_check",
    "write_transfer_report",
    "build_wave0_report",
    "render_markdown",
    "upsert_report_block",
    "Profile",
    "STUDIO",
    "M3PRO_LOCAL_MAX",
    "PROFILES",
    "get_profile",
    "list_profiles",
]
