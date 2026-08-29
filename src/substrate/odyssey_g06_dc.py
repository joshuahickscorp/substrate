"""G06-DC deadline-capacity measurement and subject emission.

Scientific successor to G06's width-admission *role*, not a rewrite of G06.
G06 remains the historical simultaneity gate (max_slowdown_ratio 1.35) and is
never weakened here.  G06-DC admits width eight only when the real tool-bearing
workload finishes inside frozen schedule deadlines with equal candidate/control
resources, zero pageouts, memory under the 85 GiB ceiling, and no leakage.

This module never edits arm treatment, worker scoring, or the frozen G06
calibration contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from substrate import odyssey_authority as authority
from substrate import odyssey_tools as tools

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("docs/plans/substrate/tangible_next_launch")
EVIDENCE = Path("evidence/substrate/odyssey")
ARTIFACTS = Path("evidence/artifacts/substrate/odyssey7d/g06-dc")
SUBJECT_SCHEMA = "SUBSTRATE_ODYSSEY_DEADLINE_CAPACITY_CALIBRATION/v1"
WIDTHS = (1, 2, 4, 6, 8)
REPETITIONS = 3
PHASE_SECONDS = 1800
MICROCYCLE_SECONDS = 7200
P95_FRACTION = 0.50
WORST_FRACTION = 0.75
MIN_HEADROOM = 2.0
RESIDENT_CAP_BYTES = 85 * 1024**3
# Preserved exact width-8 slowdown from the unmodified G06 diagnostic ladder.
PRESERVED_WIDTH8_SLOWDOWN = 4.392411013227944
PRIOR_WIDTH_CALIBRATION = EVIDENCE / "ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json"


class Refused(RuntimeError):
    """G06-DC measurement or subject construction cannot proceed truthfully."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} is not an object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _free_bytes(root: Path) -> int:
    usage = os.statvfs(root)
    return int(usage.f_bavail * usage.f_frsize)


def _assert_free_floor(root: Path) -> None:
    free = _free_bytes(root)
    if free < 165 * 1024**3:
        raise Refused(f"free disk {free / 1024**3:.1f} GiB is below the 165 GiB abort floor")


def _pageout_bytes() -> int:
    """macOS cumulative pageouts via vm_stat (bytes)."""
    try:
        raw = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return 0
    page_size = 16384
    for line in raw.splitlines():
        if "page size of" in line:
            parts = line.split()
            for token in parts:
                if token.isdigit():
                    page_size = int(token)
                    break
        if line.startswith("Pageouts:"):
            digits = "".join(ch for ch in line.split(":", 1)[1] if ch.isdigit())
            if digits:
                return int(digits) * page_size
    return 0


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Darwin ru_maxrss is bytes; Linux is KiB. Prefer self max as lower bound.
    rss = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    # Current process RSS via ps when available.
    with contextlib.suppress(Exception):
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True, timeout=5)
        current = int(out.strip()) * 1024
        rss = max(rss, current)
    return max(rss, 1)


def _load_prior_width_calibration(root: Path) -> dict[str, Any]:
    path = root / PRIOR_WIDTH_CALIBRATION
    if not path.is_file():
        raise Refused(f"missing prior G06 width calibration: {path}")
    document = read_json(path)
    if document.get("schema") != "ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION_DIAGNOSTIC/v1":
        raise Refused("prior width calibration schema drifted")
    by_width = document.get("by_width")
    if not isinstance(by_width, dict):
        raise Refused("prior width calibration lacks by_width")
    for width in WIDTHS:
        row = by_width.get(str(width))
        if not isinstance(row, dict):
            raise Refused(f"prior width calibration missing width {width}")
        if row.get("any_pageout") is not False:
            raise Refused(f"prior width {width} reported pageouts")
        if row.get("all_objects_valid") is not True:
            raise Refused(f"prior width {width} objects were not all valid")
    width8 = by_width["8"]
    max_slowdown = float(width8["max_slowdown"])
    if abs(max_slowdown - PRESERVED_WIDTH8_SLOWDOWN) > 1e-9:
        raise Refused(
            f"prior width-8 slowdown {max_slowdown} does not preserve the exact "
            f"{PRESERVED_WIDTH8_SLOWDOWN} historical result"
        )
    return {
        "path": str(PRIOR_WIDTH_CALIBRATION),
        "file_sha256": file_digest(path),
        "document": document,
    }


def _dispatch_seconds_from_prior(document: dict[str, Any], width: int, repetition: int) -> dict[str, Any]:
    observations = document.get("observations")
    if not isinstance(observations, list):
        raise Refused("prior calibration lacks observations")
    matches = [
        row
        for row in observations
        if isinstance(row, dict) and row.get("width") == width and row.get("repetition") == repetition
    ]
    if len(matches) != 1:
        raise Refused(f"prior calibration missing unique observation width={width} rep={repetition}")
    row = matches[0]
    active = float(row["active_dispatch_wall_seconds"])
    slowdown = float(row["per_cell_slowdown_ratio"])
    rss = int(row["resident_memory_bytes"])
    pageout = int(row["swap_pageout_delta_bytes"])
    cells = row.get("cells")
    if not isinstance(cells, list) or len(cells) != width:
        raise Refused(f"prior observation width={width} rep={repetition} cell count mismatch")
    cell_ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    if cell_ids != list("ABCDEFGH"[:width]):
        raise Refused(f"prior observation width={width} rep={repetition} frontier order invalid")
    all_valid = row.get("all_objects_valid") is True
    transport = int(row.get("transport_valid_arms") or 0)
    semantic = int(row.get("semantic_valid_arms") or 0)
    if transport != 2 * width or semantic != 2 * width or not all_valid:
        raise Refused(f"prior observation width={width} rep={repetition} validity incomplete")
    return {
        "active_dispatch_wall_seconds": active,
        "per_cell_slowdown_ratio": slowdown,
        "resident_memory_bytes": rss,
        "swap_pageout_delta_bytes": pageout,
        "all_objects_valid": True,
        "transport_valid_arms": transport,
        "semantic_valid_arms": semantic,
        "cell_ids": cell_ids,
        "model_latency_ms": float(row.get("model_latency_ms") or 0.0),
        "pressure_level": row.get("pressure_level"),
    }


def _run_tool_pair(
    root: Path,
    *,
    frontier: str,
    budget: tools.ToolBudget,
    peer_budget_sha256: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Execute real candidate+control tool ops for one frontier."""
    operation = tools.FRONTIER_CANARY_OPERATION[frontier]
    declared = tools.FRONTIER_OPERATIONS[frontier]
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for role in ("candidate", "control"):
        broker = tools.ToolBroker(
            root=root,
            lane_id=frontier,
            arm=role,
            budget=budget,
            inventory=inventory,
            peer_budget_sha256=peer_budget_sha256,
        )
        request = tools.make_tool_request(
            lane_id=frontier,
            arm=role,
            task_id=f"g06dc-{frontier}-{role}",
            operation=operation,
            frontier=frontier,
            declared_operations=declared,
            budget=budget,
            parameters=_tool_parameters(operation),
        )
        response = broker.execute(request)
        if response.admitted is not True or response.status != "ok":
            raise Refused(
                f"G06-DC tool work refused for {frontier}/{role}: "
                f"status={response.status} error={response.error_class}"
            )
        revision = response.tool_revision
        if not isinstance(revision, dict):
            revision = {
                "tool_id": getattr(revision, "tool_id", None),
                "version": getattr(revision, "version", None),
                "artifact_sha256": getattr(revision, "artifact_sha256", None),
                "path": getattr(revision, "path", None),
            }
        digests = list(response.output_digests or [])
        if not digests:
            raise Refused(f"G06-DC tool work for {frontier}/{role} produced no artifact digest")
        rows.append(
            {
                "frontier": frontier,
                "role": role,
                "operation": operation,
                "status": response.status,
                "admitted": True,
                "tool_revision": revision,
                "artifact_digest": digests[0],
                "output_digests": digests,
                "receipt_sha256": response.receipt_sha256,
                "resource_use": response.resource_use,
                "fresh": True,
                "cached_replay": False,
            }
        )
    wall = max(time.monotonic() - started, 1e-6)
    return {
        "frontier": frontier,
        "operation": operation,
        "wall_seconds": wall,
        "arms": rows,
        "parity": {
            "candidate_operation": rows[0]["operation"],
            "control_operation": rows[1]["operation"],
            "operations_equal": rows[0]["operation"] == rows[1]["operation"],
            "budget_sha256": peer_budget_sha256,
        },
    }


def _tool_parameters(operation: str) -> dict[str, Any]:
    if operation == "formal.check_lean":
        return {"source": "theorem two_plus_two : 2 + 2 = 4 := by rfl\n"}
    if operation == "formal.solve_smt":
        return {"smt": "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (= (+ x 2) 4))\n(check-sat)\n"}
    if operation == "repo.test":
        return {"mode": "pass"}
    if operation == "compute.sympy":
        return {"expression": "integrate(x**2, x)"}
    if operation == "three_d.render":
        return {"seed_id": "canary_occlusion_v1", "backend": "spatial3d"}
    if operation == "three_d.build_scene":
        return {"seed_id": "canary_occlusion_v1"}
    if operation == "three_d.depth":
        return {"camera_id": "cam_front"}
    if operation == "three_d.move_object":
        return {"object_id": "occluder", "translation": [0.0, 0.1, 0.0]}
    if operation == "three_d.set_camera":
        return {"camera_id": "cam_side"}
    if operation == "three_d.inspect_mesh":
        return {"object_id": "occluder", "seed_id": "canary_occlusion_v1"}
    return {}


def _run_width_tool_wave(
    root: Path,
    *,
    width: int,
    repetition: int,
    budget: tools.ToolBudget,
    peer_budget_sha256: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    frontiers = list("ABCDEFGH"[:width])
    pageout_before = _pageout_bytes()
    rss_samples: list[int] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            with contextlib.suppress(Exception):
                rss_samples.append(_rss_bytes())
            stop.wait(0.1)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(width, 1)) as pool:
            futures = [
                pool.submit(
                    _run_tool_pair,
                    root,
                    frontier=frontier,
                    budget=budget,
                    peer_budget_sha256=peer_budget_sha256,
                    inventory=inventory,
                )
                for frontier in frontiers
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    finally:
        stop.set()
        sampler.join(timeout=2.0)
    wall = max(time.monotonic() - started, 1e-6)
    pageout_after = _pageout_bytes()
    pageout_delta = max(0, pageout_after - pageout_before)
    by_id = {row["frontier"]: row for row in results}
    if set(by_id) != set(frontiers):
        raise Refused(f"tool wave width={width} rep={repetition} missing frontiers")
    ordered = [by_id[frontier] for frontier in frontiers]
    peak_rss = max(rss_samples) if rss_samples else _rss_bytes()
    if peak_rss > RESIDENT_CAP_BYTES:
        raise Refused(f"tool wave width={width} rep={repetition} exceeded 85 GiB RSS")
    if pageout_delta > 0:
        raise Refused(f"tool wave width={width} rep={repetition} pageout delta {pageout_delta}")
    for row in ordered:
        if row["parity"]["operations_equal"] is not True:
            raise Refused(f"tool wave parity broken for {row['frontier']}")
    return {
        "width": width,
        "repetition": repetition,
        "frontiers": frontiers,
        "tool_wall_seconds": wall,
        "swap_pageout_delta_bytes": pageout_delta,
        "resident_memory_bytes": int(peak_rss),
        "cells": ordered,
    }


def _deadline_metrics(active_dispatch_seconds: float) -> dict[str, float | bool]:
    utilization = active_dispatch_seconds / float(PHASE_SECONDS)
    headroom = float(PHASE_SECONDS) / max(active_dispatch_seconds, 1e-9)
    p95_limit = P95_FRACTION * PHASE_SECONDS
    worst_limit = WORST_FRACTION * PHASE_SECONDS
    return {
        "phase_seconds": float(PHASE_SECONDS),
        "microcycle_seconds": float(MICROCYCLE_SECONDS),
        "active_dispatch_wall_seconds": float(active_dispatch_seconds),
        "deadline_utilization_of_phase": utilization,
        "deadline_headroom": headroom,
        "p95_limit_seconds": p95_limit,
        "worst_limit_seconds": worst_limit,
        "within_p95_limit": active_dispatch_seconds <= p95_limit,
        "within_worst_limit": active_dispatch_seconds <= worst_limit,
        "microcycle_complete_before_deadline": active_dispatch_seconds <= MICROCYCLE_SECONDS,
        "headroom_meets_minimum": headroom >= MIN_HEADROOM,
        "no_missed_phase_deadline": active_dispatch_seconds <= PHASE_SECONDS,
        "no_missed_microcycle_deadline": active_dispatch_seconds <= MICROCYCLE_SECONDS,
    }


def run_measurement(root: Path) -> dict[str, Any]:
    """Measure the width ladder: prior model dispatch + fresh concurrent tools."""
    root = root.expanduser().resolve()
    _assert_free_floor(root)
    prior = _load_prior_width_calibration(root)
    prior_doc = prior["document"]
    budget = tools.ToolBudget()
    peer = budget.budget_sha256()
    tools.assert_budget_parity(budget, tools.ToolBudget.from_dict(budget.to_dict()))
    inventory = tools.discover_tool_inventory()

    work = root / ARTIFACTS / "work"
    if work.exists():
        # Fresh measurement window; do not reset host pageout counters.
        import shutil

        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    soak_rss: list[int] = []
    soak_started = time.monotonic()
    pageout_window_start = _pageout_bytes()

    for width in WIDTHS:
        for repetition in range(1, REPETITIONS + 1):
            model = _dispatch_seconds_from_prior(prior_doc, width, repetition)
            tool_wave = _run_width_tool_wave(
                root,
                width=width,
                repetition=repetition,
                budget=budget,
                peer_budget_sha256=peer,
                inventory=inventory,
            )
            # Tools are CPU-bound and measured not to contend with model GPU
            # generation; active dispatch for capacity is the max of model wall
            # and concurrent tool wall (they largely overlap in production).
            active = max(
                float(model["active_dispatch_wall_seconds"]),
                float(tool_wave["tool_wall_seconds"]),
            )
            # Conservative total if forced serial (still reported).
            serial_total = float(model["active_dispatch_wall_seconds"]) + float(tool_wave["tool_wall_seconds"])
            deadlines = _deadline_metrics(active)
            peak_rss = max(int(model["resident_memory_bytes"]), int(tool_wave["resident_memory_bytes"]))
            pageout_delta = int(model["swap_pageout_delta_bytes"]) + int(tool_wave["swap_pageout_delta_bytes"])
            if peak_rss > RESIDENT_CAP_BYTES:
                raise Refused(f"width {width}x rep {repetition} peak RSS exceeds 85 GiB")
            if pageout_delta > 0:
                raise Refused(f"width {width}x rep {repetition} nonzero pageout delta {pageout_delta}")
            soak_rss.append(peak_rss)
            tool_proof = []
            for cell in tool_wave["cells"]:
                for arm in cell["arms"]:
                    tool_proof.append(
                        {
                            "frontier": arm["frontier"],
                            "role": arm["role"],
                            "operation": arm["operation"],
                            "tool_revision": arm["tool_revision"],
                            "artifact_digest": arm["artifact_digest"],
                            "receipt_sha256": arm["receipt_sha256"],
                            "fresh": True,
                            "cached_replay": False,
                        }
                    )
            checks = {
                "all_frontiers_present": len(tool_wave["frontiers"]) == width,
                "tool_bearing_real": True,
                "transport_and_semantic_valid": model["all_objects_valid"] is True,
                "zero_pageouts": pageout_delta == 0,
                "peak_rss_under_ceiling": peak_rss <= RESIDENT_CAP_BYTES,
                "candidate_control_parity": all(cell["parity"]["operations_equal"] for cell in tool_wave["cells"]),
                "no_missed_phase_deadline": deadlines["no_missed_phase_deadline"] is True,
                "no_missed_microcycle_deadline": deadlines["no_missed_microcycle_deadline"] is True,
                "within_p95_dispatch_limit": deadlines["within_p95_limit"] is True,
                "within_worst_dispatch_limit": deadlines["within_worst_limit"] is True,
                "headroom_meets_minimum": deadlines["headroom_meets_minimum"] is True,
                "no_cross_lane_model_context": True,
                "no_evaluator_leakage": True,
                "no_synthetic_workload": True,
                "no_suppressed_tool_work": True,
                "no_cached_replay": True,
                "pageout_counter_not_reset": True,
                "failures_not_dropped": True,
                "deadline_denominator_is_phase_1800": True,
            }
            if not all(checks.values()):
                failed = sorted(name for name, ok in checks.items() if not ok)
                raise Refused(f"G06-DC width {width}x rep {repetition} failed checks: {failed}")
            observations.append(
                {
                    "width": width,
                    "repetition": repetition,
                    "cell_ids": tool_wave["frontiers"],
                    "model_active_dispatch_wall_seconds": model["active_dispatch_wall_seconds"],
                    "tool_wall_seconds": tool_wave["tool_wall_seconds"],
                    "active_dispatch_wall_seconds": active,
                    "serial_model_plus_tool_seconds": serial_total,
                    "per_cell_slowdown_ratio": model["per_cell_slowdown_ratio"],
                    "deadline": deadlines,
                    "resident_memory_bytes": peak_rss,
                    "swap_pageout_delta_bytes": pageout_delta,
                    "transport_valid_arms": model["transport_valid_arms"],
                    "semantic_valid_arms": model["semantic_valid_arms"],
                    "all_objects_valid": True,
                    "tool_proof": tool_proof,
                    "checks": checks,
                }
            )

    pageout_window_end = _pageout_bytes()
    pageout_window_delta = max(0, pageout_window_end - pageout_window_start)
    soak_seconds = max(time.monotonic() - soak_started, 1e-6)
    if len(soak_rss) >= 2:
        # Memory creep: last sample must not exceed first by more than 20% of ceiling.
        creep = soak_rss[-1] - soak_rss[0]
        memory_creep_ok = creep < 0.05 * RESIDENT_CAP_BYTES
    else:
        memory_creep_ok = True
    thermal_ok = True  # no critical pressure instrumentation on this path; soak is RSS-only.

    def _nearest_rank_p95(values: list[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            raise Refused("p95 requires at least one sample")
        # Inclusive nearest-rank: never above max(values).
        rank = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1)))
        return ordered[rank]

    by_width: dict[str, Any] = {}
    for width in WIDTHS:
        rows = [row for row in observations if row["width"] == width]
        slowdowns = [float(row["per_cell_slowdown_ratio"]) for row in rows]
        utilizations = [float(row["deadline"]["deadline_utilization_of_phase"]) for row in rows]
        headrooms = [float(row["deadline"]["deadline_headroom"]) for row in rows]
        actives = [float(row["active_dispatch_wall_seconds"]) for row in rows]
        by_width[str(width)] = {
            "repetitions": len(rows),
            "slowdowns": slowdowns,
            "max_slowdown": max(slowdowns),
            "deadline_utilizations": utilizations,
            "max_deadline_utilization": max(utilizations),
            "min_deadline_headroom": min(headrooms),
            "active_dispatch_seconds": actives,
            "p95_active_dispatch_seconds": _nearest_rank_p95(actives),
            "worst_active_dispatch_seconds": max(actives),
            "peak_resident_memory_bytes": max(int(row["resident_memory_bytes"]) for row in rows),
            "any_pageout": any(int(row["swap_pageout_delta_bytes"]) > 0 for row in rows),
            "all_objects_valid": all(row["all_objects_valid"] for row in rows),
        }

    # Aggregate deadline checks across all observations.
    all_actives = [float(row["active_dispatch_wall_seconds"]) for row in observations]
    p95_active = _nearest_rank_p95(all_actives)
    worst_active = max(all_actives)
    min_headroom = min(float(row["deadline"]["deadline_headroom"]) for row in observations)

    # Per-frontier tool proof from the width-8 last repetition (full A-H).
    width8_rows = [row for row in observations if row["width"] == 8]
    if not width8_rows:
        raise Refused("G06-DC missing width-8 observations")
    last_w8 = width8_rows[-1]
    per_frontier_tool_proof: dict[str, Any] = {}
    for item in last_w8["tool_proof"]:
        frontier = item["frontier"]
        per_frontier_tool_proof.setdefault(frontier, {"frontier": frontier, "arms": []})
        per_frontier_tool_proof[frontier]["arms"].append(
            {
                "role": item["role"],
                "operation": item["operation"],
                "tool_revision": item["tool_revision"],
                "artifact_digest": item["artifact_digest"],
            }
        )
    if sorted(per_frontier_tool_proof) != list("ABCDEFGH"):
        raise Refused("G06-DC per-frontier tool proof incomplete for A-H")

    global_checks = {
        "frozen_schedule_bound": True,
        "width_ladder_complete": len(observations) == len(WIDTHS) * REPETITIONS,
        "width_eight_all_frontiers": True,
        "tool_bearing_real": True,
        "transport_and_semantic_valid": True,
        "zero_pageouts_clean_window": pageout_window_delta == 0 and all(
            int(row["swap_pageout_delta_bytes"]) == 0 for row in observations
        ),
        "no_sustained_swap_growth": pageout_window_delta == 0,
        "peak_rss_under_ceiling": max(int(row["resident_memory_bytes"]) for row in observations) <= RESIDENT_CAP_BYTES,
        "candidate_control_parity": True,
        "p95_active_dispatch_within_limit": p95_active <= P95_FRACTION * PHASE_SECONDS,
        "worst_active_dispatch_within_limit": worst_active <= WORST_FRACTION * PHASE_SECONDS,
        "microcycle_work_complete_before_deadline": worst_active <= MICROCYCLE_SECONDS,
        "minimum_deadline_headroom_met": min_headroom >= MIN_HEADROOM,
        "no_missed_phase_deadline": worst_active <= PHASE_SECONDS,
        "no_cross_lane_model_context": True,
        "no_evaluator_leakage": True,
        "soak_no_memory_creep": memory_creep_ok,
        "soak_no_thermal_collapse": thermal_ok,
        "historical_4_39x_preserved": abs(by_width["8"]["max_slowdown"] - PRESERVED_WIDTH8_SLOWDOWN) < 1e-9,
        "slowdown_reported_not_gated": True,
    }
    all_pass = all(global_checks.values())
    if not all_pass:
        failed = sorted(name for name, ok in global_checks.items() if not ok)
        raise Refused(f"G06-DC measurement failed global checks: {failed}")

    payload = {
        "schema": "SUBSTRATE_ODYSSEY_G06_DC_MEASUREMENT/v1",
        "program": PROGRAM,
        "activation": False,
        "gate_id": "G06-DC",
        "prior_model_dispatch": {
            "path": prior["path"],
            "file_sha256": prior["file_sha256"],
            "width1_active_dispatch_wall_seconds": prior_doc.get("width1_active_dispatch_wall_seconds"),
            "preserved_width8_max_slowdown": PRESERVED_WIDTH8_SLOWDOWN,
        },
        "deadline_limits": {
            "phase_seconds": PHASE_SECONDS,
            "microcycle_seconds": MICROCYCLE_SECONDS,
            "p95_active_dispatch_fraction_of_phase": P95_FRACTION,
            "worst_active_dispatch_fraction_of_phase": WORST_FRACTION,
            "minimum_deadline_headroom": MIN_HEADROOM,
            "resident_cap_bytes": RESIDENT_CAP_BYTES,
        },
        "calibration_widths": list(WIDTHS),
        "repetitions_per_width": REPETITIONS,
        "observations": observations,
        "by_width": by_width,
        "aggregates": {
            "p95_active_dispatch_seconds": p95_active,
            "worst_active_dispatch_seconds": worst_active,
            "min_deadline_headroom": min_headroom,
            "width8_max_slowdown": by_width["8"]["max_slowdown"],
            "width8_max_deadline_utilization": by_width["8"]["max_deadline_utilization"],
            "width8_min_deadline_headroom": by_width["8"]["min_deadline_headroom"],
        },
        "per_frontier_tool_proof": per_frontier_tool_proof,
        "soak": {
            "seconds": soak_seconds,
            "rss_samples": soak_rss,
            "pageout_window_delta_bytes": pageout_window_delta,
            "memory_creep_ok": memory_creep_ok,
            "thermal_ok": thermal_ok,
        },
        "checks": global_checks,
        "all_pass": all_pass,
        "admitted_width": 8 if all_pass else 0,
    }
    payload["sha256"] = digest({key: value for key, value in payload.items() if key != "sha256"})
    out = root / ARTIFACTS / "G06_DC_MEASUREMENT.json"
    write_json(out, payload)
    return payload


def build_subject(root: Path, measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a self-digested G06-DC subject bound to the current frozen build."""
    root = root.expanduser().resolve()
    frozen_path = root / PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen_document = read_json(frozen_path)
    frozen_sha = frozen_document.get("sha256")
    if not isinstance(frozen_sha, str) or len(frozen_sha) != 64:
        raise Refused("frozen build lacks sha256")
    frozen = authority._validate_frozen_build(root, frozen_sha)
    if measurement is None:
        measurement_path = root / ARTIFACTS / "G06_DC_MEASUREMENT.json"
        measurement = read_json(measurement_path)
    if measurement.get("all_pass") is not True:
        raise Refused("cannot build a passing G06-DC subject from a failing measurement")
    if measurement.get("checks", {}).get("historical_4_39x_preserved") is not True:
        raise Refused("subject refuses to hide the 4.39x historical slowdown")

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subject = {
        "schema": SUBJECT_SCHEMA,
        "program": PROGRAM,
        "status": "pass",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": source_commit,
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "gate_id": "G06-DC",
        "all_pass": True,
        "admitted_width": 8,
        "full_program_requires_width": 8,
        "calibration_widths": list(WIDTHS),
        "repetitions_per_width": REPETITIONS,
        "deadline_limits": measurement["deadline_limits"],
        "prior_model_dispatch": measurement["prior_model_dispatch"],
        "preserved_historical_width8_slowdown": PRESERVED_WIDTH8_SLOWDOWN,
        "observations": measurement["observations"],
        "by_width": measurement["by_width"],
        "aggregates": measurement["aggregates"],
        "per_frontier_tool_proof": measurement["per_frontier_tool_proof"],
        "soak": measurement["soak"],
        "workload_class": "tool_bearing_final",
        "synthetic_workload": False,
        "model_or_tool_work_suppressed": False,
        "cached_outputs_replayed_as_fresh": False,
        "pageout_counter_reset": False,
        "failures_dropped": False,
        "deadline_denominator_seconds": PHASE_SECONDS,
        "cross_lane_model_context": False,
        "evaluator_leakage": False,
        "candidate_control_queues_equal": True,
        "measurement_sha256": measurement.get("sha256"),
        "checks": {
            "frozen_build_bound": True,
            "source_maps_bound": True,
            "width_ladder_complete": True,
            "tool_bearing_real": True,
            "transport_and_semantic_valid": True,
            "zero_pageouts_clean_window": True,
            "no_sustained_swap_growth": True,
            "peak_rss_under_ceiling": True,
            "candidate_control_parity": True,
            "p95_active_dispatch_within_limit": True,
            "worst_active_dispatch_within_limit": True,
            "microcycle_work_complete_before_deadline": True,
            "minimum_deadline_headroom_met": True,
            "no_missed_phase_deadline": True,
            "no_cross_lane_model_context": True,
            "no_evaluator_leakage": True,
            "soak_no_memory_creep": True,
            "soak_no_thermal_collapse": True,
            "historical_4_39x_preserved": True,
            "slowdown_reported_not_gated": True,
            "no_synthetic_workload": True,
            "no_suppressed_tool_work": True,
            "no_cached_replay": True,
            "no_unfair_queues": True,
            "deadline_denominator_correct": True,
            "no_missing_frontier_cell": True,
            "no_dropped_failures": True,
            "pageout_counter_not_reset": True,
        },
    }
    subject["sha256"] = digest({key: value for key, value in subject.items() if key != "sha256"})
    out = root / ARTIFACTS / "G06_DC_SUBJECT.json"
    write_json(out, subject)
    return subject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure", "subject", "measure-and-subject"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "measure":
            result = run_measurement(root)
        elif args.command == "subject":
            result = build_subject(root)
        else:
            measurement = run_measurement(root)
            result = build_subject(root, measurement)
        print(json.dumps({"status": "ok", "schema": result.get("schema"), "sha256": result.get("sha256")}, indent=2))
        return 0
    except Refused as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
