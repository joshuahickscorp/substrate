"""Bridge Wave-0 encode schedules into DR1 cache jobs.

The encode scheduler decides whether CPU or MPS should own the Studio cache build. This module turns
that JSON receipt into explicit DR1 gate, encode-leg, merge, and guard commands without decoding video
or loading an encoder.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .long_run import SCHEMA as DAEMON_SCHEMA

SCHEMA = "mop-dr1-schedule-plan/v1"
DEFAULT_DR1_SCRIPT = "scripts/studio/dr1_curate_bound_video.py"
DEFAULT_FACTORS = ("object", "count", "relation", "action")


def load_encode_schedule(path: Path | str) -> dict[str, Any]:
    """Read an `encode_schedule.json` receipt."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("encode schedule must be a JSON object")
    return data


def build_dr1_schedule_plan(
    schedule: dict[str, Any],
    *,
    source: str,
    cache_name: str = "vjepa2_vitl_comp_video",
    factors: Sequence[str] = DEFAULT_FACTORS,
    min_per_cell: int = 16,
    python: str = ".venv/bin/python",
    script: str = DEFAULT_DR1_SCRIPT,
    include_a6_guard: bool = True,
    include_verifier: bool = True,
    source_intake: dict[str, Any] | None = None,
    require_source_intake: bool = False,
) -> dict[str, Any]:
    """Convert an encode schedule into a DR1 command plan.

    The plan is dry metadata until a Studio daemon executes the emitted commands. If the schedule is
    blocked, the returned plan records the wall and emits no runnable jobs.
    """
    factors_tuple = tuple(str(f) for f in factors if str(f))
    gates = _schedule_gates(
        schedule,
        source=source,
        factors=factors_tuple,
        min_per_cell=min_per_cell,
        source_intake=source_intake,
        require_source_intake=require_source_intake,
    )
    ok_to_launch = all(g["ok"] for g in gates)
    winner_obj = schedule.get("winner")
    winner: dict[str, Any] = winner_obj if isinstance(winner_obj, dict) else {}
    device = str(winner.get("device", "")) if winner else ""
    effective_clips = _positive_int(schedule.get("effective_clips"))
    checkpoint_obj = schedule.get("checkpoint")
    checkpoint: dict[str, Any] = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}
    every_clips = _positive_int(checkpoint.get("every_clips"))
    ranges: list[tuple[int, int]] = []
    jobs: list[dict[str, Any]] = []

    if ok_to_launch:
        leg_size = min(every_clips, effective_clips)
        ranges = [
            (start, min(start + leg_size, effective_clips)) for start in range(0, effective_clips, leg_size)
        ]
        wall_s_per_clip = _positive_float(winner.get("wall_s_per_clip"))
        jobs.append(
            {
                "id": "dr1_caption_gate",
                "kind": "verdict-gate",
                "cmd": _base_command(python, script, source, cache_name, factors_tuple, min_per_cell)
                + ["--gate-only", "--start", "0", "--end", str(effective_clips)],
                "range": [0, effective_clips],
            }
        )
        width = max(6, len(str(effective_clips)))
        for start, end in ranges:
            jobs.append(
                {
                    "id": f"dr1_encode_{start:0{width}d}_{end:0{width}d}",
                    "kind": "encode",
                    "cmd": _base_command(python, script, source, cache_name, factors_tuple, min_per_cell)
                    + ["--start", str(start), "--end", str(end), "--device", device],
                    "range": [start, end],
                    "estimated_wall_min": None
                    if wall_s_per_clip is None
                    else round((end - start) * wall_s_per_clip / 60.0, 3),
                }
            )
        jobs.append(
            {
                "id": "dr1_merge",
                "kind": "merge",
                "cmd": _base_command(python, script, source, cache_name, factors_tuple, min_per_cell)
                + ["--merge"],
                "range": [0, effective_clips],
            }
        )
        if include_a6_guard:
            jobs.append(
                {
                    "id": "dr1_a6_guard",
                    "kind": "verdict-gate",
                    "cmd": _base_command(python, script, source, cache_name, factors_tuple, min_per_cell)
                    + ["--a6-guard"],
                    "range": [0, effective_clips],
                }
            )
        if include_verifier:
            jobs.append(
                {
                    "id": "dr1_verify",
                    "kind": "verifier",
                    "cmd": [
                        python,
                        "scripts/studio/dr1_verify.py",
                        "--cache",
                        str(Path("data") / "cache" / cache_name),
                        "--out",
                        str(Path("data") / "cache" / cache_name / "dr1_verification.json"),
                    ],
                    "range": [0, effective_clips],
                }
            )

    blocked_reasons = [g["detail"] for g in gates if not g["ok"]]
    return {
        "schema": SCHEMA,
        "ok_to_launch": ok_to_launch,
        "blocked_reasons": blocked_reasons,
        "source": source,
        "cache_name": cache_name,
        "factors": list(factors_tuple),
        "min_per_cell": int(min_per_cell),
        "schedule": {
            "profile": schedule.get("profile", {}).get("name")
            if isinstance(schedule.get("profile"), dict)
            else None,
            "effective_clips": effective_clips,
            "winner": winner or None,
            "checkpoint": checkpoint,
            "thermal_pacing": schedule.get("thermal_pacing"),
            "blocked_reasons": schedule.get("blocked_reasons", []),
        },
        "source_intake": _source_intake_summary(source_intake),
        "gates": gates,
        "ranges": [list(r) for r in ranges],
        "jobs": jobs,
        "summary": {
            "jobs": len(jobs),
            "encode_legs": len(ranges),
            "clips": effective_clips,
            "checkpoint_every_clips": every_clips,
            "device": device or None,
            "verifier": bool(include_verifier),
        },
    }


def daemon_plan_from_dr1_schedule_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a long-run daemon plan for a launchable DR1 schedule plan."""
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"DR1 schedule plan schema {plan.get('schema')!r} != {SCHEMA!r}")
    if not plan.get("ok_to_launch"):
        reasons = ", ".join(str(r) for r in plan.get("blocked_reasons", [])) or "schedule blocked"
        raise ValueError(f"cannot make daemon plan from blocked DR1 schedule: {reasons}")
    jobs = [
        {"id": str(job["id"]), "kind": str(job["kind"]), "cmd": list(job["cmd"])}
        for job in plan.get("jobs", [])
    ]
    if not jobs:
        raise ValueError("DR1 schedule plan has no jobs")
    return {"schema": DAEMON_SCHEMA, "jobs": jobs}


def write_json(data: dict[str, Any], path: Path | str) -> None:
    """Write a JSON receipt with parent directory creation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _schedule_gates(
    schedule: dict[str, Any],
    *,
    source: str,
    factors: Sequence[str],
    min_per_cell: int,
    source_intake: dict[str, Any] | None,
    require_source_intake: bool,
) -> list[dict[str, Any]]:
    winner_obj = schedule.get("winner")
    winner: dict[str, Any] | None = winner_obj if isinstance(winner_obj, dict) else None
    checkpoint_obj = schedule.get("checkpoint")
    checkpoint: dict[str, Any] = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}
    effective_clips = _positive_int(schedule.get("effective_clips"))
    every_clips = _positive_int(checkpoint.get("every_clips"))
    device = str(winner.get("device", "")) if winner else ""
    gates = [
        _gate(
            "schedule_ok",
            bool(schedule.get("ok_to_launch")),
            "schedule ok_to_launch is true"
            if schedule.get("ok_to_launch")
            else f"schedule blocked: {schedule.get('blocked_reasons', [])}",
        ),
        _gate(
            "source_present", bool(source), "source path supplied" if source else "source path is required"
        ),
        _gate(
            "factors_present",
            bool(factors),
            f"{len(factors)} factor(s)" if factors else "no factors supplied",
        ),
        _gate(
            "min_per_cell",
            int(min_per_cell) > 0,
            f"min_per_cell={min_per_cell}" if int(min_per_cell) > 0 else "min_per_cell must be positive",
        ),
        _gate(
            "winner_device",
            winner is not None and device not in {"", "blocked"},
            f"device={device}"
            if winner is not None and device not in {"", "blocked"}
            else "no usable winner device",
        ),
        _gate(
            "effective_clips",
            effective_clips > 0,
            f"{effective_clips} clips" if effective_clips > 0 else "effective_clips must be positive",
        ),
        _gate(
            "checkpoint_every_clips",
            every_clips > 0,
            f"{every_clips} clips" if every_clips > 0 else "checkpoint.every_clips must be positive",
        ),
    ]
    gates.append(_source_intake_gate(source_intake, source, factors, min_per_cell, require_source_intake))
    return gates


def _source_intake_summary(source_intake: dict[str, Any] | None) -> dict[str, Any] | None:
    if source_intake is None:
        return None
    return {
        "schema": source_intake.get("schema"),
        "all_ok": bool(source_intake.get("all_ok")),
        "source": source_intake.get("source"),
        "problems": source_intake.get("problems", []),
    }


def _source_intake_gate(
    source_intake: dict[str, Any] | None,
    source: str,
    factors: Sequence[str],
    min_per_cell: int,
    required: bool,
) -> dict[str, Any]:
    if source_intake is None:
        return _gate(
            "source_intake",
            not required,
            "source intake not required for this metadata-only plan"
            if not required
            else "missing DR1 source intake receipt",
        )
    if source_intake.get("schema") != "mop-dr1-source-intake/v1":
        return _gate(
            "source_intake",
            False,
            f"unexpected source intake schema {source_intake.get('schema')!r}",
        )
    problems: list[str] = []
    if not source_intake.get("all_ok"):
        problems.extend(str(p) for p in source_intake.get("problems", []))
    if str(source_intake.get("source")) != str(source):
        problems.append(f"source intake covers {source_intake.get('source')!r}, not {source!r}")
    if tuple(str(f) for f in source_intake.get("factors", [])) != tuple(str(f) for f in factors):
        problems.append("source intake factors do not match schedule factors")
    if int(source_intake.get("min_per_cell", -1)) != int(min_per_cell):
        problems.append("source intake min_per_cell does not match schedule")
    detail = "DR1 source intake all_ok true" if not problems else "; ".join(problems)
    return _gate("source_intake", not problems, detail)


def _base_command(
    python: str,
    script: str,
    source: str,
    cache_name: str,
    factors: Sequence[str],
    min_per_cell: int,
) -> list[str]:
    return [
        python,
        script,
        "--source",
        source,
        "--name",
        cache_name,
        "--factors",
        ",".join(factors),
        "--min-per-cell",
        str(int(min_per_cell)),
    ]


def _gate(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _positive_int(value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return 0
    return out if out > 0 else 0


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None
