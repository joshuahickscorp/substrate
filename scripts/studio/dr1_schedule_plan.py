#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.dr1_schedule import (  # noqa: E402
    DEFAULT_FACTORS,
    build_dr1_schedule_plan,
    daemon_plan_from_dr1_schedule_plan,
    load_encode_schedule,
    write_json,
)


def _load_json_object(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="turn encode_schedule.json into DR1 gate/leg commands")
    parser.add_argument("--schedule", default=str(_ROOT / "runs" / "mot" / "encode_schedule.json"))
    parser.add_argument("--source", required=True, help="dir of <cell>/<clip> plus captions.json")
    parser.add_argument("--name", default="vjepa2_vitl_comp_video", help="DR1 cache name under data/cache")
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS), help="comma-list composable factors")
    parser.add_argument("--min-per-cell", type=int, default=16)
    parser.add_argument("--python", default=".venv/bin/python", help="python executable for emitted commands")
    parser.add_argument("--script", default="scripts/studio/dr1_curate_bound_video.py")
    parser.add_argument("--out", default=str(_ROOT / "runs" / "studio_wave0" / "dr1_schedule_plan.json"))
    parser.add_argument("--daemon-out", default=None, help="optional long-run daemon plan output path")
    parser.add_argument("--source-intake", default=None, help="DR1 source intake receipt to require")
    parser.add_argument("--no-a6-guard", action="store_true", help="omit the post-merge A6 guard job")
    parser.add_argument("--no-verifier", action="store_true", help="omit the post-A6 DR1 verifier job")
    args = parser.parse_args(argv)

    factors = tuple(f for f in args.factors.split(",") if f)
    schedule = load_encode_schedule(args.schedule)
    source_intake = _load_json_object(args.source_intake)
    plan = build_dr1_schedule_plan(
        schedule,
        source=args.source,
        cache_name=args.name,
        factors=factors,
        min_per_cell=args.min_per_cell,
        python=args.python,
        script=args.script,
        include_a6_guard=not args.no_a6_guard,
        include_verifier=not args.no_verifier,
        source_intake=source_intake,
        require_source_intake=args.source_intake is not None,
    )
    plan["schedule_path"] = args.schedule
    plan["source_intake_path"] = args.source_intake
    write_json(plan, args.out)

    daemon_out = None
    if args.daemon_out:
        try:
            daemon = daemon_plan_from_dr1_schedule_plan(plan)
        except ValueError as e:
            print(json.dumps({"out": args.out, "ok_to_launch": False, "error": str(e)}, indent=2))
            return 1
        write_json(daemon, args.daemon_out)
        daemon_out = args.daemon_out

    print(
        json.dumps(
            {
                "out": args.out,
                "daemon_out": daemon_out,
                "ok_to_launch": plan["ok_to_launch"],
                "summary": plan["summary"],
                "blocked_reasons": plan["blocked_reasons"],
            },
            indent=2,
        )
    )
    return 0 if plan["ok_to_launch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
