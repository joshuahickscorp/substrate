#!/usr/bin/env python
"""List or plan Studio-native lanes without running science."""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.studio.native_lanes import (
    build_native_lane_manifest,
    write_native_daemon_plan,
    write_native_manifest,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Studio-native lane manifest and daemon-plan builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="evaluate lanes and print a manifest")
    _add_common(lp)
    lp.add_argument("--out", default=None, help="optional JSON manifest path")

    pp = sub.add_parser("plan", help="write a long-run daemon plan from ready lanes")
    _add_common(pp)
    pp.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_native_lanes_plan.json"))
    pp.add_argument(
        "--manifest-out",
        default=str(REPO_ROOT / "runs" / "studio_native_lanes_manifest.json"),
        help="sidecar manifest path with blocked-lane reasons",
    )

    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    manifest = build_native_lane_manifest(
        profile_name=a.profile,
        include_heavy=a.include_heavy,
        lane_ids=a.lane,
        inputs=_inputs(a),
    )

    if a.cmd == "list":
        if a.out:
            write_native_manifest(manifest, a.out)
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    write_native_manifest(manifest, a.manifest_out)
    try:
        plan = write_native_daemon_plan(manifest, a.out)
    except ValueError as e:
        print(json.dumps({"error": str(e), "manifest": a.manifest_out}, indent=2), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"out": a.out, "manifest": a.manifest_out, "jobs": [j["id"] for j in plan["jobs"]]}, indent=2
        )
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="studio-m1ultra")
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="allow heavy runnable lanes into the manifest or daemon plan",
    )
    parser.add_argument("--lane", action="append", default=None, help="restrict to one lane id, repeatable")
    parser.add_argument("--clip-dir", default=None, help="real .pt clip directory for DR13 lanes")
    parser.add_argument("--dr1-cache", default=None, help="merged DR1 latent cache for PR9")
    parser.add_argument(
        "--plan-path", default=None, help="inspected hosted-corpora plan path for acquisition"
    )
    parser.add_argument(
        "--encode-schedule",
        default=str(REPO_ROOT / "runs" / "mot" / "encode_schedule.json"),
        help="Wave 0 encode schedule receipt for live-encoder doctrine review",
    )


def _inputs(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "clip_dir": args.clip_dir,
        "dr1_cache": args.dr1_cache,
        "plan_path": args.plan_path,
        "encode_schedule": args.encode_schedule,
    }


if __name__ == "__main__":
    raise SystemExit(main())
