#!/usr/bin/env python
"""Long-run Studio daemon CLI.

Dry-run is the default. Pass --execute only on the Studio after the profile gate is intentional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.studio.long_run import run_daemon, write_plan_template


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="profile-gated long-run Studio daemon")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("template", help="write a starter daemon plan")
    pt.add_argument("--out", required=True)

    pr = sub.add_parser("run", help="run or dry-run a daemon plan")
    pr.add_argument("--plan", required=True)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--profile", default="studio-m1ultra")
    pr.add_argument("--execute", action="store_true")
    pr.add_argument("--heartbeat-min", type=float, default=5.0)
    pr.add_argument("--poll-s", type=float, default=5.0)
    pr.add_argument("--disk-root", default=None)

    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "template":
        plan = write_plan_template(a.out)
        print(json.dumps({"out": a.out, "jobs": len(plan["jobs"])}, indent=2))
        return 0

    state = run_daemon(
        Path(a.plan),
        out_dir=Path(a.out_dir),
        profile_name=a.profile,
        execute=bool(a.execute),
        heartbeat_s=float(a.heartbeat_min) * 60.0,
        poll_s=float(a.poll_s),
        disk_root=Path(a.disk_root) if a.disk_root else None,
    )
    print(json.dumps({"summary": state.get("summary", {}), "out_dir": a.out_dir}, indent=2))
    return 0 if not any(k in state.get("summary", {}) for k in ("failed", "blocked")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
