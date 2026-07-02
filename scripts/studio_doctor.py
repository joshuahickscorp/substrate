#!/usr/bin/env python
"""Studio readiness doctor entrypoint. Prints the full report as JSON to stdout, writes a
readable Markdown table to runs/studio_doctor.md, and logs a one-line PASS/FAIL verdict to
stderr. Exit code mirrors all_ok (0 ready, 1 not), so it can gate a launch script.

Usage: .venv/bin/python scripts/studio_doctor.py
"""

from __future__ import annotations

import json

from mop.config import REPO_ROOT
from mop.logging_utils import get_logger
from mop.studio_doctor import doctor, render_md

log = get_logger("studio_doctor")


def main() -> int:
    report = doctor()
    out = REPO_ROOT / "runs" / "studio_doctor.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(report))

    s = report["summary"]
    for c in report["checks"]:
        log.info("%-18s %s  %s", c["name"], "ok" if c["ok"] else "FAIL", c["detail"])
    log.info(
        "%s: %d/%d checks passed -> %s",
        "STUDIO READY" if report["all_ok"] else "NOT READY",
        s["passed"],
        s["total"],
        out,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
