#!/usr/bin/env python
"""Run the Mac-Studio rehearsal capsule: the whole future Studio workflow end to end on tiny
local fixtures (no downloads, no long runs). Writes runs/studio_rehearsal/{report.md,summary.json}.

Usage: python scripts/studio_rehearsal.py        # or: make rehearse
"""

from __future__ import annotations

import json

from mop.studio_rehearsal import DEFAULT_OUT, rehearse


def main(argv: list[str] | None = None) -> int:
    s = rehearse()
    print(
        json.dumps(
            {
                "overall": s["overall"],
                "stages": {st["stage"]: st["status"] for st in s["stages"]},
                "report": str(DEFAULT_OUT / "report.md"),
            },
            indent=2,
        )
    )
    return 0 if s["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
