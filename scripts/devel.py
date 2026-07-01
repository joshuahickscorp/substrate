#!/usr/bin/env python
"""Developmental capacities CLI (Frontiers 20-35). One surface over the paradigm registry, the
capacity ladder, the curriculum engine, the ablation/hypothesis engine, paper-watch, and the
metacognition report. All bounded and offline; the curriculum command does REAL probe work over
generated control corpora on the M3 Pro. Nothing here claims sentience.

Usage:
  python scripts/devel.py paradigms                       # list the paradigm frontier registry
  python scripts/devel.py capacities                      # list the developmental capacity ladder
  python scripts/devel.py paperwatch                      # offline literature watch
  python scripts/devel.py ablation --scope local [--budget-hours 8]   # ranked next-experiment plan
  python scripts/devel.py curriculum [--families a,b,c]    # next-lesson manifest (real probes)
  python scripts/devel.py metacognition                   # self-monitoring report (gated by safety rails)
  python scripts/devel.py validate                        # validate all developmental registries
JSON to stdout, human log lines to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from devsys.devel import ablation, curriculum, metacognition, paperwatch, registries


def _cmd_curriculum(a: argparse.Namespace) -> dict:
    from devsys.studio import controls
    from devsys.studio.profiles import get_profile

    fams = a.families.split(",") if a.families else ["moving_object", "hard_motion", "aleatoric_tv"]
    out_root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="devel_curriculum_"))
    gen = controls.generate_controls(
        out_root, families=fams, clips_per_class=a.clips_per_class, frames=8, h=32, w=32, seed=a.seed
    )
    sources = [{"slug": f["name"], "root": f["root"], "capacity": "abstraction"} for f in gen["families"]]
    return curriculum.next_lesson(
        sources, profile=get_profile(a.profile), eval_clips=a.eval_clips, next_clips=a.next_clips, seed=a.seed
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="developmental capacities CLI (registries, curriculum, ablation, paperwatch)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("paradigms", help="Frontier 20: list the paradigm frontier registry")
    sub.add_parser("capacities", help="Frontier 32: list the developmental capacity ladder")
    sub.add_parser("paperwatch", help="Frontier 30: offline literature watch")
    sub.add_parser("metacognition", help="Frontier 34: self-monitoring report (safety-rail gated)")
    sub.add_parser("validate", help="validate all developmental registries")
    pe = sub.add_parser(
        "experiments", help="list/validate the experiment bank; --render writes EXPERIMENTS.md"
    )
    pe.add_argument("--render", action="store_true", help="regenerate EXPERIMENTS.md from the registry")

    pa = sub.add_parser("ablation", help="Frontier 29: ranked next-experiment plan")
    pa.add_argument("--scope", default="local", choices=["local", "studio"])
    pa.add_argument("--budget-hours", type=float, default=None)

    pc = sub.add_parser("curriculum", help="Frontier 26/33: next-lesson manifest (REAL probes over controls)")
    pc.add_argument("--families", default=None, help="comma list of control families")
    pc.add_argument("--profile", default="m3pro-local-max")
    pc.add_argument("--clips-per-class", type=int, default=24)
    pc.add_argument("--eval-clips", type=int, default=48)
    pc.add_argument("--next-clips", type=int, default=64)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--out", default=None, help="where to generate control corpora (default tmp)")

    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if a.cmd == "paradigms":
        items = registries.load_paradigms()
        result = {
            "n": len(items),
            "paradigms": [
                {"slug": p["slug"], "family": p["family"], "status": p["status"], "tags": p["tags"]}
                for p in items
            ],
        }
    elif a.cmd == "capacities":
        items = registries.load_capacities()
        result = {
            "n": len(items),
            "ladder": [
                {"rung": c["rung"], "name": c["name"], "tag": c["tag"], "status": c["status"]} for c in items
            ],
        }
    elif a.cmd == "paperwatch":
        result = paperwatch.watch_report()
    elif a.cmd == "ablation":
        result = ablation.plan_ablations(scope=a.scope, budget_hours=a.budget_hours)
    elif a.cmd == "curriculum":
        result = _cmd_curriculum(a)
    elif a.cmd == "metacognition":
        result = metacognition.report()
    elif a.cmd == "validate":
        problems = registries.validate_all()
        result = {"problems": problems, "ok": not problems}
    elif a.cmd == "experiments":
        from devsys.config import REPO_ROOT

        items = registries.load_experiments()
        problems = registries.validate_experiments()
        if a.render:
            (REPO_ROOT / "EXPERIMENTS.md").write_text(registries.render_experiments_md())
        result = {
            "n": len(items),
            "by_status": {s: sum(1 for e in items if e["status"] == s) for s in registries.EXP_STATUS},
            "by_series": {
                s: sum(1 for e in items if e["series"] == s) for s in sorted({e["series"] for e in items})
            },
            "problems": problems,
            "ok": not problems,
            "rendered": bool(a.render),
        }
    else:  # unreachable (required subparser)
        ap.error(f"unknown command {a.cmd}")

    print(json.dumps(result, indent=2, default=str))
    if a.cmd in ("validate", "experiments") and not result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
