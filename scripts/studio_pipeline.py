#!/usr/bin/env python
"""The one Studio pipeline CLI: plan -> acquire -> validate -> cache -> run -> optimize -> report,
plus `local-max` (the current-device maximal rehearsal) and `profiles` (list the kill-switch envelopes).

Heavy acquisition is DRY-RUN by default. A real download needs `--execute`, a `--budget-gb`, and (for
any source with terms) `--accept-license`. The current device should use `local-max`; the future
Studio uses `plan --profile studio-1tb` then `acquire --execute`.

Usage:
  python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900
  python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json   # DRY RUN
  python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json \
      --execute --budget-gb 900 --accept-license                                           # REAL (gated)
  python scripts/studio_pipeline.py validate --plan runs/studio_pipeline/latest/plan.json
  python scripts/studio_pipeline.py cache    --plan runs/studio_pipeline/latest/plan.json [--execute]
  python scripts/studio_pipeline.py run --gated --tiers C --full
  python scripts/studio_pipeline.py optimize --cache <cache_id>
  python scripts/studio_pipeline.py report
  python scripts/studio_pipeline.py local-max --download-gb 10 --time-min 180 --cache-clips 64
  python scripts/studio_pipeline.py profiles
JSON to stdout, human log lines to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

from mop.studio import pipeline, profiles


def _parse_tiers(s: str) -> set[str]:
    return {t.strip().upper() for t in s.split(",") if t.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="the one Studio pipeline (plan/acquire/validate/cache/run/optimize/report + local-max)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="Frontier 3: knapsack acquisition plan under a profile")
    pp.add_argument(
        "--profile", default="studio-1tb", help="profile name/alias (studio-1tb, m3pro-local-max)"
    )
    pp.add_argument(
        "--budget-gb", type=float, default=None, help="usable disk budget (clamped to the profile hard cap)"
    )
    pp.add_argument("--accept-license", action="store_true", help="acknowledge signed-terms (manual) sources")
    pp.add_argument("--include", nargs="*", default=None, help="restrict to these dataset slugs")
    pp.add_argument("--exclude", nargs="*", default=None, help="drop these dataset slugs")
    pp.add_argument("--label", default=None, help="run-dir label (default timestamp)")

    pa = sub.add_parser("acquire", help="Frontier 4: acquire selected sources (DRY RUN unless --execute)")
    pa.add_argument(
        "--plan", required=True, help="path to plan.json (use runs/studio_pipeline/latest/plan.json)"
    )
    pa.add_argument("--execute", action="store_true", help="ACTUALLY acquire (omit for a safe dry run)")
    pa.add_argument(
        "--budget-gb", type=float, default=None, help="download budget (clamped to the profile hard cap)"
    )
    pa.add_argument("--accept-license", action="store_true", help="acknowledge signed-terms (manual) sources")
    pa.add_argument("--profile", default=None, help="override the profile (default: from the plan)")

    pv = sub.add_parser("validate", help="Frontier 6: validate acquired sources")
    pv.add_argument("--plan", required=True)

    pc = sub.add_parser("cache", help="Frontier 8: estimate (default) or build (--execute) latent caches")
    pc.add_argument("--plan", required=True)
    pc.add_argument("--execute", action="store_true", help="build a TINY real cache from a landed corpus")
    pc.add_argument("--encoder", default="vjepa2_vitl_fpc64_256")
    pc.add_argument("--profile", default=None)

    pr = sub.add_parser("run", help="Frontier 10/15: the gated conveyor (campaign queue behind gates)")
    pr.add_argument(
        "--gated",
        action="store_true",
        default=True,
        help="affirm gating (gates ALWAYS enforce as kill switches)",
    )
    pr.add_argument("--tiers", type=str, default="C", help="comma list of tiers (C,E,R)")
    pr.add_argument(
        "--full", action="store_true", help="full grids (Studio); default is toy validation scale"
    )
    # SAFE DEFAULT: the laptop profile, so an unqualified `run --full` fails the run-count kill switch
    # on this device instead of launching a full sweep. The Studio passes --profile studio-1tb.
    pr.add_argument(
        "--profile", default="m3pro-local-max", help="default m3pro-local-max; Studio: studio-1tb"
    )

    po = sub.add_parser("optimize", help="Frontier 12: throughput optimization lane (not science)")
    po.add_argument("--cache", default=None, help="cache id/dir to annotate the report with")
    po.add_argument(
        "--profile", default="m3pro-local-max", help="default m3pro-local-max; Studio: studio-1tb"
    )
    po.add_argument("--reps", type=int, default=1)

    sub.add_parser("report", help="Frontier 14: roll the latest run into report.md/summary.json")

    pl = sub.add_parser("local-max", help="Frontier 3B: current-device maximal rehearsal (m3pro-local-max)")
    pl.add_argument("--download-gb", type=float, default=10.0)
    pl.add_argument("--time-min", type=int, default=90)
    pl.add_argument("--cache-clips", type=int, default=64)
    pl.add_argument("--seed", type=int, default=0)

    sub.add_parser("profiles", help="list the shipped profiles (kill-switch envelopes)")

    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if a.cmd == "plan":
        out = pipeline.cmd_plan(a.profile, a.budget_gb, a.accept_license, a.include, a.exclude, a.label)
        result = {
            "run_dir": out["run_dir"],
            "totals": out["plan"]["totals"],
            "blockers": out["plan"]["license_blockers"],
        }
    elif a.cmd == "acquire":
        m = pipeline.cmd_acquire(a.plan, a.execute, a.budget_gb, a.accept_license, a.profile)
        result = {"run_dir": m["run_dir"], "mode": m["manifest"]["mode"], "totals": m["manifest"]["totals"]}
    elif a.cmd == "validate":
        out = pipeline.cmd_validate(a.plan)
        result = {"results": [{"slug": r["slug"], "validation": r["validation"]} for r in out["results"]]}
    elif a.cmd == "cache":
        out = pipeline.cmd_cache(a.plan, a.execute, a.encoder, a.profile)
        result = {"encoder": out["encoder"], "results": out["results"]}
    elif a.cmd == "run":
        out = pipeline.cmd_run(a.gated, _parse_tiers(a.tiers), a.full, a.profile)
        result = {"ran": out["ran"], "gates": out["gates"], "reason": out.get("reason")}
    elif a.cmd == "optimize":
        out = pipeline.cmd_optimize(a.cache, a.profile, a.reps)
        result = {"benches": out["benches"]}
    elif a.cmd == "report":
        result = pipeline.cmd_report()
    elif a.cmd == "local-max":
        s = pipeline.cmd_local_max(a.download_gb, a.time_min, a.cache_clips, a.seed)
        result = {
            "overall": s["overall"],
            "elapsed_min": s["elapsed_min"],
            "effective": s["effective"],
            "stages": {st["stage"]: st["status"] for st in s["stages"]},
        }
    elif a.cmd == "profiles":
        result = {"profiles": profiles.list_profiles()}
    else:  # unreachable (required subparser)
        ap.error(f"unknown command {a.cmd}")

    print(json.dumps(result, indent=2, default=str))
    # non-zero exit when a gated run was stopped or a local-max did not fully pass
    if a.cmd == "run" and not result.get("ran"):
        return 1
    if a.cmd == "local-max" and result.get("overall") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
