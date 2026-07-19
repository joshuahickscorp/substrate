"""Launch, observe, adopt, or cooperatively stop the single durable MOP research orchestrator.

  PYTHONPATH=src .venv/bin/python scripts/mop_research_orchestrator.py --detach   # launch detached
  PYTHONPATH=src .venv/bin/python scripts/mop_research_orchestrator.py --observe   # read-only live view
  PYTHONPATH=src .venv/bin/python scripts/mop_research_orchestrator.py --stop      # cooperative drain

A second launch adopts the first rather than starting a competitor. Idempotent: re-running resumes from
durable state. It coexists with and never signals any live campaign. No em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mop.campaign.orchestrator import ResearchOrchestrator

RUN_ROOT = Path("runs/campaign/mop_research")
LOG = RUN_ROOT / "orchestrator.out.log"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The durable MOP research orchestrator.")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--detach", action="store_true", help="spawn a detached singleton and exit")
    parser.add_argument("--max-lifetime", type=float, default=3600.0)
    parser.add_argument("--tick-interval", type=float, default=30.0)
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args(argv)

    orch = ResearchOrchestrator(run_root=RUN_ROOT)

    if args.observe:
        view = orch.observe()
        owner = view["live_owner"]
        print("live orchestrator:", "yes pid=" + str(owner["pid"]) if owner else "none")
        if view["status"]:
            from mop.campaign.status import render_text

            print(render_text(view["status"]))
            oc = view["status"].get("orchestrator", {})
            print(
                "launch_commit:",
                oc.get("launch_commit"),
                "horizon_v2_complete:",
                oc.get("horizon_v2_complete"),
            )
        return 0

    if args.stop:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "STOP").write_text("stop", encoding="utf-8")
        print("cooperative stop file written; the orchestrator will drain at its next tick")
        return 0

    if args.detach:
        # if a live owner already holds the lock, adopt rather than spawn
        owner = orch.lock.live_owner()
        if owner is not None:
            print(f"adopted existing orchestrator pid={owner['pid']} (singleton); not spawning a competitor")
            return 0
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        # clear any stale stop file from a prior drained run
        stop = RUN_ROOT / "STOP"
        if stop.exists():
            stop.unlink()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--max-lifetime",
            str(args.max_lifetime),
            "--tick-interval",
            str(args.tick_interval),
        ]
        if args.no_telegram:
            cmd.append("--no-telegram")
        env = {**_env_with_pythonpath()}
        with open(LOG, "ab") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=logf,
                start_new_session=True,
                env=env,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
        print(f"detached orchestrator spawned pid={proc.pid}; log -> {LOG}")
        return 0

    # foreground durable run (this is what the detached child executes)
    summary = orch.run(
        max_lifetime=args.max_lifetime, tick_interval=args.tick_interval, telegram=not args.no_telegram
    )
    print(json.dumps(summary, indent=1, default=str))
    return 0


def _env_with_pythonpath() -> dict[str, str]:
    import os

    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}:{existing}"
    return env


if __name__ == "__main__":
    sys.exit(main())
