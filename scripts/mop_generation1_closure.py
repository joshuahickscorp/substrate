"""Run the generation1-general-run-closure observer and regenerate the current-state frontier.

Idempotent and read-only over live artifacts. While the General Run is not terminal, closure is deferred:
it seals a refusal closure plus the current-state authority and exits without launching any compute. When
the General Run reaches a clean terminal, the same command replays the terminal lineage and seals the full
closure. It never signals, restarts, or modifies any live process or sealed artifact.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/mop_generation1_closure.py --execute
  PYTHONPATH=src .venv/bin/python scripts/mop_generation1_closure.py --execute --no-telegram

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running this file as a script puts scripts/ on sys.path, not the repo root; the closure lineage
# transitively imports a module that references the top-level ``scripts`` package, so put the repo root
# first (the equivalent of running ``-m mop.closure.producer`` from the repo root).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mop.closure.producer import run_closure  # noqa: E402 (import after the sys.path shim above)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="generation1-general-run-closure observer")
    parser.add_argument("--execute", action="store_true", help="seal the closure and current frontier")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--gr-root", default=str(REPO_ROOT / "runs/generation1/general-run"))
    parser.add_argument("--timestamp", default="2026-07-18T00:00:00Z")
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args(argv)

    if not args.execute:
        print("pass --execute to seal the closure (read-only, refuses while the General Run is not terminal)")
        return 0

    summary = run_closure(
        repo_root=REPO_ROOT,
        runs_root=Path(args.runs_root),
        gr_root=Path(args.gr_root),
        timestamp=args.timestamp,
        telegram=not args.no_telegram,
    )
    print(f"admitted: {summary['admitted']}  closure_status: {summary.get('closure_status')}")
    print(f"refusals: {summary.get('refusals')}")
    print(f"closure: {summary.get('closure_path')}")
    print(
        f"verification: seal_intact={summary.get('seal_intact')} "
        f"mutations_all_detected={summary.get('mutations_all_detected')}"
    )
    print(f"frontier: {summary.get('frontier_path')}")
    print(f"report: {summary.get('report_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
