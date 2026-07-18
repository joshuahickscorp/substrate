"""Launch and observe the MOP pre-substrate discovery campaign under the unified engine.

Builds the campaign manifest, adopts the live General Run and horizon chain as external resource consumers
(observed, never signaled), runs the safe local frontier concurrently under one global resource broker, and
durably queues the contracted external-input families and the precommitted decision branches. Writes a
status view and delivers campaign Telegram events. Idempotent: re-running resumes from durable state.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/run_mop_campaign.py --max-seconds 600
  PYTHONPATH=src .venv/bin/python scripts/run_mop_campaign.py --observe   # status only, no run

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.campaign.broker import ResourceBroker
from mop.campaign.dag import AuthorityResolver, refresh_eligibility
from mop.campaign.executor import CampaignScheduler
from mop.campaign.manifest import build_campaign
from mop.campaign.status import build_status, render_text
from mop.campaign.telegram import record_delivery, send_campaign_event

RUN_ROOT = Path("runs/campaign/pre_substrate_v1")
OP_ROOT = Path("proof/campaign_run")


def _live_boundary_checker() -> AuthorityResolver:
    """External authorities: the live successor horizon chain is an unavoidable boundary for any node that
    declares ``external:horizon-v2-complete``. It clears when the sealed terminal artifact appears."""

    def check(name: str) -> bool:
        if name == "external:horizon-v2-complete":
            return (
                Path("runs/generation1/generation1-successor-horizon-v2/program_state.json").exists()
                and _horizon_complete()
            )
        return False

    return AuthorityResolver(external_checker=check)


def _horizon_complete() -> bool:
    try:
        data = json.loads(
            Path("runs/generation1/generation1-successor-horizon-v2/program_state.json").read_text()
        )
        return data.get("state") in ("complete", "finished") or data.get("finished_at") not in (None, "")
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MOP pre-substrate discovery campaign.")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-ceiling", type=int, default=16)
    parser.add_argument("--observe", action="store_true", help="print status and exit without running")
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args(argv)

    camp = build_campaign()
    OP_ROOT.mkdir(parents=True, exist_ok=True)
    broker = ResourceBroker(external_labels=("mop-final-mechanic", "mop-g1-", "general-run"))
    resolver = _live_boundary_checker()

    scheduler = CampaignScheduler(camp, RUN_ROOT, broker, resolver, max_ceiling=args.max_ceiling)
    scheduler.state.load()

    if args.observe:
        refresh_eligibility(camp, scheduler.state, resolver)
        snap = broker.sample()
        status = build_status(camp, scheduler.state, snap)
        (OP_ROOT / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(render_text(status))
        return 0

    # launch event
    if not args.no_telegram:
        launch_snap = broker.sample()
        receipt = send_campaign_event(
            "campaign launched",
            f"{camp.campaign_id}: {len(camp.nodes)} nodes "
            f"({len([n for n in camp.nodes if not n.is_blocked])} runnable, "
            f"{len([n for n in camp.nodes if n.is_blocked])} contracted-external). "
            f"broker mode={launch_snap.payload()['mode']} budget~{launch_snap.cpu_budget} "
            f"coexisting with {launch_snap.external_consumers} live workers.",
            root=OP_ROOT,
            dedup_key=f"launch-{camp.digest()[:12]}",
        )
        record_delivery(OP_ROOT, receipt)
        print(
            "telegram launch:",
            receipt.get("delivered"),
            receipt.get("message_id") or receipt.get("error", ""),
        )

    summary = scheduler.run(max_seconds=args.max_seconds, poll_seconds=0.3)
    print("run summary:", json.dumps(summary["counts"]), f"elapsed={summary['elapsed_seconds']}s")

    snap = broker.snapshot()
    status = build_status(camp, scheduler.state, snap)
    (OP_ROOT / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")

    # terminal summary event
    if not args.no_telegram:
        receipt = send_campaign_event(
            "campaign frontier complete",
            render_text(status),
            root=OP_ROOT,
            dedup_key=f"summary-{camp.digest()[:12]}-{status['completed']}",
        )
        record_delivery(OP_ROOT, receipt)
        print(
            "telegram summary:",
            receipt.get("delivered"),
            receipt.get("message_id") or receipt.get("error", ""),
        )

    print("\n" + render_text(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
