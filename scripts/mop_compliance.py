"""Generate and verify the hard mandate-compliance ledger over bundle files 00, 01, 02, and 04.

Writes proof/campaign_run/COMPLIANCE_LEDGER.json and exits nonzero if any safely-executable hard
requirement is still planned or partial (an external-blocked family is contracted, not a failure).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/mop_compliance.py            # verify (nonzero if non-compliant)
  PYTHONPATH=src .venv/bin/python scripts/mop_compliance.py --print    # print the full ledger

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.campaign.compliance import build_ledger

OUT = Path("proof/campaign_run/COMPLIANCE_LEDGER.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify MOP mandate compliance.")
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args(argv)

    ledger = build_ledger()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")

    print(f"requirements={ledger['n_requirements']} "
          f"implemented/running={ledger['n_implemented_or_running']} "
          f"failures={ledger['n_failures']} compliant={ledger['compliant']}")
    for r in ledger["requirements"]:
        mark = "OK " if r["status"] in ("implemented", "running", "queued") else (
            "BLK" if r["status"] == "blocked_external" else "!! ")
        print(f"  {mark} {r['req_id']:16} {r['status']:12} {r['text'][:56]}")
        if args.show and r["is_failure"]:
            print(f"       next: {r.get('next_executable_action', '')}")
    print(f"\nledger sealed -> {OUT}")
    return 0 if ledger["compliant"] else 1


if __name__ == "__main__":
    sys.exit(main())
