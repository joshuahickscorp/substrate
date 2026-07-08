#!/usr/bin/env python
"""Run the read-only Studio transfer checklist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.studio.transfer_check import (
    DEFAULT_AUDIT_PATH,
    TransferCheckConfig,
    run_transfer_check,
    write_transfer_report,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Studio Wave-0 transfer checklist")
    ap.add_argument("--profile", default="studio-m1ultra")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--audit-path", default=None, help="override governing audit path")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-receipts", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    audit = None if a.skip_audit else Path(a.audit_path) if a.audit_path else DEFAULT_AUDIT_PATH
    cfg = TransferCheckConfig(
        repo_root=Path(a.repo_root).resolve(),
        audit_path=audit,
        profile_name=a.profile,
        allow_dirty=bool(a.allow_dirty),
        require_receipts=not bool(a.no_receipts),
    )
    report = run_transfer_check(cfg)
    if a.out:
        write_transfer_report(report, a.out)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
