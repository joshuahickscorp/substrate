#!/usr/bin/env python
"""Build or apply the Studio Wave-0 report from machine-readable receipts."""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.studio.wave0_report import (
    build_wave0_report,
    load_json,
    render_markdown,
    upsert_report_block,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="synthesize Studio Wave-0 report receipt")
    ap.add_argument("--transfer", default=str(REPO_ROOT / "runs" / "studio_wave0" / "transfer_check.json"))
    ap.add_argument("--daemon-state", default=str(REPO_ROOT / "runs" / "studio_wave0" / "daemon_state.json"))
    ap.add_argument("--encode-device", default=str(REPO_ROOT / "runs" / "mot" / "encode_device.json"))
    ap.add_argument("--encode-schedule", default=str(REPO_ROOT / "runs" / "mot" / "encode_schedule.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_wave0" / "wave0_report.json"))
    ap.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "docs" / "mixture_of_perspectives" / "STUDIO_RUN_REPORT.md"),
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="insert/replace the auto block in STUDIO_RUN_REPORT.md",
    )
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    report = build_wave0_report(
        transfer=load_json(a.transfer),
        daemon_state=load_json(a.daemon_state),
        encode_device=load_json(a.encode_device),
        encode_schedule=load_json(a.encode_schedule),
    )
    write_json(report, a.out)
    block = render_markdown(report)
    if a.apply:
        upsert_report_block(a.report_md, block)
    print(json.dumps({"out": a.out, "all_ok": report["all_ok"], "markdown": block}, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
