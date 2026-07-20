#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.process_c_gate import (  # noqa: E402
    DEFAULT_DR1_VERIFICATION,
    DEFAULT_NULL_CARD,
    DEFAULT_PR9_VERDICT,
    build_process_c_license_gate,
    load_json,
    write_process_c_license_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the Process C license gate receipt")
    parser.add_argument("--pr9-verdict", default=str(_ROOT / DEFAULT_PR9_VERDICT))
    parser.add_argument("--dr1-verification", default=str(_ROOT / DEFAULT_DR1_VERIFICATION))
    parser.add_argument("--null-card", default=str(_ROOT / DEFAULT_NULL_CARD))
    parser.add_argument("--min-params", type=int, default=1_000_000)
    parser.add_argument("--max-params", type=int, default=10_000_000)
    parser.add_argument("--out", default=str(_ROOT / "runs" / "mot" / "process_c_license_gate.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    receipt = build_process_c_license_gate(
        pr9_verdict=load_json(args.pr9_verdict),
        dr1_verification=load_json(args.dr1_verification),
        null_card_path=args.null_card,
        min_params=args.min_params,
        max_params=args.max_params,
    )
    write_process_c_license_gate(receipt, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": receipt["all_ok"],
                "status": receipt["status"],
                "decision": receipt["decision"],
                "licensed": receipt["licensed"],
                "launch_allowed": receipt["launch_allowed"],
                "licensing_sources": receipt["licensing_sources"],
                "problems": receipt["problems"],
            },
            indent=2,
        )
    )
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
