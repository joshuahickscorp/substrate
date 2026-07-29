"""Single command family for Substrate Cognitive Material Genesis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from substrate import genesis_campaign as campaign
from substrate import genesis_grok as grok
from substrate import genesis_io as io


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _exit(value: dict[str, Any], passed: bool) -> None:
    _print(value)
    raise SystemExit(0 if passed else 1)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    publish = "--no-publish" not in arguments
    arguments = [argument for argument in arguments if argument != "--no-publish"]
    command = arguments.pop(0) if arguments else "status"

    if command == "preflight":
        report = campaign.preflight(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "constitution":
        report = campaign.constitution(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "status":
        _print(campaign.status())
        return
    if command == "grok-summary":
        _print(grok.summary())
        return
    if command == "record-grok-review":
        if len(arguments) != 4:
            raise SystemExit("record-grok-review requires TASK_DIRECTORY CONTRACT_PATH ROLE ROUND")
        row = grok.invocation_record(
            Path(arguments[0]),
            Path(arguments[1]),
            expected_role=arguments[2],
            expected_round=arguments[3],
        )
        ledger = grok.record(row)
        _print(
            {
                "recorded": row["invocation_id"],
                "role": row["role"],
                "round": row["round"],
                "feasibility_out_of_20": row["feasibility_out_of_20"],
                "blocking_defect_count": row["blocking_defect_count"],
                "distinct_role_count": ledger["distinct_role_count"],
                "activation": False,
            }
        )
        return
    if command == "mark-stage":
        if len(arguments) != 2:
            raise SystemExit("mark-stage requires STAGE STATE")
        _print(campaign.mark_stage(arguments[0], arguments[1]))
        return
    if command == "stop":
        _print({"stopped": True, "stop_switch": str(io.stop()), "activation": False})
        return
    if command == "resume":
        io.resume()
        _print({"resumed": True, "activation": False})
        return
    raise SystemExit(f"unknown genesis command {command!r}")
