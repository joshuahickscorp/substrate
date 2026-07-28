"""Single command family for the Substrate final revision."""

from __future__ import annotations

import json
import sys
from typing import Any

from substrate import final_revision_campaign as campaign
from substrate import final_revision_io as io


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
    if command == "research":
        report = campaign.research(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "grok-review":
        report = campaign.grok_review(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "reproduce-null":
        report = campaign.reproduce_null(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "tournament":
        report = campaign.tournament(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "acquire":
        report = campaign.acquire(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "canaries":
        report = campaign.canaries(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "pilot":
        report = campaign.pilot(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "freeze":
        report = campaign.freeze(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "run":
        report = campaign.run(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "status":
        _print(campaign.status())
        return
    if command == "stop":
        _print({"stopped": True, "stop_switch": str(io.stop()), "activation": False})
        return
    if command == "resume":
        io.resume()
        _print({"resumed": True, "status": campaign.status(), "activation": False})
        return
    if command == "verify":
        report = campaign.verify(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "publish":
        report = campaign.publish(publish_files=publish)
        _exit(report, bool(report["all_pass"]))
    raise SystemExit(f"unknown final-revision command {command!r}")
