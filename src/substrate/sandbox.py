"""Command surface for Substrate Tangible Sandbox R2."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from substrate import sandbox_campaign as campaign


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="substrate sandbox")
    commands = root.add_subparsers(dest="command")
    for name in (
        "preflight",
        "research",
        "plan-acquisition",
        "acquire",
        "verify-acquisition",
        "generate",
        "inventory",
        "canaries",
        "pilot",
        "freeze",
        "run",
        "status",
        "stop",
        "resume",
        "verify",
        "clean-clone",
    ):
        commands.add_parser(name)
    publish = commands.add_parser("publish")
    publish.add_argument("--pr-number", type=int)
    publish.add_argument("--no-clean-clone", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(sys.argv[1:] if argv is None else argv)
    command = arguments.command or "status"
    if command == "preflight":
        document = campaign.write_preflight()
    elif command == "research":
        document = campaign.research()
    elif command == "plan-acquisition":
        document = campaign.acquisition_plan()
    elif command == "acquire":
        document = campaign.acquire()
    elif command == "verify-acquisition":
        document = campaign.verify()
    elif command == "canaries":
        document = campaign.canaries()
    elif command == "status":
        document = campaign.status()
    elif command == "stop":
        document = campaign.stop()
    elif command == "resume":
        document = campaign.resume()
    elif command == "verify":
        document = campaign.verify()
    elif command == "clean-clone":
        document = campaign.clean_clone()
    elif command == "publish":
        document = campaign.publish(
            pr_number=arguments.pr_number,
            run_clean_clone=not arguments.no_clean_clone,
        )
    else:
        document = campaign.refuse_stage(command)
    _print(document)
    if command == "verify" and not document["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
