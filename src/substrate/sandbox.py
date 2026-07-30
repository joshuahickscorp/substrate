"""Command surface for Substrate Tangible Sandbox R2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
        "prepare-public",
        "run-public",
        "run",
        "seal-continuity-repair",
        "status",
        "stop",
        "resume",
        "verify",
        "clean-clone",
    ):
        commands.add_parser(name)
    invalidation = commands.add_parser("invalidate-longitudinal")
    invalidation.add_argument("--reason", required=False)
    commands.add_parser("seal-continuity-supervision-repair")
    longitudinal = commands.add_parser("longitudinal")
    longitudinal.add_argument("--supervision-manifest", type=Path, required=False)
    commands.add_parser("launch-longitudinal")
    supervised = commands.add_parser("supervised-longitudinal")
    supervised.add_argument("--supervision-manifest", type=Path, required=True)
    commands.add_parser("longitudinal-supervision-status")
    publish = commands.add_parser("publish")
    publish.add_argument("--pr-number", type=int)
    publish.add_argument("--no-clean-clone", action="store_true")
    refusal = commands.add_parser("publish-continuity-refusal")
    refusal.add_argument("--pr-number", type=int)
    refusal.add_argument("--no-clean-clone", action="store_true")
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
    elif command == "generate":
        document = campaign.generate()
    elif command == "inventory":
        document = campaign.inventory()
    elif command == "pilot":
        document = campaign.pilot()
    elif command == "freeze":
        document = campaign.freeze()
    elif command == "prepare-public":
        document = campaign.prepare_public()
    elif command == "run-public":
        document = campaign.run_public()
    elif command == "run":
        document = campaign.run()
    elif command == "invalidate-longitudinal":
        document = campaign.invalidate_longitudinal_attempt(reason=arguments.reason)
    elif command == "seal-continuity-repair":
        document = campaign.seal_continuity_repair()
    elif command == "seal-continuity-supervision-repair":
        document = campaign.seal_continuity_supervision_repair()
    elif command == "longitudinal":
        document = campaign.longitudinal(
            supervision_manifest=arguments.supervision_manifest
        )
    elif command == "launch-longitudinal":
        document = campaign.launch_longitudinal_supervised()
    elif command == "supervised-longitudinal":
        document = campaign.supervised_longitudinal(arguments.supervision_manifest)
    elif command == "longitudinal-supervision-status":
        document = campaign.longitudinal_supervision_status()
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
    elif command == "publish-continuity-refusal":
        document = campaign.publish_continuity_refusal(
            pr_number=arguments.pr_number,
            run_clean_clone=not arguments.no_clean_clone,
        )
    else:
        document = campaign.refuse_stage(command)
    _print(document)
    if command == "verify" and not document["all_pass"]:
        raise SystemExit(1)
    if command == "supervised-longitudinal" and document["worker_returncode"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
