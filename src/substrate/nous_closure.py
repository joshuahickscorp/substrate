"""Single command family for the Substrate Nous Closure campaign."""

from __future__ import annotations

import json
import sys
from typing import Any

from substrate import nous_closure_campaign as campaign
from substrate import nous_closure_io as io


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
    if command == "audit":
        report = campaign.counterfeit_documents()[0]
        _exit(report, bool(report["all_pass"]))
    if command == "sandbox":
        report = campaign.build_sandbox()
        _exit(report, bool(report["all_required_real_structures_present"]))
    if command == "canaries":
        report = campaign.canaries(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "pilot":
        report = campaign.pilot(publish=publish)
        _exit(report, bool(report["admission"]["terminal_closed_null_authorized"] or report["admission"]["principal_launch_authorized"]))
    if command == "rehearse":
        report = campaign.rehearse(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "run":
        admission_path = io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json"
        if not admission_path.is_file():
            campaign.pilot(publish=True)
        admission = io.load_json(admission_path)
        if not admission["principal_launch_authorized"]:
            _print(
                {
                    "status": "terminally_gated",
                    "reason": admission["stop_rule"],
                    "terminal_closed_null_authorized": admission["terminal_closed_null_authorized"],
                    "principal_units_launched": 0,
                    "activation": False,
                }
            )
            return
        raise io.Refused("positive admission requires the frozen principal DAG; no implicit launch path is permitted")
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
        terminal = "--terminal" in arguments
        report = campaign.verify(publish=publish, require_terminal=terminal)
        _exit(report, bool(report["all_pass"]))
    if command == "review-package":
        report = campaign.terminalize(clean_clone_full="--representative-clean-clone" not in arguments)
        _exit(report, bool(report["verification"]["all_pass"]))
    if command == "_review-cell":
        if not arguments:
            raise SystemExit("_review-cell requires A, B, or C")
        pilot = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json")
        _print(campaign.review_cell_report(arguments[0], pilot))
        return
    raise SystemExit(f"unknown nous-closure command {command!r}")


if __name__ == "__main__":
    main()
