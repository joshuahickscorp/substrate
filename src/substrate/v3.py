"""The single Substrate v3 constitutional ascent command family."""

from __future__ import annotations

import json
import sys

from substrate import v3campaign
from substrate import v3io as io


def _print(document: dict) -> None:
    print(json.dumps(document, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "status"
    if command == "preflight":
        documents = v3campaign.seal_preflight()
        summary = {
            "all_pass": documents["preflight"]["all_pass"],
            "failed": documents["preflight"]["failed"],
            "v1_v2_immutable": documents["integrity"]["all_pass"],
            "hawking_processes": documents["hawking"]["snapshot"]["active_process_count"],
            "activation": False,
        }
        _print(summary)
        raise SystemExit(0 if summary["all_pass"] else 1)
    if command == "retrospect":
        documents = v3campaign.retrospect()
        frozen = v3campaign.freeze()
        _print(
            {
                "capabilities": documents["retrospective"]["capability_count"],
                "gaps": documents["retrospective"]["gap_count"],
                "frozen": len(frozen["sealed"]),
                "split_disjoint": frozen["split_disjoint"],
                "activation": False,
            }
        )
        return
    if command == "canaries":
        from substrate import v3canary

        documents = v3canary.run()
        summary = {
            "passed": documents["evidence"]["passed"],
            "total": documents["evidence"]["total"],
            "all_pass": documents["evidence"]["all_pass"],
            "all_terminal": documents["evidence"]["all_terminal"],
            "beds_valid": documents["bed"]["all_valid"],
            "moderate_pilot_licensed": documents["admission"]["moderate_pilot_licensed"],
            "activation": False,
        }
        _print(summary)
        raise SystemExit(0 if summary["all_pass"] and summary["all_terminal"] else 1)
    if command == "pilot":
        from substrate import v3pilot

        documents = v3pilot.run()
        summary = {
            "pilot": documents["pilot"]["all_pass"],
            "histories": documents["pilot"]["independent_histories"],
            "episodes": documents["pilot"]["episodes"],
            "failure_matrix": documents["failures"]["all_pass"],
            "selected_workers": documents["resources"]["selected_workers"],
            "principal_execution_licensed": documents["admission"]["principal_execution_licensed"],
            "activation": False,
        }
        _print(summary)
        raise SystemExit(0 if summary["principal_execution_licensed"] else 1)
    if command == "rehearse":
        from substrate import v3principal

        documents = v3principal.freeze()
        summary = {
            "unit_count": documents["manifest"]["unit_count"],
            "episodes": documents["manifest"]["expected_episodes"],
            "source_digest": documents["manifest"]["source_digest"],
            "configuration_digest": documents["manifest"]["configuration_digest"],
            "workers": documents["resources"]["workers"],
            "activation": False,
        }
        _print(summary)
        return
    if command == "run":
        from substrate import v3principal

        document = v3principal.run()
        _print(document)
        raise SystemExit(0 if document["status"]["remaining"] == 0 and not document["status"]["invalid"] and not document["failures"] else 1)
    if command == "status":
        from substrate import v3principal

        _print(v3principal.status())
        return
    if command == "stop":
        _print({"stopped": True, "stop_switch": str(io.stop()), "activation": False})
        return
    if command == "resume":
        from substrate import v3principal

        io.resume()
        document = v3principal.run()
        _print(document)
        raise SystemExit(0 if document["status"]["remaining"] == 0 and not document["status"]["invalid"] else 1)
    if command == "verify":
        from substrate import v3verify

        clean = v3verify.clean_clone()
        document = v3verify.finalize(clean)
        summary = {
            "all_pass": document["verification"]["all_pass"] and clean["all_pass"],
            "classification": document["classification"]["classification"],
            "mutations_zero_survivors": document["mutations"]["zero_survivors"],
            "clean_clone": clean["all_pass"],
            "activation": False,
        }
        _print(summary)
        raise SystemExit(0 if summary["all_pass"] else 1)
    raise SystemExit(f"unknown v3 command {command!r}")
