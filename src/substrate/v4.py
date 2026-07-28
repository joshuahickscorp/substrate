"""The single Substrate v4 structural-understanding command family."""

from __future__ import annotations

import json
import sys

from substrate import v4campaign, v4canary, v4io, v4pilot, v4principal, v4verify


def _print(document: dict) -> None:
    print(json.dumps(document, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "status"
    if command == "preflight":
        documents = v4campaign.seal_preflight()
        document = {
            "all_pass": documents["preflight"]["all_pass"],
            "failed": documents["preflight"]["failed"],
            "v1_v2_v3_immutable": documents["integrity"]["all_pass"],
            "hawking_processes": documents["hawking"]["snapshot"]["active_process_count"],
            "activation": False,
        }
        _print(document)
        raise SystemExit(0 if document["all_pass"] else 1)
    elif command == "audit":
        documents = v4campaign.root_cause()
        frozen = v4campaign.freeze()
        document = {
            "root_cause": documents["audit"]["conclusion"],
            "scientific_null_preserved": documents["audit"]["scientific_null_preserved"],
            "frozen_authorities": len(frozen["sealed"]),
            "split_disjoint": frozen["split_disjoint"],
            "activation": False,
        }
    elif command == "canaries":
        documents = v4canary.run()
        document = {
            "passed": documents["evidence"]["passed"],
            "total": documents["evidence"]["total"],
            "all_pass": documents["evidence"]["all_pass"],
            "all_terminal": documents["evidence"]["all_terminal"],
            "beds_valid": documents["bed"]["all_valid"],
            "moderate_pilot_licensed": documents["evidence"]["all_pass"] and documents["bed"]["all_valid"],
            "activation": False,
        }
        _print(document)
        raise SystemExit(0 if document["all_pass"] and document["all_terminal"] else 1)
    elif command == "pilot":
        documents = v4pilot.run()
        document = {
            "pilot": documents["pilot"]["all_primary_mechanisms_clear_sesoi"],
            "histories": documents["pilot"]["histories"],
            "episodes": documents["pilot"]["episodes"],
            "failure_matrix": documents["failures"]["all_pass"],
            "selected_workers": documents["resources"]["selected_workers"],
            "principal_execution_licensed": documents["admission"]["principal_launch_authorized"],
            "activation": False,
        }
        _print(document)
        raise SystemExit(0 if document["principal_execution_licensed"] else 1)
    elif command == "rehearse":
        documents = v4principal.prepare()
        document = {
            "unit_count": documents["manifest"]["unit_count"],
            "episodes": documents["manifest"]["total_episodes"],
            "source_digest": v4io.source_digest(),
            "configuration_digest": documents["manifest"].get("configuration_digest"),
            "workers": documents["resource_plan"]["selected_workers"],
            "activation": False,
        }
    elif command in {"run", "resume"}:
        v4io.resume()
        document = v4principal.run()
    elif command == "status":
        document = v4principal.status()
    elif command == "verify":
        document = v4verify.run_all()
        summary = {
            "all_pass": document["all_pass"],
            "classification": document["final"]["classification"]["classification"],
            "mutations_zero_survivors": document["mutation"]["zero_survived"],
            "clean_clone": document["clean_clone"]["all_pass"],
            "activation": False,
        }
        _print(summary)
        raise SystemExit(0 if summary["all_pass"] else 1)
    elif command == "stop":
        document = {"stopped": True, "stop_switch": str(v4io.stop()), "activation": False}
    else:
        raise SystemExit(f"unknown v4 command {command!r}")
    _print(document)
