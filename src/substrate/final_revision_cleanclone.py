"""Clean-checkout verification entry point for terminal orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from substrate import final_revision_campaign as campaign
from substrate import final_revision_config as C
from substrate import final_revision_io as io
from substrate import final_revision_verification as verification


def _command(*arguments: str) -> dict[str, Any]:
    completed = subprocess.run(list(arguments), cwd=io.ROOT, capture_output=True, text=True, check=False)
    return {
        "command": list(arguments),
        "returncode": completed.returncode,
        "stdout_digest": io.digest(completed.stdout),
        "stderr_digest": io.digest(completed.stderr),
        "passed": completed.returncode == 0,
    }


def verify_clean_checkout(evidence_source: Path) -> dict[str, Any]:
    if not evidence_source.is_dir():
        raise io.Refused(f"evidence source does not exist: {evidence_source}")
    initial_clean = not io.git("status", "--porcelain")
    ready_commit = io.ref_or_none(C.READY_TAG, peel=True)
    lineage = {
        "head_is_ready_tag": io.ref_or_none("HEAD") == ready_commit,
        "historical_main": io.ref_or_none("main") == C.AUTHORITATIVE_MAIN,
        "closure_terminal_preserved": io.ref_or_none(C.NOUS_CLOSURE_TERMINAL_TAG, peel=True) == C.AUTHORITATIVE_MAIN,
        "initial_checkout_clean": initial_clean,
    }
    commands = {
        "tests": _command(sys.executable, "-m", "pytest", "-q"),
        "lint": _command(sys.executable, "-m", "ruff", "check", "src", "tests"),
    }
    preflight = campaign.preflight(publish=False)
    closure = campaign.reproduce_null(publish=False)
    canaries = campaign.canaries(publish=False)
    pilot = campaign.pilot(publish=False)
    recomputations = {}
    for split, filename in (
        ("principal", "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json"),
        ("replication", "SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json"),
        ("hidden_composition", "SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json"),
    ):
        document = io.load_json(evidence_source / filename)
        recomputations[split] = verification.recomputation_matches(document)
    checks = {
        **lineage,
        "tests": commands["tests"]["passed"],
        "lint": commands["lint"]["passed"],
        "preflight": preflight["all_pass"],
        "closure": closure["all_pass"],
        "canaries": canaries["all_pass"],
        "pilot": pilot["all_pass"],
        "independent_recomputation": all(row["exact_match"] for row in recomputations.values()),
        "activation_false": C.ACTIVATION is False,
    }
    return {
        "schema": "substrate-final-revision-clean-checkout-verification/v1",
        "root": str(io.ROOT),
        "ready_commit": ready_commit,
        "checks": checks,
        "commands": commands,
        "recomputations": recomputations,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def regenerate_twice() -> dict[str, Any]:
    first = campaign._terminal_documents()
    second = campaign._terminal_documents()
    first_bytes = io.canonical_bytes(first)
    second_bytes = io.canonical_bytes(second)
    return {
        "schema": "substrate-final-revision-regeneration-check/v1",
        "first_digest": io.digest(first_bytes),
        "second_digest": io.digest(second_bytes),
        "exact_agreement": first_bytes == second_bytes,
        "normalization": "canonical JSON; no nondeterministic terminal fields",
        "activation": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "regenerate"))
    parser.add_argument("--evidence-source", type=Path)
    options = parser.parse_args(argv)
    if options.mode == "verify":
        if options.evidence_source is None:
            raise SystemExit("--evidence-source is required for verify")
        report = verify_clean_checkout(options.evidence_source.resolve())
    else:
        report = regenerate_twice()
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["all_pass" if options.mode == "verify" else "exact_agreement"] else 1)


if __name__ == "__main__":
    main()
