"""The only public Substrate command."""

from __future__ import annotations

import json
import subprocess
import sys


def _print(document: dict) -> None:
    print(json.dumps(document, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "status"

    if command == "v2":
        from substrate import v2

        v2.main(argv)
        return
    if command == "v3":
        from substrate import v3

        v3.main(argv)
        return
    if command == "v4":
        from substrate import v4

        v4.main(argv)
        return
    if command == "v5":
        from substrate import v5

        v5.main(argv)
        return
    if command == "nous-closure":
        from substrate import nous_closure

        nous_closure.main(argv)
        return
    if command == "final-revision":
        from substrate import final_revision

        final_revision.main(argv)
        return
    if command == "genesis":
        from substrate import genesis

        genesis.main(argv)
        return
    if command == "sandbox":
        from substrate import sandbox

        sandbox.main(argv)
        return
    if command == "product":
        from substrate.product import cli as product_cli

        product_cli.main(argv)
        return

    if command == "test":
        raise SystemExit(subprocess.run([sys.executable, "-m", "pytest", "tests/substrate", *argv]).returncode)

    if command == "verify":
        from substrate import audit, verification

        structural = audit.run()
        independent = verification.recompute()
        _print({"structural": structural, "independent": independent, "activation": False})
        raise SystemExit(0 if structural["all_pass"] and independent["all_pass"] else 1)
    if command == "rehearse":
        from substrate import execution

        document = execution.rehearse()
        _print(document)
        raise SystemExit(0 if document["all_pass"] else 1)
    if command == "run":
        from substrate import execution

        execution.main(["launch", *argv])
        return
    if command == "status":
        from substrate import execution

        _print(execution.status())
        return
    if command == "workers":
        from substrate import execution

        _print(execution.workers())
        return
    if command == "resources":
        from substrate import execution

        document = execution.resources()
        _print(document)
        raise SystemExit(0 if document["launch_permitted"] else 1)
    if command == "doctor":
        from substrate import execution

        document = execution.doctor()
        _print(document)
        raise SystemExit(0 if document["all_pass"] else 1)
    if command == "stop":
        from substrate import evidence

        _print({"stopped": True, "stop_switch": str(evidence.stop()), "activation": False})
        return
    if command == "resume":
        from substrate import evidence, execution

        evidence.resume()
        execution.main(["launch", *argv])
        return
    if command == "audit":
        from substrate import audit

        document = audit.run()
        _print(document)
        raise SystemExit(0 if document["all_pass"] else 1)
    if command == "regenerate":
        from substrate import authority, deliverables, execution, verification

        deliverables.main(["seal-modules"])
        deliverables.main(["write"])
        authority.main(["seal"])
        execution.main(["seal"])
        verification.main(["all"])
        return
    raise SystemExit(f"unknown command {command!r}")
