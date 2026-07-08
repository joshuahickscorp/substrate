"""Daemon plan builder for Studio claim ledger updates.

Positive ledger updates must not be hand-assembled. This module emits the required sequence:
verdict gate, artifact durability index, then the actual ledger command. The long-run daemon validates
that any positive-ledger job has the two required predecessors.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..falsification.verdict_gate import POSITIVE_VERDICTS
from .long_run import SCHEMA as DAEMON_SCHEMA
from .long_run import load_plan


def build_claim_daemon_plan(
    *,
    null_card: str,
    run_receipt: str,
    verdict_gate_out: str,
    artifact_index_out: str,
    ledger_cmd: Sequence[str],
    verifier_receipt: str | None = None,
    verdict: str = "PUBLISH-POSITIVE",
    artifact_paths: Sequence[str] = (),
    copy_dir: str | None = None,
    require_durable: bool = True,
    python: str = ".venv/bin/python",
) -> dict[str, Any]:
    """Build a daemon plan that gates a Studio ledger update."""
    ledger_command = [str(part) for part in ledger_cmd if str(part)]
    if not ledger_command:
        raise ValueError("ledger_cmd must contain at least one command token")
    positive = str(verdict) in POSITIVE_VERDICTS
    paths = _claim_artifacts(
        null_card=null_card,
        run_receipt=run_receipt,
        verdict_gate_out=verdict_gate_out,
        verifier_receipt=verifier_receipt,
        artifact_paths=artifact_paths,
    )
    plan = {
        "schema": DAEMON_SCHEMA,
        "jobs": [
            {
                "id": "claim_verdict_gate",
                "kind": "verdict-gate",
                "cmd": _verdict_gate_cmd(
                    python=python,
                    null_card=null_card,
                    run_receipt=run_receipt,
                    verifier_receipt=verifier_receipt,
                    verdict=verdict,
                    out=verdict_gate_out,
                ),
            },
            {
                "id": "claim_artifact_bundle",
                "kind": "artifact-bundle",
                "cmd": _artifact_bundle_cmd(
                    python=python,
                    paths=paths,
                    out=artifact_index_out,
                    copy_dir=copy_dir,
                    require_durable=require_durable,
                ),
            },
            {
                "id": "claim_ledger_update",
                "kind": "positive-ledger" if positive else "ledger",
                "cmd": ledger_command,
                "notes": "only runs after verdict and artifact gates succeed",
            },
        ],
    }
    problems = _validate_plan_object(plan)
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def write_claim_daemon_plan(plan: dict[str, Any], path: Path | str) -> None:
    """Write and re-load the plan so the daemon contract is checked before handoff."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    load_plan(out)


def _verdict_gate_cmd(
    *,
    python: str,
    null_card: str,
    run_receipt: str,
    verifier_receipt: str | None,
    verdict: str,
    out: str,
) -> list[str]:
    cmd = [
        python,
        "scripts/verdict_gate.py",
        "--null-card",
        null_card,
        "--run-receipt",
        run_receipt,
        "--verdict",
        verdict,
        "--out",
        out,
    ]
    if verifier_receipt is not None:
        cmd.extend(["--verifier-receipt", verifier_receipt])
    return cmd


def _artifact_bundle_cmd(
    *,
    python: str,
    paths: Sequence[str],
    out: str,
    copy_dir: str | None,
    require_durable: bool,
) -> list[str]:
    cmd = [python, "scripts/studio_artifact_bundle.py", "--only-paths", "--out", out]
    for path in paths:
        cmd.extend(["--path", path])
    if copy_dir is not None:
        cmd.extend(["--copy-dir", copy_dir])
    if require_durable:
        cmd.append("--require-durable")
    return cmd


def _claim_artifacts(
    *,
    null_card: str,
    run_receipt: str,
    verdict_gate_out: str,
    verifier_receipt: str | None,
    artifact_paths: Sequence[str],
) -> list[str]:
    paths = [null_card, run_receipt, verdict_gate_out]
    if verifier_receipt is not None:
        paths.append(verifier_receipt)
    paths.extend(str(path) for path in artifact_paths)
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _validate_plan_object(plan: dict[str, Any]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="mop_claim_plan_") as tmp:
        path = Path(tmp) / "plan.json"
        path.write_text(json.dumps(plan, indent=2) + "\n")
        try:
            load_plan(path)
        except ValueError as e:
            return [str(e)]
    return []
