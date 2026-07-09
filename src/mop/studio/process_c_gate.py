"""Process C license gate.

Process C is an explicitly conditional pilot. This module turns the doctrine rule into a durable
receipt: PR9 can license the small dense-token pilot via its verdict ledger, or DR1 can license it
only when the real-video cache is integrity-clean and the adversarial A6 condition exposes the
frozen-representation wall. The receipt decides authorization; it never trains a module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-process-c-license-gate/v1"
DEFAULT_PR9_VERDICT = "runs/mot/pr9_verdict_ledger.json"
DEFAULT_DR1_VERIFICATION = "data/cache/vjepa2_vitl_comp_video/dr1_verification.json"
DEFAULT_NULL_CARD = "proof/NULL_CARDS/process_c_dense_token_pilot.md"


def build_process_c_license_gate(
    *,
    pr9_verdict: dict[str, Any] | None,
    dr1_verification: dict[str, Any] | None,
    null_card_path: str | Path = DEFAULT_NULL_CARD,
    min_params: int = 1_000_000,
    max_params: int = 10_000_000,
) -> dict[str, Any]:
    """Build the Process C authorization receipt from PR9 and DR1 verdict artifacts."""
    null_card = _null_card_summary(null_card_path)
    pr9 = _pr9_source(pr9_verdict)
    dr1 = _dr1_source(dr1_verification)
    sources = {"pr9": pr9, "dr1": dr1}
    licensing_sources = [name for name, src in sources.items() if src["licensed"]]
    decisive_sources = [name for name, src in sources.items() if src["decisive"]]

    problems: list[str] = []
    warnings: list[str] = []
    if not null_card["exists"]:
        problems.append(f"missing Process C null card: {null_card['path']}")
    for name, source in sources.items():
        if source["fatal"]:
            problems.extend(f"{name}: {p}" for p in source["problems"])
        elif source["problems"]:
            warnings.extend(f"{name}: {p}" for p in source["problems"])
    if not decisive_sources:
        problems.append("no decisive PR9 or DR1 receipt is available to evaluate Process C licensing")

    licensed = bool(licensing_sources)
    if problems:
        status = "undecidable"
        decision = "NO-DECISION"
    elif licensed:
        status = "licensed"
        decision = "LICENSED"
    else:
        status = "not_licensed"
        decision = "NOT-LICENSED"

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "decision": decision,
        "all_ok": not problems,
        "licensed": licensed,
        "launch_allowed": bool(licensed and not problems),
        "licensing_sources": licensing_sources,
        "decisive_sources": decisive_sources,
        "null_card": null_card,
        "pilot_budget": {
            "min_params": int(min_params),
            "max_params": int(max_params),
            "rule": "the only authorized pilot is a 1 to 10M object-centric dense-token shell",
        },
        "sources": sources,
        "blockers": [] if licensed else _license_blockers(sources),
        "warnings": warnings,
        "problems": problems,
        "publish_rule": (
            "Process C cannot run unless launch_allowed is true, the dedicated null card is strict, "
            "the dense real/control cache gate is present for the training data plane, and the pilot "
            "uses matched frozen-representation controls inside the 1 to 10M parameter cap"
        ),
    }


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    """Load a JSON object if it exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data if isinstance(data, dict) else None


def write_process_c_license_gate(receipt: dict[str, Any], path: str | Path) -> None:
    """Write the Process C license gate receipt."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def _pr9_source(verdict: dict[str, Any] | None) -> dict[str, Any]:
    if verdict is None:
        return _source(
            status="missing",
            decisive=False,
            licensed=False,
            fatal=False,
            reason="missing PR9 verdict ledger",
            problems=["missing PR9 verdict ledger"],
        )
    if verdict.get("schema") != "mop-pr9-verdict-ledger/v1":
        return _source(
            status="invalid_schema",
            decisive=False,
            licensed=False,
            fatal=True,
            reason="unexpected PR9 verdict ledger schema",
            problems=[f"unexpected PR9 verdict ledger schema {verdict.get('schema')!r}"],
        )
    if not verdict.get("all_ok"):
        status = str(verdict.get("status") or "unknown")
        return _source(
            status="not_scoring",
            decisive=False,
            licensed=False,
            fatal=True,
            reason=f"PR9 verdict ledger is not complete/scoring: {status}",
            problems=list(verdict.get("problems", [])) or [f"PR9 verdict status {status} is not scoring"],
            summary=_pr9_summary(verdict),
        )
    licensed = bool(verdict.get("process_c_licensed"))
    return _source(
        status="licensed" if licensed else "complete_no_license",
        decisive=True,
        licensed=licensed,
        fatal=False,
        reason=(
            "PR9 verdict ledger licenses Process C"
            if licensed
            else "PR9 verdict ledger completed but did not license Process C"
        ),
        summary=_pr9_summary(verdict),
    )


def _dr1_source(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return _source(
            status="missing",
            decisive=False,
            licensed=False,
            fatal=False,
            reason="missing DR1 adversarial verification receipt",
            problems=["missing DR1 adversarial verification receipt"],
        )
    if report.get("schema") != "mop-dr1-adversarial-verification/v1":
        return _source(
            status="invalid_schema",
            decisive=False,
            licensed=False,
            fatal=True,
            reason="unexpected DR1 verifier schema",
            problems=[f"unexpected DR1 verifier schema {report.get('schema')!r}"],
        )
    if not report.get("integrity_ok"):
        return _source(
            status="integrity_failed",
            decisive=False,
            licensed=False,
            fatal=True,
            reason="DR1 artifact integrity failed",
            problems=list(report.get("problems", [])) or ["DR1 artifact integrity failed"],
            summary=_dr1_summary(report),
        )
    if not (report.get("independent") and report.get("adversarial")):
        return _source(
            status="not_independent",
            decisive=False,
            licensed=False,
            fatal=True,
            reason="DR1 verifier is not both independent and adversarial",
            problems=["DR1 verifier is not both independent and adversarial"],
            summary=_dr1_summary(report),
        )

    a6_survives = report.get("a6_survives")
    if a6_survives is False and not report.get("passed"):
        return _source(
            status="licensed_representational_wall",
            decisive=True,
            licensed=True,
            fatal=False,
            reason="DR1 integrity is clean and the adversarial A6 verifier refused survival",
            summary=_dr1_summary(report),
        )
    if a6_survives is True and report.get("passed"):
        return _source(
            status="complete_no_license",
            decisive=True,
            licensed=False,
            fatal=False,
            reason="DR1 verifier passed, so no DR1 representational wall licenses Process C",
            summary=_dr1_summary(report),
        )
    return _source(
        status="undecidable",
        decisive=False,
        licensed=False,
        fatal=True,
        reason="DR1 verifier lacks an explicit A6 survival decision",
        problems=["DR1 verifier lacks an explicit A6 survival decision"],
        summary=_dr1_summary(report),
    )


def _source(
    *,
    status: str,
    decisive: bool,
    licensed: bool,
    fatal: bool,
    reason: str,
    problems: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "decisive": bool(decisive),
        "licensed": bool(licensed),
        "fatal": bool(fatal),
        "reason": reason,
        "summary": summary or {},
        "problems": problems or [],
    }


def _pr9_summary(verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": verdict.get("status"),
        "decision": verdict.get("decision"),
        "all_ok": verdict.get("all_ok"),
        "process_c_licensed": verdict.get("process_c_licensed"),
        "cache": verdict.get("cache"),
        "certificate": verdict.get("certificate", {}),
        "claim_status": verdict.get("claim_status"),
    }


def _dr1_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "integrity_ok": report.get("integrity_ok"),
        "a6_survives": report.get("a6_survives"),
        "passed": report.get("passed"),
        "all_ok": report.get("all_ok"),
        "independent": report.get("independent"),
        "adversarial": report.get("adversarial"),
        "cache_dir": report.get("cache_dir"),
        "summary": report.get("summary", {}),
    }


def _license_blockers(sources: dict[str, dict[str, Any]]) -> list[str]:
    return [f"{name}:{source['reason']}" for name, source in sources.items() if not source["licensed"]]


def _null_card_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "exists": p.exists() and p.is_file()}
