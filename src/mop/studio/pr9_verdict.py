
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-pr9-verdict-ledger/v1"
DEFAULT_DR1_CACHE = "data/cache/vjepa2_vitl_comp_video"
DEFAULT_NULL_CARD = "proof/NULL_CARDS/pr9_long_stream_plasticity.md"


def build_pr9_verdict_ledger(
    *,
    result: dict[str, Any] | None,
    state: dict[str, Any] | None,
    null_card_path: str | Path = DEFAULT_NULL_CARD,
    dr1_cache: str = DEFAULT_DR1_CACHE,
) -> dict[str, Any]:
    problems: list[str] = []
    null_card = _null_card_summary(null_card_path)
    if not null_card["exists"]:
        problems.append(f"missing PR9 null card: {null_card['path']}")

    if result is None:
        return _ledger(
            status="missing",
            result=result,
            state=state,
            null_card=null_card,
            dr1_cache=dr1_cache,
            problems=[*problems, "missing PR9 result receipt"],
        )
    if state is None:
        return _ledger(
            status="pending",
            result=result,
            state=state,
            null_card=null_card,
            dr1_cache=dr1_cache,
            problems=[*problems, "missing PR9 run-state receipt"],
        )

    cache = str(result.get("cache") or "")
    if cache != dr1_cache:
        return _ledger(
            status="non_scoring",
            result=result,
            state=state,
            null_card=null_card,
            dr1_cache=dr1_cache,
            problems=[
                *problems,
                f"PR9 result cache {cache or 'missing'} is not the DR1 real cache {dr1_cache}",
            ],
        )

    state_status = str(state.get("status") or "")
    if state.get("schema") != "mop-pr9-run-state/v1" or state_status != "complete":
        return _ledger(
            status="pending",
            result=result,
            state=state,
            null_card=null_card,
            dr1_cache=dr1_cache,
            problems=[*problems, "PR9 run-state receipt is missing schema or not complete"],
        )

    status, decision, process_c_licensed, claim_status, status_problem = _classify_result(result)
    if status_problem:
        problems.append(status_problem)
    return _ledger(
        status=status,
        result=result,
        state=state,
        null_card=null_card,
        dr1_cache=dr1_cache,
        decision=decision,
        process_c_licensed=process_c_licensed,
        claim_status=claim_status,
        problems=problems,
    )


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data if isinstance(data, dict) else None


def write_pr9_verdict_ledger(ledger: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2, default=str) + "\n")


def _classify_result(result: dict[str, Any]) -> tuple[str, str, bool, str, str | None]:
    if result.get("any_zero_reinit"):
        return (
            "config_error",
            "CONFIG-ERROR",
            False,
            "do-not-publish",
            "at least one CBP arm never reinitialized",
        )
    if not result.get("lr_integral_matched_all", False):
        return (
            "null_compute_mismatch",
            "NULL-COMPUTE-MISMATCH",
            False,
            "publish-null-only",
            None,
        )
    cert = result.get("certificate") or {}
    if result.get("null_supported") is False:
        return (
            "evidence_cbp_win",
            "CANDIDATE-POSITIVE",
            False,
            "candidate-positive-needs-verdict-gate",
            None,
        )
    if result.get("null_supported") is True and cert.get("fired"):
        return (
            "null_cbp_no_win",
            "NULL-SUPPORTED",
            True,
            "publish-null-or-wall",
            None,
        )
    if result.get("null_supported") is True:
        return (
            "null_no_certificate",
            "NULL-NO-PLASTICITY-LOSS",
            True,
            "publish-null-or-wall",
            None,
        )
    return (
        "indeterminate",
        "NO-VERDICT",
        False,
        "do-not-publish",
        "PR9 result null_supported is neither true nor false",
    )


def _ledger(
    *,
    status: str,
    result: dict[str, Any] | None,
    state: dict[str, Any] | None,
    null_card: dict[str, Any],
    dr1_cache: str,
    problems: list[str],
    decision: str | None = None,
    process_c_licensed: bool = False,
    claim_status: str = "do-not-publish",
) -> dict[str, Any]:
    result = result or {}
    state = state or {}
    complete_statuses = {
        "null_compute_mismatch",
        "null_cbp_no_win",
        "null_no_certificate",
        "evidence_cbp_win",
    }
    all_ok = status in complete_statuses and not problems
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "decision": decision or "NO-VERDICT",
        "all_ok": all_ok,
        "dr1_cache": dr1_cache,
        "cache": result.get("cache"),
        "null_card": null_card,
        "run_state": {
            "schema": state.get("schema"),
            "status": state.get("status"),
            "expected_leg_count": state.get("expected_leg_count"),
            "completed_leg_count": state.get("completed_leg_count"),
            "resume_behavior": state.get("resume_behavior"),
        },
        "certificate": _certificate_summary(result.get("certificate")),
        "cbp": {
            "any_zero_reinit": bool(result.get("any_zero_reinit")),
            "lr_integral_matched_all": bool(result.get("lr_integral_matched_all")),
            "winning_rates": result.get("winning_rates", []),
            "reinit_count_total_all_rates": result.get("reinit_count_total_all_rates"),
            "best_rate_by_late_delta": result.get("best_rate_by_late_delta"),
        },
        "null_supported": result.get("null_supported"),
        "process_c_licensed": bool(process_c_licensed),
        "claim_status": claim_status,
        "publish_rule": (
            "positive PR9 claims require this ledger, the strict PR9 null card, raw result/state "
            "receipts, durable artifact index, and a separate verdict-gate path before doc mutation"
        ),
        "verdict": result.get("verdict"),
        "problems": problems,
    }


def _certificate_summary(cert: Any) -> dict[str, Any]:
    cert = cert if isinstance(cert, dict) else {}
    return {
        "fired": bool(cert.get("fired")),
        "adapt_trends_down": cert.get("adapt_trends_down"),
        "dead_trends_up": cert.get("dead_trends_up"),
        "adapt_slope_ci": cert.get("adapt_slope_ci"),
        "dead_slope_ci": cert.get("dead_slope_ci"),
    }


def _null_card_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists() and p.is_file(),
    }
