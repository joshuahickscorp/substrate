
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .null_cards import load_card, validate_card

SCHEMA = "mop-verdict-gate/v1"
POSITIVE_VERDICTS = {"PUBLISH-POSITIVE"}
PASS_KEYS = ("passed", "pass", "all_ok", "all_passed", "clean", "ok", "verified")
INDEPENDENCE_KEYS = ("independent", "independent_verifier", "adversarial", "adversarial_pass")
FORM_VERIFIER_SCHEMA = "mop-form-independent-verifier/v1"


def build_verdict_gate(
    *,
    null_card_path: Path | str,
    run_receipt_path: Path | str,
    verifier_receipt_path: Path | str | None = None,
    declared_verdict: str | None = None,
    strict_card: bool = True,
) -> dict[str, Any]:
    problems: list[str] = []
    card_path = Path(null_card_path)
    run_path = Path(run_receipt_path)
    verifier_path = Path(verifier_receipt_path) if verifier_receipt_path is not None else None

    card, card_problems = _load_card(card_path, strict_card)
    problems.extend(card_problems)
    verdict = str(declared_verdict or card.get("verdict") or "")
    positive = verdict in POSITIVE_VERDICTS

    run = _receipt_summary(run_path, "run_receipt")
    problems.extend(run["problems"])

    verifier = _verifier_summary(verifier_path, run_path)
    if positive:
        problems.extend(verifier["problems"])
        if (
            verifier.get("path")
            and run["path"]
            and Path(str(verifier["path"])).resolve() == run_path.resolve()
        ):
            problems.append("verifier_receipt must be independent: path equals run_receipt")
        if not verifier["passed"]:
            problems.append("positive verdict requires verifier receipt with a passed/all_ok/clean/ok flag")
        if not verifier["independent"]:
            problems.append("positive verdict requires verifier receipt with an independent/adversarial flag")

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_id": str(card.get("exp_id") or ""),
        "verdict": verdict,
        "positive": positive,
        "strict_card": bool(strict_card),
        "all_ok": not problems,
        "null_card": {
            "path": str(card_path),
            "exists": card_path.exists(),
            "sha256": _sha256(card_path),
            "problems": card_problems,
        },
        "run_receipt": run,
        "verifier_receipt": verifier,
        "problems": problems,
    }


def write_verdict_gate(gate: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate, indent=2, default=str) + "\n")


def _load_card(path: Path, strict: bool) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"null_card missing: {path}"]
    try:
        card = load_card(path)
    except Exception as e:
        return {}, [f"null_card parse failed: {e}"]
    return card, validate_card(card, strict=strict)


def _receipt_summary(path: Path, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path),
        "json_ok": False,
        "problems": [],
    }
    if not path.exists():
        out["problems"].append(f"{label} missing: {path}")
        return out
    try:
        json.loads(path.read_text())
    except Exception as e:
        out["problems"].append(f"{label} is not valid JSON: {e}")
        return out
    out["json_ok"] = True
    return out


def _verifier_summary(path: Path | None, run_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "sha256": _sha256(path) if path is not None else None,
        "json_ok": False,
        "passed": False,
        "independent": False,
        "problems": [],
    }
    if path is None:
        out["problems"].append("verifier_receipt missing")
        return out
    if not path.exists():
        out["problems"].append(f"verifier_receipt missing: {path}")
        return out
    if path.resolve() == run_path.resolve():
        out["problems"].append("verifier_receipt path equals run_receipt path")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        out["problems"].append(f"verifier_receipt is not valid JSON: {e}")
        return out
    out["json_ok"] = True
    out["passed"] = truthy_top_level(data, PASS_KEYS)
    out["independent"] = truthy_top_level(data, INDEPENDENCE_KEYS)
    if data.get("schema") == FORM_VERIFIER_SCHEMA:
        out["problems"].extend(_form_verifier_problems(data, run_path))
        if out["problems"]:
            out["passed"] = False
            out["independent"] = False
    return out


def truthy_top_level(obj: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(obj, dict):
        return False
    normalized = {str(key).strip().lower(): value for key, value in obj.items()}
    return any(normalized.get(key) is True for key in keys)


def _form_verifier_problems(data: dict[str, Any], run_path: Path) -> list[str]:
    problems: list[str] = []
    if data.get("passed") is not True or data.get("all_ok") is not True:
        problems.append("form verifier top-level passed/all_ok flags must both be true")
    if data.get("independent") is not True or data.get("adversarial") is not True:
        problems.append("form verifier must be both independent and adversarial")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append("form verifier checks are missing")
    elif any(not isinstance(check, dict) or check.get("passed") is not True for check in checks):
        problems.append("form verifier contains a failed or malformed check")
    if data.get("problems") not in ([], tuple()):
        problems.append("form verifier reports problems")

    fresh = data.get("fresh_seeds_or_heldout_cases")
    canonical = data.get("canonical_seeds")
    if not isinstance(fresh, list) or len(fresh) < 5 or len(set(fresh)) != len(fresh):
        problems.append("form verifier needs at least five unique fresh seeds or held-out cases")
    elif isinstance(canonical, list) and set(fresh) & set(canonical):
        problems.append("form verifier fresh seeds overlap its canonical seeds")

    frozen = data.get("frozen_live_contract")
    required_fingerprints = (
        "contract_fingerprint",
        "config_sha256",
        "class_source_sha256",
        "verifier_sha256",
    )
    if not isinstance(frozen, dict):
        problems.append("form verifier frozen live contract is missing")
    else:
        missing = [key for key in required_fingerprints if not str(frozen.get(key) or "")]
        if missing:
            problems.append(f"form verifier source fingerprints missing: {missing}")

    source_receipts = data.get("source_receipts")
    matching_source = False
    if isinstance(source_receipts, list):
        for source in source_receipts:
            if not isinstance(source, dict):
                continue
            source_path = Path(str(source.get("path") or ""))
            if not source_path.is_absolute():
                candidates = [ancestor / source_path for ancestor in run_path.parents]
                source_path = next((candidate for candidate in candidates if candidate.exists()), source_path)
            try:
                same_path = source_path.resolve() == run_path.resolve()
            except OSError:
                same_path = False
            matching_source = matching_source or (same_path and source.get("sha256") == _sha256(run_path))
    if not matching_source:
        problems.append("form verifier is not fingerprint-bound to this canonical run receipt")
    return problems


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
