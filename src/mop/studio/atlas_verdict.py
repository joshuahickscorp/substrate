"""Atlas verdict ledger.

The at-scale atlas raw JSON contains several internal nulls. This module turns it into a durable
publishability receipt by checking the paired dense-cache gate, the dedicated atlas null card, and the
full registered grid/pair status before the scorecard can count density evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-atlas-verdict-ledger/v1"
DEFAULT_NULL_CARD = "proof/NULL_CARDS/atlas_dense_multiencoder.md"


def build_atlas_verdict_ledger(
    *,
    atlas: dict[str, Any] | None,
    dense_gate: dict[str, Any] | None,
    null_card_path: str | Path = DEFAULT_NULL_CARD,
) -> dict[str, Any]:
    """Build a typed atlas verdict ledger from raw atlas and dense-gate receipts."""
    problems: list[str] = []
    null_card = _null_card_summary(null_card_path)
    if not null_card["exists"]:
        problems.append(f"missing atlas null card: {null_card['path']}")

    if dense_gate is None:
        return _ledger(
            status="dense_gate_missing",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            problems=[*problems, "missing dense/atlas cache gate receipt"],
        )
    if dense_gate.get("schema") != "mop-dense-atlas-cache-gate/v1":
        return _ledger(
            status="dense_gate_invalid",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            problems=[
                *problems,
                f"unexpected dense gate schema {dense_gate.get('schema')!r}",
            ],
        )
    if not dense_gate.get("all_ok"):
        return _ledger(
            status="dense_gate_blocked",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            problems=[*problems, *[f"dense gate: {p}" for p in dense_gate.get("problems", [])]],
        )
    if atlas is None:
        return _ledger(
            status="missing",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            problems=[*problems, "missing atlas result receipt"],
        )

    if not (atlas.get("full_registered_grid") and atlas.get("full_registered_pairs")):
        return _ledger(
            status="partial_non_scoring",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            decision="WITHHOLD-UNIVERSAL-SCOPE",
            claim_status="publish-null-or-partial-only",
            problems=[
                *problems,
                f"missing registered columns: {atlas.get('registered_columns_missing', [])}",
                f"missing registered arms: {atlas.get('registered_arms_missing', [])}",
            ],
        )

    null_supported = atlas.get("null_supported")
    if null_supported is False:
        return _ledger(
            status="candidate_positive",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            decision="CANDIDATE-POSITIVE",
            claim_status="candidate-positive-needs-verdict-gate",
            problems=problems,
        )
    if null_supported is True:
        return _ledger(
            status="null_supported",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            decision="NULL-SUPPORTED",
            claim_status="publish-null-or-wall",
            problems=problems,
        )
    if null_supported is None:
        return _ledger(
            status="no_typed_axes",
            atlas=atlas,
            dense_gate=dense_gate,
            null_card=null_card,
            decision="NO-TYPED-AXES",
            claim_status="publish-null-or-wall",
            problems=problems,
        )
    return _ledger(
        status="indeterminate",
        atlas=atlas,
        dense_gate=dense_gate,
        null_card=null_card,
        decision="NO-VERDICT",
        problems=[*problems, f"atlas null_supported has unexpected value {null_supported!r}"],
    )


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    """Load a JSON object if it exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data if isinstance(data, dict) else None


def write_atlas_verdict_ledger(ledger: dict[str, Any], path: str | Path) -> None:
    """Write the atlas verdict ledger."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2, default=str) + "\n")


def _ledger(
    *,
    status: str,
    atlas: dict[str, Any] | None,
    dense_gate: dict[str, Any] | None,
    null_card: dict[str, Any],
    problems: list[str],
    decision: str = "NO-VERDICT",
    claim_status: str = "do-not-publish",
) -> dict[str, Any]:
    atlas = atlas or {}
    dense_gate = dense_gate or {}
    complete_statuses = {"candidate_positive", "null_supported", "no_typed_axes"}
    all_ok = status in complete_statuses and not problems
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "decision": decision,
        "all_ok": all_ok,
        "null_card": null_card,
        "dense_gate": {
            "schema": dense_gate.get("schema"),
            "all_ok": dense_gate.get("all_ok"),
            "real_cache": (dense_gate.get("real_cache") or {}).get("path"),
            "randominit_cache": (dense_gate.get("randominit_cache") or {}).get("path"),
            "pair": dense_gate.get("pair", {}),
        },
        "atlas": {
            "full_registered_grid": bool(atlas.get("full_registered_grid")),
            "full_registered_pairs": bool(atlas.get("full_registered_pairs")),
            "registered_columns_missing": atlas.get("registered_columns_missing", []),
            "registered_arms_missing": atlas.get("registered_arms_missing", []),
            "null_supported": atlas.get("null_supported"),
            "verdict": atlas.get("verdict"),
            "atlas_scope": atlas.get("atlas_scope", {}),
        },
        "claim_status": claim_status,
        "publish_rule": (
            "positive atlas claims require this ledger, the strict atlas null card, dense real/control "
            "cache gate, raw atlas JSON, durable artifact index, and a separate verdict-gate path before "
            "doc mutation"
        ),
        "problems": problems,
    }


def _null_card_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "exists": p.exists() and p.is_file()}
