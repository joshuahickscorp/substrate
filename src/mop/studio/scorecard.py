
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-studio-scorecard/v1"
AUTO_START = "<!-- STUDIO-SCORECARD-AUTO:START -->"
AUTO_END = "<!-- STUDIO-SCORECARD-AUTO:END -->"
DEFAULT_DR1_CACHE = "data/cache/vjepa2_vitl_comp_video"


def build_studio_scorecard(
    *,
    wave0: dict[str, Any] | None = None,
    dr1_verification: dict[str, Any] | None = None,
    pr9_result: dict[str, Any] | None = None,
    pr9_state: dict[str, Any] | None = None,
    pr9_verdict: dict[str, Any] | None = None,
    process_c_gate: dict[str, Any] | None = None,
    dense_gate: dict[str, Any] | None = None,
    atlas_result: dict[str, Any] | None = None,
    atlas_verdict: dict[str, Any] | None = None,
    artifact_indexes: dict[str, dict[str, Any] | None] | None = None,
    spine_status: dict[str, Any] | None = None,
    dr1_cache: str = DEFAULT_DR1_CACHE,
) -> dict[str, Any]:
    indexes = artifact_indexes or {}
    axes: dict[str, dict[str, Any]] = {
        "falsification": _falsification_axis(indexes),
        "abstraction": _dr1_axis(dr1_verification),
        "moldability": _pr9_axis(pr9_result, pr9_state, pr9_verdict, dr1_cache=dr1_cache),
        "density": _atlas_axis(dense_gate, atlas_result, atlas_verdict),
        "durability": _durability_axis(indexes, spine_status),
    }
    launch = _launch_status(wave0)
    process_c = _process_c_decision(process_c_gate)
    blockers = _blockers(launch, axes, spine_status, process_c)
    scorecard: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "launch": launch,
        "axes": axes,
        "process_c": process_c,
        "spine": _spine_summary(spine_status),
        "artifact_indexes": _artifact_index_summary(indexes),
        "score": _score_summary(axes),
        "blockers": blockers,
    }
    scorecard["all_ok"] = bool(
        launch["status"] == "complete"
        and all(axis["status"] in {"evidence", "held", "complete", "walled"} for axis in axes.values())
        and process_c["status"] in {"licensed", "not_licensed"}
        and not blockers
    )
    scorecard["studio_10_ready"] = bool(
        scorecard["all_ok"]
        and axes["abstraction"]["status"] == "evidence"
        and axes["moldability"]["status"] in {"evidence", "walled"}
        and axes["density"]["status"] == "evidence"
        and axes["durability"]["status"] == "complete"
    )
    return scorecard


def load_json(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data if isinstance(data, dict) else None


def write_json(data: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n")


def render_markdown(scorecard: dict[str, Any]) -> str:
    status = "COMPLETE" if scorecard.get("studio_10_ready") else "INCOMPLETE"
    lines = [
        AUTO_START,
        "## Studio Scorecard Auto Receipt",
        "",
        f"- Status: {status}.",
        f"- Launch: {scorecard['launch']['status']} ({scorecard['launch']['detail']}).",
        "",
        "| Axis | Status | Evidence | Current score | Studio target |",
        "|---|---|---|---:|---:|",
    ]
    for axis_id, axis in scorecard["axes"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    axis_id,
                    str(axis["status"]),
                    _escape_table(str(axis["detail"])),
                    str(axis["current_score"]),
                    str(axis["target_score"]),
                ]
            )
            + " |"
        )
    next_cmd = (scorecard.get("spine") or {}).get("next_cmd_shell")
    if next_cmd:
        lines.extend(["", f"- Next spine command: `{next_cmd}`."])
    process_c = scorecard.get("process_c") or {}
    if process_c:
        lines.append(
            f"- Process C: {process_c.get('status')} ({_escape_table(str(process_c.get('detail', '')))})."
        )
    if scorecard.get("blockers"):
        lines.append(f"- Blocking receipts: {'; '.join(scorecard['blockers'])}.")
    lines.extend(["", AUTO_END])
    return "\n".join(lines) + "\n"


def upsert_report_block(report_path: Path | str, block: str) -> None:
    path = Path(report_path)
    text = path.read_text() if path.exists() else "# STUDIO RUN REPORT\n\n"
    if AUTO_START in text and AUTO_END in text:
        before, rest = text.split(AUTO_START, 1)
        _, after = rest.split(AUTO_END, 1)
        text = before.rstrip() + "\n\n" + block.rstrip() + "\n" + after
    else:
        anchor = "## Wave log"
        if anchor in text:
            text = text.replace(anchor, block + "\n" + anchor, 1)
        else:
            text = text.rstrip() + "\n\n" + block
    path.write_text(text)


def _launch_status(wave0: dict[str, Any] | None) -> dict[str, Any]:
    if wave0 is None:
        return {"status": "pending", "detail": "missing Wave 0 report"}
    if wave0.get("schema") != "mop-studio-wave0-report/v1":
        return {"status": "failed", "detail": f"unexpected schema {wave0.get('schema')!r}"}
    if wave0.get("all_ok"):
        return {"status": "complete", "detail": "Wave 0 report all_ok true"}
    return {"status": "pending", "detail": "Wave 0 report incomplete"}


def _falsification_axis(indexes: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    existing = [name for name, idx in indexes.items() if idx and idx.get("all_ok")]
    return _axis(
        "held",
        "pre-Studio falsification discipline held; positives still require verdict gate plus artifact bundle",
        current=10,
        target=10,
        receipts=existing,
    )


def _dr1_axis(dr1: dict[str, Any] | None) -> dict[str, Any]:
    if dr1 is None:
        return _axis("pending", "missing DR1 adversarial verification receipt", current=6, target=9)
    if dr1.get("schema") != "mop-dr1-adversarial-verification/v1":
        return _axis("failed", f"unexpected DR1 verifier schema {dr1.get('schema')!r}", current=6, target=9)
    if not dr1.get("integrity_ok"):
        return _axis(
            "failed",
            "DR1 artifact integrity failed",
            current=6,
            target=9,
            problems=dr1.get("problems", []),
        )
    if dr1.get("passed") and dr1.get("independent") and dr1.get("adversarial"):
        return _axis(
            "evidence",
            "DR1 integrity clean and A6 adversarial verifier passed; downstream positive still needs "
            "verdict gate",
            current=6,
            target=9,
            receipts=["dr1_verification"],
        )
    return _axis(
        "walled",
        "DR1 artifacts are preserved but the positive verifier refused A6 survival; treat as null or wall",
        current=6,
        target=9,
        problems=dr1.get("problems", []),
    )


def _pr9_axis(
    pr9: dict[str, Any] | None,
    state: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    *,
    dr1_cache: str,
) -> dict[str, Any]:
    if pr9 is None:
        return _axis("pending", "missing PR9 long-stream result", current=5, target=8)
    cache = str(pr9.get("cache") or "")
    if cache != dr1_cache:
        return _axis(
            "pending",
            f"PR9 result is not the DR1 real cache ({cache or 'missing cache'}); local smoke is non-scoring",
            current=5,
            target=8,
        )
    state_status = str((state or {}).get("status") or "")
    if state is None or state.get("schema") != "mop-pr9-run-state/v1" or state_status != "complete":
        return _axis("pending", "PR9 run-state receipt is missing or not complete", current=5, target=8)
    if verdict is None:
        return _axis("pending", "missing PR9 verdict ledger", current=5, target=8)
    if verdict.get("schema") != "mop-pr9-verdict-ledger/v1":
        return _axis(
            "failed",
            f"unexpected PR9 verdict ledger schema {verdict.get('schema')!r}",
            current=5,
            target=8,
        )
    if not verdict.get("all_ok"):
        status = str(verdict.get("status") or "")
        axis_status = "failed" if status in {"config_error", "non_scoring", "indeterminate"} else "pending"
        return _axis(
            axis_status,
            f"PR9 verdict ledger is not complete/scoring: {status or 'unknown'}",
            current=5,
            target=8,
            problems=verdict.get("problems", []),
        )
    if pr9.get("any_zero_reinit"):
        return _axis(
            "failed",
            "PR9 config error: at least one CBP arm never reinitialized",
            current=5,
            target=8,
        )
    if not pr9.get("lr_integral_matched_all", False):
        return _axis("walled", "PR9 compute match failed; comparison is a null", current=5, target=8)
    cert = pr9.get("certificate") or {}
    if pr9.get("null_supported") is False:
        return _axis(
            "evidence",
            "PR9 certificate fired and CBP restored plasticity without retention tax",
            current=5,
            target=8,
            receipts=["pr9_result", "pr9_state", "pr9_verdict_ledger"],
        )
    if pr9.get("null_supported") is True and cert.get("fired"):
        return _axis("walled", "PR9 certificate fired but CBP did not beat the null", current=5, target=8)
    return _axis(
        "walled",
        "PR9 found no certified plasticity loss to restore on the DR1 stream; Process C may be licensed",
        current=5,
        target=8,
    )


def _process_c_decision(gate: dict[str, Any] | None) -> dict[str, Any]:
    if gate is None:
        return {"status": "pending", "detail": "missing Process C license gate"}
    if gate.get("schema") != "mop-process-c-license-gate/v1":
        return {"status": "failed", "detail": f"unexpected Process C gate schema {gate.get('schema')!r}"}
    if not gate.get("all_ok"):
        return {
            "status": "pending",
            "detail": f"Process C license gate is undecidable: {gate.get('status') or 'unknown'}",
            "problems": gate.get("problems", []),
        }
    if gate.get("launch_allowed"):
        return {
            "status": "licensed",
            "detail": f"launch allowed by {gate.get('licensing_sources', [])}",
            "sources": gate.get("licensing_sources", []),
        }
    return {
        "status": "not_licensed",
        "detail": "PR9/DR1 receipts were evaluated and did not authorize Process C",
        "blockers": gate.get("blockers", []),
    }


def _atlas_axis(
    dense_gate: dict[str, Any] | None,
    atlas: dict[str, Any] | None,
    atlas_verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    if dense_gate is None:
        return _axis("pending", "missing dense/atlas cache gate", current=6, target=9)
    if dense_gate.get("schema") != "mop-dense-atlas-cache-gate/v1":
        return _axis(
            "failed",
            f"unexpected dense/atlas cache gate schema {dense_gate.get('schema')!r}",
            current=6,
            target=9,
        )
    if not dense_gate.get("all_ok"):
        return _axis(
            "pending",
            "dense/atlas cache gate is blocked; real and matched random-init dense caches are not ready",
            current=6,
            target=9,
            problems=dense_gate.get("problems", []),
        )
    if atlas is None:
        return _axis("pending", "missing dense/atlas result", current=6, target=9)
    if atlas_verdict is None:
        return _axis("pending", "missing atlas verdict ledger", current=6, target=9)
    if atlas_verdict.get("schema") != "mop-atlas-verdict-ledger/v1":
        return _axis(
            "failed",
            f"unexpected atlas verdict ledger schema {atlas_verdict.get('schema')!r}",
            current=6,
            target=9,
        )
    if not atlas_verdict.get("all_ok"):
        status = str(atlas_verdict.get("status") or "")
        axis_status = "failed" if status in {"dense_gate_invalid", "indeterminate"} else "pending"
        return _axis(
            axis_status,
            f"atlas verdict ledger is not complete/scoring: {status or 'unknown'}",
            current=6,
            target=9,
            problems=atlas_verdict.get("problems", []),
        )
    full_grid = bool(atlas.get("full_registered_grid"))
    full_pairs = bool(atlas.get("full_registered_pairs"))
    if not (full_grid and full_pairs):
        return _axis(
            "pending",
            "atlas result is partial; universal density/substrate scope is withheld",
            current=6,
            target=9,
            problems=[
                f"missing columns: {atlas.get('registered_columns_missing', [])}",
                f"missing arms: {atlas.get('registered_arms_missing', [])}",
            ],
        )
    ledger_status = str(atlas_verdict.get("status") or "")
    if ledger_status == "candidate_positive":
        return _axis(
            "evidence",
            f"atlas verdict ledger rejected the global null: {atlas.get('verdict')}",
            current=6,
            target=9,
            receipts=["dense_atlas_cache_gate", "atlas_result", "atlas_verdict_ledger"],
        )
    if ledger_status == "null_supported":
        return _axis(
            "walled",
            "atlas verdict ledger supports the density/substrate null",
            current=6,
            target=9,
        )
    if ledger_status == "no_typed_axes":
        return _axis("walled", "full atlas produced no typed columns or pairs", current=6, target=9)
    return _axis("walled", f"atlas verdict ledger ended in {ledger_status or 'unknown'}", current=6, target=9)


def _durability_axis(
    indexes: dict[str, dict[str, Any] | None],
    spine_status: dict[str, Any] | None,
) -> dict[str, Any]:
    required = ("wave0", "dr1", "pr9", "atlas", "form_substrate", "spine")
    missing = [name for name in required if indexes.get(name) is None]
    bad: list[str] = []
    for name in required:
        idx = indexes.get(name)
        if idx is not None and not idx.get("all_ok"):
            bad.append(name)
    if missing or bad:
        detail = f"missing indexes={missing}; failing indexes={bad}"
        return _axis("pending", detail, current=7, target=10)
    if spine_status is None or spine_status.get("schema") != "mop-studio-spine-status/v1":
        return _axis("pending", "missing final spine status receipt", current=7, target=10)
    return _axis(
        "complete",
        "all required artifact indexes and final spine status are present",
        current=7,
        target=10,
    )


def _artifact_index_summary(indexes: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, idx in sorted(indexes.items()):
        out[name] = (
            None if idx is None else {"all_ok": bool(idx.get("all_ok")), "summary": idx.get("summary", {})}
        )
    return out


def _spine_summary(spine_status: dict[str, Any] | None) -> dict[str, Any]:
    if spine_status is None:
        return {"status": "missing"}
    nxt = spine_status.get("next_step") or {}
    return {
        "status": "complete" if spine_status.get("all_complete") else "incomplete",
        "summary": spine_status.get("summary", {}),
        "next_step": nxt.get("id"),
        "next_cmd_shell": nxt.get("cmd_shell"),
        "missing_receipts": nxt.get("missing_receipts", []),
    }


def _score_summary(axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "current_mean": round(sum(float(axis["current_score"]) for axis in axes.values()) / len(axes), 3),
        "target_mean": round(sum(float(axis["target_score"]) for axis in axes.values()) / len(axes), 3),
        "scored_axes": sorted(axes),
    }


def _blockers(
    launch: dict[str, Any],
    axes: dict[str, dict[str, Any]],
    spine_status: dict[str, Any] | None,
    process_c: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if launch["status"] != "complete":
        blockers.append(f"launch:{launch['detail']}")
    for axis_id, axis in axes.items():
        if axis["status"] in {"pending", "failed"}:
            blockers.append(f"{axis_id}:{axis['detail']}")
    nxt = (spine_status or {}).get("next_step") or {}
    if nxt.get("id"):
        blockers.append(f"next:{nxt['id']}")
    if process_c["status"] in {"pending", "failed"}:
        blockers.append(f"process_c:{process_c['detail']}")
    return blockers


def _axis(
    status: str,
    detail: str,
    *,
    current: float,
    target: float,
    receipts: list[str] | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "detail": detail,
        "current_score": current,
        "target_score": target,
        "receipts": receipts or [],
        "problems": problems or [],
    }


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
