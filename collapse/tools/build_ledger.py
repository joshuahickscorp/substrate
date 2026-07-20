"""Refresh the durable collapse state from its compact machine authorities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COLLAPSE = ROOT / "collapse"
PYTHON_ROOTS = ("src", "tests", "scripts", "collapse/tools", "legacy_scaffolding")
ITEM_FIELDS = {
    "id",
    "section",
    "kind",
    "title",
    "status",
    "evidence_paths",
    "validation",
    "commit",
    "rollback_tag",
    "dependency",
    "next_action",
}


def sh(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def python_loc(root: str) -> int:
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in (ROOT / root).rglob("*.py"))


def proof_index() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_digest: dict[str, list[str]] = {}
    for relative in sh("git", "ls-files", "proof").splitlines():
        payload = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        by_digest.setdefault(digest, []).append(relative)
        entries.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": digest,
                "git_blob": sh("git", "hash-object", relative),
            }
        )
    value = {
        "schema": "mop-proof-index/v1",
        "files": len(entries),
        "bytes": sum(row["bytes"] for row in entries),
        "duplicate_groups": {digest: paths for digest, paths in by_digest.items() if len(paths) > 1},
        "entries": entries,
    }
    write(COLLAPSE / "MOP_PROOF_INDEX.json", value)
    return value


def reduction_totals(log: dict[str, Any]) -> dict[str, int]:
    fields = (
        "eliminated_LOC",
        "deduplicated_LOC",
        "relocated_LOC",
        "archived_LOC",
        "generated_replacement_LOC",
        "added_LOC",
        "net_reduction_LOC",
    )
    totals = {field: sum(int(row.get(field, 0)) for row in log.get("events", [])) for field in fields}
    totals["event_net_LOC"] = totals.pop("net_reduction_LOC")
    totals["net_reduction_LOC"] = totals["eliminated_LOC"] + totals["deduplicated_LOC"] - totals["added_LOC"]
    return totals


def apply_audit(state: dict[str, Any], audit: dict[str, Any]) -> None:
    rows = list(audit.get("checklist_evidence", []))
    for group in audit.get("checklist_groups", []):
        rows.extend({**group, "id": item_id} for item_id in group.get("ids", []))
    checklist = state["checklist"]
    by_id = {row["id"]: row for row in checklist}
    if len(by_id) != len(checklist):
        raise ValueError("durable checklist IDs must be unique")
    for row in checklist:
        if set(row) != ITEM_FIELDS:
            raise ValueError(f"checklist field drift: {row.get('id')}")
    for evidence in rows:
        target = by_id.get(evidence.get("id"))
        status = evidence.get("status")
        if target is None or status not in {"complete", "verified"}:
            raise ValueError(f"invalid audit checklist evidence: {evidence.get('id')}")
        target.update(
            status=status,
            evidence_paths=list(evidence.get("evidence_paths") or []),
            validation=str(evidence.get("validation") or ""),
            dependency="",
            next_action="none",
        )


def live_status() -> dict[str, Any]:
    path = Path("/Users/scammermike/Downloads/mop/runs/generation1/general-run/current_status.json")
    if not path.exists():
        return {"state": "missing"}
    status = load(path)
    return {key: status.get(key) for key in ("state", "stage", "updated_at", "counts")}


def render_ledger(state: dict[str, Any], log: dict[str, Any]) -> None:
    counts = state["checklist_summary"]["by_status"]
    current = state["current_measured"]
    active = [
        row
        for row in state["checklist"]
        if row["status"] in {"active", "partial"} and row["kind"] in {"region", "target"}
    ]
    lines = [
        "# MOP Collapse Ledger",
        "",
        "Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json`, "
        "`collapse/MOP_REDUCTION_LOG.json`, and `collapse/MOP_COMPLETION_AUDIT.json`.",
        "",
        "## Current",
        "",
        f"- Maintained Python: {current['global_maintained_python_LOC']:,} LOC; ceiling: 50,000.",
        f"- Runtime and campaign kernel: {current['runtime_campaign_kernel_LOC']:,} LOC.",
        f"- Validation: {current['validation_LOC']:,} LOC.",
        f"- Verified reduction ledger: {state['reduction_accounting_verified']['net_reduction_LOC']:,} LOC.",
        f"- Checklist: {json.dumps(counts, sort_keys=True)}.",
        "- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and "
        "`collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.",
        "",
        "## Active boundaries",
        "",
    ]
    lines.extend(f"- {row['id']}: {row['title']} -> {row['next_action']}" for row in active)
    lines.extend(
        [
            "",
            "## Recent green reductions",
            "",
            "| tag | net LOC | batch |",
            "| --- | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {row.get('tag', '')} | {int(row.get('net_reduction_LOC', 0)):,} | {row.get('batch', '')} |"
        for row in log.get("events", [])[-12:]
    )
    lines.append("")
    (ROOT / "MOP_COLLAPSE_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    state = load(ROOT / "MOP_COLLAPSE_STATE.json")
    if state.get("schema") != "mop-collapse-state/v1" or len(state.get("checklist", [])) != 261:
        raise ValueError("missing or incomplete durable collapse state")
    log = load(COLLAPSE / "MOP_REDUCTION_LOG.json")
    audit = load(COLLAPSE / "MOP_COMPLETION_AUDIT.json")
    index = proof_index()
    apply_audit(state, audit)

    source = python_loc("src")
    validation = python_loc("tests")
    maintained = sum(python_loc(root) for root in PYTHON_ROOTS)
    state["meta"].update(
        base_commit=sh("git", "rev-parse", "HEAD"),
        current_main=sh("git", "rev-parse", "origin/main"),
        live_general_run=live_status(),
    )
    state["current_measured"] = {
        "global_maintained_python_LOC": maintained,
        "runtime_campaign_kernel_LOC": source,
        "validation_LOC": validation,
        "normal_cli_count": 1,
        "installed_entrypoint_count": 1,
        "current_document_count": len(list(ROOT.glob("*.md"))),
        "current_document_LOC": sum(
            len(path.read_text(encoding="utf-8").splitlines()) for path in ROOT.glob("*.md")
        ),
        "optional_pack_LOC": 0,
    }
    state["reduction_accounting_verified"] = reduction_totals(log)
    state["key_findings"]["proof_index"] = {
        "files": index["files"],
        "bytes": index["bytes"],
        "duplicate_groups": len(index["duplicate_groups"]),
    }
    counts: dict[str, int] = {}
    for row in state["checklist"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    state["checklist_summary"] = {"total": len(state["checklist"]), "by_status": counts}
    write(ROOT / "MOP_COLLAPSE_STATE.json", state)
    render_ledger(state, log)
    print(f"refreshed {len(state['checklist'])} checklist items: {json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
