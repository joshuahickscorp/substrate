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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return value


def load(path: Path) -> dict[str, Any]:
    return (
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if path.exists()
        else {}
    )


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def decode_checklist(compact: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(compact["row_count"])
    ids = compact["ids"]
    titles = compact["titles"]
    dictionaries = compact["dictionaries"]
    vectors = {
        field: [int(value) for value in encoded.split(",")] for field, encoded in compact["vectors"].items()
    }
    if len(ids) != count or len(titles) != count or any(len(row) != count for row in vectors.values()):
        raise ValueError("normalized checklist column length drift")
    return [
        {
            "id": ids[index],
            "section": dictionaries["section"][vectors["section"][index]],
            "kind": dictionaries["kind"][vectors["kind"][index]],
            "title": titles[index],
            "status": dictionaries["status"][vectors["status"][index]],
            "evidence_paths": dictionaries["evidence_paths"][vectors["evidence_paths"][index]],
            "validation": dictionaries["validation"][vectors["validation"][index]],
            "commit": "",
            "rollback_tag": dictionaries["rollback_tag"][vectors["rollback_tag"][index]],
            "dependency": dictionaries["dependency"][vectors["dependency"][index]],
            "next_action": dictionaries["next_action"][vectors["next_action"][index]],
        }
        for index in range(count)
    ]


def decode_reductions(table: dict[str, Any]) -> list[dict[str, Any]]:
    fields = table["fields"]
    dictionaries = table["dictionaries"]
    events = []
    for encoded in table["events"]:
        cells = json.loads(f"[{encoded}]")
        if len(cells) != len(fields):
            raise ValueError("normalized reduction row width drift")
        event = {}
        for field, value in zip(fields, cells, strict=True):
            if value is None:
                continue
            event[field] = dictionaries[field][value] if field in dictionaries else value
        events.append(event)
    return events


def reduction_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "eliminated_LOC",
        "deduplicated_LOC",
        "relocated_LOC",
        "archived_LOC",
        "generated_replacement_LOC",
        "added_LOC",
        "net_reduction_LOC",
    )
    totals = {field: sum(int(row.get(field, 0)) for row in events) for field in fields}
    totals["event_net_LOC"] = totals.pop("net_reduction_LOC")
    totals["net_reduction_LOC"] = totals["eliminated_LOC"] + totals["deduplicated_LOC"] - totals["added_LOC"]
    return totals


def render_ledger(
    state: dict[str, Any], checklist: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    counts = state["checklist_summary"]["by_status"]
    current = state["current_measured"]
    active = [
        row
        for row in checklist
        if row["status"] in {"active", "partial"} and row["kind"] in {"region", "target"}
    ]
    lines = [
        "# MOP Collapse Ledger",
        "",
        "Compact generated view. Machine authority: `MOP_COLLAPSE_STATE.json`.",
        "",
        "## Current",
        "",
        f"- Maintained Python: {current['global_maintained_python_LOC']:,} LOC; ceiling: 50,000.",
        f"- Runtime and campaign kernel: {current['runtime_campaign_kernel_LOC']:,} LOC.",
        f"- Validation: {current['validation_LOC']:,} LOC.",
        f"- Verified reduction ledger: {state['reduction_accounting_verified']['net_reduction_LOC']:,} LOC.",
        f"- Checklist: {json.dumps(counts, sort_keys=True)}.",
        "- Recovery: state `legacy_authorities` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.",
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
        for row in events[-12:]
    )
    lines.append("")
    (ROOT / "MOP_COLLAPSE_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    state = load(ROOT / "MOP_COLLAPSE_STATE.json")
    if state.get("schema") != "mop-collapse-state/v2":
        raise ValueError("missing or incomplete durable collapse state")
    checklist = decode_checklist(state["checklist"])
    events = decode_reductions(state["reductions"])
    if len(checklist) != 261 or len({row["id"] for row in checklist}) != 261:
        raise ValueError("missing or duplicate durable checklist items")
    for row in checklist:
        if set(row) != ITEM_FIELDS:
            raise ValueError(f"checklist field drift: {row.get('id')}")
    index = proof_index()
    document_index = COLLAPSE / "MOP_HISTORICAL_DOCUMENT_INDEX.json"
    write(document_index, load(document_index))

    source = python_loc("src")
    validation = python_loc("tests")
    maintained = sum(python_loc(root) for root in PYTHON_ROOTS)
    state["meta"].update(
        base_commit=sh("git", "rev-parse", "HEAD"),
        current_main=sh("git", "rev-parse", "origin/main"),
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
    state["reduction_accounting_verified"] = reduction_totals(events)
    state["key_findings"]["proof_index"] = {
        "files": index["files"],
        "bytes": index["bytes"],
        "duplicate_groups": len(index["duplicate_groups"]),
    }
    counts: dict[str, int] = {}
    for row in checklist:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    state["checklist_summary"] = {"total": len(checklist), "by_status": counts}
    write(ROOT / "MOP_COLLAPSE_STATE.json", state)
    render_ledger(state, checklist, events)
    print(f"refreshed {len(checklist)} checklist items: {json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
