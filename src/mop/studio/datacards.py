"""Data cards + license ledger (Frontier 5). Every source, planned or acquired, gets a data card:
a single readable record of what it is, what it costs, what its license allows, how it was (or will
be) obtained, and its honest provenance tag. The license ledger rolls the whole registry into one
blocker table so the signed-terms / withdrawn / deferred sources are obvious BEFORE a Studio hour
is spent, not discovered mid-run.

A card merges the static registry entry with optional live state (a planner selection, an acquire
manifest entry): status, local path, acquired byte/clip counts, hash-manifest pointer, the exact
command used, and the date. Nothing here downloads or validates; it reports.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import json
from pathlib import Path

from .registry import load_datasets

# license status by source status: what a reviewer must DO before the Studio run
_BLOCKER = {
    "available": "none (open or already accessible)",
    "metadata-only": "metadata open; bulk media needs a licensed mirror (no scraping)",
    "manual": "ACTION: complete signed terms / accept the license, then re-plan with --accept-license",
    "blocked": "BLOCKED: source withdrawn or access-controlled; do not attempt",
    "deferred": "deferred by design (beyond a 1 TB local disk); not planned",
}


def data_card(entry: dict, state: dict | None = None, command: str = "", date: str = "") -> dict:
    """One source's data card as a dict (render with render_card_md). `entry` is the registry record;
    `state` optionally carries live acquisition facts (selected subset, acquired bytes/clips, dest,
    hash manifest path, status override)."""
    st = state or {}
    return {
        "slug": entry.get("slug"),
        "name": entry.get("name"),
        "status": st.get("status", entry.get("status")),
        "kind": entry.get("kind"),
        "modality": entry.get("modality"),
        "domain": entry.get("domain"),
        "local_path": st.get("dest") or st.get("local_path") or "(not acquired)",
        "license": entry.get("license"),
        "citation": entry.get("citation"),
        "allowed_use": entry.get("allowed_use"),
        "auth_steps": entry.get("auth_steps"),
        "raw_gb": entry.get("raw_gb"),
        "selected_subset": st.get("subset_clips", "(plan to set)"),
        "acquired_bytes": st.get("bytes"),
        "acquired_clips": st.get("n_clips"),
        "n_files": st.get("n_files"),
        "n_duplicates": st.get("n_duplicates"),
        "cache_gb_estimate": entry.get("cache_gb_pooled"),
        "hash_manifest": st.get("hash_manifest", "(written at acquire time)"),
        "quality_summary": st.get("quality_summary", "(populated by validate stage)"),
        "provenance_tag": entry.get("kind"),
        "command": command or "(planned)",
        "date": date or "(planned)",
        "risk": entry.get("risk"),
        "output_layout": entry.get("output_layout"),
    }


def render_card_md(card: dict) -> str:
    """A single data card as Markdown."""
    L = [
        f"# Data card: {card['name']} (`{card['slug']}`)",
        "",
        f"- **status**: {card['status']}  |  **kind/provenance**: {card['provenance_tag']}  |  "
        f"**modality/domain**: {card['modality']}/{card['domain']}",
        f"- **local path**: {card['local_path']}",
        f"- **license**: {card['license']}",
        f"- **allowed use**: {card['allowed_use']}",
        f"- **auth steps**: {card['auth_steps']}",
        f"- **citation**: {card['citation']}",
        f"- **raw size (est)**: {card['raw_gb']} GB  |  **cache size (est, pooled)**: "
        f"{card['cache_gb_estimate']} GB  |  **selected subset**: {card['selected_subset']}",
        f"- **acquired**: bytes={card['acquired_bytes']} files={card['n_files']} "
        f"clips={card['acquired_clips']} duplicates={card['n_duplicates']}",
        f"- **hash manifest**: {card['hash_manifest']}",
        f"- **output layout**: {card['output_layout']}",
        f"- **risk**: {card['risk']}",
        f"- **quality**: {card['quality_summary']}",
        f"- **command**: `{card['command']}`",
        f"- **date**: {card['date']}",
        "",
    ]
    return "\n".join(L)


def license_ledger(datasets: list[dict] | None = None) -> dict:
    """Roll the registry into a license/blocker view: one row per source with its status, license,
    and the action a reviewer must take. Returns {rows, counts, blockers} (blockers = the rows that
    need action: manual/blocked/deferred)."""
    ds = datasets if datasets is not None else load_datasets()
    rows = []
    counts: dict[str, int] = {}
    for d in ds:
        status = str(d.get("status"))
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "slug": d.get("slug"),
                "name": d.get("name"),
                "status": status,
                "license": d.get("license"),
                "action": _BLOCKER.get(status, "review"),
                "kind": d.get("kind"),
            }
        )
    blockers = [r for r in rows if r["status"] in ("manual", "blocked", "deferred")]
    return {"rows": rows, "counts": counts, "blockers": blockers, "n": len(rows)}


def render_ledger_md(ledger: dict) -> str:
    """The license ledger as Markdown: a full table plus a highlighted blockers section."""
    L = [
        "# License ledger",
        "",
        f"{ledger['n']} sources: " + ", ".join(f"{k}={v}" for k, v in sorted(ledger["counts"].items())),
        "",
        "| slug | status | kind | license | action before Studio run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in ledger["rows"]:
        L.append(f"| {r['slug']} | {r['status']} | {r['kind']} | {r['license']} | {r['action']} |")
    L += ["", "## Blockers (resolve before acquisition)", ""]
    if not ledger["blockers"]:
        L.append("- none")
    else:
        for b in ledger["blockers"]:
            L.append(f"- **{b['slug']}** ({b['status']}): {b['action']}")
    L.append("")
    return "\n".join(L)


def write_data_cards(out_dir: Path, plan: dict | None = None, manifest: dict | None = None) -> dict:
    """Write a data card per registry source plus the license ledger under out_dir. When a plan and/or
    acquire manifest are given, merge their live state into the matching cards. Returns
    {cards_dir, ledger_md, n_cards}."""
    out_dir = Path(out_dir)
    cards_dir = out_dir / "datacards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    selected = {s["slug"]: s for s in (plan or {}).get("selected", [])}
    acquired = {s["slug"]: s for s in (manifest or {}).get("sources", [])}
    datasets = load_datasets()
    for d in datasets:
        slug = d["slug"]
        state = {}
        state.update(selected.get(slug, {}))
        state.update({k: v for k, v in acquired.get(slug, {}).items() if v is not None})
        card = data_card(d, state=state or None)
        (cards_dir / f"{slug}.md").write_text(render_card_md(card))
    ledger = license_ledger(datasets)
    ledger_md = out_dir / "license_ledger.md"
    ledger_md.write_text(render_ledger_md(ledger))
    (out_dir / "license_ledger.json").write_text(json.dumps(ledger, indent=2, default=str))
    return {"cards_dir": str(cards_dir), "ledger_md": str(ledger_md), "n_cards": len(datasets)}
