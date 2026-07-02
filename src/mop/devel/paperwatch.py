"""Paper-watch command (Frontier 30). OFFLINE by default: reads registry/paperwatch.yaml and reports
the literature this project tracks, grouped by status, with the primary sources to check. Normal tests
never hit the network. An online refresh is opt-in and would only stamp last_checked + cache results;
it is intentionally NOT an auto-fetch here (no network dependency in the hot path).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from .registries import load_paperwatch


def watch_report(path=None) -> dict:
    """Offline paper-watch report: topics grouped by status, never-checked topics surfaced first."""
    topics = load_paperwatch(path)
    by_status: dict[str, list] = {}
    for t in topics:
        by_status.setdefault(t.get("status", "watching"), []).append(t["slug"])
    never = [t["slug"] for t in topics if str(t.get("last_checked", "never")).lower() == "never"]
    return {
        "n_topics": len(topics),
        "by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "never_checked": sorted(never),
        "topics": topics,
        "online": False,
        "note": "offline registry view; online refresh is opt-in and cached (not run here)",
    }


def render_md(report: dict) -> str:
    L = [
        "# Paper-watch (offline)",
        "",
        f"{report['n_topics']} topics. {report['note']}.",
        "",
        f"By status: {report['by_status']}",
        f"Never checked: {report['never_checked']}",
        "",
        "| topic | status | last checked | primary sources |",
        "| --- | --- | --- | --- |",
    ]
    for t in report["topics"]:
        primary = "; ".join(t.get("primary", []))
        L.append(f"| {t['slug']} | {t.get('status')} | {t.get('last_checked')} | {primary} |")
    return "\n".join(L) + "\n"
