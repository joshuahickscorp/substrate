from __future__ import annotations

from .registries import load_paperwatch


def watch_report(path=None) -> dict:
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
