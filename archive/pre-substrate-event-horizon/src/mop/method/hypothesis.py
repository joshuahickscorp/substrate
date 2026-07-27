"""The substrate hypothesis graph.

A linear campaign chain closes everything downstream of a null, including hypotheses the null said nothing
about. A graph closes only what depends on the failed premise, which is why four of the six explanations of
the substrate nulls are still open after five programs.

House style: no dashes.
"""

from __future__ import annotations

STATES = (
    "unopened",
    "headroom_pending",
    "instrument_pending",
    "admitted",
    "supported",
    "mixed",
    "null",
    "harm",
    "invalid",
    "superseded",
    "closed",
)

TERMINAL = ("supported", "null", "harm", "invalid", "superseded", "closed")
REQUIRED_FIELDS = (
    "id",
    "premise",
    "predecessor",
    "support",
    "contradictions",
    "required_bed",
    "required_headroom",
    "strongest_baseline",
    "cheapest_falsifier",
    "dependent_hypotheses",
    "state",
)


def validate(nodes: list[dict]) -> list[str]:
    ids = {n.get("id") for n in nodes}
    v = []
    for n in nodes:
        for f in REQUIRED_FIELDS:
            if f not in n:
                v.append(f"{n.get('id')}: missing {f}")
        if n.get("state") not in STATES:
            v.append(f"{n.get('id')}: unknown state {n.get('state')!r}")
        for p in [n.get("predecessor")] if isinstance(n.get("predecessor"), str) else (n.get("predecessor") or []):
            if p and p not in ids:
                v.append(f"{n.get('id')}: unknown predecessor {p!r}")
        for d in n.get("dependent_hypotheses") or []:
            if d not in ids:
                v.append(f"{n.get('id')}: unknown dependent {d!r}")
    return v


def propagate(nodes: list[dict], result: dict) -> dict:
    """Apply one result. A null closes only the descendants that require the failed premise."""
    by_id = {n["id"]: n for n in nodes}
    hid, verdict = result["hypothesis"], result["verdict"]
    if hid not in by_id:
        raise KeyError(f"unknown hypothesis {hid}")
    node = by_id[hid]
    node["state"] = verdict
    node.setdefault("results", []).append(result)
    closed, untouched = [], []
    if verdict in ("null", "invalid", "harm"):
        for d in node.get("dependent_hypotheses") or []:
            dep = by_id[d]
            requires = dep.get("requires_premise_of") or []
            if hid in requires:
                if dep["state"] not in TERMINAL:
                    dep["state"] = "closed"
                    dep.setdefault("closed_by", []).append(hid)
                closed.append(d)
            else:
                untouched.append(d)
    return {
        "hypothesis": hid,
        "new_state": verdict,
        "closed_descendants": closed,
        "independent_descendants_left_open": untouched,
        "still_open": [n["id"] for n in nodes if n["state"] not in TERMINAL],
    }


def summarize(nodes: list[dict]) -> dict:
    return {
        "n": len(nodes),
        "by_state": {s: [n["id"] for n in nodes if n["state"] == s] for s in STATES if any(n["state"] == s for n in nodes)},
        "open": [n["id"] for n in nodes if n["state"] not in TERMINAL],
        "terminal": [n["id"] for n in nodes if n["state"] in TERMINAL],
    }
