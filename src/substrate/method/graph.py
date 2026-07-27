"""Machine readable causal experiment graph, plus the validator that refuses to let a broken one through.

A graph declares what the experiment believes causes what, and by which kind of relation. The kind matters
more than the arrow: an implemented causal path is a claim about code that must exist, a structural
guarantee is a claim about construction that must not be reported as a measurement, and a forbidden
information path is a claim that must never be realized.

House style: no dashes.
"""

from __future__ import annotations

NODE_TYPES = (
    "scientific_construct",
    "mechanism",
    "intervention",
    "available_information",
    "hidden_information",
    "treatment",
    "control",
    "mediator",
    "confounder",
    "independent_unit",
    "primary_outcome",
    "cost",
    "verdict",
    "claim",
)

EDGE_TYPES = (
    "implemented_causal_path",  # there is code, and the arm audit can point at it
    "assumed_scientific_relation",  # believed, not established by this experiment
    "measured_relation",  # this experiment measures it
    "analytic_relation",  # closed form
    "structural_guarantee",  # true by construction
    "forbidden_information_path",  # must be proven absent, never realized
)

SCHEMA = {
    "schema": "substrate-experiment-causal-graph/v1",
    "node_types": list(NODE_TYPES),
    "edge_types": list(EDGE_TYPES),
    "node_fields": {
        "id": "unique string",
        "type": "one of node_types",
        "label": "human readable",
        "implementation": "dotted code path, required for mechanism, intervention, treatment, control",
        "time": "decision_time | future | any, required on information nodes",
        "declared": "bool, required on confounder nodes",
        "removes": "capability string, required on control nodes",
        "requires": "list of node ids, required on claim and verdict nodes",
    },
    "edge_fields": {"src": "node id", "dst": "node id", "type": "one of edge_types"},
    "rejections": [
        "mechanism without intervention",
        "reported variable without implementation",
        "output without measurement path",
        "control that still retains the removed capability",
        "future information entering a decision time mechanism",
        "analytic quantity reported as measured",
        "hidden confounder ignored without declaration",
        "claim broader than the measured path",
    ],
}


def _index(graph: dict):
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    out: dict[str, list] = {i: [] for i in nodes}
    inn: dict[str, list] = {i: [] for i in nodes}
    for e in graph.get("edges", []):
        if e.get("src") in out:
            out[e["src"]].append(e)
        if e.get("dst") in inn:
            inn[e["dst"]].append(e)
    return nodes, out, inn


def validate(graph: dict, evidence: dict | None = None) -> list[str]:
    """Return the list of rejections. Empty means the graph is admissible.

    evidence is optional runtime proof keyed by node id, for example control semantic receipts. Without it
    the structural rules still apply; with it the semantic rules apply too.
    """
    evidence = evidence or {}
    v: list[str] = []
    nodes, out, inn = _index(graph)
    if not nodes:
        return ["graph has no nodes"]

    for n in nodes.values():
        if n.get("type") not in NODE_TYPES:
            v.append(f"node {n.get('id')}: unknown type {n.get('type')!r}")
    for e in graph.get("edges", []):
        if e.get("type") not in EDGE_TYPES:
            v.append(f"edge {e.get('src')}->{e.get('dst')}: unknown type {e.get('type')!r}")
        for side in ("src", "dst"):
            if e.get(side) not in nodes:
                v.append(f"edge references unknown node {e.get(side)!r}")

    # 1 mechanism without intervention
    for i, n in nodes.items():
        if n["type"] == "mechanism" and not any(nodes.get(e["dst"], {}).get("type") == "intervention" for e in out[i]):
            v.append(f"mechanism {i} has no intervention: nothing acts on the world")

    # 2 reported variable without implementation
    for i, n in nodes.items():
        if n["type"] in ("mechanism", "intervention", "treatment", "control"):
            if not n.get("implementation"):
                v.append(f"{n['type']} {i} has no implementation path")
            elif not any(e["type"] == "implemented_causal_path" for e in out[i]):
                v.append(f"{n['type']} {i} declares an implementation but no implemented causal path")

    # 3 output without measurement path
    for i, n in nodes.items():
        if n["type"] == "primary_outcome" and not any(e["type"] == "measured_relation" for e in inn[i]):
            v.append(f"primary outcome {i} has no measured relation feeding it")

    # 4 control that still retains the removed capability
    for i, n in nodes.items():
        if n["type"] == "control":
            if not n.get("removes"):
                v.append(f"control {i} does not declare what it removes")
            ev = evidence.get(i, {})
            if ev and ev.get("all_pass") is False:
                failed = [k for k, ok in ev.items() if ok is False and k != "all_pass"]
                v.append(f"control {i} still retains {n.get('removes')!r}: failed {failed}")

    # 5 future information entering a decision time mechanism
    for e in graph.get("edges", []):
        s, d = nodes.get(e.get("src"), {}), nodes.get(e.get("dst"), {})
        if s.get("time") == "future" and d.get("type") in ("mechanism", "intervention", "treatment") and e.get("type") != "forbidden_information_path":
            v.append(f"future information {e['src']} reaches decision time node {e['dst']}")
    for e in graph.get("edges", []):
        if e.get("type") == "forbidden_information_path" and e.get("realized"):
            v.append(f"forbidden information path {e['src']}->{e['dst']} is realized")

    # 6 analytic quantity reported as measured
    for e in graph.get("edges", []):
        if e.get("type") == "measured_relation" and e.get("actually") in (
            "analytic",
            "structurally_guaranteed",
        ):
            v.append(f"edge {e['src']}->{e['dst']} is reported as measured but is {e['actually']} by construction")

    # 7 hidden confounder ignored without declaration
    for i, n in nodes.items():
        if n["type"] == "confounder" and not n.get("declared"):
            v.append(f"confounder {i} is present but not declared")
        if n["type"] == "hidden_information" and not n.get("declared"):
            v.append(f"hidden information {i} is present but not declared")

    # 8 claim broader than the measured path
    measured = {e["dst"] for e in graph.get("edges", []) if e.get("type") == "measured_relation"}
    for i, n in nodes.items():
        if n["type"] in ("claim", "verdict"):
            need = set(n.get("requires") or [])
            if not need:
                v.append(f"{n['type']} {i} requires nothing, so it cannot be checked against evidence")
            missing = need - measured
            if missing:
                v.append(f"{n['type']} {i} is broader than the measured path, unmeasured: {sorted(missing)}")
    return v


def summarize(graph: dict) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "by_node_type": {t: sum(1 for n in nodes if n.get("type") == t) for t in NODE_TYPES},
        "by_edge_type": {t: sum(1 for e in edges if e.get("type") == t) for t in EDGE_TYPES},
    }
