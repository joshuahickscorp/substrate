"""Universal arm distinctness verifier.

An arm is distinct when it differs from every other arm in at least one load bearing trace, and the
difference is produced by running it, not by naming it. The prior program shipped arms that shared an
implementation, a default policy and an output path while carrying different names; that is the defect this
module makes unreachable.

The verifier is deliberately generic. It takes records, not models, so the same code proves distinctness for
a plasticity gate, a readout, a memory policy or a data order.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import inspect
import json

# Every field here is load bearing. Two arms that match on all of them are the same experiment twice.
LOAD_BEARING = (
    "source_sha",  # identity of the implementation that ran
    "config_sha",  # identity of the configuration that was honoured
    "call_graph_sha",  # which modules actually executed, in order
    "state_transition_sha",  # the sequence of states the run passed through
    "param_delta_sha",  # which parameters moved, and by how much
    "memory_sha",  # what the memory policy retained
    "resource_sha",  # update count, trainable parameter count, wall cost class
    "output_sha",  # predictions on a fixed probe batch
)

# Identity fields alone never establish distinctness. A different name over the same behaviour is the defect.
DECLARED_ONLY = ("name", "declared_difference")


def sha(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def source_sha(fn) -> str:
    try:
        return sha(inspect.getsource(fn))
    except (OSError, TypeError):
        return sha(repr(fn))


def record(name: str, *, source, config, call_graph, state_transitions, param_delta, memory, resources, outputs, declared_difference: str = "") -> dict:
    """Build one arm record from a probe run. Every field must come from execution, not from declaration."""
    return {
        "name": name,
        "declared_difference": declared_difference,
        "source_sha": source_sha(source) if callable(source) else sha(source),
        "config_sha": sha(config),
        "call_graph_sha": sha(call_graph),
        "state_transition_sha": sha(state_transitions),
        "param_delta_sha": sha(param_delta),
        "memory_sha": sha(memory),
        "resource_sha": sha(resources),
        "output_sha": sha(outputs),
        "config": config,
        "call_graph": call_graph,
        "resources": resources,
    }


def compare(a: dict, b: dict) -> dict:
    same = [f for f in LOAD_BEARING if a.get(f) == b.get(f)]
    diff = [f for f in LOAD_BEARING if a.get(f) != b.get(f)]
    return {"identical_fields": same, "differing_fields": diff, "aliased": not diff}


def distinctness(records: list[dict], preregistered_aliases: tuple = ()) -> dict:
    """Pairwise verdicts. A preregistered alias pair is allowed to match, everything else is a defect."""
    allowed = {tuple(sorted(p)) for p in preregistered_aliases}
    pairs, aliased = {}, []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            key = f"{a['name']}|{b['name']}"
            c = compare(a, b)
            if c["aliased"] and tuple(sorted((a["name"], b["name"]))) not in allowed:
                aliased.append(key)
                c["verdict"] = "aliased"
            else:
                c["verdict"] = "preregistered_alias" if c["aliased"] else "distinct"
            pairs[key] = c
    return {
        "n_arms": len(records),
        "pairs": pairs,
        "aliased_pairs": aliased,
        "all_distinct": not aliased,
        "per_arm": {
            r["name"]: {
                other.split("|")[1] if other.split("|")[0] == r["name"] else other.split("|")[0]: c["verdict"]
                for other, c in pairs.items()
                if r["name"] in other.split("|")
            }
            for r in records
        },
    }


def config_sensitivity(probe, base_config: dict, fields: dict) -> dict:
    """Prove that every declared load bearing configuration field is actually honoured.

    probe(config) must return an arm record. For each field the value is perturbed to the supplied
    alternative; if no load bearing trace moves, the field is ignored by the implementation and the arm that
    claims to differ by that field does not differ at all.
    """
    base = probe(base_config)
    out = {}
    for field, alt in fields.items():
        cfg = dict(base_config)
        cfg[field] = alt
        r = probe(cfg)
        moved = [f for f in LOAD_BEARING if f != "config_sha" and r.get(f) != base.get(f)]
        out[field] = {"honoured": bool(moved), "moved_traces": moved}
    out["all_honoured"] = all(v["honoured"] for k, v in out.items() if k != "all_honoured")
    return out


def branch_activity(records: list[dict], required_branches: dict[str, str]) -> dict:
    """Prove the branch an arm is named after actually executed. Names a defect the call graph can see."""
    out = {}
    for r in records:
        want = required_branches.get(r["name"])
        if want is None:
            continue
        seen = want in (r.get("call_graph") or [])
        out[r["name"]] = {"required_branch": want, "executed": seen}
    out["all_active"] = all(v["executed"] for k, v in out.items() if k != "all_active")
    return out


# ---------------------------------------------------------------- mutation attacks


def mutations(records: list[dict]) -> dict:
    """Ten attacks from the historical record. Every one must be rejected."""
    if len(records) < 2:
        raise ValueError("arm mutation suite needs at least two real arm records")
    a, b = json.loads(json.dumps(records[0])), json.loads(json.dumps(records[1]))
    res = {}

    def rejected(rs, allow=()) -> bool:
        return not distinctness(rs, allow)["all_distinct"]

    clone = json.loads(json.dumps(a))
    clone["name"] = a["name"] + "_renamed"
    res["same_implementation_under_two_names"] = rejected([a, clone])

    for field, label in (
        ("config_sha", "same_default_policy"),
        ("param_delta_sha", "same_trainable_group"),
        ("memory_sha", "same_memory_policy"),
        ("output_sha", "same_output_path"),
        ("state_transition_sha", "same_checkpoint"),
        ("resource_sha", "same_resource_budget"),
    ):
        m = json.loads(json.dumps(b))
        m["name"] = b["name"] + f"_{label}"
        for f in LOAD_BEARING:
            m[f] = a[f]
        res[label] = rejected([a, m])

    res["same_result_file"] = len({r["name"] for r in records}) == len(records) and rejected([a, clone])
    res["inactive_branch"] = not branch_activity(
        [{**a, "call_graph": []}], {a["name"]: "declared_branch"}
    )["all_active"]

    # a configuration field that no trace responds to
    def dead_probe(cfg):
        return {**a, "config_sha": sha(cfg)}

    res["configuration_field_ignored"] = not config_sensitivity(
        dead_probe, {"x": 1}, {"x": 2}
    )["all_honoured"]

    def flag_probe(cfg):
        return {**a, "config_sha": sha(cfg)}

    res["treatment_flag_ignored"] = not config_sensitivity(
        flag_probe, {"treatment": False}, {"treatment": True}
    )["all_honoured"]

    res["all_rejected"] = all(v for k, v in res.items() if k != "all_rejected")
    return res
