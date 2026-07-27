"""A world model bed with state dependent decisions, built from a session nobody designed for it.

Section 12 requires the world model to operate inside the decision loop and says prediction quality alone
is insufficient. Both halves need a bed where the right action actually depends on the state, because in a
bed where one fixed action is best everywhere a world model can predict perfectly and still be worth
nothing.

The bed is the supervisor's own scheduling history. Over days it ran under three declared resource classes
with a worker cap it changed as pressure changed, and each observation is followed by an observed change
in remaining work. That gives state, action and outcome from a record that was written to run a factorial,
not to test a cognitive architecture.

The bed is only admissible if no single cap is best in every resource class. That is checked rather than
assumed, and if it fails the bed is refused with the number that refused it.

House style: no dashes.
"""

from __future__ import annotations

import ast
import json
import statistics
import sys
from collections import defaultdict

from mop.cognition import io, sessions

CAP_BUCKETS = ((0, 8), (9, 16), (17, 24), (25, 10 ** 6))


class Refused(RuntimeError):
    """A bed the admission rules will not accept."""


def _bucket(cap: int) -> str:
    for lo, hi in CAP_BUCKETS:
        if lo <= cap <= hi:
            return f"{lo}_{hi}" if hi < 10 ** 6 else f"{lo}_plus"
    return "unknown"


def _remaining_total(text: str) -> int | None:
    try:
        return sum(int(v) for v in ast.literal_eval(text).values())
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return None


def transitions(session: dict | None = None) -> list[dict]:
    """State, action and the progress that followed. Progress is measured, never assumed."""
    session = session or sessions.build()
    obs = [e for e in session["events"] if e["kind"] == "scheduling_observation"]
    rows = []
    for a, b in zip(obs, obs[1:]):
        left = _remaining_total(a["detail"]["remaining"])
        right = _remaining_total(b["detail"]["remaining"])
        if left is None or right is None:
            continue
        rows.append({
            "resource_class": a["detail"]["resource_class"],
            "workers": a["detail"]["workers"],
            "cap": a["detail"]["cap"],
            "cap_bucket": _bucket(a["detail"]["cap"]),
            "remaining_before": left,
            "progress": left - right,  # positive means work was completed
            "source_path": a["source_path"],
        })
    return rows


def build() -> dict:
    rows = transitions()
    if len(rows) < 30:
        raise Refused(f"only {len(rows)} usable transitions, which cannot support any decision claim")

    by_state_action: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_state_action[(r["resource_class"], r["cap_bucket"])].append(r["progress"])
    table = {f"{k[0]}|{k[1]}": {"n": len(v), "mean_progress": round(statistics.fmean(v), 6)}
             for k, v in by_state_action.items()}

    classes = sorted({r["resource_class"] for r in rows})
    best_by_class = {}
    for klass in classes:
        options = {b: statistics.fmean(v) for (c, b), v in by_state_action.items() if c == klass}
        if options:
            best_by_class[klass] = max(options, key=options.get)
    distinct_best = len(set(best_by_class.values()))

    # the admissibility question: does the best action depend on the state at all
    state_dependent = distinct_best > 1
    doc = {
        "schema": "substrate-world-model-bed/v1",
        "source": "the temporal supervisor's scheduling history, via the real session authority",
        "n_transitions": len(rows),
        "resource_classes": classes,
        "cap_buckets": [f"{lo}_{hi}" if hi < 10 ** 6 else f"{lo}_plus" for lo, hi in CAP_BUCKETS],
        "state_action_table": table,
        "best_action_by_class": best_by_class,
        "distinct_best_actions": distinct_best,
        "state_dependent_decision": state_dependent,
        "admissible": state_dependent,
        "refusal_reason": "" if state_dependent else (
            f"one cap bucket, {next(iter(set(best_by_class.values())), None)}, is best in every resource "
            "class, so no world model can earn a decision improvement on this bed and prediction quality "
            "would be the only thing left to report"),
        "why_this_bed": ("the record was written to run a factorial. Its states, actions and outcomes "
                         "were not chosen to make a world model look useful"),
        "activation": False,
    }
    return doc


def integrate(bed: dict | None = None) -> dict:
    """Put the model inside the loop: does a prediction change which action is selected."""
    bed = bed or build()
    rows = transitions()
    table = defaultdict(list)
    for r in rows:
        table[(r["resource_class"], r["cap_bucket"])].append(r["progress"])
    means = {k: statistics.fmean(v) for k, v in table.items()}

    def model_choice(klass: str) -> str | None:
        options = {b: m for (c, b), m in means.items() if c == klass}
        return max(options, key=options.get) if options else None

    fixed_options = defaultdict(list)
    for (c, b), v in table.items():
        fixed_options[b].extend(v)
    best_fixed = max(fixed_options, key=lambda b: statistics.fmean(fixed_options[b]))

    changed, total = 0, 0
    for r in rows:
        chosen = model_choice(r["resource_class"])
        total += 1
        if chosen != best_fixed:
            changed += 1
    model_value = statistics.fmean(
        [means.get((r["resource_class"], model_choice(r["resource_class"])), 0.0) for r in rows])
    fixed_value = statistics.fmean([statistics.fmean(fixed_options[best_fixed])] * len(rows))
    gain = model_value - fixed_value

    return {
        "schema": "substrate-world-model-battery/v1",
        "bed": {"admissible": bed["admissible"], "n_transitions": bed["n_transitions"],
                "state_dependent_decision": bed["state_dependent_decision"]},
        "inside_the_loop": True,
        "best_fixed_action": best_fixed,
        "model_action_by_class": {k: model_choice(k) for k in bed["resource_classes"]},
        "decisions_changed_by_the_model": changed,
        "decisions_total": total,
        "changed_fraction": round(changed / max(total, 1), 6),
        "expected_progress_with_model": round(model_value, 6),
        "expected_progress_best_fixed": round(fixed_value, 6),
        "decision_gain": round(gain, 6),
        "prediction_alone_is_insufficient": True,
        "verdict": ("decision_value" if gain > 0.05 and changed else
                    "limited_instrument" if changed == 0 else "no_measurable_decision_gain"),
        "reading": ("a world model earns a positive here only by changing an action and improving the "
                    "outcome. Changing nothing, or changing something and gaining nothing, is reported "
                    "as a limited instrument"),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "build"
    if command == "build":
        doc = build()
        path = io.seal("SUBSTRATE_WORLD_MODEL_BED.json", doc)
        print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                          "transitions": doc["n_transitions"], "admissible": doc["admissible"],
                          "best_action_by_class": doc["best_action_by_class"],
                          "refusal_reason": doc["refusal_reason"]}, indent=2))
    elif command == "integrate":
        doc = integrate()
        path = io.seal("SUBSTRATE_WORLD_MODEL_BATTERY.json", doc)
        print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                          "verdict": doc["verdict"], "decision_gain": doc["decision_gain"],
                          "decisions_changed": doc["decisions_changed_by_the_model"]}, indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
