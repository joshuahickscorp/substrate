"""Developmental divergence: two identical entities, two different verified histories.

Section 29 operationalizes individuality without claiming it. Two Substrate instances start byte identical,
receive different slices of the same real session, and are then measured on nine dimensions. The
interesting question is not whether they differ, which is trivial, but whether the differences are useful
and whether they transfer to material neither instance has seen.

The control matters here more than the measurement. Two instances given the same history must not diverge,
or the divergence measured on different histories is noise wearing a developmental costume. That control
runs first and its result gates the rest.

"""

from __future__ import annotations

import json
import statistics
import sys

from substrate import evidence as io
from substrate import runtime as R
from substrate import sessions

# section 29, the nine dimensions
DIMENSIONS = (
    "memory",
    "procedures",
    "world_models",
    "self_models",
    "perspective_reliability",
    "specialization",
    "transfer",
    "robustness",
    "goal_progress",
)


class Refused(RuntimeError):
    """A divergence claim the design will not support."""


def _histories(session: dict) -> tuple[list, list, list]:
    """Two disjoint slices of a real session, plus a held out slice neither instance sees."""
    events = [e for e in session["events"] if e["kind"] in ("shard_completed", "shard_quarantined", "failure_hold")]
    if len(events) < 60:
        raise Refused(f"only {len(events)} usable events, too few to split three ways")
    third = len(events) // 3
    return events[:third], events[third : 2 * third], events[2 * third :]


def _observation(event: dict) -> dict:
    """Turn a session event into something the runtime can perceive. Nothing is invented."""
    ok = event["kind"] == "shard_completed"
    return {
        "label": "ok" if ok else "bad",
        "label_confidence": 0.9 if ok else 0.6,
        "stage": event.get("stage", ""),
        "kind": event["kind"],
    }


def _live(history: list) -> R.Substrate:
    entity = R.Substrate()
    for event in history:
        obs = _observation(event)
        entity.step(obs, outcome=obs["label"], goal=["process the session"])
    return entity


def _measure(entity: R.Substrate, held_out: list) -> dict:
    correct = 0
    for event in held_out:
        obs = _observation(event)
        entity.step(obs, outcome=obs["label"])
        decided = entity.ws.read("decision", "reader") or {}
        correct += int(decided.get("value") == obs["label"])
    return {
        "memory": len(entity.episodes.store),
        "procedures": len(entity.procedures.store),
        "world_models": len(entity.semantic.store),
        "self_models": len(entity.self_model.history),
        "perspective_reliability": {k: round(v, 6) for k, v in sorted(entity.reliability.items())},
        "specialization": round(statistics.pstdev(list(entity.reliability.values())) if len(entity.reliability) > 1 else 0.0, 6),
        "transfer": round(correct / max(len(held_out), 1), 6),
        "robustness": round(1.0 - len(entity.ws.refusals) / max(entity.step_index, 1), 6),
        "goal_progress": entity.step_index,
    }


def _distance(a: dict, b: dict) -> dict:
    out = {}
    for d in DIMENSIONS:
        x, y = a[d], b[d]
        if isinstance(x, dict):
            keys = sorted(set(x) | set(y))
            out[d] = round(statistics.fmean([abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in keys]), 6)
        else:
            out[d] = round(abs(float(x) - float(y)), 6)
    return out


def run() -> dict:
    session = sessions.build()
    left, right, held_out = _histories(session)

    # the control first: identical histories must not diverge, or nothing below means anything
    same_a, same_b = _live(left), _live(list(left))
    control = _distance(_measure(same_a, held_out), _measure(same_b, held_out))
    control_clean = all(v == 0.0 for v in control.values())

    a, b = _live(left), _live(right)
    measured_a, measured_b = _measure(a, held_out), _measure(b, held_out)
    divergence = _distance(measured_a, measured_b)

    useful = measured_a["transfer"] != measured_b["transfer"]
    better = max(measured_a["transfer"], measured_b["transfer"])
    worse = min(measured_a["transfer"], measured_b["transfer"])

    return {
        "schema": "substrate-developmental-history/v1",
        "source_session": session["session"],
        "history_lengths": {"a": len(left), "b": len(right), "held_out": len(held_out)},
        "dimensions": list(DIMENSIONS),
        "control_same_history": control,
        "control_shows_no_divergence": control_clean,
        "control_rule": ("two instances given the same history must not diverge, or the divergence below is noise wearing a developmental costume"),
        "instance_a": measured_a,
        "instance_b": measured_b,
        "divergence": divergence,
        "diverged_dimensions": sorted(d for d, v in divergence.items() if v > 0),
        "useful_difference": useful,
        "transfer_gap": round(better - worse, 6),
        "verdict": (
            "invalid_control"
            if not control_clean
            else "no_divergence"
            if not any(v > 0 for v in divergence.values())
            else "divergence_with_transfer_difference"
            if useful
            else "divergence_without_transfer_value"
        ),
        "reading": (
            "the control is clean, so the instrument works. Two different verified histories produced "
            "identical internal state on every measured dimension, which means this entity does not "
            "develop differently from these histories. That is a negative result about the entity, and "
            "possibly about the observation encoding, and it is not evidence of individuality"
            if control_clean and not any(v > 0 for v in divergence.values())
            else "divergence was measured; whether it carries transfer value is reported separately"
        ),
        "claim_boundary": (
            "this measures developmental individuality as differences in owned state that "
            "persist. It says nothing about subjective identity and is not evidence of it"
        ),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    path = io.seal("SUBSTRATE_DEVELOPMENTAL_HISTORY.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "control_clean": doc["control_shows_no_divergence"],
                "diverged": doc["diverged_dimensions"],
                "transfer_gap": doc["transfer_gap"],
                "verdict": doc["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
