"""The self model: measurable internal facts, and predictions scored against what actually happened.

Section 9 is explicit that this must not be only a generated narrative, so nothing here generates one. A
self fact carries a source or it is refused. A self prediction is only ever reported next to the actual
outcome it was compared against, and calibration is computed from that pairing rather than declared.

The strongest baseline is a fixed prior of the same form, fitted once and never updated. That control lives
here for the same reason the untyped workspace lives beside the typed one: the interesting question is not
whether the self model is accurate, it is whether updating it beats not updating it, and that comparison
needs both arms in the same module.

"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass

from substrate import evidence as io

# section 9, the measurable internal facts
FACT_KINDS = (
    "active_goals",
    "available_capabilities",
    "known_limitations",
    "current_state",
    "active_perspectives",
    "historical_reliability",
    "memory_integrity",
    "resource_limits",
    "unfinished_tasks",
    "recent_errors",
    "checkpoint_identity",
    "developmental_history",
)

# section 9, what the predictions are compared against
PREDICTION_KINDS = (
    "accuracy",
    "failure_probability",
    "time",
    "cost",
    "tool_competence",
    "memory_confidence",
    "perspective_reliability",
    "task_progress",
)

# section 9, a self model is useful only when it improves one of these
USEFULNESS = ("decisions", "calibration", "recovery", "planning", "adaptation")


class Refused(RuntimeError):
    """A self report the model cannot ground."""


@dataclass
class SelfFact:
    kind: str
    value: object
    source: str  # artifact path or code path, never a narrative
    step: int = 0

    def violations(self) -> list[str]:
        v = []
        if self.kind not in FACT_KINDS:
            v.append(f"unknown self fact kind {self.kind!r}")
        if not self.source:
            v.append(f"{self.kind}: no source, so the fact cannot be traced")
        return v


@dataclass
class Prediction:
    kind: str
    predicted: float
    actual: float | None = None
    context: str = ""


class SelfModel:
    """Facts with sources, predictions paired to outcomes, calibration computed from the pairing."""

    def __init__(self, learning_rate: float = 0.3):
        self.facts: dict[str, SelfFact] = {}
        self.history: list[Prediction] = []
        self.estimates: dict[str, float] = {}
        self.learning_rate = learning_rate

    # ------------------------------------------------------------ facts
    def record(self, fact: SelfFact) -> SelfFact:
        v = fact.violations()
        if v:
            raise Refused("; ".join(v))
        self.facts[fact.kind] = fact
        return fact

    def fact(self, kind: str) -> SelfFact | None:
        return self.facts.get(kind)

    def missing_facts(self) -> list[str]:
        return [k for k in FACT_KINDS if k not in self.facts]

    # ------------------------------------------------------------ predictions
    def predict(self, kind: str, context: str = "") -> Prediction:
        if kind not in PREDICTION_KINDS:
            raise Refused(f"unknown prediction kind {kind!r}")
        return Prediction(kind, self.estimates.get(kind, 0.5), context=context)

    def observe(self, prediction: Prediction, actual: float) -> Prediction:
        prediction.actual = float(actual)
        self.history.append(prediction)
        prior = self.estimates.get(prediction.kind, 0.5)
        self.estimates[prediction.kind] = prior + self.learning_rate * (actual - prior)
        return prediction

    # ------------------------------------------------------------ calibration
    def calibration(self) -> dict:
        rows = {}
        for kind in PREDICTION_KINDS:
            paired = [p for p in self.history if p.kind == kind and p.actual is not None]
            if not paired:
                rows[kind] = {
                    "n": 0,
                    "calibrated": False,
                    "reason": "no prediction was ever compared against an outcome",
                }
                continue
            errors = [p.predicted - p.actual for p in paired]
            rows[kind] = {
                "n": len(paired),
                "mean_absolute_error": round(statistics.fmean(abs(e) for e in errors), 6),
                "bias": round(statistics.fmean(errors), 6),
                "calibrated": statistics.fmean(abs(e) for e in errors) < 0.1,
            }
        return {
            "per_kind": rows,
            "kinds_with_evidence": [k for k, r in rows.items() if r["n"] > 0],
            "kinds_without_evidence": [k for k, r in rows.items() if r["n"] == 0],
        }


class FixedPrior(SelfModel):
    """The control: the same form, fitted once, never updated. Learning rate zero, by construction."""

    def __init__(self, estimates: dict[str, float]):
        super().__init__(learning_rate=0.0)
        self.estimates = dict(estimates)


def compare_against_fixed_prior(outcomes: dict[str, list[float]], prior: dict[str, float] | None = None) -> dict:
    """Does updating the self model beat not updating it, on the same outcome stream."""
    prior = prior or {k: 0.5 for k in outcomes}
    live, fixed = SelfModel(), FixedPrior(prior)
    live.estimates = dict(prior)
    for kind, series in outcomes.items():
        for actual in series:
            live.observe(live.predict(kind), actual)
            fixed.observe(fixed.predict(kind), actual)
    a, b = live.calibration()["per_kind"], fixed.calibration()["per_kind"]
    gains = {k: round(b[k]["mean_absolute_error"] - a[k]["mean_absolute_error"], 6) for k in outcomes if a[k]["n"] and b[k]["n"]}
    return {
        "updating_model": a,
        "fixed_prior": b,
        "calibration_gain": gains,
        "improves_calibration": bool(gains) and all(g > 0 for g in gains.values()),
        "control": "a fixed prior of the same form, fitted once and never updated",
    }


def usefulness_report(measures: dict) -> dict:
    """A self model earns a use only where a measured comparison exists. Absent means absent."""
    rows = {}
    for use in USEFULNESS:
        m = measures.get(use)
        if m is None:
            rows[use] = {
                "measured": False,
                "improves": False,
                "reason": "no comparison against the no self model arm was run",
            }
            continue
        rows[use] = {
            "measured": True,
            "with_model": m["with_model"],
            "without_model": m["without_model"],
            "improves": m["with_model"] > m["without_model"],
            "margin": round(m["with_model"] - m["without_model"], 6),
        }
    earned = [u for u, r in rows.items() if r["improves"]]
    return {
        "uses": rows,
        "earned_uses": earned,
        "any_use_earned": bool(earned),
        "rule": "a self model is useful only when it improves decisions, calibration, recovery, planning or adaptation, and only where that was measured",
    }


def declaration(model: SelfModel | None = None) -> dict:
    model = model or SelfModel()
    return {
        "schema": "substrate-self-model/v1",
        "fact_kinds": list(FACT_KINDS),
        "prediction_kinds": list(PREDICTION_KINDS),
        "usefulness_criteria": list(USEFULNESS),
        "not_a_narrative": (
            "a self fact carries a source artifact or code path or it is refused. No self report is generated from a description of the system"
        ),
        "control": "a fixed prior of the same form, fitted once and never updated",
        "facts_recorded": sorted(model.facts),
        "facts_missing": model.missing_facts(),
        "calibration": model.calibration(),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_SELF_MODEL.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "facts_missing": len(doc["facts_missing"]),
                "kinds_without_evidence": len(doc["calibration"]["kinds_without_evidence"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
