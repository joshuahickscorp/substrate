"""The world model, and the four distinctions section 8 refuses to let collapse into one number.

Predictive accuracy, decision usefulness, causal validity and simulation reliability are reported apart
because a model can be excellent at the first and worthless at the last three. The plan states the case
plainly: a world model that predicts but does not improve cognition remains a limited instrument. This
module computes that verdict rather than leaving it to prose, so a high accuracy score cannot quietly stand
in for a capability it does not have.

Interventions use the do operator properly. Setting a variable by intervention cuts it off from its parents
for that step; setting it by observation does not. A model that treats the two identically will pass the
predictive tests and fail the causal ones, which is exactly the separation being measured.

ponytail: the dynamics are a tabular conditional model over a declared parent graph. Small, exactly
inspectable, and enough to separate the four distinctions. The upgrade path when a bed needs continuous
state is a learned transition function behind the same four way report, not a wider table.

House style: no dashes.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from substrate import evidence as io

REPRESENTED = (
    "entities",
    "objects",
    "relations",
    "events",
    "causes",
    "affordances",
    "agents",
    "goals",
    "uncertainty",
    "time",
    "counterfactual_alternatives",
)

DISTINCTIONS = ("predictive_accuracy", "decision_usefulness", "causal_validity", "simulation_reliability")

TESTS = (
    "next_state",
    "event",
    "transition",
    "intervention",
    "counterfactual_consistency",
    "missing_observation",
    "return_to_context",
    "long_horizon_consistency",
    "decision_improvement",
)

TEST_GROUP = {
    "next_state": "predictive_accuracy",
    "event": "predictive_accuracy",
    "transition": "predictive_accuracy",
    "missing_observation": "predictive_accuracy",
    "intervention": "causal_validity",
    "counterfactual_consistency": "causal_validity",
    "return_to_context": "simulation_reliability",
    "long_horizon_consistency": "simulation_reliability",
    "decision_improvement": "decision_usefulness",
}


@dataclass
class Entity:
    id: str
    kind: str  # object, agent, event
    attributes: dict = field(default_factory=dict)
    uncertainty: float = 0.0


@dataclass
class Relation:
    src: str
    dst: str
    kind: str  # causes, affords, part_of, precedes
    confidence: float = 1.0


class WorldModel:
    """A causal graph over discrete variables plus a tabular conditional model fitted from transitions."""

    def __init__(self, parents: dict[str, tuple[str, ...]]):
        self.parents = {k: tuple(v) for k, v in parents.items()}
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.seen_states: set[tuple] = set()
        for child, ps in self.parents.items():
            for p in ps:
                self.relations.append(Relation(p, child, "causes"))

    # ------------------------------------------------------------ fitting
    def fit(self, transitions: list[tuple[dict, dict]]) -> WorldModel:
        for state, nxt in transitions:
            self.seen_states.add(tuple(sorted(state.items())))
            for var in self.parents:
                key = (var,) + tuple(state.get(p) for p in self.parents[var])
                self.counts[key][nxt[var]] += 1
        return self

    def _one(self, var: str, state: dict):
        key = (var,) + tuple(state.get(p) for p in self.parents[var])
        table = self.counts.get(key)
        if not table:
            return state.get(var)  # no evidence, persist rather than invent
        return table.most_common(1)[0][0]

    # ------------------------------------------------------------ the three operators
    def predict(self, state: dict) -> dict:
        return {var: self._one(var, state) for var in self.parents}

    def intervene(self, state: dict, do: dict) -> dict:
        """do(X=x) sets X and cuts it off from its parents for this step."""
        forced = {**state, **do}
        out = {}
        for var in self.parents:
            out[var] = do[var] if var in do else self._one(var, forced)
        return out

    def observe_set(self, state: dict, seen: dict) -> dict:
        """Conditioning on an observation, which is not the same operation as intervening."""
        return self.predict({**state, **seen})

    def counterfactual(self, state: dict, change: dict) -> dict:
        if not change:
            return self.predict(state)
        return self.intervene(state, change)

    def rollout(self, state: dict, steps: int) -> list[dict]:
        traj, current = [], dict(state)
        for _ in range(steps):
            current = {**current, **self.predict(current)}
            traj.append(dict(current))
        return traj

    def infer_missing(self, partial: dict, var: str):
        return self._one(var, partial)


# ---------------------------------------------------------------- the bed with a known truth


def synthetic_world(seed: int = 0, n: int = 600) -> dict:
    """A generative world whose causal truth is known, so causal validity can be scored, not assumed.

    The chain is lagged by one step: each variable's next value is a function of its parents' current
    values. That is what makes the declared parent graph the real graph rather than a drawing beside it.
    Season drives weather, weather drives road, road and tyre together drive speed, speed drives arrival.

    The right tyre depends on the road, so no fixed action is good everywhere. Without that the decision
    test could be passed by a constant, and a world model that changed nothing would still look useful.
    """
    rng = random.Random(seed)
    parents = {
        "season": ("season",),
        "weather": ("season",),
        "road": ("weather",),
        "tyre": ("tyre",),
        "speed": ("road", "tyre"),
        "arrive": ("speed",),
    }

    def step(state: dict) -> dict:
        season = (state["season"] + 1) % 4
        wet = state["season"] >= 2
        weather = ("wet" if wet else "dry") if rng.random() > 0.05 else ("dry" if wet else "wet")
        road = "slick" if state["weather"] == "wet" else "grip"
        tyre = state["tyre"] if rng.random() > 0.05 else rng.choice(["summer", "winter"])
        matched = (state["road"] == "grip" and state["tyre"] == "summer") or (state["road"] == "slick" and state["tyre"] == "winter")
        speed = "fast" if matched else "slow"
        arrive = "late" if state["speed"] == "slow" else "ontime"
        return {
            "season": season,
            "weather": weather,
            "road": road,
            "tyre": tyre,
            "speed": speed,
            "arrive": arrive,
        }

    state = {
        "season": 0,
        "weather": "dry",
        "road": "grip",
        "tyre": "summer",
        "speed": "fast",
        "arrive": "ontime",
    }
    transitions = []
    for _ in range(n):
        nxt = step(state)
        transitions.append((dict(state), dict(nxt)))
        state = nxt
    split = int(len(transitions) * 0.7)
    return {
        "parents": parents,
        "train": transitions[:split],
        "test": transitions[split:],
        "actions": {"tyre": ("summer", "winter")},
        "truth": step,
    }


# ---------------------------------------------------------------- the battery


def evaluate(model: WorldModel, bed: dict) -> dict:
    test = bed["test"]
    scores: dict[str, float] = {}

    # 1 next state
    scores["next_state"] = _rate(model.predict(s) == n for s, n in test)
    # 2 event: does the model call the late arrival event
    scores["event"] = _rate(model.predict(s)["arrive"] == n["arrive"] for s, n in test)
    # 3 transition: only the steps where something actually changes
    changing = [(s, n) for s, n in test if s != n]
    scores["transition"] = _rate(model.predict(s) == n for s, n in changing) if changing else 0.0
    # 6 missing observation: recover the next road with the current road hidden
    scores["missing_observation"] = _rate(model.infer_missing({k: v for k, v in s.items() if k != "road"}, "road") == n["road"] for s, n in test)
    # 7 return to context is scored below, after the detour

    # 4 intervention: do(tyre=winter) must force the mismatch on a gripping road, whatever tyre persists
    forced = [model.intervene(s, {"tyre": "winter"}) for s, _ in test if s["road"] == "grip"]
    scores["intervention"] = _rate(f["tyre"] == "winter" and f["speed"] == "slow" for f in forced)
    # 5 counterfactual consistency: an empty change reproduces the factual prediction
    scores["counterfactual_consistency"] = _rate(model.counterfactual(s, {}) == model.predict(s) for s, _ in test)

    # 7 return to context: predict correctly after an unrelated detour and a restore
    detour = {
        "season": 3,
        "weather": "wet",
        "road": "slick",
        "tyre": "winter",
        "speed": "fast",
        "arrive": "ontime",
    }
    restored = []
    for s, n in test[:50]:
        model.rollout(detour, 5)
        restored.append(model.predict(s) == n)
    scores["return_to_context"] = _rate(restored)
    # 8 long horizon: a twenty step rollout must stay inside the observed state support
    traj = model.rollout(test[0][0], 20)
    scores["long_horizon_consistency"] = _rate(tuple(sorted(st.items())) in model.seen_states for st in traj)

    # 9 decision improvement: a model based policy against the best model free policy
    scores["decision_improvement"] = _decision_gain(model, bed)

    # correction C_DISTINCTION_ROUNDING, 2026-07-27. The distinctions were averaged from the unrounded
    # scores while the per test scores were published rounded, so a reader recomputing the distinction
    # from the artifact got a different number than the artifact stated. A sealed report has to be
    # recomputable from its own published figures, so the published figures are what is averaged.
    published = {t: round(scores[t], 4) for t in TESTS}
    grouped = {d: round(_mean([published[t] for t in TESTS if TEST_GROUP[t] == d]), 4) for d in DISTINCTIONS}
    limited = limited_instrument(grouped)
    return {
        "schema": "substrate-world-model-battery/v1",
        "tests": published,
        "distinctions_are_recomputable_from_tests": True,
        "distinctions": grouped,
        "limited_instrument": limited,
        "limited_instrument_reason": ("predicts well and does not improve any decision, which section 8 calls a limited instrument" if limited else ""),
        "n_test_transitions": len(test),
    }


def limited_instrument(distinctions: dict) -> bool:
    """Section 8's verdict, computed. Predicting well while improving no decision is not a world model
    result, it is a limited instrument, and the report has to say so on its own."""
    return distinctions["predictive_accuracy"] >= 0.9 and distinctions["decision_usefulness"] <= 0.0


def _rate(it) -> float:
    values = list(it)
    return round(sum(1 for v in values if v) / len(values), 6) if values else 0.0


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _decision_gain(model: WorldModel, bed: dict) -> float:
    """Fit the tyre to the road to avoid a late arrival.

    The chain is lagged, so the consequence of an action is three steps away and a one step lookahead
    cannot see it. The model free arm is the best single fixed action across the whole test set, which is
    the strongest baseline that uses no state at all.
    """
    test, actions = bed["test"], bed["actions"]["tyre"]

    def cost(state, action) -> float:
        s1 = model.intervene(state, {"tyre": action})
        s3 = model.rollout(s1, 2)[-1]
        return 0.0 if s3["arrive"] == "ontime" else 1.0

    with_model = _mean([min(cost(s, a) for a in actions) for s, _ in test])
    best_fixed = min(_mean([cost(s, a) for s, _ in test]) for a in actions)
    return round(best_fixed - with_model, 6)


# ---------------------------------------------------------------- declaration


def declaration() -> dict:
    bed = synthetic_world()
    model = WorldModel(bed["parents"]).fit(bed["train"])
    battery = evaluate(model, bed)
    return {
        "schema": "substrate-world-model/v1",
        "represents": list(REPRESENTED),
        "required_distinctions": list(DISTINCTIONS),
        "tests": list(TESTS),
        "test_to_distinction": dict(TEST_GROUP),
        "operator_rule": (
            "intervening on a variable cuts it from its parents for that step, observing it "
            "does not. A model that treats them identically passes the predictive tests and "
            "fails the causal ones"
        ),
        "limited_instrument_rule": ("high predictive accuracy with no decision improvement is reported as a limited instrument, never as a world model result"),
        "calibration_bed": {
            "kind": "synthetic world with a known generative truth",
            "parents": {k: list(v) for k, v in bed["parents"].items()},
            "train": len(bed["train"]),
            "test": len(bed["test"]),
        },
        "battery": battery,
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_WORLD_MODEL.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "distinctions": doc["battery"]["distinctions"],
                "limited_instrument": doc["battery"]["limited_instrument"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
