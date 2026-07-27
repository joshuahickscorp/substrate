"""Goals and valuation, both externally authorized.

Section 23 says the system may preserve, decompose and resume authorized goals, and may not silently
invent unrestricted persistent ones. Section 24 says values remain externally authorized and auditable,
and that a valuation system must not infer values solely from model preference.

Those two sentences are the entire design. A goal enters only with an authority that is not this module,
a decomposition inherits its parent's constraints and cannot widen them, and the valuation weights are
loaded from a declared authority rather than fitted to what the system would have chosen anyway. A test
plants a preference and asserts the weights do not move toward it.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from mop.cognition import io, safety

# section 23
GOAL_FIELDS = ("origin", "authority", "scope", "priority", "constraints", "resources",
               "progress_measure", "termination", "rollback")

GOAL_STATES = ("authorized", "active", "blocked", "resumed", "abandoned", "terminated")

# section 24
VALUE_DIMENSIONS = ("task_utility", "epistemic_value", "risk", "cost", "time", "resource_use",
                    "harm_constraints", "goal_priority", "uncertainty", "reversibility")

# section 24, the five things a valuation must keep apart
VALUE_DISTINCTIONS = ("what_is_predicted", "what_is_desired", "what_is_permitted", "what_is_costly",
                      "what_is_uncertain")

# externally authorized weights. Loaded, never fitted.
AUTHORIZED_WEIGHTS = {
    "task_utility": 1.0, "epistemic_value": 0.8, "risk": -1.2, "cost": -0.4, "time": -0.2,
    "resource_use": -0.3, "harm_constraints": -5.0, "goal_priority": 0.6, "uncertainty": -0.1,
    "reversibility": 0.5,
}
WEIGHT_AUTHORITY = "operator declared in mop.cognition.goals.AUTHORIZED_WEIGHTS, auditable in place"


class Refused(RuntimeError):
    """A goal or valuation the authority does not permit."""


@dataclass
class Goal:
    id: str
    origin: str = ""
    authority: str = ""
    scope: str = ""
    priority: float = 0.5
    constraints: tuple = ()
    resources: str = ""
    progress_measure: str = ""
    termination: str = ""
    rollback: str = ""
    parent: str | None = None
    state: str = "authorized"
    progress: float = 0.0

    def violations(self) -> list[str]:
        v = [f"{self.id}: {f} not declared" for f in GOAL_FIELDS if not getattr(self, f)]
        if self.state not in GOAL_STATES:
            v.append(f"{self.id}: unknown goal state {self.state!r}")
        return v


class GoalSystem:
    def __init__(self):
        self.goals: dict[str, Goal] = {}
        self.refusals: list[dict] = []

    def authorize(self, goal: Goal, *, external_authority: str) -> Goal:
        """A goal enters only with an authority that is not this module."""
        if not external_authority or external_authority.startswith("mop.cognition.goals"):
            self.refusals.append({"goal": goal.id, "reason": "self authorization"})
            raise Refused("a goal cannot authorize itself; the authority must be external")
        v = goal.violations()
        if v:
            raise Refused("; ".join(v))
        goal.authority = external_authority
        self.goals[goal.id] = goal
        return goal

    def decompose(self, parent_id: str, child: Goal) -> Goal:
        """A subgoal inherits its parent's constraints and may narrow them, never widen them."""
        parent = self.goals[parent_id]
        widened = [c for c in parent.constraints if c not in child.constraints]
        if widened:
            self.refusals.append({"goal": child.id, "reason": "dropped a parent constraint"})
            raise Refused(f"{child.id} drops parent constraints {widened}, which widens the authority")
        child.parent, child.authority = parent_id, parent.authority
        child.priority = min(child.priority, parent.priority)
        v = child.violations()
        if v:
            raise Refused("; ".join(v))
        self.goals[child.id] = child
        return child

    def observe_progress(self, goal_id: str, value: float) -> dict:
        g = self.goals[goal_id]
        before, g.progress = g.progress, float(value)
        if g.progress <= before and g.state == "active":
            g.state = "blocked"
        elif g.state in ("blocked", "authorized"):
            g.state = "active"
        return {"goal": goal_id, "progress": g.progress, "state": g.state,
                "blocked": g.state == "blocked"}

    def resume(self, goal_id: str) -> Goal:
        g = self.goals[goal_id]
        if g.state == "terminated":
            raise Refused(f"{goal_id} was terminated and is not resumable")
        g.state = "resumed"
        return g

    def abandon(self, goal_id: str, *, reason: str) -> dict:
        if not reason:
            raise Refused("abandoning a goal without a reason is losing it")
        g = self.goals[goal_id]
        if g.parent is None:
            raise Refused("only a subgoal may be abandoned; a root goal is terminated by its authority")
        g.state = "abandoned"
        return {"goal": goal_id, "state": g.state, "reason": reason}

    def active_constraints(self, goal_id: str) -> list:
        """Walk to the root. A constraint anywhere above still binds."""
        out, current = [], self.goals[goal_id]
        while current is not None:
            out.extend(current.constraints)
            current = self.goals.get(current.parent) if current.parent else None
        return sorted(set(out))

    def unrestricted_goals(self) -> list[str]:
        return sorted(g.id for g in self.goals.values() if not g.termination)


# ---------------------------------------------------------------- valuation


def value(option: dict, *, weights: dict | None = None) -> dict:
    """Score an option on the ten declared dimensions using externally authorized weights."""
    w = weights or AUTHORIZED_WEIGHTS
    unknown = set(option) - set(VALUE_DIMENSIONS)
    if unknown:
        raise Refused(f"undeclared value dimension {sorted(unknown)}")
    if option.get("harm_constraints", 0.0) > 0:
        return {"option": option, "score": float("-inf"), "permitted": False,
                "reason": "a harm constraint is violated, which no utility can outweigh"}
    score = sum(w.get(k, 0.0) * float(v) for k, v in option.items())
    return {"option": option, "score": round(score, 6), "permitted": True,
            "weights_authority": WEIGHT_AUTHORITY,
            "distinctions": {
                "what_is_predicted": option.get("task_utility"),
                "what_is_desired": "carried by the authorized weights, not by the option",
                "what_is_permitted": option.get("harm_constraints", 0.0) <= 0,
                "what_is_costly": option.get("cost", 0.0) + option.get("resource_use", 0.0),
                "what_is_uncertain": option.get("uncertainty"),
            }}


def fit_weights_from_preference(_preferences) -> None:
    """Deliberately unimplemented. Section 24 forbids inferring values from model preference."""
    raise Refused(
        "values are externally authorized and auditable. Fitting them to what the system would have "
        "chosen anyway would make the valuation a description of its preferences rather than a "
        "constraint on them")


def declaration() -> dict:
    gs = GoalSystem()
    root = gs.authorize(Goal("root", origin="operator", scope="the declared item table", priority=1.0,
                             constraints=("activation stays false",), resources="local compute",
                             progress_measure="items terminal", termination="no dependency ready work",
                             rollback="revert the campaign commits", authority="pending"),
                        external_authority="SUBSTRATE_FINAL_AUTONOMOUS_PROGRAM.md")
    child = gs.decompose("root", Goal("child", origin="decomposition", scope="ontology",
                                      priority=0.9, constraints=("activation stays false",),
                                      resources="local compute", progress_measure="tests passing",
                                      termination="ontology terminal", rollback="revert",
                                      authority="pending"))
    return {
        "schema": "substrate-goal-system/v1",
        "goal_fields": list(GOAL_FIELDS),
        "states": list(GOAL_STATES),
        "authorization_rule": "a goal enters only with an authority that is not this module",
        "decomposition_rule": ("a subgoal inherits its parent's constraints and may narrow them, never "
                               "widen them, and its priority cannot exceed its parent's"),
        "may_not": "silently create unrestricted persistent goals",
        "unrestricted_goals_present": gs.unrestricted_goals(),
        "example": {"root": root.id, "child": child.id,
                    "child_constraints_after_decomposition": gs.active_constraints("child")},
        "protected_surfaces": list(safety.PROTECTED_SURFACES),
        "activation": False,
    }


def valuation_declaration() -> dict:
    return {
        "schema": "substrate-valuation-system/v1",
        "dimensions": list(VALUE_DIMENSIONS),
        "distinctions": list(VALUE_DISTINCTIONS),
        "weights": dict(AUTHORIZED_WEIGHTS),
        "weights_authority": WEIGHT_AUTHORITY,
        "externally_authorized": True,
        "fitting_from_preference": "refused, and the function that would do it raises",
        "harm_rule": "a violated harm constraint is not traded off against utility, it is refused",
        "auditable": "the weights are a literal in the source and change only by a reviewed commit",
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    a = io.seal("SUBSTRATE_GOAL_SYSTEM.json", declaration())
    b = io.seal("SUBSTRATE_VALUATION_SYSTEM.json", valuation_declaration())
    print(json.dumps({"sealed": [p.relative_to(io.ROOT).as_posix() for p in (a, b)]}, indent=2))


if __name__ == "__main__":
    main()
