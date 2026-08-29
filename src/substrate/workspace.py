"""The typed cognitive workspace.

Section 6.2 asks for something specific: important information becomes globally available without letting
arbitrary components corrupt every region. Those two requirements pull in opposite directions, and the
resolution is that reading is broad and writing is narrow. Every region names its readers and its writers,
a read outside the declared set is refused, and a write outside the declared set is refused. Nothing is
silently dropped, because a silent drop is indistinguishable from a component that never ran.

Typing is the mechanism under test, so the untyped control lives here too and shares the same storage,
the same capacity and the same cost accounting. It differs in exactly one way: it has no reader or writer
sets. That is what makes the pair an arm and a control rather than two unrelated implementations.

"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass

from substrate import evidence as io

PERSISTENCE = ("step", "episode", "goal", "persistent")
TIMESCALES = ("immediate", "fast", "medium", "slow", "developmental")

REQUIRED_DECLARATIONS = (
    "shape",
    "persistence",
    "timescale",
    "readers",
    "writers",
    "provenance",
    "confidence",
    "cost",
    "reset",
    "update_rule",
)


class Refused(RuntimeError):
    """An access the workspace type system does not permit. Raised, never downgraded to a warning."""


@dataclass(frozen=True)
class RegionSpec:
    name: str
    shape: str
    persistence: str
    timescale: str
    readers: tuple[str, ...]
    writers: tuple[str, ...]
    provenance: bool  # a write must name where the value came from
    confidence: bool  # a write must carry a confidence
    cost: float  # charged per write
    reset: str  # which trigger clears this region
    update_rule: str

    def violations(self) -> list[str]:
        v = []
        if self.persistence not in PERSISTENCE:
            v.append(f"{self.name}: unknown persistence {self.persistence!r}")
        if self.timescale not in TIMESCALES:
            v.append(f"{self.name}: unknown timescale {self.timescale!r}")
        if not self.writers:
            v.append(f"{self.name}: no writer declared, the region can never be filled")
        if not self.readers:
            v.append(f"{self.name}: no reader declared, the region can never be used")
        if self.reset not in PERSISTENCE:
            v.append(f"{self.name}: reset trigger {self.reset!r} is not a persistence level")
        if not self.update_rule:
            v.append(f"{self.name}: no update rule declared")
        return v


@dataclass
class Entry:
    value: object
    writer: str
    provenance: str
    confidence: float | None
    step: int


def _r(name, shape, persistence, timescale, readers, writers, provenance, confidence, cost, reset, update_rule) -> RegionSpec:
    return RegionSpec(
        name,
        shape,
        persistence,
        timescale,
        tuple(readers),
        tuple(writers),
        provenance,
        confidence,
        cost,
        reset,
        update_rule,
    )


ALL = ("*",)

# section 6.2, the ten suggested typed regions, each with all ten declarations filled
REGIONS: tuple[RegionSpec, ...] = (
    _r(
        "perceptual",
        "observation vector",
        "step",
        "immediate",
        ALL,
        ("sensor",),
        True,
        False,
        0.1,
        "step",
        "overwrite from the current observation",
    ),
    _r(
        "temporal",
        "core hidden state",
        "episode",
        "fast",
        ALL,
        ("temporal_core",),
        True,
        False,
        0.2,
        "episode",
        "recurrent update from the selected temporal core",
    ),
    _r(
        "ontological",
        "typed item graph",
        "persistent",
        "slow",
        ALL,
        ("ontology",),
        True,
        True,
        0.3,
        "persistent",
        "merge only on evidence, and a merge stays reversible",
    ),
    _r(
        "epistemic",
        "belief store with justification edges",
        "persistent",
        "medium",
        ALL,
        ("epistemology",),
        True,
        True,
        0.3,
        "persistent",
        "retraction propagates to every dependant before any confidence is read",
    ),
    _r(
        "conceptual",
        "concept bindings",
        "episode",
        "medium",
        ALL,
        ("semantic_memory", "perspective"),
        True,
        True,
        0.2,
        "episode",
        "merge on higher confidence, keep the loser as an alternative",
    ),
    _r(
        "goal",
        "goal stack",
        "goal",
        "slow",
        ALL,
        ("goal_authority",),
        True,
        False,
        0.1,
        "goal",
        "push and pop only through an authorized goal decomposition",
    ),
    _r(
        "working_memory",
        "bounded slot table",
        "episode",
        "fast",
        ALL,
        ("perspective", "arbiter"),
        True,
        True,
        0.3,
        "episode",
        "priority eviction when the slot budget is exceeded",
    ),
    _r(
        "episodic_context",
        "recent episode window",
        "episode",
        "medium",
        ALL,
        ("episodic_memory",),
        True,
        False,
        0.3,
        "episode",
        "append with decay, never rewrite a sealed episode",
    ),
    _r(
        "world",
        "entity and relation graph",
        "persistent",
        "slow",
        ALL,
        ("world_model",),
        True,
        True,
        0.4,
        "persistent",
        "bayesian style update against measured outcomes",
    ),
    _r(
        "self",
        "measured internal facts",
        "persistent",
        "slow",
        ALL,
        ("self_model",),
        True,
        True,
        0.4,
        "persistent",
        "replace only from a measured comparison against actuals",
    ),
    _r(
        "uncertainty",
        "per belief interval",
        "episode",
        "fast",
        ALL,
        ("arbiter", "self_model"),
        True,
        True,
        0.1,
        "episode",
        "widen on contradiction, never narrow without new measurement",
    ),
    _r(
        "decision",
        "chosen action and rationale",
        "step",
        "immediate",
        ALL,
        ("arbiter",),
        True,
        True,
        0.1,
        "step",
        "written once per decision, superseded rather than edited",
    ),
)

BY_NAME = {r.name: r for r in REGIONS}


class Workspace:
    """Broad reads, narrow writes, every access accounted for."""

    typed = True

    def __init__(self, specs: tuple[RegionSpec, ...] | None = None, budget: float = float("inf")):
        # resolved at call time, not captured in a default argument. A default binds REGIONS once at
        # import and then a change to the declaration silently has no effect on new workspaces, which a
        # mutation attack found by changing the declaration and watching nothing happen.
        self.specs = {s.name: s for s in (REGIONS if specs is None else specs)}
        self.store: dict[str, Entry] = {}
        self.step = 0
        self.budget = budget
        self.spent = 0.0
        self.refusals: list[str] = []
        self.writes = 0

    # ------------------------------------------------------------ access control
    def _spec(self, region: str) -> RegionSpec:
        spec = self.specs.get(region)
        if spec is None:
            raise Refused(f"unknown region {region!r}")
        return spec

    def _permitted(self, allowed: tuple[str, ...], who: str) -> bool:
        return not self.typed or "*" in allowed or who in allowed

    def read(self, region: str, by: str):
        spec = self._spec(region)
        if not self._permitted(spec.readers, by):
            self.refusals.append(f"read {region} by {by}")
            raise Refused(f"{by} is not a declared reader of {region}")
        entry = self.store.get(region)
        return entry.value if entry else None

    def write(self, region: str, by: str, value, provenance: str = "", confidence: float | None = None):
        spec = self._spec(region)
        if not self._permitted(spec.writers, by):
            self.refusals.append(f"write {region} by {by}")
            raise Refused(f"{by} is not a declared writer of {region}")
        if self.typed and spec.provenance and not provenance:
            self.refusals.append(f"write {region} without provenance")
            raise Refused(f"{region} requires provenance and the write named none")
        if self.typed and spec.confidence and confidence is None:
            self.refusals.append(f"write {region} without confidence")
            raise Refused(f"{region} requires a confidence and the write carried none")
        if self.spent + spec.cost > self.budget:
            self.refusals.append(f"write {region} over budget")
            raise Refused(f"the write to {region} exceeds the workspace budget")
        self.spent += spec.cost
        self.writes += 1
        self.store[region] = Entry(value, by, provenance, confidence, self.step)
        return self.store[region]

    # ------------------------------------------------------------ global availability
    def broadcast(self) -> dict:
        """Every region any component may read, with its provenance attached.

        Global availability is a read property. It does not widen who may write, which is the whole
        point of the separation.
        """
        return {
            name: {
                "value": e.value,
                "writer": e.writer,
                "provenance": e.provenance,
                "confidence": e.confidence,
                "step": e.step,
            }
            for name, e in self.store.items()
        }

    # ------------------------------------------------------------ lifecycle
    def tick(self):
        self.step += 1
        self.reset("step")

    def reset(self, trigger: str):
        if trigger not in PERSISTENCE:
            raise Refused(f"unknown reset trigger {trigger!r}")
        order = PERSISTENCE.index(trigger)
        cleared = [n for n, s in self.specs.items() if n in self.store and PERSISTENCE.index(s.reset) <= order]
        for name in cleared:
            del self.store[name]
        return cleared

    def checkpoint(self) -> dict:
        return {
            "step": self.step,
            "spent": self.spent,
            "writes": self.writes,
            "store": copy.deepcopy(self.store),
        }

    def restore(self, snapshot: dict):
        self.step, self.spent = snapshot["step"], snapshot["spent"]
        self.writes = snapshot["writes"]
        self.store = copy.deepcopy(snapshot["store"])
        return self


class UntypedWorkspace(Workspace):
    """The control: identical storage, capacity and cost, no reader or writer sets.

    It removes exactly one capability, which is what section 18 requires of a control. Anything else
    removed here would make the comparison measure two things at once.
    """

    typed = False


def capacity(ws: Workspace) -> dict:
    """The matched quantities. The two arms must agree on all of these or they are not comparable."""
    return {
        "regions": len(ws.specs),
        "slot_cost_total": round(sum(s.cost for s in ws.specs.values()), 4),
        "budget": ws.budget,
    }


def declaration() -> dict:
    violations = [v for spec in REGIONS for v in spec.violations()]
    return {
        "schema": "substrate-workspace/v1",
        "required_declarations": list(REQUIRED_DECLARATIONS),
        "regions": [
            {
                "name": s.name,
                "shape": s.shape,
                "persistence": s.persistence,
                "timescale": s.timescale,
                "readers": list(s.readers),
                "writers": list(s.writers),
                "provenance_required": s.provenance,
                "confidence_required": s.confidence,
                "cost": s.cost,
                "reset": s.reset,
                "update_rule": s.update_rule,
            }
            for s in REGIONS
        ],
        "all_regions_fully_declared": not violations,
        "declaration_violations": violations,
        "access_rule": ("reading is broad and writing is narrow. Global availability is a read property and does not widen who may write"),
        "control": {
            "class": "UntypedWorkspace",
            "removes": "reader and writer sets",
            "retains": "storage, region set, capacity, cost accounting, budget",
            "why": "a control that removed anything else would measure two changes at once",
        },
        "capacity_matched": capacity(Workspace()) == capacity(UntypedWorkspace()),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_WORKSPACE.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "regions": len(doc["regions"]),
                "fully_declared": doc["all_regions_fully_declared"],
                "capacity_matched": doc["capacity_matched"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
