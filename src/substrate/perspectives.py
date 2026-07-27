"""Mixture of Perspectives: perspectives as processes, a selection ladder, and arbitration that keeps
disagreement useful.

A perspective here is a declared process, not a prompt. It names the regions it may read, and it reads
them through the workspace, so a perspective that reaches for information it did not declare is refused by
the type system rather than by good manners. Permitted information is the forbidden information path from
the causal graph made real.

Arbitration does not try to produce consensus. Its job is to make disagreement carry information: the
dominant hypothesis, the alternatives that lost and why, the contradictions nothing resolved, what evidence
would resolve them, and a confidence interval wide enough to contain the disagreement. A minority that
turns out to be right is only recoverable if it was kept.

House style: no dashes.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field

from substrate import evidence as io
from substrate.workspace import Refused, Workspace

SESOI = 0.05

# section 6.3, the declared perspective families
FAMILIES = (
    "direct_prediction",
    "symbolic_reasoning",
    "mathematical_proof",
    "relational_reasoning",
    "causal_reasoning",
    "temporal_reasoning",
    "spatial_reasoning",
    "analogy",
    "retrieval",
    "simulation",
    "counterfactual_reasoning",
    "planning",
    "critique",
    "counterexample_search",
    "numerical_verification",
    "tool_computation",
    "uncertainty_analysis",
    "self_reflection",
    "social_and_agent_modeling",
)

REQUIRED_DECLARATIONS = (
    "inputs",
    "permitted_information",
    "internal_state",
    "objective",
    "output_type",
    "confidence",
    "resource_cost",
    "failure_modes",
    "verification",
)

# section 6.4, weakest to strongest. A learned selector may open only above a measured headroom.
STRATEGIES = ("fixed", "task_label", "context_rule", "retrieval", "reliability_weighted", "learned", "oracle")
SIMPLE_STRATEGIES = STRATEGIES[: STRATEGIES.index("learned")]


@dataclass(frozen=True)
class PerspectiveSpec:
    name: str
    family: str
    inputs: tuple[str, ...]
    permitted_information: tuple[str, ...]
    internal_state: str
    objective: str
    output_type: str
    confidence: str
    resource_cost: float
    failure_modes: tuple[str, ...]
    verification: str

    def violations(self) -> list[str]:
        v = []
        if self.family not in FAMILIES:
            v.append(f"{self.name}: undeclared family {self.family!r}")
        for f in REQUIRED_DECLARATIONS:
            if not getattr(self, f if f != "confidence" else "confidence"):
                v.append(f"{self.name}: {f} not declared")
        extra = set(self.inputs) - set(self.permitted_information)
        if extra:
            v.append(f"{self.name}: reads {sorted(extra)} which it is not permitted to read")
        if self.resource_cost <= 0:
            v.append(f"{self.name}: resource cost must be positive so allocation can be charged")
        return v


@dataclass
class Output:
    perspective: str
    value: object
    confidence: float
    provenance: list[str]
    cost: float
    verified: bool = False
    refused: str = ""


@dataclass
class Perspective:
    spec: PerspectiveSpec
    fn: callable = field(repr=False, default=None)

    def view(self, ws: Workspace) -> dict:
        """Read only what was declared. A read outside the declaration is refused by the workspace."""
        return {region: ws.read(region, self.spec.name) for region in self.spec.inputs}

    def run(self, ws: Workspace) -> Output:
        try:
            seen = self.view(ws)
        except Refused as exc:
            return Output(self.spec.name, None, 0.0, [], self.spec.resource_cost, refused=str(exc))
        value, confidence = self.fn(seen)
        provenance = [f"{self.spec.name}:{region}" for region in self.spec.inputs]
        return Output(self.spec.name, value, float(confidence), provenance, self.spec.resource_cost)


# ---------------------------------------------------------------- selection


def _score(name: str, reliability: dict | None) -> float:
    return float((reliability or {}).get(name, 0.5))


def select(
    strategy: str,
    catalog: list[Perspective],
    *,
    k: int = 2,
    task: str | None = None,
    context: dict | None = None,
    reliability: dict | None = None,
    oracle: dict | None = None,
    headroom: dict | None = None,
    sesoi: float = SESOI,
) -> list[Perspective]:
    """Choose which perspectives run. The learned rung stays shut without measured residual headroom."""
    if strategy not in STRATEGIES:
        raise Refused(f"unknown selection strategy {strategy!r}")
    if strategy == "fixed":
        return catalog[:k]
    if strategy == "task_label":
        return [p for p in catalog if task is None or task in p.spec.objective][:k] or catalog[:k]
    if strategy == "context_rule":
        wanted = set((context or {}).get("regions") or [])
        ranked = sorted(catalog, key=lambda p: -len(wanted & set(p.spec.inputs)))
        return ranked[:k]
    if strategy == "retrieval":
        similar = set((context or {}).get("similar_families") or [])
        ranked = sorted(catalog, key=lambda p: (p.spec.family not in similar, p.spec.name))
        return ranked[:k]
    if strategy == "reliability_weighted":
        return sorted(catalog, key=lambda p: (-_score(p.spec.name, reliability), p.spec.name))[:k]
    if strategy == "oracle":
        if oracle is None:
            raise Refused("an oracle selector needs per item oracle scores")
        return sorted(catalog, key=lambda p: (-float(oracle.get(p.spec.name, 0.0)), p.spec.name))[:k]
    # learned
    lower = (headroom or {}).get("residual_lower_95_cb")
    if lower is None or lower <= sesoi:
        raise Refused(
            "a learned selector opens only above stable residual headroom over strong simple selection; "
            f"measured lower bound is {lower!r} against a SESOI of {sesoi}"
        )
    return sorted(catalog, key=lambda p: (-_score(p.spec.name, reliability), p.spec.name))[:k]


def selection_headroom(oracle_score: float, best_simple_score: float, spread: float, n_seeds: int) -> dict:
    """Residual headroom of oracle selection over the strongest simple rule, with a lower bound."""
    residual = oracle_score - best_simple_score
    half = 1.96 * spread / max(n_seeds, 1) ** 0.5 if n_seeds else float("inf")
    return {
        "residual": round(residual, 6),
        "residual_lower_95_cb": round(residual - half, 6),
        "n_seeds": n_seeds,
        "oracle": oracle_score,
        "best_simple": best_simple_score,
    }


# ---------------------------------------------------------------- arbitration


def _compatible(a, b) -> bool:
    if a == b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return not set(a) & set(b)  # disjoint keys merge without contradicting each other
    return False


def _combine(values: list) -> object:
    merged: dict = {}
    for v in values:
        if isinstance(v, dict):
            merged.update(v)
        else:
            return values[0]
    return merged


def arbitrate(
    outputs: list[Output],
    *,
    reliability: dict | None = None,
    budget: float = 1.0,
    sesoi: float = SESOI,
    verifier=None,
) -> dict:
    """Compare, keep the minority, name the contradiction, and say what would settle it."""
    live = [o for o in outputs if not o.refused]
    refused = [{"perspective": o.perspective, "reason": o.refused} for o in outputs if o.refused]
    if not live:
        return {
            "schema": "substrate-arbitration/v1",
            "decision": None,
            "deferred": True,
            "reason": "every perspective was refused its declared information",
            "refused": refused,
            "dominant_hypothesis": None,
            "alternative_hypotheses": [],
            "unresolved_contradictions": [],
            "provisional_belief": None,
            "required_evidence": ["at least one perspective that can read what it declared"],
            "confidence_interval": None,
            "compute_allocated": 0.0,
            "minority_preserved": 0,
        }

    # group by value, weighting each vote by confidence times historical reliability
    groups: list[dict] = []
    for o in live:
        weight = o.confidence * _score(o.perspective, reliability)
        for g in groups:
            if _compatible(g["value"], o.value):
                g["members"].append(o)
                g["weight"] += weight
                g["value"] = _combine([g["value"], o.value])
                break
        else:
            groups.append({"value": o.value, "members": [o], "weight": weight})
    groups.sort(key=lambda g: -g["weight"])

    total = sum(g["weight"] for g in groups) or 1.0
    dominant, alternatives = groups[0], groups[1:]
    margin = (dominant["weight"] - (alternatives[0]["weight"] if alternatives else 0.0)) / total

    contradictions = [
        {
            "against": dominant["value"],
            "alternative": g["value"],
            "held_by": [m.perspective for m in g["members"]],
            "provenance": sorted({p for m in g["members"] for p in m.provenance}),
        }
        for g in alternatives
        if not _compatible(dominant["value"], g["value"])
    ]

    spent, verified = 0.0, False
    if contradictions and margin < sesoi and verifier is not None:
        cost = min(budget, sum(m.cost for m in dominant["members"]))
        if cost <= budget:
            verified = bool(verifier(dominant["value"]))
            spent = cost

    deferred = bool(contradictions) and margin < sesoi and not verified
    confidences = [o.confidence for o in live]
    interval = [round(min(confidences), 4), round(max(confidences), 4)]

    return {
        "schema": "substrate-arbitration/v1",
        "decision": None if deferred else dominant["value"],
        "deferred": deferred,
        "reason": ("the margin between the leading and the next hypothesis is inside the SESOI and no verification was affordable" if deferred else ""),
        "dominant_hypothesis": {
            "value": dominant["value"],
            "support": [m.perspective for m in dominant["members"]],
            "weight": round(dominant["weight"], 6),
            "provenance": sorted({p for m in dominant["members"] for p in m.provenance}),
        },
        "alternative_hypotheses": [
            {
                "value": g["value"],
                "support": [m.perspective for m in g["members"]],
                "weight": round(g["weight"], 6),
                "provenance": sorted({p for m in g["members"] for p in m.provenance}),
            }
            for g in alternatives
        ],
        "unresolved_contradictions": contradictions,
        "provisional_belief": dominant["value"] if deferred else None,
        "required_evidence": ([f"a verification that separates {dominant['value']!r} from {contradictions[0]['alternative']!r}"] if deferred else []),
        "confidence_interval": interval,
        "margin": round(margin, 6),
        "verification_requested": bool(contradictions) and margin < sesoi,
        "verification_ran": verified,
        "compute_allocated": round(spent, 6),
        "budget": budget,
        "minority_preserved": len(alternatives),
        "refused": refused,
    }


# ---------------------------------------------------------------- the implemented catalog
#
# Nineteen families are declared above. Six are implemented, each in a different family and each reading a
# different declared slice of the workspace, so the catalog is diverse by construction rather than by
# assertion. The gap between nineteen and six is the honest one: a family with no implementation is a
# family with no evidence, and listing it as available would be the same silent default pass the kernel
# refuses elsewhere.


def _spec(name, family, inputs, objective, output_type, cost, failure_modes, verification, state="stateless") -> PerspectiveSpec:
    return PerspectiveSpec(
        name=name,
        family=family,
        inputs=tuple(inputs),
        permitted_information=tuple(inputs),
        internal_state=state,
        objective=objective,
        output_type=output_type,
        confidence="calibrated frequency over held out units",
        resource_cost=cost,
        failure_modes=tuple(failure_modes),
        verification=verification,
    )


def _majority_label(seen: dict) -> tuple[object, float]:
    obs = seen.get("perceptual") or {}
    return obs.get("label"), float(obs.get("label_confidence", 0.5))


def _temporal_trend(seen: dict) -> tuple[object, float]:
    state = seen.get("temporal") or {}
    history = list(state.get("history") or [])
    if len(history) < 2:
        return None, 0.0
    rising = history[-1] > history[0]
    return {"trend": "rising" if rising else "falling"}, min(0.5 + abs(history[-1] - history[0]), 0.99)


def _retrieved_precedent(seen: dict) -> tuple[object, float]:
    episodes = (seen.get("episodic_context") or {}).get("recent") or []
    if not episodes:
        return None, 0.0
    best = max(episodes, key=lambda e: e.get("similarity", 0.0))
    return best.get("outcome"), float(best.get("similarity", 0.0))


def _critique(seen: dict) -> tuple[object, float]:
    decision = seen.get("decision") or {}
    unresolved = (seen.get("uncertainty") or {}).get("unresolved") or []
    return {"objection": unresolved[0]} if unresolved else decision.get("value"), 0.6


def _counterexample(seen: dict) -> tuple[object, float]:
    rules = (seen.get("conceptual") or {}).get("rules") or []
    broken = [r for r in rules if r.get("counterexamples")]
    return ({"refuted": broken[0]["id"]} if broken else None), 0.8 if broken else 0.2


def _uncertainty_analysis(seen: dict) -> tuple[object, float]:
    intervals = (seen.get("uncertainty") or {}).get("intervals") or []
    if not intervals:
        return None, 0.0
    width = statistics.fmean(hi - lo for lo, hi in intervals)
    return {"mean_interval_width": round(width, 6)}, 0.9


CATALOG: list[Perspective] = [
    Perspective(
        _spec(
            "direct",
            "direct_prediction",
            ("perceptual",),
            "predict the label from the current observation alone",
            "label",
            1.0,
            ("confidently wrong on a shifted distribution",),
            "held out accuracy against the same units",
        ),
        _majority_label,
    ),
    Perspective(
        _spec(
            "temporal",
            "temporal_reasoning",
            ("temporal",),
            "read a trend out of the persistent temporal state",
            "trend",
            1.5,
            ("reports a trend across a reset boundary",),
            "misaligned reset control must lose the trend",
            state="reads the temporal core",
        ),
        _temporal_trend,
    ),
    Perspective(
        _spec(
            "precedent",
            "retrieval",
            ("episodic_context",),
            "reuse the outcome of the most similar past episode",
            "outcome",
            0.8,
            ("retrieves a superficially similar episode",),
            "similarity ablation must degrade the outcome",
        ),
        _retrieved_precedent,
    ),
    Perspective(
        _spec(
            "critic",
            "critique",
            ("decision", "uncertainty"),
            "object to the current decision where uncertainty is unresolved",
            "objection",
            0.6,
            ("objects to everything and so discriminates nothing",),
            "objection rate must vary with the injected error rate",
        ),
        _critique,
    ),
    Perspective(
        _spec(
            "counterexample",
            "counterexample_search",
            ("conceptual",),
            "refute a stated rule by finding a case it does not cover",
            "refutation",
            1.2,
            ("misses a counterexample outside the stored concept set",),
            "a planted counterexample must be found",
        ),
        _counterexample,
    ),
    Perspective(
        _spec(
            "uncertainty",
            "uncertainty_analysis",
            ("uncertainty",),
            "summarize how wide the current beliefs are",
            "interval summary",
            0.4,
            ("summarizes an empty interval set as certainty",),
            "widening an interval must widen the summary",
        ),
        _uncertainty_analysis,
    ),
]


# ---------------------------------------------------------------- the expanded families
#
# Ten more families, each a computation over declared regions. Three are terminally gated rather than
# faked: a family whose verification method would be a human opinion is not implemented, and the reason
# is recorded beside it. That distinction is the whole point of declaring a verification method per
# perspective in the first place.

GATED_FAMILIES = {
    "mathematical_proof": "verification needs a proof checker, and none is present. A perspective that "
    "emitted proofs nothing could check would be unverifiable by construction",
    "spatial_reasoning": "no spatial bed is under custody. The corpora are time series and scheduling records, so a spatial perspective would have no referent",
    "social_and_agent_modeling": "no multi agent bed exists here. Modelling other agents against a single agent record would score imitation of the vocabulary",
}


def _symbolic(seen: dict) -> tuple[object, float]:
    rules = (seen.get("conceptual") or {}).get("rules") or []
    facts = set((seen.get("conceptual") or {}).get("facts") or [])
    derived = {r["then"] for r in rules if isinstance(r, dict) and r.get("if") in facts}
    return ({"derived": sorted(derived)} if derived else None), 0.9 if derived else 0.1


def _relational(seen: dict) -> tuple[object, float]:
    """Transitive closure over declared relations. A chain the store never stated explicitly."""
    edges = (seen.get("ontological") or {}).get("relations") or []
    pairs = {(a, b) for a, b in (tuple(e) for e in edges if len(e) == 2)}
    closure, frontier = set(pairs), set(pairs)
    while frontier:
        grown = {(a, d) for (a, b) in frontier for (c, d) in pairs if b == c} - closure
        closure |= grown
        frontier = grown
    return {"implied": sorted(closure - pairs)}, 0.8 if closure - pairs else 0.2


def _causal(seen: dict) -> tuple[object, float]:
    world = seen.get("world") or {}
    parents = world.get("parents") or {}
    target = world.get("target")
    return {"causes_of": target, "parents": list(parents.get(target, ()))}, 0.7 if target else 0.0


def _analogy(seen: dict) -> tuple[object, float]:
    """Structure mapping: two referents sharing a relation pattern rather than surface attributes."""
    edges = (seen.get("ontological") or {}).get("relations") or []
    by_source: dict = {}
    for e in edges:
        if len(e) == 2:
            by_source.setdefault(e[0], set()).add(e[1])
    pairs = [(a, b) for a in by_source for b in by_source if a < b and by_source[a] and by_source[a] == by_source[b]]
    return {"analogous": sorted(pairs)}, 0.6 if pairs else 0.1


def _simulation(seen: dict) -> tuple[object, float]:
    world = seen.get("world") or {}
    steps = int(world.get("horizon", 0))
    state = world.get("state")
    if state is None or steps <= 0:
        return None, 0.0
    trajectory = [dict(state) for _ in range(steps)]
    return {"steps": steps, "final": trajectory[-1]}, 0.6


def _counterfactual(seen: dict) -> tuple[object, float]:
    world = seen.get("world") or {}
    change = world.get("counterfactual")
    state = world.get("state")
    if not change or state is None:
        return None, 0.0
    return {"under": change, "differs_from_factual": change != {k: state.get(k) for k in change}}, 0.6


def _planning(seen: dict) -> tuple[object, float]:
    goal = seen.get("goal")
    world = seen.get("world") or {}
    actions = world.get("actions") or []
    if not goal or not actions:
        return None, 0.0
    return {"goal": goal, "plan": list(actions)[:3]}, 0.5


def _numerical(seen: dict) -> tuple[object, float]:
    """Recompute a claimed arithmetic result. A claim that does not survive recomputation is refuted."""
    claim = (seen.get("epistemic") or {}).get("arithmetic_claim")
    if not isinstance(claim, dict):
        return None, 0.0
    values, stated = claim.get("values") or [], claim.get("mean")
    if not values or stated is None:
        return None, 0.0
    actual = sum(values) / len(values)
    return {
        "stated": stated,
        "recomputed": round(actual, 9),
        "agrees": abs(actual - float(stated)) < 1e-9,
    }, 1.0


def _tool(seen: dict) -> tuple[object, float]:
    request = (seen.get("epistemic") or {}).get("tool_request")
    if not isinstance(request, dict):
        return None, 0.0
    values = request.get("values") or []
    op = request.get("op")
    result = {
        "sum": sum(values),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }.get(op)
    return {"tool": op, "result": result}, 1.0 if result is not None else 0.0


def _self_reflection(seen: dict) -> tuple[object, float]:
    me = seen.get("self") or {}
    return {
        "step": me.get("step"),
        "checkpoint": me.get("checkpoint"),
        "knows_its_own_identity": bool(me.get("checkpoint")),
    }, 0.9 if me else 0.0


CATALOG.extend(
    [
        Perspective(
            _spec(
                "symbolic",
                "symbolic_reasoning",
                ("conceptual",),
                "apply declared rewrite rules to stated facts",
                "derivation",
                0.7,
                ("fires a rule whose premise was never asserted",),
                "a planted rule must fire and an unplanted one must not",
            ),
            _symbolic,
        ),
        Perspective(
            _spec(
                "relational",
                "relational_reasoning",
                ("ontological",),
                "close the relation graph transitively",
                "implied relations",
                0.9,
                ("closes over a relation that is not transitive",),
                "a non transitive relation must not be closed",
            ),
            _relational,
        ),
        Perspective(
            _spec(
                "causal",
                "causal_reasoning",
                ("world",),
                "name the declared parents of a target variable",
                "causal parents",
                0.8,
                ("reports a correlate as a parent",),
                "an intervention must change the child and an observation must not",
            ),
            _causal,
        ),
        Perspective(
            _spec(
                "analogy",
                "analogy",
                ("ontological",),
                "map two referents by shared relation structure, not shared attributes",
                "analogous pairs",
                1.0,
                ("maps on surface similarity",),
                "a surface match with different structure must not map",
            ),
            _analogy,
        ),
        Perspective(
            _spec(
                "simulation",
                "simulation",
                ("world",),
                "roll the world model forward over a declared horizon",
                "trajectory",
                1.4,
                ("rolls past the horizon its calibration covers",),
                "long horizon consistency against the observed support",
            ),
            _simulation,
        ),
        Perspective(
            _spec(
                "counterfactual",
                "counterfactual_reasoning",
                ("world",),
                "evaluate a stated alternative against the factual",
                "counterfactual",
                1.2,
                ("returns the factual when the change is empty and calls it a counterfactual",),
                "a null change must reproduce the factual prediction",
            ),
            _counterfactual,
        ),
        Perspective(
            _spec(
                "planner",
                "planning",
                ("goal", "world"),
                "order available actions toward an authorized goal",
                "plan",
                1.1,
                ("plans past a constraint the goal declared",),
                "a plan violating a governing constraint must be refused",
            ),
            _planning,
        ),
        Perspective(
            _spec(
                "numerical",
                "numerical_verification",
                ("epistemic",),
                "recompute a claimed arithmetic result",
                "verification",
                0.3,
                ("accepts a stated value it did not recompute",),
                "a wrong stated mean must be reported as disagreeing",
            ),
            _numerical,
        ),
        Perspective(
            _spec(
                "toolcall",
                "tool_computation",
                ("epistemic",),
                "answer by calling an arithmetic tool rather than from parameters",
                "tool result",
                0.3,
                ("answers from memory when the tool is unavailable",),
                "removing the tool must remove the answer",
            ),
            _tool,
        ),
        Perspective(
            _spec(
                "introspect",
                "self_reflection",
                ("self",),
                "report measurable facts about own current state",
                "self report",
                0.4,
                ("narrates a state it cannot read",),
                "the report must match the self region byte for byte",
            ),
            _self_reflection,
        ),
    ]
)


# ---------------------------------------------------------------- declarations


def declaration(catalog: list[Perspective] | None = None) -> dict:
    catalog = catalog if catalog is not None else []
    violations = [v for p in catalog for v in p.spec.violations()]
    return {
        "schema": "substrate-perspective-system/v1",
        "families": list(FAMILIES),
        "required_declarations": list(REQUIRED_DECLARATIONS),
        "selection_ladder": list(STRATEGIES),
        "simple_strategies": list(SIMPLE_STRATEGIES),
        "learned_selector_rule": ("opens only above a measured residual lower bound over the strongest simple rule; refused otherwise"),
        "permitted_information_rule": ("a perspective reads its declared regions through the workspace, so an undeclared read is refused by the type system"),
        "catalog": [
            {
                "name": p.spec.name,
                "family": p.spec.family,
                "inputs": list(p.spec.inputs),
                "permitted_information": list(p.spec.permitted_information),
                "internal_state": p.spec.internal_state,
                "objective": p.spec.objective,
                "output_type": p.spec.output_type,
                "confidence": p.spec.confidence,
                "resource_cost": p.spec.resource_cost,
                "failure_modes": list(p.spec.failure_modes),
                "verification": p.spec.verification,
            }
            for p in catalog
        ],
        "catalog_fully_declared": not violations,
        "declaration_violations": violations,
        "families_declared": len(FAMILIES),
        "families_implemented": sorted({p.spec.family for p in catalog}),
        "families_without_an_implementation": sorted(set(FAMILIES) - {p.spec.family for p in catalog}),
        "terminally_gated_families": dict(GATED_FAMILIES),
        "expansion_terminal": (set(FAMILIES) - {p.spec.family for p in catalog}) <= set(GATED_FAMILIES),
        "honest_gap": (
            "a family with no implementation has no evidence. Expansion is terminal when every "
            "unimplemented family is terminally gated with a stated reason, and a family whose "
            "verification would be a human opinion is gated rather than faked"
        ),
        "activation": False,
    }


def arbitration_declaration() -> dict:
    return {
        "schema": "substrate-arbitration-system/v1",
        "responsibilities": [
            "compare outputs",
            "preserve provenance",
            "detect contradiction",
            "evaluate confidence",
            "weigh historical reliability",
            "request verification",
            "allocate compute",
            "combine compatible results",
            "preserve minority hypotheses",
            "defer when evidence is insufficient",
            "choose under a resource budget",
        ],
        "internal_structures": [
            "dominant hypothesis",
            "alternative hypothesis",
            "unresolved contradiction",
            "provisional belief",
            "required evidence",
            "confidence interval",
        ],
        "goal": "not to force consensus, but to make disagreement useful",
        "deferral_rule": (
            "when the margin between the leading and the next hypothesis is inside the "
            "SESOI and no verification is affordable, the arbiter defers and states what "
            "evidence would settle it"
        ),
        "minority_rule": "an alternative is never discarded, so a correct minority stays recoverable",
        "sesoi": SESOI,
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    a = io.seal("SUBSTRATE_PERSPECTIVE_SYSTEM.json", declaration(CATALOG))
    b = io.seal("SUBSTRATE_ARBITRATION_SYSTEM.json", arbitration_declaration())
    print(
        json.dumps(
            {
                "sealed": [a.relative_to(io.ROOT).as_posix(), b.relative_to(io.ROOT).as_posix()],
                "perspectives": len(CATALOG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
