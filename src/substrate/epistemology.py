"""The epistemological substrate: how a belief is justified, and what happens when the justification fails.

Section 7 asks for beliefs with provenance and for bounded revision. The part that is easy to get wrong,
and that this module exists to get right, is dependency. A belief is rarely wrong on its own. It is wrong
because something it rested on was wrong, and a store that records confidence but not what a belief rests
on cannot propagate a retraction. It will keep reporting high confidence in a conclusion whose premise it
has already withdrawn, which is the most expensive failure an epistemology can have.

So justification here is a graph. Retracting a belief walks its dependants and reduces them. A cycle in
that graph is circular justification and is detected rather than followed. A belief whose support bottoms
out in something retracted is a hidden dependency failure and is found by walking, not by asking it how
confident it feels.

Metacognitive control operates on epistemic value rather than confidence. A belief held at 0.5 with
nothing at stake and no available test is not worth verifying; a belief held at 0.9 that everything else
rests on is. Confidence alone cannot tell those apart, so it is not what chooses.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from substrate import evidence as io

# section 7, what an epistemic item may be
KINDS = (
    "raw_observation",
    "memory",
    "testimony",
    "retrieved_record",
    "inference",
    "hypothesis",
    "assumption",
    "prediction",
    "counterfactual",
    "verified_fact",
    "disputed_claim",
    "unknown",
)

# section 7, what every belief declares
BELIEF_FIELDS = (
    "content",
    "provenance",
    "supporting_evidence",
    "contradicting_evidence",
    "method",
    "confidence",
    "calibration",
    "scope",
    "time",
    "defeaters",
    "supersession",
    "verification_status",
)

CONCEPTS = (
    "belief",
    "knowledge_claim",
    "evidence",
    "justification",
    "uncertainty",
    "ignorance",
    "contradiction",
    "defeasibility",
    "revision",
    "retraction",
    "dependency",
    "testimony",
    "source_reliability",
)

# section 7.2
EPISTEMIC_ACTIONS = (
    "accept_provisionally",
    "reject",
    "defer",
    "retrieve",
    "verify",
    "test",
    "ask",
    "simulate",
    "seek_counterexample",
    "preserve_multiple_hypotheses",
)

VERIFICATION = ("unverified", "verifying", "verified", "refuted", "unresolved")


class Refused(RuntimeError):
    """An epistemic operation the store will not perform."""


@dataclass
class Belief:
    id: str
    content: object
    kind: str = "inference"
    provenance: str = ""
    supporting_evidence: list = field(default_factory=list)
    contradicting_evidence: list = field(default_factory=list)
    method: str = ""
    confidence: float = 0.5
    calibration: dict | None = None
    scope: str = "general"
    time: tuple | None = None
    defeaters: list = field(default_factory=list)
    supersession: str | None = None
    verification_status: str = "unverified"
    depends_on: tuple = ()
    source: str = ""
    retracted: bool = False

    def violations(self) -> list[str]:
        v = []
        if self.kind not in KINDS:
            v.append(f"{self.id}: undeclared epistemic kind {self.kind!r}")
        if self.verification_status not in VERIFICATION:
            v.append(f"{self.id}: undeclared verification status {self.verification_status!r}")
        if not self.provenance:
            v.append(f"{self.id}: no provenance, so nothing can be traced to it")
        if not self.method:
            v.append(f"{self.id}: no method, so how it was arrived at is unrecorded")
        if self.kind == "verified_fact" and self.verification_status != "verified":
            v.append(f"{self.id}: claims to be a verified fact but was never verified")
        return v


# ---------------------------------------------------------------- 7.1 revision policies


# Each policy returns the winner and the margin in the quantity it actually used. The margin matters:
# a conflict is unresolved when the policy's own discriminating quantity is too close to call, and
# comparing confidences for a policy that never looked at confidence would mark a decisive source
# reliability judgement as a tie.


def _latest(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    return incoming, 1.0  # arrival order always separates


def _highest_confidence(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    margin = abs(incoming.confidence - existing.confidence)
    return (incoming if incoming.confidence > existing.confidence else existing), margin


def _net(b: Belief) -> int:
    return len(b.supporting_evidence) - len(b.contradicting_evidence)


def _source_reliability(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    a = store.source_reliability.get(existing.source, 0.5)
    b = store.source_reliability.get(incoming.source, 0.5)
    return (incoming if b > a else existing), abs(b - a)


def _evidence_weighted(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    a, b = _net(existing), _net(incoming)
    return (incoming if b > a else existing), float(abs(b - a))


def _dependency_aware(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    """A claim resting on something retracted loses, whatever its confidence says."""
    if store.rests_on_retracted(existing.id):
        return incoming, 1.0
    if store.rests_on_retracted(incoming.id):
        return existing, 1.0
    return _evidence_weighted(existing, incoming, store)


def _oracle(existing: Belief, incoming: Belief, store) -> tuple[Belief, float]:
    truth = store.oracle or {}
    for candidate in (incoming, existing):
        if truth.get(candidate.id) is True:
            return candidate, 1.0
    return existing, 0.0


@dataclass(frozen=True)
class RevisionPolicy:
    name: str
    information_used: frozenset
    available_at_decision_time: bool
    choose: callable = field(repr=False, default=None)


POLICIES: tuple[RevisionPolicy, ...] = (
    RevisionPolicy("latest_claim_wins", frozenset({"arrival_order"}), True, _latest),
    RevisionPolicy("highest_confidence_wins", frozenset({"confidence"}), True, _highest_confidence),
    RevisionPolicy("source_reliability", frozenset({"source_history"}), True, _source_reliability),
    RevisionPolicy("evidence_weighted", frozenset({"evidence_counts"}), True, _evidence_weighted),
    RevisionPolicy("dependency_aware", frozenset({"evidence_counts", "dependency_graph"}), True, _dependency_aware),
    RevisionPolicy("oracle", frozenset({"ground_truth"}), False, _oracle),
)

BY_POLICY = {p.name: p for p in POLICIES}


class Epistemology:
    """A justification graph, not a confidence table."""

    def __init__(self, policy: str = "dependency_aware"):
        if policy not in BY_POLICY:
            raise Refused(f"unknown revision policy {policy!r}")
        self.policy = BY_POLICY[policy]
        self.beliefs: dict[str, Belief] = {}
        self.source_reliability: dict[str, float] = {}
        self.unresolved: list[dict] = []
        self.history: list[dict] = []
        self.oracle: dict | None = None

    # ------------------------------------------------------------ population
    def assert_(self, belief: Belief) -> Belief:
        v = belief.violations()
        if v:
            raise Refused("; ".join(v))
        for dep in belief.depends_on:
            if dep not in self.beliefs:
                raise Refused(f"{belief.id} depends on {dep}, which is not held")
        self.beliefs[belief.id] = belief
        if self._cycle(belief.id):
            del self.beliefs[belief.id]
            raise Refused(f"{belief.id} closes a circular justification")
        self.history.append({"action": "assert", "id": belief.id})
        return belief

    def live(self) -> list[Belief]:
        return [b for b in self.beliefs.values() if not b.retracted and b.supersession is None]

    # ------------------------------------------------------------ dependency
    def _cycle(self, start: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            current = stack.pop()
            if current in seen:
                return True
            seen.add(current)
            stack.extend(self.beliefs[current].depends_on if current in self.beliefs else ())
        return False

    def dependants(self, belief_id: str) -> list[str]:
        return sorted(b.id for b in self.beliefs.values() if belief_id in b.depends_on)

    def rests_on_retracted(self, belief_id: str) -> list[str]:
        """Walk the support chain. A conclusion is only as held as its premises."""
        broken, stack, seen = [], list(self.beliefs[belief_id].depends_on), set()
        while stack:
            current = stack.pop()
            if current in seen or current not in self.beliefs:
                continue
            seen.add(current)
            if self.beliefs[current].retracted:
                broken.append(current)
            stack.extend(self.beliefs[current].depends_on)
        return sorted(broken)

    # ------------------------------------------------------------ 7.1 the eight revision moves
    def add_evidence(self, belief_id: str, evidence: str, *, supports: bool = True) -> Belief:
        b = self.beliefs[belief_id]
        (b.supporting_evidence if supports else b.contradicting_evidence).append(evidence)
        delta = 0.1 if supports else -0.15
        b.confidence = max(0.0, min(1.0, b.confidence + delta))
        self.history.append({"action": "add_evidence", "id": belief_id, "supports": supports})
        return b

    def retract(self, belief_id: str, *, reason: str) -> dict:
        """Retraction propagates. Anything resting on this drops, whatever it said about itself."""
        if not reason:
            raise Refused("a retraction without a reason is a deletion")
        b = self.beliefs[belief_id]
        b.retracted, b.verification_status = True, "refuted"
        affected = []
        for dependant in self.dependants(belief_id):
            d = self.beliefs[dependant]
            d.confidence = min(d.confidence, 0.2)
            d.verification_status = "unresolved"
            d.defeaters.append(f"rests on retracted {belief_id}")
            affected.append(dependant)
        self.history.append({"action": "retract", "id": belief_id, "propagated_to": affected})
        return {"retracted": belief_id, "reason": reason, "propagated_to": affected}

    def supersede(self, old_id: str, new: Belief) -> Belief:
        self.assert_(new)  # noqa: UP005
        self.beliefs[old_id].supersession = new.id
        self.history.append({"action": "supersede", "from": old_id, "to": new.id})
        return new

    def revise(self, incoming: Belief) -> dict:
        """Two claims about the same content. The policy chooses and the loser is kept, not deleted."""
        existing = next((b for b in self.live() if b.content == incoming.content and b.id != incoming.id), None)
        if existing is None:
            self.assert_(incoming)  # noqa: UP005
            return {"outcome": "asserted", "kept": incoming.id, "alternative": None}
        if incoming.id not in self.beliefs:
            self.beliefs[incoming.id] = incoming
        winner, margin = self.policy.choose(existing, incoming, self)
        loser = incoming if winner is existing else existing
        # the margin is in whatever quantity the policy used, so a decisive source reliability call is
        # not mistaken for a tie merely because the two confidences happen to be close
        if margin < 0.05:
            self.unresolved.append(
                {
                    "between": sorted([existing.id, incoming.id]),
                    "policy": self.policy.name,
                    "margin": round(margin, 6),
                    "reason": "neither claim dominates in the quantity this policy uses",
                }
            )
            return {
                "outcome": "unresolved",
                "kept": None,
                "margin": round(margin, 6),
                "alternative": sorted([existing.id, incoming.id]),
                "policy": self.policy.name,
            }
        loser.supersession = winner.id
        return {
            "outcome": "revised",
            "kept": winner.id,
            "alternative": loser.id,
            "margin": round(margin, 6),
            "policy": self.policy.name,
            "alternative_preserved": self.beliefs[loser.id] is not None,
        }

    def request_evidence(self, belief_id: str) -> dict:
        b = self.beliefs[belief_id]
        return {
            "belief": belief_id,
            "needed": b.defeaters or ["any independent check of the method"],
            "method": b.method,
            "current_confidence": b.confidence,
        }

    # ------------------------------------------------------------ 7.3 diagnostics
    def circular(self) -> list[str]:
        return sorted(b for b in self.beliefs if self._cycle(b))

    def hidden_dependency_failures(self) -> list[dict]:
        rows = []
        for b in self.live():
            broken = self.rests_on_retracted(b.id)
            if broken and b.confidence > 0.5:
                rows.append({"belief": b.id, "confidence": b.confidence, "rests_on_retracted": broken})
        return rows

    def unsupported_confidence(self, floor: float = 0.8) -> list[str]:
        return sorted(b.id for b in self.live() if b.confidence >= floor and not b.supporting_evidence and b.kind not in ("raw_observation", "verified_fact"))

    def ignorance(self, questions: list[str]) -> dict:
        """What the store explicitly does not know, which is not the same as low confidence."""
        held = {str(b.content) for b in self.live()}
        return {
            "asked": questions,
            "unknown": sorted(q for q in questions if q not in held),
            "rule": "an absent belief is ignorance, and is reported rather than guessed at",
        }


# ---------------------------------------------------------------- 7.2 epistemic value


def epistemic_value(belief: Belief, *, stakes: float, test_available: bool, dependants: int = 0) -> dict:
    """What an action is worth, which confidence alone cannot tell you.

    A claim at 0.5 that nothing rests on and that no test can touch is not worth verifying. A claim at 0.9
    that everything rests on is. Both facts are invisible to confidence.
    """
    uncertainty = 1.0 - abs(belief.confidence - 0.5) * 2  # peaks at 0.5, zero at either end
    leverage = 1.0 + dependants
    value = uncertainty * stakes * leverage * (1.0 if test_available else 0.0)
    if not test_available:
        action = "preserve_multiple_hypotheses" if uncertainty > 0.5 else "accept_provisionally"
    elif value > 1.0:
        action = "verify"
    elif belief.contradicting_evidence:
        action = "seek_counterexample"
    elif uncertainty > 0.5:
        action = "test"
    else:
        action = "accept_provisionally"
    return {
        "belief": belief.id,
        "confidence": belief.confidence,
        "uncertainty": round(uncertainty, 4),
        "stakes": stakes,
        "dependants": dependants,
        "test_available": test_available,
        "epistemic_value": round(value, 4),
        "action": action,
        "rule": "chosen on epistemic value, not on confidence",
    }


def compare_policies(existing: Belief, incoming: Belief, store: Epistemology) -> dict:
    chosen = {}
    for policy in POLICIES:
        try:
            winner, margin = policy.choose(existing, incoming, store)
            chosen[policy.name] = winner.id if margin >= 0.05 else "unresolved"
        except Exception as exc:  # a policy that cannot decide says so rather than crashing the compare
            chosen[policy.name] = f"undecidable: {exc.__class__.__name__}"
    return {
        "selected": chosen,
        "distinct_outcomes": len(set(chosen.values())),
        "upper_bound_only": [p.name for p in POLICIES if not p.available_at_decision_time],
    }


def declaration() -> dict:
    store = Epistemology()
    store.source_reliability = {"instrument": 0.9, "rumour": 0.2}
    store.assert_(
        Belief(
            "premise",
            "the bed is temporal",
            kind="raw_observation",
            provenance="proof/e1.json",
            method="measured",
            confidence=0.8,
            source="instrument",
        )
    )
    store.assert_(
        Belief(
            "conclusion",
            "the core is necessary",
            kind="inference",
            provenance="derived",
            method="deduction",
            confidence=0.85,
            depends_on=("premise",),
            source="instrument",
        )
    )
    before = store.beliefs["conclusion"].confidence
    store.retract("premise", reason="the verifier found the baseline unconverged")
    after = store.beliefs["conclusion"].confidence
    return {
        "schema": "substrate-epistemology/v1",
        "kinds": list(KINDS),
        "belief_fields": list(BELIEF_FIELDS),
        "concepts": list(CONCEPTS),
        "epistemic_actions": list(EPISTEMIC_ACTIONS),
        "verification_states": list(VERIFICATION),
        "justification_is_a_graph": True,
        "retraction_propagates": {
            "dependant": "conclusion",
            "confidence_before": before,
            "confidence_after": after,
            "rule": "a conclusion is only as held as its premises",
        },
        "control_rule": (
            "metacognitive control operates on epistemic value, which combines uncertainty, "
            "stakes and how much rests on the belief. Confidence alone does not choose"
        ),
        "activation": False,
    }


def revision_declaration() -> dict:
    return {
        "schema": "substrate-belief-revision/v1",
        "moves": [
            "add evidence",
            "reduce confidence",
            "increase confidence",
            "preserve alternatives",
            "retract",
            "supersede",
            "mark unresolved conflict",
            "request further evidence",
        ],
        "policies": [
            {
                "name": p.name,
                "information_used": sorted(p.information_used),
                "available_at_decision_time": p.available_at_decision_time,
            }
            for p in POLICIES
        ],
        "default_policy": "dependency_aware",
        "why": ("a claim resting on something retracted loses whatever its confidence says, which is the one thing a confidence table cannot express"),
        "alternative_rule": "the losing claim is superseded and kept, never deleted",
        "unresolved_rule": ("when neither claim dominates by the SESOI the conflict is marked unresolved rather than settled by arrival order"),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    a = io.seal("SUBSTRATE_EPISTEMOLOGY.json", declaration())
    b = io.seal("SUBSTRATE_BELIEF_REVISION.json", revision_declaration())
    print(
        json.dumps(
            {
                "sealed": [p.relative_to(io.ROOT).as_posix() for p in (a, b)],
                "kinds": len(KINDS),
                "policies": len(POLICIES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
