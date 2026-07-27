"""The memory hierarchy: working, episodic, semantic and procedural, plus consolidation and hygiene.

Four properties here are load bearing and each one exists because its absence is a known defect class.

A generated episode cannot become training material on its own. The fast state program already recorded a
replay buffer that did not replay; the mirror of that failure is a buffer that replays material the system
invented, and the only defence is that promotion requires a verification receipt rather than a flag.

A procedure is not transferable because it worked. It is transferable when it was evaluated somewhere other
than the episodes that produced it, so the transfer test names its own held out set.

Consolidation policies are compared, not chosen. Each declares the information it uses, and the oracle
policy is marked as using information that is not available at decision time, which is what makes it an
upper bound rather than a candidate.

Hygiene never deletes what audit requires. It decays, supersedes, quarantines or archives, and a delete
request against an audit required record is refused and recorded.

House style: no dashes.
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass, field

from substrate import evidence as io

# section 7.2, the twelve fields an episode may carry
EPISODE_FIELDS = (
    "context",
    "observation",
    "internal_state",
    "goal",
    "action",
    "outcome",
    "error",
    "perspectives_used",
    "verification",
    "confidence",
    "cost",
    "later_usefulness",
)

EPISODE_CLASSES = ("recent", "compressed", "verified", "failed", "unresolved", "quarantined")

HYGIENE_ACTIONS = ("decay", "supersede", "quarantine", "archive", "delete")


class Refused(RuntimeError):
    """A memory operation the hierarchy does not permit."""


# ---------------------------------------------------------------- 7.1 working memory


@dataclass
class Slot:
    key: str
    value: object
    priority: float
    activation: float = 1.0
    refreshes: int = 0
    reads: int = 0


class WorkingMemory:
    """Bounded slots with decay, refresh, priority eviction and a measurable interference cost."""

    def __init__(self, capacity: int = 7, decay: float = 0.15):
        self.capacity = capacity
        self.decay = decay
        self.slots: dict[str, Slot] = {}
        self.evicted: list[str] = []

    def write(self, key: str, value, priority: float = 0.5) -> Slot:
        if key not in self.slots and len(self.slots) >= self.capacity:
            victim = min(self.slots.values(), key=lambda s: (s.priority, s.activation))
            if victim.priority >= priority and victim.key != key:
                # a full store does not silently drop the incoming item either
                self.evicted.append(key)
                raise Refused(f"working memory is full and {key!r} does not outrank {victim.key!r}")
            del self.slots[victim.key]
            self.evicted.append(victim.key)
        self.slots[key] = Slot(key, value, float(priority))
        return self.slots[key]

    def read(self, key: str):
        slot = self.slots.get(key)
        if slot is None:
            return None
        slot.reads += 1
        return slot.value

    def refresh(self, key: str) -> bool:
        slot = self.slots.get(key)
        if slot is None:
            return False
        slot.activation = 1.0
        slot.refreshes += 1
        return True

    def tick(self) -> list[str]:
        gone = []
        for key, slot in list(self.slots.items()):
            slot.activation -= self.decay
            if slot.activation <= 0:
                del self.slots[key]
                gone.append(key)
        return gone

    def measure(self, probe: list[tuple[str, object]]) -> dict:
        """Capacity, interference and decay, measured rather than asserted."""
        fresh = WorkingMemory(self.capacity, self.decay)
        retained = []
        for i, (key, value) in enumerate(probe):
            with contextlib.suppress(Refused):
                fresh.write(key, value, priority=1.0 - i / max(len(probe), 1))
            retained.append(sum(1 for k, _ in probe[: i + 1] if k in fresh.slots))
        held = len(fresh.slots)
        first_survives = probe[0][0] in fresh.slots if probe else False
        return {
            "declared_capacity": self.capacity,
            "measured_held": held,
            "items_offered": len(probe),
            "interference": round(1.0 - held / max(len(probe), 1), 4),
            "oldest_survived": first_survives,
            "retention_curve": retained,
            "decay_per_tick": self.decay,
        }


# ---------------------------------------------------------------- 7.2 episodic memory


@dataclass
class Episode:
    id: str
    origin: str = "observed"  # observed or generated
    context: object = None
    observation: object = None
    internal_state: object = None
    goal: object = None
    action: object = None
    outcome: object = None
    error: object = None
    perspectives_used: tuple[str, ...] = ()
    verification: dict | None = None
    confidence: float | None = None
    cost: float = 0.0
    later_usefulness: float | None = None
    klass: str = "recent"

    def missing_fields(self) -> list[str]:
        return [f for f in EPISODE_FIELDS if getattr(self, f) in (None, (), "")]


class EpisodicMemory:
    def __init__(self):
        self.store: dict[str, Episode] = {}
        self.refusals: list[str] = []

    def add(self, episode: Episode) -> Episode:
        if episode.klass not in EPISODE_CLASSES:
            raise Refused(f"unknown episode class {episode.klass!r}")
        self.store[episode.id] = episode
        return episode

    def by_class(self, klass: str) -> list[Episode]:
        return [e for e in self.store.values() if e.klass == klass]

    def classify(self, episode_id: str, klass: str) -> Episode:
        if klass not in EPISODE_CLASSES:
            raise Refused(f"unknown episode class {klass!r}")
        e = self.store[episode_id]
        e.klass = klass
        return e

    def promote_to_training(self, episode_id: str) -> Episode:
        """A generated experience needs a verification receipt, not a flag."""
        e = self.store[episode_id]
        if e.origin == "generated" and not (e.verification or {}).get("verified"):
            self.refusals.append(episode_id)
            raise Refused(
                f"episode {episode_id} was generated and carries no verification receipt, so it cannot "
                "become training material"
            )
        if e.klass == "quarantined":
            self.refusals.append(episode_id)
            raise Refused(f"episode {episode_id} is quarantined")
        e.klass = "verified"
        return e

    def compress(self, episode_id: str, summary: object) -> Episode:
        e = self.store[episode_id]
        if e.klass == "verified":
            raise Refused("a verified episode is not rewritten, a compressed copy is added beside it")
        e.observation, e.klass = summary, "compressed"
        return e


# ---------------------------------------------------------------- 7.3 semantic memory


@dataclass
class Fact:
    id: str
    statement: object
    confidence: float
    provenance: str
    kind: str = "fact"  # concept, fact, relation, rule, abstraction, exception
    superseded_by: str | None = None
    supersedes: str | None = None


class SemanticMemory:
    def __init__(self):
        self.store: dict[str, Fact] = {}

    def assert_(self, fact: Fact) -> Fact:
        if not fact.provenance:
            raise Refused(f"fact {fact.id} has no provenance, so no belief can be traced to it")
        self.store[fact.id] = fact
        return fact

    def supersede(self, old_id: str, new: Fact) -> Fact:
        old = self.store[old_id]
        if not new.provenance:
            raise Refused("a superseding fact needs its own provenance")
        new.supersedes = old_id
        old.superseded_by = new.id
        self.store[new.id] = new
        return new  # the old fact stays in the store, marked, never removed

    def live(self) -> list[Fact]:
        return [f for f in self.store.values() if f.superseded_by is None]

    def chain(self, fact_id: str) -> list[str]:
        out, seen = [fact_id], {fact_id}
        current = self.store[fact_id]
        while current.supersedes and current.supersedes not in seen:
            out.append(current.supersedes)
            seen.add(current.supersedes)
            current = self.store[current.supersedes]
        return out


# ---------------------------------------------------------------- 7.4 procedural memory


@dataclass
class Procedure:
    id: str
    kind: str  # strategy, proof_motif, tool_sequence, planning_routine, debugging_method, composition
    steps: tuple[str, ...]
    source_episodes: tuple[str, ...]
    transfer: dict | None = None


class ProceduralMemory:
    def __init__(self):
        self.store: dict[str, Procedure] = {}

    def add(self, procedure: Procedure) -> Procedure:
        self.store[procedure.id] = procedure
        return procedure

    def transfer_test(
        self, procedure_id: str, evaluated_on: list[str], score: float, baseline: float
    ) -> dict:
        p = self.store[procedure_id]
        overlap = set(evaluated_on) & set(p.source_episodes)
        if overlap:
            raise Refused(
                f"procedure {procedure_id} was evaluated on {sorted(overlap)}, which produced it; a "
                "transfer test must use episodes the procedure has not seen"
            )
        if not evaluated_on:
            raise Refused("a transfer test needs at least one held out episode")
        p.transfer = {
            "evaluated_on": list(evaluated_on),
            "score": score,
            "baseline": baseline,
            "improves": score > baseline,
            "held_out": True,
        }
        return p.transfer

    def transferable(self) -> list[Procedure]:
        return [p for p in self.store.values() if (p.transfer or {}).get("improves")]


# ---------------------------------------------------------------- 7.5 consolidation


@dataclass(frozen=True)
class ConsolidationPolicy:
    name: str
    information_used: frozenset
    available_at_decision_time: bool
    select: callable = field(repr=False, default=None)


def _none(episodes, state):
    return []


def _fixed(episodes, state):
    every = state.get("every", 3)
    return [e for i, e in enumerate(episodes) if (i + 1) % every == 0]


def _boundary(episodes, state):
    return [e for e in episodes if e.context and (e.context or {}).get("boundary")]


def _performance(episodes, state):
    floor = state.get("floor", 0.5)
    return [e for e in episodes if (e.confidence or 0.0) < floor]


def _verification(episodes, state):
    return [e for e in episodes if (e.verification or {}).get("verified")]


def _repetition(episodes, state):
    seen: dict = {}
    for e in episodes:
        seen.setdefault(str(e.action), []).append(e)
    threshold = state.get("times", 2)
    return [e for group in seen.values() if len(group) >= threshold for e in group[-1:]]


def _oracle(episodes, state):
    # later usefulness is not knowable when the decision is made; this is an upper bound, not a candidate
    return [e for e in episodes if (e.later_usefulness or 0.0) > state.get("usefulness", 0.5)]


POLICIES: tuple[ConsolidationPolicy, ...] = (
    ConsolidationPolicy("none", frozenset(), True, _none),
    ConsolidationPolicy("fixed_schedule", frozenset({"episode_index"}), True, _fixed),
    ConsolidationPolicy(
        "boundary_triggered", frozenset({"episode_index", "context_boundary"}), True, _boundary
    ),
    ConsolidationPolicy(
        "performance_triggered", frozenset({"episode_index", "confidence"}), True, _performance
    ),
    ConsolidationPolicy(
        "verification_triggered", frozenset({"episode_index", "verification"}), True, _verification
    ),
    ConsolidationPolicy(
        "repetition_triggered", frozenset({"episode_index", "action_history"}), True, _repetition
    ),
    ConsolidationPolicy("oracle", frozenset({"later_usefulness"}), False, _oracle),
)

BY_POLICY = {p.name: p for p in POLICIES}


def compare_policies(episodes: list[Episode], state: dict | None = None) -> dict:
    state = state or {}
    chosen = {p.name: sorted(e.id for e in p.select(episodes, state)) for p in POLICIES}
    distinct = len({tuple(v) for v in chosen.values()})
    return {
        "selected": chosen,
        "distinct_selections": distinct,
        "policies": len(POLICIES),
        "all_distinct": distinct == len(POLICIES),
        "upper_bound_only": [p.name for p in POLICIES if not p.available_at_decision_time],
    }


# ---------------------------------------------------------------- 7.6 forgetting and hygiene


def hygiene(records: dict, *, audit_required: set, requests: list[tuple[str, str]]) -> dict:
    """Apply hygiene actions. A delete against an audit required record is refused and recorded."""
    applied, refused = [], []
    for record_id, action in requests:
        if action not in HYGIENE_ACTIONS:
            raise Refused(f"unknown hygiene action {action!r}")
        if action == "delete" and record_id in audit_required:
            refused.append(
                {
                    "record": record_id,
                    "action": action,
                    "reason": "scientific or legal auditability requires this record",
                    "substituted": "archive",
                }
            )
            applied.append({"record": record_id, "action": "archive"})
            continue
        applied.append({"record": record_id, "action": action})
    return {
        "applied": applied,
        "refused": refused,
        "audit_preserved": all(r["record"] in audit_required for r in refused),
        "nothing_audit_required_was_deleted": not any(
            a["action"] == "delete" and a["record"] in audit_required for a in applied
        ),
    }


# ---------------------------------------------------------------- declaration


def declaration() -> dict:
    # a probe stream built to separate the seven policies. If a stream cannot separate them the number
    # reported below drops, which is the honest signal that the comparison was not informative.
    probe = [
        Episode(
            f"e{i}",
            action="a" if i % 2 else "b",
            confidence=0.3 + 0.1 * (i % 4),
            context={"boundary": i % 4 == 0},
            verification={"verified": i % 5 == 0},
            later_usefulness=0.08 * i,
        )
        for i in range(1, 13)
    ]
    comparison = compare_policies(probe)
    wm = WorkingMemory()
    measured = wm.measure([(f"k{i}", i) for i in range(12)])
    return {
        "schema": "substrate-memory-system/v1",
        "working_memory": {
            "declared_capacity": wm.capacity,
            "decay_per_tick": wm.decay,
            "measured": measured,
            "measures": [
                "capacity",
                "interference",
                "decay",
                "refresh",
                "prioritization",
                "downstream value",
            ],
        },
        "episodic_memory": {
            "episode_fields": list(EPISODE_FIELDS),
            "classes": list(EPISODE_CLASSES),
            "promotion_rule": (
                "a generated episode needs a verification receipt before it can become training material"
            ),
        },
        "semantic_memory": {
            "kinds": ["concept", "fact", "relation", "rule", "abstraction", "exception"],
            "supersession_rule": (
                "a superseded fact stays in the store marked with its "
                "successor, so a belief keeps a traceable chain"
            ),
        },
        "procedural_memory": {
            "transfer_rule": (
                "a procedure is transferable only after evaluation on episodes it did not come from"
            )
        },
        "consolidation": {
            "policies": [
                {
                    "name": p.name,
                    "information_used": sorted(p.information_used),
                    "available_at_decision_time": p.available_at_decision_time,
                }
                for p in POLICIES
            ],
            "comparison_on_a_probe_stream": comparison,
        },
        "hygiene": {
            "actions": list(HYGIENE_ACTIONS),
            "rule": "a delete against an audit required record is refused and archived instead",
        },
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_MEMORY_SYSTEM.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "policies_distinct": doc["consolidation"]["comparison_on_a_probe_stream"][
                    "distinct_selections"
                ],
                "working_memory_interference": doc["working_memory"]["measured"]["interference"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
