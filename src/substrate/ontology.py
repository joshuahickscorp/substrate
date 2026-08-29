"""The ontological substrate: what kinds of things the entity can represent, and what it refuses to confuse.

Section 6 asks for a typed ontology and then, more importantly, for nine distinctions the typing must not
allow to collapse. Those distinctions are the actual content. A list of twenty seven type names is a
vocabulary; a system that refuses to merge a possible object with an actual one, or an observation with an
inference, is an ontology.

Two mechanisms carry most of the weight.

Merging is evidenced and reversible. Two names for one referent become one item only with a stated reason,
the merge keeps both original identities, and a later contradiction at the same instant reopens it. A merge
that cannot be undone is a guess that has been promoted to a fact.

Unknown is a type, not a failure. A system that must classify everything will classify wrongly, and a
wrong type propagates into every relation built on it. `unknown` is first class and carries the reason it
is unknown.

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from substrate import evidence as io

# section 6, the declared kinds
TYPES = (
    "entity",
    "object",
    "agent",
    "system",
    "component",
    "property",
    "relation",
    "event",
    "process",
    "state",
    "action",
    "observation",
    "context",
    "place",
    "time",
    "duration",
    "cause",
    "effect",
    "goal",
    "constraint",
    "resource",
    "evidence",
    "belief",
    "hypothesis",
    "counterfactual",
    "self",
    "other",
    "unknown",
)

# section 6, what every item declares
ITEM_FIELDS = (
    "identity",
    "type",
    "attributes",
    "relations",
    "temporal_extent",
    "source",
    "confidence",
    "persistence",
    "supersession",
)

# section 6, the nine distinctions the type system must not let collapse
DISTINCTIONS = {
    "object_versus_event": ({"object", "entity", "agent", "system", "component"}, {"event", "process"}),
    "state_versus_process": ({"state"}, {"process"}),
    "observation_versus_inferred": ({"observation"}, {"belief", "hypothesis", "cause", "effect"}),
    "correlation_versus_cause": ({"relation"}, {"cause"}),
    "possible_versus_actual": (set(), set()),  # carried by modality, not by type
    "self_versus_environment": ({"self"}, {"other", "object", "agent"}),
    "agent_versus_tool": ({"agent"}, {"resource", "component"}),
    "goal_versus_prediction": ({"goal"}, {"hypothesis", "effect"}),
    "evidence_versus_belief": ({"evidence", "observation"}, {"belief", "hypothesis"}),
}

MODALITIES = ("actual", "possible", "counterfactual")
SOURCES = ("observed", "inferred", "testimony", "assumed", "retrieved")

# section 6.1, weakest to strongest. Learned opens only above measured headroom.
TYPING_STRATEGIES = ("fixed", "rule_based", "prototype", "retrieval_assisted", "learned", "oracle")
SIMPLE_TYPING = TYPING_STRATEGIES[: TYPING_STRATEGIES.index("learned")]
SESOI = 0.05


class Refused(RuntimeError):
    """An ontological operation the type system does not permit."""


@dataclass
class Item:
    identity: str
    type: str
    attributes: dict = field(default_factory=dict)
    relations: list = field(default_factory=list)  # (kind, other_identity)
    temporal_extent: tuple | None = None  # (start, end) or None for timeless
    source: str = "observed"
    confidence: float = 0.5
    persistence: str = "episode"  # step, episode, persistent
    modality: str = "actual"
    supersession: str | None = None
    merged_from: tuple = ()
    unknown_reason: str = ""

    def violations(self) -> list[str]:
        v = []
        if self.type not in TYPES:
            v.append(f"{self.identity}: undeclared type {self.type!r}")
        if self.source not in SOURCES:
            v.append(f"{self.identity}: undeclared source {self.source!r}")
        if self.modality not in MODALITIES:
            v.append(f"{self.identity}: undeclared modality {self.modality!r}")
        if self.type == "unknown" and not self.unknown_reason:
            v.append(f"{self.identity}: unknown is a type and must say why it is unknown")
        if self.temporal_extent is not None:
            start, end = self.temporal_extent
            if end is not None and start is not None and end < start:
                v.append(f"{self.identity}: temporal extent ends before it starts")
        return v

    def overlaps(self, other: Item) -> bool:
        if self.temporal_extent is None or other.temporal_extent is None:
            return True
        a0, a1 = self.temporal_extent
        b0, b1 = other.temporal_extent
        a1 = float("inf") if a1 is None else a1
        b1 = float("inf") if b1 is None else b1
        return a0 <= b1 and b0 <= a1


def _side(type_name: str, pair) -> int | None:
    left, right = pair
    if type_name in left:
        return 0
    if type_name in right:
        return 1
    return None


class Ontology:
    """Items, relations, evidenced merges, and revisions that leave a chain behind."""

    def __init__(self):
        self.items: dict[str, Item] = {}
        self.merges: list[dict] = []
        self.contradictions: list[dict] = []
        self.revisions: list[dict] = []

    # ------------------------------------------------------------ population
    def add(self, item: Item) -> Item:
        v = item.violations()
        if v:
            raise Refused("; ".join(v))
        self.items[item.identity] = item
        return item

    def live(self) -> list[Item]:
        return [i for i in self.items.values() if i.supersession is None]

    def get(self, identity: str) -> Item | None:
        return self.items.get(identity)

    # ------------------------------------------------------------ the nine distinctions
    def distinguishable(self, a: str, b: str) -> dict:
        """Which declared distinction separates two items, if any."""
        x, y = self.items[a], self.items[b]
        separated = {}
        for name, pair in DISTINCTIONS.items():
            if name == "possible_versus_actual":
                separated[name] = x.modality != y.modality
                continue
            if name == "observation_versus_inferred" and x.source != y.source:
                separated[name] = (x.source == "observed") != (y.source == "observed")
                continue
            sx, sy = _side(x.type, pair), _side(y.type, pair)
            separated[name] = sx is not None and sy is not None and sx != sy
        return {
            "pair": [a, b],
            "separated_by": sorted(k for k, v in separated.items() if v),
            "any": any(separated.values()),
            "detail": separated,
        }

    # ------------------------------------------------------------ merge and split
    def merge(self, a: str, b: str, *, evidence: str, into: str | None = None) -> Item:
        """One referent under two names. Evidenced, and reversible, or it does not happen."""
        if not evidence:
            raise Refused("a merge without stated evidence is a guess promoted to a fact")
        x, y = self.items[a], self.items[b]
        blocking = self.distinguishable(a, b)
        if blocking["any"]:
            raise Refused(f"{a} and {b} are separated by {blocking['separated_by']} and cannot be one referent")
        if not x.overlaps(y):
            raise Refused(f"{a} and {b} do not share any moment, so they are not one thing over time")
        identity = into or f"{a}+{b}"
        merged = Item(
            identity=identity,
            type=x.type if x.type == y.type else "unknown",
            attributes={**y.attributes, **x.attributes},
            relations=list({*map(tuple, x.relations), *map(tuple, y.relations)}),
            temporal_extent=self._union(x.temporal_extent, y.temporal_extent),
            source="inferred",
            confidence=min(x.confidence, y.confidence),
            persistence=x.persistence,
            modality=x.modality,
            merged_from=(a, b),
            unknown_reason="" if x.type == y.type else "merged items disagreed on type",
        )
        self.add(merged)
        x.supersession = y.supersession = identity
        self.merges.append({"merged": identity, "from": [a, b], "evidence": evidence})
        return merged

    @staticmethod
    def _union(a: tuple | None, b: tuple | None) -> tuple | None:
        """The span covering both observations. Unknown extent absorbs anything."""
        if a is None or b is None:
            return None
        starts, ends = (a[0], b[0]), (a[1], b[1])
        end = None if None in ends else max(ends)
        return (min(starts), end)

    def split(self, identity: str, *, reason: str) -> list[Item]:
        """Undo a merge. The merged item stays, marked, so the mistake is part of the record."""
        item = self.items[identity]
        if not item.merged_from:
            raise Refused(f"{identity} was never merged, so it cannot be split")
        restored = []
        for origin in item.merged_from:
            self.items[origin].supersession = None
            restored.append(self.items[origin])
        item.supersession = "split"
        self.revisions.append({"kind": "split", "identity": identity, "into": list(item.merged_from), "reason": reason})
        return restored

    def detect_mistaken_merge(self, identity: str) -> dict:
        """A merged referent that holds contradictory values for one attribute at one time is two things."""
        item = self.items.get(identity)
        if item is None or not item.merged_from:
            return {"identity": identity, "mistaken": False, "reason": "not a merged item"}
        a, b = (self.items[i] for i in item.merged_from)
        clashing = [k for k in set(a.attributes) & set(b.attributes) if a.attributes[k] != b.attributes[k] and a.overlaps(b)]
        if clashing:
            self.contradictions.append({"identity": identity, "attributes": clashing})
        return {
            "identity": identity,
            "mistaken": bool(clashing),
            "clashing_attributes": sorted(clashing),
            "reason": "one referent cannot hold two values for one attribute at one time" if clashing else "",
        }

    def detect_mistaken_split(self, a: str, b: str) -> dict:
        """Two items that never disagree and always co occur were probably one thing."""
        x, y = self.items[a], self.items[b]
        shared = set(x.attributes) & set(y.attributes)
        disagree = [k for k in shared if x.attributes[k] != y.attributes[k]]
        return {
            "pair": [a, b],
            "shared_attributes": sorted(shared),
            "disagreements": sorted(disagree),
            "mistaken_split": bool(shared) and not disagree and x.overlaps(y) and not self.distinguishable(a, b)["any"],
            "reason": "no disagreement anywhere and overlapping in time",
        }

    # ------------------------------------------------------------ revision
    def revise_type(self, identity: str, new_type: str, *, evidence: str) -> Item:
        item = self.items[identity]
        if new_type not in TYPES:
            raise Refused(f"undeclared type {new_type!r}")
        if not evidence:
            raise Refused("a type revision without evidence is a relabelling")
        self.revisions.append({"kind": "type", "identity": identity, "from": item.type, "to": new_type, "evidence": evidence})
        item.type = new_type
        if new_type != "unknown":
            item.unknown_reason = ""
        return item

    def relation_consistency(self) -> dict:
        """A relation to an item that does not exist, or a part that contains its own whole."""
        dangling, cyclic = [], []
        for item in self.items.values():
            for kind, other in (tuple(r) for r in item.relations):
                if other not in self.items:
                    dangling.append({"from": item.identity, "kind": kind, "to": other})
                elif kind == "part_of":
                    back = {tuple(r) for r in self.items[other].relations}
                    if ("part_of", item.identity) in back:
                        cyclic.append({"a": item.identity, "b": other})
        return {"dangling": dangling, "cyclic_composition": cyclic, "consistent": not dangling and not cyclic}

    def temporal_consistency(self) -> dict:
        """An event inside a process it postdates, or an item whose parts outlive it."""
        problems = []
        for item in self.items.values():
            if item.temporal_extent is None:
                continue
            for kind, other in (tuple(r) for r in item.relations):
                target = self.items.get(other)
                if kind != "part_of" or target is None or target.temporal_extent is None:
                    continue
                if not item.overlaps(target):
                    problems.append(
                        {
                            "part": item.identity,
                            "whole": other,
                            "reason": "a part must share time with its whole",
                        }
                    )
        return {"problems": problems, "consistent": not problems}

    def self_environment_boundary(self) -> dict:
        selves = [i.identity for i in self.live() if i.type == "self"]
        return {
            "self_items": selves,
            "singleton": len(selves) <= 1,
            "rule": "there is one self item, and it may not merge with an item typed other",
        }


# ---------------------------------------------------------------- Substrate v3 active ontology

V3_TYPES = (
    "entity", "property", "relation", "event", "process", "state", "role", "boundary",
    "cause", "effect", "rule", "exception", "abstraction", "category", "instance", "part",
    "whole", "identity", "change", "possibility", "impossibility", "unknown",
)
V3_REVISION_OPERATIONS = ("form", "split", "merge", "supersede", "map")


@dataclass
class Concept:
    """A developmental category whose history and exceptions remain inspectable."""

    identity: str
    members: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    relations: set[tuple[str, str]] = field(default_factory=set)
    exceptions: set[str] = field(default_factory=set)
    status: str = "active"
    supersedes: tuple[str, ...] = ()

    def snapshot(self) -> dict:
        return {
            "identity": self.identity,
            "members": sorted(self.members),
            "features": sorted(self.features),
            "relations": sorted([list(row) for row in self.relations]),
            "exceptions": sorted(self.exceptions),
            "status": self.status,
            "supersedes": list(self.supersedes),
        }


@dataclass
class OntologyRevision:
    identity: str
    operation: str
    triggering_evidence: tuple[str, ...]
    old_ontology: dict
    new_ontology: dict
    predicted_benefit: float
    actual_held_out_benefit: float | None
    cost: float
    affected_beliefs: tuple[str, ...]
    affected_procedures: tuple[str, ...]
    exceptions: tuple[str, ...]
    rollback_checkpoint: dict
    admitted: bool = False
    rolled_back: bool = False

    def receipt(self) -> dict:
        return {
            "identity": self.identity,
            "operation": self.operation,
            "triggering_evidence": list(self.triggering_evidence),
            "old_ontology": self.old_ontology,
            "new_ontology": self.new_ontology,
            "predicted_benefit": self.predicted_benefit,
            "actual_held_out_benefit": self.actual_held_out_benefit,
            "cost": self.cost,
            "affected_beliefs": list(self.affected_beliefs),
            "affected_procedures": list(self.affected_procedures),
            "exceptions": list(self.exceptions),
            "rollback_checkpoint": self.rollback_checkpoint,
            "admitted": self.admitted,
            "rolled_back": self.rolled_back,
        }


class ActiveOntology:
    """Evidence-triggered concept formation with held-out admission and exact rollback."""

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.instance_features: dict[str, set[str]] = {}
        self.cross_representation_maps: dict[str, str] = {}
        self.revisions: list[OntologyRevision] = []

    def snapshot(self) -> dict:
        return {
            "concepts": {key: self.concepts[key].snapshot() for key in sorted(self.concepts)},
            "instance_features": {key: sorted(value) for key, value in sorted(self.instance_features.items())},
            "cross_representation_maps": dict(sorted(self.cross_representation_maps.items())),
        }

    def restore(self, state: dict) -> None:
        self.concepts = {
            key: Concept(
                identity=value["identity"],
                members=set(value["members"]),
                features=set(value["features"]),
                relations={tuple(row) for row in value["relations"]},
                exceptions=set(value["exceptions"]),
                status=value["status"],
                supersedes=tuple(value["supersedes"]),
            )
            for key, value in state["concepts"].items()
        }
        self.instance_features = {key: set(value) for key, value in state["instance_features"].items()}
        self.cross_representation_maps = dict(state["cross_representation_maps"])

    def observe(self, instance: str, features: set[str], *, category: str | None = None) -> None:
        self.instance_features[instance] = set(features)
        if category is not None:
            if category not in self.concepts:
                self.concepts[category] = Concept(category)
            self.concepts[category].members.add(instance)

    def form_category(self, identity: str, members: set[str], *, evidence: tuple[str, ...], predicted_benefit: float) -> OntologyRevision:
        if not evidence or not members:
            raise Refused("category formation needs verified evidence and at least one instance")
        shared = set.intersection(*(self.instance_features[member] for member in members))
        if not shared:
            raise Refused("category formation needs a shared abstraction, not only a label")
        before = self.snapshot()
        self.concepts[identity] = Concept(identity, set(members), shared)
        return self._record("form", evidence, before, predicted_benefit)

    def split_category(
        self,
        identity: str,
        left: str,
        right: str,
        *,
        discriminator: str,
        evidence: tuple[str, ...],
        predicted_benefit: float,
    ) -> OntologyRevision:
        original = self.concepts[identity]
        left_members = {member for member in original.members if discriminator in self.instance_features[member]}
        right_members = original.members - left_members
        if not evidence or not left_members or not right_members:
            raise Refused("a split needs verified exceptions and two nonempty categories")
        before = self.snapshot()
        original.status = "superseded"
        self.concepts[left] = Concept(left, left_members, original.features | {discriminator}, supersedes=(identity,))
        self.concepts[right] = Concept(right, right_members, original.features - {discriminator}, supersedes=(identity,))
        return self._record("split", evidence, before, predicted_benefit, exceptions=tuple(sorted(right_members)))

    def merge_categories(
        self,
        left: str,
        right: str,
        merged: str,
        *,
        evidence: tuple[str, ...],
        predicted_benefit: float,
    ) -> OntologyRevision:
        a, b = self.concepts[left], self.concepts[right]
        if not evidence or a.exceptions or b.exceptions:
            raise Refused("a merge needs verified equivalence and cannot erase recorded exceptions")
        if a.features != b.features:
            raise Refused("categories with different predictive features are not redundant")
        before = self.snapshot()
        a.status = b.status = "superseded"
        self.concepts[merged] = Concept(
            merged,
            a.members | b.members,
            set(a.features),
            a.relations | b.relations,
            supersedes=(left, right),
        )
        return self._record("merge", evidence, before, predicted_benefit)

    def map_representation(self, surface: str, concept: str, *, evidence: tuple[str, ...], predicted_benefit: float) -> OntologyRevision:
        if concept not in self.concepts or not evidence:
            raise Refused("cross representation mapping needs an existing concept and verified evidence")
        before = self.snapshot()
        self.cross_representation_maps[surface] = concept
        return self._record("map", evidence, before, predicted_benefit)

    def insufficient_categories(self, features: set[str]) -> bool:
        return not any(concept.status == "active" and concept.features <= features for concept in self.concepts.values())

    def complete_revision(self, revision: OntologyRevision, *, held_out_benefit: float, sesoi: float = 0.05) -> OntologyRevision:
        revision.actual_held_out_benefit = float(held_out_benefit)
        revision.admitted = held_out_benefit >= sesoi
        if not revision.admitted:
            self.restore(revision.rollback_checkpoint)
            revision.rolled_back = True
        return revision

    def _record(
        self,
        operation: str,
        evidence: tuple[str, ...],
        before: dict,
        predicted_benefit: float,
        *,
        exceptions: tuple[str, ...] = (),
    ) -> OntologyRevision:
        if operation not in V3_REVISION_OPERATIONS:
            raise Refused(f"unknown ontology revision {operation!r}")
        revision = OntologyRevision(
            identity=f"ontology-revision-{len(self.revisions):06d}",
            operation=operation,
            triggering_evidence=tuple(evidence),
            old_ontology=before,
            new_ontology=self.snapshot(),
            predicted_benefit=float(predicted_benefit),
            actual_held_out_benefit=None,
            cost=1.0,
            affected_beliefs=(),
            affected_procedures=(),
            exceptions=exceptions,
            rollback_checkpoint=before,
        )
        self.revisions.append(revision)
        return revision


# ---------------------------------------------------------------- 6.1 typing ladder


def type_of(
    observation: dict,
    strategy: str,
    *,
    rules: dict | None = None,
    prototypes: dict | None = None,
    retrieved: dict | None = None,
    oracle: dict | None = None,
    headroom: dict | None = None,
) -> str:
    if strategy not in TYPING_STRATEGIES:
        raise Refused(f"unknown typing strategy {strategy!r}")
    if strategy == "fixed":
        return "entity"
    if strategy == "rule_based":
        for key, typed in (rules or {}).items():
            if key in observation:
                return typed
        return "unknown"
    if strategy == "prototype":
        best, score = "unknown", 0
        for typed, proto in (prototypes or {}).items():
            overlap = len(set(proto) & set(observation))
            if overlap > score:
                best, score = typed, overlap
        return best
    if strategy == "retrieval_assisted":
        return (retrieved or {}).get(observation.get("id"), "unknown")
    if strategy == "oracle":
        if oracle is None:
            raise Refused("an oracle typer needs the true types")
        return oracle.get(observation.get("id"), "unknown")
    lower = (headroom or {}).get("residual_lower_95_cb")
    if lower is None or lower <= SESOI:
        raise Refused(
            "a learned typer opens only above measured residual headroom over the strongest simple rule; "
            f"measured lower bound is {lower!r} against a SESOI of {SESOI}"
        )
    return (retrieved or {}).get(observation.get("id"), "unknown")


def declaration() -> dict:
    o = Ontology()
    o.add(Item("obs1", "observation", {"colour": "red"}, source="observed", temporal_extent=(0, 1)))
    o.add(Item("belief1", "belief", {"colour": "red"}, source="inferred", temporal_extent=(0, 1)))
    probe = o.distinguishable("obs1", "belief1")
    return {
        "schema": "substrate-ontology/v1",
        "types": list(TYPES),
        "item_fields": list(ITEM_FIELDS),
        "modalities": list(MODALITIES),
        "sources": list(SOURCES),
        "distinctions": sorted(DISTINCTIONS),
        "distinction_probe": probe,
        "typing_ladder": list(TYPING_STRATEGIES),
        "learned_typing_rule": "opens only above measured residual headroom over the strongest simple rule",
        "supported": [
            "identity over time",
            "object persistence",
            "event boundaries",
            "compositional relations",
            "nested systems",
            "multiple descriptions of one referent",
            "unknown types",
            "uncertainty",
            "contradictions",
            "ontology revision",
        ],
        "merge_rule": ("a merge needs stated evidence, is refused across any declared distinction, and stays reversible. The merged item keeps both origins"),
        "unknown_rule": "unknown is a first class type and carries the reason it is unknown",
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_ONTOLOGY.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "types": len(doc["types"]),
                "distinctions": len(doc["distinctions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
