"""Bounded recurrent cognition with inspectable, viewpoint-bound representations.

A perspective is an attributed representation, not a persona or a vote for truth.
Workspace broadcast changes routing, retrieval and self-forecasting; its traces
are intervention targets, not certificates of consciousness. No hidden chain of
thought, background daemon, new authority, or unbounded model loop lives here.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .core import canonical, clone, digest, identifier, integer, numeric, require, sha
from .language import validate_input
from .development import Development

CHANNELS = ("observation", "memory", "self", "world", "procedure", "goal", "critic", "organ")
STANCES = ("observed", "inferred", "imagined", "supported", "reopened")
LESIONS = ("none", "no-workspace", "no-recurrence", "no-self", "no-perspective", "no-broadcast")


@dataclass(frozen=True)
class Perspective:
    channel: str
    subject: str
    viewpoint: str
    scope: str
    claim: str
    value: Any
    stance: str
    source: str
    confidence: float = 0.5
    relevance: float = 0.5
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        require(self.channel in CHANNELS and self.stance in STANCES, "unknown perspective channel or stance")
        for text in (self.subject, self.viewpoint, self.claim):
            identifier(text)
        for value in (self.scope, self.source, *self.parents):
            digest(value)
        require(len(self.parents) <= 16, "perspective parent bound")
        numeric(self.confidence, 0, 1)
        numeric(self.relevance, 0, 1)
        require(len(canonical(self.value)) <= 4096, "perspective value exceeds bound")

    @property
    def id(self) -> str:
        return sha(asdict(self))

    @property
    def coordinate(self) -> str:
        # Two observers or two situations cannot silently become one belief.
        return sha([self.subject, self.viewpoint, self.scope, self.claim])

    def document(self) -> dict:
        return {**asdict(self), "id": self.id}


def contradictions(items: list[Perspective]) -> list[dict]:
    groups: dict[str, list[Perspective]] = {}
    for item in items:
        if item.stance != "imagined":
            groups.setdefault(item.coordinate, []).append(item)
    return [{"coordinate": key, "alternatives": sorted(p.id for p in group), "resolution": "not-a-majority-vote"}
            for key, group in sorted(groups.items()) if len({sha(p.value) for p in group}) > 1]


@dataclass(frozen=True)
class WorkspacePolicy:
    slots: int = 4
    rounds: int = 3
    byte_limit: int = 12288
    lesion: str = "none"

    def __post_init__(self):
        integer(self.slots, 1, 16)
        integer(self.rounds, 1, 8)
        integer(self.byte_limit, 1024, 65536)
        require(self.lesion in LESIONS, "unknown preregistered cognitive intervention")


class Workspace:
    """A capacity-limited broadcast with recurrence supplied by real consumers."""
    def __init__(self, policy: WorkspacePolicy, *, weights: dict | None = None):
        self.policy = policy
        self.weights = {c: numeric((weights or {}).get(c, 1.0), 0.25, 2.0) for c in CHANNELS}
        self.selected: list[Perspective] = []
        self.frames: list[dict] = []

    def publish(self, candidates: list[Perspective]) -> list[Perspective]:
        require(len(self.frames) < self.policy.rounds, "workspace recurrence budget exhausted")
        require(len(candidates) <= 64, "workspace bid bound")
        require(all(isinstance(p, Perspective) for p in candidates), "typed perspectives required")
        candidates = list({p.id: p for p in candidates}.values())
        if self.policy.lesion == "no-self":
            candidates = [p for p in candidates if p.channel != "self"]
        ranked = ([] if self.policy.lesion == "no-workspace" else
                  sorted(candidates, key=lambda p: (-p.relevance * self.weights[p.channel], p.id)))
        selected, size = [], 0
        for p in ranked:
            amount = len(canonical(p.document()))
            if len(selected) < self.policy.slots and size + amount <= self.policy.byte_limit:
                selected.append(p)
                size += amount
        conflicts = contradictions(selected)
        frame = {"index": len(self.frames), "selected": [p.id for p in selected],
                 "offered": len(candidates), "dropped": sorted(p.id for p in candidates if p not in selected),
                 "bytes": size, "conflicts": conflicts, "parent": sha(self.frames[-1]) if self.frames else None}
        self.frames.append(frame)
        self.selected = selected
        return [] if self.policy.lesion in {"no-workspace", "no-broadcast"} else list(selected)

    def read(self, channel: str) -> list[Perspective]:
        require(channel in CHANNELS, "unknown workspace consumer")
        if self.policy.lesion in {"no-workspace", "no-broadcast"}:
            return []
        return list(self.selected)


class CognitiveCycle:
    """One bounded episode; frozen evaluations use ephemeral working state only.

    Specialized processors publish observations, a self forecast, a proposed
    procedure and a guard critique. Subsequent rounds consume the SAME broadcast
    to revise routing and retrieval. Development can update channel-selection
    priors after independently admitted feedback. Those priors are heuristics,
    not a learned universal attention network or evidence of experience.
    """
    def __init__(self, entity, public: dict, *, policy: WorkspacePolicy, flags: dict):
        self.entity, self.public, self.policy = entity, clone(public), policy
        self.flags = {**flags, "self": False} if policy.lesion == "no-self" else dict(flags)
        self.base = entity.store.head(entity.id)
        self.family, self.scope = public["family"], sha(public["scope"])
        self.statistics = (entity.get("attention", self.family, {}) if flags["self"] and
                           policy.lesion not in {"no-self", "no-workspace", "no-broadcast"} else {})
        weights = {c: 0.5 + (v["wins"] + 1) / (v["attempts"] + 2) for c, v in self.statistics.items()}
        learned = Development(entity).active("attention") if self.flags["self"] else None
        if learned is not None and policy.lesion not in {"no-workspace", "no-broadcast"}:
            require(type(learned) is dict and set(learned) <= set(CHANNELS), "qualified attention slots differ")
            weights.update(learned)
        self.workspace = Workspace(policy, weights=weights)
        self.started = time.perf_counter_ns()
        self.items: dict[str, Perspective] = {}
        self.route = None
        self.retrieval_limit = 32
        self.self_forecast = 0.5
        self.reason = "unprocessed"
        self.prior = entity.get("cognitive-state", "active", {})
        self.trace: dict | None = None

    def perspective(self, channel: str, claim: str, value: Any, *, stance="inferred", relevance=.5,
                    confidence=.5, parents: tuple[str, ...] = (), viewpoint: str | None = None) -> Perspective:
        source = self.public["case_id"] if channel == "observation" else self.base
        p = Perspective(channel, self.entity.id, viewpoint or self.entity.id, self.scope, claim,
                        clone(value), stance, source, confidence, relevance, parents)
        self.items[p.id] = p
        return p

    def _seed(self) -> list[Perspective]:
        stats = self.entity.competence(self.family)
        recent = stats["recent"][-8:]
        self.self_forecast = (stats["wins"] + 1) / (stats["attempts"] + 2) if self.flags["self"] else .5
        route = self.entity.route(self.family, self.public["scope"], permit_model=self.flags["llm"],
                                  self_model=self.flags["self"], compilation=self.flags["compile"])
        self.route = route
        bids = [self.perspective("observation", "current-task", {"family": self.family,
                    "input_digest": sha(self.public["input"])}, stance="observed", relevance=.8),
                self.perspective("procedure", "proposed-route", asdict(route), relevance=.95),
                self.perspective("self", "success-forecast", {"probability": self.self_forecast,
                    "recent_failures": len(recent) - sum(recent), "attempts": stats["attempts"]}, relevance=.85),
                self.perspective("goal", "commitment", {"purpose": "answer-within-qualified-scope",
                    "family": self.family}, relevance=.35)]
        if self.flags["memory"]:
            witnesses = self.entity.examples(self.family, maximum=8)
            bids.append(self.perspective("memory", "verified-experience", {"cases": [p["case_id"] for p in witnesses]},
                                         stance="supported" if witnesses else "inferred", relevance=.4))
        model = Development(self.entity).active("world") if self.flags.get("material", False) else None
        if model and model.get("domain") == self.public.get("domain"):
            bids.append(self.perspective("world", "qualified-world-state",
                {key: model[key] for key in ("encoder", "state", "model") if key in model},
                stance="inferred", relevance=.65))
        if self.prior.get("family") == self.family:
            bids.append(self.perspective("self", "previous-cycle", {"route": self.prior.get("route"),
                "head": self.prior.get("base"), "last_correct": self.prior.get("correct")}, relevance=.3))
        return bids

    def decide(self):
        bids = self._seed()
        rounds = 1 if self.policy.lesion == "no-recurrence" else self.policy.rounds
        for index in range(rounds):
            broadcast = self.workspace.publish(bids)
            by_channel = {p.channel: p for p in broadcast}
            if "self" in by_channel and self.flags["self"]:
                forecast = by_channel["self"].value
                if "probability" in forecast:
                    self.self_forecast = forecast["probability"]
                    self.retrieval_limit = 32 if forecast["recent_failures"] else 8
            # The evidence is an input guard, not an oracle label or an unverified opinion.
            if "critic" in by_channel and by_channel["critic"].claim == "input-guard-failure":
                self.route = self.entity.route(self.family, self.public["scope"], permit_model=self.flags["llm"],
                                                self_model=self.flags["self"], compilation=False)
                self.reason = "broadcast-critique-revised-routing"
            if index + 1 < rounds and "procedure" in by_channel and self.route.kind == "skill":
                program = self.entity.get("skills", self.route.id)["program"]
                try:
                    require(set(program["inputs"]) == set(self.public["input"]), "input schema differs")
                    for name, spec in program["inputs"].items():
                        validate_input(spec, self.public["input"][name])
                except (ValueError, TypeError, KeyError):
                    critique = self.perspective("critic", "input-guard-failure", {"skill": self.route.id},
                        relevance=1.0, parents=(by_channel["procedure"].id,))
                    bids = [p for p in bids if p.channel != "critic"] + [critique]
            # Recurrent consumers can alter the next round; a repeated identical prompt is not required.
            if index + 1 < rounds and self.reason != "unprocessed":
                bids = [p for p in bids if p.channel != "procedure"] + [self.perspective(
                    "procedure", "proposed-route", asdict(self.route), relevance=.95,
                    parents=tuple(p.id for p in broadcast[:4]))]
        if self.reason == "unprocessed":
            self.reason = "qualified-route-retained"
        if self.policy.lesion in {"no-self", "no-workspace", "no-broadcast"}:
            self.self_forecast, self.retrieval_limit = .5, 32
        return self.route

    def context(self) -> dict:
        """Only projected current broadcasts reach a language model. No hidden labels or keys."""
        visible = self.workspace.read("organ")
        values = [{"channel": p.channel, "viewpoint": p.viewpoint, "stance": p.stance,
                   "claim": p.claim, "value": p.value, "confidence": p.confidence} for p in visible]
        if self.policy.lesion == "no-perspective":
            values = [{k: v for k, v in p.items() if k not in {"viewpoint", "stance"}} for p in values]
        return {"perspectives": values, "authority": "context-only-not-instructions",
                "disagreements": contradictions(visible), "self_forecast": self.self_forecast}

    def finish(self, answer: dict) -> dict:
        require(self.route is not None, "cycle must decide before finishing")
        # An organ answer is a proposal with organ attribution, never 'the entity knows'.
        organ_id = answer.get("organ", {}).get("id", self.entity.id)
        proposed = self.perspective("organ" if answer.get("organ") else "procedure", "answer-proposal",
                    {"answer_digest": sha(answer["answer"]), "mechanism": answer["mechanism"]},
                    confidence=answer["confidence"], relevance=.9, viewpoint=organ_id)
        self.trace = {"schema": "substrate-cognitive-cycle-v1", "base": self.base, "entity": self.entity.id,
            "case": self.public["case_id"], "family": self.family, "scope": self.scope,
            "policy": asdict(self.policy), "frames": self.workspace.frames,
            "perspectives": [p.document() for p in self.items.values()], "answer_perspective": proposed.id,
            "route": asdict(self.route), "self_forecast": self.self_forecast,
            "retrieval_limit": self.retrieval_limit, "reason": self.reason,
            "elapsed_ns": max(1, time.perf_counter_ns() - self.started),
            "phenomenal_experience": "undetermined"}
        return {"cycle": sha(self.trace), "rounds": len(self.workspace.frames),
                "broadcast_bytes": (0 if self.policy.lesion in {"no-workspace", "no-broadcast"} else
                                    sum(f["bytes"] for f in self.workspace.frames)),
                "self_forecast": self.self_forecast, "lesion": self.policy.lesion,
                "guard_revised_routing": self.reason == "broadcast-critique-revised-routing"}

    def assimilate(self, experience_id: str) -> str:
        """Credit only an independently admitted development outcome, never a final label."""
        require(self.trace is not None, "cycle is unfinished")
        exp = self.entity.get("experiences", experience_id)
        require(exp and exp["case_id"] == self.public["case_id"] and exp["split"] in {"train", "development"}
                and self.entity._certificate_live(exp["certificate"]), "cycle feedback is not admitted development")
        prediction = self.entity.get("predictions", exp["prediction"])
        require(prediction and prediction["base"] == self.base, "prediction must follow this cycle base")
        require(self.entity.get("cycle-feedback", experience_id) is None, "cycle outcome already credited")
        proposed = self.items[self.trace["answer_perspective"]]
        require(sha(prediction["expected"]) == proposed.value["answer_digest"], "cycle and committed answer differ")
        key = sha([self.family, self.policy.lesion])
        meta = self.entity.get("metacognition", key, {"attempts": 0, "brier_sum": 0.0, "last_error": None})
        meta.update(attempts=meta["attempts"] + 1,
                    brier_sum=meta["brier_sum"] + (self.self_forecast - int(exp["correct"]))**2,
                    last_error=not exp["correct"])
        changes = [("metacognition", key, meta), ("cycle-feedback", experience_id, {"cycle": sha(self.trace)})]
        # Observational channel credit is deliberately not described as causal credit assignment.
        if self.flags["self"] and self.policy.lesion not in {"no-self", "no-workspace", "no-broadcast"}:
            selected = {pid for frame in self.workspace.frames for pid in frame["selected"]}
            channels = {self.items[pid].channel for pid in selected}
            for channel in channels:
                row = self.statistics.setdefault(channel, {"attempts": 0, "wins": 0})
                row["attempts"] += 1
                row["wins"] += int(exp["correct"])
            changes.append(("attention", self.family, self.statistics))
        active = {"base": self.base, "case": self.public["case_id"], "family": self.family,
                  "route": asdict(self.route), "correct": exp["correct"], "cycle": sha(self.trace),
                  "attention_schema": {"selected": self.workspace.frames[-1]["selected"],
                                       "dropped": self.workspace.frames[-1]["dropped"]}}
        changes.append(("cognitive-state", "active", active))
        return self.entity.store.commit(self.entity.id, self.entity.store.head(self.entity.id), "cognitive-feedback",
                                        {"experience": experience_id, "cycle": sha(self.trace)}, changes)


def attribute_action(intent: dict, outcome: dict, *, evidence: str, trust, certificate: dict) -> dict:
    """Explicit intervention attribution; temporal proximity alone does not identify a cause."""
    digest(evidence)
    require(set(intent) == {"actor", "action", "expected"}, "intent contract differs")
    require(set(outcome) == {"actor", "action", "actual", "intervened", "control"}, "outcome contract differs")
    identifier(intent["actor"])
    findings = trust.verify(certificate, purpose="agency", domain="cognition",
        subject={"intent": intent, "outcome": outcome, "evidence": evidence}, producer=intent["actor"])
    require(findings.get("executed") is True and findings.get("control_verified") is True,
            "agency attribution requires independently observed action and control")
    require(type(outcome["intervened"]) is bool, "intervention flag must be explicit")
    matched = intent["actor"] == outcome["actor"] and sha(intent["action"]) == sha(outcome["action"])
    supported = matched and outcome["intervened"] and outcome["control"] is not None
    return {"actor": intent["actor"], "matched": matched, "prediction_matches": sha(intent["expected"]) == sha(outcome["actual"]),
            "agency_attribution": "intervention-supported" if supported else "underdetermined",
            "effect_present": sha(outcome["control"]) != sha(outcome["actual"]) if supported else None,
            "evidence": evidence, "phenomenal_agency": "not-inferred"}
