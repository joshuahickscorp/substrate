"""Entity-owned epistemology, developmental learning and behavior-changing self knowledge.

The teacher proposes; this module admits independently attested experience.
The continuing organization, not a model session, owns projects and learned
procedures. All promotions are scoped, revocable and traceable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .core import (Cost, Refused, Trust, canonical, clone, digest, fields, identifier, integer,
                   numeric, require, scope_matches, sha, valid_signature)
from .language import CausalModel, cached_program, compile_program, equivalent, induce, mine_material, abstract_material, transfer_material
from .store import Store

STANCES = {"considered", "imagined", "believed", "supported", "knowledge", "reopened", "retracted"}
DEPENDENT_NAMESPACES = ("beliefs", "skills", "materials")


def reliability(wins: int, attempts: int) -> float:
    """Wilson lower bound used as a routing heuristic, not a research-level confidence interval."""
    integer(wins)
    integer(attempts)
    require(wins <= attempts, "wins exceed attempts")
    if attempts == 0:
        return 0.0
    z, p = 1.96, wins / attempts
    return max(0.0, (p + z * z / (2 * attempts) - z *
                    ((p * (1 - p) + z * z / (4 * attempts)) / attempts)**0.5) /
               (1 + z * z / attempts))


@dataclass(frozen=True)
class Mechanism:
    kind: str
    id: str
    family: str
    confidence: float
    reason: str


class Entity:
    def __init__(self, store: Store, entity: str, trust: Trust, *, host: str):
        require(store.role == "actor", "entity state belongs in an actor store")
        store.entity(entity)
        self.store, self.id, self.trust, self.host = store, entity, trust, digest(host)

    @classmethod
    def create(cls, store: Store, entity: str, trust: Trust, *, host: str,
               goal: str = "Develop verifiable, transferable competence") -> Entity:
        store.create_entity(entity, {"self_modify": False, "external_authority": "operator-permit", "goal": goal})
        return cls(store, entity, trust, host=host)

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        return self.store.record(self.id, namespace, key, default)

    def all(self, namespace: str) -> dict:
        return self.store.records(self.id, namespace)

    def subject(self, operation: str, proposal: Any) -> dict:
        return {"entity": self.id, "base": self.store.head(self.id),
                "operation": operation, "proposal": clone(proposal)}

    def _verify(self, purpose: str, domain: str, proposal: Any, certificate: dict, producer: str) -> dict:
        return self.trust.verify(certificate, purpose=purpose, domain=domain,
                                 subject=self.subject(purpose, proposal), producer=producer)

    def _commit(self, kind: str, payload: Any, changes: list[tuple[str, str, Any]], *, base: str) -> str:
        return self.store.commit(self.id, base, kind, payload, changes)

    def _dependencies(self, dependencies: list[str]) -> None:
        require(type(dependencies) is list and len(dependencies) <= 64
                and len(set(dependencies)) == len(dependencies), "invalid dependency list")
        for key in dependencies:
            require(self.get("beliefs", key) is not None, "unknown belief dependency")

    def _certificate_live(self, certificate: dict | None) -> bool:
        if not certificate:
            return False
        rule = self.trust.rules.get(certificate.get("signer"))
        return bool(rule and not rule.revoked and certificate.get("purpose") in rule.purposes
                    and certificate.get("domain") in rule.domains
                    and certificate["issued"] <= time.time() < certificate["expires"]
                    and valid_signature(canonical(certificate), rule.public_key))

    def live(self, belief_id: str, visiting: set[str] | None = None) -> bool:
        visiting = set() if visiting is None else set(visiting)
        if belief_id in visiting:
            return False
        visiting.add(belief_id)
        row = self.get("beliefs", belief_id)
        return bool(row and row["stance"] in {"supported", "knowledge"}
                    and self._certificate_live(row.get("certificate"))
                    and not any(self.get("revoked-sources", source) for source in row.get("sources", []))
                    and all(self.live(key, visiting) for key in row["dependencies"]))

    def think(self, proposition: str, *, domain: str, scope: dict, stance: str = "considered",
              dependencies: list[str] | None = None) -> str:
        require(stance in {"considered", "imagined", "believed"}, "thought alone cannot become knowledge")
        require(type(proposition) is str and 0 < len(proposition) <= 8192, "proposition size bound")
        require(type(scope) is dict and bool(scope), "belief needs an explicit scope")
        dependencies = dependencies or []
        self._dependencies(dependencies)
        base = self.store.head(self.id)
        row = {"text": proposition, "domain": identifier(domain), "scope": clone(scope),
               "stance": stance, "dependencies": dependencies, "confidence": None,
               "certificate": None, "sources": [], "counterevidence": [], "created": time.time()}
        key = sha(row)
        self._commit("thought", {"belief": key}, [("beliefs", key, row)], base=base)
        return key

    def support(self, belief_id: str, proposal: dict, certificate: dict) -> str:
        fields(proposal, {"belief", "producer", "confidence", "sources", "stance"})
        require(proposal["belief"] == belief_id, "belief mismatch")
        numeric(proposal["confidence"], 0, 1)
        require(proposal["stance"] in {"supported", "knowledge"}, "invalid supported stance")
        require(type(proposal["sources"]) is list and bool(proposal["sources"]), "support needs source identities")
        for source in proposal["sources"]:
            digest(source)
            require(not self.get("revoked-sources", source), "belief source revoked")
        base = self.store.head(self.id)
        row = self.get("beliefs", belief_id)
        require(row is not None and row["stance"] != "retracted", "missing or retracted belief")
        findings = self._verify("belief", row["domain"], proposal, certificate, proposal["producer"])
        require(findings.get("supported") is True and findings.get("scope") == sha(row["scope"]),
                "independent support is absent or out of scope")
        if proposal["stance"] == "knowledge":
            require(findings.get("knowledge_standard") is not None, "knowledge requires an explicit domain standard")
        require(all(self.live(dep) for dep in row["dependencies"]), "support depends on unavailable beliefs")
        row.update(stance=proposal["stance"], confidence=proposal["confidence"], sources=proposal["sources"],
                   certificate=clone(certificate))
        return self._commit("belief-supported", {"belief": belief_id, "certificate": certificate},
                            [("beliefs", belief_id, row)], base=base)

    def reopen(self, belief_id: str, reason: str, *, evidence: str | None = None) -> list[str]:
        require(self.get("beliefs", belief_id) is not None, "unknown belief")
        require(type(reason) is str and 0 < len(reason) <= 4096, "reopening requires a bounded reason")
        if evidence is not None:
            digest(evidence)
        base = self.store.head(self.id)
        invalid = {belief_id}
        beliefs = self.all("beliefs")
        changed = True
        while changed:
            before = len(invalid)
            invalid.update(key for key, row in beliefs.items() if set(row["dependencies"]) & invalid)
            changed = before != len(invalid)
        changes = []
        for key in sorted(invalid):
            row = beliefs[key]
            row["stance"] = "reopened"
            row["counterevidence"].append({"reason": reason, "evidence": evidence, "at": time.time()})
            changes.append(("beliefs", key, row))
        for namespace in ("skills", "materials"):
            for key, row in self.all(namespace).items():
                if set(row.get("dependencies", [])) & invalid:
                    row.update(status="suspended", suspension_reason=reason)
                    changes.append((namespace, key, row))
        self._commit("belief-reopened", {"beliefs": sorted(invalid), "reason": reason, "evidence": evidence},
                     changes, base=base)
        return sorted(invalid)

    def observe(self, observation: dict) -> str:
        fields(observation, {"case_id", "domain", "family", "split", "input", "source", "scope"})
        require(observation["split"] in {"train", "development"}, "sealed evaluation input is not assimilable")
        identifier(observation["domain"])
        identifier(observation["family"])
        digest(observation["source"])
        require(not self.get("revoked-sources", observation["source"]), "observation source revoked")
        require(type(observation["input"]) is dict and type(observation["scope"]) is dict, "invalid observation")
        base = self.store.head(self.id)
        key = sha(observation)
        if self.get("observations", key) is None:
            self._commit("observed", {"observation": key}, [("observations", key, observation)], base=base)
        return key

    def predict(self, observation_id: str, expected: Any, *, producer: str, mechanism: str,
                confidence: float, skill_id: str | None = None) -> str:
        observation = self.get("observations", observation_id)
        require(observation is not None, "prediction needs an observation")
        numeric(confidence, 0, 1)
        identifier(producer)
        base = self.store.head(self.id)
        row = {"observation": observation_id, "expected": clone(expected), "producer": producer,
               "mechanism": mechanism, "confidence": confidence, "skill": skill_id,
               "predicted_at": time.time(), "base": base}
        key = sha(row)
        self._commit("predicted", {"prediction": key}, [("predictions", key, row)], base=base)
        return key

    def assimilate(self, proposal: dict, certificate: dict) -> str:
        fields(proposal, {"prediction", "case_id", "unit", "actual", "producer"})
        base = self.store.head(self.id)
        prediction = self.get("predictions", proposal["prediction"])
        require(prediction is not None and prediction["producer"] == proposal["producer"], "prediction producer mismatch")
        observation = self.get("observations", prediction["observation"])
        require(observation["case_id"] == proposal["case_id"], "experience belongs to a different case")
        require(observation["split"] in {"train", "development"}, "evaluation feedback cannot enter development")
        require(not self.get("revoked-sources", observation["source"]), "experience source revoked")
        findings = self._verify("experience", observation["domain"], proposal, certificate, proposal["producer"])
        require(findings.get("outcome_verified") is True and findings.get("case_id") == proposal["case_id"],
                "outcome has not been independently verified")
        key = sha({"source": observation["source"], "case_id": proposal["case_id"]})
        require(self.get("experiences", key) is None, "same experience cannot increase competence twice")
        correct = equivalent(prediction["expected"], proposal["actual"])
        row = {"case_id": proposal["case_id"], "unit": identifier(proposal["unit"]),
               "domain": observation["domain"], "family": observation["family"], "scope": observation["scope"],
               "input": observation["input"], "actual": clone(proposal["actual"]), "correct": correct,
               "prediction": proposal["prediction"], "confidence": prediction["confidence"],
               "split": observation["split"], "certificate": clone(certificate), "at": time.time()}
        stats = self.competence(row["family"])
        recent = (stats["recent"] + [int(correct)])[-32:]
        stats.update(attempts=stats["attempts"] + 1, wins=stats["wins"] + int(correct), recent=recent,
                     brier_sum=stats["brier_sum"] + (prediction["confidence"] - int(correct))**2,
                     last_seen=time.time())
        index = self.get("experience-index", row["family"], {"keys": []})
        index["keys"] = (index["keys"] + [key])[-64:]
        mechanism_key = sha([row["family"], prediction["skill"] or prediction["mechanism"]])
        mechanism_stats = self.get("mechanism-competence", mechanism_key, {"attempts": 0, "wins": 0, "recent": []})
        mechanism_stats.update(attempts=mechanism_stats["attempts"] + 1, wins=mechanism_stats["wins"] + int(correct),
                               recent=(mechanism_stats["recent"] + [int(correct)])[-32:])
        changes = [("experiences", key, row), ("competence", row["family"], stats),
                   ("experience-index", row["family"], index), ("mechanism-competence", mechanism_key, mechanism_stats)]
        if not correct:
            changes.append(("failures", key, {"family": row["family"], "experience": key,
                                               "mechanism": prediction["mechanism"], "at": time.time()}))
            if prediction["skill"]:
                skill = self.get("skills", prediction["skill"])
                if skill:
                    skill.update(status="suspended", suspension_reason="verified predictive failure")
                    changes.append(("skills", prediction["skill"], skill))
        self._commit("experience-admitted", {"experience": key, "certificate": certificate}, changes, base=base)
        return key

    def competence(self, family: str) -> dict:
        return self.get("competence", family, {"attempts": 0, "wins": 0, "recent": [],
                                               "brier_sum": 0.0, "last_seen": 0.0})

    def examples(self, family: str, *, maximum: int = 32) -> list[dict]:
        integer(maximum, 1, 64)
        keys = self.get("experience-index", family, {"keys": []})["keys"]
        rows = []
        for key in keys:
            row = self.get("experiences", key)
            if row and row["family"] == family and self.experience_live(row):
                rows.append(dict(row, experience_id=key))
        return rows[-maximum:]

    def experience_live(self, row: dict) -> bool:
        prediction = self.get("predictions", row.get("prediction", ""))
        observation = self.get("observations", prediction["observation"]) if prediction else None
        return bool(observation and self._certificate_live(row.get("certificate"))
                    and not self.get("revoked-sources", observation["source"]))

    def propose_skill(self, program: dict, *, family: str, domain: str, scope: dict,
                      witnesses: list[str], producer: str, dependencies: list[str] | None = None,
                      source_ancestry: list[str] | None = None) -> str:
        compiled = compile_program(program)
        require(type(source_ancestry or []) is list and len(source_ancestry or []) <= 256, "source ancestry bound")
        for source in source_ancestry or []:
            digest(source)
            require(not self.get("revoked-sources", source), "ancestral source revoked")
        require(type(scope) is dict and bool(scope), "skill requires explicit scope")
        require(type(witnesses) is list and 2 <= len(set(witnesses)) <= 64,
                "skill needs 2..64 distinct training witnesses")
        self._dependencies(dependencies or [])
        input_digests, sample_ids = set(), []
        for key in witnesses:
            row = self.get("experiences", key)
            require(row and row["family"] == family and row["domain"] == domain
                    and scope_matches(scope, row["scope"]), "witness outside skill scope")
            require(self.experience_live(row), "witness source or authority expired or revoked")
            require(equivalent(compiled.run(row["input"]), row["actual"]), "program does not fit its witnesses")
            input_digests.add(sha(row["input"]))
            sample_ids.append(row["case_id"])
        require(len(input_digests) >= 2, "repeated input is not distinct support")
        base = self.store.head(self.id)
        row = {"program": compiled.document, "program_digest": compiled.digest,
               "family": identifier(family), "domain": identifier(domain), "scope": clone(scope),
               "witnesses": sorted(set(witnesses)), "training_cases": sorted(set(sample_ids)),
               "training_inputs": sorted(input_digests), "dependencies": dependencies or [],
               "producer": identifier(producer), "status": "candidate", "certificate": None,
               "host": None, "cost": None, "created": time.time(), "source_ancestry": sorted(set(source_ancestry or []))}
        key = sha(row)
        index = self.get("skill-index", family, {"keys": []})
        index["keys"] = (index["keys"] + [key])[-16:]
        self._commit("skill-proposed", {"skill": key}, [("skills", key, row), ("skill-index", family, index)], base=base)
        return key

    def qualify_skill(self, skill_id: str, certificate: dict) -> str:
        base = self.store.head(self.id)
        row = self.get("skills", skill_id)
        require(row is not None, "missing skill")
        proposal = {"skill": skill_id, "record": sha(row), "host": self.host}
        findings = self._verify("skill", row["domain"], proposal, certificate, row["producer"])
        require(findings.get("passed") is True and findings.get("split") == "qualification", "skill did not qualify")
        require(findings.get("program") == row["program_digest"] and findings.get("host") == self.host
                and findings.get("scope") == sha(row["scope"]), "qualification bindings differ")
        cases, inputs = findings.get("cases", []), findings.get("input_digests", [])
        require(type(cases) is list and type(inputs) is list and len(set(cases)) >= 8 and len(set(inputs)) >= 8,
                "qualification needs eight distinct cases and inputs")
        for value in cases + inputs:
            digest(value)
        require(len(cases) == len(inputs), "qualification case/input cardinality mismatch")
        if findings.get("validation_mode") == "exhaustive-finite":
            specs = row["program"]["inputs"]
            require(all(spec["type"] == "bool" for spec in specs.values())
                    and len(set(inputs)) == 2 ** len(specs), "finite-domain coverage is not exhaustive")
        else:
            require(not set(cases) & set(row["training_cases"]) and not set(inputs) & set(row["training_inputs"]),
                    "qualification leaks training examples")
        digest(findings.get("oracle"))
        cost = Cost(**findings["cost"])
        require(cost.model_calls == 0 and cost.elapsed_ns > 0, "native qualification must measure execution without a model")
        require(all(self.live(key) for key in row["dependencies"]), "skill depends on unavailable beliefs")
        row.update(status="qualified", certificate=clone(certificate), host=self.host, cost=cost.document())
        return self._commit("skill-qualified", {"skill": skill_id, "certificate": certificate},
                            [("skills", skill_id, row)], base=base)

    def skill_ready(self, row: dict, family: str, scope: dict) -> bool:
        return bool(row["status"] == "qualified" and row["family"] == family
                    and row["host"] == self.host and scope_matches(row["scope"], scope)
                    and self._certificate_live(row.get("certificate"))
                    and all(self.live(dep) for dep in row["dependencies"])
                    and not any(self.get("revoked-sources", source) for source in row.get("source_ancestry", []))
                    and all((w := self.get("experiences", key)) and self.experience_live(w)
                            for key in row["witnesses"]))

    def use(self, skill_id: str, inputs: dict, *, family: str, scope: dict) -> Any:
        row = self.get("skills", skill_id)
        require(row is not None and self.skill_ready(row, family, scope), "skill is not admissible here")
        return cached_program(canonical(row["program"])).run(inputs)

    def route(self, family: str, scope: dict, *, permit_model: bool = True,
              self_model: bool = True, compilation: bool = True, max_latency_ns: int | None = None) -> Mechanism:
        integer(max_latency_ns) if max_latency_ns is not None else None
        if compilation:
            indexed = [(key, self.get("skills", key)) for key in self.get("skill-index", family, {"keys": []})["keys"]]
            candidates = [(key, row) for key, row in indexed if row is not None
                          and self.skill_ready(row, family, scope)
                          and (max_latency_ns is None or row["cost"]["elapsed_ns"] <= max_latency_ns)]
            if candidates:
                key, row = min(candidates, key=lambda pair: (pair[1]["cost"]["elapsed_ns"], pair[0]))
                stats = self.get("mechanism-competence", sha([family, key]), {"attempts": 0, "wins": 0, "recent": []})
                recent = stats["recent"][-8:]
                degraded = self_model and len(recent) >= 4 and sum(recent) / len(recent) < 0.5
                if not degraded:
                    qualified_cases = len(set(row["certificate"]["findings"]["cases"]))
                    confidence = reliability(stats["wins"] + qualified_cases, stats["attempts"] + qualified_cases) if self_model else 0.5
                    return Mechanism("skill", key, family, confidence, "qualified entity-owned procedure")
        if permit_model:
            organs = [(key, row) for key, row in self.all("organs").items()
                      if row["status"] == "qualified" and row["host"] == self.host
                      and self._certificate_live(row.get("certificate")) and "reasoner" in row["roles"]]
            if organs:
                key, _ = min(organs, key=lambda pair: (pair[1].get("estimated_latency_ns", 2**63 - 1), pair[0]))
                return Mechanism("organ", key, family, 0.5, "novelty or insufficient native competence")
        return Mechanism("abstain", "none", family, 0.0, "no qualified mechanism within the declared boundary")

    def attach_organ(self, requirement: dict) -> str:
        fields(requirement, {"id", "roles", "artifact", "representation", "behavior_contract"})
        identifier(requirement["id"])
        require(requirement["representation"] in {"external-model", "NR", "NX"}, "unsupported organ representation")
        require(type(requirement["roles"]) is list and bool(requirement["roles"]), "organ needs roles")
        for role in requirement["roles"]:
            identifier(role)
        digest(requirement["artifact"])
        digest(requirement["behavior_contract"])
        base = self.store.head(self.id)
        row = {**clone(requirement), "status": "candidate", "host": None, "certificate": None}
        self._commit("organ-attached", {"requirement": requirement}, [("organs", requirement["id"], row)], base=base)
        return requirement["id"]

    def bind_organ(self, requirement: dict, certificate: dict, *, producer: str) -> str:
        """Admit a reusable external qualification without inventing an entity-specific signature."""
        fields(requirement, {"id", "roles", "artifact", "representation", "behavior_contract"})
        identifier(requirement["id"])
        digest(requirement["artifact"])
        digest(requirement["behavior_contract"])
        require(requirement["representation"] in {"external-model", "NR", "NX"}, "invalid organ representation")
        require(type(requirement["roles"]) is list and bool(requirement["roles"]), "organ roles required")
        findings = self.trust.verify(certificate, purpose="organ-binding", domain="cognition",
                                     subject={"requirement": requirement, "host": self.host}, producer=producer)
        require(findings.get("passed") is True and findings.get("executed") is True,
                "declarations are not executed organ qualifications")
        latency = integer(findings.get("latency_ns"), 1)
        base = self.store.head(self.id)
        row = {**clone(requirement), "status": "qualified", "host": self.host,
               "certificate": certificate, "estimated_latency_ns": latency}
        return self._commit("organ-bound", {"requirement": requirement}, [("organs", requirement["id"], row)], base=base)

    def qualify_organ(self, organ_id: str, certificate: dict, *, producer: str) -> str:
        base = self.store.head(self.id)
        row = self.get("organs", organ_id)
        require(row is not None, "unknown organ")
        proposal = {"organ": organ_id, "requirement": sha(row), "host": self.host}
        findings = self._verify("organ", "cognition", proposal, certificate, producer)
        require(findings.get("passed") is True and findings.get("host") == self.host,
                "organ binding has not qualified on this host")
        row.update(status="qualified", host=self.host, certificate=clone(certificate),
                   estimated_latency_ns=integer(findings["latency_ns"], 1))
        return self._commit("organ-qualified", {"organ": organ_id, "certificate": certificate},
                            [("organs", organ_id, row)], base=base)

    def project(self, name: str, *, goal: str, next_action: str, status: str = "active") -> str:
        require(status in {"active", "blocked", "complete", "abandoned"}, "invalid project status")
        require(type(goal) is str and bool(goal.strip()) and type(next_action) is str, "invalid project")
        base = self.store.head(self.id)
        old = self.get("projects", name, {})
        row = {"goal": goal, "next_action": next_action, "status": status,
               "created": old.get("created", time.time()), "updated": time.time()}
        return self._commit("project-updated", {"project": name}, [("projects", name, row)], base=base)

    def curriculum(self, families: list[str], *, maximum: int = 8) -> list[dict]:
        """Progress plus uncertainty; irreducible novelty alone cannot monopolize the budget."""
        integer(maximum, 1, 64)
        lessons = []
        for family in families:
            stats = self.competence(family)
            recent = stats["recent"]
            half = len(recent) // 2
            progress = abs(sum(recent[half:]) / max(1, len(recent) - half) -
                           sum(recent[:half]) / max(1, half)) if half else 0.0
            uncertainty = 1 / (1 + stats["attempts"])**0.5
            forgetting = 1 - sum(recent[-8:]) / len(recent[-8:]) if recent else 1.0
            lessons.append({"family": identifier(family), "priority": progress + uncertainty + 0.25 * forgetting,
                            "learning_progress": progress, "uncertainty": uncertainty,
                            "purpose": "learn or repair; never inspect final evaluation feedback"})
        return sorted(lessons, key=lambda row: (-row["priority"], row["family"]))[:maximum]

    def learn_scalar(self, family: str, *, inputs: dict, domain: str, scope: dict,
                     max_candidates: int = 5000, seconds: float = 0.25, material: bool = True) -> dict:
        integer(max_candidates, 1, 100000)
        numeric(seconds, .001, 60)
        examples = self.examples(family)
        require(2 <= len(examples) <= 64, "learning requires verified witnesses")
        samples = [{"input": row["input"], "output": row["actual"]} for row in examples]
        started = time.monotonic()
        search = {"program": None, "examined": 0, "origin": None, "qualified": False}
        if material and len(samples) >= 2:
            templates = [r for r in self.all("materials").values() if r.get("kind") == "parameterized-abstraction"
                         and not any(self.get("revoked-sources", key) for key in r.get("source_ancestry", []))][:64]
            search = transfer_material(templates, samples, inputs, max_candidates=max(1, max_candidates // 4), seconds=seconds / 4)
        remaining = seconds - (time.monotonic() - started)
        if search["program"] is None and search["examined"] < max_candidates and remaining >= .001:
            prior = search["examined"]
            search = induce(samples, inputs, max_candidates=max_candidates - prior, seconds=remaining)
            search["examined"] += prior
        search["elapsed_ns"] = int((time.monotonic() - started) * 1e9)
        search["status"] = "candidate" if search["program"] else "no-fit-within-budget"
        receipt = {"family": family, "witnesses": sha(samples), "examined": search["examined"],
                   "elapsed_ns": search["elapsed_ns"], "origin": search.get("origin"), "status": search["status"],
                   "program": sha(search["program"]) if search["program"] else None}
        self._commit("native-search", receipt, [("searches", sha(receipt), receipt)], base=self.store.head(self.id))
        if search["program"] is not None:
            search["skill"] = self.propose_skill(search["program"], family=family, domain=domain, scope=scope,
                                                 witnesses=[row["experience_id"] for row in examples],
                                                 producer="bounded-inducer",
                                                 source_ancestry=(self.get("materials", search.get("origin"), {}) or {}).get("source_ancestry", []) if search.get("origin") else [])
        return search

    def consolidate(self) -> list[str]:
        skills = [row for row in self.all("skills").values() if self.skill_ready(row, row["family"], row["scope"])]
        programs = [row["program"] for row in skills]
        sources = {}
        for row in skills:
            origin = set(row.get("source_ancestry", []))
            for witness in row["witnesses"]:
                prediction = self.get("predictions", self.get("experiences", witness)["prediction"])
                origin.add(self.get("observations", prediction["observation"])["source"])
            for dependency in row["dependencies"]:
                origin.update(self.get("beliefs", dependency)["sources"])
            sources.setdefault(row["program_digest"], set()).update(origin)
        proposals = mine_material(programs) + abstract_material(programs)
        base, changes, ids = self.store.head(self.id), [], []
        for proposal in proposals:
            key = proposal["id"]
            if self.get("materials", key) is None:
                row = {**proposal, "dependencies": [], "certificate": None,
                       "source_ancestry": sorted(set().union(*(sources.get(key, set()) for key in proposal["programs"])))}
                changes.append(("materials", key, row))
                ids.append(key)
        if changes:
            self._commit("consolidated", {"candidates": ids}, changes, base=base)
        return ids

    def propose_world(self, model: dict, *, domain: str, scope: dict, producer: str,
                      dependencies: list[str] | None = None) -> str:
        CausalModel(model)
        self._dependencies(dependencies or [])
        require(type(scope) is dict and bool(scope), "world model needs scope")
        base = self.store.head(self.id)
        row = {"kind": "causal-model", "model": clone(model), "domain": identifier(domain), "scope": scope,
               "producer": identifier(producer), "dependencies": dependencies or [], "status": "candidate",
               "certificate": None}
        key = sha(row)
        self._commit("world-proposed", {"material": key}, [("materials", key, row)], base=base)
        return key

    def qualify_world(self, material_id: str, certificate: dict) -> str:
        base = self.store.head(self.id)
        row = self.get("materials", material_id)
        require(row and row["kind"] == "causal-model", "missing causal material")
        findings = self._verify("world", row["domain"], {"material": material_id, "record": sha(row)},
                                certificate, row["producer"])
        require(findings.get("interventions_verified") is True and findings.get("scope") == sha(row["scope"]),
                "observational fit does not qualify intervention semantics")
        row.update(status="qualified", certificate=clone(certificate))
        return self._commit("world-qualified", {"material": material_id, "certificate": certificate},
                            [("materials", material_id, row)], base=base)

    def predict_world(self, material_id: str, exogenous: dict, *, scope: dict,
                      intervention: dict | None = None) -> dict:
        row = self.get("materials", material_id)
        require(row and row["kind"] == "causal-model" and row["status"] == "qualified"
                and scope_matches(row["scope"], scope) and self._certificate_live(row["certificate"])
                and all(self.live(key) for key in row["dependencies"]), "world model is not admissible")
        return CausalModel(row["model"]).predict(exogenous, do=intervention)

    def resume(self, certificate: dict, *, producer: str = "continuation-requester") -> str:
        """Continue the same identity after an operator-reviewed single-writer handoff.

        This is not a distributed lease. The operator must stop the old writer.
        Changed-host skills/organs fail their existing host guards until requalified.
        """
        head = self.store.head(self.id)
        subject = {"entity": self.id, "head": head, "host": self.host, "operation": "continue"}
        findings = self.trust.verify(certificate, purpose="continuity", domain="entity", subject=subject, producer=producer)
        require(findings.get("previous_writer_stopped") is True and findings.get("approved") is True,
                "continuation requires an explicit single-writer handoff")
        return self.store.continue_entity(self.id, head, {"host": self.host, "certificate": certificate})

    def status(self) -> dict:
        return {"entity": self.id, "head": self.store.head(self.id),
                "frozen": bool(self.store.entity(self.id)["frozen"]),
                "counts": {name: len(self.all(name)) for name in ("beliefs", "experiences", "skills", "materials", "projects")},
                "research_target": "locally-owned-general-intelligence-and-consciousness-research", "program": "substrate-odyssey",
                "claims": {"phenomenal_experience": "undetermined", "AGI": "not-established",
                           "developmental_efficacy": "requires-independent-Odyssey-evidence"}}
