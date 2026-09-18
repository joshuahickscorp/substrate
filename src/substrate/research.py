"""One Odyssey, one explicit target, independently evidenced internal chapters.

General intelligence and full consciousness are research targets, not synonyms for
passing a toy suite. Capability, consciousness indicators and phenomenal status
remain separate outputs. A chapter is a dependency in ONE continuing program;
no second Odyssey, target downgrade or automatic consciousness score is created.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from .core import clone, digest, fields, identifier, integer, numeric, require, sha
from .workspace import LESIONS, WorkspacePolicy

TARGET = "locally-owned-general-intelligence-and-consciousness-research"
AXES = {
    "generality": "Acquire and transfer skills across independently designed unfamiliar domains",
    "development": "Verified history changes abstractions, procedures and subsequent learning efficiency",
    "integration": "Selective broadcast causally connects perception, memory, self-monitoring and action",
    "recurrence": "Recurrent processing carries task-relevant state; lesions expose what is lost",
    "perspective": "Track scoped first-person, other-agent, organ and counterfactual representations without conflation",
    "metacognition": "Prospective self forecasts are calibrated and alter appropriate actions",
    "agency": "Distinguish intended, executed, externally imposed and counterfactually controlled outcomes",
    "continuity": "Projects and earned competence survive sleep, replacement and qualified migration",
    "economics": "Acquisition and execution utility survive complete storage, compute and model accounting",
}
CHAPTERS = {
    "mechanisms": {"requires": [], "axes": ["development", "economics"]},
    "integration": {"requires": ["mechanisms"], "axes": ["integration", "recurrence", "perspective", "metacognition"]},
    "apprenticeship": {"requires": ["mechanisms"], "axes": ["generality", "agency", "development"]},
    "plasticity": {"requires": ["apprenticeship"], "axes": ["development", "economics", "continuity"]},
    "continuity": {"requires": ["integration", "apprenticeship"], "axes": ["continuity", "agency"]},
    "closure": {"requires": ["plasticity", "continuity"], "axes": list(AXES)},
}


def program_plan() -> dict:
    return {"schema": "substrate-one-odyssey-v2", "id": "substrate-odyssey", "target": TARGET,
        "status": "draft", "chapters": clone(CHAPTERS), "axes": clone(AXES),
        "phenomenal_question": "Whether this organization has any subjective experience remains an empirical and theoretical question",
        "consciousness_target": "Full consciousness is investigated without presuming phenomenology from capability, labels, or self-report; theory-specific support and counterevidence remain separate",
        "not_redefined_as": ["fluent self-report", "memory database", "small-world success", "architecture labels"],
        "theory_profiles": ["global-workspace", "recurrent-processing", "higher-order", "attention-schema",
                            "predictive-processing", "biological-naturalist-challenge"],
        "required_controls": ["same-organ", "memory-only", "exact-cache", "equal-acquisition-budget",
                              "no-workspace", "no-recurrence", "no-self", "no-perspective", "no-broadcast"],
        "default_workspace": asdict(WorkspacePolicy()),
        "prohibited_shortcuts": ["counting-passing-indicators-as-consciousness", "training-on-final-labels",
                                  "unmeasured-size-as-intelligence", "self-report-as-verification"],
        "review_policy": {"phenomenal_claims": "independent-theory-aware-review", "welfare": "review-before-scaled-agentic-experiments",
                          "synthetic_distress_objective": False, "shutdown_evasion_objective": False},
        "completion": "Complete outcome coverage and independent interpretation of all target axes, not merely completed jobs"}


def validate_program(document: dict) -> dict:
    reference = program_plan()
    fields(document, set(reference))
    require(document["schema"] == reference["schema"] and document["id"] == reference["id"]
            and document["target"] == TARGET, "one Odyssey and its target cannot be silently renamed")
    require(document["status"] in {"draft", "locked"}, "unknown program lock state")
    require(document["chapters"] == reference["chapters"] and document["axes"] == reference["axes"],
            "changing the research target requires a new reviewed charter revision")
    require(all(document[k] == reference[k] for k in reference if k not in {"status", "default_workspace"}),
            "changing target language or review boundaries requires an explicit charter revision")
    WorkspacePolicy(**document["default_workspace"])
    return clone(document)


def study_profile(plan: dict) -> dict:
    """Legacy v2 plans remain usable but cannot silently receive the stronger target label."""
    profile = plan.get("cognition")
    if profile is None:
        return {"enabled": False, "workspace": asdict(WorkspacePolicy()), "program": None,
                "chapter": "mechanisms", "target": "legacy-mechanism-study"}
    fields(profile, {"enabled", "workspace", "program", "chapter", "target"})
    require(type(profile["enabled"]) is bool and profile["chapter"] in CHAPTERS, "invalid cognitive study profile")
    require(profile["program"] == "substrate-odyssey" and profile["target"] == TARGET, "study target differs from Odyssey")
    WorkspacePolicy(**profile["workspace"])
    return clone(profile)


class OdysseyLedger:
    """Append-only program evidence. Signatures authenticate claims, not consciousness."""
    def __init__(self, store, trust, charter: dict):
        self.store, self.trust, self.charter = store, trust, validate_program(charter)
        require(store.role == "actor", "program ledger requires its own actor store")
        self.id = "odyssey-program"
        found = store.db.execute("SELECT 1 FROM entities WHERE id=?", (self.id,)).fetchone()
        if not found:
            store.create_entity(self.id, {"self_modify": False, "external_authority": "operator-permit", "goal": TARGET})
            store.commit(self.id, store.head(self.id), "charter", {"charter": sha(charter)}, [("charter", "current", charter)])
        else:
            require(store.record(self.id, "charter", "current") == charter, "existing Odyssey charter differs")

    def admit(self, record: dict, certificate: dict) -> str:
        fields(record, {"chapter", "design", "source", "artifacts", "histories", "domains", "axes", "lesions",
                        "complete", "controls", "costs", "outcome_digest", "producer", "evaluation_only"})
        require(record["chapter"] in CHAPTERS and record["complete"] is True, "chapter outcome coverage is incomplete")
        require(self.charter["status"] == "locked", "lock the reviewed program before admitting research evidence")
        for field in ("design", "source", "outcome_digest"):
            digest(record[field])
        for name in ("artifacts", "histories", "domains", "controls"):
            require(type(record[name]) is list and len(record[name]) == len(set(record[name])) and bool(record[name]),
                    "unique explicit evidence factors required")
            for value in record[name]:
                digest(value) if name == "artifacts" else identifier(value)
        require(set(record["axes"]) == set(CHAPTERS[record["chapter"]]["axes"]), "chapter axis coverage is incomplete")
        require(type(record["evaluation_only"]) is bool and record["evaluation_only"], "final evidence must not teach the entity")
        require(type(record["costs"]) is dict and set(record["lesions"]) <= set(LESIONS), "invalid costs or lesions")
        for axis, result in record["axes"].items():
            fields(result, {"status", "endpoint", "control", "estimate", "interval", "limitation"})
            require(result["status"] in {"supported", "contradicted", "inconclusive"}, "invalid evidence status")
            require(type(result["limitation"]) is str and bool(result["limitation"]), "scope and limitations are mandatory")
            numeric(result["estimate"])
            require(type(result["interval"]) is list and len(result["interval"]) == 2, "endpoint interval required")
            numeric(result["interval"][0])
            numeric(result["interval"][1], result["interval"][0])
        findings = self.trust.verify(certificate, purpose="chapter", domain="odyssey", subject=record,
                                     producer=record["producer"])
        require(findings.get("independently_reviewed") is True and findings.get("executed") is True
                and findings.get("outcomes") == record["outcome_digest"], "declarations do not qualify chapters")
        chapters = self.store.records(self.id, "chapters")
        completed = set()
        for prior in chapters.values():
            r = prior["record"]
            try:
                self.trust.verify(prior["certificate"], purpose="chapter", domain="odyssey", subject=r, producer=r["producer"])
                completed.add(r["chapter"])
            except ValueError:
                continue
        require(set(CHAPTERS[record["chapter"]]["requires"]) <= completed, "chapter evidence dependency is missing")
        key = sha(record)
        require(key not in chapters, "same chapter evidence cannot be counted twice")
        return self.store.commit(self.id, self.store.head(self.id), "chapter-evidence", {"record": key},
                                 [("chapters", key, {"record": record, "certificate": certificate})])

    def assessment(self) -> dict:
        rows = self.store.records(self.id, "chapters")
        axes = {axis: {"support": [], "counterevidence": [], "inconclusive": []} for axis in AXES}
        covered = set()
        for key, row in rows.items():
            record, certificate = row["record"], row["certificate"]
            try:
                self.trust.verify(certificate, purpose="chapter", domain="odyssey", subject=record, producer=record["producer"])
            except ValueError:
                continue  # Revocation and expiry remove current authority, never historical records.
            covered.add(record["chapter"])
            for axis, result in record["axes"].items():
                bucket = {"supported": "support", "contradicted": "counterevidence", "inconclusive": "inconclusive"}[result["status"]]
                axes[axis][bucket].append({"record": key, "artifacts": record["artifacts"],
                                          "domains": record["domains"], "endpoint": result["endpoint"]})
        return {"program": self.charter["id"], "target": TARGET, "head": self.store.head(self.id), "axes": axes,
                "chapters": {name: ("evidenced-not-necessarily-supported" if name in covered else
                    "ready-for-instrument-qualification" if set(spec["requires"]) <= covered else "waiting-for-evidence")
                    for name, spec in CHAPTERS.items()},
                "missing_axes": [k for k, v in axes.items() if not any(v.values())],
                "phenomenal_consciousness": "undetermined", "target_attainment": "requires-independent-synthesis",
                "warning": "Evidence from different artifacts or domains is not automatically combinable into one qualified entity"}


def propose_inquiry(entity, family: str, *, scope: dict, source_allowlist: list[str], budget: dict) -> str:
    """Failure-driven project proposal. It cannot grant itself tools or approve its own oracle."""
    identifier(family)
    fields(budget, {"actions", "seconds"})
    integer(budget["actions"], 1, 256)
    numeric(budget["seconds"], 1, 86400)
    require(bool(source_allowlist) and len(source_allowlist) <= 64, "inquiry needs admissible source identities")
    for source in source_allowlist:
        digest(source)
    stats = entity.competence(family)
    goal = {"family": family, "scope": clone(scope), "sources": sorted(set(source_allowlist)),
            "budget": budget, "trigger": {"attempts": stats["attempts"], "recent": stats["recent"][-8:]},
            "status": "proposed", "next": "independent-source-and-oracle-admission", "steps": [],
            "created": time.time()}
    key = sha(goal)
    entity.store.commit(entity.id, entity.store.head(entity.id), "inquiry-proposed", {"inquiry": key}, [("inquiries", key, goal)])
    return key


def admit_apprenticeship(entity, inquiry: str, proposal: dict, certificate: dict) -> str:
    """Bind a real repository, proof or simulation instrument; no generic shell allowance."""
    fields(proposal, {"inquiry", "adapter", "domain", "source", "license", "oracle", "actions", "producer"})
    row = entity.get("inquiries", inquiry)
    require(row and row["status"] == "proposed" and proposal["inquiry"] == inquiry, "unknown inquiry proposal")
    for field in ("adapter", "source", "oracle"):
        digest(proposal[field])
    require(proposal["source"] in row["sources"] and type(proposal["license"]) is str and bool(proposal["license"]),
            "unreviewed apprenticeship source")
    require(type(proposal["actions"]) is list and 1 <= len(proposal["actions"]) <= 16, "bounded action schema required")
    for action in proposal["actions"]:
        identifier(action)
    findings = entity.trust.verify(certificate, purpose="apprenticeship", domain=proposal["domain"],
                                   subject=entity.subject("apprenticeship", proposal), producer=proposal["producer"])
    require(findings.get("isolation_reviewed") is True and findings.get("oracle_independent") is True
            and findings.get("executed_canary") is True, "apprenticeship instrument not qualified")
    row.update(status="admitted", instrument=clone(proposal), certificate=certificate)
    return entity.store.commit(entity.id, entity.store.head(entity.id), "apprenticeship-admitted", {"inquiry": inquiry},
                               [("inquiries", inquiry, row)])


def apprenticeship_step(entity, inquiry: str, action: dict, public: dict, expected, *, confidence: float) -> dict:
    """Prepare a durable, prospective bounded action for an already admitted site adapter.

    Returns a transport envelope. Only a separately qualified dispatcher may
    perform the action; the same independent experience-admission path handles
    the outcome. This does not pretend to implement Lean, a browser or a repo OS.
    """
    row = entity.get("inquiries", inquiry)
    require(row and row["status"] == "admitted" and entity._certificate_live(row["certificate"]), "inquiry not admitted")
    fields(action, {"kind", "arguments"})
    require(action["kind"] in row["instrument"]["actions"] and type(action["arguments"]) is dict,
            "action outside admitted instrument")
    require(len(row["steps"]) < row["budget"]["actions"]
            and time.time() - row["created"] < row["budget"]["seconds"], "inquiry budget exhausted")
    require(public["source"] == row["instrument"]["source"] and public["domain"] == row["instrument"]["domain"]
            and public["family"] == row["family"] and public["scope"] == row["scope"], "task outside inquiry scope")
    observation = entity.observe(public)
    prediction = entity.predict(observation, expected, producer="apprentice", mechanism="bounded-action",
                                 confidence=confidence)
    row["steps"].append({"prediction": prediction, "action": clone(action), "case": public["case_id"]})
    entity.store.commit(entity.id, entity.store.head(entity.id), "apprenticeship-step", {"inquiry": inquiry},
                        [("inquiries", inquiry, row)])
    return {"inquiry": inquiry, "instrument": row["instrument"]["adapter"], "action": action,
            "prediction": prediction, "case": public["case_id"], "head": entity.store.head(entity.id),
            "execution_authorized_by_this_envelope": False}
