"""Odyssey: a gated, resumable developmental experiment, not a success narrative.

Plan/preflight are inert. Initialization and workers require an operator-signed
permit bound to the exact source, plan, host and service bindings. Main runs
add independently signed qualification gates. Frozen final evaluations never
write feedback into the entity, retrieval cache or teacher context.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cognition import Entity
from .core import (Budget, Cost, Refused, Trust, canonical, clone, digest, fields, identifier, integer,
                   numeric, require, sha, source_digest)
from .language import compile_program, equivalent
from .organs import Broker, OracleClient, Organ, TransportError
from .sandbox import SandboxConfig
from .store import Store
from .worlds import FAMILIES, WORLD_VERSION, observation
from .workspace import CognitiveCycle, WorkspacePolicy
from .research import study_profile, TARGET
from .handoff import phenotype
from .development import Development, LearningPolicy

ARMS = {
    "model-only": {"memory": False, "cache": False, "compile": False, "self": False, "material": False, "teacher": False, "llm": True},
    "memory-only": {"memory": True, "cache": False, "compile": False, "self": False, "material": False, "teacher": False, "llm": True},
    "cache-only": {"memory": False, "cache": True, "compile": False, "self": False, "material": False, "teacher": False, "llm": True},
    "native-only": {"memory": True, "cache": False, "compile": True, "self": True, "material": True, "teacher": False, "llm": False},
    "developed": {"memory": True, "cache": False, "compile": True, "self": True, "material": True, "teacher": True, "llm": True},
    "no-self-model": {"memory": True, "cache": False, "compile": True, "self": False, "material": True, "teacher": True, "llm": True},
    "no-compilation": {"memory": True, "cache": False, "compile": False, "self": True, "material": True, "teacher": True, "llm": True},
    "no-material": {"memory": True, "cache": False, "compile": True, "self": True, "material": False, "teacher": True, "llm": True},
    "no-teacher": {"memory": True, "cache": False, "compile": True, "self": True, "material": True, "teacher": False, "llm": True},
}
for _lesion in ("no-workspace", "no-recurrence", "no-perspective", "no-broadcast"):
    ARMS[_lesion] = {**ARMS["developed"], "lesion": _lesion}
PHASES = {
    "train": "Prospective predictions, independent feedback, bounded induction, teacher proposals and qualification",
    "freeze": "Consolidate candidate material and freeze the entity before final feedback",
    "evaluation": "Unseen input generalization under the initial physical/organ binding",
    "transfer": "Larger-magnitude or composition-distribution transfer; no additional learning",
    "retention": "Retention probes after the full mixed-family developmental history",
    "shift": "Undisclosed mechanism changes; measure failure and confidence under misspecification",
    "organ_swap": "Replace the neural service while preserving the frozen cognitive entity",
}


def draft_plan(*, kind: str = "pilot") -> dict:
    require(kind in {"pilot", "main"}, "invalid study kind")
    return {"schema": "odyssey-plan-v2", "status": "draft", "kind": kind, "world": WORLD_VERSION,
            "units": [f"history-{i:03}" for i in range(2 if kind == "pilot" else 16)],
            "arms": ["model-only", "memory-only", "developed"] if kind == "pilot" else list(ARMS),
            "families": ["affine", "queue"] if kind == "pilot" else list(FAMILIES),
            "episodes": {"train": 12, "evaluation": 8, "transfer": 8, "retention": 8, "shift": 8, "organ_swap": 0},
            "qualification_cases": 16, "qualification_attempts": 3, "compile_every": 4,
            "induction_candidates": 5000, "induction_seconds": 0.25,
            "budget": Budget().document(), "successful_trace_every": 32,
            "primary_contrast": ["developed", "memory-only"],
            "cognition": {"enabled": True, "program": "substrate-odyssey", "target": TARGET,
                          "chapter": "mechanisms", "workspace": asdict(WorkspacePolicy())},
            "development": {"enabled": True, "policy": asdict(LearningPolicy()), "neural_jobs": "separately-authorized"},
            "preregistration": {"question": "Does verified experience become reusable entity-owned structure?",
                "primary_outcome": "evaluation_accuracy", "sample_size_rationale": None,
                "independent_unit": "developmental-history", "analysis": "fixed-horizon-paired-history-bootstrap",
                "source_license_review": False, "oracle_review": False, "isolation_review": False,
                "model_prior_disclosure": None, "stopping": "complete fixed matrix or report incomplete, not a efficacy claim"}}


def validate_plan(plan: dict) -> dict:
    fields(plan, {"schema", "status", "kind", "world", "units", "arms", "families", "episodes", "qualification_cases",
                  "qualification_attempts", "compile_every", "induction_candidates", "induction_seconds", "budget",
                  "successful_trace_every", "primary_contrast", "preregistration"}, {"cognition", "development"})
    study_profile(plan)
    if "development" in plan:
        fields(plan["development"], {"enabled", "policy", "neural_jobs"})
        require(type(plan["development"]["enabled"]) is bool and plan["development"]["neural_jobs"] == "separately-authorized", "invalid development profile")
        LearningPolicy(**plan["development"]["policy"])
    require(plan["schema"] == "odyssey-plan-v2" and plan["kind"] in {"pilot", "main"}, "unknown plan")
    require(plan["world"] == WORLD_VERSION and plan["status"] in {"draft", "locked"}, "unknown world or lock state")
    for name, allowed in (("arms", set(ARMS)), ("families", set(FAMILIES))):
        values = plan[name]
        require(type(values) is list and 1 <= len(values) <= len(allowed) and len(set(values)) == len(values)
                and set(values) <= allowed, "invalid or duplicated study factors")
    require(type(plan["units"]) is list and 2 <= len(plan["units"]) <= 1000, "plan needs 2..1000 independent histories")
    require(len(set(plan["units"])) == len(plan["units"]), "duplicate independent unit")
    for unit in plan["units"]:
        identifier(unit)
        require(len(unit) <= 32, "history identity too long")
    fields(plan["episodes"], set(PHASES) - {"freeze"})
    for split, count in plan["episodes"].items():
        integer(count, 0 if split == "organ_swap" else 1, 4096)
    trials = len(plan["units"]) * len(plan["arms"]) * len(plan["families"]) * sum(plan["episodes"].values())
    require(trials <= 1_000_000, "study matrix exceeds bounded analysis capacity; preregister internal studies within the one Odyssey")
    integer(plan["qualification_cases"], 8, 64)
    integer(plan["qualification_attempts"], 1, 64)
    integer(plan["compile_every"], 2, 64)
    integer(plan["induction_candidates"], 1, 100000)
    numeric(plan["induction_seconds"], 0.001, 60)
    integer(plan["successful_trace_every"], 1, 100000)
    Budget(**plan["budget"])
    require(len(plan["primary_contrast"]) == 2 and len(set(plan["primary_contrast"])) == 2
            and set(plan["primary_contrast"]) <= set(plan["arms"]), "primary comparison must be registered")
    fields(plan["preregistration"], {"question", "primary_outcome", "sample_size_rationale", "independent_unit",
          "analysis", "source_license_review", "oracle_review", "isolation_review", "model_prior_disclosure", "stopping"})
    require(plan["preregistration"]["primary_outcome"] == "evaluation_accuracy"
            and plan["preregistration"]["analysis"] == "fixed-horizon-paired-history-bootstrap"
            and plan["preregistration"]["independent_unit"] == "developmental-history", "unimplemented analysis contract")
    return clone(plan)


def organ_requirement(organ: Organ) -> dict:
    contract = (organ.nr or {}).get("behavior_contract", sha({"schema": "json-answer-and-closed-ir-v2"}))
    return {"id": organ.id, "roles": list(organ.roles), "artifact": organ.artifact,
            "representation": {"model": "external-model", "nr": "NR", "nx": "NX"}[organ.representation],
            "behavior_contract": contract}


def validate_bindings(bindings: dict) -> dict:
    fields(bindings, {"schema", "host", "organ", "replacement", "evaluator", "sandbox", "qualifications"})
    require(bindings["schema"] == "odyssey-bindings-v2", "unknown binding schema")
    digest(bindings["host"])
    for value in (bindings["organ"], bindings["replacement"]):
        if value is not None:
            Organ(**value)
    require(not (bindings["organ"] and bindings["replacement"])
            or bindings["organ"]["id"] != bindings["replacement"]["id"], "replacement organ needs a distinct identity")
    fields(bindings["evaluator"], {"endpoint", "token_env"})
    OracleClient(**bindings["evaluator"])
    SandboxConfig(**bindings["sandbox"])
    require(type(bindings["qualifications"]) is list, "qualifications must be a list")
    return clone(bindings)


def run_subject(plan: dict, bindings: dict, root: Path) -> dict:
    return {"kind": "odyssey-run", "plan": sha(plan), "source": source_digest(root),
            "bindings": sha(bindings), "host": bindings["host"], "budget": plan["budget"]}


def preflight(plan: dict, bindings: dict, root: Path, trust: Trust | None = None) -> dict:
    validate_plan(plan)
    validate_bindings(bindings)
    missing = []
    runtime = Path(__file__).resolve().parent
    claimed = root.resolve() / "src" / "substrate"
    runtime_paths = {p.relative_to(runtime) for p in runtime.rglob("*.py")}
    claimed_paths = {p.relative_to(claimed) for p in claimed.rglob("*.py")}
    if runtime_paths != claimed_paths or any((runtime / p).read_bytes() != (claimed / p).read_bytes() for p in runtime_paths & claimed_paths):
        missing.append("claimed source root differs from the loaded runtime source")
    if plan["status"] != "locked":
        missing.append("plan is not locked")
    if any(ARMS[arm]["llm"] for arm in plan["arms"]) and bindings["organ"] is None:
        missing.append("explicit, pinned neural organ")
    if bindings["organ"] is not None and "reasoner" not in bindings["organ"]["roles"]:
        missing.append("reasoner role on the primary neural organ")
    if bindings["replacement"] is not None and "reasoner" not in bindings["replacement"]["roles"]:
        missing.append("reasoner role on the replacement organ")
    if any(ARMS[arm]["teacher"] for arm in plan["arms"]):
        if not bindings["organ"] or "teacher" not in bindings["organ"]["roles"]:
            missing.append("teacher role on an explicit neural organ")
    if plan["episodes"]["organ_swap"] and bindings["replacement"] is None:
        missing.append("replacement organ for organ-swap phase")
    for document in (bindings["organ"], bindings["replacement"]):
        if document is not None:
            organ = Organ(**document)
            subject = {"requirement": organ_requirement(organ), "host": bindings["host"]}
            valid = False
            for certificate in bindings["qualifications"]:
                if certificate.get("purpose") == "organ-binding" and trust is not None:
                    try:
                        findings = trust.verify(certificate, purpose="organ-binding", domain="cognition", subject=subject,
                                                producer=certificate["producer"])
                        valid = findings.get("passed") is True and findings.get("executed") is True
                    except Refused:
                        continue
                    if valid:
                        break
            if not valid:
                missing.append("independently qualified organ binding: " + organ.id)
    if plan["kind"] == "main":
        prereg = plan["preregistration"]
        for key in ("source_license_review", "oracle_review", "isolation_review"):
            if prereg[key] is not True:
                missing.append(key)
        for key in ("sample_size_rationale", "model_prior_disclosure"):
            if not isinstance(prereg[key], str) or not prereg[key].strip():
                missing.append(key)
        if bindings["sandbox"]["mode"] != "oci":
            missing.append("qualified digest-pinned OCI sandbox")
        for purpose in ("canary", "oracle", "isolation", "analysis-contract"):
            subject = {"gate": purpose, "source": source_digest(root), "host": bindings["host"], "world": WORLD_VERSION}
            certificates = [c for c in bindings["qualifications"] if c.get("purpose") == purpose]
            valid = False
            for certificate in certificates:
                if trust is not None:
                    try:
                        findings = trust.verify(certificate, purpose=purpose, domain="odyssey", subject=subject,
                                                producer=certificate["producer"])
                        valid = findings.get("passed") is True and findings.get("executed") is True
                    except Refused:
                        continue
                if valid:
                    break
            if not valid:
                missing.append(f"independent executed {purpose} qualification")
    episodes = len(plan["units"]) * len(plan["arms"]) * len(plan["families"]) * sum(plan["episodes"].values())
    # A conservative upper bound; native reuse may reduce it but is not assumed in funding a run.
    neural_arms = sum(ARMS[arm]["llm"] for arm in plan["arms"])
    prediction_calls = len(plan["units"]) * neural_arms * len(plan["families"]) * sum(plan["episodes"].values())
    teachers = sum(ARMS[arm]["teacher"] for arm in plan["arms"])
    teacher_calls = len(plan["units"]) * teachers * len(plan["families"]) * (plan["episodes"]["train"] // plan["compile_every"])
    return {"schema": "odyssey-preflight-v2", "subject": run_subject(plan, bindings, root),
            "missing": missing, "ready_for_operator_permit": not missing, "network_contacted": False,
            "planned_trials": episodes, "conservative_model_calls": prediction_calls + teacher_calls,
            "call_budget_covers_upper_bound": plan["budget"]["calls"] >= prediction_calls + teacher_calls,
            "statistical_unit": "history, not episode", "efficacy": "not-evaluated"}


class Gate:
    def __init__(self, plan: dict, bindings: dict, root: Path, trust: Trust, permit: dict):
        self.plan, self.bindings, self.root, self.trust, self.permit = plan, bindings, root, trust, permit
        self.check()

    def check(self) -> None:
        check = preflight(self.plan, self.bindings, self.root, self.trust)
        require(not check["missing"], "preflight incomplete: " + ", ".join(check["missing"]))
        findings = self.trust.verify(self.permit, purpose="run", domain="odyssey", subject=check["subject"],
                                     producer="odyssey-runner")
        require(findings.get("approved") is True, "operator did not authorize execution")
        if self.plan["kind"] == "main":
            require(check["call_budget_covers_upper_bound"], "main run call budget does not cover conservative upper bound")


def jobs_for(plan: dict) -> list[dict]:
    result = []
    for unit in plan["units"]:
        for arm in plan["arms"]:
            entity = f"{unit}.{arm}"
            previous = None
            for phase in PHASES:
                if phase not in {"train", "freeze"} and not plan["episodes"][phase]:
                    continue
                key = f"{entity}.{phase}"
                result.append({"id": key, "entity": entity, "phase": phase,
                               "payload": {"unit": unit, "arm": arm, "plan": sha(plan)},
                               "depends_on": [previous] if previous else []})
                previous = key
    return result


def initialize(store: Store, gate: Gate) -> dict:
    gate.check()
    require(not store.db.execute("SELECT 1 FROM jobs").fetchone(), "initialization requires a fresh run store")
    plan, bindings = gate.plan, gate.bindings
    with store.transaction():
        store.db.executemany("INSERT INTO meta VALUES (?,?)", [("run_plan", canonical(plan)),
            ("run_bindings", canonical(sha(bindings))), ("run_started", str(time.time()))])
    for unit in plan["units"]:
        for arm in plan["arms"]:
            entity = Entity.create(store, f"{unit}.{arm}", gate.trust, host=bindings["host"])
            entity.project("odyssey", goal=plan["preregistration"]["question"], next_action="develop from prospective verified experience")
            for document in (bindings["organ"], bindings["replacement"]):
                if document is not None and ARMS[arm]["llm"]:
                    organ = Organ(**document)
                    certificates = [c for c in bindings["qualifications"] if c.get("purpose") == "organ-binding"
                                    and c.get("subject") == sha({"requirement": organ_requirement(organ), "host": bindings["host"]})]
                    require(len(certificates) == 1, "one independently qualified organ-binding receipt is required")
                    entity.bind_organ(organ_requirement(organ), certificates[0], producer=certificates[0]["producer"])
    store.enqueue(jobs_for(plan))
    return {"initialized": True, "plan": sha(plan), "entities": len(plan["units"]) * len(plan["arms"]),
            "jobs": len(jobs_for(plan)), "executed_trials": 0}


class Runner:
    def __init__(self, store: Store, gate: Gate, *, oracle=None, brokers: dict | None = None):
        require(store.meta("run_plan") == gate.plan and store.meta("run_bindings") == sha(gate.bindings),
                "store, plan and bindings differ")
        self.store, self.gate, self.plan, self.bindings = store, gate, gate.plan, gate.bindings
        self.budget = Budget(**self.plan["budget"])
        self.profile = study_profile(self.plan)
        self.oracle = oracle or OracleClient(**self.bindings["evaluator"])
        endpoints = tuple(d["endpoint"] for d in (self.bindings["organ"], self.bindings["replacement"]) if d)
        self.brokers = brokers if brokers is not None else {d["id"]: Broker(store, Organ(**d), self.budget,
            approved_endpoints=endpoints) for d in (self.bindings["organ"], self.bindings["replacement"]) if d}

    def check_budget(self, job: dict) -> None:
        require(time.time() < self.gate.permit["expires"], "run permit expired")
        require(time.time() - self.store.meta("run_started") < self.budget.wall_seconds, "run wall budget exhausted")
        require(self.store.storage()["payload_bytes"] <= self.budget.retained_bytes, "payload budget exhausted")
        self.store.heartbeat(job)

    def worker(self, owner: str, *, max_jobs: int = 1) -> dict:
        integer(max_jobs, 1, 100000)
        completed = []
        for _ in range(max_jobs):
            self.gate.check()
            job = self.store.claim(identifier(owner), max_running=self.budget.workers)
            if job is None:
                break
            try:
                self.check_budget(job)
                started = time.perf_counter_ns()
                result = self.execute(job)
                result["job_service_wall_ns"] = max(1, time.perf_counter_ns() - started)
                result["completed_at"] = time.time()
                self.store.complete(job, result)
                completed.append(job["id"])
            except BaseException as exc:
                unsettled = self.store.db.execute("SELECT 1 FROM effects WHERE job=? AND state IN ('sent','reserved','uncertain')",
                                                 (job["id"],)).fetchone()
                state = "uncertain" if unsettled or isinstance(exc, TransportError) else "failed"
                try:
                    self.store.complete(job, {"error_type": type(exc).__name__, "reason": str(exc)[:1024],
                                             "requires_operator_review": True}, state=state)
                except Refused:
                    pass  # Lease expiry itself fences the worker; no unsafe attempt to seize it back.
                raise
        return {"completed_jobs": completed, "status": status(self.store), "efficacy_claim": False}

    def _materials(self, entity: Entity, enabled: bool) -> list[dict]:
        if not enabled:
            return []
        rows = [row for row in entity.all("materials").values()
                if row.get("kind") == "abstraction-candidate" and row.get("status") != "suspended"
                and not any(entity.get("revoked-sources", key) for key in row.get("source_ancestry", []))]
        return [{"template": row["expression"]} for row in sorted(rows, key=lambda row: row["id"])[:8]]

    def answer(self, entity: Entity, arm: str, public: dict, job: dict, *, replacement: bool = False,
               cycle: CognitiveCycle | None = None) -> dict:
        flags = ARMS[arm]
        started = time.perf_counter_ns()
        cache_key = sha({"scope": public["scope"], "input": public["input"]})
        if flags["cache"]:
            cached = entity.get("answer-cache", cache_key)
            if cached:
                return {"answer": cached["actual"], "confidence": 1.0, "mechanism": "answer-cache",
                        "cost": Cost(elapsed_ns=max(1, time.perf_counter_ns() - started)).document(), "skill": None}
        route = cycle.decide() if cycle else entity.route(public["family"], public["scope"], permit_model=flags["llm"],
                                                          self_model=flags["self"], compilation=flags["compile"])
        if route.kind == "skill":
            try:
                answer = entity.use(route.id, public["input"], family=public["family"], scope=public["scope"])
                return {"answer": answer, "confidence": route.confidence, "mechanism": "native-skill", "skill": route.id,
                        "cost": Cost(elapsed_ns=max(1, time.perf_counter_ns() - started)).document()}
            except Refused:
                # A range/type refusal is not permission to weaken a guard; a qualified organ may handle novelty.
                route = entity.route(public["family"], public["scope"], permit_model=flags["llm"], compilation=False)
        if route.kind == "organ":
            organ_id = self.bindings["replacement"]["id"] if replacement else self.bindings["organ"]["id"]
            row = entity.get("organs", organ_id)
            require(row and row["status"] == "qualified" and row["host"] == entity.host
                    and entity._certificate_live(row["certificate"]), "selected organ is not currently qualified")
            examples = entity.examples(public["family"], maximum=cycle.retrieval_limit if cycle else 32) if flags["memory"] else []
            context = {"workspace": cycle.context()} if cycle else {}
            receipt = self.brokers[organ_id].query(job, role="reasoner", task=public, examples=examples,
                                                  materials=self._materials(entity, flags["material"]), **context)
            cost = {**receipt["cost"], "elapsed_ns": max(1, time.perf_counter_ns() - started)}
            proposal = receipt["proposal"] or {"answer": None, "confidence": 0.0}
            return {**proposal, "mechanism": "neural-organ" if receipt["proposal"] is not None else "abstain",
                    "cost": cost, "skill": None, "organ": receipt["organ"], "usage_known": receipt["usage_known"],
                    "invalid_proposal": receipt["invalid_proposal"]}
        return {"answer": None, "confidence": 0.0, "mechanism": "abstain", "skill": None,
                "cost": Cost(elapsed_ns=max(1, time.perf_counter_ns() - started)).document()}

    def episode(self, entity: Entity, job: dict, family: str, split: str, index: int) -> dict:
        self.check_budget(job)
        unit, arm = job["document"]["unit"], job["document"]["arm"]
        descriptor = {"unit": unit, "arm": arm, "family": family, "split": split, "index": index}
        public = self.oracle.call("case", descriptor)
        fields(public, {"case_id", "domain", "family", "split", "input", "source", "scope", "description", "input_schema"})
        require(public["family"] == family and public["split"] == split, "oracle case projection mismatch")
        observation_id = entity.observe(observation(public)) if split == "train" else None
        # Controls without developed cognition keep their original pathway and equal resource accounting.
        episode_started = time.perf_counter_ns()
        cycle = None
        if self.profile["enabled"] and arm not in {"model-only", "memory-only", "cache-only"}:
            policy = {**self.profile["workspace"], "lesion": ARMS[arm].get("lesion",
                      "no-self" if arm == "no-self-model" else self.profile["workspace"]["lesion"])}
            cycle = CognitiveCycle(entity, public, policy=WorkspacePolicy(**policy), flags=ARMS[arm])
        answer = self.answer(entity, arm, public, job, replacement=split == "organ_swap", cycle=cycle)
        cognition = cycle.finish(answer) if cycle else None
        answer["cost"]["elapsed_ns"] = max(1, time.perf_counter_ns() - episode_started)
        if split == "train":
            prediction = entity.predict(observation_id, answer["answer"], producer="odyssey-runner",
                                        mechanism=answer["mechanism"], confidence=answer["confidence"], skill_id=answer["skill"])
            proposal = {"prediction": prediction, "case_id": public["case_id"], "unit": unit, "actual": None,
                        "producer": "odyssey-runner"}
            subject = entity.subject("experience", proposal)
        else:
            require(self.store.entity(entity.id)["frozen"], "held-out evaluation requires frozen cognition")
            subject = {"entity": entity.id, "head": self.store.head(entity.id), "plan": sha(self.plan)}
        response = self.oracle.call("submit", {**descriptor, **{key: answer[key] for key in ("answer", "confidence", "mechanism", "cost")},
                                               "producer": "odyssey-runner", "subject": subject})
        metrics = {"mechanism": answer["mechanism"], "cost": answer["cost"], "confidence": answer["confidence"],
                   "usage_known": answer.get("usage_known", True), "correct": None, "sealed": split != "train"}
        if cognition is not None:
            metrics["cognition"] = cognition
        if split == "train":
            admitted = entity.assimilate(response["proposal"], response["certificate"])
            if cycle:
                cycle.assimilate(admitted)
            if self.plan.get("development", {}).get("enabled") and arm not in {"model-only", "memory-only", "cache-only"}:
                development = Development(entity, LearningPolicy(**self.plan["development"]["policy"]))
                started = time.perf_counter_ns()
                decision = development.allocate(admitted, allow_structure=ARMS[arm]["compile"], allow_neural=ARMS[arm]["llm"])
                metrics["learning_route"] = decision["route"]
                if ARMS[arm]["material"] and entity.get("replay", family)["seen"] % development.policy.consolidate_every == 0:
                    development.consolidate(development.sleep_plan())
                metrics["development_ns"] = time.perf_counter_ns() - started
            metrics["correct"] = equivalent(answer["answer"], response["actual"])
            if ARMS[arm]["cache"]:
                key = sha({"scope": public["scope"], "input": public["input"]})
                self.store.commit(entity.id, self.store.head(entity.id), "cache-learned", {"case": public["case_id"]},
                                  [("answer-cache", key, {"actual": response["actual"], "case": public["case_id"]})])
        else:
            require(response.get("sealed") is True, "evaluator leaked an unsealed final result")
        record = {"id": sha({"entity": entity.id, "case": public["case_id"]}), "entity": entity.id,
                  "unit": unit, "arm": arm, "phase": split, "family": family, "case_id": public["case_id"], "metrics": metrics}
        self.store.add_outcome(record, {"public": public, "answer": answer, "evaluator": response,
                                       "cognition": cycle.trace if cycle else None},
                               retain_success_every=self.plan["successful_trace_every"])
        return public

    def learn(self, entity: Entity, job: dict, public: dict) -> dict:
        flags = ARMS[job["document"]["arm"]]
        examples = entity.examples(public["family"])
        if len(examples) < 2 or not (flags["compile"] or flags["teacher"]):
            return {"candidate": None}
        skill_id = None
        if flags["compile"] and entity.route(public["family"], public["scope"], permit_model=False,
                                              self_model=flags["self"]).kind == "skill":
            return {"candidate": None, "reason": "qualified native skill already available"}
        if flags["compile"] and all(spec["type"] in {"int", "number"} for spec in public["input_schema"].values()):
            search = entity.learn_scalar(public["family"], inputs=public["input_schema"], domain=public["domain"], scope=public["scope"],
                                         max_candidates=self.plan["induction_candidates"], seconds=self.plan["induction_seconds"], material=flags["material"])
            skill_id = search.get("skill")
            if skill_id and entity.get("rejected-programs", entity.get("skills", skill_id)["program_digest"]):
                skill_id = None
        if skill_id is None and flags["teacher"]:
            organ_id = self.bindings["organ"]["id"]
            receipt = self.brokers[organ_id].query(job, role="teacher", task=public, examples=examples,
                                                  materials=self._materials(entity, flags["material"]))
            if receipt["proposal"] is None:
                return {"candidate": None, "reason": "teacher returned an invalid proposal; charged without retry"}
            program = receipt["proposal"]["program"]
            compiled = compile_program(program)
            try:
                fits = all(equivalent(compiled.run(row["input"]), row["actual"]) for row in examples)
            except Refused:
                fits = False
            if fits and flags["compile"]:
                skill_id = entity.propose_skill(program, family=public["family"], domain=public["domain"], scope=public["scope"],
                                                witnesses=[row["experience_id"] for row in examples], producer=organ_id)
            if flags["material"]:
                key = sha(program["body"])
                entity.store.commit(entity.id, entity.store.head(entity.id), "teacher-material-proposed", {"candidate": key},
                    [("materials", key, {"id": key, "kind": "abstraction-candidate", "status": "unqualified",
                     "expression": program["body"], "dependencies": [], "certificate": None,
                     "producer": organ_id, "training_fit": fits,
                     "source_ancestry": sorted({entity.get("observations", entity.get("predictions", r["prediction"])["observation"])["source"] for r in examples})})])
        if skill_id is not None:
            row = entity.get("skills", skill_id)
            subject = entity.subject("skill", {"skill": skill_id, "record": sha(row), "host": entity.host})
            response = self.oracle.call("qualify", {"unit": job["document"]["unit"], "arm": job["document"]["arm"],
                                                    "family": public["family"], "skill": row, "subject": subject})
            if response.get("passed") is True:
                entity.qualify_skill(skill_id, response["certificate"])
                if flags["material"]:
                    entity.consolidate()
            else:
                entity.store.commit(entity.id, entity.store.head(entity.id), "candidate-rejected", {"skill": skill_id},
                    [("rejected-programs", row["program_digest"], {"skill": skill_id, "reason": "independent qualification failed"})])
            return {"candidate": skill_id, "qualified": response.get("passed") is True}
        return {"candidate": None}

    def execute(self, job: dict) -> dict:
        entity = Entity(self.store, job["entity"], self.gate.trust, host=self.bindings["host"])
        phase, flags = job["phase"], ARMS[job["document"]["arm"]]
        if phase == "freeze":
            if flags["material"]:
                if self.plan.get("development", {}).get("enabled"):
                    development = Development(entity, LearningPolicy(**self.plan["development"]["policy"]))
                    development.consolidate(development.sleep_plan())
                else:
                    entity.consolidate()
            head = self.store.freeze(entity.id, self.store.head(entity.id))
            receipt = self.oracle.call("seal", {"unit": job["document"]["unit"], "arm": job["document"]["arm"], "head": head})
            require(receipt.get("sealed") is True and receipt.get("head") == head, "evaluator freeze acknowledgement differs")
            return {"frozen": head, "evaluator": receipt, "phenotype": phenotype(entity)}
        counts = {family: 0 for family in self.plan["families"]}
        attempts = {family: 0 for family in self.plan["families"]}
        maximum = self.plan["episodes"][phase]
        while any(count < maximum for count in counts.values()):
            candidates = [family for family, count in counts.items() if count < maximum]
            if phase == "train" and flags["self"]:
                family = entity.curriculum(candidates, maximum=1)[0]["family"]
            else:
                family = min(candidates, key=lambda name: (counts[name], name))
            public = self.episode(entity, job, family, phase, counts[family])
            counts[family] += 1
            if phase == "train" and counts[family] % self.plan["compile_every"] == 0:
                if attempts[family] < self.plan["qualification_attempts"]:
                    self.learn(entity, job, public)
                    attempts[family] += 1
        return {"phase": phase, "counts": counts, "head": self.store.head(entity.id), "scores_sealed": phase != "train"}


def status(store: Store) -> dict:
    states = {row[0]: row[1] for row in store.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")}
    return {"jobs": states, "storage": store.storage(), "model_calls_reserved_or_used": store.counter("calls"),
            "tokens_reserved_or_used": store.counter("tokens"),
            "complete": bool(states) and set(states) == {"done"}, "efficacy": "not-computed-by-status"}


def acquisition_report(store: Store) -> dict:
    """Separate developmental/service cost from answer-path cost; do not sum overlapping jobs as elapsed time."""
    groups = {}
    for job in store.db.execute("SELECT * FROM jobs ORDER BY id"):
        payload = store.get(job["payload"])
        key = payload["unit"] + "." + payload["arm"]
        group = groups.setdefault(key, {"unit": payload["unit"], "arm": payload["arm"], "phases": {},
                                        "model_calls": 0, "tokens_reserved_or_used": 0})
        result = store.get(job["result"]) if job["result"] else {}
        group["phases"][job["phase"]] = {"state": job["state"], "service_wall_ns": result.get("job_service_wall_ns")}
        for effect in store.db.execute("SELECT * FROM effects WHERE job=?", (job["id"],)):
            group["model_calls"] += effect["reserved_calls"]
            response = store.get(effect["response"]) if effect["response"] else None
            if response and response.get("usage_known"):
                group["tokens_reserved_or_used"] += response["cost"]["input_tokens"] + response["cost"]["output_tokens"]
            else:
                group["tokens_reserved_or_used"] += effect["reserved_tokens"]
    return {"status": status(store), "histories": list(groups.values()),
            "cost_scope": "all actor jobs including teacher calls, induction and evaluator round trips; parallel job times overlap",
            "evaluator_private_storage_and_energy": "not measured by actor store"}
