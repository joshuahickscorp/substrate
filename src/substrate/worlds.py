"""Versioned, secret-seeded micro-worlds and an independent evaluator service.

The service holds task parameters, held-out seeds and the verifier key. The
actor receives a projected task, never a recipe containing its hidden answer.
These are bounded mechanism experiments, not a general-intelligence benchmark.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from .core import (Cost, Trust, attest, canonical, clone, digest, fields, identifier,
                   integer, numeric, require, sha)
from .language import compile_program, equivalent
from .store import Store

WORLD_VERSION = "micro-worlds-2"
FAMILIES = ("affine", "queue", "logic", "sequence", "composed", "causal")
SPLITS = ("train", "qualification", "evaluation", "transfer", "retention", "shift", "organ_swap")
INPUTS = {
    "affine": {"x": {"type": "int", "min": -1000000, "max": 1000000}},
    "queue": {key: {"type": "int", "min": 0, "max": 1000000} for key in ("q", "arrivals", "service")},
    "logic": {key: {"type": "bool"} for key in ("a", "b", "c")},
    "sequence": {"xs": {"type": "list-int", "max_items": 128}},
    "composed": {"x": {"type": "int", "min": -1000000, "max": 1000000}},
    "causal": {"x": {"type": "int", "min": -1000000, "max": 1000000},
               "intervene": {"type": "bool"}, "set_m": {"type": "int", "min": -1000000, "max": 1000000}},
}
DESCRIPTIONS = {
    "affine": "Infer the fixed integer affine rule from verified examples.",
    "queue": "Predict next queue occupancy with nonnegative clipping and a fixed hidden capacity.",
    "logic": "Infer a three-input Boolean function. All eight inputs are a finite exhaustive domain.",
    "sequence": "Infer a bounded list transformation; preserve the specified order and multiplicities.",
    "composed": "Infer a composition of affine arithmetic, absolute value and a fixed offset.",
    "causal": "Predict output of a chain X -> M -> Y, respecting an explicit intervention on M.",
}


def keyed_seed(secret: bytes, *parts: Any) -> int:
    require(type(secret) is bytes and len(secret) >= 32, "evaluator needs a private 256-bit seed")
    return int.from_bytes(hmac.new(secret, canonical(parts), hashlib.sha256).digest(), "big")


@dataclass(frozen=True)
class Specimen:
    unit: str
    family: str
    split: str
    index: int
    public: dict
    actual: Any
    recipe: dict


class World:
    def __init__(self, secret: bytes, *, revision: str = WORLD_VERSION):
        require(revision == WORLD_VERSION, "unknown world revision")
        keyed_seed(secret, "validate")
        self.secret, self.revision = secret, revision

    def parameters(self, unit: str, family: str) -> dict:
        identifier(unit)
        require(family in FAMILIES, "unknown world family")
        rng = random.Random(keyed_seed(self.secret, "parameters", unit, family))
        return {"a": rng.choice([-2, -1, 1, 2]), "b": rng.choice([-2, -1, 0, 1, 2]),
                "capacity": rng.randrange(8, 33), "variant": rng.randrange(4)}

    def make(self, unit: str, family: str, split: str, index: int) -> Specimen:
        require(split in SPLITS, "unknown world split")
        integer(index, 0, 10**9)
        parameters = self.parameters(unit, family)
        rng = random.Random(keyed_seed(self.secret, "specimen", self.revision, unit, family, split, index))
        # Numeric partitions are disjoint; qualification inputs cannot have been training witnesses.
        base = {"train": 0, "qualification": 1000, "evaluation": 10000,
                "transfer": 100000, "retention": 200000, "shift": 300000, "organ_swap": 400000}[split]
        x = base + index * 2 + rng.randrange(2)
        if family == "affine" or family == "composed":
            inputs = {"x": -x if index % 2 else x}
        elif family == "queue":
            inputs = {"q": rng.randrange(0, 40), "arrivals": base + index,
                      "service": base + index + rng.randrange(-20, 21)}
            inputs["service"] = max(0, inputs["service"])
        elif family == "logic":
            inputs = {key: bool((index % 8) & (1 << bit)) for bit, key in enumerate(("a", "b", "c"))}
        elif family == "sequence":
            size = 3 + index % 10
            inputs = {"xs": [base + rng.randrange(-20, 21) for _ in range(size)]}
        else:
            inputs = {"x": -x if index % 2 else x, "intervene": bool(index % 2), "set_m": base + index + 3}
        if split == "shift":
            parameters = {**parameters, "b": parameters["b"] + 1, "capacity": parameters["capacity"] + 3,
                          "variant": (parameters["variant"] + 1) % 4}
        task = sha({"unit": unit, "family": family, "world": self.revision})
        case = sha({"task": task, "split": split, "index": index})
        public = {"case_id": case, "domain": "micro-world", "family": family, "split": split,
                  "input": inputs, "source": sha({"world": self.revision}),
                  "scope": {"task": task, "world": self.revision},
                  "description": DESCRIPTIONS[family], "input_schema": clone(INPUTS[family])}
        recipe = {"world": self.revision, "unit": unit, "family": family, "split": split, "index": index}
        return Specimen(unit, family, split, index, public, self.truth(family, inputs, parameters), recipe)

    @staticmethod
    def truth(family: str, inputs: dict, parameters: dict) -> Any:
        """Reference equations are deliberately not evaluated through the candidate IR."""
        a, b, variant = parameters["a"], parameters["b"], parameters["variant"]
        if family == "affine":
            return a * inputs["x"] + b
        if family == "queue":
            return min(parameters["capacity"], max(0, inputs["q"] + inputs["arrivals"] - inputs["service"]))
        if family == "logic":
            x, y, z = inputs["a"], inputs["b"], inputs["c"]
            return ((x and y) or z, (x or y) and z, (x != y) != z, x if z else y)[variant]
        if family == "sequence":
            xs = inputs["xs"]
            return (sorted(xs), xs[::-1], list(dict.fromkeys(xs)), sorted(set(xs)))[variant]
        if family == "composed":
            return abs(a * inputs["x"] + b) + variant
        if family == "causal":
            mediator = inputs["set_m"] if inputs["intervene"] else a * inputs["x"] + b
            return 2 * mediator + variant
        raise ValueError("unimplemented world")


def observation(public: dict) -> dict:
    """Positive field projection, rather than deleting a few known secret fields."""
    return {name: clone(public[name]) for name in ("case_id", "domain", "family", "split", "input", "source", "scope")}


class Evaluator:
    """Deploy in an independently credentialed process, outside the actor sandbox.

    The private store is not portable entity state. Scores for the fixed final
    matrix are released only after the full planned matrix has been submitted.
    A signature establishes provenance, not the philosophical validity of a task.
    """
    def __init__(self, store: Store, plan: dict, secret: bytes, private_key, signer: str,
                 host: str, executor, *, ttl: float = 86400 * 30):
        require(store.role == "evaluator", "oracle requires a distinct evaluator database")
        self.store, self.plan, self.world = store, clone(plan), World(secret)
        self.private_key, self.signer, self.host = private_key, identifier(signer), digest(host)
        self.executor, self.ttl = executor, numeric(ttl, 1, 31_536_000)
        with store.transaction():
            previous = store.meta("plan")
            require(previous is None or previous == sha(plan), "evaluator store belongs to a different plan")
            store.set_meta("plan", sha(plan))
            secret_binding = sha({"seed_commitment": hashlib.sha256(secret).hexdigest(), "host": host})
            previous_secret = store.meta("secret_binding")
            require(previous_secret is None or previous_secret == secret_binding, "evaluator seed or host changed")
            store.set_meta("secret_binding", secret_binding)
        store.db.executescript("""
                CREATE TABLE IF NOT EXISTS submissions(
                    id TEXT PRIMARY KEY, unit TEXT, arm TEXT, family TEXT, split TEXT,
                    index_n INTEGER, request TEXT, response TEXT, correct INTEGER);
                CREATE TABLE IF NOT EXISTS seals(entity TEXT PRIMARY KEY, head TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS qualification(
                    id TEXT PRIMARY KEY, unit TEXT, arm TEXT, family TEXT, attempt INTEGER,
                    request TEXT, response TEXT);
            """)

    def _descriptor(self, request: dict) -> Specimen:
        for name in ("unit", "arm", "family", "split", "index"):
            require(name in request, f"missing {name}")
        require(request["unit"] in self.plan["units"] and request["arm"] in self.plan["arms"], "unregistered unit or arm")
        require(request["family"] in self.plan["families"], "unregistered family")
        require(request["split"] != "qualification", "qualification recipes are not actor-readable")
        entity = request["unit"] + "." + request["arm"]
        sealed = self.store.db.execute("SELECT head FROM seals WHERE entity=?", (entity,)).fetchone()
        require((request["split"] == "train") != bool(sealed), "training/final-evaluation phase boundary refused")
        counts = self.plan["episodes"]
        require(request["split"] in counts and request["index"] < counts[request["split"]], "case outside preregistered matrix")
        return self.world.make(request["unit"], request["family"], request["split"], request["index"])

    def case(self, request: dict) -> dict:
        specimen = self._descriptor(request)
        return clone(specimen.public)

    def _sign(self, purpose: str, subject: dict, producer: str, evidence: Any, findings: dict) -> dict:
        if self.store.db.in_transaction:
            self.store.put(evidence)
        else:
            with self.store.transaction():
                self.store.put(evidence)
        return attest(self.private_key, signer=self.signer, producer=producer, purpose=purpose,
                      domain="micro-world", subject=subject, evidence=sha(evidence), findings=findings, ttl=self.ttl)

    def submit(self, request: dict) -> dict:
        fields(request, {"unit", "arm", "family", "split", "index", "answer", "confidence", "mechanism",
                         "cost", "producer", "subject"})
        specimen = self._descriptor(request)
        numeric(request["confidence"], 0, 1)
        Cost(**request["cost"])
        identity = sha({"plan": sha(self.plan), "unit": specimen.unit, "arm": request["arm"],
                        "case": specimen.public["case_id"]})
        with self.store.transaction():
            prior = self.store.db.execute("SELECT * FROM submissions WHERE id=?", (identity,)).fetchone()
            if prior:
                require(prior["request"] == sha(request), "cannot revise a published prediction")
                return self.store.get(prior["response"])
            correct = equivalent(request["answer"], specimen.actual)
            evidence = {"request": request, "recipe": specimen.recipe, "actual": specimen.actual, "correct": correct}
            if specimen.split == "train":
                subject = clone(request["subject"])
                require(subject["operation"] == "experience", "wrong training admission subject")
                proposal = subject["proposal"]
                require(proposal["case_id"] == specimen.public["case_id"] and proposal["unit"] == specimen.unit
                        and proposal["producer"] == request["producer"], "training subject mismatch")
                # The actor publishes its prediction before receiving this independently computed label.
                proposal["actual"] = clone(specimen.actual)
                response = {"actual": specimen.actual, "proposal": proposal, "certificate": self._sign(
                    "experience", subject, request["producer"], evidence,
                    {"outcome_verified": True, "case_id": specimen.public["case_id"]})}
            else:
                head = self.store.db.execute("SELECT head FROM seals WHERE entity=?",
                                            (request["unit"] + "." + request["arm"],)).fetchone()[0]
                require(request["subject"].get("head") == head and request["subject"].get("plan") == sha(self.plan),
                        "final result is not bound to the frozen entity and plan")
                response = {"sealed": True, "receipt": identity, "case_id": specimen.public["case_id"]}
            self.store.db.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?,?)",
                                  (identity, specimen.unit, request["arm"], specimen.family, specimen.split,
                                   specimen.index, self.store.put(request), self.store.put(response), int(correct)))
            return clone(response)

    def qualify(self, request: dict) -> dict:
        fields(request, {"unit", "arm", "family", "skill", "subject"})
        require(request["unit"] in self.plan["units"] and request["arm"] in self.plan["arms"], "unregistered qualification")
        require(not self.store.db.execute("SELECT 1 FROM seals WHERE entity=?",
                    (request["unit"] + "." + request["arm"],)).fetchone(), "frozen entity cannot query qualification feedback")
        family, skill = request["family"], request["skill"]
        require(family in self.plan["families"] and skill["family"] == family, "qualification family mismatch")
        specification = self.world.make(request["unit"], family, "train", 0).public
        require(skill["scope"] == specification["scope"] and skill["domain"] == "micro-world",
                "qualification scope differs from the registered task")
        program = compile_program(skill["program"])
        require(program.digest == skill["program_digest"] and skill["program"]["inputs"] == specification["input_schema"],
                "qualification program digest or input contract differs")
        require(request["subject"]["operation"] == "skill", "wrong qualification subject")
        proposal = request["subject"]["proposal"]
        require(proposal["record"] == sha(skill) and proposal["host"] == self.host, "qualification artifact/host mismatch")
        request_id = sha(request)
        with self.store.transaction():
            prior = self.store.db.execute("SELECT * FROM qualification WHERE id=?", (request_id,)).fetchone()
            if prior:
                require(prior["response"] is not None, "previous qualification outcome uncertain; operator reconciliation required")
                return self.store.get(prior["response"])
            attempt = self.store.db.execute("SELECT COUNT(*) FROM qualification WHERE unit=? AND arm=? AND family=?",
                                            (request["unit"], request["arm"], family)).fetchone()[0]
            require(attempt < self.plan["qualification_attempts"], "qualification feedback budget exhausted")
            self.store.db.execute("INSERT INTO qualification VALUES (?,?,?,?,?,?,NULL)",
                                  (request_id, request["unit"], request["arm"], family, attempt, self.store.put(request)))
        count = self.plan["qualification_cases"]
        specimens = [self.world.make(request["unit"], family, "qualification", attempt * count + i) for i in range(count)]
        inputs = [case.public["input"] for case in specimens]
        # Finite Boolean domains are exhausted, not dishonestly called unseen-input generalization.
        finite = family == "logic"
        if finite:
            specimens = specimens[:8]
            inputs = [case.public["input"] for case in specimens]
        fresh = specimens if finite else [case for case in specimens if sha(case.public["input"]) not in skill["training_inputs"]]
        if len({sha(case.public["input"]) for case in fresh}) < 8:
            response = {"passed": False, "reason": "qualification shard lacks eight fresh distinct inputs"}
        else:
            outputs, cost = self.executor.batch(skill["program"], inputs)
            passed = len(outputs) == len(specimens) and all(equivalent(a, b.actual) for a, b in zip(outputs, specimens))
            findings = {"passed": passed, "split": "qualification", "program": skill["program_digest"],
                        "host": self.host, "scope": sha(skill["scope"]), "cases": [c.public["case_id"] for c in fresh],
                        "input_digests": [sha(c.public["input"]) for c in fresh], "oracle": sha({"world": WORLD_VERSION}),
                        "cost": cost.document(), "attempt": attempt, "finite_domain": finite,
                        "validation_mode": "exhaustive-finite" if finite else "held-out"}
            response = {"passed": passed, "certificate": self._sign("skill", request["subject"], skill["producer"],
                                                                      {"request": request_id, "findings": findings, "outputs": outputs,
                                                                       "recipes": [case.recipe for case in specimens],
                                                                       "actual": [case.actual for case in specimens]}, findings)}
        with self.store.transaction():
            self.store.db.execute("UPDATE qualification SET response=? WHERE id=?", (self.store.put(response), request_id))
        return response

    def seal(self, request: dict) -> dict:
        fields(request, {"unit", "arm", "head"})
        require(request["unit"] in self.plan["units"] and request["arm"] in self.plan["arms"], "unregistered seal")
        digest(request["head"])
        entity = request["unit"] + "." + request["arm"]
        with self.store.transaction():
            count = self.store.db.execute("SELECT COUNT(*) FROM submissions WHERE unit=? AND arm=? AND split='train'",
                                         (request["unit"], request["arm"])).fetchone()[0]
            require(count == len(self.plan["families"]) * self.plan["episodes"]["train"], "training matrix incomplete")
            previous = self.store.db.execute("SELECT head FROM seals WHERE entity=?", (entity,)).fetchone()
            require(previous is None or previous[0] == request["head"], "frozen head cannot be changed")
            self.store.db.execute("INSERT OR IGNORE INTO seals VALUES (?,?)", (entity, request["head"]))
        return {"sealed": True, "entity": entity, "head": request["head"]}

    def report(self) -> dict:
        previous = self.store.meta("final_report")
        if previous is not None:
            return self.store.get(previous)
        expected = len(self.plan["units"]) * len(self.plan["arms"]) * len(self.plan["families"]) * sum(
            count for split, count in self.plan["episodes"].items() if split != "train")
        count = self.store.db.execute("SELECT COUNT(*) FROM submissions WHERE split!='train'").fetchone()[0]
        require(count == expected, "fixed-horizon matrix is incomplete; held-out scores remain sealed")
        shards, batch = [], []
        with self.store.transaction():
            for row in self.store.db.execute("SELECT * FROM submissions WHERE split!='train' ORDER BY id"):
                request = self.store.get(row["request"])
                batch.append({"id": row["id"], "unit": row["unit"], "arm": row["arm"], "family": row["family"],
                              "split": row["split"], "index": row["index_n"], "correct": bool(row["correct"]),
                              "confidence": request["confidence"], "mechanism": request["mechanism"], "cost": request["cost"]})
                if len(batch) == 128:
                    shards.append(self.store.put(batch))
                    batch = []
            if batch:
                shards.append(self.store.put(batch))
            document = {"schema": "odyssey-scores-v2", "plan": sha(self.plan), "shards": shards,
                        "expected": expected, "unit_of_replication": "independent-developmental-history"}
            response = {"document": document, "certificate": self._sign("report", document, "odyssey-runner", document,
                                                                        {"complete": True, "rows": expected})}
            self.store.set_meta("final_report", self.store.put(response))
        return response

    def scores(self, payload: dict) -> dict:
        fields(payload, {"shard"})
        report = self.report()
        require(payload["shard"] in report["document"]["shards"], "not an admitted final score shard")
        return {"id": payload["shard"], "rows": self.store.get(payload["shard"])}

    def dispatch(self, method: str, payload: dict) -> dict:
        require(method in {"case", "submit", "qualify", "seal", "report", "scores"}, "unknown evaluator method")
        if method == "report":
            require(payload == {}, "report has no actor-selected filters")
            return self.report()
        return getattr(self, method)(payload)
