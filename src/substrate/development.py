"""Reversible developmental learning, replay and action-conditioned prediction.

Observed data, verified experience and qualified competence are different things.
All weight changes here are bounded candidates; external neural fitting requires
an operator permit and independent retention review before deployment. No loop
creates its own authority or treats imagined trajectories as observations.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from .core import canonical, clone, digest, fields, identifier, integer, numeric, require, sha

TARGET = {
    "intelligence": "locally-owned-developmental-general-intelligence",
    "consciousness": "investigate-full-consciousness-with-competing-theory-interventions",
    "odyssey_programs": 1, "subjective_experience_established": False,
    "self_preservation_objective": False, "synthetic_distress_objective": False,
    "port_scope": "whole-relevant-organization-and-execution-system",
}
ROUTES = ("remember", "structuralize", "neuralize", "reopen")


@dataclass(frozen=True)
class LearningPolicy:
    native_after: int = 4
    neural_after: int = 24
    replay_per_family: int = 24
    replay_limit: int = 256
    consolidate_every: int = 32
    learning_rate: float = .01

    def __post_init__(self):
        integer(self.native_after, 2, 1024)
        integer(self.neural_after, self.native_after, 65536)
        integer(self.replay_per_family, 2, 64)
        integer(self.replay_limit, 2, 512)
        integer(self.consolidate_every, 1, 65536)
        numeric(self.learning_rate, .000001, .1)


def feature_vector(value: list, size: int) -> list[float]:
    require(type(value) is list and len(value) == size, "feature dimensions changed")
    return [numeric(x, -1, 1) for x in value]


def new_predictor(state_size: int, action_size: int, *, encoder: str) -> dict:
    """Untrained action-conditioned tanh dynamics over normalized observed features.

    Repeated prediction propagates a belief state; partial observations correct
    it. The encoder is an explicitly pinned external dependency, not fabricated
    perception. This small model is not a general learned multimodal world.
    """
    integer(state_size, 1, 128)
    integer(action_size, 1, 32)
    return {"schema": "substrate-predictive-memory-v1", "encoder": digest(encoder),
            "state_size": state_size, "action_size": action_size,
            "weights": [[0.0] * (state_size + action_size + 1) for _ in range(state_size)],
            "updates": 0, "squared_error": 0.0, "observed_components": 0}


def forecast(model: dict, state: list, action: list) -> list[float]:
    require(model["schema"] == "substrate-predictive-memory-v1", "unknown predictive model")
    n, m = integer(model["state_size"], 1, 128), integer(model["action_size"], 1, 32)
    x = feature_vector(state, n) + feature_vector(action, m) + [1.0]
    require(len(model["weights"]) == n, "weight row count differs")
    result = []
    for row in model["weights"]:
        require(len(row) == len(x), "weight width differs")
        result.append(math.tanh(sum(numeric(w, -32, 32) * a for w, a in zip(row, x))))
    return result


def fit_transition(model: dict, state: list, action: list, actual: list, observed: list,
                   *, rate: float = .01) -> tuple[dict, list]:
    numeric(rate, .000001, .1)
    result = clone(model)
    prediction = forecast(model, state, action)
    actual = feature_vector(actual, model["state_size"])
    require(type(observed) is list and len(observed) == len(actual)
            and all(type(v) is bool for v in observed) and any(observed), "observation mask required")
    x = feature_vector(state, model["state_size"]) + feature_vector(action, model["action_size"]) + [1.0]
    scale = max(1.0, sum(a * a for a in x))
    for i, present in enumerate(observed):
        if present:
            error = actual[i] - prediction[i]
            step = rate * error * (1 - prediction[i] ** 2) / scale
            result["weights"][i] = [max(-32, min(32, w + step * a)) for w, a in zip(result["weights"][i], x)]
            result["squared_error"] += error * error
            result["observed_components"] += 1
    result["updates"] += 1
    return result, [actual[i] if observed[i] else prediction[i] for i in range(len(actual))]


class Development:
    def __init__(self, entity, policy: LearningPolicy | None = None):
        self.entity = entity
        self.policy = policy or LearningPolicy()

    def _write(self, kind: str, changes: list, payload=None) -> str:
        e = self.entity
        return e._commit(kind, payload or {}, changes, base=e.store.head(e.id))

    def allocate(self, experience: str, *, allow_structure: bool = True, allow_neural: bool = True) -> dict:
        """Admission is idempotent; replay never invents additional independent support."""
        e = self.entity
        old = e.get("learning-decisions", digest(experience))
        if old:
            return old
        row = e.get("experiences", experience)
        require(row and row["split"] in {"train", "development"} and e.experience_live(row),
                "allocator accepts only live verified developmental experience")
        stats = e.competence(row["family"])
        route = "remember"
        scalar = type(row["actual"]) in (float, int) and all(type(v) in (int, float) for v in row["input"].values())
        if allow_structure and scalar and stats["attempts"] >= self.policy.native_after:
            route = "structuralize"
        if allow_neural and stats["attempts"] >= self.policy.neural_after and sum(stats["recent"]) < len(stats["recent"]) / 2:
            route = "neuralize"
        if not row["correct"] and e.get("predictions", row["prediction"]).get("skill"):
            route = "reopen"
        decision = {"experience": experience, "family": row["family"], "route": route,
                    "policy": asdict(self.policy), "status": "allocated-not-trained",
                    "criterion": "verified-recurrence-and-recent-failure", "at": time.time()}
        index = e.get("replay", row["family"], {"keys": [], "seen": 0})
        pool = list(dict.fromkeys(index["keys"] + [experience]))
        # Keep failure coverage AND successful retention controls; deterministic within each stratum.
        failures = sorted(k for k in pool if not e.get("experiences", k)["correct"])
        successes = sorted(k for k in pool if e.get("experiences", k)["correct"])
        cap = self.policy.replay_per_family
        chosen = failures[:cap // 2] + successes[:cap - min(len(failures), cap // 2)]
        chosen += [k for k in pool if k not in chosen][:cap - len(chosen)]
        index.update(keys=sorted(chosen), seen=index["seen"] + 1)
        self._write("learning-allocated", [("learning-decisions", experience, decision), ("replay", row["family"], index)])
        return decision

    def sleep_plan(self) -> dict:
        e = self.entity
        pool = [(family, key) for family, row in sorted(e.all("replay").items()) for key in row["keys"]
                if e.experience_live(e.get("experiences", key, {}))]
        # Round-robin families prevents one high-volume stream from owning consolidation.
        pool.sort(key=lambda item: (e.get("replay", item[0])["keys"].index(item[1]), item[0]))
        selected = pool[:self.policy.replay_limit]
        return {"schema": "substrate-consolidation-v1", "entity": e.id, "base": e.store.head(e.id),
                "policy": asdict(self.policy), "replay": [key for _, key in selected],
                "families": sorted({family for family, _ in selected}), "source_labels": "verified-development-only",
                "operations": ["acquired-template-mining", "retention-sampling", "inquiry-proposal"],
                "neural_fitting": "separate-authorized-job", "destructive_forgetting": False}

    def consolidate(self, plan: dict) -> dict:
        e = self.entity
        require(plan == self.sleep_plan(), "consolidation plan is stale or changed")
        before = e.store.head(e.id)
        material = e.consolidate()
        inquiries = e.curriculum(plan["families"], maximum=min(64, max(1, len(plan["families"]))))
        record = {"plan": sha(plan), "before": before, "after_material": e.store.head(e.id),
                  "material_result": material, "replay": plan["replay"], "inquiries": inquiries,
                  "independent_support_added": 0, "weights_trained": False}
        self._write("development-consolidated", [("consolidations", sha(record), record),
                    ("development-agenda", "current", {"inquiries": inquiries, "status": "proposal-no-authority"})])
        return record

    def predict_transition(self, *, stream: str, sequence: int, model: dict, state: list,
                           action: list, domain: str, source: str, unit: str) -> str:
        e = self.entity
        identifier(stream)
        integer(sequence)
        active = e.get("predictive-streams", stream)
        require(active is None or (active["status"] == "observed" and sequence == active["sequence"] + 1),
                "one prospective prediction at a time, in stream order")
        require(not e.get("revoked-sources", source), "predictive source revoked")
        if active:
            previous = e.get("transition-predictions", active["prediction"])
            require(previous["source"] == source and previous["domain"] == domain and previous["unit"] == unit, "stream provenance changed")
            require(sha(model) == active["next_model"] and state == active["belief"], "stream state discontinuity")
        predicted = forecast(model, state, action)
        record = {"stream": stream, "sequence": sequence, "model": clone(model), "state": clone(state),
                  "action": clone(action), "predicted": predicted, "domain": identifier(domain),
                  "source": digest(source), "unit": identifier(unit), "split": "development", "base": e.store.head(e.id)}
        key = sha(record)
        self._write("transition-predicted", [("transition-predictions", key, record),
                    ("predictive-streams", stream, {"prediction": key, "sequence": sequence, "status": "pending"})])
        return key

    def observe_transition(self, proposal: dict, certificate: dict) -> dict:
        fields(proposal, {"prediction", "actual", "observed", "producer"})
        e = self.entity
        prior = e.get("transition-predictions", digest(proposal["prediction"]))
        require(prior and e.get("transition-outcomes", proposal["prediction"]) is None, "missing or replayed transition")
        require(not e.get("revoked-sources", prior["source"]), "transition source revoked")
        require(e.get("predictive-streams", prior["stream"])["prediction"] == proposal["prediction"], "stream moved")
        findings = e._verify("transition", prior["domain"], proposal, certificate, proposal["producer"])
        require(findings.get("outcome_verified") is True and findings.get("source") == prior["source"], "unverified transition")
        require(findings.get("split") == "development" and findings.get("encoder") == prior["model"]["encoder"],
                "encoder or developmental boundary differs")
        model, belief = fit_transition(prior["model"], prior["state"], prior["action"], proposal["actual"],
                                      proposal["observed"], rate=self.policy.learning_rate)
        key = sha(model)
        outcome = {**clone(proposal), "certificate": clone(certificate), "model": key, "prediction": proposal["prediction"],
                   "prediction_error": [a - b if seen else None for a, b, seen in zip(proposal["actual"], prior["predicted"], proposal["observed"])],
                   "observed": proposal["observed"], "unit": prior["unit"], "source": prior["source"]}
        self._write("transition-learned", [("predictive-models", key, model),
            ("transition-outcomes", proposal["prediction"], outcome), ("predictive-streams", prior["stream"],
             {"prediction": proposal["prediction"], "sequence": prior["sequence"], "status": "observed",
              "next_model": key, "belief": belief})])
        return {"model": key, "belief": belief, "qualified_for_action": False}

    def stage_mutation(self, *, slot: str, value: dict, sources: list[str], retention: dict,
                       producer: str) -> str:
        """Immutable candidate plus rollback state; never edits an executable NX."""
        e = self.entity
        identifier(slot)
        fields(retention, {"suite", "domains", "training_groups", "evaluation_groups", "max_regression"})
        digest(retention["suite"])
        require(retention["domains"] and set(retention["training_groups"]).isdisjoint(retention["evaluation_groups"])
                and retention["evaluation_groups"], "retention needs independent groups and domains")
        numeric(retention["max_regression"], 0, 1)
        require(0 < len(sources) <= 256, "mutation source closure required")
        for source in sources:
            digest(source)
            require(not e.get("revoked-sources", source), "mutation includes revoked source")
        row = {"slot": slot, "value": clone(value), "sources": sorted(set(sources)), "retention": clone(retention),
               "producer": identifier(producer), "parent": clone(e.get("active-mutations", slot)),
               "base": e.store.head(e.id), "status": "candidate"}
        key = sha(row)
        self._write("mutation-staged", [("mutation-candidates", key, row)])
        return key

    def promote(self, mutation: str, certificate: dict) -> str:
        e = self.entity
        row = e.get("mutation-candidates", digest(mutation))
        require(row and row["status"] == "candidate", "missing candidate")
        require(e.get("active-mutations", row["slot"]) == row["parent"], "mutation parent changed")
        require(not any(e.get("revoked-sources", source) for source in row["sources"]), "revoked training source")
        proposal = {"mutation": mutation, "candidate": sha(row), "host": e.host}
        findings = e._verify("plasticity", "cognition", proposal, certificate, row["producer"])
        require(findings.get("executed") is True and findings.get("passed") is True
                and findings.get("suite") == row["retention"]["suite"], "missing executed retention review")
        require(set(findings.get("evaluation_groups", [])) == set(row["retention"]["evaluation_groups"]), "holdouts differ")
        regressions = findings.get("regression", {})
        require(set(regressions) == set(row["retention"]["domains"]), "retention domain missing")
        for value in regressions.values():
            require(numeric(value, -1, 1) <= row["retention"]["max_regression"], "retention regression exceeded")
        active = {"mutation": mutation, "value": row["value"], "sources": row["sources"], "certificate": clone(certificate),
                  "host": e.host, "parent": row["parent"], "status": "active",
                  "attested_subject": e.subject("plasticity", proposal)}
        return self._write("mutation-promoted", [("active-mutations", row["slot"], active),
                            ("mutation-candidates", mutation, {**row, "status": "promoted"})], {"mutation": mutation})

    def active(self, slot: str) -> dict | None:
        row = self.entity.get("active-mutations", identifier(slot))
        if not row or row.get("status") != "active" or row.get("host") != self.entity.host:
            return None
        if not self.entity._certificate_live(row.get("certificate")) or any(
                self.entity.get("revoked-sources", source) for source in row["sources"]):
            return None
        candidate = self.entity.get("mutation-candidates", row["mutation"])
        if not candidate or row["value"] != candidate["value"] or row["sources"] != candidate["sources"]:
            return None
        original = {**candidate, "status": "candidate"}
        if row.get("attested_subject", {}).get("proposal") != {"mutation": row["mutation"], "candidate": sha(original), "host": self.entity.host}:
            return None
        try:
            self.entity.trust.verify(row["certificate"], purpose="plasticity", domain="cognition",
                subject=row["attested_subject"], producer=candidate["producer"])
        except ValueError:
            return None
        return clone(row["value"])

    def rollback(self, slot: str, *, reason: str) -> str:
        e = self.entity
        row = e.get("active-mutations", identifier(slot))
        require(row is not None and 0 < len(reason) <= 2048, "rollback needs active lineage and reason")
        parent = row.get("parent") or {"status": "inactive"}
        return self._write("mutation-rolled-back", [("active-mutations", slot, parent)],
                           {"mutation": row.get("mutation"), "reason": reason, "history_erased": False})

    def revoke_source(self, source: str, *, reason: str) -> str:
        """Conservative dependency quarantine, not a claim of neural unlearning."""
        e = self.entity
        digest(source)
        require(type(reason) is str and 0 < len(reason) <= 2048, "revocation needs a reason")
        changes = [("revoked-sources", source, {"reason": reason, "at": time.time()})]
        for namespace in ("active-mutations", "mutation-candidates", "neural-adapters"):
            for key, row in e.all(namespace).items():
                if source in row.get("sources", []):
                    changes.append((namespace, key, {**row, "status": "quarantined", "reason": reason}))
        for key, row in e.all("beliefs").items():
            if source in row.get("sources", []):
                e.reopen(key, reason, evidence=source)
        return self._write("source-revoked", changes, {"source": source, "neural_unlearning_established": False})
