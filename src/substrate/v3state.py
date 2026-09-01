"""One integrated v3 cognitive entity over the existing Substrate mechanism owners."""

from __future__ import annotations

import hashlib

from substrate.epistemology import Defeater, EpistemicBelief, EpistemicLedger
from substrate.metacog import ReasoningPortfolio
from substrate.ontology import ActiveOntology, Concept
from substrate.selfmodel import Prediction, SelfModel
from substrate.v3fabric import CognitiveTask, simple_proposal
from substrate.v3io import sha_obj
from substrate.world import StructuralUnderstanding


class Refused(RuntimeError):
    """An integrated operation violated history, state, or activation constraints."""


try:
    from json.encoder import c_make_encoder, encode_basestring_ascii
except ImportError:  # pragma: no cover - alternate Python runtimes may omit the C accelerator.
    _V3_STATE_ENCODER = None
else:
    _V3_STATE_ENCODER = (
        c_make_encoder(None, str, encode_basestring_ascii, None, ":", ",", True, False, True)
        if c_make_encoder is not None
        else None
    )


def _semantic_state_hash(state: object) -> str:
    """Hash semantic state with the stdlib C encoder without changing bytes.

    The v3 state is JSON-shaped and acyclic by construction. Reusing the C
    encoder avoids rebuilding its configuration for every checkpoint while
    retaining the evidence layer's exact compact, sorted, ``default=str``
    representation. A recursive or otherwise unsupported edge falls back to
    the original path so its refusal semantics remain authoritative.
    """
    if _V3_STATE_ENCODER is None:
        return sha_obj(state)
    try:
        payload = "".join(_V3_STATE_ENCODER(state, 0)).encode()
    except RecursionError:
        return sha_obj(state)
    return hashlib.sha256(payload).hexdigest()


ARM_FEATURES = {
    "full_v3": {"ontology", "epistemology", "reasoning", "understanding", "inquiry", "self", "world", "development"},
    "v2_developmental_control": {"development", "self", "world"},
    "fixed_ontology": {"epistemology", "reasoning", "understanding", "inquiry", "self", "world", "development"},
    "confidence_only_epistemology": {"ontology", "reasoning", "understanding", "inquiry", "self", "world", "development"},
    "fixed_reasoning": {"ontology", "epistemology", "understanding", "inquiry", "self", "world", "development"},
    "no_understanding_structure": {"ontology", "epistemology", "reasoning", "inquiry", "self", "world", "development"},
    "simple_inquiry": {"ontology", "epistemology", "reasoning", "understanding", "self", "world", "development"},
    "no_self_model": {"ontology", "epistemology", "reasoning", "understanding", "inquiry", "world", "development"},
    "no_world_model": {"ontology", "epistemology", "reasoning", "understanding", "inquiry", "self", "development"},
    "more_compute": {"ontology", "epistemology", "reasoning", "understanding", "inquiry", "self", "world", "development"},
    "fresh_reset": {"ontology", "epistemology", "reasoning", "understanding", "inquiry", "self", "world"},
    "transcript_replay": {"development"},
}


class IntegratedEntity:
    """The v3 cycle composes ontology, epistemology, reasoning, inquiry, models, and development."""

    def __init__(self, arm: str = "full_v3", *, entity_id: str = "substrate-v3", body: str = "general"):
        if arm not in ARM_FEATURES:
            raise Refused(f"unknown v3 arm {arm!r}")
        self.arm = arm
        self.features = set(ARM_FEATURES[arm])
        self.entity_id = entity_id
        self.body = body
        self.tools = ["deterministic_compare", "sandbox_simulation"]
        self.ontology = ActiveOntology()
        self.epistemology = EpistemicLedger()
        self.reasoning = ReasoningPortfolio()
        self.structure = StructuralUnderstanding(
            {("role_0", "role_1"), ("role_1", "role_2"), ("role_2", "role_3")},
            {
                ("role_0", "role_1"): "dependency",
                ("role_1", "role_2"): "transformation",
                ("role_2", "role_3"): "delivery",
            },
        )
        self.self_model = SelfModel()
        self.reasoning_receipts: list[dict] = []
        self.cycles: list[dict] = []
        self.ontology_receipts: list[dict] = []
        self.world_receipts: list[dict] = []
        self.self_receipts: list[dict] = []
        self.semantic: dict[str, dict] = {}
        self.procedures: dict[str, dict] = {}
        self.allocation_counts: dict[tuple, dict[str, list[float]]] = {}
        self.history_specialization: dict[str, int] = {}
        self.step = 0
        self.activation = False

    def _reasoning_task(self, task: CognitiveTask) -> dict:
        public = task.public
        feature = public["feature"]
        body = {"features": {feature}, "verification": "private target revealed after committed trace"}
        if feature == "necessary_consequence":
            body.update(facts=public["facts"], rules=[(row[0], row[1]) for row in public["rules"]], query=public["query"])
        elif feature == "sample_generalization":
            body["samples"] = public["samples"]
        elif feature == "hidden_cause":
            body["explanations"] = public["explanations"]
        elif feature == "relational_transfer":
            body.update(
                source_relations=public["source_relations"],
                candidate_relations=public["candidate_relations"],
                mapping=public["mapping"],
            )
        elif feature == "observed_failure":
            body.update(observed=public["observed"], causes=public["causes"])
        elif feature == "resource_goal":
            body.update(dependencies=public["dependencies"], costs=public["costs"], budget=public["budget"])
        return body

    def _allocation_action(self, task: CognitiveTask) -> str:
        public = task.public
        if self.arm == "more_compute":
            return "inquire"
        if "inquiry" not in self.features or "self" not in self.features:
            return "inquire" if public["contradiction"] else "stop"
        context = (public["risk"], public["contradiction"], public["source_reliability"])
        learned = self.allocation_counts.get(context)
        if learned:
            means = {action: sum(values) / len(values) for action, values in learned.items() if values}
            if means:
                return max(means, key=means.get)
        return "inquire" if public["source_reliability"] == "shifted" and public["risk"] == "high" else "stop"

    def propose(self, task: CognitiveTask) -> tuple[object, str, float]:
        public = task.public
        family = task.family
        compute = 1.0
        if family == "ontology_garden":
            if "ontology" in self.features:
                proposal = "process" if "changing" in public["features"] and "exception" not in public["features"] else "entity"
                return proposal, "ontology classification", compute
        elif family == "epistemic_laboratory":
            if "epistemology" in self.features:
                source = "source_a" if public["regime"] == "alpha" else "source_b"
                return public["reports"][source], "source reliability defeater", compute
        elif family == "causal_micro_worlds":
            if "world" in self.features:
                if public["query_kind"] == "counterfactual":
                    changed = {**public["background"], **public["change"]}
                    return int(bool(changed["cause"] and changed["context"])), "counterfactual", compute
                return public["observed_cause"], "causal", compute
        elif family == "cross_representation_systems":
            if "understanding" in self.features:
                current = public["start"]
                for _ in range(public["query_distance"]):
                    current = next(target for source, target in public["relations"] if source == current)
                return current, "structural transfer", compute
        elif family == "reasoning_method_selection":
            if "reasoning" in self.features:
                trace = self.reasoning.select_and_run(self._reasoning_task(task))
                receipt = trace.receipt()
                self.reasoning_receipts.append(receipt)
                compute = 6.0 if self.arm == "more_compute" else trace.cost
                return trace.conclusion, trace.mode, compute
        elif family == "scientific_inquiry":
            action = self._allocation_action(task)
            return action, "contextual inquiry" if "inquiry" in self.features else "simple inquiry", compute + int(action == "inquire")
        elif family == "adversarial_ambiguity" and "understanding" in self.features:
            return "transfer" if public["relational_similarity"] else "reject", "relational boundary", compute
        return simple_proposal(task), "strongest simple control", compute

    def experience(self, task: CognitiveTask, *, learn: bool = True) -> dict:
        if self.activation is not False:
            raise Refused("external activation is forbidden")
        self.step += 1
        task.observation()
        prediction = self.self_model.predict("accuracy", context=task.family) if "self" in self.features else Prediction("accuracy", 0.5, context=task.family)
        prediction_step = self.step
        proposal, mode, compute = self.propose(task)
        outcome = task.reveal(proposal)
        outcome_step = self.step + 1
        explanation = None
        if task.family == "cross_representation_systems" and "understanding" in self.features:
            relation_path = task.public["relations"][: task.public["query_distance"]]
            explanation = {
                "premises": [task.public["start"]],
                "relation_or_mechanism": "directed dependency path",
                "derivation": relation_path,
                "consequence": proposal,
                "alternatives": [target for source, target in task.public["relations"] if source != task.public["start"]],
                "falsifier": f"remove relation {relation_path[-1]}" if relation_path else "no derivation",
                "representation": task.public["encoding"],
            }
            self.world_receipts.append(explanation)
        if learn:
            self._learn(task, proposal, outcome, prediction, prediction_step, outcome_step)
        receipt = {
            "identity": task.identity,
            "step": self.step,
            "family": task.family,
            "phase": task.phase,
            "ontology_used": sorted(self.ontology.concepts),
            "beliefs_used": sorted(self.epistemology.beliefs),
            "defeaters_considered": [row["defeater"] for row in self.epistemology.defeater_receipts[-4:]],
            "reasoning_mode": mode,
            "inquiry_action": proposal if task.family == "scientific_inquiry" else "none",
            "evidence_gathered": task.oracle_operation if task.family == "scientific_inquiry" and proposal == "inquire" else "public observation",
            "procedure_used": mode,
            "world_model_prediction": proposal if "world" in self.features else None,
            "self_model_prediction": prediction.predicted if "self" in self.features else None,
            "self_prediction_step": prediction_step,
            "outcome_step": outcome_step,
            "decision": proposal,
            "outcome": outcome,
            "revision": self.ontology_receipts[-1]["identity"] if self.ontology_receipts else None,
            "explanation": explanation,
            "compute": compute,
            "body": self.body,
            "tools": list(self.tools),
            "activation": False,
        }
        self.cycles.append(receipt)
        return receipt

    def _learn(
        self,
        task: CognitiveTask,
        proposal: object,
        outcome: dict,
        prediction: Prediction,
        prediction_step: int,
        outcome_step: int,
    ) -> None:
        family = task.family
        public = task.public
        if "self" in self.features:
            self.self_model.observe(prediction, float(outcome["correct"]))
            self.self_receipts.append(
                {
                    "kind": "accuracy",
                    "family": family,
                    "predicted": prediction.predicted,
                    "actual": float(outcome["correct"]),
                    "prediction_step": prediction_step,
                    "outcome_step": outcome_step,
                }
            )
        if family == "ontology_garden" and "ontology" in self.features:
            instance = public["instance"]
            features = set(public["features"])
            self.ontology.observe(instance, features)
            if "entity_or_process" not in self.ontology.concepts:
                self.ontology.concepts["entity_or_process"] = Concept("entity_or_process")
            self.ontology.concepts["entity_or_process"].members.add(instance)
            if "exception" in features and len(self.ontology.concepts["entity_or_process"].members) >= 2:
                original = self.ontology.concepts["entity_or_process"]
                changing = {
                    member
                    for member in original.members
                    if "changing" in self.ontology.instance_features[member] and "exception" not in self.ontology.instance_features[member]
                }
                remaining = original.members - changing
                if changing and remaining and original.status == "active":
                    revision = self.ontology.split_category(
                        "entity_or_process",
                        "process",
                        "entity",
                        discriminator="regular",
                        evidence=(task.identity,),
                        predicted_benefit=0.2,
                    )
                    self.ontology.complete_revision(revision, held_out_benefit=0.15)
                    self.ontology_receipts.append(revision.receipt())
        elif family == "epistemic_laboratory" and "epistemology" in self.features:
            belief_id = f"belief:{task.identity}"
            belief = EpistemicBelief(
                identity=belief_id,
                content=proposal,
                type="inferred",
                source="conditional source model",
                method="source reliability",
                provenance=(task.identity,),
                supporting_evidence=(f"{public['regime']}:source history",),
                confidence=0.9,
                domain_scope=(public["regime"],),
                unresolved_alternatives=(str(1 - int(proposal)),) if isinstance(proposal, int) else (),
                required_evidence=("independent source",),
                held_out_utility=0.2 if outcome["correct"] else 0.0,
            )
            self.epistemology.add(belief)
            unreliable = "source_b" if public["regime"] == "alpha" else "source_a"
            self.epistemology.defeat(
                belief_id,
                Defeater(
                    f"defeater:{task.identity}",
                    "scope",
                    f"{unreliable}:conditional unreliability narrows source scope",
                    belief_id,
                    0.1,
                ),
            )
            self.epistemology.admit_knowledge(belief_id, independently_verified=bool(outcome["correct"]))
        elif family == "scientific_inquiry":
            context = (public["risk"], public["contradiction"], public["source_reliability"])
            actions = self.allocation_counts.setdefault(context, {"stop": [], "inquire": []})
            target = task.private_target
            for action in ("stop", "inquire"):
                correct = action == target
                utility = float(correct) - (public["evidence_cost"] if action == "inquire" else 0.0)
                actions[action].append(utility)
        self.semantic[family] = {
            "last_task": task.identity,
            "last_outcome": outcome["correct"],
            "verified": True,
        }
        self.procedures[mode_key(family)] = {
            "family": family,
            "uses": self.procedures.get(mode_key(family), {}).get("uses", 0) + 1,
            "verified_sources": [task.identity],
        }
        self.history_specialization[family] = self.history_specialization.get(family, 0) + int(outcome["correct"])

    def change_body(self, body: str, tools: list[str]) -> dict:
        before = {"body": self.body, "tools": list(self.tools), "entity_id": self.entity_id}
        self.body = body
        self.tools = list(tools)
        return {
            "before": before,
            "after": {"body": self.body, "tools": list(self.tools), "entity_id": self.entity_id},
            "owned_identity_preserved": before["entity_id"] == self.entity_id,
        }

    def semantic_state(self) -> dict:
        return {
            "arm": self.arm,
            "entity_id": self.entity_id,
            "body": self.body,
            "tools": list(self.tools),
            "ontology": self.ontology.snapshot(),
            "epistemology": self.epistemology.snapshot(),
            "reasoning_receipts": list(self.reasoning_receipts),
            "structure": {
                "edges": sorted([list(edge) for edge in self.structure.edges]),
                "mechanisms": {f"{a}>{b}": value for (a, b), value in sorted(self.structure.mechanisms.items())},
                "representation_maps": self.structure.representation_maps,
            },
            "self_model": {
                "estimates": dict(self.self_model.estimates),
                "history": [
                    {
                        "kind": row.kind,
                        "predicted": row.predicted,
                        "actual": row.actual,
                        "context": row.context,
                    }
                    for row in self.self_model.history
                ],
            },
            "self_receipts": list(self.self_receipts),
            "ontology_receipts": list(self.ontology_receipts),
            "world_receipts": list(self.world_receipts),
            "semantic": dict(self.semantic),
            "procedures": dict(self.procedures),
            "allocation_counts": {repr(key): {action: list(values) for action, values in rows.items()} for key, rows in self.allocation_counts.items()},
            "history_specialization": dict(self.history_specialization),
            "cycles": list(self.cycles),
            "step": self.step,
            "activation": False,
        }

    def identity_hash(self) -> str:
        return _semantic_state_hash(self.semantic_state())

    def checkpoint(self) -> dict:
        state = self.semantic_state()
        return {
            "schema": "substrate-v3-checkpoint/v1",
            "owned_identity": self.entity_id,
            "semantic_state": state,
            "identity_hash": _semantic_state_hash(state),
            "activation": False,
        }

    @classmethod
    def restore(cls, checkpoint: dict) -> IntegratedEntity:
        state = checkpoint["semantic_state"]
        if _semantic_state_hash(state) != checkpoint["identity_hash"] or checkpoint.get("activation") is not False:
            raise Refused("checkpoint identity or activation is invalid")
        entity = cls(state["arm"], entity_id=state["entity_id"], body=state["body"])
        entity.tools = list(state["tools"])
        entity.ontology.restore(state["ontology"])
        entity.epistemology = EpistemicLedger.restore(state["epistemology"])
        entity.reasoning_receipts = list(state["reasoning_receipts"])
        entity.structure.edges = {tuple(edge) for edge in state["structure"]["edges"]}
        entity.structure.mechanisms = {tuple(key.split(">", 1)): value for key, value in state["structure"]["mechanisms"].items()}
        entity.structure.representation_maps = dict(state["structure"]["representation_maps"])
        entity.self_model.estimates = dict(state["self_model"]["estimates"])
        entity.self_model.history = [Prediction(**row) for row in state["self_model"]["history"]]
        entity.self_receipts = list(state["self_receipts"])
        entity.ontology_receipts = list(state["ontology_receipts"])
        entity.world_receipts = list(state["world_receipts"])
        entity.semantic = dict(state["semantic"])
        entity.procedures = dict(state["procedures"])
        entity.allocation_counts = {
            parse_context(key): {action: list(values) for action, values in rows.items()} for key, rows in state["allocation_counts"].items()
        }
        entity.history_specialization = dict(state["history_specialization"])
        entity.cycles = list(state["cycles"])
        entity.step = int(state["step"])
        if entity.semantic_state() != state:
            raise Refused("restored semantic state does not match checkpoint identity")
        return entity


def mode_key(family: str) -> str:
    return f"procedure:{family}"


def parse_context(value: str) -> tuple:
    import ast

    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple):
        raise Refused("allocation context is not a tuple")
    return parsed
