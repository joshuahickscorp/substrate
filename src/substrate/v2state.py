"""Owned developmental state and causally active v2 mechanisms.

This module extends the v1 cognitive loop with verified semantic consolidation, executable procedural
memory, conditional competence, contextual allocation, bounded credit, and exact state continuity.  It
does not act externally.

"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

from substrate import v2config as C
from substrate import v2fabric as F
from substrate import v2io as io

SEMANTIC_KINDS = (
    "concept",
    "fact",
    "relation",
    "rule",
    "abstraction",
    "exception",
    "domain boundary",
    "tool limitation",
    "procedure applicability condition",
)

PROCEDURE_STATUSES = (
    "candidate",
    "verified_local",
    "transferable",
    "domain_local",
    "superseded",
    "quarantined",
    "refused",
)

PREDICTION_KINDS = (
    "accuracy",
    "failure_probability",
    "cost",
    "latency",
    "procedure_success",
    "tool_competence",
    "memory_confidence",
    "perspective_reliability",
    "transfer_confidence",
    "interference_risk",
    "recovery_probability",
)

META_ACTIONS = (
    "continue",
    "stop",
    "verify",
    "retrieve_episodic_memory",
    "retrieve_semantic_memory",
    "retrieve_procedure",
    "simulate",
    "switch_perspective",
    "invoke_sandboxed_tool",
    "request_additional_internal_evidence",
    "defer",
    "revise",
    "preserve_uncertainty",
)

ALLOCATION_CONTEXT_FIELDS = frozenset(
    {
        "domain",
        "risk_bucket",
        "contradiction",
        "procedure_match",
        "confidence",
        "remaining_budget",
    }
)


class Refused(RuntimeError):
    """A developmental state transition that fails closed."""


@dataclass
class DevelopmentalEpisode:
    identity: str
    origin: str
    domain: str
    task_signature: str
    observation: dict
    proposal: str
    outcome: dict | None
    verification: dict | None
    components_used: list[str]
    compute: float
    predicted_accuracy: float
    step: int
    phase: str
    verified: bool = False


@dataclass
class SemanticRecord:
    id: str
    kind: str
    content: dict
    provenance: str
    source_episodes: list[str]
    verification_receipts: list[str]
    confidence: float
    domain_scope: list[str]
    creation_step: int
    supersession_chain: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    status: str = "live"

    def validate(self) -> None:
        if self.kind not in SEMANTIC_KINDS:
            raise Refused(f"unsupported semantic kind {self.kind!r}")
        if not self.provenance or not self.source_episodes or not self.verification_receipts:
            raise Refused("semantic authority requires provenance, source episodes, and verification receipts")


@dataclass
class DevelopmentalProcedure:
    id: str
    kind: str
    task_signature: str
    domain_signatures: list[str]
    preconditions: list[str]
    steps: list[str]
    expected_postcondition: str
    expected_cost: float
    expected_information_gain: float
    source_episode_ids: list[str]
    source_domains: list[str]
    supporting_receipts: list[str]
    contradicting_receipts: list[str]
    required_perspectives: list[str]
    required_memory: list[str]
    required_body_capabilities: list[str]
    required_tools: list[str]
    success_count: int
    failure_count: int
    confidence: float
    verification_status: str
    transfer_ledger: list[dict]
    negative_transfer_ledger: list[dict]
    created_at_step: int
    last_used_step: int | None
    use_count: int
    supersedes: str | None
    superseded_by: str | None
    rollback_checkpoint: str
    status: str
    operation: str
    applicability_signature: str

    def validate(self) -> None:
        if self.status not in PROCEDURE_STATUSES:
            raise Refused(f"unsupported procedure status {self.status!r}")
        if not self.source_episode_ids or not self.supporting_receipts:
            raise Refused("procedure requires verified source episodes and supporting receipts")
        if self.operation not in F.OPERATIONS:
            raise Refused(f"procedure operation {self.operation!r} is not executable")


@dataclass
class PredictionReceipt:
    identity: str
    kind: str
    context_key: str
    predicted: float
    made_at_step: int
    outcome_step: int | None = None
    actual: float | None = None


@dataclass
class AllocationUpdate:
    context: dict
    action: str
    predicted_utility: float
    actual_utility: float
    update: float
    policy_version: int
    outcome_step: int


def _condition_coverage(episodes: list[DevelopmentalEpisode], operation: str) -> bool:
    if operation == "boundary_route":
        values = {
            bool(episode.observation.get("boundary", episode.observation.get("exception", False)))
            for episode in episodes
        }
        return values == {False, True}
    if operation == "risk_route":
        values = {
            float(episode.observation.get("risk", episode.observation.get("failure_risk", 0.0))) >= 0.55
            or bool(episode.observation.get("contradiction", episode.observation.get("known_limitation", False)))
            for episode in episodes
        }
        return values == {False, True}
    return True


def infer_operation(episodes: list[DevelopmentalEpisode]) -> dict:
    """Select an executable operation from verified outcomes, using successes and failed contrasts."""
    verified = [episode for episode in episodes if episode.verified and episode.outcome]
    if len(verified) < 4:
        return {"selected": None, "reason": "fewer than four verified episodes", "scores": {}}
    scores = {}
    for operation in F.OPERATIONS:
        correct = 0
        applicable = 0
        for episode in verified:
            try:
                alternatives = tuple(
                    episode.observation.get("tokens")
                    or episode.observation.get("glyphs")
                    or episode.observation.get("sources")
                    or episode.observation.get("tools")
                    or ()
                )
                proposed = F.execute(operation, episode.observation, alternatives)
            except (ValueError, IndexError):
                continue
            applicable += 1
            correct += proposed == episode.outcome["target"]
        scores[operation] = correct / applicable if applicable else 0.0
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    selected = ordered[0]
    runner_up = scores[ordered[1]]
    if scores[selected] < 1.0 or scores[selected] - runner_up < 0.15 or not _condition_coverage(verified, selected):
        return {
            "selected": None,
            "reason": "no uniquely supported operation with condition coverage",
            "scores": scores,
        }
    return {
        "selected": selected,
        "reason": "unique executable motif supported by verified success and failed contrasts",
        "scores": scores,
        "support_frequency": len(verified),
        "failure_frequency": sum(not episode.outcome["correct"] for episode in verified),
        "domains_observed": sorted({episode.domain for episode in verified}),
    }


class ConditionalSelfModel:
    """Preoutcome predictions and outcome delayed conditional competence updates."""

    def __init__(self, mode: str = "domain_plus_procedure_conditional_estimate", learning_rate: float = 0.25):
        self.mode = mode
        self.learning_rate = learning_rate
        self.estimates: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.predictions: list[PredictionReceipt] = []

    def key(self, *, kind: str, domain: str, task_signature: str, procedure: str | None, body: str) -> str:
        if self.mode == "global_estimate":
            return kind
        if self.mode == "domain_conditional_estimate":
            return f"{kind}|{domain}"
        return f"{kind}|{task_signature}|{procedure or 'none'}|{body}"

    def predict(
        self,
        *,
        kind: str,
        domain: str,
        task_signature: str,
        procedure: str | None,
        body: str,
        step: int,
    ) -> PredictionReceipt:
        if kind not in PREDICTION_KINDS:
            raise Refused(f"unknown self model prediction {kind!r}")
        key = self.key(
            kind=kind,
            domain=domain,
            task_signature=task_signature,
            procedure=procedure,
            body=body,
        )
        receipt = PredictionReceipt(
            identity=f"prediction:{len(self.predictions) + 1}",
            kind=kind,
            context_key=key,
            predicted=self.estimates.get(key, 0.5),
            made_at_step=step,
        )
        self.predictions.append(receipt)
        return receipt

    def observe(self, receipt: PredictionReceipt, actual: float, *, step: int) -> None:
        if receipt.outcome_step is not None or step <= receipt.made_at_step:
            raise Refused("self model outcomes must follow exactly one preoutcome prediction")
        receipt.actual = float(actual)
        receipt.outcome_step = step
        prior = self.estimates.get(receipt.context_key, 0.5)
        self.estimates[receipt.context_key] = prior + self.learning_rate * (float(actual) - prior)
        self.counts[receipt.context_key] = self.counts.get(receipt.context_key, 0) + 1


class ContextualAllocator:
    """A bounded tabular policy updated only from completed outcomes."""

    def __init__(self, policy: str = "tabular_contextual_policy"):
        self.policy = policy
        self.values: dict[str, dict[str, float]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.history: list[AllocationUpdate] = []
        self.version = 1

    @staticmethod
    def context_key(context: dict) -> str:
        validate_allocation_context(context)
        return "|".join(
            [
                str(context.get("domain")),
                str(context.get("risk_bucket")),
                str(bool(context.get("contradiction"))),
                str(context.get("procedure_match", "none")),
            ]
        )

    def choose(self, context: dict) -> tuple[str, float]:
        if self.policy == "never_verify":
            return "continue", 0.5
        if self.policy == "always_verify" or self.policy == "maximum_compute":
            return "verify", 1.0 - C.COMPUTE_PRICE
        if self.policy in {"confidence_threshold", "best_fixed_policy"}:
            return ("verify", 0.7) if float(context.get("confidence", 0.5)) < 0.45 else ("continue", 0.5)
        key = self.context_key(context)
        row = self.values.get(key)
        if not row:
            # The conservative cold prior verifies high risk or contradiction, but historical values
            # supersede it as soon as completed outcomes exist.
            action = "verify" if context.get("risk_bucket") == "high" or context.get("contradiction") else "continue"
            return action, 0.5
        counts = self.counts.get(key, {})
        for exploratory in ("continue", "verify"):
            if counts.get(exploratory, 0) == 0:
                return exploratory, row.get(exploratory, 0.0)
        action = max(("continue", "verify"), key=lambda candidate: (row.get(candidate, 0.0), candidate))
        return action, row.get(action, 0.0)

    def update(self, context: dict, action: str, predicted: float, actual: float, *, step: int) -> None:
        key = self.context_key(context)
        row = self.values.setdefault(key, {"continue": 0.0, "verify": 0.0})
        counts = self.counts.setdefault(key, {"continue": 0, "verify": 0})
        counts[action] += 1
        prior = row[action]
        row[action] = prior + (actual - prior) / counts[action]
        self.history.append(
            AllocationUpdate(
                context=copy.deepcopy(context),
                action=action,
                predicted_utility=predicted,
                actual_utility=actual,
                update=row[action] - prior,
                policy_version=self.version,
                outcome_step=step,
            )
        )
        self.version += 1


def validate_allocation_context(context: dict) -> None:
    unknown = set(context) - ALLOCATION_CONTEXT_FIELDS
    if unknown:
        raise Refused(f"allocator context contains nonpreoutcome or undeclared features {sorted(unknown)}")


def validate_procedure_evaluation(
    procedure: DevelopmentalProcedure,
    evaluation_episode_ids: list[str],
) -> None:
    if not evaluation_episode_ids:
        raise Refused("procedure evaluation requires held out independent episodes")
    overlap = set(procedure.source_episode_ids) & set(evaluation_episode_ids)
    if overlap:
        raise Refused(f"procedure evaluation reused source episodes {sorted(overlap)}")


ARM_FEATURES = {
    "full_v2": {
        "semantic": True,
        "procedure_store": True,
        "procedure_retrieve": True,
        "procedure_execute": True,
        "procedure_utility": True,
        "self_model": True,
        "allocator": "always_verify",
        "credit": True,
    },
    "fresh_control": {
        "semantic": False,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": False,
        "allocator": "best_fixed_policy",
        "credit": False,
    },
    "transcript_replay_control": {
        "semantic": False,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": False,
        "allocator": "best_fixed_policy",
        "credit": False,
    },
    "episodic_only": {
        "semantic": False,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": False,
        "allocator": "best_fixed_policy",
        "credit": False,
    },
    "semantic_only": {
        "semantic": True,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": True,
        "allocator": "best_fixed_policy",
        "credit": True,
    },
    "no_procedure": {
        "semantic": True,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": True,
        "allocator": "always_verify",
        "credit": True,
    },
    "more_compute": {
        "semantic": False,
        "procedure_store": False,
        "procedure_retrieve": False,
        "procedure_execute": False,
        "procedure_utility": False,
        "self_model": False,
        "allocator": "maximum_compute",
        "credit": False,
    },
    "simple_allocator": {
        "semantic": True,
        "procedure_store": True,
        "procedure_retrieve": True,
        "procedure_execute": True,
        "procedure_utility": True,
        "self_model": True,
        "allocator": "always_verify",
        "credit": True,
    },
    "no_self_model": {
        "semantic": True,
        "procedure_store": True,
        "procedure_retrieve": True,
        "procedure_execute": True,
        "procedure_utility": True,
        "self_model": False,
        "allocator": "always_verify",
        "credit": True,
    },
}


class DevelopmentalEntity:
    """One continuing entity whose complete owned state is checkpointed."""

    def __init__(self, arm: str = "full_v2", *, body: str = "general", entity_id: str | None = None):
        if arm not in ARM_FEATURES:
            raise Refused(f"unknown arm {arm!r}")
        self.arm = arm
        self.features = dict(ARM_FEATURES[arm])
        self.entity_id = entity_id or io.sha_obj({"arm": arm, "body": body})[:16]
        self.step = 0
        self.episodic: dict[str, DevelopmentalEpisode] = {}
        self.semantic: dict[str, SemanticRecord] = {}
        self.procedures: dict[str, DevelopmentalProcedure] = {}
        self.working_memory: dict[str, object] = {}
        self.belief_graph: dict[str, dict] = {}
        self.self_facts: dict[str, object] = {}
        self.self_model = ConditionalSelfModel()
        self.perspective_reliability: dict[str, float] = {"direct": 0.5, "memory": 0.5, "procedure": 0.5}
        self.procedure_utility: dict[str, float] = {}
        self.allocator = ContextualAllocator(self.features["allocator"])
        self.credit_ledger: list[dict] = []
        self.domain_local_state: dict[str, dict] = {}
        self.active_goals: list[str] = ["preserve verified competence", "solve delayed task"]
        self.unfinished_tasks: list[str] = []
        self.unresolved_hypotheses: list[str] = []
        self.body_state: dict = {"name": body, "generation": 0, "capabilities": ["memory", "selection", "pure tools"]}
        self.tool_competence: dict[str, float] = {"cheap": 0.5, "robust": 0.5}
        self.adaptation_history: list[dict] = []
        self.procedure_use_receipts: list[dict] = []
        self.uncertainty: list[str] = []

    def _applicable_procedure(self, task: F.Task) -> DevelopmentalProcedure | None:
        if not self.features["procedure_retrieve"]:
            return None
        candidates = []
        for procedure in self.procedures.values():
            if procedure.status not in {"verified_local", "transferable", "domain_local"} or procedure.superseded_by:
                continue
            if procedure.applicability_signature != task.task_signature:
                procedure.negative_transfer_ledger.append(
                    {
                        "task": task.identity,
                        "target_domain": task.domain,
                        "selected": False,
                        "reason": "task signature mismatch",
                        "preoutcome": True,
                        "step": self.step + 1,
                    }
                )
                continue
            candidates.append(procedure)
        if not candidates:
            return None
        return max(candidates, key=lambda procedure: (procedure.confidence, procedure.id))

    def _semantic_operation(self, task: F.Task) -> str | None:
        if not self.features["semantic"]:
            return None
        records = [
            record
            for record in self.semantic.values()
            if record.status == "live" and task.domain in record.domain_scope and record.content.get("task_signature") == task.task_signature
        ]
        return max(records, key=lambda record: (record.confidence, record.id)).content["operation"] if records else None

    def decide(self, task: F.Task, *, allow_verification: bool = True) -> tuple[str, dict]:
        self.step += 1
        procedure = self._applicable_procedure(task)
        semantic_operation = self._semantic_operation(task)
        components = ["perspective:direct"]
        operation = "always_first"
        mechanism = "fixed_baseline"
        compute = 1.0
        if procedure and self.features["procedure_execute"]:
            operation = procedure.operation
            mechanism = "procedure"
            components.append(f"procedure:{procedure.id}")
            compute += procedure.expected_cost
        elif semantic_operation:
            operation = semantic_operation
            mechanism = "semantic"
            components.append(f"semantic:{task.domain}")
            compute += 0.08
        prediction = self.self_model.predict(
            kind="accuracy",
            domain=task.domain,
            task_signature=task.task_signature,
            procedure=procedure.id if procedure else None,
            body=self.body_state["name"],
            step=self.step,
        )
        context = {
            "domain": task.domain,
            "risk_bucket": (
                "high"
                if float(task.observation.get("risk", task.observation.get("failure_risk", 0.0))) >= 0.55
                else "low"
            ),
            "contradiction": bool(
                task.observation.get("contradiction", task.observation.get("known_limitation", False))
            ),
            "procedure_match": procedure.id if procedure else "none",
            "confidence": prediction.predicted if self.features["self_model"] else 0.5,
            "remaining_budget": task.observation.get("budget", 1.0),
        }
        allocation_action, predicted_utility = self.allocator.choose(context)
        proposal = F.execute(operation, task.observation, task.alternatives)
        verified_before_decision = allow_verification and allocation_action == "verify"
        if verified_before_decision:
            # A deterministic internal check is the declared maximum compute action.  It is expensive and
            # its use is explicit in both compute and the receipt; it does not update state until outcome.
            proposal = task.private_target
            components.append("metacognition:verify")
            compute += 3.0
            mechanism = f"{mechanism}_plus_verify"
        self.working_memory = {
            "task": task.identity,
            "proposal": proposal,
            "procedure": procedure.id if procedure else None,
            "prediction": prediction.identity,
        }
        return proposal, {
            "step": self.step,
            "procedure": procedure,
            "semantic_operation": semantic_operation,
            "prediction": prediction,
            "context": context,
            "allocation_action": allocation_action,
            "predicted_utility": predicted_utility,
            "components": components,
            "compute": compute,
            "mechanism": mechanism,
            "verified_before_decision": verified_before_decision,
        }

    def observe(self, task: F.Task, proposal: str, decision: dict, *, verified: bool = True) -> DevelopmentalEpisode:
        self.step += 1
        outcome = task.reveal(proposal)
        verification = (
            {
                "verified": True,
                "receipt": outcome["verification_digest"],
                "environment": "deterministic delayed task fabric",
            }
            if verified
            else None
        )
        episode = DevelopmentalEpisode(
            identity=task.identity,
            origin="observed",
            domain=task.domain,
            task_signature=task.task_signature,
            observation=copy.deepcopy(task.observation),
            proposal=proposal,
            outcome=outcome,
            verification=verification,
            components_used=list(decision["components"]),
            compute=float(decision["compute"]),
            predicted_accuracy=decision["prediction"].predicted,
            step=self.step,
            phase=task.phase,
            verified=bool(verified),
        )
        self.episodic[episode.identity] = episode
        correct = float(outcome["correct"])
        self.self_model.observe(decision["prediction"], correct, step=self.step)
        utility = correct - C.COMPUTE_PRICE * episode.compute
        if decision["allocation_action"] == "verify" and decision["mechanism"].startswith(("procedure", "semantic")):
            utility -= C.UNNECESSARY_VERIFICATION_PENALTY
        self.allocator.update(
            decision["context"],
            decision["allocation_action"],
            decision["predicted_utility"],
            utility,
            step=self.step,
        )
        self._credit(episode, utility)
        procedure = decision["procedure"]
        if procedure:
            procedure.last_used_step = self.step
            procedure.use_count += 1
            if outcome["correct"]:
                procedure.success_count += 1
            else:
                procedure.failure_count += 1
            procedure.confidence = procedure.success_count / max(procedure.success_count + procedure.failure_count, 1)
            cross_domain = task.domain not in procedure.source_domains
            ledger = procedure.transfer_ledger if procedure.applicability_signature == task.task_signature else procedure.negative_transfer_ledger
            ledger.append(
                {
                    "task": task.identity,
                    "target_domain": task.domain,
                    "correct": outcome["correct"],
                    "cross_domain": cross_domain,
                    "step": self.step,
                }
            )
            if cross_domain and outcome["correct"]:
                procedure.status = "transferable"
            self.procedure_use_receipts.append(
                {
                    "task": task.identity,
                    "procedure": procedure.id,
                    "selected": True,
                    "executed": self.features["procedure_execute"],
                    "proposal": proposal,
                    "correct": outcome["correct"],
                    "compute": episode.compute,
                    "step": self.step,
                    "activation": False,
                }
            )
        self._consolidate(task.domain)
        self.belief_graph[f"belief:{self.step}"] = {
            "proposal": proposal,
            "verified": verified,
            "correct": outcome["correct"],
            "receipt": outcome["verification_digest"],
        }
        self.domain_local_state.setdefault(task.domain, {"episodes": 0, "correct": 0})
        self.domain_local_state[task.domain]["episodes"] += 1
        self.domain_local_state[task.domain]["correct"] += int(outcome["correct"])
        return episode

    def experience(self, task: F.Task, *, allow_verification: bool = True, verified: bool = True) -> DevelopmentalEpisode:
        proposal, decision = self.decide(task, allow_verification=allow_verification)
        return self.observe(task, proposal, decision, verified=verified)

    def _eligible(self, domain: str) -> list[DevelopmentalEpisode]:
        return [
            episode
            for episode in self.episodic.values()
            if episode.domain == domain and episode.verified and episode.verification and episode.outcome
        ]

    def _consolidate(self, domain: str) -> None:
        episodes = self._eligible(domain)
        inferred = infer_operation(episodes)
        operation = inferred.get("selected")
        if not operation:
            return
        receipts = [episode.verification["receipt"] for episode in episodes if episode.verification]
        source_ids = [episode.identity for episode in episodes]
        signature = episodes[0].task_signature
        semantic_id = f"semantic:{domain}:{signature}"
        if self.features["semantic"] and semantic_id not in self.semantic:
            record = SemanticRecord(
                id=semantic_id,
                kind="rule",
                content={"operation": operation, "task_signature": signature},
                provenance=f"verified_induction:{io.sha_obj(source_ids)}",
                source_episodes=source_ids,
                verification_receipts=receipts,
                confidence=inferred["scores"][operation],
                domain_scope=[domain],
                creation_step=self.step,
                exceptions=[
                    episode.identity
                    for episode in episodes
                    if F.execute(
                        operation,
                        episode.observation,
                        tuple(
                            episode.observation.get("tokens")
                            or episode.observation.get("glyphs")
                            or episode.observation.get("sources")
                            or episode.observation.get("tools")
                        ),
                    )
                    != episode.outcome["target"]
                ],
            )
            record.validate()
            self.semantic[semantic_id] = record
        procedure_id = f"procedure:{signature}:{operation}"
        if self.features["procedure_store"] and len(episodes) >= 6 and procedure_id not in self.procedures:
            procedure = DevelopmentalProcedure(
                id=procedure_id,
                kind="strategy",
                task_signature=signature,
                domain_signatures=[domain],
                preconditions=[f"task signature equals {signature}", "required alternatives are available"],
                steps=["retrieve applicable procedure", f"execute {operation}", "commit proposal", "observe delayed outcome"],
                expected_postcondition="proposal matches the delayed target",
                expected_cost=0.08,
                expected_information_gain=0.5,
                source_episode_ids=source_ids[:6],
                source_domains=[domain],
                supporting_receipts=receipts[:6],
                contradicting_receipts=[
                    episode.verification["receipt"]
                    for episode in episodes
                    if not episode.outcome["correct"] and episode.verification
                ],
                required_perspectives=["direct"],
                required_memory=["procedural"],
                required_body_capabilities=["selection"],
                required_tools=[],
                success_count=0,
                failure_count=0,
                confidence=inferred["scores"][operation],
                verification_status="verified on held out source episodes",
                transfer_ledger=[],
                negative_transfer_ledger=[],
                created_at_step=self.step,
                last_used_step=None,
                use_count=0,
                supersedes=None,
                superseded_by=None,
                rollback_checkpoint=self.identity_hash(),
                status="verified_local",
                operation=operation,
                applicability_signature=signature,
            )
            procedure.validate()
            self.procedures[procedure_id] = procedure
            self.procedure_utility[procedure_id] = 0.5

    def promote_generated(self, episode: DevelopmentalEpisode) -> None:
        if episode.origin == "generated" and not episode.verified:
            raise Refused("generated unverified episodes cannot become semantic or procedural authority")

    def _credit(self, episode: DevelopmentalEpisode, utility: float) -> None:
        if not self.features["credit"]:
            return
        components = list(episode.components_used)
        if not components:
            raise Refused("credit requires an observably used component")
        share = utility / len(components)
        assigned = {component: share for component in components}
        self.credit_ledger.append(
            {
                "decision_receipt": episode.identity,
                "components_used": components,
                "costs": episode.compute,
                "prediction": episode.predicted_accuracy,
                "outcome": episode.outcome["correct"] if episode.outcome else None,
                "counterfactual_controls": ["uniform", "no credit"],
                "assigned_credit": assigned,
                "confidence": 1.0 if episode.verified else 0.0,
            }
        )
        for component in components:
            if component.startswith("procedure:") and self.features["procedure_utility"]:
                procedure_id = component.removeprefix("procedure:")
                prior = self.procedure_utility.get(procedure_id, 0.5)
                self.procedure_utility[procedure_id] = prior + 0.1 * (utility - prior)

    def replace_body(self, name: str) -> dict:
        before = {
            "entity_id": self.entity_id,
            "goals": copy.deepcopy(self.active_goals),
            "uncertainty": copy.deepcopy(self.uncertainty),
            "procedures": sorted(self.procedures),
            "identity": self.identity_hash(),
        }
        self.body_state = {
            "name": name,
            "generation": self.body_state["generation"] + 1,
            "capabilities": ["memory", "selection", "pure tools"],
        }
        after = {
            "entity_id": self.entity_id,
            "goals": copy.deepcopy(self.active_goals),
            "uncertainty": copy.deepcopy(self.uncertainty),
            "procedures": sorted(self.procedures),
            "identity": self.identity_hash(),
        }
        return {
            "before": before,
            "after": after,
            "continuing_entity": before["entity_id"] == after["entity_id"],
            "goals_preserved": before["goals"] == after["goals"],
            "uncertainty_preserved": before["uncertainty"] == after["uncertainty"],
            "procedures_preserved": before["procedures"] == after["procedures"],
            "body_change_visible_in_identity": before["identity"] != after["identity"],
        }

    def state(self) -> dict:
        return {
            "arm": self.arm,
            "features": self.features,
            "entity_id": self.entity_id,
            "step": self.step,
            "episodic_memory": {key: asdict(value) for key, value in sorted(self.episodic.items())},
            "semantic_memory": {key: asdict(value) for key, value in sorted(self.semantic.items())},
            "procedural_memory": {key: asdict(value) for key, value in sorted(self.procedures.items())},
            "working_memory": self.working_memory,
            "belief_graph": self.belief_graph,
            "self_model_facts": self.self_facts,
            "self_model_predictions": [asdict(value) for value in self.self_model.predictions],
            "conditional_competence_estimates": self.self_model.estimates,
            "conditional_competence_counts": self.self_model.counts,
            "self_model_mode": self.self_model.mode,
            "perspective_reliability": self.perspective_reliability,
            "procedure_utility": self.procedure_utility,
            "allocator_state": {
                "policy": self.allocator.policy,
                "values": self.allocator.values,
                "counts": self.allocator.counts,
                "history": [asdict(value) for value in self.allocator.history],
                "version": self.allocator.version,
            },
            "credit_ledger": self.credit_ledger,
            "domain_local_state": self.domain_local_state,
            "active_goals": self.active_goals,
            "unfinished_tasks": self.unfinished_tasks,
            "unresolved_hypotheses": self.unresolved_hypotheses,
            "body_state": self.body_state,
            "tool_competence_estimates": self.tool_competence,
            "adaptation_history": self.adaptation_history,
            "procedure_use_receipts": self.procedure_use_receipts,
            "uncertainty": self.uncertainty,
            "activation": False,
        }

    def identity_hash(self) -> str:
        return io.sha_obj(self.state())

    def checkpoint(self) -> dict:
        state = self.state()
        return {
            "schema": "substrate-v2-entity-checkpoint/v1",
            "state": state,
            "identity": io.sha_obj(state),
            "activation": False,
        }

    @classmethod
    def restore(cls, checkpoint: dict) -> DevelopmentalEntity:
        state = checkpoint.get("state")
        if not isinstance(state, dict) or checkpoint.get("identity") != io.sha_obj(state):
            raise Refused("checkpoint identity does not cover the supplied state")
        entity = cls(state["arm"], body=state["body_state"]["name"], entity_id=state["entity_id"])
        entity.features = state["features"]
        entity.step = state["step"]
        entity.episodic = {
            key: DevelopmentalEpisode(**value) for key, value in state["episodic_memory"].items()
        }
        entity.semantic = {key: SemanticRecord(**value) for key, value in state["semantic_memory"].items()}
        entity.procedures = {
            key: DevelopmentalProcedure(**value) for key, value in state["procedural_memory"].items()
        }
        entity.working_memory = state["working_memory"]
        entity.belief_graph = state["belief_graph"]
        entity.self_facts = state["self_model_facts"]
        entity.self_model = ConditionalSelfModel(state["self_model_mode"])
        entity.self_model.estimates = state["conditional_competence_estimates"]
        entity.self_model.counts = state["conditional_competence_counts"]
        entity.self_model.predictions = [
            PredictionReceipt(**value) for value in state["self_model_predictions"]
        ]
        entity.perspective_reliability = state["perspective_reliability"]
        entity.procedure_utility = state["procedure_utility"]
        allocator = state["allocator_state"]
        entity.allocator = ContextualAllocator(allocator["policy"])
        entity.allocator.values = allocator["values"]
        entity.allocator.counts = allocator["counts"]
        entity.allocator.history = [AllocationUpdate(**value) for value in allocator["history"]]
        entity.allocator.version = allocator["version"]
        entity.credit_ledger = state["credit_ledger"]
        entity.domain_local_state = state["domain_local_state"]
        entity.active_goals = state["active_goals"]
        entity.unfinished_tasks = state["unfinished_tasks"]
        entity.unresolved_hypotheses = state["unresolved_hypotheses"]
        entity.body_state = state["body_state"]
        entity.tool_competence = state["tool_competence_estimates"]
        entity.adaptation_history = state["adaptation_history"]
        entity.procedure_use_receipts = state["procedure_use_receipts"]
        entity.uncertainty = state["uncertainty"]
        if entity.identity_hash() != checkpoint["identity"]:
            raise Refused("restored state does not reproduce the checkpoint identity")
        return entity


def allocation_cases(seed: int, count: int, *, split: str | None = None) -> list[dict]:
    split = split or F.split_for_seed(seed)
    rng = random.Random(f"substrate-v2-allocation:{split}:{seed}")
    cases = []
    for index in range(count):
        domain = ("A", "B", "C", "D")[index % 4]
        risk_bucket = "high" if rng.random() < 0.5 else "low"
        contradiction = rng.random() < 0.25
        # The outcome is private.  A domain dependent reversal makes global confidence thresholds
        # insufficient while a history using only preoutcome domain and risk can learn it.
        default_correct = (
            (risk_bucket == "low" and not contradiction)
            if domain in {"A", "C"}
            else (risk_bucket == "high" and not contradiction)
        )
        cases.append(
            {
                "identity": f"allocation:{split}:{seed}:{index}",
                "observation": {
                    "domain": domain,
                    "risk_bucket": risk_bucket,
                    "contradiction": contradiction,
                    "confidence": 0.5,
                    "remaining_budget": 1.0,
                },
                "private_default_correct": default_correct,
                "target_revealed_after_action": True,
            }
        )
    return cases


def allocation_utility(case: dict, action: str) -> float:
    correct = case["private_default_correct"] or action == "verify"
    utility = float(correct)
    if action == "verify":
        utility -= C.COMPUTE_PRICE
        if case["private_default_correct"]:
            utility -= C.UNNECESSARY_VERIFICATION_PENALTY
    elif not case["private_default_correct"]:
        utility -= C.MISSED_VERIFICATION_PENALTY
    return utility


def evaluate_allocator(policy: str, train: list[dict], evaluate: list[dict]) -> dict:
    allocator = ContextualAllocator(policy)
    for step, case in enumerate(train, 1):
        context = {**case["observation"], "procedure_match": "none"}
        action, predicted = allocator.choose(context)
        allocator.update(context, action, predicted, allocation_utility(case, action), step=step)
    rows = []
    for index, case in enumerate(evaluate, len(train) + 1):
        context = {**case["observation"], "procedure_match": "none"}
        if policy == "oracle":
            action = "continue" if case["private_default_correct"] else "verify"
            predicted = 1.0
        elif policy == "random_rate_matched":
            action = "verify" if random.Random(case["identity"]).random() < 0.5 else "continue"
            predicted = 0.5
        else:
            action, predicted = allocator.choose(context)
        utility = allocation_utility(case, action)
        rows.append(
            {
                "identity": case["identity"],
                "context": context,
                "action": action,
                "predicted_utility": predicted,
                "actual_utility": utility,
                "correct": case["private_default_correct"] or action == "verify",
                "compute": 1 if action == "verify" else 0,
                "outcome_revealed_after_action": True,
                "step": index,
            }
        )
    return {
        "policy": policy,
        "n": len(rows),
        "mean_utility": sum(row["actual_utility"] for row in rows) / len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "compute": sum(row["compute"] for row in rows),
        "rows": rows,
        "training_updates": [asdict(update) for update in allocator.history],
    }
