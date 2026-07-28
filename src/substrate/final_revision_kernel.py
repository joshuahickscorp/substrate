"""Persistent kernels and bounded architecture prototypes for the final revision."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_io as io

EVENT_KINDS = (
    "observation",
    "memory",
    "belief",
    "knowledge",
    "goal",
    "world",
    "self",
    "reasoning",
    "inquiry",
    "model_register",
    "model_remove",
    "model_replace",
    "body_tool",
    "learning_propose",
    "learning_admit",
    "learning_rollback",
    "interrupt",
    "correction",
)

MEMORY_TYPES = (
    "working",
    "episodic",
    "semantic",
    "procedural",
    "perceptual",
    "structural",
    "developmental",
)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def learning_evaluation_receipt(
    update_id: str,
    *,
    held_out_before: Iterable[bool],
    held_out_after: Iterable[bool],
    retention_before: Iterable[bool],
    retention_after: Iterable[bool],
    evaluator_id: str = "frozen-construction-evaluator/v1",
) -> dict[str, Any]:
    """Compute a content-addressed learning receipt from raw evaluator outcomes."""
    raw: dict[str, Any] = {
        "update_id": str(update_id),
        "evaluator_id": str(evaluator_id),
        "held_out_before": [bool(value) for value in held_out_before],
        "held_out_after": [bool(value) for value in held_out_after],
        "retention_before": [bool(value) for value in retention_before],
        "retention_after": [bool(value) for value in retention_after],
        "activation": False,
    }
    if (
        not raw["held_out_before"]
        or len(raw["held_out_before"]) != len(raw["held_out_after"])
        or not raw["retention_before"]
        or len(raw["retention_before"]) != len(raw["retention_after"])
    ):
        raise io.Refused("learning evaluation requires nonempty paired raw outcomes")

    def rate(values: list[bool]) -> float:
        return sum(values) / len(values)

    document = {
        **raw,
        "computed": {
            "held_out_before": rate(raw["held_out_before"]),
            "held_out_after": rate(raw["held_out_after"]),
            "retention_before": rate(raw["retention_before"]),
            "retention_after": rate(raw["retention_after"]),
        },
    }
    document["evaluation_digest"] = io.digest(document)
    return document


@dataclass(frozen=True)
class CognitiveEvent:
    sequence: int
    logical_time: int
    kind: str
    payload: dict[str, Any]
    provenance: str
    previous_digest: str
    digest: str

    def document(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "logical_time": self.logical_time,
            "kind": self.kind,
            "payload": _copy(self.payload),
            "provenance": self.provenance,
            "previous_digest": self.previous_digest,
            "digest": self.digest,
        }


def _initial_state(identity: str) -> dict[str, Any]:
    return {
        "identity": {"id": identity, "origin": "operator", "continuity_epoch": 0},
        "time": {"logical": 0, "last_event_sequence": 0},
        "observations": [],
        "memory": {kind: {} for kind in MEMORY_TYPES},
        "beliefs": {},
        "knowledge": {},
        "goals": {},
        "world_model": {"objects": {}, "relations": [], "causal_edges": [], "counterfactuals": [], "interventions": []},
        "self_model": {"competence": {}, "limits": [], "resource_state": {}, "predictions": []},
        "reasoning": [],
        "inquiry": [],
        "model_fabric": {"models": {}, "replacements": [], "routes": []},
        "body_and_tools": {"body": {}, "tools": {}, "failures": []},
        "learning": {"pending": {}, "admitted": {}, "rollback": {}, "rejected": []},
        "unfinished_tasks": [],
        "receipts": [],
    }


class EventSourcedKernel:
    """A model-independent state owner with deterministic event replay."""

    schema = "substrate-final-revision-kernel/v1"

    def __init__(self, identity: str):
        if not identity:
            raise io.Refused("identity is required")
        self._identity = identity
        self._events: list[CognitiveEvent] = []
        self._state = _initial_state(identity)

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def events(self) -> tuple[CognitiveEvent, ...]:
        return tuple(self._events)

    @property
    def state(self) -> dict[str, Any]:
        return _copy(self._state)

    def interfaces(self) -> tuple[str, ...]:
        return C.CONTRACTS

    def append(self, kind: str, payload: Mapping[str, Any], *, provenance: str) -> CognitiveEvent:
        if kind not in EVENT_KINDS:
            raise io.Refused(f"unknown cognitive event {kind!r}")
        if not provenance:
            raise io.Refused("event provenance is required")
        sequence = len(self._events) + 1
        logical_time = int(self._state["time"]["logical"]) + 1
        previous = self._events[-1].digest if self._events else "0" * 64
        unsigned = {
            "sequence": sequence,
            "logical_time": logical_time,
            "kind": kind,
            "payload": _copy(dict(payload)),
            "provenance": provenance,
            "previous_digest": previous,
        }
        event = CognitiveEvent(**unsigned, digest=io.digest(unsigned))
        proposed = _copy(self._state)
        self._apply(proposed, event)
        proposed["time"] = {"logical": logical_time, "last_event_sequence": sequence}
        proposed["receipts"].append(
            {
                "sequence": sequence,
                "event_digest": event.digest,
                "kind": kind,
                "provenance": provenance,
            }
        )
        self._events.append(event)
        self._state = proposed
        return event

    def _apply(self, state: dict[str, Any], event: CognitiveEvent) -> None:
        payload = event.payload
        kind = event.kind
        if kind == "observation":
            required = {"modality", "content_digest", "features"}
            if not required <= set(payload):
                raise io.Refused(f"observation misses {sorted(required - set(payload))}")
            state["observations"].append(_copy(payload))
            state["memory"]["perceptual"][str(payload["content_digest"])] = _copy(payload["features"])
            return
        if kind == "memory":
            memory_type = str(payload.get("memory_type"))
            if memory_type not in MEMORY_TYPES:
                raise io.Refused(f"unknown memory type {memory_type!r}")
            state["memory"][memory_type][str(payload["key"])] = _copy(payload["value"])
            return
        if kind == "belief":
            key = str(payload["key"])
            confidence = float(payload["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise io.Refused("belief confidence must be in [0,1]")
            state["beliefs"][key] = {
                "value": _copy(payload["value"]),
                "confidence": confidence,
                "warrants": list(payload.get("warrants", [])),
                "defeated": bool(payload.get("defeated", False)),
                "event": event.digest,
            }
            return
        if kind == "knowledge":
            key = str(payload["key"])
            belief = state["beliefs"].get(key)
            if belief is None or belief["confidence"] < 0.8 or belief["defeated"]:
                raise io.Refused("knowledge admission requires an undefeated belief with confidence >= 0.8")
            if belief["value"] != payload["value"]:
                raise io.Refused("knowledge value disagrees with admitted belief")
            state["knowledge"][key] = {"value": _copy(payload["value"]), "belief_event": belief["event"]}
            return
        if kind == "goal":
            identity = str(payload["goal_id"])
            action = str(payload.get("action", "create"))
            if action == "create":
                state["goals"][identity] = {
                    "description": str(payload["description"]),
                    "status": "active",
                    "priority": float(payload.get("priority", 0.5)),
                    "constraints": list(payload.get("constraints", [])),
                    "created_event": event.digest,
                }
                if identity not in state["unfinished_tasks"]:
                    state["unfinished_tasks"].append(identity)
            elif action in {"complete", "cancel", "pause", "resume"}:
                if identity not in state["goals"]:
                    raise io.Refused(f"unknown goal {identity!r}")
                state["goals"][identity]["status"] = {"resume": "active"}.get(action, action)
                if action in {"complete", "cancel"} and identity in state["unfinished_tasks"]:
                    state["unfinished_tasks"].remove(identity)
                elif action in {"pause", "resume"} and identity not in state["unfinished_tasks"]:
                    state["unfinished_tasks"].append(identity)
            else:
                raise io.Refused(f"unknown goal action {action!r}")
            return
        if kind == "world":
            operation = str(payload["operation"])
            if operation == "object":
                state["world_model"]["objects"][str(payload["key"])] = _copy(payload["value"])
            elif operation == "relation":
                state["world_model"]["relations"].append(_copy(payload["value"]))
            elif operation == "causal_edge":
                edge = _copy(payload["value"])
                if not {"cause", "effect"} <= set(edge):
                    raise io.Refused("causal edge requires cause and effect")
                state["world_model"]["causal_edges"].append(edge)
            elif operation == "counterfactual":
                value = _copy(payload["value"])
                changed = value.get("changed")
                held_fixed = value.get("held_fixed")
                causal_rule = value.get("causal_rule")
                if not isinstance(changed, Mapping) or not isinstance(held_fixed, Mapping) or not isinstance(
                    causal_rule, Mapping
                ):
                    raise io.Refused("counterfactual must declare mapping-valued changed, held_fixed, and causal_rule")
                if not changed or set(changed) & set(held_fixed):
                    raise io.Refused("counterfactual changed and held-fixed variables must be nonempty and disjoint")
                prerequisites = causal_rule.get("door_opens_if")
                if prerequisites != ["push", "hinge_intact"]:
                    raise io.Refused("counterfactual causal rule is unknown to the bounded world model")
                declared = set(changed) | {f"{key}_intact" if key == "hinge" else key for key in held_fixed}
                if declared != set(prerequisites):
                    raise io.Refused("counterfactual changes an undeclared variable or omits a causal prerequisite")
                causal_causes = {str(edge["cause"]) for edge in state["world_model"]["causal_edges"]}
                if not set(changed) <= causal_causes:
                    raise io.Refused("counterfactual changed variable has no stored causal edge")
                state["world_model"]["counterfactuals"].append(value)
            elif operation == "intervene":
                value = _copy(payload["value"])
                if not {"branch_id", "variable", "value", "held_fixed"} <= set(value):
                    raise io.Refused("intervention requires branch_id, variable, value, and held_fixed")
                state["world_model"]["interventions"].append(value)
            else:
                raise io.Refused(f"unknown world operation {operation!r}")
            return
        if kind == "self":
            operation = str(payload["operation"])
            if operation == "competence":
                state["self_model"]["competence"][str(payload["key"])] = float(payload["value"])
            elif operation == "limit":
                state["self_model"]["limits"].append(str(payload["value"]))
            elif operation == "resource":
                state["self_model"]["resource_state"][str(payload["key"])] = _copy(payload["value"])
            elif operation == "prediction":
                state["self_model"]["predictions"].append(_copy(payload["value"]))
            else:
                raise io.Refused(f"unknown self operation {operation!r}")
            return
        if kind == "reasoning":
            if str(payload.get("method")) not in {"deductive", "inductive", "abductive", "causal", "counterfactual", "planning"}:
                raise io.Refused("reasoning method is not in the declared portfolio")
            state["reasoning"].append(_copy(payload))
            return
        if kind == "inquiry":
            state["inquiry"].append(_copy(payload))
            return
        if kind == "model_register":
            identity = str(payload["identity"])
            if identity in state["model_fabric"]["models"]:
                raise io.Refused(f"model {identity!r} already exists")
            state["model_fabric"]["models"][identity] = _copy(payload)
            return
        if kind == "model_remove":
            identity = str(payload["identity"])
            if identity not in state["model_fabric"]["models"]:
                raise io.Refused(f"model {identity!r} is not registered")
            removed = state["model_fabric"]["models"].pop(identity)
            state["model_fabric"]["replacements"].append({"removed": identity, "replacement": None, "prior": removed})
            return
        if kind == "model_replace":
            previous = str(payload["previous"])
            replacement = str(payload["replacement"])
            if previous not in state["model_fabric"]["models"]:
                raise io.Refused(f"model {previous!r} is not registered")
            old = state["model_fabric"]["models"].pop(previous)
            new = _copy(payload.get("contract", {}))
            new["identity"] = replacement
            state["model_fabric"]["models"][replacement] = new
            state["model_fabric"]["replacements"].append({"removed": previous, "replacement": replacement, "prior": old, "event": event.digest})
            return
        if kind == "body_tool":
            target = str(payload["target"])
            operation = str(payload.get("operation", "update"))
            if target not in {"body", "tools", "failures"}:
                raise io.Refused(f"unknown body/tool target {target!r}")
            if target == "failures":
                state["body_and_tools"]["failures"].append(_copy(payload["value"]))
            elif operation == "remove":
                state["body_and_tools"][target].pop(str(payload["key"]), None)
            else:
                state["body_and_tools"][target][str(payload["key"])] = _copy(payload["value"])
            return
        if kind == "learning_propose":
            update_id = str(payload["update_id"])
            if update_id in state["learning"]["pending"] or update_id in state["learning"]["admitted"]:
                raise io.Refused(f"duplicate learning update {update_id!r}")
            if str(payload.get("data_split")) != "construction":
                raise io.Refused("learning proposals may use construction data only")
            state["learning"]["pending"][update_id] = _copy(payload)
            return
        if kind == "learning_admit":
            update_id = str(payload["update_id"])
            evaluation = payload.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise io.Refused("learning admission requires a raw evaluator receipt")
            recomputed = learning_evaluation_receipt(
                update_id,
                held_out_before=evaluation.get("held_out_before", []),
                held_out_after=evaluation.get("held_out_after", []),
                retention_before=evaluation.get("retention_before", []),
                retention_after=evaluation.get("retention_after", []),
                evaluator_id=str(evaluation.get("evaluator_id", "")),
            )
            if dict(evaluation) != recomputed:
                raise io.Refused("learning evaluation receipt is corrupt or summary-injected")
            update = state["learning"]["pending"].pop(update_id, None)
            if update is None:
                raise io.Refused(f"unknown pending update {update_id!r}")
            computed = recomputed["computed"]
            held_out_gain = float(computed["held_out_after"]) - float(computed["held_out_before"])
            retention_loss = float(computed["retention_before"]) - float(computed["retention_after"])
            if held_out_gain <= 0.0 or retention_loss > 0.01:
                state["learning"]["rejected"].append(
                    {
                        "update_id": update_id,
                        "held_out_gain": held_out_gain,
                        "retention_loss": retention_loss,
                        "evaluation_digest": recomputed["evaluation_digest"],
                    }
                )
                return
            namespace = str(update["namespace"])
            key = str(update["key"])
            if namespace != "semantic":
                raise io.Refused("bounded learning admission currently permits semantic updates only")
            previous = _copy(state["memory"]["semantic"].get(key))
            state["memory"]["semantic"][key] = _copy(update["value"])
            state["learning"]["rollback"][update_id] = {"namespace": namespace, "key": key, "previous": previous}
            state["learning"]["admitted"][update_id] = {
                **update,
                "held_out_gain": held_out_gain,
                "retention_loss": retention_loss,
                "evaluation_digest": recomputed["evaluation_digest"],
                "evaluator_id": recomputed["evaluator_id"],
                "admission_event": event.digest,
            }
            return
        if kind == "learning_rollback":
            update_id = str(payload["update_id"])
            rollback = state["learning"]["rollback"].pop(update_id, None)
            if rollback is None:
                raise io.Refused(f"no rollback exists for {update_id!r}")
            key = rollback["key"]
            if rollback["previous"] is None:
                state["memory"]["semantic"].pop(key, None)
            else:
                state["memory"]["semantic"][key] = rollback["previous"]
            state["learning"]["admitted"].pop(update_id, None)
            return
        if kind == "correction":
            key = str(payload["belief_key"])
            if key in state["beliefs"]:
                state["beliefs"][key]["defeated"] = True
            state["knowledge"].pop(key, None)
            return
        if kind == "interrupt":
            return
        raise io.Refused(f"unhandled cognitive event {kind!r}")

    def query(self, contract: str, key: str | None = None) -> Any:
        if contract not in C.CONTRACTS:
            raise io.Refused(f"unknown contract {contract!r}")
        if contract == "checkpoint":
            return self.checkpoint()
        if contract == "receipts":
            return _copy(self._state["receipts"])
        value = self._state["body_and_tools"] if contract == "body_and_tools" else self._state[contract]
        if key is None:
            return _copy(value)
        if not isinstance(value, dict):
            raise io.Refused(f"contract {contract!r} is not key addressable")
        return _copy(value.get(key))

    def state_integrity_digest(self) -> str:
        return io.digest({"schema": self.schema, "semantic_state": self._state})

    def checkpoint(self) -> dict[str, Any]:
        state = _copy(self._state)
        document = {
            "schema": self.schema,
            "identity": self._identity,
            "entity_identity": _copy(state["identity"]),
            "state_integrity_digest": io.digest({"schema": self.schema, "semantic_state": state}),
            "semantic_state_digest": io.digest(state),
            "event_chain_head": self._events[-1].digest if self._events else "0" * 64,
            "events": [event.document() for event in self._events],
            "state": state,
            "activation": False,
        }
        document["checkpoint_digest"] = io.digest(document)
        return document

    @classmethod
    def restore(cls, checkpoint: Mapping[str, Any]) -> EventSourcedKernel:
        document = _copy(dict(checkpoint))
        claimed = document.pop("checkpoint_digest", None)
        if claimed != io.digest(document):
            raise io.Refused("checkpoint digest mismatch")
        if document.get("activation") is not False:
            raise io.Refused("checkpoint activation must be false")
        kernel = cls(str(document["identity"]))
        for raw in document["events"]:
            event_document = dict(raw)
            claimed_event = event_document.pop("digest")
            if io.digest(event_document) != claimed_event:
                raise io.Refused("event digest mismatch")
            event = kernel.append(
                str(event_document["kind"]),
                dict(event_document["payload"]),
                provenance=str(event_document["provenance"]),
            )
            if event.digest != claimed_event:
                raise io.Refused("event replay digest mismatch")
        if kernel.state != document["state"]:
            raise io.Refused("materialized state disagrees with replay")
        if kernel.state["identity"] != document["entity_identity"]:
            raise io.Refused("stable entity identity disagrees with checkpoint state")
        if kernel.state_integrity_digest() != document["state_integrity_digest"]:
            raise io.Refused("state integrity digest disagrees with checkpoint state")
        if io.digest(kernel.state) != document["semantic_state_digest"]:
            raise io.Refused("semantic state digest mismatch")
        return kernel


class ArchitecturePrototype:
    """One external contract with representation-specific activity receipts."""

    def __init__(self, candidate_id: str, identity: str):
        if candidate_id not in C.CANDIDATES:
            raise io.Refused(f"unknown candidate {candidate_id!r}")
        self.candidate_id = candidate_id
        self.kernel = EventSourcedKernel(identity)
        self.activity: dict[str, Any] = {
            "events": 0,
            "representation": C.CANDIDATES[candidate_id]["representation"],
        }
        self._latent = [0.0, 0.0, 0.0, 0.0]
        self._workspace: list[str] = []
        self._graph_nodes: set[str] = set()
        self._graph_edges: list[tuple[str, str]] = []
        self._last_kind: str | None = None
        self._prediction_errors = 0
        self._causal_index: list[dict[str, Any]] = []
        self._branch_store: dict[str, dict[str, Any]] = {}

    def interfaces(self) -> tuple[str, ...]:
        return self.kernel.interfaces()

    def append(self, kind: str, payload: Mapping[str, Any], *, provenance: str) -> CognitiveEvent:
        event = self.kernel.append(kind, payload, provenance=provenance)
        self.activity["events"] += 1
        self._activate(event)
        return event

    def _activate(self, event: CognitiveEvent) -> None:
        candidate = self.candidate_id
        if candidate == "A_frozen_v5_hybrid":
            touched = set(self.activity.get("specialist_domains", []))
            touched.add(event.kind)
            self.activity["specialist_domains"] = sorted(touched)
        elif candidate == "B_s2_task_independent_monolith":
            self.activity["single_transition_function_calls"] = self.activity["events"]
        elif candidate == "C_event_sourced":
            self.activity["event_chain_head"] = event.digest
        elif candidate == "D_recurrent_state_space":
            signed = int(event.digest[:8], 16) / 0xFFFFFFFF
            self._latent = [round(0.75 * value + 0.25 * (signed - 0.5) * (index + 1), 8) for index, value in enumerate(self._latent)]
            self.activity["latent_state"] = list(self._latent)
        elif candidate == "E_graph_dynamical":
            if event.kind == "world":
                key = str(event.payload.get("key", event.digest[:12]))
                self._graph_nodes.add(key)
                value = event.payload.get("value")
                if isinstance(value, dict) and {"cause", "effect"} <= set(value):
                    cause = str(value["cause"])
                    effect = str(value["effect"])
                    self._graph_nodes.update((cause, effect))
                    self._graph_edges.append((cause, effect))
            self.activity["graph_nodes"] = len(self._graph_nodes)
            self.activity["graph_edges"] = len(self._graph_edges)
        elif candidate == "F_global_workspace":
            if event.kind in {"goal", "belief", "inquiry", "correction", "interrupt"}:
                self._workspace.append(event.digest)
                self._workspace = self._workspace[-3:]
            self.activity["workspace_broadcasts"] = len(self._workspace)
            self.activity["workspace_capacity"] = 3
        elif candidate == "G_predictive_world_model":
            predicted = self._last_kind
            self._prediction_errors += int(predicted is not None and predicted != event.kind)
            self._last_kind = event.kind
            self.activity["transition_predictions"] = max(0, self.activity["events"] - 1)
            self.activity["prediction_errors"] = self._prediction_errors
        elif candidate == "H_causal_temporal_ledger":
            if event.kind == "world" and event.payload.get("operation") == "causal_edge":
                edge = _copy(event.payload["value"])
                edge["event_digest"] = event.digest
                self._causal_index.append(edge)
            elif event.kind == "world" and event.payload.get("operation") in {"intervene", "counterfactual"}:
                raw = _copy(event.payload["value"])
                if event.payload["operation"] == "intervene":
                    branch_id = str(raw["branch_id"])
                    interventions = {str(raw["variable"]): raw["value"]}
                else:
                    branch_id = f"counterfactual:{event.digest[:16]}"
                    interventions = {str(key): value for key, value in dict(raw["changed"]).items()}
                projected_delta: dict[str, Any] = {}
                for edge in self._causal_index:
                    cause = str(edge["cause"])
                    if cause not in interventions:
                        continue
                    mapping = edge.get("mapping")
                    if not isinstance(mapping, Mapping):
                        continue
                    lookup = json.dumps(interventions[cause], sort_keys=True).lower()
                    if lookup in mapping:
                        projected_delta[str(edge["effect"])] = _copy(mapping[lookup])
                self._branch_store[branch_id] = {
                    "base_event_id": event.previous_digest,
                    "interventions": interventions,
                    "held_fixed": _copy(raw["held_fixed"]),
                    "projected_delta": projected_delta,
                    "actual_writeback": False,
                    "branch_digest": io.digest(
                        {
                            "branch_id": branch_id,
                            "base_event_id": event.previous_digest,
                            "interventions": interventions,
                            "held_fixed": raw["held_fixed"],
                            "projected_delta": projected_delta,
                        }
                    ),
                }
            self.activity["causal_edges_indexed"] = len(self._causal_index)
            self.activity["active_branch_count"] = len(self._branch_store)
            self.activity["intervention_index_head"] = io.digest(
                {"causal_index": self._causal_index, "branch_store": self._branch_store}
            )
        elif candidate == "I_simplest_sufficient":
            self.activity["projection_calls"] = self.activity["events"]
            self.activity["event_chain_head"] = event.digest

    def query(self, contract: str, key: str | None = None) -> Any:
        return self.kernel.query(contract, key)

    def _graph_reachable(self, start: str, target: str) -> bool:
        frontier = [start]
        visited: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            frontier.extend(effect for cause, effect in self._graph_edges if cause == node)
        return False

    def mechanism_decision(self) -> dict[str, Any]:
        required_activity = {
            "A_frozen_v5_hybrid": "specialist_domains",
            "B_s2_task_independent_monolith": "single_transition_function_calls",
            "C_event_sourced": "event_chain_head",
            "D_recurrent_state_space": "latent_state",
            "E_graph_dynamical": "graph_nodes",
            "F_global_workspace": "workspace_broadcasts",
            "G_predictive_world_model": "transition_predictions",
            "H_causal_temporal_ledger": "intervention_index_head",
            "I_simplest_sufficient": "projection_calls",
        }
        required = required_activity[self.candidate_id]
        if required not in self.activity:
            raise io.Refused(f"{self.candidate_id} mechanism is inactive")
        active_value = _copy(self.activity[required])
        goals = self.kernel.query("goals")
        active_goals = sorted(key for key, value in goals.items() if value["status"] == "active")
        probe: dict[str, Any] = {"activity_nonempty": bool(active_value)}
        if self.candidate_id == "E_graph_dynamical":
            probe = {
                "query": ["push", "door-angle"],
                "reachable": self._graph_reachable("push", "door-angle"),
                "edge_count": len(self._graph_edges),
            }
        elif self.candidate_id == "D_recurrent_state_space":
            probe = {"latent_l1": round(sum(abs(value) for value in self._latent), 8)}
        elif self.candidate_id == "F_global_workspace":
            probe = {"broadcast_head": self._workspace[-1] if self._workspace else None}
        elif self.candidate_id == "G_predictive_world_model":
            probe = {"prediction_error_rate": self._prediction_errors / max(1, self.activity["events"] - 1)}
        elif self.candidate_id == "H_causal_temporal_ledger":
            if not self._branch_store:
                raise io.Refused("H intervention-indexed mechanism has no derived branch")
            branch_id = sorted(self._branch_store)[-1]
            branch = self._branch_store[branch_id]
            probe = {
                "branch_id": branch_id,
                "projected_delta": _copy(branch["projected_delta"]),
                "actual_writeback": branch["actual_writeback"],
                "causal_path_length": int(bool(branch["projected_delta"])),
            }
        return {
            "candidate_id": self.candidate_id,
            "mechanism_field": required,
            "mechanism_value": active_value,
            "mechanism_probe": probe,
            "mechanism_token": io.digest(
                {
                    "candidate_id": self.candidate_id,
                    "required_activity": required,
                    "active_value": active_value,
                    "active_goals": active_goals,
                }
            ),
            "decision": {"continue_goal": active_goals[0] if active_goals else None},
            "activation": False,
        }

    def representation_state(self) -> dict[str, Any]:
        """Return the complete candidate-specific state needed for exact continuation."""
        return {
            "activity": _copy(self.activity),
            "latent": list(self._latent),
            "workspace": list(self._workspace),
            "graph_nodes": sorted(self._graph_nodes),
            "graph_edges": [list(edge) for edge in self._graph_edges],
            "last_kind": self._last_kind,
            "prediction_errors": self._prediction_errors,
            "causal_index": _copy(self._causal_index),
            "branch_store": _copy(self._branch_store),
            "activation": False,
        }

    def checkpoint(self) -> dict[str, Any]:
        checkpoint = self.kernel.checkpoint()
        checkpoint["candidate_id"] = self.candidate_id
        checkpoint["mechanism_activity"] = _copy(self.activity)
        checkpoint["mechanism_internal_state"] = self.representation_state()
        checkpoint["checkpoint_digest"] = io.digest({key: value for key, value in checkpoint.items() if key != "checkpoint_digest"})
        return checkpoint

    @classmethod
    def restore(cls, checkpoint: Mapping[str, Any]) -> ArchitecturePrototype:
        candidate_id = str(checkpoint.get("candidate_id", ""))
        if candidate_id not in C.CANDIDATES:
            raise io.Refused("prototype checkpoint candidate is absent or unknown")
        kernel = EventSourcedKernel.restore(checkpoint)
        prototype = cls(candidate_id, str(checkpoint["identity"]))
        prototype.kernel = kernel
        state = checkpoint.get("mechanism_internal_state")
        activity = checkpoint.get("mechanism_activity")
        if not isinstance(state, Mapping) or not isinstance(activity, Mapping):
            raise io.Refused("prototype checkpoint mechanism state is absent")
        if state.get("activation") is not False or dict(state.get("activity", {})) != dict(activity):
            raise io.Refused("prototype checkpoint mechanism state disagrees with activity receipt")
        prototype.activity = _copy(dict(activity))
        prototype._latent = [float(value) for value in state.get("latent", [])]
        prototype._workspace = [str(value) for value in state.get("workspace", [])]
        prototype._graph_nodes = {str(value) for value in state.get("graph_nodes", [])}
        graph_edges: list[tuple[str, str]] = []
        for edge in state.get("graph_edges", []):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise io.Refused("prototype graph edge is malformed")
            graph_edges.append((str(edge[0]), str(edge[1])))
        prototype._graph_edges = graph_edges
        prototype._last_kind = str(state["last_kind"]) if state.get("last_kind") is not None else None
        prototype._prediction_errors = int(state.get("prediction_errors", 0))
        prototype._causal_index = _copy(state.get("causal_index", []))
        prototype._branch_store = _copy(state.get("branch_store", {}))
        if prototype.representation_state() != dict(state):
            raise io.Refused("prototype checkpoint representation state did not restore exactly")
        return prototype


def developmental_fixture(kernel: ArchitecturePrototype) -> dict[str, Any]:
    """Exercise every external contract without hidden answers."""

    kernel.append(
        "goal",
        {"goal_id": "old-project", "description": "finish the retained project", "constraints": ["activation=false"]},
        provenance="fixture://operator/goal",
    )
    kernel.append(
        "memory",
        {"memory_type": "developmental", "key": "lesson", "value": "prefer verified evidence"},
        provenance="fixture://teaching/lesson",
    )
    kernel.append(
        "belief",
        {"key": "door-open", "value": True, "confidence": 0.9, "warrants": ["sensor-a"]},
        provenance="fixture://sensor-a/belief",
    )
    kernel.append("knowledge", {"key": "door-open", "value": True}, provenance="fixture://epistemic/admission")
    kernel.append(
        "world",
        {"operation": "object", "key": "door", "value": {"position": [1.0, 0.0, 0.0], "open": True}},
        provenance="fixture://scene/door",
    )
    kernel.append(
        "world",
        {
            "operation": "causal_edge",
            "value": {"cause": "push", "effect": "door-angle", "mapping": {"false": 0, "true": 1}},
        },
        provenance="fixture://causal/door",
    )
    kernel.append(
        "world",
        {
            "operation": "counterfactual",
            "value": {
                "changed": {"push": False},
                "held_fixed": {"hinge": "intact"},
                "causal_rule": {"door_opens_if": ["push", "hinge_intact"]},
            },
        },
        provenance="fixture://counterfactual/door",
    )
    kernel.append(
        "world",
        {
            "operation": "intervene",
            "value": {
                "branch_id": "door-no-push",
                "variable": "push",
                "value": False,
                "held_fixed": {"hinge": "intact"},
            },
        },
        provenance="fixture://intervention/door",
    )
    kernel.append(
        "self",
        {"operation": "competence", "key": "vision", "value": 0.72},
        provenance="fixture://self/prediction",
    )
    kernel.append(
        "reasoning",
        {"method": "causal", "selected_before_outcome": True, "inputs": ["door-open"], "conclusion": "push-door"},
        provenance="fixture://reasoning/causal",
    )
    kernel.append(
        "inquiry",
        {"question": "is the path clear?", "status": "unresolved", "uncertainty": 0.4},
        provenance="fixture://inquiry/path",
    )
    kernel.append(
        "model_register",
        {"identity": "model-a", "version": "1", "roles": ["draft"], "competence": 0.65, "cost": 1.0},
        provenance="fixture://models/a",
    )
    kernel.append(
        "model_replace",
        {
            "previous": "model-a",
            "replacement": "model-b",
            "contract": {"version": "2", "roles": ["draft"], "competence": 0.7, "cost": 1.0},
        },
        provenance="fixture://models/replacement",
    )
    kernel.append(
        "body_tool",
        {"target": "body", "key": "reach", "value": 1.2},
        provenance="fixture://body/reach",
    )
    kernel.append(
        "body_tool",
        {"target": "tools", "key": "gripper", "value": {"available": True, "latency_ms": 4}},
        provenance="fixture://tools/gripper",
    )
    kernel.append(
        "learning_propose",
        {
            "update_id": "lesson-update",
            "namespace": "semantic",
            "key": "verified-lesson",
            "value": "check warrants",
            "data_split": "construction",
            "source": "human-teaching",
        },
        provenance="fixture://learning/proposal",
    )
    kernel.append(
        "learning_admit",
        {
            "update_id": "lesson-update",
            "evaluation": learning_evaluation_receipt(
                "lesson-update",
                held_out_before=[True, False, True, False],
                held_out_after=[True, True, True, False],
                retention_before=[True, True, True, True],
                retention_after=[True, True, True, True],
            ),
        },
        provenance="fixture://learning/admission",
    )
    return {
        "candidate": kernel.candidate_id,
        "interfaces": list(kernel.interfaces()),
        "state_integrity_digest": kernel.kernel.state_integrity_digest(),
        "event_chain_head": kernel.kernel.events[-1].digest,
        "unfinished_goals": kernel.query("goals"),
        "mechanism_activity": _copy(kernel.activity),
        "mechanism_decision": kernel.mechanism_decision(),
        "checkpoint": kernel.checkpoint(),
        "activation": False,
    }
