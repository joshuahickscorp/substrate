"""Counterfactual-only composition of finite G0 actor genotypes.

The G0 reference evaluator executes one actor genotype.  This module adds the
smallest mechanics layer needed to compose those isolated evaluations: explicit
single-root input ports, a bounded next-round FIFO, staged actor-local state, and
an exactly replayable trace.  It intentionally lives outside :mod:`mop.escs` and
has no path to factual effects, live scheduling, topology installation, or
scientific promotion.

This is not an optimizer and it does not decide whether a constructed coalition
is useful.  It only makes multi-actor counterfactual behavior finite and
inspectable so a future, separately sealed study can evaluate candidates without
silently inventing message, state, or accounting semantics.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self, cast

from mop.escs.accounting import WorkVector
from mop.escs.g0_evaluator import (
    G0CounterfactualEvaluation,
    G0CounterfactualRefusal,
    attempt_g0_counterfactual,
    verify_g0_counterfactual,
)
from mop.escs.g0_genotype import G0ActorGenotype
from mop.escs.perspective_registry import PerspectiveCandidateRegistry
from mop.escs.topology_grammar import TopologyGrammar
from mop.substrate.events import FrozenJSON, canonical_bytes, canonical_sha256

from .escs_g0_construction import G0ConstructionSnapshot

G0_SHADOW_EPISODE_SCHEMA = "mop-escs-g0-shadow-episode/v1"
G0_SHADOW_DELIVERY_SCHEMA = "mop-escs-g0-shadow-delivery/v1"
G0_SHADOW_ACTIVATION_SCHEMA = "mop-escs-g0-shadow-activation/v1"
G0_SHADOW_TRACE_SCHEMA = "mop-escs-g0-shadow-trace/v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_FORMS = frozenset({"json-list", "json-object", "json-value", "numeric-scalar", "numeric-vector"})
_EXECUTOR_CONTRACT = {
    "schema": "mop-escs-g0-shadow-executor-contract/v1",
    "queue_semantics": "canonical-fifo-one-delivery-per-activation",
    "delivery_semantics": "next-round-only",
    "input_port_semantics": "one-explicit-root-per-actor",
    "state_semantics": "immutable-start-snapshot-atomic-next-activation-staging",
    "failure_accounting": "full-declared-actor-envelope",
    "counterfactual_only": True,
    "activation_enabled": False,
    "shadow_execution_authorized": False,
    "factual_effects": False,
    "factual_mutation_authorized": False,
    "scientific_promotion_allowed": False,
}
G0_SHADOW_EXECUTOR_CONTRACT_SHA256 = canonical_sha256(_EXECUTOR_CONTRACT)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _validate_payload_form(value: Any, payload_form: str, label: str) -> None:
    _require(payload_form in _PAYLOAD_FORMS, f"{label} payload form is not declared")
    if payload_form == "numeric-scalar":
        _require(
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value)),
            f"{label} must be a finite numeric scalar",
        )
    elif payload_form == "numeric-vector":
        _require(isinstance(value, list) and bool(value), f"{label} must be a numeric vector")
        for item in value:
            _require(
                not isinstance(item, bool) and isinstance(item, int | float) and math.isfinite(float(item)),
                f"{label} contains a nonfinite or nonnumeric item",
            )
    elif payload_form == "json-list":
        _require(isinstance(value, list), f"{label} must be a JSON list")
    elif payload_form == "json-object":
        _require(isinstance(value, dict), f"{label} must be a JSON object")


def _result_sha256(result: G0CounterfactualEvaluation | G0CounterfactualRefusal) -> str:
    if isinstance(result, G0CounterfactualEvaluation):
        return result.evaluation_sha256
    return result.refusal_sha256


def _result_payload(
    result: G0CounterfactualEvaluation | G0CounterfactualRefusal,
) -> dict[str, Any]:
    return result.payload()


def _message_rows(result: G0CounterfactualEvaluation) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for frozen in result.staged_messages:
        value = frozen.value()
        _require(isinstance(value, dict), "G0 staged message is not an object")
        rows.append(cast(dict[str, Any], value))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class G0ShadowCaps:
    max_rounds: int
    max_activations: int
    max_queue_depth: int
    max_messages: int
    max_actor_operations: int
    max_routed_payload_bytes: int
    max_retained_state_bytes: int
    max_repeated_state_visits: int = 1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative_int(getattr(self, name), f"G0 shadow cap {name}", positive=True)

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class G0ShadowPortBinding:
    actor_id: str
    genotype_sha256: str
    root_node_id: str
    schema_id: str
    payload_form: str
    max_encoded_bytes: int

    def __post_init__(self) -> None:
        _require_id(self.actor_id, "G0 shadow port actor_id")
        _require_digest(self.genotype_sha256, "G0 shadow port genotype digest")
        _require_id(self.root_node_id, "G0 shadow port root_node_id")
        _require_id(self.schema_id, "G0 shadow port schema_id")
        _require(self.payload_form in _PAYLOAD_FORMS, "G0 shadow port payload form is not declared")
        _nonnegative_int(self.max_encoded_bytes, "G0 shadow port max_encoded_bytes", positive=True)

    def payload(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "genotype_sha256": self.genotype_sha256,
            "root_node_id": self.root_node_id,
            "schema_id": self.schema_id,
            "payload_form": self.payload_form,
            "max_encoded_bytes": self.max_encoded_bytes,
        }


@dataclass(frozen=True, slots=True)
class G0ShadowActorState:
    actor_id: str
    state: FrozenJSON

    def __post_init__(self) -> None:
        _require_id(self.actor_id, "G0 shadow state actor_id")
        _require(isinstance(self.state, FrozenJSON), "G0 shadow actor state must be frozen JSON")
        _require(isinstance(self.state.value(), dict), "G0 shadow actor state must contain an object")

    @classmethod
    def create(cls, actor_id: str, state: Mapping[str, Any]) -> Self:
        return cls(actor_id=actor_id, state=FrozenJSON.from_value(dict(state)))

    def payload(self) -> dict[str, Any]:
        return {"actor_id": self.actor_id, "state": self.state.payload()}


@dataclass(frozen=True, slots=True)
class G0ShadowSeed:
    seed_id: str
    actor_id: str
    schema_id: str
    payload_form: str
    payload_value: FrozenJSON
    seed_sha256: str

    def __post_init__(self) -> None:
        _require_id(self.seed_id, "G0 shadow seed_id")
        _require_id(self.actor_id, "G0 shadow seed actor_id")
        _require_id(self.schema_id, "G0 shadow seed schema_id")
        _require(self.payload_form in _PAYLOAD_FORMS, "G0 shadow seed payload form is not declared")
        _require(isinstance(self.payload_value, FrozenJSON), "G0 shadow seed payload must be frozen")
        _validate_payload_form(self.payload_value.value(), self.payload_form, "G0 shadow seed")
        _require_digest(self.seed_sha256, "G0 shadow seed digest")
        _require(
            self.seed_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 shadow seed self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        seed_id: str,
        actor_id: str,
        schema_id: str,
        payload_form: str,
        payload: Any,
    ) -> Self:
        frozen = FrozenJSON.from_value(payload)
        core = {
            "seed_id": seed_id,
            "actor_id": actor_id,
            "schema_id": schema_id,
            "payload_form": payload_form,
            "payload": frozen.payload(),
        }
        return cls(
            seed_id=seed_id,
            actor_id=actor_id,
            schema_id=schema_id,
            payload_form=payload_form,
            payload_value=frozen,
            seed_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seed_id": self.seed_id,
            "actor_id": self.actor_id,
            "schema_id": self.schema_id,
            "payload_form": self.payload_form,
            "payload": self.payload_value.payload(),
        }
        if include_digest:
            result["seed_sha256"] = self.seed_sha256
        return result


@dataclass(frozen=True, slots=True)
class G0ShadowEpisode:
    episode_id: str
    source_snapshot_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    executor_contract_sha256: str
    ports: tuple[G0ShadowPortBinding, ...]
    initial_states: tuple[G0ShadowActorState, ...]
    seeds: tuple[G0ShadowSeed, ...]
    caps: G0ShadowCaps
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_effects: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    episode_sha256: str
    schema: str = G0_SHADOW_EPISODE_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_SHADOW_EPISODE_SCHEMA, "unsupported G0 shadow episode schema")
        _require_id(self.episode_id, "G0 shadow episode_id")
        for value, label in (
            (self.source_snapshot_sha256, "source snapshot"),
            (self.grammar_sha256, "grammar"),
            (self.candidate_registry_sha256, "candidate registry"),
            (self.executor_contract_sha256, "executor contract"),
            (self.episode_sha256, "episode"),
        ):
            _require_digest(value, f"G0 shadow {label} digest")
        _require(
            self.executor_contract_sha256 == G0_SHADOW_EXECUTOR_CONTRACT_SHA256,
            "G0 shadow executor contract authority mismatch",
        )
        _require(isinstance(self.ports, tuple) and bool(self.ports), "G0 shadow ports are empty")
        _require(
            all(type(row) is G0ShadowPortBinding for row in self.ports),
            "G0 shadow ports must be exact bindings",
        )
        _require(
            all(type(row) is G0ShadowActorState for row in self.initial_states),
            "G0 shadow initial states must be exact records",
        )
        _require(
            isinstance(self.seeds, tuple)
            and bool(self.seeds)
            and all(type(row) is G0ShadowSeed for row in self.seeds),
            "G0 shadow seeds must be nonempty exact records",
        )
        port_ids = tuple(row.actor_id for row in self.ports)
        state_ids = tuple(row.actor_id for row in self.initial_states)
        _require(
            port_ids == tuple(sorted(set(port_ids))),
            "G0 shadow ports must be unique and canonically ordered",
        )
        _require(state_ids == port_ids, "G0 shadow state coverage must equal port coverage")
        seed_ids = tuple(row.seed_id for row in self.seeds)
        seed_actor_ids = tuple(row.actor_id for row in self.seeds)
        _require(
            seed_ids == tuple(sorted(set(seed_ids))),
            "G0 shadow seeds must be unique and canonically ordered",
        )
        _require(len(seed_actor_ids) == len(set(seed_actor_ids)), "an actor may have only one root seed")
        _require(set(seed_actor_ids) <= set(port_ids), "G0 shadow seed names an unknown actor")
        _require(type(self.caps) is G0ShadowCaps, "G0 shadow caps must be exact")
        _require(self.counterfactual_only is True, "G0 shadow episode escaped counterfactual status")
        for flag, flag_label in (
            (self.activation_enabled, "activation"),
            (self.shadow_execution_authorized, "shadow execution"),
            (self.factual_effects, "factual effects"),
            (self.factual_mutation_authorized, "factual mutation"),
            (self.scientific_promotion_allowed, "scientific promotion"),
        ):
            _require(flag is False, f"G0 shadow episode cannot authorize {flag_label}")
        _require(
            self.episode_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 shadow episode self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        source: G0ConstructionSnapshot,
        grammar: TopologyGrammar,
        candidate_registry: PerspectiveCandidateRegistry,
        ports: Sequence[G0ShadowPortBinding],
        initial_states: Sequence[G0ShadowActorState],
        seeds: Sequence[G0ShadowSeed],
        caps: G0ShadowCaps,
    ) -> Self:
        _require(type(source) is G0ConstructionSnapshot, "G0 shadow source must be an exact snapshot")
        _require(type(grammar) is TopologyGrammar, "G0 shadow grammar must be exact")
        _require(
            type(candidate_registry) is PerspectiveCandidateRegistry,
            "G0 shadow candidate registry must be exact",
        )
        _require(
            candidate_registry.sha256 == grammar.candidate_registry_sha256,
            "G0 shadow grammar and candidate registry authorities differ",
        )
        port_rows = tuple(sorted(ports, key=lambda row: row.actor_id))
        state_rows = tuple(sorted(initial_states, key=lambda row: row.actor_id))
        seed_rows = tuple(sorted(seeds, key=lambda row: row.seed_id))
        actors = {actor.candidate_id: actor for actor in source.actors}
        _require(
            {row.actor_id for row in port_rows} == set(actors),
            "G0 shadow ports must cover exactly the construction actors",
        )
        for port in port_rows:
            actor = actors[port.actor_id]
            roots = tuple(node.node_id for node in actor.operator_nodes if not node.input_node_ids)
            _require(len(roots) == 1, "G0 shadow v1 requires exactly one root node per actor")
            _require(port.root_node_id == roots[0], "G0 shadow port does not bind the actor root")
            _require(
                port.genotype_sha256 == actor.genotype_sha256,
                "G0 shadow port genotype authority mismatch",
            )
        states_by_actor = {row.actor_id: row.state for row in state_rows}
        _require(set(states_by_actor) == set(actors), "G0 shadow states must cover every actor")
        for actor_id, actor in actors.items():
            state_value = states_by_actor[actor_id].value()
            _require(isinstance(state_value, dict), "G0 shadow initial state must be an object")
            _require(
                set(state_value) == {slot.slot_id for slot in actor.state_slots},
                "G0 shadow initial state slots do not match the actor genotype",
            )
        by_port = {row.actor_id: row for row in port_rows}
        for seed in seed_rows:
            seed_port = by_port.get(seed.actor_id)
            if seed_port is None:
                raise ValueError("G0 shadow seed actor has no port")
            _require(
                seed.schema_id == seed_port.schema_id,
                "G0 shadow seed schema does not match its port",
            )
            _require(
                seed.payload_form == seed_port.payload_form,
                "G0 shadow seed payload form does not match its port",
            )
            _require(
                len(canonical_bytes(seed.payload_value.value())) <= seed_port.max_encoded_bytes,
                "G0 shadow seed exceeds its port byte cap",
            )
        _require(
            source.retained_state_bytes <= caps.max_retained_state_bytes,
            "G0 shadow construction exceeds the retained-state cap",
        )
        core = {
            "schema": G0_SHADOW_EPISODE_SCHEMA,
            "episode_id": episode_id,
            "source_snapshot_sha256": source.snapshot_sha256,
            "grammar_sha256": grammar.grammar_sha256,
            "candidate_registry_sha256": candidate_registry.sha256,
            "executor_contract_sha256": G0_SHADOW_EXECUTOR_CONTRACT_SHA256,
            "ports": [row.payload() for row in port_rows],
            "initial_states": [row.payload() for row in state_rows],
            "seeds": [row.payload() for row in seed_rows],
            "caps": caps.payload(),
            "counterfactual_only": True,
            "activation_enabled": False,
            "shadow_execution_authorized": False,
            "factual_effects": False,
            "factual_mutation_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            episode_id=episode_id,
            source_snapshot_sha256=source.snapshot_sha256,
            grammar_sha256=grammar.grammar_sha256,
            candidate_registry_sha256=candidate_registry.sha256,
            executor_contract_sha256=G0_SHADOW_EXECUTOR_CONTRACT_SHA256,
            ports=port_rows,
            initial_states=state_rows,
            seeds=seed_rows,
            caps=caps,
            counterfactual_only=True,
            activation_enabled=False,
            shadow_execution_authorized=False,
            factual_effects=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            episode_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "executor_contract_sha256": self.executor_contract_sha256,
            "ports": [row.payload() for row in self.ports],
            "initial_states": [row.payload() for row in self.initial_states],
            "seeds": [row.payload() for row in self.seeds],
            "caps": self.caps.payload(),
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_effects": self.factual_effects,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["episode_sha256"] = self.episode_sha256
        return result


class G0ShadowDeliveryKind(StrEnum):
    SEED = "seed"
    MESSAGE = "message"


@dataclass(frozen=True, slots=True)
class G0ShadowDelivery:
    sequence: int
    available_round: int
    kind: G0ShadowDeliveryKind
    source_sha256: str
    sender_actor_id: str | None
    recipient_actor_id: str
    schema_id: str
    payload_form: str
    payload_value: FrozenJSON
    counterfactual_only: bool
    activation_enabled: bool
    factual_effects: bool
    scientific_promotion_allowed: bool
    delivery_sha256: str
    schema: str = G0_SHADOW_DELIVERY_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_SHADOW_DELIVERY_SCHEMA, "unsupported G0 shadow delivery schema")
        _nonnegative_int(self.sequence, "G0 shadow delivery sequence")
        _nonnegative_int(self.available_round, "G0 shadow delivery round")
        _require(isinstance(self.kind, G0ShadowDeliveryKind), "G0 shadow delivery kind is untyped")
        _require_digest(self.source_sha256, "G0 shadow delivery source digest")
        if self.kind is G0ShadowDeliveryKind.SEED:
            _require(self.sender_actor_id is None, "G0 shadow seed delivery cannot name a sender")
        else:
            _require_id(self.sender_actor_id, "G0 shadow message sender")
        _require_id(self.recipient_actor_id, "G0 shadow delivery recipient")
        _require_id(self.schema_id, "G0 shadow delivery schema_id")
        _require(self.payload_form in _PAYLOAD_FORMS, "G0 shadow delivery payload form is undeclared")
        _require(isinstance(self.payload_value, FrozenJSON), "G0 shadow delivery payload is not frozen")
        _validate_payload_form(self.payload_value.value(), self.payload_form, "G0 shadow delivery")
        _require(self.counterfactual_only is True, "G0 shadow delivery escaped counterfactual status")
        for value, label in (
            (self.activation_enabled, "activation"),
            (self.factual_effects, "factual effects"),
            (self.scientific_promotion_allowed, "scientific promotion"),
        ):
            _require(value is False, f"G0 shadow delivery cannot authorize {label}")
        _require_digest(self.delivery_sha256, "G0 shadow delivery digest")
        _require(
            self.delivery_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 shadow delivery self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        available_round: int,
        kind: G0ShadowDeliveryKind,
        source_sha256: str,
        sender_actor_id: str | None,
        recipient_actor_id: str,
        schema_id: str,
        payload_form: str,
        payload: Any,
    ) -> Self:
        frozen = FrozenJSON.from_value(payload)
        core = {
            "schema": G0_SHADOW_DELIVERY_SCHEMA,
            "sequence": sequence,
            "available_round": available_round,
            "kind": kind.value,
            "source_sha256": source_sha256,
            "sender_actor_id": sender_actor_id,
            "recipient_actor_id": recipient_actor_id,
            "schema_id": schema_id,
            "payload_form": payload_form,
            "payload": frozen.payload(),
            "counterfactual_only": True,
            "activation_enabled": False,
            "factual_effects": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            sequence=sequence,
            available_round=available_round,
            kind=kind,
            source_sha256=source_sha256,
            sender_actor_id=sender_actor_id,
            recipient_actor_id=recipient_actor_id,
            schema_id=schema_id,
            payload_form=payload_form,
            payload_value=frozen,
            counterfactual_only=True,
            activation_enabled=False,
            factual_effects=False,
            scientific_promotion_allowed=False,
            delivery_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "sequence": self.sequence,
            "available_round": self.available_round,
            "kind": self.kind.value,
            "source_sha256": self.source_sha256,
            "sender_actor_id": self.sender_actor_id,
            "recipient_actor_id": self.recipient_actor_id,
            "schema_id": self.schema_id,
            "payload_form": self.payload_form,
            "payload": self.payload_value.payload(),
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "factual_effects": self.factual_effects,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["delivery_sha256"] = self.delivery_sha256
        return result


class G0ShadowActivationStatus(StrEnum):
    EVALUATED = "evaluated"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class G0ShadowActivation:
    sequence: int
    round_index: int
    actor_id: str
    genotype_sha256: str
    source_delivery_sha256: str
    prior_state_sha256: str
    status: G0ShadowActivationStatus
    result: G0CounterfactualEvaluation | G0CounterfactualRefusal
    state_applied: bool
    resulting_state: FrozenJSON
    emitted_message_sha256s: tuple[str, ...]
    enqueued_delivery_sha256s: tuple[str, ...]
    problems: tuple[str, ...]
    work: WorkVector
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_effects: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    activation_sha256: str
    schema: str = G0_SHADOW_ACTIVATION_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_SHADOW_ACTIVATION_SCHEMA, "unsupported G0 shadow activation schema")
        _nonnegative_int(self.sequence, "G0 shadow activation sequence")
        _nonnegative_int(self.round_index, "G0 shadow activation round")
        _require_id(self.actor_id, "G0 shadow activation actor_id")
        for value, label in (
            (self.genotype_sha256, "genotype"),
            (self.source_delivery_sha256, "source delivery"),
            (self.prior_state_sha256, "prior state"),
            (self.activation_sha256, "activation"),
        ):
            _require_digest(value, f"G0 shadow {label} digest")
        _require(isinstance(self.status, G0ShadowActivationStatus), "G0 shadow status is untyped")
        _require(
            type(self.result) in {G0CounterfactualEvaluation, G0CounterfactualRefusal},
            "G0 shadow activation result must be an exact evaluator receipt",
        )
        expected_status = (
            G0ShadowActivationStatus.EVALUATED
            if isinstance(self.result, G0CounterfactualEvaluation)
            else G0ShadowActivationStatus.REFUSED
        )
        _require(self.status is expected_status, "G0 shadow activation/result status mismatch")
        _require(self.result.genotype_sha256 == self.genotype_sha256, "G0 shadow result actor mismatch")
        _require(self.result.initial_state_sha256 == self.prior_state_sha256, "prior state mismatch")
        _require(isinstance(self.resulting_state, FrozenJSON), "resulting state is not frozen")
        _require(isinstance(self.state_applied, bool), "G0 shadow state_applied is not boolean")
        for rows, label in (
            (self.emitted_message_sha256s, "emitted messages"),
            (self.enqueued_delivery_sha256s, "enqueued deliveries"),
        ):
            _require(isinstance(rows, tuple), f"G0 shadow {label} must be immutable")
            _require(all(_DIGEST_RE.fullmatch(row) is not None for row in rows), f"bad {label} digest")
            _require(len(rows) == len(set(rows)), f"G0 shadow {label} contain duplicates")
        _require(
            isinstance(self.problems, tuple) and self.problems == tuple(sorted(set(self.problems))),
            "G0 shadow activation problems must be unique and sorted",
        )
        if self.state_applied:
            _require(
                isinstance(self.result, G0CounterfactualEvaluation),
                "only an evaluation may apply staged state",
            )
            evaluation = cast(G0CounterfactualEvaluation, self.result)
            _require(
                not self.problems,
                "only a clean evaluation may apply staged state",
            )
            _require(
                self.resulting_state.sha256 == evaluation.staged_state.sha256,
                "applied G0 shadow state differs from evaluator staging",
            )
        else:
            _require(
                self.resulting_state.sha256 == self.prior_state_sha256,
                "refused G0 shadow activation changed state",
            )
            _require(bool(self.problems), "nonapplying G0 shadow activation must explain refusal")
            _require(
                not self.enqueued_delivery_sha256s,
                "nonapplying G0 shadow activation cannot enqueue messages",
            )
        _require(self.work == self.result.work, "G0 shadow activation work mismatch")
        _require(self.counterfactual_only is True, "G0 shadow activation escaped counterfactual status")
        for flag, flag_label in (
            (self.activation_enabled, "activation"),
            (self.shadow_execution_authorized, "shadow execution"),
            (self.factual_effects, "factual effects"),
            (self.factual_mutation_authorized, "factual mutation"),
            (self.scientific_promotion_allowed, "scientific promotion"),
        ):
            _require(flag is False, f"G0 shadow activation cannot authorize {flag_label}")
        _require(
            self.activation_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 shadow activation self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        round_index: int,
        actor_id: str,
        genotype_sha256: str,
        source_delivery_sha256: str,
        prior_state: FrozenJSON,
        result: G0CounterfactualEvaluation | G0CounterfactualRefusal,
        state_applied: bool,
        resulting_state: FrozenJSON,
        emitted_message_sha256s: Sequence[str],
        enqueued_delivery_sha256s: Sequence[str],
        problems: Sequence[str],
    ) -> Self:
        status = (
            G0ShadowActivationStatus.EVALUATED
            if isinstance(result, G0CounterfactualEvaluation)
            else G0ShadowActivationStatus.REFUSED
        )
        emitted = tuple(emitted_message_sha256s)
        enqueued = tuple(enqueued_delivery_sha256s)
        problem_rows = tuple(sorted(set(problems)))
        core = {
            "schema": G0_SHADOW_ACTIVATION_SCHEMA,
            "sequence": sequence,
            "round_index": round_index,
            "actor_id": actor_id,
            "genotype_sha256": genotype_sha256,
            "source_delivery_sha256": source_delivery_sha256,
            "prior_state_sha256": prior_state.sha256,
            "status": status.value,
            "result": _result_payload(result),
            "state_applied": state_applied,
            "resulting_state": resulting_state.payload(),
            "emitted_message_sha256s": list(emitted),
            "enqueued_delivery_sha256s": list(enqueued),
            "problems": list(problem_rows),
            "work": result.work.payload(),
            "counterfactual_only": True,
            "activation_enabled": False,
            "shadow_execution_authorized": False,
            "factual_effects": False,
            "factual_mutation_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            sequence=sequence,
            round_index=round_index,
            actor_id=actor_id,
            genotype_sha256=genotype_sha256,
            source_delivery_sha256=source_delivery_sha256,
            prior_state_sha256=prior_state.sha256,
            status=status,
            result=result,
            state_applied=state_applied,
            resulting_state=resulting_state,
            emitted_message_sha256s=emitted,
            enqueued_delivery_sha256s=enqueued,
            problems=problem_rows,
            work=result.work,
            counterfactual_only=True,
            activation_enabled=False,
            shadow_execution_authorized=False,
            factual_effects=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            activation_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "sequence": self.sequence,
            "round_index": self.round_index,
            "actor_id": self.actor_id,
            "genotype_sha256": self.genotype_sha256,
            "source_delivery_sha256": self.source_delivery_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "status": self.status.value,
            "result": _result_payload(self.result),
            "state_applied": self.state_applied,
            "resulting_state": self.resulting_state.payload(),
            "emitted_message_sha256s": list(self.emitted_message_sha256s),
            "enqueued_delivery_sha256s": list(self.enqueued_delivery_sha256s),
            "problems": list(self.problems),
            "work": self.work.payload(),
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_effects": self.factual_effects,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["activation_sha256"] = self.activation_sha256
        return result


class G0ShadowTerminalReason(StrEnum):
    QUIESCENT = "quiescent"
    EVALUATION_REFUSED = "evaluation-refused"
    DELIVERY_REFUSED = "delivery-refused"
    REPEATED_STATE = "repeated-state"
    CAP_EXHAUSTED = "cap-exhausted"


@dataclass(frozen=True, slots=True)
class G0ShadowTrace:
    episode_sha256: str
    source_snapshot_sha256: str
    effective_snapshot_sha256: str
    rollback_snapshot_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    executor_contract_sha256: str
    deliveries: tuple[G0ShadowDelivery, ...]
    activations: tuple[G0ShadowActivation, ...]
    final_states: tuple[G0ShadowActorState, ...]
    terminal_reason: G0ShadowTerminalReason
    problems: tuple[str, ...]
    pending_delivery_count: int
    completed_rounds: int
    delivery_attempts: int
    produced_message_count: int
    routed_message_count: int
    routed_payload_bytes: int
    message_envelope_bytes: int
    declared_retained_state_bytes: int
    retained_state_byte_rounds: int
    work: WorkVector
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_effects: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    trace_sha256: str
    schema: str = G0_SHADOW_TRACE_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_SHADOW_TRACE_SCHEMA, "unsupported G0 shadow trace schema")
        for value, label in (
            (self.episode_sha256, "episode"),
            (self.source_snapshot_sha256, "source snapshot"),
            (self.effective_snapshot_sha256, "effective snapshot"),
            (self.rollback_snapshot_sha256, "rollback snapshot"),
            (self.grammar_sha256, "grammar"),
            (self.candidate_registry_sha256, "candidate registry"),
            (self.executor_contract_sha256, "executor contract"),
            (self.trace_sha256, "trace"),
        ):
            _require_digest(value, f"G0 shadow {label} digest")
        _require(
            self.source_snapshot_sha256 == self.effective_snapshot_sha256 == self.rollback_snapshot_sha256,
            "G0 shadow trace changed its effective or rollback construction snapshot",
        )
        _require(
            self.executor_contract_sha256 == G0_SHADOW_EXECUTOR_CONTRACT_SHA256,
            "G0 shadow trace executor authority mismatch",
        )
        _require(
            isinstance(self.deliveries, tuple)
            and all(type(row) is G0ShadowDelivery for row in self.deliveries),
            "G0 shadow deliveries must be exact immutable records",
        )
        _require(
            isinstance(self.activations, tuple)
            and all(type(row) is G0ShadowActivation for row in self.activations),
            "G0 shadow activations must be exact immutable records",
        )
        _require(
            tuple(row.sequence for row in self.deliveries) == tuple(range(len(self.deliveries))),
            "G0 shadow delivery sequence is not contiguous",
        )
        _require(
            tuple(row.sequence for row in self.activations) == tuple(range(len(self.activations))),
            "G0 shadow activation sequence is not contiguous",
        )
        delivery_by_sha = {row.delivery_sha256: row for row in self.deliveries}
        _require(len(delivery_by_sha) == len(self.deliveries), "G0 shadow delivery identity duplicated")
        _require(
            len({row.source_delivery_sha256 for row in self.activations}) == len(self.activations),
            "G0 shadow source delivery was activated twice",
        )
        states = {row.actor_id: row.state for row in self.final_states}
        _require(len(states) == len(self.final_states), "G0 shadow final state actor duplicated")
        for activation in self.activations:
            delivery = delivery_by_sha.get(activation.source_delivery_sha256)
            if delivery is None:
                raise ValueError("G0 shadow activation source delivery is absent")
            _require(delivery.recipient_actor_id == activation.actor_id, "delivery/actor mismatch")
        _require(
            isinstance(self.terminal_reason, G0ShadowTerminalReason),
            "G0 shadow terminal reason is untyped",
        )
        _require(
            isinstance(self.problems, tuple) and self.problems == tuple(sorted(set(self.problems))),
            "G0 shadow trace problems must be unique and sorted",
        )
        if self.terminal_reason is G0ShadowTerminalReason.QUIESCENT:
            _require(not self.problems and self.pending_delivery_count == 0, "bad quiescent trace")
        else:
            _require(bool(self.problems), "nonquiescent G0 shadow trace must explain termination")
        for count, count_label in (
            (self.pending_delivery_count, "pending_delivery_count"),
            (self.completed_rounds, "completed_rounds"),
            (self.delivery_attempts, "delivery_attempts"),
            (self.produced_message_count, "produced_message_count"),
            (self.routed_message_count, "routed_message_count"),
            (self.routed_payload_bytes, "routed_payload_bytes"),
            (self.message_envelope_bytes, "message_envelope_bytes"),
            (self.declared_retained_state_bytes, "declared_retained_state_bytes"),
            (self.retained_state_byte_rounds, "retained_state_byte_rounds"),
        ):
            _nonnegative_int(count, f"G0 shadow {count_label}")
        _require(
            self.pending_delivery_count == len(self.deliveries) - len(self.activations),
            "G0 shadow pending-delivery count mismatch",
        )
        expected_messages = sum(len(row.emitted_message_sha256s) for row in self.activations)
        _require(self.produced_message_count == expected_messages, "produced-message count mismatch")
        message_deliveries = [row for row in self.deliveries if row.kind is G0ShadowDeliveryKind.MESSAGE]
        _require(self.routed_message_count == len(message_deliveries), "routed-message count mismatch")
        _require(
            self.routed_payload_bytes
            == sum(len(canonical_bytes(row.payload_value.value())) for row in message_deliveries),
            "routed-message byte count mismatch",
        )
        _require(
            self.message_envelope_bytes
            == sum(
                row.result.message_envelope_bytes
                for row in self.activations
                if isinstance(row.result, G0CounterfactualEvaluation)
            ),
            "message-envelope byte count mismatch",
        )
        _require(
            self.retained_state_byte_rounds == self.declared_retained_state_bytes * self.completed_rounds,
            "retained-state byte-round count mismatch",
        )
        evaluation_work = sum((row.work for row in self.activations), WorkVector.zero())
        setup_and_dispatch = WorkVector(
            indexing_and_graph_maintenance=len(self.final_states) * 2,
            dispatch_and_exploration=self.delivery_attempts,
        )
        retention = WorkVector(retained_byte_time=self.retained_state_byte_rounds)
        _require(
            self.work == evaluation_work + setup_and_dispatch + retention,
            "G0 shadow aggregate work mismatch",
        )
        _require(self.counterfactual_only is True, "G0 shadow trace escaped counterfactual status")
        for flag, flag_label in (
            (self.activation_enabled, "activation"),
            (self.shadow_execution_authorized, "shadow execution"),
            (self.factual_effects, "factual effects"),
            (self.factual_mutation_authorized, "factual mutation"),
            (self.scientific_promotion_allowed, "scientific promotion"),
        ):
            _require(flag is False, f"G0 shadow trace cannot authorize {flag_label}")
        _require(
            self.trace_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 shadow trace self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "episode_sha256": self.episode_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "effective_snapshot_sha256": self.effective_snapshot_sha256,
            "rollback_snapshot_sha256": self.rollback_snapshot_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "executor_contract_sha256": self.executor_contract_sha256,
            "deliveries": [row.payload() for row in self.deliveries],
            "activations": [row.payload() for row in self.activations],
            "final_states": [row.payload() for row in self.final_states],
            "terminal_reason": self.terminal_reason.value,
            "problems": list(self.problems),
            "pending_delivery_count": self.pending_delivery_count,
            "completed_rounds": self.completed_rounds,
            "delivery_attempts": self.delivery_attempts,
            "produced_message_count": self.produced_message_count,
            "routed_message_count": self.routed_message_count,
            "routed_payload_bytes": self.routed_payload_bytes,
            "message_envelope_bytes": self.message_envelope_bytes,
            "declared_retained_state_bytes": self.declared_retained_state_bytes,
            "retained_state_byte_rounds": self.retained_state_byte_rounds,
            "work": self.work.payload(),
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_effects": self.factual_effects,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["trace_sha256"] = self.trace_sha256
        return result


def _message_delivery(
    *,
    row: Mapping[str, Any],
    sequence: int,
    available_round: int,
    sender_actor_id: str,
    ports: Mapping[str, G0ShadowPortBinding],
) -> tuple[G0ShadowDelivery | None, tuple[str, ...]]:
    problems: list[str] = []
    recipient = row.get("recipient")
    if not isinstance(recipient, str) or not recipient.startswith("actor:"):
        problems.append("message-recipient-invalid")
        recipient_actor = ""
    else:
        recipient_actor = recipient.removeprefix("actor:")
    port = ports.get(recipient_actor)
    if port is None:
        problems.append("message-recipient-unknown")
    schema_id = row.get("schema_id")
    payload_form = row.get("payload_form")
    payload = row.get("payload")
    if port is not None:
        if schema_id != port.schema_id:
            problems.append("message-schema-port-mismatch")
        if payload_form != port.payload_form:
            problems.append("message-form-port-mismatch")
        encoded_bytes = row.get("encoded_bytes")
        if not isinstance(encoded_bytes, int) or isinstance(encoded_bytes, bool):
            problems.append("message-encoded-bytes-invalid")
        elif encoded_bytes > port.max_encoded_bytes:
            problems.append("message-port-byte-cap-exceeded")
    if problems:
        return None, tuple(sorted(set(problems)))
    _require(isinstance(schema_id, str), "validated message schema_id is not text")
    _require(isinstance(payload_form, str), "validated message payload form is not text")
    source_value = row.get("message_sha256")
    source_sha256 = _require_digest(source_value, "G0 shadow staged message digest")
    typed_schema_id = cast(str, schema_id)
    typed_payload_form = cast(str, payload_form)
    return (
        G0ShadowDelivery.create(
            sequence=sequence,
            available_round=available_round,
            kind=G0ShadowDeliveryKind.MESSAGE,
            source_sha256=source_sha256,
            sender_actor_id=sender_actor_id,
            recipient_actor_id=recipient_actor,
            schema_id=typed_schema_id,
            payload_form=typed_payload_form,
            payload=payload,
        ),
        (),
    )


def _trace(
    *,
    episode: G0ShadowEpisode,
    source: G0ConstructionSnapshot,
    deliveries: Sequence[G0ShadowDelivery],
    activations: Sequence[G0ShadowActivation],
    states: Mapping[str, FrozenJSON],
    terminal_reason: G0ShadowTerminalReason,
    problems: Sequence[str],
    delivery_attempts: int,
) -> G0ShadowTrace:
    delivery_rows = tuple(deliveries)
    activation_rows = tuple(activations)
    final_states = tuple(G0ShadowActorState(actor_id, states[actor_id]) for actor_id in sorted(states))
    completed_rounds = max((row.round_index for row in activation_rows), default=-1) + 1
    produced_message_count = sum(len(row.emitted_message_sha256s) for row in activation_rows)
    message_deliveries = [row for row in delivery_rows if row.kind is G0ShadowDeliveryKind.MESSAGE]
    routed_payload_bytes = sum(len(canonical_bytes(row.payload_value.value())) for row in message_deliveries)
    message_envelope_bytes = sum(
        row.result.message_envelope_bytes
        for row in activation_rows
        if isinstance(row.result, G0CounterfactualEvaluation)
    )
    retained_state_byte_rounds = source.retained_state_bytes * completed_rounds
    evaluation_work = sum((row.work for row in activation_rows), WorkVector.zero())
    setup_and_dispatch = WorkVector(
        indexing_and_graph_maintenance=len(source.actors) + len(episode.ports),
        dispatch_and_exploration=delivery_attempts,
    )
    work = evaluation_work + setup_and_dispatch + WorkVector(retained_byte_time=retained_state_byte_rounds)
    problem_rows = tuple(sorted(set(problems)))
    core = {
        "schema": G0_SHADOW_TRACE_SCHEMA,
        "episode_sha256": episode.episode_sha256,
        "source_snapshot_sha256": source.snapshot_sha256,
        "effective_snapshot_sha256": source.snapshot_sha256,
        "rollback_snapshot_sha256": source.snapshot_sha256,
        "grammar_sha256": episode.grammar_sha256,
        "candidate_registry_sha256": episode.candidate_registry_sha256,
        "executor_contract_sha256": episode.executor_contract_sha256,
        "deliveries": [row.payload() for row in delivery_rows],
        "activations": [row.payload() for row in activation_rows],
        "final_states": [row.payload() for row in final_states],
        "terminal_reason": terminal_reason.value,
        "problems": list(problem_rows),
        "pending_delivery_count": len(delivery_rows) - len(activation_rows),
        "completed_rounds": completed_rounds,
        "delivery_attempts": delivery_attempts,
        "produced_message_count": produced_message_count,
        "routed_message_count": len(message_deliveries),
        "routed_payload_bytes": routed_payload_bytes,
        "message_envelope_bytes": message_envelope_bytes,
        "declared_retained_state_bytes": source.retained_state_bytes,
        "retained_state_byte_rounds": retained_state_byte_rounds,
        "work": work.payload(),
        "counterfactual_only": True,
        "activation_enabled": False,
        "shadow_execution_authorized": False,
        "factual_effects": False,
        "factual_mutation_authorized": False,
        "scientific_promotion_allowed": False,
    }
    return G0ShadowTrace(
        episode_sha256=episode.episode_sha256,
        source_snapshot_sha256=source.snapshot_sha256,
        effective_snapshot_sha256=source.snapshot_sha256,
        rollback_snapshot_sha256=source.snapshot_sha256,
        grammar_sha256=episode.grammar_sha256,
        candidate_registry_sha256=episode.candidate_registry_sha256,
        executor_contract_sha256=episode.executor_contract_sha256,
        deliveries=delivery_rows,
        activations=activation_rows,
        final_states=final_states,
        terminal_reason=terminal_reason,
        problems=problem_rows,
        pending_delivery_count=len(delivery_rows) - len(activation_rows),
        completed_rounds=completed_rounds,
        delivery_attempts=delivery_attempts,
        produced_message_count=produced_message_count,
        routed_message_count=len(message_deliveries),
        routed_payload_bytes=routed_payload_bytes,
        message_envelope_bytes=message_envelope_bytes,
        declared_retained_state_bytes=source.retained_state_bytes,
        retained_state_byte_rounds=retained_state_byte_rounds,
        work=work,
        counterfactual_only=True,
        activation_enabled=False,
        shadow_execution_authorized=False,
        factual_effects=False,
        factual_mutation_authorized=False,
        scientific_promotion_allowed=False,
        trace_sha256=canonical_sha256(core),
    )


def execute_g0_shadow_coalition(
    source: G0ConstructionSnapshot,
    episode: G0ShadowEpisode,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> G0ShadowTrace:
    """Execute one finite, counterfactual-only coalition episode."""

    _require(type(source) is G0ConstructionSnapshot, "G0 shadow source must be an exact snapshot")
    _require(type(episode) is G0ShadowEpisode, "G0 shadow episode must be exact")
    _require(type(grammar) is TopologyGrammar, "G0 shadow grammar must be exact")
    _require(
        type(candidate_registry) is PerspectiveCandidateRegistry,
        "G0 shadow candidate registry must be exact",
    )
    _require(
        episode.source_snapshot_sha256 == source.snapshot_sha256,
        "G0 shadow episode/source authority mismatch",
    )
    _require(episode.grammar_sha256 == grammar.grammar_sha256, "G0 shadow grammar authority mismatch")
    _require(
        episode.candidate_registry_sha256 == candidate_registry.sha256 == grammar.candidate_registry_sha256,
        "G0 shadow candidate-registry authority mismatch",
    )
    _require(
        episode.executor_contract_sha256 == G0_SHADOW_EXECUTOR_CONTRACT_SHA256,
        "G0 shadow executor contract authority mismatch",
    )
    _require(
        episode.caps.max_activations <= grammar.caps.max_shadow_events,
        "G0 shadow activation cap exceeds the grammar shadow-event envelope",
    )
    _require(
        episode.caps.max_rounds <= grammar.caps.max_shadow_events,
        "G0 shadow round cap exceeds the grammar shadow-event envelope",
    )
    _require(
        source.retained_state_bytes <= episode.caps.max_retained_state_bytes,
        "G0 shadow source exceeds retained-state cap",
    )
    actors: dict[str, G0ActorGenotype] = {row.candidate_id: row for row in source.actors}
    ports = {row.actor_id: row for row in episode.ports}
    states = {row.actor_id: row.state for row in episode.initial_states}
    _require(set(actors) == set(ports) == set(states), "G0 shadow actor authority coverage drifted")

    deliveries: list[G0ShadowDelivery] = []
    queue: deque[G0ShadowDelivery] = deque()
    for seed in episode.seeds:
        delivery = G0ShadowDelivery.create(
            sequence=len(deliveries),
            available_round=0,
            kind=G0ShadowDeliveryKind.SEED,
            source_sha256=seed.seed_sha256,
            sender_actor_id=None,
            recipient_actor_id=seed.actor_id,
            schema_id=seed.schema_id,
            payload_form=seed.payload_form,
            payload=seed.payload_value.value(),
        )
        deliveries.append(delivery)
        queue.append(delivery)
    _require(len(queue) <= episode.caps.max_queue_depth, "G0 shadow seed queue exceeds cap")

    activations: list[G0ShadowActivation] = []
    signature_visits: dict[tuple[str, str, str], int] = {}
    delivery_attempts = 0
    charged_actor_operations = 0
    produced_messages = 0
    routed_payload_bytes = 0
    terminal_reason = G0ShadowTerminalReason.QUIESCENT
    terminal_problems: tuple[str, ...] = ()

    while queue:
        current = queue[0]
        delivery_attempts += 1
        actor = actors[current.recipient_actor_id]
        state = states[current.recipient_actor_id]
        signature = (current.recipient_actor_id, current.payload_value.sha256, state.sha256)
        if current.available_round >= episode.caps.max_rounds:
            terminal_reason = G0ShadowTerminalReason.CAP_EXHAUSTED
            terminal_problems = ("round-cap-exhausted",)
            break
        if len(activations) >= episode.caps.max_activations:
            terminal_reason = G0ShadowTerminalReason.CAP_EXHAUSTED
            terminal_problems = ("activation-cap-exhausted",)
            break
        if signature_visits.get(signature, 0) >= episode.caps.max_repeated_state_visits:
            terminal_reason = G0ShadowTerminalReason.REPEATED_STATE
            terminal_problems = ("repeated-actor-input-state",)
            break
        if charged_actor_operations + actor.declared_operations > episode.caps.max_actor_operations:
            terminal_reason = G0ShadowTerminalReason.CAP_EXHAUSTED
            terminal_problems = ("actor-operation-cap-exhausted",)
            break
        if produced_messages + len(actor.message_edges) > episode.caps.max_messages:
            terminal_reason = G0ShadowTerminalReason.CAP_EXHAUSTED
            terminal_problems = ("message-cap-exhausted",)
            break
        if len(queue) - 1 + len(actor.message_edges) > episode.caps.max_queue_depth:
            terminal_reason = G0ShadowTerminalReason.CAP_EXHAUSTED
            terminal_problems = ("queue-cap-exhausted",)
            break

        queue.popleft()
        signature_visits[signature] = signature_visits.get(signature, 0) + 1
        result = attempt_g0_counterfactual(
            actor,
            grammar=grammar,
            candidate_registry=candidate_registry,
            external_inputs={ports[actor.candidate_id].root_node_id: current.payload_value.value()},
            initial_state=cast(dict[str, Any], state.value()),
            attempt_id=f"shadow:{episode.episode_id}/{len(activations):08d}",
        )
        charged_actor_operations += result.work.actor_execution
        if isinstance(result, G0CounterfactualRefusal):
            activation = G0ShadowActivation.create(
                sequence=len(activations),
                round_index=current.available_round,
                actor_id=actor.candidate_id,
                genotype_sha256=actor.genotype_sha256,
                source_delivery_sha256=current.delivery_sha256,
                prior_state=state,
                result=result,
                state_applied=False,
                resulting_state=state,
                emitted_message_sha256s=(),
                enqueued_delivery_sha256s=(),
                problems=(f"evaluation-refused:{result.reason}",),
            )
            activations.append(activation)
            terminal_reason = G0ShadowTerminalReason.EVALUATION_REFUSED
            terminal_problems = activation.problems
            break

        rows = _message_rows(result)
        emitted_ids = tuple(cast(str, row["message_sha256"]) for row in rows)
        produced_messages += len(rows)
        proposed_deliveries: list[G0ShadowDelivery] = []
        delivery_problems: list[str] = []
        for row in rows:
            proposed, problems = _message_delivery(
                row=row,
                sequence=len(deliveries) + len(proposed_deliveries),
                available_round=current.available_round + 1,
                sender_actor_id=actor.candidate_id,
                ports=ports,
            )
            delivery_problems.extend(problems)
            if proposed is not None:
                proposed_deliveries.append(proposed)
        proposed_bytes = sum(len(canonical_bytes(row.payload_value.value())) for row in proposed_deliveries)
        if routed_payload_bytes + proposed_bytes > episode.caps.max_routed_payload_bytes:
            delivery_problems.append("routed-payload-byte-cap-exhausted")
        if delivery_problems:
            activation = G0ShadowActivation.create(
                sequence=len(activations),
                round_index=current.available_round,
                actor_id=actor.candidate_id,
                genotype_sha256=actor.genotype_sha256,
                source_delivery_sha256=current.delivery_sha256,
                prior_state=state,
                result=result,
                state_applied=False,
                resulting_state=state,
                emitted_message_sha256s=emitted_ids,
                enqueued_delivery_sha256s=(),
                problems=delivery_problems,
            )
            activations.append(activation)
            terminal_reason = G0ShadowTerminalReason.DELIVERY_REFUSED
            terminal_problems = activation.problems
            break

        next_state = result.staged_state
        states[actor.candidate_id] = next_state
        deliveries.extend(proposed_deliveries)
        queue.extend(proposed_deliveries)
        routed_payload_bytes += proposed_bytes
        activation = G0ShadowActivation.create(
            sequence=len(activations),
            round_index=current.available_round,
            actor_id=actor.candidate_id,
            genotype_sha256=actor.genotype_sha256,
            source_delivery_sha256=current.delivery_sha256,
            prior_state=state,
            result=result,
            state_applied=True,
            resulting_state=next_state,
            emitted_message_sha256s=emitted_ids,
            enqueued_delivery_sha256s=tuple(row.delivery_sha256 for row in proposed_deliveries),
            problems=(),
        )
        activations.append(activation)

    return _trace(
        episode=episode,
        source=source,
        deliveries=deliveries,
        activations=activations,
        states=states,
        terminal_reason=terminal_reason,
        problems=terminal_problems,
        delivery_attempts=delivery_attempts,
    )


def verify_g0_shadow_trace(
    trace: G0ShadowTrace,
    source: G0ConstructionSnapshot,
    episode: G0ShadowEpisode,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> tuple[str, ...]:
    """Replay a trace from its exact authorities and return all detected problems."""

    if type(trace) is not G0ShadowTrace:
        raise ValueError("G0 shadow verifier requires an exact trace")
    problems: list[str] = []
    actors = {row.candidate_id: row for row in source.actors}
    for activation in trace.activations:
        if isinstance(activation.result, G0CounterfactualEvaluation):
            actor = actors.get(activation.actor_id)
            if actor is None:
                problems.append("activation-actor-absent")
                continue
            problems.extend(
                f"activation-{activation.sequence}:{problem}"
                for problem in verify_g0_counterfactual(
                    activation.result,
                    genotype=actor,
                    grammar=grammar,
                    candidate_registry=candidate_registry,
                )
            )
    try:
        replayed = execute_g0_shadow_coalition(
            source,
            episode,
            grammar=grammar,
            candidate_registry=candidate_registry,
        )
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"replay-refused:{exc}")
    else:
        if replayed != trace:
            problems.append("deterministic-replay-mismatch")
    return tuple(sorted(set(problems)))


__all__ = [
    "G0_SHADOW_ACTIVATION_SCHEMA",
    "G0_SHADOW_DELIVERY_SCHEMA",
    "G0_SHADOW_EPISODE_SCHEMA",
    "G0_SHADOW_EXECUTOR_CONTRACT_SHA256",
    "G0_SHADOW_TRACE_SCHEMA",
    "G0ShadowActivation",
    "G0ShadowActivationStatus",
    "G0ShadowActorState",
    "G0ShadowCaps",
    "G0ShadowDelivery",
    "G0ShadowDeliveryKind",
    "G0ShadowEpisode",
    "G0ShadowPortBinding",
    "G0ShadowSeed",
    "G0ShadowTerminalReason",
    "G0ShadowTrace",
    "execute_g0_shadow_coalition",
    "verify_g0_shadow_trace",
]
