from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

MESSAGING_REPAIR_SCHEMA = "mop-messaging-repair/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MessagingRepairRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise MessagingRepairRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise MessagingRepairRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise MessagingRepairRefusal(f"{label} must not be empty")


def _require_exact_sequence(declared: tuple[str, ...], canonical: tuple[str, ...], label: str) -> None:

    if tuple(declared) != tuple(canonical):
        raise MessagingRepairRefusal(f"{label} controls or order drift")


BOUNDED_MESSAGE_CONTROLS: tuple[str, ...] = ("no-message", "broadcast-all", "random-route")
VERIFICATION_CONTROLS: tuple[str, ...] = ("no-verify", "always-verify")
REPAIR_CONTROLS: tuple[str, ...] = ("no-message", "broadcast-all", "stale-message", "majority-vote")

ROUTING_RULE = "causal-only"
REPAIR_TRIGGER = "detected-disagreement"

ALLOWED_PRIOR_NULLS: frozenset[str] = frozenset(
    {"limited-broadcast", "disagreement-only", "always-on-verification-suffices"}
)


def verify_control_registry() -> bool:

    registry: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("bounded-message", BOUNDED_MESSAGE_CONTROLS),
        ("verification", VERIFICATION_CONTROLS),
        ("repair", REPAIR_CONTROLS),
    )
    for label, controls in registry:
        if len(controls) < 2:
            raise MessagingRepairRefusal(f"{label} control set must declare at least two arms")
        if len(set(controls)) != len(controls):
            raise MessagingRepairRefusal(f"{label} control set has duplicate arms")
        for arm in controls:
            _require_nonempty(arm, f"{label} control arm")
    if "always-verify" not in VERIFICATION_CONTROLS or "no-verify" not in VERIFICATION_CONTROLS:
        raise MessagingRepairRefusal("verification controls must bound both no-verify and always-verify")
    return True


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    messages: int
    verify_calls: int
    flops: int
    memory_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("messages", self.messages),
            ("verify_calls", self.verify_calls),
            ("flops", self.flops),
            ("memory_bytes", self.memory_bytes),
        ):
            if value <= 0:
                raise MessagingRepairRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "messages": self.messages,
            "verify_calls": self.verify_calls,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class BoundedMessageContract:
    schema: str
    bandwidth_limit: int
    max_fanout: int
    routing_rule: str
    allow_unbounded_broadcast: bool
    controls: tuple[str, ...]
    matched: MatchedBudget
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_SCHEMA:
            raise MessagingRepairRefusal(f"unsupported bounded message schema {self.schema!r}")
        if self.routing_rule != ROUTING_RULE:
            raise MessagingRepairRefusal("bounded messaging requires the causal-only routing rule")
        if self.allow_unbounded_broadcast:
            raise MessagingRepairRefusal("unbounded broadcast is refused; messaging must stay bounded")
        if self.bandwidth_limit < 1:
            raise MessagingRepairRefusal("bandwidth limit must be a positive message budget")
        if self.max_fanout < 1 or self.max_fanout > self.bandwidth_limit:
            raise MessagingRepairRefusal("max fanout must be in [1, bandwidth_limit]")
        _require_exact_sequence(self.controls, BOUNDED_MESSAGE_CONTROLS, "bounded message")
        if not self.matched_cost_required:
            raise MessagingRepairRefusal("bounded messaging must require matched full-system cost")
        if self.prior_null != "limited-broadcast":
            raise MessagingRepairRefusal("bounded messaging bar is forced by the limited-broadcast null")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("bounded message claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bandwidth_limit": self.bandwidth_limit,
            "max_fanout": self.max_fanout,
            "routing_rule": self.routing_rule,
            "allow_unbounded_broadcast": self.allow_unbounded_broadcast,
            "controls": list(self.controls),
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MessagePlan:
    schema: str
    seed: int
    bandwidth_limit: int
    routes: tuple[tuple[str, str], ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_SCHEMA:
            raise MessagingRepairRefusal(f"unsupported message plan schema {self.schema!r}")
        if self.seed < 0:
            raise MessagingRepairRefusal("message plan seed must be nonnegative")
        for src, dst in self.routes:
            _require_id(src, "message route source")
            _require_id(dst, "message route destination")
            if src == dst:
                raise MessagingRepairRefusal("a causal route cannot be a self loop")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("message plan claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "bandwidth_limit": self.bandwidth_limit,
            "routes": [list(route) for route in self.routes],
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def _assert_acyclic(children: Mapping[str, tuple[str, ...]]) -> None:

    color: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(node: str, stack: frozenset[str]) -> None:
        if node in stack:
            raise MessagingRepairRefusal("causal routing refuses a cyclic edge set")
        if color.get(node) == 1:
            return
        for child in children.get(node, ()):  # deterministic: children already sorted
            visit(child, stack | {node})
        color[node] = 1

    for node in sorted(children):
        visit(node, frozenset())


def causal_message_plan(*, edges: Sequence[tuple[str, str]], bandwidth_limit: int, seed: int) -> MessagePlan:

    if bandwidth_limit < 1:
        raise MessagingRepairRefusal("bandwidth limit must be a positive message budget")
    if seed < 0:
        raise MessagingRepairRefusal("causal routing seed must be nonnegative")
    children: dict[str, list[str]] = {}
    for src, dst in edges:
        _require_id(src, "causal edge source")
        _require_id(dst, "causal edge destination")
        if src == dst:
            raise MessagingRepairRefusal("a causal edge cannot be a self loop")
        children.setdefault(src, []).append(dst)
        children.setdefault(dst, [])
    frozen_children = {node: tuple(sorted(kids)) for node, kids in children.items()}
    _assert_acyclic(frozen_children)
    routes: list[tuple[str, str]] = []
    for node in sorted(frozen_children):
        kids = frozen_children[node]
        if len(kids) <= bandwidth_limit:
            retained = kids
        else:
            ordered = sorted(kids, key=lambda kid: canonical_sha256([seed, node, kid]))
            retained = tuple(sorted(ordered[:bandwidth_limit]))
        routes.extend((node, kid) for kid in retained)
    return MessagePlan(
        schema=MESSAGING_REPAIR_SCHEMA,
        seed=seed,
        bandwidth_limit=bandwidth_limit,
        routes=tuple(routes),
    )


@dataclass(frozen=True, slots=True)
class VerificationValueContract:
    schema: str
    selective: bool
    verify_fraction: float
    value_metric: str
    controls: tuple[str, ...]
    matched: MatchedBudget
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_SCHEMA:
            raise MessagingRepairRefusal(f"unsupported verification schema {self.schema!r}")
        if not self.selective:
            raise MessagingRepairRefusal("verification value must be declared selective, not always on")
        if not 0.0 < self.verify_fraction < 1.0:
            raise MessagingRepairRefusal(
                "selective verify fraction must lie strictly between no-verify and always-verify"
            )
        _require_id(self.value_metric, "VerificationValueContract.value_metric")
        _require_exact_sequence(self.controls, VERIFICATION_CONTROLS, "verification")
        if not self.matched_cost_required:
            raise MessagingRepairRefusal("verification value must require matched full-system cost")
        if self.prior_null != "always-on-verification-suffices":
            raise MessagingRepairRefusal(
                "verification bar is forced by the always-on-verification-suffices null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("verification claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selective": self.selective,
            "verify_fraction": self.verify_fraction,
            "value_metric": self.value_metric,
            "controls": list(self.controls),
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


REPAIR_METRICS: tuple[str, ...] = (
    "repair_latency",
    "restored_agreement",
    "message_cost",
    "false_repair_rate",
)


@dataclass(frozen=True, slots=True)
class ContradictionRepairContract:
    schema: str
    trigger_condition: str
    controls: tuple[str, ...]
    metrics: tuple[str, ...]
    matched: MatchedBudget
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_SCHEMA:
            raise MessagingRepairRefusal(f"unsupported contradiction repair schema {self.schema!r}")
        if self.trigger_condition != REPAIR_TRIGGER:
            raise MessagingRepairRefusal("repair may trigger only on detected disagreement")
        _require_exact_sequence(self.controls, REPAIR_CONTROLS, "repair")
        _require_exact_sequence(self.metrics, REPAIR_METRICS, "repair metric")
        if not self.matched_cost_required:
            raise MessagingRepairRefusal("contradiction repair must require matched full-system cost")
        if self.prior_null != "disagreement-only":
            raise MessagingRepairRefusal("repair bar is forced by the disagreement-only null")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("contradiction repair claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trigger_condition": self.trigger_condition,
            "controls": list(self.controls),
            "metrics": list(self.metrics),
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class RepairPlan:
    schema: str
    seed: int
    triggered: bool
    majority_value: int | None
    dissenters: tuple[str, ...]
    routes: tuple[tuple[str, str], ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_SCHEMA:
            raise MessagingRepairRefusal(f"unsupported repair plan schema {self.schema!r}")
        if self.seed < 0:
            raise MessagingRepairRefusal("repair plan seed must be nonnegative")
        if not self.triggered and self.routes:
            raise MessagingRepairRefusal("an untriggered repair plan cannot carry any messages")
        if not self.triggered and self.dissenters:
            raise MessagingRepairRefusal("an untriggered repair plan cannot name dissenters")
        if self.triggered and not self.dissenters:
            raise MessagingRepairRefusal("a triggered repair plan must name at least one dissenter")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("repair plan claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "triggered": self.triggered,
            "majority_value": self.majority_value,
            "dissenters": list(self.dissenters),
            "routes": [list(route) for route in self.routes],
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def _majority_value(claims: Sequence[tuple[str, int]]) -> int:

    counts: dict[int, int] = {}
    for _, value in claims:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def detect_and_repair(*, claims: Sequence[tuple[str, int]], seed: int) -> RepairPlan:

    if seed < 0:
        raise MessagingRepairRefusal("repair seed must be nonnegative")
    if not claims:
        raise MessagingRepairRefusal("repair needs at least one agent claim")
    ids = [agent_id for agent_id, _ in claims]
    if len(set(ids)) != len(ids):
        raise MessagingRepairRefusal("agent ids must be unique")
    for agent_id in ids:
        _require_id(agent_id, "agent id")
    distinct_values = {value for _, value in claims}
    if len(distinct_values) <= 1:
        return RepairPlan(
            schema=MESSAGING_REPAIR_SCHEMA,
            seed=seed,
            triggered=False,
            majority_value=None,
            dissenters=(),
            routes=(),
        )
    majority = _majority_value(claims)
    dissenters = tuple(sorted(agent_id for agent_id, value in claims if value != majority))
    anchor = sorted(agent_id for agent_id, value in claims if value == majority)[0]
    routes = tuple((anchor, dissenter) for dissenter in dissenters)
    return RepairPlan(
        schema=MESSAGING_REPAIR_SCHEMA,
        seed=seed,
        triggered=True,
        majority_value=majority,
        dissenters=dissenters,
        routes=routes,
    )


def assert_disagreement_present(claims: Sequence[tuple[str, int]]) -> None:

    if len({value for _, value in claims}) <= 1:
        raise MessagingRepairRefusal("repair refused: no detected disagreement to repair")


@dataclass(frozen=True, slots=True)
class ActivationReceipt:
    license_id: str
    authority: str
    confirmed: bool
    matched_cost_cleared: bool
    controls_cleared: bool

    def __post_init__(self) -> None:
        _require_sha256(self.license_id, "ActivationReceipt.license_id")
        _require_nonempty(self.authority, "ActivationReceipt.authority")

    def is_valid(self) -> bool:
        return bool(self.confirmed and self.matched_cost_cleared and self.controls_cleared)

    def payload(self) -> dict[str, Any]:
        return {
            "license_id": self.license_id,
            "authority": self.authority,
            "confirmed": self.confirmed,
            "matched_cost_cleared": self.matched_cost_cleared,
            "controls_cleared": self.controls_cleared,
        }


@dataclass(frozen=True, slots=True)
class MessagingActivationGate:
    activated: bool = False
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.activated:
            raise MessagingRepairRefusal("the activation gate cannot be constructed pre-activated locally")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("activation gate claim scope cannot be widened")

    def authorize(self, receipt: ActivationReceipt | None = None) -> None:

        if receipt is None:
            raise MessagingRepairRefusal(
                "activation not earned: the messaging gate is off and no confirmation receipt was supplied"
            )
        if not receipt.is_valid():
            raise MessagingRepairRefusal(
                "activation refused: confirmation receipt does not clear matched cost and controls"
            )

    def payload(self) -> dict[str, Any]:
        return {"activated": self.activated, "claim_scope": self.claim_scope}


def _default_budget() -> MatchedBudget:
    return MatchedBudget(messages=64, verify_calls=16, flops=4096, memory_bytes=8192)


def default_bounded_message_contract() -> BoundedMessageContract:
    return BoundedMessageContract(
        schema=MESSAGING_REPAIR_SCHEMA,
        bandwidth_limit=3,
        max_fanout=2,
        routing_rule=ROUTING_RULE,
        allow_unbounded_broadcast=False,
        controls=BOUNDED_MESSAGE_CONTROLS,
        matched=_default_budget(),
        matched_cost_required=True,
        prior_null="limited-broadcast",
    )


def default_verification_value_contract() -> VerificationValueContract:
    return VerificationValueContract(
        schema=MESSAGING_REPAIR_SCHEMA,
        selective=True,
        verify_fraction=0.25,
        value_metric="held_out_error_reduction_per_verify",
        controls=VERIFICATION_CONTROLS,
        matched=_default_budget(),
        matched_cost_required=True,
        prior_null="always-on-verification-suffices",
    )


def default_contradiction_repair_contract() -> ContradictionRepairContract:
    return ContradictionRepairContract(
        schema=MESSAGING_REPAIR_SCHEMA,
        trigger_condition=REPAIR_TRIGGER,
        controls=REPAIR_CONTROLS,
        metrics=REPAIR_METRICS,
        matched=_default_budget(),
        matched_cost_required=True,
        prior_null="disagreement-only",
    )
