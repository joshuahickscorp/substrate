"""Scaffold spine for the bounded causal messaging and contradiction repair cluster (epoch G1).

This module raises the SCAFFOLDING axis only. It supplies machine-checkable contracts for three
epoch sub-questions: bounded causal messaging (M1), value of verification (V1), and contradiction
repair (K1). It carries deterministic, seeded mechanics for causal message routing and for
disagreement detection and targeted repair. It declares completeness-checked control sets and a
non-vacuous matched budget for every contract, and it quarantines any real deployment behind an
activation gate that local code cannot pass without an external confirmation receipt.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. Nothing here establishes that bounded messaging, selective verification, or
contradiction repair carries any advantage. The routing and repair helpers operate on toy tuples
of ids and integer claims; they are harness plumbing, not results.

The two named prior nulls this epoch must clear are encoded as fail-closed conditions:
limited-broadcast (bounded causal messaging must beat a limited broadcast baseline) and
disagreement-only (repair may fire only on detected disagreement). Unbounded messaging and
always-on verification are refused by construction.

House style: no em or en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

MESSAGING_REPAIR_SCHEMA = "mop-messaging-repair/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MessagingRepairRefusal(ValueError):
    """Raised whenever a declaration is missing, malformed, unbounded, or outside its scope."""


# ---------------------------------------------------------------------------
# Section A. Shared validators, matched budget, control registry, and named nulls.
# ---------------------------------------------------------------------------


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
    """Fail closed if a declared control set drifts in membership or order."""

    if tuple(declared) != tuple(canonical):
        raise MessagingRepairRefusal(f"{label} controls or order drift")


# Canonical control sets. Order is load-bearing for every contract digest.
BOUNDED_MESSAGE_CONTROLS: tuple[str, ...] = ("no-message", "broadcast-all", "random-route")
VERIFICATION_CONTROLS: tuple[str, ...] = ("no-verify", "always-verify")
REPAIR_CONTROLS: tuple[str, ...] = ("no-message", "broadcast-all", "stale-message", "majority-vote")

ROUTING_RULE = "causal-only"
REPAIR_TRIGGER = "detected-disagreement"

# The named prior nulls that force each contract's bar.
ALLOWED_PRIOR_NULLS: frozenset[str] = frozenset(
    {"limited-broadcast", "disagreement-only", "always-on-verification-suffices"}
)


def verify_control_registry() -> bool:
    """Module-level completeness check: fail closed if any canonical control set is degenerate.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """The matched full-system budget a comparison must be tuned to before it counts.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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


# ---------------------------------------------------------------------------
# Section B. Bounded causal messaging (sub-question M1).
# Contract refuses unbounded broadcast; mechanics route only along causal edges.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundedMessageContract:
    """A declared bandwidth limit and a causal-only routing rule. Refuses unbounded broadcast.

    The prior null forced here is limited-broadcast: bounded causal messaging must beat a limited
    broadcast baseline at matched cost, so an unbounded broadcast is refused by construction.
    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """A deterministic, bandwidth-limited routing plan over a causal edge set.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """Refuse a non-causal edge set: any cycle violates the causal-only rule."""

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
    """Route messages along causal parent to child edges only, capped by the bandwidth limit.

    Deterministic and seeded: when a node has more children than the bandwidth limit, the retained
    children are chosen by a stable seeded hash order, never by wall-clock or unseeded randomness.
    Fails closed on a cyclic (non-causal) edge set. Claim scope: deterministic programmatic
    mechanics only; no capability claim.
    """

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


# ---------------------------------------------------------------------------
# Section C. Value of verification (sub-question V1).
# Verification is selective and matched-cost; controls are no-verify and always-verify.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationValueContract:
    """Declares that verification is SELECTIVE and matched-cost. Refuses always-on verification.

    The prior null forced here is always-on-verification-suffices: a selective verification policy
    must beat both the no-verify floor and the always-verify ceiling at matched cost, so a
    verify_fraction of exactly zero or one is refused. Claim scope: deterministic programmatic
    mechanics only; no capability claim.
    """

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


# ---------------------------------------------------------------------------
# Section D. Contradiction repair (sub-question K1).
# Repair triggers only on detected disagreement; controls span the four baselines.
# ---------------------------------------------------------------------------

REPAIR_METRICS: tuple[str, ...] = (
    "repair_latency",
    "restored_agreement",
    "message_cost",
    "false_repair_rate",
)


@dataclass(frozen=True, slots=True)
class ContradictionRepairContract:
    """Repair fires ONLY on detected disagreement. Controls span the four canonical baselines.

    The prior null forced here is disagreement-only: repair traffic that fires without detected
    disagreement is refused. Claim scope: deterministic programmatic mechanics only; no capability
    claim.
    """

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
    """A deterministic repair plan: targeted messages from an anchor to each dissenter.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """Deterministic majority: highest count, ties broken toward the smaller integer value."""

    counts: dict[int, int] = {}
    for _, value in claims:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def detect_and_repair(*, claims: Sequence[tuple[str, int]], seed: int) -> RepairPlan:
    """Detect disagreement across agent claims and emit targeted repair messages if and only if
    a disagreement exists.

    When every agent agrees, the plan is untriggered and carries no messages: this encodes the
    disagreement-only null directly. When they disagree, the lowest-id majority agent anchors a
    targeted message to each dissenter, never a broadcast. Deterministic and seeded. Claim scope:
    deterministic programmatic mechanics only; no capability claim.
    """

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
    """Fail closed if repair is requested with no detected disagreement (disagreement-only null)."""

    if len({value for _, value in claims}) <= 1:
        raise MessagingRepairRefusal("repair refused: no detected disagreement to repair")


# ---------------------------------------------------------------------------
# Section E. Activation gate. Any real deployment is quarantined behind a receipt that local code
# cannot forge; the gate is OFF by default and encodes that activation is not earned yet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActivationReceipt:
    """An external confirmation receipt required to pass the activation gate.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """A fail-closed gate. Off by default; authorization requires a valid external receipt.

    The gate exists so that scaffold code can never quietly deploy a messaging or repair policy as
    if the epoch bar had been cleared. Claim scope: deterministic programmatic mechanics only; no
    capability claim.
    """

    activated: bool = False
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.activated:
            raise MessagingRepairRefusal("the activation gate cannot be constructed pre-activated locally")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairRefusal("activation gate claim scope cannot be widened")

    def authorize(self, receipt: ActivationReceipt | None = None) -> None:
        """Raise unless a valid external confirmation receipt is supplied. Off by default."""

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


# ---------------------------------------------------------------------------
# Convenience builders (deterministic; used by tests and later wiring).
# ---------------------------------------------------------------------------


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


def coverage() -> dict[str, Sequence[str]]:
    """Static record of which epoch G1 sub-questions this scaffold supports (readiness only).

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    return {
        "M1: does bounded causal messaging beat a limited broadcast baseline?": (
            "BoundedMessageContract declares a bandwidth limit and refuses unbounded broadcast",
            "causal_message_plan routes only along causal edges, capped by the bandwidth limit",
            "controls span no-message, broadcast-all, and random-route at matched cost",
        ),
        "V1: is selective verification worth its matched cost?": (
            "VerificationValueContract requires a strictly selective verify fraction, not always on",
            "controls bound the policy by no-verify and always-verify at matched cost",
        ),
        "K1: does contradiction repair fire only on detected disagreement?": (
            "ContradictionRepairContract triggers only on detected-disagreement",
            "detect_and_repair emits an untriggered empty plan when all agents agree",
            "controls span no-message, broadcast-all, stale-message, and majority-vote",
        ),
    }
