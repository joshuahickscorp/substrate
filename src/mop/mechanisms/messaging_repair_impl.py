
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

MESSAGING_REPAIR_IMPL_SCHEMA = "mop-messaging-repair-impl/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

DEFAULT_ACTION_BUDGET = 10

NULL_REGIME = "null"
FAVORABLE_REGIME = "favorable"
_REGIME_KINDS = frozenset({NULL_REGIME, FAVORABLE_REGIME})

_AGENT_ROLES = frozenset({"source", "flaky", "sink"})

DECLARED_CONTROLS: tuple[str, ...] = (
    "no-message",
    "broadcast-all",
    "stale-message",
    "no-verify",
    "always-verify",
    "majority-vote",
)


class MessagingRepairImplError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Agent:

    agent_id: str
    estimate: int
    reliable: bool
    role: str

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise MessagingRepairImplError("agent id must not be empty")
        if self.role not in _AGENT_ROLES:
            raise MessagingRepairImplError(f"unsupported agent role {self.role!r}")


@dataclass(frozen=True, slots=True)
class Offer:

    parent: str
    child: str
    value: int
    reliable: bool

    def __post_init__(self) -> None:
        if self.parent == self.child:
            raise MessagingRepairImplError("a causal offer cannot be a self loop")


@dataclass(frozen=True, slots=True)
class MatchedCost:

    action_budget: int

    def __post_init__(self) -> None:
        if self.action_budget <= 0:
            raise MessagingRepairImplError("matched action budget must be positive (non-vacuous)")


@dataclass(frozen=True, slots=True)
class Regime:

    schema: str
    kind: str
    seed: int
    truth: int
    agents: tuple[Agent, ...]
    offers: tuple[Offer, ...]
    action_budget: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MESSAGING_REPAIR_IMPL_SCHEMA:
            raise MessagingRepairImplError(f"unsupported regime schema {self.schema!r}")
        if self.kind not in _REGIME_KINDS:
            raise MessagingRepairImplError(f"unsupported regime kind {self.kind!r}")
        if self.seed < 0:
            raise MessagingRepairImplError("regime seed must be nonnegative")
        if self.action_budget <= 0:
            raise MessagingRepairImplError("regime action budget must be positive")
        if not self.agents:
            raise MessagingRepairImplError("a regime must declare at least one agent")
        if len({agent.agent_id for agent in self.agents}) != len(self.agents):
            raise MessagingRepairImplError("agent ids must be unique")
        if self.claim_scope != CLAIM_SCOPE:
            raise MessagingRepairImplError("regime claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "seed": self.seed,
            "truth": self.truth,
            "agents": [[a.agent_id, a.estimate, a.reliable, a.role] for a in self.agents],
            "offers": [[o.parent, o.child, o.value, o.reliable] for o in self.offers],
            "action_budget": self.action_budget,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class PolicyOutcome:

    policy: str
    final_estimates: tuple[tuple[str, int], ...]
    initial_error: int
    final_error: int
    actions_used: int

    @property
    def improvement(self) -> int:

        return self.initial_error - self.final_error

    def payload(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "final_estimates": [[aid, val] for aid, val in self.final_estimates],
            "initial_error": self.initial_error,
            "final_error": self.final_error,
            "actions_used": self.actions_used,
            "improvement": self.improvement,
        }


PolicyFn = Callable[[Regime, MatchedCost], PolicyOutcome]




def _initial_state(regime: Regime) -> dict[str, int]:
    return {agent.agent_id: agent.estimate for agent in regime.agents}


def _total_error(state: dict[str, int], truth: int) -> int:
    return sum(abs(value - truth) for value in state.values())


def _consensus_value(offers: Sequence[Offer]) -> int | None:
    values = sorted(offer.value for offer in offers)
    if not values:
        return None
    return values[len(values) // 2]


def _suspicion_ordered(offers: Sequence[Offer], consensus: int | None) -> list[Offer]:

    base = consensus if consensus is not None else 0
    return sorted(offers, key=lambda offer: (abs(offer.value - base), offer.parent))


def _grouped_offers(regime: Regime) -> dict[str, list[Offer]]:
    groups: dict[str, list[Offer]] = {}
    for offer in regime.offers:
        groups.setdefault(offer.child, []).append(offer)
    return groups


def _outcome(policy: str, regime: Regime, state: dict[str, int], actions_used: int) -> PolicyOutcome:
    initial_error = _total_error(_initial_state(regime), regime.truth)
    final_error = _total_error(state, regime.truth)
    final = tuple(sorted(state.items()))
    return PolicyOutcome(
        policy=policy,
        final_estimates=final,
        initial_error=initial_error,
        final_error=final_error,
        actions_used=actions_used,
    )




def policy_mechanism(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    state = _initial_state(regime)
    remaining = budget.action_budget
    consensus = _consensus_value(regime.offers)
    anchor: int | None = None
    for child in sorted(_grouped_offers(regime)):
        if remaining <= 0:
            break
        for offer in _suspicion_ordered(_grouped_offers(regime)[child], consensus):
            if remaining <= 0:
                break
            remaining -= 1  # selective verification of this one candidate
            if offer.reliable:
                if remaining <= 0:
                    break
                state[child] = offer.value
                remaining -= 1  # bounded adoption
                anchor = offer.value
                break  # selective: do not verify the remaining offers for this child
    if anchor is not None and remaining > 0:
        for aid in sorted(agent_id for agent_id, value in state.items() if value != anchor):
            if remaining <= 0:
                break
            state[aid] = anchor  # disagreement-triggered repair toward the verified anchor
            remaining -= 1
    return _outcome("messaging-repair", regime, state, budget.action_budget - remaining)




def policy_no_message(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    return _outcome("no-message", regime, _initial_state(regime), 0)


def policy_broadcast_all(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    initial = _initial_state(regime)
    state = dict(initial)
    remaining = budget.action_budget
    ids = sorted(initial)
    for recipient in ids:
        if remaining <= 0:
            break
        for sender in ids:
            if sender == recipient:
                continue
            if remaining <= 0:
                break
            state[recipient] = initial[sender]
            remaining -= 1
    return _outcome("broadcast-all", regime, state, budget.action_budget - remaining)


def policy_stale_message(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    state = _initial_state(regime)
    remaining = budget.action_budget
    for offer in regime.offers:
        if remaining <= 0:
            break
        stale_value = state[offer.child]
        state[offer.child] = stale_value
        remaining -= 1
    return _outcome("stale-message", regime, state, budget.action_budget - remaining)


def policy_no_verify(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    state = _initial_state(regime)
    remaining = budget.action_budget
    for offer in regime.offers:
        if remaining <= 0:
            break
        state[offer.child] = offer.value
        remaining -= 1
    return _outcome("no-verify", regime, state, budget.action_budget - remaining)


def policy_always_verify(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    state = _initial_state(regime)
    remaining = budget.action_budget
    consensus = _consensus_value(regime.offers)
    verified: dict[tuple[str, str], bool] = {}
    for offer in sorted(regime.offers, key=lambda o: (o.child, o.parent)):
        if remaining <= 0:
            break
        remaining -= 1  # exhaustive verification, even of unused offers
        verified[(offer.child, offer.parent)] = offer.reliable
    anchor: int | None = None
    for child in sorted(_grouped_offers(regime)):
        if remaining <= 0:
            break
        for offer in _suspicion_ordered(_grouped_offers(regime)[child], consensus):
            if verified.get((offer.child, offer.parent)) and offer.reliable:
                state[child] = offer.value
                remaining -= 1
                anchor = offer.value
                break
    if anchor is not None and remaining > 0:
        for aid in sorted(agent_id for agent_id, value in state.items() if value != anchor):
            if remaining <= 0:
                break
            state[aid] = anchor
            remaining -= 1
    return _outcome("always-verify", regime, state, budget.action_budget - remaining)


def policy_majority_vote(regime: Regime, budget: MatchedCost) -> PolicyOutcome:

    initial = _initial_state(regime)
    counts: dict[int, int] = {}
    for value in initial.values():
        counts[value] = counts.get(value, 0) + 1
    majority = sorted(counts, key=lambda value: (-counts[value], value))[0]
    state = {agent_id: majority for agent_id in initial}
    remaining = max(0, budget.action_budget - len(initial))
    return _outcome("majority-vote", regime, state, budget.action_budget - remaining)


CONTROL_POLICIES: dict[str, PolicyFn] = {
    "no-message": policy_no_message,
    "broadcast-all": policy_broadcast_all,
    "stale-message": policy_stale_message,
    "no-verify": policy_no_verify,
    "always-verify": policy_always_verify,
    "majority-vote": policy_majority_vote,
}

MECHANISM_POLICY: PolicyFn = policy_mechanism
