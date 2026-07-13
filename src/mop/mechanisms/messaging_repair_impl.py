"""Deterministic mechanics for the messaging_repair Stage 3 workstream (epoch messaging_repair).

This module carries the runnable mechanics the runner scores: a bounded causal message router, a
selective verifier, and a disagreement-triggered contradiction repairer, together with the declared
control policies (no-message, broadcast-all, stale-message, no-verify, always-verify, majority-vote).

Every policy operates on the same toy regime under the same matched action budget, so a comparison
between the mechanism and any control is matched-cost by construction. A policy takes a Regime and a
MatchedCost and returns a PolicyOutcome carrying the final per-agent estimates, the total error
before and after, and the number of budgeted actions it spent. The mechanism spends actions on
selective verification and bounded adoption first, then on repair; a control that spends its whole
budget verifying everything has nothing left for repair, which is exactly the value-of-verification
tradeoff the epoch asks about.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim. Nothing
here shows that bounded messaging, selective verification, or contradiction repair carries any real
advantage; the numbers are toy integers on a seeded bed, harness plumbing rather than a result.

House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

MESSAGING_REPAIR_IMPL_SCHEMA = "mop-messaging-repair-impl/v1"

# Must stay byte-identical to the mechanism scaffold CLAIM_SCOPE. Duplicated as a literal so this
# mechanics module carries no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# The default matched action budget shared by every policy. Each verify, each adoption, and each
# repair message costs one action, so a comparison at this budget is matched-cost.
DEFAULT_ACTION_BUDGET = 10

NULL_REGIME = "null"
FAVORABLE_REGIME = "favorable"
_REGIME_KINDS = frozenset({NULL_REGIME, FAVORABLE_REGIME})

_AGENT_ROLES = frozenset({"source", "flaky", "sink"})

# The declared control policies for this epoch, in the order the metric names them.
DECLARED_CONTROLS: tuple[str, ...] = (
    "no-message",
    "broadcast-all",
    "stale-message",
    "no-verify",
    "always-verify",
    "majority-vote",
)


class MessagingRepairImplError(ValueError):
    """Raised when a regime, budget, or agent record is malformed. The mechanics fail closed."""


@dataclass(frozen=True, slots=True)
class Agent:
    """One agent holding a scalar estimate of the hidden target, with a reliability marker.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """A candidate causal message: a parent offers its estimate to a child along a causal edge.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    parent: str
    child: str
    value: int
    reliable: bool

    def __post_init__(self) -> None:
        if self.parent == self.child:
            raise MessagingRepairImplError("a causal offer cannot be a self loop")


@dataclass(frozen=True, slots=True)
class MatchedCost:
    """The matched action budget every policy is charged against. Non-vacuous by construction.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    action_budget: int

    def __post_init__(self) -> None:
        if self.action_budget <= 0:
            raise MessagingRepairImplError("matched action budget must be positive (non-vacuous)")


@dataclass(frozen=True, slots=True)
class Regime:
    """A deterministic seeded task: agents, causal offers, a hidden target, and a matched budget.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """The deterministic result of running one policy against one regime under the matched budget.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    policy: str
    final_estimates: tuple[tuple[str, int], ...]
    initial_error: int
    final_error: int
    actions_used: int

    @property
    def improvement(self) -> int:
        """Total error removed relative to the untouched initial state. Higher is better."""

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


# ---------------------------------------------------------------------------
# Shared deterministic helpers.
# ---------------------------------------------------------------------------


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
    """Order offers least-suspicious first: closeness to the causal consensus, then parent id."""

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


# ---------------------------------------------------------------------------
# The mechanism: bounded causal messaging + selective verification + repair.
# ---------------------------------------------------------------------------


def policy_mechanism(regime: Regime, budget: MatchedCost) -> PolicyOutcome:
    """Route bounded causal messages, verify selectively, and repair on detected disagreement.

    Per child the router considers only causal-edge offers, ordered least-suspicious first, verifies
    candidates one at a time, and adopts the first verified-reliable one (then stops: selective, it
    never verifies the unused offers). Any leftover budget funds repair, which fires only when a
    disagreement with the verified anchor remains. Claim scope: deterministic programmatic mechanics
    only; no capability claim.
    """

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


# ---------------------------------------------------------------------------
# The declared control policies. Each is charged against the same matched budget.
# ---------------------------------------------------------------------------


def policy_no_message(regime: Regime, budget: MatchedCost) -> PolicyOutcome:
    """No messaging: every agent keeps its initial estimate. The trivial floor."""

    return _outcome("no-message", regime, _initial_state(regime), 0)


def policy_broadcast_all(regime: Regime, budget: MatchedCost) -> PolicyOutcome:
    """Unbounded broadcast: every agent floods every other, no causal bound and no verification.

    Deliveries follow a fixed recipient-then-sender order and overwrite blindly until the matched
    budget is spent, so the budget is squandered on non-causal traffic and unverified values.
    """

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
    """Stale messaging: each message carries an outdated value equal to what the child already holds.

    A stale message is delivered and charged, but it carries no new information, so it never moves
    an estimate.
    """

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
    """Messaging with no verification: adopt every causal offer blindly, last write wins per child.

    Without verification there is no trusted anchor, so the disagreement repair cannot fire and any
    unreliable offer that lands last corrupts its child.
    """

    state = _initial_state(regime)
    remaining = budget.action_budget
    for offer in regime.offers:
        if remaining <= 0:
            break
        state[offer.child] = offer.value
        remaining -= 1
    return _outcome("no-verify", regime, state, budget.action_budget - remaining)


def policy_always_verify(regime: Regime, budget: MatchedCost) -> PolicyOutcome:
    """Always-on verification: verify every offer up front, then adopt and repair with the remainder.

    Verifying everything, including offers the router would never use, consumes the matched budget so
    that fewer adoptions and repairs remain affordable. This is the always-verify ceiling.
    """

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
    """Majority-vote repair: pull every agent to the global majority estimate, ignoring structure.

    It disregards causal edges and reliability, so a noisy majority overwrites the informed minority.
    """

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
