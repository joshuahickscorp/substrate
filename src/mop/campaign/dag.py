"""Unified campaign engine: the DAG frontier with real dependency and authority gates.

A node becomes runnable when every declared dependency is completed and every declared authority is
satisfied. Serial barriers exist only where a later decision truly depends on an earlier sealed result;
independent nodes (separate gate architectures, independent seeds, independent scoring corroborations,
independent reproduction axes, unrelated waves) share one frontier and run concurrently.

Authorities can be internal (a node id that must complete), a precommitted decision branch
(``branch:<parent>:<rule>`` satisfied only when that rule fires), or an external boundary (a sealed
terminal artifact path, or a live authority the ``AuthorityResolver`` knows how to check). External
boundaries are how work is durably queued behind the live campaign and auto-activated when it clears.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .specs import CampaignSpec, DependencyKind, NodeSpec
from .state import CampaignState, NodeStatus


class AuthorityResolver:
    """Resolves named authorities to satisfied/unsatisfied. Internal node ids resolve from state; decision
    branches from fired decisions; ``path:<file>`` from disk; anything else via an injected checker (for a
    live-process boundary), defaulting to unsatisfied so unknown authorities never falsely open."""

    def __init__(self, external_checker: Callable[[str], bool] | None = None) -> None:
        self._external_checker = external_checker or (lambda _name: False)

    def satisfied(self, authority: str, campaign: CampaignSpec, state: CampaignState) -> bool:
        node_ids = {n.node_id for n in campaign.nodes}
        if authority in node_ids:
            return state.is_completed(authority)
        if authority.startswith("branch:"):
            return state.decision_fired(authority)
        if authority.startswith("path:"):
            return Path(authority[len("path:") :]).exists()
        return self._external_checker(authority)


def dependencies_satisfied(
    node: NodeSpec, campaign: CampaignSpec, state: CampaignState, resolver: AuthorityResolver
) -> bool:
    for dep in node.dependencies:
        if dep.kind is DependencyKind.COMPLETION:
            # a run-after ordering constraint: satisfied by any terminal state (including skipped/failed)
            if not state.is_terminal(dep.node_id):
                return False
        elif dep.kind is DependencyKind.SEAL:
            # needs the earlier node's sealed artifact to read
            if not state.is_completed(dep.node_id):
                return False
        elif not resolver.satisfied(dep.node_id, campaign, state):
            return False
    return True


def authorities_satisfied(
    node: NodeSpec, campaign: CampaignSpec, state: CampaignState, resolver: AuthorityResolver
) -> bool:
    return all(resolver.satisfied(a, campaign, state) for a in node.authorities)


def is_runnable(
    node: NodeSpec, campaign: CampaignSpec, state: CampaignState, resolver: AuthorityResolver
) -> bool:
    if node.is_blocked:
        return False
    if state.status(node.node_id) not in (NodeStatus.PENDING, NodeStatus.ELIGIBLE):
        return False
    return dependencies_satisfied(node, campaign, state, resolver) and authorities_satisfied(
        node, campaign, state, resolver
    )


def refresh_eligibility(campaign: CampaignSpec, state: CampaignState, resolver: AuthorityResolver) -> None:
    """Promote every PENDING node whose gates are now satisfied to ELIGIBLE, and mark contracted-blocked
    nodes BLOCKED so the operator view shows exactly what is waiting on external inputs."""

    for node in campaign.nodes:
        st = state.status(node.node_id)
        if node.is_blocked and st is NodeStatus.PENDING:
            state.mark_blocked(node.node_id, node.blocked_reason)
            continue
        if st is NodeStatus.PENDING and is_runnable(node, campaign, state, resolver):
            state.set_eligible(node.node_id)


def runnable_frontier(
    campaign: CampaignSpec, state: CampaignState, resolver: AuthorityResolver
) -> list[NodeSpec]:
    """The current runnable frontier: every ELIGIBLE node. Call ``refresh_eligibility`` first."""

    return [n for n in campaign.nodes if state.status(n.node_id) is NodeStatus.ELIGIBLE]


def all_terminal(campaign: CampaignSpec, state: CampaignState) -> bool:
    """True when no node can make further progress without an external input clearing."""

    for node in campaign.nodes:
        st = state.status(node.node_id)
        if st in (NodeStatus.PENDING, NodeStatus.ELIGIBLE, NodeStatus.RUNNING):
            # a PENDING node blocked only on an unmet external boundary is not progress we can make now
            if st is NodeStatus.PENDING and node.is_blocked:
                continue
            return False
    return True
