"""Unified campaign engine: precommitted decision-rule evaluation and null-safe branch skipping.

Decision rules are fixed before any result is read (they live in the sealed campaign manifest). After a
parent node seals, each rule's small declarative predicate is evaluated against the sealed artifact; a
firing rule records ``branch:<parent>:<rule>`` so its target nodes become eligible. A rule that does not
fire leaves its branch dead: once the parent is terminal, target nodes reachable only through a dead branch
are SKIPPED. This is the null-safe stopping discipline: a null seals the parent and prunes the
positive-only follow-ups without ever changing the precommitted criteria.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

from .specs import CampaignSpec, NodeSpec
from .state import CampaignState, NodeStatus


def _get_field(artifact: dict[str, Any], dotted: str) -> Any:
    cur: Any = artifact
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def predicate_matches(when: dict[str, Any], artifact: dict[str, Any]) -> bool:
    """Evaluate a small declarative predicate over a sealed artifact.

    Supported forms (all keyed by ``field``, a dotted path):
      {"field": f, "equals": v}
      {"field": f, "in": [v, ...]}
      {"field": f, "op": ">="|">"|"<="|"<"|"==", "value": x}
      {"always": true}
    """

    if when.get("always") is True:
        return True
    field = when.get("field")
    if field is None:
        return False
    value = _get_field(artifact, field)
    if "equals" in when:
        return value == when["equals"]
    if "in" in when:
        return value in when["in"]
    if "op" in when:
        try:
            x = float(value)  # type: ignore[arg-type]
            y = float(when["value"])
        except (TypeError, ValueError):
            return False
        op = when["op"]
        return {
            ">=": x >= y,
            ">": x > y,
            "<=": x <= y,
            "<": x < y,
            "==": x == y,
        }.get(op, False)
    return False


def evaluate_decisions(
    campaign: CampaignSpec, node: NodeSpec, artifact: dict[str, Any], state: CampaignState
) -> list[str]:
    """Fire every precommitted rule of a just-sealed node whose predicate matches. Returns fired rule keys."""

    fired: list[str] = []
    for rule in node.decision_rules:
        if predicate_matches(rule.when, artifact):
            key = f"branch:{node.node_id}:{rule.name}"
            state.record_decision(key)
            fired.append(key)
    return fired


def resolve_skips(campaign: CampaignSpec, state: CampaignState) -> list[str]:
    """Skip every still-pending node that can only be reached through a dead branch of a terminal parent.

    A branch authority ``branch:<parent>:<rule>`` is dead once ``parent`` is terminal and the decision did
    not fire. A node whose branch authority is dead is null-safe skipped (idempotent)."""

    node_ids = {n.node_id for n in campaign.nodes}
    skipped: list[str] = []
    for node in campaign.nodes:
        if state.status(node.node_id) is not NodeStatus.PENDING:
            continue
        branch_auths = [a for a in node.authorities if a.startswith("branch:")]
        for auth in branch_auths:
            _, parent, _rule = auth.split(":", 2)
            parent_terminal = parent in node_ids and state.status(parent) in (
                NodeStatus.SEALED,
                NodeStatus.NULL_SEALED,
                NodeStatus.FAILED,
                NodeStatus.SKIPPED,
            )
            if parent_terminal and not state.decision_fired(auth):
                state.mark_skipped(node.node_id, f"null-safe prune: dead branch {auth}")
                skipped.append(node.node_id)
                break
    return skipped
