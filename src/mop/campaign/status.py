"""Unified campaign engine: the operator status view (resource and scientific breadth).

Reports both what the machine is doing (workers, utilization, resource mode, the live campaign it coexists
with) and what the science is doing (coverage by form family and phenomenon, evidence-level distribution,
nulls versus survivors, mechanism cards, readiness), plus the next high-information runnable work and the
exact external blockers. This is the view the mandate requires so progress is legible as breadth, not just
rung count.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

from .broker import BrokerSnapshot
from .coverage import achieved_coverage, coverage_targets_met, declared_coverage
from .specs import CampaignSpec
from .state import CampaignState, NodeStatus


def build_status(
    campaign: CampaignSpec, state: CampaignState, broker_snapshot: BrokerSnapshot | None = None
) -> dict[str, Any]:
    counts = state.counts()
    active = [nid for nid, r in state.records.items() if r.status is NodeStatus.RUNNING]
    eligible = [nid for nid, r in state.records.items() if r.status is NodeStatus.ELIGIBLE]
    blocked = [{"node_id": n.node_id, "blocker": n.blocked_reason} for n in campaign.nodes if n.is_blocked]
    survivors = [
        nid
        for nid, r in state.records.items()
        if r.status is NodeStatus.SEALED and r.verdict in ("survives",)
    ]
    nulls = [
        nid
        for nid, r in state.records.items()
        if r.status is NodeStatus.NULL_SEALED or (r.verdict and "null" in str(r.verdict))
    ]
    declared = declared_coverage(campaign)
    achieved = achieved_coverage(campaign, state)

    # next high-information work: the eligible/pending node advancing the least-covered dimension
    ach = achieved.payload()
    next_work = None
    best = 999
    for node in campaign.nodes:
        if state.status(node.node_id) in (NodeStatus.ELIGIBLE, NodeStatus.PENDING) and not node.is_blocked:
            score = (0 if node.coverage.form_family not in ach["form_families"] else 1) + (
                0 if node.coverage.phenomenon not in ach["phenomena"] else 1
            )
            if score < best:
                best, next_work = score, node.node_id

    return {
        "schema": "mop-campaign-status/v1",
        "campaign_id": campaign.campaign_id,
        "node_counts": counts,
        "n_nodes": len(campaign.nodes),
        "active_nodes": active,
        "runnable_frontier": eligible,
        "completed": counts.get("sealed", 0) + counts.get("null_sealed", 0),
        "survivors": survivors,
        "nulls": nulls,
        "blocked_external": blocked,
        "coverage_declared": declared.payload(),
        "coverage_achieved": achieved.payload(),
        "coverage_targets": coverage_targets_met(campaign, declared),
        "evidence_level_distribution": declared.payload()["evidence_levels"],
        "resource": broker_snapshot.payload() if broker_snapshot else None,
        "external_dependencies": [e.payload() for e in campaign.external_dependencies],
        "next_high_information_work": next_work,
    }


def render_text(status: dict[str, Any]) -> str:
    """A compact human-readable status line for the operator and for Telegram summaries."""

    res = status.get("resource") or {}
    cov = status.get("coverage_achieved", {})
    covd = status.get("coverage_declared", {})
    lines = [
        f"MOP campaign {status['campaign_id']}",
        f"nodes: {status['node_counts']} (of {status['n_nodes']})",
        f"active: {len(status['active_nodes'])}  frontier: {len(status['runnable_frontier'])}"
        f"  completed: {status['completed']}",
        f"forms: {cov.get('n_form_families', 0)}/{covd.get('n_form_families', 0)} declared"
        f"  phenomena: {cov.get('n_phenomena', 0)} / {covd.get('n_phenomena', 0)}",
        f"survivors: {len(status['survivors'])}  nulls: {len(status['nulls'])}"
        f"  blocked-external: {len(status['blocked_external'])}",
        f"resource: {res.get('mode', 'n/a')} workers~{res.get('cpu_budget', '?')}"
        f" nice={res.get('nice_level', '?')} hawking={res.get('hawking_active', '?')}",
        f"next: {status.get('next_high_information_work')}",
    ]
    return "\n".join(lines)
