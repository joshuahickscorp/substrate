"""Unified campaign engine: coverage-aware prioritization.

Breadth is a scheduling requirement, not an afterthought. The coverage tracker reads the coverage tags of
every node and scores how much an eligible node would improve breadth across form families, cognitive
phenomena, mechanism families, independent-unit classes, and evidence levels. Until minimum breadth is
reached, no single modality or mechanism framing should dominate the discretionary budget: a node that
opens an untested form family or phenomenon is prioritized over the tenth variation of a covered one.

This is a scheduling preference, not a scientific weighting rule; verdicts are never changed by coverage.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .specs import CampaignSpec, NodeSpec
from .state import CampaignState, NodeStatus


@dataclass
class CoverageReport:
    form_families: dict[str, int] = field(default_factory=dict)
    phenomena: dict[str, int] = field(default_factory=dict)
    mechanism_families: dict[str, int] = field(default_factory=dict)
    unit_classes: dict[str, int] = field(default_factory=dict)
    evidence_levels: dict[str, int] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "form_families": dict(sorted(self.form_families.items())),
            "phenomena": dict(sorted(self.phenomena.items())),
            "mechanism_families": dict(sorted(self.mechanism_families.items())),
            "unit_classes": dict(sorted(self.unit_classes.items())),
            "evidence_levels": dict(sorted(self.evidence_levels.items())),
            "n_form_families": len([k for k in self.form_families if k not in ("none", "cross")]),
            "n_phenomena": len([k for k in self.phenomena if k != "none"]),
            "n_mechanism_families": len([k for k in self.mechanism_families if k != "none"]),
        }


def _tally(pairs: list[tuple[str, str, str, str, str]]) -> CoverageReport:
    rep = CoverageReport()
    for form, phen, mech, unit, level in pairs:
        for key, table in (
            (form, rep.form_families),
            (phen, rep.phenomena),
            (mech, rep.mechanism_families),
            (unit, rep.unit_classes),
            (level, rep.evidence_levels),
        ):
            table[key] = table.get(key, 0) + 1
    return rep


def declared_coverage(campaign: CampaignSpec) -> CoverageReport:
    """Coverage the manifest declares across all non-blocked nodes (the campaign's breadth on paper)."""

    pairs = [
        (
            n.coverage.form_family,
            n.coverage.phenomenon,
            n.coverage.mechanism_family,
            n.coverage.unit_class,
            n.coverage.evidence_level,
        )
        for n in campaign.nodes
        if not n.is_blocked
    ]
    return _tally(pairs)


def achieved_coverage(campaign: CampaignSpec, state: CampaignState) -> CoverageReport:
    """Coverage actually EXECUTED so far (nodes that reached a sealed/null-sealed terminal state)."""

    pairs = []
    for node in campaign.nodes:
        if state.status(node.node_id) in (NodeStatus.SEALED, NodeStatus.NULL_SEALED):
            c = node.coverage
            pairs.append((c.form_family, c.phenomenon, c.mechanism_family, c.unit_class, c.evidence_level))
    return _tally(pairs)


def coverage_bonus(node: NodeSpec, achieved: CoverageReport) -> int:
    """A lower number sorts a node earlier. A node advancing an untested form family or phenomenon gets a
    negative bonus (higher priority); a node piling onto an already-covered dimension gets none."""

    bonus = 0
    c = node.coverage
    if c.form_family not in achieved.form_families and c.form_family not in ("none", "cross"):
        bonus -= 20
    if c.phenomenon not in achieved.phenomena and c.phenomenon != "none":
        bonus -= 15
    if c.mechanism_family not in achieved.mechanism_families and c.mechanism_family != "none":
        bonus -= 10
    if c.unit_class not in achieved.unit_classes and c.unit_class != "none":
        bonus -= 5
    return bonus


def coverage_targets_met(campaign: CampaignSpec, report: CoverageReport) -> dict[str, Any]:
    """Compare declared/achieved coverage against the manifest's coverage targets."""

    targets = campaign.coverage_targets
    p = report.payload()
    return {
        "targets": dict(targets),
        "form_families": {
            "have": p["n_form_families"],
            "target": targets.get("form_families", 6),
            "met": p["n_form_families"] >= targets.get("form_families", 6),
        },
        "phenomena": {
            "have": p["n_phenomena"],
            "target": targets.get("phenomena", 10),
            "met": p["n_phenomena"] >= targets.get("phenomena", 10),
        },
    }
