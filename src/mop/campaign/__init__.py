"""MOP unified campaign engine.

An append-only durable-DAG research campaign engine that replaces serial bed-by-bed orchestration. It owns
one global resource broker and one shared worker fleet, schedules the whole runnable frontier concurrently
under real dependency and authority gates, adopts live work as an external resource consumer, and durably
queues work behind exact sealed boundaries. See the pre-substrate expansion program doc (doc 31).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from .specs import (
    ArmSpec,
    BedSpec,
    CampaignSpec,
    Coverage,
    DecisionRule,
    Dependency,
    DependencyKind,
    ExternalDependency,
    NodeKind,
    NodeSpec,
    ReproductionSpec,
    ResearchQuestionSpec,
    ResourceClass,
    ResourceRequest,
    VerificationSpec,
)
from .state import CampaignState, NodeStatus

__all__ = [
    "ArmSpec",
    "BedSpec",
    "CampaignSpec",
    "CampaignState",
    "Coverage",
    "DecisionRule",
    "Dependency",
    "DependencyKind",
    "ExternalDependency",
    "NodeKind",
    "NodeSpec",
    "NodeStatus",
    "ReproductionSpec",
    "ResearchQuestionSpec",
    "ResourceClass",
    "ResourceRequest",
    "VerificationSpec",
]
