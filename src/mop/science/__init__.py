"""The one MOP scientific experiment engine: typed declaration, shared lifecycle, independent verification.

Families declare an ExperimentSpec and provide their unique mathematics as an ArmRunner and an independent
graded recompute. The engine supplies everything that used to be hand-expanded per axis (paired-arm
execution, decision rules, reproduction floor, claim ceiling, canonical sealing, report projection).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from mop.science.engine import (
    DECISION_RULES,
    ExperimentRefused,
    paired_improvements,
    paired_sign_flip_one_sided,
    render_report,
    run_experiment,
)
from mop.science.spec import ArmRunner, ArmSeedResult, ExperimentSpec, GradedRecompute, MetricSpec
from mop.science.verify import VerificationRefused, verify_artifact

__all__ = [
    "MetricSpec", "ExperimentSpec", "ArmSeedResult", "ArmRunner", "GradedRecompute",
    "run_experiment", "render_report", "paired_improvements", "paired_sign_flip_one_sided",
    "DECISION_RULES", "ExperimentRefused", "verify_artifact", "VerificationRefused",
]
