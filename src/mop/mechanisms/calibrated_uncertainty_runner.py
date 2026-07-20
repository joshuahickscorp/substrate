from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .calibrated_uncertainty_impl import MECHANISM_ARM, run_all
from .calibrated_uncertainty_scaffold import DUAL_AXES, PRIOR_NULL, REQUIRED_CONTROLS
from .joint_axis_runner import CLAIM_SCOPE, JointAxisResult, JointAxisRunner, JointAxisSpec

MECHANISM_ID = "calibrated_uncertainty"
REQUIREMENT_ID = "s3.calibrated_uncertainty"
RUNNER_SCHEMA = "mop-calibrated-uncertainty-run/v1"


class RunnerRefusal(ValueError):
    pass


SPEC = JointAxisSpec(
    MECHANISM_ID,
    REQUIREMENT_ID,
    RUNNER_SCHEMA,
    DUAL_AXES,
    REQUIRED_CONTROLS,
    PRIOR_NULL,
    MECHANISM_ARM,
    ("risk_margin", "utility_margin"),
    ("selective_risk_reduction_margin", "decision_utility_margin"),
)


@dataclass(frozen=True, slots=True)
class RunResult(JointAxisResult):
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal

    @property
    def risk_margin(self) -> float:
        return self.margin(0)

    @property
    def utility_margin(self) -> float:
        return self.margin(1)


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyRunner(JointAxisRunner):
    mechanism_id: str = MECHANISM_ID
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal
    result_type: ClassVar = RunResult
    run_all: ClassVar = staticmethod(run_all)
