from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .joint_axis_runner import CLAIM_SCOPE, JointAxisResult, JointAxisRunner, JointAxisSpec
from .stability_plasticity_impl import MECHANISM_ARM, run_all
from .stability_plasticity_scaffold import DUAL_AXES, PRIOR_NULL, REQUIRED_CONTROLS

MECHANISM_ID = "stability_plasticity"
REQUIREMENT_ID = "s3.stability_plasticity"
RUNNER_SCHEMA = "mop-stability-plasticity-run/v1"


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
    ("retention_margin", "plasticity_margin"),
    ("retention_margin", "future_learnability_margin"),
)


@dataclass(frozen=True, slots=True)
class RunResult(JointAxisResult):
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal

    @property
    def retention_margin(self) -> float:
        return self.margin(0)

    @property
    def plasticity_margin(self) -> float:
        return self.margin(1)


@dataclass(frozen=True, slots=True)
class StabilityPlasticityRunner(JointAxisRunner):
    mechanism_id: str = MECHANISM_ID
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal
    result_type: ClassVar = RunResult
    run_all: ClassVar = staticmethod(run_all)
