from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .joint_axis_runner import CLAIM_SCOPE, JointAxisResult, JointAxisRunner, JointAxisSpec
from .reducible_novelty_impl import MECHANISM_ARM, run_all
from .reducible_novelty_scaffold import DUAL_AXES, PRIOR_NULL, REQUIRED_CONTROLS

MECHANISM_ID = "reducible_novelty"
REQUIREMENT_ID = "s3.reducible_novelty"
RUNNER_SCHEMA = "mop-reducible-novelty-run/v1"


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
    ("progress_margin", "efficiency_margin"),
    ("learning_progress_margin", "allocation_efficiency_margin"),
)


@dataclass(frozen=True, slots=True)
class RunResult(JointAxisResult):
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal

    @property
    def progress_margin(self) -> float:
        return self.margin(0)

    @property
    def efficiency_margin(self) -> float:
        return self.margin(1)


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyRunner(JointAxisRunner):
    mechanism_id: str = MECHANISM_ID
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE
    spec: ClassVar = SPEC
    refusal: ClassVar = RunnerRefusal
    result_type: ClassVar = RunResult
    run_all: ClassVar = staticmethod(run_all)
