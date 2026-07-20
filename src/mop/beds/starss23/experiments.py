from __future__ import annotations

from mop.science.budget import ARM_NEVER_UPDATE, BudgetPolicy

from . import CLAIM_SCOPE, FLOP_CEILING

COUNT_BED_ID = "starss23_escs_source_counting"
COUNT_BUDGET_POLICY = BudgetPolicy(
    "mop-starss23-count-harness/v1",
    COUNT_BED_ID,
    "mae",
    "reestimations",
    "lower",
    ARM_NEVER_UPDATE,
    CLAIM_SCOPE,
    FLOP_CEILING,
    delta_key="delta_mean_mae_control_minus_candidate",
)
__all__ = [
    "COUNT_BED_ID",
    "COUNT_BUDGET_POLICY",
]
