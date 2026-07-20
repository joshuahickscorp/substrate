from __future__ import annotations

from typing import ClassVar

from .construction_search_runner import ConstructionSearchRunner, RunResults
from .construction_search_vec_impl import VecArmResult, vec_evaluate_regime


def _charged_net(result: VecArmResult, per_eval_cost: float) -> float:
    return result.raw_score - per_eval_cost * result.evaluations


class VecRunResults(RunResults):
    __slots__ = ()


class ConstructionSearchVecRunner(ConstructionSearchRunner):
    result_type: ClassVar = VecRunResults
    evaluator: ClassVar = staticmethod(vec_evaluate_regime)
    charger: ClassVar = staticmethod(_charged_net)
