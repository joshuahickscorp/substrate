
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .construction_search_bed import (
    CHEAP_CONTROLS,
    CLAIM_SCOPE,
    MECHANISM_ID,
    ORACLE_REFERENCE,
    REQUIREMENT_ID,
    SEARCH_ARM,
    STAGE,
)
from .construction_search_vec_impl import VecArmResult, vec_evaluate_regime

CONSTRUCTION_SEARCH_EVIDENCE_SCHEMA = "mop-construction-search-evidence/v1"
PRIOR_NULL = (
    "G0 formation mechanics existed without demonstrated efficacy; construction search buys nothing "
    "net of its charged cost unless the charged-cost margin over every cheap control is positive"
)


def _charged_net(result: VecArmResult, per_eval_cost: float) -> float:

    return result.raw_score - per_eval_cost * result.evaluations


def _charged_nets(arms: dict[str, VecArmResult], per_eval_cost: float) -> dict[str, float]:
    return {arm: _charged_net(result, per_eval_cost) for arm, result in arms.items()}


def _sorted_floats(values: dict[str, float]) -> dict[str, float]:
    return {key: values[key] for key in sorted(values)}


@dataclass(frozen=True, slots=True)
class VecRunResults:

    seed: int
    per_eval_cost: float
    favorable_objective_digest: str
    null_objective_digest: str
    favorable_nets: dict[str, float]
    null_nets: dict[str, float]
    favorable_margins: dict[str, float]
    null_margins: dict[str, float]
    favorable_headroom_gap: float
    favorable_beats_all: bool
    null_holds: bool
    controls_cleared: tuple[str, ...]

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "schema": CONSTRUCTION_SEARCH_EVIDENCE_SCHEMA,
            "mechanism_id": MECHANISM_ID,
            "requirement_id": REQUIREMENT_ID,
            "stage": STAGE,
            "seed": self.seed,
            "per_eval_cost": self.per_eval_cost,
            "prior_null": PRIOR_NULL,
            "favorable_objective_digest": self.favorable_objective_digest,
            "null_objective_digest": self.null_objective_digest,
            "favorable_nets": _sorted_floats(self.favorable_nets),
            "null_nets": _sorted_floats(self.null_nets),
            "favorable_margins": _sorted_floats(self.favorable_margins),
            "null_margins": _sorted_floats(self.null_margins),
            "favorable_headroom_gap": self.favorable_headroom_gap,
            "favorable_beats_all": self.favorable_beats_all,
            "null_holds": self.null_holds,
            "controls_cleared": list(self.controls_cleared),
            "claim_scope": CLAIM_SCOPE,
        }

    def evidence_digest(self) -> str:
        return canonical_sha256(self.evidence_payload())

    def detail(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "per_eval_cost": self.per_eval_cost,
            "favorable_margins": _sorted_floats(self.favorable_margins),
            "null_margins": _sorted_floats(self.null_margins),
            "favorable_search_net": self.favorable_nets[SEARCH_ARM],
            "null_search_net": self.null_nets[SEARCH_ARM],
            "favorable_headroom_gap": self.favorable_headroom_gap,
            "favorable_beats_all": self.favorable_beats_all,
            "null_holds": self.null_holds,
        }


class ConstructionSearchVecRunner:

    mechanism_id: str = MECHANISM_ID

    def run(self, bed: Bed, seed: int) -> VecRunResults:

        if seed < 0:
            raise ValueError("run seed must be nonnegative")
        favorable_spec = bed.favorable_regime(seed)
        null_spec = bed.null_regime(seed)
        per_eval_cost = favorable_spec.per_eval_cost

        favorable_arms = vec_evaluate_regime(favorable_spec, seed)
        null_arms = vec_evaluate_regime(null_spec, seed)
        favorable_nets = _charged_nets(favorable_arms, per_eval_cost)
        null_nets = _charged_nets(null_arms, per_eval_cost)

        search_favorable = favorable_nets[SEARCH_ARM]
        search_null = null_nets[SEARCH_ARM]
        favorable_margins = {
            control: search_favorable - favorable_nets[control] for control in CHEAP_CONTROLS
        }
        null_margins = {control: search_null - null_nets[control] for control in CHEAP_CONTROLS}

        favorable_beats_all = all(favorable_margins[control] > 0.0 for control in CHEAP_CONTROLS)
        null_holds = not all(null_margins[control] > 0.0 for control in CHEAP_CONTROLS)
        controls_cleared = tuple(
            control for control in CHEAP_CONTROLS if favorable_margins[control] > 0.0
        )
        headroom_gap = (
            favorable_arms[ORACLE_REFERENCE].raw_score - favorable_arms[SEARCH_ARM].raw_score
        )

        return VecRunResults(
            seed=seed,
            per_eval_cost=per_eval_cost,
            favorable_objective_digest=favorable_spec.digest(),
            null_objective_digest=null_spec.digest(),
            favorable_nets=favorable_nets,
            null_nets=null_nets,
            favorable_margins=favorable_margins,
            null_margins=null_margins,
            favorable_headroom_gap=max(0.0, headroom_gap),
            favorable_beats_all=favorable_beats_all,
            null_holds=null_holds,
            controls_cleared=controls_cleared,
        )

    def mint(self, results: VecRunResults) -> RunReceipt:

        earned = results.favorable_beats_all and results.null_holds
        verdict = VERDICT_MECHANICS_OK if earned else VERDICT_NULL
        controls_cleared = results.controls_cleared if earned else ()
        return mint_demonstration(
            mechanism_id=MECHANISM_ID,
            stage=STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=results.evidence_digest(),
            verdict=verdict,
            detail=results.detail(),
        )
