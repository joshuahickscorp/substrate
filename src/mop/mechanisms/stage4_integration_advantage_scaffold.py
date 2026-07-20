from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

STAGE4_SCHEMA = "mop-stage4-integration-advantage/v1"
STAGE3_RECEIPT_SCHEMA = "mop-stage3-promotion-receipt/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

PRIOR_NULL = "no-joint-advantage-beyond-best-single-or-static-composition"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

STRONG_BASELINES: tuple[str, ...] = (
    "scaled-monolith",
    "tuned-ensemble",
    "best-single-mechanism",
    "static-composition",
)

REQUIRED_ABLATION_ARMS: tuple[str, ...] = (
    "each-mechanism-alone",
    "best-single",
    "static-composition",
)

MIN_CONFIRMED_MECHANISMS = 2

MIN_INDEPENDENT_REPLICATIONS = 2


class Stage4Refusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise Stage4Refusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Stage4Refusal(f"{label} must be a lowercase SHA-256 digest")


def _require_positive(value: int, label: str) -> None:
    if value <= 0:
        raise Stage4Refusal(f"{label} must be positive (non-vacuous)")


@dataclass(frozen=True, slots=True)
class Stage3ConfirmationReceipt:
    mechanism_id: str
    promotion_gate_digest: str
    independent_replications: int
    confirmed: bool
    confirmation_digest: str
    schema: str = STAGE3_RECEIPT_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE3_RECEIPT_SCHEMA:
            raise Stage4Refusal(f"unsupported Stage-3 receipt schema {self.schema!r}")
        _require_id(self.mechanism_id, "Stage3ConfirmationReceipt.mechanism_id")
        _require_sha256(self.promotion_gate_digest, "Stage3ConfirmationReceipt.promotion_gate_digest")
        _require_sha256(self.confirmation_digest, "Stage3ConfirmationReceipt.confirmation_digest")
        if not self.confirmed:
            raise Stage4Refusal("a Stage-3 receipt that is not confirmed cannot enter a Stage 4 battery")
        if self.independent_replications < MIN_INDEPENDENT_REPLICATIONS:
            raise Stage4Refusal(
                "Stage-3 confirmation requires at least "
                f"{MIN_INDEPENDENT_REPLICATIONS} independent replications"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("Stage-3 receipt claim scope cannot be widened")
        if self.confirmation_digest != canonical_sha256(self._core()):
            raise Stage4Refusal("Stage-3 receipt content digest does not match its declared fields")

    def _core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "promotion_gate_digest": self.promotion_gate_digest,
            "independent_replications": self.independent_replications,
            "confirmed": self.confirmed,
        }

    def payload(self) -> dict[str, Any]:
        body = self._core()
        body["confirmation_digest"] = self.confirmation_digest
        body["claim_scope"] = self.claim_scope
        return body

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_stage3_receipt(
    *,
    mechanism_id: str,
    promotion_gate_digest: str,
    independent_replications: int = MIN_INDEPENDENT_REPLICATIONS,
) -> Stage3ConfirmationReceipt:

    _require_id(mechanism_id, "build_stage3_receipt.mechanism_id")
    _require_sha256(promotion_gate_digest, "build_stage3_receipt.promotion_gate_digest")
    core = {
        "schema": STAGE3_RECEIPT_SCHEMA,
        "mechanism_id": mechanism_id,
        "promotion_gate_digest": promotion_gate_digest,
        "independent_replications": independent_replications,
        "confirmed": True,
    }
    return Stage3ConfirmationReceipt(
        mechanism_id=mechanism_id,
        promotion_gate_digest=promotion_gate_digest,
        independent_replications=independent_replications,
        confirmed=True,
        confirmation_digest=canonical_sha256(core),
    )


def distinct_confirmed_mechanisms(receipts: Sequence[Stage3ConfirmationReceipt]) -> tuple[str, ...]:

    ids = [r.mechanism_id for r in receipts if r.confirmed]
    if len(set(ids)) != len(ids):
        raise Stage4Refusal("Stage-3 receipts must reference distinct mechanisms; a mechanism was repeated")
    return tuple(sorted(ids))


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    params: int
    flops: int
    memory_bytes: int
    wall_clock_ms: int

    def __post_init__(self) -> None:
        _require_positive(self.params, "MatchedBudget.params")
        _require_positive(self.flops, "MatchedBudget.flops")
        _require_positive(self.memory_bytes, "MatchedBudget.memory_bytes")
        _require_positive(self.wall_clock_ms, "MatchedBudget.wall_clock_ms")

    def compute_axes(self) -> tuple[int, int]:

        return (self.flops, self.wall_clock_ms)

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "wall_clock_ms": self.wall_clock_ms,
        }


@dataclass(frozen=True, slots=True)
class JointAdvantageContract:
    schema: str
    integrated_budget: MatchedBudget
    baseline_budget: MatchedBudget
    baselines: tuple[str, ...]
    frontier_metric: str
    min_effect: float
    matched_cost_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_SCHEMA:
            raise Stage4Refusal(f"unsupported joint-advantage schema {self.schema!r}")
        if not self.matched_cost_required:
            raise Stage4Refusal("joint advantage must require matched full-system cost")
        if self.integrated_budget.compute_axes() != self.baseline_budget.compute_axes():
            raise Stage4Refusal(
                "joint advantage must be measured at matched compute (identical flops and wall clock)"
            )
        if tuple(self.baselines) != STRONG_BASELINES:
            raise Stage4Refusal("joint-advantage baselines are incomplete or out of canonical order")
        _require_id(self.frontier_metric, "JointAdvantageContract.frontier_metric")
        if not (self.min_effect > 0.0):
            raise Stage4Refusal("joint advantage requires a strictly positive minimum effect")
        if self.replication_min < MIN_INDEPENDENT_REPLICATIONS:
            raise Stage4Refusal("joint advantage requires at least two independent replications")
        if self.prior_null != PRIOR_NULL:
            raise Stage4Refusal("joint advantage must name exactly the Stage 4 prior null")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("joint-advantage claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "integrated_budget": self.integrated_budget.payload(),
            "baseline_budget": self.baseline_budget.payload(),
            "baselines": list(self.baselines),
            "frontier_metric": self.frontier_metric,
            "min_effect": self.min_effect,
            "matched_cost_required": self.matched_cost_required,
            "replication_min": self.replication_min,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class AblationArm:
    arm: str
    mechanism_ids: tuple[str, ...]
    integrated: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.arm not in REQUIRED_ABLATION_ARMS:
            raise Stage4Refusal(f"unsupported ablation arm {self.arm!r}")
        if not self.mechanism_ids:
            raise Stage4Refusal("an ablation arm must name at least one mechanism")
        if len(set(self.mechanism_ids)) != len(self.mechanism_ids):
            raise Stage4Refusal("ablation arm mechanism ids must be unique")
        for mechanism_id in self.mechanism_ids:
            _require_id(mechanism_id, "AblationArm.mechanism_id")
        if self.integrated:
            raise Stage4Refusal(
                "no ablation arm may be integrated; integration is the treatment, not an ablation"
            )
        if self.arm in {"each-mechanism-alone", "best-single"} and len(self.mechanism_ids) != 1:
            raise Stage4Refusal(f"the {self.arm} arm must name exactly one mechanism")
        if self.arm == "static-composition" and len(self.mechanism_ids) < 2:
            raise Stage4Refusal("the static-composition arm must compose at least two mechanisms")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("ablation arm claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "mechanism_ids": list(self.mechanism_ids),
            "integrated": self.integrated,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class AblationLadder:
    schema: str
    mechanism_ids: tuple[str, ...]
    arms: tuple[AblationArm, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_SCHEMA:
            raise Stage4Refusal(f"unsupported ablation ladder schema {self.schema!r}")
        if len(self.mechanism_ids) < MIN_CONFIRMED_MECHANISMS:
            raise Stage4Refusal("an ablation ladder needs at least two mechanisms")
        if len(set(self.mechanism_ids)) != len(self.mechanism_ids):
            raise Stage4Refusal("ablation ladder mechanism ids must be unique")
        declared = set(self.mechanism_ids)
        labels_present = {arm.arm for arm in self.arms}
        if labels_present != set(REQUIRED_ABLATION_ARMS):
            missing = set(REQUIRED_ABLATION_ARMS) - labels_present
            raise Stage4Refusal(f"ablation ladder is missing rungs {sorted(missing)}")
        for arm in self.arms:
            if not set(arm.mechanism_ids) <= declared:
                raise Stage4Refusal("an ablation arm references a mechanism outside the declared set")
        alone = [arm for arm in self.arms if arm.arm == "each-mechanism-alone"]
        covered = [arm.mechanism_ids[0] for arm in alone]
        if len(covered) != len(declared) or set(covered) != declared:
            raise Stage4Refusal("each-mechanism-alone rungs must cover every declared mechanism exactly once")
        best_single = [arm for arm in self.arms if arm.arm == "best-single"]
        if len(best_single) != 1:
            raise Stage4Refusal("the ablation ladder needs exactly one best-single rung")
        static = [arm for arm in self.arms if arm.arm == "static-composition"]
        if len(static) != 1:
            raise Stage4Refusal("the ablation ladder needs exactly one static-composition rung")
        if set(static[0].mechanism_ids) != declared:
            raise Stage4Refusal("the static-composition rung must compose exactly the full mechanism set")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("ablation ladder claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_ids": list(self.mechanism_ids),
            "arms": [arm.payload() for arm in self.arms],
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_ablation_ladder(mechanism_ids: Sequence[str]) -> AblationLadder:

    ordered = tuple(mechanism_ids)
    if len(ordered) < MIN_CONFIRMED_MECHANISMS:
        raise Stage4Refusal("building an ablation ladder needs at least two mechanisms")
    arms: list[AblationArm] = [
        AblationArm(arm="each-mechanism-alone", mechanism_ids=(mechanism_id,), integrated=False)
        for mechanism_id in ordered
    ]
    arms.append(AblationArm(arm="best-single", mechanism_ids=(ordered[0],), integrated=False))
    arms.append(AblationArm(arm="static-composition", mechanism_ids=ordered, integrated=False))
    return AblationLadder(schema=STAGE4_SCHEMA, mechanism_ids=ordered, arms=tuple(arms))


@dataclass(frozen=True, slots=True)
class IntegrationBatteryContract:
    schema: str
    receipts: tuple[Stage3ConfirmationReceipt, ...]
    joint_advantage: JointAdvantageContract
    ablation: AblationLadder
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_SCHEMA:
            raise Stage4Refusal(f"unsupported integration battery schema {self.schema!r}")
        if len(self.receipts) < MIN_CONFIRMED_MECHANISMS:
            raise Stage4Refusal(
                f"an integration battery needs at least {MIN_CONFIRMED_MECHANISMS} confirmed Stage-3 receipts"
            )
        confirmed_ids = distinct_confirmed_mechanisms(self.receipts)
        if len(confirmed_ids) != len(self.receipts):
            raise Stage4Refusal("every battery receipt must be confirmed and reference a distinct mechanism")
        if set(self.ablation.mechanism_ids) != set(confirmed_ids):
            raise Stage4Refusal("the ablation ladder must cover exactly the confirmed mechanism set")
        if self.prior_null != PRIOR_NULL:
            raise Stage4Refusal("integration battery must name exactly the Stage 4 prior null")
        if self.joint_advantage.prior_null != PRIOR_NULL:
            raise Stage4Refusal("integration battery joint advantage must reject the Stage 4 prior null")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("integration battery claim scope cannot be widened")

    @property
    def mechanism_ids(self) -> tuple[str, ...]:
        return distinct_confirmed_mechanisms(self.receipts)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipts": [receipt.payload() for receipt in self.receipts],
            "joint_advantage": self.joint_advantage.payload(),
            "ablation": self.ablation.payload(),
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class Stage4EntryGate:
    min_confirmed_mechanisms: int = MIN_CONFIRMED_MECHANISMS
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.min_confirmed_mechanisms < MIN_CONFIRMED_MECHANISMS:
            raise Stage4Refusal(
                f"the Stage 4 entry gate requires at least {MIN_CONFIRMED_MECHANISMS} confirmed mechanisms"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4Refusal("Stage 4 entry gate claim scope cannot be widened")

    def authorize(self, receipts: Sequence[Stage3ConfirmationReceipt]) -> str:

        if not receipts:
            raise Stage4Refusal("Stage 4 entry gate is closed: no confirmed Stage-3 receipts were supplied")
        for receipt in receipts:
            if not receipt.confirmed:
                raise Stage4Refusal("Stage 4 entry gate refuses an unconfirmed Stage-3 receipt")
            if receipt.independent_replications < MIN_INDEPENDENT_REPLICATIONS:
                raise Stage4Refusal("Stage 4 entry gate refuses a receipt below the replication floor")
        confirmed_ids = distinct_confirmed_mechanisms(receipts)
        if len(confirmed_ids) < self.min_confirmed_mechanisms:
            raise Stage4Refusal(
                "Stage 4 entry gate is closed: "
                f"{len(confirmed_ids)} confirmed mechanisms supplied, "
                f"{self.min_confirmed_mechanisms} required"
            )
        return canonical_sha256(
            {
                "schema": STAGE4_SCHEMA,
                "gate": "stage4-entry",
                "min_confirmed_mechanisms": self.min_confirmed_mechanisms,
                "confirmed_mechanisms": list(confirmed_ids),
                "gate_digests": sorted(receipt.digest() for receipt in receipts),
            }
        )

    def payload(self) -> dict[str, Any]:
        return {
            "min_confirmed_mechanisms": self.min_confirmed_mechanisms,
            "claim_scope": self.claim_scope,
        }


def authorize_battery(gate: Stage4EntryGate, contract: IntegrationBatteryContract) -> str:

    return gate.authorize(contract.receipts)


SCIENTIFIC_CAPABILITY_CLAIM = False
