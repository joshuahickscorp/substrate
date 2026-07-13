"""Scaffold spine for Stage 4 of the Form Substrate mechanism ladder: integrated architecture advantage.

Stage 4 asks a single, sharp question: does composing several already confirmed Stage-3 mechanisms into one
integrated architecture buy a JOINT advantage that none of them, and no static composition of them, can reach
at matched compute. This module raises the scaffolding axis only. It supplies machine-checkable contracts that
ENCODE the bar such an advantage must clear, plus a fail-closed entry gate. It runs no model, loads no
weights, touches no network, and asserts no capability.

The load-bearing precondition is composition order: an integrated advantage may not even be measured until
every component mechanism has been INDEPENDENTLY CONFIRMED at Stage 3, each carrying a promotion-gate pass
digest. The Stage-3 modules are never imported here; a confirmation is referenced only through a receipt
dataclass carrying a schema and content digest, so this scaffold has no capability-bearing import surface.

The named prior null this stage must reject is fixed and duplicated as a constant: no joint advantage exists
beyond the best single confirmed mechanism or a static (non-integrated) composition of the confirmed
mechanisms. Every joint-advantage contract refuses to validate unless it names exactly that null and is
measured at matched compute against the strong baseline set.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or natural-data
claim. A green contract here means the declarations are complete and self-consistent, never that any
integrated architecture carries a demonstrated advantage.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

STAGE4_SCHEMA = "mop-stage4-integration-advantage/v1"
# The expected schema of an external Stage-3 promotion receipt. Referenced, never imported, so this scaffold
# depends on no Stage-3 capability code; only on the shape of a confirmation record.
STAGE3_RECEIPT_SCHEMA = "mop-stage3-promotion-receipt/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# The named prior null. An integrated architecture advantage does not exist beyond the best single confirmed
# mechanism or a static composition of the confirmed mechanisms. Stage 4 exists only to reject this null.
PRIOR_NULL = "no-joint-advantage-beyond-best-single-or-static-composition"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The strong baseline set a joint advantage must beat, in canonical order. Membership and order are
# load-bearing so the completeness check can detect a baseline family being dropped or reordered.
STRONG_BASELINES: tuple[str, ...] = (
    "scaled-monolith",
    "tuned-ensemble",
    "best-single-mechanism",
    "static-composition",
)

# The ablation ladder arms, in canonical order. "each-mechanism-alone" expands to one arm per component; the
# other two are single arms.
REQUIRED_ABLATION_ARMS: tuple[str, ...] = (
    "each-mechanism-alone",
    "best-single",
    "static-composition",
)

# The minimum number of independently confirmed Stage-3 mechanisms required before an integration
# battery may run.
MIN_CONFIRMED_MECHANISMS = 2

# The minimum independent replications a Stage-3 confirmation must carry to count as independently confirmed.
MIN_INDEPENDENT_REPLICATIONS = 2


class Stage4Refusal(ValueError):
    """Raised whenever a Stage 4 declaration is missing, malformed, or outside its declared scope."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise Stage4Refusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Stage4Refusal(f"{label} must be a lowercase SHA-256 digest")


def _require_positive(value: int, label: str) -> None:
    if value <= 0:
        raise Stage4Refusal(f"{label} must be positive (non-vacuous)")


# ---------------------------------------------------------------------------
# Section A. Stage-3 confirmation receipt. Referenced, never imported.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stage3ConfirmationReceipt:
    """An independently confirmed Stage-3 mechanism, referenced by schema and digest, never imported.

    The receipt carries the promotion-gate pass digest of the confirming Stage-3 run and a self-recomputable
    content digest for tamper detection. A receipt that does not declare confirmation, or that carries fewer
    than the minimum independent replications, fails closed at construction.

    Claim scope: deterministic programmatic mechanics only; no capability claim. Holding a receipt asserts
    nothing beyond the referenced Stage-3 module having recorded a promotion-gate pass.
    """

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
    """Deterministically build a confirmed Stage-3 receipt with a self-consistent content digest."""

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
    """Return the sorted distinct mechanism ids across confirmed receipts; fail closed on a duplicate id."""

    ids = [r.mechanism_id for r in receipts if r.confirmed]
    if len(set(ids)) != len(ids):
        raise Stage4Refusal("Stage-3 receipts must reference distinct mechanisms; a mechanism was repeated")
    return tuple(sorted(ids))


# ---------------------------------------------------------------------------
# Section B. Matched budget and matched-compute joint-advantage contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    """The full-system budget an arm is held to. Every axis must be positive so the budget is non-vacuous."""

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
        """The two axes that define matched compute: flops and wall-clock time."""

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
    """Declares the matched-compute frontier win the integrated architecture must show vs strong baselines.

    The integrated arm and the baseline arm must sit at matched compute (identical flops and wall-clock
    budget), the strong baseline set must be complete and in canonical order, the declared effect must be
    strictly positive, and the contract must name exactly the prior null this stage rejects.

    Claim scope: deterministic programmatic mechanics only; no capability claim. A valid instance is a
    preregistration of the bar, not a result.
    """

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


# ---------------------------------------------------------------------------
# Section C. Ablation ladder: each-mechanism-alone, best-single, static-composition.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One arm of the ablation ladder. No arm may declare integration; integration is the treatment, not a
    rung."""

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
    """The complete ablation ladder over the battery's mechanisms; fails closed on any missing rung.

    Every ladder label in ``REQUIRED_ABLATION_ARMS`` must appear. The each-mechanism-alone arms must cover
    every declared mechanism exactly once, best-single must be a single declared mechanism, and
    static-composition must compose exactly the full declared mechanism set. This encodes attribution: a
    joint advantage is only credible once each rung below it has been measured.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    """Deterministically build a complete ablation ladder over the given mechanism ids."""

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


# ---------------------------------------------------------------------------
# Section D. Integration battery contract: the composed precondition.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntegrationBatteryContract:
    """Requires at least two independently confirmed Stage-3 receipts before an integrated advantage is
    measured.

    The receipts, the joint-advantage contract, and the ablation ladder must all agree on one mechanism
    set. The joint-advantage contract must name the Stage 4 prior null, and the ladder must cover the same
    mechanisms the receipts confirm. Construction encodes the composition rule: no integrated advantage may
    be asserted before its component mechanisms are each confirmed.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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


# ---------------------------------------------------------------------------
# Section E. Activation gate: off by default, opens only on enough confirmed receipts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stage4EntryGate:
    """A fail-closed gate. It is closed by default and only authorizes a battery when enough Stage-3
    mechanisms are independently confirmed.

    Activation is not earned by construction; the only key that opens the gate is a set of at least
    ``min_confirmed_mechanisms`` distinct confirmed Stage-3 receipts, each carrying a promotion-gate pass
    digest. ``authorize`` with too few receipts raises ``Stage4Refusal``.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so that
    Stage 4 code can never quietly assert an integrated advantage before its components are confirmed.
    """

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
        """Open the gate only when enough distinct confirmed receipts are supplied; return an activation
        digest."""

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
    """Run the entry gate against a battery's receipts and return the activation digest; fail closed
    otherwise."""

    return gate.authorize(contract.receipts)


# ---------------------------------------------------------------------------
# Coverage record.
# ---------------------------------------------------------------------------

SCIENTIFIC_CAPABILITY_CLAIM = False


def coverage() -> dict[str, tuple[str, ...]]:
    """Static record mapping this epoch's Stage 4 sub-questions to what the scaffold encodes (readiness
    only)."""

    return {
        "S4.1 component-confirmation-precondition": (
            "IntegrationBatteryContract refuses fewer than two confirmed Stage-3 receipts",
            "Stage4EntryGate is closed by default and opens only on enough distinct confirmed mechanisms",
            "each receipt must carry a promotion-gate pass digest and a self-consistent content digest",
        ),
        "S4.2 joint-advantage-over-best-single": (
            "JointAdvantageContract names exactly the Stage 4 prior null as its rejection target",
            "the prior null fixes the bar at the best single mechanism or a static composition",
        ),
        "S4.3 matched-compute-frontier": (
            "JointAdvantageContract requires identical flops and wall-clock budget for treatment vs baseline",
            "the strong baseline set is complete and order-checked, so no baseline family can be dropped",
        ),
        "S4.4 ablation-attribution": (
            "AblationLadder requires each-mechanism-alone, best-single, and static-composition rungs",
            "each-mechanism-alone must cover every confirmed mechanism exactly once before a joint claim",
        ),
    }
