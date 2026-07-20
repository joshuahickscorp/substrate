from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..substrate.events import canonical_sha256
from .stage_ladder import FIRST_ACTIVATION_STAGE, ConfirmationReceipt, MatchedBudget

RECEIPT_SCHEMA = "mop-ladder-run-receipt/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

KIND_DEMONSTRATION = "mechanics-demonstration"
KIND_CONFIRMATION = "scientific-confirmation"
RECEIPT_KINDS = (KIND_DEMONSTRATION, KIND_CONFIRMATION)

VERDICT_NULL = "null"
VERDICT_PENDING = "pending"
VERDICT_MECHANICS_OK = "mechanics-ok"
VERDICT_CLEARED = "cleared"
DEMONSTRATION_VERDICTS = (VERDICT_NULL, VERDICT_PENDING, VERDICT_MECHANICS_OK)
CONFIRMATION_VERDICTS = (VERDICT_NULL, VERDICT_PENDING, VERDICT_CLEARED)

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LadderContractRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise LadderContractRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise LadderContractRefusal(f"{label} must be a lowercase sha256 digest")


@dataclass(frozen=True, slots=True)
class RunReceipt:
    kind: str
    mechanism_id: str
    stage: int
    requirement_id: str
    verdict: str
    controls_cleared: tuple[str, ...]
    evidence_digest: str
    overturns_null: str = ""
    matched: MatchedBudget | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    claim_scope: str = CLAIM_SCOPE
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise LadderContractRefusal(f"unsupported run receipt schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise LadderContractRefusal("run receipt claim scope cannot be widened")
        if self.kind not in RECEIPT_KINDS:
            raise LadderContractRefusal(f"unsupported receipt kind {self.kind!r}")
        _require_id(self.mechanism_id, "RunReceipt.mechanism_id")
        _require_id(self.requirement_id, "RunReceipt.requirement_id")
        _require_sha256(self.evidence_digest, "RunReceipt.evidence_digest")
        if not isinstance(self.stage, int) or isinstance(self.stage, bool) or self.stage < 0:
            raise LadderContractRefusal("RunReceipt.stage must be a nonnegative integer")
        if len(set(self.controls_cleared)) != len(self.controls_cleared):
            raise LadderContractRefusal("RunReceipt.controls_cleared must be unique")
        for control in self.controls_cleared:
            _require_id(control, "RunReceipt.controls_cleared entry")
        allowed = CONFIRMATION_VERDICTS if self.kind == KIND_CONFIRMATION else DEMONSTRATION_VERDICTS
        if self.verdict not in allowed:
            raise LadderContractRefusal(f"verdict {self.verdict!r} is not valid for kind {self.kind!r}")
        if self.verdict == VERDICT_CLEARED:
            if self.kind != KIND_CONFIRMATION:
                raise LadderContractRefusal("only a scientific-confirmation receipt can be cleared")
            if self.matched is None:
                raise LadderContractRefusal("a cleared confirmation must carry a matched budget")
            if not self.controls_cleared:
                raise LadderContractRefusal("a cleared confirmation must clear at least one control")
            if not self.overturns_null.strip():
                raise LadderContractRefusal("a cleared confirmation must name the null it overturns")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "mechanism_id": self.mechanism_id,
            "stage": self.stage,
            "requirement_id": self.requirement_id,
            "verdict": self.verdict,
            "controls_cleared": list(self.controls_cleared),
            "evidence_digest": self.evidence_digest,
            "overturns_null": self.overturns_null,
            "matched": None if self.matched is None else self.matched.payload(),
            "detail": dict(self.detail),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def is_confirmation(self) -> bool:
        return self.kind == KIND_CONFIRMATION and self.verdict == VERDICT_CLEARED


def mint_demonstration(
    *,
    mechanism_id: str,
    stage: int,
    requirement_id: str,
    controls_cleared: Sequence[str],
    evidence_digest: str,
    verdict: str = VERDICT_NULL,
    detail: Mapping[str, Any] | None = None,
) -> RunReceipt:

    if verdict not in DEMONSTRATION_VERDICTS:
        raise LadderContractRefusal(f"demonstration verdict {verdict!r} is not allowed")
    return RunReceipt(
        kind=KIND_DEMONSTRATION,
        mechanism_id=mechanism_id,
        stage=stage,
        requirement_id=requirement_id,
        verdict=verdict,
        controls_cleared=tuple(controls_cleared),
        evidence_digest=evidence_digest,
        detail=dict(detail or {}),
    )


def mint_confirmation(
    *,
    mechanism_id: str,
    stage: int,
    requirement_id: str,
    controls_cleared: Sequence[str],
    overturns_null: str,
    matched: MatchedBudget,
    evidence_digest: str,
    detail: Mapping[str, Any] | None = None,
) -> RunReceipt:

    return RunReceipt(
        kind=KIND_CONFIRMATION,
        mechanism_id=mechanism_id,
        stage=stage,
        requirement_id=requirement_id,
        verdict=VERDICT_CLEARED,
        controls_cleared=tuple(controls_cleared),
        evidence_digest=evidence_digest,
        overturns_null=overturns_null,
        matched=matched,
        detail=dict(detail or {}),
    )


def to_confirmation(receipt: RunReceipt) -> ConfirmationReceipt:

    if not receipt.is_confirmation:
        raise LadderContractRefusal("only a cleared scientific-confirmation receipt can open a stage gate")
    return ConfirmationReceipt(
        requirement_id=receipt.requirement_id,
        digest=receipt.evidence_digest,
        controls_cleared=receipt.controls_cleared,
        overturns_null=receipt.overturns_null,
        matched=receipt.matched,
    )


def scientific_confirmations(receipts: Sequence[RunReceipt]) -> list[RunReceipt]:
    return [receipt for receipt in receipts if receipt.is_confirmation]


def ladder_readiness(receipts: Sequence[RunReceipt]) -> dict[str, Any]:

    confirmations = scientific_confirmations(receipts)
    per_stage: dict[int, int] = {}
    for receipt in confirmations:
        per_stage[receipt.stage] = per_stage.get(receipt.stage, 0) + 1
    return {
        "schema": RECEIPT_SCHEMA,
        "total_receipts": len(receipts),
        "demonstrations": sum(1 for r in receipts if r.kind == KIND_DEMONSTRATION),
        "scientific_confirmations": len(confirmations),
        "confirmations_by_stage": {str(stage): count for stage, count in sorted(per_stage.items())},
        "first_activation_stage": FIRST_ACTIVATION_STAGE,
        "note": "only cleared scientific-confirmation receipts count toward a stage gate",
    }


@runtime_checkable
class Bed(Protocol):
    mechanism_id: str

    def controls(self) -> tuple[str, ...]: ...

    def matched_cost(self) -> MatchedBudget: ...

    def null_regime(self, seed: int) -> Any: ...

    def favorable_regime(self, seed: int) -> Any: ...


@runtime_checkable
class MechanismRunner(Protocol):
    mechanism_id: str

    def run(self, bed: Bed, seed: int) -> Any: ...

    def mint(self, results: Any) -> RunReceipt: ...
