from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from mop.evidence import canonical_sha256
from mop.ladder.ladder_contracts import mint_demonstration

MATCHED_BUDGET_WALL_NOTE = (
    "wall_ns is a deterministic nominal at a 1 GFLOP/s reference so the artifact is byte-reproducible; "
    "the measured wall is unsealed run provenance, and the authoritative sealed compute axes are the "
    "parameter count and the FLOP ledger"
)


class RecordRefused(ValueError):
    pass


def safety_flags() -> dict[str, bool]:

    return {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    artifact: dict[str, Any]
    verdict: str
    detail: dict[str, Any] = dataclass_field(default_factory=dict)
    prereg: dict[str, Any] | None = None
    receipt_payload: dict[str, Any] | None = None

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def demonstration_receipt(
    *,
    mechanism_id: str,
    controls_cleared: tuple[str, ...],
    evidence: Mapping[str, object],
    verdict: str,
    detail: Mapping[str, object],
    stage: int = 3,
    requirement_id: str = "stage3.confirmed_useful_mechanism",
) -> dict[str, Any]:

    return mint_demonstration(
        mechanism_id=mechanism_id,
        stage=stage,
        requirement_id=requirement_id,
        controls_cleared=controls_cleared,
        evidence_digest=canonical_sha256(evidence),
        verdict=verdict,
        detail=dict(detail),
    ).payload()


def artifact_envelope(
    *,
    schema: str,
    report: Any,
    seeds: Iterable[object],
    per_seed: object,
    stats: Mapping[str, object],
    controls: Mapping[str, object],
    flags: Mapping[str, object],
    verdict: str,
    featurizer: Mapping[str, object],
    gate: Mapping[str, object],
    receipt_payload: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:

    policy = report.policy
    body: dict[str, Any] = {
        "schema": schema,
        "stage": 3,
        "bed_id": policy.bed_id,
        "claim_scope": policy.claim_scope,
        "source_kind": report.source_kind,
        "rights_clean": True,
        "reproductions": 0,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "stats": dict(stats),
        "controls": dict(controls),
        "flags": dict(flags),
        "verdict": verdict,
        "harness": report.payload(),
        "featurizer": dict(featurizer),
        "gate": dict(gate),
        "demonstration_receipt": dict(receipt_payload),
    }
    if hasattr(report, "matched_budget"):
        body.update(
            {
                "matched_budget": report.matched_budget.payload(),
                "matched_budget_wall_note": MATCHED_BUDGET_WALL_NOTE,
                "break_even": report.break_even.payload(),
            }
        )
    additions = dict(extra or {})
    if body.keys() & additions.keys():
        raise RecordRefused("extra artifact fields overlap the shared envelope")
    body.update(additions)
    return body


def finalize_artifact(
    body: Mapping[str, object],
    *,
    verdict: str,
    detail: Mapping[str, object] | None = None,
    prereg: dict[str, Any] | None = None,
    receipt_payload: dict[str, Any] | None = None,
) -> ArtifactResult:

    if "seal" in body:
        raise RecordRefused("an unsealed artifact body is required")
    artifact = dict(body)
    artifact["seal"] = canonical_sha256(artifact)
    return ArtifactResult(
        artifact=artifact,
        verdict=verdict,
        detail=dict(detail or {}),
        prereg=prereg,
        receipt_payload=receipt_payload,
    )


__all__ = [
    "MATCHED_BUDGET_WALL_NOTE",
    "ArtifactResult",
    "RecordRefused",
    "artifact_envelope",
    "demonstration_receipt",
    "finalize_artifact",
    "safety_flags",
]
