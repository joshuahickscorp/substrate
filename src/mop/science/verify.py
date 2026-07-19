"""Shared independent-verification scaffold (spec sections 9, 17).

The engine and this scaffold supply only SHARED INTEGRITY machinery: seal recomputation, schema and identity
checks, claim-verb ceiling enforcement, and structural validation. The GRADED scientific recompute (the
independent recalculation of the metric and decision from raw results) is injected as ``graded_recompute``
and lives in the experiment family's own verifier module, so producer and verifier graded logic never share
an implementation. The scaffold refuses if the injected recompute disagrees with the sealed verdict.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mop.science.spec import ArmSeedResult, ExperimentSpec
from mop.substrate.events import canonical_sha256


class VerificationRefused(ValueError):
    """Raised when an artifact fails shared-integrity checks or disagrees with the independent recompute."""


def _reconstruct_results(artifact: dict[str, Any]) -> dict[str, list[ArmSeedResult]]:
    out: dict[str, list[ArmSeedResult]] = {}
    for arm, rows in (artifact.get("results") or {}).items():
        out[arm] = [ArmSeedResult(arm=arm, seed=int(r["seed"]), metric_value=float(r["metric_value"]),
                                  receipt=dict(r.get("receipt") or {})) for r in rows]
    return out


def verify_artifact(
    artifact: dict[str, Any],
    spec: ExperimentSpec,
    graded_recompute: Callable[[dict[str, list[ArmSeedResult]], ExperimentSpec], dict[str, Any]],
) -> dict[str, Any]:
    """Return an independent verification receipt. Raises VerificationRefused on any integrity failure.

    ``graded_recompute`` is the family's structurally separate recomputation of the decision from raw arm
    results. Its verdict must match the sealed verdict exactly, or verification refuses.
    """

    body = {k: v for k, v in artifact.items() if k != "seal"}
    if not isinstance(artifact.get("seal"), str) or artifact["seal"] != canonical_sha256(body):
        raise VerificationRefused("artifact self-seal is invalid")
    if artifact.get("schema") != spec.schema:
        raise VerificationRefused("artifact schema does not match the spec")
    if artifact.get("experiment_id") != spec.experiment_id:
        raise VerificationRefused("artifact identity does not match the spec")
    if artifact.get("activation_allowed") is not False or artifact.get("scientific_promotion") is not False:
        raise VerificationRefused("artifact asserts activation or promotion")

    # claim-verb ceiling: no forbidden verb may appear in the claim ceiling text
    claim = str(artifact.get("claim_ceiling") or "")
    for verb in spec.forbidden_claim_verbs:
        if verb in claim:
            raise VerificationRefused(f"forbidden claim verb present: {verb!r}")

    # independent graded recompute from raw results (structurally separate implementation)
    raw = _reconstruct_results(artifact)
    recomputed = graded_recompute(raw, spec)
    sealed_verdict = artifact.get("verdict")
    if recomputed.get("verdict") != sealed_verdict:
        raise VerificationRefused(
            f"independent recompute verdict {recomputed.get('verdict')!r} != sealed {sealed_verdict!r}"
        )
    return {
        "verified": True,
        "schema": spec.schema,
        "experiment_id": spec.experiment_id,
        "seal_reproduced": True,
        "independent_verdict": recomputed.get("verdict"),
        "sealed_verdict": sealed_verdict,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


__all__ = ["VerificationRefused", "verify_artifact"]
