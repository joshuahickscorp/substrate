"""Read-only audit adapters that classify the sealed STARSS23 beds against the admission battery.

These three adapters read the sealed bed proofs and CLASSIFY design sufficiency: for each bed they identify
which battery clauses that bed's design would have failed. They are strictly read-only. They never open a
sealed proof for writing, never overwrite a verdict, and never reinterpret a sealed scientific verdict; the
sealed verdict is echoed back verbatim as read, and the classification is an additional design lens layered
beside it, not a change to it.

The classifications are derived from the sealed data, not asserted blindly:

- onset (STARSS23_ESCS_BED): its energy front-end produced a null in which the candidate's advantage over
  rate-matched-random did not exceed the SESOI, so the WHEN features fail ``when_decodability`` (they do not
  decode the marginal value of recomputation above chance) and ``incremental_value`` (an energy featurizer
  cannot add value over an energy/rate/change baseline it already is).
- counting (STARSS23_COUNTING_BED): its favorable mechanics point had zero reproductions in the sealed bed,
  and the cross gate-architecture reproduction is null, so it fails ``architecture_independence``.
- direction of arrival (STARSS23_DOA_BED): its candidate macro-MAE does not beat rate-matched-random on
  either architecture and no architecture survives at the matched budget, so it fails
  ``what_absolute_sufficiency`` and ``oracle_budget_headroom``.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_PROOF_ROOT = Path(__file__).resolve().parents[3] / "proof"

_READ_ONLY_NOTE = (
    "read-only design classification; the sealed verdict is echoed verbatim, never altered, overwritten, "
    "or reinterpreted"
)


def _load_proof(proof_root: Path, filename: str) -> dict[str, Any]:
    """Read a sealed proof as JSON. Read-only; the file is never opened for writing."""

    path = Path(proof_root) / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _classification(
    *,
    bed_id: str,
    artifact: str,
    sealed_verdict: Any,
    failed_clauses: list[str],
    per_clause_reason: dict[str, str],
    derived_from: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bed_id": bed_id,
        "artifact": artifact,
        "sealed_verdict": sealed_verdict,
        "sealed_verdict_preserved": True,
        "read_only": True,
        "design_classification": {
            "failed_clauses": failed_clauses,
            "per_clause_reason": per_clause_reason,
            "derived_from": derived_from,
        },
        "note": _READ_ONLY_NOTE,
    }


def audit_onset(proof_root: Path = _DEFAULT_PROOF_ROOT) -> dict[str, Any]:
    """Classify the sealed onset bed. Design fails when_decodability and incremental_value."""

    bed = _load_proof(proof_root, "STARSS23_ESCS_BED.json")
    stats = bed.get("stats") or {}
    exceeds_sesoi = bool(stats.get("mean_delta_exceeds_sesoi"))
    verdict = bed.get("verdict")

    failed: list[str] = []
    reasons: dict[str, str] = {}
    # The energy front-end produced a null in which the candidate did not clear the SESOI over
    # rate-matched-random, so the WHEN features carry no decodable marginal recomputation value.
    if not exceeds_sesoi:
        failed.append("when_decodability")
        reasons["when_decodability"] = (
            "the energy front-end's candidate advantage over rate-matched-random did not exceed the SESOI "
            f"(mean_delta_exceeds_sesoi={exceeds_sesoi}, verdict={verdict!r}); a probe cannot decode the "
            "marginal value of recomputation from these energy WHEN features above chance"
        )
        failed.append("incremental_value")
        reasons["incremental_value"] = (
            "the WHEN featurizer is itself an energy/onset-flux front-end, so it cannot add value over an "
            "energy/rate/change heuristic baseline it already restates"
        )
    return _classification(
        bed_id=str(bed.get("bed_id") or ""),
        artifact="STARSS23_ESCS_BED.json",
        sealed_verdict=verdict,
        failed_clauses=failed,
        per_clause_reason=reasons,
        derived_from={
            "mean_delta_exceeds_sesoi": exceeds_sesoi,
            "one_sided_p": stats.get("one_sided_p"),
            "sesoi_f1": stats.get("sesoi_f1"),
        },
    )


def audit_counting(proof_root: Path = _DEFAULT_PROOF_ROOT) -> dict[str, Any]:
    """Classify the sealed counting bed. Design fails architecture_independence at reproduction."""

    bed = _load_proof(proof_root, "STARSS23_COUNTING_BED.json")
    verdict = bed.get("verdict")
    reproductions = bed.get("reproductions")

    gate_arch_verdict: Any = None
    try:
        gate_arch = _load_proof(proof_root, "STARSS23_COUNTING_REPRO_gate_arch.json")
        gate_arch_verdict = gate_arch.get("verdict")
    except FileNotFoundError:
        gate_arch_verdict = "unavailable"

    failed: list[str] = []
    reasons: dict[str, str] = {}
    # The main bed's favorable mechanics point had zero reproductions, and the dedicated cross
    # gate-architecture reproduction did not reproduce (its verdict is not mechanics-ok).
    unreproduced = (reproductions == 0) or (gate_arch_verdict not in ("mechanics-ok",))
    if unreproduced:
        failed.append("architecture_independence")
        reasons["architecture_independence"] = (
            f"the sealed bed's favorable point (verdict={verdict!r}) had reproductions={reproductions} and "
            f"the cross gate-architecture reproduction verdict is {gate_arch_verdict!r}; the favorable "
            "result did not reproduce across at least two gate architectures, so no favorable claim is "
            "admissible"
        )
    return _classification(
        bed_id=str(bed.get("bed_id") or ""),
        artifact="STARSS23_COUNTING_BED.json",
        sealed_verdict=verdict,
        failed_clauses=failed,
        per_clause_reason=reasons,
        derived_from={
            "reproductions": reproductions,
            "gate_arch_reproduction_verdict": gate_arch_verdict,
        },
    )


def audit_doa(proof_root: Path = _DEFAULT_PROOF_ROOT) -> dict[str, Any]:
    """Classify the sealed DoA bed. Design fails what_absolute_sufficiency and oracle_budget_headroom."""

    bed = _load_proof(proof_root, "STARSS23_DOA_BED.json")
    verdict = bed.get("verdict")
    stats = bed.get("stats") or {}
    architectures = bed.get("architectures") or []

    per_arch: dict[str, dict[str, Any]] = {}
    candidate_beats_control = False
    any_survives = False
    for arch in architectures:
        arch_stats = stats.get(arch) or {}
        cand = arch_stats.get("candidate_mean_macro_mae_deg")
        rmr = arch_stats.get("rate_matched_random_mean_macro_mae_deg")
        survives = bool(arch_stats.get("survives"))
        beats = cand is not None and rmr is not None and cand < rmr
        candidate_beats_control = candidate_beats_control or beats
        any_survives = any_survives or survives
        per_arch[arch] = {
            "candidate_mean_macro_mae_deg": cand,
            "rate_matched_random_mean_macro_mae_deg": rmr,
            "candidate_beats_rate_matched_random": beats,
            "survives": survives,
        }

    failed: list[str] = []
    reasons: dict[str, str] = {}
    if not candidate_beats_control:
        failed.append("what_absolute_sufficiency")
        reasons["what_absolute_sufficiency"] = (
            "the direction-of-arrival WHAT estimator does not beat rate-matched-random on any architecture "
            "(candidate macro-MAE is not below the rate-matched-random macro-MAE), so the WHAT is "
            "information-poor and is rejected"
        )
    if not any_survives:
        failed.append("oracle_budget_headroom")
        reasons["oracle_budget_headroom"] = (
            "no architecture survives at the matched compute budget; there is no realized advantage over "
            "rate-matched-random at the exact budget, so there is no oracle headroom for a learned policy"
        )
    return _classification(
        bed_id=str(bed.get("bed_id") or ""),
        artifact="STARSS23_DOA_BED.json",
        sealed_verdict=verdict,
        failed_clauses=failed,
        per_clause_reason=reasons,
        derived_from={"architectures": list(architectures), "per_architecture": per_arch},
    )


__all__ = ["audit_onset", "audit_counting", "audit_doa"]
