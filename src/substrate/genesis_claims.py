"""The ten primary claims, each reduced to a concrete measured comparison.

Every claim names the two arms it contrasts and the split it is measured on.
Where a claim is about a mechanism rather than a score, it is answered by the
canary that exercises that mechanism, and the canary is cited rather than
paraphrased.

Nothing here selects an arm after seeing the effect. The contrasts are fixed by
construction: a claim about plasticity contrasts a plastic arm with the arm
that is deprived of exactly plasticity, a claim about history order contrasts
it with the arm deprived of exactly history order, and so on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_statistics as S

# Each claim is a contrast between a treatment arm and the control that lacks
# exactly the thing the claim is about.
CONTRASTS: dict[str, dict[str, str]] = {
    "P1": {
        "treatment": "selected",
        "control": "static_frozen_field",
        "reads": "verified rewrites improve future behaviour",
        "why_this_control": "identical in every way except that it may not write durable state",
    },
    "P3": {
        "treatment": "K6_adaptive_topology_field",
        "control": "record_store_null",
        "reads": "topology development adds value beyond record storage",
        "why_this_control": "a pure append-only store is exactly the record-storage alternative",
    },
    "P4": {
        "treatment": "K7_native_mixed_radix_field",
        "control": "K1_monolithic_plastic_field",
        "reads": "precision allocation improves capability-resource tradeoffs",
        "why_this_control": "the same plastic budget without per-region radix selection",
    },
    "P6": {
        "treatment": "selected",
        "control": "shuffled_history_plastic",
        "reads": "histories produce different useful field geometry",
        "why_this_control": "identical plasticity and content, only the order destroyed",
    },
    "P9": {
        "treatment": "selected",
        "control": "record_store_null",
        "reads": "multimodal events update one shared field",
        "why_this_control": "measured on the modality-integration family only",
        "family": "new_modality_integration",
    },
    "P10": {
        "treatment": "selected",
        "control": "decisive_comparator",
        "reads": "the selected field beats the strongest equally plastic alternative",
        "why_this_control": "resolved by rule to the highest scoring fully resourced plastic control",
    },
}

# Claims answered by a mechanism canary rather than by a score contrast.
CANARY_CLAIMS = {
    "P2": ("rollback_removes_benefit", "reversing the rewrite removes the benefit"),
    "P5": ("compiled_procedure_preserves_reliability", "compiled procedures reduce cost without reliability loss"),
    "P7": ("shadow_field_does_not_write", "shadow fields support valid counterfactual thought"),
    "P8": ("migration_preserves_identity", "learning survives interruption and migration"),
}


def _history_scores(rows: Sequence[Mapping[str, Any]], *, family: str | None = None) -> list[S.HistoryScore]:
    accumulated: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if family is not None and row["family"] != family:
            continue
        accumulated.setdefault((row["history_id"], row["arm"]), []).append(row["score"])
    return [
        S.HistoryScore(history_id, arm, sum(values) / len(values))
        for (history_id, arm), values in sorted(accumulated.items())
    ]


def _resolve(name: str, *, selected: str, comparator: str) -> str:
    if name == "selected":
        return selected
    if name == "decisive_comparator":
        return comparator
    return name


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected: str,
    comparator: str,
    canaries: Mapping[str, Any],
    continuity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure every claim and apply the frozen multiplicity correction."""
    results: dict[str, Any] = {}
    p_values: dict[str, float] = {}

    for claim, contrast in CONTRASTS.items():
        treatment = _resolve(contrast["treatment"], selected=selected, comparator=comparator)
        control = _resolve(contrast["control"], selected=selected, comparator=comparator)
        family = contrast.get("family")
        scores = _history_scores(rows, family=family)
        available = {row.arm for row in scores}
        if treatment not in available or control not in available or treatment == control:
            results[claim] = {
                "claim": claim,
                "reads": contrast["reads"],
                "treatment": treatment,
                "control": control,
                "measurable": False,
                "reason": "treatment or control did not run on this split",
                "passes": False,
            }
            continue
        analysis = S.decisive_analysis(scores, candidate=treatment, comparator=control)
        p_values[claim] = analysis["p_value"]
        results[claim] = {
            "claim": claim,
            "reads": contrast["reads"],
            "why_this_control": contrast["why_this_control"],
            "treatment": treatment,
            "control": control,
            "family": family,
            "measurable": True,
            "effect": analysis["effect"],
            "confidence_lower": analysis["confidence_lower"],
            "confidence_upper": analysis["confidence_upper"],
            "p_value": analysis["p_value"],
            "histories": analysis["histories"],
            "passes": analysis["effect"] >= C.SESOI and analysis["confidence_lower"] > 0.0,
        }

    canary_results = canaries.get("canaries", {})
    for claim, (canary_name, reads) in CANARY_CLAIMS.items():
        row = canary_results.get(canary_name)
        if claim == "P8" and continuity is not None:
            passed = bool(row and row.get("all_pass")) and bool(continuity.get("all_pass"))
            results[claim] = {
                "claim": claim,
                "reads": reads,
                "evidence": [canary_name, "continuity_lane"],
                "measurable": True,
                "canary_pass": bool(row and row.get("all_pass")),
                "continuity_pass": bool(continuity.get("all_pass")),
                "passes": passed,
            }
            continue
        results[claim] = {
            "claim": claim,
            "reads": reads,
            "evidence": [canary_name],
            "measurable": row is not None,
            "passes": bool(row and row.get("all_pass")),
        }

    corrected = S.holm(p_values) if p_values else {}
    for claim, correction in corrected.items():
        results[claim]["holm_threshold"] = correction["threshold"]
        results[claim]["holm_rejected"] = correction["rejected"]

    critical = [claim for claim, row in C.CLAIMS.items() if row.get("critical")]
    passing = [claim for claim in critical if results.get(claim, {}).get("passes")]
    return {
        "claims": results,
        "multiplicity_correction": C.STATISTICS["multiplicity_correction"],
        "holm": corrected,
        "critical_claims": critical,
        "passing_claims": sorted(passing),
        "failing_claims": sorted(set(critical) - set(passing)),
        "all_critical_pass": len(passing) == len(critical),
        "decisive_claim": C.DECISIVE_CLAIM,
        "decisive_passes": bool(results.get(C.DECISIVE_CLAIM, {}).get("passes")),
        "activation": False,
    }


def demo() -> None:
    """A flat field must fail every score-based claim, including the decisive one."""
    rows = []
    for family in ("f1", "f2"):
        for history in range(32):
            for arm in (
                "K6_adaptive_topology_field",
                "K7_native_mixed_radix_field",
                "K1_monolithic_plastic_field",
                "static_frozen_field",
                "shuffled_history_plastic",
                "record_store_null",
                C.CANONICAL_S2_ID,
            ):
                rows.append({"family": family, "history_id": history, "arm": arm, "score": 0.4})
    canaries = {"canaries": {name: {"all_pass": True} for _, (name, _reads) in CANARY_CLAIMS.items()}}
    report = evaluate(
        rows,
        selected="K6_adaptive_topology_field",
        comparator=C.CANONICAL_S2_ID,
        canaries=canaries,
        continuity={"all_pass": True},
    )
    assert not report["decisive_passes"], report["claims"]["P10"]
    assert not report["all_critical_pass"]
    assert report["claims"]["P10"]["effect"] == 0.0
    # Mechanism claims backed by passing canaries still pass; that is correct,
    # because they are about the mechanism working, not about winning.
    assert report["claims"]["P2"]["passes"]
    print(f"genesis claims self-check passed: flat field fails {len(report['failing_claims'])} critical claims")


if __name__ == "__main__":
    demo()
