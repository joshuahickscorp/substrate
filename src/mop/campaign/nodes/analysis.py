"""Campaign analysis nodes: cross-question mechanism diagnosis, negative-space synthesis, mechanism-card
generation, and the executable Stage-3 readiness gate.

These read only SEALED artifacts (never unsealed sibling outcomes) and produce their own sealed artifacts.
They are how the campaign turns individual results into a substrate-relevant record: mechanism cards come
only from sealed results, nulls are clustered by causal failure family, and the readiness gate reports
which of the twelve pre-substrate evidence gates are met (currently none, honestly).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runners import NodeContext, RunResult, register_runner

_REPO = Path(__file__).resolve().parents[4]
_PROOF = _REPO / "proof"


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# The sealed value-of-computation lane results this session and prior sessions produced.
_SEALED_RESULTS = [
    {
        "id": "starss23_onset",
        "path": "STARSS23_ESCS_BED.json",
        "phenomenon": "temporal_boundary",
        "form": "native_audio",
        "mechanism": "value_of_computation",
        "verdict_hint": "null",
    },
    {
        "id": "starss23_counting",
        "path": "STARSS23_COUNTING_BED.json",
        "phenomenon": "source_counting",
        "form": "native_audio",
        "mechanism": "value_of_computation",
        "verdict_hint": "signal_not_reproduced",
    },
    {
        "id": "starss23_doa",
        "path": "STARSS23_DOA_BED.json",
        "phenomenon": "direction_of_arrival",
        "form": "native_audio",
        "mechanism": "value_of_computation",
        "verdict_hint": "null",
    },
    {
        "id": "starss23_voc_headroom",
        "path": "STARSS23_VOC_HEADROOM.json",
        "phenomenon": "value_of_computation",
        "form": "native_audio",
        "mechanism": "value_of_computation",
        "verdict_hint": "instrument",
    },
]


# ---------------------------------------------------------------------------
# Cross-question mechanism diagnosis (integrates the headroom instrument as one node).
# ---------------------------------------------------------------------------


@register_runner("analysis.mechanism_diagnosis")
def mechanism_diagnosis_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Diagnose the value-of-computation lane across the sealed onset/counting/DoA beds and the headroom
    instrument: what causes gates to fire, and why rate-matched-random matches or beats them."""

    headroom = _load(_PROOF / "STARSS23_VOC_HEADROOM.json")
    findings: list[dict[str, Any]] = []
    if headroom is not None:
        analysis = headroom.get("analysis", {})
        for scope in ("test_fold", "full_subset"):
            for fam in ("count", "doa"):
                a = analysis.get(scope, {}).get(fam)
                if a:
                    findings.append(
                        {
                            "scope": scope,
                            "target": fam,
                            "interpretation": a.get("interpretation"),
                            "refreshable_range": a.get("refreshable_range"),
                        }
                    )
    diagnosis = {
        "question": "why did the value-of-computation gates fail to beat rate-matched-random",
        "decomposition": (
            "the sealed nulls split into two shapes: a WHAT-floor collapse (refreshing the frozen estimator "
            "is worse than a constant, e.g. direction-of-arrival) versus real-but-under-realized headroom "
            "(the label-aware ceiling beats random but the trained gate does not, e.g. source counting)"
        ),
        "headroom_findings": findings,
        "gate_firing_hypotheses": [
            "gates fire on input energy and change proxies, not the marginal value of recomputation",
            "rate-matched-random already catches rare transitions once budget exceeds change density",
            "a weak or anti-informative estimator inverts the value of refreshing (negative range)",
        ],
        "next_preregistered_hypotheses": [
            "a gate that realizes the count headroom robustly across scoring unit and gate architecture",
            "a strong-estimator DoA follow-up isolating whether the null is scheduling or estimator quality",
        ],
    }
    content = {
        "schema": "mop-campaign-mechanism-diagnosis/v1",
        "node_id": ctx.node_id,
        "coverage": {
            "form_family": "native_audio",
            "phenomenon": "value_of_computation",
            "mechanism_family": "value_of_computation",
            "unit_class": "sealed_result",
            "evidence_level": "M1",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_verb": "consistent with",
        "diagnosis": diagnosis,
        "n_findings": len(findings),
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(str(path), seal, "diagnosis_sealed", is_null=False, detail={"n_findings": len(findings)})


# ---------------------------------------------------------------------------
# Negative-space synthesis (cluster nulls by causal failure family).
# ---------------------------------------------------------------------------


_FAILURE_FAMILIES = [
    "weak_or_anti_informative_estimator",
    "proxy_signal_lacks_marginal_value",
    "pseudoreplication_unit",
    "architecture_fragile",
    "absent_heterogeneity_headroom",
    "ceiling_effect",
    "replay_without_future_learning",
    "unstable_private_code",
    "matched_compute_collapse",
    "representation_destroyed_identity",
    "vacuous_control",
    "natural_transfer_failed",
]


@register_runner("analysis.negative_space")
def negative_space_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Cluster the record's sealed nulls into recurring causal failure families and emit precommitted
    replacement directions, so expensive dead shapes are closed rather than rerun with new names."""

    clusters: dict[str, list[str]] = {fam: [] for fam in _FAILURE_FAMILIES}
    # map the known sealed lane results to their diagnosed failure family
    headroom = _load(_PROOF / "STARSS23_VOC_HEADROOM.json")
    if headroom is not None:
        hl = headroom.get("headline", {})
        if hl.get("test_fold_doa_interpretation") == "what_floor_collapse":
            clusters["weak_or_anti_informative_estimator"].append("starss23_doa: negative refreshable range")
    clusters["pseudoreplication_unit"].append("starss23 corpus: 21 clips from 7 rooms are not 21 units")
    clusters["proxy_signal_lacks_marginal_value"].append(
        "value-of-computation gates fire on change/energy proxies not marginal recomputation value"
    )
    clusters["absent_heterogeneity_headroom"].append(
        "EDCM: perspectives had uneven, non-complementary niches"
    )
    clusters["replay_without_future_learning"].append(
        "P6: replay retained but gave no future-learning benefit"
    )
    clusters["matched_compute_collapse"].append("CM7: owned objective lost to random-target and frozen-init")
    nonempty = {k: v for k, v in clusters.items() if v}
    content = {
        "schema": "mop-campaign-negative-space/v1",
        "node_id": ctx.node_id,
        "coverage": {
            "form_family": "cross",
            "phenomenon": "failure_synthesis",
            "mechanism_family": "negative_space",
            "unit_class": "sealed_null",
            "evidence_level": "M0",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_verb": "consistent with",
        "failure_families": nonempty,
        "n_families_populated": len(nonempty),
        "precommitted_replacements": {
            "weak_or_anti_informative_estimator": "strengthen the WHAT before testing the WHEN scheduler",
            "proxy_signal_lacks_marginal_value": "target direct marginal-value labels, not change proxies",
            "pseudoreplication_unit": "score at the room/session/world unit and report the inference ceiling",
        },
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(str(path), seal, "synthesis_sealed", is_null=False, detail={"n_families": len(nonempty)})


# ---------------------------------------------------------------------------
# Mechanism cards from sealed results (M0..M7 replication levels).
# ---------------------------------------------------------------------------


def _evidence_level_for(result: dict[str, Any]) -> str:
    """A sealed lane result reaches at most M1 (one bounded effect) or M0 (mechanics/instrument). None of
    the value-of-computation results reach cross-domain replication, so nothing is >= M2 here."""

    hint = result.get("verdict_hint")
    if hint in ("null", "signal_not_reproduced", "instrument"):
        return "M0"
    return "M1"


@register_runner("analysis.mechanism_cards")
def mechanism_cards_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Generate machine-readable mechanism cards ONLY from sealed results, each with a failure domain and an
    evidence level, into mechanism_cards/. A mechanism with no known failure domain is not understood."""

    cards_dir = _REPO / "mechanism_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    for result in _SEALED_RESULTS:
        sealed = _load(_PROOF / result["path"])
        if sealed is None:
            continue
        card = {
            "card_id": result["id"],
            "phenomenon": result["phenomenon"],
            "form_family": result["form"],
            "mechanism_family": result["mechanism"],
            "hypothesis": f"a trained {result['mechanism']} beats matched controls on {result['phenomenon']}",
            "independent_units": "clips within a small number of rooms (not independent environments)",
            "strongest_control": "rate_matched_random at matched budget",
            "evidence_level": _evidence_level_for(result),
            "failure_domain": {
                "starss23_onset": "energy carries no onset-localizing information at this budget",
                "starss23_counting": "signal did not survive scoring-unit and gate-architecture reproduction",
                "starss23_doa": "estimator anti-informative: refreshing worse than a constant",
                "starss23_voc_headroom": "descriptive instrument; informed is a label-aware ceiling",
            }.get(result["id"], "unknown"),
            "disposition": "kill" if result["verdict_hint"] == "null" else "redesign",
            "forbidden_claim": "no activation, promotion, or confirmation from these sealed runs",
            "source_artifact": result["path"],
        }
        (cards_dir / f"{result['id']}.json").write_text(
            json.dumps(card, indent=1, sort_keys=True), encoding="utf-8"
        )
        cards.append(card)
    content = {
        "schema": "mop-campaign-mechanism-cards/v1",
        "node_id": ctx.node_id,
        "coverage": {
            "form_family": "cross",
            "phenomenon": "mechanism_cataloguing",
            "mechanism_family": "cards",
            "unit_class": "sealed_result",
            "evidence_level": "M0",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_verb": "consistent with",
        "n_cards": len(cards),
        "cards_dir": "mechanism_cards",
        "level_distribution": {
            lvl: sum(1 for c in cards if c["evidence_level"] == lvl)
            for lvl in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
        },
        "cards": cards,
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(str(path), seal, "cards_generated", is_null=False, detail={"n_cards": len(cards)})


# ---------------------------------------------------------------------------
# Executable Stage-3 readiness gate.
# ---------------------------------------------------------------------------


_READINESS_GATES = [
    "useful event or persistence mechanism survives on two form families and fresh units",
    "cross-form binding or transfer beats single-form and shuffled-referent controls",
    "structured state beats matched unstructured controls on prediction or action",
    "a world model improves a sealed decision or planning objective over reactive controls",
    "a memory organization improves both retention and future learning, revision and deletion tested",
    "a plasticity mechanism beats frozen-plus-larger-shell, restart, and fresh-init on a transition",
    "a monitoring or value-estimation mechanism changes behavior beneficially and rejects noisy-TV",
    "heterogeneous modes show reproducible context-disjoint competence",
    "dispatch or limited communication beats tuned fixed, random, homogeneous, and monolithic controls",
    "at least one mechanism survives cross-domain replication and independent reconstruction",
    "full lifecycle cost and density are known",
    "mechanism cards expose enough compatible interfaces to compose without bespoke glue",
]


@register_runner("analysis.readiness_gate")
def readiness_gate_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Evaluate the twelve pre-substrate evidence gates over the sealed record. A gate is met only when a
    sealed mechanism card at evidence level >= M5 supports it; none do yet, so readiness is false and the
    canonical substrate tournament stays closed."""

    cards_dir = _REPO / "mechanism_cards"
    high_level = 0
    if cards_dir.is_dir():
        for cf in cards_dir.glob("*.json"):
            card = _load(cf)
            if card and card.get("evidence_level") in ("M5", "M6", "M7"):
                high_level += 1
    gates = [
        {"gate": g, "met": False, "evidence": "no sealed card at evidence level >= M5"}
        for g in _READINESS_GATES
    ]
    stages = {
        "laboratory_readiness": True,  # governance, controls, falsification exist (Stage 0-2 complete)
        "mechanism_discovery": high_level > 0,
        "architecture_candidate_eligibility": all(g["met"] for g in gates),
        "integrated_substrate_evidence": False,
        "natural_training_readiness": False,
    }
    content = {
        "schema": "mop-campaign-readiness/v1",
        "node_id": ctx.node_id,
        "coverage": {
            "form_family": "cross",
            "phenomenon": "substrate_readiness",
            "mechanism_family": "readiness",
            "unit_class": "sealed_card",
            "evidence_level": "M0",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_verb": "consistent with",
        "stage_ladder": {"current_stage": 2, "of": 5, "stage3_empirically_useful_mechanisms": "not achieved"},
        "gates": gates,
        "n_gates_met": sum(1 for g in gates if g["met"]),
        "n_gates": len(gates),
        "readiness_stages": stages,
        "canonical_substrate_tournament_open": stages["architecture_candidate_eligibility"],
        "verdict": "not_ready_continue_discovery",
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        str(path), seal, "readiness_evaluated", is_null=False, detail={"n_gates_met": content["n_gates_met"]}
    )
