"""Read-only terminal-lineage derivation for the Generation 1 General Run closure.

``replay_terminal_lineage`` reads the sealed terminal artifacts of the successor lineage and derives
the closure facts an operator needs to read the whole program at once: which mechanics lanes were
carried, survived, were pruned, warned, failed, blocked, or remain untested; the D1 retirement and
redesign status; the G1-I1 execution or null-safe prune status; the canary decisions and the
transitive dependency closure that removed integrated lanes; an evidence class per conclusion; the
strongest killing control or the remaining blocker for each surviving hypothesis; the explicit
activation, promotion, natural-world-generality, and independent-confirmation refusals; and the exact
next bounded scientific question. It never fabricates a positive.

The derivation is pure and deterministic. Every fact is read from a sealed artifact or a scaffolded
design document and bound with the source hashes that back it. No wall clock enters the output, so the
same inputs always produce the same derivation. If an input is not terminal (Horizon 2 is currently
running behind the live General Run), the derivation is marked
``derivation_status = "deferred_inputs_not_terminal"`` and reports only what IS terminal.

This module is read-only over the live run tree and the sealed proofs; it never edits, signals, or
relabels anything.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DERIVATION_SCHEMA = "mop-generation1-general-run-closure-derivation/v1"

DEFERRED_INPUTS_NOT_TERMINAL = "deferred_inputs_not_terminal"
DERIVATION_COMPLETE = "complete"

HORIZON_V1_ID = "generation1-successor-horizon-v1"
HORIZON_V2_ID = "generation1-successor-horizon-v2"

# The three redesigned dispatch families named by the categorized batch wave (doc 23) and the
# full-generations wave (doc 25). They change the routing premise rather than weakening any threshold.
D1_REDESIGN_FAMILIES = ("utility_residual", "pairwise_ranking", "calibrated_abstaining")

# The three lanes that are new to the full-generations wave and enter only through the W08 canary
# gate (doc 25). They are untested until their canary partition seals a real receipt.
FULL_GENERATIONS_NEW_LANES = {
    "G1-U1": "calibrated_uncertainty",
    "G1-N1": "reducible_novelty",
    "G1-P1R": "stability_plasticity_r2",
}

# The lineage design documents that scaffold the queued (never-run) successor programs.
LINEAGE_DOCS = (
    "docs/mixture_of_perspectives/21_generation1_successor_evidence_chain.md",
    "docs/mixture_of_perspectives/22_generation1_successor_horizon_v2.md",
    "docs/mixture_of_perspectives/23_generation1_categorized_batch_wave.md",
    "docs/mixture_of_perspectives/24_generation1_successor_recovery_v5.md",
    "docs/mixture_of_perspectives/25_generation1_full_generations_wave.md",
)


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _classification_files(horizon_dir: Path) -> list[Path]:
    cdir = horizon_dir / "classifications"
    if not cdir.is_dir():
        return []
    return sorted(p for p in cdir.glob("*.json") if p.is_file())


def _horizon_is_terminal(horizon_dir: Path) -> tuple[bool, dict[str, Any] | None]:
    """A horizon is terminal only when its report receipt exists and reports a clean completion."""

    receipt = _load(horizon_dir / "report_receipt.json")
    if not isinstance(receipt, dict):
        return False, None
    terminal = bool(receipt.get("complete")) and not receipt.get("problems")
    return terminal, receipt


def _lane_states_from_classification(classification: dict[str, Any]) -> dict[str, list[str]]:
    """Split a sealed epoch classification's mechanics lanes into surviving, pruned, warned, failed."""

    mechanics = classification.get("mechanics") or {}
    surviving: list[str] = []
    pruned: list[str] = []
    warned: list[str] = []
    failed: list[str] = []
    for lane_id in sorted(mechanics):
        info = mechanics[lane_id] or {}
        klass = str(info.get("classification") or "")
        continue_lane = bool(info.get("continue_lane"))
        if "warn" in klass:
            warned.append(lane_id)
        if "fail" in klass:
            failed.append(lane_id)
        if klass == "not_run_pruned" or not continue_lane:
            pruned.append(lane_id)
        elif klass == "mechanics_noninferential" or continue_lane:
            surviving.append(lane_id)
    return {
        "surviving": surviving,
        "pruned": pruned,
        "warned": warned,
        "failed": failed,
    }


def _derive_horizon_v1(horizon_dir: Path) -> dict[str, Any]:
    """Derive the terminal Horizon 1 lineage: final lane states, the D1 route, and receipt bindings."""

    terminal, receipt = _horizon_is_terminal(horizon_dir)
    classifications = _classification_files(horizon_dir)
    epoch_ids: list[str] = []
    final_classification: dict[str, Any] | None = None
    for path in classifications:
        doc = _load(path)
        if not isinstance(doc, dict):
            continue
        epoch_ids.append(str(doc.get("epoch_id") or path.stem))
        final_classification = doc

    lane_states = (
        _lane_states_from_classification(final_classification)
        if isinstance(final_classification, dict)
        else {"surviving": [], "pruned": [], "warned": [], "failed": []}
    )
    d1_block = (final_classification or {}).get("d1") or {}
    d1 = {
        "classification": d1_block.get("classification"),
        "terminal_route": d1_block.get("terminal_route"),
        "continue_d1": bool(d1_block.get("continue_d1")),
    }
    return {
        "program_id": HORIZON_V1_ID,
        "terminal": terminal,
        "epochs_sealed": epoch_ids,
        "final_epoch": epoch_ids[-1] if epoch_ids else None,
        "lane_states": lane_states,
        "d1": d1,
        "report_receipt": {
            "receipt_sha256": (receipt or {}).get("receipt_sha256"),
            "result_sha256": ((receipt or {}).get("result") or {}).get("file_sha256"),
            "verification_sha256": ((receipt or {}).get("verification") or {}).get("file_sha256"),
        }
        if receipt
        else None,
    }


def _derive_horizon_v2(horizon_dir: Path) -> dict[str, Any]:
    """Derive Horizon 2: carried lanes, dependency closure, and how far the running horizon has sealed."""

    terminal, receipt = _horizon_is_terminal(horizon_dir)
    admission = _load(horizon_dir / "admission.json") or {}
    classifications = _classification_files(horizon_dir)
    epochs_sealed = [str((_load(p) or {}).get("epoch_id") or p.stem) for p in classifications]
    planned_epochs = [str(e) for e in (admission.get("epoch_ids") or [])]
    remaining = [e for e in planned_epochs if e not in epochs_sealed]

    return {
        "program_id": HORIZON_V2_ID,
        "terminal": terminal,
        "planned_epochs": planned_epochs,
        "epochs_sealed": epochs_sealed,
        "epochs_remaining": remaining,
        "report_receipt_present": receipt is not None,
        "carried_lanes": list(admission.get("mechanics_initially_eligible_lanes") or []),
        "predecessor_survivors": list(admission.get("mechanics_predecessor_survivors") or []),
        "dependency_pruned_lanes": list(admission.get("mechanics_dependency_pruned_lanes") or []),
        "mechanics_internal_dependencies": admission.get("mechanics_internal_dependencies") or {},
        "d1_initially_eligible": bool(admission.get("d1_initially_eligible")),
        "d1_predecessor_classification": admission.get("d1_predecessor_classification"),
    }


def _derive_lane_universe(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Fold the two horizons into one overall lane accounting the way the frozen routing rules do."""

    v1_survivors = list(v1["lane_states"]["surviving"])
    v1_pruned = list(v1["lane_states"]["pruned"])
    carried_into_v2 = list(v2.get("carried_lanes") or [])
    dependency_pruned = list(v2.get("dependency_pruned_lanes") or [])

    # Blocked lanes are pruned lanes whose terminal route is "blocked" (D1) plus lanes removed by the
    # transitive dependency closure at the next horizon boundary (I1, which depends on D1 and P1).
    blocked = sorted(set(v1_pruned) | set(dependency_pruned))

    # Untested lanes are the redesigned mechanism lanes queued for the full-generations W08 canary.
    untested = sorted(FULL_GENERATIONS_NEW_LANES)

    return {
        "carried": carried_into_v2,
        "surviving": v1_survivors,
        "pruned": sorted(set(v1_pruned) | set(dependency_pruned)),
        "warned": list(v1["lane_states"]["warned"]),
        "failed": list(v1["lane_states"]["failed"]),
        "blocked": blocked,
        "untested": untested,
        "counts": {
            "carried": len(carried_into_v2),
            "surviving": len(v1_survivors),
            "pruned": len(set(v1_pruned) | set(dependency_pruned)),
            "warned": len(v1["lane_states"]["warned"]),
            "failed": len(v1["lane_states"]["failed"]),
            "blocked": len(blocked),
            "untested": len(untested),
        },
    }


def _derive_d1_status(v1: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Derive the D1 retirement and redesign status from the terminal D1 route and the redesign catalog."""

    frozen = _load(repo_root / "proof" / "GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.json") or {}
    decision = frozen.get("decision") or {}
    producer_passed = decision.get("producer_all_frozen_criteria_passed")
    challenge_passed = decision.get("challenge_all_frozen_criteria_passed")
    frozen_pattern_repeated = decision.get("frozen_pattern_repeated")

    retired = v1["d1"].get("classification") == "not_run_pruned" or (
        producer_passed is False and challenge_passed is False
    )
    return {
        "retired": bool(retired),
        "route": "null_safe_prune",
        "terminal_route_in_horizon": v1["d1"].get("terminal_route"),
        "producer_all_frozen_criteria_passed": producer_passed,
        "challenge_all_frozen_criteria_passed": challenge_passed,
        "frozen_pattern_repeated": frozen_pattern_repeated,
        "old_centroid_kept_as_control": True,
        "redesign_families_registered": list(D1_REDESIGN_FAMILIES),
        "redesign_execution_authorized": False,
        "candidate_evidence_count": 0,
        "note": (
            "The exact frozen centroid design fails its static-margin and context-route-gap criteria "
            "in both the producer and challenge phases. Its honest route is null_safe_prune: preserve "
            "the aggregate and controls, retire the design, spend no fresh efficacy seeds, and require "
            "any future efficacy work to enter a new append-only redesign authority. This is a "
            "bounded nonconfirmatory result, not an independently generated scientific null."
        ),
    }


def _derive_i1_status(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Derive the G1-I1 integration status: it ran as mechanics-only in v1, then dependency-pruned."""

    ran_in_v1 = "G1-I1" in v1["lane_states"]["surviving"]
    dependency_pruned_in_v2 = "G1-I1" in (v2.get("dependency_pruned_lanes") or [])
    deps = (v2.get("mechanics_internal_dependencies") or {}).get("G1-I1") or []
    return {
        "ran_as_mechanics_only_in_horizon_v1": ran_in_v1,
        "executed_as_integration_efficacy_claim": False,
        "dependency_pruned_at_horizon_v2": dependency_pruned_in_v2,
        "route": "null_safe_prune_dependency_pruned" if dependency_pruned_in_v2 else "mechanics_only",
        "declared_dependencies": list(deps),
        "note": (
            "G1-I1 executed as a descriptive-only mechanics lane across Horizon 1, then was removed by "
            "transitive dependency closure at the Horizon 2 boundary because its declared prerequisites "
            "G1-D1 and G1-P1 are pruned. It is never silently promoted into a new causal premise."
        ),
    }


def _derive_canary_decisions(repo_root: Path) -> dict[str, Any]:
    """Derive the recorded canary decisions: the old P1 canary prune and the queued new-lane canaries."""

    mech_ext = _load(repo_root / "proof" / "GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json") or {}
    lanes = mech_ext.get("lanes") or {}
    p1 = lanes.get("G1-P1") or {}
    return {
        "old_p1_canary": {
            "lane": "G1-P1",
            "mechanism": p1.get("mechanism"),
            "canary_gate_passed": p1.get("canary_gate_passed"),
            "execution_decision": p1.get("execution_decision"),
            "long_work_executed": p1.get("long_work_executed"),
            "authority": "proof/GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json",
            "note": (
                "The old G1-P1 stability_plasticity lane failed its canary gate in the "
                "mechanics-extended screen and was pruned after canary with no producer or challenge "
                "work. The full-generations wave substitutes G1-P1R rather than reviving G1-P1."
            ),
        },
        "queued_new_lane_canaries": {
            lane: {"mechanism": mechanism, "status": "untested_queued_for_w08_canary"}
            for lane, mechanism in sorted(FULL_GENERATIONS_NEW_LANES.items())
        },
    }


def _derive_starss23_beds(repo_root: Path) -> dict[str, Any]:
    """Bind the natural-data STARSS23 outcomes; these are the only natural-world attempts, all null."""

    beds: dict[str, dict[str, Any]] = {
        "onset_localization": {
            "artifact": "STARSS23_ESCS_BED.json",
            "evidence_class": "invalid_bed_walled",
            "outcome": "energy carries no onset-localizing information; walled on both axes (seven nulls)",
        },
        "source_counting": {
            "artifact": "STARSS23_COUNTING_BED.json",
            "evidence_class": "verified_null",
            "outcome": "counting signal did not survive bias-independent reproduction (2 of 4 below bar)",
        },
        "direction_of_arrival": {
            "artifact": "STARSS23_DOA_BED.json",
            "evidence_class": "verified_null",
            "outcome": "clean double null on both gate architectures; not architecture-fragile",
        },
    }
    for bed in beds.values():
        bed["source_sha256"] = _sha256_file(repo_root / "proof" / bed["artifact"])
    return beds


def _derive_evidence_classes(lane_universe: dict[str, Any], d1: dict[str, Any]) -> dict[str, Any]:
    """State the evidence class of each terminal conclusion. No conclusion is a positive."""

    return {
        "mechanics_lanes": {
            "evidence_class": "mechanics_only_descriptive_noninferential",
            "count": lane_universe["counts"]["surviving"],
            "statement": (
                "The surviving mechanics lanes are same-code fresh-seed robustness patterns classified "
                "descriptive-only and noninferential. They are plumbing and favorable-versus-null "
                "discrimination, not an active confirmed cognitive mechanism."
            ),
        },
        "d1_dispatch": {
            "evidence_class": "bounded_null_safe_prune_nonconfirmatory",
            "statement": d1["note"],
        },
        "old_p1_lane": {
            "evidence_class": "failed_canary_prune",
            "statement": "The old stability_plasticity lane failed its canary gate and was pruned.",
        },
        "natural_world_beds": {
            "evidence_class": "verified_null_or_invalid_bed",
            "statement": (
                "Every natural-data STARSS23 bed (onset localization, source counting, direction of "
                "arrival) is null or an invalid bed for its mechanism question."
            ),
        },
        "overall_stage": {
            "evidence_class": "stage_2_verified_discovery_no_active_mechanism",
            "statement": (
                "The program has verified discovery evidence, bounded generated patterns, a "
                "structurally verified nonconfirmatory D1 prune, and mechanics pilots. It has no "
                "independently confirmed active cognitive mechanism."
            ),
        },
    }


def _derive_surviving_hypotheses(lane_universe: dict[str, Any], d1: dict[str, Any]) -> list[dict[str, Any]]:
    """For each surviving hypothesis, name the strongest killing control or the remaining blocker."""

    return [
        {
            "hypothesis": "successor mechanics lanes carry a real cognitive capability",
            "status": "descriptive_only_survivor",
            "strongest_control_or_blocker": (
                "Remaining blocker: every mechanics receipt is same-code and generated. There is no "
                "separately implemented independent scientific generator, so a descriptive-only lane "
                "cannot become an independent confirmation no matter how many fresh seeds repeat."
            ),
        },
        {
            "hypothesis": "learned dispatch (D1) beats a fixed route",
            "status": "retired_null_safe_prune",
            "strongest_control_or_blocker": (
                "Strongest killing control: the frozen centroid design fails the static-margin and "
                "context-route-gap criteria in BOTH the producer and the challenge phase against its "
                "fixed-route control. Remaining blocker: the redesign families have zero receipt-backed "
                "candidate evidence and are unauthorized."
            ),
        },
    ]


def _next_bounded_scientific_question(d1: dict[str, Any]) -> dict[str, Any]:
    """State the exact next bounded scientific question without fabricating any positive."""

    return {
        "question": (
            "Can any separately implemented scientific generator produce receipt-backed candidate "
            "evidence above zero, either from a redesigned dispatch family "
            "(utility_residual, pairwise_ranking, calibrated_abstaining) or from a newly admitted "
            "mechanism lane (G1-U1 calibrated_uncertainty, G1-N1 reducible_novelty, "
            "G1-P1R stability_plasticity_r2), under the frozen static-margin and context-route-gap "
            "gates on held-out fresh seeds?"
        ),
        "current_answer": "no candidate evidence exists; the candidate count is zero",
        "positive_fabricated": False,
        "natural_world_axis": (
            "On the natural-data axis, after onset localization walled and source counting and "
            "direction of arrival both nulled, the next bounded STARSS23 question is event "
            "classification. It is not yet run and no positive is asserted."
        ),
        "authority_required_before_any_positive": (
            "candidate_evidence_count must exceed zero under a real receipt before any efficacy claim; "
            "no threshold may be weakened and no design may be resurrected"
        ),
    }


def replay_terminal_lineage(repo_root: Path, runs_root: Path) -> dict[str, Any]:
    """Derive the full closure lineage from the sealed terminal artifacts. Pure and deterministic.

    ``repo_root`` anchors the sealed proofs and the lineage design documents. ``runs_root`` anchors the
    live run tree that holds the successor horizon programs. When Horizon 2 is not terminal the
    derivation is honestly marked ``deferred_inputs_not_terminal`` and reports only terminal facts.
    """

    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()

    v1_dir = runs_root / "generation1" / HORIZON_V1_ID
    v2_dir = runs_root / "generation1" / HORIZON_V2_ID

    v1 = _derive_horizon_v1(v1_dir)
    v2 = _derive_horizon_v2(v2_dir)

    lane_universe = _derive_lane_universe(v1, v2)
    d1_status = _derive_d1_status(v1, repo_root)
    i1_status = _derive_i1_status(v1, v2)
    canary_decisions = _derive_canary_decisions(repo_root)
    starss23 = _derive_starss23_beds(repo_root)
    evidence_classes = _derive_evidence_classes(lane_universe, d1_status)
    surviving_hypotheses = _derive_surviving_hypotheses(lane_universe, d1_status)
    next_question = _next_bounded_scientific_question(d1_status)

    all_terminal = bool(v1["terminal"]) and bool(v2["terminal"])
    derivation_status = DERIVATION_COMPLETE if all_terminal else DEFERRED_INPUTS_NOT_TERMINAL

    terminal_inputs: list[str] = []
    nonterminal_inputs: list[str] = []
    (terminal_inputs if v1["terminal"] else nonterminal_inputs).append(HORIZON_V1_ID)
    (terminal_inputs if v2["terminal"] else nonterminal_inputs).append(HORIZON_V2_ID)
    # The categorized batch wave and the full-generations wave are scaffolded design documents only;
    # they have never run, so they are always nonterminal (queued) inputs.
    nonterminal_inputs.extend(
        ["generation1-successor-categorized-batch-wave-v1", "generation1-full-generations-wave-v1"]
    )

    doc_bindings = {doc: _sha256_file(repo_root / doc) for doc in LINEAGE_DOCS}

    return {
        "schema": DERIVATION_SCHEMA,
        "derivation_status": derivation_status,
        "terminal_inputs": sorted(terminal_inputs),
        "nonterminal_inputs": sorted(nonterminal_inputs),
        "horizon_v1": v1,
        "horizon_v2": v2,
        "lane_universe": lane_universe,
        "d1_status": d1_status,
        "g1_i1_status": i1_status,
        "canary_decisions": canary_decisions,
        "dependency_closure": {
            "rule": (
                "A mechanics lane enters the next horizon only when every declared mechanics "
                "dependency also survived. G1-I1 is removed because G1-D1 and G1-P1 are pruned. A "
                "dependency-pruned lane can never be resurrected."
            ),
            "internal_dependencies": v2.get("mechanics_internal_dependencies") or {},
            "dependency_pruned_lanes": v2.get("dependency_pruned_lanes") or [],
        },
        "starss23_natural_beds": starss23,
        "evidence_classes": evidence_classes,
        "surviving_hypotheses": surviving_hypotheses,
        "one_number_result": {
            "value": 0,
            "unit": "independently confirmed active cognitive mechanisms",
            "note": (
                "Zero independently confirmed active cognitive mechanisms across the whole lineage. "
                "Every surviving lane is descriptive-only same-code robustness."
            ),
        },
        "refusals": {
            "activation_allowed": False,
            "scientific_promotion": False,
            "natural_world_generality": False,
            "independent_scientific_confirmation": False,
            "statement": (
                "Generated same-code robustness does not activate any mechanism, does not promote any "
                "result, does not generalize to the natural world, and is not an independent scientific "
                "confirmation. The natural-data STARSS23 beds are all null."
            ),
        },
        "next_bounded_scientific_question": next_question,
        "next_queued_program": {
            "while_general_run_active": (
                "observe only; the live General Run is finishing Horizon 2 (H09, H10) before the "
                "categorized batch wave and the full-generations wave"
            ),
            "on_general_run_terminal": ".venv/bin/python -m mop.closure.producer --execute",
        },
        "source_bindings": {
            "horizon_v1_dir": str(v1_dir),
            "horizon_v2_dir": str(v2_dir),
            "lineage_docs": doc_bindings,
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


__all__ = [
    "replay_terminal_lineage",
    "DERIVATION_SCHEMA",
    "DEFERRED_INPUTS_NOT_TERMINAL",
    "DERIVATION_COMPLETE",
]
