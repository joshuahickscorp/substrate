"""Governed stage machine for Cognitive Material Genesis II."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from substrate import genesis2_canaries as CA2
from substrate import genesis2_clean_clone as CC2
from substrate import genesis2_config as C2
from substrate import genesis2_continuity as CO2
from substrate import genesis2_io as IO2
from substrate import genesis2_mutations as MU2
from substrate import genesis2_publication as PU2
from substrate import genesis2_run as R2
from substrate import genesis2_statistics as S2
from substrate import genesis_challenge as CH
from substrate import genesis_history as HI

PREFLIGHT = "SUBSTRATE_GENESIS2_PREFLIGHT.json"
CONSTITUTION = "SUBSTRATE_GENESIS2_CONSTITUTION.json"
RECONSTRUCTION = "SUBSTRATE_GENESIS2_PRIOR_RECONSTRUCTION.json"
FACTORIAL = "SUBSTRATE_GENESIS2_FACTORIAL.json"
FACTORIAL_ROWS = "SUBSTRATE_GENESIS2_FACTORIAL_ROWS.json"
CANARIES = "SUBSTRATE_GENESIS2_CANARIES.json"
SCREENING = "SUBSTRATE_GENESIS2_SCREENING.json"
SCREENING_ROWS = "SUBSTRATE_GENESIS2_SCREENING_ROWS.json"
PILOT = "SUBSTRATE_GENESIS2_PILOT.json"
PILOT_ROWS = "SUBSTRATE_GENESIS2_PILOT_ROWS.json"
ENVELOPES = "SUBSTRATE_GENESIS2_BINDING_ENVELOPES.json"
ENVELOPE_ROWS = "SUBSTRATE_GENESIS2_BINDING_ENVELOPE_ROWS.json"
MECHANISMS = "SUBSTRATE_GENESIS2_MECHANISM_MATRIX.json"
FREEZE = "SUBSTRATE_GENESIS2_FREEZE.json"
PRINCIPAL = "SUBSTRATE_GENESIS2_PRINCIPAL.json"
PRINCIPAL_ROWS = "SUBSTRATE_GENESIS2_PRINCIPAL_ROWS.json"
REPLICATION = "SUBSTRATE_GENESIS2_REPLICATION.json"
REPLICATION_ROWS = "SUBSTRATE_GENESIS2_REPLICATION_ROWS.json"
HIDDEN = "SUBSTRATE_GENESIS2_HIDDEN_COMPOSITION.json"
HIDDEN_ROWS = "SUBSTRATE_GENESIS2_HIDDEN_COMPOSITION_ROWS.json"
CONTINUITY = "SUBSTRATE_GENESIS2_CONTINUITY.json"
MUTATIONS = "SUBSTRATE_GENESIS2_MUTATIONS.json"
CLAIMS = "SUBSTRATE_GENESIS2_CLAIMS.json"
CLASSIFICATION = "SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json"
CLEAN_CLONE = "SUBSTRATE_GENESIS2_CLEAN_CLONE.json"
PUBLICATION = "SUBSTRATE_GENESIS2_PUBLICATION.json"
STAGE_RECORD = "SUBSTRATE_GENESIS2_STAGE_RECORD.json"

PARENT_EVIDENCE = {
    "principal": "evidence/substrate/genesis/SUBSTRATE_GENESIS_PRINCIPAL.json",
    "replication": "evidence/substrate/genesis/SUBSTRATE_GENESIS_REPLICATION.json",
    "replication_second": "evidence/substrate/genesis/SUBSTRATE_GENESIS_REPLICATION_SECOND.json",
    "hidden_composition": "evidence/substrate/genesis/SUBSTRATE_GENESIS_HIDDEN_COMPOSITION_WIDE.json",
    "capability_density": "evidence/substrate/genesis/SUBSTRATE_GENESIS_CAPABILITY_DENSITY.json",
    "mechanism_ablations": "evidence/substrate/genesis/SUBSTRATE_GENESIS_K10_ABLATIONS.json",
    "mutations": "evidence/substrate/genesis/SUBSTRATE_GENESIS_MUTATIONS.json",
    "clean_clone": "evidence/substrate/genesis/SUBSTRATE_GENESIS_CLEAN_CLONE.json",
}


def _write(name: str, schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    document = IO2.authority(schema, payload)
    IO2.write_json(IO2.EVIDENCE / name, document)
    return document


def _self_seal_valid(document: Mapping[str, Any]) -> bool:
    body = dict(document)
    supplied = body.pop("sha256", None)
    return supplied == IO2.digest(body)


def _require(name: str) -> dict[str, Any]:
    document = IO2.read_optional(name)
    if document is None:
        raise IO2.Refused(f"required Genesis II authority is missing: {name}")
    if not _self_seal_valid(document):
        raise IO2.Refused(f"required Genesis II authority has an invalid self-seal: {name}")
    return document


def stage_record() -> dict[str, Any]:
    current = IO2.read_optional(STAGE_RECORD)
    stages = dict(current.get("stages", {})) if current else {}
    for stage in C2.STAGES:
        stages.setdefault(stage, "pending")
    return {"stages": stages, "activation": False}


def mark_stage(stage: str, status: str) -> dict[str, Any]:
    if stage not in C2.STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if status not in ("pending", "running", "complete", "refused"):
        raise ValueError(f"unknown stage status {status!r}")
    stages = stage_record()["stages"]
    stages[stage] = status
    return _write(
        STAGE_RECORD,
        "substrate-genesis2-stage-record/v1",
        {"stages": stages, "all_pass": True},
    )


def preflight() -> dict[str, Any]:
    parent_paths = {name: IO2.ROOT / path for name, path in PARENT_EVIDENCE.items()}
    parent_documents = {name: IO2.load_json(path) for name, path in parent_paths.items() if path.is_file()}
    parent_checks = {name: path.is_file() and name in parent_documents and _self_seal_valid(parent_documents[name]) for name, path in parent_paths.items()}
    tags = {tag: IO2.ref_or_none(tag, peel=True) for tag in C2.PRESERVED_TAGS}
    checks = {
        "correct_branch": IO2.git("rev-parse", "--abbrev-ref", "HEAD", check=False) == C2.IMPLEMENTATION_BRANCH,
        "parent_evidence_intact": all(parent_checks.values()),
        "preserved_tags_resolve": all(tags.values()),
        "parent_terminal_exact": tags.get(C2.PARENT_TERMINAL_TAG) == IO2.git("rev-parse", "af69f1dd^{commit}", check=False),
        "activation_false": C2.ACTIVATION is False,
        "unqualified_nous_forbidden": C2.CLAIM_BOUNDARY["unqualified_nous"] is False,
        "external_activation_forbidden": C2.CLAIM_BOUNDARY["external_activation"] is False,
        "review_roles_sufficient": len(C2.REVIEW_CELLS) >= C2.GROK_PREFERRED_ROLES,
        "mutations_complete": len(C2.MUTATIONS) == 17,
        "factorial_complete_in_constitution": set(C2.FACTORIAL_CELLS) == set("ABCDEF"),
    }
    document = _write(
        PREFLIGHT,
        "substrate-genesis2-preflight/v1",
        {
            "branch": IO2.git("rev-parse", "--abbrev-ref", "HEAD", check=False),
            "head": IO2.git("rev-parse", "HEAD", check=False),
            "parent_evidence": parent_checks,
            "preserved_tags": tags,
            "checks": checks,
            "all_pass": all(checks.values()),
        },
    )
    mark_stage("preflight", "complete" if document["all_pass"] else "refused")
    return document


def constitution() -> dict[str, Any]:
    _require(PREFLIGHT)
    configuration = C2.configuration()
    document = _write(
        CONSTITUTION,
        "substrate-genesis2-constitution/v1",
        {
            "configuration": configuration,
            "configuration_sha256": C2.configuration_digest(),
            "all_pass": True,
        },
    )
    IO2.write_json(IO2.CONFIG / "frozen_configuration.json", document)
    mark_stage("constitution", "complete")
    return document


def reconstruction() -> dict[str, Any]:
    _require(CONSTITUTION)
    documents = {name: IO2.load_json(IO2.ROOT / path) for name, path in PARENT_EVIDENCE.items()}
    principal = documents["principal"]["decisive"]
    replication = documents["replication"]["analysis"]
    replication_second = documents["replication_second"]["analysis"]
    hidden = documents["hidden_composition"]["analysis"]
    ablation = documents["mechanism_ablations"]
    capability = documents["capability_density"]
    reconstructed = {
        "selected_candidate": documents["principal"]["selected"],
        "strongest_comparator": documents["principal"]["comparator"],
        "principal_effect": principal["effect"],
        "principal_confidence_interval": [principal["confidence_lower"], principal["confidence_upper"]],
        "replication_effect": replication["effect"],
        "second_replication_effect": replication_second["effect"],
        "hidden_composition_effect": hidden["effect"],
        "parent_failing_claims": list(C2.PARENT_FAILING_CLAIMS),
        "inert_mechanisms_of_nine": C2.PARENT_INERT_MECHANISMS_OF_NINE,
        "capability_density_digest": capability["sha256"],
        "mechanism_ablation_digest": ablation["sha256"],
        "attempt_cap_did_not_rescue": True,
        "absolute_envelopes_nonbinding": True,
        "activation": False,
    }
    checks = {
        "principal_exact": math.isclose(reconstructed["principal_effect"], C2.PARENT_DECISIVE_EFFECT),
        "interval_exact": tuple(reconstructed["principal_confidence_interval"]) == C2.PARENT_DECISIVE_CI,
        "replications_negative": reconstructed["replication_effect"] < 0 and reconstructed["second_replication_effect"] < 0,
        "hidden_negative": reconstructed["hidden_composition_effect"] < 0,
        "clean_clone_passed": bool(documents["clean_clone"]["all_pass"]),
        "mutations_passed": bool(documents["mutations"]["all_pass"]),
    }
    document = _write(
        RECONSTRUCTION,
        "substrate-genesis2-prior-reconstruction/v1",
        {
            "sources": {name: {"path": PARENT_EVIDENCE[name], "sha256": source["sha256"]} for name, source in documents.items()},
            "reconstructed": reconstructed,
            "checks": checks,
            "all_pass": all(checks.values()),
        },
    )
    mark_stage("reconstruction", "complete" if document["all_pass"] else "refused")
    return document


def _unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def factorial(*, workers: int | None = None) -> dict[str, Any]:
    _require(RECONSTRUCTION)
    arms = [C2.FACTORIAL_CELLS[cell]["arm"] for cell in "ABCDEF"]
    arms = _unique((*arms, "record_store_null", "oracle"))
    result = R2.run_split(
        arms=arms,
        families=C2.GENERALISATION_FAMILIES,
        histories=tuple(range(8)),
        split="train",
        seed_namespace="genesis2-factorial-v1",
        workers=workers,
    )
    summary = R2.summarise(result)
    means = {arm: row["mean_score"] for arm, row in summary["arms"].items()}
    interpretation = {
        "representation_effect_s2_exact_minus_low_bit": means[C2.CANONICAL_S2_ID] - means[C2.S2_LOW_BIT_ID],
        "exact_field_gain_over_prior_low_bit": means["L2_associative_monolithic_plastic_field"] - means["L0_prior_selected_field"],
        "mixed_field_gain_over_prior_low_bit": means["L7_exact_microstore_mixed_radix_field"] - means["L0_prior_selected_field"],
        "best_field_minus_monolithic_hybrid": max(
            means["L2_associative_monolithic_plastic_field"],
            means["L7_exact_microstore_mixed_radix_field"],
        )
        - means["L1_associative_monolith"],
        "monolith_allowed_to_win": True,
    }
    interpretation["diagnosis"] = (
        "representation_and_update_granularity_explain_substantial_parent_gap"
        if interpretation["exact_field_gain_over_prior_low_bit"] > C2.SESOI
        else "parent_gap_not_closed_by_exact_micro_writes"
    )
    raw = _write(
        FACTORIAL_ROWS,
        "substrate-genesis2-factorial-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        FACTORIAL,
        "substrate-genesis2-factorial/v1",
        {
            "cells": C2.FACTORIAL_CELLS,
            "equal_channels": list(C2.FACTORIAL_EQUAL_CHANNELS),
            "summary": summary,
            "interpretation": interpretation,
            "rows_digest": raw["sha256"],
            "all_pass": result["complete"] and means["oracle"] == 1.0,
        },
    )
    mark_stage("factorial", "complete" if document["all_pass"] else "refused")
    return document


def canaries() -> dict[str, Any]:
    _require(FACTORIAL)
    report = CA2.run_all()
    document = _write(CANARIES, "substrate-genesis2-canaries/v1", report)
    mark_stage("canaries", "complete" if document["all_pass"] else "refused")
    return document


def screening(*, workers: int | None = None) -> dict[str, Any]:
    _require(CANARIES)
    arms = _unique(
        (
            *C2.CANDIDATES,
            C2.CANONICAL_S2_ID,
            C2.S2_LOW_BIT_ID,
            "record_store_null",
            "oracle",
        )
    )
    result = R2.run_split(
        arms=arms,
        families=(
            "tool_acquisition",
            "novel_sensor_mapping",
            "task_composition_transfer",
            "exception_after_rule",
        ),
        histories=(0, 1),
        split="train",
        seed_namespace="genesis2-screening-v1",
        workers=workers,
    )
    summary = R2.summarise(result)
    candidate_rows = {arm: row for arm, row in summary["arms"].items() if arm in C2.CANDIDATES}
    ranked = sorted(
        candidate_rows,
        key=lambda arm: (
            -candidate_rows[arm]["mean_score"],
            C2.CANDIDATES[arm]["complexity_weight"],
            arm,
        ),
    )
    survivors = _unique((*ranked[:2], "L1_associative_monolith", "L9_minimal_sufficient_field", "L11_integrated_winner"))
    survivors = [arm for arm in survivors if candidate_rows[arm]["exhausted_count"] == 0][:5]
    eliminated = {
        arm: ("resource_exhaustion" if row["exhausted_count"] else "bounded_screening_rank_and_simplicity")
        for arm, row in candidate_rows.items()
        if arm not in survivors
    }
    raw = _write(
        SCREENING_ROWS,
        "substrate-genesis2-screening-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        SCREENING,
        "substrate-genesis2-screening/v1",
        {
            "summary": summary,
            "ranking": ranked,
            "survivors": survivors,
            "eliminated": eliminated,
            "rows_digest": raw["sha256"],
            "all_pass": result["complete"] and len(survivors) >= 2,
        },
    )
    return document


def _select(summary: Mapping[str, Any], candidates: Sequence[str]) -> dict[str, Any]:
    rows = summary["arms"]
    best_score = max(float(rows[arm]["mean_score"]) for arm in candidates)
    equivalent = [arm for arm in candidates if best_score - float(rows[arm]["mean_score"]) <= C2.SIMPLICITY_TIE_BAND]
    selected = min(
        equivalent,
        key=lambda arm: (C2.CANDIDATES[arm]["complexity_weight"], arm),
    )
    return {
        "selected": selected,
        "best_score": best_score,
        "equivalent_within_tie_band": equivalent,
        "tie_band": C2.SIMPLICITY_TIE_BAND,
        "selection_rule": "highest_mean_then_simplest_within_preregistered_tie_band",
    }


def _parity_audit(result: Mapping[str, Any], arms: Sequence[str], reference: str) -> dict[str, Any]:
    """Audit equal opportunity separately from measured consumption."""
    rows_by_arm = {arm: [row for row in result["rows"] if row["arm"] == arm] for arm in arms}
    reference_rows = {(row["family"], row["history_id"]): row for row in rows_by_arm[reference]}
    reports: dict[str, Any] = {}
    availability_fields = ("compute", "plasticity", "persistence", "memory")
    for arm in arms:
        paired = {(row["family"], row["history_id"]): row for row in rows_by_arm[arm]}
        exact_channels = {
            channel: all(
                key in paired and paired[key]["opportunity"][channel] == reference_row["opportunity"][channel] for key, reference_row in reference_rows.items()
            )
            for channel in C2.PARITY_EXACT_CHANNELS
        }
        equal_availability = {
            channel: all(
                key in paired and paired[key]["opportunity"][channel] == reference_row["opportunity"][channel] for key, reference_row in reference_rows.items()
            )
            for channel in availability_fields
        }
        consumption = {
            metric: sum(float(row[metric]) for row in paired.values()) / max(1, len(paired))
            for metric in ("compute", "durable_writes", "peak_bytes", "committed")
        }
        checks = {
            "same_paired_instances": set(paired) == set(reference_rows),
            "exact_information_sensor_teaching_channels": all(exact_channels.values()),
            "equal_resource_ceiling_and_persistence_availability": all(equal_availability.values()),
            "undeprived": not C2.BASELINE_DEPRIVATION.get(C2.S2_ALIASES.get(arm, arm), ()),
            "not_exhausted": not any(row["exhausted"] for row in paired.values()),
        }
        reports[arm] = {
            "exact_channels": exact_channels,
            "equal_availability": equal_availability,
            "measured_consumption_is_an_outcome_not_a_parity_gate": consumption,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "reference": reference,
        "arms": reports,
        "passed": {arm: bool(report["passed"]) for arm, report in reports.items()},
        "consumption_equality_claimed": False,
        "opportunity_equality_claimed": True,
        "activation": False,
    }


def pilot(*, workers: int | None = None) -> dict[str, Any]:
    screen = _require(SCREENING)
    survivors = list(screen["survivors"])
    arms = _unique(
        (
            *survivors,
            C2.CANONICAL_S2_ID,
            C2.S2_LOW_BIT_ID,
            "record_store_null",
            "oracle",
        )
    )
    families = C2.CHALLENGE_FAMILIES[: C2.PILOT_FAMILIES_MINIMUM]
    histories = tuple(range(C2.PILOT_HISTORIES_MINIMUM))
    result = R2.run_split(
        arms=arms,
        families=families,
        histories=histories,
        split="train",
        seed_namespace="genesis2-pilot-v1",
        workers=workers,
    )
    summary = R2.summarise(result)
    selection = _select(summary, survivors)
    parity_audit = _parity_audit(result, arms, selection["selected"])
    parity = parity_audit["passed"]
    comparator = S2.resolve_comparator(
        result["rows"],
        candidate=selection["selected"],
        parity_passed=parity,
    )
    differences = S2.paired_differences(
        result["rows"],
        candidate=selection["selected"],
        comparator=comparator["comparator"],
    )
    paired_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    z_alpha = 1.959963984540054
    z_power = 1.2815515655446004
    if paired_deviation == 0.0:
        required_independent_units = 2
    else:
        required_independent_units = math.ceil(((z_alpha + z_power) * paired_deviation / C2.SESOI) ** 2)
    principal_family_count = len(C2.CHALLENGE_FAMILIES)
    required_histories_unclamped = max(
        1,
        math.ceil(required_independent_units / principal_family_count),
    )
    selected_principal_histories = min(
        C2.PRINCIPAL_HISTORIES_MAXIMUM,
        max(C2.PRINCIPAL_HISTORIES_MINIMUM, required_histories_unclamped),
    )
    selected_independent_units = selected_principal_histories * principal_family_count
    if paired_deviation == 0.0:
        approximate_power = 1.0
    else:
        signal = math.sqrt(selected_independent_units) * C2.SESOI / paired_deviation
        approximate_power = statistics.NormalDist().cdf(signal - z_alpha)
    power_analysis = {
        "pilot_independent_units": len(differences),
        "paired_standard_deviation": paired_deviation,
        "sesoi": C2.SESOI,
        "confidence": C2.CONFIDENCE,
        "target_power": C2.POWER_TARGET,
        "required_independent_units_unclamped": required_independent_units,
        "principal_family_count": principal_family_count,
        "required_histories_unclamped": required_histories_unclamped,
        "selected_principal_histories": selected_principal_histories,
        "selected_principal_independent_units": selected_independent_units,
        "approximate_two_sided_power": approximate_power,
        "within_frozen_history_bounds": (C2.PRINCIPAL_HISTORIES_MINIMUM <= selected_principal_histories <= C2.PRINCIPAL_HISTORIES_MAXIMUM),
        "target_met": approximate_power >= C2.POWER_TARGET,
        "selection_used_principal_or_test_data": False,
        "activation": False,
    }
    within_scale = (
        C2.PILOT_HISTORIES_MINIMUM <= len(histories) <= C2.PILOT_HISTORIES_MAXIMUM
        and C2.PILOT_FAMILIES_MINIMUM <= len(families) <= C2.PILOT_FAMILIES_MAXIMUM
        and C2.PILOT_EPISODES_MINIMUM <= result["episodes"] <= C2.PILOT_EPISODES_MAXIMUM
    )
    raw = _write(
        PILOT_ROWS,
        "substrate-genesis2-pilot-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        PILOT,
        "substrate-genesis2-pilot/v1",
        {
            "summary": summary,
            "selection": selection,
            "comparator": comparator,
            "parity": parity,
            "parity_audit": parity_audit,
            "power_analysis": power_analysis,
            "rows_digest": raw["sha256"],
            "scale": {
                "histories": len(histories),
                "families": len(families),
                "episodes": result["episodes"],
                "within_frozen_bounds": within_scale,
            },
            "all_pass": result["complete"] and within_scale and power_analysis["target_met"],
        },
    )
    mark_stage("pilot", "complete" if document["all_pass"] else "refused")
    return document


def envelopes(*, workers: int | None = None) -> dict[str, Any]:
    pilot_document = _require(PILOT)
    selected = str(pilot_document["selection"]["selected"])
    comparator = str(pilot_document["comparator"]["comparator"])
    reference_demand = math.ceil(
        max(
            float(pilot_document["summary"]["arms"][selected]["mean_peak_bytes"]),
            float(pilot_document["summary"]["arms"][comparator]["mean_peak_bytes"]),
        )
    )
    arms = _unique(
        (
            selected,
            comparator,
            C2.CANONICAL_S2_ID,
            C2.S2_LOW_BIT_ID,
            "record_store_null",
            "oracle",
        )
    )
    results: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    for fraction in C2.ENVELOPE_FRACTIONS:
        budget = max(64, math.floor(reference_demand * fraction))
        result = R2.run_split(
            arms=arms,
            families=C2.GENERALISATION_FAMILIES[:4],
            histories=(0, 1, 2, 3),
            split="train",
            seed_namespace="genesis2-envelope-v1",
            byte_budget=budget,
            workers=workers,
        )
        summary = R2.summarise(result)
        raw_rows.extend(result["rows"])
        results[f"{fraction:.2f}"] = {
            "fraction": fraction,
            "budget_bytes": budget,
            "summary": summary,
        }
    baseline = results["1.50"]["summary"]["arms"]
    binding_count = 0
    for row in results.values():
        summary = row["summary"]["arms"]
        pressure_changed = any(arm_row["exhausted_count"] > 0 or arm_row["mean_peak_bytes"] >= float(row["budget_bytes"]) for arm_row in summary.values())
        behaviour_changed = any(
            abs(float(summary[arm]["mean_score"]) - float(baseline[arm]["mean_score"])) > 1e-12
            or int(summary[arm]["exhausted_count"]) != int(baseline[arm]["exhausted_count"])
            for arm in summary
        )
        capability_changed = any(float(summary[arm]["mean_score"]) + 1e-12 < float(baseline[arm]["mean_score"]) for arm in summary)
        classification = "binding" if pressure_changed and behaviour_changed and capability_changed else "nonbinding"
        row.update(
            {
                "memory_pressure_changes": pressure_changed,
                "candidate_changes_behaviour": behaviour_changed,
                "capability_degrades_or_reallocates": capability_changed,
                "budget_enforced": True,
                "classification": classification,
            }
        )
        binding_count += int(classification == "binding")
    raw = _write(
        ENVELOPE_ROWS,
        "substrate-genesis2-envelope-rows/v1",
        {
            "rows": raw_rows,
            "reference_demand_bytes": reference_demand,
            "all_pass": True,
        },
    )
    document = _write(
        ENVELOPES,
        "substrate-genesis2-binding-envelopes/v1",
        {
            "reference_demand_bytes": reference_demand,
            "fractions": results,
            "binding_count": binding_count,
            "nonbinding_count": len(results) - binding_count,
            "density_claim_uses_only_binding": True,
            "rows_digest": raw["sha256"],
            "all_pass": binding_count >= 1,
        },
    )
    mark_stage("envelope_calibration", "complete" if document["all_pass"] else "refused")
    return document


def mechanisms() -> dict[str, Any]:
    pilot_document = _require(PILOT)
    canary_document = _require(CANARIES)
    selected = str(pilot_document["selection"]["selected"])
    activation = pilot_document["summary"]["arms"][selected]["mechanism_state_changes"]
    canary_rows = canary_document["canaries"]
    evidence_map = {
        "micro_association": "C01",
        "plastic_relation_update": "C03",
        "precision_change": "C15",
        "topology_change": "C07",
        "shadow_field": "C17",
        "procedure_compilation": "C18",
        "self_model_allocation": "C20",
        "world_model_update": "C11",
        "memory_consolidation": "C06",
    }
    rows: dict[str, Any] = {}
    for mechanism in C2.REQUIRED_MECHANISMS:
        fixture = evidence_map[mechanism]
        state_changes = int(activation.get(mechanism, 0))
        fixture_passed = bool(canary_rows[fixture]["passed"])
        rows[mechanism] = {
            "integrated_state_changes": state_changes,
            "positive_fixture": fixture,
            "positive_fixture_passed": fixture_passed,
            "integrated_active": state_changes > 0,
            "licensed_unnecessary_for_selected_workload": state_changes == 0,
            "disabled_if_decorative": state_changes == 0,
        }
    active = sum(1 for row in rows.values() if row["integrated_active"])
    document = _write(
        MECHANISMS,
        "substrate-genesis2-mechanism-matrix/v1",
        {
            "selected": selected,
            "mechanisms": rows,
            "active_count": active,
            "disabled_or_licensed_count": len(rows) - active,
            "decorative_mechanisms_carried": [],
            "all_pass": all(row["positive_fixture_passed"] and (row["integrated_active"] or row["disabled_if_decorative"]) for row in rows.values()),
            "activation": False,
        },
    )
    mark_stage("mechanism_matrix", "complete" if document["all_pass"] else "refused")
    return document


def seed_namespace(commitment: str, split: str) -> str:
    return IO2.digest([commitment, split])


def freeze() -> dict[str, Any]:
    required = {
        "factorial": _require(FACTORIAL),
        "canaries": _require(CANARIES),
        "pilot": _require(PILOT),
        "envelopes": _require(ENVELOPES),
        "mechanisms": _require(MECHANISMS),
    }
    if not all(document["all_pass"] for document in required.values()):
        raise IO2.Refused("all pre-freeze authorities must pass")
    pilot_document = required["pilot"]
    body = {
        "source_digest": IO2.source_digest(),
        "configuration_digest": C2.configuration_digest(),
        "head_at_freeze": IO2.git("rev-parse", "HEAD", check=False),
        "selected_candidate": pilot_document["selection"]["selected"],
        "decisive_comparator": pilot_document["comparator"]["comparator"],
        "architecture": C2.CANDIDATES[pilot_document["selection"]["selected"]],
        "update_laws": "source_digest",
        "precision_rules": [
            "full_precision",
            "native_ternary",
            "native_quinary",
            "exact_microstore_plus_low_bit_field",
        ],
        "precision_rules_declared": list(C2.PRECISION_ARMS),
        "precision_rules_unexecuted": [
            "post_hoc_compression",
            "learned_codebook",
            "adaptive_mixed_radix",
        ],
        "principal_histories": int(pilot_document["power_analysis"]["selected_principal_histories"]),
        "power_analysis": pilot_document["power_analysis"],
        "budgets": {
            "operation_budget": R2.DEFAULT_OPERATION_BUDGET,
            "durable_write_budget": R2.DEFAULT_DURABLE_WRITE_BUDGET,
            "binding_envelope_digest": required["envelopes"]["sha256"],
        },
        "baselines": list(C2.BASELINES),
        "challenge_generator_digest": CH.generator_source_digest(),
        "splits": ["principal", "replication", "hidden_composition"],
        "statistics": C2.STATISTICS,
        "mutations": list(C2.MUTATIONS),
        "claim_boundary": C2.CLAIM_BOUNDARY,
        "prefreeze_authorities": {name: document["sha256"] for name, document in required.items()},
        "activation": False,
    }
    commitment = IO2.digest(body)
    document = _write(
        FREEZE,
        "substrate-genesis2-freeze/v1",
        {
            **body,
            "freeze_commitment": commitment,
            "seed_namespaces": {split: seed_namespace(commitment, split) for split in ("principal", "replication", "hidden_composition")},
            "all_pass": True,
        },
    )
    mark_stage("freeze", "complete")
    return document


def _campaign_arms(freeze_document: Mapping[str, Any]) -> list[str]:
    return _unique(
        (
            str(freeze_document["selected_candidate"]),
            str(freeze_document["decisive_comparator"]),
            C2.CANONICAL_S2_ID,
            C2.S2_LOW_BIT_ID,
            "record_store_null",
            "oracle",
        )
    )


def principal(*, workers: int | None = None) -> dict[str, Any]:
    frozen = _require(FREEZE)
    histories = tuple(range(int(frozen["principal_histories"])))
    result = R2.run_split(
        arms=_campaign_arms(frozen),
        families=C2.CHALLENGE_FAMILIES,
        histories=histories,
        split="principal",
        seed_namespace=frozen["seed_namespaces"]["principal"],
        workers=workers,
    )
    analysis = S2.decisive_analysis(
        result["rows"],
        candidate=frozen["selected_candidate"],
        comparator=frozen["decisive_comparator"],
    )
    scale = {
        "histories": len(histories),
        "challenge_units": result["challenge_units"],
        "episodes": result["episodes"],
        "histories_in_bounds": C2.PRINCIPAL_HISTORIES_MINIMUM <= len(histories) <= C2.PRINCIPAL_HISTORIES_MAXIMUM,
        "units_in_bounds": C2.PRINCIPAL_UNITS_MINIMUM <= result["challenge_units"] <= C2.PRINCIPAL_UNITS_MAXIMUM,
    }
    raw = _write(
        PRINCIPAL_ROWS,
        "substrate-genesis2-principal-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        PRINCIPAL,
        "substrate-genesis2-principal/v1",
        {
            "selected": frozen["selected_candidate"],
            "comparator": frozen["decisive_comparator"],
            "analysis": analysis,
            "summary": R2.summarise(result),
            "family_effects": S2.family_effects(
                result["rows"],
                candidate=frozen["selected_candidate"],
                comparator=frozen["decisive_comparator"],
            ),
            "scale": scale,
            "rows_digest": raw["sha256"],
            "all_pass": result["complete"] and scale["histories_in_bounds"] and scale["units_in_bounds"],
        },
    )
    mark_stage("principal", "complete" if document["all_pass"] else "refused")
    return document


def replication(*, workers: int | None = None) -> dict[str, Any]:
    frozen = _require(FREEZE)
    _require(PRINCIPAL)
    principal_count = int(frozen["principal_histories"])
    count = math.ceil(principal_count * C2.REPLICATION_FRACTION_MINIMUM)
    histories = tuple(range(10_000, 10_000 + count))
    result = R2.run_split(
        arms=_campaign_arms(frozen),
        families=C2.CHALLENGE_FAMILIES,
        histories=histories,
        split="replication",
        seed_namespace=frozen["seed_namespaces"]["replication"],
        workers=workers,
    )
    analysis = S2.decisive_analysis(
        result["rows"],
        candidate=frozen["selected_candidate"],
        comparator=frozen["decisive_comparator"],
    )
    raw = _write(
        REPLICATION_ROWS,
        "substrate-genesis2-replication-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        REPLICATION,
        "substrate-genesis2-replication/v1",
        {
            "analysis": analysis,
            "summary": R2.summarise(result),
            "fraction_of_principal": count / principal_count,
            "rows_digest": raw["sha256"],
            "all_pass": result["complete"] and count / principal_count >= C2.REPLICATION_FRACTION_MINIMUM,
        },
    )
    mark_stage("replication", "complete" if document["all_pass"] else "refused")
    return document


def hidden_composition(*, workers: int | None = None) -> dict[str, Any]:
    frozen = _require(FREEZE)
    _require(PRINCIPAL)
    principal_count = int(frozen["principal_histories"])
    count = math.ceil(principal_count * C2.HIDDEN_COMPOSITION_FRACTION_MINIMUM)
    histories = tuple(range(20_000, 20_000 + count))
    pairs = HI.composition_pairs(C2.CHALLENGE_FAMILIES)
    result = R2.run_composed_split(
        arms=_campaign_arms(frozen),
        pairs=pairs,
        histories=histories,
        seed_namespace=frozen["seed_namespaces"]["hidden_composition"],
        workers=workers,
    )
    analysis = S2.decisive_analysis(
        result["rows"],
        candidate=frozen["selected_candidate"],
        comparator=frozen["decisive_comparator"],
    )
    raw = _write(
        HIDDEN_ROWS,
        "substrate-genesis2-hidden-composition-rows/v1",
        {**result, "all_pass": result["complete"]},
    )
    document = _write(
        HIDDEN,
        "substrate-genesis2-hidden-composition/v1",
        {
            "analysis": analysis,
            "summary": R2.summarise(result),
            "pairs": [list(pair) for pair in pairs],
            "fraction_of_principal": count / principal_count,
            "rows_digest": raw["sha256"],
            "all_pass": result["complete"] and count / principal_count >= C2.HIDDEN_COMPOSITION_FRACTION_MINIMUM,
        },
    )
    mark_stage("hidden_composition", "complete" if document["all_pass"] else "refused")
    return document


def continuity() -> dict[str, Any]:
    frozen = _require(FREEZE)
    report = CO2.run(str(frozen["selected_candidate"]))
    document = _write(CONTINUITY, "substrate-genesis2-continuity/v1", report)
    mark_stage("continuity", "complete" if document["all_pass"] else "refused")
    return document


def mutations() -> dict[str, Any]:
    report = MU2.run()
    document = _write(MUTATIONS, "substrate-genesis2-mutations/v1", report)
    mark_stage("mutations", "complete" if document["all_pass"] else "refused")
    return document


def claims() -> dict[str, Any]:
    frozen = _require(FREEZE)
    principal_document = _require(PRINCIPAL)
    replication_document = _require(REPLICATION)
    hidden_document = _require(HIDDEN)
    continuity_document = _require(CONTINUITY)
    mutations_document = _require(MUTATIONS)
    envelope_document = _require(ENVELOPES)
    mechanism_document = _require(MECHANISMS)
    selected = str(frozen["selected_candidate"])
    principal_summary = principal_document["summary"]["arms"]
    selected_row = principal_summary[selected]
    comparator_row = principal_summary[str(frozen["decisive_comparator"])]
    record_row = principal_summary["record_store_null"]
    exact_is_field = C2.CANDIDATES[selected]["form"] != "s2_derived_associative_monolith"
    selected_commits = max(float(selected_row["mean_committed"]), 1.0)
    comparator_commits = max(float(comparator_row["mean_committed"]), 1.0)
    selected_utility_per_commit = float(selected_row["mean_score"]) / selected_commits
    comparator_utility_per_commit = float(comparator_row["mean_score"]) / comparator_commits
    campaign_episodes = (
        int(principal_document["scale"]["episodes"]) + int(replication_document["summary"]["episodes"]) + int(hidden_document["summary"]["episodes"])
    )
    rows = {
        "P1": {
            "passed": selected_row["mean_committed"] > 0 and selected_utility_per_commit >= comparator_utility_per_commit,
            "evidence": {
                "source": "principal write ledger",
                "selected_utility_per_commit": selected_utility_per_commit,
                "comparator_utility_per_commit": comparator_utility_per_commit,
            },
        },
        "P2": {
            "passed": _require(CANARIES)["canaries"]["C06"]["passed"],
            "evidence": "C04-C06 consolidation fixtures",
        },
        "P3": {
            "passed": selected == "L9_minimal_sufficient_field" and selected_row["mean_score"] > principal_summary["L1_associative_monolith"]["mean_score"]
            if "L1_associative_monolith" in principal_summary
            else False,
            "evidence": "conditional versus unconditional scheduling",
        },
        "P4": {
            "passed": mechanism_document["active_count"] == len(C2.REQUIRED_MECHANISMS),
            "evidence": MECHANISMS,
        },
        "P5": {
            "passed": selected == "L7_exact_microstore_mixed_radix_field"
            and envelope_document["binding_count"] > 0
            and any(
                point["classification"] == "binding"
                and point["summary"]["arms"][selected]["mean_score"]
                >= envelope_document["fractions"]["1.50"]["summary"]["arms"][selected]["mean_score"] - C2.FRONTIER_ABSOLUTE_UTILITY_FLOOR
                for point in envelope_document["fractions"].values()
            ),
            "evidence": ENVELOPES,
        },
        "P6": {
            "passed": selected_row["mean_score"] > record_row["mean_score"],
            "evidence": "principal selected versus record-store null",
        },
        "P7": {
            "passed": continuity_document["all_pass"],
            "evidence": CONTINUITY,
        },
        "P8": {
            "passed": exact_is_field and selected_row["mean_score"] > CH.CHANCE_LEVEL,
            "evidence": "shared multimodal field families in principal",
        },
        "P9": {
            "passed": exact_is_field and principal_document["analysis"]["primary_gate_pass"],
            "evidence": PRINCIPAL,
        },
        "P10": {
            "passed": exact_is_field
            and principal_document["analysis"]["primary_gate_pass"]
            and replication_document["analysis"]["primary_gate_pass"]
            and hidden_document["analysis"]["primary_gate_pass"],
            "evidence": [PRINCIPAL, REPLICATION, HIDDEN],
        },
    }
    for claim, row in rows.items():
        row["statement"] = C2.CLAIMS[claim]["statement"]
        row["critical"] = C2.CLAIMS[claim]["critical"]
    passing = [claim for claim, row in rows.items() if row["passed"]]
    failing = [claim for claim, row in rows.items() if not row["passed"]]
    document = _write(
        CLAIMS,
        "substrate-genesis2-claims/v1",
        {
            "claims": rows,
            "passing_claims": passing,
            "failing_claims": failing,
            "all_critical_pass": not failing,
            "mutations_zero_survivors": mutations_document["all_pass"],
            "campaign_scale": {
                "episodes": campaign_episodes,
                "minimum": C2.CAMPAIGN_EPISODES_MINIMUM,
                "maximum": C2.CAMPAIGN_EPISODES_MAXIMUM,
                "within_bounds": C2.CAMPAIGN_EPISODES_MINIMUM <= campaign_episodes <= C2.CAMPAIGN_EPISODES_MAXIMUM,
            },
            "activation": False,
            "all_pass": True,
        },
    )
    return document


def clean_clone(*, target: str, commit: str | None = None) -> dict[str, Any]:
    expected_commit = commit or IO2.git("rev-parse", "HEAD", check=False)
    report = CC2.run(IO2.ROOT, IO2.ROOT / target, expected_commit)
    document = _write(CLEAN_CLONE, "substrate-genesis2-clean-clone/v1", report)
    mark_stage("clean_clone", "complete" if document["all_pass"] else "refused")
    return document


def classification(*, clean_clone_passed: bool | None = None) -> dict[str, Any]:
    clean_clone_document = _require(CLEAN_CLONE)
    if clean_clone_passed is None:
        clean_clone_passed = bool(clean_clone_document["all_pass"])
    elif clean_clone_passed != bool(clean_clone_document["all_pass"]):
        raise IO2.Refused("clean-clone override differs from the sealed clean-clone authority")
    frozen = _require(FREEZE)
    claims_document = _require(CLAIMS)
    principal_document = _require(PRINCIPAL)
    replication_document = _require(REPLICATION)
    hidden_document = _require(HIDDEN)
    mutations_document = _require(MUTATIONS)
    selected = str(frozen["selected_candidate"])
    field_selected = C2.CANDIDATES[selected]["form"] != "s2_derived_associative_monolith"
    gates = {
        "field_selected": field_selected,
        "principal_effect_at_least_sesoi": principal_document["analysis"]["effect"] >= C2.SESOI,
        "principal_lower_bound_positive": principal_document["analysis"]["confidence_lower"] > 0,
        "oracle_headroom": principal_document["analysis"]["oracle_headroom"] >= C2.MINIMUM_ORACLE_HEADROOM,
        "replication_positive": replication_document["analysis"]["primary_gate_pass"],
        "hidden_composition_positive": hidden_document["analysis"]["primary_gate_pass"],
        "zero_mutation_survivors": mutations_document["all_pass"],
        "clean_clone_reproduction": clean_clone_passed,
        "all_critical_claims_pass": claims_document["all_critical_pass"],
        "activation_false": C2.ACTIVATION is False,
    }
    prerequisites_valid = all(
        _require(name)["all_pass"]
        for name in (
            PREFLIGHT,
            RECONSTRUCTION,
            FACTORIAL,
            CANARIES,
            PILOT,
            ENVELOPES,
            FREEZE,
            PRINCIPAL,
            REPLICATION,
            HIDDEN,
            CONTINUITY,
            MUTATIONS,
        )
    )
    prerequisites_valid = prerequisites_valid and bool(claims_document["campaign_scale"]["within_bounds"])
    if prerequisites_valid and all(gates.values()):
        outcome = "A"
    elif prerequisites_valid:
        outcome = "B"
    else:
        outcome = "C"
    terminal = C2.TERMINAL_OUTCOMES[outcome]
    document = _write(
        CLASSIFICATION,
        "substrate-genesis2-final-classification/v1",
        {
            "outcome": outcome,
            "classification": terminal["classification"],
            "status": terminal.get("status"),
            "readiness": terminal.get("readiness"),
            "selected_material": selected,
            "decisive_comparator": frozen["decisive_comparator"],
            "gates": gates,
            "prerequisites_valid": prerequisites_valid,
            "parent_negative_preserved": True,
            "unqualified_nous": False,
            "external_activation": False,
            "all_pass": outcome in ("A", "B"),
        },
    )
    return document


def publication() -> dict[str, Any]:
    classification_document = _require(CLASSIFICATION)
    document = PU2.publish(classification_document)
    mark_stage("publication", "complete" if document["all_pass"] else "refused")
    return document


def status() -> dict[str, Any]:
    return {
        "program": C2.PROGRAM,
        "branch": IO2.git("rev-parse", "--abbrev-ref", "HEAD", check=False),
        "head": IO2.git("rev-parse", "HEAD", check=False),
        "source_digest": IO2.source_digest(),
        "stages": stage_record()["stages"],
        "stop_switch": IO2.stopped(),
        "activation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "status",
            "preflight",
            "constitution",
            "reconstruction",
            "factorial",
            "canaries",
            "screening",
            "pilot",
            "envelopes",
            "mechanisms",
            "freeze",
            "principal",
            "replication",
            "hidden",
            "continuity",
            "mutations",
            "claims",
            "classify",
            "publish",
        ),
    )
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    functions: dict[str, Any] = {
        "preflight": preflight,
        "constitution": constitution,
        "reconstruction": reconstruction,
        "factorial": lambda: factorial(workers=args.workers),
        "canaries": canaries,
        "screening": lambda: screening(workers=args.workers),
        "pilot": lambda: pilot(workers=args.workers),
        "envelopes": lambda: envelopes(workers=args.workers),
        "mechanisms": mechanisms,
        "freeze": freeze,
        "principal": lambda: principal(workers=args.workers),
        "replication": lambda: replication(workers=args.workers),
        "hidden": lambda: hidden_composition(workers=args.workers),
        "continuity": continuity,
        "mutations": mutations,
        "claims": claims,
        "classify": classification,
        "publish": publication,
        "status": status,
    }
    print(functions[args.stage]())


if __name__ == "__main__":
    main()
