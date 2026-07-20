#!/usr/bin/env python
"""Independently replay and attack the F22, F28, and F50 to F58 ecology receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from mop.config import REPO_ROOT
from mop.environments.ecology_battery import (
    BATTERY_UNIT_SCHEMA,
    EXPERIMENT_IDS,
    run_ecology_battery_seed,
)
from mop.environments.ecology_scaffold import verify_ecology_fixture
from mop.substrate.events import atomic_write_json, canonical_bytes

SCHEMA = "mop-ecology-scaffold-battery-run/v1"
VERIFIER_SCHEMA = "mop-ecology-scaffold-independent-verifier/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
PRIMARY_SEEDS = (11, 29, 47, 71, 89)
FRESH_SEEDS = (101, 131, 163)
IMPLEMENTATION_PATHS = (
    "registry/experiments.yaml",
    "src/mop/environments/ecology_scaffold.py",
    "src/mop/environments/ecology_battery.py",
    "scripts/run_ecology_scaffold_batteries.py",
    "scripts/verify_ecology_scaffold_batteries.py",
)
STRONGEST_CONTROLS = {
    "f22_active_form_acquisition": "uncertainty-acquisition",
    "f28_sensor_value_forecast": "post-hoc-value",
    "f50_curriculum_goldilocks_test": "random-task",
    "f51_safe_play_goal_babbling": "ungated-curiosity",
    "f52_quality_diverse_mode_ecology": "monolithic-model",
    "f53_joint_referent_establishment": "partner-policy-pattern-matching",
    "f54_communicative_repair": "larger-channel-matched-bits",
    "f55_selective_imitation": "behavioral-cloning",
    "f56_teaching_value": "uncertainty-selection",
    "f57_emergent_symbol_grounding": "direct-state",
    "f58_cultural_accumulation": "direct-imitation",
}
CALIBRATION_SPECS = {
    "f22_active_form_acquisition": (
        "untouched_downstream_accuracy",
        ("information-gain", "uncertainty-acquisition", "full-observation"),
        0.10,
        0.95,
        0.08,
    ),
    "f28_sensor_value_forecast": (
        "forecast_rank_correlation",
        ("expected-information-gain", "entropy-baseline", "historical-average"),
        -0.95,
        0.99,
        0.10,
    ),
    "f50_curriculum_goldilocks_test": (
        "downstream_transfer_per_sample",
        (
            "goldilocks-learning-progress",
            "random-task",
            "fixed-order",
            "easiest-first",
            "hardest-first",
        ),
        0.05,
        0.95,
        0.08,
    ),
    "f51_safe_play_goal_babbling": (
        "external_task_acceleration",
        ("guarded-goal-babbling", "random-exploration", "ungated-curiosity"),
        0.02,
        0.95,
        0.08,
    ),
    "f52_quality_diverse_mode_ecology": (
        "utility_per_maintenance_cost",
        ("quality-diverse-archive", "monolithic-model", "random-archive"),
        0.05,
        0.95,
        0.05,
    ),
    "f53_joint_referent_establishment": (
        "joint_referent_agreement",
        (
            "joint-referent-policy",
            "cue-blind-partner",
            "partner-policy-pattern-matching",
            "shared-label-oracle",
        ),
        0.05,
        0.95,
        0.08,
    ),
    "f54_communicative_repair": (
        "recovery_per_turn",
        ("clarification-repair", "no-repair-channel", "repetition", "larger-channel-matched-bits"),
        0.05,
        0.49,
        0.05,
    ),
    "f55_selective_imitation": (
        "causal_action_fidelity",
        ("selective-causal-imitation", "indiscriminate-imitation", "behavioral-cloning"),
        0.05,
        0.95,
        0.08,
    ),
    "f56_teaching_value": (
        "learner_gain_per_message",
        (
            "learner-progress-teacher",
            "equal-information",
            "random-demonstrations",
            "hard-example-selection",
            "uncertainty-selection",
        ),
        0.05,
        0.95,
        0.08,
    ),
    "f57_emergent_symbol_grounding": (
        "compositional_transfer",
        ("consequence-bound-code", "random-message", "fixed-message", "direct-state", "equal-bandwidth"),
        0.05,
        0.95,
        0.08,
    ),
    "f58_cultural_accumulation": (
        "final_external_task_performance",
        ("cumulative-convention", "generation-reset", "direct-imitation", "fresh-training"),
        0.05,
        0.95,
        0.08,
    ),
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_rows() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["experiments"] if row.get("id") in EXPERIMENT_IDS}


def _strictly_beats(
    arms: dict[str, dict[str, float]],
    primary: str,
    controls: tuple[str, ...],
    metrics: tuple[str, ...],
) -> bool:
    return all(
        arms[primary][metric] > arms[control][metric]
        for control in controls
        for metric in metrics
    )


def _independent_candidate(
    experiment_id: str, primary: str, arms: dict[str, dict[str, float]]
) -> bool:
    if experiment_id == "f22_active_form_acquisition":
        return (
            _strictly_beats(
                arms,
                primary,
                (
                    "random-acquisition",
                    "uncertainty-acquisition",
                    "saliency-acquisition",
                    "full-observation",
                ),
                ("capability_per_sensor_cost",),
            )
            and arms[primary]["acquisition_regret_vs_full_observation"] < 0.20
            and arms[primary]["cross_world_min_accuracy"] > 0.50
        )
    if experiment_id == "f28_sensor_value_forecast":
        return all(
            arms[primary]["forecast_rank_correlation"] > arms[control]["forecast_rank_correlation"]
            and arms[primary]["payoff_calibration_error"] < arms[control]["payoff_calibration_error"]
            for control in ("entropy-baseline", "historical-average", "post-hoc-value")
        )
    if experiment_id == "f50_curriculum_goldilocks_test":
        return _strictly_beats(
            arms,
            primary,
            ("random-task", "fixed-order", "easiest-first", "hardest-first"),
            ("downstream_transfer_per_sample",),
        )
    if experiment_id == "f51_safe_play_goal_babbling":
        return (
            _strictly_beats(
                arms,
                primary,
                ("random-exploration", "ungated-curiosity"),
                ("external_task_acceleration",),
            )
            and arms[primary]["guard_violation_rate"] == 0.0
            and arms["stop-rule-audit"]["audit_accuracy"] == 1.0
        )
    if experiment_id == "f52_quality_diverse_mode_ecology":
        return (
            _strictly_beats(
                arms,
                primary,
                ("monolithic-model", "random-archive"),
                ("archive_coverage", "utility_per_maintenance_cost"),
            )
            and arms[primary]["redundancy"] <= arms["random-archive"]["redundancy"]
        )
    if experiment_id == "f53_joint_referent_establishment":
        return (
            _strictly_beats(
                arms,
                primary,
                ("cue-blind-partner", "partner-policy-pattern-matching"),
                ("joint_referent_agreement",),
            )
            and arms[primary]["held_out_partner_transfer"] > 0.75
            and arms[primary]["joint_referent_agreement"]
            < arms["shared-label-oracle"]["joint_referent_agreement"]
        )
    if experiment_id == "f54_communicative_repair":
        return _strictly_beats(
            arms,
            primary,
            ("no-repair-channel", "repetition", "larger-channel-matched-bits"),
            ("recovery_per_bit", "recovery_per_turn"),
        )
    if experiment_id == "f55_selective_imitation":
        return _strictly_beats(
            arms,
            primary,
            ("indiscriminate-imitation", "behavioral-cloning"),
            ("causal_action_fidelity", "task_success_rate"),
        )
    if experiment_id == "f56_teaching_value":
        return _strictly_beats(
            arms,
            primary,
            (
                "equal-information",
                "random-demonstrations",
                "hard-example-selection",
                "uncertainty-selection",
            ),
            ("learner_gain_per_message", "held_out_learner_advantage"),
        )
    if experiment_id == "f57_emergent_symbol_grounding":
        return _strictly_beats(
            arms,
            primary,
            ("random-message", "fixed-message", "direct-state", "equal-bandwidth"),
            ("compositional_transfer", "causal_grounding_score", "code_stability"),
        )
    if experiment_id == "f58_cultural_accumulation":
        return _strictly_beats(
            arms,
            primary,
            ("generation-reset", "direct-imitation", "fresh-training"),
            (
                "generation_trend_external_tasks",
                "convention_retention",
                "final_external_task_performance",
            ),
        )
    raise ValueError(f"unsupported ecology experiment {experiment_id!r}")


def _aggregate(units: list[dict[str, Any]], experiment_id: str) -> dict[str, dict[str, float]]:
    first = units[0]["experiments"][experiment_id]
    return {
        arm: {
            metric: round(
                mean(unit["experiments"][experiment_id]["arms"][arm]["metrics"][metric] for unit in units),
                8,
            )
            for metric in arm_payload["metrics"]
        }
        for arm, arm_payload in first["arms"].items()
    }


def _calibration(
    experiment_id: str, aggregate: dict[str, dict[str, float]], primary: str
) -> dict[str, Any]:
    metric, arms, lower, upper, minimum_spread = CALIBRATION_SPECS[experiment_id]
    values = {arm: aggregate[arm][metric] for arm in arms}
    strongest = STRONGEST_CONTROLS[experiment_id]
    calibration_control = (
        "historical-average" if experiment_id == "f28_sensor_value_forecast" else strongest
    )
    primary_value = values[primary]
    strongest_value = aggregate[calibration_control][metric]
    spread = max(values.values()) - min(values.values())
    ceilinged_tie = primary_value == strongest_value and primary_value >= upper
    calibrated = (
        lower < primary_value < upper
        and lower < strongest_value < upper
        and spread >= minimum_spread
        and not ceilinged_tie
    )
    if experiment_id == "f28_sensor_value_forecast":
        calibrated = calibrated and aggregate["post-hoc-value"][metric] == 1.0
    return {
        "metric": metric,
        "values": values,
        "interior_bounds": [lower, upper],
        "minimum_spread": minimum_spread,
        "observed_spread": round(spread, 8),
        "primary_interior": lower < primary_value < upper,
        "strongest_control": strongest,
        "calibration_control": calibration_control,
        "strongest_control_interior": lower < strongest_value < upper,
        "structural_hindsight_or_oracle_ceiling_excluded": experiment_id
        == "f28_sensor_value_forecast",
        "ceilinged_tie": ceilinged_tie,
        "calibrated": calibrated,
    }


def _expected_experiment(
    experiment_id: str, units: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = [unit["experiments"][experiment_id] for unit in units]
    primary = rows[0]["primary_arm"]
    aggregate = _aggregate(units, experiment_id)
    calibration = _calibration(experiment_id, aggregate, primary)
    per_seed = [
        _independent_candidate(
            experiment_id,
            primary,
            {arm: payload["metrics"] for arm, payload in row["arms"].items()},
        )
        for row in rows
    ]
    aggregate_candidate = _independent_candidate(experiment_id, primary, aggregate)
    candidate = aggregate_candidate and all(per_seed) and calibration["calibrated"]
    return {
        "aggregate": aggregate,
        "per_seed": per_seed,
        "calibration": calibration,
        "candidate_result": "favorable-candidate" if candidate else "null",
    }


def _core_body(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "core_payload_sha256",
            "independent_verifier",
            "verified_outcomes",
            "payload_sha256",
        }
    }


def _base_errors(
    receipt: dict[str, Any], *, check_live_files: bool
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if receipt.get("schema") != SCHEMA:
        errors.append("ecology run schema drift")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        errors.append("ecology claim scope drift")
    if receipt.get("core_payload_sha256") != _sha(_core_body(receipt)):
        errors.append("ecology core payload digest drift")
    implementation = receipt.get("implementation", [])
    if tuple(row.get("path") for row in implementation) != IMPLEMENTATION_PATHS:
        errors.append("ecology implementation path set drift")
    for row in implementation:
        digest = row.get("sha256")
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            errors.append(f"ecology implementation digest malformed at {row.get('path')}")
            continue
        if check_live_files:
            path = REPO_ROOT / str(row.get("path"))
            if not path.is_file() or _sha_file(path) != digest:
                errors.append(f"ecology live implementation drift at {row.get('path')}")
    if tuple(receipt.get("primary_seeds", ())) != PRIMARY_SEEDS:
        errors.append("ecology primary seed declaration drift")
    units = receipt.get("units", [])
    if not isinstance(units, list) or len(units) != len(PRIMARY_SEEDS):
        return [*errors, "ecology independent unit count drift"], {"unit_count": 0}
    for expected_seed, unit in zip(PRIMARY_SEEDS, units, strict=True):
        if not isinstance(unit, dict):
            errors.append(f"ecology unit {expected_seed} is not a mapping")
            continue
        if unit.get("schema") != BATTERY_UNIT_SCHEMA or unit.get("seed") != expected_seed:
            errors.append(f"ecology unit schema or seed drift at {expected_seed}")
            continue
        if unit.get("claim_scope") != CLAIM_SCOPE:
            errors.append(f"ecology unit claim scope drift at {expected_seed}")
        body = {key: value for key, value in unit.items() if key != "unit_sha256"}
        if unit.get("unit_sha256") != _sha(body):
            errors.append(f"ecology unit digest drift at {expected_seed}")
        fixture_result = verify_ecology_fixture(unit.get("fixture", {}))
        if fixture_result.get("verified") is not True:
            errors.append(f"ecology scaffold fixture drift at {expected_seed}")
        rebuilt = run_ecology_battery_seed(expected_seed)
        if canonical_bytes(rebuilt) != canonical_bytes(unit):
            errors.append(f"ecology exact observation replay drift at {expected_seed}")
    experiments = receipt.get("experiments", {})
    if not isinstance(experiments, dict) or tuple(experiments) != EXPERIMENT_IDS:
        errors.append("ecology experiment coverage or order drift")
        experiments = {}
    registry = _registry_rows()
    candidate_ids: list[str] = []
    null_ids: list[str] = []
    for experiment_id in EXPERIMENT_IDS:
        row = experiments.get(experiment_id)
        if not isinstance(row, dict):
            errors.append(f"ecology receipt row missing for {experiment_id}")
            continue
        expected = _expected_experiment(experiment_id, units)
        registry_row = registry[experiment_id]
        binding = row.get("registry_binding", {})
        expected_binding = {
            "status": registry_row["status"],
            "null_hypothesis": registry_row["null_hypothesis"],
            "metrics": list(registry_row["metrics"]),
            "controls": list(registry_row["controls"]),
            "gates": list(registry_row["gates"]),
            "row_sha256": _sha(registry_row),
        }
        if binding != expected_binding:
            errors.append(f"ecology registry binding drift for {experiment_id}")
        first = units[0]["experiments"][experiment_id]
        if row.get("primary_arm") != first["primary_arm"]:
            errors.append(f"ecology primary arm drift for {experiment_id}")
        if row.get("declared_controls") != first["controls"]:
            errors.append(f"ecology declared control drift for {experiment_id}")
        control_evidence = row.get("control_evidence", {})
        if set(control_evidence) != set(registry_row["controls"]) or any(
            value == "missing" for value in control_evidence.values()
        ):
            errors.append(f"ecology control evidence incomplete for {experiment_id}")
        if row.get("all_declared_controls_exercised") is not True:
            errors.append(f"ecology control aggregate drift for {experiment_id}")
        primary_metric_names = set(first["arms"][first["primary_arm"]]["metrics"])
        reported_metric_names = set(first.get("reported_metrics", {}))
        expected_metric_evidence = {
            metric: (
                "primary-arm aggregate"
                if metric in primary_metric_names
                else "unit reported metric"
                if metric in reported_metric_names
                else "missing"
            )
            for metric in registry_row["metrics"]
        }
        if row.get("metric_evidence") != expected_metric_evidence or row.get(
            "all_declared_metrics_reported"
        ) is not True:
            errors.append(f"ecology metric evidence incomplete for {experiment_id}")
        if row.get("independent_unit_count") != len(PRIMARY_SEEDS):
            errors.append(f"ecology unit aggregate drift for {experiment_id}")
        if row.get("strongest_control") != STRONGEST_CONTROLS[experiment_id]:
            errors.append(f"ecology strongest control drift for {experiment_id}")
        if row.get("aggregate_metrics") != expected["aggregate"]:
            errors.append(f"ecology aggregate metric drift for {experiment_id}")
        observed_per_seed = row.get("per_seed_decisions", [])
        if [entry.get("seed") for entry in observed_per_seed] != list(PRIMARY_SEEDS):
            errors.append(f"ecology per-seed identity drift for {experiment_id}")
        elif [entry.get("candidate_favorable") for entry in observed_per_seed] != expected[
            "per_seed"
        ]:
            errors.append(f"ecology per-seed decision drift for {experiment_id}")
        if row.get("difficulty_calibration") != expected["calibration"]:
            errors.append(f"ecology difficulty calibration drift for {experiment_id}")
        if row.get("candidate_result") != expected["candidate_result"]:
            errors.append(f"ecology candidate result drift for {experiment_id}")
        if row.get("tie_is_null") is not True or row.get("promotion") is not False:
            errors.append(f"ecology tie or promotion rule drift for {experiment_id}")
        target = candidate_ids if expected["candidate_result"] == "favorable-candidate" else null_ids
        target.append(experiment_id)
    summary = receipt.get("candidate_summary", {})
    expected_summary = {
        "favorable_candidate_count": len(candidate_ids),
        "null_count": len(null_ids),
        "favorable_candidates": candidate_ids,
        "nulls": null_ids,
        "promotion_count": 0,
    }
    if summary != expected_summary:
        errors.append("ecology candidate summary drift")
    resource = receipt.get("resource_envelope", {})
    if (
        resource.get("lane") != "light-cpu"
        or resource.get("accelerator_required") is not False
        or resource.get("model_weights_loaded") is not False
        or resource.get("heavy_process_started") is not False
    ):
        errors.append("ecology resource envelope drift")
    return errors, {
        "unit_count": len(units),
        "experiment_count": len(experiments),
        "candidate_ids": candidate_ids,
        "null_ids": null_ids,
        "live_implementation_checked": check_live_files,
    }


def _fresh_challenges(
    candidate_ids: list[str], null_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    by_experiment = dict.fromkeys(EXPERIMENT_IDS, True)
    for seed in FRESH_SEEDS:
        unit = run_ecology_battery_seed(seed)
        outcomes: dict[str, bool] = {}
        for experiment_id in EXPERIMENT_IDS:
            expected = _expected_experiment(experiment_id, [unit])
            favorable = expected["candidate_result"] == "favorable-candidate"
            outcomes[experiment_id] = favorable
            if experiment_id in candidate_ids and not favorable:
                errors.append(f"fresh ecology favorable failed for {experiment_id} seed {seed}")
                by_experiment[experiment_id] = False
            if experiment_id in null_ids and favorable:
                errors.append(f"fresh ecology null reversed for {experiment_id} seed {seed}")
                by_experiment[experiment_id] = False
        rows.append(
            {
                "seed": seed,
                "unit_sha256": unit["unit_sha256"],
                "favorable": [experiment_id for experiment_id, value in outcomes.items() if value],
                "null": [experiment_id for experiment_id, value in outcomes.items() if not value],
            }
        )
    return rows, errors, by_experiment


def _rehash_core(receipt: dict[str, Any]) -> None:
    receipt["core_payload_sha256"] = _sha(_core_body(receipt))


def _mutation_tests(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[tuple[str, dict[str, Any], bool]] = []

    metric = copy.deepcopy(receipt)
    metric["units"][0]["experiments"]["f22_active_form_acquisition"]["arms"][
        "information-gain"
    ]["metrics"]["capability_per_sensor_cost"] += 0.01
    unit = metric["units"][0]
    unit["unit_sha256"] = _sha({key: value for key, value in unit.items() if key != "unit_sha256"})
    _rehash_core(metric)
    cases.append(("raw-metric", metric, False))

    control = copy.deepcopy(receipt)
    control["experiments"]["f54_communicative_repair"]["control_evidence"].pop("repetition")
    _rehash_core(control)
    cases.append(("control-drop", control, False))

    verdict = copy.deepcopy(receipt)
    verdict["experiments"]["f28_sensor_value_forecast"]["candidate_result"] = "favorable-candidate"
    _rehash_core(verdict)
    cases.append(("null-flip", verdict, False))

    calibration = copy.deepcopy(receipt)
    calibration["experiments"]["f56_teaching_value"]["difficulty_calibration"][
        "ceilinged_tie"
    ] = True
    _rehash_core(calibration)
    cases.append(("calibration-flip", calibration, False))

    seeds = copy.deepcopy(receipt)
    seeds["primary_seeds"][1] = seeds["primary_seeds"][0]
    _rehash_core(seeds)
    cases.append(("seed-overlap", seeds, False))

    source = copy.deepcopy(receipt)
    source["implementation"][0]["sha256"] = "0" * 64
    _rehash_core(source)
    cases.append(("source-drift", source, True))

    rows: list[dict[str, Any]] = []
    for identifier, payload, check_live in cases:
        errors, _ = _base_errors(payload, check_live_files=check_live)
        rows.append({"id": identifier, "rejected": bool(errors), "observed_errors": errors})
    return rows


def verify_receipt(
    receipt: dict[str, Any], *, check_live_files: bool = True, run_mutations: bool = True
) -> dict[str, Any]:
    errors, checks = _base_errors(receipt, check_live_files=check_live_files)
    fresh_rows, fresh_errors, by_experiment = _fresh_challenges(
        checks.get("candidate_ids", []), checks.get("null_ids", [])
    )
    errors.extend(fresh_errors)
    mutations: list[dict[str, Any]] = []
    mutation_runner_error: str | None = None
    if run_mutations:
        try:
            mutations = _mutation_tests(receipt)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            mutation_runner_error = f"ecology mutation battery could not execute: {exc}"
            errors.append(mutation_runner_error)
    all_mutations_rejected = bool(mutations) and all(row["rejected"] for row in mutations)
    if run_mutations and not all_mutations_rejected:
        errors.append("ecology semantic mutation rejection incomplete")
    candidate_ids = checks.get("candidate_ids", [])
    positives_verified = all(by_experiment.get(experiment_id) is True for experiment_id in candidate_ids)
    if not positives_verified:
        errors.append("ecology favorable candidates lack fresh-seed verification")
    expected_outcomes = {
        experiment_id: (
            "fresh-seed-verified-toy-favorable" if experiment_id in candidate_ids else "null"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    if "verified_outcomes" in receipt and receipt.get("verified_outcomes") != expected_outcomes:
        errors.append("ecology verified outcome summary drift")
    return {
        "schema": VERIFIER_SCHEMA,
        "implementation": "independent raw replay, strict decision recomputation, and fresh-seed attacks",
        "claim_scope": CLAIM_SCOPE,
        "core_payload_sha256": receipt.get("core_payload_sha256"),
        "checks": checks,
        "fresh_seed_challenges": fresh_rows,
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_seeds_disjoint_from_primary": not set(FRESH_SEEDS) & set(PRIMARY_SEEDS),
        "fresh_outcomes_stable_by_experiment": by_experiment,
        "favorable_candidates_all_fresh_verified": positives_verified,
        "mutation_tests": mutations,
        "all_mutations_rejected": all_mutations_rejected,
        "mutation_runner_error": mutation_runner_error,
        "independent": True,
        "adversarial": True,
        "errors": errors,
        "verified": not errors,
    }


def verify_payload_sha256(receipt: dict[str, Any]) -> bool:
    digest = receipt.get("payload_sha256")
    return isinstance(digest, str) and digest == _sha(
        {key: value for key, value in receipt.items() if key != "payload_sha256"}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt",
        nargs="?",
        default=str(REPO_ROOT / "proof" / "F22_F28_F50_F58_ECOLOGY_SCAFFOLD_RUN.json"),
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "proof" / "F22_F28_F50_F58_ECOLOGY_VERIFICATION.json"),
    )
    parser.add_argument("--skip-live-files", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    report = verify_receipt(receipt, check_live_files=not args.skip_live_files, run_mutations=True)
    report["payload_sha256_verified"] = verify_payload_sha256(receipt)
    if not report["payload_sha256_verified"]:
        report["errors"].append("ecology receipt payload digest drift")
        report["verified"] = False
    output = Path(args.report)
    atomic_write_json(output, report)
    print(
        f"wrote {output}: verified={report['verified']}, "
        f"fresh={len(report['fresh_seed_challenges'])}, mutations={len(report['mutation_tests'])}"
    )
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
