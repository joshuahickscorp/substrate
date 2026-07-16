"""Runnable Generation 1 C3 construction-search mechanics pilot.

This adapter lifts the existing deterministic construction-search bed, implementation, and runner
into a sealed multi-seed pilot.  It compares construction search with every declared cheap control
after charging the objective-evaluation cost, records the uncharged oracle headroom diagnostic, and
requires the favorable regime to separate from its paired null regime.

The result is deliberately bounded.  Both the configuration and result prohibit activation and
scientific promotion.  The two default fresh ranges are pilot replication ranges, not an
independently authored verifier.  A favorable result therefore demonstrates mechanics only and
cannot open a ladder stage or establish construction value on natural data.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.mechanisms.construction_search_bed import (
    CHEAP_CONTROLS,
    ORACLE_REFERENCE,
    SEARCH_ARM,
    ConstructionSearchBed,
)
from mop.mechanisms.construction_search_bed import (
    CLAIM_SCOPE as MECHANICS_CLAIM_SCOPE,
)
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner
from mop.substrate.events import canonical_sha256

CONFIG_SCHEMA = "mop-generation1-c3-construction-pilot-config/v1"
RESULT_SCHEMA = "mop-generation1-c3-construction-pilot-result/v1"
STUDY_ID = "G1-C3-G1-CONSTRUCTION-SEARCH-PILOT"
CANONICAL_EPOCH = "G1-G1"
CLAIM_SCOPE = (
    "deterministic generated construction-search mechanics pilot only; "
    "no scientific, capability, natural-data, or activation claim"
)
FRESH_SOURCE = "fresh_generated_outside_c1_c2"
C2_MAX_SEED = 20269192
MAX_SEEDS = 100_000
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_SEED_RANGES: tuple[dict[str, Any], ...] = (
    {
        "name": "producer-pilot",
        "role": "pilot-primary",
        "start": 20276001,
        "count": 256,
        "source": FRESH_SOURCE,
    },
    {
        "name": "challenge-pilot",
        "role": "pilot-challenge",
        "start": 20277001,
        "count": 256,
        "source": FRESH_SOURCE,
    },
)


class Generation1C3ConstructionRefusal(ValueError):
    """Raised when a pilot declaration or receipt is malformed or widens its claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Generation1C3ConstructionRefusal(message)


def _core_without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _sealed(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = dict(core)
    body[field] = canonical_sha256(body)
    return body


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    declared = value.get(field)
    expected = canonical_sha256(_core_without(value, field))
    _require(declared == expected, f"{label} self-seal is invalid")


def _copy_ranges(seed_ranges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(seed_range) for seed_range in seed_ranges]


def build_pilot_config(
    *,
    seed_ranges: Sequence[Mapping[str, Any]] | None = None,
    synergy_bonus: float = 1.0,
    per_eval_cost: float = 0.0002,
    size_penalty: float = 0.05,
    num_members: int = 12,
    num_tasks: int = 3,
    search_restarts: int = 40,
    random_samples: int = 2500,
    minimum_discrimination_fraction: float = 0.75,
) -> dict[str, Any]:
    """Build and seal a configurable fresh-range construction pilot."""

    ranges = _copy_ranges(seed_ranges if seed_ranges is not None else DEFAULT_SEED_RANGES)
    core: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "canonical_epoch": CANONICAL_EPOCH,
        "status": "pilot-authorized",
        "claim_scope": CLAIM_SCOPE,
        "mechanics_claim_scope": MECHANICS_CLAIM_SCOPE,
        "seed_ranges": ranges,
        "bed": {
            "synergy_bonus": synergy_bonus,
            "per_eval_cost": per_eval_cost,
            "size_penalty": size_penalty,
            "num_members": num_members,
            "num_tasks": num_tasks,
            "search_restarts": search_restarts,
            "random_samples": random_samples,
        },
        "controls": list(CHEAP_CONTROLS),
        "oracle_reference": ORACLE_REFERENCE,
        "mechanism_arm": SEARCH_ARM,
        "cost_accounting": {
            "charge": "per-objective-evaluation",
            "all_mechanism_and_control_evaluations_charged": True,
            "oracle_is_uncharged_headroom_only": True,
        },
        "decision_rule": {
            "minimum_discrimination_fraction_per_range": minimum_discrimination_fraction,
            "minimum_overall_discrimination_fraction": minimum_discrimination_fraction,
            "minimum_mean_favorable_charged_margin_over_each_control": 0.0,
            "minimum_null_hold_fraction_per_range": minimum_discrimination_fraction,
            "all_conditions_required": True,
        },
        "independent_verification_complete": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    config = _sealed(core, "config_sha256")
    validate_pilot_config(config)
    return config


def validate_pilot_config(config: Mapping[str, Any]) -> None:
    """Validate a configuration and refuse seed leakage, scope widening, or unsealed mutation."""

    _require(config.get("schema") == CONFIG_SCHEMA, "unsupported construction pilot config schema")
    _require(config.get("study_id") == STUDY_ID, "construction pilot study id drifted")
    _require(config.get("canonical_epoch") == CANONICAL_EPOCH, "construction epoch drifted")
    _require(config.get("status") == "pilot-authorized", "construction pilot is not authorized")
    _require(config.get("claim_scope") == CLAIM_SCOPE, "construction pilot claim scope widened")
    _require(
        config.get("mechanics_claim_scope") == MECHANICS_CLAIM_SCOPE,
        "underlying mechanics claim scope drifted",
    )
    _require(config.get("activation_allowed") is False, "construction pilot cannot activate topology")
    _require(config.get("scientific_promotion") is False, "construction pilot cannot promote science")
    _require(
        config.get("independent_verification_complete") is False,
        "same-source pilot cannot claim independent verification",
    )
    _require(config.get("controls") == list(CHEAP_CONTROLS), "charged control set drifted")
    _require(config.get("oracle_reference") == ORACLE_REFERENCE, "oracle reference drifted")
    _require(config.get("mechanism_arm") == SEARCH_ARM, "construction mechanism arm drifted")

    accounting = config.get("cost_accounting")
    _require(isinstance(accounting, Mapping), "cost accounting must be an object")
    _require(accounting.get("charge") == "per-objective-evaluation", "cost charge drifted")
    _require(
        accounting.get("all_mechanism_and_control_evaluations_charged") is True,
        "all mechanism and control evaluations must be charged",
    )
    _require(
        accounting.get("oracle_is_uncharged_headroom_only") is True,
        "oracle must remain an uncharged diagnostic only",
    )

    bed = config.get("bed")
    _require(isinstance(bed, Mapping), "construction bed declaration must be an object")
    for field in ("synergy_bonus", "per_eval_cost", "size_penalty"):
        value = bed.get(field)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"bed {field} must be finite",
        )
    _require(float(bed["synergy_bonus"]) >= 0.0, "synergy bonus cannot be negative")
    _require(float(bed["per_eval_cost"]) >= 0.0, "per-evaluation cost cannot be negative")
    _require(float(bed["size_penalty"]) > 0.0, "size penalty must be positive")
    for field, minimum in (
        ("num_members", 4),
        ("num_tasks", 2),
        ("search_restarts", 1),
        ("random_samples", 1),
    ):
        value = bed.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"bed {field} must be an integer at least {minimum}",
        )

    rule = config.get("decision_rule")
    _require(isinstance(rule, Mapping), "decision rule must be an object")
    for field in (
        "minimum_discrimination_fraction_per_range",
        "minimum_overall_discrimination_fraction",
        "minimum_null_hold_fraction_per_range",
    ):
        value = rule.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 < float(value) <= 1.0,
            f"decision rule {field} must be in (0, 1]",
        )
    _require(
        rule.get("minimum_mean_favorable_charged_margin_over_each_control") == 0.0,
        "pilot margin threshold must stay at strict positivity",
    )
    _require(rule.get("all_conditions_required") is True, "all pilot decision conditions are required")

    ranges = config.get("seed_ranges")
    _require(isinstance(ranges, list) and ranges, "at least one fresh seed range is required")
    _require(len(ranges) <= 16, "too many construction pilot seed ranges")
    names: set[str] = set()
    intervals: list[tuple[int, int, str]] = []
    total = 0
    for seed_range in ranges:
        _require(isinstance(seed_range, Mapping), "seed range must be an object")
        name = seed_range.get("name")
        role = seed_range.get("role")
        start = seed_range.get("start")
        count = seed_range.get("count")
        _require(isinstance(name, str) and _ID_RE.fullmatch(name) is not None, "seed range name invalid")
        _require(name not in names, "seed range names must be unique")
        names.add(name)
        _require(role in ("pilot-primary", "pilot-challenge"), "unsupported pilot seed range role")
        _require(seed_range.get("source") == FRESH_SOURCE, "construction pilot seeds must be fresh")
        _require(
            isinstance(start, int) and not isinstance(start, bool) and start > C2_MAX_SEED,
            "construction pilot seed range overlaps prior C1/C2 authority",
        )
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "construction pilot seed range count must be positive",
        )
        end = start + count - 1
        _require(end >= start, "construction pilot seed range overflowed")
        intervals.append((start, end, name))
        total += count
    _require(total <= MAX_SEEDS, "construction pilot exceeds the bounded seed ceiling")
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:], strict=False):
        _require(previous[1] < current[0], "construction pilot seed ranges overlap")

    _validate_seal(config, "config_sha256", "construction pilot config")


def _comparison(nets: Mapping[str, float], margins: Mapping[str, float]) -> dict[str, Any]:
    return {
        "search_charged_net": nets[SEARCH_ARM],
        "control_charged_nets": {control: nets[control] for control in CHEAP_CONTROLS},
        "charged_margins": {control: margins[control] for control in CHEAP_CONTROLS},
        "search_beats_all_controls": all(margins[control] > 0.0 for control in CHEAP_CONTROLS),
    }


def _run_seed(
    *,
    runner: ConstructionSearchRunner,
    bed: ConstructionSearchBed,
    range_name: str,
    range_role: str,
    seed: int,
) -> dict[str, Any]:
    results = runner.run(bed, seed)
    receipt = runner.mint(results)
    favorable = _comparison(results.favorable_nets, results.favorable_margins)
    null = _comparison(results.null_nets, results.null_margins)
    discriminates = bool(favorable["search_beats_all_controls"] and results.null_holds)
    core: dict[str, Any] = {
        "range_name": range_name,
        "range_role": range_role,
        "seed": seed,
        "per_eval_cost": results.per_eval_cost,
        "favorable_objective_sha256": results.favorable_objective_digest,
        "null_objective_sha256": results.null_objective_digest,
        "favorable": favorable,
        "null": null,
        "null_holds": results.null_holds,
        "favorable_oracle_headroom_gap": results.favorable_headroom_gap,
        "null_favorable_discrimination": discriminates,
        "mechanics_receipt": {
            "kind": receipt.kind,
            "verdict": receipt.verdict,
            "controls_cleared": list(receipt.controls_cleared),
            "evidence_sha256": receipt.evidence_digest,
            "receipt_sha256": receipt.digest(),
            "is_confirmation": receipt.is_confirmation,
        },
    }
    return _sealed(core, "row_sha256")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot aggregate an empty construction pilot")
    count = len(rows)
    favorable_count = sum(bool(row["favorable"]["search_beats_all_controls"]) for row in rows)
    null_count = sum(bool(row["null_holds"]) for row in rows)
    discrimination_count = sum(bool(row["null_favorable_discrimination"]) for row in rows)
    favorable_margins = {
        control: [float(row["favorable"]["charged_margins"][control]) for row in rows]
        for control in CHEAP_CONTROLS
    }
    null_margins = {
        control: [float(row["null"]["charged_margins"][control]) for row in rows]
        for control in CHEAP_CONTROLS
    }
    headroom = [float(row["favorable_oracle_headroom_gap"]) for row in rows]
    return {
        "seed_count": count,
        "favorable_beats_all_count": favorable_count,
        "favorable_beats_all_fraction": favorable_count / count,
        "null_holds_count": null_count,
        "null_holds_fraction": null_count / count,
        "discrimination_count": discrimination_count,
        "discrimination_fraction": discrimination_count / count,
        "favorable_charged_margin_by_control": {
            control: {
                "mean": _mean(favorable_margins[control]),
                "minimum": min(favorable_margins[control]),
                "maximum": max(favorable_margins[control]),
            }
            for control in CHEAP_CONTROLS
        },
        "null_charged_margin_by_control": {
            control: {
                "mean": _mean(null_margins[control]),
                "minimum": min(null_margins[control]),
                "maximum": max(null_margins[control]),
            }
            for control in CHEAP_CONTROLS
        },
        "oracle_headroom": {
            "mean_gap": _mean(headroom),
            "maximum_gap": max(headroom),
            "zero_gap_fraction": sum(value == 0.0 for value in headroom) / count,
            "interpretation": "uncharged exhaustive diagnostic only",
        },
    }


def _decision(
    *,
    rule: Mapping[str, Any],
    overall: Mapping[str, Any],
    ranges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    range_discrimination = all(
        aggregate["discrimination_fraction"]
        >= rule["minimum_discrimination_fraction_per_range"]
        for aggregate in ranges
    )
    range_nulls = all(
        aggregate["null_holds_fraction"] >= rule["minimum_null_hold_fraction_per_range"]
        for aggregate in ranges
    )
    overall_discrimination = (
        overall["discrimination_fraction"] >= rule["minimum_overall_discrimination_fraction"]
    )
    positive_mean_margins = all(
        overall["favorable_charged_margin_by_control"][control]["mean"]
        > rule["minimum_mean_favorable_charged_margin_over_each_control"]
        for control in CHEAP_CONTROLS
    )
    earned = range_discrimination and range_nulls and overall_discrimination and positive_mean_margins
    return {
        "mechanics_pattern_observed": earned,
        "verdict": "mechanics-pilot-favorable" if earned else "mechanics-pilot-null",
        "conditions": {
            "every_range_clears_discrimination_fraction": range_discrimination,
            "every_range_preserves_the_prior_null": range_nulls,
            "overall_discrimination_fraction_cleared": overall_discrimination,
            "positive_mean_charged_margin_over_every_control": positive_mean_margins,
        },
        "interpretation": (
            "toy construction-search plumbing separated a favorable synthetic regime from its null"
            if earned
            else "the configured toy construction-search pilot did not clear its mechanics rule"
        ),
        "independent_verification_complete": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def run_pilot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute every configured seed and return a deterministic, sealed pilot result."""

    validate_pilot_config(config)
    bed_decl = config["bed"]
    bed = ConstructionSearchBed(
        synergy_bonus=float(bed_decl["synergy_bonus"]),
        per_eval_cost=float(bed_decl["per_eval_cost"]),
        size_penalty=float(bed_decl["size_penalty"]),
        num_members=int(bed_decl["num_members"]),
        num_tasks=int(bed_decl["num_tasks"]),
        search_restarts=int(bed_decl["search_restarts"]),
        random_samples=int(bed_decl["random_samples"]),
    )
    runner = ConstructionSearchRunner()
    rows: list[dict[str, Any]] = []
    range_aggregates: list[dict[str, Any]] = []
    for seed_range in config["seed_ranges"]:
        range_rows = [
            _run_seed(
                runner=runner,
                bed=bed,
                range_name=seed_range["name"],
                range_role=seed_range["role"],
                seed=seed,
            )
            for seed in range(seed_range["start"], seed_range["start"] + seed_range["count"])
        ]
        rows.extend(range_rows)
        range_aggregates.append(
            {
                "range_name": seed_range["name"],
                "range_role": seed_range["role"],
                **_aggregate_rows(range_rows),
            }
        )
    overall = _aggregate_rows(rows)
    decision = _decision(rule=config["decision_rule"], overall=overall, ranges=range_aggregates)
    core: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "study_id": STUDY_ID,
        "canonical_epoch": CANONICAL_EPOCH,
        "claim_scope": CLAIM_SCOPE,
        "config": dict(config),
        "config_sha256": config["config_sha256"],
        "rows": rows,
        "range_aggregates": range_aggregates,
        "overall": overall,
        "decision": decision,
        "pilot_only": True,
        "independent_verification_complete": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = _sealed(core, "result_sha256")
    validate_pilot_result(result)
    return result


def _validate_comparison(comparison: Mapping[str, Any], label: str) -> None:
    _require(isinstance(comparison, Mapping), f"{label} comparison must be an object")
    nets = comparison.get("control_charged_nets")
    margins = comparison.get("charged_margins")
    _require(isinstance(nets, Mapping), f"{label} control nets must be an object")
    _require(isinstance(margins, Mapping), f"{label} margins must be an object")
    _require(set(nets) == set(CHEAP_CONTROLS), f"{label} charged control net set drifted")
    _require(set(margins) == set(CHEAP_CONTROLS), f"{label} charged margin set drifted")
    search_net = comparison.get("search_charged_net")
    _require(isinstance(search_net, (int, float)), f"{label} search net must be numeric")
    for control in CHEAP_CONTROLS:
        _require(
            margins[control] == search_net - nets[control],
            f"{label} charged margin for {control} is inconsistent",
        )
    expected = all(margins[control] > 0.0 for control in CHEAP_CONTROLS)
    _require(
        comparison.get("search_beats_all_controls") is expected,
        f"{label} all-controls verdict is inconsistent",
    )


def validate_pilot_result(result: Mapping[str, Any]) -> None:
    """Recompute aggregation and decisions, then validate every row and self-seal."""

    _require(result.get("schema") == RESULT_SCHEMA, "unsupported construction pilot result schema")
    _require(result.get("study_id") == STUDY_ID, "construction result study id drifted")
    _require(result.get("canonical_epoch") == CANONICAL_EPOCH, "construction result epoch drifted")
    _require(result.get("claim_scope") == CLAIM_SCOPE, "construction result scope widened")
    _require(result.get("pilot_only") is True, "construction result must remain pilot-only")
    _require(result.get("activation_allowed") is False, "construction result cannot activate")
    _require(result.get("scientific_promotion") is False, "construction result cannot promote science")
    _require(
        result.get("independent_verification_complete") is False,
        "construction pilot cannot claim independent verification",
    )
    config = result.get("config")
    _require(isinstance(config, Mapping), "construction result must embed its config")
    validate_pilot_config(config)
    _require(result.get("config_sha256") == config["config_sha256"], "config binding drifted")

    rows = result.get("rows")
    _require(isinstance(rows, list) and rows, "construction result rows must be nonempty")
    expected: dict[tuple[str, int], str] = {}
    for seed_range in config["seed_ranges"]:
        for seed in range(seed_range["start"], seed_range["start"] + seed_range["count"]):
            expected[(seed_range["name"], seed)] = seed_range["role"]
    _require(len(rows) == len(expected), "construction result seed count drifted")
    observed: set[tuple[str, int]] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "construction result row must be an object")
        key = (row.get("range_name"), row.get("seed"))
        _require(key in expected and key not in observed, "construction result seed authority drifted")
        observed.add(key)
        _require(row.get("range_role") == expected[key], "construction result range role drifted")
        _validate_comparison(row.get("favorable", {}), "favorable")
        _validate_comparison(row.get("null", {}), "null")
        null_expected = not row["null"]["search_beats_all_controls"]
        _require(row.get("null_holds") is null_expected, "row null verdict is inconsistent")
        discrimination = bool(row["favorable"]["search_beats_all_controls"] and null_expected)
        _require(
            row.get("null_favorable_discrimination") is discrimination,
            "row null/favorable discrimination is inconsistent",
        )
        headroom = row.get("favorable_oracle_headroom_gap")
        _require(
            isinstance(headroom, (int, float)) and headroom >= 0.0,
            "oracle headroom gap must be nonnegative",
        )
        for field in ("favorable_objective_sha256", "null_objective_sha256"):
            _require(
                isinstance(row.get(field), str) and _SHA256_RE.fullmatch(row[field]) is not None,
                f"row {field} is invalid",
            )
        receipt = row.get("mechanics_receipt")
        _require(isinstance(receipt, Mapping), "row mechanics receipt must be an object")
        _require(receipt.get("kind") == "mechanics-demonstration", "row receipt must be a demonstration")
        _require(receipt.get("is_confirmation") is False, "row receipt cannot be a confirmation")
        expected_verdict = "mechanics-ok" if row["favorable"]["search_beats_all_controls"] else "null"
        _require(receipt.get("verdict") == expected_verdict, "row receipt verdict is inconsistent")
        for field in ("evidence_sha256", "receipt_sha256"):
            _require(
                isinstance(receipt.get(field), str) and _SHA256_RE.fullmatch(receipt[field]) is not None,
                f"row receipt {field} is invalid",
            )
        _validate_seal(row, "row_sha256", "construction pilot row")
    _require(observed == set(expected), "construction result omitted an authorized seed")

    expected_ranges = []
    for seed_range in config["seed_ranges"]:
        range_rows = [row for row in rows if row["range_name"] == seed_range["name"]]
        expected_ranges.append(
            {
                "range_name": seed_range["name"],
                "range_role": seed_range["role"],
                **_aggregate_rows(range_rows),
            }
        )
    _require(result.get("range_aggregates") == expected_ranges, "range aggregation drifted")
    expected_overall = _aggregate_rows(rows)
    _require(result.get("overall") == expected_overall, "overall aggregation drifted")
    expected_decision = _decision(
        rule=config["decision_rule"], overall=expected_overall, ranges=expected_ranges
    )
    _require(result.get("decision") == expected_decision, "construction pilot decision drifted")
    _require(result["decision"]["activation_allowed"] is False, "decision cannot activate topology")
    _require(result["decision"]["scientific_promotion"] is False, "decision cannot promote science")
    _validate_seal(result, "result_sha256", "construction pilot result")


def atomic_write_result(path: Path | str, result: Mapping[str, Any]) -> None:
    """Validate and atomically write a pilot result as canonical JSON."""

    validate_pilot_result(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_and_validate_result(path: Path | str) -> dict[str, Any]:
    """Load a written pilot result and validate its full fail-closed envelope."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "construction pilot result file must contain an object")
    validate_pilot_result(value)
    return value
