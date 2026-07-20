
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.mechanisms.messaging_repair_bed import MessagingRepairBed, build_default_bed
from mop.mechanisms.messaging_repair_runner import (
    MessagingRepairRunner,
    RegimeScore,
    RunResult,
    build_default_runner,
)
from mop.substrate.events import canonical_sha256

CONFIG_SCHEMA = "mop-generation1-c3-communication-pilot-config/v1"
RESULT_SCHEMA = "mop-generation1-c3-communication-pilot-result/v1"
PILOT_ID = "generation1-c3-v1-m1-communication-mechanics-pilot-v1"
CLAIM_SCOPE = (
    "deterministic multi-seed mechanics pilot only; no scientific, capability, "
    "natural-data, activation, or promotion claim"
)

C2_RESULT_PATH = "proof/GENERATION1_CONTEXT_ROUTING.json"
C2_VERIFICATION_PATH = "proof/GENERATION1_CONTEXT_ROUTING.verification.json"
FIRST_FRESH_SEED = 20_270_001
MAX_SEEDS_PER_LANE = 4096

V1_CONTROLS: tuple[str, ...] = ("no-verify", "always-verify")
M1_CONTROLS: tuple[str, ...] = (
    "no-message",
    "broadcast-all",
    "stale-message",
    "majority-vote",
)
AGGREGATE_METRICS: tuple[str, ...] = (
    "mean_mechanism_improvement",
    "mean_control_improvement",
    "mean_matched_improvement_margin",
    "minimum_matched_improvement_margin",
    "strict_win_seed_fraction",
)


class CommunicationPilotError(ValueError):
    pass


def _canonical_without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommunicationPilotError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CommunicationPilotError(f"{label} must be a JSON object")
    return value


def _prerequisite(repo_root: Path) -> dict[str, Any]:
    result_path = repo_root / C2_RESULT_PATH
    verification_path = repo_root / C2_VERIFICATION_PATH
    result = _read_object(result_path, "C2 result")
    verification = _read_object(verification_path, "C2 verification")
    return {
        "result_path": C2_RESULT_PATH,
        "result_file_sha256": _file_sha256(result_path),
        "result_sha256": result.get("result_sha256"),
        "c2_complete": result.get("complete"),
        "c2_ready_to_preregister_successor": result.get("decision", {}).get(
            "ready_to_preregister_g1_c3_learned_dispatch"
        ),
        "verification_path": C2_VERIFICATION_PATH,
        "verification_file_sha256": _file_sha256(verification_path),
        "verification_sha256": verification.get("verification_sha256"),
        "independent_verification_complete": verification.get("verification_complete"),
        "all_cells_reproduced": verification.get("dataset_reproduction", {}).get(
            "all_dataset_and_metric_reproductions_passed"
        ),
        "fresh_actor_canary_passed": verification.get("fresh_actor_canary", {}).get("passed"),
        "all_mutations_rejected": verification.get("mutation_suite", {}).get("all_rejected"),
    }


def _seed_range(name: str, start: int, count: int) -> dict[str, Any]:
    return {
        "name": name,
        "start": start,
        "count": count,
        "source": "fresh_generated_mechanics_pilot",
    }


def _lane_authority(lane_id: str, controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "controls": list(controls),
        "metrics": list(AGGREGATE_METRICS),
        "decision_rule": {
            "favorable_strict_win_seed_fraction_must_equal": 1.0,
            "null_strict_win_seed_fraction_must_equal": 0.0,
            "favorable_minimum_margin_must_exceed": 0.0,
            "all_conditions_required": True,
        },
    }


def build_config(
    *,
    repo_root: Path = REPO_ROOT,
    seed_count: int = 256,
    v1_seed_start: int = 20_278_001,
    m1_seed_start: int = 20_279_001,
) -> dict[str, Any]:

    core: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "pilot_id": PILOT_ID,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics_pilot_executable",
        "prerequisite": _prerequisite(repo_root),
        "seed_ranges": {
            "G1-V1": _seed_range("v1_mechanics_pilot", v1_seed_start, seed_count),
            "G1-M1": _seed_range("m1_mechanics_pilot", m1_seed_start, seed_count),
        },
        "lanes": {
            "G1-V1": _lane_authority("G1-V1", V1_CONTROLS),
            "G1-M1": _lane_authority("G1-M1", M1_CONTROLS),
        },
        "bed_authority": {
            "mechanism_id": "messaging_repair",
            "regimes": ["null", "favorable"],
            "matched_action_budget_required": True,
            "toy_bed_constructed_for_discrimination": True,
            "independent_scientific_verifier": False,
        },
        "mechanics_pilot_execution_authorized": True,
        "scientific_execution_authorized": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "config_sha256": canonical_sha256(core)}


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise CommunicationPilotError(f"{label} drifted")


def _validate_seed_ranges(config: Mapping[str, Any]) -> None:
    ranges = config.get("seed_ranges")
    if not isinstance(ranges, Mapping):
        raise CommunicationPilotError("seed ranges must be an object")
    _require_exact(frozenset(ranges), frozenset({"G1-V1", "G1-M1"}), "seed lane inventory")
    intervals: list[tuple[int, int, str]] = []
    for lane_id, expected_name in (("G1-V1", "v1_mechanics_pilot"), ("G1-M1", "m1_mechanics_pilot")):
        item = ranges.get(lane_id)
        if not isinstance(item, Mapping):
            raise CommunicationPilotError(f"{lane_id} seed range must be an object")
        _require_exact(item.get("name"), expected_name, f"{lane_id} seed range name")
        _require_exact(
            item.get("source"), "fresh_generated_mechanics_pilot", f"{lane_id} seed source"
        )
        start = item.get("start")
        count = item.get("count")
        if isinstance(start, bool) or not isinstance(start, int) or start < FIRST_FRESH_SEED:
            raise CommunicationPilotError(f"{lane_id} seed start is not in the fresh authority")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or count > MAX_SEEDS_PER_LANE
        ):
            raise CommunicationPilotError(f"{lane_id} seed count must be in [1, {MAX_SEEDS_PER_LANE}]")
        intervals.append((start, start + count, lane_id))
    intervals.sort()
    if intervals[1][0] < intervals[0][1]:
        raise CommunicationPilotError("communication pilot seed ranges overlap")


def _validate_prerequisite(config: Mapping[str, Any], repo_root: Path) -> None:
    actual = _prerequisite(repo_root)
    prerequisite = config.get("prerequisite")
    if prerequisite != actual:
        raise CommunicationPilotError("C2 prerequisite proof binding drifted")
    required_true = (
        "c2_complete",
        "c2_ready_to_preregister_successor",
        "independent_verification_complete",
        "all_cells_reproduced",
        "fresh_actor_canary_passed",
        "all_mutations_rejected",
    )
    if not all(actual.get(field) is True for field in required_true):
        raise CommunicationPilotError("C2 prerequisite is not clean")


def validate_config(config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:

    _require_exact(config.get("schema"), CONFIG_SCHEMA, "config schema")
    _require_exact(config.get("pilot_id"), PILOT_ID, "pilot id")
    _require_exact(config.get("claim_scope"), CLAIM_SCOPE, "claim scope")
    _require_exact(config.get("status"), "mechanics_pilot_executable", "pilot status")
    expected_seal = canonical_sha256(_canonical_without(config, "config_sha256"))
    _require_exact(config.get("config_sha256"), expected_seal, "config self-seal")
    _validate_prerequisite(config, repo_root)
    _validate_seed_ranges(config)

    lanes = config.get("lanes")
    if not isinstance(lanes, Mapping):
        raise CommunicationPilotError("lanes must be an object")
    _require_exact(frozenset(lanes), frozenset({"G1-V1", "G1-M1"}), "lane inventory")
    for lane_id, controls in (("G1-V1", V1_CONTROLS), ("G1-M1", M1_CONTROLS)):
        expected = _lane_authority(lane_id, controls)
        _require_exact(lanes.get(lane_id), expected, f"{lane_id} authority")

    _require_exact(
        config.get("bed_authority"),
        {
            "mechanism_id": "messaging_repair",
            "regimes": ["null", "favorable"],
            "matched_action_budget_required": True,
            "toy_bed_constructed_for_discrimination": True,
            "independent_scientific_verifier": False,
        },
        "bed authority",
    )
    _require_exact(config.get("mechanics_pilot_execution_authorized"), True, "pilot execution flag")
    _require_exact(config.get("scientific_execution_authorized"), False, "scientific execution flag")
    _require_exact(config.get("activation_allowed"), False, "activation flag")
    _require_exact(config.get("scientific_promotion"), False, "promotion flag")


def _mean(values: Sequence[int]) -> float:
    if not values:
        raise CommunicationPilotError("cannot aggregate an empty seed inventory")
    return sum(values) / len(values)


def _aggregate_regime(scores: Sequence[RegimeScore], controls: tuple[str, ...]) -> dict[str, Any]:
    mechanisms = [score.mechanism_improvement for score in scores]
    control_records: dict[str, dict[str, float | int]] = {}
    strict_wins = 0
    for name in controls:
        improvements: list[int] = []
        margins: list[int] = []
        for score in scores:
            score_controls = dict(score.control_improvements)
            score_margins = dict(score.margins)
            if name not in score_controls or name not in score_margins:
                raise CommunicationPilotError(f"runner omitted declared control {name!r}")
            improvements.append(score_controls[name])
            margins.append(score_margins[name])
        control_records[name] = {
            "mean_control_improvement": _mean(improvements),
            "mean_matched_improvement_margin": _mean(margins),
            "minimum_matched_improvement_margin": min(margins),
            "maximum_matched_improvement_margin": max(margins),
        }
    for score in scores:
        margins = dict(score.margins)
        if all(margins[name] > 0 for name in controls):
            strict_wins += 1
    return {
        "seed_count": len(scores),
        "mean_mechanism_improvement": _mean(mechanisms),
        "controls": control_records,
        "strict_win_seed_fraction": strict_wins / len(scores),
    }


def _lane_result(
    *,
    lane_id: str,
    seed_range: Mapping[str, Any],
    controls: tuple[str, ...],
    runner: MessagingRepairRunner,
    bed: MessagingRepairBed,
) -> dict[str, Any]:
    start = int(seed_range["start"])
    count = int(seed_range["count"])
    results: list[RunResult] = [runner.run(bed, seed) for seed in range(start, start + count)]
    null = _aggregate_regime([result.null for result in results], controls)
    favorable = _aggregate_regime([result.favorable for result in results], controls)
    favorable_minimum_margin = min(
        int(record["minimum_matched_improvement_margin"])
        for record in favorable["controls"].values()
    )
    favorable_wins = favorable["strict_win_seed_fraction"] == 1.0
    null_holds = null["strict_win_seed_fraction"] == 0.0
    passed = favorable_wins and null_holds and favorable_minimum_margin > 0
    return {
        "lane_id": lane_id,
        "seed_range": dict(seed_range),
        "controls": list(controls),
        "metrics": list(AGGREGATE_METRICS),
        "matched_action_budget": bed.action_budget,
        "null": null,
        "favorable": favorable,
        "discrimination": {
            "favorable_all_controls_win_fraction": favorable["strict_win_seed_fraction"],
            "null_all_controls_win_fraction": null["strict_win_seed_fraction"],
            "favorable_minimum_margin": favorable_minimum_margin,
            "named_null_preserved": null_holds,
            "pilot_mechanics_discrimination_passed": passed,
        },
        "per_seed_result_digest_fold": canonical_sha256([result.digest() for result in results]),
    }


def _run_pilot(
    config: Mapping[str, Any],
    *,
    runner: MessagingRepairRunner,
    bed: MessagingRepairBed,
) -> dict[str, Any]:
    ranges = config["seed_ranges"]
    v1 = _lane_result(
        lane_id="G1-V1",
        seed_range=ranges["G1-V1"],
        controls=V1_CONTROLS,
        runner=runner,
        bed=bed,
    )
    m1 = _lane_result(
        lane_id="G1-M1",
        seed_range=ranges["G1-M1"],
        controls=M1_CONTROLS,
        runner=runner,
        bed=bed,
    )
    lane_passes = {
        "G1-V1": v1["discrimination"]["pilot_mechanics_discrimination_passed"],
        "G1-M1": m1["discrimination"]["pilot_mechanics_discrimination_passed"],
    }
    core: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "pilot_id": PILOT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config": dict(config),
        "config_sha256": config["config_sha256"],
        "lanes": {"G1-V1": v1, "G1-M1": m1},
        "decision": {
            "lane_pilot_discrimination": lane_passes,
            "all_lanes_discriminate_as_constructed": all(lane_passes.values()),
            "supports_real_producer_implementation": all(lane_passes.values()),
            "scientific_confirmation": False,
            "interpretation": (
                "The existing toy harness distinguishes its constructed null and favorable "
                "regimes. It does not isolate a natural verification or messaging effect."
            ),
        },
        "execution_complete": True,
        "independent_scientific_verification_complete": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def run_pilot(
    config: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    runner: MessagingRepairRunner | None = None,
    bed: MessagingRepairBed | None = None,
) -> dict[str, Any]:

    validate_config(config, repo_root=repo_root)
    return _run_pilot(
        config,
        runner=runner if runner is not None else build_default_runner(),
        bed=bed if bed is not None else build_default_bed(),
    )


def validate_result(
    result: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    replay: bool = True,
) -> None:

    _require_exact(result.get("schema"), RESULT_SCHEMA, "result schema")
    _require_exact(result.get("pilot_id"), PILOT_ID, "result pilot id")
    _require_exact(result.get("claim_scope"), CLAIM_SCOPE, "result claim scope")
    expected_seal = canonical_sha256(_canonical_without(result, "result_sha256"))
    _require_exact(result.get("result_sha256"), expected_seal, "result self-seal")
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise CommunicationPilotError("result config must be an object")
    validate_config(config, repo_root=repo_root)
    _require_exact(result.get("config_sha256"), config.get("config_sha256"), "result config binding")
    _require_exact(result.get("execution_complete"), True, "execution completion")
    _require_exact(
        result.get("independent_scientific_verification_complete"),
        False,
        "independent scientific verification flag",
    )
    _require_exact(result.get("activation_allowed"), False, "result activation flag")
    _require_exact(result.get("scientific_promotion"), False, "result promotion flag")
    decision = result.get("decision")
    if not isinstance(decision, Mapping):
        raise CommunicationPilotError("result decision must be an object")
    _require_exact(decision.get("scientific_confirmation"), False, "scientific confirmation flag")
    if replay:
        expected = _run_pilot(
            config,
            runner=build_default_runner(),
            bed=build_default_bed(),
        )
        _require_exact(result, expected, "canonical mechanics replay")


__all__ = [
    "AGGREGATE_METRICS",
    "CLAIM_SCOPE",
    "CommunicationPilotError",
    "CONFIG_SCHEMA",
    "M1_CONTROLS",
    "PILOT_ID",
    "RESULT_SCHEMA",
    "V1_CONTROLS",
    "build_config",
    "run_pilot",
    "validate_config",
    "validate_result",
]
