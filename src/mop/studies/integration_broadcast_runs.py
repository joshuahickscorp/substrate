"""Executed toy beds for f36 broadcast necessity and f37 broadcast sufficiency.

The task is a bounded binary shared-message fixture with calibrated signal noise. Every arm spends
the same declared scalar-operation budget through deterministic padding. It is programmatic R1
evidence only. In particular, an exact comparator tie is recorded as a null for sufficiency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from random import Random
from typing import Any

import yaml

from ..config import REPO_ROOT
from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.events import canonical_sha256
from .integration_battery_scaffold import make_broadcast_contract

CONFIG_SCHEMA = "mop-integration-broadcast-run-config/v1"
RECEIPT_SCHEMA = "mop-integration-broadcast-run/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "integration_broadcast_runs.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "INTEGRATION_BROADCAST_RUN.json"
EXPERIMENT_IDS = ("f36_limited_broadcast_necessity", "f37_broadcast_sufficiency")
SOURCE_PATHS = (
    "configs/experiment/integration_broadcast_runs.yaml",
    "registry/experiments.yaml",
    "src/mop/studies/integration_battery_scaffold.py",
    "src/mop/studies/integration_broadcast_runs.py",
    "src/mop/studies/integration_broadcast_verify.py",
    "scripts/run_integration_broadcast.py",
    "scripts/verify_integration_broadcast.py",
    "tests/unit/test_integration_broadcast_runs.py",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("integration broadcast run config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("integration broadcast run claim scope drift")
    if not isinstance(config.get("null_hypothesis"), str) or not config["null_hypothesis"].strip():
        raise ValueError("integration broadcast run null hypothesis is required")
    units = config.get("independent_units", {})
    seeds = tuple(int(value) for value in units.get("seeds", ()))
    fresh = tuple(int(value) for value in units.get("fresh_verifier_seeds", ()))
    minimum = int(config.get("stop_contract", {}).get("minimum_independent_units", 0))
    if len(seeds) < minimum or len(fresh) < minimum:
        raise ValueError("integration broadcast run lacks the preregistered seed minimum")
    if len(set(seeds + fresh)) != len(seeds) + len(fresh) or min(seeds + fresh) < 0:
        raise ValueError("integration primary and fresh seeds must be unique and nonnegative")
    fixture = config.get("fixture", {})
    if (
        int(fixture.get("heldout_episodes", 0)) < 100
        or int(fixture.get("n_specialists", 0)) < 4
        or int(fixture.get("n_consumers", 0)) < 2
        or int(fixture.get("recurrence_steps", 0)) < 1
        or not 0.5 < float(fixture.get("signal_reliability", 0.0)) < 1.0
        or not 0.5 < float(fixture.get("cue_reliability", 0.0)) < 1.0
        or int(fixture.get("matched_flop_budget_per_episode", 0)) < 32
    ):
        raise ValueError("integration broadcast fixture is missing a calibrated bounded task")
    expected_controls = {
        EXPERIMENT_IDS[0]: (
            "unrestricted-bus",
            "lesion-broadcast",
            "delay-broadcast",
            "restore-broadcast",
            "message-shuffled",
        ),
        EXPERIMENT_IDS[1]: (
            "unrestricted-bus",
            "matched-dense-state",
            "feed-forward-depth",
            "independent-specialists",
            "equal-flop-routing",
        ),
    }
    if tuple(config.get("experiments", {})) != EXPERIMENT_IDS:
        raise ValueError("integration broadcast experiment coverage or order drift")
    for experiment_id, controls in expected_controls.items():
        if tuple(config["experiments"][experiment_id].get("controls", ())) != controls:
            raise ValueError(f"integration broadcast control drift for {experiment_id}")
    envelope = config.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("external_data_allowed") is not False
    ):
        raise ValueError("integration broadcast run must remain one-thread CPU and self-contained")
    return config


def _registry_bindings() -> dict[str, Any]:
    registry = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in registry["experiments"]}
    bindings = {}
    for experiment_id in EXPERIMENT_IDS:
        row = index.get(experiment_id)
        if row is None or row.get("status") != "registry-only" or row.get("resource_tier") != "cpu-now":
            raise ValueError(f"live registry row {experiment_id} is absent or outside the local lane")
        bindings[experiment_id] = {
            "row_sha256": canonical_sha256(row),
            "null_hypothesis": row["null_hypothesis"],
            "metrics": list(row["metrics"]),
            "controls": list(row["controls"]),
            "status": row["status"],
        }
    return bindings


def _episodes(seed: int, fixture: dict[str, Any]) -> list[dict[str, Any]]:
    rng = Random(_seed(seed, "integration-broadcast:episodes"))
    count = int(fixture["heldout_episodes"])
    specialists = int(fixture["n_specialists"])
    signal_reliability = float(fixture["signal_reliability"])
    cue_reliability = float(fixture["cue_reliability"])
    rows = []
    for episode_index in range(count):
        target = rng.randrange(2)
        relevant = rng.randrange(specialists)
        cue = relevant
        if rng.random() >= cue_reliability:
            cue = (relevant + 1 + rng.randrange(specialists - 1)) % specialists
        signals = []
        for specialist in range(specialists):
            reliability = signal_reliability if specialist == relevant else 0.5
            signals.append(target if rng.random() < reliability else 1 - target)
        fallback = [rng.randrange(2) for _ in range(int(fixture["n_consumers"]))]
        rows.append(
            {
                "episode_index": episode_index,
                "target": target,
                "relevant_specialist": relevant,
                "cue_specialist": cue,
                "signals": signals,
                "fallback": fallback,
            }
        )
    return rows


def _padding_checksum(seed: int, arm: str, used: int, budget: int) -> int:
    if used > budget:
        raise ValueError(f"arm {arm} exceeds the matched scalar-operation budget")
    value = _seed(seed, arm) & 0xFFFF
    for index in range(budget - used):
        value = ((value ^ index) * 33 + 17) & 0xFFFFFFFF
    return value


def _accuracy(targets: list[int], predictions: list[int]) -> float:
    return sum(target == prediction for target, prediction in zip(targets, predictions, strict=True)) / len(
        targets
    )


def _predictions(
    episodes: list[dict[str, Any]],
    fixture: dict[str, Any],
    arm: str,
    *,
    delay: int = 0,
) -> tuple[list[int], int]:
    consumers = int(fixture["n_consumers"])
    predictions = []
    shuffled_messages = [
        episodes[(index + 1) % len(episodes)]["signals"][
            episodes[(index + 1) % len(episodes)]["cue_specialist"]
        ]
        for index in range(len(episodes))
    ]
    for episode_index, episode in enumerate(episodes):
        message = episode["signals"][episode["cue_specialist"]]
        for consumer in range(consumers):
            if arm in {
                "limited-broadcast",
                "unrestricted-bus",
                "unrestricted-bus-lesion",
                "restore-broadcast",
            }:
                prediction = message
            elif arm == "lesion-broadcast":
                prediction = episode["fallback"][consumer]
            elif arm == "delay-broadcast":
                prediction = message if consumer >= delay else episode["fallback"][consumer]
            elif arm == "message-shuffled":
                prediction = shuffled_messages[episode_index]
            elif arm == "matched-dense-state":
                prediction = int(sum(episode["signals"]) >= (len(episode["signals"]) / 2))
            elif arm == "feed-forward-depth":
                prediction = episode["fallback"][consumer]
            elif arm == "independent-specialists":
                prediction = episode["signals"][consumer % len(episode["signals"])]
            elif arm == "equal-flop-routing":
                route = (episode["episode_index"] + consumer) % len(episode["signals"])
                prediction = episode["signals"][route]
            else:
                raise ValueError(f"unsupported broadcast arm {arm!r}")
            predictions.append(prediction)
    budget = int(fixture["matched_flop_budget_per_episode"])
    checksum = _padding_checksum(len(episodes), arm + f":{delay}", 16, budget)
    return predictions, checksum


def _evaluate_once(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    episodes = _episodes(seed, fixture)
    consumers = int(fixture["n_consumers"])
    targets = [episode["target"] for episode in episodes for _ in range(consumers)]
    arms = (
        "limited-broadcast",
        "unrestricted-bus",
        "unrestricted-bus-lesion",
        "lesion-broadcast",
        "restore-broadcast",
        "message-shuffled",
        "matched-dense-state",
        "feed-forward-depth",
        "independent-specialists",
        "equal-flop-routing",
    )
    predictions: dict[str, list[int]] = {}
    checksums: dict[str, int] = {}
    for arm in arms:
        predictions[arm], checksums[arm] = _predictions(episodes, fixture, arm)
    delay_predictions = {}
    for delay in range(1, consumers):
        values, checksum = _predictions(episodes, fixture, "delay-broadcast", delay=delay)
        delay_predictions[str(delay)] = values
        checksums[f"delay-broadcast:{delay}"] = checksum
    scores = {arm: _accuracy(targets, values) for arm, values in predictions.items()}
    delay_scores = {key: _accuracy(targets, values) for key, values in delay_predictions.items()}
    clean = scores["limited-broadcast"]
    lesion = scores["lesion-broadcast"]
    restored = scores["restore-broadcast"]
    lesion_drop = clean - lesion
    restoration_recovery = 1.0 if lesion_drop == 0 else (restored - lesion) / lesion_drop
    max_delay = max(int(value) for value in delay_scores)
    delay_slope = (clean - delay_scores[str(max_delay)]) / max_delay
    budget = int(fixture["matched_flop_budget_per_episode"])
    cost_ledger = {
        arm: {
            "modelled_flops_per_episode": budget,
            "decision_flops": 16,
            "deterministic_padding_flops": budget - 16,
            "padding_checksum": checksum,
        }
        for arm, checksum in checksums.items()
    }
    comparator_names = (
        "unrestricted-bus",
        "matched-dense-state",
        "feed-forward-depth",
        "independent-specialists",
        "equal-flop-routing",
    )
    best_comparator = max(scores[name] for name in comparator_names)
    area = (0.5 * (scores["feed-forward-depth"] + clean) + 0.5 * (clean + scores["unrestricted-bus"])) / 2
    necessity_band = [float(value) for value in config["experiments"][EXPERIMENT_IDS[0]]["calibration_band"]]
    sufficiency_band = [
        float(value) for value in config["experiments"][EXPERIMENT_IDS[1]]["calibration_band"]
    ]
    return {
        "contract_fixture_sha256": {
            "necessity": make_broadcast_contract("necessity", seed).sha256,
            "sufficiency": make_broadcast_contract("sufficiency", seed).sha256,
        },
        "scores": {key: round(value, 8) for key, value in scores.items()},
        "necessity": {
            "lesion_drop": round(lesion_drop, 8),
            "restoration_recovery": round(restoration_recovery, 8),
            "delay_scores": {key: round(value, 8) for key, value in delay_scores.items()},
            "delay_slope": round(delay_slope, 8),
            "message_shuffle_drop": round(clean - scores["message-shuffled"], 8),
            "delta_vs_unrestricted_bus": round(clean - scores["unrestricted-bus"], 8),
            "unrestricted_bus_lesion_drop": round(
                scores["unrestricted-bus"] - scores["unrestricted-bus-lesion"], 8
            ),
            "difficulty_calibration": {
                "band": necessity_band,
                "off_floor_and_ceiling": necessity_band[0] <= clean <= necessity_band[1],
            },
        },
        "sufficiency": {
            "shared_task_score": round(clean, 8),
            "delta_vs_best_comparator": round(clean - best_comparator, 8),
            "capacity_frontier_area": round(area, 8),
            "best_comparator_score": round(best_comparator, 8),
            "tie_with_unrestricted_bus": clean == scores["unrestricted-bus"],
            "tie_is_null": clean == best_comparator,
            "difficulty_calibration": {
                "band": sufficiency_band,
                "off_floor_and_ceiling": sufficiency_band[0] <= clean <= sufficiency_band[1],
            },
        },
        "compute_match": {
            "cost_convention": "one scalar comparison, copy, or padding update counts as one modelled FLOP",
            "all_arms_exact_budget": len({row["modelled_flops_per_episode"] for row in cost_ledger.values()})
            == 1,
            "cost_ledger": cost_ledger,
        },
        "heldout_episodes": len(episodes),
        "independent_queries": len(targets),
    }


def evaluate_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    first = _evaluate_once(seed, config)
    second = _evaluate_once(seed, config)
    first["exact_replay"] = canonical_sha256(first) == canonical_sha256(second)
    return {"seed": seed, "result": first, "unit_sha256": canonical_sha256(first)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _aggregate(units: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    results = [unit["result"] for unit in units]
    f36 = config["experiments"][EXPERIMENT_IDS[0]]
    f37 = config["experiments"][EXPERIMENT_IDS[1]]
    necessity_positive = all(
        float(row["necessity"]["lesion_drop"]) > float(f36["minimum_lesion_drop"])
        and float(row["necessity"]["restoration_recovery"]) >= float(f36["minimum_restoration_fraction"])
        and float(row["necessity"]["delay_slope"]) > float(f36["minimum_delay_slope"])
        and float(row["necessity"]["message_shuffle_drop"]) > float(f36["minimum_shuffle_drop"])
        and float(row["necessity"]["unrestricted_bus_lesion_drop"]) == 0.0
        and row["necessity"]["difficulty_calibration"]["off_floor_and_ceiling"] is True
        and row["compute_match"]["all_arms_exact_budget"] is True
        and row["exact_replay"] is True
        for row in results
    )
    sufficiency_positive = all(
        float(row["sufficiency"]["delta_vs_best_comparator"]) > float(f37["minimum_margin"])
        and row["sufficiency"]["tie_is_null"] is False
        and row["sufficiency"]["difficulty_calibration"]["off_floor_and_ceiling"] is True
        and row["compute_match"]["all_arms_exact_budget"] is True
        and row["exact_replay"] is True
        for row in results
    )
    return {
        EXPERIMENT_IDS[0]: {
            "mean_lesion_drop": round(_mean([float(row["necessity"]["lesion_drop"]) for row in results]), 8),
            "mean_delay_slope": round(_mean([float(row["necessity"]["delay_slope"]) for row in results]), 8),
            "mean_shuffle_drop": round(
                _mean([float(row["necessity"]["message_shuffle_drop"]) for row in results]), 8
            ),
            "programmatic_favorable": necessity_positive,
            "verdict": "favorable-toy-pattern-pending-independent-verification"
            if necessity_positive
            else "null",
        },
        EXPERIMENT_IDS[1]: {
            "mean_shared_task_score": round(
                _mean([float(row["sufficiency"]["shared_task_score"]) for row in results]), 8
            ),
            "mean_delta_vs_best_comparator": round(
                _mean([float(row["sufficiency"]["delta_vs_best_comparator"]) for row in results]), 8
            ),
            "all_units_tie_is_null": all(row["sufficiency"]["tie_is_null"] is True for row in results),
            "programmatic_favorable": sufficiency_positive,
            "verdict": "favorable-toy-pattern-pending-independent-verification"
            if sufficiency_positive
            else "null",
        },
    }


def build_receipt(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    units = [evaluate_seed(int(seed), config) for seed in config["independent_units"]["seeds"]]
    aggregate = _aggregate(units, config)
    registry_bindings = _registry_bindings()
    per_experiment_nulls = {
        experiment_id: str(registry_bindings[experiment_id]["null_hypothesis"])
        for experiment_id in EXPERIMENT_IDS
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "R1 deterministic programmatic toy execution",
        "scientific_capability_claim": False,
        "config": _file_receipt(str(config_path.relative_to(REPO_ROOT))),
        "source_receipts": [_file_receipt(path) for path in SOURCE_PATHS],
        "registry_bindings": registry_bindings,
        "null_contract": {
            "aggregate": config["null_hypothesis"],
            "per_experiment": per_experiment_nulls,
            "per_experiment_sha256": canonical_sha256(per_experiment_nulls),
        },
        "independent_units": units,
        "aggregate": aggregate,
        "favorable_experiments_requiring_fresh_verification": sorted(
            experiment_id for experiment_id, row in aggregate.items() if row["programmatic_favorable"] is True
        ),
        "limitations": [
            "the binary shared-message bed is a structural fixture, not a natural shared task",
            "matched compute uses a declared scalar-operation convention with deterministic padding",
            "the exact unrestricted-bus tie makes f37 a null and cannot support sufficiency",
            "a favorable necessity pattern remains programmatic mechanics only",
        ],
        "resource_envelope": copy.deepcopy(config["resource_envelope"]),
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def write_receipt(output: Path = DEFAULT_OUTPUT, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    receipt = build_receipt(config_path)
    _atomic_json(output, receipt)
    return receipt


def assert_receipt(payload: dict[str, Any]) -> None:
    if payload.get("schema") != RECEIPT_SCHEMA or payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("integration broadcast receipt schema or claim scope drift")
    expected = canonical_sha256({key: value for key, value in payload.items() if key != "payload_sha256"})
    if payload.get("payload_sha256") != expected:
        raise ValueError("integration broadcast receipt payload digest mismatch")
    if payload.get("scientific_capability_claim") is not False:
        raise ValueError("integration broadcast toy receipt cannot make a capability claim")
    null_contract = payload.get("null_contract", {})
    per_experiment = null_contract.get("per_experiment", {})
    if (
        not isinstance(null_contract.get("aggregate"), str)
        or not null_contract["aggregate"].strip()
        or set(per_experiment) != set(EXPERIMENT_IDS)
        or null_contract.get("per_experiment_sha256") != canonical_sha256(per_experiment)
    ):
        raise ValueError("integration broadcast receipt null contract is incomplete")
    if payload.get("aggregate", {}).get(EXPERIMENT_IDS[1], {}).get("all_units_tie_is_null") is not True:
        raise ValueError("f37 exact tie must remain an explicit null")
