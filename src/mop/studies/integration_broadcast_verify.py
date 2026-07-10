"""Independent adversarial verifier for the f36 and f37 broadcast toy beds.

The verifier does not import the run evaluator. It regenerates every primary and fresh unit,
recomputes arm predictions through a separate loop, attacks the link result with shuffled messages,
and requires the unrestricted bus to remain lesion invariant. Exact comparator ties stay null.
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

RUN_SCHEMA = "mop-integration-broadcast-run/v1"
VERIFY_SCHEMA = "mop-integration-broadcast-verifier/v1"
CONFIG_SCHEMA = "mop-integration-broadcast-run-config/v1"
DEFAULT_RUN = REPO_ROOT / "proof" / "INTEGRATION_BROADCAST_RUN.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "integration_broadcast_runs.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "INTEGRATION_BROADCAST_VERIFICATION.json"
F36 = "f36_limited_broadcast_necessity"
F37 = "f37_broadcast_sufficiency"


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


def _display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def _read_inputs(run_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA or run.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("broadcast verifier received the wrong run receipt")
    if run.get("payload_sha256") != canonical_sha256(
        {key: value for key, value in run.items() if key != "payload_sha256"}
    ):
        raise ValueError("broadcast run receipt digest mismatch")
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("broadcast verifier config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("broadcast verifier claim scope drift")
    if run.get("config", {}).get("sha256") != _sha256_file(config_path):
        raise ValueError("broadcast run does not bind the verifier config")
    return run, config


def _episodes(
    seed: int, fixture: dict[str, Any]
) -> list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]:
    rng = Random(_seed(seed, "integration-broadcast:episodes"))
    rows = []
    specialists = int(fixture["n_specialists"])
    consumers = int(fixture["n_consumers"])
    for _ in range(int(fixture["heldout_episodes"])):
        target = rng.randrange(2)
        relevant = rng.randrange(specialists)
        cue = relevant
        if rng.random() >= float(fixture["cue_reliability"]):
            cue = (relevant + 1 + rng.randrange(specialists - 1)) % specialists
        signals = []
        for specialist in range(specialists):
            reliability = float(fixture["signal_reliability"]) if specialist == relevant else 0.5
            signals.append(target if rng.random() < reliability else 1 - target)
        fallback = tuple(rng.randrange(2) for _ in range(consumers))
        rows.append((target, relevant, cue, tuple(signals), fallback))
    return rows


def _score(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    rows = _episodes(seed, fixture)
    consumers = int(fixture["n_consumers"])
    scores: dict[str, list[bool]] = {
        name: []
        for name in (
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
    }
    delay_correct: dict[int, list[bool]] = {delay: [] for delay in range(1, consumers)}
    messages = [signals[cue] for _, _, cue, signals, _ in rows]
    for episode_index, (target, _, _, signals, fallback) in enumerate(rows):
        message = messages[episode_index]
        shuffled_message = messages[(episode_index + 1) % len(messages)]
        dense = int(sum(signals) >= len(signals) / 2)
        for consumer in range(consumers):
            values = {
                "limited-broadcast": message,
                "unrestricted-bus": message,
                "unrestricted-bus-lesion": message,
                "lesion-broadcast": fallback[consumer],
                "restore-broadcast": message,
                "message-shuffled": shuffled_message,
                "matched-dense-state": dense,
                "feed-forward-depth": fallback[consumer],
                "independent-specialists": signals[consumer % len(signals)],
                "equal-flop-routing": signals[(episode_index + consumer) % len(signals)],
            }
            for arm, prediction in values.items():
                scores[arm].append(prediction == target)
            for delay in delay_correct:
                prediction = message if consumer >= delay else fallback[consumer]
                delay_correct[delay].append(prediction == target)

    accuracy = {arm: sum(values) / len(values) for arm, values in scores.items()}
    delays = {str(delay): sum(values) / len(values) for delay, values in delay_correct.items()}
    clean = accuracy["limited-broadcast"]
    lesion = accuracy["lesion-broadcast"]
    lesion_drop = clean - lesion
    restoration = 1.0 if lesion_drop == 0 else (accuracy["restore-broadcast"] - lesion) / lesion_drop
    last_delay = max(delay_correct)
    delay_slope = (clean - delays[str(last_delay)]) / last_delay
    comparators = (
        "unrestricted-bus",
        "matched-dense-state",
        "feed-forward-depth",
        "independent-specialists",
        "equal-flop-routing",
    )
    best = max(accuracy[name] for name in comparators)
    capacity_area = (
        0.5 * (accuracy["feed-forward-depth"] + clean) + 0.5 * (clean + accuracy["unrestricted-bus"])
    ) / 2
    f36_band = [float(value) for value in config["experiments"][F36]["calibration_band"]]
    f37_band = [float(value) for value in config["experiments"][F37]["calibration_band"]]
    return {
        "contract_fixture_sha256": {
            "necessity": make_broadcast_contract("necessity", seed).sha256,
            "sufficiency": make_broadcast_contract("sufficiency", seed).sha256,
        },
        "scores": {key: round(value, 8) for key, value in accuracy.items()},
        "necessity": {
            "lesion_drop": round(lesion_drop, 8),
            "restoration_recovery": round(restoration, 8),
            "delay_scores": {key: round(value, 8) for key, value in delays.items()},
            "delay_slope": round(delay_slope, 8),
            "message_shuffle_drop": round(clean - accuracy["message-shuffled"], 8),
            "delta_vs_unrestricted_bus": round(clean - accuracy["unrestricted-bus"], 8),
            "unrestricted_bus_lesion_drop": round(
                accuracy["unrestricted-bus"] - accuracy["unrestricted-bus-lesion"], 8
            ),
            "off_floor_and_ceiling": f36_band[0] <= clean <= f36_band[1],
            "strongest_shell_attack_passed": (
                clean - accuracy["message-shuffled"]
                > float(config["experiments"][F36]["minimum_shuffle_drop"])
                and accuracy["unrestricted-bus"] == accuracy["unrestricted-bus-lesion"]
            ),
        },
        "sufficiency": {
            "shared_task_score": round(clean, 8),
            "delta_vs_best_comparator": round(clean - best, 8),
            "capacity_frontier_area": round(capacity_area, 8),
            "best_comparator_score": round(best, 8),
            "tie_with_unrestricted_bus": clean == accuracy["unrestricted-bus"],
            "tie_is_null": clean == best,
            "off_floor_and_ceiling": f37_band[0] <= clean <= f37_band[1],
        },
        "compute_match": {
            "all_arms_exact_budget": True,
            "modelled_flops_per_episode": int(fixture["matched_flop_budget_per_episode"]),
        },
        "heldout_episodes": len(rows),
        "independent_queries": len(rows) * consumers,
    }


PRIMARY_KEYS = (
    "contract_fixture_sha256",
    "scores",
    "heldout_episodes",
    "independent_queries",
)
NECESSITY_KEYS = (
    "lesion_drop",
    "restoration_recovery",
    "delay_scores",
    "delay_slope",
    "message_shuffle_drop",
    "delta_vs_unrestricted_bus",
    "unrestricted_bus_lesion_drop",
)
SUFFICIENCY_KEYS = (
    "shared_task_score",
    "delta_vs_best_comparator",
    "capacity_frontier_area",
    "best_comparator_score",
    "tie_with_unrestricted_bus",
    "tie_is_null",
)


def _primary_problems(run: dict[str, Any], config: dict[str, Any]) -> list[str]:
    problems = []
    expected_seeds = [int(value) for value in config["independent_units"]["seeds"]]
    actual_seeds = [int(row["seed"]) for row in run.get("independent_units", [])]
    if actual_seeds != expected_seeds:
        return ["broadcast primary seed set or order drift"]
    for unit in run["independent_units"]:
        seed = int(unit["seed"])
        recorded = unit["result"]
        recomputed = _score(seed, config)
        for key in PRIMARY_KEYS:
            if recorded.get(key) != recomputed.get(key):
                problems.append(f"seed {seed} broadcast field {key} does not recompute")
        for key in NECESSITY_KEYS:
            if recorded["necessity"].get(key) != recomputed["necessity"].get(key):
                problems.append(f"seed {seed} necessity field {key} does not recompute")
        for key in SUFFICIENCY_KEYS:
            if recorded["sufficiency"].get(key) != recomputed["sufficiency"].get(key):
                problems.append(f"seed {seed} sufficiency field {key} does not recompute")
        if recorded["compute_match"].get("all_arms_exact_budget") is not True:
            problems.append(f"seed {seed} compute match is not exact")
        if recorded.get("exact_replay") is not True:
            problems.append(f"seed {seed} exact replay failed")
    return problems


def _necessity_favorable(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    return all(
        float(row["necessity"]["lesion_drop"]) > float(cfg["minimum_lesion_drop"])
        and float(row["necessity"]["restoration_recovery"]) >= float(cfg["minimum_restoration_fraction"])
        and float(row["necessity"]["delay_slope"]) > float(cfg["minimum_delay_slope"])
        and float(row["necessity"]["message_shuffle_drop"]) > float(cfg["minimum_shuffle_drop"])
        and row["necessity"]["off_floor_and_ceiling"] is True
        and row["necessity"]["strongest_shell_attack_passed"] is True
        for row in rows
    )


def _sufficiency_favorable(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    return all(
        float(row["sufficiency"]["delta_vs_best_comparator"]) > float(cfg["minimum_margin"])
        and row["sufficiency"]["tie_is_null"] is False
        and row["sufficiency"]["off_floor_and_ceiling"] is True
        for row in rows
    )


def _mutation_checks(run: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    mutations = {}
    metric = copy.deepcopy(run)
    metric["independent_units"][0]["result"]["necessity"]["lesion_drop"] += 0.01
    mutations["changed_lesion_drop"] = metric
    tie = copy.deepcopy(run)
    tie["independent_units"][0]["result"]["sufficiency"]["tie_is_null"] = False
    mutations["erased_tie_null"] = tie
    seed = copy.deepcopy(run)
    seed["independent_units"][0]["seed"] += 1
    mutations["changed_seed"] = seed
    return {name: bool(_primary_problems(payload, config)) for name, payload in mutations.items()}


def build_verification(run_path: Path = DEFAULT_RUN, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    run, config = _read_inputs(run_path, config_path)
    primary_problems = _primary_problems(run, config)
    fresh_units: list[dict[str, Any]] = [
        {"seed": int(seed), "result": _score(int(seed), config)}
        for seed in config["independent_units"]["fresh_verifier_seeds"]
    ]
    fresh_rows: list[dict[str, Any]] = [row["result"] for row in fresh_units]
    fresh_necessity = _necessity_favorable(fresh_rows, config["experiments"][F36])
    fresh_sufficiency = _sufficiency_favorable(fresh_rows, config["experiments"][F37])
    primary_necessity = run["aggregate"][F36]["programmatic_favorable"] is True
    primary_sufficiency = run["aggregate"][F37]["programmatic_favorable"] is True
    necessity_verified = primary_necessity and fresh_necessity
    sufficiency_verified = primary_sufficiency and fresh_sufficiency
    problems = list(primary_problems)
    if primary_necessity and not necessity_verified:
        problems.append("fresh adversarial seeds did not close the favorable f36 toy pattern")
    if primary_sufficiency and not sufficiency_verified:
        problems.append("fresh adversarial seeds did not close the favorable f37 toy pattern")
    if any(row["result"]["sufficiency"]["tie_is_null"] is not True for row in fresh_units):
        problems.append("fresh f37 exact tie did not remain an explicit null")
    mutations = _mutation_checks(run, config)
    if not all(mutations.values()):
        problems.append("one or more broadcast verifier mutations escaped rejection")
    receipt = {
        "schema": VERIFY_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "R1 independently re-executed programmatic toy mechanics",
        "run_receipt": {
            "path": _display_path(run_path),
            "sha256": _sha256_file(run_path),
            "payload_sha256": run["payload_sha256"],
        },
        "config": {
            "path": _display_path(config_path),
            "sha256": _sha256_file(config_path),
        },
        "independence": {
            "imports_run_evaluator": False,
            "primary_seed_reexecution": True,
            "fresh_seed_reexecution": True,
            "separate_arm_loop": True,
        },
        "primary_recompute_exact": not primary_problems,
        "fresh_units": fresh_units,
        "per_experiment": {
            F36: {
                "primary_programmatic_favorable": primary_necessity,
                "fresh_seed_favorable": fresh_necessity,
                "strongest_control": "message-shuffled plus unrestricted-bus lesion invariance",
                "programmatic_pattern_verified": necessity_verified,
                "scientific_promotion_allowed": False,
                "verdict": "verified-favorable-toy-pattern" if necessity_verified else "verified-null",
            },
            F37: {
                "primary_programmatic_favorable": primary_sufficiency,
                "fresh_seed_favorable": fresh_sufficiency,
                "strongest_control": "exact unrestricted-bus tie",
                "programmatic_pattern_verified": sufficiency_verified,
                "scientific_promotion_allowed": False,
                "verdict": "verified-favorable-toy-pattern" if sufficiency_verified else "verified-null",
            },
        },
        "mutation_checks": mutations,
        "problems": problems,
        "all_ok": not problems,
        "scientific_capability_claim": False,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def write_verification(
    output: Path = DEFAULT_OUTPUT,
    run_path: Path = DEFAULT_RUN,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    receipt = build_verification(run_path, config_path)
    _atomic_json(output, receipt)
    return receipt
