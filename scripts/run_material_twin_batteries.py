#!/usr/bin/env python
"""Execute the CPU-only F61 to F64 material twin batteries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mop.config import REPO_ROOT
from mop.experiments.expansion_harness import CLAIM_SCOPE
from mop.studies.material_twin_scaffold import (
    CONTROL_FAMILIES,
    DRIFT_CONTROLS,
    DRIFT_METRICS,
    MATERIAL_TWIN_SCHEMA,
    REPAIR_POLICIES,
    TOY_PRIORS,
    DriftAdaptationContract,
    LeakyEchoStateReservoir,
    LesionSpec,
    Lineage,
    NativeDynamicsValueContract,
    TwinInterfaceContract,
    build_default_control_set,
    default_damage_repair_contract,
    validate_twin_interface,
)
from mop.substrate.events import canonical_sha256

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "mop-material-twin-batteries/v1"
PRIMARY_SEEDS = (19, 41, 73)
UNITS = 12
SESOI = 0.02
IMPLEMENTATION_PATHS = (
    "registry/experiments.yaml",
    "src/mop/studies/material_twin_scaffold.py",
    "scripts/run_material_twin_batteries.py",
    "scripts/verify_material_twin_batteries.py",
)
FloatArray = NDArray[np.float64]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(array: FloatArray) -> str:
    normalized = np.asarray(array, dtype="<f8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _drive(seed: int, label: int) -> FloatArray:
    return np.random.default_rng(seed * 1009 + label).normal(0.0, 0.3, size=UNITS)


def _similarity(reference: FloatArray, observed: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference)) + 1e-12
    return float(np.clip(1.0 - np.linalg.norm(reference - observed) / denominator, 0.0, 1.0))


def _f61_run(prior_cls: type, seed: int) -> dict[str, Any]:
    def execute() -> tuple[FloatArray, list[float], dict[str, Any]]:
        twin = prior_cls(units=UNITS, seed=seed)
        validate_twin_interface(twin)
        twin.excite(_drive(seed, 1))
        energy_trace = [twin.energy()]
        for _ in range(4):
            twin.evolve(1)
            energy_trace.append(twin.energy())
        readout = twin.readout()
        return readout, energy_trace, twin.lineage().payload()

    first, energy, lineage = execute()
    repeated, repeated_energy, repeated_lineage = execute()
    random_control = LeakyEchoStateReservoir(units=UNITS, seed=seed + 10000)
    random_control.excite(_drive(seed, 1))
    random_control.evolve(4)
    random_readout = random_control.readout()
    shape_rejected = False
    try:
        prior_cls(units=UNITS, seed=seed).excite(np.zeros(UNITS - 1, dtype=np.float64))
    except ValueError:
        shape_rejected = True
    lineage_rejected = False
    try:
        Lineage(schema=MATERIAL_TWIN_SCHEMA, ops=tuple(lineage["ops"]), head_digest="0" * 64)
    except ValueError:
        lineage_rejected = True
    metrics = {
        "interface_conformance_rate": 1.0,
        "seed_reproducibility_rate": float(
            np.array_equal(first, repeated) and energy == repeated_energy
        ),
        "lineage_digest_stability": float(lineage == repeated_lineage),
        "energy_accounting_monotonicity": float(
            all(later >= earlier for earlier, later in zip(energy, energy[1:], strict=False))
            and energy[-1] > energy[0]
        ),
    }
    controls = {
        "random-reservoir-arm": _array_sha(first) != _array_sha(random_readout),
        "shape-mismatch-reject": shape_rejected,
        "tampered-lineage-reject": lineage_rejected,
    }
    return {
        "prior": str(cast(Any, prior_cls).kind),
        "seed": seed,
        "readout": first.tolist(),
        "readout_sha256": _array_sha(first),
        "energy_trace": energy,
        "lineage": lineage,
        "lineage_sha256": canonical_sha256(lineage),
        "metrics": metrics,
        "controls": controls,
        "all_metrics_pass": all(value == 1.0 for value in metrics.values()),
        "all_controls_pass": all(controls.values()),
    }


def _ridge(train: FloatArray, target: FloatArray, test: FloatArray) -> FloatArray:
    regularizer = 1e-3 * np.eye(train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(train.T @ train + regularizer, train.T @ target)
    return test @ weights


def _control_features(
    family: str, values: FloatArray, seed: int, option: float
) -> FloatArray:
    width = UNITS
    if family == "fourier-features":
        frequencies = np.arange(1, width // 2 + 1, dtype=np.float64) * option
        return np.column_stack(
            [
                *[np.sin(freq * values) for freq in frequencies],
                *[np.cos(freq * values) for freq in frequencies],
            ]
        )
    if family == "associative-memory":
        centers = np.linspace(-1.0, 1.0, width)
        return np.exp(-option * (values[:, None] - centers[None, :]) ** 2)
    state = np.zeros(width, dtype=np.float64)
    rows: list[FloatArray] = []
    rng = np.random.default_rng(seed + sum(ord(char) for char in family))
    if family == "tuned-rnn":
        matrix = np.roll(np.eye(width), 1, axis=1) * option
        input_weights = np.linspace(0.1, 0.9, width)

        def update(old: FloatArray, value: float) -> FloatArray:
            return np.tanh(matrix @ old + input_weights * value)

    elif family == "state-space-model":
        decays = np.linspace(option, min(option + 0.35, 0.98), width)
        gains = np.linspace(0.2, 1.0, width)

        def update(old: FloatArray, value: float) -> FloatArray:
            return decays * old + gains * value

    elif family == "random-reservoir":
        matrix = rng.normal(size=(width, width))
        radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        matrix = matrix * (option / radius)
        input_weights = rng.normal(0.0, 0.2, size=width)

        def update(old: FloatArray, value: float) -> FloatArray:
            return np.tanh(matrix @ old + input_weights * value)

    else:
        raise ValueError(f"unknown control family {family!r}")
    for value in values:
        state = update(state, float(value))
        rows.append(state.copy())
    return np.asarray(rows, dtype=np.float64)


def _material_features(prior_cls: type, values: FloatArray, seed: int) -> FloatArray:
    twin = prior_cls(units=UNITS, seed=seed)
    pattern = np.linspace(0.5, 1.5, UNITS)
    rows: list[FloatArray] = []
    for value in values:
        twin.excite((pattern * value).astype(np.float64))
        twin.evolve(1)
        rows.append(twin.readout())
    return np.asarray(rows, dtype=np.float64)


def _score(target: FloatArray, prediction: FloatArray) -> float:
    mse = float(np.mean((target - prediction) ** 2))
    variance = float(np.var(target)) + 1e-12
    return float(1.0 / (1.0 + mse / variance))


def _f62_run(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_values = rng.uniform(-1.0, 1.0, size=96)
    test_values = rng.uniform(-1.0, 1.0, size=48)
    train_noise = rng.normal(0.0, 0.18, size=train_values.size)
    test_noise = rng.normal(0.0, 0.18, size=test_values.size)
    train_target = np.sin(3.0 * train_values) + 0.35 * np.cos(5.0 * train_values) + train_noise
    test_target = np.sin(3.0 * test_values) + 0.35 * np.cos(5.0 * test_values) + test_noise
    option_grids = {
        "tuned-rnn": (0.35, 0.60, 0.85),
        "state-space-model": (0.20, 0.45, 0.70),
        "random-reservoir": (0.40, 0.65, 0.90),
        "fourier-features": (0.75, 1.00, 1.25),
        "associative-memory": (2.0, 5.0, 9.0),
    }
    split = 72
    control_rows: list[dict[str, Any]] = []
    common_budget = {
        "params": UNITS,
        "flops_per_example": UNITS * UNITS,
        "memory_bytes": UNITS * 8,
        "update_steps": split,
    }
    for family in sorted(CONTROL_FAMILIES):
        candidates: list[tuple[float, float]] = []
        for option in option_grids[family]:
            features = _control_features(family, train_values, seed, option)
            validation_prediction = _ridge(
                features[:split], train_target[:split], features[split:]
            )
            candidates.append((_score(train_target[split:], validation_prediction), option))
        _, chosen = max(candidates, key=lambda row: (row[0], -row[1]))
        train_features = _control_features(family, train_values, seed, chosen)
        test_features = _control_features(family, test_values, seed, chosen)
        prediction = _ridge(train_features, train_target, test_features)
        control_rows.append(
            {
                "arm": family,
                "chosen_tuning_value": chosen,
                "score": _score(test_target, prediction),
                "prediction_sha256": _array_sha(prediction),
                "matched_budget": common_budget,
            }
        )
    material_rows: list[dict[str, Any]] = []
    for prior_cls in TOY_PRIORS:
        train_features = _material_features(prior_cls, train_values, seed)
        test_features = _material_features(prior_cls, test_values, seed)
        prediction = _ridge(train_features, train_target, test_features)
        material_rows.append(
            {
                "arm": prior_cls.kind,
                "score": _score(test_target, prediction),
                "prediction_sha256": _array_sha(prediction),
                "matched_budget": common_budget,
            }
        )
    best_control = max(control_rows, key=lambda row: row["score"])
    best_material = max(material_rows, key=lambda row: row["score"])
    margin = float(best_material["score"] - best_control["score"])
    return {
        "seed": seed,
        "task": "noisy held-out nonlinear scalar regression",
        "train_target_sha256": _array_sha(train_target),
        "test_target_sha256": _array_sha(test_target),
        "control_arms": control_rows,
        "material_arms": material_rows,
        "best_control": best_control["arm"],
        "best_control_score": best_control["score"],
        "best_material": best_material["arm"],
        "best_material_score": best_material["score"],
        "retained_win_margin": margin,
        "tie_or_below_sesoi": margin <= SESOI,
        "native_dynamics_win": margin > SESOI,
        "score_ceilinged": max(
            row["score"] for row in [*control_rows, *material_rows]
        )
        >= 0.995,
    }


def _pristine_probe(prior_cls: type, seed: int, drive: FloatArray) -> FloatArray:
    twin = prior_cls(units=UNITS, seed=seed)
    twin.excite(drive)
    twin.evolve(4)
    return twin.readout()


def _f63_run(prior_cls: type, seed: int) -> dict[str, Any]:
    novel = _drive(seed, 63)
    reference = _pristine_probe(prior_cls, seed, novel)
    drift_epochs = 200

    no_adapt = prior_cls(units=UNITS, seed=seed)
    no_adapt.apply_drift(drift_epochs)
    no_adapt.excite(novel)
    no_adapt.evolve(4)
    no_adapt_score = _similarity(reference, no_adapt.readout())

    adapted = prior_cls(units=UNITS, seed=seed)
    adapted.apply_drift(drift_epochs)
    adapted.excite(novel)
    adapted.evolve(4)
    adapted.adapt(reference)
    adapted_score = _similarity(reference, adapted.readout())

    oracle = _similarity(reference, _pristine_probe(prior_cls, seed, novel))
    full = prior_cls(units=UNITS, seed=seed)
    full.apply_drift(drift_epochs)
    full.excite(novel)
    full.evolve(4)
    for _ in range(20):
        full.adapt(reference)
    full_score = _similarity(reference, full.readout())
    scores = {
        "no-adapt": no_adapt_score,
        "adaptation": adapted_score,
        "oracle-reset": oracle,
        "full-retraining": full_score,
    }
    metrics = {
        "drift_rate": 1.0 / drift_epochs,
        "adaptation_lag": 1.0,
        "retained_function": adapted_score,
        "adaptation_energy": 1.0,
    }
    success = adapted_score > no_adapt_score + SESOI and metrics["adaptation_energy"] < 20.0
    return {
        "prior": str(cast(Any, prior_cls).kind),
        "seed": seed,
        "drift_epochs": drift_epochs,
        "novel_input_sha256": _array_sha(novel),
        "reference_sha256": _array_sha(reference),
        "controls": scores,
        "metrics": metrics,
        "margin_over_no_adapt": adapted_score - no_adapt_score,
        "energy_controls": {"adaptation": 1.0, "oracle-reset": 4.0, "full-retraining": 20.0},
        "adaptation_success": success,
    }


def _f64_policy(prior_cls: type, seed: int, policy: str) -> dict[str, Any]:
    calibration = _drive(seed, 64)
    novel = _drive(seed, 65)
    reference_twin = prior_cls(units=UNITS, seed=seed)
    reference_twin.excite(calibration)
    reference_twin.evolve(3)
    pre_damage = reference_twin.readout()
    indices = tuple(int(index) for index in np.argsort(np.abs(pre_damage))[-3:])
    reference_twin.excite(novel)
    reference_twin.evolve(4)
    reference = reference_twin.readout()

    twin = prior_cls(units=UNITS, seed=seed)
    twin.excite(calibration)
    twin.evolve(3)
    lesion = LesionSpec(unit_indices=indices, fraction=0.25, seed=seed, selective=True)
    twin.apply_damage(lesion)
    twin.excite(novel)
    twin.evolve(1)
    damaged = twin.readout()
    energy_before_repair = twin.energy()
    twin.repair(policy)
    twin.excite(novel)
    twin.evolve(4)
    restored = twin.readout()
    restored_score = _similarity(reference, restored)
    twin.adapt(reference)
    plasticity = _similarity(reference, twin.readout())
    repair_energy = max(0.0, twin.energy() - energy_before_repair)
    normative_cost = {
        "fixed-final": 0.0,
        "restart": 4.0,
        "spare": 2.0,
        "random": 2.0,
        "full-retraining": 20.0,
    }[policy]
    return {
        "policy": policy,
        "repair_time": normative_cost,
        "restored_function": restored_score,
        "future_plasticity": plasticity,
        "repair_energy": repair_energy + normative_cost,
        "topology_change": float(policy in {"spare", "random", "full-retraining"}),
        "damage_delta": 1.0 - _similarity(reference, damaged),
        "lineage_sha256": twin.lineage().head_digest,
    }


def _f64_run(prior_cls: type, seed: int) -> dict[str, Any]:
    rows = [_f64_policy(prior_cls, seed, policy) for policy in REPAIR_POLICIES]
    by_policy = {row["policy"]: row for row in rows}
    candidate = by_policy["spare"]
    restart = by_policy["restart"]
    fixed = by_policy["fixed-final"]
    full = by_policy["full-retraining"]
    margin_restart = candidate["restored_function"] - restart["restored_function"]
    success = (
        candidate["restored_function"] > fixed["restored_function"] + SESOI
        and margin_restart > SESOI
        and candidate["repair_energy"] < full["repair_energy"]
    )
    return {
        "prior": str(cast(Any, prior_cls).kind),
        "seed": seed,
        "selective_lesion_fraction": 0.25,
        "calibration_input_sha256": _array_sha(_drive(seed, 64)),
        "novel_input_sha256": _array_sha(_drive(seed, 65)),
        "inputs_are_distinct": _array_sha(_drive(seed, 64)) != _array_sha(_drive(seed, 65)),
        "policies": rows,
        "candidate_policy": "spare",
        "margin_over_fixed_final": (
            candidate["restored_function"] - fixed["restored_function"]
        ),
        "margin_over_restart": margin_restart,
        "tie_with_restart": abs(margin_restart) <= SESOI,
        "repair_success": success,
        "result": "favorable" if success else "null",
    }


def build_receipt() -> dict[str, Any]:
    interface = TwinInterfaceContract()
    controls = build_default_control_set()
    native_contract = NativeDynamicsValueContract(
        schema=MATERIAL_TWIN_SCHEMA,
        controls=tuple(sorted(CONTROL_FAMILIES)),
        matched_cost_required=True,
        retain_only=("cross-task-future-learnability", "capability-density"),
        replication_min=3,
    )
    drift_contract = DriftAdaptationContract(
        schema=MATERIAL_TWIN_SCHEMA,
        drift_model="multiplicative parameter decay over 200 epochs",
        adaptation_policy="one-step linear readout refit",
        controls=DRIFT_CONTROLS,
        metrics=DRIFT_METRICS,
    )
    damage_contract = default_damage_repair_contract()
    f61_units = [_f61_run(prior, seed) for seed in PRIMARY_SEEDS for prior in TOY_PRIORS]
    f62_units = [_f62_run(seed) for seed in PRIMARY_SEEDS]
    f63_units = [_f63_run(prior, seed) for seed in PRIMARY_SEEDS for prior in TOY_PRIORS]
    f64_units = [_f64_run(prior, seed) for seed in PRIMARY_SEEDS for prior in TOY_PRIORS]
    f61_pass = all(row["all_metrics_pass"] and row["all_controls_pass"] for row in f61_units)
    f62_win = all(row["native_dynamics_win"] for row in f62_units)
    f62_ceilinged = any(row["score_ceilinged"] for row in f62_units)
    f63_pass = all(row["adaptation_success"] for row in f63_units)
    f64_pass = all(row["repair_success"] for row in f64_units)
    score_values = [
        row["score"]
        for unit in f62_units
        for row in [*unit["control_arms"], *unit["material_arms"]]
    ]
    f63_min_separation = min(row["margin_over_no_adapt"] for row in f63_units)
    f64_min_damage = min(
        policy["damage_delta"] for row in f64_units for policy in row["policies"]
    )
    f64_min_fixed_separation = min(row["margin_over_fixed_final"] for row in f64_units)
    f63_calibrated = f63_min_separation > SESOI
    f64_calibrated = f64_min_damage > SESOI and f64_min_fixed_separation > SESOI
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "deterministic programmatic pilot over toy numpy priors",
        "status": "complete",
        "implementation": [
            {"path": path, "sha256": _sha_file(REPO_ROOT / path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "preregistration": {
            "primary_seeds": list(PRIMARY_SEEDS),
            "smallest_effect": SESOI,
            "tie_rule": "a margin at or below the SESOI is a null",
            "stop_rule": "three independent seeds across every toy prior, no adaptive extension",
        },
        "difficulty_calibration": {
            "f61_nontrivial_control_rejections": f61_pass,
            "f62_score_min": min(score_values),
            "f62_score_max": max(score_values),
            "f62_ceilinged": f62_ceilinged,
            "f63_control_separation": f63_min_separation,
            "f63_calibrated": f63_calibrated,
            "f64_damage_delta_min": f64_min_damage,
            "f64_fixed_final_separation_min": f64_min_fixed_separation,
            "f64_calibrated": f64_calibrated,
            "ceilinged_tie_promoted": False,
            "calibrated": f61_pass and not f62_ceilinged and f63_calibrated and f64_calibrated,
        },
        "f61": {
            "experiment_id": "f61_physical_reservoir_digital_twin",
            "null_hypothesis": "a common interface or deterministic accounting fails for a toy prior",
            "interface_contract": interface.payload(),
            "interface_contract_sha256": interface.interface_digest(),
            "units": f61_units,
            "result": "programmatic-mechanics-pass" if f61_pass else "null",
            "promotion": False,
        },
        "f62": {
            "experiment_id": "f62_material_native_dynamics_value",
            "null_hypothesis": (
                "no toy material prior beats every tuned matched-budget conventional control by the SESOI"
            ),
            "contract": native_contract.payload(),
            "control_declarations": controls.payload(),
            "units": f62_units,
            "replicated_native_dynamics_win": f62_win and not f62_ceilinged,
            "result": "favorable" if f62_win and not f62_ceilinged else "null",
            "strongest_control": max(
                (
                    (row["best_control_score"], row["best_control"])
                    for row in f62_units
                ),
                key=lambda item: item[0],
            )[1],
            "promotion": False,
        },
        "f63": {
            "experiment_id": "f63_drift_and_aging_adaptation",
            "null_hypothesis": (
                "one-step adaptation does not clear the SESOI over no-adapt at lower declared "
                "energy than full retraining"
            ),
            "contract": drift_contract.payload(),
            "units": f63_units,
            "replicated_adaptation_success": f63_pass,
            "result": "favorable-programmatic-pilot" if f63_pass else "null",
            "promotion": False,
        },
        "f64": {
            "experiment_id": "f64_damage_and_reattachment_recovery",
            "null_hypothesis": (
                "spare repair does not beat both fixed-final and restart by the SESOI under novel input"
            ),
            "contract": damage_contract.payload(),
            "units": f64_units,
            "replicated_repair_success": f64_pass,
            "result": "favorable-programmatic-pilot" if f64_pass else "null",
            "strongest_control": "restart",
            "promotion": False,
        },
        "scientific_scope": {
            "physical_specimens_tested": False,
            "biological_equivalence_claimed": False,
            "native_dynamics_advantage_claimed": False,
            "toy_programmatic_result_only": True,
        },
    }
    core["core_payload_sha256"] = canonical_sha256(core)

    from scripts.verify_material_twin_batteries import verify_receipt

    verification = verify_receipt(core, check_live_files=True, run_mutations=True)
    if verification["verified"] is not True:
        raise RuntimeError("independent material verification failed: " + "; ".join(verification["errors"]))
    receipt = {**core, "independent_verifier": verification}
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "proof" / "F61_F64_MATERIAL_TWIN_RUN.json")
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = build_receipt()
    output = Path(args.out)
    _atomic_write(output, receipt)
    print(
        f"wrote {output}: f62={receipt['f62']['result']}, f63={receipt['f63']['result']}, "
        f"f64={receipt['f64']['result']}, payload={receipt['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
