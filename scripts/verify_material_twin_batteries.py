#!/usr/bin/env python
"""Independently replay and attack the F61 to F64 material twin receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mop.config import REPO_ROOT

SCHEMA = "mop-material-twin-batteries/v1"
VERIFIER_SCHEMA = "mop-material-twin-independent-verifier/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
MATERIAL_SCHEMA = "mop-material-twin/v1"
PRIORS = (
    "leaky-echo-state-reservoir",
    "decaying-conductance-map",
    "oscillator-bank",
)
CONTROL_FAMILIES = (
    "associative-memory",
    "fourier-features",
    "random-reservoir",
    "state-space-model",
    "tuned-rnn",
)
REPAIR_POLICIES = ("fixed-final", "restart", "spare", "random", "full-retraining")
F61_METRICS = (
    "interface_conformance_rate",
    "seed_reproducibility_rate",
    "lineage_digest_stability",
    "energy_accounting_monotonicity",
)
F61_CONTROLS = (
    "random-reservoir-arm",
    "shape-mismatch-reject",
    "tampered-lineage-reject",
)
F63_METRICS = ("drift_rate", "adaptation_lag", "retained_function", "adaptation_energy")
F64_METRICS = (
    "repair_time",
    "restored_function",
    "future_plasticity",
    "repair_energy",
    "topology_change",
)
IMPLEMENTATION_PATHS = (
    "registry/experiments.yaml",
    "src/mop/studies/material_twin_scaffold.py",
    "scripts/run_material_twin_batteries.py",
    "scripts/verify_material_twin_batteries.py",
)
FRESH_SEEDS = (211, 223, 227)
SESOI = 0.02
UNITS = 12
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FloatArray = NDArray[np.float64]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_sha(array: FloatArray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= value <= 1.0


def _verify_f61(block: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["f61 block missing"]
    units = block.get("units", [])
    identities = {(row.get("prior"), row.get("seed")) for row in units if isinstance(row, dict)}
    seeds = {row.get("seed") for row in units if isinstance(row, dict)}
    if len(units) != 9 or len(identities) != 9 or len(seeds) != 3:
        errors.append("f61 independent unit coverage drift")
    if {prior for prior, _ in identities} != set(PRIORS):
        errors.append("f61 prior coverage drift")
    for row in units:
        if not isinstance(row, dict):
            errors.append("f61 unit is not a mapping")
            continue
        label = f"{row.get('prior')} seed {row.get('seed')}"
        readout = np.asarray(row.get("readout"), dtype=np.float64)
        if readout.shape != (UNITS,) or not np.isfinite(readout).all():
            errors.append(f"f61 readout invalid for {label}")
        elif row.get("readout_sha256") != _array_sha(readout):
            errors.append(f"f61 readout digest drift for {label}")
        energy = row.get("energy_trace")
        if (
            not isinstance(energy, list)
            or len(energy) < 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in energy)
            or not all(later >= earlier for earlier, later in zip(energy, energy[1:], strict=False))
            or energy[-1] <= energy[0]
        ):
            errors.append(f"f61 energy accounting drift for {label}")
        lineage = row.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("schema") != MATERIAL_SCHEMA:
            errors.append(f"f61 lineage missing for {label}")
        else:
            if lineage.get("head_digest") != _sha(lineage.get("ops")):
                errors.append(f"f61 lineage head drift for {label}")
            if row.get("lineage_sha256") != _sha(lineage):
                errors.append(f"f61 lineage receipt drift for {label}")
        metrics = row.get("metrics", {})
        if set(metrics) != set(F61_METRICS) or not all(value == 1.0 for value in metrics.values()):
            errors.append(f"f61 metric drift for {label}")
        controls = row.get("controls", {})
        if set(controls) != set(F61_CONTROLS) or not all(value is True for value in controls.values()):
            errors.append(f"f61 control drift for {label}")
        if row.get("all_metrics_pass") is not True or row.get("all_controls_pass") is not True:
            errors.append(f"f61 aggregate drift for {label}")
    expected_result = "programmatic-mechanics-pass" if not errors else "null"
    if block.get("result") != expected_result or block.get("promotion") is not False:
        errors.append("f61 result scope drift")
    return errors


def _verify_budget_rows(rows: Any, expected_arms: set[str], label: str) -> tuple[list[str], dict[str, float]]:
    errors: list[str] = []
    if not isinstance(rows, list) or {row.get("arm") for row in rows} != expected_arms:
        return [f"{label} arm coverage drift"], {}
    scores: dict[str, float] = {}
    budgets = []
    for row in rows:
        arm = row.get("arm")
        score = row.get("score")
        if not _finite_probability(score):
            errors.append(f"{label} score invalid for {arm}")
        else:
            scores[str(arm)] = float(score)
        digest = row.get("prediction_sha256")
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            errors.append(f"{label} prediction digest invalid for {arm}")
        budget = row.get("matched_budget")
        if not isinstance(budget, dict) or set(budget) != {
            "params",
            "flops_per_example",
            "memory_bytes",
            "update_steps",
        } or any(not isinstance(value, int) or value <= 0 for value in budget.values()):
            errors.append(f"{label} matched budget invalid for {arm}")
        else:
            budgets.append(budget)
    if budgets and any(row != budgets[0] for row in budgets[1:]):
        errors.append(f"{label} within-family budget drift")
    return errors, scores


def _verify_f62(block: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["f62 block missing"]
    units = block.get("units", [])
    if not isinstance(units, list) or len(units) != 3 or len({row.get("seed") for row in units}) != 3:
        return ["f62 independent unit coverage drift"]
    wins: list[bool] = []
    ceilinged: list[bool] = []
    strongest: list[tuple[float, str]] = []
    for unit in units:
        seed = unit.get("seed")
        control_errors, controls = _verify_budget_rows(
            unit.get("control_arms"), set(CONTROL_FAMILIES), f"f62 controls seed {seed}"
        )
        material_errors, materials = _verify_budget_rows(
            unit.get("material_arms"), set(PRIORS), f"f62 materials seed {seed}"
        )
        errors.extend(control_errors)
        errors.extend(material_errors)
        if not controls or not materials:
            continue
        control_row = max(controls.items(), key=lambda item: item[1])
        material_row = max(materials.items(), key=lambda item: item[1])
        margin = material_row[1] - control_row[1]
        ceiling = max(*controls.values(), *materials.values()) >= 0.995
        if unit.get("best_control") != control_row[0] or not math.isclose(
            unit.get("best_control_score"), control_row[1], rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"f62 strongest control drift at seed {seed}")
        if unit.get("best_material") != material_row[0] or not math.isclose(
            unit.get("best_material_score"), material_row[1], rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"f62 best material drift at seed {seed}")
        if not math.isclose(unit.get("retained_win_margin"), margin, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"f62 margin drift at seed {seed}")
        if unit.get("native_dynamics_win") is not (margin > SESOI):
            errors.append(f"f62 win decision drift at seed {seed}")
        if unit.get("tie_or_below_sesoi") is not (margin <= SESOI):
            errors.append(f"f62 tie rule drift at seed {seed}")
        if unit.get("score_ceilinged") is not ceiling:
            errors.append(f"f62 ceiling decision drift at seed {seed}")
        wins.append(margin > SESOI)
        ceilinged.append(ceiling)
        strongest.append((control_row[1], control_row[0]))
    replicated = len(wins) == 3 and all(wins) and not any(ceilinged)
    if block.get("replicated_native_dynamics_win") is not replicated:
        errors.append("f62 replication aggregate drift")
    if block.get("result") != ("favorable" if replicated else "null"):
        errors.append("f62 null or favorable verdict drift")
    if strongest and block.get("strongest_control") != max(strongest)[1]:
        errors.append("f62 strongest control aggregate drift")
    if block.get("promotion") is not False:
        errors.append("f62 promotion scope drift")
    return errors


def _verify_f63(block: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["f63 block missing"]
    units = block.get("units", [])
    if not isinstance(units, list) or len(units) != 9:
        return ["f63 independent unit coverage drift"]
    successes: list[bool] = []
    for row in units:
        label = f"{row.get('prior')} seed {row.get('seed')}"
        controls = row.get("controls", {})
        metrics = row.get("metrics", {})
        if set(controls) != {"no-adapt", "adaptation", "oracle-reset", "full-retraining"}:
            errors.append(f"f63 control coverage drift for {label}")
            continue
        if set(metrics) != set(F63_METRICS) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()
        ):
            errors.append(f"f63 metric drift for {label}")
            continue
        if not all(_finite_probability(value) for value in controls.values()):
            errors.append(f"f63 control score drift for {label}")
        margin = controls["adaptation"] - controls["no-adapt"]
        success = margin > SESOI and metrics["adaptation_energy"] < row["energy_controls"]["full-retraining"]
        if not math.isclose(row.get("margin_over_no_adapt"), margin, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"f63 margin drift for {label}")
        if row.get("adaptation_success") is not success:
            errors.append(f"f63 decision drift for {label}")
        successes.append(success)
    replicated = len(successes) == 9 and all(successes)
    if block.get("replicated_adaptation_success") is not replicated:
        errors.append("f63 replication aggregate drift")
    expected = "favorable-programmatic-pilot" if replicated else "null"
    if block.get("result") != expected or block.get("promotion") is not False:
        errors.append("f63 result scope drift")
    return errors


def _verify_f64(block: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["f64 block missing"]
    units = block.get("units", [])
    if not isinstance(units, list) or len(units) != 9:
        return ["f64 independent unit coverage drift"]
    successes: list[bool] = []
    for row in units:
        label = f"{row.get('prior')} seed {row.get('seed')}"
        policies = row.get("policies", [])
        if [item.get("policy") for item in policies] != list(REPAIR_POLICIES):
            errors.append(f"f64 repair control coverage drift for {label}")
            continue
        by_policy = {item["policy"]: item for item in policies}
        for policy, metrics in by_policy.items():
            if not set(F64_METRICS) <= set(metrics) or not all(
                isinstance(metrics[name], (int, float)) and math.isfinite(metrics[name])
                for name in F64_METRICS
            ):
                errors.append(f"f64 metric drift for {policy} {label}")
        candidate = by_policy["spare"]
        fixed = by_policy["fixed-final"]
        restart = by_policy["restart"]
        full = by_policy["full-retraining"]
        margin_fixed = candidate["restored_function"] - fixed["restored_function"]
        margin_restart = candidate["restored_function"] - restart["restored_function"]
        success = (
            margin_fixed > SESOI
            and margin_restart > SESOI
            and candidate["repair_energy"] < full["repair_energy"]
        )
        if not math.isclose(row.get("margin_over_fixed_final"), margin_fixed, abs_tol=1e-12):
            errors.append(f"f64 fixed-final margin drift for {label}")
        if not math.isclose(row.get("margin_over_restart"), margin_restart, abs_tol=1e-12):
            errors.append(f"f64 restart margin drift for {label}")
        if row.get("tie_with_restart") is not (abs(margin_restart) <= SESOI):
            errors.append(f"f64 tie rule drift for {label}")
        if row.get("repair_success") is not success:
            errors.append(f"f64 success decision drift for {label}")
        if row.get("result") != ("favorable" if success else "null"):
            errors.append(f"f64 unit verdict drift for {label}")
        if row.get("inputs_are_distinct") is not True:
            errors.append(f"f64 novelty input drift for {label}")
        successes.append(success)
    replicated = len(successes) == 9 and all(successes)
    if block.get("replicated_repair_success") is not replicated:
        errors.append("f64 replication aggregate drift")
    expected = "favorable-programmatic-pilot" if replicated else "null"
    if block.get("result") != expected or block.get("promotion") is not False:
        errors.append("f64 result scope drift")
    if block.get("strongest_control") != "restart":
        errors.append("f64 strongest control drift")
    return errors


def _drive(seed: int, label: int) -> FloatArray:
    return np.random.default_rng(seed * 1009 + label).normal(0.0, 0.3, size=UNITS)


def _similarity(reference: FloatArray, observed: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference)) + 1e-12
    return float(np.clip(1.0 - np.linalg.norm(reference - observed) / denominator, 0.0, 1.0))


def _fresh_challenges() -> tuple[list[dict[str, Any]], list[str]]:
    from mop.studies.material_twin_scaffold import (
        TOY_PRIORS,
        LesionSpec,
        Lineage,
        validate_twin_interface,
    )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for seed in FRESH_SEEDS:
        per_prior: list[dict[str, Any]] = []
        for prior_cls in TOY_PRIORS:
            def interface_run(
                prior_type: type = prior_cls, challenge_seed: int = seed
            ) -> tuple[FloatArray, list[float]]:
                twin = prior_type(units=UNITS, seed=challenge_seed)
                validate_twin_interface(twin)
                twin.excite(_drive(challenge_seed, 1))
                energies = [twin.energy()]
                for _ in range(4):
                    twin.evolve(1)
                    energies.append(twin.energy())
                return twin.readout(), energies

            first, energy = interface_run()
            repeated, repeated_energy = interface_run()
            f61_ok = np.array_equal(first, repeated) and energy == repeated_energy
            lineage_refusal = False
            try:
                Lineage(schema=MATERIAL_SCHEMA, ops=("reset:seed=0",), head_digest="0" * 64)
            except ValueError:
                lineage_refusal = True

            novel = _drive(seed, 63)
            reference_twin = prior_cls(units=UNITS, seed=seed)
            reference_twin.excite(novel)
            reference_twin.evolve(4)
            reference = reference_twin.readout()
            no_adapt = prior_cls(units=UNITS, seed=seed)
            no_adapt.apply_drift(200)
            no_adapt.excite(novel)
            no_adapt.evolve(4)
            no_adapt_score = _similarity(reference, no_adapt.readout())
            adapted = prior_cls(units=UNITS, seed=seed)
            adapted.apply_drift(200)
            adapted.excite(novel)
            adapted.evolve(4)
            adapted.adapt(reference)
            adapted_score = _similarity(reference, adapted.readout())
            f63_ok = adapted_score > no_adapt_score + SESOI

            calibration = _drive(seed, 64)
            source = prior_cls(units=UNITS, seed=seed)
            source.excite(calibration)
            source.evolve(3)
            indices = tuple(int(index) for index in np.argsort(np.abs(source.readout()))[-3:])
            lesion = LesionSpec(indices, 0.25, seed, True)
            repair_scores: dict[str, float] = {}
            for policy in ("restart", "spare"):
                twin = prior_cls(units=UNITS, seed=seed)
                twin.excite(calibration)
                twin.evolve(3)
                twin.apply_damage(lesion)
                twin.repair(policy)
                twin.excite(_drive(seed, 65))
                twin.evolve(4)
                repair_scores[policy] = float(np.linalg.norm(twin.readout()))
            f64_null = abs(repair_scores["spare"] - repair_scores["restart"]) <= SESOI
            prior_row = {
                "prior": prior_cls.kind,
                "f61_bit_exact": f61_ok,
                "f61_tampered_lineage_refused": lineage_refusal,
                "f63_margin": adapted_score - no_adapt_score,
                "f63_adaptation_success": f63_ok,
                "f64_spare_restart_tie": f64_null,
                "all_pass": f61_ok and lineage_refusal and f63_ok and f64_null,
            }
            per_prior.append(prior_row)
            if not prior_row["all_pass"]:
                errors.append(f"fresh material challenge failed for {prior_cls.kind} seed {seed}")
        rows.append({"seed": seed, "priors": per_prior, "all_pass": all(r["all_pass"] for r in per_prior)})
    return rows, errors


def _base_errors(receipt: dict[str, Any], *, check_live_files: bool) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if receipt.get("schema") != SCHEMA:
        errors.append("material receipt schema drift")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        errors.append("material claim scope drift")
    core = {
        key: value
        for key, value in receipt.items()
        if key not in {"core_payload_sha256", "independent_verifier", "payload_sha256"}
    }
    if receipt.get("core_payload_sha256") != _sha(core):
        errors.append("material core payload digest drift")
    implementation = receipt.get("implementation", [])
    if tuple(row.get("path") for row in implementation) != IMPLEMENTATION_PATHS:
        errors.append("material implementation path set drift")
    if check_live_files:
        for row in implementation:
            path = REPO_ROOT / str(row.get("path"))
            if not path.is_file() or _sha_file(path) != row.get("sha256"):
                errors.append(f"material live implementation drift at {row.get('path')}")
    errors.extend(_verify_f61(receipt.get("f61")))
    errors.extend(_verify_f62(receipt.get("f62")))
    errors.extend(_verify_f63(receipt.get("f63")))
    errors.extend(_verify_f64(receipt.get("f64")))
    calibration = receipt.get("difficulty_calibration", {})
    if (
        calibration.get("calibrated") is not True
        or calibration.get("f62_ceilinged") is not False
        or calibration.get("f63_calibrated") is not True
        or calibration.get("f64_calibrated") is not True
        or calibration.get("ceilinged_tie_promoted") is not False
        or not isinstance(calibration.get("f64_damage_delta_min"), (int, float))
        or calibration.get("f64_damage_delta_min") <= 0.0
    ):
        errors.append("material difficulty calibration drift")
    if receipt.get("status") != "complete":
        errors.append("material completion status drift")
    return errors, {
        "f61_unit_count": len(receipt.get("f61", {}).get("units", [])),
        "f62_unit_count": len(receipt.get("f62", {}).get("units", [])),
        "f63_unit_count": len(receipt.get("f63", {}).get("units", [])),
        "f64_unit_count": len(receipt.get("f64", {}).get("units", [])),
        "live_implementation_checked": check_live_files,
    }


def _mutation_tests(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[tuple[str, list[str]]] = []
    readout = copy.deepcopy(receipt["f61"])
    readout["units"][0]["readout"][0] += 1.0
    cases.append(("f61-readout", _verify_f61(readout)))

    lineage = copy.deepcopy(receipt["f61"])
    lineage["units"][0]["lineage"]["ops"].append("tampered")
    cases.append(("f61-lineage", _verify_f61(lineage)))

    budget = copy.deepcopy(receipt["f62"])
    original_budget = budget["units"][0]["control_arms"][0]["matched_budget"]
    budget["units"][0]["control_arms"][0]["matched_budget"] = {
        **original_budget,
        "params": original_budget["params"] + 1,
    }
    cases.append(("f62-budget", _verify_f62(budget)))

    f63 = copy.deepcopy(receipt["f63"])
    f63["units"][0]["metrics"]["adaptation_energy"] = 100.0
    cases.append(("f63-energy", _verify_f63(f63)))

    f64 = copy.deepcopy(receipt["f64"])
    f64["units"][0]["result"] = "favorable"
    cases.append(("f64-tie-promotion", _verify_f64(f64)))

    controls = copy.deepcopy(receipt["f64"])
    controls["units"][0]["policies"].pop()
    cases.append(("f64-control-drop", _verify_f64(controls)))
    return [
        {"id": name, "rejected": bool(errors), "observed_errors": errors}
        for name, errors in cases
    ]


def verify_receipt(
    receipt: dict[str, Any], *, check_live_files: bool = True, run_mutations: bool = True
) -> dict[str, Any]:
    errors, checks = _base_errors(receipt, check_live_files=check_live_files)
    fresh_rows, fresh_errors = _fresh_challenges()
    errors.extend(fresh_errors)
    mutations: list[dict[str, Any]] = []
    mutation_runner_error: str | None = None
    if run_mutations:
        try:
            mutations = _mutation_tests(receipt)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            mutation_runner_error = f"material mutation battery could not execute: {exc}"
            errors.append(mutation_runner_error)
    all_mutations_rejected = bool(mutations) and all(row["rejected"] for row in mutations)
    if run_mutations and not all_mutations_rejected:
        errors.append("material semantic mutation rejection incomplete")
    primary_seeds = set(receipt.get("preregistration", {}).get("primary_seeds", []))
    return {
        "schema": VERIFIER_SCHEMA,
        "implementation": "independent raw JSON replay plus fresh-seed public-API attacks",
        "claim_scope": CLAIM_SCOPE,
        "core_payload_sha256": receipt.get("core_payload_sha256"),
        "checks": checks,
        "fresh_seed_challenges": fresh_rows,
        "fresh_seed_count": len(fresh_rows),
        "fresh_seeds_disjoint_from_primary": not (set(FRESH_SEEDS) & primary_seeds),
        "mutation_tests": mutations,
        "all_mutations_rejected": all_mutations_rejected,
        "mutation_runner_error": mutation_runner_error,
        "errors": errors,
        "independent": True,
        "adversarial": True,
        "verified": not errors,
    }


def verify_payload_sha256(receipt: dict[str, Any]) -> bool:
    digest = receipt.get("payload_sha256")
    return isinstance(digest, str) and digest == _sha(
        {key: value for key, value in receipt.items() if key != "payload_sha256"}
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt",
        nargs="?",
        default=str(REPO_ROOT / "proof" / "F61_F64_MATERIAL_TWIN_RUN.json"),
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "proof" / "F61_F64_MATERIAL_TWIN_VERIFICATION.json")
    )
    parser.add_argument("--skip-live-files", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt_path = Path(args.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = verify_receipt(receipt, check_live_files=not args.skip_live_files, run_mutations=True)
    report["payload_sha256_verified"] = verify_payload_sha256(receipt)
    report["receipt_payload_sha256"] = receipt.get("payload_sha256")
    report["receipt_file_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if args.report:
        _atomic_write(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["verified"] and report["payload_sha256_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
