
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .. import config
from ..config import REPO_ROOT
from ..devel.registries import load_experiments
from ..experiments import REGISTRY
from ..harness.runner import run_experiment
from ..logging_utils import new_run_dir
from .experiment_contracts import compare_contract_sources

SCHEMA = "mop-form-independent-verifier/v1"
PROOF_DIR = Path("proof/FORM_SUBSTRATE/VERIFIERS")
RAW_RUN_ROOT = Path("runs/form_verifiers")

FRESH_SEEDS = (101, 211, 307, 401, 503)
CANDIDATE_POSITIVES = (
    "f1_form_alignment_gate",
    "f2_heldout_form_transfer",
    "f3_form_bottleneck_capacity",
    "f4_raw_payload_vs_form_tokens",
    "f5_cross_form_memory_binding",
    "f6_sensorimotor_form_closure",
    "f9_cross_form_compositional_binding",
    "f10_intrinsic_form_curriculum",
    "f12_private_form_language_stability",
    "f13_form_energy_budget",
    "f15_embodied_affordance_form",
    "f18_counterfactual_form_intervention",
)

_CAPABILITY_FIELDS = {
    "f1_form_alignment_gate": "aligned_transfer",
    "f2_heldout_form_transfer": "heldout_form_acc",
    "f3_form_bottleneck_capacity": "wide_form_acc",
    "f4_raw_payload_vs_form_tokens": "canonical_cross_form_acc",
    "f5_cross_form_memory_binding": "cross_form_recall_at_k",
    "f6_sensorimotor_form_closure": "rollout_r2",
    "f9_cross_form_compositional_binding": "heldout_combo_acc",
    "f10_intrinsic_form_curriculum": "learning_progress_transfer",
    "f12_private_form_language_stability": "cross_seed_code_transfer",
    "f13_form_energy_budget": "form_best_grid_accuracy",
    "f15_embodied_affordance_form": "affordance_decode_acc",
    "f18_counterfactual_form_intervention": "counterfactual_match_acc",
}

_SOURCE_REQUIREMENTS = {
    "f1_form_alignment_gate": (
        "_three_way_split",
        "fit_affine_alignment",
        "moment_target",
        "shuffled_src",
    ),
    "f2_heldout_form_transfer": (
        "_fit_matched",
        "x_epoch",
        "shuffled_map",
        "matched_rows_updates_head",
    ),
    "f3_form_bottleneck_capacity": (
        "q[:, :wide_dim]",
        "q[:, :small_dim]",
        "_pad",
        "shared_seed",
    ),
    "f4_raw_payload_vs_form_tokens": (
        "_F4TokenProbe",
        "requires [N,T,D]",
        "_f4_align_tokens",
        "torch.roll",
    ),
    "f5_cross_form_memory_binding": (
        "ReplayBuffer",
        "memory.add",
        "mem.retrieve",
        "shuf_store",
    ),
    "f6_sensorimotor_form_closure": (
        "_rollout_r2",
        "_goal_success",
        '"blind"',
        '"shuffled"',
    ),
    "f9_cross_form_compositional_binding": (
        "head_a",
        "head_b",
        "shuffled_idx",
        "composite_held",
    ),
    "f10_intrinsic_form_curriculum": (
        "_run_form_scheduler",
        "val_acc",
        "test_acc",
        '"novelty"',
        "heldout",
    ),
    "f12_private_form_language_stability": (
        "_hungarian",
        "fit_idx=tr",
        "shuffled_j",
        "remap",
    ),
    "f13_form_energy_budget": (
        "base_form_flops",
        "alignment_fit_flops",
        "alignment_apply_flops",
        "fusion_flops",
        "token_projection_flops",
        "train_shell_flops",
        "eval_shell_flops",
        "bytes_moved",
        "structural_bytes",
        "produced_bytes",
    ),
    "f15_embodied_affordance_form": (
        "_affordance_dataset",
        "outcomes_tr",
        '"action_shuffle"',
        "appearance_confound",
    ),
    "f18_counterfactual_form_intervention": (
        "_f18_fit_map",
        "b_interventional",
        "b_observational",
        "b_test",
        '"form_b_state"',
        "matched_within",
    ),
}


@dataclass
class ExecutionTrace:

    ci_inputs: list[list[float]] = field(default_factory=list)
    sign_inputs: list[list[float]] = field(default_factory=list)
    hungarian_calls: list[dict[str, Any]] = field(default_factory=list)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _plain(value: Any) -> dict[str, Any]:
    converted = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    return {str(key): child for key, child in converted.items()} if isinstance(converted, dict) else {}


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any, default: float = math.nan) -> float:
    return float(value) if _finite(value) else default


def _close(left: Any, right: Any, *, atol: float = 0.0015, rtol: float = 1.0e-6) -> bool:
    if not (_finite(left) and _finite(right)):
        return False
    return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any = None,
    expected: Any = None,
    detail: str = "",
    category: str = "semantic",
) -> None:
    record: dict[str, Any] = {
        "name": name,
        "category": category,
        "passed": bool(passed),
    }
    if observed is not None:
        record["observed"] = observed
    if expected is not None:
        record["expected"] = expected
    if detail:
        record["detail"] = detail
    checks.append(record)


def _independent_ci(values: Sequence[float], z: float = 1.96) -> dict[str, Any]:
    vals = [float(value) for value in values]
    n = len(vals)
    mean = sum(vals) / max(n, 1)
    variance = sum((value - mean) ** 2 for value in vals) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(variance)
    half = z * sd / math.sqrt(n) if n else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci_half": round(half, 4),
        "lo": round(mean - half, 4),
        "hi": round(mean + half, 4),
        "unstable": n < 3,
    }


def _independent_signs(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(value) for value in values]
    n_pos = sum(value > 0 for value in vals)
    n_neg = sum(value < 0 for value in vals)
    n_zero = len(vals) - n_pos - n_neg
    any_flip = bool(n_pos and n_neg)
    consistent = 0 if any_flip or not (n_pos or n_neg) else (1 if n_pos else -1)
    return {
        "n": len(vals),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "any_flip": any_flip,
        "consistent_sign": consistent,
    }


def _assignment_minimum_dp(cost: np.ndarray) -> float:
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"independent assignment requires square cost, got {matrix.shape}")
    n = int(matrix.shape[0])
    if n > 18:
        raise ValueError(f"independent subset solver is capped at n=18, got {n}")
    size = 1 << n
    dp = [math.inf] * size
    dp[0] = 0.0
    for mask in range(size):
        row = mask.bit_count()
        if row >= n or not math.isfinite(dp[mask]):
            continue
        remaining = ((1 << n) - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            column = bit.bit_length() - 1
            next_mask = mask | bit
            candidate = dp[mask] + float(matrix[row, column])
            if candidate < dp[next_mask]:
                dp[next_mask] = candidate
            remaining ^= bit
    return float(dp[-1])


@contextmanager
def _capture_execution(experiment_id: str) -> Iterator[ExecutionTrace]:
    from ..diagnostics import riskcov, seed_consistency
    from ..experiments import f_form_substrate_missing

    trace = ExecutionTrace()
    original_ci = riskcov.seed_ci
    original_sign = riskcov.sign_flip_report
    original_missing_ci = f_form_substrate_missing.seed_ci
    original_missing_sign = f_form_substrate_missing.sign_flip_report
    original_hungarian = seed_consistency._hungarian

    def traced_ci(values: Sequence[float], z: float = 1.96) -> dict[str, Any]:
        captured = [float(value) for value in values]
        trace.ci_inputs.append(captured)
        return original_ci(captured, z=z)

    def traced_sign(values: Sequence[float]) -> dict[str, Any]:
        captured = [float(value) for value in values]
        trace.sign_inputs.append(captured)
        return original_sign(captured)

    def traced_hungarian(cost: np.ndarray) -> list[int]:
        assignment = original_hungarian(cost)
        matrix = np.asarray(cost, dtype=float)
        actual = float(sum(matrix[row, column] for row, column in enumerate(assignment)))
        optimum = _assignment_minimum_dp(matrix)
        trace.hungarian_calls.append(
            {
                "n": int(matrix.shape[0]),
                "actual_cost": actual,
                "independent_optimum": optimum,
                "exact": bool(math.isclose(actual, optimum, abs_tol=1.0e-9, rel_tol=1.0e-9)),
            }
        )
        return assignment

    riskcov.seed_ci = traced_ci
    riskcov.sign_flip_report = traced_sign  # type: ignore[assignment]
    f_form_substrate_missing.seed_ci = traced_ci
    f_form_substrate_missing.sign_flip_report = traced_sign  # type: ignore[assignment]
    if experiment_id == "f12_private_form_language_stability":
        seed_consistency._hungarian = traced_hungarian
    try:
        yield trace
    finally:
        riskcov.seed_ci = original_ci
        riskcov.sign_flip_report = original_sign
        f_form_substrate_missing.seed_ci = original_missing_ci
        f_form_substrate_missing.sign_flip_report = original_missing_sign
        seed_consistency._hungarian = original_hungarian


def _freeze_live_contract(experiment_id: str, root: Path) -> dict[str, Any]:
    rows = {str(row["id"]): row for row in load_experiments(root / "registry/experiments.yaml")}
    row = rows.get(experiment_id, {})
    cls = REGISTRY[experiment_id]
    config_path = root / "configs/experiment" / f"{experiment_id}.yaml"
    experiment_config = _plain(OmegaConf.load(config_path))
    audit = compare_contract_sources(row, cls, experiment_config)
    module_path = Path(inspect.getfile(cls)).resolve()
    class_source = inspect.getsource(cls)
    module_source = module_path.read_text()
    canonical = dict(audit.get("canonical") or {})
    frozen = {
        "experiment_id": experiment_id,
        "canonical": canonical,
        "registry_row_fingerprint": _object_sha256(row),
        "config_path": _display(root, config_path),
        "config_sha256": _sha256(config_path),
        "class_source_sha256": hashlib.sha256(class_source.encode()).hexdigest(),
        "module_path": _display(root, module_path),
        "module_sha256": _sha256(module_path),
        "verifier_path": _display(root, Path(__file__).resolve()),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "source_requirements": list(_SOURCE_REQUIREMENTS[experiment_id]),
        "source_requirements_present": {
            token: token in module_source for token in _SOURCE_REQUIREMENTS[experiment_id]
        },
        "contract_problems": list(audit.get("problems") or []),
    }
    frozen["contract_fingerprint"] = _object_sha256(canonical)
    return frozen


def _add_common_checks(
    experiment_id: str,
    metrics: dict[str, Any],
    live_config: dict[str, Any],
    canonical_receipt: dict[str, Any],
    fresh_seeds: list[int],
    trace: ExecutionTrace,
    frozen_before: dict[str, Any],
    frozen_after: dict[str, Any],
    verifier_run_dir: Path,
    root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    declared = list(frozen_before.get("canonical", {}).get("metric") or [])
    canonical_seeds = list((canonical_receipt.get("execution") or {}).get("observed_seeds") or [])
    observed_seeds = [int(value) for value in metrics.get("seeds") or []]
    _check(
        checks,
        "canonical_candidate_receipt_is_positive_and_canonical",
        canonical_receipt.get("all_ok") is True
        and (canonical_receipt.get("null_decision") or {}).get("null_supported") is False
        and (canonical_receipt.get("null_decision") or {}).get("declared_verdict") == "PUBLISH-POSITIVE",
        observed={
            "all_ok": canonical_receipt.get("all_ok"),
            "null_decision": canonical_receipt.get("null_decision"),
        },
        expected="canonical PUBLISH-POSITIVE candidate",
        category="independence",
    )
    _check(
        checks,
        "fresh_effective_seeds_are_disjoint",
        len(fresh_seeds) >= 5
        and len(set(fresh_seeds)) == len(fresh_seeds)
        and not (set(fresh_seeds) & set(canonical_seeds)),
        observed={"canonical": canonical_seeds, "verifier": fresh_seeds},
        expected="at least five unique disjoint seeds",
        category="independence",
    )
    _check(
        checks,
        "fresh_seeds_reached_experiment",
        observed_seeds == fresh_seeds,
        observed=observed_seeds,
        expected=fresh_seeds,
        category="independence",
    )
    canonical_source_dir = str((canonical_receipt.get("source") or {}).get("run_dir") or "")
    _check(
        checks,
        "verifier_execution_path_is_separate",
        RAW_RUN_ROOT.as_posix() in _display(root, verifier_run_dir)
        and _display(root, verifier_run_dir) != canonical_source_dir,
        observed={"verifier": _display(root, verifier_run_dir), "canonical": canonical_source_dir},
        expected="distinct runs/form_verifiers path",
        category="independence",
    )
    _check(
        checks,
        "class_config_registry_contract_was_clean_and_frozen",
        not frozen_before.get("contract_problems")
        and frozen_before.get("contract_fingerprint") == frozen_after.get("contract_fingerprint")
        and frozen_before.get("registry_row_fingerprint") == frozen_after.get("registry_row_fingerprint")
        and frozen_before.get("config_sha256") == frozen_after.get("config_sha256")
        and frozen_before.get("class_source_sha256") == frozen_after.get("class_source_sha256")
        and frozen_before.get("module_sha256") == frozen_after.get("module_sha256")
        and frozen_before.get("verifier_sha256") == frozen_after.get("verifier_sha256"),
        observed={
            "before_contract": frozen_before.get("contract_fingerprint"),
            "after_contract": frozen_after.get("contract_fingerprint"),
            "problems": frozen_before.get("contract_problems"),
        },
        expected="exact stable live contract",
        category="independence",
    )
    source_presence = dict(frozen_after.get("source_requirements_present") or {})
    _check(
        checks,
        "claim_semantics_are_load_bearing_in_fingerprinted_source",
        bool(source_presence) and all(source_presence.values()),
        observed=source_presence,
        expected="all experiment-specific source requirements present",
        category="semantic",
    )
    _check(
        checks,
        "verifier_implementation_is_a_distinct_source_file",
        frozen_after.get("verifier_path") != frozen_after.get("module_path")
        and bool(frozen_after.get("verifier_sha256")),
        observed={
            "verifier": frozen_after.get("verifier_path"),
            "treatment": frozen_after.get("module_path"),
        },
        category="independence",
    )
    missing_or_nonfinite = [name for name in declared if not _finite(metrics.get(name))]
    recursive_nonfinite = _nonfinite_numeric_paths(metrics)
    _check(
        checks,
        "declared_and_emitted_numeric_metrics_are_finite",
        not missing_or_nonfinite and not recursive_nonfinite,
        observed={"declared_bad": missing_or_nonfinite, "emitted_bad": recursive_nonfinite[:20]},
        expected="all declared metrics and emitted numeric fields finite",
    )

    capability_field = _CAPABILITY_FIELDS[experiment_id]
    if experiment_id == "f13_form_energy_budget":
        points = list(metrics.get("frontier_points") or [])
        capability = max(
            (_number(point.get("accuracy"), -math.inf) for point in points if point.get("arm") == "form"),
            default=math.nan,
        )
    else:
        capability = _number(metrics.get(capability_field))
    _check(
        checks,
        "capability_is_non_ceiling",
        math.isfinite(capability) and 0.0 < capability < 0.995,
        observed={"field": capability_field, "value": capability},
        expected="0 < capability < 0.995",
    )

    effect_candidates = [values for values in trace.ci_inputs if len(values) >= 3]
    sign_candidates = [values for values in trace.sign_inputs if len(values) >= 3]
    effects = effect_candidates[-1] if effect_candidates else []
    sign_effects = sign_candidates[-1] if sign_candidates else []
    independent_ci = _independent_ci(effects)
    independent_signs = _independent_signs(sign_effects)
    reported_ci = dict(metrics.get("seed_ci") or {})
    reported_signs = dict(metrics.get("sign_flip_report") or metrics.get("sign_flip") or {})
    _check(
        checks,
        "live_per_seed_effects_were_captured",
        len(effects) >= 3 and effects == sign_effects,
        observed={"ci_effects": effects, "sign_effects": sign_effects},
        expected="same live effect vector at both diagnostic boundaries",
        category="independence",
    )
    _check(
        checks,
        "reported_ci_matches_independent_recomputation",
        _dict_numeric_close(reported_ci, independent_ci),
        observed={"reported": reported_ci, "independent": independent_ci},
        expected="exact rounded normal CI from captured effects",
        category="independence",
    )
    _check(
        checks,
        "reported_signs_match_independent_recomputation",
        all(reported_signs.get(key) == value for key, value in independent_signs.items()),
        observed={"reported": reported_signs, "independent": independent_signs},
        expected="exact sign counts from captured effects",
        category="independence",
    )
    _check(
        checks,
        "fresh_seed_ci_lower_bound_is_positive",
        independent_ci["n"] >= 3
        and independent_ci["unstable"] is False
        and _number(independent_ci["lo"]) > 0.0,
        observed=independent_ci,
        expected="stable CI with lo > 0",
    )
    _check(
        checks,
        "fresh_seed_effect_sign_is_stable",
        independent_signs["n"] >= 3
        and independent_signs["n_neg"] == 0
        and independent_signs["any_flip"] is False
        and independent_signs["consistent_sign"] == 1
        and independent_signs["n_pos"] > 0,
        observed=independent_signs,
        expected="no negative effects, no sign flip, and at least one positive effect",
    )
    emitted_effects = _emitted_effect_vector(metrics)
    if emitted_effects is not None:
        _check(
            checks,
            "emitted_per_seed_effects_match_live_capture",
            len(emitted_effects) == len(effects)
            and all(_close(left, right) for left, right in zip(emitted_effects, effects, strict=True)),
            observed={"emitted": emitted_effects, "captured": effects},
            expected="same per-seed effects",
            category="independence",
        )
    return {
        "canonical_seeds": canonical_seeds,
        "observed_seeds": observed_seeds,
        "captured_effects": effects,
        "independent_ci": independent_ci,
        "independent_signs": independent_signs,
        "capability_field": capability_field,
        "capability": capability,
        "live_config": live_config,
    }


def _nonfinite_numeric_paths(value: Any, prefix: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            problems.extend(_nonfinite_numeric_paths(child, path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            problems.extend(_nonfinite_numeric_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, int | float) and not isinstance(value, bool) and not math.isfinite(float(value)):
        problems.append(prefix)
    return problems


def _dict_numeric_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key, expected in right.items():
        actual = left.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif isinstance(expected, int) and key == "n":
            if actual != expected:
                return False
        elif not _close(actual, expected, atol=0.0001, rtol=0.0):
            return False
    return True


def _emitted_effect_vector(metrics: Mapping[str, Any]) -> list[float] | None:
    for key in (
        "per_seed_deltas",
        "per_seed_transfer_deltas",
        "per_seed_frontier_deltas",
    ):
        value = metrics.get(key)
        if isinstance(value, list) and all(_finite(item) for item in value):
            return [float(item) for item in value]
    return None


def _accounting_equal(accounting: Any, required_arms: Sequence[str]) -> bool:
    if not isinstance(accounting, Mapping) or not all(arm in accounting for arm in required_arms):
        return False
    values = [accounting[arm] for arm in required_arms]
    return all(isinstance(value, Mapping) for value in values) and all(
        value == values[0] for value in values[1:]
    )


def _semantic_checks(
    experiment_id: str,
    metrics: dict[str, Any],
    e: dict[str, Any],
    trace: ExecutionTrace,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    dispatch = {
        "f1_form_alignment_gate": _verify_f1,
        "f2_heldout_form_transfer": _verify_f2,
        "f3_form_bottleneck_capacity": _verify_f3,
        "f4_raw_payload_vs_form_tokens": _verify_f4,
        "f5_cross_form_memory_binding": _verify_f5,
        "f6_sensorimotor_form_closure": _verify_f6,
        "f9_cross_form_compositional_binding": _verify_f9,
        "f10_intrinsic_form_curriculum": _verify_f10,
        "f12_private_form_language_stability": _verify_f12,
        "f13_form_energy_budget": _verify_f13,
        "f15_embodied_affordance_form": _verify_f15,
        "f18_counterfactual_form_intervention": _verify_f18,
    }
    return dispatch[experiment_id](metrics, e, trace, checks)


def _strongest_delta(
    checks: list[dict[str, Any]],
    *,
    name: str,
    treatment: Any,
    controls: Mapping[str, Any],
    margin: float,
) -> tuple[float, float]:
    control_values = {name: _number(value) for name, value in controls.items()}
    strongest = max(control_values.values(), default=math.nan)
    delta = _number(treatment) - strongest
    _check(
        checks,
        name,
        math.isfinite(delta) and delta > margin,
        observed={
            "treatment": treatment,
            "controls": control_values,
            "strongest": strongest,
            "recomputed_delta": delta,
        },
        expected=f"delta > preregistered margin {margin}",
    )
    return delta, strongest


def _verify_f1(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f1_paired_alignment_beats_every_unpaired_control",
        treatment=m.get("aligned_transfer"),
        controls={
            "raw": m.get("raw_transfer"),
            "moment_matched": m.get("moment_matched_transfer"),
            "shuffled_anchor": m.get("shuffled_anchor_transfer"),
        },
        margin=float(e["margin"]),
    )
    _check(c, "f1_alignment_and_label_splits_are_disjoint", m.get("disjoint_splits") is True)
    _check(
        c,
        "f1_alignment_is_above_chance_band",
        _number(m.get("aligned_transfer")) > _number(m.get("chance")) + float(e["chance_band"]),
        observed={"aligned": m.get("aligned_transfer"), "chance": m.get("chance")},
    )
    _check(
        c, "f1_reported_strongest_control_is_recomputable", _close(m.get("best_unpaired_control"), strongest)
    )
    return {"strongest_control": strongest, "aggregate_delta": delta}


def _verify_f2(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f2_matched_exposure_transfer_beats_every_control",
        treatment=m.get("heldout_form_acc"),
        controls={
            "single_form": m.get("single_form_baseline"),
            "matched_noise": m.get("matched_noise_baseline"),
            "shuffled_alignment": m.get("shuffled_heldout_alignment_acc"),
        },
        margin=float(e["margin"]),
    )
    _check(
        c,
        "f2_rows_updates_head_and_input_are_matched",
        m.get("matched_rows_updates_head") is True
        and _accounting_equal(m.get("matched_accounting"), ("multi", "single", "matched_noise")),
        observed=m.get("matched_accounting"),
    )
    _check(c, "f2_anchor_label_test_splits_are_disjoint", m.get("disjoint_splits") is True)
    _check(
        c,
        "f2_heldout_transfer_is_above_chance_band",
        _number(m.get("heldout_form_acc")) > _number(m.get("chance")) + float(e["chance_band"]),
    )
    return {"strongest_control": strongest, "aggregate_delta": delta}


def _verify_f3(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f3_nested_wide_bottleneck_beats_small",
        treatment=m.get("wide_form_acc"),
        controls={"nested_small": m.get("small_form_acc")},
        margin=float(e["margin"]),
    )
    _check(
        c,
        "f3_projection_is_strictly_nested_and_small_is_zero_padded",
        m.get("nested_projection") is True
        and m.get("small_zero_padded") is True
        and 0 < int(m.get("small_dim", 0)) < int(m.get("wide_dim", 0)),
        observed={"small": m.get("small_dim"), "wide": m.get("wide_dim")},
    )
    _check(
        c,
        "f3_all_heads_data_and_updates_are_identical",
        m.get("identical_head_topology") is True
        and _accounting_equal(
            m.get("matched_accounting"),
            ("wide", "small", "no_bottleneck", "concat", "shuffle"),
        ),
        observed=m.get("matched_accounting"),
    )
    floor = max(_number(m.get("chance")), _number(m.get("shuffle_floor_acc")))
    _check(
        c,
        "f3_wide_bottleneck_clears_random_floor_band",
        _number(m.get("wide_form_acc")) > floor + float(e["floor_band"]),
        observed={"wide": m.get("wide_form_acc"), "floor": floor},
    )
    return {"strongest_control": strongest, "aggregate_delta": delta, "floor": floor}


def _verify_f4(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f4_ordered_tokens_beat_every_payload_control",
        treatment=m.get("canonical_cross_form_acc"),
        controls={
            "raw": m.get("raw_cross_form_acc"),
            "handcrafted": m.get("handcrafted_cross_form_acc"),
            "shuffled_referent": m.get("shuffled_referent_token_acc"),
            "permuted_token_order": m.get("token_order_permutation_acc"),
        },
        margin=float(e["margin"]),
    )
    expected_shape = [int(e["token_count"]), int(e["token_dim"])]
    _check(
        c,
        "f4_native_token_geometry_is_preserved_and_enforced",
        m.get("token_shape") == expected_shape
        and int(e["token_count"]) > 1
        and m.get("token_axis_preserved") is True
        and m.get("token_probe_rejects_flattened") is True,
        observed=m.get("token_shape"),
        expected=expected_shape,
    )
    _check(
        c,
        "f4_token_heads_rows_and_updates_are_matched",
        _accounting_equal(m.get("matched_accounting"), ("canonical", "raw", "handcrafted")),
        observed=m.get("matched_accounting"),
    )
    _check(c, "f4_anchor_label_test_splits_are_disjoint", m.get("disjoint_splits") is True)
    return {"strongest_control": strongest, "aggregate_delta": delta}


def _verify_f5(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f5_cross_form_memory_beats_raw_and_shuffled_retrieval",
        treatment=m.get("cross_form_recall_at_k"),
        controls={
            "raw_cross_form": m.get("raw_cross_form_recall_at_k"),
            "shuffled_referent": m.get("shuffled_referent_floor"),
        },
        margin=float(e["margin"]),
    )
    slots = int(m.get("store_slots", 0))
    _check(
        c,
        "f5_actual_memory_was_allocated_and_queried",
        slots > 0 and int(m.get("store_bytes", 0)) > 0 and str(m.get("index_backend") or "") != "",
        observed={"slots": slots, "bytes": m.get("store_bytes"), "backend": m.get("index_backend")},
    )
    expected_density = _number(m.get("cross_form_recall_at_k")) / max(slots, 1)
    _check(
        c, "f5_recall_per_slot_is_recomputed", _close(m.get("recall_per_slot"), expected_density, atol=1.0e-6)
    )
    oracle_gap = _number(m.get("same_form_recall_at_k")) - _number(m.get("cross_form_recall_at_k"))
    _check(
        c,
        "f5_cross_form_recall_is_within_same_form_oracle_band",
        oracle_gap <= float(e["oracle_band"]),
        observed=oracle_gap,
        expected=f"<= {e['oracle_band']}",
    )
    _check(c, "f5_store_and_query_forms_are_distinct", m.get("store_form") != m.get("query_form"))
    return {"strongest_control": strongest, "aggregate_delta": delta, "oracle_gap": oracle_gap}


def _verify_f6(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    roll = dict(m.get("rollout_r2_by_arm") or {})
    goals = dict(m.get("goal_success_by_arm") or {})
    roll_delta, roll_control = _strongest_delta(
        c,
        name="f6_action_conditioning_improves_autoregressive_rollout",
        treatment=roll.get("true"),
        controls={"blind": roll.get("blind"), "shuffled": roll.get("shuffled")},
        margin=float(e["margin"]),
    )
    goal_delta, goal_control = _strongest_delta(
        c,
        name="f6_action_conditioning_improves_goal_reachability",
        treatment=goals.get("true"),
        controls={"blind": goals.get("blind"), "shuffled": goals.get("shuffled")},
        margin=float(e["margin"]),
    )
    _check(
        c,
        "f6_controls_have_identical_architecture_and_updates",
        m.get("matched_architecture") is True and m.get("matched_updates") is True,
    )
    _check(
        c,
        "f6_action_shuffle_delta_is_recomputed",
        _close(m.get("action_shuffle_delta"), _number(roll.get("true")) - _number(roll.get("shuffled"))),
    )
    return {
        "strongest_rollout_control": roll_control,
        "strongest_goal_control": goal_control,
        "rollout_delta": roll_delta,
        "goal_delta": goal_delta,
    }


def _verify_f9(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f9_heldout_composition_beats_every_noncompositional_control",
        treatment=m.get("heldout_combo_acc"),
        controls={
            "single_form": m.get("best_single_form_acc"),
            "shuffled_label": m.get("shuffle_floor_acc"),
            "shuffled_referent": m.get("shuffled_referent_acc"),
            "conjunction_head": m.get("composite_head_heldout_acc"),
            "chance": m.get("chance"),
        },
        margin=float(e["margin"]),
    )
    _check(
        c,
        "f9_heldout_seen_generalization_gap_is_bounded",
        _number(m.get("heldout_seen_gap")) <= float(e["gap_margin"]),
        observed=m.get("heldout_seen_gap"),
        expected=f"<= {e['gap_margin']}",
    )
    _check(
        c,
        "f9_factor_cardinalities_match_contract",
        int(m.get("n_a", 0)) == int(e["n_a"]) and int(m.get("n_b", 0)) == int(e["n_b"]),
    )
    return {"strongest_control": strongest, "aggregate_delta": delta}


def _verify_f10(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f10_learning_progress_improves_untouched_form_transfer",
        treatment=m.get("learning_progress_transfer"),
        controls={"strongest_control": m.get("strongest_control_transfer")},
        margin=float(e["margin"]),
    )
    strongest_coverage = max(
        _number(m.get("uniform_coverage")),
        _number(m.get("error_coverage")),
        _number(m.get("novelty_coverage")),
    )
    coverage_delta = _number(m.get("lp_coverage")) - strongest_coverage
    _check(
        c,
        "f10_learning_progress_improves_real_form_coverage",
        coverage_delta > 0,
        observed={"lp": m.get("lp_coverage"), "strongest": strongest_coverage, "delta": coverage_delta},
    )
    noisy_delta = _number(m.get("uniform_noisy_timeshare")) - _number(m.get("noisy_form_timeshare"))
    _check(
        c,
        "f10_learning_progress_avoids_unlearnable_noisy_forms",
        noisy_delta >= float(e["noisy_margin"]),
        observed=noisy_delta,
        expected=f">= {e['noisy_margin']}",
    )
    expected_per_update = _number(m.get("lp_coverage")) / int(e["rounds"])
    _check(
        c,
        "f10_coverage_per_update_is_recomputed",
        _close(m.get("coverage_per_update"), expected_per_update, atol=5.0e-6),
    )
    return {
        "strongest_transfer_control": strongest,
        "aggregate_transfer_delta": delta,
        "coverage_delta": coverage_delta,
        "noisy_timeshare_reduction": noisy_delta,
    }


def _verify_f12(m: dict[str, Any], e: dict[str, Any], t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    agreement_delta = _number(m.get("code_agreement")) - _number(m.get("random_code_floor"))
    transfer_floor = max(_number(m.get("chance")), _number(m.get("shuffled_referent_transfer")))
    transfer_delta = _number(m.get("cross_seed_code_transfer")) - transfer_floor
    _check(
        c,
        "f12_code_agreement_beats_random_codebook_floor",
        agreement_delta > float(e["margin"]),
        observed=agreement_delta,
        expected=f"> {e['margin']}",
    )
    _check(
        c,
        "f12_heldout_code_transfer_beats_shuffled_referents_and_chance",
        transfer_delta > float(e["transfer_margin"]),
        observed={
            "transfer": m.get("cross_seed_code_transfer"),
            "floor": transfer_floor,
            "delta": transfer_delta,
        },
        expected=f"delta > {e['transfer_margin']}",
    )
    expected_calls = len(list(m.get("seeds") or [])) * (len(list(m.get("seeds") or [])) - 1) + 2 * (
        len(list(m.get("seeds") or [])) - 1
    )
    _check(
        c,
        "f12_every_live_hungarian_call_is_exact_under_independent_solver",
        len(t.hungarian_calls) >= expected_calls
        and all(call.get("exact") is True for call in t.hungarian_calls),
        observed={
            "calls": len(t.hungarian_calls),
            "expected_min": expected_calls,
            "all_exact": all(call.get("exact") is True for call in t.hungarian_calls),
        },
        expected="all live train-only and shuffled-control assignments globally optimal",
        category="independence",
    )
    _check(c, "f12_language_flag_matches_recomputed_claim", m.get("codes_recur_across_seeds") is True)
    return {
        "agreement_delta": agreement_delta,
        "transfer_floor": transfer_floor,
        "transfer_delta": transfer_delta,
        "hungarian_calls": t.hungarian_calls,
    }


def _verify_f13(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    points = [dict(point) for point in m.get("frontier_points") or [] if isinstance(point, Mapping)]
    arms = ("form", "raw", "random")
    expected_axes = set(
        itertools.product(
            arms,
            [int(value) for value in e["widths"]],
            [int(value) for value in e["token_counts"]],
            [int(value) for value in e["shell_sizes"]],
            [int(value) for value in e["replay_bytes"]],
        )
    )
    observed_axes = {
        (
            str(point.get("arm")),
            int(point.get("width", -1)),
            int(point.get("token_count", -1)),
            int(point.get("shell_size", -1)),
            int(point.get("replay_budget_bytes", -1)),
        )
        for point in points
    }
    _check(
        c,
        "f13_preregistered_full_grid_is_complete_without_duplicates",
        observed_axes == expected_axes and len(points) == len(expected_axes),
        observed={"points": len(points), "unique_axes": len(observed_axes)},
        expected=len(expected_axes),
    )
    cost_fields = ("params", "retained_bytes", "estimated_flops", "estimated_energy_joules")
    _check(
        c,
        "f13_every_grid_point_has_finite_positive_full_system_costs",
        bool(points)
        and all(
            _finite(point.get(field)) and _number(point.get(field)) > 0
            for point in points
            for field in cost_fields
        ),
        expected=list(cost_fields),
    )
    pj_per_flop = float(e["energy_pj_per_flop"])
    energy_lower_bound_ok = all(
        _number(point.get("estimated_energy_joules"))
        > _number(point.get("estimated_flops")) * pj_per_flop * 1.0e-12
        for point in points
    )
    _check(
        c,
        "f13_energy_estimate_prices_flops_plus_nonzero_byte_movement",
        energy_lower_bound_ok and m.get("energy_measured") is False,
        observed={"energy_measured": m.get("energy_measured"), "pj_per_flop": pj_per_flop},
        expected="analytical estimate strictly above FLOP-only lower bound",
    )
    by_axis = {
        (
            str(point["arm"]),
            int(point["width"]),
            int(point["token_count"]),
            int(point["shell_size"]),
            int(point["replay_budget_bytes"]),
        ): point
        for point in points
    }
    structural_cost_ok = True
    for width, tokens, shell, replay in itertools.product(
        e["widths"], e["token_counts"], e["shell_sizes"], e["replay_bytes"]
    ):
        form = by_axis.get(("form", int(width), int(tokens), int(shell), int(replay)), {})
        raw = by_axis.get(("raw", int(width), int(tokens), int(shell), int(replay)), {})
        structural_cost_ok = structural_cost_ok and (
            _number(form.get("estimated_flops")) > _number(raw.get("estimated_flops"))
            and _number(form.get("retained_bytes")) > _number(raw.get("retained_bytes"))
        )
    _check(
        c,
        "f13_alignment_fusion_and_multi_form_production_are_not_free",
        structural_cost_ok,
        expected="form cost exceeds matched raw cost at every identical grid coordinate",
    )
    test_rows = int(e["samples"]) - int(int(e["samples"]) * float(e["train_frac"]))
    per_correct_ok = all(
        0.0
        < _number(point.get("estimated_energy_joules")) / _number(point.get("estimated_energy_per_correct"))
        <= test_rows
        and math.isclose(
            _number(point.get("estimated_energy_joules"))
            / _number(point.get("estimated_energy_per_correct")),
            _number(point.get("accuracy")) * test_rows,
            rel_tol=0.08,
            abs_tol=1.0,
        )
        for point in points
    )
    _check(c, "f13_energy_per_correct_is_independently_recomputed", per_correct_ok)
    best_byte = dict(m.get("best_byte_point") or {})
    best_param = dict(m.get("best_param_point") or {})
    best_energy = dict(m.get("best_energy_point") or {})
    _check(
        c,
        "f13_declared_density_metrics_recompute_from_frontier_points",
        _close(
            m.get("accuracy_per_byte"),
            _number(best_byte.get("accuracy")) / _number(best_byte.get("retained_bytes")),
            atol=5.0e-7,
        )
        and _close(
            m.get("accuracy_per_param"),
            _number(best_param.get("accuracy")) / _number(best_param.get("params")),
            atol=1.0e-6,
        )
        and _close(
            m.get("estimated_energy_per_correct"),
            best_energy.get("estimated_energy_per_correct"),
            atol=1.0e-12,
        ),
    )
    delta = _number(m.get("form_pareto_area")) - max(
        _number(m.get("raw_pareto_area")), _number(m.get("random_pareto_area"))
    )
    _check(
        c,
        "f13_form_frontier_beats_strongest_cost_matched_arm",
        delta > float(e["margin"]) and int(m.get("dominant_form_point_count", 0)) > 0,
        observed={"recomputed_delta": delta, "dominant_points": m.get("dominant_form_point_count")},
        expected=f"area delta > {e['margin']} and at least one dominant point",
    )
    return {
        "expected_grid_points": len(expected_axes),
        "observed_grid_points": len(points),
        "aggregate_area_delta": delta,
    }


def _verify_f15(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    arms = dict(m.get("accuracy_by_arm") or {})
    delta, strongest = _strongest_delta(
        c,
        name="f15_consequence_form_beats_passive_and_action_shuffle",
        treatment=arms.get("intervention"),
        controls={"passive": arms.get("passive"), "action_shuffle": arms.get("action_shuffle")},
        margin=float(e["margin"]),
    )
    _check(
        c,
        "f15_controls_have_identical_architecture_and_updates",
        m.get("matched_architecture") is True and m.get("matched_updates") is True,
    )
    audit = dict(m.get("form_audit") or {})
    tags = set(audit.get("tags") or [])
    _check(
        c,
        "f15_observation_and_action_consequence_are_distinct_referent_bound_forms",
        audit.get("all_ok") is True and {"observation", "action_consequence"} <= tags,
        observed={"tags": sorted(tags), "all_ok": audit.get("all_ok")},
    )
    _check(
        c,
        "f15_reported_intervention_deltas_recompute",
        _close(m.get("intervention_gain"), _number(arms.get("intervention")) - _number(arms.get("passive")))
        and _close(
            m.get("action_shuffle_delta"),
            _number(arms.get("intervention")) - _number(arms.get("action_shuffle")),
        ),
    )
    return {"strongest_control": strongest, "aggregate_delta": delta}


def _verify_f18(m: dict[str, Any], e: dict[str, Any], _t: ExecutionTrace, c: list[dict[str, Any]]) -> dict:
    delta, strongest = _strongest_delta(
        c,
        name="f18_unseen_intervention_beats_every_transport_control",
        treatment=m.get("counterfactual_match_acc"),
        controls={
            "correlational": m.get("correlational_baseline_acc"),
            "random_direction": m.get("random_direction_acc"),
            "shuffled_pair": m.get("shuffled_pair_acc"),
        },
        margin=float(e["margin"]),
    )
    matched = dict(m.get("matched_compute") or {})
    flops = list(matched.get("flops") or [])
    _check(
        c,
        "f18_correlational_and_interventional_arms_match_rows_updates_params_and_flops",
        matched.get("matched") is True
        and m.get("matched_train_rows") is True
        and len(flops) == 2
        and _close(flops[0], flops[1], atol=0.0)
        and int(m.get("train_rows_per_arm", 0)) > 0
        and int(m.get("updates_per_arm", 0)) == int(e["epochs"])
        and int(m.get("transport_params", 0)) > 0,
        observed={
            "matched_compute": matched,
            "rows": m.get("train_rows_per_arm"),
            "updates": m.get("updates_per_arm"),
        },
    )
    _check(
        c,
        "f18_test_intervention_value_is_genuinely_held_out",
        int(m.get("test_delta", -1)) not in [int(value) for value in m.get("train_deltas") or []]
        and int(m.get("test_delta", -1)) == int(e["test_delta"]),
        observed={"train": m.get("train_deltas"), "test": m.get("test_delta")},
    )
    _check(
        c,
        "f18_predicts_actual_form_b_geometry_not_a_scalar_surrogate",
        m.get("predicted_object") == "form_b_state"
        and str(m.get("measurement_decoder") or "").startswith("independent-form-b")
        and int(m.get("measurement_decoder_bytes", 0)) > 0
        and _number(m.get("counterfactual_form_b_cosine")) > _number(m.get("correlational_form_b_cosine"))
        and _number(m.get("cross_form_readout_acc")) > _number(m.get("chance")),
        observed={
            "object": m.get("predicted_object"),
            "decoder": m.get("measurement_decoder"),
            "cf_cosine": m.get("counterfactual_form_b_cosine"),
            "corr_cosine": m.get("correlational_form_b_cosine"),
            "readout": m.get("cross_form_readout_acc"),
        },
    )
    _check(
        c,
        "f18_unseen_value_leakage_gap_is_bounded",
        _number(m.get("unseen_value_gap")) <= float(e["leak_margin"]),
        observed=m.get("unseen_value_gap"),
        expected=f"<= {e['leak_margin']}",
    )
    expected_density = _number(m.get("counterfactual_match_acc")) / int(m.get("transport_params", 0) or 1)
    _check(
        c,
        "f18_accuracy_per_parameter_is_recomputed",
        _close(m.get("counterfactual_acc_per_param"), expected_density, atol=1.0e-7),
    )
    return {"strongest_control": strongest, "aggregate_delta": delta}


def run_candidate_verifier(
    experiment_id: str,
    *,
    repo_root: Path | str = REPO_ROOT,
    fresh_seeds: Sequence[int] = FRESH_SEEDS,
    write: bool = True,
) -> dict[str, Any]:
    if experiment_id not in CANDIDATE_POSITIVES:
        raise ValueError(f"{experiment_id!r} is not one of the locked candidate positives")
    root = Path(repo_root).resolve()
    seeds = [int(value) for value in fresh_seeds]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("verifier needs at least five unique effective seeds")

    canonical_receipt_path = root / "proof/FORM_SUBSTRATE/RECEIPTS" / f"{experiment_id}.json"
    canonical_receipt = _json(canonical_receipt_path)
    frozen_before = _freeze_live_contract(experiment_id, root)
    live_config_path = root / "configs/experiment" / f"{experiment_id}.yaml"
    live_config = _plain(OmegaConf.load(live_config_path))
    run_dir = new_run_dir(experiment_id, root=root / RAW_RUN_ROOT)
    seed_override = ",".join(str(value) for value in seeds)
    cfg = config.compose(
        [
            f"experiment={experiment_id}",
            "device=cpu",
            "result_tag=independent-adversarial-verifier",
            f"experiment.seeds=[{seed_override}]",
            "seed=19027",
        ],
        config_dir=root / "configs",
    )
    started = time.time()
    try:
        with _capture_execution(experiment_id) as trace:
            metrics = run_experiment(cfg, run_dir=run_dir)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "experiment_id": experiment_id,
            "execution_completed": False,
            "written": False,
            "run_dir": _display(root, run_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "passed": False,
            "all_ok": False,
        }
    finished = time.time()
    frozen_after = _freeze_live_contract(experiment_id, root)
    manifest_path = run_dir / "manifest.json"
    snapshot_path = run_dir / "config.yaml"
    manifest = _json(manifest_path)
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "fresh_harness_execution_completed",
        manifest.get("status") == "ok" and manifest.get("metrics") == metrics,
        observed=manifest.get("status"),
        expected="ok",
        category="execution",
    )
    common = _add_common_checks(
        experiment_id,
        metrics,
        live_config,
        canonical_receipt,
        seeds,
        trace,
        frozen_before,
        frozen_after,
        run_dir,
        root,
        checks,
    )
    recomputed = _semantic_checks(experiment_id, metrics, live_config, trace, checks)
    problems = [
        f"{check['name']}: {check.get('detail') or 'adversarial check failed'}"
        for check in checks
        if check.get("passed") is not True
    ]
    independence_checks = [check for check in checks if check.get("category") == "independence"]
    semantic_checks = [check for check in checks if check.get("category") == "semantic"]
    independent = bool(independence_checks) and all(check["passed"] for check in independence_checks)
    adversarial = bool(semantic_checks) and all(check["passed"] for check in semantic_checks)
    passed = not problems and independent and adversarial
    output = root / PROOF_DIR / f"{experiment_id}.json"
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "passed": passed,
        "all_ok": passed,
        "independent": independent,
        "adversarial": adversarial,
        "verifier_implementation": (
            "fresh-seed canonical rerun observed through independent effect capture, strongest-control "
            "recomputation, semantic invariants, and fingerprint freeze"
        ),
        "evidence_scope": "structured synthetic local fixture only; no natural-form generalization claim",
        "source_receipts": [
            {
                "path": _display(root, canonical_receipt_path),
                "sha256": _sha256(canonical_receipt_path),
                "receipt_fingerprint": canonical_receipt.get("receipt_fingerprint"),
            }
        ],
        "canonical_seeds": common["canonical_seeds"],
        "fresh_seeds_or_heldout_cases": seeds,
        "execution": {
            "completed": True,
            "run_dir": _display(root, run_dir),
            "manifest_path": _display(root, manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "config_snapshot_path": _display(root, snapshot_path),
            "config_snapshot_sha256": _sha256(snapshot_path),
            "status": manifest.get("status"),
            "device": manifest.get("device"),
            "git": manifest.get("git"),
            "started": started,
            "finished": finished,
            "seconds": finished - started,
        },
        "frozen_live_contract": frozen_after,
        "captured_effects": {
            "values": common["captured_effects"],
            "independent_ci": common["independent_ci"],
            "independent_signs": common["independent_signs"],
            "hungarian_calls": trace.hungarian_calls,
        },
        "verifier_metrics": metrics,
        "metrics_sha256": _object_sha256(metrics),
        "recomputed": recomputed,
        "checks": checks,
        "problems": problems,
        "durable_path": _display(root, output),
    }
    receipt["receipt_fingerprint"] = _object_sha256(
        {key: value for key, value in receipt.items() if key not in {"created_at", "receipt_fingerprint"}}
    )
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    receipt["written"] = bool(write)
    return receipt


def run_candidate_verifiers(
    *,
    experiment_ids: Sequence[str] | None = None,
    repo_root: Path | str = REPO_ROOT,
    fresh_seeds: Sequence[int] = FRESH_SEEDS,
    write: bool = True,
) -> dict[str, Any]:
    selected = list(experiment_ids or CANDIDATE_POSITIVES)
    unknown = set(selected) - set(CANDIDATE_POSITIVES)
    if unknown:
        raise ValueError(f"unknown candidate verifier ids: {sorted(unknown)}")
    records = [
        run_candidate_verifier(eid, repo_root=repo_root, fresh_seeds=fresh_seeds, write=write)
        for eid in selected
    ]
    return {
        "schema": "mop-form-independent-verifier-batch/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "fresh_seeds": list(fresh_seeds),
        "records": [
            {
                "experiment_id": record["experiment_id"],
                "execution_completed": record.get("execution", {}).get("completed", False),
                "run_dir": record.get("execution", {}).get("run_dir", record.get("run_dir")),
                "passed": record.get("passed", False),
                "written": record.get("written", False),
                "problems": record.get("problems", [record.get("error")])
                if record.get("error")
                else record.get("problems", []),
            }
            for record in records
        ],
        "summary": {
            "requested": len(selected),
            "executed": sum(bool(record.get("execution", {}).get("completed")) for record in records),
            "passed": sum(record.get("passed") is True for record in records),
            "failed": sum(record.get("passed") is not True for record in records),
        },
        "all_ok": all(record.get("passed") is True for record in records),
    }


def validate_verifier_receipt(receipt: Mapping[str, Any], experiment_id: str | None = None) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != SCHEMA:
        problems.append(f"unexpected verifier schema {receipt.get('schema')!r}")
    if experiment_id is not None and receipt.get("experiment_id") != experiment_id:
        problems.append("verifier experiment_id does not match expected claim")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append("verifier checks must be a non-empty list")
    elif any(not isinstance(check, Mapping) or check.get("passed") is not True for check in checks):
        problems.append("one or more verifier checks did not pass")
    if receipt.get("passed") is not True or receipt.get("all_ok") is not True:
        problems.append("verifier top-level pass flags are not true")
    if receipt.get("independent") is not True or receipt.get("adversarial") is not True:
        problems.append("verifier is not independently adversarial")
    if receipt.get("problems") not in ([], tuple()):
        problems.append("verifier has reported problems")
    frozen = receipt.get("frozen_live_contract")
    if not isinstance(frozen, Mapping):
        problems.append("frozen_live_contract missing")
    else:
        for key in ("contract_fingerprint", "config_sha256", "class_source_sha256", "verifier_sha256"):
            if not str(frozen.get(key) or ""):
                problems.append(f"frozen_live_contract.{key} missing")
    seeds = receipt.get("fresh_seeds_or_heldout_cases")
    canonical = receipt.get("canonical_seeds")
    if not isinstance(seeds, list) or len(seeds) < 5 or len(set(seeds)) != len(seeds):
        problems.append("fresh verifier seeds are missing, duplicated, or too few")
    elif isinstance(canonical, list) and set(seeds) & set(canonical):
        problems.append("fresh verifier seeds overlap canonical seeds")
    return problems


__all__ = [
    "CANDIDATE_POSITIVES",
    "FRESH_SEEDS",
    "SCHEMA",
    "ExecutionTrace",
    "run_candidate_verifier",
    "run_candidate_verifiers",
    "validate_verifier_receipt",
]


if __name__ == "__main__":  # pragma: no cover - use the repository CLI wrapper
    print("Use scripts/form_substrate_verifiers.py", file=sys.stderr)
    raise SystemExit(2)
