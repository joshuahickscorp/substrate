"""Wave I science node: the expected value of more compute, with noisy-TV rejection.

Question. A cheap predictor and an expensive predictor disagree on some examples and agree on others.
Extra compute (running the expensive predictor) only earns its keep on the fraction of examples where the
cheap answer is wrong but the expensive answer is right. Given a fixed spend budget (extra compute on only
K of N examples) the candidate is a monitor that predicts the marginal value of extra compute per example
and spends the budget on the examples where predicted value is highest. We ask whether this selective
spend beats the pooled alternatives at the SAME budget.

Named controls (the bar the candidate must clear).
  * always-spend  : run the expensive predictor on every example. This is the unconstrained ceiling and is
                    over budget (N expensive evaluations); recorded as a reference, not in the paired delta.
  * random-spend  : pick K examples at random to receive extra compute.
  * noisy-TV      : pick the K examples with the highest value of a high-variance but UNINFORMATIVE signal.
                    This is the distractor the monitor must not chase: ranking by it is no better than
                    random, so a monitor that mistakes variance for value gains nothing.

Design. Each experimental unit is an independent synthetic task stream with its own seed. Per-example
latent difficulty sets the cheap and expensive correctness through a shared uniform draw, so the expensive
answer dominates the cheap one and the realized marginal value v_i is 1 exactly on the gap examples. The
expected value of compute is the correctness gap, which is largest at intermediate difficulty. The monitor
sees only a noisy estimate of difficulty and ranks by the predicted gap. Score is accuracy at the matched
budget. The per-unit paired delta is candidate accuracy minus the best of {random-spend, noisy-TV} on the
same stream, so a positive delta favors the value-of-compute monitor. Over the units we run the framework's
exact one-sided sign-flip and derive the neutral verdict against a small structural SESOI.

Precondition (negative control). A separate regime removes the example-level structure: cheap and expensive
correctness no longer depend on difficulty, so the realized value of compute is constant across examples.
There is then nothing to select on and every matched arm ties; the monitor cannot beat random. This is the
honest boundary: selective spend wins iff the marginal value of compute varies across examples.

Honesty. A tie or a wrong-direction result is a legitimate null and nothing here is tuned toward a
positive. One run is evidence level M1: consistent with, never a scientific confirmation.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    LifecycleCost,
    assert_matched_budget,
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

_N_UNITS = 10
_N_EXAMPLES = 800
# Spend budget: extra compute is allowed on this fraction of examples per stream (matched across arms).
_BUDGET_FRACTION = 0.25
# Correctness logits: expensive shifts the intercept up, difficulty (beta) lowers both. The gap between the
# two sigmoids is the expected value of compute and peaks at intermediate difficulty.
_ALPHA_CHEAP = 0.0
_ALPHA_EXPENSIVE = 1.5
_BETA = 1.5
# The monitor sees difficulty through this much Gaussian noise: informative but imperfect.
_MONITOR_NOISE = 0.5
# The noisy-TV distractor: a high-variance signal uncorrelated with the value of compute.
_NOISY_TV_SCALE = 5.0
# Structural SESOI in accuracy points: a gain below two points is treated as no effect.
_SESOI = 0.02


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _top_k_mask(signal: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the k highest-signal examples (ties broken by index, deterministically)."""

    mask = np.zeros(signal.shape[0], dtype=bool)
    if k > 0:
        chosen = np.argsort(-signal, kind="stable")[:k]
        mask[chosen] = True
    return mask


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 when either side has no variance."""

    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _evaluate_stream(gen: np.random.Generator, n: int, k: int, separating: bool) -> dict[str, Any]:
    """Score the candidate monitor and every named control on one task stream, and return the delta."""

    difficulty = gen.standard_normal(n)
    if separating:
        lin_cheap = _ALPHA_CHEAP - _BETA * difficulty
        lin_expensive = _ALPHA_EXPENSIVE - _BETA * difficulty
    else:
        # Negative control: correctness no longer depends on difficulty, so the value of compute is
        # constant across examples and carries no example-level structure to select on.
        lin_cheap = np.full(n, _ALPHA_CHEAP)
        lin_expensive = np.full(n, _ALPHA_EXPENSIVE)

    p_cheap = _sigmoid(lin_cheap)
    p_expensive = _sigmoid(lin_expensive)
    # A shared uniform draw couples the two predictors: the expensive answer is right whenever the cheap
    # answer is, so the realized marginal value v_i is 1 exactly on the gap examples and never negative.
    draw = gen.random(n)
    correct_cheap = (draw < p_cheap).astype(np.int64)
    correct_expensive = (draw < p_expensive).astype(np.int64)
    value = correct_expensive - correct_cheap

    # Candidate monitor: predicted expected gap from a noisy difficulty estimate. It ranks by predicted
    # value of compute, never by the realized outcome (which is unknown before the spend decision).
    dhat = difficulty + _MONITOR_NOISE * gen.standard_normal(n)
    predicted_gap = _sigmoid(_ALPHA_EXPENSIVE - _BETA * dhat) - _sigmoid(_ALPHA_CHEAP - _BETA * dhat)
    noisy_tv = _NOISY_TV_SCALE * gen.standard_normal(n)
    random_signal = gen.random(n)

    def accuracy(mask: np.ndarray) -> float:
        chosen = np.where(mask, correct_expensive, correct_cheap)
        return float(np.mean(chosen))

    acc_candidate = accuracy(_top_k_mask(predicted_gap, k))
    acc_random = accuracy(_top_k_mask(random_signal, k))
    acc_noisy_tv = accuracy(_top_k_mask(noisy_tv, k))
    acc_always = accuracy(np.ones(n, dtype=bool))
    acc_cheap_only = accuracy(np.zeros(n, dtype=bool))

    best_matched_control = max(acc_random, acc_noisy_tv)
    return {
        "acc_candidate": round(acc_candidate, 12),
        "acc_random_spend": round(acc_random, 12),
        "acc_noisy_tv": round(acc_noisy_tv, 12),
        "acc_always_spend_reference": round(acc_always, 12),
        "acc_cheap_only_reference": round(acc_cheap_only, 12),
        "best_matched_control": round(best_matched_control, 12),
        "delta": round(acc_candidate - best_matched_control, 12),
        "corr_candidate_signal_value": round(_safe_corr(predicted_gap, value.astype(np.float64)), 12),
        "corr_noisy_tv_value": round(_safe_corr(noisy_tv, value.astype(np.float64)), 12),
        "fraction_value_examples": round(float(np.mean(value)), 12),
    }


@register_runner("wave_i.expected_value_of_compute")
def wave_i_value_of_compute_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Real, deterministic Wave I mechanics: does a value-of-compute monitor beat matched-budget controls?"""

    n_units = int(params.get("n_units", _N_UNITS))
    n_examples = int(params.get("n_examples", _N_EXAMPLES))
    budget_fraction = float(params.get("budget_fraction", _BUDGET_FRACTION))
    k_spend = int(round(budget_fraction * n_examples))

    units: list[dict[str, Any]] = []
    for u in range(n_units):
        gen = rng(ctx.seed, "wave_i_sep", u)
        scored = _evaluate_stream(gen, n_examples, k_spend, separating=True)
        units.append({"unit_id": f"stream-{u:02d}", **scored})

    deltas = [unit["delta"] for unit in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], _SESOI)
    is_null = verdict != "survives"

    neg_gen = rng(ctx.seed, "wave_i_negative_control")
    negative = _evaluate_stream(neg_gen, n_examples, k_spend, separating=False)
    negative_control_ties = abs(negative["delta"]) < _SESOI

    # Matched-budget accounting: every scored arm runs the cheap predictor on all N examples and the
    # expensive predictor on exactly K, so the candidate is not bought with extra compute. Always-spend
    # runs the expensive predictor on all N examples and is over the ceiling; it is a reference only.
    costs = {
        "candidate_selective": LifecycleCost(train_flops=0, inference_flops=n_examples + k_spend),
        "random_spend": LifecycleCost(train_flops=0, inference_flops=n_examples + k_spend),
        "noisy_tv_spend": LifecycleCost(train_flops=0, inference_flops=n_examples + k_spend),
    }
    budget = assert_matched_budget(costs, ceiling=n_examples + k_spend)
    always_spend_inference = 2 * n_examples

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_i_value_of_compute/v1",
        {
            "form_family": "host_telemetry",
            "phenomenon": "marginal_value_of_computation",
            "mechanism_family": "value_of_computation",
            "unit_class": "synthetic_compute_allocation_stream",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "a value-of-compute monitor: predict the marginal value of extra compute per "
            "example from a noisy difficulty estimate and spend the budget on the highest-predicted-value "
            "examples",
            "named_controls": {
                "always_spend": "run the expensive predictor on every example (over budget; the "
                "unconstrained ceiling, not in the paired delta)",
                "random_spend": "spend the extra-compute budget on K random examples",
                "noisy_tv": "spend on the K examples with the highest high-variance uninformative signal; "
                "the monitor must not chase this distractor",
            },
            "control_description": "per-unit delta is candidate accuracy minus the best of "
            "{random_spend, noisy_tv} on the same stream at the matched spend budget; positive favors the "
            "value-of-compute monitor. Always-spend is an over-budget ceiling reference only.",
            "config": {
                "n_units": n_units,
                "n_examples": n_examples,
                "budget_fraction": budget_fraction,
                "k_spend": k_spend,
                "alpha_cheap": _ALPHA_CHEAP,
                "alpha_expensive": _ALPHA_EXPENSIVE,
                "beta": _BETA,
                "monitor_noise": _MONITOR_NOISE,
                "noisy_tv_scale": _NOISY_TV_SCALE,
                "sesoi": _SESOI,
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": _SESOI,
            "verdict": verdict,
            "is_null": is_null,
            "negative_control": {
                **negative,
                "description": "correctness no longer depends on difficulty, so the value of compute is "
                "constant across examples; there is nothing to select on and the matched arms tie",
                "ties": negative_control_ties,
            },
            "matched_budget": {
                **budget,
                "always_spend_inference_flops": always_spend_inference,
                "always_spend_over_ceiling": always_spend_inference > (n_examples + k_spend),
            },
            "noisy_tv_rejection": {
                "mean_corr_candidate_signal_value": round(
                    float(np.mean([unit["corr_candidate_signal_value"] for unit in units])), 12
                ),
                "mean_corr_noisy_tv_value": round(
                    float(np.mean([unit["corr_noisy_tv_value"] for unit in units])), 12
                ),
                "description": "the candidate signal correlates with the realized value of compute while "
                "the noisy-TV signal does not; chasing variance buys no accuracy",
            },
            "alternative_explanation": "The gain reflects the monitor reading example-level difficulty "
            "structure that the pooled controls cannot exploit at the same budget; it is bounded to the "
            "regime where the marginal value of compute varies across examples and collapses to a tie when "
            "that value is constant.",
            "failure_domain": "When the value of compute carries no example-level structure (constant gap, "
            "the negative control), selective spend does not beat random and the delta ties at zero; the "
            "monitor also gains nothing by chasing the high-variance noisy-TV distractor.",
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "negative_control_ties": negative_control_ties,
        },
    )
