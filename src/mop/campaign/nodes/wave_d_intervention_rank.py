"""Wave D science node: does access to interventional samples earn its keep for ranking interventions?

Wave D here asks a causal-agency question on independent synthetic linear structural causal models (SCMs).
Each unit is a small known DAG: k manipulable root variables A_1..A_k, one unobserved confounder H that
loads onto every A_i and also onto the outcome Y, and Y = sum_i beta_i A_i + gamma H + noise. Because H is
hidden, the observational joint distribution is confounded: the correlational picture of "which variable
moves Y most" is biased away from the true causal effect ordering.

The candidate is a model that also has access to interventional samples: for each variable it draws a batch
where that variable is randomized (a do-intervention that severs its dependence on H) and estimates the
per-variable causal slope, which is an unbiased estimate of beta_i. The named control is a purely
correlational ranker with no interventional data at all: it fits the best observational estimator available,
a multiple ordinary-least-squares regression of Y on every A jointly, and ranks by the resulting
coefficients. Both rank the variables by estimated effect magnitude; each is scored by the Spearman rank
correlation with the true ordering by |beta_i|. The paired per-unit delta is candidate_spearman minus
control_spearman, so a positive delta favors the interventional ranker. A tie, a wrong-direction delta, or
an effect below the structural SESOI is a legitimate null and is never tuned away.

Determinism: every random draw is seeded from ctx.seed via the framework rng. No wall clock enters the
sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties share the mean of their positions), stdlib-and-numpy only."""

    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(float((xc * xc).sum()) * float((yc * yc).sum())))
    if denom == 0.0:
        return 0.0
    return float((xc * yc).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks of ``x`` and ``y``."""

    return _pearson(_rankdata(x), _rankdata(y))


def _simulate_observational(
    gen: np.random.Generator,
    n: int,
    beta: np.ndarray,
    c: np.ndarray,
    gamma: float,
    sig_a: float,
    sig_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw confounded observational data from the SCM: A_i = c_i H + eps_i, Y = A beta + gamma H + eps."""

    k = len(beta)
    h = gen.normal(0.0, 1.0, n)
    a = c[None, :] * h[:, None] + gen.normal(0.0, sig_a, (n, k))
    y = a @ beta + gamma * h + gen.normal(0.0, sig_y, n)
    return a, y


def _ols_coeffs(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The strongest correlational baseline available: multiple OLS slopes with an intercept."""

    design = np.column_stack([np.ones(len(y)), a])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs[1:]


def _interventional_slope(
    gen: np.random.Generator,
    i: int,
    n: int,
    beta: np.ndarray,
    c: np.ndarray,
    gamma: float,
    sig_a: float,
    sig_y: float,
    sig_int: float,
) -> float:
    """Estimate beta_i from a do-intervention batch: A_i is randomized, severing its link to H."""

    k = len(beta)
    h = gen.normal(0.0, 1.0, n)
    a = c[None, :] * h[:, None] + gen.normal(0.0, sig_a, (n, k))
    a[:, i] = gen.normal(0.0, sig_int, n)
    y = a @ beta + gamma * h + gen.normal(0.0, sig_y, n)
    xi = a[:, i]
    var_xi = float(np.var(xi, ddof=1))
    if var_xi == 0.0:
        return 0.0
    return float(np.cov(xi, y, ddof=1)[0, 1] / var_xi)


def _run_unit(
    unit: int,
    base_seed: int,
    k: int,
    n_obs: int,
    n_int: int,
    sig_a: float,
    sig_y: float,
    sig_int: float,
) -> dict[str, Any]:
    """One independent SCM: fit both rankers, score each by Spearman against the true |beta| ordering."""

    gen = rng(base_seed, "scm", unit)
    beta = gen.uniform(0.5, 2.5, k)
    c = gen.uniform(-1.5, 1.5, k)
    gamma = float(gen.uniform(1.0, 2.0))

    a_obs, y_obs = _simulate_observational(rng(base_seed, "obs", unit), n_obs, beta, c, gamma, sig_a, sig_y)
    control_est = np.abs(_ols_coeffs(a_obs, y_obs))
    candidate_est = np.abs(
        np.array(
            [
                _interventional_slope(
                    rng(base_seed, "do", unit, i), i, n_int, beta, c, gamma, sig_a, sig_y, sig_int
                )
                for i in range(k)
            ]
        )
    )

    true_effect = beta  # all positive, so effect magnitude ordering is the ordering by beta
    candidate_spearman = _spearman(candidate_est, true_effect)
    control_spearman = _spearman(control_est, true_effect)

    return {
        "unit_id": f"scm_{unit:02d}",
        "true_effects": [round(float(v), 12) for v in true_effect],
        "confounder_loadings": [round(float(v), 12) for v in c],
        "gamma": round(gamma, 12),
        "candidate_effect_estimates": [round(float(v), 12) for v in candidate_est],
        "control_effect_estimates": [round(float(v), 12) for v in control_est],
        "candidate_spearman": round(float(candidate_spearman), 12),
        "control_spearman": round(float(control_spearman), 12),
        "delta": round(float(candidate_spearman - control_spearman), 12),
    }


@register_runner("wave_d.intervention_ranking")
def wave_d_intervention_rank_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Wave D: an interventional ranker must recover the true causal intervention ordering better than the
    strongest correlational ranker on confounded observational data before it can be called anything more
    than correlational. A tie or a below-SESOI gain is a legitimate null.
    """

    n_units = int(params.get("n_units", 10))
    k = int(params.get("k", 6))
    n_obs = int(params.get("n_obs", 600))
    n_int = int(params.get("n_int", 400))
    sig_a = float(params.get("sig_a", 0.7))
    sig_y = float(params.get("sig_y", 1.0))
    sig_int = float(params.get("sig_int", 2.0))
    n_units = max(8, min(n_units, 22))  # keep the exact sign-flip enumerable and meaningful
    k = max(3, k)

    units = [_run_unit(u, ctx.seed, k, n_obs, n_int, sig_a, sig_y, sig_int) for u in range(n_units)]
    deltas = [u["delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)

    # A small structural floor: one adjacent rank transposition's worth of Spearman correlation.
    sesoi = round(12.0 / (k * (k * k - 1)), 12)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], sesoi)
    is_null = verdict != "survives"

    mean_candidate = round(float(np.mean([u["candidate_spearman"] for u in units])), 12)
    mean_control = round(float(np.mean([u["control_spearman"] for u in units])), 12)

    content = {
        **honest_envelope(
            ctx.node_id,
            "mop-campaign-wave_d_intervention_rank/v1",
            {
                "form_family": "action",
                "phenomenon": "causal_intervention",
                "mechanism_family": "world_model",
                "unit_class": "linear_scm_dag",
                "evidence_level": "M1",
            },
        ),
        "design": {
            "n_units": n_units,
            "variables_per_unit": k,
            "n_observational": n_obs,
            "n_interventional_per_variable": n_int,
            "interventional_budget_total": n_int * k,
            "noise": {"sig_a": sig_a, "sig_y": sig_y, "sig_int": sig_int},
            "scm": (
                "A_i = c_i * H + eps_i with hidden confounder H; Y = sum_i beta_i A_i + gamma H + eps_y. "
                "H loads on every A_i and on Y, so the observational distribution is confounded and the "
                "true causal ordering is by |beta_i|, not by observational association."
            ),
        },
        "control_description": (
            "Named control is a purely correlational ranker with no interventional data: it fits the "
            "strongest observational estimator available, a multiple ordinary-least-squares regression of "
            "Y on all A jointly, and ranks variables by the magnitude of the fitted coefficients. Because "
            "H is unobserved this estimator carries omitted-variable bias proportional to the confounder "
            "loadings, so its ranking is systematically distorted relative to the true causal ordering."
        ),
        "per_unit": units,
        "ranking": {
            "score": "Spearman rank correlation between estimated |effect| ordering and true |beta| ordering",
            "delta_definition": "candidate_spearman minus control_spearman; positive favors interventions",
            "deltas": deltas,
            "sign_flip": sign_flip,
            "sesoi": sesoi,
            "sesoi_rationale": (
                "one adjacent rank transposition's worth of Spearman, 12/(k*(k*k-1)); below this a rank "
                "correlation gain is scientifically negligible"
            ),
            "mean_candidate_spearman": mean_candidate,
            "mean_control_spearman": mean_control,
        },
        "verdict": verdict,
        "is_null": is_null,
        "alternative_explanation": (
            "The interventional advantage could be an artifact of estimation variance rather than causal "
            "identification: with finite interventional batches the do-slope estimates carry sampling noise, "
            "and if the confounding in a unit is weak the correlational multiple regression is already "
            "near-unbiased, so the two rankers tie. The sign-flip over independent units and the structural "
            "SESOI guard against reading unit-level noise as a real ranking gain."
        ),
        "failure_domain": (
            "Nonlinear or non-additive dynamics a linear slope cannot summarize, settings where the "
            "confounder is actually observed (so plain regression already identifies the effect and "
            "interventions add nothing), or interventions too small in range to estimate the causal slope "
            "above outcome noise."
        ),
    }

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "mean_candidate_spearman": mean_candidate,
            "mean_control_spearman": mean_control,
        },
    )
