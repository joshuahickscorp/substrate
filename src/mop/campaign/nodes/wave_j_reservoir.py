"""Wave J science node: does a fading-memory reservoir earn its keep over conventional digital readouts?

Reservoir computing proposes that a fixed random recurrent dynamical system (the reservoir) plus a single
trained linear readout can solve temporal tasks that a memoryless map cannot, because the reservoir's state
holds a fading memory of recent inputs and mixes them nonlinearly. The honest question this bed asks is not
whether recurrence beats a strictly memoryless map (it trivially can, on any task that needs the past) but
whether the reservoir's nonlinear echo-state expansion earns its keep against a strong conventional control
that already has explicit short memory: a matched-parameter delay line feeding the same kind of linear
readout.

Design. Each of N independent units is a nonlinear short-memory sequence-prediction task. The scalar input
u_t is iid uniform. The target depends only on recent lags through nonlinear products,
y_t = c0 * u_{t-1} * u_{t-2} + c1 * u_{t-2} * u_{t-3} + c2 * u_{t-1} + observation noise, with random
per-unit coefficients. Three arms fit a ridge-regression linear readout on a shared train split and are
scored on a held-out test split by normalized mean-squared error (NMSE, error divided by target variance):

* candidate: an echo-state reservoir. A fixed random recurrent matrix scaled below unit spectral radius (the
  echo-state / fading-memory condition) is driven by the input through a random input map with a tanh
  nonlinearity; the N_res reservoir states plus a bias feed the linear readout.
* named control 1 (memoryless): a linear readout of the current input u_t plus a bias, with no memory and no
  recurrence. It anchors the floor: the target carries no u_t term, so a memoryless map can do no better than
  predicting the mean.
* named control 2 (delay line): a matched-parameter tapped delay line. The readout sees N_res raw past
  inputs u_t, u_{t-1}, ..., u_{t-N_res+1} plus a bias, so it has the same trainable readout width as the
  reservoir and explicit linear short memory, but no nonlinear mixing of lags.

The score per unit is negative NMSE. The paired per-unit delta is the candidate score minus the BEST (lowest
error) control score, which equals best_control_nmse minus reservoir_nmse: a positive delta means the
reservoir beats the strongest conventional readout available. Because the delay line already captures the
linear u_{t-1} term at matched parameter count, any reservoir advantage must come from representing the lag
products, not merely from having memory.

The exact one-sided sign-flip over the independent-unit deltas and the neutral SESOI verdict decide the
outcome. A tie or a wrong-direction result is a legitimate null and is reported as such: this bed does not
tune toward a positive.

Boundary. This is a numpy simulation of the readout geometry only. It cannot prove any physical property of
a real reservoir substrate: it says nothing about device drift, energy per inference, thermal noise, or
fabrication variability, and the artifact states this explicitly.

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

# --- Fixed design constants (no wall clock, no tuning knobs exposed to params). ---
N_UNITS = 10
N_RES = 60  # reservoir size; also the delay-line tap count so readout widths match at N_RES + 1
SPECTRAL_RADIUS = 0.9  # below 1 for the echo-state / fading-memory condition
INPUT_SCALE = 1.0
WASHOUT = 60
TRAIN = 600
TEST = 400
RIDGE_ALPHA = 1e-3
NOISE_STD = 0.05
SESOI = 0.02  # smallest normalized-variance NMSE improvement we would call a real effect
ALPHA = 0.05


def _ridge_readout(phi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge readout weights for design matrix phi and targets y."""

    gram = phi.T @ phi + alpha * np.eye(phi.shape[1])
    return np.linalg.solve(gram, phi.T @ y)


def _fit_predict_nmse(
    features: np.ndarray,
    y: np.ndarray,
    train_slice: slice,
    test_slice: slice,
    alpha: float,
) -> float:
    """Fit a bias-augmented ridge readout on the train split and return held-out NMSE on the test split."""

    ones = np.ones((features.shape[0], 1))
    phi = np.hstack([features, ones])
    weights = _ridge_readout(phi[train_slice], y[train_slice], alpha)
    pred = phi[test_slice] @ weights
    resid = pred - y[test_slice]
    y_var = float(np.var(y[test_slice]))
    if y_var <= 0.0:
        return float("inf")
    return float(np.mean(resid**2) / y_var)


def _make_task(seed: int, u_idx: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """A nonlinear short-memory sequence task: the target mixes recent lags through products."""

    gen = rng(seed, "unit", u_idx, "task")
    u = gen.uniform(-1.0, 1.0, size=length)
    u1 = np.roll(u, 1)
    u1[:1] = 0.0
    u2 = np.roll(u, 2)
    u2[:2] = 0.0
    u3 = np.roll(u, 3)
    u3[:3] = 0.0
    coeff = gen.uniform(0.3, 0.7, size=3) * gen.choice(np.array([-1.0, 1.0]), size=3)
    noise = NOISE_STD * gen.standard_normal(length)
    y = coeff[0] * u1 * u2 + coeff[1] * u2 * u3 + coeff[2] * u1 + noise
    return u, y


def _reservoir_states(seed: int, u_idx: int, u: np.ndarray) -> np.ndarray:
    """Drive a fixed random echo-state reservoir with the input sequence and collect its states."""

    gen = rng(seed, "unit", u_idx, "reservoir")
    w_rec = gen.standard_normal((N_RES, N_RES))
    radius = float(np.max(np.abs(np.linalg.eigvals(w_rec))))
    w_rec = w_rec * (SPECTRAL_RADIUS / radius)
    w_in = gen.uniform(-1.0, 1.0, size=N_RES) * INPUT_SCALE
    states = np.zeros((u.shape[0], N_RES))
    x = np.zeros(N_RES)
    for t in range(u.shape[0]):
        x = np.tanh(w_in * u[t] + w_rec @ x)
        states[t] = x
    return states


def _delay_features(u: np.ndarray, taps: int) -> np.ndarray:
    """Matched-parameter tapped delay line: columns hold u_t, u_{t-1}, ..., u_{t-(taps-1)}, zero padded."""

    feats = np.zeros((u.shape[0], taps))
    for k in range(taps):
        if k == 0:
            feats[:, 0] = u
        else:
            feats[k:, k] = u[:-k]
    return feats


def _run_unit(seed: int, u_idx: int) -> dict[str, Any]:
    length = WASHOUT + TRAIN + TEST
    u, y = _make_task(seed, u_idx, length)
    train_slice = slice(WASHOUT, WASHOUT + TRAIN)
    test_slice = slice(WASHOUT + TRAIN, length)

    states = _reservoir_states(seed, u_idx, u)
    nmse_reservoir = _fit_predict_nmse(states, y, train_slice, test_slice, RIDGE_ALPHA)

    memoryless = u.reshape(-1, 1)
    nmse_memoryless = _fit_predict_nmse(memoryless, y, train_slice, test_slice, RIDGE_ALPHA)

    delay = _delay_features(u, N_RES)
    nmse_delay = _fit_predict_nmse(delay, y, train_slice, test_slice, RIDGE_ALPHA)

    best_control_nmse = min(nmse_memoryless, nmse_delay)
    best_control = "delay_line" if nmse_delay <= nmse_memoryless else "memoryless"
    # Score is negative NMSE; delta is candidate score minus best control score = best_control - reservoir.
    delta = best_control_nmse - nmse_reservoir

    return {
        "unit_id": f"u{u_idx:02d}",
        "nmse_reservoir": round(nmse_reservoir, 9),
        "nmse_memoryless": round(nmse_memoryless, 9),
        "nmse_delay_line": round(nmse_delay, 9),
        "best_control": best_control,
        "best_control_nmse": round(best_control_nmse, 9),
        "delta": round(delta, 9),
    }


@register_runner("wave_j.reservoir_vs_conventional")
def wave_j_reservoir_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Deterministic Wave J bed: reservoir versus memoryless and matched-parameter delay-line controls."""

    units = [_run_unit(ctx.seed, u_idx) for u_idx in range(N_UNITS)]
    deltas = [u["delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    mean_delta = sign_flip["mean_delta"]
    verdict = verdict_from(mean_delta, sign_flip["one_sided_p"], SESOI, ALPHA)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_j_reservoir/v1",
        {
            "form_family": "material_sim",
            "phenomenon": "alternate_compute_dynamics",
            "mechanism_family": "fading_memory_reservoir",
            "unit_class": "nonlinear_short_memory_sequence_task",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "echo_state_reservoir_linear_readout",
            "controls": {
                "primary": "matched_parameter_delay_line_linear_readout",
                "secondary": "memoryless_linear_readout",
            },
            "control_description": (
                "The primary control is a matched-parameter tapped delay line: the readout sees N_res raw "
                "past inputs plus a bias, giving it the same trainable readout width as the reservoir and "
                "explicit linear short memory, but no nonlinear mixing of lags. The secondary control is a "
                "memoryless linear readout of only the current input plus a bias, which anchors the floor "
                "because the target carries no current-input term. The per-unit delta pits the reservoir "
                "against the best (lowest error) of the two controls, so any advantage must come from "
                "representing the lag products rather than from merely having memory."
            ),
            "design": {
                "n_units": N_UNITS,
                "reservoir_size": N_RES,
                "delay_taps": N_RES,
                "spectral_radius": SPECTRAL_RADIUS,
                "input_scale": INPUT_SCALE,
                "washout": WASHOUT,
                "train_steps": TRAIN,
                "test_steps": TEST,
                "ridge_alpha": RIDGE_ALPHA,
                "noise_std": NOISE_STD,
                "target": (
                    "y_t = c0 * u_{t-1} * u_{t-2} + c1 * u_{t-2} * u_{t-3} + c2 * u_{t-1} + noise; "
                    "random per-unit coefficients; iid uniform input"
                ),
                "score": "negative NMSE (test error divided by target variance)",
                "delta": "best_control_nmse minus reservoir_nmse; positive favors the reservoir",
                "readout_width_matched": "reservoir and delay line both use N_res + 1 readout weights",
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "verdict": verdict,
            "is_null": is_null,
            "alternative_explanation": (
                "Any reservoir advantage here is a property of this task's specific nonlinearity (pairwise "
                "lag products) meeting a linear-only delay-line control. A delay line augmented with "
                "explicit product features, or a task whose target is linear in its lags, would erase the "
                "gap, so the "
                "sign of the effect reflects the match between the reservoir's tanh mixing and the target "
                "nonlinearity, not a universal superiority of recurrence. Fixed ridge and a fixed random "
                "reservoir instantiation also leave headroom that per-unit tuning could shift either way."
            ),
            "failure_domain": (
                "Numpy simulation of the readout geometry for a single fixed reservoir topology (dense "
                "gaussian recurrence at spectral radius 0.9, tanh activation) on short-memory product tasks "
                "with iid uniform inputs. It does not speak to long-memory or chaotic-input tasks, other "
                "reservoir topologies or activations, or online rather than batch-ridge readouts."
            ),
            "simulation_boundary": (
                "Simulation only. This bed models the reservoir as an idealized numerical dynamical system "
                "and cannot demonstrate any physical property of a real reservoir substrate: it makes no "
                "claim about device drift, energy per inference, thermal noise, or fabrication variability."
            ),
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "n_units": sign_flip["n_units"],
        },
    )
