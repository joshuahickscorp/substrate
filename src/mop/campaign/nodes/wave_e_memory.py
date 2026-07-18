"""Wave E memory ecology: does experience replay improve the retention-versus-future-learning frontier?

Continual learning poses a stability-plasticity tension. A learner adapted to an old task must adapt to a
new one after an abrupt change point without erasing what it knew. Experience replay (mixing a buffer of
old samples into the new-task updates) is the standard candidate for buying stability. The honest question
this bed asks is not whether replay preserves the old task (it trivially can, by refusing to move) but
whether it improves the joint frontier: better retention of the old task WITHOUT sacrificing learning of
the new task, under a matched compute budget.

Design. Each of N independent drift streams is a linear-regression online task. Phase A trains from a zero
init toward a random target w_A while filling a replay buffer. At an abrupt change point the target switches
to a near-orthogonal w_B. Three arms then run under a matched gradient-evaluation budget:

* candidate: experience replay. Each phase-B update mixes one fresh task-B sample with a minibatch of
  replayed task-A samples at equal weight (the canonical naive 50/50 replay).
* named control 1 (no-replay): plain online SGD on the task-B stream from the same phase-A weights.
* named control 2 (fresh-init): reset the weights to zero at the change point and learn task B from scratch.
  This anchors the plasticity ceiling (best achievable future learning, zero retention) and confirms that
  the no-replay arm's forgetting is intrinsic to the drift, not an artifact of its starting point.

Retention error is the normalized task-A test error after adapting; future error is the normalized task-B
test error. Both are divided by the task's own null-predictor variance so units on different random targets
are comparable. The per-unit combined frontier score credits a retention improvement ONLY when future
learning is preserved within a small structural tolerance; if replay improves retention by giving up future
learning, the score is the (negative) size of that sacrifice. That operationalizes the mandate directly:
replay must improve the frontier, not win by preserving obsolete behavior.

The exact one-sided sign-flip over the independent-unit deltas and the neutral SESOI verdict decide the
outcome. A tie or a wrong-direction result is a legitimate null and is reported as such: this bed does not
tune toward a positive.

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

# --- Fixed design constants (no wall clock, no tuning knobs exposed to params). ---
N_UNITS = 12
DIM = 12
PHASE_A_STEPS = 300
PHASE_B_REPLAY_STEPS = 600
REPLAY_K = 4  # replayed old-task samples mixed into each phase-B update
BUFFER_CAP = 256
LR = 0.05
NOISE_STD = 0.10
N_TEST = 2000
FUTURE_TOLERANCE = 0.02  # tau: replay may cost at most this fraction of task-B variance and still count
SESOI = 0.05  # smallest normalized-variance frontier gain we would call a real effect
ALPHA = 0.05

# Matched compute: the controls take as many single-sample gradient steps as replay takes gradient
# evaluations (one fresh plus REPLAY_K replayed) so no arm is advantaged by raw compute.
CONTROL_B_STEPS = PHASE_B_REPLAY_STEPS * (1 + REPLAY_K)


def _unit_vector(gen: np.random.Generator, dim: int) -> np.ndarray:
    v = gen.standard_normal(dim)
    return v / np.linalg.norm(v)


def _make_stream(gen: np.random.Generator, w_true: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = gen.standard_normal((n, DIM))
    y = x @ w_true + NOISE_STD * gen.standard_normal(n)
    return x, y


def _sgd_online(w: np.ndarray, x: np.ndarray, y: np.ndarray, lr: float) -> np.ndarray:
    w = w.copy()
    for i in range(x.shape[0]):
        xi = x[i]
        w = w - lr * (float(xi @ w) - y[i]) * xi
    return w


def _sgd_replay(
    w: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    buf_x: np.ndarray,
    buf_y: np.ndarray,
    lr: float,
    gen: np.random.Generator,
    replay_k: int,
) -> np.ndarray:
    w = w.copy()
    n_buf = buf_x.shape[0]
    for i in range(xb.shape[0]):
        xi = xb[i]
        g_fresh = (float(xi @ w) - yb[i]) * xi
        idx = gen.integers(0, n_buf, size=replay_k)
        rx = buf_x[idx]
        resid = rx @ w - buf_y[idx]
        g_replay = (resid[:, None] * rx).mean(axis=0)
        w = w - lr * (g_fresh + g_replay)
    return w


def _normalized_error(w: np.ndarray, x_test: np.ndarray, w_true: np.ndarray) -> float:
    """Task test error as a fraction of the null-predictor variance (noiseless targets)."""

    y_true = x_test @ w_true
    pred = x_test @ w
    mse = float(np.mean((pred - y_true) ** 2))
    null = float(np.mean(y_true**2))
    return mse / null


def _run_unit(seed: int, u: int) -> dict[str, Any]:
    w_a = _unit_vector(rng(seed, "unit", u, "w_a"), DIM)
    w_b = _unit_vector(rng(seed, "unit", u, "w_b"), DIM)

    xa, ya = _make_stream(rng(seed, "unit", u, "train_a"), w_a, PHASE_A_STEPS)
    w_after_a = _sgd_online(np.zeros(DIM), xa, ya, LR)
    buf_x = xa[-BUFFER_CAP:]
    buf_y = ya[-BUFFER_CAP:]

    xb, yb = _make_stream(rng(seed, "unit", u, "train_b"), w_b, CONTROL_B_STEPS)

    w_replay = _sgd_replay(
        w_after_a,
        xb[:PHASE_B_REPLAY_STEPS],
        yb[:PHASE_B_REPLAY_STEPS],
        buf_x,
        buf_y,
        LR,
        rng(seed, "unit", u, "replay_draw"),
        REPLAY_K,
    )
    w_no_replay = _sgd_online(w_after_a, xb, yb, LR)
    w_fresh = _sgd_online(np.zeros(DIM), xb, yb, LR)

    xt_a = rng(seed, "unit", u, "test_a").standard_normal((N_TEST, DIM))
    xt_b = rng(seed, "unit", u, "test_b").standard_normal((N_TEST, DIM))

    ret_replay = _normalized_error(w_replay, xt_a, w_a)
    fut_replay = _normalized_error(w_replay, xt_b, w_b)
    ret_no_replay = _normalized_error(w_no_replay, xt_a, w_a)
    fut_no_replay = _normalized_error(w_no_replay, xt_b, w_b)
    ret_fresh = _normalized_error(w_fresh, xt_a, w_a)
    fut_fresh = _normalized_error(w_fresh, xt_b, w_b)

    retention_gain = ret_no_replay - ret_replay  # positive when replay retains the old task better
    future_sacrifice = fut_replay - fut_no_replay  # positive when replay gives up new-task learning
    preserved_future = future_sacrifice <= FUTURE_TOLERANCE
    # Combined frontier score: credit the retention gain only if future learning is preserved; otherwise
    # the score is the negative size of the sacrifice. Replay cannot win by preserving obsolete behavior.
    delta = retention_gain if preserved_future else -future_sacrifice

    return {
        "unit_id": f"u{u:02d}",
        "retention_error_replay": round(ret_replay, 9),
        "future_error_replay": round(fut_replay, 9),
        "retention_error_no_replay": round(ret_no_replay, 9),
        "future_error_no_replay": round(fut_no_replay, 9),
        "retention_error_fresh_init": round(ret_fresh, 9),
        "future_error_fresh_init": round(fut_fresh, 9),
        "retention_gain": round(retention_gain, 9),
        "future_sacrifice": round(future_sacrifice, 9),
        "preserved_future": bool(preserved_future),
        "delta": round(delta, 9),
    }


def _budget() -> dict[str, Any]:
    """Matched-budget lifecycle accounting. FLOP proxy is (gradient evaluations) times DIM per arm,
    summed across the N units; inference is the shared two-task test pass. Replay and no-replay share an
    identical phase-B gradient budget; fresh-init skips the phase-A pass by construction."""

    per_grad = DIM
    train_replay = PHASE_A_STEPS + PHASE_B_REPLAY_STEPS * (1 + REPLAY_K)
    train_no_replay = PHASE_A_STEPS + CONTROL_B_STEPS
    train_fresh = CONTROL_B_STEPS
    infer_evals = 2 * N_TEST  # both tasks, per arm
    costs = {
        "replay": LifecycleCost(N_UNITS * train_replay * per_grad, N_UNITS * infer_evals * per_grad),
        "no_replay": LifecycleCost(N_UNITS * train_no_replay * per_grad, N_UNITS * infer_evals * per_grad),
        "fresh_init": LifecycleCost(N_UNITS * train_fresh * per_grad, N_UNITS * infer_evals * per_grad),
    }
    ceiling = max(c.total for c in costs.values())
    report = assert_matched_budget(costs, ceiling)
    report["primary_pair_matched"] = costs["replay"].train_flops == costs["no_replay"].train_flops
    return report


@register_runner("wave_e.replay_retention_future_learning")
def wave_e_memory_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Deterministic Wave E bed: replay versus no-replay and fresh-init on the retention-future frontier."""

    units = [_run_unit(ctx.seed, u) for u in range(N_UNITS)]
    deltas = [u["delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    mean_delta = sign_flip["mean_delta"]
    verdict = verdict_from(mean_delta, sign_flip["one_sided_p"], SESOI, ALPHA)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_e_memory/v1",
        {
            "form_family": "memory_episode",
            "phenomenon": "continual_learning",
            "mechanism_family": "experience_replay",
            "unit_class": "linear_regression_drift_stream",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "experience_replay_equal_weight",
            "controls": {
                "primary": "no_replay_online_sgd",
                "secondary": "fresh_init_at_change_point",
            },
            "control_description": (
                "Primary control is plain online SGD on the task-B stream from the same phase-A weights "
                "(no replay). Secondary control resets the weights to zero at the change point and learns "
                "task B from scratch, anchoring the plasticity ceiling and confirming the no-replay arm's "
                "forgetting is intrinsic to the abrupt near-orthogonal drift."
            ),
            "design": {
                "n_units": N_UNITS,
                "dim": DIM,
                "phase_a_steps": PHASE_A_STEPS,
                "phase_b_replay_steps": PHASE_B_REPLAY_STEPS,
                "control_b_steps": CONTROL_B_STEPS,
                "replay_k": REPLAY_K,
                "buffer_cap": BUFFER_CAP,
                "learning_rate": LR,
                "noise_std": NOISE_STD,
                "n_test": N_TEST,
                "frontier_score": (
                    "delta = retention_gain if future_sacrifice <= tau else -future_sacrifice; "
                    "positive only when replay preserves future learning within tau"
                ),
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "future_tolerance_tau": FUTURE_TOLERANCE,
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "verdict": verdict,
            "is_null": is_null,
            "budget": _budget(),
            "alternative_explanation": (
                "A negative combined score is specific to canonical equal-weight replay under a matched "
                "compute budget. A much weaker replay mixture would trade less future learning for a "
                "smaller retention gain and could pass the future-preservation gate, so the sign of the "
                "frontier effect is a property of the replay strength and the compute split, not of "
                "replay as a category. Under matched compute, replay also diverts gradient steps onto old "
                "samples, so part of any future-learning shortfall is a data-budget effect rather than "
                "interference alone."
            ),
            "failure_domain": (
                "Linear-regression online continual learning with an abrupt, near-orthogonal target change "
                "and standard-normal inputs. It does not speak to nonlinear or learned features, gradual "
                "or recurring drift, capacity-limited or prioritized buffer selection, or regimes where the "
                "old and new tasks share structure that replay could transfer."
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
