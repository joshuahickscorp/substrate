"""Wave F science node: does a continual-reset mechanism repair lost plasticity in an online learner?

This is a deterministic mechanics experiment. Each independent unit is a non-stationary stream: a warm
started one-hidden-layer tanh network is trained by full-batch gradient descent on a sequence of random
teacher tasks, carrying its weights across task boundaries. Under plain online SGD such a learner tends to
lose plasticity: hidden units saturate (their tanh pre-activations grow large so their gradient vanishes)
and post-drift error on the later tasks rises. The candidate mechanism is continual reset, continual
backprop style: at each task boundary it re-initializes the least useful hidden units (lowest outgoing
weight times mean absolute activation), setting their outgoing weight to zero so the reset does not disturb
the current output, which unsaturates dead units and restores their ability to relearn.

Three named controls share the identical per-stream task sequence so the comparison is paired:

* plain SGD, warm started, no reset (the primary control the per-unit delta is measured against);
* a frozen representation with a larger shell, whose hidden features stay at their random initialization
  (frozen random features cannot saturate-lock) while only a wider linear readout adapts each task; and
* fresh init, which reinitializes the whole network at every task boundary (no plasticity loss but no
  transfer either).

The score per stream is the negative mean post-drift error (the last half of the task sequence). The per
unit paired delta is sgd_error minus candidate_error, so a positive delta favors the reset. The exact one
sided sign-flip over the streams plus a small structural SESOI decide the verdict, and the candidate must
additionally beat both the frozen-plus-larger-shell control and the fresh-init control on the mean or the
result is a null. A tie or wrong-direction outcome is a legitimate null and nothing here is tuned toward a
positive: the alternative explanation is that well-tuned SGD already retains plasticity so there is nothing
to repair.

Imports: stdlib, numpy, mop.campaign.runners, mop.campaign.nodes.framework only.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
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

SCHEMA = "mop-campaign-wave_f_plasticity/v1"

# Structural experiment constants. These are fixed design choices for eliciting the plasticity question,
# not knobs fitted to any outcome.
M_IN = 12  # input dimension of every stream
H = 24  # hidden width of the plain SGD and continual-reset learners
H_SHELL = 48  # larger frozen-feature shell width for the frozen-representation control
H_TEACHER = 16  # hidden width of the random teacher that generates each task
N_TASKS = 16  # tasks per stream; weights carry across task boundaries (continual learning)
N_TRAIN = 48  # training examples per task
N_TEST = 48  # held-out examples per task, used to score post-drift error
GD_STEPS = 40  # full-batch gradient descent steps per task
LR = 0.1  # learning rate
RESET_FRAC = 0.15  # fraction of hidden units the candidate re-initializes at each task boundary
POST_DRIFT_TASKS = 8  # score is averaged over the last this-many tasks (the post-drift regime)
N_STREAMS = 12  # independent experimental units (<= 22 so the exact sign-flip enumerates)
SESOI = 0.01  # smallest meaningful reduction in standardized held-out MSE
ERR_CAP = 1000.0  # finite ceiling so a diverged arm never seals a non-finite value

# Descriptive strings hoisted to module scope so the sealed-content dict lines stay within the width limit.
_CANDIDATE_DESC = (
    "continual reset: at each task boundary re-initialize the least useful "
    f"{RESET_FRAC:.2f} fraction of hidden units (lowest outgoing weight times mean absolute "
    "activation), zeroing their outgoing weight so the reset does not disturb the output"
)
_PRIMARY_SGD_DESC = "plain warm-started SGD with no reset (the per-unit delta is measured against it)"
_FROZEN_SHELL_DESC = (
    f"frozen random features of width {H_SHELL} that never update, with only the wider linear "
    "readout adapting each task"
)
_FRESH_INIT_DESC = "reinitialize the whole network at every task boundary (no plasticity loss, no transfer)"
_SCORE_DEF = "negative mean standardized held-out MSE over the last POST_DRIFT_TASKS tasks"
_DELTA_DEF = "sgd_error minus candidate_error per stream; positive favors the reset candidate"
_ALT_EXPLANATION = (
    "well-tuned SGD already retains plasticity so there is nothing to repair; the reset then only "
    "adds noise and cannot lower post-drift error"
)
_FAILURE_DOMAIN = (
    "stationary streams, streams too short for any hidden unit to saturate, or readout-only convex "
    "learners: none of these can lose plasticity so continual reset has nothing to restore there"
)

_Params = dict[str, np.ndarray]


def _finite(value: float) -> float:
    """Clamp to a finite, rounded scalar so the sealed artifact never carries NaN or Inf."""

    v = float(value)
    if not math.isfinite(v):
        return ERR_CAP
    return round(min(v, ERR_CAP), 9)


def _init_learner(gen: np.random.Generator, width: int) -> _Params:
    """Small random one-hidden-layer network with fan-in scaled weights and zero biases."""

    return {
        "W1": gen.standard_normal((width, M_IN)) / math.sqrt(M_IN),
        "b1": np.zeros(width),
        "W2": gen.standard_normal((1, width)) / math.sqrt(width),
        "b2": np.zeros(1),
    }


def _copy_params(params: _Params) -> _Params:
    return {name: array.copy() for name, array in params.items()}


def _make_teacher(gen: np.random.Generator) -> _Params:
    """A fixed random tanh teacher whose output defines one task's target function."""

    return {
        "W1": gen.standard_normal((H_TEACHER, M_IN)) / math.sqrt(M_IN),
        "b1": gen.standard_normal(H_TEACHER) * 0.5,
        "W2": gen.standard_normal((1, H_TEACHER)),
        "b2": np.zeros(1),
    }


def _teacher_output(teacher: _Params, x: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ teacher["W1"].T + teacher["b1"])
    return hidden @ teacher["W2"].T + teacher["b2"]


def _forward(params: _Params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["W1"].T + params["b1"])
    output = hidden @ params["W2"].T + params["b2"]
    return hidden, output


def _test_mse(params: _Params, x: np.ndarray, y: np.ndarray) -> float:
    _, output = _forward(params, x)
    return float(np.mean((output - y) ** 2))


def _train_task(params: _Params, x: np.ndarray, y: np.ndarray, update_hidden: bool) -> None:
    """Warm-started full-batch gradient descent for GD_STEPS steps, in place on params."""

    n = x.shape[0]
    for _ in range(GD_STEPS):
        hidden, output = _forward(params, x)
        d_out = (2.0 / n) * (output - y)
        d_w2 = d_out.T @ hidden
        d_b2 = d_out.sum(axis=0)
        if update_hidden:
            d_hidden = d_out @ params["W2"]
            d_pre = d_hidden * (1.0 - hidden**2)
            params["W1"] -= LR * (d_pre.T @ x)
            params["b1"] -= LR * d_pre.sum(axis=0)
        params["W2"] -= LR * d_w2
        params["b2"] -= LR * d_b2


def _reset_low_utility(params: _Params, x_ref: np.ndarray, gen: np.random.Generator) -> None:
    """Continual-backprop-style reset of the least useful hidden units (in place)."""

    hidden, _ = _forward(params, x_ref)
    utility = np.abs(params["W2"][0]) * np.mean(np.abs(hidden), axis=0)
    width = params["W1"].shape[0]
    n_reset = max(1, int(RESET_FRAC * width))
    victims = np.argsort(utility)[:n_reset]
    params["W1"][victims] = gen.standard_normal((n_reset, M_IN)) / math.sqrt(M_IN)
    params["b1"][victims] = 0.0
    params["W2"][0, victims] = 0.0  # zero the outgoing weight so the reset does not disturb the output


def _build_tasks(
    gen_teacher: np.random.Generator, gen_data: np.random.Generator
) -> list[dict[str, np.ndarray]]:
    """A per-stream sequence of tasks, shared byte-for-byte by all four arms for a paired comparison."""

    tasks: list[dict[str, np.ndarray]] = []
    for _ in range(N_TASKS):
        teacher = _make_teacher(gen_teacher)
        x_tr = gen_data.standard_normal((N_TRAIN, M_IN))
        x_te = gen_data.standard_normal((N_TEST, M_IN))
        y_tr_raw = _teacher_output(teacher, x_tr)
        y_te_raw = _teacher_output(teacher, x_te)
        mu = float(y_tr_raw.mean())
        sd = float(y_tr_raw.std()) + 1e-8
        tasks.append(
            {
                "x_tr": x_tr,
                "x_te": x_te,
                "y_tr": (y_tr_raw - mu) / sd,
                "y_te": (y_te_raw - mu) / sd,
            }
        )
    return tasks


def _post_drift(errors: list[float]) -> float:
    tail = errors[-POST_DRIFT_TASKS:]
    return float(np.mean(tail))


def _run_sgd(init: _Params, tasks: list[dict[str, np.ndarray]]) -> float:
    params = _copy_params(init)
    errors: list[float] = []
    for task in tasks:
        _train_task(params, task["x_tr"], task["y_tr"], update_hidden=True)
        errors.append(_test_mse(params, task["x_te"], task["y_te"]))
    return _post_drift(errors)


def _run_reset(init: _Params, tasks: list[dict[str, np.ndarray]], gen_reset: np.random.Generator) -> float:
    params = _copy_params(init)
    errors: list[float] = []
    for index, task in enumerate(tasks):
        if index > 0:
            _reset_low_utility(params, tasks[index - 1]["x_tr"], gen_reset)
        _train_task(params, task["x_tr"], task["y_tr"], update_hidden=True)
        errors.append(_test_mse(params, task["x_te"], task["y_te"]))
    return _post_drift(errors)


def _run_frozen_shell(gen_init: np.random.Generator, tasks: list[dict[str, np.ndarray]]) -> float:
    params = _init_learner(gen_init, H_SHELL)
    errors: list[float] = []
    for task in tasks:
        _train_task(params, task["x_tr"], task["y_tr"], update_hidden=False)
        errors.append(_test_mse(params, task["x_te"], task["y_te"]))
    return _post_drift(errors)


def _run_fresh_init(gen_init: np.random.Generator, tasks: list[dict[str, np.ndarray]]) -> float:
    errors: list[float] = []
    for task in tasks:
        params = _init_learner(gen_init, H)
        _train_task(params, task["x_tr"], task["y_tr"], update_hidden=True)
        errors.append(_test_mse(params, task["x_te"], task["y_te"]))
    return _post_drift(errors)


def _arm_flops(width: int, update_hidden: bool) -> LifecycleCost:
    """Rough full-lifecycle FLOP accounting per arm, so the larger frozen shell is not treated as free."""

    per_example_fwd = 2 * (M_IN * width + width)
    per_example_bwd = per_example_fwd * (2 if update_hidden else 1)
    train = GD_STEPS * N_TASKS * N_TRAIN * (per_example_fwd + per_example_bwd)
    infer = N_TASKS * N_TEST * per_example_fwd
    return LifecycleCost(train_flops=int(train), inference_flops=int(infer))


@register_runner("wave_f.plasticity_reset_vs_sgd")
def wave_f_plasticity_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Continual-reset vs plain SGD (and frozen-shell and fresh-init controls) on non-stationary streams."""

    per_unit: list[dict[str, Any]] = []
    deltas: list[float] = []
    sgd_errs: list[float] = []
    reset_errs: list[float] = []
    frozen_errs: list[float] = []
    fresh_errs: list[float] = []

    for stream in range(N_STREAMS):
        gen_teacher = rng(ctx.seed, "wave_f", "teacher", stream)
        gen_data = rng(ctx.seed, "wave_f", "data", stream)
        gen_init = rng(ctx.seed, "wave_f", "init", stream)
        gen_reset = rng(ctx.seed, "wave_f", "reset", stream)
        gen_frozen = rng(ctx.seed, "wave_f", "frozen", stream)
        gen_fresh = rng(ctx.seed, "wave_f", "fresh", stream)

        tasks = _build_tasks(gen_teacher, gen_data)
        shared_init = _init_learner(gen_init, H)  # SGD and reset share one initialization

        sgd_error = _finite(_run_sgd(shared_init, tasks))
        reset_error = _finite(_run_reset(shared_init, tasks, gen_reset))
        frozen_error = _finite(_run_frozen_shell(gen_frozen, tasks))
        fresh_error = _finite(_run_fresh_init(gen_fresh, tasks))

        delta = round(sgd_error - reset_error, 9)  # positive favors the continual-reset candidate
        deltas.append(delta)
        sgd_errs.append(sgd_error)
        reset_errs.append(reset_error)
        frozen_errs.append(frozen_error)
        fresh_errs.append(fresh_error)
        per_unit.append(
            {
                "stream": stream,
                "sgd_error": sgd_error,
                "candidate_error": reset_error,
                "frozen_shell_error": frozen_error,
                "fresh_init_error": fresh_error,
                "delta": delta,
            }
        )

    sign_flip = exact_sign_flip_one_sided(deltas)
    base_verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)

    mean_reset = float(np.mean(reset_errs))
    frozen_margin = round(float(np.mean(frozen_errs)) - mean_reset, 9)  # positive => reset beats frozen
    fresh_margin = round(float(np.mean(fresh_errs)) - mean_reset, 9)  # positive => reset beats fresh
    beats_frozen = frozen_margin > 0.0
    beats_fresh = fresh_margin > 0.0
    control_gate = beats_frozen and beats_fresh

    verdict = "survives" if (base_verdict == "survives" and control_gate) else "null"
    is_null = verdict != "survives"

    costs = {
        "sgd": _arm_flops(H, update_hidden=True),
        "candidate_reset": _arm_flops(H, update_hidden=True),
        "frozen_shell": _arm_flops(H_SHELL, update_hidden=False),
        "fresh_init": _arm_flops(H, update_hidden=True),
    }
    ceiling = max(cost.total for cost in costs.values()) * 2

    coverage = {
        "form_family": "memory_episode",
        "phenomenon": "plasticity",
        "mechanism_family": "continual_reset",
        "unit_class": "nonstationary_stream",
        "evidence_level": "M1",
    }
    content = honest_envelope(ctx.node_id, SCHEMA, coverage)
    content["candidate"] = _CANDIDATE_DESC
    content["controls"] = {
        "primary_sgd": _PRIMARY_SGD_DESC,
        "frozen_shell": _FROZEN_SHELL_DESC,
        "fresh_init": _FRESH_INIT_DESC,
    }
    content["score_definition"] = _SCORE_DEF
    content["delta_definition"] = _DELTA_DEF
    content["n_streams"] = N_STREAMS
    content["hyperparameters"] = {
        "input_dim": M_IN,
        "hidden_width": H,
        "shell_width": H_SHELL,
        "teacher_width": H_TEACHER,
        "n_tasks": N_TASKS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "gd_steps": GD_STEPS,
        "learning_rate": LR,
        "reset_fraction": RESET_FRAC,
        "post_drift_tasks": POST_DRIFT_TASKS,
    }
    content["per_unit"] = per_unit
    content["deltas"] = deltas
    content["sign_flip"] = sign_flip
    content["sesoi"] = SESOI
    content["arm_mean_post_drift_error"] = {
        "sgd": _finite(float(np.mean(sgd_errs))),
        "candidate_reset": _finite(mean_reset),
        "frozen_shell": _finite(float(np.mean(frozen_errs))),
        "fresh_init": _finite(float(np.mean(fresh_errs))),
    }
    content["control_margins"] = {
        "frozen_shell_mean_margin": frozen_margin,
        "fresh_init_mean_margin": fresh_margin,
        "beats_frozen_shell": beats_frozen,
        "beats_fresh_init": beats_fresh,
    }
    content["matched_budget"] = assert_matched_budget(costs, ceiling)
    content["verdict"] = verdict
    content["alternative_explanation"] = _ALT_EXPLANATION
    content["failure_domain"] = _FAILURE_DOMAIN

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
            "beats_frozen_shell": beats_frozen,
            "beats_fresh_init": beats_fresh,
        },
    )
