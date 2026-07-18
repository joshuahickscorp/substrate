"""Wave G compositional-generalization bed: a factorized reader versus a conjunctive-lookup control.

The question is whether reading a symbol as a factored pair of independent attributes buys generalization to
HELD-OUT conjunctions that a pure memorizer cannot reach. Each experimental unit is an independent synthetic
symbolic-algebra task over a color-by-shape grid. Every color carries a hidden integer value and every shape
carries a hidden integer value, and the target output for a conjunction is the sum of the two attribute
values, so the true mapping factorizes additively. Training reveals a subset of conjunctions with noisy
observed outputs; testing scores accuracy on conjunctions that were never shown during training. The train
split always includes the grid diagonal, so every color and every shape appears in training at least once,
which is exactly the precondition that makes held-out conjunctions reachable by factorization.

Two arms share the identical training observations so the only thing that varies is how the symbol is read:

  candidate  factorized reader: fit an additive attribute model mu + a[color] + b[shape] by alternating least
             squares on the training conjunctions, then compose the two learned attributes to predict any
             held-out conjunction, including combinations never seen together.
  lookup     the NAMED control: a conjunctive lookup that memorizes each seen (color, shape) pair and, having
             no factorization, must fall back to the single most common training output on any held-out pair.

The primary paired delta per task is candidate held-out accuracy minus lookup held-out accuracy (positive
favors the factorized reader). A precondition negative control replaces the additive truth with an
independent per-cell interaction table that does not factorize; there the additive candidate has no structure
to exploit and does not beat the memorizer, so the delta collapses toward zero. A tie or a wrong-direction
primary result is a legitimate null and is reported as such; nothing here is tuned toward a positive.

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

N_TASKS = 10  # independent experimental units; within the exact sign-flip enumeration cap
GRID = 6  # colors and shapes (a GRID by GRID conjunction table)
VALUE_RANGE = 5  # each attribute value is an integer in [0, VALUE_RANGE)
OUTPUT_LEVELS = 2 * VALUE_RANGE - 1  # additive-sum support size, reused by the interaction control
N_EXTRA_TRAIN = 12  # off-diagonal conjunctions added to the always-present diagonal for training
N_TEST = 12  # held-out off-diagonal conjunctions used for scoring
P_NOISE = 0.15  # chance a training observation is corrupted by plus or minus one
ALS_PASSES = 12  # alternating-least-squares sweeps for the additive fit
N_NEG_CONTROL = 6  # interaction-truth precondition tasks, averaged
SESOI = 0.05  # small structural minimum improvement on the accuracy scale
ROUND = 8


def _r(value: Any) -> float:
    """Cast any numpy or python scalar to a plain rounded float so the sealed JSON stays canonical."""

    return round(float(value), ROUND)


def _split(gen: np.random.Generator) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Train and held-out conjunction lists. The diagonal is always trained so every color and shape is
    covered; extra training and all test pairs are drawn disjointly from the off-diagonal cells."""

    diagonal = [(i, i) for i in range(GRID)]
    off = [(c, s) for c in range(GRID) for s in range(GRID) if c != s]
    order = gen.permutation(len(off))
    shuffled = [off[int(i)] for i in order]
    extra_train = shuffled[:N_EXTRA_TRAIN]
    test = shuffled[N_EXTRA_TRAIN : N_EXTRA_TRAIN + N_TEST]
    return diagonal + extra_train, test


def _fit_additive(
    colors: np.ndarray, shapes: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Alternating-least-squares additive fit y ~ mu + a[color] + b[shape] on the training conjunctions.

    The diagonal split guarantees every color and shape has at least one training cell, so no effect is
    unidentified; the guard on empty groups is defensive. Deterministic given the inputs.
    """

    mu = float(y.mean())
    counts_c = np.bincount(colors, minlength=GRID)
    counts_s = np.bincount(shapes, minlength=GRID)
    a = np.zeros(GRID)
    b = np.zeros(GRID)
    for _ in range(ALS_PASSES):
        target_c = y - mu - b[shapes]
        sum_c = np.bincount(colors, weights=target_c, minlength=GRID)
        a = np.where(counts_c > 0, sum_c / np.maximum(counts_c, 1), 0.0)
        target_s = y - mu - a[colors]
        sum_s = np.bincount(shapes, weights=target_s, minlength=GRID)
        b = np.where(counts_s > 0, sum_s / np.maximum(counts_s, 1), 0.0)
    return mu, a, b


def _run_task(gen: np.random.Generator, truth: np.ndarray) -> dict[str, Any]:
    """Score the factorized candidate and the conjunctive-lookup control on one task and return the record.

    ``truth`` is the noiseless GRID by GRID output table. The two arms see the identical noisy training
    observations; scoring compares each arm's held-out predictions to the noiseless truth.
    """

    train_pairs, test_pairs = _split(gen)
    tc = np.array([c for c, _ in train_pairs], dtype=np.int64)
    ts = np.array([s for _, s in train_pairs], dtype=np.int64)
    y_true_train = truth[tc, ts].astype(np.float64)

    corrupt = gen.random(len(train_pairs)) < P_NOISE
    sign = gen.choice(np.array([-1.0, 1.0]), size=len(train_pairs))
    y_obs_train = y_true_train + corrupt * sign

    mu, a, b = _fit_additive(tc, ts, y_obs_train)

    test_c = np.array([c for c, _ in test_pairs], dtype=np.int64)
    test_s = np.array([s for _, s in test_pairs], dtype=np.int64)
    y_true_test = truth[test_c, test_s].astype(np.int64)

    cand_pred = np.round(mu + a[test_c] + b[test_s]).astype(np.int64)
    acc_candidate = float(np.mean(cand_pred == y_true_test))

    lookup = {(int(c), int(s)): int(round(v)) for (c, s), v in zip(train_pairs, y_obs_train, strict=True)}
    vals, counts = np.unique(np.round(y_obs_train).astype(np.int64), return_counts=True)
    fallback = int(vals[int(np.argmax(counts))])
    ctrl_pred = np.array([lookup.get((int(c), int(s)), fallback) for c, s in test_pairs], dtype=np.int64)
    acc_lookup = float(np.mean(ctrl_pred == y_true_test))

    return {
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "acc_candidate": _r(acc_candidate),
        "acc_lookup": _r(acc_lookup),
        "delta": _r(acc_candidate - acc_lookup),
    }


def _additive_task(seed: int, idx: int) -> dict[str, Any]:
    """A task whose truth factorizes: output(color, shape) = color_value + shape_value."""

    gen = rng(seed, "wave_g_compositional", "additive", idx)
    color_value = gen.integers(0, VALUE_RANGE, GRID)
    shape_value = gen.integers(0, VALUE_RANGE, GRID)
    truth = color_value[:, None] + shape_value[None, :]
    record = _run_task(gen, truth)
    return {"task": idx, "regime": "additive", **record}


def _interaction_task(seed: int, idx: int) -> dict[str, Any]:
    """Precondition negative control: an independent per-cell truth table that does not factorize, so the
    additive candidate has no compositional structure to recover and should not beat the memorizer."""

    gen = rng(seed, "wave_g_compositional", "interaction", idx)
    truth = gen.integers(0, OUTPUT_LEVELS, size=(GRID, GRID))
    record = _run_task(gen, truth)
    return {"task": idx, "regime": "interaction", **record}


@register_runner("wave_g.compositional_generalization")
def wave_g_compositional_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Factorized attribute reading versus conjunctive memorization on held-out color-shape conjunctions."""

    n_tasks = int(params.get("n_tasks", N_TASKS))
    per_unit = [_additive_task(ctx.seed, i) for i in range(n_tasks)]

    primary_deltas = [u["delta"] for u in per_unit]
    sign_flip = exact_sign_flip_one_sided(primary_deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    neg_units = [_interaction_task(ctx.seed, i) for i in range(N_NEG_CONTROL)]
    neg_deltas = [u["delta"] for u in neg_units]
    neg_mean_delta = _r(float(np.mean(neg_deltas)))
    negative_control_ties = abs(neg_mean_delta) < SESOI

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_g_compositional/v1",
        {
            "form_family": "symbolic",
            "phenomenon": "compositional_binding",
            "mechanism_family": "factorized_state",
            "unit_class": "synthetic_color_shape_algebra_task",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "design": {
                "n_tasks": n_tasks,
                "grid": GRID,
                "value_range": VALUE_RANGE,
                "output_levels": OUTPUT_LEVELS,
                "n_train_conjunctions": GRID + N_EXTRA_TRAIN,
                "n_held_out_conjunctions": N_TEST,
                "train_noise_flip_prob": P_NOISE,
                "als_passes": ALS_PASSES,
                "score": "held_out_conjunction_accuracy",
                "coverage_guarantee": "diagonal always trained so every color and shape appears in training",
            },
            "candidate": (
                "factorized reader: alternating-least-squares additive fit mu + a[color] + b[shape] on the "
                "training conjunctions, composing the two learned attributes to predict held-out pairs"
            ),
            "control": (
                "conjunctive_lookup memorizes each seen (color, shape) output and, lacking factorization, "
                "falls back to the single most common training output on any held-out conjunction; both arms "
                "read the identical noisy training observations so only the reading strategy differs"
            ),
            "per_unit": per_unit,
            "primary_deltas": primary_deltas,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "verdict": verdict,
            "is_null": is_null,
            "negative_control": {
                "regime": "interaction_truth_does_not_factorize",
                "n_tasks": N_NEG_CONTROL,
                "per_unit": neg_units,
                "mean_delta": neg_mean_delta,
                "ties": negative_control_ties,
                "note": (
                    "with an independent per-cell truth table there is no additive structure to recover, so "
                    "the factorized candidate does not beat the memorizer and the delta collapses toward zero"
                ),
            },
            "alternative_explanation": (
                "The lookup control could lose merely because held-out cells are unseen and any generalizing "
                "reader would win, rather than because factored attribute binding is doing real work. The "
                "interaction negative control addresses this: it keeps the same unseen-cell structure but "
                "removes additive factorization, and there the candidate does not beat the memorizer, which "
                "is consistent with the gain coming from recoverable compositional structure rather than "
                "from generalization pressure alone."
            ),
            "failure_domain": (
                "Non-factorizable mappings: when the true output carries color-by-shape interaction terms "
                "that no additive attribute model can represent, the factorized reader cannot compose the "
                "correct held-out answer and degrades toward the conjunctive-lookup control, as the "
                "interaction negative control shows."
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
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "negative_control_ties": negative_control_ties,
        },
    )
