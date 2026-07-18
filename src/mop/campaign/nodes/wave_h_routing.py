"""Wave H science node: the value of competence-conditioned routing across a perspective ecology.

Question. Two readers (perspectives) have context-disjoint competence: reader 1 is accurate when a binary
context factor is 0 and near chance when it is 1, and reader 2 is the mirror image. The candidate strategy
is competence-conditioned routing: send each example to the reader that is competent in that example's
context. We ask whether this beats every pooled alternative that cannot condition on competence.

Named controls (the bar the candidate must clear).
  * best-single reader   : deploy whichever single reader scores best over the whole task.
  * homogeneous bank      : average several same-competence readers (the best of the two homogeneous banks).
  * uniform ensemble      : average the two heterogeneous readers with equal weight.
  * random routing        : send each example to a randomly chosen reader.

Design. Each experimental unit is an independent synthetic task with its own seed. Score is task accuracy.
The per-unit paired delta is candidate accuracy minus the best control accuracy on the SAME examples, so a
positive delta favors competence routing. Over N units we run the framework's exact one-sided sign-flip and
derive the neutral verdict against a small structural SESOI. Candidate inference is also the CHEAPEST arm
(one reader per example), so any win is not bought with extra compute; the matched-budget accounting records
this.

Precondition (negative control). A separate task removes the separating factor: both readers are
undifferentiated (identical, perfectly correlated, competent everywhere), so the context factor carries no
competence information. Here every strategy makes the same predictions and the delta is exactly zero. This
mirrors the density-mixture precondition: a mixture wins iff a required factor sharply separates the readers,
and ties when it does not.

Honesty. A tie or a wrong-direction result is a legitimate null; nothing here is tuned toward a positive.
One run is evidence level M1: consistent with, never a scientific confirmation.

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

# Competent-reader decision margin: Phi^{-1}(0.90), giving about 0.90 single-context accuracy under unit
# Gaussian noise. An incompetent reader has margin 0 (pure chance).
_COMPETENT_MARGIN = 1.2815515594457764
_N_UNITS = 9
_N_EXAMPLES = 600
_BANK_K = 5
# Structural SESOI in accuracy points: a gain smaller than two points is treated as no effect.
_SESOI = 0.02


def _accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Task accuracy: a positive decision score predicts label 1, and we compare to the true label."""

    preds = (scores > 0.0).astype(np.int64)
    return float(np.mean(preds == labels))


def _evaluate_task(
    labels: np.ndarray,
    factor: np.ndarray,
    score1: np.ndarray,
    score2: np.ndarray,
    bank1: np.ndarray,
    bank2: np.ndarray,
    random_route: np.ndarray,
) -> dict[str, float]:
    """Score the candidate and every named control on one task, and return the paired delta.

    ``bank1``/``bank2`` are ``K by M`` arrays of same-competence readers; their column means are the
    homogeneous-bank decision scores. ``random_route`` is a per-example bit selecting a reader at random.
    """

    candidate_scores = np.where(factor == 0, score1, score2)
    acc_candidate = _accuracy(candidate_scores, labels)

    acc_best_single = max(_accuracy(score1, labels), _accuracy(score2, labels))
    acc_homogeneous = max(
        _accuracy(bank1.mean(axis=0), labels),
        _accuracy(bank2.mean(axis=0), labels),
    )
    acc_uniform_ensemble = _accuracy(score1 + score2, labels)
    acc_random_routing = _accuracy(np.where(random_route == 0, score1, score2), labels)

    best_control = max(acc_best_single, acc_homogeneous, acc_uniform_ensemble, acc_random_routing)
    return {
        "acc_candidate": round(acc_candidate, 12),
        "acc_best_single": round(acc_best_single, 12),
        "acc_homogeneous_bank": round(acc_homogeneous, 12),
        "acc_uniform_ensemble": round(acc_uniform_ensemble, 12),
        "acc_random_routing": round(acc_random_routing, 12),
        "best_control": round(best_control, 12),
        "delta": round(acc_candidate - best_control, 12),
    }


def _separating_unit(gen: np.random.Generator, m: int, k: int) -> dict[str, float]:
    """A task where the context factor sharply separates reader competence (independent reader noise)."""

    labels = gen.integers(0, 2, m)
    signed = 2 * labels - 1
    factor = gen.integers(0, 2, m)
    competent1 = (factor == 0).astype(np.float64)  # reader 1 competent where the factor bit is 0
    competent2 = (factor == 1).astype(np.float64)  # reader 2 competent where the factor bit is 1

    score1 = signed * _COMPETENT_MARGIN * competent1 + gen.standard_normal(m)
    score2 = signed * _COMPETENT_MARGIN * competent2 + gen.standard_normal(m)
    bank1 = signed * _COMPETENT_MARGIN * competent1 + gen.standard_normal((k, m))
    bank2 = signed * _COMPETENT_MARGIN * competent2 + gen.standard_normal((k, m))
    random_route = gen.integers(0, 2, m)
    return _evaluate_task(labels, factor, score1, score2, bank1, bank2, random_route)


def _negative_control(gen: np.random.Generator, m: int, k: int) -> dict[str, float]:
    """The factor does NOT separate the readers: both readers are identical and competent everywhere.

    With perfectly correlated, undifferentiated readers every strategy makes the same predictions, so the
    delta is exactly zero. This is the required precondition check, not a tuned outcome.
    """

    labels = gen.integers(0, 2, m)
    signed = 2 * labels - 1
    factor = gen.integers(0, 2, m)
    shared = signed * _COMPETENT_MARGIN + gen.standard_normal(m)  # one signal shared by both readers
    bank = np.broadcast_to(shared, (k, m))  # a bank of identical copies adds no information
    random_route = gen.integers(0, 2, m)
    return _evaluate_task(labels, factor, shared, shared, bank, bank, random_route)


@register_runner("wave_h.competence_conditioned_routing")
def wave_h_routing_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Real, deterministic Wave H mechanics: does competence-conditioned routing beat pooled controls?"""

    n_units = int(params.get("n_units", _N_UNITS))
    m_examples = int(params.get("n_examples", _N_EXAMPLES))
    bank_k = int(params.get("bank_k", _BANK_K))

    units: list[dict[str, Any]] = []
    for u in range(n_units):
        gen = rng(ctx.seed, "wave_h_sep", u)
        scored = _separating_unit(gen, m_examples, bank_k)
        units.append({"unit_id": f"sep-{u:02d}", **scored})

    deltas = [unit["delta"] for unit in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], _SESOI)
    is_null = verdict != "survives"

    neg_gen = rng(ctx.seed, "wave_h_negative_control")
    negative = _negative_control(neg_gen, m_examples, bank_k)
    negative_control_ties = abs(negative["delta"]) < _SESOI

    # Matched-budget accounting: one reader-evaluation over M examples is the unit inference cost. The
    # candidate deploys a single reader per example, so it is the cheapest non-degenerate arm; any win is
    # not bought with extra compute.
    costs = {
        "candidate_routing": LifecycleCost(train_flops=0, inference_flops=m_examples),
        "best_single": LifecycleCost(train_flops=0, inference_flops=m_examples),
        "uniform_ensemble": LifecycleCost(train_flops=0, inference_flops=2 * m_examples),
        "homogeneous_bank": LifecycleCost(train_flops=0, inference_flops=bank_k * m_examples),
        "random_routing": LifecycleCost(train_flops=0, inference_flops=m_examples),
    }
    budget = assert_matched_budget(costs, ceiling=bank_k * m_examples)

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_h_routing/v1",
        {
            "form_family": "perspective",
            "phenomenon": "routing_value_of_computation",
            "mechanism_family": "competence_conditioned_routing",
            "unit_class": "synthetic_two_reader_task",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "competence-conditioned routing: send each example to the reader competent in its "
            "context (factor bit 0 -> reader 1, factor bit 1 -> reader 2)",
            "named_controls": {
                "best_single": "deploy the single reader with the best whole-task accuracy",
                "homogeneous_bank": f"average {bank_k} same-competence readers (best of the two banks)",
                "uniform_ensemble": "equal-weight average of the two heterogeneous readers",
                "random_routing": "send each example to a randomly chosen reader",
            },
            "control_description": "per-unit delta is candidate accuracy minus the best of "
            "{best_single, homogeneous_bank, uniform_ensemble, random_routing} on the same examples; "
            "positive favors competence routing",
            "config": {
                "n_units": n_units,
                "n_examples": m_examples,
                "bank_k": bank_k,
                "competent_margin": round(_COMPETENT_MARGIN, 12),
                "sesoi": _SESOI,
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": _SESOI,
            "verdict": verdict,
            "is_null": is_null,
            "negative_control": {
                **negative,
                "description": "factor does not separate the readers (identical, correlated, competent "
                "everywhere); every strategy agrees so the delta is exactly zero",
                "ties": negative_control_ties,
            },
            "matched_budget": budget,
            "alternative_explanation": "The gain reflects the router reading the same context factor the "
            "pooled controls also receive but cannot exploit; it is bounded to the regime where that factor "
            "sharply separates reader competence and collapses to a tie when readers are undifferentiated.",
            "failure_domain": "When the context factor does not separate reader competence (undifferentiated "
            "or perfectly correlated readers), competence-conditioned routing does not beat a uniform "
            "ensemble; the negative control ties at delta zero.",
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
