"""Wave A science node: ambiguous-referent abstention under a genuine two-way tie.

This bed asks a single uncertainty question: when some events are genuinely ambiguous (two referents
equidistant, with the ground-truth label drawn independently of the feature so no rule can beat chance on
them), does a binder that ABSTAINS on low-margin assignments score better than the identical binder forced
to always guess?

We build N independent synthetic streams (the experimental units). Each stream mixes clear events (drawn
tight around one referent centroid, unambiguous) with ambiguous events (placed at the midpoint between a
random pair of referents, ground truth a coin flip between the two). The binder computes each event's
assignment margin (distance to the second-nearest referent minus distance to the nearest) and its argmin
assignment. Clear events carry a large margin; ambiguous events carry a near-zero margin.

The candidate abstains when the margin falls below a fixed threshold and answers otherwise. Its score is
accuracy-on-answered minus a small penalty for the fraction abstained. The named control (force-assign)
is the same binder with no abstention: it answers every event, so its score is plain accuracy over all
events and it must guess on the ambiguous ones.

Per-unit paired delta is candidate_score minus force_assign_score (positive favors selective abstention).
The verdict comes from the shared exact sign-flip and a small structural SESOI. The candidate wins only if
declining truly unanswerable events is worth more than the abstention penalty; if the penalty is set at or
above the forced-guess error, or the ambiguous events are not really equidistant, this is a legitimate null.

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

# Structural geometry, fixed so the seal depends only on ctx.seed.
_N_STREAMS = 10
_K = 4
_D = 4
_S = 10.0
_SIGMA_CLEAR = 1.0
_SIGMA_AMB = 1.0
# Per-stream event counts are drawn from these ranges to give the units genuine heterogeneity.
_N_CLEAR_LOW, _N_CLEAR_HIGH = 22, 30
_N_AMB_LOW, _N_AMB_HIGH = 8, 16
# Abstention decision and scoring. The margin threshold sits well between the clear and ambiguous regimes.
_MARGIN_THRESHOLD = 5.0
_ABSTAIN_PENALTY = 0.10
# Small structural SESOI on the score scale (five score points).
_SESOI = 0.05


def _centroids() -> np.ndarray:
    """K well-separated referent centroids: scaled basis vectors, pairwise distance S * sqrt(2)."""

    return _S * np.eye(_K, _D, dtype=np.float64)


def _generate_stream(gen: np.random.Generator, cents: np.ndarray) -> dict[str, Any]:
    """Build one stream of clear and ambiguous events with ground-truth referent labels.

    Clear events sit tight around one referent (unambiguous). Ambiguous events sit at the midpoint of a
    random referent pair with the label a coin flip between the two, so no assignment rule can beat chance.
    """

    n_clear = int(gen.integers(_N_CLEAR_LOW, _N_CLEAR_HIGH + 1))
    n_amb = int(gen.integers(_N_AMB_LOW, _N_AMB_HIGH + 1))

    feats: list[np.ndarray] = []
    labels: list[int] = []
    ambiguous: list[bool] = []

    for _ in range(n_clear):
        k = int(gen.integers(_K))
        feats.append(cents[k] + _SIGMA_CLEAR * gen.standard_normal(_D))
        labels.append(k)
        ambiguous.append(False)

    for _ in range(n_amb):
        pair = gen.choice(_K, size=2, replace=False)
        k, j = int(pair[0]), int(pair[1])
        midpoint = 0.5 * (cents[k] + cents[j])
        feats.append(midpoint + _SIGMA_AMB * gen.standard_normal(_D))
        # Ground truth is independent of the feature: a coin flip between the two tied referents.
        labels.append(k if int(gen.integers(2)) == 0 else j)
        ambiguous.append(True)

    return {
        "features": np.asarray(feats, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "ambiguous": np.asarray(ambiguous, dtype=bool),
        "n_clear": n_clear,
        "n_amb": n_amb,
    }


def _assign(features: np.ndarray, cents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid assignment and the assignment margin (second-nearest minus nearest distance)."""

    dists = np.linalg.norm(features[:, None, :] - cents[None, :, :], axis=2)
    assigned = np.argmin(dists, axis=1)
    ordered = np.sort(dists, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    return assigned, margin


def _score_stream(stream: dict[str, Any], cents: np.ndarray) -> dict[str, Any]:
    """Score the abstaining candidate against the force-assign control on one stream."""

    features = stream["features"]
    labels = stream["labels"]
    assigned, margin = _assign(features, cents)
    correct = assigned == labels
    n = int(labels.shape[0])

    # Control (force-assign): answers every event, so its score is plain accuracy over all events.
    control_score = float(np.mean(correct))

    # Candidate: answer only above-margin events, then subtract a small penalty for abstaining.
    answered = margin >= _MARGIN_THRESHOLD
    n_answered = int(np.count_nonzero(answered))
    abstain_fraction = float(1.0 - n_answered / n)
    if n_answered > 0:
        acc_answered = float(np.mean(correct[answered]))
    else:
        acc_answered = 0.0
    candidate_score = acc_answered - _ABSTAIN_PENALTY * abstain_fraction

    delta = candidate_score - control_score
    return {
        "n_events": n,
        "n_clear": int(stream["n_clear"]),
        "n_amb": int(stream["n_amb"]),
        "control_accuracy": round(control_score, 6),
        "candidate_accuracy_answered": round(acc_answered, 6),
        "abstain_fraction": round(abstain_fraction, 6),
        "n_answered": n_answered,
        "candidate_score": round(candidate_score, 6),
        "control_score": round(control_score, 6),
        "delta": round(delta, 6),
    }


@register_runner("wave_a.ambiguity_abstention")
def wave_a_abstention_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Ambiguous-referent abstention bed: an abstaining binder vs a force-assign control."""

    cents = _centroids()
    per_unit: list[dict[str, Any]] = []
    for s in range(_N_STREAMS):
        stream = _generate_stream(rng(ctx.seed, "stream", s), cents)
        record = _score_stream(stream, cents)
        record["stream"] = int(s)
        per_unit.append(record)

    deltas = [u["delta"] for u in per_unit]
    sign_flip = exact_sign_flip_one_sided(deltas)
    mean_delta = float(sign_flip["mean_delta"])
    one_sided_p = float(sign_flip["one_sided_p"])
    verdict = verdict_from(mean_delta, one_sided_p, _SESOI)
    is_null = verdict != "survives"

    coverage = {
        "form_family": "events",
        "phenomenon": "uncertainty_abstention",
        "mechanism_family": "uncertainty_state",
        "unit_class": "synthetic_referent_stream",
        "evidence_level": "M1",
    }
    content = honest_envelope(ctx.node_id, "mop-campaign-wave_a_abstention/v1", coverage)
    content.update(
        {
            "n_units": int(_N_STREAMS),
            "sesoi": _SESOI,
            "margin_threshold": _MARGIN_THRESHOLD,
            "abstain_penalty": _ABSTAIN_PENALTY,
            "control_description": (
                "force-assign (never abstain): the identical nearest-centroid binder with abstention "
                "disabled, so it assigns every event to its argmin referent and must guess on the "
                "genuinely ambiguous events, where the ground-truth label is a coin flip between two "
                "equidistant referents drawn independently of the feature (chance on that subset)."
            ),
            "per_unit": per_unit,
            "deltas": [round(float(d), 6) for d in deltas],
            "sign_flip": sign_flip,
            "mean_delta": round(mean_delta, 6),
            "one_sided_p": round(one_sided_p, 12),
            "verdict": verdict,
            "is_null": bool(is_null),
            "alternative_explanation": (
                "a bookkeeping artifact of the abstention penalty being set below the forced-guess error "
                "rather than any real uncertainty signal. The ambiguous events are equidistant between two "
                "referents with the label drawn independently of the feature, so no assignment rule can "
                "exceed chance on them; both arms handle the clear events identically, so the delta "
                "isolates the value of declining the truly unanswerable subset. If the penalty is raised to "
                "the forced-guess error the delta collapses, which is exactly the honest boundary."
            ),
            "failure_domain": (
                "an abstention penalty at or above the forced-guess error (half on a two-way tie) makes "
                "declining not worth it; ambiguous events that are not truly equidistant leak the label so "
                "a forced guesser beats chance and erases the gap; a margin threshold below the ambiguous "
                "margin spread lets the candidate answer ambiguous events and inherit the guessing error; "
                "too few ambiguous events shrink the advantage below the SESOI."
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
            "mean_delta": round(mean_delta, 6),
            "one_sided_p": round(one_sided_p, 12),
            "n_units": int(_N_STREAMS),
            "n_units_favorable": int(sign_flip["n_units_favorable"]),
        },
    )
