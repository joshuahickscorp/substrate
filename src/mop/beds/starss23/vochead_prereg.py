"""STARSS23 value-of-computation HEADROOM instrument, component 2: the preregistered analysis plan.

This freezes, before any corpus number is read, exactly what the instrument measures and how each per-target
result is classified, into a self-sealed ``proof/STARSS23_VOC_HEADROOM.prereg.json``. It exists so the
interpretation labels (``what_floor_collapse`` / ``real_headroom`` / ``no_headroom_budget_saturated``)
cannot be tuned to a desired conclusion after the fact: the budget sweep, the rate-matched-random draw
discipline, the two derived quantities, and the classification thresholds are all fixed here.

This is an INSTRUMENT, not a promotable bed. It reports descriptive corpus characterization
(value-of-computation headroom), never a capability or activation claim. ``activation_allowed`` and
``scientific_promotion`` are hardcoded false, ``independent_scientific_confirmation`` is never set true by
either the producer or the verifier, and a tie is a null. Nothing here reads a corpus score.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE
from .stats import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from .vochead_analyzer import (
    BUDGET_FRACTIONS,
    INTERP_NO_HEADROOM_BUDGET_SATURATED,
    INTERP_REAL_HEADROOM,
    INTERP_WHAT_FLOOR_COLLAPSE,
    METRIC_COUNT_ABS,
    METRIC_DOA_GREATCIRCLE,
    N_RMR_DRAWS,
    RMR_BASE_SEED,
    TIE_EPS,
    VOCHEAD_ANALYZER_SCHEMA,
)

VOCHEAD_PREREG_SCHEMA = "mop-starss23-vochead-prereg/v1"
STAGE = 3
VOCHEAD_INSTRUMENT_ID = "starss23_voc_headroom"


def build_vochead_prereg(*, timestamp: str) -> dict[str, Any]:
    """Assemble the self-sealed instrument preregistration. The timestamp is passed, not read from a clock."""

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("timestamp must be a non-empty string passed by the caller")

    body: dict[str, Any] = {
        "schema": VOCHEAD_PREREG_SCHEMA,
        "stage": STAGE,
        "instrument_id": VOCHEAD_INSTRUMENT_ID,
        "analyzer_schema": VOCHEAD_ANALYZER_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_corpus_scores": True,
        "purpose": (
            "measure how much value-of-computation HEADROOM each STARSS23 re-estimation target contains, "
            "before any trained gate, to decompose why the three sealed value-of-computation beds "
            "(onset-localization, source-counting, direction-of-arrival) nulled: a WHAT-floor collapse "
            "(re-estimating is worse than a constant) is a distinct failure shape from a corpus that "
            "genuinely rewards WHEN placement but whose gate could not learn it"
        ),
        "targets": {
            "count": {
                "metric_id": METRIC_COUNT_ABS,
                "description": "coasted concurrent-source-count absolute error per frame; strong frozen "
                "eigenvalue-rank estimator (the target that produced the program's only real VoC signal)",
            },
            "doa": {
                "metric_id": METRIC_DOA_GREATCIRCLE,
                "description": "coasted great-circle direction-of-arrival error in degrees per active "
                "frame; frozen wideband active-intensity estimator (the sealed DoA bed's WHAT)",
            },
        },
        "policies": {
            "always_on": "re-estimate every frame; the perfect-WHEN unlimited-budget ceiling (coasted error "
            "equals the frozen estimator's fresh error)",
            "never_update": "never re-estimate; coast the fixed cold-start forever; the zero-budget floor",
            "informed_change_aligned": "label-aware reference: starting from never_update, greedily add the "
            "change frame whose re-estimation most reduces total coasted error, up to budget K; a strong "
            "achievable WHEN policy over the change-frame candidates, an upper reference, NOT a proven "
            "global optimum",
            "rate_matched_random": "place the same K re-estimations at random, averaged over the "
            "preregistered deterministic draws; the exact control the sealed beds use",
        },
        "budget_sweep": {
            "fractions_of_clip_frames": list(BUDGET_FRACTIONS),
            "k_rule": "k = round(fraction * n_frames), clamped to [1, n_frames]",
            "rationale": "spans the tight regime (budget below change density, where placement is "
            "decisive) through the loose regime the sealed beds operated in",
        },
        "rate_matched_random_discipline": {
            "base_seed": RMR_BASE_SEED,
            "n_draws": N_RMR_DRAWS,
            "seed_rule": "random.Random(int.from_bytes(sha256(f'{base}|{clip}|{k}|{draw}')[:8])); "
            "host-reproducible stdlib sampling so the independent verifier re-derives it exactly",
        },
        "derived_quantities": {
            "refreshable_range": "never_update_macro minus always_on_macro (clip-macro); the full error "
            "span that re-estimating can buy; negative means re-estimating is worse than a constant",
            "headroom": "rate_matched_random_macro minus informed_change_aligned_macro at each budget; "
            "positive means the label-aware policy beats random at that matched budget",
            "realization_fraction": "(never_update_macro minus informed_change_aligned_macro) divided by "
            "the refreshable_range; only defined when the range is positive",
        },
        "interpretation_rule": {
            "tie_eps": TIE_EPS,
            INTERP_WHAT_FLOOR_COLLAPSE: "refreshable_range <= tie_eps: re-estimating the frozen estimator "
            "is no better than coasting a constant, so no WHEN policy can help (a WHAT-floor failure)",
            INTERP_REAL_HEADROOM: "refreshable_range > tie_eps AND the informed change-aligned reference "
            "strictly beats rate-matched-random at EVERY swept budget: the corpus rewards WHEN placement "
            "and a gate that could locate changes would win",
            INTERP_NO_HEADROOM_BUDGET_SATURATED: "refreshable_range > tie_eps but the informed reference "
            "ties random at some budget: random already saturates the rare changes there",
            "tie_is_null": True,
        },
        "scopes": {
            "test_fold": "the native fold-4 dev-test split the sealed beds evaluate on (directly "
            "comparable to their results)",
            "full_subset": "all fixed-subset clips (train+val+test), the widest descriptive characterization",
            "synthetic_control": "self-contained toy targets with a known-strong and a known-harmful WHAT, "
            "proving the instrument reports real_headroom and what_floor_collapse respectively and is not "
            "rigged to a single label",
        },
        "claim_ceiling": {
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "rationale": "a descriptive corpus-characterization instrument; the informed policy is a "
            "label-aware upper reference, not a demonstration of any gate, and no positive is claimed",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


DEFAULT_VOCHEAD_PREREG_PATH = Path("proof/STARSS23_VOC_HEADROOM.prereg.json")


def write_vochead_prereg(body: dict[str, Any], out_path: str | Path = DEFAULT_VOCHEAD_PREREG_PATH) -> Path:
    """Write the self-sealed preregistration as canonical JSON bytes so its on-disk digest is stable."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path
