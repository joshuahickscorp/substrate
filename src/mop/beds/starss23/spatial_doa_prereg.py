"""Sealed preregistration for the STARSS23 ESCS FROZEN-FEATURIZER iteration wave.

This is a net-new, additive component. It changes no sealed scoring logic. The committed real Stage-3 run
and every gate-policy variant so far nulled: the trained value-of-computation gate, reading half-wave
rectified log-mel spectral flux, clusters its fires on high-energy regions and recovers fewer distinct
onsets than uniform-random placement at matched budget. Those variants all kept the frozen log-mel flux
FRONT-END and changed only the gate. This wave asks a different question: does swapping the frozen
zero-trained-parameter FEATURIZER for one that exposes a spatial cue the log-mel front-end discards let
the SAME gate place a fixed firing budget better than rate-matched-random?

Testing a family of frozen featurizers against one fixed test split inflates the family-wise error, so
this module preregisters, IN CODE and sealed to ``proof/STARSS23_ESCS_BED_spatial_doa.prereg.json`` BEFORE
any featurizer reads a test score:

- the metric and direction (onset F1 at the DCASE plus-or-minus 200 ms collar, greedy one-to-one, strict
  point-wise PR; direction candidate greater than rate-matched-random), restated from the committed prereg;
- the SESOI (0.05), on the identical cost-benefit basis as the committed prereg, from label-only corpus
  structure (never from a test score);
- the exact one-sided sign-flip plan (five paired seeds, min one-sided p 1/32, two-sided 0.05 unreachable
  at n equals 5);
- a multiplicity control across the THREE featurizer families (Bonferroni). At three families the
  per-family adjusted alpha is 0.05/3 = 0.016667, and the smallest achievable one-sided sign-flip p at
  five seeds is 1/32 = 0.03125, which exceeds it, so no single featurizer can clear family-wise
  significance from this family alone: a preregistered statistical wall, not a moved goalpost;
- the claim ceiling (the clip is the experimental unit, the verb is bounded to "consistent with");
- the three featurizer ids, each with a one-line falsifiable hypothesis.

Nothing here reads a test-split F1. The cost-benefit derivation uses only fixed compute anchors and label
structure known before scoring (test-clip count, pooled test onset count, train-set onset density). The
timestamp is passed by the caller, never read from the wall clock inside the sealed body, and the sealed
body carries ``activation_allowed=false`` and ``scientific_promotion=false``.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID, CLAIM_SCOPE
from .prereg import (
    DEFAULT_C_DOWN_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    PREREG_DIRECTION,
    PREREG_METRIC,
    PREREGISTERED_SESOI_F1,
    _sesoi_rationale,
    compute_cost_benefit,
)
from .schema import COLLAR_FRAMES, FRAME_MS

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_spatial_doa.prereg.json")

# The three frozen zero-trained-parameter featurizer families. Each is a distinct, falsifiable answer to
# the same diagnosed failure (the log-mel front-end discards the spatial channel, so the gate cannot tell
# two equally loud events apart by their direction). Each hypothesis is one line, scored against the same
# sealed referee and rate-matched-random control. Only ``spatial_doa`` is executed in this wave; the other
# two are preregistered members of the multiplicity family so the family-wise wall is honest.
FEATURIZERS: tuple[dict[str, str], ...] = (
    {
        "featurizer_id": "spatial_doa",
        "hypothesis": (
            "Replacing the log-mel flux front-end with per-band active-intensity direction of arrival "
            "(the source azimuth and elevation as wraparound-free direction cosines) plus a DirAC "
            "diffuseness exposes the direction a new event arrives from, which the gate can use to place "
            "the fixed firing budget on distinct directional onsets and beat rate-matched-random."
        ),
    },
    {
        "featurizer_id": "spatial_coherence",
        "hypothesis": (
            "A frozen per-band interchannel magnitude-squared coherence between the omni and the three "
            "dipole channels flags coherent single-source arrivals against reverberant or diffuse frames, "
            "so the gate withholds fires in incoherent frames and spends the budget on coherent onsets, "
            "beating uniform-random placement at matched budget."
        ),
    },
    {
        "featurizer_id": "doa_flux",
        "hypothesis": (
            "A frozen half-wave-rectified temporal flux of the per-band direction of arrival (the rate of "
            "directional change) fires the explicit new-direction novelty of an onset, so the gate keys on "
            "directional change points rather than steady directions and places its budget better than "
            "rate-matched-random."
        ),
    },
)


class FeaturizersPreregRefusal(ValueError):
    """Raised when the featurizer preregistration inputs are malformed."""


def _multiplicity_block(n_featurizers: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:
    """Bonferroni family-wise control across the featurizer family, honest about the n equals 5 floor."""

    per_family_alpha = alpha / n_featurizers
    return {
        "n_featurizers": n_featurizers,
        "correction": "Bonferroni",
        "family_alpha": alpha,
        "per_featurizer_alpha": round(per_family_alpha, 12),
        "min_achievable_one_sided_p": round(min_one_sided_p, 12),
        "family_significance_reachable_at_n5": bool(min_one_sided_p <= per_family_alpha),
        "rationale": (
            "three frozen featurizers scored against one fixed test split inflate the family-wise error, "
            f"so each featurizer is held to the Bonferroni-adjusted alpha {per_family_alpha:.6f}. With "
            f"five paired seeds the smallest achievable one-sided sign-flip p is {min_one_sided_p:.5f}, "
            "which exceeds that adjusted alpha, so no single featurizer can clear family-wise significance "
            "from this family alone. This is a preregistered statistical wall: a featurizer may still "
            "register a SESOI-exceeding effect for triangulation, but family-wise promotion needs more "
            "seeds and bias-independent reproductions, never a larger claim squeezed from n equals 5"
        ),
    }


def build_featurizers_prereg(
    *,
    timestamp: str,
    operating_firing_fraction: float,
    n_test_clips: int,
    n_test_onsets: int,
    train_onset_density: float,
    n_test_frames: int,
    sesoi_f1: float = PREREGISTERED_SESOI_F1,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_down_flops: int = DEFAULT_C_DOWN_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
    featurizers: tuple[dict[str, str], ...] = FEATURIZERS,
    base_prereg_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    """Assemble the self-sealed featurizer preregistration body. The timestamp is passed, never a clock read."""

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise FeaturizersPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise FeaturizersPreregRefusal("n_seeds must be positive")
    if not featurizers:
        raise FeaturizersPreregRefusal("at least one featurizer must be preregistered")
    ids = [entry["featurizer_id"] for entry in featurizers]
    if len(set(ids)) != len(ids):
        raise FeaturizersPreregRefusal("featurizer ids must be unique")
    for entry in featurizers:
        if not entry.get("featurizer_id") or not entry.get("hypothesis"):
            raise FeaturizersPreregRefusal("each featurizer needs a featurizer_id and a one-line hypothesis")

    cb = compute_cost_benefit(
        c_train_flops=c_train_flops,
        c_down_flops=c_down_flops,
        operating_firing_fraction=operating_firing_fraction,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
    )
    permutations = 2**n_seeds
    min_one_sided_p = 1.0 / permutations

    body: dict[str, Any] = {
        "schema": FEATURIZERS_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "F1 frozen-featurizer iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
        "metric": PREREG_METRIC,
        "collar_frames": COLLAR_FRAMES,
        "collar_ms": COLLAR_FRAMES * FRAME_MS,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "front_end_swap": (
            "each family replaces the frozen zero-trained-parameter featurizer; the trained gate, its "
            "parameter count and ceiling, the sealed referee, controls, harness, and sign-flip are "
            "unchanged, so any F1 difference is attributable to the front-end, not the model"
        ),
        "sesoi": {
            "sesoi_f1": round(float(sesoi_f1), 12),
            "provisional": False,
            "selection_method": "cost-benefit (docs/ESCS_DEEP_RESEARCH.md Q5a option 3), same as base prereg",
            "rationale": _sesoi_rationale(cb, float(sesoi_f1)),
            "cost_benefit": cb.payload(),
            "train_onset_density": round(float(train_onset_density), 12),
        },
        "operating_point_rule": (
            "the swept firing budget whose firing fraction is closest to the train-set onset density; "
            "a fixed rule set before scoring, using only train labels, never a val or test F1 argmax"
        ),
        "sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "statistic": "mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
            "min_one_sided_p": round(min_one_sided_p, 12),
            "two_sided_floor": round(2.0 / permutations, 12),
            "alpha": 0.05,
            "two_sided_alpha_reachable": (2.0 / permutations) <= 0.05,
            "phipson_smyth_applied": False,
        },
        "multiplicity": _multiplicity_block(len(featurizers), min_one_sided_p, alpha=0.05),
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_frames": int(n_test_frames),
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
            "rationale": (
                "with about "
                f"{n_test_clips} test clips the clip is the experimental unit and frames are correlated "
                "sub-samples; a frame or clip bootstrap is refused and the claim verb is bounded to "
                "'consistent with', never 'demonstrates' or 'significant'"
            ),
        },
        "featurizers": [
            {"featurizer_id": entry["featurizer_id"], "hypothesis": entry["hypothesis"]}
            for entry in featurizers
        ],
        "executed_in_this_wave": ["spatial_doa"],
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
            "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
            "same direction; a single run, and any run at n equals 5 across this three-featurizer family, "
            "can never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


def write_featurizers_prereg(
    body: dict[str, Any], out_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH
) -> Path:
    """Write the self-sealed featurizer preregistration as canonical JSON bytes for a stable digest."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path


# ---------------------------------------------------------------------------
# Label-only structural facts. Read the fold-respecting split's labels only, never a feature or a score.
# ---------------------------------------------------------------------------


def structural_facts_from_adapter(
    *,
    foa_root: str | Path | None = None,
    metadata_root: str | Path | None = None,
    max_frames: int | None = None,
    n_val_rooms: int | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Derive the label-only structural facts the cost-benefit SESOI needs. Reads no feature and no score.

    The split is the native fold-respecting split (test is exactly fold-4 dev-test), and every fact below
    is a function of the clip and onset LABELS only. No audio is featurized, so this cannot leak a score.
    """

    from .adapter import RealStarssAdapter
    from .real_artifact import (
        DEFAULT_FOA_ROOT,
        DEFAULT_METADATA_ROOT,
        DEFAULT_N_VAL_ROOMS,
        RealBedConfig,
        _fold_respecting_split,
    )

    foa = Path(foa_root) if foa_root is not None else DEFAULT_FOA_ROOT
    meta = Path(metadata_root) if metadata_root is not None else DEFAULT_METADATA_ROOT
    n_val = DEFAULT_N_VAL_ROOMS if n_val_rooms is None else int(n_val_rooms)
    rates = target_rates or RealBedConfig().target_rates

    adapter = RealStarssAdapter(foa, meta, rights_clean=True, max_frames=max_frames)
    split = _fold_respecting_split(adapter, n_val)

    train_onsets = sum(len(clip.onsets) for clip in split.train)
    train_frames = sum(clip.n_frames for clip in split.train)
    train_density = train_onsets / train_frames if train_frames > 0 else 0.0
    operating_rate = min(rates, key=lambda r: abs(r - train_density))
    return {
        "operating_firing_fraction": float(operating_rate),
        "n_test_clips": len(split.test),
        "n_test_onsets": int(sum(len(clip.onsets) for clip in split.test)),
        "train_onset_density": float(train_density),
        "n_test_frames": int(sum(clip.n_frames for clip in split.test)),
    }


def _read_base_prereg_digest() -> str | None:
    """Return the committed base prereg canonical digest for traceability, or None if absent."""

    import json

    base_path = Path("proof/STARSS23_ESCS_BED.prereg.json")
    if not base_path.is_file():
        return None
    try:
        base = json.loads(base_path.read_bytes().decode("utf-8"))
    except (ValueError, OSError):
        return None
    digest = base.get("canonical_sha256")
    return str(digest) if isinstance(digest, str) else None


def _main(argv: list[str] | None = None) -> int:
    """Seal the featurizer preregistration from label-only structural facts and print its digest."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Seal the STARSS23 ESCS frozen-featurizer preregistration."
    )
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    parser.add_argument("--foa", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default=str(DEFAULT_FEATURIZERS_PREREG_PATH))
    args = parser.parse_args(argv)

    facts = structural_facts_from_adapter(foa_root=args.foa, metadata_root=args.metadata)
    body = build_featurizers_prereg(
        timestamp=args.timestamp,
        operating_firing_fraction=facts["operating_firing_fraction"],
        n_test_clips=facts["n_test_clips"],
        n_test_onsets=facts["n_test_onsets"],
        train_onset_density=facts["train_onset_density"],
        n_test_frames=facts["n_test_frames"],
        base_prereg_canonical_sha256=_read_base_prereg_digest(),
    )
    path = write_featurizers_prereg(body, args.out)
    print(f"wrote {path}")
    print(
        json.dumps(
            {
                "canonical_sha256": body["canonical_sha256"],
                "sesoi_f1": body["sesoi"]["sesoi_f1"],
                "n_featurizers": body["multiplicity"]["n_featurizers"],
                "family_significance_reachable_at_n5": body["multiplicity"][
                    "family_significance_reachable_at_n5"
                ],
                "featurizer_ids": [v["featurizer_id"] for v in body["featurizers"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
