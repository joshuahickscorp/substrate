"""The ladder Bed adapter for the STARSS23 event-formation bed.

This implements ``mop.ladder.ladder_contracts.Bed`` so the bed plugs into the Stage 3 ladder as a
deterministic task environment. It declares the three controls, the matched full-system budget, and the
two regimes the recipe requires:

- ``null_regime``: nuisance-only. The audio carries only band-limited distractor grains and the onset
  labels are independent of the audio, so no featurizer signal predicts them. A learned gate cannot beat
  the rate-matched-random control, so the strong null holds.
- ``favorable_regime``: planted onsets. Coherent first-order-Ambisonics-steered onset grains sit at the
  labeled frames, so a gate that reads the signal-band signature can fire on them.

The Bed carries no training and no scoring itself; it hands the ladder the deterministic regimes and the
matched budget. Synthetic data caps any downstream verdict at mechanics-ok. House style: no em dashes and
no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mop.ladder.stage_ladder import MatchedBudget
from mop.science.budget import ARM_ALWAYS_ON, ARM_BEST_SINGLE, ARM_RATE_MATCHED_RANDOM
from mop.substrate.events import canonical_sha256

from . import BED_ID
from .featurizer import FLOPS_PER_FRAME
from .fixtures import (
    REGIME_FAVORABLE,
    REGIME_NULL,
    SyntheticStarssConfig,
    generate_clip,
)
from .gate import DEFAULT_EPOCHS, FLOPS_PER_INFERENCE, param_count, training_flops

BED_ADAPTER_SCHEMA = "mop-starss23-escs-bed-adapter/v1"

# The three controls the harness scores against the candidate. noisy_tv is the irreducible-noise guard.
BED_CONTROLS: tuple[str, ...] = (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv")

# Full-lifecycle matched-budget anchors from the recipe (docs/ESCS_DEEP_RESEARCH.md).
NOMINAL_TEST_FRAMES = 24_000  # a 40 clip by 60 s test set at 10 frames per second
NOMINAL_TRAIN_FRAMES = 54_000  # a 90 clip by 60 s train set
NOMINAL_FIRING_FRACTION = 0.10  # a ~10 percent firing budget
DOWNSTREAM_FLOPS_PER_FIRING = 40_000
N_PAIRED_SEEDS = 5

# Regime samples stay small so the Bed is cheap to materialize when the ladder probes it.
_REGIME_CLIPS = 2
_REGIME_ROOM = "room00"


def _matched_budget() -> MatchedBudget:
    """The deterministic full-lifecycle matched budget, asserted under the ~6e10 FLOP ceiling."""

    firings = int(NOMINAL_FIRING_FRACTION * NOMINAL_TEST_FRAMES)
    flops = (
        FLOPS_PER_FRAME * NOMINAL_TEST_FRAMES
        + FLOPS_PER_INFERENCE * NOMINAL_TEST_FRAMES
        + firings * DOWNSTREAM_FLOPS_PER_FIRING
        + training_flops(NOMINAL_TRAIN_FRAMES, DEFAULT_EPOCHS)
    )
    # A deterministic nominal wall at a 1 GFLOP/s reference keeps the budget byte-reproducible.
    return MatchedBudget(params=param_count(), flops=flops, wall_ns=flops, seeds=N_PAIRED_SEEDS)


@dataclass(frozen=True, slots=True)
class RegimeSample:
    """A deterministic regime materialization: the generated clips and a content digest."""

    regime: str
    seed: int
    clip_ids: tuple[str, ...]
    onset_counts: tuple[int, ...]
    clip_digests: tuple[str, ...]
    schema: str = BED_ADAPTER_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "seed": self.seed,
            "clip_ids": list(self.clip_ids),
            "onset_counts": list(self.onset_counts),
            "clip_digests": list(self.clip_digests),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


class Starss23EscsBed:
    """The Stage 3 ladder Bed for the STARSS23 event-formation lane. Deterministic and mechanics-only."""

    mechanism_id: str = BED_ID

    def __init__(self, *, clip_seconds: float = 6.0) -> None:
        self._clip_seconds = float(clip_seconds)

    def controls(self) -> tuple[str, ...]:
        return BED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        return _matched_budget()

    def _sample(self, seed: int, regime: str) -> RegimeSample:
        config = SyntheticStarssConfig(clip_seconds=self._clip_seconds, base_seed=int(seed))
        clip_ids: list[str] = []
        onset_counts: list[int] = []
        clip_digests: list[str] = []
        for mix in range(_REGIME_CLIPS):
            clip_id = f"fold3_room0_mix{mix:03d}"
            clip, _audio = generate_clip(
                clip_id=clip_id, room_id=_REGIME_ROOM, regime=regime, config=config
            )
            clip_ids.append(clip.clip_id)
            onset_counts.append(len(clip.onsets))
            clip_digests.append(clip.digest())
        return RegimeSample(
            regime=regime,
            seed=int(seed),
            clip_ids=tuple(clip_ids),
            onset_counts=tuple(onset_counts),
            clip_digests=tuple(clip_digests),
        )

    def null_regime(self, seed: int) -> RegimeSample:
        """Nuisance-only regime: onset labels are independent of the audio, so the strong null holds."""

        return self._sample(seed, REGIME_NULL)

    def favorable_regime(self, seed: int) -> RegimeSample:
        """Planted-onset regime: coherent Ambisonics-steered onset grains sit at the labeled frames."""

        return self._sample(seed, REGIME_FAVORABLE)


def build_bed(*, clip_seconds: float = 6.0) -> Starss23EscsBed:
    """Return the default STARSS23 ESCS ladder Bed."""

    return Starss23EscsBed(clip_seconds=clip_seconds)
