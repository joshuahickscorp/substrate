
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

BED_CONTROLS: tuple[str, ...] = (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv")

NOMINAL_TEST_FRAMES = 24_000  # a 40 clip by 60 s test set at 10 frames per second
NOMINAL_TRAIN_FRAMES = 54_000  # a 90 clip by 60 s train set
NOMINAL_FIRING_FRACTION = 0.10  # a ~10 percent firing budget
DOWNSTREAM_FLOPS_PER_FIRING = 40_000
N_PAIRED_SEEDS = 5

_REGIME_CLIPS = 2
_REGIME_ROOM = "room00"


def _matched_budget() -> MatchedBudget:

    firings = int(NOMINAL_FIRING_FRACTION * NOMINAL_TEST_FRAMES)
    flops = (
        FLOPS_PER_FRAME * NOMINAL_TEST_FRAMES
        + FLOPS_PER_INFERENCE * NOMINAL_TEST_FRAMES
        + firings * DOWNSTREAM_FLOPS_PER_FIRING
        + training_flops(NOMINAL_TRAIN_FRAMES, DEFAULT_EPOCHS)
    )
    return MatchedBudget(params=param_count(), flops=flops, wall_ns=flops, seeds=N_PAIRED_SEEDS)


@dataclass(frozen=True, slots=True)
class RegimeSample:

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

        return self._sample(seed, REGIME_NULL)

    def favorable_regime(self, seed: int) -> RegimeSample:

        return self._sample(seed, REGIME_FAVORABLE)


def build_bed(*, clip_seconds: float = 6.0) -> Starss23EscsBed:

    return Starss23EscsBed(clip_seconds=clip_seconds)
