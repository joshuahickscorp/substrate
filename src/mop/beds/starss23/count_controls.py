"""Concurrent-source-counting bed, component 4: the four honest controls.

Each control maps one-to-one onto a sealed onset control and isolates a different failure mode. Three are
reused unchanged BY IMPORT from the sealed ``controls.py`` (never edited); one, the zero-compute floor, is
a net-new trivial producer:

(a) rate-matched-random (WHERE, matched budget). Reuse ``controls.rate_matched_random_fires``: the SAME
    re-estimation COUNT K as the candidate, positions permuted uniformly at random, matched per clip and
    per seed. Any MAE gap is then WHERE re-estimations land, not how many. This is the primary control.

(b) always-on (max-compute ceiling). Reuse ``controls.always_on_fires``: R = all frames, so emitted = E
    and MAE = mean(|E - C_gt|), the estimator's own tracking error at full compute.

(c) never-update / coast-from-frame-0 (the zero-compute floor). Net-new: R = []. emitted = c0 = 0
    everywhere, so MAE = mean(|C_gt|). Free but maximally drifted.

(d) noisy-TV (irreducible-noise at-chance). Reuse ``controls.at_chance``: the gate's re-estimation rate on
    a structureless pure-aleatoric channel must stay at chance (base rate plus tolerance). A gate that
    re-estimates preferentially on noise is chasing novelty and FAILS. The channel itself (white-noise FOA
    audio featurized by the frozen count front-end and affine-matched to test marginals) is built in the
    producer, exactly as the onset real producer builds its noisy-TV channel.

This module trains nothing and edits no sealed module. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from .controls import (  # reused unchanged from the sealed onset control path
    DEFAULT_NOISY_TV_TOLERANCE,
    ControlRefusal,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)

COUNT_CONTROLS_SCHEMA = "mop-starss23-count-controls/v1"

__all__ = [
    "COUNT_CONTROLS_SCHEMA",
    "DEFAULT_NOISY_TV_TOLERANCE",
    "ControlRefusal",
    "always_on_fires",
    "at_chance",
    "rate_matched_random_fires",
    "never_update_reestimates",
]


def never_update_reestimates(n_frames: int) -> list[int]:
    """The zero-compute floor: never re-estimate. Returns the empty re-estimation set for any clip length."""

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return []
