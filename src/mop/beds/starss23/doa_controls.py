"""Direction-of-arrival bed, component 5: the four honest controls.

Three are reused unchanged BY IMPORT from the sealed ``controls.py`` (never edited); one, the zero-compute
floor, is a net-new trivial producer, mirroring ``count_controls.py`` exactly:

(a) rate-matched-random (WHERE, matched budget). Reuse ``controls.rate_matched_random_fires``: the SAME
    re-estimation COUNT K as the candidate, positions permuted uniformly, matched per clip, per seed, and
    per architecture (K may differ slightly between Architecture A and B at the same nominal operating
    rate, since their p-value distributions differ, so each architecture's candidate is matched against
    its OWN rate-matched-random realization). This is the PRIMARY control.

(b) always-on (max-compute ceiling). Reuse ``controls.always_on_fires``: R = every frame. Its coasted-MAE
    result and FLOP model do not depend on gate architecture at all (it never runs a gate), so it can be
    computed once and reused across both architecture reports.

(c) never-update / coast-from-frame-0 (the zero-compute floor). Net-new: R = []. Also architecture-
    independent, computed once, reused.

(d) noisy-TV (irreducible-noise at-chance). Reuse ``controls.at_chance``: the trained gate's re-estimation
    rate on a structureless pure-aleatoric channel must stay at chance (base rate plus tolerance), checked
    per architecture per seed.

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

DOA_CONTROLS_SCHEMA = "mop-starss23-doa-controls/v1"

__all__ = [
    "DOA_CONTROLS_SCHEMA",
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
