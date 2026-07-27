from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .adapter import domain_seed

_RMR_NAMESPACE = "mop.beds.starss23.controls.rate_matched_random"
DEFAULT_NOISY_TV_TOLERANCE = 0.05


class ControlRefusal(ValueError):
    pass


def _unique_sorted_frames(frames: Sequence[int], n_frames: int, label: str) -> list[int]:
    seen: set[int] = set()
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise ControlRefusal(f"{label} must contain frames in [0, {n_frames})")
        seen.add(frame)
    return sorted(seen)


def rate_matched_random_fires(
    candidate_fires: Sequence[int], n_frames: int, *, seed: int, clip_id: str
) -> list[int]:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ControlRefusal("clip_id must be a non-empty string")
    fires = _unique_sorted_frames(candidate_fires, n_frames, "candidate_fires")
    if len(fires) >= n_frames:
        return list(range(n_frames))
    rng = np.random.default_rng(domain_seed(seed, f"{_RMR_NAMESPACE}:{clip_id}", b"mop-starss23-controls-v1"))
    return sorted(int(position) for position in rng.choice(n_frames, size=len(fires), replace=False))


def always_on_fires(n_frames: int) -> list[int]:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return list(range(n_frames))


def never_update_reestimates(n_frames: int) -> list[int]:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return []


def at_chance(
    firing_rate_on_noise: float,
    base_rate: float,
    *,
    tolerance: float = DEFAULT_NOISY_TV_TOLERANCE,
) -> bool:
    for name, value in (("firing_rate_on_noise", firing_rate_on_noise), ("base_rate", base_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ControlRefusal(f"{name} must be a rate in [0, 1]")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0.0:
        raise ControlRefusal("tolerance must be a nonnegative number")
    return float(firing_rate_on_noise) <= float(base_rate) + float(tolerance)
