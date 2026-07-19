
from __future__ import annotations

import numpy as np

from .gate import DEFAULT_THETA, CandidateGate, GateRefusal, OnlineState
from .schema import COLLAR_FRAMES

# The refractory / NMS window is the DCASE collar width on the frozen 100 ms grid: two frames on each
# side. Committed fires are separated by strictly more than this, so no second fire lands inside an
# already-covered collar. This is a firing-policy constant, not a trained quantity.
DEFAULT_WINDOW_FRAMES = COLLAR_FRAMES  # 2 frames == plus-or-minus 200 ms


def _require_window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise GateRefusal("refractory window must be a nonnegative integer number of frames")
    return window


def refractory_nms_select(probs: np.ndarray, theta: float, window: int) -> list[int]:

    trace = np.asarray(probs, dtype=np.float64)
    if trace.ndim != 1:
        raise GateRefusal("probs must be a one-dimensional p_fire trace")
    threshold = float(theta)
    window = _require_window(window)

    fires: list[int] = []
    armed = False
    peak_frame = -1
    peak_score = -np.inf
    deadline = -1
    n = int(trace.shape[0])
    for frame in range(n):
        score = float(trace[frame])
        # Close and commit a held peak the moment we step strictly past its window.
        if armed and frame > deadline:
            fires.append(peak_frame)
            armed = False
        if score >= threshold:
            if not armed:
                armed = True
                peak_frame = frame
                peak_score = score
                deadline = frame + window
            elif score > peak_score:
                # A strictly higher score arrived inside the window: move the single held peak to it.
                peak_frame = frame
                peak_score = score
                deadline = frame + window
            # else: a non-maximum inside the window is suppressed.
    if armed:
        fires.append(peak_frame)
    return fires


class RefractoryNmsGate(CandidateGate):

    def __init__(self, *, window: int = DEFAULT_WINDOW_FRAMES, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.window = _require_window(window)

    def causal_probs(self, features: np.ndarray, theta: float = DEFAULT_THETA) -> np.ndarray:

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise GateRefusal("features must be a (n_frames, D_FEAT) block")
        threshold = float(theta)
        state = OnlineState.initial()
        probs = np.empty(features.shape[0], dtype=np.float64)
        for frame in range(features.shape[0]):
            p_fire = self.infer(features[frame], state)
            probs[frame] = p_fire
            state = state.update(features[frame], p_fire, p_fire >= threshold)
        return probs

    def refractory_fires(
        self, features: np.ndarray, theta: float, window: int | None = None
    ) -> list[int]:

        window = self.window if window is None else _require_window(window)
        probs = self.causal_probs(features, theta)
        return refractory_nms_select(probs, theta, window)


def pooled_post_nms_fraction(
    prob_traces: list[np.ndarray], theta: float, window: int
) -> float:

    total_frames = sum(int(trace.shape[0]) for trace in prob_traces)
    if total_frames <= 0:
        return 0.0
    total_fires = sum(len(refractory_nms_select(trace, theta, window)) for trace in prob_traces)
    return total_fires / total_frames


def tune_theta_for_rate(
    prob_traces: list[np.ndarray],
    target_rate: float,
    window: int,
    *,
    n_candidates: int = 60,
) -> float:

    if not prob_traces:
        raise GateRefusal("theta tuning needs at least one val p_fire trace")
    pooled = np.concatenate([np.asarray(trace, dtype=np.float64).ravel() for trace in prob_traces])
    if pooled.size == 0:
        raise GateRefusal("theta tuning needs non-empty val p_fire traces")
    rate = float(target_rate)
    if not 0.0 < rate < 1.0:
        raise GateRefusal("target_rate must be strictly between 0 and 1")

    # Post-NMS fraction is at most a raw fraction, so admit more raw budget than the target: sweep raw
    # firing fractions from a small floor up to several times the target, and read off the quantile theta.
    lo = max(rate * 0.25, 1.0 / max(1, pooled.size))
    hi = min(0.9, rate * 8.0)
    raw_fractions = np.linspace(lo, hi, int(n_candidates))
    # A threshold strictly above the maximum commits nothing; include it so the search is well posed.
    candidate_thetas = [float(np.quantile(pooled, 1.0 - frac)) for frac in raw_fractions]
    candidate_thetas.append(float(pooled.max()) + 1.0)

    best_theta = candidate_thetas[0]
    best_gap = np.inf
    for theta in candidate_thetas:
        fraction = pooled_post_nms_fraction(prob_traces, theta, window)
        gap = abs(fraction - rate)
        if gap < best_gap or (gap == best_gap and theta > best_theta):
            best_gap = gap
            best_theta = theta
    return best_theta
