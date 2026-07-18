"""E1 gate variant "refractory_nms": a post-decision refractory / non-maximum-suppression policy.

This is a net-new, additive component. It changes no sealed scoring logic and does not edit the committed
``gate.py``. The first real Stage-3 run nulled because the trained value-of-computation gate clusters
roughly 42 percent of its fires adjacently on high-energy regions and so recovers fewer distinct onsets
than uniform random placement at matched budget (docs/mixture_of_perspectives/26_escs_starss23_bed.md).

The refractory_nms variant keeps the committed gate's model, its training, its parameters, its online
state, and its per-frame ``p_fire`` trace byte-for-byte identical. The ONLY thing that differs is the
firing SELECTION policy applied to that trace: after a fire, a collar-width window suppresses further
fires unless a strictly higher score arrives, in which case the single held peak moves to the higher
score. Each cluster of supra-threshold frames therefore commits exactly ONE fire (its local maximum),
so a fixed firing budget spreads across distinct onsets instead of piling adjacent fires onto one
high-energy region. This is a post-decision policy on ``p_fire``, exactly as the sealed variant
preregistration states.

Design invariants, asserted in code and in tests:

- ``RefractoryNmsGate`` subclasses the committed ``CandidateGate`` with an identical ``__init__`` and
  ``fit``, so its trainable-parameter count (3193, hard-capped at 4096), its few-KB online state, its
  ``infer(features, state)`` interface (blind to ground truth online), and its amortized ``C_train`` are
  the committed ones unchanged. Paired-seed weights are byte-identical to ``CandidateGate`` at the same
  seed, so the variant isolates the firing policy and nothing else.
- The causal ``p_fire`` pass advances the online state exactly as the committed producer's causal pass
  does (``fired = p_fire >= theta`` provisional excitation), so the trace the selection reads is the same
  trace the committed gate produces at that threshold.
- The refractory NMS is deterministic and causal (bounded latency of ``window`` frames): it never reads a
  label, a class, or a direction of arrival, and it emits fires separated by strictly more than ``window``
  frames, so no second fire lands inside an already-covered collar.

House style: no em dashes and no en dashes.
"""

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
    """Select committed fire frames from a p_fire trace by causal collar-width non-maximum suppression.

    The rule, written out fully so an independent reader can reproduce it byte for byte:

    Sweep frames left to right. A window is "armed" by the first frame whose score reaches ``theta``; the
    armed peak is that frame and its score, and the window runs for ``window`` frames past the peak. While
    armed, a later supra-threshold frame whose score is STRICTLY higher than the held peak moves the peak
    to that frame and re-arms the window from there (this is the "unless a strictly higher score arrives"
    override); a supra-threshold frame that is not strictly higher is a non-maximum and is suppressed. The
    first frame that falls strictly past the window commits the single held peak as one fire and, if it is
    itself supra-threshold, arms the next window. Any peak still held at the end of the trace is committed.

    Consequences: each maximal cluster of supra-threshold activity commits exactly one fire (its local
    maximum), and committed fires are separated by strictly more than ``window`` frames, so no second fire
    lands inside an already-covered collar. Deterministic; ties on equal scores keep the earliest frame.
    """

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
    """The committed value-of-computation gate with a refractory / NMS firing-selection policy bolted on.

    Everything trained is inherited unchanged from ``CandidateGate``: the 264 to 12 to 1 MLP, the 3193
    trainable parameters (asserted against the 4096 ceiling by the inherited constructor), the few-KB
    ``OnlineState``, the label-blind ``infer(features, state)`` path, ``fit``, and the amortized
    ``C_train`` exposed for the FLOP ledger. Paired-seed weights are byte-identical to ``CandidateGate``
    at the same seed. Only the firing policy differs: ``causal_probs`` reproduces the committed causal
    ``p_fire`` trace, and ``refractory_fires`` selects the committed fires by collar-width NMS on it.
    """

    def __init__(self, *, window: int = DEFAULT_WINDOW_FRAMES, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.window = _require_window(window)

    def causal_probs(self, features: np.ndarray, theta: float = DEFAULT_THETA) -> np.ndarray:
        """Return the causal per-frame p_fire trace, advancing state as the committed producer does.

        The online state is advanced with the committed producer's provisional decision
        ``fired = p_fire >= theta``, so this trace is byte-identical to the committed gate's causal pass at
        the same threshold. No label ever enters here.
        """

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
        """Committed fire frames for one clip: causal p_fire trace, then collar-width refractory NMS."""

        window = self.window if window is None else _require_window(window)
        probs = self.causal_probs(features, theta)
        return refractory_nms_select(probs, theta, window)


def pooled_post_nms_fraction(
    prob_traces: list[np.ndarray], theta: float, window: int
) -> float:
    """Fraction of frames that commit a fire under refractory NMS at ``theta``, pooled over the traces."""

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
    """Pick the theta whose pooled post-NMS firing fraction is closest to ``target_rate``. Label-free.

    The committed producer sets its threshold by a val p_fire quantile so the RAW firing fraction hits the
    swept target rate. The refractory NMS commits fewer fires than it thresholds, so this tunes the
    threshold so the POST-NMS firing fraction hits the same target rate: the variant then spends the same
    firing budget as the swept rate, but spread by NMS instead of clustered. Uses only the val p_fire
    traces (no label, no test score). Candidate thresholds are val-prob quantiles spanning a band around
    the target rate; ties on the fraction gap break toward the HIGHER threshold (fewer fires,
    conservative), mirroring the best-single tuning convention.
    """

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
