
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mop.substrate.events import canonical_sha256

from .featurizer import D_FEAT
from .gate import (
    FLOPS_PER_INFERENCE,
    PARAM_CEILING,
    CandidateGate,
    GateRefusal,
    OnlineState,
)
from .referee import score_arm
from .schema import COLLAR_FRAMES

VARIANT_ID = "recurrence_spread"

N_SPREAD_PARAMS = 2

REFRACTORY_FRAMES = COLLAR_FRAMES + 1  # 3 frames

DEFAULT_DENSITY_GRID: tuple[float, ...] = (0.0, 1.0, 2.0)
DEFAULT_REFRACTORY_GRID: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)

FLOPS_INVERSE_SIGMOID = 6  # clip, ratio, log
FLOPS_PENALTY = 10  # gap, divide, rectified refractory, density ratio, two multiplies, one add
FLOPS_RESIGMOID = 6  # subtract plus a logistic
FLOPS_SPREAD_HEAD = FLOPS_INVERSE_SIGMOID + FLOPS_PENALTY + FLOPS_RESIGMOID  # 22
FLOPS_PER_INFERENCE_SPREAD = FLOPS_PER_INFERENCE + FLOPS_SPREAD_HEAD  # 6407

_PROB_EPS = 1e-12


def _sigmoid_scalar(x: float) -> float:

    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _prob_to_logit(p: float) -> float:

    clipped = min(max(float(p), _PROB_EPS), 1.0 - _PROB_EPS)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class SpreadTrainingReport:

    rate: float
    density_grid: tuple[float, ...]
    refractory_grid: tuple[float, ...]
    n_search_evals: int
    frames_per_pass: int
    search_flops: int
    base_train_f1: float
    best_train_f1: float
    density_weight: float
    refractory_weight: float
    selected_no_spread: bool

    def payload(self) -> dict[str, Any]:
        return {
            "rate": float(self.rate),
            "density_grid": [float(v) for v in self.density_grid],
            "refractory_grid": [float(v) for v in self.refractory_grid],
            "n_search_evals": self.n_search_evals,
            "frames_per_pass": self.frames_per_pass,
            "search_flops": self.search_flops,
            "base_train_f1": round(float(self.base_train_f1), 12),
            "best_train_f1": round(float(self.best_train_f1), 12),
            "density_weight": float(self.density_weight),
            "refractory_weight": float(self.refractory_weight),
            "selected_no_spread": bool(self.selected_no_spread),
        }


@dataclass
class RecurrenceSpreadGate:

    signal_gate: CandidateGate
    rho: float
    density_weight: float = 0.0
    refractory_weight: float = 0.0
    refractory_frames: int = REFRACTORY_FRAMES
    search_flops: int = 0
    spread_report: SpreadTrainingReport | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.signal_gate, CandidateGate):
            raise GateRefusal("signal_gate must be a committed CandidateGate")
        rho = float(self.rho)
        if not math.isfinite(rho) or not 0.0 < rho < 1.0:
            raise GateRefusal("rho (operating firing fraction) must lie strictly in (0, 1)")
        self.rho = rho
        for name in ("density_weight", "refractory_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise GateRefusal(f"{name} must be a nonnegative finite number")
            setattr(self, name, value)
        if (
            isinstance(self.refractory_frames, bool)
            or not isinstance(self.refractory_frames, int)
            or self.refractory_frames <= 0
        ):
            raise GateRefusal("refractory_frames must be a positive integer")
        n_params = self.n_params()
        if n_params > PARAM_CEILING:
            raise GateRefusal(
                f"recurrence_spread trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        assert n_params <= PARAM_CEILING, "recurrence_spread parameter ceiling breached"


    def n_spread_params(self) -> int:
        return N_SPREAD_PARAMS

    def n_params(self) -> int:

        return self.signal_gate.n_params() + N_SPREAD_PARAMS

    def flops_per_inference(self) -> int:
        return FLOPS_PER_INFERENCE_SPREAD

    def signal_training_flops(self, n_train_frames: int, epochs: int) -> int:
        return self.signal_gate.training_flops(n_train_frames, epochs)

    def total_training_flops(self, n_train_frames: int, epochs: int) -> int:

        return self.signal_training_flops(n_train_frames, epochs) + int(self.search_flops)

    def parameter_digest(self) -> str:

        payload = {
            "variant_id": VARIANT_ID,
            "signal_parameter_digest": self.signal_gate.parameter_digest(),
            "density_weight": float(self.density_weight),
            "refractory_weight": float(self.refractory_weight),
            "rho": float(self.rho),
            "refractory_frames": int(self.refractory_frames),
            "n_params": self.n_params(),
        }
        return canonical_sha256(payload)


    def _penalty(self, state: OnlineState) -> float:

        density = state.firing_rate_ema / self.rho  # firing_rate_ema >= 0, rho > 0 => density >= 0
        if state.last_fire_frame >= 0.0:
            gap = state.n_frames - state.last_fire_frame
            refractory = max(0.0, 1.0 - gap / float(self.refractory_frames))
        else:
            refractory = 0.0  # never fired yet: no refractory suppression
        return self.density_weight * density + self.refractory_weight * refractory

    def infer(self, features: np.ndarray, state: OnlineState) -> float:

        p_signal = self.signal_gate.infer(features, state)
        if self.density_weight == 0.0 and self.refractory_weight == 0.0:
            return p_signal  # exact committed null when no spreading is fit
        penalty = self._penalty(state)
        if penalty <= 0.0:
            return p_signal
        return _sigmoid_scalar(_prob_to_logit(p_signal) - penalty)

    def fire(
        self, features: np.ndarray, state: OnlineState, theta: float
    ) -> tuple[bool, float]:

        p_fire = self.infer(features, state)
        return (p_fire >= float(theta), p_fire)

    def causal_pass(self, features: np.ndarray, theta: float) -> tuple[list[int], np.ndarray]:

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != D_FEAT:
            raise GateRefusal(f"features must be shape (n_frames, {D_FEAT})")
        state = OnlineState.initial()
        fires: list[int] = []
        probs = np.empty(features.shape[0], dtype=np.float64)
        for frame in range(features.shape[0]):
            p_fire = self.infer(features[frame], state)
            probs[frame] = p_fire
            fired = p_fire >= float(theta)
            if fired:
                fires.append(frame)
            state = state.update(features[frame], p_fire, fired)
        return fires, probs


    def fit_spread(
        self,
        train_clips: list[tuple[np.ndarray, list[int]]],
        rate: float,
        *,
        density_grid: tuple[float, ...] = DEFAULT_DENSITY_GRID,
        refractory_grid: tuple[float, ...] = DEFAULT_REFRACTORY_GRID,
        collar_frames: int = COLLAR_FRAMES,
    ) -> SpreadTrainingReport:

        if not train_clips:
            raise GateRefusal("fit_spread needs at least one train clip")
        if not 0.0 < float(rate) < 1.0:
            raise GateRefusal("rate must lie strictly in (0, 1)")
        if 0.0 not in density_grid or 0.0 not in refractory_grid:
            raise GateRefusal("the spreading grid must contain the no-spread point (0.0, 0.0)")

        frames_per_pass = int(sum(np.asarray(f).shape[0] for f, _ in train_clips))

        def _train_f1(w_density: float, w_refractory: float) -> float:
            self.density_weight = float(w_density)
            self.refractory_weight = float(w_refractory)
            probs = np.concatenate([self.causal_pass(f, 0.5)[1] for f, _ in train_clips])
            theta = float(np.quantile(probs, 1.0 - float(rate)))
            scored = [(list(onsets), self.causal_pass(f, theta)[0]) for f, onsets in train_clips]
            return score_arm(scored, collar_frames).f1

        base_f1 = _train_f1(0.0, 0.0)
        best_f1 = base_f1
        best_weights = (0.0, 0.0)
        n_evals = 1
        for w_density in density_grid:
            for w_refractory in refractory_grid:
                if w_density == 0.0 and w_refractory == 0.0:
                    continue  # already evaluated as the base
                f1 = _train_f1(w_density, w_refractory)
                n_evals += 1
                total = w_density + w_refractory
                best_total = best_weights[0] + best_weights[1]
                if f1 > best_f1 or (f1 == best_f1 and total < best_total):
                    best_f1 = f1
                    best_weights = (float(w_density), float(w_refractory))

        self.density_weight, self.refractory_weight = best_weights
        self.search_flops = int(n_evals * 2 * frames_per_pass * FLOPS_PER_INFERENCE_SPREAD)
        report = SpreadTrainingReport(
            rate=float(rate),
            density_grid=tuple(float(v) for v in density_grid),
            refractory_grid=tuple(float(v) for v in refractory_grid),
            n_search_evals=n_evals,
            frames_per_pass=frames_per_pass,
            search_flops=self.search_flops,
            base_train_f1=float(base_f1),
            best_train_f1=float(best_f1),
            density_weight=float(best_weights[0]),
            refractory_weight=float(best_weights[1]),
            selected_no_spread=bool(best_weights == (0.0, 0.0)),
        )
        self.spread_report = report
        return report
