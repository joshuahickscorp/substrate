
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .count_featurizer import D_CFEAT, N_CHANNELS, N_MEL
from .gate import (
    C_TRAIN_ANCHOR,
    DEFAULT_EPOCHS,
    _sigmoid,
    inference_flops,
    param_count,
    training_flops,
)

COUNT_GATE_SCHEMA = "mop-starss23-count-gate/v1"

# Gate geometry. Byte-identical width to the sealed onset gate so its cost anchors reuse exactly.
N_CFEAT = D_CFEAT  # 256 frozen count features
N_CONLINE = 8  # self-derived online scalars
D_IN = N_CFEAT + N_CONLINE  # 264 gate inputs
HIDDEN = 12
N_OUT = 1

PARAM_CEILING = 4096
STATE_CEILING_BYTES = 8192

DEFAULT_THETA = 0.5
PHASE_HORIZON = 10.0  # frames per positional-clock cycle (1 second at 100 ms frames)
EMA_DECAY = 0.1

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_PONDER_LAMBDA = 0.02
COUNT_VOC_WINDOW = 1  # a re-estimation is valuable within +/- 1 frame of a count change

# The first N_MEL * N_CHANNELS = 128 features are the positive (source-enter) flux; the last 128 are the
# negative (source-leave) flux, matching count_featurizer.featurize's concatenation order.
_POS_BLOCK = N_MEL * N_CHANNELS  # 128

# Anchors reused unchanged from gate.py so the counting bed and the onset bed charge identical costs.
C_TRAIN_ANCHOR_COUNT = C_TRAIN_ANCHOR
FLOPS_PER_INFERENCE = inference_flops(D_IN, HIDDEN, N_OUT)  # 6385


class CountGateRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CountOnlineState:

    n_frames: float = 0.0
    n_reestimates: float = 0.0
    last_reestimate_frame: float = -1.0
    reestimate_rate_ema: float = 0.0
    energy_ema: float = 0.0
    pos_flux_peak_ema: float = 0.0
    neg_flux_peak_ema: float = 0.0
    last_p_reestimate: float = 0.0

    @classmethod
    def initial(cls) -> CountOnlineState:
        return cls()

    @classmethod
    def state_bytes(cls) -> int:

        return len(fields(cls)) * 8

    def to_vector(self) -> np.ndarray:

        recency = math.tanh((self.n_frames - self.last_reestimate_frame) / PHASE_HORIZON)
        phase = math.fmod(self.n_frames, PHASE_HORIZON) / PHASE_HORIZON
        cumulative_fraction = self.n_reestimates / max(1.0, self.n_frames)
        return np.array(
            [
                recency,
                self.reestimate_rate_ema,
                math.tanh(self.energy_ema),
                math.tanh(self.pos_flux_peak_ema),
                math.tanh(self.neg_flux_peak_ema),
                self.last_p_reestimate,
                phase,
                cumulative_fraction,
            ],
            dtype=np.float64,
        )

    def update(self, features: np.ndarray, p_reestimate: float, reestimated: bool) -> CountOnlineState:

        features = np.asarray(features, dtype=np.float64)
        energy = float(features.mean()) if features.size else 0.0
        pos_peak = float(features[:_POS_BLOCK].max()) if features.size else 0.0
        neg_peak = float(features[_POS_BLOCK:].max()) if features.size else 0.0
        did = 1.0 if reestimated else 0.0
        return CountOnlineState(
            n_frames=self.n_frames + 1.0,
            n_reestimates=self.n_reestimates + did,
            last_reestimate_frame=self.n_frames if reestimated else self.last_reestimate_frame,
            reestimate_rate_ema=(1.0 - EMA_DECAY) * self.reestimate_rate_ema + EMA_DECAY * did,
            energy_ema=(1.0 - EMA_DECAY) * self.energy_ema + EMA_DECAY * energy,
            pos_flux_peak_ema=(1.0 - EMA_DECAY) * self.pos_flux_peak_ema + EMA_DECAY * pos_peak,
            neg_flux_peak_ema=(1.0 - EMA_DECAY) * self.neg_flux_peak_ema + EMA_DECAY * neg_peak,
            last_p_reestimate=float(p_reestimate),
        )


@dataclass
class CountTrainingReport:

    epochs: int
    n_train_frames: int
    learning_rate: float
    ponder_lambda: float
    loss_history: tuple[float, ...]
    final_reestimate_rate: float
    c_train_flops: int

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_frames": self.n_train_frames,
            "learning_rate": self.learning_rate,
            "ponder_lambda": self.ponder_lambda,
            "loss_history": [float(value) for value in self.loss_history],
            "final_reestimate_rate": self.final_reestimate_rate,
            "c_train_flops": self.c_train_flops,
        }


class CountGateInterface:

    __slots__ = ()
    _feature_dim = N_CFEAT
    _refusal: type[ValueError] = CountGateRefusal

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise self._refusal(f"gate input must be shape (N, {self.d_in})")
        return self._forward(x)[0]

    def _assemble(self, features: np.ndarray, state: CountOnlineState) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self._feature_dim,):
            raise self._refusal(f"features must be a length {self._feature_dim} vector")
        if not isinstance(state, CountOnlineState):
            raise self._refusal("state must be a CountOnlineState")
        return np.concatenate([features, state.to_vector()])

    def infer(self, features: np.ndarray, state: CountOnlineState) -> float:
        x = self._assemble(features, state)
        return float(self.predict_proba(x[None, :])[0])

    def decide(
        self, features: np.ndarray, state: CountOnlineState, theta: float | None = None
    ) -> tuple[bool, float]:
        threshold = self.theta if theta is None else float(theta)
        probability = self.infer(features, state)
        return (probability >= threshold, probability)


class CountGate(CountGateInterface):

    _SEED_NAMESPACE = "mop.beds.starss23.count_gate.init"

    def __init__(
        self,
        *,
        seed: int = 0,
        d_in: int = D_IN,
        hidden: int = HIDDEN,
        n_out: int = N_OUT,
        theta: float = DEFAULT_THETA,
    ) -> None:
        if hidden <= 0 or d_in <= 0 or n_out <= 0:
            raise CountGateRefusal("gate dimensions must be positive")
        n_params = param_count(d_in, hidden, n_out)
        if n_params > PARAM_CEILING:
            raise CountGateRefusal(
                f"gate trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        assert n_params <= PARAM_CEILING, "count gate parameter ceiling breached"
        state_bytes = CountOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise CountGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "count gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise CountGateRefusal("theta must lie in [0, 1]")

        self.d_in = d_in
        self.hidden = hidden
        self.n_out = n_out
        self.theta = float(theta)
        self.seed = int(seed)

        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        self.W1 = rng.standard_normal((hidden, d_in)) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.standard_normal((n_out, hidden)) * math.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_out, dtype=np.float64)

    def n_params(self) -> int:

        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def flops_per_inference(self) -> int:
        return inference_flops(self.d_in, self.hidden, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops(n_train_frames, epochs, self.d_in, self.hidden, self.n_out)

    def parameter_digest(self) -> str:
        payload = {
            "schema": COUNT_GATE_SCHEMA,
            "w1_sha256": hashlib.sha256(self.W1.astype("<f8").tobytes()).hexdigest(),
            "b1_sha256": hashlib.sha256(self.b1.astype("<f8").tobytes()).hexdigest(),
            "w2_sha256": hashlib.sha256(self.W2.astype("<f8").tobytes()).hexdigest(),
            "b2_sha256": hashlib.sha256(self.b2.astype("<f8").tobytes()).hexdigest(),
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_out": self.n_out,
            "n_params": self.n_params(),
        }
        return canonical_sha256(payload)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_pre = x @ self.W1.T + self.b1
        hidden = np.maximum(0.0, hidden_pre)
        logit = hidden @ self.W2.T + self.b2
        p_reestimate = _sigmoid(logit[:, 0])
        return p_reestimate, hidden_pre, hidden

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
    ) -> CountTrainingReport:

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise CountGateRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise CountGateRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise CountGateRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise CountGateRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise CountGateRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise CountGateRefusal("ponder_lambda must be nonnegative")

        n = x.shape[0]
        eps = 1e-12
        loss_history: list[float] = []
        final_p = np.zeros(n, dtype=np.float64)
        for _ in range(epochs):
            p_reestimate, hidden_pre, hidden = self._forward(x)
            final_p = p_reestimate
            clipped = np.clip(p_reestimate, eps, 1.0 - eps)
            bce = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
            ponder = ponder_lambda * p_reestimate.mean()
            loss_history.append(float(bce + ponder))
            d_logit = ((p_reestimate - y) + ponder_lambda * p_reestimate * (1.0 - p_reestimate)) / n
            d_logit = d_logit[:, None]
            grad_w2 = d_logit.T @ hidden
            grad_b2 = d_logit.sum(axis=0)
            d_hidden = d_logit @ self.W2
            d_hidden_pre = d_hidden * (hidden_pre > 0.0)
            grad_w1 = d_hidden_pre.T @ x
            grad_b1 = d_hidden_pre.sum(axis=0)
            self.W2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1

        return CountTrainingReport(
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            loss_history=tuple(loss_history),
            final_reestimate_rate=float(final_p.mean()),
            c_train_flops=self.training_flops(n, epochs),
        )


def voc_targets_from_count_track(count_track, window: int = COUNT_VOC_WINDOW) -> np.ndarray:

    track = [int(v) for v in count_track]
    n_frames = len(track)
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise CountGateRefusal("window must be a nonnegative integer")
    changes = [t for t in range(1, n_frames) if track[t] != track[t - 1]]
    targets = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        if any(abs(t - c) <= window for c in changes):
            targets[t] = 1.0
    return targets
