from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields

import numpy as np

from mop.evidence import canonical_sha256
from mop.seeding import derive_seed32

from .count_featurizer import D_CFEAT, N_CHANNELS, N_MEL

COUNT_GATE_SCHEMA = "mop-starss23-count-gate/v1"

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
DEFAULT_EPOCHS = 8
DEFAULT_TRAIN_FRAMES = 54_000
TRAIN_STEP_FACTOR = 3
COUNT_VOC_WINDOW = 1  # a re-estimation is valuable within +/- 1 frame of a count change

_POS_BLOCK = N_MEL * N_CHANNELS  # 128


class CountGateRefusal(ValueError):
    pass


N_PARAMS = D_IN * HIDDEN + HIDDEN + HIDDEN * N_OUT + N_OUT
if N_PARAMS > PARAM_CEILING:
    raise CountGateRefusal(f"gate trainable parameters {N_PARAMS} exceed the {PARAM_CEILING} ceiling")


def inference_flops(d_in: int = D_IN, hidden: int = HIDDEN, n_out: int = N_OUT) -> int:
    return 2 * d_in * hidden + 2 * hidden + 2 * hidden * n_out + n_out


def training_flops(
    n_train_frames: int = DEFAULT_TRAIN_FRAMES,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN,
    hidden: int = HIDDEN,
    n_out: int = N_OUT,
) -> int:
    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise CountGateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise CountGateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops(d_in, hidden, n_out)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


FLOPS_PER_INFERENCE = inference_flops()
C_TRAIN_ANCHOR = training_flops()


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


class CountGate:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != D_IN:
            raise CountGateRefusal(f"gate input must be shape (N, {D_IN})")
        return self._forward(x)[0]

    def _assemble(self, features: np.ndarray, state: CountOnlineState) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (N_CFEAT,):
            raise CountGateRefusal(f"features must be a length {N_CFEAT} vector")
        if not isinstance(state, CountOnlineState):
            raise CountGateRefusal("state must be a CountOnlineState")
        return np.concatenate([features, state.to_vector()])

    def infer(self, features: np.ndarray, state: CountOnlineState) -> float:
        x = self._assemble(features, state)
        return float(self.predict_proba(x[None, :])[0])

    _SEED_NAMESPACE = "mop.beds.starss23.count_gate.init"

    def __init__(
        self,
        *,
        seed: int = 0,
        theta: float = DEFAULT_THETA,
    ) -> None:
        state_bytes = CountOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise CountGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "count gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise CountGateRefusal("theta must lie in [0, 1]")

        self.theta = float(theta)
        self.seed = int(seed)

        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        self.W1 = rng.standard_normal((HIDDEN, D_IN)) * math.sqrt(2.0 / D_IN)
        self.b1 = np.zeros(HIDDEN, dtype=np.float64)
        self.W2 = rng.standard_normal((N_OUT, HIDDEN)) * math.sqrt(2.0 / HIDDEN)
        self.b2 = np.zeros(N_OUT, dtype=np.float64)

    def n_params(self) -> int:
        return N_PARAMS

    def parameter_digest(self) -> str:
        payload = {
            "schema": COUNT_GATE_SCHEMA,
            "w1_sha256": hashlib.sha256(self.W1.astype("<f8").tobytes()).hexdigest(),
            "b1_sha256": hashlib.sha256(self.b1.astype("<f8").tobytes()).hexdigest(),
            "w2_sha256": hashlib.sha256(self.W2.astype("<f8").tobytes()).hexdigest(),
            "b2_sha256": hashlib.sha256(self.b2.astype("<f8").tobytes()).hexdigest(),
            "d_in": D_IN,
            "hidden": HIDDEN,
            "n_out": N_OUT,
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
    ) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != D_IN:
            raise CountGateRefusal(f"training inputs must be shape (N, {D_IN})")
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
        final_p = np.zeros(n, dtype=np.float64)
        for _ in range(epochs):
            p_reestimate, hidden_pre, hidden = self._forward(x)
            final_p = p_reestimate
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

        return float(final_p.mean())


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
