
from __future__ import annotations

import hashlib
import math

import numpy as np

from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .count_gate import (
    COUNT_VOC_WINDOW,
    D_IN,
    N_CFEAT,
    N_CONLINE,
    CountGateInterface,
    CountOnlineState,
    CountTrainingReport,
    voc_targets_from_count_track,
)
from .gate import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PONDER_LAMBDA,
    TRAIN_STEP_FACTOR,
    _sigmoid,
)

__all__ = [
    "COUNT_REPRO_GATE_ARCH_GATE_SCHEMA",
    "REPRO_AXIS",
    "HIDDEN1",
    "HIDDEN2",
    "PARAM_CEILING",
    "FLOPS_PER_INFERENCE_GATE_ARCH",
    "CountReproGateArchRefusal",
    "CountReproGateArchGate",
    "param_count_two_layer",
    "inference_flops_two_layer",
    "training_flops_two_layer",
    "CountOnlineState",
    "voc_targets_from_count_track",
    "COUNT_VOC_WINDOW",
]

COUNT_REPRO_GATE_ARCH_GATE_SCHEMA = "mop-starss23-count-repro-gate-arch-gate/v1"
REPRO_AXIS = "gate_arch"

N_CFEAT_GATE_ARCH = N_CFEAT  # 256 frozen count features (held identical)
N_CONLINE_GATE_ARCH = N_CONLINE  # 8 self-derived online scalars (held identical)
D_IN_GATE_ARCH = D_IN  # 264 gate inputs (held identical)
HIDDEN1 = 8
HIDDEN2 = 4
N_OUT = 1

PARAM_CEILING = 4096
STATE_CEILING_BYTES = 8192

DEFAULT_THETA = 0.5


class CountReproGateArchRefusal(ValueError):
    pass




def param_count_two_layer(
    d_in: int = D_IN_GATE_ARCH,
    hidden1: int = HIDDEN1,
    hidden2: int = HIDDEN2,
    n_out: int = N_OUT,
) -> int:

    return (
        d_in * hidden1
        + hidden1
        + hidden1 * hidden2
        + hidden2
        + hidden2 * n_out
        + n_out
    )


def inference_flops_two_layer(
    d_in: int = D_IN_GATE_ARCH,
    hidden1: int = HIDDEN1,
    hidden2: int = HIDDEN2,
    n_out: int = N_OUT,
) -> int:

    layer1 = 2 * d_in * hidden1 + hidden1 + hidden1
    layer2 = 2 * hidden1 * hidden2 + hidden2 + hidden2
    layer3 = 2 * hidden2 * n_out + n_out
    return layer1 + layer2 + layer3


def training_flops_two_layer(
    n_train_frames: int,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN_GATE_ARCH,
    hidden1: int = HIDDEN1,
    hidden2: int = HIDDEN2,
    n_out: int = N_OUT,
) -> int:

    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise CountReproGateArchRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise CountReproGateArchRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops_two_layer(
        d_in, hidden1, hidden2, n_out
    )


FLOPS_PER_INFERENCE_GATE_ARCH = inference_flops_two_layer()


class CountReproGateArchTrainingReport(CountTrainingReport):
    pass


class CountReproGateArchGate(CountGateInterface):

    _SEED_NAMESPACE = "mop.beds.starss23.count_repro_gate_arch.init"
    _feature_dim = N_CFEAT_GATE_ARCH
    _refusal = CountReproGateArchRefusal

    def __init__(
        self,
        *,
        seed: int = 0,
        d_in: int = D_IN_GATE_ARCH,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        n_out: int = N_OUT,
        theta: float = DEFAULT_THETA,
    ) -> None:
        if hidden1 <= 0 or hidden2 <= 0 or d_in <= 0 or n_out <= 0:
            raise CountReproGateArchRefusal("gate dimensions must be positive")
        n_params = param_count_two_layer(d_in, hidden1, hidden2, n_out)
        if n_params > PARAM_CEILING:
            raise CountReproGateArchRefusal(
                f"gate trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        assert n_params <= PARAM_CEILING, "count repro gate parameter ceiling breached"
        state_bytes = CountOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise CountReproGateArchRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "count repro gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise CountReproGateArchRefusal("theta must lie in [0, 1]")

        self.d_in = d_in
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.n_out = n_out
        self.theta = float(theta)
        self.seed = int(seed)

        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        self.W1 = rng.standard_normal((hidden1, d_in)) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden1, dtype=np.float64)
        self.W2 = rng.standard_normal((hidden2, hidden1)) * math.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float64)
        self.W3 = rng.standard_normal((n_out, hidden2)) * math.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_out, dtype=np.float64)

    def n_params(self) -> int:

        return int(
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
            + self.W3.size
            + self.b3.size
        )

    def flops_per_inference(self) -> int:
        return inference_flops_two_layer(self.d_in, self.hidden1, self.hidden2, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops_two_layer(
            n_train_frames, epochs, self.d_in, self.hidden1, self.hidden2, self.n_out
        )

    def parameter_digest(self) -> str:
        payload = {
            "schema": COUNT_REPRO_GATE_ARCH_GATE_SCHEMA,
            "w1_sha256": hashlib.sha256(self.W1.astype("<f8").tobytes()).hexdigest(),
            "b1_sha256": hashlib.sha256(self.b1.astype("<f8").tobytes()).hexdigest(),
            "w2_sha256": hashlib.sha256(self.W2.astype("<f8").tobytes()).hexdigest(),
            "b2_sha256": hashlib.sha256(self.b2.astype("<f8").tobytes()).hexdigest(),
            "w3_sha256": hashlib.sha256(self.W3.astype("<f8").tobytes()).hexdigest(),
            "b3_sha256": hashlib.sha256(self.b3.astype("<f8").tobytes()).hexdigest(),
            "d_in": self.d_in,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "n_out": self.n_out,
            "n_params": self.n_params(),
        }
        return canonical_sha256(payload)

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.W1.T + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(0.0, z2)
        logit = h2 @ self.W3.T + self.b3
        p_reestimate = _sigmoid(logit[:, 0])
        return p_reestimate, z1, h1, z2, h2

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
    ) -> CountReproGateArchTrainingReport:

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise CountReproGateArchRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise CountReproGateArchRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise CountReproGateArchRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise CountReproGateArchRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise CountReproGateArchRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise CountReproGateArchRefusal("ponder_lambda must be nonnegative")

        n = x.shape[0]
        eps = 1e-12
        loss_history: list[float] = []
        final_p = np.zeros(n, dtype=np.float64)
        for _ in range(epochs):
            p_reestimate, z1, h1, z2, h2 = self._forward(x)
            final_p = p_reestimate
            clipped = np.clip(p_reestimate, eps, 1.0 - eps)
            bce = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
            ponder = ponder_lambda * p_reestimate.mean()
            loss_history.append(float(bce + ponder))
            d_logit = ((p_reestimate - y) + ponder_lambda * p_reestimate * (1.0 - p_reestimate)) / n
            d_logit = d_logit[:, None]
            grad_w3 = d_logit.T @ h2
            grad_b3 = d_logit.sum(axis=0)
            d_h2 = d_logit @ self.W3
            d_z2 = d_h2 * (z2 > 0.0)
            grad_w2 = d_z2.T @ h1
            grad_b2 = d_z2.sum(axis=0)
            d_h1 = d_z2 @ self.W2
            d_z1 = d_h1 * (z1 > 0.0)
            grad_w1 = d_z1.T @ x
            grad_b1 = d_z1.sum(axis=0)
            self.W3 -= learning_rate * grad_w3
            self.b3 -= learning_rate * grad_b3
            self.W2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1

        return CountReproGateArchTrainingReport(
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            loss_history=tuple(loss_history),
            final_reestimate_rate=float(final_p.mean()),
            c_train_flops=self.training_flops(n, epochs),
        )
