
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .doa_featurizer import D_FEAT_DOA
from .gate import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PONDER_LAMBDA, TRAIN_STEP_FACTOR, _sigmoid

DOA_GATE_SCHEMA = "mop-starss23-doa-gate/v1"

N_CFEAT_DOA = D_FEAT_DOA  # 256 frozen spatial-flux features
N_ONLINE_DOA = 8  # self-derived online scalars
D_IN_DOA = N_CFEAT_DOA + N_ONLINE_DOA  # 264 gate inputs, shared by both architectures

PARAM_CEILING = 4096
STATE_CEILING_BYTES = 8192

ARCH_A_ID = "arch_a_264_12_1"
ARCH_B_ID = "arch_b_264_6_6_1"
ARCHITECTURES: tuple[str, ...] = (ARCH_A_ID, ARCH_B_ID)

DEFAULT_THETA = 0.5
PHASE_HORIZON = 10.0  # frames per positional-clock cycle (1 second at 100 ms frames)
EMA_DECAY = 0.1

_DIFFUSENESS_FLUX_INDICES = tuple(range(3, D_FEAT_DOA, 4))


class DoaGateRefusal(ValueError):
    pass




@dataclass(frozen=True, slots=True)
class DoaOnlineState:

    n_frames: float = 0.0
    n_reestimates: float = 0.0
    last_reestimate_frame: float = -1.0
    reestimate_rate_ema: float = 0.0
    energy_ema: float = 0.0  # EMA of mean(|features|)
    flux_peak_ema: float = 0.0  # EMA of max(features)
    diffuseness_flux_ema: float = 0.0  # EMA of mean over the 64 diffuseness-flux components
    last_p_reestimate: float = 0.0

    @classmethod
    def initial(cls) -> DoaOnlineState:
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
                math.tanh(self.flux_peak_ema),
                math.tanh(self.diffuseness_flux_ema),
                self.last_p_reestimate,
                phase,
                cumulative_fraction,
            ],
            dtype=np.float64,
        )

    def update(self, features: np.ndarray, p_reestimate: float, reestimated: bool) -> DoaOnlineState:

        features = np.asarray(features, dtype=np.float64)
        energy = float(np.abs(features).mean()) if features.size else 0.0
        flux_peak = float(features.max()) if features.size else 0.0
        if features.size:
            diffuseness_component = features[list(_DIFFUSENESS_FLUX_INDICES)]
            diffuseness_flux = float(diffuseness_component.mean()) if diffuseness_component.size else 0.0
        else:
            diffuseness_flux = 0.0
        did = 1.0 if reestimated else 0.0
        return DoaOnlineState(
            n_frames=self.n_frames + 1.0,
            n_reestimates=self.n_reestimates + did,
            last_reestimate_frame=self.n_frames if reestimated else self.last_reestimate_frame,
            reestimate_rate_ema=(1.0 - EMA_DECAY) * self.reestimate_rate_ema + EMA_DECAY * did,
            energy_ema=(1.0 - EMA_DECAY) * self.energy_ema + EMA_DECAY * energy,
            flux_peak_ema=(1.0 - EMA_DECAY) * self.flux_peak_ema + EMA_DECAY * flux_peak,
            diffuseness_flux_ema=(1.0 - EMA_DECAY) * self.diffuseness_flux_ema + EMA_DECAY * diffuseness_flux,
            last_p_reestimate=float(p_reestimate),
        )


@runtime_checkable
class DoaGateProtocol(Protocol):

    architecture: str

    def n_params(self) -> int: ...

    def flops_per_inference(self) -> int: ...

    def training_flops(self, n_train_frames: int, epochs: int) -> int: ...

    def infer(self, features: np.ndarray, state: DoaOnlineState) -> float: ...

    def decide(
        self, features: np.ndarray, state: DoaOnlineState, theta: float | None = None
    ) -> tuple[bool, float]: ...

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int,
        learning_rate: float,
        ponder_lambda: float,
    ) -> Any: ...

    def parameter_digest(self) -> str: ...


@dataclass
class DoaTrainingReport:

    architecture: str
    epochs: int
    n_train_frames: int
    learning_rate: float
    ponder_lambda: float
    loss_history: tuple[float, ...]
    final_reestimate_rate: float
    c_train_flops: int

    def payload(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "epochs": self.epochs,
            "n_train_frames": self.n_train_frames,
            "learning_rate": self.learning_rate,
            "ponder_lambda": self.ponder_lambda,
            "loss_history": [float(value) for value in self.loss_history],
            "final_reestimate_rate": self.final_reestimate_rate,
            "c_train_flops": self.c_train_flops,
        }


def _assemble(features: np.ndarray, state: DoaOnlineState, d_in: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.shape != (N_CFEAT_DOA,):
        raise DoaGateRefusal(f"features must be a length {N_CFEAT_DOA} vector")
    if not isinstance(state, DoaOnlineState):
        raise DoaGateRefusal("state must be a DoaOnlineState")
    x = np.concatenate([features, state.to_vector()])
    if x.shape != (d_in,):
        raise DoaGateRefusal(f"assembled gate input must be length {d_in}")
    return x


class _DoaGateInterface:

    __slots__ = ()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise DoaGateRefusal(f"gate input must be shape (N, {self.d_in})")
        return self._forward(x)[0]

    def infer(self, features: np.ndarray, state: DoaOnlineState) -> float:
        x = _assemble(features, state, self.d_in)
        return float(self.predict_proba(x[None, :])[0])

    def decide(
        self, features: np.ndarray, state: DoaOnlineState, theta: float | None = None
    ) -> tuple[bool, float]:
        threshold = self.theta if theta is None else float(theta)
        probability = self.infer(features, state)
        return (probability >= threshold, probability)



HIDDEN_A = 12
N_OUT = 1


def param_count_arch_a(d_in: int = D_IN_DOA, hidden: int = HIDDEN_A, n_out: int = N_OUT) -> int:
    return d_in * hidden + hidden + hidden * n_out + n_out


def inference_flops_arch_a(d_in: int = D_IN_DOA, hidden: int = HIDDEN_A, n_out: int = N_OUT) -> int:
    return 2 * d_in * hidden + hidden + hidden + 2 * hidden * n_out + n_out


def training_flops_arch_a(
    n_train_frames: int,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN_DOA,
    hidden: int = HIDDEN_A,
    n_out: int = N_OUT,
) -> int:
    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise DoaGateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise DoaGateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops_arch_a(d_in, hidden, n_out)


FLOPS_PER_INFERENCE_ARCH_A = inference_flops_arch_a()


class DoaGateArchA(_DoaGateInterface):

    _SEED_NAMESPACE = "mop.beds.starss23.doa_gate.arch_a.init"
    architecture = ARCH_A_ID

    def __init__(
        self,
        *,
        seed: int = 0,
        d_in: int = D_IN_DOA,
        hidden: int = HIDDEN_A,
        n_out: int = N_OUT,
        theta: float = DEFAULT_THETA,
    ) -> None:
        if hidden <= 0 or d_in <= 0 or n_out <= 0:
            raise DoaGateRefusal("gate dimensions must be positive")
        n_params = param_count_arch_a(d_in, hidden, n_out)
        if n_params > PARAM_CEILING:
            raise DoaGateRefusal(f"arch_a trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling")
        assert n_params <= PARAM_CEILING, "doa gate arch_a parameter ceiling breached"
        state_bytes = DoaOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise DoaGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "doa gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise DoaGateRefusal("theta must lie in [0, 1]")

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
        return inference_flops_arch_a(self.d_in, self.hidden, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops_arch_a(n_train_frames, epochs, self.d_in, self.hidden, self.n_out)

    def parameter_digest(self) -> str:
        payload = {
            "schema": DOA_GATE_SCHEMA,
            "architecture": self.architecture,
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
    ) -> DoaTrainingReport:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise DoaGateRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise DoaGateRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise DoaGateRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise DoaGateRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise DoaGateRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise DoaGateRefusal("ponder_lambda must be nonnegative")

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

        return DoaTrainingReport(
            architecture=self.architecture,
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            loss_history=tuple(loss_history),
            final_reestimate_rate=float(final_p.mean()),
            c_train_flops=self.training_flops(n, epochs),
        )



HIDDEN_B1 = 6
HIDDEN_B2 = 6


def param_count_arch_b(
    d_in: int = D_IN_DOA, hidden1: int = HIDDEN_B1, hidden2: int = HIDDEN_B2, n_out: int = N_OUT
) -> int:
    return d_in * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 * n_out + n_out


def inference_flops_arch_b(
    d_in: int = D_IN_DOA, hidden1: int = HIDDEN_B1, hidden2: int = HIDDEN_B2, n_out: int = N_OUT
) -> int:
    layer1 = 2 * d_in * hidden1 + hidden1 + hidden1
    layer2 = 2 * hidden1 * hidden2 + hidden2 + hidden2
    layer3 = 2 * hidden2 * n_out + n_out
    return layer1 + layer2 + layer3


def training_flops_arch_b(
    n_train_frames: int,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN_DOA,
    hidden1: int = HIDDEN_B1,
    hidden2: int = HIDDEN_B2,
    n_out: int = N_OUT,
) -> int:
    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise DoaGateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise DoaGateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops_arch_b(d_in, hidden1, hidden2, n_out)


FLOPS_PER_INFERENCE_ARCH_B = inference_flops_arch_b()


class DoaGateArchB(_DoaGateInterface):

    _SEED_NAMESPACE = "mop.beds.starss23.doa_gate.arch_b.init"
    architecture = ARCH_B_ID

    def __init__(
        self,
        *,
        seed: int = 0,
        d_in: int = D_IN_DOA,
        hidden1: int = HIDDEN_B1,
        hidden2: int = HIDDEN_B2,
        n_out: int = N_OUT,
        theta: float = DEFAULT_THETA,
    ) -> None:
        if hidden1 <= 0 or hidden2 <= 0 or d_in <= 0 or n_out <= 0:
            raise DoaGateRefusal("gate dimensions must be positive")
        n_params = param_count_arch_b(d_in, hidden1, hidden2, n_out)
        if n_params > PARAM_CEILING:
            raise DoaGateRefusal(f"arch_b trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling")
        assert n_params <= PARAM_CEILING, "doa gate arch_b parameter ceiling breached"
        state_bytes = DoaOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise DoaGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "doa gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise DoaGateRefusal("theta must lie in [0, 1]")

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
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size)

    def flops_per_inference(self) -> int:
        return inference_flops_arch_b(self.d_in, self.hidden1, self.hidden2, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops_arch_b(
            n_train_frames, epochs, self.d_in, self.hidden1, self.hidden2, self.n_out
        )

    def parameter_digest(self) -> str:
        payload = {
            "schema": DOA_GATE_SCHEMA,
            "architecture": self.architecture,
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
    ) -> DoaTrainingReport:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise DoaGateRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise DoaGateRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise DoaGateRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise DoaGateRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise DoaGateRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise DoaGateRefusal("ponder_lambda must be nonnegative")

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

        return DoaTrainingReport(
            architecture=self.architecture,
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            loss_history=tuple(loss_history),
            final_reestimate_rate=float(final_p.mean()),
            c_train_flops=self.training_flops(n, epochs),
        )


GATE_CLASSES: dict[str, type] = {ARCH_A_ID: DoaGateArchA, ARCH_B_ID: DoaGateArchB}

C_TRAIN_ANCHOR_ARCH_A = training_flops_arch_a(54_000)
C_TRAIN_ANCHOR_ARCH_B = training_flops_arch_b(54_000)


def build_gate(
    architecture: str, *, seed: int = 0, theta: float = DEFAULT_THETA
) -> DoaGateArchA | DoaGateArchB:

    if architecture not in GATE_CLASSES:
        raise DoaGateRefusal(f"unknown architecture {architecture!r}, expected one of {ARCHITECTURES}")
    return GATE_CLASSES[architecture](seed=seed, theta=theta)
