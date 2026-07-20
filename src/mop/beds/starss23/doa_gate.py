
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


def _network_param_count(*dimensions: int) -> int:
    return sum(
        d_in * d_out + d_out
        for d_in, d_out in zip(dimensions[:-1], dimensions[1:], strict=True)
    )


def _network_inference_flops(*dimensions: int) -> int:
    last = len(dimensions) - 2
    return sum(
        2 * d_in * d_out + d_out + (d_out if index < last else 0)
        for index, (d_in, d_out) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True))
    )


def _network_training_flops(n_train_frames: int, epochs: int, *dimensions: int) -> int:
    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise DoaGateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise DoaGateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * _network_inference_flops(*dimensions)


class _DoaGateInterface:
    def _init_network(self, seed: int, theta: float, *dimensions: int) -> None:
        if any(dimension <= 0 for dimension in dimensions):
            raise DoaGateRefusal("gate dimensions must be positive")
        n_params = _network_param_count(*dimensions)
        if n_params > PARAM_CEILING:
            raise DoaGateRefusal(
                f"{self.architecture} trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        state_bytes = DoaOnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise DoaGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        if not 0.0 <= theta <= 1.0:
            raise DoaGateRefusal("theta must lie in [0, 1]")
        self.d_in, self.n_out = dimensions[0], dimensions[-1]
        self.theta, self.seed, self._dimensions = float(theta), int(seed), dimensions
        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        for index, (d_in, d_out) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True), 1
        ):
            setattr(self, f"W{index}", rng.standard_normal((d_out, d_in)) * math.sqrt(2.0 / d_in))
            setattr(self, f"b{index}", np.zeros(d_out, dtype=np.float64))

    def _layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (getattr(self, f"W{index}"), getattr(self, f"b{index}"))
            for index in range(1, len(self._dimensions))
        ]

    def n_params(self) -> int:
        return sum(weight.size + bias.size for weight, bias in self._layers())

    def parameter_digest(self) -> str:
        payload = {"schema": DOA_GATE_SCHEMA, "architecture": self.architecture}
        for index, (weight, bias) in enumerate(self._layers(), 1):
            payload[f"w{index}_sha256"] = hashlib.sha256(weight.astype("<f8").tobytes()).hexdigest()
            payload[f"b{index}_sha256"] = hashlib.sha256(bias.astype("<f8").tobytes()).hexdigest()
        payload.update(self._digest_dimensions())
        payload["n_params"] = self.n_params()
        return canonical_sha256(payload)

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        layers = self._layers()
        activations = [x]
        hidden_pre = []
        for weight, bias in layers[:-1]:
            hidden_pre.append(activations[-1] @ weight.T + bias)
            activations.append(np.maximum(0.0, hidden_pre[-1]))
        output_weight, output_bias = layers[-1]
        probabilities = _sigmoid((activations[-1] @ output_weight.T + output_bias)[:, 0])
        return probabilities, tuple(hidden_pre), tuple(activations)

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
        probability = self.infer(features, state)
        return (probability >= (self.theta if theta is None else float(theta)), probability)

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
    ) -> DoaTrainingReport:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(voc_targets, dtype=np.float64)
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
        loss_history: list[float] = []
        final_p = np.zeros(n, dtype=np.float64)
        for _ in range(epochs):
            final_p, hidden_pre, activations = self._forward(x)
            clipped = np.clip(final_p, 1e-12, 1.0 - 1e-12)
            bce = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
            loss_history.append(float(bce + ponder_lambda * final_p.mean()))
            delta = ((final_p - y) + ponder_lambda * final_p * (1.0 - final_p))[:, None] / n
            layers = self._layers()
            grad_weights: list[np.ndarray] = [np.empty(0)] * len(layers)
            grad_biases: list[np.ndarray] = [np.empty(0)] * len(layers)
            for index in range(len(layers) - 1, -1, -1):
                grad_weights[index] = delta.T @ activations[index]
                grad_biases[index] = delta.sum(axis=0)
                if index:
                    delta = (delta @ layers[index][0]) * (hidden_pre[index - 1] > 0.0)
            for index in range(len(layers) - 1, -1, -1):
                np.subtract(
                    layers[index][0], learning_rate * grad_weights[index], out=layers[index][0]
                )
                np.subtract(
                    layers[index][1], learning_rate * grad_biases[index], out=layers[index][1]
                )

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


HIDDEN_A = 12
N_OUT = 1


def param_count_arch_a(d_in: int = D_IN_DOA, hidden: int = HIDDEN_A, n_out: int = N_OUT) -> int:
    return _network_param_count(d_in, hidden, n_out)


def inference_flops_arch_a(d_in: int = D_IN_DOA, hidden: int = HIDDEN_A, n_out: int = N_OUT) -> int:
    return _network_inference_flops(d_in, hidden, n_out)


def training_flops_arch_a(
    n_train_frames: int,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN_DOA,
    hidden: int = HIDDEN_A,
    n_out: int = N_OUT,
) -> int:
    return _network_training_flops(n_train_frames, epochs, d_in, hidden, n_out)


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
        self.hidden = hidden
        self._init_network(seed, theta, d_in, hidden, n_out)

    def flops_per_inference(self) -> int:
        return inference_flops_arch_a(self.d_in, self.hidden, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops_arch_a(n_train_frames, epochs, self.d_in, self.hidden, self.n_out)

    def _digest_dimensions(self) -> dict[str, int]:
        return {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_out": self.n_out,
        }


HIDDEN_B1 = 6
HIDDEN_B2 = 6


def param_count_arch_b(
    d_in: int = D_IN_DOA, hidden1: int = HIDDEN_B1, hidden2: int = HIDDEN_B2, n_out: int = N_OUT
) -> int:
    return _network_param_count(d_in, hidden1, hidden2, n_out)


def inference_flops_arch_b(
    d_in: int = D_IN_DOA, hidden1: int = HIDDEN_B1, hidden2: int = HIDDEN_B2, n_out: int = N_OUT
) -> int:
    return _network_inference_flops(d_in, hidden1, hidden2, n_out)


def training_flops_arch_b(
    n_train_frames: int,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN_DOA,
    hidden1: int = HIDDEN_B1,
    hidden2: int = HIDDEN_B2,
    n_out: int = N_OUT,
) -> int:
    return _network_training_flops(n_train_frames, epochs, d_in, hidden1, hidden2, n_out)


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
        self.hidden1, self.hidden2 = hidden1, hidden2
        self._init_network(seed, theta, d_in, hidden1, hidden2, n_out)

    def flops_per_inference(self) -> int:
        return inference_flops_arch_b(self.d_in, self.hidden1, self.hidden2, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops_arch_b(
            n_train_frames, epochs, self.d_in, self.hidden1, self.hidden2, self.n_out
        )

    def _digest_dimensions(self) -> dict[str, int]:
        return {
            "d_in": self.d_in,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "n_out": self.n_out,
        }


GATE_CLASSES: dict[str, type] = {ARCH_A_ID: DoaGateArchA, ARCH_B_ID: DoaGateArchB}

C_TRAIN_ANCHOR_ARCH_A = training_flops_arch_a(54_000)
C_TRAIN_ANCHOR_ARCH_B = training_flops_arch_b(54_000)


def build_gate(
    architecture: str, *, seed: int = 0, theta: float = DEFAULT_THETA
) -> DoaGateArchA | DoaGateArchB:

    if architecture not in GATE_CLASSES:
        raise DoaGateRefusal(f"unknown architecture {architecture!r}, expected one of {ARCHITECTURES}")
    return GATE_CLASSES[architecture](seed=seed, theta=theta)
