"""STARSS23 counting bed, gate-architecture reproduction: the re-authored trained gate.

This is a net-new, ADDITIVE component for the gate-architecture bias reproduction. It varies exactly ONE
axis of the sealed counting bed: the shape of the only trained module. Where the sealed ``count_gate.py``
gate is a single-hidden-layer MLP ``264 -> 12 -> 1`` (3193 trainable parameters), this reproduction uses a
genuinely deeper, narrower TWO-hidden-layer MLP ``264 -> 8 -> 4 -> 1`` (2161 trainable parameters, still
hard-capped at 4096). Nothing else moves: the input contract (256 frozen features plus the same 8
self-derived online scalars = 264 inputs), the training objective (binary cross-entropy plus the same
firing-rate ponder penalty on the same value-of-computation targets), the decision rule (re-estimate iff
``p_reestimate >= theta``), and the deterministic ``derive_seed32`` seeding are all held identical and are
reused BY IMPORT from the sealed gate so the reproduction cannot smuggle in a second change.

The point is adversarial: if the sealed counting bed's first positive mechanics-ok signal (the trained gate
reaching strictly lower coasted-count-MAE than rate-matched-random at matched budget) is a fingerprint of
that one gate shape, a different-shape gate trained the same way on the same data will fail to beat the same
control. If instead the mechanism is architecture-robust, the different-shape gate survives too. This module
runs no experiment and makes no claim; it only defines the alternative gate and its own analytic FLOP and
parameter cost anchors so the matched-budget harness can charge them in full-lifecycle accounting.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

# Held FIXED, reused by import from the sealed gate: the online-state contract (the 8 self-derived scalars),
# the value-of-computation target rule, the input width, and the training hyperparameter anchors. None of
# these is the gate architecture, so changing the architecture leaves every one of them byte-identical.
from .count_gate import (
    COUNT_VOC_WINDOW,
    D_IN,
    N_CFEAT,
    N_CONLINE,
    CountOnlineState,
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
    # Re-exported held-fixed pieces so the producer imports the reproduction surface from one place.
    "CountOnlineState",
    "voc_targets_from_count_track",
    "COUNT_VOC_WINDOW",
]

COUNT_REPRO_GATE_ARCH_GATE_SCHEMA = "mop-starss23-count-repro-gate-arch-gate/v1"
REPRO_AXIS = "gate_arch"

# The varied axis: a two-hidden-layer 264 -> 8 -> 4 -> 1 MLP. A genuine depth change from the single-hidden
# -layer sealed gate, still comfortably under the 4096-parameter ceiling.
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
    """Raised when the re-authored two-layer gate would breach its parameter, state, or interface contract."""


# ---------------------------------------------------------------------------
# Analytic cost model for the two-hidden-layer gate. Pure functions, host-independent integers.
# ---------------------------------------------------------------------------


def param_count_two_layer(
    d_in: int = D_IN_GATE_ARCH,
    hidden1: int = HIDDEN1,
    hidden2: int = HIDDEN2,
    n_out: int = N_OUT,
) -> int:
    """Trainable parameters of a ``d_in -> hidden1 -> hidden2 -> n_out`` MLP: W1+b1 + W2+b2 + W3+b3.

    For 264 -> 8 -> 4 -> 1 this is 264*8+8 + 8*4+4 + 4*1+1 = 2161, under the 4096 ceiling.
    """

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
    """Per-frame forward FLOPs: three matmuls (multiply-add), three bias adds, two ReLU layers.

    For 264 -> 8 -> 4 -> 1 this is (2*264*8 + 8 + 8) + (2*8*4 + 4 + 4) + (2*4*1 + 1) = 4321 FLOPs.
    """

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
    """Amortized training cost C_train: epochs * frames * TRAIN_STEP_FACTOR * per-frame inference FLOPs.

    The step factor (one forward plus a backward pass costed at about twice the forward) is reused
    unchanged from the sealed gate, so only the per-frame inference FLOPs differ from the original.
    """

    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise CountReproGateArchRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise CountReproGateArchRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops_two_layer(
        d_in, hidden1, hidden2, n_out
    )


# The per-frame inference FLOP anchor for this architecture: 4321 FLOPs for 264 -> 8 -> 4 -> 1.
FLOPS_PER_INFERENCE_GATE_ARCH = inference_flops_two_layer()


@dataclass
class CountReproGateArchTrainingReport:
    """Deterministic record of one fit call. Compute is the analytic C_train, not a measured count."""

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


class CountReproGateArchGate:
    """The re-authored two-hidden-layer re-estimation gate: 264 -> 8 -> 4 -> 1 with ReLU.

    Identical interface, objective, and decision rule to the sealed ``CountGate``; only the multilayer
    perceptron topology differs. Construction hard-asserts the 4096-parameter ceiling and the few-KB
    online-state ceiling. Weights are seeded deterministically through ``derive_seed32`` so paired seeds
    reproduce byte for byte. The forward path takes the 256 frozen features plus the 8 self-derived online
    scalars and never a label.
    """

    _SEED_NAMESPACE = "mop.beds.starss23.count_repro_gate_arch.init"

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
        # He initialization per layer, matching the sealed gate's variance convention exactly.
        self.W1 = rng.standard_normal((hidden1, d_in)) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden1, dtype=np.float64)
        self.W2 = rng.standard_normal((hidden2, hidden1)) * math.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float64)
        self.W3 = rng.standard_normal((n_out, hidden2)) * math.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_out, dtype=np.float64)

    def n_params(self) -> int:
        """Exact trainable-parameter count from the live weight arrays."""

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

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise CountReproGateArchRefusal(f"gate input must be shape (N, {self.d_in})")
        p_reestimate, _, _, _, _ = self._forward(x)
        return p_reestimate

    def _assemble(self, features: np.ndarray, state: CountOnlineState) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (N_CFEAT_GATE_ARCH,):
            raise CountReproGateArchRefusal(f"features must be a length {N_CFEAT_GATE_ARCH} vector")
        if not isinstance(state, CountOnlineState):
            raise CountReproGateArchRefusal("state must be a CountOnlineState")
        return np.concatenate([features, state.to_vector()])

    def infer(self, features: np.ndarray, state: CountOnlineState) -> float:
        """Online re-estimation probability for one frame. Takes features and self-state, never a label."""

        x = self._assemble(features, state)
        return float(self.predict_proba(x[None, :])[0])

    def decide(
        self, features: np.ndarray, state: CountOnlineState, theta: float | None = None
    ) -> tuple[bool, float]:
        """Return ``(reestimate, p_reestimate)`` for one frame; re-estimates iff p_reestimate >= theta."""

        threshold = self.theta if theta is None else float(theta)
        p_reestimate = self.infer(features, state)
        return (p_reestimate >= threshold, p_reestimate)

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
    ) -> CountReproGateArchTrainingReport:
        """Fit the two-layer gate on train-room value-of-computation targets with a ponder penalty.

        Identical objective to the sealed gate: binary cross-entropy plus ``ponder_lambda`` times the mean
        re-estimation probability, full-batch deterministic gradient descent. Only the backward pass carries
        the extra hidden layer. ``x`` is (N, d_in); ``voc_targets`` is (N,) in {0, 1}. The reported compute
        is the analytic C_train for this architecture.
        """

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
            # Same output-logit gradient as the sealed gate (BCE plus ponder on the sigmoid probability).
            d_logit = ((p_reestimate - y) + ponder_lambda * p_reestimate * (1.0 - p_reestimate)) / n
            d_logit = d_logit[:, None]
            # Output layer W3 (n_out, hidden2).
            grad_w3 = d_logit.T @ h2
            grad_b3 = d_logit.sum(axis=0)
            # Second hidden layer W2 (hidden2, hidden1).
            d_h2 = d_logit @ self.W3
            d_z2 = d_h2 * (z2 > 0.0)
            grad_w2 = d_z2.T @ h1
            grad_b2 = d_z2.sum(axis=0)
            # First hidden layer W1 (hidden1, d_in).
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
