
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.escs.accounting import WorkVector
from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .featurizer import D_FEAT
from .gate import (
    D_IN,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PONDER_LAMBDA,
    DEFAULT_THETA,
    HIDDEN,
    N_OUT,
    PARAM_CEILING,
    STATE_CEILING_BYTES,
    GateRefusal,
    OnlineState,
    _sigmoid,
    inference_flops,
    param_count,
    training_flops,
)
from .schema import COLLAR_FRAMES

# The spacing window is anchored to the sealed referee collar, not a tuned knob: two fires closer than
# COLLAR_FRAMES can match at most one distinct onset under the one-to-one matcher, so co-firing inside the
# collar is the waste the regularizer prices. Fixed in code before the run.
DEFAULT_SPACING_WINDOW = COLLAR_FRAMES  # 2 frames on the frozen 100 ms grid (the plus/minus 200 ms collar)

# Default regularizer strength. The real run selects the strength per seed on the val rooms over a
# preregistered grid (see scripts/run_starss23_escs_bed_diversity_reg.py); this default only anchors the
# gate's standalone behavior and its cost accounting.
DEFAULT_DIVERSITY_LAMBDA = 1.0

# Small floor on the firing-energy normalizer so the scale-invariant ratio is finite when all p go to 0.
_RATIO_EPS = 1e-12


class DiversityRegRefusal(GateRefusal):
    pass


def spacing_kernel(window: int) -> np.ndarray:

    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise DiversityRegRefusal("spacing window must be a positive integer")
    raw = np.array([window + 1 - j for j in range(1, window + 1)], dtype=np.float64)
    total = float(raw.sum())
    if total <= 0.0:
        raise DiversityRegRefusal("spacing kernel must have positive mass")
    return raw / total


def _clip_index(segment_lengths: Sequence[int] | None, n: int) -> np.ndarray:

    if segment_lengths is None:
        return np.zeros(n, dtype=np.int64)
    lengths = [int(length) for length in segment_lengths]
    if any(isinstance(length, bool) or length <= 0 for length in lengths):
        raise DiversityRegRefusal("segment_lengths must be positive integers")
    if sum(lengths) != n:
        raise DiversityRegRefusal("segment_lengths must sum to the number of training frames")
    return np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)


def _neighbor_probability_sum(
    p_fire: np.ndarray, clip_index: np.ndarray, kernel: np.ndarray
) -> np.ndarray:

    n = p_fire.shape[0]
    neighbor = np.zeros(n, dtype=np.float64)
    for offset, weight in enumerate(kernel, start=1):
        if offset >= n:
            break
        same_clip = clip_index[:-offset] == clip_index[offset:]
        contribution = float(weight) * same_clip
        neighbor[:-offset] += contribution * p_fire[offset:]
        neighbor[offset:] += contribution * p_fire[:-offset]
    return neighbor


@dataclass
class DiversityRegTrainingReport:

    epochs: int
    n_train_frames: int
    learning_rate: float
    ponder_lambda: float
    diversity_lambda: float
    spacing_window: int
    loss_history: tuple[float, ...]
    spacing_penalty_history: tuple[float, ...]
    final_firing_rate: float
    final_spacing_penalty: float
    c_train_flops: int

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_frames": self.n_train_frames,
            "learning_rate": self.learning_rate,
            "ponder_lambda": self.ponder_lambda,
            "diversity_lambda": self.diversity_lambda,
            "spacing_window": self.spacing_window,
            "loss_history": [float(value) for value in self.loss_history],
            "spacing_penalty_history": [float(value) for value in self.spacing_penalty_history],
            "final_firing_rate": self.final_firing_rate,
            "final_spacing_penalty": self.final_spacing_penalty,
            "c_train_flops": self.c_train_flops,
        }


class DiversityRegGate:

    # Deliberately the SAME namespace as the committed gate so paired-seed weights match byte for byte.
    _SEED_NAMESPACE = "mop.beds.starss23.gate.init"

    def __init__(
        self,
        *,
        seed: int = 0,
        d_in: int = D_IN,
        hidden: int = HIDDEN,
        n_out: int = N_OUT,
        theta: float = DEFAULT_THETA,
        diversity_lambda: float = DEFAULT_DIVERSITY_LAMBDA,
        spacing_window: int = DEFAULT_SPACING_WINDOW,
    ) -> None:
        if hidden <= 0 or d_in <= 0 or n_out <= 0:
            raise DiversityRegRefusal("gate dimensions must be positive")
        n_params = param_count(d_in, hidden, n_out)
        if n_params > PARAM_CEILING:
            raise DiversityRegRefusal(
                f"gate trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        # Hard invariant, kept as a bare assert too so a mis-edit trips even without the guard.
        assert n_params <= PARAM_CEILING, "gate parameter ceiling breached"
        state_bytes = OnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise DiversityRegRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise DiversityRegRefusal("theta must lie in [0, 1]")
        if isinstance(diversity_lambda, bool) or not isinstance(diversity_lambda, (int, float)):
            raise DiversityRegRefusal("diversity_lambda must be a real number")
        if not math.isfinite(float(diversity_lambda)) or float(diversity_lambda) < 0.0:
            raise DiversityRegRefusal("diversity_lambda must be a nonnegative finite number")

        self.d_in = d_in
        self.hidden = hidden
        self.n_out = n_out
        self.theta = float(theta)
        self.seed = int(seed)
        self.diversity_lambda = float(diversity_lambda)
        self.spacing_window = int(spacing_window)
        self._kernel = spacing_kernel(self.spacing_window)

        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        # He-style init for the ReLU layer, small output layer. All float64 for reproducibility. Identical
        # recipe to the committed gate so paired-seed initial weights match byte for byte.
        self.W1 = rng.standard_normal((hidden, d_in)) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.standard_normal((n_out, hidden)) * math.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_out, dtype=np.float64)

    # -- geometry and cost (identical formulas to the committed gate) --------

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def flops_per_inference(self) -> int:
        return inference_flops(self.d_in, self.hidden, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops(n_train_frames, epochs, self.d_in, self.hidden, self.n_out)

    def infer_work_vector(self, n_frames: int) -> WorkVector:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise DiversityRegRefusal("n_frames must be a nonnegative integer")
        return WorkVector(dispatch_and_exploration=self.flops_per_inference() * n_frames)

    def train_work_vector(self, n_train_frames: int, epochs: int) -> WorkVector:
        return WorkVector(learning=self.training_flops(n_train_frames, epochs))

    def parameter_digest(self) -> str:
        payload = {
            "w1_sha256": _sha_of(self.W1),
            "b1_sha256": _sha_of(self.b1),
            "w2_sha256": _sha_of(self.W2),
            "b2_sha256": _sha_of(self.b2),
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_out": self.n_out,
            "n_params": self.n_params(),
        }
        return canonical_sha256(payload)

    # -- forward path (no label ever enters here; byte-identical to the committed gate) --------

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_pre = x @ self.W1.T + self.b1
        hidden = np.maximum(0.0, hidden_pre)
        logit = hidden @ self.W2.T + self.b2
        p_fire = _sigmoid(logit[:, 0])
        return p_fire, hidden_pre, hidden

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise DiversityRegRefusal(f"gate input must be shape (N, {self.d_in})")
        p_fire, _, _ = self._forward(x)
        return p_fire

    def _assemble(self, features: np.ndarray, state: OnlineState) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (D_FEAT,):
            raise DiversityRegRefusal(f"features must be a length {D_FEAT} vector")
        if not isinstance(state, OnlineState):
            raise DiversityRegRefusal("state must be an OnlineState")
        return np.concatenate([features, state.to_vector()])

    def infer(self, features: np.ndarray, state: OnlineState) -> float:

        x = self._assemble(features, state)
        return float(self.predict_proba(x[None, :])[0])

    def fire(
        self, features: np.ndarray, state: OnlineState, theta: float | None = None
    ) -> tuple[bool, float]:
        threshold = self.theta if theta is None else float(theta)
        p_fire = self.infer(features, state)
        return (p_fire >= threshold, p_fire)

    # -- training (offline; committed loss plus a within-clip spacing regularizer) --------

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
        segment_lengths: Sequence[int] | None = None,
    ) -> DiversityRegTrainingReport:

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise DiversityRegRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise DiversityRegRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise DiversityRegRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise DiversityRegRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise DiversityRegRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise DiversityRegRefusal("ponder_lambda must be nonnegative")

        n = x.shape[0]
        clip_index = _clip_index(segment_lengths, n)
        eps = 1e-12
        loss_history: list[float] = []
        spacing_history: list[float] = []
        final_p = np.zeros(n, dtype=np.float64)
        final_spacing = 0.0
        for _ in range(epochs):
            p_fire, hidden_pre, hidden = self._forward(x)
            final_p = p_fire
            clipped = np.clip(p_fire, eps, 1.0 - eps)
            bce = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
            ponder = ponder_lambda * p_fire.mean()

            # Scale-invariant within-clip spacing regularizer. Skip entirely at lambda == 0 so the update
            # and the loss are byte-identical to the committed gate. The BCE and ponder gradients are means
            # (divided by n); the ratio penalty is an intensive scalar whose gradient is added directly.
            base_grad = (p_fire - y) + ponder_lambda * p_fire * (1.0 - p_fire)
            if self.diversity_lambda > 0.0:
                neighbor = _neighbor_probability_sum(p_fire, clip_index, self._kernel)
                a_adjacency = float((p_fire * neighbor).sum())  # p^T K p, within-clip adjacency energy
                q_energy = float((p_fire * p_fire).sum()) + _RATIO_EPS  # total firing energy (scale)
                rho = a_adjacency / q_energy  # fraction of firing energy that is adjacent; scale-invariant
                spacing_penalty = self.diversity_lambda * rho
                # d(rho)/d(p_a) = 2 * neighbor_a / Q - 2 * A * p_a / Q^2 (A = p^T K p is quadratic in p).
                d_rho_d_p = 2.0 * neighbor / q_energy - 2.0 * a_adjacency * p_fire / (q_energy * q_energy)
                spacing_grad_logit = self.diversity_lambda * d_rho_d_p * p_fire * (1.0 - p_fire)
                d_logit = base_grad / n + spacing_grad_logit
            else:
                spacing_penalty = 0.0
                d_logit = base_grad / n
            final_spacing = spacing_penalty
            loss_history.append(float(bce + ponder + spacing_penalty))
            spacing_history.append(float(spacing_penalty))

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

        return DiversityRegTrainingReport(
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            diversity_lambda=float(self.diversity_lambda),
            spacing_window=int(self.spacing_window),
            loss_history=tuple(loss_history),
            spacing_penalty_history=tuple(spacing_history),
            final_firing_rate=float(final_p.mean()),
            final_spacing_penalty=float(final_spacing),
            c_train_flops=self.training_flops(n, epochs),
        )


def _sha_of(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(array.astype("<f8").tobytes()).hexdigest()
