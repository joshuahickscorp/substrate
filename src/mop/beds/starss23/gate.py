"""Component 3: the candidate online value-of-computation event-formation gate.

This is the ONLY trained module in the STARSS23 bed. Per 100 ms frame it consumes the frozen
featurizer's 256 features plus 8 online state scalars (264 inputs) and emits a firing probability
p_fire = sigmoid(MLP(z)); it fires iff p_fire >= theta. The multilayer perceptron is 264 -> 12 -> 1,
so its trainable parameter count is 3193, hard-capped at 4096, and its per-stream online state is a
handful of float64 registers, hard-capped at a few kilobytes. Both caps are asserted in code.

The gate predicts the value of spending downstream compute, that is, whether firing here recovers a
matched referee true positive that a non-fire would miss. It is trained on train-room value-of-
computation targets with a firing-rate ponder penalty. It never sees ground truth, class, or
direction-of-arrival online: infer() and the online state carry features and self-derived running
statistics only. Its inference cost and its amortized training cost C_train are both exposed so the
matched-budget harness can charge them in full-lifecycle FLOP accounting.

Compute is charged analytically, not measured, so the ledger is reproducible across hosts.

Claim scope: deterministic programmatic mechanics only; no capability, learning, or efficiency claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from mop.escs.accounting import WorkVector
from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .featurizer import D_FEAT

# ---------------------------------------------------------------------------
# Frozen gate geometry. The candidate is a tiny MLP; every number here is fixed by the spec.
# ---------------------------------------------------------------------------

N_ONLINE = 8  # self-derived online scalars appended to the 256 frozen features
D_IN = D_FEAT + N_ONLINE  # 264 gate inputs
HIDDEN = 12  # hidden units; 264 -> 12 -> 1 keeps params <= 4096
N_OUT = 1  # a single firing logit

PARAM_CEILING = 4096  # hard cap on trainable parameters; owned-substrate constraint
STATE_CEILING_BYTES = 8192  # hard cap on per-stream online state; few-KB constraint

DEFAULT_THETA = 0.5  # default firing threshold; the harness tunes theta on val rooms

# Full-lifecycle training-cost anchors. C_train = epochs * train_frames * step_factor * infer_flops.
DEFAULT_EPOCHS = 8
DEFAULT_TRAIN_FRAMES = 54_000  # 90 train-room clips of 60 s at 10 frames per second
TRAIN_STEP_FACTOR = 3  # one forward plus a backward pass costed at about twice the forward

# Online-state update constants (deterministic, no learned quantity).
EMA_DECAY = 0.1  # exponential-moving-average step for the running online statistics
PHASE_HORIZON = 10.0  # frames per positional-clock cycle (1 second at 100 ms frames)

# Training defaults for the real run. Tests pass explicit values; these anchor C_train only.
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_PONDER_LAMBDA = 0.02

# Analytic per-frame inference FLOP budget (docs/ESCS_DEEP_RESEARCH.md immediate design implications).
FLOPS_MATMUL1 = 2 * D_IN * HIDDEN  # 6336 multiply-add for the first layer
FLOPS_BIAS1 = HIDDEN  # 12 first-layer bias adds
FLOPS_ACT1 = HIDDEN  # 12 ReLU activations
FLOPS_MATMUL2 = 2 * HIDDEN * N_OUT  # 24 multiply-add for the output layer
FLOPS_BIAS2 = N_OUT  # 1 output bias add
FLOPS_PER_INFERENCE = (
    FLOPS_MATMUL1 + FLOPS_BIAS1 + FLOPS_ACT1 + FLOPS_MATMUL2 + FLOPS_BIAS2
)  # 6385


class GateRefusal(ValueError):
    """Raised when the candidate gate would breach its parameter, state, or interface contract."""


# ---------------------------------------------------------------------------
# Analytic cost model. Pure functions so the harness can cost an arm without building a gate.
# ---------------------------------------------------------------------------


def param_count(d_in: int = D_IN, hidden: int = HIDDEN, n_out: int = N_OUT) -> int:
    """Trainable parameters of a d_in -> hidden -> n_out MLP: W1 + b1 + W2 + b2."""

    return d_in * hidden + hidden + hidden * n_out + n_out


def inference_flops(d_in: int = D_IN, hidden: int = HIDDEN, n_out: int = N_OUT) -> int:
    """Per-frame forward FLOPs: two matmuls (multiply-add), two bias adds, one ReLU layer."""

    return 2 * d_in * hidden + hidden + hidden + 2 * hidden * n_out + n_out


def training_flops(
    n_train_frames: int = DEFAULT_TRAIN_FRAMES,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN,
    hidden: int = HIDDEN,
    n_out: int = N_OUT,
) -> int:
    """Amortized training cost C_train charged in full: epochs * frames * step_factor * infer FLOPs."""

    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise GateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise GateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * inference_flops(d_in, hidden, n_out)


# The C_train anchor for the real run: 8 * 54000 * 3 * 6385 = 8_274_960_000 (about 8.27e9 FLOPs).
C_TRAIN_ANCHOR = training_flops()


def break_even_frames(
    per_query_saving_flops: float,
    *,
    n_train_frames: int = DEFAULT_TRAIN_FRAMES,
    epochs: int = DEFAULT_EPOCHS,
    d_in: int = D_IN,
    hidden: int = HIDDEN,
    n_out: int = N_OUT,
) -> float:
    """Break-even query count N* = C_train / per-query saving. Below N* the gate has not paid back."""

    if not isinstance(per_query_saving_flops, (int, float)) or per_query_saving_flops <= 0:
        raise GateRefusal("per_query_saving_flops must be a positive number")
    return training_flops(n_train_frames, epochs, d_in, hidden, n_out) / float(per_query_saving_flops)


# ---------------------------------------------------------------------------
# Online state. Eight self-derived running scalars, no ground truth, no direction of arrival.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnlineState:
    """Per-stream online state fed to the gate. Carries only self-derived running statistics.

    The eight registers below are updated causally from the features and the gate's own firing
    decisions. None of them is a label, a class, or a direction-of-arrival: the gate is blind to
    ground truth online by construction. ``to_vector`` returns the eight bounded scalars the MLP sees.
    """

    n_frames: float = 0.0
    n_fires: float = 0.0
    last_fire_frame: float = -1.0
    firing_rate_ema: float = 0.0
    energy_ema: float = 0.0
    energy_absdev_ema: float = 0.0
    flux_peak_ema: float = 0.0
    last_p_fire: float = 0.0

    @classmethod
    def initial(cls) -> OnlineState:
        return cls()

    @classmethod
    def state_bytes(cls) -> int:
        """Byte footprint of the per-stream state: one float64 per register. Hard-capped few-KB."""

        return len(fields(cls)) * 8

    def to_vector(self) -> np.ndarray:
        """Return the eight bounded online scalars the gate consumes, as float64."""

        recency = math.tanh((self.n_frames - self.last_fire_frame) / PHASE_HORIZON)
        phase = math.fmod(self.n_frames, PHASE_HORIZON) / PHASE_HORIZON
        cumulative_fraction = self.n_fires / max(1.0, self.n_frames)
        return np.array(
            [
                recency,
                self.firing_rate_ema,
                math.tanh(self.energy_ema),
                math.tanh(self.energy_absdev_ema),
                math.tanh(self.flux_peak_ema),
                self.last_p_fire,
                phase,
                cumulative_fraction,
            ],
            dtype=np.float64,
        )

    def update(self, features: np.ndarray, p_fire: float, fired: bool) -> OnlineState:
        """Return the next state after observing one frame and the gate's own decision on it.

        Deterministic. Uses only the features and the firing decision, never a label. The energy and
        peak summaries are self-derived from the frozen features; the exponential moving averages
        decay at a fixed rate.
        """

        features = np.asarray(features, dtype=np.float64)
        energy = float(features.mean())
        flux_peak = float(features.max()) if features.size else 0.0
        fired_value = 1.0 if fired else 0.0
        return OnlineState(
            n_frames=self.n_frames + 1.0,
            n_fires=self.n_fires + fired_value,
            last_fire_frame=self.n_frames if fired else self.last_fire_frame,
            firing_rate_ema=(1.0 - EMA_DECAY) * self.firing_rate_ema + EMA_DECAY * fired_value,
            energy_ema=(1.0 - EMA_DECAY) * self.energy_ema + EMA_DECAY * energy,
            energy_absdev_ema=(1.0 - EMA_DECAY) * self.energy_absdev_ema
            + EMA_DECAY * abs(energy - self.energy_ema),
            flux_peak_ema=(1.0 - EMA_DECAY) * self.flux_peak_ema + EMA_DECAY * flux_peak,
            last_p_fire=float(p_fire),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic sigmoid."""

    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


# ---------------------------------------------------------------------------
# The candidate gate.
# ---------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """Deterministic record of one fit call. Compute is the analytic C_train, not a measured count."""

    epochs: int
    n_train_frames: int
    learning_rate: float
    ponder_lambda: float
    loss_history: tuple[float, ...]
    final_firing_rate: float
    c_train_flops: int

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_frames": self.n_train_frames,
            "learning_rate": self.learning_rate,
            "ponder_lambda": self.ponder_lambda,
            "loss_history": [float(value) for value in self.loss_history],
            "final_firing_rate": self.final_firing_rate,
            "c_train_flops": self.c_train_flops,
        }


class CandidateGate:
    """The only trained module: an online value-of-computation firing trigger.

    Construction hard-asserts the parameter ceiling (<= 4096) and the online-state ceiling (few-KB).
    Weights are seeded deterministically through ``derive_seed32`` so paired seeds reproduce byte for
    byte. The forward path takes features plus online scalars and never a label.
    """

    _SEED_NAMESPACE = "mop.beds.starss23.gate.init"

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
            raise GateRefusal("gate dimensions must be positive")
        n_params = param_count(d_in, hidden, n_out)
        if n_params > PARAM_CEILING:
            raise GateRefusal(
                f"gate trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        # Hard invariant, kept as a bare assert as well so a mis-edit trips even without the guard.
        assert n_params <= PARAM_CEILING, "gate parameter ceiling breached"
        state_bytes = OnlineState.state_bytes()
        if state_bytes > STATE_CEILING_BYTES:
            raise GateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise GateRefusal("theta must lie in [0, 1]")

        self.d_in = d_in
        self.hidden = hidden
        self.n_out = n_out
        self.theta = float(theta)
        self.seed = int(seed)

        rng = np.random.default_rng(derive_seed32(self.seed, self._SEED_NAMESPACE))
        # He-style init for the ReLU layer, small output layer. All float64 for reproducibility.
        self.W1 = rng.standard_normal((hidden, d_in)) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.standard_normal((n_out, hidden)) * math.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_out, dtype=np.float64)

    # -- geometry and cost ---------------------------------------------------

    def n_params(self) -> int:
        """Exact trainable-parameter count from the live weight arrays."""

        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def flops_per_inference(self) -> int:
        return inference_flops(self.d_in, self.hidden, self.n_out)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops(n_train_frames, epochs, self.d_in, self.hidden, self.n_out)

    def infer_work_vector(self, n_frames: int) -> WorkVector:
        """Charge inference to dispatch and exploration: the gate decides whether to spend compute."""

        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise GateRefusal("n_frames must be a nonnegative integer")
        return WorkVector(dispatch_and_exploration=self.flops_per_inference() * n_frames)

    def train_work_vector(self, n_train_frames: int, epochs: int) -> WorkVector:
        """Charge the amortized training cost to the learning bucket, in full."""

        return WorkVector(learning=self.training_flops(n_train_frames, epochs))

    def parameter_digest(self) -> str:
        """Digest of the current weight bytes, so paired-seed reproductions can be checked."""

        payload = {
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

    # -- forward path (no label ever enters here) ----------------------------

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_fire, hidden_pre_activation, hidden_activation) for a batch of inputs."""

        hidden_pre = x @ self.W1.T + self.b1
        hidden = np.maximum(0.0, hidden_pre)
        logit = hidden @ self.W2.T + self.b2
        p_fire = _sigmoid(logit[:, 0])
        return p_fire, hidden_pre, hidden

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Firing probability for a batch of 264-dim inputs. Shape (N, d_in) -> (N,)."""

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise GateRefusal(f"gate input must be shape (N, {self.d_in})")
        p_fire, _, _ = self._forward(x)
        return p_fire

    def _assemble(self, features: np.ndarray, state: OnlineState) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (D_FEAT,):
            raise GateRefusal(f"features must be a length {D_FEAT} vector")
        if not isinstance(state, OnlineState):
            raise GateRefusal("state must be an OnlineState")
        return np.concatenate([features, state.to_vector()])

    def infer(self, features: np.ndarray, state: OnlineState) -> float:
        """Online firing probability for one frame. Takes features and self-state, never a label."""

        x = self._assemble(features, state)
        return float(self.predict_proba(x[None, :])[0])

    def fire(
        self, features: np.ndarray, state: OnlineState, theta: float | None = None
    ) -> tuple[bool, float]:
        """Return (fired, p_fire) for one frame; fires iff p_fire >= theta."""

        threshold = self.theta if theta is None else float(theta)
        p_fire = self.infer(features, state)
        return (p_fire >= threshold, p_fire)

    # -- training (offline, value-of-computation targets, ponder penalty) ----

    def fit(
        self,
        x: np.ndarray,
        voc_targets: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        ponder_lambda: float = DEFAULT_PONDER_LAMBDA,
    ) -> TrainingReport:
        """Fit the gate on train-room value-of-computation targets with a firing-rate ponder penalty.

        ``x`` is (N, d_in): each row is a frame's 256 features plus its 8 online scalars, assembled by
        the harness on the train rooms. ``voc_targets`` is (N,) in {0, 1}: 1 where firing recovers a
        matched referee true positive a non-fire would miss. The loss is binary cross-entropy plus
        ``ponder_lambda`` times the mean firing probability, which prices ponder so the gate does not
        fire everywhere. Full-batch deterministic gradient descent. The reported compute is the
        analytic C_train, charged in full regardless of the measured wall time.
        """

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(voc_targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise GateRefusal(f"training inputs must be shape (N, {self.d_in})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise GateRefusal("voc_targets must be a length-N vector aligned to the inputs")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise GateRefusal("voc_targets must be binary {0, 1}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise GateRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise GateRefusal("learning_rate must be positive")
        if ponder_lambda < 0.0:
            raise GateRefusal("ponder_lambda must be nonnegative")

        n = x.shape[0]
        eps = 1e-12
        loss_history: list[float] = []
        final_p = np.zeros(n, dtype=np.float64)
        for _ in range(epochs):
            p_fire, hidden_pre, hidden = self._forward(x)
            final_p = p_fire
            clipped = np.clip(p_fire, eps, 1.0 - eps)
            bce = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
            ponder = ponder_lambda * p_fire.mean()
            loss_history.append(float(bce + ponder))
            # Gradient of BCE plus ponder with respect to the output logit.
            d_logit = ((p_fire - y) + ponder_lambda * p_fire * (1.0 - p_fire)) / n
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

        return TrainingReport(
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            ponder_lambda=float(ponder_lambda),
            loss_history=tuple(loss_history),
            final_firing_rate=float(final_p.mean()),
            c_train_flops=self.training_flops(n, epochs),
        )
