
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.escs.accounting import WorkVector
from mop.seeding import derive_seed32
from mop.substrate.events import canonical_sha256

from .featurizer import D_FEAT
from .gate import PARAM_CEILING, STATE_CEILING_BYTES


LP_PROJ_DIM = 32  # dimension of the fixed random projection R: 256 -> 32
LP_TARGET_DIM = 16  # dimension of the fixed random RND target f*: 32 -> 16
LP_N_PARAMS = LP_PROJ_DIM * LP_TARGET_DIM

EMA_DECAY = 0.1  # causal EMA step for the running online error baseline
ONLINE_LR = 0.05  # per-frame online gradient step for the working predictor Pw
EPS_NORM = 1e-9  # floor for the projected-direction normalizer
EPS_REDUCIBLE = 1e-6  # floor in the reducible-fraction denominator so tiny e_base gives ~0 reducible
SCORE_SCALE = 1.0  # monotone squash scale; rank-invariant, so it never changes the firing set

DEFAULT_THETA = 0.5  # default firing threshold; the harness tunes theta on the val rooms

DEFAULT_EPOCHS = 8
DEFAULT_TRAIN_FRAMES = 54_000
TRAIN_STEP_FACTOR = 3  # one forward plus a backward pass costed at about twice the forward

_SEED_PROJECTION = "mop.beds.starss23.gate_lp.projection"
_SEED_TARGET = "mop.beds.starss23.gate_lp.target"
_SEED_PREDICTOR = "mop.beds.starss23.gate_lp.predictor_init"


class LPGateRefusal(ValueError):
    pass




def param_count(proj_dim: int = LP_PROJ_DIM, target_dim: int = LP_TARGET_DIM) -> int:

    return proj_dim * target_dim


def _forward_flops(d_feat: int, proj_dim: int, target_dim: int) -> int:

    project = 2 * proj_dim * d_feat
    normalize = 3 * proj_dim
    target = 2 * target_dim * proj_dim
    predict = 2 * target_dim * proj_dim
    error = 2 * target_dim
    return project + normalize + target + predict + error


def inference_flops(
    d_feat: int = D_FEAT, proj_dim: int = LP_PROJ_DIM, target_dim: int = LP_TARGET_DIM
) -> int:

    forward = _forward_flops(d_feat, proj_dim, target_dim)
    base_predict = 2 * target_dim * proj_dim + 2 * target_dim  # second predictor forward and its error
    online_update = 3 * target_dim * proj_dim  # residual, outer-product gradient, and the weight step
    return forward + base_predict + online_update


def training_flops(
    n_train_frames: int = DEFAULT_TRAIN_FRAMES,
    epochs: int = DEFAULT_EPOCHS,
    d_feat: int = D_FEAT,
    proj_dim: int = LP_PROJ_DIM,
    target_dim: int = LP_TARGET_DIM,
) -> int:

    if isinstance(n_train_frames, bool) or not isinstance(n_train_frames, int) or n_train_frames < 0:
        raise LPGateRefusal("n_train_frames must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise LPGateRefusal("epochs must be a nonnegative integer")
    return epochs * n_train_frames * TRAIN_STEP_FACTOR * _forward_flops(d_feat, proj_dim, target_dim)


C_TRAIN_ANCHOR = training_flops()




def _softplus(x: float) -> float:

    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))




@dataclass(frozen=True, slots=True)
class LPOnlineState:

    pw: np.ndarray
    error_ema: float
    n_frames: float
    initialized: bool = False

    @classmethod
    def initial(cls, p0: np.ndarray) -> LPOnlineState:
        return cls(pw=np.array(p0, dtype=np.float64, copy=True), error_ema=0.0, n_frames=0.0)

    @staticmethod
    def state_bytes(proj_dim: int = LP_PROJ_DIM, target_dim: int = LP_TARGET_DIM) -> int:

        return proj_dim * target_dim * 8 + 3 * 8




@dataclass
class LPTrainingReport:

    epochs: int
    n_train_frames: int
    learning_rate: float
    loss_history: tuple[float, ...]
    final_train_error: float
    c_train_flops: int

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_frames": self.n_train_frames,
            "learning_rate": self.learning_rate,
            "loss_history": [float(value) for value in self.loss_history],
            "final_train_error": self.final_train_error,
            "c_train_flops": self.c_train_flops,
        }


class LearningProgressGate:

    def __init__(
        self,
        *,
        seed: int = 0,
        d_feat: int = D_FEAT,
        proj_dim: int = LP_PROJ_DIM,
        target_dim: int = LP_TARGET_DIM,
        theta: float = DEFAULT_THETA,
    ) -> None:
        if d_feat <= 0 or proj_dim <= 0 or target_dim <= 0:
            raise LPGateRefusal("gate dimensions must be positive")
        n_params = param_count(proj_dim, target_dim)
        if n_params > PARAM_CEILING:
            raise LPGateRefusal(
                f"gate trainable parameters {n_params} exceed the {PARAM_CEILING} ceiling"
            )
        assert n_params <= PARAM_CEILING, "gate parameter ceiling breached"
        state_bytes = LPOnlineState.state_bytes(proj_dim, target_dim)
        if state_bytes > STATE_CEILING_BYTES:
            raise LPGateRefusal(
                f"gate online state {state_bytes} bytes exceed the {STATE_CEILING_BYTES} byte ceiling"
            )
        assert state_bytes <= STATE_CEILING_BYTES, "gate state ceiling breached"
        if not 0.0 <= theta <= 1.0:
            raise LPGateRefusal("theta must lie in [0, 1]")

        self.d_feat = d_feat
        self.proj_dim = proj_dim
        self.target_dim = target_dim
        self.theta = float(theta)
        self.seed = int(seed)

        rng_r = np.random.default_rng(derive_seed32(self.seed, _SEED_PROJECTION))
        self.R = rng_r.standard_normal((proj_dim, d_feat)) / math.sqrt(d_feat)
        rng_t = np.random.default_rng(derive_seed32(self.seed, _SEED_TARGET))
        self.target = rng_t.standard_normal((target_dim, proj_dim)) / math.sqrt(proj_dim)

        rng_p = np.random.default_rng(derive_seed32(self.seed, _SEED_PREDICTOR))
        self.P0 = rng_p.standard_normal((target_dim, proj_dim)) * math.sqrt(1.0 / proj_dim)


    def n_params(self) -> int:

        return int(self.P0.size)

    def state_bytes(self) -> int:
        return LPOnlineState.state_bytes(self.proj_dim, self.target_dim)

    def flops_per_inference(self) -> int:
        return inference_flops(self.d_feat, self.proj_dim, self.target_dim)

    def training_flops(self, n_train_frames: int, epochs: int) -> int:
        return training_flops(n_train_frames, epochs, self.d_feat, self.proj_dim, self.target_dim)

    def infer_work_vector(self, n_frames: int) -> WorkVector:

        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise LPGateRefusal("n_frames must be a nonnegative integer")
        return WorkVector(dispatch_and_exploration=self.flops_per_inference() * n_frames)

    def train_work_vector(self, n_train_frames: int, epochs: int) -> WorkVector:

        return WorkVector(learning=self.training_flops(n_train_frames, epochs))

    def parameter_digest(self) -> str:

        payload = {
            "r_sha256": hashlib.sha256(self.R.astype("<f8").tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(self.target.astype("<f8").tobytes()).hexdigest(),
            "p0_sha256": hashlib.sha256(self.P0.astype("<f8").tobytes()).hexdigest(),
            "d_feat": self.d_feat,
            "proj_dim": self.proj_dim,
            "target_dim": self.target_dim,
            "n_params": self.n_params(),
        }
        return canonical_sha256(payload)


    def _code(self, features: np.ndarray) -> np.ndarray:

        z = self.R @ np.asarray(features, dtype=np.float64)
        norm = math.sqrt(float(z @ z)) + EPS_NORM
        return z / norm

    def _codes(self, features: np.ndarray) -> np.ndarray:

        z = np.asarray(features, dtype=np.float64) @ self.R.T
        norms = np.sqrt((z * z).sum(axis=1)) + EPS_NORM
        return z / norms[:, None]

    def initial_state(self) -> LPOnlineState:

        return LPOnlineState.initial(self.P0)


    def _score_from_code(
        self, code: np.ndarray, target_vec: np.ndarray, e_base: float, state: LPOnlineState
    ) -> float:

        pred_w = state.pw @ code
        residual = pred_w - target_vec
        e_online = float(residual @ residual)
        baseline = state.error_ema if state.initialized else e_online
        surprise = _softplus((e_online - baseline) / SCORE_SCALE)
        reducible_num = e_base - e_online
        reducible = reducible_num / (e_base + EPS_REDUCIBLE) if reducible_num > 0.0 else 0.0
        score = surprise * reducible
        return math.tanh(score / SCORE_SCALE)

    def _next_state(
        self, code: np.ndarray, target_vec: np.ndarray, state: LPOnlineState
    ) -> LPOnlineState:

        pred_w = state.pw @ code
        residual = pred_w - target_vec
        e_online = float(residual @ residual)
        grad = 2.0 * np.outer(residual, code)
        pw_next = state.pw - ONLINE_LR * grad
        if state.initialized:
            error_ema = (1.0 - EMA_DECAY) * state.error_ema + EMA_DECAY * e_online
        else:
            error_ema = e_online
        return LPOnlineState(
            pw=pw_next, error_ema=error_ema, n_frames=state.n_frames + 1.0, initialized=True
        )

    def infer(self, features: np.ndarray, state: LPOnlineState) -> float:

        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.d_feat,):
            raise LPGateRefusal(f"features must be a length {self.d_feat} vector")
        if not isinstance(state, LPOnlineState):
            raise LPGateRefusal("state must be an LPOnlineState")
        code = self._code(features)
        target_vec = self.target @ code
        pred0 = self.P0 @ code
        base_residual = pred0 - target_vec
        e_base = float(base_residual @ base_residual)
        return self._score_from_code(code, target_vec, e_base, state)

    def update(self, features: np.ndarray, state: LPOnlineState) -> LPOnlineState:

        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.d_feat,):
            raise LPGateRefusal(f"features must be a length {self.d_feat} vector")
        if not isinstance(state, LPOnlineState):
            raise LPGateRefusal("state must be an LPOnlineState")
        code = self._code(features)
        target_vec = self.target @ code
        return self._next_state(code, target_vec, state)

    def fire(
        self, features: np.ndarray, state: LPOnlineState, theta: float | None = None
    ) -> tuple[bool, float]:

        threshold = self.theta if theta is None else float(theta)
        p_fire = self.infer(features, state)
        return (p_fire >= threshold, p_fire)

    def causal_scores(self, features: np.ndarray) -> np.ndarray:

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.d_feat:
            raise LPGateRefusal(f"features must be shape (n_frames, {self.d_feat})")
        n = features.shape[0]
        codes = self._codes(features)  # (n, proj_dim)
        targets = codes @ self.target.T  # (n, target_dim)
        base_pred = codes @ self.P0.T  # (n, target_dim)
        base_res = base_pred - targets
        e_base = (base_res * base_res).sum(axis=1)  # (n,)

        probs = np.empty(n, dtype=np.float64)
        state = self.initial_state()
        for i in range(n):
            probs[i] = self._score_from_code(codes[i], targets[i], float(e_base[i]), state)
            state = self._next_state(codes[i], targets[i], state)
        return probs

    def causal_fires(self, features: np.ndarray, theta: float) -> tuple[list[int], np.ndarray]:

        probs = self.causal_scores(features)
        fires = [int(i) for i in np.nonzero(probs >= float(theta))[0]]
        return fires, probs


    def fit(
        self,
        features: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = ONLINE_LR,
    ) -> LPTrainingReport:

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.d_feat:
            raise LPGateRefusal(f"training features must be shape (N, {self.d_feat})")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise LPGateRefusal("epochs must be a positive integer")
        if learning_rate <= 0.0:
            raise LPGateRefusal("learning_rate must be positive")

        codes = self._codes(features)  # (N, proj_dim)
        targets = codes @ self.target.T  # (N, target_dim)
        n = codes.shape[0]
        loss_history: list[float] = []
        final_error = 0.0
        for _ in range(epochs):
            pred = codes @ self.P0.T  # (N, target_dim)
            residual = pred - targets
            loss = float((residual * residual).sum(axis=1).mean())
            loss_history.append(loss)
            final_error = loss
            grad = 2.0 / n * residual.T @ codes
            self.P0 = self.P0 - learning_rate * grad

        return LPTrainingReport(
            epochs=epochs,
            n_train_frames=n,
            learning_rate=float(learning_rate),
            loss_history=tuple(loss_history),
            final_train_error=final_error,
            c_train_flops=self.training_flops(n, epochs),
        )
