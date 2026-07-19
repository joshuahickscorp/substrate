
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mop.beds.starss23.featurizer import D_FEAT
from mop.beds.starss23.gate import PARAM_CEILING, STATE_CEILING_BYTES
from mop.beds.starss23.gate_learning_progress import (
    C_TRAIN_ANCHOR,
    LP_N_PARAMS,
    LP_PROJ_DIM,
    LP_TARGET_DIM,
    LearningProgressGate,
    LPGateRefusal,
    LPOnlineState,
    inference_flops,
    param_count,
    training_flops,
)

_FORBIDDEN_ONLINE = ("azimuth", "elevation", "distance", "class", "onset", "label", "truth", "doa", "gt")


def _fake_clip_features(n: int = 240, seed: int = 7) -> np.ndarray:

    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal((n, D_FEAT)))




def test_param_count_under_ceiling() -> None:
    assert LP_N_PARAMS == LP_PROJ_DIM * LP_TARGET_DIM
    assert param_count() == LP_N_PARAMS
    assert LP_N_PARAMS <= PARAM_CEILING
    gate = LearningProgressGate(seed=0)
    assert gate.n_params() == LP_N_PARAMS
    assert gate.n_params() <= PARAM_CEILING


def test_param_ceiling_is_enforced() -> None:
    with pytest.raises(LPGateRefusal):
        LearningProgressGate(seed=0, proj_dim=256, target_dim=32)


def test_online_state_within_few_kb() -> None:
    gate = LearningProgressGate(seed=1)
    assert gate.state_bytes() == LPOnlineState.state_bytes(LP_PROJ_DIM, LP_TARGET_DIM)
    assert gate.state_bytes() <= STATE_CEILING_BYTES


def test_theta_must_be_a_probability() -> None:
    with pytest.raises(LPGateRefusal):
        LearningProgressGate(seed=0, theta=1.5)




def test_infer_and_update_signatures_are_label_free() -> None:
    for method in (LearningProgressGate.infer, LearningProgressGate.update):
        params = list(inspect.signature(method).parameters)
        assert params == ["self", "features", "state"], params
        for name in params:
            assert not any(token in name.lower() for token in _FORBIDDEN_ONLINE)


def test_fit_signature_is_self_supervised() -> None:
    params = list(inspect.signature(LearningProgressGate.fit).parameters)
    assert "features" in params
    for name in params:
        assert not any(token in name.lower() for token in _FORBIDDEN_ONLINE)




def test_gate_is_deterministic_in_seed() -> None:
    features = _fake_clip_features()
    a = LearningProgressGate(seed=3)
    b = LearningProgressGate(seed=3)
    assert a.parameter_digest() == b.parameter_digest()
    np.testing.assert_array_equal(a.causal_scores(features), b.causal_scores(features))


def test_distinct_seeds_give_distinct_tensors() -> None:
    assert LearningProgressGate(seed=0).parameter_digest() != LearningProgressGate(seed=1).parameter_digest()


def test_batched_causal_matches_per_frame_loop() -> None:
    features = _fake_clip_features(n=180, seed=11)
    gate = LearningProgressGate(seed=5)
    batched = gate.causal_scores(features)

    state = gate.initial_state()
    manual = np.empty(features.shape[0], dtype=np.float64)
    for i in range(features.shape[0]):
        manual[i] = gate.infer(features[i], state)
        state = gate.update(features[i], state)
    np.testing.assert_allclose(batched, manual, rtol=0.0, atol=1e-12)


def test_scores_are_probabilities_and_theta_independent() -> None:
    features = _fake_clip_features(n=200, seed=2)
    gate = LearningProgressGate(seed=2)
    probs = gate.causal_scores(features)
    assert probs.shape == (200,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    fires_a, probs_a = gate.causal_fires(features, theta=0.5)
    fires_b, probs_b = gate.causal_fires(features, theta=0.9)
    np.testing.assert_array_equal(probs_a, probs_b)
    assert set(fires_b).issubset(set(fires_a))


def test_fire_matches_threshold() -> None:
    features = _fake_clip_features(n=50, seed=4)
    gate = LearningProgressGate(seed=4)
    state = gate.initial_state()
    fired, p = gate.fire(features[0], state, theta=0.0)
    assert fired is True and 0.0 <= p <= 1.0




def test_fit_reduces_training_error_and_charges_c_train() -> None:
    features = _fake_clip_features(n=300, seed=9)
    gate = LearningProgressGate(seed=6)
    report = gate.fit(features, epochs=8, learning_rate=0.05)
    assert report.n_train_frames == 300
    assert report.loss_history[-1] <= report.loss_history[0]
    assert report.c_train_flops == training_flops(300, 8)
    assert report.c_train_flops > 0


def test_fit_rejects_wrong_shape() -> None:
    gate = LearningProgressGate(seed=0)
    with pytest.raises(LPGateRefusal):
        gate.fit(np.zeros((10, D_FEAT + 1)), epochs=2)




def test_flop_functions_are_exposed_and_positive() -> None:
    gate = LearningProgressGate(seed=0)
    assert gate.flops_per_inference() == inference_flops()
    assert gate.flops_per_inference() > 0
    assert gate.training_flops(1000, 8) == training_flops(1000, 8)
    assert training_flops() == C_TRAIN_ANCHOR
    assert training_flops(2000, 8) == 2 * training_flops(1000, 8)
    assert training_flops(1000, 16) == 2 * training_flops(1000, 8)


def test_infer_rejects_wrong_feature_length() -> None:
    gate = LearningProgressGate(seed=0)
    state = gate.initial_state()
    with pytest.raises(LPGateRefusal):
        gate.infer(np.zeros(D_FEAT + 3), state)
