
from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from mop.beds.starss23.featurizer import FrozenFeaturizer
from mop.beds.starss23.gate import (
    PARAM_CEILING,
    STATE_CEILING_BYTES,
    CandidateGate,
    GateRefusal,
    OnlineState,
)
from mop.beds.starss23.gate_refractory_nms import (
    DEFAULT_WINDOW_FRAMES,
    RefractoryNmsGate,
    pooled_post_nms_fraction,
    refractory_nms_select,
    tune_theta_for_rate,
)
from mop.beds.starss23.schema import COLLAR_FRAMES, N_CHANNELS, SAMPLES_PER_FRAME
from mop.science.gating import causal_gate_trace

_FORBIDDEN_ONLINE = ("azimuth", "elevation", "distance", "class", "onset", "label", "truth", "doa")


def _feature_block(seed: int, n_frames: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * n_frames))
    return FrozenFeaturizer().featurize(audio)


def test_window_default_is_the_collar_width() -> None:
    assert DEFAULT_WINDOW_FRAMES == COLLAR_FRAMES == 2
    assert RefractoryNmsGate(seed=0).window == 2
    assert RefractoryNmsGate(seed=0, window=3).window == 3


def test_param_budget_is_the_committed_gate_budget() -> None:
    gate = RefractoryNmsGate(seed=0)
    assert gate.n_params() == 3193
    assert gate.n_params() <= PARAM_CEILING == 4096
    with pytest.raises(GateRefusal):
        RefractoryNmsGate(seed=0, hidden=16)


def test_state_stays_within_few_kilobytes() -> None:
    assert OnlineState.state_bytes() <= STATE_CEILING_BYTES == 8192


def test_infer_interface_carries_no_label() -> None:
    assert list(inspect.signature(RefractoryNmsGate.infer).parameters) == ["self", "features", "state"]
    names = [field.name.lower() for field in fields(OnlineState)]
    for forbidden in _FORBIDDEN_ONLINE:
        assert all(forbidden not in name for name in names), forbidden


def test_c_train_is_exposed_for_the_flop_ledger() -> None:
    gate = RefractoryNmsGate(seed=0)
    assert gate.training_flops(54_000, 8) == 8_274_960_000
    assert gate.flops_per_inference() == 6385


def test_weights_are_byte_identical_to_the_committed_gate() -> None:
    assert RefractoryNmsGate(seed=2).parameter_digest() == CandidateGate(seed=2).parameter_digest()
    assert RefractoryNmsGate(seed=2).parameter_digest() != CandidateGate(seed=3).parameter_digest()


def test_training_leaves_weights_byte_identical_to_the_committed_gate() -> None:
    rng = np.random.default_rng(9)
    from mop.beds.starss23.gate import D_IN

    x = rng.standard_normal((300, D_IN))
    y = (x @ rng.standard_normal(D_IN) > 0).astype(np.float64)
    variant = RefractoryNmsGate(seed=5)
    committed = CandidateGate(seed=5)
    variant.fit(x, y, epochs=20, learning_rate=0.2, ponder_lambda=0.02)
    committed.fit(x, y, epochs=20, learning_rate=0.2, ponder_lambda=0.02)
    assert variant.parameter_digest() == committed.parameter_digest()


def test_causal_probs_are_byte_identical_to_the_committed_causal_pass() -> None:
    features = _feature_block(seed=11, n_frames=60)
    variant = RefractoryNmsGate(seed=1)
    committed = CandidateGate(seed=1)
    theta = 0.5
    _, committed_probs = causal_gate_trace(committed, features, theta, OnlineState.initial)
    variant_probs = variant.causal_probs(features, theta)
    assert np.array_equal(variant_probs, committed_probs)


def test_nms_keeps_one_local_maximum_per_cluster() -> None:
    probs = np.array([0.1, 0.6, 0.7, 0.65, 0.1, 0.1, 0.9, 0.2])
    assert refractory_nms_select(probs, 0.5, 2) == [2, 6]


def test_nms_strictly_higher_score_moves_the_single_held_peak() -> None:
    assert refractory_nms_select(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), 0.4, 2) == [4]
    assert refractory_nms_select(np.array([0.7, 0.7, 0.7]), 0.5, 2) == [0]


def test_nms_separates_committed_fires_by_more_than_the_window() -> None:
    rng = np.random.default_rng(4)
    probs = rng.random(800)
    for window in (1, 2, 3):
        fires = refractory_nms_select(probs, 0.5, window)
        if len(fires) > 1:
            assert int(np.diff(fires).min()) > window
        if window == COLLAR_FRAMES:
            for i in range(1, len(fires)):
                assert fires[i] - fires[i - 1] > COLLAR_FRAMES


def test_nms_below_threshold_fires_nothing() -> None:
    assert refractory_nms_select(np.array([0.1, 0.2, 0.3]), 0.9, 2) == []


def test_nms_is_deterministic() -> None:
    probs = np.random.default_rng(7).random(400)
    assert refractory_nms_select(probs, 0.4, 2) == refractory_nms_select(probs, 0.4, 2)


def test_refractory_fires_equal_select_on_causal_probs() -> None:
    features = _feature_block(seed=3, n_frames=120)
    gate = RefractoryNmsGate(seed=0, window=2)
    theta = 0.5
    expected = refractory_nms_select(gate.causal_probs(features, theta), theta, 2)
    assert gate.refractory_fires(features, theta) == expected
    assert gate.refractory_fires(features, theta) == gate.refractory_fires(features, theta)


def test_refractory_fires_de_clusters_relative_to_raw_threshold() -> None:
    features = _feature_block(seed=21, n_frames=400)
    gate = RefractoryNmsGate(seed=1, window=2)
    theta = 0.5
    raw_fires, _ = causal_gate_trace(gate, features, theta, OnlineState.initial)
    nms_fires = gate.refractory_fires(features, theta)
    assert len(nms_fires) <= len(raw_fires)
    if len(nms_fires) > 1:
        assert int(np.diff(nms_fires).min()) > COLLAR_FRAMES


def test_tune_theta_hits_the_post_nms_target_rate() -> None:
    rng = np.random.default_rng(2)
    traces = [rng.random(1500) for _ in range(6)]
    for rate in (0.06, 0.08, 0.10):
        theta = tune_theta_for_rate(traces, rate, 2)
        fraction = pooled_post_nms_fraction(traces, theta, 2)
        assert abs(fraction - rate) < 0.02


def test_tune_theta_signature_carries_no_label() -> None:
    params = list(inspect.signature(tune_theta_for_rate).parameters)
    assert params[:3] == ["prob_traces", "target_rate", "window"]
