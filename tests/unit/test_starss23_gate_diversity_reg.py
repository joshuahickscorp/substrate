
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mop.beds.starss23.featurizer import D_FEAT, FrozenFeaturizer
from mop.beds.starss23.gate import (
    D_IN,
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_FRAMES,
    PARAM_CEILING,
    STATE_CEILING_BYTES,
    CandidateGate,
    GateRefusal,
    OnlineState,
    inference_flops,
    param_count,
    training_flops,
)
from mop.beds.starss23.gate_diversity_reg import (
    DEFAULT_SPACING_WINDOW,
    DiversityRegGate,
    DiversityRegRefusal,
    DiversityRegTrainingReport,
    _clip_index,
    _neighbor_probability_sum,
    spacing_kernel,
)
from mop.beds.starss23.schema import COLLAR_FRAMES

_FORBIDDEN_ONLINE = ("azimuth", "elevation", "distance", "class", "onset", "label", "truth", "doa")


def _separable_voc_problem(n: int = 400, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D_IN))
    direction = rng.standard_normal(D_IN)
    score = x @ direction
    y = (score > np.median(score)).astype(np.float64)
    return x, y


def test_param_count_matches_committed_gate() -> None:
    assert DiversityRegGate(seed=0).n_params() == 3193
    assert DiversityRegGate(seed=0).n_params() == param_count()
    assert DiversityRegGate(seed=0).n_params() <= PARAM_CEILING


def test_param_ceiling_assert_fails_when_budget_exceeded() -> None:
    with pytest.raises(DiversityRegRefusal):
        DiversityRegGate(hidden=16)
    within = DiversityRegGate(hidden=15)
    assert within.n_params() == 3991
    assert within.n_params() <= PARAM_CEILING


def test_state_stays_within_few_kilobytes() -> None:
    assert OnlineState.state_bytes() <= STATE_CEILING_BYTES


def test_refusal_is_a_gate_refusal() -> None:
    assert issubclass(DiversityRegRefusal, GateRefusal)


def test_flop_cost_functions_match_committed_gate() -> None:
    gate = DiversityRegGate(seed=0)
    assert gate.flops_per_inference() == 6385
    assert gate.flops_per_inference() == inference_flops()
    assert gate.training_flops(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS) == training_flops()
    assert gate.training_flops(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS) == 8_274_960_000


def test_work_vectors_charge_the_right_buckets() -> None:
    gate = DiversityRegGate(seed=0)
    infer_work = gate.infer_work_vector(1000)
    assert infer_work.dispatch_and_exploration == 6385 * 1000
    assert infer_work.total_work == 6385 * 1000
    train_work = gate.train_work_vector(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS)
    assert train_work.learning == 8_274_960_000
    assert train_work.total_work == 8_274_960_000


def test_infer_interface_carries_no_label() -> None:
    infer_params = list(inspect.signature(DiversityRegGate.infer).parameters)
    assert infer_params == ["self", "features", "state"]


def test_no_ground_truth_in_online_state() -> None:
    from dataclasses import fields

    names = [field.name.lower() for field in fields(OnlineState)]
    for forbidden in _FORBIDDEN_ONLINE:
        assert all(forbidden not in name for name in names), forbidden


def test_init_weights_match_committed_gate() -> None:
    for seed in (0, 1, 7):
        assert DiversityRegGate(seed=seed).parameter_digest() == CandidateGate(seed=seed).parameter_digest()


def test_lambda_zero_fit_reproduces_committed_gate_byte_for_byte() -> None:
    x, y = _separable_voc_problem()
    committed = CandidateGate(seed=3)
    variant = DiversityRegGate(seed=3, diversity_lambda=0.0)
    committed_report = committed.fit(x, y, epochs=8, learning_rate=0.1, ponder_lambda=0.02)
    variant_report = variant.fit(x, y, epochs=8, learning_rate=0.1, ponder_lambda=0.02)
    assert committed.parameter_digest() == variant.parameter_digest()
    assert list(committed_report.loss_history) == list(variant_report.loss_history)
    assert variant_report.final_spacing_penalty == 0.0


def test_positive_lambda_changes_the_weights() -> None:
    x, y = _separable_voc_problem()
    baseline = DiversityRegGate(seed=3, diversity_lambda=0.0)
    regularized = DiversityRegGate(seed=3, diversity_lambda=2.0)
    baseline.fit(x, y, epochs=8, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[400])
    regularized.fit(x, y, epochs=8, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[400])
    assert baseline.parameter_digest() != regularized.parameter_digest()


def test_paired_seed_weights_are_reproducible() -> None:
    a = DiversityRegGate(seed=0, diversity_lambda=1.5)
    b = DiversityRegGate(seed=0, diversity_lambda=1.5)
    assert a.parameter_digest() == b.parameter_digest()
    assert a.parameter_digest() != DiversityRegGate(seed=1, diversity_lambda=1.5).parameter_digest()


def test_infer_is_deterministic() -> None:
    features = np.linspace(0.0, 1.0, D_FEAT)
    state = OnlineState.initial()
    first = DiversityRegGate(seed=3, diversity_lambda=1.0).infer(features, state)
    second = DiversityRegGate(seed=3, diversity_lambda=1.0).infer(features, state)
    assert first == second


def test_fit_is_deterministic() -> None:
    x, y = _separable_voc_problem()
    first = DiversityRegGate(seed=5, diversity_lambda=1.0)
    second = DiversityRegGate(seed=5, diversity_lambda=1.0)
    first.fit(x, y, epochs=20, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[200, 200])
    second.fit(x, y, epochs=20, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[200, 200])
    assert first.parameter_digest() == second.parameter_digest()


def test_spacing_kernel_is_normalized_and_decreasing() -> None:
    kernel = spacing_kernel(DEFAULT_SPACING_WINDOW)
    assert DEFAULT_SPACING_WINDOW == COLLAR_FRAMES == 2
    assert kernel == pytest.approx(np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert float(kernel.sum()) == pytest.approx(1.0)
    assert kernel[0] > kernel[1]
    with pytest.raises(DiversityRegRefusal):
        spacing_kernel(0)


def test_neighbor_sum_respects_clip_boundaries() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(12)
    clip_index = _clip_index([5, 7], 12)
    kernel = spacing_kernel(2)
    fast = _neighbor_probability_sum(p, clip_index, kernel)
    brute = np.zeros(12)
    for a in range(12):
        for b in range(12):
            distance = abs(a - b)
            if 0 < distance <= 2 and clip_index[a] == clip_index[b]:
                brute[a] += kernel[distance - 1] * p[b]
    assert np.allclose(fast, brute)
    assert clip_index[4] != clip_index[5]


def test_clip_index_rejects_bad_segment_lengths() -> None:
    with pytest.raises(DiversityRegRefusal):
        _clip_index([5, 5], 12)  # sums to 10, not 12
    with pytest.raises(DiversityRegRefusal):
        _clip_index([0, 12], 12)  # nonpositive segment


def _adjacency_energy_ratio(p: np.ndarray, clip_index: np.ndarray, kernel: np.ndarray) -> float:
    neighbor = _neighbor_probability_sum(p, clip_index, kernel)
    a_adjacency = float((p * neighbor).sum())
    q_energy = float((p * p).sum()) + 1e-12
    return a_adjacency / q_energy


def test_ratio_penalty_is_scale_invariant_unlike_raw_pairwise() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(20)
    clip_index = _clip_index([10, 10], 20)
    kernel = spacing_kernel(2)
    assert _adjacency_energy_ratio(p, clip_index, kernel) == pytest.approx(
        _adjacency_energy_ratio(3.0 * p, clip_index, kernel), rel=1e-9
    )
    raw = float((p * _neighbor_probability_sum(p, clip_index, kernel)).sum())
    raw_scaled = float((3.0 * p * _neighbor_probability_sum(3.0 * p, clip_index, kernel)).sum())
    assert raw_scaled == pytest.approx(9.0 * raw, rel=1e-9)


def test_ratio_penalty_gradient_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(15) + 0.1
    clip_index = _clip_index([15], 15)
    kernel = spacing_kernel(2)
    neighbor = _neighbor_probability_sum(p, clip_index, kernel)
    a_adjacency = float((p * neighbor).sum())
    q_energy = float((p * p).sum()) + 1e-12
    analytic = 2.0 * neighbor / q_energy - 2.0 * a_adjacency * p / (q_energy * q_energy)
    eps = 1e-6
    numerical = np.zeros_like(p)
    for i in range(p.shape[0]):
        plus = p.copy()
        plus[i] += eps
        minus = p.copy()
        minus[i] -= eps
        numerical[i] = (
            _adjacency_energy_ratio(plus, clip_index, kernel)
            - _adjacency_energy_ratio(minus, clip_index, kernel)
        ) / (2.0 * eps)
    assert np.allclose(analytic, numerical, atol=1e-5)


def test_spacing_penalty_is_zero_at_lambda_zero_and_positive_otherwise() -> None:
    x, y = _separable_voc_problem()
    zero = DiversityRegGate(seed=1, diversity_lambda=0.0).fit(
        x, y, epochs=4, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[400]
    )
    positive = DiversityRegGate(seed=1, diversity_lambda=1.0).fit(
        x, y, epochs=4, learning_rate=0.1, ponder_lambda=0.02, segment_lengths=[400]
    )
    assert zero.final_spacing_penalty == 0.0
    assert positive.final_spacing_penalty > 0.0
    assert isinstance(positive, DiversityRegTrainingReport)
    payload = positive.payload()
    assert payload["diversity_lambda"] == 1.0
    assert payload["spacing_window"] == DEFAULT_SPACING_WINDOW


def test_construction_rejects_bad_diversity_lambda() -> None:
    with pytest.raises(DiversityRegRefusal):
        DiversityRegGate(diversity_lambda=-1.0)
    with pytest.raises(DiversityRegRefusal):
        DiversityRegGate(diversity_lambda=float("nan"))
    with pytest.raises(DiversityRegRefusal):
        DiversityRegGate(diversity_lambda=True)


def test_construction_rejects_bad_spacing_window() -> None:
    with pytest.raises(DiversityRegRefusal):
        DiversityRegGate(spacing_window=0)


def test_fit_rejects_malformed_targets() -> None:
    gate = DiversityRegGate(seed=0, diversity_lambda=1.0)
    x = np.zeros((10, D_IN))
    with pytest.raises(DiversityRegRefusal):
        gate.fit(x, np.full(10, 0.5))  # not binary
    with pytest.raises(DiversityRegRefusal):
        gate.fit(x, np.zeros(9))  # misaligned length
    with pytest.raises(DiversityRegRefusal):
        gate.fit(np.zeros((10, D_IN + 3)), np.zeros(10))  # wrong input width


def test_fit_rejects_segment_lengths_that_do_not_cover_the_batch() -> None:
    gate = DiversityRegGate(seed=0, diversity_lambda=1.0)
    x, y = _separable_voc_problem(n=100)
    with pytest.raises(DiversityRegRefusal):
        gate.fit(x, y, epochs=2, segment_lengths=[50, 40])  # sums to 90, not 100


def test_predict_proba_matches_infer_and_rejects_bad_shape() -> None:
    gate = DiversityRegGate(seed=2, diversity_lambda=1.0)
    features = np.linspace(-0.5, 0.5, D_FEAT)
    state = OnlineState.initial().update(features, 0.3, fired=True)
    assembled = np.concatenate([features, state.to_vector()])
    assert gate.predict_proba(assembled[None, :])[0] == pytest.approx(gate.infer(features, state), abs=1e-12)
    with pytest.raises(DiversityRegRefusal):
        gate.predict_proba(np.zeros((4, D_IN + 1)))
    with pytest.raises(DiversityRegRefusal):
        gate.infer(np.zeros(D_FEAT + 1), OnlineState.initial())


def test_fire_respects_threshold() -> None:
    gate = DiversityRegGate(seed=2, diversity_lambda=1.0)
    features = np.linspace(-0.5, 0.5, D_FEAT)
    state = OnlineState.initial()
    fired, p_fire = gate.fire(features, state)
    assert fired == (p_fire >= gate.theta)
    assert gate.fire(features, state, theta=0.0)[0] is True
    assert gate.fire(features, state, theta=1.000001)[0] is False


def test_gate_consumes_featurizer_output_online() -> None:
    from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME

    rng = np.random.default_rng(11)
    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * 5))
    features = FrozenFeaturizer().featurize(audio)
    gate = DiversityRegGate(seed=0, diversity_lambda=1.0)
    state = OnlineState.initial()
    fired_count = 0
    for frame in features:
        fired, p_fire = gate.fire(frame, state)
        assert 0.0 <= p_fire <= 1.0
        state = state.update(frame, p_fire, fired)
        fired_count += int(fired)
    assert state.n_frames == features.shape[0]
    assert 0 <= fired_count <= features.shape[0]
