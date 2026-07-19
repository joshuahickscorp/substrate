
from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from mop.beds.starss23 import gate as gate_mod
from mop.beds.starss23.featurizer import D_FEAT, FrozenFeaturizer
from mop.beds.starss23.gate import (
    C_TRAIN_ANCHOR,
    D_IN,
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_FRAMES,
    FLOPS_PER_INFERENCE,
    HIDDEN,
    N_ONLINE,
    PARAM_CEILING,
    STATE_CEILING_BYTES,
    CandidateGate,
    GateRefusal,
    OnlineState,
    break_even_frames,
    inference_flops,
    param_count,
    training_flops,
)

_FORBIDDEN_ONLINE = ("azimuth", "elevation", "distance", "class", "onset", "label", "truth", "doa")


def _separable_voc_problem(n: int = 400, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D_IN))
    direction = rng.standard_normal(D_IN)
    score = x @ direction
    y = (score > np.median(score)).astype(np.float64)
    return x, y


def test_input_geometry() -> None:
    assert N_ONLINE == 8
    assert HIDDEN == 12
    assert D_IN == 264
    assert D_IN == D_FEAT + N_ONLINE


def test_param_count_matches_documented_formula() -> None:
    assert param_count() == 3193  # 264*12 + 12 + 12 + 1
    assert param_count(D_IN, HIDDEN) == 3193
    assert CandidateGate(seed=0).n_params() == 3193
    assert param_count() <= PARAM_CEILING
    assert PARAM_CEILING == 4096


def test_param_ceiling_assert_fails_when_budget_exceeded() -> None:
    assert param_count(D_IN, 16) == 4257
    with pytest.raises(GateRefusal):
        CandidateGate(hidden=16)
    assert param_count(D_IN, 15) == 3991
    within = CandidateGate(hidden=15)
    assert within.n_params() == 3991
    assert within.n_params() <= PARAM_CEILING


def test_state_stays_within_few_kilobytes() -> None:
    assert OnlineState.state_bytes() <= STATE_CEILING_BYTES
    assert STATE_CEILING_BYTES == 8192
    vector = OnlineState.initial().to_vector()
    assert vector.shape == (N_ONLINE,)
    assert vector.dtype == np.float64


def test_inference_flops_matches_documented_formula() -> None:
    assert inference_flops() == 6385  # 2*264*12 + 12 + 12 + 24 + 1
    assert FLOPS_PER_INFERENCE == 6385
    assert CandidateGate(seed=0).flops_per_inference() == 6385
    components = (
        gate_mod.FLOPS_MATMUL1,
        gate_mod.FLOPS_BIAS1,
        gate_mod.FLOPS_ACT1,
        gate_mod.FLOPS_MATMUL2,
        gate_mod.FLOPS_BIAS2,
    )
    assert components == (6336, 12, 12, 24, 1)
    assert sum(components) == 6385


def test_training_flops_and_c_train_anchor() -> None:
    assert training_flops() == 8_274_960_000
    assert training_flops(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS) == 8_274_960_000
    assert C_TRAIN_ANCHOR == 8_274_960_000
    assert CandidateGate(seed=0).training_flops(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS) == 8_274_960_000


def test_break_even_anchor() -> None:
    assert break_even_frames(4e4) == pytest.approx(206_874.0)
    with pytest.raises(GateRefusal):
        break_even_frames(0.0)
    with pytest.raises(GateRefusal):
        break_even_frames(-1.0)


def test_work_vectors_charge_the_right_buckets() -> None:
    gate = CandidateGate(seed=0)
    infer_work = gate.infer_work_vector(1000)
    assert infer_work.dispatch_and_exploration == FLOPS_PER_INFERENCE * 1000
    assert infer_work.total_work == FLOPS_PER_INFERENCE * 1000  # only the dispatch bucket is charged
    train_work = gate.train_work_vector(DEFAULT_TRAIN_FRAMES, DEFAULT_EPOCHS)
    assert train_work.learning == 8_274_960_000
    assert train_work.total_work == 8_274_960_000  # amortized training charged to learning only


def test_infer_interface_carries_no_label() -> None:
    infer_params = list(inspect.signature(CandidateGate.infer).parameters)
    assert infer_params == ["self", "features", "state"]
    update_params = list(inspect.signature(OnlineState.update).parameters)
    assert update_params == ["self", "features", "p_fire", "fired"]


def test_online_state_has_no_ground_truth_fields() -> None:
    names = [field.name.lower() for field in fields(OnlineState)]
    for forbidden in _FORBIDDEN_ONLINE:
        assert all(forbidden not in name for name in names), forbidden


def test_paired_seed_weights_are_reproducible() -> None:
    assert CandidateGate(seed=0).parameter_digest() == CandidateGate(seed=0).parameter_digest()
    assert CandidateGate(seed=0).parameter_digest() != CandidateGate(seed=1).parameter_digest()


def test_infer_is_deterministic() -> None:
    features = np.linspace(0.0, 1.0, D_FEAT)
    state = OnlineState.initial()
    assert CandidateGate(seed=3).infer(features, state) == CandidateGate(seed=3).infer(features, state)


def test_fit_is_deterministic() -> None:
    x, y = _separable_voc_problem()
    first = CandidateGate(seed=5)
    second = CandidateGate(seed=5)
    first.fit(x, y, epochs=25, learning_rate=0.3, ponder_lambda=0.0)
    second.fit(x, y, epochs=25, learning_rate=0.3, ponder_lambda=0.0)
    assert first.parameter_digest() == second.parameter_digest()


def test_online_state_update_is_causal_and_bounded() -> None:
    features = np.full(D_FEAT, 0.1)
    idle = OnlineState.initial()
    for _ in range(20):
        idle = idle.update(features, 0.2, fired=False)
    fired = idle.update(features, 0.9, fired=True)
    assert fired.n_frames == 21.0
    assert fired.n_fires == 1.0
    assert fired.last_fire_frame == 20.0
    assert fired.to_vector()[0] < idle.to_vector()[0]
    vector = fired.to_vector()
    assert np.all(vector >= -1.0 - 1e-9)
    assert np.all(vector <= 1.0 + 1e-9)


def test_predict_proba_matches_infer() -> None:
    gate = CandidateGate(seed=2)
    features = np.linspace(-0.5, 0.5, D_FEAT)
    state = OnlineState.initial().update(features, 0.3, fired=True)
    assembled = np.concatenate([features, state.to_vector()])
    batched = gate.predict_proba(assembled[None, :])[0]
    assert batched == pytest.approx(gate.infer(features, state), abs=1e-12)


def test_fire_respects_threshold() -> None:
    gate = CandidateGate(seed=2)
    features = np.linspace(-0.5, 0.5, D_FEAT)
    state = OnlineState.initial()
    fired, p_fire = gate.fire(features, state)
    assert fired == (p_fire >= gate.theta)
    assert gate.fire(features, state, theta=0.0)[0] is True
    assert gate.fire(features, state, theta=1.000001)[0] is False


def test_predict_proba_rejects_bad_shape() -> None:
    gate = CandidateGate(seed=0)
    with pytest.raises(GateRefusal):
        gate.predict_proba(np.zeros((4, D_IN + 1)))
    with pytest.raises(GateRefusal):
        gate.infer(np.zeros(D_FEAT + 1), OnlineState.initial())


def test_fit_learns_voc_targets() -> None:
    x, y = _separable_voc_problem()
    gate = CandidateGate(seed=1)
    report = gate.fit(x, y, epochs=400, learning_rate=0.3, ponder_lambda=0.0)
    probs = gate.predict_proba(x)
    accuracy = float(((probs >= 0.5).astype(np.float64) == y).mean())
    assert report.loss_history[-1] < 0.5 * report.loss_history[0]
    assert report.loss_history[-1] < 0.2
    assert accuracy > 0.9
    assert probs[y == 1].mean() - probs[y == 0].mean() > 0.3
    assert report.c_train_flops == training_flops(x.shape[0], 400)


def test_ponder_penalty_reduces_firing_rate() -> None:
    x, y = _separable_voc_problem()
    lenient = CandidateGate(seed=2)
    strict = CandidateGate(seed=2)
    lenient_report = lenient.fit(x, y, epochs=200, learning_rate=0.3, ponder_lambda=0.0)
    strict_report = strict.fit(x, y, epochs=200, learning_rate=0.3, ponder_lambda=1.0)
    assert strict_report.final_firing_rate < lenient_report.final_firing_rate


def test_fit_rejects_malformed_targets() -> None:
    gate = CandidateGate(seed=0)
    x = np.zeros((10, D_IN))
    with pytest.raises(GateRefusal):
        gate.fit(x, np.full(10, 0.5))  # not binary
    with pytest.raises(GateRefusal):
        gate.fit(x, np.zeros(9))  # misaligned length
    with pytest.raises(GateRefusal):
        gate.fit(np.zeros((10, D_IN + 3)), np.zeros(10))  # wrong input width


def test_gate_consumes_featurizer_output_online() -> None:
    rng = np.random.default_rng(11)
    from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME

    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * 5))
    features = FrozenFeaturizer().featurize(audio)
    gate = CandidateGate(seed=0)
    state = OnlineState.initial()
    fired_count = 0
    for frame in features:
        fired, p_fire = gate.fire(frame, state)
        assert 0.0 <= p_fire <= 1.0
        state = state.update(frame, p_fire, fired)
        fired_count += int(fired)
    assert state.n_frames == features.shape[0]
    assert 0 <= fired_count <= features.shape[0]
