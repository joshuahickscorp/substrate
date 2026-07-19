
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mop.beds.starss23.featurizer import D_FEAT
from mop.beds.starss23.gate import (
    FLOPS_PER_INFERENCE,
    PARAM_CEILING,
    CandidateGate,
    GateRefusal,
    OnlineState,
)
from mop.beds.starss23.gate_recurrence_spread import (
    DEFAULT_DENSITY_GRID,
    DEFAULT_REFRACTORY_GRID,
    FLOPS_PER_INFERENCE_SPREAD,
    FLOPS_SPREAD_HEAD,
    N_SPREAD_PARAMS,
    REFRACTORY_FRAMES,
    VARIANT_ID,
    RecurrenceSpreadGate,
)


def _gate(seed: int = 0, rho: float = 0.08, **kwargs) -> RecurrenceSpreadGate:
    return RecurrenceSpreadGate(signal_gate=CandidateGate(seed=seed), rho=rho, **kwargs)


def _synthetic_clips(
    n_clips: int = 3, n_frames: int = 60, seed: int = 7
) -> list[tuple[np.ndarray, list[int]]]:

    rng = np.random.default_rng(seed)
    clips: list[tuple[np.ndarray, list[int]]] = []
    for _c in range(n_clips):
        features = rng.standard_normal((n_frames, D_FEAT)) ** 2  # nonnegative flux-like features
        onsets = sorted(int(v) for v in rng.choice(n_frames, size=6, replace=False))
        clips.append((features, onsets))
    return clips




def test_variant_id_and_spread_param_count() -> None:
    assert VARIANT_ID == "recurrence_spread"
    assert N_SPREAD_PARAMS == 2
    gate = _gate()
    assert gate.n_spread_params() == 2
    assert gate.n_params() == 3195
    assert gate.n_params() == CandidateGate(seed=0).n_params() + N_SPREAD_PARAMS
    assert gate.n_params() <= PARAM_CEILING
    assert PARAM_CEILING == 4096


def test_param_budget_holds_for_widest_valid_signal_head() -> None:
    wide = RecurrenceSpreadGate(signal_gate=CandidateGate(seed=0, hidden=15), rho=0.08)
    assert wide.n_params() == 3993
    assert wide.n_params() <= PARAM_CEILING


def test_construction_refuses_bad_inputs() -> None:
    with pytest.raises(GateRefusal):
        RecurrenceSpreadGate(signal_gate=object(), rho=0.08)  # type: ignore[arg-type]
    for bad_rho in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(GateRefusal):
            _gate(rho=bad_rho)
    with pytest.raises(GateRefusal):
        _gate(density_weight=-1.0)
    with pytest.raises(GateRefusal):
        _gate(refractory_weight=-0.5)
    with pytest.raises(GateRefusal):
        _gate(refractory_frames=0)


def test_state_is_the_committed_few_kb_online_state() -> None:
    assert OnlineState.state_bytes() <= 8192
    assert REFRACTORY_FRAMES == 3




def test_flops_surface_matches_documented_constants() -> None:
    assert FLOPS_PER_INFERENCE_SPREAD == FLOPS_PER_INFERENCE + FLOPS_SPREAD_HEAD
    assert FLOPS_PER_INFERENCE_SPREAD == 6407
    gate = _gate()
    assert gate.flops_per_inference() == 6407
    assert gate.signal_training_flops(1000, 8) == CandidateGate(seed=0).training_flops(1000, 8)
    assert gate.total_training_flops(1000, 8) == gate.signal_training_flops(1000, 8) + gate.search_flops




def test_infer_interface_carries_no_label() -> None:
    params = list(inspect.signature(RecurrenceSpreadGate.infer).parameters)
    assert params == ["self", "features", "state"]




def test_no_spread_is_byte_identical_to_the_committed_gate() -> None:
    signal = CandidateGate(seed=3)
    gate = RecurrenceSpreadGate(signal_gate=signal, rho=0.08)  # weights default to 0.0
    rng = np.random.default_rng(1)
    features = rng.standard_normal((40, D_FEAT))
    state = OnlineState.initial()
    for row in features:
        assert gate.infer(row, state) == signal.infer(row, state)
        p = gate.infer(row, state)
        state = state.update(row, p, p >= 0.5)




def test_penalty_only_suppresses_after_a_recent_fire() -> None:
    signal = CandidateGate(seed=2)
    gate = RecurrenceSpreadGate(
        signal_gate=signal, rho=0.08, density_weight=2.0, refractory_weight=4.0
    )
    features = np.linspace(-0.5, 0.5, D_FEAT)
    fired_state = OnlineState.initial().update(features, 0.9, fired=True)
    assert gate.infer(features, fired_state) < signal.infer(features, fired_state)
    idle = OnlineState.initial()
    assert gate.infer(features, idle) == signal.infer(features, idle)


def test_effective_probability_stays_in_unit_interval() -> None:
    gate = _gate(density_weight=3.0, refractory_weight=6.0)
    rng = np.random.default_rng(5)
    features = rng.standard_normal((80, D_FEAT))
    fires, probs = gate.causal_pass(features, theta=0.3)
    assert probs.shape == (80,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert all(0 <= f < 80 for f in fires)
    assert fires == sorted(set(fires))




def test_causal_pass_is_deterministic() -> None:
    a = _gate(seed=4, density_weight=1.0, refractory_weight=2.0)
    b = _gate(seed=4, density_weight=1.0, refractory_weight=2.0)
    rng = np.random.default_rng(9)
    features = rng.standard_normal((100, D_FEAT))
    fa, pa = a.causal_pass(features, 0.4)
    fb, pb = b.causal_pass(features, 0.4)
    assert fa == fb
    assert np.array_equal(pa, pb)


def test_parameter_digest_tracks_weights_and_seed() -> None:
    base = _gate(seed=0)
    assert base.parameter_digest() == _gate(seed=0).parameter_digest()
    assert base.parameter_digest() != _gate(seed=1).parameter_digest()
    assert base.parameter_digest() != _gate(seed=0, density_weight=1.0).parameter_digest()
    assert base.parameter_digest() != _gate(seed=0, refractory_weight=1.0).parameter_digest()




def test_fit_spread_is_deterministic_and_records_honest_search_cost() -> None:
    clips = _synthetic_clips()
    first = _gate(seed=6)
    second = _gate(seed=6)
    report_a = first.fit_spread(clips, rate=0.1)
    report_b = second.fit_spread(clips, rate=0.1)
    assert (first.density_weight, first.refractory_weight) == (
        second.density_weight,
        second.refractory_weight,
    )
    assert report_a.payload() == report_b.payload()
    n_grid = len(DEFAULT_DENSITY_GRID) * len(DEFAULT_REFRACTORY_GRID)
    assert report_a.n_search_evals == n_grid
    frames = sum(f.shape[0] for f, _ in clips)
    assert first.search_flops == n_grid * 2 * frames * FLOPS_PER_INFERENCE_SPREAD
    assert first.search_flops > 0
    assert first.density_weight >= 0.0 and first.refractory_weight >= 0.0


def test_fit_spread_never_underperforms_the_no_spread_point_on_train() -> None:
    clips = _synthetic_clips(seed=11)
    gate = _gate(seed=1)
    report = gate.fit_spread(clips, rate=0.1)
    assert report.best_train_f1 >= report.base_train_f1
    if report.selected_no_spread:
        assert (gate.density_weight, gate.refractory_weight) == (0.0, 0.0)


def test_fit_spread_requires_the_no_spread_point_in_the_grid() -> None:
    gate = _gate(seed=0)
    with pytest.raises(GateRefusal):
        gate.fit_spread(_synthetic_clips(), rate=0.1, density_grid=(1.0, 2.0))
    with pytest.raises(GateRefusal):
        gate.fit_spread(_synthetic_clips(), rate=0.1, refractory_grid=(1.0, 2.0))
    with pytest.raises(GateRefusal):
        gate.fit_spread([], rate=0.1)
