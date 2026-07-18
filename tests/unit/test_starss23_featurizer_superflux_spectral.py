"""Tests for the frozen zero-parameter SuperFlux featurizer (F1 frozen-featurizer wave).

These lock the properties the ESCS constraints require of a swapped front-end: it carries zero trained
parameters, it is byte-reproducible, it emits exactly the gate's 256-dim input so the unchanged gate
consumes it, its per-frame FLOPs are the documented analytic cost and are charged into every arm under the
6e10 ceiling, and the sealed artifact and preregistration carry a reproducible seal. Claim scope:
deterministic programmatic mechanics only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23 import FLOP_CEILING
from mop.beds.starss23.featurizer import FrozenFeaturizer
from mop.beds.starss23.featurizer_superflux_spectral import (
    COLS_PER_FRAME,
    D_FEAT,
    FLOPS_COMPRESS,
    FLOPS_FLUX,
    FLOPS_MAXFILT,
    FLOPS_MEL,
    FLOPS_PER_COL_PER_CH,
    FLOPS_PER_FRAME,
    FLOPS_POWER,
    FLOPS_RFFT,
    FLOPS_WINDOW,
    MAX_FILTER_RADIUS,
    N_MEL,
    SuperfluxSpectralFeaturizer,
)
from mop.beds.starss23.gate import CandidateGate, OnlineState
from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME
from mop.substrate.events import canonical_sha256


def _fixture_audio(n_frames: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))


# -- zero trained parameters --------------------------------------------------


def test_zero_trained_parameters() -> None:
    assert SuperfluxSpectralFeaturizer().n_params() == 0


def test_parameter_digest_is_stable_and_distinct_from_the_base_front_end() -> None:
    f = SuperfluxSpectralFeaturizer()
    # Fixed DSP: two instances have byte-identical parameters (no learned weight).
    assert f.parameter_digest() == SuperfluxSpectralFeaturizer().parameter_digest()
    # The SuperFlux DSP (mu-law + frequency max filter) differs from the base flux, so its digest and any
    # cache keyed on it differ; the frozen cache can never be served this front-end and vice versa.
    assert f.parameter_digest() != FrozenFeaturizer().parameter_digest()


# -- deterministic, byte-reproducible feature bytes ---------------------------


def test_deterministic_feature_bytes_across_runs_and_instances() -> None:
    audio = _fixture_audio(8)
    a = SuperfluxSpectralFeaturizer().featurize(audio)
    b = SuperfluxSpectralFeaturizer().featurize(audio)
    assert a.tobytes() == b.tobytes()
    assert SuperfluxSpectralFeaturizer().feature_digest(a) == SuperfluxSpectralFeaturizer().feature_digest(b)


def test_silence_gives_exactly_zero_flux() -> None:
    features = SuperfluxSpectralFeaturizer().featurize(np.zeros((N_CHANNELS, 4 * SAMPLES_PER_FRAME)))
    assert float(np.abs(features).sum()) == 0.0


def test_flux_responds_to_a_planted_onset() -> None:
    n_frames = 6
    times = np.arange(n_frames * SAMPLES_PER_FRAME) / 24_000
    signal = np.zeros(n_frames * SAMPLES_PER_FRAME, dtype=np.float64)
    onset = 3 * SAMPLES_PER_FRAME
    signal[onset:] = np.sin(2.0 * np.pi * 1000.0 * times[onset:])
    audio = np.repeat(signal[None, :], N_CHANNELS, axis=0) * np.array([1.0, 0.7, 0.5, 0.3])[:, None]
    per_frame_l1 = np.abs(SuperfluxSpectralFeaturizer().featurize(audio)).sum(axis=1)
    # Leading silence carries no flux; the transition frame carries a strong positive flux.
    assert per_frame_l1[0] == pytest.approx(0.0, abs=1e-9)
    assert per_frame_l1[1] == pytest.approx(0.0, abs=1e-9)
    assert per_frame_l1[2] > 1.0
    assert per_frame_l1[2] > per_frame_l1[3]


# -- exactly 256-dim output feeds the unchanged gate with no adaptation -------


def test_output_is_256_dim_and_feeds_the_unchanged_gate() -> None:
    features = SuperfluxSpectralFeaturizer().featurize(_fixture_audio(6))
    assert features.shape == (6, D_FEAT)
    assert D_FEAT == 256 == N_MEL * N_CHANNELS
    assert features.dtype == np.float64
    # The unchanged 264-input gate consumes a 256-length feature row directly (no projection/truncation).
    p_fire = CandidateGate(seed=0).infer(features[0], OnlineState.initial())
    assert isinstance(p_fire, float)
    assert 0.0 <= p_fire <= 1.0


def test_rejects_bad_audio_shape() -> None:
    f = SuperfluxSpectralFeaturizer()
    with pytest.raises(ValueError):
        f.featurize(np.zeros((3, SAMPLES_PER_FRAME)))
    with pytest.raises(ValueError):
        f.featurize(np.zeros(SAMPLES_PER_FRAME))
    with pytest.raises(ValueError):
        f.featurize(np.zeros((N_CHANNELS, SAMPLES_PER_FRAME + 1)))


# -- FLOPs charged: the analytic per-frame cost and the matched-budget ceiling ------------------------


def test_flops_per_frame_matches_the_documented_analytic_ledger() -> None:
    assert FLOPS_WINDOW == 1024
    assert FLOPS_RFFT == 51200
    assert FLOPS_POWER == 1539
    assert FLOPS_MEL == 2048
    assert FLOPS_COMPRESS == 4 * N_MEL == 256  # mu-law companding replaces the base plain log
    assert FLOPS_MAXFILT == (2 * MAX_FILTER_RADIUS + 1) * N_MEL == 192  # frequency max filter, new
    assert FLOPS_FLUX == 192
    assert (
        FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_MEL + FLOPS_COMPRESS + FLOPS_MAXFILT + FLOPS_FLUX
        == FLOPS_PER_COL_PER_CH
        == 56451
    )
    assert FLOPS_PER_FRAME == FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME == 1_129_020


def test_flops_for_frames_are_charged_and_stay_under_the_ceiling() -> None:
    f = SuperfluxSpectralFeaturizer()
    assert f.flops_for_frames(0) == 0
    assert f.flops_for_frames(1) == FLOPS_PER_FRAME
    # The SuperFlux front-end is strictly costlier per frame than the base, and that cost is charged.
    assert FLOPS_PER_FRAME > 1_121_340
    # Featurize over the real test set (22569 frames) stays well under the 6e10 lifecycle ceiling.
    assert f.flops_for_frames(22_569) == 25_480_852_380
    assert f.flops_for_frames(22_569) < FLOP_CEILING == 60_000_000_000


def test_flops_for_frames_rejects_bad_counts() -> None:
    f = SuperfluxSpectralFeaturizer()
    with pytest.raises(ValueError):
        f.flops_for_frames(-1)
    with pytest.raises(ValueError):
        f.flops_for_frames(True)


# -- seal present: the preregistration and (if built) the sealed artifact carry a reproducible seal ---


def test_sealed_prereg_carries_a_reproducible_seal() -> None:
    from mop.beds.starss23.superflux_spectral_prereg import build_featurizers_prereg

    body = build_featurizers_prereg(
        timestamp="2026-07-17T00:00:00Z",
        operating_firing_fraction=0.06,
        n_test_clips=21,
        n_test_onsets=538,
        train_onset_density=0.02,
        n_test_frames=22_569,
    )
    seal = body["canonical_sha256"]
    assert isinstance(seal, str) and len(seal) == 64
    without = {k: v for k, v in body.items() if k != "canonical_sha256"}
    assert canonical_sha256(without) == seal
    # The 3-featurizer Bonferroni wall is preregistered: n=5 cannot clear family-wise significance.
    assert body["multiplicity"]["n_variants"] == 3
    assert body["multiplicity"]["family_significance_reachable_at_n5"] is False


def test_sealed_artifact_seal_present_if_the_proof_has_been_produced() -> None:
    proof = Path("proof/STARSS23_ESCS_BED_superflux_spectral.json")
    if not proof.is_file():
        pytest.skip("sealed superflux_spectral artifact not produced on this host yet")
    artifact = json.loads(proof.read_bytes().decode("utf-8"))
    stored = artifact.get("seal")
    assert isinstance(stored, str) and len(stored) == 64
    body = {k: v for k, v in artifact.items() if k != "seal"}
    assert canonical_sha256(body) == stored
    assert artifact["schema"] == "mop-starss23-escs-bed-superflux-spectral/v1"
    assert artifact["featurizer"]["n_params"] == 0
    assert artifact["featurizer"]["flops_per_frame"] == 1_129_020
