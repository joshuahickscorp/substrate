
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mop.beds.starss23 import FLOP_CEILING
from mop.beds.starss23.featurizer import (
    COLS_PER_FRAME,
    D_FEAT,
    FLOPS_FLUX,
    FLOPS_LOG,
    FLOPS_MEL,
    FLOPS_PER_COL_PER_CH,
    FLOPS_PER_FRAME,
    FLOPS_POWER,
    FLOPS_RFFT,
    FLOPS_WINDOW,
    HOP,
    N_BINS,
    N_FFT,
    N_MEL,
    WINDOW,
    FrozenFeaturizer,
    hann_window,
    mel_filterbank,
)
from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME


def _fixture_audio(n_frames: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))


def test_zero_trained_parameters() -> None:
    assert FrozenFeaturizer().n_params() == 0


def test_frozen_dsp_grid_constants() -> None:
    assert WINDOW == 1024
    assert N_FFT == 1024
    assert N_BINS == 513  # one-sided rFFT bins, n_fft // 2 + 1
    assert HOP == 480
    assert SAMPLES_PER_FRAME == 2400  # 100 ms at 24 kHz
    assert COLS_PER_FRAME == 5  # 2400 // 480
    assert N_MEL == 64
    assert D_FEAT == 256  # 64 mel by 4 channels
    assert D_FEAT == N_MEL * N_CHANNELS


def test_flop_component_constants_match_documented_formula() -> None:
    # docs/ESCS_DEEP_RESEARCH.md immediate design implications, per column per channel.
    assert FLOPS_WINDOW == 1024
    assert FLOPS_RFFT == 51200  # 5 * 1024 * log2(1024) = 5 * 1024 * 10
    assert FLOPS_POWER == 1539  # 3 * 513
    assert FLOPS_MEL == 2048  # 2 * 1024 nominal nonzeros
    assert FLOPS_LOG == 64
    assert FLOPS_FLUX == 192  # 3 * 64
    assert FLOPS_PER_COL_PER_CH == 56067
    assert (
        FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_MEL + FLOPS_LOG + FLOPS_FLUX
        == FLOPS_PER_COL_PER_CH
    )


def test_flops_per_frame_matches_documented_formula() -> None:
    assert FLOPS_PER_FRAME == 1_121_340  # 56067 * 4 channels * 5 columns
    assert FLOPS_PER_FRAME == FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME


def test_flops_for_frames_and_matched_ceiling() -> None:
    featurizer = FrozenFeaturizer()
    assert featurizer.flops_for_frames(0) == 0
    assert featurizer.flops_for_frames(1) == FLOPS_PER_FRAME
    # The reference 40 clip by 60 s evaluation set is 24000 frames and stays under the 6e10 ceiling.
    full_set_frames = 40 * 60 * 10
    assert full_set_frames == 24_000
    assert featurizer.flops_for_frames(full_set_frames) == 26_912_160_000
    assert featurizer.flops_for_frames(full_set_frames) < FLOP_CEILING
    assert FLOP_CEILING == 60_000_000_000


def test_flops_for_frames_rejects_bad_counts() -> None:
    featurizer = FrozenFeaturizer()
    with pytest.raises(ValueError):
        featurizer.flops_for_frames(-1)
    with pytest.raises(ValueError):
        featurizer.flops_for_frames(True)


def test_feature_block_shape() -> None:
    features = FrozenFeaturizer().featurize(_fixture_audio(6))
    assert features.shape == (6, D_FEAT)
    assert features.dtype == np.float64


def test_byte_reproducible_across_runs() -> None:
    audio = _fixture_audio(8)
    featurizer = FrozenFeaturizer()
    first = featurizer.featurize(audio)
    second = featurizer.featurize(audio)
    assert first.tobytes() == second.tobytes()
    assert featurizer.feature_digest(first) == featurizer.feature_digest(second)


def test_byte_reproducible_across_instances() -> None:
    audio = _fixture_audio(8)
    digest_a = FrozenFeaturizer().feature_digest(FrozenFeaturizer().featurize(audio))
    digest_b = FrozenFeaturizer().feature_digest(FrozenFeaturizer().featurize(audio))
    assert digest_a == digest_b


def test_frozen_parameter_digest_is_stable() -> None:
    # The window and filterbank are fixed DSP, so two instances have byte-identical parameters.
    assert FrozenFeaturizer().parameter_digest() == FrozenFeaturizer().parameter_digest()


def test_shared_spectral_primitives_pin_both_mel_resolutions() -> None:
    def digest(array: np.ndarray) -> str:
        return hashlib.sha256(array.astype("<f8").tobytes()).hexdigest()

    assert digest(hann_window()) == "92c4b7d76381d84a6eefd52fcd7ffba885abeedc69b512944ec270888f9c0f28"
    assert digest(mel_filterbank(24_000)) == "7ddf24c34f6ad2dfeaf0448761853063e252c4044d15e8b4d66d0c5760f13766"
    assert digest(mel_filterbank(24_000, 32, 513)) == (
        "13130b3b6646a543ff9b17d8a110256f3fe02e5b1a1e55ee9d0d67407f717db9"
    )


def test_mel_filterbank_is_sparse_and_deterministic() -> None:
    filterbank = FrozenFeaturizer().filterbank
    assert filterbank.shape == (N_MEL, N_BINS)
    nnz = int(np.count_nonzero(filterbank))
    # Overlapping triangular filters give about two nonzeros per rFFT bin, near the nominal 1024.
    assert 800 <= nnz <= 1200
    other = FrozenFeaturizer().filterbank
    assert filterbank.tobytes() == other.tobytes()


def test_hann_window_shape_and_edges() -> None:
    window = FrozenFeaturizer().window
    assert window.shape == (WINDOW,)
    # Periodic Hann: first sample is exactly zero, values stay within [0, 1].
    assert window[0] == pytest.approx(0.0, abs=1e-12)
    assert float(window.min()) >= 0.0
    assert float(window.max()) <= 1.0


def test_flux_responds_to_planted_onset() -> None:
    # Three silent frames then a steady 1 kHz tone. Flux is a change detector: silence gives exactly
    # zero, the silence-to-tone transition gives a strong positive flux, steady tone gives far less.
    n_frames = 6
    sample_rate = 24_000
    times = np.arange(n_frames * SAMPLES_PER_FRAME) / sample_rate
    signal = np.zeros(n_frames * SAMPLES_PER_FRAME, dtype=np.float64)
    onset = 3 * SAMPLES_PER_FRAME
    signal[onset:] = np.sin(2.0 * np.pi * 1000.0 * times[onset:])
    gains = np.array([1.0, 0.7, 0.5, 0.3])[:, None]
    audio = np.repeat(signal[None, :], N_CHANNELS, axis=0) * gains
    features = FrozenFeaturizer().featurize(audio)
    per_frame_l1 = np.abs(features).sum(axis=1)
    # Frames wholly inside the leading silence carry no flux at all.
    assert per_frame_l1[0] == pytest.approx(0.0, abs=1e-9)
    assert per_frame_l1[1] == pytest.approx(0.0, abs=1e-9)
    # The transition frame (its overlapping windows first reach the onset) carries strong flux.
    assert per_frame_l1[2] > 100.0
    # The steady-tone frame that follows carries much less flux than the transition frame.
    assert per_frame_l1[2] > per_frame_l1[3]


def test_rejects_bad_audio_shape() -> None:
    featurizer = FrozenFeaturizer()
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((3, SAMPLES_PER_FRAME)))  # wrong channel count
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros(SAMPLES_PER_FRAME))  # not two dimensional
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((N_CHANNELS, SAMPLES_PER_FRAME + 1)))  # partial frame
