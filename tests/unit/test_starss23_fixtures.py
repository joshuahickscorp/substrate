
from __future__ import annotations

import numpy as np
import pytest

from mop.beds.starss23.featurizer import N_MEL, FrozenFeaturizer
from mop.beds.starss23.fixtures import (
    NUISANCE_BAND_HZ,
    REGIME_FAVORABLE,
    REGIME_NULL,
    SIGNAL_BAND_HZ,
    SyntheticStarssConfig,
    generate_clip,
)
from mop.beds.starss23.referee import score_clip
from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME


def _band_bins(featurizer: FrozenFeaturizer, band: tuple[float, float]) -> list[int]:
    fb = featurizer.filterbank
    bin_freqs = np.linspace(0.0, featurizer.sample_rate / 2.0, fb.shape[1])
    peak_freq = [bin_freqs[int(np.argmax(fb[m]))] for m in range(N_MEL)]
    return [m for m in range(N_MEL) if band[0] <= peak_freq[m] <= band[1]]


def _band_flux(features: np.ndarray, bins: list[int]) -> np.ndarray:
    idx = np.array(bins)
    return sum(features[:, ch * N_MEL + idx].sum(axis=1) for ch in range(N_CHANNELS))


def _oracle_f1(scores: np.ndarray, onsets: list[int]) -> float:
    k = len(onsets)
    top = sorted(np.argsort(scores)[-k:].tolist())
    tp, fp, fn = score_clip(onsets, top)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _config(seed: int) -> SyntheticStarssConfig:
    return SyntheticStarssConfig(clip_seconds=6.0, onsets_per_clip=5, nuisance_per_clip=6, base_seed=seed)


def test_generate_clip_is_byte_reproducible() -> None:
    clip_a, audio_a = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(0)
    )
    clip_b, audio_b = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(0)
    )
    assert np.array_equal(audio_a, audio_b)
    assert clip_a.audio_sha256 == clip_b.audio_sha256
    assert clip_a.digest() == clip_b.digest()
    _clip_c, audio_c = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(1)
    )
    assert not np.array_equal(audio_a, audio_c)


def test_audio_shape_matches_the_frame_grid() -> None:
    config = _config(0)
    _clip, audio = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=config
    )
    assert audio.shape == (N_CHANNELS, config.n_frames * SAMPLES_PER_FRAME)


def test_favorable_onsets_have_integer_direction_of_arrival() -> None:
    clip, _audio = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(0)
    )
    assert clip.onsets
    for onset in clip.onsets:
        assert onset.azimuth == round(onset.azimuth)
        assert onset.elevation == round(onset.elevation)
        assert onset.distance == round(onset.distance)


def test_planted_onsets_are_recovered_by_a_signal_band_oracle() -> None:
    featurizer = FrozenFeaturizer()
    sig_bins = _band_bins(featurizer, SIGNAL_BAND_HZ)
    for seed in range(3):
        clip, audio = generate_clip(
            clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(seed)
        )
        features = featurizer.featurize(audio)
        onsets = list(clip.onset_frames)
        assert _oracle_f1(_band_flux(features, sig_bins), onsets) == 1.0


def test_bare_total_flux_threshold_cannot_solve_it_off_ceiling() -> None:
    featurizer = FrozenFeaturizer()
    sig_bins = _band_bins(featurizer, SIGNAL_BAND_HZ)
    nui_bins = _band_bins(featurizer, NUISANCE_BAND_HZ)
    assert set(sig_bins).isdisjoint(nui_bins)
    clip, audio = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(0)
    )
    features = featurizer.featurize(audio)
    onsets = list(clip.onset_frames)
    signal_oracle = _oracle_f1(_band_flux(features, sig_bins), onsets)
    total_flux = _oracle_f1(features.sum(axis=1), onsets)
    assert total_flux < signal_oracle
    assert total_flux < 0.75


def test_rooms_carry_distinct_correlated_backgrounds() -> None:
    _a_clip, a = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_FAVORABLE, config=_config(0)
    )
    _b_clip, b = generate_clip(
        clip_id="fold3_room0_mix000", room_id="room01", regime=REGIME_FAVORABLE, config=_config(0)
    )
    assert not np.array_equal(a, b)


def test_null_regime_onsets_are_not_recoverable_strong_null() -> None:
    featurizer = FrozenFeaturizer()
    sig_bins = _band_bins(featurizer, SIGNAL_BAND_HZ)
    f1s = []
    for seed in range(5):
        clip, audio = generate_clip(
            clip_id="fold3_room0_mix000", room_id="room00", regime=REGIME_NULL, config=_config(seed)
        )
        features = featurizer.featurize(audio)
        f1s.append(_oracle_f1(_band_flux(features, sig_bins), list(clip.onset_frames)))
    assert max(f1s) < 1.0
    assert sum(f1s) / len(f1s) < 0.6


def test_generate_clip_refuses_unknown_regime() -> None:
    with pytest.raises(ValueError):
        generate_clip(clip_id="fold3_room0_mix000", room_id="room00", regime="bogus", config=_config(0))
