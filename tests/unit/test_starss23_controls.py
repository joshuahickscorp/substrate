"""Tests for component 4: the three honest controls of the STARSS23 ESCS bed.

The task-critical properties, one test each:
- rate-matched-random fires the SAME count as the candidate, matched per seed and per clip, with the
  positions permuted (a different WHERE, an identical how-much);
- best-single is deterministic;
- a gate that preferentially fires on the injected noisy-TV channel FAILS the at-chance check while an
  honest gate passes.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from mop.beds.starss23 import controls as c
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_RATE_MATCHED_RANDOM,
)

# ---------------------------------------------------------------------------
# (a) rate-matched-random
# ---------------------------------------------------------------------------


def test_rate_matched_random_identical_count_matched_per_seed_permuted_positions() -> None:
    candidate = [3, 10, 17, 40, 55]
    n_frames = 60

    fires = c.rate_matched_random_fires(candidate, n_frames, seed=1, clip_id="clipA")
    # Identical firing COUNT to the candidate.
    assert len(fires) == len(candidate)
    # Matched per seed and per clip: byte-reproducible.
    assert fires == c.rate_matched_random_fires(candidate, n_frames, seed=1, clip_id="clipA")
    # Different seed gives a different draw; a different clip does too.
    assert fires != c.rate_matched_random_fires(candidate, n_frames, seed=2, clip_id="clipA")
    assert fires != c.rate_matched_random_fires(candidate, n_frames, seed=1, clip_id="clipB")
    # Positions are permuted, not the candidate's own positions, and stay in range and unique.
    assert fires != sorted(candidate)
    assert all(0 <= frame < n_frames for frame in fires)
    assert len(set(fires)) == len(fires)


def test_rate_matched_random_matches_count_across_many_clips() -> None:
    # The per-clip firing COUNT must match exactly so the pooled compute histogram is identical.
    rng = np.random.default_rng(0)
    control = c.RateMatchedRandomControl(seed=7)
    for clip_index in range(20):
        n_frames = int(rng.integers(20, 80))
        k = int(rng.integers(0, n_frames // 2))
        candidate = sorted(int(f) for f in rng.choice(n_frames, size=k, replace=False))
        fires = control.fires_for_clip(candidate, n_frames, f"clip{clip_index}")
        assert len(fires) == len(candidate)


def test_rate_matched_random_saturates_when_budget_reaches_frame_count() -> None:
    # When the candidate fires on every frame, the only rate-matched set is every frame.
    assert c.rate_matched_random_fires(list(range(12)), 12, seed=0, clip_id="c") == list(range(12))


def test_rate_matched_random_arm_kind_is_the_harness_control_name() -> None:
    assert c.RateMatchedRandomControl(seed=0).arm_kind == ARM_RATE_MATCHED_RANDOM


# ---------------------------------------------------------------------------
# (b) always-on and best-single
# ---------------------------------------------------------------------------


def test_always_on_fires_every_frame() -> None:
    assert c.always_on_fires(5) == [0, 1, 2, 3, 4]
    assert c.AlwaysOnControl().fires_for_clip(4) == [0, 1, 2, 3]
    assert c.AlwaysOnControl().arm_kind == ARM_ALWAYS_ON


def _toy_val_clip(onset_frames: list[int], n_frames: int, rng: np.random.Generator):
    features = np.abs(rng.standard_normal((n_frames, 256))) * 0.05
    for frame in onset_frames:
        features[frame] += 4.0  # a strong flux spike at the onset frame
    return features, onset_frames


def test_best_single_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    val = [_toy_val_clip([5, 15, 25], 30, rng), _toy_val_clip([3, 20], 30, rng)]

    first = c.tune_best_single_threshold(val)
    second = c.tune_best_single_threshold(val)
    assert first == second  # tuning is a deterministic function of the val features

    control = c.BestSingleControl.tuned(val)
    test_features, _ = _toy_val_clip([7, 14], 30, rng)
    fires_a = control.fires_for_clip(test_features)
    fires_b = control.fires_for_clip(test_features)
    assert fires_a == fires_b  # applying the threshold is deterministic
    assert control.arm_kind == ARM_BEST_SINGLE


def test_best_single_threshold_separates_flux_spikes() -> None:
    rng = np.random.default_rng(4)
    val = [_toy_val_clip([4, 12, 20], 26, rng)]
    control = c.BestSingleControl.tuned(val)
    features, onsets = _toy_val_clip([6, 18], 26, rng)
    # The tuned total-flux threshold recovers the planted high-flux frames on held-out content.
    assert control.fires_for_clip(features) == onsets


# ---------------------------------------------------------------------------
# (c) noisy-TV
# ---------------------------------------------------------------------------


class _ScheduleGate:
    """An honest gate whose firing is uncorrelated with the input: it fires on a fixed schedule."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.index = -1

    def __call__(self, _row: np.ndarray) -> bool:
        self.index += 1
        return self.index % self.period == 0


def test_noisy_tv_honest_gate_passes_at_chance() -> None:
    # A gate that fires uncorrelated with novelty fires on the noise at its base rate, so it is at chance.
    base_rate = 0.1
    result = c.noisy_tv_probe(_ScheduleGate(period=10), base_rate, seed=7, n_noise_frames=1000)
    assert result.at_chance is True
    assert result.firing_rate_on_noise <= base_rate + result.tolerance


def test_noisy_tv_coherent_signal_gate_passes_at_chance() -> None:
    # A gate keyed on a coherent signal-band condition almost never fires on incoherent pure noise.
    def honest(row: np.ndarray) -> bool:
        band = row[:64]
        return bool(band.mean() > 2.0 and band.min() > 0.5)

    result = c.noisy_tv_probe(honest, 0.1, seed=11, n_noise_frames=1000)
    assert result.at_chance is True
    assert result.firing_rate_on_noise == pytest.approx(0.0, abs=1e-9)


def test_noisy_tv_noise_chasing_gate_fails_at_chance() -> None:
    # A gate rewarded by raw RND novelty fires preferentially on the high-novelty pure-noise channel.
    rng = np.random.default_rng(0)
    normal = np.abs(rng.standard_normal((500, 256))) * 0.1
    base_rate = 0.1
    chaser = c.noise_chasing_fire_fn(normal, base_rate=base_rate, seed=7)

    # On normal content it fires at about the base rate.
    normal_rate = c.firing_rate_on_frames(chaser, normal)
    assert normal_rate == pytest.approx(base_rate, abs=0.02)

    # On the injected pure-aleatoric channel it fires preferentially, so it fails the at-chance check.
    result = c.noisy_tv_probe(chaser, base_rate, seed=7, n_noise_frames=1000)
    assert result.at_chance is False
    assert result.firing_rate_on_noise > base_rate + result.tolerance


def test_at_chance_band_is_one_sided_on_preferential_firing() -> None:
    # Firing at or below the base rate plus tolerance is honest; firing preferentially above fails.
    assert c.at_chance(0.10, 0.10, tolerance=0.05) is True
    assert c.at_chance(0.00, 0.10, tolerance=0.05) is True  # firing less on noise is honest
    assert c.at_chance(0.14, 0.10, tolerance=0.05) is True
    assert c.at_chance(0.20, 0.10, tolerance=0.05) is False  # preferential firing fails


def test_rnd_target_novelty_is_deterministic_and_higher_on_noise() -> None:
    target = c.RndTarget(seed=5)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((100, 256))
    normal = np.abs(rng.standard_normal((100, 256))) * 0.1
    # Byte-reproducible in the seed.
    assert np.array_equal(target.novelty(noise), c.RndTarget(seed=5).novelty(noise))
    # Pure aleatoric input drives raw novelty higher than small-magnitude normal content: the trap.
    assert target.novelty(noise).mean() > target.novelty(normal).mean()


def test_pure_aleatoric_channel_is_deterministic_with_the_expected_shape() -> None:
    a = c.pure_aleatoric_channel(9, 128)
    b = c.pure_aleatoric_channel(9, 128)
    assert a.shape == (128, 256)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c.pure_aleatoric_channel(10, 128))


def test_noisy_tv_result_seal_is_stable() -> None:
    result = c.noisy_tv_probe(_ScheduleGate(period=10), 0.1, seed=7, n_noise_frames=500)
    assert result.digest() == c.NoisyTvResult(**{
        "firing_rate_on_noise": result.firing_rate_on_noise,
        "base_rate": result.base_rate,
        "tolerance": result.tolerance,
        "n_noise_frames": result.n_noise_frames,
        "at_chance": result.at_chance,
    }).digest()


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_controls_refuse_malformed_inputs() -> None:
    with pytest.raises(c.ControlRefusal):
        c.rate_matched_random_fires([0, 1], 0, seed=0, clip_id="c")
    with pytest.raises(c.ControlRefusal):
        c.rate_matched_random_fires([99], 10, seed=0, clip_id="c")  # frame out of range
    with pytest.raises(c.ControlRefusal):
        c.always_on_fires(-1)
    with pytest.raises(c.ControlRefusal):
        c.tune_best_single_threshold([])
    with pytest.raises(c.ControlRefusal):
        c.at_chance(1.5, 0.1)
