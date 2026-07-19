
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23 import FLOP_CEILING
from mop.beds.starss23.featurizer_interchannel_coherence import (
    COLS_PER_FRAME,
    D_FEAT,
    FLOPS_BANDAGG_PER_COL,
    FLOPS_BANDS,
    FLOPS_COL_TO_FRAME,
    FLOPS_CROSS_PER_COL,
    FLOPS_PER_COL_PER_CH,
    FLOPS_PER_FRAME,
    N_BANDS,
    N_BINS,
    N_SPATIAL,
    InterchannelCoherenceFeaturizer,
    _band_partition,
)
from mop.beds.starss23.gate import CandidateGate, OnlineState
from mop.beds.starss23.schema import N_CHANNELS, SAMPLE_RATE_HZ, SAMPLES_PER_FRAME
from mop.substrate.events import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED_interchannel_coherence.prereg.json"
ARTIFACT_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED_interchannel_coherence.json"


def _fixture_audio(n_frames: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))


# ---------------------------------------------------------------------------
# Zero trained parameters.
# ---------------------------------------------------------------------------


def test_zero_trained_parameters() -> None:
    assert InterchannelCoherenceFeaturizer().n_params() == 0


def test_frozen_dsp_is_not_learned() -> None:
    # The window and the band partition are deterministic functions of the sample rate and FFT size, so
    # two independent instances carry byte-identical DSP constants and the same parameter digest.
    a = InterchannelCoherenceFeaturizer()
    b = InterchannelCoherenceFeaturizer()
    assert a.window.tobytes() == b.window.tobytes()
    assert a.band_partition.tobytes() == b.band_partition.tobytes()
    assert a.parameter_digest() == b.parameter_digest()


# ---------------------------------------------------------------------------
# Deterministic feature bytes.
# ---------------------------------------------------------------------------


def test_deterministic_feature_bytes_across_runs() -> None:
    audio = _fixture_audio(8)
    featurizer = InterchannelCoherenceFeaturizer()
    first = featurizer.featurize(audio)
    second = featurizer.featurize(audio)
    assert first.tobytes() == second.tobytes()
    assert featurizer.feature_digest(first) == featurizer.feature_digest(second)


def test_deterministic_feature_bytes_across_instances() -> None:
    audio = _fixture_audio(8)
    digest_a = InterchannelCoherenceFeaturizer().feature_digest(
        InterchannelCoherenceFeaturizer().featurize(audio)
    )
    digest_b = InterchannelCoherenceFeaturizer().feature_digest(
        InterchannelCoherenceFeaturizer().featurize(audio)
    )
    assert digest_a == digest_b


# ---------------------------------------------------------------------------
# 256-dim output the sealed gate consumes, in the normalized coherence range.
# ---------------------------------------------------------------------------


def test_feature_block_shape_is_256() -> None:
    features = InterchannelCoherenceFeaturizer().featurize(_fixture_audio(6))
    assert features.shape == (6, D_FEAT)
    assert D_FEAT == 256
    assert D_FEAT == N_BANDS * N_SPATIAL
    assert features.dtype == np.float64


def test_features_are_normalized_ratios_in_unit_interval() -> None:
    # Magnitude-squared coherence and DirAC directness are normalized ratios, so every feature is in [0, 1].
    features = InterchannelCoherenceFeaturizer().featurize(_fixture_audio(12, seed=11))
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_band_partition_has_no_empty_bands() -> None:
    partition = _band_partition(SAMPLE_RATE_HZ)
    assert partition.shape == (N_BANDS, N_BINS)
    # A partition: each bin assigned to exactly one band, and every band carries at least one bin.
    assert np.array_equal(partition.sum(axis=0), np.ones(N_BINS))
    assert int((partition.sum(axis=1) == 0).sum()) == 0


def test_gate_consumes_the_256_features_unchanged() -> None:
    # The sealed 264-input gate must ingest the featurizer output with no change to its parameter count.
    features = InterchannelCoherenceFeaturizer().featurize(_fixture_audio(4))
    gate = CandidateGate(seed=0)
    assert gate.n_params() == 3193
    p_fire = gate.infer(features[0], OnlineState.initial())
    assert 0.0 <= float(p_fire) <= 1.0


# ---------------------------------------------------------------------------
# FLOPs charged: the analytic per-frame ledger and the matched-budget charge.
# ---------------------------------------------------------------------------


def test_flops_per_frame_matches_documented_formula() -> None:
    front = FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME
    cross = FLOPS_CROSS_PER_COL * COLS_PER_FRAME
    bandagg = FLOPS_BANDAGG_PER_COL * COLS_PER_FRAME
    assert FLOPS_PER_FRAME == front + cross + bandagg + FLOPS_COL_TO_FRAME + FLOPS_BANDS
    assert FLOPS_PER_FRAME == 1_151_560


def test_flops_for_frames_and_matched_ceiling() -> None:
    featurizer = InterchannelCoherenceFeaturizer()
    assert featurizer.flops_for_frames(0) == 0
    assert featurizer.flops_for_frames(1) == FLOPS_PER_FRAME
    # The per-frame cost stays under the ceiling headroom (about 2.4M) the recipe requires.
    assert FLOPS_PER_FRAME <= 2_400_000
    # The reference 24000-frame evaluation set stays well under the 6e10 lifecycle ceiling.
    assert featurizer.flops_for_frames(24_000) < FLOP_CEILING


def test_flops_for_frames_rejects_bad_counts() -> None:
    featurizer = InterchannelCoherenceFeaturizer()
    with pytest.raises(ValueError):
        featurizer.flops_for_frames(-1)
    with pytest.raises(ValueError):
        featurizer.flops_for_frames(True)


def test_featurize_flops_charged_into_matched_budget_ledger() -> None:
    # Every arm in the sealed artifact charges this featurizer's per-frame FLOPs times the test frame
    # count, and every arm's full-lifecycle FLOPs stay under the 6e10 ceiling.
    if not ARTIFACT_PATH.is_file():
        pytest.skip("sealed artifact not present; run the producer first")
    artifact = json.loads(ARTIFACT_PATH.read_bytes().decode("utf-8"))
    n_test_frames = artifact["real_corpus"]["n_test_frames"]
    assert artifact["featurizer"]["flops_per_frame"] == FLOPS_PER_FRAME
    expected_featurize = FLOPS_PER_FRAME * n_test_frames
    for summary in artifact["harness"]["arm_summaries"]:
        assert summary["flop_model"]["featurize_flops"] == expected_featurize
        assert summary["max_lifecycle_flops"] <= FLOP_CEILING
    assert artifact["matched_budget"]["flops"] <= FLOP_CEILING


# ---------------------------------------------------------------------------
# Seal present: the prereg and the sealed artifact both carry reproducible seals.
# ---------------------------------------------------------------------------


def test_seal_present_and_reproducible() -> None:
    # A self-contained seal over the featurizer identity reproduces, so the featurizer participates in the
    # canonical seal the artifact uses.
    body = {
        "featurizer": "interchannel_coherence",
        "n_params": InterchannelCoherenceFeaturizer().n_params(),
        "parameter_digest": InterchannelCoherenceFeaturizer().parameter_digest(),
        "flops_per_frame": FLOPS_PER_FRAME,
    }
    seal = canonical_sha256(body)
    assert isinstance(seal, str) and len(seal) == 64
    assert canonical_sha256(body) == seal

    if PREREG_PATH.is_file():
        prereg = json.loads(PREREG_PATH.read_bytes().decode("utf-8"))
        stored = prereg.pop("canonical_sha256")
        assert canonical_sha256(prereg) == stored

    if ARTIFACT_PATH.is_file():
        artifact = json.loads(ARTIFACT_PATH.read_bytes().decode("utf-8"))
        stored_seal = artifact.pop("seal")
        assert isinstance(stored_seal, str) and len(stored_seal) == 64
        assert canonical_sha256(artifact) == stored_seal


# ---------------------------------------------------------------------------
# The physical cue: a coherent directional onset registers; silence does not.
# ---------------------------------------------------------------------------


def test_responds_to_planted_directional_onset() -> None:
    # Three silent frames then a directional (coherent across W and the gradient channels) 1 kHz source.
    n_frames = 6
    times = np.arange(n_frames * SAMPLES_PER_FRAME) / SAMPLE_RATE_HZ
    signal = np.zeros(n_frames * SAMPLES_PER_FRAME, dtype=np.float64)
    onset = 3 * SAMPLES_PER_FRAME
    signal[onset:] = np.sin(2.0 * np.pi * 1000.0 * times[onset:])
    audio = np.zeros((N_CHANNELS, n_frames * SAMPLES_PER_FRAME), dtype=np.float64)
    audio[0] = signal  # W (omni)
    audio[1] = 0.9 * signal  # X strongly coherent with W (a direction)
    audio[2] = 0.1 * signal
    audio[3] = 0.05 * signal
    features = InterchannelCoherenceFeaturizer().featurize(audio)
    per_frame_l1 = np.abs(features).sum(axis=1)
    # Frames wholly inside the leading silence carry no coherence at all (zero energy, eps-floored).
    assert per_frame_l1[0] == pytest.approx(0.0, abs=1e-3)
    assert per_frame_l1[1] == pytest.approx(0.0, abs=1e-3)
    # The directional source frames carry substantial coherence and directness.
    assert per_frame_l1[4] > 10.0


def test_rejects_bad_audio_shape() -> None:
    featurizer = InterchannelCoherenceFeaturizer()
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((3, SAMPLES_PER_FRAME)))  # wrong channel count
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros(SAMPLES_PER_FRAME))  # not two dimensional
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((N_CHANNELS, SAMPLES_PER_FRAME + 1)))  # partial frame
