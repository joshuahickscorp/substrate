"""Tests for the frozen spatial-DOA featurizer swap (F1 frozen-featurizer iteration wave).

These lock the invariants that make the spatial-DOA front-end an honest controlled experiment: it is a
frozen zero-trained-parameter featurizer that emits exactly 256 features, so it feeds the SAME unchanged
gate through the SAME sealed harness, referee, and controls, and its per-frame FLOPs are charged into the
matched-budget ledger identically to every arm and stay under the 6e10 ceiling. The featurizer is
deterministic and byte-reproducible, the sealed artifact carries a seal, and the independently authored
family verifier (importing no producer code) reproduces the sealed scores. Claim scope: deterministic
programmatic mechanics only; no capability claim.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import numpy as np
import pytest

from mop.beds.starss23 import FLOP_CEILING
from mop.beds.starss23 import spatial_doa_verifier as verifier_module
from mop.beds.starss23.feature_cache import CachedCorpus
from mop.beds.starss23.featurizer import FLOPS_PER_FRAME as FROZEN_FLOPS_PER_FRAME
from mop.beds.starss23.featurizer import FrozenFeaturizer
from mop.beds.starss23.featurizer_spatial_doa import (
    D_FEAT,
    FLOPS_PER_FRAME,
    N_BANDS,
    N_BINS,
    SpatialDoaFeaturizer,
    _band_edges,
)
from mop.beds.starss23.gate import D_FEAT as GATE_D_FEAT
from mop.beds.starss23.gate import D_IN
from mop.beds.starss23.schema import (
    N_CHANNELS,
    SAMPLES_PER_FRAME,
    Clip,
    ClipSplit,
    OnsetEvent,
)
from mop.beds.starss23.spatial_doa_producer import _flop_model
from mop.beds.starss23.spatial_doa_verifier import verify_artifact
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
)

# The committed real run's structural anchors (from proof/STARSS23_ESCS_BED_refractory_nms.json).
REAL_N_TEST_FRAMES = 22569
REAL_N_TRAIN_FRAMES = 25172
REAL_EPOCHS = 8


def _feature_block(seed: int, n_frames: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * n_frames))
    return SpatialDoaFeaturizer().featurize(audio)


# -- zero trained parameters, asserted the same way the base featurizer asserts it -----------------


def test_zero_trained_parameters() -> None:
    featurizer = SpatialDoaFeaturizer()
    assert featurizer.n_params() == 0
    # The base featurizer asserts zero the same way; the swap keeps that invariant.
    assert FrozenFeaturizer().n_params() == 0


def test_parameter_digest_hashes_only_frozen_dsp_constants() -> None:
    # Deterministic across instances, and distinct from the frozen log-mel front-end (different front-end).
    assert SpatialDoaFeaturizer().parameter_digest() == SpatialDoaFeaturizer().parameter_digest()
    assert SpatialDoaFeaturizer().parameter_digest() != FrozenFeaturizer().parameter_digest()
    # A different sample rate changes the frozen band grid, so it changes the digest.
    assert SpatialDoaFeaturizer(sample_rate=16000).parameter_digest() != SpatialDoaFeaturizer().parameter_digest()


# -- the output dimension is exactly 256, matching the unchanged gate contract ----------------------


def test_output_dim_matches_the_gate_contract() -> None:
    assert D_FEAT == 256 == GATE_D_FEAT
    assert D_IN == 264  # 256 features + 8 online scalars; the gate is not touched
    assert N_BANDS * 4 == D_FEAT


def test_featurize_emits_n_frames_by_256_float64() -> None:
    for n_frames in (1, 7, 30):
        feat = _feature_block(seed=n_frames, n_frames=n_frames)
        assert feat.shape == (n_frames, D_FEAT)
        assert feat.dtype == np.float64
        assert np.all(np.isfinite(feat))


def test_band_grid_is_non_empty_and_covers_all_bins() -> None:
    edges = _band_edges(24000)
    assert edges[0] == 0
    assert int(edges[-1]) == N_BINS
    assert bool(np.all(np.diff(edges) >= 1))  # every band owns at least one bin


def test_direction_cosines_are_unit_vectors_and_diffuseness_is_bounded() -> None:
    feat = _feature_block(seed=5, n_frames=40)
    dir_x = feat[:, :N_BANDS]
    dir_y = feat[:, N_BANDS : 2 * N_BANDS]
    dir_z = feat[:, 2 * N_BANDS : 3 * N_BANDS]
    diffuseness = feat[:, 3 * N_BANDS :]
    norm = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z
    assert np.allclose(norm, 1.0, atol=1e-9)  # az/el encoded as a unit direction, no wraparound
    assert float(diffuseness.min()) >= 0.0
    assert float(diffuseness.max()) <= 1.0


# -- deterministic, byte-reproducible ---------------------------------------------------------------


def test_feature_bytes_are_deterministic() -> None:
    rng = np.random.default_rng(11)
    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * 50))
    featurizer = SpatialDoaFeaturizer()
    a = featurizer.featurize(audio)
    b = featurizer.featurize(audio)
    assert np.array_equal(a, b)
    assert featurizer.feature_digest(a) == featurizer.feature_digest(b)
    # A fresh instance reproduces the same bytes (no hidden per-instance state).
    assert SpatialDoaFeaturizer().feature_digest(SpatialDoaFeaturizer().featurize(audio)) == featurizer.feature_digest(a)


def test_audio_length_and_channel_contract_is_enforced() -> None:
    featurizer = SpatialDoaFeaturizer()
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((N_CHANNELS, SAMPLES_PER_FRAME - 1)))  # not a whole frame
    with pytest.raises(ValueError):
        featurizer.featurize(np.zeros((3, SAMPLES_PER_FRAME)))  # wrong channel count


def test_white_noise_reads_as_highly_diffuse() -> None:
    # An incoherent field carries no single direction, so the DirAC diffuseness runs high. This is the
    # physical sanity check that the diffuseness channel means what it claims.
    feat = _feature_block(seed=99, n_frames=200)
    diffuseness = feat[:, 3 * N_BANDS :]
    assert float(diffuseness.mean()) > 0.6


# -- the featurizer feeds the unchanged gate directly, with no projection or truncation -------------


def test_features_feed_the_unchanged_gate_without_projection() -> None:
    from mop.beds.starss23.gate import CandidateGate, OnlineState
    from mop.science.gating import assemble_causal_inputs, causal_gate_trace

    feat = _feature_block(seed=3, n_frames=80)
    x = assemble_causal_inputs(feat, OnlineState.initial)
    assert x.shape == (80, D_IN)  # 256 native features + 8 online scalars, no projection
    gate = CandidateGate(seed=0)
    fires, probs = causal_gate_trace(gate, feat, 0.5, OnlineState.initial)
    assert probs.shape == (80,)
    assert isinstance(fires, list)


# -- FLOPs are analytic, host-independent, charged identically to every arm, and under the ceiling ---


def test_flops_per_frame_is_analytic_and_linear() -> None:
    featurizer = SpatialDoaFeaturizer()
    assert FLOPS_PER_FRAME == 1_124_177
    assert featurizer.flops_for_frames(1) == FLOPS_PER_FRAME
    assert featurizer.flops_for_frames(1000) == FLOPS_PER_FRAME * 1000
    assert featurizer.flops_for_frames(0) == 0
    with pytest.raises(ValueError):
        featurizer.flops_for_frames(-1)
    # Kept near the frozen front-end cost and well under the ~2.4M per-frame headroom (recipe gotcha F3).
    assert FLOPS_PER_FRAME < 2_400_000
    assert abs(FLOPS_PER_FRAME - FROZEN_FLOPS_PER_FRAME) < 10_000


def test_featurize_flops_are_charged_identically_to_every_arm() -> None:
    expected = FLOPS_PER_FRAME * REAL_N_TEST_FRAMES
    for kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE):
        model = _flop_model(kind, REAL_N_TEST_FRAMES, REAL_N_TRAIN_FRAMES, REAL_EPOCHS)
        assert model.featurize_flops == expected  # every arm pays the same DOA featurize charge
    # Only the candidate charges C_train; only candidate and rate-matched-random run the gate.
    candidate = _flop_model(ARM_CANDIDATE, REAL_N_TEST_FRAMES, REAL_N_TRAIN_FRAMES, REAL_EPOCHS)
    rmr = _flop_model(ARM_RATE_MATCHED_RANDOM, REAL_N_TEST_FRAMES, REAL_N_TRAIN_FRAMES, REAL_EPOCHS)
    always_on = _flop_model(ARM_ALWAYS_ON, REAL_N_TEST_FRAMES, REAL_N_TRAIN_FRAMES, REAL_EPOCHS)
    assert candidate.train_flops > 0
    assert rmr.train_flops == 0
    assert always_on.gate_infer_flops == 0
    # Candidate and rate-matched-random share byte-equal inference FLOPs at the same firing count.
    assert candidate.run_flops(1800) == rmr.run_flops(1800)


def test_candidate_lifecycle_stays_under_the_flop_ceiling() -> None:
    candidate = _flop_model(ARM_CANDIDATE, REAL_N_TEST_FRAMES, REAL_N_TRAIN_FRAMES, REAL_EPOCHS)
    # At the operating firing fraction (about 8 percent of 22569 frames) the full lifecycle is well under.
    firings = round(0.08 * REAL_N_TEST_FRAMES)
    lifecycle = candidate.lifecycle_flops(firings)
    assert lifecycle <= FLOP_CEILING == 60_000_000_000
    assert lifecycle > 0


# -- a sealed artifact carries a seal and the independent family verifier reproduces it -------------


def _synthetic_clip(clip_id: str, room_id: str, seed: int, n_frames: int) -> tuple[Clip, np.ndarray]:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((N_CHANNELS, SAMPLES_PER_FRAME * n_frames))
    features = SpatialDoaFeaturizer().featurize(audio)
    onset_frames = sorted(rng.choice(n_frames, size=max(2, n_frames // 12), replace=False).tolist())
    onsets = tuple(
        OnsetEvent(frame=int(f), class_id=int(i % 13), azimuth=0.0, elevation=0.0, distance=1.0)
        for i, f in enumerate(onset_frames)
    )
    clip = Clip(
        clip_id=clip_id,
        room_id=room_id,
        n_frames=n_frames,
        audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
        onsets=onsets,
    )
    return clip, features


def _synthetic_corpus() -> CachedCorpus:
    layout = [
        ("fold3_room00_a", "room00", "train"),
        ("fold3_room00_b", "room00", "train"),
        ("fold3_room01_a", "room01", "train"),
        ("fold3_room02_a", "room02", "val"),
        ("fold4_room03_a", "room03", "test"),
        ("fold4_room04_a", "room04", "test"),
    ]
    clips: list[Clip] = []
    features_by_clip: dict[str, np.ndarray] = {}
    part: dict[str, list[Clip]] = {"train": [], "val": [], "test": []}
    for i, (clip_id, room_id, where) in enumerate(layout):
        clip, feat = _synthetic_clip(clip_id, room_id, seed=100 + i, n_frames=64)
        clips.append(clip)
        features_by_clip[clip_id] = feat
        part[where].append(clip)
    split = ClipSplit(
        train=tuple(part["train"]),
        val=tuple(part["val"]),
        test=tuple(part["test"]),
        detail={"split_rule": "synthetic room-disjoint fixture for the seal test"},
    )
    return CachedCorpus(
        cache_key="synthetic-spatial-doa-seal-test",
        cache_dir=pathlib.Path("/dev/null"),
        clips=tuple(clips),
        features_by_clip=features_by_clip,
        split=split,
        featurizer_digest=SpatialDoaFeaturizer().parameter_digest(),
        flops_per_frame=FLOPS_PER_FRAME,
        truncations=(),
        foa_root="synthetic",
        metadata_root="synthetic",
        max_frames=None,
        n_val_rooms=1,
    )


def test_sealed_artifact_has_a_seal_and_is_independently_reproduced(tmp_path) -> None:
    from mop.beds.starss23.real_artifact import RealBedConfig
    from mop.beds.starss23.spatial_doa_prereg import (
        build_featurizers_prereg,
    )
    from mop.beds.starss23.spatial_doa_producer import build_spatial_doa_artifact
    from mop.substrate.events import write_canonical_json

    corpus = _synthetic_corpus()
    prereg = build_featurizers_prereg(
        timestamp="2026-07-17T00:00:00Z",
        operating_firing_fraction=0.08,
        n_test_clips=corpus.n_test_clips(),
        n_test_onsets=corpus.n_test_onsets(),
        train_onset_density=corpus.train_onset_density(),
        n_test_frames=corpus.n_test_frames(),
    )
    prereg_path = tmp_path / "spatial_doa.prereg.json"
    write_canonical_json(prereg, prereg_path)

    bed = build_spatial_doa_artifact(
        timestamp="2026-07-17T00:00:00Z",
        corpus=corpus,
        config=RealBedConfig(noisy_tv_frames=200),
        featurizers_prereg_path=prereg_path,
    )

    # The sealed artifact carries a seal that is a re-hash of its own body.
    assert "seal" in bed.artifact and isinstance(bed.seal, str) and len(bed.seal) == 64
    assert bed.verdict in ("null", "mechanics-ok")

    # The independently authored family verifier (no producer imports) reproduces the sealed scores.
    result = verify_artifact(bed.artifact)
    assert result.seal_intact
    assert result.schema_ok
    assert result.scores_reproduced
    assert result.stats_reproduced
    assert result.honesty_ok
    assert result.independent_referee_reproduction
    # A single real run never self-promotes to scientific confirmation.
    assert not result.independent_scientific_confirmation


def test_family_verifier_imports_no_producer_code() -> None:
    # The adversarial verifier must share no code with the producer: a real triangulation, not a shared bug.
    allowed = {
        "__future__", "hashlib", "itertools", "json", "math",
        "dataclasses", "typing", "collections", "argparse",
    }
    source = pathlib.Path(verifier_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    relative: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                relative.append(node.module or "<relative>")
            else:
                roots.add((node.module or "").split(".")[0])
    assert not relative, f"relative imports would pull producer code: {relative}"
    assert "mop" not in roots, "verifier must not import anything under mop"
    assert not (roots - allowed), f"verifier imported unexpected modules: {roots - allowed}"


# -- the sealed featurizer preregistration is a 3-family Bonferroni wall -----------------------------


def test_featurizer_prereg_is_a_three_family_bonferroni_wall() -> None:
    from mop.beds.starss23.spatial_doa_prereg import build_featurizers_prereg
    from mop.substrate.events import canonical_sha256

    prereg = build_featurizers_prereg(
        timestamp="2026-07-17T00:00:00Z",
        operating_firing_fraction=0.08,
        n_test_clips=21,
        n_test_onsets=538,
        train_onset_density=0.0848,
        n_test_frames=22569,
    )
    assert prereg["sesoi"]["sesoi_f1"] == 0.05
    assert prereg["multiplicity"]["n_featurizers"] == 3
    # At three families and five seeds, family-wise significance is unreachable: a preregistered wall.
    assert prereg["multiplicity"]["family_significance_reachable_at_n5"] is False
    assert "spatial_doa" in [f["featurizer_id"] for f in prereg["featurizers"]]
    # The prereg is self-sealed and re-sealing reproduces the digest.
    body = {k: v for k, v in prereg.items() if k != "canonical_sha256"}
    assert canonical_sha256(body) == prereg["canonical_sha256"]
