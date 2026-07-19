
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23 import count_repro_featurizer_estimator_verifier as V
from mop.beds.starss23.count_estimator import FrozenCountEstimator
from mop.beds.starss23.count_featurizer import FrozenCountFeaturizer
from mop.beds.starss23.count_repro_featurizer_estimator_estimator import (
    BETA,
    FLOPS_PER_REESTIMATE,
    MAX_ESTIMABLE_SOURCES,
    ReproCountEstimator,
)
from mop.beds.starss23.count_repro_featurizer_estimator_featurizer import (
    D_CFEAT,
    FLOPS_PER_FRAME_COUNT,
    N_BANDS,
    ReproCountFeaturizer,
)
from mop.beds.starss23.count_repro_featurizer_estimator_prereg import (
    CountReproPreregRefusal,
    build_repro_prereg,
    compute_repro_sesoi,
)
from mop.beds.starss23.count_repro_featurizer_estimator_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    ReproCountBedConfig,
    build_repro_count_bed_artifact,
)

FLOP_CEILING = 60_000_000_000
_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"


# ---------------------------------------------------------------------------
# Re-authored frozen featurizer: zero-parameter, 256-wide [128 pos | 128 neg], distinct from the sealed one.
# ---------------------------------------------------------------------------


def test_repro_featurizer_shape_layout_and_zero_params():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((4, 2400 * 12))
    fz = ReproCountFeaturizer()
    feats = fz.featurize(audio)
    assert feats.shape == (12, D_CFEAT) and D_CFEAT == 256
    assert N_BANDS == 32
    assert fz.n_params() == 0
    # Positive and negative polarity blocks are nonnegative sums of rectified flux.
    assert feats.min() >= 0.0
    assert np.isfinite(feats).all()
    # Byte reproducible.
    assert fz.feature_digest(feats) == fz.feature_digest(fz.featurize(audio))


def test_repro_featurizer_digest_differs_from_sealed_mel_frontend():
    new = ReproCountFeaturizer()
    old = FrozenCountFeaturizer()
    assert new.parameter_digest() != old.parameter_digest()
    # The gammatone filterbank bytes differ from the mel filterbank bytes.
    assert not np.array_equal(new.filterbank, old.filterbank)


def test_repro_featurizer_flops_keep_candidate_under_ceiling():
    # The full-scale test fold is 22569 frames; the held-fixed gate charges C_train and gate inference.
    from mop.beds.starss23.count_gate import FLOPS_PER_INFERENCE
    from mop.beds.starss23.gate import training_flops

    n_test = 22569
    featurize = FLOPS_PER_FRAME_COUNT * n_test
    gate_infer = FLOPS_PER_INFERENCE * n_test
    c_train = training_flops(25172, 8)
    k_max = int(0.10 * n_test) * FLOPS_PER_REESTIMATE
    assert featurize + gate_infer + c_train + k_max < FLOP_CEILING


# ---------------------------------------------------------------------------
# Re-authored frozen estimator: zero-parameter cumulative-energy rule, genuinely different track.
# ---------------------------------------------------------------------------


def test_repro_estimator_zero_params_and_caps_at_four():
    rng = np.random.default_rng(1)
    audio = rng.standard_normal((4, 2400 * 30))
    est = ReproCountEstimator()
    track = est.estimate_track(audio)
    assert est.n_params() == 0
    assert BETA == 0.90
    assert int(track.max()) <= MAX_ESTIMABLE_SOURCES
    assert int(track.min()) >= 0
    assert np.array_equal(track, est.estimate_track(audio))  # byte reproducible


def test_repro_estimator_silence_is_zero_and_rule_differs_from_sealed():
    est = ReproCountEstimator()
    silent = np.zeros((4, 2400 * 5))
    assert est.estimate_track(silent).tolist() == [0, 0, 0, 0, 0]
    # On structured noise the cumulative-energy rule and the eigenvalue-threshold rule disagree somewhere.
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((4, 2400 * 40))
    audio[0] += 3.0 * np.sin(np.linspace(0, 80.0, audio.shape[1]))  # inject a coherent direction
    new = est.estimate_track(audio)
    old = FrozenCountEstimator().estimate_track(audio)
    assert not np.array_equal(new, old)
    assert est.parameter_digest() != FrozenCountEstimator().parameter_digest()
    assert FLOPS_PER_REESTIMATE >= 80_000


# ---------------------------------------------------------------------------
# Reproduction preregistration: SESOI computed in code, granularity floor enforced.
# ---------------------------------------------------------------------------


def test_repro_prereg_sesoi_is_half_over_test_clips_and_sealed_before_scores():
    body = build_repro_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        n_test_clips=21,
        n_test_changes=916,
        n_test_frames=22569,
        train_change_density=0.0502,
        coast_from_zero_mae=1.2552,
    )
    assert body["preregistered_before_reading_test_scores"] is True
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert body["reproduction_axis"] == "featurizer_estimator"
    # SESOI reduces to 0.5 / n_test_clips on the pooled-frame scale.
    assert body["sesoi"]["sesoi_mae"] == pytest.approx(0.5 / 21, rel=1e-9)
    assert "canonical_sha256" in body and len(body["canonical_sha256"]) == 64


def test_repro_prereg_refuses_below_granularity_floor():
    # A corpus so small that 0.5 / n_test_clips is below 100x the per-frame granularity must be refused.
    with pytest.raises(CountReproPreregRefusal):
        compute_repro_sesoi(
            operating_reestimate_fraction=0.05,
            n_test_clips=21,
            n_test_changes=48,
            n_test_frames=2100,  # ~100 frames/clip => SESOI is only 50x the granularity
            coast_from_zero_mae=1.0,
        )


# ---------------------------------------------------------------------------
# The independent verifier imports only the standard library and nothing from the producer.
# ---------------------------------------------------------------------------


def test_verifier_imports_no_producer_or_mop_code():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (
        "mop",
        "count_referee",
        "count_repro_featurizer_estimator_producer",
        "count_harness",
        "stats",
        "harness",
        "referee",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), name
    assert set(n.split(".")[0] for n in imported) <= {
        "json",
        "hashlib",
        "itertools",
        "dataclasses",
        "__future__",
    }


# ---------------------------------------------------------------------------
# End-to-end producer + verifier on the REAL subset (small, fast config that clears the granularity floor).
# ---------------------------------------------------------------------------


_SMALL_CONFIG = ReproCountBedConfig(
    seeds=(20, 21, 22), target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=400
)


@pytest.fixture(scope="module")
def real_repro_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("count_repro_fe")
    return build_repro_count_bed_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json"
    )


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_artifact_within_ceiling(real_repro_artifact):
    art = real_repro_artifact.artifact
    assert "seal" in art and isinstance(art["seal"], str) and len(art["seal"]) == 64
    assert art["schema"] == "mop-starss23-escs-count-bed-repro-featurizer-estimator/v1"
    assert art["reproduction_axis"] == "featurizer_estimator"
    assert art["source_kind"] == "real" and art["rights_clean"] is True
    assert art["reproductions"] == 0
    assert art["verdict"] in ("mechanics-ok", "null")
    assert art["flags"] == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    for arm in art["harness"]["arm_summaries"]:
        assert arm["max_lifecycle_flops"] <= FLOP_CEILING
    assert art["matched_budget"]["flops"] <= FLOP_CEILING
    # The frozen modules carry no trained parameters.
    assert art["featurizer"]["n_params"] == 0 and art["estimator"]["n_params"] == 0


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seal_byte_reproducible(tmp_path):
    prereg_path = tmp_path / "prereg.json"
    a = build_repro_count_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    b = build_repro_count_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    assert a.artifact["seal"] == b.artifact["seal"]
    assert a.prereg["canonical_sha256"] == b.prereg["canonical_sha256"]


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_reproduces_referee_and_stats_and_withholds_confirmation(real_repro_artifact):
    result = V.verify_repro_count_artifact(real_repro_artifact.artifact)
    assert result.seal_intact is True
    assert result.scores_reproduced is True
    assert result.stats_reproduced is True
    assert result.honesty_ok is True
    assert result.independent_referee_reproduction is True
    assert result.mismatches == ()
    # A single reproduction can never self-certify scientific confirmation (reproductions == 0 < 3).
    assert result.independent_scientific_confirmation is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_score(real_repro_artifact):
    tampered = copy.deepcopy(real_repro_artifact.artifact)
    # Lower the candidate MAE on the first seed to fake a bigger win; the seal breaks and the re-score fails.
    tampered["per_seed"][0]["arm_scores"]["candidate"]["mae"] = 0.0
    result = V.verify_repro_count_artifact(tampered)
    assert result.independent_referee_reproduction is False
    assert result.survives is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_estimator_track(real_repro_artifact):
    tampered = copy.deepcopy(real_repro_artifact.artifact)
    any_clip = next(iter(tampered["corpus_tracks"]))
    track = tampered["corpus_tracks"][any_clip]["estimator_track"]
    tampered["corpus_tracks"][any_clip]["estimator_track"] = [0 for _ in track]
    result = V.verify_repro_count_artifact(tampered)
    assert result.independent_referee_reproduction is False
