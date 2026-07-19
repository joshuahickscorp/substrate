
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from mop.beds.starss23 import count_repro_data_split_verifier as V
from mop.beds.starss23.count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealCountBedConfig,
)
from mop.beds.starss23.count_repro_data_split_prereg import (
    DEFAULT_REPRO_PREREG_PATH,
    MIN_GRANULARITY_MULTIPLE,
    ReproPreregRefusal,
    build_data_split_prereg,
)
from mop.beds.starss23.count_repro_data_split_producer import (
    DATA_SPLIT_SEEDS,
    DEFAULT_REPRO_ARTIFACT_PATH,
    build_data_split_repro_artifact,
)

_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"

_SMALL_CONFIG = RealCountBedConfig(
    seeds=DATA_SPLIT_SEEDS, target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=300
)




def test_prereg_self_derived_sesoi_and_floor():
    body = build_data_split_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        n_test_clips=24,
        n_test_changes=1477,
        n_test_frames=28411,
        train_change_density=0.038,
        coast_from_zero_mae=1.2,
    )
    assert body["sesoi"]["sesoi_mae"] == pytest.approx(0.5 / 24, abs=1e-9)
    assert body["schema"] == "mop-starss23-count-repro-data-split-prereg/v1"
    assert body["reproduction_axis"] == "data_split"
    assert body["preregistered_before_reading_test_scores"] is True
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert body["sesoi"]["granularity_multiple"] >= MIN_GRANULARITY_MULTIPLE
    assert body["sign_flip_test_plan"]["min_one_sided_p"] == pytest.approx(1 / 32)
    assert body["sign_flip_test_plan"]["two_sided_alpha_reachable"] is False
    assert "canonical_sha256" in body


def test_prereg_refuses_below_granularity_floor():
    with pytest.raises(ReproPreregRefusal):
        build_data_split_prereg(
            timestamp=_TIMESTAMP,
            operating_reestimate_fraction=0.05,
            n_test_clips=24,
            n_test_changes=100,
            n_test_frames=24 * 100,  # 100 frames per clip -> only 60x granularity
            train_change_density=0.05,
            coast_from_zero_mae=1.0,
        )




def test_additive_only_repro_paths_are_net_new():
    prereg = Path(DEFAULT_REPRO_PREREG_PATH)
    artifact = Path(DEFAULT_REPRO_ARTIFACT_PATH)
    assert prereg.name == "STARSS23_COUNTING_REPRO_data_split.prereg.json"
    assert artifact.name == "STARSS23_COUNTING_REPRO_data_split.json"
    sealed = {
        "STARSS23_COUNTING_BED.json",
        "STARSS23_COUNTING_BED.prereg.json",
        "STARSS23_COUNTING_BED.verification.json",
    }
    assert prereg.name not in sealed and artifact.name not in sealed




def test_verifier_imports_no_producer_or_mop_code():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (
        "mop",
        "count_referee",
        "count_producer",
        "count_harness",
        "count_verifier",
        "count_repro_data_split_producer",
        "count_repro_data_split_prereg",
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




@pytest.fixture(scope="module")
def repro_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("count_repro_data_split")
    return build_data_split_repro_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json"
    )


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_swapped_split_trains_on_fold4_scores_on_fold3(repro_artifact):
    rc = repro_artifact.artifact["real_corpus"]
    rooms = rc["split_rooms"]
    fold4 = {"room02", "room08", "room10", "room15", "room16", "room23", "room24"}
    fold3 = {"room04", "room06", "room07", "room12", "room13", "room14", "room21", "room22"}
    assert set(rooms["train_rooms"]) | set(rooms["val_rooms"]) == fold4
    assert set(rooms["test_rooms"]) == fold3
    train, val, test = set(rooms["train_rooms"]), set(rooms["val_rooms"]), set(rooms["test_rooms"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert rooms["swapped_from_sealed"] is True
    assert rc["n_test_clips"] == 24


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_within_ceiling_and_never_self_certifies(repro_artifact):
    art = repro_artifact.artifact
    assert isinstance(art["seal"], str) and len(art["seal"]) == 64
    assert art["schema"] == "mop-starss23-escs-count-bed-repro-data-split/v1"
    assert art["source_kind"] == "real" and art["rights_clean"] is True
    assert art["reproductions"] == 0
    assert art["verdict"] in ("mechanics-ok", "null")
    assert art["flags"] == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    for arm in art["harness"]["arm_summaries"]:
        assert arm["max_lifecycle_flops"] <= 60_000_000_000
    assert art["matched_budget"]["flops"] <= 60_000_000_000


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_reproduces_referee_and_never_self_confirms(repro_artifact):
    result = V.verify_data_split_artifact(repro_artifact.artifact)
    assert result.seal_intact is True
    assert result.schema_ok is True
    assert result.split_room_disjoint is True
    assert result.scores_reproduced is True
    assert result.stats_reproduced is True
    assert result.honesty_ok is True
    assert result.independent_referee_reproduction is True
    assert result.mismatches == ()
    assert result.independent_scientific_confirmation is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_byte_reproducible_seal_same_prereg_path(tmp_path):
    prereg_path = tmp_path / "prereg.json"
    a = build_data_split_repro_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    b = build_data_split_repro_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    assert a.artifact["seal"] == b.artifact["seal"]
    assert a.prereg["canonical_sha256"] == b.prereg["canonical_sha256"]


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_candidate_mae(repro_artifact):
    art = copy.deepcopy(repro_artifact.artifact)
    art["per_seed"][0]["arm_scores"]["candidate"]["mae"] += 0.5
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    result = V.verify_data_split_artifact(art)
    assert result.seal_intact is True
    assert result.scores_reproduced is False
    assert result.independent_referee_reproduction is False
    assert result.survives is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_estimator_track(repro_artifact):
    art = copy.deepcopy(repro_artifact.artifact)
    some_clip = next(iter(art["corpus_tracks"]))
    art["corpus_tracks"][some_clip]["estimator_track"][0] += 3
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    assert V.verify_data_split_artifact(art).scores_reproduced is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_broken_seal(repro_artifact):
    art = copy.deepcopy(repro_artifact.artifact)
    art["seal"] = "0" * 64
    result = V.verify_data_split_artifact(art)
    assert result.seal_intact is False
    assert result.independent_referee_reproduction is False
