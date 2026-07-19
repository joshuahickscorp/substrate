
from __future__ import annotations

import ast
import copy
import itertools
import random
from pathlib import Path

import pytest

from mop.beds.starss23 import count_repro_scoring_unit_verifier as V
from mop.beds.starss23.count_prereg import compute_count_cost_benefit
from mop.beds.starss23.count_producer import DEFAULT_FOA_ROOT, DEFAULT_METADATA_ROOT
from mop.beds.starss23.count_producer import RealCountBedConfig
from mop.beds.starss23.count_referee import mae_clip as sealed_mae_clip
from mop.beds.starss23.count_repro_scoring_unit_prereg import (
    ClipLabelFact,
    build_count_repro_scoring_unit_prereg,
)
from mop.beds.starss23.count_repro_scoring_unit_producer import (
    build_real_count_repro_scoring_unit_artifact,
)
from mop.beds.starss23.count_repro_scoring_unit_referee import (
    _exact_sign_flip_one_sided_meet_in_middle,
    exact_sign_flip_over_clips,
    macro_score_arm,
)

_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"




def test_macro_referee_weights_each_clip_equally():
    score = macro_score_arm(
        [("A", [0, 1, 1, 2, 0], [0, 1, 2, 2, 0], [1, 3]), ("B", [0, 0], [1, 1], [])]
    )
    assert score.macro_mae == pytest.approx(0.2)
    assert score.n_clips == 2
    clip_mae = score.clip_mae()
    assert clip_mae["A"] == pytest.approx(0.4)
    assert clip_mae["B"] == pytest.approx(0.0)


def test_macro_referee_reuses_sealed_coasting_per_clip():
    gt, est, reest = [0, 1, 1, 2, 0], [0, 1, 2, 2, 0], [1, 3]
    abs_err, n = sealed_mae_clip(gt, est, reest)
    score = macro_score_arm([("A", gt, est, reest)])
    per_clip = score.per_clip[0]
    assert (per_clip.abs_error_sum, per_clip.n_frames) == (abs_err, n)


def test_macro_referee_differs_from_pooled_when_clip_lengths_differ():
    long_gt = [0] * 100
    long_est = [0] * 100
    long_reest = list(range(100))  # perfect on the long clip -> per-clip mae 0
    short_gt = [0, 3]
    short_est = [0, 0]
    short_reest = []  # emitted 0,0 vs gt 0,3 -> abs err 3 over 2 -> per-clip mae 1.5
    macro = macro_score_arm(
        [("long", long_gt, long_est, long_reest), ("short", short_gt, short_est, short_reest)]
    ).macro_mae
    assert macro == pytest.approx(0.75)
    assert macro != pytest.approx(3 / 102)


def test_clip_permutation_meet_in_middle_matches_brute_force():
    def brute(deltas):
        n = len(deltas)
        t_obs = sum(deltas)
        at_least = sum(
            1
            for signs in itertools.product((1.0, -1.0), repeat=n)
            if sum(s * d for s, d in zip(signs, deltas, strict=True)) >= t_obs - 1e-9
        )
        return t_obs, at_least / (2**n), 2**n

    rng = random.Random(2026)
    for _ in range(150):
        n = rng.randrange(1, 13)
        deltas = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        mine = _exact_sign_flip_one_sided_meet_in_middle(deltas)
        ref = brute(deltas)
        assert mine[0] == pytest.approx(ref[0])
        assert mine[1] == pytest.approx(ref[1])
        assert mine[2] == ref[2]


def test_clip_permutation_all_positive_hits_the_floor():
    cluster = exact_sign_flip_over_clips([0.1, 0.2, 0.05, 0.3, 0.15])
    assert cluster.one_sided_p == pytest.approx(1 / 32)
    assert cluster.direction_agrees is True
    assert cluster.fraction_candidate_lower == pytest.approx(1.0)




def test_prereg_macro_sesoi_uses_the_reused_rule_and_clears_the_floor():
    facts = [ClipLabelFact(clip_id=f"c{i:02d}", n_frames=1000, n_changes=40) for i in range(21)]
    body = build_count_repro_scoring_unit_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        test_clip_facts=facts,
        train_change_density=0.05,
        coast_from_zero_mae=1.2,
    )
    cb = compute_count_cost_benefit(
        c_train_flops=body["sesoi"]["cost_benefit"]["c_train_flops"],
        c_reest_flops=body["sesoi"]["cost_benefit"]["c_reest_flops"],
        operating_reestimate_fraction=0.05,
        n_test_frames=21000,
        n_test_clips=21,
        n_test_changes=840,
        coast_from_zero_mae=1.2,
    )
    assert body["sesoi"]["sesoi_mae"] == pytest.approx(cb.one_clip_change_mass_mae)
    assert body["sesoi"]["sesoi_mae"] == pytest.approx(0.5 / 21)
    assert body["sesoi"]["restatement_matches_reused_rule"] is True
    assert body["sesoi"]["clears_granularity_floor"] is True
    assert body["preregistered_before_reading_test_scores"] is True
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert "canonical_sha256" in body


def test_prereg_refuses_when_sesoi_below_the_macro_granularity_floor():
    facts = [ClipLabelFact(clip_id="only", n_frames=3, n_changes=1)]
    with pytest.raises(Exception):
        build_count_repro_scoring_unit_prereg(
            timestamp=_TIMESTAMP,
            operating_reestimate_fraction=0.05,
            test_clip_facts=facts,
            train_change_density=0.05,
            coast_from_zero_mae=1.0,
        )




def test_verifier_imports_only_stdlib_and_no_producer_or_mop_code():
    source = Path(V.__file__).read_text(encoding="utf-8")
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
    assert "mop" not in roots
    for forbidden in ("count_referee", "count_producer", "count_harness", "stats", "harness", "referee"):
        assert forbidden not in roots
    assert roots <= {"json", "hashlib", "itertools", "dataclasses", "__future__"}, roots


def test_reproduction_reuses_sealed_coasting_by_import_not_reimplementation():
    import mop.beds.starss23.count_referee as sealed
    import mop.beds.starss23.count_repro_scoring_unit_referee as macro

    assert macro.mae_clip is sealed.mae_clip




_SMALL_CONFIG = RealCountBedConfig(
    seeds=(30, 31, 32), target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=300
)


@pytest.fixture(scope="module")
def real_macro_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("count_repro_scoring_unit")
    return build_real_count_repro_scoring_unit_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json"
    )


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_clip_macro_artifact(real_macro_artifact):
    art = real_macro_artifact.artifact
    assert "seal" in art and isinstance(art["seal"], str) and len(art["seal"]) == 64
    assert art["schema"] == "mop-starss23-escs-count-repro-scoring-unit-bed/v1"
    assert art["reproduction_axis"] == "scoring_unit"
    assert art["scoring_unit"] == "clip-macro"
    assert art["source_kind"] == "real" and art["rights_clean"] is True
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
def test_verifier_reproduces_macro_scores_and_both_permutations(real_macro_artifact):
    result = V.verify_count_repro_scoring_unit_artifact(real_macro_artifact.artifact)
    assert result.seal_intact is True
    assert result.scores_reproduced is True
    assert result.stats_reproduced is True
    assert result.clip_cluster_reproduced is True
    assert result.honesty_ok is True
    assert result.independent_referee_reproduction is True
    assert result.independent_scientific_confirmation is False
    assert result.mismatches == ()


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seal_is_byte_reproducible(tmp_path):
    a = build_real_count_repro_scoring_unit_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=tmp_path / "p.json"
    )
    b = build_real_count_repro_scoring_unit_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=tmp_path / "p.json"
    )
    assert a.artifact["seal"] == b.artifact["seal"]
    assert a.prereg["canonical_sha256"] == b.prereg["canonical_sha256"]


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_macro_score(real_macro_artifact):
    art = copy.deepcopy(real_macro_artifact.artifact)
    art["per_seed"][0]["arm_scores"]["candidate"]["macro_mae"] += 0.5
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    result = V.verify_count_repro_scoring_unit_artifact(art)
    assert result.seal_intact is True
    assert result.scores_reproduced is False
    assert result.independent_referee_reproduction is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_clip_cluster(real_macro_artifact):
    art = copy.deepcopy(real_macro_artifact.artifact)
    art["clip_cluster"]["permutation"]["direction_agrees"] = not art["clip_cluster"]["permutation"][
        "direction_agrees"
    ]
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    result = V.verify_count_repro_scoring_unit_artifact(art)
    assert result.seal_intact is True
    assert result.clip_cluster_reproduced is False
    assert result.independent_referee_reproduction is False
