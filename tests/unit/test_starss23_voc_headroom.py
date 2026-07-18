"""Tests for the STARSS23 value-of-computation HEADROOM instrument.

Covers the analyzer primitives (geometry, coasting, both metrics, budgeting, the greedy label-aware
reference and its monotonicity, the rate-matched-random draw discipline), the per-target interpretation
classifier on constructed strong/harmful/saturated targets, the producer-to-verifier round trip and its
tamper detection, the stdlib-only import boundary of the verifier, and a small real-subset end-to-end run.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from mop.beds.starss23 import vochead_verifier as V
from mop.beds.starss23.vochead_analyzer import (
    METRIC_COUNT_ABS,
    METRIC_DOA_GREATCIRCLE,
    ClipTarget,
    VocHeadRefusal,
    analyze_target,
    budget_k,
    clip_error,
    coast,
    great_circle_degrees,
    greedy_informed_path,
    rate_matched_random_frames,
    rmr_mean_error,
)
from mop.beds.starss23.vochead_prereg import build_vochead_prereg
from mop.beds.starss23.vochead_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealVocHeadConfig,
    build_vochead_artifact,
)
from mop.beds.starss23.vochead_verifier import verify_vochead_artifact

_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()


def _count_target(clip_id: str, gt, est, cold_start=0) -> ClipTarget:
    n = len(gt)
    changes = tuple(t for t in range(1, n) if gt[t] != gt[t - 1])
    return ClipTarget(
        clip_id, "roomA", n, METRIC_COUNT_ABS, tuple([True] * n), tuple(gt), tuple(est), changes, cold_start
    )


# ---------------------------------------------------------------------------
# Geometry, coasting, metrics.
# ---------------------------------------------------------------------------


def test_great_circle_identity_and_antipodal():
    assert great_circle_degrees(37.0, 12.0, 37.0, 12.0) == pytest.approx(0.0, abs=1e-9)
    assert great_circle_degrees(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0, abs=1e-9)
    assert great_circle_degrees(10.0, 0.0, 10.0, 90.0) == pytest.approx(90.0, abs=1e-6)


def test_coast_holds_last_refresh_else_cold_start():
    t = _count_target("c", [0, 1, 2, 3], [5, 6, 7, 8], cold_start=0)
    assert coast(t, ()) == [0, 0, 0, 0]
    assert coast(t, (1,)) == [0, 6, 6, 6]
    assert coast(t, (0, 3)) == [5, 5, 5, 8]


def test_clip_error_count_and_doa():
    t = _count_target("c", [1, 1, 1], [1, 1, 1], cold_start=0)
    assert clip_error(t, ()) == pytest.approx(1.0)  # coast 0 vs gt 1
    assert clip_error(t, (0,)) == pytest.approx(0.0)  # refresh -> est 1 == gt
    dt = ClipTarget(
        "d",
        "roomA",
        2,
        METRIC_DOA_GREATCIRCLE,
        (True, True),
        ((0.0, 0.0), (0.0, 0.0)),
        ((180.0, 0.0), (180.0, 0.0)),
        (),
        (0.0, 0.0),
    )
    assert clip_error(dt, ()) == pytest.approx(0.0)  # coast cold start (0,0) == gt (0,0)
    assert clip_error(dt, (0, 1)) == pytest.approx(180.0)  # refresh to antipodal estimate


def test_budget_k_rounds_and_clamps():
    assert budget_k(1000, 0.05) == 50
    assert budget_k(1000, 0.0001) == 1  # clamped up to at least 1
    assert budget_k(10, 1.0) == 10
    with pytest.raises(VocHeadRefusal):
        budget_k(10, 0.0)


# ---------------------------------------------------------------------------
# Policies: greedy reference monotonicity and rate-matched-random discipline.
# ---------------------------------------------------------------------------


def test_greedy_path_monotone_and_bounded_by_never():
    gt = [0, 0, 1, 1, 1, 2, 2, 0, 0, 3, 3, 3] * 3
    t = _count_target("c", gt, gt)  # perfect estimator
    path = greedy_informed_path(t, 20)
    assert path[0] == pytest.approx(clip_error(t, ()))  # errors[0] is never_update
    for i in range(len(path) - 1):
        assert path[i + 1] <= path[i] + 1e-12  # never increases
    assert path[-1] < path[0]  # a perfect estimator strictly improves


def test_greedy_no_changes_cannot_refresh():
    t = _count_target("c", [1] * 8, [1] * 8, cold_start=0)  # constant, zero change frames
    path = greedy_informed_path(t, 5)
    assert len(path) == 1  # nothing to refresh; only the never_update point
    assert path[0] == pytest.approx(1.0)


def test_rate_matched_random_deterministic_and_sized():
    a = rate_matched_random_frames(100, 7, seed=12345)
    b = rate_matched_random_frames(100, 7, seed=12345)
    assert a == b and len(a) == 7 and a == sorted(set(a))
    t = _count_target("c", [0, 1] * 20, [0, 1] * 20)
    assert rmr_mean_error(t, 4, base_seed=1, n_draws=8) == rmr_mean_error(t, 4, base_seed=1, n_draws=8)


# ---------------------------------------------------------------------------
# Interpretation classifier: the three shapes.
# ---------------------------------------------------------------------------


def test_interpretation_real_headroom_strong_what():
    gt = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0] * 6
    t = _count_target("c", gt, gt)  # perfect fresh estimator
    result = analyze_target("strong", METRIC_COUNT_ABS, [t])
    assert result["refreshable_range"] > 0
    assert result["interpretation"] == "real_headroom"
    assert all(b["headroom_rmr_minus_informed"] > 0 for b in result["budgets"])


def test_interpretation_what_floor_collapse_harmful_what():
    gt = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0] * 6
    t = _count_target("c", gt, [9] * len(gt))  # maximally wrong estimator
    result = analyze_target("harmful", METRIC_COUNT_ABS, [t])
    assert result["refreshable_range"] < 0
    assert result["interpretation"] == "what_floor_collapse"


def test_interpretation_no_headroom_when_informed_cannot_beat_random():
    # constant target with a helpful estimator but ZERO change frames: the informed reference cannot
    # refresh (no changes), random can, so informed never strictly beats random -> no real headroom.
    t = _count_target("c", [1] * 40, [1] * 40, cold_start=0)
    result = analyze_target("saturated", METRIC_COUNT_ABS, [t])
    assert result["refreshable_range"] > 0
    assert result["interpretation"] == "no_headroom_budget_saturated"


# ---------------------------------------------------------------------------
# Producer to verifier round trip (synthetic corpus) and tamper detection.
# ---------------------------------------------------------------------------


def _tiny_artifact() -> dict:
    from mop.substrate.events import canonical_sha256

    gt = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0] * 6
    strong = _count_target("synthetic_strong_what", gt, gt)
    harmful = _count_target("synthetic_harmful_what", gt, [9] * len(gt))
    real_strong = _count_target("fold4_room1_mix001", gt, gt)
    content = {
        "schema": "mop-starss23-vochead-bed/v1",
        "stage": 3,
        "instrument_id": "starss23_voc_headroom",
        "source_kind": "real",
        "rights_clean": True,
        "corpus_targets": {
            "count": {"fold4_room1_mix001": real_strong.payload()},
            "doa": {},
            "synthetic_control": {
                "synthetic_strong_what": strong.payload(),
                "synthetic_harmful_what": harmful.payload(),
            },
        },
        "analysis": {
            "test_fold": {
                "clip_ids": ["fold4_room1_mix001"],
                "count": analyze_target("count", METRIC_COUNT_ABS, [real_strong]),
                "doa": analyze_target("doa", METRIC_DOA_GREATCIRCLE, [_doa_dummy()]),
            },
            "full_subset": {
                "clip_ids": ["fold4_room1_mix001"],
                "count": analyze_target("count", METRIC_COUNT_ABS, [real_strong]),
                "doa": analyze_target("doa", METRIC_DOA_GREATCIRCLE, [_doa_dummy()]),
            },
            "synthetic_control": {
                "clip_ids": ["synthetic_harmful_what", "synthetic_strong_what"],
                "strong_what": analyze_target("synthetic_strong_what", METRIC_COUNT_ABS, [strong]),
                "harmful_what": analyze_target("synthetic_harmful_what", METRIC_COUNT_ABS, [harmful]),
            },
        },
        "synthetic_control_ok": True,
        "flags": {
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
            "is_instrument_not_bed": True,
        },
    }
    # the tiny artifact needs a doa entry in corpus for the verifier's doa scope reconstruction
    content["corpus_targets"]["doa"] = {"fold4_room1_mix001": _doa_dummy().payload()}
    content["analysis"]["test_fold"]["clip_ids"] = ["fold4_room1_mix001"]
    return {**content, "seal": {"sha256": canonical_sha256(content)}}


def _doa_dummy() -> ClipTarget:
    n = 12
    gt = tuple((float(10 * (t % 3)), 0.0) for t in range(n))
    est = tuple((float(10 * (t % 3)), 0.0) for t in range(n))
    changes = tuple(t for t in range(1, n) if gt[t] != gt[t - 1])
    return ClipTarget(
        "fold4_room1_mix001",
        "roomA",
        n,
        METRIC_DOA_GREATCIRCLE,
        tuple([True] * n),
        gt,
        est,
        changes,
        (0.0, 0.0),
    )


def test_verifier_round_trip_all_true():
    artifact = _tiny_artifact()
    v = verify_vochead_artifact(artifact)
    assert v["seal_intact"] and v["targets_reproduced"] and v["interpretation_reproduced"]
    assert v["honesty_ok"] and v["independent_referee_reproduction"]
    assert v["independent_scientific_confirmation"] is False
    assert v["mismatches"] == []


def test_verifier_detects_seal_tamper():
    artifact = _tiny_artifact()
    artifact["analysis"]["test_fold"]["count"]["refreshable_range"] += 5.0  # forge a number
    v = verify_vochead_artifact(artifact)
    assert not v["seal_intact"] or not v["targets_reproduced"]
    assert v["mismatches"]


def test_verifier_detects_flag_tamper():
    artifact = _tiny_artifact()
    artifact["flags"]["scientific_promotion"] = True  # a forbidden promotion claim
    v = verify_vochead_artifact(artifact)
    assert not v["honesty_ok"]


def test_verifier_imports_stdlib_only():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("mop", "numpy", "vochead_analyzer", "vochead_producer", "vochead_prereg")
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), name
    allowed_stdlib = {"json", "hashlib", "math", "random", "bisect", "pathlib", "__future__", "typing"}
    assert {n.split(".")[0] for n in imported} <= allowed_stdlib


def test_prereg_seals_and_is_honest():
    body = build_vochead_prereg(timestamp="2026-07-18T00:00Z")
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert body["interpretation_rule"]["tie_is_null"] is True
    from mop.substrate.events import canonical_sha256

    declared = body.pop("canonical_sha256")
    assert declared == canonical_sha256(body)


# ---------------------------------------------------------------------------
# End-to-end on the REAL subset (small, fast config).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_real_end_to_end_small():
    config = RealVocHeadConfig(max_frames=250)
    artifact = build_vochead_artifact(config)
    assert artifact["source_kind"] == "real" and artifact["rights_clean"] is True
    assert artifact["synthetic_control_ok"] is True
    v = verify_vochead_artifact(artifact)
    assert v["seal_intact"] and v["targets_reproduced"] and v["interpretation_reproduced"]
    assert v["honesty_ok"] and v["independent_referee_reproduction"]
    assert v["independent_scientific_confirmation"] is False
    # a tamper on a re-read copy must be caught
    forged = copy.deepcopy(artifact)
    forged["analysis"]["full_subset"]["count"]["always_on_macro"] += 1.0
    assert (
        not verify_vochead_artifact(forged)["targets_reproduced"]
        or not verify_vochead_artifact(forged)["seal_intact"]
    )
