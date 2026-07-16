from __future__ import annotations

import copy
from typing import Any

import pytest

from mop.studies import generation1_d1_frozen_verify as frozen
from mop.studies import generation1_d1_redesign_v2 as redesign
from tests.unit.test_generation1_d1_frozen_verify import (
    _fake_rung_replay,
    aggregate_fixture,
)


def _ci(mean: float, lo: float, hi: float, n: int = 100) -> dict[str, float | int]:
    return {"mean": mean, "lo": lo, "hi": hi, "n": n}


def _metrics(
    *,
    mean: float = 0.02,
    lo: float = 0.01,
    gap: float = 0.01,
    favorable: float = 0.80,
    saving: float = 0.80,
) -> dict[str, Any]:
    return {
        "completed_cell_count": 100,
        "learned_dispatch_differences": {
            "global_static": _ci(mean, lo, mean + 0.01),
            "difficulty_static": _ci(mean, lo, mean + 0.01),
        },
        "favorable_seed_fraction": {
            "global_static": favorable,
            "difficulty_static": favorable,
        },
        "mean_gap_below_context_route": gap,
        "work_saving_vs_all_five_actors": saving,
    }


def _evidence_inventory(
    *,
    advisory_winner: str | None = "pairwise-ranking-margin-mlp-v2",
) -> list[dict[str, Any]]:
    rows = []
    for definition in redesign.candidate_catalog():
        metrics = _metrics() if definition["candidate_id"] == advisory_winner else _metrics(lo=0.0)
        rows.append(
            redesign.build_candidate_evidence(
                definition["candidate_id"],
                producer=metrics,
                challenge=metrics,
            )
        )
    return rows


def _authority(tmp_path, monkeypatch):
    source = tmp_path / "d1.json"
    source.write_bytes(frozen.canonical_bytes(aggregate_fixture()) + b"\n")
    rung_root = tmp_path / "rungs"
    monkeypatch.setattr(frozen, "_replay_raw_rungs", _fake_rung_replay)
    artifact = frozen.build_byte_bound_verification(source, rung_root=rung_root)
    return artifact, source, rung_root


def _reseal(value: dict[str, Any], field: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    value[field] = frozen.canonical_sha256(core)


def test_catalog_has_three_parallel_families_and_never_reenters_centroid() -> None:
    catalog = redesign.candidate_catalog()
    assert {row["family"] for row in catalog} == {
        "utility-residual",
        "pairwise-ranking",
        "calibrated-abstaining",
    }
    assert frozen.FROZEN_VARIANT_ID not in {row["candidate_id"] for row in catalog}
    assert redesign.LEGACY_CONTROL["eligible_for_screen"] is False
    assert redesign.LEGACY_CONTROL["eligible_for_freeze"] is False
    assert all(row["visible_inputs"] == list(redesign.VISIBLE_INPUTS) for row in catalog)
    assert "labeled_training_support_geometry" in redesign.VISIBLE_INPUTS
    assert all(row["forbidden_inputs"] == list(redesign.FORBIDDEN_HELDOUT_INPUTS) for row in catalog)

    with pytest.raises(ValueError, match="legacy centroid"):
        redesign.build_candidate_evidence(
            frozen.FROZEN_VARIANT_ID,
            producer=_metrics(),
            challenge=_metrics(),
        )


def test_caller_summaries_never_yield_winner_or_completed_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    artifact, source, rung_root = _authority(tmp_path, monkeypatch)
    evidence = _evidence_inventory()
    first = redesign.build_screen(
        artifact,
        evidence,
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    second = redesign.build_screen(
        artifact,
        list(reversed(evidence)),
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    redesign.validate_screen(
        first,
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )

    assert first == second
    assert first["decision"]["provisional_freeze_outcome"] == "no_candidate"
    assert first["decision"]["screen_complete"] is False
    assert first["eligible_candidate_ids"] == []
    assert first["complete"] is False
    assert all(row["complete"] is False for row in first["candidate_evidence"])
    assert all(row["eligible_for_freeze"] is False for row in first["candidate_assessments"])

    freeze = redesign.freeze_screen(
        first,
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    redesign.validate_freeze(
        freeze,
        first,
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    assert freeze["outcome"] == "no_candidate"
    assert freeze["selected_candidate"] is None
    assert freeze["ready_for_untouched_replication_design"] is False
    assert freeze["execution_authorized"] is False
    assert {row["status"] for row in freeze["categorized_program_admission"]} == {"preregistered_unexecuted"}


def test_exact_thresholds_are_advisory_but_never_admissible() -> None:
    candidate_id = redesign.candidate_catalog()[0]["candidate_id"]
    exact_pass = _metrics(
        mean=0.01,
        lo=0.000001,
        gap=0.02,
        favorable=0.75,
        saving=0.70,
    )
    candidate = redesign.build_candidate_evidence(
        candidate_id,
        producer=exact_pass,
        challenge=exact_pass,
    )
    assessment = redesign.assess_candidate_evidence(candidate)

    assert assessment["caller_summary_thresholds_satisfied"] is True
    assert assessment["eligible_for_freeze"] is False
    assert candidate["evidence_status"] == "unexecuted_preregistration_summary_only"
    assert candidate["summary_authority"] == "caller_supplied_nonadmissible"
    assert candidate["complete"] is False


def test_candidate_screen_and_freeze_reject_resealed_semantic_drift(
    tmp_path,
    monkeypatch,
) -> None:
    artifact, source, rung_root = _authority(tmp_path, monkeypatch)
    candidate = redesign.build_candidate_evidence(
        redesign.candidate_catalog()[0]["candidate_id"],
        producer=_metrics(),
        challenge=_metrics(),
    )
    leaking = copy.deepcopy(candidate)
    leaking["heldout_contract"]["visible_inputs"].append("truth")
    _reseal(leaking, "evidence_sha256")
    with pytest.raises(ValueError, match="heldout-input boundary"):
        redesign.validate_candidate_evidence(leaking)

    completed = copy.deepcopy(candidate)
    completed["complete"] = True
    completed["problems"] = []
    _reseal(completed, "evidence_sha256")
    with pytest.raises(ValueError, match="identity or safety"):
        redesign.validate_candidate_evidence(completed)

    screen = redesign.build_screen(
        artifact,
        _evidence_inventory(),
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    screen_mutant = copy.deepcopy(screen)
    screen_mutant["eligible_candidate_ids"] = ["pairwise-ranking-margin-mlp-v2"]
    _reseal(screen_mutant, "screen_sha256")
    with pytest.raises(ValueError, match="semantic projection"):
        redesign.validate_screen(
            screen_mutant,
            predecessor_source_path=source,
            predecessor_rung_root=rung_root,
        )

    freeze = redesign.freeze_screen(
        screen,
        predecessor_source_path=source,
        predecessor_rung_root=rung_root,
    )
    freeze_mutant = copy.deepcopy(freeze)
    freeze_mutant["outcome"] = "winner"
    freeze_mutant["ready_for_untouched_replication_design"] = True
    _reseal(freeze_mutant, "freeze_sha256")
    with pytest.raises(ValueError, match="semantic projection"):
        redesign.validate_freeze(
            freeze_mutant,
            screen,
            predecessor_source_path=source,
            predecessor_rung_root=rung_root,
        )


def test_screen_requires_byte_bound_live_replayed_predecessor(
    tmp_path,
    monkeypatch,
) -> None:
    artifact, source, rung_root = _authority(tmp_path, monkeypatch)
    evidence = _evidence_inventory()
    with pytest.raises(ValueError, match="exact append-only candidate inventory"):
        redesign.build_screen(
            artifact,
            evidence[:-1],
            predecessor_source_path=source,
            predecessor_rung_root=rung_root,
        )

    structural_only = frozen.verify_frozen_aggregate(aggregate_fixture())
    with pytest.raises(ValueError, match="field inventory"):
        redesign.build_screen(
            structural_only,
            evidence,
            predecessor_source_path=source,
            predecessor_rung_root=rung_root,
        )

    changed = aggregate_fixture()
    changed["rungs"][0]["result_sha256"] = frozen.canonical_sha256({"changed": True})
    core = {key: item for key, item in changed.items() if key != "result_sha256"}
    changed["result_sha256"] = frozen.canonical_sha256(core)
    source.write_bytes(frozen.canonical_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="source replay drifted"):
        redesign.build_screen(
            artifact,
            evidence,
            predecessor_source_path=source,
            predecessor_rung_root=rung_root,
        )
