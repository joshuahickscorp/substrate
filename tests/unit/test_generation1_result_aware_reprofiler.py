
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mop.config import REPO_ROOT
from mop.studio import generation1_result_aware_reprofiler as reprof
from mop.studio.generation1_result_aware_reprofiler import ReprofileRefused
from mop.studio.generation1_supervisor import canonical_sha256

_PROOF = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json"
_MECHANICS_RUNS = REPO_ROOT / "runs/generation1/generation1-successor-mechanics-extended-v1"


def _receipt_core(
    *,
    index: int,
    lane_id: str,
    mechanism: str,
    phase: str,
    rung_index: int,
    seed_start: int,
    seed_count: int,
    complete: bool = True,
) -> dict[str, object]:
    return {
        "schema": reprof.MECHANICS_RUNG_SCHEMA,
        "program_id": "synthetic-mechanics",
        "claim_scope": "synthetic",
        "item": {
            "index": index,
            "lane_id": lane_id,
            "mechanism": mechanism,
            "phase": phase,
            "rung_index": rung_index,
            "seed_start": seed_start,
            "seed_count": seed_count,
        },
        "receipt_count": seed_count,
        "verdict_counts": {"mechanics-ok": seed_count},
        "control_clear_counts": {},
        "confirmation_count": 0,
        "receipt_digest_fold": "0" * 64,
        "complete": complete,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def _write_receipt(
    root: Path,
    core: dict[str, object],
    *,
    mtime_ns: int,
    break_seal: bool = False,
) -> Path:
    sealed = {**core, "result_sha256": canonical_sha256(core)}
    if break_seal:
        sealed["result_sha256"] = "f" * 64
    item = core["item"]
    assert isinstance(item, dict)
    path = root / str(item["lane_id"]).lower() / str(item["phase"]) / f"rung_{int(item['rung_index']):03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _build_tree(root: Path) -> None:
    second = 1_000_000_000
    for rung, offset in ((0, 0), (1, 10), (2, 20)):
        _write_receipt(
            root,
            _receipt_core(
                index=rung,
                lane_id="G1-X1",
                mechanism="mech_x",
                phase="producer",
                rung_index=rung,
                seed_start=100 + rung * 100,
                seed_count=100,
            ),
            mtime_ns=1_000 * second + offset * second,
        )
    _write_receipt(
        root,
        _receipt_core(
            index=10,
            lane_id="G1-Y1",
            mechanism="mech_y",
            phase="canary",
            rung_index=0,
            seed_start=5_000,
            seed_count=256,
        ),
        mtime_ns=2_000 * second,
    )
    _write_receipt(
        root,
        _receipt_core(
            index=20,
            lane_id="G1-Z1",
            mechanism="mech_z",
            phase="producer",
            rung_index=0,
            seed_start=9_000,
            seed_count=100,
            complete=False,
        ),
        mtime_ns=3_000 * second,
    )
    _write_receipt(
        root,
        _receipt_core(
            index=21,
            lane_id="G1-Z1",
            mechanism="mech_z",
            phase="producer",
            rung_index=1,
            seed_start=9_100,
            seed_count=100,
        ),
        mtime_ns=3_010 * second,
        break_seal=True,
    )


def test_collect_counts_only_complete_receipts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    proof_root = tmp_path / "proof"
    proof_root.mkdir()
    _build_tree(runs_root)

    profile = reprof.collect_observed_rates(runs_root, proof_root)
    observed = profile["observed_rates"]

    assert set(observed) == {"mech_x"}
    assert observed["mech_x"]["observed_seconds_per_seed"] == pytest.approx(0.1)
    assert observed["mech_x"]["sample_rungs"] == 2
    assert observed["mech_x"]["complete_receipts"] == 3
    assert observed["mech_x"]["total_seeds"] == 300

    assert "mech_y" not in observed
    assert "mech_z" not in observed

    assert profile["receipts_seen"] == 6
    assert profile["receipts_skipped"] == 2


def test_collect_ignores_foreign_schema_receipts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    proof_root = tmp_path / "proof"
    proof_root.mkdir()
    _build_tree(runs_root)
    foreign = runs_root / "other" / "phase" / "rung_000.json"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text(json.dumps({"schema": "some-other-program/v1", "value": 1}), encoding="utf-8")

    profile = reprof.collect_observed_rates(runs_root, proof_root)
    assert profile["receipts_seen"] == 6
    assert profile["receipts_skipped"] == 2


def test_worker_count_memory_bound() -> None:
    math_block = reprof.recommended_worker_count(28, 96.0, 16.0)
    assert math_block["recommended_workers"] == 6
    assert math_block["cores_after_reserve"] == 24
    assert math_block["memory_bound_workers"] == 6
    assert math_block["hard_ceiling"] == 16
    assert math_block["binding_constraint"] == "memory_bound_workers"
    assert math_block["clamped_to_floor"] is False


def test_worker_count_core_bound() -> None:
    math_block = reprof.recommended_worker_count(8, 1_000.0, 1.0)
    assert math_block["recommended_workers"] == 4
    assert math_block["binding_constraint"] == "cores_after_reserve"


def test_worker_count_ceiling_bound() -> None:
    math_block = reprof.recommended_worker_count(64, 1_024.0, 16.0)
    assert math_block["recommended_workers"] == reprof.WORKER_HARD_CEILING == 16
    assert math_block["binding_constraint"] == "hard_ceiling"


def test_worker_count_clamped_to_floor() -> None:
    math_block = reprof.recommended_worker_count(5, 1.0, 16.0)
    assert math_block["memory_bound_workers"] == 0
    assert math_block["recommended_workers"] == 1
    assert math_block["clamped_to_floor"] is True


def test_worker_count_tie_breaks_toward_cores() -> None:
    math_block = reprof.recommended_worker_count(20, 256.0, 16.0)
    assert math_block["recommended_workers"] == 16
    assert math_block["binding_constraint"] == "cores_after_reserve"


@pytest.mark.parametrize(
    ("cores", "memory", "cap"),
    [(0, 96.0, 16.0), (28, 0.0, 16.0), (28, 96.0, 0.0), (-1, 96.0, 16.0), (28, -5.0, 16.0)],
)
def test_worker_count_rejects_nonpositive_inputs(cores: int, memory: float, cap: float) -> None:
    with pytest.raises(ReprofileRefused):
        reprof.recommended_worker_count(cores, memory, cap)


def test_reprofile_uses_observed_else_planned() -> None:
    planned = {"mech_x": 0.2, "mech_y": 0.001}
    observed = {"mech_x": {"observed_seconds_per_seed": 0.1}}
    result = reprof.reprofile(planned, observed, 28, 96.0, 16.0, planned_seconds_total=7_200.0)
    rates = result["de_idealized_rates"]

    assert rates["mech_x"]["source"] == "observed"
    assert rates["mech_x"]["seconds_per_seed"] == pytest.approx(0.1)
    assert rates["mech_x"]["observed_over_planned"] == pytest.approx(0.5)
    assert rates["mech_x"]["provisional"] is False

    assert rates["mech_y"]["source"] == "planned"
    assert rates["mech_y"]["seconds_per_seed"] == pytest.approx(0.001)
    assert rates["mech_y"]["observed_over_planned"] is None
    assert rates["mech_y"]["provisional"] is True

    assert result["recommended_workers"] == 6
    assert result["projection"]["serial_hours"] == pytest.approx(2.0)
    assert result["projection"]["ideal_worker_hours"] == pytest.approx(2.0 / 6.0)


def _synthetic_profile() -> dict[str, object]:
    return {
        "observed_rates": {
            "construction_search": {
                "observed_seconds_per_seed": 0.04,
                "sample_rungs": 10,
                "complete_receipts": 11,
                "total_seeds": 2_048,
            }
        },
        "continuing_lanes": ["G1-G1"],
        "pruned_lanes": ["G1-P1"],
        "lane_mechanisms": {"G1-G1": "construction_search", "G1-P1": "stability_plasticity"},
        "source_results": [
            {
                "program_id": "synthetic",
                "schema": "mop-generation1-successor-mechanics-extended/v1",
                "result_sha256": "a" * 64,
                "path": "proof/synthetic.json",
            }
        ],
        "receipts_seen": 11,
        "receipts_skipped": 0,
    }


def _planned() -> dict[str, float]:
    return {"construction_search": 0.0407, "stability_plasticity": 0.000211}


def test_artifact_round_trip_validates() -> None:
    artifact = reprof.build_reprofile_artifact(_synthetic_profile(), _planned())
    reprof.validate_reprofile(artifact)
    assert artifact["schema"] == reprof.SCHEMA
    assert artifact["advisory"] is True
    assert artifact["activation_allowed"] is False
    assert artifact["scientific_promotion"] is False
    assert artifact["provisional_mechanisms"] == ["stability_plasticity"]
    assert artifact["recommendation"]["recommended_workers"] == 16


def test_forgery_seal_flip_is_rejected() -> None:
    artifact = reprof.build_reprofile_artifact(_synthetic_profile(), _planned())
    tampered = dict(artifact)
    tampered["activation_allowed"] = True  # flip a value without re-sealing
    with pytest.raises(ReprofileRefused, match="self-seal mismatch"):
        reprof.validate_reprofile(tampered)


def test_forgery_reseal_still_rejected_on_safety_flag() -> None:
    artifact = reprof.build_reprofile_artifact(_synthetic_profile(), _planned())
    tampered = dict(artifact)
    tampered["activation_allowed"] = True
    core = {key: value for key, value in tampered.items() if key != "reprofile_sha256"}
    tampered["reprofile_sha256"] = canonical_sha256(core)  # re-seal so the seal passes
    with pytest.raises(ReprofileRefused, match="activation_allowed"):
        reprof.validate_reprofile(tampered)


def test_forgery_field_drift_is_rejected() -> None:
    artifact = reprof.build_reprofile_artifact(_synthetic_profile(), _planned())
    dropped = {key: value for key, value in artifact.items() if key != "observed"}
    with pytest.raises(ReprofileRefused, match="fields drifted"):
        reprof.validate_reprofile(dropped)

    extra = dict(artifact)
    extra["surprise"] = 1
    core = {key: value for key, value in extra.items() if key != "reprofile_sha256"}
    extra["reprofile_sha256"] = canonical_sha256(core)
    with pytest.raises(ReprofileRefused, match="fields drifted"):
        reprof.validate_reprofile(extra)


def test_determinism_double_build_is_byte_identical(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    proof_root = tmp_path / "proof"
    proof_root.mkdir()
    _build_tree(runs_root)

    profile_one = reprof.collect_observed_rates(runs_root, proof_root)
    profile_two = reprof.collect_observed_rates(runs_root, proof_root)
    artifact_one = reprof.build_reprofile_artifact(profile_one, {"mech_x": 0.2})
    artifact_two = reprof.build_reprofile_artifact(profile_two, {"mech_x": 0.2})

    assert artifact_one["reprofile_sha256"] == artifact_two["reprofile_sha256"]
    assert json.dumps(artifact_one, sort_keys=True) == json.dumps(artifact_two, sort_keys=True)


@pytest.mark.skipif(not _PROOF.is_file(), reason="sealed mechanics result not present on disk")
def test_real_result_extracts_eleven_continuing_lanes_and_pruned_p1() -> None:
    profile = reprof.collect_observed_rates(_MECHANICS_RUNS, REPO_ROOT / "proof")

    continuing = profile["continuing_lanes"]
    assert len(continuing) == 11
    assert "G1-P1" not in continuing
    assert {"G1-V1", "G1-M1", "G1-G1", "G1-C0", "G1-I1"}.issubset(set(continuing))
    assert profile["pruned_lanes"] == ["G1-P1"]
    assert profile["lane_mechanisms"]["G1-P1"] == "stability_plasticity"

    assert "stability_plasticity" not in profile["observed_rates"]
    assert "construction_search" in profile["observed_rates"]
    assert profile["observed_rates"]["construction_search"]["observed_seconds_per_seed"] == pytest.approx(
        0.0407, abs=5e-3
    )

    planned = reprof._default_planned_rates()
    artifact = reprof.build_reprofile_artifact(profile, planned)
    reprof.validate_reprofile(artifact)
    assert artifact["recommendation"]["recommended_workers"] == 16
    assert "stability_plasticity" in artifact["provisional_mechanisms"]
    assert artifact["recommendation"]["de_idealized_rates"]["construction_search"]["source"] == "observed"


@pytest.mark.skipif(not _PROOF.is_file(), reason="sealed mechanics result not present on disk")
def test_real_report_double_build_is_byte_identical() -> None:
    planned = reprof._default_planned_rates()
    profile = reprof.collect_observed_rates(_MECHANICS_RUNS, REPO_ROOT / "proof")
    artifact_one = reprof.build_reprofile_artifact(profile, planned)
    artifact_two = reprof.build_reprofile_artifact(
        reprof.collect_observed_rates(_MECHANICS_RUNS, REPO_ROOT / "proof"), planned
    )
    assert artifact_one["reprofile_sha256"] == artifact_two["reprofile_sha256"]
