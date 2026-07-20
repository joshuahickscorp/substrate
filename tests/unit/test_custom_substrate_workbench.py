import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import mop.substrate.custom_workbench as workbench_module
from mop.config import REPO_ROOT
from mop.devices import resolve
from mop.substrate.custom_model import state_sha256
from mop.substrate.custom_workbench import (
    CorpusSpec,
    ModelSpec,
    ProgrammaticVideoCorpus,
    TinyVideoSubstrate,
    WorkbenchRefused,
    attest_current_requirements,
    audit_requirements,
    audit_teacher_cache,
    build_referent_records,
    dataset_manifest,
    estimated_train_step_flops,
    parameter_count,
    run_workbench,
    snapshot_requirement_sources,
    train_arm,
)


def _failure_injection_config(tmp_path: Path) -> dict:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"schema": "fixture-evidence/v1", "all_ok": True}))
    ledger = tmp_path / "requirements.yaml"
    ledger.write_text(
        yaml.safe_dump(
            {
                "schema": "mop-custom-substrate-requirements/v1",
                "claim_scope": "fixture",
                "requirements": [
                    {
                        "id": "r1",
                        "title": "fail closed",
                        "status": "required",
                        "sources": [{"path": str(source), "role": "fixture"}],
                        "design_response": ["abort"],
                        "consumers": ["test"],
                        "promotion_gate": "abort",
                    }
                ],
            }
        )
    )
    return {
        "requirements_ledger": str(ledger),
        "strict_requirements": True,
        "data": {
            "resolution": 32,
            "frames": 2,
            "factor_a_levels": 4,
            "factor_b_levels": 4,
            "replicates": 3,
            "seed": 1,
        },
        "model": {
            "dim": 16,
            "depth": 1,
            "heads": 2,
            "mlp_ratio": 2,
            "patch_size": 16,
            "tubelet": 2,
            "max_resolution": 32,
            "max_frames": 2,
        },
        "training": {
            "objectives": ["predictive", "invariance", "reconstruction", "random_target"],
            "seeds": [0, 1],
            "steps": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "mask_ratio": 0.5,
            "ema_decay": 0.9,
            "variance_weight": 0.1,
            "checkpoint_every": 1,
            "wall_budget_seconds": 30.0,
            "min_free_disk_gb": 0.0,
        },
        "teacher": {"cache_path": ""},
        "promotion": {
            "min_params": 0,
            "max_params": 5_000_000,
            "min_seeds": 2,
            "margin": 0.03,
            "ceiling": 0.98,
            "compute_tolerance": 0.005,
            "teacher_min_rows": 64,
        },
    }


def _stub_evaluation(*args, **kwargs) -> dict:
    del args, kwargs
    return {
        "heldout_combo_score": 0.25,
        "referent_view_stability": 0.5,
        "heldout_oracle_form_alignment_r2": 0.0,
        "combination_geometry_gap": 0.0,
    }


def test_requirements_audit_hashes_every_machine_readable_source(tmp_path: Path):
    (tmp_path / "proof").mkdir()
    (tmp_path / "proof/a.json").write_text(json.dumps({"schema": "evidence/v1", "all_ok": True}))
    ledger = {
        "schema": "mop-custom-substrate-requirements/v1",
        "claim_scope": "fixture",
        "requirements": [
            {
                "id": "r1",
                "title": "identity",
                "status": "required",
                "sources": [{"path": "proof/a.json", "role": "fixture"}],
                "design_response": ["hash it"],
                "consumers": ["test"],
                "promotion_gate": "clean",
            }
        ],
    }
    (tmp_path / "requirements.yaml").write_text(yaml.safe_dump(ledger))
    audit = audit_requirements("requirements.yaml", repo_root=tmp_path)
    assert audit["all_ok"] and len(audit["aggregate_sha256"]) == 64
    assert len(audit["requirements"][0]["sources"][0]["sha256"]) == 64

    (tmp_path / "proof/a.json").write_text("not json")
    changed = audit_requirements("requirements.yaml", repo_root=tmp_path)
    assert not changed["all_ok"] and changed["aggregate_sha256"] != audit["aggregate_sha256"]


def test_programmatic_referents_are_deterministic_and_combination_disjoint():
    spec = CorpusSpec(
        resolution=64,
        frames=4,
        factor_a_levels=4,
        factor_b_levels=4,
        replicates=3,
        seed=11,
    )
    records = build_referent_records(spec)
    manifest = dataset_manifest(spec, records)
    assert manifest["disjoint_referents"] and manifest["combination_disjoint"]
    split_combos = {
        split: {(row.factor_a, row.factor_b) for row in records if row.split == split}
        for split in ("train", "val", "test")
    }
    assert not split_combos["train"] & split_combos["test"]
    assert {row.factor_a for row in records if row.split == "train"} == set(range(4))
    corpus = ProgrammaticVideoCorpus(spec, records)
    assert torch.equal(corpus.clip(records[0], view=7), corpus.clip(records[0], view=7))
    assert not torch.equal(corpus.clip(records[0], view=7), corpus.clip(records[0], view=8))


def test_tiny_video_substrate_is_token_preserving_and_in_1_to_5m_envelope():
    spec = ModelSpec(
        dim=128,
        depth=4,
        heads=4,
        patch_size=16,
        tubelet=2,
        max_resolution=64,
        max_frames=4,
    )
    model = TinyVideoSubstrate(spec)
    assert 1_000_000 <= parameter_count(model) <= 5_000_000
    clips = torch.rand(2, 3, 4, 64, 64)
    tokens = model.encode(clips)
    assert tokens.shape == (2, 32, 128)
    output = model(clips)
    assert output.dense_spatiotemporal_tokens.shape == (2, 32, 128)
    assert output.pooled_retrieval_key.shape == (2, 128)


def test_objective_flop_estimates_share_one_matched_core():
    data = CorpusSpec(resolution=256, frames=8, factor_a_levels=4, factor_b_levels=4, replicates=3)
    model = ModelSpec()
    estimates = {
        objective: estimated_train_step_flops(data, model, batch_size=4, objective=objective)
        for objective in ("predictive", "invariance", "reconstruction", "random_target")
    }
    mean = sum(estimates.values()) / len(estimates)
    assert max(abs(value - mean) / mean for value in estimates.values()) <= 0.005


def test_missing_teacher_is_optional_and_explicit():
    audit = audit_teacher_cache(None)
    assert audit["all_ok"] and not audit["configured"] and not audit["available"]


def test_current_evidence_attestation_preserves_snapshots_and_fails_on_schema_drift(
    tmp_path: Path,
):
    (tmp_path / "proof").mkdir()
    source = tmp_path / "proof/source.json"
    source.write_text(json.dumps({"schema": "evidence/v1", "value": 1}))
    ledger = {
        "schema": "mop-custom-substrate-requirements/v1",
        "claim_scope": "fixture",
        "requirements": [
            {
                "id": "r1",
                "title": "identity",
                "status": "required",
                "sources": [{"path": "proof/source.json", "role": "fixture"}],
                "design_response": ["preserve"],
                "consumers": ["test"],
                "promotion_gate": "clean",
            }
        ],
    }
    (tmp_path / "requirements.yaml").write_text(yaml.safe_dump(ledger))
    run_dir = tmp_path / "runs/pilot"
    run_dir.mkdir(parents=True)
    start = audit_requirements("requirements.yaml", repo_root=tmp_path)
    snapshot_requirement_sources(start, destination=run_dir / "requirements_sources", repo_root=tmp_path)
    (run_dir / "requirements_audit.json").write_text(json.dumps(start))
    (run_dir / "resolved_config.json").write_text(json.dumps({"requirements_ledger": "requirements.yaml"}))
    (run_dir / "workbench_receipt.json").write_text(
        json.dumps(
            {
                "promotion": {
                    "cm7_local_objective_lever_promotable": True,
                    "cm8_custom_build_promotable": False,
                    "cm7_reasons": [],
                    "cm8_reasons": [],
                }
            }
        )
    )
    implementation_snapshot = run_dir / "implementation_sources/core.py"
    implementation_snapshot.parent.mkdir()
    implementation_snapshot.write_text("# frozen implementation\n")
    implementation_hash = hashlib.sha256(implementation_snapshot.read_bytes()).hexdigest()
    (run_dir / "implementation_manifest.json").write_text(
        json.dumps(
            {
                "schema": "mop-custom-substrate-implementation-snapshot/v1",
                "files": [
                    {
                        "source_path": "core.py",
                        "snapshot_path": str(implementation_snapshot),
                        "snapshot_sha256": implementation_hash,
                    }
                ],
            }
        )
    )
    current = attest_current_requirements(run_dir, repo_root=tmp_path)
    assert current["evidence_attestation"]["scientifically_current"]

    source.write_text(json.dumps({"schema": "evidence/v2", "value": 2}))
    refused = attest_current_requirements(run_dir, repo_root=tmp_path)
    assert not refused["evidence_attestation"]["scientifically_current"]
    assert not refused["promotion"]["cm7_local_objective_lever_promotable"]
    assert any("schema drifted" in problem for problem in refused["evidence_attestation"]["problems"])


def test_required_arm_refusal_is_durable_and_aborts_all_loops(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def refuse(**kwargs):
        calls.append(kwargs["objective"])
        raise WorkbenchRefused("injected non-finite loss")

    monkeypatch.setattr(workbench_module, "train_arm", refuse)
    monkeypatch.setattr(workbench_module, "evaluate_model", _stub_evaluation)
    run_dir = tmp_path / "refusal"
    receipt = run_workbench(
        _failure_injection_config(tmp_path),
        run_dir=run_dir,
        device=resolve("cpu"),
        repo_root=REPO_ROOT,
    )
    assert calls == ["predictive"]
    assert not receipt["complete"] and receipt["stopped_for_required_arm_refusal"]
    assert not receipt["promotion"]["cm7_local_objective_lever_promotable"]
    failure_path = run_dir / "arms/seed_0/predictive/refusal_receipt.json"
    failure = json.loads(failure_path.read_text())
    assert failure["campaign_must_abort"] and failure["last_good_checkpoint"] is None
    assert "1" not in receipt["seed_results"]


def test_unexpected_arm_exception_writes_crash_then_reraises(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def crash(**kwargs):
        calls.append(kwargs["objective"])
        raise RuntimeError("injected backend recovery")

    monkeypatch.setattr(workbench_module, "train_arm", crash)
    monkeypatch.setattr(workbench_module, "evaluate_model", _stub_evaluation)
    run_dir = tmp_path / "crash"
    with pytest.raises(RuntimeError, match="backend recovery"):
        run_workbench(
            _failure_injection_config(tmp_path),
            run_dir=run_dir,
            device=resolve("cpu"),
            repo_root=REPO_ROOT,
        )
    assert calls == ["predictive"]
    crash_path = run_dir / "arms/seed_0/predictive/crash_receipt.json"
    crash_receipt = json.loads(crash_path.read_text())
    assert crash_receipt["campaign_must_abort"]
    assert "RuntimeError: injected backend recovery" in crash_receipt["traceback"]


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path):
    data = CorpusSpec(
        resolution=32,
        frames=2,
        factor_a_levels=4,
        factor_b_levels=4,
        replicates=3,
        seed=19,
    )
    model_spec = ModelSpec(
        dim=16,
        depth=1,
        heads=2,
        mlp_ratio=2,
        patch_size=16,
        tubelet=2,
        max_resolution=32,
        max_frames=2,
    )
    records = build_referent_records(data)
    corpus = ProgrammaticVideoCorpus(data, records)
    torch.manual_seed(5)
    initial = TinyVideoSubstrate(model_spec).state_dict()
    initial = {key: value.detach().clone() for key, value in initial.items()}
    common = {
        "objective": "predictive",
        "seed": 5,
        "corpus": corpus,
        "records": records,
        "data_spec": data,
        "model_spec": model_spec,
        "initial_state": initial,
        "device": resolve("cpu"),
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "mask_ratio": 0.5,
        "ema_decay": 0.9,
        "variance_weight": 0.1,
        "checkpoint_every": 1,
        "config_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "requirements_sha256": "r" * 64,
    }
    full = train_arm(**common, arm_dir=tmp_path / "full", steps=4)
    train_arm(**common, arm_dir=tmp_path / "resume", steps=2)
    resumed = train_arm(**common, arm_dir=tmp_path / "resume", steps=4)
    assert resumed["resumed"] and resumed["completed_steps"] == 4
    assert full["final_state_sha256"] == resumed["final_state_sha256"]
    checkpoint = torch.load(tmp_path / "resume/checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["initial_state_sha256"] == state_sha256(initial)
