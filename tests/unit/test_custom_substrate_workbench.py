import json
from dataclasses import asdict
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
    audit_requirements,
    audit_teacher_cache,
    build_referent_records,
    dataset_manifest,
    estimated_train_step_flops,
    parameter_count,
    run_workbench,
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
        "data": asdict(CorpusSpec(resolution=32, frames=2, replicates=3, seed=1)),
        "model": asdict(ModelSpec(16, 1, 2, 2, 16, 2, 32, 2)),
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


@pytest.mark.parametrize(
    ("error", "receipt_name"),
    [
        (WorkbenchRefused("injected non-finite loss"), "refusal_receipt.json"),
        (RuntimeError("injected backend recovery"), "crash_receipt.json"),
    ],
    ids=("scientific-refusal", "unexpected-exception"),
)
def test_required_arm_failure_is_durable_and_aborts(
    tmp_path: Path, monkeypatch, error: Exception, receipt_name: str
):
    calls: list[str] = []

    def fail(**kwargs):
        calls.append(kwargs["objective"])
        raise error

    monkeypatch.setattr(workbench_module, "train_arm", fail)
    monkeypatch.setattr(workbench_module, "evaluate_model", _stub_evaluation)
    run_dir = tmp_path / receipt_name.removesuffix("_receipt.json")
    arguments = {
        "config": _failure_injection_config(tmp_path),
        "run_dir": run_dir,
        "device": resolve("cpu"),
        "repo_root": REPO_ROOT,
    }
    if isinstance(error, WorkbenchRefused):
        receipt = run_workbench(**arguments)
    else:
        with pytest.raises(type(error), match=str(error)):
            run_workbench(**arguments)
        receipt = None
    assert calls == ["predictive"]
    failure = json.loads((run_dir / "arms/seed_0/predictive" / receipt_name).read_text())
    assert failure["campaign_must_abort"] and failure["last_good_checkpoint"] is None
    if receipt is not None:
        assert not receipt["complete"] and receipt["stopped_for_required_arm_refusal"]
        assert not receipt["promotion"]["cm7_local_objective_lever_promotable"]
        assert "1" not in receipt["seed_results"]
    else:
        assert f"{type(error).__name__}: {error}" in failure["traceback"]


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path):
    config = _failure_injection_config(tmp_path)
    data = CorpusSpec(**{**config["data"], "seed": 19})
    model_spec = ModelSpec(**config["model"])
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
