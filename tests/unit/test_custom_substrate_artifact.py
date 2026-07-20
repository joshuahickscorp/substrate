import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mop.substrate.custom_artifact import (
    ARM_SCHEMA,
    ATTESTATION_SCHEMA,
    CHECKPOINT_SCHEMA,
    DATASET_SCHEMA,
    ENVIRONMENT_SCHEMA,
    IMPLEMENTATION_SCHEMA,
    REQUIREMENTS_SCHEMA,
    VERIFIER_SCHEMA,
    WORKBENCH_SCHEMA,
    ArtifactRefused,
    PortableModelSpec,
    PortableTinyVideoSubstrate,
    export_artifact,
    json_sha256,
    load_portable_artifact,
    preflight_export,
    read_tensor_pack,
    sha256_file,
    state_sha256,
    verifier_contract,
    write_tensor_pack,
)
from mop.substrate.custom_workbench import ModelSpec, TinyVideoSubstrate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fixture(run_root: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    run_dir = run_root / "run"
    run_dir.mkdir(parents=True)
    spec = PortableModelSpec(
        dim=128,
        depth=4,
        heads=4,
        mlp_ratio=4,
        patch_size=32,
        tubelet=2,
        max_resolution=256,
        max_frames=8,
    )
    torch.manual_seed(17)
    model = PortableTinyVideoSubstrate(spec)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    parameters = sum(parameter.numel() for parameter in model.parameters())

    implementation_snapshot = run_dir / "implementation_sources/custom_workbench.py"
    implementation_snapshot.parent.mkdir()
    implementation_snapshot.write_text("# independently frozen fixture implementation\n")
    implementation_hash = sha256_file(implementation_snapshot)
    implementation_files = [
        {
            "source_path": "src/mop/substrate/custom_workbench.py",
            "source_sha256": implementation_hash,
            "snapshot_path": f"elsewhere/{implementation_snapshot.name}",
            "snapshot_sha256": implementation_hash,
            "bytes": implementation_snapshot.stat().st_size,
        }
    ]
    implementation_aggregate = json_sha256(
        [{"path": row["source_path"], "sha256": row["snapshot_sha256"]} for row in implementation_files]
    )
    _write_json(
        run_dir / "implementation_manifest.json",
        {
            "schema": IMPLEMENTATION_SCHEMA,
            "all_ok": True,
            "aggregate_sha256": implementation_aggregate,
            "files": implementation_files,
        },
    )

    evidence_snapshot = run_dir / "requirements_sources/r1__00__evidence.json"
    _write_json(evidence_snapshot, {"schema": "fixture-evidence/v1", "all_ok": True})
    evidence_hash = sha256_file(evidence_snapshot)
    ledger_hash = "a" * 64
    requirement_rows = [
        {
            "id": "r1",
            "title": "fixture identity",
            "status": "required",
            "sources": [
                {
                    "path": "proof/evidence.json",
                    "role": "fixture",
                    "exists": True,
                    "bytes": evidence_snapshot.stat().st_size,
                    "sha256": evidence_hash,
                    "snapshot_path": f"elsewhere/{evidence_snapshot.name}",
                    "snapshot_sha256": evidence_hash,
                }
            ],
        }
    ]
    requirements_aggregate = json_sha256(
        {
            "ledger_sha256": ledger_hash,
            "requirements": [
                {
                    "id": "r1",
                    "sources": [{"path": "proof/evidence.json", "sha256": evidence_hash}],
                }
            ],
        }
    )
    requirements = {
        "schema": REQUIREMENTS_SCHEMA,
        "ledger_sha256": ledger_hash,
        "aggregate_sha256": requirements_aggregate,
        "requirements": requirement_rows,
        "requirement_ids": ["r1"],
        "all_ok": True,
        "problems": [],
    }
    _write_json(run_dir / "requirements_audit.json", requirements)
    _write_json(run_dir / "requirements_current_audit.json", requirements)

    config = {"model": asdict(spec), "training": {"seeds": [0], "objectives": ["predictive"]}}
    config_sha = json_sha256(config)
    _write_json(run_dir / "resolved_config.json", config)
    dataset = {
        "schema": DATASET_SCHEMA,
        "claim_scope": "deterministic programmatic video; not natural-video evidence",
        "generator": {"module": "fixture", "source_sha256": implementation_hash},
        "spec": {"resolution": 256, "frames": 8},
        "records": [{"index": 0, "referent": "fixture:0", "split": "train"}],
        "splits": {"train": [0], "val": [], "test": []},
        "disjoint_referents": True,
        "combination_disjoint": True,
    }
    dataset["content_sha256"] = json_sha256(dataset)
    _write_json(run_dir / "dataset_manifest.json", dataset)
    _write_json(
        run_dir / "teacher_audit.json",
        {"schema": "mop-custom-substrate-teacher-audit/v1", "all_ok": True, "configured": False},
    )

    arm_dir = run_dir / "arms/seed_0/predictive"
    arm_dir.mkdir(parents=True)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "objective": "predictive",
        "step": 10,
        "config_sha256": config_sha,
        "data_sha256": dataset["content_sha256"],
        "requirements_sha256": requirements_aggregate,
        "initial_state_sha256": "b" * 64,
        "model": state,
    }
    torch.save(checkpoint, arm_dir / "checkpoint.pt")
    checkpoint_hash = sha256_file(arm_dir / "checkpoint.pt")
    arm = {
        "schema": ARM_SCHEMA,
        "objective": "predictive",
        "seed": 0,
        "complete": True,
        "requested_steps": 10,
        "completed_steps": 10,
        "config_sha256": config_sha,
        "data_sha256": dataset["content_sha256"],
        "requirements_sha256": requirements_aggregate,
        "initial_state_sha256": "b" * 64,
        "final_state_sha256": state_sha256(state),
        "checkpoint": {
            "path": "ignored/checkpoint.pt",
            "bytes": (arm_dir / "checkpoint.pt").stat().st_size,
            "sha256": checkpoint_hash,
        },
    }
    _write_json(arm_dir / "arm_receipt.json", arm)

    receipt = {
        "schema": WORKBENCH_SCHEMA,
        "claim_scope": "programmatic-video objective probe; not natural-video evidence",
        "complete": True,
        "resumable": False,
        "stopped_for_wall_budget": False,
        "stopped_for_disk_floor": False,
        "config_sha256": config_sha,
        "data_sha256": dataset["content_sha256"],
        "requirements_sha256": requirements_aggregate,
        "requirements": {"all_ok": True, "requirement_ids": ["r1"]},
        "implementation": {
            "all_ok": True,
            "manifest_path": "implementation_manifest.json",
            "aggregate_sha256": implementation_aggregate,
        },
        "dataset": {
            "rows": 1,
            "resolution": 256,
            "frames": 8,
            "disjoint_referents": True,
            "combination_disjoint": True,
            "split_counts": {"train": 1, "val": 0, "test": 0},
        },
        "model": {
            "architecture": "TinyVideoSubstrate",
            "spec": asdict(spec),
            "trainable_parameters": parameters,
            "token_count": 256,
            "teacher_independent": True,
            "exports": ["dense_spatiotemporal_tokens", "pooled_retrieval_key"],
        },
        "promotion": {
            "best_objective": "predictive",
            "cm7_local_objective_lever_promotable": True,
            "cm7_reasons": [],
            "cm8_custom_build_promotable": False,
        },
        "seed_results": {"0": {"predictive": {"training": arm}}},
    }
    receipt_path = run_dir / "raw_workbench_receipt.json"
    _write_json(receipt_path, receipt)
    raw_sha = sha256_file(receipt_path)
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "raw_training_receipt_path": "raw_workbench_receipt.json",
        "raw_training_receipt_sha256": raw_sha,
        "current_audit_path": "requirements_current_audit.json",
        "current_audit_sha256": sha256_file(run_dir / "requirements_current_audit.json"),
        "scientifically_current": True,
        "training_design_snapshot_self_verifies": True,
        "implementation_snapshot_self_verifies": True,
        "requirements_semantics_unchanged": True,
        "current_evidence_all_ok": True,
        "problems": [],
    }
    attestation_path = run_dir / "current_evidence_attestation.json"
    _write_json(attestation_path, attestation)
    environment = {
        "schema": ENVIRONMENT_SCHEMA,
        "raw_training_receipt_sha256": raw_sha,
        "implementation_manifest_sha256": sha256_file(run_dir / "implementation_manifest.json"),
        "implementation_aggregate_sha256": implementation_aggregate,
        "host": {"fixture": True},
        "runtime": {"fixture": True},
        "package_locks": {"fixture": True},
        "git": {"fixture": True},
        "source_inventory_sha256": "d" * 64,
        "all_ok": True,
    }
    environment_path = run_dir / "environment_receipt.json"
    _write_json(environment_path, environment)
    verifier = verifier_contract()
    verifier.update(
        {
            "schema": VERIFIER_SCHEMA,
            "bindings": {
                "raw_training_receipt_sha256": raw_sha,
                "current_evidence_attestation_sha256": sha256_file(attestation_path),
                "environment_receipt_sha256": sha256_file(environment_path),
            },
            "selection": {
                "candidate_objectives": ["predictive", "invariance", "reconstruction"],
                "raw_winner": "predictive",
                "selection_status": "familywise-corrected",
                "family_size": 12,
                "correction": ("Holm one-sided tests plus simultaneous Bonferroni Student-t lower bounds"),
            },
            "paired_comparisons": [{"fixture": True}],
            "gates": {"fixture": True},
            "verdict": "promote-local-objective-lever",
            "promotion": True,
            "problems": [],
            "all_ok": True,
        }
    )
    verifier_path = run_dir / "independent_verifier.json"
    _write_json(verifier_path, verifier)
    composite = json.loads(json.dumps(receipt))
    composite.update(
        {
            "raw_training_receipt": {
                "path": "raw_workbench_receipt.json",
                "sha256": raw_sha,
            },
            "current_evidence_attestation": {
                "path": "current_evidence_attestation.json",
                "sha256": sha256_file(attestation_path),
            },
            "environment_receipt": {
                "path": "environment_receipt.json",
                "sha256": sha256_file(environment_path),
            },
            "independent_verifier": {
                "path": "independent_verifier.json",
                "sha256": sha256_file(verifier_path),
            },
            "authoritative_promotion": {
                "cm7_local_objective_lever_promotable": True,
                "cm8_custom_build_promotable": False,
                "verdict": "promote-local-objective-lever",
                "raw_promotion_is_preliminary": True,
                "gates": {
                    "raw_training_complete": True,
                    "evidence_current": True,
                    "environment_all_ok": True,
                    "independent_verifier_promotes": True,
                },
                "reasons": [],
                "scope_boundary": "programmatic-video objective evidence only",
            },
        }
    )
    _write_json(run_dir / "workbench_receipt.json", composite)
    return run_dir, verifier_path, state


def _rebind_verifier(run_dir: Path, verifier_path: Path) -> None:
    raw_path = run_dir / "raw_workbench_receipt.json"
    raw = json.loads(raw_path.read_text())
    raw_sha = sha256_file(raw_path)
    attestation_path = run_dir / "current_evidence_attestation.json"
    attestation = json.loads(attestation_path.read_text())
    attestation["raw_training_receipt_sha256"] = raw_sha
    _write_json(attestation_path, attestation)
    environment_path = run_dir / "environment_receipt.json"
    environment = json.loads(environment_path.read_text())
    environment["raw_training_receipt_sha256"] = raw_sha
    _write_json(environment_path, environment)
    verifier = json.loads(verifier_path.read_text())
    verifier["bindings"] = {
        "raw_training_receipt_sha256": raw_sha,
        "current_evidence_attestation_sha256": sha256_file(attestation_path),
        "environment_receipt_sha256": sha256_file(environment_path),
    }
    _write_json(verifier_path, verifier)
    composite = json.loads(json.dumps(raw))
    composite.update(
        {
            "raw_training_receipt": {"path": raw_path.name, "sha256": raw_sha},
            "current_evidence_attestation": {
                "path": attestation_path.name,
                "sha256": sha256_file(attestation_path),
            },
            "environment_receipt": {
                "path": environment_path.name,
                "sha256": sha256_file(environment_path),
            },
            "independent_verifier": {
                "path": verifier_path.name,
                "sha256": sha256_file(verifier_path),
            },
            "authoritative_promotion": {
                "cm7_local_objective_lever_promotable": True,
                "cm8_custom_build_promotable": False,
                "verdict": "promote-local-objective-lever",
                "raw_promotion_is_preliminary": True,
                "gates": {
                    "raw_training_complete": raw.get("complete") is True,
                    "evidence_current": True,
                    "environment_all_ok": True,
                    "independent_verifier_promotes": True,
                },
                "reasons": [],
                "scope_boundary": "programmatic-video objective evidence only",
            },
        }
    )
    _write_json(run_dir / "workbench_receipt.json", composite)


def test_portable_architecture_matches_workbench_interface():
    assert PortableModelSpec is ModelSpec
    assert PortableTinyVideoSubstrate is TinyVideoSubstrate
    source_spec = ModelSpec(
        dim=16,
        depth=1,
        heads=2,
        mlp_ratio=2,
        patch_size=16,
        tubelet=2,
        max_resolution=32,
        max_frames=4,
    )
    portable_spec = PortableModelSpec.from_mapping(asdict(source_spec))
    torch.manual_seed(3)
    source = TinyVideoSubstrate(source_spec).eval()
    portable = PortableTinyVideoSubstrate(portable_spec).eval()
    portable.load_state_dict(source.state_dict(), strict=True)
    clips = torch.rand(2, 3, 4, 32, 32)
    with torch.inference_mode():
        output = portable(clips)
        source_dense = source.encode(clips)
    assert torch.equal(output.dense_spatiotemporal_tokens, source_dense)
    assert torch.equal(output.pooled_retrieval_key, source_dense.mean(dim=1))


def test_canonical_model_preserves_historical_fingerprint_and_refuses_contract_mutations():
    base = asdict(ModelSpec(16, 1, 2, 2, 16, 2, 32, 4))
    for key in base:
        mutation = dict(base)
        mutation[key] = True
        with pytest.raises(ValueError, match="integers"):
            ModelSpec.from_mapping(mutation)
    for mutation in (
        {**base, "extra": 1},
        {key: value for key, value in base.items() if key != "dim"},
        {**base, "dim": 0},
        {**base, "heads": 3},
        {**base, "max_resolution": 31},
    ):
        with pytest.raises(ValueError):
            ModelSpec.from_mapping(mutation)

    torch.manual_seed(3)
    model = TinyVideoSubstrate(ModelSpec.from_mapping(base)).eval()
    clips = torch.rand(2, 3, 4, 32, 32)
    output = model(clips)

    def digest(value):
        return hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()

    assert (
        state_sha256(model.state_dict()) == "90c10469a0355530402c344fdd637901571e3211e38d165a0eddc8d82158b932"
    )
    assert (
        digest(output.dense_spatiotemporal_tokens)
        == "cebfc000f9f1b25a2ece29b6520894397644ebf6f89b9cf050eccf0524c6b10a"
    )
    assert (
        digest(output.pooled_retrieval_key)
        == "9dce9e254ec67bba04a71e155b3f4656577f4fd04b2a88d89af4437dd215553d"
    )
    with pytest.raises(ValueError, match="bool dtype"):
        model(clips, torch.zeros(2, 8))
    with pytest.raises(ValueError, match="model maxima"):
        model(torch.zeros(1, 3, 6, 32, 32))


def test_tensor_pack_is_deterministic_pickle_free_and_exact(tmp_path: Path):
    spec = PortableModelSpec(8, 1, 2, 2, 16, 2, 32, 2)
    torch.manual_seed(4)
    state = PortableTinyVideoSubstrate(spec).state_dict()
    first, second = tmp_path / "first.mopbin", tmp_path / "second.mopbin"
    first_record = write_tensor_pack(state, first)
    second_record = write_tensor_pack(state, second)
    restored, header = read_tensor_pack(first)
    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"]
    assert header["state_sha256"] == state_sha256(state)
    assert restored.keys() == state.keys()
    assert all(torch.equal(restored[name], state[name]) for name in state)


def test_export_is_content_addressed_deterministic_and_loads_offline(tmp_path: Path):
    run_dir, verifier_path, _state = _fixture(tmp_path)
    preflight = preflight_export(run_dir, verifier_path)
    assert preflight["eligible"] and not preflight["export_performed"]

    first = export_artifact(run_dir, verifier_path, tmp_path / "out-a")
    reused = export_artifact(run_dir, verifier_path, tmp_path / "out-a")
    second = export_artifact(run_dir, verifier_path, tmp_path / "out-b")
    assert not first["reused"] and reused["reused"] and not second["reused"]
    assert first["artifact_id"] == reused["artifact_id"] == second["artifact_id"]
    first_dir, second_dir = Path(first["artifact_dir"]), Path(second["artifact_dir"])
    assert (first_dir / "manifest.json").read_bytes() == (second_dir / "manifest.json").read_bytes()
    assert (first_dir / "weights.mopbin").read_bytes() == (second_dir / "weights.mopbin").read_bytes()

    loaded = load_portable_artifact(first_dir)
    clips = torch.zeros(1, 3, 2, 64, 64)
    with torch.inference_mode():
        output = loaded.model(clips)
    assert output.dense_spatiotemporal_tokens.shape == (1, 4, 128)
    assert output.pooled_retrieval_key.shape == (1, 128)
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert loaded.manifest["evidence"]["scope"]["natural_video_evidence"] is False


def test_preflight_refuses_incomplete_self_report_and_missing_verifier(tmp_path: Path):
    run_dir, verifier_path, _state = _fixture(tmp_path)
    missing = preflight_export(run_dir, tmp_path / "missing.json")
    assert not missing["eligible"] and "verifier" in missing["problems"][0]

    verifier = json.loads(verifier_path.read_text())
    verifier["selection"]["selection_status"] = "workbench-self-report"
    _write_json(verifier_path, verifier)
    self_report = preflight_export(run_dir, verifier_path)
    assert not self_report["eligible"] and "uncorrected" in self_report["problems"][0]

    receipt_path = run_dir / "raw_workbench_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["complete"] = False
    _write_json(receipt_path, receipt)
    incomplete = preflight_export(run_dir, verifier_path)
    assert not incomplete["eligible"] and "incomplete" in incomplete["problems"][0]


def test_export_refuses_checkpoint_and_model_spec_mismatch(tmp_path: Path):
    run_dir, verifier_path, _state = _fixture(tmp_path)
    checkpoint = run_dir / "arms/seed_0/predictive/checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
    drift = preflight_export(run_dir, verifier_path)
    assert not drift["eligible"] and "checkpoint" in drift["problems"][0] and "drift" in drift["problems"][0]

    run_dir, verifier_path, _state = _fixture(tmp_path / "spec")
    receipt_path = run_dir / "raw_workbench_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["model"]["spec"]["depth"] = 3
    _write_json(receipt_path, receipt)
    _rebind_verifier(run_dir, verifier_path)
    mismatch = preflight_export(run_dir, verifier_path)
    assert not mismatch["eligible"] and "model specs disagree" in mismatch["problems"][0]


def test_preflight_refuses_composite_raw_confusion_and_chain_hash_drift(tmp_path: Path):
    run_dir, verifier_path, _state = _fixture(tmp_path)
    composite_path = run_dir / "workbench_receipt.json"
    composite = json.loads(composite_path.read_text())
    composite["model"]["spec"]["depth"] = 3
    _write_json(composite_path, composite)
    confused = preflight_export(run_dir, verifier_path)
    assert not confused["eligible"] and "changed raw training field" in confused["problems"][0]

    run_dir, verifier_path, _state = _fixture(tmp_path / "binding")
    attestation_path = run_dir / "current_evidence_attestation.json"
    attestation = json.loads(attestation_path.read_text())
    attestation["source_drift"] = [{"fixture": True}]
    _write_json(attestation_path, attestation)
    drift = preflight_export(run_dir, verifier_path)
    assert not drift["eligible"] and "receipt-chain bindings" in drift["problems"][0]


def test_offline_loader_refuses_artifact_hash_drift(tmp_path: Path):
    run_dir, verifier_path, _state = _fixture(tmp_path)
    exported = export_artifact(run_dir, verifier_path, tmp_path / "out")
    artifact_dir = Path(exported["artifact_dir"])
    weights = artifact_dir / "weights.mopbin"
    content = bytearray(weights.read_bytes())
    content[-1] ^= 1
    weights.write_bytes(content)
    with pytest.raises(ArtifactRefused, match="weight hash drift"):
        load_portable_artifact(artifact_dir)
