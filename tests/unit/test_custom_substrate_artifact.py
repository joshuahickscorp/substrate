import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mop.evidence import atomic_write_json, canonical_sha256
from mop.substrate.custom_artifact import (
    ArtifactRefused,
    PortableModelSpec,
    PortableTinyVideoSubstrate,
    export_artifact,
    load_portable_artifact,
    read_tensor_pack,
    state_sha256,
    write_tensor_pack,
)
from mop.substrate.custom_workbench import ModelSpec, TinyVideoSubstrate


def _model() -> PortableTinyVideoSubstrate:
    torch.manual_seed(17)
    return PortableTinyVideoSubstrate(PortableModelSpec(16, 1, 2, 2, 16, 2, 32, 4)).eval()


def _export(tmp_path: Path) -> tuple[dict, Path]:
    result = export_artifact(_model(), tmp_path / "out", evidence_bindings={"cm7_receipt": "a" * 64})
    return result, Path(result["artifact_dir"])


def _reidentify(path: Path, mutation) -> None:
    manifest = json.loads(path.read_text())
    mutation(manifest)
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = canonical_sha256(identity)
    atomic_write_json(path, manifest)


def test_portable_architecture_matches_workbench_interface():
    assert PortableModelSpec is ModelSpec
    assert PortableTinyVideoSubstrate is TinyVideoSubstrate
    source_spec = ModelSpec(16, 1, 2, 2, 16, 2, 32, 4)
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
        with pytest.raises(ValueError, match="integers"):
            ModelSpec.from_mapping({**base, key: True})
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
    state = _model().state_dict()
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
    model = _model()
    first = export_artifact(model, tmp_path / "out-a", evidence_bindings={"cm7_receipt": "a" * 64})
    reused = export_artifact(model, tmp_path / "out-a", evidence_bindings={"cm7_receipt": "a" * 64})
    second = export_artifact(model, tmp_path / "out-b", evidence_bindings={"cm7_receipt": "a" * 64})
    assert not first["reused"] and reused["reused"] and not second["reused"]
    assert first["artifact_id"] == reused["artifact_id"] == second["artifact_id"]
    first_dir, second_dir = Path(first["artifact_dir"]), Path(second["artifact_dir"])
    assert (first_dir / "manifest.json").read_bytes() == (second_dir / "manifest.json").read_bytes()
    assert (first_dir / "weights.mopbin").read_bytes() == (second_dir / "weights.mopbin").read_bytes()

    loaded = load_portable_artifact(first_dir)
    clips = torch.zeros(1, 3, 4, 32, 32)
    with torch.inference_mode():
        expected, actual = model(clips), loaded.model(clips)
    assert torch.equal(expected.dense_spatiotemporal_tokens, actual.dense_spatiotemporal_tokens)
    assert torch.equal(expected.pooled_retrieval_key, actual.pooled_retrieval_key)
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert loaded.manifest["evidence"]["scope"]["natural_video_evidence"] is False


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("identity", "content id mismatch"),
        ("weights", "weight hash drift"),
        ("binding", "evidence binding is invalid"),
        ("interface", "interface schema mismatch"),
        ("path", "path is unsafe"),
    ],
)
def test_offline_loader_refuses_identity_and_weight_mutations(tmp_path: Path, case: str, expected: str):
    _result, artifact_dir = _export(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    if case == "weights":
        weights = artifact_dir / "weights.mopbin"
        content = bytearray(weights.read_bytes())
        content[-1] ^= 1
        weights.write_bytes(content)
    elif case == "identity":
        manifest = json.loads(manifest_path.read_text())
        manifest["model"]["architecture"] = "drift"
        atomic_write_json(manifest_path, manifest)
    elif case == "binding":
        _reidentify(manifest_path, lambda value: value["evidence"]["bindings"].update(cm7_receipt="bad"))
    elif case == "interface":
        _reidentify(manifest_path, lambda value: value["interface"].update(schema="drift"))
    else:
        _reidentify(manifest_path, lambda value: value["weights"].update(path="../weights.mopbin"))
    with pytest.raises(ArtifactRefused, match=expected):
        load_portable_artifact(artifact_dir)
