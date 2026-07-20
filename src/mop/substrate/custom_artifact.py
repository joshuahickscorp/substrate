from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from mop.evidence import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    canonical_sha256,
    sha256_file,
)

from .custom_model import ModelSpec as PortableModelSpec
from .custom_model import TinyVideoSubstrate as PortableTinyVideoSubstrate
from .custom_model import parameter_count, state_sha256

ARTIFACT_SCHEMA = "mop-portable-custom-substrate/v1"
TENSOR_PACK_SCHEMA = "mop-portable-tensor-pack/v1"
_MAGIC = b"MOPTVS1\n"

EVIDENCE_SCOPE: dict[str, Any] = {
    "training_evidence": "deterministic_programmatic_video_only",
    "programmatic_video": True,
    "natural_video_evidence": False,
    "general_capability_evidence": False,
    "sentience_evidence": False,
    "allowed_claim": (
        "independently verified CM7 objective selection on the deterministic programmatic-video task"
    ),
}

INTERFACE_SCHEMA: dict[str, Any] = {
    "schema": "mop-portable-video-substrate-interface/v1",
    "input": {
        "name": "clips",
        "layout": "batch,channel,time,height,width",
        "channels": 3,
        "value_domain": "float tensor; CM7 training used values in [0,1]",
        "geometry": "time/height/width divide exactly by tubelet/patch and do not exceed maxima",
    },
    "optional_mask": {
        "layout": "batch,dense_token",
        "dtype": "bool",
        "meaning": "replace selected embedded patches with the learned mask token before encoding",
    },
    "outputs": {
        "dense_spatiotemporal_tokens": {
            "layout": "batch,dense_token,feature",
            "token_order": "time-major then row-major then column-major (column varies fastest)",
        },
        "pooled_retrieval_key": {
            "layout": "batch,feature",
            "reduction": "arithmetic mean of dense_spatiotemporal_tokens over dense_token",
        },
    },
}


class ArtifactRefused(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactRefused(message)


def _object(value: Any, message: str) -> dict[str, Any]:
    _require(isinstance(value, dict), message)
    return cast(dict[str, Any], value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _little_endian_bytes(tensor: torch.Tensor) -> bytes:
    _require(tensor.layout == torch.strided, "tensor pack refuses sparse or non-strided tensors")
    _require(not tensor.is_quantized, "tensor pack refuses quantized tensors")
    cpu = tensor.detach().cpu().contiguous()
    _require(cpu.dtype in _DTYPE_TO_NAME, f"tensor pack does not support dtype {cpu.dtype}")
    array = cpu.numpy()
    return array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")


def write_tensor_pack(state: Mapping[str, torch.Tensor], path: Path) -> dict[str, Any]:
    _require(bool(state), "cannot export an empty state")
    _require(all(isinstance(name, str) and name for name in state), "state names must be non-empty strings")
    names = sorted(state)
    raw_parts: list[bytes] = []
    tensors: list[dict[str, Any]] = []
    offset = 0
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        raw = _little_endian_bytes(tensor)
        tensors.append(
            {
                "name": name,
                "dtype": _DTYPE_TO_NAME[tensor.dtype],
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        raw_parts.append(raw)
        offset += len(raw)
    header = {
        "schema": TENSOR_PACK_SCHEMA,
        "byte_order": "little",
        "state_sha256": state_sha256(state),
        "tensor_count": len(tensors),
        "payload_bytes": offset,
        "tensors": tensors,
    }
    header_bytes = canonical_bytes(header)
    atomic_write_bytes(
        path,
        b"".join((_MAGIC, struct.pack(">Q", len(header_bytes)), header_bytes, *raw_parts)),
    )
    return {
        "format": TENSOR_PACK_SCHEMA,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "state_sha256": header["state_sha256"],
        "tensor_count": len(tensors),
    }


def read_tensor_pack(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _require(path.is_file(), f"tensor pack missing: {path}")
    content = path.read_bytes()
    prefix_bytes = len(_MAGIC) + 8
    _require(len(content) >= prefix_bytes and content.startswith(_MAGIC), "tensor pack magic mismatch")
    header_length = struct.unpack(">Q", content[len(_MAGIC) : prefix_bytes])[0]
    _require(0 < header_length <= 16 * 1024 * 1024, "tensor pack header length is invalid")
    payload_start = prefix_bytes + header_length
    _require(payload_start <= len(content), "tensor pack header is truncated")
    try:
        header = json.loads(content[prefix_bytes:payload_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactRefused(f"tensor pack header is invalid: {exc}") from exc
    header = _object(header, "tensor pack header must be an object")
    _require(header.get("schema") == TENSOR_PACK_SCHEMA, "tensor pack schema mismatch")
    _require(header.get("byte_order") == "little", "tensor pack byte order mismatch")
    rows = header.get("tensors")
    _require(isinstance(rows, list) and bool(rows), "tensor pack tensor table is empty")
    rows = cast(list[Any], rows)
    _require(header.get("tensor_count") == len(rows), "tensor pack tensor count mismatch")
    _require(len(rows) <= 100_000, "tensor pack tensor table is unreasonably large")
    payload = memoryview(content)[payload_start:]
    _require(header.get("payload_bytes") == len(payload), "tensor pack payload length mismatch")
    state: dict[str, torch.Tensor] = {}
    expected_offset = 0
    prior_name = ""
    for value in rows:
        row = _object(value, "tensor pack row must be an object")
        name, dtype_name, shape = row.get("name"), row.get("dtype"), row.get("shape")
        _require(isinstance(name, str) and name > prior_name, "tensor names must be unique and sorted")
        _require(dtype_name in _NAME_TO_DTYPE, f"unsupported packed dtype {dtype_name!r}")
        _require(
            isinstance(shape, list)
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape),
            f"invalid shape for tensor {name!r}",
        )
        name, dtype_name = cast(str, name), cast(str, dtype_name)
        shape = cast(list[int], shape)
        dtype = _NAME_TO_DTYPE[dtype_name]
        item_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = math.prod(shape) * item_size
        _require(row.get("offset") == expected_offset, f"non-contiguous payload offset for tensor {name!r}")
        _require(row.get("nbytes") == expected_bytes, f"payload size mismatch for tensor {name!r}")
        stop = expected_offset + expected_bytes
        _require(stop <= len(payload), f"payload is truncated for tensor {name!r}")
        raw = bytearray(payload[expected_offset:stop])
        if sys.byteorder == "big" and item_size > 1:
            for start in range(0, len(raw), item_size):
                raw[start : start + item_size] = reversed(raw[start : start + item_size])
        state[name] = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
        expected_offset, prior_name = stop, name
    _require(expected_offset == len(payload), "tensor pack has trailing payload bytes")
    actual_state_hash = state_sha256(state)
    _require(_is_sha256(header.get("state_sha256")), "tensor pack state hash is malformed")
    _require(actual_state_hash == header["state_sha256"], "tensor pack state hash mismatch")
    return state, header


def _safe_path(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is missing")
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path is unsafe")
    return root / relative


def _manifest(
    model: PortableTinyVideoSubstrate,
    weights: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    _require(bool(bindings), "portable evidence bindings are empty")
    _require(
        all(isinstance(key, str) and key and _is_sha256(value) for key, value in bindings.items()),
        "portable evidence binding is invalid",
    )
    state = model.state_dict()
    return {
        "schema": ARTIFACT_SCHEMA,
        "model": {
            "architecture": "PortableTinyVideoSubstrate",
            "source_architecture": "TinyVideoSubstrate",
            "spec": asdict(model.spec),
            "trainable_parameters": parameter_count(model),
            "state_sha256": state_sha256(state),
            "teacher_independent": True,
        },
        "interface": INTERFACE_SCHEMA,
        "evidence": {"scope": EVIDENCE_SCOPE, "bindings": dict(sorted(bindings.items()))},
        "weights": dict(weights),
    }


def export_artifact(
    model: PortableTinyVideoSubstrate,
    output_root: Path,
    *,
    evidence_bindings: Mapping[str, str],
) -> dict[str, Any]:
    _require(isinstance(model, PortableTinyVideoSubstrate), "portable export requires the canonical model")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch = output_root / f".portable-export-{os.getpid()}"
    _require(not scratch.exists(), f"temporary export path already exists: {scratch}")
    scratch.mkdir()
    try:
        weights = write_tensor_pack(model.state_dict(), scratch / "weights.mopbin")
        manifest = _manifest(model, weights, evidence_bindings)
        manifest["artifact_id"] = canonical_sha256(manifest)
        atomic_write_json(scratch / "manifest.json", manifest)
        target = output_root / f"tiny-video-substrate-{manifest['artifact_id']}"
        reused = target.exists()
        if reused:
            shutil.rmtree(scratch)
        else:
            os.replace(scratch, target)
        _require(load_portable_artifact(target).manifest == manifest, "portable artifact verification drift")
        return {
            "artifact_id": manifest["artifact_id"],
            "artifact_dir": str(target),
            "manifest_sha256": sha256_file(target / "manifest.json"),
            "reused": reused,
        }
    except Exception:
        if scratch.exists():
            shutil.rmtree(scratch)
        raise


@dataclass(frozen=True)
class LoadedPortableArtifact:
    manifest: dict[str, Any]
    model: PortableTinyVideoSubstrate


def load_portable_artifact(
    artifact_dir: Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPortableArtifact:
    root = artifact_dir.resolve()
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"portable artifact manifest missing: {manifest_path}")
    try:
        manifest = _object(json.loads(manifest_path.read_text()), "portable manifest must be an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactRefused(f"portable artifact manifest is invalid: {exc}") from exc
    _require(manifest.get("schema") == ARTIFACT_SCHEMA, "portable artifact schema mismatch")
    recorded_id = manifest.get("artifact_id")
    identity = dict(manifest)
    identity.pop("artifact_id", None)
    _require(_is_sha256(recorded_id), "portable artifact id is malformed")
    _require(canonical_sha256(identity) == recorded_id, "portable artifact content id mismatch")
    _require(manifest.get("interface") == INTERFACE_SCHEMA, "portable interface schema mismatch")
    evidence = _object(manifest.get("evidence"), "portable evidence is missing")
    _require(evidence.get("scope") == EVIDENCE_SCOPE, "portable evidence scope drift")
    bindings = _object(evidence.get("bindings"), "portable evidence bindings are missing")
    _require(
        bool(bindings) and all(_is_sha256(value) for value in bindings.values()),
        "portable evidence binding is invalid",
    )

    weights = _object(manifest.get("weights"), "portable weight identity is missing")
    model_record = _object(manifest.get("model"), "portable model identity is missing")
    _require(weights.get("format") == TENSOR_PACK_SCHEMA, "portable weight format mismatch")
    weight_path = _safe_path(root, weights.get("path"), "portable weights")
    _require(weight_path.is_file(), "portable weights are missing")
    _require(weight_path.stat().st_size == weights.get("bytes"), "portable weight size drift")
    _require(sha256_file(weight_path) == weights.get("sha256"), "portable weight hash drift")
    state, header = read_tensor_pack(weight_path)
    state_hash = state_sha256(state)
    _require(header.get("state_sha256") == state_hash, "packed state header drift")
    _require(weights.get("state_sha256") == state_hash, "packed weight state binding drift")
    _require(model_record.get("state_sha256") == state_hash, "portable model state binding drift")
    _require(model_record.get("architecture") == "PortableTinyVideoSubstrate", "architecture mismatch")
    _require(model_record.get("source_architecture") == "TinyVideoSubstrate", "source architecture drift")
    _require(model_record.get("teacher_independent") is True, "artifact is not teacher independent")
    spec_value = _object(model_record.get("spec"), "portable model spec is missing")
    try:
        spec = PortableModelSpec.from_mapping(spec_value)
    except ValueError as exc:
        raise ArtifactRefused(str(exc)) from exc
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PortableTinyVideoSubstrate(spec)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactRefused(f"portable state does not match its model spec: {exc}") from exc
    _require(
        parameter_count(model) == model_record.get("trainable_parameters"), "portable parameter count drift"
    )
    model.requires_grad_(False).to(torch.device(device)).eval()
    return LoadedPortableArtifact(manifest=manifest, model=model)
