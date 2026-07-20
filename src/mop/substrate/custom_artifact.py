from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from mop.evidence import atomic_write_json, canonical_bytes, canonical_sha256, sha256_file

from .custom_model import (
    ModelSpec as PortableModelSpec,
)
from .custom_model import (
    TinyVideoSubstrate as PortableTinyVideoSubstrate,
)
from .custom_model import (
    parameter_count,
    state_sha256,
)

ARTIFACT_SCHEMA = "mop-portable-custom-substrate/v1"
PREFLIGHT_SCHEMA = "mop-portable-custom-substrate-preflight/v1"
VERIFIER_SCHEMA = "mop-custom-substrate-cm7-independent-verifier/v1"
WORKBENCH_SCHEMA = "mop-custom-substrate-workbench/v1"
ATTESTATION_SCHEMA = "mop-custom-substrate-current-evidence-attestation/v1"
ENVIRONMENT_SCHEMA = "mop-custom-substrate-environment-receipt/v1"
ARM_SCHEMA = "mop-custom-substrate-arm/v1"
CHECKPOINT_SCHEMA = "mop-custom-substrate-checkpoint/v1"
DATASET_SCHEMA = "mop-custom-substrate-dataset/v1"
REQUIREMENTS_SCHEMA = "mop-custom-substrate-requirements/v1"
IMPLEMENTATION_SCHEMA = "mop-custom-substrate-implementation-snapshot/v1"
TENSOR_PACK_SCHEMA = "mop-portable-tensor-pack/v1"

_MAGIC = b"MOPTVS1\n"
_LEARNED_OBJECTIVES = ("predictive", "invariance", "reconstruction")
_SELECTION_RULE = "lowest_complete_seed_for_recomputed_best_objective"
_VERIFIER_CORRECTION = "Holm one-sided tests plus simultaneous Bonferroni Student-t lower bounds"
_VERIFIER_FAMILY_SIZE = 12
_CHAIN_FILENAMES = {
    "raw_training_receipt": "raw_workbench_receipt.json",
    "final_composite_receipt": "workbench_receipt.json",
    "current_evidence_attestation": "current_evidence_attestation.json",
    "environment_receipt": "environment_receipt.json",
    "independent_verifier": "independent_verifier.json",
    "implementation_manifest": "implementation_manifest.json",
    "current_requirements_audit": "requirements_current_audit.json",
}
_CHAIN_ROLES = (*_CHAIN_FILENAMES, "selected_arm_receipt")

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


json_sha256 = canonical_sha256
_atomic_json = atomic_write_json


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactRefused(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactRefused(f"{label} is not valid JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _model_spec(value: Mapping[str, Any]) -> PortableModelSpec:
    try:
        return PortableModelSpec.from_mapping(value)
    except ValueError as exc:
        raise ArtifactRefused(str(exc)) from exc


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
    little_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_dtype, copy=False).tobytes(order="C")


def write_tensor_pack(state: Mapping[str, torch.Tensor], path: Path) -> dict[str, Any]:

    _require(bool(state), "cannot export an empty state")
    _require(all(isinstance(name, str) and name for name in state), "state names must be non-empty strings")
    names = sorted(state)
    _require(len(names) == len(set(names)), "state contains duplicate tensor names")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_MAGIC)
            handle.write(struct.pack(">Q", len(header_bytes)))
            handle.write(header_bytes)
            for raw in raw_parts:
                handle.write(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
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
    _require(isinstance(header, dict), "tensor pack header must be an object")
    header = cast(dict[str, Any], header)
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
    for row in rows:
        _require(isinstance(row, dict), "tensor pack row must be an object")
        row = cast(dict[str, Any], row)
        name, dtype_name, shape = row.get("name"), row.get("dtype"), row.get("shape")
        _require(isinstance(name, str) and name > prior_name, "tensor names must be unique and sorted")
        _require(dtype_name in _NAME_TO_DTYPE, f"unsupported packed dtype {dtype_name!r}")
        _require(
            isinstance(shape, list)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape),
            f"invalid shape for tensor {name!r}",
        )
        shape = cast(list[int], shape)
        name = cast(str, name)
        dtype_name = cast(str, dtype_name)
        dtype = _NAME_TO_DTYPE[dtype_name]
        item_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = math.prod(shape) * item_size
        offset, byte_count = row.get("offset"), row.get("nbytes")
        _require(offset == expected_offset, f"non-contiguous payload offset for tensor {name!r}")
        _require(byte_count == expected_bytes, f"payload size mismatch for tensor {name!r}")
        stop = expected_offset + expected_bytes
        _require(stop <= len(payload), f"payload is truncated for tensor {name!r}")
        raw = bytearray(payload[expected_offset:stop])
        if sys.byteorder == "big" and item_size > 1:
            for start in range(0, len(raw), item_size):
                raw[start : start + item_size] = reversed(raw[start : start + item_size])
        tensor = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
        state[name] = tensor
        expected_offset = stop
        prior_name = name
    _require(expected_offset == len(payload), "tensor pack has trailing payload bytes")
    actual_state_hash = state_sha256(state)
    _require(_is_sha256(header.get("state_sha256")), "tensor pack state hash is malformed")
    _require(actual_state_hash == header["state_sha256"], "tensor pack state hash mismatch")
    return state, header


def verifier_contract() -> dict[str, Any]:

    return {
        "schema": VERIFIER_SCHEMA,
        "all_ok": True,
        "bindings": {
            "raw_training_receipt_sha256": "<sha256 R of immutable raw_workbench_receipt.json>",
            "current_evidence_attestation_sha256": "<sha256 E of current_evidence_attestation.json>",
            "environment_receipt_sha256": "<sha256 H of environment_receipt.json>",
        },
        "selection": {
            "candidate_objectives": list(_LEARNED_OBJECTIVES),
            "raw_winner": "<predictive|invariance|reconstruction>",
            "selection_status": "familywise-corrected",
            "family_size": _VERIFIER_FAMILY_SIZE,
            "correction": _VERIFIER_CORRECTION,
        },
        "paired_comparisons": "<independently recomputed comparison rows from raw R>",
        "gates": "<independently recomputed gate rows>",
        "verdict": "promote-local-objective-lever",
        "promotion": True,
        "problems": [],
    }


@dataclass(frozen=True)
class _SourceFile:
    role: str
    source: Path
    artifact_path: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class _PreparedExport:
    receipt: dict[str, Any]
    composite: dict[str, Any]
    verifier: dict[str, Any]
    checkpoint_sha256: str
    state: dict[str, torch.Tensor]
    spec: PortableModelSpec
    seed: int
    objective: str
    sources: tuple[_SourceFile, ...]


def _require_raw_receipt(receipt: Mapping[str, Any]) -> None:

    _require(receipt.get("schema") == WORKBENCH_SCHEMA, "raw training receipt schema mismatch")
    _require(receipt.get("complete") is True, "raw training receipt is incomplete")
    _require(receipt.get("resumable") is False, "raw training receipt is still resumable")
    _require(receipt.get("stopped_for_wall_budget") is False, "raw training stopped for its wall budget")
    _require(receipt.get("stopped_for_disk_floor") is False, "raw training stopped for its disk floor")
    _require(isinstance(receipt.get("model"), dict), "raw training model record is missing")
    _require(isinstance(receipt.get("seed_results"), dict), "raw training seed results are missing")


def _require_attestation(
    attestation: Mapping[str, Any],
    *,
    raw_receipt_sha256: str,
    current_audit_sha256: str,
) -> None:
    _require(attestation.get("schema") == ATTESTATION_SCHEMA, "current-evidence attestation schema mismatch")
    _require(
        attestation.get("raw_training_receipt_sha256") == raw_receipt_sha256,
        "current-evidence attestation binds a different raw training receipt",
    )
    raw_path = attestation.get("raw_training_receipt_path")
    _require(
        isinstance(raw_path, str) and Path(raw_path).name == "raw_workbench_receipt.json",
        "attestation raw training path is invalid",
    )
    _require(
        attestation.get("current_audit_sha256") == current_audit_sha256,
        "current-evidence audit hash drift",
    )
    current_path = attestation.get("current_audit_path")
    _require(
        isinstance(current_path, str) and Path(current_path).name == "requirements_current_audit.json",
        "current-evidence audit path is invalid",
    )
    for key in (
        "scientifically_current",
        "training_design_snapshot_self_verifies",
        "implementation_snapshot_self_verifies",
        "requirements_semantics_unchanged",
        "current_evidence_all_ok",
    ):
        _require(attestation.get(key) is True, f"current-evidence attestation failed: {key}")
    _require(not attestation.get("problems"), "current-evidence attestation records problems")


def _require_environment(
    environment: Mapping[str, Any],
    *,
    raw_receipt_sha256: str,
    implementation_manifest_sha256: str,
    implementation_aggregate_sha256: str,
) -> None:
    _require(environment.get("schema") == ENVIRONMENT_SCHEMA, "environment receipt schema mismatch")
    _require(environment.get("all_ok") is True, "environment receipt did not pass")
    _require(
        environment.get("raw_training_receipt_sha256") == raw_receipt_sha256,
        "environment receipt binds a different raw training receipt",
    )
    _require(
        environment.get("implementation_manifest_sha256") == implementation_manifest_sha256,
        "environment receipt implementation-manifest hash drift",
    )
    _require(
        environment.get("implementation_aggregate_sha256") == implementation_aggregate_sha256,
        "environment receipt implementation aggregate drift",
    )
    _require(_is_sha256(environment.get("source_inventory_sha256")), "environment source inventory missing")
    for key in ("host", "runtime", "package_locks", "git"):
        _require(environment.get(key) is not None, f"environment receipt field missing: {key}")


def _require_verifier(
    verifier: Mapping[str, Any],
    *,
    raw_receipt_sha256: str,
    attestation_sha256: str,
    environment_sha256: str,
    receipt: Mapping[str, Any],
) -> tuple[str, int]:
    _require(verifier.get("schema") == VERIFIER_SCHEMA, "independent verifier schema mismatch")
    _require(verifier.get("all_ok") is True, "independent verifier did not pass")
    _require(not verifier.get("problems"), "independent verifier records problems")
    _require(
        verifier.get("verdict") == "promote-local-objective-lever" and verifier.get("promotion") is True,
        "independent verifier did not promote the local objective lever",
    )
    bindings = verifier.get("bindings")
    _require(isinstance(bindings, dict), "independent verifier bindings are missing")
    bindings = cast(dict[str, Any], bindings)
    _require(
        bindings
        == {
            "raw_training_receipt_sha256": raw_receipt_sha256,
            "current_evidence_attestation_sha256": attestation_sha256,
            "environment_receipt_sha256": environment_sha256,
        },
        "independent verifier receipt-chain bindings disagree",
    )
    selection = verifier.get("selection")
    _require(isinstance(selection, dict), "verifier objective selection is missing")
    selection = cast(dict[str, Any], selection)
    candidates = selection.get("candidate_objectives")
    _require(
        isinstance(candidates, list)
        and len(candidates) == len(_LEARNED_OBJECTIVES)
        and set(candidates) == set(_LEARNED_OBJECTIVES),
        "verifier candidate objective family is invalid",
    )
    _require(selection.get("selection_status") == "familywise-corrected", "verifier selection is uncorrected")
    _require(selection.get("family_size") == _VERIFIER_FAMILY_SIZE, "verifier family size mismatch")
    _require(selection.get("correction") == _VERIFIER_CORRECTION, "verifier correction method mismatch")
    best_objective = selection.get("raw_winner")
    _require(best_objective in _LEARNED_OBJECTIVES, "verifier raw winner is invalid")
    _require(bool(verifier.get("paired_comparisons")), "verifier paired comparisons are missing")
    _require(bool(verifier.get("gates")), "verifier gates are missing")
    seed_results = receipt.get("seed_results")
    _require(isinstance(seed_results, dict), "workbench seed results are missing")
    seed_results = cast(dict[str, Any], seed_results)
    complete_seeds: list[int] = []
    for key, row in seed_results.items():
        if not isinstance(row, dict):
            continue
        objective_row = row.get(best_objective)
        training = objective_row.get("training") if isinstance(objective_row, dict) else None
        if isinstance(training, dict) and training.get("complete") is True:
            try:
                complete_seeds.append(int(key))
            except ValueError:
                continue
    _require(bool(complete_seeds), "verified best objective has no complete seed")
    return str(best_objective), min(complete_seeds)


def _chain_link(composite: Mapping[str, Any], field: str, filename: str, expected_sha256: str) -> None:
    link = composite.get(field)
    _require(isinstance(link, dict), f"composite receipt link missing: {field}")
    link = cast(dict[str, Any], link)
    path = link.get("path")
    _require(isinstance(path, str) and Path(path).name == filename, f"composite {field} path is invalid")
    _require(link.get("sha256") == expected_sha256, f"composite {field} hash mismatch")


def _require_composite(
    composite: Mapping[str, Any],
    *,
    raw_receipt: Mapping[str, Any],
    raw_receipt_sha256: str,
    attestation_sha256: str,
    environment_sha256: str,
    verifier_sha256: str,
    attestation: Mapping[str, Any],
    environment: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> None:
    _require(composite.get("schema") == WORKBENCH_SCHEMA, "final composite receipt schema mismatch")
    for key, value in raw_receipt.items():
        _require(composite.get(key) == value, f"final composite changed raw training field: {key}")
    _chain_link(composite, "raw_training_receipt", "raw_workbench_receipt.json", raw_receipt_sha256)
    _chain_link(
        composite,
        "current_evidence_attestation",
        "current_evidence_attestation.json",
        attestation_sha256,
    )
    _chain_link(composite, "environment_receipt", "environment_receipt.json", environment_sha256)
    _chain_link(composite, "independent_verifier", "independent_verifier.json", verifier_sha256)
    authoritative = composite.get("authoritative_promotion")
    _require(isinstance(authoritative, dict), "final authoritative promotion block is missing")
    authoritative = cast(dict[str, Any], authoritative)
    gates = authoritative.get("gates")
    _require(isinstance(gates, dict), "final authoritative promotion gates are missing")
    gates = cast(dict[str, Any], gates)
    expected_gates = {
        "raw_training_complete": raw_receipt.get("complete") is True,
        "evidence_current": attestation.get("scientifically_current") is True,
        "environment_all_ok": environment.get("all_ok") is True,
        "independent_verifier_promotes": verifier.get("promotion") is True,
    }
    _require(gates == expected_gates, "final authoritative promotion gates disagree")
    expected_promotion = all(expected_gates.values())
    _require(
        authoritative.get("cm7_local_objective_lever_promotable") is expected_promotion,
        "final CM7 authoritative promotion is inconsistent",
    )
    _require(expected_promotion, "final composite does not authorize CM7 artifact export")
    _require(authoritative.get("cm8_custom_build_promotable") is False, "CM8 must remain unpromoted")
    _require(
        authoritative.get("verdict") == verifier.get("verdict") == "promote-local-objective-lever",
        "final authoritative verdict disagrees with verifier",
    )
    _require(authoritative.get("raw_promotion_is_preliminary") is True, "raw promotion was treated as final")
    _require(not authoritative.get("reasons"), "final authoritative promotion records refusal reasons")
    scope = authoritative.get("scope_boundary")
    _require(isinstance(scope, str) and bool(scope.strip()), "final authoritative scope boundary is missing")


def _validate_receipt_chain(
    paths: Mapping[str, Path],
    *,
    label_prefix: str = "",
) -> tuple[tuple[dict[str, Any], ...], dict[str, str], str, int]:
    receipt = _read_json(paths["raw_training_receipt"], f"{label_prefix}raw training receipt")
    composite = _read_json(paths["final_composite_receipt"], f"{label_prefix}final composite receipt")
    attestation = _read_json(
        paths["current_evidence_attestation"], f"{label_prefix}current-evidence attestation"
    )
    environment = _read_json(paths["environment_receipt"], f"{label_prefix}environment receipt")
    verifier = _read_json(paths["independent_verifier"], f"{label_prefix}independent verifier")
    current_audit_path = paths["current_requirements_audit"]
    _require(current_audit_path.is_file(), f"{label_prefix}current requirements audit is missing")

    raw_sha = sha256_file(paths["raw_training_receipt"])
    composite_sha = sha256_file(paths["final_composite_receipt"])
    attestation_sha = sha256_file(paths["current_evidence_attestation"])
    environment_sha = sha256_file(paths["environment_receipt"])
    verifier_sha = sha256_file(paths["independent_verifier"])
    _require_raw_receipt(receipt)
    _require_attestation(
        attestation,
        raw_receipt_sha256=raw_sha,
        current_audit_sha256=sha256_file(current_audit_path),
    )
    receipt_implementation = receipt.get("implementation")
    _require(isinstance(receipt_implementation, dict), "raw implementation binding is missing")
    receipt_implementation = cast(dict[str, Any], receipt_implementation)
    _require_environment(
        environment,
        raw_receipt_sha256=raw_sha,
        implementation_manifest_sha256=sha256_file(paths["implementation_manifest"]),
        implementation_aggregate_sha256=str(receipt_implementation.get("aggregate_sha256")),
    )
    objective, seed = _require_verifier(
        verifier,
        raw_receipt_sha256=raw_sha,
        attestation_sha256=attestation_sha,
        environment_sha256=environment_sha,
        receipt=receipt,
    )
    _require_composite(
        composite,
        raw_receipt=receipt,
        raw_receipt_sha256=raw_sha,
        attestation_sha256=attestation_sha,
        environment_sha256=environment_sha,
        verifier_sha256=verifier_sha,
        attestation=attestation,
        environment=environment,
        verifier=verifier,
    )
    return (
        (receipt, composite, attestation, environment, verifier),
        {
            "raw": raw_sha,
            "composite": composite_sha,
            "attestation": attestation_sha,
            "environment": environment_sha,
            "verifier": verifier_sha,
        },
        objective,
        seed,
    )


def _snapshot_sources(
    run_dir: Path,
    rows: Sequence[Any],
    group: str,
) -> tuple[tuple[_SourceFile, ...], set[str]]:
    implementation = group == "implementation_sources"
    label = "implementation snapshot" if implementation else "requirements snapshot"
    source_hash_key = "source_sha256" if implementation else "sha256"
    role_prefix = "implementation_snapshot_" if implementation else "requirements_snapshot_"
    role_width = 2 if implementation else 3
    artifact_dir = "provenance/implementation" if implementation else "provenance/requirements"
    sources: list[_SourceFile] = []
    hashes: set[str] = set()
    for index, value in enumerate(rows):
        _require(isinstance(value, dict), f"{label} row is invalid")
        row = cast(dict[str, Any], value)
        expected = row.get("snapshot_sha256")
        _require(_is_sha256(expected), f"{label} hash is malformed")
        _require(expected == row.get(source_hash_key), f"{label} source/snapshot hash mismatch")
        recorded_path = row.get("snapshot_path")
        _require(isinstance(recorded_path, str) and bool(recorded_path), f"{group} snapshot path missing")
        recorded_path = cast(str, recorded_path)
        name = Path(recorded_path).name
        _require(name not in ("", ".", ".."), f"{group} snapshot path is unsafe")
        snapshot = run_dir / group / name
        _require(sha256_file(snapshot) == expected, f"{label} hash drift")
        _require(snapshot.stat().st_size == row.get("bytes"), f"{label} size drift")
        hashes.add(str(expected))
        sources.append(
            _SourceFile(
                f"{role_prefix}{index:0{role_width}d}",
                snapshot,
                f"{artifact_dir}/{snapshot.name}",
                str(expected),
            )
        )
    return tuple(sources), hashes


def _requirements_aggregate_payload(requirements: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ledger_sha256": requirements.get("ledger_sha256"),
        "requirements": [
            {
                "id": row.get("id"),
                "sources": [
                    {"path": source.get("path"), "sha256": source.get("sha256")}
                    for source in row.get("sources", [])
                    if isinstance(source, dict)
                ],
            }
            for row in requirements.get("requirements", [])
            if isinstance(row, dict)
        ],
    }


def _validate_provenance(
    run_dir: Path,
    receipt: Mapping[str, Any],
    chain_paths: Mapping[str, Path],
    arm_path: Path,
    attestation: Mapping[str, Any],
) -> tuple[_SourceFile, ...]:
    config_path = run_dir / "resolved_config.json"
    dataset_path = run_dir / "dataset_manifest.json"
    requirements_path = run_dir / "requirements_audit.json"
    implementation_path = run_dir / "implementation_manifest.json"
    teacher_path = run_dir / "teacher_audit.json"
    config = _read_json(config_path, "resolved config")
    dataset = _read_json(dataset_path, "dataset manifest")
    requirements = _read_json(requirements_path, "requirements audit")
    implementation = _read_json(implementation_path, "implementation manifest")
    teacher = _read_json(teacher_path, "teacher audit")

    _require(json_sha256(config) == receipt.get("config_sha256"), "resolved config hash drift")
    _require(dataset.get("schema") == DATASET_SCHEMA, "dataset manifest schema mismatch")
    dataset_without_hash = dict(dataset)
    recorded_dataset_hash = dataset_without_hash.pop("content_sha256", None)
    _require(json_sha256(dataset_without_hash) == recorded_dataset_hash, "dataset content hash drift")
    _require(recorded_dataset_hash == receipt.get("data_sha256"), "receipt dataset hash mismatch")
    _require(dataset.get("disjoint_referents") is True, "dataset referents are not disjoint")
    _require(dataset.get("combination_disjoint") is True, "dataset combinations are not disjoint")
    _require("not natural-video evidence" in str(dataset.get("claim_scope")), "dataset scope is unsafe")
    dataset_record = receipt.get("dataset")
    _require(isinstance(dataset_record, dict), "receipt dataset binding is missing")
    dataset_record = cast(dict[str, Any], dataset_record)
    dataset_spec = dataset.get("spec")
    records = dataset.get("records")
    splits = dataset.get("splits")
    _require(isinstance(dataset_spec, dict), "dataset spec is missing")
    _require(isinstance(records, list), "dataset records are missing")
    _require(isinstance(splits, dict), "dataset splits are missing")
    dataset_spec = cast(dict[str, Any], dataset_spec)
    records = cast(list[Any], records)
    splits = cast(dict[str, Any], splits)
    _require(len(records) == dataset_record.get("rows"), "receipt dataset row count mismatch")
    _require(dataset_spec.get("frames") == dataset_record.get("frames"), "receipt frame count mismatch")
    _require(
        dataset_spec.get("resolution") == dataset_record.get("resolution"),
        "receipt resolution mismatch",
    )
    _require(dataset_record.get("disjoint_referents") is True, "receipt referent-disjoint flag failed")
    _require(dataset_record.get("combination_disjoint") is True, "receipt combination-disjoint flag failed")
    split_counts = dataset_record.get("split_counts")
    _require(isinstance(split_counts, dict), "receipt dataset split counts are missing")
    split_counts = cast(dict[str, Any], split_counts)
    for split in ("train", "val", "test"):
        indices = splits.get(split)
        _require(isinstance(indices, list), f"dataset {split} split is invalid")
        indices = cast(list[Any], indices)
        _require(len(indices) == split_counts.get(split), f"receipt {split} split count mismatch")

    _require(requirements.get("schema") == REQUIREMENTS_SCHEMA, "requirements schema mismatch")
    _require(requirements.get("all_ok") is True and not requirements.get("problems"), "requirements failed")
    aggregate = _requirements_aggregate_payload(requirements)
    _require(json_sha256(aggregate) == requirements.get("aggregate_sha256"), "requirements hash drift")
    _require(
        requirements.get("aggregate_sha256") == receipt.get("requirements_sha256"),
        "receipt requirements hash mismatch",
    )

    _require(implementation.get("schema") == IMPLEMENTATION_SCHEMA, "implementation schema mismatch")
    _require(implementation.get("all_ok") is True, "implementation snapshot did not pass")
    implementation_rows = implementation.get("files")
    _require(
        isinstance(implementation_rows, list) and bool(implementation_rows),
        "implementation files missing",
    )
    implementation_rows = cast(list[Any], implementation_rows)
    implementation_aggregate = [
        {"path": row.get("source_path"), "sha256": row.get("snapshot_sha256")}
        for row in implementation_rows
        if isinstance(row, dict)
    ]
    _require(
        json_sha256(implementation_aggregate) == implementation.get("aggregate_sha256"),
        "implementation aggregate hash drift",
    )
    receipt_implementation = receipt.get("implementation")
    _require(isinstance(receipt_implementation, dict), "receipt implementation binding is missing")
    receipt_implementation = cast(dict[str, Any], receipt_implementation)
    _require(receipt_implementation.get("all_ok") is True, "receipt implementation binding failed")
    _require(
        receipt_implementation.get("aggregate_sha256") == implementation.get("aggregate_sha256"),
        "receipt implementation hash mismatch",
    )

    sources = [
        _SourceFile(role, chain_paths[role], f"evidence/{_CHAIN_FILENAMES[role]}")
        for role in _CHAIN_ROLES[:5]
    ] + [
        _SourceFile("selected_arm_receipt", arm_path, "evidence/selected_arm_receipt.json"),
        _SourceFile("resolved_config", config_path, "provenance/resolved_config.json"),
        _SourceFile("dataset_manifest", dataset_path, "provenance/dataset_manifest.json"),
        _SourceFile("requirements_audit", requirements_path, "provenance/requirements_audit.json"),
        _SourceFile(
            "implementation_manifest", implementation_path, "provenance/implementation_manifest.json"
        ),
        _SourceFile("teacher_audit", teacher_path, "provenance/teacher_audit.json"),
    ]

    current_name = attestation.get("current_audit_path")
    _require(
        isinstance(current_name, str) and Path(current_name).name == current_name,
        "current requirements audit path is unsafe",
    )
    current_name = cast(str, current_name)
    current_path = run_dir / current_name
    _require(
        sha256_file(current_path) == attestation.get("current_audit_sha256"),
        "attested current requirements audit hash drift",
    )
    current = _read_json(current_path, "current requirements audit")
    _require(current.get("schema") == REQUIREMENTS_SCHEMA, "current requirements schema mismatch")
    _require(current.get("all_ok") is True and not current.get("problems"), "current requirements failed")
    _require(
        json_sha256(_requirements_aggregate_payload(current)) == current.get("aggregate_sha256"),
        "current requirements aggregate hash drift",
    )
    _require(
        current.get("ledger_sha256") == requirements.get("ledger_sha256"),
        "current requirements ledger hash drift",
    )
    sources.append(
        _SourceFile("current_requirements_audit", current_path, "provenance/requirements_current_audit.json")
    )

    implementation_sources, implementation_hashes = _snapshot_sources(
        run_dir, implementation_rows, "implementation_sources"
    )
    sources.extend(implementation_sources)
    generator = dataset.get("generator")
    _require(isinstance(generator, dict), "dataset generator provenance is missing")
    generator = cast(dict[str, Any], generator)
    _require(
        generator.get("source_sha256") in implementation_hashes,
        "dataset generator is not present in the frozen implementation snapshots",
    )

    requirement_rows: list[Any] = []
    for requirement in requirements.get("requirements", []):
        _require(isinstance(requirement, dict), "requirements row is invalid")
        requirement = cast(dict[str, Any], requirement)
        requirement_rows.extend(requirement.get("sources", []))
    requirement_sources, _hashes = _snapshot_sources(run_dir, requirement_rows, "requirements_sources")
    _require(bool(requirement_sources), "requirements evidence snapshots are missing")
    sources.extend(requirement_sources)
    _require(
        teacher.get("schema") == "mop-custom-substrate-teacher-audit/v1",
        "teacher audit schema mismatch",
    )
    _require(teacher.get("all_ok") is True, "teacher audit failed")

    runtime_sources = (
        ("portable_runtime_source", Path(__file__).resolve(), "runtime/custom_artifact.py"),
        ("portable_model_source", Path(__file__).with_name("custom_model.py"), "runtime/custom_model.py"),
    )
    sources.extend(
        _SourceFile(role, source, artifact_path, sha256_file(source))
        for role, source, artifact_path in runtime_sources
    )
    paths = [source.artifact_path for source in sources]
    _require(len(paths) == len(set(paths)), "portable provenance paths collide")
    return tuple(sorted(sources, key=lambda source: source.artifact_path))


def _prepare_export(run_dir: Path, verifier_path: Path) -> _PreparedExport:
    run_dir = run_dir.resolve()
    verifier_path = verifier_path.resolve()
    chain_paths = {role: run_dir / name for role, name in _CHAIN_FILENAMES.items()}
    _require(
        verifier_path == chain_paths["independent_verifier"].resolve(),
        "verifier path is not the run's immutable linked verifier",
    )
    documents, _hashes, objective, seed = _validate_receipt_chain(
        chain_paths,
        label_prefix="immutable ",
    )
    receipt, composite, attestation, environment, verifier = documents
    seed_results = cast(Mapping[str, Any], receipt["seed_results"])
    seed_row = seed_results.get(str(seed))
    _require(isinstance(seed_row, dict), "selected seed is absent from workbench receipt")
    seed_row = cast(dict[str, Any], seed_row)
    objective_row = seed_row.get(objective)
    _require(isinstance(objective_row, dict), "selected objective is absent from workbench receipt")
    objective_row = cast(dict[str, Any], objective_row)
    embedded_arm = objective_row.get("training")
    _require(
        isinstance(embedded_arm, dict) and embedded_arm.get("complete") is True,
        "selected arm incomplete",
    )
    embedded_arm = cast(dict[str, Any], embedded_arm)
    arm_path = run_dir / "arms" / f"seed_{seed}" / objective / "arm_receipt.json"
    arm = _read_json(arm_path, "selected arm receipt")
    _require(arm.get("schema") == ARM_SCHEMA, "selected arm receipt schema mismatch")
    for key in (
        "objective",
        "seed",
        "complete",
        "requested_steps",
        "completed_steps",
        "config_sha256",
        "data_sha256",
        "requirements_sha256",
        "initial_state_sha256",
        "final_state_sha256",
    ):
        _require(arm.get(key) == embedded_arm.get(key), f"selected arm disagrees with receipt: {key}")
    _require(arm.get("objective") == objective and arm.get("seed") == seed, "selected arm identity mismatch")
    _require(arm.get("complete") is True, "selected arm receipt is incomplete")
    _require(arm.get("completed_steps") == arm.get("requested_steps"), "selected arm stopped early")
    checkpoint_record = arm.get("checkpoint")
    embedded_checkpoint = embedded_arm.get("checkpoint")
    _require(isinstance(checkpoint_record, dict), "selected arm checkpoint record is missing")
    _require(isinstance(embedded_checkpoint, dict), "embedded checkpoint record is missing")
    checkpoint_record = cast(dict[str, Any], checkpoint_record)
    embedded_checkpoint = cast(dict[str, Any], embedded_checkpoint)
    checkpoint_sha = checkpoint_record.get("sha256")
    _require(_is_sha256(checkpoint_sha), "selected checkpoint hash is malformed")
    _require(checkpoint_sha == embedded_checkpoint.get("sha256"), "selected checkpoint bindings disagree")
    checkpoint_path = run_dir / "arms" / f"seed_{seed}" / objective / "checkpoint.pt"
    _require(checkpoint_path.is_file(), "selected checkpoint file is missing")
    _require(
        checkpoint_record.get("bytes") == checkpoint_path.stat().st_size,
        "selected checkpoint size drift",
    )
    _require(sha256_file(checkpoint_path) == checkpoint_sha, "selected checkpoint file hash drift")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactRefused(f"selected checkpoint cannot be safely loaded: {exc}") from exc
    _require(isinstance(checkpoint, dict), "selected checkpoint payload is invalid")
    checkpoint = cast(dict[str, Any], checkpoint)
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "selected checkpoint schema mismatch")
    expected_checkpoint = {
        "objective": objective,
        "step": arm.get("completed_steps"),
        "config_sha256": arm.get("config_sha256"),
        "data_sha256": arm.get("data_sha256"),
        "requirements_sha256": arm.get("requirements_sha256"),
        "initial_state_sha256": arm.get("initial_state_sha256"),
    }
    for key, value in expected_checkpoint.items():
        _require(checkpoint.get(key) == value, f"selected checkpoint identity mismatch: {key}")
    state = checkpoint.get("model")
    _require(
        isinstance(state, dict)
        and bool(state)
        and all(isinstance(value, torch.Tensor) for value in state.values()),
        "selected checkpoint online state is invalid",
    )
    state = cast(dict[str, torch.Tensor], state)
    actual_state_hash = state_sha256(state)
    _require(actual_state_hash == arm.get("final_state_sha256"), "selected online state hash drift")
    for key in ("config_sha256", "data_sha256", "requirements_sha256"):
        _require(arm.get(key) == receipt.get(key), f"selected arm differs from raw training identity: {key}")

    model_record = receipt.get("model")
    _require(isinstance(model_record, dict), "workbench model record is missing")
    model_record = cast(dict[str, Any], model_record)
    _require(model_record.get("architecture") == "TinyVideoSubstrate", "workbench architecture mismatch")
    _require(model_record.get("teacher_independent") is True, "selected model is not teacher independent")
    _require(
        model_record.get("exports") == ["dense_spatiotemporal_tokens", "pooled_retrieval_key"],
        "workbench export interface mismatch",
    )
    spec_value = model_record.get("spec")
    _require(isinstance(spec_value, dict), "workbench model spec is missing")
    spec_value = cast(dict[str, Any], spec_value)
    spec = _model_spec(spec_value)
    config = _read_json(run_dir / "resolved_config.json", "resolved config")
    _require(config.get("model") == asdict(spec), "resolved config and receipt model specs disagree")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PortableTinyVideoSubstrate(spec)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactRefused(f"selected state does not match portable model spec: {exc}") from exc
    parameters = parameter_count(model)
    _require(parameters == model_record.get("trainable_parameters"), "model parameter count mismatch")
    _require(1_000_000 <= parameters <= 5_000_000, "model is outside the CM7 parameter envelope")
    dataset_record = receipt.get("dataset")
    _require(isinstance(dataset_record, dict), "workbench dataset record is missing")
    dataset_record = cast(dict[str, Any], dataset_record)
    frames, resolution = dataset_record.get("frames"), dataset_record.get("resolution")
    _require(isinstance(frames, int) and isinstance(resolution, int), "dataset geometry is invalid")
    frames = cast(int, frames)
    resolution = cast(int, resolution)
    expected_tokens = (frames // spec.tubelet) * (resolution // spec.patch_size) ** 2
    _require(expected_tokens == model_record.get("token_count"), "model token count mismatch")

    sources = _validate_provenance(
        run_dir,
        receipt,
        chain_paths,
        arm_path,
        attestation,
    )
    return _PreparedExport(
        receipt=receipt,
        composite=composite,
        verifier=verifier,
        checkpoint_sha256=str(checkpoint_sha),
        state={name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
        spec=spec,
        seed=seed,
        objective=objective,
        sources=sources,
    )


def preflight_export(run_dir: Path, verifier_path: Path) -> dict[str, Any]:

    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "export_performed": False,
        "run_dir": str(run_dir),
        "verifier_path": str(verifier_path),
        "required_verifier_schema": VERIFIER_SCHEMA,
        "selection_rule": _SELECTION_RULE,
        "eligible": False,
        "problems": [],
    }
    try:
        prepared = _prepare_export(run_dir, verifier_path)
    except ArtifactRefused as exc:
        result["problems"] = [str(exc)]
        return result
    result.update(
        {
            "eligible": True,
            "selection": {
                "seed": prepared.seed,
                "objective": prepared.objective,
                "checkpoint_sha256": prepared.checkpoint_sha256,
                "state_sha256": state_sha256(prepared.state),
            },
            "model_spec": asdict(prepared.spec),
            "evidence_scope": EVIDENCE_SCOPE,
            "provenance_file_count": len(prepared.sources),
        }
    )
    return result


def _copy_sources(sources: Sequence[_SourceFile], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        _require(source.source.is_file(), f"provenance source disappeared: {source.source}")
        before = sha256_file(source.source)
        if source.expected_sha256 is not None:
            _require(before == source.expected_sha256, f"provenance source hash drift: {source.role}")
        target = root / source.artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.source, target)
        after = sha256_file(source.source)
        copied = sha256_file(target)
        _require(before == after == copied, f"provenance source changed during copy: {source.role}")
        records.append(
            {
                "role": source.role,
                "path": source.artifact_path,
                "bytes": target.stat().st_size,
                "sha256": copied,
            }
        )
    return records


def export_artifact(run_dir: Path, verifier_path: Path, output_root: Path) -> dict[str, Any]:

    prepared = _prepare_export(run_dir, verifier_path)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch = output_root / f".custom-substrate-export-{os.getpid()}"
    _require(not scratch.exists(), f"temporary export path already exists: {scratch}")
    scratch.mkdir()
    try:
        weights = write_tensor_pack(prepared.state, scratch / "weights.mopbin")
        provenance = _copy_sources(prepared.sources, scratch)
        records = {row["role"]: row for row in provenance}
        manifest: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "model": {
                "architecture": "PortableTinyVideoSubstrate",
                "source_architecture": "TinyVideoSubstrate",
                "spec": asdict(prepared.spec),
                "trainable_parameters": prepared.receipt["model"]["trainable_parameters"],
                "token_count_at_training_geometry": prepared.receipt["model"]["token_count"],
                "state_sha256": state_sha256(prepared.state),
                "teacher_independent": True,
            },
            "interface": INTERFACE_SCHEMA,
            "selection": {
                "rule": _SELECTION_RULE,
                "seed": prepared.seed,
                "objective": prepared.objective,
                "state_component": "model",
                "checkpoint_sha256": prepared.checkpoint_sha256,
                "selected_arm_receipt_sha256": records["selected_arm_receipt"]["sha256"],
                "state_sha256": state_sha256(prepared.state),
            },
            "evidence": {
                "raw_training_receipt_sha256": records["raw_training_receipt"]["sha256"],
                "current_evidence_attestation_sha256": records["current_evidence_attestation"]["sha256"],
                "environment_receipt_sha256": records["environment_receipt"]["sha256"],
                "independent_verifier_sha256": records["independent_verifier"]["sha256"],
                "final_composite_receipt_sha256": records["final_composite_receipt"]["sha256"],
                "independent_verifier_schema": VERIFIER_SCHEMA,
                "independent_verifier_verdict": prepared.verifier["verdict"],
                "authoritative_promotion": prepared.composite["authoritative_promotion"],
                "scope": EVIDENCE_SCOPE,
            },
            "source_provenance": {
                "config_sha256": prepared.receipt["config_sha256"],
                "data_sha256": prepared.receipt["data_sha256"],
                "requirements_sha256": prepared.receipt["requirements_sha256"],
                "implementation_sha256": prepared.receipt["implementation"]["aggregate_sha256"],
                "files": provenance,
            },
            "weights": weights,
        }
        manifest["artifact_id"] = json_sha256(manifest)
        _atomic_json(scratch / "manifest.json", manifest)
        target = output_root / f"tiny-video-substrate-{manifest['artifact_id']}"
        if target.exists():
            existing = load_portable_artifact(target)
            _require(existing.manifest == manifest, "existing content-addressed artifact differs")
            shutil.rmtree(scratch)
            return {
                "artifact_id": manifest["artifact_id"],
                "artifact_dir": str(target),
                "manifest_sha256": sha256_file(target / "manifest.json"),
                "reused": True,
            }
        os.replace(scratch, target)
        loaded = load_portable_artifact(target)
        _require(loaded.manifest == manifest, "new artifact failed post-export verification")
        return {
            "artifact_id": manifest["artifact_id"],
            "artifact_dir": str(target),
            "manifest_sha256": sha256_file(target / "manifest.json"),
            "reused": False,
        }
    except Exception:
        if scratch.exists():
            shutil.rmtree(scratch)
        raise


def _safe_artifact_path(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is missing")
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path is unsafe")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"{label} escapes artifact directory")
    return path


def _provenance_by_role(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source = manifest.get("source_provenance")
    _require(isinstance(source, dict), "artifact source provenance is missing")
    source = cast(dict[str, Any], source)
    rows = source.get("files")
    _require(isinstance(rows, list) and bool(rows), "artifact provenance file table is empty")
    rows = cast(list[Any], rows)
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "artifact provenance row is invalid")
        row = cast(dict[str, Any], row)
        role = row.get("role")
        _require(isinstance(role, str) and role not in by_role, "artifact provenance role is invalid")
        role = cast(str, role)
        by_role[role] = row
    return by_role


def _verify_embedded_evidence(root: Path, manifest: Mapping[str, Any]) -> None:
    by_role = _provenance_by_role(manifest)
    evidence = manifest.get("evidence")
    selection = manifest.get("selection")
    _require(isinstance(evidence, dict) and isinstance(selection, dict), "artifact evidence is missing")
    evidence = cast(dict[str, Any], evidence)
    selection = cast(dict[str, Any], selection)
    _require(all(role in by_role for role in _CHAIN_ROLES), "portable artifact receipt chain is incomplete")

    def evidence_path(role: str) -> Path:
        return _safe_artifact_path(root, by_role[role].get("path"), f"embedded {role}")

    paths = {role: evidence_path(role) for role in _CHAIN_ROLES}
    arm = _read_json(paths["selected_arm_receipt"], "embedded selected arm")
    documents, hashes, objective, seed = _validate_receipt_chain(
        paths,
        label_prefix="embedded ",
    )
    composite, verifier = documents[1], documents[4]
    expected_hashes = {
        "raw_training_receipt_sha256": hashes["raw"],
        "current_evidence_attestation_sha256": hashes["attestation"],
        "environment_receipt_sha256": hashes["environment"],
        "independent_verifier_sha256": hashes["verifier"],
        "final_composite_receipt_sha256": hashes["composite"],
    }
    for field, expected in expected_hashes.items():
        _require(evidence.get(field) == expected, f"embedded receipt-chain hash drift: {field}")
    _require(
        evidence.get("authoritative_promotion") == composite.get("authoritative_promotion"),
        "embedded authoritative promotion binding drift",
    )
    _require(evidence.get("independent_verifier_schema") == VERIFIER_SCHEMA, "verifier schema binding drift")
    _require(
        evidence.get("independent_verifier_verdict") == verifier.get("verdict"),
        "verifier verdict binding drift",
    )
    _require(evidence.get("scope") == EVIDENCE_SCOPE, "artifact evidence scope drift")
    _require(seed == selection.get("seed") and objective == selection.get("objective"), "selection drift")
    _require(arm.get("seed") == seed and arm.get("objective") == objective, "embedded arm identity drift")
    arm_checkpoint = arm.get("checkpoint")
    _require(isinstance(arm_checkpoint, dict), "embedded arm checkpoint record missing")
    arm_checkpoint = cast(dict[str, Any], arm_checkpoint)
    _require(arm_checkpoint.get("sha256") == selection.get("checkpoint_sha256"), "checkpoint drift")
    _require(arm.get("final_state_sha256") == selection.get("state_sha256"), "state binding drift")


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
    manifest = _read_json(root / "manifest.json", "portable artifact manifest")
    _require(manifest.get("schema") == ARTIFACT_SCHEMA, "portable artifact schema mismatch")
    recorded_id = manifest.get("artifact_id")
    _require(_is_sha256(recorded_id), "portable artifact id is malformed")
    identity = dict(manifest)
    identity.pop("artifact_id", None)
    _require(json_sha256(identity) == recorded_id, "portable artifact content id mismatch")
    _require(manifest.get("interface") == INTERFACE_SCHEMA, "portable interface schema mismatch")

    by_role = _provenance_by_role(manifest)
    source_record = cast(Mapping[str, Any], manifest["source_provenance"])
    listed_paths: set[str] = set()
    for role, row in by_role.items():
        path_value = row.get("path")
        _require(isinstance(path_value, str) and path_value not in listed_paths, "duplicate provenance path")
        path_value = cast(str, path_value)
        listed_paths.add(path_value)
        path = _safe_artifact_path(root, path_value, f"provenance {role}")
        _require(path.is_file(), f"artifact provenance file missing: {role}")
        _require(path.stat().st_size == row.get("bytes"), f"artifact provenance size drift: {role}")
        _require(sha256_file(path) == row.get("sha256"), f"artifact provenance hash drift: {role}")
    for required_role in (
        *_CHAIN_ROLES,
        "resolved_config",
        "dataset_manifest",
        "requirements_audit",
        "portable_runtime_source",
        "portable_model_source",
    ):
        _require(required_role in by_role, f"required artifact provenance role missing: {required_role}")

    config = _read_json(
        _safe_artifact_path(root, by_role["resolved_config"]["path"], "resolved config"),
        "embedded resolved config",
    )
    _require(json_sha256(config) == source_record.get("config_sha256"), "embedded config identity drift")
    dataset = _read_json(
        _safe_artifact_path(root, by_role["dataset_manifest"]["path"], "dataset manifest"),
        "embedded dataset manifest",
    )
    dataset_identity = dict(dataset)
    dataset_sha = dataset_identity.pop("content_sha256", None)
    _require(json_sha256(dataset_identity) == dataset_sha, "embedded dataset content hash drift")
    _require(dataset_sha == source_record.get("data_sha256"), "embedded dataset identity drift")
    requirements = _read_json(
        _safe_artifact_path(root, by_role["requirements_audit"]["path"], "requirements audit"),
        "embedded requirements audit",
    )
    _require(
        requirements.get("aggregate_sha256") == source_record.get("requirements_sha256"),
        "embedded requirements identity drift",
    )
    implementation = _read_json(
        _safe_artifact_path(root, by_role["implementation_manifest"]["path"], "implementation manifest"),
        "embedded implementation manifest",
    )
    _require(
        implementation.get("aggregate_sha256") == source_record.get("implementation_sha256"),
        "embedded implementation identity drift",
    )
    _verify_embedded_evidence(root, manifest)

    weights = manifest.get("weights")
    model_record = manifest.get("model")
    selection = manifest.get("selection")
    _require(
        isinstance(weights, dict) and isinstance(model_record, dict) and isinstance(selection, dict),
        "portable model or weight identity is missing",
    )
    weights = cast(dict[str, Any], weights)
    model_record = cast(dict[str, Any], model_record)
    selection = cast(dict[str, Any], selection)
    _require(weights.get("format") == TENSOR_PACK_SCHEMA, "portable weight format mismatch")
    weight_path = _safe_artifact_path(root, weights.get("path"), "portable weights")
    _require(weight_path.is_file(), "portable weights are missing")
    _require(weight_path.stat().st_size == weights.get("bytes"), "portable weight size drift")
    _require(sha256_file(weight_path) == weights.get("sha256"), "portable weight hash drift")
    state, header = read_tensor_pack(weight_path)
    state_hash = state_sha256(state)
    _require(header.get("state_sha256") == state_hash, "packed state header drift")
    _require(weights.get("state_sha256") == state_hash, "packed weight state binding drift")
    _require(model_record.get("state_sha256") == state_hash, "portable model state binding drift")
    _require(selection.get("state_sha256") == state_hash, "portable selection state binding drift")
    _require(model_record.get("architecture") == "PortableTinyVideoSubstrate", "architecture mismatch")
    _require(model_record.get("source_architecture") == "TinyVideoSubstrate", "source architecture drift")
    _require(model_record.get("teacher_independent") is True, "artifact is not teacher independent")
    spec_value = model_record.get("spec")
    _require(isinstance(spec_value, dict), "portable model spec is missing")
    spec_value = cast(dict[str, Any], spec_value)
    spec = _model_spec(spec_value)
    _require(config.get("model") == asdict(spec), "portable spec disagrees with frozen training config")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PortableTinyVideoSubstrate(spec)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactRefused(f"portable state does not match its model spec: {exc}") from exc
    _require(
        parameter_count(model) == model_record.get("trainable_parameters"),
        "portable parameter count drift",
    )
    model.requires_grad_(False)
    model.to(torch.device(device)).eval()
    return LoadedPortableArtifact(manifest=manifest, model=model)
