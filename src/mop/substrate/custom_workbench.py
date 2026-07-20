from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from mop.evidence import atomic_write_json, canonical_bytes, canonical_sha256, sha256_file

from ..config import REPO_ROOT
from ..devices import DeviceInfo
from .cache_manifest import validate_cache_manifest
from .latent_store import LatentStore

WORKBENCH_SCHEMA = "mop-custom-substrate-workbench/v1"
DATASET_SCHEMA = "mop-custom-substrate-dataset/v1"
CHECKPOINT_SCHEMA = "mop-custom-substrate-checkpoint/v1"
ARM_SCHEMA = "mop-custom-substrate-arm/v1"
REQUIREMENTS_SCHEMA = "mop-custom-substrate-requirements/v1"
TEACHER_AUDIT_SCHEMA = "mop-custom-substrate-teacher-audit/v1"
PREFLIGHT_SCHEMA = "mop-custom-substrate-cm8-preflight/v1"

TRAIN_OBJECTIVES = ("predictive", "invariance", "reconstruction", "random_target")
OPTIONAL_OBJECTIVE = "teacher_distill"


class WorkbenchRefused(RuntimeError):
    pass


json_sha256 = canonical_sha256
_atomic_json = atomic_write_json


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _resolve_repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_observation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"top_level": type(payload).__name__}
    observation: dict[str, Any] = {"schema": payload.get("schema")}
    if isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
        observation["metric_keys"] = sorted(metrics)
        observation["null_supported"] = metrics.get("null_supported")
    if isinstance(payload.get("null_decision"), dict):
        observation["declared_verdict"] = payload["null_decision"].get("declared_verdict")
        observation["ledger_ready"] = payload["null_decision"].get("ledger_ready")
    if isinstance(payload.get("coverage"), dict):
        observation["coverage"] = payload["coverage"]
    if "hardware_boundary_conclusion" in payload:
        observation["hardware_boundary_conclusion"] = payload["hardware_boundary_conclusion"]
    if "all_ok" in payload:
        observation["all_ok"] = payload.get("all_ok")
    return observation


def audit_requirements(
    ledger_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    path = _resolve_repo_path(ledger_path, repo_root)
    problems: list[str] = []
    if not path.is_file():
        return {
            "schema": REQUIREMENTS_SCHEMA,
            "ledger_path": str(path),
            "all_ok": False,
            "problems": ["requirements ledger missing"],
            "requirements": [],
        }
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != REQUIREMENTS_SCHEMA:
        problems.append(f"requirements schema must equal {REQUIREMENTS_SCHEMA!r}")
        raw = raw if isinstance(raw, dict) else {}
    requirements = raw.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        problems.append("requirements must be a non-empty list")
        requirements = []
    seen: set[str] = set()
    audited: list[dict[str, Any]] = []
    for row in requirements:
        if not isinstance(row, dict):
            problems.append("requirement row is not a mapping")
            continue
        rid = str(row.get("id", ""))
        if not rid or rid in seen:
            problems.append(f"missing or duplicate requirement id {rid!r}")
        seen.add(rid)
        sources = row.get("sources")
        if not isinstance(sources, list) or not sources:
            problems.append(f"{rid}: sources must be non-empty")
            sources = []
        source_audit: list[dict[str, Any]] = []
        for source in sources:
            rel = str(source.get("path", "")) if isinstance(source, dict) else ""
            source_path = _resolve_repo_path(rel, repo_root)
            record: dict[str, Any] = {
                "path": rel,
                "role": source.get("role") if isinstance(source, dict) else None,
                "exists": source_path.is_file(),
            }
            if not source_path.is_file():
                problems.append(f"{rid}: source missing: {rel}")
            else:
                record.update(
                    {
                        "bytes": source_path.stat().st_size,
                        "sha256": sha256_file(source_path),
                    }
                )
                if source_path.suffix == ".json":
                    try:
                        payload = json.loads(source_path.read_text())
                        record["json_ok"] = True
                        record["observation"] = _source_observation(payload)
                    except (OSError, json.JSONDecodeError) as exc:
                        record["json_ok"] = False
                        record["error"] = str(exc)
                        problems.append(f"{rid}: invalid JSON source {rel}: {exc}")
            source_audit.append(record)
        audited.append(
            {
                "id": rid,
                "title": row.get("title"),
                "status": row.get("status"),
                "design_response": row.get("design_response", []),
                "consumers": row.get("consumers", []),
                "promotion_gate": row.get("promotion_gate"),
                "sources": source_audit,
            }
        )
    aggregate = {
        "ledger_sha256": sha256_file(path),
        "requirements": [
            {
                "id": row["id"],
                "sources": [
                    {"path": source["path"], "sha256": source.get("sha256")} for source in row["sources"]
                ],
            }
            for row in audited
        ],
    }
    return {
        "schema": REQUIREMENTS_SCHEMA,
        "ledger_path": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
        "ledger_sha256": sha256_file(path),
        "aggregate_sha256": json_sha256(aggregate),
        "claim_scope": raw.get("claim_scope"),
        "requirements": audited,
        "requirement_ids": [row["id"] for row in audited],
        "problems": problems,
        "all_ok": not problems,
    }


def snapshot_requirement_sources(
    audit: dict[str, Any],
    *,
    destination: Path,
    repo_root: Path = REPO_ROOT,
) -> None:

    destination.mkdir(parents=True, exist_ok=True)
    for requirement in audit.get("requirements", []):
        rid = str(requirement["id"])
        for index, source in enumerate(requirement.get("sources", [])):
            if not source.get("exists"):
                continue
            original = _resolve_repo_path(source["path"], repo_root)
            target = destination / f"{rid}__{index:02d}__{original.name}"
            shutil.copyfile(original, target)
            copied_hash = sha256_file(target)
            if copied_hash != source.get("sha256"):
                target.unlink(missing_ok=True)
                raise WorkbenchRefused(f"requirements source changed during snapshot: {source['path']}")
            source["snapshot_path"] = str(target)
            source["snapshot_sha256"] = copied_hash


def snapshot_implementation_sources(
    run_dir: Path,
    *,
    expected_generator_sha256: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    candidates = (
        Path(__file__).resolve(),
        repo_root / "src/mop/experiments/custom_substrate.py",
        repo_root / "scripts/custom_substrate_workbench.py",
        repo_root / "configs/experiment/mop_cm7_min_objective_probe.yaml",
        repo_root / "configs/custom_substrate/requirements.yaml",
    )
    destination = run_dir / "implementation_sources"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in candidates:
        if not source.is_file():
            raise WorkbenchRefused(f"implementation source missing before training: {source}")
        relative = source.relative_to(repo_root) if source.is_relative_to(repo_root) else Path(source.name)
        target = destination / str(relative).replace("/", "__")
        before = sha256_file(source)
        shutil.copyfile(source, target)
        copied = sha256_file(target)
        after = sha256_file(source)
        if before != copied or before != after:
            target.unlink(missing_ok=True)
            raise WorkbenchRefused(f"implementation source changed during snapshot: {source}")
        records.append(
            {
                "source_path": str(relative),
                "source_sha256": before,
                "snapshot_path": str(target),
                "snapshot_sha256": copied,
                "bytes": target.stat().st_size,
            }
        )
    generator = next(
        record for record in records if record["source_path"] == "src/mop/substrate/custom_workbench.py"
    )
    if generator["snapshot_sha256"] != expected_generator_sha256:
        raise WorkbenchRefused("dataset generator hash does not match the frozen implementation source")
    manifest = {
        "schema": "mop-custom-substrate-implementation-snapshot/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "all_ok": True,
        "files": records,
        "aggregate_sha256": json_sha256(
            [{"path": record["source_path"], "sha256": record["snapshot_sha256"]} for record in records]
        ),
    }
    _atomic_json(run_dir / "implementation_manifest.json", manifest)
    return manifest


def attest_current_requirements(
    run_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    receipt_path = run_dir / "workbench_receipt.json"
    start_path = run_dir / "requirements_audit.json"
    config_path = run_dir / "resolved_config.json"
    if not all(path.is_file() for path in (receipt_path, start_path, config_path)):
        raise WorkbenchRefused("cannot attest an incomplete workbench run")
    receipt = json.loads(receipt_path.read_text())
    start = json.loads(start_path.read_text())
    config = json.loads(config_path.read_text())
    current = audit_requirements(config["requirements_ledger"], repo_root=repo_root)
    _atomic_json(run_dir / "requirements_current_audit.json", current)

    problems: list[str] = []
    implementation_path = run_dir / "implementation_manifest.json"
    implementation_checks: list[dict[str, Any]] = []
    if not implementation_path.is_file():
        problems.append("implementation snapshot manifest missing")
    else:
        implementation = json.loads(implementation_path.read_text())
        for row in implementation.get("files", []):
            snapshot = _resolve_repo_path(row["snapshot_path"], repo_root)
            actual = sha256_file(snapshot) if snapshot.is_file() else None
            ok = actual == row.get("snapshot_sha256") and actual is not None
            implementation_checks.append(
                {
                    "source_path": row.get("source_path"),
                    "snapshot_path": row.get("snapshot_path"),
                    "expected_sha256": row.get("snapshot_sha256"),
                    "actual_sha256": actual,
                    "ok": ok,
                }
            )
            if not ok:
                problems.append(f"frozen implementation snapshot invalid: {row.get('source_path')}")
    snapshot_checks: list[dict[str, Any]] = []
    start_by_path: dict[str, dict[str, Any]] = {}
    for requirement in start.get("requirements", []):
        for source in requirement.get("sources", []):
            start_by_path[str(source.get("path"))] = source
            snapshot_value = source.get("snapshot_path")
            requirement_snapshot = _resolve_repo_path(snapshot_value, repo_root) if snapshot_value else None
            actual = (
                sha256_file(requirement_snapshot)
                if requirement_snapshot is not None and requirement_snapshot.is_file()
                else None
            )
            expected = source.get("snapshot_sha256") or source.get("sha256")
            ok = actual == expected and actual is not None
            snapshot_checks.append(
                {
                    "source_path": source.get("path"),
                    "snapshot_path": snapshot_value,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "ok": ok,
                }
            )
            if not ok:
                problems.append(f"frozen requirements snapshot invalid: {source.get('path')}")

    current_by_path: dict[str, dict[str, Any]] = {}
    for requirement in current.get("requirements", []):
        for source in requirement.get("sources", []):
            current_by_path[str(source.get("path"))] = source
    source_drift: list[dict[str, Any]] = []
    for path, original in start_by_path.items():
        live = current_by_path.get(path)
        if live is None:
            problems.append(f"current requirements source disappeared: {path}")
            continue
        old_schema = (original.get("observation") or {}).get("schema")
        new_schema = (live.get("observation") or {}).get("schema")
        schema_compatible = old_schema == new_schema
        if not schema_compatible:
            problems.append(
                f"requirements source schema drifted for {path}: {old_schema!r} -> {new_schema!r}"
            )
        if original.get("sha256") != live.get("sha256"):
            source_drift.append(
                {
                    "path": path,
                    "start_sha256": original.get("sha256"),
                    "current_sha256": live.get("sha256"),
                    "schema_compatible": schema_compatible,
                }
            )

    semantic_same = start.get("ledger_sha256") == current.get("ledger_sha256") and start.get(
        "requirement_ids"
    ) == current.get("requirement_ids")
    if not semantic_same:
        problems.append("custom-substrate requirements ledger or requirement ids changed")
    if not current.get("all_ok"):
        problems.append("current requirements evidence audit is not clean")
    all_ok = not problems
    original_receipt_sha = sha256_file(receipt_path)
    attestation = {
        "schema": "mop-custom-substrate-current-evidence-attestation/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "original_receipt_sha256": original_receipt_sha,
        "training_design_snapshot_self_verifies": all(check["ok"] for check in snapshot_checks),
        "implementation_snapshot_self_verifies": bool(implementation_checks)
        and all(check["ok"] for check in implementation_checks),
        "requirements_semantics_unchanged": semantic_same,
        "current_evidence_all_ok": current.get("all_ok"),
        "scientifically_current": all_ok,
        "source_drift": source_drift,
        "snapshot_checks": snapshot_checks,
        "implementation_checks": implementation_checks,
        "current_audit_path": "requirements_current_audit.json",
        "problems": problems,
    }
    if not all_ok:
        receipt["promotion"]["cm7_local_objective_lever_promotable"] = False
        receipt["promotion"]["cm8_custom_build_promotable"] = False
        receipt["promotion"].setdefault("cm7_reasons", []).extend(problems)
        receipt["promotion"].setdefault("cm8_reasons", []).extend(problems)
        receipt["promotion"]["cm7_reasons"] = sorted(set(receipt["promotion"]["cm7_reasons"]))
        receipt["promotion"]["cm8_reasons"] = sorted(set(receipt["promotion"]["cm8_reasons"]))
    receipt["evidence_attestation"] = attestation
    _atomic_json(receipt_path, receipt)
    return receipt


@dataclass(frozen=True)
class CorpusSpec:
    resolution: int = 256
    frames: int = 8
    factor_a_levels: int = 4
    factor_b_levels: int = 4
    replicates: int = 6
    seed: int = 4407

    def validate(self) -> None:
        if self.resolution < 32 or self.resolution % 16:
            raise ValueError("resolution must be a multiple of 16 and at least 32")
        if self.frames < 2:
            raise ValueError("frames must be at least two")
        if min(self.factor_a_levels, self.factor_b_levels) < 3:
            raise ValueError("each factor needs at least three levels for held-out combinations")
        if self.replicates < 3:
            raise ValueError("replicates must be at least three")


@dataclass(frozen=True)
class ReferentRecord:
    index: int
    referent: str
    factor_a: int
    factor_b: int
    replicate: int
    split: str


def build_referent_records(spec: CorpusSpec) -> list[ReferentRecord]:

    spec.validate()
    rows: list[ReferentRecord] = []
    for rep in range(spec.replicates):
        for a in range(spec.factor_a_levels):
            for b in range(spec.factor_b_levels):
                group = (a + b) % 4
                split = "test" if group == 0 else "val" if group == 1 else "train"
                rows.append(
                    ReferentRecord(
                        index=len(rows),
                        referent=f"factorized:a{a}:b{b}:rep{rep}",
                        factor_a=a,
                        factor_b=b,
                        replicate=rep,
                        split=split,
                    )
                )
    for split in ("train", "val", "test"):
        if not any(row.split == split for row in rows):
            raise ValueError(f"corpus split {split!r} is empty")
    train_a = {row.factor_a for row in rows if row.split == "train"}
    train_b = {row.factor_b for row in rows if row.split == "train"}
    if len(train_a) != spec.factor_a_levels or len(train_b) != spec.factor_b_levels:
        raise ValueError("training combinations do not expose every independent factor level")
    return rows


def dataset_manifest(spec: CorpusSpec, records: Sequence[ReferentRecord]) -> dict[str, Any]:
    source = Path(__file__)
    rows = [asdict(row) for row in records]
    splits = {
        split: [row.index for row in records if row.split == split] for split in ("train", "val", "test")
    }
    split_refs = {
        split: [row.referent for row in records if row.split == split] for split in ("train", "val", "test")
    }
    split_sets = [set(values) for values in split_refs.values()]
    disjoint = all(not split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3))
    payload: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "claim_scope": "deterministic programmatic video; not natural-video evidence",
        "generator": {
            "module": "mop.substrate.custom_workbench.ProgrammaticVideoCorpus",
            "source_sha256": sha256_file(source),
        },
        "spec": asdict(spec),
        "referent_scheme": "factorized:a{a}:b{b}:rep{rep}",
        "factor_semantics": {
            "factor_a": "hue",
            "factor_b": "orientation-and-drift",
            "replicate": "phase, texture, and brightness nuisance",
        },
        "records": rows,
        "splits": splits,
        "split_referent_hashes": {key: json_sha256(value) for key, value in split_refs.items()},
        "disjoint_referents": disjoint,
        "combination_disjoint": True,
    }
    payload["content_sha256"] = json_sha256(payload)
    return payload


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


class ProgrammaticVideoCorpus:
    def __init__(self, spec: CorpusSpec, records: Sequence[ReferentRecord]):
        self.spec = spec
        self.records = list(records)
        lin = torch.linspace(0.0, 1.0, spec.resolution)
        self.yy, self.xx = torch.meshgrid(lin, lin, indexing="ij")

    @staticmethod
    def _hue(index: int, levels: int) -> torch.Tensor:
        hue = index / max(1, levels)
        return torch.tensor(
            [
                0.5 + 0.5 * math.cos(2 * math.pi * hue),
                0.5 + 0.5 * math.cos(2 * math.pi * (hue + 1 / 3)),
                0.5 + 0.5 * math.cos(2 * math.pi * (hue + 2 / 3)),
            ],
            dtype=torch.float32,
        )

    def clip(self, record: ReferentRecord, *, view: int = 0) -> torch.Tensor:
        spec = self.spec
        generator = torch.Generator().manual_seed(_stable_seed(spec.seed, record.referent, view, "pixels"))
        theta = record.factor_b * math.pi / spec.factor_b_levels
        direction = 1.0 if record.factor_b % 2 == 0 else -1.0
        projection = self.xx * math.cos(theta) + self.yy * math.sin(theta)
        frequency = 3.5 + 0.35 * (record.replicate % 3)
        phase = 0.19 * record.replicate + 0.07 * view
        temporal = torch.arange(spec.frames, dtype=torch.float32) / spec.frames
        carrier = torch.sin(
            2 * math.pi * (frequency * projection[None] + direction * temporal[:, None, None] + phase)
        )
        carrier = (carrier + 1.0) / 2.0
        tint = self._hue(record.factor_a, spec.factor_a_levels)
        foreground = carrier[:, None] * tint[None, :, None, None]

        checker = torch.sin(
            2 * math.pi * ((1.0 + record.replicate % 4) * self.xx + (0.5 + view * 0.03) * self.yy)
        )
        background = (0.06 + 0.025 * checker)[None, None]
        gain = 0.78 + 0.05 * (record.replicate % 4) + 0.015 * ((view % 3) - 1)
        noise = 0.025 * torch.randn((spec.frames, 3, spec.resolution, spec.resolution), generator=generator)
        return (gain * foreground + background + noise).clamp(0.0, 1.0)

    def batch(self, indices: Sequence[int], *, view: int) -> torch.Tensor:
        return torch.stack([self.clip(self.records[index], view=view) for index in indices])


@dataclass(frozen=True)
class ModelSpec:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    patch_size: int = 32
    tubelet: int = 2
    max_resolution: int = 256
    max_frames: int = 16

    def validate(self) -> None:
        if self.dim % self.heads:
            raise ValueError("model dim must be divisible by heads")
        if self.patch_size <= 0 or self.tubelet <= 0:
            raise ValueError("patch_size and tubelet must be positive")
        if self.max_resolution % self.patch_size or self.max_frames % self.tubelet:
            raise ValueError("maximum input geometry must divide exactly into patches")

    @property
    def max_tokens(self) -> int:
        return (self.max_frames // self.tubelet) * (self.max_resolution // self.patch_size) ** 2


class TinyVideoSubstrate(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.patch_embed = nn.Conv3d(
            3,
            spec.dim,
            kernel_size=(spec.tubelet, spec.patch_size, spec.patch_size),
            stride=(spec.tubelet, spec.patch_size, spec.patch_size),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, spec.dim))
        self.position = nn.Parameter(torch.zeros(1, spec.max_tokens, spec.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=spec.dim,
            nhead=spec.heads,
            dim_feedforward=spec.dim * spec.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=spec.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(spec.dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(spec.dim),
            nn.Linear(spec.dim, spec.dim),
            nn.GELU(),
            nn.Linear(spec.dim, spec.dim),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, clips: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError("clips must be [batch,3,time,height,width]")
        patches = self.patch_embed(clips).flatten(2).transpose(1, 2)
        tokens = patches.shape[1]
        if tokens > self.position.shape[1]:
            raise ValueError(f"input has {tokens} tokens but model maximum is {self.position.shape[1]}")
        hidden = patches + self.position[:, :tokens]
        if mask is not None:
            if mask.shape != hidden.shape[:2]:
                raise ValueError("token mask shape does not match token geometry")
            hidden = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(hidden), hidden)
        return self.norm(self.blocks(hidden))

    def forward(self, clips: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(clips, mask).mean(dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def token_count(data: CorpusSpec, model: ModelSpec) -> int:
    if data.frames % model.tubelet or data.resolution % model.patch_size:
        raise ValueError("data geometry must divide exactly into model tubelets and patches")
    return (data.frames // model.tubelet) * (data.resolution // model.patch_size) ** 2


def estimated_train_step_flops(
    data: CorpusSpec,
    model: ModelSpec,
    *,
    batch_size: int,
    objective: str,
    teacher_dim: int = 0,
) -> int:

    n, d, ff = token_count(data, model), model.dim, model.dim * model.mlp_ratio
    conv = 2 * batch_size * n * d * 3 * model.tubelet * model.patch_size**2
    attention = 2 * 4 * batch_size * n * d * d + 4 * batch_size * n * n * d
    mlp = 4 * batch_size * n * d * ff
    encoder_forward = conv + model.depth * (attention + mlp)
    predictor_forward = 4 * batch_size * n * d * d
    total = 4 * encoder_forward + 3 * predictor_forward
    if objective == "reconstruction":
        total += 2 * batch_size * n * 6 * d
    elif objective == OPTIONAL_OBJECTIVE:
        total += 2 * batch_size * max(1, teacher_dim) * d
    total += 2 * batch_size * n * d + 2 * model.depth * d * d
    return int(total)


def _mask_for_step(batch: int, tokens: int, ratio: float, seed: int, step: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(_stable_seed(seed, step, "mask"))
    mask = torch.rand((batch, tokens), generator=generator) < ratio
    mask[:, 0] = True
    return mask


def _batch_for_step(indices: Sequence[int], batch_size: int, seed: int, step: int) -> list[int]:
    generator = torch.Generator().manual_seed(_stable_seed(seed, step, "batch"))
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    if batch_size <= len(indices):
        return [indices[index] for index in permutation[:batch_size]]
    return [indices[permutation[index % len(indices)]] for index in range(batch_size)]


def _fixed_projection(input_dim: int, output_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator) / math.sqrt(input_dim)
    return projection.to(device)


def _patch_descriptors(clips: torch.Tensor, model: ModelSpec) -> torch.Tensor:
    mean = F.avg_pool3d(
        clips,
        kernel_size=(model.tubelet, model.patch_size, model.patch_size),
        stride=(model.tubelet, model.patch_size, model.patch_size),
    )
    second = F.avg_pool3d(
        clips.square(),
        kernel_size=(model.tubelet, model.patch_size, model.patch_size),
        stride=(model.tubelet, model.patch_size, model.patch_size),
    )
    descriptor = torch.cat([mean, second], dim=1)
    return descriptor.flatten(2).transpose(1, 2)


def _update_ema(target: nn.Module, online: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
            target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _accelerator_bytes(device: DeviceInfo) -> int:
    if device.kind == "mps":
        return int(torch.mps.current_allocated_memory())
    if device.kind == "cuda":
        return int(torch.cuda.memory_allocated())
    return 0


def _last_checkpoint_evidence(arm_dir: Path) -> dict[str, Any] | None:
    path = arm_dir / "checkpoint.pt"
    if not path.is_file():
        return None
    evidence: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        evidence.update(
            {
                "schema": checkpoint.get("schema"),
                "step": checkpoint.get("step"),
                "objective": checkpoint.get("objective"),
                "config_sha256": checkpoint.get("config_sha256"),
                "data_sha256": checkpoint.get("data_sha256"),
                "requirements_sha256": checkpoint.get("requirements_sha256"),
            }
        )
    except Exception as exc:  # the failure receipt must survive even a damaged checkpoint
        evidence["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _write_arm_failure(
    arm_dir: Path,
    *,
    seed: int,
    objective: str,
    error: BaseException,
    device: DeviceInfo,
    kind: str,
) -> dict[str, Any]:
    payload = {
        "schema": "mop-custom-substrate-required-arm-failure/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "kind": kind,
        "required_arm": True,
        "seed": seed,
        "objective": objective,
        "exception_type": type(error).__name__,
        "reason": str(error),
        "last_good_checkpoint": _last_checkpoint_evidence(arm_dir),
        "device": device.kind,
        "max_rss_bytes": _max_rss_bytes(),
        "accelerator_allocated_bytes": _accelerator_bytes(device),
        "free_disk_bytes": shutil.disk_usage(REPO_ROOT).free,
        "campaign_must_abort": True,
        "scientific_promotion": False,
    }
    if kind == "unexpected-exception":
        payload["traceback"] = "".join(traceback.format_exception(error))
    filename = "refusal_receipt.json" if kind == "scientific-refusal" else "crash_receipt.json"
    _atomic_json(arm_dir / filename, payload)
    return payload


def _load_checkpoint(
    path: Path,
    *,
    objective: str,
    config_sha256: str,
    data_sha256: str,
    requirements_sha256: str,
    initial_state_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise WorkbenchRefused("checkpoint schema mismatch")
    expected = {
        "objective": objective,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "requirements_sha256": requirements_sha256,
        "initial_state_sha256": initial_state_sha256,
    }
    mismatched = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatched:
        raise WorkbenchRefused(f"checkpoint identity mismatch: {mismatched}")
    return checkpoint


def _save_checkpoint(
    path: Path,
    *,
    objective: str,
    step: int,
    model: TinyVideoSubstrate,
    target: TinyVideoSubstrate,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    data_sha256: str,
    requirements_sha256: str,
    initial_state_sha256: str,
    losses: Sequence[float],
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "objective": objective,
        "step": step,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "requirements_sha256": requirements_sha256,
        "initial_state_sha256": initial_state_sha256,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "target": {key: value.detach().cpu() for key, value in target.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "losses": list(losses),
    }
    _atomic_torch_save(path, payload)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _teacher_columns(path: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and raw.get("schema") == "mop-factor-sidecar/v2":
        return dict(raw.get("columns", {})), dict(raw.get("metadata", {}))
    return dict(raw), {}


def audit_teacher_cache(
    cache_path: str | Path | None,
    *,
    corpus_referents: Iterable[str] = (),
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if not cache_path:
        return {
            "schema": TEACHER_AUDIT_SCHEMA,
            "configured": False,
            "available": False,
            "all_ok": True,
            "problems": [],
            "joined_referents": 0,
        }
    path = _resolve_repo_path(cache_path, repo_root)
    problems: list[str] = []
    if not path.is_dir():
        problems.append("teacher cache directory missing")
        return {
            "schema": TEACHER_AUDIT_SCHEMA,
            "configured": True,
            "path": str(path),
            "available": False,
            "all_ok": False,
            "problems": problems,
            "joined_referents": 0,
        }
    problems.extend(validate_cache_manifest(path, citable=True))
    required = ("referents.json", "factors.json", "cache_manifest.json")
    for name in required:
        if not (path / name).is_file():
            problems.append(f"teacher cache missing {name}")
    referents: list[str] = []
    factor_columns: dict[str, list[int]] = {}
    if (path / "referents.json").is_file():
        referents = [str(value) for value in json.loads((path / "referents.json").read_text())]
    if (path / "factors.json").is_file():
        factor_columns, _ = _teacher_columns(path / "factors.json")
    manifest = json.loads((path / "cache_manifest.json").read_text())
    run_receipt = (
        json.loads((path / "run_receipt.json").read_text()) if (path / "run_receipt.json").is_file() else {}
    )
    claim_scope = str(run_receipt.get("claim_scope", ""))
    scope_lower = claim_scope.lower()
    natural_video = (
        "natural-video" in scope_lower
        and "not natural-video" not in scope_lower
        and "programmatic" not in scope_lower
    )
    corpus_set = set(corpus_referents)
    joined = sorted(corpus_set.intersection(referents))
    store = LatentStore.open(path)
    return {
        "schema": TEACHER_AUDIT_SCHEMA,
        "configured": True,
        "available": not problems,
        "all_ok": not problems,
        "path": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
        "manifest_sha256": sha256_file(path / "cache_manifest.json"),
        "manifest_schema": manifest.get("schema"),
        "form": manifest.get("form"),
        "encoder_config_hash": manifest.get("encoder_config_hash"),
        "claim_scope": claim_scope or "unknown",
        "natural_video": natural_video,
        "rows": len(store),
        "feature_shape": [len(store), *store.meta.feat_shape],
        "referent_count": len(referents),
        "factor_columns": sorted(factor_columns),
        "joined_referents": len(joined),
        "joined_referent_ids": joined,
        "problems": problems,
    }


def load_teacher_targets(cache_path: str | Path, *, repo_root: Path = REPO_ROOT) -> dict[str, torch.Tensor]:
    path = _resolve_repo_path(cache_path, repo_root)
    problems = validate_cache_manifest(path, citable=True)
    if problems:
        raise WorkbenchRefused("teacher cache is not citable: " + "; ".join(problems))
    referents = [str(value) for value in json.loads((path / "referents.json").read_text())]
    features = LatentStore.open(path).latents()
    if features.ndim > 2:
        features = features.flatten(2).mean(dim=1)
    if features.ndim != 2 or len(referents) != features.shape[0]:
        raise WorkbenchRefused("teacher feature and referent shapes disagree")
    return {referent: features[index].float() for index, referent in enumerate(referents)}


def _objective_target(
    objective: str,
    *,
    model: TinyVideoSubstrate,
    target_tokens: torch.Tensor,
    online_clips: torch.Tensor,
    records: Sequence[ReferentRecord],
    teacher_targets: Mapping[str, torch.Tensor] | None,
    step: int,
) -> torch.Tensor:
    dim = model.spec.dim
    if objective in ("predictive", "invariance"):
        return target_tokens.detach()
    if objective == "random_target":
        return target_tokens.detach().roll(1, dims=0).flip(1)
    if objective == "reconstruction":
        descriptor = _patch_descriptors(online_clips, model.spec)
        projection = _fixed_projection(6, dim, 7721, online_clips.device)
        return descriptor @ projection
    if objective == OPTIONAL_OBJECTIVE:
        if teacher_targets is None:
            raise WorkbenchRefused("teacher_distill requested without a teacher target mapping")
        missing = [row.referent for row in records if row.referent not in teacher_targets]
        if missing:
            raise WorkbenchRefused(f"teacher target missing for batch referents: {missing[:3]}")
        teacher = torch.stack([teacher_targets[row.referent] for row in records]).to(online_clips.device)
        projection = _fixed_projection(teacher.shape[1], dim, 8803, online_clips.device)
        return (teacher @ projection).unsqueeze(1).expand(-1, target_tokens.shape[1], -1)
    raise ValueError(f"unknown objective {objective!r}")


def train_arm(
    *,
    objective: str,
    seed: int,
    corpus: ProgrammaticVideoCorpus,
    records: Sequence[ReferentRecord],
    data_spec: CorpusSpec,
    model_spec: ModelSpec,
    initial_state: Mapping[str, torch.Tensor],
    arm_dir: Path,
    device: DeviceInfo,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mask_ratio: float,
    ema_decay: float,
    variance_weight: float,
    checkpoint_every: int,
    config_sha256: str,
    data_sha256: str,
    requirements_sha256: str,
    deadline: float | None = None,
    teacher_targets: Mapping[str, torch.Tensor] | None = None,
    disk_path: Path = REPO_ROOT,
    disk_floor_bytes: int = 0,
    model_factory: Callable[[], TinyVideoSubstrate] | None = None,
    flops_estimator: Callable[..., int] | None = None,
) -> dict[str, Any]:
    if objective not in (*TRAIN_OBJECTIVES, OPTIONAL_OBJECTIVE):
        raise ValueError(f"unknown objective {objective!r}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = arm_dir / "arm_receipt.json"
    initial_hash = _state_sha256(initial_state)
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        identity_ok = all(
            (
                prior.get("objective") == objective,
                prior.get("config_sha256") == config_sha256,
                prior.get("data_sha256") == data_sha256,
                prior.get("requirements_sha256") == requirements_sha256,
                prior.get("initial_state_sha256") == initial_hash,
            )
        )
        if prior.get("complete") and int(prior.get("completed_steps", 0)) >= steps and identity_ok:
            completed_checkpoint = arm_dir / "checkpoint.pt"
            if completed_checkpoint.is_file() and prior.get("checkpoint", {}).get("sha256") == sha256_file(
                completed_checkpoint
            ):
                prior["resumed_from_complete_receipt"] = True
                return prior
        if not identity_ok:
            raise WorkbenchRefused(f"completed arm receipt identity drift at {receipt_path}")

    model = TinyVideoSubstrate(model_spec) if model_factory is None else model_factory()
    model.load_state_dict(initial_state)
    target = TinyVideoSubstrate(model_spec) if model_factory is None else model_factory()
    target.load_state_dict(initial_state)
    target.requires_grad_(False)
    model.to(device.device)
    target.to(device.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    checkpoint_path = arm_dir / "checkpoint.pt"
    losses: list[float] = []
    start_step = 0
    resumed = False
    if checkpoint_path.is_file():
        loaded_checkpoint = _load_checkpoint(
            checkpoint_path,
            objective=objective,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            requirements_sha256=requirements_sha256,
            initial_state_sha256=initial_hash,
        )
        model.load_state_dict(loaded_checkpoint["model"])
        target.load_state_dict(loaded_checkpoint["target"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        torch.random.set_rng_state(loaded_checkpoint["torch_rng_state"])
        losses = [float(value) for value in loaded_checkpoint.get("losses", [])]
        start_step = int(loaded_checkpoint["step"])
        resumed = start_step > 0

    train_indices = [row.index for row in records if row.split == "train"]
    if objective == OPTIONAL_OBJECTIVE:
        available = set(teacher_targets or {})
        train_indices = [index for index in train_indices if records[index].referent in available]
        if len(train_indices) < batch_size:
            raise WorkbenchRefused(
                "teacher_distill needs at least "
                f"{batch_size} joined train referents; have {len(train_indices)}"
            )
    tokens = token_count(data_spec, model_spec)
    started = time.perf_counter()
    max_accelerator = _accelerator_bytes(device)
    complete = True
    stop_reason: str | None = None
    minimum_free_disk = shutil.disk_usage(disk_path).free
    checkpoint_record: dict[str, Any] = {}
    for step in range(start_step, steps):
        free_disk = shutil.disk_usage(disk_path).free
        minimum_free_disk = min(minimum_free_disk, free_disk)
        if disk_floor_bytes and free_disk < disk_floor_bytes:
            complete = False
            stop_reason = "disk_floor"
            break
        if deadline is not None and time.monotonic() >= deadline:
            complete = False
            stop_reason = "wall_budget"
            break
        batch_indices = _batch_for_step(train_indices, batch_size, seed, step)
        batch_records = [records[index] for index in batch_indices]
        view_a = corpus.batch(batch_indices, view=step * 2).permute(0, 2, 1, 3, 4)
        view_b = corpus.batch(batch_indices, view=step * 2 + 1).permute(0, 2, 1, 3, 4)
        view_a = view_a.to(device.device)
        view_b = view_b.to(device.device)
        mask = _mask_for_step(batch_size, tokens, mask_ratio, seed, step).to(device.device)
        optimizer.zero_grad(set_to_none=True)
        online_tokens = model.encode(view_a, mask)
        with torch.no_grad():
            target_view = view_a if objective == "predictive" else view_b
            target_tokens = target.encode(target_view)
        prediction = model.predictor(online_tokens)
        target_value = _objective_target(
            objective,
            model=model,
            target_tokens=target_tokens,
            online_clips=view_a,
            records=batch_records,
            teacher_targets=teacher_targets,
            step=step,
        )
        prediction = F.normalize(prediction, dim=-1)
        target_value = F.normalize(target_value.detach(), dim=-1)
        prediction_loss = (1.0 - (prediction * target_value).sum(dim=-1))[mask].mean()
        feature_std = online_tokens.mean(dim=1).std(dim=0, unbiased=False)
        variance_loss = F.relu(0.10 - feature_std).mean()
        loss = prediction_loss + variance_weight * variance_loss
        if not torch.isfinite(loss):
            raise WorkbenchRefused(f"non-finite {objective} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        _update_ema(target, model, ema_decay)
        if device.kind == "mps":
            torch.mps.synchronize()
        losses.append(float(loss.detach().cpu()))
        max_accelerator = max(max_accelerator, _accelerator_bytes(device))
        completed_step = step + 1
        if completed_step % checkpoint_every == 0 or completed_step == steps:
            checkpoint_record = _save_checkpoint(
                checkpoint_path,
                objective=objective,
                step=completed_step,
                model=model,
                target=target,
                optimizer=optimizer,
                config_sha256=config_sha256,
                data_sha256=data_sha256,
                requirements_sha256=requirements_sha256,
                initial_state_sha256=initial_hash,
                losses=losses,
            )

    completed_steps = len(losses)
    if not checkpoint_record or int(completed_steps) != int(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True).get("step", -1)
        if checkpoint_path.is_file()
        else -1
    ):
        checkpoint_record = _save_checkpoint(
            checkpoint_path,
            objective=objective,
            step=completed_steps,
            model=model,
            target=target,
            optimizer=optimizer,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            requirements_sha256=requirements_sha256,
            initial_state_sha256=initial_hash,
            losses=losses,
        )
    elapsed = time.perf_counter() - started
    teacher_dim = next(iter(teacher_targets.values())).numel() if teacher_targets else 0
    estimator = estimated_train_step_flops if flops_estimator is None else flops_estimator
    per_step = estimator(
        data_spec,
        model_spec,
        batch_size=batch_size,
        objective=objective,
        teacher_dim=teacher_dim,
    )
    receipt = {
        "schema": ARM_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": "programmatic-video objective probe; not natural-video evidence",
        "objective": objective,
        "seed": seed,
        "complete": complete and completed_steps >= steps,
        "stop_reason": stop_reason,
        "resumed": resumed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "batch_size": batch_size,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "requirements_sha256": requirements_sha256,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_sha256(model.state_dict()),
        "target_state_sha256": _state_sha256(target.state_dict()),
        "checkpoint": checkpoint_record,
        "loss": {
            "initial": losses[0] if losses else None,
            "final": losses[-1] if losses else None,
            "minimum": min(losses) if losses else None,
            "count": len(losses),
        },
        "compute": {
            "estimated_flops_per_step": per_step,
            "estimated_total_flops": per_step * completed_steps,
            "estimator": "conv3d plus transformer attention/mlp plus predictor/backward/EMA arithmetic",
        },
        "telemetry": {
            "seconds_this_invocation": elapsed,
            "seconds_per_completed_step_this_invocation": elapsed / max(1, completed_steps - start_step),
            "max_rss_bytes": _max_rss_bytes(),
            "max_accelerator_allocated_bytes": max_accelerator,
            "device": device.kind,
            "minimum_free_disk_bytes": minimum_free_disk,
            "free_disk_bytes_at_end": shutil.disk_usage(disk_path).free,
            "disk_floor_bytes": disk_floor_bytes,
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def load_arm_model(
    arm_dir: Path,
    model_spec: ModelSpec,
    *,
    device: DeviceInfo,
    model_factory: Callable[[], TinyVideoSubstrate] | None = None,
) -> TinyVideoSubstrate:
    checkpoint = torch.load(arm_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = TinyVideoSubstrate(model_spec) if model_factory is None else model_factory()
    model.load_state_dict(checkpoint["model"])
    model.to(device.device).eval()
    return model


def _extract_embeddings(
    model: TinyVideoSubstrate,
    corpus: ProgrammaticVideoCorpus,
    records: Sequence[ReferentRecord],
    *,
    device: DeviceInfo,
    batch_size: int,
    view: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            stop = min(len(records), start + batch_size)
            indices = list(range(start, stop))
            clips = corpus.batch(indices, view=view).permute(0, 2, 1, 3, 4).to(device.device)
            output.append(model(clips).float().cpu())
    return torch.cat(output)


def _ridge_fit_predict(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    classes: int,
    ridge: float = 1.0e-2,
) -> float:
    train = features[list(train_indices)].double()
    test = features[list(eval_indices)].double()
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-4)
    train = (train - mean) / std
    test = (test - mean) / std
    train = torch.cat([train, torch.ones(train.shape[0], 1, dtype=train.dtype)], dim=1)
    test = torch.cat([test, torch.ones(test.shape[0], 1, dtype=test.dtype)], dim=1)
    targets = F.one_hot(labels[list(train_indices)].long(), num_classes=classes).double()
    identity = torch.eye(train.shape[1], dtype=train.dtype)
    weights = torch.linalg.solve(train.T @ train + ridge * identity, train.T @ targets)
    prediction = (test @ weights).argmax(dim=1)
    return float((prediction == labels[list(eval_indices)]).double().mean())


def _ridge_alignment_r2(
    features: torch.Tensor,
    target: torch.Tensor,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    ridge: float = 1.0e-2,
) -> float:
    train = features[list(train_indices)].double()
    test = features[list(eval_indices)].double()
    train = torch.cat([train, torch.ones(train.shape[0], 1, dtype=train.dtype)], dim=1)
    test = torch.cat([test, torch.ones(test.shape[0], 1, dtype=test.dtype)], dim=1)
    identity = torch.eye(train.shape[1], dtype=train.dtype)
    weights = torch.linalg.solve(
        train.T @ train + ridge * identity, train.T @ target[list(train_indices)].double()
    )
    prediction = test @ weights
    truth = target[list(eval_indices)].double()
    residual = (prediction - truth).square().sum()
    centered = (truth - truth.mean(0, keepdim=True)).square().sum().clamp_min(1.0e-12)
    return float(1.0 - residual / centered)


def _pair_geometry(features: torch.Tensor, records: Sequence[ReferentRecord]) -> dict[str, float]:
    normalized = F.normalize(features.float(), dim=1)
    similarity = normalized @ normalized.T
    same: list[float] = []
    different: list[float] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            value = float(similarity[left, right])
            equal = (
                records[left].factor_a == records[right].factor_a
                and records[left].factor_b == records[right].factor_b
            )
            (same if equal else different).append(value)
    same_mean = sum(same) / max(1, len(same))
    different_mean = sum(different) / max(1, len(different))
    return {
        "same_combination_cosine": same_mean,
        "different_combination_cosine": different_mean,
        "combination_geometry_gap": same_mean - different_mean,
    }


def evaluate_model(
    model: TinyVideoSubstrate,
    corpus: ProgrammaticVideoCorpus,
    records: Sequence[ReferentRecord],
    *,
    device: DeviceInfo,
    batch_size: int,
) -> dict[str, Any]:
    features = _extract_embeddings(model, corpus, records, device=device, batch_size=batch_size, view=100001)
    augmented = _extract_embeddings(model, corpus, records, device=device, batch_size=batch_size, view=100002)
    train = [row.index for row in records if row.split == "train"]
    val = [row.index for row in records if row.split == "val"]
    test = [row.index for row in records if row.split == "test"]
    factor_a = torch.tensor([row.factor_a for row in records])
    factor_b = torch.tensor([row.factor_b for row in records])
    a_levels = max(factor_a).item() + 1
    b_levels = max(factor_b).item() + 1
    a_test = _ridge_fit_predict(features, factor_a, train, test, a_levels)
    b_test = _ridge_fit_predict(features, factor_b, train, test, b_levels)
    a_val = _ridge_fit_predict(features, factor_a, train, val, a_levels)
    b_val = _ridge_fit_predict(features, factor_b, train, val, b_levels)
    oracle = torch.cat([F.one_hot(factor_a, a_levels), F.one_hot(factor_b, b_levels)], dim=1).float()
    alignment_r2 = _ridge_alignment_r2(features, oracle, train, test)
    stability = float(F.cosine_similarity(features, augmented, dim=1).mean())
    return {
        "heldout_factor_a_accuracy": a_test,
        "heldout_factor_b_accuracy": b_test,
        "heldout_combo_score": (a_test + b_test) / 2.0,
        "validation_combo_score": (a_val + b_val) / 2.0,
        "chance": 0.5 * (1.0 / a_levels + 1.0 / b_levels),
        "referent_view_stability": stability,
        "heldout_oracle_form_alignment_r2": alignment_r2,
        "feature_dim": features.shape[1],
        **_pair_geometry(features, records),
    }


def oracle_difficulty_calibration(records: Sequence[ReferentRecord]) -> dict[str, Any]:
    factor_a = torch.tensor([row.factor_a for row in records])
    factor_b = torch.tensor([row.factor_b for row in records])
    a_levels = max(factor_a).item() + 1
    b_levels = max(factor_b).item() + 1
    oracle = torch.cat([F.one_hot(factor_a, a_levels), F.one_hot(factor_b, b_levels)], dim=1).float()
    train = [row.index for row in records if row.split == "train"]
    test = [row.index for row in records if row.split == "test"]
    a = _ridge_fit_predict(oracle, factor_a, train, test, a_levels)
    b = _ridge_fit_predict(oracle, factor_b, train, test, b_levels)
    return {
        "factor_a_accuracy": a,
        "factor_b_accuracy": b,
        "mean_accuracy": (a + b) / 2.0,
        "clears_floor": min(a, b) >= 0.90,
        "interpretation": "programmatic oracle proves factor labels remain solvable on held-out combinations",
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_ci(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    mean = _mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "lo": mean, "hi": mean, "half": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half, "half": half}


def _aggregate_results(seed_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    arms = sorted({arm for seed in seed_results.values() for arm in seed})
    output: dict[str, Any] = {}
    for arm in arms:
        rows = [seed[arm] for seed in seed_results.values() if arm in seed and seed[arm].get("evaluation")]
        if not rows:
            continue
        metrics = (
            "heldout_combo_score",
            "referent_view_stability",
            "heldout_oracle_form_alignment_r2",
            "combination_geometry_gap",
        )
        output[arm] = {
            "seeds": len(rows),
            **{metric: _paired_ci([float(row["evaluation"][metric]) for row in rows]) for metric in metrics},
            "estimated_total_flops": sum(
                int(row.get("training", {}).get("compute", {}).get("estimated_total_flops", 0))
                for row in rows
            ),
            "wall_seconds": sum(
                float(row.get("training", {}).get("telemetry", {}).get("seconds_this_invocation", 0.0))
                for row in rows
            ),
        }
    return output


def _promotion_decision(
    *,
    aggregate: Mapping[str, Any],
    difficulty: Mapping[str, Any],
    requirements: Mapping[str, Any],
    teacher: Mapping[str, Any],
    min_seeds: int,
    margin: float,
    ceiling: float,
    parameter_count_value: int,
    compute_match: Mapping[str, Any],
    teacher_min_rows: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not requirements.get("all_ok"):
        reasons.append("requirements evidence audit is not clean")
    if not difficulty.get("clears_floor"):
        reasons.append("programmatic difficulty oracle did not clear its floor")
    if not 1_000_000 <= parameter_count_value <= 5_000_000:
        reasons.append("trainable parameter count is outside the preregistered 1 to 5M envelope")
    if not compute_match.get("all_ok"):
        reasons.append("trained arms are not compute matched")
    learned = [arm for arm in ("predictive", "invariance", "reconstruction") if arm in aggregate]
    complete_seed_count = min(
        [
            int(aggregate[arm].get("seeds", 0))
            for arm in (*learned, "random_target", "frozen_random")
            if arm in aggregate
        ]
        or [0]
    )
    if complete_seed_count < min_seeds:
        reasons.append(f"need {min_seeds} complete seeds; have {complete_seed_count}")
    best_arm = max(
        learned,
        key=lambda arm: aggregate[arm]["heldout_combo_score"]["mean"],
        default=None,
    )
    best_vs_random: dict[str, Any] | None = None
    best_vs_frozen: dict[str, Any] | None = None
    best_vs_second: dict[str, Any] | None = None
    if best_arm and "random_target" in aggregate and "frozen_random" in aggregate:
        best_mean = aggregate[best_arm]["heldout_combo_score"]["mean"]
        random_mean = aggregate["random_target"]["heldout_combo_score"]["mean"]
        frozen_mean = aggregate["frozen_random"]["heldout_combo_score"]["mean"]
        random_delta = best_mean - random_mean
        frozen_delta = best_mean - frozen_mean
        best_vs_random = {
            "mean_delta": random_delta,
            "conservative_lower": random_delta
            - aggregate[best_arm]["heldout_combo_score"]["half"]
            - aggregate["random_target"]["heldout_combo_score"]["half"],
        }
        best_vs_frozen = {
            "mean_delta": frozen_delta,
            "conservative_lower": frozen_delta
            - aggregate[best_arm]["heldout_combo_score"]["half"]
            - aggregate["frozen_random"]["heldout_combo_score"]["half"],
        }
        runner_up = max(
            (arm for arm in learned if arm != best_arm),
            key=lambda arm: aggregate[arm]["heldout_combo_score"]["mean"],
            default=None,
        )
        if runner_up is not None:
            second_delta = best_mean - aggregate[runner_up]["heldout_combo_score"]["mean"]
            best_vs_second = {
                "objective": runner_up,
                "mean_delta": second_delta,
                "conservative_lower": second_delta
                - aggregate[best_arm]["heldout_combo_score"]["half"]
                - aggregate[runner_up]["heldout_combo_score"]["half"],
            }
            if best_vs_second["conservative_lower"] <= margin:
                reasons.append("best learned objective does not clear the strongest alternative objective")
        else:
            reasons.append("fewer than two learned objective arms are complete")
        if best_mean >= ceiling:
            reasons.append(
                f"best learned score {best_mean:.4f} reaches the preregistered {ceiling:.4f} ceiling"
            )
        if best_vs_random["conservative_lower"] <= margin:
            reasons.append("best learned objective does not clear the matched random-target control")
        if best_vs_frozen["conservative_lower"] <= margin:
            reasons.append("best learned objective does not clear the exact frozen initialization")
    else:
        reasons.append("one or more required objective/control arms are incomplete")
    cm7_promotable = not reasons

    cm8_reasons = list(reasons)
    if not teacher.get("all_ok"):
        cm8_reasons.append("optional teacher cache is absent or invalid")
    if int(teacher.get("joined_referents", 0)) < teacher_min_rows:
        cm8_reasons.append(
            f"CM8 needs {teacher_min_rows} exact same-referent teacher rows; "
            f"have {teacher.get('joined_referents', 0)}"
        )
    if not teacher.get("natural_video"):
        cm8_reasons.append("teacher control is programmatic rather than rights-cleared natural video")
    cm8_reasons.append("CM1/CM2/DR1 upstream natural-data gates are not supplied to this local receipt")
    return {
        "best_objective": best_arm,
        "best_vs_second_objective": best_vs_second,
        "best_vs_random_target": best_vs_random,
        "best_vs_frozen_random": best_vs_frozen,
        "cm7_local_objective_lever_promotable": cm7_promotable,
        "cm7_reasons": reasons,
        "cm8_custom_build_promotable": False,
        "cm8_reasons": sorted(set(cm8_reasons)),
        "scope_boundary": (
            "CM7 promotion means only that objective choice is a live lever on this deterministic "
            "programmatic task. It does not license a natural-video or general-capability claim."
        ),
    }


def _compute_match(seed_results: Mapping[str, Mapping[str, Any]], tolerance: float) -> dict[str, Any]:
    totals: dict[str, list[int]] = {objective: [] for objective in TRAIN_OBJECTIVES}
    for seed in seed_results.values():
        for objective in TRAIN_OBJECTIVES:
            row = seed.get(objective, {})
            training = row.get("training", {})
            if training.get("complete"):
                totals[objective].append(int(training["compute"]["estimated_total_flops"]))
    means = {key: _mean(values) for key, values in totals.items() if values}
    if len(means) < len(TRAIN_OBJECTIVES):
        return {
            "all_ok": False,
            "tolerance_fraction": tolerance,
            "mean_flops": means,
            "max_fractional_deviation": None,
            "reason": "one or more trained arms are incomplete",
        }
    reference = _mean(list(means.values()))
    deviation = max(abs(value - reference) / max(1.0, reference) for value in means.values())
    return {
        "all_ok": deviation <= tolerance,
        "tolerance_fraction": tolerance,
        "mean_flops": means,
        "max_fractional_deviation": deviation,
        "reference_mean_flops": reference,
    }


def run_workbench(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    device: DeviceInfo,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    run_dir.mkdir(parents=True, exist_ok=True)
    config_plain = json.loads(json.dumps(config))
    config_sha = json_sha256(config_plain)
    _atomic_json(run_dir / "resolved_config.json", config_plain)

    requirements = audit_requirements(config_plain["requirements_ledger"], repo_root=repo_root)
    snapshot_requirement_sources(
        requirements,
        destination=run_dir / "requirements_sources",
        repo_root=repo_root,
    )
    _atomic_json(run_dir / "requirements_audit.json", requirements)
    if bool(config_plain.get("strict_requirements", True)) and not requirements["all_ok"]:
        raise WorkbenchRefused("requirements audit failed: " + "; ".join(requirements["problems"]))

    data_spec = CorpusSpec(**config_plain["data"])
    model_spec = ModelSpec(**config_plain["model"])
    records = build_referent_records(data_spec)
    manifest = dataset_manifest(data_spec, records)
    if not manifest["disjoint_referents"]:
        raise WorkbenchRefused("dataset referents overlap across splits")
    _atomic_json(run_dir / "dataset_manifest.json", manifest)
    implementation = snapshot_implementation_sources(
        run_dir,
        expected_generator_sha256=manifest["generator"]["source_sha256"],
        repo_root=repo_root,
    )
    corpus = ProgrammaticVideoCorpus(data_spec, records)

    training = config_plain["training"]
    objectives = [str(value) for value in training["objectives"]]
    unknown = sorted(set(objectives) - set((*TRAIN_OBJECTIVES, OPTIONAL_OBJECTIVE)))
    if unknown:
        raise ValueError(f"unknown objectives: {unknown}")
    for required in TRAIN_OBJECTIVES:
        if required not in objectives:
            raise WorkbenchRefused(f"matched local tournament requires objective {required!r}")

    teacher_path = config_plain.get("teacher", {}).get("cache_path")
    teacher = audit_teacher_cache(
        teacher_path,
        corpus_referents=[row.referent for row in records],
        repo_root=repo_root,
    )
    _atomic_json(run_dir / "teacher_audit.json", teacher)
    teacher_targets = None
    if OPTIONAL_OBJECTIVE in objectives:
        if not teacher.get("all_ok"):
            raise WorkbenchRefused("teacher_distill requested but teacher audit is not clean")
        teacher_targets = load_teacher_targets(teacher_path, repo_root=repo_root)

    params_probe = TinyVideoSubstrate(model_spec)
    params = parameter_count(params_probe)
    min_params = int(config_plain.get("promotion", {}).get("min_params", 1_000_000))
    max_params = int(config_plain.get("promotion", {}).get("max_params", 5_000_000))
    if not min_params <= params <= max_params:
        raise WorkbenchRefused(f"model has {params} params outside [{min_params}, {max_params}]")

    seeds = [int(value) for value in training["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    wall_budget = float(training.get("wall_budget_seconds", 10800.0))
    deadline = time.monotonic() + wall_budget
    seed_results: dict[str, dict[str, Any]] = {}
    difficulty = oracle_difficulty_calibration(records)
    started = time.perf_counter()
    stopped_for_wall = False
    stopped_for_disk = False
    stopped_for_required_arm = False
    required_arm_failure: dict[str, Any] | None = None
    disk_floor_bytes = int(float(training.get("min_free_disk_gb", 40.0)) * 1_000_000_000)

    for seed in seeds:
        if shutil.disk_usage(repo_root).free < disk_floor_bytes:
            stopped_for_disk = True
            break
        torch.manual_seed(seed)
        initial_model = TinyVideoSubstrate(model_spec)
        initial_state = {
            key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()
        }
        initial_hash = _state_sha256(initial_state)
        seed_key = str(seed)
        seed_results[seed_key] = {}
        frozen_model = TinyVideoSubstrate(model_spec)
        frozen_model.load_state_dict(initial_state)
        frozen_model.to(device.device).eval()
        frozen_evaluation = evaluate_model(
            frozen_model,
            corpus,
            records,
            device=device,
            batch_size=int(training["eval_batch_size"]),
        )
        seed_results[seed_key]["frozen_random"] = {
            "control": "exact same-architecture, same-initialization frozen encoder",
            "initial_state_sha256": initial_hash,
            "evaluation": frozen_evaluation,
            "training": {
                "complete": True,
                "compute": {"estimated_total_flops": 0},
                "telemetry": {"seconds_this_invocation": 0.0},
            },
        }
        del frozen_model

        for objective in objectives:
            if time.monotonic() >= deadline:
                stopped_for_wall = True
                break
            arm_dir = run_dir / "arms" / f"seed_{seed}" / objective
            try:
                arm = train_arm(
                    objective=objective,
                    seed=seed,
                    corpus=corpus,
                    records=records,
                    data_spec=data_spec,
                    model_spec=model_spec,
                    initial_state=initial_state,
                    arm_dir=arm_dir,
                    device=device,
                    steps=int(training["steps"]),
                    batch_size=int(training["batch_size"]),
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                    mask_ratio=float(training["mask_ratio"]),
                    ema_decay=float(training["ema_decay"]),
                    variance_weight=float(training["variance_weight"]),
                    checkpoint_every=int(training["checkpoint_every"]),
                    config_sha256=config_sha,
                    data_sha256=manifest["content_sha256"],
                    requirements_sha256=requirements["aggregate_sha256"],
                    deadline=deadline,
                    teacher_targets=teacher_targets,
                    disk_path=repo_root,
                    disk_floor_bytes=disk_floor_bytes,
                )
                result: dict[str, Any] = {"training": arm}
                if arm["complete"]:
                    trained = load_arm_model(arm_dir, model_spec, device=device)
                    result["evaluation"] = evaluate_model(
                        trained,
                        corpus,
                        records,
                        device=device,
                        batch_size=int(training["eval_batch_size"]),
                    )
                    result["exact_initialization_control"] = arm["initial_state_sha256"] == initial_hash
                    del trained
                seed_results[seed_key][objective] = result
                if not arm["complete"]:
                    if arm.get("stop_reason") == "disk_floor":
                        stopped_for_disk = True
                    else:
                        stopped_for_wall = True
                    break
            except WorkbenchRefused as exc:
                required_arm_failure = _write_arm_failure(
                    arm_dir,
                    seed=seed,
                    objective=objective,
                    error=exc,
                    device=device,
                    kind="scientific-refusal",
                )
                seed_results[seed_key][objective] = {
                    "training": {
                        "complete": False,
                        "refused": True,
                        "reason": str(exc),
                        "refusal_receipt": str(arm_dir / "refusal_receipt.json"),
                    }
                }
                stopped_for_required_arm = True
                break
            except Exception as exc:
                _write_arm_failure(
                    arm_dir,
                    seed=seed,
                    objective=objective,
                    error=exc,
                    device=device,
                    kind="unexpected-exception",
                )
                raise
        if stopped_for_wall or stopped_for_disk or stopped_for_required_arm:
            break

    aggregate = _aggregate_results(seed_results)
    compute_match = _compute_match(
        seed_results, float(config_plain["promotion"].get("compute_tolerance", 0.005))
    )
    promotion = _promotion_decision(
        aggregate=aggregate,
        difficulty=difficulty,
        requirements=requirements,
        teacher=teacher,
        min_seeds=int(config_plain["promotion"]["min_seeds"]),
        margin=float(config_plain["promotion"]["margin"]),
        ceiling=float(config_plain["promotion"].get("ceiling", 0.98)),
        parameter_count_value=params,
        compute_match=compute_match,
        teacher_min_rows=int(config_plain["promotion"]["teacher_min_rows"]),
    )
    if required_arm_failure is not None:
        reason = (
            "required arm refused and aborted the campaign: "
            f"seed={required_arm_failure['seed']} objective={required_arm_failure['objective']} "
            f"reason={required_arm_failure['reason']}"
        )
        promotion["cm7_local_objective_lever_promotable"] = False
        promotion["cm8_custom_build_promotable"] = False
        promotion.setdefault("cm7_reasons", []).append(reason)
        promotion.setdefault("cm8_reasons", []).append(reason)
    completed = all(
        objective in seed_results.get(str(seed), {})
        and seed_results[str(seed)][objective].get("training", {}).get("complete")
        for seed in seeds
        for objective in objectives
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": WORKBENCH_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": (
            "custom 1 to 5M video-token substrate on deterministic programmatic video; "
            "not natural-video, intelligence, sentience, or general-capability evidence"
        ),
        "complete": completed,
        "resumable": not completed,
        "stopped_for_wall_budget": stopped_for_wall,
        "stopped_for_disk_floor": stopped_for_disk,
        "stopped_for_required_arm_refusal": stopped_for_required_arm,
        "required_arm_failure": required_arm_failure,
        "config_sha256": config_sha,
        "data_sha256": manifest["content_sha256"],
        "requirements_sha256": requirements["aggregate_sha256"],
        "requirements": {
            "all_ok": requirements["all_ok"],
            "requirement_ids": requirements["requirement_ids"],
            "audit_path": "requirements_audit.json",
        },
        "implementation": {
            "manifest_path": "implementation_manifest.json",
            "aggregate_sha256": implementation["aggregate_sha256"],
            "all_ok": implementation["all_ok"],
        },
        "dataset": {
            "manifest_path": "dataset_manifest.json",
            "rows": len(records),
            "split_counts": {
                split: sum(row.split == split for row in records) for split in ("train", "val", "test")
            },
            "disjoint_referents": manifest["disjoint_referents"],
            "combination_disjoint": manifest["combination_disjoint"],
            "resolution": data_spec.resolution,
            "frames": data_spec.frames,
        },
        "model": {
            "architecture": "TinyVideoSubstrate",
            "spec": asdict(model_spec),
            "trainable_parameters": params,
            "token_count": token_count(data_spec, model_spec),
            "teacher_independent": True,
            "exports": ["dense_spatiotemporal_tokens", "pooled_retrieval_key"],
        },
        "controls": {
            "frozen_random": "exact same state hash as every trained arm",
            "random_target": "same initialization, architecture, batches, updates, and core compute",
            "teacher": teacher,
        },
        "difficulty_calibration": difficulty,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "compute_match": compute_match,
        "promotion": promotion,
        "resource_telemetry": {
            "wall_budget_seconds": wall_budget,
            "wall_seconds_this_invocation": elapsed,
            "max_rss_bytes": _max_rss_bytes(),
            "accelerator_allocated_bytes_at_end": _accelerator_bytes(device),
            "device": device.kind,
            "free_disk_bytes_at_end": shutil.disk_usage(repo_root).free,
            "disk_floor_bytes": disk_floor_bytes,
        },
    }
    _atomic_json(run_dir / "workbench_receipt.json", receipt)
    return receipt


def cm8_preflight(
    config: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    config_plain = json.loads(json.dumps(config))
    requirements = audit_requirements(config_plain["requirements_ledger"], repo_root=repo_root)
    data_spec = CorpusSpec(**config_plain["data"])
    records = build_referent_records(data_spec)
    teacher = audit_teacher_cache(
        config_plain.get("teacher", {}).get("cache_path"),
        corpus_referents=[row.referent for row in records],
        repo_root=repo_root,
    )
    cm7_path = _resolve_repo_path(config_plain["cm7_receipt"], repo_root)
    cm7 = json.loads(cm7_path.read_text()) if cm7_path.is_file() else None
    min_rows = int(config_plain["promotion"]["teacher_min_rows"])
    blockers: list[str] = []
    if not requirements.get("all_ok"):
        blockers.append("requirements audit is not clean")
    if not cm7:
        blockers.append("durable CM7 workbench receipt is missing")
    elif not cm7.get("promotion", {}).get("cm7_local_objective_lever_promotable"):
        blockers.append("CM7 has not established an objective lever under its local controls")
    if not teacher.get("all_ok"):
        blockers.append("teacher cache is missing or invalid")
    if int(teacher.get("joined_referents", 0)) < min_rows:
        blockers.append(
            f"teacher cache has {teacher.get('joined_referents', 0)} joined referents; need {min_rows}"
        )
    if not teacher.get("natural_video"):
        blockers.append("teacher cache is programmatic rather than rights-cleared natural video")
    blockers.extend(
        [
            "CM1 natural-video compositional gate is not supplied",
            "CM2 multi-encoder atlas gate is not supplied",
            "DR1 rights-cleared dense natural-video package is not supplied",
        ]
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": "local CM8 execution preflight; no custom-build scientific result",
        "runnable_local_preflight": True,
        "scientific_execution_ready": not blockers,
        "scientific_promotion": False,
        "requirements": {
            "all_ok": requirements.get("all_ok"),
            "aggregate_sha256": requirements.get("aggregate_sha256"),
        },
        "cm7_receipt": {
            "path": str(cm7_path),
            "exists": cm7_path.is_file(),
            "sha256": sha256_file(cm7_path) if cm7_path.is_file() else None,
        },
        "teacher": teacher,
        "blockers": blockers,
        "next_scientific_action": (
            "expand an exact same-referent citable teacher cache and natural-video test package, then "
            "run teacher_distill beside the teacher-free objective winner at matched compute"
        ),
    }
