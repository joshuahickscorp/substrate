
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import resource
import threading
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..devices import DeviceInfo, apple_silicon_info
from ..diagnostics.compute import mlp_flops, param_count
from ..diagnostics.performance_density import density_block
from ..diagnostics.riskcov import seed_ci, sign_flip_report
from ..seeding import seed_everything
from ..studio.profiles import get_profile
from .base import _mean
from mop.substrate.events import sha256_file

DATASET_SCHEMA = "mop-rewrite-dataset/v1"
RIGHTS_SCHEMA = "mop-rewrite-data-rights/v1"
ENCODER_SCHEMA = "mop-rewrite-encoder-receipt/v1"
WEIGHTS_FORMAT = "mop-mlp-npz/v1"
COMPUTE_PLAN_SCHEMA = "mop-matched-compute-plan/v1"
SHELL_FAILURE_SCHEMA = "mop-shell-failure-receipt/v1"
INHERITED_BASELINE_SCHEMA = "mop-inherited-baseline-receipt/v1"
SEED_PLAN_SCHEMA = "mop-seed-plan/v1"
PREFLIGHT_SCHEMA = "mop-scientific-preflight/v2"
ATTEMPT_SCHEMA = "mop-rewrite-attempt/v1"
PROJECTION_SCHEMA = "mop-rewrite-resource-projection/v1"
ADAM_FLOPS_PER_PARAMETER_UPDATE = 12

F8_ARMS = (
    "plastic_rewrite",
    "frozen_inherited",
    "larger_frozen_shell",
    "random_init_same_arch",
)
F16_ARMS = (
    "blank_slate",
    "frozen_inherited",
    "larger_frozen_shell",
    "random_init_same_arch",
)


class ScientificExecutionRefused(RuntimeError):
    pass


@dataclass
class RewritePackage:
    inputs: torch.Tensor
    view_a: torch.Tensor
    view_b: torch.Tensor
    factor_labels: torch.Tensor
    transfer_labels: torch.Tensor
    domain_labels: torch.Tensor
    split: torch.Tensor
    referent_ids: list[str]
    inherited_features: torch.Tensor
    inherited_encoder: nn.Module
    encoder_dims: list[int]
    activation: str
    evidence_scope: str
    evidence_documents: dict[str, dict[str, Any]]
    evidence_hashes: dict[str, str]
    artifact_bytes: int
    resident_tensor_bytes: int


@dataclass
class ArmResult:
    accuracy: float
    validation_accuracy: float
    flops: int
    updates: int
    examples: int
    trainable_params: int
    representation_shift: float = 0.0
    initialization_fingerprint: str = ""


class _ManifestMLP(nn.Module):
    def __init__(self, dims: Sequence[int], activation: str):
        super().__init__()
        self.dims = [int(v) for v in dims]
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                if self.activation == "gelu":
                    x = F.gelu(x)
                elif self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "tanh":
                    x = torch.tanh(x)
                else:  # defended again during manifest validation
                    raise ValueError(f"unsupported activation {self.activation!r}")
        return x


def _activation_forward_flops(activation: str) -> int:
    return 1 if activation == "relu" else 8


def _mlp_forward_flops(dims: Sequence[int], activation: str = "none") -> int:
    linear = mlp_flops([int(value) for value in dims], batch=1)
    if activation == "none" or len(dims) <= 2:
        return linear
    return linear + _activation_forward_flops(activation) * sum(int(v) for v in dims[1:-1])


def _classification_train_flops(dims: Sequence[int], classes: int, activation: str = "none") -> int:
    return 3 * _mlp_forward_flops(dims, activation) + 10 * int(classes)


def _ssl_train_flops(dims: Sequence[int], activation: str) -> int:
    output_dim = int(dims[-1])
    return 6 * _mlp_forward_flops(dims, activation) + 24 * output_dim


def _adam_update_flops(module: nn.Module) -> int:
    return ADAM_FLOPS_PER_PARAMETER_UPDATE * param_count(module)


def _module_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _margin_null_decision(deltas: Sequence[float], margin: float, compute_matched: bool) -> dict[str, Any]:
    raw = [float(value) for value in deltas]
    adjusted = [value - float(margin) for value in raw]
    raw_ci = seed_ci(raw)
    adjusted_ci = seed_ci(adjusted)
    flips = sign_flip_report(adjusted)
    rejects = bool(
        _mean(adjusted) > 0.0
        and float(adjusted_ci["lo"]) > 0.0
        and not bool(flips["any_flip"])
        and compute_matched
    )
    return {
        "rejects": rejects,
        "raw_seed_ci": raw_ci,
        "seed_ci": adjusted_ci,
        "sign_flip": flips,
        "margin_adjusted_deltas": adjusted,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _npz_uncompressed_bytes(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as archive:
            return sum(record.file_size for record in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid NPZ archive {path}: {exc}") from exc


def _resolved_artifact(document_path: Path, value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = document_path.parent / path
    return path.resolve()


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return _peak_rss_bytes()


def _device_allocated_bytes(device: DeviceInfo) -> int:
    try:
        if device.kind == "cuda":
            return int(torch.cuda.memory_allocated(device.device))
        if device.kind == "mps":
            return int(torch.mps.current_allocated_memory())
    except (RuntimeError, AttributeError):
        return 0
    return 0


class _AttemptResourceMeter:

    def __init__(self, device: DeviceInfo, interval_seconds: float = 0.01):
        self.device = device
        self.interval_seconds = interval_seconds
        self.rss_start = _current_rss_bytes()
        self.rss_peak = self.rss_start
        self.device_start = _device_allocated_bytes(device)
        self.device_peak = self.device_start
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        self.rss_peak = max(self.rss_peak, _current_rss_bytes())
        self.device_peak = max(self.device_peak, _device_allocated_bytes(self.device))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def __enter__(self) -> _AttemptResourceMeter:
        if self.device.kind == "cuda":
            with suppress(RuntimeError):
                torch.cuda.reset_peak_memory_stats(self.device.device)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sample()
        if self.device.kind == "cuda":
            with suppress(RuntimeError):
                self.device_peak = max(
                    self.device_peak, int(torch.cuda.max_memory_allocated(self.device.device))
                )

    def snapshot(self) -> dict[str, Any]:
        self.sample()
        return {
            "schema": "mop-attempt-resource-sample/v1",
            "window_specific": True,
            "sampling_interval_seconds": self.interval_seconds,
            "rss_start_bytes": self.rss_start,
            "rss_peak_sampled_bytes": self.rss_peak,
            "rss_peak_delta_bytes": max(0, self.rss_peak - self.rss_start),
            "process_lifetime_ru_maxrss_bytes": _peak_rss_bytes(),
            "device_kind": self.device.kind,
            "device_allocated_start_bytes": self.device_start,
            "device_allocated_peak_bytes": self.device_peak,
            "device_peak_kind": (
                "cuda-runtime-peak"
                if self.device.kind == "cuda"
                else "sampled-current-allocation"
                if self.device.kind == "mps"
                else "not-applicable"
            ),
            "limitations": (
                "RSS is sampled, process allocation spikes shorter than the interval may be missed; "
                "MPS exposes current allocation but no resettable per-attempt peak API"
            ),
        }


def _load_evidence_document(
    path_text: str,
    requirements: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not path_text.strip():
        return {"status": "missing", "path": "", "problems": ["path is empty"]}, None
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        return {
            "status": "missing",
            "path": str(path),
            "problems": ["file does not exist"],
        }, None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(path), "problems": [str(exc)]}, None
    if not isinstance(value, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "problems": ["JSON root is not a mapping"],
        }, None
    problems: list[str] = []
    for key, expected in requirements.items():
        found = value.get(key)
        if expected is True and found is not True:
            problems.append(f"{key} must be true")
        elif expected == "nonempty" and not str(found or "").strip():
            problems.append(f"{key} must be nonempty")
        elif expected not in (True, "nonempty") and found != expected:
            problems.append(f"{key} must equal {expected!r}")
    return {
        "status": "valid" if not problems else "invalid",
        "path": str(path),
        "sha256": sha256_file(path),
        "problems": problems,
    }, value


def _requirements(variant: str) -> dict[str, dict[str, object]]:
    common: dict[str, dict[str, object]] = {
        "data_rights_manifest": {
            "schema": RIGHTS_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "rights_granted": True,
            "artifact_class": "nonempty",
            "dataset_path": "nonempty",
            "dataset_sha256": "nonempty",
            "license": "nonempty",
            "source": "nonempty",
            "referent_scheme": "nonempty",
        },
        "real_encoder_manifest": {
            "schema": ENCODER_SCHEMA,
            "artifact_class": "nonempty",
            "model_id": "nonempty",
            "weights_path": "nonempty",
            "weights_sha256": "nonempty",
            "inherited_features_path": "nonempty",
            "inherited_features_sha256": "nonempty",
        },
        "matched_compute_receipt": {
            "schema": COMPUTE_PLAN_SCHEMA,
            "matched_compute": True,
            "artifact_class": "nonempty",
            "arm_flops": "nonempty",
            "budget_flops": "nonempty",
            "dataset_sha256": "nonempty",
            "weights_sha256": "nonempty",
            "tolerance": "nonempty",
        },
        "seed_plan_receipt": {
            "schema": SEED_PLAN_SCHEMA,
            "artifact_class": "nonempty",
            "experiment_id": "nonempty",
            "seed_count": "nonempty",
            "seeds": "nonempty",
            "heldout_split_frozen": True,
            "dataset_sha256": "nonempty",
            "margin": "nonempty",
            "compute_tolerance": "nonempty",
            "budget_flops": "nonempty",
        },
    }
    if variant == "f8":
        common["shell_failure_receipt"] = {
            "schema": SHELL_FAILURE_SCHEMA,
            "artifact_class": "nonempty",
            "shell_controls_exhausted": True,
            "receipt_sha256": "nonempty",
            "receipt_path": "nonempty",
            "dataset_sha256": "nonempty",
            "weights_sha256": "nonempty",
        }
    elif variant == "f16":
        common["inherited_baseline_receipt"] = {
            "schema": INHERITED_BASELINE_SCHEMA,
            "artifact_class": "nonempty",
            "baseline_complete": True,
            "receipt_sha256": "nonempty",
            "receipt_path": "nonempty",
            "dataset_sha256": "nonempty",
            "weights_sha256": "nonempty",
        }
    else:
        raise ValueError(f"unknown rewrite variant {variant!r}")
    return common


def _manifest_scope_problems(
    variant: str,
    documents: Mapping[str, dict[str, Any]],
    seeds: Sequence[int],
    scientific: Mapping[str, Any],
) -> tuple[str | None, list[str]]:
    problems: list[str] = []
    scopes = {str(doc.get("artifact_class") or "") for doc in documents.values()}
    if len(scopes) != 1 or not scopes:
        problems.append(f"all evidence documents must share one artifact_class, got {sorted(scopes)}")
        scope = None
    else:
        scope = next(iter(scopes))
        if scope not in {"fixture", "natural"}:
            problems.append("artifact_class must be exactly 'fixture' or 'natural'")

    rights = documents.get("data_rights_manifest", {})
    encoder = documents.get("real_encoder_manifest", {})
    dataset_hash = str(rights.get("dataset_sha256") or "")
    weights_hash = str(encoder.get("weights_sha256") or "")
    if scope == "fixture":
        if rights.get("fixture_only") is not True or rights.get("natural_data") is not False:
            problems.append("fixture data must set fixture_only=true and natural_data=false")
        if encoder.get("weights_real") is not False or encoder.get("feature_cache_real") is not False:
            problems.append("fixture encoder evidence must explicitly mark weights/features as non-real")
    elif scope == "natural":
        if rights.get("fixture_only") is not False or rights.get("natural_data") is not True:
            problems.append("natural data must set fixture_only=false and natural_data=true")
        if rights.get("split_frozen") is not True:
            problems.append("natural data rights manifest must set split_frozen=true")
        if encoder.get("weights_real") is not True or encoder.get("feature_cache_real") is not True:
            problems.append("natural evidence requires real weights and a real inherited-feature cache")
        if not str(encoder.get("training_provenance") or "").strip():
            problems.append("natural encoder evidence requires nonempty training_provenance")

    compute = documents.get("matched_compute_receipt", {})
    expected_arms = F8_ARMS if variant == "f8" else F16_ARMS
    arm_flops = compute.get("arm_flops")
    if not isinstance(arm_flops, dict) or set(arm_flops) != set(expected_arms):
        problems.append(f"matched compute arm_flops must name exactly {list(expected_arms)}")
    try:
        budget = int(compute.get("budget_flops", 0))
    except (TypeError, ValueError):
        budget = 0
    if budget <= 0:
        problems.append("matched compute budget_flops must be a positive integer")
    if isinstance(arm_flops, dict) and budget > 0:
        for arm, value in arm_flops.items():
            try:
                planned = int(value)
            except (TypeError, ValueError):
                planned = -1
            if planned != budget:
                problems.append(f"matched compute plan for {arm} must equal budget_flops exactly")
    if float(compute.get("tolerance", float("nan"))) != float(scientific["compute_tolerance"]):
        problems.append("matched compute receipt tolerance must equal the configured tolerance")
    for name in (
        "matched_compute_receipt",
        "shell_failure_receipt" if variant == "f8" else "inherited_baseline_receipt",
    ):
        document = documents.get(name, {})
        if document.get("dataset_sha256") != dataset_hash:
            problems.append(f"{name} is not bound to the rights-manifested dataset hash")
        if document.get("weights_sha256") != weights_hash:
            problems.append(f"{name} is not bound to the inherited encoder weight hash")

    plan = documents.get("seed_plan_receipt", {})
    declared = plan.get("seeds")
    if not isinstance(declared, list) or [int(v) for v in declared] != list(seeds):
        problems.append("seed_plan_receipt seeds must exactly match the configured ordered seeds")
    if int(plan.get("seed_count", -1)) != len(seeds):
        problems.append("seed_plan_receipt seed_count must equal the configured seed count")
    expected_id = "f8_plastic_substrate_rewrite" if variant == "f8" else "f16_perfect_slate_null"
    if plan.get("experiment_id") != expected_id:
        problems.append(f"seed_plan_receipt experiment_id must equal {expected_id!r}")
    if plan.get("dataset_sha256") != dataset_hash:
        problems.append("seed_plan_receipt is not bound to the rights-manifested dataset hash")
    if float(plan.get("margin", float("nan"))) != float(scientific["margin"]):
        problems.append("seed_plan_receipt margin must equal the configured preregistered margin")
    if float(plan.get("compute_tolerance", float("nan"))) != float(scientific["compute_tolerance"]):
        problems.append("seed_plan_receipt compute_tolerance must equal the configured compute tolerance")
    if int(plan.get("budget_flops", -1)) != budget:
        problems.append("seed_plan_receipt budget_flops must equal the matched compute plan")
    if variant == "f16" and len(seeds) != 5:
        problems.append("F16 scientific execution requires exactly five preregistered seeds")
    if variant == "f8" and len(seeds) < 5:
        problems.append("F8 scientific execution requires at least five preregistered seeds")
    receipt_name = "shell_failure_receipt" if variant == "f8" else "inherited_baseline_receipt"
    if not _is_sha256(documents.get(receipt_name, {}).get("receipt_sha256")):
        problems.append(f"{receipt_name} receipt_sha256 must be a 64-character hexadecimal digest")
    return scope, problems


def _validated_scientific_config(variant: str, scientific: Mapping[str, Any]) -> list[int]:
    required = {
        "seeds",
        "batch_size",
        "lr",
        "ssl_compute_fraction",
        "variance_weight",
        "larger_shell_width",
        "larger_shell_multiplier",
        "margin",
        "compute_tolerance",
        "max_compute_flops_per_arm_seed",
        "max_package_bytes",
        "max_resident_bytes",
        "max_trainable_params",
        "max_seed_count",
        "max_total_flops",
    }
    missing = sorted(required - set(scientific))
    if missing:
        raise ValueError(f"scientific config is missing required fields {missing}")
    raw_seeds = scientific["seeds"]
    if not isinstance(raw_seeds, list):
        raise ValueError("scientific seeds must be an ordered list")
    seeds = [int(value) for value in raw_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("scientific seeds must be unique")
    if len(seeds) > int(scientific["max_seed_count"]):
        raise ValueError("scientific seed count exceeds the configured safety cap")
    if (variant == "f16" and len(seeds) != 5) or (variant == "f8" and len(seeds) < 5):
        raise ValueError("F8 requires at least five seeds and F16 requires exactly five seeds")
    batch_size = int(scientific["batch_size"])
    lr = float(scientific["lr"])
    ssl_fraction = float(scientific["ssl_compute_fraction"])
    variance_weight = float(scientific["variance_weight"])
    shell_width = int(scientific["larger_shell_width"])
    shell_multiplier = float(scientific["larger_shell_multiplier"])
    margin = float(scientific["margin"])
    tolerance = float(scientific["compute_tolerance"])
    numeric_values = (lr, ssl_fraction, variance_weight, shell_multiplier, margin, tolerance)
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("scientific floating-point hyperparameters must all be finite")
    if batch_size < 2 or lr <= 0 or not 0.1 <= ssl_fraction <= 0.9 or variance_weight < 0:
        raise ValueError("invalid batch_size, lr, ssl_compute_fraction, or variance_weight")
    if shell_width < 0 or shell_multiplier <= 1.0 or not 0.0 <= margin <= 1.0:
        raise ValueError("invalid larger-shell width/multiplier or null margin")
    if not 0.0 <= tolerance <= 0.05:
        raise ValueError("compute_tolerance must be in [0, 0.05]")
    for name in (
        "max_compute_flops_per_arm_seed",
        "max_package_bytes",
        "max_resident_bytes",
        "max_trainable_params",
        "max_seed_count",
        "max_total_flops",
    ):
        if int(scientific[name]) <= 0:
            raise ValueError(f"scientific safety cap {name} must be positive")
    return seeds


def _load_manifest_mlp(
    path: Path,
    dims: Sequence[int],
    activation: str,
) -> _ManifestMLP:
    if activation not in {"gelu", "relu", "tanh"}:
        raise ValueError(f"activation must be gelu, relu, or tanh, got {activation!r}")
    model = _ManifestMLP(dims, activation)
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load safe NPZ encoder weights: {exc}") from exc
    with archive:
        expected = {f"weight_{i}" for i in range(len(model.layers))} | {
            f"bias_{i}" for i in range(len(model.layers))
        }
        if set(archive.files) != expected:
            raise ValueError(f"weight archive keys must be exactly {sorted(expected)}")
        for index, raw_layer in enumerate(model.layers):
            layer = cast(nn.Linear, raw_layer)
            weight = np.asarray(archive[f"weight_{index}"])
            bias = np.asarray(archive[f"bias_{index}"])
            if tuple(weight.shape) != tuple(layer.weight.shape):
                raise ValueError(
                    f"weight_{index} shape {weight.shape} != declared {tuple(layer.weight.shape)}"
                )
            if tuple(bias.shape) != tuple(layer.bias.shape):
                raise ValueError(f"bias_{index} shape {bias.shape} != declared {tuple(layer.bias.shape)}")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError(f"encoder layer {index} contains non-finite weights")
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight).to(torch.float32))
                layer.bias.copy_(torch.from_numpy(bias).to(torch.float32))
    return model


def _validate_and_load_package(
    variant: str,
    documents: dict[str, dict[str, Any]],
    checks: dict[str, dict[str, Any]],
    seeds: Sequence[int],
    scientific: Mapping[str, Any],
    max_package_bytes: int,
    max_resident_bytes: int,
) -> RewritePackage:
    scope, problems = _manifest_scope_problems(variant, documents, seeds, scientific)
    rights = documents["data_rights_manifest"]
    encoder_doc = documents["real_encoder_manifest"]
    rights_path = Path(checks["data_rights_manifest"]["path"])
    encoder_path = Path(checks["real_encoder_manifest"]["path"])
    prerequisite_name = "shell_failure_receipt" if variant == "f8" else "inherited_baseline_receipt"
    prerequisite_doc = documents[prerequisite_name]
    prerequisite_doc_path = Path(checks[prerequisite_name]["path"])
    dataset_path = _resolved_artifact(rights_path, rights.get("dataset_path"))
    weights_path = _resolved_artifact(encoder_path, encoder_doc.get("weights_path"))
    features_path = _resolved_artifact(encoder_path, encoder_doc.get("inherited_features_path"))
    prerequisite_path = _resolved_artifact(prerequisite_doc_path, prerequisite_doc.get("receipt_path"))
    artifacts = {
        "dataset": (dataset_path, str(rights.get("dataset_sha256") or "")),
        "weights": (weights_path, str(encoder_doc.get("weights_sha256") or "")),
        "inherited_features": (
            features_path,
            str(encoder_doc.get("inherited_features_sha256") or ""),
        ),
        "prerequisite_receipt": (
            prerequisite_path,
            str(prerequisite_doc.get("receipt_sha256") or ""),
        ),
    }
    artifact_bytes = 0
    for role, (path, expected_hash) in artifacts.items():
        if not path.is_file():
            problems.append(f"{role} artifact does not exist: {path}")
            continue
        artifact_bytes += path.stat().st_size
        actual_hash = sha256_file(path)
        if expected_hash.lower() != actual_hash:
            problems.append(f"{role} sha256 does not match the manifest")
    if artifact_bytes > max_package_bytes:
        problems.append(
            f"evidence package is {artifact_bytes} bytes, above configured safety cap {max_package_bytes}"
        )
    if all(path.is_file() for path, _expected in artifacts.values()):
        estimated_resident = (
            _npz_uncompressed_bytes(dataset_path)
            + _npz_uncompressed_bytes(weights_path)
            + features_path.stat().st_size
        )
        if estimated_resident > max_resident_bytes:
            problems.append(
                f"uncompressed tensors require at least {estimated_resident} bytes, above configured "
                f"resident-memory safety cap {max_resident_bytes}"
            )
    if problems:
        raise ValueError("; ".join(problems))

    try:
        prerequisite_source = json.loads(prerequisite_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load prerequisite source receipt: {exc}") from exc
    if not isinstance(prerequisite_source, dict):
        raise ValueError("prerequisite source receipt root must be a mapping")
    expected_source_schema = (
        "mop-shell-control-result/v1" if variant == "f8" else "mop-inherited-baseline-result/v1"
    )
    expected_source_flag = "shell_controls_exhausted" if variant == "f8" else "baseline_complete"
    expected_experiment_id = "f8_plastic_substrate_rewrite" if variant == "f8" else "f16_perfect_slate_null"
    source_problems = []
    if prerequisite_source.get("schema") != expected_source_schema:
        source_problems.append(f"source schema must equal {expected_source_schema!r}")
    if prerequisite_source.get("experiment_id") != expected_experiment_id:
        source_problems.append(f"source experiment_id must equal {expected_experiment_id!r}")
    if prerequisite_source.get(expected_source_flag) is not True:
        source_problems.append(f"source {expected_source_flag} must be true")
    if prerequisite_source.get("dataset_sha256") != rights.get("dataset_sha256"):
        source_problems.append("source receipt dataset hash does not match")
    if prerequisite_source.get("weights_sha256") != encoder_doc.get("weights_sha256"):
        source_problems.append("source receipt weight hash does not match")
    if source_problems:
        raise ValueError("invalid prerequisite source receipt: " + "; ".join(source_problems))

    try:
        archive = np.load(dataset_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load safe NPZ dataset package: {exc}") from exc
    required_arrays = {
        "inputs",
        "view_a",
        "view_b",
        "factor_labels",
        "transfer_labels",
        "domain_labels",
        "split",
        "referent_ids",
        "view_a_referent_ids",
        "view_b_referent_ids",
    }
    with archive:
        if set(archive.files) != required_arrays:
            raise ValueError(f"dataset arrays must be exactly {sorted(required_arrays)}")
        arrays = {name: np.asarray(archive[name]) for name in required_arrays}
    inputs = arrays["inputs"]
    view_a, view_b = arrays["view_a"], arrays["view_b"]
    labels = arrays["factor_labels"]
    transfer = arrays["transfer_labels"]
    domains = arrays["domain_labels"]
    split = arrays["split"]
    referents = arrays["referent_ids"]
    view_a_referents = arrays["view_a_referent_ids"]
    view_b_referents = arrays["view_b_referent_ids"]
    if inputs.ndim != 2 or inputs.shape[0] < 12:
        raise ValueError("inputs must be a two-dimensional tensor with at least 12 referents")
    n = inputs.shape[0]
    if view_a.shape != inputs.shape or view_b.shape != inputs.shape:
        raise ValueError("view_a and view_b must have the exact input shape")
    for name, value in (
        ("factor_labels", labels),
        ("transfer_labels", transfer),
        ("domain_labels", domains),
        ("split", split),
    ):
        if value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"{name} must be a one-dimensional array with {n} rows")
    if referents.ndim != 1 or referents.shape[0] != n:
        raise ValueError(f"referent_ids must be a one-dimensional array with {n} rows")
    referent_ids = [str(v) for v in referents.tolist()]
    if len(set(referent_ids)) != n or any(not value.strip() for value in referent_ids):
        raise ValueError("referent_ids must be nonempty and globally unique")
    for name, value in (
        ("view_a_referent_ids", view_a_referents),
        ("view_b_referent_ids", view_b_referents),
    ):
        if value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"{name} must be one-dimensional with {n} rows")
        if [str(item) for item in value.tolist()] != referent_ids:
            raise ValueError(f"{name} must exactly equal referent_ids in row order")
    if not np.isfinite(inputs).all() or not np.isfinite(view_a).all() or not np.isfinite(view_b).all():
        raise ValueError("dataset inputs and views must be finite")
    split_ints = np.asarray(split).astype(np.int64)
    if not np.array_equal(split_ints, split):
        raise ValueError("split values must be exact integers, fractional values cannot be truncated")
    if set(split_ints.tolist()) != {0, 1, 2}:
        raise ValueError("split must contain frozen train=0, validation=1, and test=2 rows")
    train_rows = {np.ascontiguousarray(row).tobytes() for row in inputs[split_ints == 0]}
    test_rows = {np.ascontiguousarray(row).tobytes() for row in inputs[split_ints == 2]}
    if train_rows & test_rows:
        raise ValueError("exact duplicate input payloads cannot cross the train/test boundary")
    domain_ints = np.asarray(domains).astype(np.int64)
    if (domain_ints < 0).any() or not np.array_equal(domain_ints, domains):
        raise ValueError("domain_labels must contain nonnegative integers")
    train_domains = set(domain_ints[split_ints == 0].tolist())
    test_domains = set(domain_ints[split_ints == 2].tolist())
    if not train_domains or not test_domains or train_domains & test_domains:
        raise ValueError("test domains must be nonempty and disjoint from all training domains")
    for name, value in (("factor_labels", labels), ("transfer_labels", transfer)):
        ints = np.asarray(value).astype(np.int64)
        if (ints < 0).any() or not np.array_equal(ints, value):
            raise ValueError(f"{name} must contain nonnegative integer class labels")
        train_classes = set(ints[split_ints == 0].tolist())
        test_classes = set(ints[split_ints == 2].tolist())
        if len(train_classes) < 2 or not test_classes.issubset(train_classes):
            raise ValueError(f"{name} test classes must be represented in train, with at least two classes")
        if train_classes != set(range(max(train_classes) + 1)):
            raise ValueError(f"{name} train classes must be contiguous integers beginning at zero")

    if encoder_doc.get("weights_format") != WEIGHTS_FORMAT:
        raise ValueError(f"weights_format must be {WEIGHTS_FORMAT!r}")
    dims_raw = encoder_doc.get("architecture_dims")
    if not isinstance(dims_raw, list) or len(dims_raw) < 2:
        raise ValueError("encoder architecture_dims must be a list with at least two positive widths")
    dims = [int(v) for v in dims_raw]
    if any(value <= 0 for value in dims) or dims[0] != inputs.shape[1]:
        raise ValueError("encoder architecture_dims must be positive and begin with the input width")
    activation = str(encoder_doc.get("activation") or "")
    inherited_encoder = _load_manifest_mlp(weights_path, dims, activation)
    try:
        inherited_np = np.load(features_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load inherited feature NPY: {exc}") from exc
    inherited_np = np.asarray(inherited_np)
    if inherited_np.shape != (n, dims[-1]) or not np.isfinite(inherited_np).all():
        raise ValueError(f"inherited features must be finite with shape {(n, dims[-1])}")
    with torch.no_grad():
        recomputed = inherited_encoder(torch.from_numpy(inputs).to(torch.float32))
    tolerance = float(encoder_doc.get("feature_tolerance", 1.0e-5))
    if tolerance <= 0 or tolerance > 1.0e-2:
        raise ValueError("feature_tolerance must be in (0, 1e-2]")
    max_error = float((recomputed - torch.from_numpy(inherited_np).float()).abs().max())
    if max_error > tolerance:
        raise ValueError(
            f"inherited feature receipt is not reproducible from the checkpoint, max error "
            f"{max_error:.6g} > {tolerance:.6g}"
        )

    evidence_hashes = {
        name: str(check.get("sha256")) for name, check in checks.items() if check.get("sha256")
    }
    evidence_hashes.update(
        {
            "dataset": sha256_file(dataset_path),
            "weights": sha256_file(weights_path),
            "inherited_features": sha256_file(features_path),
            "prerequisite_receipt": sha256_file(prerequisite_path),
        }
    )
    resident_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            torch.from_numpy(inputs),
            torch.from_numpy(view_a),
            torch.from_numpy(view_b),
            torch.from_numpy(np.asarray(labels)),
            torch.from_numpy(np.asarray(transfer)),
            torch.from_numpy(domain_ints),
            torch.from_numpy(np.asarray(split)),
            torch.from_numpy(inherited_np),
        )
    ) + sum(parameter.numel() * parameter.element_size() for parameter in inherited_encoder.parameters())
    return RewritePackage(
        inputs=torch.from_numpy(inputs).to(torch.float32),
        view_a=torch.from_numpy(view_a).to(torch.float32),
        view_b=torch.from_numpy(view_b).to(torch.float32),
        factor_labels=torch.from_numpy(np.asarray(labels).astype(np.int64)),
        transfer_labels=torch.from_numpy(np.asarray(transfer).astype(np.int64)),
        domain_labels=torch.from_numpy(domain_ints),
        split=torch.from_numpy(split_ints),
        referent_ids=referent_ids,
        inherited_features=torch.from_numpy(inherited_np).to(torch.float32),
        inherited_encoder=inherited_encoder,
        encoder_dims=dims,
        activation=activation,
        evidence_scope=str(scope),
        evidence_documents=documents,
        evidence_hashes=evidence_hashes,
        artifact_bytes=artifact_bytes + sum(Path(check["path"]).stat().st_size for check in checks.values()),
        resident_tensor_bytes=resident_tensor_bytes,
    )


class _IndexStream:
    def __init__(self, indices: torch.Tensor, seed: int):
        self.indices = indices.detach().cpu().long()
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = torch.empty(0, dtype=torch.long)

    def take(self, count: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = count
        while remaining > 0:
            if len(self.pending) == 0:
                order = torch.randperm(len(self.indices), generator=self.generator)
                self.pending = self.indices[order]
            take = min(remaining, len(self.pending))
            chunks.append(self.pending[:take])
            self.pending = self.pending[take:]
            remaining -= take
        return torch.cat(chunks)


def _train_to_budget(
    *,
    optimizer: torch.optim.Optimizer,
    loss_for_indices: Callable[[torch.Tensor], torch.Tensor],
    stream: _IndexStream,
    per_sample_flops: int,
    per_update_flops: int,
    budget_flops: int,
    batch_size: int,
    minimum_batch: int = 1,
) -> tuple[int, int, int]:
    if per_sample_flops <= 0 or budget_flops <= 0:
        return 0, 0, 0
    spent = updates = examples = 0
    while budget_flops - spent >= per_update_flops + per_sample_flops * minimum_batch:
        affordable = (budget_flops - spent - per_update_flops) // per_sample_flops
        batch = int(min(batch_size, affordable))
        if batch < minimum_batch:
            break
        indices = stream.take(batch)
        optimizer.zero_grad()
        loss = loss_for_indices(indices)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        spent += per_update_flops + batch * per_sample_flops
        updates += 1
        examples += batch
    return spent, updates, examples


def _classification_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def _new_encoder(dims: Sequence[int], activation: str, seed: int) -> _ManifestMLP:
    seed_everything(seed)
    return _ManifestMLP(dims, activation)


def _ssl_then_probe(
    *,
    encoder: nn.Module,
    package: RewritePackage,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    classes: int,
    budget_flops: int,
    ssl_fraction: float,
    batch_size: int,
    lr: float,
    variance_weight: float,
    measure_rewrite: bool,
    seed: int,
    device: torch.device,
) -> ArmResult:
    seed_everything(seed)
    encoder = encoder.to(device)
    views_a = package.view_a.to(device)
    views_b = package.view_b.to(device)
    x = package.inputs.to(device)
    y = labels.to(device)
    output_dim = package.encoder_dims[-1]
    encoder_forward = _mlp_forward_flops(package.encoder_dims, package.activation)
    ssl_per_sample = _ssl_train_flops(package.encoder_dims, package.activation)
    probe_forward = _mlp_forward_flops([output_dim, classes])
    evaluation_rows = len(val_idx) + len(test_idx)
    fixed_inference_flops = encoder_forward * len(package.inputs) + probe_forward * evaluation_rows
    if measure_rewrite:
        fixed_inference_flops += encoder_forward * len(package.inputs)
    training_budget = budget_flops - fixed_inference_flops
    if training_budget <= 0:
        raise ValueError("matched compute budget is too small for candidate inference and evaluation")
    ssl_budget = int(training_budget * ssl_fraction)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    ssl_stream = _IndexStream(train_idx, seed + 11)

    def ssl_loss(indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(device)
        z1, z2 = encoder(views_a[idx]), encoder(views_b[idx])
        invariance = F.mse_loss(z1, z2)
        std1 = torch.sqrt(z1.var(0, unbiased=False) + 1.0e-4)
        std2 = torch.sqrt(z2.var(0, unbiased=False) + 1.0e-4)
        variance = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
        return invariance + variance_weight * variance

    ssl_flops, ssl_updates, ssl_examples = _train_to_budget(
        optimizer=optimizer,
        loss_for_indices=ssl_loss,
        stream=ssl_stream,
        per_sample_flops=ssl_per_sample,
        per_update_flops=_adam_update_flops(encoder),
        budget_flops=ssl_budget,
        batch_size=batch_size,
        minimum_batch=2,
    )
    encoder.eval()
    with torch.no_grad():
        representations = encoder(x).detach()
    head = nn.Linear(output_dim, classes).to(device)
    probe_per_sample = _classification_train_flops([output_dim, classes], classes)
    probe_budget = training_budget - ssl_flops
    probe_opt = torch.optim.Adam(head.parameters(), lr=lr)
    probe_stream = _IndexStream(train_idx, seed + 31)

    def probe_loss(indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(device)
        return F.cross_entropy(head(representations[idx]), y[idx])

    probe_flops, probe_updates, probe_examples = _train_to_budget(
        optimizer=probe_opt,
        loss_for_indices=probe_loss,
        stream=probe_stream,
        per_sample_flops=probe_per_sample,
        per_update_flops=_adam_update_flops(head),
        budget_flops=probe_budget,
        batch_size=batch_size,
    )
    shift = 0.0
    if measure_rewrite:
        inherited = package.inherited_features.to(device)[test_idx.to(device)]
        rewritten = representations[test_idx.to(device)]
        shift = 1.0 - float(F.cosine_similarity(inherited, rewritten, dim=1).mean())
    return ArmResult(
        accuracy=_classification_accuracy(head, representations[test_idx.to(device)], y[test_idx.to(device)]),
        validation_accuracy=_classification_accuracy(
            head, representations[val_idx.to(device)], y[val_idx.to(device)]
        ),
        flops=ssl_flops + probe_flops + fixed_inference_flops,
        updates=ssl_updates + probe_updates,
        examples=ssl_examples + probe_examples,
        trainable_params=param_count(encoder) + param_count(head),
        representation_shift=shift,
    )


def _cached_feature_head(
    *,
    package: RewritePackage,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    classes: int,
    budget_flops: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    shell_width: int | None,
) -> ArmResult:
    seed_everything(seed)
    features = package.inherited_features.to(device)
    y = labels.to(device)
    dim = features.shape[1]
    dims = [dim, classes] if shell_width is None else [dim, shell_width, classes]
    head_activation = "none" if shell_width is None else "gelu"
    if shell_width is None:
        head: nn.Module = nn.Linear(dim, classes)
    else:
        head = nn.Sequential(nn.Linear(dim, shell_width), nn.GELU(), nn.Linear(shell_width, classes))
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    stream = _IndexStream(train_idx, seed + 41)

    def loss_for_indices(indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(device)
        return F.cross_entropy(head(features[idx]), y[idx])

    per_sample = _classification_train_flops(dims, classes, head_activation)
    inherited_production = _mlp_forward_flops(package.encoder_dims, package.activation) * len(package.inputs)
    fixed_inference_flops = inherited_production + _mlp_forward_flops(dims, head_activation) * (
        len(val_idx) + len(test_idx)
    )
    training_budget = budget_flops - fixed_inference_flops
    if training_budget <= 0:
        raise ValueError("matched compute budget is too small for frozen-shell evaluation")
    spent, updates, examples = _train_to_budget(
        optimizer=optimizer,
        loss_for_indices=loss_for_indices,
        stream=stream,
        per_sample_flops=per_sample,
        per_update_flops=_adam_update_flops(head),
        budget_flops=training_budget,
        batch_size=batch_size,
    )
    return ArmResult(
        accuracy=_classification_accuracy(head, features[test_idx.to(device)], y[test_idx.to(device)]),
        validation_accuracy=_classification_accuracy(
            head, features[val_idx.to(device)], y[val_idx.to(device)]
        ),
        flops=spent + fixed_inference_flops,
        updates=updates,
        examples=examples,
        trainable_params=param_count(head),
    )


def _frozen_random_encoder_head(
    *,
    encoder: nn.Module,
    package: RewritePackage,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    classes: int,
    budget_flops: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> ArmResult:
    seed_everything(seed)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    head = nn.Linear(package.encoder_dims[-1], classes).to(device)
    x, y = package.inputs.to(device), labels.to(device)
    with torch.no_grad():
        features = encoder(x).detach()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    stream = _IndexStream(train_idx, seed + 31)

    def loss_for_indices(indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(device)
        return F.cross_entropy(head(features[idx]), y[idx])

    encoder_cost = _mlp_forward_flops(package.encoder_dims, package.activation)
    head_cost = _mlp_forward_flops([package.encoder_dims[-1], classes])
    fixed_inference_flops = encoder_cost * len(x) + head_cost * (len(val_idx) + len(test_idx))
    training_budget = budget_flops - fixed_inference_flops
    if training_budget <= 0:
        raise ValueError("matched compute budget is too small for random-encoder evaluation")
    spent, updates, examples = _train_to_budget(
        optimizer=optimizer,
        loss_for_indices=loss_for_indices,
        stream=stream,
        per_sample_flops=_classification_train_flops([package.encoder_dims[-1], classes], classes),
        per_update_flops=_adam_update_flops(head),
        budget_flops=training_budget,
        batch_size=batch_size,
    )
    return ArmResult(
        accuracy=_classification_accuracy(head, features[test_idx.to(device)], y[test_idx.to(device)]),
        validation_accuracy=_classification_accuracy(
            head, features[val_idx.to(device)], y[val_idx.to(device)]
        ),
        flops=spent + fixed_inference_flops,
        updates=updates,
        examples=examples,
        trainable_params=param_count(head),
    )


def _run_one_seed(
    variant: str,
    package: RewritePackage,
    scientific: Mapping[str, Any],
    seed: int,
    device: torch.device,
    budget_flops: int,
    progress_callback: Callable[[str, ArmResult], None] | None = None,
) -> dict[str, ArmResult]:
    train_idx = (package.split == 0).nonzero().flatten()
    val_idx = (package.split == 1).nonzero().flatten()
    test_idx = (package.split == 2).nonzero().flatten()
    labels = package.factor_labels if variant == "f8" else package.transfer_labels
    classes = int(labels.max()) + 1
    batch_size = int(scientific["batch_size"])
    lr = float(scientific["lr"])
    ssl_fraction = float(scientific["ssl_compute_fraction"])
    variance_weight = float(scientific["variance_weight"])
    configured_shell_width = int(scientific["larger_shell_width"])
    shell_multiplier = float(scientific["larger_shell_multiplier"])
    shell_width = (
        configured_shell_width
        if configured_shell_width > 0
        else int(math.ceil(package.encoder_dims[-1] * shell_multiplier))
    )
    if (
        batch_size < 2
        or not 0.1 <= ssl_fraction <= 0.9
        or shell_multiplier <= 1.0
        or shell_width <= package.encoder_dims[-1]
    ):
        raise ValueError(
            "scientific batch_size must be >=2, ssl_compute_fraction in [0.1,0.9], and "
            "the configured or derived larger shell width must exceed inherited feature width"
        )
    encoder_params = sum(
        (package.encoder_dims[index] + 1) * package.encoder_dims[index + 1]
        for index in range(len(package.encoder_dims) - 1)
    )
    candidate_params = encoder_params + (package.encoder_dims[-1] + 1) * classes
    larger_shell_params = (package.encoder_dims[-1] + 1) * shell_width + (shell_width + 1) * classes
    max_params = max(candidate_params, larger_shell_params)
    if max_params > int(scientific["max_trainable_params"]):
        raise ValueError(
            f"largest arm has {max_params} trainable parameters, above configured safety cap "
            f"{int(scientific['max_trainable_params'])}"
        )
    evaluation_rows = len(val_idx) + len(test_idx)
    encoder_forward = _mlp_forward_flops(package.encoder_dims, package.activation)
    probe_forward = _mlp_forward_flops([package.encoder_dims[-1], classes])
    shell_forward = _mlp_forward_flops([package.encoder_dims[-1], shell_width, classes], "gelu")
    inherited_production = encoder_forward * len(package.inputs)
    candidate_fixed = encoder_forward * len(package.inputs) + probe_forward * evaluation_rows
    if variant == "f8":
        candidate_fixed += inherited_production
    fixed_costs = {
        "candidate": candidate_fixed,
        "frozen_inherited": inherited_production + probe_forward * evaluation_rows,
        "larger_frozen_shell": inherited_production + shell_forward * evaluation_rows,
        "random_init_same_arch": encoder_forward * len(package.inputs) + probe_forward * evaluation_rows,
    }
    if budget_flops <= max(fixed_costs.values()):
        raise ValueError(
            f"matched compute budget {budget_flops} is too small for mandatory inference costs {fixed_costs}"
        )
    candidate_training = budget_flops - fixed_costs["candidate"]
    encoder_update = ADAM_FLOPS_PER_PARAMETER_UPDATE * encoder_params
    probe_params = (package.encoder_dims[-1] + 1) * classes
    probe_update = ADAM_FLOPS_PER_PARAMETER_UPDATE * probe_params
    if int(candidate_training * ssl_fraction) < encoder_update + 2 * _ssl_train_flops(
        package.encoder_dims, package.activation
    ):
        raise ValueError("matched compute budget cannot fund one two-sample SSL update")
    if candidate_training - int(candidate_training * ssl_fraction) < probe_update + (
        _classification_train_flops([package.encoder_dims[-1], classes], classes)
    ):
        raise ValueError("matched compute budget cannot fund one probe update")

    inherited = copy.deepcopy(package.inherited_encoder)
    candidate_name = "plastic_rewrite" if variant == "f8" else "blank_slate"
    if variant == "f8":
        candidate_encoder = inherited
        frozen_random_control_encoder = None
    else:
        blank_initialization = _new_encoder(package.encoder_dims, package.activation, seed + 701)
        candidate_encoder = copy.deepcopy(blank_initialization)
        frozen_random_control_encoder = copy.deepcopy(blank_initialization)
    candidate_initialization_fingerprint = _module_fingerprint(candidate_encoder)
    results: dict[str, ArmResult] = {}

    def record(name: str, result: ArmResult) -> None:
        results[name] = result
        if progress_callback is not None:
            progress_callback(name, result)

    candidate_result = _ssl_then_probe(
        encoder=candidate_encoder,
        package=package,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        classes=classes,
        budget_flops=budget_flops,
        ssl_fraction=ssl_fraction,
        batch_size=batch_size,
        lr=lr,
        variance_weight=variance_weight,
        measure_rewrite=variant == "f8",
        seed=seed,
        device=device,
    )
    candidate_result.initialization_fingerprint = candidate_initialization_fingerprint
    record(candidate_name, candidate_result)
    record(
        "frozen_inherited",
        _cached_feature_head(
            package=package,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            classes=classes,
            budget_flops=budget_flops,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 101,
            device=device,
            shell_width=None,
        ),
    )
    record(
        "larger_frozen_shell",
        _cached_feature_head(
            package=package,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            classes=classes,
            budget_flops=budget_flops,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 201,
            device=device,
            shell_width=shell_width,
        ),
    )
    if variant == "f8":
        random_encoder = _new_encoder(package.encoder_dims, package.activation, seed + 901)
        random_fingerprint = _module_fingerprint(random_encoder)
        random_result = _ssl_then_probe(
            encoder=random_encoder,
            package=package,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            classes=classes,
            budget_flops=budget_flops,
            ssl_fraction=ssl_fraction,
            batch_size=batch_size,
            lr=lr,
            variance_weight=variance_weight,
            measure_rewrite=False,
            seed=seed,
            device=device,
        )
        random_result.initialization_fingerprint = random_fingerprint
        record(
            "random_init_same_arch",
            random_result,
        )
    else:
        assert frozen_random_control_encoder is not None
        random_result = _frozen_random_encoder_head(
            encoder=frozen_random_control_encoder,
            package=package,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            classes=classes,
            budget_flops=budget_flops,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )
        random_result.initialization_fingerprint = candidate_initialization_fingerprint
        record(
            "random_init_same_arch",
            random_result,
        )
    return results


def _resource_projection(
    *,
    experiment_id: str,
    device: DeviceInfo,
    package_bytes: int,
    budget_flops: int | None,
    arm_count: int,
    seed_count: int,
    estimated_peak_bytes: int,
) -> dict[str, Any]:
    host = apple_silicon_info()
    profiles: dict[str, Any] = {}
    for name in ("m3pro-local-max", "studio-1tb", "studio-m1ultra"):
        profile = get_profile(name)
        compatible, problems, measured = profile.host_compatibility(host=host)
        disk_ok, free_gb = profile.free_disk_ok()
        memory_floor_bytes = int(profile.min_host_unified_memory_gb * 1e9)
        profiles[name] = {
            "procurement_status": profile.procurement_status,
            "current_host_compatible": compatible,
            "current_host_problems": problems,
            "current_host_measured": measured,
            "current_free_disk_gb": round(free_gb, 3),
            "profile_free_disk_policy_satisfied": disk_ok,
            "estimated_peak_fits_profile_memory_floor": estimated_peak_bytes <= memory_floor_bytes,
            "package_fits_profile_usable_disk": package_bytes <= int(profile.usable_gb * 1e9),
            "profile_memory_floor_gb": profile.min_host_unified_memory_gb,
            "profile_usable_disk_gb": profile.usable_gb,
            "projection_only": True,
        }
    total_flops = None if budget_flops is None else budget_flops * arm_count * seed_count
    return {
        "schema": PROJECTION_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at": _utc_now(),
        "resolved_execution_device": device.kind,
        "measured_host": host,
        "package_bytes": int(package_bytes),
        "per_arm_per_seed_budget_flops": budget_flops,
        "projected_total_training_flops": total_flops,
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "profiles": profiles,
        "wall_time_projection": None,
        "measured_hardware_wall": False,
        "boundary_conclusion": "none, this receipt is a resource projection and attempt record",
    }


def _scientific_result(
    *,
    variant: str,
    experiment_id: str,
    metric_names: Sequence[str],
    package: RewritePackage,
    scientific: Mapping[str, Any],
    device: DeviceInfo,
    projection_path: Path,
    progress_path: Path,
    resource_meter: _AttemptResourceMeter,
) -> dict[str, Any]:
    seeds = [int(value) for value in scientific["seeds"]]
    compute_doc = package.evidence_documents["matched_compute_receipt"]
    budget_flops = int(compute_doc["budget_flops"])
    planned_total_flops = budget_flops * 4 * len(seeds)
    if planned_total_flops > int(scientific["max_total_flops"]):
        raise ScientificExecutionRefused(
            f"{experiment_id} planned {planned_total_flops} estimated FLOPs, above configured total "
            f"safety cap {int(scientific['max_total_flops'])}"
        )
    max_compute = int(scientific["max_compute_flops_per_arm_seed"])
    if budget_flops > max_compute:
        raise ScientificExecutionRefused(
            f"{experiment_id} requested {budget_flops} FLOPs per arm/seed, above configured safety cap "
            f"{max_compute}; raise the explicit cap only after resource review"
        )
    tolerance = min(
        float(scientific["compute_tolerance"]),
        float(compute_doc.get("tolerance", scientific["compute_tolerance"])),
    )
    if not 0.0 <= tolerance <= 0.05:
        raise ScientificExecutionRefused("compute matching tolerance must be in [0, 0.05]")
    expected_arms = F8_ARMS if variant == "f8" else F16_ARMS
    candidate = expected_arms[0]
    arm_runs: dict[str, list[ArmResult]] = {name: [] for name in expected_arms}
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "schema": "mop-rewrite-progress/v1",
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": _utc_now(),
        "planned_seeds": seeds,
        "completed_arms": [],
        "completed_seed_count": 0,
        "consumed_estimated_flops": 0,
    }
    _write_json(progress_path, progress)
    for seed in seeds:
        progress["current_seed"] = seed

        def record_progress(arm: str, result: ArmResult, current_seed: int = seed) -> None:
            completed = cast(list[dict[str, Any]], progress["completed_arms"])
            completed.append(
                {
                    "seed": current_seed,
                    "arm": arm,
                    "estimated_flops": result.flops,
                    "updates": result.updates,
                    "examples": result.examples,
                }
            )
            progress["consumed_estimated_flops"] = int(progress["consumed_estimated_flops"]) + result.flops
            progress["last_updated_at"] = _utc_now()
            resource_meter.sample()
            progress["resource_sample"] = resource_meter.snapshot()
            _write_json(progress_path, progress)

        results = _run_one_seed(
            variant,
            package,
            scientific,
            seed,
            device.device,
            budget_flops,
            progress_callback=record_progress,
        )
        if set(results) != set(expected_arms):
            raise RuntimeError("scientific engine produced an incomplete arm set")
        for arm, result in results.items():
            arm_runs[arm].append(result)
        progress["completed_seed_count"] = int(progress["completed_seed_count"]) + 1
        _write_json(progress_path, progress)
    seconds = time.perf_counter() - started
    resource_sample = resource_meter.snapshot()
    peak_rss = int(resource_sample["rss_peak_sampled_bytes"])

    all_flops = [run.flops for values in arm_runs.values() for run in values]
    lo, hi = min(all_flops), max(all_flops)
    compute_spread = (hi - lo) / max(hi, 1)
    budget_shortfall = max((budget_flops - value) / budget_flops for value in all_flops)
    compute_matched = compute_spread <= tolerance and budget_shortfall <= tolerance
    if not compute_matched:
        raise ScientificExecutionRefused(
            f"{experiment_id} estimated total compute failed closed: spread={compute_spread:.6f}, "
            f"budget_shortfall={budget_shortfall:.6f}, tolerance={tolerance:.6f}"
        )

    accuracy_by_arm = {
        name: _mean([result.accuracy for result in values]) for name, values in arm_runs.items()
    }
    validation_by_arm = {
        name: _mean([result.validation_accuracy for result in values]) for name, values in arm_runs.items()
    }
    strongest_control = max(expected_arms[1:], key=lambda name: accuracy_by_arm[name])
    deltas = [
        arm_runs[candidate][index].accuracy
        - max(arm_runs[name][index].accuracy for name in expected_arms[1:])
        for index in range(len(seeds))
    ]
    inherited_deltas = [
        arm_runs[candidate][index].accuracy - arm_runs["frozen_inherited"][index].accuracy
        for index in range(len(seeds))
    ]
    margin = float(scientific["margin"])
    null_decision = _margin_null_decision(deltas, margin, compute_matched)
    margin_adjusted_deltas = cast(list[float], null_decision["margin_adjusted_deltas"])
    raw_ci = cast(dict[str, Any], null_decision["raw_seed_ci"])
    ci = cast(dict[str, Any], null_decision["seed_ci"])
    flips = cast(dict[str, Any], null_decision["sign_flip"])
    rejects = bool(null_decision["rejects"])
    natural_evidence_declared = package.evidence_scope == "natural"
    natural_claim_eligible = False
    promotion_eligible = False
    primary_value = accuracy_by_arm[candidate]
    vs_inherited = _mean(inherited_deltas)
    vs_strongest = _mean(deltas)
    representation_shift = _mean([result.representation_shift for result in arm_runs[candidate]])
    if variant == "f8":
        metrics = {
            metric_names[0]: primary_value,
            metric_names[1]: representation_shift,
            metric_names[2]: vs_strongest,
        }
    else:
        metrics = {
            metric_names[0]: vs_inherited,
            metric_names[1]: primary_value,
            metric_names[2]: vs_strongest,
        }
    total_flops = sum(all_flops)
    total_updates = sum(result.updates for values in arm_runs.values() for result in values)
    max_params = max(result.trainable_params for values in arm_runs.values() for result in values)
    projection = _resource_projection(
        experiment_id=experiment_id,
        device=device,
        package_bytes=package.artifact_bytes,
        budget_flops=budget_flops,
        arm_count=len(expected_arms),
        seed_count=len(seeds),
        estimated_peak_bytes=package.resident_tensor_bytes * 2 + max_params * 16,
    )
    projection["actual_attempt"] = {
        "wall_seconds": seconds,
        "resource_sample": resource_sample,
        "estimated_total_training_flops": total_flops,
        "completed": True,
    }
    _write_json(projection_path, projection)
    progress["status"] = "completed"
    progress["completed_at"] = _utc_now()
    progress["resource_sample"] = resource_sample
    _write_json(progress_path, progress)
    return {
        **{name: round(value, 6) for name, value in metrics.items()},
        "execution_status": "scientific-engine-complete",
        "scientific_result": True,
        "evidence_scope": package.evidence_scope,
        "natural_evidence_declared": natural_evidence_declared,
        "natural_claim_eligible": natural_claim_eligible,
        "promotion_eligible": promotion_eligible,
        "external_provenance_authority_required": True,
        "fixture_taint_irreversible": package.evidence_scope == "fixture",
        "null_evaluated": True,
        "null_supported": not rejects,
        "decision_rule": {
            "margin": margin,
            "mean_per_seed_strongest_control_effect_above_margin": _mean(margin_adjusted_deltas) > 0.0,
            "margin_adjusted_seed_ci_lower_above_zero": float(ci["lo"]) > 0.0,
            "no_margin_adjusted_seed_sign_flip": not bool(flips["any_flip"]),
            "estimated_total_compute_matched": compute_matched,
        },
        "candidate_arm": candidate,
        "strongest_control_arm": strongest_control,
        "control_design": {
            "frozen_inherited_encoder": True,
            "larger_shell_uses_frozen_inherited_features": True,
            "random_init_same_architecture": True,
            "blank_and_random_control_share_initial_weights": variant == "f16",
            "f16_random_control_encoder_frozen": variant == "f16",
            "f8_random_control_receives_same_ssl_curriculum": variant == "f8",
        },
        "accuracy_by_arm": {name: round(value, 6) for name, value in accuracy_by_arm.items()},
        "validation_accuracy_by_arm": {name: round(value, 6) for name, value in validation_by_arm.items()},
        "representation_cosine_shift": round(representation_shift, 6),
        "candidate_vs_frozen_inherited_accuracy_delta": round(vs_inherited, 6),
        "seeds": seeds,
        "raw_seed_ci": raw_ci,
        "seed_ci": ci,
        "seed_ci_quantity": "candidate minus per-seed strongest control minus preregistered margin",
        "sign_flip": flips,
        "compute": {
            "schema": "mop-matched-compute-estimate/v2",
            "estimator": (
                "linear multiply-adds, activation forward/backward, SSL and cross-entropy loss, "
                "parameter/input gradients, Adam moment and parameter updates, feature production, "
                "and held-out inference"
            ),
            "adam_flops_per_parameter_update": ADAM_FLOPS_PER_PARAMETER_UPDATE,
            "actual_hardware_instruction_count_measured": False,
            "hardware_time_and_energy_matched": False,
            "target_flops_per_arm_seed": budget_flops,
            "estimated_flops_by_arm_seed": {
                name: [result.flops for result in values] for name, values in arm_runs.items()
            },
            "updates_by_arm_seed": {
                name: [result.updates for result in values] for name, values in arm_runs.items()
            },
            "examples_by_arm_seed": {
                name: [result.examples for result in values] for name, values in arm_runs.items()
            },
            "initialization_fingerprints_by_arm_seed": {
                name: [result.initialization_fingerprint for result in values]
                for name, values in arm_runs.items()
                if any(result.initialization_fingerprint for result in values)
            },
            "relative_spread": compute_spread,
            "maximum_budget_shortfall": budget_shortfall,
            "tolerance": tolerance,
            "matched": compute_matched,
            "matched_quantity": "estimated end-to-end arm FLOPs within the declared convention",
        },
        "evidence_hashes": package.evidence_hashes,
        "split_counts": {
            "train": int((package.split == 0).sum()),
            "validation": int((package.split == 1).sum()),
            "test": int((package.split == 2).sum()),
        },
        "heldout_domains": {
            "train": sorted({int(value) for value in package.domain_labels[package.split == 0].tolist()}),
            "validation": sorted(
                {int(value) for value in package.domain_labels[package.split == 1].tolist()}
            ),
            "test": sorted({int(value) for value in package.domain_labels[package.split == 2].tolist()}),
            "train_test_disjoint": not bool(
                set(package.domain_labels[package.split == 0].tolist())
                & set(package.domain_labels[package.split == 2].tolist())
            ),
        },
        "referent_count": len(package.referent_ids),
        "package_accounting": {
            "artifact_bytes": package.artifact_bytes,
            "resident_tensor_bytes": package.resident_tensor_bytes,
            "peak_projection_formula": "2 * resident_tensor_bytes + 16 * peak_trainable_params",
        },
        "resource_projection": str(projection_path),
        "progress_receipt": str(progress_path),
        "resource_measurement": resource_sample,
        "density_scope": {
            "capability": "candidate arm mean held-out score",
            "flops_and_updates": (
                "estimated end-to-end compute for all four arms across every preregistered seed"
            ),
            "params": "largest simultaneously trainable arm",
            "bytes": "dataset, weights, inherited features, and evidence documents",
            "seconds_and_peak_rss": "complete scientific attempt process",
        },
        "density": density_block(
            {metric_names[0]: float(metrics[metric_names[0]])},
            params=max_params,
            flops=total_flops,
            bytes=package.artifact_bytes,
            seconds=seconds,
            updates=total_updates,
            peak_rss_bytes=peak_rss,
        ),
    }


def _smoke_result(
    *,
    experiment_id: str,
    metric_names: Sequence[str],
    cfg: DictConfig,
    device: DeviceInfo,
    run_dir: Path,
    eligible: bool,
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    e = cfg.experiment
    seed_everything(int(e.smoke.seed))
    x = torch.randn(int(e.smoke.samples), int(e.smoke.dim))
    y = torch.randint(0, int(e.smoke.classes), (int(e.smoke.samples),))
    encoder = nn.Linear(int(e.smoke.dim), int(e.smoke.width))
    head = nn.Linear(int(e.smoke.width), int(e.smoke.classes))
    optimizer = torch.optim.SGD([*encoder.parameters(), *head.parameters()], lr=0.01)
    before = float(F.cross_entropy(head(encoder(x)), y).detach())
    optimizer.zero_grad()
    loss = F.cross_entropy(head(encoder(x)), y)
    loss.backward()
    optimizer.step()
    after = float(F.cross_entropy(head(encoder(x)), y).detach())
    params = param_count(encoder) + param_count(head)
    projection = _resource_projection(
        experiment_id=experiment_id,
        device=device,
        package_bytes=0,
        budget_flops=None,
        arm_count=0,
        seed_count=1,
        estimated_peak_bytes=params * 16 + x.numel() * x.element_size(),
    )
    projection_path = _write_json(run_dir / "resource_projection.json", projection)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "experiment_id": experiment_id,
        "status": "smoke-only",
        "scientific_result": False,
        "promotion_eligible": False,
        "evidence_eligible": eligible,
        "checks_fingerprint": _json_sha256(checks),
        "resolved_device": device.kind,
        "measured_host": apple_silicon_info(),
        "resource_projection": str(projection_path),
        "measured_hardware_wall": False,
        "completed_at": _utc_now(),
    }
    attempt_path = _write_json(run_dir / "attempt_receipt.json", attempt)
    output: dict[str, Any] = {name: None for name in metric_names}
    output.update(
        {
            "execution_status": "smoke-only",
            "scientific_result": False,
            "promotion_eligible": False,
            "natural_claim_eligible": False,
            "null_evaluated": False,
            "null_supported": None,
            "smoke_mechanics_pass": bool(math.isfinite(after) and after <= before + 1.0),
            "smoke_loss_before": round(before, 6),
            "smoke_loss_after": round(after, 6),
            "evidence_eligible": eligible,
            "resource_projection": str(projection_path),
            "attempt_receipt": str(attempt_path),
            "density": density_block(
                {"smoke_mechanics_pass": 1.0 if math.isfinite(after) else 0.0},
                params=params,
                flops=3
                * (
                    mlp_flops([int(e.smoke.dim), int(e.smoke.width)], int(e.smoke.samples))
                    + mlp_flops([int(e.smoke.width), int(e.smoke.classes)], int(e.smoke.samples))
                ),
                updates=1,
            ),
        }
    )
    return output


def run_gated_rewrite(
    *,
    variant: str,
    experiment_id: str,
    metric_names: Sequence[str],
    cfg: DictConfig,
    device: DeviceInfo,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    e = cfg.experiment
    mode = str(e.execution_mode)
    if mode not in {"smoke", "scientific"}:
        raise ValueError(f"{experiment_id}: execution_mode must be smoke or scientific, got {mode!r}")
    evidence_cfg = OmegaConf.to_container(e.evidence, resolve=True)
    if not isinstance(evidence_cfg, dict):
        raise ValueError(f"{experiment_id}: evidence config must be a mapping")
    requirements = _requirements(variant)
    checks: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name, required in requirements.items():
        check, document = _load_evidence_document(str(evidence_cfg.get(name, "")), required)
        checks[name] = check
        if document is not None:
            documents[name] = document
    basic_eligible = all(check["status"] == "valid" for check in checks.values())
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "experiment_id": experiment_id,
        "requested_mode": mode,
        "evidence_eligible": basic_eligible,
        "scientific_result": False,
        "promotion_eligible": False,
        "checks": checks,
        "generated_at": _utc_now(),
    }
    preflight_path = run_dir / "preflight_receipt.json"
    if mode == "smoke":
        _write_json(preflight_path, preflight)
        output = _smoke_result(
            experiment_id=experiment_id,
            metric_names=metric_names,
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            eligible=basic_eligible,
            checks=checks,
        )
        output["preflight_receipt"] = str(preflight_path)
        return output

    started_at = _utc_now()
    projection_path = run_dir / "resource_projection.json"
    attempt_path = run_dir / "attempt_receipt.json"
    progress_path = run_dir / "scientific_progress.json"
    if not basic_eligible:
        missing = [name for name, check in checks.items() if check["status"] != "valid"]
        _write_json(preflight_path, preflight)
        _write_json(
            projection_path,
            _resource_projection(
                experiment_id=experiment_id,
                device=device,
                package_bytes=0,
                budget_flops=None,
                arm_count=4,
                seed_count=0,
                estimated_peak_bytes=0,
            ),
        )
        _write_json(
            attempt_path,
            {
                "schema": ATTEMPT_SCHEMA,
                "experiment_id": experiment_id,
                "status": "refused-preflight",
                "started_at": started_at,
                "completed_at": _utc_now(),
                "scientific_result": False,
                "promotion_eligible": False,
                "invalid_evidence": missing,
                "resolved_device": device.kind,
                "measured_hardware_wall": False,
                "resource_projection": str(projection_path),
            },
        )
        raise ScientificExecutionRefused(
            f"{experiment_id} scientific execution refused: missing or invalid evidence "
            f"{missing}; receipt={preflight_path}"
        )

    scientific_raw = OmegaConf.to_container(e.scientific, resolve=True)
    if not isinstance(scientific_raw, dict):
        raise ValueError(f"{experiment_id}: scientific config must be a mapping")
    scientific_cfg = cast(dict[str, Any], scientific_raw)
    seeds: list[int] = []
    try:
        seeds = _validated_scientific_config(variant, scientific_cfg)
        package = _validate_and_load_package(
            variant,
            documents,
            checks,
            seeds,
            scientific_cfg,
            int(scientific_cfg["max_package_bytes"]),
            int(scientific_cfg["max_resident_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks["scientific_package"] = {
            "status": "invalid",
            "path": "cross-document-and-artifact-validation",
            "problems": [str(exc)],
        }
        preflight["checks"] = checks
        preflight["evidence_eligible"] = False
        _write_json(preflight_path, preflight)
        _write_json(
            projection_path,
            _resource_projection(
                experiment_id=experiment_id,
                device=device,
                package_bytes=0,
                budget_flops=None,
                arm_count=4,
                seed_count=len(seeds),
                estimated_peak_bytes=0,
            ),
        )
        _write_json(
            attempt_path,
            {
                "schema": ATTEMPT_SCHEMA,
                "experiment_id": experiment_id,
                "status": "refused-evidence-integrity",
                "started_at": started_at,
                "completed_at": _utc_now(),
                "scientific_result": False,
                "promotion_eligible": False,
                "problem": str(exc),
                "resolved_device": device.kind,
                "measured_hardware_wall": False,
                "resource_projection": str(projection_path),
            },
        )
        raise ScientificExecutionRefused(
            f"{experiment_id} scientific execution refused: {exc}; receipt={preflight_path}"
        ) from exc

    preflight["evidence_eligible"] = True
    preflight["evidence_scope"] = package.evidence_scope
    _write_json(preflight_path, preflight)
    planned_budget = int(package.evidence_documents["matched_compute_receipt"]["budget_flops"])
    _write_json(
        projection_path,
        _resource_projection(
            experiment_id=experiment_id,
            device=device,
            package_bytes=package.artifact_bytes,
            budget_flops=planned_budget,
            arm_count=4,
            seed_count=len(seeds),
            estimated_peak_bytes=package.resident_tensor_bytes * 2,
        ),
    )
    _write_json(
        attempt_path,
        {
            "schema": ATTEMPT_SCHEMA,
            "experiment_id": experiment_id,
            "status": "started",
            "started_at": started_at,
            "scientific_result": False,
            "promotion_eligible": False,
            "evidence_scope": package.evidence_scope,
            "evidence_hashes": package.evidence_hashes,
            "resolved_device": device.kind,
            "measured_host": apple_silicon_info(),
            "resource_projection": str(projection_path),
            "measured_hardware_wall": False,
        },
    )
    attempt_started_perf = time.perf_counter()
    resource_meter = _AttemptResourceMeter(device)
    try:
        with resource_meter:
            output = _scientific_result(
                variant=variant,
                experiment_id=experiment_id,
                metric_names=metric_names,
                package=package,
                scientific=scientific_cfg,
                device=device,
                projection_path=projection_path,
                progress_path=progress_path,
                resource_meter=resource_meter,
            )
    except Exception as exc:
        resource_sample = resource_meter.snapshot()
        elapsed = time.perf_counter() - attempt_started_perf
        progress: dict[str, Any] = (
            cast(dict[str, Any], json.loads(progress_path.read_text()))
            if progress_path.is_file()
            else {
                "schema": "mop-rewrite-progress/v1",
                "status": "failed-before-first-arm",
                "completed_arms": [],
                "completed_seed_count": 0,
                "consumed_estimated_flops": 0,
            }
        )
        progress["status"] = "failed"
        progress["failed_at"] = _utc_now()
        progress["problem"] = f"{type(exc).__name__}: {exc}"
        progress["resource_sample"] = resource_sample
        _write_json(progress_path, progress)
        projection = cast(dict[str, Any], json.loads(projection_path.read_text()))
        projection["actual_attempt"] = {
            "completed": False,
            "wall_seconds": elapsed,
            "resource_sample": resource_sample,
            "completed_arm_count": len(progress.get("completed_arms", [])),
            "completed_seed_count": int(progress.get("completed_seed_count", 0)),
            "consumed_estimated_flops": int(progress.get("consumed_estimated_flops", 0)),
        }
        _write_json(projection_path, projection)
        _write_json(
            attempt_path,
            {
                "schema": ATTEMPT_SCHEMA,
                "experiment_id": experiment_id,
                "status": "refused-or-failed-execution",
                "started_at": started_at,
                "completed_at": _utc_now(),
                "scientific_result": False,
                "promotion_eligible": False,
                "evidence_scope": package.evidence_scope,
                "problem": f"{type(exc).__name__}: {exc}",
                "wall_seconds": elapsed,
                "resource_sample": resource_sample,
                "progress_receipt": str(progress_path),
                "completed_arm_count": len(progress.get("completed_arms", [])),
                "completed_seed_count": int(progress.get("completed_seed_count", 0)),
                "consumed_estimated_flops": int(progress.get("consumed_estimated_flops", 0)),
                "resolved_device": device.kind,
                "measured_hardware_wall": False,
                "resource_projection": str(projection_path),
            },
        )
        raise

    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "experiment_id": experiment_id,
        "status": "completed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "scientific_result": True,
        "evidence_scope": package.evidence_scope,
        "natural_claim_eligible": output["natural_claim_eligible"],
        "promotion_eligible": output["promotion_eligible"],
        "null_supported": output["null_supported"],
        "result_fingerprint": _json_sha256(output),
        "evidence_hashes": package.evidence_hashes,
        "resolved_device": device.kind,
        "measured_host": apple_silicon_info(),
        "resource_projection": str(projection_path),
        "progress_receipt": output["progress_receipt"],
        "resource_sample": output["resource_measurement"],
        "measured_hardware_wall": False,
    }
    _write_json(attempt_path, attempt)
    output["preflight_receipt"] = str(preflight_path)
    output["attempt_receipt"] = str(attempt_path)
    return output
