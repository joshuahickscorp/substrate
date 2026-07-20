"""Independent, fail-closed verification for the Generation-1 cognitive corpus.

The strict v2 path intentionally does not call the corpus builder, seed adapter, experiment
summariser, or receipt validators.  It recomputes their scientific claims from the frozen
configuration and the files on disk so that a shared implementation bug cannot certify itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, cast

from omegaconf import OmegaConf

from ..config import REPO_ROOT, compose
from ..experiments import REGISTRY
from mop.substrate.events import canonical_bytes

VERIFICATION_SCHEMA = "mop-generation1-cognitive-corpus-verification/v2"
CONFIG_SCHEMA = "mop-generation1-cognitive-corpus-config/v2"
SEED_SCHEMA = "mop-generation1-cognitive-seed/v2"
ATTEMPT_SCHEMA = "mop-generation1-cognitive-attempt/v2"
CORPUS_SCHEMA = "mop-generation1-cognitive-corpus/v2"
SEED_POLICY_SCHEMA = "mop-generation1-seed-authority-policy/v1"
SEED_AUTHORITY_SCHEMA = "mop-generation1-effective-seed-authority/v1"
CELL_AUTHORITY_SCHEMA = "mop-generation1-cell-authority/v1"
SEED_ALGORITHM = "sha256-domain-separated-31bit-v1"
MAX_EFFECTIVE_SEED = 2**31 - 1
VARIED = "varied"
FIXED = "fixed"
MECHANICS = "mechanics"
INFERENTIAL = "inferential"
FIXED_EVIDENCE = "fixed_case_noninferential"
MECHANICS_EVIDENCE = "mechanics_noninferential"
DIRECTIONAL_LABELS = frozenset({"stable_null", "stable_candidate_trace"})

_LEGACY_INTEGER_KEY_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "a7_comm_channel": (("per_codebook_size",),),
    "c9_systematicity_sweep": (
        ("systematicity_curve_frozen_random",),
        ("systematicity_curve_real",),
    ),
    "ex13_long_stream": (
        ("effective_rank", "frozen_random"),
        ("effective_rank", "naive"),
        ("effective_rank", "protected"),
    ),
    "ex15_rejuvenation": tuple(
        (family, arm)
        for family in ("dead_unit_count", "effective_rank")
        for arm in ("frozen_random_rejuvenated", "protected", "protected_rejuvenated")
    ),
    "ex5_local_rules_scale": (("depth_sweep",),),
    "i1_info_bottleneck": (("acc_frozen_random",), ("acc_real",)),
    "i6_mi_audit": (("ratio_by_capacity_rung",),),
    "i8_quant_robustness": (("acc_frozen_random",), ("acc_real",)),
    "i9_vq_rate_distortion": (("kmeans_curve",), ("random_curve",), ("vq_curve",)),
    "p4_intelligence_is_compression": (("capability_vs_bits",),),
}

DEFAULT_CORPUS = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.json"
DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"

_NONSCIENTIFIC_METRIC_KEYS = frozenset(
    {
        "seed",
        "seeds",
        "data_seed",
        "base_seed",
        "effective_seed",
        "effective_seeds",
        "wall_clock_s",
        "wall_clock_seconds",
        "elapsed_seconds",
        "seconds",
        "maximum_rss_bytes",
        "max_rss_bytes",
        "run_dir",
        "receipt",
        "recorded_at",
        "timestamp",
        "path",
        "resource_observation",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == _canonical_sha256(core)


def _sealed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: _canonical_sha256(core)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_authority(path: Path, *, role: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"authority is not a regular file: {resolved}")
    row: dict[str, Any] = {
        "path": _repository_path(resolved),
        "sha256": _sha256_file(resolved),
    }
    if role is not None:
        row["role"] = role
    return row


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_object(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Generation-1 corpus config must use {CONFIG_SCHEMA}")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Generation-1 corpus needs at least five distinct nonnegative seeds")
    policy = config.get("seed_authority")
    if policy is None:
        return config
    if not isinstance(policy, dict) or policy.get("schema") != SEED_POLICY_SCHEMA:
        raise ValueError(f"Generation-1 seed authority must use {SEED_POLICY_SCHEMA}")
    if policy.get("algorithm") != SEED_ALGORITHM:
        raise ValueError(f"Generation-1 seed authority must use {SEED_ALGORITHM}")
    if policy.get("variation_canary_outer_seeds") != seeds[:5]:
        raise ValueError("variation canary must be exactly the first five frozen outer seeds")
    for field in (
        "outer_seed_experiment_ids",
        "fixed_case_experiment_ids",
        "mechanics_only_experiment_ids",
    ):
        values = policy.get(field)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"seed_authority.{field} must be a duplicate-free list")
    mode_sets = [
        set(policy[field])
        for field in (
            "outer_seed_experiment_ids",
            "fixed_case_experiment_ids",
            "mechanics_only_experiment_ids",
        )
    ]
    if any(left & right for index, left in enumerate(mode_sets) for right in mode_sets[index + 1 :]):
        raise ValueError("seed-authority experiment modes overlap")
    classification = config.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("classification policy is missing")
    for field in ("minimum_complete_seeds", "minimum_boolean_observations"):
        value = classification.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"classification.{field} must be a positive integer")
    stable_fraction = classification.get("stable_fraction")
    if (
        isinstance(stable_fraction, bool)
        or not isinstance(stable_fraction, int | float)
        or not 0.5 < float(stable_fraction) <= 1.0
    ):
        raise ValueError("classification.stable_fraction must be in (0.5, 1]")
    return config


def _eligible_experiment_ids(config: dict[str, Any]) -> list[str]:
    scope = config["experiment_scope"]
    tiers = set(scope["tiers"])
    excluded = set(scope.get("excluded_ids") or [])
    mechanics = set(config["seed_authority"]["mechanics_only_experiment_ids"])
    include_f = bool(scope.get("include_f_series"))
    include_wrappers = bool(scope.get("include_wrapper_smokes", True))
    return sorted(
        experiment_id
        for experiment_id, experiment in REGISTRY.items()
        if experiment.tier in tiers
        and experiment_id not in excluded
        and (include_f or not experiment_id.startswith("f"))
        and (include_wrappers or experiment_id not in mechanics)
    )


def _mode_and_evidence(config: dict[str, Any], experiment_id: str) -> tuple[str, str]:
    policy = config["seed_authority"]
    if experiment_id in policy["mechanics_only_experiment_ids"]:
        return MECHANICS, MECHANICS_EVIDENCE
    if experiment_id in policy["fixed_case_experiment_ids"]:
        return FIXED, FIXED_EVIDENCE
    return VARIED, INFERENTIAL


def _path_value(root: dict[str, Any], dotted: str) -> Any:
    components = dotted.split(".")
    if components and components[0] == "experiment":
        components = components[1:]
    value: Any = root
    for component in components:
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"configured seed path is missing: {dotted}")
        value = value[component]
    return value


def _configured_seed_paths(
    config: dict[str, Any], experiment_id: str, experiment: dict[str, Any]
) -> list[str]:
    policy = config["seed_authority"]
    mode, _ = _mode_and_evidence(config, experiment_id)
    if mode != VARIED or experiment_id in policy["outer_seed_experiment_ids"]:
        return []
    declared = policy.get("experiment_seed_paths", {}).get(experiment_id)
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            raise ValueError(f"seed paths for {experiment_id} are invalid")
        return list(declared)
    if "seeds" in experiment:
        return ["experiment.seeds"]
    if "seed" in experiment:
        return ["experiment.seed"]
    raise ValueError(f"inferential experiment {experiment_id} has no effective seed path")


def _derived_seed(
    *,
    config: dict[str, Any],
    experiment_id: str,
    outer_seed: int,
    path: str,
    index: int,
    original: int,
    nonce: int = 0,
) -> int:
    domain = {
        "schema": SEED_AUTHORITY_SCHEMA,
        "algorithm": SEED_ALGORITHM,
        "campaign_id": config["campaign_id"],
        "result_tag": config["result_tag"],
        "experiment_id": experiment_id,
        "outer_seed": outer_seed,
        "path": path,
        "index": index,
        "original": original,
        "nonce": nonce,
    }
    return int.from_bytes(hashlib.sha256(canonical_bytes(domain)).digest()[:8], "big") % (
        2**31 - 1
    )


def _effective_value(
    value: Any,
    *,
    config: dict[str, Any],
    experiment_id: str,
    outer_seed: int,
    path: str,
) -> int | list[int]:
    if isinstance(value, bool):
        raise ValueError(f"seed field {path} cannot be boolean")
    if isinstance(value, int):
        return _derived_seed(
            config=config,
            experiment_id=experiment_id,
            outer_seed=outer_seed,
            path=path,
            index=0,
            original=value,
        )
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"seed field {path} must be an integer or nonempty integer list")
    derived: list[int] = []
    for index, original in enumerate(value):
        nonce = 0
        candidate = _derived_seed(
            config=config,
            experiment_id=experiment_id,
            outer_seed=outer_seed,
            path=path,
            index=index,
            original=original,
            nonce=nonce,
        )
        while candidate in derived:
            nonce += 1
            candidate = _derived_seed(
                config=config,
                experiment_id=experiment_id,
                outer_seed=outer_seed,
                path=path,
                index=index,
                original=original,
                nonce=nonce,
            )
        derived.append(candidate)
    return derived


def _expected_seed_authority(
    config: dict[str, Any], experiment_id: str, outer_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = compose(
        [
            f"experiment={experiment_id}",
            "device=cpu",
            f"result_tag={config['result_tag']}",
        ]
    )
    experiment_value = OmegaConf.to_container(cfg.experiment, resolve=True)
    if not isinstance(experiment_value, dict):
        raise ValueError(f"experiment config for {experiment_id} is not a mapping")
    experiment = cast(dict[str, Any], experiment_value)
    mode, evidence = _mode_and_evidence(config, experiment_id)
    original_outer = int(cfg.get("seed", 0))
    overrides: list[dict[str, Any]] = [
        {
            "role": "campaign_outer_seed",
            "path": "seed",
            "original": original_outer,
            "effective": outer_seed,
        }
    ]
    effective_paths: dict[str, Any] = {}
    if mode == VARIED:
        for path in _configured_seed_paths(config, experiment_id, experiment):
            original = _path_value(experiment, path)
            effective = _effective_value(
                original,
                config=config,
                experiment_id=experiment_id,
                outer_seed=outer_seed,
                path=path,
            )
            overrides.append(
                {
                    "role": "experiment_seed_control",
                    "path": path,
                    "original": original,
                    "effective": effective,
                }
            )
            effective_paths[path] = effective
    execute_once = mode in {FIXED, MECHANICS}
    core = {
        "schema": SEED_AUTHORITY_SCHEMA,
        "algorithm": SEED_ALGORITHM,
        "policy_sha256": _canonical_sha256(config["seed_authority"]),
        "campaign_id": config["campaign_id"],
        "result_tag": config["result_tag"],
        "experiment_id": experiment_id,
        "outer_seed": outer_seed,
        "mode": mode,
        "evidence_class": evidence,
        "effective_overrides": overrides,
        "execute_once": execute_once,
        "reference_outer_seed": int(config["seeds"][0]) if execute_once else outer_seed,
    }
    return {**core, "authority_sha256": _canonical_sha256(core)}, effective_paths


def _implementation_authorities(experiment_id: str) -> list[dict[str, Any]]:
    source_name = inspect.getsourcefile(REGISTRY[experiment_id])
    if source_name is None:
        raise ValueError(f"cannot locate implementation source for {experiment_id}")
    paths = (
        ("experiment_source", Path(source_name)),
        ("experiment_harness", REPO_ROOT / "src/mop/harness/runner.py"),
        ("generation1_driver", Path(__file__).with_name("generation1_cognitive_corpus.py")),
    )
    return [_file_authority(path, role=role) for role, path in paths]


def _expected_static_authority(
    config: dict[str, Any], experiment_id: str, outer_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_authority, effective_paths = _expected_seed_authority(
        config, experiment_id, outer_seed
    )
    mode, evidence = _mode_and_evidence(config, experiment_id)
    return (
        {
            "schema": CELL_AUTHORITY_SCHEMA,
            "evidence_class": evidence,
            "seed_mode": mode,
            "seed_authority": seed_authority,
            "experiment_config": _file_authority(
                REPO_ROOT / "configs/experiment" / f"{experiment_id}.yaml"
            ),
            "implementation_authorities": _implementation_authorities(experiment_id),
        },
        effective_paths,
    )


def _scientific_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scientific_payload(child)
            for key, child in sorted(value.items())
            if isinstance(key, str) and key.lower() not in _NONSCIENTIFIC_METRIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_payload(child) for child in value]
    return value


def _scientific_fingerprint(metrics: dict[str, Any]) -> str:
    return _canonical_sha256(_scientific_payload(metrics))


def _legacy_integer_key_fingerprint(
    metrics: dict[str, Any], experiment_id: str
) -> str | None:

    paths = _LEGACY_INTEGER_KEY_PATHS.get(experiment_id)
    if paths is None:
        return None
    payload = copy.deepcopy(_scientific_payload(metrics))
    for path in paths:
        parent: Any = payload
        for part in path[:-1]:
            if not isinstance(parent, dict) or not isinstance(parent.get(part), dict):
                return None
            parent = parent[part]
        if not isinstance(parent, dict):
            return None
        target = parent.get(path[-1])
        if not isinstance(target, dict) or not target:
            return None
        if any(
            not isinstance(key, str)
            or not key
            or (not key.isdigit() and not (key[0] == "-" and key[1:].isdigit()))
            or str(int(key)) != key
            for key in target
        ):
            return None
        parent[path[-1]] = {}
    return _canonical_sha256(payload)


def _flatten_scalars(value: Any, prefix: str = "", depth: int = 0) -> dict[str, bool | float]:
    if depth > 3:
        return {}
    if isinstance(value, bool):
        return {prefix: value} if prefix else {}
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)):
        return {prefix: float(value)} if prefix else {}
    if not isinstance(value, dict):
        return {}
    flattened: dict[str, bool | float] = {}
    for key, child in value.items():
        if isinstance(key, str):
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_scalars(child, child_prefix, depth + 1))
    return flattened


def _resolved_config_valid(
    path: Path,
    *,
    expected_authority: dict[str, Any],
    effective_paths: dict[str, Any],
    result_tag: str,
) -> bool:
    try:
        resolved = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(resolved, dict):
        return False
    seed_authority = expected_authority["seed_authority"]
    if (
        resolved.get("seed") != seed_authority["outer_seed"]
        or resolved.get("result_tag") != result_tag
        or resolved.get("generation1_seed_authority") != seed_authority
        or resolved.get("generation1_evidence_class") != expected_authority["evidence_class"]
    ):
        return False
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        return False
    try:
        return all(_path_value(experiment, path) == value for path, value in effective_paths.items())
    except ValueError:
        return False


def _receipt_core_valid(
    receipt: dict[str, Any],
    *,
    attempt_dir: Path,
    experiment_id: str,
    outer_seed: int,
    expected_static: dict[str, Any],
    effective_paths: dict[str, Any],
    result_tag: str,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    problems: list[str] = []
    if receipt.get("schema") != ATTEMPT_SCHEMA:
        problems.append("schema")
    if not _valid_seal(receipt, "attempt_sha256"):
        problems.append("seal")
    if receipt.get("experiment_id") != experiment_id or receipt.get("seed") != outer_seed:
        problems.append("identity")
    if receipt.get("run_dir") != _repository_path(attempt_dir):
        problems.append("run_dir")
    returncode = receipt.get("returncode")
    timed_out = receipt.get("timed_out")
    seconds = receipt.get("seconds")
    if (
        (returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)))
        or not isinstance(timed_out, bool)
        or (returncode is None and timed_out is not True)
        or isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
        or not isinstance(receipt.get("stdout_tail"), str)
        or not isinstance(receipt.get("stderr_tail"), str)
        or not isinstance(receipt.get("recorded_at"), str)
    ):
        problems.append("attempt_execution_fields")
    for field in (
        "evidence_class",
        "seed_mode",
        "seed_authority",
        "experiment_config",
        "implementation_authorities",
    ):
        if receipt.get(field) != expected_static[field]:
            problems.append(field)
    manifest: dict[str, Any] | None = None
    manifest_path = attempt_dir / "manifest.json"
    manifest_binding = receipt.get("manifest")
    successful = receipt.get("returncode") == 0 and receipt.get("timed_out") is False
    if successful:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            problems.append("manifest_missing")
        else:
            try:
                manifest = _load_object(manifest_path)
            except (OSError, json.JSONDecodeError, ValueError):
                problems.append("manifest_json")
            expected_binding = {
                "path": _repository_path(manifest_path),
                "sha256": _sha256_file(manifest_path),
            }
            if manifest_binding != expected_binding:
                problems.append("manifest_binding")
        resolved_path = attempt_dir / "config.yaml"
        resolved_binding = receipt.get("resolved_config")
        if (
            not resolved_path.is_file()
            or resolved_path.is_symlink()
            or resolved_binding
            != {"path": _repository_path(resolved_path), "sha256": _sha256_file(resolved_path)}
        ):
            problems.append("resolved_config_binding")
        elif not _resolved_config_valid(
            resolved_path,
            expected_authority=expected_static,
            effective_paths=effective_paths,
            result_tag=result_tag,
        ):
            problems.append("resolved_config_seed_semantics")
    elif manifest_binding is not None and (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_binding
        != {
            "path": _repository_path(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
    ):
        problems.append("failed_manifest_binding")
    return not problems, problems, manifest


def _cell_valid(
    *,
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    experiment_id: str,
    outer_seed: int,
    expected_static: dict[str, Any],
) -> tuple[bool, list[str], str]:
    problems: list[str] = []
    extra = manifest.get("extra")
    metrics = manifest.get("metrics")
    if (
        manifest.get("name") != experiment_id
        or manifest.get("seed") != outer_seed
        or manifest.get("status") != "ok"
        or manifest.get("result_tag") != expected_static["seed_authority"]["result_tag"]
        or not isinstance(metrics, dict)
        or not isinstance(extra, dict)
        or not isinstance(extra.get("contract"), dict)
    ):
        problems.append("manifest_contract")
        return False, problems, "invalid"
    cell = extra.get("generation1_cell_authority")
    if not isinstance(cell, dict):
        return False, ["cell_authority_missing"], "invalid"
    for field, expected in expected_static.items():
        if cell.get(field) != expected:
            problems.append(f"cell_{field}")
    resolved = receipt.get("resolved_config")
    if cell.get("resolved_config") != resolved:
        problems.append("cell_resolved_config")
    fingerprint = _scientific_fingerprint(metrics)
    declared_fingerprint = cell.get("scientific_metrics_sha256")
    if declared_fingerprint == fingerprint:
        fingerprint_mode = "canonical_json"
    elif declared_fingerprint == _legacy_integer_key_fingerprint(metrics, experiment_id):
        fingerprint_mode = "legacy_pre_json_integer_keys"
    else:
        fingerprint_mode = "invalid"
        problems.append("scientific_fingerprint")
    for field in (
        "evidence_class",
        "seed_mode",
        "seed_authority",
        "experiment_config",
        "implementation_authorities",
        "resolved_config",
    ):
        if receipt.get(field) != cell.get(field):
            problems.append(f"attempt_cell_{field}")
    worker = receipt.get("worker_report")
    if not isinstance(worker, dict):
        problems.append("worker_report")
    else:
        expected_worker_fields = {
            "experiment_id": experiment_id,
            "seed": outer_seed,
            "evidence_class": cell.get("evidence_class"),
            "seed_mode": cell.get("seed_mode"),
            "seed_authority": cell.get("seed_authority"),
            "experiment_config": cell.get("experiment_config"),
            "implementation_authorities": cell.get("implementation_authorities"),
            "resolved_config": cell.get("resolved_config"),
            "manifest": receipt.get("manifest"),
            "scientific_metrics_sha256": declared_fingerprint,
        }
        if any(worker.get(field) != value for field, value in expected_worker_fields.items()):
            problems.append("worker_report_binding")
    return not problems, problems, fingerprint_mode


def _audit_attempts(
    *, config: dict[str, Any], run_root: Path, experiment_ids: list[str]
) -> tuple[
    dict[tuple[int, str], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    dict[str, int],
]:
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    visited: set[Path] = set()
    all_count = 0
    valid_count = 0
    first_seed = int(config["seeds"][0])
    for seed in config["seeds"]:
        for experiment_id in experiment_ids:
            mode, _ = _mode_and_evidence(config, experiment_id)
            reference_seed = first_seed if mode in {FIXED, MECHANICS} else int(seed)
            key = (reference_seed, experiment_id)
            if key in selected:
                continue
            expected_static, effective_paths = _expected_static_authority(
                config, experiment_id, reference_seed
            )
            class_root = run_root / f"seed_{reference_seed}" / "classes" / experiment_id
            candidates = sorted(class_root.glob("attempt_[0-9][0-9][0-9]"), reverse=True)
            chosen: dict[str, Any] | None = None
            for attempt_dir in candidates:
                visited.add(attempt_dir)
                all_count += 1
                receipt_path = attempt_dir / "attempt_receipt.json"
                if receipt_path.is_symlink():
                    invalid.append(
                        {"path": _repository_path(attempt_dir), "problems": ["receipt_symlink"]}
                    )
                    continue
                try:
                    receipt = _load_object(receipt_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    invalid.append(
                        {
                            "path": _repository_path(attempt_dir),
                            "problems": [f"receipt:{type(exc).__name__}"],
                        }
                    )
                    continue
                receipt_ok, receipt_problems, manifest = _receipt_core_valid(
                    receipt,
                    attempt_dir=attempt_dir,
                    experiment_id=experiment_id,
                    outer_seed=reference_seed,
                    expected_static=expected_static,
                    effective_paths=effective_paths,
                    result_tag=str(config["result_tag"]),
                )
                if receipt_ok:
                    valid_count += 1
                else:
                    invalid.append(
                        {"path": _repository_path(attempt_dir), "problems": receipt_problems}
                    )
                if receipt_ok and manifest is not None:
                    cell_ok, cell_problems, fingerprint_mode = _cell_valid(
                        manifest=manifest,
                        receipt=receipt,
                        experiment_id=experiment_id,
                        outer_seed=reference_seed,
                        expected_static=expected_static,
                    )
                    if not cell_ok:
                        invalid.append(
                            {"path": _repository_path(attempt_dir), "problems": cell_problems}
                        )
                    elif chosen is None:
                        chosen = {
                            "attempt_dir": attempt_dir,
                            "receipt": receipt,
                            "manifest": manifest,
                            "expected_static": expected_static,
                            "fingerprint_mode": fingerprint_mode,
                        }
            if chosen is not None:
                selected[key] = chosen
    for unexpected in sorted(
        set(run_root.glob("seed_*/classes/*/attempt_[0-9][0-9][0-9]")) - visited
    ):
        all_count += 1
        invalid.append(
            {"path": _repository_path(unexpected), "problems": ["unexpected_attempt_cell"]}
        )
    superseded: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    selected_numbers = {
        (seed, experiment_id): int(row["attempt_dir"].name.rsplit("_", 1)[-1])
        for (seed, experiment_id), row in selected.items()
    }
    for row in invalid:
        path = Path(str(row.get("path", "")))
        try:
            number = int(path.name.rsplit("_", 1)[-1])
            seed = int(path.parents[2].name.removeprefix("seed_"))
            experiment_id = path.parent.name
        except (IndexError, ValueError):
            unresolved.append(row)
            continue
        if number < selected_numbers.get((seed, experiment_id), -1):
            superseded.append(row)
        else:
            unresolved.append(row)
    fingerprint_modes: dict[str, int] = defaultdict(int)
    for row in selected.values():
        fingerprint_modes[str(row["fingerprint_mode"])] += 1
    return selected, superseded, unresolved, all_count, valid_count, dict(
        sorted(fingerprint_modes.items())
    )


def _expected_cell_receipt(
    *,
    selected: dict[str, Any],
    experiment_id: str,
    requested_seed: int,
    reference_seed: int,
) -> dict[str, Any]:
    attempt_dir = selected["attempt_dir"]
    receipt = selected["receipt"]
    manifest = selected["manifest"]
    cell = manifest["extra"]["generation1_cell_authority"]
    manifest_path = attempt_dir / "manifest.json"
    receipt_path = attempt_dir / "attempt_receipt.json"
    return {
        "experiment_id": experiment_id,
        "requested_outer_seed": requested_seed,
        "reference_outer_seed": reference_seed,
        "execution": (
            "skipped_execute_once_reference"
            if requested_seed != reference_seed
            else "executed_or_resumed"
        ),
        "evidence_class": cell["evidence_class"],
        "seed_mode": cell["seed_mode"],
        "seed_authority": cell["seed_authority"],
        "experiment_config": cell["experiment_config"],
        "implementation_authorities": cell["implementation_authorities"],
        "resolved_config": cell["resolved_config"],
        "scientific_metrics_sha256": cell["scientific_metrics_sha256"],
        "manifest": {
            "path": _repository_path(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "attempt_receipt": {
            "path": _repository_path(receipt_path),
            "sha256": _sha256_file(receipt_path),
            "attempt_sha256": receipt["attempt_sha256"],
            "self_seal_valid": True,
        },
    }


def _expected_seed_coverage(
    *,
    config: dict[str, Any],
    experiment_ids: list[str],
    selected: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    first_seed = int(config["seeds"][0])
    coverage: dict[str, Any] = {}
    for seed in config["seeds"]:
        complete: list[str] = []
        executed: list[str] = []
        skipped: list[str] = []
        for experiment_id in experiment_ids:
            mode, _ = _mode_and_evidence(config, experiment_id)
            reference_seed = first_seed if mode in {FIXED, MECHANICS} else int(seed)
            if (reference_seed, experiment_id) not in selected:
                continue
            complete.append(experiment_id)
            (skipped if reference_seed != seed else executed).append(experiment_id)
        coverage[str(seed)] = {
            "complete_count": len(complete),
            "expected_count": len(experiment_ids),
            "complete": len(complete) == len(experiment_ids),
            "missing_ids": sorted(set(experiment_ids) - set(complete)),
            "executed_or_resumed_ids": executed,
            "skipped_execute_once_ids": skipped,
        }
    return coverage


def _audit_seed_receipts(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_root: Path,
    experiment_ids: list[str],
    selected: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], bool, bool]:
    first_seed = int(config["seeds"][0])
    rows: dict[str, Any] = {}
    cells_exact = True
    config_binding = {
        "path": _repository_path(config_path),
        "sha256": _sha256_file(config_path),
    }
    mode_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    for experiment_id in experiment_ids:
        mode, evidence = _mode_and_evidence(config, experiment_id)
        mode_counts[mode] += 1
        evidence_counts[evidence] += 1
    for seed in config["seeds"]:
        path = run_root / f"seed_{seed}" / "seed_receipt.json"
        row: dict[str, Any] = {"path": _repository_path(path), "valid": False, "problems": []}
        if path.is_symlink():
            row["problems"].append("receipt_symlink")
            rows[str(seed)] = row
            continue
        try:
            receipt = _load_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            row["problems"].append(f"load:{type(exc).__name__}")
            rows[str(seed)] = row
            continue
        expected_executed = sorted(
            experiment_id
            for experiment_id in experiment_ids
            if _mode_and_evidence(config, experiment_id)[0] == VARIED or seed == first_seed
        )
        expected_skipped = sorted(set(experiment_ids) - set(expected_executed))
        base_expectations = {
            "schema": SEED_SCHEMA,
            "campaign_id": config["campaign_id"],
            "claim_scope": config["claim_scope"],
            "seed": seed,
            "config": config_binding,
            "eligible_ids": experiment_ids,
            "eligible_count": len(experiment_ids),
            "executed_complete_ids": expected_executed,
            "skipped_execute_once_ids": expected_skipped,
            "execute_once_reference_seed": first_seed,
            "complete_ids": experiment_ids,
            "complete_count": len(experiment_ids),
            "remaining_ids": [],
            "all_complete": True,
            "scientific_promotion": False,
        }
        for field, expected in base_expectations.items():
            if receipt.get(field) != expected:
                row["problems"].append(field)
        if not _valid_seal(receipt, "receipt_sha256"):
            row["problems"].append("receipt_sha256")
        expected_policy_summary = {
            "schema": SEED_POLICY_SCHEMA,
            "policy_sha256": _canonical_sha256(config["seed_authority"]),
            "outer_seed": seed,
            "mode_counts": dict(sorted(mode_counts.items())),
            "evidence_class_counts": dict(sorted(evidence_counts.items())),
        }
        if receipt.get("seed_authority") != expected_policy_summary:
            row["problems"].append("seed_authority")
        expected_cells: dict[str, Any] = {}
        for experiment_id in experiment_ids:
            mode, _ = _mode_and_evidence(config, experiment_id)
            reference_seed = first_seed if mode in {FIXED, MECHANICS} else int(seed)
            cell = selected.get((reference_seed, experiment_id))
            if cell is not None:
                expected_cells[experiment_id] = _expected_cell_receipt(
                    selected=cell,
                    experiment_id=experiment_id,
                    requested_seed=int(seed),
                    reference_seed=reference_seed,
                )
        if receipt.get("cell_receipts") != expected_cells:
            row["problems"].append("cell_receipts")
            cells_exact = False
        row["valid"] = not row["problems"]
        row["receipt_sha256"] = receipt.get("receipt_sha256")
        rows[str(seed)] = row
    return rows, all(row["valid"] for row in rows.values()), cells_exact


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": round(mean(values), 8),
        "std": round(pstdev(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
    }


def _wilson_95(successes: int, observations: int) -> dict[str, Any] | None:
    if observations <= 0:
        return None
    z = 1.959963984540054
    fraction = successes / observations
    denominator = 1.0 + (z * z / observations)
    center = (fraction + z * z / (2.0 * observations)) / denominator
    radius = (
        z
        * math.sqrt(
            fraction * (1.0 - fraction) / observations
            + z * z / (4.0 * observations * observations)
        )
        / denominator
    )
    return {
        "method": "wilson_score",
        "confidence": 0.95,
        "low": round(max(0.0, center - radius), 8),
        "high": round(min(1.0, center + radius), 8),
    }


def _independent_experiment_summary(
    *, experiment_id: str, records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    records = sorted(records, key=lambda row: config["seeds"].index(row["seed"]))
    expected_mode, expected_evidence = _mode_and_evidence(config, experiment_id)
    seed_modes = {row["cell"]["seed_mode"] for row in records}
    evidence_classes = {row["cell"]["evidence_class"] for row in records}
    seed_mode = next(iter(seed_modes)) if len(seed_modes) == 1 else "inconsistent"
    evidence_class = (
        next(iter(evidence_classes)) if len(evidence_classes) == 1 else "inconsistent"
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_overrides: set[str] = set()
    for record in records:
        digest = str(record["cell"]["seed_authority"]["authority_sha256"])
        seen_overrides.add(
            _canonical_sha256(record["cell"]["seed_authority"]["effective_overrides"])
        )
        if digest not in seen:
            seen.add(digest)
            unique.append(record)
    canary_seeds = list(config["seed_authority"]["variation_canary_outer_seeds"])
    canary = [row for row in records if row["seed"] in canary_seeds]
    if expected_mode != VARIED:
        canary_status = "not_applicable"
        structurally_varied = False
        scientifically_varied = False
    elif len(canary) != len(canary_seeds) or {row["seed"] for row in canary} != set(
        canary_seeds
    ):
        canary_status = "incomplete"
        structurally_varied = False
        scientifically_varied = False
    else:
        authority_hashes = {
            row["cell"]["seed_authority"]["authority_sha256"] for row in canary
        }
        override_hashes = {
            _canonical_sha256(row["cell"]["seed_authority"]["effective_overrides"])
            for row in canary
        }
        structurally_varied = (
            len(authority_hashes) == len(canary_seeds)
            and len(override_hashes) == len(canary_seeds)
        )
        scientifically_varied = (
            len({row["cell"]["scientific_metrics_sha256"] for row in canary}) >= 2
        )
        if not structurally_varied:
            canary_status = "seed_authority_failed"
        elif not scientifically_varied:
            canary_status = "scientific_output_invariant"
        else:
            canary_status = "passed"
    boolean_values: dict[str, list[bool]] = defaultdict(list)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    for record in unique:
        for key, value in _flatten_scalars(record["manifest"]["metrics"]).items():
            if isinstance(value, bool):
                boolean_values[key].append(value)
            else:
                numeric_values[key].append(value)
    nulls = boolean_values.get("null_supported", [])
    policy = config["classification"]
    minimum_complete = int(policy["minimum_complete_seeds"])
    effective_count = len(unique)
    structural_authority_ok = bool(
        seed_mode == expected_mode
        and evidence_class == expected_evidence
        and structurally_varied
        and len(records) == effective_count
        and len(records) == len(seen_overrides)
    )
    if evidence_class == MECHANICS_EVIDENCE or expected_evidence == MECHANICS_EVIDENCE:
        classification = "mechanics_noninferential"
    elif evidence_class == FIXED_EVIDENCE or expected_evidence == FIXED_EVIDENCE:
        classification = "descriptive_fixed_case"
    elif not structural_authority_ok:
        classification = "descriptive_seed_adapter_unverified"
    elif canary_status == "scientific_output_invariant":
        classification = "descriptive_seed_invariant"
    elif canary_status != "passed":
        classification = "descriptive_seed_adapter_unverified"
    elif effective_count < minimum_complete or len(nulls) != effective_count or len(nulls) < int(
        policy["minimum_boolean_observations"]
    ):
        classification = str(policy["missing_null_label"])
    else:
        null_fraction = sum(nulls) / len(nulls)
        threshold = float(policy["stable_fraction"])
        if null_fraction >= threshold:
            classification = "stable_null"
        elif 1.0 - null_fraction >= threshold:
            classification = "stable_candidate_trace"
        else:
            classification = str(policy["tie_label"])
    expected_execution_count = 1 if expected_mode in {FIXED, MECHANICS} else len(config["seeds"])
    return {
        "experiment_id": experiment_id,
        "completed_seed_count": len(records),
        "expected_execution_count": expected_execution_count,
        "effective_observation_count": effective_count,
        "distinct_seed_authority_count": len(seen),
        "distinct_effective_override_count": len(seen_overrides),
        "coverage_complete": effective_count >= expected_execution_count,
        "minimum_directional_seed_count": minimum_complete,
        "directional_evidence_eligible": bool(
            evidence_class == INFERENTIAL
            and seed_mode == VARIED
            and canary_status == "passed"
            and effective_count >= minimum_complete
            and len(records) == effective_count
            and len(nulls) == effective_count
            and len(nulls) >= int(policy["minimum_boolean_observations"])
        ),
        "evidence_class": evidence_class,
        "seed_mode": seed_mode,
        "classification": classification,
        "variation_canary": {
            "outer_seeds": canary_seeds,
            "expected_count": len(canary_seeds),
            "observed_count": len(canary),
            "status": canary_status,
            "structurally_varied": structurally_varied,
            "scientifically_varied": scientifically_varied,
            "scientific_metrics_sha256": {
                str(row["seed"]): row["cell"]["scientific_metrics_sha256"] for row in canary
            },
            "seed_authority_sha256": {
                str(row["seed"]): row["cell"]["seed_authority"]["authority_sha256"]
                for row in canary
            },
        },
        "null_supported": {
            "observations": len(nulls),
            "true": sum(nulls),
            "false": len(nulls) - sum(nulls),
            "fraction": round(sum(nulls) / len(nulls), 8) if nulls else None,
            "wilson_95": _wilson_95(sum(nulls), len(nulls)),
        },
        "boolean_rates": {
            key: {"n": len(values), "true_fraction": round(sum(values) / len(values), 8)}
            for key, values in sorted(boolean_values.items())
        },
        "numeric_summaries": {
            key: _numeric_summary(values) for key, values in sorted(numeric_values.items())
        },
        "contract": records[0]["manifest"]["extra"]["contract"] if records else {},
    }


def _independent_aggregate(
    *,
    config: dict[str, Any],
    experiment_ids: list[str],
    selected: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (seed, experiment_id), row in sorted(selected.items()):
        cell = row["manifest"]["extra"]["generation1_cell_authority"]
        by_experiment[experiment_id].append(
            {"seed": seed, "cell": cell, "manifest": row["manifest"]}
        )
        cell_index[experiment_id].append(
            _expected_cell_receipt(
                selected=row,
                experiment_id=experiment_id,
                requested_seed=seed,
                reference_seed=seed,
            )
        )
    summaries = {
        experiment_id: _independent_experiment_summary(
            experiment_id=experiment_id,
            records=by_experiment[experiment_id],
            config=config,
        )
        for experiment_id in experiment_ids
    }
    pack_summaries: dict[str, Any] = {}
    for pack, members in config["capability_packs"].items():
        rows = [summaries[experiment_id] for experiment_id in members]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["classification"]] += 1
        pack_summaries[pack] = {
            "experiment_ids": members,
            "classification_counts": dict(sorted(counts.items())),
            "minimum_seed_coverage": min(row["effective_observation_count"] for row in rows),
            "complete_experiment_count": sum(row["coverage_complete"] is True for row in rows),
        }
    minimum = int(config["classification"]["minimum_complete_seeds"])
    complete_ids = [
        experiment_id
        for experiment_id, row in summaries.items()
        if row["coverage_complete"] is True
        and (row["seed_mode"] in {FIXED, MECHANICS} or row["effective_observation_count"] >= minimum)
    ]
    mode_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    for row in summaries.values():
        mode_counts[row["seed_mode"]] += 1
        evidence_counts[row["evidence_class"]] += 1
    varied = [row for row in summaries.values() if row["seed_mode"] == VARIED]
    structural_no_pseudoreplication = all(
        row["completed_seed_count"] == row["distinct_seed_authority_count"]
        and row["completed_seed_count"] == row["distinct_effective_override_count"]
        and row["variation_canary"]["structurally_varied"] is True
        for row in varied
    )
    structural_failures = sorted(
        row["experiment_id"]
        for row in varied
        if row["variation_canary"]["structurally_varied"] is not True
    )
    authority_summary = {
        "policy": config["seed_authority"],
        "policy_sha256": _canonical_sha256(config["seed_authority"]),
        "mode_counts": dict(sorted(mode_counts.items())),
        "evidence_class_counts": dict(sorted(evidence_counts.items())),
        "varied_experiment_count": len(varied),
        "varied_with_five_distinct_canary_authorities": sum(
            row["variation_canary"]["structurally_varied"] is True for row in varied
        ),
        "canary_failures": structural_failures,
        "structural_canary_failures": structural_failures,
        "scientific_output_invariant_ids": sorted(
            row["experiment_id"]
            for row in varied
            if row["variation_canary"]["status"] == "scientific_output_invariant"
        ),
        "no_pseudoreplication": structural_no_pseudoreplication,
    }
    return {
        "experiment_summaries": summaries,
        "capability_pack_summaries": pack_summaries,
        "cell_authority_index": dict(sorted(cell_index.items())),
        "seed_authority_summary": authority_summary,
        "complete_experiment_ids": complete_ids,
        "complete_experiment_count": len(complete_ids),
        "corpus_complete": len(complete_ids) == len(experiment_ids),
        "no_pseudoreplication": structural_no_pseudoreplication,
    }


def _operational_summary(run_root: Path, *, manifest_bytes: int) -> dict[str, Any]:

    attempts = sorted(run_root.glob("seed_*/classes/*/attempt_[0-9][0-9][0-9]"))
    valid = 0
    invalid = 0
    wall_seconds = 0.0
    maximum_rss = 0
    by_cell: dict[tuple[str, str], int] = defaultdict(int)
    invalid_attempts: list[tuple[tuple[str, str], int]] = []
    max_valid_attempt_number: dict[tuple[str, str], int] = {}
    for attempt in attempts:
        cell = (attempt.parts[-4], attempt.parts[-2])
        by_cell[cell] += 1
        attempt_number = int(attempt.name.rsplit("_", 1)[-1])
        try:
            receipt = _load_object(attempt / "attempt_receipt.json")
        except (OSError, json.JSONDecodeError, ValueError):
            invalid += 1
            invalid_attempts.append((cell, attempt_number))
            continue
        if not _valid_seal(receipt, "attempt_sha256"):
            invalid += 1
            invalid_attempts.append((cell, attempt_number))
            continue
        valid += 1
        worker = receipt.get("worker_report")
        successful = bool(
            receipt.get("returncode") == 0
            and receipt.get("timed_out") is False
            and isinstance(receipt.get("manifest"), dict)
            and isinstance(worker, dict)
            and isinstance(worker.get("manifest"), dict)
        )
        if successful:
            max_valid_attempt_number[cell] = max(
                max_valid_attempt_number.get(cell, -1), attempt_number
            )
        seconds = receipt.get("seconds")
        if isinstance(seconds, int | float) and not isinstance(seconds, bool) and math.isfinite(seconds):
            wall_seconds += float(seconds)
        rss = worker.get("maximum_rss_bytes") if isinstance(worker, dict) else None
        if isinstance(rss, int) and not isinstance(rss, bool):
            maximum_rss = max(maximum_rss, rss)
    superseded = sum(
        1 for cell, number in invalid_attempts if number < max_valid_attempt_number.get(cell, -1)
    )
    unresolved = invalid - superseded
    return {
        "attempt_directory_count": len(attempts),
        "attempt_receipt_count": valid + invalid,
        "valid_attempt_receipt_count": valid,
        "invalid_attempt_receipt_count": invalid,
        "superseded_invalid_attempt_count": superseded,
        "unresolved_invalid_attempt_count": unresolved,
        "retry_count": sum(max(0, count - 1) for count in by_cell.values()),
        "summed_attempt_wall_seconds": round(wall_seconds, 6),
        "max_observed_worker_rss_bytes": maximum_rss or None,
        "manifest_bytes": manifest_bytes,
    }


def _mutation_suite(
    *,
    corpus: dict[str, Any],
    config: dict[str, Any],
    aggregate: dict[str, Any],
    selected: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    mutation = copy.deepcopy(corpus)
    mutation["seeds"] = list(reversed(mutation.get("seeds", [])))
    mutation = _sealed(mutation, "corpus_sha256")
    results["plan_tamper"] = mutation["seeds"] != config["seeds"]

    mutation = copy.deepcopy(corpus)
    index = mutation.get("cell_authority_index", {})
    if isinstance(index, dict) and index:
        experiment_id = sorted(index)[0]
        rows = index[experiment_id]
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            authority = rows[0].get("seed_authority")
            if isinstance(authority, dict):
                authority["authority_sha256"] = "0" * 64
    mutation = _sealed(mutation, "corpus_sha256")
    results["seed_authority_tamper"] = index != aggregate["cell_authority_index"]

    mutation = copy.deepcopy(corpus)
    summaries = mutation.get("experiment_summaries", {})
    if isinstance(summaries, dict) and summaries:
        experiment_id = sorted(summaries)[0]
        row = summaries[experiment_id]
        if isinstance(row, dict):
            row["evidence_class"] = "inferential-tampered"
    mutation = _sealed(mutation, "corpus_sha256")
    results["evidence_class_tamper"] = summaries != aggregate["experiment_summaries"]

    if selected:
        (_, experiment_id), row = sorted(selected.items())[0]
        receipt = copy.deepcopy(row["receipt"])
        receipt["seed_authority"]["experiment_id"] = f"{experiment_id}-tampered"
        receipt = _sealed(receipt, "attempt_sha256")
        results["attempt_tamper"] = receipt["seed_authority"] != row["expected_static"][
            "seed_authority"
        ]
    else:
        results["attempt_tamper"] = False

    mutation = copy.deepcopy(corpus)
    summaries = mutation.get("experiment_summaries", {})
    if isinstance(summaries, dict) and summaries:
        experiment_id = sorted(summaries)[0]
        row = summaries[experiment_id]
        if isinstance(row, dict):
            row["classification"] = "stable_candidate_trace-tampered"
    mutation = _sealed(mutation, "corpus_sha256")
    results["summary_tamper"] = summaries != aggregate["experiment_summaries"]
    return results


def _verify_legacy_corpus(
    *,
    corpus: dict[str, Any],
    corpus_path: Path,
    config: dict[str, Any],
    config_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    from .generation1_cognitive_corpus import build_corpus, eligible_experiment_ids

    experiment_ids = eligible_experiment_ids(config)
    checks: dict[str, bool] = {
        "corpus_schema": corpus.get("schema") == CORPUS_SCHEMA,
        "corpus_self_hash": _valid_seal(corpus, "corpus_sha256"),
        "config_schema": config.get("schema") == CONFIG_SCHEMA,
        "config_hash_bound": corpus.get("config")
        == {"path": _repository_path(config_path), "sha256": _sha256_file(config_path)},
        "experiment_set_exact": corpus.get("eligible_experiment_ids") == experiment_ids,
        "seed_set_exact": corpus.get("seeds") == config["seeds"],
    }
    seed_rows: dict[str, Any] = {}
    for seed in config["seeds"]:
        path = run_root / f"seed_{seed}" / "seed_receipt.json"
        row: dict[str, Any] = {"path": _repository_path(path), "valid": False}
        if path.is_file():
            receipt = _load_object(path)
            row.update(
                {
                    "schema": receipt.get("schema"),
                    "seed": receipt.get("seed"),
                    "all_complete": receipt.get("all_complete"),
                    "self_hash": _valid_seal(receipt, "receipt_sha256"),
                    "experiment_set_exact": receipt.get("eligible_ids") == experiment_ids,
                }
            )
            row["valid"] = bool(
                row["schema"] == SEED_SCHEMA
                and row["seed"] == seed
                and row["all_complete"] is True
                and row["self_hash"]
                and row["experiment_set_exact"]
            )
        seed_rows[str(seed)] = row
    checks["all_seed_receipts_valid"] = all(row["valid"] for row in seed_rows.values())
    checks["full_regeneration_match"] = build_corpus(config_path, run_root) == corpus
    checks["corpus_complete"] = corpus.get("corpus_complete") is True
    checks["promotion_blocked"] = corpus.get("scientific_promotion") is False

    mutation_results: dict[str, bool] = {}
    mutations: dict[str, dict[str, Any]] = {}
    mutation = copy.deepcopy(corpus)
    mutation["scientific_promotion"] = True
    mutations["promotion_flip"] = mutation
    mutation = copy.deepcopy(corpus)
    mutation["seeds"] = list(reversed(mutation.get("seeds", [])))
    mutations["seed_order_flip"] = mutation
    mutation = copy.deepcopy(corpus)
    summaries = mutation.get("experiment_summaries", {})
    if summaries:
        summaries[sorted(summaries)[0]]["classification"] = "positive"
    mutations["classification_tamper"] = mutation
    for name, mutated in mutations.items():
        mutation_results[name] = not _valid_seal(mutated, "corpus_sha256")
    checks["all_mutations_rejected"] = all(mutation_results.values())
    problems = [name for name, passed in checks.items() if not passed]
    core = {
        "schema": VERIFICATION_SCHEMA,
        "claim_scope": config["claim_scope"],
        "corpus": {
            "path": _repository_path(corpus_path),
            "sha256": _sha256_file(corpus_path),
            "corpus_sha256": corpus.get("corpus_sha256"),
        },
        "config": {"path": _repository_path(config_path), "sha256": _sha256_file(config_path)},
        "run_root": _repository_path(run_root),
        "checks": checks,
        "seed_receipts": seed_rows,
        "mutation_suite": {
            "count": len(mutation_results),
            "rejected": sum(mutation_results.values()),
            "results": mutation_results,
        },
        "verification_complete": not problems,
        "problems": problems,
        "scientific_promotion": False,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    return {**core, "verification_sha256": _canonical_sha256(core)}


def _declared_directional_row_safe(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("classification") not in DIRECTIONAL_LABELS:
        return True
    canary = row.get("variation_canary")
    nulls = row.get("null_supported")
    return bool(
        isinstance(canary, dict)
        and isinstance(nulls, dict)
        and row.get("directional_evidence_eligible") is True
        and canary.get("status") == "passed"
        and nulls.get("observations") == row.get("effective_observation_count")
    )


def verify_corpus(
    *, corpus_path: Path, config_path: Path, run_root: Path
) -> dict[str, Any]:
    corpus = _load_object(corpus_path)
    config = _load_config(config_path)
    if "seed_authority" not in config:
        return _verify_legacy_corpus(
            corpus=corpus,
            corpus_path=corpus_path,
            config=config,
            config_path=config_path,
            run_root=run_root,
        )
    experiment_ids = _eligible_experiment_ids(config)
    (
        selected,
        superseded_attempts,
        unresolved_attempts,
        attempt_count,
        valid_attempt_count,
        fingerprint_mode_counts,
    ) = _audit_attempts(
        config=config,
        run_root=run_root,
        experiment_ids=experiment_ids,
    )
    seed_receipts, seed_receipts_valid, receipt_cells_exact = _audit_seed_receipts(
        config=config,
        config_path=config_path,
        run_root=run_root,
        experiment_ids=experiment_ids,
        selected=selected,
    )
    aggregate = _independent_aggregate(
        config=config,
        experiment_ids=experiment_ids,
        selected=selected,
    )
    expected_coverage = _expected_seed_coverage(
        config=config,
        experiment_ids=experiment_ids,
        selected=selected,
    )
    manifest_paths = {row["attempt_dir"] / "manifest.json" for row in selected.values()}
    manifest_bytes = sum(path.stat().st_size for path in manifest_paths)
    expected_operational = _operational_summary(run_root, manifest_bytes=manifest_bytes)
    declared_operational = corpus.get("operational_summary")
    if (
        isinstance(declared_operational, dict)
        and declared_operational.get("invalid_attempt_receipt_count") == 0
        and not {
            "superseded_invalid_attempt_count",
            "unresolved_invalid_attempt_count",
        }.issubset(declared_operational)
    ):
        expected_operational = {
            key: value
            for key, value in expected_operational.items()
            if key
            not in {
                "superseded_invalid_attempt_count",
                "unresolved_invalid_attempt_count",
            }
        }

    directional_safe = all(
        row["classification"] not in DIRECTIONAL_LABELS
        or (
            row["seed_mode"] == VARIED
            and row["evidence_class"] == INFERENTIAL
            and row["variation_canary"]["status"] == "passed"
            and row["effective_observation_count"]
            >= int(config["classification"]["minimum_complete_seeds"])
            and row["null_supported"]["observations"] == row["effective_observation_count"]
        )
        for row in aggregate["experiment_summaries"].values()
    )
    declared_summaries = corpus.get("experiment_summaries")
    corpus_directional_safe = isinstance(declared_summaries, dict) and all(
        _declared_directional_row_safe(row) for row in declared_summaries.values()
    )
    summary_fields = (
        "experiment_summaries",
        "capability_pack_summaries",
        "seed_authority_summary",
    )
    independent_summary_match = all(corpus.get(field) == aggregate[field] for field in summary_fields)
    full_fields: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "claim_scope": config["claim_scope"],
        "config": {"path": _repository_path(config_path), "sha256": _sha256_file(config_path)},
        "run_root": _repository_path(run_root),
        "seed_count": len(config["seeds"]),
        "seeds": config["seeds"],
        "eligible_experiment_count": len(experiment_ids),
        "eligible_experiment_ids": experiment_ids,
        "minimum_complete_seeds": int(config["classification"]["minimum_complete_seeds"]),
        "complete_experiment_count": aggregate["complete_experiment_count"],
        "complete_experiment_ids": aggregate["complete_experiment_ids"],
        "corpus_complete": aggregate["corpus_complete"],
        "seed_coverage": expected_coverage,
        "cell_authority_index": aggregate["cell_authority_index"],
        "operational_summary": expected_operational,
        "total_manifest_bytes": manifest_bytes,
        "scientific_promotion": False,
    }
    full_regeneration_match = independent_summary_match and all(
        corpus.get(field) == expected for field, expected in full_fields.items()
    )
    seed_authority_exact = not unresolved_attempts and receipt_cells_exact and (
        corpus.get("cell_authority_index") == aggregate["cell_authority_index"]
    )
    expected_effective_cell_count = sum(
        1
        if _mode_and_evidence(config, experiment_id)[0] in {FIXED, MECHANICS}
        else len(config["seeds"])
        for experiment_id in experiment_ids
    )
    all_cell_authorities_valid = (
        len(selected) == expected_effective_cell_count
        and receipt_cells_exact
        and not unresolved_attempts
    )
    mutation_results = _mutation_suite(
        corpus=corpus,
        config=config,
        aggregate=aggregate,
        selected=selected,
    )
    declared_authority_summary = corpus.get("seed_authority_summary")
    checks = {
        "corpus_schema": corpus.get("schema") == CORPUS_SCHEMA,
        "corpus_self_hash": _valid_seal(corpus, "corpus_sha256"),
        "config_schema": config.get("schema") == CONFIG_SCHEMA,
        "config_hash_bound": corpus.get("config")
        == {"path": _repository_path(config_path), "sha256": _sha256_file(config_path)},
        "experiment_set_exact": corpus.get("eligible_experiment_ids") == experiment_ids,
        "seed_set_exact": corpus.get("seeds") == config["seeds"],
        "all_seed_receipts_valid": seed_receipts_valid,
        "all_attempt_receipts_valid": valid_attempt_count >= expected_effective_cell_count
        and not unresolved_attempts,
        "all_cell_authorities_valid": all_cell_authorities_valid,
        "seed_authority_exact": seed_authority_exact,
        "no_pseudoreplication": bool(aggregate["no_pseudoreplication"])
        and isinstance(declared_authority_summary, dict)
        and declared_authority_summary.get("no_pseudoreplication") is True,
        "independent_summary_match": independent_summary_match,
        "directional_inference_fail_closed": directional_safe and corpus_directional_safe,
        "full_regeneration_match": full_regeneration_match,
        "corpus_complete": corpus.get("corpus_complete") is True
        and aggregate["corpus_complete"] is True,
        "promotion_blocked": corpus.get("scientific_promotion") is False,
        "all_mutations_rejected": all(mutation_results.values()),
    }
    problems = [name for name, passed in checks.items() if not passed]
    core = {
        "schema": VERIFICATION_SCHEMA,
        "claim_scope": config["claim_scope"],
        "corpus": {
            "path": _repository_path(corpus_path),
            "sha256": _sha256_file(corpus_path),
            "corpus_sha256": corpus.get("corpus_sha256"),
        },
        "config": {"path": _repository_path(config_path), "sha256": _sha256_file(config_path)},
        "run_root": _repository_path(run_root),
        "checks": checks,
        "seed_receipts": seed_receipts,
        "attempt_audit": {
            "attempt_directory_count": attempt_count,
            "valid_attempt_count": valid_attempt_count,
            "selected_complete_cell_count": len(selected),
            "invalid_count": len(superseded_attempts) + len(unresolved_attempts),
            "superseded_invalid_count": len(superseded_attempts),
            "superseded_invalid": superseded_attempts,
            "unresolved_invalid_count": len(unresolved_attempts),
            "unresolved_invalid": unresolved_attempts,
            "invalid": [*superseded_attempts, *unresolved_attempts],
        },
        "authority_audit": {
            "expected_effective_cell_count": expected_effective_cell_count,
            "selected_effective_cell_count": len(selected),
            "policy_sha256": _canonical_sha256(config["seed_authority"]),
            "independent_summary_sha256": _canonical_sha256(
                aggregate["experiment_summaries"]
            ),
            "scientific_fingerprint_recompute": {
                "selected_cell_count": len(selected),
                "mode_counts": fingerprint_mode_counts,
            },
        },
        "mutation_suite": {
            "count": len(mutation_results),
            "rejected": sum(mutation_results.values()),
            "results": mutation_results,
        },
        "verification_complete": not problems,
        "problems": problems,
        "scientific_promotion": False,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    return {**core, "verification_sha256": _canonical_sha256(core)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiment/generation1_cognitive_corpus.json",
    )
    parser.add_argument(
        "--run-root", type=Path, default=REPO_ROOT / "runs/generation1/cognitive_corpus"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = verify_corpus(
        corpus_path=arguments.corpus.resolve(),
        config_path=arguments.config.resolve(),
        run_root=arguments.run_root.resolve(),
    )
    _atomic_json(arguments.out.resolve(), result)
    print(
        json.dumps(
            {
                "verification_complete": result["verification_complete"],
                "problems": result["problems"],
                "output": str(arguments.out),
            },
            indent=2,
        )
    )
    return 0 if result["verification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
