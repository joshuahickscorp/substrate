"""Fresh-seed Generation-1 cognitive-capability census.

The corpus runner deliberately stays below architecture-level promotion.  It executes every
locally runnable :class:`mop.experiments.base.Experiment` in isolated subprocesses, including the
F-series that the project-exhaustion ledger keeps on a separate evidence surface.  The resulting
manifests are aggregated as stability and falsification evidence, never as natural-world proof.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..config import REPO_ROOT, compose
from ..experiments import REGISTRY
from mop.substrate.events import canonical_bytes, canonical_sha256, sha256_file

CONFIG_SCHEMA = "mop-generation1-cognitive-corpus-config/v2"
SEED_SCHEMA = "mop-generation1-cognitive-seed/v2"
ATTEMPT_SCHEMA = "mop-generation1-cognitive-attempt/v2"
CORPUS_SCHEMA = "mop-generation1-cognitive-corpus/v2"
SEED_POLICY_SCHEMA = "mop-generation1-seed-authority-policy/v1"
SEED_AUTHORITY_SCHEMA = "mop-generation1-effective-seed-authority/v1"
CELL_AUTHORITY_SCHEMA = "mop-generation1-cell-authority/v1"
SEED_ALGORITHM = "sha256-domain-separated-31bit-v1"
MAX_EFFECTIVE_SEED = 2**31 - 1
EVIDENCE_INFERENTIAL = "inferential"
EVIDENCE_MECHANICS = "mechanics_noninferential"
EVIDENCE_FIXED = "fixed_case_noninferential"
SEED_MODE_VARIED = "varied"
SEED_MODE_MECHANICS = "mechanics"
SEED_MODE_FIXED = "fixed"
DEFAULT_CONFIG = REPO_ROOT / "configs/experiment/generation1_cognitive_corpus.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/generation1/cognitive_corpus"
DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.json"
MAX_TAIL = 12_000


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


def _sealed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: canonical_sha256(core)}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Generation-1 corpus config must use {CONFIG_SCHEMA}")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("Generation-1 corpus seeds must be at least five distinct nonnegative integers")
    scope = payload.get("experiment_scope")
    if not isinstance(scope, dict) or not scope.get("tiers"):
        raise ValueError("Generation-1 experiment scope is missing")
    packs = payload.get("capability_packs")
    if not isinstance(packs, dict) or not packs:
        raise ValueError("Generation-1 capability packs are missing")
    known = set(REGISTRY)
    for name, members in packs.items():
        if not isinstance(name, str) or not isinstance(members, list) or not members:
            raise ValueError("capability packs require a name and nonempty experiment list")
        missing = sorted(set(members) - known)
        if missing:
            raise ValueError(f"capability pack {name!r} references unknown experiments {missing}")
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("Generation-1 classification policy is missing")
    threshold = classification.get("stable_fraction")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0.5 < threshold <= 1:
        raise ValueError("stable_fraction must be in (0.5, 1]")
    minimum_complete = classification.get("minimum_complete_seeds")
    minimum_boolean = classification.get("minimum_boolean_observations")
    if (
        isinstance(minimum_complete, bool)
        or not isinstance(minimum_complete, int)
        or not 1 <= minimum_complete <= len(seeds)
    ):
        raise ValueError("minimum_complete_seeds must be a positive count within the seed set")
    if (
        isinstance(minimum_boolean, bool)
        or not isinstance(minimum_boolean, int)
        or not minimum_complete <= minimum_boolean <= len(seeds)
    ):
        raise ValueError(
            "minimum_boolean_observations must cover at least minimum_complete_seeds"
        )
    seed_policy = payload.get("seed_authority")
    if seed_policy is not None:
        if not isinstance(seed_policy, dict) or seed_policy.get("schema") != SEED_POLICY_SCHEMA:
            raise ValueError(f"Generation-1 seed authority must use {SEED_POLICY_SCHEMA}")
        if seed_policy.get("algorithm") != SEED_ALGORITHM:
            raise ValueError(f"Generation-1 seed authority algorithm must be {SEED_ALGORITHM}")
        list_fields = (
            "outer_seed_experiment_ids",
            "fixed_case_experiment_ids",
            "mechanics_only_experiment_ids",
            "variation_canary_outer_seeds",
        )
        for field in list_fields:
            value = seed_policy.get(field)
            if not isinstance(value, list) or len(value) != len(set(value)):
                raise ValueError(f"seed_authority.{field} must be a duplicate-free list")
        canary = seed_policy["variation_canary_outer_seeds"]
        if len(canary) != 5 or canary != seeds[:5]:
            raise ValueError("seed-authority canary must be exactly the first five frozen outer seeds")
        id_fields = (
            "outer_seed_experiment_ids",
            "fixed_case_experiment_ids",
            "mechanics_only_experiment_ids",
        )
        declared_sets = [set(seed_policy[field]) for field in id_fields]
        overlaps = (
            left & right
            for index, left in enumerate(declared_sets)
            for right in declared_sets[index + 1 :]
        )
        if any(overlaps):
            raise ValueError("seed-authority experiment modes must be disjoint")
        declared_ids = set().union(*declared_sets)
        missing_declared = sorted(declared_ids - known)
        if missing_declared:
            raise ValueError(f"seed authority references unknown experiments {missing_declared}")
        paths = seed_policy.get("experiment_seed_paths")
        if not isinstance(paths, dict):
            raise ValueError("seed_authority.experiment_seed_paths must be an object")
        for experiment_id, values in paths.items():
            if experiment_id not in known:
                raise ValueError(f"seed authority path references unknown experiment {experiment_id!r}")
            if (
                not isinstance(values, list)
                or not values
                or len(values) != len(set(values))
                or any(
                    not isinstance(value, str)
                    or not value.startswith("experiment.")
                    or ".." in value
                    for value in values
                )
            ):
                raise ValueError(f"seed paths for {experiment_id!r} are invalid")
    return payload


def eligible_experiment_ids(config: dict[str, Any]) -> list[str]:
    scope = config["experiment_scope"]
    tiers = set(scope["tiers"])
    excluded = set(scope.get("excluded_ids") or [])
    include_f = bool(scope.get("include_f_series"))
    mechanics = set(_seed_policy(config).get("mechanics_only_experiment_ids", []))
    include_wrappers = bool(scope.get("include_wrapper_smokes", True))
    return sorted(
        experiment_id
        for experiment_id, cls in REGISTRY.items()
        if cls.tier in tiers
        and experiment_id not in excluded
        and (include_f or not experiment_id.startswith("f"))
        and (include_wrappers or experiment_id not in mechanics)
    )


def _seed_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("seed_authority")
    if isinstance(policy, dict):
        return policy
    return {
        "schema": SEED_POLICY_SCHEMA,
        "algorithm": SEED_ALGORITHM,
        "outer_seed_experiment_ids": [],
        "fixed_case_experiment_ids": [],
        "mechanics_only_experiment_ids": [],
        "experiment_seed_paths": {},
        "variation_canary_outer_seeds": list(config.get("seeds", [])[:5]),
    }


def _seed_mode(config: dict[str, Any], experiment_id: str) -> tuple[str, str]:
    policy = _seed_policy(config)
    if experiment_id in policy.get("mechanics_only_experiment_ids", []):
        return SEED_MODE_MECHANICS, EVIDENCE_MECHANICS
    if experiment_id in policy.get("fixed_case_experiment_ids", []):
        return SEED_MODE_FIXED, EVIDENCE_FIXED
    return SEED_MODE_VARIED, EVIDENCE_INFERENTIAL


def _execution_seed(config: dict[str, Any], experiment_id: str, outer_seed: int) -> int:
    mode, _ = _seed_mode(config, experiment_id)
    return int(config["seeds"][0]) if mode in {SEED_MODE_FIXED, SEED_MODE_MECHANICS} else outer_seed


def _repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_authority(path: Path, *, role: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"authority must be a regular non-symlink file: {path}")
    row: dict[str, Any] = {"path": _repository_path(path), "sha256": sha256_file(path)}
    if role is not None:
        row["role"] = role
    return row


def _experiment_config_path(experiment_id: str) -> Path:
    return REPO_ROOT / "configs" / "experiment" / f"{experiment_id}.yaml"


def _implementation_authorities(experiment_id: str) -> list[dict[str, Any]]:
    cls = REGISTRY[experiment_id]
    source_name = inspect.getsourcefile(cls)
    if source_name is None:
        raise ValueError(f"cannot locate implementation source for {experiment_id}")
    paths = (
        ("experiment_source", Path(source_name)),
        ("experiment_harness", REPO_ROOT / "src/mop/harness/runner.py"),
        ("generation1_driver", Path(__file__)),
    )
    return [_file_authority(path.resolve(), role=role) for role, path in paths]


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


def _set_path_value(root: dict[str, Any], dotted: str, replacement: Any) -> None:
    components = dotted.split(".")
    if components and components[0] == "experiment":
        components = components[1:]
    if not components:
        raise ValueError(f"configured seed path is invalid: {dotted}")
    parent: Any = root
    for component in components[:-1]:
        if not isinstance(parent, dict) or component not in parent:
            raise ValueError(f"configured seed path is missing: {dotted}")
        parent = parent[component]
    if not isinstance(parent, dict) or components[-1] not in parent:
        raise ValueError(f"configured seed path is missing: {dotted}")
    parent[components[-1]] = replacement


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
    payload = {
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
    return int.from_bytes(hashlib.sha256(canonical_bytes(payload)).digest()[:8], "big") % MAX_EFFECTIVE_SEED


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
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
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
    raise ValueError(f"seed field {path} must be an integer or a nonempty integer list")


def _configured_seed_paths(
    config: dict[str, Any], experiment_id: str, experiment: dict[str, Any]
) -> list[str]:
    policy = _seed_policy(config)
    mode, _ = _seed_mode(config, experiment_id)
    if mode != SEED_MODE_VARIED or experiment_id in policy.get("outer_seed_experiment_ids", []):
        return []
    explicit = policy.get("experiment_seed_paths", {}).get(experiment_id)
    if explicit is not None:
        return list(explicit)
    if "seeds" in experiment:
        return ["experiment.seeds"]
    if "seed" in experiment:
        return ["experiment.seed"]
    raise ValueError(
        f"inferential experiment {experiment_id} has no consumed effective seed path or outer-seed authority"
    )


def apply_effective_seed_authority(
    cfg: DictConfig,
    *,
    config: dict[str, Any],
    experiment_id: str,
    outer_seed: int,
) -> dict[str, Any]:
    if outer_seed not in config["seeds"]:
        raise ValueError(f"outer seed {outer_seed} is outside the frozen campaign")
    plain_value = OmegaConf.to_container(cfg.experiment, resolve=True)
    if not isinstance(plain_value, dict):
        raise ValueError(f"experiment config for {experiment_id} is not a mapping")
    plain = cast(dict[str, Any], plain_value)
    mode, evidence_class = _seed_mode(config, experiment_id)
    original_outer = int(cfg.get("seed", 0))
    cfg.seed = outer_seed
    overrides: list[dict[str, Any]] = [
        {
            "role": "campaign_outer_seed",
            "path": "seed",
            "original": original_outer,
            "effective": outer_seed,
        }
    ]
    if mode == SEED_MODE_VARIED:
        for path in _configured_seed_paths(config, experiment_id, plain):
            original = _path_value(plain, path)
            effective = _effective_value(
                original,
                config=config,
                experiment_id=experiment_id,
                outer_seed=outer_seed,
                path=path,
            )
            _set_path_value(plain, path, effective)
            overrides.append(
                {
                    "role": "experiment_seed_control",
                    "path": path,
                    "original": original,
                    "effective": effective,
                }
            )
    cfg.experiment = OmegaConf.create(plain)
    execute_once = mode in {SEED_MODE_FIXED, SEED_MODE_MECHANICS}
    core = {
        "schema": SEED_AUTHORITY_SCHEMA,
        "algorithm": SEED_ALGORITHM,
        "policy_sha256": canonical_sha256(_seed_policy(config)),
        "campaign_id": config["campaign_id"],
        "result_tag": config["result_tag"],
        "experiment_id": experiment_id,
        "outer_seed": outer_seed,
        "mode": mode,
        "evidence_class": evidence_class,
        "effective_overrides": overrides,
        "execute_once": execute_once,
        "reference_outer_seed": int(config["seeds"][0]) if execute_once else outer_seed,
    }
    authority = {**core, "authority_sha256": canonical_sha256(core)}
    cfg.generation1_seed_authority = authority
    cfg.generation1_evidence_class = evidence_class
    return authority


def _compose_worker_config(
    *, experiment_id: str, outer_seed: int, result_tag: str, config: dict[str, Any]
) -> tuple[DictConfig, dict[str, Any]]:
    cfg = compose(
        [
            f"experiment={experiment_id}",
            "device=cpu",
            f"result_tag={result_tag}",
        ]
    )
    authority = apply_effective_seed_authority(
        cfg,
        config=config,
        experiment_id=experiment_id,
        outer_seed=outer_seed,
    )
    return cfg, authority


def _expected_cell_authority(
    config: dict[str, Any], experiment_id: str, outer_seed: int
) -> dict[str, Any]:
    _, seed_authority = _compose_worker_config(
        experiment_id=experiment_id,
        outer_seed=outer_seed,
        result_tag=str(config["result_tag"]),
        config=config,
    )
    return {
        "schema": CELL_AUTHORITY_SCHEMA,
        "evidence_class": seed_authority["evidence_class"],
        "seed_mode": seed_authority["mode"],
        "seed_authority": seed_authority,
        "experiment_config": _file_authority(_experiment_config_path(experiment_id)),
        "implementation_authorities": _implementation_authorities(experiment_id),
    }


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == canonical_sha256(core)


def _manifest_ok(
    path: Path,
    *,
    experiment_id: str,
    seed: int,
    result_tag: str,
    expected_cell_authority: dict[str, Any] | None = None,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    base_ok = bool(
        isinstance(payload, dict)
        and payload.get("name") == experiment_id
        and payload.get("seed") == seed
        and payload.get("status") == "ok"
        and payload.get("result_tag") == result_tag
        and isinstance(payload.get("metrics"), dict)
        and isinstance(payload.get("extra", {}).get("contract"), dict)
    )
    if not base_ok or expected_cell_authority is None:
        return base_ok
    cell = payload.get("extra", {}).get("generation1_cell_authority")
    if not isinstance(cell, dict):
        return False
    for field in (
        "schema",
        "evidence_class",
        "seed_mode",
        "seed_authority",
        "experiment_config",
        "implementation_authorities",
    ):
        if cell.get(field) != expected_cell_authority.get(field):
            return False
    config_snapshot = path.parent / "config.yaml"
    resolved = cell.get("resolved_config")
    if (
        not isinstance(resolved, dict)
        or resolved.get("path") != _repository_path(config_snapshot)
        or not config_snapshot.is_file()
        or config_snapshot.is_symlink()
        or resolved.get("sha256") != sha256_file(config_snapshot)
    ):
        return False
    attempt_path = path.parent / "attempt_receipt.json"
    try:
        attempt_receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(attempt_receipt, dict)
        or attempt_receipt.get("schema") != ATTEMPT_SCHEMA
        or not _valid_seal(attempt_receipt, "attempt_sha256")
        or attempt_receipt.get("experiment_id") != experiment_id
        or attempt_receipt.get("seed") != seed
        or attempt_receipt.get("seed_authority") != cell.get("seed_authority")
        or attempt_receipt.get("evidence_class") != cell.get("evidence_class")
        or attempt_receipt.get("seed_mode") != cell.get("seed_mode")
        or attempt_receipt.get("experiment_config") != cell.get("experiment_config")
        or attempt_receipt.get("implementation_authorities")
        != cell.get("implementation_authorities")
        or attempt_receipt.get("resolved_config") != resolved
    ):
        return False
    manifest_binding = attempt_receipt.get("manifest")
    return bool(
        isinstance(manifest_binding, dict)
        and manifest_binding.get("path") == _repository_path(path)
        and manifest_binding.get("sha256") == sha256_file(path)
    )


def _verified_attempt(
    class_root: Path,
    *,
    experiment_id: str,
    seed: int,
    result_tag: str,
    expected_cell_authority: dict[str, Any] | None = None,
) -> Path | None:
    for attempt in sorted(class_root.glob("attempt_[0-9][0-9][0-9]"), reverse=True):
        manifest = attempt / "manifest.json"
        if _manifest_ok(
            manifest,
            experiment_id=experiment_id,
            seed=seed,
            result_tag=result_tag,
            expected_cell_authority=expected_cell_authority,
        ):
            return attempt
    return None


def _next_attempt(class_root: Path) -> Path:
    used = [
        int(path.name.rsplit("_", 1)[-1])
        for path in class_root.glob("attempt_[0-9][0-9][0-9]")
    ]
    return class_root / f"attempt_{max(used, default=0) + 1:03d}"


@dataclass(frozen=True, slots=True)
class Attempt:
    experiment_id: str
    seed: int
    run_dir: str
    returncode: int | None
    timed_out: bool
    seconds: float
    stdout_tail: str
    stderr_tail: str
    evidence_class: str
    seed_mode: str
    seed_authority: dict[str, Any]
    experiment_config: dict[str, Any]
    implementation_authorities: list[dict[str, Any]]
    resolved_config: dict[str, Any] | None
    manifest: dict[str, Any] | None
    worker_report: dict[str, Any] | None


def _worker_report(stdout: str) -> dict[str, Any] | None:
    prefix = "GENERATION1_WORKER="
    for line in reversed(stdout.splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _run_subprocess(
    *,
    script: Path,
    experiment_id: str,
    seed: int,
    run_dir: Path,
    result_tag: str,
    timeout_seconds: float,
    mpl_config_dir: Path,
    config_path: Path,
    cell_authority: dict[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "worker",
        "--experiment",
        experiment_id,
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--result-tag",
        result_tag,
        "--config",
        str(config_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl_config_dir),
        }
    )
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    report = _worker_report(stdout)
    attempt = Attempt(
        experiment_id=experiment_id,
        seed=seed,
        run_dir=_repository_path(run_dir),
        returncode=returncode,
        timed_out=timed_out,
        seconds=round(time.perf_counter() - started, 6),
        stdout_tail=stdout[-MAX_TAIL:],
        stderr_tail=stderr[-MAX_TAIL:],
        evidence_class=str(cell_authority["evidence_class"]),
        seed_mode=str(cell_authority["seed_mode"]),
        seed_authority=dict(cell_authority["seed_authority"]),
        experiment_config=dict(cell_authority["experiment_config"]),
        implementation_authorities=list(cell_authority["implementation_authorities"]),
        resolved_config=(
            dict(report["resolved_config"])
            if report and report.get("resolved_config")
            else None
        ),
        manifest=(dict(report["manifest"]) if report and report.get("manifest") else None),
        worker_report=report,
    )
    receipt = _sealed(
        {
            "schema": ATTEMPT_SCHEMA,
            **asdict(attempt),
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "attempt_sha256",
    )
    receipt_path = run_dir / "attempt_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {
        **asdict(attempt),
        "attempt_receipt": {
            "path": _repository_path(receipt_path),
            "sha256": sha256_file(receipt_path),
            "attempt_sha256": receipt["attempt_sha256"],
        },
    }


def run_worker(
    experiment_id: str,
    seed: int,
    run_dir: Path,
    result_tag: str,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from ..harness.runner import run_experiment

    if experiment_id not in REGISTRY:
        raise KeyError(experiment_id)
    cls = REGISTRY[experiment_id]
    if cls.tier != "cpu-now":
        raise ValueError(f"Generation-1 worker only accepts cpu-now experiments, got {cls.tier}")
    config = load_config(config_path)
    if str(config["result_tag"]) != result_tag:
        raise ValueError("Generation-1 worker result tag differs from the frozen campaign")
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg, seed_authority = _compose_worker_config(
        experiment_id=experiment_id,
        outer_seed=seed,
        result_tag=result_tag,
        config=config,
    )
    expected = _expected_cell_authority(config, experiment_id, seed)
    if seed_authority != expected["seed_authority"]:
        raise ValueError("Generation-1 worker seed authority changed during composition")
    started = time.perf_counter()
    metrics = run_experiment(cfg, run_dir=run_dir)
    resolved_config = _file_authority(run_dir / "config.yaml")
    cell_authority = {
        **expected,
        "resolved_config": resolved_config,
        "scientific_metrics_sha256": _scientific_fingerprint(metrics),
    }
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("extra"), dict):
        raise ValueError("Generation-1 worker manifest is malformed")
    manifest["extra"]["generation1_cell_authority"] = cell_authority
    _atomic_json(manifest_path, manifest)
    manifest_binding = {
        "path": _repository_path(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum_rss *= 1024
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 6),
        "maximum_rss_bytes": maximum_rss,
        "metric_keys": sorted(metrics),
        "evidence_class": cell_authority["evidence_class"],
        "seed_mode": cell_authority["seed_mode"],
        "seed_authority": seed_authority,
        "experiment_config": cell_authority["experiment_config"],
        "implementation_authorities": cell_authority["implementation_authorities"],
        "resolved_config": resolved_config,
        "manifest": manifest_binding,
        "scientific_metrics_sha256": cell_authority["scientific_metrics_sha256"],
    }


def _cell_receipt(
    *,
    attempt: Path,
    experiment_id: str,
    requested_outer_seed: int,
    reference_outer_seed: int,
) -> dict[str, Any]:
    manifest_path = attempt / "manifest.json"
    attempt_path = attempt / "attempt_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    cell = manifest.get("extra", {}).get("generation1_cell_authority", {})
    return {
        "experiment_id": experiment_id,
        "requested_outer_seed": requested_outer_seed,
        "reference_outer_seed": reference_outer_seed,
        "execution": (
            "skipped_execute_once_reference"
            if requested_outer_seed != reference_outer_seed
            else "executed_or_resumed"
        ),
        "evidence_class": cell.get("evidence_class"),
        "seed_mode": cell.get("seed_mode"),
        "seed_authority": cell.get("seed_authority"),
        "experiment_config": cell.get("experiment_config"),
        "implementation_authorities": cell.get("implementation_authorities"),
        "resolved_config": cell.get("resolved_config"),
        "scientific_metrics_sha256": cell.get("scientific_metrics_sha256"),
        "manifest": {
            "path": _repository_path(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "attempt_receipt": {
            "path": _repository_path(attempt_path),
            "sha256": sha256_file(attempt_path),
            "attempt_sha256": receipt.get("attempt_sha256"),
            "self_seal_valid": _valid_seal(receipt, "attempt_sha256"),
        },
    }


def run_seed(
    *,
    config_path: Path,
    run_root: Path,
    seed: int,
    output: Path,
    max_workers: int,
    timeout_seconds: float,
    wall_seconds: float,
    script: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if seed not in config["seeds"]:
        raise ValueError(f"seed {seed} is outside the frozen Generation-1 seed set")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    ids = eligible_experiment_ids(config)
    seed_root = run_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = run_root / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    result_tag = str(config["result_tag"])
    strict_authority = isinstance(config.get("seed_authority"), dict)
    reference_seeds = {
        experiment_id: _execution_seed(config, experiment_id, seed) for experiment_id in ids
    }
    expected_authorities = {
        experiment_id: (
            _expected_cell_authority(config, experiment_id, reference_seed)
            if strict_authority
            else None
        )
        for experiment_id, reference_seed in reference_seeds.items()
    }
    verified_before = {
        experiment_id: _verified_attempt(
            run_root / f"seed_{reference_seeds[experiment_id]}" / "classes" / experiment_id,
            experiment_id=experiment_id,
            seed=reference_seeds[experiment_id],
            result_tag=result_tag,
            expected_cell_authority=expected_authorities[experiment_id],
        )
        for experiment_id in ids
    }
    pending = [
        experiment_id
        for experiment_id in ids
        if reference_seeds[experiment_id] == seed and verified_before[experiment_id] is None
    ]
    started = time.perf_counter()
    attempts: dict[str, dict[str, Any]] = {}
    submitted: list[str] = []
    batch_size = max_workers
    for offset in range(0, len(pending), batch_size):
        remaining = wall_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        batch = pending[offset : offset + batch_size]
        per_task_timeout = min(timeout_seconds, remaining)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for experiment_id in batch:
                class_root = seed_root / "classes" / experiment_id
                run_dir = _next_attempt(class_root)
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                submitted.append(experiment_id)
                future = pool.submit(
                    _run_subprocess,
                    script=script,
                    experiment_id=experiment_id,
                    seed=seed,
                    run_dir=run_dir,
                    result_tag=result_tag,
                    timeout_seconds=per_task_timeout,
                    mpl_config_dir=mpl_config_dir,
                    config_path=config_path,
                    cell_authority=expected_authorities[experiment_id]
                    or _expected_cell_authority(config, experiment_id, seed),
                )
                futures[future] = experiment_id
            for future in as_completed(futures):
                experiment_id = futures[future]
                try:
                    attempts[experiment_id] = future.result()
                except Exception as exc:  # all failures remain visible while independent rows continue
                    attempts[experiment_id] = {"orchestrator_error": f"{type(exc).__name__}: {exc}"}
    verified = {
        experiment_id: _verified_attempt(
            run_root / f"seed_{reference_seeds[experiment_id]}" / "classes" / experiment_id,
            experiment_id=experiment_id,
            seed=reference_seeds[experiment_id],
            result_tag=result_tag,
            expected_cell_authority=expected_authorities[experiment_id],
        )
        for experiment_id in ids
    }
    complete_ids = sorted(experiment_id for experiment_id, path in verified.items() if path is not None)
    remaining_ids = sorted(set(ids) - set(complete_ids))
    skipped_execute_once_ids = sorted(
        experiment_id
        for experiment_id in complete_ids
        if reference_seeds[experiment_id] != seed
    )
    cell_receipts: dict[str, dict[str, Any]] = {}
    if strict_authority:
        for experiment_id in complete_ids:
            attempt_path = verified[experiment_id]
            if attempt_path is None:
                continue
            cell_receipts[experiment_id] = _cell_receipt(
                attempt=attempt_path,
                experiment_id=experiment_id,
                requested_outer_seed=seed,
                reference_outer_seed=reference_seeds[experiment_id],
            )
    mode_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    for experiment_id in ids:
        mode, evidence = _seed_mode(config, experiment_id)
        mode_counts[mode] += 1
        evidence_counts[evidence] += 1
    receipt = _sealed(
        {
            "schema": SEED_SCHEMA,
            "campaign_id": config["campaign_id"],
            "claim_scope": config["claim_scope"],
            "seed": seed,
            "config": {
                "path": str(config_path.resolve().relative_to(REPO_ROOT)),
                "sha256": sha256_file(config_path),
            },
            "eligible_ids": ids,
            "eligible_count": len(ids),
            "resumed_ids": sorted(
                experiment_id for experiment_id, path in verified_before.items() if path is not None
            ),
            "submitted_ids": submitted,
            "executed_complete_ids": sorted(
                experiment_id
                for experiment_id in complete_ids
                if reference_seeds[experiment_id] == seed
            ),
            "skipped_execute_once_ids": skipped_execute_once_ids,
            "execute_once_reference_seed": int(config["seeds"][0]),
            "complete_ids": complete_ids,
            "complete_count": len(complete_ids),
            "remaining_ids": remaining_ids,
            "all_complete": not remaining_ids,
            "max_workers": max_workers,
            "per_experiment_timeout_seconds": timeout_seconds,
            "wall_budget_seconds": wall_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "attempts": attempts,
            "cell_receipts": cell_receipts,
            "seed_authority": {
                "schema": SEED_POLICY_SCHEMA,
                "policy_sha256": canonical_sha256(_seed_policy(config)),
                "outer_seed": seed,
                "mode_counts": dict(sorted(mode_counts.items())),
                "evidence_class_counts": dict(sorted(evidence_counts.items())),
            },
            "scientific_promotion": False,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "receipt_sha256",
    )
    _atomic_json(output, receipt)
    return receipt


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
        if not isinstance(key, str):
            continue
        child_prefix = f"{prefix}.{key}" if prefix else key
        flattened.update(_flatten_scalars(child, child_prefix, depth + 1))
    return flattened


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


def _scientific_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scientific_payload(child)
            for key, child in sorted(value.items())
            if isinstance(key, str) and key.lower() not in _NONSCIENTIFIC_METRIC_KEYS
        }
    if isinstance(value, list):
        return [_scientific_payload(child) for child in value]
    if isinstance(value, tuple):
        return [_scientific_payload(child) for child in value]
    return value


def _scientific_fingerprint(metrics: dict[str, Any]) -> str:
    return canonical_sha256(_scientific_payload(metrics))


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


def _experiment_summary(
    *,
    experiment_id: str,
    manifests: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    strict_authority = isinstance(config.get("seed_authority"), dict)
    seed_order = {int(seed): index for index, seed in enumerate(config["seeds"])}
    manifests = sorted(manifests, key=lambda row: seed_order.get(int(row.get("seed", -1)), 10**9))
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        cell = manifest.get("extra", {}).get("generation1_cell_authority")
        if isinstance(cell, dict) and isinstance(cell.get("seed_authority"), dict):
            authority = cell["seed_authority"]
            records.append(
                {
                    "seed": int(manifest["seed"]),
                    "evidence_class": cell.get("evidence_class"),
                    "seed_mode": cell.get("seed_mode"),
                    "authority_sha256": authority.get("authority_sha256"),
                    "effective_overrides": authority.get("effective_overrides"),
                    "scientific_metrics_sha256": cell.get("scientific_metrics_sha256")
                    or _scientific_fingerprint(manifest["metrics"]),
                    "manifest": manifest,
                }
            )
        else:
            records.append(
                {
                    "seed": int(manifest["seed"]),
                    "evidence_class": EVIDENCE_INFERENTIAL,
                    "seed_mode": SEED_MODE_VARIED,
                    "authority_sha256": f"legacy-outer-seed-{manifest['seed']}",
                    "effective_overrides": [{"path": "seed", "effective": manifest["seed"]}],
                    "scientific_metrics_sha256": _scientific_fingerprint(manifest["metrics"]),
                    "manifest": manifest,
                }
            )
    evidence_classes = {str(row["evidence_class"]) for row in records}
    seed_modes = {str(row["seed_mode"]) for row in records}
    expected_mode, expected_evidence = _seed_mode(config, experiment_id)
    seed_mode = next(iter(seed_modes)) if len(seed_modes) == 1 else "inconsistent"
    evidence_class = next(iter(evidence_classes)) if len(evidence_classes) == 1 else "inconsistent"
    if not strict_authority:
        seed_mode = SEED_MODE_VARIED
        evidence_class = EVIDENCE_INFERENTIAL

    unique_records: list[dict[str, Any]] = []
    seen_authorities: set[str] = set()
    seen_override_sets: set[str] = set()
    for row in records:
        digest = str(row["authority_sha256"])
        seen_override_sets.add(canonical_sha256(row["effective_overrides"]))
        if digest in seen_authorities:
            continue
        seen_authorities.add(digest)
        unique_records.append(row)

    canary_seeds = list(
        _seed_policy(config).get("variation_canary_outer_seeds", config["seeds"][:5])
    )
    canary_rows = [row for row in records if row["seed"] in canary_seeds]
    if not strict_authority:
        canary_status = "legacy_not_enforced"
        structurally_varied = len({row["authority_sha256"] for row in canary_rows}) >= 2
        scientifically_varied = len({row["scientific_metrics_sha256"] for row in canary_rows}) >= 2
    elif expected_mode != SEED_MODE_VARIED:
        canary_status = "not_applicable"
        structurally_varied = False
        scientifically_varied = False
    elif (
        len(canary_rows) != len(canary_seeds)
        or {row["seed"] for row in canary_rows} != set(canary_seeds)
    ):
        canary_status = "incomplete"
        structurally_varied = False
        scientifically_varied = False
    else:
        structurally_varied = (
            len({row["authority_sha256"] for row in canary_rows}) == len(canary_rows)
            and len({canonical_sha256(row["effective_overrides"]) for row in canary_rows})
            == len(canary_rows)
        )
        scientifically_varied = (
            len({row["scientific_metrics_sha256"] for row in canary_rows}) >= 2
        )
        if not structurally_varied:
            canary_status = "seed_authority_failed"
        elif not scientifically_varied:
            canary_status = "scientific_output_invariant"
        else:
            canary_status = "passed"

    boolean_values: dict[str, list[bool]] = defaultdict(list)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    for record in unique_records:
        manifest = record["manifest"]
        for key, value in _flatten_scalars(manifest["metrics"]).items():
            if isinstance(value, bool):
                boolean_values[key].append(value)
            else:
                numeric_values[key].append(value)
    nulls = boolean_values.get("null_supported", [])
    policy = config["classification"]
    threshold = float(policy["stable_fraction"])
    minimum = int(policy["minimum_boolean_observations"])
    minimum_complete = int(policy["minimum_complete_seeds"])
    effective_count = len(unique_records)
    structural_authority_ok = bool(
        seed_mode == expected_mode
        and evidence_class == expected_evidence
        and structurally_varied
        and len(records) == effective_count
        and len(records) == len(seen_override_sets)
    )
    if evidence_class == EVIDENCE_MECHANICS or expected_evidence == EVIDENCE_MECHANICS:
        classification = "mechanics_noninferential"
    elif evidence_class == EVIDENCE_FIXED or expected_evidence == EVIDENCE_FIXED:
        classification = "descriptive_fixed_case"
    elif strict_authority and not structural_authority_ok:
        classification = "descriptive_seed_adapter_unverified"
    elif strict_authority and canary_status == "scientific_output_invariant":
        classification = "descriptive_seed_invariant"
    elif strict_authority and canary_status != "passed":
        classification = "descriptive_seed_adapter_unverified"
    elif (
        (strict_authority and effective_count < minimum_complete)
        or len(nulls) < minimum
        or len(nulls) != effective_count
    ):
        classification = str(policy["missing_null_label"])
    else:
        null_fraction = sum(nulls) / len(nulls)
        if null_fraction >= threshold:
            classification = "stable_null"
        elif 1.0 - null_fraction >= threshold:
            classification = "stable_candidate_trace"
        else:
            classification = str(policy["tie_label"])
    contract = manifests[0]["extra"]["contract"] if manifests else {}
    expected_execution_count = (
        1 if expected_mode in {SEED_MODE_FIXED, SEED_MODE_MECHANICS} else len(config["seeds"])
    )
    return {
        "experiment_id": experiment_id,
        "completed_seed_count": len(manifests),
        "expected_execution_count": expected_execution_count,
        "effective_observation_count": effective_count,
        "distinct_seed_authority_count": len(seen_authorities),
        "distinct_effective_override_count": len(seen_override_sets),
        "coverage_complete": effective_count >= expected_execution_count,
        "minimum_directional_seed_count": minimum_complete,
        "directional_evidence_eligible": bool(
            evidence_class == EVIDENCE_INFERENTIAL
            and seed_mode == SEED_MODE_VARIED
            and (not strict_authority or canary_status == "passed")
            and effective_count >= minimum_complete
            and len(records) == effective_count
            and len(nulls) == effective_count
            and len(nulls) >= minimum
        ),
        "evidence_class": evidence_class,
        "seed_mode": seed_mode,
        "classification": classification,
        "variation_canary": {
            "outer_seeds": canary_seeds,
            "expected_count": len(canary_seeds),
            "observed_count": len(canary_rows),
            "status": canary_status,
            "structurally_varied": structurally_varied,
            "scientifically_varied": scientifically_varied,
            "scientific_metrics_sha256": {
                str(row["seed"]): row["scientific_metrics_sha256"] for row in canary_rows
            },
            "seed_authority_sha256": {
                str(row["seed"]): row["authority_sha256"] for row in canary_rows
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
        "contract": contract,
    }


def _operational_summary(run_root: Path, *, manifest_bytes: int) -> dict[str, Any]:
    attempt_directories = sorted(run_root.glob("seed_*/classes/*/attempt_[0-9][0-9][0-9]"))
    valid_receipts = 0
    invalid_receipts = 0
    wall_seconds = 0.0
    maximum_rss = 0
    by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for attempt in attempt_directories:
        if len(attempt.parts) >= 4:
            by_cell[(attempt.parts[-4], attempt.parts[-2])] += 1
        receipt_path = attempt / "attempt_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_receipts += 1
            continue
        if not isinstance(receipt, dict) or not _valid_seal(receipt, "attempt_sha256"):
            invalid_receipts += 1
            continue
        valid_receipts += 1
        seconds = receipt.get("seconds")
        if isinstance(seconds, int | float) and not isinstance(seconds, bool) and math.isfinite(seconds):
            wall_seconds += float(seconds)
        worker = receipt.get("worker_report")
        rss = worker.get("maximum_rss_bytes") if isinstance(worker, dict) else None
        if isinstance(rss, int) and not isinstance(rss, bool):
            maximum_rss = max(maximum_rss, rss)
    return {
        "attempt_directory_count": len(attempt_directories),
        "attempt_receipt_count": valid_receipts + invalid_receipts,
        "valid_attempt_receipt_count": valid_receipts,
        "invalid_attempt_receipt_count": invalid_receipts,
        "retry_count": sum(max(0, count - 1) for count in by_cell.values()),
        "summed_attempt_wall_seconds": round(wall_seconds, 6),
        "max_observed_worker_rss_bytes": maximum_rss or None,
        "manifest_bytes": manifest_bytes,
    }


def build_corpus(config_path: Path, run_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    ids = eligible_experiment_ids(config)
    strict_authority = isinstance(config.get("seed_authority"), dict)
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_authority_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_coverage: dict[str, Any] = {}
    total_manifest_bytes = 0
    seen_manifests: set[Path] = set()
    for seed in config["seeds"]:
        complete: list[str] = []
        skipped: list[str] = []
        executed: list[str] = []
        for experiment_id in ids:
            reference_seed = _execution_seed(config, experiment_id, seed)
            expected = (
                _expected_cell_authority(config, experiment_id, reference_seed)
                if strict_authority
                else None
            )
            attempt = _verified_attempt(
                run_root / f"seed_{reference_seed}" / "classes" / experiment_id,
                experiment_id=experiment_id,
                seed=reference_seed,
                result_tag=config["result_tag"],
                expected_cell_authority=expected,
            )
            if attempt is None:
                continue
            manifest_path = attempt / "manifest.json"
            complete.append(experiment_id)
            if reference_seed != seed:
                skipped.append(experiment_id)
            else:
                executed.append(experiment_id)
            if manifest_path not in seen_manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                by_experiment[experiment_id].append(manifest)
                total_manifest_bytes += manifest_path.stat().st_size
                seen_manifests.add(manifest_path)
                if strict_authority:
                    cell_authority_index[experiment_id].append(
                        _cell_receipt(
                            attempt=attempt,
                            experiment_id=experiment_id,
                            requested_outer_seed=reference_seed,
                            reference_outer_seed=reference_seed,
                        )
                    )
        seed_coverage[str(seed)] = {
            "complete_count": len(complete),
            "expected_count": len(ids),
            "complete": len(complete) == len(ids),
            "missing_ids": sorted(set(ids) - set(complete)),
            "executed_or_resumed_ids": executed,
            "skipped_execute_once_ids": skipped,
        }
    summaries = {
        experiment_id: _experiment_summary(
            experiment_id=experiment_id,
            manifests=by_experiment[experiment_id],
            config=config,
        )
        for experiment_id in ids
    }
    domain_summaries: dict[str, Any] = {}
    for domain, members in config["capability_packs"].items():
        rows = [summaries[experiment_id] for experiment_id in members]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["classification"]] += 1
        domain_summaries[domain] = {
            "experiment_ids": members,
            "classification_counts": dict(sorted(counts.items())),
            "minimum_seed_coverage": min(row["effective_observation_count"] for row in rows),
            "complete_experiment_count": sum(row["coverage_complete"] is True for row in rows),
        }
    minimum_complete = int(config["classification"]["minimum_complete_seeds"])
    complete_experiments = [
        experiment_id
        for experiment_id, row in summaries.items()
        if row["coverage_complete"] is True
        and (
            row["seed_mode"] in {SEED_MODE_FIXED, SEED_MODE_MECHANICS}
            or row["effective_observation_count"] >= minimum_complete
        )
    ]
    mode_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    for row in summaries.values():
        mode_counts[str(row["seed_mode"])] += 1
        evidence_counts[str(row["evidence_class"])] += 1
    varied_rows = [row for row in summaries.values() if row["seed_mode"] == SEED_MODE_VARIED]
    structural_canary_failures = sorted(
        row["experiment_id"]
        for row in varied_rows
        if row["variation_canary"]["status"] != "legacy_not_enforced"
        and row["variation_canary"]["structurally_varied"] is not True
    )
    scientific_output_invariants = sorted(
        row["experiment_id"]
        for row in varied_rows
        if row["variation_canary"]["status"] == "scientific_output_invariant"
    )
    seed_authority_summary = {
        "policy": _seed_policy(config),
        "policy_sha256": canonical_sha256(_seed_policy(config)),
        "mode_counts": dict(sorted(mode_counts.items())),
        "evidence_class_counts": dict(sorted(evidence_counts.items())),
        "varied_experiment_count": len(varied_rows),
        "varied_with_five_distinct_canary_authorities": sum(
            row["variation_canary"]["structurally_varied"] is True for row in varied_rows
        ),
        "canary_failures": structural_canary_failures,
        "structural_canary_failures": structural_canary_failures,
        "scientific_output_invariant_ids": scientific_output_invariants,
        "no_pseudoreplication": all(
            row["completed_seed_count"] == row["distinct_seed_authority_count"]
            and row["completed_seed_count"] == row["distinct_effective_override_count"]
            and row["variation_canary"]["structurally_varied"] is True
            for row in varied_rows
        ),
    }
    core = {
        "schema": CORPUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": config["claim_scope"],
        "config": {
            "path": str(config_path.resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(config_path),
        },
        "run_root": str(run_root.resolve().relative_to(REPO_ROOT)),
        "seed_count": len(config["seeds"]),
        "seeds": config["seeds"],
        "eligible_experiment_count": len(ids),
        "eligible_experiment_ids": ids,
        "minimum_complete_seeds": minimum_complete,
        "complete_experiment_count": len(complete_experiments),
        "complete_experiment_ids": complete_experiments,
        "corpus_complete": len(complete_experiments) == len(ids),
        "seed_coverage": seed_coverage,
        "experiment_summaries": summaries,
        "capability_pack_summaries": domain_summaries,
        "cell_authority_index": dict(sorted(cell_authority_index.items())),
        "seed_authority_summary": seed_authority_summary,
        "operational_summary": _operational_summary(
            run_root, manifest_bytes=total_manifest_bytes
        ),
        "total_manifest_bytes": total_manifest_bytes,
        "evidence_interpretation": {
            "stable_candidate_trace": (
                "candidate defeated its own registered null on at least the frozen fraction of "
                "available outer seeds; this remains synthetic exploratory evidence"
            ),
            "stable_null": (
                "registered null was supported on at least the frozen fraction of available outer seeds"
            ),
            "mixed_or_seed_sensitive": "outer-seed direction was unstable and cannot support promotion",
            "descriptive_only": (
                "the experiment does not expose enough top-level null_supported observations for "
                "directional aggregation"
            ),
        },
        "scientific_promotion": False,
    }
    return _sealed(core, "corpus_sha256")


def write_corpus(config_path: Path, run_root: Path, output: Path) -> dict[str, Any]:
    corpus = build_corpus(config_path, run_root)
    _atomic_json(output, corpus)
    return corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--experiment", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--result-tag", required=True)
    worker.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = subparsers.add_parser("run-seed")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-workers", type=int, default=6)
    run.add_argument("--timeout-seconds", type=float, default=900.0)
    run.add_argument("--wall-seconds", type=float, default=21_600.0)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    aggregate.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    aggregate.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "worker":
        result = run_worker(
            arguments.experiment,
            arguments.seed,
            arguments.run_dir,
            arguments.result_tag,
            arguments.config.resolve(),
        )
        print("GENERATION1_WORKER=" + json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "run-seed":
        result = run_seed(
            config_path=arguments.config.resolve(),
            run_root=arguments.run_root.resolve(),
            seed=arguments.seed,
            output=arguments.out.resolve(),
            max_workers=arguments.max_workers,
            timeout_seconds=arguments.timeout_seconds,
            wall_seconds=arguments.wall_seconds,
            script=(REPO_ROOT / "scripts/generation1_cognitive_corpus.py").resolve(),
        )
        print(json.dumps({key: result[key] for key in ("seed", "complete_count", "all_complete")}, indent=2))
        return 0 if result["all_complete"] else 2
    corpus = write_corpus(arguments.config.resolve(), arguments.run_root.resolve(), arguments.out.resolve())
    print(
        json.dumps(
            {
                "corpus_complete": corpus["corpus_complete"],
                "complete_experiment_count": corpus["complete_experiment_count"],
                "eligible_experiment_count": corpus["eligible_experiment_count"],
                "output": str(arguments.out),
            },
            indent=2,
        )
    )
    return 0 if corpus["corpus_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
