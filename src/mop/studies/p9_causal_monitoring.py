
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT
from ..substrate.events import canonical_sha256
from .p9_accounting import WorkloadAccountant
from .process_resources import PeakRSSMonitor
from .runtime_integrity import (
    FORBIDDEN_MODEL_MODULES,
    deny_forbidden_runtime_imports,
    forbidden_source_imports,
)

CONFIG_SCHEMA = "mop-p9-causal-monitoring-config/v1"
DATASET_SCHEMA = "mop-p9-causal-monitoring-dataset/v1"
ROW_SCHEMA = "mop-p9-causal-monitoring-row/v1"
CHECKPOINT_SCHEMA = "mop-p9-causal-monitoring-checkpoint/v1"
PREFLIGHT_SCHEMA = "mop-p9-causal-monitoring-preflight/v1"
CLAIM_SCOPE = (
    "deterministic structural-fixture mechanics only; no natural-workload, capability, cognition, "
    "sentience, or energy claim"
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "p9_causal_monitoring_preflight.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "P9_CAUSAL_MONITORING_PREFLIGHT.json"

HISTOGRAM_ARMS = (
    "causal_abstraction",
    "correlational_proxy",
    "external_error_only",
    "final_output_decoder",
    "random_telemetry",
    "telemetry_lesion",
    "telemetry_shuffled",
)
STRUCTURAL_CONTROLS = ("fixed_threshold_pid", "oracle_structural")
ARM_ORDER = (*HISTOGRAM_ARMS, *STRUCTURAL_CONTROLS)
SPLITS = ("train", "calibration", "heldout")

IMPLEMENTATION_PATHS = (
    "configs/experiment/p9_causal_monitoring_preflight.yaml",
    "src/mop/studies/p9_causal_monitoring.py",
    "scripts/p9_causal_monitoring_preflight.py",
    "tests/unit/test_p9_causal_monitoring.py",
    "docs/P9_CAUSAL_MONITORING_PREFLIGHT.md",
    "src/mop/studies/process_resources.py",
    "src/mop/studies/runtime_integrity.py",
)
UPSTREAM_PATHS = (
    "src/mop/studies/p9_accounting.py",
    "scripts/p9_accounting_mechanics.py",
    "tests/unit/test_p9_accounting.py",
    "proof/P9_ACCOUNTING_MECHANICS.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)


def _fraction(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 8),
    }


def _from_fraction(payload: dict[str, Any]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def _mean(values: list[Fraction]) -> Fraction:
    return sum(values, Fraction()) / max(1, len(values))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_tree(item) for item in value)
    return True


def _digest_bytes(*parts: Any) -> bytes:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).digest()


def _clip(value: int, maximum: int = 8) -> int:
    return max(0, min(maximum, value))


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("P9 causal-monitoring config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("P9 causal-monitoring claim scope drift")
    if not str(config.get("null_hypothesis", "")).strip():
        raise ValueError("P9 causal-monitoring null hypothesis is required")
    fixture = config.get("fixture", {})
    if int(fixture.get("causal_bins", 0)) != 8:
        raise ValueError("P9 causal-monitoring preflight requires exactly eight bins")
    if tuple(int(value) for value in fixture.get("smoothing_candidates", ())) != (1, 2, 4):
        raise ValueError("P9 causal-monitoring smoothing candidates drift")
    interventions = fixture.get("interventions", ())
    names = tuple(str(row.get("name")) for row in interventions)
    expected_names = (
        "observational",
        "queue_pressure",
        "memory_pressure",
        "retry_pressure",
        "resource_relief",
    )
    if names != expected_names or len(set(names)) != len(names):
        raise ValueError("P9 causal-monitoring intervention set drift")
    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("histogram_arms", ())) != HISTOGRAM_ARMS:
        raise ValueError("P9 causal-monitoring matched histogram arm set drift")
    if tuple(evaluation.get("structural_controls", ())) != STRUCTURAL_CONTROLS:
        raise ValueError("P9 causal-monitoring structural control set drift")
    if evaluation.get("primary_arm") != "causal_abstraction":
        raise ValueError("P9 causal-monitoring primary arm drift")
    envelope = config.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or envelope.get("accelerator_required") is not False
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("model_downloads_allowed") is not False
        or envelope.get("external_data_allowed") is not False
    ):
        raise ValueError("P9 causal-monitoring preflight must remain one-thread CPU and self-contained")
    units = fixture.get("independent_units", ())
    minimum = int(config.get("stop_contract", {}).get("minimum_independent_units", 0))
    if len(units) < minimum:
        raise ValueError("P9 causal-monitoring independent-unit minimum is unmet")
    if len({str(row.get("unit_id")) for row in units}) != len(units):
        raise ValueError("P9 causal-monitoring unit ids must be unique")
    if len({int(row.get("seed", -1)) for row in units}) != len(units):
        raise ValueError("P9 causal-monitoring unit seeds must be unique")
    return config


def _construction(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_payload": copy.deepcopy(config),
        "config_payload_sha256": canonical_sha256(config),
        "generator": {
            "name": "p9-bounded-structural-failure-scm",
            "version": 1,
            "causal_equation": "failure := 2*queue + 2*memory + 2*retry + shock >= threshold",
            "telemetry_boundary": "queue, memory, and retry are visible; shock is hidden",
            "confounder_shift": "aligned proxy in train/calibration, reversed proxy in heldout",
        },
    }


def _split_count(config: dict[str, Any], split: str) -> int:
    key = {
        "train": "train_lineages_per_unit",
        "calibration": "calibration_lineages_per_unit",
        "heldout": "heldout_lineages_per_unit",
    }[split]
    return int(config["fixture"][key])


def _lineage_offset(split: str) -> int:
    return {"train": 0, "calibration": 10_000, "heldout": 20_000}[split]


def _base_state(unit: dict[str, Any], split: str, lineage_index: int) -> dict[str, int]:
    lineage_number = _lineage_offset(split) + lineage_index
    digest = _digest_bytes(unit["unit_id"], int(unit["seed"]), split, lineage_number)
    return {
        "queue": int(digest[0] % 6),
        "memory": int(digest[1] % 6),
        "retry": int(digest[2] % 3),
        "shock": int(digest[3] % 5) - 2,
        "lineage_number": lineage_number,
    }


def _causal_visible_score(state: dict[str, int]) -> int:
    return 2 * int(state["queue"]) + 2 * int(state["memory"]) + 2 * int(state["retry"])


def _apply_intervention(state: dict[str, int], intervention: dict[str, Any]) -> dict[str, int]:
    return {
        "queue": _clip(int(state["queue"]) + int(intervention["queue_delta"])),
        "memory": _clip(int(state["memory"]) + int(intervention["memory_delta"])),
        "retry": _clip(int(state["retry"]) + int(intervention["retry_delta"]), maximum=5),
    }


def _undigested_branch(
    config: dict[str, Any],
    unit: dict[str, Any],
    split: str,
    lineage_index: int,
    intervention: dict[str, Any],
) -> dict[str, Any]:
    base = _base_state(unit, split, lineage_index)
    post = _apply_intervention(base, intervention)
    threshold = int(unit["failure_threshold"])
    visible_score = _causal_visible_score(post)
    structural_risk = visible_score + int(base["shock"])
    failure = structural_risk >= threshold
    base_visible_score = _causal_visible_score(base)
    raw_proxy = int(base_visible_score + int(base["shock"]) >= threshold - 2)
    shifted_proxy = raw_proxy if split != "heldout" else 1 - raw_proxy
    digest = _digest_bytes(unit["unit_id"], split, base["lineage_number"], intervention["name"], "telemetry")
    lineage_id = f"lineage:p9/{unit['unit_id']}/{split}/{base['lineage_number']:05d}"
    parent_ref = f"state:p9/{unit['unit_id']}/{split}/{base['lineage_number']:05d}"
    branch_ref = f"branch:p9/{unit['unit_id']}/{split}/{base['lineage_number']:05d}/{intervention['name']}"
    causal_bin = min(int(config["fixture"]["causal_bins"]) - 1, visible_score // 4)
    lesion_score = 2 * int(post["queue"]) + 2 * int(post["retry"])
    return {
        "schema": ROW_SCHEMA,
        "unit_id": str(unit["unit_id"]),
        "split": split,
        "lineage_id": lineage_id,
        "parent_state_ref": parent_ref,
        "branch_ref": branch_ref,
        "intervention": {
            "name": str(intervention["name"]),
            "queue_delta": int(intervention["queue_delta"]),
            "memory_delta": int(intervention["memory_delta"]),
            "retry_delta": int(intervention["retry_delta"]),
        },
        "base_state": {
            "queue": int(base["queue"]),
            "memory": int(base["memory"]),
            "retry": int(base["retry"]),
        },
        "post_intervention_state": post,
        "exogenous": {"shock": int(base["shock"])},
        "failure_threshold": threshold,
        "telemetry": {
            "queue_depth": int(post["queue"]),
            "memory_pressure": int(post["memory"]),
            "retry_debt": int(post["retry"]),
            "causal_abstraction_bin": causal_bin,
            "external_error_lag": min(7, max(0, base_visible_score + int(base["shock"]) - threshold + 4)),
            "correlational_proxy": shifted_proxy,
            "proxy_regime": "aligned" if split != "heldout" else "reversed",
            "final_output_token": int(shifted_proxy * 4 + digest[4] % 4),
            "random_channel": int(digest[5] % 8),
            "telemetry_lesion_bin": min(7, lesion_score // 3),
            "telemetry_shuffled_bin": None,
            "shuffle_source_branch_ref": None,
        },
        "outcome": {
            "failure_within_horizon": failure,
            "structural_risk": structural_risk,
            "visible_risk": visible_score,
            "risk_margin": structural_risk - threshold,
            "failure_tick": max(1, 4 - min(3, structural_risk - threshold)) if failure else None,
        },
        "causal_contract": {
            "proxy_is_parent_fixed": True,
            "shock_is_hidden_from_monitor": True,
            "outcome_uses_post_intervention_state": True,
        },
    }


def _generate_chunk(construction: dict[str, Any], unit_index: int, split: str) -> dict[str, Any]:
    config = construction["config_payload"]
    unit = config["fixture"]["independent_units"][unit_index]
    groups: list[dict[str, Any]] = []
    for lineage_index in range(_split_count(config, split)):
        rows = [
            _undigested_branch(config, unit, split, lineage_index, intervention)
            for intervention in config["fixture"]["interventions"]
        ]
        groups.append(
            {
                "lineage_id": rows[0]["lineage_id"],
                "parent_state_ref": rows[0]["parent_state_ref"],
                "base_state": rows[0]["base_state"],
                "exogenous": rows[0]["exogenous"],
                "branches": rows,
            }
        )

    flat = [row for group in groups for row in group["branches"]]
    shift = max(1, len(config["fixture"]["interventions"]) + 2)
    for index, row in enumerate(flat):
        source = flat[(index + shift) % len(flat)]
        row["telemetry"]["telemetry_shuffled_bin"] = source["telemetry"]["causal_abstraction_bin"]
        row["telemetry"]["shuffle_source_branch_ref"] = source["branch_ref"]
        row["row_sha256"] = canonical_sha256(row)
    for group in groups:
        group["group_sha256"] = canonical_sha256(group)
    payload: dict[str, Any] = {
        "chunk_id": f"{unit['unit_id']}:{split}",
        "unit": copy.deepcopy(unit),
        "split": split,
        "lineage_count": len(groups),
        "branch_count": len(flat),
        "groups": groups,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _chunk_specs(construction: dict[str, Any]) -> list[tuple[int, str]]:
    units = construction["config_payload"]["fixture"]["independent_units"]
    return [(index, split) for index in range(len(units)) for split in SPLITS]


def _assemble_dataset(construction: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    config = construction["config_payload"]
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    units: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(config["fixture"]["independent_units"]):
        splits = {split: copy.deepcopy(by_id[f"{unit['unit_id']}:{split}"]) for split in SPLITS}
        unit_payload: dict[str, Any] = {
            "unit": copy.deepcopy(unit),
            "unit_index": unit_index,
            "splits": splits,
        }
        unit_payload["payload_sha256"] = canonical_sha256(unit_payload)
        units.append(unit_payload)

    intervention_count = len(config["fixture"]["interventions"])
    lineages_per_unit = sum(_split_count(config, split) for split in SPLITS)
    dataset: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "construction": copy.deepcopy(construction),
        "budget_contract": {
            "independent_units": len(units),
            "chunks": len(chunks),
            "lineages_per_unit": lineages_per_unit,
            "branches_per_lineage": intervention_count,
            "total_lineages": len(units) * lineages_per_unit,
            "total_branches": len(units) * lineages_per_unit * intervention_count,
            "split_lineages_per_unit": {split: _split_count(config, split) for split in SPLITS},
        },
        "units": units,
    }
    dataset["payload_sha256"] = canonical_sha256(dataset)
    return dataset


def build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    construction = _construction(config)
    chunks = [
        _generate_chunk(construction, unit_index, split) for unit_index, split in _chunk_specs(construction)
    ]
    return _assemble_dataset(construction, chunks)


def _checkpoint_chain(records: list[dict[str, Any]]) -> str:
    chain = "0" * 64
    for record in records:
        chain = canonical_sha256(
            {
                "previous": chain,
                "chunk_id": record["chunk_id"],
                "payload_sha256": record["payload_sha256"],
            }
        )
    return chain


def _validate_checkpoint(checkpoint: dict[str, Any], construction: dict[str, Any]) -> None:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("P9 causal-monitoring checkpoint schema drift")
    if checkpoint.get("construction_sha256") != canonical_sha256(construction):
        raise ValueError("P9 causal-monitoring checkpoint construction mismatch")
    specs = _chunk_specs(construction)
    expected_ids = [
        f"{construction['config_payload']['fixture']['independent_units'][index]['unit_id']}:{split}"
        for index, split in specs
    ]
    records = checkpoint.get("completed_chunks", [])
    observed_ids = [record.get("chunk_id") for record in records]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise ValueError("P9 causal-monitoring checkpoint chunk order mismatch")
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("P9 causal-monitoring checkpoint chunk payload is invalid")
        payload_core = {key: value for key, value in payload.items() if key != "payload_sha256"}
        if record.get("payload_sha256") != canonical_sha256(payload_core):
            raise ValueError("P9 causal-monitoring checkpoint chunk digest mismatch")
        if payload.get("payload_sha256") != record["payload_sha256"]:
            raise ValueError("P9 causal-monitoring nested chunk digest mismatch")
    if checkpoint.get("chain_sha256") != _checkpoint_chain(records):
        raise ValueError("P9 causal-monitoring checkpoint chain mismatch")


def build_dataset_resumable(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    stop_after_chunks: int | None = None,
) -> dict[str, Any] | None:
    construction = _construction(config)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _validate_checkpoint(checkpoint, construction)
    else:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "construction_sha256": canonical_sha256(construction),
            "completed_chunks": [],
            "chain_sha256": "0" * 64,
        }
        _atomic_json(checkpoint_path, checkpoint)

    specs = _chunk_specs(construction)
    while len(checkpoint["completed_chunks"]) < len(specs):
        index = len(checkpoint["completed_chunks"])
        unit_index, split = specs[index]
        payload = _generate_chunk(construction, unit_index, split)
        checkpoint["completed_chunks"].append(
            {
                "chunk_id": payload["chunk_id"],
                "payload_sha256": payload["payload_sha256"],
                "payload": payload,
            }
        )
        checkpoint["chain_sha256"] = _checkpoint_chain(checkpoint["completed_chunks"])
        _atomic_json(checkpoint_path, checkpoint)
        if stop_after_chunks is not None and len(checkpoint["completed_chunks"]) >= stop_after_chunks:
            return None

    dataset = _assemble_dataset(
        construction,
        [record["payload"] for record in checkpoint["completed_chunks"]],
    )
    checkpoint["final_dataset_sha256"] = dataset["payload_sha256"]
    _atomic_json(checkpoint_path, checkpoint)
    return dataset


def _row_without_digest(row: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(row)
    payload.pop("row_sha256", None)
    return payload


def verify_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    try:
        construction = dataset["construction"]
        config = construction["config_payload"]
        _load_config_payload(config)
        expected = build_dataset(config)
        checks["exact_deterministic_rebuild"] = canonical_sha256(dataset) == canonical_sha256(expected)
        checks["dataset_payload_digest"] = dataset.get("payload_sha256") == canonical_sha256(
            {key: value for key, value in dataset.items() if key != "payload_sha256"}
        )
        checks["construction_digest"] = construction.get("config_payload_sha256") == canonical_sha256(config)
        row_digests = True
        disjoint = True
        complete = True
        same_parent = True
        intervention_equations = True
        confounder_shift = True
        shuffle_nonidentity = True
        outcome_classes = True
        expected_interventions = {row["name"] for row in config["fixture"]["interventions"]}
        for unit in dataset["units"]:
            lineage_sets: dict[str, set[str]] = {}
            for split in SPLITS:
                groups = unit["splits"][split]["groups"]
                lineage_sets[split] = {str(group["lineage_id"]) for group in groups}
                outcomes: set[bool] = set()
                for group in groups:
                    rows = group["branches"]
                    complete &= {row["intervention"]["name"] for row in rows} == expected_interventions
                    same_parent &= len({row["parent_state_ref"] for row in rows}) == 1
                    same_parent &= len({canonical_sha256(row["base_state"]) for row in rows}) == 1
                    same_parent &= len({canonical_sha256(row["exogenous"]) for row in rows}) == 1
                    for row in rows:
                        row_digests &= row.get("row_sha256") == canonical_sha256(_row_without_digest(row))
                        post = _apply_intervention(row["base_state"], row["intervention"])
                        risk = _causal_visible_score(post) + int(row["exogenous"]["shock"])
                        intervention_equations &= post == row["post_intervention_state"]
                        intervention_equations &= risk == int(row["outcome"]["structural_risk"])
                        intervention_equations &= bool(risk >= int(row["failure_threshold"])) == bool(
                            row["outcome"]["failure_within_horizon"]
                        )
                        expected_regime = "reversed" if split == "heldout" else "aligned"
                        confounder_shift &= row["telemetry"]["proxy_regime"] == expected_regime
                        shuffle_nonidentity &= (
                            row["telemetry"]["shuffle_source_branch_ref"] != row["branch_ref"]
                        )
                        outcomes.add(bool(row["outcome"]["failure_within_horizon"]))
                outcome_classes &= outcomes == {False, True}
            disjoint &= not (lineage_sets["train"] & lineage_sets["calibration"])
            disjoint &= not (lineage_sets["train"] & lineage_sets["heldout"])
            disjoint &= not (lineage_sets["calibration"] & lineage_sets["heldout"])
        checks.update(
            {
                "row_digests": row_digests,
                "disjoint_split_lineages": disjoint,
                "complete_intervention_sets": complete,
                "same_parent_branches": same_parent,
                "intervention_equations": intervention_equations,
                "declared_confounder_shift": confounder_shift,
                "shuffled_channel_nonidentity": shuffle_nonidentity,
                "both_outcome_classes_per_split": outcome_classes,
            }
        )
    except Exception as exc:
        errors.append(f"verification exception: {exc}")
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"verified": not errors and all(checks.values()), "checks": checks, "errors": errors}


def _load_config_payload(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("embedded P9 causal-monitoring config drift")


def mutation_suite(dataset: dict[str, Any]) -> dict[str, Any]:
    mutations: dict[str, dict[str, Any]] = {}

    def reject(name: str, mutate: Any) -> None:
        changed = copy.deepcopy(dataset)
        mutate(changed)
        audit = verify_dataset(changed)
        mutations[name] = {"rejected": audit["verified"] is False, "errors": audit["errors"][:4]}

    def first(value: dict[str, Any]) -> dict[str, Any]:
        return value["units"][0]["splits"]["heldout"]["groups"][0]["branches"][0]

    reject(
        "future_outcome",
        lambda value: first(value)["outcome"].__setitem__(
            "failure_within_horizon", not first(value)["outcome"]["failure_within_horizon"]
        ),
    )
    reject(
        "intervention_delta",
        lambda value: first(value)["intervention"].__setitem__("queue_delta", 7),
    )
    reject(
        "causal_telemetry",
        lambda value: first(value)["telemetry"].__setitem__("queue_depth", 99),
    )
    reject(
        "correlational_proxy",
        lambda value: first(value)["telemetry"].__setitem__(
            "correlational_proxy", 1 - first(value)["telemetry"]["correlational_proxy"]
        ),
    )
    reject(
        "shuffle_provenance",
        lambda value: first(value)["telemetry"].__setitem__(
            "shuffle_source_branch_ref", first(value)["branch_ref"]
        ),
    )
    reject(
        "lineage_identity",
        lambda value: value["units"][0]["splits"]["heldout"]["groups"][0].__setitem__(
            "lineage_id",
            value["units"][0]["splits"]["train"]["groups"][0]["lineage_id"],
        ),
    )
    reject("row_digest", lambda value: first(value).__setitem__("row_sha256", "0" * 64))
    reject("dataset_digest", lambda value: value.__setitem__("payload_sha256", "f" * 64))
    return {
        "mutations": mutations,
        "count": len(mutations),
        "rejected": sum(row["rejected"] is True for row in mutations.values()),
        "all_rejected": all(row["rejected"] is True for row in mutations.values()),
    }


def _rows(unit: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [row for group in unit["splits"][split]["groups"] for row in group["branches"]]


def _feature_bin(arm: str, row: dict[str, Any]) -> int:
    telemetry = row["telemetry"]
    if arm == "causal_abstraction":
        return int(telemetry["causal_abstraction_bin"])
    if arm == "correlational_proxy":
        return int(telemetry["correlational_proxy"]) * 4
    if arm == "external_error_only":
        return int(telemetry["external_error_lag"])
    if arm == "final_output_decoder":
        return int(telemetry["final_output_token"])
    if arm == "random_telemetry":
        return int(telemetry["random_channel"])
    if arm == "telemetry_lesion":
        return int(telemetry["telemetry_lesion_bin"])
    if arm == "telemetry_shuffled":
        return int(telemetry["telemetry_shuffled_bin"])
    raise ValueError(f"unsupported histogram arm {arm!r}")


def _table_from_counts(counts: list[list[int]], alpha: int) -> list[Fraction]:
    return [Fraction(positive + alpha, total + 2 * alpha) for positive, total in counts]


def _brier(rows: list[dict[str, Any]], predictions: list[Fraction]) -> Fraction:
    losses = [
        (prediction - int(row["outcome"]["failure_within_horizon"])) ** 2
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    return _mean(losses)


def _fit_histogram(
    arm: str,
    train: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    bins = int(config["fixture"]["causal_bins"])
    counts = [[0, 0] for _ in range(bins)]
    for row in train:
        feature = _feature_bin(arm, row)
        counts[feature][0] += int(row["outcome"]["failure_within_horizon"])
        counts[feature][1] += 1
    candidates: list[tuple[Fraction, int, list[Fraction]]] = []
    for alpha in (int(value) for value in config["fixture"]["smoothing_candidates"]):
        table = _table_from_counts(counts, alpha)
        predictions = [table[_feature_bin(arm, row)] for row in calibration]
        candidates.append((_brier(calibration, predictions), alpha, table))
    calibration_brier, alpha, table = min(candidates, key=lambda value: (value[0], value[1]))
    return {
        "arm": arm,
        "kind": "matched_histogram",
        "selected_smoothing_alpha": alpha,
        "calibration_brier": _fraction(calibration_brier),
        "table": [
            {
                "bin": index,
                "positive_count": counts[index][0],
                "total_count": counts[index][1],
                "probability": _fraction(probability),
            }
            for index, probability in enumerate(table)
        ],
        "cost": {
            "maximum_bins": bins,
            "stored_integer_counts": bins * 2,
            "train_feature_evaluations": len(train),
            "calibration_feature_evaluations": len(calibration)
            * len(config["fixture"]["smoothing_candidates"]),
            "smoothing_candidates": len(config["fixture"]["smoothing_candidates"]),
            "trainable_parameters": 0,
        },
    }


def _predict(model: dict[str, Any], row: dict[str, Any]) -> Fraction:
    arm = str(model["arm"])
    if arm in HISTOGRAM_ARMS:
        payload = model["table"][_feature_bin(arm, row)]["probability"]
        return _from_fraction(payload)
    if arm == "fixed_threshold_pid":
        visible = int(row["outcome"]["visible_risk"])
        threshold = int(row["failure_threshold"])
        return Fraction(3, 4) if visible >= threshold else Fraction(1, 4)
    if arm == "oracle_structural":
        return Fraction(int(row["outcome"]["failure_within_horizon"]), 1)
    raise ValueError(f"unsupported P9 monitor arm {arm!r}")


def _classification_metrics(
    rows: list[dict[str, Any]], predictions: list[Fraction], decision: Fraction
) -> dict[str, Any]:
    labels = [int(row["outcome"]["failure_within_horizon"]) for row in rows]
    predicted = [int(value >= decision) for value in predictions]
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predicted, strict=True))
    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predicted, strict=True))
    sensitivity = Fraction(true_positive, positives) if positives else Fraction()
    specificity = Fraction(true_negative, negatives) if negatives else Fraction()
    balanced = (sensitivity + specificity) / 2
    positive_scores = [p for p, y in zip(predictions, labels, strict=True) if y == 1]
    negative_scores = [p for p, y in zip(predictions, labels, strict=True) if y == 0]
    pair_credit = Fraction()
    for positive in positive_scores:
        for negative in negative_scores:
            pair_credit += Fraction(int(positive > negative), 1)
            pair_credit += Fraction(int(positive == negative), 2)
    pairs = len(positive_scores) * len(negative_scores)
    auc = pair_credit / pairs if pairs else Fraction()
    calibration_error = Fraction()
    for probability in sorted(set(predictions)):
        indexes = [index for index, value in enumerate(predictions) if value == probability]
        observed = Fraction(sum(labels[index] for index in indexes), len(indexes))
        calibration_error += Fraction(len(indexes), len(rows)) * abs(observed - probability)
    early_indexes = [
        index
        for index, row in enumerate(rows)
        if labels[index] == 1 and int(row["telemetry"]["external_error_lag"]) == 0
    ]
    early_recall = (
        Fraction(sum(predicted[index] for index in early_indexes), len(early_indexes))
        if early_indexes
        else Fraction()
    )
    return {
        "rows": len(rows),
        "failures": positives,
        "nonfailures": negatives,
        "brier": _fraction(_brier(rows, predictions)),
        "expected_calibration_error": _fraction(calibration_error),
        "balanced_accuracy": _fraction(balanced),
        "roc_auc": _fraction(auc),
        "prospective_recall_before_external_error": _fraction(early_recall),
        "early_failure_rows": len(early_indexes),
    }


def _intervention_response(
    unit: dict[str, Any], model: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    pressure = set(config["evaluation"]["pressure_interventions"])
    informative = 0
    sign_agree = 0
    prediction_changes = 0
    total_pairs = 0
    branch_probability_counts: list[int] = []
    for group in unit["splits"]["heldout"]["groups"]:
        by_name = {row["intervention"]["name"]: row for row in group["branches"]}
        observed = by_name["observational"]
        observed_probability = _predict(model, observed)
        branch_probability_counts.append(len({_predict(model, row) for row in group["branches"]}))
        for name in (*sorted(pressure), str(config["evaluation"]["control_intervention"])):
            branch = by_name[name]
            total_pairs += 1
            outcome_delta = int(branch["outcome"]["failure_within_horizon"]) - int(
                observed["outcome"]["failure_within_horizon"]
            )
            probability_delta = _predict(model, branch) - observed_probability
            prediction_changes += int(probability_delta != 0)
            if outcome_delta != 0:
                informative += 1
                sign_agree += int(
                    (outcome_delta > 0 and probability_delta > 0)
                    or (outcome_delta < 0 and probability_delta < 0)
                )
    return {
        "paired_same_parent_comparisons": total_pairs,
        "informative_outcome_changes": informative,
        "prediction_changes": prediction_changes,
        "sign_agreement_on_informative_pairs": _fraction(
            Fraction(sign_agree, informative) if informative else Fraction()
        ),
        "mean_unique_probabilities_per_branch_set": _fraction(
            Fraction(sum(branch_probability_counts), len(branch_probability_counts))
        ),
    }


def _controller_metrics(
    unit: dict[str, Any], model: dict[str, Any], config: dict[str, Any], decision: Fraction
) -> dict[str, Any]:
    relief_name = str(config["evaluation"]["control_intervention"])
    failure_cost = int(config["fixture"]["failure_cost"])
    relief_cost = int(config["fixture"]["relief_cost"])
    before_failures = 0
    after_failures = 0
    relief_actions = 0
    unnecessary_relief = 0
    utility = 0
    baseline_utility = 0
    decisions: list[dict[str, Any]] = []
    for group in unit["splits"]["heldout"]["groups"]:
        by_name = {row["intervention"]["name"]: row for row in group["branches"]}
        observed = by_name["observational"]
        probability = _predict(model, observed)
        intervene = probability >= decision
        selected = by_name[relief_name] if intervene else observed
        before_failure = bool(observed["outcome"]["failure_within_horizon"])
        after_failure = bool(selected["outcome"]["failure_within_horizon"])
        before_failures += int(before_failure)
        after_failures += int(after_failure)
        relief_actions += int(intervene)
        unnecessary_relief += int(intervene and not before_failure)
        baseline_utility -= failure_cost * int(before_failure)
        utility -= failure_cost * int(after_failure) + relief_cost * int(intervene)
        decisions.append(
            {
                "lineage_id": group["lineage_id"],
                "probability": _fraction(probability),
                "selected_intervention": relief_name if intervene else "observational",
                "failure_before": before_failure,
                "failure_after": after_failure,
            }
        )
    count = len(decisions)
    return {
        "episodes": count,
        "failures_before_control": before_failures,
        "failures_after_control": after_failures,
        "failures_avoided": before_failures - after_failures,
        "relief_actions": relief_actions,
        "unnecessary_relief_actions": unnecessary_relief,
        "mean_utility": _fraction(Fraction(utility, count)),
        "mean_utility_delta_vs_no_control": _fraction(Fraction(utility - baseline_utility, count)),
        "decision_trace_sha256": canonical_sha256(decisions),
        "one_decision_per_episode": len({row["lineage_id"] for row in decisions}) == count,
    }


def _evaluate_unit(unit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    train = _rows(unit, "train")
    calibration = _rows(unit, "calibration")
    heldout = _rows(unit, "heldout")
    decision = Fraction(
        int(config["fixture"]["decision_threshold_numerator"]),
        int(config["fixture"]["decision_threshold_denominator"]),
    )
    models: dict[str, dict[str, Any]] = {
        arm: _fit_histogram(arm, train, calibration, config) for arm in HISTOGRAM_ARMS
    }
    models.update(
        {
            "fixed_threshold_pid": {
                "arm": "fixed_threshold_pid",
                "kind": "preregistered_visible-risk threshold",
                "cost": {"trainable_parameters": 0, "stored_thresholds": 1},
            },
            "oracle_structural": {
                "arm": "oracle_structural",
                "kind": "fixture sensitivity oracle with hidden shock",
                "cost": {"trainable_parameters": 0, "stored_thresholds": 1},
            },
        }
    )
    arms: dict[str, dict[str, Any]] = {}
    pressure_names = set(config["evaluation"]["pressure_interventions"])
    for arm in ARM_ORDER:
        model = models[arm]
        subsets = {
            "all_heldout": heldout,
            "observational": [row for row in heldout if row["intervention"]["name"] == "observational"],
            "pressure_interventions": [
                row for row in heldout if row["intervention"]["name"] in pressure_names
            ],
            "resource_relief": [
                row
                for row in heldout
                if row["intervention"]["name"] == config["evaluation"]["control_intervention"]
            ],
        }
        arms[arm] = {
            "role": (
                "primary"
                if arm == config["evaluation"]["primary_arm"]
                else "structural-control"
                if arm in STRUCTURAL_CONTROLS
                else "negative-control"
            ),
            "model": model,
            "heldout": {
                name: _classification_metrics(rows, [_predict(model, row) for row in rows], decision)
                for name, rows in subsets.items()
            },
            "intervention_response": _intervention_response(unit, model, config),
            "relief_controller": _controller_metrics(unit, model, config, decision),
            "mechanics_only": True,
        }

    cost_payloads = [models[arm]["cost"] for arm in HISTOGRAM_ARMS]
    matched_capacity = len({canonical_sha256(payload) for payload in cost_payloads}) == 1
    causal = arms["causal_abstraction"]
    proxy = arms["correlational_proxy"]
    fixed = arms["fixed_threshold_pid"]

    causal_brier = causal["heldout"]["all_heldout"]["brier"]
    proxy_brier = proxy["heldout"]["all_heldout"]["brier"]
    causal_auc = causal["heldout"]["all_heldout"]["roc_auc"]
    proxy_auc = proxy["heldout"]["all_heldout"]["roc_auc"]
    comparison = {
        "causal_minus_correlational": {
            "brier_improvement": _fraction(_from_fraction(proxy_brier) - _from_fraction(causal_brier)),
            "roc_auc_delta": _fraction(_from_fraction(causal_auc) - _from_fraction(proxy_auc)),
            "intervention_sign_agreement_delta": _fraction(
                _from_fraction(causal["intervention_response"]["sign_agreement_on_informative_pairs"])
                - _from_fraction(proxy["intervention_response"]["sign_agreement_on_informative_pairs"])
            ),
            "controller_utility_delta": _fraction(
                _from_fraction(causal["relief_controller"]["mean_utility"])
                - _from_fraction(proxy["relief_controller"]["mean_utility"])
            ),
        },
        "causal_minus_fixed_threshold": {
            "brier_improvement": _fraction(
                _from_fraction(fixed["heldout"]["all_heldout"]["brier"]) - _from_fraction(causal_brier)
            ),
            "controller_utility_delta": _fraction(
                _from_fraction(causal["relief_controller"]["mean_utility"])
                - _from_fraction(fixed["relief_controller"]["mean_utility"])
            ),
        },
    }
    checks = {
        "all_arms_executed": tuple(arms) == ARM_ORDER,
        "histogram_capacity_matched": matched_capacity,
        "heldout_lineages_disjoint_from_fit": not (
            {row["lineage_id"] for row in heldout}
            & ({row["lineage_id"] for row in train} | {row["lineage_id"] for row in calibration})
        ),
        "causal_monitor_responds_to_some_interventions": causal["intervention_response"]["prediction_changes"]
        > 0,
        "correlational_proxy_is_branch_invariant": proxy["intervention_response"][
            "mean_unique_probabilities_per_branch_set"
        ]["numerator"]
        == proxy["intervention_response"]["mean_unique_probabilities_per_branch_set"]["denominator"],
        "oracle_detects_fixture_signal": _from_fraction(
            arms["oracle_structural"]["heldout"]["all_heldout"]["roc_auc"]
        )
        == 1,
        "controller_uses_real_paired_relief_branch": all(
            row["relief_controller"]["one_decision_per_episode"] for row in arms.values()
        ),
        "all_metrics_finite": _finite_tree(arms),
    }
    return {
        "independent_unit": copy.deepcopy(unit["unit"]),
        "dataset_payload_sha256": unit["payload_sha256"],
        "split_payload_sha256": {split: unit["splits"][split]["payload_sha256"] for split in SPLITS},
        "arms": arms,
        "causal_comparisons": comparison,
        "matched_histogram_capacity": {
            "arms": list(HISTOGRAM_ARMS),
            "contract": cost_payloads[0],
            "matched": matched_capacity,
            "boundary": (
                "table capacity, fit rows, calibration rows, and smoothing search are exact; "
                "feature semantics intentionally differ by arm"
            ),
        },
        "checks": checks,
        "all_mechanics_ok": all(checks.values()),
        "scientific_promotion_allowed": False,
    }


def _aggregate(units: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "brier_improvement",
        "roc_auc_delta",
        "intervention_sign_agreement_delta",
        "controller_utility_delta",
    )
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [
            _from_fraction(unit["causal_comparisons"]["causal_minus_correlational"][key]) for unit in units
        ]
        aggregate[key] = {
            "mean": _fraction(_mean(values)),
            "positive_units": sum(value > 0 for value in values),
            "zero_units": sum(value == 0 for value in values),
            "negative_units": sum(value < 0 for value in values),
        }
    return aggregate


def _corrupt_checkpoint(config: dict[str, Any], source: Path, target: Path) -> bool:
    shutil.copy2(source, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["completed_chunks"][0]["payload"]["groups"][0]["branches"][0]["outcome"][
        "failure_within_horizon"
    ] = not payload["completed_chunks"][0]["payload"]["groups"][0]["branches"][0]["outcome"][
        "failure_within_horizon"
    ]
    _atomic_json(target, payload)
    try:
        build_dataset_resumable(config, target)
    except ValueError:
        return True
    return False


def _deterministic_part(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key not in {"resource_observation", "deterministic_core_sha256"}
    }


def build_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    source_import_problems = forbidden_source_imports(Path(__file__))
    with PeakRSSMonitor() as rss_monitor, deny_forbidden_runtime_imports() as runtime_import_attempts:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="p9-causal-monitoring-") as scratch_name:
            scratch = Path(scratch_name)
            accountant = WorkloadAccountant(
                workload=(
                    "P9 bounded structural telemetry, intervention, calibration, and controller preflight"
                ),
                watch_paths={"scratch": scratch},
            )
            with accountant.phase("fixture"):
                dataset = build_dataset(config)
            with accountant.phase("resume"):
                checkpoint = scratch / "resume.json"
                interrupted = build_dataset_resumable(config, checkpoint, stop_after_chunks=4)
                if interrupted is not None:
                    raise ValueError("P9 resume drill did not stop at the requested chunk boundary")
                resumed = build_dataset_resumable(config, checkpoint)
                if resumed is None:
                    raise ValueError("P9 resume drill did not finish")
                resume_exact = resumed["payload_sha256"] == dataset["payload_sha256"]
                corrupted_rejected = _corrupt_checkpoint(config, checkpoint, scratch / "corrupt.json")
                checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            with accountant.phase("fit_evaluate"):
                units = [_evaluate_unit(unit, config) for unit in dataset["units"]]
            with accountant.phase("verify"):
                verification = verify_dataset(dataset)
                mutations = mutation_suite(dataset)
            accounting = accountant.receipt()

    elapsed = time.perf_counter() - started
    max_rss = rss_monitor.peak_rss_bytes
    envelope = config["resource_envelope"]
    checks = {
        "minimum_independent_units": len(units) >= int(config["stop_contract"]["minimum_independent_units"]),
        "dataset_exact_replay": verification["verified"] is True,
        "all_dataset_mutations_rejected": mutations["all_rejected"] is True,
        "all_unit_mechanics_ok": all(unit["all_mechanics_ok"] is True for unit in units),
        "interrupted_resume_is_exact": resume_exact,
        "corrupt_checkpoint_rejected": corrupted_rejected,
        "all_scientific_promotion_blocked": all(
            unit["scientific_promotion_allowed"] is False for unit in units
        ),
        "resource_wall_envelope": elapsed <= float(envelope["maximum_wall_seconds"]),
        "resource_rss_envelope": rss_monitor.peak_increment_bytes <= int(envelope["maximum_rss_bytes"]),
        "resource_rss_sampling_complete": rss_monitor.all_ok,
        "no_forbidden_model_imports_in_source": not source_import_problems,
        "no_runtime_model_import_attempts": not runtime_import_attempts,
        "no_model_or_download_modules": not source_import_problems and not runtime_import_attempts,
    }
    core: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if all(checks.values()) else "mechanics-fail",
        "null_hypothesis": config["null_hypothesis"],
        "audit": {
            "already_existed": [
                "phase-scoped wall, CPU, RSS, accelerator, storage, retry, and idle accounting",
                "an explicit unmeasured-energy boundary",
                "generic intervention and immutable event primitives outside P9",
            ],
            "true_gap_closed": [
                "prospective causal telemetry monitoring",
                "same-parent pressure and relief interventions",
                "a shifted correlational shortcut and negative controls",
                "disjoint fit, calibration, and held-out lineages",
                "exact rational calibration and intervention-response metrics",
                "a paired relief controller with utility accounting",
                "fail-closed interrupted resume and provenance chain",
            ],
            "namespace_note": (
                "this extended-compute P9 operational-self-model surface is distinct from the "
                "registered p9_thought_without_language experiment"
            ),
        },
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(config_path),
            "payload_sha256": canonical_sha256(config),
            "payload": config,
        },
        "runtime_integrity": {
            "forbidden_module_prefixes": list(FORBIDDEN_MODEL_MODULES),
            "source_import_problems": source_import_problems,
            "runtime_import_attempts": runtime_import_attempts,
            "all_ok": not source_import_problems and not runtime_import_attempts,
        },
        "dataset": {
            "schema": dataset["schema"],
            "payload_sha256": dataset["payload_sha256"],
            "budget_contract": dataset["budget_contract"],
            "verification": verification,
        },
        "resume": {
            "interrupted_after_chunks": 4,
            "completed_chunks": len(checkpoint_payload["completed_chunks"]),
            "checkpoint_chain_sha256": checkpoint_payload["chain_sha256"],
            "final_dataset_sha256": checkpoint_payload["final_dataset_sha256"],
            "clean_dataset_sha256": dataset["payload_sha256"],
            "exact": resume_exact,
            "corrupt_checkpoint_rejected": corrupted_rejected,
        },
        "mutation_suite": mutations,
        "units": units,
        "causal_vs_correlational_aggregate": _aggregate(units),
        "checks": checks,
        "claim_boundary": {
            "mechanics_only": True,
            "natural_workloads": False,
            "physical_failures": False,
            "capability_claim": False,
            "cognition_or_sentience_claim": False,
            "energy_measured": False,
            "scientific_promotion_allowed": False,
            "remaining_evidence_gate": (
                "independent natural workload and failure episodes, prospectively registered telemetry, "
                "real bounded interventions, replicated calibration, and a metered boundary for energy"
            ),
        },
        "implementation": [
            _file_receipt(REPO_ROOT / path) for path in (*IMPLEMENTATION_PATHS, *UPSTREAM_PATHS)
        ],
        "all_mechanics_ok": all(checks.values()),
    }
    core["deterministic_core_sha256"] = canonical_sha256(core)
    return {
        **core,
        "resource_observation": {
            "elapsed_seconds": round(elapsed, 6),
            "max_rss_bytes": max_rss,
            "phase_local_peak_rss_increment_bytes": rss_monitor.peak_increment_bytes,
            "rss_limit_scope": "phase-local sampled peak increment above phase-start RSS",
            "rss_measurement": rss_monitor.receipt(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "device": "cpu",
            "cpu_threads": 1,
            "accelerator_required": False,
            "model_weights_loaded": False,
            "model_downloads_performed": False,
            "external_data_loaded": False,
            "command_executed_heavy_work": False,
            "workload_accounting": accounting,
        },
    }


def verify_preflight_receipt(receipt: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    recorded = str(receipt.get("deterministic_core_sha256", ""))
    self_hash = canonical_sha256(_deterministic_part(receipt))
    rebuilt = build_preflight(config_path)
    rebuilt_hash = str(rebuilt["deterministic_core_sha256"])
    checks = {
        "recorded_core_self_hash": recorded == self_hash,
        "exact_rebuild_hash": recorded == rebuilt_hash,
        "exact_rebuild_payload": canonical_sha256(_deterministic_part(receipt))
        == canonical_sha256(_deterministic_part(rebuilt)),
        "mechanics_pass": receipt.get("all_mechanics_ok") is True,
        "scientific_promotion_blocked": receipt.get("claim_boundary", {}).get("scientific_promotion_allowed")
        is False,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "recorded_core_sha256": recorded,
        "rebuilt_core_sha256": rebuilt_hash,
    }


def write_preflight(
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    receipt = build_preflight(config_path)
    if receipt["all_mechanics_ok"] is not True:
        raise ValueError("P9 causal-monitoring mechanics did not pass")
    _atomic_json(output_path, receipt)
    maximum = int(receipt["config"]["payload"]["resource_envelope"]["maximum_proof_bytes"])
    if output_path.stat().st_size > maximum:
        output_path.unlink()
        raise ValueError("P9 causal-monitoring proof exceeds its declared byte envelope")
    return receipt
