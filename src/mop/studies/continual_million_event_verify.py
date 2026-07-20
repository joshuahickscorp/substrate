
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from mop.config import REPO_ROOT
from mop.substrate.events import canonical_bytes

VERIFIER_SCHEMA = "mop-continual-progressive-rung-independent-verifier/v1"
RUNG_SCHEMA = "mop-continual-progressive-rung/v1"
RUNG_CONFIG_SCHEMA = "mop-continual-progressive-rungs-config/v1"
PROGRESS_SCHEMA = "mop-continual-progressive-rung-progress/v1"
CHECKPOINT_SCHEMA = "mop-continual-smoke-checkpoint/v1"
PREFLIGHT_SCHEMA = "mop-continual-million-event-preflight/v1"
STREAM_SCHEMA = "mop-continual-stream-manifest/v1"
LIFECYCLE_SCHEMA = "mop-lifecycle-journal/v1"
STREAM_SPEC_SCHEMA = "mop-continual-stream-spec/v1"
CLAIM_SCOPE = "disk-backed programmatic continual-stream mechanics only; no capability claim"
TIE_RULE = "any zero or negative endpoint within a paired seed, or any nonpositive paired mean, is a null"
RUNGS = (10_000, 100_000, 1_000_000)
SCHEDULES = ("abrupt", "gradual")
ARMS = ("replay", "no-replay", "fresh-init")
METRIC_FAMILIES = (
    "retention",
    "acquisition",
    "future_learnability",
    "stale_memory",
    "deletion",
    "resources",
)
RUNG_CONFIG_PATH = "configs/experiment/continual_million_event_rungs.yaml"
RUNG_RUNNER_PATH = "scripts/continual_million_event_rung.py"
SOURCE_PREFLIGHT_PATH = "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json"
IMPLEMENTATION_PATHS = (
    "src/mop/studies/continual_million_event_verify.py",
    "scripts/verify_continual_million_event_rung.py",
    RUNG_RUNNER_PATH,
    RUNG_CONFIG_PATH,
    SOURCE_PREFLIGHT_PATH,
)
RECORD_CORE = struct.Struct("<QHHHHHB")
RECORD = struct.Struct("<QHHHHHB32s")
FLAG_ACTIVE_SECONDARY = 1
FLAG_TRANSITION = 2
FLAG_DELETE = 4


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value}")),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON authority is not an object: {path}")
    return payload


def _repo_path(value: object, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("repository-relative authority path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"authority path escapes the repository: {value!r}")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"authority path escapes the repository: {value!r}")
    return path


def _display_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _source_payload_ok(payload: dict[str, Any]) -> bool:
    core = dict(payload)
    declared = core.pop("payload_sha256", None)
    return _is_sha256(declared) and _canonical_sha256(core) == declared


def _embedded_preflight_authority(
    preflight: dict[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    config = preflight.get("config")
    implementation = preflight.get("implementation")
    upstream = preflight.get("wave_e0")
    authority: dict[str, Any] = {"config": {}, "implementation": [], "wave_e0": {}}
    try:
        if not isinstance(config, dict):
            raise ValueError("preflight config binding missing")
        config_path = _repo_path(config.get("path"), repo_root=repo_root)
        observed_config_sha = _sha256_file(config_path)
        if observed_config_sha != config.get("sha256"):
            problems.append("preflight live config hash drift")
        config_payload = config.get("payload")
        if not isinstance(config_payload, dict):
            problems.append("preflight embedded config payload missing")
        else:
            if _canonical_sha256(config_payload) != config.get("profile_sha256"):
                problems.append("preflight embedded config digest drift")
            if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config_payload:
                problems.append("preflight live config payload drift")
        authority["config"] = {
            "path": str(config.get("path")),
            "sha256": observed_config_sha,
            "payload_sha256": str(config.get("profile_sha256")),
        }
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"preflight config authority invalid: {exc}")

    seen: set[str] = set()
    if not isinstance(implementation, list) or not implementation:
        problems.append("preflight implementation bindings missing")
    else:
        for index, row in enumerate(implementation):
            try:
                if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                    raise ValueError("binding row or path invalid")
                relative = str(row["path"])
                if relative in seen:
                    raise ValueError("duplicate binding path")
                seen.add(relative)
                observed = _sha256_file(_repo_path(relative, repo_root=repo_root))
                if observed != row.get("sha256"):
                    problems.append(f"preflight live implementation drift: {relative}")
                authority["implementation"].append({"path": relative, "sha256": observed})
            except (OSError, TypeError, ValueError) as exc:
                problems.append(f"preflight implementation binding {index} invalid: {exc}")
    if isinstance(config, dict) and config.get("path") not in seen:
        problems.append("preflight config is absent from implementation bindings")

    try:
        if not isinstance(upstream, dict):
            raise ValueError("Wave E0 binding missing")
        upstream_path = _repo_path(upstream.get("path"), repo_root=repo_root)
        observed_upstream_sha = _sha256_file(upstream_path)
        if observed_upstream_sha != upstream.get("sha256"):
            problems.append("preflight live Wave E0 hash drift")
        authority["wave_e0"] = {
            "path": str(upstream.get("path")),
            "sha256": observed_upstream_sha,
        }
    except (OSError, TypeError, ValueError) as exc:
        problems.append(f"preflight Wave E0 authority invalid: {exc}")

    authority["bindings_sha256"] = _canonical_sha256(authority)
    return authority, problems


def _live_dependencies(
    identity: dict[str, Any], source_authority: object, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    problems: list[str] = []
    rung_config: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    try:
        config_path = _repo_path(RUNG_CONFIG_PATH, repo_root=repo_root)
        runner_path = _repo_path(RUNG_RUNNER_PATH, repo_root=repo_root)
        preflight_path = _repo_path(SOURCE_PREFLIGHT_PATH, repo_root=repo_root)
        config_sha = _sha256_file(config_path)
        runner_sha = _sha256_file(runner_path)
        preflight_sha = _sha256_file(preflight_path)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("rung config is not an object")
        rung_config = loaded
        preflight = _read_json(preflight_path)
        if preflight.get("schema") != PREFLIGHT_SCHEMA or not _source_payload_ok(preflight):
            problems.append("source preflight schema or payload digest invalid")
        embedded_authority, embedded_problems = _embedded_preflight_authority(preflight, repo_root=repo_root)
        problems.extend(embedded_problems)
        expected = {
            "config_sha256": config_sha,
            "runner_sha256": runner_sha,
            "source_preflight_file_sha256": preflight_sha,
            "source_preflight_payload_sha256": preflight.get("payload_sha256"),
            "source_live_bindings_sha256": embedded_authority["bindings_sha256"],
        }
        for field, value in expected.items():
            if identity.get(field) != value:
                problems.append(f"rung identity live dependency drift: {field}")
        if source_authority != embedded_authority:
            problems.append("rung receipt source live authority drift")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"live dependency validation failed: {exc}")
    return rung_config, expected, problems


def _expected_matrix(plan: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    seeds = tuple(plan["seeds"])
    schedules = tuple(plan["schedules"])
    arms = tuple(plan["arms"])
    rows = [
        {"seed": seed, "schedule": schedule, "arm": arm}
        for seed in seeds
        for schedule in schedules
        for arm in arms
    ]
    keys = {f"seed_{row['seed']}/{row['schedule']}/{row['arm']}" for row in rows}
    return keys, rows


def _expected_plan_from_config(rung_config: dict[str, Any], *, rung: int, mode: str) -> dict[str, Any]:

    if rung_config.get("schema") != RUNG_CONFIG_SCHEMA:
        raise ValueError("live progressive rung config schema drift")
    if rung_config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("live progressive rung config claim scope drift")
    replication = rung_config.get("replication")
    profile = rung_config.get("profile")
    if not isinstance(replication, dict) or not isinstance(profile, dict):
        raise ValueError("live progressive rung replication or profile config missing")
    configured_rungs = tuple(int(value) for value in replication.get("rungs", []))
    if rung not in configured_rungs:
        raise ValueError("source rung is absent from the live progressive rung config")
    all_seeds = [int(value) for value in replication.get("seeds", [])]
    if mode == "resource-probe":
        if not all_seeds:
            raise ValueError("live progressive rung seed config is empty")
        seeds = all_seeds[:1]
        schedules = ["abrupt"]
        arms = ["replay"]
    elif mode == "replication":
        seeds = all_seeds
        schedules = [str(value) for value in replication.get("schedules", [])]
        arms = [str(value) for value in replication.get("arms", [])]
    else:
        raise ValueError(f"unsupported progressive rung mode {mode!r}")
    cells = [
        {"seed": seed, "schedule": schedule, "arm": arm}
        for seed in seeds
        for schedule in schedules
        for arm in arms
    ]
    return {
        "mode": mode,
        "rung": rung,
        "seeds": seeds,
        "schedules": schedules,
        "arms": arms,
        "cells": cells,
        "expected_cells": len(cells),
        "stream": {
            "chunk_events": max(
                int(profile["minimum_chunk_events"]),
                rung // int(profile["chunks_per_stream"]),
            ),
            "n_domains": 4,
            "n_classes": 4,
            "gradual_width_events": max(1, rung // int(profile["gradual_width_divisor"])),
            "deletion_event": (
                rung * int(profile["deletion_numerator"]) // int(profile["deletion_denominator"])
            ),
        },
        "profile": {
            "checkpoint_every": int(profile["checkpoint_every_events"]),
            "replay_capacity": int(profile["replay_capacity"]),
            "future_window_events": int(profile["future_window_events"]),
            "threshold_window_events": int(profile["threshold_window_events"]),
            "future_accuracy_threshold": float(profile["future_accuracy_threshold"]),
            "matched_updates_per_event": int(profile["matched_updates_per_event"]),
        },
    }


def _validate_structure(
    receipt: dict[str, Any],
    *,
    rung_config: dict[str, Any],
    expected_dependencies: dict[str, Any],
    authority_cells: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    problems: list[str] = []
    try:
        if receipt.get("schema") != RUNG_SCHEMA:
            problems.append("rung schema drift")
        if not _source_payload_ok(receipt):
            problems.append("rung payload digest drift")
        if receipt.get("claim_scope") != CLAIM_SCOPE:
            problems.append("rung claim scope drift")
        identity = receipt.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("rung identity missing")
        if receipt.get("identity_sha256") != _canonical_sha256(identity):
            problems.append("rung identity digest drift")
        for field, expected in expected_dependencies.items():
            if identity.get(field) != expected:
                problems.append(f"rung identity dependency drift: {field}")
        if identity.get("claim_scope") != CLAIM_SCOPE:
            problems.append("rung identity claim scope drift")

        plan = receipt.get("plan")
        if not isinstance(plan, dict) or identity.get("plan") != plan:
            raise ValueError("rung plan is absent from or differs from identity")
        rung = receipt.get("rung")
        if rung not in RUNGS or plan.get("rung") != rung:
            problems.append("rung value drift")
        mode = receipt.get("mode")
        if not isinstance(rung, int) or isinstance(rung, bool) or not isinstance(mode, str):
            raise ValueError("rung value or mode is invalid")
        expected_plan = _expected_plan_from_config(rung_config, rung=rung, mode=mode)
        if plan != expected_plan or identity.get("plan") != expected_plan:
            problems.append("rung exact live-config plan drift")
        replication = rung_config.get("replication")
        if not isinstance(replication, dict):
            raise ValueError("live replication config missing")
        seeds = tuple(plan.get("seeds", ()))
        schedules = tuple(plan.get("schedules", ()))
        arms = tuple(plan.get("arms", ()))
        if seeds != tuple(replication.get("seeds", ())):
            problems.append("full rung seed set or order drift")
        if len(seeds) < int(replication.get("minimum_independent_seeds", 0)) or len(set(seeds)) != len(seeds):
            problems.append("full rung independent seed contract invalid")
        if schedules != SCHEDULES or schedules != tuple(replication.get("schedules", ())):
            problems.append("full rung schedule set or order drift")
        if arms != ARMS or arms != tuple(replication.get("arms", ())):
            problems.append("full rung control set or order drift")
        expected_keys, expected_rows = _expected_matrix(plan)
        if plan.get("mode") != "replication" or receipt.get("mode") != "replication":
            problems.append("independent verifier accepts only full replication rungs")
        if plan.get("cells") != expected_rows or plan.get("expected_cells") != len(expected_rows):
            problems.append("full rung plan matrix drift")
        cells = receipt.get("cells")
        if not isinstance(cells, dict) or set(cells) != expected_keys:
            problems.append("full rung completed cell set drift")
            cells = cells if isinstance(cells, dict) else {}
        if receipt.get("all_mechanics_ok") is not True:
            problems.append("rung mechanics did not pass")
        if receipt.get("replication_execution_complete") is not True:
            problems.append("rung replication execution is incomplete")
        if receipt.get("independent_metric_verifier_complete") is not False:
            problems.append("source rung cannot self-assert independent verification")
        if receipt.get("scientific_promotion") is not False:
            problems.append("source rung improperly promotes mechanics")
        progress = receipt.get("progress")
        if not isinstance(progress, dict):
            problems.append("rung progress binding missing")
        elif progress.get("completed_cells") != len(expected_rows) or progress.get("expected_cells") != len(
            expected_rows
        ):
            problems.append("rung progress completion counts drift")
        for key in sorted(expected_keys & set(cells)):
            row = cells[key]
            if not isinstance(row, dict):
                problems.append(f"cell {key} is not an object")
                continue
            expected_key = f"seed_{row.get('seed')}/{row.get('schedule')}/{row.get('arm')}"
            if expected_key != key:
                problems.append(f"cell {key} coordinate drift")
            if not all(
                _is_sha256(row.get(field))
                for field in (
                    "stream_identity_sha256",
                    "stream_sha256",
                    "checkpoint_sha256",
                    "state_sha256",
                )
            ):
                problems.append(f"cell {key} identity digest invalid")
            metrics = row.get("metrics")
            if not isinstance(metrics, dict) or set(metrics) != set(METRIC_FAMILIES):
                problems.append(f"cell {key} metric family drift")
            controls = row.get("controls")
            if not isinstance(controls, dict):
                problems.append(f"cell {key} controls missing")
            else:
                arm = row.get("arm")
                expected_control_flags = {
                    "replay_enabled": arm == "replay",
                    "fresh_init_on_transition": arm == "fresh-init",
                }
                for field, expected in expected_control_flags.items():
                    if controls.get(field) is not expected:
                        problems.append(f"cell {key} control semantics drift: {field}")
                if (
                    controls.get("matched_updates_per_event") != 2
                    or controls.get("actual_updates_per_event") != 2.0
                    or controls.get("fixed_topology") is not True
                ):
                    problems.append(f"cell {key} matched control contract drift")
            if row.get("all_mechanics_ok") is not True:
                problems.append(f"cell {key} mechanics did not pass")
            if not isinstance(row.get("resumed_from_atomic_checkpoint"), bool):
                problems.append(f"cell {key} resume observation is not boolean")
            if authority_cells is not None and row != authority_cells.get(key):
                problems.append(f"cell {key} differs from independently recomputed authority")
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"rung structure invalid: {exc}")
    return problems


def _expected_stream_spec(plan: dict[str, Any], *, seed: int, schedule: str) -> dict[str, Any]:
    stream = plan["stream"]
    return {
        "schema": STREAM_SPEC_SCHEMA,
        "seed": seed,
        "total_events": int(plan["rung"]),
        "chunk_events": int(stream["chunk_events"]),
        "n_domains": int(stream["n_domains"]),
        "n_classes": int(stream["n_classes"]),
        "transition_schedule": schedule,
        "gradual_width_events": int(stream["gradual_width_events"]),
        "deletion_event": int(stream["deletion_event"]),
    }


def _stable_int(spec: dict[str, Any], sequence: int, label: str, modulus: int) -> int:
    raw = canonical_bytes(
        {
            "stream_identity_sha256": _canonical_sha256(spec),
            "sequence": sequence,
            "label": label,
        }
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % modulus


def _expected_record_fields(spec: dict[str, Any], sequence: int) -> tuple[int, int, int, int, int, int, int]:
    total = int(spec["total_events"])
    n_domains = int(spec["n_domains"])
    n_classes = int(spec["n_classes"])
    segment = total / n_domains
    default_domain = min(n_domains - 1, int(sequence / segment))
    flags = 0
    if spec["transition_schedule"] == "abrupt":
        domain_a = default_domain
        domain_b = default_domain
        if sequence > 0 and int((sequence - 1) / segment) != default_domain:
            flags |= FLAG_TRANSITION
        blend = 0
    else:
        domain_a = default_domain
        domain_b = default_domain
        blend = 0
        half_width = int(spec["gradual_width_events"]) / 2.0
        for next_domain in range(1, n_domains):
            boundary = next_domain * segment
            start = boundary - half_width
            end = boundary + half_width
            if start <= sequence < end:
                blend = int(round(1000.0 * (sequence - start) / max(1.0, end - start - 1.0)))
                blend = max(0, min(1000, blend))
                secondary = _stable_int(spec, sequence, "gradual-choice", 1000) < blend
                domain_a = next_domain - 1
                domain_b = next_domain
                flags = FLAG_TRANSITION | (FLAG_ACTIVE_SECONDARY if secondary else 0)
                break
    cue = _stable_int(spec, sequence, "cue", n_classes)
    active_domain = domain_b if flags & FLAG_ACTIVE_SECONDARY else domain_a
    label = (cue + active_domain) % n_classes
    if sequence == int(spec["deletion_event"]):
        flags |= FLAG_DELETE
    return sequence, domain_a, domain_b, blend, cue, label, flags


def _stream_composite_sha(manifest: dict[str, Any]) -> str:
    chunks = manifest.get("chunks")
    rows = chunks if isinstance(chunks, list) else []
    return _canonical_sha256(
        {
            "identity_sha256": manifest.get("identity_sha256"),
            "chunks": [
                {
                    "index": row.get("index"),
                    "start_sequence": row.get("start_sequence"),
                    "count": row.get("count"),
                    "sha256": row.get("sha256"),
                    "chain_head_sha256": row.get("chain_head_sha256"),
                }
                for row in rows
                if isinstance(row, dict)
            ],
            "chain_head_sha256": manifest.get("chain_head_sha256"),
            "generated_events": manifest.get("generated_events"),
        }
    )


def _new_replay_state(plan: dict[str, Any]) -> dict[str, Any]:
    stream = plan["stream"]
    n_classes = int(stream["n_classes"])
    n_domains = int(stream["n_domains"])
    return {
        "next_sequence": 0,
        "last_event_sha256": "0" * 64,
        "counts": [[0 for _ in range(n_classes)] for _ in range(n_classes)],
        "replay": [],
        "current_stage": 0,
        "correct": 0,
        "total": 0,
        "domain_correct": [0 for _ in range(n_domains)],
        "domain_total": [0 for _ in range(n_domains)],
        "anchor_snapshots": [],
        "future_outcomes": [],
        "future_seen": 0,
        "future_events_to_threshold": None,
        "future_rolling": [],
        "stale_opportunities": 0,
        "stale_harm_count": 0,
        "deletion_seen": False,
        "deletion_removed": 0,
        "updates": 0,
        "replay_samples": 0,
        "resets": 0,
        "max_replay_records": 0,
    }


def _event_ref(stream_identity_sha256: str, sequence: int) -> str:
    return f"event:{stream_identity_sha256[:16]}/stream:{sequence:012d}"


def _process_replayed_event(
    state: dict[str, Any],
    event: dict[str, Any],
    *,
    arm: str,
    plan: dict[str, Any],
) -> None:
    profile = plan["profile"]
    stream = plan["stream"]
    counts = state["counts"]
    replay = state["replay"]
    stage = max(int(event["domain_a"]), int(event["domain_b"]))
    if stage > int(state["current_stage"]):
        state["anchor_snapshots"].append(
            {
                "sequence": event["sequence"],
                "stage": stage,
                "domain_zero_accuracy": sum(int(_predict(row) == cue) for cue, row in enumerate(counts))
                / int(stream["n_classes"]),
            }
        )
        if arm == "fresh-init":
            n_classes = int(stream["n_classes"])
            state["counts"] = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
            counts = state["counts"]
            state["resets"] += 1
        state["current_stage"] = stage

    if event["deletion_requested"]:
        before = len(replay)
        replay[:] = [row for row in replay if int(row["domain"]) != 0]
        state["deletion_removed"] += before - len(replay)
        state["deletion_seen"] = True

    cue = int(event["cue"])
    label = int(event["label"])
    active_domain = int(event["active_domain"])
    prediction = _predict(counts[cue])
    correct = prediction == label
    state["correct"] += int(correct)
    state["total"] += 1
    state["domain_correct"][active_domain] += int(correct)
    state["domain_total"][active_domain] += 1

    if active_domain == int(stream["n_domains"]) - 1:
        state["future_seen"] += 1
        if len(state["future_outcomes"]) < int(profile["future_window_events"]):
            state["future_outcomes"].append(int(correct))
        rolling = state["future_rolling"]
        rolling.append(int(correct))
        if len(rolling) > int(profile["threshold_window_events"]):
            rolling.pop(0)
        if (
            state["future_events_to_threshold"] is None
            and len(rolling) == int(profile["threshold_window_events"])
            and sum(rolling) / len(rolling) >= float(profile["future_accuracy_threshold"])
        ):
            state["future_events_to_threshold"] = state["future_seen"]

    counts[cue][label] += 1
    state["updates"] += 1
    if arm == "replay" and replay:
        replay_row = replay[int(event["sequence"]) % len(replay)]
        before_replay = _predict(counts[cue]) == label
        stale = int(replay_row["domain"]) != active_domain
        replay_cue = int(replay_row["cue"])
        replay_label = int(replay_row["label"])
        counts[replay_cue][replay_label] += 1
        after_replay = _predict(counts[cue]) == label
        state["updates"] += 1
        state["replay_samples"] += 1
        if stale:
            state["stale_opportunities"] += 1
            state["stale_harm_count"] += int(before_replay and not after_replay)
    else:
        counts[cue][label] += 1
        state["updates"] += 1

    if arm == "replay":
        replay.append(
            {
                "event_ref": event["event_ref"],
                "event_sha256": event["content_sha256"],
                "cue": cue,
                "label": label,
                "domain": active_domain,
            }
        )
        if len(replay) > int(profile["replay_capacity"]):
            replay.pop(0)
    state["max_replay_records"] = max(int(state["max_replay_records"]), len(replay))
    state["next_sequence"] = int(event["sequence"]) + 1
    state["last_event_sha256"] = event["content_sha256"]


def _lifecycle_entry(
    *,
    memory_ref: str,
    event_ref: str,
    operation: str,
    content_value: dict[str, Any] | None,
    reason: str,
    previous: str | None,
    sequence: int,
) -> dict[str, Any]:
    content = (
        {"value": content_value, "sha256": _canonical_sha256(content_value)}
        if content_value is not None
        else None
    )
    entry: dict[str, Any] = {
        "memory_ref": memory_ref,
        "event_ref": event_ref,
        "sequence": sequence,
        "revision": sequence + 1,
        "operation": operation,
        "content": content,
        "availability_enabled": True if operation in {"record", "revise"} else None,
        "available_from_tick": 0 if operation in {"record", "revise"} else None,
        "available_until_tick": None,
        "target_revision": None,
        "reason": reason,
        "previous_entry_sha256": previous,
    }
    entry["entry_sha256"] = _canonical_sha256(entry)
    return entry


def _new_lifecycle(stream_identity_sha256: str) -> dict[str, Any]:
    return {
        "memory_ref": f"memory:{stream_identity_sha256[:16]}/continual-anchor",
        "entries": [],
        "current_stage": -1,
        "deleted": False,
    }


def _process_lifecycle_event(lifecycle: dict[str, Any], event: dict[str, Any]) -> None:
    entries = lifecycle["entries"]
    stage = max(int(event["domain_a"]), int(event["domain_b"]))
    content = {
        "stage": stage,
        "mapping": "cue-plus-domain",
        "source_event": event["content_sha256"],
    }
    if int(lifecycle["current_stage"]) < 0:
        entries.append(
            _lifecycle_entry(
                memory_ref=str(lifecycle["memory_ref"]),
                event_ref=str(event["event_ref"]),
                operation="record",
                content_value=content,
                reason="record initial stream mapping",
                previous=None,
                sequence=0,
            )
        )
        lifecycle["current_stage"] = stage
    elif stage > int(lifecycle["current_stage"]) and lifecycle["deleted"] is False:
        entries.append(
            _lifecycle_entry(
                memory_ref=str(lifecycle["memory_ref"]),
                event_ref=str(event["event_ref"]),
                operation="revise",
                content_value=content,
                reason="revise mapping at transition",
                previous=entries[-1]["entry_sha256"],
                sequence=len(entries),
            )
        )
        lifecycle["current_stage"] = stage
    if event["deletion_requested"] and lifecycle["deleted"] is False:
        entries.append(
            _lifecycle_entry(
                memory_ref=str(lifecycle["memory_ref"]),
                event_ref=str(event["event_ref"]),
                operation="delete",
                content_value=None,
                reason="delete superseded continual anchor",
                previous=entries[-1]["entry_sha256"],
                sequence=len(entries),
            )
        )
        lifecycle["deleted"] = True


def _lifecycle_payload(lifecycle: dict[str, Any]) -> dict[str, Any]:
    entries = lifecycle["entries"]
    return {
        "schema": LIFECYCLE_SCHEMA,
        "memory_ref": lifecycle["memory_ref"],
        "entries": entries,
        "head_sha256": entries[-1]["entry_sha256"] if entries else None,
    }


def _replay_stream(
    root: Path,
    *,
    plan: dict[str, Any],
    seed: int,
    schedule: str,
    arms: tuple[str, ...] = ARMS,
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    expected_spec = _expected_stream_spec(plan, seed=seed, schedule=schedule)
    expected_identity = _canonical_sha256(expected_spec)
    states = {arm: _new_replay_state(plan) for arm in arms}
    lifecycle = _new_lifecycle(expected_identity)
    disk_bytes = 0
    manifest: dict[str, Any] = {}
    previous_digest = bytes(32)
    expected_sequence = 0
    try:
        manifest = _read_json(root / "manifest.json")
        chunks = manifest.get("chunks")
        if manifest.get("schema") != STREAM_SCHEMA or manifest.get("complete") is not True:
            problems.append("stream manifest schema or completion drift")
        if manifest.get("spec") != expected_spec:
            problems.append("stream exact deterministic spec drift")
        if manifest.get("identity_sha256") != expected_identity:
            problems.append("stream canonical spec identity drift")
        if manifest.get("record_bytes") != RECORD.size:
            problems.append("stream fixed record width drift")
        if not isinstance(chunks, list) or not chunks:
            problems.append("stream chunk matrix missing")
            chunks = []
        for index, row in enumerate(chunks):
            if not isinstance(row, dict):
                problems.append(f"stream chunk {index} is invalid")
                continue
            expected_count = min(
                int(expected_spec["chunk_events"]),
                int(expected_spec["total_events"]) - expected_sequence,
            )
            expected_name = f"chunk_{index:06d}.bin"
            path = (root / expected_name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                problems.append(f"stream chunk {index} missing")
                continue
            raw = path.read_bytes()
            disk_bytes += len(raw)
            if row.get("path") != expected_name:
                problems.append(f"stream chunk {index} canonical path drift")
            if (
                row.get("index") != index
                or row.get("start_sequence") != expected_sequence
                or row.get("count") != expected_count
                or row.get("bytes") != len(raw)
                or row.get("sha256") != hashlib.sha256(raw).hexdigest()
                or row.get("chain_start_sha256") != previous_digest.hex()
            ):
                problems.append(f"stream chunk {index} metadata or file digest drift")
            if len(raw) != expected_count * RECORD.size:
                problems.append(f"stream chunk {index} fixed-width byte count drift")
                continue
            for offset in range(expected_count):
                record_raw = raw[offset * RECORD.size : (offset + 1) * RECORD.size]
                unpacked = RECORD.unpack(record_raw)
                fields = tuple(int(value) for value in unpacked[:7])
                declared_record_digest = bytes(unpacked[7])
                expected_fields = _expected_record_fields(expected_spec, expected_sequence)
                if fields != expected_fields:
                    problems.append(f"stream deterministic record field drift at {expected_sequence}")
                expected_record_digest = hashlib.sha256(
                    bytes.fromhex(expected_identity) + previous_digest + RECORD_CORE.pack(*fields)
                ).digest()
                if declared_record_digest != expected_record_digest:
                    problems.append(f"stream record digest chain drift at {expected_sequence}")
                previous_digest = declared_record_digest
                sequence, domain_a, domain_b, blend, cue, label, flags = fields
                active_domain = domain_b if flags & FLAG_ACTIVE_SECONDARY else domain_a
                event = {
                    "sequence": sequence,
                    "domain_a": domain_a,
                    "domain_b": domain_b,
                    "blend_milli": blend,
                    "cue": cue,
                    "label": label,
                    "flags": flags,
                    "active_domain": active_domain,
                    "deletion_requested": bool(flags & FLAG_DELETE),
                    "content_sha256": declared_record_digest.hex(),
                    "event_ref": _event_ref(expected_identity, sequence),
                }
                for arm, state in states.items():
                    _process_replayed_event(state, event, arm=arm, plan=plan)
                _process_lifecycle_event(lifecycle, event)
                expected_sequence += 1
            if row.get("chain_head_sha256") != previous_digest.hex():
                problems.append(f"stream chunk {index} chain head drift")
        if expected_sequence != int(plan["rung"]):
            problems.append("stream decoded event coverage drift")
        if manifest.get("generated_events") != expected_sequence:
            problems.append("stream generated event count drift")
        if manifest.get("chain_head_sha256") != previous_digest.hex():
            problems.append("stream manifest chain head drift")
        recomputed_stream_sha = _stream_composite_sha(manifest)
        if manifest.get("stream_sha256") != recomputed_stream_sha:
            problems.append("stream composite digest drift")
    except (KeyError, OSError, struct.error, TypeError, ValueError) as exc:
        problems.append(f"stream authority invalid: {exc}")
    return (
        {
            "identity_sha256": expected_identity,
            "stream_sha256": manifest.get("stream_sha256"),
            "disk_bytes": disk_bytes,
            "states": states,
            "lifecycle": _lifecycle_payload(lifecycle),
        },
        problems,
    )


def _audit_lifecycle(payload: object, declared_sha: object) -> tuple[bool, list[str]]:
    problems: list[str] = []
    deleted = False
    try:
        if not isinstance(payload, dict) or payload.get("schema") != LIFECYCLE_SCHEMA:
            raise ValueError("lifecycle schema drift")
        if _canonical_sha256(payload) != declared_sha:
            problems.append("lifecycle payload digest drift")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("lifecycle entries missing")
        previous: str | None = None
        memory_ref = payload.get("memory_ref")
        for index, row in enumerate(entries):
            if not isinstance(row, dict):
                problems.append(f"lifecycle entry {index} invalid")
                continue
            core = dict(row)
            declared = core.pop("entry_sha256", None)
            if (
                row.get("memory_ref") != memory_ref
                or row.get("sequence") != index
                or row.get("revision") != index + 1
                or row.get("previous_entry_sha256") != previous
                or _canonical_sha256(core) != declared
            ):
                problems.append(f"lifecycle entry {index} chain drift")
            previous = declared if isinstance(declared, str) else None
        if payload.get("head_sha256") != previous:
            problems.append("lifecycle head digest drift")
        deleted = entries[-1].get("operation") == "delete"
        if entries[0].get("operation") != "record" or not deleted:
            problems.append("lifecycle record-to-delete contract drift")
    except (TypeError, ValueError) as exc:
        problems.append(f"lifecycle authority invalid: {exc}")
    return deleted, problems


def _predict(row: list[int]) -> int:
    return max(range(len(row)), key=lambda label: (row[label], -label))


def _accuracy_ratio(numerator: object, denominator: object, *, label: str) -> float:
    if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError(f"{label} counts invalid")
    if not 0 <= numerator <= denominator:
        raise ValueError(f"{label} numerator outside denominator")
    return numerator / denominator


def _recompute_cell_metrics(
    state: dict[str, Any],
    *,
    arm: str,
    plan: dict[str, Any],
    stream_disk_bytes: int,
    lifecycle_deleted: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    problems: list[str] = []
    rung = int(plan["rung"])
    profile = plan["profile"]
    stream = plan["stream"]
    n_classes = int(stream["n_classes"])
    n_domains = int(stream["n_domains"])
    try:
        counts = state["counts"]
        if (
            not isinstance(counts, list)
            or len(counts) != n_classes
            or any(
                not isinstance(row, list)
                or len(row) != n_classes
                or any(not isinstance(value, int) or value < 0 for value in row)
                for row in counts
            )
        ):
            raise ValueError("learner count matrix invalid")
        domain_correct = state["domain_correct"]
        domain_total = state["domain_total"]
        if (
            not isinstance(domain_correct, list)
            or not isinstance(domain_total, list)
            or len(domain_correct) != n_domains
            or len(domain_total) != n_domains
        ):
            raise ValueError("per-domain count vectors invalid")
        future_outcomes = state["future_outcomes"]
        replay = state["replay"]
        if not isinstance(future_outcomes, list) or any(value not in (0, 1) for value in future_outcomes):
            raise ValueError("future outcome window invalid")
        if not isinstance(replay, list) or any(not isinstance(row, dict) for row in replay):
            raise ValueError("replay state invalid")
        if state.get("next_sequence") != rung or state.get("total") != rung:
            problems.append("checkpoint cursor or processed event count drift")
        if state.get("updates") != rung * int(profile["matched_updates_per_event"]):
            problems.append("checkpoint matched update count drift")
        if len(future_outcomes) != int(profile["future_window_events"]):
            problems.append("checkpoint future window length drift")
        capacity = int(profile["replay_capacity"])
        if len(replay) > capacity or int(state.get("max_replay_records", -1)) > capacity:
            problems.append("checkpoint replay capacity exceeded")
        remaining_deleted = sum(int(row.get("domain") == 0) for row in replay)
        if remaining_deleted != 0 or state.get("deletion_seen") is not True or not lifecycle_deleted:
            problems.append("checkpoint deletion contract drift")
        replay_samples = int(state["replay_samples"])
        if (arm == "replay" and replay_samples <= 0) or (arm != "replay" and replay_samples != 0):
            problems.append("checkpoint replay control accounting drift")
        resets = int(state["resets"])
        if (arm == "fresh-init" and resets != n_domains - 1) or (arm != "fresh-init" and resets != 0):
            problems.append("checkpoint fresh-init control accounting drift")

        domain_zero_accuracy = sum(int(_predict(row) == cue) for cue, row in enumerate(counts)) / n_classes
        total = int(state["total"])
        updates = int(state["updates"])
        stale_opportunities = int(state["stale_opportunities"])
        stale_harm = int(state["stale_harm_count"])
        metrics = {
            "retention": {
                "domain_zero_final_accuracy": domain_zero_accuracy,
                "transition_snapshots": state["anchor_snapshots"],
            },
            "acquisition": {
                "stream_accuracy": _accuracy_ratio(state["correct"], total, label="stream accuracy"),
                "per_domain_accuracy": [
                    _accuracy_ratio(correct, count, label=f"domain {index} accuracy")
                    for index, (correct, count) in enumerate(zip(domain_correct, domain_total, strict=True))
                ],
            },
            "future_learnability": {
                "first_window_accuracy": sum(future_outcomes) / len(future_outcomes),
                "window_events": len(future_outcomes),
                "events_to_threshold": state["future_events_to_threshold"],
                "threshold": float(profile["future_accuracy_threshold"]),
                "threshold_window_events": int(profile["threshold_window_events"]),
            },
            "stale_memory": {
                "opportunities": stale_opportunities,
                "harm_count": stale_harm,
                "harm_rate": stale_harm / max(1, stale_opportunities),
            },
            "deletion": {
                "requested": bool(state["deletion_seen"]),
                "replay_records_removed": int(state["deletion_removed"]),
                "remaining_deleted_domain_records": remaining_deleted,
                "lifecycle_deleted": lifecycle_deleted,
                "lifecycle_available_after_delete": False,
                "complete": bool(state["deletion_seen"] and remaining_deleted == 0 and lifecycle_deleted),
            },
            "resources": {
                "events_processed": total,
                "updates": updates,
                "updates_per_event": updates / total,
                "replay_samples": replay_samples,
                "max_replay_records": int(state["max_replay_records"]),
                "replay_capacity": capacity,
                "stream_disk_bytes": stream_disk_bytes,
                "checkpoint_state_bytes": len(canonical_bytes(state)),
                "model_weights_loaded": False,
                "accelerator_required": False,
            },
        }
        controls = {
            "replay_enabled": arm == "replay",
            "fresh_init_on_transition": arm == "fresh-init",
            "matched_updates_per_event": int(profile["matched_updates_per_event"]),
            "actual_updates_per_event": updates / total,
            "fixed_topology": True,
            "reset_count": resets,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"checkpoint metric state invalid: {exc}") from exc
    return metrics, controls, problems


def _checkpoint_cell(
    *,
    key: str,
    row: dict[str, Any],
    plan: dict[str, Any],
    work_root: Path,
    replay_authority: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    try:
        seed = int(row["seed"])
        schedule = str(row["schedule"])
        arm = str(row["arm"])
        expected_state = replay_authority["states"][arm]
        expected_lifecycle = replay_authority["lifecycle"]
        expected_stream_identity = replay_authority["identity_sha256"]
        expected_stream_sha = replay_authority["stream_sha256"]
        stream_bytes = int(replay_authority["disk_bytes"])
        checkpoint_path = work_root / "checkpoints" / f"seed_{seed}" / schedule / f"{arm}.json"
        checkpoint = _read_json(checkpoint_path)
        checkpoint_sha = _sha256_file(checkpoint_path)
        if checkpoint_sha != row.get("checkpoint_sha256"):
            problems.append(f"cell {key} checkpoint file digest drift")
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("complete") is not True:
            problems.append(f"cell {key} checkpoint schema or completion drift")
        checkpoint_identity = checkpoint.get("identity")
        if not isinstance(checkpoint_identity, dict) or checkpoint.get(
            "identity_sha256"
        ) != _canonical_sha256(checkpoint_identity):
            problems.append(f"cell {key} checkpoint identity digest drift")
            checkpoint_identity = checkpoint_identity if isinstance(checkpoint_identity, dict) else {}
        expected_checkpoint_identity = {
            "stream_identity_sha256": expected_stream_identity,
            "stream_sha256": expected_stream_sha,
            "arm": arm,
            "profile": plan["profile"],
            "claim_scope": CLAIM_SCOPE,
        }
        if checkpoint_identity != expected_checkpoint_identity:
            problems.append(f"cell {key} checkpoint identity join drift")
        state = checkpoint.get("state")
        if not isinstance(state, dict) or checkpoint.get("state_sha256") != _canonical_sha256(state):
            raise ValueError("checkpoint state or state digest invalid")
        if state != expected_state:
            problems.append(f"cell {key} checkpoint state differs from independent raw-event replay")
        if checkpoint.get("state_sha256") != row.get("state_sha256"):
            problems.append(f"cell {key} checkpoint state join drift")
        result = checkpoint.get("result")
        if not isinstance(result, dict):
            raise ValueError("complete checkpoint result missing")
        lifecycle_deleted, lifecycle_problems = _audit_lifecycle(
            expected_lifecycle, _canonical_sha256(expected_lifecycle)
        )
        problems.extend(f"cell {key} {problem}" for problem in lifecycle_problems)
        metrics, controls, recompute_problems = _recompute_cell_metrics(
            expected_state,
            arm=arm,
            plan=plan,
            stream_disk_bytes=stream_bytes,
            lifecycle_deleted=lifecycle_deleted,
        )
        problems.extend(f"cell {key} {problem}" for problem in recompute_problems)
        resumed = result.get("resumed_from_atomic_checkpoint")
        if not isinstance(resumed, bool):
            problems.append(f"cell {key} checkpoint result resume observation invalid")
            resumed = False
        expected_result = {
            "schema": "mop-continual-smoke-result/v1",
            "claim_scope": CLAIM_SCOPE,
            "arm": arm,
            "stream_identity_sha256": expected_stream_identity,
            "stream_sha256": expected_stream_sha,
            "complete": True,
            "resumed_from_atomic_checkpoint": resumed,
            "metrics": metrics,
            "controls": controls,
            "lifecycle": expected_lifecycle,
            "lifecycle_sha256": _canonical_sha256(expected_lifecycle),
            "lifecycle_errors": [],
            "state_sha256": _canonical_sha256(expected_state),
            "all_mechanics_ok": not recompute_problems,
        }
        if result != expected_result:
            problems.append(f"cell {key} checkpoint result differs from independent full recompute")
        authority_cell = {
            "seed": seed,
            "schedule": schedule,
            "arm": arm,
            "stream_identity_sha256": expected_stream_identity,
            "stream_sha256": expected_stream_sha,
            "checkpoint_sha256": checkpoint_sha,
            "state_sha256": _canonical_sha256(expected_state),
            "metrics": metrics,
            "controls": controls,
            "all_mechanics_ok": not recompute_problems and result == expected_result,
            "resumed_from_atomic_checkpoint": resumed,
        }
        return authority_cell, problems
    except (KeyError, OSError, TypeError, ValueError) as exc:
        problems.append(f"cell {key} checkpoint authority invalid: {exc}")
        return None, problems


def _live_cell_authorities(
    receipt: dict[str, Any], *, repo_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    problems: list[str] = []
    authority_cells: dict[str, dict[str, Any]] = {}
    progress_summary: dict[str, Any] = {}
    try:
        identity = receipt["identity"]
        plan = receipt["plan"]
        cells = receipt["cells"]
        progress_binding = receipt["progress"]
        progress_path = _repo_path(progress_binding.get("path"), repo_root=repo_root)
        progress = _read_json(progress_path)
        progress_sha = _sha256_file(progress_path)
        if progress_sha != progress_binding.get("sha256"):
            problems.append("live progress file digest drift")
        if (
            progress.get("schema") != PROGRESS_SCHEMA
            or progress.get("identity") != identity
            or progress.get("identity_sha256") != _canonical_sha256(identity)
            or progress.get("complete") is not True
            or progress.get("cells") != cells
        ):
            problems.append("live progress authority or cell join drift")
        work_root = progress_path.parent
        replay_authorities: dict[tuple[int, str], dict[str, Any]] = {}
        for seed_value in plan.get("seeds", []):
            for schedule_value in plan.get("schedules", []):
                seed = int(seed_value)
                schedule = str(schedule_value)
                replay, stream_problems = _replay_stream(
                    work_root / "streams" / f"seed_{seed}" / schedule,
                    plan=plan,
                    seed=seed,
                    schedule=schedule,
                    arms=tuple(str(value) for value in plan.get("arms", [])),
                )
                replay_authorities[(seed, schedule)] = replay
                problems.extend(f"stream seed_{seed}/{schedule} {problem}" for problem in stream_problems)
        for key, row in sorted(cells.items()):
            if not isinstance(row, dict):
                problems.append(f"cell {key} is not an object")
                continue
            coordinate = (int(row.get("seed", -1)), str(row.get("schedule")))
            replay_authority = replay_authorities.get(coordinate)
            if replay_authority is None:
                problems.append(f"cell {key} has no independently replayed stream authority")
                continue
            authority, cell_problems = _checkpoint_cell(
                key=key,
                row=row,
                plan=plan,
                work_root=work_root,
                replay_authority=replay_authority,
            )
            problems.extend(cell_problems)
            if authority is not None:
                authority_cells[key] = authority
        progress_summary = {
            "path": str(progress_binding.get("path")),
            "file_sha256": progress_sha,
            "identity_sha256": progress.get("identity_sha256"),
            "complete": progress.get("complete") is True,
            "cell_count": len(progress.get("cells", {})) if isinstance(progress.get("cells"), dict) else 0,
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        problems.append(f"live progress authority invalid: {exc}")
    return authority_cells, progress_summary, problems


def audit_rung_semantics(receipt: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:

    authority_cells, progress_summary, problems = _live_cell_authorities(receipt, repo_root=repo_root)
    plan = receipt.get("plan")
    cells = receipt.get("cells")
    if not isinstance(plan, dict) or not isinstance(cells, dict):
        problems.append("semantic rung plan or cells missing")
        expected_keys: set[str] = set()
    else:
        try:
            expected_keys, _ = _expected_matrix(plan)
        except (KeyError, TypeError, ValueError) as exc:
            expected_keys = set()
            problems.append(f"semantic rung matrix invalid: {exc}")
        if set(cells) != expected_keys:
            problems.append("semantic rung cell matrix drift")
        for key in sorted(expected_keys & set(cells)):
            if cells.get(key) != authority_cells.get(key):
                problems.append(f"semantic rung cell {key} differs from raw-event authority")
    all_ok = not problems and len(authority_cells) == len(expected_keys)
    semantic_authority = {
        "progress": progress_summary,
        "cells": authority_cells,
        "expected_cell_count": len(expected_keys),
    }
    return {
        "schema": "mop-continual-progressive-rung-semantic-audit/v1",
        "all_ok": all_ok,
        "errors": problems,
        "cell_count": len(authority_cells),
        "semantic_authority_sha256": _canonical_sha256(semantic_authority),
    }


def _decimal_mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _decision(cells: dict[str, dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    contrasts: list[dict[str, Any]] = []
    seeds = tuple(int(seed) for seed in plan["seeds"])
    for schedule in SCHEDULES:
        for control in ARMS[1:]:
            retention_deltas: list[Decimal] = []
            future_deltas: list[Decimal] = []
            paired: list[dict[str, Any]] = []
            for seed in seeds:
                replay = cells[f"seed_{seed}/{schedule}/replay"]
                comparator = cells[f"seed_{seed}/{schedule}/{control}"]
                replay_retention = Decimal(str(replay["metrics"]["retention"]["domain_zero_final_accuracy"]))
                control_retention = Decimal(
                    str(comparator["metrics"]["retention"]["domain_zero_final_accuracy"])
                )
                replay_future = Decimal(
                    str(replay["metrics"]["future_learnability"]["first_window_accuracy"])
                )
                control_future = Decimal(
                    str(comparator["metrics"]["future_learnability"]["first_window_accuracy"])
                )
                retention_delta = replay_retention - control_retention
                future_delta = replay_future - control_future
                retention_deltas.append(retention_delta)
                future_deltas.append(future_delta)
                tie = retention_delta == 0 or future_delta == 0
                nonpositive = retention_delta <= 0 or future_delta <= 0
                paired.append(
                    {
                        "seed": seed,
                        "retention_delta": float(retention_delta),
                        "future_first_window_delta": float(future_delta),
                        "tie_is_null": tie,
                        "nonpositive_is_null": nonpositive,
                    }
                )
            retention_mean = _decimal_mean(retention_deltas)
            future_mean = _decimal_mean(future_deltas)
            aggregate_tie = retention_mean == 0 or future_mean == 0
            any_seed_nonpositive = any(row["nonpositive_is_null"] for row in paired)
            null_contrast = aggregate_tie or any_seed_nonpositive
            strict_joint_gain = retention_mean > 0 and future_mean > 0 and not null_contrast
            contrasts.append(
                {
                    "schedule": schedule,
                    "control": control,
                    "independent_units": len(seeds),
                    "paired_seed_deltas": paired,
                    "retention_mean_delta": float(retention_mean),
                    "future_first_window_mean_delta": float(future_mean),
                    "aggregate_tie_is_null": aggregate_tie,
                    "any_seed_nonpositive_is_null": any_seed_nonpositive,
                    "null_contrast": null_contrast,
                    "strict_joint_gain": strict_joint_gain,
                }
            )
    favorable = all(row["strict_joint_gain"] is True for row in contrasts)
    tie_count = sum(int(row["null_contrast"]) for row in contrasts)
    return {
        "primary_endpoints": [
            "retention.domain_zero_final_accuracy",
            "future_learnability.first_window_accuracy",
        ],
        "independent_unit": "seed within transition schedule",
        "controls": list(ARMS[1:]),
        "tie_rule": TIE_RULE,
        "contrasts": contrasts,
        "aggregate_tie_count": tie_count,
        "strict_joint_gain_all_schedules_and_controls": favorable,
        "verdict": "favorable-rung-pattern" if favorable else "null",
        "null_supported": not favorable,
        "scientific_promotion": False,
        "claim_boundary": (
            "independently verified programmatic rung pattern only; mechanics cannot establish "
            "natural continual-learning capability"
        ),
    }


def _next_rung_authority(
    *,
    rung: int | None,
    source_file_sha256: str | None,
    source_identity_sha256: object,
    verification_complete: bool,
    valid_controls: bool,
    mutations_all_rejected: bool,
    favorable: bool,
) -> dict[str, Any]:
    next_rung = {10_000: 100_000, 100_000: 1_000_000}.get(rung) if rung is not None else None
    next_allowed = bool(verification_complete and next_rung is not None and favorable)
    return {
        "source_rung": rung,
        "source_rung_file_sha256": source_file_sha256,
        "source_identity_sha256": source_identity_sha256,
        "verification_complete": verification_complete,
        "valid_controls": valid_controls,
        "tie_is_null": True,
        "mutation_suite_all_rejected": mutations_all_rejected,
        "next_rung": next_rung,
        "next_rung_allowed": next_allowed,
        "next_rung_reason": (
            "strict favorable programmatic pattern requires the next scale confirmation"
            if next_allowed
            else "verified tie, null, invalid evidence, or final rung does not admit scaling"
        ),
    }


def _repair_identity_and_payload(receipt: dict[str, Any]) -> None:
    identity = receipt.get("identity")
    if isinstance(identity, dict):
        receipt["identity_sha256"] = _canonical_sha256(identity)
    core = dict(receipt)
    core.pop("payload_sha256", None)
    receipt["payload_sha256"] = _canonical_sha256(core)


def _mutation_suite(
    receipt: dict[str, Any],
    *,
    rung_config: dict[str, Any],
    expected_dependencies: dict[str, Any],
    authority_cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first_key = sorted(authority_cells)[0]
    second_key = sorted(authority_cells)[1]
    replay_key = next(key for key in sorted(authority_cells) if key.endswith("/replay"))

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("drop-completed-cell", lambda row: row["cells"].pop(first_key)),
        ("duplicate-plan-cell", lambda row: row["plan"]["cells"].append(row["plan"]["cells"][0])),
        ("cell-coordinate", lambda row: row["cells"][first_key].__setitem__("seed", -1)),
        (
            "replay-control-flag",
            lambda row: row["cells"][replay_key]["controls"].__setitem__("replay_enabled", False),
        ),
        (
            "matched-update-control",
            lambda row: row["cells"][second_key]["controls"].__setitem__("actual_updates_per_event", 1.0),
        ),
        (
            "retention-metric",
            lambda row: row["cells"][first_key]["metrics"]["retention"].__setitem__(
                "domain_zero_final_accuracy", 0.125
            ),
        ),
        (
            "stale-harm-rate",
            lambda row: row["cells"][first_key]["metrics"]["stale_memory"].__setitem__("harm_rate", 0.75),
        ),
        (
            "checkpoint-join",
            lambda row: row["cells"][first_key].__setitem__("checkpoint_sha256", "0" * 64),
        ),
        (
            "source-self-promotion",
            lambda row: row.__setitem__("scientific_promotion", True),
        ),
        (
            "progress-completion-count",
            lambda row: row["progress"].__setitem__("completed_cells", 29),
        ),
        (
            "live-source-binding",
            lambda row: row["identity"].__setitem__("source_live_bindings_sha256", "0" * 64),
        ),
        (
            "cross-cell-copy",
            lambda row: row["cells"].__setitem__(first_key, copy.deepcopy(row["cells"][second_key])),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(receipt)
        try:
            mutate(candidate)
            if isinstance(candidate.get("identity"), dict) and isinstance(
                candidate["identity"].get("plan"), dict
            ):
                candidate["identity"]["plan"] = copy.deepcopy(candidate.get("plan"))
            _repair_identity_and_payload(candidate)
            problems = _validate_structure(
                candidate,
                rung_config=rung_config,
                expected_dependencies=expected_dependencies,
                authority_cells=authority_cells,
            )
        except (KeyError, TypeError, ValueError) as exc:
            problems = [f"mutation construction or audit rejected: {exc}"]
        rows.append({"mutation": name, "rejected": bool(problems), "problems": problems})
    return {
        "count": len(rows),
        "rejected": sum(int(row["rejected"]) for row in rows),
        "all_rejected": all(row["rejected"] for row in rows),
        "mutations": rows,
    }


def build_verification_receipt(
    source_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    problems: list[str] = []
    try:
        receipt = _read_json(source)
    except (OSError, TypeError, ValueError) as exc:
        receipt = {}
        problems.append(f"source rung receipt invalid: {exc}")

    identity_value = receipt.get("identity")
    identity: dict[str, Any] = identity_value if isinstance(identity_value, dict) else {}
    rung_config, expected_dependencies, dependency_problems = _live_dependencies(
        identity,
        receipt.get("source_live_authority"),
        repo_root=repo_root,
    )
    problems.extend(dependency_problems)
    structural_problems = _validate_structure(
        receipt,
        rung_config=rung_config,
        expected_dependencies=expected_dependencies,
    )
    problems.extend(structural_problems)
    authority_cells: dict[str, dict[str, Any]] = {}
    progress_summary: dict[str, Any] = {}
    authority_problems: list[str] = []
    if not dependency_problems and not structural_problems:
        authority_cells, progress_summary, authority_problems = _live_cell_authorities(
            receipt, repo_root=repo_root
        )
        problems.extend(authority_problems)
        authority_join_problems = _validate_structure(
            receipt,
            rung_config=rung_config,
            expected_dependencies=expected_dependencies,
            authority_cells=authority_cells,
        )
        problems.extend(authority_join_problems)
        structural_problems.extend(authority_join_problems)

    plan_value = receipt.get("plan")
    plan: dict[str, Any] = plan_value if isinstance(plan_value, dict) else {}
    expected_cell_count = int(plan.get("expected_cells", 0))
    authorities_complete = len(authority_cells) == expected_cell_count == 30
    if not authorities_complete:
        problems.append("independent checkpoint authority matrix is incomplete")
    decision: dict[str, Any] = {}
    if authorities_complete:
        try:
            decision = _decision(authority_cells, plan)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            problems.append(f"independent paired decision recompute failed: {exc}")
    mutations = (
        _mutation_suite(
            receipt,
            rung_config=rung_config,
            expected_dependencies=expected_dependencies,
            authority_cells=authority_cells,
        )
        if authorities_complete and not structural_problems
        else {"count": 0, "rejected": 0, "all_rejected": False, "mutations": []}
    )
    if mutations["all_rejected"] is not True:
        problems.append("independent mutation suite did not reject every mutation")

    verification_complete = not problems
    rung_value = receipt.get("rung")
    rung = rung_value if isinstance(rung_value, int) and not isinstance(rung_value, bool) else None
    favorable = decision.get("strict_joint_gain_all_schedules_and_controls") is True
    source_file_sha = _sha256_file(source) if source.is_file() else None
    valid_controls = not any("control" in problem for problem in structural_problems + authority_problems)
    core: dict[str, Any] = {
        "schema": VERIFIER_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "source_rung": {
            "path": _display_path(source, repo_root=repo_root),
            "file_sha256": source_file_sha,
            "payload_sha256": receipt.get("payload_sha256"),
            "identity_sha256": receipt.get("identity_sha256"),
            "rung": rung,
            "mode": receipt.get("mode"),
        },
        "live_dependencies": expected_dependencies,
        "progress_authority": progress_summary,
        "independent_recompute": {
            "cell_count": len(authority_cells),
            "metric_families": list(METRIC_FAMILIES),
            "checkpoint_state_recomputed": authorities_complete,
            "controls_recomputed": authorities_complete,
            "paired_metrics_recomputed": bool(decision),
            "decision": decision,
        },
        "mutation_suite": mutations,
        "checks": {
            "source_payload_self_hash": _source_payload_ok(receipt) if receipt else False,
            "live_dependencies_current": not dependency_problems,
            "progress_and_checkpoints_current": not authority_problems and authorities_complete,
            "full_replication_structure_valid": not structural_problems,
            "all_metrics_independently_recomputed": authorities_complete,
            "all_controls_present_and_valid": valid_controls,
            "tie_is_null": decision.get("tie_rule") == TIE_RULE,
            "all_mutations_rejected": mutations["all_rejected"],
            "scientific_promotion_blocked": decision.get("scientific_promotion") is False,
        },
        "verification_complete": verification_complete,
        "errors": problems,
        "prerequisite": _next_rung_authority(
            rung=rung,
            source_file_sha256=source_file_sha,
            source_identity_sha256=receipt.get("identity_sha256"),
            verification_complete=verification_complete,
            valid_controls=valid_controls,
            mutations_all_rejected=bool(mutations["all_rejected"]),
            favorable=favorable,
        ),
        "scientific_promotion": False,
        "implementation": [
            {
                "path": path,
                "sha256": _sha256_file(_repo_path(path, repo_root=repo_root)),
            }
            for path in IMPLEMENTATION_PATHS
            if _repo_path(path, repo_root=repo_root).is_file()
        ],
    }
    core["payload_sha256"] = _canonical_sha256(core)
    return core


def write_verification_receipt(
    source_path: Path | str,
    output_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    receipt = build_verification_receipt(source_path, repo_root=repo_root)
    _atomic_json(Path(output_path).resolve(), receipt)
    return receipt
