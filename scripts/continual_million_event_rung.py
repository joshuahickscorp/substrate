#!/usr/bin/env python
"""Run one progressive continual-event rung with exact serial resume authorities.

The default mode is the full 2 schedule x 3 arm x 5 seed execution matrix. ``--resource-probe`` is
restricted to the first 10k rung and one canonical abrupt/replay cell. Every stream chunk, arm
checkpoint, progress receipt, and final receipt is atomically published. This runner never loads
model weights and never requests an accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any

import yaml

from mop.config import REPO_ROOT
from mop.studies.continual_million_event import (
    ARMS,
    CLAIM_SCOPE,
    ContinualSmokeProfile,
    run_smoke_arm,
)
from mop.substrate.continual_stream import (
    ContinualStreamSpec,
    TransitionSchedule,
    materialize_stream,
    verify_stream,
)
from mop.substrate.events import canonical_sha256

CONFIG_SCHEMA = "mop-continual-progressive-rungs-config/v1"
PROGRESS_SCHEMA = "mop-continual-progressive-rung-progress/v1"
RESULT_SCHEMA = "mop-continual-progressive-rung/v1"
SOURCE_SCHEMA = "mop-continual-million-event-preflight/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/experiment/continual_million_event_rungs.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _repo_path(value: object, *, repo_root: Path = REPO_ROOT) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("continual source binding path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"continual source binding path escapes the repository: {value!r}")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"continual source binding path escapes the repository: {value!r}")
    return path


def _validate_source_live_bindings(receipt: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Revalidate every live authority embedded by the source preflight.

    The preflight file digest alone is not enough for resume. Its purpose is to bind the live
    implementation and config that produced it, so those embedded bindings must still resolve to
    the same bytes every time a progressive rung starts or resumes.
    """

    config = receipt.get("config")
    if not isinstance(config, dict):
        raise ValueError("continual source preflight config binding missing")
    config_path = _repo_path(config.get("path"), repo_root=repo_root)
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != config.get("sha256"):
        raise ValueError("continual source preflight live config hash drift")
    config_payload = config.get("payload")
    if not isinstance(config_payload, dict):
        raise ValueError("continual source preflight embedded config payload missing")
    if canonical_sha256(config_payload) != config.get("profile_sha256"):
        raise ValueError("continual source preflight embedded config digest drift")
    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config_payload:
        raise ValueError("continual source preflight live config payload drift")

    implementation = receipt.get("implementation")
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("continual source preflight implementation bindings missing")
    implementation_rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, row in enumerate(implementation):
        if not isinstance(row, dict):
            raise ValueError(f"continual source implementation binding {index} is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise ValueError("continual source implementation binding path is missing or duplicated")
        path = _repo_path(relative, repo_root=repo_root)
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != row.get("sha256"):
            raise ValueError(f"continual source preflight live implementation drift: {relative}")
        seen_paths.add(relative)
        implementation_rows.append({"path": relative, "sha256": observed_sha256})
    if str(config.get("path")) not in seen_paths:
        raise ValueError("continual source config is absent from implementation bindings")

    upstream = receipt.get("wave_e0")
    if not isinstance(upstream, dict):
        raise ValueError("continual source preflight Wave E0 binding missing")
    upstream_path = _repo_path(upstream.get("path"), repo_root=repo_root)
    upstream_sha256 = _sha256_file(upstream_path)
    if upstream_sha256 != upstream.get("sha256"):
        raise ValueError("continual source preflight live Wave E0 hash drift")

    authority = {
        "config": {
            "path": str(config["path"]),
            "sha256": config_sha256,
            "payload_sha256": str(config["profile_sha256"]),
        },
        "implementation": implementation_rows,
        "wave_e0": {"path": str(upstream["path"]), "sha256": upstream_sha256},
    }
    authority["bindings_sha256"] = canonical_sha256(authority)
    return authority


def _source_receipt(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = config["source_preflight"]
    path = (REPO_ROOT / str(source["path"])).resolve()
    if _sha256_file(path) != source["file_sha256"]:
        raise ValueError("continual source preflight file hash drift")
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict) or receipt.get("schema") != SOURCE_SCHEMA:
        raise ValueError("continual source preflight schema drift")
    payload = dict(receipt)
    declared_payload_sha = payload.pop("payload_sha256", None)
    if canonical_sha256(payload) != declared_payload_sha or declared_payload_sha != source["payload_sha256"]:
        raise ValueError("continual source preflight payload digest drift")
    if receipt.get("all_mechanics_ok") is not True or receipt.get("status") != "mechanics-pass":
        raise ValueError("continual source preflight mechanics did not pass")
    resources = [
        arm["result"]["metrics"]["resources"]
        for schedule in receipt.get("schedules", [])
        for arm in schedule.get("arms", [])
    ]
    observed = {
        "events_per_stream": receipt.get("resource_envelope", {}).get("configured_stream_events"),
        "stream_disk_bytes": max(int(row["stream_disk_bytes"]) for row in resources),
        "max_checkpoint_state_bytes": max(int(row["checkpoint_state_bytes"]) for row in resources),
    }
    expected = {
        "events_per_stream": int(source["observed_events_per_stream"]),
        "stream_disk_bytes": int(source["observed_stream_disk_bytes"]),
        "max_checkpoint_state_bytes": int(source["observed_max_checkpoint_state_bytes"]),
    }
    if observed != expected:
        raise ValueError(f"continual source resource evidence drift: {observed} != {expected}")
    return receipt, _validate_source_live_bindings(receipt)


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"progressive continual config schema must be {CONFIG_SCHEMA}")
    if payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("progressive continual claim scope drift")
    replication = payload.get("replication")
    if not isinstance(replication, dict):
        raise ValueError("progressive continual replication config missing")
    if tuple(int(value) for value in replication.get("rungs", [])) != (10_000, 100_000, 1_000_000):
        raise ValueError("progressive continual rungs must be 10k, 100k, then 1m")
    seeds = tuple(int(value) for value in replication.get("seeds", []))
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("progressive continual replication needs at least five unique seeds")
    if tuple(replication.get("schedules", [])) != ("abrupt", "gradual"):
        raise ValueError("progressive continual schedules must be abrupt then gradual")
    if tuple(replication.get("arms", [])) != ARMS:
        raise ValueError("progressive continual arms or order drift")
    if int(replication.get("minimum_independent_seeds", 0)) < 5:
        raise ValueError("progressive continual minimum independent seeds must be at least five")
    if replication.get("independent_metric_verifier_required") is not True:
        raise ValueError("progressive continual independent metric verifier must remain required")
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("progressive continual profile config missing")
    expected_profile = {
        "replay_capacity": 128,
        "future_accuracy_threshold": 0.5,
        "matched_updates_per_event": 2,
        "checkpoint_every_events": 1_250,
        "future_window_events": 48,
        "threshold_window_events": 16,
    }
    for field, expected in expected_profile.items():
        if profile.get(field) != expected:
            raise ValueError(f"progressive continual profile {field} drift")
    source_receipt, source_live_authority = _source_receipt(payload)
    payload["_config_path"] = str(config_path)
    payload["_config_sha256"] = _sha256_file(config_path)
    payload["_source_preflight_payload_sha256"] = source_receipt["payload_sha256"]
    payload["_source_live_authority"] = source_live_authority
    return payload


def build_plan(
    config: dict[str, Any],
    rung: int,
    *,
    resource_probe: bool = False,
    seed_count: int | None = None,
    schedules: tuple[str, ...] | None = None,
    arms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    replication = config["replication"]
    allowed_rungs = tuple(int(value) for value in replication["rungs"])
    if rung not in allowed_rungs:
        raise ValueError(f"rung {rung} is not one of {allowed_rungs}")
    all_seeds = tuple(int(value) for value in replication["seeds"])
    count = len(all_seeds) if seed_count is None else int(seed_count)
    if not 1 <= count <= len(all_seeds):
        raise ValueError("seed_count is outside the preregistered seed list")
    selected_schedules = schedules or tuple(str(value) for value in replication["schedules"])
    selected_arms = arms or tuple(str(value) for value in replication["arms"])
    if resource_probe:
        if rung != 10_000 or count != 1 or selected_schedules != ("abrupt",) or selected_arms != ("replay",):
            raise ValueError("resource probe is exactly 10k, first seed, abrupt schedule, replay arm")
    elif (
        count < int(replication["minimum_independent_seeds"])
        or selected_schedules != tuple(replication["schedules"])
        or selected_arms != tuple(replication["arms"])
    ):
        raise ValueError("full rung must retain every schedule and arm with at least five seeds")
    profile = config["profile"]
    chunk_events = max(int(profile["minimum_chunk_events"]), rung // int(profile["chunks_per_stream"]))
    cells = [
        {"seed": seed, "schedule": schedule, "arm": arm}
        for seed in all_seeds[:count]
        for schedule in selected_schedules
        for arm in selected_arms
    ]
    return {
        "mode": "resource-probe" if resource_probe else "replication",
        "rung": rung,
        "seeds": list(all_seeds[:count]),
        "schedules": list(selected_schedules),
        "arms": list(selected_arms),
        "cells": cells,
        "expected_cells": len(cells),
        "stream": {
            "chunk_events": chunk_events,
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


def _spec(plan: dict[str, Any], seed: int, schedule: str) -> ContinualStreamSpec:
    stream = plan["stream"]
    return ContinualStreamSpec(
        seed=seed,
        total_events=int(plan["rung"]),
        chunk_events=int(stream["chunk_events"]),
        n_domains=int(stream["n_domains"]),
        n_classes=int(stream["n_classes"]),
        transition_schedule=TransitionSchedule(schedule),
        gradual_width_events=int(stream["gradual_width_events"]),
        deletion_event=int(stream["deletion_event"]),
    )


def _profile(plan: dict[str, Any]) -> ContinualSmokeProfile:
    return ContinualSmokeProfile(**plan["profile"])


def _revalidate_live_identity(config: dict[str, Any], identity: dict[str, Any]) -> None:
    """Refuse dependency edits between cells as well as between resume invocations."""

    if _sha256_file(Path(config["_config_path"])) != identity["config_sha256"]:
        raise ValueError("progressive continual live rung config drift")
    if _sha256_file(Path(__file__).resolve()) != identity["runner_sha256"]:
        raise ValueError("progressive continual live runner drift")
    source_receipt, source_authority = _source_receipt(config)
    expected = {
        "source_preflight_file_sha256": config["source_preflight"]["file_sha256"],
        "source_preflight_payload_sha256": source_receipt["payload_sha256"],
        "source_live_bindings_sha256": source_authority["bindings_sha256"],
    }
    if any(identity.get(field) != value for field, value in expected.items()):
        raise ValueError("progressive continual live source identity drift")
    if source_authority != config["_source_live_authority"]:
        raise ValueError("progressive continual live source authority drift")


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _update_peak_rss(progress: dict[str, Any], *, observed_bytes: int | None = None) -> int:
    """Persist the maximum runner RSS across restartable invocations."""

    observed = _max_rss_bytes() if observed_bytes is None else int(observed_bytes)
    if observed < 0:
        raise ValueError("progressive continual RSS observation must be nonnegative")
    measurement = progress.get("resource_measurement")
    if measurement is None:
        measurement = {"max_rss_bytes": 0}
        progress["resource_measurement"] = measurement
    if not isinstance(measurement, dict):
        raise ValueError("progressive continual progress resource measurement drift")
    prior = measurement.get("max_rss_bytes", 0)
    if not isinstance(prior, int) or isinstance(prior, bool) or prior < 0:
        raise ValueError("progressive continual persisted RSS peak drift")
    peak = max(prior, observed)
    measurement["max_rss_bytes"] = peak
    measurement["rss_scope"] = "maximum runner-process RSS across persisted invocations"
    return peak


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.name.endswith(".tmp")
    )


def _progress(path: Path, identity: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if path.is_file():
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != PROGRESS_SCHEMA
            or payload.get("identity") != identity
            or payload.get("identity_sha256") != canonical_sha256(identity)
            or not isinstance(payload.get("cells"), dict)
        ):
            raise ValueError("progressive continual progress identity drift")
        measurement = payload.get("resource_measurement")
        if measurement is not None and (
            not isinstance(measurement, dict)
            or not isinstance(measurement.get("max_rss_bytes"), int)
            or isinstance(measurement.get("max_rss_bytes"), bool)
            or int(measurement["max_rss_bytes"]) < 0
        ):
            raise ValueError("progressive continual progress RSS authority drift")
        return payload, True
    payload = {
        "schema": PROGRESS_SCHEMA,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "cells": {},
        "complete": False,
        "resource_measurement": {
            "max_rss_bytes": 0,
            "rss_scope": "maximum runner-process RSS across persisted invocations",
        },
    }
    _atomic_json(path, payload)
    return payload, False


def run_rung(
    config_path: Path | str,
    work_root: Path | str,
    output: Path | str,
    *,
    rung: int,
    resource_probe: bool = False,
    seed_count: int | None = None,
    schedules: tuple[str, ...] | None = None,
    arms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = build_plan(
        config,
        rung,
        resource_probe=resource_probe,
        seed_count=seed_count,
        schedules=schedules,
        arms=arms,
    )
    root = Path(work_root).resolve()
    output_path = Path(output).resolve()
    identity = {
        "config_sha256": config["_config_sha256"],
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "source_preflight_file_sha256": config["source_preflight"]["file_sha256"],
        "source_preflight_payload_sha256": config["_source_preflight_payload_sha256"],
        "source_live_bindings_sha256": config["_source_live_authority"]["bindings_sha256"],
        "plan": plan,
        "claim_scope": CLAIM_SCOPE,
    }
    _revalidate_live_identity(config, identity)
    if output_path.is_file():
        existing = json.loads(output_path.read_text())
        existing_payload = dict(existing) if isinstance(existing, dict) else {}
        declared_payload_sha = existing_payload.pop("payload_sha256", None)
        if not (
            isinstance(existing, dict)
            and existing.get("schema") == RESULT_SCHEMA
            and existing.get("identity") == identity
            and existing.get("identity_sha256") == canonical_sha256(identity)
            and existing.get("source_live_authority") == config["_source_live_authority"]
            and existing.get("all_mechanics_ok") is True
            and canonical_sha256(existing_payload) == declared_payload_sha
        ):
            raise ValueError("existing progressive continual result identity drift")
    progress_path = root / "progress.json"
    progress, resumed = _progress(progress_path, identity)
    _update_peak_rss(progress)
    _atomic_json(progress_path, progress)
    smoke_profile = _profile(plan)
    for cell in plan["cells"]:
        try:
            _revalidate_live_identity(config, identity)
            seed = int(cell["seed"])
            schedule = str(cell["schedule"])
            arm = str(cell["arm"])
            key = f"seed_{seed}/{schedule}/{arm}"
            prior = progress["cells"].get(key)
            stream_root = root / "streams" / f"seed_{seed}" / schedule
            spec = _spec(plan, seed, schedule)
            checkpoint = root / "checkpoints" / f"seed_{seed}" / schedule / f"{arm}.json"
            if isinstance(prior, dict) and prior.get("all_mechanics_ok") is True:
                prior_audit = verify_stream(stream_root, expected_spec=spec, require_complete=True)
                if not prior_audit["verified"] or not checkpoint.is_file():
                    raise ValueError(f"completed progressive cell {key} lost its resume authority")
                verified_result = run_smoke_arm(
                    stream_root=stream_root,
                    spec=spec,
                    arm=arm,
                    profile=smoke_profile,
                    checkpoint_path=checkpoint,
                )
                if any(
                    prior.get(field) != verified_result.get(field)
                    for field in (
                        "stream_identity_sha256",
                        "stream_sha256",
                        "checkpoint_sha256",
                        "state_sha256",
                        "metrics",
                        "controls",
                        "all_mechanics_ok",
                    )
                ):
                    raise ValueError(f"completed progressive cell {key} checkpoint identity drift")
                continue
            materialize_stream(stream_root, spec)
            stream_audit = verify_stream(stream_root, expected_spec=spec, require_complete=True)
            if not stream_audit["verified"]:
                raise ValueError("progressive continual stream verification failed")
            result = run_smoke_arm(
                stream_root=stream_root,
                spec=spec,
                arm=arm,
                profile=smoke_profile,
                checkpoint_path=checkpoint,
            )
            progress["cells"][key] = {
                "seed": seed,
                "schedule": schedule,
                "arm": arm,
                "stream_identity_sha256": result["stream_identity_sha256"],
                "stream_sha256": result["stream_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "state_sha256": result["state_sha256"],
                "metrics": result["metrics"],
                "controls": result["controls"],
                "all_mechanics_ok": result["all_mechanics_ok"],
                "resumed_from_atomic_checkpoint": result["resumed_from_atomic_checkpoint"],
            }
            progress["complete"] = False
        finally:
            _update_peak_rss(progress)
            _atomic_json(progress_path, progress)
    _revalidate_live_identity(config, identity)
    expected_keys = {f"seed_{cell['seed']}/{cell['schedule']}/{cell['arm']}" for cell in plan["cells"]}
    progress["complete"] = set(progress["cells"]) == expected_keys and all(
        row.get("all_mechanics_ok") is True for row in progress["cells"].values()
    )
    current_invocation_max_rss = _max_rss_bytes()
    max_rss = _update_peak_rss(progress, observed_bytes=current_invocation_max_rss)
    _atomic_json(progress_path, progress)
    work_bytes = _tree_bytes(root)
    full_replication = bool(
        plan["mode"] == "replication"
        and len(plan["seeds"]) >= int(config["replication"]["minimum_independent_seeds"])
        and tuple(plan["schedules"]) == tuple(config["replication"]["schedules"])
        and tuple(plan["arms"]) == tuple(config["replication"]["arms"])
        and progress["complete"]
    )
    receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "source_live_authority": config["_source_live_authority"],
        "mode": plan["mode"],
        "rung": rung,
        "plan": plan,
        "progress": {
            "path": str(progress_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(progress_path),
            "resumed_existing_progress": resumed,
            "completed_cells": len(progress["cells"]),
            "expected_cells": plan["expected_cells"],
        },
        "cells": progress["cells"],
        "resource_measurement": {
            "max_rss_bytes": max_rss,
            "current_invocation_max_rss_bytes": current_invocation_max_rss,
            "rss_scope": "maximum runner-process RSS across persisted invocations",
            "work_root_bytes": work_bytes,
            "work_root": str(root),
            "events_per_stream": rung,
            "measured_after_complete": bool(progress["complete"]),
        },
        "all_mechanics_ok": bool(progress["complete"]),
        "replication_execution_complete": full_replication,
        "independent_metric_verifier_complete": False,
        "scientific_promotion": False,
        "claim_boundary": (
            "execution and resume mechanics only; an independent metric verifier remains required"
        ),
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    _atomic_json(output_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rung", type=int, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resource-probe", action="store_true")
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--schedules", nargs="+", choices=("abrupt", "gradual"))
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_rung(
        args.config,
        args.work_root,
        args.out,
        rung=args.rung,
        resource_probe=args.resource_probe,
        seed_count=args.seed_count,
        schedules=tuple(args.schedules) if args.schedules else None,
        arms=tuple(args.arms) if args.arms else None,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["all_mechanics_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
