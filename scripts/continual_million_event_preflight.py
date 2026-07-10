#!/usr/bin/env python
"""Run the no-heavy continual million-event scaffold preflight.

This command materializes tiny abrupt and gradual disk streams, interrupts and resumes each control
arm, records mechanics metrics, and writes a proof receipt. It refuses million-event execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from mop.config import REPO_ROOT
from mop.experiments.ex13_long_stream import EX13
from mop.experiments.ex15_rejuvenation import EX15
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

PREFLIGHT_SCHEMA = "mop-continual-million-event-preflight/v1"
CONFIG_SCHEMA = "mop-continual-million-event-preflight-config/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "continual_million_event_preflight.yaml"
DEFAULT_WORK_ROOT = REPO_ROOT / "runs" / "continual_million_event_preflight"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "CONTINUAL_MILLION_EVENT_PREFLIGHT.json"
IMPLEMENTATION_PATHS = (
    "src/mop/substrate/continual_stream.py",
    "src/mop/studies/continual_million_event.py",
    "scripts/continual_million_event_preflight.py",
    "configs/experiment/continual_million_event_preflight.yaml",
    "src/mop/substrate/events.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/experiments/ex13_long_stream.py",
    "src/mop/experiments/ex15_rejuvenation.py",
    "docs/P6_CONTINUAL_MILLION_EVENT_AUDIT_2026_07.md",
)
EXPECTED_METRICS = (
    "retention",
    "acquisition",
    "future_learnability",
    "stale_memory",
    "deletion",
    "resources",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("continual million-event preflight config schema drift")
    if payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("continual million-event preflight claim scope drift")
    stream = payload.get("stream", {})
    runner = payload.get("runner", {})
    guard = payload.get("no_heavy_guard", {})
    total_events = int(stream.get("total_events", 0))
    schedules = tuple(stream.get("schedules", ()))
    arms = tuple(runner.get("arms", ()))
    if schedules != ("abrupt", "gradual"):
        raise ValueError("preflight must cover abrupt and gradual schedules in canonical order")
    if arms != ARMS:
        raise ValueError("preflight control arm set or order drift")
    if tuple(payload.get("preregistered_metrics", ())) != EXPECTED_METRICS:
        raise ValueError("preflight metric schema drift")
    if total_events > int(guard.get("maximum_events_per_stream", 0)):
        raise ValueError("no-heavy guard rejects the configured events per stream")
    total_arm_events = total_events * len(schedules) * len(arms)
    if total_arm_events > int(guard.get("maximum_total_arm_events", 0)):
        raise ValueError("no-heavy guard rejects the configured total arm events")
    if guard.get("accelerator_required") is not False or guard.get("model_weights_loaded") is not False:
        raise ValueError("no-heavy preflight cannot request an accelerator or model weights")
    full_gate = payload.get("full_run_gate", {})
    if int(full_gate.get("target_events_per_stream", 0)) != 1_000_000:
        raise ValueError("full-run gate must retain the million-event target")
    return payload


def _profile(config: dict[str, Any]) -> ContinualSmokeProfile:
    runner = config["runner"]
    return ContinualSmokeProfile(
        checkpoint_every=int(runner["checkpoint_every"]),
        replay_capacity=int(runner["replay_capacity"]),
        future_window_events=int(runner["future_window_events"]),
        threshold_window_events=int(runner["threshold_window_events"]),
        future_accuracy_threshold=float(runner["future_accuracy_threshold"]),
        matched_updates_per_event=int(runner["matched_updates_per_event"]),
    )


def _stream_spec(config: dict[str, Any], schedule: str) -> ContinualStreamSpec:
    stream = config["stream"]
    return ContinualStreamSpec(
        seed=int(stream["seed"]),
        total_events=int(stream["total_events"]),
        chunk_events=int(stream["chunk_events"]),
        n_domains=int(stream["n_domains"]),
        n_classes=int(stream["n_classes"]),
        transition_schedule=TransitionSchedule(schedule),
        gradual_width_events=int(stream["gradual_width_events"]),
        deletion_event=int(stream["deletion_event"]),
    )


def build_preflight(config_path: Path, work_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    profile = _profile(config)
    schedule_rows: list[dict[str, Any]] = []
    interruption = int(config["runner"]["interruption_after_events"])
    for schedule in config["stream"]["schedules"]:
        spec = _stream_spec(config, schedule)
        stream_root = work_root / "streams" / schedule
        partial_manifest = materialize_stream(stream_root, spec, max_new_chunks=1)
        atomic_chunk_checkpoint_observed = len(partial_manifest.get("chunks", [])) >= 1
        manifest = materialize_stream(stream_root, spec)
        audit = verify_stream(stream_root, expected_spec=spec, require_complete=True)

        replica_root = work_root / "identity_replicas" / schedule
        replica = materialize_stream(replica_root, spec)
        replica_audit = verify_stream(replica_root, expected_spec=spec, require_complete=True)
        replica_match = bool(
            audit["verified"]
            and replica_audit["verified"]
            and manifest["stream_sha256"] == replica["stream_sha256"]
            and manifest["chain_head_sha256"] == replica["chain_head_sha256"]
        )

        arm_rows: list[dict[str, Any]] = []
        for arm in config["runner"]["arms"]:
            checkpoint = work_root / "checkpoints" / schedule / f"{arm}.json"
            partial = run_smoke_arm(
                stream_root=stream_root,
                spec=spec,
                arm=arm,
                profile=profile,
                checkpoint_path=checkpoint,
                event_budget=interruption,
            )
            final = run_smoke_arm(
                stream_root=stream_root,
                spec=spec,
                arm=arm,
                profile=profile,
                checkpoint_path=checkpoint,
            )
            arm_rows.append(
                {
                    "arm": arm,
                    "interruption_checkpoint_observed": partial.get("complete") is False
                    or partial.get("resumed_from_atomic_checkpoint") is True,
                    "result": final,
                }
            )
        schedule_rows.append(
            {
                "schedule": schedule,
                "spec": spec.payload(),
                "identity_sha256": spec.identity_sha256,
                "stream_sha256": manifest["stream_sha256"],
                "stream_audit": audit,
                "atomic_chunk_checkpoint_observed": atomic_chunk_checkpoint_observed,
                "independent_replica_identity_match": replica_match,
                "arms": arm_rows,
            }
        )

    all_results = [row["result"] for schedule in schedule_rows for row in schedule["arms"]]
    mechanics_ok = bool(
        all(schedule["stream_audit"]["verified"] for schedule in schedule_rows)
        and all(schedule["independent_replica_identity_match"] for schedule in schedule_rows)
        and all(result.get("all_mechanics_ok") is True for result in all_results)
        and all(set(result.get("metrics", {})) == set(EXPECTED_METRICS) for result in all_results)
        and all(result.get("resumed_from_atomic_checkpoint") is True for result in all_results)
    )
    full_gate = config["full_run_gate"]
    receipt: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if mechanics_ok else "mechanics-fail",
        "no_heavy_preflight": True,
        "audit": {
            "already_existed": [
                "EX13 registered forgetting-curve contract with replay and no-replay arms",
                "EX15 long-stream plasticity and rejuvenation traces",
                "PR9 per-seed and per-arm leg resume files",
                "Wave E0 typed events and append-only lifecycle journal",
                "in-memory ReplayBuffer with bounded capacity",
            ],
            "missing_before_this_preflight": [
                "disk-backed deterministic event stream",
                "hash-chained atomic event cursor resume",
                "gradual and abrupt schedule fixtures",
                "fresh-init fixed-topology control",
                "future-learnability, stale-memory, deletion, and exact resource metrics",
            ],
            "harness_reuse": (
                "EX13 and EX15 remain the registered scientific contracts. This preflight adds a shared "
                "stream and cursor layer and does not register a competing experiment harness."
            ),
        },
        "registry_contracts": {
            "ex13_long_stream": EX13().contract(),
            "ex15_rejuvenation": EX15().contract(),
        },
        "wave_e0": {
            "path": "proof/EXPANSION_WAVE0.json",
            "sha256": _sha256_file(REPO_ROOT / "proof" / "EXPANSION_WAVE0.json"),
            "reused_primitives": ["EventRef", "EntityRef", "LifecycleJournal"],
        },
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(config_path),
            "payload": config,
            "profile_sha256": canonical_sha256(config),
        },
        "implementation": [
            {"path": path, "sha256": _sha256_file(REPO_ROOT / path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "schedules": schedule_rows,
        "checks": {
            "both_transition_schedules": [row["schedule"] for row in schedule_rows]
            == ["abrupt", "gradual"],
            "all_streams_verified": all(
                row["stream_audit"]["verified"] for row in schedule_rows
            ),
            "all_identity_replicas_match": all(
                row["independent_replica_identity_match"] for row in schedule_rows
            ),
            "all_arms_resumed_atomically": all(
                result.get("resumed_from_atomic_checkpoint") is True for result in all_results
            ),
            "all_metric_families_present": all(
                set(result.get("metrics", {})) == set(EXPECTED_METRICS) for result in all_results
            ),
            "all_deletions_complete": all(
                result.get("metrics", {}).get("deletion", {}).get("complete") is True
                for result in all_results
            ),
            "all_compute_matched": all(
                result.get("controls", {}).get("actual_updates_per_event")
                == profile.matched_updates_per_event
                for result in all_results
            ),
            "no_model_weights_loaded": all(
                result.get("metrics", {}).get("resources", {}).get("model_weights_loaded") is False
                for result in all_results
            ),
        },
        "resource_envelope": {
            "configured_stream_events": int(config["stream"]["total_events"]),
            "configured_total_arm_events": int(config["stream"]["total_events"])
            * len(config["stream"]["schedules"])
            * len(config["runner"]["arms"]),
            "guard": config["no_heavy_guard"],
            "lane": "light CPU and disk mechanics only",
        },
        "remaining_full_run_gate": {
            "status": "not-run",
            "target_events_per_stream": int(full_gate["target_events_per_stream"]),
            "current_events_per_stream": int(config["stream"]["total_events"]),
            "progressive_rungs": [10_000, 100_000, int(full_gate["target_events_per_stream"])],
            "minimum_independent_seeds": int(full_gate["minimum_independent_seeds"]),
            "required_before_claim": [
                "run every schedule and control through the admitted resource governor lane",
                "independently replay metrics from immutable checkpoints",
                "demonstrate interruption recovery at the full rung",
                "separate useful retention from replay volume and extra compute",
            ],
            "current_blocker": "execution scale and independent replication, not local implementation",
            "hardware_boundary_earned": False,
        },
        "all_mechanics_ok": mechanics_ok,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = build_preflight(args.config.resolve(), args.work_root.resolve())
    _atomic_json(args.out.resolve(), receipt)
    print(
        f"wrote {args.out}: {receipt['status']}, "
        f"events={receipt['resource_envelope']['configured_total_arm_events']}, "
        f"payload={receipt['payload_sha256']}"
    )
    return 0 if receipt["all_mechanics_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
