
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..substrate.continual_stream import (
    ContinualEvent,
    ContinualStreamSpec,
    event_at,
    iter_stream,
    read_manifest,
    verify_stream,
)
from ..substrate.events import canonical_bytes, canonical_sha256
from ..substrate.lifecycle import LifecycleJournal, MemoryRef

CHECKPOINT_SCHEMA = "mop-continual-smoke-checkpoint/v1"
RESULT_SCHEMA = "mop-continual-smoke-result/v1"
CLAIM_SCOPE = "disk-backed programmatic continual-stream mechanics only; no capability claim"
ARMS = ("replay", "no-replay", "fresh-init")


@dataclass(frozen=True, slots=True)
class ContinualSmokeProfile:
    checkpoint_every: int
    replay_capacity: int
    future_window_events: int
    threshold_window_events: int
    future_accuracy_threshold: float
    matched_updates_per_event: int = 2

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1 or self.replay_capacity < 1:
            raise ValueError("checkpoint interval and replay capacity must be positive")
        if self.future_window_events < 1 or self.threshold_window_events < 2:
            raise ValueError("future metric windows are invalid")
        if not 0.0 < self.future_accuracy_threshold <= 1.0:
            raise ValueError("future accuracy threshold must be in (0, 1]")
        if self.matched_updates_per_event != 2:
            raise ValueError("the smoke control contract requires exactly two updates per event")

    def payload(self) -> dict[str, Any]:
        return {
            "checkpoint_every": self.checkpoint_every,
            "replay_capacity": self.replay_capacity,
            "future_window_events": self.future_window_events,
            "threshold_window_events": self.threshold_window_events,
            "future_accuracy_threshold": self.future_accuracy_threshold,
            "matched_updates_per_event": self.matched_updates_per_event,
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_state(spec: ContinualStreamSpec) -> dict[str, Any]:
    return {
        "next_sequence": 0,
        "last_event_sha256": "00" * 32,
        "counts": [[0 for _ in range(spec.n_classes)] for _ in range(spec.n_classes)],
        "replay": [],
        "current_stage": 0,
        "correct": 0,
        "total": 0,
        "domain_correct": [0 for _ in range(spec.n_domains)],
        "domain_total": [0 for _ in range(spec.n_domains)],
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


def _predict(counts: list[list[int]], cue: int) -> int:
    row = counts[cue]
    return max(range(len(row)), key=lambda label: (row[label], -label))


def _update(counts: list[list[int]], cue: int, label: int) -> None:
    counts[cue][label] += 1


def _domain_accuracy(counts: list[list[int]], domain: int, n_classes: int) -> float:
    correct = sum(_predict(counts, cue) == (cue + domain) % n_classes for cue in range(n_classes))
    return correct / n_classes


def _identity(stream_manifest: dict[str, Any], arm: str, profile: ContinualSmokeProfile) -> dict[str, Any]:
    return {
        "stream_identity_sha256": stream_manifest["identity_sha256"],
        "stream_sha256": stream_manifest["stream_sha256"],
        "arm": arm,
        "profile": profile.payload(),
        "claim_scope": CLAIM_SCOPE,
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any], identity: dict[str, Any], stream_root: Path, spec: ContinualStreamSpec
) -> dict[str, Any]:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("continual smoke checkpoint schema drift")
    if checkpoint.get("identity") != identity or checkpoint.get("identity_sha256") != canonical_sha256(
        identity
    ):
        raise ValueError("continual smoke checkpoint identity drift")
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        raise ValueError("continual smoke checkpoint state missing")
    next_sequence = state.get("next_sequence")
    if not isinstance(next_sequence, int) or not 0 <= next_sequence <= spec.total_events:
        raise ValueError("continual smoke checkpoint cursor invalid")
    if next_sequence:
        prior = event_at(stream_root, next_sequence - 1)
        if state.get("last_event_sha256") != prior.content_sha256:
            raise ValueError("continual smoke checkpoint prefix identity drift")
    elif state.get("last_event_sha256") != "00" * 32:
        raise ValueError("empty continual smoke checkpoint has a nonempty prefix")
    if checkpoint.get("state_sha256") != canonical_sha256(state):
        raise ValueError("continual smoke checkpoint state digest drift")
    return state


def _checkpoint_payload(
    identity: dict[str, Any], state: dict[str, Any], *, complete: bool, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "state": state,
        "state_sha256": canonical_sha256(state),
        "complete": complete,
        "result": result,
    }


def _process_event(
    state: dict[str, Any],
    event: ContinualEvent,
    arm: str,
    spec: ContinualStreamSpec,
    profile: ContinualSmokeProfile,
) -> None:
    counts = state["counts"]
    replay: list[dict[str, Any]] = state["replay"]
    if event.stage > int(state["current_stage"]):
        state["anchor_snapshots"].append(
            {
                "sequence": event.sequence,
                "stage": event.stage,
                "domain_zero_accuracy": _domain_accuracy(counts, 0, spec.n_classes),
            }
        )
        if arm == "fresh-init":
            state["counts"] = [[0 for _ in range(spec.n_classes)] for _ in range(spec.n_classes)]
            counts = state["counts"]
            state["resets"] += 1
        state["current_stage"] = event.stage

    if event.deletion_requested:
        before = len(replay)
        replay[:] = [row for row in replay if int(row["domain"]) != 0]
        state["deletion_removed"] += before - len(replay)
        state["deletion_seen"] = True

    prediction = _predict(counts, event.cue)
    correct = prediction == event.label
    state["correct"] += int(correct)
    state["total"] += 1
    state["domain_correct"][event.active_domain] += int(correct)
    state["domain_total"][event.active_domain] += 1

    if event.active_domain == spec.n_domains - 1:
        state["future_seen"] += 1
        if len(state["future_outcomes"]) < profile.future_window_events:
            state["future_outcomes"].append(int(correct))
        rolling: list[int] = state["future_rolling"]
        rolling.append(int(correct))
        if len(rolling) > profile.threshold_window_events:
            rolling.pop(0)
        if (
            state["future_events_to_threshold"] is None
            and len(rolling) == profile.threshold_window_events
            and sum(rolling) / len(rolling) >= profile.future_accuracy_threshold
        ):
            state["future_events_to_threshold"] = state["future_seen"]

    _update(counts, event.cue, event.label)
    state["updates"] += 1
    if arm == "replay" and replay:
        replay_row = replay[event.sequence % len(replay)]
        before_replay = _predict(counts, event.cue) == event.label
        stale = int(replay_row["domain"]) != event.active_domain
        _update(counts, int(replay_row["cue"]), int(replay_row["label"]))
        after_replay = _predict(counts, event.cue) == event.label
        state["updates"] += 1
        state["replay_samples"] += 1
        if stale:
            state["stale_opportunities"] += 1
            state["stale_harm_count"] += int(before_replay and not after_replay)
    else:
        _update(counts, event.cue, event.label)
        state["updates"] += 1

    if arm == "replay":
        replay.append(
            {
                "event_ref": str(event.event_ref),
                "event_sha256": event.content_sha256,
                "cue": event.cue,
                "label": event.label,
                "domain": event.active_domain,
            }
        )
        if len(replay) > profile.replay_capacity:
            replay.pop(0)
    state["max_replay_records"] = max(int(state["max_replay_records"]), len(replay))
    state["next_sequence"] = event.sequence + 1
    state["last_event_sha256"] = event.content_sha256


def _build_lifecycle(stream_root: Path, spec: ContinualStreamSpec) -> LifecycleJournal:
    journal = LifecycleJournal(MemoryRef(f"memory:{spec.identity_sha256[:16]}/continual-anchor"))
    current_stage = -1
    deleted = False
    for event in iter_stream(stream_root):
        if current_stage < 0:
            journal.record(
                event.event_ref,
                {"stage": event.stage, "mapping": "cue-plus-domain", "source_event": event.content_sha256},
                reason="record initial stream mapping",
            )
            current_stage = event.stage
        elif event.stage > current_stage and not deleted:
            journal.revise(
                event.event_ref,
                {"stage": event.stage, "mapping": "cue-plus-domain", "source_event": event.content_sha256},
                reason="revise mapping at transition",
            )
            current_stage = event.stage
        if event.deletion_requested and not deleted:
            journal.delete(event.event_ref, reason="delete superseded continual anchor")
            deleted = True
    return journal


def _final_result(
    *,
    arm: str,
    state: dict[str, Any],
    stream_root: Path,
    spec: ContinualStreamSpec,
    profile: ContinualSmokeProfile,
    identity: dict[str, Any],
    resumed: bool,
) -> dict[str, Any]:
    lifecycle = _build_lifecycle(stream_root, spec)
    event_refs = {str(event.event_ref) for event in iter_stream(stream_root)}
    lifecycle_errors = lifecycle.verify(event_refs=event_refs)
    lifecycle_state = lifecycle.state_at()
    replay = state["replay"]
    stale_opportunities = int(state["stale_opportunities"])
    future_outcomes = state["future_outcomes"]
    stream_audit = verify_stream(stream_root, expected_spec=spec, require_complete=True)
    result = {
        "schema": RESULT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "arm": arm,
        "stream_identity_sha256": spec.identity_sha256,
        "stream_sha256": identity["stream_sha256"],
        "complete": int(state["next_sequence"]) == spec.total_events,
        "resumed_from_atomic_checkpoint": resumed,
        "metrics": {
            "retention": {
                "domain_zero_final_accuracy": _domain_accuracy(state["counts"], 0, spec.n_classes),
                "transition_snapshots": state["anchor_snapshots"],
            },
            "acquisition": {
                "stream_accuracy": state["correct"] / max(1, state["total"]),
                "per_domain_accuracy": [
                    state["domain_correct"][index] / max(1, state["domain_total"][index])
                    for index in range(spec.n_domains)
                ],
            },
            "future_learnability": {
                "first_window_accuracy": sum(future_outcomes) / max(1, len(future_outcomes)),
                "window_events": len(future_outcomes),
                "events_to_threshold": state["future_events_to_threshold"],
                "threshold": profile.future_accuracy_threshold,
                "threshold_window_events": profile.threshold_window_events,
            },
            "stale_memory": {
                "opportunities": stale_opportunities,
                "harm_count": state["stale_harm_count"],
                "harm_rate": state["stale_harm_count"] / max(1, stale_opportunities),
            },
            "deletion": {
                "requested": state["deletion_seen"],
                "replay_records_removed": state["deletion_removed"],
                "remaining_deleted_domain_records": sum(int(row["domain"] == 0) for row in replay),
                "lifecycle_deleted": lifecycle_state.deleted,
                "lifecycle_available_after_delete": lifecycle_state.available_at(spec.total_events),
                "complete": bool(
                    state["deletion_seen"]
                    and not any(int(row["domain"]) == 0 for row in replay)
                    and lifecycle_state.deleted
                    and not lifecycle_state.available_at(spec.total_events)
                ),
            },
            "resources": {
                "events_processed": state["total"],
                "updates": state["updates"],
                "updates_per_event": state["updates"] / max(1, state["total"]),
                "replay_samples": state["replay_samples"],
                "max_replay_records": state["max_replay_records"],
                "replay_capacity": profile.replay_capacity,
                "stream_disk_bytes": stream_audit["disk_bytes"],
                "checkpoint_state_bytes": len(canonical_bytes(state)),
                "model_weights_loaded": False,
                "accelerator_required": False,
            },
        },
        "controls": {
            "replay_enabled": arm == "replay",
            "fresh_init_on_transition": arm == "fresh-init",
            "matched_updates_per_event": profile.matched_updates_per_event,
            "actual_updates_per_event": state["updates"] / max(1, state["total"]),
            "fixed_topology": True,
            "reset_count": state["resets"],
        },
        "lifecycle": lifecycle.payload(),
        "lifecycle_sha256": lifecycle.sha256,
        "lifecycle_errors": lifecycle_errors,
        "state_sha256": canonical_sha256(state),
    }
    result["all_mechanics_ok"] = bool(
        result["complete"]
        and not lifecycle_errors
        and result["metrics"]["deletion"]["complete"]
        and result["metrics"]["resources"]["updates"] == spec.total_events * profile.matched_updates_per_event
        and result["controls"]["actual_updates_per_event"] == profile.matched_updates_per_event
    )
    return result


def run_smoke_arm(
    *,
    stream_root: Path | str,
    spec: ContinualStreamSpec,
    arm: str,
    profile: ContinualSmokeProfile,
    checkpoint_path: Path | str,
    event_budget: int | None = None,
) -> dict[str, Any]:

    if arm not in ARMS:
        raise ValueError(f"unknown continual smoke arm {arm!r}")
    root = Path(stream_root)
    audit = verify_stream(root, expected_spec=spec, require_complete=True)
    if not audit["verified"]:
        raise ValueError("continual smoke stream verification failed: " + "; ".join(audit["errors"]))
    manifest = read_manifest(root)
    identity = _identity(manifest, arm, profile)
    checkpoint_file = Path(checkpoint_path)
    resumed = False
    if checkpoint_file.is_file():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        state = _validate_checkpoint(checkpoint, identity, root, spec)
        if checkpoint.get("complete") is True:
            result = checkpoint.get("result")
            if not isinstance(result, dict):
                raise ValueError("complete continual smoke checkpoint has no result")
            return {
                **result,
                "resumed_from_atomic_checkpoint": True,
                "checkpoint_sha256": _sha256_file(checkpoint_file),
            }
        resumed = int(state["next_sequence"]) > 0
    else:
        state = _new_state(spec)
        _atomic_json(checkpoint_file, _checkpoint_payload(identity, state, complete=False))

    start = int(state["next_sequence"])
    stop = spec.total_events
    if event_budget is not None:
        if event_budget < 1:
            raise ValueError("event budget must be positive")
        stop = min(stop, start + event_budget)
    since_checkpoint = 0
    for event in iter_stream(root, start_sequence=start, stop_sequence=stop):
        _process_event(state, event, arm, spec, profile)
        since_checkpoint += 1
        if since_checkpoint >= profile.checkpoint_every:
            _atomic_json(checkpoint_file, _checkpoint_payload(identity, state, complete=False))
            since_checkpoint = 0
    _atomic_json(checkpoint_file, _checkpoint_payload(identity, state, complete=False))

    if int(state["next_sequence"]) < spec.total_events:
        return {
            "schema": RESULT_SCHEMA,
            "claim_scope": CLAIM_SCOPE,
            "arm": arm,
            "complete": False,
            "resumed_from_atomic_checkpoint": resumed,
            "next_sequence": state["next_sequence"],
            "stream_sha256": manifest["stream_sha256"],
            "checkpoint_sha256": _sha256_file(checkpoint_file),
        }

    result = _final_result(
        arm=arm,
        state=state,
        stream_root=root,
        spec=spec,
        profile=profile,
        identity=identity,
        resumed=resumed,
    )
    _atomic_json(checkpoint_file, _checkpoint_payload(identity, state, complete=True, result=result))
    result["checkpoint_sha256"] = _sha256_file(checkpoint_file)
    return result
