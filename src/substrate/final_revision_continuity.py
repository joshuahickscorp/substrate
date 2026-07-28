"""Genuine sequential continuity lane for the frozen final revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from substrate import final_revision_io as io
from substrate.final_revision_kernel import ArchitecturePrototype, EventSourcedKernel, developmental_fixture


def _load_or_construct(path: Path) -> tuple[EventSourcedKernel, bool]:
    if path.is_file():
        document = io.load_json(path)
        checkpoint = document.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise io.Refused("continuity checkpoint payload is absent")
        return EventSourcedKernel.restore(checkpoint), True
    prototype = ArchitecturePrototype("I_simplest_sufficient", "continuity-entity")
    developmental_fixture(prototype)
    return prototype.kernel, False


def _scheduled_events(kernel: EventSourcedKernel, segment: int) -> None:
    if segment == 0:
        kernel.append(
            "goal",
            {"goal_id": "old-project", "description": "finish the old project after future changes", "action": "create"},
            provenance="continuity://old-project",
        )
        kernel.append(
            "observation",
            {
                "modality": "filesystem",
                "content_digest": io.digest("project checkpoint created"),
                "features": {"event": "project checkpoint created", "uncertainty": 0.1},
            },
            provenance="continuity://filesystem",
        )
    elif segment == 1:
        kernel.append(
            "model_replace",
            {"previous": "model-b", "replacement": "model-c", "contract": {"roles": ["drafter", "fallback"]}},
            provenance="continuity://model-replacement",
        )
        kernel.append(
            "observation",
            {
                "modality": "video",
                "content_digest": io.digest("sensor interruption"),
                "features": {"event": "sensor interruption", "uncertainty": 1.0, "available": False},
            },
            provenance="continuity://sensor-interruption",
        )
        kernel.append(
            "body_tool",
            {"target": "body", "key": "embodiment", "value": "changed-body-v2"},
            provenance="continuity://body-tool-change",
        )
        kernel.append(
            "body_tool",
            {"target": "tools", "key": "changed-tool", "value": {"available": True}},
            provenance="continuity://body-tool-change",
        )
    else:
        kernel.append(
            "belief",
            {"key": "old-project-ready", "value": False, "confidence": 0.8, "defeated": False},
            provenance="continuity://conflicting-correction",
        )
        kernel.append(
            "correction",
            {"belief_key": "old-project-ready", "reason": "conflicting correction received"},
            provenance="continuity://conflicting-correction",
        )
        kernel.append(
            "goal",
            {
                "goal_id": "history-dependent-new-task",
                "description": "use the old project history after model, body, and sensor changes",
                "action": "create",
            },
            provenance="continuity://history-dependent-task",
        )


def run_segment(*, checkpoint_path: Path, receipt_path: Path, segment: int, duration_seconds: float) -> dict[str, Any]:
    started_wall = time.time()
    started = time.monotonic()
    kernel, restored_from_prior_process = _load_or_construct(checkpoint_path)
    state_integrity_before = kernel.state_integrity_digest()
    _scheduled_events(kernel, segment)
    iterations = 0
    accumulator = state_integrity_before
    consolidation_events = 0
    next_consolidation = started + 60.0
    while time.monotonic() - started < duration_seconds:
        if io.STOP.is_file():
            raise io.Refused("operator stop switch set during continuity lane")
        accumulator = hashlib.sha256(f"{accumulator}:{segment}:{iterations}".encode()).hexdigest()
        iterations += 1
        now = time.monotonic()
        if now >= next_consolidation:
            kernel.append(
                "memory",
                {
                    "memory_type": "working",
                    "key": f"continuity-segment-{segment}",
                    "value": {"iterations": iterations, "work_digest": accumulator},
                },
                provenance=f"continuity://background-consolidation/{segment}",
            )
            consolidation_events += 1
            next_consolidation = now + 60.0
    checkpoint = kernel.checkpoint()
    checkpoint_document = io.authority(
        "substrate-final-revision-continuity-checkpoint/v1",
        {
            "segment": segment,
            "checkpoint": checkpoint,
            "entity_identity": kernel.identity,
            "continuity_epoch": kernel.state["identity"]["continuity_epoch"],
            "state_integrity_before": state_integrity_before,
            "state_integrity_after": kernel.state_integrity_digest(),
            "restored_from_prior_process": restored_from_prior_process,
        },
        status="intermediate",
    )
    io.write_json(checkpoint_path, checkpoint_document)
    receipt = io.authority(
        "substrate-final-revision-continuity-segment/v1",
        {
            "segment": segment,
            "process_boundary": True,
            "process_started_unix": started_wall,
            "duration_seconds": time.monotonic() - started,
            "iterations": iterations,
            "work_digest": accumulator,
            "background_consolidation_events": consolidation_events,
            "restored_from_prior_process": restored_from_prior_process,
            "entity_identity": kernel.identity,
            "continuity_epoch": kernel.state["identity"]["continuity_epoch"],
            "state_integrity_before": state_integrity_before,
            "state_integrity_after": kernel.state_integrity_digest(),
            "unfinished_goals": kernel.query("goals"),
            "model_fabric": kernel.query("model_fabric"),
            "body_and_tools": kernel.query("body_and_tools"),
        },
    )
    io.write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    options = parser.parse_args(argv)
    result = run_segment(
        checkpoint_path=options.checkpoint,
        receipt_path=options.receipt,
        segment=options.segment,
        duration_seconds=options.duration_seconds,
    )
    print(json.dumps({"segment": result["segment"], "sha256": result["sha256"], "activation": False}, sort_keys=True))


if __name__ == "__main__":
    main()
