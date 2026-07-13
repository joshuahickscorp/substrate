"""Tests for the chained ladder campaign orchestrator.

These cover the pure orchestration logic (config validation, work planning, receipt collection and seal
verification, and own-worker shedding) plus one minimal end-to-end run that actually spawns worker
subprocesses under the throttler and checks the sealed ladder report. No capability is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from mop.ladder.ladder_campaign import (
    CampaignConfig,
    CampaignRefusal,
    LadderCampaign,
    WorkerHandle,
    WorkItem,
    _sealed,
    census_complete,
)
from mop.substrate.events import canonical_sha256


def tiny_config(tmp_path: Path, **overrides: Any) -> CampaignConfig:
    base: dict[str, Any] = dict(
        program_root=tmp_path / "campaign",
        seeds=(0, 1),
        reps=1,
        poll_interval_s=0.05,
        epochs=("event_formation",),
    )
    base.update(overrides)
    return CampaignConfig(**base)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def poll(self) -> int | None:
        return None


# ---------------------------------------------------------------------------
# Config validation fails closed.
# ---------------------------------------------------------------------------


def test_config_rejects_duplicate_seeds(tmp_path: Path) -> None:
    with pytest.raises(CampaignRefusal):
        tiny_config(tmp_path, seeds=(0, 0))


def test_config_rejects_zero_reps(tmp_path: Path) -> None:
    with pytest.raises(CampaignRefusal):
        tiny_config(tmp_path, reps=0)


def test_config_rejects_empty_seeds(tmp_path: Path) -> None:
    with pytest.raises(CampaignRefusal):
        tiny_config(tmp_path, seeds=())


def test_config_rejects_nonpositive_poll(tmp_path: Path) -> None:
    with pytest.raises(CampaignRefusal):
        tiny_config(tmp_path, poll_interval_s=0.0)


# ---------------------------------------------------------------------------
# Planning and sealing.
# ---------------------------------------------------------------------------


def test_plan_stage3_is_epochs_by_seeds(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, seeds=(0, 1, 2), epochs=("event_formation", "integrated_escs"))
    items = LadderCampaign(config).plan_stage3()
    assert len(items) == 6
    assert {item.epoch for item in items} == {"event_formation", "integrated_escs"}


def test_sealed_round_trips() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    sealed = _sealed(payload, "seal")
    core = {k: v for k, v in sealed.items() if k != "seal"}
    assert sealed["seal"] == canonical_sha256(core)


# ---------------------------------------------------------------------------
# Receipt collection verifies the worker seal.
# ---------------------------------------------------------------------------


def _write_receipt(path: Path, *, tamper: bool = False) -> None:
    core = {
        "schema": "mop-ladder-worker-receipt/v1",
        "epoch": "event_formation",
        "seed": 0,
        "reps": 1,
        "verdict_status": "x0-strong-null",
        "null_holds": True,
        "controls_ok": True,
        "result_digest": "0" * 64,
        "claim_scope": "deterministic programmatic mechanics only; no capability or natural-data claim",
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    if tamper:
        receipt["null_holds"] = False  # break the seal without recomputing it
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_collect_accepts_valid_receipt(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    campaign = LadderCampaign(config)
    item = WorkItem("event_formation", 0)
    receipt_path = campaign._receipt_path(item)
    _write_receipt(receipt_path)
    handle = WorkerHandle(
        item=item, process=cast(Any, _FakeProcess(1)), order=0, receipt_path=receipt_path
    )
    record = campaign._collect(handle, exit_code=0)
    assert record["ok"] is True
    assert record["null_holds"] is True


def test_collect_rejects_tampered_receipt(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    campaign = LadderCampaign(config)
    item = WorkItem("event_formation", 0)
    receipt_path = campaign._receipt_path(item)
    _write_receipt(receipt_path, tamper=True)
    handle = WorkerHandle(
        item=item, process=cast(Any, _FakeProcess(1)), order=0, receipt_path=receipt_path
    )
    record = campaign._collect(handle, exit_code=0)
    assert record["ok"] is False


def test_collect_flags_worker_failure(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    campaign = LadderCampaign(config)
    item = WorkItem("event_formation", 0)
    handle = WorkerHandle(
        item=item, process=cast(Any, _FakeProcess(1)), order=0, receipt_path=campaign._receipt_path(item)
    )
    record = campaign._collect(handle, exit_code=1)
    assert record["ok"] is False


# ---------------------------------------------------------------------------
# Shedding removes our own newest workers and requeues their items.
# ---------------------------------------------------------------------------


def test_shed_terminates_newest_and_requeues(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    campaign = LadderCampaign(config)
    procs = [_FakeProcess(pid) for pid in (10, 11, 12)]
    running = [
        WorkerHandle(
            item=WorkItem("event_formation", i),
            process=cast(Any, procs[i]),
            order=i,
            receipt_path=tmp_path / f"r{i}.json",
        )
        for i in range(3)
    ]
    requeued = campaign._shed(running, 2)
    assert len(requeued) == 2
    assert len(running) == 1  # oldest survives
    assert running[0].order == 0
    assert procs[2].terminated and procs[1].terminated
    assert not procs[0].terminated


# ---------------------------------------------------------------------------
# One minimal real run under the throttler.
# ---------------------------------------------------------------------------


def _write_census(path: Path, capsules: list[dict[str, Any]], status: str = "running") -> None:
    path.write_text(json.dumps({"status": status, "capsules": capsules}), encoding="utf-8")


def test_census_complete_true_when_absent(tmp_path: Path) -> None:
    assert census_complete(tmp_path / "nope.json") is True


def test_census_complete_true_on_status_complete(tmp_path: Path) -> None:
    state = tmp_path / "census.json"
    _write_census(state, [{"status": "running"}], status="complete")
    assert census_complete(state) is True


def test_census_complete_true_when_all_capsules_done(tmp_path: Path) -> None:
    state = tmp_path / "census.json"
    _write_census(state, [{"status": "complete"}, {"status": "complete"}])
    assert census_complete(state) is True


def test_census_complete_false_while_running(tmp_path: Path) -> None:
    state = tmp_path / "census.json"
    _write_census(state, [{"status": "complete"}, {"status": "running"}])
    assert census_complete(state) is False


def test_census_complete_handles_dict_keyed_capsules_all_done(tmp_path: Path) -> None:
    # The real census stores capsules as a dict keyed by capsule id, not a list.
    state = tmp_path / "census.json"
    state.write_text(
        json.dumps({"status": "running", "capsules": {"a": {"status": "complete"}, "b": {"status": "complete"}}}),
        encoding="utf-8",
    )
    assert census_complete(state) is True


def test_census_complete_false_when_dict_capsules_partial(tmp_path: Path) -> None:
    state = tmp_path / "census.json"
    state.write_text(
        json.dumps({"status": "running", "capsules": {"a": {"status": "complete"}, "b": {"status": "running"}}}),
        encoding="utf-8",
    )
    assert census_complete(state) is False


def test_minimal_end_to_end_run_writes_sealed_report(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, seeds=(0,), reps=1, epochs=("event_formation",))
    report = LadderCampaign(config).run()
    assert report["schema"] == "mop-ladder-campaign-report/v1"
    core = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == canonical_sha256(core)
    assert report["stage3"]["total_work"] == 1
    assert report["stage4_5"]["stage4"]["status"] == "not entered"
    assert config.report_path.is_file()
