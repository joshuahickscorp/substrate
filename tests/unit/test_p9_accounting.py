
from __future__ import annotations

import time
from pathlib import Path

import pytest

from mop.studies.p9_accounting import SCHEMA, WorkloadAccountant


def test_phase_accounting_totals(tmp_path: Path) -> None:
    accountant = WorkloadAccountant(workload="unit", watch_paths={"scratch": tmp_path})
    with accountant.phase("compute"):
        time.sleep(0.05)
    with accountant.phase("checkpoint"):
        (tmp_path / "blob.bin").write_bytes(b"x" * 4096)
    receipt = accountant.receipt()
    assert receipt["schema"] == SCHEMA
    names = [phase["name"] for phase in receipt["phases"]]
    assert names == ["compute", "checkpoint"]
    assert receipt["phases"][0]["wall_seconds"] >= 0.05
    assert receipt["phases"][1]["storage_delta_bytes"]["scratch"] == 4096
    totals = receipt["totals"]
    assert totals["span_seconds"] >= totals["accounted_phase_seconds"]
    assert totals["idle_seconds"] >= 0.0
    assert totals["storage_delta_bytes"]["scratch"] == 4096
    assert receipt["energy"]["measured"] is False


def test_storage_delta_can_be_negative(tmp_path: Path) -> None:
    victim = tmp_path / "old.bin"
    victim.write_bytes(b"y" * 2048)
    accountant = WorkloadAccountant(workload="unit", watch_paths={"scratch": tmp_path})
    with accountant.phase("cleanup"):
        victim.unlink()
    receipt = accountant.receipt()
    assert receipt["phases"][0]["storage_delta_bytes"]["scratch"] == -2048


def test_overlapping_phases_refused() -> None:
    accountant = WorkloadAccountant(workload="unit")
    with pytest.raises(RuntimeError), accountant.phase("outer"), accountant.phase("inner"):
        pass


def test_receipt_refused_while_phase_open() -> None:
    accountant = WorkloadAccountant(workload="unit")
    manager = accountant.phase("open")
    manager.__enter__()
    with pytest.raises(RuntimeError):
        accountant.receipt()
    manager.__exit__(None, None, None)
    assert accountant.receipt()["phases"][0]["name"] == "open"


def test_retry_attaches_to_named_phase() -> None:
    accountant = WorkloadAccountant(workload="unit")
    with accountant.phase("model"):
        pass
    accountant.note_retry("model")
    receipt = accountant.receipt()
    assert receipt["phases"][0]["retries"] == 1
    assert receipt["totals"]["retries"] == 1
    with pytest.raises(KeyError):
        accountant.note_retry("never-ran")


def test_write_round_trips(tmp_path: Path) -> None:
    import json

    accountant = WorkloadAccountant(workload="unit")
    with accountant.phase("noop"):
        pass
    out = accountant.write(tmp_path / "receipt.json")
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == SCHEMA
    assert loaded["workload"] == "unit"
