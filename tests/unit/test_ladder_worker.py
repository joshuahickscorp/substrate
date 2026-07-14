"""Unit tests for the ladder worker subprocess entrypoint.

These tests assert that main writes a sealed receipt whose receipt_sha256 verifies and exits zero on
a good run, that the reps determinism guard rejects a drifting demonstration, and that malformed
requests fail closed with no receipt. No capability is claimed anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mop.ladder import ladder_worker
from mop.ladder.stage3_registry import run_demonstration
from mop.substrate.events import canonical_sha256


def _run(out: Path, epoch: str = "event_formation", seed: int = 0, reps: int = 3) -> int:
    return ladder_worker.main(
        ["--epoch", epoch, "--seed", str(seed), "--reps", str(reps), "--out", str(out)]
    )


def test_main_writes_a_sealed_receipt_and_exits_zero(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    code = _run(out, epoch="event_formation", seed=0, reps=3)
    assert code == 0
    assert out.is_file()

    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["schema"] == "mop-ladder-worker-receipt/v1"
    assert receipt["epoch"] == "event_formation"
    assert receipt["seed"] == 0
    assert receipt["reps"] == 3
    assert receipt["mechanism_id"] == "event_formation"
    assert receipt["kind"] == "mechanics-demonstration"
    assert receipt["is_confirmation"] is False
    assert receipt["verdict"] in ("null", "pending", "mechanics-ok")

    # The seal is over every field except receipt_sha256, exactly as the orchestrator recomputes it.
    declared = receipt["receipt_sha256"]
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert declared == canonical_sha256(core)

    # The sealed result digest matches an independent run of the demonstration.
    assert receipt["result_digest"] == run_demonstration("event_formation", 0).digest()

    # Wall clock time is never sealed into the payload.
    assert "elapsed" not in receipt
    assert "wall" not in receipt


@pytest.mark.parametrize("epoch", ["trace_stability", "construction_search", "integrated_escs"])
def test_receipt_seal_verifies_across_epochs(tmp_path: Path, epoch: str) -> None:
    out = tmp_path / f"{epoch}.json"
    assert _run(out, epoch=epoch, seed=5, reps=2) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(core)


def test_reps_determinism_guard_rejects_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _Drifting:
        def digest(self) -> str:
            calls["n"] += 1
            return f"{calls['n']:064d}"

    def _fake(epoch: str, seed: int) -> Any:
        return _Drifting()

    monkeypatch.setattr(ladder_worker, "run_demonstration", _fake)
    out = tmp_path / "drift.json"
    code = ladder_worker.main(
        ["--epoch", "event_formation", "--seed", "0", "--reps", "3", "--out", str(out)]
    )
    assert code == 1
    assert not out.exists()


def test_build_receipt_rejects_nonpositive_reps() -> None:
    with pytest.raises(ValueError):
        ladder_worker.build_receipt("event_formation", 0, 0)


def test_unknown_epoch_exits_nonzero_without_receipt(tmp_path: Path) -> None:
    out = tmp_path / "missing.json"
    code = _run(out, epoch="not_an_epoch", seed=0, reps=1)
    assert code == 1
    assert not out.exists()


def test_negative_seed_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "neg.json"
    assert _run(out, epoch="trace_stability", seed=-1, reps=1) == 1
    assert not out.exists()
