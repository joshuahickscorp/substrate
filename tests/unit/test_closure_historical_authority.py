"""Cross-checkout historical-authority admission tests for the closure gate.

The gap these close: a General Run is sealed by the orchestrator bytes it ran from (implementation SHA A);
the closure may run from a different checkout (SHA B). Admission must bind to the run's recorded historical
authority (SHA A) and admit a clean terminal only when the referenced SHA-A bytes are available and exact,
and refuse when they are missing or altered. These build a valid ``complete`` status sealed under a
synthetic SHA A that differs from the current checkout's orchestrator SHA B.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json

import pytest

from mop.closure import admission as adm
from mop.studio import general_run_orchestrator as gro

_REL_IMPL = "src/mop/studio/general_run_orchestrator.py"


def _write_historical_impl(impl_root, body: bytes) -> str:
    """Write synthetic historical orchestrator bytes at the recorded relative path; return their SHA."""

    path = impl_root / _REL_IMPL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return gro.sha256_file(path)


def _seal(core: dict) -> dict:
    return {**core, "status_sha256": gro.canonical_sha256(core)}


def _complete_status(sha_a: str) -> dict:
    """A valid ``complete`` general-run status whose authority is SHA A (not the current checkout)."""

    supervisor = {
        "pid": 4242,
        "create_time": 1_784_000_000.0,
        "implementation_path": _REL_IMPL,
        "implementation_sha256": sha_a,
    }
    core = {
        "schema": gro.STATUS_SCHEMA,
        "program_id": gro.PROGRAM_ID,
        "created_at": "2026-07-18T10:00:00+00:00",
        "updated_at": "2026-07-18T12:00:00+00:00",
        "finished_at": "2026-07-18T12:00:00+00:00",
        "execution_enabled": True,
        "state": "complete",
        "stage": gro.STAGES[-1],
        "supervisor": supervisor,
        "parent_implementation": {"path": _REL_IMPL, "sha256": sha_a},
        "capsules": {"run_full_generations_wave": {"status": "complete"}},
        "counts": {
            "compute_complete": 4,
            "compute_total": 4,
            "legacy_complete": 3,
            "legacy_total": 3,
            "stage_index": len(gro.STAGES) - 1,
            "stage_total": len(gro.STAGES),
        },
        "reprofile": None,
        "last_admission": None,
        "problems": [],
        "signals_allowed": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _seal(core)


def _write_run(gr_root, status: dict) -> None:
    gr_root.mkdir(parents=True, exist_ok=True)
    (gr_root / "current_status.json").write_text(json.dumps(status), encoding="utf-8")
    (gr_root / "program_state.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")


def test_admits_clean_terminal_via_historical_authority_from_a_different_checkout(tmp_path, monkeypatch):
    """SHA A run + SHA B checkout + clean complete: admit only because the SHA-A bytes are present and exact."""

    impl_root = tmp_path / "campaign_tree"
    gr_root = tmp_path / "campaign_tree" / "runs" / "generation1" / "general-run"
    sha_a = _write_historical_impl(impl_root, b"# historical orchestrator SHA A bytes, not the checkout\n")
    # the current checkout's orchestrator is a different SHA (SHA B); confirm the two genuinely differ
    assert sha_a != gro.sha256_file(gro.IMPLEMENTATION_PATH)
    _write_run(gr_root, _complete_status(sha_a))
    # isolate the admission logic from the real live campaign's processes
    monkeypatch.setattr(adm, "_count_live_general_run_writers", lambda exclude_pids=None: 0)

    decision = adm.evaluate_admission(
        root=gr_root, implementation_root=impl_root, stability_wait_seconds=0.0, now_iso="t"
    )

    assert decision.historical_bytes_available is True
    assert decision.historical_authority_sha256 == sha_a
    assert decision.general_run_state == "complete"
    assert decision.admitted is True
    assert decision.refusals == []


def test_refuses_when_historical_bytes_are_missing(tmp_path, monkeypatch):
    """Clean complete status, but the referenced SHA-A implementation bytes are absent: refuse."""

    impl_root = tmp_path / "campaign_tree"
    gr_root = impl_root / "runs" / "generation1" / "general-run"
    sha_a = _write_historical_impl(impl_root, b"# SHA A bytes\n")
    _write_run(gr_root, _complete_status(sha_a))
    monkeypatch.setattr(adm, "_count_live_general_run_writers", lambda exclude_pids=None: 0)
    # remove the historical implementation bytes so they no longer resolve
    (impl_root / _REL_IMPL).unlink()

    decision = adm.evaluate_admission(
        root=gr_root, implementation_root=impl_root, stability_wait_seconds=0.0, now_iso="t"
    )

    assert decision.admitted is False
    assert decision.historical_bytes_available is False
    assert any("not available" in r for r in decision.refusals)


def test_refuses_when_historical_bytes_are_altered(tmp_path, monkeypatch):
    """Clean complete status, but the referenced implementation bytes were altered after sealing: refuse."""

    impl_root = tmp_path / "campaign_tree"
    gr_root = impl_root / "runs" / "generation1" / "general-run"
    sha_a = _write_historical_impl(impl_root, b"# SHA A bytes\n")
    _write_run(gr_root, _complete_status(sha_a))
    monkeypatch.setattr(adm, "_count_live_general_run_writers", lambda exclude_pids=None: 0)
    # alter the bytes so the on-disk SHA no longer matches the sealed authority
    (impl_root / _REL_IMPL).write_bytes(b"# tampered bytes with a different hash\n")

    decision = adm.evaluate_admission(
        root=gr_root, implementation_root=impl_root, stability_wait_seconds=0.0, now_iso="t"
    )

    assert decision.admitted is False
    assert decision.historical_bytes_available is False
    assert any("altered" in r for r in decision.refusals)


def test_refuses_when_supervisor_authority_disagrees(tmp_path, monkeypatch):
    """A reconciled or mixed shell whose supervisor authority differs from the parent is refused."""

    impl_root = tmp_path / "campaign_tree"
    gr_root = impl_root / "runs" / "generation1" / "general-run"
    sha_a = _write_historical_impl(impl_root, b"# SHA A bytes\n")
    status = _complete_status(sha_a)
    status["supervisor"]["implementation_sha256"] = "f" * 64  # supervisor disagrees with parent authority
    status = _seal({k: v for k, v in status.items() if k != "status_sha256"})
    _write_run(gr_root, status)
    monkeypatch.setattr(adm, "_count_live_general_run_writers", lambda exclude_pids=None: 0)

    decision = adm.evaluate_admission(
        root=gr_root, implementation_root=impl_root, stability_wait_seconds=0.0, now_iso="t"
    )

    assert decision.admitted is False
    assert any("supervisor authority" in r for r in decision.refusals)


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path("/Users/scammermike/Downloads/mop/runs/generation1/general-run")
    ).exists(),
    reason="live General Run tree not present",
)
def test_live_run_is_readable_and_deferred_not_authority_blocked():
    """Against the live run the fix reads the real state and defers on 'not terminal', not authority drift."""

    from pathlib import Path

    live = Path("/Users/scammermike/Downloads/mop")
    decision = adm.evaluate_admission(
        root=live / "runs/generation1/general-run",
        implementation_root=live,
        stability_wait_seconds=0.0,
        now_iso="t",
    )
    assert decision.admitted is False
    assert decision.historical_bytes_available is True  # the live SHA-A bytes resolve and hash-match
    assert decision.general_run_state is not None  # the state is now READABLE (was None before the fix)
    assert any("not terminal" in r for r in decision.refusals)
    assert not any("authority drifted" in r for r in decision.refusals)
