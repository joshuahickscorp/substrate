from __future__ import annotations

import json
from pathlib import Path

from mop.studio import telegram_rung_notifier as notifier


def _write(path: Path, value: dict) -> None:
    if path.name == "current_status.json":
        core = dict(value)
        core.setdefault("schema", "mop-generation1-test-status/v1")
        program_id = str(core.get("program_id") or "generation1-test")
        core["program_id"] = (
            program_id if program_id.startswith("generation1-") else f"generation1-{program_id}"
        )
        value = {**core, "status_sha256": notifier.canonical_sha256(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collects_completed_capsule_terminal_and_standalone_proof(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proofs = tmp_path / "proof"
    _write(
        runs / "campaign" / "current_status.json",
        {
            "program_id": "p1",
            "state": "complete",
            "problems": [],
            "adaptive_execution": {
                "average_rung_seconds": 120.0,
                "eta_seconds": 0.0,
                "workers": 1,
            },
            "capsules": {
                "rung_a": {
                    "returncode": 0,
                    "finished_at": "2026-07-15T00:00:00+00:00",
                    "attempts": 1,
                    "artifacts": [{"path": "proof/A.json", "sha256": "a" * 64, "all_ok": True}],
                }
            },
        },
    )
    _write(
        proofs / "GENERATION1_STANDALONE.json",
        {"schema": "proof/v1", "complete": True, "decision": {"verdict": "candidate"}},
    )
    events = notifier.collect_events(runs_root=runs, proof_root=proofs)
    assert {event["kind"] for event in events} == {"rung", "terminal", "proof"}
    assert next(event for event in events if event["kind"] == "rung")["progress"] == {
        "complete": 1,
        "total": 1,
    }
    assert next(event for event in events if event["kind"] == "rung")["eta"] == {
        "block_seconds": 0.0,
        "session_seconds": 0.0,
    }


def test_failed_program_becomes_important_failure_event(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "campaign" / "current_status.json",
        {
            "program_id": "p2",
            "state": "failure_hold",
            "problems": ["bad receipt"],
            "capsules": {},
        },
    )
    events = notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof")
    assert len(events) == 1
    assert events[0]["kind"] == "failure"
    message = notifier.format_event(events[0])
    assert "bad receipt" in message
    assert "Errors: 1" in message


def test_integrity_hold_is_terminal_failure_event(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "campaign" / "current_status.json",
        {
            "program_id": "p-integrity",
            "state": "integrity_hold",
            "problems": ["recorded child identity is inexact"],
            "capsules": {},
        },
    )

    events = notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof")

    assert len(events) == 1
    assert events[0]["kind"] == "failure"
    assert events[0]["state"] == "integrity_hold"
    assert "recorded child identity is inexact" in notifier.format_event(events[0])


def test_stale_dead_exact_supervisor_becomes_one_failure_event(monkeypatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "campaign" / "current_status.json",
        {
            "program_id": "p-stalled",
            "state": "running",
            "created_at": "2000-01-01T00:00:00+00:00",
            "supervisor": {"pid": 123, "create_time": 456.0},
            "problems": [],
            "capsules": {},
        },
    )
    monkeypatch.setattr(notifier, "_supervisor_identity_state", lambda identity: "gone")

    first = notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof")
    second = notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof")

    assert len(first) == 1
    assert first == second
    assert first[0]["kind"] == "failure"
    assert first[0]["state"] == "supervisor_stall"
    assert "stale for at least 10 minutes" in notifier.format_event(first[0])


def test_stall_alert_requires_identity_timestamp_and_exact_dead_result(monkeypatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    base = {
        "program_id": "p-custom",
        "state": "running",
        "created_at": "2000-01-01T00:00:00+00:00",
        "problems": [],
        "capsules": {},
    }
    _write(runs / "no-identity" / "current_status.json", base)
    _write(
        runs / "bad-time" / "current_status.json",
        {
            **base,
            "program_id": "p-bad-time",
            "created_at": "not-a-time",
            "supervisor": {"pid": 1, "create_time": 1.0},
        },
    )
    _write(
        runs / "alive" / "current_status.json",
        {**base, "program_id": "p-alive", "supervisor": {"pid": 2, "create_time": 2.0}},
    )
    monkeypatch.setattr(
        notifier,
        "_supervisor_identity_state",
        lambda identity: "alive" if isinstance(identity, dict) and identity.get("pid") == 2 else "unknown",
    )

    assert notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof") == []
    assert notifier._supervisor_identity_state({"pid": "2", "create_time": 2.0}) == "unknown"


def test_malformed_or_unsealed_status_fails_quiet(tmp_path: Path) -> None:
    path = tmp_path / "runs/campaign/current_status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-generation1-test-status/v1",
                "program_id": "generation1-test",
                "state": "integrity_hold",
                "capsules": {},
                "status_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert notifier.collect_events(runs_root=tmp_path / "runs", proof_root=tmp_path / "proof") == []


def test_dead_supervisor_has_ten_minute_grace(monkeypatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "campaign" / "current_status.json",
        {
            "program_id": "p-grace",
            "state": "running",
            "created_at": "2000-01-01T00:00:00+00:00",
            "supervisor": {"pid": 123, "create_time": 456.0},
            "problems": [],
            "capsules": {},
        },
    )
    monkeypatch.setattr(notifier, "_status_age_seconds", lambda status: 599.9)
    monkeypatch.setattr(notifier, "_supervisor_identity_state", lambda identity: "gone")

    assert notifier.collect_events(runs_root=runs, proof_root=tmp_path / "proof") == []


def test_rung_message_is_short_and_uses_simple_program_name(monkeypatch) -> None:
    monkeypatch.setattr(
        notifier,
        "host_health",
        lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0},
    )
    message = notifier.format_event(
        {
            "kind": "rung",
            "program_id": "generation1-c3-d1-router-redesign-screen-v1",
            "capsule_id": "g1_c3_router_redesign_rung_09",
            "progress": {"complete": 10, "total": 48},
            "eta": {"block_seconds": 1500.0, "session_seconds": 6848.0},
            "problems": [],
        }
    )
    assert message == (
        "🧪 MOP C3 Router Redesign\n"
        "Progress: 10/48\n"
        "Next 30 rungs: 25m\n"
        "Full queue: 1h 55m\n"
        "Health: good\n"
        "Errors: none"
    )


def test_planned_successor_programs_have_compact_labels() -> None:
    assert notifier._program_label("generation1-successor-evidence-chain-v2") == "Successor Evidence Chain"
    assert notifier._program_label("generation1-successor-evidence-chain-v3") == "Successor Evidence Chain"
    assert notifier._program_label("generation1-successor-evidence-chain-v4") == "Successor Evidence Chain"
    assert notifier._program_label("generation1-successor-horizon-v1") == "Successor Horizon"
    assert notifier._program_label("generation1-successor-extension-chain-v1") == "Successor Extension"
    assert notifier._program_label("generation1-successor-horizon-v2") == "Successor Horizon V2"
    assert (
        notifier._program_label("generation1-categorized-batch-extension-chain-v1")
        == "Categorized Batch Extension"
    )
    assert (
        notifier._program_label("generation1-successor-categorized-batch-wave-v1") == "Categorized Batch Wave"
    )


def test_event_eta_prefers_next_rung_cost() -> None:
    eta = notifier._event_eta(
        {
            "adaptive_execution": {
                "average_rung_seconds": 1.0,
                "next_rung_seconds": 32.1,
                "eta_seconds": 9_876.0,
                "workers": 1,
            }
        },
        complete=5,
        total=100,
    )
    assert eta == {"block_seconds": 963.0, "session_seconds": 9_876.0}


def test_short_block_eta_uses_seconds_instead_of_rounding_to_one_minute() -> None:
    assert notifier._format_duration(31.7) == "32s"


def test_proof_summary_extracts_c2_metrics_and_decision() -> None:
    summary = notifier.proof_summary(
        {
            "grid": {"completed_cell_count": 40, "expected_cell_count": 40},
            "overall": {
                "routed": {"mean": 0.57},
                "difficulty_static": {"mean": 0.53},
            },
            "decision": {
                "context_labeled_frozen_routing_confirmed": True,
                "ready_to_preregister_g1_c3_learned_dispatch": True,
            },
        }
    )
    assert "coverage 40/40" in summary
    assert any("routed 0.5700" in line for line in summary)
    assert any("context_labeled_frozen_routing_confirmed=True" in line for line in summary)


def test_proof_summary_uses_short_next_label_and_hides_internal_decision() -> None:
    summary = notifier.proof_summary(
        {
            "overall": {"learned_dispatch": {"mean": 0.1527}},
            "decision": {
                "ready_for_confirmatory_claim": False,
                "next_action": "freeze_best_variant_for_untouched_confirmation_design",
            },
        }
    )

    assert "Next: D1 confirmation" in summary
    assert not any("ready_for_confirmatory_claim" in line for line in summary)
    assert not any("freeze_best_variant" in line for line in summary)


def test_prime_suppresses_history_and_run_sends_only_new_event(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    first = [{"event_id": "one", "kind": "proof", "path": "a", "summary": []}]
    monkeypatch.setattr(notifier, "collect_events", lambda: first)
    assert notifier.prime(state_path=state_path)["existing_events_suppressed"] == 1
    sent: list[str] = []
    monkeypatch.setattr(notifier, "format_event", lambda event: str(event["event_id"]))
    result = notifier.run_once(
        state_path=state_path,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0
    first.append({"event_id": "two", "kind": "proof", "path": "b", "summary": []})
    result = notifier.run_once(
        state_path=state_path,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent == ["two"]


def test_run_sends_only_each_thirtieth_rung_but_keeps_terminal_immediate(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    notifier.save_state(state, state_path)
    events = [
        {
            "event_id": f"rung-{index}",
            "kind": "rung",
            "program_id": "program",
            "progress": {"complete": index, "total": 32},
        }
        for index in range(1, 32)
    ]
    events.append(
        {
            "event_id": "terminal",
            "kind": "terminal",
            "program_id": "program",
            "state": "complete",
            "progress": {"complete": 32, "total": 32},
            "problems": [],
        }
    )
    monkeypatch.setattr(notifier, "collect_events", lambda: events)
    monkeypatch.setattr(notifier, "format_event", lambda event: str(event["event_id"]))
    sent: list[str] = []
    result = notifier.run_once(
        state_path=state_path,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )

    assert result["sent"] == 2
    assert sent == ["rung-30", "terminal"]
    delivered = notifier.load_state(state_path)["delivered"]
    assert delivered["rung-29"]["status"] == "suppressed-nonmilestone"
    assert delivered["rung-31"]["status"] == "suppressed-nonmilestone"


def test_milestone_boundary_and_small_parent_chain_delivery() -> None:
    def rung(complete: int, total: int) -> dict:
        return {"kind": "rung", "progress": {"complete": complete, "total": total}}

    assert notifier._should_send(rung(1, 30)) is True
    assert notifier._should_send(rung(29, 31)) is False
    assert notifier._should_send(rung(30, 31)) is True
    assert notifier._should_send(rung(31, 31)) is True
    assert notifier._should_send(rung(59, 74)) is False
    assert notifier._should_send(rung(60, 74)) is True
    assert notifier._should_send(rung(74, 74)) is True


def test_state_seal_rejects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = notifier._new_state()
    notifier.atomic_write_json(path, state)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["primed"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        notifier.load_state(path)
    except notifier.MOPNotifierError as exc:
        assert "identity" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mutated notifier state was accepted")
