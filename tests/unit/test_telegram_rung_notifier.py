from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    # Per-rung milestone events are retired: a completed program now yields only its terminal
    # event (per-capsule progress is carried by the sub-task start/complete stream) and any
    # standalone proof.
    assert {event["kind"] for event in events} == {"terminal", "proof"}
    assert next(event for event in events if event["kind"] == "terminal")["progress"] == {
        "complete": 1,
        "total": 1,
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


def test_start_blip_shows_facet_and_full_run_eta() -> None:
    # The retired 10%-milestone ping's two ETAs (this facet, the full run) now ride on the
    # single per-sub-task START blip instead.
    message = notifier.format_event(
        {
            "kind": "subtask",
            "phase": "start",
            "program_id": "generation1-full-generations-wave-v1",
            "capsule_id": "w01_communication_repair",
            "wave_epoch": "01",
            "category": "communication_repair",
            "eta": {"facet_seconds": 420.0, "overall_seconds": 9000.0},
        }
    )
    assert message == (
        "▶️ General Run · Wave 01 — starting\n"
        "Communication & repair  (V1, M1, K1)\n"
        "→ Can perspectives fix a contradiction by messaging just the one that's wrong, "
        "instead of broadcasting to everyone?\n"
        "ETA: this facet ~7m  ·  full run ~2h 30m"
    )


def test_start_blip_omits_eta_line_when_no_estimate_is_available() -> None:
    message = notifier.format_event(
        {
            "kind": "subtask",
            "phase": "start",
            "program_id": "generation1-full-generations-wave-v1",
            "capsule_id": "w05_construction",
            "wave_epoch": "05",
            "category": "construction",
            "eta": {"facet_seconds": None, "overall_seconds": None},
        }
    )
    assert message.endswith("beat a greedy pick when there is real synergy to find?")
    assert "ETA:" not in message


def test_successor_chain_family_renders_as_unified_general_run() -> None:
    # the single-process orchestrator itself is the "General Run" umbrella
    assert notifier._program_label("general-run") == "General Run"
    # every successor-chain-family stage is presented under the unified name
    assert notifier._program_label("generation1-successor-mechanics-extended-v1") == "General Run: Mechanics"
    assert notifier._program_label("generation1-successor-evidence-chain-v2") == "General Run: Adopter"
    assert notifier._program_label("generation1-successor-evidence-chain-v4") == "General Run: Adopter"
    # the newly added v5/v6/v7 and ext-v2/ext-v3 ids join the same families
    assert notifier._program_label("generation1-successor-evidence-chain-v5") == "General Run: Adopter"
    assert notifier._program_label("generation1-successor-evidence-chain-v6") == "General Run: Adopter"
    assert notifier._program_label("generation1-successor-evidence-chain-v7") == "General Run: Adopter"
    assert notifier._program_label("generation1-successor-horizon-v1") == "General Run: Horizon 1"
    assert notifier._program_label("generation1-successor-horizon-v2") == "General Run: Horizon 2"
    assert (
        notifier._program_label("generation1-successor-extension-chain-v1") == "General Run: Horizon Waiter"
    )
    assert (
        notifier._program_label("generation1-successor-extension-chain-v3") == "General Run: Horizon Waiter"
    )
    assert (
        notifier._program_label("generation1-successor-categorized-batch-wave-v1")
        == "General Run: Categorized Wave"
    )
    assert notifier._program_label("generation1-consolidated-final-campaign-v1") == "General Run: Final"
    # the C0/C1/C2/C3/D1 one-off experiments keep their own labels, unchanged
    assert notifier._program_label("generation1-c3-d1-router-redesign-screen-v1") == "C3 Router Redesign"
    assert (
        notifier._program_label("generation1-c3-d1-frozen-producer-challenge-v1") == "D1 Frozen Replication"
    )


def test_adaptive_eta_reports_facet_and_full_run_from_the_adaptive_block() -> None:
    # facet prefers the next-rung cost over the running average; full run is the session eta.
    eta = notifier._adaptive_eta(
        {
            "adaptive_execution": {
                "average_rung_seconds": 50.0,
                "next_rung_seconds": 32.1,
                "eta_seconds": 9_876.0,
                "workers": 4,
            }
        },
        {},
        complete=5,
        total=200,
    )
    assert eta == {"facet_seconds": 32.1, "overall_seconds": 9_876.0}

    # with no next-rung cost the facet falls back to the average
    fallback = notifier._adaptive_eta(
        {"adaptive_execution": {"average_rung_seconds": 50.0, "eta_seconds": 100.0}},
        {},
        complete=1,
        total=10,
    )
    assert fallback == {"facet_seconds": 50.0, "overall_seconds": 100.0}


def test_short_block_eta_uses_seconds_instead_of_rounding_to_one_minute() -> None:
    assert notifier._format_duration(31.7) == "32s"


def _fake_host_sample(*, free_p_cores: int, hawking_active: bool):
    """A minimal but genuine controller sample: a real HostState (dataclasses.replace-able)
    plus a bounds dict, exactly the two fields _synthetic_adaptive reads."""

    from mop.studio import dynamic_worker_controller as controller

    state = controller.HostState(
        free_p_cores=free_p_cores,
        hawking_active=hawking_active,
        mem_available_gb=90.0,
        thermal_ok=True,
        on_ac=True,
        current_load=2.0,
    )
    bounds = controller.worker_bounds(state)
    return SimpleNamespace(state=state, bounds=bounds), bounds["comfortable_target"]


def test_synthetic_adaptive_derives_workers_and_eta_when_status_lacks_adaptive_execution(monkeypatch) -> None:
    """The dynamic-worker-pool stages (horizon, categorized, full-generations) never write a
    census-style adaptive_execution block, so Workers/ETA were silently blank for them. This
    fills the gap from real capsule finish timestamps plus a live controller sample."""

    idle_sample, idle_target = _fake_host_sample(free_p_cores=14, hawking_active=False)
    monkeypatch.setattr("mop.studio.dynamic_worker_controller.sample_host_state", lambda **_: idle_sample)
    capsules = {
        "a": {"finished_at": "2026-07-18T00:00:00+00:00"},
        "b": {"finished_at": "2026-07-18T00:01:00+00:00"},
        "c": {"finished_at": "2026-07-18T00:02:00+00:00"},
    }
    synthetic = notifier._synthetic_adaptive(capsules, total=74, complete=3)
    assert synthetic is not None
    assert synthetic["average_rung_seconds"] == 60.0
    # Not the cold-start ramp artifact (which would read 1 from a stateless current_workers=0
    # sample): the settled, capacity-based recommendation, matching the comfortable target.
    assert synthetic["workers"] == idle_target
    assert synthetic["workers"] > 1
    assert synthetic["mode"] == "hawking_idle"
    assert synthetic["eta_seconds"] == 60.0 * (74 - 3) / synthetic["workers"]

    # Hawking active is reflected too, and correctly overrides the settled comfortable target
    # with the small fixed reserve (the control law's own tiered logic, not re-derived here).
    from mop.studio import dynamic_worker_controller as controller

    hawking_sample, _ = _fake_host_sample(free_p_cores=14, hawking_active=True)
    monkeypatch.setattr("mop.studio.dynamic_worker_controller.sample_host_state", lambda **_: hawking_sample)
    hawking = notifier._synthetic_adaptive(capsules, total=74, complete=3)
    assert hawking is not None
    assert hawking["workers"] == controller.HAWKING_RESERVE_WORKERS
    assert hawking["mode"] == "hawking_active"

    # A real adaptive_execution block is never overridden by the synthetic fallback.
    real = {"average_rung_seconds": 5.0, "workers": 8, "mode": "hawking_idle", "eta_seconds": 100.0}
    assert (
        notifier._with_effective_adaptive({"adaptive_execution": real}, capsules, total=74, complete=3)[
            "adaptive_execution"
        ]
        == real
    )

    # Never fatal: if the controller sample fails and there are not enough timestamps to
    # derive an average either, the fallback is a clean None rather than a raised exception.
    monkeypatch.setattr(
        "mop.studio.dynamic_worker_controller.sample_host_state",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no host state available")),
    )
    assert notifier._synthetic_adaptive({}, total=74, complete=0) is None


def test_adaptive_eta_synthesizes_facet_and_full_run_for_a_dynamic_pool_stage(monkeypatch) -> None:
    """Dynamic-worker-pool stages (horizon/categorized/full-generations) never write a real
    adaptive_execution block, so the START blip's two ETAs are synthesized from capsule finish
    timestamps plus a live controller sample, not left blank."""

    idle_sample, idle_target = _fake_host_sample(free_p_cores=13, hawking_active=False)
    monkeypatch.setattr("mop.studio.dynamic_worker_controller.sample_host_state", lambda **_: idle_sample)
    capsules = {
        "g1_h01_classify": {"finished_at": "2026-07-18T00:00:00+00:00"},
        "g1_h01_d1_shard_00": {"finished_at": "2026-07-18T00:01:00+00:00"},
        "g1_h01_d1_shard_01": {"finished_at": "2026-07-18T00:02:00+00:00"},
    }
    eta = notifier._adaptive_eta({"state": "running"}, capsules, complete=3, total=74)
    # facet = the average capsule wall-time (60s between three evenly-spaced finishes)
    assert eta["facet_seconds"] == 60.0
    # full run = average * remaining / settled worker count
    assert eta["overall_seconds"] == 60.0 * (74 - 3) / idle_target
    assert eta["overall_seconds"] is not None


def test_synthetic_adaptive_hardens_average_from_a_single_finished_capsule(monkeypatch) -> None:
    # The old span method needed TWO finish timestamps and blanked to "estimating" until then.
    # The per-capsule duration (finished_at - started_at) yields a real number from just one.
    idle_sample, _ = _fake_host_sample(free_p_cores=14, hawking_active=False)
    monkeypatch.setattr("mop.studio.dynamic_worker_controller.sample_host_state", lambda **_: idle_sample)
    capsules = {
        "a": {"started_at": "2026-07-18T00:00:00+00:00", "finished_at": "2026-07-18T00:02:00+00:00"},
        "b": {"status": "pending"},
    }
    synthetic = notifier._synthetic_adaptive(capsules, total=10, complete=1)
    assert synthetic is not None
    assert synthetic["average_rung_seconds"] == 120.0
    assert synthetic["eta_seconds"] is not None


def test_synthetic_adaptive_falls_back_to_in_flight_elapsed_when_nothing_has_finished(monkeypatch) -> None:
    # Nothing finished yet, one capsule running: rather than "estimating", derive a provisional
    # per-capsule estimate from the longest in-flight elapsed so an ETA number still shows.
    idle_sample, _ = _fake_host_sample(free_p_cores=14, hawking_active=False)
    monkeypatch.setattr("mop.studio.dynamic_worker_controller.sample_host_state", lambda **_: idle_sample)
    capsules = {"a": {"started_at": "2020-01-01T00:00:00+00:00"}}
    synthetic = notifier._synthetic_adaptive(capsules, total=10, complete=0)
    assert synthetic is not None
    assert synthetic["average_rung_seconds"] is not None
    assert synthetic["average_rung_seconds"] > 0
    assert synthetic["eta_seconds"] is not None


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
        runs_root=tmp_path / "runs",
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0
    first.append({"event_id": "two", "kind": "proof", "path": "b", "summary": []})
    result = notifier.run_once(
        state_path=state_path,
        runs_root=tmp_path / "runs",
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent == ["two"]


def test_run_delivers_every_new_event_now_that_milestone_gating_is_gone(
    monkeypatch, tmp_path: Path
) -> None:
    # With the 10%-step rung stream retired, run_once no longer suppresses anything: each new
    # event (here a terminal) delivers immediately, deduped only by event_id.
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    notifier.save_state(state, state_path)
    events = [
        {
            "event_id": "terminal",
            "kind": "terminal",
            "program_id": "program",
            "state": "complete",
            "progress": {"complete": 100, "total": 100},
            "problems": [],
        }
    ]
    monkeypatch.setattr(notifier, "collect_events", lambda: events)
    monkeypatch.setattr(notifier, "format_event", lambda event: str(event["event_id"]))
    sent: list[str] = []
    result = notifier.run_once(
        state_path=state_path,
        runs_root=tmp_path / "runs",
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent == ["terminal"]

    # a second poll re-observes the same terminal and never re-sends it
    result = notifier.run_once(
        state_path=state_path,
        runs_root=tmp_path / "runs",
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0


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


def test_full_generations_programs_render_under_general_chain() -> None:
    assert notifier._program_label("generation1-full-generations-wave-v1") == "General Run: Full Generations"
    assert (
        notifier._program_label("generation1-full-generations-extension-chain-v1")
        == "General Run: Full Gen Waiter"
    )


def test_worker_line_reflects_mode_and_is_omitted_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        notifier,
        "host_health",
        lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0},
    )

    def terminal(adaptive: dict | None) -> str:
        event = {
            "kind": "terminal",
            "program_id": "generation1-c3-d1-router-redesign-screen-v1",
            "state": "complete",
            "progress": {"complete": 48, "total": 48},
            "problems": [],
        }
        if adaptive is not None:
            event["adaptive"] = adaptive
        return notifier.format_event(event)

    idle = terminal({"workers": 16, "mode": "idle_opportunistic"})
    assert idle == (
        "✅ MOP C3 Router Redesign: complete\n"
        "Progress: 48/48 (100%)\n"
        "Chain stage complete\n"
        "Workers: 16 (idle burst)\n"
        "Health: good\n"
        "Errors: none"
    )
    assert "Workers: 1 (Hawking active)" in terminal({"workers": 1, "mode": "hawking_active"})
    assert "Workers: 8 (steady)" in terminal({"workers": 8, "mode": "steady"})
    # Regression: "hawking_idle" (the real census mode string when Hawking is NOT active and
    # the pool runs at full speed) must read as idle burst, not be mislabeled "Hawking active"
    # by a bare "hawking" substring match landing before the "idle" check.
    assert "Workers: 8 (idle burst)" in terminal({"workers": 8, "mode": "hawking_idle"})
    # adaptive block absent: no worker line, original shape preserved
    assert "Workers:" not in terminal(None)


def test_terminal_chain_stage_annotation_and_worker_line(monkeypatch) -> None:
    monkeypatch.setattr(
        notifier,
        "host_health",
        lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0},
    )
    known = notifier.format_event(
        {
            "kind": "terminal",
            "program_id": "generation1-successor-mechanics-extended-v1",
            "state": "complete",
            "progress": {"complete": 12, "total": 12},
            "problems": [],
            "adaptive": {"workers": 16, "mode": "idle_opportunistic"},
        }
    )
    assert known == (
        "✅ MOP General Run: Mechanics: complete\n"
        "Progress: 12/12 (100%)\n"
        "Chain stage complete\n"
        "Workers: 16 (idle burst)\n"
        "Health: good\n"
        "Errors: none"
    )
    # unknown program: no chain-stage annotation
    unknown = notifier.format_event(
        {
            "kind": "terminal",
            "program_id": "generation1-unlisted-program-v9",
            "state": "complete",
            "progress": {"complete": 3, "total": 3},
            "problems": [],
        }
    )
    assert "Chain stage complete" not in unknown
    # a failure of a known chain stage is never labeled a completed stage
    failed_known = notifier.format_event(
        {
            "kind": "failure",
            "program_id": "generation1-successor-mechanics-extended-v1",
            "state": "failure_hold",
            "progress": {"complete": 4, "total": 12},
            "problems": ["bad receipt"],
        }
    )
    assert "Chain stage complete" not in failed_known


def test_throttle_shift_fires_once_per_transition(tmp_path: Path) -> None:
    runs = tmp_path / "runs"

    def write_mode(mode: str, workers: int) -> None:
        _write(
            runs / "flip" / "current_status.json",
            {
                "program_id": "flip",
                "state": "running",
                "problems": [],
                "capsules": {},
                "adaptive_execution": {"workers": workers, "mode": mode},
            },
        )

    # first observation only records the mode; it never announces a transition
    write_mode("hawking_active", 1)
    events, seen = notifier._collect_throttle_events({}, runs_root=runs)
    assert events == []
    assert seen == {"generation1-flip": "hawking"}

    # hawking -> idle fires exactly one throttle event
    write_mode("idle_opportunistic", 16)
    events, seen = notifier._collect_throttle_events(seen, runs_root=runs)
    assert len(events) == 1
    assert events[0]["kind"] == "throttle"
    assert events[0]["mode"] == "idle"
    assert events[0]["event_id"] == "throttle-shift/generation1-flip/idle"
    assert seen == {"generation1-flip": "idle"}
    assert notifier.format_event(events[0]) == "⚙️ MOP Flip: Hawking cleared, ramping to 16 workers"

    # a steady idle poll does not re-fire
    events, seen = notifier._collect_throttle_events(seen, runs_root=runs)
    assert events == []

    # idle -> hawking fires the opposite direction
    write_mode("hawking_active", 1)
    events, seen = notifier._collect_throttle_events(seen, runs_root=runs)
    assert len(events) == 1
    assert events[0]["mode"] == "hawking"
    assert notifier.format_event(events[0]) == "⚙️ MOP Flip: Hawking active, ceding to 1 workers"


def test_throttle_shift_delivers_once_through_run_once(monkeypatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "flip" / "current_status.json",
        {
            "program_id": "flip",
            "state": "running",
            "problems": [],
            "capsules": {},
            "adaptive_execution": {"workers": 16, "mode": "idle_opportunistic"},
        },
    )
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    state["last_seen_mode"] = {"generation1-flip": "hawking"}
    notifier.save_state(state, state_path)
    monkeypatch.setattr(notifier, "collect_events", lambda: [])
    sent: list[str] = []

    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent == ["⚙️ MOP Flip: Hawking cleared, ramping to 16 workers"]

    # the same idle status on the next poll must not re-announce the transition
    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0
    assert sent == ["⚙️ MOP Flip: Hawking cleared, ramping to 16 workers"]
    assert notifier.load_state(state_path)["last_seen_mode"] == {"generation1-flip": "idle"}


def test_lane_subtask_fires_once_on_completion_and_names_the_mechanism(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    runs = tmp_path / "runs"

    def write_lane(complete: int, total: int) -> None:
        _write(
            runs / "mechanics" / "current_status.json",
            {
                "program_id": "successor-mechanics-extended-v1",
                "state": "running",
                "problems": [],
                "capsules": {},
                "lane_progress": {"G1-E1": {"complete": complete, "total": total}},
            },
        )

    # first observation of an in-progress lane only records state; it never fires
    write_lane(128, 129)
    events, lanes, capsules = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []

    # the completing transition fires exactly one named sub-task blip
    write_lane(129, 129)
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert len(events) == 1
    assert events[0]["kind"] == "subtask"
    assert events[0]["lane_short"] == "E1"
    assert events[0]["event_id"] == "subtask/lane/generation1-successor-mechanics-extended-v1/G1-E1"
    assert notifier.format_event(events[0]) == (
        "✅ General Run · E1 done\n"
        " • E1 — event formation: carve the stream into meaningful events (the continual-learning gate)\n"
        "Progress: 129/129 (100%)  ·  Health: good"
    )

    # a steady complete poll does not re-announce the same lane
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert events == []


def test_partial_lane_progress_fires_nothing(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "mechanics" / "current_status.json",
        {
            "program_id": "successor-mechanics-extended-v1",
            "state": "running",
            "problems": [],
            "capsules": {},
            "lane_progress": {"G1-D1": {"complete": 5, "total": 10}},
        },
    )
    events, lanes, capsules = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []
    # still incomplete on a later poll: still nothing
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert events == []


def test_lane_already_complete_on_first_sight_is_treated_as_history(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "mechanics" / "current_status.json",
        {
            "program_id": "successor-mechanics-extended-v1",
            "state": "complete",
            "problems": [],
            "capsules": {},
            "lane_progress": {"G1-A1": {"complete": 129, "total": 129}},
        },
    )
    events, lanes, _ = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []
    assert lanes["generation1-successor-mechanics-extended-v1/G1-A1"] == "complete"


def test_wave_capsule_subtask_parses_epoch_and_category_and_fires_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    runs = tmp_path / "runs"

    def write_capsule(done: bool) -> None:
        row: dict = (
            {
                "returncode": 0,
                "finished_at": "2026-07-16T00:00:00+00:00",
                "artifacts": [{"path": "proof/x.json", "sha256": "b" * 64, "all_ok": True}],
            }
            if done
            else {"returncode": None, "artifacts": []}
        )
        _write(
            runs / "wave" / "current_status.json",
            {
                "program_id": "full-generations-wave-v1",
                "state": "running",
                "problems": [],
                "capsules": {"w08_construction": row},
            },
        )

    # in-progress capsule: first observation records, never fires
    write_capsule(done=False)
    events, lanes, capsules = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []

    # completed capsule fires one blip naming its epoch, category and mechanisms
    write_capsule(done=True)
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert len(events) == 1
    assert events[0]["wave_epoch"] == "08"
    assert events[0]["category"] == "construction"
    assert events[0]["event_id"] == "subtask/capsule/generation1-full-generations-wave-v1/w08_construction"
    assert notifier.format_event(events[0]) == (
        "✅ General Run · Wave 08 — Construction search done\n"
        " • G1 — construction: cost-charged topology search over shadow coalitions\n"
        "Wave progress: 1/1 (100%)  ·  Health: good"
    )

    # steady completed poll does not re-fire
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert events == []


def test_wave_capsule_formation_trace_names_all_covered_mechanisms(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    runs = tmp_path / "runs"
    _write(
        runs / "wave" / "current_status.json",
        {
            "program_id": "successor-categorized-batch-wave-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "w15_formation_trace": {
                    "returncode": 0,
                    "finished_at": "2026-07-16T00:00:00+00:00",
                    "artifacts": [{"path": "proof/x.json", "sha256": "c" * 64, "all_ok": True}],
                }
            },
        },
    )
    seed = {"generation1-successor-categorized-batch-wave-v1/w15_formation_trace": "incomplete"}
    events, _, _ = notifier._collect_subtask_events({}, seed, runs_root=runs)
    assert len(events) == 1
    assert notifier.format_event(events[0]) == (
        "✅ General Run · Wave 15 — Formation & trace done\n"
        " • C0 — trace stability: does it form the same events cross-seed and cross-session?\n"
        " • E1 — event formation: carve the stream into meaningful events (the continual-learning gate)\n"
        "Wave progress: 1/1 (100%)  ·  Health: good"
    )


def test_wave_capsule_only_fires_for_general_chain_programs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "other" / "current_status.json",
        {
            "program_id": "some-unlisted-experiment-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "w01_construction": {
                    "returncode": 0,
                    "finished_at": "2026-07-16T00:00:00+00:00",
                    "artifacts": [{"path": "proof/x.json", "sha256": "d" * 64, "all_ok": True}],
                }
            },
        },
    )
    seed = {"generation1-some-unlisted-experiment-v1/w01_construction": "incomplete"}
    events, _, _ = notifier._collect_subtask_events({}, seed, runs_root=runs)
    assert events == []


def test_wave_capsule_fires_a_start_blip_on_pending_to_running(tmp_path: Path) -> None:
    runs = tmp_path / "runs"

    def write_capsule(*, status: str, started: bool) -> None:
        row: dict = {"returncode": None, "artifacts": [], "status": status}
        if started:
            row["started_at"] = "2026-07-19T00:00:00+00:00"
        _write(
            runs / "wave" / "current_status.json",
            {
                "program_id": "full-generations-wave-v1",
                "state": "running",
                "problems": [],
                "capsules": {"w01_communication_repair": row},
            },
        )

    # first observation of a pending capsule records state and never fires
    write_capsule(status="pending", started=False)
    events, lanes, capsules = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []

    # pending -> running fires exactly one start blip, distinct id from the completion blip
    write_capsule(status="running", started=True)
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert len(events) == 1
    assert events[0]["phase"] == "start"
    assert (
        events[0]["event_id"]
        == "subtask/capsule-start/generation1-full-generations-wave-v1/w01_communication_repair"
    )
    message = notifier.format_event(events[0])
    assert message.startswith(
        "▶️ General Run · Wave 01 — starting\n"
        "Communication & repair  (V1, M1, K1)\n"
        "→ Can perspectives fix a contradiction by messaging just the one that's wrong, "
        "instead of broadcasting to everyone?"
    )
    # hardened ETA: a running capsule (started, none finished) still yields real numbers from
    # its in-flight elapsed, not "estimating"
    assert "\nETA: this facet ~" in message

    # a steady running poll does not re-announce the start
    events, lanes, capsules = notifier._collect_subtask_events(lanes, capsules, runs_root=runs)
    assert events == []


def test_wave_capsule_running_on_first_sight_never_back_announces_a_start(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "wave" / "current_status.json",
        {
            "program_id": "full-generations-wave-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "w01_construction": {
                    "returncode": None,
                    "artifacts": [],
                    "status": "running",
                    "started_at": "2026-07-19T00:00:00+00:00",
                }
            },
        },
    )
    events, _, capsules = notifier._collect_subtask_events({}, {}, runs_root=runs)
    assert events == []
    assert capsules["generation1-full-generations-wave-v1/w01_construction"] == "running"


def test_legacy_incomplete_capsule_state_never_emits_a_spurious_start(tmp_path: Path) -> None:
    # A capsule persisted under the pre-lifecycle "incomplete" vocabulary must upgrade cleanly:
    # transitioning into "running" is NOT a witnessed pending->running start, so it stays silent.
    runs = tmp_path / "runs"
    _write(
        runs / "wave" / "current_status.json",
        {
            "program_id": "full-generations-wave-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "w01_construction": {
                    "returncode": None,
                    "artifacts": [],
                    "status": "running",
                    "started_at": "2026-07-19T00:00:00+00:00",
                }
            },
        },
    )
    seed = {"generation1-full-generations-wave-v1/w01_construction": "incomplete"}
    events, _, capsules = notifier._collect_subtask_events({}, seed, runs_root=runs)
    assert events == []
    assert capsules["generation1-full-generations-wave-v1/w01_construction"] == "running"


def test_wave_capsule_start_delivers_once_through_run_once(monkeypatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "wave" / "current_status.json",
        {
            "program_id": "full-generations-wave-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "w01_construction": {
                    "returncode": None,
                    "artifacts": [],
                    "status": "running",
                    "started_at": "2026-07-19T00:00:00+00:00",
                }
            },
        },
    )
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    # seen as pending on the prior poll, so this poll witnesses the pending -> running start
    state["last_seen_complete_capsules"] = {
        "generation1-full-generations-wave-v1/w01_construction": "pending"
    }
    notifier.save_state(state, state_path)
    monkeypatch.setattr(notifier, "collect_events", lambda: [])
    sent: list[str] = []

    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent[0].startswith("▶️ General Run · Wave 01 — starting")
    assert "Construction search  (G1)" in sent[0]

    # the same running status on the next poll must not re-announce the start
    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0


def _halfway_status(runs: Path, n_complete: int, n_total: int, *, state: str = "running") -> None:
    def complete_row(i: int) -> dict:
        return {
            "returncode": 0,
            "started_at": f"2026-07-18T00:0{i}:00+00:00",
            "finished_at": f"2026-07-18T00:0{i + 1}:00+00:00",
            "artifacts": [{"all_ok": True, "sha256": str(i) * 64}],
        }

    capsules = {
        f"cap_{i}": (complete_row(i) if i < n_complete else {"status": "pending"}) for i in range(n_total)
    }
    _write(
        runs / "wave" / "current_status.json",
        {
            "program_id": "successor-categorized-batch-wave-v1",
            "state": state,
            "problems": [],
            "capsules": capsules,
        },
    )


def test_progress_halfway_fires_once_when_a_general_run_crosses_the_half_mark(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    program = "generation1-successor-categorized-batch-wave-v1"
    runs = tmp_path / "runs"

    # below half on first sight: only record the stage, never fire
    _halfway_status(runs, 1, 6)
    events, seen = notifier._collect_progress_events({}, runs_root=runs)
    assert events == []
    assert seen == {program: "below"}

    # crossing to >= 50% fires exactly one halfway ping carrying progress and a real ETA
    _halfway_status(runs, 3, 6)
    events, seen = notifier._collect_progress_events(seen, runs_root=runs)
    assert len(events) == 1
    assert events[0]["kind"] == "progress"
    assert events[0]["event_id"] == f"progress/{program}/halfway"
    assert events[0]["progress"] == {"complete": 3, "total": 6}
    message = notifier.format_event(events[0])
    assert message.startswith("⏳ General Run: Categorized Wave — halfway\nProgress: 3/6 (50%)")
    # hardened: real numbers (60s per-capsule duration), not "estimating"
    assert "ETA: this facet ~60s" in message
    assert seen[program] == "half"

    # a later poll still past half does not re-announce the checkpoint
    _halfway_status(runs, 4, 6)
    events, seen = notifier._collect_progress_events(seen, runs_root=runs)
    assert events == []


def test_progress_halfway_is_suppressed_for_a_terminal_run(tmp_path: Path) -> None:
    program = "generation1-successor-categorized-batch-wave-v1"
    runs = tmp_path / "runs"
    _halfway_status(runs, 3, 6, state="complete")
    events, _ = notifier._collect_progress_events({program: "below"}, runs_root=runs)
    assert events == []


def test_progress_halfway_ignores_non_general_run_programs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(
        runs / "other" / "current_status.json",
        {
            "program_id": "some-unlisted-experiment-v1",
            "state": "running",
            "problems": [],
            "capsules": {
                "cap_0": {
                    "returncode": 0,
                    "finished_at": "2026-07-18T00:01:00+00:00",
                    "artifacts": [{"all_ok": True, "sha256": "a" * 64}],
                },
                "cap_1": {"status": "pending"},
            },
        },
    )
    events, seen = notifier._collect_progress_events({}, runs_root=runs)
    assert events == []
    assert seen == {}


def test_progress_halfway_delivers_once_through_run_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    program = "generation1-successor-categorized-batch-wave-v1"
    runs = tmp_path / "runs"
    _halfway_status(runs, 3, 6)
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    state["last_seen_progress"] = {program: "below"}  # witnessed below the mark last poll
    notifier.save_state(state, state_path)
    monkeypatch.setattr(notifier, "collect_events", lambda: [])
    sent: list[str] = []

    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent[0].startswith("⏳ General Run: Categorized Wave — halfway")

    # a second poll still at half must not re-send the checkpoint
    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0
    assert notifier.load_state(state_path)["last_seen_progress"] == {program: "half"}


def test_lane_subtask_delivers_once_through_run_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        notifier, "host_health", lambda: {"pressure": 1, "swap_mb": 11.0, "disk_free_gb": 182.0}
    )
    runs = tmp_path / "runs"
    _write(
        runs / "mechanics" / "current_status.json",
        {
            "program_id": "successor-mechanics-extended-v1",
            "state": "running",
            "problems": [],
            "capsules": {},
            "lane_progress": {"G1-E1": {"complete": 129, "total": 129}},
        },
    )
    state_path = tmp_path / "state.json"
    state = notifier._new_state()
    state["primed"] = True
    state["last_seen_complete_lanes"] = {"generation1-successor-mechanics-extended-v1/G1-E1": "incomplete"}
    notifier.save_state(state, state_path)
    monkeypatch.setattr(notifier, "collect_events", lambda: [])
    sent: list[str] = []

    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 1
    assert sent == [
        "✅ General Run · E1 done\n"
        " • E1 — event formation: carve the stream into meaningful events (the continual-learning gate)\n"
        "Progress: 129/129 (100%)  ·  Health: good"
    ]

    # the same complete lane on the next poll must not re-announce the sub-task
    result = notifier.run_once(
        state_path=state_path,
        runs_root=runs,
        sender=lambda text: sent.append(text) or {"message_id": len(sent), "sent_at": "now"},
    )
    assert result["sent"] == 0
    assert sent == [
        "✅ General Run · E1 done\n"
        " • E1 — event formation: carve the stream into meaningful events (the continual-learning gate)\n"
        "Progress: 129/129 (100%)  ·  Health: good"
    ]
    assert notifier.load_state(state_path)["last_seen_complete_lanes"] == {
        "generation1-successor-mechanics-extended-v1/G1-E1": "complete"
    }
