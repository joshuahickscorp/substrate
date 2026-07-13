from __future__ import annotations

import plistlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.config import REPO_ROOT
from mop.studio import campaign_recovery as recovery


def _sealed(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: recovery.canonical_sha256(core)}


def _mock_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    router_state_name: str = "waiting_campaign",
    campaign_state_name: str = "backoff",
    current_stage: str | None = "phase1",
    registry_runs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    stage = SimpleNamespace(stage_id="phase1", plan_path=tmp_path / "campaign.json")
    plan = SimpleNamespace(
        router_id="router-v1",
        path=tmp_path / "router.json",
        sha256="1" * 64,
        out_dir=tmp_path / "router-run",
        stages=(stage,),
    )
    campaign = SimpleNamespace(
        campaign_id="campaign-v1",
        path=stage.plan_path,
        sha256="2" * 64,
        out_dir=tmp_path / "campaign-run",
        state_root=tmp_path / "throttle",
        policy=SimpleNamespace(sha256="3" * 64),
    )
    router_state = {
        "status": router_state_name,
        "current_stage": current_stage,
        "supervisor": {"pid": 101, "create_time": 100.0},
    }
    router_status = {"state": router_state_name}
    campaign_state = {
        "status": campaign_state_name,
        "supervisor": {"pid": 202, "create_time": 100.0},
    }
    campaign_status = {"state": campaign_state_name}
    monkeypatch.setattr(recovery, "validate_live_router_plan", lambda _plan: {"valid": True})
    monkeypatch.setattr(
        recovery,
        "_validate_router_snapshots",
        lambda _plan: (router_state, router_status),
    )
    monkeypatch.setattr(recovery, "load_campaign_plan", lambda _path: campaign)
    monkeypatch.setattr(
        recovery,
        "_validate_campaign_snapshots",
        lambda _campaign: (campaign_state, campaign_status),
    )
    monkeypatch.setattr(
        recovery,
        "_raw_registry",
        lambda _campaign: {
            "schema": recovery.ACTIVE_REGISTRY_SCHEMA,
            "updated_at": None,
            "runs": registry_runs or {},
        },
    )
    monkeypatch.setattr(
        recovery,
        "_checkpoint_authority",
        lambda _campaign, _state: {
            "step_id": "x0",
            "aggregate_sha256": "4" * 64,
            "run_id": "x0-leg02",
        },
    )
    return plan, campaign


def _missing_process(pid: int) -> recovery.ProcessObservation:
    return recovery.ProcessObservation(pid, False, None, ())


def test_ready_only_after_exact_seals_checkpoint_and_reboot_stale_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(monkeypatch, tmp_path)
    decision = recovery.build_recovery_plan(plan, process_probe=_missing_process, boot_time=200.0)
    assert decision["disposition"] == "ready"
    assert decision["safe_to_resume"] is True
    assert decision["facts"]["router"]["process"]["status"] == "stale-after-reboot"
    assert decision["facts"]["campaign"]["process"]["status"] == "stale-after-reboot"
    assert decision["facts"]["campaign"]["checkpoint"]["aggregate_sha256"] == "4" * 64
    core = dict(decision)
    assert core.pop("recovery_sha256") == recovery.canonical_sha256(core)


def test_reused_live_pid_is_refused_without_signaling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(monkeypatch, tmp_path)

    def reused(pid: int) -> recovery.ProcessObservation:
        return recovery.ProcessObservation(pid, True, 150.0, ("unrelated",))

    decision = recovery.build_recovery_plan(plan, process_probe=reused, boot_time=50.0)
    assert decision["disposition"] == "refused"
    assert decision["safe_to_resume"] is False
    assert decision["signals_sent"] is False
    assert decision["facts"]["router"]["process"]["status"] == "pid-reused"


def test_nonempty_raw_registry_defers_even_when_row_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(
        monkeypatch,
        tmp_path,
        registry_runs={"old-run": {"scheduler_pid": 999999}},
    )
    decision = recovery.build_recovery_plan(plan, process_probe=_missing_process, boot_time=200.0)
    assert decision["disposition"] == "deferred"
    assert decision["facts"]["raw_active_registry"]["run_ids"] == ["old-run"]
    assert "raw throttle registry is not exactly empty" in decision["problems"]


def test_corrupt_snapshot_or_config_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        recovery,
        "_validate_router_snapshots",
        lambda _plan: (_ for _ in ()).throw(ValueError("router state state_sha256 mismatch")),
    )
    decision = recovery.build_recovery_plan(plan, process_probe=_missing_process, boot_time=200.0)
    assert decision["disposition"] == "refused"
    assert decision["facts"] == {}
    assert "state_sha256 mismatch" in decision["problems"][0]


def test_clean_terminal_router_is_not_restarted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(
        monkeypatch,
        tmp_path,
        router_state_name="complete_null_stop",
        current_stage=None,
    )
    decision = recovery.build_recovery_plan(plan, process_probe=_missing_process, boot_time=200.0)
    assert decision["disposition"] == "already-terminal"
    assert decision["safe_to_resume"] is False
    assert set(decision["facts"]) == {"router"}


def test_router_validation_runtime_error_becomes_sealed_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, _ = _mock_snapshots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        recovery,
        "validate_live_router_plan",
        lambda _plan: (_ for _ in ()).throw(RuntimeError("live authority drift")),
    )
    decision = recovery.build_recovery_plan(plan, process_probe=_missing_process, boot_time=200.0)
    assert decision["disposition"] == "refused"
    assert decision["safe_to_resume"] is False
    assert decision["facts"] == {}
    assert decision["problems"] == ["RuntimeError: live authority drift"]
    core = dict(decision)
    assert core.pop("recovery_sha256") == recovery.canonical_sha256(core)


def test_checkpoint_authority_requires_a_valid_resumable_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = "4" * 64
    campaign = SimpleNamespace(state_root=tmp_path)
    state = {
        "current_step": "x0",
        "steps": {"x0": {"last_checkpoint_sha256": checkpoint}},
        "launches": [{"step_id": "x0", "run_id": "x0-leg01"}],
    }
    monkeypatch.setattr(
        recovery,
        "probe_run_outcome",
        lambda _run_id, _root: SimpleNamespace(
            status="complete",
            checkpoint_sha256=checkpoint,
            receipt_path=str(tmp_path / "run_receipt.json"),
        ),
    )
    with pytest.raises(ValueError, match="not joined to a valid governor receipt"):
        recovery._checkpoint_authority(campaign, state)


def test_exact_live_router_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, campaign = _mock_snapshots(monkeypatch, tmp_path)
    commands = {101: recovery._router_command(plan), 202: recovery._campaign_command(campaign)}

    def exact(pid: int) -> recovery.ProcessObservation:
        return recovery.ProcessObservation(pid, True, 100.0, commands[pid])

    decision = recovery.build_recovery_plan(plan, process_probe=exact, boot_time=50.0)
    assert decision["disposition"] == "already-running"
    assert decision["safe_to_resume"] is False


def test_resume_deferred_exits_cleanly_without_calling_launcher(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace()
    monkeypatch.setattr(recovery, "load_router_plan", lambda _path: plan)
    monkeypatch.setattr(
        recovery,
        "build_recovery_plan",
        lambda _plan: {"disposition": "deferred", "safe_to_resume": False},
    )
    monkeypatch.setattr(
        recovery,
        "start_router_detached",
        lambda *_args, **_kwargs: pytest.fail("unsafe recovery called the launcher"),
    )
    assert recovery.main(["resume", "--config", "unused.json", "--execute"]) == 0
    assert '"disposition": "deferred"' in capsys.readouterr().out


def test_resume_inconsistent_ready_flag_fails_closed_without_launching(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace()
    monkeypatch.setattr(recovery, "load_router_plan", lambda _path: plan)
    monkeypatch.setattr(
        recovery,
        "build_recovery_plan",
        lambda _plan: {"disposition": "ready", "safe_to_resume": False},
    )
    monkeypatch.setattr(
        recovery,
        "start_router_detached",
        lambda *_args, **_kwargs: pytest.fail("inconsistent recovery decision called the launcher"),
    )
    assert recovery.main(["resume", "--config", "unused.json", "--execute"]) == 0
    assert '"safe_to_resume": false' in capsys.readouterr().out


def test_plan_is_read_only_even_when_recovery_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace()
    monkeypatch.setattr(recovery, "load_router_plan", lambda _path: plan)
    monkeypatch.setattr(
        recovery,
        "build_recovery_plan",
        lambda _plan: {"disposition": "ready", "safe_to_resume": True},
    )
    monkeypatch.setattr(
        recovery,
        "start_router_detached",
        lambda *_args, **_kwargs: pytest.fail("read-only plan called the launcher"),
    )
    assert recovery.main(["plan", "--config", "unused.json"]) == 0
    assert '"disposition": "ready"' in capsys.readouterr().out


def test_resume_requires_execute_without_calling_launcher(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace()
    monkeypatch.setattr(recovery, "load_router_plan", lambda _path: plan)
    monkeypatch.setattr(
        recovery,
        "build_recovery_plan",
        lambda _plan: {"disposition": "ready", "safe_to_resume": True},
    )
    monkeypatch.setattr(
        recovery,
        "start_router_detached",
        lambda *_args, **_kwargs: pytest.fail("resume without --execute called the launcher"),
    )
    assert recovery.main(["resume", "--config", "unused.json"]) == 2
    assert "requires explicit --execute" in capsys.readouterr().err


def test_resume_execute_launches_only_a_fresh_ready_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace()
    calls: list[tuple[Any, bool, bool]] = []
    monkeypatch.setattr(recovery, "load_router_plan", lambda _path: plan)
    monkeypatch.setattr(
        recovery,
        "build_recovery_plan",
        lambda _plan: {"disposition": "ready", "safe_to_resume": True},
    )

    def launch(candidate: Any, *, execute: bool, use_caffeinate: bool) -> dict[str, Any]:
        calls.append((candidate, execute, use_caffeinate))
        return {"launched_pid": 303}

    monkeypatch.setattr(recovery, "start_router_detached", launch)
    assert recovery.main(["resume", "--config", "unused.json", "--execute"]) == 0
    assert calls == [(plan, True, True)]
    assert '"launched_pid": 303' in capsys.readouterr().out


def test_launchd_template_is_noninstalled_periodic_fail_closed_resume() -> None:
    path = REPO_ROOT / "configs/campaign/org.mop.null-safe-campaign.plist.template"
    payload = plistlib.loads(path.read_bytes())
    arguments = payload["ProgramArguments"]
    assert payload["Label"] == "org.mop.null-safe-campaign"
    assert payload["RunAtLoad"] is True and payload["StartInterval"] == 300
    assert arguments[1].endswith("/scripts/plan_campaign_recovery.py")
    assert arguments[2:] == [
        "resume",
        "--config",
        "__MOP_REPO_ROOT__/configs/campaign/mac_studio_substrate_null_safe_router.json",
        "--execute",
    ]
    assert all("kill" not in value and "launchctl" not in value for value in arguments)


def test_seal_validator_rejects_tampering() -> None:
    payload = _sealed({"schema": "fixture", "state": "backoff"}, "state_sha256")
    recovery._validate_seal(payload, "state_sha256", "fixture")
    payload["state"] = "running"
    with pytest.raises(ValueError, match="state_sha256 mismatch"):
        recovery._validate_seal(payload, "state_sha256", "fixture")
