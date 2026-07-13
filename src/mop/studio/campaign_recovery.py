"""Fail-closed reboot recovery for the detached null-safe campaign router.

Planning is read-only.  The explicit resume operation starts the existing router
only after the sealed controller snapshots, live implementation authorities,
durable checkpoint, process identities, and raw throttle registry all agree.
This module never sends signals and never deletes or rewrites stale state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from ..config import REPO_ROOT
from .campaign_supervisor import (
    CONTROL_FILE as CAMPAIGN_CONTROL_FILE,
)
from .campaign_supervisor import (
    RUN_RESUMABLE_STATES,
    load_campaign_plan,
    probe_run_outcome,
)
from .campaign_supervisor import (
    STATE_FILE as CAMPAIGN_STATE_FILE,
)
from .campaign_supervisor import (
    STATE_SCHEMA as CAMPAIGN_STATE_SCHEMA,
)
from .campaign_supervisor import (
    STATUS_FILE as CAMPAIGN_STATUS_FILE,
)
from .campaign_supervisor import (
    STATUS_SCHEMA as CAMPAIGN_STATUS_SCHEMA,
)
from .campaign_supervisor import (
    TERMINAL_STATES as CAMPAIGN_TERMINAL_STATES,
)
from .local_throttle import IMPLEMENTATION_PATH as THROTTLE_IMPLEMENTATION_PATH
from .null_safe_campaign_router import (
    ROUTER_CONTROL_FILE,
    ROUTER_STATE_FILE,
    ROUTER_STATE_SCHEMA,
    ROUTER_STATUS_FILE,
    ROUTER_STATUS_SCHEMA,
    RouterPlan,
    load_router_plan,
    start_router_detached,
    validate_live_router_plan,
)

RECOVERY_SCHEMA = "mop-null-safe-campaign-recovery-plan/v1"
ACTIVE_REGISTRY_SCHEMA = "mop-local-throttle-active-registry/v1"
ROUTER_CLEAN_TERMINAL = frozenset({"complete", "complete_null_stop", "drained"})
ROUTER_HOLD_STATES = frozenset({"failure_hold"})


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str, *, maximum_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0 or stat.st_size > maximum_bytes:
        raise ValueError(f"{label} byte envelope is invalid")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise ValueError(f"{label} changed while being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise ValueError(f"{label} {field} mismatch")


@dataclass(frozen=True, slots=True)
class ProcessObservation:
    pid: int
    present: bool
    create_time: float | None
    cmdline: tuple[str, ...]
    error: str | None = None


ProcessProbe = Callable[[int], ProcessObservation]


def observe_process(pid: int) -> ProcessObservation:
    try:
        process = psutil.Process(pid)
        return ProcessObservation(
            pid=pid,
            present=True,
            create_time=process.create_time(),
            cmdline=tuple(process.cmdline()),
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return ProcessObservation(pid, False, None, ())
    except (psutil.AccessDenied, OSError) as exc:
        return ProcessObservation(pid, True, None, (), f"{type(exc).__name__}: {exc}")


def _identity_report(
    identity: object,
    *,
    expected_command: tuple[str, ...],
    process_probe: ProcessProbe,
    boot_time: float,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("controller process identity is missing")
    pid = identity.get("pid")
    create_time = identity.get("create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(create_time, bool)
        or not isinstance(create_time, int | float)
        or not math.isfinite(float(create_time))
    ):
        raise ValueError("controller process identity is invalid")
    observation = process_probe(pid)
    report = {
        "recorded": {"pid": pid, "create_time": float(create_time)},
        "boot_time": float(boot_time),
        "expected_command": list(expected_command),
        "observation": asdict(observation),
    }
    if observation.error is not None:
        return {**report, "status": "unobservable", "safe": False}
    if not observation.present:
        reason = "stale-after-reboot" if float(create_time) < boot_time else "stale-exited"
        return {**report, "status": reason, "safe": True}
    if observation.create_time is None or not math.isclose(
        observation.create_time,
        float(create_time),
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        return {**report, "status": "pid-reused", "safe": False}
    if observation.cmdline != expected_command:
        return {**report, "status": "identity-command-mismatch", "safe": False}
    return {**report, "status": "exact-live", "safe": True}


def _validate_router_snapshots(plan: RouterPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _read_object(plan.out_dir / ROUTER_STATE_FILE, "router state")
    status = _read_object(plan.out_dir / ROUTER_STATUS_FILE, "router status")
    if state.get("schema") != ROUTER_STATE_SCHEMA or state.get("router_id") != plan.router_id:
        raise ValueError("router state identity drifted")
    if status.get("schema") != ROUTER_STATUS_SCHEMA or status.get("router_id") != plan.router_id:
        raise ValueError("router status identity drifted")
    _validate_seal(state, "state_sha256", "router state")
    _validate_seal(status, "status_sha256", "router status")
    authority = {"path": str(plan.path), "sha256": plan.sha256}
    if state.get("router_plan") != authority or status.get("router_plan") != authority:
        raise ValueError("router snapshot plan authority drifted")
    joins = {
        "supervisor": "supervisor",
        "execution_enabled": "execution_enabled",
        "status": "state",
        "current_stage": "current_stage",
        "stage_index": "stage_index",
        "stage_results": "stage_results",
        "started_at": "started_at",
        "updated_at": "updated_at",
        "finished_at": "finished_at",
        "problems": "problems",
    }
    for state_field, status_field in joins.items():
        if state.get(state_field) != status.get(status_field):
            raise ValueError(f"router state/status join drifted at {state_field}")
    if state.get("execution_enabled") is not True:
        raise ValueError("router snapshot was not execution-enabled")
    return state, status


def _validate_campaign_snapshots(
    campaign: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _read_object(campaign.out_dir / CAMPAIGN_STATE_FILE, "campaign state")
    status = _read_object(campaign.out_dir / CAMPAIGN_STATUS_FILE, "campaign status")
    if state.get("schema") != CAMPAIGN_STATE_SCHEMA or state.get("campaign_id") != campaign.campaign_id:
        raise ValueError("campaign state identity drifted")
    if status.get("schema") != CAMPAIGN_STATUS_SCHEMA or status.get("campaign_id") != campaign.campaign_id:
        raise ValueError("campaign status identity drifted")
    _validate_seal(state, "state_sha256", "campaign state")
    _validate_seal(status, "status_sha256", "campaign status")
    plan_authority = {"path": str(campaign.path), "sha256": campaign.sha256}
    policy_authority = {"path": str(campaign.policy_path), "sha256": campaign.policy.sha256}
    throttle_sha256 = sha256_file(THROTTLE_IMPLEMENTATION_PATH)
    if (
        state.get("plan") != plan_authority
        or state.get("plan_sha256") != campaign.sha256
        or status.get("plan") != plan_authority
    ):
        raise ValueError("campaign snapshot plan authority drifted")
    if state.get("policy") != policy_authority or status.get("policy") != policy_authority:
        raise ValueError("campaign snapshot policy authority drifted")
    if (
        state.get("throttle_implementation")
        != {
            "path": str(THROTTLE_IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
            "sha256": throttle_sha256,
        }
        or status.get("throttle_implementation_sha256") != throttle_sha256
    ):
        raise ValueError("campaign snapshot throttle implementation authority drifted")
    supervisor = state.get("supervisor")
    if not isinstance(supervisor, Mapping):
        raise ValueError("campaign supervisor authority is missing")
    supervisor_path = supervisor.get("implementation_path")
    if not isinstance(supervisor_path, str):
        raise ValueError("campaign supervisor implementation path is missing")
    implementation = (REPO_ROOT / supervisor_path).resolve()
    if not implementation.is_relative_to(REPO_ROOT.resolve()) or not implementation.is_file():
        raise ValueError("campaign supervisor implementation path is invalid")
    if supervisor.get("implementation_sha256") != sha256_file(implementation):
        raise ValueError("campaign supervisor implementation authority drifted")
    if supervisor.get("loaded_throttle_sha256") != throttle_sha256:
        raise ValueError("campaign supervisor loaded throttle authority drifted")
    current_step_id = state.get("current_step")
    steps = state.get("steps")
    expected_current = steps.get(current_step_id) if isinstance(steps, dict) else None
    joins = {
        "supervisor": status.get("supervisor"),
        "execution_enabled": status.get("execution_enabled"),
        "status": status.get("state"),
        "steps": status.get("steps"),
        "problems": status.get("problems"),
    }
    for field, value in joins.items():
        if state.get(field) != value:
            raise ValueError(f"campaign state/status join drifted at {field}")
    if status.get("current_step") != expected_current:
        raise ValueError("campaign state/status current-step join drifted")
    if state.get("execution_enabled") is not True:
        raise ValueError("campaign snapshot was not execution-enabled")
    return state, status


def _checkpoint_authority(campaign: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
    current_step = state.get("current_step")
    steps = state.get("steps")
    if not isinstance(current_step, str) or not isinstance(steps, Mapping):
        return None
    step = steps.get(current_step)
    if not isinstance(step, Mapping):
        raise ValueError("current campaign step is missing")
    checkpoint = step.get("last_checkpoint_sha256")
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError("campaign checkpoint authority is invalid")
    launches = state.get("launches")
    if not isinstance(launches, list):
        raise ValueError("campaign launch ledger is missing")
    for launch in reversed(launches):
        if not isinstance(launch, Mapping) or launch.get("step_id") != current_step:
            continue
        run_id = launch.get("run_id")
        if not isinstance(run_id, str):
            continue
        outcome = probe_run_outcome(run_id, campaign.state_root)
        if (
            outcome is not None
            and outcome.status in RUN_RESUMABLE_STATES
            and outcome.checkpoint_sha256 == checkpoint
        ):
            return {
                "step_id": current_step,
                "run_id": run_id,
                "aggregate_sha256": checkpoint,
                "governor_receipt": outcome.receipt_path,
                "governor_receipt_sha256": (
                    sha256_file(Path(outcome.receipt_path)) if outcome.receipt_path is not None else None
                ),
                "governor_status": outcome.status,
            }
    raise ValueError("campaign checkpoint is not joined to a valid governor receipt")


def _raw_registry(campaign: Any) -> dict[str, Any]:
    path = campaign.state_root / "active.json"
    if not path.is_file():
        return {"schema": ACTIVE_REGISTRY_SCHEMA, "updated_at": None, "runs": {}}
    payload = _read_object(path, "raw throttle registry")
    if payload.get("schema") != ACTIVE_REGISTRY_SCHEMA or not isinstance(payload.get("runs"), dict):
        raise ValueError("raw throttle registry schema is invalid")
    return payload


def _router_command(plan: RouterPlan) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/mop_null_safe_campaign.py"),
        "run",
        "--config",
        str(plan.path),
        "--execute",
    )


def _campaign_command(campaign: Any) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/mop_campaign.py"),
        "run",
        "--plan",
        str(campaign.path),
        "--out-dir",
        str(campaign.out_dir),
        "--execute",
    )


def build_recovery_plan(
    plan: RouterPlan,
    *,
    process_probe: ProcessProbe = observe_process,
    boot_time: float | None = None,
) -> dict[str, Any]:
    """Return a sealed read-only decision; never launch or mutate controller state."""

    problems: list[str] = []
    facts: dict[str, Any] = {}
    disposition = "refused"
    recovery_command = (
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/plan_campaign_recovery.py"),
        "resume",
        "--config",
        str(plan.path),
        "--execute",
    )
    try:
        live_validation = validate_live_router_plan(plan)
        if live_validation.get("valid") is not True:
            raise ValueError("live router validation did not pass")
        router_state, router_status = _validate_router_snapshots(plan)
        current_boot = float(psutil.boot_time() if boot_time is None else boot_time)
        router_process = _identity_report(
            router_state.get("supervisor"),
            expected_command=_router_command(plan),
            process_probe=process_probe,
            boot_time=current_boot,
        )
        router_facts = {
            "id": plan.router_id,
            "plan": {"path": str(plan.path), "sha256": plan.sha256},
            "state": router_state.get("status"),
            "process": router_process,
            "implementation_sha256": sha256_file(
                REPO_ROOT / "src/mop/studio/null_safe_campaign_router.py"
            ),
        }
        router_state_name = str(router_status.get("state"))
        if router_state_name in ROUTER_CLEAN_TERMINAL:
            facts = {"router": router_facts}
            disposition = "already-terminal"
        else:
            current_stage = router_state.get("current_stage")
            stage = next(
                (candidate for candidate in plan.stages if candidate.stage_id == current_stage),
                None,
            )
            if stage is None:
                raise ValueError("router current stage is not declared by the router plan")
            campaign = load_campaign_plan(stage.plan_path)
            campaign_state, campaign_status = _validate_campaign_snapshots(campaign)
            campaign_process = _identity_report(
                campaign_state.get("supervisor"),
                expected_command=_campaign_command(campaign),
                process_probe=process_probe,
                boot_time=current_boot,
            )
            registry = _raw_registry(campaign)
            checkpoint = _checkpoint_authority(campaign, campaign_state)
            facts = {
                "router": router_facts,
                "campaign": {
                    "id": campaign.campaign_id,
                    "plan": {"path": str(campaign.path), "sha256": campaign.sha256},
                    "policy_sha256": campaign.policy.sha256,
                    "state": campaign_state.get("status"),
                    "process": campaign_process,
                    "checkpoint": checkpoint,
                },
                "raw_active_registry": {
                    "path": str(campaign.state_root / "active.json"),
                    "schema": registry.get("schema"),
                    "updated_at": registry.get("updated_at"),
                    "run_ids": sorted(str(key) for key in registry["runs"]),
                    "empty": not registry["runs"],
                },
                "controls_absent": {
                    "router": not (plan.out_dir / ROUTER_CONTROL_FILE).exists(),
                    "campaign": not (campaign.out_dir / CAMPAIGN_CONTROL_FILE).exists(),
                },
            }
            campaign_state_name = str(campaign_status.get("state"))
            if router_state_name in ROUTER_HOLD_STATES:
                problems.append(f"router is in hold state {router_state_name}")
            elif campaign_state_name in CAMPAIGN_TERMINAL_STATES and campaign_state_name != "complete":
                problems.append(f"campaign is in terminal state {campaign_state_name}")
            elif router_process["safe"] is not True or campaign_process["safe"] is not True:
                problems.append(
                    "a recorded controller PID is live under a different or unobservable identity"
                )
            elif router_process["status"] == "exact-live":
                disposition = "already-running"
            elif registry["runs"]:
                disposition = "deferred"
                problems.append("raw throttle registry is not exactly empty")
            elif not all(facts["controls_absent"].values()):
                disposition = "deferred"
                problems.append("a sealed drain request exists")
            else:
                disposition = "ready"
    except Exception as exc:  # every observation or validation failure must fail closed
        problems.append(f"{type(exc).__name__}: {exc}")
    core = {
        "schema": RECOVERY_SCHEMA,
        "disposition": disposition,
        "safe_to_resume": disposition == "ready",
        "read_only_plan": True,
        "signals_sent": False,
        "facts": facts,
        "problems": problems,
        "resume_command": list(recovery_command),
    }
    return {**core, "recovery_sha256": canonical_sha256(core)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="publish a read-only recovery decision")
    plan.add_argument("--config", type=Path, required=True)
    resume = subparsers.add_parser("resume", help="recheck and start the detached router if safe")
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--execute", action="store_true")
    resume.add_argument("--no-caffeinate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        plan = load_router_plan(arguments.config)
        decision = build_recovery_plan(plan)
        if arguments.command == "plan":
            print(json.dumps(decision, indent=2, sort_keys=True))
            return 0
        if not arguments.execute:
            print(json.dumps({"error": "resume requires explicit --execute"}, indent=2), file=sys.stderr)
            return 2
        if decision.get("disposition") != "ready" or decision.get("safe_to_resume") is not True:
            print(json.dumps(decision, indent=2, sort_keys=True))
            return 0
        launch = start_router_detached(
            plan,
            execute=True,
            use_caffeinate=not arguments.no_caffeinate,
        )
        print(json.dumps({"decision": decision, "launch": launch}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # watchdog invocations must fail closed without a traceback
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


__all__ = [
    "ACTIVE_REGISTRY_SCHEMA",
    "ProcessObservation",
    "RECOVERY_SCHEMA",
    "build_recovery_plan",
    "canonical_sha256",
    "main",
    "observe_process",
]
