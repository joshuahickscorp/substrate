from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from mop.studio import generation1_successor_chain_v6 as v6
from mop.studio import generation1_successor_extension_chain_v3 as extension_chain
from mop.studio import generation1_successor_recovery_launcher as recovery
from mop.studio.generation1_supervisor import canonical_sha256

NOW = "2026-07-17T04:00:00+00:00"


def _sealed(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "status_sha256": canonical_sha256(core)}


def _adopter_status(state: str, *, pid: int, create_time: float | None = None) -> dict[str, Any]:
    capsules = {spec.stage_id: v6.v3._empty_capsule() for spec in v6.legacy_specs(v6.REPO_ROOT)}
    capsules["successor_horizon"] = v6.v3._empty_capsule()
    complete = state == "complete"
    if complete:
        for capsule_id, row in capsules.items():
            row.update(
                {
                    "status": "complete",
                    "returncode": 0,
                    "finished_at": NOW,
                    "artifacts": [{"path": f"proof/{capsule_id}.json"}],
                }
            )
    elif state == "waiting_horizon":
        for capsule_id, row in capsules.items():
            if capsule_id == "successor_horizon":
                continue
            row.update(
                {
                    "status": "complete",
                    "returncode": 0,
                    "finished_at": NOW,
                    "artifacts": [{"path": f"proof/{capsule_id}.json"}],
                }
            )
    elif state == "failure_hold":
        next(iter(capsules.values())).update(
            {"status": "failure_hold", "finished_at": NOW, "last_problem": "legacy prerequisite failed"}
        )
    completed = sum(row["status"] == "complete" for row in capsules.values())
    implementation = v6._implementation_authority(v6.REPO_ROOT)
    core: dict[str, Any] = {
        "schema": v6.STATUS_SCHEMA,
        "program_id": v6.CHAIN_ID,
        "created_at": NOW,
        "updated_at": NOW,
        "supervisor": {
            "pid": pid,
            "create_time": float(pid) + 0.5 if create_time is None else create_time,
            "implementation_path": implementation["path"],
            "implementation_sha256": implementation["sha256"],
        },
        "execution_enabled": True,
        "state": state,
        "horizon_program": v6._horizon_authority(v6.DEFAULT_HORIZON_PROGRAM, v6.REPO_ROOT),
        "capsules": capsules,
        "counts": {
            "complete": completed,
            "total": len(capsules),
            "remaining": len(capsules) - completed,
        },
        "finished_at": NOW if complete or state in recovery.UNSAFE_TERMINAL_STATES else None,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "supersedes": "generation1-successor-evidence-chain-v5",
    }
    return _sealed(core)


def _extension_status(state: str, *, pid: int, create_time: float | None = None) -> dict[str, Any]:
    capsules = {
        "predecessor_chain_v6": extension_chain._empty_row(),
        "successor_horizon_v2": extension_chain._empty_row(),
    }
    complete = state == "complete"
    if complete:
        for capsule_id, row in capsules.items():
            row.update(
                {
                    "status": "complete",
                    "returncode": 0,
                    "finished_at": NOW,
                    "artifacts": [{"path": f"proof/{capsule_id}.json"}],
                }
            )
    elif state == "failure_hold":
        capsules["predecessor_chain_v6"].update(
            {"status": "failure_hold", "finished_at": NOW, "last_problem": "predecessor failed"}
        )
    completed = sum(row["status"] == "complete" for row in capsules.values())
    implementation = extension_chain._implementation_authority(extension_chain.REPO_ROOT)
    core: dict[str, Any] = {
        "schema": extension_chain.STATUS_SCHEMA,
        "program_id": extension_chain.PROGRAM_ID,
        "created_at": NOW,
        "updated_at": NOW,
        "finished_at": NOW if complete or state in recovery.UNSAFE_TERMINAL_STATES else None,
        "execution_enabled": True,
        "state": state,
        "supervisor": {
            "pid": pid,
            "create_time": float(pid) + 0.5 if create_time is None else create_time,
            "implementation_path": implementation["path"],
            "implementation_sha256": implementation["sha256"],
        },
        "parent_implementation": implementation,
        "predecessor": extension_chain._predecessor_observation_binding(
            extension_chain.DEFAULT_PREDECESSOR_STATUS,
            extension_chain.REPO_ROOT,
        ),
        "target_program": extension_chain._target_program_authority(
            extension_chain.DEFAULT_TARGET_PROGRAM,
            extension_chain.REPO_ROOT,
        ),
        "capsules": capsules,
        "counts": {
            "complete": completed,
            "total": len(capsules),
            "remaining": len(capsules) - completed,
        },
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core)


def test_recovery_requires_explicit_execution() -> None:
    with pytest.raises(recovery.SuccessorChainRecoveryRefused, match="explicit --execute"):
        recovery.start_recovery(execute=False)


def test_recovery_starts_v6_before_the_observation_only_extension_v3() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def adopter(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("adopter", kwargs))
        return {"status": _adopter_status("waiting_legacy", pid=41), "launched_pid": 41}

    def extension(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("extension", kwargs))
        return {"status": _extension_status("waiting_predecessor", pid=42), "launched_pid": 42}

    result = recovery.start_recovery(
        execute=True,
        use_caffeinate=False,
        adopter_starter=adopter,
        extension_starter=extension,
    )

    assert calls == [
        ("adopter", {"execute": True, "use_caffeinate": False}),
        ("extension", {"execute": True, "use_caffeinate": False}),
    ]
    assert result["schema"] == "mop-generation1-successor-chain-recovery-start/v1"
    assert result["program_id"] == "generation1-successor-chain-recovery-v1"
    assert result["start_order"] == [
        "generation1-successor-evidence-chain-v6",
        "generation1-successor-extension-chain-v3",
    ]
    assert result["signals_allowed"] is False
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False
    assert result["acknowledgement_attempts"] == {"adopter": 1, "extension": 1}
    assert result["all_ok"] is True


def test_recovery_is_idempotent_when_both_parents_already_running() -> None:
    def adopter(**_kwargs: Any) -> Mapping[str, Any]:
        return {"already_running": True, "status": _adopter_status("waiting_horizon", pid=41)}

    def extension(**_kwargs: Any) -> Mapping[str, Any]:
        return {"already_running": True, "status": _extension_status("waiting_predecessor", pid=42)}

    result = recovery.start_recovery(
        execute=True,
        adopter_starter=adopter,
        extension_starter=extension,
    )

    assert result["all_ok"] is True
    assert "launched_pid" not in result["adopter"]
    assert "launched_pid" not in result["extension"]


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_terminal_v6_state_prevents_extension_start(state: str) -> None:
    extension_calls: list[dict[str, Any]] = []

    with pytest.raises(recovery.SuccessorChainRecoveryRefused, match=f"unsafe state {state}"):
        recovery.start_recovery(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": _adopter_status(state, pid=41)},
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    assert extension_calls == []


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_terminal_extension_v3_state_refuses_success_receipt(state: str) -> None:
    with pytest.raises(recovery.SuccessorChainRecoveryRefused, match=f"unsafe state {state}"):
        recovery.start_recovery(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": _adopter_status("waiting_legacy", pid=41)},
            extension_starter=lambda **_kwargs: {"status": _extension_status(state, pid=42)},
        )


def test_recovery_rejects_self_sealed_adopter_supersedes_forgery() -> None:
    forged = _adopter_status("waiting_legacy", pid=41)
    core = dict(forged)
    core.pop("status_sha256")
    core["supersedes"] = "generation1-successor-evidence-chain-v4"
    resealed = _sealed(core)

    with pytest.raises(recovery.SuccessorChainRecoveryRefused, match="not authoritative"):
        recovery.start_recovery(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": resealed},
            extension_starter=lambda **_kwargs: {"status": _extension_status("waiting_predecessor", pid=42)},
        )


def test_recovery_parser_exposes_one_idempotent_start_command() -> None:
    parser = recovery.build_parser()
    arguments = parser.parse_args(["start", "--execute"])

    assert arguments.command == "start"
    assert arguments.execute is True
    assert arguments.no_caffeinate is False
