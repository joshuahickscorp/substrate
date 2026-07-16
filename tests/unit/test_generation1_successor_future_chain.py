from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from mop.studio import generation1_successor_chain_v4 as v4
from mop.studio import generation1_successor_extension_chain as extension_chain
from mop.studio import generation1_successor_future_chain as future
from mop.studio.generation1_supervisor import canonical_sha256

NOW = "2026-07-16T12:00:00+00:00"


def _sealed(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "status_sha256": canonical_sha256(core)}


def _adopter_status(
    state: str,
    *,
    pid: int,
    create_time: float | None = None,
) -> dict[str, Any]:
    capsules = {spec.stage_id: v4.v3._empty_capsule() for spec in v4.legacy_specs(v4.REPO_ROOT)}
    capsules["successor_horizon"] = v4.v3._empty_capsule()
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
        first = next(iter(capsules.values()))
        first.update(
            {
                "status": "failure_hold",
                "finished_at": NOW,
                "last_problem": "legacy prerequisite failed",
            }
        )
    completed = sum(row["status"] == "complete" for row in capsules.values())
    implementation = v4._implementation_authority(v4.REPO_ROOT)
    core: dict[str, Any] = {
        "schema": v4.STATUS_SCHEMA,
        "program_id": v4.CHAIN_ID,
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
        "horizon_program": v4._horizon_authority(
            v4.DEFAULT_HORIZON_PROGRAM,
            v4.REPO_ROOT,
        ),
        "capsules": capsules,
        "counts": {
            "complete": completed,
            "total": len(capsules),
            "remaining": len(capsules) - completed,
        },
        "finished_at": NOW if complete or state in future.UNSAFE_TERMINAL_STATES else None,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "supersedes": "generation1-successor-evidence-chain-v3",
    }
    return _sealed(core)


def _extension_status(
    state: str,
    *,
    pid: int,
    create_time: float | None = None,
) -> dict[str, Any]:
    capsules = {
        "predecessor_chain_v4": extension_chain._empty_row(),
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
        capsules["predecessor_chain_v4"].update(
            {
                "status": "failure_hold",
                "finished_at": NOW,
                "last_problem": "predecessor failed",
            }
        )
    completed = sum(row["status"] == "complete" for row in capsules.values())
    implementation = extension_chain._implementation_authority(extension_chain.REPO_ROOT)
    core: dict[str, Any] = {
        "schema": extension_chain.STATUS_SCHEMA,
        "program_id": extension_chain.PROGRAM_ID,
        "created_at": NOW,
        "updated_at": NOW,
        "finished_at": NOW if complete or state in future.UNSAFE_TERMINAL_STATES else None,
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


def _reseal(status: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    core = dict(status)
    core.pop("status_sha256", None)
    core.update(updates)
    return _sealed(core)


def test_future_chain_requires_explicit_execution() -> None:
    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match="explicit --execute",
    ):
        future.start_future_chain(execute=False)


def test_future_chain_starts_v4_before_the_observation_only_extension() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def adopter(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("adopter", kwargs))
        return {"status": _adopter_status("waiting_legacy", pid=41), "launched_pid": 41}

    def extension(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("extension", kwargs))
        return {
            "status": _extension_status("waiting_predecessor", pid=42),
            "launched_pid": 42,
        }

    result = future.start_future_chain(
        execute=True,
        use_caffeinate=False,
        adopter_starter=adopter,
        extension_starter=extension,
    )

    assert calls == [
        ("adopter", {"execute": True, "use_caffeinate": False}),
        ("extension", {"execute": True, "use_caffeinate": False}),
    ]
    assert result["start_order"] == [
        "generation1-successor-evidence-chain-v4",
        "generation1-successor-extension-chain-v1",
    ]
    assert result["signals_allowed"] is False
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False
    assert result["acknowledgement_attempts"] == {"adopter": 1, "extension": 1}
    assert result["all_ok"] is True


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_terminal_v4_state_prevents_extension_start(state: str) -> None:
    extension_calls: list[dict[str, Any]] = []

    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match=f"unsafe state {state}",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": _adopter_status(state, pid=41)},
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    assert extension_calls == []


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_terminal_extension_state_refuses_success_receipt(state: str) -> None:
    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match=f"extension is terminal in unsafe state {state}",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": _adopter_status("waiting_horizon", pid=41)},
            extension_starter=lambda **_kwargs: {"status": _extension_status(state, pid=42)},
        )


@pytest.mark.parametrize(
    "receipt",
    (
        {},
        {"status": None},
        {"status": {"state": "waiting_legacy"}},
    ),
)
def test_missing_or_unsealed_v4_acknowledgement_prevents_extension_start(
    receipt: Mapping[str, Any],
) -> None:
    extension_calls: list[dict[str, Any]] = []

    with pytest.raises(future.SuccessorFutureChainRefused):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: receipt,
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    assert extension_calls == []


def test_exact_component_validators_reject_self_sealed_fabrications() -> None:
    extension_calls: list[dict[str, Any]] = []

    minimal_row = _adopter_status("waiting_legacy", pid=41)
    capsules = dict(minimal_row["capsules"])
    capsule_id = next(iter(capsules))
    capsules[capsule_id] = {"status": "pending"}
    minimal_row = _reseal(minimal_row, capsules=capsules)

    extra_field_core = dict(_adopter_status("waiting_legacy", pid=41))
    extra_field_core.pop("status_sha256")
    extra_field_core["fabricated"] = True
    extra_field = _sealed(extra_field_core)

    complete_with_pending_capsules = _reseal(
        _adopter_status("waiting_legacy", pid=41),
        state="complete",
        finished_at=NOW,
    )
    all_complete_but_waiting = _reseal(
        _adopter_status("complete", pid=41),
        state="waiting_legacy",
        finished_at=None,
    )
    pending_horizon_with_launch_evidence = _adopter_status(
        "waiting_legacy",
        pid=41,
    )
    adopter_capsules = {
        key: dict(value) for key, value in pending_horizon_with_launch_evidence["capsules"].items()
    }
    horizon = dict(adopter_capsules["successor_horizon"])
    horizon.update(
        {
            "attempts": 1,
            "process": {"pid": 51, "create_time": 51.5},
            "launch_requested_at": NOW,
            "launched_pid": 51,
        }
    )
    adopter_capsules["successor_horizon"] = horizon
    pending_horizon_with_launch_evidence = _reseal(
        pending_horizon_with_launch_evidence,
        capsules=adopter_capsules,
    )
    horizon_adoption_before_legacy = _adopter_status("waiting_legacy", pid=41)
    adopter_capsules = {key: dict(value) for key, value in horizon_adoption_before_legacy["capsules"].items()}
    horizon = dict(adopter_capsules["successor_horizon"])
    horizon.update(
        {
            "status": "adoption_wait",
            "last_problem": "fabricated early horizon",
        }
    )
    adopter_capsules["successor_horizon"] = horizon
    horizon_adoption_before_legacy = _reseal(
        horizon_adoption_before_legacy,
        state="adoption_wait",
        capsules=adopter_capsules,
    )
    legacy_adoption_after_horizon_active = _adopter_status(
        "waiting_horizon",
        pid=41,
    )
    adopter_capsules = {
        key: dict(value) for key, value in legacy_adoption_after_horizon_active["capsules"].items()
    }
    legacy_id = next(key for key in adopter_capsules if key != "successor_horizon")
    legacy = dict(adopter_capsules[legacy_id])
    legacy.update(
        {
            "status": "adoption_wait",
            "returncode": None,
            "finished_at": None,
            "artifacts": [],
            "last_problem": "fabricated regressed legacy",
        }
    )
    adopter_capsules[legacy_id] = legacy
    horizon = dict(adopter_capsules["successor_horizon"])
    horizon.update(
        {
            "status": "running",
            "process": {"pid": 52, "create_time": 52.5},
            "last_problem": "fabricated active horizon",
        }
    )
    adopter_capsules["successor_horizon"] = horizon
    completed = sum(row["status"] == "complete" for row in adopter_capsules.values())
    legacy_adoption_after_horizon_active = _reseal(
        legacy_adoption_after_horizon_active,
        state="adoption_wait",
        capsules=adopter_capsules,
        counts={
            "complete": completed,
            "total": len(adopter_capsules),
            "remaining": len(adopter_capsules) - completed,
        },
    )
    impossible_adopted_horizon = _adopter_status("waiting_horizon", pid=41)
    adopter_capsules = {key: dict(value) for key, value in impossible_adopted_horizon["capsules"].items()}
    horizon = dict(adopter_capsules["successor_horizon"])
    horizon["status"] = "adopted"
    adopter_capsules["successor_horizon"] = horizon
    impossible_adopted_horizon = _reseal(
        impossible_adopted_horizon,
        capsules=adopter_capsules,
    )

    partial_authority = _reseal(
        _extension_status("waiting_predecessor", pid=42),
        parent_implementation={},
    )
    dirty_complete = _reseal(
        _extension_status("complete", pid=42),
        execution_enabled=False,
        problems=["fabricated completion"],
    )

    garbage_row = _extension_status("waiting_predecessor", pid=42)
    extension_capsules = dict(garbage_row["capsules"])
    target = dict(extension_capsules["successor_horizon_v2"])
    target["status"] = "garbage"
    extension_capsules["successor_horizon_v2"] = target
    garbage_row = _reseal(garbage_row, capsules=extension_capsules)
    extension_all_complete_but_waiting = _reseal(
        _extension_status("complete", pid=42),
        state="waiting_predecessor",
        finished_at=None,
    )
    pending_target_with_launch_evidence = _extension_status(
        "waiting_predecessor",
        pid=42,
    )
    extension_capsules = {
        key: dict(value) for key, value in pending_target_with_launch_evidence["capsules"].items()
    }
    target = dict(extension_capsules["successor_horizon_v2"])
    target.update(
        {
            "attempts": 1,
            "process": {"pid": 61, "create_time": 61.5},
            "launch_requested_at": NOW,
            "launched_pid": 61,
        }
    )
    extension_capsules["successor_horizon_v2"] = target
    pending_target_with_launch_evidence = _reseal(
        pending_target_with_launch_evidence,
        capsules=extension_capsules,
    )

    for forged in (
        minimal_row,
        extra_field,
        complete_with_pending_capsules,
        all_complete_but_waiting,
        pending_horizon_with_launch_evidence,
        horizon_adoption_before_legacy,
        legacy_adoption_after_horizon_active,
        impossible_adopted_horizon,
    ):
        with pytest.raises(future.SuccessorFutureChainRefused, match="not authoritative"):
            future.start_future_chain(
                execute=True,
                adopter_starter=lambda forged=forged, **_kwargs: {"status": forged},
                extension_starter=lambda **kwargs: extension_calls.append(kwargs),
            )

    for forged in (
        partial_authority,
        garbage_row,
        dirty_complete,
        extension_all_complete_but_waiting,
        pending_target_with_launch_evidence,
    ):
        with pytest.raises(future.SuccessorFutureChainRefused, match="not authoritative"):
            future.start_future_chain(
                execute=True,
                adopter_starter=lambda **_kwargs: {"status": _adopter_status("waiting_horizon", pid=41)},
                extension_starter=lambda forged=forged, **_kwargs: {"status": forged},
            )

    assert extension_calls == []


def test_complete_v4_acknowledgement_requires_matching_durable_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _adopter_status("complete", pid=41)
    extension_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        future,
        "read_validated_complete_chain_status",
        lambda *, root: _adopter_status("complete", pid=99),
    )

    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match="disagrees with durable replay",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": status},
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    assert extension_calls == []


def test_matching_complete_v4_durable_replay_allows_extension_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _adopter_status("complete", pid=41)
    monkeypatch.setattr(
        future,
        "read_validated_complete_chain_status",
        lambda *, root: status,
    )

    result = future.start_future_chain(
        execute=True,
        adopter_starter=lambda **_kwargs: {"status": status},
        extension_starter=lambda **_kwargs: {"status": _extension_status("waiting_predecessor", pid=42)},
    )

    assert result["all_ok"] is True


def test_complete_extension_acknowledgement_requires_matching_durable_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _extension_status("complete", pid=42)
    monkeypatch.setattr(
        future,
        "read_validated_complete_extension_status",
        lambda *, root: _extension_status("complete", pid=99),
    )

    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match="extension complete acknowledgement disagrees with durable replay",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": _adopter_status("waiting_horizon", pid=41)},
            extension_starter=lambda **_kwargs: {"status": status},
        )


def test_matching_complete_extension_durable_replay_allows_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _extension_status("complete", pid=42)
    monkeypatch.setattr(
        future,
        "read_validated_complete_extension_status",
        lambda *, root: status,
    )

    result = future.start_future_chain(
        execute=True,
        adopter_starter=lambda **_kwargs: {"status": _adopter_status("waiting_horizon", pid=41)},
        extension_starter=lambda **_kwargs: {"status": status},
    )

    assert result["all_ok"] is True


def test_unknown_or_safety_drifted_v4_state_prevents_extension_start() -> None:
    extension_calls: list[dict[str, Any]] = []
    unknown = _reseal(_adopter_status("waiting_legacy", pid=41), state="mystery")
    with pytest.raises(future.SuccessorFutureChainRefused, match="not authoritative"):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": unknown},
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    drifted = _reseal(
        _adopter_status("waiting_legacy", pid=41),
        activation_allowed=True,
    )
    with pytest.raises(future.SuccessorFutureChainRefused, match="not authoritative"):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {"status": drifted},
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
        )

    assert extension_calls == []


def test_observed_v4_process_is_retried_until_matching_status_acknowledges() -> None:
    adopter_calls = 0
    extension_calls = 0

    def adopter(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal adopter_calls
        adopter_calls += 1
        if adopter_calls == 1:
            return {
                "already_running": True,
                "status": None,
                "observed_process": {"pid": 41, "create_time": 41.5},
            }
        return {
            "already_running": True,
            "status": _adopter_status("waiting_legacy", pid=41),
        }

    def extension(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal extension_calls
        extension_calls += 1
        return {
            "status": _extension_status("waiting_predecessor", pid=42),
        }

    result = future.start_future_chain(
        execute=True,
        adopter_starter=adopter,
        extension_starter=extension,
        ack_attempts=2,
        ack_interval_seconds=0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert adopter_calls == 2
    assert extension_calls == 1
    assert result["acknowledgement_attempts"] == {"adopter": 2, "extension": 1}


def test_observed_v4_process_retries_transient_invalid_matching_status() -> None:
    adopter_calls = 0
    extension_calls = 0

    def adopter(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal adopter_calls
        adopter_calls += 1
        status = _adopter_status("waiting_legacy", pid=41)
        if adopter_calls == 1:
            status = _reseal(status, activation_allowed=True)
        return {
            "already_running": True,
            "status": status,
            "observed_process": {"pid": 41, "create_time": 41.5},
        }

    def extension(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal extension_calls
        extension_calls += 1
        return {
            "status": _extension_status("waiting_predecessor", pid=42),
        }

    result = future.start_future_chain(
        execute=True,
        adopter_starter=adopter,
        extension_starter=extension,
        ack_attempts=2,
        ack_interval_seconds=0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert adopter_calls == 2
    assert extension_calls == 1
    assert result["acknowledgement_attempts"] == {"adopter": 2, "extension": 1}


def test_unacknowledged_observed_v4_process_times_out_without_extension() -> None:
    extension_calls: list[dict[str, Any]] = []

    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match="acknowledgement remained pending",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=lambda **_kwargs: {
                "already_running": True,
                "status": None,
                "observed_process": {"pid": 41, "create_time": 41.5},
            },
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
            ack_attempts=2,
            ack_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )

    assert extension_calls == []


def test_observed_v4_process_identity_remains_pinned_across_retries() -> None:
    adopter_calls = 0
    extension_calls: list[dict[str, Any]] = []

    def adopter(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal adopter_calls
        adopter_calls += 1
        if adopter_calls == 1:
            return {
                "already_running": True,
                "status": None,
                "observed_process": {"pid": 41, "create_time": 41.5},
            }
        return {
            "already_running": True,
            "status": _adopter_status("waiting_legacy", pid=99),
        }

    with pytest.raises(
        future.SuccessorFutureChainRefused,
        match="acknowledgement remained pending",
    ):
        future.start_future_chain(
            execute=True,
            adopter_starter=adopter,
            extension_starter=lambda **kwargs: extension_calls.append(kwargs),
            ack_attempts=2,
            ack_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )

    assert adopter_calls == 2
    assert extension_calls == []


def test_future_chain_parser_exposes_one_idempotent_start_command() -> None:
    arguments = future.build_parser().parse_args(["start", "--execute", "--no-caffeinate"])

    assert (arguments.command, arguments.execute, arguments.no_caffeinate) == (
        "start",
        True,
        True,
    )
