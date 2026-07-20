from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mop.studio import generation1_full_generations_extension_chain as extension_chain
from mop.studio import generation1_successor_long_chain_v3 as long_chain


def test_launcher_targets_extension_parent_by_uniform_label() -> None:
    assert extension_chain.PARENT_LABEL == "mop:fullgen:extension"


def _v2_receipt() -> dict[str, Any]:
    return {
        "schema": long_chain.LONG_CHAIN_V2_RECEIPT_SCHEMA,
        "program_id": long_chain.LONG_CHAIN_V2_PROGRAM_ID,
        "start_order": [
            "generation1-successor-future-chain-v2",
            "generation1-categorized-batch-extension-chain-v1",
        ],
        "target_order": [
            "generation1-successor-evidence-chain-v5",
            "generation1-successor-horizon-v2",
            "generation1-successor-categorized-batch-wave-v1",
        ],
        "signals_allowed": False,
        "activation_allowed": False,
        "scientific_promotion": False,
        "all_ok": True,
    }


def test_long_chain_requires_explicit_execution(tmp_path: Path) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match="explicit --execute",
    ):
        long_chain.start_long_chain(execute=False, root=tmp_path)


def test_long_chain_starts_v2_chain_before_full_generations_waiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    status = {
        "state": "waiting_predecessor",
        "supervisor": {"pid": 43, "create_time": 43.5},
    }
    monkeypatch.setattr(
        long_chain,
        "validate_full_generations_extension_status",
        lambda value: str(value["state"]),
    )

    def v2_chain(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("long_chain_v2", kwargs))
        return _v2_receipt()

    def full_generations(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("full_generations", kwargs))
        return {
            "status": status,
            "launched_pid": 42,
            "observed_process": {"pid": 43, "create_time": 43.5},
        }

    result = long_chain.start_long_chain(
        execute=True,
        use_caffeinate=False,
        root=tmp_path,
        long_chain_v2_starter=v2_chain,
        full_generations_starter=full_generations,
    )

    assert calls == [
        ("long_chain_v2", {"execute": True, "use_caffeinate": False}),
        ("full_generations", {"execute": True, "use_caffeinate": False}),
    ]
    assert result["start_order"] == [
        "generation1-successor-future-chain-v2",
        "generation1-categorized-batch-extension-chain-v1",
        "generation1-full-generations-extension-chain-v1",
    ]
    assert result["target_order"] == [
        "generation1-successor-evidence-chain-v5",
        "generation1-successor-horizon-v2",
        "generation1-successor-categorized-batch-wave-v1",
        "generation1-full-generations-wave-v1",
    ]
    assert result["full_generations_state"] == "waiting_predecessor"
    assert result["full_generations_extension"]["observed_process"]["pid"] == 43
    assert result["signals_allowed"] is False
    assert result["all_ok"] is True


def test_bad_v2_receipt_prevents_full_generations_waiter(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match="no clean success receipt",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            long_chain_v2_starter=lambda **_kwargs: {**_v2_receipt(), "all_ok": False},
            full_generations_starter=lambda **_kwargs: calls.append("full_generations") or {},
        )
    assert calls == []


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_full_generations_state_refuses_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr(
        long_chain,
        "validate_full_generations_extension_status",
        lambda _value: state,
    )
    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match=f"unsafe state {state}",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            long_chain_v2_starter=lambda **_kwargs: _v2_receipt(),
            full_generations_starter=lambda **_kwargs: {"status": {"state": state}},
        )


def test_observed_waiter_retries_until_same_process_is_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def validate(value: Mapping[str, Any]) -> str:
        return str(value["state"])

    monkeypatch.setattr(
        long_chain,
        "validate_full_generations_extension_status",
        validate,
    )

    def full_generations(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "status": None,
                "observed_process": {"pid": 42, "create_time": 42.5},
            }
        return {
            "status": {
                "state": "waiting_predecessor",
                "supervisor": {"pid": 42, "create_time": 42.5},
            },
            "observed_process": {"pid": 42, "create_time": 42.5},
        }

    result = long_chain.start_long_chain(
        execute=True,
        root=tmp_path,
        long_chain_v2_starter=lambda **_kwargs: _v2_receipt(),
        full_generations_starter=full_generations,
        ack_interval_seconds=0.25,
        sleep_fn=sleeps.append,
    )

    assert attempts == 2
    assert sleeps == [0.25]
    assert result["acknowledgement_attempts"] == {"full_generations_extension": 2}


def test_observed_waiter_identity_cannot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr(
        long_chain,
        "validate_full_generations_extension_status",
        lambda value: str(value["state"]),
    )

    def full_generations(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        return {
            "status": None,
            "observed_process": {
                "pid": 42 + attempts,
                "create_time": 42.5 + attempts,
            },
        }

    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match="changed before acknowledgement",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            long_chain_v2_starter=lambda **_kwargs: _v2_receipt(),
            full_generations_starter=full_generations,
            ack_attempts=2,
            ack_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )


def test_invalid_observed_waiter_identity_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match="inexact observed process",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            long_chain_v2_starter=lambda **_kwargs: _v2_receipt(),
            full_generations_starter=lambda **_kwargs: {
                "status": None,
                "observed_process": {"pid": True, "create_time": float("nan")},
            },
        )


def test_fresh_waiter_launch_requires_visible_parent_binding(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainV3Refused,
        match="returned no visible parent",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            long_chain_v2_starter=lambda **_kwargs: _v2_receipt(),
            full_generations_starter=lambda **_kwargs: {
                "launched_pid": 42,
                "status": {
                    "state": "waiting_predecessor",
                    "supervisor": {"pid": 43, "create_time": 43.5},
                },
            },
        )
