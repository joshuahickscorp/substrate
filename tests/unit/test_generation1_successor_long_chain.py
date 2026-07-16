from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mop.studio import generation1_successor_long_chain as long_chain


def _future_receipt() -> dict[str, Any]:
    return {
        "schema": long_chain.FUTURE_CHAIN_RECEIPT_SCHEMA,
        "program_id": long_chain.FUTURE_CHAIN_PROGRAM_ID,
        "signals_allowed": False,
        "activation_allowed": False,
        "scientific_promotion": False,
        "all_ok": True,
    }


def test_long_chain_requires_explicit_execution(tmp_path: Path) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainRefused,
        match="explicit --execute",
    ):
        long_chain.start_long_chain(execute=False, root=tmp_path)


def test_long_chain_starts_future_before_categorized_waiter(
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
        "validate_categorized_extension_status",
        lambda value: str(value["state"]),
    )

    def future(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("future", kwargs))
        return _future_receipt()

    def categorized(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("categorized", kwargs))
        return {
            "status": status,
            "launched_pid": 42,
            "observed_process": {"pid": 43, "create_time": 43.5},
        }

    result = long_chain.start_long_chain(
        execute=True,
        use_caffeinate=False,
        root=tmp_path,
        future_starter=future,
        categorized_starter=categorized,
    )

    assert calls == [
        ("future", {"execute": True, "use_caffeinate": False}),
        ("categorized", {"execute": True, "use_caffeinate": False}),
    ]
    assert result["target_order"] == [
        "generation1-successor-evidence-chain-v4",
        "generation1-successor-horizon-v2",
        "generation1-successor-categorized-batch-wave-v1",
    ]
    assert result["categorized_state"] == "waiting_predecessor"
    assert result["categorized_extension"]["observed_process"]["pid"] == 43
    assert result["signals_allowed"] is False
    assert result["all_ok"] is True


def test_bad_future_receipt_prevents_categorized_waiter(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    with pytest.raises(
        long_chain.SuccessorLongChainRefused,
        match="no clean success receipt",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            future_starter=lambda **_kwargs: {**_future_receipt(), "all_ok": False},
            categorized_starter=lambda **_kwargs: calls.append("categorized") or {},
        )
    assert calls == []


@pytest.mark.parametrize("state", ("failure_hold", "integrity_hold", "drained"))
def test_unsafe_categorized_state_refuses_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr(
        long_chain,
        "validate_categorized_extension_status",
        lambda _value: state,
    )
    with pytest.raises(
        long_chain.SuccessorLongChainRefused,
        match=f"unsafe state {state}",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            future_starter=lambda **_kwargs: _future_receipt(),
            categorized_starter=lambda **_kwargs: {"status": {"state": state}},
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
        "validate_categorized_extension_status",
        validate,
    )

    def categorized(**_kwargs: Any) -> Mapping[str, Any]:
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
        future_starter=lambda **_kwargs: _future_receipt(),
        categorized_starter=categorized,
        ack_interval_seconds=0.25,
        sleep_fn=sleeps.append,
    )

    assert attempts == 2
    assert sleeps == [0.25]
    assert result["acknowledgement_attempts"] == {"categorized_extension": 2}


def test_observed_waiter_identity_cannot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr(
        long_chain,
        "validate_categorized_extension_status",
        lambda value: str(value["state"]),
    )

    def categorized(**_kwargs: Any) -> Mapping[str, Any]:
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
        long_chain.SuccessorLongChainRefused,
        match="changed before acknowledgement",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            future_starter=lambda **_kwargs: _future_receipt(),
            categorized_starter=categorized,
            ack_attempts=2,
            ack_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )


def test_invalid_observed_waiter_identity_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainRefused,
        match="inexact observed process",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            future_starter=lambda **_kwargs: _future_receipt(),
            categorized_starter=lambda **_kwargs: {
                "status": None,
                "observed_process": {"pid": True, "create_time": float("nan")},
            },
        )


def test_fresh_waiter_launch_requires_visible_parent_binding(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        long_chain.SuccessorLongChainRefused,
        match="returned no visible parent",
    ):
        long_chain.start_long_chain(
            execute=True,
            root=tmp_path,
            future_starter=lambda **_kwargs: _future_receipt(),
            categorized_starter=lambda **_kwargs: {
                "launched_pid": 42,
                "status": {
                    "state": "waiting_predecessor",
                    "supervisor": {"pid": 43, "create_time": 43.5},
                },
            },
        )
