from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from mop.studio import generation1_successor_future_chain as future


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
        return {"status": {"state": "waiting_legacy"}, "launched_pid": 41}

    def extension(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(("extension", kwargs))
        return {"status": {"state": "waiting_predecessor"}, "launched_pid": 42}

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
            adopter_starter=lambda **_kwargs: {"status": {"state": state}},
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
            adopter_starter=lambda **_kwargs: {"status": {"state": "complete"}},
            extension_starter=lambda **_kwargs: {"status": {"state": state}},
        )


def test_future_chain_parser_exposes_one_idempotent_start_command() -> None:
    arguments = future.build_parser().parse_args(["start", "--execute", "--no-caffeinate"])

    assert (arguments.command, arguments.execute, arguments.no_caffeinate) == (
        "start",
        True,
        True,
    )
