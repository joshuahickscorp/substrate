"""One-command launcher for the append-only Generation 1 future chain."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mop.studio.generation1_successor_chain_v4 import (
    SuccessorChainRefused,
    start_chain_detached,
)
from mop.studio.generation1_successor_extension_chain import (
    SuccessorExtensionRefused,
    start_extension_detached,
)

PROGRAM_ID = "generation1-successor-future-chain-v1"
RECEIPT_SCHEMA = "mop-generation1-successor-future-chain-start/v1"
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})

Starter = Callable[..., Mapping[str, Any]]


class SuccessorFutureChainRefused(RuntimeError):
    """The complete future chain could not be started at an exact safe boundary."""


def _reported_state(result: Mapping[str, Any]) -> str | None:
    status = result.get("status")
    if not isinstance(status, Mapping):
        return None
    state = status.get("state")
    return state if isinstance(state, str) else None


def start_future_chain(
    *,
    execute: bool,
    use_caffeinate: bool = True,
    adopter_starter: Starter = start_chain_detached,
    extension_starter: Starter = start_extension_detached,
) -> dict[str, Any]:
    """Start or resume v4, then start the observation-only post-v4 waiter."""

    if not execute:
        raise SuccessorFutureChainRefused("future-chain start requires explicit --execute")
    adopter = dict(
        adopter_starter(
            execute=True,
            use_caffeinate=use_caffeinate,
        )
    )
    adopter_state = _reported_state(adopter)
    if adopter_state in UNSAFE_TERMINAL_STATES:
        raise SuccessorFutureChainRefused(f"v4 adopter is terminal in unsafe state {adopter_state}")
    extension = dict(
        extension_starter(
            execute=True,
            use_caffeinate=use_caffeinate,
        )
    )
    extension_state = _reported_state(extension)
    if extension_state in UNSAFE_TERMINAL_STATES:
        raise SuccessorFutureChainRefused(f"extension is terminal in unsafe state {extension_state}")
    return {
        "schema": RECEIPT_SCHEMA,
        "program_id": PROGRAM_ID,
        "adopter": adopter,
        "extension": extension,
        "start_order": [
            "generation1-successor-evidence-chain-v4",
            "generation1-successor-extension-chain-v1",
        ],
        "signals_allowed": False,
        "activation_allowed": False,
        "scientific_promotion": False,
        "all_ok": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--execute", action="store_true")
    start.add_argument("--no-caffeinate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        payload = start_future_chain(
            execute=arguments.execute,
            use_caffeinate=not arguments.no_caffeinate,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        SuccessorFutureChainRefused,
        SuccessorChainRefused,
        SuccessorExtensionRefused,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"all_ok": False, "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2


__all__ = [
    "PROGRAM_ID",
    "RECEIPT_SCHEMA",
    "SuccessorFutureChainRefused",
    "build_parser",
    "main",
    "start_future_chain",
]


if __name__ == "__main__":
    raise SystemExit(main())
