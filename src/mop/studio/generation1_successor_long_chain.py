"""One-command launcher for the full append-only Generation 1 successor chain."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_categorized_batch_extension_chain import (
    CategorizedBatchExtensionRefused,
    read_validated_complete_categorized_extension_status,
    start_categorized_extension_detached,
    validate_categorized_extension_status,
)
from mop.studio.generation1_successor_future_chain import (
    PROGRAM_ID as FUTURE_CHAIN_PROGRAM_ID,
)
from mop.studio.generation1_successor_future_chain import (
    RECEIPT_SCHEMA as FUTURE_CHAIN_RECEIPT_SCHEMA,
)
from mop.studio.generation1_successor_future_chain import (
    SuccessorFutureChainRefused,
    start_future_chain,
)
from mop.studio.generation1_supervisor import FileLock, Generation1Refused

PROGRAM_ID = "generation1-successor-long-chain-v1"
RECEIPT_SCHEMA = "mop-generation1-successor-long-chain-start/v1"
DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
START_LOCK_FILE = "control/start.lock"
ACK_ATTEMPTS = 31
ACK_INTERVAL_SECONDS = 1.0
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})
PROCESS_TIME_TOLERANCE_SECONDS = 0.02

Starter = Callable[..., Mapping[str, Any]]
SleepFn = Callable[[float], None]


class SuccessorLongChainRefused(RuntimeError):
    """The full successor chain could not be started at a safe boundary."""


def _valid_process_identity(identity: Mapping[str, Any]) -> bool:
    pid = identity.get("pid")
    created = identity.get("create_time")
    return (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and isinstance(created, int | float)
        and not isinstance(created, bool)
        and math.isfinite(float(created))
        and float(created) > 0
    )


def _same_process(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    return (
        _valid_process_identity(first)
        and _valid_process_identity(second)
        and first["pid"] == second["pid"]
        and math.isclose(
            float(first["create_time"]),
            float(second["create_time"]),
            rel_tol=0.0,
            abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
        )
    )


def _start_waiter_until_acknowledged(
    *,
    starter: Starter,
    execute: bool,
    use_caffeinate: bool,
    ack_attempts: int,
    ack_interval_seconds: float,
    sleep_fn: SleepFn,
) -> tuple[dict[str, Any], str, int]:
    if ack_attempts < 1:
        raise SuccessorLongChainRefused("categorized acknowledgement attempts must be positive")
    observed: dict[str, Any] | None = None
    for attempt in range(1, ack_attempts + 1):
        raw = starter(execute=execute, use_caffeinate=use_caffeinate)
        if not isinstance(raw, Mapping):
            raise SuccessorLongChainRefused("categorized waiter starter returned no receipt")
        result = dict(raw)
        fresh_launch = "launched_pid" in result
        launched_pid = result.get("launched_pid")
        if fresh_launch and (
            not isinstance(launched_pid, int) or isinstance(launched_pid, bool) or launched_pid <= 0
        ):
            raise SuccessorLongChainRefused("categorized waiter returned an invalid detached launch process")
        current_observed = result.get("observed_process")
        if fresh_launch and current_observed is None:
            raise SuccessorLongChainRefused("categorized waiter detached launch returned no visible parent")
        if current_observed is not None:
            if not isinstance(current_observed, Mapping):
                raise SuccessorLongChainRefused("categorized waiter returned an invalid observed process")
            identity = dict(current_observed)
            if not _valid_process_identity(identity):
                raise SuccessorLongChainRefused("categorized waiter returned an inexact observed process")
            if observed is None:
                observed = identity
            elif not _same_process(identity, observed):
                raise SuccessorLongChainRefused(
                    "categorized waiter observed process changed before acknowledgement"
                )
        status = result.get("status")
        if not isinstance(status, Mapping):
            if observed is not None and attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainRefused("categorized waiter returned no authoritative status")
        try:
            state = validate_categorized_extension_status(status)
        except CategorizedBatchExtensionRefused as exc:
            if observed is not None and attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainRefused(f"categorized waiter acknowledgement is invalid: {exc}") from exc
        supervisor = status.get("supervisor")
        if observed is not None and (
            not isinstance(supervisor, Mapping) or not _same_process(supervisor, observed)
        ):
            if attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainRefused("categorized waiter observed process was never sealed")
        return result, state, attempt
    raise SuccessorLongChainRefused("categorized waiter acknowledgement remained pending")


def start_long_chain(
    *,
    execute: bool,
    use_caffeinate: bool = True,
    root: Path = DEFAULT_ROOT,
    future_starter: Starter = start_future_chain,
    categorized_starter: Starter = start_categorized_extension_detached,
    ack_attempts: int = ACK_ATTEMPTS,
    ack_interval_seconds: float = ACK_INTERVAL_SECONDS,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Start current future-chain components, then the categorized waiter."""

    if not execute:
        raise SuccessorLongChainRefused("successor long-chain start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        future = future_starter(
            execute=True,
            use_caffeinate=use_caffeinate,
        )
        if (
            not isinstance(future, Mapping)
            or future.get("schema") != FUTURE_CHAIN_RECEIPT_SCHEMA
            or future.get("program_id") != FUTURE_CHAIN_PROGRAM_ID
            or future.get("signals_allowed") is not False
            or future.get("activation_allowed") is not False
            or future.get("scientific_promotion") is not False
            or future.get("all_ok") is not True
        ):
            raise SuccessorLongChainRefused("future-chain starter returned no clean success receipt")
        categorized, categorized_state, attempts = _start_waiter_until_acknowledged(
            starter=categorized_starter,
            execute=True,
            use_caffeinate=use_caffeinate,
            ack_attempts=ack_attempts,
            ack_interval_seconds=ack_interval_seconds,
            sleep_fn=sleep_fn,
        )
        if categorized_state in UNSAFE_TERMINAL_STATES:
            raise SuccessorLongChainRefused(
                f"categorized waiter is terminal in unsafe state {categorized_state}"
            )
        if categorized_state == "complete":
            replayed = read_validated_complete_categorized_extension_status()
            if categorized.get("status") != replayed:
                raise SuccessorLongChainRefused(
                    "categorized complete acknowledgement disagrees with durable replay"
                )
        return {
            "schema": RECEIPT_SCHEMA,
            "program_id": PROGRAM_ID,
            "future_chain": dict(future),
            "categorized_extension": categorized,
            "categorized_state": categorized_state,
            "start_order": [
                "generation1-successor-future-chain-v1",
                "generation1-categorized-batch-extension-chain-v1",
            ],
            "target_order": [
                "generation1-successor-evidence-chain-v4",
                "generation1-successor-horizon-v2",
                "generation1-successor-categorized-batch-wave-v1",
            ],
            "acknowledgement_attempts": {"categorized_extension": attempts},
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
    start.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        payload = start_long_chain(
            execute=arguments.execute,
            use_caffeinate=not arguments.no_caffeinate,
            root=arguments.root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        SuccessorLongChainRefused,
        SuccessorFutureChainRefused,
        CategorizedBatchExtensionRefused,
        Generation1Refused,
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
    "DEFAULT_ROOT",
    "PROGRAM_ID",
    "RECEIPT_SCHEMA",
    "SuccessorLongChainRefused",
    "build_parser",
    "main",
    "start_long_chain",
]


if __name__ == "__main__":
    raise SystemExit(main())
