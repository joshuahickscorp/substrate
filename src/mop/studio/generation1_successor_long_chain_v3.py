"""One-command launcher for the extended append-only Generation 1 successor chain."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_full_generations_extension_chain import (
    FullGenerationsExtensionRefused,
    read_validated_complete_full_generations_extension_status,
    start_full_generations_extension_detached,
    validate_full_generations_extension_status,
)
from mop.studio.generation1_successor_long_chain_v2 import (
    PROGRAM_ID as LONG_CHAIN_V2_PROGRAM_ID,
)
from mop.studio.generation1_successor_long_chain_v2 import (
    RECEIPT_SCHEMA as LONG_CHAIN_V2_RECEIPT_SCHEMA,
)
from mop.studio.generation1_successor_long_chain_v2 import (
    SuccessorLongChainRefused,
)
from mop.studio.generation1_successor_long_chain_v2 import (
    start_long_chain as start_long_chain_v2,
)
from mop.studio.generation1_supervisor import FileLock, Generation1Refused

PROGRAM_ID = "generation1-successor-long-chain-v3"
RECEIPT_SCHEMA = "mop-generation1-successor-long-chain-start/v3"
DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
START_LOCK_FILE = "control/start.lock"
ACK_ATTEMPTS = 31
ACK_INTERVAL_SECONDS = 1.0
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})
PROCESS_TIME_TOLERANCE_SECONDS = 0.02

FULL_GENERATIONS_EXTENSION_PROGRAM_ID = "generation1-full-generations-extension-chain-v1"
FULL_GENERATIONS_WAVE_PROGRAM_ID = "generation1-full-generations-wave-v1"

Starter = Callable[..., Mapping[str, Any]]
SleepFn = Callable[[float], None]


class SuccessorLongChainV3Refused(RuntimeError):
    pass


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
        raise SuccessorLongChainV3Refused("full-generations acknowledgement attempts must be positive")
    observed: dict[str, Any] | None = None
    for attempt in range(1, ack_attempts + 1):
        raw = starter(execute=execute, use_caffeinate=use_caffeinate)
        if not isinstance(raw, Mapping):
            raise SuccessorLongChainV3Refused("full-generations waiter starter returned no receipt")
        result = dict(raw)
        fresh_launch = "launched_pid" in result
        launched_pid = result.get("launched_pid")
        if fresh_launch and (
            not isinstance(launched_pid, int) or isinstance(launched_pid, bool) or launched_pid <= 0
        ):
            raise SuccessorLongChainV3Refused(
                "full-generations waiter returned an invalid detached launch process"
            )
        current_observed = result.get("observed_process")
        if fresh_launch and current_observed is None:
            raise SuccessorLongChainV3Refused(
                "full-generations waiter detached launch returned no visible parent"
            )
        if current_observed is not None:
            if not isinstance(current_observed, Mapping):
                raise SuccessorLongChainV3Refused(
                    "full-generations waiter returned an invalid observed process"
                )
            identity = dict(current_observed)
            if not _valid_process_identity(identity):
                raise SuccessorLongChainV3Refused(
                    "full-generations waiter returned an inexact observed process"
                )
            if observed is None:
                observed = identity
            elif not _same_process(identity, observed):
                raise SuccessorLongChainV3Refused(
                    "full-generations waiter observed process changed before acknowledgement"
                )
        status = result.get("status")
        if not isinstance(status, Mapping):
            if observed is not None and attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainV3Refused("full-generations waiter returned no authoritative status")
        try:
            state = validate_full_generations_extension_status(status)
        except FullGenerationsExtensionRefused as exc:
            if observed is not None and attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainV3Refused(
                f"full-generations waiter acknowledgement is invalid: {exc}"
            ) from exc
        supervisor = status.get("supervisor")
        if observed is not None and (
            not isinstance(supervisor, Mapping) or not _same_process(supervisor, observed)
        ):
            if attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorLongChainV3Refused("full-generations waiter observed process was never sealed")
        return result, state, attempt
    raise SuccessorLongChainV3Refused("full-generations waiter acknowledgement remained pending")


def start_long_chain(
    *,
    execute: bool,
    use_caffeinate: bool = True,
    root: Path = DEFAULT_ROOT,
    long_chain_v2_starter: Starter = start_long_chain_v2,
    full_generations_starter: Starter = start_full_generations_extension_detached,
    ack_attempts: int = ACK_ATTEMPTS,
    ack_interval_seconds: float = ACK_INTERVAL_SECONDS,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, Any]:

    if not execute:
        raise SuccessorLongChainV3Refused("successor long-chain start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        v2 = long_chain_v2_starter(
            execute=True,
            use_caffeinate=use_caffeinate,
        )
        if (
            not isinstance(v2, Mapping)
            or v2.get("schema") != LONG_CHAIN_V2_RECEIPT_SCHEMA
            or v2.get("program_id") != LONG_CHAIN_V2_PROGRAM_ID
            or v2.get("signals_allowed") is not False
            or v2.get("activation_allowed") is not False
            or v2.get("scientific_promotion") is not False
            or v2.get("all_ok") is not True
        ):
            raise SuccessorLongChainV3Refused("long-chain-v2 starter returned no clean success receipt")
        v2_start_order = v2.get("start_order")
        v2_target_order = v2.get("target_order")
        if (
            not isinstance(v2_start_order, list)
            or not isinstance(v2_target_order, list)
            or any(
                not isinstance(program_id, str) or not program_id
                for program_id in [*v2_start_order, *v2_target_order]
            )
        ):
            raise SuccessorLongChainV3Refused("long-chain-v2 receipt program orders drifted")
        full_generations, full_generations_state, attempts = _start_waiter_until_acknowledged(
            starter=full_generations_starter,
            execute=True,
            use_caffeinate=use_caffeinate,
            ack_attempts=ack_attempts,
            ack_interval_seconds=ack_interval_seconds,
            sleep_fn=sleep_fn,
        )
        if full_generations_state in UNSAFE_TERMINAL_STATES:
            raise SuccessorLongChainV3Refused(
                f"full-generations waiter is terminal in unsafe state {full_generations_state}"
            )
        if full_generations_state == "complete":
            replayed = read_validated_complete_full_generations_extension_status()
            if full_generations.get("status") != replayed:
                raise SuccessorLongChainV3Refused(
                    "full-generations complete acknowledgement disagrees with durable replay"
                )
        return {
            "schema": RECEIPT_SCHEMA,
            "program_id": PROGRAM_ID,
            "long_chain_v2": dict(v2),
            "full_generations_extension": full_generations,
            "full_generations_state": full_generations_state,
            "start_order": [
                *v2_start_order,
                FULL_GENERATIONS_EXTENSION_PROGRAM_ID,
            ],
            "target_order": [
                *v2_target_order,
                FULL_GENERATIONS_WAVE_PROGRAM_ID,
            ],
            "acknowledgement_attempts": {"full_generations_extension": attempts},
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
        SuccessorLongChainV3Refused,
        SuccessorLongChainRefused,
        FullGenerationsExtensionRefused,
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
    "FULL_GENERATIONS_EXTENSION_PROGRAM_ID",
    "FULL_GENERATIONS_WAVE_PROGRAM_ID",
    "LONG_CHAIN_V2_PROGRAM_ID",
    "LONG_CHAIN_V2_RECEIPT_SCHEMA",
    "PROGRAM_ID",
    "RECEIPT_SCHEMA",
    "SuccessorLongChainV3Refused",
    "build_parser",
    "main",
    "start_long_chain",
]


if __name__ == "__main__":
    raise SystemExit(main())
