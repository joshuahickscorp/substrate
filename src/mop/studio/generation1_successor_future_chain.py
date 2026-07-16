"""One-command launcher for the append-only Generation 1 future chain."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mop.studio.generation1_successor_chain_v4 import (
    DEFAULT_ROOT as ADOPTER_ROOT,
)
from mop.studio.generation1_successor_chain_v4 import (
    SuccessorChainRefused,
    read_validated_complete_chain_status,
    start_chain_detached,
    validate_chain_status,
)
from mop.studio.generation1_successor_extension_chain import (
    DEFAULT_ROOT as EXTENSION_ROOT,
)
from mop.studio.generation1_successor_extension_chain import (
    SuccessorExtensionRefused,
    read_validated_complete_extension_status,
    start_extension_detached,
    validate_extension_status,
)

PROGRAM_ID = "generation1-successor-future-chain-v1"
RECEIPT_SCHEMA = "mop-generation1-successor-future-chain-start/v1"
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})
ACK_ATTEMPTS = 31
ACK_INTERVAL_SECONDS = 1.0
PROCESS_TIME_TOLERANCE_SECONDS = 0.02

Starter = Callable[..., Mapping[str, Any]]
SleepFn = Callable[[float], None]
StatusValidator = Callable[[Mapping[str, Any]], str]


class SuccessorFutureChainRefused(RuntimeError):
    """The complete future chain could not be started at an exact safe boundary."""


def _validated_status_state(
    status: Mapping[str, Any],
    *,
    label: str,
    validator: StatusValidator,
) -> str:
    try:
        return validator(status)
    except (
        SuccessorChainRefused,
        SuccessorExtensionRefused,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SuccessorFutureChainRefused(
            f"{label} acknowledgement is not authoritative: {type(exc).__name__}: {exc}"
        ) from exc


def _replay_complete_adopter(status: Mapping[str, Any]) -> None:
    try:
        replayed = read_validated_complete_chain_status(root=ADOPTER_ROOT)
    except (
        SuccessorChainRefused,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SuccessorFutureChainRefused(
            f"v4 adopter complete acknowledgement failed durable replay: {type(exc).__name__}: {exc}"
        ) from exc
    if dict(status) != replayed:
        raise SuccessorFutureChainRefused("v4 adopter complete acknowledgement disagrees with durable replay")


def _replay_complete_extension(status: Mapping[str, Any]) -> None:
    try:
        replayed = read_validated_complete_extension_status(
            root=EXTENSION_ROOT,
        )
    except (
        SuccessorChainRefused,
        SuccessorExtensionRefused,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SuccessorFutureChainRefused(
            f"extension complete acknowledgement failed durable replay: {type(exc).__name__}: {exc}"
        ) from exc
    if dict(status) != replayed:
        raise SuccessorFutureChainRefused("extension complete acknowledgement disagrees with durable replay")


def _status_matches_observed_process(
    status: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    supervisor = status.get("supervisor")
    observed_pid = observed.get("pid")
    observed_created = observed.get("create_time")
    if (
        not isinstance(supervisor, Mapping)
        or isinstance(observed_pid, bool)
        or not isinstance(observed_pid, int)
        or observed_pid <= 0
        or isinstance(observed_created, bool)
        or not isinstance(observed_created, int | float)
        or not math.isfinite(float(observed_created))
        or float(observed_created) <= 0
        or isinstance(supervisor.get("pid"), bool)
        or not isinstance(supervisor.get("pid"), int)
        or supervisor["pid"] <= 0
        or isinstance(supervisor.get("create_time"), bool)
        or not isinstance(supervisor.get("create_time"), int | float)
        or not math.isfinite(float(supervisor["create_time"]))
        or float(supervisor["create_time"]) <= 0
    ):
        return False
    return supervisor["pid"] == observed_pid and math.isclose(
        float(supervisor["create_time"]),
        float(observed_created),
        rel_tol=0.0,
        abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
    )


def _start_until_acknowledged(
    *,
    starter: Starter,
    label: str,
    validator: Callable[[Mapping[str, Any]], str],
    execute: bool,
    use_caffeinate: bool,
    ack_attempts: int,
    ack_interval_seconds: float,
    sleep_fn: SleepFn,
) -> tuple[dict[str, Any], str, int]:
    if ack_attempts < 1:
        raise SuccessorFutureChainRefused("acknowledgement attempts must be positive")
    pending_observed: dict[str, Any] | None = None
    for attempt in range(1, ack_attempts + 1):
        raw_result = starter(
            execute=execute,
            use_caffeinate=use_caffeinate,
        )
        if not isinstance(raw_result, Mapping):
            raise SuccessorFutureChainRefused(f"{label} starter returned no receipt")
        result = dict(raw_result)
        status = result.get("status")
        observed = result.get("observed_process")
        if observed is not None:
            if not isinstance(observed, Mapping):
                raise SuccessorFutureChainRefused(
                    f"{label} starter returned an invalid observed-process receipt"
                )
            current_observed = dict(observed)
            if not _status_matches_observed_process(
                {"supervisor": current_observed},
                current_observed,
            ):
                raise SuccessorFutureChainRefused(
                    f"{label} starter returned an invalid observed-process identity"
                )
            if pending_observed is None:
                pending_observed = current_observed
            elif not _status_matches_observed_process(
                {"supervisor": current_observed},
                pending_observed,
            ):
                raise SuccessorFutureChainRefused(
                    f"{label} observed-process identity changed before acknowledgement"
                )
        if pending_observed is not None and (
            not isinstance(status, Mapping) or not _status_matches_observed_process(status, pending_observed)
        ):
            if attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise SuccessorFutureChainRefused(
                f"{label} acknowledgement remained pending after {ack_attempts} attempts"
            )
        if not isinstance(status, Mapping):
            raise SuccessorFutureChainRefused(
                f"{label} starter returned no authoritative status acknowledgement"
            )
        try:
            state = validator(status)
        except SuccessorFutureChainRefused:
            if pending_observed is not None and attempt < ack_attempts:
                sleep_fn(ack_interval_seconds)
                continue
            raise
        return result, state, attempt
    raise SuccessorFutureChainRefused(f"{label} acknowledgement remained pending")


def start_future_chain(
    *,
    execute: bool,
    use_caffeinate: bool = True,
    adopter_starter: Starter = start_chain_detached,
    extension_starter: Starter = start_extension_detached,
    ack_attempts: int = ACK_ATTEMPTS,
    ack_interval_seconds: float = ACK_INTERVAL_SECONDS,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Start or resume v4, then start the observation-only post-v4 waiter."""

    if not execute:
        raise SuccessorFutureChainRefused("future-chain start requires explicit --execute")
    adopter, adopter_state, adopter_attempts = _start_until_acknowledged(
        starter=adopter_starter,
        label="v4 adopter",
        validator=lambda status: _validated_status_state(
            status,
            label="v4 adopter",
            validator=validate_chain_status,
        ),
        execute=True,
        use_caffeinate=use_caffeinate,
        ack_attempts=ack_attempts,
        ack_interval_seconds=ack_interval_seconds,
        sleep_fn=sleep_fn,
    )
    if adopter_state in UNSAFE_TERMINAL_STATES:
        raise SuccessorFutureChainRefused(f"v4 adopter is terminal in unsafe state {adopter_state}")
    if adopter_state == "complete":
        _replay_complete_adopter(adopter["status"])
    extension, extension_state, extension_attempts = _start_until_acknowledged(
        starter=extension_starter,
        label="extension",
        validator=lambda status: _validated_status_state(
            status,
            label="extension",
            validator=validate_extension_status,
        ),
        execute=True,
        use_caffeinate=use_caffeinate,
        ack_attempts=ack_attempts,
        ack_interval_seconds=ack_interval_seconds,
        sleep_fn=sleep_fn,
    )
    if extension_state in UNSAFE_TERMINAL_STATES:
        raise SuccessorFutureChainRefused(f"extension is terminal in unsafe state {extension_state}")
    if extension_state == "complete":
        _replay_complete_extension(extension["status"])
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
        "acknowledgement_attempts": {
            "adopter": adopter_attempts,
            "extension": extension_attempts,
        },
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
