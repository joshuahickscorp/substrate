"""A deterministic seeded organized memory and the four control recall policies.

The organized memory supports the four capabilities the epoch requires. Revision lets a newer
reliable observation overwrite an older one. Provenance keeps a trusted value and refuses to let an
unreliable observation overwrite it. Deletion retracts a key so a dead value is never recalled and
the decision falls back to the observable default. Action-conditioned recall turns the current
recalled value into a decision through the shared action bucket. Each control is a deliberately
weaker policy: no-memory ignores the store, flat-memory trusts the latest observation regardless of
provenance or retraction, replay-only aggregates raw observations by frequency, and stale-memory
freezes at the first observation and never applies a revision or a deletion.

Every policy is a pure function of the ops it is shown and the observable default, so a run is fully
deterministic in the seed with no wall clock and no unseeded randomness.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim. A policy
recalling a value is a mechanics witness, not evidence that any memory organization carries value.

House style: no em or en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .memory_organization_bed import ORGANIZED_ARM, MemoryOp


class MemoryImplRefusal(ValueError):
    """Raised when a recall is asked for an unknown arm. The dispatcher fails closed."""


def _ordered(ops: Sequence[MemoryOp], key: int) -> list[MemoryOp]:
    return sorted((op for op in ops if op.key == key), key=lambda op: op.step)


def organized_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """Organized memory: apply revision, honor provenance, and respect deletion; else use default."""

    value: int | None = None
    value_reliable = False
    deleted = False
    for op in _ordered(ops, key):
        if op.retract:
            deleted = True
            value = None
            value_reliable = False
            continue
        if value_reliable and not op.reliable:
            # Provenance: an unreliable observation cannot overwrite a trusted value.
            continue
        value = op.value
        value_reliable = op.reliable
        deleted = False
    if deleted or value is None:
        return default_feature
    return value


def no_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """No-memory control: ignore the store entirely and decide from the observable default."""

    return default_feature


def flat_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """Flat-memory control: trust the latest observation, ignoring provenance and retraction."""

    ordered = _ordered(ops, key)
    if not ordered:
        return default_feature
    return ordered[-1].value


def replay_only_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """Replay-only control: aggregate raw observations by frequency, ignoring recency and provenance."""

    ordered = _ordered(ops, key)
    if not ordered:
        return default_feature
    counts: dict[int, int] = {}
    for op in ordered:
        counts[op.value] = counts.get(op.value, 0) + 1
    # Deterministic tie break: highest count, then smallest value.
    return max(counts, key=lambda candidate: (counts[candidate], -candidate))


def stale_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """Stale-memory control: freeze at the first observation and never revise or delete."""

    ordered = _ordered(ops, key)
    if not ordered:
        return default_feature
    return ordered[0].value


ARM_POLICIES: dict[str, Callable[[Sequence[MemoryOp], int, int], int]] = {
    ORGANIZED_ARM: organized_recall,
    "no-memory": no_memory_recall,
    "flat-memory": flat_memory_recall,
    "replay-only": replay_only_recall,
    "stale-memory": stale_memory_recall,
}


def recall(arm: str, ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:
    """Dispatch a recall to one arm's policy. Fails closed on an unknown arm."""

    policy = ARM_POLICIES.get(arm)
    if policy is None:
        raise MemoryImplRefusal(f"unknown memory arm {arm!r}")
    return policy(ops, key, default_feature)
