
from __future__ import annotations

from collections.abc import Callable, Sequence

from .memory_organization_bed import ORGANIZED_ARM, MemoryOp


class MemoryImplRefusal(ValueError):
    pass


def _ordered(ops: Sequence[MemoryOp], key: int) -> list[MemoryOp]:
    return sorted((op for op in ops if op.key == key), key=lambda op: op.step)


def organized_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:

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
            continue
        value = op.value
        value_reliable = op.reliable
        deleted = False
    if deleted or value is None:
        return default_feature
    return value


def no_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:

    return default_feature


def flat_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:

    ordered = _ordered(ops, key)
    if not ordered:
        return default_feature
    return ordered[-1].value


def replay_only_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:

    ordered = _ordered(ops, key)
    if not ordered:
        return default_feature
    counts: dict[int, int] = {}
    for op in ordered:
        counts[op.value] = counts.get(op.value, 0) + 1
    return max(counts, key=lambda candidate: (counts[candidate], -candidate))


def stale_memory_recall(ops: Sequence[MemoryOp], key: int, default_feature: int) -> int:

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

    policy = ARM_POLICIES.get(arm)
    if policy is None:
        raise MemoryImplRefusal(f"unknown memory arm {arm!r}")
    return policy(ops, key, default_feature)
