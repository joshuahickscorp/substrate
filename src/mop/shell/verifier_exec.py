from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mop.diagnostics.hardness import SLOT_CARD, SLOT_ORDER

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


@dataclass(frozen=True)
class Predicate:
    slot: str
    op: str
    value: int

    def __post_init__(self) -> None:
        if self.slot not in SLOT_ORDER:
            raise ValueError(f"unknown slot {self.slot!r}, expected one of {SLOT_ORDER}")
        if self.op not in _OPS:
            raise ValueError(f"unknown op {self.op!r}, expected one of {sorted(_OPS)}")
        if not (0 <= int(self.value) < SLOT_CARD[self.slot]):
            raise ValueError(f"value {self.value} out of range for slot {self.slot}")

    def eval(self, col: torch.Tensor) -> torch.Tensor:
        return _OPS[self.op](col, self.value)


@dataclass(frozen=True)
class Program:
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

    def slot_index(self, slot: str) -> int:
        return SLOT_ORDER.index(slot)

    def execute_on_slots(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.dim() != 2 or slots.shape[1] != len(SLOT_ORDER):
            raise ValueError(f"slots must be (n, {len(SLOT_ORDER)}), got {tuple(slots.shape)}")
        out = torch.ones(slots.shape[0], dtype=torch.bool)
        for p in self.predicates:
            out = out & p.eval(slots[:, self.slot_index(p.slot)])
        return out.long()


def target_program() -> Program:
    return Program(
        predicates=(
            Predicate("count", "ge", 2),
            Predicate("color", "ne", 0),
        )
    )


class ExecutableVerifier:
    def __init__(self, program: Program | None = None):
        self.program = program or target_program()

    def execute(self, cand_slots: torch.Tensor) -> torch.Tensor:
        return self.program.execute_on_slots(cand_slots)

    def consistent(self, cand_slots: torch.Tensor, cand_label: torch.Tensor) -> torch.Tensor:
        executed = self.execute(cand_slots)
        return executed == cand_label

    def flops_per_check(self, batch: int = 1) -> int:
        return int(batch * len(self.program.predicates))
