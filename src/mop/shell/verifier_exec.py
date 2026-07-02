"""A3 executable verifier and a minimal DSL over the (shape, color, motion, count) slots. This is the
deterministic, CPU-executed checker the shell's test-time-compute loop queries: it is real code
execution, not a learned scorer. Given a candidate slot decoding of a sample (four predicted integers),
the verifier runs a tiny program and returns correct or incorrect. The DSL is the small program space
the shell can emit (a conjunction of slot predicates) and the verifier can execute.

Why this can bite: a learned verifier (the ex18 line) can only carry the signal the training loss
happened to install, and the audit showed it ties a shuffled control. A DSL executor is exact by
construction: it reads the candidate slots and computes the SAME boolean the label was defined by, with
zero learned parameters. So verifier-guided iteration can, in principle, select the refinement whose
decoded slots actually satisfy the target program, a signal no matched-FLOP feedforward pass gets for
free. If it STILL ties feedforward on the hard bin, test-time compute is dead at this substrate even
with a perfect oracle checker, which is the sharpest possible kill-switch.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mop.diagnostics.hardness import SLOT_CARD, SLOT_ORDER

# operators the DSL predicates may use, all deterministic integer comparisons over one slot
_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


@dataclass(frozen=True)
class Predicate:
    """One DSL atom: compare slot `slot` against constant `value` under operator `op` (eq/ne/ge/le).
    Evaluates elementwise on an integer slot column (a torch tensor)."""

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
    """A minimal DSL program: a CONJUNCTION of slot predicates (the label is true iff every predicate
    holds). This is the whole emittable program space the shell searches over and the verifier executes.
    Deliberately tiny so the executor is provably exact and cheap (a handful of integer comparisons)."""

    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

    def slot_index(self, slot: str) -> int:
        return SLOT_ORDER.index(slot)

    def execute_on_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """Execute the program on an integer slot table (n, 4) in SLOT_ORDER, returning the integer
        label (n,) in {0, 1}: 1 iff all predicates hold. This is the ground-truth label generator AND
        the per-candidate checker (same code path), so the verifier is exact by construction."""
        if slots.dim() != 2 or slots.shape[1] != len(SLOT_ORDER):
            raise ValueError(f"slots must be (n, {len(SLOT_ORDER)}), got {tuple(slots.shape)}")
        out = torch.ones(slots.shape[0], dtype=torch.bool)
        for p in self.predicates:
            out = out & p.eval(slots[:, self.slot_index(p.slot)])
        return out.long()


def target_program() -> Program:
    """The fixed target program the graded task labels by (hardness.make_graded_slot_task calls this).
    A two-predicate conjunction over the count and color slots (the count predicate exercises a numeric
    >= op), chosen so the positive rate is near 0.45: a balanced label whose majority-class baseline is
    only ~0.55, leaving real headroom above chance for test-time compute to matter. A correct decision
    still depends on reading two independent slots correctly through the corruption."""
    return Program(
        predicates=(
            Predicate("count", "ge", 2),
            Predicate("color", "ne", 0),
        )
    )


class ExecutableVerifier:
    """CPU code-execution checker over candidate slot decodings. Holds the target Program and, given a
    batch of candidate slot integers (n, 4), executes the program to produce the candidate label and
    compares it to a reference. No learned parameters: this is the oracle the test-time-compute loop
    consults. The loop decodes slots from a latent (a trained slot head), the verifier runs the DSL on
    those decoded slots, and disagreement between the executed label and the head's own label prediction
    flags a candidate as unverified (trigger for more compute or for selecting a better candidate)."""

    def __init__(self, program: Program | None = None):
        self.program = program or target_program()

    def execute(self, cand_slots: torch.Tensor) -> torch.Tensor:
        """Run the DSL on candidate slot integers (n, 4), returning the executed label (n,) in {0,1}."""
        return self.program.execute_on_slots(cand_slots)

    def consistent(self, cand_slots: torch.Tensor, cand_label: torch.Tensor) -> torch.Tensor:
        """Per-candidate verifier verdict: does the candidate's own predicted label match the label the
        DSL computes from the candidate's decoded slots. True means self-consistent (accept), False
        means the decoded slots and the predicted label disagree (reject, spend more compute). This is
        the only signal the loop reads, and it is pure code execution over the candidate's own outputs
        (no test-label leakage: it never sees the ground-truth y)."""
        executed = self.execute(cand_slots)
        return executed == cand_label

    def flops_per_check(self, batch: int = 1) -> int:
        """FLOP charge for one verifier check over a batch: one integer comparison per predicate per
        sample (the executor's whole cost). Charged to the verifier-guided arm so it cannot win on
        unbilled compute (matched-FLOP honesty)."""
        return int(batch * len(self.program.predicates))
