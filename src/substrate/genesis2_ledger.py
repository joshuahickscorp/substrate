"""The common update ledger, and the rewrite allocator scored against it.

Genesis I compared arms on attempt count.  That is not a fair comparison
because the update units differ: S2's attempt is a one-byte exact association
and the field's attempt is a structural rewrite.  Counting them as one thing
each either starves the monolith or flatters it, depending on which way the cap
is set.

This module replaces attempt count with a common ledger.  Every proposal, from
whichever arm, records what it actually cost and what it was actually worth, so
utility per written byte and utility per compute unit are comparable across
architectures that do not share an update unit.

The allocator is the other half.  It picks an update granularity *before* the
outcome is known -- from the shape of the evidence, not from the verdict -- and
is scored against always-micro, always-structural, fixed thresholds, the S2
policy, and an oracle that is allowed to see which granularity actually paid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from substrate import genesis2_config as C2

_ACTIVATION = False


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# One ledger row per proposal
# --------------------------------------------------------------------------


@dataclass
class UpdateRecord:
    """Everything the constitution requires a proposal to declare."""

    proposal_id: str
    arm: str
    granularity: str
    bytes_read: int = 0
    bytes_written: int = 0
    information_introduced: int = 0
    compute: int = 0
    latency_us: int = 0
    scope: str = ""
    durability: str = "durable"
    rollback_cost: int = 0
    committed: bool = False
    future_utility: float = 0.0

    def cost_weight(self) -> float:
        return C2.GRANULARITY_COST_WEIGHT.get(self.granularity, 1.0)

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateLedger:
    """The write economy of one arm over one developmental history."""

    arm: str
    records: list[UpdateRecord] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    def record(self, record: UpdateRecord) -> UpdateRecord:
        self.records.append(record)
        return record

    def settle(self, proposal_id: str, *, committed: bool, future_utility: float, rollback_cost: int = 0) -> None:
        """Attach the verdict to a proposal that was already attempted."""
        for row in reversed(self.records):
            if row.proposal_id == proposal_id:
                row.committed = bool(committed)
                row.future_utility = float(future_utility)
                row.rollback_cost = int(rollback_cost)
                return
        raise KeyError(f"{self.arm}: no ledger row for proposal {proposal_id!r}")

    # -- the seven required reports --------------------------------------

    def attempt_count(self) -> int:
        return len(self.records)

    def committed_count(self) -> int:
        return sum(1 for row in self.records if row.committed)

    def useful_commits(self) -> int:
        return sum(1 for row in self.records if row.committed and row.future_utility > 0.0)

    def total_utility(self) -> float:
        return sum(row.future_utility for row in self.records if row.committed)

    def utility_per_commit(self) -> float:
        committed = self.committed_count()
        return self.total_utility() / committed if committed else 0.0

    def utility_per_written_byte(self) -> float:
        written = sum(row.bytes_written for row in self.records if row.committed)
        return self.total_utility() / written if written else 0.0

    def utility_per_compute_unit(self) -> float:
        compute = sum(row.compute for row in self.records)
        return self.total_utility() / compute if compute else 0.0

    def utility_per_wall_time(self) -> float:
        return self.total_utility() / self.wall_clock_seconds if self.wall_clock_seconds > 0 else 0.0

    def granularity_mix(self) -> dict[str, int]:
        mix = {granularity: 0 for granularity in C2.UPDATE_GRANULARITIES}
        for row in self.records:
            if row.committed:
                mix[row.granularity] = mix.get(row.granularity, 0) + 1
        return mix

    def weighted_cost(self) -> float:
        return sum(row.cost_weight() for row in self.records if row.committed)

    def topology_share(self) -> float:
        committed = self.committed_count()
        if not committed:
            return 0.0
        return self.granularity_mix().get("topology_revision", 0) / committed

    def report(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "attempt_count": self.attempt_count(),
            "committed_count": self.committed_count(),
            "useful_commits": self.useful_commits(),
            "total_utility": self.total_utility(),
            "utility_per_commit": self.utility_per_commit(),
            "utility_per_written_byte": self.utility_per_written_byte(),
            "utility_per_compute_unit": self.utility_per_compute_unit(),
            "utility_per_wall_time": self.utility_per_wall_time(),
            "granularity_mix": self.granularity_mix(),
            "weighted_cost": self.weighted_cost(),
            "topology_share": self.topology_share(),
            "bytes_written": sum(row.bytes_written for row in self.records if row.committed),
            "bytes_read": sum(row.bytes_read for row in self.records),
            "activation": _ACTIVATION,
        }

    def digest(self) -> str:
        return _digest([row.row() for row in self.records])


def misallocation_audit(ledgers: Sequence[UpdateLedger]) -> dict[str, Any]:
    """Fail an arm that spends topology on problems a micro-write already solves."""
    rows = {}
    for ledger in ledgers:
        share = ledger.topology_share()
        rows[ledger.arm] = {
            "topology_share": share,
            "ceiling": C2.TOPOLOGY_MISALLOCATION_CEILING,
            "within_ceiling": share <= C2.TOPOLOGY_MISALLOCATION_CEILING,
            "granularity_mix": ledger.granularity_mix(),
        }
    return {
        "arms": rows,
        "all_pass": all(row["within_ceiling"] for row in rows.values()),
        "activation": _ACTIVATION,
    }


# --------------------------------------------------------------------------
# The rewrite allocator
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the allocator is allowed to see before it chooses a granularity.

    Every field here is derivable from observations and from the material's own
    state.  None of it is an evaluator verdict, a sealed answer, or a probe
    outcome: choosing granularity *after* the outcome is the
    ``precision_promotion_reads_outcomes`` mutation.
    """

    repeat_count: int
    distinct_values_at_address: int
    scope_support: int
    scope_rule_fitted: bool
    contradiction_seen: bool
    residency_pressure: float
    novel_key: bool


#: Fixed thresholds the conditional allocator is compared against.  These are
#: the "fixed thresholds" comparator of master plan section 6.
FIXED_THRESHOLDS = {
    "promote_after_repeats": 3,
    "consolidate_after_support": 4,
    "topology_after_contradictions": 1,
}


def allocate(evidence: Evidence) -> str:
    """Choose the cheapest sufficient update granularity.

    The ladder is strict: nothing reaches topology that a micro-write, a low-bit
    adjustment, a promotion or a consolidation could have handled.
    """
    # A contradiction inside a scope that already carries a fitted rule is the
    # one thing a cheaper update genuinely cannot express: the general rule must
    # survive and the exception must live somewhere else.
    if evidence.contradiction_seen and evidence.scope_rule_fitted:
        return "topology_revision"
    # Enough support to fit a rule, and the rule is not yet installed.
    if evidence.scope_support >= FIXED_THRESHOLDS["consolidate_after_support"] and not evidence.scope_rule_fitted:
        return "structural_consolidation"
    # A repeatedly confirmed association is worth promoting to a durable fact.
    if evidence.repeat_count >= FIXED_THRESHOLDS["promote_after_repeats"] and evidence.distinct_values_at_address == 1:
        return "association_promotion"
    # Under residency pressure the cheapest useful change is a precision cut.
    if evidence.residency_pressure >= 0.8 and not evidence.novel_key:
        return "local_low_bit_adjustment"
    return "micro_association"


def allocate_always_micro(_evidence: Evidence) -> str:
    return "micro_association"


def allocate_always_structural(_evidence: Evidence) -> str:
    return "structural_consolidation"


def allocate_fixed_thresholds(evidence: Evidence) -> str:
    """Thresholds with no conditional ordering: the first rule that matches wins."""
    if evidence.repeat_count >= FIXED_THRESHOLDS["promote_after_repeats"]:
        return "association_promotion"
    if evidence.scope_support >= FIXED_THRESHOLDS["consolidate_after_support"]:
        return "structural_consolidation"
    if evidence.contradiction_seen:
        return "topology_revision"
    return "micro_association"


def allocate_s2_policy(_evidence: Evidence) -> str:
    """S2 has exactly one update unit: the exact associative write."""
    return "micro_association"


def allocate_oracle(evidence: Evidence, *, paid: Mapping[str, float]) -> str:
    """The upper reference.  Allowed to see which granularity actually paid."""
    if not paid:
        return allocate(evidence)
    return max(paid, key=lambda granularity: (paid[granularity], -C2.GRANULARITY_COST_WEIGHT.get(granularity, 1.0)))


ALLOCATORS = {
    "conditional": allocate,
    "always_micro": allocate_always_micro,
    "always_structural": allocate_always_structural,
    "fixed_thresholds": allocate_fixed_thresholds,
    "s2_policy": allocate_s2_policy,
}


def demo() -> None:
    """Runnable self-check for the ledger arithmetic and the allocation ladder."""
    ledger = UpdateLedger(arm="demo")
    ledger.wall_clock_seconds = 2.0
    ledger.record(UpdateRecord("p0", "demo", "micro_association", bytes_written=32, compute=4))
    ledger.record(UpdateRecord("p1", "demo", "structural_consolidation", bytes_written=96, compute=40))
    ledger.record(UpdateRecord("p2", "demo", "topology_revision", bytes_written=256, compute=200))
    ledger.settle("p0", committed=True, future_utility=0.25)
    ledger.settle("p1", committed=True, future_utility=0.5)
    ledger.settle("p2", committed=False, future_utility=0.0, rollback_cost=256)

    assert ledger.attempt_count() == 3
    assert ledger.committed_count() == 2
    assert ledger.useful_commits() == 2
    assert abs(ledger.total_utility() - 0.75) < 1e-9
    assert abs(ledger.utility_per_commit() - 0.375) < 1e-9
    assert abs(ledger.utility_per_written_byte() - 0.75 / 128) < 1e-12
    assert abs(ledger.utility_per_compute_unit() - 0.75 / 244) < 1e-12
    assert abs(ledger.utility_per_wall_time() - 0.375) < 1e-9
    # A refused topology revision is not counted as spent structure.
    assert ledger.granularity_mix()["topology_revision"] == 0
    assert ledger.topology_share() == 0.0

    # An arm that spends topology on everything fails the misallocation audit.
    greedy = UpdateLedger(arm="greedy")
    for index in range(4):
        greedy.record(UpdateRecord(f"g{index}", "greedy", "topology_revision", bytes_written=256))
        greedy.settle(f"g{index}", committed=True, future_utility=0.01)
    audit = misallocation_audit([ledger, greedy])
    assert audit["arms"]["demo"]["within_ceiling"] is True
    assert audit["arms"]["greedy"]["within_ceiling"] is False
    assert audit["all_pass"] is False

    # The ladder: cheapest sufficient change, and topology only for a real exception.
    plain = Evidence(1, 1, 0, False, False, 0.0, True)
    assert allocate(plain) == "micro_association"
    repeated = Evidence(3, 1, 0, False, False, 0.0, False)
    assert allocate(repeated) == "association_promotion"
    supported = Evidence(1, 1, 6, False, False, 0.0, True)
    assert allocate(supported) == "structural_consolidation"
    exception = Evidence(1, 2, 6, True, True, 0.0, False)
    assert allocate(exception) == "topology_revision"
    # A contradiction with no fitted rule is still not worth topology.
    early_contradiction = Evidence(1, 2, 1, False, True, 0.0, False)
    assert allocate(early_contradiction) == "micro_association"
    pressed = Evidence(1, 1, 0, False, False, 0.9, False)
    assert allocate(pressed) == "local_low_bit_adjustment"

    # The oracle prefers the cheaper granularity when two paid the same.
    assert allocate_oracle(plain, paid={"micro_association": 0.5, "topology_revision": 0.5}) == "micro_association"

    print("genesis2 ledger self-check passed: seven reports, five allocators, misallocation audit")


if __name__ == "__main__":
    demo()
