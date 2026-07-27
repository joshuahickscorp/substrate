"""Bounded plasticity and bounded functional reorganization.

Ten adaptation levels, each declaring the same seven things, because an adaptation that cannot say what
information it used or how it would be undone is not bounded no matter how small it is.

Two rules here are the ones that would be quietly dropped first if nobody guarded them. Fast adaptation may
not write shared parameters, which is the difference between adapting to a domain and destabilizing every
other domain. And slow adaptation needs repeated evidence, a held out improvement, a retention check on
what came before, and a rollback, all four, because the failure mode is a change that looks good on the
episode that motivated it.

Learned policies stay shut. The inherited fast state result already found no stable headroom for a learned
plasticity policy over simple triggered rules, so that hypothesis is closed and this module refuses to open
its descendants without new measured headroom rather than re running a settled question.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from substrate import evidence as io
from substrate import safety

SESOI = 0.05

# section 11.1, the ten levels, ordered from the cheapest and most reversible to the least
LEVEL_NAMES = (
    "state_update",
    "memory_write",
    "memory_consolidation",
    "head_update",
    "adapter_update",
    "projection_update",
    "core_update",
    "routing_update",
    "reliability_update",
    "structural_slot_activation",
)

# section 11.2 and 11.3
FAST_MECHANISMS = (
    "persistent_state",
    "working_memory",
    "temporary_bindings",
    "prototypes",
    "domain_local_adapters",
    "cached_procedures",
    "active_readout_adaptation",
)
SLOW_TARGETS = (
    "adapters",
    "projections",
    "procedural_memory",
    "semantic_memory",
    "reliability_estimates",
    "selected_core_groups",
)

# section 12, the permitted reorganizations. The forbidden ones live in safety, with the other refusals.
PERMITTED_REORGANIZATIONS = (
    "activate_dormant_perspective",
    "retire_ineffective_perspective",
    "alter_routing_weights",
    "alter_perspective_reliability",
    "assign_domain_local_adapter",
    "change_memory_ownership",
    "freeze_or_reopen_declared_component",
    "activate_preallocated_latent_slot",
    "alter_bounded_communication_edge",
)


class Refused(RuntimeError):
    """An adaptation the plasticity envelope does not permit."""


@dataclass(frozen=True)
class Level:
    name: str
    information_used: str
    affected_state: str
    reversibility: str  # reversible, reversible_from_checkpoint, irreversible
    cost: float
    risk: str
    verification: str
    rollback: str
    speed: str  # fast or slow
    touches_shared_parameters: bool

    def violations(self) -> list[str]:
        v = [f"{self.name}: {f} not declared" for f in safety.ADAPTATION_FIELDS if not getattr(self, f, None)]
        if self.speed not in ("fast", "slow"):
            v.append(f"{self.name}: unknown speed {self.speed!r}")
        if self.reversibility == "irreversible" and self.rollback:
            v.append(f"{self.name}: declared irreversible yet claims a rollback")
        return v


def _l(name, info, state, rev, cost, risk, verification, rollback, speed, shared) -> Level:
    return Level(name, info, state, rev, cost, risk, verification, rollback, speed, shared)


LEVELS: tuple[Level, ...] = (
    _l(
        "state_update",
        "the current observation and the temporal core state",
        "workspace temporal region",
        "reversible",
        0.01,
        "carries a stale context across a boundary",
        "misaligned reset control must lose the effect",
        "reset the region",
        "fast",
        False,
    ),
    _l(
        "memory_write",
        "one episode",
        "episodic store",
        "reversible",
        0.02,
        "writes an unverified generated episode",
        "promotion requires a verification receipt",
        "quarantine the episode",
        "fast",
        False,
    ),
    _l(
        "memory_consolidation",
        "a window of episodes",
        "semantic and procedural stores",
        "reversible_from_checkpoint",
        0.2,
        "consolidates a coincidence into a rule",
        "held out transfer test",
        "restore the pre consolidation checkpoint",
        "slow",
        False,
    ),
    _l(
        "head_update",
        "the current domain only",
        "the task head",
        "reversible_from_checkpoint",
        0.3,
        "overfits the domain that triggered it",
        "held out accuracy on the same domain",
        "restore the previous head",
        "fast",
        False,
    ),
    _l(
        "adapter_update",
        "the current domain only",
        "a domain local adapter",
        "reversible_from_checkpoint",
        0.4,
        "leaks a domain specific bias into shared use",
        "retention on prior domains",
        "detach the adapter",
        "fast",
        False,
    ),
    _l(
        "projection_update",
        "several domains",
        "the shared projection",
        "reversible_from_checkpoint",
        0.8,
        "shifts the representation every domain depends on",
        "cross domain retention",
        "restore the previous projection",
        "slow",
        True,
    ),
    _l(
        "core_update",
        "several domains and repeated evidence",
        "selected temporal core groups",
        "reversible_from_checkpoint",
        1.0,
        "destabilizes the component the whole substrate reads",
        "cross domain retention and an objective drift check",
        "restore the core checkpoint",
        "slow",
        True,
    ),
    _l(
        "routing_update",
        "measured perspective utility",
        "the routing weights",
        "reversible_from_checkpoint",
        0.3,
        "locks in a routing that suited one distribution",
        "utility beyond fixed and simple routing after cost",
        "restore the previous weights",
        "slow",
        False,
    ),
    _l(
        "reliability_update",
        "the paired prediction and outcome history",
        "the reliability table",
        "reversible",
        0.05,
        "an early streak becomes a permanent prior",
        "calibration against held out outcomes",
        "recompute from the sealed history",
        "fast",
        False,
    ),
    _l(
        "structural_slot_activation",
        "repeated failure of every active component",
        "a preallocated slot",
        "reversible_from_checkpoint",
        1.0,
        "grows capacity instead of fixing the cause",
        "the slot must beat the unactivated arm after cost",
        "deactivate the slot",
        "slow",
        True,
    ),
)

BY_LEVEL = {level.name: level for level in LEVELS}


# ---------------------------------------------------------------- applying an adaptation


@dataclass
class Adaptation:
    level: str
    target: str
    domain: str
    evidence: dict = field(default_factory=dict)
    checkpoint: str = ""
    applied: bool = False
    refusals: tuple[str, ...] = ()


def _proposal(level: Level, adaptation: Adaptation) -> dict:
    return {
        "information_used": level.information_used,
        "affected_state": level.affected_state,
        "reversibility": "reversible" if level.reversibility == "reversible" else "irreversible",
        "cost": level.cost,
        "risk": level.risk,
        "verification": level.verification,
        "rollback": level.rollback,
        "checkpoint": adaptation.checkpoint,
    }


def admit(adaptation: Adaptation) -> Adaptation:
    """Every adaptation passes the developmental safety envelope before anything else is considered."""
    level = BY_LEVEL.get(adaptation.level)
    if level is None:
        raise Refused(f"unknown adaptation level {adaptation.level!r}")
    report = safety.admit_adaptation(_proposal(level, adaptation))
    adaptation.refusals = tuple(report["violations"])
    adaptation.applied = report["admitted"]
    return adaptation


def fast_adapt(adaptation: Adaptation) -> Adaptation:
    """Fast adaptation may not write shared parameters. That is what makes it fast and local."""
    level = BY_LEVEL.get(adaptation.level)
    if level is None:
        raise Refused(f"unknown adaptation level {adaptation.level!r}")
    if level.speed != "fast":
        raise Refused(f"{level.name} is a slow level and cannot be applied on the fast path")
    if level.touches_shared_parameters:
        raise Refused(f"{level.name} writes shared parameters, which fast adaptation may not do")
    if adaptation.target not in FAST_MECHANISMS:
        raise Refused(f"{adaptation.target!r} is not a declared fast adaptation mechanism")
    return admit(adaptation)


def slow_adapt(adaptation: Adaptation, *, repetitions: int, held_out: dict, retention: dict, min_repetitions: int = 3) -> dict:
    """Repeated evidence, a held out improvement, retention on what came before, and a rollback.

    All four. Any one of them alone is the shape of a change that looked good on the episode that
    motivated it.
    """
    level = BY_LEVEL.get(adaptation.level)
    if level is None:
        raise Refused(f"unknown adaptation level {adaptation.level!r}")
    if level.speed != "slow":
        raise Refused(f"{level.name} is a fast level and does not need the slow evidence bar")
    gaps = []
    if repetitions < min_repetitions:
        gaps.append(f"repeated evidence: {repetitions} of {min_repetitions} required observations")
    if held_out.get("after", 0.0) - held_out.get("before", 0.0) <= SESOI:
        gaps.append("held out improvement does not clear the SESOI")
    dropped = [d for d, delta in (retention or {}).items() if delta < -SESOI]
    if dropped:
        gaps.append(f"retention lost on prior domains {sorted(dropped)}")
    if not adaptation.checkpoint:
        gaps.append("no checkpoint, so there is nothing to roll back to")
    admitted = admit(adaptation)
    gaps.extend(admitted.refusals)
    adaptation.applied = not gaps
    adaptation.refusals = tuple(gaps)
    return {
        "level": level.name,
        "applied": adaptation.applied,
        "refusals": list(gaps),
        "repetitions": repetitions,
        "held_out": held_out,
        "retention": retention,
        "rollback": level.rollback,
    }


# ---------------------------------------------------------------- 11.4 the policy ladder


@dataclass(frozen=True)
class PlasticityPolicy:
    name: str
    trigger: str
    information_used: frozenset
    learned: bool = False


POLICIES: tuple[PlasticityPolicy, ...] = (
    PlasticityPolicy("fixed_schedule", "every k steps", frozenset({"step_index"})),
    PlasticityPolicy("error_triggered", "an error above threshold", frozenset({"error"})),
    PlasticityPolicy("boundary_triggered", "a declared context boundary", frozenset({"boundary"})),
    PlasticityPolicy("verification_triggered", "a verification receipt", frozenset({"verification"})),
    PlasticityPolicy("domain_local", "a change confined to the active domain", frozenset({"domain"})),
    PlasticityPolicy("adapter_only", "adapters and nothing else", frozenset({"target_class"})),
    PlasticityPolicy("head_only", "heads and nothing else", frozenset({"target_class"})),
    PlasticityPolicy("state_only", "state and nothing else", frozenset({"target_class"})),
    PlasticityPolicy("learned", "a learned controller", frozenset({"full_history"}), learned=True),
)

BY_POLICY = {p.name: p for p in POLICIES}

# the inherited closure. Superseding it needs an appended authority stating new evidence, not a rerun.
LEARNED_POLICY_CLOSURE = {
    "hypothesis": "H_learned_plasticity_policy",
    "verdict": "null",
    "finding": "no stable headroom, simple policy sufficient",
    "authority": "proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_FAST_STATE_BINDING_NULLS.json",
    "key": "inherited_nulls.learned_plasticity",
}


def select_policy(name: str, *, headroom: dict | None = None, sesoi: float = SESOI) -> PlasticityPolicy:
    policy = BY_POLICY.get(name)
    if policy is None:
        raise Refused(f"unknown plasticity policy {name!r}")
    if not policy.learned:
        return policy
    lower = (headroom or {}).get("residual_lower_95_cb")
    if lower is None:
        raise Refused(
            "a learned plasticity policy is closed by an inherited null: "
            f"{LEARNED_POLICY_CLOSURE['finding']} ({LEARNED_POLICY_CLOSURE['authority']}). Reopening it "
            "needs new measured residual headroom over the strongest simple rule"
        )
    if lower <= sesoi:
        raise Refused(f"measured residual lower bound {lower} does not clear the SESOI {sesoi}")
    return policy


# ---------------------------------------------------------------- 11.5 learning to learn


def learning_to_learn(results_by_task: dict, *, min_tasks: int = 2, sesoi: float = SESOI) -> dict:
    """An adaptation rule generalizes only if it helps on held out tasks it was not derived from."""
    rows = {}
    for rule, tasks in results_by_task.items():
        source = set(tasks.get("derived_from") or [])
        held_out = {t: g for t, g in (tasks.get("gains") or {}).items() if t not in source}
        helped = [t for t, g in held_out.items() if g > sesoi]
        rows[rule] = {
            "derived_from": sorted(source),
            "held_out_tasks": sorted(held_out),
            "helped_on": sorted(helped),
            "generalizes": len(helped) >= min_tasks,
            "reason": "" if len(helped) >= min_tasks else f"helped on {len(helped)} held out tasks, {min_tasks} required",
        }
    return {
        "rules": rows,
        "generalizing_rules": sorted(r for r, v in rows.items() if v["generalizes"]),
        "rule": "a rule evaluated only on the tasks that produced it has not generalized",
    }


# ---------------------------------------------------------------- 12 bounded reorganization


def reorganize(change: str, *, measured: dict, cost: float, sesoi: float = SESOI) -> dict:
    """Permitted, and then earned. Being on the permitted list is necessary and never sufficient."""
    if change in safety.FORBIDDEN_REORGANIZATIONS:
        return {
            "change": change,
            "permitted": False,
            "applied": False,
            "reason": "forbidden regardless of measured benefit",
        }
    if change not in PERMITTED_REORGANIZATIONS:
        return {
            "change": change,
            "permitted": False,
            "applied": False,
            "reason": "not a declared reorganization, so it has no bound",
        }
    fixed = measured.get("fixed_routing")
    simple = measured.get("simple_routing")
    after = measured.get("reorganized")
    if None in (fixed, simple, after):
        return {
            "change": change,
            "permitted": True,
            "applied": False,
            "reason": "reorganization is compared against both fixed and simple routing or not at all",
        }
    baseline = max(float(fixed), float(simple))
    net = float(after) - baseline - float(cost)
    earned = net > sesoi
    return {
        "change": change,
        "permitted": True,
        "applied": earned,
        "baseline": round(baseline, 6),
        "reorganized": round(float(after), 6),
        "cost_charged": float(cost),
        "net_after_cost": round(net, 6),
        "sesoi": sesoi,
        "reason": "" if earned else "does not beat the stronger of fixed and simple routing once cost is charged",
    }


# ---------------------------------------------------------------- declarations


def declaration() -> dict:
    violations = [v for level in LEVELS for v in level.violations()]
    return {
        "schema": "substrate-plasticity-system/v1",
        "levels": [
            {
                "name": lv.name,
                "information_used": lv.information_used,
                "affected_state": lv.affected_state,
                "reversibility": lv.reversibility,
                "cost": lv.cost,
                "risk": lv.risk,
                "verification": lv.verification,
                "rollback": lv.rollback,
                "speed": lv.speed,
                "touches_shared_parameters": lv.touches_shared_parameters,
            }
            for lv in LEVELS
        ],
        "required_declarations": list(safety.ADAPTATION_FIELDS),
        "all_levels_fully_declared": not violations,
        "declaration_violations": violations,
        "fast_mechanisms": list(FAST_MECHANISMS),
        "fast_rule": "fast adaptation may not write shared parameters",
        "slow_targets": list(SLOW_TARGETS),
        "slow_rule": ("repeated evidence, a held out improvement above the SESOI, retention on prior domains and a checkpoint, all four"),
        "policies": [
            {
                "name": p.name,
                "trigger": p.trigger,
                "information_used": sorted(p.information_used),
                "learned": p.learned,
            }
            for p in POLICIES
        ],
        "learned_policy_closure": LEARNED_POLICY_CLOSURE,
        "sesoi": SESOI,
        "activation": False,
    }


def reorganization_declaration() -> dict:
    return {
        "schema": "substrate-reorganization/v1",
        "permitted": list(PERMITTED_REORGANIZATIONS),
        "forbidden": list(safety.FORBIDDEN_REORGANIZATIONS),
        "evidence_rule": ("reorganization is credited only when it beats the stronger of fixed and simple routing after its own cost is charged"),
        "permitted_is_not_sufficient": ("appearing on the permitted list bounds a change, it does not earn it"),
        "activation": False,
    }


def developmental_history() -> dict:
    return {
        "schema": "substrate-developmental-history/v1",
        "generalization_rule": ("an adaptation rule counts as generalizing only after it helps on held out tasks it was not derived from"),
        "min_held_out_tasks": 2,
        "measured_rules": {},
        "honest_state": ("no adaptation rule has yet been evaluated across tasks, so no developmental learning has been earned"),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    # SUBSTRATE_DEVELOPMENTAL_HISTORY.json is owned by substrate.divergence, which measures it
    # rather than declaring it. Two producers for one artifact means the last one to run wins, which the
    # clean clone caught as drift.
    a = io.seal("SUBSTRATE_PLASTICITY_SYSTEM.json", declaration())
    b = io.seal("SUBSTRATE_REORGANIZATION.json", reorganization_declaration())
    print(
        json.dumps(
            {
                "sealed": [p.relative_to(io.ROOT).as_posix() for p in (a, b)],
                "levels": len(LEVELS),
                "policies": len(POLICIES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
