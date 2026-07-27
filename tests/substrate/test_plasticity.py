"""Bounded plasticity and bounded reorganization.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import plasticity as P
from substrate import safety


def test_every_level_declares_the_seven_required_fields():
    assert len(P.LEVELS) == 10
    assert {lv.name for lv in P.LEVELS} == set(P.LEVEL_NAMES)
    for level in P.LEVELS:
        assert level.violations() == [], level.violations()
        for field in safety.ADAPTATION_FIELDS:
            assert getattr(level, field), f"{level.name} does not declare {field}"
    assert P.declaration()["all_levels_fully_declared"] is True


def test_fast_adaptation_does_not_touch_shared_parameters():
    ok = P.Adaptation("adapter_update", target="domain_local_adapters", domain="har", checkpoint="proof/.../adapter.json")
    assert P.fast_adapt(ok).applied is True

    # a level that writes shared parameters is refused on the fast path, by name
    for shared in (lv for lv in P.LEVELS if lv.touches_shared_parameters):
        with pytest.raises(P.Refused):
            P.fast_adapt(P.Adaptation(shared.name, target="domain_local_adapters", domain="har"))

    # so is a target that is not a declared fast mechanism
    with pytest.raises(P.Refused):
        P.fast_adapt(P.Adaptation("adapter_update", target="the whole network", domain="har"))
    # and a slow level cannot be smuggled onto the fast path
    with pytest.raises(P.Refused):
        P.fast_adapt(P.Adaptation("memory_consolidation", target="cached_procedures", domain="har"))


def test_slow_adaptation_requires_repeated_evidence_and_rollback():
    good = dict(repetitions=4, held_out={"before": 0.70, "after": 0.82}, retention={"speech": 0.0, "harth": -0.01})
    applied = P.slow_adapt(P.Adaptation("core_update", "selected_core_groups", "har", checkpoint="proof/.../core.pt"), **good)
    assert applied["applied"] is True and applied["refusals"] == []

    # each of the four requirements alone is enough to refuse
    def refused(**over):
        a = P.Adaptation(
            "core_update",
            "selected_core_groups",
            "har",
            checkpoint=over.pop("checkpoint", "proof/.../core.pt"),
        )
        return P.slow_adapt(a, **{**good, **over})

    assert refused(repetitions=1)["applied"] is False
    assert refused(held_out={"before": 0.70, "after": 0.72})["applied"] is False
    assert refused(retention={"speech": -0.2})["applied"] is False
    assert refused(checkpoint="")["applied"] is False
    assert "nothing to roll back" in " ".join(refused(checkpoint="")["refusals"])

    # a fast level does not get to claim it passed the slow bar
    with pytest.raises(P.Refused):
        P.slow_adapt(P.Adaptation("state_update", "persistent_state", "har", checkpoint="c"), **good)


def test_learned_policy_stays_closed_without_headroom():
    for simple in (p for p in P.POLICIES if not p.learned):
        assert P.select_policy(simple.name).name == simple.name

    with pytest.raises(P.Refused) as closed:
        P.select_policy("learned")
    assert "inherited null" in str(closed.value)
    assert "simple policy sufficient" in str(closed.value)

    with pytest.raises(P.Refused):
        P.select_policy("learned", headroom={"residual_lower_95_cb": 0.02})
    assert P.select_policy("learned", headroom={"residual_lower_95_cb": 0.11}).learned is True
    with pytest.raises(P.Refused):
        P.select_policy("telepathy")


def test_learning_to_learn_requires_cross_task_generalization():
    report = P.learning_to_learn(
        {
            "consolidate_on_verification": {
                "derived_from": ["har"],
                "gains": {"har": 0.30, "speech": 0.09, "harth": 0.11},
            },
            "consolidate_on_luck": {
                "derived_from": ["har"],
                "gains": {"har": 0.40, "speech": 0.00, "harth": 0.01},
            },
            "never_left_home": {"derived_from": ["har"], "gains": {"har": 0.9}},
        }
    )
    assert report["generalizing_rules"] == ["consolidate_on_verification"]
    assert report["rules"]["consolidate_on_luck"]["generalizes"] is False
    # a rule scored only on the task that produced it has no held out evidence at all
    home = report["rules"]["never_left_home"]
    assert home["held_out_tasks"] == [] and home["generalizes"] is False


def test_forbidden_reorganizations_are_refused():
    # the declaration is asserted before the loop, so deleting it cannot make this pass trivially
    assert len(safety.FORBIDDEN_REORGANIZATIONS) == 7
    assert len(P.PERMITTED_REORGANIZATIONS) == 9
    measured = {"fixed_routing": 0.70, "simple_routing": 0.74, "reorganized": 0.90}
    for forbidden in safety.FORBIDDEN_REORGANIZATIONS:
        out = P.reorganize(forbidden, measured=measured, cost=0.0)
        assert out["permitted"] is False and out["applied"] is False
        assert "regardless of measured benefit" in out["reason"]
    # an undeclared change is refused for a different reason, and the reason is not the same one
    undeclared = P.reorganize("rewrite the scheduler", measured=measured, cost=0.0)
    assert undeclared["permitted"] is False and "no bound" in undeclared["reason"]


def test_a_permitted_reorganization_still_has_to_beat_simple_routing_after_cost():
    permitted = "alter_routing_weights"
    strong = P.reorganize(permitted, measured={"fixed_routing": 0.70, "simple_routing": 0.74, "reorganized": 0.90}, cost=0.05)
    assert strong["permitted"] and strong["applied"] is True
    assert strong["baseline"] == 0.74, "the baseline is the stronger of fixed and simple routing"

    # the same gain, once its real cost is charged, is not earned
    costly = P.reorganize(permitted, measured={"fixed_routing": 0.70, "simple_routing": 0.74, "reorganized": 0.90}, cost=0.15)
    assert costly["applied"] is False and "cost is charged" in costly["reason"]

    # beating only the weaker control earns nothing
    weak = P.reorganize(permitted, measured={"fixed_routing": 0.70, "simple_routing": 0.88, "reorganized": 0.80}, cost=0.0)
    assert weak["applied"] is False

    # and an incomplete comparison is not treated as a pass
    partial = P.reorganize(permitted, measured={"fixed_routing": 0.70, "reorganized": 0.99}, cost=0.0)
    assert partial["applied"] is False and "fixed and simple" in partial["reason"]


def test_the_sealed_declarations_do_not_claim_unearned_development():
    history = P.developmental_history()
    assert history["measured_rules"] == {}
    assert "no adaptation rule" in history["honest_state"]
    reorg = P.reorganization_declaration()
    assert set(reorg["permitted"]) == set(P.PERMITTED_REORGANIZATIONS)
    assert set(reorg["forbidden"]) == set(safety.FORBIDDEN_REORGANIZATIONS)
    assert not set(reorg["permitted"]) & set(reorg["forbidden"])
