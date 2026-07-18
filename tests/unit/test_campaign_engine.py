"""Tests for the unified campaign engine: specs, DAG frontier, decisions, null-safe skipping, durable
state and lease recovery, broker admission, receipt-invariance, coverage, the atlas, and the compliance
verifier. These exercise the engine without disturbing any live campaign.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import tempfile

import pytest

from mop.campaign.broker import ResourceBroker
from mop.campaign.dag import AuthorityResolver, refresh_eligibility, runnable_frontier
from mop.campaign.decisions import evaluate_decisions, predicate_matches, resolve_skips
from mop.campaign.invariance import run_invariance_sweep
from mop.campaign.manifest import build_campaign
from mop.campaign.specs import (
    BedSpec,
    CampaignSpec,
    DecisionRule,
    Dependency,
    DependencyKind,
    ReproductionSpec,
    ResearchQuestionSpec,
    ResourceClass,
    ResourceRequest,
    SpecError,
    VerificationSpec,
)
from mop.campaign.state import CampaignState, Lease, NodeStatus


def _tiny_campaign() -> CampaignSpec:
    nodes = [
        BedSpec(node_id="bed", title="b", entrypoint="m:f"),
        ResearchQuestionSpec(node_id="arm_a", title="a", entrypoint="m:f", dependencies=(Dependency("bed"),)),
        ResearchQuestionSpec(node_id="arm_b", title="b", entrypoint="m:f", dependencies=(Dependency("bed"),)),
        VerificationSpec(node_id="verify", title="v", entrypoint="m:f",
                         dependencies=(Dependency("arm_a", DependencyKind.SEAL),
                                       Dependency("arm_b", DependencyKind.SEAL)),
                         decision_rules=(DecisionRule("survives", {"field": "verdict", "equals": "survives"},
                                                      ("repro",)),)),
        ReproductionSpec(node_id="repro", title="r", entrypoint="m:f",
                         authorities=("branch:verify:survives",)),
        BedSpec(node_id="ext", title="e", entrypoint="m:f", blocked_reason="needs external data"),
    ]
    return CampaignSpec(campaign_id="t", title="t", nodes=tuple(nodes))


def test_spec_validation_rejects_bad_refs():
    with pytest.raises(SpecError):
        CampaignSpec(campaign_id="x", title="x",
                     nodes=(BedSpec(node_id="a", title="a", entrypoint="m:f",
                                    dependencies=(Dependency("missing"),)),))
    with pytest.raises(SpecError):
        BedSpec(node_id="a", title="a", entrypoint="no_colon")


def test_frontier_dependencies_and_concurrency():
    camp = _tiny_campaign()
    with tempfile.TemporaryDirectory() as d:
        st = CampaignState(d, "t", [n.node_id for n in camp.nodes])
        res = AuthorityResolver()
        refresh_eligibility(camp, st, res)
        assert [n.node_id for n in runnable_frontier(camp, st, res)] == ["bed"]
        assert st.status("ext") is NodeStatus.BLOCKED
        st.mark_sealed("bed", "p", "s", "ok", False)
        refresh_eligibility(camp, st, res)
        assert sorted(n.node_id for n in runnable_frontier(camp, st, res)) == ["arm_a", "arm_b"]


def test_decision_branch_and_null_safe_skip():
    camp = _tiny_campaign()
    res = AuthorityResolver()
    with tempfile.TemporaryDirectory() as d:
        st = CampaignState(d, "t", [n.node_id for n in camp.nodes])
        for nid in ("bed", "arm_a", "arm_b"):
            st.mark_sealed(nid, "p", "s", "ok", False)
        evaluate_decisions(camp, camp.node("verify"), {"verdict": "survives"}, st)
        st.mark_sealed("verify", "p", "s", "survives", False)
        refresh_eligibility(camp, st, res)
        assert [n.node_id for n in runnable_frontier(camp, st, res)] == ["repro"]
    with tempfile.TemporaryDirectory() as d:
        st = CampaignState(d, "t", [n.node_id for n in camp.nodes])
        for nid in ("bed", "arm_a", "arm_b"):
            st.mark_sealed(nid, "p", "s", "ok", False)
        evaluate_decisions(camp, camp.node("verify"), {"verdict": "null"}, st)
        st.mark_sealed("verify", "p", "s", "null", True)
        assert "repro" in resolve_skips(camp, st)
        assert st.status("repro") is NodeStatus.SKIPPED


def test_predicate_operators():
    assert predicate_matches({"field": "v", "op": ">=", "value": 0.5}, {"v": 0.5})
    assert not predicate_matches({"field": "v", "op": ">", "value": 0.5}, {"v": 0.5})
    assert predicate_matches({"field": "v", "in": ["a", "b"]}, {"v": "b"})
    assert predicate_matches({"always": True}, {})


def test_durable_state_and_stale_lease_recovery():
    camp = _tiny_campaign()
    with tempfile.TemporaryDirectory() as d:
        st = CampaignState(d, "t", [n.node_id for n in camp.nodes])
        st.mark_running("bed", pid=999999, create_time=1.0)
        st.records["bed"].lease = Lease(pid=999999, create_time=1.0, started_at=1.0, heartbeat_at=1.0)
        st.save()
        st2 = CampaignState(d, "t", [n.node_id for n in camp.nodes])
        st2.load()
        assert st2.status("bed") is NodeStatus.RUNNING
        recovered = st2.recover_stale_leases()  # pid 999999 is dead -> recovered
        assert "bed" in recovered
        assert st2.status("bed") is NodeStatus.ELIGIBLE


def test_broker_admission_and_exclusive():
    broker = ResourceBroker()
    # a fake snapshot so the test does not depend on the live host
    from mop.campaign.broker import BrokerSnapshot

    broker._snapshot = BrokerSnapshot(cpu_budget=2, cpu_in_use=0, mem_available_gb=100.0, mem_in_use_gb=0.0,
                                      hawking_active=False, nice_level=5, external_consumers=0,
                                      recommended_workers=2, binding_constraint="ceiling", created_at=0.0)
    light = ResourceRequest(resource_class=ResourceClass.CPU_LIGHT, cpu_slots=1)
    excl = ResourceRequest(resource_class=ResourceClass.EXCLUSIVE, cpu_slots=8, exclusive=True)
    assert broker.admit(light, "n1", []) is not None  # first admits
    assert broker.admit(light, "n3", [light, light]) is None  # over budget of 2
    assert broker.admit(excl, "e", []) is not None  # exclusive admits on a clear fleet regardless of budget
    assert broker.admit(excl, "e", [light]) is None  # exclusive waits for a clear fleet
    assert broker.admit(light, "n", [excl]) is None  # nothing runs alongside an exclusive node


def test_receipt_invariance_across_widths():
    result = run_invariance_sweep(n_items=800, widths=[1, 2, 4], seed=123)
    assert result["receipt_invariant"] is True
    assert result["receipt"] is not None
    assert len({r["receipt"] for r in result["per_width"]}) == 1  # identical receipt at every width


def test_manifest_builds_and_covers_breadth():
    camp = build_campaign()
    forms = {n.coverage.form_family for n in camp.nodes if not n.is_blocked
             and n.coverage.form_family not in ("none", "cross")}
    phenomena = {n.coverage.phenomenon for n in camp.nodes if not n.is_blocked and n.coverage.phenomenon != "none"}
    assert len(forms) >= 6
    assert len(phenomena) >= 10
    assert len([n for n in camp.nodes if n.is_blocked]) >= 16  # contracted external families
    assert len(camp.external_dependencies) >= 1  # the live run is adopted


def test_atlas_generates_full_rows():
    from mop.campaign.atlas import build_atlas

    atlas = build_atlas()
    assert atlas["n_rows"] == 30 or atlas["n_rows"] >= 30
    required = {"estimand" if False else "target_capability", "sesoi", "verifier_strategy",
                "alternative_explanation", "substrate_consequence_if_null", "current_evidence_level"}
    for row in atlas["rows"]:
        assert required <= set(row.keys())


def test_compliance_verifier_runs():
    from mop.campaign.compliance import build_ledger

    ledger = build_ledger()
    assert ledger["n_requirements"] == 13
    ids = {r["req_id"] for r in ledger["requirements"]}
    assert {"R-DAG-1", "R-BROKER-1", "R-LOCAL-24", "R-LAUNCH-1"} <= ids
    # the engine requirements must verify as implemented (they check real importable code)
    by_id = {r["req_id"]: r for r in ledger["requirements"]}
    assert by_id["R-DAG-1"]["status"] == "implemented"
    assert by_id["R-BROKER-1"]["status"] == "implemented"
    assert by_id["R-FRAMEWORK-1"]["status"] == "implemented"
