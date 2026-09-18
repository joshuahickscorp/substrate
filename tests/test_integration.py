"""Unrun contracts for the one-Odyssey target, cognitive loop and physical forms.

All signatures below use artificial fixtures. They are not study authorization,
consciousness evidence, real organ qualifications or measured performance.
"""
import json
import struct
from dataclasses import asdict

import pytest

from conftest import HOST, INPUTS, SCOPE, certificate, experience, native_skill
from substrate.core import Refused, attest, clone, sha
from substrate.language import abstract_material, compile_program, make_program, transfer_material
from substrate.handoff import (NativeImage, growth, lower_native_regions, phenotype, portable_organization,
                               safetensors_manifest)
from substrate.odyssey import ARMS, draft_plan, validate_plan
from substrate.research import (AXES, TARGET, OdysseyLedger, program_plan, propose_inquiry, study_profile,
                                validate_program, apprenticeship_step)
from substrate.workspace import CognitiveCycle, Perspective, Workspace, WorkspacePolicy, attribute_action, contradictions


def public_case(x=7):
    return {"case_id": sha(["workspace-case", x]), "family": "affine", "domain": "micro-world", "split": "train",
            "input": {"x": x}, "scope": SCOPE, "source": sha("fixture-source")}


def perspective(value, *, viewpoint="fixture", stance="inferred", channel="world"):
    return Perspective(channel, "fixture", viewpoint, sha(SCOPE), "location", value, stance, sha("witness"))


def test_one_odyssey_keeps_the_requested_target():
    charter = program_plan()
    assert charter["target"] == TARGET
    assert charter["id"] == "substrate-odyssey" and "closure" in charter["chapters"]
    assert validate_program(charter) == charter


@pytest.mark.parametrize("field,value", [("id", "odyssey-two"), ("target", "memory-agent"),
                                         ("completion", "all jobs done")])
def test_charter_cannot_silently_lower_the_bar(field, value):
    charter = program_plan()
    charter[field] = value
    with pytest.raises(Refused):
        validate_program(charter)


def test_legacy_study_is_not_retroactively_relabelled():
    plan = draft_plan()
    assert study_profile(plan)["target"] == TARGET
    del plan["cognition"]
    assert not study_profile(validate_plan(plan))["enabled"]
    assert study_profile(plan)["target"] == "legacy-mechanism-study"


def test_main_has_separate_workspace_interventions():
    plan = draft_plan(kind="main")
    for arm in ("no-workspace", "no-recurrence", "no-perspective", "no-broadcast"):
        assert arm in plan["arms"] and ARMS[arm]["lesion"] == arm


def test_viewpoint_and_scope_prevent_false_contradiction():
    assert contradictions([perspective(1), perspective(2, viewpoint="other")]) == []
    assert len(contradictions([perspective(1), perspective(2)])) == 1


def test_imagining_does_not_contradict_observed_fact():
    assert contradictions([perspective(1, stance="observed"), perspective(2, stance="imagined")]) == []


def test_workspace_selects_within_capacity_and_reports_dropped():
    space = Workspace(WorkspacePolicy(slots=1))
    bids = [perspective(1), perspective(2)]
    assert len(space.publish(bids)) == 1
    assert len(space.frames[0]["dropped"]) == 1 and space.frames[0]["bytes"] <= space.policy.byte_limit


def test_broadcast_ablation_has_no_consumer_access():
    space = Workspace(WorkspacePolicy(lesion="no-broadcast"))
    assert space.publish([perspective(1)]) == []
    assert space.read("organ") == []


def test_self_ablation_removes_self_representation():
    space = Workspace(WorkspacePolicy(lesion="no-self"))
    assert all(p.channel != "self" for p in space.publish([perspective(1, channel="self"), perspective(2)]))


def test_recurrent_critique_changes_route_not_just_trace(actor, key):
    native_skill(actor, key)
    public = public_case(10_000_001)  # Outside the independently qualified program's input guard.
    actor.observe(public)
    cycle = CognitiveCycle(actor, public, policy=WorkspacePolicy(), flags=ARMS["native-only"])
    assert cycle.decide().kind == "abstain"
    assert cycle.reason == "broadcast-critique-revised-routing"


def test_recurrence_lesion_prevents_second_round_revision(actor, key):
    native_skill(actor, key)
    public = public_case(10_000_001)
    actor.observe(public)
    cycle = CognitiveCycle(actor, public, policy=WorkspacePolicy(lesion="no-recurrence"), flags=ARMS["native-only"])
    assert cycle.decide().kind == "skill"  # Runner still enforces the input guard; no unsafe execution.
    assert len(cycle.workspace.frames) == 1


def test_verified_cycle_feedback_is_prospective_and_replay_safe(actor, key):
    public = public_case()
    observed = actor.observe(public)
    cycle = CognitiveCycle(actor, public, policy=WorkspacePolicy(), flags=ARMS["native-only"])
    cycle.decide()
    answer = {"answer": 15, "confidence": .5, "mechanism": "fixture"}
    cycle.finish(answer)
    prediction = actor.predict(observed, 15, producer="tester", mechanism="fixture", confidence=.5)
    proposal = {"prediction": prediction, "case_id": public["case_id"], "unit": "u", "actual": 15, "producer": "tester"}
    signed = certificate(key, actor, "experience", proposal, {"outcome_verified": True, "case_id": public["case_id"]})
    exp = actor.assimilate(proposal, signed)
    cycle.assimilate(exp)
    assert actor.get("cognitive-state", "active")["correct"] is True
    with pytest.raises(Refused):
        cycle.assimilate(exp)


def test_frozen_working_cycle_does_not_modify_entity(actor, key):
    native_skill(actor, key)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    before = actor.store.head(actor.id)
    cycle = CognitiveCycle(actor, public_case(), policy=WorkspacePolicy(), flags=ARMS["native-only"])
    cycle.decide()
    cycle.finish({"answer": 15, "confidence": .5, "mechanism": "fixture"})
    assert actor.store.head(actor.id) == before


def test_counterfactual_or_other_view_does_not_become_self(actor):
    cycle = CognitiveCycle(actor, public_case(), policy=WorkspacePolicy(), flags=ARMS["native-only"])
    other = cycle.perspective("world", "expected-outcome", 9, viewpoint="other-agent", stance="imagined")
    self_view = cycle.perspective("world", "expected-outcome", 3, stance="inferred")
    assert other.coordinate != self_view.coordinate and contradictions([other, self_view]) == []


def test_agency_requires_a_verified_control(key, trust):
    intent = {"actor": "fixture", "action": "toggle", "expected": 1}
    outcome = {"actor": "fixture", "action": "toggle", "actual": 1, "intervened": True, "control": 0}
    ev = sha("paired-control")
    signed = attest(key, signer="oracle", producer="fixture", purpose="agency", domain="cognition",
        subject={"intent": intent, "outcome": outcome, "evidence": ev}, evidence=ev,
        findings={"executed": True, "control_verified": True}, ttl=3600)
    result = attribute_action(intent, outcome, evidence=ev, trust=trust, certificate=signed)
    assert result["effect_present"] is True and result["phenomenal_agency"] == "not-inferred"
    with pytest.raises(Refused):
        attribute_action(intent, {**outcome, "actor": "operator"}, evidence=ev, trust=trust, certificate=signed)


def test_acquired_parameter_structure_transfers_to_new_variable():
    a = make_program(INPUTS, ["add", ["mul", ["const", 2], ["var", "x"]], ["const", 1]])
    b = make_program({"y": INPUTS["x"]}, ["add", ["mul", ["const", 1], ["var", "y"]], ["const", 2]])
    materials = abstract_material([a, b])
    assert len(materials) == 1 and materials[0]["status"] == "unqualified"
    result = transfer_material(materials, [{"input": {"z": v}, "output": 2*v-1} for v in (3, 7)],
                               {"z": INPUTS["x"]}, seconds=1)
    assert result["program"] is not None and not result["qualified"]
    assert compile_program(result["program"]).run({"z": 12}) == 23


def test_repeated_identical_program_is_not_two_abstraction_witnesses():
    p = make_program(INPUTS, ["mul", ["const", 2], ["var", "x"]])
    assert abstract_material([p, p]) == []


def test_phenotype_reports_unknown_model_residency_as_unknown(actor, key):
    before = phenotype(actor)
    experience(actor, key, 1)
    after = phenotype(actor)
    delta = growth(before, after)
    assert after["external_organ_bytes"] is None and after["resident_process_bytes"] is None
    assert not after["size_is_intelligence"]
    assert delta["delta"]["experience"]["raw_bytes"] > 0


def test_whole_entity_portable_package_is_not_hawking_compatibility(actor, key, tmp_path):
    native_skill(actor, key)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    result = portable_organization(actor, tmp_path / "engineer.substrate", runtime=sha("runtime"))
    doc = json.loads((tmp_path / "engineer.substrate/organization.nr.json").read_text())
    assert result["nr_pin"] == sha(doc) and doc["full_entity_included"] is True
    assert doc["hawking_abi_qualified"] is False and doc["contains_credentials"] is False
    assert (tmp_path / "engineer.substrate/entity.bundle").exists()


def test_portable_export_requires_checkpoint_freeze(actor, tmp_path):
    with pytest.raises(Refused):
        portable_organization(actor, tmp_path / "entity", runtime=sha("runtime"))


def test_native_image_executes_only_after_independent_qualification(actor, key, trust, tmp_path):
    skill = native_skill(actor, key)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    runtime = sha("runtime")
    exported = portable_organization(actor, tmp_path / "native-source", runtime=runtime)
    nr = exported["nr_pin"]
    source = json.loads((tmp_path / "native-source/organization.nr.json").read_text())
    image = lower_native_regions(actor, nr=source, nr_pin=nr, runtime=runtime, machine=HOST)
    pin = sha(image)
    subject = {"image": pin, "nr": nr, "machine": HOST, "runtime": runtime}
    signed = attest(key, signer="oracle", producer="native-lowerer", purpose="native-image", domain="cognition",
        subject=subject, evidence=sha("native-canary"),
        findings={"executed": True, "fidelity_passed": True, "scope": sha(sorted(image["regions"]))}, ttl=3600)
    loaded = NativeImage(image, pin=pin, runtime=runtime, machine=HOST, trust=trust, certificate=signed)
    assert loaded.run(skill, {"x": 12}, family="affine", scope=SCOPE) == 25
    with pytest.raises(Refused):
        NativeImage(image, pin=pin, runtime=runtime, machine=sha("other-host"), trust=trust, certificate=signed)


def tensor_bytes(header, payload=b"\x00" * 4):
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw + payload


def test_tensor_inventory_counts_actual_elements_not_intelligence(tmp_path):
    path = tmp_path / "counts.safetensors"
    path.write_bytes(tensor_bytes({"counts": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}))
    inv = safetensors_manifest(path)
    assert inv["elements"] == 1 and "not automatically" in inv["semantics"]


@pytest.mark.parametrize("offsets,shape", [([0, 8], [1]), ([1, 4], [1]), ([0, 4], [2])])
def test_tensor_inventory_refuses_misdeclared_payload(tmp_path, offsets, shape):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(tensor_bytes({"x": {"dtype": "F32", "shape": shape, "data_offsets": offsets}}))
    with pytest.raises(Refused):
        safetensors_manifest(path)


def test_empty_program_ledger_does_not_award_consciousness(actor, trust):
    ledger = OdysseyLedger(actor.store, trust, program_plan())
    result = ledger.assessment()
    assert set(result["missing_axes"]) == set(AXES)
    assert result["phenomenal_consciousness"] == "undetermined"
    assert result["target"] == TARGET


def test_proposed_inquiry_is_not_tool_authorization(actor):
    inquiry = propose_inquiry(actor, "affine", scope=SCOPE, source_allowlist=[sha("fixture-source")],
                              budget={"actions": 2, "seconds": 60})
    assert actor.get("inquiries", inquiry)["status"] == "proposed"
    with pytest.raises(Refused):
        apprenticeship_step(actor, inquiry, {"kind": "read", "arguments": {}}, public_case(), 3, confidence=.5)


def test_organization_roundtrip_preserves_frozen_identity(actor, key, trust, tmp_path):
    from substrate.handoff import restore_organization
    from substrate.store import Store
    native_skill(actor, key)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    directory, runtime = tmp_path / "portable", sha("runtime")
    exported = portable_organization(actor, directory, runtime=runtime)
    with Store(tmp_path / "restored.sqlite", create=True) as destination:
        result = restore_organization(destination, directory, nr_pin=exported["nr_pin"], runtime=runtime)
        assert result["entity"] == actor.id and destination.head(actor.id) == actor.store.head(actor.id)
        assert destination.entity(actor.id)["frozen"] and not result["host_authority_inherited"]


def test_no_workspace_is_distinct_from_unbroadcast_private_selection():
    absent = Workspace(WorkspacePolicy(lesion="no-workspace"))
    private = Workspace(WorkspacePolicy(lesion="no-broadcast"))
    assert absent.publish([perspective(1)]) == private.publish([perspective(1)]) == []
    assert absent.frames[0]["selected"] == [] and private.frames[0]["selected"] != []


def test_workspace_cannot_grow_past_recurrence_budget():
    space = Workspace(WorkspacePolicy(rounds=1))
    space.publish([perspective(1)])
    with pytest.raises(Refused):
        space.publish([perspective(2)])
