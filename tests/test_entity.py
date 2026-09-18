"""Unrun developmental contracts. Artificial signatures test admission, not scientific validity."""
import base64
import time
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from substrate.cognition import Entity, reliability
from substrate.core import Refused, Trust, TrustRule, attest, sha
from substrate.handoff import hawking_request, training_rows
from substrate.language import make_program
from substrate.odyssey import organ_requirement
from substrate.organs import Organ
from substrate.store import Store
from conftest import HOST, INPUTS, SCOPE, certificate, experience, native_skill


@pytest.mark.parametrize("stance", ["supported", "knowledge", "reopened", "true"])
def test_thinking_cannot_promote_itself(actor, stance):
    with pytest.raises(Refused):
        actor.think("P", domain="micro-world", scope=SCOPE, stance=stance)


@pytest.mark.parametrize("stance", ["considered", "imagined", "believed"])
def test_epistemic_stances_remain_distinct(actor, stance):
    key = actor.think("P", domain="micro-world", scope=SCOPE, stance=stance)
    assert actor.get("beliefs", key)["stance"] == stance
    assert actor.live(key) is False


def supported_belief(actor, key, text="P", dependencies=None):
    belief = actor.think(text, domain="micro-world", scope=SCOPE, dependencies=dependencies)
    proposal = {"belief": belief, "producer": "tester", "confidence": .9, "sources": [sha("source")], "stance": "supported"}
    signed = certificate(key, actor, "belief", proposal, {"supported": True, "scope": sha(SCOPE)})
    actor.support(belief, proposal, signed)
    return belief


def test_independent_support_changes_admissibility(actor, key):
    belief = supported_belief(actor, key)
    assert actor.live(belief)


def test_self_asserted_knowledge_needs_a_domain_standard(actor, key):
    belief = actor.think("P", domain="micro-world", scope=SCOPE)
    proposal = {"belief": belief, "producer": "tester", "confidence": 1, "sources": [sha("source")], "stance": "knowledge"}
    signed = certificate(key, actor, "belief", proposal, {"supported": True, "scope": sha(SCOPE)})
    with pytest.raises(Refused):
        actor.support(belief, proposal, signed)


def test_reopening_propagates_through_beliefs_and_material(actor, key):
    parent = supported_belief(actor, key, "P")
    child = supported_belief(actor, key, "Q depends on P", [parent])
    store = actor.store
    store.commit(actor.id, store.head(actor.id), "material", {}, [("materials", "m", {"dependencies": [child], "status": "qualified"})])
    reopened = actor.reopen(parent, "counterexample", evidence=sha("counterexample"))
    assert set(reopened) == {parent, child}
    assert not actor.live(child)
    assert actor.get("materials", "m")["status"] == "suspended"


def test_revocation_changes_existing_belief_admissibility(actor, key):
    belief = supported_belief(actor, key)
    actor.trust.rules["oracle"] = replace(actor.trust.rules["oracle"], revoked=True)
    assert not actor.live(belief)


def test_key_rotation_does_not_silently_revalidate_old_certificates(actor, key):
    belief = supported_belief(actor, key)
    new_key = Ed25519PrivateKey.generate()
    public = new_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    actor.trust.rules["oracle"] = replace(actor.trust.rules["oracle"], public_key=base64.b64encode(public).decode())
    assert not actor.live(belief)


def test_observation_cannot_admit_heldout_feedback(actor):
    with pytest.raises(Refused):
        actor.observe({"case_id": sha("x"), "domain": "micro-world", "family": "affine", "split": "evaluation",
                       "input": {"x": 1}, "source": sha("source"), "scope": SCOPE})


def test_experience_requires_a_prospective_prediction(actor, key):
    proposal = {"prediction": sha("absent"), "case_id": sha("case"), "unit": "u", "actual": 1, "producer": "tester"}
    signed = certificate(key, actor, "experience", proposal, {"outcome_verified": True, "case_id": proposal["case_id"]})
    with pytest.raises(Refused):
        actor.assimilate(proposal, signed)


def test_replayed_experience_does_not_inflate_competence(actor, key):
    experience(actor, key, 1)
    with pytest.raises(Refused):
        experience(actor, key, 1)
    assert actor.competence("affine")["attempts"] == 1


def test_training_examples_are_bounded(actor, key):
    for x in range(10):
        experience(actor, key, x)
    assert len(actor.examples("affine", maximum=4)) == 4


def test_procedure_requires_distinct_inputs(actor, key):
    witness = experience(actor, key, 1)
    with pytest.raises(Refused):
        actor.propose_skill(make_program(INPUTS, ["var", "x"]), family="affine", domain="micro-world", scope=SCOPE,
                            witnesses=[witness, witness], producer="tester")


def test_program_must_fit_verified_witnesses(actor, key):
    witnesses = [experience(actor, key, x) for x in (1, 2)]
    with pytest.raises(Refused):
        actor.propose_skill(make_program(INPUTS, ["var", "x"]), family="affine", domain="micro-world", scope=SCOPE,
                            witnesses=witnesses, producer="tester")


def test_native_reuse_without_a_model(actor, key):
    skill = native_skill(actor, key)
    assert actor.use(skill, {"x": 100}, family="affine", scope=SCOPE) == 201
    assert actor.route("affine", SCOPE, permit_model=False).kind == "skill"


def test_procedure_is_scope_guarded(actor, key):
    skill = native_skill(actor, key)
    with pytest.raises(Refused):
        actor.use(skill, {"x": 100}, family="affine", scope={"task": "different", "world": "fixture-world"})


def test_host_change_suspends_native_eligibility(actor, key):
    native_skill(actor, key)
    moved = Entity(actor.store, actor.id, actor.trust, host=sha("new-host"))
    assert moved.route("affine", SCOPE, permit_model=False).kind == "abstain"


def test_verified_prediction_failure_suspends_the_actual_skill(actor, key):
    skill = native_skill(actor, key)
    experience(actor, key, 20, expected=41, actual=42, skill=skill)
    assert actor.get("skills", skill)["status"] == "suspended"
    assert actor.route("affine", SCOPE, permit_model=False).kind == "abstain"


def test_unrelated_model_failures_do_not_poison_new_native_routing(actor, key):
    for x in range(10, 14):
        experience(actor, key, x, expected=-999)
    native_skill(actor, key)
    assert actor.route("affine", SCOPE, permit_model=False).kind == "skill"


def test_compilation_ablation_changes_route(actor, key):
    native_skill(actor, key)
    assert actor.route("affine", SCOPE, permit_model=False, compilation=False).kind == "abstain"


def test_projects_survive_organ_attachment(actor):
    actor.project("research", goal="Complete the proof", next_action="Find a counterexample")
    before = actor.get("projects", "research")
    actor.attach_organ({"id": "new-organ", "roles": ["reasoner"], "artifact": sha("model"),
                        "representation": "external-model", "behavior_contract": sha("contract")})
    assert actor.get("projects", "research") == before


def test_organ_declaration_is_not_qualification(actor):
    actor.attach_organ({"id": "organ", "roles": ["reasoner"], "artifact": sha("model"),
                        "representation": "external-model", "behavior_contract": sha("contract")})
    assert actor.route("affine", SCOPE).kind == "abstain"


def test_global_organ_receipt_requires_actual_execution(actor, key):
    requirement = {"id": "organ", "roles": ["reasoner"], "artifact": sha("model"),
                   "representation": "external-model", "behavior_contract": sha("contract")}
    signed = attest(key, signer="oracle", producer="tester", purpose="organ-binding", domain="cognition",
                    subject={"requirement": requirement, "host": HOST}, evidence=sha("e"),
                    findings={"passed": True, "executed": False, "latency_ns": 1000})
    with pytest.raises(Refused):
        actor.bind_organ(requirement, signed, producer="tester")


def test_authorized_continuation_keeps_the_entity_identity(actor, key):
    head = actor.store.freeze(actor.id, actor.store.head(actor.id))
    signed = attest(key, signer="oracle", producer="continuation-requester", purpose="continuity", domain="entity",
                    subject={"entity": actor.id, "head": head, "host": HOST, "operation": "continue"}, evidence=sha("handoff"),
                    findings={"approved": True, "previous_writer_stopped": True})
    actor.resume(signed)
    assert actor.id == "fixture"
    assert not actor.store.entity(actor.id)["frozen"]
    assert actor.store.audit(actor.id)["audit"] == "intact"


def test_development_does_not_claim_agi(actor):
    assert actor.status()["claims"]["AGI"] == "not-established"


def test_curriculum_is_bounded_and_no_feedback_is_fabricated(actor):
    plan = actor.curriculum(["affine", "queue"], maximum=1)
    assert len(plan) == 1
    assert "never inspect final evaluation feedback" in plan[0]["purpose"]


def test_training_export_requires_a_frozen_source(actor, key):
    experience(actor, key, 1)
    with pytest.raises(Refused):
        training_rows([actor], allowed_sources={sha("fixture-source"): "fixture-license"}, purpose="test")


def test_training_export_keeps_sources_and_whole_history_folds(actor, key):
    for x in (1, 2):
        experience(actor, key, x)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    rows, excluded = training_rows([actor], allowed_sources={sha("fixture-source"): "fixture-license"}, purpose="test")
    assert len(rows) == 2 and len({r["fold"] for r in rows}) == 1
    assert all(r["provenance"]["license"] == "fixture-license" for r in rows)


def test_training_export_excludes_unlicensed_sources(actor, key):
    experience(actor, key, 1)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    rows, excluded = training_rows([actor], allowed_sources={sha("different"): "license"}, purpose="test")
    assert rows == [] and excluded["source-not-licensed"] == 1


def test_hawking_preparation_cannot_train_a_frozen_nx():
    with pytest.raises(Refused):
        hawking_request(dataset_manifest={}, source_model={"artifact": sha("a"), "revision": "r", "tokenizer": sha("t"),
                                                          "representation": "NX"}, role_contract={}, machine_contract=None)
