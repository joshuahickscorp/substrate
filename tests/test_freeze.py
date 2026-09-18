"""Unrun freeze contracts. Fixture signatures are not real qualifications.

No test here is collected or executed by the source-only release inspector.
Fitting/provider tests exercise refusal boundaries without launching a trainer.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from conftest import HOST, SCOPE, certificate, experience, native_skill
from substrate.core import Refused, Trust, attest, sha
from substrate.development import Development, LearningPolicy, fit_transition, forecast, new_predictor
from substrate.exchange import capsule, import_capsule, propose_imported_skill
from substrate.hawking import (BRIDGE_SURFACE, INHERITANCE, HawkingClient, UPSTREAM_COMMIT,
                               genome_digest, seal_payload_sha256_v1, validate_nr, validate_nx, validate_nova)
from substrate.local import Corpus, LOCAL_OPERATIONS, fitting_plan, tree_manifest
from substrate.workspace import CognitiveCycle, WorkspacePolicy


def broaden(actor):
    actor.trust = Trust({name: replace(rule, purposes=tuple(rule.purposes) + ("transition", "plasticity", "exchange"))
                         for name, rule in actor.trust.rules.items()})


def source_spec(group="independent-one"):
    return {"origin": "fixture-owned", "license": "fixture-permission", "group": group,
            "classification": "private", "rights": ["read", "train"], "split": "development"}


def retention():
    return {"suite": sha("retention-suite"), "domains": ["math", "language"],
            "training_groups": ["develop-A"], "evaluation_groups": ["heldout-B"], "max_regression": .05}


def stage_attention(actor):
    return Development(actor).stage_mutation(slot="attention", value={"world": 1.4},
        sources=[sha("fixture-source")], retention=retention(), producer="candidate-trainer")


def promote_attention(actor, key, mutation):
    row = actor.get("mutation-candidates", mutation)
    proposal = {"mutation": mutation, "candidate": sha(row), "host": HOST}
    findings = {"passed": True, "executed": True, "suite": retention()["suite"],
                "evaluation_groups": ["heldout-B"], "regression": {"math": 0., "language": .01}}
    signed = certificate(key, actor, "plasticity", proposal, findings, producer="candidate-trainer", domain="cognition")
    return Development(actor).promote(mutation, signed)


def test_learning_policy_is_bounded():
    with pytest.raises(Refused):
        LearningPolicy(replay_per_family=500)


def test_allocator_counts_an_experience_once(actor, key):
    item = experience(actor, key, 3)
    development = Development(actor)
    decision = development.allocate(item)
    head = actor.store.head(actor.id)
    assert development.allocate(item) == decision and actor.store.head(actor.id) == head
    assert actor.get("replay", "affine")["seen"] == 1


def test_allocator_rejects_nonexistent_experience(actor):
    with pytest.raises(Refused):
        Development(actor).allocate(sha("not-admitted"))


def test_source_revocation_stops_examples_and_native_reuse(actor, key):
    skill = native_skill(actor, key)
    Development(actor).revoke_source(sha("fixture-source"), reason="fixture revocation")
    assert actor.examples("affine") == []
    assert not actor.skill_ready(actor.get("skills", skill), "affine", SCOPE)


def test_consolidation_does_not_multiply_support(actor, key):
    development = Development(actor)
    development.allocate(experience(actor, key, 1))
    attempts = actor.competence("affine")["attempts"]
    result = development.consolidate(development.sleep_plan())
    assert result["independent_support_added"] == 0 and not result["weights_trained"]
    assert actor.competence("affine")["attempts"] == attempts


def test_stale_sleep_plan_is_refused(actor, key):
    development = Development(actor)
    plan = development.sleep_plan()
    experience(actor, key, 9)
    with pytest.raises(Refused):
        development.consolidate(plan)


def test_dynamics_updates_only_observed_components():
    model = new_predictor(2, 1, encoder=sha("encoder"))
    changed, belief = fit_transition(model, [0., 0.], [1.], [.5, -.5], [True, False])
    assert changed["weights"][0] != model["weights"][0]
    assert changed["weights"][1] == model["weights"][1]
    assert belief[1] == forecast(model, [0., 0.], [1.])[1]
    assert model["updates"] == 0 and changed["updates"] == 1


def test_dynamics_rejects_empty_observation_and_nonfinite_input():
    model = new_predictor(1, 1, encoder=sha("encoder"))
    with pytest.raises(Refused):
        fit_transition(model, [0.], [0.], [0.], [False])
    with pytest.raises(Refused):
        forecast(model, [float("nan")], [0.])


def test_prospective_transition_requires_independent_observation(actor, key):
    broaden(actor)
    development = Development(actor)
    prediction = development.predict_transition(stream="sensor", sequence=0,
        model=new_predictor(1, 1, encoder=sha("encoder")), state=[0.], action=[.5],
        domain="cognition", source=sha("sensor-source"), unit="history-0")
    proposal = {"prediction": prediction, "actual": [.25], "observed": [True], "producer": "sensor"}
    signed = certificate(key, actor, "transition", proposal,
        {"outcome_verified": True, "source": sha("sensor-source"), "split": "development", "encoder": sha("encoder")},
        producer="sensor", domain="cognition")
    result = development.observe_transition(proposal, signed)
    assert actor.get("predictive-models", result["model"])["updates"] == 1
    assert result["qualified_for_action"] is False
    with pytest.raises(Refused):
        development.observe_transition(proposal, signed)


def test_transition_cannot_skip_pending_prediction(actor):
    development = Development(actor)
    arguments = dict(stream="sensor", sequence=0, model=new_predictor(1, 1, encoder=sha("enc")),
                     state=[0.], action=[0.], domain="cognition", source=sha("sensor"), unit="unit-0")
    development.predict_transition(**arguments)
    with pytest.raises(Refused):
        development.predict_transition(**arguments)


def test_mutation_is_inactive_until_review_and_can_roll_back(actor, key):
    broaden(actor)
    mutation = stage_attention(actor)
    assert Development(actor).active("attention") is None
    promote_attention(actor, key, mutation)
    assert Development(actor).active("attention") == {"world": 1.4}
    Development(actor).rollback("attention", reason="operator rollback")
    assert Development(actor).active("attention") is None
    assert actor.get("mutation-candidates", mutation)["status"] == "promoted"


def test_reviewed_attention_reaches_cognitive_consumer(actor, key):
    broaden(actor)
    promote_attention(actor, key, stage_attention(actor))
    public = {"case_id": sha("task"), "family": "affine", "scope": SCOPE, "input": {"x": 1}, "domain": "micro-world"}
    flags = {"self": True, "llm": False, "compile": False, "memory": False}
    cycle = CognitiveCycle(actor, public, policy=WorkspacePolicy(), flags=flags)
    assert cycle.workspace.weights["world"] == 1.4


def test_revocation_disables_a_reviewed_mutation(actor, key):
    broaden(actor)
    promote_attention(actor, key, stage_attention(actor))
    Development(actor).revoke_source(sha("fixture-source"), reason="remove influence")
    assert Development(actor).active("attention") is None


def test_mutation_training_and_retention_groups_cannot_overlap(actor):
    bad = retention()
    bad["evaluation_groups"] = ["develop-A"]
    with pytest.raises(Refused):
        Development(actor).stage_mutation(slot="world", value={}, sources=[sha("data")], retention=bad, producer="trainer")


def test_corpus_ingestion_is_attributed_not_verified(actor, tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("photonic calibration belongs in the experiment record", encoding="utf-8")
    corpus = Corpus(actor)
    result = corpus.ingest(file, source_spec())
    assert result["verified_facts"] == 0 and corpus.ingest(file, source_spec())["duplicate"]
    assert corpus.search("photonic")[0]["source"] == result["source"]
    assert actor.all("beliefs") == {} and actor.all("experiences") == {}


def test_corpus_revocation_removes_future_retrieval(actor, tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("sensitive needle")
    corpus = Corpus(actor)
    source = corpus.ingest(file, source_spec())["source"]
    assert corpus.search("needle")
    Development(actor).revoke_source(source, reason="permission withdrawn")
    assert corpus.search("needle") == []


def test_bad_utf8_source_remains_unsearchable(actor, tmp_path):
    file = tmp_path / "broken.txt"
    file.write_bytes(b"hidden phrase" + b"\xff")
    corpus = Corpus(actor)
    with pytest.raises(UnicodeDecodeError):
        corpus.ingest(file, source_spec())
    assert corpus.search("hidden") == []


def test_corpus_refuses_final_evaluation_data(actor, tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("holdout")
    with pytest.raises(Refused):
        Corpus(actor).ingest(file, {**source_spec(), "split": "evaluation"})


def test_duplicate_training_text_cannot_cross_group_partitions(actor, tmp_path):
    corpus = Corpus(actor)
    groups = {}
    for index in range(100):
        group = f"group-{index}"
        fold = int(sha([group, "adapt"])[:16], 16) % 5 == 0
        groups.setdefault(fold, group)
    assert len(groups) == 2
    file = tmp_path / "source.txt"
    file.write_text("same example across apparently independent groups")
    sources = [corpus.ingest(file, source_spec(group))["source"] for group in groups.values()]
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    with pytest.raises(Refused):
        corpus.training(tmp_path / "dataset", sources=sources, purpose="adapt")
    assert not (tmp_path / "dataset").exists()


def test_payload_inventory_includes_all_files_and_refuses_links(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "tensor.dat").write_bytes(b"fixture")
    assert tree_manifest(root)["bytes"] == 7
    (root / "link").symlink_to(root / "tensor.dat")
    with pytest.raises(Refused):
        tree_manifest(root)


def fixture_nr():
    return {"nr_kind": "hawking.nos.noetic_representation", "nr_version": "1.0.0",
            "semantic_provenance": {"parent_model": "fixture", "parent_revision": "fixture", "parameter_count": 0},
            "representation": {}, "kernel_requirements": [{"requires": "fixture-family", "note": "semantic only"}]}


def fixture_genome():
    machine = {"chipset": "fixture", "gpu_cores": 1, "unified_memory_bytes": 1024,
               "metal_family": "fixture", "measured_roof_gb_s": 1.}
    return {**machine, "genome_digest": genome_digest(machine)}


def test_hawking_nr_rejects_nested_machine_bindings():
    nr = fixture_nr()
    assert validate_nr(nr)["execution_qualified"] is False
    nr["representation"]["nested"] = {"threadgroup": 128}
    with pytest.raises(Refused):
        validate_nr(nr)


def test_hawking_legacy_seal_is_not_substrate_compact_hash():
    import hashlib
    value = {"z": 1, "a": 2}
    assert seal_payload_sha256_v1(value) == hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    assert seal_payload_sha256_v1(value) != sha(value)


def test_hawking_nx_binds_exact_nr_bytes_and_measured_machine():
    import hashlib
    nr, machine = fixture_nr(), fixture_genome()
    raw = json.dumps(nr, indent=2).encode()
    nx = {"nx_kind": "hawking.nos.noetic_executable_genome", "nx_version": "1.0.0",
          "compiled_for_machine_genome": machine,
          "lowers_nr": {"nr_content_sha256": hashlib.sha256(raw).hexdigest(), "nr_kind": nr["nr_kind"],
                        "requirements_the_nx_satisfies": [{"requires": "fixture-family"}]}}
    assert validate_nx(nx, nr_bytes=raw, machine=machine)["execution_qualified"] is False
    with pytest.raises(Refused):
        validate_nx(nx, nr_bytes=json.dumps(nr).encode(), machine=machine)
    changed = {**machine, "gpu_cores": 2}
    changed["genome_digest"] = genome_digest(changed)
    with pytest.raises(Refused):
        validate_nx(nx, nr_bytes=raw, machine=changed)


def test_hawking_missing_nova_lineage_does_not_pass():
    with pytest.raises(Refused):
        validate_nova({}, parent_identity="parent", descendant=sha("candidate"))


def test_unimplemented_hawking_route_refused_before_network():
    client = HawkingClient("http://127.0.0.1:8080")
    with pytest.raises(Refused):
        client.query("POST /v1/responses", {}, surface={"document": {"endpoints": []}})
    assert BRIDGE_SURFACE["POST /v1/embeddings"] == "partial"


def test_whole_hawking_port_is_not_limited_to_organs():
    assert INHERITANCE["whole_system_eligible"] and INHERITANCE["permanent_partial_port_cap"] is None
    assert not INHERITANCE["external_hawking_service_required"]
    assert len(UPSTREAM_COMMIT) == 40


def public_rights():
    return {sha("fixture-source"): {"license": "fixture-share", "share_derived": True, "privacy_review": True,
                                    "classification": "public", "review_digest": sha("privacy-review")}}


def test_capsule_export_requires_rights_closure(actor, key):
    skill = native_skill(actor, key)
    actor.store.freeze(actor.id, actor.store.head(actor.id))
    with pytest.raises(Refused):
        capsule(actor, skills=[skill], rights={}, author="publisher")
    document = capsule(actor, skills=[skill], rights=public_rights(), author="publisher")
    assert not document["raw_personal_data_included"] and not document["authority_inherited"]
    assert document["programs"][0]["import_status"] == "unqualified"


def test_local_command_surface_has_no_implicit_qualification():
    assert "fit-run" in LOCAL_OPERATIONS and "promote" in LOCAL_OPERATIONS
    assert "tests" not in LOCAL_OPERATIONS and "odyssey" not in LOCAL_OPERATIONS
    assert "not scientific" in LOCAL_OPERATIONS["fit-authorize"]


def test_completed_source_cannot_reanimate_partial_tail(actor, tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("valid corpus")
    corpus = Corpus(actor)
    source = corpus.ingest(file, source_spec())["source"]
    actor._commit("fixture-leftover", {}, [("document-chunks", sha("tail"),
        {"source": source, "index": 99, "text": "stale poison"})], base=actor.store.head(actor.id))
    assert corpus.search("poison") == [] and corpus.search("valid")


def test_actual_nr_can_be_mapped_only_to_an_unqualified_organ(tmp_path):
    from substrate.hawking import organ_candidate
    path = tmp_path / "nr.json"
    path.write_text(json.dumps(fixture_nr()))
    spec = {"id": "fixture-organ", "model": "fixture", "artifact": sha("payload"), "nr_artifact": sha("payload"),
            "behavior_contract": sha("behavior"), "state_schema": sha("state"), "tokenizer": sha("tokenizer"),
            "endpoint": "http://127.0.0.1:8080/v1/chat/completions", "roles": ["reasoner"]}
    result = organ_candidate(path, spec)
    assert result["organ"]["representation"] == "nr" and not result["execution_qualified"]
    assert not result["complete_payload_verified"] and not result["server_artifact_verified"]
