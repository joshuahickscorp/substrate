"""Unrun source contracts for the shared store, typed language and trust boundary."""
import base64
import json
import math
import zipfile

import pytest

from substrate.core import (Budget, Conflict, Cost, Refused, Trust, TrustRule, attest, canonical, clone,
                            digest, fields, identifier, integer, numeric, parse, sha, scope_matches)
from substrate.language import CausalModel, apply, compile_program, equivalent, induce, make_program, mine_material
from substrate.store import Store
from conftest import INPUTS


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.25, "snow ☃", [1, False], {"b": 1, "a": [2]}])
def test_canonical_roundtrip(value):
    assert parse(canonical(value)) == value


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', 'NaN', 'Infinity', '-Infinity', '{bad}', '1e999'])
def test_json_refuses_ambiguous_or_nonfinite(raw):
    with pytest.raises(Refused):
        parse(raw)


def test_canonical_key_order():
    assert sha({"x": 1, "y": 2}) == sha({"y": 2, "x": 1})


def test_clone_has_no_shared_mutable_children():
    original = {"x": [1]}
    duplicate = clone(original)
    duplicate["x"].append(2)
    assert original == {"x": [1]}


@pytest.mark.parametrize("value", ["", "../x", "/x", "x y", "x:y", "x" * 129, None])
def test_identifier_refusal(value):
    with pytest.raises(Refused):
        identifier(value)


@pytest.mark.parametrize("value", ["0" * 63, "z" * 64, "A" * 64, None])
def test_digest_refusal(value):
    with pytest.raises(Refused):
        digest(value)


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "3", 10**10000])
def test_numeric_refusal(value):
    with pytest.raises(Refused):
        numeric(value)


def test_exact_fields_reject_unknown_authority():
    with pytest.raises(Refused):
        fields({"name": "x", "approved": True}, {"name"})


def test_scope_requires_every_declared_condition():
    assert scope_matches({"version": 2}, {"version": 2, "x": 1})
    assert not scope_matches({"version": 2}, {"version": 1})


def test_unknown_resources_are_not_zero():
    assert Cost().energy_j is None and Cost().peak_memory_bytes is None


@pytest.mark.parametrize("kwargs", [{"calls": -1}, {"tokens": True}, {"workers": 65}, {"wall_seconds": 0}])
def test_budget_is_bounded(kwargs):
    with pytest.raises(Refused):
        Budget(**kwargs)


def test_attestation_verifies(key, trust):
    subject = {"case": 1}
    cert = attest(key, signer="oracle", producer="student", purpose="skill", domain="micro-world",
                  subject=subject, evidence=sha("evidence"), findings={"passed": True}, now=100, ttl=10)
    assert trust.verify(cert, purpose="skill", domain="micro-world", subject=subject, producer="student", now=105)["passed"]


@pytest.mark.parametrize("mutation", ["subject", "purpose", "producer", "signature", "expires"])
def test_attestation_tamper_refused(key, trust, mutation):
    cert = attest(key, signer="oracle", producer="student", purpose="skill", domain="micro-world",
                  subject={}, evidence=sha("e"), findings={}, now=100, ttl=10)
    cert[mutation] = 1000 if mutation == "expires" else "changed"
    with pytest.raises(Refused):
        trust.verify(cert, purpose="skill", domain="micro-world", subject={}, producer="student", now=105)


def test_attestation_cannot_self_verify(key, trust):
    cert = attest(key, signer="oracle", producer="oracle", purpose="skill", domain="micro-world",
                  subject={}, evidence=sha("e"), findings={}, now=100, ttl=10)
    with pytest.raises(Refused):
        trust.verify(cert, purpose="skill", domain="micro-world", subject={}, producer="oracle", now=105)


@pytest.mark.parametrize("now", [99, 110, 111])
def test_attestation_time_window(key, trust, now):
    cert = attest(key, signer="oracle", producer="student", purpose="skill", domain="micro-world",
                  subject={}, evidence=sha("e"), findings={}, now=100, ttl=10)
    with pytest.raises(Refused):
        trust.verify(cert, purpose="skill", domain="micro-world", subject={}, producer="student", now=now)


def test_content_addressed_dedup(actor):
    store = actor.store
    with store.transaction():
        first = store.put({"text": "repeat" * 1000})
        size = store.counter("payload_bytes")
        assert store.put({"text": "repeat" * 1000}) == first
        assert store.counter("payload_bytes") == size
    assert store.get(first) == {"text": "repeat" * 1000}


def test_missing_object_reference_refused(actor):
    with pytest.raises(Refused), actor.store.transaction():
        actor.store.put({"$ref": sha("absent")})


def test_objects_need_transaction(actor):
    with pytest.raises(Refused):
        actor.store.put({"x": 1})


def test_compare_and_swap_fences_old_state(actor):
    store = actor.store
    old = store.head(actor.id)
    store.commit(actor.id, old, "change", {}, [("facts", "x", {"value": 1})])
    with pytest.raises(Conflict):
        store.commit(actor.id, old, "stale", {}, [])


def test_transaction_rolls_back_records(actor):
    store = actor.store
    before = store.counter("payload_bytes")
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put({"will": "rollback"})
            raise RuntimeError("stop")
    assert store.counter("payload_bytes") == before


def test_audit_detects_materialization_tamper(actor):
    store = actor.store
    store.commit(actor.id, store.head(actor.id), "change", {}, [("facts", "x", 1)])
    with store.transaction():
        store.db.execute("DELETE FROM records WHERE entity=?", (actor.id,))
    with pytest.raises(Refused):
        store.audit(actor.id)


def test_store_rejects_wrong_actor_evaluator_role(tmp_path):
    path = tmp_path / "role.sqlite"
    with Store(path, create=True, role="evaluator"):
        pass
    with pytest.raises(Refused):
        Store(path, role="actor")


def test_store_does_not_overwrite_existing(tmp_path):
    path = tmp_path / "existing.sqlite"
    with Store(path, create=True):
        pass
    with pytest.raises(FileExistsError):
        Store(path, create=True)


def test_store_refuses_symlink(tmp_path):
    original, link = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    with Store(original, create=True):
        pass
    link.symlink_to(original)
    with pytest.raises(Refused):
        Store(link)


def test_compressed_corruption_is_detected(actor):
    store = actor.store
    with store.transaction():
        key = store.put({"payload": "x" * 10000})
        store.db.execute("UPDATE objects SET payload=? WHERE id=?", (b"broken", key))
    with pytest.raises(Refused):
        store.get(key)


def test_record_export_restore_is_pinned(actor, tmp_path):
    store = actor.store
    store.commit(actor.id, store.head(actor.id), "change", {}, [("facts", "x", {"value": 1})])
    store.freeze(actor.id, store.head(actor.id))
    path = tmp_path / "entity.zip"
    pin = store.export_entity(actor.id, path)
    with Store(tmp_path / "restored.sqlite", create=True) as restored:
        assert restored.restore_entity(path, manifest_pin=pin) == actor.id
        assert restored.audit(actor.id)["head"] == store.head(actor.id)
        assert restored.record(actor.id, "facts", "x") == {"value": 1}


def test_bundle_pin_cannot_come_from_another_bundle(actor, tmp_path):
    path = tmp_path / "entity.zip"
    actor.store.export_entity(actor.id, path)
    with Store(tmp_path / "restore.sqlite", create=True) as store, pytest.raises(Refused):
        store.restore_entity(path, manifest_pin=sha("wrong"))


def test_freeze_refuses_further_learning(actor):
    store = actor.store
    store.freeze(actor.id, store.head(actor.id))
    with pytest.raises(Refused):
        actor.think("A new thought", domain="micro-world", scope={"x": 1})


def test_fork_has_explicit_lineage_and_omits_ablation_namespace(actor):
    store = actor.store
    store.commit(actor.id, store.head(actor.id), "change", {}, [("facts", "x", 1), ("skills", "y", 2)])
    store.freeze(actor.id, store.head(actor.id))
    store.fork(actor.id, "forked", intervention="no-skills", omit_namespaces=("skills",))
    assert store.record("forked", "facts", "x") == 1
    assert store.records("forked", "skills") == {}
    assert store.audit("forked")["audit"] == "intact"


def job_document(actor, name="j", depends=None):
    return {"id": name, "entity": actor.id, "phase": "train", "payload": {"x": name}, "depends_on": depends or []}


def test_job_dependency_blocks_claim(actor):
    store = actor.store
    store.enqueue([job_document(actor, "a"), job_document(actor, "b", ["a"])])
    job = store.claim("worker")
    assert job["id"] == "a"
    assert store.claim("other") is None
    store.complete(job, {})
    assert store.claim("worker")["id"] == "b"


def test_job_graph_refuses_cycle(actor):
    with pytest.raises(Refused):
        actor.store.enqueue([job_document(actor, "a", ["b"]), job_document(actor, "b", ["a"])])


def test_job_definition_cannot_be_rewritten(actor):
    actor.store.enqueue([job_document(actor)])
    changed = job_document(actor)
    changed["payload"] = {"different": True}
    with pytest.raises(Refused):
        actor.store.enqueue([changed])


def test_old_worker_fence_cannot_complete(actor):
    store = actor.store
    store.enqueue([job_document(actor)])
    job = store.claim("worker")
    with store.transaction():
        store.db.execute("UPDATE jobs SET fence=fence+1 WHERE id='j'")
    with pytest.raises(Conflict):
        store.complete(job, {})


def test_expired_work_becomes_uncertain_not_replayed(actor):
    store = actor.store
    store.enqueue([job_document(actor)])
    store.claim("worker")
    with store.transaction():
        store.db.execute("UPDATE jobs SET expires=0 WHERE id='j'")
    assert store.claim("new-worker") is None
    assert store.db.execute("SELECT state FROM jobs WHERE id='j'").fetchone()[0] == "uncertain"


def test_effect_reservation_precedes_dispatch(actor):
    store = actor.store
    store.enqueue([job_document(actor)])
    job = store.claim("worker")
    effect = store.reserve_effect(job, {"prompt": "x"}, max_tokens=100, call_limit=1, token_limit=100)
    assert store.counter("calls") == 1 and store.counter("tokens") == 100
    store.sent_effect(job, effect)
    with pytest.raises(Refused):
        store.sent_effect(job, effect)
    store.settle_effect(job, effect, {"response": 1}, actual_tokens=20)
    assert store.counter("tokens") == 20
    store.complete(job, {})


def test_unknown_model_usage_retains_full_reservation(actor):
    store = actor.store
    store.enqueue([job_document(actor)])
    job = store.claim("worker")
    effect = store.reserve_effect(job, {}, max_tokens=100, call_limit=1, token_limit=100)
    store.sent_effect(job, effect)
    store.settle_effect(job, effect, {}, actual_tokens=None)
    assert store.counter("tokens") == 100


def test_unsettled_request_cannot_be_success(actor):
    store = actor.store
    store.enqueue([job_document(actor)])
    job = store.claim("worker")
    store.reserve_effect(job, {}, max_tokens=100, call_limit=1, token_limit=100)
    with pytest.raises(Refused):
        store.complete(job, {})


def test_all_compact_outcomes_survive_trace_sampling(actor):
    store = actor.store
    for i in range(20):
        record = {"id": sha(i), "entity": actor.id, "unit": "u", "arm": "a", "phase": "evaluation",
                  "family": "affine", "case_id": sha([i]), "metrics": {"correct": None, "sealed": True}}
        store.add_outcome(record, {"verbose": "x" * 100}, retain_success_every=100000)
    assert len(list(store.outcomes())) == 20
    assert all(row["trace_digest"] for row in store.outcomes())


@pytest.mark.parametrize("body,inputs,expected", [
    (["add", ["var", "x"], ["const", 2]], {"x": 3}, 5),
    (["mul", ["var", "x"], ["const", -2]], {"x": 3}, -6),
    (["abs", ["var", "x"]], {"x": -3}, 3),
    (["mod", ["var", "x"], ["const", 2]], {"x": 5}, 1),
    (["if", ["lt", ["var", "x"], ["const", 0]], ["const", -1], ["const", 1]], {"x": -1}, -1),
])
def test_typed_programs(body, inputs, expected):
    program = compile_program(make_program(INPUTS, body))
    assert program.run(inputs) == expected
    assert program.cost()["model_calls"] == 0


@pytest.mark.parametrize("operation", ["eval", "exec", "open", "__import__", "system", "getattr", "socket"])
def test_ir_has_no_host_execution_escape(operation):
    with pytest.raises(Refused):
        make_program(INPUTS, [operation, ["var", "x"]])


def test_ir_lazy_branch_does_not_evaluate_unreachable_division():
    program = make_program(INPUTS, ["if", ["const", True], ["const", 3], ["floordiv", ["var", "x"], ["const", 0]]])
    assert compile_program(program).run({"x": 2}) == 3


def test_ir_rejects_out_of_range_input():
    program = compile_program(make_program(INPUTS, ["var", "x"]))
    with pytest.raises(Refused):
        program.run({"x": 1000001})


def test_ir_rejects_extra_input_name():
    with pytest.raises(Refused):
        compile_program(make_program(INPUTS, ["var", "x"])).run({"x": 2, "extra": 1})


def test_ir_type_checks_branches():
    with pytest.raises(Refused):
        make_program(INPUTS, ["if", ["const", True], ["const", 1], ["const", False]])


def test_ir_deduplicates_common_subexpressions():
    expr = ["add", ["var", "x"], ["const", 1]]
    program = compile_program(make_program(INPUTS, ["mul", expr, expr]))
    assert len(program.instructions) == 4


@pytest.mark.parametrize("op,expected", [("sort", [1, 2, 2]), ("reverse", [2, 1, 2]), ("unique", [2, 1])])
def test_list_ir(op, expected):
    spec = {"xs": {"type": "list-int", "max_items": 16}}
    program = compile_program(make_program(spec, [op, ["var", "xs"]]))
    assert program.run({"xs": [2, 1, 2]}) == expected


def test_bool_is_not_numeric_success():
    assert not equivalent(True, 1)
    assert equivalent(1, 1.0)


def test_induction_is_candidate_not_qualification():
    samples = [{"input": {"x": i}, "output": i + 1} for i in (2, 4, 8)]
    result = induce(samples, INPUTS, seconds=.5)
    assert result["program"] is not None
    assert result["qualified"] is False


def test_causal_intervention_replaces_equation():
    mid = make_program(INPUTS, ["add", ["var", "x"], ["const", 1]])
    downstream = make_program({"m": INPUTS["x"]}, ["mul", ["var", "m"], ["const", 2]])
    model = CausalModel({"inputs": INPUTS, "equations": {"m": mid, "y": downstream}})
    assert model.predict({"x": 2})["y"] == 6
    assert model.predict({"x": 2}, do={"m": 10})["y"] == 20


def test_causal_cycle_refused():
    with pytest.raises(Refused):
        CausalModel({"inputs": INPUTS, "equations": {"m": make_program({"y": INPUTS["x"]}, ["var", "y"]),
                                                   "y": make_program({"m": INPUTS["x"]}, ["var", "m"])}})
