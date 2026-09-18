"""Unrun experimental-boundary contracts. All fake transports stay local to tests."""
from dataclasses import asdict

import pytest

from substrate.core import Budget, Cost, Refused, canonical, sha
from substrate.odyssey import ARMS, draft_plan, jobs_for, preflight, validate_plan
from substrate.organs import Broker, JsonTransport, Organ, validate_hawking
from substrate.sandbox import SandboxConfig, execute_request
from substrate.statistics import paired_history_interval, summarize, verify_scores
from substrate.store import Store
from substrate.worlds import Evaluator, FAMILIES, World, observation
from conftest import HOST, INPUTS


class CountingTransport:
    def __init__(self, content='{"answer":3,"confidence":0.8}', *, tools=False, usage=True):
        self.calls, self.requests = 0, []
        self.content, self.tools, self.usage = content, tools, usage

    def post(self, request):
        self.calls += 1
        self.requests.append(request)
        message = {"content": self.content}
        if self.tools:
            message["tool_calls"] = [{"function": {"name": "delete"}}]
        response = {"choices": [{"finish_reason": "stop", "message": message}]}
        if self.usage:
            response["usage"] = {"prompt_tokens": 50, "completion_tokens": 10}
        return response


def broker_fixture(actor, transport):
    organ = Organ("local-model", "pinned-model-revision", sha("model"), "http://127.0.0.1:8080/v1/chat/completions",
                  ("reasoner", "teacher"))
    broker = Broker(actor.store, organ, Budget(), approved_endpoints=(organ.endpoint,), transport=transport)
    actor.store.enqueue([{"id": "job", "entity": actor.id, "phase": "train", "payload": {}, "depends_on": []}])
    job = actor.store.claim("worker")
    task = {"description": "a bounded task", "input_schema": INPUTS, "input": {"x": 1}, "oracle_secret": "DO-NOT-SEND"}
    return broker, job, task


def test_llm_is_an_actual_priced_transport_path(actor):
    transport = CountingTransport()
    broker, job, task = broker_fixture(actor, transport)
    result = broker.query(job, role="reasoner", task=task, examples=[])
    assert transport.calls == 1
    assert result["proposal"]["answer"] == 3
    assert actor.store.counter("calls") == 1 and actor.store.counter("tokens") == 60


def test_llm_projection_excludes_evaluator_fields(actor):
    transport = CountingTransport()
    broker, job, task = broker_fixture(actor, transport)
    broker.query(job, role="reasoner", task=task, examples=[])
    assert b"DO-NOT-SEND" not in canonical(transport.requests)


def test_model_generated_tool_calls_are_not_dispatched(actor):
    broker, job, task = broker_fixture(actor, CountingTransport(tools=True))
    result = broker.query(job, role="reasoner", task=task, examples=[])
    assert result["proposal"] is None and result["invalid_proposal"] is not None
    assert actor.store.db.execute("SELECT state FROM effects").fetchone()[0] == "complete"


@pytest.mark.parametrize("content", ['{"approved":true}', '{"answer":3,"confidence":5}', '{"answer":NaN,"confidence":1}', 'not json'])
def test_invalid_model_output_is_charged_not_admitted(actor, content):
    broker, job, task = broker_fixture(actor, CountingTransport(content))
    result = broker.query(job, role="reasoner", task=task, examples=[])
    assert result["proposal"] is None
    assert actor.store.counter("calls") == 1


def test_unknown_provider_usage_keeps_reservation(actor):
    broker, job, task = broker_fixture(actor, CountingTransport(usage=False))
    result = broker.query(job, role="reasoner", task=task, examples=[])
    assert not result["usage_known"]
    assert actor.store.counter("tokens") == result["reserved_tokens"]


def test_operator_endpoint_allowlist_is_binding(actor):
    organ = Organ("organ", "model", sha("model"), "http://127.0.0.1:8080/v1/chat/completions", ("reasoner",))
    with pytest.raises(Refused):
        Broker(actor.store, organ, Budget(), approved_endpoints=())


@pytest.mark.parametrize("url", ["file:///secret", "http://example.com/api", "https://user:pass@example.com/api",
                                 "https://example.com/api?key=x", "https://example.com/api#secret"])
def test_transport_refuses_unsafe_endpoint_shapes(url):
    with pytest.raises(Refused):
        JsonTransport(url)


def test_oci_requires_a_digest_pin():
    with pytest.raises(Refused):
        SandboxConfig(mode="oci", image="python:latest")


def test_main_cannot_use_native_mode_as_security_sandbox():
    from substrate.sandbox import Executor
    with pytest.raises(Refused):
        Executor(SandboxConfig(), main=True)


def test_worker_only_accepts_closed_ir():
    with pytest.raises(Refused):
        execute_request({"schema": "substrate-worker-v2", "program": "import os", "inputs": [{"x": 1}]})


def test_worker_batch_has_an_explicit_bound():
    with pytest.raises(Refused):
        execute_request({"schema": "substrate-worker-v2", "program": {}, "inputs": []})


@pytest.mark.parametrize("family", FAMILIES)
def test_worlds_are_reproducible_and_projected(family):
    world = World(b"private-fixture-seed" * 2)
    one = world.make("u0", family, "train", 1)
    two = world.make("u0", family, "train", 1)
    assert one.public == two.public and one.actual == two.actual
    assert "actual" not in one.public and "recipe" not in one.public
    assert set(observation(one.public)) == {"case_id", "domain", "family", "split", "input", "source", "scope"}


def test_world_parameters_are_not_shared_between_independent_histories():
    world = World(b"seed" * 8)
    assert len({sha(world.parameters(f"u{i}", "affine")) for i in range(20)}) > 1


def test_numeric_world_splits_are_disjoint():
    world = World(b"seed" * 8)
    train = {sha(world.make("u", "affine", "train", i).public["input"]) for i in range(16)}
    heldout = {sha(world.make("u", "affine", "evaluation", i).public["input"]) for i in range(16)}
    assert not train & heldout


def test_boolean_world_is_finite_not_falsely_novel():
    world = World(b"seed" * 8)
    assert len({sha(world.make("u", "logic", "train", i).public["input"]) for i in range(32)}) == 8


def test_world_reference_does_not_use_candidate_interpreter(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("candidate interpreter was invoked by oracle")
    monkeypatch.setattr("substrate.language.compile_program", refuse)
    assert World.truth("affine", {"x": 3}, {"a": 2, "b": 1, "variant": 0}) == 7


def tiny_plan():
    plan = draft_plan()
    plan.update(status="locked", arms=["developed", "memory-only"], families=["affine"])
    plan["episodes"] = {"train": 2, "evaluation": 2, "transfer": 1, "retention": 1, "shift": 1, "organ_swap": 0}
    return plan


@pytest.fixture
def evaluator(tmp_path, key):
    with Store(tmp_path / "oracle.sqlite", create=True, role="evaluator") as store:
        yield Evaluator(store, tiny_plan(), b"seed" * 8, key, "oracle", HOST, executor=None)


def descriptor(evaluator, *, split="train", index=0, unit=None, arm=None):
    return {"unit": unit or evaluator.plan["units"][0], "arm": arm or evaluator.plan["arms"][0],
            "family": "affine", "split": split, "index": index}


def publish(evaluator, request, *, head=None):
    public = evaluator.case(request)
    subject = ({"entity": request["unit"] + "." + request["arm"], "base": sha("base"), "operation": "experience",
                "proposal": {"prediction": sha(request), "case_id": public["case_id"], "unit": request["unit"],
                             "producer": "tester", "actual": None}} if request["split"] == "train" else
               {"head": head, "plan": sha(evaluator.plan)})
    return evaluator.submit({**request, "answer": None, "confidence": 0.0, "mechanism": "abstain",
                             "cost": Cost(elapsed_ns=1).document(), "producer": "tester", "subject": subject})


def test_evaluation_inputs_require_training_and_freeze(evaluator):
    with pytest.raises(Refused):
        evaluator.case(descriptor(evaluator, split="evaluation"))


def test_same_prediction_cannot_be_revised_after_feedback(evaluator):
    request = descriptor(evaluator)
    first = publish(evaluator, request)
    assert first["proposal"]["actual"] == first["actual"]
    assert publish(evaluator, request) == first


def test_final_scores_stay_sealed_until_matrix_complete(evaluator):
    publish(evaluator, descriptor(evaluator))
    with pytest.raises(Refused):
        evaluator.report()


def test_freeze_requires_all_training_examples(evaluator):
    with pytest.raises(Refused):
        evaluator.seal({"unit": evaluator.plan["units"][0], "arm": evaluator.plan["arms"][0], "head": sha("head")})


def test_training_cannot_resume_after_final_seal(evaluator):
    for index in range(2):
        publish(evaluator, descriptor(evaluator, index=index))
    request = descriptor(evaluator)
    evaluator.seal({"unit": request["unit"], "arm": request["arm"], "head": sha("head")})
    with pytest.raises(Refused):
        evaluator.case(request)


def test_complete_score_report_is_sharded_and_verified(evaluator, trust):
    for unit in evaluator.plan["units"]:
        for arm in evaluator.plan["arms"]:
            for index in range(2):
                publish(evaluator, descriptor(evaluator, unit=unit, arm=arm, index=index))
            head = sha([unit, arm])
            evaluator.seal({"unit": unit, "arm": arm, "head": head})
            for split, count in evaluator.plan["episodes"].items():
                if split != "train":
                    for index in range(count):
                        result = publish(evaluator, descriptor(evaluator, unit=unit, arm=arm, split=split, index=index), head=head)
                        assert result["sealed"] and "correct" not in result
    signed = evaluator.report()
    shards = {key: evaluator.scores({"shard": key})["rows"] for key in signed["document"]["shards"]}
    rows = verify_scores(evaluator.plan, signed, shards, trust)
    assert len(rows) == 20 and all(row["correct"] is False for row in rows)


def test_plan_is_inert_and_explicitly_draft():
    plan = draft_plan(kind="main")
    assert plan["status"] == "draft"
    assert plan["preregistration"]["sample_size_rationale"] is None
    assert set(plan["arms"]) == set(ARMS)


def test_ablation_flags_are_not_identical():
    assert ARMS["developed"] != ARMS["no-material"]
    assert ARMS["developed"] != ARMS["no-teacher"]
    assert ARMS["memory-only"]["compile"] is False
    assert ARMS["native-only"]["llm"] is False


def test_job_graph_places_freeze_before_heldout():
    jobs = jobs_for(tiny_plan())
    by_id = {job["id"]: job for job in jobs}
    evaluation = next(job for job in jobs if job["phase"] == "evaluation")
    assert by_id[evaluation["depends_on"][0]]["phase"] == "freeze"


def test_duplicate_histories_refused():
    plan = draft_plan()
    plan["units"] = ["same", "same"]
    with pytest.raises(Refused):
        validate_plan(plan)


def test_preflight_does_not_contact_services(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("preflight contacted a service")
    monkeypatch.setattr(JsonTransport, "post", refuse)
    bindings = {"schema": "odyssey-bindings-v2", "host": HOST, "organ": None, "replacement": None,
                "evaluator": {"endpoint": "http://127.0.0.1:8091/", "token_env": "TOKEN"},
                "sandbox": asdict(SandboxConfig()), "qualifications": []}
    report = preflight(draft_plan(kind="main"), bindings, tmp_path)
    assert report["network_contacted"] is False and report["missing"]


def test_paired_statistics_count_histories_not_trials():
    result = paired_history_interval([.1, .2, -.1, .3], seed=1, repetitions=100)
    assert result["histories"] == 4
    assert result["not_optional_stopping_valid"]


def test_resource_summary_preserves_unknown_measurements():
    rows = [{"correct": True, "confidence": .8, "mechanism": "native-skill", "cost": Cost(elapsed_ns=1000).document()}]
    report = summarize(rows)
    assert report["energy_j"] is None and report["peak_memory_bytes"] is None


def test_nx_must_bind_its_exact_nr_and_contract():
    nr = {"schema": "substrate-hawking-nr-bridge-v1", "artifact": sha("nr"), "behavior_contract": sha("behavior"),
          "state_schema": sha("state"), "lineage": [], "tokenizer": sha("tokenizer")}
    nx = {"schema": "substrate-hawking-nx-bridge-v1", "artifact": sha("nx"), "nr": sha("wrong-nr"),
          "machine_contract": sha("machine"), "behavior_contract": sha("behavior"), "state_schema": sha("state"),
          "execution_plan": sha("plan"), "qualification": sha("qualification")}
    with pytest.raises(Refused):
        validate_hawking(nr, nx, sha("nx"), "nx")
