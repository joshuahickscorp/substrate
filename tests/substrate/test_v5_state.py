from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from substrate import v5io as io
from substrate import v5state as S


def contract(
    identity: str,
    checkpoint: str,
    *,
    roles: tuple[str, ...] = ("independent_performer", "specialist"),
) -> S.ModelContract:
    return S.ModelContract(
        identity=identity,
        checkpoint_identity=checkpoint,
        allowed_roles=roles,
        training_provenance=("local deterministic fixture",),
    )


def populated_entity() -> S.PermanentEntity:
    entity = S.PermanentEntity("entity:continuing")
    entity.upsert_goal("goal:learn", "retain the learned scene", priority=0.9)
    entity.upsert_task("task:return", "return to the scene", goal_ids=("goal:learn",))
    entity.upsert_hypothesis(
        "hypothesis:hidden",
        {"object": "object:cup", "location": "shelf"},
        confidence=0.6,
        evidence=("observation:1",),
    )
    entity.update_world(
        "tracked_objects",
        "object:cup",
        {"class": "cup", "track": [1, 2], "occluded": True},
    )
    entity.record_memory(
        "semantic",
        "memory:cup",
        {"class": "container"},
        provenance=("observation:1",),
        verification=("check:1",),
    )
    entity.record_memory(
        "procedural",
        "memory:inspect",
        {"steps": ["change viewpoint", "compare track"]},
        provenance=("demonstration:1",),
        verification=("check:2",),
    )
    return entity


def reseal(document: dict) -> dict:
    return io.sealed_document(
        {key: value for key, value in document.items() if key != "sha256"}
    )


def test_v5_canonical_json_keeps_legacy_bytes_and_rejects_nonfinite_values() -> None:
    value = {
        "z": (1, 2.0),
        "a": {2: "two", 1: "one"},
        "large": 9007199254740993,
        "unicode": "é",
    }
    normalized = io._normal_json(value)  # noqa: SLF001 - parity with the prior canonical path
    expected = (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert io.canonical_json(value) == expected
    with pytest.raises(io.Refused, match="finite canonical JSON"):
        io.canonical_json({"not_a_number": math.nan})


def test_v5_canonical_json_fast_encoder_preserves_cycle_refusal() -> None:
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(io.Refused, match="finite canonical JSON"):
        io.canonical_json(cyclic)


def test_v5_stable_json_preserves_legacy_digest_bytes_and_cycle_refusal() -> None:
    value = {"unicode": "naïve café", "tuple": (1, 2.0), "nan": math.nan}
    expected = json.dumps(
        value,
        allow_nan=True,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    assert io.stable_json(value) == expected

    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Circular reference detected"):
        io.stable_json(cyclic)


def test_normalized_seal_matches_general_seal_and_falls_back_for_tuples() -> None:
    document = {"payload": {"items": [1, 2, 3]}, "activation": False}
    expected = io.sealed_document(document)
    assert io._sealed_normalized_document(document) == expected  # noqa: SLF001
    assert document == {"payload": {"items": [1, 2, 3]}, "activation": False}

    noncanonical = {"payload": {"items": (1, 2, 3)}, "activation": False}
    assert io._sealed_normalized_document(noncanonical) == io.sealed_document(noncanonical)  # noqa: SLF001

    with pytest.raises(io.Refused, match="activation"):
        io._sealed_normalized_document({"activation": True})  # noqa: SLF001


def test_event_payload_fast_path_preserves_canonical_key_order() -> None:
    payload = {"z": {"z": 2, "a": 3}, "a": 1}
    event = S.CognitiveEvent.create(
        sequence=1,
        event_time=1,
        kind="entity_created",
        payload=payload,
        previous_sha256=None,
    )
    assert event.payload == io._normal_json(payload)  # noqa: SLF001
    assert list(event.payload) == ["a", "z"]
    assert list(event.payload["z"]) == ["a", "z"]


def test_event_payload_and_semantic_snapshots_remain_detached() -> None:
    payload = {"layer": "active_context", "value": {"nested": ["before"]}}
    entity = S.PermanentEntity("entity:isolation")
    appended = entity.append_event("context_updated", payload)

    assert appended == entity.events[-1].to_dict()
    appended["payload"]["value"]["nested"].append("return-mutation")
    assert entity.events[-1].payload["value"]["nested"] == ["before"]

    payload["value"]["nested"].append("caller-mutation")
    assert entity.state["active_context"]["nested"] == ["before"]

    snapshot = entity.semantic_state()
    snapshot["active_context"]["nested"].append("snapshot-mutation")
    assert entity.state["active_context"]["nested"] == ["before"]


def test_reducer_copy_on_write_preserves_prior_state_branches() -> None:
    entity = S.PermanentEntity("entity:copy-on-write")

    def assert_prior_untouched(operation: Callable[[], object]) -> None:
        prior = entity._state  # noqa: SLF001 - inspect the reducer boundary
        expected = copy.deepcopy(prior)
        operation()
        assert prior == expected

    assert_prior_untouched(
        lambda: entity.update_context("active_context", {"topic": "copy-on-write"})
    )
    assert_prior_untouched(
        lambda: entity.upsert_goal("goal:copy", "preserve prior branches")
    )
    assert_prior_untouched(
        lambda: entity.attach_sensor(
            "sensor:copy", {"modality": "text", "coordinate_frame": "document"}
        )
    )
    assert_prior_untouched(
        lambda: entity.observe_sensor(
            "sensor:copy", {"observation": "detached"}, source_timestamp=1
        )
    )
    assert_prior_untouched(lambda: entity.interrupt_sensor("sensor:copy"))
    assert_prior_untouched(
        lambda: entity.register_model(contract("model:copy", "sha256:copy"))
    )
    assert_prior_untouched(
        lambda: entity.replace_model(
            "model:copy", contract("model:copy-v2", "sha256:copy-v2")
        )
    )
    assert_prior_untouched(
        lambda: entity.enqueue("learning_queue", {"identity": "item:copy"})
    )
    assert_prior_untouched(lambda: entity.dequeue("learning_queue", "item:copy"))


def test_v5_io_is_atomic_content_addressed_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / "substrate" / "v5"
    monkeypatch.setattr(io, "RUNS", root)
    path = io.run_json("receipts/example.json", {"value": [3, 1, 2]})
    loaded = io.load_json(path)
    assert loaded["activation"] is False
    assert loaded["sha256"] == io.sha_obj(
        {key: value for key, value in loaded.items() if key != "sha256"}
    )
    object_path = root / ".objects" / loaded["sha256"][:2] / (
        f"{loaded['sha256']}.json"
    )
    assert object_path.read_bytes() == path.read_bytes()
    assert io.run_json("receipts/example.json", {"value": [3, 1, 2]}) == path

    with pytest.raises(io.Refused, match="activation"):
        io.run_json("receipts/active.json", {"nested": {"activation": True}})
    with pytest.raises(io.Refused, match="outside"):
        io.publish_json(tmp_path / "escape.json", {"activation": False})

    nested_active = {
        "program": io.PROGRAM,
        "source_commit": "0" * 40,
        "source_digest": "0" * 64,
        "activation": False,
        "payload": {"activation": True},
    }
    nested_active["sha256"] = io.sha_obj(
        {key: value for key, value in nested_active.items() if key != "sha256"}
    )
    nested_active_path = root / "receipts" / "nested-active.json"
    nested_active_path.parent.mkdir(parents=True, exist_ok=True)
    nested_active_path.write_bytes(io.canonical_json(nested_active))
    with pytest.raises(io.Refused, match="activation"):
        io.load_json(nested_active_path)

    corrupt = copy.deepcopy(loaded)
    corrupt["value"].append(9)
    io.atomic_write(path, io.canonical_json(corrupt).decode())
    with pytest.raises(io.Refused, match="self-seal"):
        io.load_json(path)


def test_event_sourced_state_has_stable_identity_and_monotonic_time() -> None:
    entity = populated_entity()
    identity = entity.state["identity"]
    state_sha256 = entity.state_identity()
    assert state_sha256 == entity.state_identity()
    event = entity.advance_time(100, reason="simulated week-scale idle")
    assert event["event_time"] == 100
    assert entity.state_identity() != state_sha256
    assert entity.checkpoint()["state_sha256"] == entity.state_identity()
    assert entity.state["continuous_time"]["gaps"][-1]["missing_intervals"] > 0
    assert entity.state["identity"] == identity
    assert entity.state["active_goals"]["goal:learn"]["status"] == "active"
    assert entity.state["unfinished_tasks"]["task:return"]["goal_ids"] == [
        "goal:learn"
    ]
    assert entity.state["unresolved_hypotheses"]["hypothesis:hidden"]
    assert entity.state["tracked_objects"]["object:cup"]["occluded"]
    assert entity.activation is False
    assert all(
        later.event_time > earlier.event_time
        and later.previous_sha256 == earlier.event_sha256
        for earlier, later in zip(entity.events, entity.events[1:], strict=False)
    )
    with pytest.raises(S.Refused, match="increase"):
        entity.append_event("late", {}, event_time=99)
    with pytest.raises(S.Refused, match="unsupported cognitive event"):
        entity.append_event("purchase", {"sku": "unknown-event"})
    with pytest.raises(S.Refused, match="activation"):
        entity.append_event(
            "context_updated",
            {"layer": "active_context", "value": {"activation": True}},
        )


def test_knowledge_admission_requires_registered_distinct_authorities_and_evidence() -> None:
    entity = S.PermanentEntity("entity:knowledge")
    entity.attach_sensor(
        "sensor:source",
        {"modality": "text", "coordinate_frame": "document"},
    )
    entity.register_model(
        contract(
            "model:verifier",
            "sha256:verifier",
            roles=("independent_performer", "verifier"),
        )
    )
    admitted = entity.admit_knowledge(
        "knowledge:checked",
        {"claim": "fixture is internally consistent"},
        provenance=("sensor:source",),
        verification=("model:verifier",),
        verification_evidence=("receipt:independent-check-1",),
    )
    assert admitted["activation"] is False
    record = entity.state["knowledge"]["knowledge:checked"]
    assert record["provenance"] == ["sensor:source"]
    assert record["verification"] == ["model:verifier"]
    assert record["verification_evidence"] == ["receipt:independent-check-1"]

    cases = (
        {
            "provenance": ("sensor:source",),
            "verification": ("model:verifier",),
            "verification_evidence": (),
        },
        {
            "provenance": ("sensor:unregistered",),
            "verification": ("model:verifier",),
            "verification_evidence": ("receipt:check",),
        },
        {
            "provenance": ("sensor:source",),
            "verification": ("model:unregistered",),
            "verification_evidence": ("receipt:check",),
        },
        {
            "provenance": ("model:verifier",),
            "verification": ("model:verifier",),
            "verification_evidence": ("receipt:check",),
        },
    )
    for index, arguments in enumerate(cases):
        with pytest.raises(S.Refused):
            entity.admit_knowledge(
                f"knowledge:rejected-{index}",
                {"claim": "must not be admitted"},
                **arguments,
            )
    assert not any(
        identity.startswith("knowledge:rejected-")
        for identity in entity.state["knowledge"]
    )

    nonverifier = contract("model:not-verifier", "sha256:not-verifier")
    entity.register_model(nonverifier)
    with pytest.raises(S.Refused, match="lack the verifier role"):
        entity.admit_knowledge(
            "knowledge:wrong-role",
            {"claim": "must not be admitted"},
            provenance=("sensor:source",),
            verification=(nonverifier.identity,),
            verification_evidence=("receipt:check",),
        )
    with pytest.raises(S.Refused, match="identity lists"):
        entity.append_event(
            "knowledge_upserted",
            {
                "record": {
                    "identity": "knowledge:malformed",
                    "content": {"claim": "must not be admitted"},
                    "provenance": "sensor:source",
                    "verification": "model:verifier",
                    "verification_evidence": "receipt:check",
                }
            },
        )


def test_checkpoint_restore_is_exact_and_rejects_state_or_event_corruption() -> None:
    entity = populated_entity()
    checkpoint = entity.checkpoint()
    assert entity._checkpoint_internal() == checkpoint  # noqa: SLF001
    restored = S.PermanentEntity.restore(checkpoint)
    assert restored.checkpoint() == checkpoint
    assert restored.state_identity() == checkpoint["state_sha256"]

    state_corrupt = copy.deepcopy(checkpoint)
    state_corrupt["state"]["active_goals"]["goal:learn"]["description"] = "changed"
    state_corrupt["state_sha256"] = io.sha_obj(state_corrupt["state"])
    state_corrupt = reseal(state_corrupt)
    with pytest.raises(S.Refused, match="exact deterministic event projection"):
        S.PermanentEntity.restore(state_corrupt)

    event_corrupt = copy.deepcopy(checkpoint)
    event_corrupt["events"][1]["payload"]["goal"]["description"] = "changed"
    event_corrupt = reseal(event_corrupt)
    with pytest.raises(S.Refused, match="corrupt cognitive event"):
        S.PermanentEntity.restore(event_corrupt)

    seal_corrupt = copy.deepcopy(checkpoint)
    seal_corrupt["owned_identity"] = "other"
    with pytest.raises(S.Refused, match="seal"):
        S.PermanentEntity.restore(seal_corrupt)


def test_checkpoint_cache_is_detached_and_invalidated_after_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = populated_entity()
    seal_calls = 0
    original_seal = io._sealed_normalized_document  # noqa: SLF001

    def counted_seal(document: dict) -> dict:
        nonlocal seal_calls
        seal_calls += 1
        return original_seal(document)

    monkeypatch.setattr(io, "_sealed_normalized_document", counted_seal)  # noqa: SLF001
    first = entity.checkpoint()
    second = entity.checkpoint()
    assert first == second
    assert seal_calls == 1

    first["state"]["active_goals"]["goal:learn"]["description"] = "caller mutation"
    first["events"][0]["payload"]["entity_id"] = "caller mutation"
    assert entity.checkpoint() == second
    assert seal_calls == 1

    entity.events[1].payload["goal"]["description"] = "event mutation"
    changed_by_event = entity.checkpoint()
    assert changed_by_event != second
    assert changed_by_event["events"][1]["payload"]["goal"]["description"] == "event mutation"
    assert seal_calls == 2

    entity.set_mode("observing")
    changed = entity.checkpoint()
    assert changed["event_chain_head"] != second["event_chain_head"]
    assert seal_calls == 3


def test_events_and_checkpoint_persist_under_v5_as_self_sealed_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / "substrate" / "v5"
    monkeypatch.setattr(io, "RUNS", root)
    entity = S.PermanentEntity("entity:persisted", storage_root=root)
    entity.upsert_goal("goal:resume", "survive a process restart")
    checkpoint_path = entity.save_checkpoint()
    assert checkpoint_path.is_file()
    assert len(list((root / "entities" / "entity:persisted" / "events").glob("*.json"))) == 2

    restored = S.PermanentEntity.load_checkpoint(
        checkpoint_path,
        storage_root=root,
    )
    assert restored.checkpoint() == entity.checkpoint()
    assert restored.state["active_goals"]["goal:resume"]


def test_models_remain_independent_and_replacement_preserves_owned_state() -> None:
    entity = populated_entity()
    draft = S.DeterministicModel(
        contract("model:draft", "sha256:draft"),
        lambda request: {"candidate": request["value"] + 1},
    )
    verifier = S.DeterministicModel(
        contract("model:verify", "sha256:verify"),
        lambda request: {"verified": request["candidate"] == 4},
    )
    entity.register_model(draft)
    entity.register_model(verifier)
    entity.relate_models(
        "model:draft",
        "model:verify",
        "drafts_for",
        measured=True,
        evidence=("canary:model-support",),
    )
    assert entity.call_model("model:draft", {"value": 3})["output"] == {
        "candidate": 4
    }
    assert entity.call_model("model:verify", {"candidate": 4})["output"] == {
        "verified": True
    }

    replacement = S.DeterministicModel(
        contract("model:verify-v2", "sha256:verify-v2"),
        lambda request: {"verified": request["candidate"] % 2 == 0},
    )
    report = entity.replace_model(
        "model:verify",
        replacement,
        measured=True,
        evidence=("canary:replacement",),
    )
    assert report["entity_identity_preserved"]
    assert report["goals_preserved"]
    assert report["memory_preserved"]
    assert report["world_preserved"]
    assert entity.call_model("model:verify-v2", {"candidate": 4})["output"][
        "verified"
    ]
    with pytest.raises(S.Refused, match="unavailable"):
        entity.call_model("model:verify", {"candidate": 4})


def test_model_invocation_keeps_runner_mutation_out_of_input_identity() -> None:
    registry = S.ModelRegistry()

    def mutating_runner(request: dict) -> dict:
        request["nested"]["items"].append("runner-mutation")
        return {"items": request["nested"]["items"]}

    registry.register(contract("model:isolated", "sha256:isolated"), mutating_runner)
    request = {"nested": {"items": ["caller"]}}
    result = registry.invoke("model:isolated", request)

    assert request == {"nested": {"items": ["caller"]}}
    assert result["output"]["items"] == ["caller", "runner-mutation"]
    assert result["receipt"]["input_sha256"] == io.sha_obj(request)


def test_sensor_interruption_and_body_replacement_preserve_cognition() -> None:
    entity = populated_entity()
    entity.attach_sensor(
        "sensor:camera",
        {"modality": "video", "coordinate_frame": "camera:1"},
    )
    entity.observe_sensor(
        "sensor:camera",
        {
            "raw_data_reference": "sha256:frame",
            "observation": {"pixels": "referenced"},
            "hypothesis": {"object": "object:cup"},
        },
        source_timestamp=10,
    )
    interruption = entity.interrupt_sensor("sensor:camera")
    assert interruption["entity_identity_preserved"]
    assert interruption["goals_preserved"]
    assert interruption["memory_preserved"]
    assert interruption["world_preserved"]
    assert entity.state["sensors"]["sensor:camera"]["status"] == "interrupted"
    assert entity.state["tracked_objects"]["object:cup"]["occluded"]

    body = entity.replace_body(
        {
            "identity": "body:simulator-v2",
            "sensors": ["sensor:camera"],
            "actuators": ["rotate"],
            "coordinate_frames": ["body:base"],
            "capabilities": ["viewpoint_change"],
        }
    )
    assert body["entity_identity_preserved"]
    assert body["goals_preserved"]
    assert body["memory_preserved"]
    assert body["world_preserved"]
    assert entity.state["body_state"]["generation"] == 1


def test_consolidation_and_workspace_projection_are_bounded_and_traceable() -> None:
    entity = S.PermanentEntity("entity:bounded")
    for index in range(12):
        entity.record_memory(
            "episodic",
            f"episode:{index:02d}",
            {"index": index, "detail": "x" * 20},
            provenance=(f"event:{index}",),
        )
    event_count = len(entity.events)
    consolidation = entity.consolidate(max_active_episodic=4, batch_size=5)
    assert consolidation is not None
    assert len(entity.state["episodic_memory"]) == 7
    summary = next(iter(entity.state["semantic_memory"].values()))
    assert len(summary["source_memory_ids"]) == 5
    assert len(entity.events) == event_count + 1

    projection = entity.workspace_projection(max_items=6, max_bytes=2_048)
    assert projection["bounds"]["included_items"] <= 6
    assert len(io.canonical_json(projection)) <= 2_048
    assert "events" not in projection
    assert projection["identity"]["entity_id"] == "entity:bounded"


def test_workspace_projection_byte_ledger_preserves_exact_boundary() -> None:
    entity = S.PermanentEntity("entity:projection-budget")
    for index in range(40):
        entity.update_context("active_context", {"index": index, "payload": ["x"] * 8})

    budget = 4_096
    for _ in range(5):
        projection = entity.workspace_projection(max_items=32, max_bytes=budget)
        size = len(io.canonical_json(projection))
        if size == budget:
            break
        budget = size

    exact = entity.workspace_projection(max_items=32, max_bytes=budget)
    assert exact == projection
    assert len(io.canonical_json(exact)) <= budget

    for smaller_budget in range(budget - 1, 511, -1):
        smaller = entity.workspace_projection(max_items=32, max_bytes=smaller_budget)
        if smaller["bounds"]["included_items"] < exact["bounds"]["included_items"]:
            break
    else:
        raise AssertionError("no smaller budget rejected a projection candidate")
    assert len(io.canonical_json(smaller)) <= smaller_budget


def test_schema_migration_and_rollback_recover_the_exact_prior_checkpoint() -> None:
    entity = S.PermanentEntity("entity:migration", schema_version=1)
    entity.upsert_goal("goal:migrate", "retain exact pre-migration state")
    before = entity.checkpoint()
    receipt = entity.migrate_schema(2)
    assert receipt.semantic_identity_preserved
    assert entity.state["schema_version"] == 2
    assert entity.state["archival_tiers"] == {"episodic": []}
    assert S.PermanentEntity.restore(entity.checkpoint()).checkpoint() == entity.checkpoint()

    rolled_back = entity.rollback_migration(receipt)
    assert rolled_back.checkpoint() == before
    assert S.rollback_checkpoint(receipt) == before
    with pytest.raises(S.Refused, match="changed since migration"):
        entity.upsert_goal("goal:later", "make rollback stale")
        entity.rollback_migration(receipt)


def test_service_lifecycle_modes_health_and_restore_are_deterministic() -> None:
    service = S.EntityService(populated_entity())
    assert service.start()["status"] == "running"
    service.entity.set_mode("observing")
    health = service.health()
    assert health["status"] == "running"
    assert health["mode"] == "observing"
    assert health["state_integrity"]
    snapshot = service.snapshot()
    assert service.pause()["status"] == "paused"
    assert service.resume()["status"] == "running"
    assert service.stop()["status"] == "stopped"
    restored = service.restore(snapshot)
    assert restored["status"] == "restored"
    assert service.entity.checkpoint() == snapshot
    assert service.entity.activation is False
