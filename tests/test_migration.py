"""Unrun migration, source-binding and experiment-integrity regression obligations."""
import pytest

from substrate.cognition import Entity
from substrate.core import Refused, clone, sha
from substrate.handoff import migrate_v1, training_rows
from substrate.odyssey import draft_plan, validate_plan
from substrate.store import Store
from conftest import HOST, experience


def legacy_bundle():
    manifest = {"entity_id": "legacy-entity", "schema_version": "substrate-product-v1"}
    state = {"entity_id": "legacy-entity", "schema_version": "substrate-product-v1",
             "cognition": {"projects": {"study": {"goal": "Continue a proof", "next_action": "Check the lemma"}},
                           "beliefs": {"P": {"status": "supported"}}}}
    unsigned = {"sequence": 1, "kind": "entity_initialized", "payload": {"manifest_sha256": sha(manifest)},
                "previous_sha256": None, "schema_version": "substrate-product-v1", "timestamp": "2026-09-18T00:00:00Z"}
    receipt = {**unsigned, "sha256": sha(unsigned)}
    checkpoint = {"developmental_state_sha256": sha(state), "ledger_length": 1, "ledger_tail_sha256": receipt["sha256"],
                  "manifest_sha256": sha(manifest), "schema_version": "substrate-product-v1"}
    document = {"checkpoint": checkpoint, "manifest": manifest, "manifest_sha256": sha(manifest),
                "receipts": [receipt], "schema_version": "substrate-product-v1", "state": state}
    return {**document, "sha256": sha(document)}


def test_migration_is_quarantined_not_claimed_equivalent(tmp_path):
    document = legacy_bundle()
    with Store(tmp_path / "migrated.sqlite", create=True) as store:
        result = migrate_v1(store, document, pin=document["sha256"])
        assert result["semantic_equivalence"] == "not-established"
        assert result["qualified_skills_inherited"] == 0
        assert result["projects_preserved"] == 1
        assert not store.records("legacy-entity", "beliefs")
        assert store.records("legacy-entity", "legacy-state")
        assert store.audit("legacy-entity")["audit"] == "intact"


def test_migration_rejects_wrong_transport_pin(tmp_path):
    with Store(tmp_path / "migrated.sqlite", create=True) as store:
        with pytest.raises(Refused):
            migrate_v1(store, legacy_bundle(), pin=sha("wrong"))


def test_migration_detects_changed_legacy_checkpoint(tmp_path):
    document = legacy_bundle()
    document["checkpoint"]["ledger_length"] = 2
    document["sha256"] = sha({k: v for k, v in document.items() if k != "sha256"})
    with Store(tmp_path / "migrated.sqlite", create=True) as store:
        with pytest.raises(Refused):
            migrate_v1(store, document, pin=document["sha256"])


def test_control_arm_exports_are_not_deduplicated_away(tmp_path, trust, key):
    with Store(tmp_path / "actor.sqlite", create=True) as store:
        first = Entity.create(store, "unit-0.developed", trust, host=HOST)
        second = Entity.create(store, "unit-0.memory-only", trust, host=HOST)
        for entity in (first, second):
            experience(entity, key, 1)
            store.freeze(entity.id, store.head(entity.id))
        rows, excluded = training_rows([first, second], allowed_sources={sha("fixture-source"): "fixture"}, purpose="fitting")
        assert len(rows) == 2 and {row["control_arm"] for row in rows} == {"developed", "memory-only"}
        assert len({row["fold"] for row in rows}) == 1
        assert not excluded


def test_single_study_matrix_is_memory_bounded():
    plan = draft_plan(kind="main")
    plan["units"] = [f"history-{i:03}" for i in range(1000)]
    plan["episodes"]["evaluation"] = 4096
    with pytest.raises(Refused):
        validate_plan(plan)


def test_fork_failure_does_not_leave_a_partial_child(actor, monkeypatch):
    store = actor.store
    store.freeze(actor.id, store.head(actor.id))
    original = store._commit
    def fail_inherit(entity, expected, kind, payload, changes, **kwargs):
        if kind == "inherit":
            raise Refused("injected-fork-interruption")
        return original(entity, expected, kind, payload, changes, **kwargs)
    monkeypatch.setattr(store, "_commit", fail_inherit)
    with pytest.raises(Refused):
        store.fork(actor.id, "child", intervention="ablation")
    assert store.db.execute("SELECT 1 FROM entities WHERE id='child'").fetchone() is None
