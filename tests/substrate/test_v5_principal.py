from __future__ import annotations

import copy

from substrate import v5config as C
from substrate import v5principal as P


def test_v5_principal_dag_is_frozen_within_master_bounds() -> None:
    units = P.work_units()
    episodes = sum(unit.event_count for unit in units)
    assert len(units) == 5_760
    assert 5_000 <= len(units) <= 30_000
    assert 500_000 <= episodes <= 5_000_000
    assert len(P.SPLIT_SEEDS["principal"]) == 48
    assert len(C.ARMS) == 18
    assert len(C.PHASES) == 20
    assert all(unit.document()["activation"] is False for unit in units[:10])


def test_v5_continuing_entity_survives_interrupt_replacement_and_body_change() -> None:
    predecessor = None
    receipts = []
    checkpoints = []
    for shard in range(P.SHARDS):
        unit = P.WorkUnit("principal", 5_000, "full_v5", shard)
        receipt, checkpoint = P.execute_unit(unit, predecessor)
        assert P.validate(receipt, checkpoint, unit, predecessor)
        receipts.append(receipt)
        checkpoints.append(checkpoint)
        predecessor = checkpoint
    identities = {
        receipt["summary"]["entity_identity"] for receipt in receipts
    }
    assert len(identities) == 1
    final = checkpoints[-1]["state"]
    assert final["sensor_interruptions"] == 1
    assert final["restorations"] == 1
    assert final["model_replacements"] == 1
    assert final["model_identity"] == "vision-temporal-beta"
    assert final["body_changes"] == 1
    assert final["unfinished_goals"] == ["return-to-scene"]
    assert final["activation"] is False


def test_v5_fresh_reset_is_a_real_identity_control() -> None:
    predecessor = None
    identities = []
    for shard in range(P.SHARDS):
        unit = P.WorkUnit("replication", 6_000, "fresh_reset", shard)
        _, checkpoint = P.execute_unit(unit, predecessor)
        identities.append(checkpoint["state"]["entity_identity"])
        predecessor = checkpoint
    assert len(set(identities)) == P.SHARDS


def test_v5_unit_validation_rejects_checkpoint_drift() -> None:
    unit = P.WorkUnit("open_world_review", 7_000, "full_v5", 0)
    receipt, checkpoint = P.execute_unit(unit)
    corrupted = copy.deepcopy(checkpoint)
    corrupted["state"]["entity_identity"] = "mutated"
    assert not P.validate(receipt, corrupted, unit)
