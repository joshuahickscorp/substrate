from __future__ import annotations

import json

import pytest

from substrate import v2config as C
from substrate import v2fabric as F


def test_splits_are_disjoint_and_generator_is_deterministic():
    seeds = [seed for values in C.SPLITS.values() for seed in values]
    assert len(seeds) == len(set(seeds))
    assert F.generate_task(0, "A", 3) == F.generate_task(0, "A", 3)
    assert F.generate_task(0, "A", 3) != F.generate_task(1, "A", 3)


@pytest.mark.parametrize("domain", tuple(C.DOMAIN_CATALOG))
def test_targets_are_delayed_and_oracle_solves(domain):
    task = F.generate_task(0, domain, 7)
    public = task.public()
    encoded = json.dumps(public, sort_keys=True)
    assert "private_target" not in encoded
    assert '"target"' not in encoded
    assert task.private_target not in encoded
    assert F.leakage(task)["passes"]
    proposal = F.execute(task.required_operation, task.observation, task.alternatives)
    assert task.reveal(proposal)["correct"]
    assert task.reveal(proposal)["revealed_after_commitment"]
    assert task.reveal(proposal)["activation"] is False


def test_bed_screen_has_four_valid_nonsaturated_domains_and_real_headroom():
    report = F.screen(per_domain=8)
    assert report["all_valid"]
    assert len(report["domains"]) == 4
    assert not report["task_identity_collisions"]
    assert not any(report["split_seed_overlap"].values())
    for row in report["domains"].values():
        assert row["not_saturated"]
        assert row["oracle_headroom"] > C.SESOI
        assert not row["answer_leakage_failures"]


def test_negative_transfer_operation_is_wrong_for_tool_domain():
    tasks = [F.generate_task(0, "D", index) for index in range(16)]
    assert all(task.task_signature != C.DOMAIN_CATALOG["A"]["task_signature"] for task in tasks)
    wrong = [F.execute("boundary_route", task.observation, task.alternatives) for task in tasks]
    right = [F.execute("risk_route", task.observation, task.alternatives) for task in tasks]
    assert all(proposal == task.private_target for proposal, task in zip(right, tasks, strict=True))
    assert any(a != b for a, b in zip(wrong, right, strict=True))
