from __future__ import annotations

from mop.studies import generation1_c3_d1_frozen_queue as d1
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as horizon
from mop.studies import generation1_successor_horizon_verify as verifier
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studies.generation1_c3_dispatch import canonical_sha256


def test_horizon_is_five_fresh_cycles_and_more_than_one_parallel_day() -> None:
    assert horizon.EPOCH_CYCLES == (2, 3, 4, 5, 6)
    assert horizon.IDLE_WORKERS == 20
    assert horizon.planned_horizon_compute_seconds() >= 230 * 60 * 60
    assert horizon.planned_horizon_compute_seconds() / horizon.IDLE_WORKERS >= 11 * 60 * 60


def test_shard_plan_is_complete_unique_and_below_five_serial_hours() -> None:
    assert sorted(index for shard in horizon.D1_PARTITIONS for index in shard) == list(
        range(d1.DEFAULT_RUNG_COUNT)
    )
    assert sorted(index for shard in horizon.MECHANICS_PARTITIONS for index in shard) == list(
        range(len(mechanics.WORK_ITEMS))
    )
    assert (
        max(len(shard) * consolidated.D1_PLANNED_RUNG_SECONDS for shard in horizon.D1_PARTITIONS)
        < 5 * 60 * 60
    )
    assert (
        max(
            sum(
                mechanics.WORK_ITEMS[index].seed_count
                * mechanics.PLANNED_SECONDS_PER_SEED[mechanics.WORK_ITEMS[index].mechanism]
                for index in shard
            )
            for shard in horizon.MECHANICS_PARTITIONS
        )
        < 5 * 60 * 60
    )


def test_all_horizon_seed_spaces_are_disjoint() -> None:
    d1_intervals = []
    mechanics_intervals = []
    for epoch_index in range(len(horizon.EPOCH_IDS)):
        for source_index in range(d1.DEFAULT_RUNG_COUNT):
            work = horizon._d1_work(epoch_index, source_index)
            config = consolidated.fresh_d1_config(work)
            d1_intervals.extend(
                (
                    int(config[f"{kind}_seed_start"]),
                    int(config[f"{kind}_seed_start"]) + int(config[f"{kind}_seed_count"]),
                )
                for kind in ("train", "heldout")
            )
        for source_index in range(len(mechanics.WORK_ITEMS)):
            work = horizon._mechanics_work(epoch_index, source_index)
            item = consolidated.fresh_mechanics_item(work)
            mechanics_intervals.append((item.seed_start, item.seed_start + item.seed_count))
    assert verifier._intervals_disjoint(d1_intervals)
    assert verifier._intervals_disjoint(mechanics_intervals)
    assert max(end for _, end in d1_intervals) < min(start for start, _ in mechanics_intervals)


def _fake_rung(*, learned: float, global_static: float, difficulty_static: float, context: float):
    cells = []
    for seed in (1, 2, 3, 4):
        for difficulty in range(5):
            cells.append(
                {
                    "seed": seed,
                    "difficulty_index": difficulty,
                    "variant_accuracy": {d1.FROZEN_VARIANT_ID: learned},
                    "control_accuracy": {
                        "global_static": global_static,
                        "difficulty_static": difficulty_static,
                        "random_actor": global_static,
                        "context_route_nonpromotable": context,
                    },
                }
            )
    return {
        "cells": cells,
        "grid": {
            "train_seed_count": len({cell["seed"] for cell in cells}),
            "heldout_seed_count": len({cell["seed"] for cell in cells}),
        },
    }


def test_separate_streaming_verifier_reproduces_frozen_phase_gate() -> None:
    passing = _fake_rung(
        learned=0.63,
        global_static=0.55,
        difficulty_static=0.58,
        context=0.64,
    )
    failing = _fake_rung(
        learned=0.60,
        global_static=0.55,
        difficulty_static=0.58,
        context=0.64,
    )
    assert verifier._phase_gate([passing]) is d1.summarize_phase([passing])["all_frozen_criteria_passed"]
    assert verifier._phase_gate([failing]) is d1.summarize_phase([failing])["all_frozen_criteria_passed"]
    accumulator = horizon._D1PhaseAccumulator()
    accumulator.add(passing)
    assert accumulator.result()["all_frozen_criteria_passed"] is True
    assert accumulator.result()["grid"]["rung_count"] == 1


def test_separate_canonicalizer_matches_producer_and_duplicate_intervals_fail() -> None:
    payload = {"unicode": "perspective-\u03bb", "nested": {"z": 1, "a": False}}
    assert verifier.canonical_sha256(payload) == canonical_sha256(payload)
    assert verifier._intervals_disjoint([(1, 2), (2, 3)]) is True
    assert verifier._intervals_disjoint([(1, 2), (1, 2)]) is False


def test_resealed_semantic_mutations_are_all_rejected() -> None:
    core = {
        "schema": horizon.RESULT_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": horizon.CLAIM_SCOPE,
        "grid": {"epoch_count": len(horizon.EPOCH_IDS)},
        "decision": {"independent_scientific_confirmation": False},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = {**core, "result_sha256": canonical_sha256(core)}
    mutations = verifier._mutation_suite(result)
    assert mutations == {"count": 9, "rejected": 9, "all_rejected": True}
