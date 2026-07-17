"""Rung-level proof that the wave's vectorized construction path equals the sealed scalar path.

The full-generations wave routes the G1-G1 ``construction_search`` lane through the net-new vectorized
executor (``_execute_construction_work`` / ``_construction_rung_payload``) instead of the sealed scalar
``build_pair`` path that ``generation1_successor_mechanics_queue._run_item_payload`` drives. This module
proves that routing is receipt-preserving at the rung boundary: for real G1-G1 fresh-cycle work items,
reduced to a small ``seed_count`` so the sweep stays fast, the vectorized rung JSON is byte identical to
the scalar rung JSON, down to the same ``result_sha256``. A single differing byte fails this test, so
the wave can only ever route construction through the vectorized runner while the receipt is unchanged.

The reduction shortens only the seed_count; the real fresh-cycle ``seed_start`` (the exact
``base + rung*seeds_per_rung + MECHANICS_FRESH_BASE + cycle*MECHANICS_CYCLE_STRIDE`` offset) is
untouched, so the fold order, schema, item block, verdict tallies, and canonical seal are all the real
ones. A separate case drives the actual executors end to end and proves the on-disk raw artifact the
vectorized executor writes is byte identical to the scalar executor's, at the same raw path.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import replace

from mop.studies import generation1_full_generations_wave as wave
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import canonical_bytes, sha256_file

# A short seed band keeps each rung fast while still folding several receipts in ascending seed order.
_REDUCED_SEED_COUNT = 6
# First, a middle, and the last epoch exercise three distinct fresh-cycle seed offsets (cycles 19, 26,
# and 32). The first and last producer/challenge rung indices exercise the rung stride inside a band.
_EPOCH_SAMPLE = (0, len(wave.EPOCH_IDS) // 2, len(wave.EPOCH_IDS) - 1)
_RUNG_SAMPLE = (0, 479)


def _reduced_construction_items() -> list[mechanics.WorkItem]:
    """Real G1-G1 fresh-cycle rung items across phases, epochs, and rungs, shortened to few seeds."""

    items: list[mechanics.WorkItem] = []
    for epoch_index in _EPOCH_SAMPLE:
        for work in wave.category_work_items(epoch_index, "construction"):
            source = wave._fresh_item(work)
            if source.mechanism != "construction_search":
                continue
            if source.rung_index not in _RUNG_SAMPLE:
                continue
            items.append(replace(source, seed_count=_REDUCED_SEED_COUNT))
    return items


def test_reduced_item_sample_covers_all_phases_and_multiple_epochs() -> None:
    """The sample really spans canary, producer, and challenge across several fresh cycles."""

    items = _reduced_construction_items()
    assert items, "expected a non-empty reduced construction rung sample"
    phases = {item.phase.split("_", 2)[-1] for item in items}
    assert phases == {"canary", "producer", "challenge"}, phases
    # Every sampled item is the G1-G1 construction lane at the reduced seed count on a real band.
    assert all(item.lane_id == "G1-G1" and item.mechanism == "construction_search" for item in items)
    assert all(item.seed_count == _REDUCED_SEED_COUNT for item in items)
    assert all(item.seed_start >= mechanics.WORK_ITEMS[0].seed_start for item in items)
    # The three sampled epochs land on distinct fresh-cycle seed offsets.
    canary_starts = {item.seed_start for item in items if item.phase.endswith("canary")}
    assert len(canary_starts) == len(_EPOCH_SAMPLE)


def test_wave_vec_construction_rung_is_byte_identical_to_scalar_rung() -> None:
    """Every vectorized construction rung equals the scalar rung byte for byte (same result_sha256)."""

    items = _reduced_construction_items()
    assert items, "expected a non-empty reduced construction rung sample"

    mismatches: list[tuple] = []
    verdicts_seen: set[str] = set()
    for item in items:
        scalar = mechanics._run_item_payload(item)
        vec = wave._construction_rung_payload(item)
        # The sealed scalar validator must also accept the vectorized rung against the same item.
        mechanics.validate_rung(vec, item)
        verdicts_seen.update(vec["verdict_counts"])
        byte_identical = (
            canonical_bytes(vec) == canonical_bytes(scalar)
            and vec == scalar
            and vec["result_sha256"] == scalar["result_sha256"]
        )
        if not byte_identical:
            mismatches.append(
                (item.phase, item.rung_index, item.seed_start, scalar["result_sha256"], vec["result_sha256"])
            )

    assert not mismatches, (
        f"{len(mismatches)} vec/scalar construction rung mismatches over {len(items)} rungs; "
        f"first diverging (phase, rung, seed_start, scalar_sha, vec_sha): {mismatches[0]}"
    )
    # The real construction bed earns the mechanics-ok verdict, so the fold actually exercised receipts.
    assert verdicts_seen == {"mechanics-ok"}, verdicts_seen


def test_construction_lane_routes_through_vectorized_executor_only() -> None:
    """Routing sends only the construction old lane to the vectorized executor; others are unchanged."""

    construction_works = wave.category_work_items(0, "construction")
    assert construction_works
    for work in construction_works:
        assert wave._work_mechanism(work) == "construction_search"
        assert wave._runner_for(work) is wave._execute_construction_work

    # A non-construction old lane keeps the sealed scalar path.
    formation_works = wave.category_work_items(0, "formation_trace")
    assert formation_works
    for work in formation_works:
        assert wave._runner_for(work) is wave._execute_work

    # A new-mechanism lane keeps the new-mechanisms path.
    uncertainty_works = wave.category_work_items(0, "uncertainty_curiosity")
    new_works = [work for work in uncertainty_works if work.origin == wave._NEW_ORIGIN]
    assert new_works
    for work in new_works:
        assert wave._runner_for(work) is wave._execute_new_work


def test_vec_executor_writes_identical_artifact_at_the_scalar_path(tmp_path) -> None:
    """End to end, the vectorized executor's raw artifact equals the scalar executor's, same path."""

    # The W08 construction canary is the smallest real construction rung (256 seeds), so both executors
    # run it end to end quickly while still folding the full canary band in ascending seed order.
    canary = next(
        work
        for work in wave.category_work_items(0, "construction")
        if wave._fresh_item(work).phase.endswith("canary")
    )

    scalar_root = tmp_path / "scalar"
    vec_root = tmp_path / "vec"

    scalar_key, _scalar_seconds = wave._execute_work(canary, str(scalar_root))
    vec_key, _vec_seconds = wave._execute_construction_work(canary, str(vec_root))
    assert scalar_key == vec_key == canary.key

    scalar_path = wave._artifact_path(scalar_root, canary)
    vec_path = wave._artifact_path(vec_root, canary)
    # Both executors write to the identical raw sub-path (mechanics_fresh/<key>.json) under their roots.
    assert scalar_path.relative_to(scalar_root) == vec_path.relative_to(vec_root)
    assert scalar_path.is_file() and vec_path.is_file()
    # The written bytes are identical, so the raw receipt digest folded into the category result matches.
    assert scalar_path.read_bytes() == vec_path.read_bytes()
    assert sha256_file(scalar_path) == sha256_file(vec_path)

    # And the on-disk vectorized artifact validates against the sealed rung validator.
    item = wave._fresh_item(canary)
    mechanics.validate_rung(wave._read_object(vec_path), item)
