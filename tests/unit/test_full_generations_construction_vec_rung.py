
from __future__ import annotations

from dataclasses import replace

from mop.studies import generation1_full_generations_wave as wave
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import canonical_bytes, sha256_file

_REDUCED_SEED_COUNT = 6
_EPOCH_SAMPLE = (0, len(wave.EPOCH_IDS) // 2, len(wave.EPOCH_IDS) - 1)
_RUNG_SAMPLE = (0, 479)


def _reduced_construction_items() -> list[mechanics.WorkItem]:

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

    items = _reduced_construction_items()
    assert items, "expected a non-empty reduced construction rung sample"
    phases = {item.phase.split("_", 2)[-1] for item in items}
    assert phases == {"canary", "producer", "challenge"}, phases
    assert all(item.lane_id == "G1-G1" and item.mechanism == "construction_search" for item in items)
    assert all(item.seed_count == _REDUCED_SEED_COUNT for item in items)
    assert all(item.seed_start >= mechanics.WORK_ITEMS[0].seed_start for item in items)
    canary_starts = {item.seed_start for item in items if item.phase.endswith("canary")}
    assert len(canary_starts) == len(_EPOCH_SAMPLE)


def test_wave_vec_construction_rung_is_byte_identical_to_scalar_rung() -> None:

    items = _reduced_construction_items()
    assert items, "expected a non-empty reduced construction rung sample"

    mismatches: list[tuple] = []
    verdicts_seen: set[str] = set()
    for item in items:
        scalar = mechanics._run_item_payload(item)
        vec = wave._construction_rung_payload(item)
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
    assert verdicts_seen == {"mechanics-ok"}, verdicts_seen


def test_construction_lane_routes_through_vectorized_executor_only() -> None:

    construction_works = wave.category_work_items(0, "construction")
    assert construction_works
    for work in construction_works:
        assert wave._work_mechanism(work) == "construction_search"
        assert wave._runner_for(work) is wave._execute_construction_work

    formation_works = wave.category_work_items(0, "formation_trace")
    assert formation_works
    for work in formation_works:
        assert wave._runner_for(work) is wave._execute_work

    uncertainty_works = wave.category_work_items(0, "uncertainty_curiosity")
    new_works = [work for work in uncertainty_works if work.origin == wave._NEW_ORIGIN]
    assert new_works
    for work in new_works:
        assert wave._runner_for(work) is wave._execute_new_work


def test_vec_executor_writes_identical_artifact_at_the_scalar_path(tmp_path) -> None:

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
    assert scalar_path.relative_to(scalar_root) == vec_path.relative_to(vec_root)
    assert scalar_path.is_file() and vec_path.is_file()
    assert scalar_path.read_bytes() == vec_path.read_bytes()
    assert sha256_file(scalar_path) == sha256_file(vec_path)

    item = wave._fresh_item(canary)
    mechanics.validate_rung(wave._read_object(vec_path), item)
