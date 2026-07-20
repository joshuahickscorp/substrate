
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.ladder.stage3_successor_registry import build_pair
from mop.process_labels import set_process_label
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256
from mop.studies.generation1_successor_mechanics_queue import LaneSpec, WorkItem

PROGRAM_ID = "generation1-new-mechanisms-queue-v1"
RUNG_SCHEMA = "mop-generation1-new-mechanisms-rung/v1"
CLAIM_SCOPE = "deterministic generated new-mechanism mechanics stress only; no confirmation or activation"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID

NEW_LANES = (
    LaneSpec(
        "G1-U1",
        "calibrated_uncertainty",
        ("G1-C2",),
        False,
        171_000_001,
        172_000_001,
        177_000_001,
        64,
        65_536,
    ),
    LaneSpec(
        "G1-N1", "reducible_novelty", ("G1-C0",), False, 182_000_001, 183_000_001, 188_000_001, 64, 65_536
    ),
    LaneSpec(
        "G1-P1R",
        "stability_plasticity_r2",
        ("G1-C0",),
        False,
        193_000_001,
        194_000_001,
        199_000_001,
        64,
        65_536,
    ),
)

CANARY_SEEDS = 256

NEW_PLANNED_SECONDS_PER_SEED = {
    "calibrated_uncertainty": 0.000_154_417,
    "reducible_novelty": 0.000_124_245,
    "stability_plasticity_r2": 0.000_229_5,
}


def build_new_work_items() -> tuple[WorkItem, ...]:

    items = []
    for lane in NEW_LANES:
        items.append(
            WorkItem(
                index=len(items),
                lane_id=lane.lane_id,
                mechanism=lane.mechanism,
                phase="canary",
                rung_index=0,
                seed_start=lane.canary_start,
                seed_count=CANARY_SEEDS,
            )
        )
    for lane in NEW_LANES:
        for phase, base in (("producer", lane.producer_start), ("challenge", lane.challenge_start)):
            for rung_index in range(lane.rungs_per_phase):
                items.append(
                    WorkItem(
                        index=len(items),
                        lane_id=lane.lane_id,
                        mechanism=lane.mechanism,
                        phase=phase,
                        rung_index=rung_index,
                        seed_start=base + rung_index * lane.seeds_per_rung,
                        seed_count=lane.seeds_per_rung,
                    )
                )
    return tuple(items)


NEW_WORK_ITEMS = build_new_work_items()


def _item_path(root: Path, item: WorkItem) -> Path:
    return root / item.lane_id.lower() / item.phase / f"rung_{item.rung_index:03d}.json"


def _run_item_payload(item: WorkItem) -> dict[str, Any]:
    bed, runner = build_pair(item.mechanism)
    verdict_counts: dict[str, int] = {}
    control_clear_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    confirmations = 0
    for seed in range(item.seed_start, item.seed_start + item.seed_count):
        results = runner.run(bed, seed)
        receipt = runner.mint(results)
        if receipt.mechanism_id != item.mechanism:
            raise ValueError("new mechanisms runner receipt identity drifted")
        confirmations += int(receipt.is_confirmation)
        verdict_counts[receipt.verdict] = verdict_counts.get(receipt.verdict, 0) + 1
        for control in receipt.controls_cleared:
            control_clear_counts[control] = control_clear_counts.get(control, 0) + 1
        digest.update(receipt.digest().encode("ascii"))
    core = {
        "schema": RUNG_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "item": {
            "index": item.index,
            "lane_id": item.lane_id,
            "mechanism": item.mechanism,
            "phase": item.phase,
            "rung_index": item.rung_index,
            "seed_start": item.seed_start,
            "seed_count": item.seed_count,
        },
        "receipt_count": item.seed_count,
        "verdict_counts": verdict_counts,
        "control_clear_counts": control_clear_counts,
        "confirmation_count": confirmations,
        "receipt_digest_fold": digest.hexdigest(),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_rung(value: Mapping[str, Any], item: WorkItem) -> None:
    core = {key: row for key, row in value.items() if key != "result_sha256"}
    expected_item = {
        "index": item.index,
        "lane_id": item.lane_id,
        "mechanism": item.mechanism,
        "phase": item.phase,
        "rung_index": item.rung_index,
        "seed_start": item.seed_start,
        "seed_count": item.seed_count,
    }
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("new mechanisms rung seal drifted")
    if (
        value.get("schema") != RUNG_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("item") != expected_item
        or value.get("receipt_count") != item.seed_count
        or value.get("confirmation_count") != 0
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("new mechanisms rung identity or safety drifted")


def _item_by_index(index: int) -> WorkItem:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(NEW_WORK_ITEMS):
        raise ValueError(f"unknown new mechanisms work item index {index!r}")
    return NEW_WORK_ITEMS[index]


def run_item(index: int, root: Path = DEFAULT_ROOT) -> dict[str, Any]:

    item = _item_by_index(index)
    root = Path(root).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("new mechanisms queue root must remain inside the repository")
    set_process_label(f"mop-{item.lane_id.lower()}-{item.phase[:4]}-r{item.rung_index:03d}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    result = _run_item_payload(item)
    validate_rung(result, item)
    atomic_write_json(_item_path(root, item), result)
    result_with_timing = dict(result)
    result_with_timing["wall_seconds"] = time.perf_counter() - started
    return result_with_timing


def status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:

    root = Path(root).resolve()
    lane_progress: dict[str, dict[str, int]] = {
        lane.lane_id: {"complete": 0, "total": 0} for lane in NEW_LANES
    }
    complete = 0
    for item in NEW_WORK_ITEMS:
        lane_progress[item.lane_id]["total"] += 1
        path = _item_path(root, item)
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                continue
            validate_rung(value, item)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        lane_progress[item.lane_id]["complete"] += 1
        complete += 1
    return {
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "lane_progress": lane_progress,
        "counts": {
            "complete": complete,
            "total": len(NEW_WORK_ITEMS),
            "remaining": len(NEW_WORK_ITEMS) - complete,
        },
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="New Generation 1 mechanisms queue pilot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-item", help="run one work item and seal its rung receipt")
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    status_parser = subparsers.add_parser("status", help="summarize sealed rung receipts under root")
    status_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "run-item":
        result = run_item(args.index, args.root)
        print(json.dumps({key: result[key] for key in ("item", "verdict_counts", "wall_seconds")}, indent=2))
        return 0
    print(json.dumps(status(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
