#!/usr/bin/env python
"""Map the frozen SANPO smoke pack's session attributes onto DR1/CM1 factor requirements.

Metadata-only: this script never decodes a pixel and never opens an official-test session
directory, description, or frame. It reads the frozen bridge plan, the intake receipts, and the
train/validation session descriptions, then reports which attribute pairs could serve as DR1/CM1
bound-factor designs, at what cell coverage, and how many sessions a promotable design would need.

The output is a hash-bound receipt, proof/SANPO_DR1_CM1_ATTRIBUTE_MAP.json. Its claim scope is
mapping mechanics only: with eight development sessions nothing here is promotable, and the map
says so explicitly. A tie or an empty cell is reported as-is; fail closed on any hash mismatch.

Usage: python scripts/sanpo_attribute_map.py [--out proof/SANPO_DR1_CM1_ATTRIBUTE_MAP.json]

No em dashes or en dashes (BLACKHOLE.md). Engineering vocabulary only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "sanpo_real_smoke_v0"
BRIDGE_PREFLIGHT = REPO_ROOT / "proof" / "SANPO_CUSTOM_SUBSTRATE_BRIDGE_PREFLIGHT.json"
INTAKE_RECEIPT = REPO_ROOT / "proof" / "SANPO_REAL_SMOKE_INTAKE.json"
SCHEMA = "mop-sanpo-dr1-cm1-attribute-map/v1"

# DR1 curation gate defaults (src/mop/studio/dr1_schedule.py) and CM1 registry requirements:
# two independently decodable bound factors, held-out combinations, minimum replicates per cell.
DR1_MIN_PER_CELL = 16
CM1_MIN_FACTORS = 2

# Session-level scalar attributes considered as candidate factors. List-valued attributes
# (environment_types, weather_conditions, elevation_changes, ground_appearances) are reported
# but only their first element is offered as a factor level, stated in the receipt.
SCALAR_ATTRIBUTES = (
    "human_traffic",
    "vehicular_traffic",
    "animal_traffic",
    "num_obstacles",
    "visibility",
    "ego_motion",
    "motion_blur",
)
LIST_ATTRIBUTES = (
    "environment_types",
    "weather_conditions",
    "elevation_changes",
    "ground_appearances",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "SANPO_DR1_CM1_ATTRIBUTE_MAP.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    problems: list[str] = []

    preflight = _read_json(BRIDGE_PREFLIGHT)
    intake = _read_json(INTAKE_RECEIPT)
    splits = _read_json(SOURCE_ROOT / "splits.json")
    if splits.get("schema") != "mop-sanpo-real-explicit-splits/v1":
        problems.append("splits schema mismatch")
    if splits.get("official_test_tuning_allowed") is not False:
        problems.append("splits do not seal official test tuning")

    roles = splits.get("roles") or {}
    development_ids = list(roles.get("train") or []) + list(roles.get("validation") or [])
    test_ids = list(roles.get("test") or [])
    if len(development_ids) != 8 or len(test_ids) != 2:
        problems.append(
            f"expected 8 development and 2 test sessions, found {len(development_ids)} and {len(test_ids)}"
        )

    # Metadata-only read of development sessions; test session directories are never opened.
    sessions: list[dict] = []
    is_park_by_session = {
        str(record.get("session_id")): bool(record.get("is_park"))
        for record in intake.get("sessions", [])
        if record.get("role") in ("train", "validation")
    }
    for session_id in development_ids:
        description_path = SOURCE_ROOT / "sessions" / session_id / "description.json"
        description = _read_json(description_path)
        metadata = description.get("session_video_metadata") or {}
        if not metadata:
            problems.append(f"session {session_id} has no session_video_metadata")
            continue
        row: dict[str, object] = {
            "session_id": session_id,
            "role": "train" if session_id in (roles.get("train") or []) else "validation",
            "description_sha256": _sha256_file(description_path),
            "is_park": is_park_by_session.get(session_id),
        }
        for key in SCALAR_ATTRIBUTES:
            row[key] = metadata.get(key)
        for key in LIST_ATTRIBUTES:
            value = metadata.get(key)
            row[key] = value
            row[f"{key}_primary"] = value[0] if isinstance(value, list) and value else None
        sessions.append(row)

    # Candidate factor set: is_park plus scalar attributes plus primary elements of list attributes.
    factor_keys = ["is_park", *SCALAR_ATTRIBUTES, *[f"{key}_primary" for key in LIST_ATTRIBUTES]]
    factor_levels: dict[str, dict[str, int]] = {}
    for key in factor_keys:
        counts = Counter(str(row.get(key)) for row in sessions)
        factor_levels[key] = dict(sorted(counts.items()))

    # Pairwise cell coverage over the eight development sessions.
    pairs = []
    for key_a, key_b in combinations(factor_keys, 2):
        levels_a = {str(row.get(key_a)) for row in sessions}
        levels_b = {str(row.get(key_b)) for row in sessions}
        if len(levels_a) < 2 or len(levels_b) < 2:
            continue
        cells = Counter((str(row.get(key_a)), str(row.get(key_b))) for row in sessions)
        total_cells = len(levels_a) * len(levels_b)
        occupied = len(cells)
        min_cell = min(cells.values()) if occupied == total_cells else 0
        sessions_needed = total_cells * DR1_MIN_PER_CELL
        pairs.append(
            {
                "factor_a": key_a,
                "factor_b": key_b,
                "levels_a": sorted(levels_a),
                "levels_b": sorted(levels_b),
                "total_cells": total_cells,
                "occupied_cells": occupied,
                "complete_design": occupied == total_cells,
                "min_sessions_per_cell": min_cell,
                "dr1_min_per_cell": DR1_MIN_PER_CELL,
                "dr1_gate_met_at_smoke_scale": min_cell >= DR1_MIN_PER_CELL,
                "sessions_needed_for_dr1_gate": sessions_needed,
                "cells": {f"{a}|{b}": count for (a, b), count in sorted(cells.items())},
            }
        )
    pairs.sort(key=lambda item: (-int(item["complete_design"]), -item["min_sessions_per_cell"]))

    complete = [p for p in pairs if p["complete_design"]]
    best = complete[0] if complete else None
    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "metadata-only; no pixel decode; official-test sessions never opened",
        "bindings": {
            "bridge_preflight_sha256": _sha256_file(BRIDGE_PREFLIGHT),
            "bridge_plan_identity_sha256": preflight.get("plan_identity_sha256"),
            "content_set_sha256": preflight.get("content_set_sha256"),
            "intake_receipt_sha256": _sha256_file(INTAKE_RECEIPT),
            "splits_sha256": _sha256_file(SOURCE_ROOT / "splits.json"),
        },
        "development_sessions": sessions,
        "test_sessions_excluded": {
            "count": len(test_ids),
            "metadata_read": False,
            "pixels_decoded": False,
            "note": "official test stays sealed; not even description.json was opened",
        },
        "factor_levels": factor_levels,
        "candidate_factor_pairs": pairs,
        "dr1_cm1_verdict": {
            "cm1_min_factors": CM1_MIN_FACTORS,
            "complete_two_factor_designs_at_n8": len(complete),
            "best_complete_pair": (
                {k: best[k] for k in ("factor_a", "factor_b", "total_cells", "min_sessions_per_cell")}
                if best
                else None
            ),
            "any_pair_meets_dr1_min_per_cell": any(p["dr1_gate_met_at_smoke_scale"] for p in pairs),
            "statement": (
                "The eight-session smoke pack maps which SANPO session attributes could form DR1/CM1 "
                "bound-factor designs and proves the mapping mechanics. No pair reaches the DR1 "
                "minimum of 16 sessions per cell at this scale, so every design here is nonpromotable. "
                "Promotion requires the session-diverse full intake sized by "
                "sessions_needed_for_dr1_gate for the chosen pair, with session-disjoint splits."
            ),
        },
        "claim_boundary": {
            "natural_video_scientific_claim": False,
            "scientific_promotion": False,
            "test_sessions_touched": False,
            "statement": (
                "Attribute-mapping mechanics over frozen, hash-bound development metadata only. "
                "Not a representation, capability, or natural-video result."
            ),
        },
        "problems": problems,
        "all_ok": not problems,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "all_ok": receipt["all_ok"], "problems": problems}, indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
