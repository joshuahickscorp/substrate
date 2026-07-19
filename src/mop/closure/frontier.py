"""The machine-readable current-state authority: ``proof/MOP_CURRENT_FRONTIER.json``.

Current-state drift is eliminated by making one generated artifact the authority for the facts that keep
going stale in prose: the current facet count, the P4 and P5 state, the active heavy lane, the current
worker policy, the latest STARSS23 outcomes, the hardware boundary, and the exact next executable command.
Facts are DERIVED from live artifacts where possible (the atlas facet count, the General Run status stage)
and bound with source hashes, so a stale generated frontier is detectable and regeneratable.

This module is read-only over live artifacts and sealed proofs; it never edits them.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

MOP_CURRENT_FRONTIER_SCHEMA = "mop-current-frontier/v1"


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_current_frontier(
    *, repo_root: Path, runs_root: Path | None = None, timestamp: str
) -> dict[str, Any]:
    """Derive the current-state authority from live artifacts. ``timestamp`` is passed, not clock-read."""

    repo_root = repo_root.resolve()
    runs_root = (runs_root or (repo_root / "runs")).resolve()

    atlas_path = repo_root / "proof" / "MOP_POTENTIAL_ATLAS.json"
    atlas = _load(atlas_path) or {}
    facets = atlas.get("facets") or []
    facet_count = len(facets) if isinstance(facets, list) else None

    # active heavy lane is the live General Run (its stage), derived from its sealed status.
    gr_status = _load(runs_root / "generation1" / "general-run" / "current_status.json") or {}
    gr_state = gr_status.get("state")
    gr_stage = gr_status.get("stage")
    gr_counts = gr_status.get("counts") or {}
    gr_terminal = gr_state in ("complete", "failure_hold", "integrity_hold", "drained")

    # latest STARSS23 outcomes, keyed to the sealed bed artifacts (read-only; verdicts not overwritten).
    starss23 = {
        "onset_localization": {
            "artifact": "STARSS23_ESCS_BED.json",
            "outcome": "walled: energy carries no onset-localizing information (seven nulls)",
        },
        "source_counting": {
            "artifact": "STARSS23_COUNTING_BED.json",
            "outcome": "signal did not survive bias-independent reproduction (2 of 4 below the 3 bar)",
            "reproductions": "STARSS23_COUNTING_REPRO_{scoring_unit,gate_arch,featurizer_estimator,split}",
        },
        "direction_of_arrival": {
            "artifact": "STARSS23_DOA_BED.json",
            "outcome": "clean null on both gate architectures; not architecture-fragile",
        },
    }

    content: dict[str, Any] = {
        "schema": MOP_CURRENT_FRONTIER_SCHEMA,
        "timestamp": timestamp,
        "claim_scope": "machine-readable current-state authority; supersedes stale current-facing prose",
        "facet_count": facet_count,
        "facet_count_source": "proof/MOP_POTENTIAL_ATLAS.json",
        "stale_facet_count_rejected": 37,
        "p4_state": "governor-closed: all 12 cells across five seeds; a bounded pilot; not promotable",
        "p4_is_active_heavy_lane": False,
        "p5_state": "smoke admission failed closed on local category-1 state; no measured hardware blocker; "
        "next heavy boot is p5smoke_cpu leg4 when the admission gate is healthy",
        "active_heavy_lane": {
            "program_id": "general-run",
            "state": gr_state,
            "stage": gr_stage,
            "terminal": gr_terminal,
            "counts": gr_counts,
            "note": "the live General Run (Generation 1 successor horizon chain) is the exclusive heavy lane",
        },
        "worker_policy": {
            "controller": "mop.studio.dynamic_worker_controller",
            "range": "1 to 20 workers; measured seeded-hash optimum at 20; 24 regressed",
            "hawking": "sheds worker count to a tiny reserve and lowers QoS; never signals Hawking",
            "obsolete_descriptions_rejected": ["fixed eight-worker pool", "static ideal-eight-worker hours"],
        },
        "starss23_latest_outcomes": starss23,
        "hardware_boundary": {
            "studio_earned": False,
            "statement": "no current receipt earns a Studio boundary; categories 8 and 9 are empty; the "
            "evidence does not justify a Studio purchase",
        },
        "next_executable_command": {
            "while_general_run_active": "observe only; run no heavy compute in parallel with the General Run",
            "on_general_run_terminal": ".venv/bin/python -m mop.closure.producer --execute",
            "current_facing_authority": "proof/MOP_CURRENT_FRONTIER.json",
        },
        "obsolete_commands_rejected": [
            "scripts/mop_generation1_successor_long_chain.py start --execute (superseded)",
            "P4 five-seed response surface as the active next heavy step (P4 is closed)",
        ],
        "source_hashes": {
            "atlas": _sha256_file(atlas_path),
            "general_run_status": gr_status.get("status_sha256"),
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**content, "seal": {"sha256": canonical_sha256(content)}}


def write_current_frontier(frontier: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(canonical_bytes(frontier))
    return out_path
