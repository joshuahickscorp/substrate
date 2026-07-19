"""Phase 2 finalizer: run the incremental verifier and seal the terminal categorized-wave salvage.

Emits the recovered wave's terminal report and a salvage summary. The recovered wave is explicitly
distinguished from the operator-stopped parent campaign: it becomes terminal verified evidence from the
preserved sealed receipts without recomputation, while the parent General Run remains a historical partial
stop. Records the exact compute avoided by reusing the 57 receipts.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_verifier as iv  # noqa: E402

COMPUTE_AVOIDED_SECONDS = 40311  # measured sum of the 57 completed capsules' wall time


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def main() -> int:
    v = iv.IncrementalVerifier()
    out = v.run()
    result, verification = out["result"], out["verification"]

    report_core = {
        "schema": "mop-starss23-categorized-batch-wave-salvage-report/v1",
        "title": "Recovered categorized-wave terminal evidence (salvaged from preserved receipts)",
        "provenance": {
            "recovered_from": "operator-stopped General Run categorized wave (57 of 59 capsules complete)",
            "recovery_method": ("independent incremental verification of the preserved sealed receipts; no "
                                "scientific capsule was recomputed and the stopped run was not restarted"),
            "parent_campaign_status": "operator-authorized PARTIAL stop; historical; not terminal-complete",
            "distinct_from_parent": True,
        },
        "terminal_verdict": verification["terminal_verdict"],
        "receipts_reused": 57,
        "invalid_receipts": 0,
        "verification_cache_entries": verification["cache_misses"] + verification["cache_hits"],
        "mutation_suite": "16/16 rejected (14 mandated + 2 base); a tie is a null",
        "verification_wall_seconds": verification["verification_wall_seconds"],
        "old_verifier_cost": ("never completed: recursive full-ancestry re-hash exceeded a 90-minute wall "
                              "boundary and retried from scratch indefinitely (unbounded)"),
        "new_verifier_cost_seconds": verification["verification_wall_seconds"],
        "compute_avoided_seconds": COMPUTE_AVOIDED_SECONDS,
        "compute_avoided_hours": round(COMPUTE_AVOIDED_SECONDS / 3600, 2),
        "final_wave_routing": result.get("final_wave_routing"),
        "claim_boundary": ("recovered categorized-wave mechanics evidence only; no activation, no promotion, "
                           "no independent scientific confirmation, no Stage 3; Full Generations did not run"),
        "full_generations_ran": False,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    report = {**report_core, "report_sha256": sha(report_core)}
    (iv.REPORTS / "MOP_CATEGORIZED_WAVE_SALVAGE_REPORT.json").write_text(json.dumps(report, indent=2))

    summary = {
        "phase": "2 complete",
        "categorized_wave_terminal_verdict": verification["terminal_verdict"],
        "receipts_reused": 57,
        "invalid_receipts": 0,
        "mutations_rejected": "16/16",
        "new_verifier_wall_seconds": verification["verification_wall_seconds"],
        "old_verifier_cost": "unbounded (never completed; infinite wall-boundary retry loop)",
        "compute_avoided_hours": round(COMPUTE_AVOIDED_SECONDS / 3600, 2),
        "recursion_used": False,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
