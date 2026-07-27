"""Generation 3 terminal synthesis from the sealed C1 and C3 results. House style: no dashes."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

W = Path("/Users/scammermike/Downloads/mop-gen3")
R = W / "gen3/reports"


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def L(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main():
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(W)).stdout.strip()
    c1 = {os.path.basename(f): L(f) for f in sorted(glob.glob(str(R / "MOP_GEN3_C1_*.json")))}
    c3 = L(R / "MOP_GEN3_C3_PRECOMPUTE_RESULT.json")

    # C1 verdict across stages
    c1_classes = {k: v["classification"] for k, v in c1.items()}
    headroom = next((v for k, v in c1.items() if "A2_headroom" in k), None)

    syn = {
        "schema": "mop-generation3-terminal-synthesis/v1", "source_commit": commit,
        "program_id": "mop-generation3-discovery-v1", "branch": "agent/mop-generation3-discovery",
        "streams": {
            "C1_p1r_priority": {
                "premise": "P1R replay value as a soft SAMPLING PRIORITY over a fixed independently-maintained buffer, keep policy separate from replay-frequency",
                "sources_tested": {"EMNIST_full_budget": c1_classes.get("MOP_GEN3_C1_B_canary_emnist_RESULT.json"),
                                   "HAR_non_image": c1_classes.get("MOP_GEN3_C1_C_confirm_har_RESULT.json"),
                                   "EMNIST_high_forgetting": (headroom or {}).get("classification", "pending")},
                "key_finding": ("INVALID BED across all three tested regimes. At normal training budgets GDumb-uniform "
                                "captures nearly all oracle value (EMNIST oracle headroom about +0.01 unstable, HAR "
                                "about +0.006; HAR activity recognition is too separable, all methods about 0.92). "
                                "The precompute headroom (+0.169) was a TWO-SEED ARTIFACT: at the same high-forgetting "
                                "regime measured over five stable seeds, oracle_priority (0.320) beats uniform (0.309) "
                                "by only +0.010, below the 0.02 validity threshold, and shuffled/random concentration "
                                "(0.318/0.314) match the oracle. So GDumb-uniform captures essentially all recoverable "
                                "value at every stable budget, making the bed invalid for the C1 question. Separately, "
                                "the LEARNED P1R priority is actively HARMFUL where a tiny oracle edge exists (0.18-0.19, "
                                "worse than uniform, loss, oracle, shuffled, and random; per-task effect lcb -0.15 to "
                                "-0.21): concentrating replay by a noisy learned value reduces diversity and hurts."),
                "meta_finding": ("the two-seed precompute gate produced a false-positive headroom signal that a proper "
                                 "five-seed power analysis overturned; adequate seeding (mandate gate 7) is decisive."),
                "headroom_regime_detail": (headroom or {}).get("detail"),
                "terminal_classification": "invalid_bed (no valid bed found; learned priority harmful where a tiny oracle edge exists)",
            },
            "C3_model_error_aware_simulation": {
                "premise": "allocate bounded simulation only where expected decision benefit exceeds expected model-error cost; condition depth/trust on estimated rollout error",
                "precompute_verdict": (c3 or {}).get("verdict"),
                "gate5_oracle_error_aware_vs_reactive": (c3 or {}).get("gate5_oracle_error_aware_vs_reactive"),
                "gate6_residual_vs_planner": (c3 or {}).get("gate6_residual_vs_best_planner"),
                "informative_units": (c3 or {}).get("informative_units"),
                "key_finding": ("Terminated at the precompute gate. Only 2 of 10 env units are informative (both "
                                "CartPole; Acrobot and MountainCar are unsolvable by a linear model), and on those "
                                "the oracle error-aware allocator is WORSE than reactive (mean about -0.54). Even a "
                                "perfect model-error oracle cannot make learned-model simulation help on these "
                                "environments. C3 is closed before any principal canary, continuing the Gen2 S1 null."),
                "terminal_classification": "precompute_gate_failure_no_canary",
            },
        },
        "surviving_subsystem": "none new. P1R remains the strongest surviving internal hypothesis (Gen2), unchanged: same-team benchmark positives bounded by an external replication null, no downstream license.",
        "construction_integration_activation": "unlicensed; activation false (unchanged from Gen2)",
        "forbidden_claims": ["P1R-priority is a validated replay mechanism (headroom is budget-specific and not "
                             "shown recoverable by a learned priority on a valid stable bed)",
                             "C3 model-error-aware simulation has any headroom (it does not)",
                             "any new mechanism is confirmed or activated"],
        "exact_next_frontier": ("If EMNIST_high_forgetting shows a learned priority robustly beating loss-priority "
                                "and uniform on a VALID bed, C1 becomes a bounded replay-subsystem candidate needing "
                                "external-method replication. Otherwise the null map's replay and simulation premises "
                                "are exhausted at this compute scale, and the next genuinely new premise must target "
                                "a domain with a strong non-image sequential source that has real forgetting AND "
                                "oracle headroom the established method leaves open."),
    }
    syn["sha256"] = sha(syn)
    (W / "gen3/MOP_GENERATION3_TERMINAL_SYNTHESIS.json").write_text(json.dumps(syn, indent=2))
    print("C1 classes:", c1_classes)
    print("C3 verdict:", (c3 or {}).get("verdict"))
    print("headroom-regime C1:", (headroom or {}).get("classification", "pending"))
    return syn


if __name__ == "__main__":
    main()
