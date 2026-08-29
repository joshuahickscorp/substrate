"""Instrument-liveness check for the final-revision discrimination bed.

Post-freeze supplementary check. Writes no evidence. Answers one question:
can the bed report a nonzero P3 effect at all, or is the exact tie an artifact
of a scorer that cannot move?

Method: run a small bed unperturbed, then re-run with one system's answer
function degraded on a single answer class, and compare the reported
P3 effect. A live instrument moves; a dead one stays at 0.0.
"""

from __future__ import annotations

import json
import sys

from substrate import final_revision_experiment as E
from substrate import final_revision_io as io

SEEDS = range(90_000, 90_006)
EPISODES = 2
KEY = "P3_selected_minus_strongest_persistent_alternative"

_kernel_answers = E._kernel_answers
_monolith_answers = E._monolith_answers


def _bed(split: str) -> dict:
    return E.run_discrimination_bed(split=split, seeds=SEEDS, episodes_per_family=EPISODES)


def _effect(document: dict) -> dict:
    row = document["effects"][KEY]
    return {
        "mean_paired_effect": row["mean_paired_effect"],
        "confidence_interval_95": row["confidence_interval_95"],
        "clears_sesoi": row.get("clears_sesoi"),
    }


def main() -> None:
    modes = {}

    modes["control_unperturbed"] = _effect(_bed("instrument_liveness_control"))

    def broken_monolith(monolith, family):
        answers = _monolith_answers(monolith, family)
        answers[0] = "__degraded__"
        return answers

    E._monolith_answers = broken_monolith
    try:
        modes["baseline_S2_degraded_on_class_0"] = _effect(_bed("instrument_liveness_baseline_degraded"))
    finally:
        E._monolith_answers = _monolith_answers

    def broken_kernel(kernel, family):
        answers = _kernel_answers(kernel, family)
        answers[0] = "__degraded__"
        return answers

    E._kernel_answers = broken_kernel
    try:
        modes["candidate_degraded_on_class_0"] = _effect(_bed("instrument_liveness_candidate_degraded"))
    finally:
        E._kernel_answers = _kernel_answers

    control = modes["control_unperturbed"]
    baseline_degraded = modes["baseline_S2_degraded_on_class_0"]
    candidate_degraded = modes["candidate_degraded_on_class_0"]

    checks = {
        "control_is_exact_tie": control["mean_paired_effect"] == 0.0 and control["confidence_interval_95"] == [0.0, 0.0],
        "degrading_baseline_moves_effect_positive": baseline_degraded["mean_paired_effect"] > 0.0,
        "degrading_candidate_moves_effect_negative": candidate_degraded["mean_paired_effect"] < 0.0,
        "moved_effects_have_nondegenerate_intervals": (
            baseline_degraded["confidence_interval_95"] != [0.0, 0.0]
            and candidate_degraded["confidence_interval_95"] != [0.0, 0.0]
        ),
    }

    document = io.authority(
        "substrate-final-revision-instrument-liveness/v1",
        {
            "purpose": "establish that the discrimination bed can report a nonzero P3 effect, so the frozen exact tie is a behavioral result rather than a dead scorer",
            "status_note": "post-freeze supplementary check; not a required deliverable and not a scored endpoint",
            "seeds": [int(seed) for seed in SEEDS],
            "episodes_per_family": EPISODES,
            "perturbation": "replace one answer class of one system with a constant wrong value; no threshold, seed, split, generator or scoring rule is changed",
            "modes": modes,
            "checks": checks,
            "all_pass": all(checks.values()),
        },
        status="complete" if all(checks.values()) else "invalid",
    )
    print(json.dumps(document, sort_keys=True))
    sys.exit(0 if document["all_pass"] else 1)


if __name__ == "__main__":
    main()
