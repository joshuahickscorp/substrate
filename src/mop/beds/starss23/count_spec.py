"""STARSS23 counting axis, expressed as one declaration on the shared science engine (spec sections 10, 11).

This is the collapsed form of the counting bed: the scientific DECLARATION only. Everything it used to
hand-expand across count_prereg (prereg assembly and write), count_harness (arm pairing, budget ceilings,
paired deltas, decision), and count_producer (artifact assembly and sealing) is now supplied by
``mop.science.engine``. The unique counting mathematics (featurizer, causal re-estimator, online gate) stays
in its own provider modules and is bound through an ArmRunner; the independent verifier graded recompute
stays in count_verifier. The declared values are the exact preregistered counting-bed values.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from mop.science.spec import ExperimentSpec, MetricSpec

# Declared, preregistered counting-bed values (previously scattered as module constants across the axis).
COUNT_SESOI_MAE = 0.02
N_PAIRED_SEEDS = 5
MIN_REPRODUCTIONS = 3
MAX_GATE_PARAMS = 4096

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"

COUNT_SPEC = ExperimentSpec(
    experiment_id="starss23_escs_source_counting",
    schema="mop-starss23-escs-count-bed/v2-engine",
    stage=3,
    question=(
        "does a bounded causal re-estimation gate reduce coasted concurrent-source count MAE versus its "
        "rate-matched random control at matched inference budget"
    ),
    null_hypothesis=(
        "the candidate re-estimation gate does not reduce coasted count MAE versus the rate-matched random "
        "control"
    ),
    metric=MetricSpec(name="coasted concurrent-source count MAE", direction="lower", sesoi=COUNT_SESOI_MAE),
    seeds=tuple(range(1, N_PAIRED_SEEDS + 1)),
    arms=(ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE),
    primary_control=ARM_RATE_MATCHED_RANDOM,
    decision_rule="paired_sign_flip_one_sided",
    min_reproductions=MIN_REPRODUCTIONS,
    claim_ceiling="deterministic programmatic mechanics only; no capability or natural-data claim",
    param_ceiling=MAX_GATE_PARAMS,
    allowed_claim_verbs=("consistent with", "suggestive"),
    forbidden_claim_verbs=("proves", "demonstrates", "establishes capability", "generalizes"),
    extra={"bed_id": "starss23_escs_source_counting", "voc_window_frames": 1},
)

__all__ = ["COUNT_SPEC", "COUNT_SESOI_MAE", "N_PAIRED_SEEDS", "MIN_REPRODUCTIONS", "MAX_GATE_PARAMS",
           "ARM_CANDIDATE", "ARM_RATE_MATCHED_RANDOM", "ARM_ALWAYS_ON", "ARM_NEVER_UPDATE"]
