from __future__ import annotations

from typing import Any

from mop.experiments.base import PROGRAM, ExperimentSpec, bind
from mop.science.budget import ARM_NEVER_UPDATE, BudgetPolicy

from . import CLAIM_SCOPE, FLOP_CEILING

COUNT_BED_ID = "starss23_escs_source_counting"
COUNT_BUDGET_POLICY = BudgetPolicy(
    "mop-starss23-count-harness/v1",
    COUNT_BED_ID,
    "mae",
    "reestimations",
    "lower",
    ARM_NEVER_UPDATE,
    CLAIM_SCOPE,
    FLOP_CEILING,
    delta_key="delta_mean_mae_control_minus_candidate",
)


def _record(
    experiment_id: str,
    name: str,
    question: str,
    null: str,
    metric: str,
    provider: str,
    verifier: str,
    *,
    split: dict[str, Any],
    treatments: list[str],
    sesoi: float,
    multiplicity: dict[str, Any],
) -> ExperimentSpec:
    return bind(
        {
            "id": experiment_id,
            "name": name,
            "question": question,
            "null_hypothesis": null,
            "metrics": [metric],
            "controls": ["rate-matched-random", "always-on", "never-update"],
            "source": {"corpus": "STARSS23", "identity_required": True},
            "split": split,
            "unit": {"experimental": "clip", "correlated_subsamples": "frames"},
            "treatments": treatments,
            "sesoi": {"value": sesoi, "provisional": False},
            "multiplicity": multiplicity,
            "budget": {"flop_ceiling": FLOP_CEILING, "rule": "matched-charged-compute"},
            "stop": {"rule": "paired-sign-flip", "tie": "null", "min_reproductions": 3},
            "claim_ceiling": {
                "statement": CLAIM_SCOPE,
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_confirmation": False,
            },
            "provider": provider,
            "verifier": verifier,
            "program": list(PROGRAM),
            "status": "historical",
            "resource_tier": "sealed-history",
        },
        None,
        None,
    )


ONSET = _record(
    "starss23_escs_event_formation",
    "event_formation",
    "does a trained gate place the same firing budget at onsets better than rate-matched random",
    "candidate onset F1 does not exceed rate-matched-random onset F1",
    "onset_f1_dcase_200ms_collar",
    "starss23_onset_provider",
    "starss23_onset_verifier",
    split={"rule": "room-disjoint", "train_fold": 3, "test_fold": 4},
    treatments=["learning-progress-gate"],
    sesoi=0.05,
    multiplicity={"rule": "none"},
)
COUNTING = _record(
    COUNT_BED_ID,
    "source_counting",
    "does a trained gate lower coasted count MAE at the same re-estimation budget",
    "candidate count MAE is not lower than rate-matched-random count MAE",
    "coasted_count_mae_frame_micro_average",
    "mop.beds.starss23.count_producer.produce_real_count_artifact",
    "mop.beds.starss23.count_verifier.verify_count_artifact",
    split={"rule": "room-disjoint", "train_fold": 3, "test_fold": 4},
    treatments=["count-gate"],
    sesoi=0.02,
    multiplicity={"rule": "none"},
)
DOA = _record(
    "starss23_escs_direction_of_arrival",
    "direction_of_arrival",
    "does a trained gate lower clip-macro great-circle DoA MAE under both gate architectures",
    "candidate DoA MAE is not lower than rate-matched random under both architectures",
    "great_circle_doa_mae_degrees_clip_macro",
    "starss23_doa_provider",
    "starss23_doa_verifier",
    split={"rule": "room-disjoint", "train_fold": 3, "test_fold": 4},
    treatments=["doa-architecture-a", "doa-architecture-b"],
    sesoi=1.0,
    multiplicity={"rule": "both-architectures-required"},
)
COUNTING_DATA_SPLIT_REPRO = _record(
    f"{COUNT_BED_ID}/data_split",
    "source_counting_data_split_reproduction",
    "does the counting result survive when train and test room folds are swapped",
    "the counting advantage is specific to the original room partition",
    "coasted_count_mae_frame_micro_average",
    "starss23_count_data_split_provider",
    "starss23_count_data_split_verifier",
    split={"rule": "room-disjoint", "train_fold": 4, "test_fold": 3},
    treatments=["count-gate"],
    sesoi=0.02,
    multiplicity={"rule": "reproduction-axis", "of": COUNT_BED_ID},
)
STARSS_RECORDS = (ONSET, COUNTING, DOA, COUNTING_DATA_SPLIT_REPRO)

__all__ = [
    "COUNT_BED_ID",
    "COUNT_BUDGET_POLICY",
    "COUNTING",
    "COUNTING_DATA_SPLIT_REPRO",
    "DOA",
    "ONSET",
    "STARSS_RECORDS",
]
