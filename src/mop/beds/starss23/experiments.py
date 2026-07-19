"""STARSS23 onset, counting, DoA, and reproduction declarations for the shared science engine."""

from __future__ import annotations

from mop.science import PROGRAM, seal_record

CLAIM = "deterministic programmatic mechanics only; no capability or natural-data claim"
FORBIDDEN = ("proves", "demonstrates", "significant", "establishes capability", "generalizes")
CONTROLS = ("rate_matched_random", "always_on", "never_update")


def _record(*, experiment_id: str, schema: str, question: str, null: str, metric: str, direction: str,
            sesoi: float, seeds: tuple[int, ...], providers: tuple[str, ...], treatments: tuple[str, ...],
            split: dict[str, object], multiplicity: dict[str, object],
            verification: str) -> dict[str, object]:
    return seal_record({
        "id": experiment_id, "schema": schema, "stage": 3, "question": question, "null": null,
        "source": {"corpus": "STARSS23", "adapter": "starss23", "rights_clean": True,
                   "real_corpus": False, "identity_required": True},
        "split": split, "unit": {"experimental": "clip", "correlated_subsamples": "frames"},
        "providers": providers, "treatments": treatments,
        "controls": {"primary": "rate_matched_random", "arms": CONTROLS},
        "metric": {"name": metric, "direction": direction, "rule_provider": providers[-1]},
        "sesoi": {"value": sesoi, "provisional": False, "selection": "cost_benefit_before_test_scores"},
        "multiplicity": multiplicity,
        "budget": {"flop_ceiling": 60_000_000_000, "rule": "matched_inference_plus_charged_training"},
        "stop": {"decision": "paired_sign_flip_one_sided", "alpha": 0.05, "tie": "null",
                 "min_reproductions": 3, "single_run_never_promotes": True},
        "claims": {"ceiling": CLAIM, "forbidden_verbs": FORBIDDEN, "activation_allowed": False,
                   "scientific_promotion": False},
        "verification": {"provider": verification, "separate_process": True, "graded_logic_shared": False},
        "seeds": seeds, "program": PROGRAM,
    })


ONSET = _record(
    experiment_id="starss23_escs_event_formation", schema="mop-starss23-escs-bed/v1",
    question="does a trained gate place the same firing budget at onsets better than rate-matched random",
    null="candidate onset F1 does not exceed rate-matched-random onset F1",
    metric="onset F1 at DCASE plus or minus 200 ms collar", direction="higher", sesoi=0.05,
    seeds=(0, 1, 2, 3, 4), providers=("spectral_flux", "online_gate", "onset_referee"),
    treatments=("learning_progress_gate",),
    split={"rule": "room_disjoint", "train_fold": 3, "test_fold": 4},
    multiplicity={"kind": "none", "members": ()}, verification="starss23_onset_verifier",
)

COUNTING = _record(
    experiment_id="starss23_escs_source_counting", schema="mop-starss23-escs-count-bed/v1",
    question="does a trained gate lower coasted count MAE at the same re-estimation budget",
    null="candidate count MAE is not lower than rate-matched-random count MAE",
    metric="coasted concurrent-source count MAE, pooled frame micro-average", direction="lower", sesoi=0.02,
    seeds=(0, 1, 2, 3, 4), providers=("count_featurizer", "count_estimator", "count_referee"),
    treatments=("count_gate",), split={"rule": "room_disjoint", "train_fold": 3, "test_fold": 4},
    multiplicity={"kind": "none", "members": ()}, verification="starss23_count_verifier",
)

DOA = _record(
    experiment_id="starss23_escs_direction_of_arrival", schema="mop-starss23-doa-bed/v1",
    question="does a trained gate lower clip-macro great-circle DoA MAE under both gate architectures",
    null="candidate DoA MAE is not lower than rate-matched random under both architectures",
    metric="great-circle direction-of-arrival error in degrees, clip-macro", direction="lower", sesoi=1.0,
    seeds=(0, 1, 2, 3, 4), providers=("doa_featurizer", "doa_estimator", "doa_referee"),
    treatments=("doa_arch_a", "doa_arch_b"),
    split={"rule": "room_disjoint", "train_fold": 3, "test_fold": 4},
    multiplicity={"kind": "architectures", "members": ("doa_arch_a", "doa_arch_b"),
                  "all_required": True}, verification="starss23_doa_verifier",
)

COUNTING_DATA_SPLIT_REPRO = _record(
    experiment_id="starss23_escs_source_counting/data_split",
    schema="mop-starss23-escs-count-bed-repro-data-split/v1",
    question="does the counting result survive when train and test room folds are swapped",
    null="the counting advantage is specific to the original room partition",
    metric="coasted concurrent-source count MAE, pooled frame micro-average", direction="lower", sesoi=0.02,
    seeds=(10, 11, 12, 13, 14), providers=("count_featurizer", "count_estimator", "count_referee"),
    treatments=("count_gate",), split={"rule": "room_disjoint", "train_fold": 4, "test_fold": 3},
    multiplicity={"kind": "reproduction_axis", "members": ("data_split",), "of": COUNTING["id"]},
    verification="starss23_count_data_split_verifier",
)

RECORDS = (ONSET, COUNTING, DOA, COUNTING_DATA_SPLIT_REPRO)

__all__ = ["ONSET", "COUNTING", "DOA", "COUNTING_DATA_SPLIT_REPRO", "RECORDS"]
