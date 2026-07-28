"""Matched-history analysis for Substrate v5 pilot and principal evidence."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from substrate import v5config as C
from substrate import v5experiment as E
from substrate import v5stats

Table = dict[int, dict[str, dict[int, dict[str, Any]]]]

ENDPOINTS: dict[str, dict[str, Any]] = {
    "H_M1": {
        "name": "persistent_structured_long_history",
        "phases": (16, 18, 19),
        "controls": ("transcript_replay", "fresh_reset", "single_multimodal_model"),
    },
    "H_M2": {
        "name": "continuous_video_object_event_value",
        "phases": (2, 3),
        "controls": ("no_video_state",),
    },
    "H_M3": {
        "name": "spatial_and_3d_value",
        "phases": (6, 7),
        "controls": ("no_3d_state",),
    },
    "H_M4": {
        "name": "cross_modal_binding_value",
        "phases": (1, 5, 11),
        "controls": ("disconnected_specialists",),
    },
    "H_M5": {
        "name": "active_perception_value",
        "phases": (9,),
        "controls": ("no_active_perception",),
    },
    "H_M6": {
        "name": "model_fabric_routing_value",
        "phases": (10,),
        "controls": (
            "fixed_model_routing",
            "largest_model_always",
            "disconnected_specialists",
        ),
    },
    "H_M7": {
        "name": "model_support_value",
        "phases": (10,),
        "controls": ("no_model_support",),
    },
    "H_M8": {
        "name": "body_schema_value",
        "phases": (8,),
        "controls": ("no_body_schema",),
    },
    "H_M9": {
        "name": "verified_continual_learning_value",
        "phases": (12,),
        "controls": ("no_continual_learning",),
    },
    "H_M10": {
        "name": "human_multimodal_teaching_value",
        "phases": (11,),
        "controls": ("no_human_multimodal_teaching",),
    },
    "H_M11": {
        "name": "selected_kernel_integrated_value",
        "phases": tuple(range(20)),
        "controls": ("v4_cognitive_core_control",),
    },
    "H_M12": {
        "name": "continuing_entity_integrated_advantage",
        "phases": (17, 18, 19),
        "controls": (
            "single_multimodal_model",
            "disconnected_specialists",
            "more_compute_disconnected",
        ),
    },
    "H_M14": {
        "name": "model_replacement_continuity",
        "phases": (13, 14, 16),
        "controls": ("fresh_reset", "single_multimodal_model"),
    },
    "H_M15": {
        "name": "integrated_coherence_under_change",
        "phases": (15, 16, 18, 19),
        "controls": ("fresh_reset", "transcript_replay", "disconnected_specialists"),
    },
}


def evaluate_histories(
    split: str,
    history_seeds: Iterable[int],
    *,
    arms: Iterable[str] = C.ARMS,
) -> Table:
    selected_arms = tuple(arms)
    if "full_v5" not in selected_arms:
        raise v5stats.Refused("full_v5 is required for matched contrasts")
    table: Table = {}
    for seed in history_seeds:
        table[int(seed)] = {}
        for arm in selected_arms:
            if arm not in C.ARMS:
                raise v5stats.Refused(f"unknown arm {arm!r}")
            table[int(seed)][arm] = {
                phase: E.phase_result(
                    split=split,
                    history_seed=int(seed),
                    arm=arm,
                    phase_index=phase,
                )
                for phase in range(len(C.PHASES))
            }
    return table


def table_from_receipts(receipts: Iterable[Mapping[str, Any]]) -> Table:
    table: Table = {}
    for receipt in receipts:
        unit = receipt["unit"]
        seed = int(unit["history_seed"])
        arm = str(unit["arm"])
        history = table.setdefault(seed, {})
        phases = history.setdefault(arm, {})
        for row in receipt["phase_results"]:
            index = int(row["phase_index"])
            if index in phases:
                raise v5stats.Refused(
                    f"duplicate phase {index} for history {seed} arm {arm}"
                )
            phases[index] = dict(row)
    expected = set(range(len(C.PHASES)))
    for seed, history in table.items():
        for arm, phases in history.items():
            if set(phases) != expected:
                raise v5stats.Refused(
                    f"incomplete phases for history {seed} arm {arm}"
                )
    return table


def _metric(
    table: Table,
    arm: str,
    phases: tuple[int, ...],
) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed, history in table.items():
        if arm not in history:
            raise v5stats.Refused(f"history {seed} is missing arm {arm}")
        values[seed] = statistics.fmean(
            float(history[arm][phase]["utility"]) for phase in phases
        )
    return values


def effects(table: Table) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for hypothesis, authority in ENDPOINTS.items():
        phases = tuple(int(value) for value in authority["phases"])
        results[hypothesis] = v5stats.paired_contrast(
            _metric(table, "full_v5", phases),
            {
                control: _metric(table, control, phases)
                for control in authority["controls"]
            },
            str(authority["name"]),
            sesoi=C.SESOI,
        )
        results[hypothesis]["hypothesis"] = C.HYPOTHESES[hypothesis]
        results[hypothesis]["phases"] = [C.PHASES[index] for index in phases]
    retention_values = {
        seed: statistics.fmean(
            (
                float(history["full_v5"][0]["utility"]),
                float(history["full_v5"][19]["utility"]),
            )
        )
        - 0.78
        for seed, history in table.items()
    }
    results["H_M13"] = v5stats.paired_effect(
        retention_values.values(),
        "v4_structural_reflective_retention_margin",
        sesoi=C.SESOI,
    )
    results["H_M13"].update(
        {
            "hypothesis": C.HYPOTHESES["H_M13"],
            "retention_floor": 0.78,
            "full_arm": "full_v5",
            "controls": ["frozen_v4_retention_floor"],
        }
    )
    ordered = {name: results[name] for name in C.HYPOTHESES}
    correction = v5stats.holm(
        {name: float(row["exact_sign_p"]) for name, row in ordered.items()}
    )
    for name, row in ordered.items():
        row["holm_reject_zero"] = correction["rows"][name]["reject_zero"]
        row["passes"] = bool(row["clears_sesoi"]) and bool(
            row["holm_reject_zero"]
        )
    return {
        "effects": ordered,
        "holm": correction,
        "all_pass": all(bool(row["passes"]) for row in ordered.values()),
        "activation": False,
    }
