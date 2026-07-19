"""Preregistered constants and the 8-clause Mechanism Admission contract.

This module is the frozen, self-sealed preregistration for the Mechanism Admission Battery. It exists so
that the bar a proposed WHEN/WHAT mechanism must clear is written down once, before any test scores are
looked at, and is then referenced by digest everywhere the battery runs. Nothing here reads live data,
launches compute, or touches a sealed proof. It only declares thresholds.

The contract has eight clauses. A mechanism is admitted only if ALL eight pass. The clauses jointly encode
the lessons the STARSS23 beds paid for: an information-poor WHAT is rejected, a favorable result must clear
its EXACT compute budget against rate-matched-random, the WHEN features must decode the marginal value of
recomputation (not mere target presence) and add value over energy/rate/change heuristics, validity must
hold under the true grouped unit, the design must be adequately powered and multiplicity-corrected, the
controls must behave, and any favorable claim must reproduce across at least two gate architectures.

The preregistration dict is self-sealed with ``canonical_sha256`` over its content (the seal excluded), so a
drifted copy is detectable. As with every sealed artifact in this repository, activation, scientific
promotion, and independent scientific confirmation are hardcoded false here.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

from mop.substrate.events import canonical_sha256

MECHANISM_ADMISSION_PREREG_SCHEMA = "mop-mechanism-admission-prereg/v1"

# Preregistered scalar constants. These are the defaults the eight clauses read.
SESOI_DEFAULT = 0.05
"""Smallest effect size of interest, in the WHAT metric's own units (default F1-like scale)."""

ALPHA = 0.05
"""Per-family significance level before multiplicity correction."""

MULTIPLICITY_CORRECTION = "holm-bonferroni"
"""How the eight-clause family and any within-clause comparisons are corrected."""

MIN_INDEPENDENT_UNITS = 12
"""Minimum number of distinct grouped units (rooms/sessions/referents), not pooled frames."""

POWER_MIN = 0.8
"""Minimum preregistered statistical power at the SESOI."""

FUTILITY_STOP_RULE = "stop for futility when the conditional power at the SESOI drops below 0.2"
"""The declared stop rule; a mechanism with no stop rule fails design adequacy."""

ORACLE_HEADROOM_MIN_LIFT = 0.10
"""At the EXACT proposed budget, an oracle change-aligned policy must beat rate-matched-random by at least
this relative lift, else there is no headroom for a learned policy to be worth its compute."""

DECODABILITY_FLOOR_OVER_CHANCE = 0.05
"""A simple probe on the WHEN features must decode the marginal value of recomputation with balanced
accuracy at least this far above the 0.5 chance floor."""

INCREMENTAL_VALUE_FLOOR = 0.05
"""The WHEN features must improve marginal-value prediction over energy/rate/change heuristics by at least
this much (same units as the decodability score), else they add nothing beyond the baselines."""

NOISY_TV_TOLERANCE = 0.02
"""A noisy-TV control is at chance when its firing/reestimate rate on noise does not exceed the base rate by
more than this tolerance."""

CHANCE_BAND = 0.05
"""Shuffled-target and wrong-time controls must collapse to within this band of their chance level."""

PRIMARY_CONTROL = "rate_matched_random"
"""The required primary control. Any other primary control fails control behavior."""

MIN_GATE_ARCHITECTURES = 2
"""A favorable result must reproduce across at least this many gate architectures before any claim."""


def _clause_contract() -> list[dict[str, Any]]:
    """The eight-clause contract, in admission order. Each clause is a frozen descriptor."""

    return [
        {
            "id": "what_absolute_sufficiency",
            "statement": (
                "The WHAT estimator error must beat the constant, empirical-prior, frozen-random, and a "
                "handcrafted control by the SESOI; an information-poor WHAT is rejected."
            ),
            "thresholds": {
                "sesoi": SESOI_DEFAULT,
                "baselines": [
                    "constant",
                    "empirical_prior",
                    "frozen_random",
                    "handcrafted_control",
                ],
            },
        },
        {
            "id": "oracle_budget_headroom",
            "statement": (
                "At the EXACT proposed compute budget, an oracle change-aligned policy must beat "
                "rate-matched-random by at least the preregistered lift, else there is no headroom."
            ),
            "thresholds": {"min_relative_lift": ORACLE_HEADROOM_MIN_LIFT, "primary_control": PRIMARY_CONTROL},
        },
        {
            "id": "when_decodability",
            "statement": (
                "A simple probe must decode the MARGINAL VALUE of recomputation (not mere target presence) "
                "from the WHEN features above the chance floor."
            ),
            "thresholds": {"decodability_floor_over_chance": DECODABILITY_FLOOR_OVER_CHANCE},
        },
        {
            "id": "incremental_value",
            "statement": (
                "The WHEN features must add value beyond simple energy/rate/change heuristics; features that "
                "merely restate an energy heuristic add nothing."
            ),
            "thresholds": {"incremental_value_floor": INCREMENTAL_VALUE_FLOOR},
        },
        {
            "id": "group_disjoint_validity",
            "statement": (
                "Validity must hold under the true room/session/referent grouped split, not pooled frames; "
                "an effect carried only by frame pooling is rejected."
            ),
            "thresholds": {"sesoi": SESOI_DEFAULT, "min_independent_units": MIN_INDEPENDENT_UNITS},
        },
        {
            "id": "design_adequacy",
            "statement": (
                "SESOI, power, multiplicity correction, and a stop rule must all be declared and adequate, "
                "and the grouped unit count must meet the minimum."
            ),
            "thresholds": {
                "sesoi": SESOI_DEFAULT,
                "power_min": POWER_MIN,
                "multiplicity_correction": MULTIPLICITY_CORRECTION,
                "stop_rule": FUTILITY_STOP_RULE,
                "min_independent_units": MIN_INDEPENDENT_UNITS,
            },
        },
        {
            "id": "control_behavior",
            "statement": (
                "Noisy-TV must be rejected (at chance on noise), shuffled-target and wrong-time must "
                "collapse to chance, and rate-matched-random must be the primary control."
            ),
            "thresholds": {
                "noisy_tv_tolerance": NOISY_TV_TOLERANCE,
                "chance_band": CHANCE_BAND,
                "primary_control": PRIMARY_CONTROL,
            },
        },
        {
            "id": "architecture_independence",
            "statement": (
                "A favorable result must reproduce across at least two gate architectures before any "
                "favorable claim is made."
            ),
            "thresholds": {"min_gate_architectures": MIN_GATE_ARCHITECTURES},
        },
    ]


# The canonical order of clause ids; battery and audit adapters both key on this.
CLAUSE_IDS: tuple[str, ...] = tuple(clause["id"] for clause in _clause_contract())


def build_prereg() -> dict[str, Any]:
    """Return the frozen, self-sealed preregistration dict. Deterministic; no clock or live reads."""

    content: dict[str, Any] = {
        "schema": MECHANISM_ADMISSION_PREREG_SCHEMA,
        "claim_scope": "preregistered admission thresholds only; no capability or natural-data claim",
        "constants": {
            "sesoi_default": SESOI_DEFAULT,
            "alpha": ALPHA,
            "multiplicity_correction": MULTIPLICITY_CORRECTION,
            "min_independent_units": MIN_INDEPENDENT_UNITS,
            "power_min": POWER_MIN,
            "futility_stop_rule": FUTILITY_STOP_RULE,
            "oracle_headroom_min_lift": ORACLE_HEADROOM_MIN_LIFT,
            "decodability_floor_over_chance": DECODABILITY_FLOOR_OVER_CHANCE,
            "incremental_value_floor": INCREMENTAL_VALUE_FLOOR,
            "noisy_tv_tolerance": NOISY_TV_TOLERANCE,
            "chance_band": CHANCE_BAND,
            "primary_control": PRIMARY_CONTROL,
            "min_gate_architectures": MIN_GATE_ARCHITECTURES,
        },
        "clauses": _clause_contract(),
        "admission_rule": "admitted only if all eight clauses pass",
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**content, "seal": {"sha256": canonical_sha256(content)}}


def verify_prereg_seal(prereg: dict[str, Any]) -> bool:
    """Return True when the prereg's seal matches a recomputation over its content (seal excluded)."""

    seal = prereg.get("seal") or {}
    content = {key: value for key, value in prereg.items() if key != "seal"}
    return isinstance(seal, dict) and seal.get("sha256") == canonical_sha256(content)


# The module-level frozen preregistration. Import this as the default contract.
MECHANISM_ADMISSION_PREREG: dict[str, Any] = build_prereg()


__all__ = [
    "MECHANISM_ADMISSION_PREREG",
    "MECHANISM_ADMISSION_PREREG_SCHEMA",
    "CLAUSE_IDS",
    "SESOI_DEFAULT",
    "ALPHA",
    "MULTIPLICITY_CORRECTION",
    "MIN_INDEPENDENT_UNITS",
    "POWER_MIN",
    "FUTILITY_STOP_RULE",
    "ORACLE_HEADROOM_MIN_LIFT",
    "DECODABILITY_FLOOR_OVER_CHANCE",
    "INCREMENTAL_VALUE_FLOOR",
    "NOISY_TV_TOLERANCE",
    "CHANCE_BAND",
    "PRIMARY_CONTROL",
    "MIN_GATE_ARCHITECTURES",
    "build_prereg",
    "verify_prereg_seal",
]
