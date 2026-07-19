"""The redesign route taken when a mechanism fails the admission battery.

When the current features fail the battery, the honest next step is not to retune the same energy front-end.
It is to change the representation. This module points to a relational-temporal representation redesign:
persistent source and event identity, short-horizon state carried across frames, explicit change relations
between sources, and causal or predictive features rather than instantaneous energy. It exists so that a
failing mechanism is routed toward the representation change that could actually carry the missing
information, and away from the four cheap moves that have never converted a null in these beds.

The route explicitly FORBIDS four default routes: trying another MLP, adding more seeds, weakening
thresholds, and adding another spacing regularizer. Those are re-searches of the same representation; they
do not add the relational-temporal structure the battery failures point to.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FORBIDDEN_DEFAULT_ROUTES: tuple[str, ...] = (
    "try another MLP",
    "more seeds",
    "weaker thresholds",
    "another spacing regularizer",
)

RELATIONAL_TEMPORAL_TARGETS: tuple[str, ...] = (
    "persistent source and event identity carried across frames",
    "short-horizon state (a small recurrent or windowed memory), not instantaneous energy",
    "explicit change relations between sources (appear, vanish, move, split, merge)",
    "causal or predictive features (does recomputing now reduce future error), not mere target presence",
)


def relational_temporal_redesign_route(
    battery_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the relational-temporal redesign route.

    When ``battery_result`` is supplied and not admitted, the route is engaged and lists the failed clauses
    it is responding to. When it is admitted, the route reports that no redesign is required. The four
    forbidden default routes are always declared so no caller can mistake them for the route.
    """

    engaged = True
    failed_clauses: list[str] = []
    if battery_result is not None:
        engaged = not bool(battery_result.get("admitted"))
        clauses = battery_result.get("clauses") or {}
        failed_clauses = [name for name, entry in clauses.items() if not entry.get("passed")]

    return {
        "schema": "mop-relational-temporal-redesign-route/v1",
        "engaged": engaged,
        "diagnosis": (
            "the current features failed the admission battery; the missing information is relational and "
            "temporal, so the representation must change rather than the search over the same representation"
        )
        if engaged
        else "the mechanism was admitted; no representation redesign is required",
        "responding_to_failed_clauses": failed_clauses,
        "redesign_target": "relational-temporal representation",
        "redesign_target_components": list(RELATIONAL_TEMPORAL_TARGETS),
        "forbidden_default_routes": list(FORBIDDEN_DEFAULT_ROUTES),
        "forbidden_default_routes_rationale": (
            "each re-searches the same instantaneous-energy representation; none adds persistent identity, "
            "short-horizon state, change relations, or predictive structure, so none can convert a null "
            "whose cause is a missing relational-temporal representation"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


__all__ = [
    "relational_temporal_redesign_route",
    "FORBIDDEN_DEFAULT_ROUTES",
    "RELATIONAL_TEMPORAL_TARGETS",
]
