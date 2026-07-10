from __future__ import annotations

from scripts.build_extended_compute_requirements import registry_rows


def test_registry_only_rows_are_preregistration_only_requirements() -> None:
    rows, registry_ids = registry_rows()
    by_id = {row["id"]: row for row in rows}

    assert len(registry_ids) == 227
    assert by_id["f21_asynchronous_temporal_binding"]["primary_category"] == 2
    assert by_id["f21_asynchronous_temporal_binding"]["measured"]["present"] is False
    assert by_id["f21_asynchronous_temporal_binding"]["classification_basis"] == (
        "registry-preregistration-only"
    )

    assert by_id["f65_specimen_to_specimen_transfer"]["primary_category"] == 6
    assert by_id["f65_specimen_to_specimen_transfer"]["required_rung"] == "L6"
    assert by_id["f66_cross_substrate_form_portability"]["primary_category"] == 2
    assert by_id["f66_cross_substrate_form_portability"]["post_blocker_local_rung"] == "L0"

    category2_current = sum(
        row["scope"] == "current_registry" and row["primary_category"] == 2 for row in rows
    )
    assert category2_current == 39
