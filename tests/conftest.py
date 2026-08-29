"""Central test-tier policy for the frozen Substrate checkout.

Tier labels live here so the suite remains discoverable without adding imports
or decorators to every historical test module. The full collection still runs
unchanged; markers only provide honest selectors for the Makefile entry points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

QUALIFICATION = {
    "test_admission",
    "test_batteries",
    "test_body",
    "test_entity_integrity",
    "test_epistemology",
    "test_memory",
    "test_metacog",
    "test_nous",
    "test_ontology",
    "test_perspectives",
    "test_plasticity",
    "test_program",
    "test_repository_layout",
    "test_runtime",
    "test_safety",
    "test_selfmodel",
    "test_verification",
    "test_world",
    "test_workspace",
    "test_worldbed",
}

INTEGRATION = {
    "test_odyssey7d_telegram_notifier",
    "test_odyssey_model_canary",
    "test_odyssey_telegram_probe",
    "test_odyssey_tools",
    "test_r2_continuity_verifier",
    "test_r2_provenance_verifier",
    "test_sandbox_r2",
    "test_source_adapters",
}

EXPENSIVE = {
    "test_experiments",
    "test_final_revision",
    "test_final_revision_field",
    "test_genesis2_core",
    "test_genesis_canaries",
    "test_genesis_challenge",
    "test_genesis_controls",
    "test_genesis_core",
    "test_genesis_k_advanced",
    "test_genesis_k_structural",
    "test_genesis_parity",
    "test_odyssey7d",
    "test_odyssey_arms",
    "test_odyssey_authority",
    "test_odyssey_density",
    "test_odyssey_detachment",
    "test_odyssey_g06_dc",
    "test_odyssey_machine_subjects",
    "test_odyssey_manifest_materializer",
    "test_odyssey_mutations",
    "test_odyssey_rehearsal",
    "test_odyssey_task_bank",
    "test_odyssey_worker",
    "test_spatial3d",
    "test_tangible_next",
    "test_v2_executor",
    "test_v2_fabric",
    "test_v2_principal",
    "test_v2_state",
    "test_v2_verification",
    "test_v3_campaign",
    "test_v3_mechanisms",
    "test_v4_mechanisms",
    "test_v5_analysis",
    "test_v5_authorities",
    "test_v5_campaign",
    "test_v5_canary",
    "test_v5_experiment",
    "test_v5_kernels",
    "test_v5_pilot",
    "test_v5_sensorium",
    "test_v5_state",
    "test_v5_stats",
}

CERTIFICATION = {
    "test_certification",
    "test_final_program",
    "test_final_revision_terminal_gate",
    "test_final_revision_terminal_harden",
    "test_genesis_k_basic",
    "test_genesis_reference",
    "test_no_private_keys_tracked",
    "test_odyssey_clean_clone",
    "test_portability",
    "test_product_assimilation_hardening",
    "test_product_pack_cache",
    "test_product_scaffold",
    "test_product_tool_bundles",
    "test_v5_cli",
    "test_v5_principal",
    "test_v5_verify",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach one execution tier while preserving qualification in normal."""
    for item in items:
        module = Path(str(item.fspath)).stem
        if module in INTEGRATION:
            item.add_marker("integration")
        elif module in EXPENSIVE:
            item.add_marker("expensive")
        elif module in CERTIFICATION:
            item.add_marker("certification")
        else:
            item.add_marker("normal")
        if module in QUALIFICATION:
            item.add_marker("qualification")
