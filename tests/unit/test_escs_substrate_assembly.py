from __future__ import annotations

import copy
import json

import pytest

from mop.config import REPO_ROOT
from mop.escs.perspective_registry import (
    EvidenceStanding,
    PerspectiveFacet,
    load_perspective_candidate_registry,
)
from mop.escs.substrate_assembly import (
    SlotMode,
    SubstrateAssembly,
    load_substrate_assembly,
)

REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"
ASSEMBLY_PATH = REPO_ROOT / "configs/experiment/escs_substrate_assembly.json"


def test_all_requested_perspectives_are_installed_but_quiescent() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    assembly = load_substrate_assembly(ASSEMBLY_PATH)

    assert len(assembly.slots) == len(PerspectiveFacet) == 31
    assert assembly.default_quiescent is True
    assert assembly.scientific_promotion_allowed is False
    assert all(slot.activation_enabled is False for slot in assembly.slots)
    assert all(slot.scientific_promotion_allowed is False for slot in assembly.slots)
    assert assembly.validate_registry(registry) == ()
    assert assembly == SubstrateAssembly.create(registry)


def test_evidence_standing_routes_slots_without_relabeling_nulls_or_failures() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    assembly = load_substrate_assembly(ASSEMBLY_PATH)
    slots = {slot.candidate_id: slot for slot in assembly.slots}

    for candidate in registry.candidates:
        slot = slots[candidate.candidate_id]
        if candidate.evidence_standing is EvidenceStanding.NULL:
            assert slot.mode is SlotMode.CONTROL_INERT
        if candidate.evidence_standing is EvidenceStanding.FAILED:
            assert slot.mode is SlotMode.EXCLUDED
        if candidate.evidence_standing in {
            EvidenceStanding.PENDING,
            EvidenceStanding.BLOCKED,
            EvidenceStanding.UNTESTED,
        }:
            assert slot.mode is SlotMode.SANDBOX_STUB_INERT


def test_curiosity_novelty_uncertainty_and_simulation_keep_effect_boundaries() -> None:
    assembly = load_substrate_assembly(ASSEMBLY_PATH)
    slots = {slot.candidate_id: slot for slot in assembly.slots}

    for candidate_id in ("curiosity", "novelty_detection", "uncertainty_estimation"):
        assert slots[candidate_id].mode in {SlotMode.CONTROL_INERT, SlotMode.EXCLUDED}
        assert slots[candidate_id].activation_enabled is False
    for candidate_id in ("imagination", "simulation"):
        assert slots[candidate_id].effect_boundary.value == "counterfactual-only"


def test_assembly_tampering_or_registry_splice_fails_closed() -> None:
    payload = json.loads(ASSEMBLY_PATH.read_text())
    payload["slots"][0]["activation_enabled"] = True
    with pytest.raises(ValueError, match="must be disabled"):
        SubstrateAssembly.from_payload(payload)

    payload = json.loads(ASSEMBLY_PATH.read_text())
    payload["candidate_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash mismatch"):
        SubstrateAssembly.from_payload(payload)

    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    assembly = load_substrate_assembly(ASSEMBLY_PATH)
    spliced = copy.deepcopy(registry.payload())
    spliced["candidates"][0]["label"] = "spliced"
    changed = type(registry).from_payload(spliced)
    assert "candidate-registry-authority-mismatch" in assembly.validate_registry(changed)
