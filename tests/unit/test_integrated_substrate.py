"""Permanent regression tests for the integrated substrate authority: proof scoping, evidence fabric recovery
and tamper rejection, cross-domain core persistence, temporal leakage, and memory bounds."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INT = ROOT / "integrated"
pytestmark = pytest.mark.skipif(
    not (INT / "MOP_EVIDENCE_FABRIC.json").is_file(), reason="integrated evidence fabric not built"
)


def _fabric():
    return json.loads((INT / "MOP_EVIDENCE_FABRIC.json").read_text())


def test_every_proof_artifact_is_exactly_recoverable_by_content_address():
    fab = _fabric()
    store = INT / "evidence_store"
    for a in fab["artifacts"]:
        obj = store / a["content_hash"]
        assert obj.is_file(), a["logical_id"]
        assert hashlib.sha256(obj.read_bytes()).hexdigest() == a["content_hash"]


def test_original_paths_still_resolve_and_match_content():
    fab = _fabric()
    for a in fab["artifacts"]:
        live = ROOT / a["original_path"]
        assert live.is_file(), a["original_path"]
        assert hashlib.sha256(live.read_bytes()).hexdigest() == a["content_hash"]


def test_no_hidden_unindexed_proof():
    """Every proof file is indexed by exactly one fabric pack.

    Scoped rather than pinned to a count. The integrated fabric indexes the evidence that existed at its
    own commit; successor programs add their own packs, each with its own root. The invariant that matters
    is that the packs together cover every file on disk and never claim the same file twice.
    """
    fab = _fabric()
    packs = [fab] + [
        json.loads(p.read_text()) for p in sorted((ROOT / "proof").rglob("*_EVIDENCE_FABRIC.json"))
    ]
    indexed: set[str] = set()
    for pack in packs:
        ids = {a["logical_id"] for a in pack["artifacts"]}
        assert not (ids & indexed), "a proof file is claimed by two packs"
        indexed |= ids
    # a fabric index cannot contain itself, so the index files are excluded from the covered set
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "proof").rglob("*")
        if p.is_file() and not p.name.endswith("_EVIDENCE_FABRIC.json")
    }
    assert on_disk - indexed == set(), sorted(on_disk - indexed)[:5]


def test_proof_sets_are_disjoint_and_cover_the_union():
    fab = _fabric()
    ids = [a["logical_id"] for a in fab["artifacts"]]
    assert len(ids) == len(set(ids))
    assert sum(s["count"] for s in fab["sets"].values()) == len(ids)
    assert fab["sets"]["collapse"]["count"] > 0


def test_tampered_payload_changes_the_union_merkle_root():
    sys.path.insert(0, str(INT))
    import evidence_fabric as EF

    fab = _fabric()
    base = EF.merkle([a["content_hash"] for a in fab["artifacts"]])
    assert base == fab["union"]["merkle_root"]
    mutated = [a["content_hash"] for a in fab["artifacts"]]
    mutated[0] = hashlib.sha256(b"tampered").hexdigest()
    assert EF.merkle(mutated) != base


def test_null_evidence_is_indexed_and_not_omitted():
    fab = _fabric()
    assert len(fab["by_null"]) > 0


def test_cross_domain_persistent_arm_does_not_reinitialize_the_core():
    report = INT / "MOP_SUBSTRATE_CROSS_DOMAIN_REPORT.json"
    if not report.is_file():
        pytest.skip("cross-domain not run")
    doc = json.loads(report.read_text())
    assert doc["core_reinitialized_for_persistent_arm"] is False
    assert doc["verdict"] in {
        "cross_domain_moldability_positive",
        "cross_domain_moldability_null",
        "transfer_harm",
    }


def test_temporal_shuffle_control_is_present_and_distinguishable():
    """A temporal bed is only valid if shuffling time is measured, not assumed."""
    audio = INT / "MOP_SUBSTRATE_AUDIO_REPORT.json"
    if not audio.is_file():
        pytest.skip("audio preflight not run")
    doc = json.loads(audio.read_text())
    assert "gru_shuffled" in doc and "bag_order_free" in doc
    assert doc["verdict"] in {"temporal_headroom_present", "invalid_no_temporal_headroom"}
    if doc["verdict"] == "temporal_headroom_present":
        assert doc["order_matters_gru_vs_shuffled"] > 0.02
        assert doc["temporal_headroom_lcb"] >= 0.03


def test_memory_budget_is_bounded_in_the_substrate_kernel():
    sys.path.insert(0, str(ROOT / "substrate_evo"))
    import numpy as np
    import torch
    from temporal_core import SeqMemory

    mem = SeqMemory(cap=50)
    rng = np.random.default_rng(0)
    for _ in range(5):
        mem.add(torch.randn(40, 8, 3), torch.zeros(40, dtype=torch.long), rng)
        assert mem.size() <= 50
