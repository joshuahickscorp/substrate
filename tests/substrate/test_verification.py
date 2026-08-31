"""Independent recomputation, mutation attacks, and the consolidation artifacts.

The mutation suite itself is run from the CLI because each mutation costs a subprocess. What is asserted
here is the property that makes the sealed report meaningful: every mutation names a distinct guard, and
none of them survived.

"""

from __future__ import annotations

import json

from substrate import deliverables as D
from substrate import evidence as io
from substrate import program as P
from substrate import safety
from substrate import verification as V


def test_recomputation_agrees_with_every_sealed_artifact():
    report = V.recompute()
    assert report["disagreements"] == [], report["disagreements"]
    assert report["broken_seals"] == [], report["broken_seals"]
    assert report["all_pass"] is True
    assert len(report["checks"]) >= 25
    # the recomputation must not be trivially self confirming: every check names the route it took
    assert all(c["route"] for c in report["checks"].values())


def test_a_tampered_seal_is_detected():
    doc = json.loads((io.PROOF / "SUBSTRATE_WORKSPACE.json").read_text())
    assert V._seal_intact(doc) is True
    assert V._seal_intact({**doc, "regions": []}) is False


def test_recomputation_reads_each_sealed_artifact_once(monkeypatch):
    calls = []
    original = V._sealed

    def counted(name):
        calls.append(name)
        return original(name)

    monkeypatch.setattr(V, "_sealed", counted)
    report = V.recompute()
    assert report["all_pass"] is True
    assert len(calls) == len(set(calls))


def test_every_mutation_names_a_distinct_guard_and_died():
    guards = [m[3] for m in V.MUTATIONS]
    names = [m[0] for m in V.MUTATIONS]
    assert len(names) == len(set(names))
    assert len(V.MUTATIONS) >= 20
    # more than one mutation may aim at the same guard, but every guard must be a real test node
    for node in set(guards):
        path, _, _ = node.partition("::")
        assert (io.ROOT / path).is_file(), node

    report = json.loads((io.PROOF / "SUBSTRATE_MUTATION_REPORT.json").read_text())
    assert report["survivors"] == [], report["survivors"]
    assert report["all_rejected"] is True
    assert report["rejected"] == report["total"] >= 20


def test_the_capability_map_never_outruns_the_item_table():
    st = P.state()
    caps = D.capability_map(st)
    assert caps["total"] == 20
    for row in caps["capabilities"]:
        for item in row["items"]:
            assert item in st["items"], f"{row['capability']} names an item that does not exist: {item}"
        # a capability may only claim evidence through an item that carries a classification
        for item in row["evidence_earned"]:
            assert (st["items"][item]["result"] or {}).get("scientific") is True
    assert caps["with_evidence_count"] <= caps["implemented_count"]

    arch = D.architecture(st)
    assert arch["components_present"] == arch["components_declared"]


def test_the_entity_report_makes_no_forbidden_claim():
    st = P.state()
    arch, caps = D.architecture(st), D.capability_map(st)
    temporal = D.temporal_core_record()
    spec = D.entity_spec(st, arch, caps)
    text = D.entity_report(st, spec, arch, caps, temporal)
    assert safety.check_claim(text) == []
    # the four separations the report is required to make
    assert "implementation" in text and "evidence" in text
    assert "None of that is a claim about experience" in text
    assert "next frontier" in text.lower()
    assert spec["claim_class"]["unsupported_claim"] == []
    assert set(spec["not_claimed"]) == set(safety.FORBIDDEN_CLAIM_TERMS)


def test_the_temporal_core_record_tracks_the_live_program():
    record = D.temporal_core_record()
    counts = record["receipt_counts"]
    assert counts["e2_principal"] <= record["principal_shards_expected"]
    assert record["principal_complete"] == (counts["e2_principal"] >= 24)
    # v1 cannot be named while the factorial is unfinished or its verification does not pass
    if not record["terminal"]:
        assert record["named_v1"] is False
        assert "not named yet" in record["honest_state"]
    assert record["remains_one_component"] is True


def test_no_sealed_substrate_artifact_is_undeclared():
    """The no placeholder rule, checked against what is actually on disk."""
    # only this program's own proof root. An artifact in an inherited root is that program's to write.
    declared = {e for item in P.ITEMS for e in item.evidence if ":" not in e}
    declared |= {
        "SUBSTRATE_STATE.json",
        "SUBSTRATE_LEDGER.md",
        "SUBSTRATE_HYPOTHESIS_GRAPH.json",
        "SUBSTRATE_NULL_MAP.json",
        "SUBSTRATE_NEXT_FRONTIER.json",
        "SUBSTRATE_FINAL_LEDGER.md",
    }
    # SUBSTRATE_CLEAN_CLONE.json is produced by cloning the commit that would have to contain it, so it
    # can never be present in that commit. The bootstrap exception is declared rather than silently
    # tolerated, and the artifact is still required to be declared by an item.
    bootstrap = {"SUBSTRATE_CLEAN_CLONE.json"}
    on_disk = {p.name for p in io.PROOF.glob("SUBSTRATE_*")}
    assert on_disk - declared == set(), "an undeclared artifact was sealed into the proof root"
    assert declared - on_disk - bootstrap == set(), "a declared deliverable was never written"
