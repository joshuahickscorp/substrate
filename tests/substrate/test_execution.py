"""The frozen long run: units, freeze, rehearsal and the claim ceiling.

"""

from __future__ import annotations

import json
import shutil

import pytest

from substrate import evidence as io
from substrate import execution as L
from substrate import safety as SF
from substrate import verification as V


def test_every_unit_is_certified_or_necessary():
    """A unit in the run names what licensed it. Nothing is there because it was already written."""
    assert len(L.UNIT_LIST) >= 15
    for unit in L.UNIT_LIST:
        assert unit.certified, f"{unit.identity} does not say what licensed it"
        assert unit.module, f"{unit.identity} has nothing to run"
    ids = [u.identity for u in L.UNIT_LIST]
    assert len(set(ids)) == len(ids)
    # declared in dependency order, so a resume never waits on something later in the list
    seen = set()
    for unit in L.UNIT_LIST:
        assert set(unit.depends_on) <= seen, unit.identity
        seen.add(unit.identity)


def test_the_frozen_manifest_covers_everything_the_authority_lists():
    man = L.manifest()
    for key in (
        "source_commit",
        "source_digest",
        "source_tree",
        "sessions",
        "splits",
        "perspectives",
        "bodies",
        "seeds",
        "controls",
        "sesoi",
        "stop_rules",
        "checkpoint_policy",
        "retries",
        "claim_ceiling",
    ):
        assert man.get(key) not in (None, ""), key
    assert man["manifest_sha256"]
    assert man["sesoi"] == L.SESOI
    assert man["run_classification"] == "terminal deterministic synthesis"
    assert man["scientific_work_unit_count"] == 0
    assert "zero new scientific trials" in man["completion"]


def test_terminal_source_digest_matches_the_immutable_terminal_tree():
    assert L.source_digest() == "cc7cf719ae5fc6de2a235e3bef052438ed341e48037693c3da70c530e2971aa4"


def test_a_live_edit_after_the_freeze_is_detectable():
    man = L.manifest()
    assert L.live_edit_detected(man)["live_edit"] is False
    # advancing the commit is not a live edit, because committing the authority does exactly that
    assert L.live_edit_detected({**man, "source_commit": "0" * 40})["live_edit"] is False
    drifted = L.live_edit_detected({**man, "source_digest": "0" * 64})
    assert drifted["live_edit"] is True
    assert "source_digest" in drifted["drifted_keys"]


def test_a_unit_cannot_be_claimed_twice():
    L.release("test_probe")
    try:
        assert L.claim("test_probe") is True
        assert L.claim("test_probe") is False
    finally:
        L.release("test_probe")


def test_completion_is_units_not_wall_time():
    plan = L.resource_plan()
    assert plan["completion_criterion"] == "all units terminal"
    assert "independent of elapsed time" in plan["not_a_wall_clock"]
    assert plan["unit_count"] == len(L.UNIT_LIST)
    assert plan["scientific_work_units"] == 0


def test_the_rehearsal_breaks_things_and_survives():
    doc = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_REHEARSAL.json").read_text())
    assert doc["all_pass"] is True, doc["failed"]
    for required in (
        "injected_failure_preserves_completed_work",
        "stop_switch_halts",
        "worker_claim_race_and_exclusive_writers",
        "checkpoint_resume",
        "stale_artifact_refusal",
        "terminal_closure",
    ):
        assert doc["checks"][required]["ok"] is True, required
    assert "only proves the happy path" in doc["note"]


def test_the_claim_ceiling_forbids_what_it_must():
    doc = L.claim_boundary()
    assert set(doc["forbidden"]) == set(SF.FORBIDDEN_CLAIM_TERMS)
    assert "No result from this run can license either" in doc["requires_separate_authority"]
    assert doc["current_claims_supported"] == []
    assert "No category has a positive" in doc["current_evidence"]
    # the permitted terms are permitted only with a classification behind them
    assert "classification" in doc["permitted_only_when"]


def test_the_launch_gate_refuses_without_a_green_admission(tmp_path, monkeypatch):
    sealed = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_AUTHORITY.json").read_text())
    assert sealed["admission"] in ("green", "refused")
    assert set(sealed["commands"]) == {"launch", "status", "stop", "resume"}
    if sealed["admission"] == "green":
        assert sealed["refusal_reason"] == ""
        assert sealed["audit"]["all_pass"] is True
        assert sealed["certification"]["green"] is True
        assert sealed["rehearsal"]["all_pass"] is True


def _capsule() -> dict:
    return json.loads(L.LAUNCH_CAPSULE.read_text())


@pytest.mark.parametrize(
    ("binding", "check"),
    (
        ("source_digest", "source_digest"),
        ("source_tree_sha256", "source_tree"),
        ("configuration_sha256", "configuration"),
        ("runtime", "runtime"),
        ("historical_authority_sha256", "historical_authority"),
        ("historical_objects_sha256", "historical_objects"),
        ("data_custody_sha256", "data_custody"),
        ("session_authority_sha256", "session_authority"),
        ("perspective_system_sha256", "perspective_system"),
        ("body_artifacts_sha256", "body_artifacts"),
        ("dag_sha256", "dag"),
        ("registry_sha256", "registry"),
        ("verifier_source_sha256", "verifier"),
        ("mutations_sha256", "mutations"),
        ("claim_boundary_sha256", "claim_boundary"),
        ("expected_artifacts_sha256", "artifacts"),
        ("expected_unit_receipt_sha256", "receipt_set"),
        ("expected_reports_sha256", "reports"),
    ),
)
def test_every_capsule_identity_mismatch_fails_closed(binding, check):
    capsule = _capsule()
    capsule["bindings"][binding] = "mismatch"
    capsule["capsule_sha256"] = io.sha_obj({key: value for key, value in capsule.items() if key != "capsule_sha256"})
    result = L.validate_launch_capsule(capsule)
    assert result["all_pass"] is False
    assert check in result["failed"]


def test_missing_and_tampered_cached_artifacts_fail_closed(tmp_path, monkeypatch):
    proof = tmp_path / "evidence"
    shutil.copytree(io.PROOF, proof)
    monkeypatch.setattr(io, "PROOF", proof)
    victim = proof / "SUBSTRATE_STRUCTURAL_AUDIT.json"
    original = victim.read_bytes()
    victim.unlink()
    with pytest.raises((L.Refused, FileNotFoundError)):
        L.validate_launch_capsule(_capsule())
    victim.write_bytes(original + b"tamper")
    result = L.validate_launch_capsule(_capsule())
    assert result["all_pass"] is False
    assert "artifacts" in result["failed"]


def _receipt_sandbox(tmp_path, monkeypatch):
    synthesis = tmp_path / "terminal_synthesis"
    monkeypatch.setattr(L, "SYNTHESIS_ROOT", synthesis)
    monkeypatch.setattr(L, "UNITS", synthesis / "units")
    monkeypatch.setattr(L, "LOCKS", synthesis / "locks")
    monkeypatch.setattr(L, "STAGING", synthesis / "staging")
    identity = L._receipt_identity()
    return [L._logical_receipt(unit, identity=identity) for unit in L.UNIT_LIST]


def test_partial_receipt_set_never_becomes_authoritative(tmp_path, monkeypatch):
    receipts = _receipt_sandbox(tmp_path, monkeypatch)
    with pytest.raises(L.Refused, match="partial"):
        L.publish_receipts(receipts[:-1])
    assert not L.UNITS.exists()


def test_terminal_publication_failure_rolls_back_the_prior_complete_set(tmp_path, monkeypatch):
    receipts = _receipt_sandbox(tmp_path, monkeypatch)
    L.publish_receipts(receipts)
    before = {path.name: path.read_bytes() for path in L.UNITS.glob("*.json")}
    with pytest.raises(OSError, match="injected"):
        L.publish_receipts(receipts, inject_failure="after_old_swap")
    after = {path.name: path.read_bytes() for path in L.UNITS.glob("*.json")}
    assert after == before
    assert len(after) == 19


def test_a_death_between_receipt_swaps_is_recovered(tmp_path, monkeypatch):
    _receipt_sandbox(tmp_path, monkeypatch)
    transaction = L.STAGING / "receipt-transaction-death"
    old = transaction / "old-units"
    old.mkdir(parents=True)
    (old / "sentinel.json").write_text('{"ok": true}')
    recovered = L.recover_receipt_transaction()
    assert recovered["recovered"] == ["receipt-transaction-death"]
    assert (L.UNITS / "sentinel.json").is_file()


def test_a_mutation_pool_failure_propagates_instead_of_becoming_a_pass(monkeypatch):
    def broken(*args):
        raise RuntimeError("injected pool failure")

    monkeypatch.setattr(V, "run_mutation", broken)
    with pytest.raises(RuntimeError, match="injected pool failure"):
        V.mutation_report(only=[V.MUTATIONS[0][0]], workers=2)


def test_stop_refuses_capsule_work_and_resume_clears_the_switch(tmp_path, monkeypatch):
    stop = tmp_path / "stop"
    monkeypatch.setattr(L, "STOP", stop)
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("operator stop\n")
    with pytest.raises(L.Refused, match="stop switch"):
        L.run_capsule()
    stop.unlink()
    assert not stop.exists()


def test_fast_and_full_receipts_have_exact_normalized_parity():
    identity = L._receipt_identity()
    fast = {unit.identity: L._logical_receipt(unit, identity=identity)["receipt_sha256"] for unit in L.UNIT_LIST}
    full = {unit.identity: L._logical_receipt(unit, wall_seconds=index + 0.125, identity=identity)["receipt_sha256"] for index, unit in enumerate(L.UNIT_LIST)}
    assert fast == full == _capsule()["bindings"]["expected_unit_receipt_sha256"]
