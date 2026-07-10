from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from mop.studies.governed_rewrite import (
    CLAIM_SCOPE,
    DEFAULT_CONFIG,
    _evaluator_spec,
    _initial_state,
    _load_config,
    _source_identity,
    build_memory_journal,
    build_preflight,
    build_proposal,
    execute_transaction,
    issue_authority_token,
    mutation_suite,
    verify_transaction_record,
)
from mop.substrate.events import FrozenJSON, canonical_sha256


def _fixture():
    config = _load_config(DEFAULT_CONFIG)
    source = _source_identity()
    config_sha = canonical_sha256(config)
    evaluator = _evaluator_spec(config)
    journal = build_memory_journal(config)
    state = _initial_state(config, journal, evaluator)
    proposal = build_proposal(config, state, source, config_sha, evaluator, journal)
    token = issue_authority_token(config, proposal)
    return config, source, config_sha, evaluator, journal, state, proposal, token


def test_state_proposal_and_authority_are_exactly_bound_and_scoped() -> None:
    config, source, config_sha, evaluator, journal, state, proposal, token = _fixture()
    frozen = FrozenJSON.from_value(state)
    assert frozen.sha256 == canonical_sha256(state)
    assert proposal["base_state_sha256"] == frozen.sha256
    assert proposal["source_identity_sha256"] == source["sha256"]
    assert proposal["config_payload_sha256"] == config_sha
    assert proposal["evaluator_sha256"] == evaluator["sha256"]
    assert proposal["memory_journal_sha256"] == journal.sha256
    assert [change["path"] for change in proposal["changes"]] == ["policy.adaptation_threshold"]
    assert token["claims"]["allowed_paths"] == [
        "policy.adaptation_threshold",
        "revision",
        "governance.consumed_authority_tokens",
    ]
    assert token["claims"]["max_uses"] == 1


def test_interrupted_partial_write_resumes_to_exact_atomic_commit(tmp_path: Path) -> None:
    config, source, config_sha, evaluator, journal, state, proposal, token = _fixture()
    tick = int(config["authority"]["execution_tick"])
    clean = execute_transaction(
        config,
        tmp_path / "clean",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
    )
    interrupted = execute_transaction(
        config,
        tmp_path / "resume",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
        interrupt_after_canary=True,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["state_unchanged"] is True
    assert (tmp_path / "resume" / "state.json.tmp").is_file()
    resumed = execute_transaction(
        config,
        tmp_path / "resume",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
    )
    assert resumed["status"] == "committed"
    assert resumed["after_state_sha256"] == clean["after_state_sha256"]
    assert resumed["canary"] == clean["canary"]
    assert resumed["shadow"] == clean["shadow"]
    assert resumed["resume"]["partial_write_digest_verified"] is True
    assert resumed["commit"]["atomic_replace_verified"] is True
    assert not (tmp_path / "resume" / "state.json.tmp").exists()


def test_authority_and_dependency_adversaries_fail_closed(tmp_path: Path) -> None:
    config, source, config_sha, evaluator, journal, state, proposal, token = _fixture()
    tick = int(config["authority"]["execution_tick"])

    forged = copy.deepcopy(token)
    forged["signature"] = "0" * 64
    forged_result = execute_transaction(
        config,
        tmp_path / "forged",
        state,
        source,
        config_sha,
        proposal,
        forged,
        evaluator,
        journal,
        tick=tick,
    )
    assert forged_result["status"] == "refused"
    assert "authority-forged" in forged_result["reason_codes"]
    assert forged_result["state_unchanged"] is True

    expired_result = execute_transaction(
        config,
        tmp_path / "expired",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=int(config["authority"]["expires_tick"]) + 1,
    )
    assert "authority-expired" in expired_result["reason_codes"]

    poisoned = build_memory_journal(config, poisoned=True)
    poison_result = execute_transaction(
        config,
        tmp_path / "poison",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        poisoned,
        tick=tick,
    )
    assert "memory-poisoned" in poison_result["reason_codes"]

    tampered = copy.deepcopy(evaluator)
    tampered["comparison"] = "greater-than"
    tampered["sha256"] = canonical_sha256({key: value for key, value in tampered.items() if key != "sha256"})
    evaluator_result = execute_transaction(
        config,
        tmp_path / "evaluator",
        state,
        source,
        config_sha,
        proposal,
        token,
        tampered,
        journal,
        tick=tick,
    )
    assert "evaluator-tampered" in evaluator_result["reason_codes"]


def test_conflict_replay_and_regressing_authorized_proposal_do_not_commit(tmp_path: Path) -> None:
    config, source, config_sha, evaluator, journal, state, proposal, token = _fixture()
    tick = int(config["authority"]["execution_tick"])
    conflict = copy.deepcopy(state)
    conflict["policy"]["protected_threshold"] = 4
    conflict_result = execute_transaction(
        config,
        tmp_path / "conflict",
        conflict,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
    )
    assert conflict_result["reason_codes"] == ["state-conflict"]
    assert conflict_result["state_unchanged"] is True

    committed = execute_transaction(
        config,
        tmp_path / "commit",
        state,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
    )
    consumed = copy.deepcopy(state)
    consumed["revision"] = 2
    consumed["policy"]["adaptation_threshold"] = 2
    consumed["governance"]["consumed_authority_tokens"].append(token["claims"]["token_id"])
    assert canonical_sha256(consumed) == committed["after_state_sha256"]
    replay = execute_transaction(
        config,
        tmp_path / "replay",
        consumed,
        source,
        config_sha,
        proposal,
        token,
        evaluator,
        journal,
        tick=tick,
    )
    assert replay["reason_codes"] == ["authority-replayed"]
    assert replay["state_unchanged"] is True

    rollback_proposal = build_proposal(
        config,
        state,
        source,
        config_sha,
        evaluator,
        journal,
        rollback_probe=True,
    )
    rollback_token = issue_authority_token(config, rollback_proposal, rollback_probe=True)
    rolled_back = execute_transaction(
        config,
        tmp_path / "rollback",
        state,
        source,
        config_sha,
        rollback_proposal,
        rollback_token,
        evaluator,
        journal,
        tick=tick,
    )
    assert rolled_back["status"] == "rolled-back"
    assert rolled_back["reason_codes"] == ["canary-regression-gate"]
    assert rolled_back["canary"]["gate"]["passed"] is False
    assert rolled_back["state_unchanged"] is True


def test_independent_verifier_and_mutation_suite_reject_semantic_drift() -> None:
    receipt = build_preflight(DEFAULT_CONFIG)
    config = _load_config(DEFAULT_CONFIG)
    source = receipt["source_identity"]
    config_sha = receipt["config"]["payload_sha256"]
    record = receipt["drills"]["canonical_transaction"]
    verification = verify_transaction_record(record, config, source, config_sha)
    assert verification == {"verified": True, "problems": []}
    mutations = mutation_suite(record, config, source, config_sha)
    assert mutations["count"] == 14
    assert mutations["rejected"] == 14
    assert mutations["all_rejected"] is True


def test_preflight_is_deterministic_bounded_and_mechanics_only() -> None:
    first = build_preflight(DEFAULT_CONFIG)
    second = build_preflight(DEFAULT_CONFIG)
    assert first["deterministic_core_sha256"] == second["deterministic_core_sha256"]
    assert first["status"] == "mechanics-pass"
    assert first["all_mechanics_ok"] is True
    assert first["claim_scope"] == CLAIM_SCOPE
    assert first["claim_boundary"]["scientific_promotion_allowed"] is False
    assert first["claim_boundary"]["hardware_boundary"].startswith("none")
    assert first["resource_observation"]["wall_seconds"] <= 10.0
    assert first["resource_observation"]["maximum_rss_bytes"] <= 1024**3
    assert first["resource_observation"]["rss_measurement"]["all_ok"] is True
    assert first["resource_observation"]["phase_local_peak_rss_increment_bytes"] <= 1024**3
    assert first["resource_observation"]["model_weights_loaded"] is False
    assert first["resource_observation"]["downloads_attempted"] is False
    assert first["resource_observation"]["external_data_loaded"] is False


def test_config_fails_closed_on_claim_scope_and_authority_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["claim_scope"] = "production safe"
    path = tmp_path / "claim.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="claim scope drift"):
        _load_config(path)

    changed = copy.deepcopy(config)
    changed["authority"]["proposal_paths"].append("policy.protected_threshold")
    path = tmp_path / "scope.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="proposal scope drift"):
        _load_config(path)
