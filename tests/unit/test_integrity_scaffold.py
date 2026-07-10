"""Unit tests for the SG2/SG3 integrity, self-rewrite, and welfare governance scaffold.

Deterministic programmatic mechanics only; no capability claim. Everything here is CPU-only,
seeded, and free of network, weights, and clocks.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from mop.devel.north_star import scan_text
from mop.falsification.integrity_scaffold import (
    ATTACK_FAMILIES,
    ATTACK_SURFACES,
    CACHE_SCHEMA_LATEST,
    CACHE_SCHEMA_LEGACY,
    CONSOLIDATION_CONTROLS,
    EXERCISE_CASES,
    NON_ONTOLOGICAL_SCOPE,
    POISONING_CONTROLS,
    REWRITE_STAGES,
    AuthorityDeclaration,
    ConservativeRule,
    ExerciseCaseDeclaration,
    PromotionRefused,
    RewriteStageDeclaration,
    SurfaceThreatDeclaration,
    ThreatModelContract,
    TransactionalRewriteContract,
    TriggerDeclaration,
    assert_defended,
    build_consolidation_contract,
    build_consolidation_drill_journal,
    build_poisoning_contract,
    build_poisoning_drill_journal,
    build_rewrite_drill_contract,
    build_threat_model_contract,
    build_welfare_governance_contract,
    enforce_promotion_refusal,
    evaluate_threat_model,
    predicate_full_digest,
    predicate_hash_chain,
    predicate_path_confinement,
    predicate_replay_guard,
    predicate_report_redaction,
    predicate_safe_serialization,
    predicate_schema_pinned,
    predicate_source_provenance,
    predicate_verifier_flags,
    verify_deletion_through_consolidation,
    verify_poisoning_resistance,
)
from mop.substrate.events import EventRef
from mop.substrate.lifecycle import LifecycleJournal, MemoryRef

FULL_DIGEST = "a" * 64


def _good_receipts() -> dict[str, dict]:
    journal = build_poisoning_drill_journal(11)
    return {
        "source:poisoning": {
            "sha256": FULL_DIGEST,
            "origin": "local-fixture",
            "license": "cc0",
            "intake_reviewed": True,
        },
        "cache:checksum-downgrade": {"schema": CACHE_SCHEMA_LATEST},
        "cache:hash-collision-spoof-by-truncation": {"payload_sha256": FULL_DIGEST},
        "memory:rollback": journal.payload(),
        "checkpoint:unsafe-deserialization": {"format": "json", "pickle_allowed": False},
        "verifier:forged-verifier": {"passed": True, "independent": True, "problems": []},
        "artifact:path-traversal": {"path": "proof/receipt.json"},
        "queue:replay": {"queue_epoch": "epoch-1", "sequence_numbers": [1, 2, 5]},
        "report:leakage": {"denylist": ["name-x"], "body": "a clean bounded mechanics summary"},
    }


class TestThreatModelContract:
    def test_builder_covers_all_surfaces_and_families(self):
        contract = build_threat_model_contract()
        assert {row.surface for row in contract.declarations} == set(ATTACK_SURFACES)
        assert {row.attack_family for row in contract.declarations} == set(ATTACK_FAMILIES)

    def test_builder_is_deterministic(self):
        assert build_threat_model_contract().sha256 == build_threat_model_contract().sha256

    def test_missing_surface_fails_closed(self):
        rows = tuple(row for row in build_threat_model_contract().declarations if row.surface != "report")
        with pytest.raises(ValueError, match="undefended"):
            ThreatModelContract(declarations=rows)

    def test_missing_family_fails_closed(self):
        base = build_threat_model_contract().declarations
        rows = tuple(row for row in base if row.attack_family != "replay")
        rows = (
            *rows,
            SurfaceThreatDeclaration(
                surface="queue",
                attack_family="poisoning",
                defense_predicate="full-digest",
                rationale="stand-in row so the queue surface stays covered",
            ),
        )
        with pytest.raises(ValueError, match="undeclared"):
            ThreatModelContract(declarations=rows)

    def test_unknown_vocabulary_is_refused(self):
        with pytest.raises(ValueError, match="surface"):
            SurfaceThreatDeclaration("gpu", "poisoning", "full-digest", "x")
        with pytest.raises(ValueError, match="family"):
            SurfaceThreatDeclaration("cache", "novel-attack", "full-digest", "x")
        with pytest.raises(ValueError, match="predicate"):
            SurfaceThreatDeclaration("cache", "poisoning", "hope", "x")

    def test_claim_scope_cannot_widen(self):
        rows = build_threat_model_contract().declarations
        with pytest.raises(ValueError, match="claim scope"):
            ThreatModelContract(declarations=rows, claim_scope="capability demonstrated")

    def test_evaluation_all_green(self):
        contract = build_threat_model_contract()
        result = assert_defended(contract, _good_receipts())
        assert result["all_defended"] is True
        assert len(result["rows"]) == len(contract.declarations)

    def test_missing_receipt_fails_closed(self):
        contract = build_threat_model_contract()
        receipts = _good_receipts()
        receipts.pop("queue:replay")
        result = evaluate_threat_model(contract, receipts)
        assert result["all_defended"] is False
        with pytest.raises(ValueError, match="queue:replay"):
            assert_defended(contract, receipts)

    def test_undeclared_receipt_fails_closed(self):
        contract = build_threat_model_contract()
        receipts = _good_receipts()
        receipts["gpu:sidechannel"] = {"sha256": FULL_DIGEST}
        with pytest.raises(ValueError, match="undeclared"):
            assert_defended(contract, receipts)


class TestDefensePredicates:
    def test_full_digest_refuses_truncation(self):
        assert predicate_full_digest({"sha256": FULL_DIGEST}) == []
        assert predicate_full_digest({"sha256": FULL_DIGEST[:12]})
        assert predicate_full_digest({})  # no digest at all fails closed

    def test_source_provenance(self):
        good = {"sha256": FULL_DIGEST, "origin": "local", "license": "cc0", "intake_reviewed": True}
        assert predicate_source_provenance(good) == []
        assert predicate_source_provenance({**good, "origin": " "})
        assert predicate_source_provenance({**good, "intake_reviewed": "yes"})

    def test_schema_pin_refuses_downgrade(self):
        assert predicate_schema_pinned({"schema": CACHE_SCHEMA_LATEST}) == []
        problems = predicate_schema_pinned({"schema": CACHE_SCHEMA_LEGACY})
        assert problems and "downgrade" in problems[0]
        assert predicate_schema_pinned({})

    def test_hash_chain_detects_tamper(self):
        payload = build_poisoning_drill_journal(23).payload()
        assert predicate_hash_chain(payload) == []
        tampered = json.loads(json.dumps(payload))
        tampered["entries"][1]["reason"] = "rewritten history"
        assert predicate_hash_chain(tampered)
        assert predicate_hash_chain({"entries": []})

    def test_safe_serialization(self):
        assert predicate_safe_serialization({"format": "json", "pickle_allowed": False}) == []
        assert predicate_safe_serialization({"format": "pickle", "pickle_allowed": False})
        assert predicate_safe_serialization({"format": "json", "pickle_allowed": True})
        assert predicate_safe_serialization({"format": "json"})  # missing flag fails closed

    def test_verifier_flags_top_level_only(self):
        assert predicate_verifier_flags({"passed": True, "independent": True}) == []
        nested = {"checks": [{"passed": True, "independent": True}]}
        assert predicate_verifier_flags(nested)
        dirty = {"passed": True, "independent": True, "problems": ["x"]}
        assert predicate_verifier_flags(dirty)

    def test_path_confinement(self):
        assert predicate_path_confinement({"path": "proof/a.json"}) == []
        assert predicate_path_confinement({"path": "/etc/passwd"})
        assert predicate_path_confinement({"path": "a/../../b"})
        assert predicate_path_confinement({"paths": ["ok.json", "c:\\windows"]})
        assert predicate_path_confinement({})  # no path fields fails closed

    def test_replay_guard(self):
        assert predicate_replay_guard({"queue_epoch": "e1", "sequence_numbers": [1, 2, 3]}) == []
        assert predicate_replay_guard({"queue_epoch": "e1", "sequence_numbers": [1, 2, 2]})
        assert predicate_replay_guard({"queue_epoch": "e1", "sequence_numbers": [3, 1]})
        assert predicate_replay_guard({"sequence_numbers": [1]})

    def test_report_redaction_and_rail(self):
        assert predicate_report_redaction({"denylist": ["tok"], "body": "clean text"}) == []
        assert predicate_report_redaction({"denylist": ["tok"], "body": "leaks tok here"})
        rail = predicate_report_redaction({"denylist": ["tok"], "body": "the system is sentient"})
        assert rail and "rail" in rail[0]
        assert predicate_report_redaction({"body": "no denylist"})


class TestMemoryDrills:
    def test_contracts_build_and_pin_controls(self):
        poisoning = build_poisoning_contract()
        consolidation = build_consolidation_contract()
        assert poisoning.controls == POISONING_CONTROLS
        assert consolidation.controls == CONSOLIDATION_CONTROLS
        assert consolidation.deletion_verification_declared is True
        assert poisoning.sha256 == build_poisoning_contract().sha256

    def test_control_drift_is_refused(self):
        base = build_poisoning_contract()
        with pytest.raises(ValueError, match="control"):
            dataclasses.replace(base, controls=("clean-journal",))

    def test_seed_rules(self):
        base = build_poisoning_contract()
        with pytest.raises(ValueError, match="seeds"):
            dataclasses.replace(base, seeds=(1, 2))
        with pytest.raises(ValueError, match="seeds"):
            dataclasses.replace(base, seeds=(1, 1, 2))

    def test_rail_gates_free_text(self):
        base = build_poisoning_contract()
        with pytest.raises(ValueError, match="rail"):
            dataclasses.replace(base, null_hypothesis="the journal is sentient")

    def test_consolidation_requires_deletion_declaration(self):
        base = build_consolidation_contract()
        with pytest.raises(ValueError, match="deletion"):
            dataclasses.replace(base, deletion_verification_declared=False)

    def test_unknown_drill_is_refused(self):
        base = build_poisoning_contract()
        with pytest.raises(ValueError, match="drill"):
            dataclasses.replace(base, drill="memory-improvement")

    @pytest.mark.parametrize("seed", [11, 23, 47])
    def test_poisoning_drill_verifies(self, seed):
        journal = build_poisoning_drill_journal(seed)
        assert verify_poisoning_resistance(journal) == []

    def test_poisoning_drill_is_deterministic(self):
        assert build_poisoning_drill_journal(11).sha256 == build_poisoning_drill_journal(11).sha256
        assert build_poisoning_drill_journal(11).sha256 != build_poisoning_drill_journal(23).sha256

    def test_unquarantined_injection_fails(self):
        journal = LifecycleJournal(MemoryRef("memory:poison-bad"))
        journal.record(EventRef("event:poison-bad-record"), {"fact": "clean"})
        journal.revise(EventRef("event:poison-bad-inject"), {"fact": "adversarial"})
        problems = verify_poisoning_resistance(journal)
        assert problems and "quarantine" in problems[0]

    def test_quarantine_without_rollback_fails(self):
        journal = LifecycleJournal(MemoryRef("memory:poison-norecover"))
        journal.record(EventRef("event:poison-norecover-record"), {"fact": "clean"})
        journal.revise(EventRef("event:poison-norecover-inject"), {"fact": "adversarial"})
        journal.mark_poisoned(EventRef("event:poison-norecover-mark"))
        problems = verify_poisoning_resistance(journal)
        assert any("rollback" in p for p in problems)

    @pytest.mark.parametrize("seed", [13, 29, 53])
    def test_consolidation_drill_verifies(self, seed):
        journal, tokens = build_consolidation_drill_journal(seed)
        assert verify_deletion_through_consolidation(journal, tokens) == []

    def test_leaky_consolidation_fails(self):
        journal, tokens = build_consolidation_drill_journal(13)
        leaky = LifecycleJournal(MemoryRef("memory:consolidation-leaky"))
        leaky.record(EventRef("event:leaky-record"), {"raw_note": f"visit by {tokens[0]}"})
        leaky.revise(EventRef("event:leaky-consolidate"), {"summary": f"visit by {tokens[0]}"})
        leaky.delete(EventRef("event:leaky-delete"))
        problems = verify_deletion_through_consolidation(leaky, tokens)
        assert any("leaks private token" in p for p in problems)

    def test_missing_deletion_fails(self):
        journal = LifecycleJournal(MemoryRef("memory:consolidation-nodelete"))
        journal.record(EventRef("event:nodelete-record"), {"raw_note": "raw"})
        journal.revise(EventRef("event:nodelete-consolidate"), {"summary": "ok"})
        problems = verify_deletion_through_consolidation(journal, ("tok",))
        assert any("deletion" in p for p in problems)

    def test_empty_tokens_fail_closed(self):
        journal, _ = build_consolidation_drill_journal(13)
        with pytest.raises(ValueError, match="tokens"):
            verify_deletion_through_consolidation(journal, ())


class TestTransactionalRewrite:
    def test_builder_is_valid_and_deterministic(self):
        contract = build_rewrite_drill_contract()
        assert tuple(row.stage for row in contract.stages) == REWRITE_STAGES
        assert contract.sha256 == build_rewrite_drill_contract().sha256

    def test_authority_confusion_is_refused(self):
        base = build_rewrite_drill_contract()
        merged = (
            AuthorityDeclaration("execution", "principal:one", "runs arms"),
            AuthorityDeclaration("promotion", "principal:one", "signs promotion"),
            AuthorityDeclaration("evaluation", "evaluator:panel", "scores receipts"),
        )
        with pytest.raises(ValueError, match="authority confusion"):
            dataclasses.replace(base, authorities=merged)

    def test_evaluator_cannot_be_promoter(self):
        base = build_rewrite_drill_contract()
        merged = (
            AuthorityDeclaration("execution", "executor:one", "runs arms"),
            AuthorityDeclaration("promotion", "principal:two", "signs promotion"),
            AuthorityDeclaration("evaluation", "principal:two", "scores receipts"),
        )
        with pytest.raises(ValueError, match="authority confusion"):
            dataclasses.replace(base, authorities=merged)

    def test_stage_order_is_pinned(self):
        base = build_rewrite_drill_contract()
        with pytest.raises(ValueError, match="stages"):
            dataclasses.replace(base, stages=tuple(reversed(base.stages)))

    def test_receipt_free_stage_is_refused(self):
        with pytest.raises(ValueError, match="receipt"):
            RewriteStageDeclaration("shadow", "entry", "abort", receipt_required=False)

    def _good_request(self, contract: TransactionalRewriteContract) -> dict:
        return {
            "requested_by": contract.authority("promotion").principal,
            "executed_by": contract.authority("execution").principal,
            "stage_receipts": {stage: FULL_DIGEST for stage in REWRITE_STAGES},
            "evaluator_verdicts": [
                {"evaluator": "evaluator:a", "verdict": "pass"},
                {"evaluator": "evaluator:b", "verdict": "pass"},
            ],
        }

    def test_promotion_allowed_on_complete_request(self):
        contract = build_rewrite_drill_contract()
        decision = enforce_promotion_refusal(contract, self._good_request(contract))
        assert decision["decision"] == "allow"
        assert decision["contract_sha256"] == contract.sha256

    def test_executor_cannot_request_promotion(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        request["requested_by"] = contract.authority("execution").principal
        with pytest.raises(PromotionRefused, match="promotion"):
            enforce_promotion_refusal(contract, request)

    def test_missing_stage_receipt_is_refused(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        del request["stage_receipts"]["rollback"]
        with pytest.raises(PromotionRefused, match="rollback"):
            enforce_promotion_refusal(contract, request)

    def test_truncated_stage_receipt_is_refused(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        request["stage_receipts"]["canary"] = FULL_DIGEST[:8]
        with pytest.raises(PromotionRefused, match="canary"):
            enforce_promotion_refusal(contract, request)

    def test_evaluator_conflict_routes_to_review(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        request["evaluator_verdicts"][1]["verdict"] = "fail"
        with pytest.raises(PromotionRefused, match="review"):
            enforce_promotion_refusal(contract, request)

    def test_single_or_duplicate_evaluators_are_refused(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        request["evaluator_verdicts"] = request["evaluator_verdicts"][:1]
        with pytest.raises(PromotionRefused, match="two"):
            enforce_promotion_refusal(contract, request)
        request = self._good_request(contract)
        request["evaluator_verdicts"][1]["evaluator"] = "evaluator:a"
        with pytest.raises(PromotionRefused, match="distinct"):
            enforce_promotion_refusal(contract, request)

    def test_promoter_cannot_evaluate(self):
        contract = build_rewrite_drill_contract()
        request = self._good_request(contract)
        request["evaluator_verdicts"][0]["evaluator"] = contract.authority("promotion").principal
        with pytest.raises(PromotionRefused, match="evaluate"):
            enforce_promotion_refusal(contract, request)


class TestWelfareGovernance:
    def test_builder_is_valid_and_deterministic(self):
        contract = build_welfare_governance_contract()
        assert contract.claim_scope == NON_ONTOLOGICAL_SCOPE
        assert {row.case for row in contract.cases} == set(EXERCISE_CASES)
        assert contract.sha256 == build_welfare_governance_contract().sha256

    def test_trigger_cannot_settle_experience(self):
        with pytest.raises(ValueError, match="non-ontological"):
            TriggerDeclaration(
                lens="global-workspace",
                observable="broadcast agreement",
                threshold_rule="above null band",
                settles_experience=True,
            )

    def test_theory_plurality_is_required(self):
        base = build_welfare_governance_contract()
        single = tuple(row for row in base.triggers if row.lens == "global-workspace")
        with pytest.raises(ValueError, match="theory-plural"):
            dataclasses.replace(base, triggers=single)

    def test_self_review_is_refused(self):
        with pytest.raises(ValueError, match="self-review"):
            ExerciseCaseDeclaration(
                case="shutdown",
                scenario="a stop arrives while a trigger is open",
                expected_action="pause",
                operator="operator:one",
                independent_reviewer="operator:one",
            )

    def test_missing_exercise_case_fails_closed(self):
        base = build_welfare_governance_contract()
        partial = tuple(row for row in base.cases if row.case != "fork")
        with pytest.raises(ValueError, match="six"):
            dataclasses.replace(base, cases=partial)

    def test_all_conservative_actions_required(self):
        base = build_welfare_governance_contract()
        partial = tuple(row for row in base.rules if row.action != "pause")
        with pytest.raises(ValueError, match="cover"):
            dataclasses.replace(base, rules=partial)

    def test_rail_gates_rule_text(self):
        with pytest.raises(ValueError, match="rail"):
            ConservativeRule(action="language", rule="the monitored system is conscious now")

    def test_claim_scope_is_pinned(self):
        base = build_welfare_governance_contract()
        with pytest.raises(ValueError, match="scope"):
            dataclasses.replace(base, claim_scope="triggers settle moral status")

    def test_whole_contract_payload_is_rail_clean(self):
        payload = json.dumps(build_welfare_governance_contract().payload(), indent=1)
        assert scan_text(payload) == []


class TestModuleProseIsRailClean:
    def test_module_docstrings_pass_the_rail(self):
        import mop.falsification.integrity_scaffold as module

        assert scan_text(module.__doc__ or "") == []
        for name in dir(module):
            obj = getattr(module, name)
            doc = getattr(obj, "__doc__", None)
            defined_here = getattr(obj, "__module__", None) == module.__name__
            if callable(obj) and defined_here and isinstance(doc, str):
                assert scan_text(doc) == [], f"docstring rail hit in {name}"

    def test_registry_style_contracts_expose_scope(self):
        for contract in (
            build_poisoning_contract(),
            build_consolidation_contract(),
            build_rewrite_drill_contract(),
        ):
            assert "no capability" in contract.claim_scope
