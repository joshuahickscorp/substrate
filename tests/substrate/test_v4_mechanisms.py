from __future__ import annotations

import copy
import json

import pytest

from substrate import v4, v4campaign, v4canary
from substrate import v4config as C
from substrate import v4fabric as F
from substrate import v4principal as P
from substrate.runtime import Refused, StructuralSubstrate


def test_v4_observation_has_no_answer_or_latent_authority() -> None:
    task = F.generate_task(
        C.SPLITS["construction"][0],
        "cross_representation_isomorphisms",
        0,
        "construction",
    )
    observation = json.dumps(task.observation(), sort_keys=True)
    assert "private_target" not in observation
    assert "latent_family" not in observation
    assert "oracle_mapping" not in observation


def test_v4_one_model_executes_intervention_counterfactual_and_alignment() -> None:
    seed = C.SPLITS["construction"][1]
    entity = StructuralSubstrate()
    training = F.generate_task(seed, "causal_systems", 0, "construction", include_training=True)
    intervention = entity.step_structural(training)
    counterfactual = entity.step_structural(F.generate_task(seed, "counterfactual_planning", 1, "construction", include_training=True))
    target_representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)]
    alignment = entity.step_structural(
        F.generate_task(
            seed,
            "cross_representation_isomorphisms",
            2,
            "construction",
            representation=target_representation,
            include_training=False,
        )
    )
    assert intervention["outcome"]["correct"]
    assert counterfactual["outcome"]["correct"]
    assert alignment["outcome"]["correct"]
    assert intervention["structural_execution"]["model"] == counterfactual["structural_execution"]["model"]
    assert alignment["structural_execution"]["model"] == intervention["structural_execution"]["model"]


def test_v4_checkpoint_is_exact_and_corruption_is_refused() -> None:
    seed = C.SPLITS["construction"][2]
    entity = StructuralSubstrate()
    entity.step_structural(F.generate_task(seed, "causal_systems", 0, "construction", include_training=True))
    checkpoint = entity.checkpoint()
    restored = StructuralSubstrate().restore(copy.deepcopy(checkpoint))
    assert restored.checkpoint()["identity"] == checkpoint["identity"]
    corrupted = copy.deepcopy(checkpoint)
    model = next(iter(corrupted["extension"]["structural_world"]["models"].values()))
    model["causal_edges"].pop()
    with pytest.raises(Refused):
        StructuralSubstrate().restore(corrupted)


def test_v4_canary_gate_is_complete() -> None:
    result = v4canary.run()["evidence"]
    assert result["total"] == 46
    assert result["passed"] == 46
    assert result["failed"] == []
    assert result["all_pass"]
    assert result["activation"] is False


def test_v4_principal_dag_and_unit_are_frozen_and_executable() -> None:
    units = P.work_units()
    episodes = len(units) * len(C.PHASES) * P.EPISODES_PER_PHASE
    assert 1_500 <= len(units) <= 6_000
    assert 150_000 <= episodes <= 600_000
    assert len(C.CORE_ARMS) == 14
    assert len(C.PHASES) == 17
    unit = next(unit for unit in units if unit.split == "principal" and unit.arm == "full_v4")
    receipt = P.execute_unit(unit)
    assert P.validate_receipt(receipt, unit)
    assert receipt["summary"]["checkpoint_exact"]
    assert receipt["summary"]["body_continuity"]
    assert receipt["summary"]["history_specialization"]["mean_specialization_margin"] >= C.SESOI


def test_v4_exact_command_surface_and_stage_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        v4campaign,
        "root_cause",
        lambda: {"audit": {"conclusion": "audited", "scientific_null_preserved": True}},
    )
    monkeypatch.setattr(
        v4campaign,
        "freeze",
        lambda: {"sealed": ["authority"], "split_disjoint": True},
    )
    v4.main(["audit"])
    audit = json.loads(capsys.readouterr().out)
    assert audit["root_cause"] == "audited"
    assert audit["frozen_authorities"] == 1
    monkeypatch.setattr(
        v4canary,
        "run",
        lambda: {
            "evidence": {"passed": 46, "total": 46, "all_pass": True, "all_terminal": True},
            "bed": {"all_valid": True},
        },
    )
    with pytest.raises(SystemExit) as canary_exit:
        v4.main(["canaries"])
    assert canary_exit.value.code == 0
    canaries = json.loads(capsys.readouterr().out)
    assert canaries["moderate_pilot_licensed"]
    status = P.status()
    assert status["current_stage"] in status["stages"]
    assert set(status["stages"]) == {
        "root_cause_audit",
        "mechanism_construction",
        "cheap_admission",
        "moderate_pilot",
        "principal_development",
        "replication",
        "open_world_review",
        "verification",
        "terminal_classification",
    }
    assert "SUBSTRATE_V4_TERMINAL_REPORT.md" in C.DELIVERABLES
    assert all(digest != "missing" for _, digest in P.work_units()[0].input_digests)
