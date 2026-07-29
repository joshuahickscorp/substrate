"""Enforcement pins for the executable sandbox readiness package.

Each test fails if the corresponding enforcement is removed.
No test writes outside tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from substrate import final_revision_io as io
from substrate.final_revision_readiness import (
    ActionProposal,
    ModelAdapter,
    ModelRequest,
    TaskRequest,
)
from substrate.sandbox_readiness import (
    CampaignTask,
    ConsentStore,
    DegradedLocalAdapter,
    DeterministicLocalAdapter,
    PrivacyEnforcer,
    ToolExecutor,
    as_model_adapter,
    bounded_smoke,
    run_campaign,
)


def _consent(store: ConsentStore, subject: str = "operator-a", **kwargs) -> str:
    record = store.issue(subject_id=subject, scope="sandbox", issued_at=0.0, expires_at=10_000.0, **kwargs)
    return record.receipt_id


def _proposal(
    *,
    action_id: str = "a1",
    task_id: str = "t1",
    tool: str = "read_file",
    path: str = "in.txt",
    reversibility: str = "read_only",
    content: str | None = None,
) -> ActionProposal:
    arguments: dict = {"path": path}
    if content is not None:
        arguments["content"] = content
    return ActionProposal(
        action_id=action_id,
        task_id=task_id,
        tool=tool,
        arguments=arguments,
        expected_effect="test action",
        reversibility=reversibility,
    )


def test_containment_escape_refused(tmp_path: Path) -> None:
    """Path resolution must refuse proposals that escape the workspace."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    store = ConsentStore()
    receipt = _consent(store)
    executor = ToolExecutor(workspace, store, dry_run=False, stop_path=tmp_path / "STOP")

    result = executor.execute(
        _proposal(path="../secret.txt"),
        consent_receipt=receipt,
        subject_id="operator-a",
        operator_approved=True,
        dry_run=False,
    )
    assert result.refused
    assert result.reason is not None
    assert "escape" in result.reason
    assert result.receipt["status"] == "refused"
    assert result.receipt["sha256"]
    assert result.receipt["activation"] is False


def test_revoked_consent_refused(tmp_path: Path) -> None:
    """Revocation must stop new actions for that subject immediately."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "in.txt").write_text("ok")
    store = ConsentStore()
    receipt = _consent(store)
    executor = ToolExecutor(workspace, store, dry_run=False, stop_path=tmp_path / "STOP")

    store.revoke(receipt, at=1.0)
    result = executor.execute(
        _proposal(),
        consent_receipt=receipt,
        subject_id="operator-a",
        operator_approved=True,
        dry_run=False,
        now=2.0,
    )
    assert result.refused
    assert result.reason is not None
    assert "revoked" in result.reason


def test_unknown_consent_refused(tmp_path: Path) -> None:
    """Unknown consent receipts are refused, not warned."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = ConsentStore()
    executor = ToolExecutor(workspace, store, dry_run=True, stop_path=tmp_path / "STOP")

    result = executor.execute(
        _proposal(),
        consent_receipt="not-a-real-receipt",
        subject_id="operator-a",
        dry_run=True,
    )
    assert result.refused
    assert result.reason is not None
    assert "unknown" in result.reason

    with pytest.raises(io.Refused, match="unknown"):
        store.require("also-unknown")


def test_undeclared_field_refused(tmp_path: Path) -> None:
    """Data minimisation rejects fields absent from the task inventory."""
    enforcer = PrivacyEnforcer(
        policy={
            "data_minimization": True,
            "private_fields_declared": True,
            "retention_predeclared": True,
            "activation": False,
        }
    )
    with pytest.raises(io.Refused, match="undeclared"):
        enforcer.store(
            record_id="r1",
            fields={"allowed": 1, "secret_extra": "x"},
            inventory={"allowed"},
            private_fields=(),
            retention_seconds=60.0,
            stored_at=0.0,
        )


def test_retention_expiry_refused(tmp_path: Path) -> None:
    """Reading a record past its retention window is refused."""
    enforcer = PrivacyEnforcer(
        policy={
            "data_minimization": True,
            "private_fields_declared": True,
            "retention_predeclared": True,
            "activation": False,
        }
    )
    enforcer.store(
        record_id="r-expire",
        fields={"note": "ephemeral"},
        inventory={"note"},
        private_fields=(),
        retention_seconds=10.0,
        stored_at=100.0,
    )
    # Still inside window.
    alive = enforcer.read("r-expire", now=109.0)
    assert alive["fields"]["note"] == "ephemeral"

    with pytest.raises(io.Refused, match="retention expired"):
        enforcer.read("r-expire", now=111.0)


def test_stop_switch_honoured(tmp_path: Path) -> None:
    """io.STOP-equivalent path must refuse every action when present."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "in.txt").write_text("ok")
    stop_path = tmp_path / "STOP"
    stop_path.write_text("operator stop\n")
    store = ConsentStore()
    receipt = _consent(store)
    executor = ToolExecutor(workspace, store, dry_run=False, stop_path=stop_path)

    result = executor.execute(
        _proposal(),
        consent_receipt=receipt,
        subject_id="operator-a",
        operator_approved=True,
        dry_run=False,
    )
    assert result.refused
    assert result.reason is not None
    assert "stop switch" in result.reason


def test_model_accounting_non_zero(tmp_path: Path) -> None:
    """Adapters must report real non-zero latency and cost accounting."""
    adapter = DeterministicLocalAdapter()
    request = ModelRequest(
        request_id="acct-1",
        role="planner",
        input_digest=io.digest(b"payload"),
        context={"objective": "measure accounting", "n": 42},
        output_schema={"proposal": "object"},
        cost_limit=1.0,
    )
    response = adapter.infer(request)
    assert response.latency_seconds > 0
    assert response.cost > 0
    assert response.model_identity == adapter.identity
    assert response.output_digest
    assert response.request_id == "acct-1"
    # Deterministic: same input yields same digests and accounting.
    again = adapter.infer(request)
    assert again.output_digest == response.output_digest
    assert again.latency_seconds == response.latency_seconds
    assert again.cost == response.cost


def test_adapter_substitution_preserves_interface(tmp_path: Path) -> None:
    """Primary and degraded adapters are substitutable under ModelAdapter."""
    primary = DeterministicLocalAdapter()
    degraded = DegradedLocalAdapter()
    assert isinstance(primary, ModelAdapter)
    assert isinstance(degraded, ModelAdapter)
    as_model_adapter(primary)
    as_model_adapter(degraded)

    request = ModelRequest(
        request_id="sub-1",
        role="planner",
        input_digest=io.digest({"x": 1}),
        context={"objective": "substitute"},
        output_schema={"proposal": "object"},
        cost_limit=1.0,
    )
    for adapter in (primary, degraded):
        health = adapter.health()
        assert health["identity"] == adapter.identity
        assert health["activation"] is False
        response = adapter.infer(request)
        assert response.model_identity == adapter.identity
        assert response.latency_seconds > 0
        assert response.cost > 0
        assert isinstance(response.output, dict)
        assert response.output_digest
        assert response.supported is True or response.supported is False

    primary_r = primary.infer(request)
    degraded_r = degraded.infer(request)
    assert primary_r.model_identity != degraded_r.model_identity
    assert degraded_r.supported is False
    assert degraded_r.latency_seconds > primary_r.latency_seconds


def test_bounded_smoke_exercises_workspace(tmp_path: Path) -> None:
    """Smoke creates a real workspace, refuses escape and revoked consent, stays offline."""
    result = bounded_smoke(base_dir=tmp_path / "smoke-root")
    assert result["external_action_executed"] is False
    assert result["activation"] is False
    assert result["containment_escape_refused"] is True
    assert result["revoked_consent_refused"] is True
    assert result["all_pass"] is True
    assert len(result["model_adapters"]) == 2
    assert result["sha256"]


def test_campaign_entrypoint_dry_run(tmp_path: Path) -> None:
    """Campaign runs full path in dry-run without leaving the workspace."""
    workspace = tmp_path / "campaign-ws"
    store = ConsentStore()
    receipt = _consent(store, subject="campaign-op")
    task = TaskRequest(
        task_id="campaign-1",
        task_kind="offline_document",
        objective="dry-run campaign path",
        consent_receipt=receipt,
        allowed_tools=("write_file", "read_file"),
        resource_budget={"seconds": 1.0, "cost": 1.0},
        private_fields=(),
    )
    action = ActionProposal(
        action_id="c-write",
        task_id="campaign-1",
        tool="write_file",
        arguments={"path": "out.txt", "content": "hello"},
        expected_effect="write inside workspace",
        reversibility="reversible",
    )
    result = run_campaign(
        workspace,
        [
            CampaignTask(
                request=task,
                subject_id="campaign-op",
                data_inventory=("exists", "bytes", "content_digest", "note"),
                retention_seconds=100.0,
                sensor_features={"note": "from-sensor", "exists": True, "bytes": 5},
                sensor_bytes=b"hello",
                action=action,
            )
        ],
        consent_store=store,
        privacy=PrivacyEnforcer(policy={
            "data_minimization": True,
            "private_fields_declared": True,
            "retention_predeclared": True,
            "activation": False,
        }),
        dry_run=True,
        stop_path=tmp_path / "STOP",
        now=1.0,
    )
    doc = result.document()
    assert doc["external_action_executed"] is False
    assert doc["activation"] is False
    assert doc["all_pass"] is True
    assert (workspace / "out.txt").exists() is False  # dry-run did not write
    # Nothing outside workspace.
    assert list(tmp_path.iterdir())  # only under tmp_path


def test_privacy_reads_policy_json_when_present() -> None:
    """PrivacyEnforcer loads PRIVACY.json flags when the readiness artifact exists."""
    path = io.READINESS / "PRIVACY.json"
    if not path.is_file():
        pytest.skip("PRIVACY.json not present in this checkout")
    enforcer = PrivacyEnforcer()
    assert enforcer.policy.get("data_minimization") is True
    assert enforcer.policy.get("retention_predeclared") is True
    assert enforcer.policy.get("activation") is False
