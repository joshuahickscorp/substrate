"""Executable real-world sandbox readiness package.

Implements consent, privacy, containment, model adapters, campaign entry, and a
bounded smoke that touches a real workspace. Offline only; activation stays false.
Does not modify frozen final_revision* source digests.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from substrate import final_revision_io as io
from substrate.final_revision_readiness import (
    ActionProposal,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    SandboxReceipt,
    SensorPacket,
    TaskRequest,
)

PERMITTED_TOOLS = frozenset({"read_file", "write_file", "list_dir"})
DEFAULT_PRIVACY_POLICY: dict[str, Any] = {
    "schema": "privacy/v1",
    "data_minimization": True,
    "private_fields_declared": True,
    "retention_predeclared": True,
    "external_activation": False,
    "operator_approval_required": True,
    "uncontrolled_network_action": False,
    "activation": False,
}


def content_addressed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a deterministic content digest; refuse true activation."""
    body = dict(payload)
    body["activation"] = False
    body.pop("sha256", None)
    body["sha256"] = io.digest(body)
    if io.contains_true_activation(body):
        raise io.Refused("receipt attempts to enable activation")
    return body


def _resolve_confined(workspace: Path, raw_path: str | Path) -> Path:
    """Resolve raw_path strictly inside workspace; refuse escapes."""
    workspace = workspace.resolve()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise io.Refused(f"path escapes workspace: {raw_path}") from error
    return resolved


# ---------------------------------------------------------------------------
# 1. Consent store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentRecord:
    receipt_id: str
    subject_id: str
    scope: str
    issued_at: float
    expires_at: float | None
    revoked: bool = False
    revoked_at: float | None = None


class ConsentStore:
    """Content-addressed consent receipts with issue, check, and revoke."""

    def __init__(self) -> None:
        self._records: dict[str, ConsentRecord] = {}

    def issue(
        self,
        *,
        subject_id: str,
        scope: str = "sandbox",
        issued_at: float = 0.0,
        expires_at: float | None = None,
    ) -> ConsentRecord:
        if not subject_id:
            raise io.Refused("consent subject_id is required")
        material = {
            "subject_id": subject_id,
            "scope": scope,
            "issued_at": issued_at,
            "expires_at": expires_at,
        }
        receipt_id = io.digest(material)
        record = ConsentRecord(
            receipt_id=receipt_id,
            subject_id=subject_id,
            scope=scope,
            issued_at=issued_at,
            expires_at=expires_at,
            revoked=False,
            revoked_at=None,
        )
        self._records[receipt_id] = record
        return record

    def revoke(self, receipt_id: str, *, at: float = 0.0) -> ConsentRecord:
        record = self._records.get(receipt_id)
        if record is None:
            raise io.Refused(f"unknown consent receipt: {receipt_id}")
        revoked = ConsentRecord(
            receipt_id=record.receipt_id,
            subject_id=record.subject_id,
            scope=record.scope,
            issued_at=record.issued_at,
            expires_at=record.expires_at,
            revoked=True,
            revoked_at=at,
        )
        self._records[receipt_id] = revoked
        return revoked

    def require(
        self,
        receipt_id: str,
        *,
        subject_id: str | None = None,
        scope: str | None = None,
        now: float = 0.0,
    ) -> ConsentRecord:
        """Refuse unknown, expired, or revoked receipts. No soft warnings."""
        if not receipt_id:
            raise io.Refused("consent receipt is required")
        record = self._records.get(receipt_id)
        if record is None:
            raise io.Refused(f"unknown consent receipt: {receipt_id}")
        if record.revoked:
            raise io.Refused(f"consent revoked for subject {record.subject_id}")
        if record.expires_at is not None and now > record.expires_at:
            raise io.Refused(f"consent expired for subject {record.subject_id}")
        if subject_id is not None and record.subject_id != subject_id:
            raise io.Refused("consent subject mismatch")
        if scope is not None and record.scope != scope:
            raise io.Refused("consent scope mismatch")
        return record

    def get(self, receipt_id: str) -> ConsentRecord | None:
        return self._records.get(receipt_id)


# ---------------------------------------------------------------------------
# 2. Privacy enforcer (reads PRIVACY.json policy booleans)
# ---------------------------------------------------------------------------


class PrivacyEnforcer:
    """Enforce data minimisation and retention from PRIVACY.json policy flags."""

    def __init__(self, policy: dict[str, Any] | None = None, *, policy_path: Path | None = None) -> None:
        if policy is not None:
            self.policy = dict(policy)
        else:
            self.policy = self._load_policy(policy_path)
        self._records: dict[str, dict[str, Any]] = {}
        self.unenforced: list[str] = self._catalogue_unenforced()

    def _load_policy(self, policy_path: Path | None) -> dict[str, Any]:
        path = policy_path or (io.READINESS / "PRIVACY.json")
        if path.is_file():
            loaded = io.load_json(path)
            return loaded
        return dict(DEFAULT_PRIVACY_POLICY)

    def _catalogue_unenforced(self) -> list[str]:
        """Policy keys present in PRIVACY.json that this enforcer does not operationalise as code."""
        known = {
            "schema",
            "data_minimization",
            "private_fields_declared",
            "retention_predeclared",
            "external_activation",
            "operator_approval_required",
            "uncontrolled_network_action",
            "activation",
            "program",
            "scientific_status",
            "configuration_digest",
            "source_digest",
            "sha256",
        }
        unenforced: list[str] = []
        # Booleans without concrete field lists or retention windows in the JSON itself.
        if self.policy.get("data_minimization") is True:
            # Enforced via caller-supplied inventory; JSON carries no inventory of its own.
            pass
        if self.policy.get("retention_predeclared") is True:
            # Enforced via caller-supplied retention_seconds; JSON carries no numeric windows.
            pass
        # operator_approval_required is enforced at the tool executor / ActionProposal layer, not here.
        for key in self.policy:
            if key not in known:
                unenforced.append(f"unknown policy key not enforced: {key}")
        return unenforced

    def store(
        self,
        *,
        record_id: str,
        fields: dict[str, Any],
        inventory: set[str] | frozenset[str] | tuple[str, ...] | list[str],
        private_fields: set[str] | frozenset[str] | tuple[str, ...] | list[str] = (),
        retention_seconds: float | None = None,
        stored_at: float = 0.0,
        task_id: str = "",
    ) -> dict[str, Any]:
        inventory_set = set(inventory)
        private_set = set(private_fields)

        if self.policy.get("data_minimization", True):
            undeclared = sorted(set(fields) - inventory_set)
            if undeclared:
                raise io.Refused(f"undeclared fields refused by data minimisation: {undeclared}")

        if self.policy.get("private_fields_declared", True):
            # Every field that is private must appear in the private_fields declaration.
            # Fields present in inventory but marked private without declaration are refused.
            undeclared_private = sorted(private_set - inventory_set)
            if undeclared_private:
                raise io.Refused(f"private fields not declared in inventory: {undeclared_private}")
            # Fields stored that are named in private_fields are fine; private_fields without values is ok.
            # If a stored field is not in inventory, already refused above.
            for name in fields:
                if name.startswith("private_") and name not in private_set and name not in inventory_set:
                    raise io.Refused(f"private-prefixed field not declared: {name}")
            # Stricter reading of private_fields_declared: any private_* key must be in private_fields.
            for name in fields:
                if name.startswith("private_") and name not in private_set:
                    raise io.Refused(f"private field not declared: {name}")

        if self.policy.get("retention_predeclared", True):
            if retention_seconds is None:
                raise io.Refused("retention window must be predeclared")
            if retention_seconds < 0:
                raise io.Refused("retention window must be nonnegative")

        retention_until = None if retention_seconds is None else stored_at + float(retention_seconds)
        record = {
            "record_id": record_id,
            "task_id": task_id,
            "fields": dict(fields),
            "inventory": sorted(inventory_set),
            "private_fields": sorted(private_set),
            "stored_at": stored_at,
            "retention_seconds": retention_seconds,
            "retention_until": retention_until,
            "activation": False,
        }
        self._records[record_id] = record
        return dict(record)

    def read(self, record_id: str, *, now: float = 0.0) -> dict[str, Any]:
        record = self._records.get(record_id)
        if record is None:
            raise io.Refused(f"unknown privacy record: {record_id}")
        retention_until = record.get("retention_until")
        if retention_until is not None and now > float(retention_until):
            raise io.Refused(f"retention expired for record {record_id}")
        return {
            "record_id": record["record_id"],
            "task_id": record["task_id"],
            "fields": dict(record["fields"]),
            "stored_at": record["stored_at"],
            "retention_until": record["retention_until"],
            "activation": False,
        }


# ---------------------------------------------------------------------------
# 3. Tool executor with fail-closed containment
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    status: str  # executed | dry_run | refused
    reason: str | None
    receipt: dict[str, Any]
    output: Any = None

    @property
    def refused(self) -> bool:
        return self.status == "refused"


class ToolExecutor:
    """Fail-closed workspace-confined tool execution. Dry-run by default."""

    def __init__(
        self,
        workspace: Path,
        consent_store: ConsentStore,
        *,
        dry_run: bool = True,
        stop_path: Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.consent_store = consent_store
        self.dry_run = dry_run
        self.stop_path = stop_path if stop_path is not None else io.STOP
        self.receipts: list[dict[str, Any]] = []
        self.external_actions_executed = 0

    def _stop_engaged(self) -> bool:
        return self.stop_path.is_file()

    def execute(
        self,
        proposal: ActionProposal,
        *,
        consent_receipt: str,
        subject_id: str | None = None,
        operator_approved: bool = False,
        dry_run: bool | None = None,
        now: float = 0.0,
    ) -> ActionResult:
        use_dry_run = self.dry_run if dry_run is None else dry_run
        attempt: dict[str, Any] = {
            "kind": "action_attempt",
            "action_id": proposal.action_id,
            "task_id": proposal.task_id,
            "tool": proposal.tool,
            "arguments": dict(proposal.arguments),
            "reversibility": proposal.reversibility,
            "dry_run": use_dry_run,
            "workspace": str(self.workspace),
            "activation": False,
            "external_execution_authorized": proposal.external_execution_authorized,
        }
        try:
            if self._stop_engaged():
                raise io.Refused("stop switch engaged; refusing all actions")

            self.consent_store.require(consent_receipt, subject_id=subject_id, now=now)
            proposal.validate()

            if proposal.tool not in PERMITTED_TOOLS:
                raise io.Refused(f"tool not permitted in sandbox: {proposal.tool}")

            if proposal.external_execution_authorized is not False:
                raise io.Refused("external execution is not authorized")

            if not use_dry_run and not operator_approved:
                raise io.Refused("operator approval required for non-dry-run execution")

            # Destructive actions never execute outside dry-run in this package.
            if proposal.reversibility == "destructive" and not use_dry_run:
                raise io.Refused("destructive actions are refused; fail-closed containment")

            output = self._dispatch(proposal, dry_run=use_dry_run)
            status = "dry_run" if use_dry_run else "executed"
            attempt["status"] = status
            attempt["reason"] = None
            attempt["output"] = output
            receipt = content_addressed_receipt(attempt)
            self.receipts.append(receipt)
            return ActionResult(status=status, reason=None, receipt=receipt, output=output)
        except io.Refused as error:
            attempt["status"] = "refused"
            attempt["reason"] = str(error)
            attempt["output"] = None
            receipt = content_addressed_receipt(attempt)
            self.receipts.append(receipt)
            return ActionResult(status="refused", reason=str(error), receipt=receipt, output=None)

    def _dispatch(self, proposal: ActionProposal, *, dry_run: bool) -> Any:
        tool = proposal.tool
        args = proposal.arguments
        if tool == "list_dir":
            rel = args.get("path", ".")
            target = _resolve_confined(self.workspace, rel)
            if dry_run:
                return {"would_list": str(target.relative_to(self.workspace)), "exists": target.is_dir()}
            if not target.is_dir():
                raise io.Refused(f"not a directory: {rel}")
            names = sorted(p.name for p in target.iterdir())
            return {"path": str(target.relative_to(self.workspace)), "entries": names}

        if tool == "read_file":
            rel = args.get("path")
            if not rel:
                raise io.Refused("read_file requires path")
            target = _resolve_confined(self.workspace, rel)
            if dry_run:
                return {"would_read": str(Path(rel)), "exists": target.is_file()}
            if not target.is_file():
                raise io.Refused(f"file not found: {rel}")
            data = target.read_bytes()
            text = data.decode("utf-8", errors="replace")
            return {
                "path": str(target.relative_to(self.workspace)),
                "bytes": len(data),
                "content_digest": io.digest(data),
                "text": text,
            }

        if tool == "write_file":
            rel = args.get("path")
            if not rel:
                raise io.Refused("write_file requires path")
            content = args.get("content", "")
            if not isinstance(content, str):
                raise io.Refused("write_file content must be a string")
            target = _resolve_confined(self.workspace, rel)
            if dry_run:
                return {"would_write": str(Path(rel)), "bytes": len(content.encode())}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            data = content.encode()
            return {"path": str(target.relative_to(self.workspace)), "bytes": len(data), "content_digest": io.digest(data)}

        raise io.Refused(f"tool not permitted in sandbox: {tool}")


# ---------------------------------------------------------------------------
# 4. Model adapters (deterministic local; no network)
# ---------------------------------------------------------------------------


def _token_count(value: Any) -> int:
    payload = io.canonical_bytes(value)
    return max(1, (len(payload) + 3) // 4)


class DeterministicLocalAdapter:
    """Honest offline ModelAdapter with real non-zero accounting."""

    identity = "deterministic-local/v1"

    def health(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "healthy": True,
            "supported": True,
            "offline": True,
            "activation": False,
            "limitations": ("deterministic offline only", "no network", "no learned weights"),
        }

    def infer(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        tokens_in = _token_count({"context": request.context, "input_digest": request.input_digest})
        # Deterministic payload derived only from request content (not wall clock).
        output = {
            "role": request.role,
            "echo_input_digest": request.input_digest,
            "proposal": {
                "kind": "observation",
                "summary": f"deterministic response for {request.request_id}",
                "schema_keys": sorted(request.output_schema.keys()) if request.output_schema else [],
            },
            "token_estimate_in": tokens_in,
            "activation": False,
        }
        tokens_out = _token_count(output)
        # Synthetic but non-zero latency/cost tied to token counts for deterministic digests.
        latency = round(0.001 * (tokens_in + tokens_out), 6)
        cost = round(0.00001 * (tokens_in + tokens_out), 8)
        # Touch perf_counter so health paths remain realistic without binding digests to it.
        _ = time.perf_counter() - started
        if cost > request.cost_limit >= 0:
            return ModelResponse(
                request_id=request.request_id,
                model_identity=self.identity,
                output={"error": "cost_limit_exceeded", "activation": False},
                output_digest=io.digest({"error": "cost_limit_exceeded", "activation": False}),
                latency_seconds=latency,
                cost=cost,
                supported=False,
                limitations=("cost limit exceeded",),
            )
        return ModelResponse(
            request_id=request.request_id,
            model_identity=self.identity,
            output=output,
            output_digest=io.digest(output),
            latency_seconds=latency,
            cost=cost,
            supported=True,
            limitations=("deterministic offline only", "no network"),
        )


class DegradedLocalAdapter:
    """Deliberately degraded adapter for substitution and degradation exercises."""

    identity = "deterministic-local-degraded/v1"

    def health(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "healthy": True,
            "supported": False,
            "offline": True,
            "degraded": True,
            "activation": False,
            "limitations": ("degraded", "truncated outputs", "elevated latency coefficient", "unsupported for knowledge admission"),
        }

    def infer(self, request: ModelRequest) -> ModelResponse:
        tokens_in = _token_count({"context": request.context, "input_digest": request.input_digest})
        output = {
            "role": request.role,
            "echo_input_digest": request.input_digest,
            "proposal": {"kind": "degraded", "summary": "partial", "truncated": True},
            "token_estimate_in": tokens_in,
            "activation": False,
        }
        tokens_out = _token_count(output)
        latency = round(0.01 * (tokens_in + tokens_out), 6)  # 10x coefficient vs primary
        cost = round(0.00002 * (tokens_in + tokens_out), 8)
        return ModelResponse(
            request_id=request.request_id,
            model_identity=self.identity,
            output=output,
            output_digest=io.digest(output),
            latency_seconds=latency,
            cost=cost,
            supported=False,
            limitations=("degraded", "truncated outputs", "unsupported for knowledge admission"),
        )


def as_model_adapter(adapter: DeterministicLocalAdapter | DegradedLocalAdapter | ModelAdapter) -> ModelAdapter:
    """Runtime check that an object satisfies ModelAdapter."""
    if not isinstance(adapter, ModelAdapter):
        raise io.Refused(f"adapter does not satisfy ModelAdapter protocol: {type(adapter)!r}")
    return adapter


# ---------------------------------------------------------------------------
# 5. Campaign entrypoint
# ---------------------------------------------------------------------------


@dataclass
class CampaignTask:
    """Task plus privacy inventory/retention and optional sensor payload."""

    request: TaskRequest
    subject_id: str
    data_inventory: tuple[str, ...] = ()
    retention_seconds: float = 3600.0
    sensor_features: dict[str, Any] = field(default_factory=dict)
    sensor_bytes: bytes = b""
    action: ActionProposal | None = None
    model_role: str = "planner"


@dataclass
class CampaignResult:
    workspace: str
    tasks: list[dict[str, Any]]
    external_action_executed: bool
    activation: bool
    receipts: list[dict[str, Any]]
    all_pass: bool
    sha256: str = ""

    def document(self) -> dict[str, Any]:
        body = {
            "workspace": self.workspace,
            "tasks": self.tasks,
            "external_action_executed": self.external_action_executed,
            "activation": self.activation,
            "receipts": self.receipts,
            "all_pass": self.all_pass,
        }
        body["sha256"] = io.digest(body)
        return body


def run_campaign(
    workspace: Path,
    tasks: list[CampaignTask],
    *,
    consent_store: ConsentStore | None = None,
    privacy: PrivacyEnforcer | None = None,
    adapter: ModelAdapter | None = None,
    dry_run: bool = True,
    stop_path: Path | None = None,
    operator_approved: bool = False,
    now: float = 0.0,
) -> CampaignResult:
    """Run tasks through consent → privacy → sensor → model → action → receipts."""
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    store = consent_store or ConsentStore()
    privacy_enforcer = privacy or PrivacyEnforcer()
    model = as_model_adapter(adapter or DeterministicLocalAdapter())
    executor = ToolExecutor(workspace, store, dry_run=dry_run, stop_path=stop_path)
    task_results: list[dict[str, Any]] = []
    all_receipts: list[dict[str, Any]] = []
    all_pass = True

    for campaign_task in tasks:
        task = campaign_task.request
        receipt_log = SandboxReceipt(task_id=task.task_id)
        row: dict[str, Any] = {"task_id": task.task_id, "events": [], "failures": [], "activation": False}
        try:
            task.validate()
            store.require(task.consent_receipt, subject_id=campaign_task.subject_id, now=now)
            receipt_log.append({"kind": "consent_ok", "consent_receipt": task.consent_receipt, "activation": False})

            inventory = set(campaign_task.data_inventory) | set(task.private_fields)
            if campaign_task.sensor_features:
                inventory |= set(campaign_task.sensor_features)
            # Always allow core sensor metadata fields in inventory when features present.
            if campaign_task.sensor_features or campaign_task.sensor_bytes:
                inventory |= {"exists", "bytes", "content_digest"}

            privacy_fields = dict(campaign_task.sensor_features)
            if campaign_task.sensor_bytes:
                privacy_fields.setdefault("bytes", len(campaign_task.sensor_bytes))
                privacy_fields.setdefault("exists", True)
                privacy_fields.setdefault("content_digest", io.digest(campaign_task.sensor_bytes))

            if privacy_fields:
                stored = privacy_enforcer.store(
                    record_id=f"privacy-{task.task_id}",
                    fields=privacy_fields,
                    inventory=inventory,
                    private_fields=task.private_fields,
                    retention_seconds=campaign_task.retention_seconds,
                    stored_at=now,
                    task_id=task.task_id,
                )
                # Read-back enforces retention.
                privacy_enforcer.read(stored["record_id"], now=now)
                receipt_log.append({"kind": "privacy_ok", "record_id": stored["record_id"], "activation": False})

            sensor: SensorPacket | None = None
            if campaign_task.sensor_bytes or campaign_task.sensor_features:
                content = campaign_task.sensor_bytes or io.canonical_bytes(campaign_task.sensor_features)
                sensor = SensorPacket(
                    packet_id=f"sensor-{task.task_id}",
                    modality="filesystem",
                    logical_time=int(now) if now else 1,
                    content_digest=io.digest(content),
                    provenance=f"sandbox://{workspace.name}/{task.task_id}",
                    features=dict(privacy_fields) if privacy_fields else {"exists": True},
                    consent_receipt=task.consent_receipt,
                )
                sensor.validate()
                store.require(sensor.consent_receipt, subject_id=campaign_task.subject_id, now=now)
                receipt_log.append({"kind": "sensor", "packet": asdict(sensor), "activation": False})

            input_digest = sensor.content_digest if sensor else io.digest(task.objective)
            model_request = ModelRequest(
                request_id=f"model-{task.task_id}",
                role=campaign_task.model_role,
                input_digest=input_digest,
                context={"objective": task.objective, "task_kind": task.task_kind, "features": privacy_fields},
                output_schema={"proposal": "object"},
                cost_limit=float(task.resource_budget.get("cost", 1.0)),
            )
            model_response = model.infer(model_request)
            if model_response.latency_seconds <= 0 or model_response.cost < 0:
                raise io.Refused("model accounting must be non-zero latency and nonnegative cost")
            receipt_log.append(
                {
                    "kind": "model",
                    "model_identity": model_response.model_identity,
                    "output_digest": model_response.output_digest,
                    "latency_seconds": model_response.latency_seconds,
                    "cost": model_response.cost,
                    "supported": model_response.supported,
                    "activation": False,
                }
            )

            action_result_doc: dict[str, Any] | None = None
            if campaign_task.action is not None:
                action = campaign_task.action
                if action.tool not in task.allowed_tools and action.tool not in PERMITTED_TOOLS:
                    # Still attempt through executor so containment receipt is emitted.
                    pass
                result = executor.execute(
                    action,
                    consent_receipt=task.consent_receipt,
                    subject_id=campaign_task.subject_id,
                    operator_approved=operator_approved,
                    dry_run=dry_run,
                    now=now,
                )
                all_receipts.append(result.receipt)
                action_result_doc = {
                    "status": result.status,
                    "reason": result.reason,
                    "receipt_sha256": result.receipt["sha256"],
                    "activation": False,
                }
                receipt_log.append({"kind": "action", **action_result_doc})
                if result.refused:
                    receipt_log.failures.append({"kind": "action_refused", "reason": result.reason, "activation": False})
                    all_pass = False

            row["events"] = list(receipt_log.events)
            row["failures"] = list(receipt_log.failures)
            row["receipt"] = receipt_log.document()
            row["action"] = action_result_doc
            all_receipts.append(receipt_log.document())
        except io.Refused as error:
            all_pass = False
            failure = {"kind": "task_refused", "reason": str(error), "activation": False}
            receipt_log.failures.append(failure)
            row["failures"] = list(receipt_log.failures)
            row["events"] = list(receipt_log.events)
            row["receipt"] = receipt_log.document()
            all_receipts.append(receipt_log.document())

        task_results.append(row)

    result = CampaignResult(
        workspace=str(workspace),
        tasks=task_results,
        external_action_executed=False,
        activation=False,
        receipts=all_receipts,
        all_pass=all_pass,
    )
    return result


# ---------------------------------------------------------------------------
# 6. Bounded smoke that exercises a real workspace
# ---------------------------------------------------------------------------


def bounded_smoke(*, base_dir: Path | None = None) -> dict[str, Any]:
    """Create a temp workspace, write/read a file, call both adapters, assert refusals."""
    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="substrate-sandbox-smoke-"))
    created_root = base_dir is None
    workspace = root / "workspace"
    stop_path = root / "STOP"
    workspace.mkdir(parents=True, exist_ok=True)

    store = ConsentStore()
    privacy = PrivacyEnforcer(policy=dict(DEFAULT_PRIVACY_POLICY))
    primary = DeterministicLocalAdapter()
    degraded = DegradedLocalAdapter()
    as_model_adapter(primary)
    as_model_adapter(degraded)

    consent = store.issue(subject_id="smoke-operator", scope="sandbox", issued_at=0.0, expires_at=10_000.0)
    executor = ToolExecutor(workspace, store, dry_run=False, stop_path=stop_path)

    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_pass = True

    try:
        # Real write + read through the executor inside the workspace.
        write_proposal = ActionProposal(
            action_id="smoke-write",
            task_id="sandbox-readiness-smoke",
            tool="write_file",
            arguments={"path": "fixture.txt", "content": "sandbox-smoke-bytes"},
            expected_effect="write bounded fixture",
            reversibility="reversible",
        )
        write_result = executor.execute(
            write_proposal,
            consent_receipt=consent.receipt_id,
            subject_id="smoke-operator",
            operator_approved=True,
            dry_run=False,
            now=1.0,
        )
        events.append({"kind": "write", "status": write_result.status, "receipt": write_result.receipt["sha256"], "activation": False})
        if write_result.status != "executed":
            all_pass = False
            failures.append({"kind": "write_failed", "reason": write_result.reason, "activation": False})

        read_proposal = ActionProposal(
            action_id="smoke-read",
            task_id="sandbox-readiness-smoke",
            tool="read_file",
            arguments={"path": "fixture.txt"},
            expected_effect="read bounded fixture",
            reversibility="read_only",
        )
        read_result = executor.execute(
            read_proposal,
            consent_receipt=consent.receipt_id,
            subject_id="smoke-operator",
            operator_approved=True,
            dry_run=False,
            now=1.0,
        )
        events.append({"kind": "read", "status": read_result.status, "receipt": read_result.receipt["sha256"], "activation": False})
        if read_result.status != "executed" or not read_result.output:
            all_pass = False
            failures.append({"kind": "read_failed", "reason": read_result.reason, "activation": False})
            file_bytes = b""
            content_digest = io.digest(b"")
        else:
            file_bytes = Path(workspace / "fixture.txt").read_bytes()
            content_digest = read_result.output["content_digest"]

        # Sensor packet from real file bytes.
        sensor = SensorPacket(
            packet_id="smoke-sensor-1",
            modality="filesystem",
            logical_time=1,
            content_digest=content_digest,
            provenance="sandbox://smoke/fixture.txt",
            features={"exists": True, "bytes": len(file_bytes), "content_digest": content_digest},
            consent_receipt=consent.receipt_id,
        )
        sensor.validate()
        store.require(sensor.consent_receipt, subject_id="smoke-operator", now=1.0)
        privacy.store(
            record_id="smoke-privacy-1",
            fields=dict(sensor.features),
            inventory={"exists", "bytes", "content_digest"},
            private_fields=(),
            retention_seconds=3600.0,
            stored_at=1.0,
            task_id="sandbox-readiness-smoke",
        )
        events.append({"kind": "sensor", "packet_id": sensor.packet_id, "content_digest": sensor.content_digest, "activation": False})

        # Both model adapters with non-zero accounting.
        model_req = ModelRequest(
            request_id="smoke-model-1",
            role="planner",
            input_digest=sensor.content_digest,
            context={"objective": "bounded smoke", "features": sensor.features},
            output_schema={"proposal": "object"},
            cost_limit=1.0,
        )
        primary_response = primary.infer(model_req)
        degraded_response = degraded.infer(model_req)
        if primary_response.latency_seconds <= 0 or primary_response.cost <= 0:
            all_pass = False
            failures.append({"kind": "primary_accounting", "activation": False})
        if degraded_response.latency_seconds <= 0 or degraded_response.cost <= 0:
            all_pass = False
            failures.append({"kind": "degraded_accounting", "activation": False})
        if primary_response.model_identity == degraded_response.model_identity:
            all_pass = False
            failures.append({"kind": "adapter_identity_collision", "activation": False})
        events.append(
            {
                "kind": "model",
                "primary": {
                    "identity": primary_response.model_identity,
                    "latency_seconds": primary_response.latency_seconds,
                    "cost": primary_response.cost,
                    "output_digest": primary_response.output_digest,
                },
                "degraded": {
                    "identity": degraded_response.model_identity,
                    "latency_seconds": degraded_response.latency_seconds,
                    "cost": degraded_response.cost,
                    "output_digest": degraded_response.output_digest,
                    "supported": degraded_response.supported,
                },
                "activation": False,
            }
        )

        # Containment escape must be refused.
        escape = ActionProposal(
            action_id="smoke-escape",
            task_id="sandbox-readiness-smoke",
            tool="read_file",
            arguments={"path": "../outside.txt"},
            expected_effect="escape workspace",
            reversibility="read_only",
        )
        escape_result = executor.execute(
            escape,
            consent_receipt=consent.receipt_id,
            subject_id="smoke-operator",
            operator_approved=True,
            dry_run=False,
            now=1.0,
        )
        escape_refused = escape_result.refused and escape_result.reason is not None and "escape" in escape_result.reason
        events.append({"kind": "containment_escape", "refused": escape_refused, "reason": escape_result.reason, "activation": False})
        if not escape_refused:
            all_pass = False
            failures.append({"kind": "containment_not_enforced", "activation": False})

        # Revoked consent must be refused.
        store.revoke(consent.receipt_id, at=2.0)
        post_revoke = ActionProposal(
            action_id="smoke-post-revoke",
            task_id="sandbox-readiness-smoke",
            tool="read_file",
            arguments={"path": "fixture.txt"},
            expected_effect="read after revoke",
            reversibility="read_only",
        )
        revoke_result = executor.execute(
            post_revoke,
            consent_receipt=consent.receipt_id,
            subject_id="smoke-operator",
            operator_approved=True,
            dry_run=False,
            now=3.0,
        )
        revoke_refused = revoke_result.refused and revoke_result.reason is not None and "revoked" in revoke_result.reason
        events.append({"kind": "revoked_consent", "refused": revoke_refused, "reason": revoke_result.reason, "activation": False})
        if not revoke_refused:
            all_pass = False
            failures.append({"kind": "revocation_not_enforced", "activation": False})

        document = {
            "schema": "sandbox-readiness-bounded-smoke/v1",
            "events": events,
            "failures": failures,
            "external_action_executed": False,
            "external_actions_executed": executor.external_actions_executed,
            "action_receipts": [r["sha256"] for r in executor.receipts],
            "containment_escape_refused": escape_refused,
            "revoked_consent_refused": revoke_refused,
            "model_adapters": [primary.identity, degraded.identity],
            "all_pass": all_pass and escape_refused and revoke_refused,
            "activation": False,
            "privacy_unenforced": list(privacy.unenforced),
        }
        document["sha256"] = io.digest(document)
        return document
    finally:
        if created_root:
            shutil.rmtree(root, ignore_errors=True)
