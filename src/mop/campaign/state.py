"""Unified campaign engine: durable campaign state, node leases, stale-lease recovery, and adoption.

The scheduler is operational infrastructure, so it legitimately uses real wall time for leases and
heartbeats. This does not touch scientific determinism: sealed artifacts remain clock-free and
byte-reproducible; only the operational bookkeeping here reads the clock.

State is a single JSON file under the campaign run root. On restart it is re-read and each node resumes at
its exact recorded status. A node marked RUNNING whose lease process is dead (or whose heartbeat is stale)
is recovered to ELIGIBLE so the frontier can re-launch it, exactly once accounting for the failed attempt.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

STATE_SCHEMA = "mop-campaign-state/v1"
STALE_LEASE_SECONDS = 900.0  # a RUNNING lease with no heartbeat this long, or a dead pid, is recoverable


class NodeStatus(StrEnum):
    PENDING = "pending"  # dependencies/authorities not yet satisfied
    ELIGIBLE = "eligible"  # runnable frontier; awaiting a broker slot
    RUNNING = "running"  # leased to a worker
    SEALED = "sealed"  # terminal, produced a sealed artifact (positive-or-null both seal)
    NULL_SEALED = "null_sealed"  # terminal, sealed as a null (still a completion)
    FAILED = "failed"  # terminal after exhausting attempts
    BLOCKED = "blocked"  # contracted but blocked on a named external input
    SKIPPED = "skipped"  # pruned by a precommitted null-safe decision


TERMINAL = {NodeStatus.SEALED, NodeStatus.NULL_SEALED, NodeStatus.FAILED, NodeStatus.SKIPPED}
COMPLETED = {NodeStatus.SEALED, NodeStatus.NULL_SEALED}


@dataclass
class Lease:
    pid: int
    create_time: float
    started_at: float
    heartbeat_at: float

    def payload(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "create_time": round(self.create_time, 6),
            "started_at": round(self.started_at, 6),
            "heartbeat_at": round(self.heartbeat_at, 6),
        }

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> Lease:
        return cls(int(d["pid"]), float(d["create_time"]), float(d["started_at"]), float(d["heartbeat_at"]))


@dataclass
class NodeRecord:
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    artifact_path: str | None = None
    seal_sha256: str | None = None
    verdict: str | None = (
        None  # e.g. survives / null / architecture_fragile / real_headroom / what_floor_collapse
    )
    lease: Lease | None = None
    enqueued_by: str | None = None  # a decision rule that made this node eligible
    updated_at: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "artifact_path": self.artifact_path,
            "seal_sha256": self.seal_sha256,
            "verdict": self.verdict,
            "lease": self.lease.payload() if self.lease else None,
            "enqueued_by": self.enqueued_by,
            "updated_at": round(self.updated_at, 6),
            "detail": self.detail,
        }

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> NodeRecord:
        return cls(
            node_id=d["node_id"],
            status=NodeStatus(d["status"]),
            attempts=int(d.get("attempts", 0)),
            artifact_path=d.get("artifact_path"),
            seal_sha256=d.get("seal_sha256"),
            verdict=d.get("verdict"),
            lease=Lease.from_payload(d["lease"]) if d.get("lease") else None,
            enqueued_by=d.get("enqueued_by"),
            updated_at=float(d.get("updated_at", 0.0)),
            detail=dict(d.get("detail", {})),
        )


def _pid_alive_with_create_time(pid: int, create_time: float) -> bool:
    """True if pid is alive and (best effort) matches the recorded create time, so we never adopt a reused
    pid. Falls back to a plain liveness check when psutil is unavailable."""

    try:
        import psutil  # type: ignore

        if not psutil.pid_exists(pid):
            return False
        try:
            proc = psutil.Process(pid)
            return abs(proc.create_time() - create_time) < 2.0
        except Exception:
            return False
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class CampaignState:
    """The durable, restart-safe state of one campaign run."""

    def __init__(self, root: str | Path, campaign_id: str, node_ids: list[str]) -> None:
        self.root = Path(root)
        self.campaign_id = campaign_id
        self.path = self.root / "state.json"
        self.records: dict[str, NodeRecord] = {nid: NodeRecord(nid) for nid in node_ids}
        self.created_at = time.time()
        self.fired_decisions: set[str] = set()  # "branch:<parent>:<rule>" keys of precommitted rules fired

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.created_at = float(data.get("created_at", self.created_at))
        self.fired_decisions = set(data.get("fired_decisions", []))
        for nid, rec in data.get("records", {}).items():
            if nid in self.records:
                self.records[nid] = NodeRecord.from_payload(rec)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        body = {
            "schema": STATE_SCHEMA,
            "campaign_id": self.campaign_id,
            "created_at": round(self.created_at, 6),
            "updated_at": round(time.time(), 6),
            "fired_decisions": sorted(self.fired_decisions),
            "records": {nid: rec.payload() for nid, rec in self.records.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    # -- lease recovery ----------------------------------------------------

    def recover_stale_leases(self) -> list[str]:
        """Return RUNNING nodes whose worker is dead or whose heartbeat is stale to ELIGIBLE. Idempotent."""

        recovered: list[str] = []
        now = time.time()
        for rec in self.records.values():
            if rec.status is not NodeStatus.RUNNING or rec.lease is None:
                continue
            dead = not _pid_alive_with_create_time(rec.lease.pid, rec.lease.create_time)
            stale = (now - rec.lease.heartbeat_at) > STALE_LEASE_SECONDS
            if dead or stale:
                rec.status = NodeStatus.ELIGIBLE
                rec.lease = None
                rec.updated_at = now
                recovered.append(rec.node_id)
        return recovered

    # -- transitions -------------------------------------------------------

    def mark_running(self, node_id: str, pid: int, create_time: float) -> None:
        now = time.time()
        rec = self.records[node_id]
        rec.status = NodeStatus.RUNNING
        rec.attempts += 1
        rec.lease = Lease(pid=pid, create_time=create_time, started_at=now, heartbeat_at=now)
        rec.updated_at = now

    def heartbeat(self, node_id: str) -> None:
        rec = self.records[node_id]
        if rec.lease is not None:
            rec.lease.heartbeat_at = time.time()
            rec.updated_at = rec.lease.heartbeat_at

    def mark_sealed(
        self,
        node_id: str,
        artifact_path: str,
        seal: str,
        verdict: str | None,
        is_null: bool,
        detail: dict[str, Any] | None = None,
    ) -> None:
        rec = self.records[node_id]
        rec.status = NodeStatus.NULL_SEALED if is_null else NodeStatus.SEALED
        rec.artifact_path = artifact_path
        rec.seal_sha256 = seal
        rec.verdict = verdict
        rec.lease = None
        rec.updated_at = time.time()
        if detail:
            rec.detail.update(detail)

    def mark_failed(self, node_id: str, reason: str) -> None:
        rec = self.records[node_id]
        rec.status = NodeStatus.FAILED
        rec.lease = None
        rec.detail["failure"] = reason
        rec.updated_at = time.time()

    def mark_blocked(self, node_id: str, reason: str) -> None:
        rec = self.records[node_id]
        rec.status = NodeStatus.BLOCKED
        rec.detail["blocked_reason"] = reason
        rec.updated_at = time.time()

    def mark_skipped(self, node_id: str, reason: str) -> None:
        rec = self.records[node_id]
        rec.status = NodeStatus.SKIPPED
        rec.detail["skipped_reason"] = reason
        rec.updated_at = time.time()

    def set_eligible(self, node_id: str, enqueued_by: str | None = None) -> None:
        rec = self.records[node_id]
        if rec.status is NodeStatus.PENDING:
            rec.status = NodeStatus.ELIGIBLE
            rec.enqueued_by = enqueued_by
            rec.updated_at = time.time()

    # -- queries -----------------------------------------------------------

    def record_decision(self, key: str) -> None:
        self.fired_decisions.add(key)

    def decision_fired(self, key: str) -> bool:
        return key in self.fired_decisions

    def status(self, node_id: str) -> NodeStatus:
        return self.records[node_id].status

    def is_completed(self, node_id: str) -> bool:
        return self.records[node_id].status in COMPLETED

    def is_terminal(self, node_id: str) -> bool:
        """Terminal for ordering: sealed, null-sealed, failed, or skipped. A COMPLETION dependency is a
        run-after ordering constraint satisfied by any terminal state; a SEAL dependency needs an artifact."""

        return self.records[node_id].status in TERMINAL

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records.values():
            out[rec.status.value] = out.get(rec.status.value, 0) + 1
        return out
