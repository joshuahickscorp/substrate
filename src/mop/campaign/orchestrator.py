"""The single durable MOP research campaign orchestrator.

One process owns future MOP campaign scheduling: it holds a singleton lock, runs the runnable frontier
under one global resource broker with one shared worker fleet, waits on external authority boundaries and
auto-activates queued work when they clear, throttles dynamically, delivers Telegram events, and writes an
atomic observable status. A second invocation adopts the first rather than starting a competitor.

It coexists with, and never signals, any live campaign (the running General Run and successor horizon
chain are adopted as external resource consumers). Cooperative drain is a stop file the loop honors; the
loop is otherwise idempotent and restart-safe (durable state plus lease recovery in the scheduler).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.process_labels import set_process_label

from .broker import ResourceBroker
from .dag import AuthorityResolver
from .executor import CampaignScheduler
from .manifest import build_campaign
from .specs import CampaignSpec
from .state import NodeStatus
from .status import build_status, render_text
from .telegram import record_delivery, send_campaign_event

ORCHESTRATOR_LABEL = "mop:research:orchestrator"
PROGRAM_ID = "mop-research"
# Native-thread pinning for pool workers, so process-count times BLAS-threads does not oversubscribe.
_THREAD_PIN = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _proc_alive_matching(pid: int, create_time: float) -> bool:
    try:
        import psutil  # type: ignore

        if not psutil.pid_exists(pid):
            return False
        return abs(psutil.Process(pid).create_time() - create_time) < 2.0
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


@dataclass
class SingletonLock:
    """A pidfile singleton lock binding pid, create-time, launch commit, and implementation identity."""

    path: Path
    launch_commit: str

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def live_owner(self) -> dict[str, Any] | None:
        payload = self.read()
        if payload and _proc_alive_matching(int(payload["pid"]), float(payload["create_time"])):
            return payload
        return None

    def acquire(self) -> tuple[bool, dict[str, Any] | None]:
        """Return (acquired, existing_owner). If a live owner holds the lock, do not acquire (adopt it)."""

        owner = self.live_owner()
        if owner is not None:
            return False, owner
        self.path.parent.mkdir(parents=True, exist_ok=True)
        create_time = 0.0
        try:
            import psutil  # type: ignore

            create_time = psutil.Process(os.getpid()).create_time()
        except Exception:
            create_time = time.time()
        payload = {
            "pid": os.getpid(),
            "create_time": create_time,
            "launch_commit": self.launch_commit,
            "program_id": PROGRAM_ID,
            "acquired_at": time.time(),
        }
        tmp = self.path.with_suffix(".lock.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        return True, payload

    def release(self) -> None:
        owner = self.read()
        if owner and int(owner.get("pid", -1)) == os.getpid():
            with contextlib.suppress(OSError):
                self.path.unlink()


def _git_head(repo: Path) -> str:
    try:
        import subprocess

        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


class ResearchOrchestrator:
    """The durable orchestrator. Owns the singleton lock, the broker, and the shared scheduler."""

    def __init__(
        self,
        run_root: str | Path = "runs/campaign/mop_research",
        op_root: str | Path = "proof/campaign_run",
        campaign: CampaignSpec | None = None,
        max_ceiling: int = 20,
    ) -> None:
        self.run_root = Path(run_root)
        self.op_root = Path(op_root)
        self.repo = Path(__file__).resolve().parents[3]
        self.launch_commit = _git_head(self.repo)
        self.campaign = campaign or build_campaign()
        self.lock = SingletonLock(self.run_root / "orchestrator.lock", self.launch_commit)
        self.broker = ResourceBroker(external_labels=("mop-final-mechanic", "mop-g1-", "general-run"))
        self.resolver = AuthorityResolver(external_checker=self._external_boundary)
        self.max_ceiling = max_ceiling
        self.stop_file = self.run_root / "STOP"

    # -- external authority boundaries (never signal live work) ------------

    def _external_boundary(self, name: str) -> bool:
        if name == "external:horizon-v2-complete":
            return self._horizon_complete()
        return False

    def _horizon_complete(self) -> bool:
        for rel in (
            "runs/generation1/generation1-successor-horizon-v2/program_state.json",
            "runs/generation1/generation1-successor-horizon-v2/current_status.json",
        ):
            path = self.repo / rel
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            state = data.get("state")
            if data.get("finished_at") not in (None, "") or state in ("complete", "finished", "drained"):
                return True
        return False

    # -- observability -----------------------------------------------------

    def write_status(self, scheduler: CampaignScheduler) -> dict[str, Any]:
        snap = self.broker.snapshot() or self.broker.sample()
        status = build_status(self.campaign, scheduler.state, snap)
        status["orchestrator"] = {
            "program_id": PROGRAM_ID,
            "label": ORCHESTRATOR_LABEL,
            "pid": os.getpid(),
            "launch_commit": self.launch_commit,
            "horizon_v2_complete": self._horizon_complete(),
        }
        self.op_root.mkdir(parents=True, exist_ok=True)
        tmp = self.op_root / "orchestrator_status.json.tmp"
        tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
        (tmp).replace(self.op_root / "orchestrator_status.json")
        return status

    def _pending_behind_boundary(self, scheduler: CampaignScheduler) -> list[str]:
        out = []
        for node in self.campaign.nodes:
            if scheduler.state.status(node.node_id) is NodeStatus.PENDING and not node.is_blocked:
                out.append(node.node_id)
        return out

    def observe(self) -> dict[str, Any]:
        """Read-only live view: the lock owner and the latest status, without acquiring anything."""

        owner = self.lock.live_owner()
        status_path = self.op_root / "orchestrator_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else None
        return {"live_owner": owner, "status": status}

    # -- the durable loop --------------------------------------------------

    def run(
        self, *, max_lifetime: float = 3600.0, tick_interval: float = 30.0, telegram: bool = True
    ) -> dict[str, Any]:
        """Run the durable orchestrator loop. Returns a terminal summary. Adopts if already live."""

        acquired, owner = self.lock.acquire()
        if not acquired:
            return {
                "adopted": True,
                "owner": owner,
                "note": "a live orchestrator already owns the singleton lock; this invocation adopts it",
            }

        for key, value in _THREAD_PIN.items():
            os.environ.setdefault(key, value)
        set_process_label(ORCHESTRATOR_LABEL)
        started = time.time()
        scheduler = CampaignScheduler(
            self.campaign, self.run_root, self.broker, self.resolver, max_ceiling=self.max_ceiling
        )

        if telegram:
            snap = self.broker.sample()
            receipt = send_campaign_event(
                "orchestrator launched",
                f"{PROGRAM_ID} live at {self.launch_commit[:12]}: {len(self.campaign.nodes)} nodes, "
                f"broker mode={snap.payload()['mode']} budget~{snap.cpu_budget}, "
                f"coexisting with {snap.external_consumers} live workers; horizon-v2 complete="
                f"{self._horizon_complete()}.",
                root=self.op_root,
                dedup_key=f"orch-launch-{self.launch_commit[:12]}",
            )
            record_delivery(self.op_root, receipt)

        last_mode: str | None = None
        try:
            while True:
                if self.stop_file.exists():
                    break
                # run the eligible frontier to quiescence this tick under the shared fleet
                scheduler.run(max_seconds=max(30.0, tick_interval), poll_seconds=0.3)
                status = self.write_status(scheduler)
                mode = (status.get("resource") or {}).get("mode")
                if telegram and mode and mode != last_mode:
                    rc = send_campaign_event(
                        "resource mode",
                        f"broker mode -> {mode}",
                        root=self.op_root,
                        dedup_key=f"mode-{mode}-{int((time.time() - started) // 300)}",
                    )
                    record_delivery(self.op_root, rc)
                    last_mode = mode
                # done iff nothing is pending/eligible/running that could still progress on its own
                pending = self._pending_behind_boundary(scheduler)
                counts = scheduler.state.counts()
                live = counts.get("eligible", 0) + counts.get("running", 0)
                if not pending and not live:
                    break
                if (time.time() - started) > max_lifetime:
                    break
                # throttle: sleep between ticks; pending nodes are waiting on an external boundary
                time.sleep(min(tick_interval, max(1.0, max_lifetime - (time.time() - started))))
            counts = scheduler.state.counts()
            summary: dict[str, Any] = {
                "adopted": False,
                "counts": counts,
                "elapsed_seconds": round(time.time() - started, 1),
                "pending_behind_boundary": self._pending_behind_boundary(scheduler),
                "horizon_v2_complete": self._horizon_complete(),
            }
            if telegram:
                n_sealed = counts.get("sealed", 0)
                rc = send_campaign_event(
                    "orchestrator quiescent",
                    render_text(self.write_status(scheduler)),
                    root=self.op_root,
                    dedup_key=f"orch-quiescent-{self.launch_commit[:12]}-{n_sealed}",
                )
                record_delivery(self.op_root, rc)
            return summary
        finally:
            self.lock.release()
