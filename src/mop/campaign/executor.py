"""Unified campaign engine: one shared executor that runs the runnable frontier concurrently.

This is the single campaign parent's worker fleet. It does NOT create a pool per bed. It samples the global
broker, submits every admitted eligible node to one shared ``ProcessPoolExecutor``, seals results, fires
precommitted decision rules, applies null-safe skipping, and persists durable state after every event so a
restart resumes exactly. Stale leases (dead worker or missed heartbeat) are recovered to eligible.

The parent process itself does little CPU work; its child workers do. Interactive responsiveness is kept
through the broker's nice level, not through chronic underutilization.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from .broker import ResourceBroker
from .dag import AuthorityResolver, refresh_eligibility, runnable_frontier
from .decisions import evaluate_decisions, resolve_skips
from .runners import NodeContext, resolve_entrypoint
from .specs import CampaignSpec, NodeSpec, ResourceRequest
from .state import CampaignState

_NULL_VERDICTS = {"null", "what_floor_collapse", "receipt_variant", "no_headroom_budget_saturated", "refused"}


def _derive_seed(campaign_id: str, node_id: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{campaign_id}|{node_id}".encode()).digest()[:4], "big")


def _execute_node(
    entrypoint: str, node_id: str, params: dict[str, Any], seed: int, root: str, nice_level: int
) -> dict[str, Any]:
    """Worker-process entry: run one node's runner and return a picklable result. Never raises to the pool."""

    try:
        if nice_level > 0:
            with contextlib.suppress(OSError):
                os.nice(nice_level)
        runner = resolve_entrypoint(entrypoint)
        workdir = Path(root) / "nodes" / node_id
        workdir.mkdir(parents=True, exist_ok=True)
        ctx = NodeContext(node_id=node_id, workdir=workdir, seed=seed, proof_root=Path(root) / "proof")
        result = runner(params, ctx)
        return {
            "ok": True,
            "node_id": node_id,
            "artifact_path": result.artifact_path,
            "seal": result.seal,
            "verdict": result.verdict,
            "is_null": bool(result.is_null),
            "detail": dict(result.detail),
        }
    except Exception as exc:  # noqa: BLE001 (a node failure must not kill the fleet)
        return {"ok": False, "node_id": node_id, "error": f"{type(exc).__name__}: {exc}"}


class CampaignScheduler:
    """Runs a campaign to quiescence under one shared executor and global broker."""

    def __init__(
        self,
        campaign: CampaignSpec,
        root: str | Path,
        broker: ResourceBroker,
        resolver: AuthorityResolver | None = None,
        max_ceiling: int = 20,
    ) -> None:
        self.campaign = campaign
        self.root = Path(root)
        self.broker = broker
        self.resolver = resolver or AuthorityResolver()
        self.max_ceiling = max_ceiling
        self.state = CampaignState(self.root, campaign.campaign_id, [n.node_id for n in campaign.nodes])

    def _request_of(self, node: NodeSpec) -> ResourceRequest:
        return node.resources

    def run(self, max_seconds: float | None = None, poll_seconds: float = 0.5) -> dict[str, Any]:
        """Drive the frontier until nothing else can progress, or a time budget elapses. Return a summary."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.state.load()
        recovered = self.state.recover_stale_leases()
        self.state.save()
        started = time.time()

        running: dict[Future[dict[str, Any]], tuple[str, ResourceRequest]] = {}
        with ProcessPoolExecutor(max_workers=self.max_ceiling) as pool:
            own = {p.pid for p in (pool._processes or {}).values() if p.pid is not None}
            self.broker.set_own_pids(own)
            while True:
                # 1. bookkeeping: recover leases, refresh eligibility, prune dead branches
                self.state.recover_stale_leases()
                refresh_eligibility(self.campaign, self.state, self.resolver)
                resolve_skips(self.campaign, self.state)

                # 2. sample the broker and admit as many eligible nodes as the budget allows
                self.broker.sample(current_workers=len(running))
                frontier = runnable_frontier(self.campaign, self.state, self.resolver)
                frontier.sort(key=lambda n: (n.priority, n.coverage.evidence_level, n.node_id))
                for node in frontier:
                    running_reqs = [req for (_nid, req) in running.values()]
                    grant = self.broker.admit(self._request_of(node), node.node_id, running_reqs)
                    if grant is None:
                        continue
                    seed = _derive_seed(self.campaign.campaign_id, node.node_id)
                    fut = pool.submit(
                        _execute_node,
                        node.entrypoint,
                        node.node_id,
                        node.params,
                        seed,
                        str(self.root),
                        grant.nice_level,
                    )
                    self.state.mark_running(node.node_id, os.getpid(), started)
                    running[fut] = (node.node_id, node.resources)
                self.state.save()

                # 3. stop conditions: quiescent when nothing running and nothing admitted this pass
                if not running and not runnable_frontier(self.campaign, self.state, self.resolver):
                    break
                if max_seconds is not None and (time.time() - started) > max_seconds:
                    break

                # 4. wait for at least one node to finish, then process it
                if running:
                    done, _pending = wait(list(running), timeout=poll_seconds, return_when=FIRST_COMPLETED)
                    for fut in done:
                        node_id, _req = running.pop(fut)
                        self._handle_result(fut.result())
                    self.state.save()

        summary = {
            "campaign_id": self.campaign.campaign_id,
            "recovered_leases": recovered,
            "counts": self.state.counts(),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        return summary

    def _handle_result(self, result: dict[str, Any]) -> None:
        node_id = result["node_id"]
        if not result.get("ok"):
            self.state.mark_failed(node_id, result.get("error", "unknown"))
            return
        verdict = result.get("verdict")
        is_null = bool(result.get("is_null")) or (verdict in _NULL_VERDICTS)
        self.state.mark_sealed(
            node_id, result["artifact_path"], result["seal"], verdict, is_null, detail=result.get("detail")
        )
        node = self.campaign.node(node_id)
        try:
            artifact = self._read_artifact(result["artifact_path"])
        except Exception:
            artifact = {"verdict": verdict}
        evaluate_decisions(self.campaign, node, {**artifact, "verdict": verdict}, self.state)

    @staticmethod
    def _read_artifact(path: str) -> dict[str, Any]:
        import json

        return json.loads(Path(path).read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        return {
            "counts": self.state.counts(),
            "records": {nid: rec.payload() for nid, rec in self.state.records.items()},
        }
