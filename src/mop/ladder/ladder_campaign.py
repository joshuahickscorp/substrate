"""Chained autonomous ladder campaign: climbs Stage 0 to 5 under the dynamic throttler.

This is the "new and improved" detached run. It reuses the disciplined patterns of the sealed
Generation 1 supervisor (sealed receipts, fail-closed stage gates, self-sealed state, a run that can be
detached) but drives admission with the self-managing DynamicThrottleController instead of the static
one-heavy-lane policy. That lets it MAXIMIZE CPU and memory utilization around whatever else the host is
doing, for example an external quantize job, without ever crossing an out-of-memory floor: it scales
worker concurrency up when there is headroom and sheds its OWN newest workers when there is not. It
never signals any process it did not spawn.

Stage 0 to 2 are recorded as satisfied by the constitution and the competence census. Stage 3 runs the
nine mechanism demonstrations as isolated worker subprocesses across a seed sweep, throttled. Stage 4
and Stage 5 stay fail-closed until their entry receipts exist, which the honest null demonstrations do
not mint. The campaign writes a sealed ladder report.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..substrate.events import canonical_sha256
from .dynamic_throttle import DynamicThrottleController, ThrottlePolicy, read_host_sample
from .stage3_demonstrations import STAGE3_EPOCHS

SCHEMA = "mop-ladder-campaign/v1"
REPORT_SCHEMA = "mop-ladder-campaign-report/v1"
STATE_SCHEMA = "mop-ladder-campaign-state/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
WORKER_MODULE = "mop.ladder.ladder_worker"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROGRAM_ROOT = REPO_ROOT / "runs" / "ladder_campaign" / "stage0_to_5_v1"
CENSUS_STATE = (
    REPO_ROOT / "runs" / "generation1" / "generation1-empirical-cognitive-corpus-v2" / "program_state.json"
)


class CampaignRefusal(RuntimeError):
    """Raised when a campaign configuration or state is malformed. Fails closed."""


def census_complete(state_path: Path = CENSUS_STATE) -> bool:
    """True if the competence census has finished (or is absent, so there is nothing to wait for)."""

    if not state_path.is_file():
        return True
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("status") == "complete":
        return True
    capsules = state.get("capsules", [])
    return bool(capsules) and all(
        isinstance(row, dict) and row.get("status") == "complete" for row in capsules
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sealed(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    core = {key: value for key, value in payload.items() if key != field_name}
    return {**core, field_name: canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Immutable campaign parameters."""

    program_root: Path = DEFAULT_PROGRAM_ROOT
    seeds: tuple[int, ...] = tuple(range(24))
    reps: int = 24
    poll_interval_s: float = 0.4
    per_worker_peak_gb: float = 0.5
    max_workers: int | None = None
    worker_python: str = sys.executable
    epochs: tuple[str, ...] = STAGE3_EPOCHS
    wait_for_census: bool = False
    census_poll_s: float = 60.0
    census_max_wait_s: float = 43200.0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise CampaignRefusal("campaign needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise CampaignRefusal("campaign seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise CampaignRefusal("campaign seeds must be nonnegative")
        if self.reps < 1:
            raise CampaignRefusal("campaign reps must be at least 1")
        if self.poll_interval_s <= 0:
            raise CampaignRefusal("poll interval must be positive")
        if self.per_worker_peak_gb <= 0:
            raise CampaignRefusal("per-worker peak estimate must be positive")
        if not self.epochs:
            raise CampaignRefusal("campaign needs at least one epoch")

    @property
    def state_path(self) -> Path:
        return self.program_root / "campaign_state.json"

    @property
    def report_path(self) -> Path:
        return self.program_root / "ladder_report.json"

    @property
    def pid_path(self) -> Path:
        return self.program_root / "campaign.pid"

    @property
    def log_path(self) -> Path:
        return self.program_root / "campaign.log"


@dataclass(frozen=True, slots=True)
class WorkItem:
    epoch: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.epoch}.seed{self.seed}"


@dataclass
class WorkerHandle:
    item: WorkItem
    process: subprocess.Popen[bytes]
    order: int
    receipt_path: Path


class LadderCampaign:
    """Throttler-driven, stage-chained, self-sealing campaign runner."""

    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        self._start_ns = time.monotonic_ns()

    # -- planning -----------------------------------------------------------------

    def plan_stage3(self) -> list[WorkItem]:
        return [WorkItem(epoch, seed) for epoch in self.config.epochs for seed in self.config.seeds]

    def _build_policy(self) -> ThrottlePolicy:
        sample = read_host_sample(worker_pids=[])
        return ThrottlePolicy.autoscaled(
            sample,
            per_worker_peak_gb=self.config.per_worker_peak_gb,
            max_workers=self.config.max_workers,
        )

    # -- worker lifecycle ---------------------------------------------------------

    def _receipt_path(self, item: WorkItem) -> Path:
        return self.config.program_root / "stage3_receipts" / f"{item.label}.json"

    def _spawn(self, item: WorkItem, order: int) -> WorkerHandle:
        receipt = self._receipt_path(item)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        args = [
            self.config.worker_python,
            "-m",
            WORKER_MODULE,
            "--epoch",
            item.epoch,
            "--seed",
            str(item.seed),
            "--reps",
            str(self.config.reps),
            "--out",
            str(receipt),
        ]
        process = subprocess.Popen(
            args,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return WorkerHandle(item=item, process=process, order=order, receipt_path=receipt)

    def _collect(self, handle: WorkerHandle, exit_code: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "epoch": handle.item.epoch,
            "seed": handle.item.seed,
            "exit_code": exit_code,
        }
        if exit_code == 0 and handle.receipt_path.is_file():
            try:
                receipt = json.loads(handle.receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                record["ok"] = False
                record["error"] = f"receipt unreadable: {exc}"
                return record
            declared = receipt.get("receipt_sha256")
            core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            record["ok"] = declared == canonical_sha256(core)
            record["verdict_status"] = receipt.get("verdict_status")
            record["null_holds"] = receipt.get("null_holds")
            record["controls_ok"] = receipt.get("controls_ok")
            if not record["ok"]:
                record["error"] = "receipt seal mismatch"
        else:
            record["ok"] = False
            record["error"] = "worker failed or produced no receipt"
        return record

    def _shed(self, running: list[WorkerHandle], count: int) -> list[WorkItem]:
        """Terminate our own newest workers and return their items for requeue. Never touches others."""

        if count <= 0 or not running:
            return []
        running.sort(key=lambda handle: handle.order)
        victims = running[-count:]
        del running[-count:]
        requeued: list[WorkItem] = []
        for handle in victims:
            with contextlib.suppress(ProcessLookupError):
                handle.process.terminate()
            requeued.append(handle.item)
        return requeued

    # -- stage 3 pool -------------------------------------------------------------

    def run_stage3(self) -> dict[str, Any]:
        queue = self.plan_stage3()
        total = len(queue)
        controller = DynamicThrottleController(self._build_policy())
        running: list[WorkerHandle] = []
        completed: list[dict[str, Any]] = []
        order = 0
        peak_concurrency = 0
        shed_total = 0

        while queue or running:
            still: list[WorkerHandle] = []
            for handle in running:
                code = handle.process.poll()
                if code is None:
                    still.append(handle)
                else:
                    completed.append(self._collect(handle, code))
            running = still

            sample = read_host_sample(worker_pids=[handle.process.pid for handle in running])
            decision = controller.decide(sample, running=len(running))

            if decision.must_shed > 0:
                requeued = self._shed(running, decision.must_shed)
                shed_total += len(requeued)
                queue[0:0] = requeued

            if decision.admit and queue:
                item = queue.pop(0)
                running.append(self._spawn(item, order))
                order += 1
                peak_concurrency = max(peak_concurrency, len(running))
                continue

            time.sleep(self.config.poll_interval_s)

        ok = [row for row in completed if row.get("ok")]
        by_epoch: dict[str, dict[str, Any]] = {}
        for epoch in self.config.epochs:
            rows = [row for row in ok if row["epoch"] == epoch]
            by_epoch[epoch] = {
                "seeds_ok": len(rows),
                "seeds_total": len(self.config.seeds),
                "null_holds_all": bool(rows) and all(row.get("null_holds") is True for row in rows),
                "controls_ok_all": bool(rows) and all(row.get("controls_ok") is True for row in rows),
                "verdicts": sorted({str(row.get("verdict_status")) for row in rows}),
            }
        return {
            "total_work": total,
            "completed": len(completed),
            "ok": len(ok),
            "failed": len(completed) - len(ok),
            "peak_concurrency": peak_concurrency,
            "shed_total": shed_total,
            "by_epoch": by_epoch,
        }

    # -- stage records ------------------------------------------------------------

    def _stage0(self) -> dict[str, Any]:
        return {
            "stage": 0,
            "name": "governance, measurement, falsification, recovery",
            "status": "complete",
            "basis": "constitution and Generation 0 verified nulls carry forward",
        }

    def _stage12(self) -> dict[str, Any]:
        census_root = REPO_ROOT / "runs" / "generation1" / "generation1-empirical-cognitive-corpus-v2"
        census_state = census_root / "program_state.json"
        detail: dict[str, Any] = {
            "census_program_root": str(census_root),
            "census_present": census_state.is_file(),
        }
        if census_state.is_file():
            try:
                state = json.loads(census_state.read_text(encoding="utf-8"))
                capsules = state.get("capsules", [])
                detail["census_complete"] = sum(
                    1 for row in capsules if isinstance(row, dict) and row.get("status") == "complete"
                )
                detail["census_total"] = len(capsules)
                detail["census_status"] = state.get("status")
            except (OSError, json.JSONDecodeError):
                detail["census_present"] = False
        return {
            "stage": "1-2",
            "name": "programmable mechanics and counterfactual ecology",
            "status": "complete",
            "basis": "the competence census is the Stage 1 to 2 empirical layer",
            "detail": detail,
        }

    def _stage45(self, stage3: dict[str, Any]) -> dict[str, Any]:
        confirmed = sum(
            1
            for epoch in stage3.get("by_epoch", {}).values()
            if epoch.get("null_holds_all") is False and epoch.get("seeds_ok", 0) > 0
        )
        return {
            "stage4": {
                "name": "integrated architecture advantage",
                "status": "not entered",
                "reason": (
                    f"entry needs at least 2 confirmed Stage 3 mechanisms; {confirmed} confirmed. "
                    "the honest null demonstrations mint no confirmation receipt"
                ),
                "confirmed_mechanisms": confirmed,
            },
            "stage5": {
                "name": "natural, session-disjoint general validity",
                "status": "not entered",
                "reason": "entry needs Stage 4 plus measured session-disjoint validity across every axis",
            },
        }

    # -- run ----------------------------------------------------------------------

    def _write_state(self, status: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "schema": STATE_SCHEMA,
            "status": status,
            "program_root": str(self.config.program_root),
            "pid": os.getpid(),
            "seeds": list(self.config.seeds),
            "reps": self.config.reps,
            "epochs": list(self.config.epochs),
        }
        if extra:
            payload.update(extra)
        _atomic_write_json(self.config.state_path, _sealed(payload, "state_sha256"))

    def _wait_for_census(self) -> None:
        """Idle politely (a sleep loop, negligible resource) until the census finishes or we time out."""

        waited = 0.0
        while not census_complete() and waited < self.config.census_max_wait_s:
            self._write_state("waiting_for_census", {"waited_seconds": round(waited, 1)})
            time.sleep(self.config.census_poll_s)
            waited += self.config.census_poll_s

    def run(self) -> dict[str, Any]:
        self.config.program_root.mkdir(parents=True, exist_ok=True)
        if self.config.wait_for_census:
            self._wait_for_census()
        self._write_state("running")
        stage0 = self._stage0()
        stage12 = self._stage12()
        stage3 = self.run_stage3()
        stage45 = self._stage45(stage3)
        elapsed_s = (time.monotonic_ns() - self._start_ns) / 1e9
        report = {
            "schema": REPORT_SCHEMA,
            "claim_scope": CLAIM_SCOPE,
            "program_root": str(self.config.program_root),
            "elapsed_seconds": round(elapsed_s, 3),
            "seeds": list(self.config.seeds),
            "reps": self.config.reps,
            "stage0": stage0,
            "stage1_2": stage12,
            "stage3": stage3,
            "stage4_5": stage45,
            "ladder_position": "Stage 2 of 5 (Stage 3 demonstrations run; activation not earned)",
        }
        sealed = _sealed(report, "report_sha256")
        _atomic_write_json(self.config.report_path, sealed)
        self._write_state("complete", {"report_sha256": sealed["report_sha256"]})
        return sealed


# -- CLI ---------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    return CampaignConfig(
        program_root=Path(args.program_root),
        seeds=tuple(range(args.seeds)),
        reps=args.reps,
        per_worker_peak_gb=args.per_worker_gb,
        max_workers=args.max_workers,
        wait_for_census=args.wait_for_census,
    )


def _start_detached(config: CampaignConfig) -> dict[str, Any]:
    config.program_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    log = config.log_path.open("ab")
    args = [
        config.worker_python,
        "-m",
        "mop.ladder.ladder_campaign",
        "run",
        "--program-root",
        str(config.program_root),
        "--seeds",
        str(len(config.seeds)),
        "--reps",
        str(config.reps),
        "--per-worker-gb",
        str(config.per_worker_peak_gb),
    ]
    if config.max_workers is not None:
        args += ["--max-workers", str(config.max_workers)]
    if config.wait_for_census:
        args.append("--wait-for-census")
    process = subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    config.pid_path.write_text(str(process.pid), encoding="utf-8")
    return {"launched": True, "pid": process.pid, "program_root": str(config.program_root)}


def _status(config: CampaignConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        return {"state": "absent"}
    try:
        return json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "unreadable", "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chained autonomous ladder campaign (Stage 0 to 5).")
    parser.add_argument("verb", choices=["run", "start", "status", "validate"])
    parser.add_argument("--program-root", default=str(DEFAULT_PROGRAM_ROOT))
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--per-worker-gb", type=float, default=0.5)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--wait-for-census", action="store_true")
    args = parser.parse_args(argv)
    config = _config_from_args(args)

    if args.verb == "validate":
        print(json.dumps({"ok": True, "program_root": str(config.program_root), "work_items": len(
            LadderCampaign(config).plan_stage3()
        )}))
        return 0
    if args.verb == "status":
        print(json.dumps(_status(config), indent=2))
        return 0
    if args.verb == "start":
        print(json.dumps(_start_detached(config)))
        return 0
    report = LadderCampaign(config).run()
    print(json.dumps({"complete": True, "report_sha256": report["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
