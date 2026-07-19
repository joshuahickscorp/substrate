"""Prelaunch chaos gauntlet: adversarially prove the orchestrator, singleton lock, and broker.

Every scenario uses a disposable run root or a receipt-invariant fixture and touches no live work. Prints
PASS/FAIL per scenario and exits nonzero if any fails. This is a bounded launch gate, not a research detour.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from mop.campaign.broker import BrokerSnapshot, ResourceBroker
from mop.campaign.dag import AuthorityResolver
from mop.campaign.executor import CampaignScheduler
from mop.campaign.orchestrator import SingletonLock
from mop.campaign.specs import (
    BedSpec,
    CampaignSpec,
    Dependency,
    SpecError,
)
from mop.campaign.state import CampaignState, Lease, NodeStatus
from mop.substrate.events import canonical_sha256

_OK = "mop.campaign.nodes.gauntlet_fixtures:ok_runner"
_FAIL = "mop.campaign.nodes.gauntlet_fixtures:failing_runner"
_COUNT = "mop.campaign.nodes.gauntlet_fixtures:counting_runner"

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _campaign(entrypoint: str, ids: list[str]) -> CampaignSpec:
    nodes = [BedSpec(node_id=i, title=i, entrypoint=entrypoint) for i in ids]
    return CampaignSpec(campaign_id="gauntlet", title="gauntlet", nodes=tuple(nodes))


def _broker() -> ResourceBroker:
    b = ResourceBroker()
    b._snapshot = BrokerSnapshot(
        cpu_budget=8,
        cpu_in_use=0,
        mem_available_gb=100.0,
        mem_in_use_gb=0.0,
        hawking_active=False,
        nice_level=5,
        external_consumers=0,
        recommended_workers=8,
        binding_constraint="ceiling",
        created_at=0.0,
    )
    return b


# ---------------------------------------------------------------------------


def scenario_singleton_and_adoption() -> None:
    print("scenario: double launch, singleton and adoption")
    with tempfile.TemporaryDirectory() as d:
        lock = SingletonLock(Path(d) / "orchestrator.lock", "commitA")
        acquired, owner = lock.acquire()  # current pid is alive -> acquires
        check("first launch acquires the singleton lock", acquired and owner is not None)
        lock2 = SingletonLock(Path(d) / "orchestrator.lock", "commitA")
        acquired2, owner2 = lock2.acquire()  # live owner present -> adopts
        check(
            "second launch adopts (does not acquire a competitor)",
            (not acquired2) and owner2 is not None and int(owner2["pid"]) == owner["pid"],
        )
        # a stale lock (dead pid) must NOT be adopted
        stale = json.loads((Path(d) / "orchestrator.lock").read_text())
        stale["pid"] = 999999
        stale["create_time"] = 1.0
        (Path(d) / "orchestrator.lock").write_text(json.dumps(stale))
        lock3 = SingletonLock(Path(d) / "orchestrator.lock", "commitA")
        acquired3, _ = lock3.acquire()
        check("stale lock (dead pid) is not adopted; a fresh owner acquires", acquired3)


def scenario_crash_restart_stale_lease() -> None:
    print("scenario: parent crash, exact restart, stale-lease recovery, no double execution")
    with tempfile.TemporaryDirectory() as d:
        camp = _campaign(_COUNT, ["a", "b", "c"])
        counter = Path(d) / "exec_count.txt"
        CampaignScheduler(camp, d, _broker(), AuthorityResolver()).run(max_seconds=60)
        first = counter.read_text().splitlines() if counter.exists() else []
        check(
            "clean run seals every node exactly once",
            sorted(first) == ["a", "b", "c"],
            f"exec={sorted(first)}",
        )
        # simulate a crash: node c was RUNNING with a now-dead worker lease
        st = CampaignState(d, camp.campaign_id, [n.node_id for n in camp.nodes])
        st.load()
        st.records["c"].status = NodeStatus.RUNNING
        st.records["c"].lease = Lease(pid=999999, create_time=1.0, started_at=1.0, heartbeat_at=1.0)
        st.records["c"].artifact_path = None
        st.records["c"].seal_sha256 = None
        st.save()
        CampaignScheduler(camp, d, _broker(), AuthorityResolver()).run(max_seconds=60)
        after = counter.read_text().splitlines()
        # a and b sealed -> not recomputed (still once each); c was incomplete -> recovered and re-run (twice)
        check(
            "sealed nodes are not recomputed on restart",
            after.count("a") == 1 and after.count("b") == 1,
            f"a={after.count('a')} b={after.count('b')}",
        )
        check(
            "incomplete node is recovered from stale lease and completes",
            after.count("c") == 2,
            f"c ran {after.count('c')} times (1 original + 1 recovery)",
        )
        st2 = CampaignState(d, camp.campaign_id, [n.node_id for n in camp.nodes])
        st2.load()
        check(
            "all nodes terminal after restart, no deadlock",
            all(st2.status(n).value in ("sealed", "null_sealed") for n in ["a", "b", "c"]),
        )


def scenario_worker_failure() -> None:
    print("scenario: worker failure isolation")
    with tempfile.TemporaryDirectory() as d:
        nodes = [
            BedSpec(node_id="ok1", title="ok1", entrypoint=_OK),
            BedSpec(node_id="boom", title="boom", entrypoint=_FAIL),
            BedSpec(node_id="ok2", title="ok2", entrypoint=_OK),
        ]
        camp = CampaignSpec(campaign_id="wf", title="wf", nodes=tuple(nodes))
        CampaignScheduler(camp, d, _broker(), AuthorityResolver()).run(max_seconds=60)
        st = CampaignState(d, camp.campaign_id, [n.node_id for n in camp.nodes])
        st.load()
        check("failing node is marked failed, not sealed", st.status("boom") is NodeStatus.FAILED)
        check(
            "unrelated nodes still complete",
            st.status("ok1") is NodeStatus.SEALED and st.status("ok2") is NodeStatus.SEALED,
        )
        check(
            "no partial artifact accepted for the failed node", not (Path(d) / "proof" / "boom.json").exists()
        )


def scenario_mutation_fail_closed() -> None:
    print("scenario: state and authority mutation fail closed")
    # manifest/spec mutation: a dangling dependency must be rejected at construction
    try:
        CampaignSpec(
            campaign_id="x",
            title="x",
            nodes=(BedSpec(node_id="a", title="a", entrypoint=_OK, dependencies=(Dependency("ghost"),)),),
        )
        check("dangling dependency rejected", False, "no SpecError raised")
    except SpecError:
        check("dangling dependency rejected at construction", True)
    # seal mutation: a forged artifact number must not reproduce its seal
    content = {"schema": "x", "value": 10}
    seal = canonical_sha256(content)
    forged = {**content, "value": 11}
    check("forged artifact fails its seal", canonical_sha256(forged) != seal)
    # lock identity mutation: a lock whose create_time no longer matches the process is not a live owner
    with tempfile.TemporaryDirectory() as d:
        lock = SingletonLock(Path(d) / "orchestrator.lock", "c")
        lock.acquire()
        payload = json.loads((Path(d) / "orchestrator.lock").read_text())
        payload["create_time"] = payload["create_time"] + 12345.0  # tamper with identity
        (Path(d) / "orchestrator.lock").write_text(json.dumps(payload))
        check(
            "tampered lock identity is not treated as a live owner",
            SingletonLock(Path(d) / "orchestrator.lock", "c").live_owner() is None,
        )


def scenario_scheduling_invariance() -> None:
    print("scenario: scheduling order and pool width invariance")
    seals: list[dict[str, str]] = []
    for ceiling in (1, 4, 8):
        with tempfile.TemporaryDirectory() as d:
            camp = _campaign(_OK, ["n0", "n1", "n2", "n3", "n4"])
            CampaignScheduler(camp, d, _broker(), AuthorityResolver(), max_ceiling=ceiling).run(
                max_seconds=60
            )
            st = CampaignState(d, camp.campaign_id, [n.node_id for n in camp.nodes])
            st.load()
            seals.append({nid: st.records[nid].seal_sha256 or "" for nid in ["n0", "n1", "n2", "n3", "n4"]})
    check(
        "per-node seals identical across pool widths 1/4/8",
        all(s == seals[0] for s in seals),
        f"widths gave {'identical' if all(s == seals[0] for s in seals) else 'DIFFERENT'} seals",
    )
    from mop.campaign.invariance import run_invariance_sweep

    inv = run_invariance_sweep(n_items=1200, widths=[1, 2, 4, 8], seed=7)
    check("receipt-invariance sweep: identical merged receipt across worker widths", inv["receipt_invariant"])


def scenario_telegram() -> None:
    print("scenario: real Telegram delivery")
    from mop.campaign.telegram import send_campaign_event

    with tempfile.TemporaryDirectory() as d:
        r = send_campaign_event(
            "gauntlet rehearsal",
            "prelaunch gauntlet delivery check",
            root=Path(d),
            dedup_key="gauntlet-rehearsal-unique",
        )
        check(
            "telegram rehearsal delivered",
            bool(r.get("delivered")),
            f"message_id={r.get('message_id')}" if r.get("delivered") else f"error={r.get('error')}",
        )


def main() -> int:
    print("=== MOP prelaunch chaos gauntlet ===")
    scenario_singleton_and_adoption()
    scenario_crash_restart_stale_lease()
    scenario_worker_failure()
    scenario_mutation_fail_closed()
    scenario_scheduling_invariance()
    scenario_telegram()
    n_pass = sum(1 for _n, ok, _d in _RESULTS if ok)
    n_fail = sum(1 for _n, ok, _d in _RESULTS if not ok)
    print(f"\n=== gauntlet: {n_pass} passed, {n_fail} failed ===")
    out = Path("proof/campaign_run/GAUNTLET_RESULTS.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": n_pass,
                "failed": n_fail,
                "checks": [{"name": n, "pass": ok, "detail": dd} for n, ok, dd in _RESULTS],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"sealed -> {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
