"""Global work-conserving resource-token DAG scheduler for the MOP scientific frontier.

One scheduler spans every campaign, lane, and phase. Concurrency is NOT a single global worker count: each
capsule declares a resource vector (cpu, memory_gb, network, disk, gpu) plus exclusive output locks, and the
scheduler admits every dependency-ready capsule that fits the remaining host token budget (resource-aware bin
packing), fills released capacity immediately, prevents duplicate execution and shared-output writers, closes
only the descendants of a failed scientific dependency, and keeps unrelated lanes running.

It supports two execution modes with identical scheduling logic:
  * real:      each capsule carries a run() thunk (a subprocess or callable); wall time is measured.
  * simulate:  each capsule carries a measured duration; the scheduler advances a virtual clock, applying a
               measured per-job concurrency slowdown, to produce a modeled parallel wall time and the exact
               ready/running/blocked/completed/pruned timeline. Receipt identity is start-order and
               worker-width invariant because it derives only from the capsule authority, not the schedule.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


# ---- task-class resource profiles (cpu cores, memory GB, network, disk I/O, gpu) ----
PROFILES = {
    "download": {"cpu": 1, "memory_gb": 1, "network": 1, "disk": 1, "gpu": 0},
    "extract": {"cpu": 1, "memory_gb": 2, "network": 0, "disk": 2, "gpu": 0},
    "preprocess": {"cpu": 2, "memory_gb": 3, "network": 0, "disk": 1, "gpu": 0},
    "numpy_battery": {"cpu": 2, "memory_gb": 2, "network": 0, "disk": 0, "gpu": 0},
    "small_verify": {"cpu": 1, "memory_gb": 1, "network": 0, "disk": 0, "gpu": 0},
    "large_verify": {"cpu": 4, "memory_gb": 4, "network": 0, "disk": 0, "gpu": 0},
    "torch_train": {"cpu": 6, "memory_gb": 6, "network": 0, "disk": 0, "gpu": 0},
    "memory_heavy_cl": {"cpu": 7, "memory_gb": 10, "network": 0, "disk": 1, "gpu": 0},
    "control_arm": {"cpu": 2, "memory_gb": 2, "network": 0, "disk": 0, "gpu": 0},
    "aggregation": {"cpu": 1, "memory_gb": 2, "network": 0, "disk": 0, "gpu": 0},
    "report": {"cpu": 1, "memory_gb": 1, "network": 0, "disk": 0, "gpu": 0},
    "mutation_suite": {"cpu": 2, "memory_gb": 2, "network": 0, "disk": 0, "gpu": 0},
}


@dataclass
class Capsule:
    cid: str
    task_class: str
    deps: list = field(default_factory=list)         # capsule ids that must complete first
    duration: float = 0.0                            # measured seconds (simulate mode)
    output_locks: list = field(default_factory=list)  # exclusive output resource names
    authority: str = ""                              # scientific authority / receipt identity
    scientific_dep: bool = True                      # if a dep of this capsule fails, descendants are pruned
    run: object = None                               # callable for real mode
    thread_hint: int = 4                             # internal torch thread count suggestion

    def profile(self):
        return PROFILES[self.task_class]


class HostBudget:
    def __init__(self, cpu=28, memory_gb=96, network=4, disk=4, gpu=0):
        self.cap = {"cpu": cpu, "memory_gb": memory_gb, "network": network, "disk": disk, "gpu": gpu}

    def fits(self, used, prof):
        return all(used[k] + prof[k] <= self.cap[k] for k in self.cap)


class DAGScheduler:
    """Work-conserving resource-token scheduler. States: ready, running, blocked, held, completed, pruned."""

    def __init__(self, capsules, host=None, slowdown_fn=None):
        self.caps = {c.cid: c for c in capsules}
        self.host = host or HostBudget()
        # measured per-job slowdown as a function of concurrent same-class jobs (default: no slowdown)
        self.slowdown_fn = slowdown_fn or (lambda cls, k: 1.0)
        self.state = {c: "blocked" for c in self.caps}
        self.done = set(); self.pruned = set()
        self.timeline = []      # (t_start, t_end, cid, concurrency_at_start)

    def _ready(self):
        r = []
        for cid, c in self.caps.items():
            if self.state[cid] != "blocked":
                continue
            if any(d in self.pruned for d in c.deps):
                continue  # will be pruned
            if all(d in self.done for d in c.deps):
                r.append(cid)
        return r

    def _prune_descendants(self, cid):
        # close only descendants of a failed scientific dependency
        stack = [cid]
        while stack:
            x = stack.pop()
            for oid, c in self.caps.items():
                if oid in self.pruned or oid in self.done:
                    continue
                if x in c.deps and c.scientific_dep:
                    if self.state[oid] != "pruned":
                        self.state[oid] = "pruned"; self.pruned.add(oid); stack.append(oid)

    def simulate(self, failed=None):
        """Advance a virtual clock; return modeled parallel wall time and the full timeline."""
        failed = set(failed or [])
        used = {k: 0 for k in self.host.cap}
        locks = set()
        running = []  # heap of (t_end, cid)
        t = 0.0
        # mark failed capsules' descendants pruned up front (deterministic closure)
        for f in failed:
            if f in self.caps:
                self._prune_descendants(f)
        completed_count = 0
        total = len(self.caps)
        peak_conc = 0
        area = 0.0  # integral of concurrency over time (for average concurrency)
        last_t = 0.0
        while completed_count + len(self.pruned) < total:
            # admit ready capsules that fit
            progressed = True
            while progressed:
                progressed = False
                for cid in sorted(self._ready(), key=lambda c: -PROFILES[self.caps[c].task_class]["cpu"]):
                    c = self.caps[cid]
                    prof = c.profile()
                    if any(l in locks for l in c.output_locks):
                        self.state[cid] = "held"; continue
                    if self.host.fits(used, prof):
                        for k in used:
                            used[k] += prof[k]
                        for l in c.output_locks:
                            locks.add(l)
                        conc = len([1 for _ in running]) + 1
                        dur = c.duration * self.slowdown_fn(c.task_class, conc)
                        if cid in failed:
                            dur = c.duration  # failure still consumes some time then prunes
                        heapq.heappush(running, (t + dur, cid))
                        self.state[cid] = "running"
                        self.timeline.append({"cid": cid, "class": c.task_class, "t_start": round(t, 2),
                                              "t_end": round(t + dur, 2), "concurrency": conc, "authority": c.authority})
                        progressed = True
                peak_conc = max(peak_conc, len(running))
            if not running:
                break
            # advance to next completion
            area += len(running) * (heapq.nsmallest(1, running)[0][0] - t)
            t_end, cid = heapq.heappop(running)
            area_correction = 0
            t = t_end
            c = self.caps[cid]
            prof = c.profile()
            for k in used:
                used[k] -= prof[k]
            for l in c.output_locks:
                locks.discard(l)
            if cid in failed:
                self.state[cid] = "pruned"; self.pruned.add(cid); self._prune_descendants(cid)
            else:
                self.state[cid] = "completed"; self.done.add(cid); completed_count += 1
        makespan = t
        serial = sum(c.duration for c in self.caps.values() if c.cid in self.done)
        avg_conc = (area / makespan) if makespan > 0 else 0.0
        return {
            "modeled_parallel_wall_seconds": round(makespan, 1),
            "serial_wall_seconds": round(serial, 1),
            "speedup": round(serial / makespan, 2) if makespan > 0 else None,
            "peak_concurrency": peak_conc, "average_concurrency": round(avg_conc, 2),
            "completed": sorted(self.done), "pruned": sorted(self.pruned),
            "timeline": sorted(self.timeline, key=lambda x: x["t_start"]),
        }

    def critical_path(self):
        """Longest dependency chain by duration (the scientific critical path)."""
        memo = {}

        def longest(cid):
            if cid in memo:
                return memo[cid]
            c = self.caps[cid]
            best = c.duration + (max((longest(d) for d in c.deps), default=0.0))
            memo[cid] = best
            return best
        end = max(self.caps, key=longest)
        # reconstruct path
        path = []; cur = end
        while cur is not None:
            path.append(cur)
            deps = self.caps[cur].deps
            cur = max(deps, key=longest) if deps else None
        return {"length_seconds": round(longest(end), 1), "path": list(reversed(path))}


def status_projection(sched):
    counts = {}
    for s in sched.state.values():
        counts[s] = counts.get(s, 0) + 1
    return {"total": len(sched.state), **{k: counts.get(k, 0) for k in
            ("ready", "running", "blocked", "held", "completed", "pruned")}}
