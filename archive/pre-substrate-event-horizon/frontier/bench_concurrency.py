"""Measure real concurrent torch-training throughput at 1..K simultaneous capsules on this host.

Each capsule runs an identical fixed workload (train a small CNN for a fixed number of steps on in-memory
random data, no I/O), so the only variable is contention. For each concurrency level we launch that many
identical worker processes at once and record each worker's wall time; per-job slowdown = mean-worker-time /
solo-time, aggregate throughput = jobs / makespan. This produces the measured slowdown_fn the DAG scheduler
uses to model the parallel frontier wall time. House style: no dashes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")
PY = "/Users/scammermike/Downloads/mop/.venv/bin/python"
WORKER = "/tmp/bench_worker.py"
THREADS = 5           # torch threads per capsule (candidate: 4-6)
STEPS = 220           # fixed workload
LEVELS = [1, 2, 3, 4]

WORKER_SRC = f'''
import os, time
os.environ["OMP_NUM_THREADS"]="{THREADS}"; os.environ["MKL_NUM_THREADS"]="{THREADS}"
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads({THREADS})
torch.manual_seed(0)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.c1=nn.Conv2d(1,32,3,padding=1); s.c2=nn.Conv2d(32,64,3,padding=1)
        s.fc=nn.Linear(64*7*7,128); s.h=nn.Linear(128,47)
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.c1(x)),2); x=F.max_pool2d(F.relu(s.c2(x)),2); return s.h(F.relu(s.fc(x.flatten(1))))
x=torch.randn(256,1,28,28); y=torch.randint(0,47,(256,))
net=Net(); opt=torch.optim.Adam(net.parameters(),1e-3)
t=time.time()
for _ in range({STEPS}):
    opt.zero_grad(); F.cross_entropy(net(x),y).backward(); opt.step()
dt=time.time()-t
print(f"WORKER_DONE {{dt:.3f}} {{{STEPS}*256/dt:.1f}}")
'''


def main():
    Path(WORKER).write_text(WORKER_SRC)
    results = {}
    solo = None
    for k in LEVELS:
        procs = []
        t0 = time.time()
        for _ in range(k):
            procs.append(subprocess.Popen([PY, WORKER], stdout=subprocess.PIPE, text=True))
        times = []
        for p in procs:
            out, _ = p.communicate()
            for line in out.splitlines():
                if line.startswith("WORKER_DONE"):
                    times.append(float(line.split()[1]))
        makespan = time.time() - t0
        mean_t = sum(times) / len(times) if times else 0.0
        if k == 1:
            solo = mean_t
        slowdown = mean_t / solo if solo else 1.0
        throughput = k / makespan  # jobs per second
        agg_sps = k * STEPS * 256 / makespan
        results[k] = {"concurrency": k, "mean_worker_seconds": round(mean_t, 2),
                      "makespan_seconds": round(makespan, 2), "per_job_slowdown": round(slowdown, 3),
                      "jobs_per_second": round(throughput, 4), "aggregate_samples_per_second": round(agg_sps, 1),
                      "worker_times": [round(x, 2) for x in times]}
        print(f"  K={k}: mean_job={mean_t:.1f}s slowdown={slowdown:.2f}x makespan={makespan:.1f}s "
              f"agg_sps={agg_sps:.0f}", flush=True)
    # pick fastest safe throughput profile
    best_k = max(results, key=lambda k: results[k]["aggregate_samples_per_second"])
    report = {
        "schema": "mop-frontier-parallel-benchmark/v1", "host_cores": os.cpu_count(),
        "torch_threads_per_capsule": THREADS, "fixed_workload_steps": STEPS, "levels": results,
        "recommended_concurrency": best_k,
        "recommended_note": f"aggregate throughput maximized at {best_k} concurrent torch-train capsules with {THREADS} threads each",
        "measured_slowdown_by_concurrency": {str(k): results[k]["per_job_slowdown"] for k in results},
    }
    (OUT / "MOP_FRONTIER_PARALLEL_BENCHMARK.json").write_text(json.dumps(report, indent=2))
    print("BENCH_DONE best_k=", best_k, flush=True)
    return report


if __name__ == "__main__":
    main()
