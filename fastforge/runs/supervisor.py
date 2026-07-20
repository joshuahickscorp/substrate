"""Stage supervisor.

One global dependency ready queue, driven by shard files on disk rather than by an in memory plan, so the
pipeline survives a restart and resumes from the ledger. It keeps the host near the measured concurrency
optimum: it will not start a stage that would push the worker count past the cap, and it will not leave the
host idle while dependency ready work exists.

Stop switch: ~/.mop_fast_state_forge_stop

House style: no dashes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from fastforge.runs import io

PY = sys.executable
STOP = Path.home() / ".mop_fast_state_forge_stop"
LOGS = io.ROOT / "logs"
CAP = int(os.environ.get("FASTFORGE_WORKERS", "18"))
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH="src")
DIRS = ["har-speech", "speech-har"]
SEEDS8 = list(range(8))
SEEDS5 = list(range(5))


def workers() -> int:
    r = subprocess.run(["pgrep", "-f", "fastforge.runs"], capture_output=True, text=True)
    return max(0, len([x for x in r.stdout.split() if x]) - 1)


def launch(mod, args, log):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / log, "w") as f:
        subprocess.Popen([PY, "-m", mod] + args, cwd=io.ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)


def have(kind, names):
    d = io.RUNS / kind
    return all((d / f"{n}.json").is_file() for n in names)


def missing(kind, names):
    d = io.RUNS / kind
    return [n for n in names if not (d / f"{n}.json").is_file()]


def run_sync(mod, args=()):
    r = subprocess.run([PY, "-m", mod] + list(args), cwd=io.ROOT, env=ENV, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-4:]
    print(f"[{mod}] exit {r.returncode}: {' | '.join(tail)}", flush=True)
    return r.returncode == 0


def main():
    bench_names = [
        f"{d}_{k}"
        for d in ("har", "speech")
        for k in ("gru", "lstm", "separate", "shared_adapters", "matched_capacity", "bag")
    ]
    cross_names = [f"{d}_{s}" for d in DIRS for s in SEEDS8]
    within_names = [f"{d}_{s}" for d in ("har", "speech") for s in SEEDS5]
    inter_names = [f"{d}_{s}" for d in DIRS for s in SEEDS8]
    round_names = [f"{d}_{s}" for d in DIRS for s in SEEDS8]
    started: set[str] = set()
    done: set[str] = set()

    while not STOP.exists():
        free = max(0, CAP - workers())

        if "bench" not in done and have("bench", bench_names):
            run_sync("fastforge.runs.bench")
            done.add("bench")

        if "cross" not in done and have("crossdomain", cross_names):
            run_sync("fastforge.runs.crossdomain")
            done.add("cross")

        # interference and within domain are ready as soon as the architectures are sealed
        for kind, mod, names in (
            ("interference", "fastforge.runs.interference", inter_names),
            ("within", "fastforge.runs.withinrun", within_names),
        ):
            if kind in done:
                continue
            if have(kind, names):
                run_sync(mod)
                done.add(kind)
                continue
            for n in missing(kind, names):
                tag = f"{kind}:{n}"
                if tag in started or free <= 0:
                    continue
                d, s = n.rsplit("_", 1)
                launch(mod, ["shard", f"{d}:{s}"], f"{kind}_{n}.log")
                started.add(tag)
                free -= 1

        # rounds depend on the interference map and the first matrix
        if "cross" in done and "interference" in done and "rounds" not in done:
            if have("rounds", round_names):
                run_sync("fastforge.runs.rounds")
                done.add("rounds")
            else:
                for n in missing("rounds", round_names):
                    tag = f"rounds:{n}"
                    if tag in started or free <= 0:
                        continue
                    d, s = n.rsplit("_", 1)
                    launch("fastforge.runs.rounds", ["shard", f"{d}:{s}"], f"rounds_{n}.log")
                    started.add(tag)
                    free -= 1

        if "cross" in done and "interference" in done and "plasticity" not in done:
            run_sync("fastforge.runs.plasticity")
            done.add("plasticity")
        if "cross" in done and "reorg" not in done:
            run_sync("fastforge.runs.reorg")
            done.add("reorg")
        if "cross" in done and "represent" not in done:
            run_sync("fastforge.runs.represent")
            done.add("represent")
        if "cross" in done and "mutations" not in done:
            run_sync("fastforge.runs.mutations")
            done.add("mutations")
        if "cross" in done and "verify" not in done:
            run_sync("fastforge.verify")
            done.add("verify")

        run_sync("fastforge.runs.domainvalidity")
        run_sync("fastforge.runs.thirddomain")
        run_sync("fastforge.runs.codereport")
        run_sync("fastforge.runs.fabric")
        run_sync("fastforge.runs.ledger")

        remaining = [
            k
            for k in (
                "bench",
                "cross",
                "interference",
                "within",
                "rounds",
                "plasticity",
                "reorg",
                "represent",
                "mutations",
                "verify",
            )
            if k not in done
        ]
        print(f"[supervisor] workers={workers()} done={sorted(done)} remaining={remaining}", flush=True)
        if not remaining:
            break
        time.sleep(120)

    run_sync("fastforge.runs.testreport")
    run_sync("fastforge.runs.fabric")
    run_sync("fastforge.runs.ledger")
    run_sync("fastforge.runs.synthesis")
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
