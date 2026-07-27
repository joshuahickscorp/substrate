"""One resource token DAG for the whole program.

Dependency ready work is driven by shard files on disk rather than by an in memory plan, so a restart
resumes from what already landed. The cap is a worker count, not a thread count: every child runs with two
BLAS threads, which is where the throughput benchmark put the knee on this host.

Stop switch: ~/.mop_experimental_method_stop

House style: no dashes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from mop.method import io
from mop.method.runs import exp1, exp4

PY = sys.executable
LOGS = io.ROOT / "logs"
CAP = int(os.environ.get("METHOD_WORKERS", "10"))
ENV = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH="src")


def workers() -> int:
    r = subprocess.run(["pgrep", "-f", "mop.method.runs"], capture_output=True, text=True)
    return max(0, len([x for x in r.stdout.split() if x]) - 1)


def launch(mod: str, args: list[str], log: str):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / log, "a") as f:
        subprocess.Popen([PY, "-m", mod] + args, cwd=io.ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)


def run_sync(mod: str, args=()) -> bool:
    r = subprocess.run([PY, "-m", mod] + list(args), cwd=io.ROOT, env=ENV, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    print(f"[{mod} {' '.join(args)}] exit {r.returncode}: {' | '.join(tail)}", flush=True)
    return r.returncode == 0


def have(sub: str, names: list[str]) -> bool:
    return all((io.RUNS / sub / f"{n}.json").is_file() for n in names)


def missing(sub: str, names: list[str]) -> list[str]:
    return [n for n in names if not (io.RUNS / sub / f"{n}.json").is_file()]


def main():
    e1_shards = [f"{b}_{s}" for b in exp1.BEDS for s in exp1.PRINCIPAL_SEEDS]
    e4_shards = [f"{b}_{s}" for b in exp4.BEDS for s in exp4.PRINCIPAL_SEEDS]
    e1_scouts = [f"scout_{b}" for b in exp1.BEDS]
    e4_scouts = [f"scout_{b}" for b in exp4.BEDS]
    started: set[str] = set()

    while not io.STOP.exists():
        free = max(0, CAP - workers())

        # tier 1: no compute, always cheap, keeps the authority current
        for mod in ("mop.method.runs.authority", "mop.method.runs.acceptance_run"):
            if not io.exists("MOP_METHOD_ACCEPTANCE_RESULT.json"):
                run_sync(mod)

        licensed = io.exists("MOP_METHOD_ACCEPTANCE_RESULT.json") and io.load(
            "MOP_METHOD_ACCEPTANCE_RESULT.json"
        )["principal_execution_licensed_by_this_gate"]
        if not licensed:
            print("[supervisor] acceptance gate is not green, principal execution stays closed", flush=True)
            time.sleep(60)
            continue

        # tier 2: admissions, then scouts, then principal shards, per experiment
        for exp, mod, beds, scouts, scout_sub, shards, shard_sub, admission in (
            ("E1", "mop.method.runs.exp1", exp1.BEDS, e1_scouts, "scout", e1_shards, "principal",
             "MOP_E1_ADMISSION.json"),
            ("E4", "mop.method.runs.exp4", exp4.BEDS, e4_scouts, "e4_scout", e4_shards, "e4_principal",
             "MOP_E4_ADMISSION.json"),
        ):
            if not io.exists(admission):
                run_sync(mod, ["admit"])
                continue
            if not io.load(admission)["admission"]["licensed"]:
                print(f"[supervisor] {exp} admission blocked, no compute spent", flush=True)
                continue
            if not have(scout_sub, scouts):
                for b in beds:
                    tag = f"{exp}:scout:{b}"
                    if tag in started or free <= 0 or (io.RUNS / scout_sub / f"scout_{b}.json").is_file():
                        continue
                    launch(mod, ["scout", b], f"{exp.lower()}_scout_{b}.log")
                    started.add(tag)
                    free -= 1
                continue
            for n in missing(shard_sub, shards):
                tag = f"{exp}:principal:{n}"
                if tag in started or free <= 0:
                    continue
                b, s = n.rsplit("_", 1)
                launch(mod, ["principal", b, s], f"{exp.lower()}_principal_{n}.log")
                started.add(tag)
                free -= 1

        e1_ready = have("principal", e1_shards)
        e4_ready = have("e4_principal", e4_shards)
        if e1_ready and not io.exists("MOP_PRINCIPAL_EXPERIMENT_1.json"):
            run_sync("mop.method.runs.analyze1")
        if e4_ready and not io.exists("MOP_PRINCIPAL_EXPERIMENT_2.json"):
            run_sync("mop.method.runs.analyze4")

        remaining = {
            "e1_scouts": missing("scout", e1_scouts),
            "e4_scouts": missing("e4_scout", e4_scouts),
            "e1_principal": missing("principal", e1_shards),
            "e4_principal": missing("e4_principal", e4_shards),
        }
        print(f"[supervisor] workers={workers()} remaining="
              f"{ {k: len(v) for k, v in remaining.items()} }", flush=True)
        if not any(remaining.values()):
            break
        time.sleep(45)

    # terminal order: analyses, then the two verification roles, then reports, then the fabric last
    for mod in (
        "mop.method.runs.analyze1",
        "mop.method.runs.analyze4",
        "mop.method.runs.verify",
        "mop.method.runs.audit",
        "mop.method.runs.reports",
        "mop.method.runs.synthesis",
        "mop.method.runs.fabric",
    ):
        run_sync(mod)
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
