"""One resource token DAG for the temporal core program.

Shard files on disk are the plan, so a restart resumes from what already landed. Stage order is a dependency
order, not a wish: nothing principal starts before calibration is green and the scout has landed, and nothing
is analysed before its shards are complete.

Stop switch: ~/.mop_temporal_core_mechanism_stop

House style: no dashes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from mop.temporal import io
from mop.temporal.runs import e2

PY = sys.executable
LOGS = io.ROOT / "logs"
CAP = int(os.environ.get("TEMPORAL_WORKERS", "20"))
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH="src")
BEDS = ("har_stream", "speech_stream", "harth_stream")


def workers() -> int:
    r = subprocess.run(["pgrep", "-f", "mop.temporal.runs"], capture_output=True, text=True)
    return max(0, len([x for x in r.stdout.split() if x]) - 1)


def launch(args: list[str], log: str):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / log, "a") as f:
        subprocess.Popen([PY, "-m", "mop.temporal.runs.e2", *args], cwd=io.ROOT, env=ENV,
                         stdout=f, stderr=subprocess.STDOUT)


def run_sync(mod: str, args=()) -> bool:
    r = subprocess.run([PY, "-m", mod, *args], cwd=io.ROOT, env=ENV, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    print(f"[{mod} {' '.join(args)}] exit {r.returncode}: {' | '.join(tail)}", flush=True)
    return r.returncode == 0


def missing(sub: str, names: list[str]) -> list[str]:
    d = io.RUNS / sub
    return [n for n in names if not (d / f"{n}.json").is_file()]


def main():
    scout_shards = [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]
    conv_shards = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    principal_shards = [f"{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    started: set[str] = set()

    for mod in ("mop.temporal.runs.authority", "mop.temporal.runs.custody_run",
                "mop.temporal.runs.method_ext", "mop.temporal.runs.codelife"):
        if not io.exists("MOP_TEMPORAL_CORE_START_AUTHORITY.json") or mod.endswith("codelife"):
            run_sync(mod)
    if not io.exists("MOP_E2_CALIBRATION.json"):
        run_sync("mop.temporal.runs.e2", ["calibration"])
    if not io.load("MOP_E2_CALIBRATION.json")["all_pass"]:
        print("[supervisor] calibration is not green, principal execution stays closed", flush=True)
        return

    while not io.STOP.exists():
        free = max(0, CAP - workers())
        pending_scout = missing("e2_scout", scout_shards)
        pending_conv = missing("e2_converge", conv_shards)
        stage1_done = not pending_scout and not pending_conv

        for n in pending_conv:
            tag = f"c:{n}"
            if tag in started or free <= 0:
                continue
            _, b, i = n.split("_")[0], "_".join(n.split("_")[1:-1]), n.split("_")[-1]
            launch(["converge_shard", b, i], f"e2_conv_{b}.log")
            started.add(tag)
            free -= 1
        for n in pending_scout:
            tag = f"s:{n}"
            if tag in started or free <= 0:
                continue
            b, s = "_".join(n.split("_")[1:-1]), n.split("_")[-1]
            launch(["scout_shard", b, s], f"e2_scout_{b}.log")
            started.add(tag)
            free -= 1

        if stage1_done:
            for b in BEDS:
                if not (io.RUNS / "e2_scout" / f"scout_{b}.json").is_file():
                    run_sync("mop.temporal.runs.e2", ["scout", b])
                if not (io.RUNS / "e2_converge" / f"converge_{b}.json").is_file():
                    run_sync("mop.temporal.runs.e2", ["converge", b])
            for n in missing("e2_principal", principal_shards):
                tag = f"p:{n}"
                if tag in started or free <= 0:
                    continue
                b, s = "_".join(n.split("_")[:-1]), n.split("_")[-1]
                launch(["principal", b, s], f"e2_principal_{b}.log")
                started.add(tag)
                free -= 1

        rem = {"scout": len(pending_scout), "converge": len(pending_conv),
               "principal": len(missing("e2_principal", principal_shards))}
        print(f"[supervisor] workers={workers()} remaining={rem}", flush=True)
        if not any(rem.values()):
            break
        time.sleep(60)

    for mod in ("mop.temporal.runs.bedvalid", "mop.temporal.runs.analyze", "mop.temporal.runs.coresel",
                "mop.temporal.runs.successors", "mop.temporal.runs.mutations",
                "mop.temporal.runs.verify", "mop.temporal.runs.reports", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric"):
        run_sync(mod)
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
