"""One resource token DAG for the temporal core program.

Shard files on disk are the plan, so a restart resumes from what already landed. Stage order is a dependency
order, not a wish: nothing principal starts before calibration is green and the scout has landed, and nothing
is analysed before its shards are complete.

Stop switch: ~/.mop_temporal_core_mechanism_stop

House style: no dashes.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
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
LOCKS = io.RUNS / "locks"


def workers() -> int:
    r = subprocess.run(["pgrep", "-f", "mop.temporal.runs"], capture_output=True, text=True)
    return max(0, len([x for x in r.stdout.split() if x]) - 1)


def _lock_path(tag: str) -> Path:
    return LOCKS / f"{tag.replace(':', '_')}.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def lock_active(tag: str) -> bool:
    p = _lock_path(tag)
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        if _pid_alive(int(d["pid"])):
            return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    p.unlink(missing_ok=True)
    return False


def launch(args: list[str], log: str, tag: str) -> bool:
    LOGS.mkdir(exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(tag)
    if lock_active(tag):
        return False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps({"pid": os.getpid(), "tag": tag, "state": "reserved"}).encode())
    os.close(fd)
    with open(LOGS / log, "a") as f:
        env = dict(ENV, TEMPORAL_SHARD_LOCK=str(lock), TEMPORAL_SHARD_TAG=tag)
        proc = subprocess.Popen([PY, "-m", "mop.temporal.runs.e2", *args], cwd=io.ROOT, env=env,
                                stdout=f, stderr=subprocess.STDOUT)
    lock.write_text(json.dumps({"pid": proc.pid, "tag": tag, "args": args, "state": "active"}))
    return True


def run_sync(mod: str, args=()) -> bool:
    r = subprocess.run([PY, "-m", mod, *args], cwd=io.ROOT, env=ENV, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    print(f"[{mod} {' '.join(args)}] exit {r.returncode}: {' | '.join(tail)}", flush=True)
    return r.returncode == 0


def missing(sub: str, names: list[str]) -> list[str]:
    d = io.RUNS / sub
    return [n for n in names if not (d / f"{n}.json").is_file()]


def status() -> dict:
    scout = [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]
    conv = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    ext = [f"xshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    principal = [f"{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    active = []
    if LOCKS.is_dir():
        for p in sorted(LOCKS.glob("*.json")):
            tag = p.stem.replace("_", ":", 1)
            if lock_active(tag):
                active.append(json.loads(p.read_text()))
    return {
        "schema": "mop-temporal-supervisor-status/v1",
        "stop_switch_active": io.STOP.exists(),
        "process_workers": workers(),
        "active_shards": active,
        "completed": {
            "scout": len(scout) - len(missing("e2_scout", scout)),
            "convergence": len(conv) - len(missing("e2_converge", conv)),
            "extended_convergence": len(ext) - len(missing("e2_converge_extended", ext)),
            "principal": len(principal) - len(missing("e2_principal", principal)),
        },
        "missing": {
            "scout": missing("e2_scout", scout),
            "convergence": missing("e2_converge", conv),
            "extended_convergence": missing("e2_converge_extended", ext),
            "principal": missing("e2_principal", principal),
        },
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv and argv[0] == "status":
        print(json.dumps(status(), indent=2))
        return
    scout_shards = [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]
    conv_shards = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    ext_shards = [f"xshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
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
        started = {tag for tag in started if lock_active(tag)}
        free = max(0, CAP - workers())
        pending_scout = missing("e2_scout", scout_shards)
        pending_conv = missing("e2_converge", conv_shards)
        pending_ext = missing("e2_converge_extended", ext_shards)
        stage1_done = not pending_scout and not pending_conv and not pending_ext

        for n in pending_conv:
            tag = f"c:{n}"
            if tag in started or free <= 0:
                continue
            _, b, i = n.split("_")[0], "_".join(n.split("_")[1:-1]), n.split("_")[-1]
            if launch(["converge_shard", b, i], f"e2_conv_{b}.log", tag):
                started.add(tag)
                free -= 1
        for n in pending_scout:
            tag = f"s:{n}"
            if tag in started or free <= 0:
                continue
            b, s = "_".join(n.split("_")[1:-1]), n.split("_")[-1]
            if launch(["scout_shard", b, s], f"e2_scout_{b}.log", tag):
                started.add(tag)
                free -= 1

        if not pending_conv:
            for n in pending_ext:
                tag = f"x:{n}"
                if tag in started or lock_active(tag) or free <= 0:
                    continue
                _, b, i = n.split("_")[0], "_".join(n.split("_")[1:-1]), n.split("_")[-1]
                if launch(["extend_converge_shard", b, i], f"e2_conv_extended_{b}.log", tag):
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
                if launch(["principal", b, s], f"e2_principal_{b}.log", tag):
                    started.add(tag)
                    free -= 1

        rem = {"scout": len(pending_scout), "converge": len(pending_conv), "extended": len(pending_ext),
               "principal": len(missing("e2_principal", principal_shards))}
        print(f"[supervisor] workers={workers()} remaining={rem}", flush=True)
        if not any(rem.values()):
            break
        time.sleep(60)

    if io.STOP.exists() or any(status()["missing"].values()):
        print("[supervisor] stopped before downstream aggregation because shard work is not terminal", flush=True)
        return
    run_sync("mop.temporal.runs.e2", ["scout_result"])
    for b in BEDS:
        run_sync("mop.temporal.runs.e2", ["converge", b])
    for mod in ("mop.temporal.runs.bedvalid", "mop.temporal.runs.analyze",
                "mop.temporal.runs.replicate", "mop.temporal.runs.mutations",
                "mop.temporal.runs.verify", "mop.temporal.runs.coresel",
                "mop.temporal.runs.successors", "mop.temporal.runs.reports", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric"):
        run_sync(mod)
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
