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
CAP_OVERRIDE = int(os.environ["TEMPORAL_WORKERS"]) if os.environ.get("TEMPORAL_WORKERS") else None
CAP_SMALL = 24
CAP_LARGE = 16
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


def _large_convergence_name(name: str) -> bool:
    try:
        idx = int(name.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return False
    return e2.CONVERGE_CONFIGS[idx].get("tier") == "large"


def scheduling_class(pending_extended: list[str]) -> tuple[int, list[str], str]:
    """Large curves run at their measured optimum before small curves use the wider optimum."""
    if CAP_OVERRIDE is not None:
        return CAP_OVERRIDE, pending_extended, "operator_override"
    active_large = False
    if LOCKS.is_dir():
        for p in LOCKS.glob("x_*.json"):
            try:
                d = json.loads(p.read_text())
                active_large = active_large or _large_convergence_name(d["tag"].split(":", 1)[1])
            except (OSError, KeyError, json.JSONDecodeError):
                continue
    large = [n for n in pending_extended if _large_convergence_name(n)]
    if active_large or large:
        return CAP_LARGE, large, "large_class_isolated"
    return CAP_SMALL, pending_extended, "small"


def launch(args: list[str], log: str, tag: str, module: str = "mop.temporal.runs.e2") -> bool:
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
        proc = subprocess.Popen([PY, "-m", module, *args], cwd=io.ROOT, env=env,
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


def invalid(sub: str, names: list[str]) -> list[str]:
    d, out = io.RUNS / sub, []
    for n in names:
        p = d / f"{n}.json"
        if not p.is_file():
            continue
        try:
            json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            out.append(n)
    return out


def partials(sub: str) -> list[str]:
    d = io.RUNS / sub
    return sorted(p.name for p in d.glob(".*.partial.*")) if d.is_dir() else []


def status() -> dict:
    scout = [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]
    conv = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    ext = [f"xshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    principal = [f"{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    e3_shards = [f"{source}_to_{target}_{seed}" for source, target in (
        ("har_stream", "speech_stream"), ("speech_stream", "har_stream"))
                 for seed in e2.PRINCIPAL_SEEDS]
    third = [f"harth_preflight_{seed}" for seed in e2.PRINCIPAL_SEEDS]
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
            "e3": len(e3_shards) - len(missing("e3", e3_shards)),
            "third_bed_preflight": len(third) - len(missing("third_bed_preflight", third)),
        },
        "missing": {
            "scout": missing("e2_scout", scout),
            "convergence": missing("e2_converge", conv),
            "extended_convergence": missing("e2_converge_extended", ext),
            "principal": missing("e2_principal", principal),
            "e3": missing("e3", e3_shards),
            "third_bed_preflight": missing("third_bed_preflight", third),
        },
        "invalid": {
            "scout": invalid("e2_scout", scout),
            "convergence": invalid("e2_converge", conv),
            "extended_convergence": invalid("e2_converge_extended", ext),
            "principal": invalid("e2_principal", principal),
            "e3": invalid("e3", e3_shards),
            "third_bed_preflight": invalid("third_bed_preflight", third),
        },
        "partial_receipts": {
            "scout": partials("e2_scout"),
            "convergence": partials("e2_converge"),
            "extended_convergence": partials("e2_converge_extended"),
            "principal": partials("e2_principal"),
            "e3": partials("e3"),
            "third_bed_preflight": partials("third_bed_preflight"),
        },
        "orchestration_incidents": sorted(
            p.name for p in (io.RUNS / "orchestration").glob("*.json")
        ) if (io.RUNS / "orchestration").is_dir() else [],
    }


def run_e3() -> bool:
    from mop.temporal.runs import e3

    names = [f"{source}_to_{target}_{seed}" for source, target in e3.DIRECTIONS for seed in e3.SEEDS]
    started: set[str] = set()
    while not io.STOP.exists():
        started = {tag for tag in started if lock_active(tag)}
        pending = missing("e3", names)
        if not pending:
            return run_sync("mop.temporal.runs.e3", ["aggregate"])
        free = max(0, CAP_SMALL - workers())
        for name in pending:
            tag = f"e3:{name}"
            if tag in started or lock_active(tag) or free <= 0:
                continue
            source_target, seed = name.rsplit("_", 1)
            source, target = source_target.split("_to_", 1)
            if launch(["shard", source, target, seed], "e3.log", tag,
                      module="mop.temporal.runs.e3"):
                started.add(tag)
                free -= 1
        print(f"[supervisor] E3 workers={workers()} cap={CAP_SMALL} remaining={len(pending)}", flush=True)
        time.sleep(60)
    return False


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
        pending_scout = missing("e2_scout", scout_shards)
        pending_conv = missing("e2_converge", conv_shards)
        pending_ext = missing("e2_converge_extended", ext_shards)
        if pending_conv:
            cap, eligible_ext, resource_class = (CAP_OVERRIDE or CAP_SMALL), [], "small_base"
        else:
            cap, eligible_ext, resource_class = scheduling_class(pending_ext)
        free = max(0, cap - workers())
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
            for n in eligible_ext:
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
        print(f"[supervisor] workers={workers()} cap={cap} class={resource_class} remaining={rem}", flush=True)
        if not any(rem.values()):
            break
        time.sleep(60)

    remaining = status()["missing"]
    invalid_receipts = status()["invalid"]
    scientific_missing = any(remaining[k] for k in ("scout", "convergence", "extended_convergence",
                                                     "principal"))
    if io.STOP.exists() or scientific_missing or any(invalid_receipts[k] for k in (
            "scout", "convergence", "extended_convergence", "principal")):
        print("[supervisor] stopped before downstream aggregation because shard work is not terminal", flush=True)
        return
    run_sync("mop.temporal.runs.e2", ["scout_result"])
    for b in BEDS:
        run_sync("mop.temporal.runs.e2", ["converge", b])
    run_sync("mop.temporal.runs.thirdbed", ["all"])
    for mod in ("mop.temporal.runs.bedvalid", "mop.temporal.runs.analyze",
                "mop.temporal.runs.replicate", "mop.temporal.runs.mutations",
                "mop.temporal.runs.verify", "mop.temporal.runs.coresel",
                "mop.temporal.runs.successors"):
        run_sync(mod)
    queue = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json") if io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json") else {}
    if "E3_shared_versus_local" in (queue.get("licensed_top_two") or []):
        run_e3()
        run_sync("mop.temporal.runs.mutations")
        run_sync("mop.temporal.runs.verify")
        run_sync("mop.temporal.runs.successors")
    for mod in ("mop.temporal.runs.reports", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric"):
        run_sync(mod)
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
