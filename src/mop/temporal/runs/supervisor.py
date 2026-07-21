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
    return max(0, len([x for x in r.stdout.split() if x]) - 1, sum(lock_active(p.stem.replace("_", ":", 1)) for p in LOCKS.glob("*.json")) if LOCKS.is_dir() else 0)
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
        for p in LOCKS.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                if d["tag"].startswith(("c:", "x:")):
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
            doc = json.loads(p.read_text())
            if not isinstance(doc, dict):
                raise ValueError("receipt top level must be an object")
            version = doc.get("result_hash_version")
            if version == "canonical_json_v2" and doc.get("result_sha256") != io.sha_obj(
                    {k: v for k, v in doc.items() if k != "result_sha256"}):
                raise ValueError("receipt content hash mismatch")
            if version not in (None, "canonical_json_v2"):
                raise ValueError("unknown receipt content hash version")
            if sub in ("e2_converge", "e2_converge_extended"):
                idx = int(n.rsplit("_", 1)[1])
                expected = e2.Fx.cell_name(**e2.CONVERGE_CONFIGS[idx])
                spec = doc.get("spec")
                if not isinstance(spec, dict) or not set(e2.Fx.REFERENCE) <= set(spec) \
                        or doc.get("cell") != expected or e2.Fx.cell_name(**spec) != expected:
                    raise ValueError("convergence index identity mismatch")
                if sub == "e2_converge_extended":
                    extension = doc.get("extends")
                    if not isinstance(extension, dict) or not isinstance(extension.get("path"), str) \
                            or not isinstance(extension.get("sha256"), str):
                        raise ValueError("extended convergence dependency shape invalid")
                    base_path = io.ROOT / extension["path"]
                    base = json.loads(base_path.read_text())
                    if not isinstance(base, dict) or extension["sha256"] != io.sha_file(base_path) \
                            or base.get("cell") != expected or e2.Fx.cell_name(**base.get("spec", {})) != expected:
                        raise ValueError("extended convergence base identity mismatch")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            out.append(n)
    return out
def partials(sub: str) -> list[str]:
    d = io.RUNS / sub
    return sorted(p.name for p in d.glob(".*.partial.*")) if d.is_dir() else []
def recoverable_pending(sub: str, names: list[str]) -> list[str]:
    """Move invalid receipts to append only quarantine before making their identities runnable again."""
    for name in invalid(sub, names):
        source = io.RUNS / sub / f"{name}.json"
        digest = io.sha_file(source)
        quarantine = io.RUNS / "quarantine" / "invalid_receipts"
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / f"{sub}_{name}_{digest[:12]}.json"
        if target.exists():
            target = quarantine / f"{sub}_{name}_{digest[:12]}_{time.time_ns()}.json"
        source.rename(target)
        io.run_json(f"quarantine_{sub}_{name}_{digest[:12]}.json", {
            "schema": "mop-temporal-invalid-receipt-quarantine/v1", "stage": sub, "identity": name,
            "original_path": source.relative_to(io.ROOT).as_posix(),
            "quarantine_path": target.relative_to(io.ROOT).as_posix(), "sha256": digest,
            "classification": "invalid_receipt_excluded_pending_replacement"}, "orchestration")
        print(f"[supervisor] quarantined invalid {sub}/{name} as {target.name}", flush=True)
    return missing(sub, names)
def status() -> dict:
    scout = [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]
    conv = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    ext = [f"xshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    principal = [f"{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    principal_corrections = [f"capacity_{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    convergence_corrections = [f"convergence_{b}" for b in BEDS]
    optimization_corrections = [f"optimization_{b}_{tier}" for b in BEDS for tier in ("small", "large")]
    e3_shards = [f"{source}_to_{target}_{seed}" for source, target in (
        ("har_stream", "speech_stream"), ("speech_stream", "har_stream"))
                 for seed in e2.PRINCIPAL_SEEDS]
    third = [f"harth_preflight_{seed}" for seed in e2.PRINCIPAL_SEEDS]
    hybrid = [f"{bed}_{seed}" for bed in e2.B.PRINCIPAL for seed in e2.PRINCIPAL_SEEDS]
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
            "principal_corrections": len(principal_corrections) - len(
                missing("e2_principal_corrections", principal_corrections)),
            "convergence_corrections": len(convergence_corrections) - len(
                missing("e2_converge_corrections", convergence_corrections)),
            "optimization_corrections": len(optimization_corrections) - len(
                missing("e2_optimization_corrections", optimization_corrections)),
            "e3": len(e3_shards) - len(missing("e3", e3_shards)),
            "third_bed_preflight": len(third) - len(missing("third_bed_preflight", third)),
            "hybrid": len(hybrid) - len(missing("hybrid", hybrid)),
        },
        "missing": {
            "scout": missing("e2_scout", scout),
            "convergence": missing("e2_converge", conv),
            "extended_convergence": missing("e2_converge_extended", ext),
            "principal": missing("e2_principal", principal),
            "principal_corrections": missing("e2_principal_corrections", principal_corrections),
            "convergence_corrections": missing("e2_converge_corrections", convergence_corrections),
            "optimization_corrections": missing("e2_optimization_corrections", optimization_corrections),
            "e3": missing("e3", e3_shards),
            "third_bed_preflight": missing("third_bed_preflight", third),
            "hybrid": missing("hybrid", hybrid),
        },
        "invalid": {
            "scout": invalid("e2_scout", scout),
            "convergence": invalid("e2_converge", conv),
            "extended_convergence": invalid("e2_converge_extended", ext),
            "principal": invalid("e2_principal", principal),
            "principal_corrections": invalid("e2_principal_corrections", principal_corrections),
            "convergence_corrections": invalid("e2_converge_corrections", convergence_corrections),
            "optimization_corrections": invalid("e2_optimization_corrections", optimization_corrections),
            "e3": invalid("e3", e3_shards),
            "third_bed_preflight": invalid("third_bed_preflight", third),
            "hybrid": invalid("hybrid", hybrid),
        },
        "partial_receipts": {
            "scout": partials("e2_scout"),
            "convergence": partials("e2_converge"),
            "extended_convergence": partials("e2_converge_extended"),
            "principal": partials("e2_principal"),
            "principal_corrections": partials("e2_principal_corrections"),
            "convergence_corrections": partials("e2_converge_corrections"),
            "optimization_corrections": partials("e2_optimization_corrections"),
            "e3": partials("e3"),
            "third_bed_preflight": partials("third_bed_preflight"),
            "hybrid": partials("hybrid"),
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
        pending = recoverable_pending("e3", names)
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
def _artifact_true(name: str, field: str) -> bool:
    if not io.exists(name):
        return False
    try:
        doc = io.load(name)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(doc, dict) and doc.get(field) is True
def _core_selected() -> bool:
    return _artifact_true("MOP_OWNED_TEMPORAL_CORE_V1.json", "selected")
def prepare_verified_core() -> bool:
    """Select, independently verify, and withdraw a selection that fails its post selection check."""
    if not run_sync("mop.temporal.runs.coresel"):
        return False
    verifier_ran = run_sync("mop.temporal.runs.verify")
    if verifier_ran and _artifact_true("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "all_pass"):
        return True
    print("[supervisor] post core verification failed; withdrawing the selection", flush=True)
    selector_ran = run_sync("mop.temporal.runs.coresel")
    withdrawn = selector_ran and not _core_selected()
    verifier_reran = run_sync("mop.temporal.runs.verify")
    return bool(withdrawn and verifier_reran and _artifact_true(
        "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "all_pass"))
def run_verified_successor_gates() -> bool:
    """Never compute successor licences from a core that did not survive independent verification."""
    if not prepare_verified_core():
        print("[supervisor] successor licensing stays closed because no verified core state exists", flush=True)
        return False
    return run_sync("mop.temporal.runs.successors")
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
        pending_scout = recoverable_pending("e2_scout", scout_shards)
        pending_conv = recoverable_pending("e2_converge", conv_shards)
        pending_ext = recoverable_pending("e2_converge_extended", ext_shards)
        ready_ext = [n for n in pending_ext if (io.RUNS / "e2_converge" /
                     f"cshard_{n.removeprefix('xshard_')}.json").is_file()]
        cap, eligible, resource_class = scheduling_class(pending_conv + ready_ext)
        eligible = set(eligible)
        free = max(0, cap - workers())
        stage1_done = not pending_scout and not pending_conv and not pending_ext
        for n in pending_conv:
            tag = f"c:{n}"
            if n not in eligible or tag in started or free <= 0:
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
        for n in ready_ext:
            tag = f"x:{n}"
            if n not in eligible or tag in started or lock_active(tag) or free <= 0:
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
    for mod in ("mop.temporal.runs.corrections", "mop.temporal.runs.bedvalid", "mop.temporal.runs.analyze",
                "mop.temporal.runs.replicate", "mop.temporal.runs.mutations",
                "mop.temporal.runs.verify", "mop.temporal.runs.analyze", "mop.temporal.runs.verify"):
        run_sync(mod)
    if not run_verified_successor_gates():
        return
    queue = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json") if io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json") else {}
    if "E3_shared_versus_local" in (queue.get("licensed_top_two") or []):
        run_e3()
        run_sync("mop.temporal.runs.mutations")
        run_sync("mop.temporal.runs.verify")
        run_sync("mop.temporal.runs.successors")
    if "hybrid_adaptation" in (queue.get("licensed_top_two") or []):
        run_sync("mop.temporal.runs.hybrid", ["all"])
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
