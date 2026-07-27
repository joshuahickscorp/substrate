"""Measure the terminal deterministic synthesis without crossing its launch boundary."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = ROOT / "artifacts" / "substrate" / "execution-forge"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _unit_map():
    from substrate import execution

    return execution.BY_UNIT


def _units():
    from substrate import execution

    return execution.UNIT_LIST


def _prepare(root: Path) -> None:
    from substrate import evidence

    shutil.copytree(evidence.PROOF, root / "evidence", dirs_exist_ok=True)
    if evidence.RUNS.is_dir():
        shutil.copytree(evidence.RUNS, root / "runs", dirs_exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)


def _configure(root: Path, thread_budget: int) -> None:
    for name in THREAD_VARIABLES:
        os.environ[name] = str(thread_budget)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(ROOT)
    from substrate import evidence, program

    evidence.PROOF = root / "evidence"
    evidence.RUNS = root / "runs"
    evidence.ARTIFACTS = root / "artifacts"
    program.PROOF_ROOTS[""] = evidence.PROOF
    program._REACHABLE.clear()
    evidence.commit.cache_clear()


def _reset_mutable_state(root: Path, thread_budget: int) -> None:
    _configure(root, thread_budget)
    import random

    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass


def _direct_unit(identity: str, root: Path, thread_budget: int) -> dict:
    _reset_mutable_state(root, thread_budget)
    unit = _unit_map()[identity]
    started = time.perf_counter()
    try:
        with tempfile.TemporaryFile(mode="w+") as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            module = importlib.import_module(unit.module)
            module.main(list(unit.args))
        missing = [name for name in unit.produces if not (root / "evidence" / name).is_file()]
        ok = not missing
        detail = "" if ok else f"missing outputs: {missing}"
    except BaseException as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "unit": identity,
        "ok": ok,
        "detail": detail,
        "wall_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }


def _pool_initializer(root: str, thread_budget: int) -> None:
    _configure(Path(root), thread_budget)


def _pool_task(identity: str, root: str, thread_budget: int) -> dict:
    return _direct_unit(identity, Path(root), thread_budget)


def _thread_budget_environment(thread_budget: int) -> dict:
    return {name: str(thread_budget) for name in THREAD_VARIABLES}


def _run_subprocess(identity: str, root: Path, thread_budget: int) -> dict:
    env = {
        **os.environ,
        **_thread_budget_environment(thread_budget),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "unit",
            identity,
            "--root",
            str(root),
            "--threads",
            str(thread_budget),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError:
        document = {
            "unit": identity,
            "ok": False,
            "detail": (result.stdout or result.stderr)[-500:],
            "pid": None,
        }
    document["wall_seconds"] = time.perf_counter() - started
    return document


def _descendant_metrics(parent_pid: int, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.wait(0.2):
        result = subprocess.run(
            ["ps", "axo", "pid=,ppid=,rss=,command="],
            capture_output=True,
            text=True,
        )
        rows = {}
        for line in result.stdout.splitlines():
            fields = line.strip().split(None, 3)
            if len(fields) < 4:
                continue
            try:
                pid, ppid, rss = map(int, fields[:3])
            except ValueError:
                continue
            rows[pid] = {"pid": pid, "ppid": ppid, "rss_kib": rss, "threads": 1, "command": fields[3]}
        descendants, frontier = set(), {parent_pid}
        while frontier:
            children = {pid for pid, row in rows.items() if row["ppid"] in frontier and pid not in descendants}
            descendants |= children
            frontier = children
        selected = [rows[pid] for pid in descendants if pid in rows]
        for row in selected:
            thread_result = subprocess.run(["ps", "-M", "-p", str(row["pid"])], capture_output=True, text=True)
            row["threads"] = max(1, len(thread_result.stdout.splitlines()) - 1)
        samples.append(
            {
                "processes": len(selected),
                "threads": sum(row["threads"] for row in selected),
                "rss_kib": sum(row["rss_kib"] for row in selected),
                "tree": selected,
            }
        )


def _swap_used_mib() -> float | None:
    result = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    for token_index, token in enumerate(result.stdout.split()):
        if token == "used" and token_index + 2 < len(result.stdout.split()):
            value = result.stdout.split()[token_index + 2]
            if value.endswith("M"):
                return float(value[:-1])
    return None


def _memory_free_percent() -> int | None:
    result = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True)
    marker = "System-wide memory free percentage:"
    if marker in result.stdout:
        return int(result.stdout.split(marker, 1)[1].strip().rstrip("%"))
    return None


def _thermal() -> dict:
    result = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
    return {"available": result.returncode == 0, "detail": result.stdout.strip()[-1000:]}


def _normalized_value(value):
    volatile = {
        "source_commit",
        "sha256",
        "wall_seconds",
        "cpu_seconds",
        "observed_at",
        "timestamp",
        "created_at",
        "updated_at",
    }
    if isinstance(value, dict):
        return {key: _normalized_value(item) for key, item in sorted(value.items()) if key not in volatile}
    if isinstance(value, list):
        return [_normalized_value(item) for item in value]
    return value


def _artifact_hashes(root: Path) -> dict:
    import hashlib

    names = sorted({name for unit in _units() for name in unit.produces})
    hashes = {}
    for name in names:
        path = root / "evidence" / name
        if not path.is_file():
            hashes[name] = None
            continue
        if path.suffix == ".json":
            payload = json.dumps(_normalized_value(json.loads(path.read_text())), sort_keys=True, separators=(",", ":")).encode()
        else:
            payload = path.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def _waves(completed: set[str]) -> list:
    return [unit for unit in _units() if unit.identity not in completed and set(unit.depends_on) <= completed]


def _execute_model(root: Path, model: str, workers: int, thread_budget: int) -> list[dict]:
    records: list[dict] = []
    if model == "subprocess":
        for unit in _units():
            row = _run_subprocess(unit.identity, root, thread_budget)
            records.append(row)
            if not row["ok"]:
                break
        return records

    completed: set[str] = set()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_pool_initializer,
        initargs=(str(root), thread_budget),
    ) as pool:
        while len(completed) < len(_units()):
            ready = _waves(completed)
            if not ready:
                break
            exclusive = [unit for unit in ready if not unit.concurrency_safe]
            selected = exclusive[:1] if exclusive else ready[:workers]
            futures = {pool.submit(_pool_task, unit.identity, str(root), thread_budget): unit for unit in selected}
            wave = []
            for future in as_completed(futures):
                row = future.result()
                wave.append(row)
            for row in sorted(wave, key=lambda item: [unit.identity for unit in _units()].index(item["unit"])):
                records.append(row)
                if row["ok"]:
                    completed.add(row["unit"])
            if any(not row["ok"] for row in wave):
                break
    return records


def measure_once(model: str, workers: int, thread_budget: int) -> dict:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".benchmark-", dir=ARTIFACT_ROOT))
    _prepare(temporary)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_swap = _swap_used_mib()
    before_memory = _memory_free_percent()
    before_thermal = _thermal()
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_descendant_metrics, args=(os.getpid(), stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        units = _execute_model(temporary, model, workers, thread_budget)
        wall = time.perf_counter() - started
        hashes = _artifact_hashes(temporary)
        output_bytes = sum((temporary / "evidence" / name).stat().st_size for name in hashes if (temporary / "evidence" / name).is_file())
    finally:
        stop.set()
        sampler.join(timeout=2)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    after_swap = _swap_used_mib()
    document = {
        "model": model,
        "workers": workers,
        "thread_budget_per_worker": thread_budget,
        "native_thread_environment": _thread_budget_environment(thread_budget),
        "wall_seconds": wall,
        "user_cpu_seconds": after.ru_utime - before.ru_utime,
        "system_cpu_seconds": after.ru_stime - before.ru_stime,
        "total_cpu_seconds": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
        "cpu_utilization_percent": (100 * ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)) / wall if wall else 0),
        "effective_core_utilization": (((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)) / wall if wall else 0),
        "peak_process_count": max((sample["processes"] for sample in samples), default=0),
        "peak_thread_count": max((sample["threads"] for sample in samples), default=0),
        "peak_rss_mib": max((sample["rss_kib"] for sample in samples), default=0) / 1024,
        "block_input_operations": after.ru_inblock - before.ru_inblock,
        "block_output_operations": after.ru_oublock - before.ru_oublock,
        "output_bytes": output_bytes,
        "swap_delta_mib": None if before_swap is None or after_swap is None else after_swap - before_swap,
        "memory_free_percent_before": before_memory,
        "memory_free_percent_after": _memory_free_percent(),
        "thermal_before": before_thermal,
        "thermal_after": _thermal(),
        "units": units,
        "completed_units": sum(row["ok"] for row in units),
        "failed_units": [row["unit"] for row in units if not row["ok"]],
        "success": len(units) == len(_units()) and all(row["ok"] for row in units),
        "normalized_artifact_hashes": hashes,
        "sampled_process_tree": max(samples, key=lambda item: item["rss_kib"], default={"tree": []})["tree"],
    }
    shutil.rmtree(temporary, ignore_errors=True)
    return document


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault((record["model"], record["workers"], record["thread_budget_per_worker"]), []).append(record)
    rows = []
    for (model, workers, threads), trials in sorted(groups.items()):
        walls = [trial["wall_seconds"] for trial in trials if trial["success"]]
        cpus = [trial["total_cpu_seconds"] for trial in trials if trial["success"]]
        rows.append(
            {
                "model": model,
                "workers": workers,
                "thread_budget_per_worker": threads,
                "trials": len(trials),
                "successes": sum(trial["success"] for trial in trials),
                "failures": sum(not trial["success"] for trial in trials),
                "failure_rate": sum(not trial["success"] for trial in trials) / len(trials),
                "median_wall_seconds": statistics.median(walls) if walls else None,
                "p95_wall_seconds": _percentile(walls, 0.95) if walls else None,
                "median_total_cpu_seconds": statistics.median(cpus) if cpus else None,
                "throughput_units_per_second": len(_units()) / statistics.median(walls) if walls else 0,
                "peak_memory_mib": max((trial["peak_rss_mib"] for trial in trials), default=0),
                "peak_process_count": max((trial["peak_process_count"] for trial in trials), default=0),
                "peak_thread_count": max((trial["peak_thread_count"] for trial in trials), default=0),
                "swap_delta_mib": max((trial["swap_delta_mib"] or 0 for trial in trials), default=0),
                "deterministic_parity_within_configuration": len(
                    {json.dumps(trial["normalized_artifact_hashes"], sort_keys=True) for trial in trials if trial["success"]}
                )
                <= 1,
                "restart_loss_units": 1,
            }
        )
    return rows


def benchmark(repetitions: int) -> dict:
    configurations = [
        ("subprocess", 1, 1, "A sequential subprocess low"),
        ("subprocess", 1, 8, "A sequential subprocess medium"),
        ("subprocess", 1, 28, "A sequential subprocess full machine"),
        ("persistent", 1, 1, "B sequential persistent low"),
        ("persistent", 1, 8, "B sequential persistent medium"),
        ("persistent", 1, 28, "B sequential persistent full machine"),
        ("persistent", 2, 2, "C two workers conservative"),
        ("persistent", 2, 7, "C two workers balanced"),
        ("persistent", 2, 14, "C two workers aggressive"),
        ("persistent", 4, 1, "D four workers low"),
        ("persistent", 4, 4, "D four workers balanced"),
        ("persistent", 8, 1, "E adaptive candidate one thread"),
        ("persistent", 8, 2, "E adaptive candidate few threads"),
    ]
    records = []
    for model, workers, threads, label in configurations:
        for repetition in range(repetitions):
            record = measure_once(model, workers, threads)
            record["label"] = label
            record["repetition"] = repetition + 1
            records.append(record)
            print(
                json.dumps(
                    {
                        "label": label,
                        "repetition": repetition + 1,
                        "wall_seconds": round(record["wall_seconds"], 4),
                        "success": record["success"],
                        "peak_rss_mib": round(record["peak_rss_mib"], 1),
                    }
                ),
                flush=True,
            )
    summary = summarize(records)
    reference = next(row for row in summary if row["model"] == "subprocess" and row["workers"] == 1 and row["thread_budget_per_worker"] == 1)
    reference_hashes = next(
        record["normalized_artifact_hashes"]
        for record in records
        if record["model"] == "subprocess" and record["workers"] == 1 and record["thread_budget_per_worker"] == 1 and record["success"]
    )
    for row in summary:
        row["speedup_vs_reference"] = reference["median_wall_seconds"] / row["median_wall_seconds"] if row["median_wall_seconds"] else 0
        matching = [
            record
            for record in records
            if record["model"] == row["model"]
            and record["workers"] == row["workers"]
            and record["thread_budget_per_worker"] == row["thread_budget_per_worker"]
            and record["success"]
        ]
        row["artifact_parity_with_reference"] = all(record["normalized_artifact_hashes"] == reference_hashes for record in matching)
    document = {
        "schema": "substrate-worker-matrix/v1",
        "classification": "terminal deterministic synthesis benchmark; the scientific run was not launched",
        "machine": {"chip": "Apple M3 Ultra", "logical_cores": 28, "memory_gib": 96},
        "repetitions": repetitions,
        "thread_controls": {
            "set": list(THREAD_VARIABLES),
            "numpy_blas": "Accelerate",
            "applicability": {
                "VECLIB_MAXIMUM_THREADS": "applicable to the installed Accelerate build",
                "OMP_NUM_THREADS": "recorded; no OpenMP kernel was observed in this workload",
                "OPENBLAS_NUM_THREADS": "not applicable to the installed Accelerate build",
                "MKL_NUM_THREADS": "not applicable to the installed Accelerate build",
                "NUMEXPR_NUM_THREADS": "not applicable because NumExpr is not a dependency",
            },
        },
        "summary": summary,
        "trials": records,
        "scientific_run_launched": False,
        "activation": False,
    }
    _atomic_json(ARTIFACT_ROOT / "SUBSTRATE_WORKER_MATRIX.json", document)
    return document


def unit_command(args) -> None:
    print(json.dumps(_direct_unit(args.identity, Path(args.root), args.threads)))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    unit = subparsers.add_parser("unit")
    unit.add_argument("identity")
    unit.add_argument("--root", required=True)
    unit.add_argument("--threads", type=int, required=True)
    matrix = subparsers.add_parser("benchmark")
    matrix.add_argument("--repetitions", type=int, default=3)
    once = subparsers.add_parser("once")
    once.add_argument("--model", choices=("subprocess", "persistent"), required=True)
    once.add_argument("--workers", type=int, required=True)
    once.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    if args.command == "unit":
        unit_command(args)
    elif args.command == "benchmark":
        document = benchmark(args.repetitions)
        print(json.dumps({"configurations": len(document["summary"]), "trials": len(document["trials"])}, indent=2))
    elif args.command == "once":
        document = measure_once(args.model, args.workers, args.threads)
        print(
            json.dumps(
                {
                    key: document[key]
                    for key in (
                        "wall_seconds",
                        "completed_units",
                        "failed_units",
                        "success",
                        "peak_rss_mib",
                        "peak_process_count",
                        "peak_thread_count",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
