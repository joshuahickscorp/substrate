from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mop.temporal import io
from mop.temporal.runs import e2
PY, LOGS = sys.executable, io.ROOT / "logs"
CAP_OVERRIDE = int(os.environ["TEMPORAL_WORKERS"]) if os.environ.get("TEMPORAL_WORKERS") else None
CAP_SMALL, CAP_LARGE, PARTIAL_STALE_SECONDS = 12, 8, 300
ENV = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH="src")
BEDS = ("har_stream", "speech_stream", "harth_stream")
LOCKS = io.RUNS / "locks"
HEX = set("0123456789abcdef")
LEGACY_AUTHORITY = "b3f7421e6545527b3385f1368784ac2f0e1602a6"
LEGACY_BINDING = "legacy_receipt_hash_normalization_20260721.json"
def workers() -> int:
    r = subprocess.run(["pgrep", "-f", "mop.temporal.runs"], capture_output=True, text=True)
    return max(0, len([x for x in r.stdout.split() if x]) - 1, sum(lock_active(p.stem.replace("_", ":", 1)) for p in LOCKS.glob("*.json")) if LOCKS.is_dir() else 0)
def _lock_path(tag: str) -> Path: return LOCKS / f"{tag.replace(':', '_')}.json"
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
def launch_environment() -> tuple[dict, dict]:
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                            "src", "fastforge"], cwd=io.ROOT, capture_output=True, text=True)
    if dirty.returncode or dirty.stdout.strip():
        raise RuntimeError(f"refusing shard launch from dirty source tree: {dirty.stdout.strip()[:300]}")
    commit = io.commit()
    tree = subprocess.run(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=io.ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    if not _hex(commit, 40) or not _hex(tree, 40):
        raise RuntimeError("git did not return an exact commit and source tree binding")
    stable = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "fastforge"],
                            cwd=io.ROOT, capture_output=True, text=True)
    if stable.returncode or stable.stdout.strip() or io.commit() != commit:
        raise RuntimeError("source tree changed while launch provenance was being bound")
    binding = {"source_commit": commit, "source_tree_oid": tree}
    env = dict(ENV, MOP_LAUNCH_SOURCE_COMMIT=commit, MOP_LAUNCH_SOURCE_TREE_OID=tree)
    return env, binding
def scheduling_class(pending_extended: list[str]) -> tuple[int, list[str], str]:
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
        child_env, binding = launch_environment()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[supervisor] launch refused for {tag}: {exc}", flush=True)
        return False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps({"pid": os.getpid(), "tag": tag, "state": "reserved", **binding}).encode())
    os.close(fd)
    try:
        with open(LOGS / log, "a") as f:
            env = dict(child_env, TEMPORAL_SHARD_LOCK=str(lock), TEMPORAL_SHARD_TAG=tag)
            proc = subprocess.Popen([PY, "-m", module, *args], cwd=io.ROOT, env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
    except OSError as exc:
        lock.unlink(missing_ok=True)
        print(f"[supervisor] launch failed for {tag}: {exc}", flush=True)
        return False
    lock.write_text(json.dumps({"pid": proc.pid, "tag": tag, "args": args, "state": "active",
                                **binding}))
    return True
def run_sync(mod: str, args=()) -> bool:
    try:
        child_env, _ = launch_environment()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[{mod}] launch refused: {exc}", flush=True)
        return False
    r = subprocess.run([PY, "-m", mod, *args], cwd=io.ROOT, env=child_env,
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    print(f"[{mod} {' '.join(args)}] exit {r.returncode}: {' | '.join(tail)}", flush=True)
    return r.returncode == 0
def missing(sub: str, names: list[str]) -> list[str]:
    return [n for n in names if not (io.RUNS / sub / f"{n}.json").is_file()]
def _hex(value, size: int) -> bool: return isinstance(value, str) and len(value) == size and set(value.lower()) <= HEX
def _legacy_authorized() -> bool:
    p = io.RUNS / "orchestration" / LEGACY_BINDING
    try:
        d = json.loads(p.read_text())
        return all((d.get("schema") == "mop-temporal-orchestration-incident/v1",
            d.get("source_commit") == LEGACY_AUTHORITY, d.get("classification") ==
            "legacy_pre_persistence_object_hash_not_recomputable_after_json_normalization",
            "whole-file" in d.get("compatibility", ""), d.get("result_sha256") == io.sha_obj(
            {k: v for k, v in d.items() if k != "result_sha256"})))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
def _provenance_valid(doc: dict, allow_legacy: bool = False) -> bool:
    version = doc.get("result_hash_version")
    if version == "canonical_json_v2":
        return (doc.get("program") == io.PROGRAM and _hex(doc.get("source_commit"), 40) and
                _hex(doc.get("source_tree_oid"), 40) and doc.get("result_sha256") == io.sha_obj(
                {k: v for k, v in doc.items() if k != "result_sha256"}))
    if version is not None:
        return False
    source = doc.get("source_commit")
    return (allow_legacy and (source is None or _hex(source, 40)) and doc.get("source_tree_oid") is None
            and _legacy_authorized())
def _named_bed_seed(name: str, prefix: str = "") -> tuple[str, int]:
    bed, seed = name.removeprefix(prefix).rsplit("_", 1)
    if bed not in BEDS: raise ValueError("undeclared bed")
    return bed, int(seed)
def _runs_exact(doc: dict, bed: str, seed: int, cells=None) -> None:
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(r, dict) or r.get("bed") != bed
                                                     or r.get("seed") != seed for r in runs): raise ValueError("run identity mismatch")
    if cells is not None and ({r.get("cell") for r in runs} != set(cells) or len(runs) != len(cells)): raise ValueError("run cell set mismatch")
def _convergence_binding(doc: dict, bed: str, cells) -> None:
    from mop.temporal import factorial as Fx
    cells = set(cells)
    authority = doc.get("convergence_authority")
    path = io.RUNS / "e2_converge" / f"converge_{bed}.json"
    expected_path = path.relative_to(io.ROOT).as_posix()
    if (not isinstance(authority, dict) or authority.get("path") != expected_path or
            not _hex(authority.get("sha256"), 64) or authority["sha256"] != io.sha_file(path)): raise ValueError("convergence authority path or hash mismatch")
    convergence = json.loads(path.read_text())
    configs = convergence.get("configs") if isinstance(convergence, dict) else None
    checkpoints = authority.get("selected_checkpoints")
    all_cells = {Fx.cell_name(**spec) for spec in Fx.sweep_cells()["_all"]}
    if (not isinstance(configs, dict) or set(configs) != all_cells or convergence.get("bed") != bed or
            convergence.get("schema") != "mop-e2-convergence/v4" or
            not isinstance(convergence.get("shard_index"), list) or len(convergence["shard_index"]) != len(all_cells) or
            not _provenance_valid(convergence) or not isinstance(checkpoints, dict) or set(checkpoints) != cells): raise ValueError("convergence authority inventory mismatch")
    expected = {cell: int(configs[cell]["selected_checkpoint"]) for cell in cells}
    if checkpoints != expected or any(r.get("steps") != expected[r["cell"]] or
                                      r.get("updates") != expected[r["cell"]] for r in doc["runs"]): raise ValueError("principal budget is not the selected convergence checkpoint")
def _curve_exact(doc: dict, grid) -> None:
    curve = doc.get("curve")
    if not isinstance(curve, dict) or {int(k) for k in curve} != set(grid): raise ValueError("curve grid mismatch")
    if doc.get("selected_checkpoint") not in set(grid) or not isinstance(doc.get("converged"), bool): raise ValueError("curve decision shape invalid")
    scores, records = doc.get("seed_scores"), doc.get("arm_records")
    if (not isinstance(scores, dict) or not isinstance(records, dict) or
            {int(k) for k in scores} != set(grid) or {int(k) for k in records} != set(grid)): raise ValueError("raw convergence grid missing")
    get = lambda mapping, budget: mapping.get(str(budget), mapping.get(budget))
    for budget in grid:
        values, rows = get(scores, budget), get(records, budget)
        if (not isinstance(values, list) or not isinstance(rows, list) or len(values) != len(e2.CONVERGENCE_SEEDS)
                or [r.get("seed") for r in rows] != list(e2.CONVERGENCE_SEEDS)
                or any(r.get("updates") != budget or r.get("score") != score or
                       not _hex(r.get("checkpoint_sha"), 64) for r, score in zip(rows, values, strict=True))):
            raise ValueError("raw convergence arm record mismatch")
def _stage_valid(sub: str, name: str, doc: dict) -> None:
    from mop.temporal import factorial as Fx
    if sub in ("e2_scout", "e2_principal"):
        if sub == "e2_scout" and name.startswith("scout_"):
            bed, runs = name.removeprefix("scout_"), doc.get("runs")
            cells = {Fx.cell_name(**s) for s in Fx.sweep_cells()["_all"]}
            shards = [json.loads((io.RUNS / sub / f"shard_{bed}_{s}.json").read_text())
                      for s in e2.SCOUT_SEEDS]
            for seed, shard in zip(e2.SCOUT_SEEDS, shards, strict=True):
                if not _provenance_valid(shard): raise ValueError("scout aggregate has unbound shard")
                _stage_valid(sub, f"shard_{bed}_{seed}", shard)
            expected_runs = [row for shard in shards for row in shard["runs"]]
            if (bed not in BEDS or doc.get("schema") != "mop-e2-scout/v1" or doc.get("bed") != bed or
                    doc.get("seeds") != list(e2.SCOUT_SEEDS) or doc.get("n_cells") != len(cells) or
                    runs != expected_runs): raise ValueError("scout aggregate is stale or incomplete")
            return
        prefix = "shard_" if sub == "e2_scout" else ""
        bed, seed = _named_bed_seed(name, prefix)
        if doc.get("bed") != bed or doc.get("seed") != seed: raise ValueError("receipt identity mismatch")
        cells = [Fx.cell_name(**s) for s in Fx.sweep_cells()["_all"]]
        _runs_exact(doc, bed, seed, cells)
        if sub == "e2_principal":
            if doc.get("schema") != "mop-e2-principal-shard/v2" or doc.get("n_cells") != len(cells): raise ValueError("principal schema or cell count mismatch")
            _convergence_binding(doc, bed, cells)
    elif sub in ("e2_converge", "e2_converge_extended"):
        if sub == "e2_converge" and name.startswith("converge_"):
            bed = name.removeprefix("converge_")
            configs = doc.get("configs")
            expected = {Fx.cell_name(**spec) for spec in e2.CONVERGE_CONFIGS}
            if (bed not in BEDS or doc.get("schema") != "mop-e2-convergence/v4" or doc.get("bed") != bed or
                    not isinstance(configs, dict) or set(configs) != expected or
                    doc.get("seeds_per_budget") != list(e2.CONVERGENCE_SEEDS)): raise ValueError("convergence aggregate identity mismatch")
            selected = []
            for idx, spec in enumerate(e2.CONVERGE_CONFIGS):
                cell = Fx.cell_name(**spec)
                correction = e2.CORRECTED_CONVERGENCE_CELLS.get(cell)
                candidates = ([((io.RUNS / "e2_converge_corrections" / correction.format(bed=bed)),
                    "e2_converge_corrections", f"convergence_{bed}")] if correction else []) + [
                    (io.RUNS / "e2_converge_extended" / f"xshard_{bed}_{idx}.json", "e2_converge_extended",
                     f"xshard_{bed}_{idx}"), (io.RUNS / "e2_converge" / f"cshard_{bed}_{idx}.json",
                     "e2_converge", f"cshard_{bed}_{idx}")]
                source, source_sub, source_name = next((r for r in candidates if r[0].is_file()), (None,) * 3)
                raw = json.loads(source.read_text()) if source else None
                if not isinstance(raw, dict) or not _provenance_valid(raw) or configs[cell] != raw: raise ValueError("convergence aggregate is stale or unbound")
                _stage_valid(source_sub, source_name, raw)
                selected.append({"cell": cell, "path": source.relative_to(io.ROOT).as_posix(),
                                 "sha256": io.sha_file(source)})
            if doc.get("shard_index") != selected: raise ValueError("convergence aggregate shard index mismatch")
            return
        prefix = "cshard_" if sub == "e2_converge" else "xshard_"
        bed_idx = name.removeprefix(prefix)
        bed, idx_text = bed_idx.rsplit("_", 1)
        idx = int(idx_text)
        if bed not in BEDS or not 0 <= idx < len(e2.CONVERGE_CONFIGS): raise ValueError("convergence bounds invalid")
        spec, expected = e2.CONVERGE_CONFIGS[idx], Fx.cell_name(**e2.CONVERGE_CONFIGS[idx])
        if (doc.get("bed") != bed or doc.get("spec") != spec or doc.get("cell") != expected or
                doc.get("seeds") != list(e2.CONVERGENCE_SEEDS)): raise ValueError("convergence index identity mismatch")
        grid = e2.CONVERGENCE_GRID if sub == "e2_converge" else \
            e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID
        _curve_exact(doc, grid)
        if sub == "e2_converge_extended":
            extension = doc.get("extends")
            base_path = io.RUNS / "e2_converge" / f"cshard_{bed}_{idx}.json"
            expected_path = base_path.relative_to(io.ROOT).as_posix()
            if (not isinstance(extension, dict) or extension.get("path") != expected_path or
                    not _hex(extension.get("sha256"), 64) or extension.get("grid") != list(e2.CONVERGENCE_GRID)): raise ValueError("extended convergence dependency shape invalid")
            base = json.loads(base_path.read_text())
            if extension["sha256"] != io.sha_file(base_path): raise ValueError("extended convergence dependency hash mismatch")
            if not isinstance(base, dict) or not _provenance_valid(base): raise ValueError("extended convergence base provenance invalid")
            _stage_valid("e2_converge", f"cshard_{bed}_{idx}", base)
    elif sub == "e2_principal_corrections":
        from mop.temporal.runs import corrections
        bed, seed = _named_bed_seed(name, "capacity_")
        if (doc.get("schema") != "mop-e2-capacity-tier-correction-shard/v2" or
                doc.get("bed") != bed or doc.get("seed") != seed): raise ValueError("principal correction identity mismatch")
        cells = [Fx.cell_name(**s) for s in corrections.SPECS]
        _runs_exact(doc, bed, seed, cells)
        _convergence_binding(doc, bed, cells)
    elif sub == "e2_converge_corrections":
        from mop.temporal.runs import corrections
        bed = name.removeprefix("convergence_")
        if (bed not in BEDS or doc.get("bed") != bed or doc.get("spec") != corrections.LINEAR or
                doc.get("cell") != Fx.cell_name(**corrections.LINEAR) or
                doc.get("seeds") != list(e2.CONVERGENCE_SEEDS)): raise ValueError("convergence correction identity mismatch")
        _curve_exact(doc, e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID)
    elif sub == "e2_optimization_corrections":
        from mop.temporal.runs import corrections
        bed, tier = name.removeprefix("optimization_").rsplit("_", 1)
        spec = corrections.OPTIMIZATION_SPECS.get(tier)
        roles = doc.get("four_contrast_roles")
        expected_roles = {f"{tier}_model_at_strict_selected_convergence",
            "large_model_at_same_update_count" if tier == "large" else "small_model_at_same_compute"}
        if (bed not in BEDS or spec is None or doc.get("bed") != bed or doc.get("tier") != tier or
                doc.get("spec") != spec or doc.get("cell") != Fx.cell_name(**spec) or
                doc.get("seeds") != list(e2.CONVERGENCE_SEEDS) or
                not isinstance(roles, dict) or set(roles) != expected_roles): raise ValueError("optimization correction identity mismatch")
        regular = e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID
        match, arm_records = doc.get("compute_match"), doc.get("arm_records")
        extra = match.get("small_steps") if isinstance(match, dict) and tier == "small" else None
        measured = set(regular) | ({extra} if isinstance(extra, int) else set())
        if (doc.get("regular_grid") != list(regular) or not isinstance(arm_records, dict) or
                {int(k) for k in arm_records} != measured or set(map(int, doc.get("curve", {}))) != measured): raise ValueError("optimization measured grid mismatch")
        for role in roles.values():
            records, budget = (role.get("records"), role.get("budget")) if isinstance(role, dict) else (None, None)
            if (budget not in measured or records != arm_records.get(str(budget), arm_records.get(budget)) or
                    not isinstance(records, list) or [r.get("seed") for r in records] != list(e2.CONVERGENCE_SEEDS)): raise ValueError("optimization role records mismatch")
    elif sub == "e3":
        from mop.temporal.runs import e3
        direction, seed_text = name.rsplit("_", 1)
        source, target = direction.split("_to_", 1)
        arms = doc.get("arms")
        if ((source, target) not in e3.DIRECTIONS or int(seed_text) not in e3.SEEDS or
                doc.get("source_bed") != source or doc.get("target_bed") != target or
                doc.get("seed") != int(seed_text) or not isinstance(arms, dict) or set(arms) != set(e3.ARMS) or
                not all(isinstance(arms[a], dict) for a in e3.ARMS) or
                not isinstance(doc.get("source_training"), dict) or
                not isinstance(doc.get("target_training_match"), dict)): raise ValueError("E3 identity or arm mismatch")
    elif sub == "third_bed_preflight":
        seed = int(name.removeprefix("harth_preflight_"))
        if (seed not in e2.PRINCIPAL_SEEDS or doc.get("bed") != "harth_stream" or doc.get("seed") != seed or
                not isinstance(doc.get("receipts"), dict) or set(doc["receipts"]) != {
                "pretrain", "pooled_control_pretrain", "adapt_B", "recover_A_readout"} or
                not isinstance(doc.get("temporal_order_permutation"), dict)): raise ValueError("third bed identity mismatch")
    elif sub == "hybrid":
        from mop.temporal.runs import hybrid
        bed, seed = _named_bed_seed(name)
        arms = doc.get("arms")
        if (bed not in hybrid.BEDS or seed not in hybrid.SEEDS or doc.get("bed") != bed or
                doc.get("seed") != seed or not isinstance(doc.get("selected_core_spec"), dict) or
                not isinstance(arms, dict) or set(arms) != set(hybrid.ARMS) or any(not isinstance(arms[a], dict)
                or not isinstance(arms[a].get("trace"), dict) for a in hybrid.ARMS)): raise ValueError("hybrid identity or arm mismatch")
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
            allow_legacy = sub not in {"e2_scout", "e2_principal", "e2_converge", "e2_converge_extended",
                "e2_principal_corrections", "e2_converge_corrections", "e2_optimization_corrections", "e3",
                "third_bed_preflight", "hybrid"}
            if not _provenance_valid(doc, allow_legacy):
                raise ValueError("receipt provenance invalid")
            _stage_valid(sub, n, doc)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            out.append(n)
    return out
def partials(sub: str) -> list[str]:
    d = io.RUNS / sub; return sorted(p.name for p in d.glob(".*.partial.*")) if d.is_dir() else []
def _relative(path: Path) -> str:
    try:
        return path.relative_to(io.ROOT).as_posix()
    except ValueError:
        return path.relative_to(io.RUNS).as_posix()
def _quarantine(source: Path, sub: str, identity: str, kind: str) -> None:
    digest = io.sha_file(source)
    quarantine = io.RUNS / "quarantine" / kind
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"{sub}_{identity}_{digest[:12]}{source.suffix}"
    if target.exists():
        target = quarantine / f"{sub}_{identity}_{digest[:12]}_{time.time_ns()}{source.suffix}"
    source.rename(target)
    io.run_json(f"quarantine_{kind}_{target.stem}.json", {
        "schema": "mop-temporal-receipt-quarantine/v2", "stage": sub, "identity": identity,
        "original_path": _relative(source), "quarantine_path": _relative(target), "sha256": digest,
        "classification": f"{kind}_excluded_pending_replacement",
        "preservation": "append only move; original bytes retained at quarantine_path"}, "orchestration")
    print(f"[supervisor] quarantined {kind} {sub}/{identity} as {target.name}", flush=True)
def quarantine_stale_partials(sub: str, stale_after: int = PARTIAL_STALE_SECONDS) -> list[str]:
    moved = []
    now = time.time()
    for name in partials(sub):
        source = io.RUNS / sub / name
        try:
            pid = int(name.rsplit(".", 1)[1])
        except (ValueError, IndexError):
            pid = 0
        try:
            stale = now - source.stat().st_mtime >= stale_after
        except OSError:
            continue
        if not stale or (pid and _pid_alive(pid)):
            continue
        identity = name.removeprefix(".").split(".json.partial.", 1)[0]
        _quarantine(source, sub, identity, "stale_partials")
        moved.append(name)
    return moved
def recoverable_pending(sub: str, names: list[str]) -> list[str]:
    quarantine_stale_partials(sub)
    for name in invalid(sub, names):
        source = io.RUNS / sub / f"{name}.json"
        _quarantine(source, sub, name, "invalid_receipts")
    return missing(sub, names)
def receipt_plan() -> dict[str, tuple[str, list[str]]]:
    seeds, configs = e2.PRINCIPAL_SEEDS, range(len(e2.CONVERGE_CONFIGS))
    return {
        "scout": ("e2_scout", [f"shard_{b}_{s}" for b in BEDS for s in e2.SCOUT_SEEDS]),
        "scout_aggregates": ("e2_scout", [f"scout_{b}" for b in BEDS]),
        "convergence": ("e2_converge", [f"cshard_{b}_{i}" for b in BEDS for i in configs]),
        "extended_convergence": ("e2_converge_extended", [f"xshard_{b}_{i}" for b in BEDS for i in configs]),
        "convergence_aggregates": ("e2_converge", [f"converge_{b}" for b in BEDS]),
        "principal": ("e2_principal", [f"{b}_{s}" for b in BEDS for s in seeds]),
        "principal_corrections": ("e2_principal_corrections",
                                  [f"capacity_{b}_{s}" for b in BEDS for s in seeds]),
        "convergence_corrections": ("e2_converge_corrections", [f"convergence_{b}" for b in BEDS]),
        "optimization_corrections": ("e2_optimization_corrections",
                                     [f"optimization_{b}_{t}" for b in BEDS for t in ("small", "large")]),
        "e3": ("e3", [f"{a}_to_{b}_{s}" for a, b in (("har_stream", "speech_stream"),
                 ("speech_stream", "har_stream")) for s in seeds]),
        "third_bed_preflight": ("third_bed_preflight", [f"harth_preflight_{s}" for s in seeds]),
        "hybrid": ("hybrid", [f"{b}_{s}" for b in e2.B.PRINCIPAL for s in seeds])}
def status() -> dict:
    plan = receipt_plan()
    absent = {key: missing(sub, names) for key, (sub, names) in plan.items()}
    bad = {key: invalid(sub, names) for key, (sub, names) in plan.items()}
    active = []
    if LOCKS.is_dir():
        for p in sorted(LOCKS.glob("*.json")):
            tag = p.stem.replace("_", ":", 1)
            if lock_active(tag):
                active.append(json.loads(p.read_text()))
    return {"schema": "mop-temporal-supervisor-status/v1", "stop_switch_active": io.STOP.exists(),
        "process_workers": workers(), "active_shards": active,
        "completed": {key: len(names) - len(absent[key]) - len(bad[key]) for key, (_, names) in plan.items()},
        "missing": absent,
        "invalid": bad,
        "partial_receipts": {key: partials(sub) for key, (sub, _) in plan.items()},
        "orchestration_incidents": sorted(
            p.name for p in (io.RUNS / "orchestration").glob("*.json")
        ) if (io.RUNS / "orchestration").is_dir() else [],
    }
def _load_field(name: str, field: str) -> bool:
    try:
        return io.exists(name) and io.load(name).get(field) is True
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return False
def reconcile_live_state(phase: str, snapshot=None) -> dict:
    from mop.temporal.runs.synthesis import stage_dependencies
    snapshot = snapshot or status()
    clean = lambda key: not (snapshot["missing"][key] or snapshot["invalid"][key]
                             or snapshot["partial_receipts"][key])
    stages = {
        "start_authority": io.exists("MOP_TEMPORAL_CORE_START_AUTHORITY.json"), "binding_results": io.exists("MOP_TEMPORAL_CORE_BINDING_RESULTS.json"),
        "data_custody": io.exists("MOP_DATA_CUSTODY_AUTHORITY.json"), "code_lifecycle": io.exists("MOP_CODE_LIFECYCLE_AUTHORITY.json"),
        "method_extension": io.exists("MOP_TEMPORAL_METHOD_EXTENSION.json"),
        "e2_calibration": _load_field("MOP_E2_CALIBRATION.json", "all_pass"),
        "scout": clean("scout") and clean("scout_aggregates"),
        "convergence": clean("convergence") and clean("extended_convergence") and clean("convergence_aggregates"),
        "third_bed_preflight": clean("third_bed_preflight")
        and io.exists("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json"),
        "e2_principal": clean("principal"),
        "capacity_corrections": clean("principal_corrections")
        and clean("convergence_corrections") and clean("optimization_corrections")
        and _load_field("MOP_E2_CAPACITY_TIER_CORRECTION.json", "all_pass"),
        "bed_validity": io.exists("MOP_E2_FACTORIAL_AUTHORITY.json"), "independent_replication": _load_field("MOP_E2_INDEPENDENT_REPLICATION.json", "all_pass"),
        "mutations": _load_field("MOP_TEMPORAL_CORE_MUTATION_REPORT.json", "all_rejected"),
        "verification": _load_field("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "all_pass"),
        "core_selection": io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json"), "successor_gates": io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json"),
        "successors_terminal": all(io.exists(n) for n in (
            "MOP_E3_SHARED_LOCAL_RESULT.json", "MOP_E5_SELF_SUPERVISED_RESULT.json",
            "MOP_HYBRID_ADAPTATION_RESULT.json")),
        "tests_and_coverage": _load_field("MOP_TEMPORAL_CORE_TEST_REPORT.json", "passed")
        and io.exists("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json"),
        "preclone_deliverables": False, "clean_clone": _load_field("MOP_TEMPORAL_CORE_CLEAN_CLONE.json", "all_pass"),
        "evidence_fabric": io.exists("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"), "terminal_metadata": False,
    }
    dependencies = stage_dependencies()
    for name in stages:
        stages[name] = stages[name] and all(stages.get(dep, False) for dep in dependencies[name])
    ready = sorted(name for name, value in stages.items() if not value
                   and all(stages.get(dep, False) for dep in dependencies[name]))
    items = {f"stage:{name}": {"status": "terminal" if value else "incomplete",
             "dependencies": [f"stage:{dep}" for dep in dependencies[name]],
             "next_action": "none" if value else "resume supervisor stage"}
             for name, value in stages.items()}
    for key in snapshot["missing"]:
        items[f"receipts:{key}"] = {
            "status": "terminal" if clean(key) else "incomplete", "dependencies": [],
            "missing": snapshot["missing"][key], "invalid": snapshot["invalid"][key],
            "partial": snapshot["partial_receipts"][key],
            "next_action": "none" if clean(key) else "quarantine invalid or stale partial and resume"}
    doc = {"schema": "mop-temporal-core-state/v2", "program_id": io.PROGRAM,
           "mode": "live_supervisor_reconciliation", "phase": phase, "stages": stages,
           "stage_dependencies": dependencies, "dependency_ready": ready, "items": items,
           "active_shards": snapshot["active_shards"], "stop_switch_active": snapshot["stop_switch_active"],
           "all_terminal": False, "no_dependency_ready_work": not ready,
           "resume": "python3.12 -m mop.temporal.runs.supervisor resumes exact receipt identities",
           "activation": False}
    io.seal("MOP_TEMPORAL_CORE_STATE.json", doc)
    return doc
def _run_and_reconcile(mod: str, args=(), phase: str | None = None) -> bool:
    ok = run_sync(mod, args)
    if not mod.endswith((".synthesis", ".fabric")):
        reconcile_live_state(phase or f"after:{mod}")
    return ok
def run_recoverable_all(mod: str, sub: str, names: list[str]) -> bool:
    while not io.STOP.exists():
        pending = recoverable_pending(sub, names)
        if not pending and not partials(sub):
            return True
        if pending and not _run_and_reconcile(mod, ["all"], f"after:{sub}"):
            return False
        if not pending: time.sleep(5)
    return False
def run_recoverable_phase(mod: str, args: list[str], specs: list[tuple[str, list[str]]]) -> bool:
    while not io.STOP.exists():
        pending = sum((recoverable_pending(sub, names) for sub, names in specs), [])
        dirty = any(partials(sub) for sub, _ in specs)
        if not pending and not dirty: return True
        if pending and not _run_and_reconcile(mod, args): return False
        if not pending: time.sleep(5)
    return False
def run_e3() -> bool:
    from mop.temporal.runs import e3
    names = [f"{source}_to_{target}_{seed}" for source, target in e3.DIRECTIONS for seed in e3.SEEDS]
    started: set[str] = set()
    while not io.STOP.exists():
        started = {tag for tag in started if lock_active(tag)}
        pending = recoverable_pending("e3", names)
        if not pending and not partials("e3"):
            return _run_and_reconcile("mop.temporal.runs.e3", ["aggregate"], "after:e3")
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
        reconcile_live_state("e3_shard_loop")
        time.sleep(60)
    return False
def _core_selected() -> bool: return _load_field("MOP_OWNED_TEMPORAL_CORE_V1.json", "selected")
def quarantine_owned_checkpoints() -> bool:
    root = io.PROOF / "checkpoints"
    sources = sorted(root.glob("owned_temporal_core_v1_*.pt")) if root.is_dir() else []
    if not sources: return True
    incident = f"withdrawal_{time.time_ns()}"
    target_root = io.PROOF / "checkpoint_quarantine" / incident
    try:
        target_root.mkdir(parents=True)
        entries = []
        for source in sources:
            digest, target = io.sha_file(source), target_root / source.name
            source.rename(target)
            entries.append({"original_path": source.relative_to(io.ROOT).as_posix(),
                            "quarantine_path": target.relative_to(io.ROOT).as_posix(), "sha256": digest})
        io.seal("incident.json", {"schema": "mop-temporal-checkpoint-withdrawal/v1",
            "classification": "failed_selection_checkpoint_withdrawal", "checkpoints": entries,
            "preservation": "append only move; packaged bytes retained", "activation": False},
            f"checkpoint_quarantine/{incident}")
        return not list(root.glob("owned_temporal_core_v1_*.pt"))
    except (OSError, ValueError) as exc:
        print(f"[supervisor] checkpoint withdrawal failed closed: {exc}", flush=True)
        return False
def prepare_verified_core() -> bool:
    if not run_sync("mop.temporal.runs.coresel"):
        return False
    verifier_ran = run_sync("mop.temporal.runs.verify")
    if verifier_ran and _load_field("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "all_pass"):
        return True
    print("[supervisor] post core verification failed; withdrawing the selection", flush=True)
    if not quarantine_owned_checkpoints(): return False
    selector_ran = run_sync("mop.temporal.runs.coresel")
    withdrawn = selector_ran and not _core_selected()
    verifier_reran = run_sync("mop.temporal.runs.verify")
    return bool(withdrawn and verifier_reran and _load_field(
        "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "all_pass"))
def run_verified_successor_gates() -> bool:
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
    scout_aggregates = [f"scout_{b}" for b in BEDS]
    conv_shards = [f"cshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    ext_shards = [f"xshard_{b}_{i}" for b in BEDS for i in range(len(e2.CONVERGE_CONFIGS))]
    conv_aggregates = [f"converge_{b}" for b in BEDS]
    principal_shards = [f"{b}_{s}" for b in BEDS for s in e2.PRINCIPAL_SEEDS]
    pre_c_names, pre_o_names = receipt_plan()["convergence_corrections"][1], receipt_plan()["optimization_corrections"][1]
    pre_specs = [("e2_converge_corrections", pre_c_names), ("e2_optimization_corrections", pre_o_names)]
    started: set[str] = set()
    reconcile_live_state("supervisor_start")
    startup = (("mop.temporal.runs.authority", "MOP_TEMPORAL_CORE_START_AUTHORITY.json"),
               ("mop.temporal.runs.custody_run", "MOP_DATA_CUSTODY_AUTHORITY.json"),
               ("mop.temporal.runs.method_ext", "MOP_TEMPORAL_METHOD_EXTENSION.json"),
               ("mop.temporal.runs.codelife", None))
    for mod, artifact in startup:
        if (artifact is None or not io.exists(artifact)) and not _run_and_reconcile(mod):
            print(f"[supervisor] required startup stage failed: {mod}", flush=True)
            return
    if not io.exists("MOP_E2_CALIBRATION.json"):
        if not _run_and_reconcile("mop.temporal.runs.e2", ["calibration"]):
            return
    if not _load_field("MOP_E2_CALIBRATION.json", "all_pass"):
        print("[supervisor] calibration is not green, principal execution stays closed", flush=True)
        return
    while not io.STOP.exists():
        started = {tag for tag in started if lock_active(tag)}
        pending_scout = recoverable_pending("e2_scout", scout_shards)
        pending_scout_aggregates = recoverable_pending("e2_scout", scout_aggregates)
        pending_conv = recoverable_pending("e2_converge", conv_shards)
        pending_ext = recoverable_pending("e2_converge_extended", ext_shards)
        pending_aggregates = recoverable_pending("e2_converge", conv_aggregates)
        pending_principal = recoverable_pending("e2_principal", principal_shards)
        pending_pre_c = recoverable_pending("e2_converge_corrections", pre_c_names)
        pending_pre_o = recoverable_pending("e2_optimization_corrections", pre_o_names)
        base_partials = sum(len(partials(sub)) for sub in ("e2_scout", "e2_converge",
                            "e2_converge_extended", "e2_principal"))
        pre_partials = sum(len(partials(sub)) for sub, _ in pre_specs)
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
                if f"scout_{b}" in pending_scout_aggregates:
                    if not _run_and_reconcile("mop.temporal.runs.e2", ["scout", b]):
                        return
                if f"converge_{b}" in pending_aggregates:
                    if not _run_and_reconcile("mop.temporal.runs.e2", ["converge", b]):
                        return
            pending_aggregates = recoverable_pending("e2_converge", conv_aggregates)
            pending_scout_aggregates = recoverable_pending("e2_scout", scout_aggregates)
            if not pending_aggregates and not pending_scout_aggregates and (pending_pre_c or pending_pre_o or pre_partials):
                if not run_recoverable_phase("mop.temporal.runs.corrections", ["preprincipal"], pre_specs): return
                pending_pre_c = recoverable_pending("e2_converge_corrections", pre_c_names)
                pending_pre_o = recoverable_pending("e2_optimization_corrections", pre_o_names)
                pending_aggregates = recoverable_pending("e2_converge", conv_aggregates)
                pre_partials = sum(len(partials(sub)) for sub, _ in pre_specs)
            if not pending_aggregates and not pending_scout_aggregates and not pending_pre_c and not pending_pre_o and not base_partials and not pre_partials:
                for n in pending_principal:
                    tag = f"p:{n}"
                    if tag in started or free <= 0:
                        continue
                    b, s = "_".join(n.split("_")[:-1]), n.split("_")[-1]
                    if launch(["principal", b, s], f"e2_principal_{b}.log", tag):
                        started.add(tag)
                        free -= 1
        rem = {"scout": len(pending_scout), "converge": len(pending_conv), "extended": len(pending_ext),
               "aggregates": len(pending_aggregates) + len(pending_scout_aggregates),
               "preprincipal_corrections": len(pending_pre_c) + len(pending_pre_o),
               "partials": base_partials + pre_partials, "principal": len(pending_principal)}
        print(f"[supervisor] workers={workers()} cap={cap} class={resource_class} remaining={rem}", flush=True)
        reconcile_live_state("e2_shard_loop")
        if not any(rem.values()):
            break
        time.sleep(60)
    remaining = status()["missing"]
    invalid_receipts = status()["invalid"]
    partial_receipts = status()["partial_receipts"]
    terminal_keys = ("scout", "scout_aggregates", "convergence", "extended_convergence",
                     "convergence_aggregates", "principal")
    scientific_missing = any(remaining[k] for k in terminal_keys)
    if io.STOP.exists() or scientific_missing or any(invalid_receipts[k] for k in (
            terminal_keys)) or any(partial_receipts[k] for k in terminal_keys):
        print("[supervisor] stopped before downstream aggregation because shard work is not terminal", flush=True)
        return
    if not _run_and_reconcile("mop.temporal.runs.e2", ["scout_result"]):
        return
    third_names = [f"harth_preflight_{seed}" for seed in e2.PRINCIPAL_SEEDS]
    if not run_recoverable_all("mop.temporal.runs.thirdbed", "third_bed_preflight", third_names):
        return
    if not _run_and_reconcile("mop.temporal.runs.thirdbed", ["aggregate"]):
        return
    post_names = receipt_plan()["principal_corrections"][1]
    if not run_recoverable_phase("mop.temporal.runs.corrections", ["postprincipal"],
                                 [("e2_principal_corrections", post_names)]): return
    if not _run_and_reconcile("mop.temporal.runs.corrections", ["aggregate"]): return
    for mod in ("mop.temporal.runs.bedvalid", "mop.temporal.runs.analyze",
                "mop.temporal.runs.replicate", "mop.temporal.runs.mutations",
                "mop.temporal.runs.verify", "mop.temporal.runs.analyze", "mop.temporal.runs.verify"):
        if not _run_and_reconcile(mod):
            print(f"[supervisor] required downstream stage failed: {mod}", flush=True)
            return
    if not run_verified_successor_gates():
        return
    reconcile_live_state("after:successor_gates")
    queue = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json") if io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json") else {}
    if "E3_shared_versus_local" in (queue.get("licensed_top_two") or []):
        if not run_e3():
            return
        for mod in ("mop.temporal.runs.mutations", "mop.temporal.runs.verify",
                    "mop.temporal.runs.successors"):
            if not _run_and_reconcile(mod):
                return
    if "hybrid_adaptation" in (queue.get("licensed_top_two") or []):
        hybrid_names = [f"{bed}_{seed}" for bed in e2.B.PRINCIPAL for seed in e2.PRINCIPAL_SEEDS]
        if not run_recoverable_all("mop.temporal.runs.hybrid", "hybrid", hybrid_names):
            return
        if not _run_and_reconcile("mop.temporal.runs.hybrid", ["aggregate"]):
            return
        for mod in ("mop.temporal.runs.mutations", "mop.temporal.runs.verify",
                    "mop.temporal.runs.successors"):
            if not _run_and_reconcile(mod):
                return
    for mod in ("mop.temporal.runs.reports", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric", "mop.temporal.runs.synthesis",
                "mop.temporal.runs.fabric"):
        if not _run_and_reconcile(mod):
            print(f"[supervisor] required terminal artifact stage failed: {mod}", flush=True)
            return
    terminal = io.load("MOP_TEMPORAL_CORE_SYNTHESIS.json") if io.exists(
        "MOP_TEMPORAL_CORE_SYNTHESIS.json") else {}
    print("SUPERVISOR_DONE" if terminal.get("all_terminal") else
          "SUPERVISOR_PRECLONE_DONE", flush=True)
if __name__ == "__main__":
    main()
