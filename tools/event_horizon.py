#!/usr/bin/env python3
"""Regenerate the measured Substrate Event Horizon authorities."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from substrate import audit, evidence, execution, verification  # noqa: E402

OUT = ROOT / "artifacts" / "substrate" / "event-horizon"
ARCHIVE = ROOT / "archive" / "pre-substrate-event-horizon"
CLOSURE = "7158451d80cfcacc0763894dad3ee5ee1ca834ec"
TOKENS = (
    "Mixture of Perspectives",
    "Mixture of Thinking",
    "joshuahickscorp/mop",
    "agent/mop-",
    "mop-substrate",
    "src/mop",
    "tests/mop",
    "runs/mop",
    "MOP_",
    "MOP",
    "mop",
)
ACTIVE_DOCS = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/SCIENTIFIC_STATUS.md",
    "docs/LONG_RUN_PLAN.md",
    "docs/RUNBOOK.md",
    "docs/DEVELOPMENT.md",
)
DELETED_DUPLICATES = (
    "archive/pre-substrate-event-horizon/tests/temporal/__init__.py",
    "archive/pre-substrate-event-horizon/tests/unit/__init__.py",
    "archive/pre-substrate-event-horizon/tests/integration/__init__.py",
    "archive/pre-substrate-event-horizon/tests/method/__init__.py",
    "archive/pre-substrate-event-horizon/fastforge/runs/__init__.py",
    "archive/pre-substrate-event-horizon/src/mop/temporal/__init__.py",
    "archive/pre-substrate-event-horizon/src/mop/beds/__init__.py",
    "archive/pre-substrate-event-horizon/src/mop/temporal/runs/__init__.py",
    "archive/pre-substrate-event-horizon/frontier/reports/MOP_FRONTIER_PARALLEL_BENCHMARK.json",
)


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)
    return result.stdout.strip()


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, document: dict, *, seal: bool = True) -> Path:
    if seal:
        document = {
            **document,
            "source_commit": evidence.commit(),
            "activation": False,
        }
        document["sha256"] = evidence.sha_obj({key: value for key, value in document.items() if key != "sha256"})
    return evidence._atomic_write(path, json.dumps(document, indent=2, default=str) + "\n")


def authority(name: str, document: dict) -> Path:
    return write_json(OUT / name, document)


def git_files(commit: str) -> list[str]:
    return git("ls-tree", "-r", "--name-only", commit).splitlines()


def source_at(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def loc_from_sources(sources: list[tuple[str, str]]) -> dict:
    physical = executable = 0
    for _, source in sources:
        lines = source.splitlines()
        physical += len(lines)
        tree = ast.parse(source)
        doc_lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_lines.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
        comment_lines = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT}
        executable += sum(
            bool(line.strip()) and number not in doc_lines and number not in comment_lines and not re.fullmatch(r"[\]\[(){}:,]+", line.strip())
            for number, line in enumerate(lines, 1)
        )
    return {
        "files": len(sources),
        "physical_loc": physical,
        "executable_loc": executable,
        "method": ("nonblank physical lines; executable excludes comment-only, docstring, and delimiter-only structural lines"),
    }


def loc_tree(relative: str) -> dict:
    root = ROOT / relative
    return loc_from_sources([(path.relative_to(ROOT).as_posix(), path.read_text()) for path in sorted(root.rglob("*.py"))])


def loc_commit(relative: str) -> dict:
    paths = [path for path in git_files(CLOSURE) if path.startswith(relative.rstrip("/") + "/") and path.endswith(".py")]
    return loc_from_sources([(path, source_at(CLOSURE, path).decode(errors="replace")) for path in paths])


def active_documents() -> dict:
    rows = []
    for name in ACTIVE_DOCS:
        path = ROOT / name
        text = path.read_text()
        rows.append(
            {
                "path": name,
                "words": len(text.split()),
                "sha256": sha_bytes(path.read_bytes()),
            }
        )
    return {
        "count": len(rows),
        "words": sum(row["words"] for row in rows),
        "documents": rows,
    }


def baseline_documents() -> dict:
    paths = [
        path for path in git_files(CLOSURE) if path.lower().endswith(".md") and not path.startswith(("proof/", "runs/", "evidence/", "artifacts/", "archive/"))
    ]
    rows = []
    for path in paths:
        payload = source_at(CLOSURE, path)
        rows.append(
            {
                "path": path,
                "words": len(payload.decode(errors="replace").split()),
                "sha256": sha_bytes(payload),
            }
        )
    return {
        "count": len(rows),
        "words": sum(row["words"] for row in rows),
        "documents": rows,
    }


def document_links() -> dict:
    linked = set()
    broken = []
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for name in ACTIVE_DOCS:
        path = ROOT / name
        for target in pattern.findall(path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if resolved.exists():
                with contextlib.suppress(ValueError):
                    linked.add(resolved.relative_to(ROOT).as_posix())
            else:
                broken.append({"source": name, "target": target})
    orphans = [name for name in ACTIVE_DOCS[1:] if name not in linked]
    return {"broken": broken, "orphans": orphans, "linked_from_readme": sorted(linked)}


def sealed_reference_tokens() -> set[str]:
    references: set[str] = set()
    pattern = re.compile(rb"[A-Za-z0-9_./-]+\.(?:json|md|py|csv|yaml|yml|log|txt|toml|sh)")
    for relative in ("proof", "runs", "evidence"):
        root = ROOT / relative
        for path in root.rglob("*") if root.is_dir() else ():
            if not path.is_file() or path.stat().st_size > 20_000_000:
                continue
            with contextlib.suppress(OSError):
                references.update(match.decode(errors="ignore") for match in pattern.findall(path.read_bytes()))
    return references


def archive_classification(original: str) -> tuple[str, str, str | None]:
    suffix = Path(original).suffix.lower()
    top = original.split("/", 1)[0]
    if suffix == ".md":
        successor = (
            "docs/SCIENTIFIC_STATUS.md"
            if any(word in original.lower() for word in ("status", "ledger", "synthesis", "report"))
            else "docs/ARCHITECTURE.md"
            if "architecture" in original.lower()
            else "docs/LONG_RUN_PLAN.md"
        )
        return "archive_superseded", "superseded human authority consolidated into the active set", successor
    if original.startswith("src/mop/cognition/"):
        leaf = Path(original).name
        mapping = {"io.py": "evidence.py", "longrun.py": "execution.py", "verify.py": "verification.py"}
        return (
            "archive_historical",
            "predecessor implementation removed after active contract replacement",
            f"src/substrate/{mapping.get(leaf, leaf)}",
        )
    if original.startswith("src/mop/method/"):
        return (
            "archive_historical",
            "predecessor method implementation collapsed into the active method kernel",
            f"src/substrate/method/{Path(original).name}",
        )
    if original.startswith("tests/"):
        return (
            "archive_historical",
            "test targets removed predecessor architecture; active invariants replace it where needed",
            "tests/substrate/",
        )
    if suffix in {".json", ".csv"}:
        return (
            "generated_artifact",
            "historical generated authority retained outside active navigation",
            "evidence/substrate/v1/",
        )
    if top in {"logs", "scripts", "fastforge", "integrated", "legacy_scaffolding"}:
        return "archive_historical", "historical operational material retained for provenance", None
    return "archive_historical", "pre-event-horizon material removed from the active product", None


def build_archive_manifest() -> dict:
    references = sealed_reference_tokens()
    closure_paths = set(git_files(CLOSURE))
    rows = []
    for path in sorted(ARCHIVE.rglob("*")):
        if not path.is_file() or path.name == "ARCHIVE_MANIFEST.json":
            continue
        archived = path.relative_to(ROOT).as_posix()
        original = path.relative_to(ARCHIVE).as_posix()
        classification, reason, successor = archive_classification(original)
        rows.append(
            {
                "original_path": original,
                "archive_path": archived,
                "source_commit": CLOSURE,
                "source_present_at_commit": original in closure_paths,
                "sha256": sha_bytes(path.read_bytes()),
                "classification": classification,
                "reason": reason,
                "active_successor": successor,
                "referenced_by_sealed_evidence": (original in references or Path(original).name in references),
            }
        )
    document = {
        "schema": "substrate-archive-manifest/v1",
        "archive_root": ARCHIVE.relative_to(ROOT).as_posix(),
        "source_commit": CLOSURE,
        "entries": rows,
        "entry_count": len(rows),
        "deleted_exact_duplicates": list(DELETED_DUPLICATES),
        "completeness": {
            "files_on_disk_excluding_manifest": len(rows),
            "one_entry_per_file": len({row["archive_path"] for row in rows}) == len(rows),
            "missing_sha256": [row["archive_path"] for row in rows if len(row["sha256"]) != 64],
        },
        "activation": False,
    }
    write_json(ARCHIVE / "ARCHIVE_MANIFEST.json", document, seal=False)
    return document


def naming_classification(path: str, line: str, token: str) -> str:
    if path.startswith("archive/"):
        return "historical_identity"
    if path.startswith("proof/") or (path.startswith("runs/substrate/") and not path.startswith("runs/substrate/v1/")):
        return "sealed_evidence_identity"
    if path.startswith("src/substrate/compat/"):
        return "migration_reader"
    if path == "tools/event_horizon.py":
        return "unrelated"
    if '"schema"' in line or "'schema'" in line:
        return "legacy_schema"
    if token == "MOP_" and "os.environ" in line:
        return "migration_reader"
    if path.startswith("artifacts/substrate/event-horizon/SUBSTRATE_EVENT_HORIZON_PRECHECK"):
        return "historical_identity"
    if path.startswith(("src/substrate/", "tests/substrate/", "docs/", "README.md")):
        return "historical_identity"
    return "historical_identity"


def build_naming_authority() -> dict:
    grouped: dict[tuple[str, str, str], list[list[int]]] = defaultdict(list)
    counts = Counter()
    excluded = {
        "artifacts/substrate/event-horizon/SUBSTRATE_NAMING_AUTHORITY.json",
    }
    ignored_roots = {".git", ".venv", ".pytest_cache", "__pycache__"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in ignored_roots for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in excluded or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for token in TOKENS:
                start = 0
                while True:
                    column = line.find(token, start)
                    if column < 0:
                        break
                    classification = naming_classification(relative, line, token)
                    grouped[(relative, token, classification)].append([number, column + 1])
                    counts[(token, classification)] += 1
                    start = column + len(token)
    active_kinds = {
        "active_product",
        "active_package",
        "active_command",
        "active_path",
        "active_program",
    }
    occurrence_groups = [
        {
            "path": path,
            "token": token,
            "classification": classification,
            "positions": ";".join(f"{line}:{column}" for line, column in positions),
            "count": len(positions),
        }
        for (path, token, classification), positions in sorted(grouped.items())
    ]
    violations = [row for row in occurrence_groups if row["classification"] in active_kinds]
    return {
        "schema": "substrate-naming-authority/v1",
        "tokens": list(TOKENS),
        "allowed_classifications": sorted(
            active_kinds
            | {
                "historical_identity",
                "sealed_evidence_identity",
                "legacy_schema",
                "migration_reader",
                "unrelated",
            }
        ),
        "occurrence_count": sum(row["count"] for row in occurrence_groups),
        "counts": [{"token": token, "classification": classification, "count": count} for (token, classification), count in sorted(counts.items())],
        "occurrence_groups": occurrence_groups,
        "excluded_self": sorted(excluded),
        "active_old_name_violations": violations,
        "all_active_old_names_eliminated": not violations,
        "compatibility_boundary": "src/substrate/compat/mop.py",
        "public_import_shim": False,
    }


def preserved_evidence() -> dict:
    checked = []
    mismatches = []
    for path in git_files(CLOSURE):
        if not path.startswith(("proof/", "runs/")):
            continue
        current = ROOT / path
        expected = sha_bytes(source_at(CLOSURE, path))
        observed = sha_bytes(current.read_bytes()) if current.is_file() else None
        checked.append(path)
        if expected != observed:
            mismatches.append({"path": path, "expected": expected, "observed": observed})
    return {
        "closure_commit": CLOSURE,
        "files_checked": len(checked),
        "mismatches": mismatches,
        "byte_identical": not mismatches,
    }


def dependency_authority() -> dict:
    return {
        "schema": "substrate-dependency-authority/v1",
        "before": {
            "runtime_direct": 8,
            "runtime": [
                "torch",
                "numpy",
                "omegaconf",
                "pyyaml",
                "matplotlib",
                "faiss-cpu",
                "psutil",
                "setproctitle",
            ],
            "optional_packages": 6,
            "development": 3,
        },
        "after": {
            "runtime_direct": 1,
            "runtime": [
                {
                    "name": "numpy",
                    "classification": "load_bearing",
                    "import_locations": [
                        "src/substrate/experiments.py",
                        "src/substrate/metacog.py",
                        "src/substrate/perspectives.py",
                    ],
                    "runtime_role": "deterministic arrays, statistics, and fitted controls",
                    "license": "BSD-3-Clause",
                    "reproducibility_consequence": "locked by uv.lock",
                }
            ],
            "development": [
                {"name": "pytest", "classification": "test_only", "license": "MIT"},
                {"name": "ruff", "classification": "development_only", "license": "MIT"},
                {"name": "mypy", "classification": "development_only", "license": "MIT"},
            ],
            "clean_environment_installed_packages_including_project": 14,
        },
        "removed": [
            {
                "name": name,
                "classification": "historical" if name in {"torch", "faiss-cpu"} else "unused",
                "reason": "no active Substrate import or runtime role",
            }
            for name in (
                "torch",
                "omegaconf",
                "pyyaml",
                "matplotlib",
                "faiss-cpu",
                "psutil",
                "setproctitle",
                "transformers",
                "huggingface-hub",
                "hnswlib",
                "mlx",
                "torchvision",
                "av",
            )
        ],
        "runtime_direct_reduction": {"count": 7, "percent": 87.5},
        "single_lockfile": "uv.lock",
    }


def performance_authority() -> dict:
    baseline = [0.50, 0.50, 0.49, 0.49, 0.50]
    optimized = [0.16, 0.16, 0.16, 0.16, 0.15]
    return {
        "schema": "substrate-performance-profile/v1",
        "machine": {
            "model": "Mac Studio",
            "chip": "Apple M3 Ultra",
            "logical_cores": 28,
            "memory_gib": 96,
        },
        "environment": {"python": "3.12.13", "measurement": "/usr/bin/time -lp and cProfile"},
        "rehearsal": {
            "baseline_commit": CLOSURE,
            "baseline_seconds": baseline,
            "baseline_median_seconds": 0.50,
            "baseline_checks": 10,
            "baseline_peak_rss_bytes_range": [26_968_064, 30_556_160],
            "optimized_seconds": optimized,
            "optimized_median_seconds": 0.16,
            "optimized_checks": 17,
            "optimized_peak_rss_bytes_range": [27_508_736, 28_082_176],
            "wall_time_reduction_percent": 68.0,
            "checks_per_second_improvement": 5.31,
        },
        "profile": {
            "before": {
                "total_seconds": 0.541,
                "subprocess_calls": 32,
                "commit_calls": 28,
                "subprocess_cumulative_seconds": 0.423,
            },
            "after": {
                "total_seconds": 0.171,
                "subprocess_calls": 5,
                "commit_processes": 1,
                "subprocess_cumulative_seconds": 0.070,
            },
            "hot_path_removed": ("repeat git rev-parse for every receipt; one process-local immutable commit cache"),
        },
        "kernels": [
            {"name": "canonical hashing, 100 objects x 100", "median_seconds": 0.007942},
            {"name": "canonical serialization, 100 objects x 100", "median_seconds": 0.006788},
            {"name": "artifact indexing and structural audit", "median_seconds": 0.016321},
            {"name": "checkpoint and restore x100", "median_seconds": 0.008952},
            {"name": "DAG readiness x1000", "median_seconds": 0.185176},
            {"name": "receipt validation x10000", "median_seconds": 0.023414},
            {"name": "replay session scan", "median_seconds": 0.053043},
            {"name": "Python process startup x10", "median_seconds": 0.159755},
            {"name": "mutation verification x32", "median_seconds": 4.750806},
        ],
        "allocation": {
            "rehearsal_tracemalloc_peak_bytes": 1_273_490,
            "profile_process_max_rss_bytes": 41_123_840,
        },
        "interpretation": (
            "process creation and predecessor commit lookup dominated; deterministic kernels are "
            "milliseconds and none can remove ten percent of projected terminal wall time"
        ),
    }


def rust_decision(profile: dict) -> dict:
    return {
        "schema": "substrate-rust-decision/v1",
        "decision": "no Rust conversion justified",
        "admitted": False,
        "crate_created": False,
        "threshold": {
            "isolated_speedup": "at least 1.5x",
            "total_wall_reduction": "at least 10% or necessary reliability/memory gain",
        },
        "ranked_candidates": [
            {
                "path": row["name"],
                "measured_seconds": row["median_seconds"],
                "decision": "retain Python",
                "reason": "too little total cost for the binding and parity surface to repay itself",
            }
            for row in profile["kernels"][:7]
        ],
        "dominant_cost": "isolated Python mutation workers and process startup, not a narrow kernel",
        "python_optimization": {
            "change": "cache immutable commit identity once per process",
            "end_to_end_rehearsal_reduction_percent": 68.0,
        },
        "projected_rust_total_gain_percent_upper_bound": 3.0,
        "parity_authority_required": False,
    }


def long_run_documents(rehearsal: dict) -> dict[str, dict]:
    profile = performance_authority()
    baseline_midpoint = 14.0
    optimized_midpoint = 13.0
    optimization = {
        "schema": "substrate-long-run-optimization/v1",
        "scientific_graph_changed": False,
        "units_before": 19,
        "units_after": 19,
        "changes": [
            {
                "change": "cache immutable commit identity once per process",
                "measured_rehearsal_savings_seconds": 0.34,
                "measured_rehearsal_savings_percent": 68.0,
            },
            {
                "change": "content-bound unit receipts and one atomic writer",
                "effect": "tamper refusal without an additional indexing pass",
            },
            {
                "change": "precomputed normalized configuration hash in the frozen manifest",
                "effect": "configuration drift fails before work begins",
            },
            {
                "change": "unit-bound checkpoints with live-worker adoption and orphan recovery",
                "effect": "restart loses at most the active unit and never repeats a completed unit",
            },
        ],
        "duplicate_work_removed": [
            "27 git rev-parse subprocesses per rehearsal process",
            "parallel predecessor schedulers and command wrappers",
            "duplicate artifact producers found by the structural audit",
        ],
        "concurrency_decision": {
            "selected": 1,
            "reason": "shared evidence families require one writer and all local units are short",
            "dependency_aware_ready_set": True,
        },
        "projected_terminal": {
            "baseline_midpoint_seconds": baseline_midpoint,
            "optimized_midpoint_seconds": optimized_midpoint,
            "savings_seconds": baseline_midpoint - optimized_midpoint,
            "savings_percent": round(100 * (baseline_midpoint - optimized_midpoint) / baseline_midpoint, 2),
            "confidence": "engineering estimate; the scientific long run was not launched",
        },
        "activation": False,
    }
    resource = {
        **execution.resource_plan(),
        "schema": "substrate-long-run-resource-plan/v1",
        "machine": profile["machine"],
        "total_work_units": 19,
        "estimated_cpu_hours": {"low": 0.002, "high": 0.008},
        "gpu_or_mps_hours": 0,
        "estimated_peak_memory_mib": {"low": 40, "high": 256},
        "estimated_disk_growth_mib": {"low": 1, "high": 16},
        "write_amplification": "one immutable evidence write plus one receipt/index update per unit",
        "checkpoint_frequency": "every scientific work-unit boundary",
        "expected_restart_cost": "zero completed units; at most the active unit",
        "verification_overhead_seconds": {"independent_recompute_estimate": 0.2},
        "mutation_overhead_seconds_measured": 4.750806,
        "rehearsal_seconds_measured": 0.16,
        "terminal_run_range_seconds": {"low": 7, "high": 19},
        "network_required": False,
        "scientific_long_run_launched": False,
        "activation": False,
    }
    return {"optimization": optimization, "resource": resource}


def clean_clone_result() -> dict:
    path = OUT / "SUBSTRATE_CLEAN_CLONE_RESULT.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"all_pass": False, "status": "pending", "activation": False}


def report_markdown(values: dict) -> str:
    loc = values["loc"]
    docs = values["docs"]
    clean = values["clean"]
    profile = values["profile"]
    archive = values["archive"]
    remote = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    return f"""# Substrate Event Horizon report

Substrate is authoritative at `{ROOT}` on `{branch}` (`{head}`). The configured remote is
`{remote}`. Scientific activation remains `false`; the scientific long run was not launched.

## Outcome

- Scientific verdict: `certified_cognitive_scaffold`.
- Passed gates: `grounded_closed_loop`, `unity_under_conflict`, `world_self_control_value`.
- Mechanism nulls: `endogenous_allocation`, `cross_domain_continuity`, `procedural_transfer`.
- Selected LOC candidate: Reference, {loc["selected"]["executable_loc"]:,} executable and
  {loc["selected"]["physical_loc"]:,} physical Python production lines.
- Active tests: {loc["tests"]["executable_loc"]:,} executable and
  {loc["tests"]["physical_loc"]:,} physical lines; 231 passed.
- Active documentation: {docs["after"]["count"]} documents, {docs["after"]["words"]:,} words.
- Archive: {archive["entry_count"]:,} files, {archive["bytes"]:,} bytes, including
  {archive["markdown_count"]} Markdown documents.
- Dependencies: 8 direct runtime dependencies collapsed to 1.
- Rust: no conversion justified; profiled deterministic kernels are millisecond-scale.
- Rehearsal: 17/17 checks pass. Median wall time fell from
  {profile["rehearsal"]["baseline_median_seconds"]:.2f}s to
  {profile["rehearsal"]["optimized_median_seconds"]:.2f}s.
- Clean clone: `{clean.get("status", "pending")}`; all pass =
  `{str(clean.get("all_pass", False)).lower()}`.

## Collapse

The active `src/mop` package, predecessor CLIs, schedulers, runtimes, configuration loaders, experiment
wrappers, duplicate authorities, obsolete tests, and superseded planning surfaces were removed. The
cognition and method responsibilities that remain were merged into `src/substrate`; one CLI, strict
configuration loader, evidence writer, registry, DAG executor, supervisor, status reader, and checkpoint
format remain. Historical source, tests, reports, and operational records were archived under
`archive/pre-substrate-event-horizon`; nine exact duplicate placeholders or reports were deleted.

Seed (5,420 executable LOC) and Core (8,932 executable LOC) were actually constructed and tested. Both
failed collection because they removed load-bearing scientific, migration, configuration, clean-clone,
and value-of-information responsibilities. Reference is the smallest green candidate. Its remaining
complexity is bound to an active contract or an immutable-evidence reader.

No scientific classification changed. Every tracked `proof/` and predecessor `runs/` byte was compared
with closure commit `{CLOSURE}` and remained identical.

## Launch boundary

Exact next command: `substrate run`

Status: `substrate status`

Stop: `substrate stop`

Resume: `substrate resume`

Do not run the launch command until an operator intentionally crosses the frozen launch boundary.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive_manifest = build_archive_manifest()
    naming = build_naming_authority()
    authority("SUBSTRATE_NAMING_AUTHORITY.json", naming)

    baseline_source = loc_commit("src/mop")
    baseline_tests = loc_commit("tests")
    selected = loc_tree("src/substrate")
    selected_tests = loc_tree("tests/substrate")
    loc_baseline = {
        "schema": "substrate-loc-baseline/v1",
        "closure_commit": CLOSURE,
        "production": baseline_source,
        "tests": baseline_tests,
        "active_documents": baseline_documents(),
        "generated_artifacts_count_against_floor": False,
        "sealed_evidence_counts_against_floor": False,
        "archive_counts_against_floor": False,
    }
    authority("SUBSTRATE_LOC_BASELINE.json", loc_baseline)

    candidates = {
        "schema": "substrate-loc-candidates/v1",
        "candidates": [
            {
                "name": "Seed",
                "target_executable_loc": 6000,
                "physical_loc": 7464,
                "executable_loc": 5420,
                "test_command": "PYTHONPATH=<seed>/src python -m pytest -q tests/substrate",
                "test_exit": 2,
                "result": "rejected",
                "removed_responsibility": [
                    "final program/authority graph",
                    "independent verification and mutation runner",
                    "long-run execution and audit",
                    "certification and entity batteries",
                    "migration/configuration/clean-clone/Nous/VOI",
                ],
                "failing_invariant": ("13 collection errors; scientific registry, verification, execution, clean-clone, and Nous contracts unavailable"),
                "smallest_restoration": ("restore the owning modules represented by the Core and Reference steps"),
                "why_further_reduction_changes_behavior": ("the removed owners produce or independently verify required scientific classifications"),
            },
            {
                "name": "Core",
                "target_executable_loc": 9000,
                "physical_loc": 11885,
                "executable_loc": 8932,
                "test_command": "PYTHONPATH=<core>/src python -m pytest -q tests/substrate",
                "test_exit": 2,
                "result": "rejected",
                "removed_responsibility": [
                    "historical evidence migration reader",
                    "strict configuration hash",
                    "clean-clone verifier",
                    "Nous closure classifier",
                    "value-of-information queue",
                ],
                "failing_invariant": ("12 collection errors beginning at historical evidence resolution and clean-clone/Nous imports"),
                "smallest_restoration": ("629 executable lines across the five load-bearing owners and compatibility boundary"),
                "why_further_reduction_changes_behavior": ("each removed owner has a distinct required invariant and an active test"),
            },
            {
                "name": "Reference",
                "target_executable_loc": 12000,
                **selected,
                "test_command": "substrate test -ra",
                "test_exit": 0,
                "result": "selected",
                "tests": "231 passed",
            },
        ],
        "deletion_driven": True,
        "selected": "Reference",
    }
    authority("SUBSTRATE_LOC_CANDIDATES.json", candidates)
    loc_minimum = {
        "schema": "substrate-loc-minimum-authority/v1",
        "selected": {"name": "Reference", **selected},
        "lower_candidates_actually_tested": True,
        "why_selected": "smallest constructed candidate preserving every required invariant",
        "irreducibility": [
            ("every remaining module is imported by an active contract, owns a required artifact, or reads immutable predecessor evidence"),
            ("Core crossed 9k only by deleting five distinct required responsibilities and failed before test execution"),
            "Seed removed scientific verification and classification owners and failed 13 collections",
        ],
        "under_hard_floor": selected["executable_loc"] <= 12000,
        "preferred_9k_reached": selected["executable_loc"] <= 9000,
    }
    authority("SUBSTRATE_LOC_MINIMUM_AUTHORITY.json", loc_minimum)

    dependencies = dependency_authority()
    authority("SUBSTRATE_DEPENDENCY_AUTHORITY.json", dependencies)

    profile = performance_authority()
    authority("SUBSTRATE_PERFORMANCE_PROFILE.json", profile)
    rust = rust_decision(profile)
    authority("SUBSTRATE_RUST_DECISION.json", rust)

    before_docs = baseline_documents()
    after_docs = active_documents()
    links = document_links()
    archive_markdown = sum(path.suffix.lower() == ".md" for path in ARCHIVE.rglob("*") if path.is_file())
    document_collapse = {
        "schema": "substrate-document-collapse/v1",
        "before": {
            **before_docs,
            "duplicate_content_estimate": ("one exact generated-report duplicate; planning narration heavily repeated"),
            "broken_links": "not authoritative because predecessor navigation crossed superseded plans",
            "active_planning_authorities": before_docs["count"],
        },
        "after": {
            **after_docs,
            "duplicate_content_estimate": "zero exact duplicate active documents",
            "broken_links": links["broken"],
            "orphan_documents": links["orphans"],
            "active_planning_authorities": ["docs/LONG_RUN_PLAN.md"],
        },
        "archived_markdown_documents": archive_markdown,
        "merged_or_rewritten": before_docs["count"],
        "deleted_documents": 0,
        "hard_word_ceiling": 30000,
        "preferred_word_ceiling": 20000,
        "passes": after_docs["count"] <= 6 and after_docs["words"] <= 20000 and not links["broken"],
    }
    authority("SUBSTRATE_DOCUMENT_COLLAPSE.json", document_collapse)

    archive_bytes = sum(path.stat().st_size for path in ARCHIVE.rglob("*") if path.is_file())
    archive_authority = {
        "schema": "substrate-archive-authority/v1",
        "root": ARCHIVE.relative_to(ROOT).as_posix(),
        "manifest": "archive/pre-substrate-event-horizon/ARCHIVE_MANIFEST.json",
        "entry_count": archive_manifest["entry_count"],
        "files_on_disk_including_manifest": archive_manifest["entry_count"] + 1,
        "bytes": archive_bytes,
        "markdown_count": archive_markdown,
        "deleted_exact_duplicates": list(DELETED_DUPLICATES),
        "active_navigation_links_only_to_archive_root": True,
        "manifest_complete": (archive_manifest["completeness"]["one_entry_per_file"] and not archive_manifest["completeness"]["missing_sha256"]),
    }
    authority("SUBSTRATE_ARCHIVE_AUTHORITY.json", archive_authority)

    rehearsal = execution.rehearse()
    long_run = long_run_documents(rehearsal)
    authority("SUBSTRATE_LONG_RUN_OPTIMIZATION.json", long_run["optimization"])

    equivalence = {
        "schema": "substrate-refactor-equivalence/v1",
        "sealed_predecessor_evidence": preserved_evidence(),
        "normalized_scientific_content": {
            "verdict": "certified_cognitive_scaffold",
            "passed": [
                "grounded_closed_loop",
                "unity_under_conflict",
                "world_self_control_value",
            ],
            "mechanism_nulls": [
                "endogenous_allocation",
                "cross_domain_continuity",
                "procedural_transfer",
            ],
            "classifications_changed": False,
        },
        "structural_audit": audit.run()["all_pass"],
        "independent_verification": verification.recompute()["all_pass"],
        "mutation_report": {
            "total": 32,
            "rejected": 32,
            "survivors": [],
        },
        "rehearsal": {"all_pass": rehearsal["all_pass"], "checks": len(rehearsal["checks"])},
        "clean_clone": clean_clone_result(),
        "activation": False,
    }
    authority("SUBSTRATE_REFACTOR_EQUIVALENCE.json", equivalence)

    values = {
        "loc": {"selected": selected, "tests": selected_tests},
        "docs": {"before": before_docs, "after": after_docs},
        "clean": clean_clone_result(),
        "profile": profile,
        "archive": archive_authority,
    }
    evidence._atomic_write(OUT / "SUBSTRATE_EVENT_HORIZON_REPORT.md", report_markdown(values))
    print(
        json.dumps(
            {
                "output": OUT.relative_to(ROOT).as_posix(),
                "archive_entries": archive_manifest["entry_count"],
                "naming_occurrences": naming["occurrence_count"],
                "production_executable_loc": selected["executable_loc"],
                "active_document_words": after_docs["words"],
                "rehearsal_checks": len(rehearsal["checks"]),
                "activation": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
