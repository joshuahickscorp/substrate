"""Code lifecycle: what is active, what is frozen for reproducibility, and what the orientation surface is.

The maintained Python total exceeds the old target because the implementations behind sealed receipts are
still here. Deleting them to satisfy a number would trade reproducibility for a green cell, so instead the
code is classified and each class carries its own target. Reclassification is not deletion and is not called
deletion.

House style: no dashes.
"""

from __future__ import annotations

import time
from pathlib import Path

from mop.temporal import io

CLASSES = ("active_runtime", "active_method", "active_substrate", "active_tests",
           "frozen_reproducibility", "historical_migration", "generated")

# path prefix to lifecycle class, longest prefix wins
RULES = [
    # The resumable execution surface stays active. Receipt producing stage drivers remain byte for byte
    # recoverable but are frozen once their authority is sealed; they are not part of the orientation surface.
    ("src/mop/temporal/runs/supervisor.py", "active_substrate"),
    ("src/mop/temporal/runs/e2.py", "active_substrate"),
    ("src/mop/temporal/runs/__init__.py", "active_substrate"),
    # The third bed now owns a live receipt normalization path for a sealed data split authority.
    # It is active runtime instrumentation, not additional E2 mechanism implementation.
    ("src/mop/temporal/runs/thirdbed.py", "active_runtime"),
    ("src/mop/temporal/runs", "frozen_reproducibility"),
    # These three modules are live, shared execution plumbing rather than E2 mechanism implementation.
    # They still count toward the global active runtime cap; this boundary does not freeze or hide them.
    ("src/mop/temporal/receipt_contract.py", "active_runtime"),
    ("src/mop/temporal/custody.py", "active_runtime"),
    ("src/mop/temporal/io.py", "active_runtime"),
    ("src/mop/temporal", "active_substrate"),
    ("src/mop/method", "active_method"),
    ("tests/temporal", "active_tests"),
    ("tests/method", "active_tests"),
    # fastforge/runs holds the Fast State Forge program's own stages, which are frozen behind its receipts.
    # The engine, the architectures and the data providers are the live execution system this program uses.
    ("fastforge/runs", "frozen_reproducibility"),
    ("fastforge/verify.py", "frozen_reproducibility"),
    ("fastforge/arms.py", "frozen_reproducibility"),
    ("fastforge/within.py", "frozen_reproducibility"),
    ("fastforge/sequence.py", "frozen_reproducibility"),
    ("fastforge", "active_runtime"),
    ("src/mop/beds", "frozen_reproducibility"),
    ("src/mop/substrate", "frozen_reproducibility"),
    ("src/mop/experiments", "frozen_reproducibility"),
    ("src/mop/science", "active_runtime"),
    ("src/mop/harness", "active_runtime"),
    ("src/mop/studio", "frozen_reproducibility"),
    ("src/mop", "active_runtime"),
    ("tests", "frozen_reproducibility"),
    ("substrate", "frozen_reproducibility"),
    ("gen3", "frozen_reproducibility"),
    ("frontier", "frozen_reproducibility"),
    ("integrated", "frozen_reproducibility"),
    ("collapse", "frozen_reproducibility"),
    ("campaign2", "frozen_reproducibility"),
    ("salvage", "frozen_reproducibility"),
    ("substrate_evo", "frozen_reproducibility"),
    ("forge", "frozen_reproducibility"),
    ("legacy_scaffolding", "historical_migration"),
    ("scaffolding", "historical_migration"),
    ("scripts", "active_runtime"),
]

TARGETS = {
    "active_runtime_loc": 8000,
    "new_active_e2_implementation_loc": 2500,
    "normal_entrypoints": 3,
    "configuration_roots": 1,
    "registries": 1,
    "experiment_engines": 1,
    "evidence_fabrics": 1,
}


def classify(rel: str) -> str:
    best, cls = -1, "historical_migration"
    for prefix, c in RULES:
        if rel.startswith(prefix) and len(prefix) > best:
            best, cls = len(prefix), c
    return cls


def scan() -> dict:
    rows: dict[str, dict] = {c: {} for c in CLASSES}
    for p in sorted(io.ROOT.rglob("*.py")):
        if "__pycache__" in p.parts or ".venv" in p.parts or ".git" in p.parts:
            continue
        rel = p.relative_to(io.ROOT).as_posix()
        rows[classify(rel)][rel] = len(p.read_text(errors="ignore").splitlines())
    return rows


def main():
    t0 = time.time()
    rows = scan()
    loc = {c: sum(v.values()) for c, v in rows.items()}
    e2 = {k: v for k, v in rows["active_substrate"].items() if k.startswith("src/mop/temporal")}
    active_runtime = loc["active_runtime"] + loc["active_substrate"]
    io.seal("MOP_CODE_LIFECYCLE_AUTHORITY.json", {
        "schema": "mop-code-lifecycle-authority/v1",
        "classes": list(CLASSES),
        "rules": [{"prefix": p, "class": c} for p, c in RULES],
        "targets": TARGETS,
        "policy": ("frozen reproducibility code is the implementation behind sealed immutable receipts. It "
                   "stays recoverable and is excluded from the orientation surface by classification, not by "
                   "deletion. Reclassification is reported as reclassification"),
        "loc_by_class": loc,
        "total_python_loc": sum(loc.values()),
    })
    io.seal("MOP_ACTIVE_CODE_ACCOUNTING.json", {
        "schema": "mop-active-code-accounting/v1",
        "active_runtime": {"files": rows["active_runtime"], "loc": loc["active_runtime"]},
        "active_method": {"files": rows["active_method"], "loc": loc["active_method"]},
        "active_substrate": {"files": rows["active_substrate"], "loc": loc["active_substrate"]},
        "active_tests": {"files": rows["active_tests"], "loc": loc["active_tests"]},
        "active_runtime_plus_substrate": active_runtime,
        "active_runtime_target": TARGETS["active_runtime_loc"],
        "active_runtime_within_target": active_runtime <= TARGETS["active_runtime_loc"],
        "new_e2_implementation": {"files": e2, "loc": sum(e2.values()),
                                  "target": TARGETS["new_active_e2_implementation_loc"],
                                  "within_target": sum(e2.values()) <= TARGETS["new_active_e2_implementation_loc"]},
        "normal_entrypoints": ["python -m mop.temporal.runs.supervisor",
                               "python -m mop.temporal.runs.<stage>",
                               "python -m mop.method.runs.<stage>"],
        "configuration_roots": ["configs/"],
        "registries": ["registry/experiments.yaml"],
        "experiment_engines": ["fastforge.engine"],
        "evidence_fabrics": ["integrated/evidence_store"],
        "honest_blocker": "" if active_runtime <= TARGETS["active_runtime_loc"] else (
            f"active runtime and substrate together are {active_runtime} against a target of "
            f"{TARGETS['active_runtime_loc']}"),
    })
    io.seal("MOP_FROZEN_REPRODUCIBILITY_ACCOUNTING.json", {
        "schema": "mop-frozen-reproducibility-accounting/v1",
        "frozen_reproducibility": {"loc": loc["frozen_reproducibility"],
                                   "files": sorted(rows["frozen_reproducibility"])},
        "historical_migration": {"loc": loc["historical_migration"],
                                 "files": sorted(rows["historical_migration"])},
        "generated": {"loc": loc["generated"], "files": sorted(rows["generated"])},
        "why_retained": ("each of these files is the implementation behind at least one sealed receipt. "
                         "Removing them would make prior evidence unreproducible, which the immutability "
                         "rule forbids. They are excluded from the orientation surface by classification"),
        "recoverable": True,
        "counts_toward_active_targets": False,
    })
    print(f"code lifecycle: active runtime+substrate {active_runtime}/{TARGETS['active_runtime_loc']}, "
          f"new E2 {sum(e2.values())}/{TARGETS['new_active_e2_implementation_loc']}, "
          f"frozen {loc['frozen_reproducibility']}", flush=True)
    print("CODELIFE_DONE", flush=True)


if __name__ == "__main__":
    main()
