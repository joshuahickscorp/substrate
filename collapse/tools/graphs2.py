"""Second census pass: safety record + authority/command/duplication graphs (spec sections 7, 9, 12, 14).

Deterministic and read-only. Emits into collapse/:
  MOP_LIVE_NO_TOUCH.json      the immutable live-run boundary (safety authority)
  MOP_AUTHORITY_GRAPH.json    every implementation of the evidence/integrity primitives (section 9 targets)
  MOP_COMMAND_GRAPH.json      scripts/ inventory + thin-wrapper classification (section 14 targets)
  MOP_DUPLICATION_GRAPH.json  lifecycle-boilerplate clusters with member files (section 12 targets)

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse"
OUT.mkdir(exist_ok=True)
LIVE = Path("/Users/scammermike/Downloads/mop")


def sh(*a: str, cwd: Path = ROOT) -> str:
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True).stdout.strip()


def tracked() -> list[str]:
    return [x for x in sh("git", "ls-files").splitlines() if x]


def phys_loc(p: Path) -> int:
    try:
        d = p.read_bytes()
    except Exception:
        return 0
    return d.count(b"\n") + (0 if d.endswith(b"\n") or not d else 1)


files = tracked()
py = [f for f in files if f.endswith(".py")]

# ---------- LIVE_NO_TOUCH ----------
live_status = {}
sp = LIVE / "runs/generation1/general-run/current_status.json"
if sp.exists():
    try:
        s = json.loads(sp.read_text())
        live_status = {k: s.get(k) for k in ("state", "stage", "updated_at", "counts",
                                             "parent_implementation")}
    except Exception:
        live_status = {"state": "unreadable"}
no_touch = {
    "schema": "mop-collapse-live-no-touch/v1",
    "rule": ("The detached General Run and its live tree are immutable while active. Condensation occurs "
             "only in the isolated worktree. Never edit, signal, restart, retune, or merge into the live "
             "tree or its processes."),
    "live_tree": str(LIVE),
    "live_branch_head": sh("git", "rev-parse", "HEAD", cwd=LIVE),
    "live_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=LIVE),
    "run_root": str(LIVE / "runs/generation1/general-run"),
    "live_status": live_status,
    "forbidden_paths_for_edit": [str(LIVE)],
    "worktree_is_separate_checkout": True,
    "note": ("Worktree edits cannot affect the live run: the live processes execute from the live tree's "
             "own physical files on branch save-mop-stable-work, which are distinct from this worktree."),
}
(OUT / "MOP_LIVE_NO_TOUCH.json").write_text(json.dumps(no_touch, indent=2), encoding="utf-8")

# ---------- AUTHORITY_GRAPH: every implementation of the integrity/evidence primitives ----------
PRIMS = [
    ("canonical_bytes", re.compile(r"^\s*def\s+canonical_bytes\b")),
    ("canonical_sha256", re.compile(r"^\s*def\s+canonical_sha256\b")),
    ("atomic_write_json", re.compile(r"^\s*def\s+atomic_write_json\b")),
    ("atomic_write", re.compile(r"^\s*def\s+_?atomic_write\b")),
    ("sha256_file", re.compile(r"^\s*def\s+sha256_file\b")),
    ("validate_seal", re.compile(r"^\s*def\s+_?validate_seal\b")),
    ("self_seal", re.compile(r"^\s*def\s+_?self_seal\b|^\s*def\s+_?seal\b")),
    ("canonical_json", re.compile(r"^\s*def\s+canonical_json\b")),
    ("file_sha256", re.compile(r"^\s*def\s+file_sha256\b|^\s*def\s+_sha256_file\b")),
    ("read_json", re.compile(r"^\s*def\s+_?read_json\b")),
]
authority: dict[str, list[dict]] = defaultdict(list)
for f in py:
    p = ROOT / f
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in PRIMS:
            if rx.match(line):
                authority[name].append({"path": f, "line": i})
authority_graph = {
    "schema": "mop-collapse-authority-graph/v1",
    "purpose": "Every owned implementation of an integrity/evidence primitive (section 9 unify targets).",
    "primitives": {k: sorted(v, key=lambda r: r["path"]) for k, v in sorted(authority.items())},
    "implementation_counts": {k: len(v) for k, v in sorted(authority.items(), key=lambda kv: -len(kv[1]))},
}
(OUT / "MOP_AUTHORITY_GRAPH.json").write_text(json.dumps(authority_graph, indent=2), encoding="utf-8")

# ---------- COMMAND_GRAPH: scripts inventory + thin-wrapper classification ----------
scripts = [f for f in py if f.startswith("scripts/")]
cmd_records = []
wrapper = 0
for f in scripts:
    p = ROOT / f
    loc = phys_loc(p)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
    has_argparse = "argparse" in text
    has_main = "__main__" in text
    imports_mop = bool(re.search(r"\b(from|import)\s+mop\b", text))
    calls_main = bool(re.search(r"\.main\(|run_closure\(|def main\(", text))
    # heuristic classification
    if loc <= 60 and imports_mop and (calls_main or "sys.exit" in text):
        cls = "thin_wrapper"
        wrapper += 1
    elif loc <= 40:
        cls = "thin_wrapper"
        wrapper += 1
    elif "migrat" in f or "backfill" in f:
        cls = "one_off_migration"
    elif "build_" in Path(f).name or "generate" in f or "gen_" in Path(f).name:
        cls = "generator"
    else:
        cls = "real_implementation"
    cmd_records.append({"path": f, "physical_LOC": loc, "class": cls,
                        "argparse": has_argparse, "has_main": has_main, "imports_mop": imports_mop})
cmd_graph = {
    "schema": "mop-collapse-command-graph/v1",
    "scripts_total": len(scripts),
    "scripts_LOC": sum(r["physical_LOC"] for r in cmd_records),
    "class_counts": dict(sorted(
        ((c, sum(1 for r in cmd_records if r["class"] == c))
         for c in {r["class"] for r in cmd_records}), key=lambda kv: -kv[1])),
    "target_cli_surface": ["mop status", "mop run <program>", "mop inspect <identity>", "mop explain",
                           "mop verify", "mop drain", "mop resume", "mop packs", "mop dev ..."],
    "records": sorted(cmd_records, key=lambda r: -r["physical_LOC"]),
}
(OUT / "MOP_COMMAND_GRAPH.json").write_text(json.dumps(cmd_graph, indent=2), encoding="utf-8")

# ---------- DUPLICATION_GRAPH: lifecycle boilerplate clusters with members ----------
SUFFIXES = ["_scaffold.py", "_runner.py", "_bed.py", "_gate.py", "_impl.py", "_producer.py",
            "_verifier.py", "_prereg.py", "_harness.py", "_referee.py", "_featurizer.py", "_estimator.py"]
clusters: dict[str, list[dict]] = defaultdict(list)
for f in py:
    name = Path(f).name
    for suf in SUFFIXES:
        if name.endswith(suf):
            clusters[suf].append({"path": f, "physical_LOC": phys_loc(ROOT / f)})
dup_graph = {
    "schema": "mop-collapse-duplication-graph/v1",
    "purpose": "Repeated per-experiment lifecycle files (section 12/10 collapse targets).",
    "cluster_summary": {suf: {"files": len(v), "LOC": sum(x["physical_LOC"] for x in v)}
                        for suf, v in sorted(clusters.items(), key=lambda kv: -len(kv[1]))},
    "total_boilerplate_files": sum(len(v) for v in clusters.values()),
    "total_boilerplate_LOC": sum(x["physical_LOC"] for v in clusters.values() for x in v),
    "clusters": {suf: sorted(v, key=lambda r: -r["physical_LOC"]) for suf, v in clusters.items()},
}
(OUT / "MOP_DUPLICATION_GRAPH.json").write_text(json.dumps(dup_graph, indent=2), encoding="utf-8")

print(json.dumps({
    "live_no_touch": {"live_branch": no_touch["live_branch"], "run_state": live_status.get("state")},
    "authority_primitive_impl_counts": authority_graph["implementation_counts"],
    "scripts_total": cmd_graph["scripts_total"],
    "scripts_class_counts": cmd_graph["class_counts"],
    "duplication_total_files": dup_graph["total_boilerplate_files"],
    "duplication_total_LOC": dup_graph["total_boilerplate_LOC"],
    "duplication_cluster_summary": dup_graph["cluster_summary"],
}, indent=2))
