"""Deterministic authoritative census of the MOP owned system (MOP_ACCRETION_COLLAPSE section 7).

Read-only. Operates on the tracked files of the current worktree (based on origin/main). Emits:

  collapse/MOP_CODEBASE_CENSUS.json      per-file records + global rollups
  collapse/MOP_IMPORT_GRAPH.json         module import edges + reverse edges + SCCs
  collapse/MOP_GLOBAL_ACCOUNTING.json    honest global LOC accounting (section 4)
  collapse/MOP_CONTEXT_SURFACE.json      orientation cost (section 6)

Every number here is computed from bytes on disk, never assumed. Physical LOC = newline count of the
file (matching `wc -l` semantics used elsewhere in the repo). No minification, no packing: physical lines.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse"
OUT.mkdir(exist_ok=True)

CODE_EXT = {".py"}
DOC_EXT = {".md"}
CONFIG_EXT = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def phys_loc(path: Path) -> int:
    try:
        data = path.read_bytes()
    except Exception:
        return 0
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def module_name(rel: str) -> str | None:
    """Dotted module name for a tracked python file under src/, else None (non-package scripts)."""
    if rel.endswith("__init__.py"):
        rel = rel[: -len("/__init__.py")] if "/" in rel else rel[: -len("__init__.py")]
    else:
        rel = rel[:-3]
    if rel.startswith("src/"):
        return rel[len("src/"):].replace("/", ".")
    return None


def parse_py(path: Path):
    """Return (imports:set[str], public_symbols:list[str], is_entry:bool, has_main:bool)."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return set(), [], False, False
    imports: set[str] = set()
    public: list[str] = []
    has_main = "__main__" in src and "__name__" in src
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public.append(node.name)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_") and t.id.isupper():
                    public.append(t.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(node.module)
    return imports, public, has_main, has_main


def tarjan_scc(adj: dict[str, set[str]]) -> list[list[str]]:
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []
    sys.setrecursionlimit(100000)

    def strongconnect(v: str):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj.get(v, ()):  # noqa
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    # iterative-safe wrapper via explicit recursion but with raised limit; trees here are shallow enough
    for v in list(adj):
        if v not in index:
            try:
                strongconnect(v)
            except RecursionError:
                # extremely deep chain: record singleton and move on rather than crash the census
                if v not in index:
                    result.append([v])
    return result


def main() -> int:
    files = tracked_files()
    records: dict[str, dict] = {}
    modmap: dict[str, str] = {}  # dotted module -> rel path

    for rel in files:
        p = ROOT / rel
        ext = p.suffix.lower()
        loc = phys_loc(p)
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        lang = (
            "python" if ext in CODE_EXT else
            "markdown" if ext in DOC_EXT else
            "config" if ext in CONFIG_EXT else
            ext.lstrip(".") or "none"
        )
        rec = {
            "path": rel,
            "language": lang,
            "physical_LOC": loc,
            "bytes": size,
            "imports": [],
            "importers": [],
            "public_symbols": [],
            "is_entrypoint": False,
            "module": None,
        }
        if ext == ".py":
            m = module_name(rel)
            rec["module"] = m
            if m:
                modmap[m] = rel
        records[rel] = rec

    # second pass: parse python for imports/symbols/entry
    for rel, rec in records.items():
        if rec["language"] != "python":
            continue
        imports, public, is_entry, _ = parse_py(ROOT / rel)
        rec["imports"] = sorted(imports)
        rec["public_symbols"] = public
        rec["is_entrypoint"] = bool(is_entry) or rel.startswith("scripts/")

    # build import edges over internal modules (resolve dotted import to nearest tracked module)
    internal_mods = set(modmap)

    def resolve(dotted: str) -> str | None:
        # match longest known module prefix (module or its package)
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in internal_mods:
                return cand
        return None

    adj: dict[str, set[str]] = defaultdict(set)
    importers: dict[str, set[str]] = defaultdict(set)
    for rel, rec in records.items():
        if not rec["module"]:
            continue
        src_mod = rec["module"]
        for imp in rec["imports"]:
            tgt = resolve(imp)
            if tgt and tgt != src_mod:
                adj[src_mod].add(tgt)
                importers[tgt].add(src_mod)
    # write importers back
    for rel, rec in records.items():
        m = rec["module"]
        if m:
            rec["importers"] = sorted(modmap[im] for im in importers.get(m, ()) if im in modmap)

    # SCCs (cyclic clusters) over internal module graph
    sccs = [c for c in tarjan_scc({k: v for k, v in adj.items()}) if len(c) > 1]

    # ---- global accounting (section 4) by physical LOC ----
    def area(rel: str) -> str:
        if rel.startswith("collapse/") or rel.startswith(".collapse/"):
            return "collapse_tooling"
        if rel.startswith("src/mop/"):
            return "src"
        if rel.startswith("tests/"):
            return "tests"
        if rel.startswith("scripts/"):
            return "scripts"
        if rel.startswith("configs/") or rel.startswith("campaign/"):
            return "config"
        if rel.startswith("docs/") or (("/" not in rel) and rel.endswith(".md")):
            return "docs"
        if rel.startswith("proof/") or rel.startswith("runs/"):
            return "proof_runs"
        if rel.startswith("registry/"):
            return "registry"
        if rel.endswith(".json") or rel.endswith((".yaml", ".yml")):
            return "config"
        if rel.endswith(".md"):
            return "docs"
        return "other"

    loc_by_area: dict[str, int] = defaultdict(int)
    files_by_area: dict[str, int] = defaultdict(int)
    loc_by_lang: dict[str, int] = defaultdict(int)
    for rel, rec in records.items():
        loc_by_area[area(rel)] += rec["physical_LOC"]
        files_by_area[area(rel)] += 1
        loc_by_lang[rec["language"]] += rec["physical_LOC"]

    py_src_loc = sum(r["physical_LOC"] for r in records.values()
                     if r["language"] == "python" and r["path"].startswith("src/mop/"))
    py_test_loc = sum(r["physical_LOC"] for r in records.values()
                      if r["language"] == "python" and r["path"].startswith("tests/"))
    py_scripts_loc = sum(r["physical_LOC"] for r in records.values()
                         if r["language"] == "python" and r["path"].startswith("scripts/"))
    doc_loc = sum(r["physical_LOC"] for r in records.values() if r["language"] == "markdown")
    cfg_loc = sum(r["physical_LOC"] for r in records.values() if r["language"] == "config")

    global_maintained = py_src_loc + py_test_loc + py_scripts_loc  # owned python we maintain
    def _harness(p: str) -> bool:
        return p.startswith("collapse/") or p.startswith(".collapse/")
    collapse_tooling_loc = sum(r["physical_LOC"] for r in records.values()
                               if r["language"] == "python" and _harness(r["path"]))
    # owned MOP system python EXCLUDES the collapse harness (analysis infrastructure, tracked separately)
    global_owned_source = sum(r["physical_LOC"] for r in records.values()
                              if r["language"] == "python" and not _harness(r["path"]))

    accounting = {
        "measured_at_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "tracked_files": len(files),
        "python_files": sum(1 for r in records.values() if r["language"] == "python"),
        "global_owned_source_LOC": global_owned_source,
        "global_maintained_source_LOC": global_maintained,
        "active_src_mop_LOC": py_src_loc,
        "test_LOC": py_test_loc,
        "scripts_LOC": py_scripts_loc,
        "documentation_LOC": doc_loc,
        "configuration_LOC": cfg_loc,
        "collapse_tooling_LOC_excluded": collapse_tooling_loc,
        "loc_by_area": dict(sorted(loc_by_area.items(), key=lambda kv: -kv[1])),
        "files_by_area": dict(sorted(files_by_area.items(), key=lambda kv: -kv[1])),
        "loc_by_language": dict(sorted(loc_by_lang.items(), key=lambda kv: -kv[1])),
        "reduction_accounting": {
            "eliminated_LOC": 0, "deduplicated_LOC": 0, "relocated_LOC": 0,
            "archived_LOC": 0, "generated_replacement_LOC": 0, "added_LOC": 0,
            "net_global_reduction_LOC": 0,
        },
        "note": "baseline snapshot; reduction_accounting is zero at precheck and grows as regions collapse",
    }

    # ---- context surface (section 6) ----
    src_mods = [r for r in records.values() if r["module"]]
    entrypoints = [r["path"] for r in records.values() if r["is_entrypoint"]]
    public_symbol_count = sum(len(r["public_symbols"]) for r in records.values())
    import_edges = sum(len(v) for v in adj.values())
    directories = sorted({str(Path(f).parent) for f in files})
    context = {
        "tracked_files": len(files),
        "tracked_directories": len(directories),
        "python_modules": len(src_mods),
        "public_symbols": public_symbol_count,
        "import_edges": import_edges,
        "strongly_connected_components_gt1": len(sccs),
        "largest_scc_size": max((len(c) for c in sccs), default=0),
        "entrypoints": len(entrypoints),
        "authoritative_documents_root_md": sum(
            1 for r in records.values() if r["language"] == "markdown" and "/" not in r["path"]
        ),
        "all_markdown_documents": sum(1 for r in records.values() if r["language"] == "markdown"),
        "configuration_files": sum(1 for r in records.values() if r["language"] == "config"),
        "default_reading_bytes_root_docs": sum(
            r["bytes"] for r in records.values()
            if r["language"] == "markdown" and "/" not in r["path"]
        ),
    }

    # top duplication candidates: files sharing an identical basename stem pattern
    stem_clusters: dict[str, list[str]] = defaultdict(list)
    for rel, rec in records.items():
        if rec["language"] == "python":
            name = Path(rel).name
            for suf in ("_scaffold.py", "_impl.py", "_bed.py", "_runner.py", "_producer.py",
                        "_verifier.py", "_prereg.py", "_referee.py", "_harness.py", "_gate.py"):
                if name.endswith(suf):
                    stem_clusters[suf].append(rel)
    dup_summary = {suf: len(v) for suf, v in sorted(stem_clusters.items(), key=lambda kv: -len(kv[1]))}

    # ---- emit ----
    census = {
        "schema": "mop-collapse-census/v1",
        "root": str(ROOT),
        "commit": accounting["measured_at_commit"],
        "summary": {
            "tracked_files": len(files),
            "python_files": accounting["python_files"],
            "global_owned_source_LOC": global_owned_source,
            "global_maintained_source_LOC": global_maintained,
            "active_src_mop_LOC": py_src_loc,
            "test_LOC": py_test_loc,
            "documentation_LOC": doc_loc,
        },
        "duplication_suffix_clusters": dup_summary,
        "files": [records[k] for k in sorted(records)],
    }
    (OUT / "MOP_CODEBASE_CENSUS.json").write_text(json.dumps(census, indent=2), encoding="utf-8")
    (OUT / "MOP_IMPORT_GRAPH.json").write_text(json.dumps({
        "schema": "mop-collapse-import-graph/v1",
        "modules": len(src_mods),
        "edges": {k: sorted(v) for k, v in sorted(adj.items())},
        "reverse_edges": {k: sorted(v) for k, v in sorted(importers.items())},
        "strongly_connected_components": sorted((c for c in sccs), key=len, reverse=True),
    }, indent=2), encoding="utf-8")
    (OUT / "MOP_GLOBAL_ACCOUNTING.json").write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    (OUT / "MOP_CONTEXT_SURFACE.json").write_text(json.dumps(context, indent=2), encoding="utf-8")

    print(json.dumps({
        "tracked_files": len(files),
        "python_files": accounting["python_files"],
        "global_owned_source_LOC": global_owned_source,
        "global_maintained_source_LOC": global_maintained,
        "active_src_mop_LOC": py_src_loc,
        "test_LOC": py_test_loc,
        "scripts_LOC": py_scripts_loc,
        "documentation_LOC": doc_loc,
        "configuration_LOC": cfg_loc,
        "python_modules": len(src_mods),
        "entrypoints": len(entrypoints),
        "import_edges": import_edges,
        "cyclic_clusters_gt1": len(sccs),
        "largest_scc": context["largest_scc_size"],
        "root_md_docs": context["authoritative_documents_root_md"],
        "all_md_docs": context["all_markdown_documents"],
        "config_files": context["configuration_files"],
        "duplication_suffix_clusters": dup_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
