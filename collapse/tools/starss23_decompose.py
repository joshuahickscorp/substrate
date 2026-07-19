"""Emit the exact top-level line-range ownership map for the STARSS23 migration.

The map partitions every physical line of every maintained STARSS23 Python file.  Classification is based
on syntax, symbol role, and explicit family policy, not whole-file suffix alone.  Mixed producer, prereg,
cache, schema, referee, and verifier files are therefore split at top-level implementation boundaries.
Ranges marked review_required must be inspected before deletion; the map is a deletion gate, not automatic
permission to remove code.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "src/mop/beds/starss23"
OUT = ROOT / "collapse/MOP_STARSS23_SOURCE_DECOMPOSITION.json"

VERIFIER_FILES = {
    "verifier.py", "count_verifier.py", "doa_verifier.py", "spatial_doa_verifier.py",
    "verifier_superflux_spectral.py", "verifier_interchannel_coherence.py",
    "count_repro_data_split_verifier.py", "count_repro_featurizer_estimator_verifier.py",
    "count_repro_gate_arch_verifier.py", "count_repro_scoring_unit_verifier.py",
}
ADAPTER_FILES = {"adapter.py", "synthetic_corpus.py", "fixtures.py", "bed.py"}
PROVIDER_HINTS = (
    "featurizer", "estimator", "gate", "labels", "referee",
)
INTEGRITY_HINTS = (
    "canonical", "sha256", "seal", "digest", "atomic", "manifest", "cache_key", "feature_bytes",
    "read_json", "write_json", "write_", "_read_", "verify_cache_bytes",
)
LIFECYCLE_HINTS = (
    "main", "build", "run", "produce", "load_or_build", "report", "payload", "artifact", "prereg",
    "sweep", "assemble", "mint", "config", "cache_dir", "load_cached", "load_feature", "harness",
)
MATH_HINTS = (
    "featur", "estimate", "reestimate", "fit", "forward", "predict", "target", "score", "metric",
    "coast", "onset", "angle", "great_circle", "mae", "f1", "flop", "pareto", "break_even", "delta",
    "sign_flip", "sesoi", "density", "majority", "match", "infer", "update",
)

OWNERS = {
    "unique_scientific_mathematics": "src/mop/beds/starss23/providers (small explicit provider modules)",
    "scientific_declaration": "src/mop/beds/starss23/experiments.py",
    "shared_lifecycle": "src/mop/science/__init__.py",
    "shared_integrity": "src/mop/substrate/events.py",
    "independent_verifier_mathematics": "same stdlib-only STARSS23 independent verifier",
    "adapter_or_dataset_logic": "src/mop/beds/starss23/adapter.py or fixture provider",
    "historical_or_compatibility_logic": "sealed compatibility reader or history index",
    "generated_or_evidence_only_material": "sealed evidence index",
}

PARITY = {
    "unique_scientific_mathematics": "known-answer provider parity and unchanged decisive mathematics",
    "scientific_declaration": "field-for-field preregistration and scientific-identity parity",
    "shared_lifecycle": "artifact, budget, seed, control, decision, stop, resume, and refusal parity",
    "shared_integrity": "canonical-byte, hash, seal, atomic-write, malformed-input, and mutation parity",
    "independent_verifier_mathematics": "separate-process replay and import-isolation parity",
    "adapter_or_dataset_logic": "source, rights, split, unit, decode, truncation, and label parity",
    "historical_or_compatibility_logic": "old authority replay and old path resolution",
    "generated_or_evidence_only_material": "byte identity or content-addressed semantic identity",
}


def _name(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        return ",".join(names) or type(node).__name__
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports"
    if isinstance(node, ast.If):
        return "__main__" if "__main__" in ast.unparse(node.test) else "conditional"
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return "module_docstring"
    return type(node).__name__


def _classify(path: Path, node: ast.AST) -> tuple[str, bool, str]:
    name = _name(node)
    lowered = name.lower()
    filename = path.name
    if name == "module_docstring" or isinstance(node, (ast.Import, ast.ImportFrom)):
        return "scientific_declaration", False, "orientation or dependency declaration"
    if name == "__main__" or lowered in {"_main", "main"}:
        return "shared_lifecycle", True, "per-module command wrapper"
    if filename in VERIFIER_FILES:
        if any(hint in lowered for hint in INTEGRITY_HINTS):
            return "shared_integrity", True, "neutral verifier integrity scaffolding"
        if any(hint in lowered for hint in LIFECYCLE_HINTS):
            return "shared_lifecycle", True, "verifier CLI, parsing, or report lifecycle"
        return "independent_verifier_mathematics", False, "independently re-derived graded logic"
    if filename in ADAPTER_FILES:
        if any(hint in lowered for hint in INTEGRITY_HINTS):
            return "shared_integrity", True, "dataset integrity primitive"
        return "adapter_or_dataset_logic", False, "source, fixture, split, decode, or label adapter"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return "scientific_declaration", False, "constant or registry declaration"
    if filename.startswith("feature_cache"):
        if any(hint in lowered for hint in INTEGRITY_HINTS):
            return "shared_integrity", True, "repeated cache integrity lifecycle"
        return "shared_lifecycle", True, "repeated feature-cache build/load lifecycle"
    if filename in {"schema.py"}:
        if any(hint in lowered for hint in INTEGRITY_HINTS):
            return "shared_integrity", True, "repeated schema integrity helper"
        return "scientific_declaration", True, "shared contract replaces family schema boilerplate"
    if filename in {"stats.py", "controls.py", "count_controls.py", "doa_controls.py"}:
        return "shared_lifecycle", True, "global decision/control provider replaces family copy"
    if any(hint in filename for hint in PROVIDER_HINTS) and not filename.endswith("_prereg.py"):
        if any(hint in lowered for hint in INTEGRITY_HINTS):
            return "shared_integrity", True, "provider-local integrity boilerplate"
        lifecycle = any(hint in lowered for hint in LIFECYCLE_HINTS)
        mathematics = any(hint in lowered for hint in MATH_HINTS)
        if lifecycle and not mathematics:
            return "shared_lifecycle", True, "provider-local lifecycle wrapper"
        return "unique_scientific_mathematics", False, "provider state or decisive mathematics"
    if any(hint in lowered for hint in INTEGRITY_HINTS):
        return "shared_integrity", True, "repeated integrity implementation"
    if any(hint in lowered for hint in MATH_HINTS):
        return "unique_scientific_mathematics", False, "mixed file contains provider mathematics"
    if any(hint in lowered for hint in LIFECYCLE_HINTS):
        return "shared_lifecycle", True, "engine-owned lifecycle"
    if isinstance(node, ast.ClassDef):
        return "scientific_declaration", True, "contract class replaced by record schema"
    return "shared_lifecycle", True, "unclassified orchestration defaults fail-closed to review"


def _ranges(path: Path) -> list[dict[str, object]]:
    source = path.read_text(encoding="utf-8")
    total = source.count("\n") + (0 if source.endswith("\n") else 1)
    tree = ast.parse(source)
    ranges: list[dict[str, object]] = []
    cursor = 1
    for node in tree.body:
        decorators = getattr(node, "decorator_list", ())
        node_start = min([node.lineno, *(decorator.lineno for decorator in decorators)])
        start = cursor
        end = int(node.end_lineno or node.lineno)
        if node_start > cursor and ranges:
            start = cursor
        category, deletion, reason = _classify(path, node)
        ranges.append({
            "start_line": start,
            "end_line": end,
            "symbol": _name(node),
            "category": category,
            "reason": reason,
            "deletion_candidate": deletion,
            "replacement_owner": OWNERS[category],
            "parity_requirement": PARITY[category],
            "rollback_path": f"git show 4dcb1e7^:{path.relative_to(ROOT)}",
            "review_required": deletion or category == "unique_scientific_mathematics",
        })
        cursor = end + 1
    if cursor <= total:
        category = ranges[-1]["category"] if ranges else "scientific_declaration"
        ranges.append({
            "start_line": cursor, "end_line": total, "symbol": "trailing_source",
            "category": category, "reason": "comments or whitespace attached to preceding authority",
            "deletion_candidate": bool(ranges[-1]["deletion_candidate"]) if ranges else False,
            "replacement_owner": OWNERS[category], "parity_requirement": PARITY[category],
            "rollback_path": f"git show 4dcb1e7^:{path.relative_to(ROOT)}", "review_required": False,
        })
    return ranges


def main() -> int:
    files = []
    totals: Counter[str] = Counter()
    for path in sorted(FAMILY.glob("*.py")):
        ranges = _ranges(path)
        for row in ranges:
            totals[row["category"]] += int(row["end_line"]) - int(row["start_line"]) + 1
        files.append({
            "path": str(path.relative_to(ROOT)),
            "physical_loc": sum(1 for _ in path.open(encoding="utf-8")),
            "ranges": ranges,
        })
    payload = {
        "schema": "mop-starss23-source-decomposition/v1",
        "family": "src/mop/beds/starss23",
        "measured_at_commit": "4dcb1e7",
        "method": "complete physical-line partition at Python top-level syntax boundaries",
        "safety": "review_required ranges are not deletion authority until their named parity gate passes",
        "categories": list(OWNERS),
        "category_loc": dict(sorted(totals.items())),
        "files": files,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(files)} files, {sum(totals.values())} LOC")
    print(json.dumps(dict(sorted(totals.items())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
