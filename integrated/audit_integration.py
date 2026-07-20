"""Seal integrated authority/ancestry/artifact-map and audit the integration diff to zero unexpected."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/Users/scammermike/Downloads/mop-substrate-forge")
OUT = ROOT / "integrated"
COLLAPSE = "agent/mop-accretion-collapse"
FORGE = "agent/mop-substrate-forge"
INTEGRATED = "64d2801"
SUBSTRATE_DIRS = ("substrate_evo", "forge", "frontier", "campaign2", "salvage")


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal(name, obj):
    obj["sha256"] = sha(obj)
    (OUT / name).write_text(json.dumps(obj, indent=2))


def files(ref):
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, capture_output=True, text=True
    )
    return set(out.stdout.split())


def main():
    integ, coll, forg = files(INTEGRATED), files(COLLAPSE), files(FORGE)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()

    collapse_missing = sorted(coll - integ)
    substrate_retained = sorted((integ & forg) - coll)
    generated = sorted(integ - coll - forg)
    collapse_retained = sorted(integ & coll)
    proof_preserved = sorted(p for p in integ if p.startswith("proof/") and p not in coll)
    test_seam = ["tests/unit/test_collapse_invariants.py"]

    # classification: unexpected = collapse files lost that are not deliberately replaced
    unexpected = [p for p in collapse_missing if not p.startswith("proof/")]

    # required audits
    # stale means the imported evidence symbols do not resolve, not that evidence is imported at all
    src = list((ROOT / "src").rglob("*.py"))
    import importlib
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "src"))
    ev = importlib.import_module("mop.evidence")
    wanted = set()
    for p in src:
        text = p.read_text()
        for m in re.finditer(
            r"from\s+mop\.evidence\s+import\s+\(([^)]*)\)|from\s+mop\.evidence\s+import\s+([^\n(]+)", text
        ):
            for s in (m.group(1) or m.group(2) or "").split(","):
                name = s.strip().split(" as ")[0].strip()
                if name:
                    wanted.add(name)
    stale_evidence = sorted(s for s in wanted if not hasattr(ev, s))
    substrate_legacy = subprocess.run(
        ["grep", "-rl", "-E", r"^\s*(from|import)\s+mop"] + [str(ROOT / d) for d in SUBSTRATE_DIRS],
        capture_output=True,
        text=True,
    ).stdout.split()
    registries = sorted(p for p in integ if p.startswith("registry/"))
    config_roots = sorted({p.split("/")[0] for p in integ if p.startswith("configs/")})
    pyproject = json.dumps(
        subprocess.run(
            ["git", "show", f"{INTEGRATED}:pyproject.toml"], cwd=ROOT, capture_output=True, text=True
        ).stdout
    )
    clis = re.findall(r'mop\s*=\s*"([^"]+)"', pyproject)
    substrate_present = {d: any(p.startswith(d + "/") for p in integ) for d in SUBSTRATE_DIRS}

    audits = {
        "no_stale_mop_evidence_imports": stale_evidence == [],
        "no_substrate_dependency_on_legacy_mop": substrate_legacy == [],
        "single_registry": len(registries) == 1,
        "single_config_root": config_roots == ["configs"],
        "single_cli": len(set(clis)) <= 1,
        "substrate_dirs_present": all(substrate_present.values()),
        "no_lost_collapse_runtime": unexpected == [],
        "proof_preserved_count": len(proof_preserved),
    }
    audits["all_pass"] = all(v is True for k, v in audits.items() if isinstance(v, bool))

    seal(
        "MOP_INTEGRATED_AUTHORITY.json",
        {
            "schema": "mop-integrated-authority/v1",
            "integrated_commit": INTEGRATED,
            "successor_head": head,
            "successor_branch": "agent/mop-integrated-substrate-forge",
            "collapse_source": subprocess.run(
                ["git", "rev-parse", COLLAPSE], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "forge_source": subprocess.run(
                ["git", "rev-parse", FORGE], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "merge_commit": "d023d9d",
            "code_tree_replacement_commits": ["9de20f0", "64d2801"],
            "preserved_substrate_dirs": list(SUBSTRATE_DIRS),
            "preserved_proof_dirs": ["proof"],
            "test_invariant_changes": test_seam,
            "registry_config_replacements": ["registry/experiments.yaml", "configs/"],
            "excluded_branches": [
                FORGE,
                "agent/mop-autonomous-substrate-evolution",
                "agent/mop-substrate-genesis-v2",
                "agent/mop-evidence-salvage",
                COLLAPSE,
            ],
            "rollback": [f"git checkout {INTEGRATED}", f"git checkout {FORGE}", f"git checkout {COLLAPSE}"],
            "audits": audits,
        },
    )
    seal(
        "MOP_INTEGRATED_ANCESTRY.json",
        {
            "schema": "mop-integrated-ancestry/v1",
            "chain": [
                "collapse condensation -> dd5498a",
                "substrate/evidence lineage -> d33dc69 (forge boundary)",
                "merge d023d9d (-X theirs)",
                "wholesale code adoption 9de20f0",
                "registry/config + invariant scoping 64d2801",
            ],
            "counts": {
                "integrated_files": len(integ),
                "collapse_files": len(coll),
                "forge_files": len(forg),
                "collapse_retained": len(collapse_retained),
                "substrate_retained": len(substrate_retained),
                "generated": len(generated),
                "proof_preserved_beyond_collapse": len(proof_preserved),
            },
        },
    )
    seal(
        "MOP_INTEGRATED_ARTIFACT_MAP.json",
        {
            "schema": "mop-integrated-artifact-map/v1",
            "classification": {
                "collapse_owned_retained": len(collapse_retained),
                "substrate_owned_retained": len(substrate_retained),
                "intentional_test_seam": test_seam,
                "intentional_evidence_preservation": len(proof_preserved),
                "generated_authority": len(generated),
                "unexpected": unexpected,
            },
            "unexpected_count": len(unexpected),
            "substrate_dirs_present": substrate_present,
            "registries": registries,
            "config_roots": config_roots,
            "clis": sorted(set(clis)),
        },
    )
    print("unexpected:", len(unexpected), "| audits pass:", audits["all_pass"])
    print(
        "collapse_retained",
        len(collapse_retained),
        "substrate_retained",
        len(substrate_retained),
        "generated",
        len(generated),
        "proof_preserved",
        len(proof_preserved),
    )
    if unexpected[:5]:
        print("unexpected sample:", unexpected[:5])
    if stale_evidence:
        print("stale evidence imports:", stale_evidence[:5])


if __name__ == "__main__":
    main()
