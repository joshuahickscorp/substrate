#!/usr/bin/env python

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DOCS = ("README.md", "STATUS.md")

ACTIVE_HARDWARE_DOCS = ("README.md", "ARCHITECTURE.md", "GO.md")
_STALE_ACTIVE_HARDWARE = (
    re.compile(r"\bViT-H\b"),
    re.compile(r"\bViT-g\b"),
    re.compile(r"\bViT-G\b"),
    re.compile(r"(?<![A-Za-z0-9])vith(?![A-Za-z0-9])", re.I),
    re.compile(r"(?<![A-Za-z0-9])vitg(?![A-Za-z0-9])", re.I),
    re.compile(r"\bvit_huge\b", re.I),
    re.compile(r"(?<![A-Za-z0-9])L/H/g(?![A-Za-z0-9])", re.I),
    re.compile(r"(?<![A-Za-z0-9])H/g(?![A-Za-z0-9])", re.I),
    re.compile(
        r"\b(?:ENCODER_SCALE|REAL_ENCODER)_(?:VITH|VITG)(?:_[A-Z0-9]+)*\.json\b",
        re.I,
    ),
    re.compile(r"\b(?:VJEPA_SCALE_ATLAS_LOCAL|FACTORIZED_STIMULUS_IDENTITY)\.json\b", re.I),
    re.compile(r"\bM1 Ultra\b"),
    re.compile(r"\bStudio-gated\b", re.I),
    re.compile(r"\bstudio-scale\b", re.I),
    re.compile(r"\bgpu-later\b", re.I),
    re.compile(r"delivered Apple .*Mac Studio", re.I),
)

_VITB_PROVENANCE_TOKEN = "vjepa2_1_vitb_dist_vitG_384.pt"

CURRENT_MD = (
    "README.md",
    "ARCHITECTURE.md",
    "STATUS.md",
    "MOP_COLLAPSE_LEDGER.md",
)
LEDGER_MD = frozenset(CURRENT_MD)
_MD_SKIP_DIRS = (".venv", ".git", ".claude", "runs", ".pytest_cache", ".ruff_cache", ".mypy_cache", "data")
_MD_OPTIONAL = frozenset({"GO.md"})

_PHONY = re.compile(r"^\.PHONY:\s*(.+)$", re.M)
_SCRIPT_REF = re.compile(r"scripts/[A-Za-z0-9_]+\.py")
_MAKE_REF = re.compile(r"\bmake\s+([a-z][a-z0-9-]+)\b")
_STUDIO_SUB_REF = re.compile(r"studio_pipeline\.py\s+([a-z][a-z-]+)")
_ADD_PARSER = re.compile(r'add_parser\(\s*"([a-z][a-z-]+)"')
_STEP = re.compile(r'step\(\s*"([^"]+)"')
_TESTS_CLAIM = re.compile(r"(\d+)\s+tests\b")
_EXPERIMENTS_CLAIM = re.compile(r"(\d+)\s+experiments\b")
_RATIO_CLAIM = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def _read(name: str) -> str:
    p = ROOT / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def collected_test_count() -> int:
    p = subprocess.run(
        [PY, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = p.stdout + "\n" + p.stderr
    per_file = sum(int(m) for m in re.findall(r":\s*(\d+)\s*$", out, re.M))
    summary = 0
    m = re.search(r"(\d+)\s+tests collected", out) or re.search(r"collected\s+(\d+)\s+items?", out)
    if m:
        summary = int(m.group(1))
    best = max(per_file, summary)
    if best == 0:
        return -1 if p.returncode not in (0, 5) else 0
    return best


def experiment_registry_size() -> int:
    from mop.experiments import REGISTRY

    return len(REGISTRY)


def acceptance_check_count() -> int:
    src = _read("scripts/acceptance.py")
    return len(set(_STEP.findall(src)))


def _make_targets() -> set[str]:
    m = _PHONY.search(_read("Makefile"))
    return set(m.group(1).split()) if m else set()


def _studio_subcommands() -> set[str]:
    return set(_ADD_PARSER.findall(_read("scripts/studio_pipeline.py")))


def _project_markdown() -> list[str]:
    out = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT)
        if any(part in _MD_SKIP_DIRS for part in rel.parts):
            continue
        rel_text = str(rel)
        if rel_text in _MD_OPTIONAL:
            continue
        out.append(rel_text)
    return out


def _markdown_ledger_problems() -> list[str]:
    if not (ROOT / "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json").exists():
        return []  # not the real repo (a test fixture root): the ledger check does not apply
    problems: list[str] = []
    on_disk = set(_project_markdown())
    for md in sorted(on_disk - LEDGER_MD):
        problems.append(f"unexpected markdown {md} not in the ledger (consolidate it, or add to LEDGER_MD)")
    for md in sorted(LEDGER_MD - on_disk):
        problems.append(f"ledger lists {md} but it is missing on disk (update the ledger)")
    return problems


def _active_hardware_drift_problems() -> list[str]:
    problems: list[str] = []
    for name in ACTIVE_HARDWARE_DOCS:
        text = _read(name)
        if not text:
            continue
        text = text.replace(_VITB_PROVENANCE_TOKEN, "<official-vitb-checkpoint>")
        for pattern in _STALE_ACTIVE_HARDWARE:
            if pattern.search(text):
                problems.append(
                    f"{name}: current entrypoint contains historical hardware-first phrase "
                    f"matching {pattern.pattern!r}"
                )
    return problems


def check_docs() -> list[str]:
    problems: list[str] = []

    real_tests = collected_test_count()
    real_experiments = experiment_registry_size()
    real_accept = acceptance_check_count()
    make_targets = _make_targets()
    studio_subs = _studio_subcommands()

    problems += _markdown_ledger_problems()  # Frontier 36: stale markdown must not regrow
    problems += _active_hardware_drift_problems()

    for name in DOCS:
        text = _read(name)
        if not text:
            continue

        if real_tests >= 0:
            for n in (int(x) for x in _TESTS_CLAIM.findall(text)):
                if n > real_tests:
                    problems.append(
                        f"{name}: claims {n} tests but only {real_tests} are collected "
                        f"(over-claim; update or remove the number)"
                    )

        for n in (int(x) for x in _EXPERIMENTS_CLAIM.findall(text)):
            if n != real_experiments:
                problems.append(f"{name}: claims {n} experiments but REGISTRY has {real_experiments}")

        for line in text.splitlines():
            if "acceptance" not in line.lower():
                continue
            for a, b in ((int(a), int(b)) for a, b in _RATIO_CLAIM.findall(line)):
                if a == b and b != real_accept:
                    problems.append(
                        f"{name}: acceptance ratio {a}/{b} but acceptance.py has {real_accept} checks"
                    )

        for ref in sorted(set(_SCRIPT_REF.findall(text))):
            if not (ROOT / ref).exists():
                problems.append(f"{name}: references {ref} which does not exist on disk")

        for tgt in sorted(set(_MAKE_REF.findall(text))):
            if tgt not in make_targets:
                problems.append(f"{name}: references `make {tgt}` not in Makefile .PHONY")

        if studio_subs:  # only enforce once the CLI is parseable
            for sub in sorted(set(_STUDIO_SUB_REF.findall(text))):
                if sub not in studio_subs:
                    problems.append(
                        f"{name}: references `studio_pipeline.py {sub}` not a CLI subcommand "
                        f"(have {sorted(studio_subs)})"
                    )

    return problems


def main() -> int:
    problems = check_docs()
    if not problems:
        print(
            f"docs OK: {collected_test_count()} tests, {experiment_registry_size()} experiments, "
            f"{acceptance_check_count()} acceptance checks; no stale numbers, all refs resolve"
        )
        return 0
    print(f"DOCS DRIFT: {len(problems)} problem(s)", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
