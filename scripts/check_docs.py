#!/usr/bin/env python
"""Frontier 6: docs drift gate. Computes the ground truth (collected pytest test count, the
experiment-registry size, the acceptance-check count) and scans README.md + STATUS.md for
hardcoded claims that CONTRADICT it: an over-claimed test count, a wrong experiment count, a
wrong acceptance ratio. Also verifies that every script path (scripts/foo.py) and make target
(make foo) named in the docs actually exists on disk / in the Makefile.

Tolerance is the whole point: a doc that states no number is fine, and in a suite that only
grows, a build-log line stating an OLD (smaller) count is behind, not wrong. Only a number that
is impossible (claims more tests than exist) or flatly disagrees (experiment count, acceptance
ratio) is flagged. check_docs()->list[str] (empty == clean); main() prints them and exits
nonzero on any problem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
DOCS = ("README.md", "STATUS.md")

# Markdown ledger (Frontier 36): the COMPLETE set of markdown the project intends to keep, so stale
# docs cannot silently regrow. canonical = doctrine (corpus + BLACKHOLE + active Studio plan);
# operational = concise summaries / references. Old generated reports and maximal-goal prompts are
# consolidated outside the repo in the project retrospective ledger, not kept as competing docs.
# Anything on disk and not in this ledger is flagged by the docs gate (consolidate it, or add it here
# deliberately).
CANONICAL_MD = (
    "developmental_jepa_corpus.md",
    "developmental_jepa_corpus_vol2.md",
    "developmental_jepa_corpus_vol3.md",
    "BLACKHOLE.md",
    "docs/STUDIO_MAXIMIZATION_2026_06_27.md",
    # the Mixture-of-Perspectives architecture review (master index + its section files)
    "docs/mixture_of_perspectives/MIXTURE_OF_THINKING.md",
    "docs/mixture_of_perspectives/EXECUTION_MANIFEST.md",
    "docs/mixture_of_perspectives/DEEP_RESEARCH_2026_07.md",
    "docs/mixture_of_perspectives/SEMANTIC_POSITIONS.md",
    "docs/mixture_of_perspectives/M3PRO_RUN_REPORT.md",
    "docs/mixture_of_perspectives/POTENTIAL_AUDIT.md",
    "docs/mixture_of_perspectives/SCAFFOLD.md",
    "docs/mixture_of_perspectives/HANDOFF.md",
    "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md",
    "docs/mixture_of_perspectives/STUDIO_TURNKEY_PLAN.md",
    "docs/mixture_of_perspectives/CONDENSATION_PLAN.md",
    "docs/mixture_of_perspectives/RESULTS_LEDGER.md",
    "docs/mixture_of_perspectives/EXPAND_PHASE_PLAN.md",
    "docs/mixture_of_perspectives/STUDIO_POTENTIAL_AUDIT.md",
    "docs/mixture_of_perspectives/STUDIO_GOAL_PROMPT.md",
    "docs/mixture_of_perspectives/01_thesis_and_definition.md",
    "docs/mixture_of_perspectives/03_thinking_modes.md",
    "docs/mixture_of_perspectives/04_reasoning_program.md",
    "docs/mixture_of_perspectives/05_plasticity_program.md",
    "docs/mixture_of_perspectives/06_cognitive_currencies_atlas.md",
    "docs/mixture_of_perspectives/07_workspace_layer.md",
    "docs/mixture_of_perspectives/08_09_custom_model_pathway_and_architectures.md",
    "docs/mixture_of_perspectives/10_compute_tiers.md",
    "docs/mixture_of_perspectives/11_experiment_registry.md",
    "docs/mixture_of_perspectives/12_metrics.md",
    "docs/mixture_of_perspectives/13_code_scaffolding.md",
    "docs/mixture_of_perspectives/15_custom_model_skepticism.md",
    "docs/mixture_of_perspectives/16_form_substrate_program.md",
    # the Form Substrate Program root docs (the active paradigm; see PARADIGM_MIGRATION.md)
    "FORM_SUBSTRATE_PROGRAM.md",
    "FORM_SUBSTRATE_DOCTRINE.md",
    "FORM_SUBSTRATE_CODEMAP.md",
    "FORM_SUBSTRATE_EXPERIMENTS.md",
    "PERFORMANCE_DENSITY_DOCTRINE.md",
    "OPERATIONAL_AWARENESS.md",
    "PARADIGM_MIGRATION.md",
    "MIGRATION_PHASES.md",
    "LEGACY_INDEX.md",
    "FORM_SUBSTRATE_IMPLEMENTATION_PLAN.md",
)
OPERATIONAL_MD = (
    "GO.md",
    "README.md",
    "STATUS.md",
    "DECISIONS.md",
    "ISSUES.md",
    "SCALING.md",
    "APPLE_SILICON.md",
    "ARCHITECTURE.md",
    "EXPERIMENTS.md",
    "STUDIO_HANDOFF.md",
    "DOCTRINE_SYNTHESIS.md",
    "CONDENSE_LEDGER.md",
    "CONDENSE_AUDIT.md",
    "CONDENSE_DOCS_REVIEW.md",
)
# archive pointer indexes (legacy docs are archived IN PLACE; statuses live in LEGACY_INDEX.md)
HISTORICAL_MD = (
    "docs/archive/mop_legacy/README.md",
    "docs/archive/vjepa_legacy/README.md",
    "docs/archive/biology_levers_legacy/README.md",
)
# Frontier 36 proof system (Section 10): the standalone proof instruments and their templates,
# registered deliberately so a missing instrument is caught while a new unledgered doc is still flagged.
PROOF_MD = (
    "proof/README.md",
    "proof/ATLAS.md",
    "proof/CORPUS_CARD.md",
    "proof/FAILURE_TAXONOMY.md",
    "proof/OBITUARIES.md",
    "proof/REPRODUCE_ONE_PLOT.md",
    "proof/DO_NOT_CITE_AS_INTELLIGENCE.md",
    "proof/NULL_CARDS/_TEMPLATE.md",
    "proof/NULL_CARDS/third_party/README.md",
    # axis-ceiling falsification-lane negative-registry entries
    "proof/NULL_CARDS/EX-A6-NUISANCE-CARRIER.md",
    "proof/NULL_CARDS/EX-AT3-TEMPORAL-CURRENCY.md",
    "proof/NULL_CARDS/EX-ROUTER-DENSITY.md",
    "proof/NULL_CARDS/EX-SUBSTRATE-SPLIT-FRAGILITY.md",
    # pre-Studio verdict cards + facet 12 (preregistration frozen this session)
    "proof/NULL_CARDS/FACET12-ROLLOUT-FIDELITY.md",
    "proof/NULL_CARDS/b5_degeneracy_robustness.md",
    "proof/NULL_CARDS/ex2_latent_planning.md",
    "proof/NULL_CARDS/e7.md",
    "proof/NULL_CARDS/ex5_local_rules_scale.md",
    "proof/NULL_CARDS/atlas_dense_multiencoder.md",
    "proof/NULL_CARDS/mop_dr1_video_cache.md",
    "proof/NULL_CARDS/pr9_long_stream_plasticity.md",
    "proof/NULL_CARDS/process_c_dense_token_pilot.md",
    "proof/NULL_CARDS/ex13_long_stream.md",
    "proof/NULL_CARDS/ex15_rejuvenation.md",
)
LEDGER_MD = frozenset(CANONICAL_MD + OPERATIONAL_MD + HISTORICAL_MD + PROOF_MD)
# directories whose markdown is tooling/output, not project docs (excluded from the ledger scan)
_MD_SKIP_DIRS = (".venv", ".git", ".claude", "runs", ".pytest_cache", ".ruff_cache", ".mypy_cache", "data")

# the canonical make targets live on the Makefile .PHONY line; these are referenced in prose too
_PHONY = re.compile(r"^\.PHONY:\s*(.+)$", re.M)
_SCRIPT_REF = re.compile(r"scripts/[A-Za-z0-9_]+\.py")
_MAKE_REF = re.compile(r"\bmake\s+([a-z][a-z0-9-]+)\b")
# studio_pipeline.py subcommands referenced in docs must be real CLI subcommands (no stale verbs)
_STUDIO_SUB_REF = re.compile(r"studio_pipeline\.py\s+([a-z][a-z-]+)")
_ADD_PARSER = re.compile(r'add_parser\(\s*"([a-z][a-z-]+)"')
_STEP = re.compile(r'step\(\s*"([^"]+)"')
# "68 tests", "108 tests green", "9 tests" (parenthetical build-log counts included)
_TESTS_CLAIM = re.compile(r"(\d+)\s+tests\b")
# "11 experiments"
_EXPERIMENTS_CLAIM = re.compile(r"(\d+)\s+experiments\b")
# "10/10" acceptance ratio (n/n form)
_RATIO_CLAIM = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def _read(name: str) -> str:
    p = ROOT / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def collected_test_count() -> int:
    """Ground-truth collected test count via `pytest --collect-only -q`.

    This pytest prints one `path: N` line per test file under -q (not `path::node` nodes), so the
    robust signal is the sum of those trailing per-file counts. We cross-check against an
    `N tests collected` / `N items` line when the run emits one and trust the larger (collection
    can be pruned by a stale cache; the per-file sum is the floor). Returns -1 if pytest cannot
    collect, so the caller can skip the comparison instead of asserting against a bogus 0.
    """
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
        # collection produced nothing parseable: only treat as real if pytest actually ran clean
        return -1 if p.returncode not in (0, 5) else 0
    return best


def experiment_registry_size() -> int:
    from mop.experiments import REGISTRY

    return len(REGISTRY)


def acceptance_check_count() -> int:
    """Distinct step("...") names in acceptance.py (try/except pairs reuse one name), which equals
    the denominator acceptance prints as N/N."""
    src = _read("scripts/acceptance.py")
    return len(set(_STEP.findall(src)))


def _make_targets() -> set[str]:
    m = _PHONY.search(_read("Makefile"))
    return set(m.group(1).split()) if m else set()


def _studio_subcommands() -> set[str]:
    """The real studio_pipeline.py subcommands, read off its argparse add_parser calls, so the docs
    are checked against the ACTUAL CLI (self-consistent, no hardcoded list to drift)."""
    return set(_ADD_PARSER.findall(_read("scripts/studio_pipeline.py")))


def _project_markdown() -> list[str]:
    """Every project markdown file (repo-relative), excluding tooling/output dirs. The ledger scan
    runs over this so a new competing-doctrine doc cannot accumulate unnoticed."""
    out = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT)
        if any(part in _MD_SKIP_DIRS for part in rel.parts):
            continue
        out.append(str(rel))
    return out


def _markdown_ledger_problems() -> list[str]:
    """Frontier 36 anti-regrowth: every markdown on disk must be in the ledger (consolidate or add it
    deliberately), and every ledgered doc must still exist (no dangling ledger entry). Runs only over
    the real repo (canonical doctrine present); a monkeypatched fixture ROOT skips it."""
    if not (ROOT / "BLACKHOLE.md").exists():
        return []  # not the real repo (a test fixture root): the ledger check does not apply
    problems: list[str] = []
    on_disk = set(_project_markdown())
    for md in sorted(on_disk - LEDGER_MD):
        problems.append(f"unexpected markdown {md} not in the ledger (consolidate it, or add to LEDGER_MD)")
    for md in sorted(LEDGER_MD - on_disk):
        problems.append(f"ledger lists {md} but it is missing on disk (update the ledger)")
    return problems


def check_docs() -> list[str]:
    """Return a list of human-readable drift problems (empty == docs are consistent)."""
    problems: list[str] = []

    real_tests = collected_test_count()
    real_experiments = experiment_registry_size()
    real_accept = acceptance_check_count()
    make_targets = _make_targets()
    studio_subs = _studio_subcommands()

    problems += _markdown_ledger_problems()  # Frontier 36: stale markdown must not regrow

    for name in DOCS:
        text = _read(name)
        if not text:
            continue

        # test counts: a claim is WRONG only if it over-claims (more tests than exist). A smaller
        # number is an older build-log snapshot of a growing suite, not a contradiction.
        if real_tests >= 0:
            for n in (int(x) for x in _TESTS_CLAIM.findall(text)):
                if n > real_tests:
                    problems.append(
                        f"{name}: claims {n} tests but only {real_tests} are collected "
                        f"(over-claim; update or remove the number)"
                    )

        # experiment count must match the registry exactly (it does not silently grow in docs)
        for n in (int(x) for x in _EXPERIMENTS_CLAIM.findall(text)):
            if n != real_experiments:
                problems.append(f"{name}: claims {n} experiments but REGISTRY has {real_experiments}")

        # acceptance ratio "k/k": a full-pass ratio whose denominator disagrees with the real
        # acceptance-check count is stale (e.g. doc says 10/10 but acceptance now has 12 steps).
        for a, b in ((int(a), int(b)) for a, b in _RATIO_CLAIM.findall(text)):
            if a == b and b != real_accept:
                problems.append(
                    f"{name}: acceptance ratio {a}/{b} but acceptance.py has {real_accept} checks"
                )

        # every referenced script path must exist on disk
        for ref in sorted(set(_SCRIPT_REF.findall(text))):
            if not (ROOT / ref).exists():
                problems.append(f"{name}: references {ref} which does not exist on disk")

        # every referenced make target must exist in the Makefile
        for tgt in sorted(set(_MAKE_REF.findall(text))):
            if tgt not in make_targets:
                problems.append(f"{name}: references `make {tgt}` not in Makefile .PHONY")

        # every studio_pipeline.py subcommand named in the docs must be a real CLI subcommand
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
