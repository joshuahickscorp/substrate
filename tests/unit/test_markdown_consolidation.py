"""Frontier 36 consolidation safety: the canonical doctrine docs and referenced operational docs still
exist, old run/goal reports are gone, the markdown ledger matches disk exactly, and the docs-drift gate
(now including the anti-regrowth ledger check) stays clean."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_docs as CD  # noqa: E402


def test_canonical_doctrine_docs_exist():
    for md in CD.CANONICAL_MD:
        assert (ROOT / md).exists(), f"canonical doctrine {md} missing"


def test_referenced_docs_not_deleted():
    # consolidation must not delete docs other files link to
    for md in CD.OPERATIONAL_MD + CD.HISTORICAL_MD:
        assert (ROOT / md).exists(), f"referenced doc {md} was deleted"


def test_dead_scaffold_doc_removed():
    assert not (ROOT / "scripts" / "_scaffold_api.md").exists()


def test_markdown_ledger_matches_disk():
    on_disk = set(CD._project_markdown())
    assert on_disk == set(CD.LEDGER_MD), f"ledger/disk mismatch: {on_disk ^ set(CD.LEDGER_MD)}"


def test_no_ledger_problems():
    assert CD._markdown_ledger_problems() == []


def test_docs_gate_clean_after_consolidation():
    assert CD.check_docs() == []


def test_old_run_reports_are_consolidated_out_of_repo():
    # These now live in /Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md
    # so they cannot compete with the active Studio plan.
    for md in ("RUN_REPORT.md", "CPU_RUN_REPORT.md", "STUDIO_MAXIMAL_GOAL.md"):
        assert md not in CD.LEDGER_MD
        assert not (ROOT / md).exists(), f"{md} should be consolidated, not kept as live doctrine"
