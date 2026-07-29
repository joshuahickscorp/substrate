"""Collect campaign evidence from the ready-tag worktree into the branch.

Refuses rather than copies when anything is off. Run from the repository root
after `substrate final-revision run` exits.

Guards, in order:

1. the campaign must have exited (its run log must be non-empty)
2. the three decisive bed documents must match the committed copies on every
   field except the allowed nondeterminism (`runtime_seconds`, and the `sha256`
   computed over a document containing it) -- a scientific difference is a
   reproducibility failure and stops everything
3. the five campaign documents and the readiness manifest must all be present
4. the Grok authority, ledger and review-isolation documents are never copied:
   the run rewrites them at ready-tag vintage and the branch holds newer ones
   including the terminal review rounds
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from substrate import final_revision_io as io

WORKTREE = Path("/private/tmp/substrate-fr-ready-afeb")
RUN_LOG = Path(
    "/private/tmp/claude-503/-Users-scammermike-Downloads-substrate/"
    "8c9149d6-a426-4202-a16d-64a9f03f70f2/scratchpad/runlogs/run.log"
)
ALLOWED_NONDETERMINISM = ("runtime_seconds", "sha256")

BEDS = (
    "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json",
)
COLLECT = (
    "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_MUTATION_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json",
    "SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json",
)
NEVER_COPY = (
    "SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_REVIEW_ISOLATION.json",
)
MANIFEST = "REAL_WORLD_SANDBOX_READINESS_MANIFEST.json"


def normalized(document: dict) -> str:
    stripped = {k: v for k, v in document.items() if k not in ALLOWED_NONDETERMINISM}
    return io.digest(stripped)


def main() -> None:
    apply = "--apply" in sys.argv
    source = WORKTREE / "evidence" / "substrate" / "final_revision"
    failures: list[str] = []

    if not RUN_LOG.is_file() or not RUN_LOG.stat().st_size:
        failures.append("campaign run log is empty; the run has not exited")

    for name in BEDS:
        rerun, committed = source / name, io.EVIDENCE / name
        if not rerun.is_file():
            failures.append(f"rerun bed document absent: {name}")
            continue
        a, b = json.loads(rerun.read_text()), json.loads(committed.read_text())
        if normalized(a) != normalized(b):
            failures.append(f"REPRODUCIBILITY FAILURE: {name} differs beyond allowed nondeterminism")
        else:
            p3 = a["effects"]["P3_selected_minus_strongest_persistent_alternative"]
            print(f"  reproduced  {name}  P3={p3['mean_paired_effect']} CI={p3['confidence_interval_95']}")

    for name in COLLECT:
        if not (source / name).is_file():
            failures.append(f"campaign document absent: {name}")
    if not (WORKTREE / "artifacts" / "substrate" / "final_revision" / MANIFEST).is_file():
        failures.append(f"readiness manifest absent: {MANIFEST}")

    continuity = source / "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json"
    if continuity.is_file():
        document = json.loads(continuity.read_text())
        if document.get("meets_12_hour_minimum") is not True:
            failures.append("continuity result does not meet the 12 hour minimum")
        if document.get("all_pass") is not True:
            failures.append("continuity result did not pass")
        print(f"  continuity  {document.get('actual_duration_seconds')}s  all_pass={document.get('all_pass')}")

    mutation = source / "SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json"
    if mutation.is_file():
        document = json.loads(mutation.read_text())
        if document.get("zero_survivors") is not True:
            failures.append(f"mutation survivors present: {document.get('survivors')}")

    counterfeit = source / "SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json"
    if counterfeit.is_file():
        document = json.loads(counterfeit.read_text())
        if document.get("all_rejected") is not True:
            failures.append("not every counterfeit was rejected")

    if failures:
        print("\nREFUSED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nall guards passed")
    if not apply:
        print("dry run; pass --apply to copy")
        raise SystemExit(0)

    for name in COLLECT:
        shutil.copy2(source / name, io.EVIDENCE / name)
        print(f"  copied {name}")
    shutil.copy2(WORKTREE / "artifacts" / "substrate" / "final_revision" / MANIFEST, io.ARTIFACTS / MANIFEST)
    print(f"  copied {MANIFEST}")
    print(f"  deliberately not copied: {', '.join(NEVER_COPY)}")


if __name__ == "__main__":
    main()
