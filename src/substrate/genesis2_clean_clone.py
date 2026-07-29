"""Clean-clone reproduction and raw-result collection for Genesis II."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from substrate import genesis2_io as IO2
from substrate import genesis2_statistics as S2

FREEZE = "SUBSTRATE_GENESIS2_FREEZE.json"
PRINCIPAL = "SUBSTRATE_GENESIS2_PRINCIPAL.json"
PRINCIPAL_ROWS = "SUBSTRATE_GENESIS2_PRINCIPAL_ROWS.json"
REPLICATION = "SUBSTRATE_GENESIS2_REPLICATION.json"
REPLICATION_ROWS = "SUBSTRATE_GENESIS2_REPLICATION_ROWS.json"
HIDDEN = "SUBSTRATE_GENESIS2_HIDDEN_COMPOSITION.json"
HIDDEN_ROWS = "SUBSTRATE_GENESIS2_HIDDEN_COMPOSITION_ROWS.json"
CLAIMS = "SUBSTRATE_GENESIS2_CLAIMS.json"


class CleanCloneRefused(RuntimeError):
    """Raised when the clone or its published evidence does not reproduce."""


def _self_seal_valid(document: Mapping[str, Any]) -> bool:
    body = dict(document)
    supplied = body.pop("sha256", None)
    return supplied == IO2.digest(body)


def _source_digest(repository: Path) -> str:
    rows: list[tuple[str, str]] = []
    for path in sorted((repository / "src" / "substrate").glob("genesis2*.py")):
        rows.append((str(path.relative_to(repository)), IO2.file_digest(path)))
    for path in sorted((repository / "tests" / "substrate").glob("test_genesis2*.py")):
        rows.append((str(path.relative_to(repository)), IO2.file_digest(path)))
    plan = repository / "docs" / "genesis2_master_plan.md"
    if plan.is_file():
        rows.append((str(plan.relative_to(repository)), IO2.file_digest(plan)))
    return IO2.digest(rows)


def _same_analysis(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    fields = (
        "candidate",
        "comparator",
        "histories",
        "effect",
        "confidence_lower",
        "confidence_upper",
        "p_value",
        "oracle_headroom",
        "primary_gate_pass",
        "robust_gate_pass",
    )
    return all(expected.get(field) == observed.get(field) for field in fields)


def verify(repository: Path, expected_commit: str) -> dict[str, Any]:
    """Verify committed authorities and recompute every decisive contrast."""
    repository = repository.resolve()
    evidence = repository / "evidence" / "substrate" / "genesis2"
    required = (
        FREEZE,
        PRINCIPAL,
        PRINCIPAL_ROWS,
        REPLICATION,
        REPLICATION_ROWS,
        HIDDEN,
        HIDDEN_ROWS,
        CLAIMS,
    )
    documents = {name: IO2.load_json(evidence / name) for name in required}
    frozen = documents[FREEZE]
    candidate = str(frozen["selected_candidate"])
    comparator = str(frozen["decisive_comparator"])

    recomputed: dict[str, Any] = {}
    matches: dict[str, bool] = {}
    for name, raw_name in (
        (PRINCIPAL, PRINCIPAL_ROWS),
        (REPLICATION, REPLICATION_ROWS),
        (HIDDEN, HIDDEN_ROWS),
    ):
        observed = S2.decisive_analysis(
            documents[raw_name]["rows"],
            candidate=candidate,
            comparator=comparator,
        )
        recomputed[name] = observed
        matches[name] = _same_analysis(documents[name]["analysis"], observed)

    head = IO2.git("-C", str(repository), "rev-parse", "HEAD", check=False)
    source_digest = _source_digest(repository)
    checks = {
        "expected_commit_checked_out": head == expected_commit,
        "freeze_commit_is_ancestor": subprocess.run(
            ["git", "-C", str(repository), "merge-base", "--is-ancestor", str(frozen["head_at_freeze"]), head],
            check=False,
        ).returncode
        == 0,
        "source_digest_matches_freeze": source_digest == frozen["source_digest"],
        "all_required_authorities_self_sealed": all(_self_seal_valid(document) for document in documents.values()),
        "all_authorities_inactive": not any(IO2.contains_true_activation(document) for document in documents.values()),
        "principal_raw_recomputation_exact": matches[PRINCIPAL],
        "replication_raw_recomputation_exact": matches[REPLICATION],
        "hidden_raw_recomputation_exact": matches[HIDDEN],
        "git_worktree_clean": not IO2.git("-C", str(repository), "status", "--porcelain", check=False),
    }
    return {
        "repository": str(repository),
        "expected_commit": expected_commit,
        "head": head,
        "freeze_head": frozen["head_at_freeze"],
        "source_digest": source_digest,
        "recomputed": recomputed,
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def _command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
) -> dict[str, Any]:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": list(command),
        "returncode": process.returncode,
        "stdout": process.stdout[-20_000:],
        "stderr": process.stderr[-20_000:],
        "passed": process.returncode == 0,
        "activation": False,
    }


def run(
    source_repository: Path,
    target: Path,
    expected_commit: str,
    *,
    python: Path | None = None,
) -> dict[str, Any]:
    """Clone a committed result, run independent checks, and recompute results."""
    source_repository = source_repository.resolve()
    target = target.resolve()
    if target.exists():
        raise CleanCloneRefused(f"clean-clone target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--no-local", "--quiet", str(source_repository), str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if clone.returncode != 0:
        raise CleanCloneRefused(f"git clone failed: {clone.stderr.strip()}")
    checkout = subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", "--quiet", expected_commit],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if checkout.returncode != 0:
        raise CleanCloneRefused(f"git checkout failed: {checkout.stderr.strip()}")

    interpreter = (python or Path(sys.executable)).resolve()
    tool_root = interpreter.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target / "src")
    sources = [str(path.relative_to(target)) for path in sorted((target / "src" / "substrate").glob("genesis2_*.py"))]
    commands = (
        (str(tool_root / "ruff"), "check", *sources, "tests/substrate/test_genesis2_core.py"),
        (str(tool_root / "mypy"), *sources),
        (str(interpreter), "-m", "pytest", "-q", "tests/substrate/test_genesis2_core.py"),
    )
    command_results = [_command(command, cwd=target, env=env, timeout=600) for command in commands]
    verification = verify(target, expected_commit)
    checks = {
        "clone_succeeded": clone.returncode == 0,
        "checkout_succeeded": checkout.returncode == 0,
        "quality_commands_passed": all(row["passed"] for row in command_results),
        "evidence_recomputed": verification["all_pass"],
    }
    return {
        "source_repository": str(source_repository),
        "clone_repository": str(target),
        "expected_commit": expected_commit,
        "commands": command_results,
        "verification": verification,
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--repository", type=Path, default=Path.cwd())
    verify_parser.add_argument("--expected-commit", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--source-repository", type=Path, default=Path.cwd())
    run_parser.add_argument("--target", type=Path, required=True)
    run_parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify(args.repository, args.expected_commit)
    else:
        report = run(args.source_repository, args.target, args.expected_commit)
    print(report)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
