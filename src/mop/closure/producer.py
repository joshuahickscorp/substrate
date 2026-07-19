"""Produce and seal the Generation 1 General Run closure.

``run_closure`` is the single entry point that closes the program honestly. It:

1. evaluates the clean-terminal admission of the live General Run (it REFUSES while the run is not a
   clean terminal, which is the correct current behavior);
2. always rebuilds and writes the current-state authority ``proof/MOP_CURRENT_FRONTIER.json`` (this is
   a current-state fact, not gated on terminal admission);
3. calls the read-only replay to derive whatever terminal lineage exists (the derivation self-marks
   deferred while Horizon 2 is running and never fabricates a positive);
4. seals ``proof/GENERATION1_GENERAL_RUN_CLOSURE.json`` with a canonical seal, the admitted flag, the
   admission payload, and either the authoritative derivation (when admitted) or the deferral (when
   refused);
5. calls the structurally separate verifier and seals
   ``proof/GENERATION1_GENERAL_RUN_CLOSURE.verification.json``;
6. writes a markdown closure report;
7. sends a terminal Telegram digest best-effort, wrapped so that any notifier failure can never affect
   the closure.

Running it while the General Run is not terminal is expected and produces the honest current output:
``admitted = false`` with a non-empty refusal list. Re-running with the same inputs is idempotent.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mop.closure.admission import evaluate_admission
from mop.closure.frontier import build_current_frontier, write_current_frontier
from mop.closure.replay import replay_terminal_lineage
from mop.closure.report import render_closure_report
from mop.closure.verifier import verify_closure
from mop.substrate.events import canonical_bytes, canonical_sha256

CLOSURE_SCHEMA = "mop-generation1-general-run-closure/v1"
PROGRAM_ID = "generation1-general-run-closure"

# The live tree is the authority the General Run status was sealed against; admission always checks it.
LIVE_TREE = Path("/Users/scammermike/Downloads/mop")

CLOSURE_FILENAME = "GENERATION1_GENERAL_RUN_CLOSURE.json"
VERIFICATION_FILENAME = "GENERATION1_GENERAL_RUN_CLOSURE.verification.json"
REPORT_FILENAME = "GENERATION1_GENERAL_RUN_CLOSURE_REPORT.md"
FRONTIER_FILENAME = "MOP_CURRENT_FRONTIER.json"


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _final_v1_classification(runs_root: Path) -> dict[str, Any]:
    """Load Horizon 1's final sealed epoch classification (the raw artifact the verifier recomputes)."""

    cdir = runs_root / "generation1" / "generation1-successor-horizon-v1" / "classifications"
    if not cdir.is_dir():
        return {}
    files = sorted(p for p in cdir.glob("*.json") if p.is_file())
    return _load(files[-1]) if files else {}


def _v2_admission(runs_root: Path) -> dict[str, Any]:
    return _load(runs_root / "generation1" / "generation1-successor-horizon-v2" / "admission.json")


def run_closure(
    repo_root: Path,
    runs_root: Path,
    gr_root: Path,
    timestamp: str,
    telegram: bool = True,
    proof_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full closure and return a summary of what was sealed. Idempotent for fixed inputs."""

    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    gr_root = Path(gr_root).resolve()
    proof_dir = Path(proof_dir).resolve() if proof_dir is not None else (repo_root / "proof")
    proof_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean-terminal admission of the live General Run. It refuses while the run is not terminal.
    admission = evaluate_admission(root=gr_root, repo_root=LIVE_TREE, now_iso=timestamp)
    admitted = bool(admission.admitted)

    # 2. Always rebuild and write the current-state authority (not gated on terminal admission).
    frontier = build_current_frontier(repo_root=repo_root, runs_root=runs_root, timestamp=timestamp)
    frontier_path = write_current_frontier(frontier, proof_dir / FRONTIER_FILENAME)

    # 3. Read-only replay of whatever IS terminal; self-marks deferred while Horizon 2 is running.
    terminal_lineage = replay_terminal_lineage(repo_root, runs_root)

    # The raw general run state is read directly for the record and for the independent verifier.
    raw_status = _load(gr_root / "current_status.json")
    general_run_state = raw_status.get("state")

    refusals = {
        "activation_allowed": False,
        "scientific_promotion": False,
        "natural_world_generality": False,
        "independent_scientific_confirmation": False,
        "statement": (
            "Generated same-code robustness does not activate any mechanism, does not promote any "
            "result, does not generalize to the natural world, and is not an independent scientific "
            "confirmation."
        ),
    }

    if admitted:
        closure_status = "sealed_complete"
        deferral: dict[str, Any] | None = None
        derivation_authoritative = True
    else:
        closure_status = "deferred_general_run_not_terminal"
        derivation_authoritative = False
        deferral = {
            "reason": (
                "The live General Run is not a clean terminal, so the closure derivation is not yet "
                "authoritative. The terminal-lineage snapshot below records only what IS terminal."
            ),
            "refusals": list(admission.refusals),
            "general_run_state": general_run_state,
            "terminal_inputs_available": terminal_lineage.get("terminal_inputs", []),
            "nonterminal_inputs": terminal_lineage.get("nonterminal_inputs", []),
        }

    content: dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "program_id": PROGRAM_ID,
        "timestamp": timestamp,
        "closure_status": closure_status,
        "admitted": admitted,
        "derivation_authoritative": derivation_authoritative,
        "admission": admission.payload(),
        "general_run_state": general_run_state,
        "general_run_status_sha256": raw_status.get("status_sha256"),
        "terminal_lineage": terminal_lineage,
        "deferral": deferral,
        "refusals": refusals,
        "frontier_artifact": {
            "path": str(frontier_path.relative_to(repo_root))
            if _is_relative(frontier_path, repo_root)
            else str(frontier_path),
            "sha256": (frontier.get("seal") or {}).get("sha256"),
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "natural_world_generality": False,
    }
    closure_artifact = {**content, "seal": {"sha256": canonical_sha256(content)}}

    closure_path = proof_dir / CLOSURE_FILENAME
    closure_path.write_bytes(canonical_bytes(closure_artifact))

    # 5. Structurally separate verification over the raw bound artifacts.
    raw_artifacts = {
        "general_run_status": raw_status,
        "horizon_v1_final_classification": _final_v1_classification(runs_root),
        "horizon_v2_admission": _v2_admission(runs_root),
    }
    verification = verify_closure(closure_artifact, raw_artifacts)
    verification_path = proof_dir / VERIFICATION_FILENAME
    verification_path.write_bytes(canonical_bytes(verification))

    # 6. Markdown closure report.
    report_text = render_closure_report(closure_artifact)
    report_path = proof_dir / REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")

    # 7. Terminal Telegram digest, best-effort. A notifier failure never affects the closure.
    telegram_sent = False
    if telegram:
        telegram_sent = _send_digest_best_effort(closure_artifact, verification)

    return {
        "admitted": admitted,
        "closure_status": closure_status,
        "general_run_state": general_run_state,
        "refusals": list(admission.refusals),
        "closure_path": str(closure_path),
        "verification_path": str(verification_path),
        "report_path": str(report_path),
        "frontier_path": str(frontier_path),
        "seal_intact": bool(verification.get("seal_intact")),
        "mutations_all_detected": bool((verification.get("mutations_detected") or {}).get("all_detected")),
        "telegram_sent": telegram_sent,
    }


def _is_relative(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _send_digest_best_effort(closure_artifact: dict[str, Any], verification: dict[str, Any]) -> bool:
    """Send a terminal Telegram digest. Any failure is swallowed so closure is never affected."""

    try:
        from mop.studio.telegram_rung_notifier import send_message

        lineage = closure_artifact.get("terminal_lineage") or {}
        counts = (lineage.get("lane_universe") or {}).get("counts") or {}
        one_number = lineage.get("one_number_result") or {}
        text = (
            "MoP Generation 1 General Run closure\n"
            f"status: {closure_artifact.get('closure_status')}\n"
            f"admitted: {closure_artifact.get('admitted')}\n"
            f"general_run_state: {closure_artifact.get('general_run_state')}\n"
            f"result: {one_number.get('value')} {one_number.get('unit')}\n"
            f"lanes surviving/pruned/untested: "
            f"{counts.get('surviving')}/{counts.get('pruned')}/{counts.get('untested')}\n"
            f"seal_intact: {verification.get('seal_intact')}; "
            f"mutations_detected: {(verification.get('mutations_detected') or {}).get('all_detected')}"
        )
        send_message(text)
        return True
    except Exception:
        return False


def _default_timestamp(gr_root: Path) -> str:
    """A deterministic, content-derived timestamp so repeated runs are idempotent for a fixed state."""

    status = _load(gr_root / "current_status.json")
    return str(status.get("updated_at") or status.get("created_at") or "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce and seal the Generation 1 General Run closure.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the closure. Without it, nothing is written.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Worktree that holds the sealed proofs, atlas, and lineage docs.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=LIVE_TREE / "runs",
        help="Live run tree that holds the successor horizon programs.",
    )
    parser.add_argument(
        "--gr-root",
        type=Path,
        default=LIVE_TREE / "runs" / "generation1" / "general-run",
        help="Live General Run root.",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Skip the best-effort Telegram digest.",
    )
    args = parser.parse_args(argv)

    if not args.execute:
        parser.print_help()
        return 0

    timestamp = _default_timestamp(args.gr_root)
    summary = run_closure(
        repo_root=args.repo_root,
        runs_root=args.runs_root,
        gr_root=args.gr_root,
        timestamp=timestamp,
        telegram=not args.no_telegram,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
