#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.config import REPO_ROOT
from mop.falsification.experiment_contracts import build_contract_audit, write_contract_audit
from mop.falsification.form_evidence import (
    CONTRACT_AUDIT_PATH,
    PROOF_ROOT,
    build_campaign_scorecard,
    build_form_verdict_gates,
    collect_run_receipts,
    run_fail_closed_preflights,
    run_local_campaign,
    write_null_cards,
    write_scorecard_inputs,
)
from mop.studio.artifact_bundle import build_artifact_index, preset_paths, write_artifact_index
from mop.studio.form_boundary import (
    DEFAULT_PATH as BOUNDARY_PATH,
)
from mop.studio.form_boundary import (
    build_form_pre_studio_boundary,
    write_form_pre_studio_boundary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="durable F-series evidence campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="write exact class/config/registry contract audit")
    prereg = sub.add_parser("preregister", help="write/verify registry-backed F1-F20 null cards")
    prereg.add_argument("--overwrite", action="store_true", help="explicitly replace existing cards")
    collect = sub.add_parser("collect", help="promote current implemented F runs into proof receipts")
    collect.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="EXPERIMENT_ID=RUN_DIR",
        help="override one source run directory",
    )
    sub.add_parser("preflight", help="run durable fail-closed F8/F16 mechanics preflights")
    run_local = sub.add_parser("run-local", help="run all implemented F legs at canonical local scale")
    run_local.add_argument("--only", action="append", default=[], help="run only this F id, repeatable")
    sub.add_parser("gate", help="write verdict gates; positives require independent verifiers")
    sub.add_parser("scorecard", help="write F campaign, OA-input, and density-input scorecards")
    sub.add_parser("boundary", help="write the measured pre-Studio boundary classification")
    bundle = sub.add_parser("bundle", help="write the durable form-substrate artifact index")
    bundle.add_argument(
        "--allow-missing",
        action="store_true",
        help="audit mode only; a handoff bundle should not use this",
    )
    finalize = sub.add_parser("finalize", help="audit, lock cards, collect, score, classify, and index")
    finalize.add_argument("--overwrite-cards", action="store_true")
    finalize.add_argument("--allow-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "audit":
        result = _audit()
    elif args.command == "preregister":
        result = write_null_cards(repo_root=REPO_ROOT, overwrite=bool(args.overwrite))
    elif args.command == "collect":
        result = _collect(_sources(args.source))
    elif args.command == "preflight":
        result = run_fail_closed_preflights(repo_root=REPO_ROOT)
    elif args.command == "run-local":
        result = run_local_campaign(repo_root=REPO_ROOT, experiment_ids=list(args.only) or None)
    elif args.command == "gate":
        result = build_form_verdict_gates(repo_root=REPO_ROOT)
    elif args.command == "scorecard":
        result = _scorecard()
    elif args.command == "boundary":
        result = _boundary()
    elif args.command == "bundle":
        result = _bundle(allow_missing=bool(args.allow_missing))
    elif args.command == "finalize":
        result = _finalize(overwrite_cards=bool(args.overwrite_cards), allow_missing=bool(args.allow_missing))
    else:  # pragma: no cover - argparse owns the closed command set
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("all_ok") else 1


def _audit() -> dict:
    audit = build_contract_audit(repo_root=REPO_ROOT, series="F", implemented_only=False)
    write_contract_audit(audit, REPO_ROOT / CONTRACT_AUDIT_PATH)
    return audit


def _collect(sources: dict[str, Path]) -> dict:
    result = collect_run_receipts(repo_root=REPO_ROOT, source_dirs=sources)
    write_contract_audit(result["contract_audit"], REPO_ROOT / CONTRACT_AUDIT_PATH)
    return result


def _scorecard() -> dict:
    scorecard = build_campaign_scorecard(repo_root=REPO_ROOT)
    write_scorecard_inputs(scorecard, repo_root=REPO_ROOT)
    return scorecard


def _boundary() -> dict:
    scorecard = build_campaign_scorecard(repo_root=REPO_ROOT)
    write_scorecard_inputs(scorecard, repo_root=REPO_ROOT)
    boundary = build_form_pre_studio_boundary(repo_root=REPO_ROOT, scorecard=scorecard)
    write_form_pre_studio_boundary(boundary, REPO_ROOT / BOUNDARY_PATH)
    return boundary


def _bundle(*, allow_missing: bool) -> dict:
    index = build_artifact_index(
        list(preset_paths("form-substrate")),
        repo_root=REPO_ROOT,
        require_durable=True,
        allow_missing=allow_missing,
    )
    write_artifact_index(index, REPO_ROOT / "proof/ARTIFACT_INDEX/form_substrate.json")
    return index


def _finalize(*, overwrite_cards: bool, allow_missing: bool) -> dict:
    stages: dict[str, dict] = {}
    stages["audit"] = _audit()
    stages["preregister"] = write_null_cards(repo_root=REPO_ROOT, overwrite=overwrite_cards)
    stages["preflight"] = run_fail_closed_preflights(repo_root=REPO_ROOT)
    stages["collect"] = _collect({})
    stages["verdict_gates"] = build_form_verdict_gates(repo_root=REPO_ROOT)
    stages["scorecard"] = _scorecard()
    stages["boundary_before_bundle"] = _boundary()
    stages["bundle"] = _bundle(allow_missing=allow_missing)
    stages["boundary"] = _boundary()
    stages["bundle_final"] = _bundle(allow_missing=allow_missing)
    required = (
        "audit",
        "preregister",
        "preflight",
        "collect",
        "verdict_gates",
        "scorecard",
        "boundary",
        "bundle_final",
    )
    all_ok = all(stages[name].get("all_ok", False) for name in required)
    return {
        "schema": "mop-form-campaign-finalize/v1",
        "proof_root": str(PROOF_ROOT),
        "stages": stages,
        "all_ok": all_ok,
    }


def _sources(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        eid, sep, raw_path = value.partition("=")
        if not sep or not eid or not raw_path:
            raise ValueError(f"--source must be EXPERIMENT_ID=RUN_DIR, got {value!r}")
        if eid in out:
            raise ValueError(f"duplicate --source for {eid}")
        out[eid] = Path(raw_path)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
