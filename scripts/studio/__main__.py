"""Single command surface for Studio receipts, gates, plans, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.logging_utils import get_logger
from mop.studio.artifact_bundle import build_artifact_index, preset_paths, write_artifact_index
from mop.studio.claim_plan import build_claim_daemon_plan, write_claim_daemon_plan
from mop.studio.density_receipt import DensityReceiptConfig, build_density_receipt, write_density_receipt
from mop.studio.disk_recovery import DiskRecoveryConfig, build_disk_recovery_plan, write_disk_recovery_plan
from mop.studio.long_run import load_plan, run_daemon, write_plan_template
from mop.studio.native_lanes import (
    build_native_lane_manifest,
    write_native_daemon_plan,
    write_native_manifest,
)
from mop.studio.objective_audit import build_studio_objective_audit, write_studio_objective_audit
from mop.studio.scorecard import (
    build_studio_scorecard,
)
from mop.studio.scorecard import (
    load_json as load_scorecard_json,
)
from mop.studio.scorecard import (
    render_markdown as render_scorecard_markdown,
)
from mop.studio.scorecard import (
    upsert_report_block as upsert_scorecard_block,
)
from mop.studio.scorecard import (
    write_json as write_scorecard_json,
)
from mop.studio.spine_plan import (
    DEFAULT_DENSE_CACHE,
    DEFAULT_DR1_CACHE,
    DEFAULT_SPINE_DIR,
    StudioSpineConfig,
    build_studio_spine_plan,
    build_studio_spine_status,
    load_studio_spine_plan,
    write_spine_wave0_plan,
    write_studio_spine_plan,
    write_studio_spine_status,
)
from mop.studio.transfer_check import (
    DEFAULT_AUDIT_PATH,
    TransferCheckConfig,
    run_transfer_check,
    write_transfer_report,
)
from mop.studio.wave0_report import (
    build_wave0_report,
)
from mop.studio.wave0_report import (
    load_json as load_wave0_json,
)
from mop.studio.wave0_report import (
    render_markdown as render_wave0_markdown,
)
from mop.studio.wave0_report import (
    upsert_report_block as upsert_wave0_block,
)
from mop.studio.wave0_report import (
    write_json as write_wave0_json,
)
from mop.studio_doctor import doctor, render_md
from mop.studio_rehearsal import DEFAULT_OUT, rehearse

log = get_logger("studio_doctor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MOP Studio command surface")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_artifact_bundle(sub)
    _add_claim_plan(sub)
    _add_daemon(sub)
    _add_density_receipt(sub)
    _add_disk_recovery(sub)
    _add_doctor(sub)
    _add_native_lanes(sub)
    _add_objective_audit(sub)
    _add_rehearse(sub)
    _add_scorecard(sub)
    _add_spine_plan(sub)
    _add_transfer_check(sub)
    _add_wave0_report(sub)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


def _add_artifact_bundle(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("artifact-bundle", help="write a durable artifact index")
    parser.add_argument(
        "--preset",
        choices=["pre-studio", "wave0", "dr1", "pr9", "atlas", "spine"],
        default="pre-studio",
    )
    parser.add_argument("--path", action="append", default=[], help="extra artifact path, repeatable")
    parser.add_argument("--only-paths", action="store_true", help="ignore the preset and use only --path")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "pre_studio.json"),
        help="artifact index JSON path",
    )
    parser.add_argument("--copy-dir", default=None, help="copy untracked small receipts into this bundle dir")
    parser.add_argument("--max-copy-mb", type=float, default=5.0)
    parser.add_argument("--require-durable", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.set_defaults(func=_cmd_artifact_bundle)


def _cmd_artifact_bundle(args: argparse.Namespace) -> int:
    paths = list(args.path) if args.only_paths else [*preset_paths(args.preset), *args.path]
    index = build_artifact_index(
        paths,
        copy_dir=args.copy_dir,
        max_copy_bytes=int(args.max_copy_mb * 1_000_000),
        require_durable=args.require_durable,
        allow_missing=args.allow_missing,
    )
    write_artifact_index(index, args.out)
    _print({"out": args.out, "all_ok": index["all_ok"], "summary": index["summary"]})
    return 0 if index["all_ok"] else 1


def _add_claim_plan(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("claim-plan", help="write a gated daemon plan for a Studio claim")
    parser.add_argument("--null-card", required=True)
    parser.add_argument("--run-receipt", required=True)
    parser.add_argument("--verifier-receipt", default=None)
    parser.add_argument("--verdict", default="PUBLISH-POSITIVE")
    parser.add_argument("--verdict-gate-out", required=True)
    parser.add_argument("--artifact-index-out", required=True)
    parser.add_argument("--artifact-path", action="append", default=[], help="extra receipt path")
    parser.add_argument("--copy-dir", default=None)
    parser.add_argument("--no-require-durable", action="store_true", help="draft only")
    parser.add_argument(
        "--ledger-cmd-json",
        required=True,
        help='JSON array command, for example ["python","-m","scripts.studio","wave0-report","--apply"]',
    )
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_claim_plan.json"))
    parser.set_defaults(func=_cmd_claim_plan)


def _cmd_claim_plan(args: argparse.Namespace) -> int:
    try:
        ledger_cmd = json.loads(args.ledger_cmd_json)
    except json.JSONDecodeError as e:
        _print({"ok": False, "error": f"ledger-cmd-json parse failed: {e}"}, err=True)
        return 2
    if not isinstance(ledger_cmd, list) or not all(isinstance(part, str) for part in ledger_cmd):
        _print({"ok": False, "error": "ledger-cmd-json must be a JSON array of strings"}, err=True)
        return 2
    try:
        plan = build_claim_daemon_plan(
            null_card=args.null_card,
            run_receipt=args.run_receipt,
            verifier_receipt=args.verifier_receipt,
            verdict=args.verdict,
            verdict_gate_out=args.verdict_gate_out,
            artifact_index_out=args.artifact_index_out,
            artifact_paths=args.artifact_path,
            copy_dir=args.copy_dir,
            require_durable=not args.no_require_durable,
            ledger_cmd=ledger_cmd,
            python=args.python,
        )
        write_claim_daemon_plan(plan, args.out)
    except ValueError as e:
        _print({"ok": False, "error": str(e)}, err=True)
        return 1
    _print({"ok": True, "out": args.out, "jobs": [job["id"] for job in plan["jobs"]]})
    return 0


def _add_daemon(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("daemon", help="profile-gated long-run Studio daemon")
    daemon = parser.add_subparsers(dest="daemon_command", required=True)
    template = daemon.add_parser("template", help="write a starter daemon plan")
    template.add_argument("--out", required=True)
    template.set_defaults(func=_cmd_daemon_template)

    validate = daemon.add_parser("validate", help="validate a daemon plan without running it")
    validate.add_argument("--plan", required=True)
    validate.set_defaults(func=_cmd_daemon_validate)

    run = daemon.add_parser("run", help="run or dry-run a daemon plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--profile", default="studio-m1ultra")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--heartbeat-min", type=float, default=5.0)
    run.add_argument("--poll-s", type=float, default=5.0)
    run.add_argument("--disk-root", default=None)
    run.set_defaults(func=_cmd_daemon_run)


def _cmd_daemon_template(args: argparse.Namespace) -> int:
    plan = write_plan_template(args.out)
    _print({"out": args.out, "jobs": len(plan["jobs"])})
    return 0


def _cmd_daemon_validate(args: argparse.Namespace) -> int:
    try:
        jobs = load_plan(Path(args.plan))
    except Exception as e:
        _print({"plan": args.plan, "ok": False, "error": str(e)}, err=True)
        return 1
    _print({"plan": args.plan, "ok": True, "jobs": [job.job_id for job in jobs]})
    return 0


def _cmd_daemon_run(args: argparse.Namespace) -> int:
    state = run_daemon(
        Path(args.plan),
        out_dir=Path(args.out_dir),
        profile_name=args.profile,
        execute=bool(args.execute),
        heartbeat_s=float(args.heartbeat_min) * 60.0,
        poll_s=float(args.poll_s),
        disk_root=Path(args.disk_root) if args.disk_root else None,
    )
    _print({"summary": state.get("summary", {}), "out_dir": args.out_dir})
    return 0 if not any(k in state.get("summary", {}) for k in ("failed", "blocked")) else 1


def _add_density_receipt(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("density-receipt", help="write the Studio density receipt")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--disk-recovery",
        default=str(REPO_ROOT / "runs" / "studio_wave0" / "disk_recovery.json"),
    )
    parser.add_argument("--largest-limit", type=int, default=25)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_wave0" / "density_receipt.json"))
    parser.set_defaults(func=_cmd_density_receipt)


def _cmd_density_receipt(args: argparse.Namespace) -> int:
    receipt = build_density_receipt(
        DensityReceiptConfig(
            repo_root=Path(args.repo_root),
            disk_recovery_path=Path(args.disk_recovery),
            largest_limit=args.largest_limit,
        )
    )
    write_density_receipt(receipt, args.out)
    _print(
        {
            "out": args.out,
            "all_ok": receipt["all_ok"],
            "workspace": {
                "total_files": receipt["workspace"]["total_files"],
                "total_human": receipt["workspace"]["total_human"],
            },
            "source_loc": {
                "total_files": receipt["source_loc"]["total_files"],
                "total_lines": receipt["source_loc"]["total_lines"],
            },
            "cleanup": {
                "deleted_human": receipt["cleanup"]["deleted_human"],
                "would_delete_human": receipt["cleanup"]["would_delete_human"],
            },
            "problems": receipt["problems"],
        }
    )
    return 0 if receipt["all_ok"] else 1


def _add_disk_recovery(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("disk-recovery", help="write a disk recovery receipt")
    parser.add_argument("--profile", default="m3pro-local-max")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--scan-path", action="append", default=[], help="extra path to classify")
    parser.add_argument("--no-defaults", action="store_true", help="scan only explicit --scan-path values")
    parser.add_argument("--out", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-class", action="append", default=[])
    parser.add_argument("--allow-path", action="append", default=[])
    parser.add_argument("--max-receipts", type=int, default=50)
    parser.set_defaults(func=_cmd_disk_recovery)


def _cmd_disk_recovery(args: argparse.Namespace) -> int:
    cfg = DiskRecoveryConfig(
        repo_root=Path(args.repo_root),
        profile_name=args.profile,
        scan_paths=tuple(args.scan_path),
        include_defaults=not args.no_defaults,
        execute=bool(args.execute),
        allow_classes=tuple(args.allow_class),
        allow_paths=tuple(args.allow_path),
        max_receipts=args.max_receipts,
    )
    report = build_disk_recovery_plan(cfg)
    if args.out:
        write_disk_recovery_plan(report, args.out)
    _print(
        {
            "out": args.out,
            "all_ok": report["all_ok"],
            "dry_run": report["dry_run"],
            "free_disk": report["free_disk"],
            "summary": report["summary"],
            "problems": report["problems"],
        }
    )
    return 0 if report["all_ok"] else 1


def _add_doctor(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("doctor", help="run the Studio readiness doctor")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", default=None, help="optional JSON receipt path")
    parser.set_defaults(func=_cmd_doctor)


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor(args.profile)
    out = REPO_ROOT / "runs" / "studio_doctor.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(report))
    if args.out:
        requested = Path(args.out)
        json_out = requested if requested.is_absolute() else REPO_ROOT / requested
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    summary = report["summary"]
    for check in report["checks"]:
        log.info("%-18s %s  %s", check["name"], "ok" if check["ok"] else "FAIL", check["detail"])
    log.info(
        "%s: %d/%d checks passed -> %s",
        "STUDIO READY" if report["all_ok"] else "NOT READY",
        summary["passed"],
        summary["total"],
        out,
    )
    _print(report)
    return 0 if report["all_ok"] else 1


def _add_native_lanes(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("native-lanes", help="list or plan Studio-native lanes")
    native = parser.add_subparsers(dest="native_command", required=True)
    list_parser = native.add_parser("list", help="evaluate lanes and print a manifest")
    _add_native_common(list_parser)
    list_parser.add_argument("--out", default=None, help="optional JSON manifest path")
    list_parser.set_defaults(func=_cmd_native_lanes_list)

    plan_parser = native.add_parser("plan", help="write a daemon plan from ready lanes")
    _add_native_common(plan_parser)
    plan_parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_native_lanes_plan.json"))
    plan_parser.add_argument(
        "--manifest-out",
        default=str(REPO_ROOT / "runs" / "studio_native_lanes_manifest.json"),
    )
    plan_parser.set_defaults(func=_cmd_native_lanes_plan)


def _add_native_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="studio-m1ultra")
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--lane", action="append", default=None)
    parser.add_argument("--clip-dir", default=None)
    parser.add_argument("--dr1-cache", default=None)
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--encode-schedule", default=str(REPO_ROOT / "runs" / "mot" / "encode_schedule.json"))
    parser.add_argument("--pr9-verdict", default=None)
    parser.add_argument("--dr1-verification", default=None)


def _native_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return build_native_lane_manifest(
        profile_name=args.profile,
        include_heavy=args.include_heavy,
        lane_ids=args.lane,
        inputs={
            "clip_dir": args.clip_dir,
            "dr1_cache": args.dr1_cache,
            "plan_path": args.plan_path,
            "encode_schedule": args.encode_schedule,
            "pr9_verdict": args.pr9_verdict,
            "dr1_verification": args.dr1_verification,
        },
    )


def _cmd_native_lanes_list(args: argparse.Namespace) -> int:
    manifest = _native_manifest(args)
    if args.out:
        write_native_manifest(manifest, args.out)
    _print(manifest)
    return 0


def _cmd_native_lanes_plan(args: argparse.Namespace) -> int:
    manifest = _native_manifest(args)
    write_native_manifest(manifest, args.manifest_out)
    try:
        plan = write_native_daemon_plan(manifest, args.out)
    except ValueError as e:
        _print({"error": str(e), "manifest": args.manifest_out}, err=True)
        return 1
    _print({"out": args.out, "manifest": args.manifest_out, "jobs": [job["id"] for job in plan["jobs"]]})
    return 0


def _add_objective_audit(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("objective-audit", help="build the Studio objective point audit")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_objective_audit_local.json"))
    parser.add_argument("--spine-status", default=None)
    parser.add_argument("--scorecard", default=None)
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.set_defaults(func=_cmd_objective_audit)


def _cmd_objective_audit(args: argparse.Namespace) -> int:
    paths = {}
    if args.spine_status:
        paths["spine_status"] = args.spine_status
    if args.scorecard:
        paths["scorecard"] = args.scorecard
    audit = build_studio_objective_audit(repo_root=REPO_ROOT, paths=paths)
    write_studio_objective_audit(audit, args.out)
    _print(
        {
            "out": args.out,
            "studio_10_ready": audit["studio_10_ready"],
            "summary": audit["summary"],
            "incomplete": [
                {
                    "id": req["id"],
                    "status": req["status"],
                    "credit": req["credit"],
                    "detail": req["detail"],
                }
                for req in audit["requirements"]
                if req["status"] != "complete"
            ],
        }
    )
    return 0 if audit["studio_10_ready"] or args.allow_not_ready else 1


def _add_rehearse(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("rehearse", help="run the Studio rehearsal capsule")
    parser.set_defaults(func=_cmd_rehearse)


def _cmd_rehearse(args: argparse.Namespace) -> int:
    summary = rehearse()
    _print(
        {
            "overall": summary["overall"],
            "stages": {stage["stage"]: stage["status"] for stage in summary["stages"]},
            "report": str(DEFAULT_OUT / "report.md"),
        }
    )
    return 0 if summary["overall"] == "pass" else 1


def _add_scorecard(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("scorecard", help="synthesize the receipt-backed Studio scorecard")
    parser.add_argument("--wave0", default=str(REPO_ROOT / "runs" / "studio_wave0" / "wave0_report.json"))
    parser.add_argument(
        "--dr1-verification",
        default=str(REPO_ROOT / "data" / "cache" / "vjepa2_vitl_comp_video" / "dr1_verification.json"),
    )
    parser.add_argument(
        "--pr9-result", default=str(REPO_ROOT / "runs" / "mot" / "pr9_continual_backprop.json")
    )
    parser.add_argument(
        "--pr9-state",
        default=str(REPO_ROOT / "runs" / "mot" / "pr9_continual_backprop.json.state.json"),
    )
    parser.add_argument("--pr9-verdict", default=str(REPO_ROOT / "runs" / "mot" / "pr9_verdict_ledger.json"))
    parser.add_argument(
        "--process-c-gate", default=str(REPO_ROOT / "runs" / "mot" / "process_c_license_gate.json")
    )
    parser.add_argument(
        "--dense-gate", default=str(REPO_ROOT / "runs" / "mot" / "dense_atlas_cache_gate.json")
    )
    parser.add_argument("--atlas", default=str(REPO_ROOT / "runs" / "mot" / "atlas_multi_encoder_grid.json"))
    parser.add_argument(
        "--atlas-verdict", default=str(REPO_ROOT / "runs" / "mot" / "atlas_verdict_ledger.json")
    )
    parser.add_argument(
        "--spine-status", default=str(REPO_ROOT / "runs" / "studio_spine" / "spine_status.json")
    )
    parser.add_argument("--artifact-index", action="append", default=[], help="name=path, repeatable")
    parser.add_argument("--dr1-cache", default="data/cache/vjepa2_vitl_comp_video")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_scorecard.json"))
    parser.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "docs" / "mixture_of_perspectives" / "STUDIO_RUN_REPORT.md"),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.set_defaults(func=_cmd_scorecard)


def _cmd_scorecard(args: argparse.Namespace) -> int:
    scorecard = build_studio_scorecard(
        wave0=load_scorecard_json(args.wave0),
        dr1_verification=load_scorecard_json(args.dr1_verification),
        pr9_result=load_scorecard_json(args.pr9_result),
        pr9_state=load_scorecard_json(args.pr9_state),
        pr9_verdict=load_scorecard_json(args.pr9_verdict),
        process_c_gate=load_scorecard_json(args.process_c_gate),
        dense_gate=load_scorecard_json(args.dense_gate),
        atlas_result=load_scorecard_json(args.atlas),
        atlas_verdict=load_scorecard_json(args.atlas_verdict),
        artifact_indexes=_artifact_indexes(args.artifact_index),
        spine_status=load_scorecard_json(args.spine_status),
        dr1_cache=args.dr1_cache,
    )
    write_scorecard_json(scorecard, args.out)
    if args.apply:
        upsert_scorecard_block(args.report_md, render_scorecard_markdown(scorecard))
    _print(
        {
            "out": args.out,
            "all_ok": scorecard["all_ok"],
            "studio_10_ready": scorecard["studio_10_ready"],
            "blockers": scorecard["blockers"],
        }
    )
    return 0 if scorecard["all_ok"] or args.allow_incomplete else 1


def _artifact_indexes(raw_items: list[str]) -> dict[str, dict | None]:
    defaults: dict[str, str | Path] = {
        "wave0": REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "wave0.json",
        "dr1": REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "dr1.json",
        "pr9": REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "pr9.json",
        "atlas": REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "atlas.json",
        "spine": REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "spine.json",
    }
    for item in raw_items:
        if "=" not in item:
            raise SystemExit(f"--artifact-index must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        defaults[name] = path
    return {name: load_scorecard_json(path) for name, path in defaults.items()}


def _add_spine_plan(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("spine-plan", help="write or inspect the staged Studio spine plan")
    parser.add_argument("--source", default=None)
    parser.add_argument("--profile", default="studio-m1ultra")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--out", default=str(REPO_ROOT / DEFAULT_SPINE_DIR / "spine_plan.json"))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan", default=str(REPO_ROOT / DEFAULT_SPINE_DIR / "spine_plan.json"))
    parser.add_argument("--status-out", default=None)
    parser.add_argument(
        "--wave0-plan-out", default=str(REPO_ROOT / DEFAULT_SPINE_DIR / "wave0_daemon_plan.json")
    )
    parser.add_argument("--dr1-cache", default=DEFAULT_DR1_CACHE)
    parser.add_argument("--source-card", default=None)
    parser.add_argument("--dense-cache", default=DEFAULT_DENSE_CACHE)
    parser.add_argument("--dense-encoder", default="vjepa21_vitl")
    parser.add_argument("--planned-clips", type=int, default=1000)
    parser.add_argument("--dense-planned-clips", type=int, default=1000)
    parser.add_argument("--pr9-seeds", default="0-9")
    parser.add_argument("--atlas-seeds", default="0-9")
    parser.set_defaults(func=_cmd_spine_plan)


def _cmd_spine_plan(args: argparse.Namespace) -> int:
    if args.status:
        try:
            plan = load_studio_spine_plan(args.plan)
            status = build_studio_spine_status(plan)
            if args.status_out:
                write_studio_spine_status(status, args.status_out)
        except Exception as e:
            _print({"ok": False, "error": str(e)}, err=True)
            return 1
        next_step = status.get("next_step")
        _print(
            {
                "ok": True,
                "plan": args.plan,
                "status_out": args.status_out,
                "summary": status["summary"],
                "all_complete": status["all_complete"],
                "next_step": None if next_step is None else _next_step(next_step),
            }
        )
        return 0

    if not args.source:
        _print({"ok": False, "error": "--source is required unless --status is set"})
        return 2

    out = Path(args.out)
    wave0_out = Path(args.wave0_plan_out)
    try:
        write_spine_wave0_plan(wave0_out)
        plan = build_studio_spine_plan(
            StudioSpineConfig(
                source=args.source,
                profile_name=args.profile,
                python=args.python,
                spine_dir=_parent_relative_dir(out),
                dr1_cache=args.dr1_cache,
                dense_cache=args.dense_cache,
                dense_encoder=args.dense_encoder,
                source_card=None if args.source_card is None else Path(args.source_card),
                planned_clips=args.planned_clips,
                dense_planned_clips=args.dense_planned_clips,
                pr9_seeds=args.pr9_seeds,
                atlas_seeds=args.atlas_seeds,
            )
        )
        display = str(_display_path(wave0_out))
        plan["subplans"]["wave0_daemon_plan"] = display
        _set_arg(plan["steps"][0]["cmd"], "--plan", display)
        plan["steps"][0]["expected_receipts"] = [display]
        _set_arg(plan["steps"][1]["cmd"], "--plan", display)
        plan["expected_receipts"] = _dedupe_receipts(plan["steps"])
        write_studio_spine_plan(plan, out)
    except Exception as e:
        _print({"ok": False, "error": str(e)}, err=True)
        return 1

    _print(
        {
            "ok": True,
            "out": str(out),
            "wave0_plan_out": str(wave0_out),
            "steps": len(plan["steps"]),
            "phases": plan["summary"]["phases"],
        }
    )
    return 0


def _next_step(next_step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": next_step["id"],
        "phase": next_step["phase"],
        "status": next_step["status"],
        "cmd": next_step["cmd"],
        "missing_receipts": next_step["missing_receipts"],
    }


def _add_transfer_check(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("transfer-check", help="run the read-only Studio transfer checklist")
    parser.add_argument("--profile", default="studio-m1ultra")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-receipts", action="store_true")
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=_cmd_transfer_check)


def _cmd_transfer_check(args: argparse.Namespace) -> int:
    audit = None if args.skip_audit else Path(args.audit_path) if args.audit_path else DEFAULT_AUDIT_PATH
    report = run_transfer_check(
        TransferCheckConfig(
            repo_root=Path(args.repo_root).resolve(),
            audit_path=audit,
            profile_name=args.profile,
            allow_dirty=bool(args.allow_dirty),
            require_receipts=not bool(args.no_receipts),
        )
    )
    if args.out:
        write_transfer_report(report, args.out)
    _print(report)
    return 0 if report["all_ok"] else 1


def _add_wave0_report(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("wave0-report", help="build or apply the Studio Wave-0 report")
    parser.add_argument(
        "--transfer", default=str(REPO_ROOT / "runs" / "studio_wave0" / "transfer_check.json")
    )
    parser.add_argument("--doctor", default=str(REPO_ROOT / "runs" / "studio_wave0" / "studio_doctor.json"))
    parser.add_argument(
        "--disk-recovery", default=str(REPO_ROOT / "runs" / "studio_wave0" / "disk_recovery.json")
    )
    parser.add_argument(
        "--daemon-state", default=str(REPO_ROOT / "runs" / "studio_wave0" / "daemon_state.json")
    )
    parser.add_argument("--encode-device", default=str(REPO_ROOT / "runs" / "mot" / "encode_device.json"))
    parser.add_argument("--encode-schedule", default=str(REPO_ROOT / "runs" / "mot" / "encode_schedule.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_wave0" / "wave0_report.json"))
    parser.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "docs" / "mixture_of_perspectives" / "STUDIO_RUN_REPORT.md"),
    )
    parser.add_argument("--apply", action="store_true")
    parser.set_defaults(func=_cmd_wave0_report)


def _cmd_wave0_report(args: argparse.Namespace) -> int:
    report = build_wave0_report(
        transfer=load_wave0_json(args.transfer),
        doctor=load_wave0_json(args.doctor),
        disk_recovery=load_wave0_json(args.disk_recovery),
        daemon_state=load_wave0_json(args.daemon_state),
        encode_device=load_wave0_json(args.encode_device),
        encode_schedule=load_wave0_json(args.encode_schedule),
    )
    write_wave0_json(report, args.out)
    block = render_wave0_markdown(report)
    if args.apply:
        upsert_wave0_block(args.report_md, block)
    _print({"out": args.out, "all_ok": report["all_ok"], "markdown": block})
    return 0 if report["all_ok"] else 1


def _parent_relative_dir(path: Path) -> Path:
    try:
        return path.parent.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return path.parent


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return path


def _dedupe_receipts(steps: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for receipt in step.get("expected_receipts", []):
            if receipt not in seen:
                seen.add(receipt)
                out.append(receipt)
    return out


def _set_arg(cmd: list[str], flag: str, value: str) -> None:
    try:
        cmd[cmd.index(flag) + 1] = value
    except (ValueError, IndexError) as e:
        raise ValueError(f"command has no {flag} argument: {cmd}") from e


def _print(payload: Any, *, err: bool = False) -> None:
    print(json.dumps(payload, indent=2, default=str), file=sys.stderr if err else sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
