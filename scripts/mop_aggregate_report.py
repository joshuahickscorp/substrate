#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR1_GATE = REPO_ROOT / "runs" / "pre_studio" / "pr1_mode_error_disjointness.json"

NOT_EVALUABLE_PREFIXES = ("NO ", "NO-", "NO_", "DEGENERATE", "UNREADABLE", "SKIPPED", "UNMATCHED")

EXPECTED = (
    "at4_programmatic_features",
    "at4_handcrafted_features",
    "at5_probe_class_sweep",
    "at4_programmatic_ceiling",
    "mt5_adaptive_halting",
    "mt6_confidence_stop",
    "dr9_verify_revise",
    "dr8_fixed_point_vjepa",
    "mt7_beam_search",
    "dr6_rollout_planning",
    "dr13_horizon_limit",
    "dr11_mc_rollouts",
    "dr10_retrieve_reason",
    "pr8_retrieval_head",
    "pr7_fast_slow",
    "pr4_epistemic_gate",
    "pr5_content_gated_cp",
    "pr6_sleep_consolidation",
    "al1_uncertainty_router",
    "dr12_disagreement",
    "mt8_latent_debate",
    "mt123_router_pilots",
    "dr14_corruption",
    "dr2_sparse_real_pilot",
    "ws5_slot_ablation_pilot",
    "cm4_workspace_pilot",
    "pr2_plasticity_substrates",
    "at3_time_axis",
    "dr8_fixed_point_randominit",
    "ws1_agreement_vs_confidence",
    "ws2_fusion_tournament",
    "ws4_bandwidth_sweep",
    "ws3_arbitration",
    "at1_grid_pilot",
    "al2_alignment_pilot",
)


def find_verdict_blocks(node, path: str = "") -> list[dict]:
    found: list[dict] = []
    if isinstance(node, dict):
        if {"ci", "sign_flips", "win"} <= set(node):
            ci = node.get("ci") or {}
            flips = node.get("sign_flips") or {}
            found.append(
                {
                    "at": path or "top",
                    "win": bool(node.get("win")),
                    "mean": ci.get("mean"),
                    "sd": ci.get("sd"),
                    "n_seeds": ci.get("n"),
                    "unstable": bool(ci.get("unstable", False)),
                    "any_sign_flip": bool(flips.get("any_flip", False)),
                }
            )
        for k, v in node.items():
            found.extend(find_verdict_blocks(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(find_verdict_blocks(v, f"{path}[{i}]"))
    return found


def summarize_file(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as err:
        return {"file": path.name, "experiment": path.stem, "error": f"unreadable: {err}"}
    if not isinstance(doc, dict):
        return {"file": path.name, "experiment": path.stem, "error": "not a json object"}
    blocks = find_verdict_blocks(doc)
    contract = doc.get("contract") if isinstance(doc.get("contract"), dict) else {}
    row = {
        "file": path.name,
        "experiment": str(doc.get("experiment") or doc.get("id") or path.stem),
        "null_supported": doc.get("null_supported"),
        "null_hypothesis": contract.get("null_hypothesis"),
        "n_verdict_blocks": len(blocks),
        "n_wins": sum(b["win"] for b in blocks),
        "any_sign_flip": any(b["any_sign_flip"] for b in blocks),
        "any_unstable_ci": any(b["unstable"] for b in blocks),
        "verdict_blocks": blocks,
        "seconds": doc.get("seconds"),
    }
    if doc.get("verdict") is not None and not isinstance(doc.get("verdict"), dict):
        row["verdict"] = doc.get("verdict")
    return row


def pr1_gate_context() -> dict:
    if not PR1_GATE.exists():
        return {"present": False, "note": "PR1 verdict json missing; MT rows unreadable as live tests"}
    try:
        doc = json.loads(PR1_GATE.read_text())
    except (json.JSONDecodeError, OSError) as err:
        return {"present": False, "note": f"PR1 verdict unreadable: {err}"}
    keep = {k: doc[k] for k in ("verdict", "null_supported", "experiment") if k in doc}
    return {"present": True, "path": str(PR1_GATE.relative_to(REPO_ROOT)), **keep}


def not_evaluable(row: dict) -> bool:
    v = row.get("verdict")
    return isinstance(v, str) and v.upper().startswith(NOT_EVALUABLE_PREFIXES)


def aggregate(run_dir: Path) -> dict:
    skip = {"aggregate_report.json"}
    files = sorted(p for p in run_dir.glob("*.json") if p.name not in skip)
    rows = [summarize_file(p) for p in files]
    present = {p.stem for p in files}
    missing = [name for name in EXPECTED if name not in present]
    unevaluated = [r for r in rows if not_evaluable(r)]
    nulls = [r for r in rows if r.get("null_supported") is True and not not_evaluable(r)]
    wins = [r for r in rows if r.get("null_supported") is False and not not_evaluable(r)]
    return {
        "run_dir": str(run_dir),
        "pr1_gate": pr1_gate_context(),
        "n_jsons": len(rows),
        "n_null_supported": len(nulls),
        "n_null_rejected": len(wins),
        "n_not_evaluable": len(unevaluated),
        "null_rejected_ids": sorted(r["experiment"] for r in wins),
        "not_evaluable_ids": sorted(r["experiment"] for r in unevaluated),
        "any_sign_flip_ids": sorted(r["experiment"] for r in rows if r.get("any_sign_flip")),
        "unstable_ci_ids": sorted(r["experiment"] for r in rows if r.get("any_unstable_ci")),
        "missing_expected": missing,
        "rows": rows,
    }


def render_table(report: dict) -> str:
    header = f"{'experiment':<38} {'null':<6} {'wins':<5} {'flips':<6} {'unstable':<9} seconds"
    lines = [header, "-" * len(header)]
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"{r['experiment']:<38} ERROR {r['error']}")
            continue
        null = {True: "yes", False: "NO", None: "?"}[r["null_supported"]]
        if not_evaluable(r):
            null = "n/e"
        lines.append(
            f"{r['experiment']:<38} {null:<6} {r['n_wins']:<5} "
            f"{str(r['any_sign_flip']):<6} {str(r['any_unstable_ci']):<9} {r['seconds']}"
        )
    for name in report["missing_expected"]:
        lines.append(f"{name:<38} MISSING (unrun queue row)")
    gate = report["pr1_gate"]
    lines.append(f"PR1 gate: {gate}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="aggregate MoT verdicts into runs/mot/aggregate_report.json")
    ap.add_argument("--dir", default=str(REPO_ROOT / "runs" / "mot"))
    ap.add_argument("--out", default=None, help="default: <dir>/aggregate_report.json")
    a = ap.parse_args(argv)
    run_dir = Path(a.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = aggregate(run_dir)
    out = Path(a.out) if a.out else run_dir / "aggregate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(render_table(report))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
