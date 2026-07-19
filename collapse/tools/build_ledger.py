"""Build the durable progress ledger from the binding spec (MOP_ACCRETION_COLLAPSE section 25).

Emits, at the worktree root:
  MOP_COLLAPSE_STATE.json   machine-readable authority: meta, baseline, and the full checklist
  MOP_COLLAPSE_LEDGER.md    human-readable ledger rendered from the same authority

Every numbered section, artifact, metric, invariant, target, gate, completion condition, and forbidden
outcome in the spec becomes a uniquely identified checklist item with the required fields. Reconciliation
is by ID: this generator is the single place the item set is defined, so nothing from the spec disappears.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLLAPSE = ROOT / "collapse"


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def load(name: str) -> dict:
    p = COLLAPSE / name
    return json.loads(p.read_text()) if p.exists() else {}


ITEM_FIELDS = ("id", "section", "kind", "title", "status", "evidence_paths", "validation",
               "commit", "rollback_tag", "dependency", "next_action")


def item(id_, section, kind, title, status="pending", evidence=None, validation="",
         commit="", rollback_tag="", dependency="", next_action=""):
    return {
        "id": id_, "section": section, "kind": kind, "title": title, "status": status,
        "evidence_paths": evidence or [], "validation": validation, "commit": commit,
        "rollback_tag": rollback_tag, "dependency": dependency, "next_action": next_action,
    }


def build_checklist() -> list[dict]:
    items: list[dict] = []
    A = items.append

    # ---- Section 2/24: workspace + git ----
    A(item("WS-1", 2, "workspace", "Isolated worktree + branch off current origin/main",
           status="complete", evidence=["/Users/scammermike/Downloads/mop-accretion-collapse"],
           validation="worktree HEAD == origin/main (a19ebe6); live tree untouched on f6d107b",
           next_action="none"))
    A(item("WS-2", 2, "workspace", "Separate build/cache/test-temp/report/pack roots",
           status="complete", evidence=[".collapse/build", ".collapse/cache", ".collapse/testtmp",
                                         ".collapse/reports", ".collapse/packs"],
           validation="created under worktree .collapse/", next_action="none"))
    A(item("WS-3", 2, "invariant", "Absolute non-interference with the live General Run (2.1)",
           status="active", validation="read-only audits only; no signal/edit/merge into live tree",
           next_action="re-verify live PID 52934 alive and untouched at every checkpoint"))
    A(item("GIT-1", 24, "git", "Draft PR against current main; not ready/merge until all gates",
           status="pending", dependency="census+precheck commit pushed",
           next_action="push precheck commit, open draft PR via gh"))

    # ---- Section 24: rollback tags ----
    for tag in ["mop-collapse-precheck", "mop-collapse-evidence", "mop-collapse-experiment-engine",
                "mop-collapse-starss23", "mop-collapse-mechanisms", "mop-collapse-controller",
                "mop-collapse-registry-config", "mop-collapse-validation", "mop-collapse-docs",
                "mop-collapse-packs", "mop-collapse-300k", "mop-collapse-250k", "mop-collapse-200k",
                "mop-collapse-150k", "mop-collapse-125k", "mop-collapse-100k", "mop-collapse-75k",
                "mop-collapse-50k", "mop-collapse-event-horizon"]:
        A(item(f"TAG-{tag}", 24, "rollback_tag", f"Create rollback tag {tag}",
               status="pending", next_action=f"tag {tag} at its green checkpoint"))

    # ---- Section 3: PR #9 disposition ----
    for sid, title, act in [
        ("PR9-1", "Inspect every PR #9 file", "diff origin/agent/mop-extreme-condensation vs main; list files"),
        ("PR9-2", "Test PR #9 controller against current main", "run its accounting controller read-only"),
        ("PR9-3", "Port or rewrite only useful mechanisms", "port LOC accounting, no-minify/no-pack gates, hydration"),
        ("PR9-4", "Discard assumptions invalidated by Generation-1 era", "record discarded assumptions"),
        ("PR9-5", "Replace active-checkout LOC metric with honest global reduction", "global accounting is primary"),
        ("PR9-6", "Open a new draft PR from current main", "gh pr create --draft"),
        ("PR9-7", "Keep PR #9 open until protections exist on replacement", "do not close prematurely"),
        ("PR9-8", "Close PR #9 only after exact retained-vs-retired mapping exists", "write mapping artifact"),
    ]:
        A(item(sid, 3, "pr9", title, next_action=act))

    # ---- Section 6: context surface artifacts + metrics ----
    A(item("ART-CONTEXT-JSON", 6, "artifact", "MOP_CONTEXT_SURFACE.json", status="complete",
           evidence=["collapse/MOP_CONTEXT_SURFACE.json"], validation="emitted by census.py",
           next_action="extend with cold_import/test_collection timings under host headroom"))
    A(item("ART-CONTEXT-MD", 6, "artifact", "MOP_CONTEXT_SURFACE.md (orientation benchmark 10 Qs)",
           next_action="render md from json + run clean-agent orientation benchmark"))
    A(item("MET-CONTEXT", 6, "metric", "Context/orientation metrics measured", status="partial",
           evidence=["collapse/MOP_CONTEXT_SURFACE.json"],
           validation="files/dirs/modules/public_symbols/import_edges/SCC/entrypoints measured",
           next_action="add reading tokens + cold_import + collection/docs-validation timings (heavy: queue)"))

    # ---- Section 7: census artifacts ----
    census_arts = {
        "MOP_CODEBASE_CENSUS.json": "complete", "MOP_CODEBASE_CENSUS.md": "pending",
        "MOP_IMPORT_GRAPH.json": "complete", "MOP_CALL_GRAPH.json": "pending",
        "MOP_COMMAND_GRAPH.json": "complete", "MOP_SCHEMA_GRAPH.json": "pending",
        "MOP_CONFIG_GRAPH.json": "pending", "MOP_TEST_OWNERSHIP.json": "pending",
        "MOP_DOCUMENTATION_GRAPH.json": "pending", "MOP_DUPLICATION_GRAPH.json": "complete",
        "MOP_AUTHORITY_GRAPH.json": "complete", "MOP_HISTORICAL_BOUNDARY.json": "pending",
        "MOP_IRREDUCIBLE_KERNEL_ESTIMATE.json": "pending", "MOP_LIVE_NO_TOUCH.json": "complete",
    }
    for name, st in census_arts.items():
        ev = [f"collapse/{name}"] if st in ("complete", "partial") else []
        A(item(f"ART-{name}", 7, "artifact", name, status=st, evidence=ev,
               next_action=("none" if st == "complete" else f"generate {name}")))
    A(item("CENSUS-CLASSIFY", 7, "census", "Classify every file into exactly one of 16 categories; unknown->0",
           status="pending", dependency="import/call/authority graphs",
           next_action="run classification over census records grounded in imports/tests/proofs/git"))

    # ---- Section 4: global accounting metrics ----
    for met in ["global_owned_source_LOC", "global_maintained_source_LOC", "active_kernel_LOC",
                "active_product_LOC", "default_validation_LOC", "optional_pack_LOC", "laboratory_LOC",
                "compatibility_LOC", "historical_source_LOC", "generated_owned_LOC", "test_LOC",
                "documentation_LOC", "configuration_LOC", "CI_build_LOC", "fixture_LOC", "third_party_LOC"]:
        st = "partial" if met in ("global_owned_source_LOC", "global_maintained_source_LOC",
                                   "test_LOC", "documentation_LOC", "configuration_LOC") else "pending"
        A(item(f"MET-{met}", 4, "metric", f"Measure {met}", status=st,
               evidence=["collapse/MOP_GLOBAL_ACCOUNTING.json"] if st == "partial" else [],
               next_action="derive from classified census" if st == "pending" else "refine via classification"))
    for met in ["eliminated_LOC", "deduplicated_LOC", "relocated_LOC", "archived_LOC",
                "generated_replacement_LOC", "added_LOC", "net_global_reduction_LOC"]:
        A(item(f"RED-{met}", 4, "reduction_metric", f"Track {met} (relocation!=elimination)",
               status="pending", next_action="update per region collapse"))

    # ---- Section 5: architectural targets + checkpoint ladder ----
    for tid, title in [
        ("TGT-KERNEL", "active kernel <=25000 (stretch 18000) LOC"),
        ("TGT-GLOBAL", "global maintained <=75000 (extreme 50000) LOC"),
        ("TGT-TESTS", "default test harness <=15000 LOC"),
        ("TGT-DOCS", "current-facing docs <=8 documents and <=8000 lines"),
        ("TGT-ENTRYPOINTS", "normal executable entrypoints <=10"),
        ("TGT-CONTROLLER", "production controllers exactly 1"),
        ("TGT-EVIDENCE", "receipt/evidence engines exactly 1"),
        ("TGT-EXPERIMENT", "experiment execution frameworks exactly 1"),
        ("TGT-REGISTRY", "capability/mechanism registries exactly 1"),
        ("TGT-CONFIG", "normal configuration roots exactly 1 typed tree"),
        ("TGT-CLI", "normal user-facing CLI exactly 1"),
    ]:
        A(item(tid, 5, "target", title, next_action="drive region collapses toward target; measure"))
    for cp in ["300k", "250k", "200k", "150k", "125k", "100k", "75k", "50k"]:
        A(item(f"CKPT-{cp}", 5, "checkpoint", f"Reach green global checkpoint {cp}",
               next_action=f"tag mop-collapse-{cp} when global maintained LOC crosses {cp}"))
    A(item("ESCAPE-RULE", 5, "gate", "Two-architecture escape rule before rejecting a lower target",
           next_action="only after 2 architectures implemented+failed for measured reasons + green restore + sealed receipt"))

    # ---- Sections 8-20: architecture work regions ----
    for sid, sec, title, act in [
        ("SEC-8", 8, "Canonical end-state architecture (core/science/mechanisms/substrate/campaign/packs/interface)",
         "converge domains without wrapper dirs"),
        ("SEC-9", 9, "One evidence authority (compact evidence core; verifier structurally independent)",
         "audit all seal/hash/encode impls; unify integrity; keep graded logic independent"),
        ("SEC-10", 10, "One experiment engine (ExperimentSpec..IndependentVerifier)",
         "build engine; simple<=150 LOC, complex<=400 LOC declarations"),
        ("SEC-11", 11, "STARSS23 first high-pressure region collapse (12-step process)",
         "prove method: parity byte-for-byte, replay, delete superseded, recovery map"),
        ("SEC-12", 12, "Mechanism-family collapse (one provider contract)",
         "replace *_scaffold/_impl/_bed/_runner boilerplate (152 files)"),
        ("SEC-13", 13, "One campaign controller (AFTER live run terminal + PR30 closure)",
         "build vs fixtures only while live; archive historical bytes; replay-equivalence then delete"),
        ("SEC-14", 14, "Entrypoint and script collapse (313 -> ~10 CLI verbs)",
         "classify scripts/; remove wrappers/bootstraps/argparse dup"),
        ("SEC-15", 15, "One registry (typed capability registry)",
         "unify experiment/mechanism/dataset/instrument/verifier registries"),
        ("SEC-16", 16, "One typed configuration authority",
         "separate frozen-identity/runtime-policy/machine-profile/overrides"),
        ("SEC-17", 17, "Validation condensation (properties/matrices/mutation; coverage-equivalence receipts)",
         "reduce handwritten test LOC; keep adversarial rigor + producer/verifier split"),
        ("SEC-18", 18, "Documentation collapse (<=8 front-door docs; sealed history index)",
         "consolidate 34 root md + 169 total; generate current tables from authorities"),
        ("SEC-19", 19, "Proof/evidence compaction (content-addressed index; no claim reduction)",
         "build evidence index; dedupe byte-identical payloads; move to packs after run releases"),
        ("SEC-20", 20, "Packs follow collapse (no pack owns a 2nd controller/engine/registry/CLI)",
         "collapse before packing; report relocation separate from elimination"),
    ]:
        A(item(sid, sec, "region", title, next_action=act))

    # ---- Section 10 experiment LOC targets ----
    A(item("TGT-EXP-SIMPLE", 10, "target", "simple experiment <=150 LOC declaration + math", next_action="enforce"))
    A(item("TGT-EXP-COMPLEX", 10, "target", "complex experiment <=400 LOC declaration + math", next_action="enforce"))

    # ---- Section 9/23: scientific invariants ----
    for inv in ["frozen_instruments", "owned_substrate_separation", "nulls", "controls",
                "independent_units", "SESOI", "multiplicity", "stop_rules", "negative_results",
                "exact_evidence_classes", "independent_scientific_recomputation",
                "no_activation_or_promotion_without_authority", "honest_hardware_boundaries",
                "crash_safe_writes", "deterministic_resume", "historical_authority_replay",
                "producer_verifier_structural_independence"]:
        A(item(f"INV-{inv}", 23, "invariant", f"Preserve invariant: {inv}",
               status="active", validation="no LOC target may weaken this",
               next_action="assert in every region parity+mutation gate"))

    # ---- Gates ----
    for gid, title in [
        ("GATE-NO-MINIFY", "no minification"), ("GATE-NO-LINE-PACK", "no line packing"),
        ("GATE-PARITY", "behavior + receipt parity"), ("GATE-MUTATION", "receipt/verifier mutation attacks"),
        ("GATE-REPLAY", "sealed proof replay"), ("GATE-CRASH-RESUME", "crash and deterministic resume"),
        ("GATE-REGEN", "deterministic regeneration"), ("GATE-COVERAGE-EQUIV", "coverage-equivalence receipt per replaced cluster"),
        ("GATE-CLEAN-CLONE", "clean clone builds+validates"), ("GATE-OFFLINE-HYDRATION", "offline pack hydration"),
        ("GATE-RELOCATION-ACCOUNTING", "relocation/archive/pack counted separately from elimination"),
        ("GATE-PERF-2PCT", "perf regressions >2% investigated"),
    ]:
        A(item(gid, 21, "gate", f"Gate: {title}", status="active",
               next_action="apply at each region checkpoint; queue heavy variants until host free"))

    # ---- Section 26: completion conditions 1-30 ----
    cc = [
        "current main and live-run identities verified", "PR #9 useful machinery ported or explicitly retired",
        "complete owned-system census exists", "unknown classifications are zero", "global accounting is honest",
        "one evidence authority remains", "one experiment engine remains", "one campaign controller remains active",
        "one registry remains", "one typed configuration authority remains", "one normal CLI remains",
        "STARSS23 framework duplication removed", "mechanism-family boilerplate removed",
        "script wrappers collapsed", "validation uses shared matrices and properties",
        "current-facing docs consolidated", "historical docs and code sealed and indexed",
        "packs contain no duplicate authorities", "sealed results remain replayable",
        "independent verifiers remain structurally independent", "crash/resume and rollback pass",
        "clean clone passes", "offline hydration passes", "no live-run source was modified",
        "full release validation passes after run releases host", "global LOC reduction measured",
        "orientation-token reduction measured", "lowest green checkpoint tagged", "rollback documented",
        "draft PR contains the complete measured result",
    ]
    for i, text in enumerate(cc, 1):
        st = "complete" if i == 24 else ("partial" if i in (1, 3) else "pending")
        A(item(f"CC-{i}", 26, "completion_condition", text, status=st,
               next_action="evidence required per spec; nothing complete from prose"))

    # ---- Section 27: forbidden outcomes 1-16 ----
    fo = ["census only", "plan only", "new abstraction beside every old abstraction", "pack-only reduction",
          "smaller default checkout with unchanged global owned code", "duplicated old and new experiment engines",
          "duplicated old and new controllers", "deferred documentation consolidation",
          "deletion candidates without deletion", "permanent wrappers around legacy implementations",
          "an under-tested generic engine", "hidden generated code", "deleted scientific evidence",
          "reduced independent-verifier rigor", "request to finish next region in another session",
          "claim of irreducible before two architectures attempted"]
    for i, text in enumerate(fo, 1):
        A(item(f"FO-{i}", 27, "forbidden_outcome", f"MUST NOT end with: {text}",
               status="active", validation="checked at final report",
               next_action="guard against; do not conclude in this state"))

    # ---- Section 28: final report items 1-30 ----
    for i in range(1, 31):
        A(item(f"RPT-{i}", 28, "report_item", f"Final report clause {i}", status="pending",
               next_action="populate from measured artifacts at conclusion"))

    return items


def main() -> int:
    census = load("MOP_CODEBASE_CENSUS.json")
    acct = load("MOP_GLOBAL_ACCOUNTING.json")
    ctx = load("MOP_CONTEXT_SURFACE.json")
    authority = load("MOP_AUTHORITY_GRAPH.json")
    command = load("MOP_COMMAND_GRAPH.json")
    dup = load("MOP_DUPLICATION_GRAPH.json")
    checklist = build_checklist()

    # live run state (read-only), for the ledger header
    live_status = {}
    live_path = Path("/Users/scammermike/Downloads/mop/runs/generation1/general-run/current_status.json")
    if live_path.exists():
        try:
            s = json.loads(live_path.read_text())
            live_status = {"state": s.get("state"), "stage": s.get("stage"),
                           "updated_at": s.get("updated_at"), "counts": s.get("counts")}
        except Exception:
            live_status = {"state": "unreadable"}

    by_status: dict[str, int] = {}
    for it in checklist:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1

    state = {
        "schema": "mop-collapse-state/v1",
        "spec": "MOP_ACCRETION_COLLAPSE.md",
        "governing_principle": ("One scientific kernel. One evidence language. One experiment engine. "
                                "One controller. One registry. One interface. Full breadth. Minimal mass."),
        "meta": {
            "branch": "agent/mop-accretion-collapse",
            "worktree": "/Users/scammermike/Downloads/mop-accretion-collapse",
            "base_commit": sh("git", "rev-parse", "HEAD"),
            "current_main": sh("git", "rev-parse", "origin/main"),
            "live_tree": "/Users/scammermike/Downloads/mop (agent/save-mop-stable-work, DO NOT TOUCH)",
            "live_general_run": live_status,
        },
        "baseline_measured": {
            "commit": (census.get("commit") or ""),
            "tracked_files": (census.get("summary") or {}).get("tracked_files"),
            "python_files": (census.get("summary") or {}).get("python_files"),
            "global_owned_source_LOC": (census.get("summary") or {}).get("global_owned_source_LOC"),
            "global_maintained_source_LOC": acct.get("global_maintained_source_LOC"),
            "active_src_mop_LOC": acct.get("active_src_mop_LOC"),
            "test_LOC": acct.get("test_LOC"),
            "scripts_LOC": acct.get("scripts_LOC"),
            "documentation_LOC": acct.get("documentation_LOC"),
            "configuration_LOC": acct.get("configuration_LOC"),
            "python_modules": ctx.get("python_modules"),
            "entrypoints": ctx.get("entrypoints"),
            "root_md_docs": ctx.get("authoritative_documents_root_md"),
            "all_md_docs": ctx.get("all_markdown_documents"),
            "duplication_suffix_clusters": census.get("duplication_suffix_clusters"),
        },
        "targets": {
            "kernel_LOC": {"primary": 25000, "stretch": 18000},
            "global_maintained_LOC": {"primary": 75000, "extreme": 50000},
            "test_LOC": {"primary": 15000},
            "docs": {"documents": 8, "lines": 8000},
            "entrypoints": 10, "controllers": 1, "evidence_engines": 1, "experiment_frameworks": 1,
            "registries": 1, "config_roots": 1, "cli": 1,
        },
        "key_findings": {
            "duplicate_integrity_primitive_definitions": (authority.get("implementation_counts") or {}),
            "duplicate_integrity_total": sum((authority.get("implementation_counts") or {}).values()),
            "scripts_class_counts": (command.get("class_counts") or {}),
            "lifecycle_boilerplate_files": dup.get("total_boilerplate_files"),
            "lifecycle_boilerplate_LOC": dup.get("total_boilerplate_LOC"),
            "highest_pressure_first_region": ("section 9 evidence authority: 168 duplicate integrity "
                                              "definitions collapse to one evidence core, provable by "
                                              "byte-parity + mutation tests (pure functions, live-safe)"),
        },
        "checklist_summary": {"total": len(checklist), "by_status": by_status},
        "checklist": checklist,
    }
    (ROOT / "MOP_COLLAPSE_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ---- render ledger md ----
    b = state["baseline_measured"]
    lines = []
    lines.append("# MOP Collapse Ledger")
    lines.append("")
    lines.append("Durable progress ledger for MOP_ACCRETION_COLLAPSE.md. Machine authority: "
                 "`MOP_COLLAPSE_STATE.json`. This file is rendered from it; edit the generator, not this file, "
                 "for structural changes. Per-checkpoint measurements are appended under History.")
    lines.append("")
    lines.append("## Governing principle")
    lines.append("")
    lines.append("> " + state["governing_principle"])
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append(f"- Branch `{state['meta']['branch']}` @ `{state['meta']['base_commit'][:7]}` "
                 f"(base = current origin/main `{state['meta']['current_main'][:7]}`).")
    lines.append(f"- Live tree: {state['meta']['live_tree']}.")
    lines.append(f"- Live General Run: {json.dumps(state['meta']['live_general_run'])}.")
    lines.append("- Only light work while the run occupies the host; heavy validation is queued.")
    lines.append("")
    lines.append("## Baseline (measured, not assumed)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ["tracked_files", "python_files", "global_owned_source_LOC", "global_maintained_source_LOC",
              "active_src_mop_LOC", "test_LOC", "scripts_LOC", "documentation_LOC", "configuration_LOC",
              "python_modules", "entrypoints", "root_md_docs", "all_md_docs"]:
        lines.append(f"| {k} | {b.get(k)} |")
    lines.append("")
    lines.append(f"Lifecycle-boilerplate suffix clusters: `{json.dumps(b.get('duplication_suffix_clusters'))}`")
    lines.append("")
    lines.append("## Checklist status")
    lines.append("")
    lines.append(f"Total items: {len(checklist)}. By status: `{json.dumps(by_status)}`.")
    lines.append("")
    lines.append("| id | sec | kind | status | title | next action |")
    lines.append("|---|---|---|---|---|---|")
    for it in checklist:
        t = it["title"].replace("|", "/")
        na = (it["next_action"] or "").replace("|", "/")
        lines.append(f"| {it['id']} | {it['section']} | {it['kind']} | {it['status']} | {t} | {na} |")
    lines.append("")
    lines.append("## History")
    lines.append("")
    lines.append("### precheck (this checkpoint)")
    lines.append("")
    lines.append(f"- commit: pending; base {state['meta']['base_commit'][:7]}")
    lines.append(f"- global_owned_source_LOC: {b.get('global_owned_source_LOC')}")
    lines.append(f"- global_maintained_source_LOC: {b.get('global_maintained_source_LOC')}")
    lines.append("- eliminated_LOC: 0; relocated_LOC: 0; archived_LOC: 0; added_LOC: (ledger+census tooling)")
    lines.append("- rollback_tag: mop-collapse-precheck (to be created at commit)")
    lines.append("- next_exact_edit: generate remaining census graphs (call/command/schema/config/authority/"
                 "historical-boundary/live-no-touch), classify unknown->0, then port PR #9 protections")
    lines.append("")
    (ROOT / "MOP_COLLAPSE_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"checklist items: {len(checklist)}")
    print(f"by status: {json.dumps(by_status)}")
    print("wrote MOP_COLLAPSE_STATE.json + MOP_COLLAPSE_LEDGER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
