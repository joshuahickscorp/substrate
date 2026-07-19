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
         "deletion map ready (collapse/MOP_EVIDENCE_EQUIVALENCE.json): 64 byte-identical primitive defs "
         "collapsible onto one core; implement core, redirect, delete, run parity+mutation+replay "
         "(HEAVY: queue behind live run per section 2)"),
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
    equiv = load("MOP_EVIDENCE_EQUIVALENCE.json")
    redlog = load("MOP_REDUCTION_LOG.json")
    starss = load("MOP_STARSS23_ANATOMY.json")
    architecture = load("MOP_STARSS23_ARCHITECTURE_COMPARISON.json")
    decomposition = load("MOP_STARSS23_SOURCE_DECOMPOSITION.json")
    checklist = build_checklist()

    # reconcile: STARSS23 vertical slice, engine built, floor corrected
    for it in checklist:
        if it["id"] == "SEC-11":
            it["status"] = "active"
            it["evidence_paths"] = ["collapse/MOP_STARSS23_ANATOMY.json", "src/mop/science/",
                                    "collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json",
                                    "collapse/MOP_STARSS23_SOURCE_DECOMPOSITION.json",
                                    "src/mop/beds/starss23/experiments.py",
                                    "src/mop/beds/starss23/feature_cache.py",
                                    "tests/unit/test_science_engine.py"]
            it["validation"] = (
                f"measured collapsible={starss.get('collapsible_loc')} "
                f"preserved={starss.get('preserved_loc')} "
                "per_axis_declaration~21; architecture B selected after implemented A/B comparison; "
                "29/29 selected-engine tests and 22/22 B mutation attacks; cache/control/statistics "
                "cluster physically deleted with net owned Python reduction of 1174 LOC; three "
                "matched-budget harnesses deleted in favor of one policy engine, net 1138 LOC; "
                "producer budget projection and canonical writes centralized, net 378 LOC; producer "
                "result, receipt, finalization, seed-record, and prereg-write paths centralized, net 316 LOC; "
                "statistics, noisy-TV controls, and safety projections centralized, net 41 LOC; common "
                "artifact envelopes and matched-budget provenance centralized, net 31 LOC; four count "
                "seed execution loops centralized with exact provider-specific records, net 191 LOC; native "
                "dev/fold splits and one-time corpus provider mapping centralized, net 91 LOC; causal input, "
                "gate-trace, and marginal-noise lifecycle centralized, net 73 LOC; producer fire-spread "
                "diagnostics and sealed family prereg reads centralized, net 108 LOC; four family prereg "
                "analysis plans centralized with local family declarations retained, net 138 LOC; family "
                "label-fact, Bonferroni, and CLI projections centralized, net 28 LOC; exact-clone Hann, "
                "mel-warp, and triangular-filterbank DSP centralized, net 74 LOC; domain-separated "
                "control and producer seed derivation centralized with exact streams, net 21 LOC; arm FLOP "
                "projection, count provider binding, and onset density centralized, net 34 LOC; frozen "
                "provider parameter, feature-byte, and frame-cost introspection centralized, net 75 LOC; "
                "topology-neutral count and DoA gate interfaces centralized, net 58 LOC; three frozen-"
                "featurizer producer lifecycles and spread projections centralized behind local evidence "
                "declarations, net 260 LOC; five concurrent-count producer artifact lifecycles centralized "
                "behind provider, split, preregistration, scoring, and readout declarations, net 407 LOC; "
                "base onset, refractory-NMS, and learning-progress producer lifecycles centralized behind "
                "the onset executor, and embedded STARSS narrative documentation removed, net 3320 LOC; "
                "four family prereg assemblers centralized, eight local command wrappers deleted, and "
                "remaining comment-only STARSS documentation removed, net 1717 LOC")
            it["dependency"] = ("physical deletion of *_producer/*_harness needs sealed-artifact parity; "
                                "heavy real-audio validation remains deferred to host headroom")
            it["next_action"] = ("finish the remaining dual-architecture DoA producer shell and measure "
                                 "the residual STARSS lifecycle surface")
        if it["id"] == "SEC-10":
            it["status"] = "active"
            it["evidence_paths"] = ["src/mop/science/", "collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json"]
            it["validation"] = ("Architecture B selected: one 198-LOC sealed-record interpreter; "
                                 "Architecture A physically deleted")

    checklist.append(item(
        "ART-MOP_STARSS23_ARCHITECTURE_COMPARISON.json", 11, "artifact",
        "MOP_STARSS23_ARCHITECTURE_COMPARISON.json (implemented A/B selection)", status="complete",
        evidence=["collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json"],
        validation="four axes in both designs; LOC/API/dependency/import/runtime/audit/mutation measured",
        next_action="none"))
    checklist.append(item(
        "ART-MOP_STARSS23_SOURCE_DECOMPOSITION.json", 11, "artifact",
        "MOP_STARSS23_SOURCE_DECOMPOSITION.json (complete line-range ownership map)", status="complete",
        evidence=["collapse/MOP_STARSS23_SOURCE_DECOMPOSITION.json",
                  "collapse/tools/starss23_decompose.py"],
        validation=(f"{len(decomposition.get('files') or [])} files and "
                    f"{sum(len(f.get('ranges') or []) for f in (decomposition.get('files') or []))} "
                    "top-level ranges partition every physical line exactly once"),
        next_action="use named parity and rollback fields as the deletion gate for each cluster"))
    checklist.append(item(
        "RED-starss23-architecture-b", 10, "verified_reduction",
        "Select Architecture B and physically delete Architecture A", status="verified",
        evidence=["collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json",
                  "collapse/MOP_REDUCTION_LOG.json"],
        validation="539 replaced Python LOC, 402 added, net -137; 29/29 focused green",
        rollback_tag="mop-collapse-starss23-architecture-b",
        next_action="wire real STARSS23 providers and delete superseded family lifecycle"))
    checklist.append(item(
        "RED-starss23-cache-controls-statistics", 11, "verified_reduction",
        "Collapse STARSS23 cache, control, and producer-statistics lifecycle", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/feature_cache.py",
                  "src/mop/science/statistics.py", "tests/unit/test_starss23_feature_cache.py"],
        validation=("1816 replaced Python LOC, 642 added, net -1174; historical cache identities, "
                    "crash-safe writes, exact statistics, controls, and full focused STARSS suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-1",
        next_action="collapse the three STARSS23 budget harnesses onto one shared implementation"))
    checklist.append(item(
        "RED-starss23-matched-budget-harnesses", 11, "verified_reduction",
        "Replace three STARSS23 matched-budget harnesses with one policy engine", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/budget.py",
                  "src/mop/beds/starss23/experiments.py", "tests/unit/test_starss23_harness.py",
                  "tests/unit/test_starss23_counting_bed.py", "tests/unit/test_starss23_doa_bed.py"],
        validation=("2142 replaced Python LOC, 1004 added, net -1138; onset/count/DoA payloads "
                    "byte-equal and old canonical digests pinned; full focused STARSS suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-2",
        next_action="centralize producer budget projection and crash-safe canonical artifact writes"))
    checklist.append(item(
        "RED-starss23-producer-projection-writes", 11, "verified_reduction",
        "Centralize STARSS23 producer budget projection and canonical artifact writes", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/budget.py",
                  "src/mop/substrate/events.py", "tests/unit/test_starss23_harness.py",
                  "tests/unit/test_starss23_end_to_end.py"],
        validation=("618 replaced Python LOC, 240 added, net -378; exact BudgetPoint projection, "
                    "canonical byte sealing, crash-safe replacement, and full focused STARSS suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-3",
        next_action="centralize producer results, receipts, seed records, and preregistration writes"))
    checklist.append(item(
        "RED-starss23-producer-results-receipts", 11, "verified_reduction",
        "Centralize STARSS23 producer results, receipts, seed records, and preregistration writes",
        status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/__init__.py",
                  "src/mop/science/budget.py", "src/mop/substrate/events.py",
                  "tests/unit/test_science_engine.py", "tests/unit/test_starss23_end_to_end.py"],
        validation=("586 replaced Python LOC, 270 added, net -316; canonical nonmutating finalization, "
                    "exact evidence-digest receipts, crash-safe prereg writes, and focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-4",
        next_action="centralize producer statistics, noisy-TV controls, and safety projections"))
    checklist.append(item(
        "RED-starss23-producer-projections", 11, "verified_reduction",
        "Centralize STARSS23 producer statistics, controls, and safety projections", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/__init__.py",
                  "src/mop/science/budget.py", "src/mop/science/statistics.py",
                  "tests/unit/test_starss23_stats.py", "tests/unit/test_starss23_harness.py"],
        validation=("393 replaced Python LOC, 352 added, net -41; production source net -110; exact "
                    "statistics/control shapes, fresh closed safety flags, and focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-5",
        next_action="collapse the common artifact envelope across all thirteen producers"))
    checklist.append(item(
        "RED-starss23-artifact-envelopes", 11, "verified_reduction",
        "Centralize STARSS23 producer artifact envelopes and matched-budget provenance", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/__init__.py",
                  "tests/unit/test_science_engine.py", "tests/unit/test_starss23_end_to_end.py"],
        validation=("649 replaced Python LOC, 618 added, net -31; production source net -70; 13/13 "
                    "field inventories and migrated expressions exact; closed-authority mutation refused; "
                    "full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-6",
        next_action="measure and collapse the next repeated producer execution lifecycle"))
    checklist.append(item(
        "RED-starss23-count-seed-lifecycle", 11, "verified_reduction",
        "Centralize STARSS23 counting per-seed execution lifecycle", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json",
                  "src/mop/beds/starss23/count_producer.py",
                  "tests/unit/test_starss23_counting_bed.py"],
        validation=("372 replaced Python LOC, 181 added, net -191; production source net -234; complete "
                    "legacy BudgetSeedRun digests exact across micro, clip-macro, swapped-provider, and "
                    "alternate-gate axes; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-7",
        next_action="collapse repeated room-disjoint split and corpus-preparation lifecycle"))
    checklist.append(item(
        "RED-starss23-native-split-corpus-map", 11, "verified_reduction",
        "Centralize STARSS23 native split and one-time corpus provider mapping", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/adapter.py",
                  "tests/unit/test_starss23_adapter.py"],
        validation=("245 replaced Python LOC, 154 added, net -91; production source net -106; onset, "
                    "count, and DoA pre/post split projections byte-identical; swapped-fold reproduction "
                    "remains independent; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-8",
        next_action="collapse repeated causal gate passes and marginal-matched noise controls"))
    checklist.append(item(
        "RED-starss23-causal-gate-noise", 11, "verified_reduction",
        "Centralize STARSS23 causal gate and marginal-noise lifecycle", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/gating.py",
                  "src/mop/beds/starss23/adapter.py", "tests/unit/test_starss23_counting_bed.py"],
        validation=("226 replaced Python LOC, 153 added, net -73; production source net -84; nine "
                    "pre/post numerical hashes exact across onset/count/DoA state inputs, traces, events, "
                    "and seeded noise; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-9",
        next_action="collapse repeated fire-spread diagnostics and sealed prereg readers"))
    checklist.append(item(
        "RED-starss23-fire-spread-prereg-read", 11, "verified_reduction",
        "Centralize STARSS23 producer fire-spread and sealed prereg reads", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/referee.py",
                  "src/mop/science/__init__.py", "tests/unit/test_starss23_referee.py",
                  "tests/unit/test_science_engine.py"],
        validation=("305 replaced Python LOC, 197 added, net -108; production source net -159; four "
                    "spread-artifact and four checked-in prereg body hashes exact; local refusal surfaces "
                    "preserved; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-10",
        next_action="measure and collapse the next repeated STARSS producer/prereg lifecycle"))
    checklist.append(item(
        "RED-starss23-family-prereg-plan", 11, "verified_reduction",
        "Centralize STARSS23 family preregistration analysis plans", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/prereg.py",
                  "tests/unit/test_starss23_featurizer_spatial_doa.py"],
        validation=("338 replaced Python LOC, 200 added, net -138; production source net -163; complete "
                    "outer-body and embedded seals exact across all four family preregistrations; local "
                    "multiplicity rationales and hypotheses preserved; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-11",
        next_action="measure remaining family structural-fact and prereg CLI lifecycle"))
    checklist.append(item(
        "RED-starss23-family-prereg-facts-cli", 11, "verified_reduction",
        "Centralize STARSS23 family prereg facts, multiplicity, and CLI projections", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/prereg.py"],
        validation=("170 replaced Python LOC, 142 added, net -28; four complete body/seal pairs and four "
                    "CLI summary hashes exact; hand-computable split facts exact; local data sources and "
                    "family vocabulary preserved; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-12",
        next_action="collapse exact-clone frozen spectral primitives"))
    checklist.append(item(
        "RED-starss23-frozen-spectral-primitives", 11, "verified_reduction",
        "Centralize STARSS23 frozen spectral primitives", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/featurizer.py",
                  "tests/unit/test_starss23_featurizer.py"],
        validation=("121 replaced Python LOC, 47 added, net -74; production source net -89; periodic-Hann, "
                    "64-mel and 32-mel bytes exact; seven complete frontend parameter digests exact; full "
                    "focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-13",
        next_action="collapse repeated noisy-TV namespace and frontend wrappers"))
    checklist.append(item(
        "RED-starss23-domain-seeds", 11, "verified_reduction",
        "Centralize STARSS23 domain-separated seed derivation", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/adapter.py",
                  "tests/unit/test_starss23_adapter.py"],
        validation=("82 replaced Python LOC, 61 added, net -21; production source net -37; exact "
                    "uint32 producer seeds across ordinary and oversized roots, rate-matched-random "
                    "positions, RndTarget matrix bytes, and aleatoric-control bytes preserved; full "
                    "focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-14",
        next_action="measure remaining repeated producer FLOP models and onset-density wrappers"))
    checklist.append(item(
        "RED-starss23-flop-provider-projection", 11, "verified_reduction",
        "Centralize STARSS23 arm FLOP projection and count provider binding", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/science/budget.py",
                  "src/mop/beds/starss23/count_producer.py", "tests/unit/test_starss23_harness.py"],
        validation=("163 replaced Python LOC, 129 added, net -34; production source net -54; ten complete "
                    "arm-charge tables exact, alternate count gate shares one binding by identity, real "
                    "producer/verifier path green, and independent verifier formulas remain local"),
        rollback_tag="mop-collapse-starss23-lifecycle-15",
        next_action="collapse repeated frozen-provider introspection methods"))
    checklist.append(item(
        "RED-starss23-frozen-provider-introspection", 11, "verified_reduction",
        "Centralize STARSS23 frozen-provider introspection", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/adapter.py",
                  "tests/unit/test_starss23_featurizer.py", "tests/unit/test_starss23_counting_bed.py"],
        validation=("130 replaced Python LOC, 55 added, net -75; ten dataclass and slot layouts, ten "
                    "parameter digests, seven feature digests, seven frame costs, and all refusal types "
                    "and messages exact; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-16",
        next_action="measure common causal gate state, forward, and decision kernels"))
    checklist.append(item(
        "RED-starss23-topology-neutral-gate-interfaces", 11, "verified_reduction",
        "Centralize STARSS23 topology-neutral gate interfaces", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/count_gate.py",
                  "src/mop/beds/starss23/doa_gate.py", "tests/unit/test_starss23_doa_bed.py"],
        validation=("126 replaced Python LOC, 68 added, net -58; four parameter digests, probability "
                    "vectors, online decisions, refusal surfaces, and count report object shapes exact; "
                    "forward topologies and optimizers remain local; full focused suite green"),
        rollback_tag="mop-collapse-starss23-lifecycle-17",
        next_action="remeasure remaining variant producer shells against the declaration target"))
    checklist.append(item(
        "RED-starss23-frozen-featurizer-variant-lifecycle", 11, "verified_reduction",
        "Centralize frozen-featurizer producer execution and spread projections", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json",
                  "src/mop/beds/starss23/featurizer_variant_producer.py",
                  "tests/unit/test_starss23_featurizer_spatial_doa.py"],
        validation=("842 replaced Python LOC, 582 added, net -260; all three complete sealed artifact "
                    "dictionaries exactly match checkpoint b7fcb0b under fixed clocks; independent "
                    "verifier battery 44/44 and full focused suite 493/493 green"),
        rollback_tag="mop-collapse-starss23-lifecycle-18",
        next_action="remeasure adjacent count and DoA producer shells against the declaration target"))
    checklist.append(item(
        "RED-starss23-count-variant-artifact-lifecycle", 11, "verified_reduction",
        "Centralize concurrent-count producer artifact lifecycles", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json",
                  "src/mop/beds/starss23/count_variant_producer.py",
                  "tests/unit/test_starss23_counting_bed.py",
                  "tests/unit/test_starss23_count_repro_scoring_unit.py"],
        validation=("1379 replaced Python LOC, 972 added, net -407; five complete real-data sealed "
                    "artifact dictionaries exactly match checkpoint a7916c2 under fixed clocks; "
                    "independent producer/verifier battery 73/73 and full focused suite 493/493 green"),
        rollback_tag="mop-collapse-starss23-lifecycle-19",
        next_action="measure remaining DoA, refractory, and learning-progress producer shells"))
    checklist.append(item(
        "RED-starss23-onset-variant-and-embedded-docs", 11, "verified_reduction",
        "Centralize remaining onset variants and remove embedded STARSS narratives", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json",
                  "src/mop/beds/starss23/featurizer_variant_producer.py",
                  "src/mop/beds/starss23/real_artifact.py",
                  "src/mop/beds/starss23/refractory_nms_producer.py",
                  "src/mop/beds/starss23/learning_progress_producer.py"],
        validation=("728 lifecycle LOC replaced by 456 executor/declaration LOC; 3092 source/test "
                    "documentation LOC eliminated with 44 pass replacements; net -3320; base onset, "
                    "refractory-NMS, and learning-progress sealed artifact dictionaries exactly match "
                    "checkpoint 3b309f6 under fixed clocks; full STARSS suite 493/493 green"),
        rollback_tag="mop-collapse-starss23-lifecycle-20",
        next_action="finish the dual-architecture DoA producer shell and measure STARSS residual"))
    checklist.append(item(
        "RED-starss23-family-prereg-cli-comments", 11, "verified_reduction",
        "Centralize family preregistrations and delete local commands and comments", status="verified",
        evidence=["collapse/MOP_REDUCTION_LOG.json",
                  "src/mop/beds/starss23/prereg.py",
                  "src/mop/beds/starss23/gate_variants_prereg.py",
                  "tests/unit/test_starss23_featurizer_spatial_doa.py"],
        validation=("528 duplicate preregistration/CLI LOC replaced by 219 shared/declaration LOC; "
                    "1408 comment-only source/test LOC eliminated; eight executable wrappers removed; "
                    "four prereg seals exact; focused 52/52 and full STARSS suite 493/493 green; net -1717"),
        rollback_tag="mop-collapse-starss23-lifecycle-21",
        next_action="finish the dual-architecture DoA producer shell and measure STARSS residual"))

    # accumulate verified reductions from the append-only log
    red = {"eliminated_LOC": 0, "deduplicated_LOC": 0, "relocated_LOC": 0, "archived_LOC": 0,
           "generated_replacement_LOC": 0, "added_LOC": 0}
    for ev in (redlog.get("events") or []):
        for k in red:
            red[k] += int(ev.get(k, 0) or 0)
    red["net_global_reduction_LOC"] = (red["eliminated_LOC"] + red["deduplicated_LOC"]
                                       + red["archived_LOC"] - red["added_LOC"])

    # reconcile: record the evidence-authority deletion map and move SEC-9 into active analysis
    checklist.append(item(
        "ART-MOP_EVIDENCE_EQUIVALENCE.json", 9, "artifact",
        "MOP_EVIDENCE_EQUIVALENCE.json (evidence-primitive deletion map)", status="complete",
        evidence=["collapse/MOP_EVIDENCE_EQUIVALENCE.json"],
        validation="normalized-AST body clustering of every owned primitive definition",
        next_action="none"))
    checklist.append(item(
        "ART-MOP_EVIDENCE_MIGRATION.json", 9, "artifact",
        "MOP_EVIDENCE_MIGRATION.json (per-duplicate migration table)", status="complete",
        evidence=["collapse/MOP_EVIDENCE_MIGRATION.json"],
        validation="123 rows; batches: batch1_studies_safe/verifier-defer/controller-defer/inspect",
        next_action="execute remaining batches under their gates"))
    checklist.append(item(
        "RED-batch1", 9, "verified_reduction",
        "Evidence core batch1: 9 studies modules deduplicated onto mop.substrate.events",
        status="verified", evidence=["collapse/MOP_REDUCTION_LOG.json"],
        validation="77 LOC removed; byte-identical + py_compile + 9/9 import parity + known-answer",
        commit="", rollback_tag="mop-collapse-evidence-batch1",
        next_action="next batch: sha256_file dominant cluster (9), then _atomic_write (6), then distinct-body inspection"))
    collapsible = ((equiv.get("totals") or {}).get("redundant_definitions_collapsible"))
    for it in checklist:
        if it["id"] == "SEC-9":
            it["status"] = "active"
            it["evidence_paths"] = ["collapse/MOP_AUTHORITY_GRAPH.json",
                                    "collapse/MOP_EVIDENCE_EQUIVALENCE.json"]
            it["validation"] = (f"{collapsible} byte-identical primitive defs identified as collapsible; "
                                "distinct-body defs flagged for inspection")
            it["dependency"] = "heavy parity/mutation/replay suite must run under host headroom (live run active)"

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
            "evidence_primitive_defs_analyzed": (equiv.get("totals") or {}).get("primitive_definitions"),
            "evidence_primitive_defs_collapsible": (equiv.get("totals") or {}).get(
                "redundant_definitions_collapsible"),
            "highest_pressure_first_region": ("section 9 evidence authority: 168 duplicate integrity "
                                              "definitions collapse to one evidence core, provable by "
                                              "byte-parity + mutation tests (pure functions, live-safe)"),
            "floor_correction": ("duplicate-function analysis (14.3k byte-identical, 27k structural) measures "
                                 "similarity WITHIN the current shape, NOT the architectural-collapse ceiling. "
                                 "Measured STARSS23 anatomy: 18669 collapsible LOC (68 percent) fold into the "
                                 "shared engine. The 50k global target is not disproven."),
            "starss23_collapsible_loc": (starss.get("collapsible_loc") if starss else None),
            "starss23_preserved_loc": (starss.get("preserved_loc") if starss else None),
            "shared_engine": ("src/mop/science architecture B (269 LOC), including shared producer "
                              "receipt, finalization, and safety paths; Architecture A deleted"),
            "shared_budget_engine": ("src/mop/science/budget.py (704 LOC), three old harnesses, eight "
                                     "producer budget-point assemblers, and five seed-record copies deleted"),
            "shared_statistics_engine": ("src/mop/science/statistics.py (298 LOC), including shared "
                                          "onset and count artifact projections"),
            "selected_experiment_architecture": (architecture.get("selection") or {}).get("selected"),
            "starss23_source_decomposition": {
                "files": len(decomposition.get("files") or []),
                "ranges": sum(len(f.get("ranges") or []) for f in (decomposition.get("files") or [])),
                "category_loc": decomposition.get("category_loc") or {},
            },
        },
        "reduction_accounting_verified": red,
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
    lines.append("### STARSS23 architecture selection (current checkpoint)")
    lines.append("")
    lines.append("- candidate A: 374 engine LOC + 83 declaration LOC; 4 engine modules; 13 public symbols.")
    lines.append("- candidate B: 190 measured engine LOC + 83 declaration LOC; 1 engine module; 9 public symbols.")
    lines.append("- selected implementation: Architecture B, 198 engine LOC + 83 declaration LOC.")
    lines.append("- rejected implementation: Architecture A physically deleted after green selection.")
    lines.append("- parity: onset, counting, DoA, and data-split reproduction 4/4; selected tests 29/29.")
    lines.append("- mutation: Architecture B 22/22 attacks refused; canonical scientific authority drift refused.")
    lines.append("- net global owned Python LOC reduction: 137 (cumulative verified reduction: 339).")
    lines.append("- performance: cold import -1.1%; trivial fixture +9.793 us from fail-closed authority hashing.")
    lines.append("- rollback_tag: mop-collapse-starss23-architecture-b.")
    lines.append("- next_exact_edit: lifecycle parity and physical deletion.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 1 (current checkpoint)")
    lines.append("")
    lines.append("- source decomposition: 75 files, 30,968 physical lines, 2,364 exact top-level ranges.")
    lines.append("- shared lifecycle classification: 13,229 LOC; shared integrity classification: 1,003 LOC.")
    lines.append("- feature caches: three implementations (1,447 LOC) replaced by one 412-LOC "
                 "typed policy engine.")
    lines.append("- controls: count and DoA wrapper modules deleted; one shared never-update "
                 "authority remains.")
    lines.append("- statistics: 385-LOC family module replaced by 222-LOC producer-side shared module; "
                 "selected engine now uses the preregistered exact sign-flip test.")
    lines.append("- source reduction: 1,277 LOC; tests added net: 103 LOC; owned Python net "
                 "reduction: 1,174 LOC.")
    lines.append("- cumulative verified owned Python reduction: 1,513 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-1.")
    lines.append("- next_exact_edit: matched-budget harness consolidation.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 2 (current checkpoint)")
    lines.append("")
    lines.append("- old harnesses: onset 714 LOC + count 502 LOC + DoA 640 LOC = 1,856 LOC deleted.")
    lines.append("- replacement: one 625-LOC policy-driven budget engine; three declarations live with "
                 "the experiment records.")
    lines.append("- byte parity: representative onset, count, and dual-architecture DoA report payloads exact.")
    lines.append("- canonical payload digests are pinned in the corresponding three unit-test families.")
    lines.append("- source reduction: 1,176 LOC; tests added net: 38 LOC; owned Python net reduction: "
                 "1,138 LOC.")
    lines.append("- cumulative verified owned Python reduction: 2,651 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-2.")
    lines.append("- next_exact_edit: centralize producer budget projection and canonical artifact writes.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 3 (current checkpoint)")
    lines.append("")
    lines.append("- budget projection: eight producer-local assemblers replaced by one 39-LOC generic projection "
                 "in the shared budget engine; the diversity entry point no longer imports a removed private helper.")
    lines.append("- artifact writes: twelve producer-local final writers deleted; all entry points use one "
                 "canonical, crash-safe atomic writer in the evidence core.")
    lines.append("- parity: direct BudgetPoint structure exact; canonical file bytes reproduce through the "
                 "independent verifier; atomic replacement behavior remains green.")
    lines.append("- validation: full STARSS-focused suite 477/477 in 141.78s under nice -n 10.")
    lines.append("- source reduction: 423 LOC; tests added net: 33 LOC; scripts added net: 12 LOC; "
                 "owned Python net reduction: 378 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,029 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-3.")
    lines.append("- next_exact_edit: centralize producer results, receipts, seed records, and prereg writes.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 4 (current checkpoint)")
    lines.append("")
    lines.append("- result containers: thirteen producer-local artifact result dataclasses replaced by one "
                 "ArtifactResult in the selected science engine.")
    lines.append("- seed records: five byte-identical producer seed-run dataclasses replaced by one BudgetSeedRun.")
    lines.append("- receipts and seals: thirteen producer mint/finalize paths use one canonical evidence receipt "
                 "and one nonmutating, fail-closed artifact finalizer.")
    lines.append("- preregistration writes: twelve local writers deleted; every prereg uses the shared crash-safe "
                 "canonical JSON writer.")
    lines.append("- validation: full STARSS-focused suite 479/479 in 145.60s under nice -n 10.")
    lines.append("- source reduction: 356 LOC; tests added net: 40 LOC; owned Python net reduction: 316 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,345 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-4.")
    lines.append("- next_exact_edit: centralize producer statistics, noisy-TV controls, and safety projections.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 5 (current checkpoint)")
    lines.append("")
    lines.append("- safety: thirteen repeated three-flag blocks replaced by one fresh fail-closed projection.")
    lines.append("- controls: twelve repeated noisy-TV/control-arm blocks replaced by one policy-ordered projection.")
    lines.append("- statistics: twelve onset/count sign-flip artifact payloads now project from the one shared "
                 "decisive statistic without changing the independent verifiers.")
    lines.append("- validation: full STARSS-focused suite 483/483 in 155.47s under nice -n 10.")
    lines.append("- production source reduction: 110 LOC; tests added: 69 LOC; owned Python net reduction: 41 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,386 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-5.")
    lines.append("- next_exact_edit: collapse the common artifact envelope across all thirteen producers.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 6 (current checkpoint)")
    lines.append("")
    lines.append("- envelopes: thirteen producer-local artifact bodies now use one closed shared envelope; "
                 "producer-specific evidence remains explicit in each declaration.")
    lines.append("- budget provenance: twelve identical matched-budget payload, wall-note, and break-even blocks "
                 "are projected once; the DoA dual-budget exception remains exact.")
    lines.append("- parity: 13/13 old/new field inventories and every migrated expression are normalized-AST exact; "
                 "attempted shared-field shadowing refuses before sealing.")
    lines.append("- validation: full STARSS-focused suite 484/484 under nice -n 10.")
    lines.append("- production source reduction: 70 LOC; tests added: 39 LOC; owned Python net reduction: 31 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,417 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-6.")
    lines.append("- next_exact_edit: measure and collapse the next repeated producer execution lifecycle.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 7 (current checkpoint)")
    lines.append("")
    lines.append("- count execution: four producer-local per-seed training, validation-threshold, budget-sweep, "
                 "control, scoring, operating-point, and noisy-TV loops now execute once.")
    lines.append("- providers: sealed frame-micro, clip-macro, swapped featurizer-estimator, and alternate-gate "
                 "mathematics remain explicit callbacks; the held-fixed gate path is imported by reference.")
    lines.append("- parity: complete BudgetSeedRun digests from checkpoint 48a587d are exact for all four axes; "
                 "the sealed frame-micro digest is pinned in the default test suite.")
    lines.append("- validation: full STARSS-focused suite 485/485 in 161.37s under nice -n 10.")
    lines.append("- production source reduction: 234 LOC; tests added: 43 LOC; owned Python net reduction: 191 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,608 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-7.")
    lines.append("- next_exact_edit: collapse repeated room-disjoint split and corpus-preparation lifecycle.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 8 (current checkpoint)")
    lines.append("")
    lines.append("- split authority: synthetic and real adapters share one native dev-fold projector; onset, count, "
                 "DoA, caches, and preregistrations share one fold-3/validation/fold-4 split.")
    lines.append("- corpus preparation: seven producer-local one-time audio/provider comprehensions now map through "
                 "one stable clip-identity authority.")
    lines.append("- independence: the adversarial swapped-fold reproduction retains its distinct split mathematics.")
    lines.append("- parity: checkpoint 0936cc0 onset ClipSplit and count/DoA tuple projections are byte-identical.")
    lines.append("- validation: full STARSS-focused suite 486/486 in 149.91s under nice -n 10.")
    lines.append("- production source reduction: 106 LOC; tests added: 15 LOC; owned Python net reduction: 91 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,699 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-8.")
    lines.append("- next_exact_edit: collapse repeated causal gate passes and marginal-matched noise controls.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 9 (current checkpoint)")
    lines.append("")
    lines.append("- causal state: onset, count, alternate-gate count, and DoA share one label-free input assembly "
                 "and probability/event pass while retaining their state and gate providers.")
    lines.append("- noise controls: onset, count, swapped-featurizer count, and DoA retain independent seed "
                 "identities but share one STARSS-shaped white-noise marginal-matching implementation.")
    lines.append("- parity: nine checkpoint 3b2367e numerical hashes are exact; the count noise bytes are pinned in tests.")
    lines.append("- validation: full STARSS-focused suite 487/487 in 153.43s under nice -n 10.")
    lines.append("- production source reduction: 84 LOC; tests added net: 11 LOC; owned Python net reduction: 73 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,772 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-9.")
    lines.append("- next_exact_edit: collapse repeated fire-spread diagnostics and sealed prereg readers.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 10 (current checkpoint)")
    lines.append("")
    lines.append("- fire-spread authority: four variant producers now derive adjacency, distinct-onset counts, "
                 "and ordered per-seed summaries through the onset referee while retaining their distinct "
                 "wording and anchors.")
    lines.append("- prereg authority: three featurizer lanes and refractory NMS share one schema-and-membership "
                 "reader; their local refusal types, paths, schemas, family identities, and messages remain "
                 "explicit inputs.")
    lines.append("- parity: four checkpoint 33a7bd8 spread-artifact hashes and four checked-in prereg body "
                 "hashes are exact.")
    lines.append("- validation: full STARSS-focused suite 489/489 in 146.66s under nice -n 10.")
    lines.append("- production source reduction: 159 LOC; tests added: 51 LOC; owned Python net reduction: "
                 "108 LOC.")
    lines.append("- cumulative verified owned Python reduction: 3,880 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-10.")
    lines.append("- next_exact_edit: measure and collapse the next repeated STARSS producer/prereg "
                 "lifecycle.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 11 (current checkpoint)")
    lines.append("")
    lines.append("- analysis plan: four gate/featurizer families share one SESOI, operating-point, exact "
                 "sign-flip, claim-ceiling, and base-prereg traceability lifecycle.")
    lines.append("- declarations: each family retains its own member ids, hypotheses, multiplicity rationale, "
                 "front-end or gate contract, promotion bar, schema, and refusal surface.")
    lines.append("- parity: complete outer-body and embedded canonical hashes from checkpoint 2f3671f are exact "
                 "for all four family preregistrations.")
    lines.append("- validation: full STARSS-focused suite 490/490 in 154.46s under nice -n 10.")
    lines.append("- production source reduction: 163 LOC; tests added: 25 LOC; owned Python net reduction: "
                 "138 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,018 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-11.")
    lines.append("- next_exact_edit: measure remaining family structural-fact and prereg CLI lifecycle.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 12 (current checkpoint)")
    lines.append("")
    lines.append("- label facts: cache-backed and native-split prereg lanes share one label-only reduction while "
                 "retaining their distinct source loaders and the spatial lane's public dict shape.")
    lines.append("- reporting: family-specific Bonferroni vocabulary and precision feed one projection; four "
                 "command-line summaries share one stable report shape.")
    lines.append("- parity: four checkpoint 237e4b4 body/seal pairs and all four CLI summary hashes are exact.")
    lines.append("- validation: full STARSS-focused suite 490/490 in 154.66s under nice -n 10.")
    lines.append("- production and owned Python net reduction: 28 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,046 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-12.")
    lines.append("- next_exact_edit: collapse exact-clone frozen spectral primitives.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 13 (current checkpoint)")
    lines.append("")
    lines.append("- frozen DSP: onset, count, reproduction, coherence, spatial, SuperFlux, and DoA frontends "
                 "share one periodic Hann window; mel-based lanes share one parameterized triangular bank.")
    lines.append("- preservation: every frontend keeps its unique band geometry, temporal statistic, parameter "
                 "schema, feature layout, and independently charged FLOP ledger.")
    lines.append("- parity: window bytes, both mel-bank resolutions, and seven checkpoint d873a13 parameter "
                 "digests are exact and the byte hashes are pinned in tests.")
    lines.append("- validation: full STARSS-focused suite 491/491 in 160.35s under nice -n 10.")
    lines.append("- production source reduction: 89 LOC; tests added: 15 LOC; owned Python net reduction: 74 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,120 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-13.")
    lines.append("- next_exact_edit: collapse repeated noisy-TV namespace and frontend wrappers.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 14 (current checkpoint)")
    lines.append("")
    lines.append("- seed authority: controls and four sealed producers share one explicit domain-and-key "
                 "SHA-256 derivation while retaining every experiment's original namespace and uint32 stream.")
    lines.append("- parity: onset, count, reproduced-count, and DoA seeds are exact for roots 0, 7, and "
                 "2**32+3; rate-matched-random positions, RndTarget matrix bytes, and aleatoric bytes are exact.")
    lines.append("- validation: four producer namespace outputs are pinned at the shared boundary; full "
                 "STARSS-focused suite 492/492 in 143.67s under nice -n 10.")
    lines.append("- production source reduction: 37 LOC; tests added: 16 LOC; owned Python net reduction: "
                 "21 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,141 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-14.")
    lines.append("- next_exact_edit: measure repeated producer FLOP models and onset-density wrappers.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 15 (current checkpoint)")
    lines.append("")
    lines.append("- FLOP authority: nine producers retain local front-end, gate, downstream, and training-cost "
                 "providers but project the conventional four arm fields through one budget primitive.")
    lines.append("- provider binding: sealed and alternate count gates share one frame-micro seed binding; onset "
                 "and learning-progress lanes share one label-only density reducer.")
    lines.append("- parity: ten complete charge tables are exact across all affected frontends and both DoA "
                 "architectures; the real alternate-gate producer/verifier path is green.")
    lines.append("- validation: candidate-only training evaluation is pinned directly; full STARSS-focused suite "
                 "493/493 in 154.28s under nice -n 10.")
    lines.append("- production source reduction: 54 LOC; tests added: 20 LOC; owned Python net reduction: "
                 "34 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,175 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-15.")
    lines.append("- next_exact_edit: collapse repeated frozen-provider introspection methods.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 16 (current checkpoint)")
    lines.append("")
    lines.append("- provider authority: ten frozen estimators/frontends inherit zero-parameter introspection; "
                 "seven frontends also share feature-byte sealing and analytic frame-cost validation.")
    lines.append("- preservation: local parameter digests, per-frame constants, custom refusal classes, dataclass "
                 "fields, and slot-only layouts remain exact.")
    lines.append("- parity: ten parameter identities, seven feature digests, seven frame charges, and every "
                 "ordinary/custom error pair match checkpoint ec5257e.")
    lines.append("- validation: focused provider battery 175/175; full STARSS-focused suite 493/493 in 153.14s "
                 "under nice -n 10.")
    lines.append("- production and owned Python net reduction: 75 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,250 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-16.")
    lines.append("- next_exact_edit: measure common causal gate state, forward, and decision kernels.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 17 (current checkpoint)")
    lines.append("")
    lines.append("- count interface: sealed and alternate-topology count gates share held-fixed feature/state "
                 "assembly, batch probability, online inference, threshold decision, and report shape.")
    lines.append("- DoA interface: both gate architectures share only topology-neutral probability, online inference, "
                 "and threshold decision; their forward and optimizer mathematics remain separate.")
    lines.append("- parity: four parameter digests and all probability, decision, refusal, report payload, repr, and "
                 "dict surfaces match checkpoint b8e98dd.")
    lines.append("- validation: focused gate battery 91/91; full STARSS-focused suite 493/493 in 153.21s under "
                 "nice -n 10.")
    lines.append("- production and owned Python net reduction: 58 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,308 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-17.")
    lines.append("- next_exact_edit: remeasure remaining variant producer shells against the declaration target.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 18 (current checkpoint)")
    lines.append("")
    lines.append("- lifecycle authority: spatial DoA, SuperFlux spectral, and interchannel coherence now declare "
                 "only corpus preparation, frontend identity, provenance payloads, and variant anchors; one "
                 "shared executor owns prereg reads, seeded runs, FLOP budgets, statistics, controls, receipts, "
                 "artifact envelopes, and finalization.")
    lines.append("- preservation: all three independent verifiers remain untouched, and frontend-specific cache/raw "
                 "adapter preparation, FLOP charges, hypotheses, refusal classes, and artifact vocabulary stay local.")
    lines.append("- parity: all three complete sealed artifact dictionaries exactly match checkpoint b7fcb0b under "
                 "fixed clocks, including receipts, spread evidence, and final seals.")
    lines.append("- validation: focused independent-verifier battery 44/44; full STARSS-focused suite 493/493 in "
                 "155.83s under nice -n 10.")
    lines.append("- production and owned Python net reduction: 260 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,568 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-18.")
    lines.append("- next_exact_edit: remeasure adjacent count and DoA producer shells against the declaration target.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 19 (current checkpoint)")
    lines.append("")
    lines.append("- lifecycle authority: the base count bed plus swapped-fold, re-authored-provider, alternate-gate, "
                 "and clip-macro reproductions now share one corpus-preparation and sealed-artifact executor.")
    lines.append("- preservation: provider-specific noisy-TV namespaces, seed runners, FLOP models, splits, preregistration "
                 "builders, gate/estimator identities, clip-cluster readout, survival conjunctions, refusal classes, "
                 "and all five independent verifier implementations remain local.")
    lines.append("- parity: five complete real-data artifact dictionaries and seals exactly match checkpoint a7916c2 "
                 "under fixed clocks, including the swapped split and clip-macro evidence extensions.")
    lines.append("- validation: focused producer and independent-verifier battery 73/73 in 72.92s; full STARSS-focused "
                 "suite 493/493 in 145.36s under nice -n 10.")
    lines.append("- production and owned Python net reduction: 407 LOC.")
    lines.append("- cumulative verified owned Python reduction: 4,975 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-19.")
    lines.append("- next_exact_edit: measure remaining DoA, refractory, and learning-progress producer shells.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 20 (current checkpoint)")
    lines.append("")
    lines.append("- lifecycle authority: base onset, refractory-NMS, and learning-progress now declare their "
                 "distinct seed runner, gate, preregistration, FLOP, diagnostic, and evidence projections "
                 "against the existing onset-variant executor.")
    lines.append("- documentation deletion: 3,092 non-executable STARSS source/test docstring LOC removed; "
                 "44 one-line pass bodies retained where Python requires a class body; no code was packed.")
    lines.append("- preservation: all unique gate/training/statistical mathematics, preregistration payloads, "
                 "refusal surfaces, and independent verifier implementations remain executable and separate.")
    lines.append("- parity: base onset, refractory-NMS, and learning-progress complete artifact dictionaries and "
                 "seals exactly match checkpoint 3b309f6 under fixed clocks.")
    lines.append("- validation: full STARSS-focused suite 493/493 in 159.64s under nice -n 10; changed lifecycle "
                 "modules pass ruff and the complete STARSS package compiles.")
    lines.append("- source change: 500 added, 3,540 deleted, net -3,040; tests: 0 added, 280 deleted; "
                 "total owned Python net reduction: 3,320 LOC.")
    lines.append("- cumulative verified owned Python reduction: 8,295 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-20.")
    lines.append("- next_exact_edit: finish the dual-architecture DoA producer shell and measure STARSS residual.")
    lines.append("")
    lines.append("### STARSS23 lifecycle cluster 21 (current checkpoint)")
    lines.append("")
    lines.append("- preregistration authority: gate, coherence, spatial-DoA, and SuperFlux families now "
                 "declare their schemas, members, multiplicity vocabulary, and evidence fields against "
                 "one family preregistration assembler.")
    lines.append("- interface deletion: eight unreferenced module-local argparse/JSON command wrappers removed; "
                 "all producer, verifier, cache, and preregistration callable APIs remain.")
    lines.append("- documentation deletion: 1,408 comment-only source/test LOC removed; no executable statement, "
                 "scientific case, test case, verifier calculation, or source line was packed.")
    lines.append("- parity: the four complete family preregistration seals remain exactly "
                 "5c74b42a/7a41d355/afefd5d6/dd2cce3e for the pinned structural fixture.")
    lines.append("- validation: preregistration/verifier battery 52/52 in 2.93s; full STARSS-focused suite "
                 "493/493 in 155.78s under nice -n 10; complete package compiles and critical ruff checks pass.")
    lines.append("- source change: 219 added, 1,349 deleted, net -1,130; tests: 0 added, 587 deleted; "
                 "total owned Python net reduction: 1,717 LOC; executable entrypoints: -8.")
    lines.append("- cumulative verified owned Python reduction: 10,012 LOC.")
    lines.append("- rollback_tag: mop-collapse-starss23-lifecycle-21.")
    lines.append("- next_exact_edit: finish the dual-architecture DoA producer shell and measure STARSS residual.")
    lines.append("")
    (ROOT / "MOP_COLLAPSE_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"checklist items: {len(checklist)}")
    print(f"by status: {json.dumps(by_status)}")
    print("wrote MOP_COLLAPSE_STATE.json + MOP_COLLAPSE_LEDGER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
