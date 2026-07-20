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

import hashlib
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


def build_proof_index() -> dict:
    entries = []
    by_sha256: dict[str, list[str]] = {}
    total_bytes = 0
    for relative in sh("git", "ls-files", "proof").splitlines():
        path = ROOT / relative
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        total_bytes += len(raw)
        by_sha256.setdefault(sha256, []).append(relative)
        entries.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": sha256,
                "git_blob": sh("git", "hash-object", relative),
            }
        )
    duplicates = {digest: paths for digest, paths in by_sha256.items() if len(paths) > 1}
    index = {
        "schema": "mop-proof-index/v1",
        "files": len(entries),
        "bytes": total_bytes,
        "duplicate_groups": duplicates,
        "entries": entries,
    }
    (COLLAPSE / "MOP_PROOF_INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


ITEM_FIELDS = (
    "id",
    "section",
    "kind",
    "title",
    "status",
    "evidence_paths",
    "validation",
    "commit",
    "rollback_tag",
    "dependency",
    "next_action",
)


def item(
    id_,
    section,
    kind,
    title,
    status="pending",
    evidence=None,
    validation="",
    commit="",
    rollback_tag="",
    dependency="",
    next_action="",
):
    return {
        "id": id_,
        "section": section,
        "kind": kind,
        "title": title,
        "status": status,
        "evidence_paths": evidence or [],
        "validation": validation,
        "commit": commit,
        "rollback_tag": rollback_tag,
        "dependency": dependency,
        "next_action": next_action,
    }


def build_checklist() -> list[dict]:
    items: list[dict] = []
    A = items.append

    # ---- Section 2/24: workspace + git ----
    A(
        item(
            "WS-1",
            2,
            "workspace",
            "Isolated worktree + branch off current origin/main",
            status="complete",
            evidence=["/Users/scammermike/Downloads/mop-accretion-collapse"],
            validation="worktree HEAD == origin/main (a19ebe6); live tree untouched on f6d107b",
            next_action="none",
        )
    )
    A(
        item(
            "WS-2",
            2,
            "workspace",
            "Separate build/cache/test-temp/report/pack roots",
            status="complete",
            evidence=[
                ".collapse/build",
                ".collapse/cache",
                ".collapse/testtmp",
                ".collapse/reports",
                ".collapse/packs",
            ],
            validation="created under worktree .collapse/",
            next_action="none",
        )
    )
    A(
        item(
            "WS-3",
            2,
            "invariant",
            "Absolute non-interference with the live General Run (2.1)",
            status="active",
            validation="read-only audits only; no signal/edit/merge into live tree",
            next_action="re-verify live PID 52934 alive and untouched at every checkpoint",
        )
    )
    A(
        item(
            "GIT-1",
            24,
            "git",
            "Draft PR against current main; not ready/merge until all gates",
            status="pending",
            dependency="census+precheck commit pushed",
            next_action="push precheck commit, open draft PR via gh",
        )
    )

    # ---- Section 24: rollback tags ----
    for tag in [
        "mop-collapse-precheck",
        "mop-collapse-evidence",
        "mop-collapse-experiment-engine",
        "mop-collapse-starss23",
        "mop-collapse-mechanisms",
        "mop-collapse-controller",
        "mop-collapse-registry-config",
        "mop-collapse-validation",
        "mop-collapse-docs",
        "mop-collapse-packs",
        "mop-collapse-300k",
        "mop-collapse-250k",
        "mop-collapse-200k",
        "mop-collapse-150k",
        "mop-collapse-125k",
        "mop-collapse-100k",
        "mop-collapse-75k",
        "mop-collapse-50k",
        "mop-collapse-35k",
        "mop-collapse-event-horizon",
    ]:
        done = tag in {
            "mop-collapse-50k",
            "mop-collapse-35k",
            "mop-collapse-registry-config",
            "mop-collapse-event-horizon",
        }
        A(
            item(
                f"TAG-{tag}",
                24,
                "rollback_tag",
                f"Create rollback tag {tag}",
                status="complete" if done else "pending",
                evidence=["collapse/MOP_REDUCTION_LOG.json"] if done else [],
                next_action="none" if done else f"tag {tag} at its green checkpoint",
            )
        )

    # ---- Section 3: PR #9 disposition ----
    for sid, title, act in [
        (
            "PR9-1",
            "Inspect every PR #9 file",
            "diff origin/agent/mop-extreme-condensation vs main; list files",
        ),
        ("PR9-2", "Test PR #9 controller against current main", "run its accounting controller read-only"),
        (
            "PR9-3",
            "Port or rewrite only useful mechanisms",
            "port LOC accounting, no-minify/no-pack gates, hydration",
        ),
        ("PR9-4", "Discard assumptions invalidated by Generation-1 era", "record discarded assumptions"),
        (
            "PR9-5",
            "Replace active-checkout LOC metric with honest global reduction",
            "global accounting is primary",
        ),
        ("PR9-6", "Open a new draft PR from current main", "gh pr create --draft"),
        ("PR9-7", "Keep PR #9 open until protections exist on replacement", "do not close prematurely"),
        (
            "PR9-8",
            "Close PR #9 only after exact retained-vs-retired mapping exists",
            "write mapping artifact",
        ),
    ]:
        A(item(sid, 3, "pr9", title, next_action=act))

    # ---- Section 6: context surface artifacts + metrics ----
    A(
        item(
            "ART-CONTEXT-JSON",
            6,
            "artifact",
            "MOP_CONTEXT_SURFACE.json",
            status="complete",
            evidence=["collapse/MOP_CONTEXT_SURFACE.json"],
            validation="emitted by census.py",
            next_action="extend with cold_import/test_collection timings under host headroom",
        )
    )
    A(
        item(
            "ART-CONTEXT-MD",
            6,
            "artifact",
            "MOP_CONTEXT_SURFACE.md (orientation benchmark 10 Qs)",
            next_action="render md from json + run clean-agent orientation benchmark",
        )
    )
    A(
        item(
            "MET-CONTEXT",
            6,
            "metric",
            "Context/orientation metrics measured",
            status="partial",
            evidence=["collapse/MOP_CONTEXT_SURFACE.json"],
            validation="files/dirs/modules/public_symbols/import_edges/SCC/entrypoints measured",
            next_action=(
                "add reading tokens + cold_import + collection/docs-validation timings (heavy: queue)"
            ),
        )
    )

    # ---- Section 7: census artifacts ----
    census_arts = {
        "MOP_CODEBASE_CENSUS.json": "complete",
        "MOP_CODEBASE_CENSUS.md": "pending",
        "MOP_IMPORT_GRAPH.json": "complete",
        "MOP_CALL_GRAPH.json": "pending",
        "MOP_COMMAND_GRAPH.json": "complete",
        "MOP_SCHEMA_GRAPH.json": "pending",
        "MOP_CONFIG_GRAPH.json": "pending",
        "MOP_TEST_OWNERSHIP.json": "pending",
        "MOP_DOCUMENTATION_GRAPH.json": "pending",
        "MOP_DUPLICATION_GRAPH.json": "complete",
        "MOP_AUTHORITY_GRAPH.json": "complete",
        "MOP_HISTORICAL_BOUNDARY.json": "pending",
        "MOP_IRREDUCIBLE_KERNEL_ESTIMATE.json": "pending",
        "MOP_LIVE_NO_TOUCH.json": "complete",
    }
    for name, st in census_arts.items():
        ev = [f"collapse/{name}"] if st in ("complete", "partial") else []
        A(
            item(
                f"ART-{name}",
                7,
                "artifact",
                name,
                status=st,
                evidence=ev,
                next_action=("none" if st == "complete" else f"generate {name}"),
            )
        )
    A(
        item(
            "CENSUS-CLASSIFY",
            7,
            "census",
            "Classify every file into exactly one of 16 categories; unknown->0",
            status="pending",
            dependency="import/call/authority graphs",
            next_action="run classification over census records grounded in imports/tests/proofs/git",
        )
    )

    # ---- Section 4: global accounting metrics ----
    for met in [
        "global_owned_source_LOC",
        "global_maintained_source_LOC",
        "active_kernel_LOC",
        "active_product_LOC",
        "default_validation_LOC",
        "optional_pack_LOC",
        "laboratory_LOC",
        "compatibility_LOC",
        "historical_source_LOC",
        "generated_owned_LOC",
        "test_LOC",
        "documentation_LOC",
        "configuration_LOC",
        "CI_build_LOC",
        "fixture_LOC",
        "third_party_LOC",
    ]:
        st = (
            "partial"
            if met
            in (
                "global_owned_source_LOC",
                "global_maintained_source_LOC",
                "test_LOC",
                "documentation_LOC",
                "configuration_LOC",
            )
            else "pending"
        )
        A(
            item(
                f"MET-{met}",
                4,
                "metric",
                f"Measure {met}",
                status=st,
                evidence=["collapse/MOP_GLOBAL_ACCOUNTING.json"] if st == "partial" else [],
                next_action="derive from classified census"
                if st == "pending"
                else "refine via classification",
            )
        )
    for met in [
        "eliminated_LOC",
        "deduplicated_LOC",
        "relocated_LOC",
        "archived_LOC",
        "generated_replacement_LOC",
        "added_LOC",
        "net_global_reduction_LOC",
    ]:
        A(
            item(
                f"RED-{met}",
                4,
                "reduction_metric",
                f"Track {met} (relocation!=elimination)",
                status="pending",
                next_action="update per region collapse",
            )
        )

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
    for cp in ["300k", "250k", "200k", "150k", "125k", "100k", "75k", "50k", "35k"]:
        A(
            item(
                f"CKPT-{cp}",
                5,
                "checkpoint",
                f"Reach green global checkpoint {cp}",
                status="complete",
                evidence=["collapse/MOP_REDUCTION_LOG.json"],
                validation="tracked Python is 33513 LOC and the full retained suite is green",
                rollback_tag="mop-collapse-35k",
                next_action="none",
            )
        )
    A(
        item(
            "ESCAPE-RULE",
            5,
            "gate",
            "Two-architecture escape rule before rejecting a lower target",
            next_action=(
                "only after 2 architectures implemented+failed for measured reasons + green restore "
                "+ sealed receipt"
            ),
        )
    )

    # ---- Sections 8-20: architecture work regions ----
    for sid, sec, title, act in [
        (
            "SEC-8",
            8,
            "Canonical end-state architecture (core/science/mechanisms/substrate/campaign/packs/interface)",
            "converge domains without wrapper dirs",
        ),
        (
            "SEC-9",
            9,
            "One evidence authority (compact evidence core; verifier structurally independent)",
            "deletion map ready (collapse/MOP_EVIDENCE_EQUIVALENCE.json): 64 byte-identical primitive defs "
            "collapsible onto one core; implement core, redirect, delete, run parity+mutation+replay "
            "(HEAVY: queue behind live run per section 2)",
        ),
        (
            "SEC-10",
            10,
            "One experiment engine (ExperimentSpec..IndependentVerifier)",
            "build engine; simple<=150 LOC, complex<=400 LOC declarations",
        ),
        (
            "SEC-11",
            11,
            "STARSS23 first high-pressure region collapse (12-step process)",
            "prove method: parity byte-for-byte, replay, delete superseded, recovery map",
        ),
        (
            "SEC-12",
            12,
            "Mechanism-family collapse (one provider contract)",
            "replace *_scaffold/_impl/_bed/_runner boilerplate (152 files)",
        ),
        (
            "SEC-13",
            13,
            "One campaign controller (AFTER live run terminal + PR30 closure)",
            "build vs fixtures only while live; archive historical bytes; replay-equivalence then delete",
        ),
        (
            "SEC-14",
            14,
            "Entrypoint and script collapse (313 -> ~10 CLI verbs)",
            "classify scripts/; remove wrappers/bootstraps/argparse dup",
        ),
        (
            "SEC-15",
            15,
            "One registry (typed capability registry)",
            "unify experiment/mechanism/dataset/instrument/verifier registries",
        ),
        (
            "SEC-16",
            16,
            "One typed configuration authority",
            "separate frozen-identity/runtime-policy/machine-profile/overrides",
        ),
        (
            "SEC-17",
            17,
            "Validation condensation (properties/matrices/mutation; coverage-equivalence receipts)",
            "reduce handwritten test LOC; keep adversarial rigor + producer/verifier split",
        ),
        (
            "SEC-18",
            18,
            "Documentation collapse (<=8 front-door docs; sealed history index)",
            "consolidate 34 root md + 169 total; generate current tables from authorities",
        ),
        (
            "SEC-19",
            19,
            "Proof/evidence compaction (content-addressed index; no claim reduction)",
            "build evidence index; dedupe byte-identical payloads; move to packs after run releases",
        ),
        (
            "SEC-20",
            20,
            "Packs follow collapse (no pack owns a 2nd controller/engine/registry/CLI)",
            "collapse before packing; report relocation separate from elimination",
        ),
    ]:
        A(item(sid, sec, "region", title, next_action=act))

    # ---- Section 10 experiment LOC targets ----
    A(
        item(
            "TGT-EXP-SIMPLE",
            10,
            "target",
            "simple experiment <=150 LOC declaration + math",
            next_action="enforce",
        )
    )
    A(
        item(
            "TGT-EXP-COMPLEX",
            10,
            "target",
            "complex experiment <=400 LOC declaration + math",
            next_action="enforce",
        )
    )

    # ---- Section 9/23: scientific invariants ----
    for inv in [
        "frozen_instruments",
        "owned_substrate_separation",
        "nulls",
        "controls",
        "independent_units",
        "SESOI",
        "multiplicity",
        "stop_rules",
        "negative_results",
        "exact_evidence_classes",
        "independent_scientific_recomputation",
        "no_activation_or_promotion_without_authority",
        "honest_hardware_boundaries",
        "crash_safe_writes",
        "deterministic_resume",
        "historical_authority_replay",
        "producer_verifier_structural_independence",
    ]:
        A(
            item(
                f"INV-{inv}",
                23,
                "invariant",
                f"Preserve invariant: {inv}",
                status="active",
                validation="no LOC target may weaken this",
                next_action="assert in every region parity+mutation gate",
            )
        )

    # ---- Gates ----
    for gid, title in [
        ("GATE-NO-MINIFY", "no minification"),
        ("GATE-NO-LINE-PACK", "no line packing"),
        ("GATE-PARITY", "behavior + receipt parity"),
        ("GATE-MUTATION", "receipt/verifier mutation attacks"),
        ("GATE-REPLAY", "sealed proof replay"),
        ("GATE-CRASH-RESUME", "crash and deterministic resume"),
        ("GATE-REGEN", "deterministic regeneration"),
        ("GATE-COVERAGE-EQUIV", "coverage-equivalence receipt per replaced cluster"),
        ("GATE-CLEAN-CLONE", "clean clone builds+validates"),
        ("GATE-OFFLINE-HYDRATION", "offline pack hydration"),
        ("GATE-RELOCATION-ACCOUNTING", "relocation/archive/pack counted separately from elimination"),
        ("GATE-PERF-2PCT", "perf regressions >2% investigated"),
    ]:
        A(
            item(
                gid,
                21,
                "gate",
                f"Gate: {title}",
                status="active",
                next_action="apply at each region checkpoint; queue heavy variants until host free",
            )
        )

    # ---- Section 26: completion conditions 1-30 ----
    cc = [
        "current main and live-run identities verified",
        "PR #9 useful machinery ported or explicitly retired",
        "complete owned-system census exists",
        "unknown classifications are zero",
        "global accounting is honest",
        "one evidence authority remains",
        "one experiment engine remains",
        "one campaign controller remains active",
        "one registry remains",
        "one typed configuration authority remains",
        "one normal CLI remains",
        "STARSS23 framework duplication removed",
        "mechanism-family boilerplate removed",
        "script wrappers collapsed",
        "validation uses shared matrices and properties",
        "current-facing docs consolidated",
        "historical docs and code sealed and indexed",
        "packs contain no duplicate authorities",
        "sealed results remain replayable",
        "independent verifiers remain structurally independent",
        "crash/resume and rollback pass",
        "clean clone passes",
        "offline hydration passes",
        "no live-run source was modified",
        "full release validation passes after run releases host",
        "global LOC reduction measured",
        "orientation-token reduction measured",
        "lowest green checkpoint tagged",
        "rollback documented",
        "draft PR contains the complete measured result",
    ]
    for i, text in enumerate(cc, 1):
        st = "complete" if i == 24 else ("partial" if i in (1, 3) else "pending")
        A(
            item(
                f"CC-{i}",
                26,
                "completion_condition",
                text,
                status=st,
                next_action="evidence required per spec; nothing complete from prose",
            )
        )

    # ---- Section 27: forbidden outcomes 1-16 ----
    fo = [
        "census only",
        "plan only",
        "new abstraction beside every old abstraction",
        "pack-only reduction",
        "smaller default checkout with unchanged global owned code",
        "duplicated old and new experiment engines",
        "duplicated old and new controllers",
        "deferred documentation consolidation",
        "deletion candidates without deletion",
        "permanent wrappers around legacy implementations",
        "an under-tested generic engine",
        "hidden generated code",
        "deleted scientific evidence",
        "reduced independent-verifier rigor",
        "request to finish next region in another session",
        "claim of irreducible before two architectures attempted",
    ]
    for i, text in enumerate(fo, 1):
        A(
            item(
                f"FO-{i}",
                27,
                "forbidden_outcome",
                f"MUST NOT end with: {text}",
                status="active",
                validation="checked at final report",
                next_action="guard against; do not conclude in this state",
            )
        )

    # ---- Section 28: final report items 1-30 ----
    for i in range(1, 31):
        A(
            item(
                f"RPT-{i}",
                28,
                "report_item",
                f"Final report clause {i}",
                status="pending",
                next_action="populate from measured artifacts at conclusion",
            )
        )

    return items


def main() -> int:
    proof_index = build_proof_index()
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
            it["evidence_paths"] = [
                "collapse/MOP_STARSS23_ANATOMY.json",
                "src/mop/science/",
                "collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json",
                "collapse/MOP_STARSS23_SOURCE_DECOMPOSITION.json",
                "src/mop/beds/starss23/experiments.py",
                "src/mop/beds/starss23/feature_cache.py",
                "tests/unit/test_science_engine.py",
            ]
            it["validation"] = (
                f"measured collapsible={starss.get('collapsible_loc')} "
                f"preserved={starss.get('preserved_loc')} "
                "per_axis_declaration~21; architecture B selected after implemented A/B comparison; "
                "29/29 selected-engine tests and 22/22 B mutation attacks; cache/control/statistics "
                "cluster physically deleted with net owned Python reduction of 1174 LOC; three "
                "matched-budget harnesses deleted in favor of one policy engine, net 1138 LOC; "
                "producer budget projection and canonical writes centralized, net 378 LOC; producer "
                "result, receipt, finalization, seed-record, and prereg-write paths centralized, "
                "net 316 LOC; "
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
                "remaining comment-only STARSS documentation removed, net 1717 LOC; duplicated DoA "
                "architecture training implementations and the local seed projection collapsed onto shared "
                "dimension-driven and budget-seed lifecycles, exact numerical fingerprint and full 493-case "
                "STARSS suite preserved, net 118 executable LOC; remaining architecture-specific control, "
                "budget, statistics, gate, and detail projections collapsed onto indexed records with exact "
                "fixed-clock artifact parity, net 10 LOC"
            )
            it["dependency"] = (
                "physical deletion of *_producer/*_harness needs sealed-artifact parity; "
                "remaining producer families need sealed-artifact parity"
            )
            it["next_action"] = "measure and collapse the next residual STARSS producer family"
        if it["id"] == "SEC-10":
            it["status"] = "complete"
            it["evidence_paths"] = ["src/mop/science/", "src/mop/experiments/base.py"]
            it["validation"] = "one shared experiment lifecycle serves the two active declarations"
            it["next_action"] = "none"
        if it["id"] == "TGT-GLOBAL":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_REDUCTION_LOG.json"]
            it["validation"] = "tracked maintained Python is 16995 LOC, below the 35000 challenge"
            it["next_action"] = "prevent regrowth"
        if it["id"] == "TGT-KERNEL":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_REDUCTION_LOG.json"]
            it["validation"] = "the complete src/mop tree is 10890 LOC, below the 18000 stretch target"
            it["next_action"] = "prevent regrowth"
        if it["id"] == "TGT-TESTS":
            it["status"] = "complete"
            it["evidence_paths"] = ["tests/"]
            it["validation"] = "retained test harness is 2970 LOC"
            it["next_action"] = "prevent regrowth"
        if it["id"] in {"TGT-REGISTRY", "SEC-15", "CC-9"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["registry/experiments.yaml"]
            it["validation"] = "registry/experiments.yaml is the sole maintained registry"
            it["next_action"] = "prevent parallel registries"
        if it["id"] in {"TGT-CONFIG", "SEC-16", "CC-10"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["configs/config.yaml", "src/mop/config.py"]
            it["validation"] = "one configs/ tree is composed through the single mop.config authority"
            it["next_action"] = "prevent alternate config loaders"
        if it["id"] in {"TGT-ENTRYPOINTS", "TGT-CLI"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["pyproject.toml", "scripts/"]
            it["validation"] = "one installed CLI plus eight bounded developer entrypoints remain"
            it["next_action"] = "prevent wrapper regrowth"
        if it["id"] in {"TGT-EVIDENCE", "SEC-9", "CC-6"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["src/mop/evidence.py", "src/mop/beds/starss23/count_verifier.py"]
            it["validation"] = (
                "mop.evidence is the sole production serializer and hasher; "
                "the STARSS verifier keeps its required independent implementation"
            )
            it["next_action"] = "prevent local evidence writers"
        if it["id"] in {"TGT-CONTROLLER", "CC-8"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["src/mop/harness/runner.py"]
            it["validation"] = "mop.harness.runner is the sole installed experiment controller"
            it["next_action"] = "prevent parallel controllers"
        if it["id"] == "TGT-EXPERIMENT":
            it["status"] = "complete"
            it["evidence_paths"] = ["src/mop/science/", "src/mop/experiments/base.py"]
            it["validation"] = "one experiment execution framework remains"
            it["next_action"] = "none"
        if it["id"] == "CC-7":
            it["status"] = "complete"
            it["evidence_paths"] = ["src/mop/harness/runner.py", "src/mop/experiments/"]
            it["validation"] = "one installed experiment lifecycle serves the two active declarations"
            it["next_action"] = "none"
        if it["id"] in {"SEC-17", "CC-15"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["tests/", "scripts/acceptance.py"]
            it["validation"] = "162 retained tests and all ten acceptance checks pass in 2970 test LOC"
            it["next_action"] = "prevent duplicated fixture-specific suites"
        if it["id"] == "TGT-DOCS":
            it["status"] = "complete"
            it["evidence_paths"] = [
                "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json",
            ]
            it["validation"] = "exactly four current Markdown authorities totaling under 8000 LOC"
            it["next_action"] = "documentation is intentionally non-gating during experimentation"
        if it["id"] == "SEC-18":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json"]
            it["validation"] = (
                "170 original document versions sealed by SHA-256 and Git blob; "
                "162 superseded files deleted; four current authorities under 800 LOC; "
                "all runtime-unused embedded Python documentation and comment-only prose "
                "removed without statement packing; scanner reports zero eligible lines"
            )
            it["next_action"] = "none; enforce anti-regrowth gate"
        if it["id"] == "SEC-13":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_CODE_INDEX.json"]
            it["validation"] = (
                "historical campaign controllers are tag-recoverable and no parallel controller remains"
            )
            it["next_action"] = "none"
        if it["id"] == "SEC-12":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_CODE_INDEX.json"]
            it["validation"] = (
                "all completed mechanism scaffold, bed, implementation, and runner families are retired"
            )
            it["next_action"] = "none"
        if it["id"] == "CC-16":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json"]
            it["validation"] = "current-facing documentation consolidated to four files and 91 lines"
            it["next_action"] = "none"
        if it["id"] == "CC-17":
            it["status"] = "partial"
            it["evidence_paths"] = [
                "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
            ]
            it["validation"] = (
                "historical documentation and the superseded Generation-1 controller branch "
                "sealed and indexed; remaining historical code boundaries remain"
            )
            it["next_action"] = "index each subsequent physical historical-code deletion"
        if it["id"] == "CC-13":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_CODE_INDEX.json"]
            it["validation"] = "mechanism-family boilerplate was physically retired at the 50k checkpoint"
            it["next_action"] = "none"
        if it["id"] == "SEC-14":
            it["status"] = "complete"
            it["evidence_paths"] = ["scripts/studio/__main__.py", "collapse/MOP_HISTORICAL_CODE_INDEX.json"]
            it["validation"] = (
                "parallel Studio command wrappers and completed campaign entrypoints are retired; "
                "the host operations parser retains only doctor and profiles"
            )
            it["next_action"] = "none"
        if it["id"] == "CC-14":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_HISTORICAL_CODE_INDEX.json"]
            it["validation"] = "forty-one one-off script wrappers were retired at the 35k checkpoint"
            it["next_action"] = "none"
        if it["id"] == "CC-26":
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_REDUCTION_LOG.json"]
            it["validation"] = "global maintained Python is measured at every checkpoint"
            it["next_action"] = "none"
        if it["id"] in {"CC-28", "CC-29"}:
            it["status"] = "complete"
            it["evidence_paths"] = ["collapse/MOP_REDUCTION_LOG.json"]
            it["validation"] = "the event-horizon checkpoint is tagged and every deletion has tag recovery"
            it["next_action"] = "none"

    checklist.append(
        item(
            "ART-MOP_STARSS23_ARCHITECTURE_COMPARISON.json",
            11,
            "artifact",
            "MOP_STARSS23_ARCHITECTURE_COMPARISON.json (implemented A/B selection)",
            status="complete",
            evidence=["collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json"],
            validation="four axes in both designs; LOC/API/dependency/import/runtime/audit/mutation measured",
            next_action="none",
        )
    )
    checklist.append(
        item(
            "ART-MOP_STARSS23_SOURCE_DECOMPOSITION.json",
            11,
            "artifact",
            "MOP_STARSS23_SOURCE_DECOMPOSITION.json (complete line-range ownership map)",
            status="complete",
            evidence=[
                "collapse/MOP_STARSS23_SOURCE_DECOMPOSITION.json",
            ],
            validation=(
                f"{len(decomposition.get('files') or [])} files and "
                f"{sum(len(f.get('ranges') or []) for f in (decomposition.get('files') or []))} "
                "top-level ranges partition every physical line exactly once"
            ),
            next_action="use named parity and rollback fields as the deletion gate for each cluster",
        )
    )
    checklist.append(
        item(
            "RED-starss23-architecture-b",
            10,
            "verified_reduction",
            "Select Architecture B and physically delete Architecture A",
            status="verified",
            evidence=[
                "collapse/MOP_STARSS23_ARCHITECTURE_COMPARISON.json",
                "collapse/MOP_REDUCTION_LOG.json",
            ],
            validation="539 replaced Python LOC, 402 added, net -137; 29/29 focused green",
            rollback_tag="mop-collapse-starss23-architecture-b",
            next_action="wire real STARSS23 providers and delete superseded family lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-starss23-cache-controls-statistics",
            11,
            "verified_reduction",
            "Collapse STARSS23 cache, control, and producer-statistics lifecycle",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/feature_cache.py",
                "src/mop/science/statistics.py",
                "tests/unit/test_starss23_feature_cache.py",
            ],
            validation=(
                "1816 replaced Python LOC, 642 added, net -1174; historical cache identities, "
                "crash-safe writes, exact statistics, controls, and full focused STARSS suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-1",
            next_action="collapse the three STARSS23 budget harnesses onto one shared implementation",
        )
    )
    checklist.append(
        item(
            "RED-starss23-matched-budget-harnesses",
            11,
            "verified_reduction",
            "Replace three STARSS23 matched-budget harnesses with one policy engine",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/budget.py",
                "src/mop/beds/starss23/experiments.py",
                "tests/unit/test_starss23_harness.py",
                "tests/unit/test_starss23_counting_bed.py",
                "tests/unit/test_starss23_doa_bed.py",
            ],
            validation=(
                "2142 replaced Python LOC, 1004 added, net -1138; onset/count/DoA payloads "
                "byte-equal and old canonical digests pinned; full focused STARSS suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-2",
            next_action="centralize producer budget projection and crash-safe canonical artifact writes",
        )
    )
    checklist.append(
        item(
            "RED-starss23-producer-projection-writes",
            11,
            "verified_reduction",
            "Centralize STARSS23 producer budget projection and canonical artifact writes",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/budget.py",
                "src/mop/substrate/events.py",
                "tests/unit/test_starss23_harness.py",
                "tests/unit/test_starss23_end_to_end.py",
            ],
            validation=(
                "618 replaced Python LOC, 240 added, net -378; exact BudgetPoint projection, "
                "canonical byte sealing, crash-safe replacement, and full focused STARSS suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-3",
            next_action="centralize producer results, receipts, seed records, and preregistration writes",
        )
    )
    checklist.append(
        item(
            "RED-starss23-producer-results-receipts",
            11,
            "verified_reduction",
            "Centralize STARSS23 producer results, receipts, seed records, and preregistration writes",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/__init__.py",
                "src/mop/science/budget.py",
                "src/mop/substrate/events.py",
                "tests/unit/test_science_engine.py",
                "tests/unit/test_starss23_end_to_end.py",
            ],
            validation=(
                "586 replaced Python LOC, 270 added, net -316; canonical nonmutating finalization, "
                "exact evidence-digest receipts, crash-safe prereg writes, and focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-4",
            next_action="centralize producer statistics, noisy-TV controls, and safety projections",
        )
    )
    checklist.append(
        item(
            "RED-starss23-producer-projections",
            11,
            "verified_reduction",
            "Centralize STARSS23 producer statistics, controls, and safety projections",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/__init__.py",
                "src/mop/science/budget.py",
                "src/mop/science/statistics.py",
                "tests/unit/test_starss23_stats.py",
                "tests/unit/test_starss23_harness.py",
            ],
            validation=(
                "393 replaced Python LOC, 352 added, net -41; production source net -110; exact "
                "statistics/control shapes, fresh closed safety flags, and focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-5",
            next_action="collapse the common artifact envelope across all thirteen producers",
        )
    )
    checklist.append(
        item(
            "RED-starss23-artifact-envelopes",
            11,
            "verified_reduction",
            "Centralize STARSS23 producer artifact envelopes and matched-budget provenance",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/__init__.py",
                "tests/unit/test_science_engine.py",
                "tests/unit/test_starss23_end_to_end.py",
            ],
            validation=(
                "649 replaced Python LOC, 618 added, net -31; production source net -70; 13/13 "
                "field inventories and migrated expressions exact; closed-authority mutation refused; "
                "full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-6",
            next_action="measure and collapse the next repeated producer execution lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-starss23-count-seed-lifecycle",
            11,
            "verified_reduction",
            "Centralize STARSS23 counting per-seed execution lifecycle",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/count_producer.py",
                "tests/unit/test_starss23_counting_bed.py",
            ],
            validation=(
                "372 replaced Python LOC, 181 added, net -191; production source net -234; complete "
                "legacy BudgetSeedRun digests exact across micro, clip-macro, swapped-provider, and "
                "alternate-gate axes; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-7",
            next_action="collapse repeated room-disjoint split and corpus-preparation lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-starss23-native-split-corpus-map",
            11,
            "verified_reduction",
            "Centralize STARSS23 native split and one-time corpus provider mapping",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/adapter.py",
                "tests/unit/test_starss23_adapter.py",
            ],
            validation=(
                "245 replaced Python LOC, 154 added, net -91; production source net -106; onset, "
                "count, and DoA pre/post split projections byte-identical; swapped-fold reproduction "
                "remains independent; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-8",
            next_action="collapse repeated causal gate passes and marginal-matched noise controls",
        )
    )
    checklist.append(
        item(
            "RED-starss23-causal-gate-noise",
            11,
            "verified_reduction",
            "Centralize STARSS23 causal gate and marginal-noise lifecycle",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/gating.py",
                "src/mop/beds/starss23/adapter.py",
                "tests/unit/test_starss23_counting_bed.py",
            ],
            validation=(
                "226 replaced Python LOC, 153 added, net -73; production source net -84; nine "
                "pre/post numerical hashes exact across onset/count/DoA state inputs, traces, events, "
                "and seeded noise; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-9",
            next_action="collapse repeated fire-spread diagnostics and sealed prereg readers",
        )
    )
    checklist.append(
        item(
            "RED-starss23-fire-spread-prereg-read",
            11,
            "verified_reduction",
            "Centralize STARSS23 producer fire-spread and sealed prereg reads",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/referee.py",
                "src/mop/science/__init__.py",
                "tests/unit/test_starss23_referee.py",
                "tests/unit/test_science_engine.py",
            ],
            validation=(
                "305 replaced Python LOC, 197 added, net -108; production source net -159; four "
                "spread-artifact and four checked-in prereg body hashes exact; local refusal surfaces "
                "preserved; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-10",
            next_action="measure and collapse the next repeated STARSS producer/prereg lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-starss23-family-prereg-plan",
            11,
            "verified_reduction",
            "Centralize STARSS23 family preregistration analysis plans",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/prereg.py",
                "tests/unit/test_starss23_featurizer_spatial_doa.py",
            ],
            validation=(
                "338 replaced Python LOC, 200 added, net -138; production source net -163; complete "
                "outer-body and embedded seals exact across all four family preregistrations; local "
                "multiplicity rationales and hypotheses preserved; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-11",
            next_action="measure remaining family structural-fact and prereg CLI lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-starss23-family-prereg-facts-cli",
            11,
            "verified_reduction",
            "Centralize STARSS23 family prereg facts, multiplicity, and CLI projections",
            status="verified",
            evidence=["collapse/MOP_REDUCTION_LOG.json", "src/mop/beds/starss23/prereg.py"],
            validation=(
                "170 replaced Python LOC, 142 added, net -28; four complete body/seal pairs and four "
                "CLI summary hashes exact; hand-computable split facts exact; local data sources and "
                "family vocabulary preserved; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-12",
            next_action="collapse exact-clone frozen spectral primitives",
        )
    )
    checklist.append(
        item(
            "RED-starss23-frozen-spectral-primitives",
            11,
            "verified_reduction",
            "Centralize STARSS23 frozen spectral primitives",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/featurizer.py",
                "tests/unit/test_starss23_featurizer.py",
            ],
            validation=(
                "121 replaced Python LOC, 47 added, net -74; production source net -89; periodic-Hann, "
                "64-mel and 32-mel bytes exact; seven complete frontend parameter digests exact; full "
                "focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-13",
            next_action="collapse repeated noisy-TV namespace and frontend wrappers",
        )
    )
    checklist.append(
        item(
            "RED-starss23-domain-seeds",
            11,
            "verified_reduction",
            "Centralize STARSS23 domain-separated seed derivation",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/adapter.py",
                "tests/unit/test_starss23_adapter.py",
            ],
            validation=(
                "82 replaced Python LOC, 61 added, net -21; production source net -37; exact "
                "uint32 producer seeds across ordinary and oversized roots, rate-matched-random "
                "positions, RndTarget matrix bytes, and aleatoric-control bytes preserved; full "
                "focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-14",
            next_action="measure remaining repeated producer FLOP models and onset-density wrappers",
        )
    )
    checklist.append(
        item(
            "RED-starss23-flop-provider-projection",
            11,
            "verified_reduction",
            "Centralize STARSS23 arm FLOP projection and count provider binding",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/science/budget.py",
                "src/mop/beds/starss23/count_producer.py",
                "tests/unit/test_starss23_harness.py",
            ],
            validation=(
                "163 replaced Python LOC, 129 added, net -34; production source net -54; ten complete "
                "arm-charge tables exact, alternate count gate shares one binding by identity, real "
                "producer/verifier path green, and independent verifier formulas remain local"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-15",
            next_action="collapse repeated frozen-provider introspection methods",
        )
    )
    checklist.append(
        item(
            "RED-starss23-frozen-provider-introspection",
            11,
            "verified_reduction",
            "Centralize STARSS23 frozen-provider introspection",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/adapter.py",
                "tests/unit/test_starss23_featurizer.py",
                "tests/unit/test_starss23_counting_bed.py",
            ],
            validation=(
                "130 replaced Python LOC, 55 added, net -75; ten dataclass and slot layouts, ten "
                "parameter digests, seven feature digests, seven frame costs, and all refusal types "
                "and messages exact; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-16",
            next_action="measure common causal gate state, forward, and decision kernels",
        )
    )
    checklist.append(
        item(
            "RED-starss23-topology-neutral-gate-interfaces",
            11,
            "verified_reduction",
            "Centralize STARSS23 topology-neutral gate interfaces",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/count_gate.py",
                "src/mop/beds/starss23/doa_gate.py",
                "tests/unit/test_starss23_doa_bed.py",
            ],
            validation=(
                "126 replaced Python LOC, 68 added, net -58; four parameter digests, probability "
                "vectors, online decisions, refusal surfaces, and count report object shapes exact; "
                "forward topologies and optimizers remain local; full focused suite green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-17",
            next_action="remeasure remaining variant producer shells against the declaration target",
        )
    )
    checklist.append(
        item(
            "RED-starss23-frozen-featurizer-variant-lifecycle",
            11,
            "verified_reduction",
            "Centralize frozen-featurizer producer execution and spread projections",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/featurizer_variant_producer.py",
                "tests/unit/test_starss23_featurizer_spatial_doa.py",
            ],
            validation=(
                "842 replaced Python LOC, 582 added, net -260; all three complete sealed artifact "
                "dictionaries exactly match checkpoint b7fcb0b under fixed clocks; independent "
                "verifier battery 44/44 and full focused suite 493/493 green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-18",
            next_action="remeasure adjacent count and DoA producer shells against the declaration target",
        )
    )
    checklist.append(
        item(
            "RED-starss23-count-variant-artifact-lifecycle",
            11,
            "verified_reduction",
            "Centralize concurrent-count producer artifact lifecycles",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/count_variant_producer.py",
                "tests/unit/test_starss23_counting_bed.py",
                "tests/unit/test_starss23_count_repro_scoring_unit.py",
            ],
            validation=(
                "1379 replaced Python LOC, 972 added, net -407; five complete real-data sealed "
                "artifact dictionaries exactly match checkpoint a7916c2 under fixed clocks; "
                "independent producer/verifier battery 73/73 and full focused suite 493/493 green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-19",
            next_action="measure remaining DoA, refractory, and learning-progress producer shells",
        )
    )
    checklist.append(
        item(
            "RED-starss23-onset-variant-and-embedded-docs",
            11,
            "verified_reduction",
            "Centralize remaining onset variants and remove embedded STARSS narratives",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/featurizer_variant_producer.py",
                "src/mop/beds/starss23/real_artifact.py",
                "src/mop/beds/starss23/refractory_nms_producer.py",
                "src/mop/beds/starss23/learning_progress_producer.py",
            ],
            validation=(
                "728 lifecycle LOC replaced by 456 executor/declaration LOC; 3092 source/test "
                "documentation LOC eliminated with 44 pass replacements; net -3320; base onset, "
                "refractory-NMS, and learning-progress sealed artifact dictionaries exactly match "
                "checkpoint 3b309f6 under fixed clocks; full STARSS suite 493/493 green"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-20",
            next_action="finish the dual-architecture DoA producer shell and measure STARSS residual",
        )
    )
    checklist.append(
        item(
            "RED-starss23-family-prereg-cli-comments",
            11,
            "verified_reduction",
            "Centralize family preregistrations and delete local commands and comments",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/prereg.py",
                "src/mop/beds/starss23/gate_variants_prereg.py",
                "tests/unit/test_starss23_featurizer_spatial_doa.py",
            ],
            validation=(
                "528 duplicate preregistration/CLI LOC replaced by 219 shared/declaration LOC; "
                "1408 comment-only source/test LOC eliminated; eight executable wrappers removed; "
                "four prereg seals exact; focused 52/52 and full STARSS suite 493/493 green; net -1717"
            ),
            rollback_tag="mop-collapse-starss23-lifecycle-21",
            next_action="finish the dual-architecture DoA producer shell and measure STARSS residual",
        )
    )
    checklist.append(
        item(
            "RED-current-documentation-authority",
            18,
            "verified_reduction",
            "Collapse current documentation to eight recoverable authorities",
            status="verified",
            evidence=[
                "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json",
            ],
            validation=(
                "170 source versions sealed with SHA-256 and Git blobs; 162 files physically "
                "deleted; Markdown 95 added/43508 deleted; docs gate reports 8 files under 800 LOC; "
                "documentation tests 32/32 green; owned Python net -187"
            ),
            rollback_tag="mop-collapse-docs",
            next_action="finish the dual-architecture DoA producer shell",
        )
    )
    checklist.append(
        item(
            "RED-embedded-python-documentation",
            18,
            "verified_reduction",
            "Remove runtime-unused embedded Python documentation",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "tests/unit/test_integrity_scaffold.py",
                "tests/unit/test_sensing_scaffold.py",
            ],
            validation=(
                "54 required pass bodies added, 11294 docstring/comment/cleanup LOC deleted, "
                "net -11240; module docs consumed by CLIs, shebangs, legal headers, directives, and "
                "all executable statements retained; compile-all, critical ruff, 774 passed/2 skipped"
            ),
            rollback_tag="mop-collapse-embedded-docs",
            next_action="finish the dual-architecture DoA producer shell",
        )
    )
    checklist.append(
        item(
            "RED-doa-gate-complete-python-surface",
            11,
            "verified_reduction",
            "Centralize DoA gates and seed records; finish embedded Python documentation removal",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/beds/starss23/doa_gate.py",
                "src/mop/beds/starss23/doa_producer.py",
                "tests/unit/test_starss23_doa_bed.py",
            ],
            validation=(
                "297 DoA lifecycle LOC replaced by 179 shared/declaration LOC with exact numerical "
                "fingerprint; 9339 remaining documentation/whitespace/unused-import LOC eliminated; "
                "total 253 added/9636 deleted, net -9383; scanner zero; compile-all and critical ruff "
                "green; targeted DoA/integrity/sensing/docs battery green in 80.36s; STARSS 493/493 "
                "green in 166.99s"
            ),
            rollback_tag="mop-collapse-python-surface",
            next_action=("collapse remaining DoA statistics/control projections and measure STARSS residual"),
        )
    )
    checklist.append(
        item(
            "RED-doa-projection-controller-predecessors",
            13,
            "verified_reduction",
            "Collapse DoA projections and delete superseded Generation-1 controller branches",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studio/generation1_successor_chain_v7.py",
                "tests/unit/test_generation1_successor_chain_v7.py",
            ],
            validation=(
                "145 replaced Python LOC plus 24435 superseded Python LOC removed for 134 replacement "
                "LOC, net -24446; exact fixed-clock DoA artifact parity; active controller/DoA battery "
                "109/109 green with worktree-first PYTHONPATH; compile-all, ruff, and docs gates green"
            ),
            rollback_tag="mop-collapse-controller-predecessors",
            next_action="collapse surviving base/extension/recovery controller modules",
        )
    )
    checklist.append(
        item(
            "RED-joint-axis-construction-search",
            12,
            "verified_reduction",
            "Centralize joint-axis runners and select the construction-search vector engine",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/mechanisms/joint_axis_runner.py",
                "src/mop/mechanisms/construction_search_runner.py",
                "src/mop/mechanisms/construction_search_vec_impl.py",
            ],
            validation=(
                "760 duplicate and 855 superseded Python LOC removed for 314 replacement LOC, net "
                "-1301; exact four-mechanism and construction-search fingerprints; 123 focused cases "
                "green; compile and ruff clean"
            ),
            rollback_tag="mop-collapse-mechanism-runners",
            next_action="collapse matching joint-axis scaffold and bed lifecycle",
        )
    )
    checklist.append(
        item(
            "RED-joint-axis-scaffolds",
            12,
            "verified_reduction",
            "Centralize joint-axis mechanism scaffold lifecycles",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "src/mop/mechanisms/joint_axis_runner.py",
                "src/mop/mechanisms/calibrated_uncertainty_scaffold.py",
                "src/mop/mechanisms/reducible_novelty_scaffold.py",
                "src/mop/mechanisms/stability_plasticity_r2_scaffold.py",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
            ],
            validation=(
                "1472 duplicate Python LOC and 107 prose-only Python LOC removed for 685 replacement "
                "LOC, net -894; exact four-mechanism scaffold fingerprint; 115 focused cases green; "
                "3829-case collection, compile-all, ruff, and docs gates green"
            ),
            rollback_tag="mop-collapse-mechanism-scaffolds",
            next_action="collapse matching joint-axis bed and remaining mechanism lifecycles",
        )
    )
    checklist.append(
        item(
            "RED-retired-stability-plasticity-v1",
            12,
            "verified_reduction",
            "Delete retired stability-plasticity v1 and executable coverage prose",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/mechanisms/stability_plasticity_r2_bed.py",
                "src/mop/mechanisms/stability_plasticity_r2_impl.py",
                "src/mop/mechanisms/stability_plasticity_r2_runner.py",
            ],
            validation=(
                "1765 superseded or prose-only Python LOC removed for 34 active-registry/test LOC, "
                "net -1731; six v1 files recover from the prior tag and the complete predecessor "
                "fingerprint is sealed; 405 focused cases green; 3779-case collection, compile-all, "
                "ruff, and docs gates green"
            ),
            rollback_tag="mop-collapse-retired-mechanism-v1",
            next_action="collapse remaining active mechanism impl/bed/runner lifecycles",
        )
    )
    checklist.append(
        item(
            "RED-cli-bed-stage3-authorities",
            12,
            "verified_reduction",
            "Centralize successor beds, Studio commands, and Stage-3 registry authority",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/mechanisms/joint_axis_runner.py",
                "scripts/studio/__main__.py",
                "src/mop/ladder/stage3_registry.py",
            ],
            validation=(
                "219 duplicate and 618 superseded Python LOC removed for 166 shared/migration LOC, "
                "net -671; exact three-mechanism lifecycle fingerprint; unified Studio parser retains "
                "13 commands; 209 focused cases green; 3756-case collection, compile-all, ruff, and "
                "docs gates green"
            ),
            rollback_tag="mop-collapse-cli-bed-authorities",
            next_action="collapse remaining active mechanism and command lifecycles",
        )
    )
    checklist.append(
        item(
            "RED-starss23-orphaned-count-reproductions",
            11,
            "verified_reduction",
            "Delete orphaned STARSS23 count reproduction variants",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/beds/starss23/count_producer.py",
                "src/mop/beds/starss23/count_verifier.py",
                "tests/unit/test_starss23_counting_bed.py",
            ],
            validation=(
                "4691 orphaned source LOC and 870 dedicated-test LOC physically deleted, net -5561; "
                "repository-wide reachability found no executable external consumer; retained "
                "STARSS23 battery 163/163 green; 3709-case collection, compile-all, critical ruff, "
                "and docs gate clean; all deleted paths recover from the prior tag"
            ),
            rollback_tag="mop-collapse-starss23-orphans",
            next_action="delete or centralize the next test-only experimental authority cluster",
        )
    )
    checklist.append(
        item(
            "RED-starss23-dead-scaffolds",
            11,
            "verified_reduction",
            "Delete broken and duplicate STARSS23 scaffolds",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/beds/starss23/fixtures.py",
                "src/mop/beds/starss23/artifact.py",
                "src/mop/beds/starss23/experiments.py",
            ],
            validation=(
                "1144 broken, duplicate, test-only, or phantom-record Python LOC removed for 3 "
                "active-record/test LOC, net -1141; retained STARSS23 battery 178/178 green; "
                "3672-case collection, compile-all, critical ruff, and docs gate clean; deleted "
                "files recover from the prior tag"
            ),
            rollback_tag="mop-collapse-starss23-dead-scaffolds",
            next_action="continue the dead-authority audit outside the active controller chain",
        )
    )
    checklist.append(
        item(
            "RED-starss23-null-explorations",
            11,
            "verified_reduction",
            "Delete null additional STARSS23 gate variants",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/beds/starss23/gate.py",
                "src/mop/beds/starss23/gate_variants_prereg.py",
                "src/mop/beds/starss23/artifact.py",
            ],
            validation=(
                "2456 exploratory Python LOC and 1.36 MB of null output artifacts physically "
                "deleted; both variants were outside the sealed family and had no campaign/config "
                "consumer; retained 71-case core battery, 3630-case collection, compile-all, "
                "critical ruff, and docs gate clean; all paths recover from the prior tag"
            ),
            rollback_tag="mop-collapse-starss23-null-explorations",
            next_action="delete the next sealed-null or unreachable experimental vertical slice",
        )
    )
    checklist.append(
        item(
            "RED-starss23-null-family",
            11,
            "verified_reduction",
            "Delete null STARSS23 variant family and DoA axis",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/beds/starss23/artifact.py",
                "src/mop/beds/starss23/count_producer.py",
                "src/mop/beds/starss23/count_verifier.py",
            ],
            validation=(
                "8738 null, unreachable, dedicated-test, or orphaned-output Python LOC removed for "
                "2 active-record/test LOC, net -8736; 6.37 MiB of proof output removed; retained "
                "base/count battery 161/161 green; 3520-case collection, compile-all, critical "
                "ruff, and docs gate clean; all 62 deleted paths recover from the prior tag"
            ),
            rollback_tag="mop-collapse-starss23-null-family",
            next_action="audit whether the null base onset bed should remain executable or historical only",
        )
    )
    checklist.append(
        item(
            "RED-starss23-count-kernel",
            11,
            "verified_reduction",
            "Retire null base onset bed and isolate counting kernel",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/beds/starss23/count_featurizer.py",
                "src/mop/beds/starss23/count_gate.py",
                "src/mop/beds/starss23/count_producer.py",
            ],
            validation=(
                "5275 null/base-only Python LOC removed for 77 count-kernel LOC, net -5198; "
                "spectral arrays and gate accounting are exact before/after; retained 121-case "
                "count boundary battery, 3384-case collection, compile-all, critical ruff, and "
                "docs gate clean; all 25 deleted paths recover from the prior tag"
            ),
            rollback_tag="mop-collapse-starss23-count-kernel",
            next_action="audit remaining STARSS23 package layers around the mechanics-ok counting kernel",
        )
    )
    checklist.append(
        item(
            "RED-controller-extension-v4-recovery-v2",
            13,
            "verified_reduction",
            "Delete superseded extension-v4 and recovery-v2 controller branch",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studio/general_run_orchestrator.py",
                "src/mop/studio/generation1_successor_chain_v7.py",
                "src/mop/studio/generation1_categorized_batch_extension_chain.py",
            ],
            validation=(
                "3083 unreachable controller/wrapper/test LOC physically deleted after live terminal "
                "closure; retained General Run path 59/59 green; 3366-case collection, compile-all, "
                "critical ruff, and docs gate clean; all six paths recover; protected checkout "
                "unchanged"
            ),
            rollback_tag="mop-collapse-controller-extension-v4",
            next_action="collapse duplication within the surviving General Run controller path",
        )
    )
    checklist.append(
        item(
            "RED-controller-general-run-orchestrator",
            13,
            "verified_reduction",
            "Delete isolated General Run and legacy v3/v7 orchestration layer",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studio/generation1_categorized_batch_extension_chain.py",
                "src/mop/studio/generation1_full_generations_extension_chain.py",
            ],
            validation=(
                "7046 unreachable coordinator/wrapper/test LOC physically deleted after terminal "
                "closure; retained direct stage controllers 25/25 green; 3306-case collection, "
                "compile-all, critical ruff, and docs gate clean; all nine paths recover; "
                "protected checkout unchanged"
            ),
            rollback_tag="mop-collapse-controller-orchestrator",
            next_action="collapse duplication between the two surviving direct stage controllers",
        )
    )
    checklist.append(
        item(
            "RED-full-generations-future-phase",
            8,
            "verified_reduction",
            "Delete the never-started full-generations future phase",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studies/generation1_successor_categorized_batch_wave.py",
                "src/mop/studio/generation1_categorized_batch_extension_chain.py",
                "src/mop/studio/generation1_subaccomplishment_emitter.py",
            ],
            validation=(
                "8036 future-phase Python LOC removed for 18 generic-observer/test LOC, net -8018; "
                "the dedicated 1678-line manifest/policy surface was also deleted; no run or proof "
                "artifact ever existed; retained boundary 67/67 green; 3267-case collection, "
                "compile-all, critical ruff, and docs gate clean; all 13 paths recover"
            ),
            rollback_tag="mop-collapse-full-generations",
            next_action="audit the categorized wave for completed or unexecuted experimental slices",
        )
    )
    checklist.append(
        item(
            "RED-categorized-wave-execution-framework",
            8,
            "verified_reduction",
            "Retire the stopped categorized-wave execution framework",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studies/generation1_successor_mechanics_queue.py",
                "src/mop/studio/generation1_supervisor.py",
                "src/mop/studio/generation1_subaccomplishment_emitter.py",
            ],
            validation=(
                "8153 stopped phase and observer Python LOC removed for 12 survivor LOC, net -8141; "
                "the protected sealed result is complete with promotion disabled and 57/59 capsules "
                "complete; its 68 MB evidence tree remains untouched; retained boundary 64/64 green; "
                "3225-case collection, compile-all, critical ruff, and docs gate clean; all 12 paths "
                "recover"
            ),
            rollback_tag="mop-collapse-categorized-wave",
            next_action="audit the remaining Generation-1 successor mechanics and final-campaign surfaces",
        )
    )
    checklist.append(
        item(
            "RED-successor-horizon-v1-v2",
            8,
            "verified_reduction",
            "Retire completed successor horizon v1/v2 campaign stacks",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studies/generation1_context_routing.py",
                "src/mop/studio/generation1_supervisor.py",
                "src/mop/studio/local_throttle.py",
            ],
            validation=(
                "8166 completed campaign/scheduler/test Python LOC removed for 12 survivor and "
                "integrity-pin LOC, net -8154; dedicated 1345-line manifest/policy surface also "
                "deleted; four live result/verification artifacts are complete and their 100 MB "
                "evidence trees remain untouched; retained C2 direct assertions and 45-case "
                "notifier/supervisor boundary green; 3177-case collection, compile-all, critical "
                "ruff, and docs gate clean; all 16 paths recover"
            ),
            rollback_tag="mop-collapse-successor-horizons",
            next_action="audit completed Generation-1 mechanics, context-routing, and final-campaign layers",
        )
    )
    checklist.append(
        item(
            "RED-completed-generation1-program",
            8,
            "verified_reduction",
            "Retire the completed Generation-1 program and generated proof surface",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/studio/local_throttle.py",
                "configs/local_execution_throttle.yaml",
            ],
            validation=(
                "37467 campaign, controller, notifier, script, and dedicated-test Python LOC "
                "removed for 10 generic-governor LOC, net -37457; generated outputs add 265643 "
                "non-Python deletions; 2879-case collection, compile-all, 39-pass/8-skip retained "
                "coexistence boundary, direct scheduler assertions, and 32-case docs gate green; "
                "all 127 paths recover and protected live evidence remains present"
            ),
            rollback_tag="mop-collapse-generation1-program",
            next_action="delete the next completed experimental vertical slice",
        )
    )
    checklist.append(
        item(
            "RED-pre-generation1-campaign-escs",
            8,
            "verified_reduction",
            "Retire the pre-Generation1 campaign, throttle, and ESCS substrate stack",
            status="verified",
            evidence=[
                "collapse/MOP_REDUCTION_LOG.json",
                "collapse/MOP_HISTORICAL_CODE_INDEX.json",
                "src/mop/ladder/stage3_registry.py",
                "src/mop/beds/starss23/adapter.py",
            ],
            validation=(
                "95905 campaign, evidence, controller, throttle, ESCS substrate, and dedicated-test "
                "Python LOC removed for 19 retained-kernel LOC, net -95886; generated outputs add "
                "88876 non-Python deletions; 2110-case collection, compile-all, 24-case registry/"
                "ladder boundary, 119-case Stage-3/event/STARSS boundary, and 32-case docs gate "
                "green; all 222 deleted paths recover and protected live evidence remains present"
            ),
            rollback_tag="mop-collapse-pre-generation1-campaign",
            next_action="delete the next sealed campaign or unreachable experimental framework",
        )
    )

    # accumulate verified reductions from the append-only log
    red = {
        "eliminated_LOC": 0,
        "deduplicated_LOC": 0,
        "relocated_LOC": 0,
        "archived_LOC": 0,
        "generated_replacement_LOC": 0,
        "added_LOC": 0,
    }
    for ev in redlog.get("events") or []:
        for k in red:
            red[k] += int(ev.get(k, 0) or 0)
    red["net_global_reduction_LOC"] = (
        red["eliminated_LOC"] + red["deduplicated_LOC"] + red["archived_LOC"] - red["added_LOC"]
    )

    # reconcile: record the evidence-authority deletion map and move SEC-9 into active analysis
    checklist.append(
        item(
            "ART-MOP_EVIDENCE_EQUIVALENCE.json",
            9,
            "artifact",
            "MOP_EVIDENCE_EQUIVALENCE.json (evidence-primitive deletion map)",
            status="complete",
            evidence=["collapse/MOP_EVIDENCE_EQUIVALENCE.json"],
            validation="normalized-AST body clustering of every owned primitive definition",
            next_action="none",
        )
    )
    checklist.append(
        item(
            "ART-MOP_EVIDENCE_MIGRATION.json",
            9,
            "artifact",
            "MOP_EVIDENCE_MIGRATION.json (per-duplicate migration table)",
            status="complete",
            evidence=["collapse/MOP_EVIDENCE_MIGRATION.json"],
            validation="123 rows; batches: batch1_studies_safe/verifier-defer/controller-defer/inspect",
            next_action="execute remaining batches under their gates",
        )
    )
    checklist.append(
        item(
            "RED-batch1",
            9,
            "verified_reduction",
            "Evidence core batch1: 9 studies modules deduplicated onto mop.substrate.events",
            status="verified",
            evidence=["collapse/MOP_REDUCTION_LOG.json"],
            validation="77 LOC removed; byte-identical + py_compile + 9/9 import parity + known-answer",
            commit="",
            rollback_tag="mop-collapse-evidence-batch1",
            next_action=(
                "next batch: sha256_file dominant cluster (9), then _atomic_write (6), "
                "then distinct-body inspection"
            ),
        )
    )
    collapsible = (equiv.get("totals") or {}).get("redundant_definitions_collapsible")
    for it in checklist:
        if it["id"] == "SEC-9":
            it["status"] = "complete"
            it["evidence_paths"] = [
                "src/mop/evidence.py",
                "src/mop/beds/starss23/count_verifier.py",
            ]
            it["validation"] = (
                f"the original map identified {collapsible} byte-identical definitions; the final tree "
                "has one production serializer/hasher and retains the required independent verifier"
            )
            it["dependency"] = ""
            it["next_action"] = "prevent parallel evidence authorities"

    # Completion is reconciled only from item-specific evidence above. A final audit may override
    # remaining items after its exact requirement, command, and artifact checks have passed.

    # live run state (read-only), for the ledger header
    live_status = {}
    live_path = Path("/Users/scammermike/Downloads/mop/runs/generation1/general-run/current_status.json")
    if live_path.exists():
        try:
            s = json.loads(live_path.read_text())
            live_status = {
                "state": s.get("state"),
                "stage": s.get("stage"),
                "updated_at": s.get("updated_at"),
                "counts": s.get("counts"),
            }
        except Exception:
            live_status = {"state": "unreadable"}

    by_status: dict[str, int] = {}
    for it in checklist:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1

    state = {
        "schema": "mop-collapse-state/v1",
        "spec": "MOP_ACCRETION_COLLAPSE.md",
        "governing_principle": (
            "One scientific kernel. One evidence language. One experiment engine. "
            "One controller. One registry. One interface. Full breadth. Minimal mass."
        ),
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
            "entrypoints": 10,
            "controllers": 1,
            "evidence_engines": 1,
            "experiment_frameworks": 1,
            "registries": 1,
            "config_roots": 1,
            "cli": 1,
        },
        "key_findings": {
            "duplicate_integrity_primitive_definitions": (authority.get("implementation_counts") or {}),
            "duplicate_integrity_total": sum((authority.get("implementation_counts") or {}).values()),
            "scripts_class_counts": (command.get("class_counts") or {}),
            "lifecycle_boilerplate_files": dup.get("total_boilerplate_files"),
            "lifecycle_boilerplate_LOC": dup.get("total_boilerplate_LOC"),
            "evidence_primitive_defs_analyzed": (equiv.get("totals") or {}).get("primitive_definitions"),
            "evidence_primitive_defs_collapsible": (equiv.get("totals") or {}).get(
                "redundant_definitions_collapsible"
            ),
            "highest_pressure_first_region": (
                "section 9 evidence authority: 168 duplicate integrity "
                "definitions collapse to one evidence core, provable by "
                "byte-parity + mutation tests (pure functions, live-safe)"
            ),
            "floor_correction": (
                "duplicate-function analysis (14.3k byte-identical, 27k structural) measures "
                "similarity WITHIN the current shape, NOT the architectural-collapse ceiling. "
                "Measured STARSS23 anatomy: 18669 collapsible LOC (68 percent) fold into the "
                "shared engine. The 50k global target is not disproven."
            ),
            "starss23_collapsible_loc": (starss.get("collapsible_loc") if starss else None),
            "starss23_preserved_loc": (starss.get("preserved_loc") if starss else None),
            "shared_engine": (
                "src/mop/science architecture B (269 LOC), including shared producer "
                "receipt, finalization, and safety paths; Architecture A deleted"
            ),
            "shared_budget_engine": (
                "src/mop/science/budget.py (704 LOC), three old harnesses, eight "
                "producer budget-point assemblers, and five seed-record copies deleted"
            ),
            "shared_statistics_engine": (
                "src/mop/science/statistics.py (298 LOC), including shared "
                "onset and count artifact projections"
            ),
            "selected_experiment_architecture": (architecture.get("selection") or {}).get("selected"),
            "proof_index": {
                "files": proof_index["files"],
                "bytes": proof_index["bytes"],
                "duplicate_groups": len(proof_index["duplicate_groups"]),
            },
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

    # Render a compact human view. Full checklist and history stay machine-readable.
    tracked_python = [
        ROOT / path for path in sh("git", "ls-files", "*.py").splitlines() if (ROOT / path).is_file()
    ]
    current_python_loc = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in tracked_python)
    active = [
        it for it in checklist if it["status"] in {"active", "partial"} and it["kind"] in {"region", "target"}
    ]
    recent = (redlog.get("events") or [])[-12:]
    lines = [
        "# MOP Collapse Ledger",
        "",
        "Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json` and "
        "`collapse/MOP_REDUCTION_LOG.json`.",
        "",
        "## Current",
        "",
        f"- Maintained Python: {current_python_loc:,} LOC; ceiling: 50,000.",
        f"- Verified net Python reduction: {red['net_global_reduction_LOC']:,} LOC.",
        f"- Checklist: {json.dumps(by_status, sort_keys=True)}.",
        "- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and "
        "`collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.",
        "",
        "## Active boundaries",
        "",
    ]
    lines.extend(f"- {it['id']}: {it['title']} -> {it['next_action']}" for it in active)
    lines.extend(
        [
            "",
            "## Recent green reductions",
            "",
            "| tag | net LOC | batch |",
            "| --- | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {event.get('tag', '')} | {int(event.get('net_reduction_LOC', 0)):,} | {event.get('batch', '')} |"
        for event in recent
    )
    lines.extend(
        [
            "",
            "Older checkpoints, proof text, and exact accounting remain in the machine log.",
            "",
        ]
    )
    (ROOT / "MOP_COLLAPSE_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"checklist items: {len(checklist)}")
    print(f"by status: {json.dumps(by_status)}")
    print("wrote MOP_COLLAPSE_STATE.json + MOP_COLLAPSE_LEDGER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
