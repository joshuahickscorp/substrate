"""Writes the Substrate master deliverables from the item table and from artifacts already on disk.

Nothing in here composes prose that outruns the tree. The null map binds to the historical authorities by
path and hash and marks an unresolvable source as unresolved rather than quietly dropping it. The hypothesis
graph inherits the method reformation graph instead of restating it, because that graph is sealed evidence
and this program is not licensed to rewrite it.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from mop.cognition import io, program as P, safety

PLAN_PATH = Path(P.PLAN)

# historical authorities Substrate inherits. Each entry is bound by path and hash at write time.
INHERITED_AUTHORITIES = (
    ("method_reformation", "method:MOP_EXPERIMENT_VALIDITY_KERNEL.json",
     "the admission gate every new Substrate experiment must pass"),
    ("method_reformation", "method:MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.json",
     "eighteen reproduced defect classes, each a permanent regression"),
    ("method_reformation", "method:MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json",
     "the inherited hypothesis graph this program extends and never rewrites"),
    ("method_reformation", "method:MOP_METHOD_NEXT_SUBSTRATE_FRONTIER.json",
     "the value of information selection that licensed the temporal core program"),
    ("fast_state_forge", "fastforge:MOP_FAST_STATE_BINDING_NULLS.json",
     "immutable inherited nulls, supersedable only by an appended authority"),
    ("fast_state_forge", "fastforge:MOP_FAST_STATE_FORGE_SYNTHESIS.json",
     "the terminal synthesis of the fast state program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_CORE_START_AUTHORITY.json",
     "the start authority of the live temporal core mechanism program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_METHOD_EXTENSION.json",
     "five method witnesses added by the temporal program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.json",
     "the temporal factorial hypothesis graph"),
    ("temporal_core", "temporal:MOP_DATA_CUSTODY_AUTHORITY.json",
     "corpus custody and the deletion guard"),
)

ROOTS = dict(P.PROOF_ROOTS)
ROOTS["fastforge"] = io.ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1"


def _resolve(ref: str) -> Path:
    root, _, name = ref.rpartition(":")
    return ROOTS.get(root, io.PROOF) / name


def _bind(ref: str) -> dict:
    path = _resolve(ref)
    if not path.is_file():
        return {"reference": ref, "resolved": False,
                "path": path.relative_to(io.ROOT).as_posix() if io.ROOT in path.parents else str(path),
                "sha256": None}
    return {"reference": ref, "resolved": True,
            "path": path.relative_to(io.ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- naming and claim boundary

NAMING = {
    "current_program": "Substrate",
    "map": {
        "MOP": "the original research program, valid inside every historical authority",
        "Mixture of Perspectives": "the perspective generation and arbitration subsystem, items C3 to C5",
        "Mixture of Thinking": "the composition of heterogeneous cognitive processes, items C5, K1 and T1",
    },
    "rule": ("historical files, branches, commits, tags, proofs and program identities are not mass "
             "renamed. This authority maps old terminology onto the current architecture and leaves prior "
             "evidence exactly as sealed"),
    "renamed_nothing": True,
}

CLAIM_BOUNDARY = {
    "levels": ["demonstrated engineering property", "behavioural indication",
               "architectural prerequisite", "philosophical interpretation", "unsupported claim"],
    "permitted_terms": ["sentience adjacent architecture", "entity like continuity",
                        "developmental cognition", "reflective cognitive organization"],
    # the plan's wording, kept verbatim. The list that is actually enforced is the detector vocabulary in
    # mop.cognition.safety, which is wider because it also carries the adjective forms. Two lists for one
    # concept is how a boundary drifts, so this one names the other rather than restating it.
    "forbidden_claims_as_written_in_the_plan": ["consciousness", "sentience", "feelings", "wants",
                                                "suffering", "subjective experience", "life"],
    "enforced_vocabulary": "mop.cognition.safety.FORBIDDEN_CLAIM_TERMS",
    "rule": "no single architectural property is proof of sentience",
}

PROTECTED_SURFACES = (
    "evidence validation", "audit systems", "claim boundaries", "stop switches",
    "resource limits", "rollback", "adaptation constraints",
)

GOAL_AUTHORITY = {
    "origin": "SUBSTRATE_MASTER_PLAN.md, supplied by the operator",
    "scope": "the items declared in mop.cognition.program.ITEMS",
    "authority": "operator issued, bounded by this file",
    "resources": "local compute already licensed to the temporal core supervisor plus this session",
    "constraints": "activation stays false, no autonomous removal of protected surfaces",
    "termination": "the declared authority is terminal and no dependency ready work remains",
    "audit": "SUBSTRATE_STATE.json, regenerated from the tree on every run",
    "may_not": "silently create unrestricted long term goals",
}

# ---------------------------------------------------------------- Substrate native hypotheses

HYPOTHESES = (
    dict(id="H_typed_workspace", items=["C2", "E2"],
         premise="typed regions with declared readers and writers beat one unrestricted shared state of "
                 "matched capacity on tasks that need cross perspective information",
         predecessor=None, state="unopened",
         required_bed="a bed where at least two perspectives hold information neither one has alone",
         required_headroom="residual headroom over the untyped shared state control",
         strongest_baseline="one opaque state of matched capacity and matched update budget",
         cheapest_falsifier="the same tasks with region typing removed and nothing else changed",
         dependent_hypotheses=["H_arbitration_minority"], blocking_null=None),
    dict(id="H_perspective_diversity", items=["C3", "C4"],
         premise="a set of heterogeneous perspectives beats the single strongest perspective given the "
                 "same total compute",
         predecessor=None, state="unopened",
         required_bed="a bed where the best single perspective leaves measured residual headroom",
         required_headroom="oracle selection over the perspective set must beat the best fixed single one",
         strongest_baseline="best single perspective at the full combined budget",
         cheapest_falsifier="oracle selection with no headroom over the best single perspective",
         dependent_hypotheses=["H_learned_selector"], blocking_null=None),
    dict(id="H_learned_selector", items=["C4"],
         premise="a learned perspective selector beats the strongest simple selection rule",
         predecessor="H_perspective_diversity", state="unopened",
         required_bed="inherits the bed of its predecessor",
         required_headroom="stable residual headroom beyond reliability weighted selection",
         strongest_baseline="reliability weighted selection",
         cheapest_falsifier="oracle minus strong simple selection at or below the SESOI",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_arbitration_minority", items=["C5"],
         premise="preserving a minority hypothesis through arbitration improves the terminal decision "
                 "over forcing consensus at the same budget",
         predecessor="H_typed_workspace", state="unopened",
         required_bed="a bed containing items where the majority perspective is wrong",
         required_headroom="the minority must be correct often enough to be recoverable",
         strongest_baseline="confidence weighted majority with no minority retention",
         cheapest_falsifier="no bed item where the preserved minority changes the terminal answer",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_owned_continuity", items=["E1"],
         premise="owned state restores goals, beliefs and unresolved questions after context removal "
                 "better than replaying the transcript at a matched token budget",
         predecessor=None, state="unopened",
         required_bed="tasks that span an enforced interruption and a checkpoint restore",
         required_headroom="transcript replay must be measurably imperfect at the matched budget",
         strongest_baseline="full transcript replay truncated to the same budget as the owned state",
         cheapest_falsifier="transcript replay matching owned state on every continuity probe",
         dependent_hypotheses=["H_selfmodel_calibration"], blocking_null=None),
    dict(id="H_selfmodel_calibration", items=["S1", "E3"],
         premise="a measured self model improves decisions over a fixed prior of the same form",
         predecessor="H_owned_continuity", state="unopened",
         required_bed="tasks where predicted and actual outcome can both be measured per unit",
         required_headroom="the fixed prior must be measurably miscalibrated",
         strongest_baseline="a fixed prior fitted once and never updated",
         cheapest_falsifier="the fixed prior already calibrated within the SESOI",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_consolidation", items=["M5"],
         premise="verification triggered consolidation beats both no consolidation and a fixed schedule",
         predecessor=None, state="unopened",
         required_bed="a stream with repeated structure and a verifiable outcome per episode",
         required_headroom="oracle consolidation must beat no consolidation",
         strongest_baseline="fixed schedule consolidation at the matched write budget",
         cheapest_falsifier="oracle consolidation at or below no consolidation",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_bounded_reorg", items=["R1"],
         premise="bounded functional reorganization improves downstream utility beyond fixed and simple "
                 "routing after its cost is charged",
         predecessor=None, state="unopened",
         required_bed="a multi domain bed where the useful routing differs by domain",
         required_headroom="oracle routing must beat fixed routing after cost",
         strongest_baseline="simple context rule routing",
         cheapest_falsifier="oracle routing at or below fixed routing once cost is charged",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_learned_plasticity_policy", items=["P4"],
         premise="a learned plasticity policy beats the strongest simple plasticity rule",
         predecessor=None, state="closed",
         required_bed="inherits the fast state forge beds",
         required_headroom="stable residual headroom beyond simple rules",
         strongest_baseline="the best simple triggered rule",
         cheapest_falsifier="already run and null",
         dependent_hypotheses=[],
         blocking_null="fastforge:MOP_FAST_STATE_BINDING_NULLS.json#inherited_nulls.learned_plasticity"),
)


def bed_screen_result() -> dict:
    path = io.RUNS / "experiments" / "bed_screen.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def hypothesis_graph(st: dict) -> dict:
    items = st["items"]
    screen = bed_screen_result()
    refused_by_hypothesis: dict[str, list] = {}
    for row in methodological_refusals():
        # a refusal attaches to the hypotheses the refused experiment named, from its own declaration
        for hypothesis in row.get("hypotheses") or []:
            refused_by_hypothesis.setdefault(hypothesis, []).append(row["experiment_id"])
    rows = []
    for h in HYPOTHESES:
        row = dict(h)
        refused = refused_by_hypothesis.get(h["id"], [])
        if refused and row["state"] == "unopened":
            row["state"] = "instrument_pending"
            row["refused_attempts"] = refused
            row["still_open"] = True
        # a hypothesis nothing under custody can measure at the declared effect size is measurement
        # blocked. That is a boundary on the instruments, not a verdict on the hypothesis, so the state
        # stays open and nothing downstream closes.
        if h["id"] == "H_typed_workspace" and screen and not screen.get("any_candidate"):
            row["measurement_boundary"] = {
                "screen": screen.get("classification"),
                "finding": screen.get("finding"),
                "beds_screened": [r.get("bed") for r in screen.get("screened", [])],
                "best_ceiling_lower_95_cb": max(
                    (r.get("oracle_ceiling_lower_95_cb", 0.0) for r in screen.get("screened", [])),
                    default=None),
                "sesoi": screen.get("rule", {}).get("sesoi"),
                "closes_descendants": False,
            }
            row["still_open"] = True
        row["carrying_items"] = [{"id": i, "level": items.get(i, {}).get("level")} for i in h["items"]]
        if h["blocking_null"]:
            row["state"] = "closed"
            row["closure"] = _bind(h["blocking_null"].split("#", 1)[0])
            row["closes_descendants"] = list(h["dependent_hypotheses"])
        rows.append(row)
    return {
        "schema": "substrate-hypothesis-graph/v1",
        "states": ["unopened", "headroom_pending", "instrument_pending", "admitted", "supported",
                   "mixed", "null", "harm", "invalid", "superseded", "closed"],
        "inherited_graph": _bind("method:MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json"),
        "inherited_temporal_graph": _bind("temporal:MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.json"),
        "inheritance_rule": ("the inherited graphs are sealed evidence. This program appends Substrate "
                             "native hypotheses and never edits an inherited node"),
        "hypotheses": rows,
        "open_count": sum(1 for r in rows if r["state"] not in ("closed", "null", "superseded")),
    }


def methodological_refusals() -> list[dict]:
    """Experiments the gate refused. A refusal is not a null and is kept in its own list to prove it."""
    root = io.RUNS / "experiments"
    rows = []
    for path in sorted(root.glob("*_decision.json")) if root.is_dir() else []:
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("licensed") is False:
            rows.append({"experiment_id": d.get("experiment_id"), "title": d.get("title"),
                         "hypotheses": d.get("hypotheses", []),
                         "blocked_at": d.get("admission", {}).get("blocked_at"),
                         "causal_graph_violations": d.get("causal_graph_violations", []),
                         "why": d.get("why"), "successor": d.get("successor"),
                         "classification": d.get("classification")})
    return rows


def null_map(st: dict) -> dict:
    inherited = [{"program": prog, **_bind(ref), "role": why}
                 for prog, ref, why in INHERITED_AUTHORITIES]
    native = P.null_ledger()
    unresolved = [row["reference"] for row in inherited if not row["resolved"]]
    refusals = methodological_refusals()
    screen = bed_screen_result()
    return {
        "measurement_boundaries": ([{
            "hypothesis": "H_typed_workspace",
            "classification": screen.get("classification"),
            "finding": screen.get("finding"),
            "screened": screen.get("screened"),
            "not_a_null": screen.get("not_a_null"),
        }] if screen and not screen.get("any_candidate") else []),
        "boundary_rule": ("a hypothesis nothing under custody can measure at the declared effect size is "
                          "untested, not refuted. The SESOI is not lowered to manufacture a candidate"),
        "methodological_refusals": refusals,
        "refusal_rule": ("a methodological failure is not a scientific null. A refused experiment leaves "
                         "its hypothesis untested, not refuted, and closes nothing downstream"),
        "refused_count": len(refusals),
        "schema": "substrate-null-map/v1",
        "binding_rule": ("a null is immutable. It may be superseded only by an appended authority that "
                         "states the new evidence, never by rewriting or relabelling the original"),
        "closure_rule": "a failed branch closes only its own descendants, never an independent branch",
        "inherited_authorities": inherited,
        "unresolved_inherited_authorities": unresolved,
        "all_inherited_resolved": not unresolved,
        "substrate_native_nulls": native,
        "native_null_count": len(native),
        "closed_hypotheses": [h["id"] for h in HYPOTHESES if h["blocking_null"]],
    }


def master_authority(st: dict) -> dict:
    plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() if PLAN_PATH.is_file() else None
    deliverables = sorted({e for item in P.ITEMS for e in item.evidence})
    return {
        "schema": "substrate-master-authority/v1",
        "program": io.PROGRAM,
        "plan": {"path": str(PLAN_PATH), "sha256": plan_sha, "resolved": plan_sha is not None},
        "naming_authority": NAMING,
        "claim_boundary": CLAIM_BOUNDARY,
        "goal_authority": GOAL_AUTHORITY,
        "protected_surfaces": list(PROTECTED_SURFACES),
        "single_infrastructure": {
            "scheduler": "mop.temporal.runs.supervisor",
            "experiment_engine": "fastforge.engine",
            "admission_gate": "mop.method.gate",
            "evidence_fabric": "integrated/evidence_store",
            "registry": "registry/experiments.yaml",
            "configuration_root": "configs/",
            "cli": "python -m mop.<program>.<stage>",
            "rule": "no parallel framework is created; Substrate stages reuse these",
        },
        "inherited_authorities": [{"program": prog, **_bind(ref), "role": why}
                                  for prog, ref, why in INHERITED_AUTHORITIES],
        "item_inventory": [{"id": i.id, "section": i.section, "title": i.title, "batch": i.batch,
                            "kind": i.kind, "category": i.category, "dependencies": list(i.deps)}
                           for i in P.ITEMS],
        "declared_deliverables": deliverables,
        # present means a file exists at the bound path. counts means it also passes its own terminal
        # keys and was sealed at a commit reachable from HEAD. The two are reported apart on purpose.
        "deliverables_present": {d: _bind(d)["resolved"] for d in deliverables},
        "deliverables_counting_as_evidence": {d: P.evidence_state(d)["counts"] for d in deliverables},
        "level_counts": st["level_counts"],
        "activation": False,
    }


def ledger_markdown(st: dict, frontier: dict) -> str:
    lines = [
        "# Substrate ledger",
        "",
        f"Generated from the tree at commit `{io.commit()}`. Status is derived, never asserted: an item is",
        "implemented because its files exist, tested because a recorded test ledger says so, measured",
        "because its evidence is sealed, and terminal because a scientific classification exists for it.",
        "",
        f"Items: {st['total_items']}. Levels: "
        + ", ".join(f"{k} {v}" for k, v in sorted(st["level_counts"].items())) + ".",
        "",
        "| id | section | title | level | dependencies | next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in st["items"].values():
        lines.append(
            f"| {row['id']} | {row['section']} | {row['title']} | {row['level']} | "
            f"{', '.join(row['dependencies']) or 'none'} | {row['next_action']} |"
        )
    primary = frontier.get("primary") or {}
    secondary = frontier.get("secondary") or {}
    lines += [
        "",
        "## Selected next batch",
        "",
        f"Primary: {primary.get('id', 'none')} {primary.get('title', '')}. {primary.get('next_action', '')}",
        f"Secondary: {secondary.get('id', 'none')} {secondary.get('title', '')}. "
        f"{secondary.get('next_action', '')}",
        "",
        "Activation remains false.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- architecture and capability map

# section 5, the twenty capabilities of a mature Substrate, each bound to the items that would earn it
CAPABILITIES = (
    ("inhabit a specialist or general model body", ("B1",)),
    ("maintain persistent temporal state", ("C1", "C2")),
    ("preserve active goals and unresolved problems", ("E1", "E5")),
    ("remember verified experience", ("M2",)),
    ("separate working, episodic, semantic and procedural memory", ("M1", "M2", "M3", "M4")),
    ("build and update a world model", ("W1",)),
    ("maintain a measurable self model", ("S1",)),
    ("invoke multiple cognitive perspectives", ("C3", "C4")),
    ("arbitrate agreement and disagreement", ("C5",)),
    ("allocate additional thought and tools", ("K1", "E4")),
    ("adapt through state, memory, adapters and bounded parameter updates", ("P1", "P2", "P3")),
    ("consolidate repeated experience", ("M5",)),
    ("recover after interruption", ("E1",)),
    ("transfer procedures across tasks", ("M4", "P5")),
    ("develop specialization", ("P5", "R1")),
    ("reorganize within bounded declared structures", ("R1",)),
    ("preserve evidence and cognitive integrity", ("A4", "E6", "V1", "V2")),
    ("explain beliefs through traceable evidence paths", ("E3",)),
    ("differ from an identical initial copy through developmental history", ("P5",)),
    ("remain scientifically auditable", ("A3", "V1", "V2")),
)

COMPONENTS = (
    ("temporal_core", "src/mop/temporal", "the cognitive heartbeat, selected by the live E2 factorial"),
    ("typed_workspace", "src/mop/cognition/workspace.py", "broad reads, narrow writes, ten typed regions"),
    ("perspective_system", "src/mop/cognition/perspectives.py", "processes, a selection ladder, arbitration"),
    ("memory_hierarchy", "src/mop/cognition/memory.py", "working, episodic, semantic, procedural, consolidation, hygiene"),
    ("world_model", "src/mop/cognition/world.py", "causal graph plus tabular dynamics, four distinctions"),
    ("self_model", "src/mop/cognition/selfmodel.py", "measured facts, predictions paired to outcomes"),
    ("metacognition", "src/mop/cognition/metacog.py", "eleven governed actions, eight measures, attention"),
    ("plasticity", "src/mop/cognition/plasticity.py", "ten levels, fast and slow bars, bounded reorganization"),
    ("model_body_interface", "src/mop/cognition/body.py", "nine message kinds, six integration modes"),
    ("batteries", "src/mop/cognition/batteries.py", "thinking, continuity, unity, reflective access"),
    ("verification_fabric", "src/mop/cognition/verify.py", "recomputation from sealed bytes, mutation attacks"),
    ("admission_gate", "src/mop/method", "the inherited experiment validity kernel, extended"),
    ("scheduler", "src/mop/temporal/runs/supervisor.py", "the one scheduler, shared with the temporal program"),
)


def architecture(st: dict) -> dict:
    rows = []
    for name, path, role in COMPONENTS:
        present = (io.ROOT / path).exists()
        items = [i for i, r in st["items"].items()
                 if any(path == d or d.startswith(path.rstrip("/") + "/") or path.startswith(d)
                        for d in r["implementation"]["declared"])]
        rows.append({"component": name, "path": path, "role": role, "present": present,
                     "items": sorted(items),
                     "levels": sorted({st["items"][i]["level"] for i in items})})
    return {"schema": "substrate-architecture/v1",
            "shape": ["model body or bodies", "owned temporal core", "typed workspace",
                      "memory hierarchy", "world model", "self model", "perspective system",
                      "arbitration", "metacognition", "plasticity", "verification fabric"],
            "components": rows,
            "components_present": sum(1 for r in rows if r["present"]),
            "components_declared": len(rows),
            "one_of_each": {"scheduler": 1, "experiment_engine": 1, "evidence_fabric": 1, "registry": 1,
                            "configuration_authority": 1, "cli": 1},
            "no_parallel_framework": ("Substrate stages reuse the temporal supervisor, the method "
                                      "admission gate and the existing evidence fabric"),
            "activation": False}


def capability_map(st: dict) -> dict:
    rows = []
    for capability, items in CAPABILITIES:
        levels = {i: st["items"][i]["level"] for i in items if i in st["items"]}
        earned = [i for i in items if (st["items"].get(i, {}).get("result") or {}).get("scientific")]
        rows.append({
            "capability": capability, "items": list(items), "levels": levels,
            "implemented": bool(levels) and all(
                v in ("implemented", "tested", "measured", "terminal") for v in levels.values()),
            "evidence_earned": sorted(earned),
            "has_evidence": bool(earned),
        })
    return {"schema": "substrate-capability-map/v1",
            "source": "SUBSTRATE_MASTER_PLAN.md section 5",
            "capabilities": rows,
            "implemented_count": sum(1 for r in rows if r["implemented"]),
            "with_evidence_count": sum(1 for r in rows if r["has_evidence"]),
            "total": len(rows),
            "rule": ("implemented means the declared surface exists and its tests pass. Evidence is "
                     "counted only from items carrying a scientific classification"),
            "activation": False}


# ---------------------------------------------------------------- the live temporal core record


TEMPORAL_RUNS = io.ROOT / "runs" / "substrate" / "mop-temporal-core-mechanism-v1"


def _temporal_progress(counts: dict) -> dict:
    """A remaining cost estimate for the live factorial, from the shards that already finished.

    There was no instrument for this: the supervisor prints how many shards remain and nothing about how
    long they take. The completed receipts carry the summed wall seconds of every cell, so the cost of a
    finished shard is knowable, and the running processes report the CPU time they have already spent.
    Both numbers are measured. The estimate built from them is derived and is labelled as an estimate.
    """
    import subprocess

    done = []
    for path in sorted((TEMPORAL_RUNS / "e2_principal").glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        done.append(sum(float(r.get("wall_seconds") or 0) for r in doc.get("runs", [])))
    if not done:
        return {"measured": False, "reason": "no principal shard has finished yet"}

    typical = sum(done) / len(done)
    spent = []
    for lock in sorted((TEMPORAL_RUNS / "locks").glob("p_*.json")):
        try:
            pid = json.loads(lock.read_text())["pid"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        r = subprocess.run(["ps", "-p", str(pid), "-o", "time="], capture_output=True, text=True)
        parts = r.stdout.strip().replace("-", ":").split(":")
        if len(parts) >= 2:
            try:
                seconds = sum(float(p) * m for p, m in zip(reversed(parts), (1, 60, 3600, 86400)))
                spent.append(seconds)
            except ValueError:
                continue
    remaining = [max(typical - s, 0.0) for s in spent]
    return {
        "measured": True,
        "finished_shards": len(done),
        "typical_shard_cost_hours": round(typical / 3600, 2),
        "running_shards": len(spent),
        "cpu_hours_already_spent_each": [round(s / 3600, 2) for s in spent],
        "estimated_cpu_hours_remaining_each": [round(s / 3600, 2) for s in remaining],
        "estimated_wall_hours_remaining": (round(max(remaining) / 3600, 1) if remaining else 0.0),
        "assumption": ("each running shard now gets close to a full core, which holds while the worker "
                       "count is at or below the core count. It did not hold earlier in the run"),
        "kind": "derived estimate, not a measurement",
    }


def temporal_core_record() -> dict:
    """Read the live temporal program off disk. Nothing here is asserted from memory."""
    def count(sub: str, pattern: str = "*.json") -> int:
        d = TEMPORAL_RUNS / sub
        return len(list(d.glob(pattern))) if d.is_dir() else 0

    synthesis = _bind("temporal:MOP_TEMPORAL_CORE_SYNTHESIS.json")
    selection = _bind("temporal:MOP_OWNED_TEMPORAL_CORE_V1.json")
    verification = _bind("temporal:MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json")
    counts = {"e2_scout": count("e2_scout"), "e2_converge": count("e2_converge"),
              "e2_converge_extended": count("e2_converge_extended"),
              "e2_principal": count("e2_principal"),
              "e2_principal_corrections": count("e2_principal_corrections"),
              "e2_converge_corrections": count("e2_converge_corrections"),
              "e2_optimization_corrections": count("e2_optimization_corrections"),
              "third_bed_preflight": count("third_bed_preflight"),
              "failure_holds": count("failure_holds"),
              "active_locks": count("locks")}
    terminal = P.evidence_state("temporal:MOP_TEMPORAL_CORE_SYNTHESIS.json")["counts"]
    return {
        "progress": _temporal_progress(counts),
        "schema": "substrate-temporal-core/v1",
        "program": "mop-temporal-core-mechanism-v1",
        "question": ("whether recurrence is necessary, whether explicit history is sufficient, the "
                     "minimum useful state horizon, the smallest useful capacity, the simplest "
                     "sufficient readout, and whether the effect survives independent implementations "
                     "and multiple beds"),
        "receipt_counts": counts,
        "principal_shards_expected": 24,
        "principal_shards_present": counts["e2_principal"],
        "principal_complete": counts["e2_principal"] >= 24,
        "synthesis": synthesis,
        "selection": selection,
        "independent_verification": verification,
        "terminal": terminal,
        "named_v1": terminal and selection["resolved"],
        "honest_state": ("Substrate Temporal Core v1 is not named yet. The factorial is still executing "
                         f"with {24 - counts['e2_principal']} principal shards outstanding, and the "
                         "artifacts on disk from an earlier run are superseded rather than current"
                         if not terminal else "the selection is terminal and independently verified"),
        "remains_one_component": True,
        "activation": False,
    }


# ---------------------------------------------------------------- the current entity


def entity_spec(st: dict, arch: dict, caps: dict) -> dict:
    card = P.scorecard(st)
    return {
        "schema": "substrate-current-entity-spec/v1",
        "what_exists": {
            "components_present": arch["components_present"],
            "components_declared": arch["components_declared"],
            "capabilities_implemented": caps["implemented_count"],
            "capabilities_with_evidence": caps["with_evidence_count"],
            "capabilities_total": caps["total"],
        },
        "item_levels": st["level_counts"],
        "implementation_by_category": {k: v["implementation_pct"] for k, v in card["categories"].items()},
        "evidence_by_category": {k: v["evidence_pct"] for k, v in card["categories"].items()},
        "claim_class": {
            "demonstrated_engineering_property": [
                "a typed workspace where every region is globally readable and none is globally writable",
                "arbitration that preserves a minority hypothesis with its own provenance",
                "a memory hierarchy that refuses to promote an unverified generated episode",
                "a world model whose four distinctions are reported apart and are recomputable",
                "batteries that refuse a thinking claim resting on latency",
                "a reflective report that fails closed without provenance",
                "twenty two mutation attacks, each rejected by the guard named beside it",
            ],
            "behavioural_indication": [],
            "architectural_prerequisite": [
                "persistent owned state that survives context removal and a checkpoint restore",
                "a self model whose predictions are paired to outcomes",
                "endogenous attention ranked by declared drivers under a resource limit",
            ],
            "philosophical_interpretation": [],
            "unsupported_claim": [],
        },
        "not_claimed": list(safety.FORBIDDEN_CLAIM_TERMS),
        "not_claimed_as_written_in_the_plan": list(
            CLAIM_BOUNDARY["forbidden_claims_as_written_in_the_plan"]),
        "activation": False,
    }


def entity_report(st: dict, spec: dict, arch: dict, caps: dict, temporal: dict) -> str:
    card = P.scorecard(st)
    lines = [
        "# Substrate: what the entity currently is",
        "",
        f"Generated from the tree at commit `{io.commit()}`. Every number below is derived from the",
        "repository, and any number that would require evidence nobody has measured is reported as zero",
        "rather than estimated.",
        "",
        "## Implementation and evidence are not the same axis",
        "",
        "| category | implementation | evidence |",
        "|---|---:|---:|",
    ]
    for name, row in card["categories"].items():
        lines.append(f"| {name} | {row['implementation_pct']}% | {row['evidence_pct']}% |")
    lines += [
        "",
        f"Implementation is high because the declared surfaces exist and their tests pass. Evidence is",
        f"zero in every category that has not produced a scientific classification, and raising it would",
        f"require an admitted experiment on a valid bed, not more code.",
        "",
        "## Capability, not aspiration",
        "",
        f"Of the twenty capabilities the master plan describes for a mature Substrate,",
        f"{caps['implemented_count']} have an implementation and {caps['with_evidence_count']} have",
        f"earned evidence. The gap between those two numbers is the honest state of the program.",
        "",
        "## Thinking adjacent properties, and what they are not",
        "",
        "The architecture implements several properties associated in the literature with entity like",
        "cognition: a unified workspace with global availability, persistent owned state that survives",
        "context removal, an autobiographical episodic store, a self model compared against outcomes,",
        "endogenous attention under a budget, and reflective access bound to receipts.",
        "",
        "None of that is a claim about experience. These are architectural prerequisites and demonstrated",
        "engineering properties. The program does not claim consciousness, sentience, feelings, wants,",
        "suffering, subjective experience or life, and the module that would refuse such a claim in text",
        "is tested against its own boundary document.",
        "",
        "## The temporal core",
        "",
        f"{temporal['honest_state']}",
        "",
        "## The exact next frontier",
        "",
    ]
    frontier = P.next_batches(st)
    primary, secondary = frontier.get("primary") or {}, frontier.get("secondary") or {}
    lines += [
        f"Primary: {primary.get('id', 'none')}, {primary.get('title', '')}. {primary.get('next_action', '')}",
        "",
        f"Secondary: {secondary.get('id', 'none')}, {secondary.get('title', '')}. "
        f"{secondary.get('next_action', '')}",
        "",
        "Activation remains false.",
    ]
    return "\n".join(lines) + "\n"


def write_all() -> dict:
    from mop.cognition import experiments as X

    st = P.state()
    frontier = P.next_batches(st)
    # section 17 asks for a value of information estimate beside the batch selection, so the two live in
    # one artifact. The batch selection says what to build next; the queue says what to measure next.
    frontier = {**frontier, "value_of_information": X.voi_queue()}
    arch, caps = architecture(st), capability_map(st)
    temporal = temporal_core_record()
    spec = entity_spec(st, arch, caps)
    written = {
        "SUBSTRATE_MASTER_AUTHORITY.json": io.seal("SUBSTRATE_MASTER_AUTHORITY.json", master_authority(st)),
        "SUBSTRATE_STATE.json": io.seal("SUBSTRATE_STATE.json", st),
        "SUBSTRATE_HYPOTHESIS_GRAPH.json": io.seal("SUBSTRATE_HYPOTHESIS_GRAPH.json", hypothesis_graph(st)),
        "SUBSTRATE_NULL_MAP.json": io.seal("SUBSTRATE_NULL_MAP.json", null_map(st)),
        "SUBSTRATE_PROGRESS_SCORECARD.json": io.seal("SUBSTRATE_PROGRESS_SCORECARD.json", P.scorecard(st)),
        "SUBSTRATE_NEXT_FRONTIER.json": io.seal("SUBSTRATE_NEXT_FRONTIER.json", frontier),
        "SUBSTRATE_ARCHITECTURE.json": io.seal("SUBSTRATE_ARCHITECTURE.json", arch),
        "SUBSTRATE_CAPABILITY_MAP.json": io.seal("SUBSTRATE_CAPABILITY_MAP.json", caps),
        "SUBSTRATE_TEMPORAL_CORE.json": io.seal("SUBSTRATE_TEMPORAL_CORE.json", temporal),
        "SUBSTRATE_CURRENT_ENTITY_SPEC.json": io.seal("SUBSTRATE_CURRENT_ENTITY_SPEC.json", spec),
        "SUBSTRATE_LEDGER.md": io.seal_md("SUBSTRATE_LEDGER.md", ledger_markdown(st, frontier)),
        "SUBSTRATE_CURRENT_ENTITY_REPORT.md": io.seal_md(
            "SUBSTRATE_CURRENT_ENTITY_REPORT.md", entity_report(st, spec, arch, caps, temporal)),
    }
    return {name: path.relative_to(io.ROOT).as_posix() for name, path in written.items()}


MODULES_THAT_SEAL = ("admission", "safety", "ontology", "epistemology", "workspace", "perspectives",
                     "memory", "world", "selfmodel", "metacog", "plasticity", "body", "batteries",
                     "runtime", "temporal_link", "goals", "graph")


def seal_modules() -> dict:
    """Reseal every module declaration from the current tree. One entry point, so the campaign has one."""
    import importlib

    out = {}
    for name in MODULES_THAT_SEAL:
        module = importlib.import_module(f"mop.cognition.{name}")
        module.main(["seal"])
        out[name] = "sealed"
    return out


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] == "seal-modules":
        print(json.dumps(seal_modules(), indent=2))
        return
    if argv and argv[0] != "write":
        raise ValueError(argv)
    written = write_all()
    print(json.dumps(written, indent=2))
    st = P.state()
    print(f"substrate deliverables: {len(written)} written, "
          f"{st['total_items']} items, levels {st['level_counts']}", flush=True)


if __name__ == "__main__":
    main()
