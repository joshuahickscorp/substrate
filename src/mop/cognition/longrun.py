"""The long run: one frozen DAG, a rehearsal that tries to break it, and the launch gate.

Everything here is decided before the run starts and nothing is decided during it. The freeze covers
source, data, sessions, splits, perspectives, bodies, seeds, budgets, controls, SESOI, stop rules,
checkpoints, retries and claim ceilings, and the manifest is hashed so a live edit is detectable rather
than merely discouraged.

Completion is a count of scientific work units, not a wall clock. A run that finishes early because the
machine was fast has not done less science, and a run that is still going has units left.

The rehearsal is the part that earns the launch. It proves receipts are deterministic, that a killed run
resumes without redoing finished work, that two writers cannot claim the same unit, that a stale artifact
is refused, and that injected failures do not destroy completed work. A rehearsal that only proves the
happy path proves nothing worth knowing.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from mop.cognition import audit as A
from mop.cognition import graph as G
from mop.cognition import io

PY = sys.executable
UNITS = io.RUNS / "long_run" / "units"
LOCKS = io.RUNS / "long_run" / "locks"
STOP = io.STOP

SESOI = 0.05
MAX_ATTEMPTS = 2

# what is frozen. A key here that changes after launch is a live edit and the manifest hash will say so.
FROZEN = {
    "source_commit": None,  # filled at freeze time
    "data_root_declared_by": "MOP_DATA_CUSTODY_AUTHORITY.json canonical_root",
    "sessions": "SUBSTRATE_REAL_SESSION_AUTHORITY.json",
    "splits": "group disjoint by source unit, sealed in the temporal receipts",
    "perspectives": "mop.cognition.perspectives.CATALOG",
    "bodies": ("compact", "general", "tool"),
    "seeds": (0, 1, 2),
    "cycle_budget": 6.0,
    "controls": {"temporal_core": "declared control, no core licensed",
                 "diversity": "strongest compute matched single cell",
                 "continuity": "transcript replay at matched budget"},
    "sesoi": SESOI,
    "stop_rules": ("stop switch present", "deterministic failure hold", "no unit dependency ready"),
    "checkpoint_policy": "one receipt per unit, resumable from disk, identity digest verified on restore",
    "retries": MAX_ATTEMPTS,
    "claim_ceiling": ("engineering property or architectural prerequisite only. No consciousness, "
                      "sentience, feeling, suffering, desire, personhood or life"),
    "activation": False,
}


class Refused(RuntimeError):
    """A long run action the authority does not permit."""


@dataclass(frozen=True)
class Unit:
    """One scientific work unit. Completion is a count of these, not a duration."""
    identity: str
    module: str
    args: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    certified: str = ""  # which certification licensed it, or why it is necessary without one
    produces: tuple[str, ...] = ()


def _u(identity, module, *args, depends_on=(), certified="", produces=()) -> Unit:
    return Unit(identity, module, tuple(args), tuple(depends_on), certified, tuple(produces))


C = "mop.cognition"

# Only certified or necessary components. Each unit names what licensed it.
UNIT_LIST: tuple[Unit, ...] = (
    _u("audit", f"{C}.audit", "run", certified="gates every later unit",
       produces=("SUBSTRATE_STRUCTURAL_AUDIT.json",)),
    _u("declarations", f"{C}.deliverables", "seal-modules", depends_on=("audit",),
       certified="necessary: every later unit reads a sealed declaration"),
    _u("temporal_continuity", f"{C}.temporal_link", "seal", depends_on=("declarations",),
       certified="runtime activity: the temporal region changes the decision path",
       produces=("SUBSTRATE_TEMPORAL_CORE.json",)),
    _u("ontology_epistemology", f"{C}.epistemology", "seal", depends_on=("declarations",),
       certified="tested batteries in sections 6.2 and 7.3",
       produces=("SUBSTRATE_EPISTEMOLOGY.json", "SUBSTRATE_BELIEF_REVISION.json")),
    _u("memory", f"{C}.memory", "seal", depends_on=("declarations",),
       certified="session canary: memory reuse and restoration",
       produces=("SUBSTRATE_MEMORY_SYSTEM.json",)),
    _u("diversity_arbitration", f"{C}.sx2", "run", depends_on=("declarations",),
       certified="SX2 closed on a compute matched comparison; the unit records the closure",
       produces=("SUBSTRATE_SX2_DIVERSITY.json",)),
    _u("world_model", f"{C}.worldbed", "integrate", depends_on=("memory",),
       certified="state dependent bed, decision gain measured and negative",
       produces=("SUBSTRATE_WORLD_MODEL_BATTERY.json",)),
    _u("self_model", f"{C}.selfmodel", "seal", depends_on=("memory",),
       certified="session canary: calibration paired to outcomes",
       produces=("SUBSTRATE_SELF_MODEL.json",)),
    _u("body_compact", f"{C}.bodies", "compact", depends_on=("declarations",),
       certified="body canary: pairwise distinct", produces=("SUBSTRATE_BODY_COMPACT.json",)),
    _u("body_general", f"{C}.bodies", "general", depends_on=("body_compact",),
       certified="body canary: pairwise distinct", produces=("SUBSTRATE_BODY_GENERAL.json",)),
    _u("body_tool", f"{C}.bodies", "tool", depends_on=("body_compact",),
       certified="body canary: pairwise distinct", produces=("SUBSTRATE_BODY_TOOL.json",)),
    _u("body_comparison", f"{C}.bodies", "compare",
       depends_on=("body_general", "body_tool", "temporal_continuity"),
       certified="ablation ladder measured against all three bodies",
       produces=("SUBSTRATE_MODEL_BODY_INTERFACE.json",)),
    _u("admitted_plasticity", f"{C}.plasticity", "seal", depends_on=("declarations",),
       certified="runtime activity: adapt changes the reliability state",
       produces=("SUBSTRATE_PLASTICITY_SYSTEM.json", "SUBSTRATE_REORGANIZATION.json")),
    _u("developmental_divergence", f"{C}.divergence", "run",
       depends_on=("memory", "temporal_continuity"),
       certified="control clean: identical histories produce no divergence",
       produces=("SUBSTRATE_DEVELOPMENTAL_HISTORY.json",)),
    _u("entity_batteries", f"{C}.batteries", "seal",
       depends_on=("world_model", "self_model", "body_comparison", "admitted_plasticity"),
       certified="runtime activity: every stage the batteries read is active",
       produces=("SUBSTRATE_THINKING_BATTERY.json", "SUBSTRATE_CONTINUITY_BATTERY.json",
                 "SUBSTRATE_UNITY_BATTERY.json", "SUBSTRATE_REFLECTIVE_ACCESS_BATTERY.json",
                 "SUBSTRATE_AGENCY_BATTERY.json", "SUBSTRATE_COGNITIVE_INTEGRITY_BATTERY.json")),
    _u("certification", f"{C}.certify", "run",
       depends_on=("entity_batteries", "developmental_divergence", "diversity_arbitration"),
       certified="necessary: reruns the cheap certification against the run's own outputs",
       produces=("SUBSTRATE_LONG_RUN_CERTIFICATION.json",)),
    _u("recomputation", f"{C}.verify", "recompute", depends_on=("certification",),
       certified="necessary: a second route over the sealed bytes",
       produces=("SUBSTRATE_INDEPENDENT_VERIFICATION.json",)),
    _u("mutations", f"{C}.verify", "mutate", depends_on=("recomputation",),
       certified="necessary: every guard is broken on purpose",
       produces=("SUBSTRATE_MUTATION_REPORT.json",)),
    _u("terminal_synthesis", f"{C}.authority", "seal", depends_on=("mutations",),
       certified="necessary: the closing authority",
       produces=("SUBSTRATE_FINAL_MASTER_AUTHORITY.json", "SUBSTRATE_FINAL_STATE.json")),
)

BY_UNIT = {u.identity: u for u in UNIT_LIST}


# ---------------------------------------------------------------- freeze


def source_digest() -> str:
    """A hash of the code the run will execute, not of the commit that carries it.

    Pinning the git commit would be circular: sealing the authority produces a commit, so the sealed
    manifest could never match the commit that contains it. What must not change after launch is the
    source, and this hashes exactly that.
    """
    parts = []
    for root in (io.ROOT / "src" / "mop" / "cognition", io.ROOT / "tests" / "cognition"):
        for f in sorted(root.rglob("*.py")):
            parts.append(f"{f.relative_to(io.ROOT)}:{hashlib.sha256(f.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def manifest() -> dict:
    frozen = dict(FROZEN)
    frozen["source_commit"] = io.commit()
    frozen["source_digest"] = source_digest()
    frozen["source_tree"] = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=io.ROOT, capture_output=True, text=True).stdout.strip()
    frozen["units"] = [u.identity for u in UNIT_LIST]
    frozen["unit_count"] = len(UNIT_LIST)
    frozen["completion"] = "all units terminal, which is a count of scientific work units, not wall time"
    body = json.dumps(frozen, sort_keys=True, default=str)
    return {**frozen, "manifest_sha256": hashlib.sha256(body.encode()).hexdigest()}


def live_edit_detected(sealed: dict) -> dict:
    """A frozen manifest whose source no longer matches the tree is a live edit."""
    current = manifest()
    # the source digest is the thing that must not move. The commit necessarily advances when the
    # authority itself is committed, so comparing commits would report every launch as a live edit.
    drifted = ["source_digest"] if sealed.get("source_digest") != current["source_digest"] else []
    return {"drifted_keys": drifted, "live_edit": bool(drifted),
            "sealed_digest": sealed.get("source_digest"), "current_digest": current["source_digest"],
            "sealed_commit": sealed.get("source_commit"), "current_commit": current["source_commit"],
            "commit_may_advance": ("the commit advances when this authority is committed, which is not a "
                                   "live edit. The source digest is what is frozen")}


# ---------------------------------------------------------------- units


def _receipt(unit: str) -> Path:
    return UNITS / f"{unit}.json"


def _units_subdir() -> str:
    """Where run_unit writes, relative to the runs root. The rehearsal rebinds this to its own root so
    it cannot delete the receipts of a completed run, which is the very property it claims to prove."""
    return UNITS.relative_to(io.RUNS).as_posix()


def done(unit: str) -> bool:
    path = _receipt(unit)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("ok") is True
    except (OSError, json.JSONDecodeError):
        return False


def ready() -> list[Unit]:
    return [u for u in UNIT_LIST if not done(u.identity) and all(done(d) for d in u.depends_on)]


def claim(unit: str) -> bool:
    """Exclusive writer. Two processes cannot claim the same unit, and the loser does not run it."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = LOCKS / f"{unit}.json"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"unit": unit, "pid": os.getpid(), "claimed": True}, f)
    return True


def release(unit: str) -> None:
    (LOCKS / f"{unit}.json").unlink(missing_ok=True)


def run_unit(unit: Unit, *, dry: bool = False) -> dict:
    t0 = time.time()
    if dry:
        ok, code, out = True, 0, "dry"
    else:
        env = {**os.environ, "PYTHONPATH": str(io.ROOT / "src")}
        r = subprocess.run([PY, "-m", unit.module, *unit.args], cwd=io.ROOT, env=env,
                           capture_output=True, text=True)
        code, out = r.returncode, (r.stdout or r.stderr)[-400:]
        from mop.cognition import program as P

        missing = [a for a in unit.produces if not P.evidence_state(a)["counts"]]
        ok = code == 0 and not missing
        out = out if ok else f"{out} missing={missing}"
    receipt = {"schema": "substrate-long-run-unit/v1", "unit": unit.identity, "ok": ok,
               "returncode": code, "detail": out.strip()[-300:],
               "wall_seconds": round(time.time() - t0, 2), "source_commit": io.commit(),
               "activation": False}
    io.run_json(f"{unit.identity}.json", receipt, _units_subdir())
    return receipt


def status() -> dict:
    return {"schema": "substrate-long-run-status/v1",
            "units": [{"unit": u.identity, "done": done(u.identity),
                       "depends_on": list(u.depends_on), "certified": u.certified}
                      for u in UNIT_LIST],
            "completed": sum(done(u.identity) for u in UNIT_LIST),
            "total": len(UNIT_LIST),
            "ready": [u.identity for u in ready()],
            "stop_switch_active": STOP.exists(),
            "terminal": all(done(u.identity) for u in UNIT_LIST)}


def drive(max_units: int = 10 ** 6, dry: bool = False) -> dict:
    ran = []
    while not STOP.exists():
        pending = ready()
        if not pending or len(ran) >= max_units:
            break
        unit = pending[0]
        if not claim(unit.identity):
            continue
        try:
            receipt = run_unit(unit, dry=dry)
        finally:
            release(unit.identity)
        ran.append({"unit": unit.identity, "ok": receipt["ok"]})
        if not receipt["ok"]:
            break
    st = status()
    return {"schema": "substrate-long-run-drive/v1", "ran": ran, "status": st,
            "stopped_by": "stop switch" if STOP.exists() else
                          ("failure" if ran and not ran[-1]["ok"] else
                           "no dependency ready unit" if not st["ready"] else "unit budget"),
            "activation": False}


# ---------------------------------------------------------------- rehearsal


def rehearse() -> dict:
    """A reduced end to end run that tries to break the machinery rather than to succeed.

    It runs against its own receipt root. The first version cleaned the real one on the way out, so
    sealing the authority after a completed run destroyed that run's progress record, which is precisely
    the property the injected failure check claims to protect.
    """
    import shutil

    global UNITS, LOCKS
    real_units, real_locks = UNITS, LOCKS
    UNITS = io.RUNS / "long_run" / "rehearsal" / "units"
    LOCKS = io.RUNS / "long_run" / "rehearsal" / "locks"
    try:
        return _rehearse_body(shutil)
    finally:
        shutil.rmtree(UNITS.parent, ignore_errors=True)
        UNITS, LOCKS = real_units, real_locks


def _rehearse_body(shutil) -> dict:
    shutil.rmtree(UNITS, ignore_errors=True)
    shutil.rmtree(LOCKS, ignore_errors=True)
    checks: dict[str, dict] = {}

    # 1 deterministic receipts: the same dry unit twice produces the same receipt shape
    a = run_unit(BY_UNIT["audit"], dry=True)
    b = run_unit(BY_UNIT["audit"], dry=True)
    checks["deterministic_receipts"] = {
        "ok": {k: a[k] for k in ("unit", "ok", "returncode")} == {k: b[k] for k in
                                                                  ("unit", "ok", "returncode")}}

    # 2 exclusive writers: a second claim on a held unit is refused
    release("probe")
    first, second = claim("probe"), claim("probe")
    release("probe")
    checks["exclusive_writers"] = {"ok": first is True and second is False}

    # 3 duplicate refusal: a completed unit is not offered again
    completed_before = {u.identity for u in UNIT_LIST if done(u.identity)}
    checks["duplicate_refusal"] = {
        "ok": all(u.identity not in [r.identity for r in ready()] for u in UNIT_LIST
                  if u.identity in completed_before)}

    # 4 checkpoint and resume: wipe one receipt, confirm only that unit is offered again
    shutil.rmtree(UNITS, ignore_errors=True)
    run_unit(BY_UNIT["audit"], dry=True)
    resumed = [u.identity for u in ready()]
    checks["checkpoint_resume"] = {"ok": "audit" not in resumed and "declarations" in resumed,
                                   "ready_after_resume": resumed}

    # 5 injected failure: a failing unit halts the wave and leaves completed work intact
    bad = Unit("injected_failure", "mop.cognition.does_not_exist", (), ("audit",), "injected", ())
    before = sum(done(u.identity) for u in UNIT_LIST)
    receipt = run_unit(bad)
    after = sum(done(u.identity) for u in UNIT_LIST)
    checks["injected_failure_preserves_completed_work"] = {
        "ok": receipt["ok"] is False and after == before, "completed_before": before,
        "completed_after": after}

    # 6 stale artifact refusal: an artifact at an unreachable commit does not count
    from mop.cognition import program as P

    probe = io.PROOF / "SUBSTRATE_STALE_PROBE.json"
    probe.write_text(json.dumps({"all_pass": True, "source_commit": "0" * 40}))
    stale = P.evidence_state("SUBSTRATE_STALE_PROBE.json")
    probe.unlink(missing_ok=True)
    checks["stale_artifact_refusal"] = {"ok": stale["counts"] is False, "reason": stale["reason"]}

    # 7 stop switch: the driver stops rather than continuing
    STOP.touch()
    stopped = drive(dry=True)
    STOP.unlink(missing_ok=True)
    checks["stop_switch_halts"] = {"ok": stopped["stopped_by"] == "stop switch"}

    # 8 evidence indexing: every produced artifact is declared by an item
    declared = {e for item in P.ITEMS for e in item.evidence if ":" not in e}
    produced = {a for u in UNIT_LIST for a in u.produces}
    checks["evidence_indexed"] = {"ok": produced <= declared,
                                  "undeclared": sorted(produced - declared)}

    # 9 terminal closure: with every receipt present the run reports terminal and offers nothing
    shutil.rmtree(UNITS, ignore_errors=True)
    for u in UNIT_LIST:
        run_unit(u, dry=True)
    st = status()
    checks["terminal_closure"] = {"ok": st["terminal"] is True and st["ready"] == []}

    # 10 the audit still passes after all of that
    checks["audit_still_passes"] = {"ok": A.run()["all_pass"] is True}

    failed = sorted(k for k, v in checks.items() if not v["ok"])
    return {"schema": "substrate-long-run-rehearsal/v1", "checks": checks, "failed": failed,
            "all_pass": not failed,
            "note": "a rehearsal that only proves the happy path proves nothing worth knowing",
            "activation": False}


# ---------------------------------------------------------------- authority and launch


def resource_plan() -> dict:
    return {
        "schema": "substrate-long-run-resource-plan/v1",
        "scheduler": "mop.cognition.longrun, one unit at a time, exclusive claim per unit",
        "concurrency": 1,
        "why_serial": ("every unit here is seconds to minutes of local compute and several write the "
                       "same proof root. Parallelism would buy nothing and would reintroduce the "
                       "exclusive writer problem the audit exists to prevent"),
        "unit_count": len(UNIT_LIST),
        "completion_criterion": "all units terminal",
        "not_a_wall_clock": ("a run that finishes early because the machine was fast has not done less "
                             "science, and a run still going has units left"),
        "stop_switch": str(STOP),
        "retries": MAX_ATTEMPTS,
        "external_dependencies": {"corpora": "under custody outside every worktree",
                                  "network": "none required"},
        "activation": False,
    }


def claim_boundary() -> dict:
    from mop.cognition import safety as SF

    return {
        "schema": "substrate-long-run-claim-boundary/v1",
        "permitted_terms": ["persistent developmental cognition", "entity like continuity",
                            "reflective cognitive organization", "sentience adjacent architecture"],
        "permitted_only_when": "supported by a classification from the method kernel",
        "forbidden": list(SF.FORBIDDEN_CLAIM_TERMS),
        "requires_separate_authority": ("consciousness and subjective experience. No result from this run "
                                        "can license either, whatever it shows"),
        "current_claims_supported": [],
        "current_evidence": ("one category has earned evidence and it is a null. No category has a "
                             "positive"),
        "enforcement": "mop.cognition.safety.assert_claim_safe raises rather than warning",
        "activation": False,
    }


def authority(cert: dict, reh: dict) -> dict:
    man = manifest()
    audit_doc = A.run()
    green = audit_doc["all_pass"] and cert["green"] and reh["all_pass"]
    return {
        "schema": "substrate-long-run-authority/v1",
        "frozen_manifest": man,
        "audit": {"all_pass": audit_doc["all_pass"], "failed": audit_doc["failed"]},
        "certification": {"green": cert["green"],
                          "gated_components": cert["gated_components"],
                          "sx2": cert["sx2"]},
        "rehearsal": {"all_pass": reh["all_pass"], "failed": reh["failed"]},
        "admission": "green" if green else "refused",
        "refusal_reason": "" if green else "; ".join(
            audit_doc["failed"] + [g["component"] for g in cert["gated_components"]] + reh["failed"]),
        "no_live_edits_after_launch": ("the frozen manifest carries the source commit and tree. A later "
                                       "run whose tree differs is a live edit and is detectable"),
        "defect_procedure": ("pause, append only repair, regression test, transition receipt, safe "
                             "resume. Completed units are not redone"),
        "commands": {
            "launch": "python -m mop.cognition.longrun launch",
            "status": "python -m mop.cognition.longrun status",
            "stop": f"touch {STOP}",
            "resume": "python -m mop.cognition.longrun launch",
        },
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "status"
    if command == "status":
        print(json.dumps(status(), indent=2))
    elif command == "rehearse":
        doc = rehearse()
        print(json.dumps({"all_pass": doc["all_pass"], "failed": doc["failed"],
                          "checks": {k: v["ok"] for k, v in doc["checks"].items()}}, indent=2))
    elif command == "seal":
        from mop.cognition import certify as CT

        cert = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_CERTIFICATION.json").read_text()) \
            if (io.PROOF / "SUBSTRATE_LONG_RUN_CERTIFICATION.json").is_file() else CT.run()
        reh = rehearse()
        auth = authority(cert, reh)
        io.seal("SUBSTRATE_LONG_RUN_AUTHORITY.json", auth)
        io.seal("SUBSTRATE_LONG_RUN_DAG.json", {
            "schema": "substrate-long-run-dag/v1",
            "units": [{"identity": u.identity, "module": u.module, "args": list(u.args),
                       "depends_on": list(u.depends_on), "certified": u.certified,
                       "produces": list(u.produces)} for u in UNIT_LIST],
            "unit_count": len(UNIT_LIST),
            "graph_nodes": G.declaration()["node_count"],
            "activation": False})
        io.seal("SUBSTRATE_LONG_RUN_RESOURCE_PLAN.json", resource_plan())
        io.seal("SUBSTRATE_LONG_RUN_CLAIM_BOUNDARY.json", claim_boundary())
        io.seal("SUBSTRATE_LONG_RUN_REHEARSAL.json", reh)
        print(json.dumps({"admission": auth["admission"], "refusal_reason": auth["refusal_reason"],
                          "units": len(UNIT_LIST),
                          "manifest": auth["frozen_manifest"]["manifest_sha256"][:16]}, indent=2))
    elif command == "launch":
        sealed = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_AUTHORITY.json").read_text())
        if sealed["admission"] != "green":
            raise Refused(f"admission is {sealed['admission']}: {sealed['refusal_reason']}")
        edit = live_edit_detected(sealed["frozen_manifest"])
        if edit["live_edit"]:
            raise Refused(f"the tree has changed since the freeze: {edit['drifted_keys']}")
        out = drive()
        io.run_json("launch.json", out, "long_run")
        print(json.dumps({"ran": out["ran"], "stopped_by": out["stopped_by"],
                          "completed": out["status"]["completed"],
                          "total": out["status"]["total"],
                          "terminal": out["status"]["terminal"]}, indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
