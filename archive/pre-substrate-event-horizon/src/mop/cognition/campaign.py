"""The Substrate campaign: a resumable stage graph for a run measured in days, not minutes.

This is not a second scheduler. The temporal supervisor is an E2 specific driver with bed names and shard
identities compiled into it, so it cannot run this program, and generalising it would mean editing the
module behind sealed evidence. What is reused is everything that can be: the process liveness and lock
primitives, the deterministic failure hold shape, the resume from disk contract, and the rule that a stage
opens only when its dependencies have receipts.

Three properties carry the long run.

Resume from disk. A stage is done because its receipt exists, not because a variable says so, so killing
the process at any point and starting it again loses at most one stage.

Deterministic failure holds. A stage that fails the same way twice under the same source commit stops
being retried. Without that a long run turns into an infinite loop that looks like progress.

Bounded, and it stops. The campaign ends when no stage is dependency ready, and the stop switch ends it
sooner. There is no path in which it keeps running because it has nothing better to do.

House style: no dashes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from mop.cognition import io
from mop.temporal.runs.supervisor import _pid_alive  # process liveness, unchanged and shared

PY = sys.executable
LOCKS = io.RUNS / "locks"
RECEIPTS = io.RUNS / "stages"
HOLDS = io.RUNS / "failure_holds"
LOGS = io.ROOT / "logs"
MAX_ATTEMPTS = 2


class Refused(RuntimeError):
    """A campaign action the driver will not take."""


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()  # proof artifacts that must exist afterwards
    repeatable: bool = False  # a stage that reruns every wave rather than once
    detail: str = ""


# The wave. Ordered by dependency, not by ambition.
STAGES: tuple[Stage, ...] = (
    Stage("declarations", "mop.cognition.deliverables", ("seal-modules",), deps=(),
          produces=("SUBSTRATE_WORKSPACE.json", "SUBSTRATE_PERSPECTIVE_SYSTEM.json",
                    "SUBSTRATE_MEMORY_SYSTEM.json", "SUBSTRATE_WORLD_MODEL.json",
                    "SUBSTRATE_SELF_MODEL.json", "SUBSTRATE_METACOGNITION.json",
                    "SUBSTRATE_PLASTICITY_SYSTEM.json", "SUBSTRATE_MODEL_BODY_INTERFACE.json",
                    "SUBSTRATE_RUNTIME.json"),
          repeatable=True, detail="seal every module declaration from the current tree"),
    Stage("tests", "mop.cognition.program", ("tests",), deps=("declarations",), repeatable=True,
          detail="run every declared test node and record the ledger"),
    Stage("experiments", "mop.cognition.experiments", ("seal",), deps=("tests",), repeatable=True,
          detail="admit the declared experiments and record each decision with its reason"),
    Stage("bed_screen", "mop.cognition.experiments", ("screen",), deps=("experiments",),
          repeatable=True, detail="screen every bed under custody against the declared SESOI"),
    Stage("sx1b", "mop.cognition.experiments", ("sx1b",), deps=("bed_screen",), repeatable=True,
          detail="the typed workspace successor, refused on power and recorded as refused"),
    Stage("sx5", "mop.cognition.experiments", ("sx5",), deps=("bed_screen",), repeatable=True,
          detail="self model calibration on the temporal program's sealed receipts"),
    Stage("deliverables", "mop.cognition.deliverables", ("write",), deps=("sx1b", "sx5"),
          produces=("SUBSTRATE_MASTER_AUTHORITY.json", "SUBSTRATE_STATE.json",
                    "SUBSTRATE_HYPOTHESIS_GRAPH.json", "SUBSTRATE_NULL_MAP.json",
                    "SUBSTRATE_PROGRESS_SCORECARD.json", "SUBSTRATE_NEXT_FRONTIER.json",
                    "SUBSTRATE_ARCHITECTURE.json", "SUBSTRATE_CAPABILITY_MAP.json",
                    "SUBSTRATE_TEMPORAL_CORE.json", "SUBSTRATE_CURRENT_ENTITY_SPEC.json"),
          repeatable=True, detail="regenerate every master artifact from the tree"),
    Stage("recompute", "mop.cognition.verify", ("recompute",), deps=("deliverables",),
          produces=("SUBSTRATE_INDEPENDENT_VERIFICATION.json",), repeatable=True,
          detail="recompute every sealed number by a second route from the sealed bytes"),
    Stage("mutate", "mop.cognition.verify", ("mutate",), deps=("recompute",),
          produces=("SUBSTRATE_MUTATION_REPORT.json",), repeatable=True,
          detail="break every declared guard on purpose and require the guard to notice"),
    Stage("classify", "mop.cognition.experiments", ("classify",), deps=("mutate",), repeatable=True,
          detail="turn a verified measurement into a recorded classification, and only then"),
    Stage("consolidate", "mop.cognition.deliverables", ("write",), deps=("classify",), repeatable=True,
          detail="regenerate the frontier and the value queue with the new classification in them"),
)

BY_NAME = {s.name: s for s in STAGES}


# ---------------------------------------------------------------- receipts and holds


def _receipt(stage: str) -> Path:
    return RECEIPTS / f"{stage}.json"


def done(stage: str) -> bool:
    path = _receipt(stage)
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return doc.get("ok") is True and doc.get("wave") == wave_index()


def wave_index() -> int:
    """Which pass over the graph this is. A repeatable stage is done for this wave, not forever."""
    path = io.RUNS / "campaign_wave.json"
    try:
        return int(json.loads(path.read_text())["wave"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return 1


def open_next_wave() -> int:
    nxt = wave_index() + 1
    io.run_json("campaign_wave.json", {"schema": "substrate-campaign-wave/v1", "wave": nxt,
                                       "source_commit": io.commit()})
    return nxt


def implementation_authority() -> dict:
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=io.ROOT,
                          capture_output=True, text=True)
    return {"source_commit": io.commit(),
            "source_tree_oid": tree.stdout.strip() if tree.returncode == 0 else ""}


def holds(stage: str | None = None) -> list[dict]:
    if not HOLDS.is_dir():
        return []
    authority, out = implementation_authority(), []
    for path in sorted(HOLDS.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("implementation_authority") != authority:
            continue  # a different implementation, so the hold is released
        if stage is None or doc.get("stage") == stage:
            out.append(doc)
    return out


def record_failure(stage: str, error: str) -> dict:
    authority = implementation_authority()
    prior = [h for h in holds(stage)]
    attempt = (prior[0]["attempts"] + 1) if prior else 1
    hold = {"schema": "substrate-campaign-failure-hold/v1", "stage": stage,
            "error": error[:400], "attempts": attempt,
            "state": "retry_allowed" if attempt < MAX_ATTEMPTS else "implementation_change_required",
            "implementation_authority": authority,
            "release_rule": "retry prohibited until source_commit or source_tree_oid changes",
            "activation": False}
    io.run_json(f"{stage}.json", hold, "failure_holds")
    return hold


def blocked(stage: str) -> bool:
    return any(h["state"] == "implementation_change_required" for h in holds(stage))


# ---------------------------------------------------------------- running


def ready() -> list[Stage]:
    return [s for s in STAGES
            if not done(s.name) and not blocked(s.name) and all(done(d) for d in s.deps)]


def run_stage(stage: Stage) -> dict:
    LOGS.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(io.ROOT / "src")}
    t0 = time.time()
    with open(LOGS / "substrate_campaign.log", "a") as log:
        log.write(f"\n=== {stage.name} wave {wave_index()} {time.strftime('%F %T')} ===\n")
        log.flush()
        result = subprocess.run([PY, "-m", stage.module, *stage.args], cwd=io.ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    missing = [a for a in stage.produces if not (io.PROOF / a).is_file()]
    ok = result.returncode == 0 and not missing
    receipt = {"schema": "substrate-campaign-stage/v1", "stage": stage.name, "wave": wave_index(),
               "ok": ok, "returncode": result.returncode, "missing_artifacts": missing,
               "wall_seconds": round(time.time() - t0, 1),
               "source_commit": io.commit(), "activation": False}
    io.run_json(f"{stage.name}.json", receipt, "stages")
    if not ok:
        record_failure(stage.name, f"exit {result.returncode}, missing {missing}")
    return receipt


def run(max_waves: int = 1) -> dict:
    """Drive the graph. Stops on the stop switch, on a hold, or when nothing is dependency ready."""
    waves = []
    for _ in range(max_waves):
        if io.STOP.exists():
            waves.append({"wave": wave_index(), "stopped": "stop switch"})
            break
        ran = []
        while True:
            if io.STOP.exists():
                break
            pending = ready()
            if not pending:
                break
            receipt = run_stage(pending[0])
            ran.append({"stage": pending[0].name, "ok": receipt["ok"]})
            if not receipt["ok"]:
                break
        blocked_stages = [s.name for s in STAGES if blocked(s.name)]
        unfinished = [s.name for s in STAGES if not done(s.name)]
        waves.append({"wave": wave_index(), "ran": ran, "blocked": blocked_stages,
                      "unfinished": unfinished,
                      "terminal": not unfinished})
        if unfinished:
            break  # a wave that could not complete is not followed by another one
        open_next_wave()
    return {"schema": "substrate-campaign-run/v1", "waves": waves,
            "stop_switch": str(io.STOP), "activation": False}


def status() -> dict:
    return {
        "schema": "substrate-campaign-status/v1",
        "wave": wave_index(),
        "stages": [{"stage": s.name, "done": done(s.name), "blocked": blocked(s.name),
                    "deps": list(s.deps), "detail": s.detail} for s in STAGES],
        "ready": [s.name for s in ready()],
        "holds": holds(),
        "stop_switch_active": io.STOP.exists(),
        "terminal": all(done(s.name) for s in STAGES),
    }


# ---------------------------------------------------------------- the long run
#
# Six waves toward section 22. Three of them are blocked on something that does not exist yet, and saying
# which three is the point of writing this down. A plan whose later waves quietly assume a component
# nobody has built is a plan that will look on schedule right up until it stops.

WAVES = (
    dict(wave="W1_close_the_loop",
         goal="every stage in the graph above reaches a receipt and the frontier is regenerated with the "
              "new classifications in it",
         entry="none, this is the wave the driver runs today",
         exit="all campaign stages done, SUBSTRATE_NEXT_FRONTIER.json regenerated",
         work=["seal declarations", "run declared tests", "admit experiments", "screen beds",
               "recompute independently", "mutate every guard", "classify what survived"],
         horizon="hours", blocked_on_absent_prerequisite=None),
    dict(wave="W2_perspective_families",
         goal="close the gap between nineteen declared perspective families and the six implemented",
         entry="W1 terminal",
         exit="every family that has an implementable verification method is implemented, and the ones "
              "that do not are listed with the reason",
         work=["implement the families whose verification is a computation rather than a judgement",
               "refuse the ones whose verification would be a human opinion, and say so",
               "extend the catalog declaration and its distinctness test"],
         horizon="hours", blocked_on_absent_prerequisite=None),
    dict(wave="W3_sx2_perspective_diversity",
         goal="test whether a heterogeneous perspective set beats the strongest single perspective at "
              "matched compute",
         entry="W2 terminal, and a bed that was not built to answer this question",
         exit="a classification, or a recorded refusal with the number that decides it",
         work=["treat the temporal factorial's seventy six cells as a real perspective set with per unit "
               "scores that already exist",
               "match compute honestly: selecting among k perspectives costs k times one perspective",
               "measure the oracle ceiling before spending anything on the fitted arm"],
         horizon="hours",
         blocked_on_absent_prerequisite=None,
         risk="the compute matched bar is severe. A set of seventy six cells must beat the best single "
              "cell given seventy six times its budget, and it may simply not"),
    dict(wave="W4_world_model_in_the_loop",
         goal="the world model changes a decision inside a cycle rather than scoring well beside it",
         entry="a bed where a prediction about the next state changes which action the arbiter selects",
         exit="a decision improvement measured inside the runtime, or a recorded finding that the bed "
              "does not contain such a decision",
         work=["wire the world model into the select and arbitrate stages",
               "find or build a bed with state dependent actions that nobody designed for this"],
         horizon="days",
         blocked_on_absent_prerequisite="a bed with state dependent decisions that this program did not "
                                        "design. The limited instrument verdict already exists precisely "
                                        "because prediction without decision improvement is not enough"),
    dict(wave="W5_model_body",
         goal="attach a real model body through the nine message kinds and run the same Substrate against "
              "more than one",
         entry="a body that implements the contract",
         exit="three conformance reports, one per declared body class",
         work=["implement an adapter for a compact specialist",
               "implement an adapter for a larger general model",
               "implement a tool dominant adapter",
               "run the same runtime against each and compare what has to live in weights"],
         horizon="days to weeks",
         blocked_on_absent_prerequisite="no model body is attached and none is implemented. The contract "
                                        "is testable and a partial body reports as partial, but there is "
                                        "nothing on the other end of it"),
    dict(wave="W6_entity_batteries_on_real_sessions",
         goal="run the thinking, continuity, unity and reflective batteries on sessions nobody wrote for "
              "the purpose",
         entry="W5, because a real session needs a body to produce it",
         exit="a classification per battery, or a recorded measurement boundary per battery",
         work=["replace the synthetic demo session with a recorded one",
               "rerun continuity with a transcript whose length nobody chose",
               "score thinking against all six declared alternatives"],
         horizon="weeks",
         blocked_on_absent_prerequisite="the same missing body as W5, plus a source of real sessions. "
                                        "Until then the continuity margin is a property of how much "
                                        "filler the generator added"),
)


def plan() -> dict:
    st = status()
    return {
        "schema": "substrate-campaign-plan/v1",
        "authority": "SUBSTRATE_MASTER_PLAN.md section 22",
        "driver": "python -m mop.cognition.campaign run <waves>",
        "stop_switch": str(io.STOP),
        "stage_graph": [{"stage": s.name, "deps": list(s.deps), "detail": s.detail} for s in STAGES],
        "current_status": {"wave": st["wave"], "ready": st["ready"], "terminal": st["terminal"]},
        "waves": list(WAVES),
        "waves_blocked_on_an_absent_prerequisite": [
            {"wave": w["wave"], "missing": w["blocked_on_absent_prerequisite"]}
            for w in WAVES if w["blocked_on_absent_prerequisite"]],
        "honest_horizon": ("W1 to W3 are runnable now and are measured in hours. W4 to W6 are not "
                           "schedulable, because each waits on a component or a data source that does "
                           "not exist yet. Half of this plan is a list of what would have to be true"),
        "termination": {
            "rule": ("the campaign ends when no stage is dependency ready, when every open hypothesis is "
                     "measurement blocked, or when the stop switch exists. It does not end by reaching a "
                     "target score"),
            "what_would_not_end_it": "running out of ideas is not a terminal condition, it is a queue "
                                     "that needs refilling with measurable questions",
        },
        "explicitly_not_planned": [
            "activation. No wave sets it true and no wave proposes a step that would",
            "unbounded goal creation. Every wave entry and exit is declared here in advance",
            "raising an evidence score without a classification",
            "lowering a SESOI to make a wave exit reachable",
        ],
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "status"
    if command == "status":
        print(json.dumps(status(), indent=2))
    elif command == "run":
        waves = int(argv[1]) if len(argv) > 1 else 1
        out = run(waves)
        print(json.dumps(out, indent=2))
    elif command == "plan":
        path = io.seal("SUBSTRATE_CAMPAIGN_PLAN.json", plan())
        doc = plan()
        print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                          "waves": [w["wave"] for w in doc["waves"]],
                          "blocked_on_something_that_does_not_exist":
                              [w["wave"] for w in doc["waves"] if w["blocked_on_absent_prerequisite"]],
                          "termination": doc["termination"]["rule"]}, indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
