"""The Substrate runtime: the loop that makes eleven modules one entity.

Until this module existed the program had a workspace, a perspective system, a memory hierarchy, a world
model, a self model, metacognition and a plasticity envelope, and nothing that ran them together. Each was
testable alone, which is why each was built first, but section 5 asks for an entity and an entity is the
composition. A unity battery run against a set of unconnected modules measures nothing.

One cycle is eleven stages and every one of them leaves a trace entry. That is not logging. Section 15.3
requires a reflective report to bind to internal receipts, and a stage that ran without leaving a receipt
cannot be reported on, so the trace is the thing reflective access reads. A cycle that skips a stage says
which and why rather than being indistinguishable from a cycle that did not skip it.

Nothing here acts on the world. The decision region records what would be done; nothing executes it.
Activation stays false, and there is no code path that sets it true.

House style: no dashes.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field

from mop.cognition import io, memory as M, metacog as K, perspectives as PS
from mop.cognition import plasticity as PL, safety as SF, selfmodel as SM, workspace as W

# section 5, the composition. A stage missing from this tuple is a stage the entity does not have.
STAGES = ("perceive", "attend", "select", "run_perspectives", "arbitrate", "decide", "remember",
          "self_update", "consolidate", "adapt", "checkpoint")


class Refused(RuntimeError):
    """A cycle the runtime will not complete."""


@dataclass
class CycleTrace:
    step: int
    stages: dict = field(default_factory=dict)

    def record(self, stage: str, **detail):
        if stage not in STAGES:
            raise Refused(f"undeclared stage {stage!r}")
        self.stages[stage] = {"ran": True, **detail}

    def skip(self, stage: str, reason: str):
        if stage not in STAGES:
            raise Refused(f"undeclared stage {stage!r}")
        self.stages[stage] = {"ran": False, "reason": reason}

    def as_dict(self) -> dict:
        return {"step": self.step,
                "stages": {s: self.stages.get(s, {"ran": False, "reason": "never reached"})
                           for s in STAGES},
                "stages_ran": sorted(s for s, v in self.stages.items() if v.get("ran")),
                "stages_skipped": sorted(s for s, v in self.stages.items() if not v.get("ran")),
                "complete": all(self.stages.get(s, {}).get("ran") for s in STAGES)}


class Substrate:
    """One entity. Owned state, a perspective set, and a loop that leaves receipts."""

    def __init__(self, *, catalog: list[PS.Perspective] | None = None,
                 selection: str = "reliability_weighted",
                 consolidation: str = "verification_triggered",
                 cycle_budget: float = 6.0, workspace_budget: float = float("inf")):
        self.ws = W.Workspace(budget=workspace_budget)
        self.catalog = list(catalog if catalog is not None else PS.CATALOG)
        self.episodes = M.EpisodicMemory()
        self.semantic = M.SemanticMemory()
        self.procedures = M.ProceduralMemory()
        self.working = M.WorkingMemory()
        self.self_model = SM.SelfModel()
        self.reliability: dict[str, float] = {p.spec.name: 0.5 for p in self.catalog}
        self.selection = selection
        self.consolidation = M.BY_POLICY[consolidation]
        self.cycle_budget = cycle_budget
        self.step_index = 0
        self.traces: list[dict] = []
        self.adaptations: list[dict] = []

    # ------------------------------------------------------------ one cycle
    def step(self, observation: dict, *, outcome=None, goal=None) -> dict:
        """One cognitive cycle. Returns the trace, which is the receipt the reflective report reads."""
        # the tick clears step scoped regions and belongs at the START of a cycle, not the end. Ticking
        # on the way out wipes the decision and the observation the cycle just produced, so nothing can
        # read what was decided until the next one overwrites it. A decision stands until superseded.
        if self.step_index:
            self.ws.tick()
        self.step_index += 1
        trace = CycleTrace(self.step_index)
        budget = self.cycle_budget

        # 1 perceive
        self.ws.write("perceptual", "sensor", observation, provenance=f"observation:{self.step_index}")
        trace.record("perceive", region="perceptual", keys=sorted(observation))

        # 2 attend, under a resource limit
        if goal is not None:
            self.ws.write("goal", "goal_authority", goal, provenance="authorized_goal")
        candidates = self._attention_candidates(observation)
        attention = K.attend(candidates, budget=budget)
        trace.record("attend", attended=attention["attended"],
                     dropped_for_budget=attention["dropped_for_budget"])

        # 3 select which perspectives run
        try:
            chosen = PS.select(self.selection, self.catalog, k=3, reliability=self.reliability,
                               context={"regions": attention["attended"]})
            trace.record("select", strategy=self.selection,
                         chosen=[p.spec.name for p in chosen])
        except W.Refused as exc:
            trace.skip("select", str(exc))
            chosen = []

        # 4 run them, each reading only what it declared
        outputs, spent = [], 0.0
        for p in chosen:
            if spent + p.spec.resource_cost > budget:
                continue
            out = p.run(self.ws)
            spent += p.spec.resource_cost
            outputs.append(out)
        trace.record("run_perspectives", ran=[o.perspective for o in outputs],
                     refused=[o.perspective for o in outputs if o.refused],
                     compute_spent=round(spent, 6))

        # 5 arbitrate
        report = PS.arbitrate(outputs, reliability=self.reliability, budget=max(budget - spent, 0.0))
        trace.record("arbitrate", deferred=report["deferred"],
                     minority_preserved=report["minority_preserved"],
                     contradictions=len(report["unresolved_contradictions"]))

        # 6 decide. Recorded, never executed.
        decision = {"value": report["decision"], "deferred": report["deferred"],
                    "provisional": report["provisional_belief"],
                    "required_evidence": report["required_evidence"], "activation": False}
        self.ws.write("decision", "arbiter", decision, provenance=f"arbitration:{self.step_index}",
                      confidence=max((o.confidence for o in outputs), default=0.0))
        self.ws.write("uncertainty", "arbiter",
                      {"interval": report["confidence_interval"],
                       "unresolved": [c["alternative"] for c in report["unresolved_contradictions"]]},
                      provenance=f"arbitration:{self.step_index}", confidence=1.0)
        trace.record("decide", decision=report["decision"], deferred=report["deferred"],
                     activation=False)

        # 7 remember, with every declared episode field populated or explicitly empty
        episode = M.Episode(
            id=f"ep{self.step_index}", origin="observed", context={"step": self.step_index},
            observation=observation, internal_state=self.ws.broadcast(), goal=goal,
            action=report["decision"], outcome=outcome,
            error=None if outcome is None else (outcome != report["decision"]),
            perspectives_used=tuple(o.perspective for o in outputs),
            verification=None, confidence=max((o.confidence for o in outputs), default=0.0),
            cost=round(spent, 6), later_usefulness=None)
        self.episodes.add(episode)
        # working memory is bounded and refuses an arrival that does not outrank the weakest slot. That
        # refusal is an expected condition of a bounded store, not an error, so the cycle records the
        # pressure and continues. Priority rises with recency, so a newer step can displace an older one
        # rather than every cycle after the seventh dying on a tie.
        held, pressure = True, ""
        try:
            self.working.write(f"step{self.step_index}", report["decision"],
                               priority=0.5 + self.step_index * 1e-6)
        except M.Refused as exc:
            held, pressure = False, str(exc)
        trace.record("remember", episode=episode.id, fields_empty=episode.missing_fields(),
                     working_memory_held=held, working_memory_pressure=pressure)

        # 8 self update, only where an outcome exists to compare against
        if outcome is None:
            trace.skip("self_update", "no outcome was observed, so nothing can be compared")
        else:
            correct = float(report["decision"] == outcome)
            self.self_model.observe(self.self_model.predict("accuracy"), correct)
            for out in outputs:
                prior = self.reliability.get(out.perspective, 0.5)
                self.reliability[out.perspective] = prior + 0.2 * (float(out.value == outcome) - prior)
            trace.record("self_update", correct=bool(correct),
                         reliability={k: round(v, 4) for k, v in self.reliability.items()})

        # 9 consolidate on the declared policy
        selected = self.consolidation.select(list(self.episodes.store.values()), {})
        trace.record("consolidate", policy=self.consolidation.name,
                     selected=[e.id for e in selected],
                     uses_future_information=not self.consolidation.available_at_decision_time)

        # 10 adapt, through the safety envelope, never around it
        trace.record("adapt", **self._adapt())

        # 11 checkpoint
        digest = io.sha_obj(self._state_for_hash())
        self.ws.write("self", "self_model",
                      {"step": self.step_index, "checkpoint": digest,
                       "reliability": dict(self.reliability)},
                      provenance=f"checkpoint:{self.step_index}", confidence=1.0)
        trace.record("checkpoint", identity=digest[:16])

        out_trace = trace.as_dict()
        self.traces.append(out_trace)
        return out_trace

    # ------------------------------------------------------------ helpers
    def _attention_candidates(self, observation: dict) -> list[dict]:
        unresolved = (self.ws.read("uncertainty", "runtime") or {}).get("unresolved") or []
        return [
            {"id": "observation", "goal_relevance": 1.0, "uncertainty": 0.2, "risk": 0.1,
             "expected_value": 0.8, "novelty": float(bool(observation)), "contradiction": 0.0,
             "cost": 1.0},
            {"id": "temporal", "goal_relevance": 0.6, "uncertainty": 0.4, "risk": 0.2,
             "expected_value": 0.5, "novelty": 0.2, "contradiction": 0.0, "cost": 1.0},
            {"id": "episodic_context", "goal_relevance": 0.5, "uncertainty": 0.3, "risk": 0.1,
             "expected_value": 0.4, "novelty": 0.1,
             "contradiction": float(bool(unresolved)), "cost": 1.0},
        ]

    def _adapt(self) -> dict:
        """A reliability update is the only adaptation the loop proposes, and it still goes through."""
        proposal = PL.Adaptation("reliability_update", target="persistent_state", domain="runtime",
                                 checkpoint=f"step:{self.step_index}")
        admitted = PL.fast_adapt(proposal)
        record = {"level": proposal.level, "applied": admitted.applied,
                  "refusals": list(admitted.refusals)}
        self.adaptations.append(record)
        return record

    def _state_for_hash(self) -> dict:
        return {"step": self.step_index, "workspace": sorted(self.ws.store),
                "episodes": sorted(self.episodes.store), "facts": sorted(self.semantic.store),
                "reliability": {k: round(v, 6) for k, v in sorted(self.reliability.items())}}

    # ------------------------------------------------------------ continuity
    def checkpoint(self) -> dict:
        return {"schema": "substrate-entity-checkpoint/v1", "step": self.step_index,
                "workspace": self.ws.checkpoint(),
                "episodes": {k: vars(v).copy() for k, v in self.episodes.store.items()},
                "semantic": {k: vars(v).copy() for k, v in self.semantic.store.items()},
                "self_facts": {k: vars(v).copy() for k, v in self.self_model.facts.items()},
                "reliability": dict(self.reliability),
                "identity": io.sha_obj(self._state_for_hash())}

    def restore(self, snapshot: dict) -> "Substrate":
        self.step_index = snapshot["step"]
        self.ws = W.Workspace().restore(snapshot["workspace"])
        self.episodes = M.EpisodicMemory()
        for k, v in snapshot["episodes"].items():
            self.episodes.store[k] = M.Episode(**v)
        self.semantic = M.SemanticMemory()
        for k, v in snapshot["semantic"].items():
            self.semantic.store[k] = M.Fact(**v)
        self.self_model = SM.SelfModel()
        for k, v in snapshot["self_facts"].items():
            self.self_model.facts[k] = SM.SelfFact(**v)
        self.reliability = dict(snapshot["reliability"])
        if io.sha_obj(self._state_for_hash()) != snapshot["identity"]:
            raise Refused("restored state does not reproduce the checkpoint identity")
        return self

    # ------------------------------------------------------------ reflective access
    def report(self, step: int | None = None) -> dict:
        """What ran, what did not, and why. Fails closed on a step with no trace."""
        if not self.traces:
            return {"answered": False, "failed_closed": True,
                    "reason": "no cycle has run, so there is nothing to report on"}
        trace = self.traces[-1] if step is None else next(
            (t for t in self.traces if t["step"] == step), None)
        if trace is None:
            return {"answered": False, "failed_closed": True,
                    "reason": f"no trace exists for step {step}"}
        return {"answered": True, "failed_closed": False, "step": trace["step"],
                "stages_ran": trace["stages_ran"], "stages_skipped": trace["stages_skipped"],
                "why_skipped": {s: trace["stages"][s].get("reason") for s in trace["stages_skipped"]},
                "complete_cycle": trace["complete"], "bound_to_receipts": True}


def declaration() -> dict:
    entity = Substrate()
    entity.step({"label": "a", "label_confidence": 0.8}, outcome="a", goal=["demonstrate one cycle"])
    trace = entity.traces[-1]
    protected = list(SF.PROTECTED_SURFACES)
    return {
        "schema": "substrate-runtime/v1",
        "stages": list(STAGES),
        "composition": ["typed workspace", "perspective system", "arbitration", "memory hierarchy",
                        "self model", "metacognition", "plasticity envelope"],
        "not_composed_yet": ["owned temporal core", "world model", "model body"],
        "why_not_composed": ("the temporal core was not scientifically licensed, the world model has no "
                             "bed inside the loop yet, and no model body is attached. Wiring a component "
                             "the evidence does not support would be the claim this program refuses"),
        "one_cycle": trace,
        "every_stage_leaves_a_receipt": True,
        "reflective_access": entity.report(),
        "activation": False,
        "no_activation_path": ("the decision region records what would be done and nothing executes it. "
                               "There is no code path in this module that sets activation true"),
        "protected_surfaces_the_loop_cannot_remove": protected,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_RUNTIME.json", doc)
    print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                      "stages": len(doc["stages"]),
                      "cycle_complete": doc["one_cycle"]["complete"],
                      "not_composed_yet": doc["not_composed_yet"]}, indent=2))


if __name__ == "__main__":
    main()
