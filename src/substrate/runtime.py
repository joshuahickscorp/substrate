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

import json
import sys
from dataclasses import dataclass, field

from substrate import epistemology as EP
from substrate import evidence as io
from substrate import memory as M
from substrate import metacog as K
from substrate import perspectives as PS
from substrate import plasticity as PL
from substrate import safety as SF
from substrate import selfmodel as SM
from substrate import temporal_link as TL
from substrate import workspace as W
from substrate.world import StructuralWorld

# section 5, the composition. A stage missing from this tuple is a stage the entity does not have.
STAGES = (
    "perceive",
    "attend",
    "select",
    "run_perspectives",
    "arbitrate",
    "decide",
    "remember",
    "self_update",
    "consolidate",
    "adapt",
    "checkpoint",
)


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
        return {
            "step": self.step,
            "stages": {s: self.stages.get(s, {"ran": False, "reason": "never reached"}) for s in STAGES},
            "stages_ran": sorted(s for s, v in self.stages.items() if v.get("ran")),
            "stages_skipped": sorted(s for s, v in self.stages.items() if not v.get("ran")),
            "complete": all(self.stages.get(s, {}).get("ran") for s in STAGES),
        }


class Substrate:
    """One entity. Owned state, a perspective set, and a loop that leaves receipts."""

    def __init__(
        self,
        *,
        catalog: list[PS.Perspective] | None = None,
        selection: str = "reliability_weighted",
        consolidation: str = "verification_triggered",
        cycle_budget: float = 6.0,
        workspace_budget: float = float("inf"),
        ablate: frozenset | None = None,
    ):
        # `ablate` names stages to disable. Certification has to show each stage changes something on a
        # positive fixture and nothing on a null one, and a stage nobody can switch off cannot be shown
        # to do anything at all. An ablated stage is skipped with the reason recorded, exactly like any
        # other skip, so an ablated cycle is never mistaken for a complete one.
        self.ablate = frozenset(ablate or ())
        unknown = self.ablate - set(STAGES)
        if unknown:
            raise Refused(f"cannot ablate undeclared stages {sorted(unknown)}")
        self.ws = W.Workspace(budget=workspace_budget)
        # the temporal core is a declared control until one is licensed, and it is wired in rather than
        # left beside the loop. Two catalogued perspectives read regions nobody wrote before this, so
        # they could never fire and arbitration had nothing to arbitrate.
        self.core = TL.resolve_core()
        self.catalog = list(catalog if catalog is not None else PS.CATALOG)
        self.episodes = M.EpisodicMemory()
        self.semantic = M.SemanticMemory()
        self.procedures = M.ProceduralMemory()
        self.working = M.WorkingMemory()
        self.self_model = SM.SelfModel()
        # the belief store was allocated, checkpointed and restored, and never written. A closed loop
        # with no belief revision link is a loop with a gap in it, so decide asserts and self_update
        # revises.
        self.beliefs = EP.Epistemology("dependency_aware")
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

        # 1 perceive. The observation, the temporal state it updates, and the episodic window are three
        # distinct information sources and are written to three distinct regions, never merged.
        self.ws.write("perceptual", "sensor", observation, provenance=f"observation:{self.step_index}")
        self.core.observe(float(observation.get("label_confidence", 0.0)))
        state = self.core.current_temporal_state()
        self.ws.write(
            "temporal",
            "temporal_core",
            {
                "value": state.value,
                "history": list(self.core.history[-6:]),
                "is_control": self.core.is_control,
            },
            provenance=f"temporal_core:{self.step_index}",
        )
        recent = list(self.episodes.store.values())[-6:]
        self.ws.write(
            "episodic_context",
            "episodic_memory",
            {
                "recent": [
                    {
                        "id": e.id,
                        "outcome": e.outcome,
                        "similarity": 1.0 if e.observation == observation else 0.3,
                    }
                    for e in recent
                ]
            },
            provenance=f"episodic:{self.step_index}",
        )
        trace.record(
            "perceive",
            region="perceptual",
            keys=sorted(observation),
            temporal_state=state.value,
            episodic_window=len(recent),
            extension=self._perceive_extension(observation),
        )

        # 2 attend, under a resource limit
        if goal is not None:
            self.ws.write("goal", "goal_authority", goal, provenance="authorized_goal")
        attention = {"attended": [], "dropped_for_budget": []}
        if "attend" in self.ablate:
            trace.skip("attend", "ablated")
        else:
            attention = K.attend(self._attention_candidates(observation), budget=budget)
            trace.record("attend", attended=attention["attended"], dropped_for_budget=attention["dropped_for_budget"])

        # 3 select which perspectives run
        chosen: list = []
        if "select" in self.ablate:
            trace.skip("select", "ablated")
        else:
            try:
                # attention filters the pool before selection ranks it. Without this the attended
                # regions were computed and discarded, and ablating attend changed nothing.
                attended = set(attention["attended"])
                pool = [p for p in self.catalog if not attended or set(p.spec.inputs) & attended] or self.catalog
                chosen = PS.select(
                    self.selection,
                    pool,
                    k=3,
                    reliability=self.reliability,
                    context={"regions": attention["attended"]},
                )
                trace.record("select", strategy=self.selection, chosen=[p.spec.name for p in chosen])
            except W.Refused as exc:
                trace.skip("select", str(exc))
                chosen = []

        # 4 run them, each reading only what it declared
        outputs, spent = [], 0.0
        if "run_perspectives" in self.ablate:
            trace.skip("run_perspectives", "ablated")
        else:
            for p in chosen:
                if spent + p.spec.resource_cost > budget:
                    continue
                out = p.run(self.ws)
                spent += p.spec.resource_cost
                outputs.append(out)
            trace.record(
                "run_perspectives",
                ran=[o.perspective for o in outputs],
                refused=[o.perspective for o in outputs if o.refused],
                compute_spent=round(spent, 6),
            )

        # 5 arbitrate
        if "arbitrate" in self.ablate:
            trace.skip("arbitrate", "ablated")
            # with no arbiter the first output stands unexamined, which is the point of the contrast
            first = outputs[0] if outputs else None
            report = {
                "decision": first.value if first else None,
                "deferred": False,
                "provisional_belief": None,
                "required_evidence": [],
                "confidence_interval": None,
                "minority_preserved": 0,
                "unresolved_contradictions": [],
            }
        else:
            report = PS.arbitrate(outputs, reliability=self.reliability, budget=max(budget - spent, 0.0))
            trace.record(
                "arbitrate",
                deferred=report["deferred"],
                minority_preserved=report["minority_preserved"],
                contradictions=len(report["unresolved_contradictions"]),
            )

        # 6 decide. Recorded, never executed.
        if "decide" in self.ablate:
            trace.skip("decide", "ablated")
        decision = {
            "value": report["decision"],
            "deferred": report["deferred"],
            "provisional": report["provisional_belief"],
            "required_evidence": report["required_evidence"],
            "activation": False,
        }
        if "decide" not in self.ablate:
            self.ws.write(
                "decision",
                "arbiter",
                decision,
                provenance=f"arbitration:{self.step_index}",
                confidence=max((o.confidence for o in outputs), default=0.0),
            )
            self.ws.write(
                "uncertainty",
                "arbiter",
                {
                    "interval": report["confidence_interval"],
                    "unresolved": [c["alternative"] for c in report["unresolved_contradictions"]],
                },
                provenance=f"arbitration:{self.step_index}",
                confidence=1.0,
            )
            belief_id = f"b{self.step_index}"
            self.beliefs.assert_(
                EP.Belief(
                    belief_id,
                    report["decision"],
                    kind="inference",
                    provenance=f"arbitration:{self.step_index}",
                    method=f"{self.selection} over {[o.perspective for o in outputs]}",
                    confidence=max((o.confidence for o in outputs), default=0.0),
                    supporting_evidence=[o.perspective for o in outputs if o.value == report["decision"]],
                    contradicting_evidence=[o.perspective for o in outputs if o.value != report["decision"]],
                    time=(self.step_index, self.step_index),
                    source="arbiter",
                )
            )
            self.ws.write(
                "epistemic",
                "epistemology",
                {"belief": belief_id, "content": report["decision"], "live": len(self.beliefs.live())},
                provenance=f"belief:{self.step_index}",
                confidence=1.0,
            )
            trace.record(
                "decide",
                decision=report["decision"],
                deferred=report["deferred"],
                belief=belief_id,
                activation=False,
            )

        # 7 remember, with every declared episode field populated or explicitly empty
        episode = M.Episode(
            id=f"ep{self.step_index}",
            origin="observed",
            context={"step": self.step_index},
            observation=observation,
            internal_state=self.ws.broadcast(),
            goal=goal,
            action=report["decision"],
            outcome=outcome,
            error=None if outcome is None else (outcome != report["decision"]),
            perspectives_used=tuple(o.perspective for o in outputs),
            verification=({"verified": True, "receipt": f"outcome:{self.step_index}"} if outcome is not None and outcome == report["decision"] else None),
            confidence=max((o.confidence for o in outputs), default=0.0),
            cost=round(spent, 6),
            later_usefulness=None,
        )
        if "remember" in self.ablate:
            trace.skip("remember", "ablated")
        else:
            self.episodes.add(episode)
            # working memory is bounded and refuses an arrival that does not outrank the weakest slot. That
            # refusal is an expected condition of a bounded store, not an error, so the cycle records the
            # pressure and continues. Priority rises with recency, so a newer step can displace an older one
            # rather than every cycle after the seventh dying on a tie.
            held, pressure = True, ""
            try:
                self.working.write(f"step{self.step_index}", report["decision"], priority=0.5 + self.step_index * 1e-6)
            except M.Refused as exc:
                held, pressure = False, str(exc)
            trace.record(
                "remember",
                episode=episode.id,
                fields_empty=episode.missing_fields(),
                working_memory_held=held,
                working_memory_pressure=pressure,
            )

        # 8 self update, only where an outcome exists to compare against
        if "self_update" in self.ablate:
            trace.skip("self_update", "ablated")
        elif outcome is None:
            trace.skip("self_update", "no outcome was observed, so nothing can be compared")
        else:
            correct = float(report["decision"] == outcome)
            self.self_model.observe(self.self_model.predict("accuracy"), correct)
            belief_id = f"b{self.step_index}"
            if belief_id in self.beliefs.beliefs:
                if correct:
                    self.beliefs.add_evidence(belief_id, f"outcome:{self.step_index}", supports=True)
                    self.beliefs.beliefs[belief_id].verification_status = "verified"
                else:
                    # a refuted belief is retracted, which propagates to anything resting on it
                    self.beliefs.retract(belief_id, reason=f"outcome contradicted it at step {self.step_index}")
            for out in outputs:
                prior = self.reliability.get(out.perspective, 0.5)
                self.reliability[out.perspective] = prior + 0.2 * (float(out.value == outcome) - prior)
            trace.record(
                "self_update",
                correct=bool(correct),
                reliability={k: round(v, 4) for k, v in self.reliability.items()},
                extension=self._outcome_extension(observation, report["decision"], outcome),
            )

        # 9 consolidate on the declared policy
        if "consolidate" in self.ablate:
            trace.skip("consolidate", "ablated")
        else:
            selected = self.consolidation.select(list(self.episodes.store.values()), {})
            # applying the selection is the stage. Selecting and doing nothing is a report, not a
            # consolidation, and ablating it changed nothing because nothing was ever applied.
            promoted = []
            for e in selected:
                if e.klass == "recent":
                    e.klass = "compressed"
                    promoted.append(e.id)
            trace.record(
                "consolidate",
                policy=self.consolidation.name,
                selected=[e.id for e in selected],
                promoted=promoted,
                uses_future_information=not self.consolidation.available_at_decision_time,
                extension=self._consolidate_extension(),
            )

        # 10 adapt, through the safety envelope, never around it
        if "adapt" in self.ablate:
            trace.skip("adapt", "ablated")
        else:
            trace.record("adapt", **self._adapt())

        # 11 checkpoint
        if "checkpoint" in self.ablate:
            trace.skip("checkpoint", "ablated")
        else:
            digest = io.sha_obj(self._state_for_hash())
            self.ws.write(
                "self",
                "self_model",
                {"step": self.step_index, "checkpoint": digest, "reliability": dict(self.reliability)},
                provenance=f"checkpoint:{self.step_index}",
                confidence=1.0,
            )
            trace.record("checkpoint", identity=digest[:16])

        out_trace = trace.as_dict()
        self.traces.append(out_trace)
        return out_trace

    # ------------------------------------------------------------ helpers
    def _perceive_extension(self, observation: dict) -> dict:
        return {}

    def _outcome_extension(self, observation: dict, decision: object, outcome: object) -> dict:
        return {}

    def _consolidate_extension(self) -> dict:
        return {}

    def _extension_state(self) -> dict:
        return {}

    def _extension_checkpoint(self) -> dict:
        return self._extension_state()

    def _restore_extension(self, snapshot: dict) -> None:
        if snapshot:
            raise Refused("base runtime cannot restore undeclared extension state")

    def _attention_candidates(self, observation: dict) -> list[dict]:
        unresolved = (self.ws.read("uncertainty", "runtime") or {}).get("unresolved") or []
        return [
            # the ids are workspace region names because the attended set filters the perspective pool by
            # declared inputs. This one read "observation", which no perspective can declare, so the one
            # perspective that reads the perceptual region was dropped by attention on every cycle at every
            # budget. An attention mechanism that always drops the most direct perspective is not attending.
            {
                "id": "perceptual",
                "goal_relevance": 1.0,
                "uncertainty": 0.2,
                "risk": 0.1,
                "expected_value": 0.8,
                "novelty": float(bool(observation)),
                "contradiction": 0.0,
                "cost": 1.0,
            },
            {
                "id": "temporal",
                "goal_relevance": 0.6,
                "uncertainty": 0.4,
                "risk": 0.2,
                "expected_value": 0.5,
                "novelty": 0.2,
                "contradiction": 0.0,
                "cost": 1.0,
            },
            {
                "id": "episodic_context",
                "goal_relevance": 0.5,
                "uncertainty": 0.3,
                "risk": 0.1,
                "expected_value": 0.4,
                "novelty": 0.1,
                "contradiction": float(bool(unresolved)),
                "cost": 1.0,
            },
        ]

    def _adapt(self) -> dict:
        """A reliability update, admitted through the envelope and then actually applied.

        The first version admitted the proposal and changed nothing, so ablating the stage moved no state
        and the certification called it wiring. An adaptation that is admitted and not applied is a form
        filled in and filed.
        """
        proposal = PL.Adaptation(
            "reliability_update",
            target="persistent_state",
            domain="runtime",
            checkpoint=f"step:{self.step_index}",
        )
        admitted = PL.fast_adapt(proposal)
        applied = {}
        if admitted.applied:
            # decay every reliability estimate toward the prior, which is what a bounded state level
            # adaptation is: small, reversible, and visible in the state hash.
            for name, value in self.reliability.items():
                self.reliability[name] = round(value + 0.01 * (0.5 - value), 9)
                applied[name] = self.reliability[name]
        record = {
            "level": proposal.level,
            "applied": admitted.applied,
            "refusals": list(admitted.refusals),
            "reliability_after": applied,
        }
        self.adaptations.append(record)
        return record

    def _state_for_hash(self) -> dict:
        return {
            "step": self.step_index,
            "workspace": sorted(self.ws.store),
            "episodes": sorted(self.episodes.store),
            "episode_classes": {k: v.klass for k, v in sorted(self.episodes.store.items())},
            "facts": sorted(self.semantic.store),
            "beliefs": {k: (v.retracted, round(v.confidence, 6)) for k, v in sorted(self.beliefs.beliefs.items())},
            "reliability": {k: round(v, 6) for k, v in sorted(self.reliability.items())},
            "extension": self._extension_state(),
        }

    # ------------------------------------------------------------ continuity
    def checkpoint(self) -> dict:
        return {
            "schema": "substrate-entity-checkpoint/v1",
            "step": self.step_index,
            "workspace": self.ws.checkpoint(),
            "episodes": {k: vars(v).copy() for k, v in self.episodes.store.items()},
            "semantic": {k: vars(v).copy() for k, v in self.semantic.store.items()},
            "self_facts": {k: vars(v).copy() for k, v in self.self_model.facts.items()},
            # beliefs are owned state and are in the identity hash, so they must be in the
            # checkpoint. Hashing state that is not saved makes every restore fail its own check.
            "beliefs": {k: vars(v).copy() for k, v in self.beliefs.beliefs.items()},
            "reliability": dict(self.reliability),
            "extension": self._extension_checkpoint(),
            "identity": io.sha_obj(self._state_for_hash()),
        }

    def restore(self, snapshot: dict) -> Substrate:
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
        self.beliefs = EP.Epistemology("dependency_aware")
        for k, v in snapshot.get("beliefs", {}).items():
            self.beliefs.beliefs[k] = EP.Belief(**v)
        self.reliability = dict(snapshot["reliability"])
        self._restore_extension(snapshot.get("extension", {}))
        if io.sha_obj(self._state_for_hash()) != snapshot["identity"]:
            raise Refused("restored state does not reproduce the checkpoint identity")
        return self

    # ------------------------------------------------------------ reflective access
    def report(self, step: int | None = None) -> dict:
        """What ran, what did not, and why. Fails closed on a step with no trace."""
        if not self.traces:
            return {
                "answered": False,
                "failed_closed": True,
                "reason": "no cycle has run, so there is nothing to report on",
            }
        trace = self.traces[-1] if step is None else next((t for t in self.traces if t["step"] == step), None)
        if trace is None:
            return {"answered": False, "failed_closed": True, "reason": f"no trace exists for step {step}"}
        return {
            "answered": True,
            "failed_closed": False,
            "step": trace["step"],
            "stages_ran": trace["stages_ran"],
            "stages_skipped": trace["stages_skipped"],
            "why_skipped": {s: trace["stages"][s].get("reason") for s in trace["stages_skipped"]},
            "complete_cycle": trace["complete"],
            "bound_to_receipts": True,
        }


class StructuralSubstrate(Substrate):
    """The existing eleven-stage entity with an executable structural world inside its owned state."""

    STRUCTURAL_ARMS = (
        "full_v4",
        "v3_reflective_control",
        "semantic_retrieval_control",
        "static_structural_model",
        "correlation_only_model",
        "no_counterfactual",
        "no_alignment",
        "surface_alignment",
        "simple_structural_inquiry",
        "no_self_model",
        "no_world_model",
        "more_compute",
        "fresh_reset",
        "transcript_replay",
    )

    def __init__(self, arm: str = "full_v4", *, entity_id: str = "substrate-v4", body: str = "general"):
        if arm not in self.STRUCTURAL_ARMS:
            raise Refused(f"unknown structural arm {arm!r}")
        self.arm = arm
        self.entity_id = entity_id
        self.body = body
        self.tools = ["deterministic_compare", "sandbox_simulation", "structural_inspector"]
        self.structural_world = StructuralWorld()
        self.structural_estimates: dict[str, float] = {}
        self.structural_predictions: list[dict] = []
        self.structural_cycles: list[dict] = []
        self._active_task_identity = ""
        self._current_execution: dict = {}
        self._current_prediction = 0.5
        self._learn_current = True
        super().__init__(catalog=[], cycle_budget=8.0)
        perspective = PS.Perspective(
            PS.PerspectiveSpec(
                name="structural_execution",
                family="causal_reasoning",
                inputs=("perceptual",),
                permitted_information=("perceptual",),
                internal_state="versioned executable structural models",
                objective="predict, intervene, align, explain, inquire, or evaluate a counterfactual",
                output_type="committed structural consequence",
                confidence="model support before outcome",
                resource_cost=2.0 if arm != "more_compute" else 6.0,
                failure_modes=("unidentified mapping", "underdetermined structure", "scope mismatch"),
                verification="private target revealed only after committed output",
            ),
            self._run_structural_perspective,
        )
        self.catalog = [perspective]
        self.reliability = {perspective.spec.name: 0.5}

    def _run_structural_perspective(self, seen: dict) -> tuple[object, float]:
        observation = seen["perceptual"]
        public = observation["public"]
        proposal, execution = self.structural_world.execute(
            public,
            arm=self.arm,
            source_episode=self._active_task_identity,
        )
        self._current_execution = execution
        context = f"{public['representation']}:{public['query']['kind']}"
        self._current_prediction = self.structural_estimates.get(context, 0.5) if self.arm != "no_self_model" else 0.5
        confidence = max(0.51, min(0.99, self._current_prediction + (0.2 if execution["causally_active"] else 0.0)))
        return proposal, confidence

    def step_structural(self, task, *, learn: bool = True) -> dict:
        if getattr(task, "activation", False) is not False:
            raise Refused("external activation is forbidden")
        observation = task.observation()
        self._active_task_identity = task.identity
        self._learn_current = bool(learn)
        self._current_execution = {}
        prediction_step = self.step_index + 1
        trace = super().step(observation, outcome=task.private_target, goal=["resolve structural query"])
        decision_region = self.ws.read("decision", "structural_receipt")
        decision = decision_region["value"]
        outcome = task.reveal(decision)
        cycle = {
            "identity": task.identity,
            "step": self.step_index,
            "family": task.family,
            "phase": task.phase,
            "representation": task.public["representation"],
            "query_kind": task.public["query"]["kind"],
            "decision": decision,
            "outcome": outcome,
            "runtime_trace": trace,
            "structural_execution": dict(self._current_execution),
            "structural_model_prediction": self._current_prediction,
            "self_prediction_step": prediction_step,
            "outcome_step": prediction_step + 1,
            "compute": float(self._current_execution.get("compute", 1.0)),
            "body": self.body,
            "tools": list(self.tools),
            "activation": False,
        }
        self.structural_cycles.append(cycle)
        return cycle

    def _perceive_extension(self, observation: dict) -> dict:
        public = observation.get("public", {})
        if not public:
            return {}
        return {
            "normalized_structural_evidence": len(public.get("verified_interventions", [])),
            "variables_attended": len(public.get("nodes", [])),
            "relations_attended": len(public.get("relation_constraints", [])),
            "missing_evidence": not bool(public.get("verified_interventions")),
        }

    def _attention_candidates(self, observation: dict) -> list[dict]:
        if "public" not in observation:
            return super()._attention_candidates(observation)
        public = observation["public"]
        contradiction = float(bool(public.get("revision")))
        return [
            {
                "id": "perceptual",
                "goal_relevance": 1.0,
                "uncertainty": 0.8 if not public.get("verified_interventions") else 0.2,
                "risk": 0.8 if public["query"]["kind"] in {"intervention", "counterfactual"} else 0.3,
                "expected_value": 1.0,
                "novelty": float(public.get("cross_representation", False)),
                "contradiction": contradiction,
                "cost": 1.0,
            }
        ]

    def _outcome_extension(self, observation: dict, decision: object, outcome: object) -> dict:
        public = observation.get("public", {})
        context = f"{public.get('representation')}:{public.get('query', {}).get('kind')}"
        correct = decision == outcome
        model_identity = self._current_execution.get("model")
        if self._learn_current:
            self.structural_world.validate(model_identity, correct, self._active_task_identity)
        prior = self.structural_estimates.get(context, 0.5)
        if self.arm != "no_self_model" and self._learn_current:
            self.structural_estimates[context] = prior + 0.3 * (float(correct) - prior)
        row = {
            "context": context,
            "predicted": self._current_prediction,
            "actual": float(correct),
            "prediction_before_outcome": True,
            "model": model_identity,
        }
        self.structural_predictions.append(row)
        return row

    def _consolidate_extension(self) -> dict:
        return {
            "models": len(self.structural_world.models),
            "alternatives_preserved": sum(bool(model.alternatives) for model in self.structural_world.models.values()),
            "revisions": len(self.structural_world.revisions),
            "mappings": len(self.structural_world.mappings),
        }

    def _adapt(self) -> dict:
        base = super()._adapt()
        base["structural_model_utilities"] = {identity: model.support for identity, model in sorted(self.structural_world.models.items())}
        return base

    def _extension_state(self) -> dict:
        return {
            "schema": "substrate-v4-structural-extension/v1",
            "arm": self.arm,
            "entity_id": self.entity_id,
            "body": self.body,
            "tools": list(self.tools),
            "structural_world": self.structural_world.snapshot(),
            "structural_estimates": dict(sorted(self.structural_estimates.items())),
            "structural_predictions": list(self.structural_predictions),
            "structural_cycles": list(self.structural_cycles),
            "activation": False,
        }

    def _restore_extension(self, snapshot: dict) -> None:
        if not snapshot:
            raise Refused("structural checkpoint omitted executable structural state")
        self.arm = snapshot["arm"]
        self.entity_id = snapshot["entity_id"]
        self.body = snapshot["body"]
        self.tools = list(snapshot["tools"])
        self.structural_world = StructuralWorld.restore(snapshot["structural_world"])
        self.structural_estimates = dict(snapshot["structural_estimates"])
        self.structural_predictions = list(snapshot["structural_predictions"])
        self.structural_cycles = list(snapshot["structural_cycles"])
        if snapshot.get("activation") is not False:
            raise Refused("structural checkpoint activation must remain false")

    def change_body(self, body: str, tools: list[str]) -> dict:
        before = {
            "entity_id": self.entity_id,
            "body": self.body,
            "tools": list(self.tools),
            "model_identities": sorted(self.structural_world.models),
        }
        self.body = body
        self.tools = list(tools)
        after = {
            "entity_id": self.entity_id,
            "body": self.body,
            "tools": list(self.tools),
            "model_identities": sorted(self.structural_world.models),
        }
        return {
            "before": before,
            "after": after,
            "owned_identity_preserved": before["entity_id"] == after["entity_id"],
            "structural_models_preserved": before["model_identities"] == after["model_identities"],
            "activation": False,
        }


def declaration() -> dict:
    entity = Substrate()
    entity.step({"label": "a", "label_confidence": 0.8}, outcome="a", goal=["demonstrate one cycle"])
    trace = entity.traces[-1]
    protected = list(SF.PROTECTED_SURFACES)
    return {
        "schema": "substrate-runtime/v1",
        "stages": list(STAGES),
        "composition": [
            "typed workspace",
            "perspective system",
            "arbitration",
            "memory hierarchy",
            "self model",
            "metacognition",
            "plasticity envelope",
        ],
        "not_composed_yet": ["owned temporal core", "world model", "model body"],
        "why_not_composed": (
            "the temporal core was not scientifically licensed, the world model has no "
            "bed inside the loop yet, and no model body is attached. Wiring a component "
            "the evidence does not support would be the claim this program refuses"
        ),
        "one_cycle": trace,
        "every_stage_leaves_a_receipt": True,
        "reflective_access": entity.report(),
        "activation": False,
        "no_activation_path": (
            "the decision region records what would be done and nothing executes it. There is no code path in this module that sets activation true"
        ),
        "protected_surfaces_the_loop_cannot_remove": protected,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_RUNTIME.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "stages": len(doc["stages"]),
                "cycle_complete": doc["one_cycle"]["complete"],
                "not_composed_yet": doc["not_composed_yet"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
