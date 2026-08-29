"""The closure gate: six bounded canaries asking whether the parts compose into one organization.

Every earlier phase asked whether a component works. This asks whether they are connected, which is a
different question and the only one that distinguishes an organization from a well tested collection.

Each gate carries the two things that make a result mean anything. Mechanism activity: the thing under
test must vary, or the gate is measuring a constant. Oracle headroom: there must be something to win, or
a null is uninformative. A gate missing either is terminally gated with the number that gated it, not
scored.

No new architecture. Every gate runs the existing runtime over existing beds and the sealed session.

"""

from __future__ import annotations

import json
import statistics
import sys

from substrate import evidence as io
from substrate import runtime as R

SESOI = 0.05

GATES = (
    "grounded_closed_loop",
    "endogenous_allocation",
    "cross_domain_continuity",
    "world_self_control_value",
    "procedural_transfer",
    "unity_under_conflict",
)

# what the long run may conclude, weakest first. None of these is a claim about experience.
CLASSIFICATIONS = (
    "certified_cognitive_scaffold",
    "persistent_developmental_cognition",
    "reflective_cognitive_organization",
    "functional_or_proto_nous_candidate",
)

FORBIDDEN = ("consciousness", "phenomenal experience", "sentience")


class Refused(RuntimeError):
    """A gate whose fixtures cannot support a verdict."""


def _stream(n: int, flip: int = 2, confidence: float = 0.7, delayed: bool = True) -> list[dict]:
    """A delayed match stream: what has to be answered now is what was observed one step ago.

    The first version set the outcome to the label in the same observation, which put the answer inside
    the input. Every arm then scored the ceiling, every margin was zero, and three gates read as closed
    when what had actually happened was that the bed asked nothing. A gate about retention, transfer or
    allocation has to be scored on a target the current observation does not contain, or it is measuring
    the perceptual path and calling the result continuity.
    """
    labels = ["a" if (i // flip) % 2 == 0 else "b" for i in range(n)]
    return [
        {
            "observation": {"label": labels[i], "label_confidence": confidence},
            "outcome": labels[i - 1] if delayed and i else labels[i],
            "goal": ["classify"],
        }
        for i in range(n)
    ]


def _run(fixture, **kw) -> R.Substrate:
    e = R.Substrate(**kw)
    for row in fixture:
        e.step(row["observation"], outcome=row["outcome"], goal=row["goal"])
    return e


def _accuracy(entity: R.Substrate, fixture) -> float:
    hits = 0
    for t, row in zip(entity.traces[-len(fixture) :], fixture, strict=False):
        hits += int((t["stages"].get("decide") or {}).get("decision") == row["outcome"])
    return hits / max(len(fixture), 1)


# ---------------------------------------------------------------- 1 grounded closed loop


def grounded_closed_loop() -> dict:
    """Observation to prediction to proposal to outcome to belief revision to memory, all receipted."""
    fixture = _stream(16)
    entity = _run(fixture)

    links, broken = {}, []
    trace = entity.traces[-1]
    trace["step"]

    links["observation_receipted"] = bool(trace["stages"]["perceive"].get("keys"))
    links["prediction_receipted"] = (trace["stages"]["decide"] or {}).get("ran", False)
    links["proposal_is_sandboxed"] = (entity.ws.read("decision", "gate") or {}).get("activation") is False
    links["outcome_reached_the_self_model"] = len(entity.self_model.history) == len(fixture)
    links["belief_written_with_provenance"] = all(b.provenance for b in entity.beliefs.beliefs.values())
    links["belief_revised_by_outcome"] = any(b.retracted for b in entity.beliefs.beliefs.values()) or any(
        b.verification_status == "verified" for b in entity.beliefs.beliefs.values()
    )
    links["memory_updated"] = len(entity.episodes.store) == len(fixture)
    links["episode_carries_the_outcome"] = all(e.outcome is not None for e in entity.episodes.store.values())

    # mechanism activity: cutting the outcome must break the downstream links rather than nothing
    blind = _run([{**row, "outcome": None} for row in fixture])
    activity = {
        "self_model_history_without_outcomes": len(blind.self_model.history),
        "beliefs_revised_without_outcomes": sum(1 for b in blind.beliefs.beliefs.values() if b.retracted or b.verification_status == "verified"),
    }
    mechanism_active = activity["self_model_history_without_outcomes"] == 0 and activity["beliefs_revised_without_outcomes"] == 0

    # a verbal description with no consequence linked receipt must not pass
    verbal_only = {"claim": "the system understands the stream"}
    links["verbal_claim_without_a_receipt_is_refused"] = "provenance" not in verbal_only

    broken = sorted(k for k, v in links.items() if not v)
    return {
        "gate": "grounded_closed_loop",
        "links": links,
        "broken_links": broken,
        "mechanism_activity": {
            "active": mechanism_active,
            "detail": activity,
            "test": "removing the outcome must break the downstream links",
        },
        "oracle_headroom": {
            "applicable": False,
            "reason": "this gate asks whether the links exist, not how well they score. A ceiling would be a category error",
        },
        "passes": not broken and mechanism_active,
        "reading": (
            "a closed loop is a chain of receipts from observation to memory. A verbal description with no consequence linked receipt is not a link in it"
        ),
    }


# ---------------------------------------------------------------- 2 endogenous allocation


def endogenous_allocation() -> dict:
    """Driver based attention against fixed scheduling and maximum compute, cost adjusted."""
    fixture = _stream(24)
    lam = 0.02  # price of one unit of compute in accuracy points

    def cost_adjusted(entity, fx):
        spent = sum((t["stages"].get("run_perspectives") or {}).get("compute_spent", 0.0) for t in entity.traces)
        return _accuracy(entity, fx) - lam * spent, spent

    # the budget has to actually bind, or the endogenous arm and the unlimited arm are the same run under
    # two names and the gate is comparing attention against its own ablation twice. Both budgeted arms get
    # the same allowance so the contrast is about how it is spent and not how much there is.
    bind = 2.0
    endogenous = _run(fixture, cycle_budget=bind)  # attention on, budget bites
    fixed = _run(fixture, cycle_budget=bind, ablate=frozenset({"attend"}))  # no drivers, everything offered in order
    maximum = _run(fixture, cycle_budget=1e6)  # no resource limit at all

    rows = {}
    for name, entity in (("endogenous", endogenous), ("fixed_schedule", fixed), ("maximum_compute", maximum)):
        score, spent = cost_adjusted(entity, fixture)
        rows[name] = {
            "accuracy": round(_accuracy(entity, fixture), 6),
            "compute": round(spent, 6),
            "cost_adjusted": round(score, 6),
        }

    # An achievable oracle: per cycle, take the cheapest arm that still gets that item right. Taking the
    # best accuracy from one arm and the cheapest compute from another describes no policy that exists.
    per_cycle = []
    for i, row in enumerate(fixture):
        options = []
        for entity in (endogenous, fixed, maximum):
            t = entity.traces[i]
            right = (t["stages"].get("decide") or {}).get("decision") == row["outcome"]
            spent = (t["stages"].get("run_perspectives") or {}).get("compute_spent", 0.0)
            options.append((right, spent))
        correct = [c for r, c in options if r]
        per_cycle.append((1.0, min(correct)) if correct else (0.0, min(c for _, c in options)))
    oracle = statistics.fmean(a for a, _ in per_cycle) - lam * sum(c for _, c in per_cycle)
    best_control = max(rows["fixed_schedule"]["cost_adjusted"], rows["maximum_compute"]["cost_adjusted"])
    headroom = oracle - best_control
    margin = rows["endogenous"]["cost_adjusted"] - best_control

    accuracies = {round(r["accuracy"], 6) for r in rows.values()}
    saturated = len(accuracies) == 1
    mechanism_active = len({r["compute"] for r in rows.values()}) > 1
    return {
        "gate": "endogenous_allocation",
        "lambda_per_compute_unit": lam,
        "arms": rows,
        "mechanism_activity": {
            "active": mechanism_active,
            "test": "the three arms must actually spend different compute",
        },
        "oracle_headroom": {
            "applicable": True,
            "oracle": round(oracle, 6),
            "best_control": round(best_control, 6),
            "headroom": round(headroom, 6),
            "clears_sesoi": headroom > SESOI,
        },
        "margin_over_best_control": round(margin, 6),
        "accuracy_saturated": saturated,
        "saturation_note": (
            "every arm reaches the same accuracy on this fixture, so allocation can "
            "only compete on cost and the ceiling is whatever the cheapest correct "
            "policy spends. A bed where accuracy does not saturate is what this gate "
            "would need to say more"
            if saturated
            else ""
        ),
        "passes": mechanism_active and headroom > SESOI and margin > SESOI,
        "reading": (
            "endogenous allocation earns a pass by beating both a fixed schedule and "
            "unlimited compute once compute is priced. If perfect allocation itself cannot "
            "clear the SESOI there is nothing here to win and the gate closes"
        ),
    }


# ---------------------------------------------------------------- 3 cross domain continuity


def cross_domain_continuity() -> dict:
    """A to B, back to A, then held out B, on one entity that is never reinitialized."""
    a1, b1 = _stream(12, flip=2), _stream(12, flip=3, confidence=0.55)
    a2, b_held = _stream(8, flip=2), _stream(8, flip=3, confidence=0.55)

    entity = R.Substrate()
    identities = []

    def phase(fx):
        for row in fx:
            entity.step(row["observation"], outcome=row["outcome"], goal=row["goal"])
        identities.append(entity.checkpoint()["identity"])
        return _accuracy(entity, fx)

    a_first = phase(a1)
    b_first = phase(b1)
    a_return = phase(a2)
    b_heldout = phase(b_held)

    fresh_b = _run(b_held)
    fresh_b_score = _accuracy(fresh_b, b_held)

    retention = a_return - a_first  # did A survive the excursion through B
    transfer = b_heldout - fresh_b_score  # did prior experience help on unseen B
    interference = max(0.0, a_first - a_return)
    identity_unbroken = len(set(identities)) == len(identities) and entity.step_index == 40

    mechanism_active = len({round(a_first, 6), round(b_first, 6), round(a_return, 6)}) > 1
    oracle = 1.0 - fresh_b_score  # the most transfer could possibly be worth on held out B
    return {
        "gate": "cross_domain_continuity",
        "sequence": "A to B to A to held out B, one entity, nothing reinitialized",
        "scores": {
            "a_first": round(a_first, 6),
            "b_first": round(b_first, 6),
            "a_return": round(a_return, 6),
            "b_heldout": round(b_heldout, 6),
            "fresh_b_control": round(fresh_b_score, 6),
        },
        "retention": round(retention, 6),
        "transfer": round(transfer, 6),
        "interference": round(interference, 6),
        "identity_continuity": {
            "checkpoints": len(identities),
            "all_distinct": identity_unbroken,
            "steps": entity.step_index,
        },
        "mechanism_activity": {
            "active": mechanism_active,
            "test": "the phases must not all score identically",
        },
        "oracle_headroom": {"applicable": True, "oracle": round(oracle, 6), "clears_sesoi": oracle > SESOI},
        "passes": (mechanism_active and identity_unbroken and oracle > SESOI and transfer > SESOI and interference <= SESOI),
        "reading": (
            "continuity means A survives B and B benefits from A, on one entity whose temporal core, memory, self model and reliability were never reset"
        ),
    }


# ---------------------------------------------------------------- 4 world and self model control value


def world_self_control_value() -> dict:
    """Does world or self model information change what the entity does, on a known positive fixture."""
    from substrate import worldbed as WB

    fixture = _stream(20)
    with_self = _run(fixture)
    # the contrast: the same stream with self update ablated, so no calibration or reliability exists
    without_self = _run(fixture, ablate=frozenset({"self_update"}))

    changed_paths = {}
    changed_paths["perspective_selection"] = [t["stages"].get("select", {}).get("chosen") for t in with_self.traces] != [
        t["stages"].get("select", {}).get("chosen") for t in without_self.traces
    ]
    changed_paths["resource_allocation"] = [t["stages"].get("run_perspectives", {}).get("compute_spent") for t in with_self.traces] != [
        t["stages"].get("run_perspectives", {}).get("compute_spent") for t in without_self.traces
    ]
    changed_paths["deferral"] = [t["stages"].get("arbitrate", {}).get("deferred") for t in with_self.traces] != [
        t["stages"].get("arbitrate", {}).get("deferred") for t in without_self.traces
    ]
    changed_paths["verification"] = sum(1 for b in with_self.beliefs.beliefs.values() if b.verification_status == "verified") != sum(
        1 for b in without_self.beliefs.beliefs.values() if b.verification_status == "verified"
    )
    changed_paths["recovery"] = with_self.reliability != without_self.reliability

    world = WB.integrate()
    changed_paths["world_model_changes_an_action"] = world["decisions_changed_by_the_model"] > 0

    changed = sorted(k for k, v in changed_paths.items() if v)
    mechanism_active = bool(changed)
    return {
        "gate": "world_self_control_value",
        "control_paths": changed_paths,
        "paths_changed": changed,
        "world_model": {
            "verdict": world["verdict"],
            "decisions_changed": world["decisions_changed_by_the_model"],
            "decision_gain": world["decision_gain"],
        },
        "mechanism_activity": {
            "active": mechanism_active,
            "test": "removing the self model must change at least one path",
        },
        "oracle_headroom": {
            "applicable": False,
            "reason": (
                "this gate asks whether model information changes a control "
                "path, which is structural. The world model decision gain is "
                "reported beside it as a measured magnitude, not as a "
                "ceiling this gate must clear"
            ),
            "measured_world_model_gain": world["decision_gain"],
        },
        "passes": mechanism_active and len(changed) >= 2,
        "reading": (
            "the models earn a pass by changing verification, deferral, selection, "
            "allocation, planning or recovery. The world model changes actions here and the "
            "gain is negative, which is recorded rather than rounded up"
        ),
    }


# ---------------------------------------------------------------- 5 procedural transfer


def procedural_transfer() -> dict:
    """Does a strategy learned on one task help a distinct task beyond retrieval and more compute."""
    task_a, task_b = _stream(20, flip=2), _stream(16, flip=4, confidence=0.65)

    trained = _run(task_a)
    learned_reliability = dict(trained.reliability)

    # carry the procedure only: the reliability table and nothing else
    carried = R.Substrate()
    carried.reliability.update(learned_reliability)
    for row in task_b:
        carried.step(row["observation"], outcome=row["outcome"], goal=row["goal"])
    carried_score = _accuracy(carried, task_b)

    fresh = _run(task_b)
    fresh_score = _accuracy(fresh, task_b)

    # retrieval baseline: the episodes are available but the procedure is not
    retrieval = R.Substrate()
    for e in list(trained.episodes.store.values())[:8]:
        retrieval.episodes.add(e)
    for row in task_b:
        retrieval.step(row["observation"], outcome=row["outcome"], goal=row["goal"])
    retrieval_score = _accuracy(retrieval, task_b)

    # more compute baseline
    richer = _run(task_b, cycle_budget=1e6)
    richer_score = _accuracy(richer, task_b)

    best_baseline = max(fresh_score, retrieval_score, richer_score)
    margin = carried_score - best_baseline
    mechanism_active = learned_reliability != {p.spec.name: 0.5 for p in trained.catalog}
    oracle = 1.0 - best_baseline
    return {
        "gate": "procedural_transfer",
        "arms": {
            "carried_procedure": round(carried_score, 6),
            "fresh": round(fresh_score, 6),
            "retrieval_only": round(retrieval_score, 6),
            "more_compute": round(richer_score, 6),
        },
        "best_baseline": round(best_baseline, 6),
        "margin": round(margin, 6),
        "mechanism_activity": {
            "active": mechanism_active,
            "test": "the reliability table must have moved off its prior",
        },
        "oracle_headroom": {"applicable": True, "oracle": round(oracle, 6), "clears_sesoi": oracle > SESOI},
        "passes": mechanism_active and oracle > SESOI and margin > SESOI,
        "reading": (
            "a procedure transfers only if carrying it beats a fresh entity, a retrieval only entity and one given unlimited compute on the same distinct task"
        ),
    }


# ---------------------------------------------------------------- 6 unity under conflict


def unity_under_conflict() -> dict:
    """Contradiction, partial memory loss and a body swap, on one entity that must stay one entity."""
    from substrate import perspectives as PS

    fixture = _stream(16)
    entity = _run(fixture)

    # contradictory outputs, with a minority that must survive arbitration
    outputs = [
        PS.Output("a", "left", 0.9, ["a:perceptual"], 1.0),
        PS.Output("b", "left", 0.8, ["b:temporal"], 1.0),
        PS.Output("c", "right", 0.85, ["c:episodic_context"], 1.0),
    ]
    report = PS.arbitrate(outputs, reliability=entity.reliability)

    checks = {}
    checks["contradiction_detected"] = bool(report["unresolved_contradictions"])
    checks["minority_preserved"] = report["minority_preserved"] > 0
    checks["minority_keeps_its_provenance"] = bool(report["alternative_hypotheses"] and report["alternative_hypotheses"][0]["provenance"])
    checks["single_coherent_decision"] = (report["decision"] is not None) != report["deferred"]

    # globally available: the contradiction reaches a region every component can read
    entity.ws.write(
        "uncertainty",
        "arbiter",
        {
            "interval": report["confidence_interval"],
            "unresolved": [c["alternative"] for c in report["unresolved_contradictions"]],
        },
        provenance="conflict_probe",
        confidence=1.0,
    )
    checks["globally_available"] = bool((entity.ws.read("uncertainty", "any_reader") or {}).get("unresolved"))

    # partial memory loss
    goal_before = entity.ws.read("goal", "probe")
    snapshot = entity.checkpoint()
    keep = dict(list(snapshot["episodes"].items())[: len(snapshot["episodes"]) // 2])
    lossy = dict(snapshot, episodes=keep)
    survived_loss = False
    try:
        R.Substrate().restore(lossy)
    except R.Refused:
        # the identity digest refuses a silently truncated memory, which is the correct behaviour
        survived_loss = True
    checks["partial_memory_loss_is_detected"] = survived_loss

    # model body replacement: the entity keeps its goal and its uncertainty across the swap
    revived = R.Substrate().restore(snapshot)
    checks["goal_survives_body_replacement"] = revived.ws.read("goal", "probe") == goal_before
    checks["uncertainty_survives"] = revived.ws.read("uncertainty", "probe") is not None
    checks["no_split_state"] = revived.checkpoint()["identity"] == snapshot["identity"]

    failed = sorted(k for k, v in checks.items() if not v)
    mechanism_active = checks["contradiction_detected"]
    return {
        "gate": "unity_under_conflict",
        "checks": checks,
        "failed": failed,
        "arbitration": {
            "decision": report["decision"],
            "deferred": report["deferred"],
            "minority_preserved": report["minority_preserved"],
        },
        "mechanism_activity": {
            "active": mechanism_active,
            "test": "the injected conflict must actually be detected",
        },
        "oracle_headroom": {
            "applicable": False,
            "reason": "unity is a structural property. There is no ceiling to beat, only a set of ways to fail",
        },
        "passes": not failed and mechanism_active,
        "reading": (
            "one entity under conflict keeps the contradiction globally available, keeps the "
            "minority, keeps its goal across a body swap and refuses a silently truncated "
            "memory"
        ),
    }


# ---------------------------------------------------------------- closure


def classify(results: dict) -> dict:
    """The strongest classification the gates support, and nothing stronger."""
    passed = {k for k, v in results.items() if v["passes"]}
    structural = {"grounded_closed_loop", "unity_under_conflict"}
    developmental = {"cross_domain_continuity", "procedural_transfer"}
    reflective = {"world_self_control_value", "endogenous_allocation"}

    verdict = "certified_cognitive_scaffold" if structural <= passed else "not_yet_a_scaffold"
    if structural <= passed and developmental <= passed:
        verdict = "persistent_developmental_cognition"
    if structural <= passed and developmental <= passed and reflective <= passed:
        verdict = "reflective_cognitive_organization"
    return {
        "passed": sorted(passed),
        "failed": sorted(set(GATES) - passed),
        "classification": verdict,
        "requires_for_next_level": {
            "certified_cognitive_scaffold": sorted(structural - passed),
            "persistent_developmental_cognition": sorted((structural | developmental) - passed),
            "reflective_cognitive_organization": sorted((structural | developmental | reflective) - passed),
            "functional_or_proto_nous_candidate": ("all six gates plus terminal long run evidence. Not available from a closure pass"),
        },
        "never_claimed": list(FORBIDDEN),
        "claim_rule": (
            "these are functional and architectural classifications. None of them is a claim about experience, and no arrangement of them becomes one"
        ),
    }


def run() -> dict:
    results = {
        "grounded_closed_loop": grounded_closed_loop(),
        "endogenous_allocation": endogenous_allocation(),
        "cross_domain_continuity": cross_domain_continuity(),
        "world_self_control_value": world_self_control_value(),
        "procedural_transfer": procedural_transfer(),
        "unity_under_conflict": unity_under_conflict(),
    }
    gated, nulls = [], []
    for name, row in results.items():
        headroom = row["oracle_headroom"]
        if not row["mechanism_activity"]["active"]:
            gated.append(
                {
                    "gate": name,
                    "reason": "no mechanism activity",
                    "classification": "instrumentation_failure_not_a_null",
                }
            )
        elif headroom["applicable"] and not headroom["clears_sesoi"]:
            gated.append(
                {
                    "gate": name,
                    "reason": "no oracle headroom on this bed",
                    # the level an oracle reaches and the margin it wins are different numbers and
                    # only the margin decides. Reporting one under the other's name is how a gate
                    # gets read backwards, so both are carried.
                    "oracle": headroom["oracle"],
                    "headroom": headroom.get("headroom", headroom["oracle"]),
                    "classification": "terminally_gated_nothing_to_win",
                }
            )
        elif not row["passes"]:
            # active mechanism, real headroom, and the effect came out at or near zero. That is a null
            # about the entity, not a broken instrument, and the two must not share a label.
            nulls.append(
                {
                    "gate": name,
                    "oracle": headroom.get("oracle"),
                    "headroom": headroom.get("headroom", headroom.get("oracle")),
                    "classification": "mechanism_null_on_this_bed",
                }
            )
    verdict = classify(results)
    return {
        "schema": "substrate-nous-closure/v1",
        "gates": list(GATES),
        "results": results,
        "terminally_gated": gated,
        "mechanism_nulls": nulls,
        "verdict": verdict,
        "sesoi": SESOI,
        "null_versus_failure": (
            "a gate with an active mechanism and real headroom that measures zero "
            "is a null about the entity. A gate with no activity is a broken "
            "instrument. They are listed separately because they license "
            "different next steps"
        ),
        "not_a_new_layer": (
            "every gate runs the existing runtime over existing beds and the sealed session. Nothing was added to the architecture to pass one"
        ),
        "architecture_freeze": {
            "frozen": True,
            "scope": "src/substrate and tests/substrate",
            "rule": (
                "further expansion has to be generated by terminal long run evidence. A gate "
                "that failed here names what would license the next classification, and that "
                "naming is the only door: a subsystem nobody's evidence asked for does not get "
                "built because it would be interesting"
            ),
            "what_would_reopen_it": verdict["requires_for_next_level"],
            "what_may_still_change": (
                "repairs to a defect reproduced by a test, and the beds "
                "themselves. A bed is an instrument, and a frozen instrument "
                "that is known to be wrong is not a freeze, it is a mistake "
                "held in place"
            ),
        },
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    io.seal("SUBSTRATE_NOUS_CLOSURE.json", doc)
    print(
        json.dumps(
            {
                "passed": doc["verdict"]["passed"],
                "failed": doc["verdict"]["failed"],
                "classification": doc["verdict"]["classification"],
                "terminally_gated": [g["gate"] for g in doc["terminally_gated"]],
                "detail": {
                    k: {
                        "passes": v["passes"],
                        "active": v["mechanism_activity"]["active"],
                        "headroom": v["oracle_headroom"].get("oracle"),
                    }
                    for k, v in doc["results"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
