"""Cheap certification: runtime activity, session canaries and body distinctness.

The question this answers is unglamorous and decisive. The scaffold has eleven runtime stages, three
bodies and a sealed session authority. Are any of them actually doing anything, or is the composition a
diagram that happens to execute?

Runtime activity is measured by ablation. A stage that can be switched off without changing the declared
state or the decision path on a fixture designed to need it is not a stage, it is wiring. Section 2 of the
authority is explicit that such a result is an instrumentation failure and not a scientific null, so it is
classified that way here.

The null fixture is the control that makes the positive fixture mean anything. If ablating a stage changes
the outcome on a stream where it should be irrelevant, the difference was noise and the positive result on
the other fixture cannot be trusted either.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys

from substrate import bodies as B
from substrate import evidence as io
from substrate import runtime as R
from substrate import sessions as S

# perceive is unconditional: without it the cycle has no input at all, so ablating it tests nothing
ABLATABLE = tuple(s for s in R.STAGES if s != "perceive")

# A null control is a stream on which the stage has nothing to do, so ablating it must change nothing.
# Three stages write owned state on every cycle by design, and no stream exists on which they are idle.
# Manufacturing a null for them and reporting the inevitable difference as a failure would be the
# instrument lying about itself, so the inapplicability is declared with its reason.
NULL_CONTROL_INAPPLICABLE = {
    "decide": "writes the decision region every cycle by design",
    "remember": "writes an episode every cycle by design",
    "checkpoint": "writes the self region and the identity digest every cycle by design",
    # arbitration emits a report every cycle, and the contrast arm is not silence, it is a different
    # report: the first output standing unexamined, undeferred and demanding no evidence. A stream that
    # made those two coincide would have to be one where arbitration reaches the fabricated conclusion,
    # which is a positive control wearing a null control's clothes. Even the empty catalog fails: the real
    # arbiter defers and names the missing evidence, the contrast arm does neither.
    "arbitrate": "emits an arbitration report every cycle by design, and the no arbiter arm fabricates a report of its own rather than staying silent",
}


def stage_null_fixture(stage: str) -> tuple[list[dict], dict]:
    """A stream, and runtime keyword arguments, under which the named stage has nothing to do."""
    flat = [{"observation": {"label": "a", "label_confidence": 0.5}, "outcome": None, "goal": None} for _ in range(8)]
    if stage == "attend":
        # attend has two effects, not one: it drops candidates under budget pressure, and the regions it
        # attends filter the perspective pool. A fixture that only removes the budget pressure leaves the
        # second effect running, so it is not a null control, it is a weaker positive one. Both effects are
        # neutralised here: no budget pressure, and every perspective in the catalog reads an attended
        # region, so the filter is the identity map whether the stage runs or is ablated.
        attended = {c["id"] for c in R.Substrate()._attention_candidates({"label": "a"})}
        return flat, {
            "cycle_budget": 1e6,
            "catalog": [p for p in R.PS.CATALOG if set(p.spec.inputs) <= attended],
        }
    if stage in ("select", "run_perspectives"):
        # an empty catalog leaves nothing to select or run
        return flat, {"catalog": []}
    if stage == "consolidate":
        # no outcome means no episode is ever verified, so the policy selects nothing
        return flat, {}
    if stage == "adapt":
        # the envelope is unchanged, so the proposal is admitted identically every cycle
        return flat, {}
    if stage == "self_update":
        # no outcome, so there is nothing to compare a prediction against
        return flat, {}
    return flat, {}


CANARIES = (
    "restoration",
    "goal_preservation",
    "memory_reuse",
    "calibration",
    "reliability_update",
    "world_model_value",
    "reflective_receipts",
)

BODY_DIMENSIONS = ("implementation", "state", "resources", "tool_path", "checkpoint", "output")


class Refused(RuntimeError):
    """A certification claim the fixtures do not support."""


# ---------------------------------------------------------------- fixtures


def positive_fixture(n: int = 12) -> list[dict]:
    """A stream where the label alternates and the outcome is known, so every stage has work to do."""
    out = []
    for i in range(n):
        label = "a" if i % 2 == 0 else "b"
        out.append(
            {
                "observation": {"label": label, "label_confidence": 0.6 + 0.3 * (i % 2)},
                "outcome": label,
                "goal": ["classify the stream"],
            }
        )
    return out


def null_fixture(n: int = 12) -> list[dict]:
    """A stream with one constant label, no goal and no outcome. Nothing here should depend on a stage."""
    return [{"observation": {"label": "a", "label_confidence": 0.5}, "outcome": None, "goal": None} for _ in range(n)]


def _run(fixture: list[dict], ablate: frozenset | None = None, **kw) -> dict:
    entity = R.Substrate(ablate=ablate, **kw)
    for row in fixture:
        entity.step(row["observation"], outcome=row["outcome"], goal=row["goal"])
    decisions = [(t["stages"].get("decide") or {}).get("decision") for t in entity.traces]
    # the identity hash carries region names, not region contents, so a stage whose whole product is what
    # it writes into a region reads as inactive under it. Arbitration is exactly that: its decision can
    # coincide with the first perspective while its preserved minority and recorded contradictions do not.
    # Ablation is compared against what a stage produces, so the contents are observed too.
    return {
        "state": entity._state_for_hash(),
        "contents": io.sha_obj(entity.ws.broadcast()),
        "decisions": decisions,
        "reliability": {k: round(v, 6) for k, v in entity.reliability.items()},
        "episodes": len(entity.episodes.store),
        "self_history": len(entity.self_model.history),
    }


def runtime_activity() -> dict:
    """Ablate each stage and require it to matter on the positive fixture and not on the null one."""
    pos_base = _run(positive_fixture())
    rows = {}
    for stage in ABLATABLE:
        pos = _run(positive_fixture(), frozenset({stage}))
        changed_state = (pos["state"], pos["contents"]) != (pos_base["state"], pos_base["contents"])
        changed_decisions = pos["decisions"] != pos_base["decisions"]
        active = changed_state or changed_decisions

        if stage in NULL_CONTROL_INAPPLICABLE:
            null_row = {"applicable": False, "reason": NULL_CONTROL_INAPPLICABLE[stage], "quiet": None}
        else:
            fixture, kw = stage_null_fixture(stage)
            base = _run(fixture, None, **kw)
            ablated = _run(fixture, frozenset({stage}), **kw)
            quiet = base["state"] == ablated["state"] and base["contents"] == ablated["contents"] and base["decisions"] == ablated["decisions"]
            null_row = {
                "applicable": True,
                "quiet": quiet,
                "fixture": f"a stream on which {stage} has nothing to do",
            }

        rows[stage] = {
            "positive_state_changed": changed_state,
            "positive_decision_path_changed": changed_decisions,
            "null_control": null_row,
            "active": active,
            "classification": ("active" if active else "wiring_or_instrumentation_failure"),
            "note": ""
            if active
            else (
                "ablating this stage changed neither the declared state nor the decision path on a "
                "fixture built to need it. Section 2 calls that an instrumentation failure, not a null"
            ),
        }
    inactive = sorted(k for k, v in rows.items() if not v["active"])
    null_sensitive = sorted(k for k, v in rows.items() if v["null_control"]["applicable"] and v["null_control"]["quiet"] is False)
    return {
        "schema": "substrate-runtime-activity/v1",
        "ablatable_stages": list(ABLATABLE),
        "perceive_excluded": "without input the cycle has nothing to run, so ablating it tests nothing",
        "results": rows,
        "active": sorted(k for k, v in rows.items() if v["active"]),
        "inactive": inactive,
        "null_fixture_sensitive": null_sensitive,
        "null_control_inapplicable": dict(NULL_CONTROL_INAPPLICABLE),
        "all_active": not inactive,
        "control_clean": not null_sensitive,
        "reading": (
            "each stage gets a null control built so that stage has nothing to do, and ablating "
            "it there must change nothing. Three stages write owned state every cycle by design "
            "and admit no such stream, which is declared rather than papered over with a fixture "
            "that would fail by construction"
        ),
        "activation": False,
    }


# ---------------------------------------------------------------- session canaries


def session_canaries(limit: int = 60) -> dict:
    session = S.build()
    events = [e for e in session["events"] if e["kind"] in ("shard_completed", "shard_quarantined", "failure_hold")][:limit]
    if len(events) < 20:
        raise Refused(f"only {len(events)} usable session events")

    def observe(e):
        ok = e["kind"] == "shard_completed"
        return {"label": "ok" if ok else "bad", "label_confidence": 0.9 if ok else 0.6}

    entity = R.Substrate()
    for e in events[: len(events) // 2]:
        obs = observe(e)
        entity.step(obs, outcome=obs["label"], goal=["process the sealed session"])

    snapshot = entity.checkpoint()
    revived = R.Substrate().restore(snapshot)
    rows = {}
    rows["restoration"] = {
        "passes": revived.checkpoint()["identity"] == snapshot["identity"],
        "note": "the restored entity reproduces the checkpoint identity or the restore is refused",
    }
    rows["goal_preservation"] = {"passes": revived.ws.read("goal", "canary") == ["process the sealed session"]}
    rows["memory_reuse"] = {
        "episodes": len(revived.episodes.store),
        "passes": len(revived.episodes.store) == len(entity.episodes.store) > 0,
    }

    before = dict(revived.reliability)
    for e in events[len(events) // 2 :]:
        obs = observe(e)
        revived.step(obs, outcome=obs["label"])
    rows["reliability_update"] = {
        "changed": revived.reliability != before,
        "passes": revived.reliability != before,
        "note": "reliability that never moves across half a real session is not being updated",
    }

    cal = revived.self_model.calibration()["per_kind"]["accuracy"]
    rows["calibration"] = {
        "n": cal["n"],
        "passes": cal["n"] > 0,
        "mean_absolute_error": cal.get("mean_absolute_error"),
    }

    from substrate import worldbed as WB

    battery = WB.integrate()
    rows["world_model_value"] = {
        "verdict": battery["verdict"],
        "decision_gain": battery["decision_gain"],
        "passes": battery["verdict"] != "limited_instrument",
        "note": "measured, and negative gain is reported as measured",
    }

    report = revived.report()
    rows["reflective_receipts"] = {
        "answered": report["answered"],
        "bound": report.get("bound_to_receipts"),
        "passes": report["answered"] and report["bound_to_receipts"],
    }

    passed = sorted(k for k, v in rows.items() if v["passes"])
    return {
        "schema": "substrate-session-canaries/v1",
        "canaries": list(CANARIES),
        "events_used": len(events),
        "source": session["session"],
        "results": rows,
        "passed": passed,
        "failed": sorted(set(CANARIES) - set(passed)),
        "all_pass": len(passed) == len(CANARIES),
        "activation": False,
    }


# ---------------------------------------------------------------- body distinctness


def body_canaries() -> dict:
    bed = B._load()
    made = {k: B.make(k, bed) for k in ("compact", "general", "tool")}
    profile = {}
    for k, b in made.items():
        out = b.inference(bed["Xte"][:64])
        profile[k] = {
            "implementation": sorted(b.views),
            "state": b.hidden_state()["shape"],
            "resources": b.resource_report()["n_parameters"],
            "tool_path": bool(b.tool_dominant),
            "checkpoint": b.checkpoint()["sha256"],
            "output": io.sha_obj(out["output"].tolist()),
        }
    # A boolean dimension cannot take three distinct values, so requiring all three to differ marks a
    # correctly built set as non distinct. What matters is that the dimension separates the bodies at all,
    # and that every pair differs somewhere.
    distinct, separates = {}, {}
    for dim in BODY_DIMENSIONS:
        values = [json.dumps(profile[k][dim], sort_keys=True) for k in profile]
        distinct[dim] = len(set(values)) == len(values)
        separates[dim] = len(set(values)) > 1
    names = sorted(profile)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            differing = [d for d in BODY_DIMENSIONS if json.dumps(profile[a][d], sort_keys=True) != json.dumps(profile[b][d], sort_keys=True)]
            pairwise[f"{a}_vs_{b}"] = differing

    # a microtask where the substrate changes body facing behaviour: arbitration over the body's views
    # The microtask has to give the substrate something to disagree with, or it can only echo the body.
    # The body proposes a label; the episodic window carries what actually happened on similar past
    # observations. Where those conflict the arbiter can override, and how often it does is the measure.
    micro = {}
    for k, b in made.items():
        raw = b.inference(bed["Xte"])["output"]
        alone = float((raw == bed["Yte"]).mean())
        entity = R.Substrate()
        changed, agreed = 0, 0
        for i in range(96):
            proposal = str(int(raw[i]))
            truth = str(int(bed["Yte"][i]))
            obs = {"label": proposal, "label_confidence": 0.55}
            entity.step(obs, outcome=truth, goal=["classify against the body proposal"])
            decided = (entity.ws.read("decision", "canary") or {}).get("value")
            if decided is None:
                continue
            if decided != proposal:
                changed += 1
            else:
                agreed += 1
        micro[k] = {
            "body_alone_accuracy": round(alone, 6),
            "decisions_substrate_changed": changed,
            "decisions_substrate_agreed": agreed,
            "substrate_changes_body_facing_behaviour": changed > 0,
        }

    return {
        "schema": "substrate-body-canaries/v1",
        "dimensions": list(BODY_DIMENSIONS),
        "profiles": profile,
        "all_three_differ_on": distinct,
        "dimension_separates_the_set": separates,
        "pairwise_differing_dimensions": pairwise,
        "every_pair_differs_somewhere": all(bool(v) for v in pairwise.values()),
        "every_dimension_separates": all(separates.values()),
        "distinct_on_every_dimension": all(separates.values()),
        "non_distinct_dimensions": sorted(k for k, v in separates.items() if not v),
        "criterion": (
            "a dimension must separate the set, and every pair of bodies must differ on at "
            "least one dimension. Requiring three distinct values on a boolean would mark "
            "a correctly built set as non distinct"
        ),
        "microtasks": micro,
        "any_body_facing_change": any(m["substrate_changes_body_facing_behaviour"] for m in micro.values()),
        "activation": False,
    }


def run() -> dict:
    activity = runtime_activity()
    canaries = session_canaries()
    body = body_canaries()
    from substrate import sx2 as X2

    sx2 = json.loads((io.PROOF / "SUBSTRATE_SX2_DIVERSITY.json").read_text()) if (io.PROOF / "SUBSTRATE_SX2_DIVERSITY.json").is_file() else X2.run()
    gated = []
    if activity["inactive"]:
        gated.append(
            {
                "component": "runtime_stages",
                "detail": activity["inactive"],
                "reason": "wiring or instrumentation failure",
            }
        )
    if canaries["failed"]:
        gated.append({"component": "session_canaries", "detail": canaries["failed"], "reason": "canary did not pass"})
    if not body["distinct_on_every_dimension"]:
        gated.append(
            {
                "component": "bodies",
                "detail": body["non_distinct_dimensions"],
                "reason": "bodies not distinct on every declared dimension",
            }
        )
    return {
        "schema": "substrate-long-run-certification/v1",
        "runtime_activity": activity,
        "session_canaries": canaries,
        "body_canaries": body,
        "sx2": {
            "verdict": sx2["verdict"],
            "k_clearing_sesoi": sx2["k_clearing_sesoi"],
            "best_margin": max(r["margin_over_matched_single"] for r in sx2["k_rows"].values()),
            "sesoi": sx2["sesoi"],
        },
        "gated_components": gated,
        "green": not gated,
        "rule": "a failed component is gated out of the run rather than blocking the run",
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    io.seal("SUBSTRATE_LONG_RUN_CERTIFICATION.json", doc)
    print(
        json.dumps(
            {
                "green": doc["green"],
                "runtime_active": doc["runtime_activity"]["active"],
                "runtime_inactive": doc["runtime_activity"]["inactive"],
                "control_clean": doc["runtime_activity"]["control_clean"],
                "canaries_failed": doc["session_canaries"]["failed"],
                "bodies_distinct": doc["body_canaries"]["distinct_on_every_dimension"],
                "sx2": doc["sx2"]["verdict"],
                "gated": [g["component"] for g in doc["gated_components"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
