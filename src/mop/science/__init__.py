"""The one MOP scientific engine: sealed records interpreted through one explicit lifecycle.

An experiment is an ordinary data record.  The record names its unique mathematics providers and declares
its question, null, source, split, independent unit, treatments, controls, metric, SESOI, multiplicity,
budget, stop rule, claim ceiling, and structurally independent verifier.  The engine validates the record's
canonical authority, executes the declared lifecycle, and seals the artifact.  Graded scientific
recomputation remains in the experiment family's independent verifier.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from mop.science.statistics import exact_sign_flip
from mop.substrate.events import canonical_sha256

Result = dict[str, object]
Provider = Callable[[str, int, object], Result]
Verifier = Callable[[dict[str, list[Result]], Mapping[str, object]], Mapping[str, object]]

PROGRAM = ("run_arms", "pair_primary", "decide", "project", "seal")
REQUIRED = {
    "id", "schema", "stage", "question", "null", "source", "split", "unit", "providers",
    "treatments", "controls", "metric", "sesoi", "multiplicity", "budget", "stop", "claims",
    "verification", "seeds", "program", "record_sha256",
}


class RecordRefused(ValueError):
    """The record, provider output, artifact, or independent recomputation is unsafe."""


def seal_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return a copy bound to the canonical scientific identity of every declared field."""

    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**body, "record_sha256": canonical_sha256(body)}


def validate_record(record: Mapping[str, object]) -> None:
    """Refuse incomplete, drifted, unsafe, or unknown experiment programs."""

    missing = REQUIRED - record.keys()
    if missing:
        raise RecordRefused(f"record is missing {sorted(missing)}")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != canonical_sha256(body):
        raise RecordRefused("the scientific record authority has drifted")
    metric = record["metric"]
    controls = record["controls"]
    claims = record["claims"]
    if not isinstance(metric, Mapping) or metric.get("direction") not in ("lower", "higher"):
        raise RecordRefused("metric.direction must be lower or higher")
    if not isinstance(controls, Mapping) or controls.get("primary") not in controls.get("arms", ()):
        raise RecordRefused("the primary control must be a declared control arm")
    if not isinstance(claims, Mapping) or claims.get("activation_allowed") is not False:
        raise RecordRefused("activation must be explicitly refused")
    if claims.get("scientific_promotion") is not False:
        raise RecordRefused("scientific promotion must be explicitly refused")
    if tuple(record["program"]) != PROGRAM:
        raise RecordRefused("unknown or reordered lifecycle program")
    if not tuple(record["seeds"]):
        raise RecordRefused("at least one independent seed is required")


def _run_arms(state: dict[str, Any], provider: Provider, inputs: object) -> None:
    record = state["record"]
    results: dict[str, list[Result]] = defaultdict(list)
    arms = ("candidate", *record["controls"]["arms"])
    for arm in arms:
        for seed in record["seeds"]:
            row = provider(arm, seed, inputs)
            if row.get("arm") != arm or row.get("seed") != seed:
                raise RecordRefused("provider returned a mismatched arm or seed")
            if not isinstance(row.get("metric_value"), (int, float)):
                raise RecordRefused("provider returned a non-numeric metric")
            results[arm].append(row)
    state["results"] = results


def _pair_primary(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    record, results = state["record"], state["results"]
    candidate = {row["seed"]: row["metric_value"] for row in results["candidate"]}
    primary = record["controls"]["primary"]
    control = {row["seed"]: row["metric_value"] for row in results[primary]}
    if candidate.keys() != control.keys():
        raise RecordRefused("candidate and primary control seeds differ")
    lower = record["metric"]["direction"] == "lower"
    state["deltas"] = [
        (control[seed] - candidate[seed]) if lower else (candidate[seed] - control[seed])
        for seed in sorted(candidate)
    ]


def _decide(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    deltas = state["deltas"]
    favorable = sum(delta > 0 for delta in deltas)
    against = sum(delta < 0 for delta in deltas)
    decisive = favorable + against
    record = state["record"]
    exact = exact_sign_flip(deltas, alpha=record["stop"]["alpha"])
    reproduced = exact.one_sided_significant and exact.mean_delta >= record["sesoi"]["value"]
    state["decision"] = {**exact.payload(), "rule": "paired_sign_flip_one_sided",
                         "favorable": favorable, "against": against, "decisive_pairs": decisive,
                         "ties": len(deltas) - decisive,
                         "proportion": favorable / decisive if decisive else 0.0,
                         "p_value": exact.one_sided_p}
    state["verdict"] = "reproduced_effect" if reproduced else "null_or_inconclusive"


def _project(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    record = state["record"]
    state["artifact"] = {
        "schema": record["schema"], "experiment_id": record["id"], "stage": record["stage"],
        "question": record["question"], "null_hypothesis": record["null"],
        "source": record["source"], "split": record["split"], "unit": record["unit"],
        "providers": record["providers"], "treatments": record["treatments"],
        "controls": record["controls"], "metric": record["metric"], "sesoi": record["sesoi"],
        "multiplicity": record["multiplicity"], "budget": record["budget"], "stop": record["stop"],
        "claims": record["claims"], "verification": record["verification"],
        "record_sha256": record["record_sha256"], "seeds": list(record["seeds"]),
        "results": {key: value for key, value in sorted(state["results"].items())},
        "paired_improvements": state["deltas"], "decision": state["decision"],
        "verdict": state["verdict"], "activation_allowed": False, "scientific_promotion": False,
    }


def _seal(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    state["artifact"]["seal"] = canonical_sha256(state["artifact"])


OPS = {"run_arms": _run_arms, "pair_primary": _pair_primary, "decide": _decide,
       "project": _project, "seal": _seal}


def run_experiment(
    record: Mapping[str, object], provider: Provider, inputs: object = None
) -> dict[str, Any]:
    """Validate and execute one declarative experiment record."""

    validate_record(record)
    state: dict[str, Any] = {"record": record}
    for operation in record["program"]:
        OPS[operation](state, provider, inputs)
    return state["artifact"]


def verify_artifact(
    artifact: Mapping[str, object], record: Mapping[str, object], verifier: Verifier
) -> dict[str, object]:
    """Verify integrity, invariants, and a structurally independent graded recomputation."""

    body = {key: value for key, value in artifact.items() if key != "seal"}
    if artifact.get("seal") != canonical_sha256(body):
        raise RecordRefused("artifact seal is invalid")
    if artifact.get("experiment_id") != record["id"] or artifact.get("schema") != record["schema"]:
        raise RecordRefused("artifact authority differs from its record")
    expected = {
        "stage": record["stage"], "question": record["question"], "null_hypothesis": record["null"],
        "source": record["source"], "split": record["split"], "unit": record["unit"],
        "providers": record["providers"], "treatments": record["treatments"],
        "controls": record["controls"], "metric": record["metric"], "sesoi": record["sesoi"],
        "multiplicity": record["multiplicity"], "budget": record["budget"], "stop": record["stop"],
        "claims": record["claims"], "verification": record["verification"],
        "record_sha256": record["record_sha256"], "seeds": list(record["seeds"]),
    }
    for field, declared in expected.items():
        if artifact.get(field) != declared:
            raise RecordRefused(f"artifact {field} differs from the scientific record")
    if artifact.get("activation_allowed") is not False or artifact.get("scientific_promotion") is not False:
        raise RecordRefused("artifact widens activation or promotion")
    for verb in record["claims"]["forbidden_verbs"]:
        if verb in record["claims"]["ceiling"]:
            raise RecordRefused(f"forbidden claim verb present: {verb}")
    recomputed = verifier(artifact["results"], record)
    if recomputed.get("verdict") != artifact.get("verdict"):
        raise RecordRefused("independent verifier disagrees with the artifact")
    return {"verified": True, "experiment_id": record["id"], "seal_reproduced": True,
            "independent_verdict": recomputed["verdict"], "independent_scientific_confirmation": False}


def render_report(artifact: Mapping[str, object]) -> str:
    """Render the compact common audit view."""

    decision = artifact["decision"]
    return (f"# {artifact['experiment_id']} (stage {artifact['stage']})\n\n"
            f"- verdict: {artifact['verdict']}\n"
            f"- favorable/against/ties: {decision['favorable']}/{decision['against']}/{decision['ties']}\n"
            f"- activation_allowed: false; scientific_promotion: false\n")


__all__ = [
    "PROGRAM", "Provider", "RecordRefused", "Result", "Verifier", "render_report", "run_experiment",
    "seal_record", "validate_record", "verify_artifact",
]
