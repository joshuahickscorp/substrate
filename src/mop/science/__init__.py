from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from mop.ladder.ladder_contracts import mint_demonstration
from mop.science.statistics import exact_sign_flip
from mop.substrate.events import canonical_sha256

Result = dict[str, object]
Provider = Callable[[str, int, object], Result]
Verifier = Callable[[dict[str, list[Result]], Mapping[str, object]], Mapping[str, object]]
MATCHED_BUDGET_WALL_NOTE = (
    "wall_ns is a deterministic nominal at a 1 GFLOP/s reference so the artifact is byte-reproducible; "
    "the measured wall is unsealed run provenance, and the authoritative sealed compute axes are the "
    "parameter count and the FLOP ledger"
)

PROGRAM = ("run_arms", "pair_primary", "decide", "project", "seal")
REQUIRED = {
    "id",
    "schema",
    "stage",
    "question",
    "null",
    "source",
    "split",
    "unit",
    "providers",
    "treatments",
    "controls",
    "metric",
    "sesoi",
    "multiplicity",
    "budget",
    "stop",
    "claims",
    "verification",
    "seeds",
    "program",
    "record_sha256",
}


class RecordRefused(ValueError):
    pass


def safety_flags() -> dict[str, bool]:

    return {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


def read_sealed_prereg_member(
    path: str | Path,
    *,
    expected_schema: str,
    family_field: str,
    member_field: str,
    member_id: str,
    family_label: str,
    refusal: type[ValueError],
) -> dict[str, Any]:

    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise refusal(
            f"the sealed {family_label} preregistration {prereg_path} is missing; seal it before the run"
        )
    body = json.loads(prereg_path.read_bytes().decode("utf-8"))
    if body.get("schema") != expected_schema:
        raise refusal(f"unexpected {family_label} prereg schema {body.get('schema')!r}")
    ids = [entry[member_field] for entry in body.get(family_field, [])]
    if member_id not in ids:
        raise refusal(f"{member_id!r} is not preregistered in {prereg_path}")
    return body


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    artifact: dict[str, Any]
    verdict: str
    detail: dict[str, Any] = dataclass_field(default_factory=dict)
    prereg: dict[str, Any] | None = None
    receipt_payload: dict[str, Any] | None = None

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def demonstration_receipt(
    *,
    mechanism_id: str,
    controls_cleared: tuple[str, ...],
    evidence: Mapping[str, object],
    verdict: str,
    detail: Mapping[str, object],
    stage: int = 3,
    requirement_id: str = "stage3.confirmed_useful_mechanism",
) -> dict[str, Any]:

    return mint_demonstration(
        mechanism_id=mechanism_id,
        stage=stage,
        requirement_id=requirement_id,
        controls_cleared=controls_cleared,
        evidence_digest=canonical_sha256(evidence),
        verdict=verdict,
        detail=dict(detail),
    ).payload()


def artifact_envelope(
    *,
    schema: str,
    report: Any,
    seeds: Iterable[object],
    per_seed: object,
    stats: Mapping[str, object],
    controls: Mapping[str, object],
    flags: Mapping[str, object],
    verdict: str,
    featurizer: Mapping[str, object],
    gate: Mapping[str, object],
    receipt_payload: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:

    policy = report.policy
    body: dict[str, Any] = {
        "schema": schema,
        "stage": 3,
        "bed_id": policy.bed_id,
        "claim_scope": policy.claim_scope,
        "source_kind": report.source_kind,
        "rights_clean": True,
        "reproductions": 0,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "stats": dict(stats),
        "controls": dict(controls),
        "flags": dict(flags),
        "verdict": verdict,
        "harness": report.payload(),
        "featurizer": dict(featurizer),
        "gate": dict(gate),
        "demonstration_receipt": dict(receipt_payload),
    }
    if hasattr(report, "matched_budget"):
        body.update(
            {
                "matched_budget": report.matched_budget.payload(),
                "matched_budget_wall_note": MATCHED_BUDGET_WALL_NOTE,
                "break_even": report.break_even.payload(),
            }
        )
    additions = dict(extra or {})
    if body.keys() & additions.keys():
        raise RecordRefused("extra artifact fields overlap the shared envelope")
    body.update(additions)
    return body


def finalize_artifact(
    body: Mapping[str, object],
    *,
    verdict: str,
    detail: Mapping[str, object] | None = None,
    prereg: dict[str, Any] | None = None,
    receipt_payload: dict[str, Any] | None = None,
) -> ArtifactResult:

    if "seal" in body:
        raise RecordRefused("an unsealed artifact body is required")
    artifact = dict(body)
    artifact["seal"] = canonical_sha256(artifact)
    return ArtifactResult(
        artifact=artifact,
        verdict=verdict,
        detail=dict(detail or {}),
        prereg=prereg,
        receipt_payload=receipt_payload,
    )


def seal_record(record: Mapping[str, object]) -> dict[str, object]:

    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**body, "record_sha256": canonical_sha256(body)}


def validate_record(record: Mapping[str, object]) -> None:

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
    state["decision"] = {
        **exact.payload(),
        "rule": "paired_sign_flip_one_sided",
        "favorable": favorable,
        "against": against,
        "decisive_pairs": decisive,
        "ties": len(deltas) - decisive,
        "proportion": favorable / decisive if decisive else 0.0,
        "p_value": exact.one_sided_p,
    }
    state["verdict"] = "reproduced_effect" if reproduced else "null_or_inconclusive"


def _project(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    record = state["record"]
    state["artifact"] = {
        "schema": record["schema"],
        "experiment_id": record["id"],
        "stage": record["stage"],
        "question": record["question"],
        "null_hypothesis": record["null"],
        "source": record["source"],
        "split": record["split"],
        "unit": record["unit"],
        "providers": record["providers"],
        "treatments": record["treatments"],
        "controls": record["controls"],
        "metric": record["metric"],
        "sesoi": record["sesoi"],
        "multiplicity": record["multiplicity"],
        "budget": record["budget"],
        "stop": record["stop"],
        "claims": record["claims"],
        "verification": record["verification"],
        "record_sha256": record["record_sha256"],
        "seeds": list(record["seeds"]),
        "results": {key: value for key, value in sorted(state["results"].items())},
        "paired_improvements": state["deltas"],
        "decision": state["decision"],
        "verdict": state["verdict"],
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def _seal(state: dict[str, Any], _provider: Provider, _inputs: object) -> None:
    state["artifact"]["seal"] = canonical_sha256(state["artifact"])


OPS = {
    "run_arms": _run_arms,
    "pair_primary": _pair_primary,
    "decide": _decide,
    "project": _project,
    "seal": _seal,
}


def run_experiment(record: Mapping[str, object], provider: Provider, inputs: object = None) -> dict[str, Any]:

    validate_record(record)
    state: dict[str, Any] = {"record": record}
    for operation in record["program"]:
        OPS[operation](state, provider, inputs)
    return state["artifact"]


def verify_artifact(
    artifact: Mapping[str, object], record: Mapping[str, object], verifier: Verifier
) -> dict[str, object]:

    body = {key: value for key, value in artifact.items() if key != "seal"}
    if artifact.get("seal") != canonical_sha256(body):
        raise RecordRefused("artifact seal is invalid")
    if artifact.get("experiment_id") != record["id"] or artifact.get("schema") != record["schema"]:
        raise RecordRefused("artifact authority differs from its record")
    expected = {
        "stage": record["stage"],
        "question": record["question"],
        "null_hypothesis": record["null"],
        "source": record["source"],
        "split": record["split"],
        "unit": record["unit"],
        "providers": record["providers"],
        "treatments": record["treatments"],
        "controls": record["controls"],
        "metric": record["metric"],
        "sesoi": record["sesoi"],
        "multiplicity": record["multiplicity"],
        "budget": record["budget"],
        "stop": record["stop"],
        "claims": record["claims"],
        "verification": record["verification"],
        "record_sha256": record["record_sha256"],
        "seeds": list(record["seeds"]),
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
    return {
        "verified": True,
        "experiment_id": record["id"],
        "seal_reproduced": True,
        "independent_verdict": recomputed["verdict"],
        "independent_scientific_confirmation": False,
    }


def render_report(artifact: Mapping[str, object]) -> str:

    decision = artifact["decision"]
    return (
        f"# {artifact['experiment_id']} (stage {artifact['stage']})\n\n"
        f"- verdict: {artifact['verdict']}\n"
        f"- favorable/against/ties: {decision['favorable']}/{decision['against']}/{decision['ties']}\n"
        f"- activation_allowed: false; scientific_promotion: false\n"
    )


__all__ = [
    "MATCHED_BUDGET_WALL_NOTE",
    "PROGRAM",
    "ArtifactResult",
    "Provider",
    "RecordRefused",
    "Result",
    "Verifier",
    "artifact_envelope",
    "demonstration_receipt",
    "finalize_artifact",
    "render_report",
    "run_experiment",
    "safety_flags",
    "seal_record",
    "validate_record",
    "verify_artifact",
]
