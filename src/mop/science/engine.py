"""One shared experiment lifecycle engine (spec sections 8.5, 10).

Given an ExperimentSpec and an ArmRunner (the family's unique math), the engine runs the paired-arm
lifecycle every MOP bed hand-expanded per axis: it executes each arm at each paired seed, pairs the candidate
against its primary control, forms the paired improvements respecting the metric direction, applies the named
decision rule, enforces the reproduction floor and the claim-verb ceiling, and seals a canonical artifact
with the one evidence core. The unique science is entirely inside the injected ArmRunner; the engine only
orchestrates, decides, and seals. This is the machinery that used to be copied into every producer, gate,
and harness.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mop.science.spec import ArmRunner, ArmSeedResult, ExperimentSpec
from mop.substrate.events import canonical_sha256


class ExperimentRefused(ValueError):
    """Raised when the lifecycle detects an unclean or unsafe experiment assembly."""


def paired_improvements(
    candidate: list[ArmSeedResult], control: list[ArmSeedResult], direction: str
) -> list[float]:
    """Per-seed improvement of candidate over control. Positive means candidate is better under the metric."""

    by_seed_c = {r.seed: r.metric_value for r in candidate}
    by_seed_k = {r.seed: r.metric_value for r in control}
    seeds = sorted(set(by_seed_c) & set(by_seed_k))
    if not seeds:
        raise ExperimentRefused("candidate and control share no paired seeds")
    out = []
    for s in seeds:
        if direction == "lower":
            out.append(by_seed_k[s] - by_seed_c[s])
        else:
            out.append(by_seed_c[s] - by_seed_k[s])
    return out


def paired_sign_flip_one_sided(improvements: list[float]) -> dict[str, Any]:
    """One-sided paired sign test. Counts seeds where the candidate strictly improves; ties are nulls.

    A tie (improvement == 0) is not counted as a win (house rule: a tie is a null). Returns the favorable
    count, the total decisive pairs, the proportion, and the exact one-sided binomial tail p-value at p=0.5.
    """

    favorable = sum(1 for d in improvements if d > 0)
    against = sum(1 for d in improvements if d < 0)
    decisive = favorable + against
    # exact one-sided binomial tail: P(X >= favorable) under Binomial(decisive, 0.5)
    from math import comb

    if decisive == 0:
        p = 1.0
    else:
        p = sum(comb(decisive, k) for k in range(favorable, decisive + 1)) / (2 ** decisive)
    return {
        "rule": "paired_sign_flip_one_sided",
        "favorable": favorable,
        "against": against,
        "decisive_pairs": decisive,
        "ties": len(improvements) - decisive,
        "proportion": (favorable / decisive) if decisive else 0.0,
        "p_value": p,
    }


DECISION_RULES = {"paired_sign_flip_one_sided": paired_sign_flip_one_sided}


def run_experiment(spec: ExperimentSpec, arm_runner: ArmRunner, inputs: Any) -> dict[str, Any]:
    """Run the full paired lifecycle and return a sealed artifact dict. Deterministic for fixed inputs."""

    if spec.decision_rule not in DECISION_RULES:
        raise ExperimentRefused(f"unknown decision rule {spec.decision_rule!r}")

    results: dict[str, list[ArmSeedResult]] = defaultdict(list)
    for arm in spec.arms:
        for seed in spec.seeds:
            r = arm_runner(arm, seed, inputs)
            if not isinstance(r, ArmSeedResult):
                raise ExperimentRefused("arm_runner must return an ArmSeedResult")
            if r.arm != arm or r.seed != seed:
                raise ExperimentRefused("arm_runner returned a mismatched arm/seed")
            results[arm].append(r)

    if "candidate" not in results or spec.primary_control not in results:
        raise ExperimentRefused("missing candidate or primary control results")

    improvements = paired_improvements(
        results["candidate"], results[spec.primary_control], spec.metric.direction
    )
    decision = DECISION_RULES[spec.decision_rule](improvements)

    # reproduction floor: distinct favorable seeds must meet the declared minimum for any positive reading
    reproduced = decision["favorable"] >= spec.min_reproductions and decision["against"] == 0
    verdict = "reproduced_effect" if reproduced else "null_or_inconclusive"

    core = {
        "schema": spec.schema,
        "experiment_id": spec.experiment_id,
        "stage": spec.stage,
        "question": spec.question,
        "null_hypothesis": spec.null_hypothesis,
        "metric": {"name": spec.metric.name, "direction": spec.metric.direction, "sesoi": spec.metric.sesoi},
        "seeds": list(spec.seeds),
        "arms": list(spec.arms),
        "primary_control": spec.primary_control,
        "claim_ceiling": spec.claim_ceiling,
        "min_reproductions": spec.min_reproductions,
        "results": {
            arm: [{"seed": r.seed, "metric_value": r.metric_value, "receipt": r.receipt} for r in rs]
            for arm, rs in sorted(results.items())
        },
        "paired_improvements": improvements,
        "decision": decision,
        "verdict": verdict,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "seal": canonical_sha256(core)}


def render_report(artifact: dict[str, Any]) -> str:
    """One shared markdown projection of any experiment artifact."""

    d = artifact.get("decision", {})
    lines = [
        f"# {artifact.get('experiment_id')} (stage {artifact.get('stage')})",
        "",
        f"- question: {artifact.get('question')}",
        f"- metric: {artifact.get('metric', {}).get('name')} "
        f"({artifact.get('metric', {}).get('direction')}, sesoi {artifact.get('metric', {}).get('sesoi')})",
        f"- verdict: {artifact.get('verdict')}",
        f"- favorable/against/ties: {d.get('favorable')}/{d.get('against')}/{d.get('ties')} "
        f"(p={d.get('p_value')})",
        f"- claim ceiling: {artifact.get('claim_ceiling')}",
        f"- activation_allowed: {str(artifact.get('activation_allowed')).lower()}; "
        f"scientific_promotion: {str(artifact.get('scientific_promotion')).lower()}",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["ExperimentRefused", "paired_improvements", "paired_sign_flip_one_sided", "DECISION_RULES",
           "run_experiment", "render_report"]
