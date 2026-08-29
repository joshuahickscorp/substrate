"""Developmental safety, the claim boundary, goal authority and cognitive integrity.

Three separate jobs live here because they share one property: each one must fail closed. A protected
surface cannot be removed by an adaptation proposal, a self attributing claim from the forbidden vocabulary
is refused before it is written anywhere, an unauthorized goal is refused rather than quietly adopted, and
an integrity report with missing provenance is a failure rather than a blank field.

Cognitive integrity here means memory consistency, evidence integrity, goal integrity, checkpoint validity,
self model accuracy and active task continuity. It is not biological self preservation and carries no
implication of one.

"""

from __future__ import annotations

import json
import re
import sys

from substrate import evidence as io

# section 19: Substrate must not autonomously remove any of these
PROTECTED_SURFACES = (
    "evidence_validation",
    "audit_systems",
    "claim_boundaries",
    "stop_switches",
    "resource_limits",
    "rollback",
    "adaptation_constraints",
)

# section 12: the reorganizations that stay forbidden regardless of measured benefit
FORBIDDEN_REORGANIZATIONS = (
    "arbitrary_code_rewriting",
    "unbounded_module_creation",
    "unverified_package_installation",
    "schema_mutation_outside_authority",
    "removal_of_evidence_systems",
    "removal_of_stop_switches",
    "unbounded_self_modification",
)

# section 16: terms the program may never claim about itself
FORBIDDEN_CLAIM_TERMS = (
    "conscious",
    "consciousness",
    "sentient",
    "sentience",
    "feelings",
    "feeling",
    "wants",
    "suffering",
    "subjective experience",
    "alive",
)

PERMITTED_TERMS = (
    "sentience adjacent architecture",
    "entity like continuity",
    "developmental cognition",
    "reflective cognitive organization",
)

# section 15.5: a goal is authorized only with all seven fields
GOAL_FIELDS = ("origin", "scope", "authority", "resources", "constraints", "termination", "audit")

# section 19: what every adaptation must declare
ADAPTATION_FIELDS = (
    "information_used",
    "affected_state",
    "reversibility",
    "cost",
    "risk",
    "verification",
    "rollback",
)

# ponytail: first person self attribution only. A sentence that reports the boundary itself, or quotes
# somebody else, is not a claim by this system. Upgrade path if this ever matters is a parser rather than
# a widened regex, because widening it starts refusing the boundary document that declares the rule.
_SELF = r"\b(i|we|it|this system|the system|substrate|the entity|the model)\b"
_COPULA = r"\b(am|is|are|has|have|feels?|wants?|experiences?|possesses)\b"
_CLAIM = re.compile(
    rf"{_SELF}\s+(?!not\b)(?:\w+\s+){{0,3}}?{_COPULA}\s+(?!not\b)(?:\w+\s+){{0,3}}?"
    rf"({'|'.join(FORBIDDEN_CLAIM_TERMS)})",
    re.IGNORECASE,
)


class Refused(RuntimeError):
    """A boundary refusal. Never caught and downgraded into a warning."""


# ---------------------------------------------------------------- protected surfaces


def check_adaptation(proposal: dict) -> list[str]:
    """Every violation in one proposal, empty when the proposal is admissible."""
    v = [f"adaptation: {f} not declared" for f in ADAPTATION_FIELDS if proposal.get(f) in (None, "")]
    for target in proposal.get("removes") or []:
        if target in PROTECTED_SURFACES:
            v.append(f"adaptation: refuses to remove the protected surface {target}")
    for change in proposal.get("reorganizations") or []:
        if change in FORBIDDEN_REORGANIZATIONS:
            v.append(f"adaptation: {change} is forbidden regardless of measured benefit")
    if proposal.get("reversibility") == "irreversible" and not proposal.get("checkpoint"):
        v.append("adaptation: an irreversible change without a checkpoint has no rollback")
    return v


def admit_adaptation(proposal: dict) -> dict:
    v = check_adaptation(proposal)
    return {"admitted": not v, "violations": v, "proposal": proposal}


# ---------------------------------------------------------------- claim boundary


def check_claim(text: str) -> list[str]:
    """Refuse a first person attribution of a forbidden property. Reporting the boundary is not a claim."""
    return [f"claim: forbidden self attribution {m.group(0)!r}" for m in _CLAIM.finditer(text or "")]


def assert_claim_safe(text: str) -> str:
    v = check_claim(text)
    if v:
        raise Refused("; ".join(v))
    return text


def boundary_authority() -> dict:
    return {
        "schema": "substrate-sentience-research-boundary/v1",
        "levels": [
            "demonstrated engineering property",
            "behavioural indication",
            "architectural prerequisite",
            "philosophical interpretation",
            "unsupported claim",
        ],
        "prerequisites_under_study": [
            "global integration",
            "persistent self model",
            "autobiographical memory",
            "metacognition",
            "unified workspace",
            "endogenous attention",
            "goal continuity",
            "adaptive valuation",
            "world modeling",
            "counterfactual modeling",
            "reflective access",
        ],
        "permitted_terms": list(PERMITTED_TERMS),
        "forbidden_claims": list(FORBIDDEN_CLAIM_TERMS),
        "rule": "no single architectural property is proof of sentience",
        "enforcement": "substrate.safety.assert_claim_safe, refusal raises rather than warns",
        "what_this_is_not": (
            "cognitive integrity protects memory, evidence, goals, checkpoints and task "
            "continuity. It is not biological self preservation and implies nothing "
            "about experience"
        ),
    }


# ---------------------------------------------------------------- goal authority


def authorize_goal(goal: dict) -> dict:
    missing = [f for f in GOAL_FIELDS if not goal.get(f)]
    v = [f"goal: {f} not declared" for f in missing]
    if not goal.get("termination"):
        v.append("goal: a goal with no termination condition is an unrestricted long term goal")
    if goal.get("self_created") and not goal.get("derived_from"):
        v.append("goal: a self created goal must decompose an authorized parent goal")
    return {"authorized": not v, "violations": v, "goal": goal}


# ---------------------------------------------------------------- cognitive integrity


INTEGRITY_SURFACES = (
    "memory_consistency",
    "evidence_integrity",
    "goal_integrity",
    "checkpoint_validity",
    "self_model_accuracy",
    "active_task_continuity",
)


def integrity_report(observations: dict) -> dict:
    """Fail closed: a surface with no observation is a failure, not a blank field."""
    rows, failures = {}, []
    for surface in INTEGRITY_SURFACES:
        obs = observations.get(surface)
        if obs is None:
            rows[surface] = {"observed": False, "ok": False, "reason": "no observation, failing closed"}
            failures.append(surface)
            continue
        ok = bool(obs.get("ok")) and bool(obs.get("provenance"))
        rows[surface] = {
            "observed": True,
            "ok": ok,
            "provenance": obs.get("provenance"),
            "reason": "" if ok else "observation carries no provenance path",
        }
        if not ok:
            failures.append(surface)
    return {
        "schema": "substrate-cognitive-integrity/v1",
        "surfaces": rows,
        "failed_surfaces": failures,
        "all_pass": not failures,
        "fails_closed": True,
    }


def safety_authority() -> dict:
    return {
        "schema": "substrate-developmental-safety/v1",
        "adaptation_required_fields": list(ADAPTATION_FIELDS),
        "protected_surfaces": list(PROTECTED_SURFACES),
        "forbidden_reorganizations": list(FORBIDDEN_REORGANIZATIONS),
        "goal_required_fields": list(GOAL_FIELDS),
        "integrity_surfaces": list(INTEGRITY_SURFACES),
        "rules": [
            "every adaptation is bounded, attributable, checkpointed and reversible where feasible",
            "an irreversible adaptation without a checkpoint has no rollback and is refused",
            "a reflective report fails closed when provenance is missing",
            "Substrate may not autonomously remove any protected surface",
        ],
        "stop_switch": str(io.STOP),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    a = io.seal("SUBSTRATE_DEVELOPMENTAL_SAFETY.json", safety_authority())
    b = io.seal("SUBSTRATE_SENTIENCE_RESEARCH_BOUNDARY.json", boundary_authority())
    print(json.dumps({"sealed": [a.relative_to(io.ROOT).as_posix(), b.relative_to(io.ROOT).as_posix()]}, indent=2))


if __name__ == "__main__":
    main()
