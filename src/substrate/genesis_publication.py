"""The terminal classification and what it is allowed to say.

The three outcomes are decided by the gates the constitution froze before any
principal instance existed. Outcome C is not a fallback for a disappointing
result: it is reserved for a broken prerequisite. A sound program whose decisive
claim comes back null is Outcome B, and reporting it as anything else would be
the failure this whole apparatus exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_io as io

# Prerequisites. If one of these is false the program cannot interpret its own
# measurements, and the outcome is C regardless of what the numbers say.
PREREQUISITES = (
    "canaries_pass",
    "distinctness_pass",
    "parity_pass",
    "mutations_zero_survivors",
    "record_store_at_chance",
    "reference_learner_solves_families",
    "oracle_reaches_ceiling",
    "counterfeits_rejected",
    "activation_false",
)

# The decisive gates, exactly as frozen.
OUTCOME_A_GATES = (
    "decisive_effect_at_least_sesoi",
    "decisive_lower_bound_above_zero",
    "all_critical_claims_pass",
    "replication_positive",
    "hidden_composition_positive",
    "oracle_headroom_at_least_minimum",
    "clean_clone_reproduction",
)


def classify(
    *,
    prerequisites: Mapping[str, bool],
    gates: Mapping[str, bool],
    robust: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Decide the terminal outcome from the frozen gates."""
    missing_prerequisites = [name for name in PREREQUISITES if not prerequisites.get(name)]
    missing_gates = [name for name in OUTCOME_A_GATES if not gates.get(name)]

    if missing_prerequisites:
        outcome = "C"
    elif not missing_gates:
        outcome = "A"
    else:
        outcome = "B"

    terminal = dict(C.TERMINAL_OUTCOMES[outcome])
    robust_pass = bool(robust) and all(robust.values())
    return {
        "outcome": outcome,
        **terminal,
        "prerequisites": dict(prerequisites),
        "failed_prerequisites": missing_prerequisites,
        "gates": dict(gates),
        "failed_gates": missing_gates,
        "robust_gate": dict(robust or {}),
        "robust_gate_pass": robust_pass,
        "nous_status": C.STARTING_NOUS_STATUS,
        "claim_boundary": C.CLAIM_BOUNDARY,
        "unqualified_nous_assigned": False,
        "external_activation": False,
        "preserved_starting_classification": C.STARTING_CLASSIFICATION,
        "preserved_starting_p3_effect": C.STARTING_P3_EFFECT,
        "activation": False,
    }


def terminal_report(classification: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    """The published prose. Every positive and every null stated plainly."""
    outcome = classification["outcome"]
    lines: list[str] = []
    add = lines.append

    add("# Substrate Cognitive Material Genesis — terminal report")
    add("")
    add("External activation is `false`. No unqualified Nous is assigned.")
    add("")
    add("## Terminal classification")
    add("")
    add("```text")
    add(f"outcome:        {outcome}")
    add(f"classification: {classification['classification']}")
    if "status" in classification:
        add(f"status:         {classification['status']}")
    if "readiness" in classification:
        add(f"readiness:      {classification['readiness']}")
    add(f"Nous status:    {classification['nous_status']}")
    add("```")
    add("")

    add("## The question")
    add("")
    add("> Can a frozen generic cognitive material develop new useful internal")
    add("> organization from previously unseen experience, and can that")
    add("> development outperform equally resourced static, replayed,")
    add("> precompiled, and equally plastic alternatives?")
    add("")

    decisive = evidence.get("decisive", {})
    if decisive:
        add("## The decisive answer")
        add("")
        add("```text")
        add(f"selected candidate: {decisive.get('candidate')}")
        add(f"comparator:         {decisive.get('comparator')}")
        add(f"effect:             {decisive.get('effect')}")
        add(f"95% CI:             [{decisive.get('confidence_lower')}, {decisive.get('confidence_upper')}]")
        add(f"histories:          {decisive.get('histories')}")
        add(f"oracle headroom:    {decisive.get('oracle_headroom')}")
        add("```")
        add("")

    add("## Prerequisites")
    add("")
    for name in PREREQUISITES:
        state = "pass" if classification["prerequisites"].get(name) else "FAIL"
        add(f"- `{name}` — {state}")
    add("")

    add("## Decisive gates")
    add("")
    for name in OUTCOME_A_GATES:
        state = "pass" if classification["gates"].get(name) else "null" if outcome == "B" else "FAIL"
        add(f"- `{name}` — {state}")
    add("")

    claims = evidence.get("claims", {})
    if claims:
        add("## Primary claims")
        add("")
        for claim in sorted(C.CLAIMS):
            row = claims.get(claim, {})
            verdict = "pass" if row.get("passes") else "null"
            effect = row.get("effect")
            detail = f", effect {effect:+.4f}" if isinstance(effect, (int, float)) else ""
            add(f"- **{claim}** {C.CLAIMS[claim]['statement']} — {verdict}{detail}")
        add("")

    add("## What this does not claim")
    add("")
    add("This program assigns no unqualified Nous, and makes no claim of")
    add("consciousness, sentience, phenomenal experience, moral status, human")
    add("equivalence, or unrestricted autonomy. The inherited Final Revision")
    add("result is preserved unchanged, including its decisive effect of 0.0")
    add("with a 95% confidence interval of [0, 0].")
    add("")

    return "\n".join(lines) + "\n"


def publish(classification: Mapping[str, Any], evidence: Mapping[str, Any], *, publish_files: bool = True) -> dict[str, Any]:
    document = io.authority("substrate-genesis-final-classification/v1", {**classification, "all_pass": True})
    if publish_files:
        io.write_json(io.EVIDENCE / "SUBSTRATE_GENESIS_FINAL_CLASSIFICATION.json", document)
        io.write_text(io.ARTIFACTS / "SUBSTRATE_GENESIS_TERMINAL_REPORT.md", terminal_report(classification, evidence))
    return document


def demo() -> None:
    every = dict.fromkeys(PREREQUISITES, True)
    passing = dict.fromkeys(OUTCOME_A_GATES, True)

    assert classify(prerequisites=every, gates=passing)["outcome"] == "A"

    null_gates = dict(passing)
    null_gates["decisive_effect_at_least_sesoi"] = False
    assert classify(prerequisites=every, gates=null_gates)["outcome"] == "B"

    broken = dict(every)
    broken["mutations_zero_survivors"] = False
    result = classify(prerequisites=broken, gates=passing)
    assert result["outcome"] == "C", result
    assert result["failed_prerequisites"] == ["mutations_zero_survivors"]

    # A broken prerequisite outranks a passing decisive gate: a program that
    # cannot interpret its measurements does not get to claim a win.
    assert classify(prerequisites=broken, gates=passing)["outcome"] == "C"

    text = terminal_report(classify(prerequisites=every, gates=null_gates), {"decisive": {"effect": 0.0}})
    assert "No unqualified Nous is assigned" in text
    assert "cognitive_material_foundation_complete" in text
    print("genesis publication self-check passed")


if __name__ == "__main__":
    demo()
