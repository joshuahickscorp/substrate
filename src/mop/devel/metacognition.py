
from __future__ import annotations

from .. import provenance
from .north_star import DEVELOPMENTAL_LOOP, assert_no_sentience_claims, safety_rail_note
from .registries import load_capacities

FAILURE_CATEGORIES = ("data", "representation", "memory", "plasticity", "optimization", "evaluation")


def report(
    curriculum_manifest: dict | None = None,
    ablation_plan: dict | None = None,
    cache_provenance: dict | None = None,
) -> dict:
    rungs = load_capacities()
    by_status: dict[str, list] = {}
    for r in rungs:
        by_status.setdefault(r["status"], []).append(r["name"])

    can_decode = [r["name"] for r in rungs if r["status"] == "measured-now"]
    cannot_yet = [r["name"] for r in rungs if r["status"] in ("studio-later", "deferred")]
    prototyping = [r["name"] for r in rungs if r["status"] == "prototype-local"]

    wants_next = None
    if curriculum_manifest and curriculum_manifest.get("chosen"):
        wants_next = {
            "study": curriculum_manifest["chosen"],
            "why": curriculum_manifest.get("reason"),
            "capacity": curriculum_manifest.get("expected_capacity"),
            "rejected_noise": curriculum_manifest.get("rejected_noise", []),
        }
    next_mechanism = None
    if ablation_plan and ablation_plan.get("next_best"):
        nb = ablation_plan["next_best"]
        next_mechanism = {"slug": nb["slug"], "tests": nb["metric"], "eig_per_hour": nb["eig_per_hour"]}

    return {
        "north_star": DEVELOPMENTAL_LOOP,
        "capacities_by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "what_i_can_decode": sorted(can_decode),
        "what_i_cannot_decode_yet": sorted(cannot_yet),
        "what_i_am_prototyping": sorted(prototyping),
        "uncertainty_note": (
            "self-monitoring rung (calibration + noisy-TV epistemic collapse) is measured-now; "
            "calibration ECE and OOD uncertainty are the operational confidence signals"
        ),
        "what_data_i_want_next": wants_next,
        "which_mechanism_next": next_mechanism,
        "failure_attribution_categories": list(FAILURE_CATEGORIES),
        "claim_ledger": {
            "real": "rungs tagged measured-now with a passing gate (e.g. sensory grounding, plastic "
            "adaptation/E1, episodic memory, self-monitoring, provenance continuity)",
            "provisional": "frozen-random substrate / synthetic-content results; tiny-scale probes",
            "auxiliary": "comparison/distilled/quantized encoders; never the canonical V-JEPA result",
            "deferred": "studio-later / deferred rungs (object slots, world-model, language-mediated)",
        },
        "cache_provenance": cache_provenance,
        "safety_rails": safety_rail_note(),
    }


def render_md(rep: dict) -> str:
    L = [
        "# Metacognition report (structured self-monitoring, not self-awareness)",
        "",
        rep["safety_rails"],
        "",
        f"North star: `{rep['north_star']}`",
        "",
        "## What I can decode now (measured)",
        "",
        *[f"- {c}" for c in rep["what_i_can_decode"]],
        "",
        "## What I am prototyping locally",
        "",
        *[f"- {c}" for c in rep["what_i_am_prototyping"]],
        "",
        "## What I cannot decode yet (deferred / Studio)",
        "",
        *[f"- {c}" for c in rep["what_i_cannot_decode_yet"]],
        "",
        "## What data to acquire next (the curriculum target)",
        "",
        f"- {rep['what_data_i_want_next']}",
        "",
        "## Which mechanism to try next",
        "",
        f"- {rep['which_mechanism_next']}",
        "",
        "## Uncertainty",
        "",
        f"- {rep['uncertainty_note']}",
        "",
        "## Claim ledger (capacity measurements vs philosophy)",
        "",
        *[f"- **{k}**: {v}" for k, v in rep["claim_ledger"].items()],
        "",
        f"Failure is attributed to one of: {rep['failure_attribution_categories']} (never to feelings).",
        "",
    ]
    text = "\n".join(L)
    assert_no_sentience_claims(text, where="metacognition report")  # the rail gates the artifact
    return text


def metacognition_provenance(seed: int = 0) -> dict:
    return provenance.provenance(
        seed=seed, device="cpu", result_tag="provisional", extra={"stage": "metacognition"}
    )
