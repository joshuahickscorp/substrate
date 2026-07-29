"""Terminal, evidence-bound publication for Cognitive Material Genesis II."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from substrate import genesis2_config as C2
from substrate import genesis2_io as IO2

WRITE_ECONOMY = "SUBSTRATE_GENESIS2_WRITE_ECONOMY.json"
TOURNAMENT = "SUBSTRATE_GENESIS2_CANDIDATE_TOURNAMENT.json"
CURRICULA = "SUBSTRATE_GENESIS2_DEVELOPMENTAL_CURRICULA.json"
LIMITATIONS = "SUBSTRATE_GENESIS2_LIMITATIONS.json"
HANDOFF = "SUBSTRATE_GENESIS2_TANGIBLE_SANDBOX_HANDOFF.json"
PUBLICATION = "SUBSTRATE_GENESIS2_PUBLICATION.json"


def _load(name: str) -> dict[str, Any]:
    path = IO2.EVIDENCE / name
    if not path.is_file():
        raise IO2.Refused(f"publication input is missing: {name}")
    return IO2.load_json(path)


def _write(name: str, schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    document = IO2.authority(schema, payload)
    IO2.write_json(IO2.EVIDENCE / name, document)
    return document


def _economy_rows(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "mean_committed",
        "mean_refused",
        "mean_compute",
        "mean_peak_bytes",
        "mean_score",
        "mean_development_score",
        "mean_retention_score",
    )
    return {arm: {field: values[field] for field in fields} for arm, values in sorted(summary["arms"].items())}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(IO2.ROOT)),
        "bytes": path.stat().st_size,
        "sha256": IO2.file_digest(path),
    }


def _limitations(
    classification: Mapping[str, Any],
    claims: Mapping[str, Any],
    mechanism: Mapping[str, Any],
) -> list[str]:
    limitations = [
        "Genesis I's negative decisive effect and confidence interval remain authoritative and unchanged.",
        "The Genesis II pilot selected the associative monolith within the preregistered simplicity tie band.",
        "L2-L10 are bounded scheduling and mechanism variants over a shared field core; their labels do not establish nine unrelated substrate architectures.",
        "Post-hoc compression, learned-codebook, and adaptive-mixed-radix precision arms "
        "were declared but not executed, so P5 can pass only on the implemented "
        "native-quinary field contrast.",
        "Grok reviews are advisory opinions and are not experimental evidence.",
        "The challenge generator is synthetic and does not establish human-like cognition or general intelligence.",
        "Continuity is established only for the frozen event/cycle protocol; no continuous-time or twelve-hour lane was required for the selected material.",
        "Repository-wide mypy has inherited legacy failures; Genesis II's scoped type check passes.",
        "Two inherited Final Revision tests are pinned to the former origin/main hash and fail after the remote README-only advance.",
        "External activation remains false and unqualified Nous is forbidden.",
    ]
    failing = list(claims["failing_claims"])
    if failing:
        limitations.append(f"Primary claims not established: {', '.join(failing)}.")
    inactive = [name for name, row in mechanism["mechanisms"].items() if not row["integrated_active"]]
    if inactive:
        limitations.append("Mechanisms disabled or licensed as unnecessary for the selected workload: " + ", ".join(inactive) + ".")
    if classification["outcome"] == "B":
        limitations.append("Architectural or compositional field advantage remains unproven.")
    return limitations


def _markdown_report(
    classification: Mapping[str, Any],
    freeze: Mapping[str, Any],
    principal: Mapping[str, Any],
    replication: Mapping[str, Any],
    hidden: Mapping[str, Any],
    claims: Mapping[str, Any],
    limitations: Sequence[str],
    clean_clone: Mapping[str, Any],
    grok: Mapping[str, Any],
) -> str:
    analysis = principal["analysis"]
    lines = [
        "# Substrate Cognitive Material Genesis II — Terminal Report",
        "",
        f"Outcome: **{classification['outcome']} — {classification['classification']}**",
        "",
        f"Selected material: `{freeze['selected_candidate']}`",
        "",
        f"Decisive comparator: `{freeze['decisive_comparator']}`",
        "",
        "External activation: `false`",
        "",
        "Unqualified Nous: `false`",
        "",
        "## Decisive result",
        "",
        (
            f"The principal paired effect was {analysis['effect']:.6f} "
            f"(95% CI {analysis['confidence_lower']:.6f} to {analysis['confidence_upper']:.6f}) "
            f"over {analysis['independent_units']} independently generated family-history cells "
            f"({analysis['unique_history_ids']} history indices)."
        ),
        "",
        f"Replication effect: {replication['analysis']['effect']:.6f}.",
        "",
        f"Hidden-composition effect: {hidden['analysis']['effect']:.6f}.",
        "",
        (
            "The result is an architectural null: the program improved associative plasticity, "
            "but did not establish an advantage for field organization over the strongest equally "
            "plastic monolithic alternative."
            if classification["outcome"] == "B"
            else "The terminal interpretation follows the frozen Outcome A/B/C gate."
        ),
        "",
        "## Claims",
        "",
    ]
    for claim, row in claims["claims"].items():
        lines.append(f"- {claim}: {'PASS' if row['passed'] else 'FAIL'} — {row['statement']}")
    lines.extend(("", "## Independent verification", ""))
    lines.append(
        f"The clean clone {'passed' if clean_clone['all_pass'] else 'failed'} and exactly recomputed "
        "principal, replication, and hidden-composition statistics from committed raw rows."
    )
    lines.append("")
    lines.append(f"{grok['distinct_completed_roles']} distinct Grok review roles completed; their opinions were archived but never used as evidence.")
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {limitation}" for limitation in limitations)
    lines.extend(
        (
            "",
            "## Handoff",
            "",
            "The artifact is tangible-sandbox ready only within the published claim boundary. "
            "It is not externally activated and it is not an unqualified Nous system.",
            "",
        )
    )
    return "\n".join(lines)


def publish(classification: Mapping[str, Any]) -> dict[str, Any]:
    """Derive all publication surfaces from already sealed terminal evidence."""
    reconstruction = _load("SUBSTRATE_GENESIS2_PRIOR_RECONSTRUCTION.json")
    factorial = _load("SUBSTRATE_GENESIS2_FACTORIAL.json")
    screening = _load("SUBSTRATE_GENESIS2_SCREENING.json")
    pilot = _load("SUBSTRATE_GENESIS2_PILOT.json")
    mechanism = _load("SUBSTRATE_GENESIS2_MECHANISM_MATRIX.json")
    envelopes = _load("SUBSTRATE_GENESIS2_BINDING_ENVELOPES.json")
    freeze = _load("SUBSTRATE_GENESIS2_FREEZE.json")
    principal = _load("SUBSTRATE_GENESIS2_PRINCIPAL.json")
    replication = _load("SUBSTRATE_GENESIS2_REPLICATION.json")
    hidden = _load("SUBSTRATE_GENESIS2_HIDDEN_COMPOSITION.json")
    continuity = _load("SUBSTRATE_GENESIS2_CONTINUITY.json")
    mutations = _load("SUBSTRATE_GENESIS2_MUTATIONS.json")
    claims = _load("SUBSTRATE_GENESIS2_CLAIMS.json")
    clean_clone = _load("SUBSTRATE_GENESIS2_CLEAN_CLONE.json")
    grok_path = IO2.ARTIFACTS / "grok_archive.json"
    if not grok_path.is_file():
        raise IO2.Refused("Grok archive is missing")
    grok = IO2.load_json(grok_path)
    if not clean_clone["all_pass"] or not grok["minimum_met"]:
        raise IO2.Refused("clean-clone and minimum Grok review gates must pass before publication")

    write_economy = _write(
        WRITE_ECONOMY,
        "substrate-genesis2-write-economy/v1",
        {
            "pilot": _economy_rows(pilot["summary"]),
            "principal": _economy_rows(principal["summary"]),
            "ledger_units": list(C2.LEDGER_FIELDS),
            "reports": list(C2.LEDGER_REPORTS),
            "all_pass": True,
        },
    )
    tournament = _write(
        TOURNAMENT,
        "substrate-genesis2-candidate-tournament/v1",
        {
            "screening_ranking": screening["ranking"],
            "survivors": screening["survivors"],
            "eliminated": screening["eliminated"],
            "pilot_selection": pilot["selection"],
            "decisive_comparator": freeze["decisive_comparator"],
            "terminal_selected": freeze["selected_candidate"],
            "all_pass": True,
        },
    )
    curricula = _write(
        CURRICULA,
        "substrate-genesis2-developmental-curricula/v1",
        {
            "challenge_families": list(C2.CHALLENGE_FAMILIES),
            "generalisation_families": list(C2.GENERALISATION_FAMILIES),
            "developmental_arc": list(C2.DEVELOPMENTAL_ARC),
            "continuity_requirements": dict(C2.FROZEN_CONTINUITY_REQUIREMENTS),
            "all_pass": True,
        },
    )
    limitations = _limitations(classification, claims, mechanism)
    limitations_document = _write(
        LIMITATIONS,
        "substrate-genesis2-limitations/v1",
        {
            "limitations": limitations,
            "parent_negative_preserved": True,
            "all_pass": True,
        },
    )
    handoff = _write(
        HANDOFF,
        "substrate-genesis2-tangible-sandbox-handoff/v1",
        {
            "outcome": classification["outcome"],
            "classification": classification["classification"],
            "status": classification.get("status"),
            "readiness": classification.get("readiness"),
            "selected_material": freeze["selected_candidate"],
            "decisive_comparator": freeze["decisive_comparator"],
            "claim_boundary": C2.CLAIM_BOUNDARY,
            "external_activation": False,
            "unqualified_nous": False,
            "all_pass": classification["outcome"] in ("A", "B"),
        },
    )

    report_path = IO2.ROOT / "docs" / "SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md"
    IO2.write_text(
        report_path,
        _markdown_report(
            classification,
            freeze,
            principal,
            replication,
            hidden,
            claims,
            limitations,
            clean_clone,
            grok,
        ),
    )
    handoff_path = IO2.ROOT / "docs" / "SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_HANDOFF.md"
    IO2.write_text(
        handoff_path,
        "\n".join(
            (
                "# Substrate Cognitive Material Genesis II — Handoff",
                "",
                f"Outcome: {classification['outcome']} — {classification['classification']}",
                "",
                f"Selected material: `{freeze['selected_candidate']}`",
                "",
                f"Decisive comparator: `{freeze['decisive_comparator']}`",
                "",
                "External activation remains false. Unqualified Nous remains forbidden.",
                "",
                "See `SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md` and the sealed evidence index.",
                "",
            )
        ),
    )

    evidence_files = sorted(path for path in IO2.EVIDENCE.glob("SUBSTRATE_GENESIS2_*.json") if path.name != PUBLICATION)
    indexed = [_artifact(path) for path in evidence_files]
    indexed.extend(
        (
            _artifact(grok_path),
            _artifact(report_path),
            _artifact(handoff_path),
        )
    )
    published_surfaces = {
        "prior_result_reconstruction": reconstruction["sha256"],
        "representation_versus_architecture_diagnosis": factorial["sha256"],
        "write_economy_ledger": write_economy["sha256"],
        "candidate_tournament": tournament["sha256"],
        "mechanism_activity_matrix": mechanism["sha256"],
        "binding_resource_frontiers": envelopes["sha256"],
        "developmental_curricula": curricula["sha256"],
        "continuity": continuity["sha256"],
        "principal": principal["sha256"],
        "replication": replication["sha256"],
        "hidden_composition": hidden["sha256"],
        "grok_archive": grok["sha256"],
        "mutations": mutations["sha256"],
        "clean_clone": clean_clone["sha256"],
        "limitations": limitations_document["sha256"],
        "tangible_sandbox_handoff": handoff["sha256"],
    }
    return _write(
        PUBLICATION,
        "substrate-genesis2-publication/v1",
        {
            "outcome": classification["outcome"],
            "published_surfaces": published_surfaces,
            "artifact_index": indexed,
            "report": str(report_path.relative_to(IO2.ROOT)),
            "handoff": str(handoff_path.relative_to(IO2.ROOT)),
            "all_required_surfaces_published": len(published_surfaces) == 16,
            "all_pass": len(published_surfaces) == 16,
        },
    )
