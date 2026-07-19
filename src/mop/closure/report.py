"""Render a human-readable markdown closure report from a sealed closure artifact.

The report states the completed chain, the one-number result, the evidence class of each conclusion,
the strongest killing control, the surviving and pruned lane counts, the explicit refusals, and the
exact next queued program. It reads only the closure artifact; it derives nothing new and asserts no
positive that the artifact does not already carry.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any


def _fmt_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _counts_line(counts: dict[str, Any]) -> str:
    order = ["carried", "surviving", "pruned", "warned", "failed", "blocked", "untested"]
    parts = [f"{name}={counts.get(name, 0)}" for name in order]
    return ", ".join(parts)


def render_closure_report(closure_artifact: dict[str, Any]) -> str:
    """Return the closure report as a markdown string."""

    admitted = bool(closure_artifact.get("admitted"))
    closure_status = closure_artifact.get("closure_status", "unknown")
    gr_state = closure_artifact.get("general_run_state")
    lineage = closure_artifact.get("terminal_lineage") or {}
    derivation_status = lineage.get("derivation_status", "unknown")
    lane_universe = lineage.get("lane_universe") or {}
    counts = lane_universe.get("counts") or {}
    d1 = lineage.get("d1_status") or {}
    i1 = lineage.get("g1_i1_status") or {}
    one_number = lineage.get("one_number_result") or {}
    evidence = lineage.get("evidence_classes") or {}
    hypotheses = lineage.get("surviving_hypotheses") or []
    refusals = lineage.get("refusals") or closure_artifact.get("refusals") or {}
    next_question = lineage.get("next_bounded_scientific_question") or {}
    next_program = lineage.get("next_queued_program") or {}
    admission = closure_artifact.get("admission") or {}
    admission_refusals = admission.get("refusals") or []

    lines: list[str] = []
    lines.append("# Generation 1 General Run Closure")
    lines.append("")
    lines.append(f"- Closure status: `{closure_status}`")
    lines.append(f"- Admitted: {_fmt_bool(admitted)}")
    lines.append(f"- General Run state: `{gr_state}`")
    lines.append(f"- Terminal-lineage derivation: `{derivation_status}`")
    lines.append("")

    lines.append("## One-number result")
    lines.append("")
    value = one_number.get("value")
    unit = one_number.get("unit", "")
    lines.append(f"**{value} {unit}.** {one_number.get('note', '')}".strip())
    lines.append("")

    lines.append("## Completed chain")
    lines.append("")
    for program in lineage.get("terminal_inputs") or []:
        lines.append(f"- terminal: `{program}`")
    for program in lineage.get("nonterminal_inputs") or []:
        lines.append(f"- not terminal (queued or running): `{program}`")
    lines.append("")

    lines.append("## Lane accounting")
    lines.append("")
    lines.append(_counts_line(counts))
    lines.append("")
    lines.append(f"- surviving lanes: {', '.join(lane_universe.get('surviving') or []) or 'none'}")
    lines.append(f"- pruned lanes: {', '.join(lane_universe.get('pruned') or []) or 'none'}")
    lines.append(f"- blocked lanes: {', '.join(lane_universe.get('blocked') or []) or 'none'}")
    lines.append(f"- untested lanes: {', '.join(lane_universe.get('untested') or []) or 'none'}")
    lines.append("")

    lines.append("## D1 and G1-I1")
    lines.append("")
    lines.append(
        f"- D1: retired={_fmt_bool(d1.get('retired'))}, route=`{d1.get('route')}`, "
        f"candidate_evidence_count={d1.get('candidate_evidence_count')}, "
        f"redesign_authorized={_fmt_bool(d1.get('redesign_execution_authorized'))}"
    )
    lines.append(
        f"- G1-I1: route=`{i1.get('route')}`, "
        f"executed_as_integration_efficacy_claim="
        f"{_fmt_bool(i1.get('executed_as_integration_efficacy_claim'))}"
    )
    lines.append("")

    lines.append("## Evidence class per conclusion")
    lines.append("")
    for name in sorted(evidence):
        block = evidence[name] or {}
        lines.append(f"- `{name}`: {block.get('evidence_class')} - {block.get('statement')}")
    lines.append("")

    lines.append("## Strongest killing control or remaining blocker")
    lines.append("")
    for item in hypotheses:
        lines.append(f"- {item.get('hypothesis')} ({item.get('status')})")
        lines.append(f"  - {item.get('strongest_control_or_blocker')}")
    lines.append("")

    lines.append("## Refusals")
    lines.append("")
    lines.append(f"- activation_allowed: {_fmt_bool(refusals.get('activation_allowed'))}")
    lines.append(f"- scientific_promotion: {_fmt_bool(refusals.get('scientific_promotion'))}")
    lines.append(f"- natural_world_generality: {_fmt_bool(refusals.get('natural_world_generality'))}")
    lines.append(
        "- independent_scientific_confirmation: "
        f"{_fmt_bool(refusals.get('independent_scientific_confirmation'))}"
    )
    if refusals.get("statement"):
        lines.append(f"- {refusals.get('statement')}")
    if admission_refusals:
        lines.append("")
        lines.append("Admission refusals (why the closure is deferred):")
        for refusal in admission_refusals:
            lines.append(f"- {refusal}")
    lines.append("")

    lines.append("## Next bounded scientific question")
    lines.append("")
    lines.append(next_question.get("question", ""))
    lines.append("")
    lines.append(f"Current answer: {next_question.get('current_answer', '')}")
    natural_world_axis = next_question.get("natural_world_axis")
    if natural_world_axis:
        lines.append("")
        lines.append(natural_world_axis)
    lines.append("")

    lines.append("## Next queued program")
    lines.append("")
    lines.append(f"- while the General Run is active: {next_program.get('while_general_run_active', '')}")
    lines.append(
        f"- on the General Run reaching a clean terminal: `{next_program.get('on_general_run_terminal', '')}`"
    )
    lines.append("")

    return "\n".join(lines)


__all__ = ["render_closure_report"]
