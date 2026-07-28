"""Distinct, content-addressable Grok review prompts."""

from __future__ import annotations

from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_io as io

PREFREEZE_ROUND_ASSIGNMENT = {
    **{role: "blind_independent_review" for role in C.REVIEW_CELLS[0:8]},
    **{role: "cross_examination" for role in C.REVIEW_CELLS[8:12]},
    **{role: "architecture_proposals" for role in C.REVIEW_CELLS[12:16]},
    **{role: "test_and_baseline_proposals" for role in C.REVIEW_CELLS[16:20]},
    **{role: "code_and_implementation_review" for role in C.REVIEW_CELLS[20:24]},
    **{role: "post_pilot_review" for role in C.REVIEW_CELLS[24:32]},
}


def prompt_for(role: str, round_identity: str, *, evidence_commit: str, evidence_scope: str) -> str:
    if role not in C.REVIEW_CELLS:
        raise io.Refused(f"unknown Grok review role {role!r}")
    if round_identity not in C.REVIEW_ROUNDS:
        raise io.Refused(f"unknown Grok review round {round_identity!r}")
    facet_lines = "\n".join(f"{index}. {name}" for index, name in enumerate(C.FACETS, start=1))
    special = {
        "closure_null_defender": "Argue the strongest possible case that S2 invalidates an architectural Nous claim.",
        "closure_null_challenger": (
            "Test task compilation, capacity, resource mismatch, answer encoding, saturation, implementation access, and metric design. "
            "You may conclude S2 is fully valid."
        ),
        "architecture_proposals": (
            "Propose a Candidate-H substrate-native architecture with mechanism, state representation, executable bounded prototype, "
            "failure modes, resource costs, and a discriminating ablation. Do not merely rename an existing candidate."
        ),
        "code_and_implementation_review": (
            "Inspect correctness, leakage, architecture, simplicity, performance, security, checkpoint coverage, and claim alignment."
        ),
        "post_pilot_review": "Audit raw decision receipts, controls, oracle headroom, complexity, and every null before recommending selection.",
        "publication_and_claim_boundary_review": "Audit every public sentence against frozen effects and forbid unqualified Nous.",
    }
    special_instruction = special.get(role, special.get(round_identity, "Search for the strongest role-specific falsification."))
    return f"""You are an independent Grok review cell for the Substrate final revision.

ROLE: {role}
ROUND: {round_identity}
PUBLIC EVIDENCE COMMIT: {evidence_commit}
EVIDENCE SCOPE: {evidence_scope}

Do not assume any project claim, receipt, test, benchmark, or narrative is true.
Do not invent repository or execution access. State access limitations exactly.
Historical result `terminal_closed_null` is immutable: candidate minus S2 was
0.0 with 95% CI [0,0]; a tie and any favorable effect below SESOI 0.05 are
nulls. Grok opinion is not a primary endpoint.

ROLE-SPECIFIC MANDATE:
{special_instruction}

Grade these 20 facets independently with binary 0 or 1:
{facet_lines}

Return STRICT JSON only. Required top-level fields:
- role (must equal {role!r})
- round (must equal {round_identity!r})
- evidence_scope
- access_limitations
- assumptions_prohibited
- facets: exactly 20 objects with facet_number 1..20, name,
  score_binary 0|1, discussion_credit 0|0.5|1, rationale
- total_binary_out_of_20
- confidence: low|medium|high
- blocking_defects: array
- nonblocking_concerns: array
- strongest_evidence
- strongest_falsification_evidence
- falsification_tests: nonempty array
- concrete_revisions: array
- recommended_terminal_classification
- minority_or_uncertain_points: array
- candidate_h_proposal: object or null

Every proposed revision must be testable. Preserve minority objections. Do not
turn architecture presence, long prompts, transcript replay, model calls,
digests, modality labels, or reviewer votes into cognitive evidence."""


def prefreeze_manifest(evidence_commit: str) -> list[dict[str, Any]]:
    rows = []
    for role in C.REVIEW_CELLS:
        round_identity = PREFREEZE_ROUND_ASSIGNMENT[role]
        scope = {
            "blind_independent_review": "historical authorities, research ledger, and preflight evidence",
            "cross_examination": "conflicting historical and architectural interpretations",
            "architecture_proposals": "candidate contract, tournament prototypes, and Candidate-H design",
            "test_and_baseline_proposals": "new generator, strongest baselines, fairness, and mutation plan",
            "code_and_implementation_review": "final_revision source, focused tests, checkpoint and activation logic",
            "post_pilot_review": "moderate-pilot raw receipts, decision implementations, headroom, nulls, and resources",
        }[round_identity]
        prompt = prompt_for(role, round_identity, evidence_commit=evidence_commit, evidence_scope=scope)
        rows.append(
            {
                "role": role,
                "round": round_identity,
                "prompt": prompt,
                "prompt_digest": io.digest(prompt),
                "evidence_commit": evidence_commit,
                "evidence_scope": scope,
                "activation": False,
            }
        )
    return rows
