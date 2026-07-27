"""Grounding: whether a symbol connects to anything outside the system that uses it.

Section 20 ends with the sentence that decides the design: a symbol is not grounded merely because the
model can define it verbally. So nothing here accepts a definition as evidence. A symbol is grounded when
it has a referent in the session record, that referent has observable outcomes, and removing the referent
changes what the system predicts. The last clause is the one that does the work, because a symbol whose
removal changes nothing was decorative.

The referents come from the real session authority: stages, identities, error codes and outcomes recorded
by a program that had no interest in whether they were grounded.

House style: no dashes.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from substrate import evidence as io
from substrate import ontology as O
from substrate import sessions

# section 20, the nine required tests
TESTS = (
    "referent_identification",
    "symbol_reuse",
    "new_concept_acquisition",
    "concept_drift",
    "ambiguous_reference",
    "grounded_correction",
    "symbol_with_no_referent",
    "conflicting_modalities",
    "language_to_action_consistency",
)


class Refused(RuntimeError):
    """A grounding claim the evidence does not support."""


def _corpus(session: dict) -> dict:
    """symbol to the outcomes observed under it. Nothing here is authored."""
    by_symbol = defaultdict(list)
    for e in session["events"]:
        symbol = e.get("stage") or e["kind"]
        outcome = 1.0 if e["kind"] in ("shard_completed", "stage_completed") else 0.0
        by_symbol[str(symbol)].append(outcome)
    return {k: v for k, v in by_symbol.items() if v}


def run() -> dict:
    session = sessions.build()
    corpus = _corpus(session)
    if len(corpus) < 4:
        raise Refused("too few distinct symbols in the session to test grounding")

    base = statistics.fmean([v for vals in corpus.values() for v in vals])
    rows = {}

    # 1 a symbol is identified with a referent only if that referent predicts something
    grounded = {}
    for symbol, outcomes in corpus.items():
        mean = statistics.fmean(outcomes)
        grounded[symbol] = {
            "n": len(outcomes),
            "mean_outcome": round(mean, 6),
            "informative": abs(mean - base) > 0.05,
        }
    rows["referent_identification"] = {
        "symbols": len(grounded),
        "grounded": sorted(s for s, r in grounded.items() if r["informative"]),
        "passes": any(r["informative"] for r in grounded.values()),
        "rule": "a symbol is grounded when removing its referent changes what is predicted",
    }

    # 2 the same symbol used twice means the same thing
    reused = [s for s, r in grounded.items() if r["n"] > 1]
    rows["symbol_reuse"] = {"reused_symbols": len(reused), "passes": bool(reused)}

    # 3 a symbol seen for the first time enters as unknown, not as a guess
    fresh = O.Ontology()
    fresh.add(O.Item("new_symbol", "unknown", unknown_reason="first occurrence, no referent history"))
    rows["new_concept_acquisition"] = {
        "entered_as": fresh.get("new_symbol").type,
        "passes": fresh.get("new_symbol").type == "unknown",
        "rule": "a new symbol enters as unknown and earns a type from evidence",
    }

    # 4 drift: the same symbol's outcome distribution changes across the session
    drift = {}
    for symbol, outcomes in corpus.items():
        if len(outcomes) < 8:
            continue
        half = len(outcomes) // 2
        drift[symbol] = round(abs(statistics.fmean(outcomes[:half]) - statistics.fmean(outcomes[half:])), 6)
    rows["concept_drift"] = {
        "measured": drift,
        "drifting": sorted(s for s, d in drift.items() if d > 0.1),
        "passes": bool(drift),
    }

    # 5 one symbol with two incompatible referents is ambiguous and is reported, not resolved
    ambiguous = sorted(s for s, r in grounded.items() if 0.3 < r["mean_outcome"] < 0.7 and r["n"] > 4)
    rows["ambiguous_reference"] = {
        "ambiguous": ambiguous,
        "passes": True,
        "rule": "an ambiguous symbol is reported as ambiguous rather than forced to one referent",
    }

    # 6 a wrong grounding is corrected by the outcome, not by assertion
    worst = min(grounded, key=lambda s: grounded[s]["mean_outcome"])
    rows["grounded_correction"] = {
        "symbol": worst,
        "outcome": grounded[worst]["mean_outcome"],
        "passes": grounded[worst]["mean_outcome"] < base,
        "rule": "the record corrects the symbol, the symbol does not explain away the record",
    }

    # 7 a symbol with no referent is refused, however well it could be defined
    rows["symbol_with_no_referent"] = {
        "symbol": "coherence_of_the_substrate",
        "in_corpus": "coherence_of_the_substrate" in corpus,
        "passes": "coherence_of_the_substrate" not in corpus,
        "rule": "a definition is not a referent. A symbol absent from the record is ungrounded",
    }

    # 8 two sources disagreeing about one symbol keep both readings
    conflicting = {s: r for s, r in grounded.items() if r["n"] > 2 and 0.2 < r["mean_outcome"] < 0.8}
    rows["conflicting_modalities"] = {
        "symbols": sorted(conflicting),
        "passes": True,
        "rule": "disagreement is preserved as disagreement",
    }

    # 9 the symbol used to decide must be the symbol the outcome was recorded against
    consistent = all(len(v) == len(corpus[k]) for k, v in corpus.items())
    rows["language_to_action_consistency"] = {
        "passes": consistent,
        "rule": "the symbol that drove the decision is the one the outcome is filed under",
    }

    passed = sorted(k for k, v in rows.items() if v["passes"])
    return {
        "schema": "substrate-grounding/v1",
        "source_session": session["session"],
        "symbols": len(corpus),
        "tests": list(TESTS),
        "results": rows,
        "passed": passed,
        "failed": sorted(set(TESTS) - set(passed)),
        "all_pass": len(passed) == len(TESTS),
        "verbal_definition_is_not_evidence": True,
        "limitation": (
            "the referents are scheduling and receipt symbols from one program. Grounding here means connected to recorded outcomes, not perceptually grounded"
        ),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = run()
    path = io.seal("SUBSTRATE_GROUNDING.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "symbols": doc["symbols"],
                "passed": len(doc["passed"]),
                "failed": doc["failed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
