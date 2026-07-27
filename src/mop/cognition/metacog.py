"""Metacognition and endogenous attention.

Section 10 lists eleven things metacognition governs and then says to start with simple policies, because
a learned controller with nothing to beat is an expensive way to reproduce a threshold. So the simple
policies are here, the learned one is refused without measured oracle headroom over the best of them, and
the eight required measures are computed from paired decisions and outcomes rather than declared.

Two of those measures are the ones a flattering report leaves out. Unnecessary thought counts the times
the system verified something that was already right, and missed verification counts the times it did not
verify something that was wrong. A policy that always verifies looks perfect on one and terrible on the
other, which is exactly the tradeoff that should be visible.

House style: no dashes.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field

from mop.cognition import io

SESOI = 0.05

# section 10, what metacognition governs
ACTIONS = ("continue", "stop", "verify", "retrieve", "simulate", "switch_perspective", "invoke_tool",
           "request_external_evidence", "defer", "revise", "preserve_uncertainty")

# section 10, what to measure
MEASURES = ("decision_quality", "correction", "compute", "latency", "unnecessary_thought",
            "missed_verification", "calibration", "transfer")

# section 15.4, what endogenous attention selects on
DRIVERS = ("goal_relevance", "uncertainty", "risk", "expected_value", "novelty", "contradiction")


class Refused(RuntimeError):
    """A metacognitive policy that is not licensed to open."""


@dataclass
class Situation:
    """One decision point, with the outcome that was actually observed beside it."""
    confidence: float
    novelty: float = 0.0
    contradiction: bool = False
    correct_without_verification: bool = True
    verification_cost: float = 1.0
    verification_fixes: bool = True


# ---------------------------------------------------------------- 10 policies


@dataclass(frozen=True)
class Policy:
    name: str
    information_used: frozenset
    decide: callable = field(repr=False, default=None)
    learned: bool = False


def _always_continue(s: Situation, budget: float) -> str:
    return "continue"


def _always_verify(s: Situation, budget: float) -> str:
    return "verify" if budget >= s.verification_cost else "continue"


def _threshold(s: Situation, budget: float, cut: float = 0.7) -> str:
    if s.confidence < cut and budget >= s.verification_cost:
        return "verify"
    return "continue"


def _contradiction_first(s: Situation, budget: float) -> str:
    if s.contradiction and budget >= s.verification_cost:
        return "verify"
    return "preserve_uncertainty" if s.confidence < 0.5 else "continue"


def _novelty_retrieval(s: Situation, budget: float) -> str:
    if s.novelty > 0.6 and budget >= s.verification_cost:
        return "retrieve"
    return _threshold(s, budget)


def _oracle(s: Situation, budget: float) -> str:
    # uses the outcome, which nobody has at decision time. An upper bound, not a candidate.
    if not s.correct_without_verification and s.verification_fixes and budget >= s.verification_cost:
        return "verify"
    return "continue"


POLICIES: tuple[Policy, ...] = (
    Policy("never_verify", frozenset(), _always_continue),
    Policy("always_verify", frozenset({"budget"}), _always_verify),
    Policy("confidence_threshold", frozenset({"confidence", "budget"}), _threshold),
    Policy("contradiction_first", frozenset({"contradiction", "confidence", "budget"}),
           _contradiction_first),
    Policy("novelty_retrieval", frozenset({"novelty", "confidence", "budget"}), _novelty_retrieval),
    Policy("oracle", frozenset({"outcome"}), _oracle),
    Policy("learned", frozenset({"full_history"}), None, learned=True),
)

BY_POLICY = {p.name: p for p in POLICIES}
SIMPLE = tuple(p for p in POLICIES if not p.learned and p.name != "oracle")


def evaluate(policy_name: str, situations: list[Situation], budget: float = 1e9) -> dict:
    """The eight measures, computed from what the policy decided and what was actually true."""
    policy = BY_POLICY.get(policy_name)
    if policy is None or policy.decide is None:
        raise Refused(f"policy {policy_name!r} cannot be evaluated without an implementation")
    remaining = budget
    correct, compute, unnecessary, missed, corrections = 0, 0.0, 0, 0, 0
    for s in situations:
        action = policy.decide(s, remaining)
        verified = action in ("verify", "retrieve", "request_external_evidence")
        if verified:
            remaining -= s.verification_cost
            compute += s.verification_cost
        right = s.correct_without_verification or (verified and s.verification_fixes)
        correct += bool(right)
        if verified and s.correct_without_verification:
            unnecessary += 1
        if not verified and not s.correct_without_verification:
            missed += 1
        if verified and not s.correct_without_verification and s.verification_fixes:
            corrections += 1
    n = max(len(situations), 1)
    return {
        "policy": policy_name,
        "decision_quality": round(correct / n, 6),
        "correction": round(corrections / n, 6),
        "compute": round(compute, 6),
        "latency": round(compute / n, 6),
        "unnecessary_thought": round(unnecessary / n, 6),
        "missed_verification": round(missed / n, 6),
        "calibration": round(1.0 - statistics.fmean(
            abs(s.confidence - float(s.correct_without_verification)) for s in situations), 6),
        "transfer": None,  # only a second bed can fill this, and it is not invented here
        "budget_left": round(remaining, 6) if budget < 1e9 else None,
    }


def headroom(situations: list[Situation], budget: float = 1e9) -> dict:
    """Oracle metacognition against the strongest simple policy. This is what a learned policy must beat."""
    oracle = evaluate("oracle", situations, budget)["decision_quality"]
    simple = {p.name: evaluate(p.name, situations, budget)["decision_quality"] for p in SIMPLE}
    best = max(simple.values())
    return {"oracle": oracle, "per_simple_policy": simple, "best_simple": best,
            "residual": round(oracle - best, 6), "n_situations": len(situations)}


def select_policy(name: str, *, oracle_headroom: dict | None = None, sesoi: float = SESOI) -> Policy:
    policy = BY_POLICY.get(name)
    if policy is None:
        raise Refused(f"unknown metacognitive policy {name!r}")
    if not policy.learned:
        return policy
    residual = (oracle_headroom or {}).get("residual")
    if residual is None or residual <= sesoi:
        raise Refused(
            "learned metacognition opens only above measured oracle headroom over the strongest simple "
            f"policy; residual is {residual!r} against a SESOI of {sesoi}")
    return policy


# ---------------------------------------------------------------- 15.4 endogenous attention


def attend(candidates: list[dict], *, weights: dict | None = None, budget: float) -> dict:
    """Rank what deserves thought, then stop at the resource limit rather than at the end of the list."""
    weights = weights or {d: 1.0 for d in DRIVERS}
    unknown = set(weights) - set(DRIVERS)
    if unknown:
        raise Refused(f"undeclared attention driver {sorted(unknown)}")
    scored = []
    for c in candidates:
        missing = [d for d in DRIVERS if d not in c]
        score = sum(weights.get(d, 0.0) * float(c.get(d, 0.0)) for d in DRIVERS)
        scored.append({"id": c["id"], "score": round(score, 6), "cost": float(c.get("cost", 1.0)),
                       "undeclared_drivers": missing})
    scored.sort(key=lambda r: (-r["score"], r["id"]))
    attended, spent = [], 0.0
    for row in scored:
        if spent + row["cost"] > budget:
            continue
        attended.append(row["id"])
        spent += row["cost"]
    return {"ranked": scored, "attended": attended, "spent": round(spent, 6), "budget": budget,
            "dropped_for_budget": [r["id"] for r in scored if r["id"] not in attended],
            "drivers": list(DRIVERS)}


# ---------------------------------------------------------------- declaration


def _probe(n: int = 60) -> list[Situation]:
    """A situation stream where verification sometimes matters and sometimes does not."""
    out = []
    for i in range(n):
        wrong = i % 3 == 0
        out.append(Situation(confidence=0.9 if not wrong else 0.4,
                             novelty=(i % 7) / 7, contradiction=i % 5 == 0,
                             correct_without_verification=not wrong,
                             verification_cost=1.0, verification_fixes=i % 6 != 0))
    return out


def declaration() -> dict:
    probe = _probe()
    scores = {p.name: evaluate(p.name, probe) for p in POLICIES if p.decide is not None}
    h = headroom(probe)
    return {
        "schema": "substrate-metacognition/v1",
        "governs": list(ACTIONS),
        "measures": list(MEASURES),
        "policies": [{"name": p.name, "information_used": sorted(p.information_used),
                      "learned": p.learned, "implemented": p.decide is not None} for p in POLICIES],
        "simple_first": [p.name for p in SIMPLE],
        "oracle_uses_information_unavailable_at_decision_time": True,
        "learned_rule": ("learned metacognition opens only above measured oracle headroom over the "
                         "strongest simple policy"),
        "measured_on_a_probe_stream": scores,
        "oracle_headroom": h,
        "learned_currently_licensed": h["residual"] > SESOI,
        "transfer_note": ("transfer is null in every row above because it needs a second bed, and "
                          "inventing a number for it would be the defect this program exists to stop"),
        "attention": {"drivers": list(DRIVERS),
                      "rule": "ranking is by declared drivers and stops at the resource limit"},
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_METACOGNITION.json", doc)
    print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                      "oracle_headroom": doc["oracle_headroom"]["residual"],
                      "learned_licensed": doc["learned_currently_licensed"]}, indent=2))


if __name__ == "__main__":
    main()
