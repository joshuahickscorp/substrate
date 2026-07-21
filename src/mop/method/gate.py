"""The admission gate: the part that makes an invalid experiment unable to reach principal execution.

The previous method validated instrumentation after expensive scientific execution, so an invalid control
could consume a campaign before anyone found out. Here the order is inverted. Every stage below opens only
when the previous one passed, and the stages that cost nothing come first.

The second job of this module is classification. A result is only a mechanism null when the instrument, the
bed, the mechanism, the baseline and the power all held. Otherwise the failure belongs to the method, and
saying so is the whole point.

House style: no dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mop.method.contracts import Contract

# the sequence from the reformed philosophy, cheap and blocking stages first
SEQUENCE = (
    "scientific_premise",
    "causal_model",
    "measurement_model",
    "instrument_calibration",
    "arm_distinctness",
    "control_semantics",
    "bed_validity",
    "baseline_convergence",
    "oracle_headroom",
    "power_and_units",
    "scout",
    "independent_reproduction",
    "principal",
    "independent_verification",
    "mutation_attacks",
    "terminal_classification",
    "hypothesis_update",
)

CONTRACT_STAGE = {
    "ExperimentQuestion": "scientific_premise",
    "CausalModel": "causal_model",
    "MeasurementModel": "measurement_model",
    "InstrumentContract": "instrument_calibration",
    "ArmContract": "arm_distinctness",
    "ControlContract": "control_semantics",
    "DatasetContract": "bed_validity",
    "IndependentUnitContract": "bed_validity",
    "BaselineContract": "baseline_convergence",
    "OracleContract": "oracle_headroom",
    "PowerContract": "power_and_units",
    "ExecutionContract": "principal",
    "VerificationContract": "independent_verification",
    "MutationContract": "mutation_attacks",
    "ResultContract": "terminal_classification",
    "ClaimContract": "terminal_classification",
}

# stages that must be green before any principal compute is spent
PRE_PRINCIPAL = SEQUENCE[: SEQUENCE.index("principal")]


@dataclass
class Preregistration:
    experiment_id: str
    title: str
    contracts: list[Contract] = field(default_factory=list)
    mechanism_activity: dict | None = None
    stop_switch: str = "~/.mop_experimental_method_stop"

    def by_stage(self) -> dict[str, list[Contract]]:
        out: dict[str, list[Contract]] = {s: [] for s in SEQUENCE}
        for c in self.contracts:
            out[CONTRACT_STAGE.get(c.kind, "terminal_classification")].append(c)
        return out

    def admit(self, stage: str = "principal") -> dict:
        """Walk the sequence up to stage. The first failing stage blocks everything after it."""
        by = self.by_stage()
        report, blocked_at, blocking = {}, None, []
        limit = SEQUENCE.index(stage)
        for s in SEQUENCE[: limit + 1]:
            viol = [v for c in by[s] for v in c.violations()]
            if s == "arm_distinctness" and not by[s]:
                viol.append("arm_distinctness: no arm contracts declared, distinctness was never proven")
            if s == "control_semantics" and self.mechanism_activity is not None:
                if not self.mechanism_activity.get("active"):
                    viol.append(
                        f"mechanism_activity: {self.mechanism_activity.get('classification')} "
                        f"({self.mechanism_activity.get('failed')})"
                    )
            report[s] = {
                "contracts": [c.name for c in by[s]],
                "violations": viol,
                "passed": not viol,
                "reached": blocked_at is None,
            }
            if viol and blocked_at is None:
                blocked_at, blocking = s, viol
        licensed = blocked_at is None
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "requested_stage": stage,
            "stages": report,
            "blocked_at": blocked_at,
            "blocking_violations": blocking,
            "licensed": licensed,
            "principal_execution_licensed": licensed and stage == "principal",
        }

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "sequence": list(SEQUENCE),
            "contracts": [c.as_dict() for c in self.contracts],
            "mechanism_activity": self.mechanism_activity,
            "stop_switch": self.stop_switch,
            "admission": self.admit(),
        }


# ---------------------------------------------------------------- terminal classification


def classify_result(*, effect: dict, instrument_valid: bool, bed_valid: bool, mechanism_active: bool,
                    baseline_valid: bool, estimator_sufficient: bool, verifier_agrees: bool,
                    mutations_rejected: bool, implementations_agreeing: int = 1,
                    cost_adjusted_pass: bool = True) -> dict:
    """The positive and null evidence standards, applied in the only order that is safe.

    Method failures are checked before scientific ones, because a method failure makes the scientific
    question unanswered rather than answered in the negative.
    """
    for ok, cls, why in (
        (instrument_valid, "invalid_instrument", "the instrument did not pass its own calibration"),
        (bed_valid, "invalid_bed", "the bed is not a valid principal bed"),
        (mechanism_active, "inactive_mechanism", "the mechanism produced no measurable causal effect"),
        (baseline_valid, "unconverged_baseline", "the baseline was not converged or not the declared one"),
        (estimator_sufficient, "insufficient_power", "the design cannot see an effect of the declared size"),
    ):
        if not ok:
            return {"classification": cls, "reason": why, "scientific": False, "effect": effect}
    v = effect.get("verdict")
    if v == "positive":
        gaps = []
        if implementations_agreeing < 2:
            gaps.append("only one capable implementation agreed")
        if not verifier_agrees:
            gaps.append("the independent scientific verifier did not agree")
        if not mutations_rejected:
            gaps.append("at least one mutation was not rejected")
        if not cost_adjusted_pass:
            gaps.append("the cost adjusted value did not pass")
        if gaps:
            return {
                "classification": "provisional_positive",
                "reason": "; ".join(gaps),
                "scientific": True,
                "effect": effect,
            }
        return {"classification": "positive", "reason": "", "scientific": True, "effect": effect}
    if v == "harm":
        return {"classification": "harm", "reason": "effect below the harm boundary", "scientific": True, "effect": effect}
    if v == "wrong_direction_failure":
        return {"classification": "failure_wrong_direction", "reason": "", "scientific": True, "effect": effect}
    if not verifier_agrees:
        return {
            "classification": "scientifically_unresolved",
            "reason": "the independent verifier did not reproduce the null",
            "scientific": False,
            "effect": effect,
        }
    return {"classification": "mechanism_null", "reason": "", "scientific": True, "effect": effect}


# ---------------------------------------------------------------- coverage


def coverage_gate(statement: float, branch: float, *, scope: list[str], excluded: list[str],
                  target_statement: float = 92.0, target_branch: float = 82.0) -> dict:
    """Coverage misses stay visible. Narrowing the scope to reach the target is itself a finding."""
    return {
        "statement": round(float(statement), 1),
        "branch": round(float(branch), 1),
        "target_statement": target_statement,
        "target_branch": target_branch,
        "statement_met": statement >= target_statement,
        "branch_met": branch >= target_branch,
        "scope": scope,
        "excluded_from_scope": excluded,
        "scope_narrowing_declared": bool(excluded),
        "met": statement >= target_statement and branch >= target_branch,
        "honest_blocker": (
            ""
            if statement >= target_statement and branch >= target_branch
            else f"statement {statement} of {target_statement}, branch {branch} of {target_branch}"
        ),
    }
