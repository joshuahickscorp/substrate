"""The contract vocabulary of the experiment validity kernel.

Every contract answers one question: is the thing it names proven to be what it was declared to be. A
contract with no evidence never passes. Absence of measurement is a violation, not a default pass, because
the whole class of defects this kernel exists to stop looked exactly like a silent default pass.

Quantities carry a provenance label. A number that was true by construction may not be reported as measured.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- quantity provenance

QUANTITY_KINDS = (
    "measured",  # produced by executing the instrument on data
    "recomputed",  # independently derived from sealed receipts by a second implementation
    "derived",  # arithmetic over measured quantities
    "analytic",  # closed form, no data touched
    "assumed",  # declared, not established
    "structurally_guaranteed",  # true by construction, for example a frozen group cannot drift
)


@dataclass
class Quantity:
    """A reported number and how it came to exist."""

    value: Any
    kind: str
    source: str = ""  # artifact path plus JSON pointer, or the code path that computed it

    def violations(self, label: str) -> list[str]:
        v = []
        if self.kind not in QUANTITY_KINDS:
            v.append(f"{label}: unknown quantity kind {self.kind!r}")
        if self.value is None:
            v.append(f"{label}: quantity resolved to None")
        if not self.source:
            v.append(f"{label}: quantity has no evidence path")
        return v

    def as_dict(self) -> dict:
        return {"value": self.value, "kind": self.kind, "source": self.source}


def measured(value, source: str) -> Quantity:
    return Quantity(value, "measured", source)


def analytic(value, source: str) -> Quantity:
    return Quantity(value, "analytic", source)


def structural(value, source: str) -> Quantity:
    return Quantity(value, "structurally_guaranteed", source)


# ---------------------------------------------------------------- contract base


@dataclass
class Contract:
    name: str = ""
    kind: str = "Contract"
    declared: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def _need(self, *keys: str) -> list[str]:
        return [f"{self.kind}:{self.name}: missing evidence {k!r}" for k in keys if k not in self.evidence]

    def violations(self) -> list[str]:
        return self._need(*self.declared.get("required_evidence", []))

    @property
    def passed(self) -> bool:
        return not self.violations()

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "declared": self.declared,
            "evidence": self.evidence,
            "violations": self.violations(),
            "passed": self.passed,
        }


# ---------------------------------------------------------------- the seventeen


@dataclass
class ExperimentQuestion(Contract):
    kind: str = "ExperimentQuestion"

    def violations(self) -> list[str]:
        v = []
        for k in ("question", "hypotheses", "predictions"):
            if not self.declared.get(k):
                v.append(f"question:{self.name}: {k} not declared")
        preds = self.declared.get("predictions") or {}
        hyps = set(self.declared.get("hypotheses") or [])
        # every hypothesis must predict each possible outcome, or the experiment cannot discriminate
        for h in hyps:
            if h not in preds:
                v.append(f"question:{self.name}: hypothesis {h!r} makes no prediction")
        if len({tuple(sorted(p.items())) if isinstance(p, dict) else p for p in preds.values()}) < 2:
            v.append(f"question:{self.name}: all hypotheses predict the same result, no discrimination")
        return v


@dataclass
class CausalModel(Contract):
    kind: str = "CausalModel"

    def violations(self):
        from substrate.method import graph

        g = self.declared.get("graph")
        if not g:
            return [f"causal:{self.name}: no graph declared"]
        return [f"causal:{self.name}: {x}" for x in graph.validate(g)]


@dataclass
class MeasurementModel(Contract):
    kind: str = "MeasurementModel"

    def violations(self):
        v = []
        for outcome, spec in (self.declared.get("outcomes") or {}).items():
            if not spec.get("estimator"):
                v.append(f"measurement:{self.name}: outcome {outcome} has no estimator")
            if not spec.get("unit"):
                v.append(f"measurement:{self.name}: outcome {outcome} has no independent unit")
        if not self.declared.get("outcomes"):
            v.append(f"measurement:{self.name}: no outcomes declared")
        return v


@dataclass
class InstrumentContract(Contract):
    kind: str = "InstrumentContract"

    def violations(self):
        cal = self.evidence.get("calibration")
        if cal is None:
            return [f"instrument:{self.name}: not calibrated"]
        bad = [k for k, ok in cal.items() if k != "all_pass" and not ok]
        return [f"instrument:{self.name}: calibration case {k} failed" for k in bad]


@dataclass
class ArmContract(Contract):
    kind: str = "ArmContract"

    def violations(self):
        v = []
        d = self.evidence.get("distinctness")
        if d is None:
            return [f"arm:{self.name}: distinctness never proven"]
        for other, verdict in d.items():
            if verdict == "aliased":
                v.append(f"arm:{self.name}: aliased with {other}")
        return v


@dataclass
class ControlContract(Contract):
    kind: str = "ControlContract"

    def violations(self):
        s = self.evidence.get("semantic")
        if s is None:
            return [f"control:{self.name}: semantics never proven"]
        return [f"control:{self.name}: fails {k}" for k, ok in s.items() if k != "all_pass" and ok is False]


@dataclass
class BaselineContract(Contract):
    kind: str = "BaselineContract"

    def violations(self):
        v = []
        c = self.evidence.get("convergence")
        if c is None:
            return [f"baseline:{self.name}: no convergence receipt"]
        if not c.get("converged"):
            v.append(f"baseline:{self.name}: not converged ({c.get('reason')})")
        if not c.get("resource_matched", True):
            v.append(f"baseline:{self.name}: resource budget not matched to the treatment")
        if self.declared.get("identity") and c.get("identity") != self.declared["identity"]:
            v.append(f"baseline:{self.name}: identity mismatch, declared {self.declared['identity']!r} but receipt says {c.get('identity')!r}")
        return v


@dataclass
class OracleContract(Contract):
    kind: str = "OracleContract"

    def violations(self):
        h = self.evidence.get("headroom")
        if h is None:
            return [f"oracle:{self.name}: headroom never measured"]
        v = []
        if h.get("n_seeds", 0) < 3:
            v.append(f"oracle:{self.name}: headroom rests on {h.get('n_seeds')} seeds, needs at least 3")
        if h.get("residual_lower_95_cb") is None:
            v.append(f"oracle:{self.name}: no residual headroom lower bound")
        elif h["residual_lower_95_cb"] <= 0:
            v.append(f"oracle:{self.name}: residual headroom lower bound is not positive")
        return v


@dataclass
class DatasetContract(Contract):
    kind: str = "DatasetContract"

    def violations(self):
        b = self.evidence.get("bed_validity")
        if b is None:
            return [f"dataset:{self.name}: bed validity never established"]
        cls = b.get("classification")
        if cls not in ("valid_principal_bed", "valid_secondary_bed"):
            return [f"dataset:{self.name}: bed classified {cls}"]
        return []


@dataclass
class IndependentUnitContract(Contract):
    kind: str = "IndependentUnitContract"

    def violations(self):
        v = []
        e = self.evidence.get("units")
        if e is None:
            return [f"units:{self.name}: unit structure never verified"]
        if not e.get("group_disjoint"):
            v.append(f"units:{self.name}: splits are not group disjoint")
        if e.get("n_units", 0) < 2:
            v.append(f"units:{self.name}: fewer than two independent units")
        if e.get("test_touched"):
            v.append(f"units:{self.name}: the test split was touched before the terminal measurement")
        return v


@dataclass
class PowerContract(Contract):
    kind: str = "PowerContract"

    def violations(self):
        v = []
        for k in ("sesoi", "seeds", "futility", "harm"):
            if self.declared.get(k) in (None, ""):
                v.append(f"power:{self.name}: {k} not preregistered")
        p = self.evidence.get("power")
        if p is None:
            v.append(f"power:{self.name}: no power estimate")
        elif p.get("mde") is not None and self.declared.get("sesoi") is not None and p["mde"] > self.declared["sesoi"]:
            v.append(
                f"power:{self.name}: minimum detectable effect {p['mde']} exceeds SESOI "
                f"{self.declared['sesoi']}, the design cannot see the effect it declares interesting"
            )
        return v


@dataclass
class ExecutionContract(Contract):
    kind: str = "ExecutionContract"

    def violations(self):
        v = []
        e = self.evidence.get("execution")
        if e is None:
            return [f"execution:{self.name}: no execution receipt"]
        if e.get("undeclared_changes"):
            v.append(f"execution:{self.name}: parameters changed outside the declared groups")
        if not e.get("seeds"):
            v.append(f"execution:{self.name}: no seeds recorded")
        return v


@dataclass
class VerificationContract(Contract):
    kind: str = "VerificationContract"

    def violations(self):
        r = self.evidence.get("roles")
        if r is None:
            return [f"verification:{self.name}: no role receipts"]
        return [f"verification:{self.name}: role {k} did not pass" for k, ok in r.items() if not ok]


@dataclass
class MutationContract(Contract):
    kind: str = "MutationContract"

    def violations(self):
        m = self.evidence.get("mutations")
        if m is None:
            return [f"mutation:{self.name}: no mutation suite"]
        return [f"mutation:{self.name}: mutation {k} was not rejected" for k, ok in m.items() if not ok]


@dataclass
class ResultContract(Contract):
    kind: str = "ResultContract"

    def violations(self):
        v = []
        for label, q in (self.evidence.get("quantities") or {}).items():
            q = q if isinstance(q, Quantity) else Quantity(**q)
            v.extend(f"result:{self.name}: {x}" for x in q.violations(label))
        if not self.declared.get("classification"):
            v.append(f"result:{self.name}: no terminal classification")
        return v


@dataclass
class ClaimContract(Contract):
    kind: str = "ClaimContract"

    def violations(self):
        v = []
        measured_paths = set(self.evidence.get("measured_paths") or [])
        for claim in self.declared.get("claims") or []:
            need = set(claim.get("requires") or [])
            missing = need - measured_paths
            if missing:
                v.append(f"claim:{self.name}: {claim.get('text')!r} needs unmeasured paths {sorted(missing)}")
        return v


CONTRACT_TYPES = {
    c.__name__: c
    for c in (
        ExperimentQuestion,
        CausalModel,
        MeasurementModel,
        InstrumentContract,
        ArmContract,
        ControlContract,
        BaselineContract,
        OracleContract,
        DatasetContract,
        IndependentUnitContract,
        PowerContract,
        ExecutionContract,
        VerificationContract,
        MutationContract,
        ResultContract,
        ClaimContract,
    )
}
