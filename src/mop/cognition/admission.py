"""Substrate admission: no new Substrate experiment reaches principal compute unproven.

This does not reimplement the experiment validity kernel. The kernel already walks the stage sequence and
blocks at the first failure. What Substrate adds is completeness: the kernel passes a stage that has no
contracts at all, because an empty stage has no violations. Section 18 of the master plan says every
subsystem requires all twelve, so a Substrate preregistration that simply omits a contract kind is refused
here rather than sliding through as a silent default pass. That is the same failure shape the kernel exists
to stop, one level up.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys

from mop.cognition import io
from mop.method import gate

# section 18: every subsystem requires all of these. Absence is a violation, not a default pass.
MANDATORY_KINDS = (
    "ExperimentQuestion", "CausalModel", "MeasurementModel", "InstrumentContract", "ArmContract",
    "ControlContract", "DatasetContract", "IndependentUnitContract", "BaselineContract",
    "OracleContract", "PowerContract",
)

# a positive normally requires all of these, checked after principal compute rather than before
POSITIVE_STANDARD = (
    "two capable implementations", "two valid beds where appropriate",
    "lower confidence bound above SESOI", "cost adjusted value", "complete mutation rejection",
)


def completeness(prereg: gate.Preregistration) -> list[str]:
    present = {c.kind for c in prereg.contracts}
    missing = [k for k in MANDATORY_KINDS if k not in present]
    v = [f"completeness:{prereg.experiment_id}: no {k} declared" for k in missing]
    if prereg.mechanism_activity is None:
        v.append(f"completeness:{prereg.experiment_id}: mechanism activity never measured")
    return v


def admit(prereg: gate.Preregistration, stage: str = "principal") -> dict:
    """The kernel verdict, refused further if any mandatory contract kind is simply absent."""
    report = prereg.admit(stage)
    gaps = completeness(prereg)
    if gaps:
        report = {**report, "licensed": False, "principal_execution_licensed": False,
                  "blocked_at": report["blocked_at"] or "completeness",
                  "blocking_violations": list(report["blocking_violations"]) + gaps}
    report["completeness_violations"] = gaps
    report["substrate_admission"] = "licensed" if report["licensed"] else "refused"
    return report


def license_principal(prereg: gate.Preregistration) -> dict:
    """Return the admission report. The caller must not spend principal compute unless licensed is true."""
    report = admit(prereg, "principal")
    io.run_json(f"admission_{prereg.experiment_id}.json",
                {"schema": "substrate-admission/v1", **report}, "admissions")
    return report


def requirements_authority() -> dict:
    doc = {
        "schema": "substrate-experimental-requirements/v1",
        "kernel": "mop.method.gate",
        "sequence": list(gate.SEQUENCE),
        "pre_principal_stages": list(gate.PRE_PRINCIPAL),
        "mandatory_contract_kinds": list(MANDATORY_KINDS),
        "substrate_addition": ("the kernel passes a stage with no contracts because an empty stage has no "
                               "violations. Substrate refuses a preregistration that omits a mandatory "
                               "contract kind, so absence of measurement stays a violation one level up"),
        "positive_standard": list(POSITIVE_STANDARD),
        "defect_rule": ("a reproduced defect overrides reviewer votes, becomes an append only correction "
                        "and becomes a permanent regression test"),
        "method_failure_rule": "a methodological failure is not a scientific null",
    }
    return doc


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    path = io.seal("SUBSTRATE_EXPERIMENTAL_REQUIREMENTS.json", requirements_authority())
    print(f"substrate experimental requirements sealed: {path.relative_to(io.ROOT)}", flush=True)
    print(json.dumps({"mandatory_kinds": len(MANDATORY_KINDS),
                      "pre_principal_stages": len(gate.PRE_PRINCIPAL)}))


if __name__ == "__main__":
    main()
