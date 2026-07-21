"""Focused regressions for secondary bed admission and core evidence gates."""

from mop.temporal.runs import coresel, replicate


def test_third_bed_cannot_replicate_without_preflight_admission():
    assert not replicate.third_bed_admitted({"selected": []})
    assert replicate.third_bed_classification({"selected": []}, True) == "invalid_secondary_bed"
    assert replicate.third_bed_classification({"selected": ["harth_stream"]}, True) == "replicated"
    assert (replicate.third_bed_classification({"selected": ["harth_stream"]}, False)
            == "valid_secondary_bed_did_not_reproduce_the_principal_effect")


def test_core_requires_consistent_admission_and_replication_artifacts():
    preflight = {"selected": ["harth_stream"]}
    replication = {"third_bed_classification": "replicated"}
    result = {"bed": "harth_stream", "classification": "replicated"}
    assert coresel.third_bed_licensed(preflight, replication, result)
    assert not coresel.third_bed_licensed({"selected": []}, replication, result)
    assert not coresel.third_bed_licensed(preflight, replication,
                                          {"bed": "harth_stream", "classification": "invalid_secondary_bed"})


def test_core_factorial_gate_requires_exact_two_principal_bed_checks():
    beds = list(coresel.e2.B.PRINCIPAL)
    authority = {
        "all_principal_beds_valid": True,
        "principal_beds": {bed: {"checks": {"all_pass": True}} for bed in beds},
    }
    assert all(coresel.factorial_evidence_gates(authority).values())
    authority["principal_beds"]["extra_bed"] = {"checks": {"all_pass": True}}
    assert not coresel.factorial_evidence_gates(authority)["exact_principal_bed_checks"]
    authority["principal_beds"].pop("extra_bed")
    authority["all_principal_beds_valid"] = False
    assert not coresel.factorial_evidence_gates(authority)["all_principal_beds_valid"]
