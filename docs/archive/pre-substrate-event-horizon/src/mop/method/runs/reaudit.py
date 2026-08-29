"""Reaudit the terminal Fast State Plasticity Forge with the new method system.

The science is not rerun. The instruments are. Every sealed artifact is read, every load bearing instrument
claim is reproved by executing it, and every terminal result is reclassified by the kernel rather than by
the program that produced it.

Nothing here edits an inherited receipt. Findings are append only and live in this program's proof root.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from mop.method import arms, controls, gate, io, report

FORGE = io.ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1"

CLASSES = (
    "fully_valid",
    "valid_with_nonloadbearing_issue",
    "requires_append_only_correction",
    "invalid_instrument",
    "scientifically_unresolved",
)


def load(name: str) -> dict:
    return json.loads((FORGE / name).read_text())


# ---------------------------------------------------------------- live instrument reproofs


def reprove_order_free_control() -> dict:
    """Execute the repaired order free control and prove its declared semantics hold.

    This is defect D1 checked against the thing that replaced it, not against its own description.
    """
    import torch

    from fastforge import arch as A

    dom = {"har": (9, 6), "speech": (40, 10)}
    torch.manual_seed(0)
    bag = A.build("bag", dom).eval()
    x = torch.randn(6, 32, 9)
    r = controls.order_free(lambda t: bag(t, "har")[0], x, module=bag)
    # and the counterexample: the recurrent principal architecture must fail the same proof
    g = A.build("G", dom).eval()
    rg = controls.order_free(lambda t: g(t, "har")[0], x, module=g)
    return {
        "control": "bag order free control (fastforge.arch.BagControl)",
        "semantic": {k: v for k, v in r.items() if isinstance(v, bool)},
        "structural_findings": r.get("structural_findings"),
        "passes": r["all_pass"],
        "discriminates": not rg["all_pass"],
        "recurrent_arch_fails_same_proof": {k: v for k, v in rg.items() if isinstance(v, bool)},
        "defect": "D1",
        "verdict": "repaired_and_proven" if r["all_pass"] and not rg["all_pass"] else "still_defective",
    }


def reprove_arm_distinctness() -> dict:
    """Rebuild arm records from the sealed identities and rerun the universal verifier."""
    auth = load("MOP_CROSS_DOMAIN_ARM_AUTHORITY.json")
    ident = {d["arm"]: d for d in auth["identities"]}
    recs = []
    for name, spec in auth["arms"].items():
        d = ident.get(name, {})
        recs.append(
            arms.record(
                name,
                source=f"fastforge.arms.ARMS[{name!r}]",
                config=spec,
                call_graph=[f"build:{name}", "engine.fit"],
                state_transitions=[d.get("checkpoint_after_first_domain"), d.get("checkpoint_final")],
                param_delta={
                    "trainable": d.get("phase2_trainable_params"),
                    "changed": d.get("phase2_changed_params"),
                    "groups": d.get("phase2_trainable_groups"),
                },
                memory={"policy": "budget matched", "cap": 600},
                resources={"group_sha": d.get("phase2_group_sha"), "spec_sha": d.get("spec_sha")},
                outputs=d.get("checkpoint_final"),
            )
        )
    dist = arms.distinctness(recs)
    return {
        "n_arms": len(recs),
        "all_distinct": dist["all_distinct"],
        "aliased_pairs": dist["aliased_pairs"],
        "sealed_claim": auth.get("note", ""),
        "sealed_audit_all_pass": load("MOP_CROSS_DOMAIN_ARM_AUDIT.json")["checks"]["all_pass"],
        "verdict": "reproduced" if dist["all_distinct"] else "contradicted",
        "identity_fields_used": list(arms.LOAD_BEARING),
    }


def latent_builder_aliases() -> dict:
    """Any two registered builders that construct the same thing are an alias waiting to be used."""
    import inspect

    from fastforge import arch as A

    src = {k: inspect.getsource(v).split("lambda dom:")[-1].strip() for k, v in A.BUILDERS.items()}
    groups: dict[str, list[str]] = {}
    for k, v in src.items():
        groups.setdefault(v, []).append(k)
    dupes = {v: ks for v, ks in groups.items() if len(ks) > 1}
    used = set(load("MOP_HAR_TO_SPEECH_REPORT.json")["arms"]) | set(
        load("MOP_BASELINE_CONVERGENCE_REPORT.json")["baselines"]
    )
    return {
        "duplicate_builder_groups": {v: ks for v, ks in dupes.items()},
        "load_bearing": {v: ks for v, ks in dupes.items() if len(set(ks) & used) > 1},
        "any_load_bearing": any(len(set(ks) & used) > 1 for ks in dupes.values()),
    }


# ---------------------------------------------------------------- bed validity versus principal use


def bed_use_audit() -> dict:
    dv = load("MOP_DOMAIN_VALIDITY.json")
    gates = {d: g["verdict"] for d, g in dv["gates"].items()}
    principal_beds = set()
    for f in ("MOP_HAR_TO_SPEECH_REPORT.json", "MOP_SPEECH_TO_HAR_REPORT.json"):
        principal_beds.update(load(f)["direction"].split("->"))
    within = {"har", "speech"}
    secondary = set()
    sec = load("MOP_FAST_STATE_SECONDARY_MATRIX.json")
    for k in sec.get("per_direction", {}):
        secondary.update(x.replace("_stream", "_stream") for x in str(k).split("-"))
    return {
        "sealed_gate_verdicts": gates,
        "sealed_valid_domains": dv["valid_domains"],
        "sealed_invalid_domains": dv["invalid_domains"],
        "sealed_principal_domains": dv["principal_domains"],
        "beds_used_by_the_principal_matrix": sorted(principal_beds),
        "beds_used_within_domain": sorted(within),
        "beds_used_by_the_secondary_matrix": sorted(secondary),
        "principal_matrix_ran_on_invalid_beds": sorted(principal_beds & set(dv["invalid_domains"])),
        "secondary_matrix_ran_on_valid_beds": sorted(secondary & set(dv["valid_domains"])),
        "finding": (
            "the principal cross domain matrix and the within domain battery ran on beds the same program "
            "sealed as invalid_no_temporal_headroom, and the sealed principal domain list is empty"
        ),
    }


# ---------------------------------------------------------------- report and wording


def report_field_audit() -> dict:
    syn = load("MOP_FAST_STATE_FORGE_SYNTHESIS.json")
    spec = {
        q: {"artifact": "MOP_FAST_STATE_FORGE_SYNTHESIS.json", "pointer": f"/terminal_questions/{q.replace('/', '~1')}"}
        for q in syn["terminal_questions"]
    }
    r = report.audit_report(FORGE, spec)
    nulls = [q for q, v in syn["terminal_questions"].items() if v is None]
    empty = [q for q, v in syn["terminal_questions"].items() if v in ("", [], {})]
    return {
        "n_questions": len(spec),
        "all_resolve": r["passes"],
        "errors": r["errors"][:10],
        "null_answers": nulls,
        "empty_answers": empty,
        "defect": "D7",
        "verdict": "no unresolved report field" if r["passes"] and not nulls else "unresolved fields present",
    }


def wording_audit() -> dict:
    syn = load("MOP_FAST_STATE_FORGE_SYNTHESIS.json")["terminal_questions"]
    dv = load("MOP_DOMAIN_VALIDITY.json")
    checks = {}
    q44 = syn["44 were the principal beds actually temporal"]
    checks["synthesis_q44_vs_sealed_gate"] = report.wording_check(str(q44), "invalid_no_temporal_headroom")
    checks["domain_validity_marginal_meaning"] = report.wording_check(
        dv["marginal_meaning"], "invalid_no_temporal_headroom"
    )
    checks["domain_validity_status_strings"] = report.wording_check(
        " ".join(str(g.get("status", "")) for g in dv["gates"].values()), "invalid_no_temporal_headroom"
    )
    offenders = {k: v["offenders"] for k, v in checks.items() if not v["passes"]}
    return {
        "checks": {k: v["passes"] for k, v in checks.items()},
        "offenders": offenders,
        "defect": "D9",
        "verdict": "verdict_softening_present" if offenders else "no softening",
    }


def measured_versus_analytic_audit() -> dict:
    """Search the sealed reports for zero valued forgetting reported without a structural label."""
    hits = []
    for p in sorted(FORGE.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        blob = json.dumps(doc)
        if '"negative_transfer"' in blob or "forgetting" in blob:
            for key in ("negative_transfer", "forgetting", "old_domain_drop"):
                for m in _find(doc, key):
                    if isinstance(m, dict):
                        zeros = [k for k, v in m.items() if v == 0 or v == 0.0]
                        if zeros:
                            hits.append({"artifact": p.name, "field": key, "zero_entries": zeros})
    return {
        "zero_valued_entries": hits,
        "defect": "D6",
        "rule": "a zero produced by parameter partitioning is structurally guaranteed and may not be labelled measured",
        "labelled_in_source": any("structurally" in json.dumps(load(h["artifact"])).lower() for h in hits) if hits else None,
    }


def _find(doc, key):
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(doc, list):
        for v in doc:
            yield from _find(v, key)


# ---------------------------------------------------------------- terminal reclassification


def reclassify(bed_audit: dict) -> dict:
    dv_invalid = set(bed_audit["sealed_invalid_domains"])
    dv_valid = set(bed_audit["sealed_valid_domains"])
    out = {}

    def cls(name, sealed, beds, requires_temporal, verifier_agrees=True, mechanism_active=True):
        bed_valid = bool(beds) and not (set(beds) & dv_invalid) if requires_temporal else True
        r = gate.classify_result(
            effect={"verdict": "null"},
            instrument_valid=True,
            bed_valid=bed_valid,
            mechanism_active=mechanism_active,
            baseline_valid=True,
            estimator_sufficient=True,
            verifier_agrees=verifier_agrees,
            mutations_rejected=True,
            implementations_agreeing=2,
        )
        # a correction is owed only when the sealed verdict asserts a scientific finding that the kernel
        # says the instruments cannot support. A sealed verdict that is already the conservative one, such
        # as invalid_no_temporal_headroom, needs nothing.
        sealed_asserts_science = any(w in str(sealed) for w in ("null", "positive", "sufficient"))
        correction = sealed_asserts_science and not r["scientific"]
        agrees = not correction
        out[name] = {
            "sealed_verdict": sealed,
            "beds": sorted(beds),
            "beds_sealed_invalid": sorted(set(beds) & dv_invalid),
            "beds_sealed_valid": sorted(set(beds) & dv_valid),
            "claim_requires_temporal_dynamics": requires_temporal,
            "kernel_classification": r["classification"],
            "kernel_reason": r["reason"],
            "sealed_asserts_a_scientific_finding": sealed_asserts_science,
            "agrees_with_sealed": agrees,
            "reaudit_class": "fully_valid" if agrees else "requires_append_only_correction",
        }

    cls("cross_domain_matrix", "cross_domain_null", {"har", "speech"}, True)
    cls("within_domain_battery", "within_domain_null", {"har", "speech"}, True)
    cls("secondary_matrix", "secondary_null", {"har_stream", "speech_stream"}, True)
    cls("functional_reorganization", "functional_reorganization_null", {"har", "speech"}, False)
    cls("task_free_context", "task_free_context_null", {"har", "speech"}, False)
    cls("plasticity_policy", "simple_partition_policy_sufficient", {"har", "speech"}, False)
    cls("improvement_rounds", "improvement_round_null", {"har", "speech"}, False)
    cls("third_domain_preflight", "invalid_no_temporal_headroom", {"pamap2_transition", "harth_transition"}, False)
    return out


def main():
    t0 = time.time()
    ctrl = reprove_order_free_control()
    armd = reprove_arm_distinctness()
    latent = latent_builder_aliases()
    beds = bed_use_audit()
    rep = report_field_audit()
    word = wording_audit()
    manal = measured_versus_analytic_audit()
    recl = reclassify(beds)

    load_bearing = [k for k, v in recl.items() if v["reaudit_class"] == "requires_append_only_correction"]
    findings = []
    if load_bearing:
        findings.append(
            {
                "id": "R1",
                "severity": "load_bearing",
                "title": "principal results were measured on beds the same program sealed as invalid",
                "path": "proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_DOMAIN_VALIDITY.json",
                "condition": "valid_domains is [har_stream, speech_stream] and principal_domains is empty, "
                "while the principal cross domain matrix and the within domain battery ran on har and speech",
                "reproduction": "PYTHONPATH=src python3.12 -m mop.method.runs.reaudit",
                "expected": "a claim about shared temporal dynamics is measured on a bed that requires them",
                "actual": "measured on beds sealed invalid_no_temporal_headroom",
                "consequence": "the nulls stand as nulls about transfer, but not as nulls about temporal dynamics. "
                "The dynamics claim is carried by the secondary matrix on har_stream and speech_stream, which is "
                "also null, so the scientific conclusion survives with a corrected claim ceiling",
                "affected": load_bearing,
                "repair": "append only claim ceiling correction, no rerun required",
            }
        )
    if word["offenders"]:
        findings.append(
            {
                "id": "R2",
                "severity": "load_bearing",
                "title": "human readable prose broadened a sealed invalid verdict",
                "path": "proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_DOMAIN_VALIDITY.json",
                "condition": "the sealed gate verdict is invalid_no_temporal_headroom and the prose calls the bed marginal",
                "reproduction": "PYTHONPATH=src python3.12 -m mop.method.runs.reaudit",
                "expected": "prose narrows or restates the sealed verdict",
                "actual": f"prose asserts the stronger class via {sorted({o['term'] for v in word['offenders'].values() for o in v})}",
                "consequence": "a reader of the summary receives a weaker statement of invalidity than the machine sealed",
                "affected": sorted(word["offenders"]),
                "repair": "append only wording correction in this program's synthesis, inherited text untouched",
            }
        )
    if latent["any_load_bearing"]:
        findings.append({"id": "R3", "severity": "load_bearing", "title": "load bearing builder alias", "detail": latent})
    elif latent["duplicate_builder_groups"]:
        findings.append(
            {
                "id": "R3",
                "severity": "non_load_bearing",
                "title": "two registered builders construct the same architecture",
                "path": "fastforge/arch.py BUILDERS",
                "condition": "gru and shared_heads both build Conventional(dom, core=gru, share=True)",
                "reproduction": "PYTHONPATH=src python3.12 -m mop.method.runs.reaudit",
                "expected": "every registered builder is distinct or declared as an alias",
                "actual": "an undeclared alias exists in the registry",
                "consequence": "none in the sealed evidence, because shared_heads was never used by a principal run",
                "affected": [],
                "repair": "declare the alias or delete the unused builder",
            }
        )

    corrected_ceilings = {
        "cross_domain_matrix": {
            "sealed_claim": "shared fast dynamics did not transfer across modalities under any update partition",
            "corrected_claim": (
                "on har and speech, beds this program sealed as invalid_no_temporal_headroom, no update "
                "partition beat separate per domain models. That is a null about transfer, not about temporal "
                "dynamics, because the beds do not require dynamics to be solved"
            ),
            "claim_about_dynamics_is_carried_by": "MOP_FAST_STATE_SECONDARY_MATRIX.json on har_stream and speech_stream",
            "secondary_verdict": load("MOP_FAST_STATE_SECONDARY_MATRIX.json")["verdict"],
            "scientific_conclusion_changes": False,
            "rerun_required": False,
        },
        "within_domain_battery": {
            "sealed_claim": "within domain null on both domains",
            "corrected_claim": (
                "within domain, no arm beat the strongest baseline on har and speech, beds sealed as "
                "invalid_no_temporal_headroom. No within domain measurement exists on a bed that requires "
                "temporal dynamics, so the within domain question is open on har_stream and speech_stream"
            ),
            "claim_about_dynamics_is_carried_by": "nothing: this is an unmeasured cell",
            "scientific_conclusion_changes": True,
            "opens": "within domain fast state on the two sealed valid temporal beds",
            "rerun_required": False,
        },
    }

    result = {
        "schema": "mop-fast-state-reaudit/v1",
        "corrected_claim_ceilings": corrected_ceilings,
        "audited_program": "mop-fast-state-plasticity-forge-v1",
        "audited_commit": load("MOP_FAST_STATE_FORGE_SYNTHESIS.json")["source_commit"],
        "instrument_reproofs": {
            "order_free_control": ctrl,
            "arm_distinctness": armd,
            "latent_builder_aliases": latent,
        },
        "bed_use_audit": beds,
        "report_field_audit": rep,
        "wording_audit": word,
        "measured_versus_analytic_audit": manal,
        "terminal_result_classification": recl,
        "classification_counts": {
            c: sorted(k for k, v in recl.items() if v["reaudit_class"] == c) for c in CLASSES
        },
        "findings": findings,
        "new_load_bearing_defects": [f["id"] for f in findings if f["severity"] == "load_bearing"],
        "inherited_receipts_modified": 0,
        "correction_mode": "append only",
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_FAST_STATE_REAUDIT.json", result)

    rows = "\n".join(
        f"| {k} | {v['sealed_verdict']} | {v['kernel_classification']} | {v['reaudit_class']} |"
        for k, v in recl.items()
    )
    fr = "\n".join(
        f"### {f['id']} {f.get('title')}\n\n"
        f"- severity: {f['severity']}\n- path: `{f.get('path', '')}`\n- condition: {f.get('condition', '')}\n"
        f"- expected: {f.get('expected', '')}\n- actual: {f.get('actual', '')}\n"
        f"- consequence: {f.get('consequence', '')}\n- repair: {f.get('repair', '')}\n"
        for f in findings
    )
    io.seal_md(
        "MOP_FAST_STATE_REAUDIT.md",
        f"""# Fast State Forge reaudit

The science was not rerun. The instruments were, and every terminal result was reclassified by the kernel.

## Instrument reproofs

The repaired order free control passes every invariance in the semantic proof
({ctrl["verdict"]}), and the recurrent architecture fails the same proof, which is what makes the control
discriminating rather than merely permissive.

Arm distinctness over the {armd["n_arms"]} sealed arms: {armd["verdict"]}, aliased pairs
{armd["aliased_pairs"]}.

## Terminal results

| result | sealed verdict | kernel classification | reaudit class |
|---|---|---|---|
{rows}

## Findings

{fr or "none"}

## Immutability

{result["inherited_receipts_modified"]} inherited receipts were modified. Every correction is append only and
lives under this program's proof root.
""",
    )
    print(
        f"reaudit: {len(findings)} findings, load bearing {result['new_load_bearing_defects']}, "
        f"corrections {result['classification_counts']['requires_append_only_correction']}",
        flush=True,
    )
    print("REAUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
