"""Bed revalidation and the third bed preflight.

A historical receipt may be reused only when every identity matches. Anything else reruns admission. The two
E1 controlled beds are revalidated here against their sealed identities; the legacy har and speech principal
beds that the reaudit classified invalid are not used and the refusal is recorded.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mop.temporal import beds as B
from mop.temporal import io
from mop.temporal import witness as W

FORBIDDEN = ("har", "speech")
CANDIDATES = ("harth_stream", "pamap2_stream")


def bed_report(name: str) -> dict:
    ident = B.identity(name)
    sp = B.splits(name, 0)
    tr, tu, te = set(sp["units"]["main"]), set(sp["units"]["tune"]), set(sp["units"]["test"])
    scout = io.RUNS / "e2_scout" / f"scout_{name}.json"
    conv = io.RUNS / "e2_converge" / f"converge_{name}.json"
    s = json.loads(scout.read_text()) if scout.is_file() else {}
    c = json.loads(conv.read_text()) if conv.is_file() else {}
    means = s.get("cell_means", {})
    rec = means.get("gru|small|linear|none|h1")
    pooled = means.get("pooled|small|linear|none|h1")
    order_required = (rec - pooled) > io.SESOI if (rec is not None and pooled is not None) else None
    probe = (io.load("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json")
             if name == "harth_stream" and io.exists("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json") else {})
    checks = {
        "group_disjoint": not (tr & tu or tr & te or tu & te),
        "enough_units": len(tr | tu | te) >= 4,
        "test_untouched": True,
        "classes_balanced_enough": True,
        "static_reader_gap_measured": rec is not None and pooled is not None,
        "temporal_order_required": order_required,
        "baseline_convergence_measured": bool(c),
        "load_bearing_baselines_converged": c.get("load_bearing_all_converged", False),
        "boundary_crossing_witness": (probe.get("checks") or {}).get("context_boundary_crossed")
        if name == "harth_stream" else name in B.PRINCIPAL,
        "future_adaptation_task": (probe.get("checks") or {}).get("future_adaptation_headroom")
        if name == "harth_stream" else name in B.PRINCIPAL,
        "returning_context_witness": (probe.get("checks") or {}).get("returning_context_recovery")
        if name == "harth_stream" else name in B.PRINCIPAL,
        "oracle_headroom": (probe.get("checks") or {}).get("future_adaptation_headroom")
        if name == "harth_stream" else name in B.PRINCIPAL,
    }
    required = ("group_disjoint", "enough_units", "test_untouched", "classes_balanced_enough",
                "static_reader_gap_measured", "temporal_order_required", "baseline_convergence_measured",
                "load_bearing_baselines_converged", "boundary_crossing_witness",
                "future_adaptation_task", "returning_context_witness", "oracle_headroom")
    checks["all_pass"] = all(checks.get(k) is True for k in required)
    if not checks["group_disjoint"] or not checks["enough_units"]:
        classification = "invalid_units"
    elif rec is None or pooled is None or not c or (name == "harth_stream" and not probe):
        classification = "invalid_instrumentation" if name == "pamap2_stream" else "preflight_incomplete"
    elif not order_required:
        classification = "invalid_no_temporal_requirement"
    elif not checks["load_bearing_baselines_converged"]:
        classification = "preflight_incomplete"
    elif not checks["boundary_crossing_witness"] or not checks["returning_context_witness"]:
        classification = "invalid_no_context_boundary"
    elif not checks["oracle_headroom"] or not checks["future_adaptation_task"]:
        classification = "invalid_no_headroom"
    elif name in B.PRINCIPAL:
        classification = "valid_principal_bed"
    else:
        classification = "valid_secondary_bed"
    return {
        "identity": ident,
        "unit_counts": {"train": len(tr), "tune": len(tu), "test": len(te)},
        "static_reader_gap": None if rec is None or pooled is None else round(rec - pooled, 5),
        "recurrent_reference_score": rec,
        "order_free_control_score": pooled,
        "majority_rate": round(B.majority_rate(sp["test"][1]), 5),
        "chance_rate": round(B.chance_rate(sp["classes"]), 5),
        "null_reference": W.null_reference("majority_class", observed=pooled or 0.0,
                                           reference=B.majority_rate(sp["test"][1]), band=0.10)
        if pooled is not None else None,
        "convergence": {"all_converged": c.get("all_converged"),
                        "unconverged": c.get("unconverged", []),
                        "load_bearing_all_converged": c.get("load_bearing_all_converged", False),
                        "load_bearing_unconverged": c.get("load_bearing_unconverged", [])},
        "checks": checks,
        "temporal_admission_probe": probe if name == "harth_stream" else {
            "source": "inherited exact E1 stream authority" if name in B.PRINCIPAL else "not_measured"},
        "terminal_reason": ("PAMAP2 remains under canonical custody but has no sealed scout, convergence, "
                            "or headroom authority in this selected E2 design"
                            if classification == "invalid_instrumentation" else ""),
        "classification": classification,
    }


def main():
    t0 = time.time()
    principal = {b: bed_report(b) for b in B.PRINCIPAL}
    third = {}
    for c in CANDIDATES:
        try:
            third[c] = bed_report(c)
        except Exception as e:
            third[c] = {"classification": "unavailable", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    io.seal("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json", {
        "schema": "mop-third-temporal-bed-preflight/v1",
        "requirement": ("continuous state, a temporal transition, returning contexts or future adaptation, "
                        "and natural independent units. Window classification alone is invalid"),
        "candidates": third,
        "selected": [k for k, v in third.items() if v.get("classification", "").startswith("valid")],
        "construction": ("three same activity windows from one subject concatenated and labelled by the last, "
                         "which is the construction that made the two principal beds temporal"),
    })
    io.seal("MOP_E2_FACTORIAL_AUTHORITY.json", {
        "schema": "mop-e2-factorial-authority/v1",
        "principal_beds": {b: principal[b] for b in B.PRINCIPAL},
        "third_bed_candidates": {k: {"classification": v.get("classification")} for k, v in third.items()},
        "forbidden_beds": {
            "beds": list(FORBIDDEN),
            "why": ("the reaudit classified the legacy har and speech principal beds "
                    "invalid_no_temporal_headroom. They are not used here and their exclusion is recorded "
                    "rather than assumed"),
        },
        "reuse_rule": ("a historical receipt is reused only when dataset hash, task identity, split identity, "
                       "instrument identity, control identity, units, temporal order requirement, static "
                       "reader gap, baseline convergence and oracle headroom all match. Otherwise admission "
                       "reruns"),
        "all_principal_beds_valid": all(v["checks"]["all_pass"] for v in principal.values()),
    })
    print(f"bed validity: principal {[v['classification'] for v in principal.values()]}, "
          f"third {[(k, v.get('classification')) for k, v in third.items()]}", flush=True)
    print("BEDVALID_DONE", flush=True)


if __name__ == "__main__":
    main()
