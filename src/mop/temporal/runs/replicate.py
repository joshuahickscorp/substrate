"""Independent recurrent implementation comparison and explicit third bed replication result."""

from __future__ import annotations

import json
import time

import torch.nn as nn

from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2

BEDS = ("har_stream", "speech_stream", "harth_stream")
GRU = Fx.cell_name(**dict(Fx.REFERENCE, family="gru"))
MGU = Fx.cell_name(**dict(Fx.REFERENCE, family="mgu"))
HISTORY = Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", history_k="full_window"))


def third_bed_admitted(preflight: dict) -> bool:
    """The sealed preflight is the sole authority that may admit the secondary bed."""
    selected = preflight.get("selected")
    return isinstance(selected, list) and "harth_stream" in selected


def third_bed_classification(preflight: dict, effects_reproduce: bool) -> str:
    if not third_bed_admitted(preflight):
        return "invalid_secondary_bed"
    if effects_reproduce:
        return "replicated"
    return "valid_secondary_bed_did_not_reproduce_the_principal_effect"


def _runs(bed: str) -> list[dict]:
    out = []
    for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
        out.extend(json.loads(p.read_text())["runs"])
    return out


def _convergence(bed: str, cell: str) -> dict:
    p = io.RUNS / "e2_converge" / f"converge_{bed}.json"
    if not p.is_file():
        return {"classification": "not_measured"}
    return (json.loads(p.read_text()).get("configs") or {}).get(cell, {"classification": "not_measured"})


def _effect(bed: str, series: dict, units: dict, cell: str) -> dict:
    d = AN.contrast(series, cell, HISTORY, e2.PREREG, units)
    conv = {c: _convergence(bed, c).get("classification") for c in (cell, HISTORY)}
    d["convergence"] = conv
    d["load_bearing_positive"] = (
        d.get("verdict") == "positive"
        and (d.get("group_lower_95_cb") or float("-inf")) >= io.SESOI
        and all(v == "converged" for v in conv.values())
    )
    return d


def implementation_audit() -> dict:
    sp = B.splits("har_stream", 0)
    gru = Fx.build_cell(sp, seed=0, **dict(Fx.REFERENCE, family="gru"))[0]
    mgu = Fx.build_cell(sp, seed=0, **dict(Fx.REFERENCE, family="mgu"))[0]
    return {
        "gru": {
            "call_path": "Cell.represent -> Recurrent.forward -> torch.nn.GRU",
            "uses_fused_recurrent_base": any(isinstance(m, nn.RNNBase) for m in gru.modules()),
            "parameters": A.count(gru),
            "state_transition": "torch.nn.GRU",
        },
        "mgu": {
            "call_path": "Cell.represent -> Recurrent.forward -> explicit timestep loop -> MGUCell.step",
            "uses_fused_recurrent_base": any(isinstance(m, (nn.RNNBase, nn.RNNCellBase)) for m in mgu.modules()),
            "parameters": A.count(mgu),
            "state_transition": "independent one gate cell implemented in arch.py",
        },
        "shared_recurrent_core_code": False,
        "shared_allowed_code": ["data loaders", "factorial schema", "training engine", "readout schema"],
        "reset_semantics": "both receive the same explicit reset index list from factorial.reset_schedule",
        "history_visibility": "both declare state carried from previous observations",
        "output_contribution": "both representations feed the identical readout interface",
        "pass": (any(isinstance(m, nn.RNNBase) for m in gru.modules())
                 and not any(isinstance(m, (nn.RNNBase, nn.RNNCellBase)) for m in mgu.modules())),
    }


def main():
    t0 = time.time()
    preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
    per_bed = {}
    for bed in BEDS:
        runs = _runs(bed)
        series, units = e2._series(runs)
        effects = {"torch_gru_vs_full_history": _effect(bed, series, units, GRU),
                   "explicit_mgu_vs_full_history": _effect(bed, series, units, MGU)}
        per_bed[bed] = {
            "n_seeds": len({r["seed"] for r in runs}),
            "effects": effects,
            "implementations_agree": all(v["load_bearing_positive"] for v in effects.values()),
        }
    audit = implementation_audit()
    principal_pass = audit["pass"] and all(per_bed[b]["implementations_agree"] for b in B.PRINCIPAL)
    third = per_bed["harth_stream"]
    third_admitted = third_bed_admitted(preflight)
    third_classification = third_bed_classification(preflight, third["implementations_agree"])
    doc = {
        "schema": "mop-e2-independent-replication/v1",
        "reference_control": HISTORY,
        "per_bed": per_bed,
        "implementation_audit": audit,
        "principal_beds_pass": principal_pass,
        "third_bed_admitted": third_admitted,
        "third_bed_classification": third_classification,
        "all_pass": principal_pass,
        "rule": ("a load bearing recurrence positive requires both independent recurrent implementations, "
                 "both principal beds, a group lower bound above the SESOI, and converged alternatives"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_E2_INDEPENDENT_REPLICATION.json", doc)
    io.seal("MOP_THIRD_TEMPORAL_BED_RESULT.json", {
        "schema": "mop-third-temporal-bed-result/v1",
        "bed": "harth_stream",
        "admitted": third_admitted,
        "admission": (preflight.get("candidates") or {}).get("harth_stream"),
        "effects": third["effects"],
        "classification": third_classification,
        "claim_ceiling": "secondary natural bed only; it is not promoted to a principal adaptation bed",
    })
    print(f"independent replication: principal {principal_pass}, third {doc['third_bed_classification']}")
    print("REPLICATE_DONE")


if __name__ == "__main__":
    main()
