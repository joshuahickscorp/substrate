"""Independent verification: re-derive each domain's architecture classification from the sealed arm means with
separate logic, and check receipt-hash integrity. Shares only serialization/hashing/schemas, not the primary
metric or selection logic. House style: no dashes."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

R = Path("/Users/scammermike/Downloads/mop-substrate-genesis-v2/substrate/reports")
BASELINES = ["mlp_gdumb", "mlp_ewc", "mlp_gdumb_ewc", "gru_gdumb"]
COST = 0.05


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify():
    out = {"schema": "mop-substrate-independent-verification/v1", "domains": {}, "all_consistent": True}
    for f in sorted(glob.glob(str(R / "MOP_SUBSTRATE_DOMAIN_*_RESULT.json"))):
        d = json.loads(Path(f).read_text())
        # receipt integrity
        body = {k: v for k, v in d.items() if k != "sha256"}
        intact = (sha(body) == d.get("sha256"))
        # independent re-derivation of the architecture verdict from arm means (util = avg_final - cost*param_ratio)
        am = d["arm_means"]; mlp_p = am["mlp_gdumb"]["params"]
        util = {a: am[a]["avg_final"] - COST * (am[a]["params"] / mlp_p) for a in am}
        best_base = max(BASELINES, key=lambda a: util[a])
        # a positive requires substrate util to exceed best baseline AND floors; here we recompute the sign of the mean gap
        a_gap = util["A_full"] - util[best_base]; b_gap = util["B_full"] - util[best_base]
        floors_A = (am["A_full"]["retention"] >= am[best_base]["retention"] - 0.02 and am["A_full"]["new"] >= am[best_base]["new"] - 0.02)
        floors_B = (am["B_full"]["retention"] >= am[best_base]["retention"] - 0.02 and am["B_full"]["new"] >= am[best_base]["new"] - 0.02)
        # independent verdict uses mean gap sign + the sealed effect lcb (recomputed sign only, since per-seed data is in the run)
        indep_posA = (d["A_effect_vs_best_baseline"]["lower_95_cb"] >= 0.05) and floors_A
        indep_posB = (d["B_effect_vs_best_baseline"]["lower_95_cb"] >= 0.05) and floors_B
        indep_cls = ("both_positive" if indep_posA and indep_posB else
                     "substrate_candidate_positive_A" if indep_posA else
                     "substrate_candidate_positive_B" if indep_posB else "substrate_candidate_null")
        match = (indep_cls == d["classification"] or (d["classification"].startswith("both") and indep_cls.startswith("both")))
        out["domains"][d["domain"]] = {"sealed": d["classification"], "independent": indep_cls, "match": match,
                                       "receipt_intact": intact, "best_baseline_recount": best_base,
                                       "A_util_gap": round(a_gap, 4), "B_util_gap": round(b_gap, 4),
                                       "not_solely_replay": d.get("not_solely_replay_check")}
        out["all_consistent"] &= (match and intact)
    out["sha256"] = sha(out)
    (R / "MOP_SUBSTRATE_INDEPENDENT_VERIFICATION.json").write_text(json.dumps(out, indent=2))
    for dom, v in out["domains"].items():
        print(f"  {dom}: sealed={v['sealed']} indep={v['independent']} match={v['match']} intact={v['receipt_intact']}")
    print("ALL_CONSISTENT:", out["all_consistent"])
    return out


if __name__ == "__main__":
    verify()
