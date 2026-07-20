"""Substrate Genesis terminal synthesis + architecture comparison/selection + inventories, from sealed results.
House style: no dashes."""

from __future__ import annotations

import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

W = Path("/Users/scammermike/Downloads/mop-substrate-genesis-v2")
R = W / "substrate/reports"
S = W / "substrate"


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def L(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def param_inventory():
    sys.path.insert(0, str(S))
    import torch  # noqa
    from engine import make
    inv = {}
    for name in ["A", "B"]:
        m = make(name, 784, 30)
        groups = {g: int(sum(p.numel() for p in ps)) for g, ps in m.param_groups.items()}
        inv[name] = {"total_params": int(sum(p.numel() for p in m.parameters())), "param_groups": groups,
                     "timescales": ["fast (per-sequence GRU state)", "episodic (bounded replay memory)",
                                    "slow (projection + workspace/module params under eligibility)",
                                    "policy (consolidation/eligibility/routing schedule)"]}
    return inv


def main(close_if_terminal=False):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(W)).stdout.strip()
    pf = L(R / "MOP_SUBSTRATE_DOMAIN_PREFLIGHT.json")
    doms = {L(f)["domain"]: L(f) for f in sorted(glob.glob(str(R / "MOP_SUBSTRATE_DOMAIN_*_RESULT.json")))}
    ver = L(R / "MOP_SUBSTRATE_INDEPENDENT_VERIFICATION.json")
    inv = param_inventory()
    (S / "MOP_SUBSTRATE_PARAMETER_INVENTORY.json").write_text(json.dumps(
        {"schema": "mop-substrate-parameter-inventory/v1", "architectures": inv}, indent=2))

    # architecture comparison + selection across domains
    per_dom_cls = {d: doms[d]["classification"] for d in doms}
    a_pos = [d for d in doms if doms[d]["classification"] == "substrate_candidate_positive_A" or doms[d]["classification"] == "both_positive"]
    b_pos = [d for d in doms if doms[d]["classification"] == "substrate_candidate_positive_B" or doms[d]["classification"] == "both_positive"]
    both_null = all(doms[d]["classification"] == "substrate_candidate_null" for d in doms)
    if both_null:
        selection, sel_cls = None, "no_substrate_candidate_selected (both architectures null on all evaluated domains)"
    elif len(a_pos) >= len(b_pos) and a_pos:
        selection, sel_cls = "A_shared_latent_workspace", "select_A"
    elif b_pos:
        selection, sel_cls = "B_sparse_modular_substrate", "select_B"
    else:
        selection, sel_cls = None, "no_substrate_candidate_selected"
    comparison = {"schema": "mop-substrate-architecture-comparison/v1", "source_commit": commit,
                  "per_domain_classification": per_dom_cls,
                  "domain_details": {d: {"best_baseline": doms[d]["best_baseline"],
                                         "A_util": doms[d]["util_mean"].get("A_full"), "B_util": doms[d]["util_mean"].get("B_full"),
                                         "best_baseline_util": doms[d]["util_mean"].get(doms[d]["best_baseline"]),
                                         "A_effect_lcb": doms[d]["A_effect_vs_best_baseline"]["lower_95_cb"],
                                         "B_effect_lcb": doms[d]["B_effect_vs_best_baseline"]["lower_95_cb"]} for d in doms},
                  "selection": selection, "selection_class": sel_cls,
                  "independent_verification_consistent": (ver or {}).get("all_consistent")}
    comparison["sha256"] = sha(comparison)
    (S / "MOP_SUBSTRATE_ARCHITECTURE_COMPARISON.json").write_text(json.dumps(comparison, indent=2))

    # terminal synthesis (answers the 34 questions compactly)
    syn = {"schema": "mop-substrate-genesis-synthesis/v1", "source_commit": commit, "branch": "agent/mop-substrate-genesis-v2",
           "owned_substrate_implemented": True,
           "owned_state_and_params": "owned trainable projection + latent workspace/modules + slow params + fast GRU state (both architectures); frozen providers supply only raw observations",
           "architectures": {"A": "Shared Latent Workspace (" + str(inv["A"]["total_params"]) + " params)",
                             "B": "Sparse Modular Plastic Substrate (" + str(inv["B"]["total_params"]) + " params)"},
           "timescales": inv["A"]["timescales"],
           "domains_evaluated": {d: {"is_image": doms[d]["is_image"], "classification": doms[d]["classification"]} for d in doms},
           "preflight": {d: pf["domains"][d]["classification"] for d in pf["domains"]} if pf else None,
           "architecture_comparison": {"selection": selection, "selection_class": sel_cls,
                                       "per_domain": per_dom_cls},
           "moldability_finding": ("both architectures NULL vs strong matched baselines on all evaluated domains"
                                   if both_null else "see architecture comparison"),
           "simple_policy_sufficient": both_null,
           "learned_plasticity": "not opened unless a substrate-specific oracle-headroom gate passes (deferred/pending)",
           "cross_domain": "pending or checkpointed" if both_null else "see transfer report",
           "independent_verification": (ver or {}).get("all_consistent"),
           "activation": False,
           "evidence_ceiling": ("no owned-substrate architecture beats strong matched continual-learning baselines "
                                "(GDumb, EWC) on cost-adjusted moldability on the evaluated domains" if both_null
                                else "an owned-substrate architecture shows a matched-budget advantage on at least one domain, pending transfer/mutation/verification"),
           "forbidden_claims": ["the substrate succeeds by more params/memory/updates/data/time/frozen-encoder/state-change",
                                "any activation is licensed", "any learned controller is validated"],
           "substrate_candidate": ("no candidate selected (both null)" if both_null else selection),
           "next_frontier": ("If both architectures are null under matched budgets, the owned-substrate premise "
                             "at this scale does not beat strong established continual learners; the next step is "
                             "either a temporal-native domain where the fast timescale carries real information "
                             "(the image domains make fast state degenerate) or a substrate-specific plasticity "
                             "headroom gate on the owned state. Absent a matched-budget win, no Owned Substrate v0 "
                             "candidate is licensed.")}
    syn["sha256"] = sha(syn)
    (S / "MOP_SUBSTRATE_GENESIS_SYNTHESIS.json").write_text(json.dumps(syn, indent=2))
    print("synthesis sealed. selection:", selection, "| class:", sel_cls, "| per-domain:", per_dom_cls)
    return syn


if __name__ == "__main__":
    main("--close-if-terminal" in sys.argv)
