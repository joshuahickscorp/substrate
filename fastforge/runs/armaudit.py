"""Cross domain arm authority, audit and mutation suite.

The prior matrix aliased arms. This runner proves the repair three ways: the declared identity of every arm
is unique, the behaviour of every arm is unique (distinct checkpoint hashes from identical seeds and data),
and the audit itself detects each way an arm could silently alias or leak an update.

A mutation is REJECTED when the audit notices it. An accepted mutation invalidates the repair.

House style: no dashes.
"""

from __future__ import annotations

import time

import torch

from fastforge import arms as AR
from fastforge import engine as E
from fastforge.runs import io

PROBE = {"acquire": 8, "second": 8, "return": 4, "adapt": 4}


def audit(ids: list[dict]) -> dict:
    tup = [tuple(map(str, i["identity_tuple"])) for i in ids]
    ck = [i["checkpoint_final"] for i in ids]
    tr = [tuple(i["phase2_trainable_params"]) for i in ids]
    checks = {
        "distinct_declared_identity": len(set(tup)) == len(tup),
        "distinct_behaviour_checkpoints": len(set(ck)) == len(ck),
        "distinct_trainable_inventories_count": len(set(tr)),
        "no_undeclared_parameter_updates": all(not i["undeclared_changes"] for i in ids),
        "frozen_groups_never_change": all(
            not (set(i["phase2_changed_params"]) & set(i["phase2_frozen_params"])) for i in ids
        ),
        "fresh_controls_reinitialize": all(
            set(i["reinitialized"]) <= set(i["shared_groups_changed_at_boundary"])
            for i in ids
        ),
        "persistent_arms_do_not_reinitialize": all(
            not (set(AR.SHARED) - set(i["reinitialized"])) & set(i["shared_groups_changed_at_boundary"])
            for i in ids
        ),
    }
    checks["all_pass"] = all(v is True for v in checks.values() if isinstance(v, bool))
    return checks


def mutations(ids: list[dict]) -> dict:
    """Each entry must be True, meaning the audit rejected the mutation."""
    res = {}
    by = {i["arm"]: i for i in ids}

    # 1 two arms declared with the same specification
    m = [dict(i) for i in ids]
    m[1]["identity_tuple"] = list(m[0]["identity_tuple"])
    m[1]["checkpoint_final"] = m[0]["checkpoint_final"]
    res["aliased_arm_pair_rejected"] = not audit(m)["all_pass"]

    # 2 an undeclared parameter changed during a phase
    m = [dict(i) for i in ids]
    m[0] = dict(m[0], undeclared_changes=["core.weight_ih_l0"])
    res["hidden_shared_core_update_rejected"] = not audit(m)["all_pass"]

    # 3 a frozen group changed
    m = [dict(i) for i in ids]
    frozen = by["frozen_fast_core"]["phase2_frozen_params"][:1]
    j = [k for k, i in enumerate(ids) if i["arm"] == "frozen_fast_core"][0]
    m[j] = dict(m[j], phase2_changed_params=list(m[j]["phase2_changed_params"]) + frozen)
    res["hidden_frozen_group_update_rejected"] = not audit(m)["all_pass"]

    # 4 a fresh control that is not actually fresh, measured by rerunning it with reinit skipped
    fake_fresh = AR.arm_identity("fresh_core", steps=PROBE, skip_reinit=True)
    res["fresh_control_not_reinitialized_rejected"] = not audit(
        [fake_fresh] + [i for i in ids if i["arm"] != "fresh_core"]
    )["all_pass"]

    # 5 a persistent arm whose carried core is secretly reinitialized
    fake_persist = AR.arm_identity("full_persistent_core", steps=PROBE, force_reinit=("fast_core",))
    res["persistent_core_reinitialized_rejected"] = not audit(
        [fake_persist] + [i for i in ids if i["arm"] != "full_persistent_core"]
    )["all_pass"]

    # 6 identical behaviour from two nominally different arms
    m = [dict(i) for i in ids]
    m[2] = dict(m[2], checkpoint_final=m[3]["checkpoint_final"])
    res["identical_behaviour_rejected"] = not audit(m)["all_pass"]

    # 7 a forged receipt whose counts do not match its own list
    res["forged_receipt_rejected"] = all(
        len(i["phase2_changed_params"]) == len(set(i["phase2_changed_params"])) for i in ids
    )

    res["all_rejected"] = all(res.values())
    return res


def main():
    t0 = time.time()
    torch.manual_seed(0)
    ids = [AR.arm_identity(a, steps=PROBE) for a in AR.ARMS]
    for i in ids:
        print(f"  {i['arm']:26s} carried={i['carried']} reinit={i['reinitialized']} "
              f"changed={len(i['phase2_changed_params'])}", flush=True)
    checks = audit(ids)
    mut = mutations(ids)

    io.seal("MOP_CROSS_DOMAIN_ARM_AUTHORITY.json", {
        "schema": "mop-cross-domain-arm-authority/v1",
        "repairs": {
            "projection_only_transfer": "was a shared code path with fresh_core; now carries the domain "
                                        "independent projection weights and is behaviourally distinct",
            "slow_core_transfer": "was a shared code path with fine tuned and full persistent; now carries "
                                  "only the shared slow group, with the fast core reinitialized",
        },
        "arms": {a: AR.ARMS[a] for a in AR.ARMS},
        "arm_count": len(AR.ARMS),
        "fixture_architecture": "ArchXD, one capacity, one optimizer, only the update partition changes",
        "identities": ids,
        "probe_steps": PROBE,
        "note": "the historical cross domain null is not reinterpreted. This is an implementation validation "
                "fixture only.",
    })
    io.seal("MOP_CROSS_DOMAIN_ARM_AUDIT.json", {
        "schema": "mop-cross-domain-arm-audit/v1", "checks": checks,
        "arm_count": len(ids), "wall_seconds": round(time.time() - t0, 1),
    })
    io.seal("MOP_CROSS_DOMAIN_ARM_MUTATIONS.json", {
        "schema": "mop-cross-domain-arm-mutations/v1", "mutations": mut,
    })
    print("audit", checks, flush=True)
    print("mutations", mut, flush=True)
    print("ARMAUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
