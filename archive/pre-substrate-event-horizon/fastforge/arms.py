"""The eleven repaired cross domain transfer arms.

The prior matrix aliased arms: projection_only shared a code path with fresh_core, and slow_core_transfer
shared one with fine_tuned and full_persistent. Repair means every arm now differs in at least one of

    carried groups | reinitialized groups | trainable groups per phase | warmup schedule | learning rate

and the difference is proven behaviourally, by distinct trainable inventories and distinct checkpoint hashes,
not merely declared in a name. Arms are configurations over one fixture architecture (ArchXD) because that is
what makes the comparison controlled: the same capacity, the same data, the same optimizer, and only the
update partition changes.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from . import arch as A
from . import data as D
from . import engine as E

SHARED = ("fast_core", "slow_core")
PHASE_STEPS = {"acquire": 150, "second": 150, "return": 50, "adapt": 50}


def spec_sha(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# carried: shared groups whose phase 1 weights survive into phase 2
# frozen_shared: shared groups present but not trainable during phase 2
# copy_proj: the domain independent part of the projection is copied from the first domain
# warmup: fraction of phase 2 during which every shared group is frozen, then reopened at lr_late
ARMS: dict[str, dict] = {
    "fresh_core": dict(carried=[], frozen_shared=[], copy_proj=False, warmup=0.0, lr_late=None),
    "projection_only_transfer": dict(carried=[], frozen_shared=[], copy_proj=True, warmup=0.0, lr_late=None),
    "fast_core_transfer": dict(
        carried=["fast_core"], frozen_shared=[], copy_proj=False, warmup=0.0, lr_late=None
    ),
    "slow_core_transfer": dict(
        carried=["slow_core"], frozen_shared=[], copy_proj=False, warmup=0.0, lr_late=None
    ),
    "frozen_fast_core": dict(
        carried=["fast_core"], frozen_shared=["fast_core"], copy_proj=False, warmup=0.0, lr_late=None
    ),
    "frozen_slow_core": dict(
        carried=["slow_core"], frozen_shared=["slow_core"], copy_proj=False, warmup=0.0, lr_late=None
    ),
    "fine_tuned_fast_core": dict(
        carried=["fast_core"], frozen_shared=[], copy_proj=False, warmup=0.5, lr_late=1e-4
    ),
    "fine_tuned_slow_core": dict(
        carried=["slow_core"], frozen_shared=[], copy_proj=False, warmup=0.5, lr_late=1e-4
    ),
    "full_persistent_core": dict(
        carried=["fast_core", "slow_core"], frozen_shared=[], copy_proj=True, warmup=0.0, lr_late=None
    ),
    "domain_local_adapter": dict(
        carried=["fast_core", "slow_core"],
        frozen_shared=["fast_core", "slow_core"],
        copy_proj=False,
        warmup=0.0,
        lr_late=None,
        local_only=["adapter", "head", "proj"],
    ),
    "domain_local_slow_state": dict(
        carried=["fast_core", "slow_core"],
        frozen_shared=["fast_core", "slow_core"],
        copy_proj=False,
        warmup=0.0,
        lr_late=None,
        local_only=["local_slow", "head", "proj"],
    ),
}


def reinit_groups(model, groups, seed: int):
    """Fresh initialization of the named groups. A fresh control must actually be fresh."""
    torch.manual_seed(90000 + seed)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for g in groups:
            for n in model.param_groups[g]:
                p = named[n]
                if p.dim() >= 2:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)


def local_groups(model, d: str, kinds=("proj_conv", "proj_lin", "adapter", "local_slow", "norm", "head")):
    return [f"{k}.{d}" for k in kinds if f"{k}.{d}" in model.param_groups]


def _phase_data(name: str, seed: int):
    """Return (main_x, main_y, test_x, test_y, adapt_x, adapt_y, adapt_eval_x, adapt_eval_y).

    main pool excludes the future units entirely. The adaptation pool is a row split inside the future units:
    adapt on some of their rows, evaluate on the rest, which is what future unit adaptation actually means.
    """
    dom = D.domain(name)
    main, future = D.held_out_units(name, seed)
    rng = np.random.default_rng(7000 + seed)
    fut = future[rng.permutation(len(future))]
    cut = int(0.7 * len(fut))
    return (
        dom["x"][main],
        dom["y"][main],
        dom["xte"],
        dom["yte"],
        dom["x"][fut[:cut]],
        dom["y"][fut[:cut]],
        dom["x"][fut[cut:]],
        dom["y"][fut[cut:]],
    )


def run_arm(
    arm: str,
    direction: tuple[str, str],
    seed: int,
    steps=None,
    arch="XD",
    skip_reinit: bool = False,
    force_reinit=(),
):
    """Four phase sequence: acquire A, acquire B, return to A, adapt to future units of B.

    skip_reinit and force_reinit exist only so the mutation suite can inject the two failures that matter,
    a fresh control that is not actually fresh and a persistent arm whose core is secretly reinitialized.
    Principal runs never set them.
    """
    steps = steps or PHASE_STEPS
    spec = ARMS[arm]
    a, b = direction
    torch.manual_seed(1000 + seed)
    da, db = D.domain(a), D.domain(b)
    model = A.build(arch, {a: (da["channels"], da["classes"]), b: (db["channels"], db["classes"])})
    rng = np.random.default_rng(seed)
    receipts = []

    aX, aY, aXte, aYte, *_ = _phase_data(a, seed)
    bX, bY, bXte, bYte, bAdX, bAdY, bEvX, bEvY = _phase_data(b, seed)

    # phase 1: acquire domain A with every shared and A local group trainable
    receipts.append(
        E.fit(
            model,
            a,
            aX,
            aY,
            train_groups=list(SHARED) + local_groups(model, a),
            steps=steps["acquire"],
            rng=rng,
        )
    )
    m = {"first_acquisition": E.evaluate(model, a, aXte, aYte)}
    sha_after_phase1 = E.checkpoint_sha(model)
    group_sha_phase1 = E.group_sha(model, sorted(model.param_groups))

    # arm policy applies at the domain boundary
    drop = [g for g in SHARED if g not in spec["carried"]]
    if drop and not skip_reinit:
        reinit_groups(model, drop, seed)
    if force_reinit:
        reinit_groups(model, list(force_reinit), seed)
    if spec["copy_proj"]:
        with torch.no_grad():
            model.projs[b].lin.weight.copy_(model.projs[a].lin.weight)
            model.projs[b].lin.bias.copy_(model.projs[a].lin.bias)
    group_sha_boundary = E.group_sha(model, sorted(model.param_groups))

    # phase 2: acquire domain B
    b_local = local_groups(model, b)
    if "local_only" in spec:
        b_local = [g for g in b_local if g.split(".")[0] in spec["local_only"] or g.startswith("proj")]
    trainable_shared = [g for g in SHARED if g not in spec["frozen_shared"]]
    if spec["warmup"] > 0:
        w = int(spec["warmup"] * steps["second"])
        receipts.append(E.fit(model, b, bX, bY, train_groups=b_local, steps=w, rng=rng))
        receipts.append(
            E.fit(
                model,
                b,
                bX,
                bY,
                train_groups=b_local + trainable_shared,
                steps=steps["second"] - w,
                lr=spec["lr_late"],
                rng=rng,
            )
        )
    else:
        receipts.append(
            E.fit(model, b, bX, bY, train_groups=b_local + trainable_shared, steps=steps["second"], rng=rng)
        )
    m["second_acquisition"] = E.evaluate(model, b, bXte, bYte)
    m["first_retention"] = E.evaluate(model, a, aXte, aYte)

    # phase 3: return to domain A. shared groups follow the same arm policy.
    receipts.append(
        E.fit(
            model,
            a,
            aX,
            aY,
            train_groups=local_groups(model, a) + trainable_shared,
            steps=steps["return"],
            rng=rng,
        )
    )
    m["return_recovery"] = E.evaluate(model, a, aXte, aYte)

    # phase 4: adapt to future units of domain B
    receipts.append(
        E.fit(model, b, bAdX, bAdY, train_groups=b_local + trainable_shared, steps=steps["adapt"], rng=rng)
    )
    m["future_adaptation"] = E.evaluate(model, b, bEvX, bEvY)
    m["second_after_adapt"] = E.evaluate(model, b, bXte, bYte)

    m["negative_transfer"] = m["first_acquisition"] - m["first_retention"]
    m["backward_transfer"] = m["return_recovery"] - m["first_acquisition"]
    m["params"] = A.count_params(model)
    m["trainable_phase2"] = receipts[-3]["trainable_param_count"]
    m["changed_phase2"] = receipts[-3]["changed_param_count"]
    return {
        "arm": arm,
        "arch": arch,
        "direction": f"{a}->{b}",
        "seed": seed,
        "spec": spec,
        "spec_sha": spec_sha({**spec, "arm": arm}),
        "metrics": m,
        "receipts": receipts,
        "checkpoint_after_first_domain": sha_after_phase1,
        "group_sha_after_first_domain": group_sha_phase1,
        "group_sha_at_boundary": group_sha_boundary,
        "shared_groups_changed_at_boundary": sorted(
            g for g in SHARED if group_sha_phase1[g] != group_sha_boundary[g]
        ),
        "checkpoint_final": E.checkpoint_sha(model),
        "undeclared_changes": sorted({n for r in receipts for n in r["undeclared_changes"]}),
        "carried_groups": sorted(spec["carried"]),
        "reinitialized_groups": sorted(drop),
    }


def arm_identity(arm: str, direction=("har", "speech"), seed=0, steps=None, **kw):
    """Cheap identity probe used by the non aliasing tests and by the arm audit."""
    r = run_arm(arm, direction, seed, steps or {"acquire": 6, "second": 6, "return": 3, "adapt": 3}, **kw)
    p2 = r["receipts"][-3]
    return {
        "arm": arm,
        "spec_sha": r["spec_sha"],
        "carried": r["carried_groups"],
        "reinitialized": r["reinitialized_groups"],
        "phase2_trainable_groups": p2["trainable_groups"],
        "phase2_trainable_params": p2["trainable_params"],
        "phase2_frozen_params": p2["frozen_params"],
        "phase2_changed_params": p2["changed_params"],
        "phase2_group_sha": p2["group_sha_after"],
        "checkpoint_after_first_domain": r["checkpoint_after_first_domain"],
        "shared_groups_changed_at_boundary": r["shared_groups_changed_at_boundary"],
        "checkpoint_final": r["checkpoint_final"],
        "undeclared_changes": r["undeclared_changes"],
        "identity_tuple": [
            arm,
            sorted(r["carried_groups"]),
            sorted(r["reinitialized_groups"]),
            sorted(p2["trainable_groups"]),
            r["spec"]["warmup"],
            r["spec"]["copy_proj"],
        ],
    }
