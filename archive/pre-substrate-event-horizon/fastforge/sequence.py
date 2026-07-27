"""The general four phase cross domain sequence, shared by Architecture G, Architecture H and every
conventional comparison arm.

    phase 1  acquire domain A
    phase 2  acquire domain B, under the arm update partition
    phase 3  return to domain A
    phase 4  adapt to future units of domain B

Every arm is one declarative policy over the same engine, the same data splits and the same optimizer. That
is what makes an arm comparison a controlled comparison rather than a comparison of implementations.

Tuning signals come from the tune split only. The test split is never read by a policy, only by the final
evaluation.

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

# Per domain budgets come from the baseline convergence curves, not from convenience. A domain whose
# baselines are still improving is not a domain where any substrate result is terminal.
BUDGET = {
    "har": 550,
    "speech": 1200,
    "speech_stream": 1200,
    "har_stream": 1200,
    "harth_transition": 800,
    "pamap2_transition": 800,
}
LR = {
    "har": 3e-3,
    "speech": 3e-3,
    "speech_stream": 3e-3,
    "har_stream": 3e-3,
    "harth_transition": 3e-3,
    "pamap2_transition": 3e-3,
}
RETURN_FRACTION = 3
MEM_CAP = 600


def budget(dom, phase="acquire"):
    b = BUDGET.get(dom, 800)
    return b if phase == "acquire" else max(30, b // RETURN_FRACTION)


def lr_for(dom):
    return LR.get(dom, 1e-3)


def _sha(v) -> str:
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]


def local_groups(model, d, kinds=("proj_conv", "proj_lin", "adapter", "norm", "local_slow", "head")):
    return [f"{k}.{d}" for k in kinds if f"{k}.{d}" in model.param_groups]


def shared_group_name(model) -> str | None:
    """The reopenable shared group. G-R3 splits the core, and only the hidden to hidden half is reopenable,
    so the input to hidden half is trained once in phase 1 and frozen for the rest of the sequence."""
    for g in ("fast_core", "fast_delta", "fast_core_hh"):
        if g in model.param_groups:
            return g
    return None


def reinit(model, groups, seed):
    torch.manual_seed(90000 + seed)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for g in groups:
            for n in model.param_groups[g]:
                p = named[n]
                torch.nn.init.xavier_uniform_(p) if p.dim() >= 2 else torch.nn.init.zeros_(p)


# ---------------------------------------------------------------- arm policies

# arch          which architecture to build
# carry         True keeps the shared group across the domain boundary, False reinitializes it
# copy_proj     copy the domain independent projection weights from A to B
# shared_p2     shared group trainable during second domain acquisition
# shared_p3     shared group trainable on return to the first domain
# gate          simple rule controlling shared updates during phase 2, None means no gate
# thresh        gate threshold
# local_kinds   which domain local groups may train (None means all of them)
# memory        replay policy, matched budget
# ewc           EWC penalty weight on the shared group
ARMS: dict[str, dict] = {
    # required conventional and control arms of the bidirectional matrix
    "fresh_independent": dict(arch="G", carry=False, shared_p2=True, shared_p3=True),
    "separate_per_domain": dict(arch="separate", carry=True, shared_p2=True, shared_p3=True),
    "projection_only_transfer": dict(arch="G", carry=False, copy_proj=True, shared_p2=True, shared_p3=True),
    "lstm_gdumb": dict(arch="lstm", carry=True, shared_p2=True, shared_p3=True, memory="gdumb"),
    "matched_capacity_multi_domain": dict(
        arch="matched_capacity", carry=True, shared_p2=True, shared_p3=True
    ),
    "shared_adapters_conventional": dict(arch="shared_adapters", carry=True, shared_p2=True, shared_p3=True),
    "ewc_shared_core": dict(arch="lstm", carry=True, shared_p2=True, shared_p3=True, ewc=50.0),
    # Architecture G required arms
    "G0_always_trainable": dict(arch="G", carry=True, shared_p2=True, shared_p3=True),
    "G1_frozen_after_first": dict(arch="G", carry=True, shared_p2=False, shared_p3=False),
    "G2_reopened_at_return": dict(arch="G", carry=True, shared_p2=False, shared_p3=True),
    "G3_reopened_on_drop": dict(
        arch="G", carry=True, shared_p2=True, shared_p3=True, gate="perf", thresh=0.02
    ),
    "G4_adapters_only": dict(
        arch="G", carry=True, shared_p2=False, shared_p3=False, local_kinds=("adapter", "norm", "head")
    ),
    "G5_projection_and_head": dict(
        arch="G", carry=True, shared_p2=False, shared_p3=False, local_kinds=("proj_conv", "proj_lin", "head")
    ),
    "G6_oracle_schedule": dict(arch="G", carry=True, shared_p2=True, shared_p3=True, oracle=True),
    # Architecture H required arms and controls
    "H_always_update": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="always"),
    "H_never_update": dict(arch="H", carry=True, shared_p2=False, shared_p3=False, gate="never"),
    "H_cosine_gate": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="cosine", thresh=0.0),
    "H_probe_gate": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="probe", thresh=0.05),
    "H_drift_gate": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="drift", thresh=1.0),
    "H_perf_gate": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="perf", thresh=0.02),
    "H_random_gate": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="random"),
    "H_shuffled_gate": dict(
        arch="H", carry=True, shared_p2=True, shared_p3=True, gate="shuffled", shuffle_of="H_cosine_gate"
    ),
    "H_wrong_domain_gate": dict(
        arch="H", carry=True, shared_p2=True, shared_p3=True, gate="wrong", thresh=0.0
    ),
    "H_task_label_freeze": dict(arch="H", carry=True, shared_p2=False, shared_p3=True),
    "H_oracle_schedule": dict(arch="H", carry=True, shared_p2=True, shared_p3=True, oracle=True),
}


def run(
    arm: str,
    direction: tuple[str, str],
    seed: int,
    steps=None,
    override: dict | None = None,
    replay_decisions=None,
    memory_cap=MEM_CAP,
    scale=1.0,
):
    p = dict(ARMS[arm]) if arm in ARMS else {}
    if override:
        p.update(override)
    a, b = direction
    steps = steps or {
        "acquire": int(scale * budget(a)),
        "second": int(scale * budget(b)),
        "return": int(scale * budget(a, "r")),
        "adapt": int(scale * budget(b, "r")),
    }
    sa, sb = D.splits(a, seed), D.splits(b, seed)
    torch.manual_seed(1000 + seed)
    model = A.build(p["arch"], {a: (sa["channels"], sa["classes"]), b: (sb["channels"], sb["classes"])})
    shared = shared_group_name(model)
    rng = np.random.default_rng(seed)
    kinds = p.get("local_kinds") or ("proj_conv", "proj_lin", "adapter", "norm", "local_slow", "head")
    la, lb = local_groups(model, a, kinds), local_groups(model, b, kinds)
    if p["arch"] == "separate":  # per domain cores, there is no shared group to partition
        la, lb = la + [f"fast_core.{a}"], lb + [f"fast_core.{b}"]
        shared = None
    # memory is domain local: sequences from different domains have different channel counts and lengths,
    # so a shared buffer is not merely unhelpful, it is not representable. Budget is matched per domain.
    mems = {
        a: E.Memory(p.get("memory", "gdumb"), memory_cap),
        b: E.Memory(p.get("memory", "gdumb"), memory_cap),
    }
    receipts, m = [], {}

    # phase 1
    g1 = la + ([shared] if shared else []) + [g for g in p.get("phase1_extra", []) if g in model.param_groups]
    receipts.append(
        E.fit(
            model,
            a,
            *sa["main"],
            train_groups=g1,
            steps=steps["acquire"],
            lr=lr_for(a),
            rng=rng,
            memory=mems[a],
            update_recent=True,
        )
    )
    mems[a].add(*sa["main"], rng)
    m["first_acquisition"] = E.evaluate(model, a, *sa["test"])
    m["first_tune"] = E.evaluate(model, a, *sa["tune"])
    sha1 = E.checkpoint_sha(model)
    gsha1 = E.group_sha(model, sorted(model.param_groups))

    # boundary policy
    dropped = []
    if shared and not p.get("carry", True):
        reinit(model, [shared], seed)
        dropped = [shared]
    if p.get("copy_proj"):
        with torch.no_grad():
            model.projs[b].lin.weight.copy_(model.projs[a].lin.weight)
            model.projs[b].lin.bias.copy_(model.projs[a].lin.bias)
    gshaB = E.group_sha(model, sorted(model.param_groups))

    # gate wiring: the reference signal always comes from the tune split of the reference domain
    gate = ewc = None
    ref_batch = ref_domain = None
    if p.get("gate") and shared:
        gate = E.Gate(
            p["gate"], p.get("thresh", 0.0), np.random.default_rng(500 + seed), rate=p.get("rate", 0.5)
        )
        rdom = b if p["gate"] == "wrong" else a
        rx, ry = (sb if p["gate"] == "wrong" else sa)["tune"]
        bi = rng.choice(len(rx), min(128, len(rx)), replace=False)
        ref_batch, ref_domain = (rx[bi], ry[bi]), rdom
        gate.set_reference(
            E.reference_gradient(model, model.param_groups[shared], rx[bi], ry[bi], rdom),
            E.probe_loss(model, rx[bi], ry[bi], rdom),
        )
        if replay_decisions is not None:
            gate.replay(replay_decisions)
    if p.get("ewc") and shared:
        ewc = E.fisher_diag(model, a, *sa["main"], model.param_groups[shared], rng)

    # phase 2
    g2 = lb + ([shared] if shared and p.get("shared_p2", True) else [])
    receipts.append(
        E.fit(
            model,
            b,
            *sb["main"],
            train_groups=g2,
            steps=steps["second"],
            lr=lr_for(b),
            rng=rng,
            memory=mems[b],
            gate=gate,
            shared_group=shared,
            ref_batch=ref_batch,
            ref_domain=ref_domain,
            ewc=ewc,
            ewc_lambda=p.get("ewc", 0.0),
            update_recent=True,
        )
    )
    mems[b].add(*sb["main"], rng)
    m["second_acquisition"] = E.evaluate(model, b, *sb["test"])
    m["second_tune"] = E.evaluate(model, b, *sb["tune"])
    m["first_retention"] = E.evaluate(model, a, *sa["test"])

    # phase 3
    g3 = la + ([shared] if shared and p.get("shared_p3", True) else [])
    receipts.append(
        E.fit(
            model,
            a,
            *sa["main"],
            train_groups=g3,
            steps=steps["return"],
            lr=lr_for(a),
            rng=rng,
            memory=mems[a],
            update_recent=True,
        )
    )
    m["return_recovery"] = E.evaluate(model, a, *sa["test"])
    m["return_tune"] = E.evaluate(model, a, *sa["tune"])

    # phase 4
    g4 = lb + ([shared] if shared and p.get("shared_p2", True) else [])
    receipts.append(
        E.fit(
            model,
            b,
            *sb["adapt"],
            train_groups=g4,
            steps=steps["adapt"],
            lr=lr_for(b),
            rng=rng,
            memory=mems[b],
            update_recent=True,
        )
    )
    m["future_adaptation"] = E.evaluate(model, b, *sb["adapt_eval"])
    m["second_after_return"] = E.evaluate(model, b, *sb["test"])

    m["negative_transfer"] = m["first_acquisition"] - m["first_retention"]
    m["backward_transfer"] = m["return_recovery"] - m["first_acquisition"]
    m["forward_transfer"] = m["second_acquisition"]
    m["params"] = A.count_params(model)
    m["checkpoint_bytes"] = int(sum(p_.numel() * p_.element_size() for p_ in model.parameters()))
    m["train_updates"] = sum(r["updates"] for r in receipts)
    m["train_seconds"] = round(sum(r["wall_seconds"] for r in receipts), 2)
    m["trainable_phase2"] = receipts[1]["trainable_param_count"]
    m["changed_phase2"] = receipts[1]["changed_param_count"]
    # tuning utility: the only score any policy or oracle is allowed to select on
    m["tune_utility"] = float(np.mean([m["second_tune"], m["return_tune"]]))

    return {
        "arm": arm,
        "arch": p["arch"],
        "direction": f"{a}->{b}",
        "seed": seed,
        "policy": {k: v for k, v in p.items()},
        "policy_sha": _sha({**p, "arm": arm}),
        "metrics": m,
        "receipts": receipts,
        "shared_group": shared,
        "carried": [] if dropped else ([shared] if shared else []),
        "reinitialized_at_boundary": dropped,
        "shared_changed_at_boundary": sorted(g for g in gsha1 if gsha1[g] != gshaB[g]),
        "gate_decisions": None if gate is None else gate.decisions,
        "gate_block_rate": None
        if gate is None
        else round(float(np.mean([1 - d for d in gate.decisions])) if gate.decisions else 0.0, 4),
        "checkpoint_after_first_domain": sha1,
        "checkpoint_final": E.checkpoint_sha(model),
        "undeclared_changes": sorted({n for r in receipts for n in r["undeclared_changes"]}),
    }
