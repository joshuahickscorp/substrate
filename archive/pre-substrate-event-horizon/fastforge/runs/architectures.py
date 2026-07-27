"""Seal the two owned architectures: declared state, parameter group partition, update policy and arms.

This is a specification artifact, not a result. It exists so that every later claim about which parameters
persisted, which stayed domain local and which were frozen can be checked against a declaration made before
the results existed.

House style: no dashes.
"""

from __future__ import annotations

import inspect

import torch

from fastforge import arch as A
from fastforge import sequence as S
from fastforge.runs import io

DOMS = {"har": (9, 6), "speech": (40, 10)}


def inventory(kind):
    m = A.build(kind, DOMS)
    ok, undeclared, duplicated = A.check_partition(m)
    named = dict(m.named_parameters())
    groups = {
        g: {"parameters": sorted(ns), "count": int(sum(named[n].numel() for n in ns))}
        for g, ns in m.param_groups.items()
    }
    buffers = {n: list(b.shape) for n, b in m.named_buffers()}
    return {
        "total_parameters": A.count_params(m),
        "trainable_parameters": int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
        "parameter_groups": groups,
        "non_trainable_buffers": buffers,
        "partition_is_exact": ok,
        "undeclared_parameters": undeclared,
        "duplicated_parameters": duplicated,
    }


def main():
    torch.manual_seed(0)
    g_arms = {k: v for k, v in S.ARMS.items() if k.startswith("G")}
    h_arms = {k: v for k, v in S.ARMS.items() if k.startswith("H")}
    io.seal(
        "MOP_ARCHITECTURE_G.json",
        {
            "schema": "mop-architecture-g/v1",
            "name": "Fast Shared Core with Domain Local Plasticity",
            "premise": "a persistent fast temporal representation transfers across domains while domain "
            "local "
            "adapters absorb domain specific slow change",
            "owned_state": {
                "domain specific trainable projection": "Projection, conv over channels plus a shape shared"
                " linear",
                "shared causal fast core": "single layer GRU over the projected latent",
                "domain local slow adapter": "bottleneck residual, zero initialized so it starts as identity",
                "domain local normalization": "LayerNorm per domain",
                "domain local task head": "linear per domain",
                "bounded recent state": "non trainable EMA buffer",
                "simple episodic memory": "GDumb over sequences, matched budget, no learned selector",
                "checkpointed domain state": "group level content hashes recorded in every receipt",
            },
            "explicitly_absent": [
                "shared medium state",
                "shared trainable slow workspace",
                "predictive auxiliary objective",
                "learned replay or retrieval",
            ],
            "absence_reason": "each was closed by sealed prior evidence and is not reopened",
            "default_update_policy": {
                "shared fast core": "trains during the first domain, frozen during initial second domain "
                "acquisition, reopened only under a declared rule",
                "domain local adapter": "trains on the active domain",
                "active head": "trains",
                "inactive domain adapters and heads": "frozen",
            },
            "arms": g_arms,
            "inventory": inventory("G"),
            "implementation": {
                "module": "fastforge/arch.py",
                "class": "ArchG",
                "loc": len(inspect.getsource(A.ArchG).splitlines()),
            },
        },
    )
    io.seal(
        "MOP_ARCHITECTURE_H.json",
        {
            "schema": "mop-architecture-h/v1",
            "name": "Anchored Fast Dynamics with Interference Gated Adaptation",
            "premise": "shared plasticity is safe when it is explicitly bounded and reversible, and is "
            "opened "
            "only when the measured interference signal permits it",
            "materially_different_from_G": [
                "the shared core is a frozen anchor plus a bounded trainable delta, not a directly "
                "trainable core",
                "the delta is bounded by a smooth tanh bound of tau, so shared drift has a hard ceiling",
                "the delta can be reverted toward the anchor, so shared plasticity is reversible",
                "domain local capacity is low rank rather than a bottleneck residual",
                "the update decision is driven by measured interference, not by a fixed schedule",
            ],
            "owned_state": {
                "shared fast dynamics core": "GRU parameters evaluated functionally as anchor plus bounded "
                "delta",
                "frozen reference anchor": "registered buffers, never trainable",
                "trainable fast delta": f"parameter per anchor tensor, bounded by tau = {A.TAU}",
                "domain local low rank adapters": "rank 8 residual per domain",
                "domain local heads": "linear per domain",
                "interference statistics": "gradient conflict, reference probe loss, weight space drift",
            },
            "gate_rules": {k: "simple rule, no learned parameters" for k in S.E.Gate.KINDS},
            "learned_gate_policy": "a learned gate opens only through a stable headroom authority",
            "arms": h_arms,
            "inventory": inventory("H"),
            "implementation": {
                "module": "fastforge/arch.py",
                "class": "ArchH",
                "loc": len(inspect.getsource(A.ArchH).splitlines()),
            },
        },
    )
    for k in ("G", "H"):
        inv = inventory(k)
        print(
            k,
            "params",
            inv["total_parameters"],
            "partition_exact",
            inv["partition_is_exact"],
            "groups",
            len(inv["parameter_groups"]),
            flush=True,
        )
    print("ARCHITECTURES_DONE", flush=True)


if __name__ == "__main__":
    main()
