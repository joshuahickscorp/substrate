"""Wave G science node: the value of a role-indexed binding over a role-blind bag-of-symbols reader.

Question. A sequence is a set of role-to-filler bindings, and the SAME filler appears in different roles
across sequences, so a symbol alone never determines its role. The candidate is a role-indexed binding
representation that keeps the role tag beside each filler. The named control is a bag-of-symbols reader that
discards role tags and answers a query about role r from the multiset of fillers plus a training-learned
role affinity. We ask whether the binding representation beats the bag reader on role-swapped held-out
queries, where fillers are deliberately placed in roles other than the ones they typically occupy.

Named control (the bar the candidate must clear).
  * bag-of-symbols reader : encode the observed sequence as its multiset of fillers, discarding role tags,
    then answer "what fills role r?" by picking the in-bag filler with the highest training affinity for r.

Design. Each experimental unit is an independent synthetic task with its own seed and its own random
role-affinity structure. Training sequences are role-respecting, so the bag reader can learn which filler
typically fills which role. Held-out test sequences are role-swapped: fillers are placed in non-typical
roles. The observation is lossy for BOTH readers in the same way (each slot's filler is corrupted with a
small probability), and the ground truth is the original, pre-corruption binding. The per-unit paired delta
is candidate accuracy minus control accuracy on the SAME queries, so a positive delta favors the binding
representation. Over N units we run the framework exact one-sided sign-flip and derive the neutral verdict
against a small structural SESOI. The candidate never trains, so any win is not bought with extra compute;
the matched-budget accounting records this.

Precondition (negative control). A separate rigid regime removes the role swap: test sequences are
consistent (each filler sits in its typical role) and carry exactly one filler per role class. There the
bag reader can invert the bag back to the binding, so both readers are correct on exactly the same
uncorrupted queries and the delta collapses to about zero. This is the failure domain: when the bag already
determines the binding, a role-indexed representation buys nothing.

Honesty. A tie or a wrong-direction result is a legitimate null; nothing here is tuned toward a positive.
One run is evidence level M1: consistent with, never a scientific confirmation.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    LifecycleCost,
    assert_matched_budget,
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

_N_UNITS = 10
_N_ROLES = 5
_POOL = 16
_N_TRAIN = 260
_N_TEST = 260
_N_TEST_NEG = 600
# Per-slot filler corruption probability, applied identically to what both readers observe.
_NOISE = 0.05
# Structural SESOI in accuracy points: a gain below two points is treated as no effect.
_SESOI = 0.02


def _draw_typ(gen: np.random.Generator, pool: int, n_roles: int) -> np.ndarray:
    """Typical role per filler. The first n_roles ids seed one filler per role so no class is empty."""

    typ = np.empty(pool, dtype=np.int64)
    typ[:n_roles] = np.arange(n_roles)
    typ[n_roles:] = gen.integers(0, n_roles, pool - n_roles)
    return typ


def _assign(
    gen: np.random.Generator, fillers: np.ndarray, typ: np.ndarray, n_roles: int, respect: bool
) -> np.ndarray:
    """Bijective role-to-filler assignment over ``fillers``. respect=True prefers a filler whose typical
    role matches the slot; respect=False prefers a filler whose typical role differs (the role swap)."""

    order_roles = gen.permutation(n_roles)
    order_fill = gen.permutation(len(fillers))
    perm = fillers[order_fill]
    used = np.zeros(len(perm), dtype=bool)
    binding = np.full(n_roles, -1, dtype=np.int64)
    for r in order_roles:
        avail = np.nonzero(~used)[0]
        if respect:
            pref = [i for i in avail if typ[perm[i]] == r]
        else:
            pref = [i for i in avail if typ[perm[i]] != r]
        chosen = pref[0] if pref else int(avail[0])
        binding[r] = perm[chosen]
        used[chosen] = True
    return binding


def _corrupt(gen: np.random.Generator, binding: np.ndarray, pool: int, noise: float) -> np.ndarray:
    """Return the observed binding: each slot is replaced by a fresh out-of-bag filler with prob ``noise``.

    Both readers see this same observed binding; the ground truth stays the original, pre-corruption filler.
    """

    present = {int(x) for x in binding}
    observed = binding.copy()
    for r in range(len(observed)):
        if gen.random() < noise:
            old = int(observed[r])
            f = int(gen.integers(0, pool))
            tries = 0
            while f in present and tries < 4 * pool:
                f = int(gen.integers(0, pool))
                tries += 1
            if f in present:
                continue
            present.discard(old)
            present.add(f)
            observed[r] = f
    return observed


def _train_affinity(
    gen: np.random.Generator, typ: np.ndarray, pool: int, n_roles: int, n_train: int
) -> np.ndarray:
    """Learn the role affinity counts[f, r] from role-respecting, uncorrupted training sequences."""

    counts = np.zeros((pool, n_roles), dtype=np.int64)
    for _ in range(n_train):
        fillers = gen.choice(pool, n_roles, replace=False)
        binding = _assign(gen, fillers, typ, n_roles, respect=True)
        for r in range(n_roles):
            counts[binding[r], r] += 1
    return counts


def _score(
    gen: np.random.Generator,
    typ: np.ndarray,
    counts: np.ndarray,
    pool: int,
    n_roles: int,
    n_test: int,
    noise: float,
    swapped: bool,
    rigid: bool,
) -> tuple[float, float]:
    """Accuracy of the binding reader and the bag reader over ``n_test`` role queries.

    swapped selects role-swapped held-out bags; rigid draws one filler per role class (used by the
    precondition). The binding reader answers the observed role tag; the bag reader answers the in-bag
    filler with the highest training affinity for the queried role, tie-broken by smallest id.
    """

    members = [np.nonzero(typ == r)[0] for r in range(n_roles)]
    cand_correct = 0
    ctrl_correct = 0
    for _ in range(n_test):
        if rigid:
            fillers = np.array([int(gen.choice(members[r])) for r in range(n_roles)], dtype=np.int64)
        else:
            fillers = gen.choice(pool, n_roles, replace=False)
        binding = _assign(gen, fillers, typ, n_roles, respect=not swapped)
        observed = _corrupt(gen, binding, pool, noise)
        r = int(gen.integers(0, n_roles))
        truth = int(binding[r])
        cand = int(observed[r])
        aff = counts[observed, r]
        ctrl = int(observed[aff == aff.max()].min())
        cand_correct += int(cand == truth)
        ctrl_correct += int(ctrl == truth)
    return cand_correct / n_test, ctrl_correct / n_test


@register_runner("wave_g.variable_binding")
def wave_g_variable_binding_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Real, deterministic Wave G mechanics: does a role-indexed binding beat a role-blind bag reader?"""

    n_units = int(params.get("n_units", _N_UNITS))
    pool = int(params.get("pool", _POOL))
    n_roles = int(params.get("n_roles", _N_ROLES))
    n_train = int(params.get("n_train", _N_TRAIN))
    n_test = int(params.get("n_test", _N_TEST))
    noise = float(params.get("noise", _NOISE))

    units: list[dict[str, Any]] = []
    for u in range(n_units):
        gen = rng(ctx.seed, "wave_g_swap", u)
        typ = _draw_typ(gen, pool, n_roles)
        counts = _train_affinity(gen, typ, pool, n_roles, n_train)
        cand_acc, ctrl_acc = _score(gen, typ, counts, pool, n_roles, n_test, noise, swapped=True, rigid=False)
        units.append(
            {
                "unit_id": f"swap-{u:02d}",
                "acc_candidate": round(cand_acc, 12),
                "acc_control_bag": round(ctrl_acc, 12),
                "delta": round(cand_acc - ctrl_acc, 12),
            }
        )

    deltas = [unit["delta"] for unit in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], _SESOI)
    is_null = verdict != "survives"

    neg_gen = rng(ctx.seed, "wave_g_rigid_consistent")
    neg_typ = _draw_typ(neg_gen, pool, n_roles)
    neg_counts = _train_affinity(neg_gen, neg_typ, pool, n_roles, n_train)
    neg_cand, neg_ctrl = _score(
        neg_gen, neg_typ, neg_counts, pool, n_roles, _N_TEST_NEG, noise, swapped=False, rigid=True
    )
    neg_delta = neg_cand - neg_ctrl
    advantage_absent = neg_delta < _SESOI

    # Matched-budget accounting: both readers consider n_roles fillers per query. The bag reader must first
    # train its affinity counts; the candidate keeps the binding directly and never trains, so it is the
    # cheaper arm and any win is not bought with extra compute.
    costs = {
        "candidate_binding": LifecycleCost(train_flops=0, inference_flops=n_roles * n_test),
        "control_bag": LifecycleCost(train_flops=n_roles * n_train, inference_flops=n_roles * n_test),
    }
    budget = assert_matched_budget(costs, ceiling=n_roles * (n_train + n_test))

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_g_variable_binding/v1",
        {
            "form_family": "symbolic",
            "phenomenon": "role_filler_binding",
            "mechanism_family": "relation_state",
            "unit_class": "synthetic_role_filler_task",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "role-indexed binding: keep the role tag beside each filler and answer a role "
            "query by reading the filler bound to that role",
            "named_control": "bag-of-symbols reader: discard role tags, encode the sequence as its multiset "
            "of fillers, and answer role r by the in-bag filler with the highest training affinity for r",
            "control_description": "per-unit delta is candidate accuracy minus bag-of-symbols accuracy on "
            "the same role-swapped held-out queries; positive favors the binding representation",
            "config": {
                "n_units": n_units,
                "pool": pool,
                "n_roles": n_roles,
                "n_train": n_train,
                "n_test": n_test,
                "n_test_negative": _N_TEST_NEG,
                "noise": noise,
                "sesoi": _SESOI,
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": _SESOI,
            "verdict": verdict,
            "is_null": is_null,
            "negative_control": {
                "acc_candidate": round(neg_cand, 12),
                "acc_control_bag": round(neg_ctrl, 12),
                "delta": round(neg_delta, 12),
                "description": "rigid consistent regime: no role swap and one filler per role class, so the "
                "bag already determines the binding and both readers agree on the uncorrupted queries",
                "advantage_absent": advantage_absent,
            },
            "matched_budget": budget,
            "alternative_explanation": "The gain reflects the binding reader retaining the role tags that "
            "the bag reader deliberately discards; it is bounded to the role-swapped regime where a symbol "
            "alone does not determine its role and collapses to a tie when the bag already fixes the "
            "binding.",
            "failure_domain": "When roles are rigid and unswapped so the multiset of fillers determines the "
            "binding, the bag-of-symbols reader inverts the bag and matches the binding reader; the "
            "negative control ties at delta about zero.",
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "negative_control_delta": round(neg_delta, 12),
            "advantage_absent": advantage_absent,
        },
    )
