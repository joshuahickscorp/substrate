"""Wave E episodic memory: does episodic key-value retrieval recall rare specifics a semantic summary loses?

Complementary-learning-systems theory splits memory into a fast episodic store that keeps individual
experiences and a slow semantic store that compresses experience into prototypes. This bed asks the honest
version of the question: given the SAME stream of episodes, does episodic retrieval outperform a semantic
mean-summary on the queries where it should matter, namely rare specific episodes whose values violate the
common background structure? Predicting whether a semantic summary preserves the background regularities is
not the test (it trivially can). The test is whether it can recall a surprising one-off that averaging
washes away, and whether episodic retrieval can.

Design. Each of ten independent streams is a memory ecology. A common background is a handful of dense key
clusters whose values follow a smooth regional base. Sprinkled among them are a few rare specific episodes:
isolated keys whose stored values carry a large surprising offset relative to their local background base.
Both arms see the identical memory of (key, value) pairs.

* candidate: episodic key-value retrieval. For a query it returns the value of the single nearest stored
  episode by Euclidean key distance. Its store scales with experience, so a rare one-off remains addressable.
* named control: a semantic mean-summary predictor. It compresses the same memory into a fixed budget of C
  prototypes by deterministic k-means, storing each prototype's centroid and the MEAN value of the episodes
  assigned to it, then predicts the mean value of the nearest prototype. Rare singletons get absorbed into
  a background-dominated prototype and their surprising value is averaged out.

Score per unit is accuracy on rare-episode queries: a query is a tiny-noise probe of a rare episode's key,
its true value is that episode's value, and a prediction counts as correct when it lands within a small
tolerance. Per-unit delta is candidate accuracy minus control accuracy. The exact one-sided sign-flip over
the ten independent-unit deltas and the neutral SESOI verdict decide the outcome. If episodic only ties the
summary, that is a legitimate null. This bed does not tune toward a positive: when the semantic budget
happens to isolate a rare episode in its own prototype, the summary recovers it too and the delta shrinks.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

# --- Fixed design constants (no wall clock, no tuning knobs exposed to params). ---
N_UNITS = 10
DIM = 8
N_BG_CLUSTERS = 5
BG_PER_CLUSTER = 40  # 200 dense background episodes per unit
N_RARE = 8
QUERIES_PER_RARE = 5  # 40 rare-episode queries per unit
SEMANTIC_BUDGET_C = 8  # fixed prototype capacity of the semantic store, independent of experience
KMEANS_ITERS = 12
CLUSTER_SPREAD = 3.0  # separation of background cluster centers
BG_KEY_STD = 0.5  # within-cluster key scatter
BG_NOISE_STD = 0.10  # background value noise
RARE_OFFSET_LO = 3.0  # a rare episode's value departs from its local base by this much at least
RARE_OFFSET_HI = 5.0
QUERY_NOISE_STD = 0.01  # tiny probe noise, so a rare episode stays the nearest stored key
CORRECT_TOL = 0.5  # a prediction within this absolute value tolerance counts as correct
SESOI = 0.05  # smallest rare-query accuracy gain we would call a real effect
ALPHA = 0.05


def _build_memory(seed: int, u: int) -> dict[str, np.ndarray]:
    """Construct one unit's identical memory: dense background clusters plus rare surprising episodes."""

    g = rng(seed, "unit", u, "world")
    centers = g.standard_normal((N_BG_CLUSTERS, DIM)) * CLUSTER_SPREAD
    base_values = g.standard_normal(N_BG_CLUSTERS)  # smooth regional base value per cluster

    bg_keys = np.empty((N_BG_CLUSTERS * BG_PER_CLUSTER, DIM))
    bg_vals = np.empty(N_BG_CLUSTERS * BG_PER_CLUSTER)
    for k in range(N_BG_CLUSTERS):
        lo = k * BG_PER_CLUSTER
        hi = lo + BG_PER_CLUSTER
        bg_keys[lo:hi] = centers[k] + BG_KEY_STD * g.standard_normal((BG_PER_CLUSTER, DIM))
        bg_vals[lo:hi] = base_values[k] + BG_NOISE_STD * g.standard_normal(BG_PER_CLUSTER)

    rare_keys = g.standard_normal((N_RARE, DIM)) * CLUSTER_SPREAD
    nearest_center = np.argmin(((rare_keys[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    signs = g.choice(np.array([-1.0, 1.0]), size=N_RARE)
    offsets = signs * g.uniform(RARE_OFFSET_LO, RARE_OFFSET_HI, size=N_RARE)
    rare_vals = base_values[nearest_center] + offsets  # surprising relative to the local background base

    mem_keys = np.vstack([bg_keys, rare_keys])
    mem_vals = np.concatenate([bg_vals, rare_vals])
    return {"mem_keys": mem_keys, "mem_vals": mem_vals, "rare_keys": rare_keys, "rare_vals": rare_vals}


def _kmeans(keys: np.ndarray, vals: np.ndarray, c: int, gen: np.random.Generator) -> dict[str, np.ndarray]:
    """Deterministic k-means giving the semantic store: centroids plus the MEAN value of each prototype."""

    m = keys.shape[0]
    init = gen.choice(m, size=c, replace=False)
    centroids = keys[init].copy()
    assign = np.zeros(m, dtype=int)
    for _ in range(KMEANS_ITERS):
        dists = ((keys[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(dists, axis=1)
        for j in range(c):
            members = assign == j
            if np.any(members):
                centroids[j] = keys[members].mean(axis=0)
    proto_vals = np.empty(c)
    for j in range(c):
        members = assign == j
        proto_vals[j] = vals[members].mean() if np.any(members) else 0.0
    return {"centroids": centroids, "proto_vals": proto_vals}


def _episodic_predict(mem_keys: np.ndarray, mem_vals: np.ndarray, queries: np.ndarray) -> np.ndarray:
    dists = ((queries[:, None, :] - mem_keys[None, :, :]) ** 2).sum(axis=2)
    return mem_vals[np.argmin(dists, axis=1)]


def _semantic_predict(store: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    dists = ((queries[:, None, :] - store["centroids"][None, :, :]) ** 2).sum(axis=2)
    return store["proto_vals"][np.argmin(dists, axis=1)]


def _run_unit(seed: int, u: int) -> dict[str, Any]:
    mem = _build_memory(seed, u)
    store = _kmeans(mem["mem_keys"], mem["mem_vals"], SEMANTIC_BUDGET_C, rng(seed, "unit", u, "kmeans"))

    qg = rng(seed, "unit", u, "queries")
    queries = np.repeat(mem["rare_keys"], QUERIES_PER_RARE, axis=0)
    queries = queries + QUERY_NOISE_STD * qg.standard_normal(queries.shape)
    true_vals = np.repeat(mem["rare_vals"], QUERIES_PER_RARE)

    cand_pred = _episodic_predict(mem["mem_keys"], mem["mem_vals"], queries)
    ctrl_pred = _semantic_predict(store, queries)
    cand_acc = float(np.mean(np.abs(cand_pred - true_vals) <= CORRECT_TOL))
    ctrl_acc = float(np.mean(np.abs(ctrl_pred - true_vals) <= CORRECT_TOL))
    delta = cand_acc - ctrl_acc

    return {
        "unit_id": f"u{u:02d}",
        "n_rare_queries": int(queries.shape[0]),
        "n_semantic_prototypes": SEMANTIC_BUDGET_C,
        "episodic_rare_accuracy": round(cand_acc, 9),
        "semantic_rare_accuracy": round(ctrl_acc, 9),
        "delta": round(delta, 9),
    }


@register_runner("wave_e.episodic_vs_semantic")
def wave_e_episodic_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Deterministic Wave E bed: episodic key-value retrieval versus a semantic mean-summary on rare items."""

    units = [_run_unit(ctx.seed, u) for u in range(N_UNITS)]
    deltas = [u["delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    mean_delta = sign_flip["mean_delta"]
    verdict = verdict_from(mean_delta, sign_flip["one_sided_p"], SESOI, ALPHA)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_e_episodic/v1",
        {
            "form_family": "memory_episode",
            "phenomenon": "episodic_memory",
            "mechanism_family": "episodic_retrieval",
            "unit_class": "key_value_memory_ecology",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "episodic_nearest_episode_retrieval",
            "control": "semantic_mean_summary_prototypes",
            "control_description": (
                "The named control is a semantic mean-summary predictor. It compresses the identical memory "
                "into a fixed budget of C prototypes by deterministic k-means, stores each prototype's "
                "centroid and the MEAN value of its assigned episodes, and predicts the mean value of the "
                "nearest prototype. Both arms see the same episodes; only the summary is lossy for rare "
                "singletons that a background-dominated prototype averages away."
            ),
            "design": {
                "n_units": N_UNITS,
                "dim": DIM,
                "n_bg_clusters": N_BG_CLUSTERS,
                "bg_per_cluster": BG_PER_CLUSTER,
                "n_rare": N_RARE,
                "queries_per_rare": QUERIES_PER_RARE,
                "semantic_budget_c": SEMANTIC_BUDGET_C,
                "kmeans_iters": KMEANS_ITERS,
                "rare_offset_range": [RARE_OFFSET_LO, RARE_OFFSET_HI],
                "query_noise_std": QUERY_NOISE_STD,
                "correct_tolerance": CORRECT_TOL,
                "score": "rare-episode-query accuracy; delta = episodic accuracy minus semantic accuracy",
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "verdict": verdict,
            "is_null": is_null,
            "alternative_explanation": (
                "Any gap is a property of a bounded semantic summary meeting rare, surprising values, not of "
                "episodic retrieval as a category. A semantic store whose prototype budget grew with the "
                "number of distinct episodes, or that kept per-episode residuals instead of only prototype "
                "means, would recover the rare specifics and close the gap. When k-means happens to isolate "
                "a rare episode in its own prototype, the summary already recovers it and the unit delta "
                "shrinks toward a tie. Episodic retrieval also pays an unbounded storage cost that this "
                "accuracy score does not charge it for."
            ),
            "failure_domain": (
                "A static key-value memory ecology with dense Gaussian background clusters, isolated rare "
                "episodes carrying large value offsets, tiny-noise probes, and Euclidean key distance. It "
                "does not speak to learned or nonstationary key representations, interference between nearby "
                "episodes, adaptive or prioritized summary budgets, or regimes where rare episodes are "
                "consistent with the background and thus carry no surprising value for a summary to lose."
            ),
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "n_units": sign_flip["n_units"],
        },
    )
