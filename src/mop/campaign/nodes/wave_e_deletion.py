"""Wave E deletion: does deleting invalidated entries reduce stale-memory harm on later queries?

An agent's episodic store accumulates facts, some of which are later revised or invalidated. The stability
temptation is to keep everything (append-only) and let retrieval sort it out. The honest question this bed
asks is whether a memory that actively DELETES an invalidated entry and tracks provenance (which entry is
the current one for a key) suffers less harm on post-invalidation queries than an append-only memory that
keeps the stale entry alongside the fresh one.

Design. Each of N independent streams is a content-addressed key-value store over many distinct facts. Every
key is first written with an original value at a slightly noisy key embedding. A subset of keys is then
invalidated: a fresh observation of the same key arrives with a new value at a fresh noisy embedding. Two
memories process the identical write-then-invalidate stream and answer the identical post-invalidation query
stream (queries land only on invalidated keys, where staleness can bite):

* candidate: a deleting, provenance-tracking store. On invalidation it removes the key's old entry and keeps
  only the fresh (key, embedding, value). Exactly one entry per key survives.
* named control: an append-only store. On invalidation it appends the fresh entry but retains the stale one,
  so an invalidated key has two near-identical embeddings, one carrying the obsolete value.

Retrieval is content addressing: return the value of the nearest stored entry to the noisy query embedding.
Because the append-only store keeps both the stale and fresh entries at nearly the same location, query noise
makes it retrieve the obsolete value on roughly half of invalidated-key queries; the deleting store has only
the fresh entry to return. Per-unit harm is the mean squared error between the retrieved value and the true
current value (lower is better). The per-unit delta is control_error minus candidate_error, so a positive
delta favors the deleting candidate.

The exact one-sided sign-flip over the independent-unit deltas and the neutral SESOI verdict decide the
outcome. A tie or a wrong-direction result is a legitimate null and is reported as such: this bed does not
tune toward a positive.

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
N_KEYS = 40  # distinct facts per stream (the invalidated key plus content-addressing distractors)
N_INVALID = 20  # keys that get a later, value-changing revision
EMBED_DIM = 16
OBS_NOISE = 0.05  # jitter on a stored key embedding at write time
QUERY_NOISE = 0.05  # jitter on the query embedding at read time
N_Q_PER_KEY = 12  # post-invalidation queries per invalidated key
SESOI = 0.05  # smallest squared-value harm reduction we would call a real effect
ALPHA = 0.05


def _unit_embeddings(gen: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """n random unit-norm key embeddings; distinct keys sit far apart, so distractors rarely win a query."""

    e = gen.standard_normal((n, dim))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _query_result(
    q_emb: np.ndarray, mem_emb: np.ndarray, mem_val: np.ndarray, true_vals: np.ndarray
) -> tuple[float, float]:
    """Nearest-entry content addressing. Returns mean squared retrieval error and the stale-hit fraction."""

    d2 = ((q_emb[:, None, :] - mem_emb[None, :, :]) ** 2).sum(axis=2)
    nn = np.argmin(d2, axis=1)
    retrieved = mem_val[nn]
    err = float(np.mean((retrieved - true_vals) ** 2))
    stale = float(np.mean(np.abs(retrieved - true_vals) > 1e-9))
    return err, stale


def _run_unit(seed: int, u: int) -> dict[str, Any]:
    base = _unit_embeddings(rng(seed, "unit", u, "base"), N_KEYS, EMBED_DIM)

    g_val = rng(seed, "unit", u, "values")
    v_old = g_val.standard_normal(N_KEYS)
    v_new = g_val.standard_normal(N_KEYS)

    inval_idx = rng(seed, "unit", u, "invalidate").choice(N_KEYS, size=N_INVALID, replace=False)
    inval_mask = np.zeros(N_KEYS, dtype=bool)
    inval_mask[inval_idx] = True

    emb_write = base + OBS_NOISE * rng(seed, "unit", u, "write").standard_normal((N_KEYS, EMBED_DIM))
    emb_fresh = base + OBS_NOISE * rng(seed, "unit", u, "revise").standard_normal((N_KEYS, EMBED_DIM))

    # Candidate: delete the stale entry, keep exactly one current entry per key.
    cand_emb = emb_write.copy()
    cand_val = v_old.copy()
    cand_emb[inval_mask] = emb_fresh[inval_mask]
    cand_val[inval_mask] = v_new[inval_mask]

    # Control: append-only, so invalidated keys retain a stale entry beside the fresh one.
    ctrl_emb = np.concatenate([emb_write, emb_fresh[inval_mask]], axis=0)
    ctrl_val = np.concatenate([v_old, v_new[inval_mask]], axis=0)

    q_keys = np.repeat(inval_idx, N_Q_PER_KEY)
    q_emb = base[q_keys] + QUERY_NOISE * rng(seed, "unit", u, "query").standard_normal(
        (q_keys.size, EMBED_DIM)
    )
    true_vals = v_new[q_keys]

    cand_err, cand_stale = _query_result(q_emb, cand_emb, cand_val, true_vals)
    ctrl_err, ctrl_stale = _query_result(q_emb, ctrl_emb, ctrl_val, true_vals)
    delta = ctrl_err - cand_err  # positive favors the deleting candidate (lower stale-memory harm)

    return {
        "unit_id": f"u{u:02d}",
        "n_queries": int(q_keys.size),
        "candidate_error": round(cand_err, 9),
        "control_error": round(ctrl_err, 9),
        "candidate_stale_fraction": round(cand_stale, 9),
        "control_stale_fraction": round(ctrl_stale, 9),
        "delta": round(delta, 9),
    }


@register_runner("wave_e.deletion_and_stale_memory")
def wave_e_deletion_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Deterministic Wave E bed: a deleting, provenance-tracking store versus an append-only store on
    post-invalidation retrieval harm."""

    units = [_run_unit(ctx.seed, u) for u in range(N_UNITS)]
    deltas = [u["delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    mean_delta = sign_flip["mean_delta"]
    verdict = verdict_from(mean_delta, sign_flip["one_sided_p"], SESOI, ALPHA)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_e_deletion/v1",
        {
            "form_family": "memory_episode",
            "phenomenon": "memory_revision_deletion",
            "mechanism_family": "provenance_state",
            "unit_class": "content_addressed_kv_memory_stream",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "candidate": "deleting_provenance_tracking_store",
            "control": "append_only_store_keeps_stale",
            "control_description": (
                "The named control is an append-only key-value store that processes the identical "
                "write-then-invalidate stream and answers the identical post-invalidation queries, but on "
                "invalidation it appends the fresh entry while retaining the stale one. The only difference "
                "from the candidate is deletion plus provenance: the candidate removes the old entry so "
                "exactly one current entry per key survives. Both memories use the same nearest-entry "
                "content-addressed retrieval, so any gap is attributable to keeping stale state, not to a "
                "different reader."
            ),
            "design": {
                "n_units": N_UNITS,
                "n_keys": N_KEYS,
                "n_invalidated": N_INVALID,
                "embed_dim": EMBED_DIM,
                "obs_noise": OBS_NOISE,
                "query_noise": QUERY_NOISE,
                "queries_per_key": N_Q_PER_KEY,
                "score": (
                    "per-unit harm is mean squared error between the retrieved value and the true current "
                    "value on invalidated-key queries; delta = control_error minus candidate_error"
                ),
            },
            "units": units,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "mean_delta": mean_delta,
            "one_sided_p": sign_flip["one_sided_p"],
            "verdict": verdict,
            "is_null": is_null,
            "alternative_explanation": (
                "The gap is specific to a purely content-addressed store with no way to tell stale from "
                "fresh. A lighter form of provenance would close much of it without deletion: an "
                "append-only store that recorded a write timestamp and broke retrieval ties toward the "
                "most recent entry would recover the current value on the same queries. So the effect is a "
                "property of state and provenance tracking, not of physical deletion in itself. It also "
                "depends on query and write noise being small enough that the stale and fresh embeddings "
                "for a key stay nearly coincident, which is what makes the append-only reader ambiguous."
            ),
            "failure_domain": (
                "Content-addressed key-value memory over scalar values with unit-norm embeddings, nearest "
                "entry retrieval, and abrupt whole-value invalidation. It does not speak to memories with "
                "semantic or relational structure, partial or gradual revision, retrieval that already uses "
                "recency or provenance metadata, or settings where retaining superseded values carries "
                "value (audit trails, provenance queries, or reversible edits)."
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
