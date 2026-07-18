"""STARSS23 value-of-computation HEADROOM instrument, component 1: the target-agnostic headroom math.

This is a net-new, additive instrument. It edits no sealed onset, counting, or direction-of-arrival
module and touches no live campaign path. It sits beside the three sealed STARSS23 value-of-computation
beds (onset-localization, source-counting, direction-of-arrival) and asks a question none of them
measured directly: for each re-estimation target, how much value-of-computation HEADROOM does the corpus
actually contain, before any trained gate is involved?

A value-of-computation bed pits a trained WHEN gate against a rate-matched-random control that spends the
identical re-estimation budget at random. Three such beds have run to a conclusion and all nulled. This
instrument decomposes WHY, per target, with two label-only reference policies that bracket what ANY WHEN
gate could possibly do at a matched budget:

* ``always_on`` re-estimates every frame (the perfect-WHEN, unlimited-budget ceiling: the coasted error
  equals the frozen estimator's own fresh error).
* ``never_update`` never re-estimates (coast the fixed cold-start forever: the zero-budget floor).
* ``informed_change_aligned`` is a label-AWARE reference: starting from never_update it greedily adds the
  change frame whose re-estimation most reduces the total coasted error, up to budget K. It is a strong
  achievable policy (a WHEN ceiling reference over the change-frame candidates), not a proven global
  optimum, and it is stated as such.
* ``rate_matched_random`` places the same K re-estimations at random, averaged over a preregistered set of
  deterministic stdlib draws (the exact control the sealed beds use).

Two derived quantities carry the finding:

* the ``refreshable_range`` = ``never_update - always_on``. If this is NEGATIVE, re-estimating the frozen
  estimator is WORSE than coasting a constant: the WHAT floor has collapsed and no WHEN policy can help.
  This is a distinct, mechanistic failure shape from "the gate could not learn the WHEN signal".
* the per-budget ``headroom`` = ``rate_matched_random - informed_change_aligned`` (positive means the
  label-aware policy beats random at that matched budget). If this is positive across the swept budgets,
  the corpus genuinely rewards WHEN placement and a gate that could locate changes would win; if it is a
  tie, random already saturates the rare changes at that budget and there is nothing for a gate to win.

Nothing here reads a test-split score to tune anything: the policies are fixed functions of the labels and
the frozen estimator output. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

VOCHEAD_ANALYZER_SCHEMA = "mop-starss23-vochead-analyzer/v1"

# The two target metric families this instrument understands. Both reduce an emitted-versus-ground-truth
# comparison to a nonnegative per-active-frame error whose clip mean is the clip score (lower is better).
METRIC_COUNT_ABS = "count_abs"  # concurrent-source count: absolute integer error per frame
METRIC_DOA_GREATCIRCLE = "doa_greatcircle"  # direction of arrival: great-circle degrees per active frame
SUPPORTED_METRICS = (METRIC_COUNT_ABS, METRIC_DOA_GREATCIRCLE)

# Preregistered budget sweep, as fractions of a clip's frame count, spanning the tight regime (budget below
# the change density, where placement is decisive) through the loose regime the sealed beds operated in.
BUDGET_FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)

# Rate-matched-random discipline: a fixed number of deterministic draws with host-reproducible stdlib
# seeding, so the control is byte-reproducible and the independent verifier can re-derive it exactly.
N_RMR_DRAWS = 32
RMR_BASE_SEED = 20260718

# A strictly-positive margin below which an aggregate difference is a tie, hence a null.
TIE_EPS = 1e-9


class VocHeadRefusal(ValueError):
    """Raised when a target, a budget, or a policy input violates the headroom instrument contract."""


# ---------------------------------------------------------------------------
# Per-frame error metrics.
# ---------------------------------------------------------------------------


def great_circle_degrees(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angular distance in degrees between two (azimuth, elevation) directions.

    Owned locally (stdlib math only) so the instrument and its stdlib verifier share one definition and
    neither imports a bed's private geometry helper.
    """

    a1, e1, a2, e2 = (math.radians(v) for v in (az1, el1, az2, el2))
    v1 = (math.cos(e1) * math.cos(a1), math.cos(e1) * math.sin(a1), math.sin(e1))
    v2 = (math.cos(e2) * math.cos(a2), math.cos(e2) * math.sin(a2), math.sin(e2))
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))
    return math.degrees(math.acos(dot))


def _frame_error(metric_id: str, emitted: Any, truth: Any) -> float:
    if metric_id == METRIC_COUNT_ABS:
        return float(abs(int(emitted) - int(truth)))
    if metric_id == METRIC_DOA_GREATCIRCLE:
        return great_circle_degrees(float(emitted[0]), float(emitted[1]), float(truth[0]), float(truth[1]))
    raise VocHeadRefusal(f"unsupported metric_id {metric_id!r}")


# ---------------------------------------------------------------------------
# The per-clip target: ground truth, frozen estimator output, active mask, and label change frames.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipTarget:
    """One clip reduced to what the headroom math needs. Values are ints (count) or (az, el) pairs (doa)."""

    clip_id: str
    room_id: str
    n_frames: int
    metric_id: str
    active_mask: tuple[bool, ...]
    gt_values: tuple[Any, ...]
    est_values: tuple[Any, ...]
    change_frames: tuple[int, ...]
    cold_start: Any

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise VocHeadRefusal("ClipTarget.clip_id must be a nonempty string")
        if self.metric_id not in SUPPORTED_METRICS:
            raise VocHeadRefusal(f"ClipTarget.metric_id must be one of {SUPPORTED_METRICS}")
        if not isinstance(self.n_frames, int) or isinstance(self.n_frames, bool) or self.n_frames <= 0:
            raise VocHeadRefusal("ClipTarget.n_frames must be a positive integer")
        if not (len(self.active_mask) == len(self.gt_values) == len(self.est_values) == self.n_frames):
            raise VocHeadRefusal("active_mask, gt_values, est_values must each have length n_frames")
        if not any(self.active_mask):
            raise VocHeadRefusal("a ClipTarget with no active frame cannot be scored")
        previous = -1
        for frame in self.change_frames:
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < self.n_frames:
                raise VocHeadRefusal(f"change_frames must lie in [0, {self.n_frames})")
            if frame <= previous:
                raise VocHeadRefusal("change_frames must be strictly sorted and unique")
            previous = frame

    @property
    def n_active_frames(self) -> int:
        return sum(1 for flag in self.active_mask if flag)

    def _canonical_value(self, value: Any) -> Any:
        if self.metric_id == METRIC_COUNT_ABS:
            return int(value)
        return [float(value[0]), float(value[1])]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": VOCHEAD_ANALYZER_SCHEMA,
            "clip_id": self.clip_id,
            "room_id": self.room_id,
            "n_frames": self.n_frames,
            "metric_id": self.metric_id,
            "active_mask": [bool(flag) for flag in self.active_mask],
            "gt_values": [self._canonical_value(v) for v in self.gt_values],
            "est_values": [self._canonical_value(v) for v in self.est_values],
            "change_frames": list(self.change_frames),
            "cold_start": self._canonical_value(self.cold_start),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Coasting and clip scoring, shared by every policy.
# ---------------------------------------------------------------------------


def coast(target: ClipTarget, reestimate_frames: Sequence[int]) -> list[Any]:
    """Return the causal coasted emitted track: hold the last re-estimate, else the cold start."""

    n_frames = target.n_frames
    reestimate_set = set()
    previous = -1
    for frame in reestimate_frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < n_frames:
            raise VocHeadRefusal(f"reestimate_frames must lie in [0, {n_frames})")
        if frame <= previous:
            raise VocHeadRefusal("reestimate_frames must be strictly sorted and unique")
        previous = frame
        reestimate_set.add(frame)
    emitted: list[Any] = []
    current = target.cold_start
    for t in range(n_frames):
        if t in reestimate_set:
            current = target.est_values[t]
        emitted.append(current)
    return emitted


def clip_error(target: ClipTarget, reestimate_frames: Sequence[int]) -> float:
    """Mean per-active-frame error of the coasted emitted track. Lower is better."""

    emitted = coast(target, reestimate_frames)
    total = 0.0
    count = 0
    for t in range(target.n_frames):
        if target.active_mask[t]:
            total += _frame_error(target.metric_id, emitted[t], target.gt_values[t])
            count += 1
    if count == 0:
        raise VocHeadRefusal("a clip with no active frame cannot be scored")
    return total / count


# ---------------------------------------------------------------------------
# Budget and the four policies.
# ---------------------------------------------------------------------------


def budget_k(n_frames: int, fraction: float) -> int:
    """The integer re-estimation budget for a clip at a budget fraction: round, clamped to [1, n_frames]."""

    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise VocHeadRefusal("fraction must be a real number")
    if not 0.0 < float(fraction) <= 1.0:
        raise VocHeadRefusal("fraction must lie in (0, 1]")
    k = int(round(float(fraction) * n_frames))
    return max(1, min(n_frames, k))


def greedy_informed_path(target: ClipTarget, k_max: int) -> list[float]:
    """Return ``errors[k]`` for ``k = 0 .. min(k_max, usable)``: the clip mean error of the greedy label-

    aware policy that, starting from never_update, repeatedly adds the change frame whose re-estimation most
    reduces the total coasted error, stopping when no candidate strictly reduces it. This is a strong
    achievable WHEN reference (an upper reference over the change-frame candidates), NOT a proven global
    optimum, and is stated as such. ``errors[0]`` is the never_update mean error.
    """

    if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 0:
        raise VocHeadRefusal("k_max must be a nonnegative integer")
    n_frames = target.n_frames
    gt = target.gt_values
    est = target.est_values
    active = target.active_mask
    metric_id = target.metric_id
    n_active = target.n_active_frames

    emitted: list[Any] = [target.cold_start] * n_frames
    total = math.fsum(_frame_error(metric_id, emitted[t], gt[t]) for t in range(n_frames) if active[t])
    path = [total / n_active]

    refresh: list[int] = []
    remaining = list(target.change_frames)
    while len(refresh) < k_max and remaining:
        best_c: int | None = None
        best_delta = -TIE_EPS  # require a strict improvement
        best_region = (0, 0)
        for c in remaining:
            idx = bisect.bisect_right(refresh, c)
            nxt = refresh[idx] if idx < len(refresh) else n_frames
            new_value = est[c]
            delta = 0.0
            for t in range(c, nxt):
                if active[t]:
                    delta += _frame_error(metric_id, new_value, gt[t])
                    delta -= _frame_error(metric_id, emitted[t], gt[t])
            if delta < best_delta:
                best_delta, best_c, best_region = delta, c, (c, nxt)
        if best_c is None:
            break
        start, stop = best_region
        new_value = est[best_c]
        for t in range(start, stop):
            emitted[t] = new_value
        bisect.insort(refresh, best_c)
        remaining.remove(best_c)
        total += best_delta
        path.append(total / n_active)
    return path


def greedy_informed_error(target: ClipTarget, k: int) -> float:
    """The greedy informed reference's clip mean error at budget K (see :func:`greedy_informed_path`)."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise VocHeadRefusal("k must be a nonnegative integer")
    path = greedy_informed_path(target, k)
    return path[min(k, len(path) - 1)]


def rmr_seed(base_seed: int, clip_id: str, k: int, draw: int) -> int:
    """A host-reproducible stdlib seed for one rate-matched-random draw, derived by SHA-256."""

    payload = f"{base_seed}|{clip_id}|{k}|{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rate_matched_random_frames(n_frames: int, k: int, seed: int) -> list[int]:
    """One deterministic rate-matched-random re-estimation set of size ``min(k, n_frames)``, sorted."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise VocHeadRefusal("k must be a nonnegative integer")
    if k == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_frames), min(k, n_frames)))


def rmr_mean_error(target: ClipTarget, k: int, base_seed: int, n_draws: int) -> float:
    """Mean clip error over ``n_draws`` deterministic rate-matched-random re-estimation sets at budget K."""

    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws <= 0:
        raise VocHeadRefusal("n_draws must be a positive integer")
    total = 0.0
    for draw in range(n_draws):
        frames = rate_matched_random_frames(target.n_frames, k, rmr_seed(base_seed, target.clip_id, k, draw))
        total += clip_error(target, frames)
    return total / n_draws


# ---------------------------------------------------------------------------
# The per-target sweep and interpretation.
# ---------------------------------------------------------------------------


INTERP_WHAT_FLOOR_COLLAPSE = "what_floor_collapse"
INTERP_REAL_HEADROOM = "real_headroom"
INTERP_NO_HEADROOM_BUDGET_SATURATED = "no_headroom_budget_saturated"


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """Clip-macro aggregates at one budget fraction across all clips of a target."""

    fraction: float
    mean_k: float
    informed_macro: float
    rmr_macro: float
    headroom: float
    realization: float | None

    def payload(self) -> dict[str, Any]:
        return {
            "fraction": round(self.fraction, 12),
            "mean_k": round(self.mean_k, 6),
            "informed_change_aligned_macro": round(self.informed_macro, 12),
            "rate_matched_random_macro": round(self.rmr_macro, 12),
            "headroom_rmr_minus_informed": round(self.headroom, 12),
            "realization_fraction": None if self.realization is None else round(self.realization, 12),
        }


def _macro_mean(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        raise VocHeadRefusal("cannot take a clip-macro mean over zero clips")
    return math.fsum(values) / len(values)


def analyze_target(
    label: str,
    metric_id: str,
    targets: Sequence[ClipTarget],
    *,
    budgets: Sequence[float] = BUDGET_FRACTIONS,
    base_seed: int = RMR_BASE_SEED,
    n_draws: int = N_RMR_DRAWS,
) -> dict[str, Any]:
    """Run the full budget sweep for one target and classify its value-of-computation headroom shape."""

    targets = list(targets)
    if not targets:
        raise VocHeadRefusal("analyze_target needs at least one clip target")
    for target in targets:
        if target.metric_id != metric_id:
            raise VocHeadRefusal(f"clip {target.clip_id} metric {target.metric_id} != target {metric_id}")

    always_macro = _macro_mean([clip_error(t, tuple(range(t.n_frames))) for t in targets])
    never_macro = _macro_mean([clip_error(t, ()) for t in targets])
    refreshable_range = never_macro - always_macro

    total_active = sum(t.n_active_frames for t in targets)
    total_changes = sum(len(t.change_frames) for t in targets)
    change_density = total_changes / total_active if total_active else 0.0

    # One greedy pass per clip to the largest budget needed, snapshotted so every budget reuses it.
    per_clip_k = {t.clip_id: [budget_k(t.n_frames, f) for f in budgets] for t in targets}
    greedy_paths = {t.clip_id: greedy_informed_path(t, max(per_clip_k[t.clip_id])) for t in targets}

    points: list[BudgetPoint] = []
    for fraction in budgets:
        ks = [budget_k(t.n_frames, fraction) for t in targets]
        informed = [
            greedy_paths[t.clip_id][min(k, len(greedy_paths[t.clip_id]) - 1)]
            for t, k in zip(targets, ks, strict=True)
        ]
        rmr = [rmr_mean_error(t, k, base_seed, n_draws) for t, k in zip(targets, ks, strict=True)]
        informed_macro = _macro_mean(informed)
        rmr_macro = _macro_mean(rmr)
        headroom = rmr_macro - informed_macro
        realization: float | None
        if refreshable_range > TIE_EPS:
            realization = (never_macro - informed_macro) / refreshable_range
        else:
            realization = None
        points.append(
            BudgetPoint(
                float(fraction), math.fsum(ks) / len(ks), informed_macro, rmr_macro, headroom, realization
            )
        )

    if refreshable_range <= TIE_EPS:
        interpretation = INTERP_WHAT_FLOOR_COLLAPSE
    elif all(point.headroom > TIE_EPS for point in points):
        interpretation = INTERP_REAL_HEADROOM
    else:
        interpretation = INTERP_NO_HEADROOM_BUDGET_SATURATED

    return {
        "schema": VOCHEAD_ANALYZER_SCHEMA,
        "label": label,
        "metric_id": metric_id,
        "n_clips": len(targets),
        "n_rooms": len({t.room_id for t in targets}),
        "total_active_frames": total_active,
        "total_changes": total_changes,
        "change_density": round(change_density, 12),
        "always_on_macro": round(always_macro, 12),
        "never_update_macro": round(never_macro, 12),
        "refreshable_range": round(refreshable_range, 12),
        "budgets": [point.payload() for point in points],
        "rmr_discipline": {
            "base_seed": base_seed,
            "n_draws": n_draws,
            "seed_rule": "sha256(base|clip|k|draw)[:8]",
        },
        "interpretation": interpretation,
        "interpretation_rule": (
            "refreshable_range <= 0 is what_floor_collapse (re-estimating the frozen estimator is worse "
            "than coasting a constant, so no WHEN policy can help); a positive range with the informed "
            "change-aligned reference strictly beating rate-matched-random at every swept budget is "
            "real_headroom (a WHEN gate that could locate changes would win); a positive range where the "
            "informed reference ties random at some budget is no_headroom_budget_saturated (random already "
            "catches the rare changes at that budget). A tie is a null."
        ),
    }
