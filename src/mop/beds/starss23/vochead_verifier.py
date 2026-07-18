"""STARSS23 value-of-computation HEADROOM instrument, component 4: the independent stdlib-only verifier.

This module imports NOTHING from ``mop`` and nothing from any headroom producer module. It re-derives the
entire instrument from the sealed artifact alone: it re-implements the canonical JSON seal, the coasting,
both per-frame metrics, all four policies, the rate-matched-random draw discipline, the clip-macro
aggregation, the refreshable range, and the interpretation classification, in pure Python, then checks that
every sealed number and label reproduces and that the honesty flags are what the instrument requires.

It sets ``independent_scientific_confirmation`` to false unconditionally: this is a descriptive
corpus-characterization instrument, and no single run of a producer plus a verifier authored in the same
session can confer scientific confirmation. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from typing import Any

VOCHEAD_VERIFIER_SCHEMA = "mop-starss23-vochead-verifier/v1"

_TIE_EPS = 1e-9
_NUM_TOL = 1e-9

_SUPPORTED_METRICS = ("count_abs", "doa_greatcircle")
_INTERP_WHAT_FLOOR_COLLAPSE = "what_floor_collapse"
_INTERP_REAL_HEADROOM = "real_headroom"
_INTERP_NO_HEADROOM = "no_headroom_budget_saturated"
_ALLOWED_INTERPRETATIONS = (_INTERP_WHAT_FLOOR_COLLAPSE, _INTERP_REAL_HEADROOM, _INTERP_NO_HEADROOM)
_FORBIDDEN_VERBS = ("proves", "demonstrates", "significant", "confirms", "establishes")


# ---------------------------------------------------------------------------
# Independent canonical seal (re-implemented, imports nothing from mop).
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# Independent metrics, coasting, policies (re-derived from scratch).
# ---------------------------------------------------------------------------


def _great_circle_degrees(a: Any, b: Any) -> float:
    a1, e1 = math.radians(float(a[0])), math.radians(float(a[1]))
    a2, e2 = math.radians(float(b[0])), math.radians(float(b[1]))
    v1 = (math.cos(e1) * math.cos(a1), math.cos(e1) * math.sin(a1), math.sin(e1))
    v2 = (math.cos(e2) * math.cos(a2), math.cos(e2) * math.sin(a2), math.sin(e2))
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))
    return math.degrees(math.acos(dot))


def _frame_error(metric_id: str, emitted: Any, truth: Any) -> float:
    if metric_id == "count_abs":
        return float(abs(int(emitted) - int(truth)))
    return _great_circle_degrees(emitted, truth)


def _coast(target: dict[str, Any], reestimate_frames: list[int]) -> list[Any]:
    n_frames = target["n_frames"]
    rset = set(reestimate_frames)
    emitted: list[Any] = []
    current = target["cold_start"]
    est = target["est_values"]
    for t in range(n_frames):
        if t in rset:
            current = est[t]
        emitted.append(current)
    return emitted


def _clip_error(target: dict[str, Any], reestimate_frames: list[int]) -> float:
    emitted = _coast(target, reestimate_frames)
    metric_id = target["metric_id"]
    gt = target["gt_values"]
    active = target["active_mask"]
    total = 0.0
    count = 0
    for t in range(target["n_frames"]):
        if active[t]:
            total += _frame_error(metric_id, emitted[t], gt[t])
            count += 1
    return total / count


def _budget_k(n_frames: int, fraction: float) -> int:
    k = int(round(float(fraction) * n_frames))
    return max(1, min(n_frames, k))


def _greedy_informed_path(target: dict[str, Any], k_max: int) -> list[float]:
    """Independent re-derivation of the greedy label-aware reference: errors[k] for k = 0 .. usable."""

    n_frames = target["n_frames"]
    gt = target["gt_values"]
    est = target["est_values"]
    active = target["active_mask"]
    metric_id = target["metric_id"]
    n_active = sum(1 for flag in active if flag)

    emitted: list[Any] = [target["cold_start"]] * n_frames
    total = math.fsum(_frame_error(metric_id, emitted[t], gt[t]) for t in range(n_frames) if active[t])
    path = [total / n_active]

    refresh: list[int] = []
    remaining = list(target["change_frames"])
    while len(refresh) < k_max and remaining:
        best_c = None
        best_delta = -_TIE_EPS
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


def _rmr_seed(base_seed: int, clip_id: str, k: int, draw: int) -> int:
    payload = f"{base_seed}|{clip_id}|{k}|{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rmr_frames(n_frames: int, k: int, seed: int) -> list[int]:
    if k == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_frames), min(k, n_frames)))


def _rmr_mean_error(target: dict[str, Any], k: int, base_seed: int, n_draws: int) -> float:
    total = 0.0
    for draw in range(n_draws):
        frames = _rmr_frames(target["n_frames"], k, _rmr_seed(base_seed, target["clip_id"], k, draw))
        total += _clip_error(target, frames)
    return total / n_draws


def _macro_mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _reanalyze(
    targets: list[dict[str, Any]], budgets: list[float], base_seed: int, n_draws: int
) -> dict[str, Any]:
    always = _macro_mean([_clip_error(t, list(range(t["n_frames"]))) for t in targets])
    never = _macro_mean([_clip_error(t, []) for t in targets])
    refreshable_range = never - always
    per_clip_k = {t["clip_id"]: [_budget_k(t["n_frames"], f) for f in budgets] for t in targets}
    greedy_paths = {t["clip_id"]: _greedy_informed_path(t, max(per_clip_k[t["clip_id"]])) for t in targets}
    points = []
    for fraction in budgets:
        ks = [_budget_k(t["n_frames"], fraction) for t in targets]
        informed = _macro_mean(
            [
                greedy_paths[t["clip_id"]][min(k, len(greedy_paths[t["clip_id"]]) - 1)]
                for t, k in zip(targets, ks, strict=True)
            ]
        )
        rmr = _macro_mean(
            [_rmr_mean_error(t, k, base_seed, n_draws) for t, k in zip(targets, ks, strict=True)]
        )
        points.append({"fraction": fraction, "informed": informed, "rmr": rmr, "headroom": rmr - informed})
    if refreshable_range <= _TIE_EPS:
        interp = _INTERP_WHAT_FLOOR_COLLAPSE
    elif all(p["headroom"] > _TIE_EPS for p in points):
        interp = _INTERP_REAL_HEADROOM
    else:
        interp = _INTERP_NO_HEADROOM
    return {"always": always, "never": never, "range": refreshable_range, "points": points, "interp": interp}


# ---------------------------------------------------------------------------
# The verification entry point.
# ---------------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= _NUM_TOL


def verify_vochead_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Independently re-derive and check the sealed headroom artifact. Returns a verification receipt."""

    mismatches: list[str] = []

    # 1. Seal.
    content = {k: v for k, v in artifact.items() if k != "seal"}
    recomputed_seal = _canonical_sha256(content)
    seal_intact = recomputed_seal == artifact.get("seal", {}).get("sha256")
    if not seal_intact:
        mismatches.append("seal sha256 does not reproduce")

    corpus = artifact["corpus_targets"]
    analysis = artifact["analysis"]

    def targets_for(family: str, clip_ids: list[str]) -> list[dict[str, Any]]:
        family_key = "synthetic_control" if family in ("strong_what", "harmful_what") else family
        return [corpus[family_key][cid] for cid in clip_ids if cid in corpus[family_key]]

    # 2. Re-derive each real scope and family.
    targets_reproduced = True
    interpretation_reproduced = True
    checks = [
        ("test_fold", "count"),
        ("test_fold", "doa"),
        ("full_subset", "count"),
        ("full_subset", "doa"),
    ]
    for scope, family in checks:
        sealed = analysis[scope][family]
        subset = targets_for(family, analysis[scope]["clip_ids"])
        budgets = [b["fraction"] for b in sealed["budgets"]]
        redone = _reanalyze(
            subset, budgets, sealed["rmr_discipline"]["base_seed"], sealed["rmr_discipline"]["n_draws"]
        )
        for name, got, want in (
            ("always_on_macro", redone["always"], sealed["always_on_macro"]),
            ("never_update_macro", redone["never"], sealed["never_update_macro"]),
            ("refreshable_range", redone["range"], sealed["refreshable_range"]),
        ):
            if not _close(got, want):
                targets_reproduced = False
                mismatches.append(f"{scope}/{family} {name}: {got} != {want}")
        for point, sp in zip(redone["points"], sealed["budgets"], strict=True):
            if not _close(point["informed"], sp["informed_change_aligned_macro"]):
                targets_reproduced = False
                mismatches.append(f"{scope}/{family} informed@{sp['fraction']}")
            if not _close(point["rmr"], sp["rate_matched_random_macro"]):
                targets_reproduced = False
                mismatches.append(f"{scope}/{family} rmr@{sp['fraction']}")
            if not _close(point["headroom"], sp["headroom_rmr_minus_informed"]):
                targets_reproduced = False
                mismatches.append(f"{scope}/{family} headroom@{sp['fraction']}")
        if redone["interp"] != sealed["interpretation"]:
            interpretation_reproduced = False
            mismatches.append(
                f"{scope}/{family} interpretation {redone['interp']} != {sealed['interpretation']}"
            )

    # 3. Synthetic controls: the instrument must report the two known shapes.
    control = analysis["synthetic_control"]
    control_expectations = (
        ("strong_what", _INTERP_REAL_HEADROOM),
        ("harmful_what", _INTERP_WHAT_FLOOR_COLLAPSE),
    )
    for family, expected in control_expectations:
        sealed = control[family]
        subset = targets_for(family, [sealed["label"]])
        budgets = [b["fraction"] for b in sealed["budgets"]]
        redone = _reanalyze(
            subset, budgets, sealed["rmr_discipline"]["base_seed"], sealed["rmr_discipline"]["n_draws"]
        )
        if redone["interp"] != sealed["interpretation"] or sealed["interpretation"] != expected:
            interpretation_reproduced = False
            mismatches.append(f"synthetic_control/{family} interpretation")

    # 4. Honesty.
    flags = artifact.get("flags", {})
    honesty_ok = (
        flags.get("activation_allowed") is False
        and flags.get("scientific_promotion") is False
        and flags.get("independent_scientific_confirmation") is False
        and artifact.get("synthetic_control_ok") is True
        and artifact.get("source_kind") == "real"
        and artifact.get("rights_clean") is True
    )
    for scope, family in checks:
        if analysis[scope][family]["interpretation"] not in _ALLOWED_INTERPRETATIONS:
            honesty_ok = False
            mismatches.append(f"{scope}/{family} interpretation not in allowed set")
    blob = json.dumps(artifact).lower()
    for verb in _FORBIDDEN_VERBS:
        if verb in blob:
            honesty_ok = False
            mismatches.append(f"forbidden claim verb present: {verb}")
    if not honesty_ok and "honesty" not in " ".join(mismatches):
        mismatches.append("honesty flags not as required")

    independent_referee_reproduction = seal_intact and targets_reproduced and interpretation_reproduced
    return {
        "schema": VOCHEAD_VERIFIER_SCHEMA,
        "seal_intact": seal_intact,
        "targets_reproduced": targets_reproduced,
        "interpretation_reproduced": interpretation_reproduced,
        "honesty_ok": honesty_ok,
        "independent_referee_reproduction": independent_referee_reproduction,
        "independent_scientific_confirmation": False,
        "recomputed_seal": recomputed_seal,
        "mismatches": mismatches,
    }


_DEFAULT_VERIFICATION_PATH = "proof/STARSS23_VOC_HEADROOM.verification.json"


def write_vochead_verification(
    verification: dict[str, Any], out_path: str = _DEFAULT_VERIFICATION_PATH
) -> str:
    """Write the verification receipt as canonical JSON bytes so its on-disk digest is stable."""

    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(verification))
    return str(path)
