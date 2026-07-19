
from __future__ import annotations

import hashlib
import itertools
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-doa-bed-verification/v1"

EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-doa-bed/v1"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"

ARCH_A_ID = "arch_a_264_12_1"
ARCH_B_ID = "arch_b_264_6_6_1"
ARCHITECTURES = (ARCH_A_ID, ARCH_B_ID)

PRIMARY_ALPHA = 0.05
ROOM_MAJORITY_ALPHA = 0.10
MIN_REPRODUCTIONS = 3
_MAX_CLIP_ENUMERATION = 40

ALLOWED_CLAIM_VERBS = ("consistent with", "suggestive")
FORBIDDEN_CLAIM_VERBS = (
    "demonstrates",
    "demonstrate",
    "significant",
    "proves",
    "prove",
    "establishes",
    "confirms",
    "confirmed",
)

_TOL = 1e-6


class DoaVerificationRefusal(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _agree(a: object, b: object, tol: float = _TOL) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= tol


def _direction_to_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def _great_circle_degrees(az1: float, el1: float, az2: float, el2: float) -> float:
    v1 = _direction_to_vector(az1, el1)
    v2 = _direction_to_vector(az2, el2)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _as_direction_track(track: object, label: str) -> list[list[float] | None]:
    if not isinstance(track, list):
        raise DoaVerificationRefusal(f"{label} must be a list")
    cleaned: list[list[float] | None] = []
    for item in track:
        if item is None:
            cleaned.append(None)
            continue
        if not isinstance(item, list) or len(item) != 2:
            raise DoaVerificationRefusal(f"{label} entries must be null or an [azimuth, elevation] pair")
        az, el = item
        az_ok = not isinstance(az, bool) and isinstance(az, (int, float))
        el_ok = not isinstance(el, bool) and isinstance(el, (int, float))
        if not az_ok or not el_ok:
            raise DoaVerificationRefusal(f"{label} entries must hold real numbers")
        cleaned.append([float(az), float(el)])
    return cleaned


def _as_reestimates(frames: object, n_frames: int, label: str) -> list[int]:
    if not isinstance(frames, list):
        raise DoaVerificationRefusal(f"{label} must be a list of frame indices")
    cleaned: list[int] = []
    last = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise DoaVerificationRefusal(f"{label} indices must lie in [0, {n_frames})")
        if frame <= last:
            raise DoaVerificationRefusal(f"{label} must be strictly sorted and unique")
        last = frame
        cleaned.append(frame)
    return cleaned


def _coast(
    estimator_track: list[list[float]], reestimates: list[int], cold_start: list[float]
) -> list[list[float]]:
    fire = set(reestimates)
    emitted: list[list[float]] = []
    held = list(cold_start)
    for t in range(len(estimator_track)):
        if t in fire:
            held = estimator_track[t]
        emitted.append(held)
    return emitted


def _clip_mae(
    gt_track: list[list[float] | None],
    estimator_track: list[list[float]],
    reestimates: list[int],
    cold_start: list[float],
) -> tuple[float, int]:
    emitted = _coast(estimator_track, reestimates, cold_start)
    errors: list[float] = []
    for t, gt in enumerate(gt_track):
        if gt is None:
            continue
        errors.append(_great_circle_degrees(gt[0], gt[1], emitted[t][0], emitted[t][1]))
    if not errors:
        raise DoaVerificationRefusal("a clip with no active frame cannot be scored")
    return sum(errors) / len(errors), len(errors)


def _clip_pooled(
    gt_track: list[list[float] | None],
    estimator_track: list[list[float]],
    reestimates: list[int],
    cold_start: list[float],
) -> tuple[float, int]:
    emitted = _coast(estimator_track, reestimates, cold_start)
    total = 0.0
    n = 0
    for t, gt in enumerate(gt_track):
        if gt is None:
            continue
        total += _great_circle_degrees(gt[0], gt[1], emitted[t][0], emitted[t][1])
        n += 1
    return total, n


def _reestimates_for_arm(arm: str, clip_id: str, n_frames: int, reestimates_by_clip: dict) -> list[int]:
    if arm == ARM_ALWAYS_ON:
        return list(range(n_frames))
    if arm == ARM_NEVER_UPDATE:
        return []
    stored = reestimates_by_clip.get(clip_id, {})
    if not isinstance(stored, dict):
        raise DoaVerificationRefusal(f"clip {clip_id!r} reestimate_frames must be an object")
    return _as_reestimates(stored.get(arm), n_frames, f"{arm} reestimate_frames on {clip_id}")


def _sign_flip_one_sided_brute_force(deltas: list[float]) -> tuple[float, float, int]:
    n = len(deltas)
    if n == 0:
        raise DoaVerificationRefusal("the sign-flip test needs at least one paired delta")
    observed = sum(deltas) / n
    at_least = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        flipped_mean = sum(s * d for s, d in zip(signs, deltas, strict=True)) / n
        if flipped_mean >= observed - _TOL:
            at_least += 1
    return observed, at_least / total, total


def _clip_sign_flip_meet_in_middle(deltas: list[float]) -> tuple[float, float, int]:
    n = len(deltas)
    if n == 0:
        raise DoaVerificationRefusal("the clip sign-flip needs at least one paired delta")
    if n > _MAX_CLIP_ENUMERATION:
        raise DoaVerificationRefusal(
            f"exact clip enumeration is capped at n={_MAX_CLIP_ENUMERATION}; got n={n}"
        )
    t_observed = math.fsum(deltas)
    half = n // 2
    left, right = deltas[:half], deltas[half:]

    def partial_sums(part: list[float]) -> list[float]:
        sums = [0.0]
        for value in part:
            nxt: list[float] = []
            for acc in sums:
                nxt.append(acc + value)
                nxt.append(acc - value)
            sums = nxt
        return sums

    right_sums = sorted(partial_sums(right))
    threshold_base = t_observed - _TOL
    count = 0
    for left_sum in partial_sums(left):
        cutoff = threshold_base - left_sum
        count += len(right_sums) - bisect_left(right_sums, cutoff)
    permutations = 2**n
    return t_observed, count / permutations, permutations


def _room_majority(clip_deltas_by_room: dict[str, list[float]]) -> tuple[int, float, int, list[dict]]:
    if not clip_deltas_by_room:
        raise DoaVerificationRefusal("room majority needs at least one room")
    n_favorable = 0
    per_room: list[dict] = []
    for room_id in sorted(clip_deltas_by_room):
        deltas = clip_deltas_by_room[room_id]
        if not deltas:
            raise DoaVerificationRefusal(f"room {room_id!r} has no clip deltas")
        n_pos = sum(1 for d in deltas if d > 0.0)
        favorable = n_pos * 2 > len(deltas)
        if favorable:
            n_favorable += 1
        per_room.append(
            {"room_id": room_id, "n_clips": len(deltas), "n_favorable_clips": n_pos, "favorable": favorable}
        )
    n_rooms = len(per_room)
    permutations = 2**n_rooms
    one_sided_p = sum(math.comb(n_rooms, k) for k in range(n_favorable, n_rooms + 1)) / permutations
    return n_favorable, one_sided_p, n_rooms, per_room


@dataclass(frozen=True, slots=True)
class DoaVerificationResult:
    seal_intact: bool
    schema_ok: bool
    scores_reproduced: bool
    stats_reproduced: bool
    honesty_ok: bool
    independent_referee_reproduction: bool
    independent_scientific_confirmation: bool
    source_kind: str
    rights_clean: bool
    reproductions: int
    mismatches: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.independent_referee_reproduction


def _rescore_seed_block(
    seed_block: dict, corpus: dict, cold_start: list[float], mismatches: list[str], architecture: str
) -> dict[str, tuple[float, float]]:

    seed = seed_block.get("seed")
    reestimates_by_clip = seed_block.get("reestimate_frames", {})
    arm_scores_macro = seed_block.get("arm_scores_macro", {})
    arm_scores_pooled = seed_block.get("arm_scores_pooled", {})
    if not isinstance(arm_scores_macro, dict) or not isinstance(arm_scores_pooled, dict):
        raise DoaVerificationRefusal(f"{architecture} seed {seed} is missing arm_scores blocks")

    candidate_counts: dict[str, int] = {}
    out: dict[str, tuple[float, float]] = {}
    for arm in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE):
        clip_maes: list[tuple[str, float, int]] = []
        pooled_total = 0.0
        pooled_n = 0
        for clip_id in sorted(corpus):
            block = corpus[clip_id]
            gt = _as_direction_track(block.get("gt_track"), f"{clip_id} gt_track")
            estimator = _as_direction_track(block.get("estimator_track"), f"{clip_id} estimator_track")
            if any(v is None for v in estimator):
                raise DoaVerificationRefusal(f"clip {clip_id!r} estimator_track must have no null entries")
            n = len(estimator)
            reestimates = _reestimates_for_arm(arm, clip_id, n, reestimates_by_clip)
            if arm == ARM_CANDIDATE:
                candidate_counts[clip_id] = len(reestimates)
            if arm == ARM_RATE_MATCHED_RANDOM and len(reestimates) != candidate_counts.get(clip_id):
                mismatches.append(
                    f"{architecture} seed {seed} clip {clip_id}: rate_matched_random spends "
                    f"{len(reestimates)} re-estimations but candidate spends {candidate_counts.get(clip_id)}"
                )
            mae, n_active = _clip_mae(gt, estimator, reestimates, cold_start)
            clip_maes.append((clip_id, mae, n_active))
            p_sum, p_n = _clip_pooled(gt, estimator, reestimates, cold_start)
            pooled_total += p_sum
            pooled_n += p_n

        recomputed_macro = sum(mae for _cid, mae, _n in clip_maes) / len(clip_maes)
        recomputed_pooled = pooled_total / pooled_n if pooled_n else float("nan")
        out[arm] = (recomputed_macro, recomputed_pooled)

        claimed_macro = arm_scores_macro.get(arm, {})
        if not _agree(claimed_macro.get("macro_mae_deg"), recomputed_macro):
            mismatches.append(
                f"{architecture} seed {seed} arm {arm} macro_mae_deg stored "
                f"{claimed_macro.get('macro_mae_deg')} recomputed {recomputed_macro}"
            )
        claimed_per_clip = claimed_macro.get("per_clip", {})
        for clip_id, mae, n_active in clip_maes:
            entry = claimed_per_clip.get(clip_id, {})
            if not _agree(entry.get("mae_deg"), mae) or entry.get("n_active_frames") != n_active:
                mismatches.append(
                    f"{architecture} seed {seed} arm {arm} clip {clip_id} per-clip score disagrees"
                )

        claimed_pooled = arm_scores_pooled.get(arm, {})
        if not _agree(claimed_pooled.get("pooled_mae_deg"), recomputed_pooled):
            mismatches.append(
                f"{architecture} seed {seed} arm {arm} pooled_mae_deg stored "
                f"{claimed_pooled.get('pooled_mae_deg')} recomputed {recomputed_pooled}"
            )

    return out


def _rescore_architecture(
    artifact: dict, architecture: str, cold_start: list[float], mismatches: list[str]
) -> dict[str, object]:
    corpus = artifact.get("corpus_tracks")
    per_seed = artifact.get("per_seed", {}).get(architecture)
    if not isinstance(per_seed, list) or not per_seed:
        raise DoaVerificationRefusal(f"per_seed[{architecture}] must be a nonempty list")

    per_clip_recomputed: dict[str, list[float]] = {}  # clip_id -> [delta_seed0, delta_seed1, ...]
    seed_level_deltas: list[float] = []
    room_of: dict[str, str] = {clip_id: block.get("room_id", "") for clip_id, block in corpus.items()}

    for seed_block in per_seed:
        recomputed = _rescore_seed_block(seed_block, corpus, cold_start, mismatches, architecture)
        cand_macro, _cand_pooled = recomputed[ARM_CANDIDATE]
        rmr_macro, _rmr_pooled = recomputed[ARM_RATE_MATCHED_RANDOM]
        seed_level_deltas.append(rmr_macro - cand_macro)

        claimed_per_clip_cand = seed_block["arm_scores_macro"][ARM_CANDIDATE]["per_clip"]
        claimed_per_clip_rmr = seed_block["arm_scores_macro"][ARM_RATE_MATCHED_RANDOM]["per_clip"]
        for clip_id in sorted(corpus):
            delta = float(claimed_per_clip_rmr[clip_id]["mae_deg"]) - float(
                claimed_per_clip_cand[clip_id]["mae_deg"]
            )
            per_clip_recomputed.setdefault(clip_id, []).append(delta)

    per_clip_deltas: list[tuple[str, float]] = [
        (clip_id, sum(values) / len(values)) for clip_id, values in sorted(per_clip_recomputed.items())
    ]
    flat_deltas = [delta for _cid, delta in per_clip_deltas]
    t_observed, primary_p, permutations = _clip_sign_flip_meet_in_middle(flat_deltas)
    mean_clip_delta = t_observed / len(flat_deltas)

    by_room: dict[str, list[float]] = {}
    for clip_id, delta in per_clip_deltas:
        by_room.setdefault(room_of.get(clip_id, ""), []).append(delta)
    n_room_favorable, room_p, n_rooms, per_room = _room_majority(by_room)

    seed_observed, secondary_p, seed_permutations = _sign_flip_one_sided_brute_force(seed_level_deltas)

    candidate_means = []
    rmr_means = []
    for seed_block in per_seed:
        candidate_means.append(float(seed_block["arm_scores_macro"][ARM_CANDIDATE]["macro_mae_deg"]))
        rmr_means.append(float(seed_block["arm_scores_macro"][ARM_RATE_MATCHED_RANDOM]["macro_mae_deg"]))
    candidate_mean = sum(candidate_means) / len(candidate_means)
    rmr_mean = sum(rmr_means) / len(rmr_means)
    point_estimate_holds = candidate_mean < rmr_mean

    return {
        "candidate_mean_macro_mae_deg": candidate_mean,
        "rate_matched_random_mean_macro_mae_deg": rmr_mean,
        "point_estimate_holds": point_estimate_holds,
        "mean_clip_delta": mean_clip_delta,
        "primary_one_sided_p": primary_p,
        "primary_permutations": permutations,
        "primary_significant": primary_p <= PRIMARY_ALPHA,
        "secondary_mean_delta": seed_observed,
        "secondary_one_sided_p": secondary_p,
        "secondary_permutations": seed_permutations,
        "secondary_direction_agrees": seed_observed > 0.0,
        "room_n_favorable": n_room_favorable,
        "room_one_sided_p": room_p,
        "room_n_rooms": n_rooms,
        "room_per_room": per_room,
    }


def verify_doa_artifact(artifact: dict) -> DoaVerificationResult:

    if not isinstance(artifact, dict):
        raise DoaVerificationRefusal("artifact must be a JSON object")
    mismatches: list[str] = []

    stored_seal = artifact.get("seal")
    body = {k: v for k, v in artifact.items() if k != "seal"}
    seal_intact = isinstance(stored_seal, str) and stored_seal == _canonical_sha256(body)
    if not seal_intact:
        mismatches.append("stored seal does not match a re-hash of the artifact body")

    schema_ok = True
    if artifact.get("schema") != EXPECTED_ARTIFACT_SCHEMA:
        schema_ok = False
        mismatches.append(f"schema is {artifact.get('schema')!r}, expected {EXPECTED_ARTIFACT_SCHEMA!r}")
    if artifact.get("stage") != EXPECTED_STAGE:
        schema_ok = False
        mismatches.append(f"stage is {artifact.get('stage')!r}, expected {EXPECTED_STAGE}")
    if artifact.get("claim_scope") != EXPECTED_CLAIM_SCOPE:
        schema_ok = False
        mismatches.append("claim_scope was widened away from the frozen DoA bed contract")
    architectures = artifact.get("architectures")
    if architectures != list(ARCHITECTURES):
        schema_ok = False
        mismatches.append(f"architectures is {architectures!r}, expected {list(ARCHITECTURES)!r}")

    corpus = artifact.get("corpus_tracks")
    if not isinstance(corpus, dict) or not corpus:
        raise DoaVerificationRefusal("artifact.corpus_tracks must be a nonempty object")
    cold_start = artifact.get("cold_start")
    if not isinstance(cold_start, list) or len(cold_start) != 2:
        raise DoaVerificationRefusal("artifact.cold_start must be an [azimuth, elevation] pair")

    scores_reproduced = True
    stats_reproduced = True
    per_architecture: dict[str, dict] = {}
    for architecture in ARCHITECTURES:
        before = len(mismatches)
        rescored = _rescore_architecture(artifact, architecture, cold_start, mismatches)
        if len(mismatches) > before:
            scores_reproduced = False
        per_architecture[architecture] = rescored

        stats_block = artifact.get("stats", {}).get(architecture, {})
        if not isinstance(stats_block, dict):
            raise DoaVerificationRefusal(f"stats[{architecture}] must be present")

        primary = stats_block.get("primary_clip_level_sign_flip", {})
        if not _agree(primary.get("mean_delta"), rescored["mean_clip_delta"]) or not _agree(
            primary.get("one_sided_p"), rescored["primary_one_sided_p"]
        ):
            stats_reproduced = False
            mismatches.append(f"{architecture} primary clip-level sign-flip disagrees with recomputation")

        secondary = stats_block.get("secondary_seed_sign_flip", {})
        if not _agree(secondary.get("mean_delta"), rescored["secondary_mean_delta"]) or not _agree(
            secondary.get("one_sided_p"), rescored["secondary_one_sided_p"]
        ):
            stats_reproduced = False
            mismatches.append(f"{architecture} secondary 5-seed sign-flip disagrees with recomputation")

        room = stats_block.get("room_majority", {})
        if room.get("n_rooms_favorable") != rescored["room_n_favorable"] or not _agree(
            room.get("one_sided_p"), rescored["room_one_sided_p"]
        ):
            stats_reproduced = False
            mismatches.append(f"{architecture} room-majority collapse disagrees with recomputation")

        if stats_block.get("point_estimate_holds") != rescored["point_estimate_holds"]:
            stats_reproduced = False
            mismatches.append(f"{architecture} point_estimate_holds disagrees with recomputation")

        sesoi_block = stats_block.get("sesoi", {})
        sesoi_deg = sesoi_block.get("sesoi_f1")
        if isinstance(sesoi_deg, (int, float)) and not isinstance(sesoi_deg, bool):
            expected_exceeds = rescored["mean_clip_delta"] >= float(sesoi_deg)
            if bool(sesoi_block.get("exceeds_sesoi")) != expected_exceeds:
                stats_reproduced = False
                mismatches.append(
                    f"{architecture} sesoi.exceeds_sesoi disagrees with the recomputed mean delta"
                )
        else:
            stats_reproduced = False
            mismatches.append(f"{architecture} stats.sesoi.sesoi_f1 is missing or not numeric")

        expected_survives = bool(
            rescored["point_estimate_holds"]
            and isinstance(sesoi_deg, (int, float))
            and not isinstance(sesoi_deg, bool)
            and rescored["mean_clip_delta"] >= float(sesoi_deg)
            and rescored["primary_significant"]
            and rescored["secondary_direction_agrees"]
            and rescored["room_one_sided_p"] <= ROOM_MAJORITY_ALPHA
        )
        if stats_block.get("survives") != expected_survives:
            stats_reproduced = False
            mismatches.append(
                f"{architecture} survives stored {stats_block.get('survives')} recomputed {expected_survives}"
            )
        rescored["survives"] = expected_survives

    both_survive = bool(per_architecture[ARCH_A_ID]["survives"] and per_architecture[ARCH_B_ID]["survives"])
    exactly_one = bool(per_architecture[ARCH_A_ID]["survives"] != per_architecture[ARCH_B_ID]["survives"])
    if artifact.get("stats", {}).get("both_architectures_survive") != both_survive:
        stats_reproduced = False
        mismatches.append(
            "stats.both_architectures_survive disagrees with the recomputed per-architecture survival"
        )
    if artifact.get("stats", {}).get("architecture_fragile") != exactly_one:
        stats_reproduced = False
        mismatches.append(
            "stats.architecture_fragile disagrees with the recomputed per-architecture survival"
        )

    expected_verdict = "mechanics-ok" if both_survive else ("architecture-fragile" if exactly_one else "null")
    if artifact.get("verdict") != expected_verdict:
        stats_reproduced = False
        mismatches.append(f"verdict stored {artifact.get('verdict')!r} recomputed {expected_verdict!r}")

    honesty_ok = True
    flags = artifact.get("flags")
    if not isinstance(flags, dict):
        raise DoaVerificationRefusal("artifact.flags must be present")
    if flags.get("activation_allowed") is not False:
        honesty_ok = False
        mismatches.append("flags.activation_allowed must be hardcoded false")
    if flags.get("scientific_promotion") is not False:
        honesty_ok = False
        mismatches.append("flags.scientific_promotion must be hardcoded false")
    if flags.get("independent_scientific_confirmation") is not False:
        honesty_ok = False
        mismatches.append("producer must not self-certify independent_scientific_confirmation")
    for architecture in ARCHITECTURES:
        claim_verb = artifact.get("stats", {}).get(architecture, {}).get("claim_verb")
        if claim_verb in FORBIDDEN_CLAIM_VERBS or claim_verb not in ALLOWED_CLAIM_VERBS:
            honesty_ok = False
            mismatches.append(
                f"{architecture} claim_verb {claim_verb!r} exceeds the clip-limited claim ceiling"
            )

    independent_referee_reproduction = (
        seal_intact and schema_ok and scores_reproduced and stats_reproduced and honesty_ok
    )

    source_kind = str(artifact.get("source_kind", ""))
    rights_clean = artifact.get("rights_clean") is True
    reproductions = artifact.get("reproductions")
    if isinstance(reproductions, bool) or not isinstance(reproductions, int) or reproductions < 0:
        reproductions = 0
    noisy_tv_both = artifact.get("noisy_tv_at_chance_both_architectures") is True

    independent_scientific_confirmation = (
        independent_referee_reproduction
        and source_kind == "real"
        and rights_clean
        and noisy_tv_both
        and reproductions >= MIN_REPRODUCTIONS
    )

    return DoaVerificationResult(
        seal_intact=seal_intact,
        schema_ok=schema_ok,
        scores_reproduced=scores_reproduced,
        stats_reproduced=stats_reproduced,
        honesty_ok=honesty_ok,
        independent_referee_reproduction=independent_referee_reproduction,
        independent_scientific_confirmation=independent_scientific_confirmation,
        source_kind=source_kind,
        rights_clean=rights_clean,
        reproductions=reproductions,
        mismatches=tuple(mismatches),
        detail={
            "per_architecture": per_architecture,
            "both_architectures_survive": both_survive,
            "architecture_fragile": exactly_one,
            "expected_verdict": expected_verdict,
            "noisy_tv_at_chance_both_architectures": noisy_tv_both,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def doa_verification_payload(result: DoaVerificationResult) -> dict:

    body = {
        "schema": VERIFIER_SCHEMA,
        "seal_intact": result.seal_intact,
        "schema_ok": result.schema_ok,
        "scores_reproduced": result.scores_reproduced,
        "stats_reproduced": result.stats_reproduced,
        "honesty_ok": result.honesty_ok,
        "independent_referee_reproduction": result.independent_referee_reproduction,
        "independent_scientific_confirmation": result.independent_scientific_confirmation,
        "source_kind": result.source_kind,
        "rights_clean": result.rights_clean,
        "reproductions": result.reproductions,
        "mismatches": list(result.mismatches),
        "detail": result.detail,
    }
    body["seal"] = _canonical_sha256(body)
    return body


def verify_sealed_doa_file(in_path: str) -> dict:

    with open(in_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return doa_verification_payload(verify_doa_artifact(artifact))


def write_doa_verification(payload: dict, out_path: str) -> None:

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
