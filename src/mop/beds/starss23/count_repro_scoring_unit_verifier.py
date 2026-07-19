
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-count-repro-scoring-unit-verification/v1"

EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-count-repro-scoring-unit-bed/v1"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
EXPECTED_REPRO_AXIS = "scoring_unit"
EXPECTED_SCORING_UNIT = "clip-macro"
ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"
COLD_START = 0

MIN_REPRODUCTIONS = 3

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

_TOL = 1e-9
_TIE_EPS = 1e-9


class CountReproScoringUnitVerificationRefusal(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _as_count_track(track: object, label: str) -> list[int]:
    if not isinstance(track, (list, tuple)):
        raise CountReproScoringUnitVerificationRefusal(f"{label} must be a list of nonnegative integers")
    cleaned: list[int] = []
    for item in track:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CountReproScoringUnitVerificationRefusal(f"{label} must contain only nonnegative integers")
        cleaned.append(item)
    return cleaned


def _as_reestimates(frames: object, n_frames: int, label: str) -> list[int]:
    if not isinstance(frames, (list, tuple)):
        raise CountReproScoringUnitVerificationRefusal(f"{label} must be a list of frame indices")
    cleaned: list[int] = []
    last = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise CountReproScoringUnitVerificationRefusal(f"{label} indices must lie in [0, {n_frames})")
        if frame <= last:
            raise CountReproScoringUnitVerificationRefusal(f"{label} must be strictly sorted and unique")
        last = frame
        cleaned.append(frame)
    return cleaned


def _coast(estimator: list[int], reestimates: list[int], cold_start: int = COLD_START) -> list[int]:

    fire = set(reestimates)
    emitted: list[int] = []
    held = cold_start
    for t in range(len(estimator)):
        if t in fire:
            held = estimator[t]
        emitted.append(held)
    return emitted


def _abs_error_sum(gt: list[int], emitted: list[int]) -> int:
    if len(gt) != len(emitted):
        raise CountReproScoringUnitVerificationRefusal("ground-truth and emitted tracks differ in length")
    return sum(abs(emitted[t] - gt[t]) for t in range(len(gt)))


def _reestimates_for_arm(arm: str, clip_id: str, n_frames: int, reestimates_by_clip: dict) -> list[int]:
    if arm == ARM_ALWAYS_ON:
        return list(range(n_frames))
    if arm == ARM_NEVER_UPDATE:
        return []
    stored = reestimates_by_clip.get(clip_id, {})
    if not isinstance(stored, dict):
        raise CountReproScoringUnitVerificationRefusal(f"clip {clip_id!r} reestimate_frames must be an object")
    return _as_reestimates(stored.get(arm), n_frames, f"{arm} reestimate_frames on {clip_id}")


def _sign_flip_one_sided_small(deltas: list[float]) -> tuple[float, float, int]:

    n = len(deltas)
    if n == 0:
        raise CountReproScoringUnitVerificationRefusal("the sign-flip test needs at least one paired delta")
    observed = sum(deltas) / n
    at_least = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        flipped_mean = sum(s * d for s, d in zip(signs, deltas, strict=True)) / n
        if flipped_mean >= observed - _TOL:
            at_least += 1
    return observed, at_least / total, total


def _count_ge(sorted_vals: list[float], threshold: float) -> int:

    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < threshold:
            lo = mid + 1
        else:
            hi = mid
    return len(sorted_vals) - lo


def _partial_sums(part: list[float]) -> list[float]:
    sums = [0.0]
    for value in part:
        nxt: list[float] = []
        for acc in sums:
            nxt.append(acc + value)
            nxt.append(acc - value)
        sums = nxt
    return sums


def _sign_flip_one_sided_clips(deltas: list[float]) -> tuple[float, float, int]:

    n = len(deltas)
    if n == 0:
        raise CountReproScoringUnitVerificationRefusal("the clip sign-flip needs at least one paired delta")
    t_observed = sum(deltas)
    half = n // 2
    right_sums = sorted(_partial_sums(deltas[half:]))
    threshold_base = t_observed - _TIE_EPS
    count = 0
    for left_sum in _partial_sums(deltas[:half]):
        count += _count_ge(right_sums, threshold_base - left_sum)
    permutations = 2**n
    return t_observed, count / permutations, permutations


@dataclass(frozen=True, slots=True)
class CountReproScoringUnitVerificationResult:

    seal_intact: bool
    schema_ok: bool
    scores_reproduced: bool
    stats_reproduced: bool
    clip_cluster_reproduced: bool
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


def _agree(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= _TOL


def _macro_score_arm(
    clip_ids: list[str],
    corpus: dict,
    reestimates_by_clip: dict,
    arm: str,
    candidate_count_by_clip: dict[str, int],
    mismatches: list[str],
) -> tuple[float, dict[str, tuple[int, int, float]], bool]:

    per_clip: dict[str, tuple[int, int, float]] = {}
    budget_ok = True
    for clip_id in clip_ids:
        block = corpus.get(clip_id)
        if not isinstance(block, dict):
            raise CountReproScoringUnitVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
        gt = _as_count_track(block.get("gt_count_track"), "gt_count_track")
        estimator = _as_count_track(block.get("estimator_track"), "estimator_track")
        if len(gt) != len(estimator):
            raise CountReproScoringUnitVerificationRefusal(
                f"clip {clip_id!r} gt and estimator tracks differ in length"
            )
        n = len(estimator)
        reestimates = _reestimates_for_arm(arm, clip_id, n, reestimates_by_clip)
        if arm == ARM_RATE_MATCHED_RANDOM and len(reestimates) != candidate_count_by_clip.get(clip_id):
            budget_ok = False
            mismatches.append(
                f"clip {clip_id} rate_matched_random spends {len(reestimates)} re-estimations but the "
                f"candidate spends {candidate_count_by_clip.get(clip_id)}: budget not matched"
            )
        emitted = _coast(estimator, reestimates, COLD_START)
        abs_error = _abs_error_sum(gt, emitted)
        if n == 0:
            raise CountReproScoringUnitVerificationRefusal(f"clip {clip_id!r} scored zero frames")
        per_clip[clip_id] = (abs_error, n, abs_error / n)
    if not per_clip:
        raise CountReproScoringUnitVerificationRefusal("an arm scored zero clips")
    macro_mae = sum(triple[2] for triple in per_clip.values()) / len(per_clip)
    return macro_mae, per_clip, budget_ok


def verify_count_repro_scoring_unit_artifact(
    artifact: dict,
) -> CountReproScoringUnitVerificationResult:

    if not isinstance(artifact, dict):
        raise CountReproScoringUnitVerificationRefusal("artifact must be a JSON object")
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
        mismatches.append("claim_scope was widened away from the frozen count bed contract")
    if artifact.get("reproduction_axis") != EXPECTED_REPRO_AXIS:
        schema_ok = False
        mismatches.append(f"reproduction_axis is {artifact.get('reproduction_axis')!r}, expected scoring_unit")
    if artifact.get("scoring_unit") != EXPECTED_SCORING_UNIT:
        schema_ok = False
        mismatches.append(f"scoring_unit is {artifact.get('scoring_unit')!r}, expected clip-macro")
    if artifact.get("primary_control") != ARM_RATE_MATCHED_RANDOM:
        raise CountReproScoringUnitVerificationRefusal("artifact.primary_control must be rate_matched_random")

    corpus = artifact.get("corpus_tracks")
    if not isinstance(corpus, dict) or not corpus:
        raise CountReproScoringUnitVerificationRefusal("artifact.corpus_tracks must be a nonempty object")
    per_seed = artifact.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        raise CountReproScoringUnitVerificationRefusal("artifact.per_seed must be a nonempty list")

    scores_reproduced = True
    budget_matched = True
    recomputed_deltas: list[float] = []
    candidate_clip_mae_by_seed: list[dict[str, float]] = []
    control_clip_mae_by_seed: list[dict[str, float]] = []
    reference_clip_ids: list[str] | None = None
    for seed_block in per_seed:
        clips = seed_block.get("clips")
        arm_scores = seed_block.get("arm_scores")
        if not isinstance(clips, list) or not clips or not isinstance(arm_scores, dict):
            raise CountReproScoringUnitVerificationRefusal(
                "each per_seed entry needs a nonempty clips list and arm_scores"
            )

        clip_ids: list[str] = []
        reestimates_by_clip: dict[str, dict] = {}
        candidate_count_by_clip: dict[str, int] = {}
        for clip in clips:
            clip_id = clip.get("clip_id")
            if not isinstance(clip_id, str):
                raise CountReproScoringUnitVerificationRefusal("each per_seed clip needs a string clip_id")
            block = corpus.get(clip_id)
            if not isinstance(block, dict):
                raise CountReproScoringUnitVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
            n = len(_as_count_track(block.get("estimator_track"), "estimator_track"))
            clip_ids.append(clip_id)
            reestimates_by_clip[clip_id] = clip.get("reestimate_frames", {})
            candidate_count_by_clip[clip_id] = len(
                _as_reestimates(
                    reestimates_by_clip[clip_id].get(ARM_CANDIDATE)
                    if isinstance(reestimates_by_clip[clip_id], dict)
                    else None,
                    n,
                    f"candidate reestimate_frames on {clip_id}",
                )
            )
        if reference_clip_ids is None:
            reference_clip_ids = list(clip_ids)
        elif clip_ids != reference_clip_ids:
            scores_reproduced = False
            mismatches.append("per_seed clip ordering is not stable across seeds")

        recomputed_macro: dict[str, float] = {}
        recomputed_per_clip: dict[str, dict[str, tuple[int, int, float]]] = {}
        for arm in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE):
            macro_mae, per_clip, budget_ok = _macro_score_arm(
                clip_ids, corpus, reestimates_by_clip, arm, candidate_count_by_clip, mismatches
            )
            budget_matched = budget_matched and budget_ok
            recomputed_macro[arm] = macro_mae
            recomputed_per_clip[arm] = per_clip
            claimed = arm_scores.get(arm)
            seed = seed_block.get("seed")
            if not isinstance(claimed, dict):
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} carries no stored score")
                continue
            if claimed.get("scoring_unit") != EXPECTED_SCORING_UNIT:
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} scoring_unit is not clip-macro")
            if not _agree(claimed.get("macro_mae"), macro_mae):
                scores_reproduced = False
                mismatches.append(
                    f"seed {seed} arm {arm} macro_mae stored {claimed.get('macro_mae')} recomputed {macro_mae}"
                )
            stored_per_clip = claimed.get("per_clip")
            if not isinstance(stored_per_clip, dict) or len(stored_per_clip) != len(per_clip):
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} per_clip block is missing or the wrong size")
            else:
                for clip_id, (abs_err, nfr, mae_c) in per_clip.items():
                    stored_clip = stored_per_clip.get(clip_id)
                    if not isinstance(stored_clip, dict):
                        scores_reproduced = False
                        mismatches.append(f"seed {seed} arm {arm} clip {clip_id} has no stored per-clip score")
                        continue
                    if stored_clip.get("abs_error_sum") != abs_err:
                        scores_reproduced = False
                        mismatches.append(
                            f"seed {seed} arm {arm} clip {clip_id} abs_error_sum stored "
                            f"{stored_clip.get('abs_error_sum')} recomputed {abs_err}"
                        )
                    if stored_clip.get("n_frames") != nfr:
                        scores_reproduced = False
                        mismatches.append(
                            f"seed {seed} arm {arm} clip {clip_id} n_frames stored "
                            f"{stored_clip.get('n_frames')} recomputed {nfr}"
                        )
                    if not _agree(stored_clip.get("mae"), mae_c):
                        scores_reproduced = False
                        mismatches.append(
                            f"seed {seed} arm {arm} clip {clip_id} mae stored {stored_clip.get('mae')} "
                            f"recomputed {mae_c}"
                        )

        recomputed_deltas.append(
            recomputed_macro[ARM_RATE_MATCHED_RANDOM] - recomputed_macro[ARM_CANDIDATE]
        )
        candidate_clip_mae_by_seed.append(
            {cid: recomputed_per_clip[ARM_CANDIDATE][cid][2] for cid in clip_ids}
        )
        control_clip_mae_by_seed.append(
            {cid: recomputed_per_clip[ARM_RATE_MATCHED_RANDOM][cid][2] for cid in clip_ids}
        )

    if not budget_matched:
        scores_reproduced = False

    stats = artifact.get("stats")
    if not isinstance(stats, dict):
        raise CountReproScoringUnitVerificationRefusal("artifact.stats must be present")
    stats_reproduced = True
    claimed_deltas = stats.get("deltas")
    if not isinstance(claimed_deltas, list) or len(claimed_deltas) != len(recomputed_deltas):
        stats_reproduced = False
        mismatches.append("stats.deltas is missing or the wrong length")
    else:
        for i, (claimed_d, mine_d) in enumerate(zip(claimed_deltas, recomputed_deltas, strict=True)):
            if not _agree(claimed_d, mine_d):
                stats_reproduced = False
                mismatches.append(f"stats delta {i} stored {claimed_d} recomputed {mine_d}")

    t_obs, one_sided_p, n_perm = _sign_flip_one_sided_small(recomputed_deltas)
    if not _agree(stats.get("t_obs"), t_obs):
        stats_reproduced = False
        mismatches.append(f"stats.t_obs stored {stats.get('t_obs')} recomputed {t_obs}")
    if not _agree(stats.get("one_sided_p"), one_sided_p):
        stats_reproduced = False
        mismatches.append(f"stats.one_sided_p stored {stats.get('one_sided_p')} recomputed {one_sided_p}")
    if stats.get("n_permutations") != n_perm:
        stats_reproduced = False
        mismatches.append(f"stats.n_permutations stored {stats.get('n_permutations')} recomputed {n_perm}")
    expected_two_sided_reachable = (2.0 / n_perm) <= 0.05
    if bool(stats.get("two_sided_005_reachable")) != expected_two_sided_reachable:
        stats_reproduced = False
        mismatches.append("stats.two_sided_005_reachable disagrees with the exact discrete floor")

    mean_delta = t_obs
    sesoi = None
    prereg = artifact.get("prereg")
    if isinstance(prereg, dict):
        sesoi = prereg.get("sesoi_mae")
    if sesoi is None:
        sesoi = stats.get("sesoi_mae")
    if isinstance(sesoi, (int, float)) and not isinstance(sesoi, bool):
        exceeds = mean_delta > float(sesoi)
        if bool(stats.get("mean_delta_exceeds_sesoi")) != exceeds:
            stats_reproduced = False
            mismatches.append("stats.mean_delta_exceeds_sesoi disagrees with the recomputed mean delta")

    clip_cluster_reproduced = True
    clip_ids_ref = reference_clip_ids or []
    n_seeds = len(candidate_clip_mae_by_seed)
    per_clip_delta: dict[str, float] = {}
    for cid in clip_ids_ref:
        cand_mean = sum(seed_map[cid] for seed_map in candidate_clip_mae_by_seed) / n_seeds
        rmr_mean = sum(seed_map[cid] for seed_map in control_clip_mae_by_seed) / n_seeds
        per_clip_delta[cid] = rmr_mean - cand_mean
    cluster_deltas = [per_clip_delta[cid] for cid in clip_ids_ref]
    cluster_t_obs, cluster_p, cluster_perm = _sign_flip_one_sided_clips(cluster_deltas)
    cluster_mean_delta = cluster_t_obs / len(cluster_deltas)
    cluster_direction_agrees = cluster_mean_delta > 0.0
    cluster_fraction_candidate_lower = sum(1 for d in cluster_deltas if d > 0.0) / len(cluster_deltas)

    clip_cluster = artifact.get("clip_cluster")
    if not isinstance(clip_cluster, dict):
        raise CountReproScoringUnitVerificationRefusal("artifact.clip_cluster must be present")
    permutation = clip_cluster.get("permutation")
    if not isinstance(permutation, dict):
        raise CountReproScoringUnitVerificationRefusal("artifact.clip_cluster.permutation must be present")
    if permutation.get("n_clips") != len(cluster_deltas):
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster n_clips disagrees with the number of test clips")
    if permutation.get("permutations") != cluster_perm:
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster permutations count disagrees with 2^n_clips")
    if not _agree(permutation.get("one_sided_p"), cluster_p):
        clip_cluster_reproduced = False
        mismatches.append(
            f"clip_cluster one_sided_p stored {permutation.get('one_sided_p')} recomputed {cluster_p}"
        )
    if not _agree(permutation.get("mean_delta"), cluster_mean_delta):
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster mean_delta disagrees with the recomputed per-clip mean delta")
    if not _agree(permutation.get("fraction_candidate_lower"), cluster_fraction_candidate_lower):
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster fraction_candidate_lower disagrees with the recomputed fraction")
    if bool(permutation.get("direction_agrees")) != cluster_direction_agrees:
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster direction_agrees disagrees with the recomputed sign of the mean delta")
    if bool(stats.get("clip_cluster_direction_agrees")) != cluster_direction_agrees:
        clip_cluster_reproduced = False
        mismatches.append("stats.clip_cluster_direction_agrees disagrees with the recomputed direction")
    stored_per_clip_delta = clip_cluster.get("per_clip_delta")
    if isinstance(stored_per_clip_delta, dict):
        for cid in clip_ids_ref:
            if not _agree(stored_per_clip_delta.get(cid), per_clip_delta[cid]):
                clip_cluster_reproduced = False
                mismatches.append(f"clip_cluster per_clip_delta for {cid} disagrees with the re-derivation")
    else:
        clip_cluster_reproduced = False
        mismatches.append("clip_cluster.per_clip_delta is missing")

    honesty_ok = True
    flags = artifact.get("flags")
    if not isinstance(flags, dict):
        raise CountReproScoringUnitVerificationRefusal("artifact.flags must be present")
    if flags.get("activation_allowed") is not False:
        honesty_ok = False
        mismatches.append("flags.activation_allowed must be hardcoded false")
    if flags.get("scientific_promotion") is not False:
        honesty_ok = False
        mismatches.append("flags.scientific_promotion must be hardcoded false")
    if flags.get("independent_scientific_confirmation") is not False:
        honesty_ok = False
        mismatches.append("producer must not self-certify independent_scientific_confirmation")
    claim_verb = stats.get("claim_verb")
    if claim_verb in FORBIDDEN_CLAIM_VERBS or claim_verb not in ALLOWED_CLAIM_VERBS:
        honesty_ok = False
        mismatches.append(f"claim_verb {claim_verb!r} exceeds the clip-limited claim ceiling")

    independent_referee_reproduction = (
        seal_intact
        and schema_ok
        and scores_reproduced
        and stats_reproduced
        and clip_cluster_reproduced
        and honesty_ok
    )

    source_kind = str(artifact.get("source_kind", ""))
    rights_clean = artifact.get("rights_clean") is True
    reproductions = artifact.get("reproductions")
    if isinstance(reproductions, bool) or not isinstance(reproductions, int) or reproductions < 0:
        reproductions = 0
    controls = artifact.get("controls") if isinstance(artifact.get("controls"), dict) else {}
    noisy_tv_at_chance = controls.get("noisy_tv_at_chance") is True

    independent_scientific_confirmation = (
        independent_referee_reproduction
        and source_kind == "real"
        and rights_clean
        and noisy_tv_at_chance
        and reproductions >= MIN_REPRODUCTIONS
    )

    return CountReproScoringUnitVerificationResult(
        seal_intact=seal_intact,
        schema_ok=schema_ok,
        scores_reproduced=scores_reproduced,
        stats_reproduced=stats_reproduced,
        clip_cluster_reproduced=clip_cluster_reproduced,
        honesty_ok=honesty_ok,
        independent_referee_reproduction=independent_referee_reproduction,
        independent_scientific_confirmation=independent_scientific_confirmation,
        source_kind=source_kind,
        rights_clean=rights_clean,
        reproductions=reproductions,
        mismatches=tuple(mismatches),
        detail={
            "recomputed_deltas": recomputed_deltas,
            "recomputed_t_obs": t_obs,
            "recomputed_one_sided_p": one_sided_p,
            "recomputed_mean_delta_control_minus_candidate": mean_delta,
            "n_permutations": n_perm,
            "sesoi_mae": sesoi,
            "clip_cluster_one_sided_p": cluster_p,
            "clip_cluster_mean_delta": cluster_mean_delta,
            "clip_cluster_direction_agrees": cluster_direction_agrees,
            "clip_cluster_fraction_candidate_lower": cluster_fraction_candidate_lower,
            "clip_cluster_permutations": cluster_perm,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "budget_matched": budget_matched,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def count_repro_scoring_unit_verification_payload(
    result: CountReproScoringUnitVerificationResult,
) -> dict:

    body = {
        "schema": VERIFIER_SCHEMA,
        "reproduction_axis": EXPECTED_REPRO_AXIS,
        "scoring_unit": EXPECTED_SCORING_UNIT,
        "seal_intact": result.seal_intact,
        "schema_ok": result.schema_ok,
        "scores_reproduced": result.scores_reproduced,
        "stats_reproduced": result.stats_reproduced,
        "clip_cluster_reproduced": result.clip_cluster_reproduced,
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


def verify_sealed_count_repro_scoring_unit_file(in_path: str) -> dict:

    with open(in_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return count_repro_scoring_unit_verification_payload(
        verify_count_repro_scoring_unit_artifact(artifact)
    )


def write_count_repro_scoring_unit_verification(payload: dict, out_path: str) -> None:

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
