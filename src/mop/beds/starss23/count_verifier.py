from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-count-bed-verification/v1"

EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed/v1"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
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


class CountVerificationRefusal(ValueError):
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
        raise CountVerificationRefusal(f"{label} must be a list of nonnegative integers")
    cleaned: list[int] = []
    for item in track:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CountVerificationRefusal(f"{label} must contain only nonnegative integers")
        cleaned.append(item)
    return cleaned


def _as_reestimates(frames: object, n_frames: int, label: str) -> list[int]:
    if not isinstance(frames, (list, tuple)):
        raise CountVerificationRefusal(f"{label} must be a list of frame indices")
    cleaned: list[int] = []
    last = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise CountVerificationRefusal(f"{label} indices must lie in [0, {n_frames})")
        if frame <= last:
            raise CountVerificationRefusal(f"{label} must be strictly sorted and unique")
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
        raise CountVerificationRefusal("ground-truth and emitted tracks differ in length")
    return sum(abs(emitted[t] - gt[t]) for t in range(len(gt)))


def _reestimates_for_arm(
    arm: str,
    clip_id: str,
    n_frames: int,
    reestimates_by_clip: dict,
) -> list[int]:

    if arm == ARM_ALWAYS_ON:
        return list(range(n_frames))
    if arm == ARM_NEVER_UPDATE:
        return []
    stored = reestimates_by_clip.get(clip_id, {})
    if not isinstance(stored, dict):
        raise CountVerificationRefusal(f"clip {clip_id!r} reestimate_frames must be an object")
    return _as_reestimates(stored.get(arm), n_frames, f"{arm} reestimate_frames on {clip_id}")


def _sign_flip_one_sided(deltas: list[float]) -> tuple[float, float, int]:
    n = len(deltas)
    if n == 0:
        raise CountVerificationRefusal("the sign-flip test needs at least one paired delta")
    observed = sum(deltas) / n
    at_least = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        flipped_mean = sum(s * d for s, d in zip(signs, deltas, strict=True)) / n
        if flipped_mean >= observed - _TOL:
            at_least += 1
    return observed, at_least / total, total


@dataclass(frozen=True, slots=True)
class CountVerificationResult:
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


def _agree(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= _TOL


def _score_arm_pooled(
    clip_ids: list[str],
    corpus: dict,
    reestimates_by_clip: dict,
    arm: str,
    candidate_count_by_clip: dict[str, int],
    mismatches: list[str],
) -> tuple[int, int, bool]:

    abs_error = 0
    frames = 0
    budget_ok = True
    for clip_id in clip_ids:
        block = corpus.get(clip_id)
        if not isinstance(block, dict):
            raise CountVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
        gt = _as_count_track(block.get("gt_count_track"), "gt_count_track")
        estimator = _as_count_track(block.get("estimator_track"), "estimator_track")
        if len(gt) != len(estimator):
            raise CountVerificationRefusal(f"clip {clip_id!r} gt and estimator tracks differ in length")
        n = len(estimator)
        reestimates = _reestimates_for_arm(arm, clip_id, n, reestimates_by_clip)
        if arm == ARM_RATE_MATCHED_RANDOM and len(reestimates) != candidate_count_by_clip.get(clip_id):
            budget_ok = False
            mismatches.append(
                f"clip {clip_id} rate_matched_random spends {len(reestimates)} re-estimations but the "
                f"candidate spends {candidate_count_by_clip.get(clip_id)}: budget not matched"
            )
        emitted = _coast(estimator, reestimates, COLD_START)
        abs_error += _abs_error_sum(gt, emitted)
        frames += n
    return abs_error, frames, budget_ok


def verify_count_artifact(artifact: dict) -> CountVerificationResult:

    if not isinstance(artifact, dict):
        raise CountVerificationRefusal("artifact must be a JSON object")
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
    if artifact.get("primary_control") != ARM_RATE_MATCHED_RANDOM:
        raise CountVerificationRefusal("artifact.primary_control must be rate_matched_random")

    corpus = artifact.get("corpus_tracks")
    if not isinstance(corpus, dict) or not corpus:
        raise CountVerificationRefusal("artifact.corpus_tracks must be a nonempty object")
    per_seed = artifact.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        raise CountVerificationRefusal("artifact.per_seed must be a nonempty list")

    scores_reproduced = True
    budget_matched = True
    recomputed_deltas: list[float] = []
    for seed_block in per_seed:
        clips = seed_block.get("clips")
        arm_scores = seed_block.get("arm_scores")
        if not isinstance(clips, list) or not clips or not isinstance(arm_scores, dict):
            raise CountVerificationRefusal("each per_seed entry needs a nonempty clips list and arm_scores")

        clip_ids: list[str] = []
        reestimates_by_clip: dict[str, dict] = {}
        candidate_count_by_clip: dict[str, int] = {}
        for clip in clips:
            clip_id = clip.get("clip_id")
            if not isinstance(clip_id, str):
                raise CountVerificationRefusal("each per_seed clip needs a string clip_id")
            block = corpus.get(clip_id)
            if not isinstance(block, dict):
                raise CountVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
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

        recomputed_mae: dict[str, float] = {}
        for arm in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE):
            abs_error, frames, budget_ok = _score_arm_pooled(
                clip_ids, corpus, reestimates_by_clip, arm, candidate_count_by_clip, mismatches
            )
            budget_matched = budget_matched and budget_ok
            if frames == 0:
                raise CountVerificationRefusal("a seed scored zero frames")
            mae = abs_error / frames
            recomputed_mae[arm] = mae
            claimed = arm_scores.get(arm)
            seed = seed_block.get("seed")
            if not isinstance(claimed, dict):
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} carries no stored score")
                continue
            if claimed.get("abs_error_sum") != abs_error:
                scores_reproduced = False
                mismatches.append(
                    f"seed {seed} arm {arm} abs_error_sum stored {claimed.get('abs_error_sum')} "
                    f"recomputed {abs_error}"
                )
            if claimed.get("n_frames") != frames:
                scores_reproduced = False
                mismatches.append(
                    f"seed {seed} arm {arm} n_frames stored {claimed.get('n_frames')} recomputed {frames}"
                )
            if not _agree(claimed.get("mae"), mae):
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} mae stored {claimed.get('mae')} recomputed {mae}")

        recomputed_deltas.append(recomputed_mae[ARM_RATE_MATCHED_RANDOM] - recomputed_mae[ARM_CANDIDATE])

    if not budget_matched:
        scores_reproduced = False

    stats = artifact.get("stats")
    if not isinstance(stats, dict):
        raise CountVerificationRefusal("artifact.stats must be present")
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

    t_obs, one_sided_p, n_perm = _sign_flip_one_sided(recomputed_deltas)
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

    honesty_ok = True
    flags = artifact.get("flags")
    if not isinstance(flags, dict):
        raise CountVerificationRefusal("artifact.flags must be present")
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
        seal_intact and schema_ok and scores_reproduced and stats_reproduced and honesty_ok
    )

    source_kind = str(artifact.get("source_kind", ""))
    rights_clean = artifact.get("rights_clean") is True
    reproductions = artifact.get("reproductions")
    if isinstance(reproductions, bool) or not isinstance(reproductions, int) or reproductions < 0:
        reproductions = 0
    controls = artifact.get("controls")
    if not isinstance(controls, dict):
        controls = {}
    noisy_tv_at_chance = controls.get("noisy_tv_at_chance") is True

    independent_scientific_confirmation = (
        independent_referee_reproduction
        and source_kind == "real"
        and rights_clean
        and noisy_tv_at_chance
        and reproductions >= MIN_REPRODUCTIONS
    )

    return CountVerificationResult(
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
            "recomputed_deltas": recomputed_deltas,
            "recomputed_t_obs": t_obs,
            "recomputed_one_sided_p": one_sided_p,
            "recomputed_mean_delta_control_minus_candidate": mean_delta,
            "n_permutations": n_perm,
            "sesoi_mae": sesoi,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "budget_matched": budget_matched,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def count_verification_payload(result: CountVerificationResult) -> dict:

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
