
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-count-repro-data-split-verification/v1"

EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-data-split/v1"
EXPECTED_AXIS = "data_split"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

CANDIDATE = "candidate"
RATE_MATCHED_RANDOM = "rate_matched_random"
ALWAYS_ON = "always_on"
NEVER_UPDATE = "never_update"
COLD_START = 0

MIN_REPRODUCTIONS = 3

ALLOWED_CLAIM_VERBS = ("consistent with", "suggestive")
FORBIDDEN_CLAIM_VERBS = (
    "demonstrates",
    "demonstrate",
    "shows",
    "significant",
    "proves",
    "prove",
    "establishes",
    "confirms",
    "confirmed",
)

_TOL = 1e-9


class ReproVerificationRefusal(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _close(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= _TOL


def _count_track(track: object, label: str) -> list[int]:
    if not isinstance(track, (list, tuple)):
        raise ReproVerificationRefusal(f"{label} must be a list of nonnegative integers")
    out: list[int] = []
    for item in track:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ReproVerificationRefusal(f"{label} must contain only nonnegative integers")
        out.append(item)
    return out


def _reestimates(frames: object, n_frames: int, label: str) -> list[int]:
    if not isinstance(frames, (list, tuple)):
        raise ReproVerificationRefusal(f"{label} must be a list of frame indices")
    out: list[int] = []
    previous = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise ReproVerificationRefusal(f"{label} indices must lie in [0, {n_frames})")
        if frame <= previous:
            raise ReproVerificationRefusal(f"{label} must be strictly sorted and unique")
        previous = frame
        out.append(frame)
    return out


def _coast_abs_error(gt: list[int], estimator: list[int], reestimates: list[int]) -> int:

    fire = set(reestimates)
    held = COLD_START
    total = 0
    for t in range(len(estimator)):
        if t in fire:
            held = estimator[t]
        total += abs(held - gt[t])
    return total


def _arm_reestimates(arm: str, clip_id: str, n_frames: int, stored_by_clip: dict) -> list[int]:
    if arm == ALWAYS_ON:
        return list(range(n_frames))
    if arm == NEVER_UPDATE:
        return []
    stored = stored_by_clip.get(clip_id, {})
    if not isinstance(stored, dict):
        raise ReproVerificationRefusal(f"clip {clip_id!r} reestimate_frames must be an object")
    return _reestimates(stored.get(arm), n_frames, f"{arm} reestimate_frames on {clip_id}")


def _sign_flip(deltas: list[float]) -> tuple[float, float, int]:
    n = len(deltas)
    if n == 0:
        raise ReproVerificationRefusal("the sign-flip test needs at least one paired delta")
    observed = sum(deltas) / n
    at_least = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        flipped = sum(s * d for s, d in zip(signs, deltas, strict=True)) / n
        if flipped >= observed - _TOL:
            at_least += 1
    return observed, at_least / total, total


@dataclass(frozen=True, slots=True)
class ReproVerificationResult:

    seal_intact: bool
    schema_ok: bool
    split_room_disjoint: bool
    scores_reproduced: bool
    stats_reproduced: bool
    honesty_ok: bool
    independent_referee_reproduction: bool
    independent_scientific_confirmation: bool
    source_kind: str
    rights_clean: bool
    reproductions: int
    survives: bool
    mismatches: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.independent_referee_reproduction


def _check_split_disjoint(artifact: dict, mismatches: list[str]) -> bool:
    real = artifact.get("real_corpus")
    if not isinstance(real, dict):
        mismatches.append("real_corpus block is missing")
        return False
    rooms = real.get("split_rooms")
    if not isinstance(rooms, dict):
        mismatches.append("real_corpus.split_rooms is missing")
        return False
    train = set(rooms.get("train_rooms", []) or [])
    val = set(rooms.get("val_rooms", []) or [])
    test = set(rooms.get("test_rooms", []) or [])
    if not train or not val or not test:
        mismatches.append("a swapped-split partition has no rooms")
        return False
    ok = True
    if train & val or train & test or val & test:
        ok = False
        mismatches.append("the swapped split is not room-disjoint across train, val, and test")
    if rooms.get("swapped_from_sealed") is not True:
        ok = False
        mismatches.append("the split is not marked as swapped from the sealed bed")
    return ok


def _score_pooled(
    clip_ids: list[str],
    corpus: dict,
    stored_by_clip: dict,
    arm: str,
    candidate_count: dict,
    mismatches: list[str],
) -> tuple[int, int, bool]:
    abs_error = 0
    frames = 0
    budget_ok = True
    for clip_id in clip_ids:
        block = corpus.get(clip_id)
        if not isinstance(block, dict):
            raise ReproVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
        gt = _count_track(block.get("gt_count_track"), "gt_count_track")
        estimator = _count_track(block.get("estimator_track"), "estimator_track")
        if len(gt) != len(estimator):
            raise ReproVerificationRefusal(f"clip {clip_id!r} gt and estimator tracks differ in length")
        n = len(estimator)
        reest = _arm_reestimates(arm, clip_id, n, stored_by_clip)
        if arm == RATE_MATCHED_RANDOM and len(reest) != candidate_count.get(clip_id):
            budget_ok = False
            mismatches.append(
                f"clip {clip_id} rate_matched_random spends {len(reest)} re-estimations but the candidate "
                f"spends {candidate_count.get(clip_id)}: budget not matched"
            )
        abs_error += _coast_abs_error(gt, estimator, reest)
        frames += n
    return abs_error, frames, budget_ok


def verify_data_split_artifact(artifact: dict) -> ReproVerificationResult:

    if not isinstance(artifact, dict):
        raise ReproVerificationRefusal("artifact must be a JSON object")
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
    if artifact.get("reproduction_axis") != EXPECTED_AXIS:
        schema_ok = False
        mismatches.append(f"reproduction_axis is {artifact.get('reproduction_axis')!r}, expected {EXPECTED_AXIS!r}")
    if artifact.get("stage") != EXPECTED_STAGE:
        schema_ok = False
        mismatches.append(f"stage is {artifact.get('stage')!r}, expected {EXPECTED_STAGE}")
    if artifact.get("claim_scope") != EXPECTED_CLAIM_SCOPE:
        schema_ok = False
        mismatches.append("claim_scope was widened away from the frozen contract")
    if artifact.get("primary_control") != RATE_MATCHED_RANDOM:
        raise ReproVerificationRefusal("artifact.primary_control must be rate_matched_random")

    split_room_disjoint = _check_split_disjoint(artifact, mismatches)

    corpus = artifact.get("corpus_tracks")
    if not isinstance(corpus, dict) or not corpus:
        raise ReproVerificationRefusal("artifact.corpus_tracks must be a nonempty object")
    per_seed = artifact.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        raise ReproVerificationRefusal("artifact.per_seed must be a nonempty list")

    scores_reproduced = True
    budget_matched = True
    recomputed_deltas: list[float] = []
    for seed_block in per_seed:
        clips = seed_block.get("clips")
        arm_scores = seed_block.get("arm_scores")
        if not isinstance(clips, list) or not clips or not isinstance(arm_scores, dict):
            raise ReproVerificationRefusal("each per_seed entry needs a nonempty clips list and arm_scores")

        clip_ids: list[str] = []
        stored_by_clip: dict = {}
        candidate_count: dict = {}
        for clip in clips:
            clip_id = clip.get("clip_id")
            if not isinstance(clip_id, str):
                raise ReproVerificationRefusal("each per_seed clip needs a string clip_id")
            block = corpus.get(clip_id)
            if not isinstance(block, dict):
                raise ReproVerificationRefusal(f"corpus_tracks is missing clip {clip_id!r}")
            n = len(_count_track(block.get("estimator_track"), "estimator_track"))
            clip_ids.append(clip_id)
            reest_frames = clip.get("reestimate_frames", {})
            stored_by_clip[clip_id] = reest_frames
            candidate_count[clip_id] = len(
                _reestimates(
                    reest_frames.get(CANDIDATE) if isinstance(reest_frames, dict) else None,
                    n,
                    f"candidate reestimate_frames on {clip_id}",
                )
            )

        recomputed_mae: dict = {}
        for arm in (CANDIDATE, RATE_MATCHED_RANDOM, ALWAYS_ON, NEVER_UPDATE):
            abs_error, frames, budget_ok = _score_pooled(
                clip_ids, corpus, stored_by_clip, arm, candidate_count, mismatches
            )
            budget_matched = budget_matched and budget_ok
            if frames == 0:
                raise ReproVerificationRefusal("a seed scored zero frames")
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
                    f"seed {seed} arm {arm} abs_error_sum stored {claimed.get('abs_error_sum')} recomputed {abs_error}"
                )
            if claimed.get("n_frames") != frames:
                scores_reproduced = False
                mismatches.append(
                    f"seed {seed} arm {arm} n_frames stored {claimed.get('n_frames')} recomputed {frames}"
                )
            if not _close(claimed.get("mae"), mae):
                scores_reproduced = False
                mismatches.append(f"seed {seed} arm {arm} mae stored {claimed.get('mae')} recomputed {mae}")

        recomputed_deltas.append(recomputed_mae[RATE_MATCHED_RANDOM] - recomputed_mae[CANDIDATE])

    if not budget_matched:
        scores_reproduced = False

    stats = artifact.get("stats")
    if not isinstance(stats, dict):
        raise ReproVerificationRefusal("artifact.stats must be present")
    stats_reproduced = True
    claimed_deltas = stats.get("deltas")
    if not isinstance(claimed_deltas, list) or len(claimed_deltas) != len(recomputed_deltas):
        stats_reproduced = False
        mismatches.append("stats.deltas is missing or the wrong length")
    else:
        for i, (claimed_d, mine_d) in enumerate(zip(claimed_deltas, recomputed_deltas, strict=True)):
            if not _close(claimed_d, mine_d):
                stats_reproduced = False
                mismatches.append(f"stats delta {i} stored {claimed_d} recomputed {mine_d}")

    t_obs, one_sided_p, n_perm = _sign_flip(recomputed_deltas)
    if not _close(stats.get("t_obs"), t_obs):
        stats_reproduced = False
        mismatches.append(f"stats.t_obs stored {stats.get('t_obs')} recomputed {t_obs}")
    if not _close(stats.get("one_sided_p"), one_sided_p):
        stats_reproduced = False
        mismatches.append(f"stats.one_sided_p stored {stats.get('one_sided_p')} recomputed {one_sided_p}")
    if stats.get("n_permutations") != n_perm:
        stats_reproduced = False
        mismatches.append(f"stats.n_permutations stored {stats.get('n_permutations')} recomputed {n_perm}")
    if bool(stats.get("two_sided_005_reachable")) != ((2.0 / n_perm) <= 0.05):
        stats_reproduced = False
        mismatches.append("stats.two_sided_005_reachable disagrees with the exact discrete floor")

    mean_delta = t_obs
    sesoi = None
    prereg = artifact.get("prereg")
    if isinstance(prereg, dict):
        sesoi = prereg.get("sesoi_mae")
    if sesoi is None:
        sesoi = stats.get("sesoi_mae")
    sesoi_exceeded = False
    if isinstance(sesoi, (int, float)) and not isinstance(sesoi, bool):
        sesoi_exceeded = mean_delta >= float(sesoi)
        if bool(stats.get("mean_delta_exceeds_sesoi")) != sesoi_exceeded:
            stats_reproduced = False
            mismatches.append("stats.mean_delta_exceeds_sesoi disagrees with the recomputed mean delta")

    honesty_ok = True
    flags = artifact.get("flags")
    if not isinstance(flags, dict):
        raise ReproVerificationRefusal("artifact.flags must be present")
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
        and split_room_disjoint
        and scores_reproduced
        and stats_reproduced
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

    candidate_strictly_lower = mean_delta > _TOL
    sign_flip_clears = one_sided_p <= (1.0 / n_perm) + _TOL
    survives = bool(
        independent_referee_reproduction
        and candidate_strictly_lower
        and sesoi_exceeded
        and sign_flip_clears
    )

    return ReproVerificationResult(
        seal_intact=seal_intact,
        schema_ok=schema_ok,
        split_room_disjoint=split_room_disjoint,
        scores_reproduced=scores_reproduced,
        stats_reproduced=stats_reproduced,
        honesty_ok=honesty_ok,
        independent_referee_reproduction=independent_referee_reproduction,
        independent_scientific_confirmation=independent_scientific_confirmation,
        source_kind=source_kind,
        rights_clean=rights_clean,
        reproductions=reproductions,
        survives=survives,
        mismatches=tuple(mismatches),
        detail={
            "recomputed_deltas": recomputed_deltas,
            "recomputed_t_obs": t_obs,
            "recomputed_one_sided_p": one_sided_p,
            "recomputed_mean_delta_control_minus_candidate": mean_delta,
            "recomputed_mean_delta_candidate_minus_control": -mean_delta,
            "n_permutations": n_perm,
            "sesoi_mae": sesoi,
            "sesoi_exceeded": sesoi_exceeded,
            "candidate_strictly_lower_mae": candidate_strictly_lower,
            "sign_flip_clears_floor": sign_flip_clears,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "budget_matched": budget_matched,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def data_split_verification_payload(result: ReproVerificationResult) -> dict:

    body = {
        "schema": VERIFIER_SCHEMA,
        "reproduction_axis": EXPECTED_AXIS,
        "seal_intact": result.seal_intact,
        "schema_ok": result.schema_ok,
        "split_room_disjoint": result.split_room_disjoint,
        "scores_reproduced": result.scores_reproduced,
        "stats_reproduced": result.stats_reproduced,
        "honesty_ok": result.honesty_ok,
        "independent_referee_reproduction": result.independent_referee_reproduction,
        "independent_scientific_confirmation": result.independent_scientific_confirmation,
        "source_kind": result.source_kind,
        "rights_clean": result.rights_clean,
        "reproductions": result.reproductions,
        "survives": result.survives,
        "mismatches": list(result.mismatches),
        "detail": result.detail,
    }
    body["seal"] = _canonical_sha256(body)
    return body


def verify_sealed_data_split_file(in_path: str) -> dict:

    with open(in_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return data_split_verification_payload(verify_data_split_artifact(artifact))


def write_data_split_verification(payload: dict, out_path: str) -> None:

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
