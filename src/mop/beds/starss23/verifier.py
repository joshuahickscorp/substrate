
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-escs-bed-verification/v1"

# Re-declared here rather than imported, so the verifier shares no symbol with the producer.
EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-bed/v1"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
EXPECTED_CANDIDATE_ARM = "candidate"

# The promotion bar from the deep-research recipe: at least three bias-independent reproductions.
MIN_REPRODUCTIONS = 3

# Claim ceiling: with the clip as the experimental unit only these verbs are honest.
ALLOWED_CLAIM_VERBS = ("consistent with", "suggestive")
FORBIDDEN_CLAIM_VERBS = (
    "demonstrates",
    "demonstrate",
    "significant",
    "proves",
    "prove",
    "establishes",
    "confirms",
)

_FLOAT_TOL = 1e-9


class VerificationRefusal(ValueError):
    pass


# ---------------------------------------------------------------------------
# Canonical seal, re-implemented from specification.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Referee, re-implemented from specification (no shared code with referee.py).
# ---------------------------------------------------------------------------


def _clean_frames(frames: object, label: str) -> list[int]:
    if not isinstance(frames, (list, tuple)):
        raise VerificationRefusal(f"{label} must be a list of integer frames")
    out: list[int] = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise VerificationRefusal(f"{label} must contain only nonnegative integer frames")
        out.append(frame)
    out.sort()
    return out


def _match_counts(gt_frames: object, pred_frames: object, collar: int) -> tuple[int, int, int]:

    gt = _clean_frames(gt_frames, "gt_onsets")
    pred = _clean_frames(pred_frames, "fires")

    pairs: list[tuple[int, int, int, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            gap = p - g if p >= g else g - p
            if gap <= collar:
                pairs.append((gap, g, p, gi, pi))
    pairs.sort()

    gt_taken = [False] * len(gt)
    pred_taken = [False] * len(pred)
    tp = 0
    for _gap, _g, _p, gi, pi in pairs:
        if gt_taken[gi] or pred_taken[pi]:
            continue
        gt_taken[gi] = True
        pred_taken[pi] = True
        tp += 1
    fn = len(gt) - tp
    fp = len(pred) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    return precision, recall, f1


def _pool_arm(clips: list, arm: str, collar: int) -> dict:

    tp = fp = fn = 0
    for clip in clips:
        fires = clip.get("fires", {})
        if arm not in fires:
            raise VerificationRefusal(f"clip {clip.get('clip_id')!r} is missing fires for arm {arm!r}")
        clip_tp, clip_fp, clip_fn = _match_counts(clip.get("gt_onsets"), fires[arm], collar)
        tp += clip_tp
        fp += clip_fp
        fn += clip_fn
    precision, recall, f1 = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Exact sign-flip permutation, re-implemented from specification.
# ---------------------------------------------------------------------------


def _sign_flip_one_sided_p(deltas: list) -> tuple[float, float, int]:

    n = len(deltas)
    if n == 0:
        raise VerificationRefusal("sign-flip test needs at least one paired delta")
    t_obs = sum(deltas) / n
    at_least = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        t = sum(s * d for s, d in zip(signs, deltas, strict=True)) / n
        total += 1
        if t >= t_obs - _FLOAT_TOL:
            at_least += 1
    return t_obs, at_least / total, total


# ---------------------------------------------------------------------------
# Verification result and the top-level checks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationResult:

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


def _floats_agree(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= _FLOAT_TOL


def verify_artifact(artifact: dict) -> VerificationResult:

    if not isinstance(artifact, dict):
        raise VerificationRefusal("artifact must be a JSON object")
    mismatches: list[str] = []

    # 1. Seal integrity. The seal covers everything except itself; a re-hash must reproduce it.
    stored_seal = artifact.get("seal")
    body = {k: v for k, v in artifact.items() if k != "seal"}
    recomputed_seal = _canonical_sha256(body)
    seal_intact = isinstance(stored_seal, str) and stored_seal == recomputed_seal
    if not seal_intact:
        mismatches.append("seal does not match a re-hash of the artifact body")

    # 2. Schema, stage, and claim-scope must be exactly the frozen bed contract, never widened.
    schema_ok = True
    if artifact.get("schema") != EXPECTED_ARTIFACT_SCHEMA:
        schema_ok = False
        mismatches.append(f"schema is {artifact.get('schema')!r}, expected {EXPECTED_ARTIFACT_SCHEMA!r}")
    if artifact.get("stage") != EXPECTED_STAGE:
        schema_ok = False
        mismatches.append(f"stage is {artifact.get('stage')!r}, expected {EXPECTED_STAGE}")
    if artifact.get("claim_scope") != EXPECTED_CLAIM_SCOPE:
        schema_ok = False
        mismatches.append("claim_scope was widened away from the frozen bed contract")

    collar = artifact.get("collar_frames")
    if isinstance(collar, bool) or not isinstance(collar, int) or collar < 0:
        raise VerificationRefusal("artifact.collar_frames must be a nonnegative integer")
    primary_control = artifact.get("primary_control")
    if not isinstance(primary_control, str) or not primary_control:
        raise VerificationRefusal("artifact.primary_control must name the headline control arm")

    per_seed = artifact.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        raise VerificationRefusal("artifact.per_seed must be a nonempty list")

    # 3. Re-score every arm of every seed from the raw fires, and re-derive the paired deltas.
    scores_reproduced = True
    recomputed_deltas: list[float] = []
    for seed_block in per_seed:
        clips = seed_block.get("clips")
        arm_scores = seed_block.get("arm_scores")
        if not isinstance(clips, list) or not isinstance(arm_scores, dict):
            raise VerificationRefusal("each per_seed entry needs clips and arm_scores")
        recomputed_by_arm: dict[str, dict] = {}
        for arm, claimed in arm_scores.items():
            recomputed = _pool_arm(clips, arm, collar)
            recomputed_by_arm[arm] = recomputed
            for key in ("tp", "fp", "fn"):
                if claimed.get(key) != recomputed[key]:
                    scores_reproduced = False
                    mismatches.append(
                        f"seed {seed_block.get('seed')} arm {arm} {key} claimed "
                        f"{claimed.get(key)} recomputed {recomputed[key]}"
                    )
            for key in ("precision", "recall", "f1"):
                if not _floats_agree(claimed.get(key), recomputed[key]):
                    scores_reproduced = False
                    mismatches.append(
                        f"seed {seed_block.get('seed')} arm {arm} {key} claimed "
                        f"{claimed.get(key)} recomputed {recomputed[key]}"
                    )
        if EXPECTED_CANDIDATE_ARM not in recomputed_by_arm:
            raise VerificationRefusal("per_seed entry has no candidate arm to score")
        if primary_control not in recomputed_by_arm:
            raise VerificationRefusal(f"per_seed entry has no {primary_control} control arm to score")
        delta = recomputed_by_arm[EXPECTED_CANDIDATE_ARM]["f1"] - recomputed_by_arm[primary_control]["f1"]
        recomputed_deltas.append(delta)

    # 4. Re-run the exact sign-flip test on the re-derived deltas.
    stats = artifact.get("stats")
    if not isinstance(stats, dict):
        raise VerificationRefusal("artifact.stats must be present")
    stats_reproduced = True
    claimed_deltas = stats.get("deltas")
    if not isinstance(claimed_deltas, list) or len(claimed_deltas) != len(recomputed_deltas):
        stats_reproduced = False
        mismatches.append("stats.deltas is missing or the wrong length")
    else:
        pairs = zip(claimed_deltas, recomputed_deltas, strict=True)
        for i, (claimed_d, recomputed_d) in enumerate(pairs):
            if not _floats_agree(claimed_d, recomputed_d):
                stats_reproduced = False
                mismatches.append(
                    f"stats delta {i} claimed {claimed_d} recomputed {recomputed_d}"
                )
    t_obs, one_sided_p, n_perm = _sign_flip_one_sided_p(recomputed_deltas)
    if not _floats_agree(stats.get("t_obs"), t_obs):
        stats_reproduced = False
        mismatches.append(f"stats.t_obs claimed {stats.get('t_obs')} recomputed {t_obs}")
    if not _floats_agree(stats.get("one_sided_p"), one_sided_p):
        stats_reproduced = False
        mismatches.append(f"stats.one_sided_p claimed {stats.get('one_sided_p')} recomputed {one_sided_p}")
    if stats.get("n_permutations") != n_perm:
        stats_reproduced = False
        mismatches.append(f"stats.n_permutations claimed {stats.get('n_permutations')} recomputed {n_perm}")
    two_sided_floor = 2.0 / n_perm
    expected_reachable = two_sided_floor <= 0.05
    if bool(stats.get("two_sided_005_reachable")) != expected_reachable:
        stats_reproduced = False
        mismatches.append("stats.two_sided_005_reachable disagrees with the exact discrete floor")

    # 5. Honesty flags. The producer never self-certifies and never asserts activation or promotion.
    honesty_ok = True
    flags = artifact.get("flags")
    if not isinstance(flags, dict):
        raise VerificationRefusal("artifact.flags must be present")
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
    controls = artifact.get("controls") if isinstance(artifact.get("controls"), dict) else {}
    noisy_tv_at_chance = controls.get("noisy_tv_at_chance") is True

    independent_scientific_confirmation = (
        independent_referee_reproduction
        and source_kind == "real"
        and rights_clean
        and noisy_tv_at_chance
        and reproductions >= MIN_REPRODUCTIONS
    )

    return VerificationResult(
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
            "n_permutations": n_perm,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def verification_payload(result: VerificationResult) -> dict:

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


def verify_sealed_file(in_path: str) -> dict:

    with open(in_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return verification_payload(verify_artifact(artifact))


def write_verification(payload: dict, out_path: str) -> None:

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
