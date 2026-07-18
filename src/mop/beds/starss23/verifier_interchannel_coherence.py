"""Independent verifier for the STARSS23 ESCS interchannel-coherence featurizer-swap artifact.

This module is authored separately from the producer and imports NONE of the producer's code and nothing
under ``mop``. Its whole purpose is to re-derive every scored number in the sealed featurizer-swap
artifact from specification, using only the standard library, so that agreement between the two is real
triangulation and not a shared bug. The committed ``verifier.py`` is schema-locked to the base
``mop-starss23-escs-bed/v1`` artifact and refuses the featurizer-swap schema, so this net-new verifier
covers the ``mop-starss23-escs-bed-interchannel-coherence/v1`` schema with the SAME import discipline:
only ``json``, ``hashlib``, ``itertools``, and ``dataclasses``.

What it re-implements from the written specification
----------------------------------------------------
1. Canonical JSON sealing: ``json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
   allow_nan=False)`` hashed with SHA-256. The seal covers the whole artifact except the ``seal`` key.
2. The onset-F1 referee: greedy one-to-one nearest-first matching within the plus or minus two frame
   collar, strict point-wise precision, recall, and F1 pooled across the test clips.
3. The exact sign-flip permutation test: enumerate all two-to-the-n sign assignments of the paired
   per-seed deltas; one-sided p is the fraction whose mean reaches the observed mean. At n = 5 the
   minimum one-sided p is 1/32 and two-sided 0.05 is unreachable.
4. The fire-spread diagnostics: the pooled adjacency fraction and the pooled distinct-onset true
   positives per arm at the operating point, so the mechanistic wall (the candidate clusters and recovers
   fewer distinct onsets than free random) is re-derived, never merely trusted.

What it checks about the matched budget and the frozen featurizer
-----------------------------------------------------------------
5. Every arm's stored full-lifecycle FLOPs stay under the 6e10 ceiling, and the candidate and the
   rate-matched-random control have byte-equal inference (featurize + gate + firing) FLOPs at every
   budget, so the comparison is matched ex-training.
6. The featurizer carries zero trained parameters and the featurize charge in the ledger equals its
   sealed per-frame FLOPs times the test frame count, so the new front-end is priced honestly.

The verifier re-scores from the rawest data the artifact carries (the per-clip ground-truth onsets and
the per-arm fire frames), so tampering with a stored score, a fire list, a FLOP figure, or the seal is
detected. Because the run is a single real run at n = 5 across a three-featurizer family, scientific
confirmation stays false by construction: mechanics reproduce, science does not promote.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

VERIFIER_SCHEMA = "mop-starss23-escs-bed-interchannel-coherence-verification/v1"

# Re-declared here rather than imported, so the verifier shares no symbol with the producer.
EXPECTED_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-interchannel-coherence/v1"
EXPECTED_STAGE = 3
EXPECTED_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
EXPECTED_CANDIDATE_ARM = "candidate"
EXPECTED_VARIANT_ID = "interchannel_coherence"

FLOP_CEILING = 60_000_000_000
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
)

_FLOAT_TOL = 1e-9
_COLLAR = 2


class VerificationRefusal(ValueError):
    """Raised when an artifact is too malformed to even attempt independent re-scoring."""


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
    """Independent greedy one-to-one nearest-first matcher. Returns ``(tp, fp, fn)``."""

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
    """Recompute an arm's pooled score across clips from raw ground truth and fires."""

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


def _adjacency_fraction(clip_fire_lists: list, collar: int) -> float:
    """Pooled fraction of fires within ``collar`` frames of another fire on the same clip."""

    total = 0
    adjacent = 0
    for fires in clip_fire_lists:
        ordered = sorted(fires)
        total += len(ordered)
        for index, frame in enumerate(ordered):
            near_prev = index > 0 and frame - ordered[index - 1] <= collar
            near_next = index < len(ordered) - 1 and ordered[index + 1] - frame <= collar
            if near_prev or near_next:
                adjacent += 1
    return adjacent / total if total > 0 else 0.0


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
    """The outcome of an independent verification pass over one sealed featurizer-swap artifact."""

    seal_intact: bool
    schema_ok: bool
    scores_reproduced: bool
    stats_reproduced: bool
    spread_reproduced: bool
    budget_ok: bool
    featurizer_ok: bool
    honesty_ok: bool
    independent_referee_reproduction: bool
    independent_scientific_confirmation: bool
    reproduced_verdict: str
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
    """Independently re-score a sealed featurizer-swap artifact and decide referee reproduction."""

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

    # 2. Schema, stage, claim scope, and featurizer identity must be the frozen contract, never widened.
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
    if artifact.get("variant_id") != EXPECTED_VARIANT_ID:
        schema_ok = False
        mismatches.append(f"variant_id is {artifact.get('variant_id')!r}, expected {EXPECTED_VARIANT_ID!r}")

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
    spread_reproduced = True
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
        for i, (claimed_d, recomputed_d) in enumerate(zip(claimed_deltas, recomputed_deltas, strict=True)):
            if not _floats_agree(claimed_d, recomputed_d):
                stats_reproduced = False
                mismatches.append(f"stats delta {i} claimed {claimed_d} recomputed {recomputed_d}")
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

    # 5. Re-derive the fire-spread diagnostics per arm at the operating point.
    spread = artifact.get("variant", {}).get("fire_spread_diagnostic", {})
    for arm in (EXPECTED_CANDIDATE_ARM, primary_control):
        claimed = spread.get(arm if arm != primary_control else "rate_matched_random")
        recomputed_tp: list[int] = []
        recomputed_adj: list[float] = []
        recomputed_fires: list[int] = []
        for seed_block in per_seed:
            fire_lists = [list(clip["fires"][arm]) for clip in seed_block["clips"]]
            pairs = [(list(clip["gt_onsets"]), list(clip["fires"][arm])) for clip in seed_block["clips"]]
            tp = sum(_match_counts(gt, fr, collar)[0] for gt, fr in pairs)
            recomputed_tp.append(tp)
            recomputed_adj.append(_adjacency_fraction(fire_lists, collar))
            recomputed_fires.append(sum(len(f) for f in fire_lists))
        if not isinstance(claimed, dict):
            spread_reproduced = False
            mismatches.append(f"fire_spread_diagnostic missing arm {arm}")
            continue
        if claimed.get("per_seed_distinct_onset_tp") != recomputed_tp:
            spread_reproduced = False
            mismatches.append(f"spread {arm} distinct_onset_tp claimed {claimed.get('per_seed_distinct_onset_tp')} recomputed {recomputed_tp}")
        if claimed.get("per_seed_fires") != recomputed_fires:
            spread_reproduced = False
            mismatches.append(f"spread {arm} fires claimed {claimed.get('per_seed_fires')} recomputed {recomputed_fires}")
        claimed_adj = claimed.get("per_seed_adjacency_fraction") or []
        if len(claimed_adj) != len(recomputed_adj) or not all(
            _floats_agree(a, b) for a, b in zip(claimed_adj, recomputed_adj, strict=False)
        ):
            spread_reproduced = False
            mismatches.append(f"spread {arm} adjacency_fraction disagrees")

    # 6. Matched budget: every arm under the ceiling, candidate and rate-matched-random byte-equal ex-training.
    budget_ok = True
    harness = artifact.get("harness")
    if not isinstance(harness, dict):
        raise VerificationRefusal("artifact.harness must be present")
    arm_summaries = harness.get("arm_summaries", [])
    for summary in arm_summaries:
        max_life = summary.get("max_lifecycle_flops")
        if not isinstance(max_life, int) or max_life > FLOP_CEILING:
            budget_ok = False
            mismatches.append(f"arm {summary.get('name')!r} max lifecycle {max_life} exceeds ceiling {FLOP_CEILING}")
    per_budget_rows = harness.get("per_budget_candidate_vs_rate_matched_random", [])
    by_budget: dict[str, dict] = {}
    for summary in arm_summaries:
        name = summary.get("name", "")
        if "@" in name:
            kind, budget = name.split("@", 1)
            by_budget.setdefault(budget, {})[kind] = summary
    for budget, kinds in by_budget.items():
        cand = kinds.get("candidate")
        rmr = kinds.get("rate_matched_random")
        if not cand or not rmr:
            continue
        fm_c = cand["flop_model"]
        fm_r = rmr["flop_model"]
        # inference (ex-training) run FLOPs per fired frame: featurize + gate infer + K * C_down, K equal.
        c_fires = cand["seed_results"][0]["firings"]
        r_fires = rmr["seed_results"][0]["firings"]
        if c_fires != r_fires:
            budget_ok = False
            mismatches.append(f"budget {budget}: candidate firings {c_fires} != rate_matched_random {r_fires}")
        run_c = fm_c["featurize_flops"] + fm_c["gate_infer_flops"] + c_fires * fm_c["downstream_flops_per_firing"]
        run_r = fm_r["featurize_flops"] + fm_r["gate_infer_flops"] + r_fires * fm_r["downstream_flops_per_firing"]
        if run_c != run_r:
            budget_ok = False
            mismatches.append(f"budget {budget}: candidate run FLOPs {run_c} != rate_matched_random {run_r}")
        if fm_c["train_flops"] <= 0:
            budget_ok = False
            mismatches.append(f"budget {budget}: candidate must charge a positive training cost")
        if fm_r["train_flops"] != 0:
            budget_ok = False
            mismatches.append(f"budget {budget}: control must charge zero training cost")

    # 7. Featurizer: zero trained parameters and honest featurize charge in the ledger.
    featurizer_ok = True
    featurizer = artifact.get("featurizer", {})
    if featurizer.get("n_params") != 0:
        featurizer_ok = False
        mismatches.append(f"featurizer n_params is {featurizer.get('n_params')!r}, must be 0 (zero trained parameters)")
    fpf = featurizer.get("flops_per_frame")
    n_test_frames = artifact.get("real_corpus", {}).get("n_test_frames")
    if isinstance(fpf, int) and isinstance(n_test_frames, int) and arm_summaries:
        expected_featurize = fpf * n_test_frames
        for summary in arm_summaries:
            got = summary["flop_model"]["featurize_flops"]
            if got != expected_featurize:
                featurizer_ok = False
                mismatches.append(
                    f"arm {summary.get('name')!r} featurize charge {got} != flops_per_frame*n_test_frames {expected_featurize}"
                )
                break

    # 8. Honesty flags. The producer never self-certifies and never asserts activation or promotion.
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

    # The independently reproduced verdict, from the re-derived numbers alone.
    sesoi = stats.get("sesoi_f1")
    exceeds_sesoi = isinstance(sesoi, (int, float)) and t_obs >= float(sesoi)
    dominates = bool(harness.get("candidate_strictly_dominates_rate_matched_random"))
    one_sided_significant = one_sided_p <= 0.05
    reproduced_verdict = (
        "mechanics-ok" if (dominates and one_sided_significant and exceeds_sesoi) else "null"
    )
    if artifact.get("verdict") != reproduced_verdict:
        mismatches.append(
            f"stored verdict {artifact.get('verdict')!r} disagrees with reproduced {reproduced_verdict!r}"
        )

    independent_referee_reproduction = (
        seal_intact
        and schema_ok
        and scores_reproduced
        and stats_reproduced
        and spread_reproduced
        and budget_ok
        and featurizer_ok
        and honesty_ok
        and artifact.get("verdict") == reproduced_verdict
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
        and reproduced_verdict == "mechanics-ok"
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
        spread_reproduced=spread_reproduced,
        budget_ok=budget_ok,
        featurizer_ok=featurizer_ok,
        honesty_ok=honesty_ok,
        independent_referee_reproduction=independent_referee_reproduction,
        independent_scientific_confirmation=independent_scientific_confirmation,
        reproduced_verdict=reproduced_verdict,
        source_kind=source_kind,
        rights_clean=rights_clean,
        reproductions=reproductions,
        mismatches=tuple(mismatches),
        detail={
            "recomputed_deltas": recomputed_deltas,
            "recomputed_t_obs": t_obs,
            "recomputed_one_sided_p": one_sided_p,
            "n_permutations": n_perm,
            "reproduced_mean_delta_exceeds_sesoi": exceeds_sesoi,
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "min_reproductions": MIN_REPRODUCTIONS,
        },
    )


def verification_payload(result: VerificationResult) -> dict:
    """Assemble the sealed verification body from a result."""

    body = {
        "schema": VERIFIER_SCHEMA,
        "seal_intact": result.seal_intact,
        "schema_ok": result.schema_ok,
        "scores_reproduced": result.scores_reproduced,
        "stats_reproduced": result.stats_reproduced,
        "spread_reproduced": result.spread_reproduced,
        "budget_ok": result.budget_ok,
        "featurizer_ok": result.featurizer_ok,
        "honesty_ok": result.honesty_ok,
        "independent_referee_reproduction": result.independent_referee_reproduction,
        "independent_scientific_confirmation": result.independent_scientific_confirmation,
        "reproduced_verdict": result.reproduced_verdict,
        "source_kind": result.source_kind,
        "rights_clean": result.rights_clean,
        "reproductions": result.reproductions,
        "mismatches": list(result.mismatches),
        "detail": result.detail,
    }
    body["seal"] = _canonical_sha256(body)
    return body


def verify_sealed_file(in_path: str) -> dict:
    """Read a sealed artifact from disk, verify it, and return the sealed verification payload."""

    with open(in_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return verification_payload(verify_artifact(artifact))


def write_verification(payload: dict, out_path: str) -> None:
    """Write a verification payload as canonical JSON."""

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Independently verify a sealed STARSS23 interchannel-coherence artifact."
    )
    parser.add_argument("in_path", help="sealed artifact JSON path")
    parser.add_argument("--out", default=None, help="optional path to write the sealed verification JSON")
    args = parser.parse_args(argv)

    payload = verify_sealed_file(args.in_path)
    if args.out:
        write_verification(payload, args.out)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("detail", "seal")}, indent=2, sort_keys=True))
    print("detail:", json.dumps(payload["detail"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
