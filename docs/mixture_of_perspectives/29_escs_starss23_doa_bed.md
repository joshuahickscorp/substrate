# ESCS STARSS23 Direction-of-Arrival Bed and Its First Real Result

> **Append-only Stage-3 experimental lane.** This is the third real question run on the STARSS23 ESCS
> program, after onset-localization (walled, [doc 26](./26_escs_starss23_bed.md), 7 nulls) and
> source-counting (a signal that did not survive bias-independent reproduction,
> [doc 27](./27_escs_starss23_counting_bed.md) / [doc 28](./28_escs_starss23_counting_reproductions.md)).
> It is net-new only: ten `doa_*` modules under `src/mop/beds/starss23/` plus the sealed
> `proof/STARSS23_DOA_BED.json`, `proof/STARSS23_DOA_BED.prereg.json`, and
> `proof/STARSS23_DOA_BED.verification.json`. It edits no sealed onset module (`referee.py` `stats.py`
> `controls.py` `harness.py` `gate.py` `prereg.py`), no sealed `count_*` module, and no existing proof;
> every one of those 16 files is byte-identical to the committed HEAD. It touches no live campaign or
> process file.

- **Status:** built and run once on real MIT STARSS23, under two independent trained-gate architectures
  in the same run. **Verified null on both architectures.** Not architecture-fragile: both axes fail
  the same way.
- **Snapshot date:** 2026-07-18
- **Dataset:** STARSS23 (zenodo.org/records/7880637), MIT license, commercial-use OK; the same fixed
  real 4-channel FOA subset and native room-disjoint fold split the counting bed used (fold-4 dev-test
  is the test split; val is carved from the tail of fold-3 rooms; train is the rest of fold-3).
- **Claim scope:** deterministic matched-budget adaptive-compute mechanics on a real rights-clean bed;
  no activation, no scientific promotion, no natural-world generality claim; a single run is
  "consistent with," never a demonstration. `independent_scientific_confirmation` is `false` in the
  sealed artifact and in the verification artifact, set by neither producer nor verifier.

## 1. What was tested and why

Two prior ESCS questions on this corpus are resolved. Onset-localization is walled: seven independent
nulls across four gate variants and three frozen front-ends found that windowed spatial energy carries
no onset-localizing information at this budget. Source-counting produced the program's first positive
mechanics signal, but it died on bias-independent reproduction: 2 of 4 reproductions survived, below the
preregistered 3-survivor promotion bar, and the two failures were specifically the clip-macro scoring-unit
swap and the gate re-architecture swap (doc 28), exactly the two axes a positive is most likely to be an
artifact of if they were never checked in the original run.

This bed asks a third, distinct question: **direction-of-arrival (DoA) re-estimation value-of-computation.**
Given a frozen, zero-trained-parameter DoA estimator that is cheap to call but stale unless re-invoked,
does a trained gate decide *when* to re-invoke it well enough to reach a lower coasted great-circle
error than spending the identical re-estimation budget at random? DoA is a different signal from onset
presence (counting) or event timing (onset-localization): it is a continuous-valued regression target
with a coasted-track dynamic, present at every active frame (not just at onsets), so it exercises the
value-of-computation mechanism in a genuinely new way.

Because this is the third question and the first two failure modes are now known and specific, the bed
was **built hardened against both from the start**, not patched after a positive:

- **Clip-macro is the primary statistic, never pooled-frame.** Every arm scores as an equal-weight mean
  of per-clip mean angular errors first; the clip is the paired experimental unit for the sign-flip and
  the SESOI. Pooled-frame is computed and sealed as a labeled secondary corroborating readout only,
  never as a survive criterion. This directly targets the counting bed's `scoring_unit` failure mode.
- **Both gate architectures are checked in the same run.** Architecture A (264 -> 12 -> 1, 3,193
  params, byte-identical shape to the sealed onset and counting gates) and Architecture B
  (264 -> 6 -> 6 -> 1, 1,639 params, a genuinely deeper two-hidden-layer shape) are both trained, both
  scored, both required to survive independently, before any verdict is reported. This directly targets
  the counting bed's `gate_arch` failure mode, and prevents the bed from ever reporting a positive that
  is silently architecture-specific: the preregistered rule requires `SURVIVES(arch_a) AND
  SURVIVES(arch_b)`; exactly one surviving is labeled `architecture-fragile`, not a survive.

## 2. The hardened design

- **Frozen, zero-trained-parameter featurizer** (`doa_featurizer.DoaFeaturizer`): a 256-dim per-band
  spatial-flux-of-change front end (64 bands x [flux of 3 direction cosines + diffuseness]), 1,127,761
  FLOPs/frame, byte-reproducible. This is the WHEN signal.
- **Frozen, zero-trained-parameter estimator** (`doa_estimator.FrozenDoaEstimator`): a single wideband
  time-domain active-intensity direction per frame (33,687 FLOPs/reestimate, about 33x cheaper than the
  featurizer and about 5x Architecture A's gate). This is the WHAT signal, shared unchanged by every arm
  and every seed; only the re-estimation frame set differs per arm.
- **The only trained module is the gate**, in two co-equal architectures, both <= 4,096 params, sharing
  one 8-scalar online state (64 bytes), one training objective, and one `derive_seed32` seeding
  discipline so architecture is the only axis that moves between them.
- **Four controls**, three reused unchanged by import from the sealed `controls.py` plus one net-new
  trivial floor, mirroring `count_controls.py`: `rate_matched_random` (PRIMARY, matched re-estimation
  count per clip per seed per architecture), `always_on` (max-compute ceiling, architecture-independent),
  `never_update` (zero-compute floor, coast from the cold-start boresight forever, net-new), and
  `noisy_tv` (a real white-noise channel through the frozen featurizer, required to fire no more than
  the in-domain base rate).
- **Matched-budget harness**, dual-architecture: every arm, every seed, every architecture stays under
  the shared 6e10 FLOP ceiling in full-lifecycle accounting (training charged to the candidate only).
  `assert_within_ceiling` and `assert_matched_ex_training` are reimplemented locally (not imported from
  the onset `harness.py`) because this bed's arms are shaped differently (`.reestimations`, a minimized
  unbounded great-circle MAE) than the onset harness's `.firings`/maximized-F1 shape, the same reason
  `count_harness.py` already left that import path.
- **Statistics:** the PRIMARY test is an exact one-sided sign-flip permutation over the 21 test clips
  (meet-in-the-middle, reimplemented locally rather than imported from the counting bed's private
  scoring-unit helper), each clip's delta averaged over the 5 paired seeds first. A 5-seed secondary
  sign-flip (`stats.exact_sign_flip`, reused unchanged) is required to agree in direction. A
  **room-majority collapse** (carried forward as an explicit calibration-note discipline, not the
  literal design spec) collapses each of the 7 test rooms' 3 within-room clip deltas to a majority
  vote and runs an exact binomial-style sign-flip over rooms, since same-room clips share acoustics and
  are not fully exchangeable test units.
- **SESOI preregistered in code before any test score**, at `proof/STARSS23_DOA_BED.prereg.json`:
  0.233464 degrees, a structural, architecture-independent number (one test clip's worth of a single
  missed direction change, coasted half its typical inter-change dwell, folded into the clip-macro mean
  by a double `1/n_clips` normalization), used directly rather than approximated by a separate hardcoded
  default. `SURVIVES(X)` for architecture X requires, as one preregistered conjunction: the point
  estimate strictly favors the candidate; the mean clip-level delta clears the SESOI; the primary
  clip-level sign-flip clears alpha = 0.05; the 5-seed secondary agrees in direction; and the
  room-majority one-sided p does not exceed 0.10. Any tie is a null for that architecture.
- **Independent verifier** (`doa_verifier.py`): stdlib-only (`json`, `hashlib`, `itertools`, `math`,
  `bisect`, `dataclasses`), imports nothing from `mop` and nothing from any DoA producer module. It
  re-derives the great-circle geometry, the coasting, both scoring units, the clip-level and 5-seed
  sign-flips, the room-majority collapse, and the full `SURVIVES(X)` conjunction independently, for both
  architectures, from the raw `corpus_tracks` and per-seed `reestimate_frames` only. It sets
  `independent_scientific_confirmation` only on real data, clean rights, both-architecture at-chance
  noisy-TV, and `reproductions >= 3`, so it is `false` for this single run regardless of outcome.

## 3. The sealed result, honestly

Real STARSS23 subset: 45 clips total (18 train / 6 val / 21 test, matching the counting bed's split),
7 test rooms of 3 clips each, 19,344 active test frames carrying 601 ground-truth direction changes
(mean jump 48.2 degrees). 5 paired seeds (0-4) per architecture, swept re-estimation budget with the
operating point chosen from train-set change density only (5%), before any test score was read. Matched
lifecycle FLOPs: 2.97e10 (Architecture A) and 2.83e10 (Architecture B), both comfortably under the 6e10
ceiling. Noisy-TV control at chance for both architectures (0.7% firing on pure noise against an 11.5%
in-domain base rate for A; 20.3% against 28.8% for B: both gates fire distinctly *less* on structureless
noise than on real content, the honest direction, ruling out a degenerate always-fire gate).

**The trained gate did not beat rate-matched-random on either architecture. On the clip-macro primary,
it lost, on average, on both.**

| Result (clip-macro, primary) | Architecture A (264-12-1) | Architecture B (264-6-6-1) |
| --- | --- | --- |
| candidate mean MAE (degrees) | 112.123 | 109.987 |
| rate-matched-random mean MAE (degrees) | 109.149 | 108.492 |
| point estimate favors candidate | **False** | **False** |
| mean clip-level delta (rmr minus candidate) | **-2.974** | **-1.495** |
| clips individually favoring the candidate | 7 / 21 (33%) | 8 / 21 (38%) |
| primary one-sided sign-flip p (2,097,152 perms) | 0.918 | 0.792 |
| 5-seed secondary one-sided p (32 perms) | 0.938 | 0.875 |
| pooled-frame secondary delta (corroboration only) | -1.525 | -0.353 |
| room-majority: rooms favoring candidate | 2 / 7 | 1 / 7 |
| room-majority one-sided p (does-not-contradict bar 0.10) | 0.938 | 0.992 |
| SESOI (0.233 deg) cleared | False | False |
| **SURVIVES(architecture)** | **False** | **False** |

Bed-level verdict: **`null`**. `both_architectures_survive = false`, `architecture_fragile = false`:
this is not one architecture rescuing the other; both fail the identical clause (the point estimate
itself runs the wrong way) and every corroborating readout (pooled-frame secondary, room-majority) agrees
in the same negative direction. Delta sign convention matches doc 28: positive would mean the candidate
is lower (better); every delta above is negative, meaning rate-matched-random placement of the identical
budget beat the trained gate's placement, on average, under both architectures, at both the clip level
and the room level.

## 4. What kind of null this is

The task this bed inherits from the onset wall and the counting reproduction cohort is to say which
failure shape a new null resembles, honestly, rather than default to "not significant" as a single
undifferentiated bucket.

**This is not the counting bed's failure shape.** The counting positive was directionally correct and
cleared its SESOI on every axis; what killed two of its four reproductions was per-seed sign-flip
exactness at n=5, specifically once the scoring unit moved to clip-macro and once the gate was
reshaped, axes this bed deliberately made load-bearing in the original run instead of bolting them on
afterward. There is no comparable story available here: the point estimate itself is on the wrong side
for both architectures, at the clip level and the room level and under the secondary pooled-frame
readout, so there is no directionally-correct signal for a reproduction axis to have killed. By
construction, because clip-macro and dual-architecture were both primary from run one, this bed cannot
exhibit the counting bed's specific failure signature even in principle: there is no staged, later
positive to protect.

**This more closely resembles the onset bed's very first real result** (doc 26): a trained gate that
loses to rate-matched-random outright, consistently, rather than a signal that is present but statistically
fragile. The onset bed's first run lost at every firing budget and on every one of 5 seeds
(one-sided p = 1.0). This bed's result is the same qualitative shape (decisively wrong-directioned,
not narrowly missed), reproduced independently on a second, architecturally distinct gate in the same
run, which the onset bed's first null did not itself check.

**One observation is genuinely new and worth flagging for whoever looks at this next, stated as a
hypothesis and not a claim.** Both arms' absolute error is large: candidate and rate-matched-random both
average above 108 degrees of great-circle error, worse than the 90-degree expected error of two
*independent uniformly random* directions on a sphere. STARSS23 source placement is not actually
sphere-uniform (real recordings cluster in azimuth and a limited elevation band), so this is not
directly a "worse than chance" claim, but it does suggest the shared frozen WHAT (the wideband
active-intensity estimator, coasted) may itself carry a weak or biased direction signal on this corpus,
independent of the WHEN question this bed is built to isolate. If the estimator's own signal floor is
low, WHEN it is sampled could matter less than it would for a stronger estimator, which is a genuine
interaction the WHAT/WHEN separation does not distinguish on its own. This is not evaluated further
here; it is recorded as a candidate follow-up question, not folded into this bed's verdict.

Under this program's discipline: no positive is claimed, none was found, and none is promoted. This is a
legitimate, informative outcome on the third of three real questions this program has now run to a
conclusion on this corpus.

## 5. Independent verification

A separately authored verifier (`doa_verifier.py`, stdlib-only, importing no `mop` module and no DoA
producer module) re-read the sealed artifact and independently re-derived every graded number for both
architectures from raw `corpus_tracks` and per-seed `reestimate_frames`: the great-circle geometry, the
coasting, the clip-macro and pooled-frame scores, the primary clip-level sign-flip, the 5-seed secondary,
the room-majority collapse, the SESOI comparison, and the full `SURVIVES(X)` conjunction and bed-level
fold.

```
seal_intact: true
schema_ok: true
scores_reproduced: true
stats_reproduced: true
honesty_ok: true
independent_referee_reproduction: true
independent_scientific_confirmation: false
mismatches: []
```

A fully independent, third implementation (a brute-force `itertools.product` enumeration over the full
2^21 sign space, written separately from both the producer's and the verifier's meet-in-the-middle
implementations) reproduced the seal, a hand-picked clip's coasted MAE, and both architectures' primary
one-sided p exactly, closing the loop against a shared meet-in-the-middle algorithmic bug. All 32
unit and end-to-end tests pass (`tests/unit/test_starss23_doa_bed.py`), and the full STARSS23 and
compute-accounting regression suite (465 tests across the onset, counting, and DoA lanes) passes with
no regression in any pre-existing test.

`independent_scientific_confirmation` is `false` in the verification artifact and remains unset here: it
requires real data, clean rights, both-architecture at-chance noisy-TV, and at least three
bias-independent reproductions, none of which a single run, survive or null, can satisfy on its own.

## 6. Additive-only and rights posture

- No sealed onset module (`referee.py`, `stats.py`, `controls.py`, `harness.py`, `gate.py`, `prereg.py`)
  and no sealed `count_*` module was edited; all 16 are byte-identical to the committed HEAD (direct
  SHA-256 comparison, not just an empty diff).
- No existing proof was modified. `proof/STARSS23_DOA_BED.json`,
  `proof/STARSS23_DOA_BED.prereg.json`, and `proof/STARSS23_DOA_BED.verification.json` are net-new.
- All ten `doa_*` modules under `src/mop/beds/starss23/` are net-new; none is imported by, and none
  imports, any live campaign or orchestration path. Nothing under `runs/` or `studies/` was read or
  written by this lane.
- Real, rights-clean, room-disjoint STARSS23 data throughout (`source_kind: "real"`,
  `rights_clean: true`); `activation_allowed`, `scientific_promotion`, and
  `independent_scientific_confirmation` are hardcoded `false` in the sealed artifact.
