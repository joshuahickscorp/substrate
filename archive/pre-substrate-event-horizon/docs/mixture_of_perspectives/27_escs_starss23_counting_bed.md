# ESCS STARSS23 Concurrent-Source-Counting Bed and its First Real Positive

> **Append-only Stage-3 experimental lane.** This is a net-new bed package under
> `src/mop/beds/starss23/` (all modules are `count_*`). It opens a DIFFERENT question from the seven
> sealed onset-localization nulls of doc 26: not WHEN a source onsets, but HOW MANY concurrent sources
> are active. It touches no sealed onset scoring path (`referee.py`, `stats.py`, `controls.py`,
> `harness.py`, `gate.py`, `prereg.py` are byte-unchanged) and no live campaign file. It reuses the
> onset bed's exact-sign-flip statistic and rate-matched-random control by import only.

- **Status:** built, run once on real MIT STARSS23. **First result is a POSITIVE mechanics-demonstration
  that exceeds the preregistered SESOI and clears the one-sided sign-flip. It is NOT a scientific
  confirmation.** It is flagged loudly below for the promotion bar: at least three bias-independent
  reproductions on real rights-clean, session-disjoint data.
- **Snapshot date:** 2026-07-18
- **Dataset:** STARSS23 (zenodo.org/records/7880637), MIT license, commercial-use OK; 4-channel FOA
  spatial audio, native room-disjoint dev split. The per-event active intervals in the metadata make the
  per-frame concurrent-source count directly derivable, so the label needs no model.
- **Claim scope:** deterministic matched-budget adaptive-compute mechanics on a real rights-clean bed; no
  activation, no scientific promotion, no natural-world generality claim; a single run is "consistent
  with," never a demonstration.

## 1. What the bed tests

The bed frames concurrent-source counting as an adaptive-compute value-of-computation gate. A count
estimate goes stale between re-estimations; a re-estimation costs compute; so the honest question is
saving-at-parity: given a fixed re-estimation budget K, does a trained gate that chooses WHEN to
re-estimate beat spending the same K re-estimations at uniformly random positions, measured by coasted
count error? If it cannot, the trained module is not earning its cost. The load-bearing pieces, all
net-new and additive:

- **Frozen zero-trained-parameter count featurizer** (`count_featurizer.py`): 256 features per frame from
  the 4-channel FOA, `n_params() == 0`, byte-reproducible; 1,120,700 FLOPs per frame, charged identically
  to every arm.
- **Frozen zero-trained-parameter count estimator** (`count_estimator.py`): maps features to an integer
  count in [0, 4], `n_params() == 0`, deterministic, silence maps to 0; 80,000 FLOPs per re-estimation.
  The estimator track E is a property of the clip and is shared by every arm, so only the re-estimation
  set R differs per arm.
- **The trained count gate is the ONLY trained module** (`count_gate.py`): MLP with 3,193 trainable
  parameters (hard ceiling 4,096), online state 64 bytes, 6,385 FLOPs per inference. It takes features and
  its own state, never a label. Its full-lifecycle training cost `C_train` = 8.27e9 FLOPs is charged.
- **Four reference and control arms** (`count_controls.py`): the primary control `rate_matched_random`
  (same re-estimation count K as the candidate, positions permuted uniformly, matched per seed and per
  clip); `always_on` (re-estimate every frame, the full-budget reference whose MAE reduces to
  `mean|E - C_gt|`); `never_update` (re-estimate nothing, coast from 0, MAE reduces to `mean|C_gt|`); and a
  `noisy_tv` RND channel with a required at-chance re-estimation rate.
- **Sealed coasted-count-MAE referee** (`count_referee.py`): `emitted(t) = E[max{r in R : r <= t}]` else
  cold-start 0, pooled frame micro-average `sum_clips sum_t |emitted(t) - C_gt(t)| / sum_clips T`. A level
  metric needs no matching, so the referee is deterministic by construction; lower is better; a tie is a
  null.
- **Matched-budget harness** (`count_harness.py`): 5 paired seeds, each arm total at most 6e10 FLOPs,
  candidate equal to rate-matched-random ex-training (equal K, byte-equal inference FLOPs).
- **Stats:** exact sign-flip permutation over 2^5 = 32 signs (reusing the onset bed's `exact_sign_flip` by
  import), one-sided minimum p = 1/32; two-sided 0.05 unreachable at n = 5. A compute-normalized SESOI of
  0.02 pooled MAE, preregistered in code at `count_prereg.py` before any test score was read, with a
  quantified cost-benefit rationale (0.02 is 451x the per-frame granularity floor and about one test
  clip's catchable change mass).
- **Independent verifier** (`count_verifier.py`): a separately authored module that imports none of the
  producer, referee, harness, or anything under `mop` (only `json`, `hashlib`, `itertools`,
  `dataclasses`); it re-reads the sealed artifact, re-coasts and re-scores every arm from the raw tracks,
  re-runs the sign-flip, and flips `independent_scientific_confirmation` only on real rights-clean data
  meeting the full promotion bar.

## 2. First real result: a positive that clears the SESOI and the sign-flip

Run on the room-disjoint dev subset (45 clips: 18 train, 6 val, 21 test, distinct rooms; 25,172 train
frames, 22,569 test frames, 916 test changes), 5 paired seeds, at the operating re-estimation fraction
0.05, matched lifecycle FLOPs 2.955e10 (under the 6e10 ceiling). Sealed at
`proof/STARSS23_COUNTING_BED.json`; SESOI preregistered first at
`proof/STARSS23_COUNTING_BED.prereg.json`; independently verified at
`proof/STARSS23_COUNTING_BED.verification.json`.

**At a fixed 5 percent re-estimation budget the trained gate reached a strictly lower coasted-count-MAE
than rate-matched-random on all five paired seeds, exceeding the preregistered SESOI and clearing the
one-sided sign-flip at its exact discrete floor.**

| Result | Value |
| --- | --- |
| candidate strictly dominates rate-matched-random | **True** (harness verdict `mechanics-ok`) |
| mean coasted-count-MAE delta (candidate minus random) | **-0.0436** (lower MAE is better; candidate lower on all 5 seeds) |
| per-seed candidate-minus-random deltas | -0.0261, -0.0028, -0.0241, -0.1355, -0.0296 (all negative) |
| one-sided sign-flip p (candidate lower than random) | **0.03125** (= 1/32, the exact floor at n=5, all five deltas share sign) |
| two-sided 0.05 reachable at n=5 | **False** (minimum two-sided is 2/32 = 0.0625) |
| preregistered SESOI (pooled count-MAE) | 0.02, **exceeded** (mean delta 0.0436 is about 2.2x the SESOI) |
| candidate mean MAE / rate-matched-random mean MAE | **0.8208 / 0.8644** |
| always-on reference MAE (100% re-estimation) / never-update MAE (coast from 0) | 0.8291 / 1.2552 |
| noisy-TV at chance | **True** (re-estimates 3.35% on noise vs a 7.48% content base rate; does not chase noise) |
| independent referee reproduction | **True** (zero mismatches, no producer import, seal intact) |
| **independent scientific confirmation** | **False** (one run; the bar is at least 3 bias-independent reproductions) |
| flags | activation_allowed=false, scientific_promotion=false |

**Honest claim sentence.** On the real STARSS23 room-disjoint dev-test (21 clips, the experimental unit),
this single run is consistent with a value-of-computation gate that, at a fixed 5 percent re-estimation
budget, places its re-estimations better than uniform random placement for concurrent-source counting
(mean coasted-count-MAE delta -0.0436, one-sided sign-flip p = 0.03125, preregistered SESOI 0.02
exceeded). By the pseudoreplication ceiling this is a suggestive single-run result, never a demonstration.

## 3. LOUD FLAG: this is a positive, and the promotion bar is NOT met

Unlike the onset bed (doc 26), whose seven sealed results are all nulls, this bed produced a positive on
its first real run. **That positive must not be read as a confirmed result, and nothing downstream may
treat it as one.** The reasons it stays unpromoted, made explicit:

- **`independent_scientific_confirmation` is `false`** in the sealed artifact and stays `false` under
  independent verification. `reproductions` is 0; the promotion bar is at least 3 bias-independent
  reproductions on real rights-clean, session-disjoint data.
- **The significance is at the exact discrete floor and is one-sided.** With n = 5 all-same-sign the
  minimum one-sided p is 1/32 = 0.03125 and two-sided 0.05 is unreachable. The one-sided framing was
  preregistered for a directional value-of-computation hypothesis, but it is load-bearing: a two-sided
  test at this n cannot reach 0.05. A single run at the floor is exactly the regime where a lucky sign
  pattern is most plausible, which is why the bar demands independent reproductions rather than a smaller p.
- **One seed carries most of the effect.** Seed 3 contributes a -0.1355 delta against a -0.0028 to -0.0296
  spread on the other four. The mean clears the SESOI comfortably, but the per-seed distribution is
  skewed, so reproduction on fresh seeds and fresh session splits is what would show the effect is not a
  single-seed artifact.
- **The confirmation gate trusts the artifact's own reproduction counter.** The verifier grants
  confirmation only when `reproductions >= 3`, but it reads that integer from the artifact under audit. A
  legitimate single-run producer writes 0 (as here), so the guardrail holds for this run; but promotion
  must be adjudicated by an external ledger of at least three independently produced, independently sealed
  artifacts with distinct provenance, not by any single artifact's self-reported counter. Do not promote
  on the strength of one file.

Until three bias-independent reproductions are on record, the counting substrate stays at Stage 2 exactly
as the onset substrate does, and this positive is a mechanics-demonstration only.

## 4. Independent verification (this phase)

The sealed artifact was re-derived from scratch by a separately authored verifier that imports no producer
code (only the standard library). Every number below was recomputed from the raw per-clip ground-truth
count tracks, the shared frozen estimator track, and the per-arm re-estimation frames; the stored scores
were used only to compare against, never to compute.

- **All four arms on all five seeds reproduced with zero mismatches.** The two cross-checks that use the
  shared estimator track both held: `always_on` MAE equals `mean|E - C_gt|`, and `never_update` MAE equals
  `mean|C_gt|` = 1.2552 = the sealed `test_coast_from_zero_mae`.
- **The budget match held.** `rate_matched_random` spends exactly the candidate's re-estimation count K on
  every clip and every seed.
- **The paired deltas, the mean, and the sign-flip reproduced.** Recomputed mean delta (control minus
  candidate) = 0.043608489521024404 = the sealed `t_obs`; exact sign-flip over 32 permutations gave
  one-sided p = 0.03125; the recomputed mean exceeds the SESOI 0.02.
- **Integrity.** The artifact seal `a474dc88...` re-hashes exactly (canonical JSON over the body minus the
  seal); the prereg canonical hash `5481f1db...` re-hashes exactly and matches the value the artifact
  records; `written_before_test_scores` is true; the claim verb is "consistent with".
- **Tamper resistance was exercised.** Bumping a stored MAE without re-sealing breaks the seal; bumping it
  and re-sealing passes the seal but fails the referee re-derivation; corrupting the shared estimator track
  and re-sealing fails the re-derivation; a zeroed seal is caught. The verifier is not trivially passing.

`independent_referee_reproduction` is therefore `true` and `independent_scientific_confirmation` is `false`
on this single run, which is the intended separation.

## 5. Why the win is real and where it does and does not reach

The mechanism is legible. At a 5 percent re-estimation budget the gate learns to spend its re-estimations
near count changes rather than uniformly, so its coasted track tracks the true count with fewer stale
frames than random placement, on all five seeds. The effect is a budget-allocation win, not a raw-accuracy
win: the estimator and featurizer are frozen and shared, so no arm can out-estimate another on a given
frame; the only lever is WHERE the fixed budget is spent. The noisy-TV control stayed at chance (3.35% on
noise vs 7.48% on content), so the gate is placing re-estimations on real count structure, not chasing
energy.

Two honest boundaries on the reach of this result:

- **It is a placement win, and at this budget it is competitive with always-on.** The candidate's mean
  MAE (0.8208) is slightly below the always-on reference (0.8291), meaning 5 percent of the re-estimation
  budget spent in learned positions is about as good as re-estimating every frame on this subset. That is
  the interesting direction, but it is a single-run observation on one operating point and is bounded by
  the same promotion bar; it is not a claim that learned placement beats full re-estimation in general.
- **It is a count-tracking result, not a counting-capability result.** The estimator's per-frame count is
  frozen and hand-built; the bed measures how well a gate schedules re-estimations of that estimator, not
  whether a model can count sources from scratch. The value-of-computation claim is exactly scoped to
  scheduling under a fixed budget.

## 6. What is next

This bed now exists to test whether the positive reproduces. The single promotable path is triangulation:
at least three bias-independent reproductions on real rights-clean, session-disjoint data, each a fresh
append-only run with the SESOI preregistered before any test score is read, each independently sealed and
independently verified, adjudicated by an external ledger rather than a self-reported counter. Distinct
session splits, distinct seed families, and a second operating budget point are the natural axes of
independence. Until then the counting substrate stays at Stage 2, honestly, and this result is carried as
a flagged positive awaiting the promotion bar, not as a confirmation.
