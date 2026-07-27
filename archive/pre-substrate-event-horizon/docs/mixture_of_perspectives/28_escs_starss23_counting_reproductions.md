# ESCS STARSS23 Counting Signal: Four Bias-Independent Reproductions

> **Append-only Stage-3 experimental lane.** This document is the reproduction cohort for the single
> positive reported in [27_escs_starss23_counting_bed.md](./27_escs_starss23_counting_bed.md). It is
> net-new only: four `count_repro_*` module families under `src/mop/beds/starss23/` plus their sealed
> `proof/STARSS23_COUNTING_REPRO_*.json` artifacts. It edits no sealed `count_*` or onset scoring
> module and no existing proof; every original `count_*` bed module is byte-unchanged in git. It
> touches no live campaign or process file. It reuses the exact-sign-flip statistic and the
> rate-matched-random control by import only.

- **Status:** all four reproductions built, run once each on real MIT STARSS23, sealed, and
  independently verified with zero verifier mismatch. **Two of four survived the strict preregistered
  conjunction; two are verified nulls.** The single-run positive of doc 27 did NOT clear the
  preregistered promotion bar of at least three bias-independent survivors.
- **Snapshot date:** 2026-07-18
- **Dataset:** STARSS23 (zenodo.org/records/7880637), MIT license, commercial-use OK; 4-channel FOA
  spatial audio, native room-disjoint dev split. All four runs used real, rights-clean,
  room-disjoint data with `source_kind = "real"` in every sealed proof.
- **Claim scope:** deterministic matched-budget adaptive-compute mechanics on a real rights-clean bed;
  no activation, no scientific promotion, no natural-world generality claim. `independent_scientific_confirmation`
  is `false` in all four verification artifacts and is set here by no one. This is not Stage 3.

## 1. What was tested and why

Doc 27 reported a first positive mechanics signal on the STARSS23 concurrent-source-counting bed: a
trained value-of-computation gate reached a strictly lower coasted-count-MAE than rate-matched-random
at a matched re-estimation budget. One run is not a capability. The honest adversarial question is
whether that saving-at-parity advantage is a generic property of learned re-estimation placement, or an
artifact of one specific choice in the sealed pipeline. Four reproductions each vary one likely source
of spurious advantage, hold every other axis fixed, use a seed family disjoint from the original bed
(0-4) and from each other, preregister their own SESOI and sign-flip floor IN CODE before any test
score, and are checked by a separately-authored verifier that recomputes every sealed number.

Each reproduction survives only under the strict conjunction, exactly as preregistered:

1. candidate mean count-MAE strictly below rate-matched-random (same direction, lower), AND
2. mean paired delta at or above the registered SESOI, AND
3. one-sided exact sign-flip p at or below 1/32 (the five-seed floor, 0.03125), AND
4. independent verifier reproduces every score and stat with zero mismatch.

Any tie or reversal is a null. A directionally-correct mean that misses the sign-flip floor is a null.

## 2. The honest table

All numbers below are read directly from the sealed `proof/STARSS23_COUNTING_REPRO_*.json` and their
`.verification.json`, at each reproduction's own preregistered operating budget point (`rate_0.05`).
Delta is defined so that positive means the candidate is lower (better): delta = MAE(rate_matched_random)
minus MAE(candidate), mean over the five paired seeds.

| Reproduction axis | Bias varied | Seeds | Mean delta (rmr - cand) | SESOI | SESOI cleared | One-sided sign-flip p | Floor 1/32 cleared | Verifier mismatches | Survives |
|---|---|---|---|---|---|---|---|---|---|
| `data_split` | room-fold swap: train on original test rooms, score on original train rooms | 10-14 | +0.116462 | 0.020833 | yes | 0.03125 | yes (= 1/32) | 0 | **YES** |
| `featurizer_estimator` | swapped featurizer and estimator | 20-24 | +0.164429 | 0.023810 | yes | 0.03125 | yes (= 1/32) | 0 | **YES** |
| `scoring_unit` | clip as experimental unit (clip-macro mean, no frame pseudoreplication) | 30-34 | +0.068121 | 0.023810 | yes | 0.0625 | no (2/32) | 0 | **NO (null)** |
| `gate_arch` | two-hidden-layer gate 264 -> 8 -> 4 -> 1 (vs sealed 264 -> 12 -> 1) | 40-44 | +0.052647 | 0.023810 | yes | 0.125 | no (4/32) | 0 | **NO (null)** |

Provenance held constant and verified in every sealed proof: `source_kind = "real"`, room-disjoint
split, 5 paired seeds, three controls (primary `rate_matched_random` plus `always_on`/`never_update`,
with `noisy_tv` at chance), matched budget under the 6e10 FLOP ceiling (data_split 3.53e10,
featurizer_estimator 4.37e10, scoring_unit 2.96e10, gate_arch 2.98e10), and a separately-authored
verifier reporting `mismatches: []` and `independent_scientific_confirmation: false`.

## 3. What killed the two nulls

Both nulls fail on exactly one clause: the strict one-sided sign-flip floor. Their means are
directionally correct and clear the SESOI, so what breaks is per-seed consistency, the hardest of the
four conditions.

- **`scoring_unit`** (clip-macro): four of five paired seeds put the candidate lower, but one seed
  reversed (per-seed deltas `[0.073, -0.028, 0.092, 0.111, 0.093]`), giving an exact sign-flip
  p = 2/32 = 0.0625, above the 1/32 floor. Note the corroborating clip-clustered readout does agree in
  direction (its own p = 0.0033 over 2,097,152 permutations), but the preregistered survive rule
  requires the primary five-seed sign-flip to clear the floor first, and it does not. Verified null.
- **`gate_arch`** (two hidden layers, 2,161 params): two of five paired seeds were non-positive
  (per-seed deltas `[-4.4e-05, -0.0018, 0.101, 0.038, 0.126]`), giving an exact sign-flip
  p = 4/32 = 0.125, well above the floor. The smaller two-layer gate reproduces the direction on the
  mean but not with the per-seed reliability the floor demands. Verified null.

## 4. The honest reading

**Survivors: 2 of 4.** The preregistered promotion bar was at least three bias-independent survivors.
Two is below that bar. The single-run counting saving-at-parity positive of doc 27 therefore did NOT
survive bias-independent reproduction to the promotion threshold. Under the discipline of this program
that means it is not established as a real saving-at-parity capability. It is a mapped boundary, which
is a legitimate and informative outcome.

The boundary is specific and honestly stated. The advantage is robust to the two bias axes that would
most plausibly have manufactured a spurious win from data-partition luck or featurization detail: it
survives the room-fold swap (`data_split`) and the featurizer-plus-estimator swap
(`featurizer_estimator`), on disjoint seeds, clearing all four clauses including the exact sign-flip
floor at its 1/32 minimum. It does NOT survive when the experimental unit is changed from the pooled
frame micro-average to the clip macro-average (`scoring_unit`), nor when the gate is reshaped to a
smaller two-hidden-layer network (`gate_arch`). In both failing cases the mean effect is present and
directionally consistent and clears the SESOI, but the per-seed exactness required by the sign-flip
floor does not hold. That the two survivors and the two nulls all share the same sign on the mean is
worth recording as boundary detail, but it is not a promotion argument and is not used as one here: a
directionally-correct mean that misses the floor is preregistered as a null, and it is scored as a null.

No positive is promoted here. `independent_scientific_confirmation` stays `false` in every artifact and
is set by nothing in this document. Stage 3 is not reached. Whether to attempt further reproductions to
probe the boundary (for example, more seeds to power the sign-flip on the two null axes, or additional
bias axes) is a human decision and is not taken here.

## 5. Additive-only and rights posture

- No sealed `count_*` bed module (`count_controls`, `count_estimator`, `count_featurizer`, `count_gate`,
  `count_harness`, `count_labels`, `count_prereg`, `count_producer`, `count_referee`, `count_verifier`)
  and no onset scoring module was edited; all are byte-unchanged in git.
- No existing proof was modified. The four `STARSS23_COUNTING_REPRO_*` proof, prereg, and verification
  artifacts are net-new.
- Each reproduction preregisters its own SESOI and sign-flip floor in code before any test score, and is
  checked by a separately-authored verifier (`count_repro_*_verifier.py`, distinct from
  `count_repro_*_producer.py`) that reports zero mismatch.
- Real, rights-clean, room-disjoint STARSS23 data throughout; no live process or campaign file touched.
