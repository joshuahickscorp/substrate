# ESCS STARSS23 Event-Formation Bed and its First Real Result

> **Append-only Stage-3 experimental lane.** This is the first real matched-budget bed in the
> Generation 1 program. It is the X0-R successor to the verified X0 strong null, and Generation 1
> row G1-E1 (event formation). It does not touch any sealed Generation 1 campaign file; the bed is a
> net-new package under `src/mop/beds/starss23/`.

- **Status:** built, run once on real MIT STARSS23; first result is a verified null.
- **Snapshot date:** 2026-07-17
- **Dataset:** STARSS23 (zenodo.org/records/7880637), MIT license, commercial-use OK; 4-channel FOA
  spatial audio at 24 kHz, 100 ms onset labels, native room-disjoint dev split.
- **Claim scope:** deterministic matched-budget adaptive-compute mechanics on a real rights-clean bed;
  no activation, no scientific promotion, no natural-world generality claim; a single run is
  "consistent with," never a demonstration.

## 1. What the bed tests

The bed frames event formation as an adaptive-compute value-of-computation gate. The honest question
is saving-at-parity: a trained gate must beat a rate-matched-random gate on onset F1 at the same total
compute. If it cannot, spending the firing budget at random is as good, and the trained module is not
earning its cost. The design and its numeric recipe come from `docs/ESCS_DEEP_RESEARCH.md`
(adversarially verified). The load-bearing pieces:

- **Frozen zero-trained-parameter featurizer** (log-mel plus half-wave-rectified spectral flux over
  the 4-channel FOA at 100 ms frames), byte-reproducible; 1,121,340 FLOPs per frame, charged
  identically to every arm.
- **Candidate gate is the only trained module:** MLP 264 to 12 to 1 = 3,193 trainable parameters
  (hard ceiling 4,096), online state under 8 KB. It predicts decision value, not raw novelty. Its
  training cost `C_train` = 8.27e9 FLOPs is charged in full-lifecycle accounting; break-even
  `N* = C_train / per-query saving` is about 224,863 frames (~6.2 hours of audio).
- **Three mandatory controls:** rate-matched-random (same firing count, positions permuted uniformly,
  matched per-seed), always-on / best-single reference, and a noisy-TV RND channel with a required
  at-chance firing rate.
- **Sealed referee:** onset F1 at a DCASE plus-or-minus 200 ms collar, greedy one-to-one nearest-first
  matching, strict point-wise PR (point-adjustment is forbidden; it lets a uniform-random detector
  reach F1 above 0.96).
- **Matched-budget harness:** 5 paired seeds, each arm total at most 6e10 FLOPs, candidate equal to
  rate-matched-random ex-training.
- **Stats:** exact sign-flip permutation over 2^5 = 32 signs, one-sided minimum p = 1/32; two-sided
  0.05 is unreachable at n = 5. A preregistered compute-normalized SESOI and a pseudoreplication claim
  ceiling (the clip is the experimental unit).
- **Independent verifier:** a separately authored module that imports none of the producer scorer and
  re-scores the referee and sign-flip from the sealed artifact; it flips
  `independent_scientific_confirmation` only on real rights-clean data meeting the full promotion bar.

## 2. First real result: a verified null

Run on the room-disjoint dev subset (24 train clips, 21 test clips, distinct rooms), 5 paired seeds,
matched lifecycle FLOPs 2.95e10 (under the 6e10 ceiling). Sealed at
`proof/STARSS23_ESCS_BED.json`; SESOI preregistered first at `proof/STARSS23_ESCS_BED.prereg.json`;
independently verified at `proof/STARSS23_ESCS_BED.verification.json`.

**The trained gate did not beat rate-matched-random. It lost at every firing budget and on every seed.**

| Result | Value |
| --- | --- |
| candidate strictly dominates rate-matched-random | **False** (harness verdict `null`) |
| mean onset-F1 delta (candidate minus random) | **-0.023** (all five per-seed deltas negative) |
| one-sided sign-flip p (candidate greater than random) | **1.0** (floor 1/32) |
| two-sided 0.05 reachable at n=5 | False |
| Pareto-optimal arm | the non-learned `best_single` flux threshold, F1 = 0.209 |
| preregistered SESOI (onset F1) | 0.05, **not exceeded** (fails on magnitude and direction) |
| noisy-TV at chance | **True** (fires 2.2% on noise vs 13.7% on content, does not chase noise) |
| independent referee reproduction | **True** (zero mismatches, no producer import) |
| independent scientific confirmation | **False** (one run; the bar is at least 3 bias-independent reproductions) |
| flags | activation_allowed=false, scientific_promotion=false |

**Honest claim sentence.** On the real STARSS23 room-disjoint dev-test (21 clips, the experimental
unit), this single run is consistent with the value-of-computation gate providing no onset-localization
advantage over spending the same firing budget at random (mean onset-F1 delta -0.023, one-sided
sign-flip p = 1.0, preregistered SESOI 0.05 not exceeded). By the pseudoreplication ceiling this is a
suggestive single-run result, never a demonstration.

## 3. Why this is a real null, not a wiring bug

The gate trains correctly (loss 0.537 to 0.444, firing rate tracks its target) and fires a
non-degenerate spread, but it clusters roughly 42% of its fires adjacently on high-energy regions, so
it recovers fewer distinct onsets (204 true positives) than uniformly spread random placement (237)
at matched budget. That is exactly the failure mode the SkipNet/BlockDrop rate-matched-random control
exists to catch, and it caught it. No control, referee, or statistic was weakened to move the number.

## 4. What this means and what is next

The bed works: it distinguishes an honest attempt from free random placement and refuses to manufacture
a false positive. The first gate design (a small MLP on log-mel plus flux, trained on a
value-of-computation target) is not a winning event former on this bed.

The bed now exists to test better gate designs against the same sealed referee and controls. Iterating
the gate (for example a recurrence-aware or temporally-structured trigger that spreads its fires rather
than clustering on energy) is the natural next step, and each attempt is a fresh append-only run. A
Stage-3 confirmation still requires a candidate that exceeds the SESOI, clears the one-sided sign-flip,
and is triangulated by at least three bias-independent reproductions on real rights-clean,
session-disjoint data. Until then the substrate stays at Stage 2, honestly.
