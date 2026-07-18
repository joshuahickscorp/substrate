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

## 5. E1 gate-variant iteration wave: four spread-fire designs, four nulls

The diagnosis in section 3 was specific: the base gate wastes budget by clustering roughly 42% of its
fires adjacently, so it recovers fewer distinct onsets than uniform random placement. The E1 wave built
four net-new gate variants (additive modules under `src/mop/beds/starss23/`, no sealed scoring logic
touched, gate parameters within the 4,096 ceiling, matched lifecycle FLOPs under the 6e10 ceiling),
each aimed squarely at that clustering, and scored every one against the same sealed referee and the
same rate-matched-random control over five paired seeds. The SESOI (0.05 onset F1) was preregistered in
code before any variant read a test score, and the four-variant family carries a Bonferroni multiplicity
wall that is unreachable at n = 5 by construction, so no single wave can promote.

**All four variants nulled. None beat rate-matched-random; none exceeded the SESOI; none cleared the
one-sided sign-flip. A tie is a null, so the wave is a null.**

| Variant | Mechanism (spread strategy) | mean onset-F1 delta (cand - random) | one-sided sign-flip p | SESOI 0.05 exceeded | adjacency fraction (cand) | distinct-onset TP cand vs matched random | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `refractory_nms` | post-decision collar-width non-maximum suppression on `p_fire`; one fire per supra-threshold cluster | **+0.00264** | 0.344 | No | **0.00** (from 0.51 base) | 210.4 vs 206.4 | null |
| `recurrence_spread` | recurrent density plus refractory penalty subtracted from the signal logit (2 extra params) | **-0.00369** | 0.750 | No | 0.097 (from ~0.42 base) | 192.8 vs 199.0 | null |
| `learning_progress` | RND reducible-surprise firing (fire on surprise being reduced, not on energy); 512 params | **-0.00343** | 0.875 | No | 0.152 (from 0.363 base, same definition) | pooled 872 vs 892 | null |
| `diversity_reg` | scale-invariant determinantal spacing regularizer added to the training loss | **-0.01189** | 0.781 | No | **0.400** (matched random 0.347) | 235.4 vs 266.6 | null |

No adversarial verification pass was run this wave, because a verification pass only gates a claimed
positive and there was no positive to gate: no variant produced a SESOI-exceeding, sign-flip-clearing
win. `independent_scientific_confirmation` therefore stays `false`, unchanged.

### The mechanistic wall this wave establishes

De-clustering the fires is necessary but not sufficient, and the four variants map out why cleanly:

- **`refractory_nms` proves the adjacency diagnosis was correct and still insufficient.** It drove the
  adjacency fraction to exactly 0 (no two fires inside one collar) and, for the first time, pushed the
  candidate's distinct-onset true positives slightly ABOVE matched random (210.4 vs 206.4). But the
  resulting onset-F1 gain (mean +0.00264) is an order of magnitude below the 0.05 SESOI, is not
  sign-flip significant (p = 0.344), and does not hold at the tightest firing budget (per-budget delta
  goes negative at rate 0.06). Perfectly spacing the fires recovers only the handful of distinct onsets
  the clustered budget was double-covering, and those onsets are no more onset-predictive than the ones
  random placement already finds.
- **`recurrence_spread` and `learning_progress` over-suppress.** Both cut adjacency sharply (to 0.097
  and 0.152) but at the cost of withholding fires on real onsets: their distinct-onset TP fell to or
  below matched random (192.8 vs 199.0; pooled 872 vs 892). `learning_progress` is instructive: firing
  on reducible surprise shrank the matched-control TP deficit from ~210 to ~20 pooled, the largest
  structural improvement in the wave, yet it fires far fewer times in absolute terms (~1,748 vs the base
  gate's ~2,957 at the intended 8% budget) so its absolute recall is lower and it still loses. Reducing
  the deficit is not the same as reversing it.
- **`diversity_reg` fails at the mechanism level.** Its spacing penalty lives in the training loss, but
  the ranking that the harness thresholds did not actually de-cluster at inference: the candidate's
  adjacency fraction stayed at 0.400, HIGHER than matched random's 0.347, and its distinct-onset TP
  (235.4) fell below matched random (266.6). It also showed the highest seed variance (seed 0 spiked to
  0.207 then collapsed to ~0.085), and its hyperparameter-search cost has unsealed provenance
  (~7.7e10 FLOPs, above the 6e10 per-arm ceiling), so even a spike could not be promoted. A
  training-time spacing prior did not survive into an inference-time firing pattern here.

The wave's honest lesson: the base gate's value-of-computation signal carries no onset-localizing
information beyond energy that survives de-clustering. Once you stop double-covering loud regions, the
redistributed budget lands on frames no more onset-bearing than uniform random placement chooses, so
distinct-onset recall barely moves and onset F1 does not clear the SESOI. Spreading the fires is a real
and now-measured constraint, not the missing capability. As X0 and section 3 both indicated, the gap is
relational and temporal event interpretation (knowing WHICH frame is an onset), not firing sparsity or
firing spacing. That is a proven wall with a mechanistic reason, which is a legitimate wave outcome.

The substrate stays at Stage 2. The bed and its sealed referee did their job again: they distinguished
four honest attempts from free random placement and refused to manufacture a positive from any of them.

## 6. E1 featurizer wave: three frozen front-ends, three nulls

Sections 3 and 5 bracketed the trained gate: the base gate wastes budget by clustering its fires, and
redesigning the gate's spread strategy (four variants) still did not clear the SESOI. This wave holds
the trained gate fixed and attacks the other half of the pipeline. It asks whether a richer frozen
zero-trained-parameter front-end carries onset-localizing information the unchanged gate can exploit.
Three net-new featurizers were built (additive modules under `src/mop/beds/starss23/`, no sealed scoring
logic touched), each emitting exactly 256 features natively so it feeds the unchanged 3,193-parameter
gate with no projection and no truncation, each `n_params() == 0` and byte-reproducible, and each
charged its own honest per-frame FLOP cost identically per arm. All three ran on the REAL room-disjoint
STARSS23 FOA dev subset (45 clips, 21 test clips, 538 onsets, 22,569 test frames, 25,172 train frames),
five paired seeds. The SESOI (0.05 onset F1) was preregistered in code before any featurizer read a test
score, and the three-featurizer family carries a Bonferroni multiplicity wall (per-featurizer alpha
0.0167, minimum achievable one-sided p 0.03125) that is unreachable at n = 5 by construction, so no
single wave can promote.

**All three featurizers nulled. None beat rate-matched-random; none exceeded the SESOI; none cleared the
one-sided sign-flip. A tie is a null, so the wave is a null.**

| Featurizer | Mechanism (frozen front-end) | mean onset-F1 delta (cand - random) | one-sided sign-flip p | SESOI 0.05 exceeded | noisy-TV at chance | matched FLOPs | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `spatial_doa` | per-band active-intensity direction of arrival from the FOA B-format, `I = Re{conj(W)[X,Y,Z]}` reduced per band to direction cosines plus DirAC diffuseness (64 bands x 4 = 256) | **-0.0031** | 0.6875 | No | True (4.5% on noise vs 10.8% content) | 2.96e10 | null |
| `superflux_spectral` | frozen SuperFlux onset detector (Boeck and Widmer, DAFx 2013): mu-law log-mel with maximum-filter vibrato suppression and positive spectral flux (256) | **-0.0182** | 1.0 | No | **False** (53.4% on noise vs 37.5% content) | 3.04e10 | null |
| `interchannel_coherence` | magnitude-squared coherence between the FOA omni W and each gradient channel X, Y, Z per band, plus DirAC directness (64 bands x 4 = 256) | **-0.0290** | 1.0 | No | True (2.2% on noise vs 10.1% content) | 3.02e10 | null |

Each proof is sealed net-new (`proof/STARSS23_ESCS_BED_spatial_doa.json`,
`proof/STARSS23_ESCS_BED_superflux_spectral.json`, `proof/STARSS23_ESCS_BED_interchannel_coherence.json`)
with its SESOI preregistered first in the matching `.prereg.json`. No adversarial verification pass gated
this wave, because a verification pass only gates a claimed positive and there was no positive to gate.
`independent_scientific_confirmation` therefore stays `false`, unchanged.

### The mechanistic wall this wave establishes

The three front-ends fail in an ordered, mechanistically legible way, and together with section 5 they
close the wall from both sides:

- **`spatial_doa` is the closest to break-even and still loses.** Encoding where each band's source sits
  (direction cosines plus DirAC diffuseness) is the richest spatial information the FOA format exposes,
  and two of its five seeds actually went positive (+0.018, +0.003), but the mean nets slightly negative
  (-0.0031, p = 0.6875), an order of magnitude short of the 0.05 SESOI. Direction of arrival tells the
  gate WHERE a source sits, not WHEN it onsets, so at matched budget it places fires no better than
  uniform random. It keeps the noisy-TV control at chance (4.5% on noise vs a 10.8% content base rate),
  so this is an honest no-signal null, not a noise-chaser.
- **`superflux_spectral` fails twice over.** SuperFlux is the one front-end designed as a sharper onset
  detector than the base half-wave-rectified flux, and its preregistered hypothesis was that sharper
  onset structure would raise onset F1. Instead it lost (mean -0.0182, p = 1.0) AND broke the
  noisy-TV-at-chance control (it fires 53.4% on pure noise versus a 37.5% content base rate). That is the
  tell: max-filtered positive spectral flux responds to any broadband energy transient, including noise,
  so its extra "onset structure" is partly a noise response rather than an onset response. A stronger
  generic transient detector is not a stronger onset localizer here.
- **`interchannel_coherence` loses cleanly.** All five paired seeds are negative (mean -0.0290, p = 1.0),
  the worst of the wave. Magnitude-squared coherence measures how phase-locked the omni and gradient FOA
  channels are, a spatial-directness cue close to orthogonal to onset timing; it keeps noisy-TV at chance
  (2.2% on noise vs a 10.1% content base rate) but simply carries no onset-localizing information, so it
  strictly loses to free random placement.

The wave's honest lesson: a richer frozen front-end does not carry onset-localizing information the
unchanged gate can exploit. Adding spatial direction, a stronger hand-crafted onset detector, or
interchannel coherence to the front-end does not move onset F1 past the SESOI and does not beat
rate-matched-random, and the one front-end built to sharpen onsets did so only by also chasing noise.
Section 5 showed that redesigning the gate's firing pattern does not clear the wall; this wave shows that
enriching the frozen representation the gate reads does not clear it either. The wall is therefore
neither in the front-end's feature richness nor in the gate's firing sparsity or spacing. As X0 and
section 3 both indicated, the missing capability is relational and temporal event interpretation (knowing
WHICH frame is an onset), which neither a frozen zero-parameter front-end nor a sub-4,096-parameter gate
supplies at matched compute. That is a proven wall with a mechanistic reason, which is a legitimate wave
outcome.

The substrate stays at Stage 2. The bed and its sealed referee did their job a third time: they
distinguished three honest frozen front-ends from free random placement and refused to manufacture a
positive from any of them.
