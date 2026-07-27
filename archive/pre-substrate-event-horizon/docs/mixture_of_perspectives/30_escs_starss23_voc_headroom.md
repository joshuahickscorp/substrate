# ESCS STARSS23 Value-of-Computation Headroom Instrument and Its First Result

> **Append-only Stage-3 instrument.** This is a measurement instrument, not a fourth value-of-computation
> bed. It sits beside the three sealed STARSS23 value-of-computation beds (onset-localization, walled,
> [doc 26](./26_escs_starss23_bed.md); source-counting, a signal that did not survive reproduction,
> [doc 27](./27_escs_starss23_counting_bed.md) / [doc 28](./28_escs_starss23_counting_reproductions.md);
> direction-of-arrival, a clean double null, [doc 29](./29_escs_starss23_doa_bed.md)) and asks a question
> none of them measured directly: how much value-of-computation HEADROOM does each corpus target actually
> contain, before any trained gate is involved? It is net-new only: five `vochead_*` modules under
> `src/mop/beds/starss23/`, one runner, one test module, and the sealed `proof/STARSS23_VOC_HEADROOM.json`,
> `.prereg.json`, and `.verification.json`. It edits no sealed onset, counting, or DoA module and no
> existing proof; it imports the sealed count and DoA label and estimator modules read-only. It touches no
> live campaign or process file.

- **Status:** built and run once on the real MIT STARSS23 subset, plus two self-contained synthetic control
  targets. **Independently verified**, with the producer and a stdlib-only verifier reproducing every
  number and a third, separately written adversarial recompute confirming the two headline shapes.
- **Snapshot date:** 2026-07-18
- **Dataset:** STARSS23 (zenodo.org/records/7880637), MIT license; the same fixed real 4-channel FOA subset
  and native fold-4 test split the counting and DoA beds used.
- **Claim scope:** descriptive corpus characterization of value-of-computation headroom. No activation, no
  scientific promotion, no natural-world generality claim. The informed reference is a label-aware upper
  bound, not a demonstration of any trained gate. `independent_scientific_confirmation` is `false` in the
  sealed artifact and in the verification artifact, set by neither producer nor verifier.

## 1. What was measured and why

Three value-of-computation beds on this corpus have run to a conclusion and all nulled. A value-of-
computation bed pits a trained WHEN gate, deciding when to spend a re-estimation, against a rate-matched-
random control that spends the identical budget at random. The trained gate never beat rate-matched-random.
Three nulls invite one question the beds never answered head on: is there any headroom to win at all, and
if so, of what shape?

This instrument decomposes the question with policies that bracket what ANY WHEN gate could do at a matched
budget, per target, using only the labels and the frozen estimator (no trained gate):

- **`always_on`** re-estimates every frame: the perfect-WHEN unlimited-budget ceiling. Its coasted error is
  the frozen estimator's own fresh error.
- **`never_update`** never re-estimates: coast the fixed cold-start forever, the zero-budget floor.
- **`informed_change_aligned`** is a label-AWARE reference: starting from `never_update`, greedily add the
  change frame whose re-estimation most reduces total coasted error, up to budget K. It is a strong
  achievable WHEN policy over the change-frame candidates, an upper reference, NOT a proven global optimum,
  and it is stated as such throughout. It knows the ground truth, so it upper-bounds any label-blind gate.
- **`rate_matched_random`** places the same K re-estimations at random, averaged over 32 deterministic
  host-reproducible draws: the exact control the sealed beds use.

Two derived quantities carry the finding:

- the **refreshable range** = `never_update - always_on`. If this is NEGATIVE, re-estimating the frozen
  estimator is worse, in aggregate, than coasting a constant: the WHAT floor has collapsed and no WHEN
  policy can rescue it. This is a mechanistically distinct failure shape from "the gate could not learn the
  WHEN signal."
- the per-budget **headroom** = `rate_matched_random - informed_change_aligned` (positive means the
  label-aware reference beats random at that matched budget).

The whole plan, including the interpretation classification thresholds, is preregistered in code at
`proof/STARSS23_VOC_HEADROOM.prereg.json` before any corpus number is read, so the labels
`what_floor_collapse` / `real_headroom` / `no_headroom_budget_saturated` cannot be tuned after the fact. A
tie is a null.

## 2. The two real targets and the synthetic controls

- The **count** target reuses the sealed counting bed's frozen zero-parameter eigenvalue-rank estimator and
  its per-frame concurrent-source-count ground truth (a strong WHAT: the target that produced the program's
  only real value-of-computation signal). Metric: absolute integer count error per frame, scoring every
  frame.
- The **doa** target reuses the sealed DoA bed's frozen zero-parameter wideband active-intensity estimator
  and its per-frame ground-truth direction (a weak WHAT). Metric: great-circle degrees per active frame.
- Two **synthetic controls** are self-contained count-metric toys: one with a perfect fresh estimator
  (`est == gt`, a known-strong WHAT) and one with a maximally wrong estimator (a known-harmful WHAT). The
  instrument must report `real_headroom` for the first and `what_floor_collapse` for the second, proving it
  is not rigged to a single label. It does (`synthetic_control_ok: true`).

## 3. The sealed result, honestly

Real STARSS23, fold-4 test scope (21 clips, 7 rooms). Budgets are re-estimation counts as a fraction of a
clip's frames, from the tight regime through the loose regime the sealed beds operated in.

**Count target: `real_headroom`.** Refreshable range `+0.498` (never_update 1.363, always_on 0.865). The
label-aware informed reference reaches a coasted count error of about `0.349` at every swept budget, which
is lower than both rate-matched-random (0.888 to 1.001) and, notably, lower than `always_on` (0.865): the
count estimator is noisier per frame than the true piecewise-constant count, so refreshing selectively at
changes and coasting a good sample through a stable run beats re-sampling noisily every frame.

| Count, fold-4 (lower error is better) | budget 0.5% | 1% | 2% | 5% | 10% |
| --- | --- | --- | --- | --- | --- |
| informed_change_aligned (label-aware ceiling) | 0.355 | 0.349 | 0.349 | 0.349 | 0.349 |
| rate_matched_random (control) | 1.001 | 0.947 | 0.926 | 0.903 | 0.888 |
| headroom (rmr minus informed) | +0.646 | +0.598 | +0.577 | +0.554 | +0.539 |

**DoA target: `what_floor_collapse`.** Refreshable range `-36.86` (never_update 76.12, always_on 112.97):
re-estimating the active-intensity estimator every frame (about 113 degrees of coasted error) is worse than
coasting a fixed boresight (about 76 degrees). The estimator is anti-informative in aggregate on this
corpus. The informed reference does reach a lower error (about 55 degrees) by label-aware cherry-picking of
the rare frames where the broken estimator happens to be near the truth, but that is not realizable headroom:
a label-blind gate cannot know which of the estimator's outputs are accidentally correct, and the honest
aggregate signal is the negative refreshable range. The full-subset scope (45 clips) reproduces both shapes
(count range `+0.373` real_headroom; DoA range `-37.98` what_floor_collapse).

## 4. What kind of finding this is

The three sealed nulls decompose into two mechanistically distinct shapes, and this instrument names them:

- **Onset-localization and direction-of-arrival are WHAT-floor collapses.** For DoA the refreshable range is
  negative: refreshing the frozen estimator is worse than a constant, so no WHEN gate, however well trained,
  could have won. The DoA null (doc 29), which already flagged the estimator's own signal floor as a suspect,
  is here quantified: the floor did not just limit the gate, it inverted the value of refreshing.
- **Source-counting has real headroom that the trained gate under-realized.** For count the refreshable range
  is clearly positive and the label-aware ceiling beats random by a large, budget-robust margin (0.54 to 0.65
  count-MAE), on 21 of 21 clips. This is consistent with the counting bed's history: its signal was real but
  fragile (doc 28), dying on scoring-unit and gate-architecture reproduction axes, not on absence of headroom.
  The mechanism can win on this target if a gate learns to place re-estimations at count changes robustly;
  the headroom is not the obstacle.

This reframes the program's read of the value-of-computation lane. It is not uniformly dead on STARSS23: it
is WHAT-floor-bounded on the direction and onset targets and headroom-rich but under-realized on the count
target. The decision-relevant next question is therefore narrower and better posed: a gate that realizes the
measured count headroom robustly across scoring unit and architecture, not another target whose WHAT floor
has already collapsed. This is a bounded instrument reading, `consistent with` these shapes, never a
demonstration that any gate wins.

## 5. Independent verification

A separately authored verifier (`vochead_verifier.py`, stdlib-only, importing no `mop` module and no
headroom producer module) re-implements the canonical seal, the coasting, both per-frame metrics, all four
policies, the rate-matched-random draw discipline, the clip-macro aggregation, the refreshable range, and
the interpretation classifier, then reproduces every sealed number and label.

```
seal_intact: true
targets_reproduced: true
interpretation_reproduced: true
honesty_ok: true
independent_referee_reproduction: true
independent_scientific_confirmation: false
mismatches: []
synthetic_control_ok: true
```

A third, separately written adversarial recompute (a non-incremental full-rescore greedy, a different code
path from both the producer's and the verifier's incremental greedy) re-derived the two refreshable ranges
from the sealed raw tracks, confirmed the greedy reference is monotone non-increasing on every clip, and
confirmed the count `informed beats random` result survives a different random seed base and four times the
draws, on 21 of 21 clips (a one-sided clip-level sign test p about 4.8e-7). All 16 unit and end-to-end tests
pass (`tests/unit/test_starss23_voc_headroom.py`).

`independent_scientific_confirmation` is `false` and remains unset: this is a descriptive instrument, the
informed reference is a label-aware upper bound rather than a trained gate, and no single run of a producer
plus a verifier authored in one session can confer scientific confirmation.

## 6. Additive-only and rights posture

- No sealed onset module, no sealed `count_*` module, and no sealed `doa_*` module was edited; the
  instrument imports the count and DoA label and estimator modules read-only.
- No existing proof was modified. `proof/STARSS23_VOC_HEADROOM.json`, `.prereg.json`, and
  `.verification.json` are net-new.
- All five `vochead_*` modules under `src/mop/beds/starss23/` are net-new; none is imported by, and none
  imports, any live campaign or orchestration path. Nothing under `runs/` or `studies/` was read or written.
- Real, rights-clean, room-disjoint STARSS23 data throughout (`source_kind: "real"`, `rights_clean: true`);
  `activation_allowed`, `scientific_promotion`, and `independent_scientific_confirmation` are hardcoded
  `false` in the sealed artifact.
