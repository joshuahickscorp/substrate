# Null card: facet-12 predictor-rollout fidelity null (real V-JEPA 2 predictor)

World-model predictor lane (facet 12), first drive of V-JEPA 2's PREDICTOR rather than its encoder.
Form per BLACKHOLE.md: no em or en dashes, no agency/understanding/intelligence language. The
encoder is frozen and never trained; the predictor is loaded frozen from the HF cache and only a
linear ridge readout is fit.

## Claim under test

The rollout lane bet: the frozen V-JEPA 2 predictor, rolled open-loop through several temporal-slot
lookaheads, stays faithful enough to be a usable latent world model, that is faithful enough to
support multi-step latent planning and counterfactual/interventional abstraction (the ex2 latent
planning precursor). If true, facet 12 moves off 0.

## Control (the bar the real predictor must clear)

24 synthetic bound-nuisance clips, 32 temporal slots by 16x16 spatial grid (256 patches per slot),
3 seed buckets, T_START = 4, horizons h in {1,2,3,4,6,8}. Three preregistered non-vacuous controls:
persistence (copy last ground-truth context slot, no dynamics), random_init (same predictor
architecture, freshly initialized weights, identical pipe), shuffled_target (real predictor output
scored against a different clip's true target slot, sharing the predictor-vs-encoder
representational gap). Preregistered usable rule: largest contiguous h from h=1 where
real_nmse.hi < every control_nmse.lo (non-overlapping seed CI) AND real_nmse.mean < 0.5 *
best_control_nmse. A directional-but-sub-usable signal is a NULL; a tie is a NULL. Instrument is
bit-exact against the top-level VJEPA2Model path (pred max|diff| 0.0, teacher max|diff| 0.0) and
independently re-derived on disjoint seeds by verify_facet12.py (all six checks pass).

## Result (24 clips, 3 seed buckets, CPU)

Real beats all three controls by a non-overlapping seed CI at EVERY horizon (a genuine,
sign-stable, adversarially verified world-model signal), but never approaches the 0.5 usability bar:
the real/best-control nmse ratio is 0.931 (h=1), 0.947 (h=2), 0.954 (h=3), 0.954 (h=4), 0.950
(h=6), 0.989 (h=8), a margin of only about 5 to 7 percent that collapses to about 1 percent by h=8.
usable_horizon = 0. The leakage probe (V3) shows the roughly 0.77 one-step error is mostly a
predictor-vs-encoder representational gap (in-context nmse 0.750, future h=1 nmse 0.775), so the
marginal one-step forecast cost is only about 0.025 nmse; the gap does not rescue usability, it is
itself a reason encoder-space rollout is hard. Wave 2 (task-relevant criterion: does the rollout
keep the moving object's position decodable) returned a second null, WALL null-by-ill-posedness:
under the encoder-trained readout the rollout decodes position at the random/shuffled floor
(position R2 0.09/0.06/0.06/0.07/0.05/-0.005 at h=1/2/3/4/6/8) versus persistence
0.35/0.33/0.35/0.23/0.29/0.21 and true ceiling 0.40/0.37/0.43/0.41/0.42/0.32, so retention vs
persistence is negative at every horizon. The adversarial re-derive falsified the build's own motion
premise: true object displacement is sub-patch at every horizon (median 0.60 to 4.84 px; h=8 is 0.30
patch), so the synthetic clipset cannot pose the motion-tracking question (motion_testable = FALSE).

## Why it is an asset

This is the first time the corpus drives the predictor as a world model, and it converts the lever
from unmeasured to measured with a preregistered, bit-exact, adversarially verified instrument and a
decisive number: the real predictor carries a real signal and is far below usable fidelity on this
clipset, so the rollout / ex2 latent-planning lane is NOT licensed here. The new mechanistic finding
survives adversarial re-derive: object position SURVIVES the compounded rollout but the predictor
writes it into a sub-space the encoder-trained head cannot read zero-shot (in-domain probe R2 0.73
vs encoder-trained 0.09 at h=1), which names a concrete Studio fix (a per-representation-space
readout adapter) rather than a dead end. The verdict is provisional on content: the clips are
synthetic and out of distribution for a predictor trained on real video, and motion is intrinsically
sub-patch, so the licensed re-test is real moving video with supra-patch motion (facet 14 corpora)
decoded through a readout adapter fit on rollout latents. No axis score is moved on synthetic
ill-posed content.

```yaml
exp_id:            FACET12-ROLLOUT-FIDELITY
title:             frozen V-JEPA 2 predictor rollout beats every control at each horizon but never reaches the usability bar
hypothesis:        the frozen predictor rolled open-loop stays faithful enough to be a usable latent world model for multi-step planning
null_hypothesis:   H0: the real predictor never reaches the usability bar (real_nmse.hi < every control_nmse.lo AND real_nmse.mean < 0.5 * best_control_nmse) at any contiguous horizon from h=1; a tie is a null
baseline:          persistence (copy last ground-truth context slot), random_init (same predictor architecture, fresh weights, identical pipe), shuffled_target (real output vs a different clip's true target slot, sharing the representational gap)
ablation:          leakage probe V3 (in-context nmse 0.750 vs future h=1 0.775) isolates the predictor-vs-encoder representational gap from forecast failure; wave-2 in-domain vs encoder-trained position probe (R2 0.73 vs 0.09 at h=1) isolates sub-space shift from content destruction
metric:            rollout_error
probe_dependency:
  factor:          position (object centroid cx, cy; the content the rollout lane actually needs)
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       facet12b decodability calibration (scratchpad/facet12b/decodability_retention.py); true-latent ceiling R2 0.40/0.37/0.43/0.41/0.42/0.32 at h=1/2/3/4/6/8, persistence floor 0.35/0.33/0.35/0.23/0.29/0.21
  decodable:       marginal
  acc_above_chance: 0.05
encoder_scale:     L
seeds:
  n:               3
  sem:             null
  sign_stability:  stable at S>=3 (real beats all three controls by non-overlapping seed CI at every horizon; real_nmse ci_half 0.0003 to 0.003)
provenance_tag:    provisional
result:            real/best-control nmse ratio 0.931/0.947/0.954/0.954/0.950/0.989 at h=1/2/3/4/6/8, margin about 5 to 7 percent, usable_horizon 0; wave-2 position retention vs persistence negative at every horizon (encoder-trained R2 0.09 at h=1 vs persistence 0.35); motion sub-patch (median 0.60 to 4.84 px, h=8 is 0.30 patch), motion_testable FALSE
taxonomy_category: 3
verdict:           DOWNGRADE-TIE
badges:            [substrate-blindspot]
raw_run_id:        runs/mot/dr13_predictor_fidelity.json (facet 12, verdict null, elapsed 716.4s); wave-2 method scratchpad/facet12b/ (decodability_retention.py, adversarial_pass.py, motion_validation); instrument scripts/mop_dr13_predictor_fidelity.py; write-up docs/mixture_of_perspectives/RESULTS_LEDGER.md
repro_level:       R2
```

## What this closes

It closes the synthetic-clipset question for the rollout / world-model predictor lane: the frozen
V-JEPA 2 predictor is a real but weak simulator of this content and is not licensed for multi-step
latent planning or counterfactual rollout at usable fidelity, so facet 12 stays at 0 on usability
grounds. It leaves open exactly one licensed next step, the Studio re-test on real moving video with
supra-patch object motion (facet 14 feeds facet 12) decoded through a readout adapter, since the
synthetic clipset is out of distribution for the predictor and its motion is intrinsically sub-patch.
