# Null card: substrate-special single-split p-value is fragile (demoted headline to fragility-HOLD)

Registry-completion entry (axis-ceiling falsification lane). Form per BLACKHOLE.md:
no em or en dashes, no agency language. Encoders frozen; only linear reader heads and resampling
statistics are computed.

## Claim under test

Substrate-special headline: a real pretrained V-JEPA ViT-L decodes shape at 256px far better than a
random-init same-architecture ViT-L (0.517 vs 0.241), and the difference on the reported 29-clip test
split (V-JEPA 15/29 correct, random-init 7/29) is significant at one-sided Fisher p = 0.0285. Read as
a headline, this says pretraining, not architecture, carries the shape signal, with p < 0.05.

## Control (single-split fragility, preregistered before reading any number)

The p = 0.0285 rests on ONE 29-clip split. The adversarial control resamples the 29 test clips WITH
replacement (B = 10000, seed 0) and recomputes the one-sided Fisher p on each resample, reporting the
fraction of resamples that keep p < 0.05 (the split's robustness) plus a one-clip-swing sensitivity.
A direction check is added on the on-disk 200-clip vjepa vs randominit_vitl_nuisance caches (a
DIFFERENT, larger split we actually possess; NOT a new Studio multi-seed re-encode).
Preregistered thresholds (a tie is a null): HARDEN iff > 90 percent of resamples keep p < 0.05; HOLD
iff > 50 percent; DEMOTE iff < 50 percent (the single split is a coin-flip artifact).

## Result

  - Bootstrap: only 63.7 percent of resamples keep p < 0.05. Median resampled p = 0.0285, but the
    90th-percentile p = 0.1372, well above 0.05.
  - One-clip-swing: a single adverse clip (V-JEPA 14/29, random-init 8/29) pushes p to 0.0877, across
    the 0.05 line. A single favorable clip drops it to 0.007. The headline p is one clip from
    non-significance in either direction.
  - Direction check (200-clip on-disk caches): V-JEPA 0.793 vs random-init 0.243 shape accuracy,
    delta mean 0.550, CI [0.5041, 0.5959], no sign flip across 10 seeds. The DIRECTION of the
    pretraining gap is solid; the SINGLE-SPLIT p-value is not.
By the preregistered rule this lands in the HOLD band (63.7 percent, between 50 and 90). The headline
"p = 0.0285" is demoted: the single-split significance is fragile, corroborated in direction only.

## Why it is an asset

This is the workflow demoting one of its own strongest-looking positives. The point estimate looked
publishable (p under 0.05); the bootstrap shows better than a third of equally valid resamples of the
same test set would not clear 0.05, and one clip flips the verdict. Banking "p = 0.0285" as a clean
win would have been the whitewash; recording it as a fragility-flagged HOLD, with the honest split
between a solid direction (200-clip delta CI lo 0.504) and a fragile single-split p, is the rigor
working. It also states precisely what would resolve it: a Studio multi-seed re-encode at 256px
(B5), not more resampling of the same 29 clips. A demotion of a real over-claim raises the
falsification-engine axis; this is that demotion, done to the program's own headline.

```yaml
exp_id:            EX-SUBSTRATE-SPLIT-FRAGILITY
title:             substrate-special single-split Fisher p is fragile, held on direction only
hypothesis:        pretraining not architecture carries the 256px shape signal, significant at one-sided Fisher p<0.05 on the reported 29-clip split
null_hypothesis:   the single 29-clip split significance is a coin-flip artifact: fewer than 50 percent of with-replacement resamples of the same test clips keep p<0.05
baseline:          random-init same-architecture ViT-L on the identical clips (0.241 at 256px; 0.243 shape acc on the 200-clip on-disk cache); Fisher exact one-sided, not a strawman t-test
ablation:          B=10000 test-clip bootstrap (frac p<0.05, median, 90th pct); one-clip-swing sensitivity; 200-clip on-disk direction check across 10 seeds
metric:            probe_acc
probe_dependency:
  factor:          identity
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes
  acc_above_chance: 0.317
encoder_scale:     L
seeds:
  n:               10
  sem:             0.0234
  sign_stability:  stable at S>=3 on DIRECTION (200-clip delta consistent_sign 1); single-split p FRAGILE (63.7 pct bootstrap keep p<0.05)
provenance_tag:    real-encoder
result:            reported single-split Fisher p=0.0285 (15/29 vs 7/29); bootstrap frac p<0.05 = 0.637, median 0.0285, 90th pct 0.1372; one-clip adverse swing p=0.0877; 200-clip direction delta 0.550 CI [0.5041, 0.5959] no flip; verdict HOLD (fragility-flagged)
taxonomy_category: 6
verdict:           SEED-UNSTABLE
badges:            [seed-instability]
raw_run_id:        reaudit T4 (bootstrap B=10000 seed 0); 200-clip direction check on data/cache/vjepa2_vitl_nuisance vs randominit_vitl_nuisance; runs/mot mirror survivor_reaudit T4_substrate_special
repro_level:       R2
```
