# Null card: trained-router capability-density null (real cache)

Registry-completion entry (axis-ceiling falsification lane). Form per BLACKHOLE.md:
no em or en dashes, no agency/understanding/intelligence language. The encoder is frozen and never
trained; only linear reader heads and a small gate are fit.

## Claim under test

The MoP density bet: a trained gate (router) that routes each clip to the best of several
heterogeneous frozen readers extracts a capability the readers do not share, and so beats a tuned
single-substrate baseline at matched compute. If true, heterogeneous routing is worth its extra
FLOPs on real V-JEPA / DINOv2 latents.

## Control (the tuned baselines the router must beat)

Target: shape under nuisance, 5-way, chance 0.20. Three real caches, identical 200 clips:
vjepa2_vitl_nuisance (1024d), dinov2s_nuisance_real (384d), vjepa2_vitl_singleframe (1024d). Split:
per-class 4 reader-train / 2 router-train / 2 eval. Two preregistered baselines, both TUNED:
  - R1 best single reader: the single strongest reader by eval accuracy (dino on all 10 seeds). The
    router arm runs 3 readers, so it spends about 3x the FLOPs; a TIE here is a density LOSS.
  - R2 homogeneous k-copy bank: k copies of the best reader, FLOP-matched and param-matched to the
    router arm (per-seed FLOP ratio 1.003, param ratio 1.006). This isolates heterogeneity from raw
    capacity: same compute, one substrate.
Preregistered null H0: the trained router ties or loses to at least one baseline at matched FLOPs.
Reject only if the router beats BOTH R1 and R2 with seed-CI lo > 0 and no per-seed sign flip. A tie
is a null. No threshold was tuned toward a win; the kill-switch was honored.

## Result (10 seeds)

  - router minus best-single: mean -0.010, CI [-0.0343, +0.0143], sign flips (n_pos 3 / n_neg 5 /
    n_zero 2, consistent_sign 0). Does NOT beat R1.
  - router minus homogeneous bank (FLOP+param matched): mean -0.016, CI [-0.0427, +0.0107], sign
    flips (n_pos 4 / n_neg 5 / n_zero 1, consistent_sign 0). Does NOT beat R2.
  - Gates: R1_beats_best_single = false, R2_beats_homo_bank_matched = false, R3_compute_matched =
    true. Both CIs straddle zero and both flip sign, so the null is not rejected on either arm.

Mechanism (why the null holds, not just that it holds): heterogeneity headroom is nearly absent on
the real cache. The oracle (any-of-3 readers correct) reaches 0.942 vs best-single 0.878, an oracle
gain of only 0.073 (independent re-run: 0.0733, CI [0.0504, 0.0963]). PR1's hand-built synthetic
sub-population mixture had an oracle gain of 0.1553, more than double. The three readers decode the
SAME emergent shape signal (dino best on all 10 seeds) with correlated per-clip errors (phi 0.27 to
0.71), so there is almost nothing for a gate to exploit. The complementarity PR1 manufactured is
largely a property of the synthetic mixture, not of real V-JEPA / DINOv2 latents.

## Why it is an asset

This is the density bet's own headline falsified on the cache it most wanted to win on. It converts
a promising synthetic result (PR1 oracle gain 0.155) into a bounded negative on real encoders (gain
0.073, router ties or loses at matched compute). The matched-FLOP homogeneous-bank control is the
decisive one: it strips capacity out of the comparison, so the tie cannot be explained away as "the
router just needed more compute." A registry that hides this and keeps only the synthetic win would
be a whitewash; showing it is the rigor working. It also sharpens the live claim: routing is worth
its cost only where reader errors are genuinely independent, which these caches are not.

```yaml
exp_id:            EX-ROUTER-DENSITY
title:             trained gate over 3 real-cache readers does not beat a tuned single or matched homogeneous bank
hypothesis:        a trained router exploits reader heterogeneity to beat a tuned baseline at matched compute
null_hypothesis:   H0: the trained router ties or loses to at least one baseline (best-single R1 or FLOP/param-matched homogeneous bank R2) at matched FLOPs; reject only if it beats BOTH with CI lo>0 and no sign flip
baseline:          R1 best single frozen reader (dino, per-seed argmax by eval acc) and R2 homogeneous k-copy bank of the best reader, FLOP-matched (ratio 1.003) and param-matched (ratio 1.006)
ablation:          matched-FLOP homogeneous bank isolates heterogeneity from capacity; oracle-gain headroom (0.073) contrasted against PR1 synthetic mixture (0.1553)
metric:            probe_acc
probe_dependency:
  factor:          identity
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes
  acc_above_chance: 0.678
encoder_scale:     L
seeds:
  n:               10
  sem:             0.0124
  sign_stability:  unstable (router-vs-best consistent_sign 0; router-vs-homo consistent_sign 0)
provenance_tag:    real-encoder
result:            router minus best-single mean -0.010 CI [-0.0343, +0.0143] flips; router minus matched homo bank mean -0.016 CI [-0.0427, +0.0107] flips; oracle gain 0.073 CI [0.0504, 0.0963] vs PR1 synthetic 0.1553
taxonomy_category: 3
verdict:           DOWNGRADE-TIE
badges:            [tuned-baseline-tie]
raw_run_id:        router_mechanism (real cache, 10 seeds); repo mirror runs/mot/router_mechanism.json; independent re-verify axis_falsification/verify_nulls.py (oracle gain 0.0733 CI [0.0504, 0.0963])
repro_level:       R2
```
