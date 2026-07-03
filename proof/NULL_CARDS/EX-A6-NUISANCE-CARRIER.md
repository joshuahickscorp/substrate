# Null card: A6 cross-modal shared code is nuisance geometry, not abstract semantics

Registry-completion entry (axis-ceiling falsification lane). Form per BLACKHOLE.md:
no em or en dashes, no agency/understanding language. Encoders are frozen; only linear ridge maps
and train-fit residualizers are computed.

## Claim under test

The MoP shared-code north star: two perspectives (a frozen vision encoder and a text/descriptor
column) share a metric geometry BEYOND the discrete label partition, i.e. their neighbor graphs
still align after the known generative factors are projected out. al2 flagged
vjepa2_vitl_nuisance -> qwen05b_textified_real as "genuine-shared-structure". This card asks WHAT
carries that alignment.

## Control (partial-out ladder, train-fit, no test leakage)

Metric: al2 rank-k ridge map, kNN neighbor-recall on the test split, permuted-pairing floor of equal
rank; delta = learned recall minus permuted recall; a pair "shares structure" only when the delta
seed-CI lower bound is strictly above zero with no per-seed sign flip (al2 classify_pair). Conditions
project out, on TRAIN only, the column space of a per-condition design: raw / minus_color /
minus_shape_color / minus_nuisance (the 6 factors r, x0, y0, vx, vy with rotation as sin/cos) /
minus_all. Preregistered nulls: N1 color-carried, N2 label-partition-only, N3 nuisance-geometry-
carried, N4 nothing-beyond-named-factors. 10 seeds, ranks 8 and 32 (primary 32), 4 pairs.

Residualizer VALIDITY was audited separately on a held-out split (a6_control_residualization_audit):
minus_shape_color drives vision shape decodability from 0.85 (vjepa) / 0.86 (dino) down to 0.165 /
0.172, at chance 0.20; minus_nuisance drives every one of the 6 nuisance R2 values to within +/-0.04
of zero on test. So the partial-outs demonstrably remove what they claim, and the collapse below is
not a too-weak-removal artifact.

## Result (10 seeds, primary rank 32)

Stable genuine-shared-structure survivors per condition: raw 3, minus_color 4, minus_shape_color 2,
minus_nuisance 0, minus_all 0.
  - N2 (label-partition-only) REJECTED: alignment survives removing shape+color for the pixel-text
    pairs (vjepa->text and vjepa->handcrafted stay genuine-shared-structure through minus_shape_color).
  - N3 and N4 HOLD: removing the 6 nuisance factors sends EVERY pair to non-replicating or alignment-
    artifact (0 survivors), and minus_all is likewise 0.
Carrier verdict: the surviving cross-modal alignment is the spatiotemporal NUISANCE geometry
(position, size, orientation, motion), NOT the semantic (shape, color) label partition and NOT any
abstract code beyond the named generative factors. The "shared code" is the shared pose of the same
rendered scene, visible to both a vision encoder and a pixel-derived caption.

## Companion: shape-axis bounding null (a6 ... shapecap)

The strongest possible version of the claim uses a caption cache that genuinely carries shape
(qwen05b_shapecap_real, real shape linear-decode 0.6167 > chance 0.20, so the testability kill-switch
did NOT fire). Preregistered decisive rule: alignment SURVIVES only if a vision->shapecap pair is
"genuine-shared-structure" under minus_all; anything else COLLAPSES. Result: both vision->shapecap
pairs are alignment-artifact under minus_all (survivors 0). Even a shape-carrying caption shares no
abstract geometry with vision once shape+color+nuisance are removed, exactly as the color-grid
textification collapsed. This bounds the null on the axis where it was most likely to break.

## Why it is an asset

This demotes the shared-code north star from "vision and text share an abstract semantic geometry"
to the mundane, honest reading: they share the pose of the same synthetic scene. The rejection of N2
is kept (the alignment is more than the color channel), so the card is not a strawman kill; it is a
precise carrier attribution. The separately audited residualizer makes the collapse unimpeachable:
the factors really were removed on held-out data, so "collapses under minus_nuisance" cannot be
dismissed as a weak control. The shapecap bounding null closes the obvious rebuttal ("your text side
just could not see shape"). Publishing this instead of the raw al2 "genuine-shared-structure" flag is
the falsification engine working: a real over-claim demoted.

```yaml
exp_id:            EX-A6-NUISANCE-CARRIER
title:             cross-modal vision-text neighbor alignment is carried by nuisance pose, not abstract shape or color
hypothesis:        a frozen vision encoder and a text/descriptor column share metric geometry beyond the discrete label partition
null_hypothesis:   N3 nuisance-geometry-carried (after removing r, x0, y0, vx, vy, rot the alignment is within the permutation floor) and N4 nothing-beyond-named-factors (after removing shape+color+nuisance, within the floor)
baseline:          same rank-k ridge map refit on a row-shuffled target pairing (topology permutation null); residualizer coefficients fit on TRAIN only; kNN recall on held-out test
ablation:          partial-out ladder raw / minus_color / minus_shape_color / minus_nuisance / minus_all at rank 8 and 32; residualizer validity audited on held-out (shape 0.85->0.165 at chance 0.20; nuisance R2 to +/-0.04)
metric:            recall_at_k
probe_dependency:
  factor:          cross_modal_correspondence
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/cross_modal_correspondence.json
  decodable:       marginal
  acc_above_chance: null
encoder_scale:     L
seeds:
  n:               10
  sem:             null
  sign_stability:  stable at S>=3 (minus_nuisance and minus_all: 0 survivors across all pairs, no genuine-shared-structure verdict)
provenance_tag:    structured-synthetic
result:            survivors per condition raw 3 / minus_color 4 / minus_shape_color 2 / minus_nuisance 0 / minus_all 0; carrier = nuisance geometry; N2 rejected, N3 and N4 hold; shapecap minus_all survivors 0 (bounding null); residualizer audit conclusion_survives = true
taxonomy_category: 3
verdict:           SUBSTRATE-BOUND
badges:            [substrate-blindspot]
raw_run_id:        runs/mot/a6_residual_alignment.json (base, carrier_verdict nuisance-geometry) + runs/mot/a6_residual_alignment_shapecap.json (shape-axis COLLAPSES) + runs/mot/a6_control_residualization_audit.json (removal validity); clipset bound_nuisance_v1
repro_level:       R2
```
