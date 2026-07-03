# A6 Result: What the Cross-Modal Shared Code Actually Carries

## 1. Headline (honest)

Vision and language embeddings on the bound-nuisance clipset do share stable geometric
structure, and that structure SURVIVES projecting out the semantic (shape, color)
abstraction, but it COLLAPSES the moment the six spatiotemporal nuisance factors
(position, size, orientation, motion) are projected out. The cross-modal shared code is
nuisance geometry (where / how-big / oriented-how / moving-how), NOT the semantic label
partition. This carrier conclusion survived three adversarial controls (a random-dimensionality
control, a floor-versus-learned decomposition, and a residualization correctness audit).
The separate shape-axis bet, which asked whether a caption that genuinely carries shape
would share abstract shape geometry with vision, returned a clean BOUNDING NULL: the
shape-caption was fully testable (killswitch did not fire) yet vision-to-shapecap alignment
still died at the permutation floor once shape+color+nuisance were all removed. No
first-genuine cross-modal semantic-abstraction result. The nuisance-geometry conclusion is
the standing result.

## 2. The corrected premise

The earlier program narrative claimed the Qwen text cache was "text-of-labels" and therefore
that any vision-to-text alignment was trivially explained by both sides seeing the class label.
That premise is FALSE. `scripts/cache_qwen_textified.py` textifies from PIXELS ONLY: a 4x4
palette-color grid plus the brightest-cell position. Labels never enter the text. So the Qwen
text substrate is a label-free, pixel-derived featurization, and any alignment it shares with
vision cannot be a label short-circuit.

That correction sharpens the real question, because shape behaves very differently from color
under cheap featurization. Shape is emergent from large-scale pretraining and is not
recoverable by cheap label-free features; color is recoverable everywhere.

Linear-decode accuracy on the identical 200 clips (shape chance = 0.20):

| substrate                     | shape decode | reading                                  |
|-------------------------------|-------------:|------------------------------------------|
| dinov2s_nuisance_real (image) |         0.87 | shape emergent from pretraining          |
| vjepa2_vitl_nuisance (video)  |         0.78 | shape emergent from pretraining          |
| qwen05b_textified_real (text) |         0.267| at chance: color-grid text carries no shape |
| handcrafted_descriptors (HOG) |         0.217| at chance: cheap pixel features carry no shape |

Color, by contrast, decodes from 0.52 to 1.0 across every substrate. So shape is the hard,
pretraining-dependent axis and is exactly the axis a nuisance-only explanation would predict
the label-free text CANNOT share with vision.

## 3. Residual-alignment result

Method: the AL2 topology metric (ridge fit, rank truncation, kNN neighbor-recall delta versus
a permutation floor, `classify_pair` verdicts) run on vision-to-text pairs under five
residualization conditions (raw, minus_color, minus_shape_color, minus_nuisance, minus_all).
The residualizer projects out a design matrix's column space, fit on the TRAIN split only,
rotation entered as sin/cos. Primary rank 32, seeds 0-9. Only `genuine-shared-structure`
(seed-CI lower bound strictly > 0 and no sign flip) clears the null; a sign flip is
`non-replicating`; sitting at the floor is `alignment-artifact`.

Stable-genuine survivors per condition (across all four base pairs):

| condition          | survivors | primary-pair delta (vjepa2 -> qwen text) |
|--------------------|----------:|-----------------------------------------:|
| raw                |         3 | 0.0843 (genuine)                         |
| minus_color        |         4 | 0.0754 (genuine)                         |
| minus_shape_color  |         2 | 0.0649 (genuine)                         |
| minus_nuisance     |         0 | 0.0250 (non-replicating)                 |
| minus_all          |         0 | 0.0140 (non-replicating)                 |

Reading: alignment persists through removal of shape+color (the primary pair stays genuine at
0.0649, and vjepa2 -> handcrafted actually holds at 0.0716), then goes to ZERO stable-genuine
survivors across all four pairs once the six nuisance factors are projected out. The shared
code is spatiotemporal nuisance geometry, not the semantic abstraction.

Preregistered nulls (fixed in code before running), all FALSE: N1 color-carried, N2
label-partition-only, N3 nuisance-carried-all-pairs, N4 all-named-factors-all-pairs.

### Adversarial controls this conclusion survived

1. Random-dimensionality control (`a6_control_random_dim`). Tests whether the collapse is
   merely a dimensionality artifact of projecting out ANY 17 directions. Projecting out 17
   random Gaussian columns retains 85.5% of the raw delta (0.0721, still genuine); projecting
   out 17 structurally-matched permuted-one-hot columns retains 80.0% (0.0674, still genuine);
   projecting out the 7 REAL nuisance directions collapses to 0.025 (non-replicating). The
   preregistered survive rule (both random variants genuine, delta >= 60% of raw and >= 1.5x
   nuisance, and minus_nuisance a non-survivor) was met. The collapse is the SPECIFIC nuisance
   geometry, not the column count. SURVIVES.

2. Floor-versus-learned decomposition (`a6_control_floor_decomp`). Tests whether the collapse
   is real (learned recall falling) or floor inflation (permutation floor rising to meet a
   still-high learned recall). The permutation floor is essentially flat across conditions
   (swing ~0.009). 90.2% of the raw->minus_nuisance collapse and 102.3% of the raw->minus_all
   collapse is learned recall falling, not floor rising. SURVIVES.

3. Residualization correctness audit (`a6_control_residualization_audit`). Tests, on a held-out
   split with coefficients fit on train only, that the partial-out actually removes what it
   claims. No train->test leakage. minus_shape_color drives held-out shape decode on the vision
   columns to chance (vjepa2 0.850 -> 0.165, dinov2 0.860 -> 0.172, chance 0.20), so "survives
   removing shape+color" is not overstated. minus_nuisance drives all six factors to near-zero
   held-out R^2 for every column, so the collapse is genuine nuisance removal, not a weak
   partial-out. SURVIVES.

### Caveats the controls raised (do not round up)

- Under minus_nuisance the primary pair lands at `non-replicating` (sign-unstable), NOT at the
  fully-floored `alignment-artifact`. The floor decomposition confirms residual learned recall
  is still marginally above the raw floor (0.2108 vs 0.1800 at rank 32). So the honest phrasing
  is "collapses to a non-survivor / degrades to instability," not "extinguished to zero." A
  linear 7-column design cannot remove nonlinear nuisance geometry, so a thin residual is
  expected. It does not clear the null and does not rescue a semantic-abstraction reading.
- The reverse direction qwen text -> vjepa2 is weak and non-replicating even at raw
  (delta 0.0086, a sign flip across seeds). The signal is carried by the forward vision->text
  direction. This does not affect the carrier conclusion.

## 4. Shape caption + shape-axis bet

To test the semantic axis directly, `scripts/cache_qwen_shapecap.py` builds a LABEL-FREE shape
caption: it segments the foreground object per frame (background = mean of corner pixels,
largest connected component via pure-torch label propagation), computes 13 color-independent
shape invariants (7 sign-preserving log Hu moments plus circularity, eccentricity, extent,
log-area, aspect), verbalizes them into a "shape report ..." caption, and appends the reused
palette-color grid. Labels never touch the text.

This caption succeeds where every prior label-free substrate failed. Shape linear-decode from
the LLM mid-layer state: real 0.6167 versus chance 0.20 (randominit 0.50, color 0.45). The
preregistered killswitch (KILLSWITCH_SHAPE_ACC = 0.30) did NOT fire. The shape axis is now
linguistically testable, and real beats randominit (0.62 vs 0.50), a modest but genuine
pretraining lift over identical text.

The bet: does a caption that genuinely carries shape share abstract shape geometry with vision
beyond the six nuisance factors? Preregistered rule (fixed in code before running): SURVIVES
iff a vision->shapecap pair is `genuine-shared-structure` at minus_all; a minus_all that merely
ties the floor is a COLLAPSE, never rounded up.

Outcome: BOUNDING NULL. `shape_axis_alignment_survives_minus_all = false`. Both vision->shapecap
pairs land at `alignment-artifact` at minus_all: vjepa2 -> shapecap mean_delta +0.0091
(sign flips, CI crosses 0), dinov2 -> shapecap -0.0007 (at floor). The stronger pair,
vjepa2 -> shapecap, holds genuine all the way through minus_nuisance (raw +0.0518 ->
minus_nuisance +0.0224) and then dies at minus_all (+0.0091). So even a caption that
demonstrably carries shape shares NO abstract shape geometry with vision once the six nuisance
factors are also removed. The prior working conclusion survives its own strongest adversarial
test. This is a clean bounding null, not a moot result: the killswitch did not fire, the axis
was real and testable, and the alignment still collapsed.

## 5. A6 verdict and the DR1 handoff

Verdict: the cross-modal shared code on this clipset is spatiotemporal NUISANCE geometry, not
the semantic (shape, color) abstraction. Alignment survives removing shape+color and collapses
removing nuisance; three adversarial controls confirm the collapse is specific, learned-driven,
and correctly measured; and a caption that genuinely carries shape still shares no abstract
shape geometry with vision beyond nuisance. There is no first-genuine cross-modal
semantic-abstraction result here.

What A6 hands to DR1: an ACCEPTANCE CRITERION before spending Studio-tier encode. The target
attribute must be label-free-recoverable IN THE CAPTION and probe-verified above chance BEFORE
any Studio encode is spent. The shapecap pipeline is the template: it lifted a hard,
pretraining-dependent axis (shape) from chance-in-cheap-features to 0.62 linear-decode without
ever reading labels, and it verified this with a preregistered killswitch. DR1 should treat
"caption carries the attribute, probe-verified, killswitch armed" as a gate, not an assumption.
A6 also warns DR1 that clearing that gate is necessary but NOT sufficient: a probe-verified
caption can still share zero abstract geometry with vision beyond nuisance (the shape-axis
bounding null), so DR1 must keep the nuisance-residualized alignment test as the real success
criterion, not raw alignment.

## 6. Verdict-ledger line (for HANDOFF.md section 2)

A6 (residual cross-modal alignment): NULL for semantic abstraction, POSITIVE for nuisance
geometry. Vision-text shared code survives minus_shape_color (primary pair 0.0649, genuine) but
collapses to zero stable survivors at minus_nuisance / minus_all; carrier = spatiotemporal
nuisance geometry, not the label partition. Survived 3 controls (random-dim: 85.5%/80.0% delta
retained vs 0.025 nuisance; floor-decomp: 90-102% learned-driven; residualization audit: shape
to chance, nuisance to R^2 ~0 held-out). Shape-axis bet = BOUNDING NULL (shapecap probe 0.6167,
killswitch did not fire, yet vision->shapecap dies at minus_all, alignment-artifact). Caveat:
minus_nuisance degrades to non-replicating, not fully floored. Hands DR1 a label-free-caption +
probe-verified acceptance gate, with the warning that the gate is necessary but not sufficient.
