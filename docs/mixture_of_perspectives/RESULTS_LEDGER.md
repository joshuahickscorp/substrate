# RESULTS LEDGER

The lane-result records, merged from the former standalone A6_RESULT / LAPTOP_LANES_RESULT /
AXIS_CEILING_RESULT / ROLLOUT_LANE_RESULT files (condense docs consolidation, content verbatim). Each
section is one lane's result. No em or en dashes.

## Contents
- A6: cross-modal shared code (below)
- Laptop lanes: density, router, re-audit, plasticity
- Axis ceiling: the 5-round scorecard
- Rollout lane: facet 12 predictor rollouts


<!-- ===== merged from A6_RESULT.md ===== -->

## A6 Result: What the Cross-Modal Shared Code Actually Carries

### 1. Headline (honest)

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

### 2. The corrected premise

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

### 3. Residual-alignment result

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

#### Adversarial controls this conclusion survived

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

#### Caveats the controls raised (do not round up)

- Under minus_nuisance the primary pair lands at `non-replicating` (sign-unstable), NOT at the
  fully-floored `alignment-artifact`. The floor decomposition confirms residual learned recall
  is still marginally above the raw floor (0.2108 vs 0.1800 at rank 32). So the honest phrasing
  is "collapses to a non-survivor / degrades to instability," not "extinguished to zero." A
  linear 7-column design cannot remove nonlinear nuisance geometry, so a thin residual is
  expected. It does not clear the null and does not rescue a semantic-abstraction reading.
- The reverse direction qwen text -> vjepa2 is weak and non-replicating even at raw
  (delta 0.0086, a sign flip across seeds). The signal is carried by the forward vision->text
  direction. This does not affect the carrier conclusion.

### 4. Shape caption + shape-axis bet

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

### 5. A6 verdict and the DR1 handoff

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

### 6. Verdict-ledger line (for HANDOFF.md section 2)

A6 (residual cross-modal alignment): NULL for semantic abstraction, POSITIVE for nuisance
geometry. Vision-text shared code survives minus_shape_color (primary pair 0.0649, genuine) but
collapses to zero stable survivors at minus_nuisance / minus_all; carrier = spatiotemporal
nuisance geometry, not the label partition. Survived 3 controls (random-dim: 85.5%/80.0% delta
retained vs 0.025 nuisance; floor-decomp: 90-102% learned-driven; residualization audit: shape
to chance, nuisance to R^2 ~0 held-out). Shape-axis bet = BOUNDING NULL (shapecap probe 0.6167,
killswitch did not fire, yet vision->shapecap dies at minus_all, alignment-artifact). Caveat:
minus_nuisance degrades to non-replicating, not fully floored. Hands DR1 a label-free-caption +
probe-verified acceptance gate, with the warning that the gate is necessary but not sufficient.


<!-- ===== merged from LAPTOP_LANES_RESULT.md ===== -->

## Laptop Lanes Result: one axis per ideology, run in parallel, adversarially verified

This is the consolidated result of four laptop lanes run in parallel (one per audit ideology axis) alongside
the A6 workflow, then integrated. Every lane had a build agent and an independent adversarial verifier, and
every headline below is one that SURVIVED its verifier. House style: no em or en dashes. Companion:
`RESULTS_LEDGER.md` (the abstraction axis), `POTENTIAL_AUDIT.md` (the scorecard this updates).

### 0. What moved, in one paragraph

The falsification engine got sharper by turning on the positives, and it cost a headline: at3 temporal
currency is DEMOTED (its motion and speed decode were reading the injected velocity draw, not integrating
time). Density got its first real instrument: two honest, gaming-guarded Pareto frontiers (capability per
FLOP, capability per parameter) plotted from existing data, moving density from an unmeasured 3 to about
4.5, though the frontiers show the mixture arms DOMINATED by single specialized modes. The density mechanism
bet came back a clean null: a trained router on the real cache loses to both a tuned best-single reader and
a compute-matched homogeneous bank. Moldability did not move (it cannot on the laptop), but its Studio
instrument is now validated and turnkey. The through-line with A6 is exact: on this synthetic clipset,
apparent positives keep resolving to the injected generative nuisance (velocity, position, size), not to
abstract structure. That is the single most important thing the laptop can teach before the Studio.

### 1. Falsification lane: positive-survivor re-audit (axis 6/10 -> 7/10)

Applied A6-grade adversarial controls to the three Studio-bound positives and the substrate headline
(`runs/mot/survivor_reaudit.json`, `scripts/mop_survivor_reaudit.py`). Verdicts:

- **at3 temporal currency: DEMOTE.** The temporal labels (motion_dir4, speed2) are derived from the injected
  (vx, vy) draw. Under the honest strong control (project out r, vx, vy, vx^2, vy^2, |v|, sin/cos of the
  motion angle, train-fit), the full-vs-single-frame edge collapses: motion_dir4 shrinks 100 percent,
  speed2 shrinks 96.6 percent, both to chance. at3 was reading the injected motion parameters, not
  integrating temporal currency. This removes one of the three positives before it reached the Studio.
- **at1 cross-substrate invariance: HARDEN.** The two survivors' per-clip shape-probe correctness is only
  modestly correlated (phi 0.329), so they contribute genuinely independent invariance evidence, not one
  signal double-counted.
- **pr7 plasticity: HOLD.** The delta-rule null holds (CI entirely below the Hebbian floor). The Hebbian
  fast-store gain is real but tiny (+0.029, CI lo 0.027): a single-mechanism flicker, not a headline.
- **substrate-special: HOLD, fragile.** Bootstrapping the single 29-clip split, 63.7 percent of resamples
  keep the Fisher p below 0.05, and a one-clip adverse swing crosses it (p 0.0877). The direction is
  corroborated by the 200-clip on-disk caches (delta CI lo 0.504), but the single-split p is fragile as the
  file admits. Multi-seeding it is Studio B5, still LAST.

Why this raises the axis: the audit's 6-not-8 gap was that rigor was applied to nulls and not to the
positives. This lane applied it to the positives and demoted one on the merits.

### 2. Density lane: the first capability-density frontiers (axis 3/10 -> ~4.5/10)

`runs/mot/density_frontier.json`, `scripts/mop_density_frontier.py`. Before this lane the word "density"
appeared zero times in the run report and nothing was scored as a ratio, though density is the north star
and `mop/diagnostics/compute.py` already did matched-FLOP accounting. Now two of the four density axes have
real, non-trivial, gaming-guarded Pareto frontiers computed from existing data:

- **capability per FLOP** (source `mt123_router_pilots`): 5 arms, frontier = {reactive, sparse}; planner,
  routed, and blend_full are dominated. The routed mixture is strictly dominated by its own best single mode
  (sparse) at matched compute, consistent with the source mt1 null. FLOP counts are the real per-sample
  convention from the source config.
- **capability per parameter** (substrate shape-decode, recomputed via `linear_probe`): DINOv2 Pareto-
  dominates all: 0.861 shape accuracy at a 384d readout, about 3x V-JEPA per readout-parameter. The ranking
  survives a common-128d-readout control, so it is substrate quality, not readout capacity.
- **retention per byte** and **adaptation per update**: no existing run exposes bytes-per-exemplar or a
  matched updates/adaptation ratio, so these are marked Studio-only and left unfaked.

Honest reading: this closes a MEASUREMENT gap, not a capability gap. The density story the frontiers tell is
"single specialized modes win, the mixture is dominated," which is why the axis moves to about 4.5 and not
higher, and half the density north star (retention, adaptation) stays a real Studio gap.

### 3. Density mechanism lane: trained router on the real cache (null, an asset)

`runs/mot/router_mechanism.json`, `scripts/mop_router_mechanism.py`. Attempted to convert PR1's
oracle-existence result into a trained-mechanism win. Result: NULL, matched-compute honest. A trained
heterogeneous router (shape under nuisance, 0.860) loses to both a tuned best-single DINOv2 reader (0.870,
best on all 10 seeds) and a compute-matched homogeneous 40-copy bank (0.876). Both preregistered rejection
gates fail (router_vs_best CI lo -0.034 with a sign flip; router_vs_homo CI lo -0.043 with a sign flip). The
density-mechanism null the audit flagged is confirmed on the REAL cache, not just the synthetic mt123
regime. This kills the "maybe a trained router beats the baseline on the real cache" hope cleanly.

### 4. Moldability lane: the plasticity-loss certificate, validated (axis stays 2/10)

`runs/mot/plasticity_certificate.json`, `scripts/mop_plasticity_certificate.py`. The laptop cannot induce
Studio-scale plasticity loss, so the score does not move, and this lane does not pretend otherwise. What it
delivers is the down-payment: the instrument Studio PR9 needs, validated to fire and not false-fire. Under a
fixed plain-SGD baseline over a 150-task stream, the certificate measures early-minus-late learning
accuracy. On a concept-drift stream it FIRES (gap +0.513, CI [0.498, 0.528], dead ReLU units rise 0 to
0.75, the mechanistic signature PR9 targets). On a matched stationary stream it stays QUIET (gap ~0, CI
contains 0). The verifier added the decisive position-vs-identity control (shuffled task order preserves the
gap, fresh-net difficulty is flat across stream position), upgrading the interpretation from asserted to
demonstrated. Studio PR9 is now turnkey and de-risked.

### 5. The through-line, and what it hands the Studio

A6 found the cross-modal shared code is nuisance geometry, not semantic abstraction, and its shape-axis bet
was a bounding null even with a caption that provably carried shape. This re-audit found at3 was reading the
injected velocity, not time. Same lesson twice: on a synthetic clipset where position, size, orientation,
and motion are injected as nuisance, apparent higher-order positives keep resolving to that injected
nuisance. This is not a defect of the method, it is the method working: the partial-out controls are doing
exactly their job. It hands DR1 a precise mandate: real video is what dissociates an abstraction from the
injected nuisance, so DR1 must (a) enforce the A6 acceptance gate (the target attribute is label-free
recoverable in the caption, probe-verified) and (b) keep the nuisance-residualized alignment as the success
criterion, never raw alignment.


<!-- ===== merged from AXIS_CEILING_RESULT.md ===== -->

## Axis-Ceiling Result: pushing every laptop-reachable ideology axis to its honest maximum

Four parallel, scratchpad-isolated workflows (one per audit ideology axis), each with real builds and an
independent adversarial verifier, run to push each axis to its true ceiling ON THIS DEVICE. The mandate,
enforced in every workflow: do NOT fake a score. A structural ceiling below 10 with a stated reason is the
correct answer where it applies (a frozen encoder caps moldability and at-scale abstraction). Faking would
inflate the very falsification axis being raised. House style: no em or en dashes. Companions:
`RESULTS_LEDGER.md`, `RESULTS_LEDGER.md`, `POTENTIAL_AUDIT.md`.

### 0. The scorecard, honest (after FIVE rounds; each axis at its PROVEN ceiling with a mechanistic reason)

| Axis | Audit | R1 | R2 | R3 | R4 | R5 | What proves the ceiling |
|------|------:|---:|---:|---:|---:|---:|-------------------------|
| Falsification | 6 | 9 | 10 | 10 | 10 | **10** | Vacuous frozen-random control retired at the gate (3 verdicts flip, each validated); method axis, maxed |
| Abstraction | 2 | 3 | 3 | 4 | 5 | **6** | FOUR controlled wins (systematicity, cross-substrate analogy, 3-way cross-substrate); boundary MAPPED: 3-factor compositionality breaks and vision->language fails (text is shape-blind) |
| Density | 3 | 4 | 6 | 6 | 6 | **6** | Matched-compute mixture-of-perspectives WIN; a NATURAL task provably does not convert (the win needs the factorization supplied, not discovered) |
| Moldability | 2 | 5 | 5 | 5 | 5 | **5** | Structurally capped, PROVEN concrete: on the one substrate-specific forgetting stream the joint-training ORACLE itself hits chance (0.300 vs 0.328), so no mechanism can retain what the frozen features cannot represent |
| **Overall** | **3.0** | ~4.7 | ~6.0 | ~6.25 | ~6.5 | **~6.75** | Two axes high (method maxed, abstraction climbed 2->6), two capped with a MECHANISTIC reason, not an assumption |

The FIVE rounds are the exhaustive answer to "the absolute ceiling on this device." Each round pulled the
levers the prior one left, and repeatedly a lever CLOSER to the ideology surfaced a real win the earlier
framing had buried. SIX genuine positives now stand where the audit found zero mechanism wins: the
matched-compute mixture-of-perspectives density win, and FOUR abstraction wins that climbed the axis
2->3->4->5->6 (within-encoder systematicity, then substrate-invariant pairwise cross-substrate analogy, then
3-way cross-substrate consistency across three real encoders), plus the synthetic-stream plasticity repair
and the retired vacuous control. Crucially, every ceiling is now PROVEN with a mechanistic reason, not
assumed: moldability's cap is the joint-training ORACLE itself hitting chance on the one substrate-specific
forgetting stream (frozen features cannot serve two orthogonal tasks in one head); abstraction's cap is the
double wall that 3-factor compositionality begins to memorize conjunctions and the vision->language family
transfer fails because the label-free text is shape-blind; density's is that a natural (unconstructed)
complementary task does not robustly convert. The adversarial verifiers did their job throughout, killing
FOUR separate over-claims (the R2 mistuned-baseline CBP "win", the R3 LR-confound developmental "win", the R3b
operand-confound language<->math "win", and the R4 build-agent developmental claim) and holding every other
mechanism to a matched-compute tie. Moldability (5) and abstraction beyond 6 need the Studio (PR9, DR1) or
un-freezing; on THIS device they are at their proven maxima. Round-1..3 detail is in sections 1-7; the
Round-4/5 abstraction climb and the moldability/density boundary proofs are summarized in section 8.

The two rounds together are the honest answer to "the absolute ceiling on this device." Round 2 pulled the
levers Round 1 left: it turned a "no density win" into the program's first thesis-level mechanism win, and it
turned two soft ceilings (moldability, abstraction) into PROVABLE ones with citable reasons. The two genuinely
new POSITIVES: the matched-compute mixture-of-perspectives density win (Round 2) and the continual-backprop
plasticity repair on a synthetic stream (Round 1). Neither moldability nor abstraction can pass its Round-2
ceiling on this device: the real substrate has no plasticity loss to repair, and synthetic count is
inextricably geometry-confounded. Those two need the Studio (PR9 on real content, DR1 real video) or a
trainable encoder, by construction, not for lack of effort.

Round-1 detail follows in sections 1-4; the Round-2 levers are in section 6.

### 1. Moldability 2 -> 5: the first mechanism win (`scripts/mop_cbp_plasticity_repair.py`)

Continual-backprop (Dohare selective reinit of low-utility units) vs plain SGD on the VALIDATED 150-task
concept-drift plasticity-loss stream (the one `mop_plasticity_certificate.py` proved induces loss). Result:
CBP fully repairs the loss. SGD gap 0.513 (late-task learning collapses, dead units 0 -> 0.75); CBP closes
it to ~0 at ALL FIVE reinit rates (712 to 65536 reinits), 8/8 seeds, gap-close CI lo +0.496, dead units
back to ~0, without crippling early learning. Reproduced bit-for-bit (`runs/mot/cbp_plasticity_repair.json`).

Adversarial verifier caught and corrected two things (this is why the number is trustworthy):
- The FIRST on-disk CBP artifact was a STALE FALSE NULL: PR9's default replacement_rate 1e-4 gives
  int(1e-4 * 48) = 0 reinits/step on a 48-unit layer, so the mechanism never fired. That "null" was a
  switched-off mechanism, not evidence. The corrected fractional-budget run is authoritative.
- The plain-SGD baseline is fair (loss present at lr 0.05, 0.1, 0.2), not a strawman.

The shell-continual replay result (BWT +0.155 on the real cache) is DEMOTED: it wins equally on a pure
spatial-position nuisance (`runs/mot/shell_continual_position_control.json`), so it is generic anti-forgetting
replay dynamics, not identity-specific moldability. Ceiling is 5 not higher because the CBP win is on a
synthetic teacher-student stream, the only real-substrate effect is generic anti-forgetting, and the frozen
encoder structurally caps shell accuracy at the ~0.78 joint-training oracle. Bands 9-10 require a trainable
encoder (Process C), off the laptop surface by construction.

### 2. Abstraction 2 -> 3: one new slot, the abstract-code bet still a null (`scripts/mop_abstraction_richer_slots.py`)

Rendered a 112-clip richer clipset (shape, color, count, size, relation) and encoded it with the cheap
available encoders (dinov2-small image, Qwen2.5-0.5B label-free text), then added a CODE (DSL via
`verifier_exec`) and a MATH (numerosity) perspective. Findings:
- COUNT is a genuine NEW controls-surviving slot: it decodes from all four perspectives (image, text, code,
  math), survives a foreground-area partial-out, pixel read-through corr 0.757. The instrument grows from 2
  slots (shape, color) to 3 (shape, color, count).
- SIZE is an AREA ARTIFACT (collapses once biggest-blob-area is removed). RELATION is a flat null (primary
  object identity is not pixel-recoverable after occlusion). Both honestly demoted within the lane.
- The decisive cross-perspective ABSTRACT-shared-code bet is a BOUNDING NULL: cross-perspective alignment on
  count survives minus_all but is ~95% foreground-area-carried and partly reproduced by a random encoder, so
  it is not the abstract count code the thesis wanted. There is a qualified positive (image<->code and
  image<->math carry a trained shared code beyond pixel statistics) but it is a finer scene/visual code.
This is the A6/at3 lesson a third time: apparent cross-perspective structure is nuisance/geometry-carried.
The decisive scale test (real nameable-object video) remains Studio DR1.

### 3. Density 4.5 -> 4: fully instrumented, zero wins (`scripts/mop_density_retention_byte.py`, `mop_density_adaptation_update.py`)

All four density sub-axes are now honestly instrumented at pilot scale, gaming-guarded, adversarially
reproduced, and not one is a clean density win:
- capability/FLOP: NULL (routed mixture dominated by a single mode at matched compute).
- capability/param: a controlled family-level lead only (DINOv2 leads at every swept readout dim), but the
  native DINOv2-vs-VJEPA pair is a TIE (gap 0.060 < pooled CI 0.095).
- retention/byte: a real frontier with a knee at K=16 (327,680 bytes); K=32 is Pareto-dominated (top-end
  tie), so there is a genuine tradeoff only BELOW the knee.
- adaptation/update: NULL in 12/12 cells (a Reptile meta-init buys nothing per gradient update over a plain
  init on a convex linear readout; ratios' CIs all span 1.0).
The audit's "density was never measured" gap is now fully closed; the substrate simply shows no density
advantage at pilot scale. The ceiling drops from an optimistic 4.5 to an honest 4 precisely because rigorous
measurement returned ties. At-scale density (retention/byte, adaptation/update at DR1 scale) stays
Studio-gated.

### 4. Falsification 7 -> 9: rigor turned on the positives and on our own controls (`scripts/mop_survivor_completeness.py`, `mop_meta_control_audit.py`)

- Survivor-completeness audit (2 HARDEN, 2 HOLD, both HOLDs are demotions on the merits):
  - substrate-special HARDEN: the fragile single 29-clip p is superseded by a 200-clip multi-split bootstrap
    (pretraining-minus-randinit gap 0.537, 2.5th pct 0.400, 100% of resamples gap>0). Survived all four
    red-team angles (second encoder family, matched resolution, probe-capacity ceiling, split selection).
  - compositional factoring HARDEN: genuine novel-combo generalization (real held-out 0.749 vs a
    label-permutation leakage floor 0.218 vs matched-256px randinit 0.074; seen-minus-heldout CI includes 0).
  - PR1 oracle gain HOLD (demoted): mean edge survives but per-seed het>hom in only 85% of seeds (< 90%).
  - shapecap lift HOLD (demoted): the random-init caption text already decodes shape at 0.54, so the caption
    STRING carries the shape, not what pretraining adds to the encoder.
- Meta-control audit: matched-arch/resolution controls PASS (byte-level, identical clip checksums),
  permutation nulls correctly constructed PASS, seed determinism PASS. FOUND: three still-live VACUOUS
  controls in the corpus (`src/mop/diagnostics/substrate_ablation.py` delta_frozen_random / needs_real, a
  square full-rank projection that is probe-absorbed, consumed by `a_perception.py` and `s_semiotics.py`
  grounded_index). These demote genuine positives via false negatives. Logged as a defect in `ISSUES.md`;
  the fix touches ~10 experiment modules and 8 test files, so it is a scoped follow-up, not a rushed edit.

Ceiling is 9 not 10 because two named items are off-laptop or out-of-scope: the 29-clip headline needs the
Studio-tier B5 re-encode to fully RESOLVE (the 200-clip result is a robust replacement, not the same test),
and unwiring the vacuous control is a real refactor deferred to keep the test suite honest.

### 5. What this hands the next expansion

The laptop ceiling is now reached and mapped honestly. The two axes that stayed at 2->5 and 2->3 are capped
by the frozen encoder and by synthetic content; both are lifted only by the Studio moves already specified:
PR9 (run the now-validated CBP repair on a real-latent long stream, the first substrate-touching plasticity
test) and DR1 (real nameable-object video, where an attribute like count is dissociable from injected
nuisance so the abstract-shared-code bet can finally be posed on non-nuisance content). Two concrete
laptop-side follow-ups remain: unwire the vacuous `delta_frozen_random` control (falsification 9 -> ~10), and
carry the CBP mechanism onto the real cached-latent stream.

### 6. Round 2: exhausting the levers Round 1 left unpulled

Round 1 assigned ceilings against the ideological goal; a stop-check correctly noted that is not the same as
the laptop's honest maximum. Round 2 pulled the specific unpulled levers, one workflow per axis, each
adversarially verified.

#### Density 4 -> 6: the Mixture-of-Perspectives thesis, won at matched compute (`scripts/mop_density_mixture_win.py`)
Round 1's density null was regime-specific: it tested shape-alone, where all readers decode the same emergent
signal (correlated errors, ~0.04 headroom), so no mixture can win. On a COMPOSITE task where readers are
genuinely complementary (color decodes from DINOv2; motion direction only from full-clip V-JEPA), a
matched-FLOP, matched-param FACTORED heterogeneous mixture beats: the best single reader (+0.170 CI
[0.121,0.219]), every matched homogeneous bank (+0.073 to +0.227), the concat monolith (+0.212), the MLP
monolith (+0.253), 10/10 seeds, no sign flip, mechanistic guard passed (swapping the motion expert to the
motion-blind single-frame reader destroys the win, so heterogeneity is load-bearing). The NEGATIVE control
(shape x motion, where readers are not sharply separated) correctly TIES (+0.005, sign-flipping) and pins the
falsifiable precondition: the mixture wins iff a required factor sharply separates the readers. This is the
program's first thesis-level "mechanism beats a tuned baseline at matched compute" result. Plus a second win:
the substrate is capability-dense per readout-param (V-JEPA +0.552, DINOv2 +0.617 vs random-init,
nonlinear-robust). Held to 6 because the mixture win is on a CONSTRUCTED complementary task (a clean existence
proof with an explicit precondition, not a naturally-arising win) and retention/byte still ties out at pilot
scale. `runs/mot/density_mixture_win.json`, `density_capability_param.json`.

#### Falsification 9 -> 10: the vacuous control retired (`runs/mot/falsification_vacuous_fix.diff`)
The `delta_frozen_random` gate was unwired and replaced with the shuffled-floor gate (the honest latent-level
meaning: decodable-above-chance, not substrate-specific). Three experiment verdicts flip, each independently
validated as genuinely correct with no manufactured positive: `a_perception` A1 `null_supported` True->False
(real 1.0 vs shuffle floor 0.517), `s_semiotics` S1 `grounded_index` False->True (earned on the stricter
MI-over-random-code 1.32 AND RSA-over-shuffled 0.94), S10 `null_supported` True->False (S10 now detects the
vacuity it was blind to). `frozen_random` is kept but truthfully labeled vacuous-for-linear-metrics and a
genuinely-lossy `rank_reduced` control is added. Applied to the tree, full gates green, 5 files changed
(`substrate_ablation.py`, `a_perception.py`, `s_semiotics.py`, the test, `registry/experiments.yaml`). The
same class of vacuity one level down (the direct frozen-random ARM in S3/S5/S6 and ~7 others) is flagged in
`ISSUES.md` as a scoped follow-up, not rushed.

#### Moldability 5 (held): real-substrate plasticity is a NULL (`runs/mot/moldability_real_stream_*.json`)
A long stream was built from the 200 real V-JEPA latents to run the substrate-touching plasticity test. The
adversarial verifier overturned a build-agent over-claim: the apparent continual-backprop "win" compared
CBP at lr 0.5 against a plain-SGD baseline MISTUNED into the dead-ReLU regime. Well-tuned plain SGD (lr 0.1)
already retains full plasticity on the real-latent stream (late 1.0, gap 0.0, zero dead units), so there is
nothing to repair; best-vs-best delta +0.0000 = a tie = null. The real substrate (low effective dimension,
well-conditioned) does not exhibit plasticity loss under task relabeling; the synthetic-Gaussian Round-1 win
does not transfer. Moldability is genuinely frozen-capped at 5; bands 9-10 need a trainable encoder.

#### Abstraction 3 (held): count is inextricably geometry-confounded (`runs/mot/abstraction_dissociated_*.json`)
An area-DISSOCIATED clipset (total foreground area held constant while count 1-4 varies, corr(area,count)
-0.047) with parity/ordinal slots, encoded with stronger models (dinov2-large, Qwen 3B). Count and parity DO
decode from both perspectives against the majority floor, so the instrument gained slots. But the cross
perspective abstract-code bet is a bounding null: dissociating area just moved the confound to PERIMETER
(+0.806) and SPACING (+0.722); under a perimeter+spacing control count decode collapses below chance
(0.58 -> 0.13), and a random-init encoder REPRODUCES the alignment (the decisive C3 failure). The honest
conclusion: on any synthetic clipset, count is entangled with some low-level geometry, so the laptop cannot
demonstrate abstract cross-perspective count code. It requires DR1 (real nameable-object video).

### 7. Round 3: pulling the levers closer to the ideology (one more win, two ceilings proven)

Round 2 assigned ceilings; a stop-check pushed again. Round 3 pulled levers that hadn't been tried and are
arguably closer to the actual ideology. One surfaced a real win; the other two proved the ceilings.

#### Abstraction 3 -> 4: analogical/compositional abstraction on real latents (`scripts/mop_abstraction_systematicity.py`)
The synthetic-count route was a dead end (geometry-confounded). But the ideology names compositional and
analogical thought, and that IS present in the real V-JEPA latents on the 5x5 (shape,color) grid, where an
untrained ViT is not: (Test A, analogy) a shape offset transfers across color contexts (retrieval top-1
0.336, CI-lo 0.312 vs shuffle floor 0.056; random-init substrate 0.0; permutation p=0.000). (Test B,
systematicity) a shape probe generalizes to NOVEL shape-color conjunctions (0.730) while the matched
untrained ViT collapses to 0.055 (a pure conjunction-memorizer). Confound-corrected: the color axis is a
trivial pixel statistic the untrained net "wins" (0.99), so the tests target the SHAPE axis a pixel statistic
cannot fake (clean double dissociation: real shape/color analogy 0.55/0.00, randinit 0.00/0.99). Held to 4:
real latents but a SYNTHETIC rendered grid, not real video; above 4 needs DR1. The relational same/different
lever was a well-controlled bounding null (geometry-dissociable but the alignment dies once geometry is
projected out, and a random encoder reproduces it). `runs/mot/abstraction_systematicity.json`.

The language/code/math lever (re-run tractably on a synthetic arithmetic/logic set, `runs/mot/
abstraction_langcodemath.json`) is a NULL: the strong language<->code alignment (delta 0.61) is pure
tokenizer/architecture coupling (a random-init Qwen reproduces it at 0.617 and a surface-shuffle collapses
it), and the tokenizer-free language<->math / code<->math pairs fail once the operand distribution is
decorrelated (MATH decodes the operation at 0.155, ci-lo 0.142 below chance 0.167: its apparent decode was
operand-distribution leakage, not abstract operation structure). No genuine cross-perspective operation
abstraction on the laptop; the systematicity win is the abstraction result and the axis holds at 4. This was
the last unmeasured laptop lever; the axis is now exhausted.

#### Density 6 (held): a NATURAL complementary task does not robustly convert (`runs/mot/density_natural_mixture.json`)
The R2 mixture win used a hand-constructed composite label. On the dataset's OWN factor structure (shape,
color, velocity are real independent generator factors, complementarity confirmed by Cramer's V), the mixture
does NOT convert to a preregistered win: a data-driven product-of-experts gets all-positive means but
sign-flips on 1-3 of 10 seeds; concat overfits and loses. So heterogeneity pays when the factorization is
SUPPLIED, not discovered from the natural joint at pilot scale. A second sub-axis win was also sought and
found null (retention/byte still sign-flips at the 4->8 doubling at 20 seeds; adaptation/update's stronger
methods are FLOP-unmatched or lose). Density is firmly 6; the boundary is now sharp.

#### Moldability 5 (held): every developmental lever fails on the real substrate (`runs/mot/moldability_*.json`)
Three round-3 levers, all null or non-surviving: (developmental critical-period) a build-agent claimed a
sensitive-window WIN (+0.185 early-vs-late), but the adversarial verifier decomposed it to the early arm
training with a 4.33x higher trunk learning rate; swap only that LR and it collapses to +0.002,
sign-inconsistent, with no developmental gradient. (neuromodulation/metaplasticity) an honest matched-compute
tie (BWT delta +0.045, CI-lo -0.001, sign-flipping) on both real substrates. (full-latent stream) produced no
completed output. Across three rounds every plasticity mechanism either wins only on synthetic streams or ties
the tuned baseline on the real frozen substrate. Moldability is exhaustively frozen-capped at 5.

### 8. Rounds 4-5: the abstraction climb to 6, and both frozen-caps proven concrete

Rounds 4 and 5 kept pulling the axis that kept paying (abstraction) and nailed shut the two that did not.

#### Abstraction 4 -> 6: substrate-invariant analogy (`scripts/mop_abstraction_cross_substrate.py`, `mop_abstraction_three_way.py`)
- R4 WIN (cross-substrate analogy): the shape-offset parallelogram is SUBSTRATE-INVARIANT. A shape offset in
  V-JEPA's space, carried through a shared label-free ridge map, predicts the shape analogy in DINOv2's space
  (delta real-random 0.471, CI [0.425,0.517], no flip; a broken-map null confirms it needs real correspondence,
  a leak-check confirms the offset arithmetic is necessary and sufficient). Concept-blending and
  prototype/typicality were honest nulls in the same round. `runs/mot/abstraction_cross_substrate.json`.
- R5 WIN (3-way cross-substrate): the same abstract shape code is consistent across THREE independent real
  encoders (V-JEPA, DINOv2, V-JEPA-singleframe), both cross-architecture pairs beating their matched
  random-init controls (survives broken-map, color-confound, permuted-eval, disjoint-seed). The non-independent
  V-JEPA-family pair is honestly excluded from the verdict. `runs/mot/abstraction_three_way.json`.
- R5 NULLS that MAP the ceiling: 3-factor systematicity (shape x color x motion) beats the untrained null by
  +0.700 but FAILS the beat-the-non-compositional-baseline clause (delta -0.066, below the locked -0.10 floor):
  compositional generalization is intact at 2 factors and begins to memorize conjunctions at 3. And
  vision<->text systematicity is a NULL because the label-free text substrate is shape-blind (shape decode
  0.26 ~ its own random 0.22), so there is no shared shape structure to carry into a language perspective.
So abstraction's laptop reach is: strong, substrate-invariant compositional/analogical structure WITHIN the
vision family at 2 factors; it does not extend to higher-order compositionality or to a non-vision family on
synthetic content. That is an exhausted ceiling, not a premature one.

#### Moldability 5 (proven concrete): factor-orthogonality on a frozen representation
R4 tested continual-learning-without-catastrophic-forgetting (the moldability sub-goal that any anti-forgetting
mechanism should satisfy). NULL: where a real forgetting surface exists (naive BWT -0.145 to -0.318), replay is
sub-oracle (77-91%) and generic (equal gain on random-init features, so not substrate-specific); and the one
substrate-specific stream is FROZEN-ENCODER-CAPPED, its single-shared-head joint-training oracle landing at
chance (0.300 vs 0.328).

FRAMING CORRECTION (from the ceiling adversarial audit, `runs/mot/ceiling_audit_verdict.md`): that
oracle-at-chance is a SINGLE-SHARED-HEAD limit, not the encoder lacking the information. A task-conditioned
MULTI-HEAD oracle on the same frozen features recovers both factors (shape 0.792, color 0.658, mean 0.725),
and it discriminates substrate (random-init shape decode 0.235 ~ chance while color, a pixel statistic, is
0.998). So the frozen encoder DOES carry both factors; the honest wall is one level deeper: on a frozen
(un-reshapeable) representation two orthogonal factors either live in ISOLATED heads (zero interference, so
nothing to mold: measured BWT +0.0000, structural growth is vacuous) or COMPETE for a shared trainable
bottleneck (real forgetting, but the only repair is generic anti-forgetting that fails the position-nuisance
control, and pretraining on factor A HURTS transfer to factor B, negative FWT -0.15/-0.22 on both substrates).
A genuine moldability win requires a TRAINABLE encoder (Process C) where the representation itself can be
remolded so orthogonal factors need not compete for a fixed bottleneck. Moldability is at its device ceiling
of 5. `runs/mot/moldability_continual_no_forgetting.json`, `moldability_forgetting_surface.json`.

### 9. The ceiling, adversarially confirmed (`runs/mot/ceiling_audit_verdict.md`)

After the five rounds, the exhaustion claim itself was put on trial: three independent adversaries were tasked
to DISPROVE that the laptop is at its ceiling by finding one genuinely-novel, testable, controls-beating lever
per axis, and a completeness critic adjudicated (adversary-of-the-adversary). Verdict: NO real unpulled lever
exists. Every candidate the adversaries could invent was redundant, confounded, or walled, and each wall
survived the non-vacuous controls:
- moldability: multi-head isolation is vacuous (zero interference to mold); shared-adapter repair is the
  demoted generic anti-forgetting (fails the position-nuisance control); curriculum gives negative transfer
  (orthogonal factors compete); generative replay is dominated by free exact replay.
- abstraction: position/motion offsets are decodable but NOT analogically composable (the parallelogram
  operation needs a categorical semantically-anchored factor, which on this clipset is only shape); a
  cross-family position "win" is the retired pixel-statistic confound (random-init Qwen reads position for
  free); causal and part-whole forms have no ground truth in the action-free single-object clips.
- density: the 40-seed re-test confirms a data-driven mixture cannot self-discover the factorization the
  handed win supplied; product-of-experts fails no-sign-flip; ridge experts do not fix it.
The three ROOT walls: the frozen encoder (moldability), factor orthogonality plus only two composable factors
(abstraction, density), and the action-free / part-free synthetic clipset (abstraction). Raising any score
requires off-device resources (a Process C trainable encoder, or DR1 real action/multi-part video). The device
maximum is ~6.75 and it is now proven by an adversarial search that tried to break it and could not.

### Note on lint scope

Nine of the promoted analysis scripts carry an E501-only per-file ignore in `pyproject.toml` (same rationale
the repo already uses for `tests/*`): one-off provenance scripts whose dense preregistered-hypothesis and
verdict strings read worse wrapped. The remaining promoted scripts wrap cleanly and carry no ignore. All
correctness lints (F, B, SIM, E4/E7, I, UP) and mypy apply to every one.


<!-- ===== merged from ROLLOUT_LANE_RESULT.md ===== -->

## Rollout Lane Result: What the V-JEPA 2 Predictor Actually Forecasts (Facet 12)

### 1. Headline (honest)

The entire MoP corpus used V-JEPA 2's ENCODER and discarded its PREDICTOR, the learned
latent-space simulator of video dynamics. This is the first time the program drives the
predictor as a world model. Facet 12's acceptance gate (DR13 rollout-error compounding, now on
REAL predictor rollouts instead of synthetic transitions) returns a preregistered NULL on
usability: on 24 synthetic bound-nuisance clips the frozen predictor beats all three controls
(persistence, matched random-init predictor, shuffled-target) by a non-overlapping seed CI at
EVERY horizon 1 to 8, but by only about 5 to 7 percent, never within reach of the preregistered
usability bar (real error below half the best control). A directional-but-sub-usable signal is a
null under the preregistered rule, and a tie is a null.

So the predictor carries a genuine, sign-stable, adversarially-verified world-model signal on our
content, and that signal is far too weak to support multi-step latent planning. The rollout lane
(counterfactual and interventional abstraction, the ex2 latent-planning precursor, DR7 latent
chain-of-thought) is NOT licensed at usable fidelity by this evidence. Facet 12 does not move off
0 on usability grounds.

A second wave (section 11) tested the sharper, task-relevant criterion (does the rolled-out latent
keep the moving object's POSITION decodable, the thing counterfactual abstraction and planning
actually need) and reached the same verdict: WALL, correctly labeled null-by-ill-posedness because
the synthetic clips move sub-patch at every horizon. It also produced a genuinely new,
adversarially-survived mechanistic finding: object position SURVIVES the compounded rollout but in a
representational sub-space the encoder-trained head cannot read (in-domain probe R2 0.73 vs
encoder-trained 0.09 at h=1), which names a concrete Studio fix (a readout adapter) and confirms the
representational gap of wave 1 is a systematic sub-space shift, not noise.

This verdict is PROVISIONAL on the clipset. The clips are synthetic and out of distribution for a
predictor trained on real video, and the whole-future-slot masking is out of distribution for the
predictor's training mask pattern, so the honest real-scale verdict requires the Studio re-run on
hosted real corpora (facet 14 feeds facet 12), which the same instrument runs turnkey via
`--clip-dir`. What DID convert is the lever itself: from an unmeasured capability to a measured one
with a preregistered, bit-exact, adversarially-verified instrument and a decisive synthetic-clipset
number.

### 2. What the lever is

V-JEPA 2 ships an encoder plus a PREDICTOR: a masked spatiotemporal-patch predictor over one clip's
32 temporal-slot by 16x16 spatial patch grid (256 patches per slot, 8192 total). Given encoder
hidden states at CONTEXT patch indices plus TARGET patch indices, it forecasts the target patch
representations. The teacher for that forecast is the encoder's OWN representation of the target
patches, which is exactly V-JEPA's training signal. Rolling the forecast forward (substituting the
predicted slot representations back into the context buffer before predicting the next slot) is an
open-loop latent rollout, and the question is whether that rollout stays faithful for enough steps
to be a usable world model, or whether error compounds so fast the lane is bounded to one-step
counterfactuals (the terminal ex2/DR13 wall the audit anticipated).

### 3. The instrument is real and correct (validated, not assumed)

- The model loads as VJEPA2Model from the local HF cache (`facebook/vjepa2-vitl-fpc64-256`,
  `local_files_only`), float32, CPU. MPS overflows 64-frame V-JEPA on the M3 Pro, so CPU.
- Grid confirmed: 32 temporal slots, 16x16 spatial, 256 patches per slot.
- BIT-EXACT teacher: the harness's direct-submodule predictor call and its teacher target were
  checked against the top-level VJEPA2Model path. pred(top-level) vs pred(submodule) max|diff| =
  0.0; teacher(top-level target_hidden_state) vs harness teacher (encoder full-clip state at the
  target slot) max|diff| = 0.0; nmse identical across paths. The harness measures exactly the
  model's own masked-prediction objective, not a proxy.

### 4. Preregistration (fixed in code before any number)

From `scripts/mop_dr13_predictor_fidelity.py`, committed before running:

- Horizons h in {1, 2, 3, 4, 6, 8} temporal-slot lookaheads. Rollout compounds by feeding predicted
  slots back into the context buffer (true open-loop rollout).
- Usable horizon = the largest CONTIGUOUS h (from h=1) at which real_nmse.hi < every control_nmse.lo
  (non-overlapping seed CI, lower is better) AND real_nmse.mean < 0.5 * best_control_nmse. A tie is
  a NULL.
- Three non-vacuous controls:
  (a) persistence: copy the last ground-truth context slot (no dynamics).
  (b) random_init: the SAME predictor architecture with freshly initialized weights, identical pipe.
  (c) shuffled_target: the real predictor output scored against a DIFFERENT clip's true target slot.
      This control shares the predictor-vs-encoder representational gap (see section 6), so beating
      it isolates genuine clip-specific dynamics net of that gap.
- Verdict rule: convert if usable horizon >= 2; wall-to-1-step if real is cleanly usable only at
  h=1; null otherwise.
- 24 clips bucketed into 3 seed groups for a seed CI, T_START = 4.

### 5. Results (24 synthetic clips, CPU, 2026-07-03)

Per-horizon nmse (lower is better). "real/best" is the ratio of real error to the best (smallest)
control error; usable requires it below 0.5.

| h | real nmse | persistence | random-init | shuffled | best/real ratio | beats all 3 (CI) | usable |
|---|-----------|-------------|-------------|----------|-----------------|------------------|--------|
| 1 | 0.767     | 0.964       | 1.035       | 0.824    | 0.93            | yes              | no     |
| 2 | 0.816     | 0.906       | 1.038       | 0.862    | 0.95            | yes              | no     |
| 3 | 0.833     | 0.906       | 1.040       | 0.873    | 0.95            | yes              | no     |
| 4 | 0.833     | 1.138       | 1.044       | 0.873    | 0.95            | yes              | no     |
| 6 | 0.815     | 1.022       | 1.037       | 0.858    | 0.95            | yes              | no     |
| 8 | 0.971     | 1.064       | 1.033       | 0.982    | 0.99            | yes              | no     |

- Real beats all three controls by a non-overlapping seed CI at every horizon. That is a real
  signal, not chance: the seed CIs are tight (real_nmse ci_half about 0.001 to 0.003).
- Real never comes close to the 0.5 usability bar. The margin over the strongest control
  (shuffled_target) is about 5 to 7 percent and collapses to about 1 percent by h=8.
- Encoder adjacent-slot nmse scale = 0.948. Adjacent V-JEPA temporal slots are nearly decorrelated
  in nmse, so one step of true motion is about 0.95; the predictor's 0.767 is below that but
  captures only about 20 percent of the transition.
- Cosine distance tells the same story more gently: real cosd 0.349 at h=1 (cosine similarity 0.65)
  rising to 0.82 by h=8.

### 6. The representational-gap finding (from the leakage probe)

The independent verifier's leakage probe (V3) asked the predictor to reconstruct a slot that is
ALREADY in its context. If masking leaked, that would be a trivial near-zero copy. It is not:
in-context nmse = 0.750, future (h=1) nmse = 0.775. Two consequences, both load-bearing for honesty:

1. The predictor is NOT trivially copying the target (future is genuinely harder than in-context),
   so the h=1 signal is real prediction, not leakage.
2. The predictor is LOSSY even on fully visible content (0.750), so most of the roughly 0.77
   one-step "rollout error" is a predictor-vs-encoder REPRESENTATIONAL GAP, not forecast failure.
   The MARGINAL one-step forecast cost is only about 0.025 nmse above that representational floor.

The shuffled_target control shares this representational gap (it is also predictor-output vs an
encoder slot), so "real beats shuffled at every horizon" isolates genuine clip-specific dynamics
NET of the gap. The gap does not rescue usability, though: the predictor's outputs do not live
cleanly in encoder space, which is itself a reason latent-planning-by-rollout in encoder space is
hard, and the raw error is what a downstream rollout consumer would actually see.

### 7. Independent adversarial verification (all pass)

`scripts/mop_dr13_verify.py` does NOT import the harness. It re-derives the load-bearing claims by an
independent path with DISJOINT seeds (fresh clips seeded 50000+, harness clips seeded 1000+;
random-init seed 999 vs harness 12345) and tries to break them:

- V1a random-init weights differ from real: mean|dw| = 0.154 (control b non-vacuous). PASS
- V1b shuffled-target is a genuinely different clip: all clips true != other. PASS
- V2a real < persistence at h=1 on fresh clips: 0.766 vs 0.961. PASS
- V2b real < random-init at h=1 on fresh clips: 0.766 vs 1.023. PASS
- V3 leakage probe, future harder than in-context: 0.775 > 0.750. PASS
- V4 harness h=1 reproduces on fresh clips: verify 0.766 vs harness 0.767. PASS

A shrinkage artifact is ruled out by construction: nmse rewards matching the target magnitude, so a
predictor shrinking toward zero would push nmse toward 1, not down; real's low nmse plus the cosine
alignment (0.65 at h=1) confirm directional prediction, not magnitude collapse.

### 8. Verdict and what it licenses

NULL on usability. The rollout lane is NOT licensed for multi-step latent planning or counterfactual
rollout at usable fidelity ON THIS SYNTHETIC CLIPSET. There IS a genuine, adversarially-verified,
sign-stable world-model signal (real beats all three controls at every horizon), so the predictor
is a real if weak simulator of our content; it is just far below the fidelity a rollout consumer
needs. Per the goal loop, a proven wall with a mechanism is success: the mechanism here is that the
predictor's one-step forecast, while better than every baseline, sits deep inside a large
representational gap and against a latent whose adjacent slots are nearly decorrelated, so open-loop
rollouts diverge immediately.

The verdict is PROVISIONAL on content. These clips are synthetic and out of distribution for a
predictor trained on real video, and the whole-future-slot masking is out of distribution for the
training mask pattern. The licensed real-scale verdict requires the Studio re-run on hosted real
corpora. Two outcomes for the Studio:

- If on real video the margin widens past the usability bar: the rollout lane is licensed and facet
  12 moves off 0 toward its ceiling (counterfactual abstraction, ex2 latent planning, DR7).
- If it stays weak on real video: that is the honest ex2/DR13 wall at real scale, exactly as the
  audit anticipated ("bounded to one-step counterfactuals... itself the honest ex2/DR13 verdict").

### 9. Scoring impact

Facet 12 stays at 0 on usability. The lever converts from UNMEASURED to MEASURED: the program now
owns a preregistered, bit-exact, adversarially-verified DR13-on-real-predictor instrument, a
decisive synthetic-clipset number, and a turnkey real-corpora next step. No axis score is claimed
or moved on the strength of a synthetic OOD clipset. This is an M3 Pro early-lever result
(doctrine-sanctioned to run ahead of the spine); it does not complete Studio WAVE 0, which still
requires the M1 Ultra (MPS-vs-CPU microbench at 128 GB, 1000-clip real cache rebuild, full gates on
that box). This result is folded into STUDIO_RUN_REPORT.md when the Studio creates it at WAVE 0.

### 10. Reproduction

```
PYTHONPATH=src:scripts:. OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/mop_dr13_predictor_fidelity.py --n-clips 24
## real corpora (Studio): add --clip-dir DIR of .pt clip tensors [frames,3,H,W]
```

Instrument: `scripts/mop_dr13_predictor_fidelity.py` (preregistered, in code). Synthetic-transition
sibling (the compounding-with-horizon reference): `scripts/mop_dr13_horizon_limit.py`. Audit context:
`docs/mixture_of_perspectives/STUDIO_POTENTIAL_AUDIT.md` facet 12.

### 11. Wave 2: decodability-retention (does the rollout track motion?)

Raw nmse (waves above) is not what the rollout lane exists for. Counterfactual/interventional
abstraction and latent planning need the rolled-out latent to keep TASK CONTENT decodable,
specifically to track WHERE A MOVING OBJECT GOES. Wave 2 preregistered that criterion: fit a linear
ridge probe on TRUE encoder latents at slot T_START+h to decode the object centroid (cx, cy), apply
the same probe unchanged to the compounded open-loop rollout latent, and require the rollout to beat
PERSISTENCE (hold the last real slot, the "assume nothing moved" floor) plus the random-init and
shuffled controls by a non-overlapping seed CI, retaining at least half the true-latent
above-persistence decodability. Persistence is the sharp control: a world model must track motion
better than assuming nothing moved.

VERDICT: WALL, correctly labeled NULL-by-ill-posedness. Facet 12 does not move off 0.

- Under the realistic encoder-trained readout the rollout decodes position at the random/shuffled
  floor: position R2 0.09 / 0.06 / 0.06 / 0.07 / 0.05 / -0.005 at h = 1/2/3/4/6/8, versus persistence
  0.35 / 0.33 / 0.35 / 0.23 / 0.29 / 0.21 and the true ceiling 0.40 / 0.37 / 0.43 / 0.41 / 0.42 /
  0.32. Retention vs persistence is NEGATIVE at every horizon; beats-persistence by seed CI is FALSE
  at every horizon; usable_horizon = 0.

The independent adversarial re-derive (fresh seeds, fresh code, shuffled clip-disjoint split) held
the wall on all four vectors AND corrected one over-claim in the build's own write-up, which is the
adversarial discipline working as intended:

- MOTION PREMISE FALSIFIED. The build's pixel-centroid motion gate (which reported h=8 displacement
  18.8 px and set motion_testable = true) was EXTRACTION NOISE. Reconstructing the object's true
  trajectory analytically from the generator's own RNG draw order (r, rot, x0, y0, vx, vy; _hue
  consumes no RNG, verified against `scripts/compositional_under_nuisance.py`) shows true
  displacement is SUB-PATCH at every horizon (median 0.60 / 1.21 / 1.81 / 2.42 / 3.63 / 4.84 px; h=8
  is 0.30 patch). Pixel-vs-analytic per-clip displacement is NEGATIVELY correlated at every horizon
  (about -0.55 short, -0.30 at h=8): the clips the extractor thought moved most actually moved least.
  make_bound_nuisance_clip draws vx, vy in [-0.2, 0.2] normalized (about vx/32 per slot), so motion
  is intrinsically sub-patch and there is no velocity argument to amplify. Corrected label:
  motion_testable = FALSE, the clipset cannot pose the motion-tracking question. This is a null on an
  ill-posed clipset, not a demonstrated dynamics wall.
- PROBE HONESTY CLEAN: the probe is fit on true encoder latents and applied to predicted latents (not
  re-fit on predicted latents, so the representational gap is not laundered); the split is
  clip-disjoint and the wall reproduces under a shuffled split.
- COMPOUNDING GENUINELY OPEN-LOOP: open-loop vs teacher-forced rollout L2 divergence is 0.0 at h=1
  and monotone for h>=2 (0.54, 0.61, 1.26 at h=2, 4, 8), so predicted latents are truly fed back.

NEW MECHANISTIC FINDING (survives adversarial). The wall is NOT total content destruction. A probe
fit IN-DOMAIN on rollout latents recovers position well: R2 0.73 / 0.69 / 0.64 / 0.45 at h =
1/2/4/8, versus 0.05 to 0.09 for the encoder-trained probe. So object position SURVIVES the
compounded rollout, but the predictor writes it into a representational sub-space the encoder-trained
head cannot read zero-shot. This confirms wave 1's representational gap is a systematic sub-space
shift, not noise. Even the generous in-domain probe beats in-domain persistence by seed CI at ONLY
h=1 (fragile, sub-patch motion, wide CIs), so it is not a licensing signal; but it names a concrete
mechanistic fix for the Studio: a per-representation-space READOUT ADAPTER between predictor output
and any downstream planning/counterfactual head.

Licensed re-test (Studio): real moving video with genuinely supra-patch object motion (facet 14
corpora), decoded through a readout adapter fit on rollout latents. Method and provenance:
`scratchpad/facet12b/` (decodability_retention.py, adversarial_pass.py, motion_validation +
analytic-trajectory correction). No axis score is moved on synthetic ill-posed content.

ADAPTER SCAFFOLD (turnkey re-test, `scripts/mop_dr13_readout_adapter.py`, `--clip-dir` for real video):
a linear predictor-space to encoder-space adapter, fit self-supervised on VISIBLE slots, is built and
smoke-run. Finding (synthetic, provisional): the adapter genuinely HALVES the representational gap on
visible slots (in-context nmse 0.727 to 0.357) but that correction does NOT transfer to the open-loop
rollout (adapted rollout nmse 0.82 to 1.20, slightly worse than the raw 0.76 to 0.83), because the
compounded rollout latents drift out of the visible-slot distribution. So a NAIVE visible-slot adapter
is insufficient; the Studio should fit the adapter on actual rollout predictions (teacher-forced targets)
or per-horizon, on real moving video. The scaffold produces this verdict preregistered and de-risked.

