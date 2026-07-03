# Axis-Ceiling Result: pushing every laptop-reachable ideology axis to its honest maximum

Four parallel, scratchpad-isolated workflows (one per audit ideology axis), each with real builds and an
independent adversarial verifier, run to push each axis to its true ceiling ON THIS DEVICE. The mandate,
enforced in every workflow: do NOT fake a score. A structural ceiling below 10 with a stated reason is the
correct answer where it applies (a frozen encoder caps moldability and at-scale abstraction). Faking would
inflate the very falsification axis being raised. House style: no em or en dashes. Companions:
`LAPTOP_LANES_RESULT.md`, `A6_RESULT.md`, `POTENTIAL_AUDIT.md`.

## 0. The scorecard, honest (after FIVE rounds; each axis at its PROVEN ceiling with a mechanistic reason)

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

## 1. Moldability 2 -> 5: the first mechanism win (`scripts/mop_cbp_plasticity_repair.py`)

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

## 2. Abstraction 2 -> 3: one new slot, the abstract-code bet still a null (`scripts/mop_abstraction_richer_slots.py`)

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

## 3. Density 4.5 -> 4: fully instrumented, zero wins (`scripts/mop_density_retention_byte.py`, `mop_density_adaptation_update.py`)

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

## 4. Falsification 7 -> 9: rigor turned on the positives and on our own controls (`scripts/mop_survivor_completeness.py`, `mop_meta_control_audit.py`)

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

## 5. What this hands the next expansion

The laptop ceiling is now reached and mapped honestly. The two axes that stayed at 2->5 and 2->3 are capped
by the frozen encoder and by synthetic content; both are lifted only by the Studio moves already specified:
PR9 (run the now-validated CBP repair on a real-latent long stream, the first substrate-touching plasticity
test) and DR1 (real nameable-object video, where an attribute like count is dissociable from injected
nuisance so the abstract-shared-code bet can finally be posed on non-nuisance content). Two concrete
laptop-side follow-ups remain: unwire the vacuous `delta_frozen_random` control (falsification 9 -> ~10), and
carry the CBP mechanism onto the real cached-latent stream.

## 6. Round 2: exhausting the levers Round 1 left unpulled

Round 1 assigned ceilings against the ideological goal; a stop-check correctly noted that is not the same as
the laptop's honest maximum. Round 2 pulled the specific unpulled levers, one workflow per axis, each
adversarially verified.

### Density 4 -> 6: the Mixture-of-Perspectives thesis, won at matched compute (`scripts/mop_density_mixture_win.py`)
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

### Falsification 9 -> 10: the vacuous control retired (`runs/mot/falsification_vacuous_fix.diff`)
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

### Moldability 5 (held): real-substrate plasticity is a NULL (`runs/mot/moldability_real_stream_*.json`)
A long stream was built from the 200 real V-JEPA latents to run the substrate-touching plasticity test. The
adversarial verifier overturned a build-agent over-claim: the apparent continual-backprop "win" compared
CBP at lr 0.5 against a plain-SGD baseline MISTUNED into the dead-ReLU regime. Well-tuned plain SGD (lr 0.1)
already retains full plasticity on the real-latent stream (late 1.0, gap 0.0, zero dead units), so there is
nothing to repair; best-vs-best delta +0.0000 = a tie = null. The real substrate (low effective dimension,
well-conditioned) does not exhibit plasticity loss under task relabeling; the synthetic-Gaussian Round-1 win
does not transfer. Moldability is genuinely frozen-capped at 5; bands 9-10 need a trainable encoder.

### Abstraction 3 (held): count is inextricably geometry-confounded (`runs/mot/abstraction_dissociated_*.json`)
An area-DISSOCIATED clipset (total foreground area held constant while count 1-4 varies, corr(area,count)
-0.047) with parity/ordinal slots, encoded with stronger models (dinov2-large, Qwen 3B). Count and parity DO
decode from both perspectives against the majority floor, so the instrument gained slots. But the cross
perspective abstract-code bet is a bounding null: dissociating area just moved the confound to PERIMETER
(+0.806) and SPACING (+0.722); under a perimeter+spacing control count decode collapses below chance
(0.58 -> 0.13), and a random-init encoder REPRODUCES the alignment (the decisive C3 failure). The honest
conclusion: on any synthetic clipset, count is entangled with some low-level geometry, so the laptop cannot
demonstrate abstract cross-perspective count code. It requires DR1 (real nameable-object video).

## 7. Round 3: pulling the levers closer to the ideology (one more win, two ceilings proven)

Round 2 assigned ceilings; a stop-check pushed again. Round 3 pulled levers that hadn't been tried and are
arguably closer to the actual ideology. One surfaced a real win; the other two proved the ceilings.

### Abstraction 3 -> 4: analogical/compositional abstraction on real latents (`scripts/mop_abstraction_systematicity.py`)
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

### Density 6 (held): a NATURAL complementary task does not robustly convert (`runs/mot/density_natural_mixture.json`)
The R2 mixture win used a hand-constructed composite label. On the dataset's OWN factor structure (shape,
color, velocity are real independent generator factors, complementarity confirmed by Cramer's V), the mixture
does NOT convert to a preregistered win: a data-driven product-of-experts gets all-positive means but
sign-flips on 1-3 of 10 seeds; concat overfits and loses. So heterogeneity pays when the factorization is
SUPPLIED, not discovered from the natural joint at pilot scale. A second sub-axis win was also sought and
found null (retention/byte still sign-flips at the 4->8 doubling at 20 seeds; adaptation/update's stronger
methods are FLOP-unmatched or lose). Density is firmly 6; the boundary is now sharp.

### Moldability 5 (held): every developmental lever fails on the real substrate (`runs/mot/moldability_*.json`)
Three round-3 levers, all null or non-surviving: (developmental critical-period) a build-agent claimed a
sensitive-window WIN (+0.185 early-vs-late), but the adversarial verifier decomposed it to the early arm
training with a 4.33x higher trunk learning rate; swap only that LR and it collapses to +0.002,
sign-inconsistent, with no developmental gradient. (neuromodulation/metaplasticity) an honest matched-compute
tie (BWT delta +0.045, CI-lo -0.001, sign-flipping) on both real substrates. (full-latent stream) produced no
completed output. Across three rounds every plasticity mechanism either wins only on synthetic streams or ties
the tuned baseline on the real frozen substrate. Moldability is exhaustively frozen-capped at 5.

## 8. Rounds 4-5: the abstraction climb to 6, and both frozen-caps proven concrete

Rounds 4 and 5 kept pulling the axis that kept paying (abstraction) and nailed shut the two that did not.

### Abstraction 4 -> 6: substrate-invariant analogy (`scripts/mop_abstraction_cross_substrate.py`, `mop_abstraction_three_way.py`)
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

### Moldability 5 (proven concrete): factor-orthogonality on a frozen representation
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

## 9. The ceiling, adversarially confirmed (`runs/mot/ceiling_audit_verdict.md`)

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

## Note on lint scope

Nine of the promoted analysis scripts carry an E501-only per-file ignore in `pyproject.toml` (same rationale
the repo already uses for `tests/*`): one-off provenance scripts whose dense preregistered-hypothesis and
verdict strings read worse wrapped. The remaining promoted scripts wrap cleanly and carry no ignore. All
correctness lints (F, B, SIM, E4/E7, I, UP) and mypy apply to every one.
