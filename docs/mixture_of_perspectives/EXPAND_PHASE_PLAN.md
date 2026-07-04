# EXPAND PHASE PLAN: from the proven laptop ceiling to the off-device moves

The laptop is at its adversarially-proven device ceiling (~6.75/10: falsification 10, abstraction 6, density 6,
moldability 5; see `RESULTS_LEDGER.md`). This plan translates each PROVEN wall into the specific
off-device experiment that breaks it, carrying the methods the laptop validated. It is execution, not design:
the laptop rounds already found what works, what the controls must be, and where each wall is. House style: no
em or en dashes.

## 0. The three root walls (from the ceiling adversarial audit) and the move that breaks each

| Wall (proven on the laptop) | Caps | The off-device move |
|-----------------------------|------|---------------------|
| The FROZEN encoder cannot reshape its representation | moldability (5) | Process C (trainable-encoder arm) + Studio PR9 (real long stream) |
| Factor orthogonality + only TWO composable factors (shape, color) on the clipset | abstraction (6), density (6) | Studio DR1 (real video with many genuine composable factors + a real language perspective) |
| The synthetic clipset is ACTION-FREE and PART-FREE (no causal, no part-whole content) | abstraction (6) | Studio DR1 (real action-conditioned, multi-object video) |

Everything the laptop proved POSITIVE transfers as a validated method; everything it proved NEGATIVE is a
precise, pre-registered target for the Studio. Nothing here needs to be re-derived.

## 1. Track A: Moldability, break the frozen-encoder wall (Studio PR9, then Process C)

WHAT THE LAPTOP PROVED. Continual-backprop (Dohare) REPAIRS provably-induced plasticity loss on a synthetic
stream (gap 0.513 -> ~0, all rates, 8/8 seeds, reproduced). But the frozen real substrate exhibits no
plasticity loss to exploit: a multi-head oracle recovers both factors (0.725), so the encoder carries the
information; the wall is that two orthogonal factors must either ISOLATE (zero interference, nothing to mold)
or COMPETE for a fixed trainable bottleneck (real forgetting, but only generic anti-forgetting repair, which
fails the position-nuisance control, plus negative transfer -0.15/-0.22). A moldability win therefore needs a
representation that can be REMOLDED.

STUDIO PR9 (first, cheaper). Run the validated continual-backprop mechanism on a REAL-latent LONG stream at
Studio scale (the laptop's 200-clip / short-stream regime cannot induce Studio-scale loss). Instrument: ship
the validated plasticity-loss certificate (`scripts/mop_plasticity_certificate.py` logic: late-vs-early
learn-accuracy gap under a WELL-TUNED SGD baseline, dead-unit fraction) so the result is not a mistuned
baseline artifact (the laptop caught that over-claim twice). Script: `scripts/studio/pr9_continual_backprop.py`.
KILL-SWITCH: if CBP ties a well-tuned baseline on a stream that provably induces loss (certificate fires),
moldability is dead at a frozen substrate and Process C is LICENSED.

PROCESS C (the real lever, gated by PR9). A trainable-encoder arm where the representation itself can remold
so orthogonal factors need not compete for a fixed bottleneck. This is the ONLY path to high moldability and
the audit named it as the root wall. Sanctioned pilot per doctrine: a 1-10M object-centric module on frozen
dense tokens first; a from-scratch V-JEPA-scale model stays out of scope (~60 GPU-year moonshot). Control: the
random-init same-arch encoder at matched resolution, never a square latent projection (the laptop retired that
vacuous control at falsification 10).

## 2. Track B: Abstraction, break the composable-factor and content walls (Studio DR1)

WHAT THE LAPTOP PROVED (four abstraction wins, all on the SHAPE factor). The real substrate supports
substrate-invariant compositional/analogical abstraction: within-encoder systematicity (shape offset transfers
across color; novel conjunctions generalize where an untrained ViT collapses), pairwise cross-substrate
analogy (V-JEPA -> DINOv2), and 3-way cross-substrate consistency. It BREAKS at (a) 3-factor compositionality
(memorizes conjunctions) and (b) cross-family transfer to language (the label-free pixel-text is shape-blind).
The parallelogram operation needs a categorical, semantically-anchored factor; the clipset has exactly one
(shape). It has no causal or part-whole content at all.

STUDIO DR1 (the decisive unblocker). Curate a real bound-attribute VIDEO cache with MANY genuine composable
factors (object identity, count, relation, action) AND a PAIRED real language perspective (real captions that
carry object identity, unlike the shape-blind synthetic pixel-text). Script: `scripts/studio/dr1_curate_bound_video.py`,
which already carries the ACCEPTANCE GATE the laptop mandated: the target attribute must be LABEL-FREE
RECOVERABLE from the paired caption, probe-verified above chance on a held-out split, BEFORE Studio encode is
spent (do not burn compute on an attribute a caption cannot carry). Carry these VALIDATED laptop methods:
- the systematicity/analogy parallelogram test (`scripts/mop_abstraction_systematicity.py`) as the abstraction
  probe, now run on 3+ REAL composable factors (the wall was that the clipset had only shape);
- the cross-substrate transfer test (`scripts/mop_abstraction_cross_substrate.py`, `mop_abstraction_three_way.py`)
  for substrate-invariance across real encoders and the atlas;
- the A6 residualized-alignment method (`scripts/mop_a6_residual_alignment.py`: project out nuisance, test
  residual topology over a permutation floor) so any cross-modal win is not the retired nuisance-geometry
  confound.
GATES: (b1) does systematicity extend to 3+ REAL composable factors without memorizing conjunctions? (b2) does
cross-family (vision <-> language) abstraction hold on a real caption that DOES carry the attribute (the exact
wall the laptop hit)? (b3) do causal/interventional and part-whole abstraction probes register on real
action/multi-part content? Each is a laptop null with a precise real-content retest.

## 3. Track C: Density, the natural-complementarity retest on real factors (Studio DR1, same cache)

WHAT THE LAPTOP PROVED. A matched-compute heterogeneous mixture-of-perspectives BEATS every homogeneous
control (+0.073 to +0.253, 10/10 seeds, mechanistic guard), with a FALSIFIABLE PRECONDITION: the win appears
iff a required factor sharply separates the readers. But a NATURAL (unconstructed) complementary task does not
robustly convert at pilot scale (a data-driven mixture cannot self-discover the factorization from the natural
joint at n=140). Retention/byte and adaptation/update tie.

STUDIO DR1 (rides the same real-video cache). Test whether the mixture-of-perspectives win arises NATURALLY on
real content where genuinely different perspectives (vision, video-motion, language, audio) are sharply
separated by the real factor structure, with a data-driven mixture at scale (n >> 140, so a linear reader can
self-discover the subspace assignment). Method: `scripts/mop_density_mixture_win.py` (matched FLOPs and params,
PR1 mean-copy guard, no best-of-K). Also the at-scale retention/byte and adaptation/update ratios the laptop
could only tie at pilot scale. GATE: does a data-driven mixture beat the mean-copy homogeneous bank at matched
compute on a natural real-content task, sign-stable? If yes, density's constructed-task caveat is removed.

## 4. What transfers as validated method (do not rebuild)

- Falsification hygiene at 10: no vacuous controls in any gate (the frozen-random gate was retired; see
  `ISSUES.md` for the remaining direct-arm refactor, a reviewed follow-up). Every Studio claim uses a matched
  random-init encoder, a shuffle/permutation floor, matched compute, and a no-sign-flip rule.
- Adversarial verification discipline: every candidate positive gets an independent skeptic. On the laptop this
  killed FOUR over-claims (mistuned-baseline CBP, LR-confound developmental, operand-confound language-math,
  build-agent developmental). Run the same on every Studio positive.
- The A6 residualized-alignment, the systematicity parallelogram, the cross-substrate transfer, the CBP
  mechanism, the plasticity-loss certificate, and the matched-compute mixture test are all validated and ready.

## 5. Priority and gates

1. DR1 first (Tracks B and C ride the same real-video cache): it is the single unblocker of the most axes
   (abstraction beyond 6, density's natural-complementarity, the causal/part-whole probes). Its acceptance
   gate prevents wasted encode.
2. PR9 second: cheaper than Process C, and its kill-switch decides whether Process C is licensed.
3. Process C only if PR9 ties on a loss-inducing stream (moldability dead at a frozen substrate) OR DR1 shows
   the composable-factor wall is representational, not content. Today's evidence already points here: the
   frozen encoder is the proven moldability wall.

The through-line: the laptop turned four ideology axes from "asserted" into "measured with a mechanistic
ceiling," produced six genuine wins, and reduced the whole program to three off-device walls with a
pre-registered retest for each. The expand phase is now execution.
