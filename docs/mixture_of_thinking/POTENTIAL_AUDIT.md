# POTENTIAL AUDIT: is MoP/MoT reaching its potential and fulfilling its ideology

Audit date: 2026-07-02. Scope: the M3 Pro run against the four north-star ideological goals. Adversarial by
mandate: the default assumption is that we are NOT reaching potential, and the burden is on the evidence to
overturn it. House style: no em or en dashes. Every claim below is grounded in a named file, number, or
experiment id.

## 1. The verdict (decision-first)

**Overall reaching-potential score: 3.0 / 10.** The MoP program is, at this moment, an exceptionally honest
negative-mapping machine that has not yet won a single bet of the type its own thesis requires. After a full
M3 Pro run the tally is three "real-decisive" positives, and every one of the three is either a substrate
READOUT fact (shape decodes under nuisance; held-out (shape,color) factoring equals seen) or an oracle
EXISTENCE measurement (PR1 het_oracle_gain 0.155 vs hom 0.118, an upper bound, not a trained router beating a
tuned baseline). **Zero of the three headline positives is a mechanism beating a tuned baseline at scale, which
is precisely the shape of result the density thesis demands.** The single most important truth: the program has
built world-class machinery for not fooling itself and has pointed almost none of that machinery at the two
positives (al2, ws2) it is now promoting to the Studio, both of which fail a re-grade against their own
preregistered controls. The honesty is real; the yield, measured against the ideology rather than against the
process, is one run with no mechanism-wins and a to-build list (DR1, PR9, an executable verifier) that
`ls scripts/` confirms is still unbuilt. This is not a refuted program. It is a rigorous program that has so
far spent its rigor confirming what its instrument cannot do, on the one axis (shape+color) its instrument can
address.

## 2. Per-ideology scorecard

### Ideology 1: MOLDABILITY (moldable as a child's brain; deep plasticity; continual learning without catastrophic forgetting; developmental acquisition). Score: 2 / 10.

**For.** The program is structurally honest about a fatal tension it did not have to admit:
`SEMANTIC_POSITIONS.md:1450` states in writing that "moldable as a child's brain is, at the substrate level,
false by construction." The plasticity nulls are real controlled nulls, not absences: `pr2_plasticity_substrates.json`
carries a matched random-init-ViT baseline and a preregistered verdict rule. The honesty machinery caught its own
overreach in real time, demoting `dr2_sparse_real` from a 5-seed positive to UNSTABLE at 10 seeds and deflating the
corpus-standing e7_sparse story (`M3PRO_RUN_REPORT.md` section 4).

**Against.** This is the north-star goal and it is not merely untested, it is unreachable by the chosen
instrument. The only plasticity test on the REAL substrate runs backwards: `pr2_plasticity_substrates.json`
per-seed shows real V-JEPA `bwt_delta` -0.25/-0.20 (worse), `speed_delta_steps` -1.0/-1.67 (slower), with both
arms pinned at `final_acc 0.333` (chance), so the task was too degenerate to register learning at all; the
verdict concedes the +0.31 substrate structure is readout-only. Every other plasticity PR ties or hurts (PR4,
PR5 sign-flipped, PR6 ceiling with delta forced to 0.0, PR8 retrieval head LOSES to plain kNN). The lone
plasticity WIN, PR7 fast-slow, runs on dim-64 synthetic Gaussians and is self-flagged as a Hebbian FLOOR a
delta-rule provably dominates. **PR9 continual-backprop (Dohare, Nature 2024), described in both reports as the
one frontier-certified baseline-beater, was NEVER run: it exists only as planned item #5, with no
`continual_backprop`/`dohare` file anywhere in `scripts/`.** The one mechanism certified to restore lost
plasticity is unbuilt, and the test that could force un-freezing (SEM-PLAS-12/C3) needs a trainable-encoder arm
that violates both the frozen constraint AND the live-encoder ban, so it is structurally unrunnable, not
deferred.

### Ideology 2: ABSTRACTION ACROSS PERSPECTIVES (thought as an ecology of vision, language, code, math, physics). Score: 2 / 10.

**For.** at3_time_axis is the one pilot that genuinely instruments a second axis of perspective on real cached
latents: full-clip decodes motion_dir4 (+0.200, CI [0.165,0.235]) and speed2 (+0.245, CI [0.204,0.286]) that a
token-matched static frame cannot, at 10 seeds cleanly off zero. al2 uses the correct non-vacuous controls
(random-map-of-equal-rank plus shuffled-pairing floor) and honestly records `null_supported: false` rather than
laundering it.

**Against.** The multi-perspective thesis is almost entirely asserted, not tested, and the survivor advertised
as its evidence says the opposite of the claim. The instrument exposes exactly TWO semantic slots (shape,
color) on a single 200-clip cache; of ~86 semantic positions in `SEMANTIC_POSITIONS.md`, 51 are needs-new-cache,
and every CODE, MATH, and PHYS decoding cluster is retiered off the runnable list because no DSL, executor,
physics clipset, numerosity render, or paired-language encoder exists and the live-encoder ban forbids building
them now. **al2 is materially misreported.** `al2_alignment_pilot_seeds10.json` shows the learned rank-k map has
NEGATIVE `learned_r2` in the large majority of pair-cells (it predicts worse than the mean); the only substantive
positive is `vjepa2_vitl_singleframe -> dinov2s_nuisance_real` (rank-32 delta +0.278), i.e. two frozen VISION
encoders of the identical clips. Every cross-MODAL pair fails below the floor (dinov2->handcrafted delta ~-0.42;
vision->qwen, vision->wav2vec2 all negative). Worse, the real substrate arm `vjepa2_vitl_nuisance_real` is in
`arms_missing`: the alignment claim was never even tested on the pooled V-JEPA substrate the program is built on.
Yet `M3PRO_RUN_REPORT.md:50-53` calls al2 "perspectives share an alignable code" and "conceptually the most
important survivor." That is false for every perspective except vision-vision. ex2, the one planning positive,
touches no V-JEPA latent (a synthetic 8-d toy, `SEMANTIC_POSITIONS.md:443,1472`).

### Ideology 3: CAPABILITY DENSITY (reasoning per FLOP, retention per byte, adaptation per update, abstraction per parameter). Score: 3 / 10.

**For.** The concept is not hand-waved at the spec level: `12_metrics.md` defines each density metric with a
"how it can be gamed" field, and the matched-FLOP accounting is real, tested infrastructure
(`devsys/diagnostics/compute.py` with unit tests). The one experiment that computes the true thesis ratio,
`mt123_router_pilots.json`, honestly reported its NULL and its sign-flips rather than cherry-picking the two
positive seeds.

**Against.** The word "density" appears **zero times** in `M3PRO_RUN_REPORT.md` (grep confirmed). Every one of
the three real-decisive headlines and all five pilot survivors is scored on a raw accuracy or NLL delta, never a
ratio. There is no capability-per-FLOP, retention-per-byte, or capability-per-param frontier curve anywhere in
`runs/mot/`. The single thesis-level density test NULLed and sign-flipped: `mt123.mt1_router_vs_best_mode` delta
mean -0.2356, per-seed [0.337, 0.064, -0.535, -0.622, -0.422], `consistent_sign 0`, `null_supported true`. And
`mt3_hetero_vs_homogeneous` is STRONGLY and CONSISTENTLY negative: delta -1.01, per-seed all five negative
[-0.34,-0.75,-1.45,-1.33,-1.18], `consistent_sign -1`, meaning the heterogeneous router LOSES to a param-matched
homogeneous k-copy bank on the density metric. The only positive, MT2 (+0.062), is accuracy-vs-uniform-blend at
matched budget, not a density-frontier win over the best single mode. Retention/byte and adaptation/update have
ZERO M3 Pro test surface (`mt123` config: "accuracy/FLOP density only; retention/byte is the DR1-scale Studio
question"). The program claims density as its north star and has never plotted one, tested it once at thesis
level, and that test came back NULL while its sibling MT3 came back a clean LOSS.

### Ideology 4: THE FALSIFICATION ENGINE (non-vacuous controls, matched compute, honest nulls as assets). Score: 6.0 / 10.

**For.** This is the strongest axis and the score reflects it. The five-grade taxonomy is real machinery: it
demoted dr2/pr5/ws5 on sign-flips (`aggregate_report.json` any_sign_flip_ids), deflated e7_sparse, and refused
verdicts on five not-evaluable experiments rather than fake them. The vacuous-control catch is genuinely
internalized where it KILLS things: `compositional_binding.json` is a clean self-catch (real=frozen_random=1.0,
verdict CEILING, honestly declared uninformative). The reasoning-lane null is diagnosed to a mechanism with a
per-null rescue and a real kill-switch (`DEEP_RESEARCH_2026_07.md` lines 198-201) that would retire the whole
test-time-compute branch. The banning of the square `frozen_random_projection` as VACUOUS (delta forced to
0.000, `vacuous_control_finding.json`) is a correct methodological win.

**Against (why this is not 8+).** The falsification discipline is applied rigorously to NULLS and not to the two
POSITIVES being promoted. **ws2 fails its own preregistered dual (acc AND nll) contract yet is graded "NULL
REJECTED."** `ws2_fusion_tournament_seeds10.json` arms: cross_attention `acc_win false`, `nll -18.87`;
gwt_broadcast acc +0.033 (win) but nll CI [-0.11, +0.60] straddles zero (`nll_win false`); learned_linear
`nll_win false`. No single arm wins both metrics; the reported win is a post-hoc max over three arms, the exact
winner's-curse PR1 guards against with its mean-copy baseline and ws2 does not. **al2 is promoted despite
`null_supported: false` and a majority of negative-R2 cells.** And the redirect the nulls generate is entirely
on paper: `ls scripts/` shows no dsl, executor, verifier, hardness, or continual_backprop file; the D3 hardness
gradient that would tell whether the reasoning nulls are substrate-limited or task-limited has never fired
across two runs (mt5 UNREADABLE). A nulls-are-assets doctrine only earns a high score if the assets compound
into an eventual positive; here they compound into an unbuilt to-build list.

## 3. The central tension, stated plainly

The north-star goals are moldability (a brain that reshapes) and multi-perspective thought (an ecology of
vision, language, code, math, physics). The instrument is a FROZEN V-JEPA substrate that exposes exactly TWO
crisp slots, shape and color, on one 200-clip cache. These are not compatible, and the program half-knows it.

A frozen substrate can only re-weight a fixed basis, never reshape it, so ideology 1 is "false by construction"
(`SEMANTIC_POSITIONS.md:1450`); the one test that could force un-freezing (SEM-PLAS-12/C3) requires a
trainable-encoder arm that violates both the frozen constraint AND the live-encoder ban. This is not a deferral,
it is a structural foreclosure: roughly 25 percent of the north-star ideology is definitionally out of scope for
this method, which alone caps the honest ceiling of "fulfilling the ideology" well below what a naive average of
the lens scores implies.

The foreclosure is self-reinforcing, and this is the part the plan avoids rather than confronts. Frozen
justifies cached-latent-first; cached-first plus the 21s/clip live-encoder cost justifies the live-encoder ban;
the ban then blocks producing every modality cache (physics, numerosity, DSL, paired-language, real
bound-attribute video) that would test whether frozen is bounded. The instrument can only re-ask the two-slot
question it already answered. The plan's response to this is telling: the Studio handoff sequences DR1, the
named "decisive enabler" of the entire semantic layer (`MIXTURE_OF_THINKING` lines 291, 356, 529), at priority
#4 of 5, behind multi-seeding a p-value the program already owns. The revealed preference is to sharpen the one
axis the instrument can address rather than to spend against the artifact that would let the instrument address
a third. That is avoidance dressed as sequencing.

There is also a comfortable story that must be named: the "correct-regime-null" framing for the 24-null
test-time-compute lane. It is defensible as far as it goes (matched-FLOP ties are real and honest), but the
deeper fact is that the program CANNOT YET POSE A HARD TASK. No verifier, executor, or D3 hardness gradient
exists; every reasoning null is pre-labeled "regime-driven, rescuable" (`DEEP_RESEARCH_2026_07.md` lines
174-176). Until D3 fires, "the task was in the wrong regime" is indistinguishable from "we never tested
reasoning," and it is functioning as a virtuous-sounding label for an inability to build a difficulty-graded
benchmark.

## 4. What is genuinely working (survivors of the self-deception lens)

Only three things survive an adversarial re-read of the raw JSONs:

1. **at3_time_axis.** Temporal currency is real: `at3_time_axis_seeds10.json` motion_dir4 +0.200
   (CI [0.165,0.235]) and speed2 +0.245 (CI [0.204,0.286]) over a token-matched single-frame control across 10
   seeds. It is also the least surprising possible result (a static frame cannot decode motion), but it is honest
   cached-latent evidence for a second perspective axis.

2. **PR1 router gate.** The one place winner's-curse is handled correctly: `het_oracle_gain 0.155` (sd 0.014) vs
   a MEAN-copy homogeneous baseline 0.118, not a best-of-K max, with the mean-copy rationale written into config.
   GREEN survives. It is an EXISTENCE result (a perfect router would help), not a trained-router win.

3. **The substrate-is-special direction, with a large asterisk.** `substrate_vs_random_init_vit_interpreted.json`
   isolates pretraining from architecture and resolution (V-JEPA 0.517 vs random-init ViT-L 0.241 at matched
   256px) and points KEEP the encoder. But the headline p=0.0285 rests on 15/29 vs 7/29 at n_test=29, single
   split, single seed, with the file itself admitting split-noise SD ~0.08-0.09; a one-clip swing to 14/8 pushes p
   past 0.05. It is one storm-cloud from ambiguity and has not been multi-seeded.

Everything else being carried to the Studio (al2, ws2) does not survive its own preregistered controls, and the
entire plasticity and reasoning lanes are nulls.

## 5. Prioritized action list (ranked by expected value against the ideology)

**STOP over-investing in:**
- Multi-seeding the substrate headline p-value FIRST. It refines a number on the one axis (shape under nuisance)
  the program already owns. It is currently handoff priority #1; it should not be.
- Promoting al2 and ws2 as-is. They are handoff priority #2 and both fail a re-grade. Promoting them launders a
  winner's-curse and a vision-vision-only alignment into the Studio.
- Generating additional rigorous nulls on the two-slot instrument. The marginal null on shape+color teaches
  nothing new; several already "replicate prior corpus nulls exactly" (M3PRO section 0).

**DO, in order of leverage:**

1. **Build DR1 (the non-additive bound-attribute natural-video cache) FIRST, with count and relation slots.**
   It is the sole named unblocker of GATE C1 and ~70 semantic positions. Bundle a paired vision+text encoder
   pass on identical referents (the Qwen cache is text-of-labels, not parallel LLM states), which alone unblocks
   the SEM-LANG cluster and the language-independent-abstraction north star. This is the difference between having
   and not having a science on the multi-perspective ideology.

2. **Re-grade al2 and ws2 against their own preregistered controls before any promotion (zero new compute).**
   For al2: report the census, restate the claim as "two frozen vision encoders of identical content are
   alignable; no cross-modal alignment survives," and re-run with the missing `vjepa2_vitl_nuisance_real` arm and
   a kNN-topology permutation null instead of ridge R2. For ws2: enforce the dual acc-AND-nll metric with the
   mean-baseline guard PR1 already uses; on current data ws2 does not clear its contract and should be demoted to
   null.

3. **Build the D3 hardness gradient and one executable verifier, then re-run exactly one dead reasoning
   mechanism (dr8 fixed-point) against it.** This converts the reasoning nulls from a prose rescue-list into a
   live falsification and fires the standing kill-switch if the mechanism still ties. Right now the gating
   instrument has never been built across two runs.

4. **Run PR9 continual-backprop on a long real-latent stream, gated to the kill-switch.** It is the only
   plasticity mechanism in the literature certified to beat a tuned baseline on the exact failure mode
   (plasticity loss over a long stream), and it is the only one not yet run. Either it wins (first substrate
   touching plasticity positive) or it ties (moldability is honestly dead at this substrate). Both resolve the
   lens; nothing currently run does.

5. **Run CM1 and cache one small batch of real bound-attribute video through the existing frozen encoder** to
   make GATE C1 falsifiable without violating the frozen doctrine. Until at least one custom-model gate is
   runnable on real non-ceiling content, "keep it frozen" is unfalsifiable-by-construction on the ideology's
   core.

## 6. The honest bottom line

MoP is, today, a rigorous program that is slowly and honestly proving that its chosen instrument cannot reach
its stated goals on two of four axes, and has not yet tested itself on the axis (density) it names as its north
star. The moldability goal is false-by-construction for a frozen substrate and the mechanism certified to
rescue it is unbuilt. The multi-perspective goal is instrumented at exactly two visual slots, and the only
cross-perspective evidence in the corpus (al2) points the wrong way for every non-vision pair. The density goal
was tested once and NULLed, with its sibling test a clean loss. The falsification engine is genuinely excellent
and is the one thing keeping this honest, but it has so far been an engine for producing citable negatives on a
narrow axis, not for winning a bet.

The program is NOT on a path to something real yet, because the moves that could put it on that path (DR1, a
paired text cache, D3, PR9) are all deferred behind refinements of what it already knows. It is one decisive
build away from becoming falsifiable on its own ideology, and one comfortable quarter away from becoming a
permanent negative-mapping machine that generates rigorous nulls on shape and color indefinitely. Which of those
two it becomes is decided entirely by whether DR1 and the verifier get built before the next round of headline
multi-seeding. On the current handoff ordering, the comfortable outcome is the default. That is why the score is
3.0 and not higher: the honesty is a genuine 6, the ideological yield is a genuine 2, and the plan's own
sequencing is currently protecting the former at the expense of the latter.
