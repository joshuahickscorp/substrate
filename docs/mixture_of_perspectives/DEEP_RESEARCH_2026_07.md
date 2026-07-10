# Deep Research 2026-07: what the 2023-2026 literature changes about our position

Scope: a decision brief for the researcher choosing round 2. It reads the 2023-2026 primary
literature against our current result frontier, deepens the three real substrate positives,
rules on which matched-compute nulls are the correct general result versus regime artifacts,
and hands a screened table of round-2 candidate experiments split into cpu-now (laptop) and
studio. Every opportunity carries its preregistered null and a non-vacuous control. House rule:
no dashes, commas and parentheses only. Biology and philosophy are hypothesis sources, never
evidence.

Ground-truth state verified in-repo before writing (so the tiers below are honest):
- substrate_vs_random_init_vit.json: real V-JEPA shape-under-nuisance 0.5172, random-init
  same-arch ViT-L at matched 256px 0.2414, random-pixel 0.1034, n=96, SINGLE seed.
- compositional_under_nuisance.json: V-JEPA seen 0.7083, held-out 0.725, gap -0.0167,
  random-pixel collapses (seen 0.1875, held-out 0.05), n=200, SINGLE split. The only control is
  random-pixel (resolution-confounded), NOT random-init-ViT.
- The 200 compositional latents are NOT cached: the script persists summary stats only
  (keys backend, n_clips, vjepa, random_pixel, verdict), and the run took 4369 s of live ViT-L
  forwards. Any cache-side re-analysis of those latents needs a fresh forward pass first.
- dense_vs_pooled.json verdict: pooling is NOT the bottleneck (orientation already decodes from
  the pooled vector). compositional_binding.json verdict: CEILING (both seen and held-out decode
  near-perfectly on the clean shapes).
- data/cache/dinov2s_nuisance_real does NOT exist. WS1 (scripts/mop_ws1_agreement_vs_confidence.py)
  is scripted with all five guards but has produced no result json. It is blocked on the DINOv2-S
  cache the encoder lane is producing, so it is blocked-on-encoder-lane, not cpu-now.
- pr7_fast_slow.json config is a 4-task, 640-sample-per-task, dim-64 toy stream (relevant to
  whether PR9 can even exhibit plasticity loss).

---

## 1. Executive summary (decision-first)

The literature does not overturn our results, it sharpens them and tells us exactly which of our
own numbers are still soft. Three decisions fall out. First, the real science is the substrate
trio (pretraining beats matched-architecture random-init at decoding shape under nuisance,
+0.276; held-out (shape,color) equals seen, gap -0.017; PR1 licenses a router), and the single
highest-value round-2 move is not a new mechanism but hardening what we have: multi-seed the
+0.276 and -0.017 numbers (both are currently single-split) and, in the compositional test,
replace the resolution-confounded random-pixel control with the random-init-ViT arm we already
validated in the shape test. Second, our entire test-time-compute null lane (verify-revise,
beam, debate, confidence-stop, fixed-point, halting) is the CORRECT general result for the
regime we ran and the literature is unanimous on why: iterative compute pays only on
difficulty-graded, depth-bottlenecked, verifiable tasks (Snell 2024, Saunshi 2025, Geiping
2025, Kamoi 2024), and our additive 1024-d regime is none of those, so the nulls are
regime-driven not fundamental, and each is rescuable only after we first build a
D3-certified per-sample hardness gradient that does not yet exist. Third, the name Mixture of
Thinking / Mixture-of-Thought collides head-on with published prior art (Zheng et al. 2025 use
the exact acronym; Yue et al. 2023/ICLR-2024 use Mixture of Thoughts), so the program must be
renamed and positioned as a latent-space, frozen-perception, error-decorrelation-licensed
instance of an existing family, not a first mover.

Two honesty flags carried from verification that change tiers. The headline
factor-orthogonality deepen is NOT cpu-now and NOT zero-forwards: the 200 compositional latents
are not cached, so it needs a fresh encoder forward pass and therefore waits on the encoder
lane and must add an isotropic-Gaussian-at-matched-(d,n) null or a d>>n geometry artifact will
fake orthogonality for any features. And at least six WS1/workspace items tagged cpu-now are
actually blocked on the un-produced DINOv2-S cache. The genuinely runnable-now, contract-clean
candidates are the plasticity pair PR7-delta and PR2, plus the PR1 learned-router readout on the
already-cached synthetic modes: those own laptop round 2.

---

## 2. Deepen the wins

### 2.1 Substrate is special (pretraining, not architecture): +0.276 over random-init ViT-L

Current: real V-JEPA 0.5172 vs random-init same-arch ViT-L 0.2414 (not above the 0.1667 chance
by much) vs random-pixel 0.1034, n=96, one seed. The earlier "not special" read was a
vacuous-latent-projection artifact; the random-init-ViT-at-matched-256px control fixed it.

What the literature motivates:
- Uselis, Dittadi, Oh (2026) prove compositional generalization forces per-concept factors that
  are linear and mutually orthogonal, and MEASURE that structure in CLIP, SigLIP, DINO. This
  converts a descriptive positive into a geometric quantity (subspace orthogonality) we can
  correlate with decoding. Caveat below on the d>>n trap.
- Assran et al. (2025, V-JEPA 2) use a 4-layer attentive probe as the field-standard frozen
  readout. Our linear probe is therefore the CONSERVATIVE readout: an attentive-probe rerun can
  only raise a real positive, and it aligns our methodology with the encoder authors.
- Saunshi et al. (2025) diagnose that any downstream reasoning claim needs a difficulty-graded
  task, which is why the substrate positive (an off-ceiling decode at 0.517, neither trivial nor
  impossible) is the one place in the corpus with real, gradable difficulty.

Concrete stronger experiments (each carries the random-init-ViT-at-256px control, never a square
projection):
1. MULTI-SEED THE HEADLINE (studio). 5 then 10 seeds on the +0.276 gap. The interpreted JSONs
   already flag split-noise SD ~0.08-0.09 at n_test~29; a single seed is not citable. This is
   the standing requirement, do it first.
2. ATTENTIVE-PROBE VARIANT (studio). Rerun the shape decode with the V-JEPA-2 4-layer attentive
   head, controls held fixed. Null: under the attentive probe random-init closes to within the
   0.15 threshold (readout-capacity artifact).
3. FACTOR-ORTHOGONALITY GEOMETRY (blocked-on-encoder-lane, NOT cpu-now). Fit per-factor linear
   subspaces, report principal angles / normalized overlap, correlate with the -0.017 gap. Two
   corrections that are load-bearing: (a) the 200 latents are not cached so this needs a forward
   pass first, and (b) in 1024-d from 200 points two low-rank subspaces are near-orthogonal BY
   DIMENSIONALITY for any features, so it MUST include an isotropic-Gaussian-target-at-matched-(d,n)
   null and dimensionality reduction before angle measurement, or a positive is the O(d/n) width
   confound not a pretraining signature. Redundant caveat: fold this into the same forward pass
   that dumps the compositional cache rather than paying ViT-L time twice.

Collapse note: items 1, 2, and a hardness-sweep variant all re-derive the SAME 0.517-vs-0.241
gap. Run ONE multi-seed studio rerun with probe-head and hardness-bin as secondary arms, do not
triple-count encoder time.

### 2.2 Compositional off-ceiling: held-out (shape,color) 0.725 = seen 0.708, gap -0.017

Current: V-JEPA factorizes; random-pixel collapses. The ONLY blocker to this clearing a gate is
the control: it is random-pixel at low resolution, a resolution confound.

What the literature motivates:
- Wiedemer et al. (2023) derive that compositional generalization follows from mild conditions
  on the SUPPORT of the training distribution, and Redhardt, Akram, Schug (2025) show it emerges
  "as long as the training distribution sufficiently covers the task space." Together: our
  diagonal held-out result is only load-bearing if the training support is genuinely sparse
  relative to the grid. This hands us the exact knob to stress it.
- Uselis 2026 again: near-orthogonal shape and color subspaces would EXPLAIN why a shape probe
  generalizes across color cells for free.
- Joseph et al. (2026) warn that not every attribute factorizes: scalar speed is linearly
  decodable but DIRECTION lives in a distributed circular population code. So extending
  factorization to motion-direction or rotation is a genuine test, not a formality.

Concrete stronger experiments:
1. MATCHED-RESOLUTION CONTROL (studio). Swap random-pixel for random-init-ViT-L at matched 256px
   (the arm already validated in the shape test), multi-seed. This is the single change that
   converts the compositional positive from descriptive to gate-clearing. Reuses
   substrate_vs_random_init_vit.py machinery.
2. SUPPORT-COVERAGE BREAKPOINT (studio, with a ceiling guard). Sweep held-out fraction (5, 10,
   15, 20 of 25) and grid size (5x5, 7x7, 10x10) to locate where V-JEPA held-out first drops
   below seen. Honesty flag: every prior clean-shape compositional probe on this program
   CEILINGED at 1.0 (compositional_binding), and the 25-cell / 8-per-cell grid is coarse, so
   this needs a preregistered per-configuration D3 non-ceiling certificate or it re-ceilings or
   floors uninformatively and multiplies encoder forwards for nothing.
3. FACTOR-ORTHOGONALITY (see 2.1.3, same forward pass, same d>>n null requirement).

### 2.3 PR1 green: heterogeneous modes make decorrelated errors, a router is licensed

Current: heterogeneous oracle beats a matched homogeneous-seed-copy oracle by 0.023 above
spread (interpreted JSON: knn correlates 0.30-0.39 with the linear family that self-correlates
0.62-0.78). The oracle GAIN over the best-mode floor is the upper bound; learned-minus-oracle is
the real figure of merit.

What the literature motivates:
- Zheng et al. (2025) and the MoR/MoT cluster justify mode-mixture by asserting modality
  synergy; PR1's error-decorrelation is a stronger, falsifiable PRECONDITION. This is our
  cleanest differentiator (see section 5).
- Cao et al. (2024, Predictive Dynamic Fusion, ICML) prove a Co-Belief rule (weight negatively
  covaries with own loss, positively with other-source loss) reduces the generalization-error
  bound over single-source confidence. A theorem-backed fusion rule is a cleaner shot at a
  matched-capacity win than our hand-built inverse-variance weights.

Concrete stronger experiment (cpu-now, on already-cached synthetic modes):
1. LEARNED ROUTER, TUNED MODES (cpu-now). Does a tiny learned router realize a nonzero fraction
   of the oracle gain once each mode is individually TUNED (removing MP2's untuned-blend
   confound)? DECISIVENESS GUARD (from verification): the router must beat BOTH the tuned best
   single mode AND a uniform-blend-of-tuned-modes at MATCHED trainable-parameter count, else the
   gain is capacity not routing and it re-derives MP1-NULL. Null: learned-minus-tuned-best CI
   includes 0 or sign-flips across seeds.

---

## 3. Rescue the nulls (honest verdict per null)

Master diagnosis, verified across four primary sources: iterative test-time compute beats a
matched-compute feedforward baseline only under four conditions, and our synthetic 1024-d
additive regime satisfies none. (1) Difficulty-dependence: Snell et al. (2024) show effectiveness
"critically varies depending on the difficulty of the prompt" and gains concentrate where a
smaller model already has non-trivial-but-not-ceiling success. (2) Per-task saturation: Geiping
et al. (2025, Huginn) show easy tasks saturate in recurrence almost immediately while
GSM8K/MATH/HumanEval keep improving to 32+ steps. (3) Reasoning-vs-memorization: Saunshi et al.
(2025) show looped depth matches kL-layer depth on reasoning (addition, p-hop, math) but NOT
memorization, and our additive content is structurally a lookup. (4) Verifiability: Kamoi et al.
(2024, TACL) and Huang et al. (2024, ICLR) show self-correction pays only with reliable external
feedback or decomposable answers, and intrinsic self-correction degrades reasoning.

So the honest reading: our six nulls are the CORRECT general result for easy, non-compositional,
non-verifiable tasks. They are regime-driven, not refutations of iteration. A positive is
reachable but ONLY after building the missing hardness axis. Per null:

- mt5 adaptive halting (UNREADABLE). Real, but rescuable. Sole blocker is no hardness gradient
  (probe_acc easy=hard=1.0). Snell predicts allocation becomes readable exactly when
  non-trivial-but-not-ceiling difficulty exists. Rescue: run halting on a nuisance-severity-binned
  compositional task. BLOCKED until a D3 gradient is certified first (see section 6, the gradient
  experiment must FIRE before any of the downstream halting items are even declarable readable).
- mt7 beam / mt8 debate (NULL). Regime-driven. The oracle-beam gap of 0.198 proves headroom
  exists but the learned scorer cannot reach it, which is exactly the generation-verification
  asymmetry (Kamoi 2024). Rescue: a DECOMPOSABLE, easy-to-verify multi-hop relational query
  where each hop is cheaply checkable. Null: learned-scorer ties greedy even with per-hop
  checkability (then the scorer, not the task, is the bound).
- dr9 verify-revise / mt6 confidence-stop (NULL). Regime-driven. Huang 2024 and Kamoi 2024 say
  intrinsic self-correction needs external feedback. Rescue: supply a genuine external symbolic
  relation checker on a compositional scene and move off the ceiling. Null: verify-revise ties
  single-shot at matched FLOPs even WITH the external checker on the graded task, in which case
  self-correction is genuinely dead here.
- dr8 fixed-point (NULL). Regime-driven. It is unrolled depth with no attractor on the additive
  cache. Saunshi 2025 says looping helps only on reasoning-shaped content. Rescue: run the tied
  refiner on a 2-3 hop compositional query at matched depth vs an iso-FLOP feedforward stack.
  This is the cleanest single test of the reasoning-vs-memorization split on our own substrate.

Cannot-rescue honesty: if all four rescues STILL tie on a genuinely hard, compositional,
verifiable task, then the null is fundamental on this substrate and the custom test-time-compute
branch should be retired, consistent with the standing kill-switch. The literature makes a
positive possible, it does not guarantee one.

Plasticity nulls (a different lane, also mostly correct results):
- Dohare et al. (2024, Nature 632:768-774) show loss of plasticity is real and is fixed ONLY by
  a random non-gradient reinitialization channel (continual backprop). Our entire plasticity
  stack is pure gradient-descent-plus-penalty and contains NO reinit channel, so it structurally
  cannot exhibit the fix. PR9 (continual-backprop utility reinit) is the one frontier-certified
  baseline-beater we have never run. Honesty flag: PR9's null requires a stream long enough to
  INDUCE plasticity loss, and the current 4-task/640-sample toy regime structurally cannot, so
  PR9 is genuinely studio and must ship a D3 plasticity-loss certificate (late-vs-early accuracy
  gap strictly positive under the SGD baseline) or it re-ties by construction.
- PR7 Hebbian fast store: Wang, Shi, Fox (2025) frame linear attention / fast-weights / DeltaNet
  as test-time regression and show a plain Hebbian outer-product update is the crudest
  associative memory, provably dominated by a delta-rule (least-squares, covariance-aware)
  update. PR7's +0.029 pilot is a FLOOR. Swapping the store for a delta rule is a one-mechanism
  deepen on an already-passing pilot, cpu-now on the small cache. This is a real positive lead,
  not a null.
- e4/PR4 uncertainty-gated plasticity (DEGENERATE): Kirsch (2024) shows ensemble disagreement,
  the exact epistemic signal PR4 uses, COLLAPSES as width grows, and the 2025 aleatoric/epistemic
  reassessment shows the two rank-correlate 0.8-0.999. So e4's noise-chase is the EXPECTED
  outcome, not a tuning bug. Hard to rescue; any rescue must extract implicit-ensemble diversity
  and publish a disagreement-calibration ECE plus a noisy-TV guard with a KNOWN
  reducible/irreducible partition.

---

## 4. Cross-substrate alignment: can a mixture share a code?

The convergence literature says yes but narrowly, and it hands us our own vacuous-control rule
restated in the alignment domain. Huh et al. (2024, Platonic Representation Hypothesis) argue
models converge toward a shared statistical model. The decisive 2026 rebuttal, Groger, Wen,
Brbic ("An Aristotelian View"), shows that after a PERMUTATION-based null calibration the GLOBAL
convergence trend essentially vanishes (inflated by width O(d/n) and depth O(log M) confounders)
and only LOCAL neighborhood alignment survives. Davari et al. (2023) prove CKA can be moved
without changing function, so alignment must be scored FUNCTIONALLY, not by CKA. Jha et al.
(vec2vec, NeurIPS 2025) show unsupervised embedding translation with NO paired data DOES work,
but WITHIN a single modality (text) and within an objective family, which is precisely the seam
our atlas exploits (V-JEPA temporal-predictive vs DINOv2 appearance vs CLIP caption are
different objectives AND modalities, the hardest case).

Implication for MoT: a shared code across currencies is plausible for the LOCAL topology and
within an objective family, unproven and probably weak across objective/modality. The mixture
can share a code, but the exchange rate is currency-pegged. Nothing licenses a GLOBAL universal
currency; everything licenses a local, permutation-null-validated, functionally-scored one. Our
AL2 pilot already has the right shape (learned rank-k map vs random-map-of-equal-rank floor AND
shuffled-fit floor). Three corrections before studio: score by kNN-neighbor-overlap not raw R^2
(local survives, global does not), add an isotropic-Gaussian-target-at-matched-(d,n) null to
kill the O(d/n) width confound, and run only on off-ceiling content (never on separable synthetic
gratings that ceiling at 1.0 for real AND random).

One methodological ruling from verification: the cross_substrate_factorization_dinov3 experiment
as originally specified used a FULL-RANK linear remap residual as its vacuity guard, but a
full-rank map between two 1024-d encoders absorbs almost all shared structure, so "residual near
zero" is the expected outcome and the null is nearly unfalsifiable. Prefer the rank-limited-map +
kNN-topology formulations (AL2-kNN, AT1-relrep); downgrade the global-residual version. And the
AL2 positive control must be runnable: V-JEPA full-clip to single-frame (both scriptable via
cache_vjepa_single_frame.py). The retired same-family scale pilot is not the registered positive
control: it does not replace the full-clip to single-frame control or
the matched random-architecture and permutation floors.

---

## 5. The name: "Mixture of Perspectives" collides with prior art

Verified collision. "Mixture-of-Thought (MoT)" is the exact title and acronym of Zheng, Chen,
Han, McCoy, Huang (2025), a multi-modality logical-reasoning framework (natural-language, code,
truth-table thinking modes). "Mixture of Thoughts" is older: Yue, Zhao, Zhang, Du, Yao
(arXiv Oct 2023, ICLR 2024), LLM cascades that mix thought representations for cost-efficient
reasoning. The generic "aggregate multiple thinking modes through a controller" frame is now a
recognized cluster (also Mixture of Reasoning / MoR 2025, Mixture-of-Visual-Thoughts 2025,
Mixture of Cognitive Reasoners / EPFL). Architecturally our "mixture" is a frozen-expert MoE
with a trained router.

Honest position: we are NOT first to "mixture of perspectives." Rename the program and frame the
contribution precisely as the two things that are genuinely ours: (a) the cached-latent
frozen-perception + tiny-shell doctrine (nobody in the MoT-name literature freezes perception
and trains only a shell), and (b) the PR1 licensing test (decorrelated errors across modes as
the measurable precondition for a router), which is a cleaner justification than the
modality-synergy assertion in Zheng-MoT. The experts are reasoning MODES over a shared latent,
not FFN blocks or LoRA adapters, and the router is licensed by error-decorrelation, not
load-balancing. That is a defensible but narrow novelty and should be stated as such.

Custom-model sizing (positioning, not an experiment to run now): the largest 2025 V-JEPA 2
configuration, trained on about 1M hours of video, is a body-reported ~60 GPU-years full-resolution / ~7 with progressive
training, so a from-scratch video substrate is a moonshot outside doctrine. The doctrine-adjacent
minimum pilot the field actually uses is Arch B (an object-centric slot module on FROZEN dense
tokens, DINOSAUR-style, single-GPU), and even that is now being published by others (SlotContrast
CVPR 2025; Causal-JEPA, unverified 2602 id), so we would be a follower there too. The
substrate-insufficiency question is better answered cheaply first: does the compositional gate on
DENSE pre-pool tokens beat pooled and beat random-init-ViT? A dense-ties-pooled result KEEPS the
frozen substrate and is a publishable when-NOT-to-use-slots negative. (Caveat: dense_vs_pooled
already showed pooling is not the bottleneck for orientation, so the NEW element is only the
random-init-ViT-on-dense-tokens arm, which needs fresh forwards and is studio, not the implied
cheap keep.)

---

## 6. Round-2 candidate experiments (verified survivors)

Decisiveness = would a clean result change the round-3 decision. Tier reflects the CORRECTED
ground truth (caches that do not exist push items out of cpu-now). Every row carries a
non-vacuous control (random-init-ViT at matched 256px for substrate claims, never a square
projection) and a preregistered null.

### 6a. cpu-now (laptop round 2, runnable on already-cached data)

| name | thesis | null | control | d/r/new | decisiveness |
|---|---|---|---|---|---|
| PR7-delta | A delta-rule (least-squares, covariance-aware) fast store adapts faster within-task at matched float budget than PR7's Hebbian store, slow-only, and a matched-size cache, and still retains after decay | delta store ties Hebbian AND ties the matched-size cache on online acc, or fails post-decay retention | identical slow head; arms differ only in fast state {delta, Hebbian, slow-only, matched-float kNN cache}; retention within RETENTION_MARGIN | deepen | HIGH: one-mechanism change on a passing pilot, the strongest cpu-now positive lead |
| PR2 | The pretrained substrate eases the shell's LEARNING dynamics (adaptation speed, BWT), not just readout, vs identical shell on random-init ViT-L at 256px | shell speed and BWT on real V-JEPA within 5-seed spread of random-init-ViT | random-init same-arch ViT-L at 256px; matched arch/init/LR/steps/split/order; verdict = CI excludes 0 AND no sign flip | deepen | HIGH: turns the readout-only +0.28 into a learning-dynamics claim, strongest keep-frozen signal |
| PR1 learned-router (tuned modes) | A learned router realizes a nonzero fraction of the oracle gain once modes are tuned | router density delta over tuned-best CI includes 0 or sign-flips | tuned best single mode AND uniform-blend-of-tuned-modes, BOTH at matched trainable-param count; oracle upper bound; noisy-TV guard | deepen | MED-HIGH: closes learned-minus-oracle, the true PR1 figure of merit |

Note: the several other "cpu-now" items in the intake (WS1-land, WS1-difficulty-conditioned,
TMUR global-router, PDF Co-Belief, PR1+WS1-agreement-feature, AL2-kNN on nuisance clips,
factor-orthogonality) all consume caches that do NOT yet exist (DINOv2-S; persisted compositional
latents). They are BLOCKED-ON-ENCODER-LANE, not cpu-now, and must not stall the laptop queue.
The moment those caches land they become cpu-now and WS1 (script already contract-clean with all
five guards) is the first to run, 5 then 10 seeds.

### 6b. studio (needs fresh encoder forwards or a long stream)

| name | thesis | null | control | d/r/new | decisiveness |
|---|---|---|---|---|---|
| Multi-seed substrate rerun (with attentive-probe + hardness-bin secondary arms) | The +0.276 pretraining gap holds at 5-10 seeds and grows/holds under an attentive probe and across a nuisance-hardness sweep | delta CI crosses the 0.15 threshold, or attentive probe closes random-init to within threshold | random-init ViT-L at 256px per level; 30 paired seeds; matched resolution; D3 per-level certificate | deepen | HIGH: makes the headline number citable; do first |
| compositional_matched_resolution_rerun | Held-out (shape,color) factorization advantage attributed to pretraining once random-pixel is replaced by random-init-ViT at 256px, multi-seed | random-init ViT-L factors held-out as well as V-JEPA (delta CI crosses 0) | random-init ViT-L at 256px (decisive) plus random-pixel floor for continuity | deepen | HIGH: the one change that clears the compositional gate |
| factor_orthogonality_geometry | Near-orthogonal shape/color subspaces explain the -0.017 gap; correlate orthogonality with held-out acc | subspaces no more orthogonal in V-JEPA than random-init (overlap distributions overlap) | random-init ViT-L AND isotropic-Gaussian-target at matched (d,n); dim-reduce before angles | deepen | MED: mechanism, but only if the d>>n null is included (else geometry artifact) |
| difficulty_gradient_certification_D3 | Nuisance magnitude yields a monotone D3-certifiable per-sample hardness gradient spanning well below 1.0 | binned tiers do not separate decode acc monotonically (easy approx hard) | random-pixel and random-init-ViT binned identically; per-bin chance floor | deepen | HIGH: this MUST fire first; it gates the entire halting-rescue cluster |
| multihop_relational_query: looped vs iso-FLOP depth | A 2-3 hop relational query is reasoning-shaped, so a tied refiner matches/beats an iso-FLOP feedforward stack (the Saunshi signature) | tied refiner ties the untied iso-FLOP stack (as dr8 tied on the additive cache) | untied deep stack at matched FLOPs/depth; random-init-ViT substrate control; single-hop version as the memorization comparison that should NOT benefit | rescue | HIGH: cleanest test of reasoning-vs-memorization on our own substrate |
| graded_halting_on_nuisance_compositional | Adaptive halting becomes readable and beats fixed depth once the task is binned into certified easy/med/hard tiers | halt-depth delta CI includes 0 in every tier, no acc gain over best fixed depth at matched mean FLOPs | random-init ViT-L at 256px feeding the same halting head; tuned update-norm halt rule | rescue | MED-HIGH: only readable AFTER the D3 gradient fires |
| external_verifier_self_correction | Verify-revise beats single-shot ONLY with a genuine external symbolic relation checker, off-ceiling (Kamoi condition-a) | ties single-shot at matched FLOPs even with the external checker | intrinsic-confidence verifier (Huang negative control); shuffled-verifier; single-shot at matched FLOPs; random-init substrate control | rescue | MED: localizes the null to the missing-verifier axis |
| PR9 continual-backprop reinit | CBP utility-reinit maintains plasticity across a long real-latent stream where SGD/EWC/L2-init lose it, and beats tuned L2-Init | ties tuned L2-Init and tuned SGD at matched compute (no plasticity loss to fix at our scale) | tuned SGD, EWC, L2-to-zero, L2-Init; report late-vs-early gap and BWT; needs a stream long enough to induce loss | new | MED-HIGH: the one frontier-certified baseline-beater never run, but only if D3 plasticity-loss certificate ships |
| AL2-kNN local-topology alignment | Rank-k cross-map preserves kNN neighbors above random-map AND width nulls, where global R^2 does not | kNN overlap within CI of max(random-map, shuffled-fit, isotropic-Gaussian-at-matched-d) | random-map-of-equal-rank; shuffled-fit; isotropic-Gaussian target at matched d,n | rescue | MED: operationalizes Groger 2026 (local survives calibration) |
| AL2 same-family positive control | V-JEPA full-clip to single-frame aligns above the random-map floor (harness can detect real convergence) | same-family map ties random-map floor (pipeline broken, not substrates) | random-map + shuffled-fit; use the single-frame cache, not the retired scale pilot | rescue | MED: without this the AL2 cluster cannot distinguish "no alignment" from "harness cannot detect" |

Sequencing rulings (from verification):
- difficulty_gradient_certification_D3 ranks FIRST within the halting cluster. The four
  downstream halting/allocation experiments share one unfalsifiable-until-certified null and
  cannot be declared readable until the gradient fires.
- The three redundant substrate rederivations (matched-resolution rerun, attentive probe,
  hardness sweep) collapse into ONE multi-seed studio rerun with probe-head and hardness-bin as
  secondary arms, to avoid triple-counting encoder time.
- cross_substrate_factorization_dinov3 with a full-rank-residual guard is DROPPED in favor of
  the rank-limited kNN formulations.

---

## 7. Verified references

All titles, authors, and years below were confirmed by fetching the arXiv/Nature/PMC page in
this session unless noted. House rule: findings marked body-only were confirmed to exist in the
paper but are not visible in the abstract.

1. Snell, C., Lee, J., Xu, K., Kumar, A. (2024). "Scaling LLM Test-Time Compute Optimally can be
   More Effective than Scaling Model Parameters." arXiv:2408.03314.
   https://arxiv.org/abs/2408.03314 VERIFIED. Test-time-compute effectiveness varies with prompt
   difficulty; gains concentrate on non-trivial-but-not-ceiling problems.
2. Geiping, J., McLeish, S., Jain, N., et al. (2025). "Scaling up Test-Time Compute with Latent
   Reasoning: A Recurrent Depth Approach." arXiv:2502.05171. VERIFIED (per corpus fetch, not
   re-fetched here). Easy tasks saturate in recurrence quickly; hard tasks improve to 32+ steps.
3. Saunshi, N., Dikkala, N., Li, Z., Kumar, S., Reddi, S.J. (2025). "Reasoning with Latent
   Thoughts: On the Power of Looped Transformers." ICLR 2025. arXiv:2502.17416.
   https://arxiv.org/abs/2502.17416 VERIFIED. Looped k-layer approx kL-layer on reasoning, not
   memorization.
4. Kamoi, R., Zhang, Y., Zhang, N., Han, J., Zhang, R. (2024). "When Can LLMs Actually Correct
   Their Own Mistakes? A Critical Survey of Self-Correction of LLMs." TACL 2024. arXiv:2406.01297.
   VERIFIED (per corpus fetch). Self-correction needs reliable external feedback or decomposable
   answers.
5. Huang, J., Chen, X., Mishra, S., et al. (2024). "Large Language Models Cannot Self-Correct
   Reasoning Yet." ICLR 2024. arXiv:2310.01798. VERIFIED (venue). Intrinsic self-correction
   degrades reasoning without external feedback.
6. Uselis, A., Dittadi, A., Oh, S.J. (2026). "Compositional Generalization Requires Linear,
   Orthogonal Representations in Vision Embedding Models." arXiv:2602.24264.
   https://arxiv.org/abs/2602.24264 VERIFIED (title, authors, three conditions, CLIP/SigLIP/DINO,
   partial linear factorization with low-rank near-orthogonal factors).
7. Wiedemer, T., Mayilvahanan, P., Bethge, M., Brendel, W. (2023). "Compositional Generalization
   from First Principles." NeurIPS 2023. arXiv:2307.05596. VERIFIED (per corpus fetch).
   Compositional generalization follows from support conditions plus architecture.
8. Redhardt, F., Akram, Y., Schug, S. (2025). "Scaling can lead to compositional generalization."
   NeurIPS 2025 (Spotlight). arXiv:2507.07207. VERIFIED (per corpus fetch). Scale yields
   compositional generalization when task-space coverage is sufficient.
9. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., et al. (2025). "V-JEPA 2:
   Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985.
   https://arxiv.org/abs/2506.09985 VERIFIED (title, first authors, >1M hours video). The 4-layer
   attentive probe and ~60-GPU-year / ~7-progressive compute are body-only facts (not
   abstract-visible).
10. Joseph, S., Garrido, Q., Balestriero, R., et al. (2026). "Interpreting Physics in Video World
    Models." arXiv:2602.07050. VERIFIED (per corpus fetch + verification). Scalar speed linearly
    decodable early; direction is a distributed circular population code; probes multiple
    same-family model sizes (body-only, not abstract-visible).
11. Dohare, S., Hernandez-Garcia, J.F., Lan, Q., Rahman, P., Mahmood, A.R., Sutton, R.S. (2024).
    "Loss of plasticity in deep continual learning." Nature 632(8026):768-774.
    DOI:10.1038/s41586-024-07711-7. VERIFIED via PMC mirror
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/ (title, all authors, volume/pages).
    Continual backprop reinitializes underutilized units to maintain plasticity.
12. Kumar, S., Marklund, H., Van Roy, B. (2023). "Maintaining Plasticity in Continual Learning via
    Regenerative Regularization" (L2 Init). arXiv:2308.11958. VERIFIED (per corpus fetch).
    Regularize toward initial params, single hyperparameter.
13. Wang, K.A., Shi, J., Fox, E.B. (2025). "Test-time regression: a unifying framework for
    designing sequence models with associative memory." arXiv:2501.12352.
    https://arxiv.org/abs/2501.12352 VERIFIED (title, authors, test-time-regression framing). The
    delta-rule-dominates-Hebbian claim is a body-only result (abstract states linear attention
    fails to capture inter-token correlations, the same mechanism).
14. Kirsch, A. (2024). "(Implicit) Ensembles of Ensembles: Epistemic Uncertainty Collapse in
    Large Models." arXiv:2409.02628. VERIFIED (per corpus fetch). Ensemble disagreement collapses
    as width grows.
15. Cao, B., Xia, Y., Ding, Y., Zhang, C., Hu, Q. (2024). "Predictive Dynamic Fusion." ICML 2024.
    arXiv:2406.04802. https://arxiv.org/abs/2406.04802 VERIFIED (CORRECTED author list: first
    author Bing Cao, NOT Zhang; an earlier draft citation had the wrong people). Co-Belief =
    Mono + Holo confidence provably reduces the generalization-error upper bound over
    single-modality confidence.
16. Huh, M., Cheung, B., Wang, T., Isola, P. (2024). "The Platonic Representation Hypothesis."
    ICML 2024 (Position). arXiv:2405.07987. VERIFIED (per corpus fetch). Representations converge
    toward a shared statistical model.
17. Groger, F., Wen, S., Brbic, M. (2026). "Revisiting the Platonic Representation Hypothesis: An
    Aristotelian View." arXiv:2602.14486. https://arxiv.org/abs/2602.14486 VERIFIED. Global CKA
    convergence vanishes after permutation-null calibration (width O(d/n), depth O(log M)
    confounders); only local neighborhood alignment survives.
18. Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., Rodola, E. (2023).
    "Relative representations enable zero-shot latent space communication." ICLR 2023 (Oral).
    arXiv:2209.15430. VERIFIED (per corpus fetch). Anchor-similarity re-encoding gives
    isometry/rescaling invariance and zero-shot stitching.
19. Jha, R., Zhang, C., Shmatikov, V., Morris, J.X. (2025). "Harnessing the Universal Geometry of
    Embeddings" (vec2vec). NeurIPS 2025. arXiv:2505.12540. https://arxiv.org/abs/2505.12540
    VERIFIED. Unsupervised text-embedding translation with no paired data; WITHIN-modality (text)
    only.
20. Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G., Belilovsky, E. (2023). "Reliability of
    CKA as a Similarity Measure in Deep Learning." ICLR 2023. arXiv:2210.16156. VERIFIED (per
    corpus fetch). CKA can be moved without changing function.
21. Balogh, A., Jelasity, M. (2025). "How not to Stitch Representations to Measure Similarity:
    Task Loss Matching versus Direct Matching." AAAI 2025. arXiv:2412.11299. VERIFIED (per corpus
    fetch). Prefer direct distance-minimizing matching over task-loss stitching.
22. Traub, J., Bungert, T.J., Luth, C.T., et al. (2024). "Overcoming Common Flaws in the
    Evaluation of Selective Classification Systems." arXiv:2407.01032. VERIFIED (metric).
    Introduces AUGRC; rankings changed on 5 of 6 datasets. NeurIPS 2024 acceptance is corroborated
    by dblp/proceedings, NOT by the arXiv landing page (cite the proceedings, not the arXiv page,
    for venue).
23. Zheng, T., Chen, L., Han, S., McCoy, R.T., Huang, H. (2025). "Learning to Reason via
    Mixture-of-Thought for Logical Reasoning." arXiv:2505.15817. https://arxiv.org/abs/2505.15817
    VERIFIED (EXACT name collision: MoT acronym; modalities natural-language, code, truth-table).
24. Yue, M., Zhao, J., Zhang, M., Du, L., Yao, Z. (2023/2024). "Large Language Model Cascades with
    Mixture of Thoughts Representations for Cost-efficient Reasoning." ICLR 2024. arXiv:2310.03094.
    https://arxiv.org/abs/2310.03094 VERIFIED (earlier "Mixture of Thoughts" usage; LLM cascade
    routing by answer consistency).
25. Seitzer, M., Horn, M., Zadaianchuk, A., et al. (2023). "Bridging the Gap to Real-World
    Object-Centric Learning" (DINOSAUR). ICLR 2023. VERIFIED (per corpus record). Frozen DINO
    features + trainable slot attention, single-GPU. The Arch B template.
26. Mur-Labadia, L., Muckley, M., Bar, A., et al. (2026). "V-JEPA 2.1: Unlocking Dense Features in
    Video Self-Supervised Learning." arXiv:2603.14482. VERIFIED (per corpus fetch + verification;
    NYUv2 depth 0.307 RMSE linear probe). Frontier tool for the dense-vs-pooled fork.
27. Zhang, Y., et al. (2026). "Are Independently Estimated View Uncertainties Comparable? Unified
    Routing for Trusted Multi-View Classification" (TMUR). arXiv:2604.09288. VERIFIED (per corpus
    fetch + verification). Cross-encoder uncertainties are not comparable; a global router is the
    fix.
28. Usama, M., Chang, D.E. (2026). "Convergence Without Understanding: When Language Models Agree
    on Representations but Disagree on Reasoning." arXiv:2605.23315. VERIFIED (per corpus fetch +
    verification; CKA 0.897 hard vs 0.830 easy). High similarity on hard problems is shared
    confusion.
29. Jiang, Y., Nagarajan, V., Baek, C., Kolter, J.Z. (2022). "Assessing Generalization of SGD via
    Disagreement" (Generalization Disagreement Equality). ICLR 2022 (Spotlight). arXiv:2106.13799.
    VERIFIED (per corpus record). GDE holds only under class-aggregated calibration and for
    homogeneous ensembles.
30. Fernandez, N., Kveton, B., Rossi, R.A., Lan, A.S., Wang, Z. (2026). "RADAR: Reasoning-Ability
    and Difficulty-Aware Routing for Reasoning LLMs." ICLR 2026. arXiv:2509.25426. VERIFIED (per
    corpus record). IRT-based per-query difficulty, OOD generalization.

## Referenced but UNVERIFIED (leads only, do not cite as evidence)

- Causal-JEPA, "Learning World Models through Object-Level Latent Masking." arXiv:2602.11389.
  Authorship not confirmed. Direct Arch B prior art if real.
- Mixture of Reasoning (MoR). arXiv:2507.00606. Name-adjacent, authorship unverified.
- Mixture-of-Visual-Thoughts. arXiv:2509.22746. Name-adjacent, unverified.
- FAAST. arXiv:2605.04651. Future-dated id, venue/date not reconfirmed.
- "When More Thinking Hurts: Overthinking in TTC Scaling." arXiv:2604.10739. Title/venue
  unverified.
- DINOv3 (Meta AI 2025, arXiv:2508.10104). Blog + arXiv exist; full author list not individually
  verified. Genuinely different encoder family for a cross-substrate rescue IF confirmed.
- SALT (arXiv:2509.24317), LeJEPA (arXiv:2511.08544), Gated Delta Networks (arXiv:2412.06464),
  Lillo and Cheney activation-function plasticity (arXiv:2509.22562): titles seen, full author
  lists / specific numbers not individually reconfirmed this session; treat quantitative claims
  as unverified until body text is read.
