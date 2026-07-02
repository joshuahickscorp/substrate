# Section 15: Skepticism About Building Our Own Model (The Brake)

This section is the brake on the whole custom-model line. Its job is not to be fair to the idea of training
our own model, it is to argue against it as hard as the evidence permits, then to state precisely what
evidence would overcome each objection, and only then to propose the smallest custom-model pilot that could
survive those objections. It inherits and is subordinate to DOCTRINE_SYNTHESIS.md and to the corpus state
recorded there. Nothing below is a run result; every empirical claim traces to an experiment id on disk or
to a proposed experiment with a null and a baseline it must beat.

House style (BLACKHOLE.md): no em dashes or en dashes anywhere, commas and parentheses and colons only.

---

## 0. What "our own model" even means here, so the brake targets the right thing

"Custom model" is dangerously ambiguous and the ambiguity is where cargo cult hides. There are at least five
distinct things a person could mean, and they differ by roughly six orders of magnitude in compute and in
scientific risk. The brake applies to all of them, but with very different force.

- **(A) A custom trainable SHELL.** New head architectures, routers, predictors, planners, buffers over the
  FROZEN cached latents. This is not "our own model" in the dangerous sense at all. It is what the entire
  corpus already is. e7_sparse (sparse/gated heads) and ex2_latent_planning (latent MPC) are exactly this,
  and they are the two surviving positives. This is cheap, cached-latent-first, doctrine-compliant, and needs
  no brake. When someone says "let us build our own model" and means this, agree instantly.
- **(B) A custom ADAPTER on top of a frozen encoder.** A small learned re-projection, a trained pooling
  operator, a slot-attention front end, a learned probe that is deeper than linear. Still frozen substrate,
  still cached-latent-first if the adapter is post-cache, cheap. Needs almost no brake, only the standing
  controls (it must beat a random adapter of equal capacity and beat the pooled-linear baseline).
- **(C) FINE-TUNING or LoRA on the frozen encoder.** Unfreezing V-JEPA, or bolting trainable low-rank
  deltas onto its weights. This BREAKS the frozen-substrate doctrine directly (requires_grad=False is a
  code-enforced invariant, checked by a grad-free unit test in every experiment). It also breaks
  cached-latent-first: if the encoder weights move, every cached latent is stale and must be recomputed at
  the 21 s/clip CPU floor. This needs a hard brake and a very high bar.
- **(D) Training a NEW ENCODER from scratch (or continuing V-JEPA pretraining on our own video).** A
  ViT-scale perceptual module trained on pixels. This is frontier-adjacent compute (the thing the doctrine
  explicitly forbids: "no frontier compute"). It needs the hardest brake in the document.
- **(E) A custom END-TO-END model** (encoder plus shell trained jointly on pixels). This is (D) plus (C) plus
  giving up every amortization the program is built on. It is the maximal violation of every doctrinal
  pillar simultaneously. The brake here is close to absolute.

The rest of this section is about (C), (D), and (E), because (A) and (B) are just "keep doing the program".
When this section says "custom model" with no qualifier, it means training weights that are today frozen or
that do not yet exist, that is (C)/(D)/(E). The proposed pilot in section 7 is the smallest possible member
of class (D), deliberately chosen because (D) is the only class where the corpus has produced a positive
signal that even points at a custom encoder, and it is scoped to be killable in a few hours.

---

## 1. Why a custom model might be necessary (the honest steelman)

The brake is only credible if it first states the strongest case FOR custom training. There are exactly three
arguments that survive contact with the corpus, and one that does not.

**1.1 The substrate is now known to be SPECIAL, which means it could also be BOUNDED.** The single most
important recent result (substrate_vs_random.json) is that real V-JEPA decodes shape under heavy nuisance at
0.379 versus random-pixel untrained features at 0.069 (chance 0.167), a delta of +0.31 off-ceiling. For the
first time the program has valid evidence the frozen substrate carries real learned structure, not a
full-rank-projection artifact. The symmetric worry: if pretraining buys nuisance-invariance, then a DIFFERENT
pretraining objective, or pretraining on developmentally-ordered egocentric video, might buy a DIFFERENT and
better-suited geometry for the specific abstractions the program keeps failing to find (compositional binding,
factoring shape from color, object permanence). V-JEPA 2 was trained on internet video for a mask-prediction
objective that is not obviously the objective a developmental abstraction program would choose. If the ceiling
we keep hitting is a property of THIS encoder's objective rather than of frozen encoders in general, only a
custom encoder can test that, and that is a scientifically legitimate reason.

**1.2 The binding constraint is the TEST BED, and one honest reading is that the test bed cannot be fixed
without new data that no off-the-shelf frozen model was trained to represent.** The corpus is explicit that
synthetic gratings and hand-bound objects ceiling (dense_vs_pooled.json: every cell 1.0; compositional_binding
= 1.0) and are additive by construction. The deferred prerequisite is real natural video with non-additively
bound attributes. That is a DATA problem, not a model problem, and it is solvable by caching real video
through the EXISTING frozen encoder. So this argument does NOT actually justify a custom model, it justifies
better data through the frozen encoder first. It is listed here because it is the argument people will REACH
for, and it is a category error: the test bed being the constraint is a reason NOT to build a model yet.

**1.3 Moldability remains undemonstrated and might be structurally impossible on a frozen substrate.** The
corpus has no critical window (d6 substrate_specific_window=False), no path-dependence beyond an optimizer
artifact (y4 area ~0), no U-shaped signature (d4=0). ex15_rejuvenation shows plasticity loss only appears past
toy scale. One reading: the shell is too small and too downstream of a frozen geometry to exhibit
developmental dynamics, and only a model whose REPRESENTATION itself develops (an encoder that trains) could
show a critical window. This is a real argument, but it is weak, because the corpus has not yet run the
moldability battery on real cached latents at scale (item 6 of the open questions), so "frozen shell cannot
be moldable" is not yet established, only "toy frozen shell was not moldable".

**1.4 The one argument that does NOT survive: "a custom model would just be better / more ours / more
principled".** This is pure cargo cult and section 6 dismantles it. Ownership is not a scientific variable.

Net: only argument 1.1 (the substrate may be objective-bounded, and only a custom encoder can test that) is a
clean justification, and even it is contingent on results not yet on disk (the random-init-ViT control and the
compositional-under-nuisance control are both IN FLIGHT).

---

## 2. Why it is premature RIGHT NOW (the core of the brake)

Every one of the following is a reason the custom-model decision cannot be made yet, ranked by how decisively
it blocks.

**2.1 Two controls that would REDIRECT the whole question are in flight and not on disk.** The +0.31 nuisance
result has a named, acknowledged confound: V-JEPA saw 256px, random-pixel saw 32px (RAM-forced downsample), so
part of the delta is resolution, not learned weights. The honest post-audit delta is ~0.21 to 0.23.
substrate_vs_random_init_vit.py (real vs random-init same-arch ViT-L at the SAME 256px) is the gold-standard
control that isolates pretraining from architecture-plus-resolution, and its result is not yet written. If
that control shows the delta largely SURVIVES against a random-init ViT-L at matched resolution, the frozen
substrate is confirmed special and the fork tilts hard toward KEEPING it (custom encoder loses motivation). If
it shows the delta COLLAPSES (most of +0.31 was resolution or architecture, not pretraining), then the door to
"a differently-trained encoder could do better" opens a crack. You cannot rationally decide to train an
encoder before you know which world you are in. Deciding now is deciding on a coin you have not yet flipped.

**2.2 The frozen substrate has never been shown to be BOUNDED on any task we care about.** A custom encoder is
only justified if the existing one demonstrably CANNOT do something. The corpus shows the opposite trend
lately: the frozen encoder is special (+0.31), pooled-vs-dense found orientation decodes PERFECTLY from the
mean-pooled vector (dense_vs_pooled.json real_acc 1.0, refuting the "pooling destroys spatial structure"
hypothesis). Not one experiment has produced a clean "frozen V-JEPA hits a ceiling that is a property of its
weights and not of our additive test bed". Training a model to beat a ceiling nobody has located is building a
key for a lock that has not been found.

**2.3 The ceiling we DO see is a test-bed artifact, and custom training would inherit it.** Everything
ceilings at 1.0 because synthetic content is linearly separable and additive. If we trained a custom encoder
and evaluated it on the same synthetic gratings, it would ALSO ceiling, and we would learn nothing except that
we spent compute. Worse, a custom model on ceilinged data invites the most seductive cargo cult of all:
"our model gets 1.0 too, so it is as good as V-JEPA", a vacuous tie dressed as a win. The test bed must be
fixed FIRST, on the frozen encoder, or a custom model is untestable by construction.

**2.4 The vacuous-control lesson applies with full force to any custom-model claim.** The single most
important methodological correction in the corpus is that a full-rank projection is vacuous for probes. A
custom encoder evaluated by a linear or MLP probe against V-JEPA is at acute risk of the SAME class of error:
if both encoders produce full-rank features of the same dimension, a probe can absorb differences and report a
tie or a win that is a property of probe capacity, not of the representation. Any custom-model comparison MUST
use the non-vacuous controls (held-out combination generalization, nuisance-invariance deltas, trained-shell
continual-learning dynamics), and the program has only JUST learned how to do this correctly. Building a model
before the evaluation methodology is trustworthy repeats the exact mistake the corpus spent its most important
correction unlearning.

**2.5 Compute reality.** The stated hardware is an M3 Pro (~18 to 19 GB unified, CPU-bound at 21 s/clip for
real encoding, a hard per-buffer MPS ceiling on 64-frame ViT-L, ~60 GB disk floor already tight), plus a
future Mac Studio (Apple Silicon, NO CUDA) and a rented-CUDA path reserved for environment rollouts. Training a
ViT-L encoder from scratch on video is a multi-GPU multi-week job (V-JEPA 2 itself was trained on large GPU
clusters). Even continued-pretraining or a from-scratch SMALL encoder is far outside Tier 0 and Tier 1. The
program has no budget line for this and the doctrine forbids frontier compute. A custom encoder is not
"expensive", it is off the compute map the entire program is defined against.

**2.6 Amortization collapse.** The whole program is cached-latent-first: the frozen forward is paid ONCE and
every shell experiment reads ~8 KB pooled latents. The moment an encoder trains, the cache is invalid and
every downstream experiment pays the 21 s/clip forward again on every weight update. The economic engine of
the program (thousands of cheap shell experiments over a paid-once cache) is destroyed by unfreezing anything.

---

## 3. What frozen models, the shell, and the atlas can teach us FIRST (the cheaper substitutes)

Before any custom weights, four cheap moves extract most of what a custom model would tell us, at Tier 0 or
Tier 1 cost, on cached latents.

**3.1 Multi-encoder frozen census (already staged).** The repo has dinov2-large and VideoMAEv2-Base fully
downloaded, plus config stubs for V-JEPA 2 ViT-H and ViT-g. Running the nuisance-invariance and
held-out-combination tests across these DIFFERENT frozen encoders (different objectives: image-SSL DINOv2,
masked-video VideoMAE, mask-prediction V-JEPA) is the frozen, cheap, doctrine-compliant version of "does the
pretraining objective matter". This is the cross-substrate-convergence standing control (the newest one). If
all frozen encoders converge on the same ceiling and the same +delta pattern, the abstraction we want is
either universal or universally absent, and a custom objective is unlikely to add it. If they DIVERGE sharply
by objective, that is the FIRST real evidence that objective matters, and it costs zero training. This is
strictly dominant over training a custom encoder to test the same hypothesis.

**3.2 Encoder-SCALE falsifier (ex12_atlas).** Pull ViT-H and ViT-g (deferred only by the laptop disk floor,
trivial on the Studio) and ask whether bigger frozen perception raises decodability on the non-ceiling
nuisance test. If scale helps, the answer to "we need a better encoder" is "use a bigger FROZEN one", not
"train our own". If scale does not help, a custom SAME-SIZE encoder is very unlikely to help either.

**3.3 The adapter ceiling (class B).** Before training an encoder, train a small ADAPTER on the frozen cache
(a learned re-projection or slot front end) and measure how far it moves the non-ceiling metric. If a cheap
post-cache adapter closes most of the gap between frozen-V-JEPA and whatever target, the encoder is fine and
the shell was just too shallow. Only if the adapter saturates well below target is there residual signal that
lives in the encoder weights themselves.

**3.4 The atlas / decodability battery on REAL latents at scale (STUDIO_HANDOFF item).** The single
highest-value deferred delta is running ex12_atlas, d1_geometry, a1_affordance_decode, c1_held_out_combination
on REAL ViT-L pooled latents at larger scale instead of the synthetic Gaussian-cluster proxy. This tells us
what the frozen encoder actually can and cannot represent on real content, which is the exact map you need
before claiming it is bounded. Building a custom model before this map exists is navigating without it.

If all four of these are run and the frozen encoder still demonstrably cannot represent a target that matters,
THEN and only then does custom training have a locatable lock to open.

---

## 4. The exact failures that WOULD justify custom training (preregistered triggers)

A custom-model program is scientifically justified only if ALL of the following are true, and each is a
concrete, checkable outcome, not a vibe. These are the preregistered triggers; if they do not fire, the brake
holds.

- **T1 (substrate bounded, non-vacuously).** On the real non-ceiling non-additive test bed (real video,
  non-additively bound attributes), frozen V-JEPA fails a held-out-combination or nuisance-invariance target
  that a KNOWN-separable reference (difficulty_calibration D3) certifies is achievable, and the failure
  survives a trained adapter of matched capacity (section 3.3). That is: the information is provably present
  in principle, a cheap adapter cannot extract it, so the missing structure is in the encoder weights.
- **T2 (objective-specific, not scale-specific).** The multi-encoder census (3.1) and encoder-scale falsifier
  (3.2) show the failure PERSISTS across DINOv2, VideoMAE, and larger frozen V-JEPA. If a bigger or
  differently-trained FROZEN encoder fixes it, custom training is not justified (use that frozen encoder). T2
  fires only if no available frozen encoder clears T1.
- **T3 (the confound is resolved).** substrate_vs_random_init_vit.py has landed and shows the pretraining
  delta is real and largely resolution-independent, so "pretraining objective matters" is an established fact
  and not a resolution artifact. Without T3, T1 could be a resolution story that a bigger frozen input fixes.
- **T4 (a specific, testable objective hypothesis exists).** We can name the property the custom objective
  would install (for example, a slot/binding-structured objective that factors shape from color, or
  developmentally-ordered egocentric pretraining that installs object permanence) AND a preregistered metric
  on which the custom encoder must beat the best frozen encoder by a margin exceeding seed spread. "Train and
  see" is not a hypothesis.
- **T5 (moldability specifically requires a training representation).** If and only if the moldability battery
  (critical window, path-dependence, U-shape) is run on real cached latents at scale and comes back FLAT for
  the frozen shell, AND a theoretical argument shows the flatness is caused by the representation being fixed
  rather than by the shell being small, is a training-representation model justified for the moldability
  question specifically.

If T1 through T4 all fire, class (D) (a small custom encoder) is justified for the abstraction question. If
T5 additionally fires, a larger developmental-encoder program is justified. Absent these, custom training is
premature.

---

## 5. The exact successes that would make it UNNECESSARY (the off-ramps)

Symmetric to section 4: these are the outcomes that KILL the custom-model line entirely, and several are
likely given recent results.

- **S1.** substrate_vs_random_init_vit.py shows the +0.31 nuisance delta largely survives at matched
  resolution. The frozen substrate is confirmed special, the fork tilts to KEEPING it, custom encoder loses
  its main motivation. (This is the most probable near-term outcome and it directly weakens the whole line.)
- **S2.** compositional_under_nuisance.py shows the frozen substrate FACTORS shape from color off-ceiling on
  held-out combinations. Then the frozen encoder already does compositional abstraction and there is nothing
  for a custom objective to add on that axis. CAVEAT (the current script CANNOT fire this off-ramp honestly):
  as shipped, compositional_under_nuisance.py:125-131 gates its "substrate-specific" verdict on beating
  random_pixel_features, whose control (substrate_vs_random_features.py:104-113) avg-pools 256px down to 32px,
  so V-JEPA sees 64x more spatial info than the control. That is the exact resolution-confounded vacuous
  control section 2.1 flags (why the raw +0.31 is honestly ~0.21 to 0.23). A delta over 32px random pixels is
  NOT a compositional-factoring win, it is the resolution artifact. So S2 remains UNTESTED until the control is
  swapped to the random-init same-arch ViT-L at matched 256px (the gold-standard control, same as S1's
  substrate_vs_random_init_vit.py); the random-pixel version must be read as "resolution-confounded, not
  off-ramp-clearing" and must NOT close the custom-model line on the compositional axis.
- **S3.** A cheap post-cache ADAPTER (3.3) closes the gap to any target that matters. Then the encoder was
  never the bottleneck, the shell was, and the program stays in class (A)/(B) forever.
- **S4.** The multi-encoder census (3.1) shows cross-substrate CONVERGENCE: all frozen objectives hit the same
  ceiling and the same +delta pattern. Then the abstraction is universal-or-universally-absent and a custom
  objective is very unlikely to break the pattern.
- **S5.** The real-video non-ceiling test bed, once built, shows the frozen encoder CLEARS the compositional
  and binding targets. Then the ceiling was purely a synthetic-additivity artifact, the frozen encoder is
  sufficient, and the entire custom-model line was chasing a test-bed bug.

Any one of S1 through S5 substantially weakens the case; S3 or S5 alone would close it.

---

## 6. Cargo-cult model building vs scientifically justified model building

The distinction the brake must enforce, stated as a checklist so it can be applied mechanically.

**Cargo cult (reject on sight):**
- "We should train our own model so it is truly ours / end-to-end / principled / not dependent on Meta." Owner-
  ship and end-to-end-ness are not measured variables. The doctrine does not reward provenance, it rewards
  capability density per unit compute.
- "Big labs train their own encoders, so a serious program trains its own." Imitating the form of frontier
  labs (their models) without their compute, data, or the specific bottleneck that motivated their training is
  the definition of cargo cult. The program's whole thesis is that it wins on DENSITY precisely by NOT doing
  this.
- "The custom model gets 1.0 on our benchmark too." A tie on a ceilinged, additive, linearly-separable test
  bed is vacuous (section 2.3, and the full-rank-projection lesson of section 2.4). This is a vacuous win in
  new clothes.
- "Fine-tuning the encoder a little improved the downstream probe." If the probe is linear or MLP and the
  encoder is full-rank, this risks the vacuous-control artifact and, separately, breaks the frozen invariant
  and the cache. Improvement must be shown on a non-vacuous metric AND justified against the amortization
  collapse it causes.
- "We have a Mac Studio now, so we can afford to train." The Studio buys throughput, dataset scale, and
  replication (section 10), NOT a bigger trainable model. It has no CUDA. Confusing "more machine" with "train
  a model" is the tier confusion the compute-tiers section warns against.

**Scientifically justified (the only admissible form):**
- The frozen encoder is shown BOUNDED on a certified-achievable, non-ceiling, non-vacuous target that a cheap
  adapter cannot reach (T1), the bound is objective-specific not scale-specific across available frozen
  encoders (T2), the resolution confound is resolved (T3), a specific installable property and a
  preregistered beat-the-best-frozen-encoder margin exist (T4), and the smallest model that could test the
  hypothesis is chosen (section 7). The custom model then has a null hypothesis (the custom objective ties the
  best frozen encoder on the target metric), a baseline it must beat (best frozen encoder, and a
  matched-compute random-init encoder), and a preregistered failure interpretation.

The dividing line: justified model building starts from a LOCATED failure of the frozen substrate and a NAMED
property to install; cargo cult starts from wanting a model.

---

## 7. The minimum custom-model pilot (proposed, class D, deliberately tiny)

Only ONE pilot is proposed, and it is scoped to be the smallest thing that could move the custom-model
decision, killable in a few hours on the available hardware, with a preregistered null and a
better-than-frozen bar. It is class (D) (a from-scratch tiny encoder), NOT (C) fine-tuning V-JEPA (which
breaks the frozen invariant and the cache) and NOT (E) end-to-end (maximal violation). It is deliberately a
DIAGNOSTIC pilot: its purpose is to test whether the OBJECTIVE matters at all at a scale we can afford, before
anyone proposes a real encoder run.

**CM1: minimum-objective encoder probe.** Train a DELIBERATELY tiny encoder (a small ViT or even a shallow
conv-plus-transformer, on the order of 1 to 5M params, on the SAME nuisance-clip content used by
substrate_vs_random_features.py, at the SAME 256px input to eliminate the resolution confound by
construction) under TWO contrasting self-supervised objectives: (a) a mask-prediction objective (the V-JEPA
family objective, scaled down) and (b) an invariance/contrastive objective that explicitly pulls together the
same shape under nuisance transforms. Freeze each after training and run the EXACT substrate_vs_random
nuisance-invariance and held-out-combination protocol on its features. The question is narrow and answerable
at toy scale: at fixed tiny capacity and fixed data, does the OBJECTIVE change the nuisance-invariance and
compositional-factoring delta, and does either tiny custom objective beat the random-init same-arch encoder
(the gold-standard control) by more than seed spread. This does NOT try to beat V-JEPA (a tiny encoder on toy
data cannot and should not), it tries to establish whether objective choice is a live lever at all before
anyone spends real compute on a real encoder.

- Null hypothesis: at matched tiny capacity, matched data, and matched 256px resolution, both custom
  objectives tie the random-init same-architecture encoder on nuisance-invariance and held-out-combination
  accuracy (objective is not a lever at this scale; the +0.31 was scale/data/architecture, not objective).
- Baseline it must beat: the random-init same-arch encoder (the gold-standard control from
  substrate_vs_random_init_vit.py), AND the two must be compared to each other, so the invariance objective
  must beat the mask objective by more than seed spread for "objective matters" to fire.
- Metric: held-out-combination accuracy and nuisance-invariance delta over chance, with the D3
  difficulty-calibration certification that the target is achievable, and NON-vacuous controls only (no
  full-rank-projection comparison).
- Failure interpretation: a tie (null holds) is a STRONG negative that closes the custom-encoder line, because
  if objective does not move the metric even when we control capacity, data, and resolution perfectly, then a
  bigger custom encoder is very unlikely to be worth frontier compute. A win by the invariance objective is
  NOT a green light to train a real encoder, it is the single result that would justify PROPOSING (not yet
  running) a scaled study, and only after T1 through T4 have also fired on the real test bed.
- Compute honesty: this is a real training job (it violates cached-latent-first, because the point is to test
  the encoder objective), so it must run on the Studio or a small rented slice at TINY scale, NOT on the M3
  Pro while any V-JEPA encode is in flight, and it must be scoped (few M params, hundreds of clips, minutes to
  a few hours) so a null result costs almost nothing. It is a probe of the DECISION, not a step toward a
  product encoder.
- Why exactly one pilot: the honest state is that the custom-model question is dominated by two in-flight
  controls (S1/S2) and a data problem (the real test bed). CM1 is the ONLY custom-training experiment that
  earns its violation of the doctrine right now, because it directly tests the single clean justification
  (argument 1.1, objective-boundedness) at a scale where a negative is cheap and decisive. Any second custom
  experiment would be premature until CM1 and the in-flight controls report.

---

## 8. Standing-control wiring for any custom-model claim (non-negotiable)

Every custom-model result, including CM1, is gated by the standing controls exactly as every other experiment
is, with three that bite hardest here.

- **Non-vacuous substrate control.** Never compare a custom encoder to V-JEPA (or to random features) with a
  full-rank-projection probe. Use random-init same-arch (isolates pretraining/objective from
  architecture-plus-resolution), held-out-combination generalization, and nuisance-invariance deltas.
- **Matched compute and matched resolution.** A custom encoder must be compared at matched input resolution
  (the resolution confound is the whole reason CM1 fixes 256px) and matched capacity, or any delta is an
  architecture/resolution artifact, not an objective result.
- **Cross-substrate convergence.** The custom objective's result must be read against the multi-encoder frozen
  census: if the custom objective merely reproduces what a cheaper frozen encoder already does, it is not a
  win, it is a vacuous re-derivation at frontier cost.

The brake position, stated once more plainly: do not train weights that are currently frozen or that do not
yet exist until (T1 through T4) fire on a real non-ceiling test bed, and until the two in-flight controls
(random-init ViT, compositional-under-nuisance) have reported. The cheapest thing that could change this
verdict is not a custom model, it is finishing the controls already in flight and building the real-video test
bed on the frozen encoder. CM1 exists only to make the eventual custom-encoder decision on the basis of a
cheap, controlled objective probe rather than on the basis of wanting a model.
