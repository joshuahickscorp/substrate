# Section 6: From Multiple Substrates to Multiple Cognitive Currencies

## 0. Frame and discipline

The doctrine says: one frozen substrate (V-JEPA 2 ViT-L, 1024-dim, `requires_grad=False`, called only
under `no_grad`), a tiny trainable shell, cached-latent-first, no frontier compute. This section does
NOT propose abandoning that. It proposes taking the newest standing control seriously: cross-substrate
convergence. A property that holds for exactly one substrate is a property OF that substrate, not a
property of cognition. A property that holds across many is a candidate universal. We cannot tell which
we have until we run more than one substrate. That is the whole reason this section exists.

The organizing idea: a substrate is not just "an encoder". It is a **cognitive currency**, a specific
kind of value the shell can spend on downstream reasoning. V-JEPA's currency is temporal-predictive,
motion-and-invariance-denominated. DINOv2's is appearance-and-part denominated. CLIP's is
caption-denominated. A small LLM's hidden states are token-prediction denominated. These are not
interchangeable. The shell that spends one currency well may be bankrupt in another. The point of a
cross-substrate atlas is to learn the exchange rates: which factors are decodable in which currency,
which currency is dominant for which reasoning, and (the load-bearing question) which results are
currency-agnostic (substrate-universal) versus currency-pegged (modality/objective/architecture-specific).

Hard warnings carried in from the corpus, wired into everything below:

- **The vacuous-control trap.** `frozen_random_projection` is a square full-rank 1024x1024 Gaussian, so
  a linear or MLP probe absorbs its inverse and the delta is mathematically forced to 0.000. Every
  probe-based "real ties frozen-random" result was measuring a property of a full-rank matrix, not of the
  substrate. In this section, the ONLY admissible substrate control is real-features-vs-random-ENCODER
  features (a genuinely different feature space), never a square projection of the same latent. This is a
  non-negotiable design constraint on every AT/AL experiment here.
- **The ceiling trap.** Synthetic gratings and hand-built bound objects are trivially separable in 1024-d
  (`dense_vs_pooled.json` and `compositional_binding.json` both ceiling at 1.0 for real AND random). Any
  cross-substrate comparison run on such content will report ties for a reason that has nothing to do with
  substrates. Cross-substrate work is only meaningful on NON-ceiling content: heavy-nuisance shape identity
  (the +0.31 regime) or real natural video with non-additively bound attributes.
- **The resolution confound.** The one landed positive (real V-JEPA 0.379 vs random-pixel 0.069 under
  nuisance) is contaminated by 256px-vs-32px. Any cross-substrate delta must be read the same way: control
  input resolution and preprocessing, or the delta is partly a pipeline artifact. Same-arch-random-init at
  same resolution (the in-flight `substrate_vs_random_init_vit.py` design) is the template.

Nothing below claims a substrate is special until it beats a random-ENCODER control at matched resolution
on non-ceiling content, and holds across seeds. That is the price of entry.

---

## 1. The candidate substrates, each as a cognitive currency

Each entry answers the eleven required questions: what it naturally represents; what it likely cannot;
what reasoning it supports; what plasticity shell it needs; what memory it pairs with; its characteristic
failure; how it talks to the global workspace; how to align it with others; its random-init control; its
matched baseline. All framing is provisional until run; every claim carries an implied null.

### 1.1 V-JEPA 2 ViT-L (the incumbent, canonical, `vjepa2-vitl-fpc64-256`)

- **Naturally represents.** Temporal-predictive structure of video: motion, short-horizon dynamics, and
  (the one landed positive) invariance to position/scale/rotation/color/clutter, i.e. nuisance-robust
  shape identity. Trained by masked latent prediction, so its currency is "what stays predictable across
  time and appearance change".
- **Likely cannot.** Fine-grained static appearance and text-alignable semantics (no caption objective).
  Long-horizon planning structure (fpc64 is a short clip). Anything requiring a spatial index after mean
  pooling (though `dense_vs_pooled.json` shows single spatial factors survive pooling, so this is milder
  than assumed). Compositional factoring of non-additively bound attributes is UNKNOWN, not disproven
  (`compositional_under_nuisance.py` is the pending test).
- **Reasoning it supports.** Short-horizon latent MPC (the ex2 positive: MPC beats flat reactive and
  action-shuffle on true dynamics), continual retention on latent streams, invariance-dependent
  classification.
- **Plasticity shell.** Small predictor + heads; sparse/gated heads (e7_sparse) halve forgetting. No
  demonstrated need for biological plasticity signatures (all refuted).
- **Memory it pairs with.** A latent replay buffer over pooled 1024-d vectors; prioritized replay bought
  no clean win over random (e2), so a plain buffer is the honest baseline.
- **Characteristic failure.** Ties a random projection on any easily-separable probe (the whole ceiling
  problem). Amplifies error on aleatoric noise if neuromodulated (e4, wrong direction 30/30).
- **Global-workspace interface.** Broadcasts a single pooled 1024-d state per clip; naturally a
  low-bandwidth global broadcast, weak at addressing.
- **Alignment with others.** As the video anchor in a shared-latent alignment: learn a thin linear (or
  low-rank) map from each other substrate into V-JEPA space, or into a neutral joint space; validate the
  map beats a random map of equal rank (the alignment-artifact control).
- **Random-init control.** `substrate_vs_random_init_vit.py`: random-init same-arch ViT-L at 256px. THIS
  is the gold standard, isolating pretraining from architecture and resolution.
- **Matched baseline.** Random-pixel features at matched resolution; a tuned linear probe on raw
  downsampled pixels.

### 1.2 V-JEPA 2 (ViT-H / ViT-G scale variants)

- **Naturally represents.** Same currency as ViT-L, more of it: the question is whether scale buys MORE
  nuisance-invariance or the same asymptote. A pure scaling arm.
- **Likely cannot.** Change modality or objective; scaling does not add a caption or a text axis.
- **Reasoning supported.** Same as ViT-L; the experiment is whether the +0.31 nuisance delta grows.
- **Plasticity shell / memory.** Identical to ViT-L; the shell must not grow with the substrate (doctrine:
  tiny shell). Cache footprint grows if dim grows, a real cost.
- **Characteristic failure.** Diminishing returns invisible under ceiling content; only the nuisance regime
  can see it.
- **Workspace / alignment / controls.** Same as ViT-L; the random-init control is the same-arch random-init
  at the larger scale, and the matched baseline is ViT-L itself (does bigger beat smaller off-ceiling).

### 1.3 V-JEPA 2.1 dense (announced, NOT on HF, `available: false`)

- **Naturally represents.** Hypothetically a DENSE token interface (no forced mean-pool), i.e. a spatial
  index V-JEPA-L lacks at the pooled interface.
- **Likely cannot.** Be tested at all today; weights are not published (`models.yaml` pins it unavailable,
  `replaces_canonical: false`). Any claim about it is speculative and must be flagged so.
- **Reasoning supported (IF it lands).** Spatially-indexed binding, object-before-pool routing (the P7 lane).
  But note the pooling-bottleneck motivation is already WEAKER than assumed (orientation survives pooling),
  so dense-2.1's advantage must be shown on genuinely compositional non-additive content, not asserted.
- **Everything else.** Deferred until weights verify on HF. This row exists to hold the fork open honestly,
  not to justify a branch.

### 1.4 DINOv2 ViT-L (image, `facebook/dinov2-large`, auxiliary)

- **Naturally represents.** Self-distilled static appearance: parts, textures, object identity from single
  frames. Strong dense patch features, strong at part correspondence.
- **Likely cannot.** Represent motion or temporal dynamics (image-only, no time axis). Anything requiring
  the predict-across-time currency V-JEPA has.
- **Reasoning supported.** Static part-whole, viewpoint correspondence, appearance-based retrieval; a
  natural complement to V-JEPA's motion currency.
- **Plasticity shell.** Same tiny shell; but heads tuned to dense patch tokens, not a pooled vector.
- **Memory it pairs with.** A patch-indexed memory (slots per region), unlike V-JEPA's single-vector buffer.
- **Characteristic failure.** Blind to dynamics; will tie chance on any motion-only factor. Also ceilings
  on easy static content, same trap.
- **Global-workspace interface.** Rich dense broadcast (many patch tokens), higher bandwidth than V-JEPA's
  pooled state; the workspace must down-select.
- **Alignment with others.** Frame-aligned: on the same clips, does a linear map from DINOv2 frame features
  predict V-JEPA clip features (and vice versa) above a random-map floor? A convergence-vs-complementarity
  test.
- **Random-init control.** Random-init same-arch ViT-L (image), same resolution.
- **Matched baseline.** V-JEPA on single frames (strip its time axis) and random-pixel image features.

### 1.5 CLIP (image-text, auxiliary)

- **Naturally represents.** Caption-alignable semantics: the axes a human sentence would name (object
  category, scene type, salient attribute). Its currency is explicitly language-shaped.
- **Likely cannot.** Represent fine dynamics, or anything a caption would not mention; and critically for
  Brain's SECOND doctrinal question (abstraction NOT routed through language), CLIP is the WRONG substrate
  to claim non-linguistic abstraction from, because its objective IS linguistic. Any abstraction result on
  CLIP is objective-specific by construction. This makes CLIP valuable precisely as a CONTRAST: a factor
  decodable in V-JEPA but NOT in CLIP is candidate non-linguistic structure.
- **Reasoning supported.** Zero-shot-style category readout; a semantic anchor for alignment.
- **Plasticity shell / memory.** Same tiny shell; pairs with a caption-keyed memory if one exists (it does
  not in the current shell, so this is a lane, not a claim).
- **Characteristic failure.** Reports what is nameable, misses what is not; will look "abstract" for a
  linguistic reason.
- **Global-workspace interface.** A semantic broadcast that is trivially human-readable, hence the
  interpretability-vs-usefulness tension (s9) surfaces here.
- **Alignment.** The natural pivot for cross-modal alignment (S2/S8 lanes), but the alignment-artifact
  control is essential: a random map into CLIP space can look aligned via linguistic priors.
- **Random-init control.** Random-init same-arch ViT; matched baseline is V-JEPA and DINOv2 on identical
  content, to separate "nameable" from "perceptual".

### 1.6 Small frozen LLM hidden states (text-only substrate)

- **Naturally represents.** Token-prediction structure: syntax, co-occurrence, symbol manipulation.
  Currency is discrete-symbolic and sequential.
- **Likely cannot.** Represent anything perceptual or continuous-dynamical without a perceptual front-end;
  grounding is absent by construction. This is the substrate that would make an abstraction claim
  MAXIMALLY suspect for Brain's second question (all symbols, all language).
- **Reasoning supported.** Symbolic/programmatic manipulation, sequence completion; a hostile control for
  "abstraction without language" (if a factor is ONLY decodable here, it is linguistic).
- **Plasticity shell / memory.** Shell reads hidden states as the latent; pairs with a token/episodic
  buffer. Same tiny-shell discipline.
- **Characteristic failure.** Fluent-but-ungrounded; the classic private-language / idiolect failure the
  corpus already saw (p5/s5/y3 below frozen-random floor).
- **Global-workspace interface.** Serial token broadcast; naturally a language-of-thought bottleneck, which
  is exactly what p9 (thought without language) says is NOT where the gains are.
- **Alignment.** Grounding alignment: map perceptual latents to LLM hidden states via a thin adapter, test
  above a random-adapter floor. This is the substrate whose alignment-artifact risk is highest.
- **Random-init control.** Random-init same-arch transformer (untrained LM); matched baseline is a bag-of-
  tokens / n-gram feature.

### 1.7 Audio / speech SSL (e.g. an SSL speech encoder, auxiliary)

- **Naturally represents.** Temporal acoustic structure: prosody, phonetic and event-level regularity.
  A SECOND temporal-predictive currency, in a different modality from V-JEPA, which makes it the cleanest
  test of modality-specific-vs-objective-specific (same objective family, different modality).
- **Likely cannot.** Represent anything visual or spatial.
- **Reasoning supported.** Temporal prediction and segmentation in audio; cross-modal event binding if
  paired with video.
- **Plasticity shell / memory.** Same tiny shell; a temporal buffer, like V-JEPA's, over audio latents.
- **Characteristic failure.** Ties chance on any non-acoustic factor; ceilings on easy audio.
- **Global-workspace interface.** Temporal broadcast; a natural partner to video in a multimodal workspace.
- **Alignment.** Audio-video temporal alignment on the same clips (do event boundaries co-locate above a
  shuffled-time floor). This directly tests whether "temporal-predictive invariance" is a universal or
  a V-JEPA-visual peculiarity.
- **Random-init control.** Random-init same-arch audio encoder; matched baseline is raw spectrogram features.

### 1.8 Image-only vs video-only (a modality-axis pair, not new weights)

- These are not separate weight sets but a DESIGN AXIS realized by DINOv2 (image-only) vs V-JEPA
  (video-capable), and by stripping V-JEPA's time axis (single-frame) vs full clip. The atlas question:
  which factors need TIME (decodable video-only, chance image-only) versus which are static (decodable
  either way). This is the sharpest lever for the modality-specific verdict, and it requires NO new
  substrate, only a controlled ablation of the time axis, which keeps it cpu-now.
- **Characteristic failure to guard.** Confounding "needs time" with "needs more tokens" (a video clip has
  more frames hence more raw information); the matched baseline must equalize token/frame count.

### 1.9 Symbolic / programmatic substrate (handcrafted or LLM-emitted programs)

- **Naturally represents.** Exact, compositional, extrapolating structure: a program generalizes to
  held-out combinations by construction. This is the currency that trivially PASSES the compositional test
  the perceptual substrates ceiling on, which makes it the ideal UPPER-BOUND reference, not a substrate to
  ship.
- **Likely cannot.** Be grounded in perception; it is the opposite failure mode from the LLM (too crisp,
  not too fluent).
- **Reasoning supported.** Systematicity, held-out-combination extrapolation (the c-series questions).
- **Use in the atlas.** As the difficulty-calibration reference (D3-style): if a factor is decodable in
  the programmatic substrate but NOT in any perceptual one, the perceptual substrates are genuinely
  bounded on it, which is a real negative about the substrate, not the test. This is how we distinguish
  "test too easy" from "substrate too weak".
- **Random-init control / matched baseline.** N/A in the usual sense; it IS the ceiling. Its role is to
  make other substrates' ties interpretable.

### 1.10 Handcrafted latent descriptors (classical features: HOG/optical-flow/color-histogram-style)

- **Naturally represents.** Exactly the hand-specified factor and nothing else. The honest low bar: a
  learned substrate must beat these to justify its existence.
- **Likely cannot.** Generalize beyond the hand-specified axis; brittle to nuisance (the whole point of
  the nuisance test is that these should FAIL it, and random-pixel features at 0.069 already show the
  no-invariance floor).
- **Use in the atlas.** The matched baseline of first resort. Cheap, interpretable, cpu-now. If a learned
  substrate does not beat an optical-flow descriptor on a motion factor, it earned nothing on that factor.
- **Random-init control.** They ARE deterministic; the control is a random linear combination of the same
  descriptors (a random read-head), to check the probe is not doing the work.

### 1.11 Random-init controls (the spine of the whole section)

- **What they represent.** Architecture + resolution + preprocessing, WITHOUT learned weights. The
  difference between real and random-init is exactly "what pretraining bought", isolated from everything
  else. This is not a substrate to reason with; it is the denominator of every substrate claim.
- **Two flavors, do not confuse them.**
  1. Random-ENCODER / random-INIT (untrained network of the SAME architecture at the SAME resolution):
     the CORRECT, non-vacuous control. `substrate_vs_random_init_vit.py`.
  2. Random-PIXEL (random projection of downsampled raw pixels): a genuinely different feature space, also
     valid, but confounded by resolution when the real encoder sees higher-res input (the +0.31 caveat).
  3. Random square PROJECTION of the real latent: VACUOUS for probes, forbidden as a substrate control here.
- **Characteristic role.** Every AT/AL row below names which random control it uses; a row without a
  non-vacuous random control is INVALID by the harness rule (`needs_real`).

### 1.12 Eventual custom substrate (JEPA / object-centric / action-conditioned)

- **Naturally represents (hypothetically).** Object-centric slots (binding before pooling, the P7 lane),
  or action-conditioned dynamics (the ex2 planning positive suggests action-conditioning is where the
  signal is). This is the "go custom" branch of THE fork.
- **Likely cannot be justified yet.** The fork is explicitly REOPENED but unanswered: no test has shown
  the frozen substrate is BOUNDED. Building custom weights is frontier compute and violates doctrine
  UNLESS a bound is demonstrated first. So this row's real content is a PRECONDITION, not a plan: custom
  is justified only after (a) a non-ceiling non-additive compositional test exists AND (b) the frozen
  substrate demonstrably fails it while the programmatic reference passes.
- **Reasoning supported (if built).** Compositional binding, factored slots, planning over object states.
- **Alignment / controls.** Its own random-init (same custom arch, untrained) is the mandatory control;
  its matched baseline is the frozen V-JEPA shell it must beat, at matched compute (a custom model that
  ties frozen-plus-shell at equal FLOPs earned nothing).

---

## 2. The exchange-rate summary (what pairs with what)

A compact reading of Section 1, meant as a hypothesis table, not a result:

- **Temporal-predictive currencies** (V-JEPA, audio SSL): pair with a temporal buffer, support short-horizon
  MPC, fail on static-only factors. Convergence between them (video and audio) would be the strongest
  evidence for a substrate-UNIVERSAL temporal-invariance property.
- **Appearance/part currencies** (DINOv2, CLIP): pair with patch-indexed / caption-keyed memory, support
  static correspondence and category readout, fail on dynamics.
- **Symbolic currencies** (LLM, programmatic): pair with episodic/token memory, support systematicity, fail
  on grounding. Their role is as CONTRAST and CEILING, not as Brain's cognitive core (the corpus already
  says the gains are not in the language-of-thought bottleneck).
- **The workspace** is where currencies are exchanged: a global broadcast that must accept heterogeneous
  bandwidths (V-JEPA's one vector, DINOv2's many patches, an LLM's token stream). The alignment maps ARE
  the exchange rates, and every one needs the alignment-artifact control (a random map of equal rank),
  because a linguistic or full-rank map can fake alignment.

---

## 3. The Cross-Substrate Atlas Protocol

The protocol classifies EVERY result into exactly one of nine verdicts. It extends `build_atlas_row.py`
(one row = one factor x one encoder, with a shuffle-label chance floor and a reproducibility level) to a
GRID: rows = factors, columns = substrates, plus the mandatory control columns. A cell is a decodability
number with its chance floor and seed spread. A RESULT is a pattern across the grid.

### 3.1 The grid

- **Rows (factors).** Only NON-CEILING factors qualify: nuisance-invariant shape identity (the 0.379 vs
  0.069 regime), and (once cached) real-natural-video bound attributes (color/shape/position/motion,
  non-additive). Any factor that ceilings for real AND random is DROPPED from the atlas (it can produce
  only probe-specific artifacts).
- **Columns (substrates).** Each Section-1 substrate that is `available: true`, plus its own random-init
  control column at matched resolution, plus the programmatic ceiling reference column.
- **Cells.** Linear-probe accuracy with shuffle-label floor, 5-seed spread, and the delta-over-random-init
  (NOT delta-over-square-projection, which is banned).

### 3.2 The nine verdicts (mutually exclusive, applied to a pattern across the grid)

1. **substrate-universal.** Factor decodable ABOVE its random-init control in EVERY substrate that
   plausibly carries it (across modality AND objective AND architecture), with consistent seed sign.
   The rare, valuable verdict. Candidate universals: temporal-invariance if video AND audio both beat
   their random-inits.
2. **modality-specific.** Decodable above control only within one modality (e.g. video-only, chance in
   audio and text), replicating across architectures WITHIN that modality. Distinguished from
   architecture-specific by holding across arch within the modality.
3. **objective-specific.** Decodable only for substrates trained with one objective family (e.g. only
   caption-trained CLIP, not predict-trained V-JEPA), across modalities/arch that share the objective.
   The verdict that flags a factor as (e.g.) linguistic.
4. **architecture-specific.** Decodable only for one architecture family, tying its random-init in others
   with the SAME objective and modality. Isolated by holding objective and modality fixed and varying arch.
5. **dataset-specific.** Decodable only for substrates pretrained on one dataset family; requires two
   substrates matched on arch+objective+modality but differing on pretraining data (a harder column to
   populate, flagged when unavailable).
6. **probe-specific.** Appears only under one probe class (linear vs MLP vs nonlinear-gain) and vanishes
   under another; the census already found several of these. Detected by running >=2 probe classes per cell.
7. **random-control-artifact.** The delta over the CORRECT random-init control is within seed spread, i.e.
   pretraining bought nothing here. This is the honest home of most historical "ties" once the vacuous
   square-projection control is replaced by random-init. Expected to be the MODAL verdict.
8. **alignment-artifact.** A cross-substrate "agreement" that a random map of equal rank also produces;
   the map, not the substrates, did the work. Detected by the random-map floor in every alignment claim.
9. **non-replicating.** Sign-flips or fails to reproduce across seeds; published as instability per the
   seed-stability standing control, never as a positive.

### 3.3 Decision order (apply top-down, first match wins)

1. If delta-over-random-init within seed spread -> **random-control-artifact** (stop).
2. Else if it vanishes under a second probe class -> **probe-specific** (stop).
3. Else if seed sign-flips -> **non-replicating** (stop).
4. Else if it is a cross-substrate agreement matched by a random map -> **alignment-artifact** (stop).
5. Else classify the SURVIVING signal by its scope: universal / modality / objective / architecture /
   dataset, using the hold-fixed-vary-one logic in 3.2.

This order enforces the doctrine: a signal is guilty (artifact) until it survives every control, and only
then does it earn a scope verdict.

### 3.4 What the protocol costs and where it runs

Cached-latent-first: each substrate encodes the shared clip set ONCE (the encode is the only expensive
step, and it is gated by the running-job constraint, so it is a Studio task, never run here). All
classification is linear probes on cached latents, cpu-now. The grid is small (a handful of factors x a
handful of available substrates); the discipline, not the compute, is the deliverable.

### 3.5 How this feeds THE fork

- If nuisance-invariant shape identity is **substrate-universal** (V-JEPA, DINOv2, audio-analog all beat
  random-init), the frozen-encoder branch is strongly supported: the property is real and not a V-JEPA
  quirk.
- If it is **architecture-specific to V-JEPA** but real, keep V-JEPA specifically.
- If the compositional-factoring factor (shape-from-color off-ceiling) is **random-control-artifact** for
  every frozen substrate but PASSES in the programmatic reference, that is the first real evidence the
  frozen substrates are BOUNDED, which is the ONLY thing that would justify the custom branch.

---

## 4. Emitted experiments

Two families, both previously unused: **AT** (Atlas, cross-substrate decodability grid and verdicts) and
**AL** (Alignment, cross-substrate maps and their artifact controls). All are cached-latent-first; every
encode is a Studio step (never run in this environment given the live job). Every row names its
non-vacuous random control and its matched baseline.

- **AT1 cross_substrate_nuisance_grid** (studio): the nuisance-invariant shape-identity factor decoded from
  V-JEPA, DINOv2, and a single-frame V-JEPA (time-stripped), each vs its OWN random-init control at matched
  resolution. Verdict: universal / modality-specific / architecture-specific / random-control-artifact.
- **AT2 time_axis_ablation** (cpu-now on cached V-JEPA): which factors need TIME (decodable full-clip,
  chance single-frame) vs static, with token/frame count matched so "needs time" is not "needs more tokens".
- **AT3 programmatic_ceiling_reference** (cpu-now): run the programmatic/handcrafted-descriptor substrate on
  the SAME factors as a difficulty-calibration upper bound, so a perceptual tie can be read as
  substrate-bounded (real negative) vs test-too-easy.
- **AT4 probe_class_sweep** (cpu-now): every atlas cell under >=2 probe classes (linear, MLP, nonlinear-gain)
  to surface probe-specific verdicts, the failure the census already caught.
- **AL1 shared_latent_alignment** (studio): thin linear/low-rank map between each substrate pair on shared
  clips, scored above a random-map-of-equal-rank floor (the alignment-artifact control).
- **AL2 audio_video_temporal_alignment** (studio): do audio-SSL and V-JEPA event boundaries co-locate above
  a shuffled-time floor, testing whether temporal-predictive invariance is substrate-universal.

Substrate slugs referenced (`registry/models.yaml`): `dinov2_vitl`, `vjepa2_vitl_fpc64_256` (canonical
config), `videomae_v2_vitb`; all auxiliary rows are `available: false` until weights verify on HF, so every
studio-tier row is gated on that verification and carries `replaces_canonical: false`.
