# Mixture of Perspectives: Master Index

This is the document to open first. It is the decision-first synthesis of the whole Mixture-of-Perspectives (MoT)
reframing of the Brain program. It states the refined thesis, the formal definition, the full taxonomy tables,
the experiment registry pointer, the three compute-tier execution plans, the custom-model decision tree, the
code roadmap, a concrete 72-hour checklist, the do-not-do-yet list, and the two lists of falsifiable claims
that would make MoT real or kill it. Every section links to its detailed source file for depth.

House style (BLACKHOLE.md): no em dashes or en dashes anywhere. Commas, colons, and parentheses only.

Governing honesty constraint inherited from the corpus (DOCTRINE_SYNTHESIS.md, section 3d): the standing
`frozen_random_projection` control is a full-rank invertible 1024x1024 Gaussian matrix, so a linear or MLP
probe absorbs its inverse and the delta is mathematically forced to 0.000. It is VACUOUS for any probe-based
metric. The valid substrate control is real V-JEPA versus random-ENCODER or random-init same-arch ViT features
at matched resolution. Every claim below is graded against that correction.

Section source files (this index links to them, it does not restate them):
[01 thesis+definition](01_thesis_and_definition.md), [03 thinking modes](03_thinking_modes.md),
[04 reasoning program](04_reasoning_program.md), [05 plasticity program](05_plasticity_program.md),
[06 cognitive currencies atlas](06_cognitive_currencies_atlas.md), [07 workspace layer](07_workspace_layer.md),
[08/09 custom-model pathway and architectures](08_09_custom_model_pathway_and_architectures.md),
[10 compute tiers](10_compute_tiers.md), [11 experiment registry](11_experiment_registry.md),
[12 metrics](12_metrics.md), [13 code scaffolding](13_code_scaffolding.md),
[15 custom-model skepticism](15_custom_model_skepticism.md).

---

## 1. The refined thesis

One sentence (from 01, section 1.1): intelligence worth building on a laptop is a coordinated ecology of
several cheap, qualitatively distinct thinking modes over one frozen perceptual substrate, and the win, if
any, is measured in capability DENSITY (capability per unit of compute, memory, and example), not in peak
capability.

The core wager, stated so it can be falsified (from 01, section 2.5): a learned or measured coordinator over
an ecology of representational geometries, learning speeds, memory systems, uncertainty signals, and reasoning
modes achieves strictly higher capability density than (a) the best single mode at matched compute, (b) the
uniform ensemble of the same modes, and (c) a monolithic model of equal parameters and compute, with at least
one mode's advantage surviving a NON-vacuous substrate control.

Why density and not peak (01, section 1.7): a compute-poor program structurally cannot win on peak (no
frontier compute, a frozen encoder). Density is a ratio a small system can win. MoT amortizes the expensive
frozen forward once into a cache, spends deliberative compute only on episodes the router judges worth it, and
protects retention with cheap sparse structure rather than by growing parameters. The transferable finding is
a new EFFICIENCY, not a new capability, which is exactly what a compute-rich lab has no incentive to find.

The one honest fact the reframe must carry (01, section 0): after ~119 catalogued experiments the corpus is a
rigorously honest NEGATIVE map with three survivors. Two are algorithmic/architectural: e7_sparse (sparse/gated
heads halve catastrophic forgetting versus a parameter-matched dense baseline, a trained-shell dynamics metric
that survives the vacuous control, +0.075 to +0.124 over 30 runs) and ex2_latent_planning (short-horizon MPC
beats flat reactive and action-shuffle on true dynamics, all 3 seeds). One is a substrate positive: real V-JEPA
decodes shape under heavy nuisance at 0.379 versus random-pixel features at 0.069 (chance 0.167), delta +0.31
off-ceiling (honestly ~0.21 to 0.23 after a 256px-vs-32px resolution discount), the first valid evidence the
frozen substrate is SPECIAL. MoT does not get to reintroduce refuted mechanisms (e4 neuromod, b4 homeostasis,
e3 staged plasticity) unchanged, and it does not get to call a probe artifact "coordination".

Where a frozen substrate may not be enough (01, section 1.8): a frozen geometry cannot be RESHAPED by
experience. If moldability or non-linguistic abstraction requires the representation itself to change, a frozen
substrate is by construction unable to demonstrate it, and MoT is only moving deck chairs in the shell.

---

## 2. Formal definition

MoT IS (01, section 2.1): a strict two-tier system. One inherited, inference-only frozen perceptual substrate
(V-JEPA 2 ViT-L, 1024-d, requires_grad=False, called only under no_grad) feeds a tiny trainable shell. The
shell hosts (i) a SET of thinking modes, each DISTINCT on at least one of five axes, and (ii) a cheap
coordinator (router) that selects or blends modes per episode or per task on a scalar cost signal, trained on
cached latents. The five distinguishing axes: representational geometry, learning speed, memory system,
uncertainty signal, reasoning mode.

Admission rules: a mode qualifies only if it is distinct on at least one axis AND passes a non-vacuous
substrate or matched-compute control on its own; a coordinator qualifies only if it beats both the best single
mode and the uniform ensemble at matched compute.

MoT IS NOT (01, section 2.2): not a peak-capability play, not a new representation learner (the substrate stays
frozen), not a claim that any mode "thinks" phenomenally, not vindicated by any probe result (a mode whose only
evidence is a probe that beats the vacuous control is PASS-VACUOUS), and not an excuse to revive refuted
mechanisms unchanged.

How it differs from adjacent architectures (01, section 2.3): unlike Mixture of Experts (homogeneous FFN
sub-blocks routed per token to save FLOPs), MoT experts are HETEROGENEOUS computations over a FROZEN shared
representation, routed per episode/task, for density across qualitatively different modes. Unlike ensembles (no
router, every member runs), MoT ROUTES to spend compute selectively. Unlike multimodal fusion (combines input
modalities), MoT combines computations over the SAME latent. Unlike tool use (external oracles), all modes are
internal learned computations. Unlike ordinary modular nets, modes must be distinct on a named axis, routing is
learned/measured and per-episode, and everything is scored on density under non-vacuous controls.

The falsifiable core, five hypotheses each with a null and a baseline (01, section 2.5):

| Hyp | Claim | Null (refuted if) | Baseline it must beat |
|-----|-------|-------------------|-----------------------|
| H1 | Routing beats best single mode | density delta <= 0 or inside seed spread | best single mode, compute-matched |
| H2 | Routing beats uniform ensemble | equal-weight blend matches the router | uniform ensemble, same members, same total compute |
| H3 | >=1 mode is substrate-dependent (non-vacuous) | every mode's advantage reproduced by random-init-ViT | random-init same-arch ViT at matched 256px |
| H4 | Heterogeneity is load-bearing | k copies of the best mode (homogeneous MoE) match the mixture | homogeneous k-copy MoE, same params/FLOPs |
| H5 | Density, not peak, is the win | advantage vanishes at unmatched compute (bought FLOPs) | monolith given the router's total compute in one pass |

Wholesale refutation (01, section 2.6): on a difficulty-calibrated non-ceiling test bed, if the best single
mode matches the routed ecology AND the uniform ensemble matches the router AND every mode ties the non-vacuous
substrate control, MoT is refuted and the correct conclusion is monolithic: pick the one best mode and ship it.
This is a real, reachable outcome given the corpus's negative prior.

---

## 3. Table of thinking modes

Five families (representation, reasoning, learning, memory, plasticity). Full tables with mechanism-in-code,
metric, null, tier, and interactions are in [03 thinking modes](03_thinking_modes.md). Compressed here to the
mode, the metric that would prove it helps, the null that falsifies it, and the current corpus verdict.

### Representation modes

| Mode | Proof metric | Falsifying null | Corpus verdict |
|------|--------------|-----------------|----------------|
| Pooled-vector (default) | it is the baseline | (null for others) | the reference |
| Dense-token | held-out-combo of a spatial factor beats pooled off-ceiling | orientation already decodes at 1.0 from pooled; dense buys nothing until non-additive | untested off-ceiling |
| Discrete-code (VQ) | capability-per-bit rises or reasoning improves at matched bits | i8/i9 at ceiling; codes are idiolects below the frozen-random floor | refuted/ceiling |
| Slot / object-centric | held-out-combo on unseen slot fillings beats pooled + random-encoder | synthetic bound objects ceiling at 1.0 | untested off-ceiling |
| Relational-graph | relation transfer to novel pairs beats frequency baseline + random-encoder | e6/d9 tie baselines on synthetic | untested on real relations |
| Predictive-code residual | head on PC-residual beats head on raw latent at matched compute | n3 ties residual; a linear head undoes the reparameterization | refuted |

### Reasoning modes

| Mode | Proof metric | Falsifying null | Corpus verdict |
|------|--------------|-----------------|----------------|
| Reactive (single-pass) | baseline | (null) | the reference |
| Iterative refinement | beats depth-and-FLOP-matched single pass | ex17/p9 gain=0.0, behaves like unrolled depth | refuted at matched compute |
| Fixed-point / attractor | converged + basin-stable + beats matched depth | y1/n9 no convergence, trajectories drift | refuted |
| Adaptive halting (ACT-lite) | accuracy-at-matched-mean-FLOPs beats fixed-step | halt allocates by noise not difficulty | open (MP5 the honest framing) |
| Latent planning (MPC) | beats flat reactive AND action-shuffle on true dynamics, all seeds | sysid says actions do not move state, or ties reactive | SURVIVING POSITIVE (ex2) |
| Verify-and-revise | beats single-shot at matched compute; score tracks true error | ex18/n11/y9 verifier carries no usable signal | refuted |
| Ensemble deliberation | beats single wider member at matched total compute | ties wider member; disagreement tracks aleatoric noise | refuted at matched compute |
| Compression-as-reasoning | lower-bit solution generalizes better at equal accuracy | i-series capability-per-bit flat | refuted |

### Learning modes

| Mode | Proof metric | Falsifying null | Corpus verdict |
|------|--------------|-----------------|----------------|
| SGD/Adam backprop (default) | baseline | (null) | the reference |
| Local rules (Hebbian/PC) | matches/beats matched-STEP SGD | ex5 win was an Adam artifact; matched-step SGD ties it | REFUTED |
| Meta-learned fast weights | faster adaptation at matched compute vs from-scratch | ties a warm-started baseline | refuted (transfer, not meta) |
| Test-time adaptation | beats frozen shell on shift, does not collapse on noise | collapses under noisy-TV or ties frozen | open |
| Replay-driven consolidation | BWT/frontier-AUC beats naive + matched-buffer random replay | prioritized ties random, or replay ties no-replay | refuted at synthetic scale |
| Curiosity-shaped sampling | learns faster than uniform AND ignores noise (noisy-TV) | e4 CONFIRMED NEGATIVE (30/30 amplify error on noise) | strongest negative |
| Sparse / gated learning (MoE-heads) | halves forgetting vs param-matched dense (30-run) | (the survivor) | SURVIVING POSITIVE (e7) +0.075 to +0.124 |

### Memory modes

| Mode | Proof metric | Falsifying null | Corpus verdict |
|------|--------------|-----------------|----------------|
| Episodic replay buffer | retention beats no-replay at matched buffer | replay ties no-replay (stream too short) | refuted at synthetic scale |
| Prioritized replay (PER) | beats uniform replay at matched buffer AND passes noisy-TV | prioritized ties random; surprise-priority chases noise | refuted |
| Content-addressable (KV) retrieval | conjunctive what+where beats retrieve-then-intersect | pooled latent binds nothing, ties intersect | untested on real bound video |
| Working memory (slots) | delayed-match/n-back beats memoryless head at matched params | ties a wider feedforward head (memory == capacity) | refuted (n7) |
| Event segmentation / chunking | chunked replay beats uniform windows; boundaries align with events | boundaries track noise; ties fixed windows | refuted |
| Buffer compression | retention-per-byte beats raw latents at equal memory | compression loses retention faster than it saves bytes | refuted |
| Object-permanence memory | decodes hidden object above frame-only + random-encoder | pooled latent cannot hold bound identity through occlusion | untested (needs slots + real video) |

---

## 4. Table of plasticity mechanisms

Full MVP/stronger impl, baseline, null, positive-proof scope, and E1-E4 relations in
[05 plasticity program](05_plasticity_program.md) and the mode rows in [03](03_thinking_modes.md), section 3.5.
Doctrinal question 1 (is the shell developmentally moldable) lives here. Every biological signature tied or lost
its non-biological baseline. Each row must beat a matched-LR-integral / matched-compute / matched-capacity
control, scored on trained-shell-dynamics metrics (BWT, forgetting, adaptation), not the vacuous probe control.

| Mechanism | Proof metric | Falsifying null | Corpus verdict |
|-----------|--------------|-----------------|----------------|
| Critical / sensitive window | early-placed concept beats late at matched data + compute | d6 substrate_specific_window=False; early ties late | negative |
| Signal-triggered reopening | reopened learning recovers new task without re-forgetting, beats fixed schedule | n5 Fisher-reopen ties fixed; reopening chases noise | negative |
| Consolidation rigidity (EWC/SI) | BWT beats naive AND beats L2-to-init at matched strength | n4/b4 tie the baseline; plain L2 helps too | ties baseline |
| Perineuronal-net rigidity | frozen-fraction retains better than matched-magnitude random regularizer | PNN mask ties a random freeze mask | negative |
| Structural growth | beats a fixed shell of the grown FINAL width | b8 ties matched-final-capacity fixed shell | negative |
| Rejuvenation / plasticity restoration | restores adaptation on a new task net of retention cost | no plasticity to lose at toy scale (dim 64) | scale-dependent LEAD (ex15) |
| Neuromodulatory gating (DA/ACh/NE) | gated beats ungated at matched compute AND passes noisy-TV | e4 CONFIRMED NEGATIVE (30/30 amplify error on noise) | strongest negative |
| U-shaped overgeneralization | reliable non-monotone down-then-up curve, seed-stable | d4 signature FLAT ZERO at every scale | negative |
| Sparse / gated (MoE-heads) plasticity | halves forgetting vs param-matched dense | (the survivor, restated) | SURVIVING POSITIVE (e7) |

The Plasticity Stack (05): 9 components (frozen substrate, plastic shell, fast-weight adapter, episodic
replay+KV, epistemic uncertainty gate, content-gated critical-period schedule, consolidation+sleep, compressed
LTM, self-eval head), everything defaulting OFF until it beats a matched baseline. Load-bearing dependency:
confirm e7_sparse/modular plasticity (PR3) and offline consolidation (PR6) on real cached latents.

---

## 5. Table of reasoning mechanisms

Full designs (null, matched-compute baseline, metrics, controls, diagnostic gates, failure interpretations,
tier, sci-value) in [04 reasoning program](04_reasoning_program.md). The governing prior: iteration ties depth
(ex17/n9/y1/ex18 nulls). The design job is to find the one framing where it does not. Rows most likely to reveal
the frozen substrate is INSUFFICIENT are flagged.

| Mechanism (04 id) | Proof metric | Falsifying null | Substrate-bound flag |
|-------------------|--------------|-----------------|----------------------|
| MP1 adaptive compute | beats fixed depth at matched AVERAGE FLOPs (allocation) | ties fixed depth; halt collapses to constant | weak (per-sample hardness) |
| MP2 confidence stopping | trained halt beats free update-norm rule at matched FLOPs | confidence == update-norm | no |
| MP3 beam/tree search | K scored trajectories + pruning beat greedy at matched total FLOPs | search == unrolled depth | no |
| MP4 latent debate | two seeded modules + referee beat single AND ensemble at matched FLOPs | debate == unrolled ensemble | no |
| MP5 mixture router | learned router beats single best strategy at matched mean FLOPs | reduces to bought compute (p9/ex17) | indirect (shell-lever) |
| DR1 fixed point vs drift | input-dependent attractor, V-JEPA-specific | no geometric decay, unrolled depth (n9/y1) | yes (geometry vs substrate) |
| DR2 latent CoT (no text) | intermediate-latent trace beats one-shot at matched compute; shuffle hurts | chain == bag of extra layers | weak |
| DR3 latent scratchpad | slot memory beats residual-only when WM load exceeds residual width | ties at all loads (pooled discarded the detail) | YES (pooled-width bound, most likely to bite) |
| DR4 causal intervention | do-ops match true counterfactuals better than correlational, low leakage | intervened <= correlational, high leakage | YES (entanglement bound) |
| DR5 cross-substrate consistency | gain replicates on a 2nd encoder and beats random-init-ViT | identical everywhere (shell) or gone on encoder 2 | yes (universality verdict) |
| DR6 internal simulation (ex2 ext) | rollout beats reactive + action-shuffle at matched compute | ties reactive or action-shuffle | weak |
| DR9 memory-first retrieve-then-reason | kNN-conditioned refiner beats from-scratch + random retrieval | frozen neighbor metric not task-aligned | weak |
| DR10 reasoning under corruption | flatter accuracy-vs-corruption slope than single-pass (missing channel) | equal slopes | YES (redundancy bound) |
| DR11 planning-horizon limit | crossover horizon beyond which planning stops beating reactive | no horizon where planning wins, or only H=1 | weak (bounds simulation) |
| DR12 disagreement-as-uncertainty | disagreement predicts error better than single-head confidence, passes noisy-TV | AUROC <= confidence, or chases noise (e4) | no |

Most-likely-to-show-insufficiency ranking (04, section 4.5): DR3 scratchpad (pooled-width working-memory
bound), DR10 missing-channel (redundancy bound), DR4 causal-intervention leakage (entanglement bound), DR2
CoT-shuffle, DR11 planning-horizon crossover.

---

## 6. Table of workspace architectures

Full mechanism, capabilities, limits, baseline, null, metrics, matched-capacity control, and cached/studio/
custom flags in [07 workspace layer](07_workspace_layer.md). Governing rule: structured fusion is presumed
decoration until it beats an equal-parameter, equal-input, unstructured concat-MLP (the workspace analogue of
frozen-random). WS1 (agreement vs confidence) gates all the rest and requires a genuinely different second
frozen encoder (DINOv2), not an invertible remap.

| Arch (07 id) | What it adds | Baseline to beat | Matched-capacity concern |
|--------------|--------------|------------------|--------------------------|
| A1 Concat MLP | reference floor | (it is the floor) | maximal info, minimal structure |
| A2 Learned linear fusion | weighted combine | concat floor | trivial extra params |
| A3 Attention fusion | input-dependent source down-weighting | concat floor | must beat at matched capacity |
| A4 GWT broadcast | narrow shared slot + broadcast-back | matched-capacity fusion + tuned regularization | narrowness must beat capacity effect |
| A5 Recurrent workspace | multi-step integration | matched-depth recurrence | risks unrolled depth (ex17) |
| A6 Graph workspace | explicit relational edges | concat floor | structure vs capacity |
| A7 Object-token workspace | binding BEFORE pooling (P7 lane) | dense-token probe without slots | strongly toward custom |
| A8 Memory-augmented workspace | external episodic slots | matched-param no-memory head | memory vs capacity (n7 risk) |
| A9 Uncertainty-weighted fusion | inverse-variance combine (~0 params) | equal-weight averaging | THE capacity-neutral clean win |
| A10 Disagreement-driven workspace | allocate compute on disagreement | uniform compute + disagreement-shuffle | must pass noisy-TV (e4 risk) |
| A11 Predictive-coding workspace | residual/surprise exchange | matched-compute single pass | reparameterization a head undoes |
| A12 Active-inference workspace | query source that most cuts uncertainty | passive fusion | closed-loop, deferred |
| A13 Modular router + shared slot | MoE arbitration + broadcast (extends e7) | slot-ablated sparse routing | slot must add over routing alone |
| A14 Latent-language / shared-code | discrete shared code between modules | continuous fusion | idiolect risk (p5/s5/y3) |

Central test WS1 (07, section 7.2): does cross-substrate agreement (V-JEPA + DINOv2) predict correctness better
than the best single substrate's confidence, decided by risk-coverage / correct-incorrect AUROC, with five
pre-registered nulls including an invertible-remap vacuity guard (N2) that directly mirrors the corpus's
vacuous-control discovery. A positive is the strongest available justification for a multi-substrate workspace;
a negative says stay single-substrate.

---

## 7. Table of custom-model stages

The custom pathway is an evidence-gated contingency, not a plan. Full purpose/evidence/compute/data/metric per
stage in [08/09](08_09_custom_model_pathway_and_architectures.md), section 8.2; the brake is
[15](15_custom_model_skepticism.md). You do not climb to rung N+1 until rung N's success metric is met AND the
gate for N+1 is cleared.

| Stage | What it is | Required evidence to LEAVE | Trains perception? | Compute |
|-------|-----------|----------------------------|--------------------|---------|
| 0 Cached-latent shells (WHERE WE ARE) | exhaust the frozen shell on cached latents | a gate (C1/C2/C3) FAILS on REAL content, not synthetic | never | Tier 0/1 |
| 1 Multi-substrate atlas | run the failing gate across frozen encoders (V-JEPA L/H/g, DINOv2, video-contrastive) | gate clears on a different substrate (KEEP it) or fails on ALL (universal, escalate) | never | studio encode |
| 2 Dense latent scaling | test whether the bound is the POOLING interface | dense/coarse-grid clears where full pool did not, or ties (escalate) | never | studio (memory pressure) |
| 3 Workspace shell | shell-side routing workspace (MoT at shell scale) | beats param-matched dense AND matched-compute unrolled depth on e7/ex2 metrics | never | studio shell |
| 4 Repair / adaptation shell | rejuvenation/growth restores plasticity at studio scale | restoration real, beats frozen-random + matched-compute; if it needs an adapting substrate, C3 fails | only a small adapter in the C3 arm | studio |
| 5 Custom substrate pilot | train a SMALL encoder/adapter on the implicated objective | pilot beats BEST frozen substrate off-ceiling at MATCHED compute, seed-stable | YES (small) | wider-box |
| 6 Custom MoT model | compose multiple substrates through a trained workspace | beats best single substrate + workspace, the pilot alone, and the strongest open model | mixture of frozen/trained | wider-box to small cluster |

Licensing logic (the AND, not the OR): Stage 5 is licensed only when (C1 fails on real bound-attribute video
after both a substrate swap and dense tokens also failed) OR (C3 fails at studio scale after workspace and
repair shells also failed). C2 does not license custom substrate; it redirects to shell stages. Stage 6 is
licensed only when Stage 5 cleared its bar AND the atlas showed composing substrates beats any single one.
Six candidate architectures (A custom JEPA, B object-centric JEPA, C action-conditioned world model, D MoT
substrate model, E developmental model, F compressed capability-density model) are detailed in
[09](08_09_custom_model_pathway_and_architectures.md), section 9; B and D-minimum stay closest to doctrine
(frozen backbone, only the binding shell or router trains).

---

## 8. Experiment registry pointer and per-prefix counts

The single deduplicated registry with full schemas (null, falsification, positive/negative interp, metrics,
controls, matched baseline, random control, dependency risk, custom-model decision) is
[11 experiment registry](11_experiment_registry.md). New label families do not collide with the 119 catalogued
corpus ids.

| Prefix | Meaning | Count | Central / most-decisive rows |
|--------|---------|-------|------------------------------|
| MT | Mixture of Perspectives routing and adaptive compute | 8 | MP1 router-vs-best-mode (core thesis), MP2 routing-vs-ensemble, MP3 hetero-vs-homo |
| DR | Deliberative reasoning primitives + deferred prerequisite | 15 | DR1 real bound-attribute video cache (decisive enabler), DR2 sparse-head on real latents, DR3 scratchpad |
| PR | Plasticity and learning | 8 | PR1 mode-error disjointness (cheap gate), PR2 plasticity real-vs-random-encoder (decisive), PR3 modular plasticity |
| WS | Workspace / arbitration | 5 | WS1 agreement-vs-confidence (gates all WS) |
| AT | Atlas / substrate typing (non-vacuous controls) | 5 | AT1 cross-substrate nuisance grid (decisive), AT4 programmatic ceiling (decisive) |
| AL | Alignment + uncertainty router | 3 | AL1 uncertainty router with noisy-TV guard |
| CM | Custom model gates and pilots | (see 08/09 + 15) | CM1 minimum pilot: is pretraining OBJECTIVE a live lever at matched capacity/data/256px |

Total new proposed rows across MT/DR/PR/WS/AT/AL: 44, plus the CM custom-model gates/pilot. All cached-latent
first; studio-tier rows are blocked on the deferred non-additive bound-attribute video cache (DR1) or on the
random-init-ViT encode, which must not run concurrently with the in-flight jobs (OOM risk).

---

## 9. Laptop execution plan (Tier 0, M3 Pro)

Detail: [10 compute tiers](10_compute_tiers.md), Tier 0. Cached-latent-first, CPU-bound for one thing only.
The whole catalogued corpus ran here. The hard ceiling: a 64-frame ViT-L forward (8192 tokens) HANGS the MPS
graph compiler and overflows the per-buffer limit even at batch=1, so real-encoder caching falls back to CPU at
a measured ~21 s/clip. Everything else (heads, predictors, ensembles, buffer, the whole shell) runs fine on
MPS.

Do now on the laptop (all cpu-now, all cached-latent, none loading a second encoder while the V-JEPA encode is
in flight):
- PR1 mode-error disjointness: pure diagnostic, no new module, computes whether reasoning modes make
  decorrelated per-sample errors so an oracle router could beat the single best mode. Cheapest gate for the
  whole MT line; a null here says stop before any studio spend.
- MP1 / MP2 / MP3 pilots on the existing synthetic harness: router-vs-best-mode, routing-vs-ensemble,
  heterogeneous-vs-homogeneous. Synthetic answers only, but they exercise the router and expose obvious nulls.
- MP5 adaptive halting versus fixed depth at matched AVERAGE FLOPs (the one honest iteration-beats-depth
  framing), MP6 confidence stopping, DR6 rollout planning (ex2 extension), DR9 memory-first retrieval, DR12
  disagreement-as-uncertainty, AL1 uncertainty router with the noisy-TV guard.
- AT3 time-axis ablation, AT4 programmatic ceiling reference, AT5 probe-class sweep: all cheap, all
  methodological guards on later verdicts.

Do NOT attempt on the laptop: two concurrent torch/encoder jobs (the 19.3 GB pool cannot safely hold both),
unbounded dense latent caches (~32 MB/clip, 10k dense clips ~= 313 GB), or trusting
any probe-based real-ties-frozen-random result (vacuous by construction).

The earlier blanket ViT-H/g download and MPS-forward prohibitions are superseded. All three V-JEPA scale
weights are pinned and staged within the derived 40 GB safety floor. Each forward now runs through a
supervised, receipt-writing probe; only its measured result may establish a memory or throughput wall.

Move-up trigger (to Studio): the corrected substrate control needs its resolution confound removed at real
scale, OR e7_sparse needs a significance test on real latents at 5+ seeds, OR the bound-attribute test bed needs
building, OR the encoder-scale question needs ViT-H/g. All are throughput or dataset-scale gates, none is a
train-a-bigger-model gate.

---

## 10. Mac Studio execution plan (Tier 1, M2 Max class, 96 GB, 2 TB)

Detail: [10 compute tiers](10_compute_tiers.md), Tier 1. Same MPS code, bigger unified pool, not a CUDA box.
Wall-clock is explicitly a non-constraint: optimize for corpus scale, replication, and thoroughness. Shipped
profile studio-1tb (max_cache_clips=2e6, max_run_count=100000, allowed_tiers {C, E}).

First thing to verify (a hypothesis, not an assumption): whether the ~38-core / 96 GB Studio LIFTS the 64-frame
MPS block. Smoke the smallest real forward; PASS = latents return without hanging; if so measure MPS s/clip
against the ~21 s/clip CPU floor; if it still hangs, fall back to CPU encode (fine under unlimited wall-clock).

Highest-value Studio jobs, in order:
- The permanent multi-encoder cached-latent corpus: real natural video cached through ViT-L, ViT-H, ViT-g
  (pooled is tiny, ~3 GB for 100k clips across all three). Frozen encoder means it never goes stale.
- The corrected substrate test at 256px at REAL scale (substrate_vs_random_init_vit.py logic): isolates
  PRETRAINING from architecture+resolution and settles whether the honest delta is near +0.31 or the
  discounted ~0.21 to 0.23. This is the cleanest early Studio win and directly reweights THE fork (AT1, AT2).
- e7_sparse (PR3/DR2) on REAL cached latents at 5+ seeds with a formal significance test: moves the strongest
  surviving positive from synthetic Gaussian clusters to real data, separating a head-architecture fact from a
  substrate fact.
- Build the deferred non-additive bound-attribute natural-video test bed (DR1): the prerequisite that unblocks
  every compositional/binding/permanence mode. This is a data-curation and design problem, not a compute one.
- WS1 agreement-vs-confidence with DINOv2 as the genuinely different second encoder (one extra cached pass).
- The encoder-scale falsifier: does bigger frozen perception change WHICH shell mechanisms help (same shell,
  same clips, same seeds, only the latent source differs). Science the laptop literally cannot produce.

Do NOT on the Studio: run device=cuda (Apple Silicon, no CUDA); cache dense latents by default (~313 GB for
10k clips); treat E10's bounded action contract as open-ended science (population/environment generation is still absent);
let auxiliary encoders stand in for the canonical V-JEPA in a
science result; scale a probe-based experiment to buy confidence in a vacuous tie.

Move-up trigger (to a training box): ONLY a bounding result that a frozen encoder across all three scales (and
dense 2.1 once it ships) provably cannot supply. Absent that, Tier 2 is out of scope by doctrine.

---

## 11. Wider-training-box execution plan (Tier 2, rented CUDA / cluster)

Detail: [10 compute tiers](10_compute_tiers.md), Tier 2. This tier exists to violate the frozen-substrate
doctrine on purpose under a pre-registered hypothesis, OR to run the two genuinely environment-gated legs. NOT
the default growth path.

Two sanctioned uses:
- Environment rollouts that keep the encoder frozen: the bounded E5/CM10 adapter now runs locally. Substrate-
  grounded ex2/CM10 still need rendered citable trajectories and exact controls; E10 needs population and
  environment generation. None has earned rented CUDA merely by being interactive.
- The doctrine-questioning use (out of scope until a bounding result forces it): train or fine-tune an encoder
  from scratch at the SAME resolution and frame budget to test whether a task-specific frontend beats the
  frozen general V-JEPA on the bounding target, with the frozen encoder as the control it must beat by a stated
  margin, and its features must also beat random-init same-arch at the same resolution.

Do NOT: enter without a bounding result; treat a rented GPU as license to abandon cached-latent-first for the
shell; scale env rollouts before the local adapter is connected to the exact scientific referents and controls.

Stay-or-abandon: stay only if the trained frontend beats the frozen baseline by a stated margin on the bounding
target, survives matched compute, and beats random-init same-arch at the same resolution. If it ties (the
expected outcome given the +0.31 evidence), that is a strong confirmation of the frozen-substrate doctrine and
the program returns to Tier 1.

---

## 12. Custom-model DECISION TREE

Drawn from the gates in [08/09](08_09_custom_model_pathway_and_architectures.md), section 8.1, and the brake in
[15](15_custom_model_skepticism.md). Read top to bottom; each IF/THEN either KEEPS the frozen encoder (terminal)
or ESCALATES one rung. The default and most likely terminal state is KEEP FROZEN.

```
START: are the two in-flight controls on disk?
  substrate_vs_random_init_vit.py  AND  compositional_under_nuisance.py
  |
  |-- NO  -> STOP. No custom decision is licensed. Land the controls first. (15: prematurity angle 1)
  |
  |-- YES -> continue.

GATE C1 (abstraction): on REAL non-additively bound video, D3-certified separable, probed with the
           held-out-combination gate against random-init same-arch ViT at matched resolution:
  Does the frozen substrate FACTOR the bound attributes (held-out delta over random-init ViT > +0.05)?
  |
  |-- YES (it factors)          -> C1 does NOT fail. KEEP FROZEN on the abstraction axis.
  |
  |-- collapses to chance        -> C1 FAILS. But do NOT jump to custom. First:
        Stage 1 atlas: does a DIFFERENT frozen encoder (DINOv2, video-contrastive, ViT-H/g) clear it?
          |-- YES -> KEEP that better FROZEN encoder. Terminal. No custom work.
          |-- NO (all substrates fail) -> continue.
        Stage 2 dense: does dense / coarse-grid-pooled representation clear it where full pool did not?
          |-- YES -> KEEP V-JEPA, adopt DENSE (V-JEPA 2.1 branch). Terminal for weights.
          |-- NO (dense ties pooled) -> C1 bound is in the WEIGHTS. Stage 5 (custom pilot) LICENSED via C1.

GATE C2 (adaptation): do e7_sparse forgetting and ex2 planning IMPROVE as substrate quality rises across the
           atlas on real latents?
  |
  |-- YES -> substrate-carried; keep improving frozen substrate.
  |
  |-- FLAT across substrates -> adaptation bounds are SHELL-architectural. C2 does NOT license custom substrate.
        Redirect to Stage 3 (workspace shell) then Stage 4 (repair shell). Both shell-side, in doctrine.

GATE C3 (moldability): after Stage 3 + Stage 4, do the developmental signatures (d6 window, y4 path-dependence,
           d4 U-shape) appear ONLY when the substrate itself adapts, and are provably absent with every frozen
           substrate + large shell at studio scale?
  |
  |-- NO (frozen shell reaches them, or they never appear) -> KEEP FROZEN. Moldability answerable without a
        trainable substrate, or is a non-phenomenon. Terminal.
  |
  |-- YES -> C3 FAILS. Stage 5 developmental variant (arch E) LICENSED via C3.

STAGE 5 (custom substrate pilot, licensed by C1-via-weights OR C3):
  Does the pilot beat the BEST frozen substrate off-ceiling at MATCHED compute, seed-stable, on real content?
  |
  |-- NO (ties at matched compute) -> STOP. Custom bought nothing. Return to Stage 0/1. KEEP FROZEN.
  |
  |-- YES -> continue.

STAGE 6 (custom MoT model, licensed only if Stage 5 cleared AND atlas showed composing substrates beats one):
  Does the full mixture beat best-single-substrate+workspace, the pilot alone, and the strongest open model,
  at documented compute, all controls passing?
  |
  |-- NO -> descope to the single best component. Terminal.
  |
  |-- YES -> the terminal custom architecture is justified.

STANDING KILL-SWITCHES (apply at every rung):
  (a) if a cheaper frozen-shell config matches the custom result at matched compute -> STOP.
  (b) if the substrate-is-special delta (+0.21 to +0.31) GROWS as content gets harder -> the frozen encoder is
      winning -> do not begin custom substrate work.
```

The pathway is deliberately hard to walk. As of today two of three gates are UNTESTED and the one valid
positive (+0.31) points toward KEEPING the frozen encoder.

---

## 13. Code scaffolding roadmap

Full module-by-module audit against the actual src/mop tree in [13 code scaffolding](13_code_scaffolding.md).
Headline: 16 of 24 spec modules already EXIST (extend only), 7 are PARTIAL (need a thin aggregator or
promotion), and only 2 are genuinely NEW. Building parallel WorkspaceShell/ReplayMemory/PlasticityController
classes is explicitly rejected as a duplicate that would fork the codebase.

- EXISTS (extend only, 16): SubstrateRegistry (encoder_registry.py), LatentStore, ProbeSuite, ReplayMemory
  (buffer.py), PlasticityController, NeuromodulationGate, ConsolidationEngine, CuriositySelector,
  UncertaintyEstimator (ensemble.py), ReasoningLoop (refine.py), CompressionDoctor (bottleneck.py),
  ExperimentRegistry, NullHypothesisRegistry, NegativeResultTaxonomy, MetricsLogger, ReproducibilityHarness.
- PARTIAL (promote/aggregate, 7): SubstrateAdapter (promote from scripts to substrate/adapter.py), AlignmentSuite
  (add diagnostics/alignment.py over geometry + seed_consistency), WorkspaceShell (compose existing shell
  modules), LatentScratchpad (WorkingMemory exists), FastWeightMemory (ex4-local), CriticalPeriodScheduler
  (controller knobs), MixtureArbitrator (promote e7 MoE router ONLY when a second experiment reuses it).
- NEW (justified, 2): CrossSubstrateAgreement (diagnostics/cross_substrate.py, the missing outer loop over
  substrates for standing-control 8) and the substrate-LEVEL MixtureArbitrator (gated behind
  CrossSubstrateAgreement showing complementarity).

Load-bearing consequence: the ONLY new code the corrected-substrate research actually requires is (1) promoting
SubstrateAdapter + CrossSubstrateAgreement from scripts into reusable diagnostics, and (2) DENSE real-video
bound-attribute caches (a DATA gap, not a code gap; LatentStore already supports dense shapes). Every module
whose output feeds a probe carries the vacuous-control caveat: substrate specialness is answered only by
real-encoder-vs-random-ENCODER arms, never a within-latent projection.

---

## 14. RUTHLESS PRIORITIZATION (ten ranked lists)

Decision-first. This is the section to act on. Every list is ordered most-important first.

### 14.1 Immediate laptop (cpu-now, no second encoder, cached-latent)
1. PR1 mode-error disjointness (the cheapest gate for the whole MT line; a null stops studio spend).
2. MP5 adaptive halting vs fixed depth at matched AVERAGE FLOPs (the one honest iteration-beats-depth framing).
3. MP1/MP2/MP3 router pilots on the synthetic harness (expose obvious nulls before studio).
4. DR12 disagreement-as-uncertainty + AL1 uncertainty router (both must pass the noisy-TV guard, retrying e4's failure mode as a router INPUT only).
5. AT4 programmatic ceiling reference + AT5 probe-class sweep (methodological guards on every later verdict).
6. DR6 rollout planning (ex2 extension) and DR9 memory-first retrieval.

### 14.2 Scaffold before Studio (code to promote/aggregate now, laptop-safe)
1. Promote SubstrateAdapter into substrate/adapter.py with the honest resolution metadata machine-readable.
2. Add diagnostics/alignment.py (thin aggregator over existing geometry + seed_consistency; do NOT reimplement CKA).
3. Add diagnostics/cross_substrate.py (CrossSubstrateAgreement, the missing outer loop) with a shuffled-label null.
4. Compose shell/workspace.py from existing modules behind a config flag (pure composition, no new science).
5. Add MT/PR/WS/CM/AT/AL/DR rows to registry/experiments.yaml with null + baseline + falsifier per row.

### 14.3 Studio (throughput and dataset-scale unlocks)
1. substrate_vs_random_init_vit.py at 256px at REAL scale (removes the resolution confound behind +0.31; AT1/AT2).
2. DR1 build the non-additive bound-attribute natural-video test bed (unblocks every compositional mode).
3. PR3/DR2 e7_sparse on REAL cached latents at 5+ seeds with a significance test.
4. The permanent multi-encoder cached-latent corpus (ViT-L/H/g over one validated raw corpus).
5. WS1 agreement-vs-confidence with DINOv2 as the genuinely different second encoder.
6. The encoder-scale falsifier (does bigger frozen perception change which shell mechanisms help).

### 14.4 Wider box (only under a pre-registered hypothesis)
1. Extend the proven local adapter only where the exact row requires rendered/substrate or generated-ecology evidence.
2. ex2/CM10 closed-loop MPC on true action-conditioned dynamics with the declared controls.
3. (Only if a bounding result lands) CM1 / Stage 5 pilot: train a SMALL encoder to test whether the pretraining OBJECTIVE is a live lever at matched capacity/data/256px.

### 14.5 Custom-model prerequisites (what must exist before any training is licensed)
1. Both in-flight controls on disk (substrate_vs_random_init_vit, compositional_under_nuisance).
2. A D3-certified, non-ceiling, non-additive REAL bound-attribute video cache (DR1).
3. The multi-substrate atlas run (Stage 1): a bound must be shown NOT to clear on any frozen encoder.
4. Dense-token result (Stage 2): the bound must be shown to survive dense, not be a pooling artifact.
5. A random-init same-arch ViT control at matched resolution for every substrate claim.

### 14.6 Reasoning / plasticity (the two doctrinal questions)
1. PR2 plasticity real-vs-random-encoder (decisive: does the +0.31 structure ease LEARNING, not just readout).
2. DR3 latent scratchpad (most likely to expose a pooled-width substrate bound).
3. DR10 reasoning under missing-channel corruption (redundancy bound).
4. DR4 causal intervention leakage (entanglement bound).
5. PR5 content-gated critical period (doctrinal Q1 lead, must beat matched-LR-integral where e3/d6 lost).
6. PR6 offline sleep consolidation, PR7 fast/slow weights (supply modes to MoT).

### 14.7 Best negative results (the corpus's most load-bearing nulls, defend and reuse)
1. e4 neuromod: 30/30 runs amplify error on noise, wrong direction (strongest negative; kills LR-gating).
2. The vacuous-control discovery: frozen_random_projection is invertible, so probe deltas are forced to 0.000.
3. ex17/p9 iteration=depth (gain 0.0, no fixed-point convergence): guts naive deliberative-reasoning claims.
4. d4 U-shape FLAT ZERO, d6 no substrate-specific window: no developmental moldability at toy scale.
5. Idiolects not languages (p5/s5/y3 below the frozen-random floor): emergent codes are not shared.
6. The ceiling problem: synthetic gratings/objects are trivially separable (all cells 1.0), so no compositional test can bite.

### 14.8 Sounds cool but not ready
1. A14 latent-language / shared-code workspace (idiolect risk; p5/s5/y3 already below floor).
2. A12 active-inference closed-loop workspace (needs an environment that does not exist).
3. Any dense latent caching on the laptop (~313 GB for 10k clips).
4. Stage 6 custom MoT model (licensed only after Stage 5 clears AND the atlas shows composition wins).
5. Arch E developmental model (highest mysticism risk; licensed only by a C3 failure).
6. Reviving e4/b4/e3 as modes unchanged (their negatives stand until a NEW non-vacuous control resurrects them).

### 14.9 Risks of fooling ourselves (the traps to pre-register against)
1. The vacuous full-rank projection: any probe delta versus frozen_random is meaningless; use random-ENCODER.
2. The ceiling trap: a tie on additive synthetic content is uninformative; require D3 certification first.
3. The 256px-vs-32px resolution confound: part of +0.31 may be resolution, not pretraining (AT1/AT2 settle it).
4. Unrolled depth masquerading as reasoning: match FLOPs, not just accuracy (the p9/ex17 lesson).
5. Bought compute masquerading as routing: H5 requires beating a monolith at equal TOTAL compute.
6. Renamed biology: every biological mechanism must beat a tuned NON-biological baseline (not a do-nothing arm).
7. Seed sign-flips published as instability, swept via harness.sweep.run_sweep (config-only seed is a silent no-op).

### 14.10 Results that would PROVE the direction (the wins worth chasing)
1. AT1/AT2: the +0.31 nuisance signal survives the random-init-ViT control at matched 256px (pretraining is real).
2. PR2: the special substrate eases LEARNING dynamics, not just readout (strongest reason to keep frozen).
3. MP1 + H4: a heterogeneous router beats the best single mode AND a homogeneous k-copy MoE at matched compute.
4. WS1: cross-substrate agreement beats single-substrate confidence past the invertible-remap guard.
5. DR3/DR10/DR4: a real substrate BOUND appears off-ceiling (would reweight the fork toward dense or custom).
6. PR3/DR2: e7_sparse's forgetting advantage replicates on real latents with a significance test.

---

## 15. DO NOT DO YET

- Do NOT run a second torch/encoder job while the CPU-bound V-JEPA encode is in flight (18 GB OOMs on two encoders).
- Do NOT trust or ship any probe-based real-ties-frozen-random result: it is vacuous by construction.
- Do NOT decide THE fork (frozen dense V-JEPA 2.1 vs custom): both in-flight controls must land first, and neither branch is justified until a non-ceiling, non-additive test bed exists.
- Do NOT cache dense latents on the laptop (~32 MB/clip; 10k clips ~= 313 GB, past the disk floor and the MPS block).
- Do NOT attempt the 64-frame ViT-L forward on MPS (per-buffer overflow regardless of RAM).
- Do NOT build parallel WorkspaceShell / ReplayMemory / PlasticityController classes: the primitives exist; extend them.
- Do NOT train any perception (Stage 5+) until a gate FAILS on REAL content after cheaper frozen substitutes (atlas swap, dense tokens) are exhausted.
- Do NOT revive e4 neuromod, b4 homeostasis, or e3 staged plasticity as modes unchanged: their negatives stand.
- Do NOT scale env rollouts before the local adapter's exact evidence and control gates pass.
- Do NOT scale a vacuous probe to more clips to buy confidence: the invertible-matrix problem does not go away.

---

## 16. Falsifiable claims that would MAKE Mixture of Perspectives real

Each is a concrete, pre-registered win with a named baseline. If these land (on a D3-certified, non-ceiling
test bed, seed-stable, at matched compute), MoT is a real efficiency finding.

1. H1: a per-episode router over {reactive, ex2 planner, e7 sparse} beats the best single mode on capability density at matched compute, delta outside the seed spread.
2. H4: the heterogeneous mixture beats a homogeneous k-copy MoE at equal params and FLOPs (diversity is load-bearing, not generic mixture capacity).
3. H2: the router beats a uniform equal-weight ensemble of the same modes at matched total compute (the coordinator, not averaging, carries the gain).
4. H3 / AT2: at least one mode's advantage survives real V-JEPA versus random-init same-arch ViT at matched 256px (substrate-dependent, non-vacuously).
5. AT1 / PR2: the +0.31 nuisance-invariance signal survives the random-init-ViT control (pretraining, not resolution) AND the special substrate eases learning dynamics, not just readout.
6. MP5: per-sample adaptive halting beats fixed depth at equal AVERAGE FLOPs by allocating depth to hard inputs (iteration beats depth via allocation, the one honest framing).
7. WS1: cross-substrate agreement (V-JEPA + DINOv2) predicts correctness better than the best single substrate's confidence, past the invertible-remap vacuity guard.
8. PR3 / DR2: e7_sparse's halved forgetting replicates on real cached latents with a formal significance test.

---

## 17. Falsifiable claims that would KILL Mixture of Perspectives

Each is a reachable null given the corpus's negative prior. If these hold on a difficulty-calibrated non-ceiling
bed, MoT is refuted and the correct move is monolithic (pick the one best mode and ship it).

1. H1 null: the routed system's density equals or trails its best constituent mode at matched compute.
2. H2 null: an equal-weight blend of the same modes matches the router (routing adds nothing over averaging).
3. H3 null: every mode's advantage is reproduced by random-init-ViT or random-encoder features (all modes tie the non-vacuous control).
4. H4 null: k copies of the best single mode (a homogeneous MoE) match the heterogeneous mixture (diversity is decoration).
5. H5 null: any routed advantage disappears when a monolith is given the router's total compute in one pass (the win was bought FLOPs / unrolled depth, the p9/ex17 failure).
6. PR2 null: shell adaptation and BWT on real V-JEPA sit inside the seed spread of the same shell on random-init-ViT (the +0.31 is readout-only; the learning problem is substrate-agnostic).
7. AT1 null: every substrate's decodability delta over its matched-resolution random-init control is within the seed spread (pretraining bought no invariance beyond architecture and resolution; the +0.31 was the confound).
8. WS1 null: agreement AUROC minus single-substrate-confidence AUROC <= 0 in CI, or the win fails the invertible-remap guard (one substrate's confidence suffices; the workspace is decoration).
9. DR1 / ceiling null: even real bound-attribute video ceilings at 1.0 on held-out-combination for real AND random-encoder features, so no compositional test can bite and the whole compositional question stays unanswerable.

The combined wholesale-refutation condition (01, section 2.6): H1 null AND H2 null AND H3 null all holding on a
non-ceiling bed refutes MoT outright.

---

## Audit corrections applied

Adversarial-review fixes applied as surgical edits (compute realism, cargo-cult/premature-custom, registry integrity):

1. **10_compute_tiers.md, Tier 0 "Relative cost" (5k/20k real cache un-runnable under the shipped profile).**
   Added an envelope-reconciliation paragraph: M3PRO_LOCAL_MAX hard-caps max_cache_clips=128, so clamp_clips()
   silently truncates a 5k/20k request; that cache is a Tier-1 job (studio-1tb max_cache_clips=2e6) or a
   Tier-0 job ONLY under a logged manual override, never a default-profile activity.
2. **10_compute_tiers.md, Tier 0 (raw-video disk footprint unbudgeted).** Added a raw-decode disk line: 5k
   slice ~5 to 15 GB raw, 20k ~20 to 60 GB raw; against a 53 to 63 GB free / 60 GB floor laptop these caches
   are DISK-blocked (free_disk_ok refuses) before they are time-blocked, so they belong on Tier 1's 900 GB
   budget.
3. **10_compute_tiers.md, Tier 1 "Expected bottlenecks" (48h max_wall_min vs the 83-day corpus).** Reconciled
   the studio-1tb max_wall_min=48h against the "wall-clock is free" posture: the corpus must be CHUNKED into
   resumable per-clip-range legs each under 48h (the cache is resumable per-clip), so the 83-day figure is the
   SUM of many sub-48h legs and never trips the wall-time kill switch; raising the cap is the wrong fix.
4. **08_09_custom_model_pathway.md, Gate C1, and 15_custom_model_skepticism.md, off-ramp S2 (vacuous
   random-pixel control).** Annotated both that compositional_under_nuisance.py:125-131 gates its
   "substrate-specific" verdict on random_pixel_features, which avg-pools 256px to 32px (64x less spatial
   info): a resolution-confounded vacuous control. C1 and S2 remain UNTESTED until the random-init same-arch
   ViT-L at matched 256px (substrate_vs_random_init_vit.py) reruns; the random-pixel verdict is downgraded to
   "resolution-confounded, not gate-clearing/off-ramp-clearing" and must NOT trigger or close the fork.
5. **11_experiment_registry.md, MP1 (broken intra-registry cross-reference).** Changed both "blocked on DR9"
   occurrences in the MP1 Data/task and Dependency-risk cells to "blocked on DR1" (DR1 is the real
   bound-attribute video cache prerequisite; DR9 is verify-revise and has no cache role).
