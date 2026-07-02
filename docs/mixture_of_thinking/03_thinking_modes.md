# Section 3: The Mixture-of-Thinking Taxonomy of Thinking Modes

This section catalogs the "modes" a frozen-substrate + tiny-trainable-shell system could run, grouped
into five families: representation, reasoning, learning, memory, plasticity. A "mode" is not a claim
that the brain does X. It is a concrete, buildable shell configuration plus the exact measurement that
would show it earns its keep. Every mode below maps to code that already exists in
`src/devsys/shell/` and `src/devsys/diagnostics/`, or to a small module a competent afternoon could
add. The point of a Mixture-of-Thinking (MoT) framing is not novelty for its own sake: it is that the
shell is small enough that we can afford several specialized heads/predictors/update-rules and a router,
and the honest question is whether ROUTING among modes beats the single best mode at matched compute.
That router-level question is itself an experiment (DR series), and it is the one most likely to bite,
because every individual mode below has a strong null.

## 3.0 Standing rules that gate every mode (do not relitigate per-row)

These are wired at the harness level (`src/devsys/diagnostics/`) and apply to every claim in this
section. A mode's "null" column is the mode-specific falsifier ON TOP of these:

- **Beat the right substrate control.** The old `frozen_random_projection` (square 1024x1024 invertible
  Gaussian, `substrate_ablation.py`) is VACUOUS for any linear/MLP probe: the probe absorbs the inverse,
  delta forced to 0.000. So probe-based modes must instead beat a random-ENCODER control
  (`scripts/substrate_vs_random_features.py`, random-init same-arch ViT-L). Trained-shell-dynamics modes
  (forgetting, BWT, path-integral) DO get bitten by `frozen_random_projection` and may use it.
- **Match compute** (`diagnostics/compute.py`, `matched_within`, tol 0.10). Iteration that helps must
  beat unrolled depth at equal FLOPs, not just raw accuracy. Applies hardest to every reasoning mode.
- **Beat a tuned baseline** (guards renamed-biology confounds: the e4/n5/n6 graveyard).
- **Noisy-TV guard** (`diagnostics/noisy_tv.py`) for any curiosity/uncertainty-driven mode.
- **Seed-stability** (`diagnostics/seed_consistency.py`); sign-flips are published as instability. Sweep
  seeds via `devsys.harness.sweep.run_sweep` (cfg.seed override), NOT `experiment.seeds` (silent no-op).
- **Difficulty calibration** (`diagnostics/difficulty_calibration.py`, D3): certify the regime carries
  real structure via a known-separable reference before any tie is trusted. THE ceiling problem: every
  synthetic probe currently ceilings at 1.0, so most compositional modes cannot be scored yet and are
  gated behind "real bound-attribute video" (prerequisite DR-VIDEO-CACHE below).
- **Cross-substrate convergence** (newest standing control): is a mode universal or specific to
  V-JEPA's modality/objective/architecture.

Tier vocabulary used in the tables: **cpu-now** (runs on cached latents on the M3, seconds-to-minutes),
**studio** (needs the studio box for real video caching / 30-run sweeps), **wider-box** (needs more RAM
or a real GPU for a random-init ViT-L or large replay), **moonshot** (needs an interactive environment
or weights we do not have).

---

## 3.1 Representation modes

What the shell treats as the unit of thought: a pooled vector, dense tokens, a discrete code, a slot
set, a relational graph. The substrate emits a 1024-d pooled vector (and, if we choose, dense tokens);
representation modes are shell-side re-encodings of that.

| Mode | What it does | Mechanism on frozen substrate + shell | Metric that proves it helps | Null that falsifies it | Tier | Interacts with |
|---|---|---|---|---|---|---|
| **Pooled-vector** (default) | One 1024-d vector per clip; the current default everywhere | Identity on the substrate's mean-pooled output; heads read it directly (`shell/heads.py`) | It is the baseline; other modes must beat it | (it is the null for the others) | cpu-now | baseline for every mode |
| **Dense-token** | Keep the ~8192 patch tokens instead of pooling; preserve within-frame spatial/binding structure | Do not pool; feed token set to a slot/attention head or a `WorkingMemory` read (`shell/modulation.py`) | Held-out-combination decode of a spatial factor beats pooled at matched probe capacity, off-ceiling | `dense_vs_pooled.json`: orientation already decodes at 1.0 from the POOLED vector, so pooling does NOT destroy single spatial factors; dense buys nothing until the test is non-additive | studio | slot-attention, object-binding, compositional reasoning |
| **Discrete-code (VQ)** | Quantize the latent to a codebook; a symbol-like token stream | VQ head on cached latents (I9 line, `i9_vq_rate_distortion`); measure capability-per-bit (`diagnostics/bottleneck.py`) | Capability-per-bit rises OR downstream reasoning improves vs continuous at matched bits | i8/i9 quantization is at CEILING (4-bit ties full precision); codes are idiolects (p5/s5/y3 below frozen-random floor), not shared languages | cpu-now | code-stability, compression-as-reasoning |
| **Slot / object-centric** | Factor the scene into K object slots before any head reads it | Slot-attention over dense tokens (`ex9_slot_attention`); K learned queries compete for tokens | Held-out-combination (`diagnostics/held_out_combo.py`) decodes an object attribute on unseen slot fillings, beats pooled and random-encoder, off-ceiling | On synthetic bound objects the whole probe ceilings at 1.0 (`compositional_binding=1.0`); slots cannot be shown to help until non-additive video exists | studio | dense-token, compositional/relational reasoning, object-permanence memory |
| **Relational-graph** | Represent pairwise relations (above/left-of/causes) rather than absolute features | Small relation head over pairs of slot/token embeddings (E6 relational line) | Relation transfer to novel object pairs beats a within-pair-frequency baseline and random-encoder | e6/d9 relational gates tie their baselines on synthetic data; needs real relations | studio | slot, held-out-combination, causal reasoning |
| **Predictive-code residual** | Represent a clip as its deviation from what the predictor expected (a surprise-coded latent) | `IterativeRefiner(mode="predictive_coding")` in `shell/refine.py`: `u = rate*(block(LN z) - z)` | Downstream head on the PC-residual beats head on raw latent at matched compute | n3 PC-refiner ties residual mode; PC residual is a re-parameterization a linear head undoes | cpu-now | iterative-refinement reasoning, surprise-gated memory write |

**Notes.** The representation family is where the ceiling problem bites hardest. The single most
important correction in the corpus (section 3d of the doctrine) is that a full-rank random projection is
invertible, so any probe-based representation comparison is vacuous. The ONLY representation result that
has survived is the nuisance-invariance one (`substrate_vs_random.json`, +0.31 shape-under-nuisance real
vs random-pixel), and even that is confounded by 256px-vs-32px resolution pending the random-init-ViT
control. Consequence: dense/slot/relational modes are all real-code-buildable TODAY but cannot be scored
honestly until DR-VIDEO-CACHE lands. That is exactly why the highest-value representation experiment is
the caching prerequisite, not any single mode.

---

## 3.2 Reasoning modes

How the shell turns a representation into an answer: one forward pass, iterated refinement, planning
rollouts, verify-and-revise, ensemble deliberation. Every reasoning mode's central adversary is
`diagnostics/compute.py`: reasoning ADDS compute, so it must beat unrolled depth at equal FLOPs.

| Mode | What it does | Mechanism on frozen substrate + shell | Metric that proves it helps | Null that falsifies it | Tier | Interacts with |
|---|---|---|---|---|---|---|
| **Reactive (single-pass)** | One MLP forward; the default | `Predictor` / `ClassHead` one shot (`shell/predictor.py`, `heads.py`) | Baseline the others must beat | (it is the null) | cpu-now | baseline for all |
| **Iterative refinement** | Refine the latent over N residual steps before the head reads it | `IterativeRefiner` weight-tied recurrence (`shell/refine.py`); control is an untied MLP of equal block count | Accuracy beats a depth-and-FLOP-matched single-pass MLP (`matched_within`) | ex17/p9 gain=0.0; iteration behaves like plain unrolled depth, no separation from matched depth | cpu-now | halting, verification, PC-residual representation |
| **Fixed-point / attractor** | Iterate to a stable latent that error-corrects perturbations | Unroll `IterativeRefiner`; classify dynamics with `convergence_report`, `basin_stability` (`diagnostics/convergence.py`) | `converged=True` (contraction_factor<0) AND basin `contraction_ratio<1`, AND accuracy beats matched depth | y1/n9: no fixed-point convergence, trajectories drift; refiner is not an attractor | cpu-now | iterative refinement, halting |
| **Adaptive halting (ACT-lite)** | Spend more steps on hard samples, fewer on easy ones | `IterativeRefiner(halt=True)`: per-step halt head, stop at cumulative-prob threshold | Accuracy-at-matched-mean-FLOPs beats fixed-step; hard/easy step split correlates with true difficulty | halt head allocates uniformly / by noise not difficulty; ties fixed-step at matched mean compute | cpu-now | difficulty-calibration, verification, curiosity |
| **Latent planning (MPC)** | Roll the action-conditioned predictor forward, pick the action that reaches the goal latent | AC `Predictor` + short-horizon rollout (`ex2_latent_planning`); licensed only after `sysid_report` says actions move the state | Beats a flat reactive head AND an action-shuffle control on TRUE dynamics, all seeds | `sysid.planning_licensed=False` (Gramian rank-deficient or action_delta<0.05); or ties reactive | studio | system-id gate, world-model memory, curiosity |
| **Verify-and-revise** | Score the answer, trigger an extra refine step only on low-confidence samples | `Verifier` head (`shell/refine.py`) scores z; re-refine below threshold | Verify-revise beats single-shot at MATCHED compute; verifier score correlates with true error | ex18/n11/y9: verifier carries no usable signal above matched-compute baseline (taxonomy 4) | cpu-now | halting, iterative refinement, ensemble |
| **Ensemble deliberation** | Several small predictors vote; disagreement flags "think harder" | `Ensemble` (`shell/ensemble.py`), `mean_and_disagreement` | Ensemble beats single member at matched TOTAL params/FLOPs; disagreement predicts error | ensemble ties a single wider member at matched compute; disagreement tracks aleatoric noise (fails noisy-TV) | cpu-now | verification, curiosity, uncertainty-gated learning |
| **Compression-as-reasoning** | Solve by finding the shortest code (MDL) that predicts the target | Rate-distortion / bottleneck head (`diagnostics/bottleneck.py`, I1/I4-line) | Lower-bit solution generalizes better than a higher-bit one at equal accuracy | i-series: capability-per-bit flat; compression does not buy generalization on these latents | cpu-now | discrete-code representation, MDL learning |

**Notes.** This family is a graveyard of matched-compute ties, which is the honest finding: on pooled
V-JEPA latents, "reasoning" that is really "more forward FLOPs" does not separate from depth. The ONE
survivor is `ex2_latent_planning`, and it survives because it beats a NON-projection control
(action-shuffle) on true dynamics, not a probe. The lesson for MoT: a reasoning mode earns its slot only
when its control is a matched-compute or shuffle control, never a random-projection probe. The
highest-value reasoning experiment is therefore not another refinement variant (p9 closed that) but a
ROUTER that learns WHEN reactive vs plan vs verify pays off, scored against always-using-the-single-best
mode at matched mean compute (DR-ROUTER below).

---

## 3.3 Learning modes

How the shell's weights change: gradient descent, local rules, meta-learned fast weights, replay-driven
consolidation, curiosity-shaped sampling. The substrate never learns; every learning mode is a shell
update rule on cached latents.

| Mode | What it does | Mechanism on frozen substrate + shell | Metric that proves it helps | Null that falsifies it | Tier | Interacts with |
|---|---|---|---|---|---|---|
| **SGD/Adam backprop** (default) | Standard gradient step on the shell | Optimizer over predictor/heads | Baseline | (it is the null) | cpu-now | everything |
| **Local rules (Hebbian/pred-coding)** | Update weights with a local signal, no global backprop | `i4_backprop_alternatives` / e9 local-rule head | Matches or beats matched-STEP SGD backprop on retention/accuracy | ex5 REFUTED: local-rule "win" was an Adam artifact; matched-step SGD ties it | cpu-now | plasticity gating, PC-residual |
| **Meta-learned fast weights** | An inner-loop fast adaptation the outer loop shapes | `ex4_fast_weights` / `ex7_meta_learning`: fast weights added to slow head | Faster adaptation to a new task at matched compute vs from-scratch fine-tune | fast-weights tie a warm-started baseline; the "meta" gain is transfer any warm start gives | studio | test-time adaptation, working memory |
| **Test-time adaptation** | Adapt the shell on the test stream with a self-supervised loss | `ex3_test_time_adaptation`: entropy/consistency loss at eval | Beats frozen shell on distribution shift, does not collapse on noise | TTA collapses under noisy-TV (adapts to noise); or ties frozen | cpu-now | curiosity, noisy-TV guard, plasticity |
| **Replay-driven consolidation** | Interleave stored latents to fight forgetting | `ReplayBuffer` + `Consolidation` (EWC/SI) (`shell/buffer.py`, `consolidation.py`) | Backward-transfer / frontier-AUC beats naive sequential AND matched-buffer random replay | e1/e2: prioritized ties random, or replay ties no-replay (stream too short / latents not distinct) | cpu-now | prioritized memory, plasticity floor |
| **Curiosity-shaped sampling** | Sample/learn where prediction error or disagreement is high | `Neuromodulation` DA/ACh/NE gains (`shell/neuromod.py`) drive priority | Learns faster than uniform sampling AND ignores irreducible noise (noisy-TV) | e4 CONFIRMED NEGATIVE: 30/30 runs amplify error on noise, wrong direction; e5/curiosity ties uniform | cpu-now | replay priority, ensemble disagreement, plasticity reopening |
| **Sparse / gated learning (MoE-heads)** | Route each sample to a sparse subset of head units, reducing interference | `ContextGating` / kWTA sparse heads (`shell/modulation.py`; `e7_sparse`) | Halves catastrophic forgetting vs parameter-matched dense (30-run sweep) | (SURVIVING POSITIVE: +0.075 to +0.124, survives frozen-random as a trained-shell metric) | cpu-now | consolidation, plasticity, context memory |

**Notes.** `e7_sparse` is one of the two surviving positives in the entire corpus and it lives in this
family. Critically it survives `frozen_random_projection` NOT because the substrate is special but
because forgetting is a trained-shell DYNAMICS metric (input conditioning matters, and the random
projection changes conditioning), so it is not vacuous the way a probe would be. It is reframed as a
head-ARCHITECTURE fact, not V-JEPA geometry. Open lead: is the advantage substrate-specific or purely
head-architectural once run on REAL cached latents (only synthetic Gaussian-cluster so far, no formal
significance test)? That is a clean, high-value, cpu-now-to-studio experiment (DR-SPARSE-REAL below).
The rest of the family mostly ties; the curiosity sub-family is the strongest confirmed negative in the
whole program (e4).

---

## 3.4 Memory modes

How the shell stores and retrieves past experience. Because the encoder is frozen, cached latents never
go stale, which is the one place the frozen constraint actively HELPS a memory system.

| Mode | What it does | Mechanism on frozen substrate + shell | Metric that proves it helps | Null that falsifies it | Tier | Interacts with |
|---|---|---|---|---|---|---|
| **Episodic replay buffer** | Store recent latents, sample them back | `ReplayBuffer` reservoir/fifo (`shell/buffer.py`) | Retention beats no-replay at matched buffer size | replay ties no-replay (stream too short, e2) | cpu-now | consolidation, prioritization |
| **Prioritized replay (PER)** | Sample by surprise/reward/learning-progress, not uniform | `ReplayBuffer(prioritized=True)`, PER `p^alpha`, IS weights | Beats uniform replay at matched buffer AND passes noisy-TV (does not chase noise) | e2: prioritized ties random; e4 shows surprise-priority chases aleatoric noise | cpu-now | curiosity, neuromod, buffer eviction |
| **Content-addressable (KV) retrieval** | Retrieve by nearest-neighbor in latent space, not by index | `KVIndex` (faiss or exact cdist) in `shell/buffer.py`; conjunctive key retrieval | Conjunctive what+where retrieval beats retrieve-then-intersect and recency-only (a3 what-where-when) | pooled latent binds nothing, so conjunctive KV ties independent-feature intersect | cpu-now | slot/binding representation, object-permanence |
| **Working memory (slots)** | A small recurrent scratchpad across a sequence | `WorkingMemory` gated read/write slots (`shell/modulation.py`) | Delayed-match / n-back accuracy beats a memoryless head at matched params | WM ties a wider feedforward head (the "memory" is just capacity, n7) | cpu-now | reasoning (planning), chunking |
| **Event segmentation / chunking** | Cut the stream at high-surprise boundaries into episodes | `Chunking` boundary detector (`shell/modulation.py`) | Chunked replay improves retention vs uniform windows; boundaries align with true events | boundaries track noise not structure; chunking ties fixed windows | cpu-now | prioritized replay, working memory, curiosity |
| **Buffer compression** | Store more history per byte (quantized / prototype latents) | `diagnostics/buffer_compression.py` `retention_per_byte`; VQ prototypes | Retention-per-byte beats storing raw latents at equal memory | compression loses retention faster than it saves bytes (a3 buffer-compression) | cpu-now | discrete-code, rate-distortion replay |
| **Object-permanence memory** | Hold a latent for an occluded object and re-bind it on reappearance | KV retrieval keyed on pre-occlusion latent (`a6`/`n8` object-permanence) | Decodes the hidden object above a frame-only baseline and above random-encoder | pooled latent cannot hold bound identity through occlusion (needs slots + real video) | studio | slot representation, KV retrieval, world-model |

**Notes.** The memory family is the best-engineered part of the shell (the buffer carries three cleanly
separated knobs: prioritization, KV index, eviction) but its scientific wins are thin because the
POOLED latent binds nothing: any conjunctive-retrieval or object-permanence claim ties an
independent-feature baseline on synthetic data. Like the representation family, the interesting memory
modes (conjunctive KV, object-permanence) are blocked on DR-VIDEO-CACHE. The one memory-adjacent
positive is really the sparse-heads forgetting result (3.3). A genuinely open, cheap lead: does
event-segmentation chunking of the replay stream beat uniform windows once the stream is long enough
(`ex13_long_stream` scale)? That is studio-tier and unbuilt.

---

## 3.5 Plasticity modes

How the shell's CAPACITY TO LEARN itself changes over time: critical/sensitive windows, signal-triggered
reopening, consolidation rigidity, structural growth, rejuvenation. This is doctrinal question (1):
can the shell be developmentally moldable. The honest state is that almost every biological-plasticity
signature TIES its non-biological baseline.

| Mode | What it does | Mechanism on frozen substrate + shell | Metric that proves it helps | Null that falsifies it | Tier | Interacts with |
|---|---|---|---|---|---|---|
| **Critical / sensitive window** | Learning rate high early, decays to a floor; early data matters more | `PlasticityController` hard/soft schedules (`shell/plasticity.py`) | Early-placed concept beats late-placed at matched total data AND matched compute (d6) | d6: `substrate_specific_window=False`; early-vs-late ties once data and compute are matched | cpu-now | consolidation, scheduling of learning modes |
| **Signal-triggered reopening** | Surprise/novelty above threshold REOPENS plasticity | `PlasticityController.lr_scale(signal>reopen_threshold)`; driven by `Neuromodulation` | Reopened learning recovers a new task WITHOUT re-forgetting, beats a fixed-schedule control | n5 Fisher-reopen ties fixed schedule; reopening chases noise (shares e4's failure mode) | cpu-now | neuromod, curiosity, consolidation |
| **Consolidation rigidity (EWC/SI)** | Grow per-weight rigidity so important weights resist change | `EWC` (diag Fisher) / `SI` (path integral) (`shell/consolidation.py`) | Backward transfer beats naive AND beats L2-to-init at matched strength | n4 tag-and-capture, b4 homeostatic scaling tie the non-biological baseline; SI/EWC help but so does plain L2 | cpu-now | replay, learning-rate schedule |
| **Perineuronal-net rigidity** | Freeze a fixed fraction of weights / grow rigidity where a weight is stable | `PlasticityController.init_pnn`, `update_rigidity`, `rigidity_penalty` (`shell/plasticity.py`) | Frozen-fraction retains better than a matched-magnitude random regularizer | PNN mask ties a random freeze mask at matched frozen fraction | cpu-now | consolidation, structural growth |
| **Structural growth** | Add capacity (width/units) when the current shell saturates | `b8_structural_growth`: grow width on a saturation trigger | Beats a fixed shell whose width EQUALS the grown FINAL width (matched-final-capacity) | b8: grown shell ties the matched-final-capacity fixed shell; growth just delays reaching the same size | studio | sparse heads, consolidation |
| **Rejuvenation / plasticity restoration** | Detect and reverse loss of plasticity (dormant units) | `ex15_rejuvenation`: reset/perturb dormant units | Restores adaptation speed on a new task after long training, net of any retention cost | at toy scale (dim 64) there is no plasticity to lose; effect only appears past dim 256 / thousands of tasks | wider-box | structural growth, consolidation, meta-learning |
| **Neuromodulatory gating (DA/ACh/NE)** | Global scalar signals gate LR / memory write / reset | `Neuromodulation.gates` (`shell/neuromod.py`) | Gated learning beats ungated at matched compute AND passes noisy-TV | e4 CONFIRMED NEGATIVE (strongest in corpus): 30/30 runs amplify error on noise, wrong direction | cpu-now | curiosity, plasticity reopening, prioritized replay |
| **U-shaped overgeneralization** | A developmental dip: performance gets worse before better as a rule over-applies | Track a rule-application metric over training (`d4_ushaped_overgen`) | A reliable non-monotone (down-then-up) curve appears and is seed-stable | d4: signature is FLAT ZERO; no U-shape at any tested scale | cpu-now | curriculum, consolidation |

**Notes.** This family is the clearest negative map in the corpus: e4 (neuromod gating) is the single
strongest confirmed negative, and n5/n6/n4/b4/e3/d6/d4 all tie or lose. Moldability is UNDEMONSTRATED at
toy scale. The one live thread is `ex15_rejuvenation`: plasticity loss (and its reversal) is
SCALE-DEPENDENT, appearing only past dim 256 and thousands of tasks, which is exactly the regime the M3
cannot reach. So the highest-value plasticity experiment is a scale sweep of rejuvenation on real cached
latents on the studio/wider-box (DR-PLASTICITY-SCALE below), because every toy-scale plasticity result
is a null BY CONSTRUCTION if the phenomenon only exists at scale.

---

## 3.6 Cross-mode interactions and the router (the actual MoT thesis)

The reason to catalog modes rather than pick one is that a small shell can hold several and ROUTE.
The MoT thesis is: routing among modes beats the single best mode at matched mean compute. This is
distinct from every single-mode experiment above and is the highest-value proposal in this section,
because (a) it does not require any mode to individually win, only that their ERRORS be different and a
cheap router can exploit that, and (b) it is directly buildable on the existing `Ensemble` +
`Verifier` + `IterativeRefiner` primitives.

Key documented interactions (each is a potential router feature or a confound to control):

- **Iteration x compute**: any reasoning mode that iterates is a compute confound; the router must be
  scored at matched MEAN FLOPs (`diagnostics/compute.py`), else it "wins" by spending more.
- **Curiosity x memory x plasticity**: surprise drives prioritized replay AND plasticity reopening AND
  neuromod gating from the SAME signal. e4 shows that signal chases aleatoric noise, so any router using
  a surprise feature inherits e4's failure unless it passes noisy-TV.
- **Representation x reasoning**: slot/dense representations are the prerequisite for compositional and
  object-permanence reasoning; a router over reasoning modes is uninformative on pooled latents because
  the representation ceilings first.
- **Ensemble x verify x halt**: these three are near-substitutes (all "spend more when uncertain"). A
  router must not double-count; the honest control is the single best of the three at matched compute.
- **Sparse-heads x consolidation**: the one surviving positive (e7) is a representation-of-tasks routing
  already; a mode-router is its natural generalization and should be built on the same kWTA gate.

---

## 3.7 Highest-value experiments proposed by this section (PR / DR)

The experiments list returned in the structured output prioritizes:

1. **DR-ROUTER** (studio): the core MoT test, a learned router over {reactive, refine, plan, verify}
   scored against the single best fixed mode at matched mean FLOPs. Highest value because it can win
   even when no single mode does, and it is the only proposal that tests the MoT thesis directly.
2. **DR-VIDEO-CACHE** (studio): the deferred prerequisite for ALL representation, slot, compositional,
   and object-permanence memory modes: real natural video with non-additively bound attributes cached
   through the frozen encoder, so a compositional test can finally be non-ceiling and non-additive.
   Nothing in 3.1/3.4 can be scored until this exists.
3. **PR-MODE-ERROR-DISJOINTNESS** (cpu-now): the cheap precondition for DR-ROUTER. Measure whether the
   modes' per-sample errors are actually DIFFERENT (low error correlation). If errors are correlated, a
   router cannot help and DR-ROUTER is not worth studio time. A pure diagnostic on cached latents.
4. **DR-SPARSE-REAL** (studio): move the one surviving learning positive (e7 sparse heads halve
   forgetting) from synthetic Gaussian clusters onto real cached latents with a formal significance
   test, and check whether the advantage is substrate-specific or purely head-architectural.
5. **DR-PLASTICITY-SCALE** (wider-box): the scale sweep of `ex15_rejuvenation` on real latents past dim
   256 / thousands of tasks, the only regime where a moldability signal has ever appeared. Every
   toy-scale plasticity result is a null by construction if the effect only exists at scale.

All five carry the standing controls of 3.0. DR-ROUTER and PR-MODE-ERROR-DISJOINTNESS are the pair most
likely to bite because they do not depend on any single mode winning.
