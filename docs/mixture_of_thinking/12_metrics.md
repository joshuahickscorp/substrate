# Section 12: Reasoning, Plasticity, and Workspace Metrics Overhaul

## 0. Scope, stance, and the one rule that governs this section

This section defines every metric the mixture-of-thinking spec needs, in three
groups: reasoning metrics, learning/plasticity metrics, and workspace/mixture
metrics. It is a metrics contract, not a mechanism proposal, so it returns no new
experiments (the experiments array is empty). Where a metric implies a specific
validation it says so and points at the standing control that already covers it.

The stance is set by the corpus history. Almost every candidate positive in this
program died to one of six controls, and one control (frozen-random projection)
turned out to be silently vacuous for the entire class of probe metrics, which
invalidated a whole pillar of "the substrate is not special" without anyone
noticing for ~119 experiments. The lesson is not "add more metrics." It is: a
metric is only worth defining if you can state, in advance, exactly how it can be
gamed and which cheaper control kills the game. So every metric below carries a
"how it can be gamed" and a "false-positive prevention" field, and each one is
wired to an EXISTING diagnostic in `src/devsys/diagnostics/` rather than
reimplemented. Duplication is how the vacuous control survived so long (each
experiment rolled its own frozen-random arm and none checked whether the arm was
invertible); a single shared implementation with a single audited failure mode is
the correction.

The single most important standing fact, restated because it silently reshapes
half the metrics here:

> `frozen_random_projection` (`substrate_ablation.py:28`) multiplies a pooled
> latent by a square 1024x1024 Gaussian `w = randn(d,d)/sqrt(d)`, which is
> full-rank and therefore INVERTIBLE. Any linear-or-MLP probe absorbs the inverse,
> so probe accuracy is mathematically EXACTLY invariant to it (measured delta
> 0.000). A probe-based "real ties frozen-random" result is a theorem about
> full-rank matrices, not a fact about V-JEPA. `needs_real` on a probe metric is
> near-meaningless.

Consequences that thread through everything below:
- The genuine substrate control is real-V-JEPA vs random-ENCODER features
  (`scripts/substrate_vs_random_features.py`, landed at +0.31; gold-standard
  `substrate_vs_random_init_vit.py` in flight), NOT frozen-random projection.
- Controls that SURVIVE the vacuous finding are the ones whose dynamics depend on
  input CONDITIONING rather than linear separability: SGD forgetting / BWT, and
  the non-projection controls (shuffle, matched-compute, tuned-baseline,
  action-shuffle, seed-stability). Every metric here is graded on whether its
  gating control is a survivor or a casualty of the vacuous finding, and that
  grade is stated explicitly.

Notational conventions: "cached latent" means a pooled 1024-d V-JEPA vector (or a
dense token grid where noted) already on disk under `runs/`; no metric here loads
the encoder or torch models that would OOM the running job. All compute is
CPU-fine on cached tensors. All metric verdicts are booleans with an explicit
margin, following the existing house style (`needs_real`, `decodable`,
`compositional`, `stable`, `planning_licensed`), because a bare float invites the
reader to move the threshold after seeing the result.

---

## Group A: Reasoning metrics

Reasoning in this system is not chain-of-text; it is latent iterative computation
(the `IterativeRefiner` in `shell/refine.py`), short-horizon latent planning (ex2),
compositional generalization to unseen factor combinations, and self-verification
(ex18). The corpus verdict so far is bleak: p9/ex17 gain = 0.0, no fixed-point
convergence, latent iteration behaves like plain unrolled depth. The metrics below
are built to make that negative sharp and to catch the one way a reasoning claim
can still be true: a gain that survives matched-compute AND is not a probe artifact.

### A1. Contraction factor / fixed-point residual (latent iteration is computation, not depth)

- **What it measures.** Unroll the refiner past its trained step count and fit the
  slope of `log||z_{t+1} - z_t||` vs step. Negative slope = geometric contraction
  to a fixed point (an error-correcting attractor, i.e. genuine iterative
  computation); near-zero = drift/plateau (unrolled depth wearing a costume).
- **Why it matters.** "Latent reasoning" is only a distinct capability if iterating
  the same operator BUYS something a single deeper pass does not. Contraction to a
  content-dependent fixed point is the falsifiable version of that claim.
- **How to compute.** `convergence.convergence_report(refiner, z, steps)`. It
  returns `contraction_factor`, `fixed_point_residual`, and a `classification` in
  {converges, drifts, limit_cycle}. Runs on any cached latent `z`; the refiner is a
  tiny shell module, not the encoder.
- **Baseline it needs.** Two, and they are the whole game: (1) a single-pass MLP of
  EQUAL total FLOPs (`compute.depth_for_matched_flops` gives the block count;
  `compute.matched_within` certifies the match), so the refiner must beat unrolled
  depth at equal compute, not just beat a shallow head; (2) the same unroll on a
  frozen-random-projected `z` (this control is NOT a probe, it is a dynamical
  system, so it is a SURVIVOR of the vacuous finding: a linear map does not commute
  with a nonlinear refiner's iteration).
- **How it can be gamed.** (i) Report `converged` off `fixed_point_residual` alone;
  a refiner that shrinks every input toward the origin (a contraction map with a
  trivial single fixed point) "converges" for all inputs and computes nothing. (ii)
  Cherry-pick `steps` so the plateau looks like convergence. (iii) Use a refiner
  whose Lipschitz constant < 1 by construction (e.g. a residual block with a tiny
  gain), which guarantees contraction independent of content.
- **False-positive prevention.** Require BOTH `contraction_factor < 0` AND that the
  fixed point is CONTENT-DEPENDENT: different input classes must reach
  DISTINGUISHABLE fixed points (probe the fixed points, verify they separate class
  above chance, otherwise the attractor is trivial). Pair with A2 (basin stability
  must be < 1 for a genuine basin, but a global collapse also gives ratio 0, so A1
  content-dependence disambiguates). Sweep `steps` and publish the full
  `update_norms` curve, never a single step count. And the load-bearing one: the
  matched-FLOP single-pass baseline must LOSE. p9/ex17 gain = 0.0 is the standing
  null; a positive here must beat it with seed-stability (A-group publishes
  sign-flips as instability per the seed harness).

### A2. Basin contraction ratio (error correction vs perturbation amplification)

- **What it measures.** Perturb the input by Gaussian noise of scale eps, unroll
  both clean and perturbed, and take `||z*_clean - z*_perturbed|| / ||noise||`.
  Ratio < 1 = the refiner pulls perturbed inputs back together (a stable
  error-correcting basin); ratio >= 1 = it amplifies perturbations (the e4 failure
  mode, which amplified error on noise in 30/30 runs).
- **Why it matters.** An attractor that corrects errors is the mechanistic payoff of
  iteration. This is the metric that told e4 it was going the wrong direction; it is
  reused, not reinvented.
- **How to compute.** `convergence.basin_stability(refiner, z, eps, steps)` returns
  `contraction_ratio` and `stable`.
- **Baseline it needs.** The same ratio on a frozen-random-projected input (survivor
  control, non-probe) and on the matched-FLOP single-pass network (a single pass
  has ratio ~1 by construction if it is near-linear, so beating it is the point).
- **How it can be gamed.** A refiner that collapses ALL inputs to one point has
  ratio 0 (perfectly "stable") and is useless. Small eps makes ratio look better if
  the map saturates locally.
- **False-positive prevention.** Sweep eps across at least a decade and require the
  ratio to stay < 1 across the range, not just at one scale. Cross-check against A1
  content-dependence so "stable" cannot be trivial collapse. Verdict is only
  meaningful jointly with A1: converges + content-dependent + stable + beats
  matched-FLOP depth.

### A3. Held-out-combination decodability gap (compositional generalization)

- **What it measures.** Train a probe to decode factor A on a subset of (A,B) cells,
  test on held-out cells never seen paired. Small seen-to-heldout gap = the code
  factorizes A independently of B (systematic reuse); large gap = it memorized
  conjunctions.
- **Why it matters.** This is the abstraction question that does not route through
  language: does the representation compose factors it never saw combined? It is the
  gate dense V-JEPA 2.1 (or a custom head) must clear.
- **How to compute.** `held_out_combo.held_out_combination(x, y_a, y_b)` returns
  `heldout_acc`, `gap`, `compositional`; `compositionality_report` wraps it with the
  arms.
- **Baseline it needs.** THIS IS A PROBE METRIC, so its bundled frozen-random arm is
  a CASUALTY of the vacuous finding: `compositional_binding = 1.0` and the whole
  probe ceilings, so the frozen-random comparison inside `compositionality_report`
  is invertible-projection-vacuous and proves nothing. The metric is only
  interpretable when (i) the regime is difficulty-calibrated (below) AND (ii) the
  genuine substrate control is real-V-JEPA vs random-ENCODER features, NOT
  frozen-random. The additive-vs-entangled `interaction` knob in `factorized_latents`
  is a sanity axis: additive should factor, entangled should not.
- **How it can be gamed.** (i) Ceiling: synthetic factors are trivially separable in
  1024-d, so `heldout_acc` = 1.0 for real AND frozen-random AND random-pixel; the
  gap is zero for everyone and "compositional" fires vacuously. This is the binding
  constraint of the entire program. (ii) Additive construction: if factors are
  independent by design (interaction = 0), factorization is guaranteed and says
  nothing about the substrate. (iii) Class-imbalanced held-out cells inflate accuracy.
- **False-positive prevention.** Gate on `difficulty_calibration.reference_separation`
  FIRST: a compositional tie or win is only interpretable if a known-separable
  reference actually separates AND the probe is not at ceiling (require base
  seen_acc strictly below 1.0, else the test cannot bite). Require
  `interaction > 0` (non-additive binding) so factorization is not built in. Use the
  real-encoder-vs-random-encoder delta as the substrate arm. The honest published
  statement per the ground truth: no synthetic compositional test can bite until
  real natural-video content with non-additively-bound attributes is cached; this
  metric is defined so it is READY for that content, not so it can be run on
  ceilinged synthetic gratings and produce a fake positive.

### A4. Readout-contribution index (nonlinear structure the substrate carries beyond any projection)

- **What it measures.** Difference-in-differences:
  `(nonlinear - linear on real) - (nonlinear - linear on frozen-random)`. Positive =
  the real encoder carries NONLINEARLY-decodable structure a random projection
  cannot replicate; zero = the nonlinear gain is the probe's contribution
  (theory-ladenness), not the substrate's.
- **Why it matters.** It is the one probe-family construction that is NOT vacuous
  under the frozen-random finding: the invertible projection is absorbed identically
  by BOTH the linear and nonlinear probe on the frozen-random arm, so it cancels in
  the difference, leaving only structure the real encoder has that a rotation does
  not. It is the honest probe-based substrate test.
- **How to compute.** `nonlinear_probe.readout_contribution(x, y, hidden)` returns
  `readout_contribution_index` and `substrate_carries_nonlinear_structure`.
- **Baseline it needs.** Built in (linear-vs-nonlinear on real-vs-frozen-random). The
  capacity cap on the MLP probe (`hidden = 64`) IS the baseline discipline.
- **How it can be gamed.** (i) Uncap the probe hidden width: a big MLP fits anything
  and inflates the real nonlinear gain, but ALSO inflates the frozen-random nonlinear
  gain, so the difference-in-differences is the guard, provided the cap is fixed
  BEFORE seeing results. (ii) Overfit: with a small held-out set the nonlinear probe
  can memorize, faking a positive index. (iii) Report the index without
  seed-stability; nonlinear-probe fits are seed-sensitive.
- **False-positive prevention.** Fix `hidden` in advance and never tune it per
  result. Enforce the train/test split (already in `nonlinear_probe`, `test_frac`).
  Require the index to survive the seed harness (publish sign-flips as instability).
  Cross-check: on the census re-audit p10 nonlinear-gain was GUARANTEED-VACUOUS as a
  linear-probe delta but the readout-contribution difference is the corrected form;
  a positive index is the only clean probe-based substrate signal, so treat any
  positive here with the same skepticism as +0.31 (ask what the honest number is
  after discounting probe capacity, as the +0.31 was discounted for resolution).

### A5. Sysid / planning-licensed gate (short-horizon latent planning is real control, not memorized rollout)

- **What it measures.** Fit linear dynamics `z' = Az + Ba` on action-conditioned
  cached transitions; report one-step and k-step rollout R2, controllability Gramian
  rank/condition, and the action-shuffle delta.
- **Why it matters.** ex2_latent_planning is one of two surviving positives, and it
  survives BECAUSE its controls are non-probe (true dynamics + action-shuffle). This
  gate certifies planning is even licensed before spending environment compute; a
  rank-deficient Gramian or a zero action-delta means planning is deferred (a strong
  useful negative).
- **How to compute.** `sysid.sysid_report(Z, A_act, Znext, k)` returns `one_step_r2`,
  `kstep_r2`, `action_delta`, `gramian`, `planning_licensed`.
- **Baseline it needs.** The action-shuffle fit (permute actions, refit): a SURVIVOR
  control (it breaks the x-y correspondence in a way a linear probe cannot absorb,
  because the dynamics fit is not a probe on a fixed target). This is why ex2 held.
- **How it can be gamed.** (i) Autoregressive leakage: if `Znext` overlaps the window
  used to build `Z`, one-step R2 is inflated. (ii) The offline k-step proxy in
  `sysid_report` reuses the one-step fit for k>1 (a known simplification, noted in
  the source); a claim of long-horizon planning must NOT lean on this proxy. (iii) A
  degenerate action space where actions are collinear with state gives a spurious
  action_delta.
- **False-positive prevention.** Score ex2 on TRUE dynamics for k-step (the honest
  scoring that ex2 already used across 3 seeds), not the offline proxy. Require
  `planning_licensed` = one_step R2 > 0.5 AND action_delta > 0.05 AND Gramian
  full-rank, all three. Publish per-seed; ex2 held on all 3 seeds, that is the bar.

---

## Group B: Learning / plasticity metrics

The doctrinal question here is whether the tiny shell can be developmentally
moldable/plastic. The corpus verdict is a hard negative so far: every
biological-plasticity signature (Fisher-reopen n5, staged plasticity e3, ACh/NE n6,
tag-and-capture n4, homeostatic b4, neuromod e4) ties or loses its non-biological
baseline, no critical window (d6), no U-shaped signature (d4 = 0), no path
dependence beyond an optimizer artifact (y4 ~ 0). The one surviving plasticity-
adjacent positive is e7_sparse (sparse/gated heads halve forgetting), and it
survives PRECISELY because it is a trained-shell-dynamics metric (BWT), not a probe.
These metrics are built to keep that distinction sharp.

### B1. Backward transfer / forgetting (BWT) (continual retention, a SURVIVOR metric)

- **What it measures.** Accuracy on task 0 AFTER training through the full task
  stream, minus accuracy right after task 0. Negative = catastrophic forgetting.
- **Why it matters.** This is THE metric that survives the vacuous frozen-random
  finding, because its dynamics depend on input CONDITIONING (how SGD moves the head
  through the loss landscape), not on linear separability. e7_sparse's two surviving
  deltas are exactly BWT deltas (mean_gain +0.075, dense_BWT +0.124). Any plasticity
  claim should be expressed as a trained-shell metric like this, not a probe.
- **How to compute.** The BWT computation already lives inside
  `buffer_compression._bwt_at_bits` (acc_end minus acc_first on task 0 through a
  continual stream with a `ReplayBuffer`). Factor that BWT core out and reuse it; do
  not reimplement the continual loop per experiment (the duplication that let the
  vacuous control hide).
- **Baseline it needs.** The parameter-matched DENSE head (e7_sparse's baseline: same
  parameter count, no sparsity/gating), AND the frozen-random arm which HERE is
  legitimate because it is a trained-shell metric, not a probe (the +0.075/+0.124
  deltas are real). Also matched compute so the sparse head is not just spending more.
- **How it can be gamed.** (i) Replay everything at full precision so nothing is ever
  forgotten, then attribute retention to the mechanism instead of the buffer. (ii)
  Tiny task stream where forgetting never sets in (the toy-scale trap: ex15 showed
  plasticity loss only appears past dim 256 / thousands of tasks). (iii) Measure BWT
  on synthetic Gaussian clusters (all e7_sparse evidence so far) and claim it
  transfers to real latents without a significance test.
- **False-positive prevention.** Hold the replay budget IDENTICAL across arms
  (bytes-per-exemplar, via `retention_per_byte`). Scale the stream until the dense
  baseline actually forgets (calibrate difficulty first). Run on REAL cached latents,
  not only synthetic clusters, before claiming substrate-specificity (open question 5
  in the ground truth). Sweep seeds via `devsys.harness.sweep.run_sweep` with
  per-run `seed={s}` overrides, NOT `experiment.seeds` (the latter is a silent no-op
  for modules that read `cfg.seed`, an error already made once for e4/e7).

### B2. Fisher-trace critical-period signature (is there a sensitivity window at all)

- **What it measures.** The Fisher-information trace (sum of mean squared gradients)
  over training checkpoints; a rise-then-fall with the peak strictly inside the curve
  is the critical-period signature.
- **Why it matters.** Moldability with a developmental character requires a window
  where plasticity is high then declines. d6 found no substrate-specific window; this
  metric is how you would detect one if it existed.
- **How to compute.** `fisher_trace.fisher_trace_over_training` then
  `critical_period_signature`, which returns `rise_then_fall` and `peak_index`.
- **Baseline it needs.** A non-biological control schedule (constant LR, no staged
  plasticity) must NOT show the same rise-then-fall, else the signature is generic
  SGD warmup, not a sensitivity window. And a shuffled-task-order control (a real
  window is order-dependent; generic warmup is not).
- **How it can be gamed.** (i) Any optimizer with warmup produces a Fisher bump;
  calling it a critical period is renamed-biology (the e3/n5 failure mode). (ii)
  Coarse checkpointing hides the true peak location. (iii) Reading a rise-then-fall
  off a single seed's noisy trace.
- **False-positive prevention.** Require the signature to be ABSENT under the matched
  non-biological schedule (the tuned-baseline control), so the window is attributable
  to the developmental schedule and not to SGD. Require order-dependence. Publish the
  full trace, not just the boolean. The standing null is d6 = False; a positive must
  overturn it with seed-stability.

### B3. Retention-per-byte frontier (is memory the constraint, or the mechanism)

- **What it measures.** BWT as a function of stored-latent bit depth; whether low-bit
  replay ties full precision (memory is cheap) or collapses early (retention needs
  precision).
- **Why it matters.** It separates "the mechanism retains" from "the buffer retains,"
  the confound that can fake any continual-learning positive.
- **How to compute.** `buffer_compression.retention_per_byte(tasks, ...)` returns the
  BWT-per-bit curve, bytes-per-exemplar, `ties_full_precision`, `frontier_present`.
- **Baseline it needs.** The full-precision (32-bit) arm is the internal baseline;
  the mechanism-vs-dense comparison (B1) must be run at EQUAL bit depth so memory is
  controlled.
- **How it can be gamed.** Give the favored arm more effective memory (more bits or
  more exemplars) and credit the mechanism.
- **False-positive prevention.** Pin bytes-per-exemplar identical across mechanism
  arms; report the frontier so any memory advantage is visible. A retention claim
  that vanishes when memory is equalized is a memory result, not a plasticity result.

### B4. Transfer matrix / cross-task structure (does learning one task help another)

- **What it measures.** A T-by-T grid where cell [i][j] is a probe trained on task i,
  evaluated on task j; off-diagonal mean above chance = shared structure across
  tasks; asymmetry index = directional transfer.
- **Why it matters.** Positive transfer is a plasticity payoff (learning is
  cumulative, not siloed). It is also a cheap diagnostic reused across
  continual/compositional/analogy series instead of hand-rolled loops.
- **How to compute.** `transfer_matrix.transfer_matrix(tasks)` returns the grid,
  `off_diag_mean`, `chance`, `asymmetry_index`, `cross_task_structure`.
- **Baseline it needs.** CAUTION: each cell is a linear probe, so the per-cell
  frozen-random comparison would be vacuous. The legitimate baseline is the
  DIAGONAL (within-task accuracy) as the ceiling and `chance` as the floor;
  cross-task structure is off-diag ABOVE chance, which is a real claim (probe
  invariance to invertible projections does not manufacture cross-task transfer, it
  only preserves within-task decodability). Still, do NOT bolt a frozen-random arm on
  this and read a delta.
- **How it can be gamed.** (i) Overlapping label semantics across tasks (same classes
  relabeled) fake transfer. (ii) A dominant shared nuisance direction (lighting,
  scale) that all tasks share inflates off-diagonal accuracy without any conceptual
  transfer.
- **False-positive prevention.** Ensure tasks are label-disjoint or explicitly note
  shared labels. Regress out or hold constant known shared nuisances before reading
  transfer as conceptual. Report the asymmetry index; symmetric transfer is more
  likely a shared-nuisance artifact than directed reuse.

### B5. Rejuvenation / plasticity-loss recovery (scale-dependent moldability)

- **What it measures.** Whether trained-shell plasticity (capacity to learn a new
  task) DEGRADES with accumulated training, and whether a rejuvenation intervention
  (re-init a fraction of head units, reset optimizer state) RESTORES it, at what
  retention cost.
- **Why it matters.** ex15_rejuvenation is the one early lead that plasticity loss is
  real but only past toy scale (dim 256, thousands of tasks). This is the metric that
  operationalizes the only surviving moldability lead.
- **How to compute.** Track new-task learning speed/asymptote over the stream (reuse
  the B1 continual loop and BWT core), then apply the reset and re-measure. New-task
  accuracy-after-fixed-budget is the plasticity proxy; the recovery delta after
  rejuvenation is the signal; the drop in old-task accuracy is the cost.
- **Baseline it needs.** (i) No-reset control (plasticity keeps declining). (ii) A
  matched-compute control: rejuvenation adds training, so a matched-FLOP no-reset arm
  must not recover as much. (iii) Small-scale control: the effect must be ABSENT at
  toy scale (dim < 256), else it is not the scale-dependent phenomenon claimed.
- **How it can be gamed.** (i) Reset = more training, and more training on recent
  tasks looks like recovered plasticity; matched compute is the guard. (ii) Cherry
  scale where the effect appears. (iii) Ignore the retention cost and report only the
  recovery.
- **False-positive prevention.** Always report the recovery-vs-retention-cost pair
  (rejuvenation that recovers plasticity by forgetting everything is trivial).
  Require the effect to be scale-dependent (present at scale, absent at toy scale) as
  a positive discriminator, not just present at one scale. Matched compute mandatory.

---

## Group C: Workspace / mixture metrics

The mixture-of-thinking framing is: a small set of specialized shell modules
(sparse/gated MoE-style heads from e7, an ensemble of predictors, an iterative
refiner, a planner) that are ROUTED or COMBINED per input. The metrics here measure
whether routing/mixing does real work beyond a single monolithic head of equal
compute. The prior from the corpus is skeptical: gains from "more modules" are
usually bought compute or unrolled depth, and emergent codes across modules are
idiolects (p5/s5/y3 below the frozen-random floor). These metrics are built to
force a mixture claim through matched-compute and seed-stability.

### C1. Routing utilization / load balance (are experts actually used, or is routing a no-op)

- **What it measures.** The distribution of routing weight (or hard top-k selection
  frequency) across experts/heads over a held-out set: entropy of the average
  routing distribution, per-expert utilization, and dead-expert count.
- **Why it matters.** A MoE that collapses to one expert is a single head with wasted
  parameters; a MoE that routes uniformly regardless of input is not routing on
  content. Both are the null for "mixture helps," and both are common failure modes.
- **How to compute.** From cached latents, run the (already-trained, tiny) gating
  network to get routing weights; compute the mean routing distribution and its
  entropy, and the fraction of experts below a utilization floor. Pure tensor stats
  on cached activations, no encoder. This is NEW glue but it is stats over an
  existing shell module (`heads.py` kWTA/MoE), not a new diagnostic family.
- **Baseline it needs.** Two nulls: (i) uniform routing (entropy = log(num_experts),
  the "routing carries no information" ceiling on entropy), and (ii) input-shuffled
  routing (route with weights computed from a DIFFERENT input) as the "routing is not
  content-dependent" control (a survivor, non-probe control).
- **How it can be gamed.** (i) High utilization with random routing looks balanced but
  is content-blind. (ii) Auxiliary load-balancing losses can force uniform
  utilization that LOOKS healthy while destroying content-routing (utilization and
  content-dependence trade off; measure both). (iii) Report entropy alone: a
  perfectly content-dependent 2-expert router has low entropy and is fine.
- **False-positive prevention.** Utilization is a HYGIENE metric (it rules out
  collapse and dead experts), never a positive on its own. Pair it with C2
  (content-dependence) and C3 (does mixing beat matched-compute monolith). The
  input-shuffle control is the one that certifies routing is on content, not noise.

### C2. Routing content-dependence / specialization (do experts specialize on interpretable structure)

- **What it measures.** Mutual-information-style association between the routed expert
  and a known latent factor (class, nuisance, task): does expert selection PREDICT a
  factor above chance, and do different experts have distinguishable input
  distributions.
- **Why it matters.** Specialization is the mechanistic claim behind "mixture." If
  experts do not partition inputs by any recoverable structure, the mixture is
  cosmetic.
- **How to compute.** Treat hard expert assignment as a discrete code over cached
  latents. Then: (i) `linear_probe` from the latent to predicted expert (can expert
  choice be recovered, i.e. is it lawful) and (ii) cross-tab expert vs known factor
  for association. For stability across seeds, reuse
  `seed_consistency.hungarian_code_agreement` / `code_stability` treating expert
  assignments as the discrete code (k = num_experts): does the expert PARTITION recur
  across seeds above the 1/k floor, or is it a per-run idiolect (the p5/s5/y3 verdict)?
- **Baseline it needs.** The 1/k random-relabeling floor (built into
  `code_stability.chance`) and a random-codebook control. For factor association,
  chance = marginal factor frequency.
- **How it can be gamed.** (i) With many experts, some expert will correlate with some
  factor by chance (multiple comparisons); (ii) a factor that is trivially decodable
  from the latent will be trivially associated with any lawful routing, so
  association does not prove the ROUTING adds anything.
- **False-positive prevention.** Cross-seed stability is the load-bearing test: an
  expert partition that does not recur across seeds above 1/k is an idiolect, not
  specialization, and the instability IS the result (never published as a positive),
  exactly as in the private-language series. Correct for multiple comparisons across
  experts and factors. Association is necessary, not sufficient; the sufficiency test
  is C3.

### C3. Mixture gain over matched-compute monolith (the only positive that counts)

- **What it measures.** Task metric (accuracy, BWT, rollout R2, whichever the mixture
  targets) of the routed/mixed system MINUS the same metric for a single monolithic
  head/predictor with EQUAL total FLOPs and parameters.
- **Why it matters.** This is the one metric that can license "mixture helps." Every
  other workspace metric is hygiene or mechanism-characterization; this is the payoff,
  and matched compute is what separates a real routing gain from bought compute or
  unrolled depth (the standing failure mode).
- **How to compute.** Run both systems on cached latents; use `compute.accounting` and
  `compute.matched_within` to certify FLOP/param parity within tolerance BEFORE
  comparing scores. For continual metrics use the B1 BWT core; for reasoning use the
  A-group refiner metrics; for planning use A5.
- **Baseline it needs.** The matched-FLOP, matched-param monolith is THE baseline. For
  e7_sparse specifically the monolith is the parameter-matched dense head (that is how
  the +0.075/+0.124 survived). Also a tuned baseline (the monolith must be
  hyperparameter-tuned, not a strawman) and seed-stability.
- **How it can be gamed.** (i) Count only active FLOPs for the sparse system (top-k)
  but total FLOPs for the monolith, or vice versa; the accounting must use the SAME
  convention (report both active and total, and match on the one being claimed). (ii)
  Under-tune the monolith. (iii) Win on a synthetic regime that ceilings or that is
  too easy to forget (the difficulty trap). (iv) Report one seed.
- **False-positive prevention.** `matched_within` gate must pass (ratio <= 1.1) and
  the FLOP convention must be stated and identical. Difficulty-calibrate the regime
  (`reference_separation`) so a tie or win is interpretable. Tune the monolith with
  the same budget as the mixture. Sweep seeds via `run_sweep` (per-run `seed={s}`)
  and publish sign-flips. e7_sparse cleared exactly this bar on a 30-run sweep; that
  is the standard, and the open question (5) is whether it holds on REAL cached
  latents with a formal significance test, not just synthetic Gaussian clusters.

### C4. Ensemble disagreement / epistemic-vs-aleatoric separation (mixture as uncertainty, noisy-TV guard)

- **What it measures.** Variance across ensemble members (epistemic disagreement) vs
  raw prediction error (which includes aleatoric noise) on a learnable region vs an
  irreducible-noise region.
- **Why it matters.** The mixture-as-ensemble claim is that member disagreement is a
  usable uncertainty signal that a point predictor lacks. It must pass the noisy-TV
  guard: disagreement collapses on irreducible noise (nothing to learn) even while
  raw error stays high. e4 FAILED this (amplified error on noise 30/30); this metric
  is how e4 was caught.
- **How to compute.** `ensemble.Ensemble.mean_and_disagreement` for the raw signal;
  `noisy_tv.noisy_tv_diagnostic` for the full contract, returning
  `noise_error_stays_high`, `epistemic_collapses_on_noise`,
  `learning_progress_separates`.
- **Baseline it needs.** The point-predictor (single member) which CANNOT separate
  epistemic from aleatoric, plus the streamed-fresh-noise construction (so the
  ensemble cannot memorize a finite noise set and fake learnability).
- **How it can be gamed.** (i) A finite, reused noise set becomes memorizable, so
  disagreement drops on noise for the wrong reason (looks like correct collapse);
  the fresh-sample streaming in `noisy_tv_diagnostic` is the guard. (ii) Members that
  are near-identical (poor initialization diversity) have low disagreement
  everywhere, faking the collapse-on-noise signature. (iii) Learning progress
  measured over too short a window.
- **False-positive prevention.** Require ALL THREE booleans jointly
  (`noise_error_stays_high` AND `epistemic_collapses_on_noise` AND
  `learning_progress_separates`); any single one is gameable. Verify member diversity
  (disagreement must be HIGH on the learnable region, not just low on noise).
  Learning progress is the signal intrinsically immune to noisy-TV, so it is the
  tiebreaker. This is the standing curiosity guard for any exploration/uncertainty
  claim in the workspace.

### C5. Workspace-code seed-stability (shared representation vs per-run idiolect)

- **What it measures.** Whether the representation the mixture converges on (the
  routing partition, the shared "workspace" latent that modules read/write, the
  emergent inter-module code) RECURS across seeds: mean pairwise CKA for continuous
  workspace latents, Hungarian-matched agreement for discrete codes.
- **Why it matters.** A "global workspace" that different seeds fill with unrelated
  content is an idiolect, not a shared medium. The corpus verdict on emergent
  inter-module codes is that they sit BELOW the frozen-random floor (p5/s5/y3);
  this metric enforces that skepticism on any workspace claim.
- **How to compute.** `seed_consistency.cross_seed_cka` (continuous) and
  `code_stability` (discrete). CKA is rotation-invariant, so for a CONTINUOUS
  workspace latent it is a SURVIVOR-adjacent measure (invariant to the invertible
  projection by construction, which is why it is the RIGHT tool: it asks whether
  structure recurs, not whether a probe can decode it).
- **Baseline it needs.** The frozen-random floor for CKA and the 1/k floor for code
  agreement (both built in). NOTE: for CKA the frozen-random floor is legitimate
  because CKA measures representational-structure similarity, not probe decodability,
  so the invertibility that makes probes vacuous does not apply the same way (two
  random projections of the SAME latent have high CKA to the latent but the question
  is whether two SEEDS produce high CKA to EACH OTHER).
- **How it can be gamed.** (i) A workspace dominated by a single high-variance
  direction (high anisotropy) has high CKA across seeds trivially; check
  `geometry.anisotropy` and `effective_rank` first. (ii) Averaging away the per-seed
  variation before measuring.
- **False-positive prevention.** Report `geometry.effective_rank`/`anisotropy`
  alongside CKA: high CKA on a rank-1 (collapsed) workspace is meaningless. Require
  cross-seed CKA to clear the frozen-random floor by margin, and code agreement to
  clear 1/k by margin. Publish the instability as the result when it fails (the
  private-language contract). A stable workspace code is a prerequisite for any claim
  that modules share meaning; instability refutes it directly.

---

## 13. Cross-cutting guardrails (applied to every metric above)

These are the standing controls from the harness, restated as the acceptance
checklist a metric result must pass before it is published as a positive. They are
wired ONCE at the harness level and reused, not reimplemented per metric.

1. **Substrate control, corrected.** Frozen-random projection is VACUOUS for probe
   metrics (invertible). For any probe-family metric (A3, C2-linear-probe pieces,
   B4 cells), the legitimate substrate control is real-V-JEPA vs random-ENCODER
   features (`substrate_vs_random_features.py`, gold-standard
   `substrate_vs_random_init_vit.py`), and/or the difference-in-differences
   construction (A4). Frozen-random is legitimate ONLY for trained-shell-dynamics
   metrics (B1 BWT, A5 sysid, A1/A2 dynamics) and for CKA-style structure similarity
   (C5). Each metric above states which case it is in.
2. **Matched compute.** Any metric where one arm can spend more FLOPs (A1, A2, B5,
   C3) must pass `compute.matched_within` BEFORE scores are compared; iteration that
   helps must beat unrolled depth at equal FLOPs.
3. **Tuned baseline.** The comparison arm must be hyperparameter-tuned with the same
   budget (guards renamed-biology and strawman-monolith confounds).
4. **Noisy-TV guard.** Any curiosity/uncertainty/exploration signal (C4) must ignore
   irreducible aleatoric noise via the three-boolean contract on streamed fresh noise.
5. **Seed-stability.** Sweep via `devsys.harness.sweep.run_sweep` with per-run
   `seed={s}` overrides (NOT `experiment.seeds`, a silent no-op for cfg.seed
   modules). Publish sign-flips as instability; an unstable positive is not a
   positive (A4, B2, C2, C3, C5 especially).
6. **Difficulty calibration.** Before trusting any TIE or near-ceiling result (A3,
   B1, C3), certify the regime with `difficulty_calibration.reference_separation`;
   a tie in an uncalibrated regime is uninterpretable, and a win at ceiling is
   vacuous. This is the binding constraint of the whole program: synthetic gratings
   and bound objects ceiling at 1.0, so no compositional/mixture metric can bite
   until real natural-video content with non-additively-bound attributes is cached.
7. **Determinism.** CPU is bit-identical (`determinism.assert_reproducible`); report
   Metal/CPU spread if any metric is run on device.

The through-line: every metric here extends an existing diagnostic, carries its own
gaming vector and the specific control that kills it, and states whether that
control is a SURVIVOR or a CASUALTY of the vacuous frozen-random finding. No metric
is licensed as a positive on its own; a positive is the conjunction of the metric
clearing its threshold AND the corrected substrate control AND matched compute AND
seed-stability AND difficulty calibration. That conjunction is exactly what
e7_sparse (C3/B1) and ex2 (A5) cleared and what the refuted positives did not.
