# Sections 8 and 9: The Custom-Model Pathway and Candidate Custom Architectures

## What this document is, and is not

This is the part of the review that most wants to be a wish list and must not be. Brain's doctrine is frozen substrate plus a tiny trainable shell, cached-latent-first, no frontier compute. A custom-trained model of any kind is a violation of that doctrine, so the burden of proof is on the pathway, not on the doctrine. Nothing here proposes training anything. Sections 8 and 9 define the evidence gates that would license a custom model, the staged ladder that would build one only after each gate is cleared, and the specific architectures that ladder could terminate in, each one paired with the null it must beat and the open model it must be better than. If the gates are never cleared, the honest outcome is: keep the frozen encoder, keep the shell small, ship the taxonomy-3 bounds, and never train a substrate. That is a legitimate terminal state and the document says so repeatedly.

The single most important fact framing everything below: as of today the corpus does NOT justify a custom model. The one valid substrate-is-special result (the +0.31 nuisance-invariance delta, honestly ~0.21 to 0.23 after the resolution confound) points the OTHER way: it says the frozen encoder carries real learned structure and should be KEPT. No experiment has yet shown the frozen substrate is BOUNDED in a way a custom model would fix. Two of the three custom-triggering conditions are therefore currently UNMET and one is UNTESTED. This document is a contingency ladder, not a plan of record.

---

# SECTION 8: The Custom-Model Pathway

## 8.1 The decision gates: exactly what must fail before Brain trains a substrate

The doctrine forbids custom training by default. The only thing that overturns a default is evidence. So the pathway is defined negatively: it lists the exact failing conditions that, if and only if they are observed on real (not synthetic, not ceilinged) content, license the next stage. A custom model is not a goal to be reached; it is a fallback that is unlocked only by a demonstrated ceiling on the cheaper option.

### Gate C1 (substrate-bounded on abstraction): the frozen substrate cannot FACTOR bound attributes off-ceiling

Failing condition, stated precisely: on real natural-video content with non-additively bound attributes (color, shape, position, motion entangled so that no single factor is linearly free), run through the frozen encoder and probed with the held-out-combination gate against the CORRECT control (real V-JEPA vs random-init same-arch ViT at matched resolution, NOT the vacuous full-rank latent projection), the frozen substrate's held-out-combination accuracy collapses to chance (delta over random-init ViT below +0.05) while a difficulty-calibration reference (D3) certifies the regime carries real separable structure. In plain terms: the content is genuinely compositional, a known-good featurizer could solve it, and the frozen substrate cannot factor the bound attributes.

Why this is the gate and not something cheaper: `compositional_under_nuisance.py` is exactly this test at synthetic scale and it is IN FLIGHT. If it returns the NON-COMPOSITIONAL / MEMORIZED verdict (v_held at or below chance + 0.1) on content that D3 certifies is separable, that is the first real evidence of a substrate bound. But synthetic shapes have ceilinged five times; the gate must be cleared on REAL bound-attribute video (open question 4), which does not exist on disk. So C1 is currently UNTESTED, not failed.

CONTROL CAVEAT (do NOT let the current script's "COMPOSITIONAL AND SUBSTRATE-SPECIFIC" verdict count as C1 evidence): as shipped, `compositional_under_nuisance.py:125-131` fires its substrate-specific verdict when real V-JEPA beats `random_pixel_features` by delta>0.1, and that control (`substrate_vs_random_features.py:104-113`) avg-pools 256px down to 32px, so the control sees 64x less spatial information than V-JEPA. That is precisely the resolution-confounded, vacuous-control artifact the corpus spent its most important correction unlearning (see section 15 sec 2.1 and 8_09 sec 8.1's own control mandate: the CORRECT control is real V-JEPA vs random-init same-arch ViT-L at matched 256px, NEVER random pixels). A delta over 32px random pixels cannot clear C1: it conflates pretraining with a 64x resolution gap. C1 therefore remains UNTESTED until the random-init-ViT rerun (`substrate_vs_random_init_vit.py` at matched 256px) lands; any "substrate-specific" reading of the random-pixel version is "resolution-confounded, not gate-clearing" and must not trigger the fork.

What clearing C1 unlocks: Stage 1 (multi-substrate atlas) and Stage 2 (dense latent scaling), because a factoring failure could be a pooling artifact (dense tokens fix it) or a V-JEPA-objective artifact (a different substrate fixes it) before it is a "need custom weights" failure. C1 does NOT directly unlock Stage 5 (custom pilot); it unlocks the cheaper substrate-swap experiments that must be exhausted first.

### Gate C2 (substrate-bounded on adaptation): every substrate ties on the trained-shell dynamics that actually survive controls

Failing condition: the two surviving positives (e7_sparse's continual-learning deltas, ex2's planning-on-true-dynamics wins) do NOT improve as substrate quality improves across the atlas (V-JEPA L / H / g, plus a different-objective encoder such as an image DINO or a video-contrastive model). Specifically, if e7_sparse's forgetting advantage and ex2's planning advantage are FLAT across substrates on real cached latents (per-encoder BWT and planning-gain within seed spread), the adaptation dynamics are shell-architectural and substrate-agnostic, so no substrate swap and no custom substrate can help them; the ceiling is in the shell, not the encoder.

Why it matters for the pathway: C2 failing is a STOP signal for custom SUBSTRATE work on the adaptation axis and a redirect toward Stage 3 (workspace shell) and Stage 4 (repair/adaptation), which are shell-side. It is a pathway gate because it tells you WHICH branch (substrate vs shell) is bounded. C2 is currently UNTESTED (e7_sparse ran on synthetic Gaussian clusters only, no cross-substrate sweep, open question 5).

### Gate C3 (moldability requires trainable substrate): plasticity phenomena appear only past frozen-shell scale

Failing condition: the developmental signatures the corpus has repeatedly failed to find on a frozen substrate (critical/sensitive window d6, path-dependence y4, U-shaped overgeneralization d4) appear ONLY when the substrate itself is allowed to adapt, and are provably absent with any frozen substrate plus trainable shell at studio scale. ex15_rejuvenation is the early lead here: plasticity loss and its rejuvenation appear past toy scale (dim 256, thousands of tasks). If, at studio scale on real latents, moldability phenomena are present with a trainable substrate and absent with every frozen one plus a large shell, the moldability doctrinal question cannot be answered without a trainable substrate.

Why it is the strictest gate: moldability is one of the two central doctrinal questions and the corpus has zero substrate-level positive on it. C3 failing is the ONLY condition that licenses Stage 5's developmental variant (a substrate trained with a plasticity objective), and even then only after Stages 1 to 4 have shown the frozen route cannot reach it. C3 is UNTESTED at studio scale.

### The gating logic (the AND, not the OR)

A custom substrate pilot (Stage 5) is licensed only when: (C1 fails on real bound-attribute video after both a substrate swap in the atlas and dense tokens have been tried and also failed) OR (C3 fails at studio scale after the workspace and repair shells have been tried and also failed). C2 does not license custom substrate; it redirects effort to shell stages. The full Mixture-of-Thinking custom model (Stage 6) is licensed only when Stage 5's single-substrate pilot has itself cleared its success metric AND a multi-substrate atlas (Stage 1) has shown that composing several substrates beats any single one. This is deliberately hard to satisfy. That is the point: the pathway should be reachable only by evidence that does not currently exist.

Two standing kill-switches apply to every stage below: (a) if a cheaper frozen-shell configuration matches the custom result at matched compute, the custom work is unjustified and stops; (b) if the substrate-is-special delta (currently +0.21 to +0.31) grows rather than shrinks as content gets harder, the frozen encoder is winning and custom substrate work should not even begin.

---

## 8.2 The seven stages

Each stage is a rung. You do not climb to rung N+1 until rung N's success metric is met AND the gate for N+1 is cleared. Every rung names what stops or pivots it. Compute is quoted against the two real machines: the M3 Pro (~18 core GPU, 18 GB, where the pre-Studio corpus ran) and the Mac Studio M2 Max (~38 core GPU, 96 GB, the near-term target); "wider-box" means a rented multi-GPU CUDA box; "moonshot" means frontier compute Brain does not have and does not plan to buy.

### Stage 0: cached-latent shells (WHERE WE ARE)

- Purpose: exhaust everything the frozen substrate plus a tiny trainable shell can do on cached latents before touching a substrate. This is the doctrine's home base and the null for every later stage.
- Required evidence to LEAVE this stage: a demonstrated ceiling. Specifically, a gate (C1, C2, or C3) failing on real content, not synthetic. Until then Stage 0 is not finished.
- Compute: seconds to minutes per experiment on cached latents, CPU or single-GPU, exactly the pre-Studio regime. No substrate training.
- Data: cached V-JEPA latents of real video (the deferred prerequisite, open question 4). The synthetic caches are exhausted (they ceiling).
- Implementation: the existing shell {predictor, heads, ensemble, buffer, plasticity, consolidation, neuromod, modulation, refine} plus the standing controls harness. The immediate work is the two in-flight scripts (`substrate_vs_random_init_vit.py`, `compositional_under_nuisance.py`) landing, then re-running them on real bound-attribute video.
- Success metric: a clean answer to each gate (fail or clear) on real content, with the CORRECT control (random-init ViT, not full-rank projection) and a D3 difficulty certification.
- Failure metric: another ceiling (all cells 1.0), which means the content is still too easy and Stage 0 has produced no new information.
- What it unlocks: whichever of Stages 1 to 4 the failing gate points to.
- Stop/pivot: if real bound-attribute video also ceilings after difficulty calibration, the honest conclusion is the substrate is UNBOUNDED at every difficulty Brain can construct, and the whole custom pathway is unjustified; ship the frozen shell and stop.

### Stage 1: multi-substrate atlas

- Purpose: before assuming a bound is intrinsic to perception, test whether it is intrinsic to THIS substrate. Run the same failing gate across a small atlas of frozen encoders (V-JEPA L / H / g, an image DINOv2, a video-contrastive model) to separate universal bounds from V-JEPA-objective-specific ones. This directly serves the newest standing control (cross-substrate convergence: universal vs modality/objective/architecture-specific).
- Required evidence to leave: the gate that failed at Stage 0 either (a) clears on a different substrate (bound was V-JEPA-specific, KEEP the better frozen substrate, no custom work) or (b) fails on ALL substrates (bound is universal, escalate to Stage 2/5).
- Compute: encoding is the cost. On the Studio, one ViT-L forward is ~seconds on Metal (measurement pending per STUDIO_MAXIMIZATION); a full atlas over a few thousand real clips at three model sizes is studio-scale, hours to a couple of days of encode, then cached forever. No training.
- Data: the same real bound-attribute video, encoded once per substrate.
- Implementation: the encoder loader already supports L/H/g via config; adding DINOv2 and a video-contrastive backbone is a loader change plus a weights download, NOT custom training. The atlas factor field already exists in the registry (`proof.atlas_factor`).
- Success metric: for at least one gate, a substrate exists whose delta over its own random-init control exceeds +0.15 off-ceiling where V-JEPA-L did not, OR all substrates tie (universal bound confirmed).
- Failure metric: encoding cost dominates and no substrate moves the needle beyond seed spread on a D3-certified regime (underpowered, needs more data or harder content).
- What it unlocks: a substrate-specific clear KEEPS a frozen encoder and ends the pathway for that gate; a universal fail unlocks Stage 2 then Stage 5.
- Stop/pivot: if a cheap off-the-shelf frozen substrate clears the gate, STOP, adopt it, never train. This is the most likely good outcome and the cheapest.

### Stage 2: dense latent scaling

- Purpose: test whether the bound is the POOLING interface, not the weights. The current encoder mean-pools to a 1024-d vector; dense (pre-pool spatial-token) latents may carry the compositional structure the pooled vector loses. This is open question 4's cheaper cousin and the P7 object-binding-before-pooling lane.
- Required evidence to leave: a dense or coarse-grid-pooled (2x2, 4x4) representation clears the failing gate where full mean-pool did not (pooling was the bound, no custom weights needed) OR dense ties pooled (the bound is in the weights, escalate).
- Compute: dense latents are far larger (thousands of tokens per clip vs one pooled vector), so this is the first real MEMORY pressure point. Studio 96 GB makes dense caches feasible where the M3 Pro's 18 GB did not; a `studio-2tb` disk profile (documented as a one-line change) may be needed for dense caches. Encode-once, then cached. No training.
- Data: real video re-encoded WITHOUT the final pooling, or with coarse-grid pooling exposed by the existing encoder (a code check per DOCTRINE_SYNTHESIS 3b, not a new download if pre-pool tokens are exposed).
- Implementation: expose the encoder's pre-pool tokens (loader flag), add coarse-grid pooling as an intermediate, re-run the gate's probe on dense features. Note the dense-vs-pooled probe already REFUTED "pooling destroys simple spatial factors" (orientation decodes at 1.0 from the pooled vector), so this stage is specifically about COMPOSITIONAL structure at non-ceiling difficulty, not single factors.
- Success metric: held-out-combination delta (dense minus pooled) exceeds +0.1 off-ceiling on a D3-certified regime.
- Failure metric: dense ties pooled (both at chance or both at ceiling), meaning the interface is not the bound.
- What it unlocks: a dense win motivates KEEPING V-JEPA but adopting dense V-JEPA 2.1 (the frozen-dense branch of THE fork, currently reopened by the +0.31 result); a dense tie escalates to Stage 5.
- Stop/pivot: if dense caches blow the disk or memory budget without a positive, STOP; the dense branch is not worth a 2 TB commitment on a tie.

### Stage 3: workspace shell

- Purpose: a shell-side rung, unlocked by C2 (adaptation bounds are shell-architectural). Build a global-workspace-style routing shell (a small trainable controller that composes the frozen substrate's features across a shared latent workspace, the Mixture-of-Thinking idea at SHELL scale before substrate scale). Tests whether a richer shell, not a richer substrate, lifts the adaptation and planning positives.
- Required evidence to leave: the workspace shell beats a parameter-matched dense shell AND matched-compute unrolled depth on e7_sparse's forgetting metric and ex2's planning-on-true-dynamics, on real latents, across seeds.
- Compute: shell training only, cached latents, seconds to minutes per run on the Studio; a 30-run seed/axis sweep is hours. No substrate training. This is squarely inside doctrine.
- Data: cached real latents (same as Stage 0).
- Implementation: extend the existing {ensemble, modulation, refine} modules into a routed workspace; the MoT experiments (Section 9, arch D minimum version) are the concrete build.
- Success metric: workspace shell BWT beats dense-shell BWT by more than the e7_sparse margin (+0.075) AND beats matched-compute unrolled depth, seed-stable (no sign flips).
- Failure metric: workspace ties dense or ties unrolled depth (the win is capacity or depth, not routing), or sign-flips across seeds (instability, published as such).
- What it unlocks: a workspace win is a doctrine-COMPLIANT positive (no custom substrate needed) and may itself be the shippable result; a tie escalates the adaptation question to Stage 4.
- Stop/pivot: if the workspace never beats matched-compute unrolled depth, STOP the routing line; iteration is just depth (the p9/ex17 finding recurs).

### Stage 4: repair / adaptation shell

- Purpose: the last shell-side rung for moldability, unlocked by C3 partially (ex15_rejuvenation's scale-dependent plasticity loss). Test whether a repair mechanism (rejuvenation, structural growth b8, targeted re-initialization) restores plasticity at studio scale WITHOUT touching the substrate, and whether that restoration carries a retention cost.
- Required evidence to leave: rejuvenation measurably restores plasticity at studio scale (dim and task counts past the toy regime where ex15 first saw it) AND the effect is absent for a frozen-random control AND survives matched-compute. If restoration is real but ONLY works when the substrate adapts, C3 fails and Stage 5 is licensed.
- Compute: shell training at larger dim (256 to low thousands) and thousands of tasks, cached latents, studio-scale (hours to a day per sweep). No substrate training unless the C3-failing variant is run, which trains ONLY a small adapter, not the full encoder.
- Data: cached real latents, long task streams.
- Implementation: extend ex15_rejuvenation and b8_structural_growth to studio scale; add the C3 discriminating arm (small trainable substrate adapter vs frozen substrate).
- Success metric: rejuvenation restores plasticity with a quantified retention cost, seed-stable, beating frozen-random and matched-compute.
- Failure metric: plasticity loss never appears at reachable scale (moldability is a non-phenomenon here), or rejuvenation ties doing nothing.
- What it unlocks: a frozen-substrate restoration is a moldability positive inside doctrine; a C3 failure (restoration needs an adapting substrate) is the ONLY moldability-based license for Stage 5.
- Stop/pivot: if plasticity loss is not reachable at studio scale, STOP the moldability-via-scale line; the phenomenon is a toy artifact.

### Stage 5: custom substrate pilot

- Purpose: the first stage that trains a substrate, and only a SMALL one, as a pilot. Unlocked strictly by (C1 fail on real video after atlas and dense both failed) OR (C3 fail after workspace and repair both failed). Trains a small custom encoder (or a small trainable adapter on a frozen backbone) with the specific objective the failing gate implicated (compositional factoring for C1, plasticity for C3).
- Required evidence to leave: the pilot custom substrate beats the BEST frozen substrate from the atlas on the exact failing gate, off-ceiling, at MATCHED compute (a frozen substrate given equal shell FLOPs must lose), seed-stable, on real content. This is the hardest bar in the document.
- Compute: this is the first rung that needs a wider-box (rented multi-GPU CUDA). A small pilot substrate (ViT-S/B scale, not L) on a modest real-video dataset is days on a single A100-class GPU, not frontier. The Studio can prototype the objective and data pipeline; the actual training is wider-box.
- Data: real video, now used for TRAINING not just encoding; the dataset must be large enough that a small substrate can learn the implicated structure (compositional bound attributes or a plasticity-inducing curriculum).
- Implementation: a minimal custom-JEPA or object-centric-JEPA (Section 9, arch A/B minimum versions) trained on the implicated objective; frozen backbone plus trainable adapter is the cheaper variant to try first.
- Success metric: custom-minus-best-frozen delta exceeds +0.15 off-ceiling at matched shell compute, seed-stable.
- Failure metric: custom ties the best frozen substrate (custom training bought nothing the frozen atlas did not), or the win vanishes at matched compute (it was capacity/data, not the custom objective).
- What it unlocks: a pilot win is the ONLY thing that licenses Stage 6; a tie ENDS the custom pathway and returns to Stage 0/1 (keep frozen).
- Stop/pivot: any tie at matched compute STOPS the pathway. A custom substrate that does not beat a frozen one at equal compute is a doctrine violation with no payoff.

### Stage 6: custom Mixture-of-Thinking model

- Purpose: the terminal rung. A full Mixture-of-Thinking model that composes MULTIPLE trained-or-frozen substrates through a trained workspace, routing "thinking modes" (perceptual, predictive, planning, compositional) as a mixture. Unlocked only when Stage 5's pilot cleared its bar AND Stage 1's atlas showed composing substrates beats any single one.
- Required evidence to leave (i.e., to declare success): the full MoT model beats (a) the best single frozen substrate plus workspace shell, (b) the Stage 5 pilot alone, and (c) the strongest open model on the same held-out compositional-and-adaptation battery, all at documented compute, seed-stable, with every standing control passing.
- Compute: this is the only rung that approaches (but should still avoid) moonshot. A multi-substrate MoT with a trained workspace and at least one trained substrate is wider-box to small-cluster, days to weeks. If it needs frontier compute, the design has failed the doctrine and should be descoped.
- Data: the full real-video corpus plus any modality the atlas added (image, action-conditioned).
- Implementation: Section 9 arch D full version, composing arch A/B/C substrates through the Stage 3 workspace.
- Success metric: beats all three references above on the combined battery; every gate that licensed the pathway is now cleared by the MoT model where the frozen shell could not.
- Failure metric: fails to beat the single-substrate-plus-workspace baseline at matched compute (the mixture bought nothing), or needs frontier compute (out of doctrine).
- What it unlocks: nothing further; this is the terminal architecture. Success here is the only outcome that retroactively justifies the whole pathway.
- Stop/pivot: if the mixture does not beat its best single-substrate component at matched compute, descope to that single component and stop; a mixture that does not beat its parts is pure overhead.

---

# SECTION 9: Candidate custom architectures

Six candidates, A to F. Each is a contingency: it is built only if the Section 8 gates route to it. For each I give what it PROVES (the doctrinal claim it would settle), dataset, objective, losses, trainable-vs-frozen split, the diagnostics that must pass, the null it must beat, why it would be better than the best existing open model (it must clear a real bar, not just exist), and minimum / studio / wider-box / moonshot versions. Every architecture inherits the standing controls: beat the CORRECT substrate control (random-init same-arch, never the vacuous full-rank projection), match compute, beat a tuned baseline, seed-stability, D3 difficulty certification, and cross-substrate convergence.

## Arch A: Custom JEPA world model

- Proves: whether a substrate trained with Brain's OWN objective on Brain's OWN content beats a general-purpose frozen V-JEPA on the specific compositional/adaptation gate that failed. Settles the C1 substrate-bound question for the prediction-based-representation family.
- Dataset: real bound-attribute video (the deferred prerequisite), plus the difficulty-calibrated compositional battery.
- Objective: masked-latent prediction (JEPA), but with the mask policy TARGETED at the failing factor (mask one bound attribute, predict its latent from the others) so the objective directly pressures factoring.
- Losses: JEPA latent-prediction loss (predictor MSE in latent space) plus a variance-covariance regularizer to prevent collapse; NO pixel reconstruction (the whole point of JEPA).
- Trainable vs frozen: encoder trainable at pilot scale (ViT-S/B), predictor trainable; alternatively a frozen V-JEPA backbone with a trainable factoring adapter (the cheaper C1 variant tried first).
- Diagnostics that must pass: held-out-combination gate off-ceiling, D3 difficulty certification, matched-compute vs a frozen V-JEPA given equal adapter FLOPs, seed-stability, and the collapse check (variance floor, since JEPA can trivially collapse).
- Null: a random-init same-arch ViT and the best frozen atlas substrate both tie the custom model on held-out combinations. If the null holds, the custom objective bought nothing.
- Better than existing open models how: it must beat frozen V-JEPA 2 / 2.1 AND a frozen DINOv2 on the held-out compositional battery at matched compute; "better" means a measured +0.15 off-ceiling delta, not a qualitative story.
- Minimum (Studio): frozen backbone + tiny factoring adapter, cached-latent training, hours. Studio: same, larger adapter, full seed sweep. Wider-box: train a ViT-S JEPA from scratch on real video, days on one GPU. Moonshot (avoid): ViT-L from scratch, weeks on a cluster.

## Arch B: Object-centric JEPA

- Proves: whether explicit object slots (slot-attention-style) BEFORE pooling give the compositional factoring the pooled substrate lacks. Settles whether the C1 bound is a binding-before-pooling problem (P7 lane) rather than a weights problem.
- Dataset: real multi-object video with bound attributes; the compositional-under-nuisance content scaled to multiple simultaneous objects.
- Objective: JEPA latent prediction over SLOT latents (predict masked slots from visible slots), forcing each slot to carry a bindable object.
- Losses: slot-JEPA prediction loss, plus an entropy/assignment regularizer on slot attention to prevent slot collapse, plus the variance-covariance anti-collapse term.
- Trainable vs frozen: frozen V-JEPA dense tokens as input, TRAINABLE slot-attention module and predictor; the substrate stays frozen, only the binding shell trains. This keeps it closer to doctrine than arch A.
- Diagnostics that must pass: held-out-combination on multi-object scenes, a binding-specificity test (swap one object's attribute, the correct slot's latent should change and others should not), matched compute vs dense-token probe without slots, seed-stability.
- Null: dense tokens WITHOUT slots (Stage 2 output) tie the slotted model on held-out multi-object combinations. If so, slots add nothing over dense.
- Better than existing open models how: must beat both frozen dense V-JEPA and an off-the-shelf slot-attention model (e.g., SAVi) on multi-object held-out binding at matched compute; the bar is binding-specificity, not reconstruction quality.
- Minimum (Studio): slot module on frozen dense tokens, cached, hours. Studio: full multi-object battery, seed sweep. Wider-box: co-train slots with a small trainable backbone. Moonshot (avoid): large-scale object-centric pretraining.

## Arch C: Action-conditioned world model

- Proves: whether conditioning the predictor on ACTIONS turns the ex2 planning positive into a substrate-level capability (planning that improves as the world model improves), and whether action-conditioning is where a custom substrate finally beats a frozen one on the adaptation axis. Relates to C2 (adaptation bounds).
- Dataset: real action-conditioned video (agent acting, or a simulator with logged actions); this is the one architecture that is BLOCKED on an interactive environment adapter, not on compute (per STUDIO_MAXIMIZATION section on environments).
- Objective: predict next latent given current latent AND action (forward model), trained to minimize latent prediction error on true rollouts.
- Losses: action-conditioned latent-prediction loss, plus a rollout-consistency loss (multi-step prediction matches multi-step ground truth), plus anti-collapse.
- Trainable vs frozen: frozen V-JEPA encoder, TRAINABLE action-conditioned predictor (the b9_cerebellar_forward_model line); substrate frozen, forward model trained. Closest to doctrine of the world-model archs.
- Diagnostics that must pass: ex2's true-dynamics scoring (planner must beat flat reactive head AND action-shuffle), matched-compute vs unrolled reactive depth, multi-step rollout calibration (no compounding hallucination), seed-stability.
- Null: action-shuffle (predictor given permuted actions) ties the true-action model; if so, the model ignores actions and the "world model" is a video predictor.
- Better than existing open models how: must beat frozen V-JEPA 2's own action-conditioned variant (V-JEPA 2-AC exists) on planning-on-true-dynamics at matched compute; the bar is planning success, not prediction PSNR.
- Minimum (Studio): trainable forward model on cached latents from a logged-action dataset, hours. Studio: full ex2 battery on real actions. Wider-box: co-train forward model with a small trainable encoder on a live simulator. Moonshot (avoid): large-scale robotics-video pretraining.

## Arch D: Mixture-of-Thinking substrate model

- Proves: the central Stage 6 claim: that COMPOSING multiple substrates and thinking modes through a trained workspace beats any single substrate on a combined compositional-and-adaptation battery. This is the terminal architecture and the namesake of the review.
- Dataset: the union of the atlas datasets (real bound-attribute video, action-conditioned video, multi-object scenes).
- Objective: a routing/mixture objective, a trained workspace controller selects and composes frozen (and/or arch A/B/C) substrate features per input, trained end-to-end on the downstream battery with a load-balancing term so no expert is starved.
- Losses: downstream task loss (compositional held-out + continual BWT), a mixture load-balancing loss, a routing-sparsity term (each input uses few experts, the MoE/kWTA idea from e7_sparse lifted to substrate mixture), anti-collapse.
- Trainable vs frozen: substrates frozen (or the arch A/B/C pilots), TRAINABLE workspace router and heads. The mixture is the trained part; the experts can stay frozen, keeping it maximally doctrine-compliant.
- Diagnostics that must pass: beats best single substrate + workspace (Stage 3), beats the Stage 5 pilot alone, matched compute (the mixture must beat a single expert given equal total FLOPs, or the mixture is overhead), seed-stability, and cross-substrate convergence (does the mixture find the same routing across seeds).
- Null: a single best-expert-plus-workspace ties the full mixture at matched compute. If so, the mixture is pure overhead and should be descoped to the single expert.
- Better than existing open models how: must beat the best single frozen substrate AND the strongest open video-understanding model on the combined battery at documented compute; the bar is the COMBINED score (compositional + adaptation), which no single-objective open model targets.
- Minimum (Studio): 2-expert mixture (frozen V-JEPA + frozen DINOv2) with a trained router, cached latents, this is buildable NOW at Stage 3 scale as the workspace shell. Studio: 3 to 4 frozen experts, full battery. Wider-box: include arch A/B/C trained experts. Moonshot (avoid): many large trained experts, cluster-scale.

## Arch E: Developmental model

- Proves: the moldability doctrinal question, whether a substrate trained with a DEVELOPMENTAL curriculum (sensitive windows, staged unfreezing, curiosity-gated data ordering) exhibits the plasticity signatures (critical window d6, path-dependence y4, U-shaped overgeneralization d4) that no frozen substrate has shown. Unlocked only by a C3 failure at Stage 4.
- Dataset: a curriculum-ordered real-video stream, with a difficulty schedule and (for the sensitive-window test) a time-limited exposure to a target factor.
- Objective: JEPA prediction under a developmental schedule, staged plasticity (learning-rate/unfreezing schedule over "development"), curiosity-gated ordering with the noisy-TV guard wired in.
- Losses: JEPA prediction loss plus a plasticity-regularization schedule (the metaplasticity term that opens and closes learning), plus the noisy-TV-guarded curiosity signal for data ordering.
- Trainable vs frozen: a small trainable substrate with a SCHEDULED plasticity (this is the one architecture that genuinely needs an adapting substrate, since moldability is about the substrate itself); the C3-cheaper variant is a trainable adapter with scheduled plasticity on a frozen backbone.
- Diagnostics that must pass: a genuine sensitive window (target factor learnable during the window, NOT after, and this must be substrate-specific, d6's substrate_specific_window must flip to True), path-dependence beyond an optimizer artifact (y4 substrate-corrected area above zero), U-shaped overgeneralization (d4 non-flat), the noisy-TV guard, and matched-compute vs a non-scheduled baseline.
- Null: a non-developmental substrate trained on the same data with a flat schedule ties the developmental one on all plasticity signatures; if so, the "development" is cosmetic.
- Better than existing open models how: no open model targets developmental plasticity signatures, so the bar is INTERNAL: it must beat its own flat-schedule baseline and beat a frozen substrate on the moldability battery, and the signatures must be substrate-specific and seed-stable. This is the highest-risk, most-mystical-prone architecture and the diagnostics are correspondingly strict.
- Minimum (Studio): scheduled-plasticity adapter on frozen backbone, cached latents, testing the sensitive-window signal at studio scale, hours to a day. Studio: full developmental battery on real latents. Wider-box: train a small substrate from scratch with a developmental schedule. Moonshot (avoid): large-scale developmental pretraining.

## Arch F: Compressed capability-density model

- Proves: whether a SMALL custom substrate, trained to be capability-dense (distilled or compressed from the atlas), retains the substrate-is-special nuisance-invariance (+0.21 to +0.31) at a fraction of the parameters, i.e., whether the learned invariance is cheap to carry. Serves the "no frontier compute" doctrine by testing the minimum substrate that preserves the one valid positive.
- Dataset: the nuisance-heavy shape/attribute battery (substrate_vs_random_features content) scaled to real video, plus the compositional battery.
- Objective: distillation, match a frozen V-JEPA teacher's latents (or its nuisance-invariance behavior) with a much smaller student, plus a compression/rate penalty (the i-series MDL/rate-distortion line lifted to substrate scale).
- Losses: latent-distillation loss (student latents predict teacher latents), a rate/MDL penalty on student capacity, and the nuisance-invariance behavioral-matching loss (student must match teacher's +0.31 delta, not just its latents).
- Trainable vs frozen: trainable small student substrate, frozen V-JEPA teacher; this trains a substrate but a deliberately tiny one.
- Diagnostics that must pass: the student's nuisance-invariance delta (vs its own random-init control) must match the teacher's off-ceiling, at documented compression ratio, matched-compute vs an equally small random-init substrate, seed-stability, and a D3 certification that the regime is real.
- Null: an equally small random-init substrate, or a same-size student trained WITHOUT the invariance-matching loss, ties the distilled student on nuisance-invariance. If so, the compression bought no capability density.
- Better than existing open models how: must retain the teacher's nuisance-invariance delta at, say, 10x fewer parameters than V-JEPA-L, beating both a same-size random-init substrate and a same-size non-distilled student; the bar is invariance-per-parameter, a capability-density metric no open model reports.
- Minimum (Studio): distill a ViT-S student from frozen V-JEPA-L on cached teacher latents, cached-latent training, hours. Studio: full nuisance + compositional battery, seed sweep. Wider-box: distill on a large real-video corpus. Moonshot (avoid): distill a foundation-scale student.

---

## 9.x How this maps back to the gates

- Arch A and B are the Stage 5 pilots for a C1 (compositional-factoring) failure: A tries a new objective, B tries binding-before-pooling. B is preferred first because it keeps the substrate frozen.
- Arch C is the Stage 5 pilot for the adaptation/planning axis, blocked on an environment adapter, not compute.
- Arch D is Stage 6, licensed only after a Stage 5 pilot clears its bar and the atlas shows composition beats a single substrate. Its MINIMUM version (2 frozen experts + trained router) is buildable at Stage 3 (workspace shell) NOW and is the least-risky place to start if C2 routes there.
- Arch E is the Stage 5 developmental variant, licensed only by a C3 failure at Stage 4, and is the only architecture that genuinely requires an adapting substrate.
- Arch F is orthogonal: a capability-density check that can run once ANY substrate (frozen atlas or Stage 5 pilot) is chosen, to find the smallest substrate that preserves the one valid positive, serving the no-frontier-compute doctrine.

The through-line: every custom architecture must beat the CORRECT substrate control (random-init same-arch, never the full-rank latent projection that silently propped up the old "not special" reading), must win at matched compute, must be seed-stable, and must clear a D3-certified non-ceiling regime. Absent all four, the honest outcome is to keep the frozen encoder and never train a substrate. The pathway exists to be hard to walk, and the corpus as it stands has not yet cleared its first real gate.
