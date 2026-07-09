# EXPERIMENTS

GENERATED from registry/experiments.yaml by `python scripts/devel.py experiments --render` (do not hand-edit; edit the registry). Every row is a preregistration: a null, a headline metric, a falsifier, and a proof/FAILURE_TAXONOMY.md slot, committed before it runs.

195 catalogued: implemented=124, registry-only=69, deferred=2.

Tiers: exp_tier is the runnable Experiment.tier (cpu-now, gpu-later, env-later, 2.1-only); resource_tier is the planning class (cpu-now, studio-scale, environment-needed, weights-needed, moonshot). status is implemented / registry-only / deferred.

## Conducted bank (E1-E10)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| e10_openended | Minimal open-ended JEPA (CAPSTONE) | a solo agent plateaus; open-endedness is a population/env-generation property a solo agent cannot exhibit | env-later | deferred | 7 |
| e1_baseline | Baseline continual harness (GATE) | with shuffled task labels or matched capacity/difficulty, naive and protected show no retention gap; no forgetting means tasks too easy, no learning means target not decodable | cpu-now | implemented | 5 |
| e2_replay | Latent hippocampus (replay) | prioritized replay ties random replay, or replay ties no-replay (stream too short or latents not distinct) | cpu-now | implemented | 5 |
| e3_plasticity | Critical-period schedule (staged plasticity) | staged plasticity ties constant LR and tuned cosine decay (an LR trick, nothing to shape on a frozen substrate) | cpu-now | implemented | 2 |
| e4_neuromod | Neuromodulation (uncertainty gating) | point prediction-error gating chases the noisy-TV as much as ungated (conflates epistemic and aleatoric); needs a distributional signal | cpu-now | implemented | 8 |
| e5_curiosity | Curiosity as self-curriculum | prediction-error/RND curiosity equals random on a learnable-vs-noisy env; only learning-progress survives | env-later | implemented | 5 |
| e6_relational | Relational map over latents | structured head ties parameter-matched flat baseline; gain on 2.1 not larger than on 2 (pooled latent lacks object factorization) | 2.1-only | deferred | 3 |
| e7_sparse | Sparse / modular predictor | sparse/modular ties parameter-matched dense (the reduction was just capacity) | cpu-now | implemented | 4 |
| e8_dendritic | Dendritic predictor | dendritic block ties a matched MLP (the analogy adds complexity, no benefit) | cpu-now | implemented | 1 |
| e9_local | Local learning head | local rules fail to reach within margin of backprop and offer no memory/stability win | cpu-now | implemented | 9 |

## Bleeding-edge experiment bank (EX-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| ex10_cross_modal | Cross-modal / cross-domain binding (synthetic arm) | cross-modal prediction does not improve retention or abstraction; the modalities are not aligned enough to bind, or it is just a regularizer | cpu-now | implemented | 3 |
| ex11_causal_probing | Causal / interventional probing (do-operations on control families) | the shell cannot learn an interventional map beyond the observational one; it predicts seen interventions but fails unseen values | cpu-now | implemented | 10 |
| ex12_atlas | Representational atlas (what is decodable, by encoder scale) | probe accuracy does not exceed the shuffle-label floor for a target (not in the latent); bigger frozen perception does not raise decodability beyond seed spread | cpu-now | implemented | 3 |
| ex13_long_stream | Long-stream continual learning (the forgetting curve) | retention is flat in stream length within the seed spread, or every mechanism degrades identically (protection only delays the same collapse) | cpu-now | implemented | 8 |
| ex14_memory_bakeoff | Uncertainty-indexed and associative memory at scale (memory bake-off) | associative memory does not beat FIFO capacity and uncertainty indexing ties random eviction, even at scale | cpu-now | implemented | 4 |
| ex15_rejuvenation | Shrink-and-perturb rejuvenation against loss of plasticity | rejuvenation does not restore plasticity, or it restores plasticity at the cost of retention; the frozen-latent shell does not suffer loss of plasticity at this scale | cpu-now | implemented | 8 |
| ex16_codebook_sr | Discrete codebook / VQ abstraction and successor representation | the codebook adds no decodable structure over raw latents (random codebook ties it) and the SR provides no transfer over raw-latent features | cpu-now | implemented | 3 |
| ex17_latent_reasoning | Latent iterative reasoning (refinement + adaptive halting) | at matched compute, N-step refinement ties the depth-and-flop-matched single-pass head (iteration was just depth) | cpu-now | implemented | 4 |
| ex18_self_verification | Latent self-verification / self-correction | verify-revise ties single-shot at matched compute; the verifier carries no usable correction signal | cpu-now | implemented | 4 |
| ex1_generative_replay | Generative latent replay vs stored-buffer replay (dreaming) | generated replay does not match stored-buffer replay on BWT at matched budget; the distribution gap costs retention | cpu-now | implemented | 4 |
| ex2_latent_planning | Model-based planning in latent space (Dreamer/MuZero-style), synthetic arm | the learned dynamics does not enable planning the flat shell cannot do, or rollout error is too high to plan against | cpu-now | implemented | 7 |
| ex3_test_time_adaptation | Test-time training / adaptation on frozen latents | TTA does not beat the frozen head at matched parameters; the unlabeled proxy carries no adaptation signal, or it corrupts the base | cpu-now | implemented | 3 |
| ex4_fast_weights | Fast-weight / hypernetwork shells (in-context vs gradient plasticity) | in-context plasticity does not match gradient plasticity, or the hypernet collapses to a context-independent average shell | cpu-now | implemented | 4 |
| ex5_local_rules_scale | Local learning rules at scale (extend I4 to retention) | no local rule comes within the accuracy margin of backprop AND none offers a continual-learning or memory advantage that justifies the gap | cpu-now | implemented | 9 |
| ex6_active_inference | Active-inference / free-energy shell objective | the free-energy objective does not improve calibration or selection; the complexity term is a regularizer, or it chases the noisy-TV | cpu-now | implemented | 8 |
| ex7_meta_learning | Meta-learning across the task stream (MAML / Reptile) | the meta-learned init does not reduce adaptation steps vs a control init; the task family is too homogeneous or the shell too small | cpu-now | implemented | 5 |
| ex8_curiosity_bakeoff | Intrinsic-motivation curriculum bake-off (LP vs RND vs disagreement vs info-gain) | prediction-error and RND equal random on a learnable-vs-noisy budget; only learning-progress and disagreement survive; sharper, even LP ties uniform on a homogeneous stream | cpu-now | implemented | 10 |
| ex9_slot_attention | Object-centric / slot attention over pooled latents (E6 precursor) | slot attention over pooled latents ties the parameter-matched flat baseline; pooled features carry no per-slot structure to factor | cpu-now | implemented | 3 |

## Cross-cutting comparisons (I4) + information-theory experiments (I-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| i1_info_bottleneck | information_bottleneck_capability_per_bit | accuracy vs retained bits is flat down to a tiny width (the factor needs only a handful of bits) or the real curve is indistinguishable from a frozen-random projection of equal width | cpu-now | implemented | 3 |
| i2_mdl_selection | mdl_model_selection | MDL ranks shells differently from held-out loss (low rank correlation) or MDL is monotone in parameter count and adds nothing over counting weights | cpu-now | implemented | 4 |
| i3_compression_reasoning | compression_damages_reasoning | the two accuracy-vs-bits slopes are equal within seed spread (reasoning is not specially fragile) or the refiner has no advantage to lose even uncompressed | cpu-now | implemented | 4 |
| i4_backprop_alts | Backprop-alternatives comparison | no alternative comes within the stated accuracy margin of backprop at matched budget | cpu-now | implemented | 9 |
| i4i_redundancy_reduction | redundancy_reduction_retention | redundancy reduction ties raw-latent replay at matched bytes (the pooled latent redundancy is not costing retention) or the same gain appears on a frozen-random projection | cpu-now | implemented | 2 |
| i5_rate_distortion_replay | rate_distortion_replay | importance-weighted allocation lies on the same RD curve as uniform within seed spread (the importance signal does not predict which exemplars need precision) or the RD curve is a flat cliff | cpu-now | implemented | 4 |
| i6_mi_audit | mutual_information_audit | the exploitation ratio is near 1 for every decodable factor (the shell already uses essentially all available information) or the MI estimators disagree so much that the ratio is uninterpretable | cpu-now | implemented | 6 |
| i7_predictive_information | predictive_information_bottleneck | the predictive-only code does no better than the raw latent on rollout R2 at matched dimension (predictive and instantaneous information are entangled) or predictive information is near zero | cpu-now | implemented | 5 |
| i8_quant_robustness | quantization_robustness | the real encoder degradation curve overlaps the frozen-random curve (robustness is a generic high-dimension property, not a V-JEPA property, consistent with projection-invariant linear decodability) | cpu-now | implemented | 3 |
| i9_vq_rate_distortion | vq_rate_distortion_usability | VQ ties k-means and random codebook on usability per bit at matched rate (no decodable structure over raw latents, restating the codebook null in rate-distortion terms) or usability is flat in rate | cpu-now | implemented | 3 |

## Neuroscience + neuroscience-of-thought (N-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| n10_halting_difficulty | confidence_halting_tracks_difficulty | Halting is uncorrelated with difficulty (the halt head is noise), or adaptive halting loses accuracy with no compute saving. | cpu-now | implemented | 4 |
| n11_self_verification | latent_self_verification_error_monitoring | Verify-revise ties single-shot at matched compute (the verifier carries no usable correction signal). | cpu-now | implemented | 4 |
| n1_replay_ordering | replay_ordering_surprise_priority | Reverse and surprise-priority replay tie uniform rehearsal at matched count, or all replay ties no-replay (stream too short, latents too uniform). | cpu-now | implemented | 5 |
| n3_pc_refiner | predictive_coding_refiner_matched_compute | The PC-shaped refiner ties the generic refiner and ties the compute-matched untied MLP (iteration was just depth). | cpu-now | implemented | 9 |
| n4_tag_and_capture | synaptic_tag_and_capture_consolidation | Tagged-and-captured consolidation ties uniform consolidation (the schedule adds no benefit, or a simpler priority-by-confidence already captures it). | cpu-now | implemented | 2 |
| n5_fisher_reopen | fisher_triggered_plasticity_reopening | Triggered reopening ties tuned cosine decay (an LR trick), or the Fisher signal fails to localize the shift (estimator too weak). | cpu-now | implemented | 2 |
| n6_ach_ne_dissociation | ach_ne_expected_unexpected_uncertainty | The two-signal split ties a single disagreement scalar (the dissociation adds no benefit, or helps only combined). | cpu-now | implemented | 1 |
| n7_wm_delayed_match | working_memory_delayed_match_to_sample | WM slots tie a feedforward head with concatenated cue (the delay is trivial or the task does not need memory across it). | cpu-now | implemented | 5 |
| n8_object_permanence_bound | object_permanence_pooled_latent_bound | Occluded identity is not decodable from the pooled latent (substrate loses it), or the predictor ties the last-frame baseline. | cpu-now | implemented | 3 |
| n9_attractor_convergence | latent_refiner_attractor_convergence | The refiner drifts rather than converging (a compute story not a representational one), or converged accuracy ties the compute-matched untied MLP. | cpu-now | implemented | 9 |

## Reusable diagnostics (D) + developmental-psychology experiments (D-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| d1_fast_mapping | fast_mapping_mutual_exclusivity | 1-shot mutual-exclusivity-gated binding ties the nearest-centroid baseline on 1-shot accuracy and is no more interference-robust than a vanilla softmax head | cpu-now | implemented | 2 |
| d1_geometry | Representation geometry battery | the substrate geometry is indistinguishable from a frozen-random projection on these measures | cpu-now | implemented | 3 |
| d2_object_permanence_voe | object_permanence_voe_precursor | the occluded-object-present factor is not linearly decodable from pooled latents (the gate fails), or illegal-vanish prediction error equals legal-continue error | cpu-now | implemented | 3 |
| d2_substrate_ablation | Substrate-ablation control | the result is unchanged under a frozen-random or shuffled substrate (it did not need V-JEPA) | cpu-now | implemented | 3 |
| d3_blicket_causal | blicket_causal_intervention | interventional training does not beat observational-only on held-out cause attribution (the shell learns the observational map and confound, not the causal one) | cpu-now | implemented | 10 |
| d3_difficulty_calibration | Difficulty calibration | a tie is uninterpretable because no method separates in this regime | cpu-now | implemented | 6 |
| d4_transfer_matrix | Transfer matrix | off-diagonal transfer is at chance (no cross-task structure) | cpu-now | implemented | 5 |
| d4_ushaped_overgen | ushaped_overgeneralization_dip | exception-item accuracy is monotone (no over-regularization dip), or any dip is just capacity starvation removed by width alone | cpu-now | implemented | 5 |
| d5_compute_accounting | Compute accounting | a measured gain disappears once compute is matched (it was extra compute, not the mechanism) | cpu-now | implemented | 9 |
| d5_lp_self_curriculum | lp_self_curriculum_developmental_order | LP-ordering ties random ordering on final accuracy and LP entry order is uncorrelated with measured family difficulty | cpu-now | implemented | 1 |
| d6_rollout_gate | Rollout-predictability gate | rollout R2 is below the floor, so planning is not licensed | cpu-now | implemented | 7 |
| d6_sensitive_window | Sensitive / critical developmental window | there is no sensitive window; the same evidence presented early vs late produces the same final competence at matched total data, or any order effect also appears under frozen-random | cpu-now | implemented | 5 |
| d7_scaffolding | scaffolding_external_frontier_gating | scaffolding ties self-curriculum and unscaffolded training (external frontier gating adds no measured benefit over self-generated learning progress) | cpu-now | implemented | 8 |
| d8_imitation_conditioned_rollout | imitation_demonstration_conditioned_rollout | demonstration conditioning gives no rollout improvement on novel starts (a simpler unconditioned control ties it), or true imitation needs real action | cpu-now | implemented | 7 |
| d9_relation_transfer_gate | relation_transfer_gate_precursor | the relation factor is not decodable from pooled latents (taxonomy 3), or there is no transfer beyond surface features | cpu-now | implemented | 3 |

## Biology and evolution (B-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| b10_energy_budget | energy_budget_frontier | The energy penalty only moves accuracy and FLOPs along the same line a width sweep already traces (no special frontier), or sparsity collapses accuracy with no usable operating point at this shell scale | cpu-now | implemented | 2 |
| b1_clonal_selection | clonal_selection_eviction | At matched slots clonal-selection eviction ties PER and random on backward transfer within the seed spread, and the somatic-mutation arm does not beat the no-mutation arm | cpu-now | implemented | 2 |
| b2_baldwin_meta_init | baldwin_meta_init | The meta-init confers no retention advantage to a naive learner over a generic init (nothing assimilated, protection lives only in the online mechanism) | cpu-now | implemented | 5 |
| b3_stigmergic_curriculum | stigmergic_curriculum | The stigmergic field ties explicit LP selection (it is a reparameterization of the same bandit), and a raw-error deposit chases the noisy-TV unless the deposit uses LP | cpu-now | implemented | 2 |
| b4_homeostatic_scaling | homeostatic_scaling | Homeostatic scaling alone does not beat naive on backward transfer (a normalization trick not protection), and it adds nothing on top of EWC (the two protect the same directions) | cpu-now | implemented | 1 |
| b5_degeneracy_robustness | degeneracy_robustness | Degeneracy ties the matched-param single predictor and ties pure redundancy on both perturbation robustness and retention (any win was capacity removed by the matched-compute control) | cpu-now | implemented | 9 |
| b6_offline_consolidation | offline_consolidation | Offline interleaving ties online replay at matched replay budget (scheduling does not matter only sample count), and the generative dreaming arm fails to match stored replay | cpu-now | implemented | 2 |
| b7_developmental_curriculum | developmental_curriculum | Ordered curriculum ties shuffled at matched samples (order does not matter on a frozen pooled substrate, the stream is too uniform or short for staging to bite) | cpu-now | implemented | 5 |
| b8_structural_growth | Morphogenetic structural growth vs fixed final capacity | growth-as-process does not beat a fixed shell of the same final capacity; any gain is just the extra capacity, not the growth process | cpu-now | implemented | 4 |
| b9_cerebellar_forward_model | cerebellar_forward_model | The forward-model head ties the flat predictor at matched compute (the correction loop is just extra depth), or rollout R2 is below the floor so the model is not predictive enough to correct against | cpu-now | implemented | 9 |

## Philosophy operationalized (P-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| p10_theory_ladenness | theory_ladenness_of_perception | the nonlinear probe gain is identical on real and frozen-random substrates (the recovered structure is contributed by the probe, not the encoder; perception is theory-laden all the way down) | cpu-now | implemented | 3 |
| p1_concept_no_labels | concept_formation_without_labels | shell hidden states decode the held-out factor no better than the raw frozen latent, or the same purity appears under frozen-random (projection-invariant geometry, not concept formation) | cpu-now | implemented | 2 |
| p2_memorize_vs_concept | memorization_vs_concept_interpolation_extrapolation | the shell ties the nearest-neighbor memorization baseline on held-out values (it memorized, did not form a concept), or it fails extrapolation exactly where lookup fails (no rule learned) | cpu-now | implemented | 2 |
| p3_knowledge_is_prediction | knowledge_is_prediction_dissociation | prediction error and factor decodability are monotonically coupled at this substrate, so the knowledge-is-prediction identity is trivially supported and uninformative | cpu-now | implemented | 1 |
| p4_intelligence_is_compression | intelligence_is_compression_capability_per_bit | capability decays monotonically with compression (compression is pure fidelity loss, not abstraction); there is no regime where compressing improves generalization | cpu-now | implemented | 4 |
| p5_private_language | private_language_cross_seed_codes | codes are seed-idiosyncratic, cross-seed alignment is at the frozen-random level and cross-shell probes are at chance (the latent code is a private language, not a shareable scheme) | cpu-now | implemented | 9 |
| p6_meaning_without_symbols | meaning_without_symbols_continuous_vs_vq | the discrete-symbol shell ties the matched-capacity continuous shell (symbols add nothing), or it only ties because the random-codebook also ties (structure was the bottleneck, not symbols) | cpu-now | implemented | 1 |
| p9_thought_without_language | thought_without_language_reasoning_gain | at matched compute the refiner ties the untied-depth control (the gain was depth, not iteration, so there is no language-free reasoning signal, only more parameters) | cpu-now | implemented | 9 |

## Cognitive science (C-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| c1_held_out_combination | held_out_combination_gate | Held-out-combination accuracy collapses to chance while seen-combination accuracy stays high, or both factors are jointly undecodable so the test is uninterpretable | cpu-now | implemented | 3 |
| c2_latent_analogy | latent_analogy_structure_mapping | Offset vectors do not transfer across object contents so analogy retrieval ties the shuffled-offset control, or the transform factor is not decodable at all | cpu-now | implemented | 3 |
| c3_prototype_typicality | prototype_typicality_geometry | No typicality gradient (latents uniformly distributed within category, an exemplar bag), or a frozen-random projection shows the same cluster geometry so the structure is not V-JEPA specific | cpu-now | implemented | 2 |
| c4_event_segmentation | event_segmentation_chunking | Surprise boundaries do not align with true transitions better than a uniform-rate segmenter (pooled clip latents too coarse temporally), or the substrate gives no usable surprise signal | cpu-now | implemented | 5 |
| c5_transfer_matrix | transfer_matrix_analogical_mapping | Off-diagonal transfer is at chance (each family is its own surface code), or transfer is symmetric and content-driven not relation-driven | cpu-now | implemented | 5 |
| c6_dual_process_halting | dual_process_halting | Halting step is uncorrelated with difficulty (refiner halts on a content-irrelevant signal), or at matched average compute the adaptive policy ties the single-pass head | cpu-now | implemented | 4 |
| c7_chunking_expertise | chunking_as_expertise | Chunked input ties uniform-window input at matched capacity (chunking is a re-binning with no acquisition benefit), or the boundaries are not reliable enough to help | cpu-now | implemented | 1 |
| c8_concept_blending | concept_blending_interpolation | Midpoint blends are off-manifold and undecodable (the latent space is not convex for concepts), or frozen-random shows identical interpolation behavior | cpu-now | implemented | 3 |
| c9_systematicity_sweep | systematicity_stress_sweep | Accuracy cliffs immediately once any combination is held out (pure conjunction memorization), or the curve is identical to frozen-random so the substrate does not buy systematicity | cpu-now | implemented | 3 |

## Dynamical systems and cybernetics (Y-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| y1_fixed_point_convergence | fixed_point_convergence | trajectories do not settle (the update norm neither decays geometrically nor stabilizes) or the head loss at the fixed point is no better than at trained N within the margin | cpu-now | implemented | 9 |
| y2_basin_stability | basin_stability | the refiner shows no basin structure (sensitivity ratio near 1) or amplifies noise (sensitivity above 1) without improving perturbed-input accuracy over the raw-latent head | cpu-now | implemented | 2 |
| y3_seed_consistent_fixed_points | seed_consistent_fixed_points | fixed points are seed-dependent (cross-seed CKA at the frozen-random floor), the attractor is an artifact of initialization not imposed by task or substrate | cpu-now | implemented | 2 |
| y4_hysteresis | Hysteresis / phase transition in retention | retention is a single-valued function of the swept parameter (no hysteresis, no path dependence), or any loop also appears under frozen-random | cpu-now | implemented | 5 |
| y5_homeostatic_lr_loop | homeostatic_lr_loop | the closed-loop homeostat ties the best tuned open-loop schedule on frontier AUC, or it is unstable and underperforms, or on noisy-TV it chases aleatoric error | cpu-now | implemented | 2 |
| y6_free_energy_vs_lp | free_energy_vs_lp | free energy does not beat learning-progress on coverage or noisy-TV rejection and the epistemic term is rank-correlated with learning-progress near 1 (free energy is learning-progress relabeled) | cpu-now | implemented | 8 |
| y7_controllability_sysid_gate | controllability_sysid_gate | actions do not linearly move the pooled latent (B indistinguishable from the action-shuffle, controllability Gramian rank-deficient), so the pooled latent is not a controllable state and planning is not licensed | cpu-now | implemented | 3 |
| y8_recurrent_bibs_stability | recurrent_bibs_stability | the recurrent state diverges or saturates (no useful bounded dynamics) or when stable provides no forward-transfer gain over the stateless head at matched compute (the stream is too uniform to carry usable state) | cpu-now | implemented | 5 |
| y9_verifier_contraction | verifier_contraction | V does not decrease monotonically (the verifier carries no usable correction direction, the shuffled-verifier control ties it) or verify-revise ties plain EX17 and the compute-matched single-shot head (self-correction was just more depth) | cpu-now | implemented | 4 |

## Semiotics and universal latent language (S-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| s10_anti_self_deception | anti_self_deception_meta_test | an S-style decode test carries no genuine signal: its real score does not beat the shuffled chance floor (the valid certifier). A frozen-random tie is expected for a linear metric and is not evidence of vacuity. | cpu-now | implemented | 2 |
| s1_symbol_grounding | symbol_grounding_gate | codes carry no more world-variable information than a random codebook at matched k, and code-adjacency RSA ties the shuffled floor (an arbitrary label, not a grounded index) | cpu-now | implemented | 3 |
| s3_concept_arithmetic | latent_concept_arithmetic | offset arithmetic ties a random matched-norm offset, OR it survives equally under a frozen-random projection (a generic property of any linear space, not of V-JEPA concepts) | cpu-now | implemented | 9 |
| s4_latent_vs_discrete | latent_vs_discrete_reasoning | at matched compute, latent reasoning ties the discrete-bottleneck arm (symbol serialization is not a measurable bottleneck on this task) | cpu-now | implemented | 9 |
| s5_code_stability | private_language_stability | cross-seed code agreement ties the random / frozen-random floor (a per-run private idiolect, not a stable shared language) | cpu-now | implemented | 4 |
| s6_compositionality | compositionality_probe | held-out combinations are not decodable above the frozen-random floor (the pooled latent represents whole-scene gestalts, not factors compositionally; taxonomy 3) | cpu-now | implemented | 3 |
| s7_learned_vs_designed | learned_vs_designed_symbols | learned codes tie human-designed symbols at matched capacity (optimizing the vocabulary for use buys nothing over the designer ontology) | cpu-now | implemented | 1 |
| s9_interpretable_vs_useful | interpretable_vs_useful | usefulness and interpretability are uncorrelated (the useful code is not the namable code, an opaque-but-useful private code) | cpu-now | implemented | 9 |

## Reusable ablations (A) + perception and animal-cognition experiments (A-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| a1_affordance_decode | affordance_decodability_probe | action-relevance labels do not exceed the shuffle-label chance floor (the affordance contrast is not decodable from the pooled latent at all). Clearing the floor establishes decodability only, NOT encoder-specificity (a frozen-random LINEAR projection is invertible so a linear probe re-learns through it; a substrate-specificity claim needs a random-init encoder comparison that lives in the caching path). | cpu-now | implemented | 3 |
| a1_frozen_random_arm | Frozen-random substrate arm | the effect is unchanged under a frozen-random projection | cpu-now | implemented | 3 |
| a2_matched_compute_arm | Matched-compute arm | the gain disappears at matched compute | cpu-now | implemented | 9 |
| a2_viewpoint_invariance | viewpoint_motion_invariance | event identity does not transfer across viewpoint conditions beyond a pixel-difference baseline, OR the real encoder invariance ties frozen-random so invariance is trivial averaging not structure | cpu-now | implemented | 3 |
| a3_buffer_compression | Replay-buffer compression | low-bit replay ties full-precision replay (memory is cheap) or collapses (memory needs precision) | cpu-now | implemented | 4 |
| a3_what_where_when | episodic_what_where_when | integrated what-where-when retrieval does not beat independent single-feature retrieval intersected post hoc, OR the where and when tags are not decodable from the stored latent at all (taxonomy 3) | cpu-now | implemented | 4 |
| a4_cognitive_map | cognitive_map_latent_navigation | latents support only observed-transition prediction; held-out shortcut reachability ties a transition-frequency baseline, OR position is not decodable above floor (taxonomy 3) | cpu-now | implemented | 10 |
| a4_latent_robustness | Latent robustness | accuracy is flat under perturbation (the head ignores the latent) or collapses immediately (brittle) | cpu-now | implemented | 3 |
| a5_action_loop | perception_action_loop_necessity | action-conditioning does not reduce next-latent error beyond the action-blind predictor at matched compute, OR the gain survives action-shuffling so it was added capacity (taxonomy 9) | cpu-now | implemented | 9 |
| a6_object_permanence | object_permanence_persistence | mid-occlusion the latent carries no decodable trace of the hidden object beyond the empty frame (taxonomy 3), OR a single-frame baseline already separates the classes | cpu-now | implemented | 3 |
| a7_comm_channel | symbolic_communication_channel | the code transmits no more target information than a random code of the same size, OR target direction or distance is not decodable from the pooled latent at all (taxonomy 3) | cpu-now | implemented | 3 |
| a8_affordance_curiosity | affordance_driven_curiosity | affordance weighting ties pure learning-progress on coverage, OR it helps only because of the static prior, OR it re-chases the noisy-TV (taxonomy 8 or 2) | cpu-now | implemented | 8 |

## Form substrate, any-data interface, and post-V-JEPA experiments (F-series)

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| f10_intrinsic_form_curriculum | intrinsic_form_curriculum | learning-progress form selection ties uniform sampling or chases noisy forms, so curriculum is not form-aware | cpu-now | registry-only | 8 |
| f11_form_dream_replay | form_dream_replay | generated form replay underperforms stored replay at matched memory or drifts off the factor manifold | gpu-later | registry-only | 4 |
| f12_private_form_language_stability | private_form_language_stability | cross-seed code transfer sits at chance or below random-code agreement, so the form language is private | cpu-now | registry-only | 2 |
| f13_form_energy_budget | form_energy_budget | all form interfaces lie on the same density frontier as raw features, so form structure buys no efficiency | cpu-now | registry-only | 2 |
| f14_lifelong_form_expansion | lifelong_form_expansion | adding the new form causes old-form forgetting or fails to transfer, so the interface is not expandable | cpu-now | registry-only | 5 |
| f15_embodied_affordance_form | embodied_affordance_form | consequence-conditioned form tokens tie passive labels or action-shuffle, so affordance structure is not learned from intervention | env-later | registry-only | 7 |
| f16_perfect_slate_null | perfect_slate_null | the blank substrate ties frozen inherited features plus a larger shell, or ties random-init same-arch, so the perfect-slate story bought nothing at this scale | gpu-later | registry-only | 10 |
| f17_missing_form_recovery | missing_form_recovery | recovery ties the best remaining single form, or confidence does not predict correctness under a missing form, so the forms were redundant channels and the monitor is uninformative | cpu-now | implemented | 2 |
| f18_counterfactual_form_intervention | counterfactual_form_intervention | the intervention predictor leaks (predicts only seen intervention values) or ties the correlational predictor, so the matrix binds appearances rather than intervention structure | cpu-now | registry-only | 10 |
| f19_cross_scale_referent_binding | cross_scale_referent_binding | hierarchical referent memory ties flat clip memory or single-scale stores at matched bytes, so scale structure buys no retrieval and memory stays clip-shaped | cpu-now | registry-only | 3 |
| f1_form_alignment_gate | form_alignment_gate | paired referent alignment ties raw transfer or shuffled-anchor alignment, so the form interface is just a coordinate relabeling and not a usable cross-form bridge | cpu-now | implemented | 2 |
| f20_substrate_crisis_test | substrate_crisis_test | the crisis detector ties the raw error signal or triggers on aleatoric noise, so substrate insufficiency is not predictable from the exposed signals | cpu-now | registry-only | 4 |
| f2_heldout_form_transfer | heldout_form_transfer | multi-form training ties the single-form baseline on a held-out observation family, or the held-out family stays near chance after alignment | cpu-now | implemented | 2 |
| f3_form_bottleneck_capacity | form_bottleneck_capacity | the wide canonical bottleneck ties the small bottleneck, so interface width is not the bound, or both sit near the shuffled-label floor | cpu-now | implemented | 4 |
| f4_raw_payload_vs_form_tokens | raw_payload_vs_canonical_form_tokens | canonical form tokens tie raw flattened or handcrafted per-form features on every factor, so the form layer is ceremony | cpu-now | implemented | 2 |
| f5_cross_form_memory_binding | cross_form_memory_binding | cross-form retrieval ties per-form nearest neighbor or shuffled referents, so memory is form-local rather than referent-bound | cpu-now | implemented | 2 |
| f6_sensorimotor_form_closure | sensorimotor_form_closure | action-shuffle or action-blind controls tie true action-conditioned form closure, so the substrate is still a passive recorder | env-later | registry-only | 7 |
| f7_developmental_form_growth | developmental_form_growth | fixed-final-size or random-growth schedules tie developmental growth, so structural plasticity is cosmetic | gpu-later | registry-only | 1 |
| f8_plastic_substrate_rewrite | plastic_substrate_rewrite | a larger frozen-shell control ties the plastic substrate, so the gain is shell capacity rather than representation rewrite | gpu-later | registry-only | 9 |
| f9_cross_form_compositional_binding | cross_form_compositional_binding | held-out cross-form combinations collapse to chance while seen pairs remain high, so the system memorized form-specific conjunctions | cpu-now | implemented | 3 |

## AL

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_al1_uncertainty_router | uncertainty_router_with_noisy_tv_guard | the uncertainty router matches random episode selection on adaptation-per-update, OR it chases irreducible aleatoric noise (fails the noisy-TV guard), replicating the e4 negative in router clothing | cpu-now | registry-only | 8 |
| mop_al2_shared_latent_alignment | shared_latent_alignment_vs_random_map_floor | a random map of equal rank predicts the target as well as the learned map (alignment-artifact): any apparent cross-substrate agreement is the map's capacity, not shared structure | gpu-later | registry-only | 2 |
| mop_al3_audio_video_alignment | audio_video_temporal_alignment | audio-SSL and V-JEPA event boundaries co-locate no better than under a shuffled-time permutation: temporal-predictive structure is modality-specific, not universal | gpu-later | registry-only | 3 |

## AT

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_at1_nuisance_grid | cross_substrate_nuisance_grid | every substrate's decodability delta over its matched-resolution random-init control is within seed spread (random-control-artifact): pretraining bought no nuisance invariance beyond architecture and resolution in any substrate | gpu-later | registry-only | 3 |
| mop_at2_mode_substrate_dep | mode_substrate_dependence | every mode's advantage is reproduced by random-init-ViT (or random-encoder), so no mode depends on the pretrained substrate rather than architecture/resolution or generic separability | gpu-later | registry-only | 3 |
| mop_at3_time_axis | time_axis_ablation | with token/frame count equalized, full-clip and single-frame decodability are within seed spread for every factor: no factor genuinely requires the temporal axis | cpu-now | registry-only | 3 |
| mop_at4_programmatic_ceiling | programmatic_ceiling_reference | the programmatic reference does not exceed the perceptual substrates on any atlas factor (trivially separable everywhere), so no perceptual tie can be attributed to a substrate bound | cpu-now | registry-only | 5 |
| mop_at5_probe_class_sweep | probe_class_sweep | every cell's verdict is invariant to probe class, so probe choice never changes decodability and probe-specific is an empty verdict | cpu-now | registry-only | 4 |

## CM

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_cm10_action_forward_model | action_conditioned_forward_model | action-shuffle ties the true-action forward model on planning success: the model ignores actions and is a plain video predictor | env-later | registry-only | 7 |
| mop_cm11_developmental_plasticity | developmental_scheduled_plasticity | a non-developmental substrate trained on identical data with a flat schedule ties the developmental one on every plasticity signature (window, path-dependence, U-shape): the development is cosmetic | gpu-later | registry-only | 1 |
| mop_cm12_mop_substrate_capstone | mixture_of_perspectives_substrate_capstone | a single best-expert-plus-workspace ties the full mixture at matched total FLOPs: the mixture is pure overhead and should be descoped to the single expert | gpu-later | registry-only | 2 |
| mop_cm1_compositional_gate | compositional_bound_on_real_video_c1_gate | held-out-combination accuracy for real V-JEPA ties random-init same-arch ViT (delta < +0.05) on a D3-certified separable regime: the substrate memorizes conjunctions rather than factoring them | gpu-later | registry-only | 3 |
| mop_cm2_atlas_gate | multi_substrate_atlas_gate | all frozen substrates tie their own random-init controls on the failing gate (no substrate exceeds +0.15 off-ceiling where V-JEPA-L did not): no substrate swap helps | gpu-later | registry-only | 3 |
| mop_cm3_dense_vs_pooled | dense_vs_pooled_compositional | dense and coarse-grid tie full mean-pool on held-out combos (delta < +0.1 off-ceiling on a D3 regime): the pooling interface is not the bound (single-factor pooling already refuted, orientation decodes at 1.0 pooled) | gpu-later | registry-only | 3 |
| mop_cm4_workspace_shell | workspace_shell_routing | the workspace shell ties param-matched dense OR ties matched-compute unrolled depth on BWT and planning, or sign-flips across seeds: the win is capacity/depth, not routing | gpu-later | registry-only | 2 |
| mop_cm5_studio_rejuvenation | studio_scale_rejuvenation_c3 | plasticity loss never appears at reachable scale, OR rejuvenation ties doing-nothing, OR frozen-substrate restoration ties the adapting-substrate arm (no C3 signal) | gpu-later | registry-only | 1 |
| mop_cm6_distilled_density | compressed_capability_density | a same-size random-init substrate OR a same-size student trained without the invariance-matching loss ties the distilled student on the nuisance-invariance delta: compression bought no capability density | gpu-later | registry-only | 2 |
| mop_cm7_min_objective_probe | minimum_objective_encoder_probe | at matched tiny capacity, matched data, matched 256px, both custom objectives tie random-init same-arch AND tie each other; objective is not a lever at this scale and the +0.31 was scale/data/architecture/resolution. A tie is a strong negative closing the custom-encoder line | gpu-later | registry-only | 3 |
| mop_cm8_custom_jepa_pilot | custom_jepa_factoring_pilot | the custom-JEPA pilot ties both a random-init same-arch ViT and the best frozen atlas substrate on held-out combinations at matched compute: the custom objective bought nothing | gpu-later | registry-only | 3 |
| mop_cm9_slot_jepa_binding | object_centric_slot_jepa_binding | dense tokens without slots tie the slotted model on multi-object held-out combinations, OR the binding-specificity swap shows non-target slots change too: slots add nothing over dense | gpu-later | registry-only | 2 |

## DR

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_dr10_retrieve_reason | memory_first_retrieve_then_reason | retrieval ties from-scratch OR ties random retrieval: the frozen neighbor metric is not task-aligned | cpu-now | registry-only | 3 |
| mop_dr11_mc_rollouts | monte_carlo_latent_rollouts | MC ties the matched single longer rollout (samples collapse to the mean) OR wins only by exploiting irreducible aleatoric noise (fails noisy-TV) | cpu-now | registry-only | 8 |
| mop_dr12_disagreement | disagreement_as_uncertainty | disagreement ties single-head confidence AND gating on it ties uniform compute, OR it chases irreducible noise (fails noisy-TV, the e4 pattern) | cpu-now | registry-only | 8 |
| mop_dr13_horizon_limit | planning_horizon_limit | planning never beats reactive at any horizon (contradicts ex2) OR beats it only at horizon 1 (not really planning) | cpu-now | registry-only | 9 |
| mop_dr14_corruption | reasoning_under_corruption_compression | reasoning and single-pass degrade at the same rate under every corruption: iteration only processes what survives | gpu-later | registry-only | 3 |
| mop_dr15_modality_general | modality_general_reasoning | the gain is identical across all three modalities (a shell/task property) OR present only on the most separable modality (an artifact) | gpu-later | registry-only | 2 |
| mop_dr1_video_cache | real_bound_attribute_video_cache | even on real video the pooled and dense latents ceiling at 1.0 on held-out-combination decode (D3 says still trivially separable), so no compositional mode scores above the random-encoder floor | gpu-later | registry-only | 5 |
| mop_dr2_sparse_real | sparse_head_forgetting_on_real_latents | on real latents with a D3 certificate and a paired significance test, sparse ties param-matched dense (and ties a dense head with matched activation-sparsity penalty); the synthetic win was cluster separability | gpu-later | registry-only | 2 |
| mop_dr3_latent_scratchpad | latent_scratchpad_vs_residual_stream | scratchpad ties residual-only at all memory loads: the pooled frozen latent already discarded the detail external memory would store (a substrate bound) | gpu-later | registry-only | 3 |
| mop_dr4_causal_intervention | counterfactual_causal_latent_intervention | intervened rollouts do not beat the correlational baseline and single-factor interventions leak into other factors: the frozen latent entangles factors | gpu-later | registry-only | 10 |
| mop_dr5_cross_substrate_consistency | cross_substrate_reasoning_consistency | the reasoning gain is identical across all substrates (a shell/task property, not substrate-carried) OR vanishes on a second real encoder (fragile) | gpu-later | registry-only | 2 |
| mop_dr6_rollout_planning | internal_simulation_rollout_planning | planning ties reactive at matched compute OR ties action-shuffle: rollouts carry no usable future info beyond one step | cpu-now | registry-only | 9 |
| mop_dr7_latent_cot | latent_chain_of_thought | the latent chain ties one-shot at matched compute and shuffling does not hurt: the chain is a bag of extra layers (ex17 generalized) | gpu-later | registry-only | 9 |
| mop_dr8_fixed_point | recurrent_refinement_fixed_point_vs_drift | update norms do not decay geometrically and unrolling past N worsens loss: no fixed point, it is unrolled depth (the n9/y1 result) | cpu-now | registry-only | 9 |
| mop_dr9_verify_revise | verify_revise_under_corrected_controls | verify-revise ties single-shot at matched compute and the trained verifier ties the shuffled one: no correction-relevant signal (the ex18 result) | cpu-now | registry-only | 4 |

## MT

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_mt1_router_vs_best | router_beats_single_best_mode | routed density equals or trails its best constituent mode at matched compute, or the gap sits inside the seed spread | cpu-now | registry-only | 2 |
| mop_mt2_routing_vs_ensemble | routing_beats_uniform_ensemble | equal-weight blend matches the routed system's density at matched total compute; the router adds nothing over averaging | cpu-now | registry-only | 2 |
| mop_mt3_hetero_vs_homo | heterogeneous_beats_homogeneous_moe | homogeneous k-copy MoE matches the heterogeneous mixture at equal param/FLOP; diversity adds nothing beyond generic mixture capacity | cpu-now | registry-only | 2 |
| mop_mt4_reasoning_router | router_over_reasoning_primitives | at matched mean FLOPs the router ties or loses to the single best fixed mode; any win is bought compute or vanishes under seed sweep | gpu-later | registry-only | 2 |
| mop_mt5_adaptive_halting | adaptive_halting_vs_fixed_depth | at equal average FLOPs, adaptive halting ties fixed-depth: no exploitable hardness heterogeneity, or the halt head collapses to a constant | cpu-now | registry-only | 9 |
| mop_mt6_confidence_stop | confidence_stopping_vs_free_early_exit | the trained halt head ties the free update-norm rule; confidence is just the latent ceasing to move | cpu-now | registry-only | 9 |
| mop_mt7_beam_search | beam_tree_search_vs_greedy_refinement | at matched total FLOPs, search ties a deeper greedy chain because the scorer cannot outrank the refiner's own step | cpu-now | registry-only | 9 |
| mop_mt8_latent_debate | latent_debate_vs_single_module | debate ties the single matched module AND ties a plain ensemble; structured exchange is an unrolled ensemble/MoE | cpu-now | registry-only | 2 |

## PR

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_pr1_error_disjointness | mode_error_disjointness_router_precondition | per-sample errors are highly correlated (oracle-router gain over single-best is within seed spread), so no router can help and MP1/MP4 are not worth building | cpu-now | registry-only | 2 |
| mop_pr2_plasticity_substrates | plasticity_advantage_real_vs_random_encoder | shell adaptation speed and BWT on real V-JEPA are within the seed spread of the same shell on random-init-ViT (matched resolution): the substrate's structure does not ease trained-shell learning dynamics | cpu-now | registry-only | 3 |
| mop_pr3_modular_real | modular_plasticity_on_real_latents | on real latents with a D3 certificate and a paired significance test, sparse ties param-matched dense (and a matched activation-sparsity dense head): synthetic advantage was cluster separability | cpu-now | registry-only | 2 |
| mop_pr4_epistemic_gate | epistemic_gate_vs_noisy_tv | on the noisy-TV split the disagreement gate allocates no more LR-integral (or query budget) to the reducible partition than an ungated arm: replicates the e4 epistemic/aleatoric conflation | cpu-now | registry-only | 8 |
| mop_pr5_content_gated_cp | content_gated_critical_period | at matched LR-integral and shuffled task order, content-gated shows no retention or reopening advantage over cosine decay / tuned constant LR: pure LR annealing (the e3/d6 negative) | cpu-now | registry-only | 2 |
| mop_pr6_sleep_consolidation | offline_sleep_consolidation | offline replay-only ties online replay at matched total gradient steps: any sleep benefit is added compute, not phase separation | cpu-now | registry-only | 2 |
| mop_pr7_fast_slow | fast_slow_two_timescale_weights | fast weights tie a slow-only head (and a matched-size replay buffer) on within-task adaptation: the fast store is redundant capacity or a small cache | cpu-now | registry-only | 2 |
| mop_pr8_retrieval_head | memory_augmented_retrieval_head | retrieval-conditioned head ties plain kNN (parametric part adds nothing) OR ties the no-memory parametric head (memory adds nothing), at matched capacity | cpu-now | registry-only | 2 |

## WS

| id | name | null hypothesis | exp_tier | status | tax |
|---|---|---|---|---|---|
| mop_ws1_agreement_vs_confidence | agreement_vs_confidence_central_gate | agreement AUROC minus single-substrate-confidence AUROC <= 0 within seed CI; OR the win fails the invertible-remap vacuity guard (B is a linear remap of A), the shuffled-cross-source control, or noisy-TV | cpu-now | registry-only | 2 |
| mop_ws2_fusion_tournament | matched_capacity_fusion_tournament | at matched capacity every structured fusion ties the concat-MLP on held-out accuracy and NLL: structure buys nothing beyond param count | cpu-now | registry-only | 2 |
| mop_ws3_arbitration | uncertainty_vs_disagreement_arbitration | precision weights are uninformative so inverse-variance ties equal-weight averaging, and disagreement is uncorrelated with correctness so disagreement routing ties random routing at matched rate; OR disagreement fires on aleatoric noise (fails noisy-TV) | cpu-now | registry-only | 2 |
| mop_ws4_bandwidth_sweep | broadcast_bottleneck_bandwidth_sweep | the bottleneck's benefit is indistinguishable from generic regularization at matched capacity, and broadcast-back adds nothing over write-only: narrow-vs-wide is just a capacity effect | cpu-now | registry-only | 2 |
| mop_ws5_router_slot | modular_router_plus_shared_slot | the shared slot adds nothing over sparse routing alone: slot-ablation ties the full model on BWT and accuracy at matched capacity; sparse routing does all the work | cpu-now | registry-only | 2 |

## Negative-result taxonomy (every null maps to one)

See proof/FAILURE_TAXONOMY.md for the 10 categories. 1 biology-mapping adds nothing; 2 simpler control explains it; 3 frozen latent lacks/gains the factor; 4 capacity/estimator too weak; 5 stream too uniform/short; 6 tiny shell capacity bound; 7 needs embodiment/action (Tier R); 8 only helps combined; 9 representational-vs-compute claim separated; 10 conceptually beyond frozen-latent prediction.

## Diagnostics gate the experiments

linear-probe before any X-dependent mechanism; noisy-TV before a curiosity/uncertainty signal is trusted; calibration for probabilistic heads; Fisher trace for the critical-period signature; determinism before any cross-condition delta; the EX12 atlas + D1 geometry battery bound what every mechanism can use; D5 compute-accounting enforces matched compute for the reasoning experiments.
