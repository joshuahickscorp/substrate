# RESULTS_PRE_STUDIO.md

## 100 CPU-Now Experiments, Run for Real, on a Laptop

The first 93 (series N, D, B, P, C, I, Y, S, A, E, EX) are analyzed in full detail below, each adversarially re-checked. 7 more (series EX: EX1, EX4, EX6, EX7, EX11, EX14, EX18) were completed afterward and are covered in the addendum at the end. All 100 ran against a frozen V-JEPA 2 pooled-latent substrate (or, where noted, a synthetic Gaussian-cluster stand-in for that substrate). Every experiment declared an explicit null before running. Most of them held: an honest null is an asset, not a failure, because it tells us precisely what the frozen pooled substrate and the tiny trainable shell do not do. A separate adversarial verification pass then re-checked every candidate "positive" (every `null_supported=False` row) against the three standing controls the house doctrine requires (beat frozen-random, match compute, beat a tuned baseline) plus seed stability. Two of the flagged `seed_stability` failures (e4_neuromod, e7_sparse) were then genuinely re-run through the harness's real per-seed sweep, since the modules read a single `cfg.seed` rather than a list and a naive config override had been a no-op: e4_neuromod's negative finding came back robustly confirmed across 30 real runs, and e7_sparse's disqualifying objection did not survive contact with the real sweep, so it is now a **provisionally confirmed positive**, the one candidate in the corpus that has cleared its specific standing control. A later informal read flagged EX6 (from the addendum) as a second possible survivor, but a full three-pass adversarial check (two independent code re-analyses plus a numerical resimulation) refuted it decisively: the apparent "active inference" effect is a variance-magnitude artifact of a hardcoded hyperparameter, confirmed by reproducing the exact result with a trivial zero-learning heuristic and by an inversion test that flips the effect's direction. Every other candidate positive either never ran the relevant control, or ran it and it failed, or the "rejection" was a boolean-logic artifact of how the null was coded rather than a real effect in the intended direction.

This is not a bad outcome. It is what pre-registration and adversarial verification are for. The value delivered here is a clean map: which mechanisms are honestly dead at toy scale, which comparisons are underpowered by ceiling effects and need a harder task or more seeds, which land exactly where the doctrine predicts pooled latents should land (taxonomy-3, retarget to dense V-JEPA 2.1), which one candidate (e7_sparse) is worth prioritizing for a formal Studio-scale confirmation, and which of the other 24 "positives" are worth re-running with the missing control actually wired in, because the underlying mechanism still looks promising even though this run doesn't prove it.

## Summary Counts

| Category | Count | Notes |
|---|---|---|
| Total experiments run | 93 | Series N(11), D(8), B(9), P(8), C(9), I(9), Y(9), S(8), A(8), E(10), EX(5, of 5 covered in detail) — remaining EX rows summarized in series takeaways |
| Honest nulls held (`null_held`) | 51 | Mechanism tested cleanly against its control(s) and did not beat them; includes e4_neuromod after re-verification |
| Taxonomy-3 pooled-substrate bounds (`taxonomy3_pooled_bound`) | 9 | Mean-pooling provably erases the target factor; expected, citable, ships as-is |
| Ambiguous / gated / schema-artifact verdicts | 15 | Ceiling effects, failed resolvability gates, or AND/OR null-logic artifacts made the result uninterpretable either way |
| Candidate positives (`positive_delta_over_control` or `null_supported=False`) before adversarial review | 25 | Listed as "positive" in the raw run output |
| Candidate positives that survived adversarial verification | **1 (provisional)** | e7_sparse, after a genuine 30-run per-seed/per-axis sweep resolved its seed-stability objection; see CONFIRMED POSITIVES below |
| Seed-instability flags | 1 | s5_code_stability (explicit); e4_neuromod's and e7_sparse's seed-count gaps have now been closed with a real 30-run grid each |

93 does not sum exactly across the three verdict buckets above because a handful of experiments (e.g., b2_baldwin_meta_init, b6_offline_consolidation, b7_developmental_curriculum) are `ambiguous` for reasons distinct from either a clean null or a taxonomy-3 bound (underpowered gap, failed resolvability gate, or total ceiling saturation across all arms) and are counted once in the ambiguous row.

---

## Series N — Neuro-Inspired Mechanisms (Replay, Consolidation, Reopening, Dissociation, Attractors, Verification)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| n1_replay_ordering | null_held | BWT: no_replay=-0.269, uniform=0.0, reverse=-0.25, surprise=-0.238 | Replay beats no-replay clearly, but neither reverse nor surprise-priority beats plain uniform rehearsal. Ordering hypothesis honestly falsified. |
| n3_pc_refiner | null_held (underpowered) | pc_acc=generic_acc=matched_untied_acc=frozen_random_acc=1.0 | All arms including frozen-random saturate at 1.0; task too easy to separate PC-shaped refiner from controls. |
| n4_tag_and_capture | null_held | BWT: uniform=0.0, confidence=-0.269, tag_capture=-0.269 (identical) | Tag-and-capture collapses to plain confidence-priority; the capture window never triggers. |
| n5_fisher_reopen | null_held | post-shift AUC all=1.0; shift_localization_acc=0.5 (chance) | Ceiling on adaptation AND the Fisher trigger's localization is exactly chance. Both halves of the falsifier fail. |
| n6_ach_ne_dissociation | null_held | adapt_step: single=3.0, split=1.0; noise_rejection: single=0.292, split=0.104 | Split gate adapts faster but rejects noise much worse. Trade-off, not a win; falsifier requires both. |
| n7_wm_delayed_match | null_held | wm_acc=0.467 vs ff_acc=1.0; ff_shuffled_acc=1.0 (leak) | WM slots badly underperform, but FF-on-shuffled ceiling flags the task construction as degenerate. |
| n8_object_permanence_bound | REFUTED (see below) | occluded_probe_acc=0.9924 = frozen_random=0.9924 (exact tie) | Reported null_supported=False, but frozen-random ties the real substrate exactly. Artifact, not evidence. |
| n9_attractor_convergence | null_held | fixed_point_residual=18.46 (threshold 0.2); basin_stable_fraction=0.0 | Refiner drifts, does not converge; accuracy also ties compute-matched control. Clean double null. |
| n10_halting_difficulty | REFUTED (see below) | corr=0.236 vs random_halt=0.057; compute saved 66% | Clears its own margin against random-halt, but frozen-random control was never run; adversarial check found the gap likely survives any linear architecture, not just this one. |
| n11_self_verification | null_held (underpowered) | all arms incl. shuffled-verifier = 1.0; revise_fraction=0.0 | Verifier never fires on this toy task; ceiling, not a real absence of effect. |

**Series N takeaway:** six of ten mechanisms are clean, honest nulls at toy scale. Three (n3, n5, n11) are ceiling-saturated and underpowered rather than sharp negatives. n8 and n10 both looked like the strongest positives in the series before adversarial review; both failed the frozen-random control.

---

## Series D — Developmental Capacities (Fast Mapping, Object Permanence, Causal Learning, Curriculum)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| d1_fast_mapping | null_held | acc_gain=0.0 (both at ceiling); retention_advantage=0.0 | ME-gated binder ties nearest-centroid exactly. No measurable benefit. |
| d2_object_permanence_voe | taxonomy3_pooled_bound | occluded_present_decodability=0.45 vs chance=0.5 vs frozen_random=0.41 (gate needs >0.6) | Gate fails exactly as predicted: mean-pooling dilutes the occluded-object trace below the noise floor. Honest published limit. |
| d3_blicket_causal | null_held (underpowered) | held-out acc: obs=1.0, interventional=1.0 (tie) | Both arms ceiling; intervention adds nothing measurable, but task may be too easy to discriminate the regimes. |
| d4_ushaped_overgen | null_held | u_dip_depth=0.0 across all arms | No U-shaped dip appeared at all. Clean, unambiguous null. |
| d5_lp_self_curriculum | REFUTED (see below) | rank_corr=0.55 (clears threshold); lp_vs_random_gain=-0.036 (LP worse) | null_supported=False fires only because of an AND-gated null artifact; LP actually underperforms random ordering. |
| d7_scaffolding | REFUTED (see below) | scaf=0.778 vs self_lp=0.733 vs unscaf=0.733 vs random_mask=0.689 | Looked like a clean win over three controls; adversarial review found the gate (extra trainable params) was enabled only for the winning arms, confounding capacity with frontier information. |
| d8_imitation_conditioned_rollout | REFUTED (see below) | unconditioned_err=1.713 vs conditioned_err=0.861 (halved) | Large, clean-looking gain; adversarial review found no compute-matched control and a task construction that hands the answer key to the conditioned arm. |
| d9_relation_transfer_gate | taxonomy3_pooled_bound | relation_decodability=0.326 vs chance=0.333 vs frozen_random=0.296 (gate needs >0.433) | Gate fails exactly as predicted: mean-pooling a token pair cancels the relation-encoding difference. Honest published limit. |

**Series D takeaway:** D2 and D9 are the series' clean taxonomy-3 bounds, confirmed by chance-level decodability matching frozen-random. D7 and D8 looked like the series' strongest positives; both failed a standing control on adversarial review (matched-compute in both cases).

---

## Series B — Biology/Evolution Analogies

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| b1_clonal_selection | null_held | BWT: clonal=0.634 vs PER=0.631 vs random=0.684 (random wins) | Clonal-selection eviction ties or loses to simple baselines. |
| b2_baldwin_meta_init | ambiguous | naive_from_meta=0.956 vs from_generic=0.866 (gap 0.091, spread 0.134) | Suggestive 9-point gap does not clear the 2-seed spread. Underpowered, not a clean null. |
| b3_stigmergic_curriculum | REFUTED (see below) | stigmergic_coverage=0.546 vs LP_coverage=0.898 (LP wins by 35 points) | null_supported=False is a loss for the mechanism, not a win; also fails its own noisy-TV gate. |
| b4_homeostatic_scaling | null_held | homeostasis alone: -0.238 vs naive; on top of EWC: -0.281 | Actively harmful, not merely inert. |
| b5_degeneracy_robustness | REFUTED (see below) | BWT: degenerate=0.878 vs single=0.575 vs copies=0.688 | Retention gap survives matched-compute but no frozen-random control was ever run; adversarial review flags this as unconfirmed. |
| b6_offline_consolidation | ambiguous | all three arms = 1.0 exactly, spread=0.0 | Total ceiling effect; uninformative confirmation. |
| b7_developmental_curriculum | ambiguous | ordered=1.0 vs shuffled=0.588 vs hardfirst=0.531, but resolvability gate fails | Large gap undermined by the experiment's own difficulty-calibration gate failing. |
| b9_cerebellar_forward_model | null_held | flat=1.0 vs forward_model=0.675, using 1.78x more compute | Forward-model correction hurts despite more compute and a good rollout fit. |
| b10_energy_budget | null_held | width_sweep=2403.8 capability/FLOP vs energy=1701.6 | Plain width sweep dominates the sparsity frontier. |

**Series B takeaway:** every cleanly-evaluated biology analogy lost to its standing control. B5 (degeneracy) is the one candidate positive that survived matched-compute, but adversarial review flags the missing frozen-random arm as an open gap, not a confirmed result.

---

## Series P — Philosophy of Mind Probes (Concepts, Compression, Language, Symbols)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| p1_concept_no_labels | null_held | hidden_purity=raw_purity=0.760; frozen_random_purity=0.758 | Shell adds zero purity advantage; raw latent already separable by frozen-random. |
| p2_memorize_vs_concept | REFUTED (see below) | gen_gap: shell=0.108 vs NN=0.412; extrapolation ratio 0.69 | Shell beats memorization, but adversarial review found no frozen-random control and a task where the target is an exactly linear function, near-guaranteeing this outcome. |
| p3_knowledge_is_prediction | null_held (underpowered) | pred_decode_correlation=0.0; decodability saturated at all 4 capacity rungs | Ceiling from the smallest rung onward; identity untested in the interesting regime. |
| p4_intelligence_is_compression | null_held (underpowered) | accuracy flat at 1.0 for all bit depths 2-8 | No headroom for a non-monotone peak to appear. |
| p5_private_language | null_held | cross_seed_cka=0.985 vs frozen_random=0.896 (clears margin); cross_shell_probe_acc=0.264 (chance=0.25) | The decisive transfer gate sits at chance. Codes do not transfer between shells: private language, correctly held. |
| p6_meaning_without_symbols | null_held (underpowered) | continuous=vq=1.0; random_codebook=0.267 | VQ ties continuous exactly; ceiling limits how much room there was to detect a symbol advantage. |
| p9_thought_without_language | null_held | refiner_acc=control_acc=1.0; params 4x fewer with tying | Zero measured gain from iteration at matched compute; parameter-efficiency footnote only. |
| p10_theory_ladenness | null_held (underpowered) | nonlinear_gain real=frozen_random=0.0 everywhere | Both linear and nonlinear probes saturate identically; floor artifact, not demonstrated absence of theory-ladenness. |

**Series P takeaway:** every result in series P ran on a synthetic Gaussian-cluster proxy at only 2 seeds, and 6 of 8 hit metric ceilings. p2 looked like the one real positive; it did not survive adversarial review.

---

## Series C — Cognitive Science Probes (Composition, Analogy, Prototypes, Events, Transfer, Blending, Systematicity)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| c1_held_out_combination | REFUTED (see below) | heldout=0.838 vs frozen_random=0.871 (frozen-random WINS) | Literal null rejected by beating chance, but the code's own compositional_above_floor gate is False because frozen-random beats the real substrate. Artifact, not compositionality. |
| c2_latent_analogy | null_held (underpowered) | real=shuffled_offset=frozen_random=1.0 (all tied) | Ceiling; no differential transfer signal detectable. |
| c3_prototype_typicality | null_held | typicality_correlation=0.039 (near zero); silhouette real=0.629 vs frozen_random=0.630 | No typicality gradient; frozen-random reproduces cluster geometry exactly. |
| c4_event_segmentation | null_held | beats_uniform=false, beats_shuffled=true | Real structure is used (beats shuffled-latent) but ties the naive uniform-rate segmenter, which the gate requires beating. |
| c5_transfer_matrix | REFUTED (see below) | off_diagonal=0.711 vs frozen_random=0.699 (delta only +0.012, gate needs +0.05) | Beats chance but fails the frozen-random margin the experiment itself defines as meaningful. |
| c6_dual_process_halting | null_held (underpowered) | halts more on hard items, but matched-compute gain=0.0 (ceiling) | Halting correlates with difficulty, but the matched-compute comparison is uninformative at ceiling. |
| c7_chunking_expertise | null_held | chunked=uniform=2.5 steps; random_boundaries=1.5 (faster) | Chunking ties uniform and is slower than random boundaries. Opposite of predicted benefit. |
| c8_concept_blending | null_held | midpoint_decodability=1.0 = frozen_random=1.0 | Blends decodable, but identically so under frozen-random. Generic convex-space property. |
| c9_systematicity_sweep | null_held (underpowered) | real AUC slightly below frozen_random AUC (-0.028) | Graceful degradation, but not more systematic than a random linear projection. |

**Series C takeaway:** C1 and C5 both looked like rejected nulls but both fail their own frozen-random gate on inspection, exactly the artifact pattern the doctrine warns about.

---

## Series I — Information-Theoretic Probes (Bottleneck, MDL, Compression, Backprop Alternatives, Redundancy)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| i1_info_bottleneck | null_held (underpowered) | acc_real=1.0 at every width; acc_frozen_random=0.979 to 1.0 | A single retained dimension already saturates accuracy for both real and frozen-random. |
| i2_mdl_selection | null_held | mdl_vs_heldout_spearman=-0.8 (negative!); mdl_param_count_spearman=1.0 | MDL collapses to a parameter-count penalty; clean negative, not underpowered. |
| i3_compression_reasoning | null_held (underpowered/ambiguous) | both readouts pinned at 1.0 across all bit depths | No headroom to separate reasoning from classification fragility. |
| i4_backprop_alts | REFUTED (see below) | all 6 alternative rules within margin=0.03 of backprop ceiling | Task is trivially separable (verified: a zero-training nearest-centroid classifier alone scores 100%); the "tie" says nothing about the rules' real credit-assignment quality. |
| i4i_redundancy_reduction | null_held | BWT: raw=0.0 vs whitened=-0.3 (whitening hurts) | ZCA whitening actively hurts retention; clean, unambiguous negative. |
| i5_rate_distortion_replay | null_held (underpowered) | BWT flat at 0.0 for every arm and every rate; flat cliff | No forgetting signal exists in this toy setup to trace a rate-distortion frontier over. |
| i6_mi_audit | null_held (underpowered) | exploitation_ratio=0.977 at every capacity rung | Little headroom for shell-capacity to matter regardless of true effect. |
| i7_predictive_information | null_held (underpowered) | code_gain_vs_raw=-0.023; code_gain_vs_frozen_random=-0.006 (both negative) | Predictive-only bottleneck ties/loses to raw and frozen-random; little room for it to add value. |
| i8_quant_robustness | taxonomy3_pooled_bound | real and frozen_random degrade identically at every bit depth down to 2 bits | Robustness is projection-invariant, not V-JEPA-specific. Clearest, most directly interpretable taxonomy-3 result in the corpus. |
| i9_vq_rate_distortion | null_held | VQ ties k-means exactly (margin 0.0); random codebook nearly as good | Gradient-trained codebooks buy nothing over cheap alternatives. |

**Series I takeaway:** i8 is the cleanest taxonomy-3 result across the whole corpus and is worth carrying forward unchanged. i4 looked like a positive (local rules matching backprop) but the task was verified trivially separable even without training.

---

## Series Y — Dynamical-Systems Framing (Fixed Points, Basins, Homeostasis, Free Energy, Controllability, Recurrence, Verification)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| y1_fixed_point_convergence | null_held | converged_fraction=0.0; unrolling past training horizon worsens loss (0.0 to 3.40) | No beneficial fixed point; feedforward stack, not a contractive attractor. |
| y2_basin_stability | null_held | sensitivity=3.65, contraction_ratio=4.05 (amplification, not contraction) | Refiner amplifies noise and is worse than the raw-latent head under perturbation. |
| y3_seed_consistent_fixed_points | null_held | cross_seed_cka=0.621 BELOW frozen_random_floor=0.922 | Fixed points are seed-idiosyncratic, below even the random-projection floor. Strong, clean null. |
| y5_homeostatic_lr_loop | null_held | pi_minus_tuned=+0.013 (below the 0.02 margin) | Numerically ahead of tuned cosine decay but under the pre-declared margin. Honest tie, not inflated. |
| y6_free_energy_vs_lp | REFUTED (see below) | noisy_tv_rejection: efe=0.0 vs lp=1.0; coverage delta=-0.56 | null_supported=False is a boolean artifact of the AND-gated null; EFE loses on every substantive metric. |
| y7_controllability_sysid_gate | taxonomy3_pooled_bound (undecided) | R2=0.999, full-rank Gramian, but fit on a hand-built synthetic linear system, not real latents | Proves the diagnostic machinery works, not that real latents are controllable. |
| y8_recurrent_bibs_stability | null_held | bounded_fraction=0.0 (diverges); forward_transfer gain=0.0 at matched compute | Unbounded recurrent state, zero transfer gain even where it doesn't diverge. |
| y9_verifier_contraction | null_held | monotone_descent=1.0, but shuffled-verifier control ties the real verifier exactly | Descent is attributable to extra depth, not a usable error-correction signal. |

**Series Y takeaway:** every dynamical-systems mechanism tested (fixed points, basins, recurrence, verify-revise) failed to show beneficial attractor dynamics anywhere. This is a coherent negative result: iterative/recurrent depth on this shell behaves like plain feedforward depth.

---

## Series S — Semiotics / Universal Latent Language Probes

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| s1_symbol_grounding | REFUTED (see below) | mi_gain_vs_random=1.316 nats; but needs_real=False (frozen_random ties real exactly) | The experiment's own anti-artifact flag says this is vacuous; large-looking MI gain is a generic property of separable clusters. |
| s3_concept_arithmetic | taxonomy3_pooled_bound | gain_vs_random_offset=0.75, but gain_vs_frozen_random=0.0 (exact tie) | Analogy arithmetic is a generic linear-space property, not V-JEPA-specific geometry. |
| s4_latent_vs_discrete | null_held (underpowered) | acc_latent=1.0 vs acc_discrete=0.993 (gap inside margin) | Discrete bottleneck costs nothing measurable at matched compute; ceiling-limited. |
| s5_code_stability | REFUTED / seed_instability (see below) | code_agreement clears chance, but cross_seed_cka=0.519 vs frozen_random=0.949 (badly fails) | Learned codebooks are less cross-seed-consistent than random projections. Genuine instability, not a stable code. |
| s6_compositionality | null_held | heldout=1.0 vs frozen_random=0.978 (margin needs 0.1, got 0.022) | Apparent "perfect compositional generalization" fails the frozen-random margin. |
| s7_learned_vs_designed | null_held (underpowered) | learned=designed=1.0 (tied); both beat random_codebook=0.486 | Both vocabularies capture real structure but tie each other exactly. |
| s9_interpretable_vs_useful | null_held | spearman=0.065 (near zero); top-useful named LESS than random | Usefulness and interpretability decoupled, and reproduced by frozen-random. |
| s10_anti_self_deception | null_held (load-bearing) | all 4 sub-tests PASS-VACUOUS, delta_frozen_random=0.0 across the board | Retroactively explains why s1, s3, s6, s9 all show the same vacuity. The load-bearing result of the series. |

**Series S takeaway:** s10 is the load-bearing meta-result: it shows, by direct construction, that this toy substrate cannot currently distinguish a real V-JEPA-specific effect from generic linearly-separable-cluster geometry under any random projection. That single finding explains most of the series' apparent positives evaporating on adversarial review.

---

## Series A — Action, Affordance, and Spatial Cognition Probes

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| a1_affordance_decode | null_held | real=frozen_random=1.0 on all 3 contrasts | Projection-invariant capacity, not substrate-specific affordance structure. |
| a2_viewpoint_invariance | null_held | cross_viewpoint_transfer real=frozen_random=0.0136; pixel_baseline=0.111 (higher) | Near-zero and identical to frozen-random; both beaten by a trivial pixel-difference baseline. |
| a3_what_where_when | null_held | conjunctive=0.225 vs independent=0.188 vs recency_only=0.413 (recency wins) | Where/when tags ARE decodable, but conjunctive retrieval loses to a naive recency heuristic. |
| a4_cognitive_map | REFUTED (see below) | shortcut_acc: SR=0.92 vs transition_freq=0.775 | Registry requires a frozen-random arm; it was never implemented in code, and the "latent" is itself a random projection of ground-truth coordinates by construction. |
| a5_action_loop | REFUTED (see below) | conditioned_error=1.774 vs blind_error=1.812; shuffle collapses the gain | Matched-compute and shuffle-collapse both pass, but the registry-required frozen-random-substrate control was never implemented. |
| a6_object_permanence | ambiguous | temporal=frame_only=frozen_random=0.9924 (all identical) | Ceiling-effect confound from a synthetic leak parameter (permanence_leak=0.15), not a clean taxonomy-3 finding. |
| a7_comm_channel | REFUTED (see below) | learned code beats random code at every codebook size; shuffle collapses it | No frozen-random-projection control was run, and the synthetic latent is constructed directly from the target label, near-guaranteeing separability. |
| a8_affordance_curiosity | REFUTED (see below) | affordance_coverage=0.886 vs lp_only=0.795 vs prior_only=0.670 | Verified: substituting hardcoded hand-picked constants for the entire "affordance probe" reproduces the same gain. The result is an artifact of fixed rescaling, not affordance signal. |

**Series A takeaway:** four of eight experiments looked like clean positives after clearing their declared controls; all four failed adversarial review, mostly because the registry-required frozen-random arm was never implemented in code, not because it was implemented and failed.

---

## Series E — Continual Learning Mechanisms (Replay, Plasticity, Neuromodulation, Curiosity, Relational, Sparse, Local Rules, Dendritic, Open-Ended)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| e1_baseline | ambiguous | naive_bwt=-0.0038 (needed <=-0.15 for the gate to even engage) | The gate every downstream E-series result depends on did not pass; stream too easy to forget on. |
| e2_replay | null_held (underpowered) | all 4 arms: avg_accuracy=1.0, BWT=0.0 | No forgetting occurred for any scheme to prevent. Same underpowered-stream issue as e1. |
| e3_plasticity | null_held | staged/triggered beat neither constant LR nor tuned decay; Fisher peak at checkpoint 0 | No shapeable sensitivity window detected; honest bound. |
| e4_neuromod | UPDATED: robustly confirmed negative (see below) | disagreement gate_on_noise mean=20.07 (std=1.25, n=30, min=17.57 max=22.04); point_error mean=9.57 (std=2.75) | Re-run through the harness's real per-seed sweep (all 6 axis combos x 5 seeds, 30 genuine runs, not the single seed=0 the module's `cfg.seed` had silently defaulted to before): BOTH gates amplify error on noise in 30/30 runs, the wrong direction in every single config. This is no longer merely underpowered; it is a seed-and-axis-stable negative. |
| e5_curiosity | REFUTED / not a positive (see below) | pe_chases_noise=true, but lp_resists_noise=false too (0.797 noise fraction) | Reframing: this is a failed replication of the intended mechanism, not a discovered effect; only single-seed, campaign leg never actually run. |
| e6_relational | taxonomy3_pooled_bound (undecided) | both pooled and dense arms fell back to frozen_random weights (unavailable), both heads scored exactly 0.0 | Total degeneracy this session; no claim possible until real 2.1 weights load. |
| e7_sparse | UPDATED: seed_stability objection resolved, strongest surviving candidate (see below) | BWT gain over dense across the real 30-run grid: moe mean=0.139 (std=0.093, min=-0.000, max=0.260), kwta mean=0.180 (std=0.111, min=-0.010, max=0.322); clears the 0.02 margin in 27/30 (moe) and 26/30 (kwta) runs; regime_calibration.reference_score=1.0 (chance=0.167), confirming the stream itself carries real, decodable class structure | The single-seed run's "seed_stability" refutation does not survive contact with the real per-seed sweep (the module reads `cfg.seed`, not `experiment.seeds`; overriding the latter is a no-op, so a genuine sweep must go through the harness's `run_sweep`). Direction is robust across nearly every axis x seed combination; magnitude varies and a handful of configs land near zero. `d3_difficulty_calibration` was wired directly into the module this session (a known-separable reference decodes the stream far above chance), so the BWT gap is not an artifact of an uncalibrated regime. This is now the single strongest candidate in the entire 100-experiment corpus and the top Studio-priority item: a formal significance test on real (not synthetic) latents would settle it. |
| e8_dendritic | null_held | 5 seeds; dendritic acc=0.952 vs mlp=1.0; dendritic needed 2.65x more steps | Clean, seed-stable null across a real 5-seed sweep. |
| e9_local | REFUTED (see below) | forward_forward and target_prop within margin of backprop, but no frozen-random arm, and backprop ceiling is itself weak (4-epoch toy task, separation=0.7) | Stitched-together positive: no single rule is both accuracy-competitive and a memory winner. |
| e10_openended | ambiguous | diversity ladder: 23, 27, 37, 25 (regressed on the fullest-component agent) | Scaffold-only, env-later stub; regression is real for this toy implementation but says little about a real open-ended setup. |

**Series E takeaway:** e8_dendritic and (after the real sweep) e4_neuromod are the series' clean, seed-stable nulls. e7_sparse is the corpus's one genuinely promising surviving lead: its seed-instability objection dissolves under a real 30-run grid, sparse consistently beats dense in direction even though magnitude varies, and it deserves a proper significance test on real latents before the Studio, not more toy tuning.

---

## Series EX — Extended Probes (Test-Time Adaptation, Curiosity Bake-off, Atlas, Codebook, Latent Reasoning)

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| ex3_test_time_adaptation | null_held | shift_acc_tta=0.370 vs frozen=0.403 (TTA slightly worse) | Label-free entropy-minimization TTA carries no usable adaptation signal at this scale. |
| ex8_curiosity_bakeoff | REFUTED (see below) | learning_progress rejects noise (-0.0002 on noise side), but disagreement does NOT (opposite of predicted) | Internal inconsistency: the claimed two-signal pattern is actually one thin, near-zero-margin signal; no frozen-random ablation was run. |
| ex12_atlas | null_held | identity probe clears shuffle floor; random-label control stays at floor | Self-check passes exactly as designed; a healthy sanity gate, not a substantive discovery. |
| ex16_codebook_sr | REFUTED (see below) | cluster_purity gain=0.634 vs random codebook, but code_probe_acc ties raw_latent_probe_acc exactly (gain=0.0) | Purity win restates information the raw latent already carries; ceiling-bound, not new abstraction. |
| ex17_latent_reasoning | null_held | refiner_acc=control_acc=1.0 at matched FLOPs; refiner uses 4x fewer params | Iteration bought nothing beyond depth; parameter-efficiency footnote only. |

**Series EX takeaway:** three of five are clean, healthy negative-control results (ex3, ex12, ex17). ex8 and ex16 both show a split pattern where the null is only partly rejected and neither survives as a clean positive.

---

## CONFIRMED POSITIVES

**UPDATE (three live contentions closed with decisive controls):** the picture is now sharper than "one provisional positive." Three targeted control experiments were run (scripts `close_e7_sparse.py`, `close_ex5_local_rules.py`, `close_ex2_planning.py`; results in `runs/pre_studio/close_*.json`):

1. **e7_sparse: REAL but REFRAMED, not substrate-specific.** The sparse-minus-dense backward-transfer advantage survives a frozen-random projection of the inputs nearly intact (real gain +0.239, frozen-random gain +0.164, ratio 0.69, both clear the null margin at all 5 seeds). So sparse/gated heads genuinely forget less, but it is a GENERIC ARCHITECTURAL fact about the shell head, not a property of the real V-JEPA substrate geometry. Still the most useful positive (structure matters), now correctly characterized.

2. **ex5_local_rules_scale: REFUTED (Adam artifact).** Plain-SGD backprop at a step size matched to the local rules' effective per-step update magnitude forgets far less than Adam backprop (+0.157 BWT) and now ties or beats both local rules on accuracy and backward transfer. The "local rules beat backprop" finding was an Adam optimizer artifact (adaptive normalization plus cross-task momentum state), not a property of local credit assignment. The one plausible-but-unverified lead is now a negative.

3. **ex2_latent_planning: SURVIVES honest grading (a genuine positive).** Scored on the TRUE synthetic dynamics (the planner's selected action sequence executed through the real environment, graded identically to the flat baseline and the shuffle control), the MPC planner STILL beats both on terminal distance to goal on all 3 seeds (planner 5.67 vs flat 6.98 vs shuffle 6.97, positive every seed). The in-belief optimism the adversarial review flagged is real (0.83 gap between belief and true distance) but was not the whole story: the planner wins by roughly 26x the margin even when graded honestly. Short-horizon latent MPC genuinely reaches goals a reactive head cannot on this synthetic action-conditioned family.

**Net: two real positives, both about the shell's structure/algorithm, neither about the frozen substrate carrying special structure.** e7_sparse (sparse architecture forgets less, substrate-agnostic) and ex2 (latent planning reaches goals). This is a consistent signal: what works is mechanism/architecture in the shell, not the pooled V-JEPA features doing special work. The dense-vs-pooled probe (running) tests whether the SUBSTRATE side of that story is a pooling artifact or a real bound.

---

The original provisional framing, retained for the record: of the 25 original candidate positives, e7_sparse survived its seed-instability objection via the harness's real per-seed sweep (sparse beats dense across 27-30 of 30 seed x axis combinations). The other 24 failure modes break down as follows:

| Failed control | Count | ids |
|---|---|---|
| frozen_random (control never run, or run and tied/beat the real substrate) | 17 | a4_cognitive_map, a5_action_loop, a7_comm_channel, a8_affordance_curiosity, b3_stigmergic_curriculum, b5_degeneracy_robustness, c1_held_out_combination, c5_transfer_matrix, d5_lp_self_curriculum, e9_local, ex16_codebook_sr, ex8_curiosity_bakeoff, i4_backprop_alts, n10_halting_difficulty, n8_object_permanence_bound, p2_memorize_vs_concept, s1_symbol_grounding, s5_code_stability, y7_controllability_sysid_gate |
| matched_compute | 2 | d7_scaffolding, d8_imitation_conditioned_rollout |
| seed_stability, RESOLVED (see above) | 1 | e4_neuromod (30-run grid confirms the negative finding, does not become a positive) |
| seed_stability, OBJECTION DOES NOT HOLD (see above) | 1 | e7_sparse (promoted to provisional confirmed positive) |
| tuned_baseline | 1 | y6_free_energy_vs_lp |
| other (mechanism itself failed on inspection) | 1 | e5_curiosity |

Every "REFUTED" entry in the series tables above (except e7_sparse, now provisional) is one of these 25. None is being discarded as worthless: each has a specific, actionable studio_priority (mostly `needs_real_latents` or `rerun_at_scale`) recorded in the handoff document, because the underlying mechanism in several cases (a4, a5, a7, b5, d7, d8, p2) still looks directionally interesting once the missing control is actually wired in and run on real latents with proper seed coverage. What can be said honestly today is: **one candidate (e7_sparse) survives its specific objection at toy scale and deserves priority confirmation on the Studio; no other candidate positive in this corpus has cleared its required standing control.**

---

## HONEST NULLS / TAXONOMY-3 BOUNDS (the citable results)

These are the results that can be cited today as genuine, control-passing findings from the frozen pooled V-JEPA 2 substrate and its tiny trainable shell, at toy/synthetic scale:

**Clean, well-controlled negatives** (mechanism tested against a real control and lost, not merely tied at ceiling):
- **i2_mdl_selection**: MDL selection collapses to a parameter-count penalty (spearman 1.0 with param count, -0.8 with held-out loss). Retire this MDL formulation.
- **i4i_redundancy_reduction**: ZCA whitening before replay actively hurts backward transfer (-0.3 vs 0.0 raw). Decorrelating pooled latents before replay is harmful, not neutral.
- **b4_homeostatic_scaling**: Synaptic-scaling rescaling destroys retention both alone and stacked on EWC.
- **b9_cerebellar_forward_model**: Forward-model correction hurts accuracy despite more compute and a good rollout fit.
- **b10_energy_budget**: A plain width sweep dominates the L1-activation-energy frontier.
- **y1/y2/y3/y8/y9 (series Y)**: the dynamical-systems framing of the shell (fixed points, basins, recurrence, verify-revise) coherently fails everywhere it is tested. This reads as one negative result, not five flukes.
- **c7_chunking_expertise**: Chunked summaries are no faster to competence than uniform windows, and slower than random boundaries.
- **e8_dendritic**: The one 5-seed-verified null in the corpus. Dendritic head adds complexity with no benefit.

**Taxonomy-3 pooled-substrate bounds** (mean-pooling provably erases the target factor; this is the doctrine's predicted, ship-as-is outcome for object/binding/relation tests on a pooled substrate):
- **d2_object_permanence_voe**: occluded-object trace diluted to chance, matches frozen-random exactly.
- **d9_relation_transfer_gate**: pairwise relation difference cancelled by mean-pooling, matches frozen-random exactly.
- **i8_quant_robustness**: the cleanest, most directly interpretable taxonomy-3 result in the whole corpus — real and frozen-random degrade identically under quantization down to 2 bits, confirming pooled linear decodability is projection-invariant by construction.
- **s3_concept_arithmetic, s6_compositionality**: apparent wins against naive controls evaporate exactly at the frozen-random margin.
- **s10_anti_self_deception**: the load-bearing meta-result — all four semiotics decode tests are reproduced exactly by a frozen-random projection.

**The fix for every taxonomy-3 bound above is dense V-JEPA 2.1, not more toy tuning at this scale.**

---

## What a Positive Here Would and Would Not Prove

If a future re-run genuinely clears frozen-random, matched-compute, and a tuned baseline, with seed-stable results (5+ seeds) on this synthetic Gaussian-cluster substrate, that would prove the proposed mechanism has some effect on a small trainable shell layered over an easily-separable toy task. It would **not** prove the mechanism does anything on the real frozen V-JEPA 2 pooled substrate, because most of the "latents" in this corpus are synthetic stand-ins (`make_task_stream` Gaussian clusters), not cached real pooled features, and many of the toy tasks are constructed to be close to linearly separable by design, which is exactly the condition under which a frozen-random projection can trivially match a trained representation. A positive on real cached V-JEPA 2 latents, at a task difficulty where frozen-random and raw-latent probes are NOT already at ceiling, is the actual bar for a claim about the substrate. Nothing in this batch has cleared that bar yet, and that is the honest, useful state to hand off to the Studio.

---

## ADDENDUM 2: The Studio-Gated Rows That Turned Out To Be Implementable

Of the 9 originally Studio-gated registry rows, 5 were blocked on unwritten code (or on a live-environment arm separable from a synthetic precursor) rather than genuine hardware impossibility: ex13_long_stream, ex5_local_rules_scale, then a second batch (ex15_rejuvenation, ex9_slot_attention, ex2_latent_planning synthetic arm) after the user asked to migrate everything implementable off the Studio list. All were implemented for real this session and run at TWO scales: the shipped "scaled" default config (real minutes on a CPU laptop) and a larger "grind" override (pushed further to genuinely exercise the machine, some via a long sequential background run). Registry rows flipped to `implemented`, `resource_tier: cpu-now`. Only 3 rows now remain genuinely blocked (e6_relational and ex10_cross_modal need real weights that do not exist yet; e10_openended needs multi-agent/environment infrastructure).

### ex13_long_stream (the forgetting curve)

| scale | naive final anchor acc | protected final anchor acc | frozen_random final anchor acc | divergence (protected minus naive) | survives frozen-random |
|---|---|---|---|---|---|
| shipped (240 tasks, dim 48) | 0.444 (threshold crossed at task 25) | 0.722 (never crosses) | 0.667 (never crosses) | +0.250 | **No** |
| grind (3000 tasks, dim 96, hidden 128) | ran in 50.7s | | | +0.528 | **No** |

At both scales, naive-sequential training clearly forgets (accuracy collapses well below the retention threshold within the first few dozen tasks) while replay+EWC protection clearly retains (never crosses the threshold, even after 3000 tasks). That divergence is real and, at the larger scale, gets substantially LARGER (0.25 to 0.53), not smaller, which is itself informative: the protection gap does not shrink with stream length in this toy regime, it is a durable effect. But it does **not** clearly survive the frozen-random-substrate control: the frozen-random arm retains almost as well as the protected arm (0.667 vs 0.722 at shipped scale), both far above naive's collapse. One honest caveat on the control's fairness: `n_tasks_control` (80 at shipped scale, 1000 at grind scale) is smaller than the main arms' `n_tasks` (240 / 3000), so the frozen-random arm faces a shorter stream than naive/protected, not a perfectly length-matched comparison; a tighter version of this control would run all three arms at identical length. `null_supported=True` at both scales (an honest, unforced result): replay+EWC clearly beats naive, but that specific advantage has not been shown to require the real substrate at this toy scale, so it should not yet be reported as a substrate-specific finding.

### ex5_local_rules_scale (local rules at scale, extended to retention)

| scale | backprop acc / BWT | feedback_alignment acc / BWT | predictive_coding acc / BWT | rules with BWT advantage | null_supported |
|---|---|---|---|---|---|
| shipped (80 tasks, 3 widths, 3 seeds, 112s) | 0.336 / -0.723 | 0.497 / advantage | 0.398 / advantage | FA, PC (both) | **False** |
| grind (300 tasks, 4 widths, 3 seeds, ~25 min) | 0.284 / -0.604 | 0.404 / -0.640 | 0.404 / -0.649 | **none** | still False (accuracy only) |

At the shipped scale this looked like an unforced, genuinely surprising positive: both feedback_alignment (fixed random backward matrix, no weight transport) and predictive_coding (local energy descent) beat backprop on BOTH final accuracy and backward transfer, ranking stable across a depth sweep.

**But the retention half of that finding did NOT survive scaling up.** Run again at nearly 4x the stream length (300 tasks vs 80) and a wider depth sweep, the local rules' backward-transfer advantage vanished entirely: at grind scale FA and PC both have SLIGHTLY WORSE BWT than backprop (-0.640 and -0.649 vs backprop's -0.604), so `rules_with_bwt_advantage` came back empty. The accuracy advantage persisted (FA/PC at 0.404 vs backprop 0.284), which is why null_supported stays False, but the specific, doctrine-interesting "local rules resist catastrophic forgetting better than backprop" claim is now REFUTED by its own larger-scale rerun.

This is exactly what the adversarial verifier predicted and is worth stating plainly: the shipped-scale "retention advantage" was a fragile, scale-dependent artifact, most likely of backprop's Adam optimizer (adaptive, momentum, state compounding across task boundaries) versus the local rules' plain delta updates at a matched nominal but unmatched EFFECTIVE learning rate. At a longer stream, backprop's own forgetting is no longer categorically worse. The remaining honest, non-artifactual finding is narrower: on this synthetic domain-incremental stream, the two local rules reach higher final accuracy than backprop (feedback_alignment consistently highest), which is itself interesting but is NOT the "local credit assignment is more retention-stable than global backprop" claim the shipped scale appeared to make. The still-open Studio check (backprop with plain SGD or matched-weight-decay) would isolate whether even the accuracy gap is an optimizer artifact.

### ex15_rejuvenation (shrink-and-perturb against loss of plasticity)

Built directly on ex13_long_stream's harness: three arms all on the replay+EWC-protected recipe, comparing no-rejuvenation vs shrink-and-perturb rejuvenation (every K tasks: theta <- shrink*theta + (1-shrink)*theta_init + noise, optimizer/buffer/EWC state untouched) vs the same rejuvenation on a frozen-random substrate. Tracks effective_rank, dead_unit_count, and retained_accuracy through the stream. At the shipped scale (240 tasks, dim 48, hidden 64) `null_supported=True` via the honest "nothing to restore" clause: the small frozen-latent shell simply does not exhibit loss of plasticity at that scale (no dead units appear, effective rank barely moves), so there is nothing for rejuvenation to fix. This is not a coding artifact: mid-scale probes (1000-3000 tasks at dim 96 / hidden 128) do surface real plasticity loss, and rejuvenation does then measurably restore effective rank, sometimes at a retention cost (0.083 at 1000 tasks, 0.389 at 3000) and sometimes not. So the mechanism is real once the substrate is pushed hard enough; the honest cpu-now finding is that a small frozen-latent shell needs substantial scale before loss of plasticity even appears.

**Big-run grind result (45000 tasks, dim 256, hidden 256, ran 41 min): `null_supported=False`, but weak and most likely the null's own "restores at a retention cost" clause.** At this scale loss of plasticity IS observed (plasticity_loss_observed=True) and rejuvenation raises effective rank (rank_restored=True: 5.98 vs the no-rejuvenation arm's 5.40), which flips the boolean because the module's null requires restoration WITHOUT a retention cost. But the honest numbers undercut the flip three ways: (1) both effective ranks are tiny, about 5.7 of 256 hidden units (~2%), so "restoration" is a marginal move on an already-collapsed representation; (2) the rejuvenated arm's final anchor accuracy actually DROPPED, 0.361 vs the no-rejuvenation arm's 0.417, a 5.5-point retention cost that the module's own threshold happened to score as retention_cost_paid=False; and (3) substrate_specific=False, the frozen-random arm rejuvenates similarly, so this is not a property of the real substrate. Read straight, this is essentially the null's predicted "rejuvenation restores a little plasticity at the cost of retention" outcome, dressed up as a boolean rejection by a threshold judgment on whether a 5.5-point accuracy drop counts as a cost. It should get the same adversarial verification as ex2 and ex5 (examine the retention-cost threshold, add seeds) before being cited as a rejuvenation positive; provisionally it is a weak, non-substrate-specific effect, not a moldability win. See `runs/pre_studio/ex15_rejuvenation_grind.json`.

### ex9_slot_attention (object-centric slots over pooled latents, E6 precursor)

K learned slot queries cross-attending over a window of synthetic pooled per-clip latents on a relation-change task, versus a parameter-matched flat-pooled MLP, plus a frozen-random control. The pooled per-clip latent is by construction the NOISY SUM of both entities' positions (no per-object channel), faithful to what a real pooled V-JEPA encoder outputs for a clip containing multiple entities. Shipped-scale result: `null_supported=True` — slot attention ties (does not beat) the parameter-matched flat baseline. This is the EXPECTED taxonomy-3-style outcome and a clean, citable substrate bound: pooled per-clip latents carry no factorizable per-slot structure for slots to recover, because pooling already summed the entities together before the shell ever sees them. A slot-count sweep (4/8/16/32 slots, up to 40000 windows each) was launched to confirm the tie is not just a capacity limit; see `runs/pre_studio/ex9_slot_attention_grind_s*.json`. This is precisely the kind of honest negative the doctrine values: it quantifies exactly what the pooled substrate throws away (per-object binding within a clip), and it is the empirical bound that a dense V-JEPA 2.1 retarget of E6 would have to beat.

### ex2_latent_planning (model-based latent planning, synthetic arm)

The synthetic (cpu-now) arm of latent MPC planning: fit a learned dynamics model g(z,a)->z' (a 2-layer MLP) on a synthetic mildly-nonlinear action-conditioned system, then shooting/CEM plan to reach goal latents, versus a flat-reactive-head baseline and an action-shuffle control, gated on rollout predictability. Shipped-scale result reported `null_supported=False` (planner beats both controls: terminal distance 4.78 vs flat-head 7.07 vs action-shuffle 5.94, one-step rollout R2 0.97, planning licensed).

**Adversarial verification: PLAUSIBLE-BUT-UNVERIFIED, leaning weak.** One real, non-obvious flaw was found: the planner's terminal distance is scored IN-BELIEF (against its own learned model's prediction of where it landed), while the flat-reactive baseline is rolled out and scored in the TRUE synthetic environment, so the two are not measured on the same yardstick, and "planner beats flat head" partly reflects a model's optimistic self-assessment beating an honestly-graded open-loop policy. Additional weaknesses: `flat_head_goal_success=0.0` signals a broken/strawman baseline rather than a strong one; goal_success=0.11 sits at or below the task's trivial floor (mean start distance ~8.0, success threshold ~4.0, planner terminal 4.78 is barely inside its own success bar); the rollout-predictability gate passes trivially because the synthetic dynamics are easy to fit; and it is single-seed. The ONE clean signal is the action-shuffle control: the planner clears it by 1.16, which does mean actions carry real, non-permutable information (a permuted action sequence plans worse). To promote this beyond "plausible", the Studio must (a) score the planner's terminal distance against the true dynamics, not its own model, (b) give the flat head closed-loop replanning or a matched grading yardstick, and (c) run multiple seeds. A harder/higher-dim big-run grind (dim 96, horizon 20, 10000 trials, ~95 min) was launched; see `runs/pre_studio/ex2_latent_planning_grind.json`.

---

## ADDENDUM: 7 More Experiments (EX1, EX4, EX6, EX7, EX11, EX14, EX18)

These 7 were built and run after the main 93-experiment adversarial-verification pass above completed, closing out the last registry-only cpu-now rows left over from the earlier EX-expansion session. All 7 declare the doctrine contract and ran for real on this laptop; results are at `runs/pre_studio/<id>.json`. 6 of 7 nulls held. The 2 candidate positives (EX4, EX6) were both checked adversarially and both refuted: EX4 by direct inspection against its own tuned control, EX6 by three independent adversarial passes (two skeptic re-analyses of the code plus a numerical resimulation), all converging on the same mechanism.

| id | verdict | key numbers | interpretation |
|---|---|---|---|
| ex1_generative_replay | null_held | BWT: no_replay=0.233, stored_buffer=0.625, generated=0.625 (exact tie); generator_probe_transfer=1.0 | A per-class diagonal-Gaussian generator ties real stored-buffer replay exactly on this toy stream; both clearly beat no-replay. The generator's distribution gap costs nothing measurable here, but the task's clusters are close to Gaussian by construction, which favors this outcome. |
| ex4_fast_weights | REFUTED (tuned_baseline) | hypernet_acc=0.328 vs static_acc=0.222 (beats it) vs gradient_tta_acc=1.000 (loses badly) | The reported null_supported=False keys only off "beats the weak static-head control," which it does. Against the TUNED control the mechanism is actually compared to in its own docstring (gradient-TTA), it loses by 67 points and gradient-TTA sits at ceiling. This is the same tuned-baseline-tie/loss pattern that refuted d7, d8, e9, ex16, i4, p2, y6 above. Not a real positive; only 2 seeds and 8 held-out evals besides. |
| ex6_active_inference | REFUTED (variance-magnitude artifact, three independent adversarial passes) | free_energy: fraction_time_learnable=1.0, rejects_noise=True; learning_progress: 0.489/False; standard_predictor_loss: 0.0/False | Looked like a clean same-architecture ablation at first read; it is not. Reanalysis found the `standard_predictor_loss` control differs from `free_energy` in TWO ways at once (no complexity term AND an opposite-polarity selection rule), so it never isolated the complexity term. A numerical resimulation then showed risk (≈plain NLL) drives ~99% of the EFE gap while the KL-complexity term is negligible or even slightly opposing. Decisive: a trivial zero-learning "always avoid the higher-target-variance region" heuristic reproduces fraction_time_learnable=1.0 exactly, and an inversion test (dropping the noise region's variance below the learnable region's own residual scale) makes the "free-energy" selector pick the irreducible noise 100% of the time, the opposite of "rejecting noise." The entire effect is an artifact of the hardcoded noise_scale=5.0 being large relative to the learnable region's residual variance, not evidence of active-inference or complexity-driven reducibility detection. |
| ex7_meta_learning | null_held | adaptation_speed: meta=2.50 vs random=2.78 vs hypernet=2.67 steps (gain 0.28, spread 2.0); forward_transfer: meta=0.965 vs random=0.942 (gain 0.023, spread 0.109) | Reptile meta-init nominally adapts faster and transfers better than random-init, but both gaps sit well inside the seed spread. Honest null: the synthetic task family is too homogeneous (or the 3-seed run too underpowered) to separate meta-learning from a lucky random init. |
| ex11_causal_probing | null_held | post_intervention_error=0.098 (seen range) vs extrapolation_error=0.243 (unseen do(v) values, gap 0.145); beats_observational_control=True | The interventional predictor generalizes reasonably to unseen intervention values (gap well inside the 0.5 margin) and clearly beats a control trained without any do()-interventions. A modest, real result on a toy causal task; worth a harder/wider extrapolation range at Studio scale to stress it further. |
| ex14_memory_bakeoff | null_held | recall@5: fifo=0.906 vs random=0.839 vs uncertainty=0.761 (uncertainty LOSES); BWT: fifo=-0.183 vs random=-0.163 vs uncertainty=-0.188 | Uncertainty-indexed eviction does not beat FIFO or random eviction on either recall or retention at matched buffer capacity; if anything it is the worst of the three. Clean, unambiguous null for this mechanism. |
| ex18_self_verification | null_held | single_shot_acc=0.361 vs verify_revise_acc=0.375 (gain 0.014, inside margin 0.03) vs shuffled_verifier_acc=0.368 (gain 0.007, nearly as good) | The trained verifier's correction gain does not clear its own margin, and a SHUFFLED, untrained verifier gets nearly the same gain — the small improvement from "verify and revise" is attributable to extra iteration, not a trained correction signal. Also: compute was not actually matched (`compute_matched.matched: false`, ratio 2.5x), so even the marginal gain isn't a fair comparison. Honest double null. |

**Addendum takeaway:** the pattern from the main 93 repeats exactly, including here. EX4 looked like a positive and failed the same tuned-baseline check that sank several other candidates. EX6 also looked like a clean positive on first read and did not survive adversarial scrutiny either, once three independent passes actually isolated its complexity term instead of trusting the module's own arm labels. The other 5 are honest nulls, one of them (EX11) genuinely informative (the shell does generalize somewhat past its training intervention range, a real if modest result). Combined with the main 93, this means across the full 100-experiment corpus exactly one candidate (e7_sparse) currently survives adversarial review, and only provisionally.
