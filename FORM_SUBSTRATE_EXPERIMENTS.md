# FORM SUBSTRATE EXPERIMENTS

The F-series bank and the migration map from every prior experiment family. The registry
(registry/experiments.yaml) remains the single machine-readable preregistration; this document is
the human view: what each experiment is for, what it must beat, and what it unlocks.

House style: no em or en dashes. Companions: FORM_SUBSTRATE_PROGRAM.md (architecture stack),
FORM_SUBSTRATE_DOCTRINE.md (controls), PERFORMANCE_DENSITY_DOCTRINE.md (every row needs a capability
metric, a cost metric, and a matched-cost baseline).

---

## 1. Bank status

| ID | Name | Layer | Status | Tier | Job |
|---|---|---|---|---|---|
| F1 | form_alignment_gate | 2 | implemented | cpu-now | paired referents align forms beyond raw and shuffled controls |
| F2 | heldout_form_transfer | 2-4 | implemented | cpu-now | multi-form training transfers to an unseen observation family |
| F3 | form_bottleneck_capacity | 4 | implemented | cpu-now | canonical form width is load-bearing |
| F4 | raw_payload_vs_form_tokens | 1-4 | implemented | cpu-now | form tokens beat raw ad hoc featurizers |
| F5 | cross_form_memory_binding | 6 | implemented | cpu-now | memory retrieves the same referent across forms |
| F6 | sensorimotor_form_closure | 8 | implemented | cpu-now | action as a form improves prediction and planning |
| F7 | developmental_form_growth | 10 | implemented | cpu-now | growth beats matched-final-capacity controls |
| F8 | plastic_substrate_rewrite | 10 | registry-only | env-later | trainable substrate beats frozen plus larger shell |
| F9 | cross_form_compositional_binding | 2-4 | implemented | cpu-now | held-out factor combos survive across forms |
| F10 | intrinsic_form_curriculum | 7 | implemented | cpu-now | learning progress chooses which form to study |
| F11 | form_dream_replay | 6 | implemented | cpu-now | generated replay protects referents at lower bytes |
| F12 | private_form_language_stability | 4 | implemented | cpu-now | form codes recur across seeds (vs idiolects) |
| F13 | form_energy_budget | 9 | implemented | cpu-now | capability per byte, param, FLOP, energy proxy |
| F14 | lifelong_form_expansion | 1-6 | implemented | cpu-now | new form added without remapping old memory |
| F15 | embodied_affordance_form | 8 | implemented | cpu-now | affordances from consequences, not labels |
| F16 | perfect_slate_null | 10 | registry-only | env-later | blank trainable substrate vs inherited frozen controls |
| F17 | missing_form_recovery | 7 | implemented | cpu-now | recovery and calibration when a form is absent |
| F18 | counterfactual_form_intervention | 2-8 | implemented | cpu-now | one form's factor change predicts another form's change |
| F19 | cross_scale_referent_binding | 6 | implemented | cpu-now | one referent bound at object, scene, episode, task scale |
| F20 | substrate_crisis_test | 7-10 | implemented | cpu-now | detect when the form interface is insufficient |

All 18 implemented F experiments run locally today across the two F-series modules and their configs as
controlled mechanics; their job is falsification plumbing, not headline science. F8 and F16 are the two
registry-only lanes. Their complete fixture-scientific engine also runs locally, but natural execution is
environment-gated by a rights-clean training source, executable real-weight receipt, and trusted external
provenance authority. No M3 failure receipt exists, so neither lane is classified as Studio hardware work.
Each result carries a performance-density block and every candidate positive remains R0 until its evidence
gate is satisfied.

## 2. The four new experiments (added by Master Plan v2)

### F17: Missing-Form Recovery

Hypothesis: the substrate degrades gracefully and knows it is degraded when a form is absent.
Forms: any trained matrix minus one arm at test time. Referents: shared, as trained.
Modules: substrate/form.py, diagnostics/operational_awareness.py (OA1, OA2).
Capability metric: recovery accuracy under absence. Cost metric: recovery latency.
Density metric: recovered accuracy per extra FLOP spent on imputation.
Baselines: impute-by-mean, best remaining single form, concat model with zeros.
Null: recovery ties best remaining form, or confidence fails to drop under absence.
Failure interpretation: forms are redundant channels, not complementary evidence (taxonomy 2), or
the OA layer is uncalibrated (taxonomy 4).
Unlock: OA1/OA2 become experimental facts; licenses missing-form routing in Layer 7.

### F18: Counterfactual Form Intervention

Hypothesis: intervening on a factor visible in form A predicts the induced change in form B over the
same referent, beyond correlation.
Forms: paired forms with programmatic factor control (synthetic first; DR1 bound-attribute video
plus captions at Studio scale). Referents: before-and-after intervention pairs.
Modules: f_form_substrate.py, diagnostics/held_out_combo.py, ex11 causal-probing harness (reused).
Capability metric: counterfactual match accuracy on unseen intervention values.
Cost metric: intervention-model FLOPs vs correlational predictor FLOPs (matched).
Density metric: counterfactual accuracy per parameter.
Baselines: correlational predictor, random intervention direction, shuffled counterfactual pairs.
Null: interventions leak (seen-value memorization) or tie correlation.
Failure interpretation: the matrix binds appearances, not causes (taxonomy 10 or 3).
Unlock: cross-form claims may use intervention language; feeds F6/F15 action forms.

### F19: Cross-Scale Referent Binding

Hypothesis: the same referent can be stored and retrieved at object, scene, episode, and task scale
without a separate memory per scale.
Forms: any; the variable is referent granularity. Referents: nested ids (object within scene within
episode within task), enforced at intake.
Modules: substrate/form.py (referent hierarchy), shell/buffer.py KVIndex, F5 harness extended.
Capability metric: retrieval accuracy across scale jumps (store object, query episode, and reverse).
Cost metric: memory bytes vs flat per-scale stores at matched recall.
Density metric: cross-scale recall per byte.
Baselines: single-scale memory, flat clip memory, random hierarchy.
Null: hierarchy ties flat memory at matched bytes.
Failure interpretation: scale is a nuisance dimension for this content (taxonomy 5), or pooling
destroyed sub-referent structure (taxonomy 3).
Unlock: memory claims stop being clip-shaped; episodic structure becomes testable.

### F20: Substrate Crisis Test

Hypothesis: a monitor over the form matrix detects, before the fact, when frozen forms plus larger
shells will fail a target, and stays quiet on noise.
Forms: matrices constructed to contain known-insufficient arms (the A6 nuisance record and the
FACET12 predictor wall provide real positive-crisis exemplars; noisy-TV streams provide the
false-alarm bed). Referents: shared.
Modules: diagnostics/operational_awareness.py (OA6, OA7), diagnostics/riskcov.py,
diagnostics/noisy_tv.py, falsification/verdict_gate.py (crisis verdicts feed the Layer 10 gate).
Capability metric: crisis detection AUROC against realized probe failure.
Cost metric: monitor overhead FLOPs (must be a rounding error next to the shell).
Density metric: avoided-wasted-compute per monitor FLOP.
Baselines: fixed confidence threshold, raw error signal, random trigger at matched rate.
Null: crisis detector ties raw error or triggers on aleatoric noise.
Failure interpretation: insufficiency is not predictable from the signals exposed (taxonomy 4), or
the exemplar set is too narrow (taxonomy 5).
Unlock: the ONLY evidence stream that can open Layer 10 ahead of a completed failure: F8 and F16
remain gated on doc 15 triggers, and F20 is how trigger evidence is collected honestly.

## 3. Promotion rules for the bank

A form-substrate positive clears, in order: non-ceiling difficulty, matched controls, referent
alignment audit, shuffled floors, matched compute for trainable comparisons, larger-shell-on-frozen
for substrate-training claims, seed stability, null card, density block, no prose escalation. This
list is law (FORM_SUBSTRATE_DOCTRINE.md section 2; PERFORMANCE_DENSITY_DOCTRINE.md section 3).

## 4. Experiment migration map (old ID to new paradigm role)

### Keep active (unchanged instruments the F-series depends on)

| Old | Role now |
|---|---|
| mop_dr1_video_cache (DR1) | Phase 3 real-forms intake: the first natural referent corpus with paired vision and caption forms; its acceptance gates become the multi-form intake standard |
| mop_pr9 lane (PR9) | plasticity kill-switch feeding the F8/F16 license (process_c gate) |
| atlas lanes (ex12, AT1-AT5, dense atlas gates) | form currency atlas: per-arm decodability floor every F-series probe_dependency cites |
| e1_baseline | the continual gate under every retention claim |
| B5 multi-seed substrate split | the inheritance-is-real anchor F16 must beat |
| e7_sparse lineage (DR2/PR3 real-latent retest) | sparse mode inside Layer 5, retest scheduled |
| ex2_latent_planning lineage | planning mode inside Layer 5; blocked at horizon by FACET12 wall |
| d3/difficulty calibration, d5/compute, a1/a2 controls | standing gates, unchanged |

### Convert to F-series (rewritten under form vocabulary)

| Old | New |
|---|---|
| ex10 cross-modal binding, AL1-AL3 alignment | F1/F2 at scale (real paired forms) |
| A6 shared-code lane (nuisance-residualized) | F1's residualization protocol + F18's exemplar bank |
| 06 cognitive currency atlas | form currency atlas (AT-series continues under Layer 2/4) |
| WS1-WS5 workspace arbitration | Layer 5 mode-ecology tests; WS1 agreement-vs-confidence feeds OA2 |
| MT/MP router rows | Layer 5 routing under the EX-ROUTER-DENSITY null; must beat the recorded control |
| e5/ex8 curiosity | F10 intrinsic form curriculum |
| ex14 memory bakeoff, n7 WM | F5/F19 memory binding at scale |
| ex16 codebook, s5/p5/y3 code stability | F12 private form language stability |
| ex1 generative replay, n1 replay ordering | F11 form dream replay |
| b8 structural growth | F7 developmental form growth |
| CM1-CM12 custom-model ladder | F8 (rewrite) and F16 (perfect slate) gates; CM triggers T1-T5 unchanged |
| ex11 causal probing | F18 counterfactual form intervention |
| a1-a8 affordance capacities | F15 embodied affordance form (env-later) |
| ex13 long-stream | F14 lifelong expansion's retention harness |

### Archive as negative map (evidence, not agenda)

e4 neuromod (30/30 wrong direction), e3/n5/n6 critical periods, b4/y5 homeostasis, d4 U-shape (dip
exactly 0), d1 fast mapping (ceiling), ex5 local rules (Adam artifact), ex15 rejuvenation (nothing
to repair), ex17/ex18/p9/n9/y1 and the whole test-time-compute lane (iteration equals depth, 24
nulls), pr5, ws5, ws2, at3 temporal currency (injected velocity), al2 shared code, EX-ROUTER-DENSITY
(router loses on real cache), s10 PASS-VACUOUS (the apparatus warning), p1/s1/s6/c1/c9 abstraction
vacuities, idiolect results (p5/s5/y3). These rows stay in the registry with their verdicts; they are
the priors every F-series design must not re-lose to.

### Re-run under new controls (not wrong, underpowered or reframed)

| Old | Re-run as |
|---|---|
| compositional binding (ceilinged synthetic) | F9 on DR1 real bound-attribute video |
| dense vs pooled (additive synthetic) | dense-token F4/F9 on non-additive referents behind the dense atlas gate |
| e7 sparse on real latents (sign-flipped at 10 seeds) | DR2/PR3 30-run settle |
| cross-form retrieval toys | F5 with shuffled-referent controls on real paired forms |
| density under routing (constructed-task win, natural null) | F13 with the matrix predicting when heterogeneity headroom exists |
| PR2 real-vs-random plasticity (ran at chance) | Studio-scale retest on the DR1 stream |

## 5. What the bank refuses

No experiment enters the bank without a null. No F-series row may cite a probe factor the atlas
marks not-decodable. No trainable-substrate row may run without its license receipt. No density row
may report peak without cost. No new module may be written for an experiment that an existing module
already serves (FORM_SUBSTRATE_CODEMAP.md section 2 refusal table).
