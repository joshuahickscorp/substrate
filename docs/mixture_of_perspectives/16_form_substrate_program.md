# Section 16: Form Substrate Program

Breaking out of the V-JEPA-shaped format without losing the falsification discipline.

House style: no em or en dashes. This document is a plan and scaffold, not a result. It does not claim
sentience, consciousness, or human-level cognition. It names the next object of study: a form substrate.

---

## 0. The Pivot

The old center of gravity was:

`video -> frozen V-JEPA -> pooled latent -> tiny shell`

That was a useful first instrument. It made the program concrete, cheap enough to run, and hard to
mythologize because every claim had a null. But it also shaped the imagination too much. Even when the docs
said "not a JEPA", the whole system still breathed through a V-JEPA-shaped pipe: 64 frames, pooled visual
latents, cached feature stores, and a shell trained around that inherited geometry.

The new center of gravity is:

`any observation form -> referent-aligned form substrate -> modes and memory -> falsified capability`

V-JEPA remains valuable. It is no longer the cathedral. It is one inherited perceptual arm inside a wider
substrate ecology. The unit of analysis is not "what can we do on this video latent", but "what structure
does a system preserve, align, rewrite, and act through when dropped into different data forms".

The sci-fi image, the perfect orb or blank slate, becomes falsifiable like this:

- A blank slate is not magic.
- A substrate is not proved by being general in prose.
- A form is not proved by accepting arbitrary tensors.
- A learned substrate is not better because it is trainable.
- A "human-like" claim is inadmissible unless it reduces to measurable capacities with controls.

So the question becomes: can a substrate receive multiple kinds of data, preserve referents and factors,
align them across forms, adapt its internal structure when experience demands it, and beat inherited frozen
or raw-feature controls on density, transfer, memory, and action?

That is a real research program. It is also much less LLM-like, because the substrate is not a sequence model
with language as the privileged format, and much less V-JEPA-like, because video is not the privileged input.

---

## 1. Vocabulary

### Form

A form is an observation family plus its machine-readable contract:

- vision frames or dense visual tokens
- audio waveforms or audio SSL states
- text-derived metadata or language hidden states
- symbolic scene graphs
- scalar telemetry
- action traces
- code states
- math objects
- event logs
- latent stores
- any future learned substrate arm

A form must carry:

- a tag
- a kind
- a source
- an objective family
- a feature dimension or token shape
- referent ids
- optional factor labels
- a matched control when it is used as evidence

Without referent ids, it is just a pile of vectors. Without controls, it is just a story.

### Referent

A referent is the shared thing multiple forms are about. It can be a clip, object, event, episode, program,
trajectory, environment state, or task instance. The referent is the anchor that lets the system ask whether
vision, audio, text, symbols, and action are talking about the same world point.

### Form Substrate

A form substrate is the normalized interface that presents arbitrary forms as aligned features over shared
referents. It is not necessarily trainable. It can start as pure cached tensors. It becomes a trainable
substrate only when an experiment allows the form layer itself to update and beats frozen controls.

### Perfect Slate

The perfect slate is the aspirational object: a substrate with minimal assumptions that can be dropped into
a data world and grow the structures needed for that world. In this repo it becomes a null-first hypothesis:

> A blank trainable substrate trained on the same small curriculum must beat inherited frozen substrates,
> random-init same-architecture controls, and larger-shell-on-frozen controls at matched total compute.

If it ties, the blank-slate story bought nothing at that scale. That negative is useful.

---

## 2. What We Are Moving Away From

We are moving away from V-JEPA as the format of thought.

Not away from V-JEPA entirely. It remains a strong inherited perceptual substrate, and current evidence says
pretraining buys real nuisance invariance and some compositional structure. But it cannot be the only shape
the system knows how to receive.

The old failure mode:

1. Build a clever shell mechanism.
2. Run it on pooled V-JEPA or synthetic latents.
3. Discover the effect was ceilinged, projection-vacuous, matched-compute dead, or substrate-bounded.
4. Add another shell mechanism.
5. Repeat.

The new failure mode to avoid:

1. Rename everything "forms".
2. Pass arbitrary tensors through the same classifier.
3. Declare generality.

The new discipline:

1. Every form must align over referents.
2. Every substantive form must have a matched control.
3. Every transfer claim must beat raw, shuffled, and single-form baselines.
4. Every trainable-substrate claim must beat frozen-substrate-plus-larger-shell controls.
5. Every density claim must report per byte, per parameter, per update, or per estimated energy.
6. Every positive must survive seed stability.
7. Every negative maps to the failure taxonomy.

---

## 3. Architecture Scaffold

### Layer A: Form Intake

Purpose: accept arbitrary observation families without giving any one of them privileged status.

Code scaffold:

- `src/mop/substrate/form.py`
- `FormMeta`
- `FormBatch`
- `TensorFormAdapter`
- `FormMatrix`
- `build_form_matrix`
- `form_audit`

This layer is deliberately dumb. It does not train. It says: here are N referents, here are K forms over
those referents, here are the feature tensors, here is how the controls attach.

Minimum acceptance:

- all forms share the same referent set
- duplicate referents are refused
- duplicate tags are refused
- features must match declared flattened dimension
- audit names substantive forms with no controls
- audit names trainable arms so substrate and shell effects cannot be confused

### Layer B: Form Alignment

Purpose: align one form to another by shared referents, then make the alignment itself a baseline.

Code scaffold:

- `fit_affine_alignment`
- `apply_affine_alignment`
- F1 form alignment gate

Why affine alignment first? Because it is cheap, explicit, and easy to beat or refute. If a future form
substrate cannot beat a paired affine map, it has not earned a complicated architecture. If affine alignment
already solves the toy case, that becomes the baseline for harder natural data.

Controls:

- raw target transfer
- shuffled-referent alignment
- source-form ceiling
- chance floor

### Layer C: Cross-Form Learning

Purpose: train on several aligned forms and test a held-out form.

Code scaffold:

- F2 held-out form transfer

This asks whether the interface generalizes beyond the form families that provided labeled training. It is
not enough that vision transfers to audio if audio examples were directly labeled. The stronger question is
whether unlabeled referent alignment plus multi-form training gives the system a usable held-out form.

Controls:

- single-reference-form baseline
- chance floor
- alignment audit

### Layer D: Form Bottleneck

Purpose: make the substrate interface capacity itself load-bearing.

Code scaffold:

- F3 form bottleneck capacity

The perfect substrate is not "anything goes". If the canonical bottleneck is too narrow, it should lose
information. If it does not, the task is too easy or the factor is too low-bit. F3 turns bottleneck width
into a scored variable.

Controls:

- small bottleneck
- wide bottleneck
- shuffled-label floor
- matched head and data

### Layer E: Cross-Form Memory

Purpose: store through one form and retrieve through another.

Planned experiment:

- F5 cross-form memory binding

This is the memory version of the form problem. If an object is stored visually, can it be retrieved from
audio, text, action, or symbolic query forms? If not, memory is form-local, not referent-bound.

Controls:

- per-form nearest neighbor
- shuffled referents
- matched memory slots

### Layer F: Sensorimotor Closure

Purpose: introduce action as a first-class form.

Planned experiments:

- F6 sensorimotor form closure
- F15 embodied affordance form

This is the first step away from passive perception. A substrate that only watches is bounded. Action forms
let us ask whether the substrate represents consequence, intervention, and controllability, rather than only
appearance.

Controls:

- action-blind
- action-shuffle
- matched compute
- rollout predictability gate

### Layer G: Plastic Substrate

Purpose: allow the substrate itself to change, but only under hard controls.

Planned experiments:

- F7 developmental form growth
- F8 plastic substrate rewrite
- F16 perfect slate null

This is the real departure from the inherited frozen-substrate program. It is also where overclaim risk is
highest. A trainable substrate must beat:

- frozen inherited substrate
- larger shell on frozen substrate
- random-init same architecture
- matched total compute
- seed-stability floor

If it does not, unfreezing was just extra capacity or optimizer luck.

---

## 4. The F Series

### Runnable now

| ID | Name | Job |
|---|---|---|
| F1 | form_alignment_gate | prove paired referents can align forms better than raw and shuffled controls |
| F2 | heldout_form_transfer | test whether multi-form training transfers to an unseen observation family |
| F3 | form_bottleneck_capacity | test whether the canonical form width is load-bearing |
| F5 | cross_form_memory_binding | test whether memory retrieves the same referent across forms |

These are small toy experiments. Their job is mechanics and falsification, not headline science.

### Registry-only backlog

| ID | Name | Why it exists |
|---|---|---|
| F4 | raw_payload_vs_canonical_form_tokens | tests whether form tokens beat raw ad hoc featurizers |
| F6 | sensorimotor_form_closure | adds action as a form and checks planning |
| F7 | developmental_form_growth | tests structural growth and pruning |
| F8 | plastic_substrate_rewrite | tests representational rewrite beyond frozen controls |
| F9 | cross_form_compositional_binding | re-asks compositionality outside vision-only content |
| F10 | intrinsic_form_curriculum | chooses which form to study next by learning progress |
| F11 | form_dream_replay | generates canonical form states for replay |
| F12 | private_form_language_stability | tests whether form codes recur across seeds |
| F13 | form_energy_budget | scores density per byte, param, and energy proxy |
| F14 | lifelong_form_expansion | adds a new form after training without remapping old memory |
| F15 | embodied_affordance_form | learns affordances from consequences, not passive labels |
| F16 | perfect_slate_null | tests blank trainable substrate against inherited frozen controls |

This list is intentionally broader than V-JEPA. It includes passive forms, active forms, memory, plasticity,
density, and blank-slate tests.

---

## 5. The New North Star

Old north star:

`perceive -> remember -> predict -> surprise -> adapt -> consolidate -> abstract -> transfer`

New form-substrate north star:

`receive any form -> bind referents -> preserve factors -> align perspectives -> remember across forms -> act and predict consequences -> adapt the interface when frozen form fails -> compress into dense capability`

This is not a claim that the system is human. It is a better experimental object because it measures the
things a mind-like substrate would need to do without pretending that one video encoder and a pooled latent
are enough.

---

## 6. Core Hypotheses

### H1: Form Alignment

Claim: paired referents enable transfer between forms beyond raw coordinate transfer.

Null: aligned transfer ties raw target transfer or shuffled anchors.

First experiment: F1.

### H2: Held-Out Form Generalization

Claim: a concept learned through multiple forms transfers to an unseen form after unlabeled alignment.

Null: multi-form training ties single-form baseline or remains near chance.

First experiment: F2.

### H3: Bottleneck Load

Claim: the canonical form width is a real capacity variable.

Null: wide and small bottlenecks tie, or both sit at the shuffled floor.

First experiment: F3.

### H4: Referent-Bound Memory

Claim: memory stores referents, not only same-form features.

Null: cross-form retrieval ties form-local nearest neighbor or shuffled referents.

First experiment: F5.

### H5: Action Closure

Claim: representing action consequences as a form improves prediction and planning.

Null: action-blind or action-shuffle controls tie.

First experiment: F6, then F15.

### H6: Plastic Form Rewrite

Claim: a trainable substrate can expose factors frozen substrates plus larger shells cannot.

Null: frozen inherited substrate plus larger shell ties the plastic substrate.

First experiment: F8, gated by F1 to F5 and Studio-scale content.

### H7: Perfect Slate

Claim: a small blank substrate trained on the same curriculum beats inherited frozen substrates under
matched compute.

Null: blank substrate ties inherited frozen features plus shell controls, or ties random-init same-arch.

First experiment: F16.

---

## 7. What Counts As A Real Positive

A form-substrate positive must clear all applicable gates:

1. Non-ceiling difficulty.
2. Matched controls.
3. Referent alignment audit.
4. Shuffled-referent or shuffled-label floor.
5. Matched compute for any trainable comparison.
6. Larger-shell-on-frozen control for any substrate-training claim.
7. Seed stability.
8. Null card before promotion.
9. Density metric when the claim is efficiency.
10. No prose escalation from "feature transfer" to "understanding".

The last point matters. The F series can make the system less V-JEPA-like. It cannot license mystical
language. If a result is "cross-form transfer improved by 0.12", say that. Do not call it human
understanding.

---

## 8. Roadmap

### Phase 1: Local Form Mechanics

Status: scaffolded.

Tasks:

- implement `src/mop/substrate/form.py`
- add unit tests for referent alignment and audit behavior
- implement F1 to F3 and F5
- add configs and registry rows
- render `EXPERIMENTS.md`

This phase proves the repo can run a non-video-shaped experiment without breaking the doctrine contract.

### Phase 2: Local Cross-Form Memory and Compositionality

Next experiments:

- F4
- F5
- F9
- F10
- F12
- F13
- F14

These should remain CPU-now and synthetic at first. The target is mechanics plus negative-resistant
controls. A toy negative is fine if the control is sharp.

### Phase 3: Studio DR1 Forms

Inputs:

- real bound-attribute video
- paired language/caption features
- possibly audio
- programmatic reference sidecars
- random-init same-architecture controls

Goal:

- replace synthetic form worlds with real referents
- test F5/F9/F12 on actual paired forms
- test whether form memory and cross-form compositionality survive beyond toy content

### Phase 4: Action Forms

Inputs:

- environment or offline action-conditioned trajectories
- action labels or control signals
- outcome factors

Experiments:

- F6
- F15

Goal:

- determine whether form substrate can represent consequence and controllability
- avoid pretending passive video is embodiment

### Phase 5: Plastic Substrate

Experiments:

- F7
- F8
- F16

Goal:

- allow the substrate itself to change
- beat frozen controls and larger-shell controls
- decide whether the blank-slate branch is live or a beautiful dead end

---

## 9. Decision Gates

### Gate A: If F1 Fails

If paired-referent alignment ties raw and shuffled controls, the form interface is not yet real. Do not build
plastic substrate machinery. Fix referent pairing, feature calibration, or task difficulty first.

### Gate B: If F2 Fails

If held-out form transfer ties single-form baseline, multi-form training did not create form-general
structure. Proceed to F5 memory and F9 compositionality, but do not claim substrate generality.

### Gate C: If F3 Fails

If small and wide bottlenecks tie, the task is too easy or the factor is too low-bit. Increase classes,
factor count, nuisance, or held-out combinations before using bottleneck results for architecture decisions.

### Gate D: If F5 Fails

If memory is form-local, then the substrate is not referent-bound. Form alignment may still be useful, but
the memory story is bounded.

### Gate E: If F6/F15 Fail

If action-shuffle ties true action, the substrate has not learned consequence. Keep action out of the core
claim until an environment or dataset can pose it.

### Gate F: If F8/F16 Fail

If trainable substrates tie frozen inherited features plus larger shells, the blank-slate branch is not
worth escalating. Keep V-JEPA and other inherited encoders as frozen arms, and spend effort on forms,
memory, and routing.

---

## 10. Why This Is Less LLM-Like

The LLM-shaped assumption is that the privileged substrate is text tokens and the master operation is next
token prediction or instruction-conditioned generation. The F series does not privilege text. Text can be a
form, but so can audio, action, symbolic state, telemetry, or dense visual tokens.

The system is asked to:

- align forms by shared referents
- transfer concepts between forms
- store and retrieve across forms
- preserve held-out factor combinations
- choose which form to study
- add new forms without remapping old ones
- represent action consequences
- rewrite its own substrate only if frozen controls fail

None of that requires a language model as the center. A language model can appear as one perspective arm,
but it is not the substrate definition.

---

## 11. Why This Is Less V-JEPA-Like

The V-JEPA-shaped assumption is that inherited visual perception is the substrate and everything else is a
shell around its pooled features. The form program demotes that assumption:

- V-JEPA is one form arm, not the ontology.
- Video clips are one referent type, not the only world.
- Pooled latents are one feature geometry, not the interface.
- Dense tokens are one possible form, not the destination.
- A custom substrate is gated by controls, not desire.

The new interface can ingest V-JEPA, DINO, audio SSL, language hidden states, symbolic graphs, action
traces, and future trainable substrates under the same matrix and audit.

That is the break. Not abandoning V-JEPA, but refusing to think in its shape.

---

## 12. Immediate File Map

New or updated scaffolding:

- `src/mop/substrate/form.py`: generic form interface and alignment primitives.
- `src/mop/experiments/f_form_substrate.py`: F1 to F3 and F5 runnable toy experiments.
- `configs/experiment/f1_form_alignment_gate.yaml`: F1 default.
- `configs/experiment/f2_heldout_form_transfer.yaml`: F2 default.
- `configs/experiment/f3_form_bottleneck_capacity.yaml`: F3 default.
- `configs/experiment/f5_cross_form_memory_binding.yaml`: F5 default.
- `tests/unit/test_form_substrate.py`: form contract tests.
- `registry/experiments.yaml`: F1 to F16 registry expansion.
- `EXPERIMENTS.md`: generated bank view after registry render.

This is enough for the next turn to run F1 to F3 and F5, implement F4 or F9, or start replacing toy forms
with real paired feature caches.

---

## 13. The Principle To Keep

The project gets more ambitious only by becoming more falsifiable.

The form substrate can be broad, strange, and expensive. It can aim at the blank-orb intuition. But each step
must keep the old spine:

- baseline
- ablation
- metric
- null
- control
- gate
- negative taxonomy

That is how this becomes its own thing rather than another quasi-LLM or quasi-V-JEPA story.
