# Null card: CM7 objective selection is not a live lever at 1.65M params on programmatic video

Five-seed familywise-corrected verdict on the custom substrate workbench. Form per BLACKHOLE.md:
no em or en dashes, engineering vocabulary only. Inherited encoders stay frozen; the custom
substrate lane is the only trainable lane, and this card retires one exact regime inside it.

## Claim under test

A 1 to 5M parameter token-preserving video substrate (TinyVideoSubstrate, 1,646,080 trainable
parameters, 256px, 8 frames, 256 dense tokens) trained under predictive, invariance, or
reconstruction objectives with identical initialization, architecture, batches, updates, and
matched estimated core FLOPs shows better held-out factor structure than a compute-matched
random-target control and an exact frozen copy of the same initialization. If at least one
learned objective clears both controls and the strongest alternative objective, objective choice
is a live lever at this scale.

## Controls

- random_target: compute-matched negative control, same updates and tokens, permuted targets.
- frozen_random: the exact shared initialization, never trained, same readout budget.
- Difficulty calibration (D3): a programmatic oracle solves the same held-out combination split
  at mean accuracy 1.0 (clears_floor true), so the split is solvable and the arms are off-ceiling.
- Compute match: all arms within tolerance (receipt compute_match.all_ok true).

## Result (5 seeds, Holm plus simultaneous Bonferroni t bounds, alpha .05, family of 12)

Primary metric heldout_combo_score (linear factor probe on held-out factor combinations):

- reconstruction 0.684, predictive 0.616, invariance 0.600 (learned arms)
- frozen_random 0.709, random_target 0.725 (controls)

Raw winner among learned arms is reconstruction. It clears nothing after correction:

- reconstruction vs frozen_random: mean delta -0.025, simultaneous lower bound -0.212, Holm p 1.0
- reconstruction vs random_target: mean delta -0.041, simultaneous lower bound -0.335, Holm p 1.0
- reconstruction vs predictive: mean delta +0.069, lower bound -0.222, Holm p 1.0
- reconstruction vs invariance: mean delta +0.084, lower bound -0.030, Holm p 0.49

Every learned arm trails both controls in mean. No comparison clears the 0.03 margin after
correction; per-seed paired deltas mix signs on the primary comparison. The verifier gate
winner_clears_all_corrected_comparisons is false; verdict not-promoted. The preregistered null
(objective is not a lever at this scale) HOLDS. A tie is a null; this is a tie or worse.

## What is retired and what carries forward

Retired: the exact 1.65M-parameter, 1,000-update, 256px, 8-frame programmatic regime as a place
where objective choice changes held-out factor structure. Training under any of the three learned
objectives added nothing over the untrained initialization at matched readout. Do not rerun this
regime with more seeds; the simultaneous bounds already exclude the registered margin.

Carries forward (platform, not science): the content-addressed checkpoint and resume contract,
the four-receipt immutable chain (raw receipt, current-evidence attestation, environment receipt,
independent familywise-corrected verifier), the combination-disjoint dataset manifest machinery,
the compute-match harness, and the D3 oracle calibration gate. CM8 and P4 (capability-density
response surface around CM7: 0.5x to 4x parameter budgets, more updates, natural referents) are
the named successors; each needs its own preregistration.

## Machine-readable card

```yaml
exp_id: mop_cm7_min_objective_probe
title: objective selection is not a live lever at 1.65M params on deterministic programmatic video
hypothesis: a 1-5M token-preserving video encoder trained on immutable 256px programmatic
  referents under predictive, invariance, reconstruction, and random-target objectives
  with identical initialization, architecture, batches, updates, and estimated core
  FLOPs; an inherited frozen teacher is an optional citable cache control rather than
  a dependency
null_hypothesis: at matched tiny capacity, matched data, matched 256px, both custom
  objectives tie random-init same-arch AND tie each other; objective is not a lever
  at this scale and the +0.31 was scale/data/architecture/resolution. A tie is a strong
  negative closing the custom-encoder line
baseline: exact-frozen-random-init-same-arch, compute-matched-random-target, optional-citable-frozen-teacher,
  d3-calibration, matched-capacity-data-resolution-updates-flops
ablation: every learned objective ties or trails both the exact frozen initialization and the
  compute-matched random-target arm; no corrected comparison clears the 0.03 margin
metric: probe_acc
probe_dependency:
  factor: identity
  encoder: vjepa2_vitl_fpc64_256
  atlas_row: proof/atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable: 'yes'
  acc_above_chance: 0.8414
encoder_scale: L
seeds:
  'n': 5
  sem: 0.0385
  sign_stability: unstable
provenance_tag: structured-synthetic
result: 'learned arms heldout_combo_score reconstruction 0.684, predictive 0.616, invariance
  0.600 vs frozen_random 0.709 and random_target 0.725; best-vs-frozen mean delta -0.025
  (simultaneous lower -0.212), best-vs-random-target -0.041 (lower -0.335); Holm p 1.0 on
  both control comparisons; five seeds, family of 12, alpha .05, df 4, t_crit 4.851'
taxonomy_category: 2
verdict: DOWNGRADE-TIE
badges:
- preregistered
- tuned-baseline-tie
raw_run_id: runs/custom_substrate/cm7_local180_citable_v3 (config_sha256 70ee47eba6479a77,
  data_sha256 8fadca50f346de56, requirements_sha256 27491c1ed6dfaabc; chain receipts raw
  ff0bc5ba, attestation 1d52d5b6, environment eec3acd3, verifier 81fbe054; composite
  proof/CUSTOM_SUBSTRATE_PILOT.json)
repro_level: R2
```

Notes: probe_dependency cites the teacher-scale atlas row for the optional frozen ViT-L teacher
lane; the decisive calibration for this card is the programmatic oracle (D3) at 1.0 on the same
held-out split. taxonomy_category 2 because a simpler control (the untrained initialization)
already captures everything the learned objectives produced at this regime; the adjacent reading
(category 4, capacity or update budget too small) is exactly what the P4 response surface is
preregistered to separate. MPS run; same-machine-class tolerance applies per the Metal caveat.
