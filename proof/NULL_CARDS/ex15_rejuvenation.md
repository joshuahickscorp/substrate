# Null card: shrink-and-perturb rejuvenation is not substrate-specific

Registry entry ex15_rejuvenation (EX series, taxonomy slot 8), promotes shrink_and_perturb.
Form per BLACKHOLE.md: no em or en dashes, no agency or intelligence language. The encoder is
frozen and never trained; only the small reader shell over frozen identity latents is fit.

## Claim under test

Periodic shrink-and-perturb rejuvenation (shrink 0.6, noise_std 0.02, every 25 tasks) restores
plasticity over a long 240-task stream without paying a retention cost, and does so specifically
because it acts on the trained reader shell (a protected-latent substrate) rather than being a
generic reset that a frozen-random substrate would show equally. The falsifier is: rejuvenation
recovers effective rank and dead-unit count while retaining accuracy, in a substrate-specific way.

## Control (no-rejuvenation and frozen-random-substrate)

Three arms, identical 240-task stream, chance 0.25, 3 anchor tasks, eval every 10:
  - protected (no rejuvenation): the tuned baseline. Final anchor acc 0.7222, final effective rank
    4.453, final dead units 0, 0 rejuvenation events.
  - protected_rejuvenated: same substrate plus the schedule. Final acc 0.5834, final rank 4.1085,
    final dead units 0, 9 rejuvenation events.
  - frozen_random_rejuvenated (substrate control): a frozen-random shell with the same schedule.
    Final acc 0.6666, final rank 4.8803, final dead units 0, 3 rejuvenation events.
Substrate-specificity gate: rejuvenation is substrate-specific only if the protected arm gains from
it in a way the frozen-random arm does not. It does not. The rejuvenated protected arm (0.5834)
underperforms both the frozen-random-rejuvenated control (0.6666) and its own no-rejuvenation
baseline (0.7222), so substrate_specific = false in the run.

## Result

Preconditions for the falsifier are absent, so there is nothing to restore. dead units stay at 0 in
every arm across the whole stream and effective rank never collapses, so plasticity_loss_observed =
false at this 240-task scale. Given no loss of plasticity, the schedule cannot restore it:
restores_plasticity = false, rank_restored = false, dead_units_reduced = false. Rejuvenation instead
lowers retained accuracy on the protected substrate by 0.1388 (0.7222 -> 0.5834), so
retention_cost_paid = true (a cost with no matching plasticity benefit). And it is not
substrate-specific: substrate_specific = false. The null (rejuvenation does not restore plasticity,
or restores it at a retention cost, and the frozen-latent shell does not suffer loss of plasticity
at this scale) is supported: null_supported = true. The honest verdict is DOWNGRADE-TIE: the
schedule is not a substrate-specific plasticity restorer here, it is a net retention loss on a shell
that never lost plasticity to begin with.

## Non-degeneracy (task calibration for a plasticity metric)

This is a continual-learning / plasticity metric, so the required check is difficulty calibration,
not a representational atlas factor. The anchor tasks are genuinely learnable and non-degenerate:
peak mean anchor acc reaches 0.8889 in both the protected and protected_rejuvenated arms and 0.8611
in the frozen-random arm, all well above the 0.25 chance floor (peak above chance 0.6389). So the
retention drop and the flat effective rank are real signals on a task that can be learned, not a
degenerate floor effect. The stream is calibrated so a plasticity loss would be visible if it
occurred; at 240 tasks it does not occur, which is exactly the null the registry pre-declared for
this scale. The _grind sidecar pushes the same harness to a longer stream (rejuvenation intervals
out to 42500) and still shows no substrate-specific rejuvenation benefit, consistent with this card.

```yaml
exp_id:            EX15
title:             shrink-and-perturb rejuvenation does not restore plasticity and is not substrate-specific at 240-task scale
hypothesis:        periodic shrink-and-perturb restores effective rank and dead units while retaining accuracy, specifically on the protected-latent shell
null_hypothesis:   rejuvenation does not restore plasticity, or it restores plasticity at the cost of retention; the frozen-latent shell does not suffer loss of plasticity at this scale
baseline:          protected arm with no rejuvenation (tuned no-schedule long-stream shell); final anchor acc 0.7222, final effective rank 4.453, 0 dead units
ablation:          frozen-random-substrate arm with the same schedule (final acc 0.6666) isolates substrate-specificity; it matches or beats the rejuvenated protected arm (0.5834), so no substrate specificity
metric:            retained_accuracy
probe_dependency:
  factor:          identity
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes
  acc_above_chance: 0.6389
encoder_scale:     L
seeds:
  n:               3
  sem:             null
  sign_stability:  stable at S>=3 (all three arms agree: no dead units, no rank collapse, no substrate-specific gain)
provenance_tag:    real-encoder
result:            protected_rejuvenated final acc 0.5834 vs protected 0.7222 (retention_cost 0.1388, paid) and vs frozen_random_rejuvenated 0.6666; plasticity_loss_observed false, restores_plasticity false, rank_restored false, dead_units_reduced false, substrate_specific false; peak acc 0.8889 vs chance 0.25; null_supported true
taxonomy_category: 8
verdict:           DOWNGRADE-TIE
badges:            [tuned-baseline-tie]
raw_run_id:        runs/pre_studio/ex15_rejuvenation.json (240-task stream) + runs/pre_studio/ex15_rejuvenation_grind.json (long-stream sidecar)
repro_level:       R1
```

## What this closes

This closes the shrink_and_perturb rejuvenation claim at cpu-now scale: on a frozen-latent identity
shell that shows no loss of plasticity over 240 tasks (0 dead units, stable effective rank near
4.4), the rejuvenation schedule restores nothing and instead costs 0.1388 of retained accuracy, and
the frozen-random control shows the schedule is not substrate-specific. It does not close the claim
at Studio scale: the registry names a thousands-of-tasks rerun that forces real loss of plasticity
as the natural extension, where a rejuvenation benefit could still appear once dead units actually
accumulate. Until then the honest verdict is DOWNGRADE-TIE, not a positive.
