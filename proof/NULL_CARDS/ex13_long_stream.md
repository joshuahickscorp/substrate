# Null card: long-stream continual-learning substrate-blindspot

Long-stream forgetting curve (EX series, taxonomy slot 8). Form per BLACKHOLE.md: no em or
en dashes, no agency or understanding language. The encoder is frozen and never trained;
only linear anchor-task reader heads are fit as the stream advances.

## Claim under test

Replay plus EWC (the protected arm) retains anchor-task accuracy across a long domain-incremental
stream where naive-sequential forgets, and the gap widens with stream length. The protected arm does
beat naive on divergence (protected-minus-naive anchor backward-transfer = +0.2499 at 240 tasks,
+0.5278 at 3000 tasks, growing with length). The question is whether that advantage needs the real
protected substrate, or whether an equally-sized frozen-random substrate buys the same retention.

## Control (the frozen-random-substrate arm the protected arm must beat)

The frozen-random arm holds a fixed random substrate of matched size and fits the same anchor heads.
At 240 tasks it retains final anchor accuracy 0.6666 (anchor bwt -0.1945) against protected 0.7222
(anchor bwt -0.1667): a gap of only 0.0278 bwt, so the protected arm does not clear the control. At
3000 tasks the two arms tie exactly (protected anchor bwt -0.2222, frozen_random anchor bwt -0.2222).
Pre-declared gate survives_frozen_random_control is False in BOTH the 240-task run and the 3000-task
grind. A tie against the control is a null: the divergence over naive is real, but it is not shown to
require the protected substrate rather than any substrate of the same size.

Open caveat (voids a positive read, kept honest): the frozen-random arm ran a SHORTER stream than the
protected and naive arms in both runs (n_tasks_control 80 vs 240 in the base run; 1000 vs 3000 in the
grind). The control therefore has not been stressed to the full stream length where divergence is
largest. This is a genuine confound in the control, not evidence for the substrate; Studio reruns all
three arms at matched length before any survival claim.

## Probe dependency (task is non-degenerate)

The anchor task is the 6-way visual identity label, which the atlas shows is linearly decodable from
the frozen ViT-L latent (linear_acc 1.0, chance_floor 0.1586, decodable yes). Retention is therefore
measured on a signal that is present, not on noise. Difficulty calibration within the run confirms the
same: peak_mean_anchor_acc is 0.8889 (base) and 1.0 (grind) against chance 0.25, so the anchor heads
learn a real signal before the stream erodes it. The forgetting is a retention effect, not a floor
effect.

## Why it is an asset

The divergence over naive-sequential is real and grows with stream length, which is the headline the
experiment wanted. But the matched-substrate control refuses the stronger claim: replay plus EWC does
not beat an equally-sized frozen-random substrate on retention, so the mechanism is not shown to need
the protected substrate. Showing this (rather than reporting the naive gap alone) is the control
working. The shorter-control caveat is logged as the one open thread that a matched-length Studio rerun
must close.

```yaml
exp_id:            ex13_long_stream
title:             replay plus EWC beats naive on a long stream but ties a matched frozen-random substrate on retention
hypothesis:        replay plus EWC retention over a long domain-incremental stream needs the protected substrate, not merely a same-size substrate
null_hypothesis:   retention is flat in stream length within the seed spread, or every mechanism degrades identically (protection only delays the same collapse)
baseline:          naive-sequential (protected beats it: divergence +0.2499 at 240 tasks, +0.5278 at 3000) and the frozen-random-substrate control (protected ties it)
ablation:          frozen-random-substrate arm, matched size; at 240 tasks anchor bwt -0.1945 vs protected -0.1667 (gap 0.0278); at 3000 tasks anchor bwt -0.2222 vs protected -0.2222 (exact tie)
metric:            bwt
probe_dependency:
  factor:          identity
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       proof/atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes
  acc_above_chance: 0.8414
encoder_scale:     L
seeds:
  n:               3
  sem:             null
  sign_stability:  divergence sign stable over stream length (+0.2499 at 240, +0.5278 at 3000); control tie stable across both runs
provenance_tag:    structured-synthetic
result:            protected-minus-naive anchor bwt +0.2499 at 240 tasks and +0.5278 at 3000; protected-minus-frozen_random gap 0.0278 bwt at 240 tasks and 0.0 at 3000; survives_frozen_random_control False in both runs; open caveat, control stream shorter (80 vs 240; 1000 vs 3000)
taxonomy_category: 3
verdict:           DOWNGRADE-TIE
badges:            [substrate-blindspot]
raw_run_id:        runs/pre_studio/ex13_long_stream.json (240 tasks) and runs/pre_studio/ex13_long_stream_grind.json (3000 tasks)
repro_level:       R1
```

## What this closes

This closes the long-stream forgetting-curve claim at cpu-now scale: the protected arm's advantage over
naive-sequential is real and grows with stream length, but it is not shown to need the protected
substrate, since a matched-size frozen-random substrate ties it on retention. The single open thread is
the shorter control stream, which a matched-length Studio rerun of all three arms must close before any
survival claim.
