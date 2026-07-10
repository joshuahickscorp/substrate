# Null card / survival card template

Copy this to `NULL_CARDS/<exp_id>.md` and fill every field. The machine-checkable field
list is `null_card.schema.json` in this directory. A survival card is a null card whose
verdict is a rejected null; wins and nulls use this identical template, which is the
point (nulls are shown, not buried). Form per BLACKHOLE.md: no em dashes, no agency,
intelligence, understanding, consciousness, sentience, or self language. The encoder is
frozen and never trained.

`probe_dependency` is REQUIRED. A representational claim with no probe_dependency, or one
citing a factor the atlas shows is not decodable, is INVALID (the claim is void and
becomes a substrate bound, taxonomy entry 3).

```yaml
exp_id:            EX12                      # stable id (E1 | E3 | EX12 | ...)
title:             one line, no agency language
hypothesis:        the mechanism claim, engineering vocabulary only
null_hypothesis:   the pre-declared null this must beat (verbatim from the experiment)
baseline:          the TUNED baseline (name the tuning: cosine decay, optimizer, capacity-matched head)
ablation:          the capacity/LR/seed ablation run and its outcome
metric:            frontier_auc | bwt | adaptation_steps_to_threshold | ece | recall_at_k | probe_acc
probe_dependency:                            # REQUIRED
  factor:          identity                  # the atlas factor the claim depends on
  encoder:         vjepa2_vitl_fpc64_256      # the encoder scale (or all-three)
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes                       # yes | no | marginal (from the atlas row)
  acc_above_chance: null                     # probe acc minus chance floor
encoder_scale:     primary                   # substrate role or registered id; archive-only for history
seeds:
  n:               5                          # >= 3 (the measured sign-stability threshold)
  sem:             null
  sign_stability:  stable at S>=3             # or 'unstable'
provenance_tag:    real-encoder              # natural-video | real-encoder | structured-synthetic | provisional
result:            the numbers with confidence intervals
taxonomy_category: 3                         # 1..10 (Section 10.5) for a null; 'null rejected' for a survival card
verdict:           SUBSTRATE-BOUND           # PUBLISH-POSITIVE | DOWNGRADE-TIE | SUBSTRATE-BOUND | SEED-UNSTABLE | CAPACITY-ARTIFACT
badges:            [substrate-blindspot]     # seed-instability | capacity-artifact | substrate-blindspot | tuned-baseline-tie
raw_run_id:        <run hash> + <config path under runs/>
repro_level:       R0                        # R0..R5 (Section 10.6)
```

## What voids this card (any one of these)

- No `probe_dependency`, or it cites a factor the atlas shows is not decodable, while
  still making a representational claim.
- The baseline is untuned (a strawman cosine, default optimizer, capacity-mismatched head).
- Fewer seeds than the sign-stability threshold (S>=3), or no SEM.
- No `raw_run_id`, or the id does not reproduce the reported numbers within tolerance.
- A provenance tag richer than the cache actually supports (the cache validator refuses it).
- Any sentence drifting into agency/consciousness language (the north_star scanner
  refuses to render it).

## Third-party reruns

External reruns go under `third_party/<exp_id>__<who>.md` using this same schema plus a
hardware string and the delta to the reference. A failed replication is added (never
hidden) with `verdict: REPLICATION-FAILED`; if it overturns a published claim, the
original card is marked SUPERSEDED with a link, never deleted.
