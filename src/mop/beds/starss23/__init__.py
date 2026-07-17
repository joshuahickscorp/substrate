"""STARSS23 event-formation bed: the first real Stage 3 experimental lane.

This package implements the adversarially verified recipe in docs/ESCS_DEEP_RESEARCH.md against
synthetic STARSS23-shaped fixtures. It is additive-only mechanics infrastructure. Every artifact it
produces is hardcoded to activation_allowed=false, scientific_promotion=false, and
independent_scientific_confirmation=false. On synthetic data the producer refuses any verdict above
mechanics-ok, so the bed can never open a stage gate.

The hard ESCS constraints, enforced in code and in tests:

- only the candidate gate is trained (<= 4096 parameters plus a few-KB online state);
- the front-end featurizer is a frozen zero-trained-parameter deterministic DSP;
- candidate and every control run at the same total inference compute under a ~6e10 FLOP ceiling with
  the same paired seeds; the gate's amortized training cost is charged in full-lifecycle accounting;
- the referee is byte-sealed onset F1 at a DCASE plus or minus 200 ms collar with strict point-wise PR;
- statistics are an exact sign-flip permutation over five paired seeds (min one-sided p = 1/32);
- the clip is the experimental unit, so claims are worded "consistent with", never "demonstrates".

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

BED_ID = "starss23_escs_event_formation"
BED_SCHEMA = "mop-starss23-escs-bed/v1"

# Byte-identical to the ladder contract claim scope so a receipt minted from this bed cannot widen it.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# The named prior null this bed sits under. Matches stage_ladder.STAGE3_FORCING_NULL exactly so a
# would-be confirmation could only ever attach to the real Stage 3 bar. Synthetic data cannot clear it.
STAGE3_FORCING_NULL = (
    "gen0 first-mechanism nulls (X0 event-formation strong null; CM7 familywise-corrected null)"
)

# The full-lifecycle compute ceiling every arm must stay under (per-arm total FLOPs).
FLOP_CEILING = 60_000_000_000
