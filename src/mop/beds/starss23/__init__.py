
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
