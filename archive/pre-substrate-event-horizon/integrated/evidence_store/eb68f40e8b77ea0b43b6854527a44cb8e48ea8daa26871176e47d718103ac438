# Data custody and recovery

Canonical root: `/Users/scammermike/Downloads/mop-data`. Nothing under a worktree.

| corpus | retention | state | canonical path |
|---|---|---|---|
| uci_har | principal_active | present | `/Users/scammermike/Downloads/mop-data/har/UCI HAR Dataset` |
| speech_commands | principal_active | present | `/Users/scammermike/Downloads/mop-data/speech/speech_commands` |
| speech_features_cache | derived_rebuildable | present | `/Users/scammermike/Downloads/mop-data/speech/speech_feats.npz` |
| pamap2 | secondary_active | ABSENT | `/Users/scammermike/Downloads/mop-data/pamap2/PAMAP2_Dataset` |
| harth | secondary_active | ABSENT | `/Users/scammermike/Downloads/mop-data/harth/harth` |
| starss23 | historical_reproducibility | present | `/Users/scammermike/Downloads/mop-data/starss23` |

## What went wrong

Two worktrees pointed absolute data paths at each other. Removing one destroyed the only local copies of
PAMAP2 and HARTH. Nothing sealed depended on them, which is why it went unnoticed, and that is the reason the
guard exists rather than a reason it does not need to.

## Recovery

Every corpus records a re download command. For the two lost corpora:

```
python3.12 -m mop.temporal.runs.custody_run recover pamap2
python3.12 -m mop.temporal.runs.custody_run recover harth
```

## The guard

`mop.temporal.custody.guard` refuses removal of any directory that uniquely holds raw data, a non rebuildable
cache, a split authority, a principal checkpoint or unindexed evidence. A publicly recoverable corpus can be
released only through an explicit override, never by default. 10 guard mutations are
sealed in `MOP_DATA_CUSTODY_MUTATIONS.json`.
