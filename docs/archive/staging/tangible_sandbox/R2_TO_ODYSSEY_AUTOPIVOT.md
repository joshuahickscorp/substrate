# Deterministic R2 → Odyssey autopivot

The active R2 remediation is the one permitted automatic mechanical repair:
the full 24-hour schedule runs with trace serialization fixed before launch.
It is not shortened and does not reuse elapsed time or candidate state from the
invalid trace.

After it ends, the autopivot has only these choices:

```text
valid worker result + independent verification
    → bind the verified R2 digest to the Odyssey authority gate

worker failure or verification failure
    → preserve trace, Telegram alert, safe hold; no automatic retry

verified R2 + any Odyssey gate pending
    → prepare-only safe hold

verified R2 + sealed authority + all fifteen Odyssey gates pass
    → one detached Odyssey supervisor may launch
```

The controller cannot choose a curriculum, task, candidate, control, score
weight, duration, or result interpretation. It routes immutable state only.
That gives the program autonomous timing without turning expected results into
an intervention.
