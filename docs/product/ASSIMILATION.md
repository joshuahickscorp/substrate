# Evidence and assimilation

Experience becomes development only through a controlled evidence path:

```text
approved source or worker result
  → quarantine
  → verifier and evaluator checks
  → provenance-bearing evidence packet
  → single entity writer
  → receipt, checkpoint, and reversible lineage
```

Workers do not write entity state. A source cache promotion is not, by itself,
an entity update. The current v1 bridge requires the active apprenticeship,
source policy, selected pack scope, sealed acquisition plan, and a typed
`CacheAttestedEvidenceAuthority`. That authority revalidates a signed,
locally trusted cache receipt for the exact content and plan before the single
writer can record sanitized source evidence.

It does not yet provide evaluator-backed competence updates, a production
evidence-packet format, or an autonomous assimilation engine.
