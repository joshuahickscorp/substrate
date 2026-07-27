# Terminal deterministic synthesis plan

This is the sole human-readable plan for the 19-unit terminal deterministic synthesis. It preserves the
frozen scientific meaning and stops at the launch boundary. The synthesis performs zero new scientific
trials; it verifies, recomputes, mutates, regenerates, and packages already admitted evidence.

## Admission boundary

Launch is permitted only when:

- the repository and normalized configuration hashes match the sealed authority;
- activation is false;
- the structural audit, scientific certification, mutation suite, rehearsal, and clean-clone checks pass;
- no prior synthesis receipt, duplicate supervisor, stale lock, source drift, or configuration drift exists;
- the operator explicitly invokes `substrate run`.

Verification and regeneration never invoke that command.

## Work graph

The 19 units are declared in `substrate.execution.UNIT_LIST`. The graph covers structural audit,
declarations, temporal control, ontology and epistemology, memory, diversity arbitration, world model,
self model, three bodies and their comparison, plasticity, developmental divergence, entity batteries,
certification, independent recomputation, mutations, and terminal synthesis.

Each unit has one producer and one receipt. A unit becomes ready only when all dependency receipts are
valid. Completion means every unit is terminal; it is not a duration target.

## Determinism and recovery

- Configuration is read once and content-hashed.
- Inputs and manifests are stably sorted.
- Evidence is content-addressed and written atomically.
- Unit receipts are append-only at scientific work-unit boundaries.
- Exclusive claim files prevent duplicate workers.
- Resume adopts valid completed receipts and does not recompute them.
- Source or configuration drift refuses execution.
- Verification is batched after dependency-complete families.
- Logs and indexes are bounded by work-unit identity.

## Resource policy

The synthesis is CPU-only on the current machine. Measurements select one persistent worker with one native
thread. Two workers improved on that baseline by only 7.6 percent, below the declared 15 percent materiality
threshold, while larger pools raised memory and variance.

The generated authority `SUBSTRATE_LONG_RUN_RESOURCE_PLAN.json` records unit count, projected CPU time,
peak memory, disk growth, write amplification, checkpoint cost, restart loss, verification overhead,
mutation overhead, rehearsal cost, and the terminal range.

## Commands

```bash
substrate verify
substrate rehearse
substrate status
substrate run
```

The Execution Forge campaign must not run the last command. It prepares and certifies the boundary, then
stops. That command launches terminal synthesis, not a new scientific experiment.
