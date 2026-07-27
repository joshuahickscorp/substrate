# Architecture

Substrate has one installed Python package, `src/substrate`, and one public command, `substrate`. There is
no active `mop` package or command.

## Runtime

`runtime.Substrate` owns the canonical live state and the eleven-stage transition:

```text
perceive → attend → select → run perspectives → arbitrate → decide
→ remember → self update → consolidate → adapt → checkpoint
```

The runtime records every stage, records skipped stages with a reason, revises beliefs, updates memory and
self-model state, and creates a content-bound checkpoint. It only records what would be done; no path sets
activation to true.

The model is divided by invariant:

- `ontology`, `epistemology`, `workspace`, `memory`, `world`, `selfmodel`, and `goals` own typed state.
- `perspectives`, `metacog`, `plasticity`, `bodies`, and `temporal_link` implement cognitive mechanisms
  composed by the runtime.
- `experiments` and `method` own experiment declarations, causal validation, admission, result
  classification, and value-of-information ranking.
- `program`, `graph`, `deliverables`, `authority`, `audit`, and `verification` derive and independently
  check evidence.
- `execution` is the single DAG executor, process supervisor, checkpoint/resume authority, and status
  reader.
- `cli` is the only public command surface.

## Configuration

`configs/substrate/config.json`, loaded by `substrate.config`, is the single configuration path. Unknown fields
are rejected. `SUBSTRATE_DATA_ROOT` and `SUBSTRATE_STATE_ROOT` are the only active environment overrides.
The normalized configuration is content-hashed, and activation must remain false.

## State, evidence, and artifacts

`substrate.evidence` is the sole JSON/Markdown writer. It canonicalizes JSON, content-hashes sealed
evidence, stamps the source commit, and publishes atomically.

```text
configs/substrate/config.json      configuration
src/substrate/                     active implementation
tests/substrate/                   active contracts
evidence/substrate/v1/             current immutable evidence
runs/substrate/v1/                 mutable receipts, locks, and checkpoints
artifacts/substrate/event-horizon/ generated migration authorities and reports
proof/                             sealed historical evidence
archive/pre-substrate-event-horizon/ historical source, tests, and documents
```

Historical MOP paths and schemas are read-only and reachable only through `substrate.compat.mop`. That
module cannot execute a historical runtime and does not provide an import alias.

## Execution

`substrate.execution.UNIT_LIST` is the one scientific DAG. Each unit names its dependencies, licensing
evidence, and produced artifacts. An atomic exclusive claim prevents duplicate writers. A unit is complete
only when its receipt is valid. Resume reads receipts from disk and does not repeat completed work.

The stop switch is `${SUBSTRATE_STATE_ROOT:-$HOME/.substrate}/stop`. Status, stop, and resume operate on
the same state model.
