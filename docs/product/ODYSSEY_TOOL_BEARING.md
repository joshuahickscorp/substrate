# Odyssey tool-bearing arms

Odyssey frontiers are named for formal, software, media, spatial, and causal
work.  Arms are no longer model-only: they propose **only** closed, registry
operations; a shared broker validates; a portable sandbox executes; the product
cache quarantines; a local verifier admits; and the arm may consume **only**
admitted results.

## Flow

```text
model proposes tool_proposals
        │
        ▼
broker validates
  · operation ∈ REGISTRY_OPERATIONS
  · operation ∈ task declared surface ∩ frontier surface
  · candidate/control budget digests identical
  · no evaluator-only tokens, no cross-lane cache access
        │
        ▼
sandbox executes (typed argv only)
  · mounts: inputs (ro) / work (ephemeral) / output (quarantine)
  · no arbitrary shell, no network (except source.read_cached of admitted bytes)
        │
        ▼
product cache quarantines outputs
        │
        ▼
local Ed25519 verifier admits
        │
        ▼
arm may read admitted digests only
```

## Registry

See `src/substrate/odyssey_tools.py` (`REGISTRY_OPERATIONS`).  Frontiers A–H
each declare a minimum surface implied by the frontier name
(`FRONTIER_OPERATIONS`).

## Parity

Candidate and control share:

- the same closed registry
- the same `ToolBudget` (cpu, memory, wall, max output, max calls, attempts=1)
- the same host tool inventory and revision digests
- the same deadline policy

The **only** intended causal difference remains endogenous developmental memory
in the arm adapters.

## Protocol v2

Unchanged: `MAX_OUTPUT_TOKENS = 64`, `GENERATION_NUM_PREDICT = 1024`,
`ARM_TRANSPORT_ATTEMPTS = 1`.  Tool calls are outside the substantive answer
envelope.

## Evidence

Public canary: `artifacts/substrate/odyssey7d/tool-bearing-canary/TOOL_BEARING_CANARY.json`

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from substrate.odyssey_tools import run_frontier_canary; print(run_frontier_canary(Path("."))["all_admitted"])'
```
