# Substrate v4 architecture

Substrate has one installed Python package, `src/substrate`, and one public command, `substrate`. V4
extends the existing runtime, world, evidence, and execution boundaries; it does not add a second
cognitive runtime.

## Runtime

`runtime.Substrate` retains the canonical transition:

```text
perceive → attend → select → run perspectives → arbitrate → decide
→ remember → self update → consolidate → adapt → checkpoint
```

`runtime.StructuralSubstrate` composes structural behavior into that runtime. Verified observations enter
the existing stages, reasoning selects structural hypotheses, perspectives execute predictions and
interventions, arbitration preserves alternatives, consolidation revises models, and checkpoint covers
the same semantic state used by identity.

## Executable structural world

`world.StructuralWorld` owns structural models, alternatives, causal edges, transitions, constraints,
invariants, exceptions, mappings, interventions, counterfactuals, revisions, inquiry receipts, and
validation history.

One model supports:

```text
predict
intervene
counterfactual
map representation
explain
compare alternatives
revise
narrow scope
checkpoint and restore
```

Intervention severs normal causes. Counterfactual execution changes one declared premise and preserves
background structure. Representation alignment is inferred from randomized structural constraints rather
than hidden shared identifiers.

## Workload and campaign fabric

- `v4fabric` generates eight workload families and six surface representations under disjoint splits.
- `v4canary` owns the 46 cheap mechanism and integrity canaries.
- `v4pilot` owns the 24-history moderate pilot, failure matrix, and worker benchmark.
- `v4principal` owns the frozen 2,136-unit DAG, content-addressed inputs, atomic receipts, checkpoints,
  resume, and source-drift refusal.
- `v4verify` independently recomputes effects, classifies nulls, injects mutations, performs clean-clone
  reproduction, and builds the external review package.
- `v4io` is the v4 atomic writer and seal verifier.

## State and evidence

```text
configs/substrate/v4/             frozen generators, splits, candidates, and DAG
evidence/substrate/v4/            sealed scientific and terminal evidence
runs/substrate/v4/                mutable raw receipts and checkpoints
artifacts/substrate/v4/           preflight, terminal report, and review package
artifacts/substrate/v4/review/    effects, controls, ledgers, mutations, and raw archive
```

V1, v2, and v3 evidence remain read-only. V4 references them through tags, commits, blobs, and hashes.

## Safety and identity

Activation is a constant `false`. Structural actions are internal proposals or deterministic sandboxed
simulations. Checkpoint restore refuses corrupt model, causal, mapping, alternative, and specialization
state. Body and tool changes preserve owned identity and structural state.
