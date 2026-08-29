# Substrate architecture

Substrate is one installed Python package, `src/substrate`, with one public
command, `substrate`. The repository contains a frozen research framework, its
independent verification fabric, and a separate non-executing product
foundation. Campaign names describe sealed scientific scopes; they are not
multiple competing runtimes.

## Runtime core

`runtime.Substrate` retains the canonical state transition:

```text
perceive → attend → select → run perspectives → arbitrate → decide
→ remember → self update → consolidate → adapt → checkpoint
```

The runtime composes memory, perspective, metacognition, world modelling,
plasticity, continuity, safety, and identity-like state. The structural v4
runtime extends those existing stages rather than adding a second cognitive
engine. `world.StructuralWorld` owns structural models, alternatives, causal
edges, transitions, constraints, invariants, mappings, interventions,
counterfactuals, revisions, inquiry receipts, and validation history.

One structural model supports:

```text
predict · intervene · counterfactual · map representation · explain
compare alternatives · revise · narrow scope · checkpoint and restore
```

Intervention severs declared normal causes. Counterfactual execution changes
one declared premise while preserving background structure. Representation
alignment is inferred from randomized structural constraints rather than hidden
shared identifiers.

## Frozen campaign fabric

Campaign modules are deterministic adapters around the core, not alternate
product entrypoints. They own their declared generators, controls, splits,
seeds, budgets, receipts, and classifications:

- v2–v5 and Genesis preserve historical developmental and structural scopes.
- Genesis II, Nous Closure, Final Revision, the tangible sandbox, and Odyssey
  preserve later sealed scopes and their explicit activation boundaries.
- `verification.py` independently recomputes sealed records, attacks declared
  invariants with mutations, and refuses mismatched evidence.

The v4 fabric separates cheap mechanism canaries, the moderate pilot, the
principal DAG, raw receipt/checkpoint handling, and independent verification.
Its maintained authorities are arranged as:

```text
ops/configs/substrate/v4/                 frozen generators, splits, DAG
evidence/substrate/v4/                    sealed classifications and results
evidence/artifacts/substrate/v4/          reports, review package, raw archive
runs/substrate/v4/                         mutable local receipts/checkpoints
```

The same ownership rule applies to the other campaigns: committed evidence
and retained reports live under `evidence/`, while new run output belongs in
ignored runtime namespaces. Historical predecessor paths are resolved only by
the migration authority in `substrate.evidence`, which maps them to the
canonical checkout for comparison without rewriting the historical record.

## Product foundation

The `substrate product` surface is a separate planning and state format. It
records portable entity state, specialist requirements, source plans,
capability packs, cache attestations, and future sandbox plans. It does not
launch a campaign, container, browser, downloader, model, worker, or external
action. Workers, if implemented in a future phase, must return untrusted output
through quarantine and one authoritative assimilation writer.

The product contracts are documented in [`docs/product`](product/ARCHITECTURE.md)
and its linked entity, pack, source, sandbox, security, and portability
documents. They are not evidence for the scientific classifications above.

## Repository ownership

| Area | Authority |
| --- | --- |
| `src/` | Installed Python package and the small native policy package. |
| `tests/` | Unit, invariant, campaign, and independent-verification tests. |
| `ops/configs/` | Frozen configuration and campaign inputs. |
| `ops/tools/` and `ops/operations/` | Audits, validators, and operational helpers. |
| `docs/` | Current architecture, reproduction, scientific status, product contracts, and archive index. |
| `evidence/` | Sealed classifications, proof ledgers, receipts, and retained reports. |
| ignored `runs/`, `artifacts/`, `data/`, and caches | Mutable local execution state, never scientific authority. |

There is no second active source tree under an old root name. Historical Git
trees and retained evidence may still mention predecessor paths; those names
are provenance data and are handled through the explicit migration map.

## Safety and identity

Activation is a constant `false`. Structural actions are internal proposals or
deterministic sandboxed simulations. Checkpoint restore refuses corrupt model,
causal, mapping, alternative, and specialization state. Body and tool changes
preserve owned identity and structural state. No classification, report,
fixture, or product plan authorizes uncontrolled external action or implies
consciousness, sentience, personhood, life, or moral status.
