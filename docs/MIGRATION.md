# This drop: v2 to the one-Odyssey target

The current v2 public module owners and test definitions are retained. `workspace.py` and `research.py` are new. Physical manifestation is consolidated into existing `handoff.py`, not a second overlapping packaging subsystem. No SQL schema migration is required; new namespace records use the existing transactional event store.

`odyssey-plan-v2` accepts an optional explicit `cognition` profile. Newly generated plans include the one-program target and workspace; legacy plans without it retain legacy mechanism-study status and do not silently inherit stronger claims. Four added arms are explicit factors, not mutations of existing signed plans. Changing source, plan, context schema or artifact identity invalidates old run qualifications.

Existing database content remains readable, but neither old test results nor historical evidence qualifies this release. New `cycle-feedback`, `attention`, `metacognition`, `cognitive-state`, `searches` and `inquiries` namespaces preserve their provenance. Frozen older entities require explicit continuation/qualification before new learning; no automatic host-authority migration occurs.

The first source refactor's 209 archived test obligations still remain in their inventory. This drop also retains every v2 test definition and adds unrun contracts; none is reported as passing. Keep the v2 archive for rollback and evidence comparison. The former research narrative is superseded by the explicit single-Odyssey target but remains available in that archive.

## Earlier migration history

# V1 shell → V2 Odyssey kernel

## This is a breaking consolidation

This release is a smaller developmental research kernel. It is not a claim that every predecessor public API, method library, media/3D integration, deployment wrapper or historical research test remains runnable unchanged. Reducing the whole active tree below 10,000 lines was accomplished through semantic consolidation and scope reduction, not merely line folding.

The predecessor source ZIP remains the immutable reference for old source and tests. The full v2 distribution separately carries every historical evidence file unchanged; its results concern the implementations and conditions that originally produced them. Those files do not qualify the new kernel.

`refactor/MIGRATION.json` records every predecessor module and test path with its digest, proposed owner/disposition and lack of established equivalence. `refactor/TEST_OBLIGATIONS.jsonl` inventories the predecessor test definitions. These are an audit trail, not an assertion that renaming an old test proves the new behavior.

## Ownership and disposition

| Predecessor area | V2 owner and treatment |
|---|---|
| Durable state, manifests, receipts and portable entities | `store`, `core`, `handoff`; new SQLite schema and stricter separation of host/evaluator state. |
| Developmental control, epistemology, beliefs, self-model, goals and competence | `cognition`; unified behavior-changing state rather than disconnected managers. |
| Scalar/causal/program representations | `language`; one bounded typed IR and explicit causal equations. Not every old typed operation is preserved. |
| Plan-only Odyssey and duplicated phase declarations | `odyssey`, `worlds`, `statistics`; actual gated runner plus clear unimplemented frontier. |
| External model/provider roles and capability qualification | `organs`, `sandbox`, `core`; pinned/priced broker and independent authority contracts. |
| Portable training/evidence exchange | `handoff`; licensed development-only export and proposed NR/NX bridge. |
| Media, vision/audio/3D, specialized product surfaces, broad method libraries and old delivery campaigns | Retired from this active kernel. Preserve source in the predecessor; reintroduce as small independently qualified adapters only when an experiment needs them. |
| Prior tests, including unresolved `substrate.deliverables` imports | Preserve in predecessor and inventory obligations. Do not retain misleadingly runnable test paths or pretend their behavior was re-proven. |

## Moving existing entity state

`migrate-v1` accepts the predecessor's serialized `substrate-product-v1` portable bundle, not an arbitrary directory, source ZIP or SQLite file. Supply the verified bundle document pin. The importer checks the top-level document, manifest digest, receipt ordering/hash chain and checkpoint. It does not import or execute predecessor code.

All legacy logical state and receipts are archived in quarantined namespaces. Project goal/next-action text can be projected into blocked projects for review. Old skill qualification, beliefs, host permissions and organ authority do not silently become new-schema verified capabilities. The migration result explicitly reports zero automatically inherited qualified skills/host permissions and unestablished semantic equivalence.

```sh
substrate migrate-v1 --db runs/migrated.sqlite --input path/to/v1-bundle.json --pin <verified-document-sha256>
```

This creates a new database and never overwrites the source. A partially failed migration is not a valid continued entity; preserve the failed artifact for diagnosis and retry into a new destination after review. Individual legacy objects over the new object bound need a reviewed segmentation transformation before admission. Successful hash verification is not evidence that every historical claim remains true.

## New-schema continuity

Use `bundle` to produce an entity-only archive, keep its returned manifest pin, and use `restore` on a new host. Frozen state remains frozen. Review host-local trust and independently requalify bindings before continuing the same identity with a signed `continue-entity` receipt. Use `fork` for an experimental branch; do not use it to pretend two concurrently writable copies are one uniquely owned entity.

The full project ZIP is a source/evidence distribution, not an entity checkpoint. Do not pass it to `restore` or `migrate-v1`.

## Validation status

Only source-level parsing/compilation, simple binding/import inspection, static inventory and archive integrity checks were performed during this edit. No tests were collected or run. No source-only inspection establishes behavioral equivalence. Qualification requires a separately authorized environment, tested oracles, tested isolation, real pinned organs, preregistered hypotheses and executed evidence.
