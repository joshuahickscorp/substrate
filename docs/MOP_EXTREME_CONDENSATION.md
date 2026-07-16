# MOP extreme condensation

MOP now treats **25,000 active-repository LOC as the recommended extreme
target**. Unlike a continuously shipped product, MOP is primarily a governed study,
result, and evidence archive. Its permanent operational checkout should therefore
be a small kernel that summons immutable owned packs when a study, verifier, replay,
or historical authority is needed. The target is an architectural constraint, not
permission to weaken the system. The independently measured runtime-kernel stretch
target is 18,000 LOC.

The active live checkout is not the condensation workspace. Generation 1 mechanics,
the consolidated final campaign, recovery v5, both successor horizons, the extension
chain, and the categorized wave still form one live authority chain. Condensation
work stays on an isolated branch. Nothing is merged, installed, or used by those
processes until the whole chain is terminal clean and a release/supersession audit
passes.

## Honest measurement contract

Every checkpoint reports:

- `active_repo_LOC`: physical lines in tracked runtime, campaign-control,
  laboratory, validation, configuration, and build/operations text;
- `runtime_core_LOC`: the independently classified active runtime surface, checked
  against the 18k stretch target;
- `hydrated_owned_LOC`: active LOC plus every checksum-locked external pack;
- `relocated_LOC`: owned source moved into verified packs;
- `eliminated_LOC`: LOC removed by a sealed before/after refactor entry;
- `added_LOC`: growth that offsets elimination;
- `tracked_text_LOC`: all tracked text, including evidence and documentation;
- `surface_LOC`: runtime, campaign, laboratory, validation, evidence,
  documentation, generated-manifest, asset, and operations totals.

The verifier counts every UTF-8 tracked file in its declared surface rather than
trusting a hand-picked source-extension list. Active binary files and active
symlinks fail closed. Very long active lines fail the no-line-packing gate.
Generated campaign manifests are reported as their own data surface because several
are intentionally serialized as one physical line; their bytes and hashes remain
visible and they receive no code-elimination credit.

No checkpoint may claim reduction through minification, packed lines, comment
stripping, renamed extensions, generated source blobs, hidden Git-history loaders,
network-only hydration, or removal of validation/evidence.

The machine-readable authority is
[`condensation.json`](../condensation.json). Repository shape, live bindings, the
pack graph, and run receipts use canonical SHA-256 identities.

## Target architecture

The active repository converges on:

- the MOP runtime and lean ESCS kernel;
- one data-driven campaign state machine;
- immutable receipt, provenance, recovery, and supersession authority;
- an offline-only pack resolver and lockfile;
- compact configuration, build, CI, and operational documentation.

Checksum-locked packs contain:

- research studies, experiments, falsification programs, builders, and registries;
- the complete logical validation and deterministic-replay inventory;
- immutable historical/live-compatible Studio implementations;
- optional encoder, substrate, SANPO, media, and machine-specific backends;
- bulk proof, generated manifests, checkpoints, and authoritative run artifacts.

Packs are source distributions, not opaque LOC tricks. Every payload file records
its path, byte size, physical LOC, and SHA-256. The resolver accepts only a sealed
lockfile and a verified local cache. Normal runtime hydration never uses the network.
The v2 lock and elimination ledger use immediate-parent hash chains. Every
superseded head is retained byte-for-byte under `condensation/history/`; invented
or missing ancestry fails closed.
Only active-code pack LOC contributes to `hydrated_owned_LOC`; evidence,
documentation, and generated-data pack LOC remains visible in hydrated tracked-text
and per-surface accounting. Relocation credit additionally requires a byte-identical
active file from the sealed baseline to be absent from the active checkout.

Genuine deletion credit is separate. [`condensation.eliminations.json`](../condensation.eliminations.json)
must bind baseline inputs, current outputs, exact LOC delta, and validation evidence.
Any missing baseline active path or net reduction that lacks relocation or
elimination evidence fails verification. Elimination entries are bound to the
immutable `full` gate profile; a quick receipt cannot authorize deletion credit.

A validation pack receives no logical-test credit from declarations alone. Its
sealed node IDs must be collected offline from exact manifest-listed validation
files in the cache, collected again from the staged hydration copy, and collected
again from an existing immutable mount. The full release profile then executes the
hydrated suite; collection proves identity, while execution proves behavior.

## Descent ladder

The architecture is designed backward from 25k and verified forward one rung at a
time:

| Checkpoint | Principal work |
|---:|---|
| 250k | Extract research/laboratory surfaces and make registration lazy. |
| 225k | Establish the signed validation pack and declarative parity inventory. |
| 200k | Externalize bulk evidence and generated manifests behind compact indexes. |
| 175k | Replace overlapping successor controllers with one profile-driven state machine. |
| 150k | Consolidate mechanism-family scaffold/runner/bed boilerplate. |
| 125k | Split concrete substrate/backends from stable interfaces and unify CLI plumbing. |
| 100k | Move the remaining study-specific mechanism and evaluator families behind stable contracts. |
| 75k | Reduce the active checkout to runtime, campaign kernel, pack resolver, and operational tests. |
| 50k | Consolidate recovery, receipt, registry, and CLI plumbing into data-driven engines. |
| 35k | Retain only the lean ESCS/substrate interfaces and minimal campaign authority. |
| 25k | Minimal complete operational repository; all studies, history, validation, and evidence are summoned from locked packs. |

The separate 18k runtime-kernel stretch retains only the CLI-reachable kernel,
interfaces, receipts, and pack resolver. It does not exclude required validation,
study code, or evidence from hydrated owned accounting.

A lower checkpoint is rejected only after two materially different architectures
fail for structural reasons, the irreducible subsystem budget is measured, and the
previous green rung is restored and reverified.

## Execution waves

Wave 0 freezes measurement, the empty initial pack lock, the exact live-binding
snapshot, rollback rules, and a reproducible baseline.

Wave 1 runs pack-boundary work in parallel: research, validation, evidence, and
optional backends. Relocation is credited only after offline hydration reproduces
every file and hash.

Wave 2 serializes the shared control-plane change: one campaign engine, receipt
engine, recovery authority, and profile registry. Historical implementations remain
available through an immutable replay pack.

Wave 3 combines the lean ESCS kernel, declarative mechanism families, substrate
interfaces, and one registry-driven CLI, then descends from 100k through 75k, 50k,
35k, and the recommended 25k operational floor.

Wave 4 walks the checkpoint ladder serially. Every green rung receives a commit,
tag, rollback receipt, exact accounting, and draft-PR update.

## Gates

Each release checkpoint must preserve the full logical validation inventory,
scientific semantics, result/proof hashes, deterministic replay, crash-resume,
rollback, fresh offline hydration, and supported capabilities. Formatting, lint,
typing, tests, documentation, and performance gates must be green. A meaningful
throughput regression is investigated rather than hidden by the LOC result.
Repository verification must succeed before hydration or gates are allowed to run.

Wave 0 records two inherited release debts instead of rewriting live authority:
whole-repository Ruff formatting currently identifies 82 files, and whole-repository
mypy reports 42 errors across seven older files. Ruff lint and the documentation
inventory are green, and the new condensation controller is independently
format-, lint-, type-, and test-clean. The inherited debts remain mandatory before
any release checkpoint can activate.

The static live boundary is sealed in
[`condensation.live-bindings.json`](../condensation.live-bindings.json). It is a
minimum frozen set, not permission to change other transitively imported live code.
The operational rule remains stronger: do not activate any condensation commit in
the live checkout before terminal release.

## Commands

Inspect the plan and current shape:

```sh
PYTHONPATH=src python3.12 scripts/mop_condense.py plan
PYTHONPATH=src python3.12 scripts/mop_condense.py measure
```

After the Wave 0 controller itself is committed and the checkout is clean, seal
that exact commit as the immutable origin for all later relocation and elimination:

```sh
PYTHONPATH=src python3.12 scripts/mop_condense.py baseline
```

Verify no-gaming rules, the baseline, pack lock, and all frozen live bindings:

```sh
PYTHONPATH=src python3.12 scripts/mop_condense.py verify
```

Run the complete quick sequence and emit a self-sealed receipt on stdout:

```sh
PYTHONPATH=src python3.12 scripts/mop_condense.py run --profile quick
```

The `full` profile is the release gate. It is intentionally not used as an
activation signal while Generation 1 is live.
