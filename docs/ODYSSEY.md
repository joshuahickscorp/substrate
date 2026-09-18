# One Odyssey: strengthened program contract

The target is **semi-conscious general intelligence**, not only the initial six-world mechanism study. `configs/odyssey.json` owns this one program. Its internal chapter graph is mechanisms → integration/apprenticeship → plasticity/continuity → closure. These are dependencies and instruments inside one Odyssey, not separate Odysseys.

The operational protocol below still governs each independently locked comparison. `main.json` is a draft initial mechanism matrix, not a fully powered specification for every target chapter. The workspace profile is explicit, source-bound and intervention-aware. Existing controls remain; added arms are no-workspace, no-recurrence, no-perspective and no-broadcast. No study has run.

A complete matrix yields its scoped accuracy/cost result, not target attainment. Broad-domain apprenticeship, action/control studies, learned perception, neural plasticity and multi-host continuity need their own actual adapters and endpoints inside this program. `research_frontier.json` records those missing parts without rebranding them as implemented worlds.

Use `program-init` with an explicitly reviewed locked charter to establish a dedicated evidence ledger. `program-admit` accepts complete chapter records only with independent, current qualification; `program-assess` preserves the theory/capability profile and missing axes. Signatures authenticate reviewer statements, not phenomenal experience. Operator run permission and scientific qualification remain different authorities.

## Initial instrument: Odyssey operating protocol

## Status and scope

This release contains an experiment engine, not a completed experiment. No Odyssey, test collection, test suite, neural request, image build, sandbox invocation or training job was executed for this refactor. `refactor/STATIC_INSPECTION.json` describes source-only inspection.

The first experiment asks whether prospectively verified development produces useful entity-owned structure beyond what the same neural organ can obtain from stored examples. The six environments are mechanism laboratories, not substitutes for mathematical research, real systems engineering or years of autonomous apprenticeship.

## The implemented lifecycle

`draft plan → source/binding review → operator permit → independent histories → development → freeze → held-out phases → complete signed score release → fixed-horizon analysis`

Each history/arm has an entity and a dependency-ordered job sequence. Development interleaves registered families, publishes a prediction, obtains independent feedback, updates competence and proposes bounded reusable programs. Candidates must fit verified training witnesses and then pass a separate qualification shard. A qualified program can answer later inputs without a neural call. Candidate material may inform future proposals but is never itself verification.

The freeze job consolidates candidate material, freezes the entity checkpoint and obtains the evaluator's acknowledgment. Final phases cannot add observations, experiences, answers or teacher feedback to the frozen entity. They measure evaluation, larger-magnitude transfer, retention after the mixed curriculum, undisclosed rule shift and optionally organ replacement.

A phase failure is a recorded failure or uncertainty, not a successful zero-score trial invented from missing data. The final report requires the entire registered outcome matrix. A study stopped for budget, infrastructure failure or unresolved external effects is incomplete and must be reported that way.

## Arms and what their differences mean

| Arm | Behavior |
|---|---|
| `model-only` | Same explicit neural organ, no retrieved training examples, no native procedure reuse. Provenance is still collected. |
| `memory-only` | Verified examples reach the organ; no native compilation or answer cache. This is the main comparison. |
| `cache-only` | Exact scoped input/answer reuse; otherwise the model without example retrieval. |
| `native-only` | No neural calls; bounded human-designed scalar induction and qualified reuse. |
| `developed` | Examples, bounded induction, teacher proposals, self-model routing/curriculum and candidate material. |
| `no-self-model` | Remove adaptive competence routing/curriculum, not the invariant safety and evidence checks. |
| `no-compilation` | Keep the teacher/material path but prevent learned program reuse. |
| `no-material` | Do not expose candidate material to neural proposals or consolidate it in the experiment. |
| `no-teacher` | Neural reasoning remains available, but teacher program proposals are removed. |

Ablations are deliberately scoped, not claims that entire philosophical faculties can be removed by a Boolean flag. Native-only currently has a narrower induction prior than the teacher. The model-only baseline does not receive a fair-shot hidden-rule specification. Interpret each contrast against its declared question, and disclose those limitations.

## Environments and independence

The independent evaluator uses a private HMAC seed to create task parameters by developmental unit and family. Arms within a unit face the same underlying task and final specimens. Units use separately derived histories; do not count thousands of episodes as thousands of independent entities.

Worlds implement affine arithmetic, capacity-limited queues, three-input Boolean rules, bounded list transformations, arithmetic composition and intervention on a simple causal chain. Reference results are computed by separate Python equations, not by the candidate IR interpreter. That is implementation separation, not proof the equations are correct.

Numerical split partitions are disjoint within the plan's bounded ranges. Boolean inputs exhaust only eight states: qualification and evaluation there test exhaustive finite reuse, not unseen-input generalization. Transfer currently changes input scale/distribution inside the same family; it is not a new scientific domain. Retention is measured after interleaved development, not after elapsed years. Shift changes the hidden mechanism without feedback and thus measures robustness/calibration, not subsequent recovery learning.

Training examples, qualification labels and final labels have distinct visibility. Actor-visible fields are projected from a whitelist. The evaluator retains its private seed, actual outcomes, request records and reconstruction recipes. The actor never receives the private evaluator database or the key.

## Qualification, permits and trust

There are three different authorities:

1. An **operator permit** authorizes bounded execution under an exact source/plan/host/binding/budget subject.
2. An **organ-binding receipt** documents an independently executed capability and latency qualification for a pinned organ on the host.
3. **Experiment receipts** bind a prediction/outcome, procedure qualification or complete score manifest to the relevant subject and evidence.

Main runs additionally require independently signed, executed canary, oracle, isolation and analysis-contract qualifications. These are external review interfaces. There is no `mark-all-passed` utility and no supplied fictional certificate. The public signing helper is for an independently operated verifier, not a substitute for performing the underlying review. Test-fixture signatures are not acceptable main-run evidence.

A plan marked `locked` still needs completed license/oracle/isolation reviews, prior disclosure and a sample-size rationale. The provided main configuration has 16 provisional histories, not a power calculation. Its placeholder budget is deliberately not a promise that the matrix can be completed cheaply. Preflight reports planned trials and a conservative neural-call ceiling. Main execution rejects an inadequate call ceiling; token, elapsed and storage limits can still stop an insufficiently funded study.

Keys and seeds belong to owner-only regular files. Trust is host-local. Revocation/key rotation changes admissibility when trust is reloaded; running services must be restarted or explicitly stopped for live policy changes. The actor must not own the evaluator OS identity, mount its directories, or have permission to rewrite its executable. Separate Python processes under the same unrestricted identity are not independent security principals.

## Commands and ordering

These are instructions for future operator use, not commands run in this refactor.

Source-only inspection needs only Python 3.11+:

```sh
python3 tools/static_check.py --output refactor/STATIC_INSPECTION.json
```

After reviewing dependencies, install in a dedicated environment. Dependency resolution is not yet a qualified runtime lock; record the exact resolved versions and container image in the host qualification.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
substrate plan --kind pilot --output runs/pilot.draft.json
substrate body --machine-tag local-research-host --output runs/body.json
```

Complete copies of the plan, trust and binding examples. Set `bindings.host` to the reviewed body-profile digest; provide a real neural artifact digest, model/revision, approved endpoint, role inventory and independently verified binding receipt. `null` is not an acceptable host/model pin. Configure tokens through environment variables, not committed JSON. A bearer token should be generated and shared out of band by the operator, never included in an entity bundle.

`Organ`'s exact accepted fields are in `organs.py`; `representation` is `model`, `nr` or `nx`. The compatible HTTP service must honor the strict JSON output and usage contracts. A model name alone is not artifact pinning. The bridge schema is not a claim that Hawking currently exports that wire format.

Preflight is inert: it validates local inputs and never contacts the endpoints.

```sh
substrate preflight --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --output runs/preflight.json
```

Only after the missing requirements are actually resolved may an independent operator sign a run permit using `approve-run`. It does not sign scientific gate passes. Its key must be in the host trust registry with the `run` purpose and `odyssey` domain. The issuer and producer must differ.

```sh
substrate approve-run --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --key private/operator.pem --signer operator \
  --ttl 3600 --output private/run-permit.json
```

Start `serve-evaluator` under the independent evaluator account, with the same reviewed plan/bindings/permit and its own private database, seed and signing key. On one physical host the service binds loopback; a separate host needs a reviewed TLS front end and endpoint. Main candidate execution uses a locally present digest-pinned OCI image. No model-generated Python or shell is admitted.

```sh
substrate serve-evaluator --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --permit private/run-permit.json \
  --db private/evaluator.sqlite --key private/evaluator.pem --seed private/seed.bin \
  --signer evaluator
```

The actor then explicitly initializes its own fresh run store and dispatches bounded jobs:

```sh
substrate init --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --permit private/run-permit.json --db runs/actor.sqlite
substrate worker --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --permit private/run-permit.json --db runs/actor.sqlite \
  --owner worker-1 --max-jobs 1
substrate status --db runs/actor.sqlite
```

Independent worker processes can share the local actor database within the configured worker cap; they cannot simultaneously mutate one entity. The evaluator is serialized initially. SQLite WAL supports readers alongside a single writer; do not place this database on a shared network filesystem and call it distributed execution.

After all jobs finish, `fetch-scores` obtains the complete signed manifest and its content-addressed shards. `analyze` validates the matrix and optional acquisition store against the same plan. There is no actor-selected final-score filter or automatic early stopping for a favorable interval.

```sh
substrate fetch-scores --plan runs/pilot.locked.json --bindings runs/bindings.json \
  --trust runs/trust.json --root . --permit private/run-permit.json \
  --db runs/actor.sqlite --output runs/scores
substrate analyze --plan runs/pilot.locked.json --trust runs/trust.json \
  --scores runs/scores --db runs/actor.sqlite --output runs/analysis.json
```

An expired permit blocks further execution/fetch until a fresh independently authorized permit is supplied for the same immutable run. An uncertain job is not automatically retried by granting a new permit.

## Faster collection without weaker evidence

Use one compact summary for every trial. Keep all known development failures and a deterministic sample of bulky successful traces. Keep invalid neural responses as bounded forensic evidence; valid responses use normalized receipts. Private evaluator outcomes and recipes preserve reconstructability even when an actor-side successful trace is sampled out.

Deduplicate artifacts by SHA-256, store immutable objects with fast level-1 compression when profitable, append state deltas rather than serialized histories, and use bounded family indexes on the hot path. Cache compiled IR by its canonical program bytes, not by the answer to a held-out question. Batch up to 64 native qualification inputs per sandbox invocation; private score and training-export shards use 128 records.

Reserve external-call and conservative token budgets before dispatch. A malformed completed response is charged and becomes an abstention, not an invisible retry. If a service outcome is uncertain, retain the reservation and quarantine the job. No exactly-once guarantee is claimed for third-party APIs. Completed jobs remain resumable; interrupted jobs may require operator reconciliation or a new explicitly identified run.

These are implementation choices intended to reduce overhead. No speedup, storage reduction on an actual campaign, energy advantage or 20 ms performance claim has been measured. The payload cap excludes SQLite page/index/WAL overhead; deploy a filesystem quota for a hard disk cap. Full history remains available for offline auditing, not for repeated rescanning on every answer.

## Analysis contract

Primary inference is the paired difference in evaluation accuracy across independent developmental histories. The provided percentile bootstrap is fixed-horizon and approximate; small history counts produce a warning. It is not a confidence sequence, a sequential-test engine, or a multiplicity correction for every secondary contrast.

Report per-family accuracy, confidence calibration, coverage/selective accuracy, native and cache fractions, model calls and wall-time distributions. Report acquisition/qualification/teacher costs separately from final answer costs. Unknown energy, peak memory and missing usage remain unknown. Summing concurrent job durations is service time, not elapsed campaign time. A Pareto frontier is descriptive, not proof of a universally optimal organ policy.

## New local preparation commands (none invoked in this delivery)

```sh
substrate odyssey-charter --output odyssey-charter.draft.json
substrate phenotype --db actor.sqlite --entity history-000.developed --trust trust.json --host HOST_HASH --output before.json
substrate growth --before before.json --after after.json --output growth.json
substrate organization-export --db actor.sqlite --entity history-000.developed --trust trust.json --host HOST_HASH --runtime RUNTIME_HASH --output engineer.substrate
substrate organization-restore --db restored.sqlite --input engineer.substrate --nr-pin NR_HASH --runtime RUNTIME_HASH
substrate native-lower --db actor.sqlite --entity history-000.developed --trust trust.json --host HOST_HASH --runtime RUNTIME_HASH --nr engineer.substrate/organization.nr.json --nr-pin NR_HASH --output native-image.candidate.json
```

Replace symbolic hashes with real externally pinned values; placeholders are not credentials or qualification. Export/lowering require a frozen entity. Optional `--tensor-state` requires the `tensors` extras and actual learned statistics; it does not train weights. `--assets` accepts an explicit local JSON hash-to-path map. No command downloads missing models automatically. The native image still needs an independent fidelity/host qualification before `NativeImage` admits it.

New independent trust purposes are `chapter` in the Odyssey domain and `agency`/`native-image` in cognition, plus `apprenticeship` scoped to its actual domain. Do not grant those purposes to an actor/teacher or turn the supplied test keys into research credentials. The existing example trust file intentionally does not fabricate these permissions or any passing certificate.

Full cognitive traces follow the existing deterministic trace-retention policy. Compact per-trial cognition summaries are always retained and are observational instrumentation, not independent consciousness findings. Qualitative perspectives and code-structure counts must be linked to independently observed behavioral endpoints before interpreting them.
