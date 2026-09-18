# The compact developmental entity

## One durable organization, replaceable instruments

The canonical entity is not an LLM session. It owns an identity and lineage, a constitution that does not grant itself execution authority, scoped beliefs, prospective predictions, admitted experiences, procedures, candidate cognitive material, causal models, competence, projects, failures and a mutation history. A neural organ is an explicitly priced external instrument with an independently qualified role and host binding.

The implementation is organized into fourteen source modules:

| Owner | Responsibility |
|---|---|
| `core.py` | Canonical data, bounded values, scope, cost vectors, body facts, scoped signatures and local trust. |
| `store.py` | SQLite transactions, content-addressed objects, delta history, entity export/restore/forks, job fencing, effect reservations, compact outcomes. |
| `language.py` | Closed typed program IR, shared-expression DAG compilation, bounded scalar induction, candidate subtree mining and structural causal equations. |
| `cognition.py` | Operational epistemology, evidence admission, competence, routing, procedure/world qualification, projects, curriculum and continuation. |
| `worlds.py` | Six versioned hidden-parameter worlds and the independent evaluator's private state and scoring protocol. |
| `sandbox.py` | Bounded IR worker and explicit OCI execution boundary. |
| `organs.py` | Priced neural broker, strict provider-neutral chat transport, evaluator client and proposed Hawking representation bridges. |
| `odyssey.py` | Study factors, source-bound authorization, dependency graph, developmental/frozen phases and acquisition accounting. |
| `statistics.py` | Complete signed-score validation and fixed-horizon, history-paired analysis. |
| `handoff.py` | Migration, training export, growth accounting, tensor inventory, whole-organization export/restore and qualified native images. |
| `workspace.py` | Viewpoint-bound representations, recurrent capacity-limited broadcast, routing/retrieval consumers and metacognitive feedback. |
| `research.py` | One Odyssey, full-target evidence ledger and prospective apprenticeship contracts. |
| `cli.py` | Explicit operational commands; no experiment runs by default. |
| `__init__.py` | Package identity only. |

The closed IR worker needs only the standard library. Cryptographic implementations are loaded by the authority paths rather than imported merely to execute arithmetic. The source provides no arbitrary-code `eval`, model-controlled shell, plugin import, tool dispatch or self-edit path.

## Epistemic state that changes behavior

`considered`, `imagined` and `believed` are not supported knowledge. Belief admission checks independent signer scope, exact subject, producer separation, evidence identity, freshness and declared conditions. Knowledge admission additionally requires a named domain standard. The standards themselves must be independently reviewed; a certificate cannot create a philosophically infallible oracle.

Reopening a belief propagates through registered dependencies and suspends dependent beliefs, materials and skills. Witness or qualification authority that expires or is revoked can make an old skill unavailable. Repeated failures update mechanism-specific competence rather than allowing failures of a previous LLM to permanently discredit a newly qualified native procedure.

The self-model is operational but limited. It estimates competence/calibration, tracks recent failures, proposes a registered-family curriculum and affects routing. It does not infer subjective experience, generate unrestricted goals or fully model its neural organs. The curriculum heuristic is not an optimal Bayesian information-gain solver.

## Native cognition is a bounded executable description

A candidate program has typed inputs, an expression tree and explicit structural limits. Compilation validates the tree and shares repeated subexpressions in a DAG. Known qualified procedures use a bounded in-memory compiled-program cache; no model is called to rediscover the same program.

Promotion requires verified training witnesses and a separately signed qualification on fresh inputs, or a clearly labeled exhaustive finite Boolean domain. Qualification is host-, scope- and artifact-bound. Runtime guards stay active even in self-model ablations. The native interpreter is not a JIT, universal program synthesizer, theorem prover or Hawking runtime. Its closed DAG can now be packaged as a qualified native image; whole-organization NR-like packaging remains a distinct contract.

The scalar enumerator searches a finite human-designed vocabulary under candidate/time budgets. Material mining recognizes repeated subtrees and parameter/variable roles across distinct qualified numeric programs. Acquired templates feed bounded native search before primitive enumeration, as well as candidate context where appropriate. This is structural role normalization, not arbitrary semantic equivalence or a general learned search grammar. The no-material arm disables acquired-template search as well as context use. New procedures still pass the original witness and qualification boundaries.

Structural causal models support explicit acyclic equations and interventions. Counterfactuals reuse supplied exogenous values; there is no general abduction from observed data and no learned physical/perceptual world model. The simple causal task family provides an initial intervention experiment, not a proof of general causal understanding.

## Storage and authority boundaries

Actor and evaluator databases have different roles and are separate files owned by different operating principals. Content addressing protects integrity and supports deduplication; it does not encrypt private records. Entity deltas are committed with an expected-head check and linked into an auditable history. Freeze changes are transactional. Fork creation and inheritance are a single transaction with explicit parentage.

A portable entity bundle includes its reachable manifest, logical records, provenance and history. It does not include service credentials, evaluator keys/seeds, run leases or other entities. Restore requires a manifest pin, validates bounded archive members and checks history/materialization. A restored frozen entity stays frozen until an explicit continuation or branch operation.

`continue-entity` preserves identity only after a signed handoff declaring that the old writer stopped. `fork` creates a distinct identity and intervention lineage. These are operator-enforced single-writer contracts, not a distributed consensus or global uniqueness service. Moving to a new host invalidates host-bound eligibility until that capability is requalified. Continuity of project content is not automatic continuity of every capability.

Object size is bounded at 4 MiB; score/data exports are sharded. Database payload quota is not the whole on-disk footprint. WAL and SQLite page overhead are reported separately. No automatic garbage collector, distributed state replication or safe restart of every partially completed external effect is claimed.

## Physical sandbox

The native worker is a closed interpreter, not a security boundary for arbitrary Python. Main experiments additionally require a digest-pinned locally present OCI image, no network, no host mounts, non-root execution, read-only filesystem, dropped capabilities, no new privileges, bounded memory/CPU/PIDs, a restricted temporary filesystem and bounded process I/O/time. Client timeout also triggers explicit container cleanup; uncertain cleanup requires host review.

The container image/runtime, host kernel, evaluator implementation, verifier keys and orchestration code remain in the trusted computing base. Same-host Docker does not prove freedom from kernel escape or protect evaluator secrets from the host administrator. The evaluator's separate OS identity and filesystem policy must be tested under the independently reviewed isolation gate.

Native qualification batches amortize container launch overhead. The measured receipt includes validation, launch and cleanup; it is not a kernel-only hardware timing. Current physical reporting has real unknowns: no GPU/FPGA discovery, energy meter or hard system-wide inference-resource enforcement has been implemented.

## Hawking: prepare now, do not invent a runtime

The user's Hawking materials distinguish mutable, inspectable portable NR from an immutable, machine-bound NX derived under an accepted behavior contract. The bridge preserves that distinction. A Substrate IR procedure is not relabeled as a neural NR, and a JSON manifest is not treated as a loadable neural executable.

The implemented external path is:

`verified development → licensed/grouped experience shards → fitting review → mutable model or NR → capability/retention freeze → Gravity → accepted NR → machine-bound NX → independent organ qualification → shadow attachment`

`prepare-training` exports only frozen entities' currently verified development experiences with an explicit source/license allowlist. It excludes qualification and final-evaluation labels. All arms belonging to one developmental history stay in the same fitting split; control-arm predictions are retained rather than deduplicated across arms. Source licenses are operator declarations to review, not licenses inferred from a URL. The export currently materializes eligible rows before writing 128-record deterministic gzip shards, so it is not an unlimited streaming data lake.

`hawking-request` prepares a content-bound request with source model/revision/tokenizer, role behavior/state contracts, retention/transfer suites and optional target machine. It never invokes a trainer or edits a frozen NX. No Hawking ABI has been assumed, no model has been downloaded, and no NR/NX execution claim has been established. Mapping these bridges to the actual Hawking service is a site-specific integration with independent qualification.

The user materials consulted were `HAWKING_MASTER_LONG_HAUL_V2.md` (NR and NX sections), `H-ROADMAP(8).md` (teacher query policy, training-before-Gravity freeze), and `HAWKING_NR_NX_RESEARCH(1).md` (representation versus executable and scope of earlier observations). They informed the contract; this refactor did not inspect or mutate the live Hawking repository.

## What cannot be inferred from this release

The implemented interfaces do not establish years of acquired competence, general concept formation, autonomous scientific discovery, sentience, AGI, a successful body/organ migration campaign, or faster inference. Broad perception, mathematics/formal-system integrations, external source study, robotics, distributed identity, neural weight fitting and accelerator lowering remain distinct frontier work.

The test source is a new contract suite for this smaller kernel. It has not been collected or executed. The prior test suite and retired APIs are preserved in the predecessor artifact, not falsely counted as active compatibility.

## Integrated perspective and one-program evidence

The entity's epistemic records and the active workspace have different purposes. Epistemic admission determines what can be relied on; the workspace determines which currently relevant representations reach consumers. The latter cannot promote a proposition to knowledge. Viewpoint and scope bind representations, and imagination is not treated as evidence.

The worker calls `CognitiveCycle.decide`, passes projected context to `Broker.query`, and admits cycle feedback only after `Entity.assimilate`. The same broadcast affects retrieval and the next procedural-routing round. No-workspace/no-recurrence/no-self/no-perspective/no-broadcast conditions remain explicit interventions. The current no-perspective condition ablates labels reaching the organ, not a learned theory-of-mind network.

The full target is semi-conscious general intelligence, owned by `configs/odyssey.json` and `OdysseyLedger`. A single program may contain independently locked comparison matrices and different instrument adapters. Its mechanism instrument and broad target must not share an automatic pass flag. Evidence profiles preserve contradictory and inconclusive findings, and require interpretation on the same qualified artifact lineage.

## Whole organization and size

`phenotype` separates current namespace categories and historical reachable payload, with byte-identical objects counted once. Physical process/device residency and external model size are unknown without instruments. `growth` produces a storage delta, not an intelligence scalar. There is no destructive knowledge compactor or automatic evidence erasure.

`organization.nr.json` binds complete `entity.bundle` state, record index, runtime contract, optional learned-statistics Safetensors and explicitly supplied assets. Freeze and final-head checks prevent a silently mixed evolving snapshot. Restore validates wrapper, bundle, state index, assets and tensor metadata; the entity remains frozen and host authority is not inherited.

Native image lowering serializes actual compiled instruction regions from currently qualified skills. Loading checks the source IR, exact instruction tape, host/runtime and a separate independent image qualification. New or modified procedures need a new artifact. Mutable entity records do not silently modify the frozen image. Neither a whole-organization manifest nor a closed-DAG image establishes compatibility with Hawking's actual ABI or whole neural-graph lowering.
