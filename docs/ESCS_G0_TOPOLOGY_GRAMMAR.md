# ESCS G0 topology grammar scaffold

`src/mop/escs/topology_grammar.py` is a mechanics-only, disabled-by-default contract for testing
whether ESCS can reorganize its topology. It does not modify a live `CoalitionRuntime` and supports
no capability or scientific-promotion claim.

The committed `configs/experiment/escs_g0_topology_grammar.json` declares a finite nine-operator
grammar: actor-slot addition and retirement, routing-subscription addition and removal, peer-edge
addition and removal, factor-scope addition and removal, and a deterministic swap to a registered
inactive spare. Every operator is currently disabled, `activation_enabled` is false, the grammar is
not frozen, and no freeze authority exists.

The same artifact records the complete proposed G0 construction vocabulary from the ESCS plan:
bounded distribution/table/deque/factor-graph/recurrent state; eight finite local operator primitives;
six bounded construction-mutation families; explicit node, depth, state, activation, message, and edge
caps; and hard prohibitions on generated code, recursion, undeclared operators, silent schemas, and
ungoverned mutation. `implementation_complete` is false. Capturing that vocabulary plus the isolated
counterfactual reference evaluator below is not an audited, activation-capable implementation, so this
scaffold cannot satisfy `G0_FREEZE` yet.

Topology proposals are pure transformations between immutable, content-addressed snapshots. A
transaction binds the exact ordered mutation sequence, base and proposed topology digests, declared
lifecycle work, and retained-state delta. Assessment replays the sequence and enforces mutation,
operator, topology-delta, work, and retained-state caps.

The grammar also binds the canonical permissive perspective registry by content digest. Unknown or
excluded candidates are structurally rejected, and a newly active slot remains blocked while its
registry entry is disabled. Inert candidates and registered spares can therefore be assembled without
silently gaining activation authority.

Future `G0_FREEZE` may enable an exact subset only in a new self-hashed grammar revision bound to a
separate immutable freeze-authority artifact, after every construction primitive has an audited bounded
implementation. A caller must independently verify that artifact and
explicitly join it during assessment. The static v1 genotype assessor still cannot authorize shadow
execution; the reference evaluator may only produce isolated counterfactual traces. Shadow authorization
requires a separate audited authority/integration revision as well as an enabled candidate registry entry.
Even then, shadow authorization is not factual authority:
canary evidence, a chassis commitment, a rollback snapshot equal to the base topology, and a later
consequence remain separate requirements. The grammar can never grant scientific promotion.

This realizes the campaign's light/build-lane requirement: topology structure and adversarial
contracts can be prepared now, while every actual mutation remains inert until X1 or X2 evidence
permits a reviewed finite `G0_FREEZE`.

## Finite genotype representation

`src/mop/escs/g0_genotype.py` now gives each actor a content-addressed finite genotype: bounded typed
state slots, a cycle-free DAG drawn only from the declared G0 operator vocabulary, bounded typed message
edges, explicit outputs, declared operations, and encoded/state/message budgets. Nested generated-code,
recursion, silent-schema, and undeclared-operator requests are rejected before assessment.

The DAG validator is iterative, so adversarial depth cannot turn a declared cycle check into Python call
stack recursion. Input-node order is preserved as identity-bearing operator semantics; set-like state,
message, output, and record collections are canonicalized. Every node executes at most once in the static
per-activation budget; every declared state slot is referenced; every graph sink is an explicit output; and
every message edge has exactly one emit-node owner and permits at most one encoded message per activation.
Any `recipient_cap` must equal the explicitly declared edges. Parameter schema references must bind to state
or message schemas used by that same node. This is a local identity join, not proof that a schema has an
implemented codec. These rules make operation, output, and message totals conservative and prevent aliases
from hiding work.

The static assessor binds the supplied perspective registry digest to the exact digest named by the grammar,
checks the candidate's disabled activation bit, and binds both authorities into the assessment. In addition
to the individual state/output/genotype limits, it checks their conservative aggregate against the grammar's
per-actor byte cap. A valid genotype is structurally describable under the current scaffold, but shadow and
factual activation remain refused because candidate activation and genotype shadow authorization are
disabled, the construction implementation is incomplete, G0 is unfrozen and disabled, and no freeze
authority exists.
This closes the representation gap without pretending that the operator implementations or an adaptive
topology result have been established.

## Counterfactual reference evaluator

`src/mop/escs/g0_evaluator.py` supplies deterministic reference semantics for all eight declared
operator primitives. It is deliberately a counterfactual trace evaluator, not an ESCS actor runtime:
it cannot access a live coalition, issue a chassis commitment, mutate topology, apply a factual effect,
or grant scientific promotion. Every result and staged message records those refusals explicitly and
is content-addressed for independent replay.

The evaluator joins the genotype, grammar, exact registry digest, and a self-hashed v1 evaluator contract.
Those authorities plus the immutable input and activation-start state snapshots determine the branch ID.
The branch suffix is a full SHA-256 over the complete authority/input/state tuple; it does not truncate
individual authorities. Successful traces embed the frozen input and initial-state snapshots, and
`verify_g0_counterfactual` reruns them against the exact genotype, grammar, and registry.
Every and only DAG roots receive external inputs. Ordered multi-parent fan-in is preserved as an ordered
JSON list; typed emit nodes may stage one message per canonically ordered declared edge. Each message binds
its node, branch, input, state, genotype, grammar, registry, and evaluator-contract provenance. Recipient
references must use the `actor:` namespace and a finite set of payload forms is checked, while schema IDs
remain local identity joins—not claims that an external codec exists.

State use is typed: affine updates may stage bounded recurrent or vector state; retrieval and table
operations use categorical-table state; temporal accumulation uses a bounded deque; and graph aggregation
uses a typed factor subgraph. Rollout, constraint, and message operations cannot silently bind state. All
reads observe the immutable activation-start snapshot, all writes are staged, and a slot has at most one
writer. A recurrent affine update explicitly concatenates its prior state; vector-distribution state uses
replace semantics; stateless affine nodes declare `state_mode=none`. Independent node naming therefore
cannot create an undeclared read-after-write dependency. Every primitive has an exact mode-specific
parameter schema, so misspelled or ignored controls fail rather than silently changing semantics.

`used_operations` is a deterministic logical-operation proxy, not a Python instruction or energy count.
It counts such units as scalar arithmetic, inspected records/items/edges, transitions, table entries,
active temporal-window entries, and message transformations. JSON byte movement is recorded separately as
input, state, output, payload, and envelope bytes; it is not relabeled as abstract work. Temporal operators
touch only the active bounded window, and rollout output sizing is incremental rather than quadratic.
Numbers are finite IEEE-754 values with integers restricted to the exactly representable range; nonlinear
reference activations are identity, ReLU, and deterministic softsign. Iterative input validation caps JSON
depth, node count, container width, and scalar size before the recursive standard encoder is reached. Iterator
frames keep validator memory proportional to nesting depth. Cycles and non-string object keys are rejected.
Noncanonical attempts become identity-incomplete conservative refusals and require a caller-supplied stable
`attempt_id`, so distinct append-only attempts cannot collapse into one accounting receipt.

Every successful result contains self-hashed per-node cost rows joined to the aggregate counters. Structural
search uses `attempt_g0_counterfactual`: a late refusal returns an immutable receipt charged at the full
declared genotype envelope, so failed candidates cannot make work disappear. This conservative charge may
overestimate a failure but cannot undercount admitted actor work. The mechanics proof separately binds the
exact evaluator source bytes; the evaluator-contract digest binds the semantic revision in each trace.

This reference implementation makes the finite vocabulary executable enough for mechanics tests and
construction search in an isolated branch. It does not make `implementation_complete` true and does
not satisfy `G0_FREEZE`: production scheduling, transactional state integration, rollback/canary paths,
and experimental evidence for adaptive construction are still separate requirements.
