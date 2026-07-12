# EDCM-1 v3 preregistration audit

Status: implemented and intentionally unexecuted while the P5 heavy lane owns
local compute. The config authority is frozen; the external implementation
authority must be finalized after the last scoped code/test/document edit.

## Scope and claim boundary

EDCM-1 v3 tests one synthetic mechanics claim: after an independent
complementarity gate, can hard event-triggered dispatch of heterogeneous,
stateful computations preserve always-on utility while reducing deterministic
abstract work? Even a favorable receipt is barred from promotion to a claim
about general intelligence, consciousness, biological equivalence, or trained
system superiority.

The JSON envelope (retaining the historical `.yaml` filename) hashes its
complete semantic payload. The authority digest is also compiled into the
implementation; official execution requires the repository config, contract
id, and exact digest. Changed configs
require `--exploratory` and cannot produce an official scientific verdict.

Official execution also requires the canonical manifest at
`proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.implementation-authority.json`.
That self-hashed manifest binds the config authority, review status, and exact
byte length/SHA-256 receipt of every scoped EDCM config, module, runner, test,
and audit file. Its digest must also be supplied independently to official run
and verify commands; the manifest cannot authorize a replacement for itself.
The pinned digest is part of checkpoint and receipt identity.
Exploratory execution must instead name an explicit manifest whose mode is
`exploratory`.

## World, cognition, and messages

The evaluator owns rotations, physical actions, niche/noise labels, walls, and
absolute position. Cognition sees only local command-space blocking, relative
goal, public action/reward history, and novelty channels. Tune, gate, and
held-out generators use disjoint seed offsets. Rejection sampling guarantees a
connected layout and bounded shortest path. Every transition, including the
terminal one, retains its actual visible successor observation.

The proposer roles are mechanically different:

- reactive spatial control uses current geometry and a small action trace;
- episodic retrieval uses provenance-bearing visible transitions and freshness;
- short-horizon planning learns recent `(visible state, action) -> successor`
  distributions and rolls them out.

The contradiction verifier is relational: it requires multiple proposals and
can endorse, contradict, or abstain, but never proposes an action. Hard dispatch
means only selected proposers run or update. At most one separately accounted
verifier round follows the initial proposer round.

Messages bind integrity, stable referents, source events, age, producer state
digests, and producer work. The producer work claim is sampled only after the
complete producer state has been serialized and hashed, making state-size
changes visible to accounting.

## Controls

Held-out controls include always-on, tuned best proposer, equal-budget
recurrent, balanced homogeneous, two marginal round controls, and one intact
coalition control.

Periodic and shuffled round-matched controls preserve each marginal
`(role, round)` call count. They do **not** claim to preserve within-tick
coalitions. The separate `shuffled_coalition_matched` control permutes whole
`ActivationRecord` objects and must preserve each episode's multiset of
initial coactivations and verifier rounds.

The homogeneous control replaces only initial proposer roles with rotating
copies of the gate-selected proposer. A persistent relational verifier remains
in every matched extra round. Initial proposer calls and verifier calls are
matched and checked separately.

A clean fixed-schedule replay must reproduce every event-arm action and utility.
The verifier additionally replays every stored action sequence through the
deterministic world and recomputes return, success, utility, and niche values;
rehashing a forged action sequence is therefore insufficient.

The recurrent comparator performs meaningful reservoir sweeps and online TD
updates only. Its tune budget is the measured abstract work of the tune-split
event arm, not a fixed allowance. Before bootstrapping, TD encodes
`transition.after`, charges that successor sweep, and uses the successor hidden
state. Held-out work uses the measured held-out event budget; no padding exists.

## Causal interventions

Direct intervention capture is limited to the preregistered
`intervention_episodes`. Evidence is stored as compact count/sum/mean/range
records plus a digest of the underlying ordered values; unbounded raw vectors
are excluded from proof and checkpoint artifacts.

No-message, proposer-link lesions, wrong-planner messages, and verifier lesions
are evaluated from common state and a common produced packet. The delay claim
uses a true two-tick channel assay:

1. clean, delayed, and lesioned branches clone the same origin;
2. the delayed and lesioned branches withhold the origin planner message at
   tick `t`;
3. at `t+1`, the delayed branch receives that exact age-one origin message
   while its current planner message is withheld;
4. the lesion branch receives neither planner message; and
5. paired two-tick values compare clean versus delay and delay versus lesion.

Origin/fork hashes and exact delayed-message identity are invariants. This is
not stale-message substitution in an unrelated trajectory.

Restoration clones the planner-lesioned common state at the restoration tick
into continued-lesion and restored-link branches. Only their paired future
difference is called restoration gain. At visibly nonzero noise ticks, a paired
quiet observation differing only in novelty must produce the same activation.

## Gate and verdict

All five complementarity-gate seeds finish before held-out routing. The gate
uses disjoint tune and gate splits and requires oracle headroom, unique wins for
every proposer, niche advantages, an off-ceiling best single, verifier benefit
on disagreement, and negligible agreement effect. Recurrent candidate keys are
canonicalized before aggregation, so JSON-loaded checkpoints and fresh rows
cannot disagree merely because mapping insertion order changed.

After a passing gate, the held-out positive pattern requires the preregistered
utility/work bounds, positive paired margins over every control, Pareto
contribution, selective communication effects, two-tick delay weaker than
lesion, restoration, noise invariance, change selectivity, all matching
invariants, and the required direction in every seed.

## Accounting, artifacts, and verification

Abstract work separately records scalar operations, comparisons,
nonlinearities, table reads/writes, hashed bytes, and serialized bytes. Each arm
publishes component totals and one-at-a-time 0.5x/2x weight sensitivity. A
conservative saving compares the event arm's worst sensitivity scenario with
always-on's best scenario. CPU time, wall time, energy, and RSS remain a
post-v3 non-verdict benchmark.

Before computation, a prospective bound estimates all episode records against
90% of both proof and checkpoint byte envelopes. Every checkpoint and receipt
also has an actual serialized-size guard. Receipt/checkpoint size is rejected
from filesystem metadata before content is read; after parsing, canonical
encoded size and bytes must equal the on-disk artifact exactly. Unknown
top-level receipt/checkpoint, gate/held-out row, or arm-summary keys fail
closed. Each config, manifest, receipt, or checkpoint validation derives all
semantic checks and any source receipt from the same single byte snapshot;
validation never reopens a path to assemble one authority claim. Checkpoints
bind config authority, frozen implementation authority,
runtime, exact seed prefixes, and hashes of every gate and held-out row; writes
fsync contents, atomically replace, and fsync the directory. Config, manifest,
checkpoint, and receipt paths must be pairwise distinct, preventing a mutable
artifact from overwriting an authority input or its own checkpoint.

The final receipt is bound to the exact checkpoint file hash, checkpoint
self-hash, and exact gate/held-out row hashes. Algebraic/schema/replay checks are
diagnostics, not official evidence authority. The official verifier
deterministically reruns `run_gate_seed` for every completed gate-prefix row
and, after deriving gate-selected controls, reruns `run_heldout_seed` for
every completed held-out-prefix row. Each regenerated row must be canonically
identical to the stored row. Thus gate verifier assays, compact causal evidence,
actions, work, schedules, controls, and aggregates are authenticated together.
This intentionally expensive full-regeneration path is mandatory for official
verification. The verifier is in the same implementation, not an independently
implemented second codebase; the external reviewed implementation manifest
freezes which implementation has authority.

Execution states are explicit:

- `partial`: incomplete execution, resumable, `all_ok=false`, CLI exit 2;
- `terminal_scientific_stop`: completed gate failure, not resumable, exit 0;
- `complete`: completed held-out study, exit 0.

Verification exits 0 only after a valid receipt completes its configured
verification mode. Any cap, canonical encoding, schema, authority, checkpoint
join, or regeneration mismatch fails closed and exits nonzero. The lighter
structural-diagnostics-only verifier mode is permitted only by an exploratory
receipt with an explicit nonofficial implementation authority.

### Finalize the implementation authority

After final review, and only after no further edits will be made to the five
scoped EDCM files, generate the external manifest:

~~~bash
PYTHONPATH=src python -c 'from mop.studies.edcm1_event_triggered_coalition import write_implementation_authority; d = write_implementation_authority(); print(d["manifest_sha256"])'
~~~

Any later scoped edit invalidates the manifest. Repeat review and regenerate it
before official execution. The manifest itself is outside the scoped file list,
so writing it does not create a hash cycle. Freeze the printed digest in the
review/release record and export it for both official commands:

~~~bash
export EDCM1_IMPL_SHA256=<reviewed manifest_sha256>
~~~

After the heavy lane clears, intended commands are:

```bash
python scripts/run_edcm1_event_triggered_coalition.py \
  --implementation-authority-sha256 "$EDCM1_IMPL_SHA256"
python scripts/run_edcm1_event_triggered_coalition.py \
  --verify proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json \
  --checkpoint proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.checkpoint.json \
  --implementation-authority-sha256 "$EDCM1_IMPL_SHA256"
```

## Known limits

- Worlds and specialists are hand-designed synthetic mechanisms, not learned
  general-purpose agents.
- Abstract work is not FLOPs, latency, energy, or silicon cost.
- Compact evidence omits raw vectors from artifacts; official verification
  authenticates it through deterministic regeneration of the complete row.
- Five seeds can reject only the preregistered mechanics null.
- No result is claimed until official execution and receipt verification finish.
