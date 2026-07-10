# P6 continual million-event audit

## Scope

This audit concerns the operational P6 frontier named
`frontier_continual_million_event_learning` in the extended-compute matrix. It is distinct from the
registered `p6_meaning_without_symbols` philosophy experiment, which already has its own executable
pilot. No registry row was renamed or reclassified.

## What already existed

| Component | Existing value | Missing for the frontier |
|---|---|---|
| EX13 | Registered forgetting-curve contract, replay and no-replay comparison, anchor retention, effective rank | Disk stream, event cursor checkpoint, fresh-init control, future learnability, deletion |
| EX15 | Long-stream plasticity and rejuvenation traces | Million-event data plane and exact interruption replay |
| PR9 | Per-seed and per-arm leg files | Atomic event-level cursor, strict content identity, deletion-aware replay state |
| ReplayBuffer | Bounded in-memory replay and sampling | Portable state identity, source event references, deletion provenance |
| Wave E0 | Typed event and entity references, exact branches, append-only lifecycle with rollback and deletion | Disk-backed repeated-event source and long-stream endpoints |

The prior components were useful and remain authoritative in their scopes. None alone satisfied the
frontier requirement, and their presence did not justify a hardware classification.

## Smallest missing scaffold now implemented

- Fixed-width, disk-backed event chunks with a SHA-256 chain across every record.
- Atomic chunk publication and a manifest that refuses spec, byte, order, or chain drift.
- Exact arm checkpoints bound to stream, profile, prefix digest, cursor, learner state, and replay state.
- Abrupt and gradual transition schedules from one deterministic generator.
- Replay, compute-matched no-replay, and fixed-topology fresh-init controls.
- Shared Wave E0 `EventRef`, `EntityRef`, and `LifecycleJournal` primitives.
- Retention, acquisition, future-learnability, stale-memory, deletion, and exact resource metrics.
- A preregistered 384-event smoke profile with an explicit no-heavy guard.

The smoke runner is not a new scientific registry harness. EX13 and EX15 remain the registered
contracts. The new layer supplies their previously missing disk stream and resume mechanics.

## Current evidence boundary

`proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json` is mechanics-only. It executes 384 events per stream,
both schedules, and all three controls. It loads no model weights and requests no accelerator. A
positive smoke metric is not evidence for a continual-learning advantage.

## Remaining full-run gate

The implementation blocker is closed. The unexecuted gate is scale and replication:

1. Admit progressive 10,000, 100,000, and 1,000,000 event rungs through the resource governor after
   the heavy lane is free.
2. Run abrupt and gradual schedules for replay, no-replay, and fresh-init controls.
3. Use at least five independent seeds and independently replay metrics from immutable checkpoints.
4. Demonstrate interruption recovery at the million-event rung.
5. Separate retention from replay volume and update count before any mechanism claim.

No measured hardware boundary exists. A Mac Studio is not earned by this scaffold or its smoke run.
