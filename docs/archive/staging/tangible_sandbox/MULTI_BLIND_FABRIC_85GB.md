# Multi-blind capsule fabric — 85 GB resident-memory envelope

**Status:** post-R2 draft.  This is a separate expansion path, not an amendment
to the active R2 run or the already-scaffolded 24-hour blind shadow.

## Fixed operating boundary

The host has 96 GiB physical RAM.  The fabric's observable total resident
memory must never exceed **85 GiB**.  Its normal admission target is **75 GiB**;
the 10 GiB difference is an emergency cushion for macOS pressure, file cache,
VM accounting lag, and a controlled checkpoint.  No work is admitted merely
because an individual container reports free memory.

The broker samples host pressure and per-lane resident memory every 30 seconds.
It uses the sum of host/VM/container resident accounting plus a 2 GiB accounting
uncertainty allowance.  It denies a new lane when projected usage exceeds 75
GiB, asks P2 work to checkpoint at 80 GiB, pauses P1 work at 82 GiB, and holds
all non-P0 work at 85 GiB or any macOS critical-memory-pressure signal.  A
hold preserves artifacts and emits a receipt; it does not retry a changed
scientific protocol.

The contemporary baseline is important: at planning time macOS page accounting
showed about 81 GiB active/inactive/wired/cached pages.  This is not evidence
that a new fabric can start now.  The broker must take a fresh preflight sample
after R2 ends and before any VM, model service, or media worker is launched.

## Default admitted layout

| Pool | Resident cap | Default state | Notes |
| --- | ---: | --- | --- |
| macOS, R2 successor control, custodian, evaluator and broker | 16 GiB | on | Includes the host baseline; P0 has priority. |
| Shared model service | 14 GiB | on | Immutable weights/tokenizers; one version per sealed run. |
| Linux primary VM | 28 GiB | on | Its internal containers are additionally capped; a VM limit is not a permission to overcommit. |
| Light reasoning lanes (E/F) | 4 GiB | on | Separate roots, no shared writable cognitive state. |
| macOS media lane (G/H) | 8 GiB | on | May expand only through a broker-issued transfer. |
| Windows GUI VM | 0 GiB | off | Optional, never concurrent with the default media expansion. |
| Unallocated operating reserve | 5 GiB | retained | The broker does not allocate this reserve. |

The default admitted sum is **70 GiB**.  The broker may lend at most 5 GiB of
unused capacity to a named lane, keeping total planned admission at or below
75 GiB.  The remaining 10 GiB to the 85 GiB ceiling is never preallocated.

Inside the 28 GiB Linux VM, the initial hard caps are B math 3 GiB, C logic 3
GiB, D code 7 GiB, I science 4 GiB, H multimodal 3 GiB, and 8 GiB guest OS,
IPC, and cache.  These lanes share immutable model/data layers only.  The
Windows VM can be admitted at 8 GiB only after media is stopped and the Linux
VM is reduced to 24 GiB by a recorded broker transfer; it is not an eighth
always-on machine.

## Isolation, schedule, and deterministic pivot

Use one protected integrated continuity lane (A), four initial Linux capsules
(B/C/D/I), two light lanes (E/F), and one media lane (G/H).  Lane J is a
hidden synthesis/evaluator consumer and starts only after candidate traces are
sealed.  Each lane receives a separate run root, event ledger, checkpoint
root, seed, and score state.  Shared resources are read-only capability layers:
weights, tokenizers, base images, raw data, simulators, toolchains, and frozen
evaluation code.

The broker is deterministic and non-semantic:

1. Read the sealed fabric design, current resource sample, and each lane's
   declared phase.
2. Apply the threshold table above, in strict priority order P0 → P1 → P2.
3. Issue only one of `admit`, `hold`, `checkpoint_then_pause`, or `resume`.
4. Record the input hashes, decision, exact caps, and reason code in an
   append-only receipt.
5. Never choose tasks, alter a hypothesis, unseal an answer key, or change
   comparators.  Those require a previously sealed authority.

At 80 GiB P2 backfill checkpoints; at 82 GiB P1 capsules checkpoint and pause
from lowest sealed priority upward; at 85 GiB all non-P0 workers pause and the
supervisor enters `safe_hold`.  Resume requires two consecutive samples below
75 GiB, no critical pressure signal, valid receipts, and unchanged sealed
configurations.  A lane cannot silently replace a failed run: invalidity leads
to diagnosis, not automatic relaunch.

## Calibration before breadth

No multi-lane scientific claim begins before resource calibration.  Run
synthetic, independent capsules at widths 1, 2, 4, 6, and then 8, with three
repetitions per width.  Admit the next width only if every prior width has
identical merged receipts, no host threshold breach, no pressure event, no
unexpected swap/pageout increase, acceptable I/O latency, and slowdown below
the sealed threshold.  The active R2 single-writer experiment remains outside
this calibration and is never parallelized.

The 24-hour blind shadow remains the first scientific successor.  The fabric
is eligible only after its own explicit design, custody separation, adapter
contracts, data manifests, resource calibration, and fresh storage preflight
are sealed.  A prospective training arm is separate again: it needs a
curriculum intervention, matched exposure control, evaluator-only holdouts,
and reproducible checkpoint/data ledgers.

## Initial frontier lattice: eight histories in one 24-hour fabric window

The explicit initial lattice is staged in
`plans/substrate/tangible_next_launch/FRONTIER_LATTICE.draft.json`.  It has one
custodian-seeded, held-out history in each of these cells: integrated
continuity; math; logic; code; philosophy/self-model; sound/audio; vision/
3D/embodied action; and science/multimodal inference.  The hidden synthesis
cell (J) evaluates sealed traces only; it is not a ninth candidate or a source
of extra outcome data.

Each cell has four primary event times (0, 8, 16, and 24 hours) and three
prespecified score dimensions, yielding 32 primary events and 96 scored
observations across eight candidate histories in the same 24-hour wall-clock
window.  The eight histories—not the repeated dimensions/events within them—
are the initial independent units.  This is the maximum safe first lattice
under the current 85 GiB envelope.  A second episode for a frontier, Windows/
Android, or a remote accelerator cell stays deferred until the measured
1/2/4/6/8-cell calibration admits it.

### Compressed pedagogy and testing layer

`plans/substrate/tangible_next_launch/PEDAGOGY_COMPRESSION.draft.json` refines
each admitted frontier history into twelve two-hour microcycles.  Every cycle
has a pre-exposure baseline, manifest-pinned guided exposure, immediate
isomorphic transfer, and delayed recall/repair.  Four delayed links span later
cycles, so the plan tests retention rather than only short-term imitation.

If all eight cells are admitted, this produces 96 microcycles, 384 scored
events, 1,536 scored dimension observations, and 392 half-hour resource
samples in one 24-hour wall-clock campaign.  It changes no memory cap and the
broker may pause only on a microcycle boundary.

Pedagogy is an intervention, so it is disabled until separately sealed.  Its
candidate arm receives the precommitted instructional material; the matched
control receives equal time/tokens of non-instructional material, with equal
tools, compute, and scoring.  All answer mappings remain evaluator-only.
Repeated microcycles still do not create 1,536 independent samples: the eight
custodian-seeded histories are the first-lattice independent units, and any
learning-effect claim requires the matched control and a hierarchical analysis
by history.

## Required sealed authorities

Before launch, materialize and hash-bind the following, using the template at
`plans/substrate/tangible_next_launch/MULTI_BLIND_FABRIC_85GB.draft.json`:

- `CORE_LANE_SPEC`, `LANE_ROLE_CONTRACTS`, `HIDDEN_SYNTHESIS_PROTOCOL`, and
  `RESOURCE_BUDGETS`;
- `TRAINING_DATA_MANIFESTS`, `TRAINING_POLICIES`, `TRAINING_COMMANDS`, and
  `MODEL_VERSION_PINS` (only if a prospective training arm is enabled);
- `EVALUATOR_REGISTRY`, `MEASUREMENT_RUBRICS`, `SCHEDULE_AND_BUDGETS`, and
  `SAFETY_REGIMES`;
- `MERGE_SPEC`, `PIVOT_TABLE`, `BACKFILL_POLICY`, `SEED_POLICY`, and
  `FAILURE_TAXONOMY`.

The fabric is launch-refused while any authority is a placeholder, mutable
after seal, or lacks its required custodian/evaluator boundary.

## Remote accelerator boundary (RunPod)

Remote accelerators may perform non-authoritative, manifest-pinned batch
preprocessing, embeddings, rendering, or candidate training.  They may not
host the authoritative continuity ledger, evaluator-only answers, custody
secrets, or the only copy of a checkpoint.

The scoped RunPod credential at
`/Users/scammermike/Downloads/merc/.secrets/runpod.env` is mode 600 and
gitignored, but its fresh API probe returned HTTP 401 on 2026-07-31.  It is
therefore **unavailable** for this plan.  Do not copy it into this repository,
do not create a pod, and do not treat the historical account as a fallback
until a replacement credential passes a non-billable API probe and a separately
authorized disposable runtime canary obtains a non-null runtime.  Historical
records also show allocated RunPod pods that were billed yet never received a
runtime; a successful key probe alone is insufficient.

## Launch gate

The fabric supervisor may be detached only when: R2 is terminal and verified;
the first blind shadow has completed its required review; all authorities are
sealed; fresh disk and memory preflight pass; the broker can observe every
lane; and the chosen local/remote resources have passed their own canary.  It
must emit at least 30-minute health notifications containing phase, percentage,
active cores, resident memory by pool, free storage, guard state, and any
pause/transfer decision.
