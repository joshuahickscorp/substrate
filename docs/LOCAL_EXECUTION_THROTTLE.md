# Five-hour local execution throttle

The current M3 Pro operational envelope is now 300 minutes with the same 40 GB free-disk floor.
This is a host scheduler, not a scientific-config migration. P4 and P5 keep their existing hashed
10,800-second invocation budgets. When either runner returns its documented resumable exit code 2,
the throttle repeats the exact argv inside the same five-hour operational leg. Completed checkpoint
identities therefore do not change.

The machine-readable policy is `configs/local_execution_throttle.yaml`. Every runnable task declares
its lane, CPU cores, unified-memory authority, MPS peak, forecast writes, atomic-write overhead,
wall cap, checkpoint globs, restart exit codes, dependencies, and exact argv. A declaration also
records the basis for its resource estimate. The policy loader refuses a task above 300 minutes, a
disk floor other than 40 GB, a heavy task without an atomic resume authority, or any drift in the
canonical P4 CPU, P4 clean-MPS, ordered P5 CPU, or ordered P6 CPU argv.

## Admission and dynamic throttling

One heavy lane is absolute. A second lane may only be `cpu`, `network`, or `light`; it is admitted
only with the stricter second-lane thresholds. A second MPS owner is never allowed because macOS does
not expose trustworthy global per-process Metal allocation telemetry to the Python supervisor.

Each sample records:

- normalized one-minute CPU load, instantaneous CPU utilization, and logical CPUs;
- VM available memory plus the native `memory_pressure -Q` result;
- swap use;
- current-process MPS allocation, driver allocation, and recommended working set, explicitly scoped;
- free disk and projected free disk after every active task's declared writes, atomic temp space,
  and a 25% uncertainty reserve;
- `pmset` thermal/performance status and AC/battery state;
- the frontmost application and the presence of Blender, Final Cut Pro, DaVinci Resolve, Premiere,
  or HandBrake;
- known MOP heavy processes that were not launched through the scheduler.

Missing required telemetry fails closed. Admission requires three consecutive healthy samples. Two
consecutive unsafe runtime samples pause a scheduler-owned task; a critical disk or memory decision
pauses immediately. Resume requires three healthy samples and a 60-second cooldown. If Blender or
another declared foreground workload is present, one experiment lane may continue, but a second lane
is denied.

CPU load and utilization ceilings are admission-only. Once an owned CPU task is running, its intended
core saturation is not treated as foreign pressure and cannot make it pause itself. Runtime monitoring
continues to enforce memory, swap, thermal, power, disk, lane, foreground, and unmanaged-process gates.

The throttle never signals a discovered user process. It creates a new process group only for the
command it launches, records that PID, and confines `SIGSTOP`, `SIGCONT`, `SIGINT`, and `SIGTERM` to
that owned group. It never uses `SIGKILL`. At a pause or five-hour boundary it first stops that group,
hashes only atomically published checkpoint files (never `.tmp` files), records the latest durable
resume authority, and then resumes or gracefully interrupts the owned command. Work after the most
recent checkpoint may replay; a published checkpoint is never inferred from a partial write.

The active-lane registry is atomically locked at launch, so two simultaneous schedulers cannot both
claim the same slot. A corrupt registry is a refusal, not an empty-machine assumption. Receipts and
registry updates use write-then-rename publication.

## Dry-run and current host receipt

The following performs three real host samples and launches nothing:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py decide \
  --task p4_resume_cpu \
  --samples 3 \
  --interval-seconds 0.1 \
  --out proof/LOCAL_EXECUTION_THROTTLE_P4_CPU_DRY_RUN.json
```

The receipt is refreshed at each execution boundary and launches no command. It binds the exact
policy and scheduler implementation hashes, records all three host samples, and preserves every
failed gate rather than collapsing the result to one boolean. Historical dry receipts are evidence
only for the policy and host state they bind.

## P4 historical closure

P4 is complete. Its CPU campaign closed all twelve registered cells across five seeds, and the
governor ended with no active lane. The execution-time governor receipt remains immutable and keeps
the policy hash that actually supervised that run. Current policy changes do not rewrite it. The P4
task declarations remain available only as exact historical resume authorities; no partial P4
trajectory is active and no P4 command is next.

## Canonical P5 CPU branches

P5 is pinned to CPU. Every heavy step runs through the governor. The common entrypoint is:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5smoke_cpu \
  --run-id p5smoke_20260711_leg4 \
  --execute \
  --out proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json
```

The smoke result either stops at a verified terminal null or admits the grid and pilot. The pilot
then selects a null verifier or a separately governed fresh challenge.

If the f64 same-initialization trainability gate is a scientific null, the smoke receipt is complete
and all_ok, but explicitly terminal. No training grid or pilot is meaningful. The independent null
verifier runs as a light task:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5verify_smoke_null_cpu --run-id p5-smoke-null-verify --execute
```

If smoke clears trainability, the full branch is:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5_traingrid_memory_probe_cpu --run-id p5-traingrid-cpu --execute

PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5pilot_cpu --run-id p5-pilot-cpu --execute

# Pilot null only:
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5verify_pilot_null_cpu --run-id p5-pilot-null-verify --execute

# Favorable pilot only:
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5fresh_challenge_cpu --run-id p5-fresh-challenge-cpu --execute

PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5verify_cpu --run-id p5-verify-cpu --execute
```

The fresh challenge is conditional. The pilot emits `fresh_challenge_required` only as a
non-evidentiary authorization hint when a strict f64 or f32 confidence interval lies wholly beyond
one SESOI boundary. Equality is a null. A pilot null skips the heavy challenge and proceeds only to
`p5verify_pilot_null_cpu`. A favorable pilot cannot enter `p5verify_cpu` until
`p5fresh_challenge_cpu` has a current sealed governor receipt. Three complete full-surface training
runs at disjoint seeds 5101, 5102, and 5103 must reproduce the direction. A ceilinged or one-unit
mechanics result is not P6-ready.

Smoke and pilot retain their hashed 10,800-second internal configs and exact CPU checkpoint
directories. Every checkpoint snapshot includes `checkpoint.pt`, `arm_receipt.json`, the seed
receipt, the frame receipt, the root run receipt, the resolved config, and the published proof. Each
seed and checkpoint binds the current config, cell registry, P5 source aggregate, and derived
checkpoint requirements digest. The verifier independently reconstructs the exact seed set licensed
by trainability, off-ceiling staging, and the three-seed futility rule. A valid terminal scientific
null exits successfully and is not resumable. Wall or disk stops remain incomplete, publish no
proof, return exit code 2, and resume only under the same identity. Other incomplete states return
exit code 1 and publish no proof.

The training-grid probe writes
`proof/P5_TRAINGRID_MEMORY_TRACE.json.progress.json` atomically after every successful cold child.
That progress identity binds the script, boundary trace, full cell grid, batch rows, repeats, seed,
memory guard, device, live P5 config, P5 CLI, and P5 runner. An exact rerun skips only verified
finite rows and refuses identity or source drift. The final mechanics receipt remains
non-promotable by construction.

The final P5 verifier reopens the published proof, raw run receipt, resolved config, frame receipts,
seed results, source bindings, checkpoint identities, controls, and mutation suite. It rejects 18
base mutations and 23 mutations when a fresh challenge is required. Checkpoints are opened with
`weights_only`, and model plus target state hashes are independently recomputed. Held-out scores are
recomputed from durable per-seed receipts but are not re-evaluated from model checkpoints. The
governor rebuilds every fresh-pattern decision from the bound raw challenge cells. It emits a P6
prerequisite only for a verified null or `favorable-programmatic-only` result. Scientific and
confirmatory promotion remain false on every branch.

## P6 progressive continual-event order

P6 is a CPU-and-disk mechanics lane, not a heavy-model lane. Every progressive runner task is
exclusive and may not overlap P4, P5, Blender, another scheduler lane, or an unmanaged known
process. The canonical order is:

1. `p6_10k_resource_probe_cpu`, one 10k abrupt/replay cell on the first seed;
2. `p6_10k_replication_cpu`, two schedules by three arms by five seeds;
3. `p6_10k_verify_cpu`, independent checkpoint-backed verification;
4. `p6_100k_replication_cpu`, the same 30-cell matrix, only if the 10k verifier authorizes it;
5. `p6_100k_verify_cpu`, independent checkpoint-backed verification;
6. `p6_1m_replication_cpu`, the same 30-cell matrix, only if the 100k verifier authorizes it;
7. `p6_1m_verify_cpu`, final independent checkpoint-backed verification.

The 384-event receipt measured 19,584 stream bytes and at most 22,612 serialized checkpoint-state
bytes, but it did not record process RSS. The policy consequently contains no invented P6 memory
number. The first 10k resource probe is an explicitly unmeasured, exclusive task protected by live
memory-pressure, swap, thermal, AC-power, and disk gates. Full 10k admission requires that probe's
complete live-bound max-RSS receipt with at least 16 MiB measured RSS. The 100k task requires the
completed and independently favorable 10k receipt; the 1m task requires the completed and
independently favorable 100k receipt. Each prior measured max RSS receives the global 25%
uncertainty margin before admission.

Disk declarations are derived mechanically from the 384-event receipt. Stream bytes scale by rung,
shared once per seed and schedule; serialized checkpoint state is counted for every arm. A 2x
P6-specific container/serialization margin is applied first. The scheduler then adds its separate
25% uncertainty and minimum 0.5 GB reserve. The declared forecasts are approximately 0.0011 GB for
the one-cell 10k probe, 0.0116 GB for full 10k, 0.1034 GB for full 100k, and 1.0214 GB for full 1m.
The policy loader recomputes these values from the pinned receipt and refuses drift.

The progressive runner is `scripts/continual_million_event_rung.py`, configured by
`configs/experiment/continual_million_event_rungs.yaml`. Stream chunks and manifests, every
schedule/arm/seed checkpoint, the rung progress file, and final proof are exact atomic resume
authorities. Temporary files are ignored. The runner supports only 10k, 100k, and 1m; a full rung
cannot reduce the two schedules, three arms, or five seeds. Every start, cell transition, resume,
and finalization revalidates the live rung config, runner, source preflight payload, embedded
implementation bindings, and Wave E0 authority.

Before replay, the independent verifier reconstructs the complete expected plan from the live
config, including stream geometry, checkpoint interval, replay profile, cells, mode, and rung. Any
plan drift stops verification before raw replay begins. If any non-pause-safe owned child encounters
a critical runtime gate, the governor terminates it and withholds completion authority.

The independent verifier joins all 30 result rows to the live progress file and checkpoint state,
recomputes all registered metric families and controls, and rejects twelve rehashed mutations. An
exact tie is a null. Scale-up requires strict positive replay gains on both primary endpoints against
both controls in both schedules, with every paired seed delta positive and no tie. Any null stops the
ladder. Every result remains programmatic mechanics evidence and scientific promotion stays false.

Before the final P5 verifier exists, the only permitted P6 action is a dry decision:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py decide \
  --task p6_10k_resource_probe_cpu \
  --samples 3 \
  --interval-seconds 0.1 \
  --out proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json
```

The expected decision is denial with `command_executed=false` and a failed P5 receipt-prerequisite
gate. No P6 rung may start until the P5 verifier, exclusive-lane gate, and live telemetry gates are
simultaneously green.

Omitting `--execute` is always a dry-run. It writes no experiment data and starts no command.
