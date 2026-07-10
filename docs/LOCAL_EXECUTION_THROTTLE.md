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

An initial pass correctly waited while the normalized one-minute load was 0.8668, just above the
0.85 first-lane ceiling after an earlier P4 leg ended. The latest receipt, collected after cooldown,
admitted all three samples at 0.6744 normalized load. No required telemetry was missing; memory,
swap, thermal, power, disk, MPS reporting, foreground, and unmanaged-heavy gates passed; the write
forecast left about 87.9 GB free against the 40 GB floor. No command ran. The receipt binds both the
policy and scheduler implementation hashes.

## Exact P4 resume invocation

First inspect the decision. Then, when the heavy lane is intentionally available, execute the exact
registered command through the throttle:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p4_resume_cpu \
  --run-id p4-local300-cpu \
  --execute
```

The child argv is exactly:

```text
.venv/bin/python scripts/p4_capability_density.py --profile p4screen --device cpu --run-dir runs/p4_screen/p4screen --out proof/P4_CAPABILITY_DENSITY_SCREEN.json
```

P4's internal 180-minute budget remains unchanged. If it exits 2, the supervisor snapshots the
checkpoints and invokes that same argv again while the 300-minute outer leg has time. The run receipt
is `runs/local_throttle/p4-local300-cpu/run_receipt.json`; child stdout and stderr are separated by
invocation under its `logs` directory.

The active partial P4 trajectory must remain on CPU. `p4_resume_mps` is separately pinned to
`runs/p4_screen/p4screen_mps_clean` and
`proof/P4_CAPABILITY_DENSITY_SCREEN_MPS_CLEAN.json`; it may start or resume only that clean MPS
trajectory. It never opens the CPU checkpoint directory.

## Canonical P5 CPU order

P5 is pinned to CPU in this order after P4 releases the heavy lane:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5smoke_cpu --run-id p5-smoke-cpu --execute

PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5_traingrid_memory_probe_cpu --run-id p5-traingrid-cpu --execute

PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task p5pilot_cpu --run-id p5-pilot-cpu --execute
```

The smoke and pilot retain their hashed 10,800-second internal configs and exact CPU checkpoint
directories. The training-grid probe writes
`proof/P5_TRAINGRID_MEMORY_TRACE.json.progress.json` atomically after every successful cold child.
That progress identity binds the script, boundary trace, full cell grid, batch rows, repeats, seed,
memory guard, and device; an exact rerun skips only verified finite rows and refuses identity drift.
The final mechanics receipt remains non-promotable by construction.

## P6 progressive continual-event order

P6 is a CPU-and-disk mechanics lane, not a heavy-model lane, but every P6 task is exclusive. It may
not overlap P4, P5, Blender, another scheduler lane, or an unmanaged known process. The canonical
order is:

1. `p6_10k_resource_probe_cpu`, one 10k abrupt/replay cell on the first seed;
2. `p6_10k_replication_cpu`, two schedules by three arms by five seeds;
3. `p6_100k_replication_cpu`, the same 30-cell matrix;
4. `p6_1m_replication_cpu`, the same 30-cell matrix.

The 384-event receipt measured 19,584 stream bytes and at most 22,612 serialized checkpoint-state
bytes, but it did not record process RSS. The policy consequently contains no invented P6 memory
number. The first 10k resource probe is an explicitly unmeasured, exclusive task protected by live
memory-pressure, swap, thermal, AC-power, and disk gates. Full 10k admission requires that probe's
positive measured max-RSS receipt. The 100k task requires the completed 10k replication receipt;
the 1m task requires the completed 100k receipt. Each prior measured max RSS receives the global
25% uncertainty margin before admission.

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
cannot reduce the two schedules, three arms, or five seeds. Every result remains mechanics-only and
requires an independent metric verifier before any scientific promotion.

While P4 is active, the only permitted P6 action is a dry decision:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py decide \
  --task p6_10k_resource_probe_cpu \
  --samples 3 \
  --interval-seconds 0.1 \
  --out proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json
```

The expected decision is denial with `command_executed=false`. No P6 rung may start until the
exclusive-lane and telemetry gates are simultaneously green.

Omitting `--execute` is always a dry-run. It writes no experiment data and starts no command.
