# APPLE SILICON

MOP is Apple-Silicon-native on the current M3 Pro. CPU and MPS are workload choices selected by
measurement. A larger Apple Silicon host is not an assumed destination.

## Current host

- chip: Apple M3 Pro;
- logical CPUs: 12;
- unified memory: about 19.3 GB;
- MPS: available;
- active heavy lanes: one;
- operational maximum: 300 minutes;
- free-disk floor: 40 GB after forecast and atomic-write margin;
- heavy power source: AC;
- runtime governor: `scripts/local_execution_throttle.py`.

Unified memory is shared by the OS, applications, CPU tensors, MPS allocations, file cache, and
other processes. Installed memory is not usable accelerator memory. The governor separately observes
available memory, memory pressure, swap, MPS allocation where visible, and the recommended MPS
working set.

## Device boundary

Device selection goes through `devices.resolve(cfg.device.kind)`. `auto` may prefer MPS, but a
scientific command pins the measured path explicitly. CPU remains the deterministic fallback and is
the active P4 path. Unsupported Metal operations may fall back to CPU only when the experiment's
device and numerical contract permits it.

Every device comparison records:

- exact source and configuration;
- tensor geometry and precision;
- warmup and synchronization policy;
- output finiteness and numerical parity;
- process-tree RSS;
- MPS current and driver allocation where available;
- wall-time distribution;
- failures, fallbacks, and retries;
- thermal and power state.

A faster device path is rejected if it changes the scientific decision beyond the preregistered
parity tolerance.

## Current dense-instrument result

The official dense ViT-B checkpoint is retained locally and strictly loads into 86.8M parameters.
Finite CPU forwards pass at 8 and 64 frames. The native 64-frame forward produces shape
`[1, 18432, 768]` in about 25.2 seconds with about 1.33 GB maximum observed process-tree RSS.

This establishes local runtime availability. It does not establish learned-code value, natural-data
performance, task integration, or a reason to scale the inherited instrument.

## CPU execution

CPU is appropriate when:

- deterministic replay is the priority;
- the MPS graph or allocator path is less stable;
- a workload is small or memory-bound;
- the model executes serially within the scientific deadline;
- CPU leaves MPS available for user applications.

Thread and process counts must be declared. Do not infer useful parallelism from logical CPU count.
Measure oversubscription, BLAS threads, memory bandwidth, and thermal behavior. The adaptive governor
allows declared task CPU saturation after admission while retaining memory, swap, thermal, power,
disk, and foreground gates.

## MPS execution

MPS is appropriate only after a workload-specific parity and stability pilot. Use explicit
synchronization around measured work. Record fallback behavior and refuse silent device changes.

For training:

- checkpoint at exact arm and update boundaries;
- synchronize before publishing progress;
- write crash and refusal receipts atomically;
- use a separate run directory when CPU and MPS trajectories cannot share exact state;
- do not recover a transient failure into a citable trajectory unless the registered policy and
  verifier allow it.

The governor sees only its own process MPS counters plus declared child peak. macOS does not supply a
single project-wide per-process Metal accounting API through PyTorch, so headroom remains
conservative.

## Precision

Lower precision is a scientific intervention, not a free optimization. BF16, FP16, INT8, or lower
precision requires:

- finite outputs and gradients;
- no overflow or underflow beyond the declared rule;
- representation geometry parity on held-out referents;
- endpoint and decision parity;
- objective-by-precision interaction check where training is involved.

If parity fails, use the exact precision or treat the precision change as a separate experiment.

## Caching

Frozen instruments should be encoded once only when caching preserves the intended stochastic and
view contract. A citable cache binds:

- checkpoint and revision;
- source object bytes;
- decode and preprocessing code;
- frame indices, crop, view, resolution, and normalization;
- RNG and precision;
- layer, pooling, token shape, and time axis;
- referent, event, split, and independent-unit identity;
- output hashes and manifest schema.

Frozen weights do not make a cache timeless. Any identity change requires a new cache or an explicit
parity receipt.

## Adaptive governor

Heavy tasks use a declared entry in `configs/local_execution_throttle.yaml`.

```bash
.venv/bin/python scripts/local_execution_throttle.py decide \
  --task <task-id> --samples 3 --interval-seconds 2 --out <decision.json>

PYTHONPATH=src .venv/bin/python scripts/local_execution_throttle.py run \
  --task <task-id> --run-id <unique-id> --execute --out <run.json>
```

Admission requires three good samples. Runtime monitoring may pause only the scheduler-owned process
group. It never signals user processes. Known foreground work may coexist with at most one experiment
lane, and resource pressure can pause that lane until hysteresis clears.

## Larger-host gate

A larger Apple Silicon host is considered only after a survivor supplies:

1. a named scientific requirement;
2. three valid measured local failures;
3. failed or scientifically invalid streaming, caching, factorization, recurrence, precision, and
   sequential alternatives;
4. proof that a reduction changes the estimand or decision;
5. the smallest enabling memory, bandwidth, or latency calculation;
6. a parity-preserving next-host pilot.

More unified memory or GPU cores can buy throughput. They establish scientific necessity only when a
non-factorizable resident state, intrinsic real-time deadline, or inseparable synchronized
interaction cannot be preserved locally.

The current requirements matrix has no such row.
