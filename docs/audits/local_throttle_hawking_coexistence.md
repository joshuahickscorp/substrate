# Hawking coexistence cutover

Status: implemented and fail-closed on 2026-07-12.

The first substrate campaign originally treated every recognized Hawking worker as an absolute
admission veto. That was safe but prevented progress while long-lived audit ladders occupied only a
changing subset of the Mac Studio. The replacement is task-scoped: only the EDCM and X0 producer and
verifier commands may use the coexistence profile. P6 remains under its original exclusive resource
probe contract.

The cutover authority is `proof/LOCAL_THROTTLE_HAWKING_COEXISTENCE_CUTOVER.json`. It binds the terminal
old campaign, an empty MOP lane registry, the new policy/governor/router bytes, the historical overlay,
the new reproducible overlay, and the external profile. It changes no experiment seed, control,
horizon, threshold, verdict rule, implementation authority, or scientific-promotion boundary.

## External profile

Hawking is observed but never owned or signalled. Admission requires the exact current UID and root,
the reviewed Python and quantizer executables, exact argv grammars and parent/model joins, stable PID
creation identities, and CPU-only sanitized environment evidence. The profile snapshots and hashes
`studio_run.py`, `audit_ladder.py`, `doctor.py`, and `quantize-model` through regular non-symlink file
descriptors. Any path, byte, process-tree, environment, thread-count, RSS, or CPU drift restores the
veto.

The external envelope permits at most three audit roots, eight recognized processes, one active
compute child per root, four quantizer threads, 64 decimal GB aggregate RSS, and 2700% aggregate CPU.
Every report is self-sealed, observation-only, and nonpromoting.

## MOP lane

Each reviewed MOP command is one-core and CPU-only. Darwin `taskpolicy` applies background priority,
throttled disk I/O, background QoS, a kernel 4096-MiB memory limit, and kill-on-limit process control.
BLAS/OpenMP thread environments are capped at one. Only one MOP lane may exist, and the producer yields
after one complete atomic seed boundary so host admission is repeated before every next seed.

During validated coexistence, the normalized one-minute load gate remains live at the reviewed
first-lane ceiling of 0.85. Instantaneous
CPU bursts may use the host while Darwin background QoS makes the one-core MOP child yield; the profile
still reserves one logical CPU and sustained saturation raises the load gate. Both VM availability and
`memory_pressure` must retain at
least 40%, at least 40 decimal GB must remain available, `memory_pressure` must report at least 75%
free, swap must remain exactly zero, thermals must be normal, and AC power plus the 40-GB disk floor
remain mandatory. Critical pressure or profile drift pauses only the owned MOP process group.

Every exit-2 seed yield publishes a self-sealed progress authority binding the atomic checkpoint,
command, task, policy, governor, task-policy authority, return code, and measured process-tree RSS.
The campaign rejects an unsealed or spliced progress receipt and re-runs admission before resuming.

## Parallel substrate work

The shadow-only G0 construction plane lives in `src/mop/studies/escs_g0_construction.py`. It implements
all six declared construction-mutation families in new, non-authoritative files. It cannot activate a
slot, mutate the live chassis, claim factual effects, or grant promotion. Existing mechanics and
substrate-preflight hashes remain unchanged.
