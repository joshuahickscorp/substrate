# MOP Generation 2 Reproduction Guide

How an outside reviewer reproduces the terminal Generation 2 results without traversing repository history.
Environment: Python 3.12 venv at `/Users/scammermike/Downloads/mop/.venv`; torch 2.13, torchvision 0.28,
numpy 2.5, gymnasium 1.3.0. Absent: torchaudio, scipy, sklearn, librosa. House rules: no dashes.

## Layout (branch `agent/mop-scientific-frontier`, worktree `mop-scientific-frontier`)
- Calibrated admission battery: `salvage/lanes2/repaired_battery.py`, `estimators.py`, `calibration.py`.
- Frontier lane beds: `frontier/lanes/{selection_beds,temporal_beds,control_beds,lane_p_emnist}.py`.
- Runners: `frontier/{run_admission,run_temporal,run_control,run_lane_p}.py`.
- DAG scheduler + benchmark: `frontier/{scheduler,bench_concurrency,finalize_frontier}.py`.
- Sealed results: `frontier/reports/MOP_FRONTIER_*_ADMISSION_RESULT.json`, `MOP_FRONTIER_P_RESULT.json`.
- Closure + Gen3: `frontier/MOP_GENERATION2_*`, `frontier/closure/`, `frontier/generation3/`.

## Reproduce the seven admissions
```
cd frontier
PY=/Users/scammermike/Downloads/mop/.venv/bin/python
$PY run_admission.py            # via run_selection driver: V, K, M (battery lanes)
$PY run_temporal.py             # E, C
$PY run_control.py              # A, S (gymnasium classic-control)
```
Each writes `frontier/reports/MOP_FRONTIER_<L>_ADMISSION_RESULT.json`. Expected classifications: V
architecture_dependent; K, M, E, C, A, S pruned_mechanism. Beds cache to npz on first run; delete the npz under
`runs/generation2/...` to rebuild from raw data.

## Reproduce Lane P (P1R external replication)
```
$PY run_lane_p.py               # EMNIST-balanced class-incremental, 5 torch-seeded streams, 6 methods
```
Expected: faithful P1R final avg approx 0.10, GDumb approx 0.59, reservoir approx 0.58, no-replay approx 0.07;
classification replication_null (harm relabeled to null: harm is defined relative to no-replay).

## Reproduce the concurrency benchmark and schedule model
```
$PY bench_concurrency.py        # measured slowdown at 1..4 concurrent torch capsules
$PY finalize_frontier.py        # observed schedule + parallel replay (decomposed Lane P)
```

## Independent verification
```
$PY verify_frontier.py          # re-derives all seven admission verdicts from sealed clauses; checks receipt hashes
```

## Where the strongest scientific uncertainty remains
- Whether P1R value used as a soft sampling PRIORITY over a representative buffer beats GDumb (Gen3 candidate C1;
  precompute in `frontier/generation3/`).
- Whether V1 verification value is decodable by a richer capable family (Gen3 candidate C2; precompute showed no
  headroom under a strict single-feature control).
- External INDEPENDENT replication of P1R (second team or external code) has never been attempted.
