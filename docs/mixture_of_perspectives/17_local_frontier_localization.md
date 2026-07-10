# Local frontier localization

This is the active correction to the old `studio-scale`, `gpu-later`, and `moonshot` planning
labels for the non-F experiment bank. The machine-readable source is
`proof/FRONTIER_LOCALIZATION.json`; bounded execution details are in
`proof/LOCAL_FRONTIER_PREFLIGHTS.json`. Planning tags are not hardware evidence.

## Result

The current M3 Pro has **zero experiment-specific, measured hardware-blocked non-F rows**. The
strict host doctor passes, the frozen video encoder has executed locally over eight 64-frame,
256-pixel programmatic clips in 178.7 seconds (22.34 seconds per clip, 2.59 GB peak RSS), and five
formerly registry-only frontier mechanisms now execute across three seeds in about 1.2 seconds at
fixture scale. Under the user-authorized 300-minute wall, the measured encoder rate projects to 644
sequential clips after reserving 20% of the wall. That last number is a linear projection from eight
programmatic clips, not a throughput guarantee and not natural-video evidence.

The custom-substrate lane also ran locally. CM7 completed one MPS training-calibration seed in 17.31
seconds at 582,057,984 bytes peak RSS with 1,646,080 trainable parameters and compute matching within
0.008%; its programmatic-only objective-lever result is not promotable until four more seeds and an
independent verifier exist. CM8's no-training preflight executes and refuses promotion. Separately, a
canonical class smoke measured 0.99 seconds and 322,961,408 bytes peak RSS for CM8. A three-scale
real-weight atlas executed serially over eight shared programmatic referents. That retires
model availability for AL2/DR5, but not their data, sample-size, compatible-task, or matched-random
controls.

The old labels hid four different states:

1. Mechanics already run locally.
2. Mechanics now run locally, but their scientific inputs or prerequisites are absent.
3. The first blocker is rights-cleared data or a citable upstream cache/model family.
4. No full runner exists. That is an implementation gap, not a hardware wall.

No fixture result below is promoted to a scientific result. Every full launch is receipt-gated and
currently fails closed when a semantic prerequisite is absent.

All 24 historical frontier rows have now been reclassified away from those three hardware-flavored
planning tags. The registry has zero remaining non-F `studio-scale`, `gpu-later`, or `moonshot`
rows. Rights/task rows are `environment-needed/env-later`; cache/model rows are
`weights-needed/env-later`; locally executable mechanics use `cpu-now`.

## Newly localized mechanics

| Row | What now runs locally | Controls actually exercised | Remaining scientific gate |
|---|---|---|---|
| MT4 reasoning router | Learned selection over fixed-depth, adaptive-halt, beam, memory, plan, and verify fixture primitives | strongest fixed primitive, exactly matched mean FLOPs, shuffled selection, three seeds | compatible outputs on the same real referents, distinct-error certificate, independent verifier |
| AT2 mode/substrate dependence | One head/mode path over shape-matched real-like and random-like temporal views | same probe code, matched shape/resolution, disjoint train/test, three seeds | verified winning mode plus citable real and same-architecture random-init caches over identical referents |
| CM5 rejuvenation | Adapting/frozen substrate crossed with rejuvenation/no-op | equal optimizer updates, equal intervention intervals, frozen arm, no-op accounting arm, three seeds | first induce calibrated plasticity loss at a progressive local rung, then independently verify restoration |
| CM11 developmental plasticity | Early/middle/late onset, forward/reverse order, scheduled/flat plasticity | matched updates, flat schedule, reverse order, noisy-TV weighting guard, three seeds | calibrated non-ceiling curriculum and independent recomputation of window/path/U-shape signatures |
| CM12 substrate mixture capstone | Learned router over four fixture experts | best single expert, random router, exact total-FLOP padding, three seeds | cleared compatible expert pilots, common battery, declared open-model control, independent verifier |

Registry planning semantics now follow the executable path:

- MT4, CM5, and CM11 are `cpu-now` / `cpu-now` / `minutes`.
- AT2 and CM12 are `weights-needed` / `cpu-now` / `minutes`: the mechanism is local; their first
  blocker is an upstream evidence set, not compute.
- All five remain scientific `registry-only` because an executable fixture is not the registered
  claim. Their proof level is R1 and points to the local preflight receipt.

## Complete historical frontier-tag audit

| Rows | Localized state | First blocker or next action |
|---|---|---|
| MT4, AT2, CM5, CM11, CM12 | local mechanics proven | satisfy the exact receipt gates in the table above |
| DR1, CM1 | rights-data blocked | rights-cleared bound-attribute natural video; programmatic video does not substitute |
| DR3, DR4, DR7, CM3, CM9 | rights/task blocked through DR1 | dense tokens plus working-memory, counterfactual, intermediate-state, or binding annotations |
| AL3 | rights-data blocked | temporally aligned audio-video clips and citable caches |
| DR2, DR14 | upstream-cache blocked | full real-latent or dense-token cache at the registered scope |
| AL2, DR5 | local multi-encoder availability proven | three frozen real-weight encoders ran on the same eight programmatic referents; expand meaningful shared/task rows and add the exact matched-random controls |
| DR15, AT1, CM2, CM4, CM6 | upstream-model/cache blocked | matched multi-encoder/random controls, teacher/student checkpoints, or prerequisite gate output |
| CM7 | local training calibration proven | one programmatic-video seed completed with matched compute; four seeds and independent verification remain |
| CM8 | local fail-closed preflight proven | `proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json` verifies the no-training code path; CM1/CM2/CM7/DR1 and same-referent teacher rows still gate science |

This table accounts for all 24 non-F rows that carried one of the historical frontier labels when
the localization pass began. The JSON receipt preserves each row separately, including its current
registry values and first blocker.

## Fail-closed usage

Run and refresh all local mechanics and the global audit:

```bash
.venv/bin/python scripts/frontier_localization.py
```

Ask whether a full row is scientifically launchable. This command is expected to exit 2 today and
name missing receipt-backed requirements:

```bash
.venv/bin/python scripts/frontier_localization.py \
  --assert-scientific-ready mop_at2_mode_substrate_dep
```

Supplying prose or booleans cannot clear a gate. `--receipt-map` accepts only repository-local JSON
receipt paths. A file is still only a necessary integrity primitive; the independent verifier must
check its semantic claim before the result can advance beyond R1.

## What changes in the project

The project no longer treats a larger machine as the default next step for these pillars. MT4,
CM5, and CM11 can be expanded progressively on this device; AT2 and CM12 can be wired and tested
here as soon as their upstream evidence arrives. The real migration problem is now explicit:

- build compatible referent-level outputs across modes and substrates;
- acquire or generate rights-clean tasks without confusing generated fixtures for natural evidence;
- create matched random-init and multi-encoder caches;
- train and verify small adapting substrates locally before considering a wider rung;
- promote only independent, multi-seed evidence, never the preflight's deliberately constructed
  fixture advantages.

That converts “wait for hardware” into an actionable queue while preserving the original nulls and
controls.
