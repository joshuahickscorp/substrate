# Substrate

Substrate is a Python research program that ran repeatable experiments on
persistent cognitive material. Historical campaigns used seeded simulators.
Tangible Sandbox R2 added pinned public archives, locally generated files, and
hash-sealed receipts.

It exists to test one question: does building the entity out of many cooperating
parts with persistent memory beat a much simpler program given the same
information and budget? So far the measured answer is no.

The program is frozen. Work stopped after the Odyssey launch-gate set was sealed
and reconciled, before a seven-day worker launched.

## What the tree is

A package under `src/substrate` with campaign, sandbox, verification, and a
separate non-executing `substrate product` surface. Historical campaigns
(v1–v5, Nous Closure, Final Revision, Cognitive Material Genesis, Genesis II)
are seeded, offline, and deterministic. They feed synthetic events of eight
kinds — text, image, video, motion, audio, speech, depth/3D, and body/tool —
into small fixture modules (v5 declares thirteen) and keep identity, goals,
memory, world, and body on a hash-chained event log. A checkpoint refuses a
broken seal, non-monotonic time, or activation set true.

R2 is the live-download Core tier. It downloaded 86,036,677,480 bytes of
preregistered FSD50K and LibriSpeech archives, SHA-256 validated with zero
mismatches, and cloned eleven public git sources at pinned revisions. OSWorld-V2,
WorkArena, MLE-bench, and Common Voice stayed gated. It materialized STSC-1
`1.0.0-r2` after a generator commitment, froze a model-and-tool panel
(`qwen3:8b` as the local organ), and passed 28/28 canaries. One SWE-bench
Verified gold-patch canary (`astropy__astropy-12907`) resolved.

`substrate product` records portable specialist state and plans apprenticeships.
It does not launch a campaign, a container, or a tool.

## Where the work stopped

Cognitive Material Genesis II is the last architectural campaign. Outcome B:
`cognitive_material_genesis_ii_complete`, status
`compositional_advantage_unproven`. The preregistered simplicity rule selected
`L1_associative_monolith` rather than a field.

| Comparison | Effect | Interval / n |
|---|---:|---|
| Genesis I, selected field `K8` vs S2 | −0.247768 | 95% CI [−0.256737, −0.238393], 32 histories |
| Genesis II, selected L1 vs exact S2 | 0.393415 | 95% CI [0.372210, 0.414509], 1,792 family-history cells |
| Genesis II replication | 0.415282 | 602 cells |
| Genesis II hidden composition | 0.316777 | 602 cells |
| Genesis II best field vs best equal-plastic monolith | −0.004167 | representation/architecture factorial |
| Nous Closure, modular entity vs S2 | 0.000000 | both scored 1.0 on the publication sandbox |

The Genesis II campaign covered 4,245,640 episodes. Four of ten primary claims
passed. All 17 planted defects were detected. The selected non-continuous-time
material passed 250,000 events, 16 process interruptions, 32 scheduled
checkpoints, four migrations, and four model and body replacements. Genesis I’s
negative result is unchanged. Nous Closure had already closed as
`terminal_closed_null`.

R2’s published classification file remains Outcome C,
`terminal_tangible_sandbox_null`, from a protected-disk-floor refusal. Later
receipts do not overwrite it. After the floor was restored, Core acquisition,
freeze, canaries, and a 24-hour longitudinal lane completed (86,400.039 s, nine
checkpoints, run `r2-af2873803d99429594c557732f4140ac`). Six earlier
longitudinal attempts were invalidated. The public lane’s sealed effect is 0.0
under conservative abstention (235 tasks). A custom STSC-1 receipt scores
`L1_full` at 0.5 against S2 and `project_state_database` at 1.0 (effect −0.5,
64 histories); the published classification still records H_T12 as
`not_tested`.

Odyssey is the unlaunched successor. At HEAD, all fifteen launch-gate receipts
are `pass`, including G06-DC in place of G06. Formal G06 simultaneity (limit
1.35) still fails: on `gpt-oss:20b` MXFP4, a 1/2/4/6/8 × 3 ladder measured
width-8 max slowdown 4.392411, 3.15% of the 1,800 s phase, 31.74× deadline
headroom, zero pageouts, and peak RSS 61,843,158,492 bytes (57.59 GiB) against
an 85 GiB ceiling. G06-DC admitted width eight on deadline capacity, not
simultaneity. Preflight status is `admitted_waiting_for_authority_seal`;
`launch_allowed` is false. The two-wave alternative was refused. Native
`substrate_spatial3d` is Frontier G’s renderer: seven canaries pass; same-seed
RGB and depth are byte-identical; an occluded pyramid has 0 front pixels and
295 from the side. Blender is optional. A portability manifest records host
tools and corpora. Capability-density work (warm workers, CPU/GPU overlap,
compact receipts) is in the tree and does not change the treatment.

## What is not true yet

- No field, compositional, or low-precision architectural advantage was
  established. Given the same information and budget, the simpler program was
  not beaten.
- R2 did not publish a replacement scientific classification. H_T12 remains
  `not_tested` in that file.
- Odyssey’s seven-day experiment did not run. No scientific worker launched.
- This is not a claim about consciousness, phenomenal experience, sentience,
  feeling, suffering, desire, personhood, life, or moral status.
- It is not itself a trained model. Historical modules are deterministic
  fixtures. R2 froze a local organ. Odyssey pinned `gpt-oss:20b` and did not
  launch it as a campaign.
- It is not evidence of unrestricted real-world ability. Every finished task
  sat inside a frozen benchmark or a local sandbox.
- It is not externally reviewed. The reviewers that graded the closure package
  were internal simulations (`external_independence_claimed: false`).
- It is not switched on. Activation is `false` throughout. There is no path in
  the runtime that sets it true.
- `substrate v5 verify` does not work from a fresh clone. It needs a raw run
  tree that is not committed.

"Nous" is this project’s name for the property under test. Naming it is not
claiming it.

## Next

Nothing is scheduled. The tree is the record of a program that asked the
architectural question, measured a null, built a tangible sandbox and an Odyssey
launch-gate set, and stopped before the seven-day run.

## Evidence

- [Genesis II terminal report](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md)
  and [handoff](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_HANDOFF.md)
- [Genesis II classification](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json)
  and [limitations](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_LIMITATIONS.json)
- [Nous Closure terminal report](artifacts/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md)
  and [limitations](artifacts/substrate/nous_closure/external_review/LIMITATIONS.md)
- [R2 published classification](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json)
  and [acquisition result](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json)
- [v5 scientific status](docs/V5_SCIENTIFIC_STATUS.md)

The Odyssey receipts this README cites — the G06-DC measurement, the G06
simultaneity limitation, the preflight, and the spatial3d evidence — are not on
`main`. They sit on the unmerged `codex/odyssey-preflight-repair` branch, which
also carries a large accidental vendored toolchain and per-campaign signing keys
and so has not been published as-is.
