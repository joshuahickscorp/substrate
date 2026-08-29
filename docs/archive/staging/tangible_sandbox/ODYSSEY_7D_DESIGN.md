# Substrate Odyssey — seven-day design

**Status:** staged, inactive post-R2 design. It is deliberately more than a
compressed 24-hour measurement lattice. It may not start until every launch
gate in `ODYSSEY_7D.draft.json` is sealed and admitted.

## What makes it worth seven days

The Odyssey keeps eight separately custodian-seeded frontier histories alive
for 168 hours: integrated continuity, math, logic, code, philosophy/self-model,
sound, vision/spatial/embodied action, and science/multimodal inference. The
same histories return across a precommitted curriculum, competing work,
interruption, recovery, cross-modal transfer, and closed-book assessment.

Parallelism buys frontier breadth; the week buys temporal depth. It does not
turn repeated observations into independent samples. The initial design has
eight independent histories, 672 two-hour microcycles, 2,688 scored events,
10,752 scored dimension observations, and 2,696 half-hour resource samples.

## Arc

| Day | Arc | Hard question |
|---|---|---|
| 1 | Map and baseline | What can each system do before the intervention? |
| 2 | Foundation and guided practice | Can it use the specified curriculum under matched controls? |
| 3 | Near transfer and revision | Can it apply, revise, and retain rather than replay? |
| 4 | Interference and recovery | Can it recover durable state after a precommitted disruption? |
| 5 | Far / cross-modal transfer | Does learning survive a new representation or tool body? |
| 6 | Long-horizon project synthesis | Can it maintain commitments through substantial integrated work? |
| 7 | Cold retention and blind exam | What survives after delayed, instruction-free evaluation? |

Each two-hour block is baseline/retrieval → sealed guided or matched control
exposure → novel transfer/project work → delayed recall, repair, or checkpoint.
Day 7 gives no new instruction. Daily results cannot alter the curriculum;
only a resource or safety breach may safely pause the campaign.

## Density without invalidity

The design is dense because it reuses time intelligently, not because it
allows result-dependent task selection:

- all task identities, instructional materials, and answer mappings are
  committed before the candidate runs;
- the matched control receives equal wall-clock, token/instruction budget,
  compute, tools, stimulus family, and rubric; only the named pedagogical
  intervention differs;
- every transfer task has evaluator-only scoring material and follows a locked
  trace;
- every cell has distinct writable roots, ledgers, checkpoints, and seed;
- daily checkpoints create recoverability, not permission to revise protocol.

## Hard operating limits

The 85 GiB resident cap is unchanged. Normal admission is capped at 75 GiB;
P2 checkpoints at 80 GiB, P1 pauses at 82 GiB, and all non-P0 work holds at 85
GiB. The measured 1/2/4/6/8-cell calibration decides whether eight cells can
run; a failed width means the Odyssey launches only at the lower admitted width
or does not launch. No remote job is part of the authoritative run: the
available RunPod credential currently fails its non-billable API probe.

## Launch sequence

1. R2 reaches terminal state and independent verification.
2. Select and hash-pin exactly one held-out history per frontier; accept terms,
   parity, and evaluator-only splits.
3. Run the resource calibration and fresh dynamic disk/memory preflight.
4. Bind candidate, control, evaluator, custody, seed, and notification
   versions; seal the complete authority set.
5. Probe 30-minute Telegram reporting, then detach one supervisor from the
   frozen 168-hour manifest.

The supervisor reports phase, day, microcycle, completion percentage, active
cores, memory pool use, free storage, guard state, and any broker decision
every 30 minutes. A safety hold preserves receipts and does not choose a new
scientific path.
