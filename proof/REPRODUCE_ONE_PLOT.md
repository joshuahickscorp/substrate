# Reproduce one plot (the five-minute path)

The clone-to-reproduced-plot path. The goal is that an outside reader, on their own
Apple-Silicon Mac, regenerates exactly ONE reference plot from a card and compares it to
the card's numbers within a stated tolerance. One plot, one command, a few MB of latents,
not the whole corpus. Form per BLACKHOLE.md: no em dashes; engineering vocabulary only.

This is a STUB. The concrete shard name and `repro_cmd` are filled per card once the
Studio produces the first real atlas rows and null cards (the card carries its own
`raw_run_id` and the shard it needs).

## The steps

1. Clone and install:
   - `git clone <repo>`
   - `uv pip install -e ".[dev,encoder,apple]"`
2. Fetch the tiny named latent shard referenced in the target card (a few MB, not the
   full corpus). The shard name is carried in the card's `raw_run_id` config.
3. Run the single `repro_cmd` from the card, for example:
   - `python scripts/studio_pipeline.py run --exp EX12 --shard <shard> --seeds 5`
4. Compare the produced plot to the card's reference numbers within the stated tolerance.
5. Optionally file a third-party card (`NULL_CARDS/third_party/<exp_id>__<who>.md`) with
   your hardware string and the observed delta.

## The Metal-determinism tolerance (state it wherever a repro level is claimed)

Cross-backend bitwise reproducibility is not achievable (floating-point
non-associativity). Metal is roughly 50 percent byte-identical at temperature 0; CPU is
bit-identical and is the tolerance baseline. So "reproducible on a Mac" means
SAME-MACHINE-CLASS reproducibility within a stated tolerance, never bitwise-matches-CUDA.
Every R3+ claim publishes the tolerance it was checked against.

## Repro levels (Section 10.6)

| Level | Meaning |
|---|---|
| R0 | Private run (numbers exist in `runs/` only; not citable) |
| R1 | Command + config published |
| R2 | Artifact hash + metrics pinned |
| R3 | One-command local repro, same machine class (the target before launch) |
| R4 | Third-party Mac repro filed |
| R5 | Atlas row or null card cited externally |

## First reproduce-one-plot target (MVP)

The minimum-viable reproduce-one-plot is the ViT-L identity probe row of the atlas on the
EPIC 5k licensed shard, paired with the EX12 atlas null card. That single plot (probe
accuracy vs chance floor with CI) is the cheapest, most decisive thing an outsider can
regenerate to test the instrument.
