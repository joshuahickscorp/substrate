# The Representational Atlas (index + how to read it)

The atlas is the empirical floor under the whole program. It maps what each frozen
V-JEPA 2 encoder (ViT-L / ViT-H / ViT-g) linearly and nonlinearly affords, factor by
factor, including the factors that are NOT decodable. It is the linear-probe gate every
mechanism claim already depends on, systematized into a browsable reference. Nothing
downstream is trusted past what the atlas shows is there.

Form follows BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses
only). No agency, intelligence, understanding, consciousness, sentience, or self
language. The encoder is frozen and never trained. Vocabulary is engineering only
(novelty, uncertainty, learning progress, prediction error, decodability).

This file is the INDEX and the READING GUIDE. Publication rows live as machine-readable JSON
under `atlas/<encoder>/<factor>.json` and are summarized in `atlas/atlas_summary.csv`.
The current local scale pilot lives separately at `VJEPA_SCALE_ATLAS_LOCAL.json`: it verifies
serial execution and shared-referent mechanics for all three published V-JEPA 2 scale points,
but does not populate promotable factor rows because n=8 programmatic stimuli and the incomplete
matched-control set do not meet this file's evidence standard.

## How to read a row

Each row answers one question: is FACTOR decodable from ENCODER's frozen pooled
latents, linearly and nonlinearly, above the shuffle-label chance floor, with what
confidence and how reproducibly. A row is one of:

- decodable: linear or nonlinear accuracy clears the chance floor beyond the seed
  spread. A mechanism may build a representational claim on this factor (it must still
  cite the row as its `probe_dependency`).
- not decodable: accuracy does not clear the chance floor. Any mechanism needing this
  factor is bounded out (failure taxonomy entry 3, substrate blind spot). The "not
  decodable" rows are shown as prominently as the decodable ones; they are the
  bounded-substrate contribution the V-JEPA and continual-learning communities want.
- marginal: clears chance by less than the seed spread. Treated as not-yet-decodable
  for claim purposes until more seeds or a tighter probe resolve it.

A row is INVALID (and must not back any claim) if it is missing a chance floor (no
shuffle-label control), missing a repro level, or its `raw_run_id` does not reproduce
the reported numbers within tolerance.

## The factor list (frozen here so the Studio only populates, never invents)

These are the factors the atlas probes. The first three (identity, action, relation)
are the minimum-viable atlas (Section 10.9 MVP) and gate the first mechanism cards. The
rest extend coverage. "Not decodable" is a first-class, expected outcome for several of
these on a pooled (per-clip, non-dense) substrate.

| factor | what the probe target is | MVP | likely-bounded note |
|---|---|---|---|
| identity | which object/class is present in the clip | yes | expected decodable (linear probe acc 1.0 on record at n=96, real ViT-L) |
| action | which action/motion class the clip depicts | yes | the action-motion target; SSv2 is the clean source once unlocked |
| relation | spatial/temporal relation between two objects (above/below, swap) | yes | may be marginal on pooled latents; the relation control family probes it |
| permanence | object persists vs vanishes behind an occluder | no | candidate bound on pooled latents (no dense per-patch tokens) |
| count | number of objects/events | no | candidate not-decodable (pooled vector discards count cues) |
| motion | direction/type of motion (left/right, dolly/pan) | no | expected decodable (the moving/navigation control families) |
| temporal_order | order of sub-events within the clip | no | candidate marginal; tests whether pooling kept order |
| controllability | whether an outcome depends on an action (agent-controllable) | no | candidate not-decodable without action labels; needed by EX2/EX3 |
| intervention_effect | effect of an intervention (interventional structure) | no | expected not-decodable (taxonomy entry 10, beyond frozen-latent prediction) |
| cross_modal_correspondence | audio/text/video correspondence | no | metadata-only sources; future rung-10 work |

Rule: a mechanism card that depends on FACTOR must cite the atlas row for (FACTOR x
encoder) as its `probe_dependency`. If the row reads "not decodable," the dependent
mechanism claim is void and the result is a substrate bound, not a mechanism failure.

## The atlas row schema (each `atlas/<encoder>/<factor>.json`, fields exact)

```
encoder:        vjepa2_vitl_fpc64_256 | vjepa2_vith | vjepa2_vitg
factor:         identity | action | controllability | relation | permanence | count |
                intervention_effect | cross_modal_correspondence | motion | temporal_order
linear_acc:     probe accuracy (linear), with CI
nonlinear_acc:  probe accuracy (small MLP), with CI
chance_floor:   shuffle-label accuracy (decodability must exceed this or the row reads "not decodable")
decodable:      yes | no | marginal (relative to chance_floor + seed spread)
seeds:          n, SEM
provenance_tag: natural-video | real-encoder | structured-synthetic | provisional
repro_level:    R0..R5
raw_run_id:     run hash + config path
```

A template row lives at `atlas/_TEMPLATE.factor.json`. The per-encoder directories
(`vjepa2_vitl_fpc64_256/`, `vjepa2_vith/`, `vjepa2_vitg/`) are reserved for one promotable
JSON row per probed factor. Local availability and pilot geometry stay in their dedicated
receipts until natural content, sample size, and matched controls clear the row gate.

## The summary matrix

`atlas/atlas_summary.csv` is the flat view: one line per (encoder x factor x
probe-type), carrying decodability, chance floor, accuracy, CI, seeds, provenance, and
repro level. The header is scaffolded; the Studio appends real rows.

## Provenance and reproducibility (every row carries both)

Provenance tag ordering (richest first): natural-video > real-encoder >
structured-synthetic > provisional. A row never wears a tag richer than the cache that
produced it supports (the cache validator refuses that). Reproducibility level R0..R5
per Section 10.6; the honest Metal-determinism caveat applies (same-machine-class
reproducibility within a stated tolerance, never bitwise-matches-CUDA). Most atlas rows
target R3 (one-command local repro on an equivalent Apple-Silicon machine) before
launch, R4 (third-party Mac repro) soon after.

## Minimum-viable vs gold-standard

- MVP: ViT-L only, on the EPIC 5k licensed shard, linear-probe rows for
  identity/action/relation, plus the E1 and EX12 null cards, plus one
  reproduce-one-plot path.
- Gold: all three encoders, all factors including the "not decodable" rows, linear and
  nonlinear probes, the full E and EX null-card gallery with third-party reruns, a
  citable corpus tag, and the obituary appendix.
