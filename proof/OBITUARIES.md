# Mechanism obituaries

One entry per mechanism that was retired, with the cleanest explanation of the dead end.
A mechanism obituary is a survival card's mirror: it states what was tried, the
pre-declared null it failed to beat, the taxonomy category it lands in, and what the
roadmap did in response. Shown prominently, never buried. Form per BLACKHOLE.md: no em
dashes; engineering vocabulary only; no agency language.

Entries are added as mechanisms are retired against their declared nulls. Each entry
links to the null card that retired it.

## Entry template

```
mechanism:        e.g. prioritized replay (E2 prioritized arm)
retired_against:  the null it failed to beat (verbatim)
taxonomy_category: 1..10 (FAILURE_TAXONOMY.md)
verdict:          DOWNGRADE-TIE | SUBSTRATE-BOUND | SEED-UNSTABLE | CAPACITY-ARTIFACT
null_card:        NULL_CARDS/<exp_id>.md
roadmap_response: what the roadmap retargeted to (capacity sweep, hybrid arm, 2.1 dense, different factor)
one_line:         the cleanest explanation of the dead end
```

## Entries

### 2026-07-10 CM7 learned objectives at the exact 1.65M programmatic regime

```
mechanism:        CM7 learned training objectives (predictive, invariance, reconstruction) at the exact 1.646M-parameter, 1,000-update, 256px, 8-frame programmatic regime
retired_against:  at matched tiny capacity, matched data, matched 256px, both custom objectives tie random-init same-arch AND tie each other; objective is not a lever at this scale
taxonomy_category: 2
verdict:          DOWNGRADE-TIE
null_card:        NULL_CARDS/mop_cm7_min_objective_probe.md
roadmap_response: platform contracts (receipt chain, resume, manifest, compute match, D3 oracle gate) carry to CM8 and the P4 capability-density response surface; no rerun of this exact regime
one_line:         five seeds and a familywise-corrected verifier show every learned arm ties or trails both the untrained initialization and the random-target control on held-out factor structure; training bought nothing here
```
