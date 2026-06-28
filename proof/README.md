# proof/ (the trust surface)

This is the first artifact: the Representational Atlas bundled with the Null-Card
Gallery, the most reusable and most-citable thing Brain emits, useful even to someone who
never touches the shell. The tree mirrors Section 10.9 of
`docs/STUDIO_MAXIMIZATION_2026_06_27.md` exactly. Form follows BLACKHOLE.md: no em dashes
or en dashes (commas, colons, parentheses only); no agency, intelligence, understanding,
consciousness, or sentience language; the frozen encoder is never trained; the
linear-probe gate precedes every mechanism claim.

This tree is SCAFFOLD. It carries the schemas, the frozen factor list, and the templates,
so the Studio populates real rows and cards instead of inventing structure. Empty atlas
directories and stub cards are expected at this stage.

## The tree

```
proof/
  README.md                     # this file
  ATLAS.md                      # the atlas index + how to read it (factor list + row schema)
  atlas/
    _TEMPLATE.factor.json       # the per-(encoder x factor) row template
    atlas_summary.csv           # the flat matrix header (encoder x factor x probe-type x decodability)
    vjepa2_vitl_fpc64_256/      # (empty) the Studio writes one <factor>.json per probed factor
    vjepa2_vith/                # (empty)
    vjepa2_vitg/                # (empty)
  NULL_CARDS/
    null_card.schema.json       # the machine-checkable Section 10.3 field list (probe_dependency REQUIRED)
    _TEMPLATE.md                # the human-readable card template
    third_party/                # external reruns, including REPLICATION-FAILED cards
  CORPUS_CARD.md                # the cached-latent corpus card (Section 10.10)
  REPRODUCE_ONE_PLOT.md         # the clone-to-reproduced-plot quickstart
  FAILURE_TAXONOMY.md           # the 10 categories + badge legend (Section 10.5)
  OBITUARIES.md                 # the mechanism-obituary appendix
  DO_NOT_CITE_AS_INTELLIGENCE.md  # the language-boundary note (Section 10.11)
```

## The trust surface order (what an evaluator meets first)

1. The Representational Atlas (`ATLAS.md` + `atlas/`), including the "not decodable" rows.
2. The Null-Card Gallery (`NULL_CARDS/`), wins and nulls in one identical template.
3. The Reproduce-One-Plot quickstart (`REPRODUCE_ONE_PLOT.md`).
4. The flagship bounded writeup (later).
5. The failure taxonomy (`FAILURE_TAXONOMY.md`) + obituaries (`OBITUARIES.md`).

Nothing that is not proof-shaped goes above these five.

## Minimum-viable first artifact

ViT-L only, on the EPIC 5k licensed shard, with linear-probe atlas rows for
identity/action/relation, plus null cards for E1 and EX12, plus one reproduce-one-plot
path. This scaffold is the structure that holds those.

---

## HUMAN LICENSE TASKS (start now, external latency of days)

Two manual-access datasets are blocked on a person completing terms. The planner will NOT
auto-select them until access is granted AND `--accept-license` is passed. Both approvals
can take days, so starting now is what lets the Studio pull them on arrival. No download
happens on the laptop for either; these are access grants only. Do NOT pull SSv2 or Ego4D
video on the laptop (Studio-only).

### 1. Something-Something V2 (SSv2), slug `ssv2`, status manual

The action-motion anchor and the cleanest labeled-motion source.

- [ ] Register at developer.qualcomm.com.
- [ ] Accept the Something-Something V2 terms (Qualcomm/20BN terms).
- [ ] Obtain the download token.
- [ ] Record the token somewhere the Studio session can read it.

### 2. Ego4D subset, slug `ego4d_subset`, status manual

The egocentric anchor; the curated ~200 GB subset, NEVER the full multi-TB corpus (full
Ego4D and Ego-Exo4D are `status: deferred` and are never planned by default).

- [ ] Sign the Ego4D License Agreement.
- [ ] Receive the AWS credentials.
- [ ] Install the `ego4d` CLI.
- [ ] Record the AWS creds somewhere the Studio session can read them.

When both are done, the Studio runs the planner with `--accept-license` and the planner
will select them.
