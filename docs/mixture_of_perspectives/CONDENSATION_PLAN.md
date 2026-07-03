# CONDENSATION PLAN: aggressive batching for an easier-to-scaffold tree

Audit date 2026-07-03 (five parallel area auditors + a synthesizer, grounded in actual file counts). The
mandate: shrink the tree so future scaffolding and review are easier, accepting modest immediate runtime
and ergonomic sacrifice. Guardrails: the gates (ruff, mypy on src/mop, pytest, check_docs, acceptance) stay
green after every phase; the Studio spine and facet-12 scripts keep working; the falsification, registry,
and provenance systems stay intact; the frozen-encoder doctrine is untouched. House style: no em or en dashes.

## Headline

Baseline: 608 tracked files, ~65k Python LOC (src/mop 139 files / 28k, scripts 105 / 28k, tests 82 / 8.5k),
153 config YAMLs, 66 markdown. Target after all phases: about 460 files (~24 percent fewer) and 5,000 to
5,700 fewer LOC (~8 to 9 percent). The file-count win is concentrated in the low-risk P0 and P1 tier
(configs, preregistration mirrors, boilerplate hoists); the P2 tier is a smaller LOC win at higher risk.

## Phases (each ends gates-green plus one commit; the verifying gate is named)

PHASE 1 (P0, config-first, the biggest low-risk file-count drop):
- Collapse the 34 `configs/experiment/mop_*.yaml` preregistration-MIRROR configs (NOT compose-loaded) into
  one `configs/experiment/mot_mirrors.yaml` keyed by id; relax the `src/mop/harness/validate.py`
  null_hypothesis check to read the map. 34 -> 1.
- Flatten `campaign/legs/trackNN/` subfolders into a flat `campaign/legs/` and rewrite the ~35 explicit
  `sweep:` path strings in `campaign/run_queue.yaml` (pure move + manifest edit; leg paths are explicit
  strings, not globs). 13 folders removed.
- GATE: pytest (validate.py null-hypothesis + queue load_leg) + acceptance.

PHASE 2 (P0, shared-primitive hoists, MUST precede the scripts that use them):
- `src/mop/experiments/script_lib.py` = canonical parse_seeds / resolve_out / make_graded_split / pearson /
  verdict / nmse (currently duplicated in 17 scripts and re-imported cross-script from mop_mt5).
- `src/mop/experiments/_plotting.py` = frontier_scatter + a use-Agg shim (replaces 11 inline plot methods).
- `tests/conftest.py` = a cpu fixture, a compose helper, a check_docs fixture (hoist the per-file
  sys.path/device boilerplate).
- Fold the 3 orphan experiment singletons into their letter-series file; collapse `learning/alternatives/`
  into a flat `learning/`.
- GATE: ruff + mypy on src/mop + pytest + check_docs. (Entrypoint filenames unchanged, zero doc churn.)

PHASE 3 (P1, low-risk dedup on top of the Phase-2 primitives):
- Dedupe the 6 synthetic clip generators into `src/mop/substrate/synthetic_clips.py` and re-point callers
  (this centralizes the synthetic-clip contract the facet-12 gate depends on).
- Extract the shared frozen-random + rank-reduced control loop from the 4 `close_*.py` into
  `mop.diagnostics.substrate_ablation` (files stay, LOC shrinks).
- Move the orphan `studio_*.py` into the `studio/` package; merge the `metrics/` 2-file package; collapse
  the 3 single-consumer `devel/` tools.
- GATE: ruff + mypy + pytest.

PHASE 4 (P1, the batching merges with import and doc updates):
- `run_*.py` (8) -> `scripts/run.py` subcommands; e2-e10 (9) -> `e_series.py`; ex1-ex18 (18) -> 3 theme
  modules; the 24 diagnostics -> ~8 family modules (ship 1-line shims for the 135 by-path importers, or
  sed-rewrite); track13 (16) -> one grouped sweep set; the E-series gate tests (11) -> one parametrized
  file; consolidate the numbered MoP section docs + RESULT docs PAIRED with the `check_docs.py` CANONICAL_MD
  edit in the same commit.
- GATE: full suite (ruff + mypy + pytest + check_docs ledger==disk + acceptance).

PHASE 5 (P2, high-risk file collapses, one family per commit, each with a green acceptance run):
- mt/dr reasoning-halting cluster (7) -> `scripts/mop_reasoning_lane.py` subcommands.
- cache_*.py encoder cachers (8-10) -> `scripts/cache.py` subcommands, PRESERVING
  `scripts.cache_randominit_vitl_features.assert_encoder_lane_free` (a Studio import) as its own file or a
  re-export.
- the 4 dr13 facet-12 scripts -> `scripts/mop_dr13_predictor.py` subcommands (this is the facet-12
  acceptance gate; keep `verify` a distinct adversarial subcommand). Update the 4 doc references + the
  `test_mot_rollout` import.
- merge the substrate cache pair; fold the 7 studio unit tests into 2 modules.
- GATE per commit: check_docs (the `_SCRIPT_REF` regex fails if any `scripts/foo.py` named in a doc is
  missing) + acceptance (facet-12) + pytest.

## Must NOT be condensed

1. The Studio spine and its load-bearing cross-module imports: `scripts/studio/{dr1_curate_bound_video,
   pr9_continual_backprop, atlas_multi_encoder_grid, dr1_smoke}.py`, and the symbols other scripts import
   from them (`cache_randominit_vitl_features.assert_encoder_lane_free`, `mop_at1_grid_pilot.{COLUMNS,
   evaluate_grid,parse_seeds}`, `mop_al2_alignment_pilot.evaluate_pairs`, `featurize_programmatic.CACHE_ROOT`).
   `shell/` (12 modules, 1:1 with ARCHITECTURE.md levers) and `studies/` (dynamic `__import__` on exact
   filenames from `cpu_campaign.py`) are NOT batched.
2. The registry id->class mapping and the falsification/provenance system: the registry YAMLs, the
   null_hypothesis contract in `base.py.__init_subclass__`, NULL_CARDS provenance, and `proof/` (every entry
   ledgered in check_docs PROOF_MD) stay intact.
3. The frozen-encoder doctrine: encoder configs and the encoder-registry honesty layer stay; the deferred
   `_base.yaml` encoder hoist only moves `frozen: true` into a shared default, never weakens it.

## Honest tradeoff (what the size win costs)

- Import-time cost: batching singletons into series/family modules means importing one experiment pulls its
  whole series' torch/matplotlib deps, raising cold-import and pytest-collection time. Real but modest.
- Subcommand indirection: `python scripts/foo.py` becomes `python scripts/run.py foo ...` for the collapsed
  families; muscle memory and any hardcoded wrapper path breaks, and a crash trace points at a shared module
  rather than a purpose-named file.
- Shim tax: the 135 by-path diagnostics importers are kept working via 1-line forwarders, so the import
  graph gains a hop (net file win preserved).
- Merge diffs are large and git-blame history is disrupted for the merged files.
None of these touch correctness or the gates; they trade per-invocation ergonomics and a little import
latency for a materially smaller, easier-to-scaffold tree.

## Recommended execution posture

Phases 1 to 3 (P0 plus low-risk P1) capture most of the file-count win with near-zero Studio-facing churn and
no subcommand indirection on the scripts the Studio runs. Phase 4 is the batching bulk. Phase 5 (the
subcommand collapses of the facet-12 / cache / reasoning scripts) touches the exact scripts the Studio uses
imminently, so it is best deferred until AFTER the Studio run unless the churn is explicitly wanted now.
