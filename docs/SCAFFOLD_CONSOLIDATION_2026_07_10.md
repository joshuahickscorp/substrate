# Scaffold consolidation wave (2026-07-10)

Session-authored record of the combined deep-research, deep-audit, and deep-scaffolding effort that
raises the SCAFFOLDING (S) axis of the potential atlas toward 10 for every S<=7 facet, and answers
the breadth question (does the 37-facet taxonomy miss anything). Companion to
docs/INDEPENDENT_REVERIFICATION_2026_07_10.md (same session, prior wave). No generated proof was
hand-edited; the atlas rescoring itself happens only through its driver at the next regeneration.

## 1. What was built (ten new files, all gates green)

Five scaffold spines, each contracts plus deterministic fixtures plus fail-closed refusal rules plus
unit tests (208 tests total, ruff and mypy clean; no existing file was modified except the four
post-review fixes inside these same new modules):

| Cluster | Module + test | Facets served (S before) | Assigned proposed rows |
|---|---|---|---|
| Sensing and evidence | src/mop/substrate/sensing_scaffold.py | SR3 (5), SR5 (7), SR6 (7), EV4 (7) | f21, f26, f27 |
| Interactive ecology | src/mop/environments/ecology_scaffold.py | RA5 (7), RA6 (6), PA7 (7), PA8 (6), PA9 (7) | f22, f28, f50-f58 |
| Workspace and integration theory | src/mop/studies/integration_battery_scaffold.py | PA6 (6), plus PA4/PA5 support | f31-f38 |
| Security, integrity, welfare | src/mop/falsification/integrity_scaffold.py | SG2 (7), SG3 (7) | f59, f60 |
| Bio-morphogenic-material | src/mop/studies/material_twin_scaffold.py | BM1 (7), BM2 (7), BM3 (6), BM4 (6) | f61-f66 |

Everything composes the Wave E0 substrate (events, lifecycle, scenario factory, expansion harness
claim scope) rather than forking it. Each module's facet card coverage, and the bullets that remain
DATA or ENVIRONMENT gated rather than scaffold gated, are recorded in the workflow receipts and
summarized per facet in the module docstrings. Adversarial doctrine review found four minor issues
(claim-scope forks, one ungated free-text field, one private-name import), all fixed in this wave;
its two blocking findings were both about landing registry rows, which is why rows are staged, not
landed (section 4).

## 2. Staged registry rows

registry/staged/f21_f66_scaffold_rows.yaml holds 30 schema-shaped rows (f21, f22, f26-f28, f31-f38,
f50-f66), status registry-only, evidence R0, honest nulls and controls, sentience-rail-clean, with
the one out-of-vocabulary tier pair (f65) corrected to environment-needed / env-later. The file is
inert: no driver loads registry/staged/. The remaining proposed rows (f23-f25, f29, f30, f39-f49)
belong to facets already at S>=8 whose scaffolds exist (Wave E0 covers f23/f29/f39 as sentinels;
the memory ladder rows ride the P6 lane).

## 3. Breadth verdict: four missing facets, two conflations

External-framework research (capability batteries, developmental robotics, AI safety and evaluation
frameworks, ALife open-endedness, neuromorphic assessment, welfare governance) judged the 37-facet
taxonomy broad and largely complete, with four genuinely unsubsumed candidates:

1. Interpretability and internal-representation legibility of the owned substrate (an external
   analyst's ability to read learned internals; PA5 is the system's own self-report, SG2 is tamper
   resistance; neither covers analyst-side legibility).
2. Alignment, honesty, and anti-deception under incentive (strategic misreporting, sandbagging,
   specification-gaming resistance; PA5 tests causal grounding of reports, not incentive-conditional
   honesty).
3. Dangerous-capability red-lines and elicitation thresholds (evaluation-driven capability ceilings
   with preregistered thresholds; SG1 covers corrigibility, OP4 covers harness autonomy, neither
   sets capability red-lines).
4. Scale-extrapolation and capability-forecasting validity (does small-scale evidence predict
   larger-scale behavior; EV5 governs power at fixed scale, OP2 measures one operating point).

Conflation fixes for the next atlas regeneration: the PA6 card spans several distinct integration
constructs under a non-standard label and should name them; SG2 bundles evaluator integrity with
classical security and should split the two threat surfaces.

These become facets only through the atlas driver, scored honestly (all four would start with low S).

## 4. Landing sequence (the next coordinated wave; do not land piecemeal)

The doctrine reviewer verified both blockers that make row-landing a sequenced operation:
validate_experiments now runs the exact contract audit over all F rows, so registry-only rows must
either be exempted from config/class expectations by the audit's implemented_only semantics or land
with config stubs; and any registry change stales the exhaustion, requirements, claim-audit, and
atlas chain. The wave, in the execution plan section 18 order:

1. Confirm build_contract_audit treats status registry-only rows as preregistration-only (no config
   or class required); adjust its expectations if the current code demands more.
2. Land registry/staged/f21_f66_scaffold_rows.yaml into registry/experiments.yaml; re-render
   EXPERIMENTS.md; run validate_experiments.
3. Regenerate project exhaustion, then frontier localization, then rebuild and check the
   extended-compute requirements matrix (the category-2 registry row count will grow; that is the
   point), then refresh the completion-claim audit bound inputs, then regenerate the potential
   atlas so the S-axis rescoring of the sixteen served facets happens through the driver.
4. In the same wave, close reverification defects D2 (finalize all_ok hardcode), D3 (wave0 verifier
   mutation-skip), D10 (closure script --out flags), and promote verdict_gate's private predicate to
   a public name (integrity_scaffold carries a documented temporary re-declaration), regenerating
   the receipts that bind those scripts' hashes.
5. Commit the encoder retirement and the claim-audit artifact so neither remains unversioned (D1,
   D8), and either driver-regenerate or annotate the post-hoc-edited scale-atlas receipt (D7).

## 5. What scaffolding cannot buy

S is the only axis this wave moves. The overall facet scores stay capped by confirmation until real
experiments run on appropriate independent units: the four missing-facet candidates start at low S,
SR3 still needs a rights-clean native audiovisual cohort, the ecology contracts need executed
worlds, and the material twins need executed damage/repair batteries. That execution is the goal
prompt's job, behind P4/P5 on the governed heavy lane.
