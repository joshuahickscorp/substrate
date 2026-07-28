# Historical Evidence Map

Reader map of Substrate campaign evidence as present in this repository.

Every tag in this repository is an **annotated** tag, so `git rev-parse <tag>` returns the tag object, not the commit. Every commit hash below is therefore `git rev-parse <tag>^{}` — the peeled commit. Classification strings are taken from the cited final classification documents, not from README prose.

## Lineage

| generation | terminal tag | terminal commit (40-hex) | evidence directory | frozen classification | classification source | scope |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | `substrate-v1-terminal` | `20ba38ed097b6ccfc46bc0b2a34b82923b873aed` | `evidence/substrate/v1` | `certified_cognitive_scaffold` | SUBSTRATE_NOUS_CLOSURE.json → verdict.classification (no SUBSTRATE_*_FINAL_CLASSIFICATION.json in this directory) | Established a certified cognitive scaffold with implemented runtime stages, admission/evidence fabric, and mechanism nulls on three scored gates; no positive category on the final scorecard. |
| V2 | `substrate-v2-terminal` | `4c81bf445c3889f579c28de9a4f079b81aae6743` | `evidence/substrate/v2` | `persistent_developmental_cognition` | SUBSTRATE_V2_FINAL_CLASSIFICATION.json → classification | Established persistent developmental cognition with principal developmental histories and a no_oracle_headroom closure on endogenous allocation. |
| V3 | `substrate-v3-terminal` | `a605fef265a58fa5a8905a61705a111c2140c420` | `evidence/substrate/v3` | `reflective_cognitive_organization` | SUBSTRATE_V3_FINAL_CLASSIFICATION.json → classification | Established reflective cognitive organization under the ordered classification ladder with H_N4 and H_N7 remaining false and independent verification listed among missing conditions. |
| V4 | `substrate-v4-terminal` | `ea22a1dea4a67c8e45c97c8630e412b03ea4e7cf` | `evidence/substrate/v4` | `functional_proto_nous_candidate` | SUBSTRATE_V4_FINAL_CLASSIFICATION.json → classification | Established functional_proto_nous_candidate with open-world pass and independent replication failing to clear SESOI 0.05 (replication mean 0.03098, 95% CI [0.02314, 0.03980]). |
| V5 | `substrate-v5-terminal` | `731cd116191f16247c3c2e99f164502b233f974f` | `evidence/substrate/v5` | `multimodal_nous_ready_for_review` | SUBSTRATE_V5_FINAL_CLASSIFICATION.json → classification | Established multimodal_nous_ready_for_review as the maximum automatic V5 classification on a multimodal sensorium campaign; unqualified_nous remains false. |
| Nous Closure | `substrate-nous-closure-terminal` | `be78aa3a750fb73f103245367ef20215ae8daaf5` | `evidence/substrate/nous_closure` | `terminal_closed_null` | SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json → classification (scientific_status: mechanism_null; outcome: B) | Closed integrated modular advantage versus equal-resource S2 as terminal_closed_null; principal was terminally gated at admission with units_launched=0. |
| Final Revision | `unavailable (tag not present)` | `unavailable (no terminal tag in this checkout)` | `evidence/substrate/final_revision` | `mechanism_null` | SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json → classification (scientific_status: mechanism_null). No SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json is present. Field foundation scientific_status: foundation_feasibility_only. Tag substrate-final-revision-terminal is absent. | Reproduced the historical terminal_closed_null, ran a principal discrimination bed that remains mechanism_null versus S2 and full transcript replay (P1 and P3 mean_paired_effect 0.0, 95% CI [0.0, 0.0], SESOI 0.05), and recorded an adversarial Grok review programme; terminal tag and final classification document are not present in this checkout. |

## Ready and preflight tags

Freeze-then-run discipline as reflected by annotated tags currently present (`git tag --list`). Commits are peeled with `^{}`.

| generation | ready tag | ready commit | preflight tag | preflight commit |
| --- | --- | --- | --- | --- |
| V1 | `substrate-v1-launch-ready` | `facd39e954b34be9549c509b7c797d6297d5da56` | `substrate-v1-pre-three-second-seal` | `facd39e954b34be9549c509b7c797d6297d5da56` |
| V2 | `substrate-v2-developmental-ready` | `553539bc5edc01737fa53b2c4ee4096eb32a2f23` | `substrate-v2-pre-development` | `20ba38ed097b6ccfc46bc0b2a34b82923b873aed` |
| V3 | `substrate-v3-nous-ready` | `36c5c284001b0e4c2eed9cf52471a690bc395314` | `substrate-v3-pre-constitutional-ascent` | `4c81bf445c3889f579c28de9a4f079b81aae6743` |
| V4 | `substrate-v4-structural-ready` | `3eec1611e4f887d3955c6f41604dcceeccbe0f81` | `substrate-v4-pre-structural-understanding` | `a605fef265a58fa5a8905a61705a111c2140c420` |
| V5 | `substrate-v5-sensorium-ready` | `9988a70e418998fcab7b3bb869fba06dd273c811` | `substrate-v5-pre-sensorium` | `ea22a1dea4a67c8e45c97c8630e412b03ea4e7cf` |
| Nous Closure | `unavailable` | `unavailable (no ready tag in this checkout)` | `substrate-nous-closure-preflight` | `731cd116191f16247c3c2e99f164502b233f974f` |
| Final Revision | `substrate-final-revision-ready` | `afebfa2cf931e7495ce344da048f92c01dbdd6fb` | `substrate-final-revision-preflight` | `be78aa3a750fb73f103245367ef20215ae8daaf5` |

## What each null actually says

Preserved nulls with effect, confidence interval, SESOI, and strongest baseline they lost or tied against, as recorded in evidence. Numbers are quoted from the cited files.

### V1 — S1 self-model (SX5) `mechanism_null`

Source: `evidence/substrate/v1/SUBSTRATE_STATE.json` item `S1.result`. Classification `mechanism_null`. Effect estimate 0.025966 with lower 95% CB 0.000752; SESOI 0.05. Directional effects were 0.032393 (har→harth) and 0.019539 (harth→har), both below SESOI. Strongest comparison in the evidence is the naive/fixed-prior/updating error triad on each direction; updating improves calibration but does not clear SESOI 0.05. Reading recorded in evidence: learning the offset roughly halves calibration error, improvement between 0.02 and 0.03.

### V1 — NOUS_CLOSURE mechanism nulls on three gates

Source: `evidence/substrate/v1/SUBSTRATE_NOUS_CLOSURE.json` → `mechanism_nulls` and `verdict.classification` (`certified_cognitive_scaffold`).
- Gate `endogenous_allocation`: classification `mechanism_null_on_this_bed`; oracle -0.051; headroom 0.214. SESOI for the program is 0.05 (recorded on Y5/SX2 evidence). These gates are scored mechanism nulls on this bed; margins on failed gates in `SUBSTRATE_STATE.json` item `Y5` are endogenous_allocation −0.138333, cross_domain_continuity 0.0, procedural_transfer 0.0.
- Gate `cross_domain_continuity`: classification `mechanism_null_on_this_bed`; oracle 0.5; headroom 0.5. SESOI for the program is 0.05 (recorded on Y5/SX2 evidence). These gates are scored mechanism nulls on this bed; margins on failed gates in `SUBSTRATE_STATE.json` item `Y5` are endogenous_allocation −0.138333, cross_domain_continuity 0.0, procedural_transfer 0.0.
- Gate `procedural_transfer`: classification `mechanism_null_on_this_bed`; oracle 0.4375; headroom 0.4375. SESOI for the program is 0.05 (recorded on Y5/SX2 evidence). These gates are scored mechanism nulls on this bed; margins on failed gates in `SUBSTRATE_STATE.json` item `Y5` are endogenous_allocation −0.138333, cross_domain_continuity 0.0, procedural_transfer 0.0.

### V1 — SX2 diversity `closed_no_headroom`

Source: `evidence/substrate/v1/SUBSTRATE_SX2_DIVERSITY.json`. Verdict `closed_no_headroom`; SESOI 0.05; `k_clearing_sesoi` is empty. Best single unmatched accuracy 0.945426. Oracle top-k margins over compute-matched single do not clear SESOI at any recorded k (for example k=1 margin 0.0). Strongest baseline is the compute-matched single cell `gru|medium|linear|horizon_45|h1`.

### V2 — endogenous_allocation `no_oracle_headroom`

Source: `evidence/substrate/v2/SUBSTRATE_V2_FINAL_CLASSIFICATION.json` → `mechanism_nulls.endogenous_allocation`. Classification `no_oracle_headroom`; oracle_residual 0.04570312500000001; margin 0.030468750000000006; SESOI 0.05. The residual is below SESOI, so the bed is gated rather than reporting a positive learned-allocation effect. Related canary headroom in `SUBSTRATE_V2_ALLOCATION_HEADROOM.json` records oracle_residual 0.04375 against SESOI 0.05.

### V4 — independent replication fails SESOI

Source: `evidence/substrate/v4/SUBSTRATE_V4_INDEPENDENT_VERIFICATION.json` → `replication`. Mean effect 0.030980392156862713; bootstrap 95% CI [0.02313725490196074, 0.03980392156862741]; SESOI 0.05; `passes` False. Full arm `full_v4`. Strongest control by history is uniformly `no_counterfactual`. Final state strongest_missing_condition: independent replication effect must clear the preregistered 0.05 SESOI (`SUBSTRATE_V4_FINAL_STATE.json`).

### Nous Closure — `terminal_closed_null` (H_NC20 and instrument pair)

Source: `evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json` (classification `terminal_closed_null`, scientific_status `mechanism_null`, outcome `B`) and `SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json`.

Instrument 2 / H_NC20: candidate minus `S2_monolithic_deterministic_state_machine` mean paired effect 0.0 with 95% CI [0.0, 0.0]; SESOI 0.05; status `mechanism_null`; strongest baseline `S2_monolithic_deterministic_state_machine`. Instrument 2 records candidate_mean_accuracy 1.0 and baseline_mean_accuracy 1.0 with the same CI [0.0, 0.0].

Instrument 1 (stateless / public-cue bed): strongest baseline `S0_stateless_direct_phase_frozen_policy` with baseline_mean_accuracy 0.95125, candidate_mean_accuracy 0.93796875, classification `no_oracle_headroom`, oracle_headroom 0.04874999999999996. Candidate minus strongest stateless effect is -0.013281250000000022 (recorded as −0.01328125 in final-revision immutability and as v5_terminal_full mean_paired_effect −0.013281249999999988 in `SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json`). Oracle headroom 0.04875 is below SESOI 0.05.

Principal result: `SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_RESULT.json` status `terminally_gated`, units_launched 0, reason records H_NC20 below SESOI and instrument 1 no oracle headroom over S0.

### Final Revision — principal `mechanism_null` (P1 and P3)

Source: `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json` classification `mechanism_null`, scientific_status `mechanism_null`, strongest_baseline `S2_task_independent_monolithic_persistent_core`, co_strongest_baseline `full_transcript_replay`.

P3 selected-minus-S2: mean_paired_effect 0.0, 95% CI [0.0, 0.0], SESOI 0.05, histories 96, clears_sesoi False. Strongest baseline tied: `S2_task_independent_monolithic_persistent_core`.

P1 selected-minus-full_transcript_replay: mean_paired_effect 0.0, 95% CI [0.0, 0.0], SESOI 0.05, histories 96. Co-strongest baseline `full_transcript_replay`.

Pilot strongest-baseline document (`SUBSTRATE_FINAL_REVISION_STRONGEST_BASELINE.json`): scientific_status `mechanism_null`, mean_paired_effect 0.0, 95% CI [0.0, 0.0], SESOI 0.05, identity `S2_task_independent_monolithic_persistent_core`.

Closure reproduction (`SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json`) re-records instrument_2 mechanism_null effect 0.0 CI [0.0, 0.0] against `S2_monolithic_deterministic_state_machine` and instrument_1 candidate_effect −0.013281249999999988 with oracle_headroom 0.04874999999999996.

## Immutability

Historical paths that must not be rewritten, as enforced by final-revision preflight and immutability publication:

- `evidence/substrate/nous_closure`
- `artifacts/substrate/nous_closure`
- `configs/substrate/nous_closure`
- `src/substrate/nous_closure*.py`

The preflight drift check also includes explicit source files (not only the namespace globs above):

- `evidence/substrate/nous_closure`
- `artifacts/substrate/nous_closure`
- `configs/substrate/nous_closure`
- `src/substrate/nous_closure.py`
- `src/substrate/nous_closure_campaign.py`
- `src/substrate/nous_closure_config.py`
- `src/substrate/nous_closure_experiment.py`
- `src/substrate/nous_closure_io.py`

Guard code:

- `src/substrate/final_revision_campaign.py:43-45` — `_git_diff_names` runs `git diff --name-only` against the preflight tag for the protected paths.
- `src/substrate/final_revision_campaign.py:620-630` — builds `historical_drift` over the nous_closure evidence/artifacts/config/source paths.
- `src/substrate/final_revision_campaign.py:672-693` — publishes `SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json` with `historical_evidence_untouched: not historical_drift` and status `invalid` if drift is found.
- `src/substrate/final_revision_campaign.py:734` — preflight `all_pass` requires `immutable_document["historical_evidence_untouched"]`.
- `tests/substrate/test_final_revision.py:102-107` — `test_preflight_preserves_historical_closure_null` asserts `historical_evidence_untouched` and immutable null effect 0.0 with CI [0.0, 0.0].

Earlier campaigns also ship generation-local immutability checks (for example `src/substrate/v5campaign.py` `immutability()`, `src/substrate/nous_closure_campaign.py` `immutability()`, and tests under `tests/substrate/test_v5_campaign.py`). Those protect prior version tags and trees relative to each later campaign.

Recorded immutability evidence: `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json` has `historical_evidence_untouched: true`, `historical_diff_from_preflight: []`, `starting_result: terminal_closed_null`.

## How to re-derive any claim

From a clean clone of this repository (read-only verification path; do not run write campaigns against historical evidence):

```bash
git clone <repository-url> substrate && cd substrate
git fetch --tags
# Inspect a generation terminal. Tags are annotated, so peel with ^{} to get commits.
git rev-parse substrate-v1-terminal^{} substrate-v2-terminal^{} substrate-v3-terminal^{} \
  substrate-v4-terminal^{} substrate-v5-terminal^{} substrate-nous-closure-terminal^{}
git rev-parse substrate-final-revision-ready^{} substrate-final-revision-preflight^{}

uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

# Package and test surface (Makefile)
make verify-install
make test
make accept   # substrate verify

# Read sealed classification documents
python -c "import json; print(json.load(open('evidence/substrate/v2/SUBSTRATE_V2_FINAL_CLASSIFICATION.json'))['classification'])"
python -c "import json; print(json.load(open('evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json'))['classification'])"
python -c "import json; d=json.load(open('evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json')); print(d['primary_effects']['H_NC20'])"
python -c "import json; d=json.load(open('evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json')); print(d['classification'], d['effects']['P3_selected_minus_strongest_persistent_alternative'])"

# Final-revision status only (read-only; do not pass write-producing campaign commands)
substrate final-revision status
```

To bind a review to a frozen generation, check out that generation's terminal tag and read only that generation's evidence directory. Do not rewrite files under `evidence/`, `artifacts/`, `runs/`, `proof/`, or `archive/`.

Note: `substrate final-revision verify`, `run`, `freeze`, and `publish` write evidence and are not part of read-only re-derivation.

