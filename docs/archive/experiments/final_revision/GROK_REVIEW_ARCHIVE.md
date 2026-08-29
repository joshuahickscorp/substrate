# Grok Review Archive

Human-readable archive of the Substrate Final Revision adversarial Grok review programme. Tables and quotations are taken from `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json` and `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json`. Reviewer role identities were checked against the frozen `REVIEW_CELLS` tuple in `src/substrate/final_revision_config.py` and the authority `review_cells` list. The executed review contracts live at `runs/substrate/final_revision/grok_contracts/c9dbf03802e2/` — one file per review cell (including `-retry` variants) plus `manifest.json`, each pinning `PUBLIC EVIDENCE COMMIT: c9dbf03802e22ee9c4e3d9852a8d67cd9da0cd08`. That directory is git-ignored working state; the authoritative record of each contract is its `inputs.contract_digest` and `inputs.task_prompt_digest` in the invocation ledger, which the recorder verifies byte-for-byte against the executed `task.md` before crediting a review. Scores and defects below are read from the ledger, not from the contracts.

## Caveat (prominence)

**Grok reviewer grades are not independent external validation, are not a primary endpoint, and cannot override deterministic evidence.** The authority document records `grok_is_not_an_oracle: true` and `grok_agreement_is_not_a_primary_endpoint: true` (`SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json`). Deterministic sealed receipts and classification documents remain the scientific authority.

## Programme summary

- Validated invocations (ledger `validated_output_count`): **34**
- Invocations listed in ledger: 42; output_count: 41; fabricated_outputs: 0; guest_prompt_without_response_count: 1
- Distinct reviewer roles completed: **32**
- Frozen minimum distinct reviewers: **24**
- Preferred target: **48** (upper_target: 64)
- Rounds completed (8): `architecture_proposals`, `blind_independent_review`, `code_and_implementation_review`, `cross_examination`, `final_candidate_review`, `post_pilot_review`, `publication_and_claim_boundary_review`, `test_and_baseline_proposals`
- Review rounds defined in authority (8): `blind_independent_review`, `cross_examination`, `architecture_proposals`, `test_and_baseline_proposals`, `code_and_implementation_review`, `post_pilot_review`, `final_candidate_review`, `publication_and_claim_boundary_review`
- `minimum_complete`: True; `prefreeze_complete`: True; `terminal_complete`: True; `current_blocker`: None
- Architecture proposers counted: 4
- Rejected invocations recorded: 8

The distinct-role count is **32**. That exceeds the frozen minimum of 24 and is below the preferred target of 48. The reviewer role set is the 32-tuple `REVIEW_CELLS` in `src/substrate/final_revision_config.py` (lines 187–220), republished as `review_cells` in `SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json`. Candidate freeze records `scientific_source_edits_after_launch: False` (`SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json`). Adding roles after candidate freeze would require editing that frozen role set (a post-launch source edit relative to the freeze). No additional roles beyond the 32-cell set appear in the authority or ledger.

## Review cells

Every validated invocation with received structured output (not in `rejected_invocations`). Binary score is `output.total_binary_out_of_20` (out of 20). Blocking-defect count is `len(output.blocking_defects)`.

| role | round | invocation id | binary score / 20 | confidence | blocking defects |
| --- | --- | --- | ---: | --- | ---: |
| `historical_evidence_auditor` | `blind_independent_review` | `fr-01-historical-evidence-auditor-pty-20260728-093046` | 0 | high | 6 |
| `closure_null_defender` | `blind_independent_review` | `fr-02-closure-null-defender-pty-20260728-093046` | 0 | high | 7 |
| `closure_null_challenger` | `blind_independent_review` | `fr-03-closure-null-challenger-pty-20260728-093046` | 0 | high | 6 |
| `minimal_architecture_reviewer` | `blind_independent_review` | `fr-04-minimal-architecture-reviewer-pty-20260728-093700` | 0 | high | 5 |
| `radical_architecture_reviewer` | `blind_independent_review` | `fr-05-radical-architecture-reviewer-pty-20260728-093700` | 0 | high | 6 |
| `monolithic_systems_reviewer` | `blind_independent_review` | `fr-06-monolithic-systems-reviewer-pty-20260728-093700` | 0 | high | 5 |
| `hybrid_explicit_latent_reviewer` | `blind_independent_review` | `fr-07-hybrid-explicit-latent-reviewer-pty-20260728-094144` | 0 | high | 7 |
| `graph_relational_dynamics_reviewer` | `cross_examination` | `fr-09-graph-relational-dynamics-reviewer-pty-20260728-094144` | 0 | high | 6 |
| `event_sourced_cognition_reviewer` | `blind_independent_review` | `fr-08-event-sourced-cognition-reviewer-retry-20260728-094954` | 0 | high | 8 |
| `predictive_processing_reviewer` | `cross_examination` | `fr-10-predictive-processing-reviewer-pty-20260728-094837` | 0 | high | 8 |
| `state_space_recurrent_systems_reviewer` | `cross_examination` | `fr-11-state-space-recurrent-systems-reviewer-retry-20260728-095428` | 0 | high | 5 |
| `global_workspace_reviewer` | `cross_examination` | `fr-12-global-workspace-reviewer-pty-20260728-095428` | 0 | high | 6 |
| `sensorium_reviewer` | `architecture_proposals` | `fr-13-sensorium-reviewer-pty-20260728-095428` | 0 | high | 5 |
| `motion_temporal_perception_reviewer` | `architecture_proposals` | `fr-14-motion-temporal-perception-reviewer-pty-20260728-100127` | 3 | medium | 5 |
| `model_fabric_reviewer` | `architecture_proposals` | `fr-16-model-fabric-reviewer-pty-20260728-100127` | 0 | high | 6 |
| `continual_learning_reviewer` | `test_and_baseline_proposals` | `fr-17-continual-learning-reviewer-pty-20260728-100610` | 0 | high | 10 |
| `epistemology_reasoning_reviewer` | `test_and_baseline_proposals` | `fr-18-epistemology-reasoning-reviewer-pty-20260728-100610` | 0 | high | 8 |
| `spatial_3d_reviewer` | `architecture_proposals` | `fr-15-spatial-3d-reviewer-retry-2-20260728-101155` | 0 | high | 5 |
| `self_model_metacognition_reviewer` | `test_and_baseline_proposals` | `fr-19-self-model-metacognition-reviewer-pty-20260728-101155` | 0 | high | 7 |
| `goal_agency_reviewer` | `test_and_baseline_proposals` | `fr-20-goal-agency-reviewer-pty-20260728-101155` | 0 | high | 6 |
| `evaluation_security_reviewer` | `code_and_implementation_review` | `fr-22-evaluation-security-reviewer-pty-20260728-102207` | 0 | high | 8 |
| `runtime_performance_reviewer` | `code_and_implementation_review` | `fr-23-runtime-performance-reviewer-pty-20260728-102207` | 0 | high | 8 |
| `red_team_shortcut_compilation` | `post_pilot_review` | `fr-25-red-team-shortcut-compilation-pty-20260728-102852` | 0 | high | 8 |
| `statistical_reviewer` | `code_and_implementation_review` | `fr-21-statistical-reviewer-retry-20260728-102852` | 0 | high | 7 |
| `red_team_resource_parity` | `post_pilot_review` | `fr-26-red-team-resource-parity-pty-20260728-103213` | 0 | high | 9 |
| `red_team_answer_leakage` | `post_pilot_review` | `fr-27-red-team-answer-leakage-pty-20260728-103419` | 0 | high | 10 |
| `publication_reviewer` | `code_and_implementation_review` | `fr-24-publication-reviewer-pty-20260728-102852` | 0 | high | 8 |
| `red_team_checkpoint_coverage` | `post_pilot_review` | `fr-28-red-team-checkpoint-coverage-pty-20260728-103620` | 0 | high | 12 |
| `red_team_causal_counterfactuals` | `post_pilot_review` | `fr-31-red-team-causal-counterfactuals-pty-20260728-104129` | 0 | high | 8 |
| `red_team_multimodal_counterfeits` | `post_pilot_review` | `fr-29-red-team-multimodal-counterfeits-retry-20260728-104347` | 0 | high | 6 |
| `red_team_activation_security` | `post_pilot_review` | `fr-32-red-team-activation-security-pty-20260728-104506` | 0 | high | 10 |
| `red_team_learning_poisoning` | `post_pilot_review` | `fr-30-red-team-learning-poisoning-pty-20260728-103931` | 0 | high | 8 |
| `minimal_architecture_reviewer` | `final_candidate_review` | `fr-terminal-final-candidate-afeb-retry-20260728-125315` | 0 | high | 0 |
| `publication_reviewer` | `publication_and_claim_boundary_review` | `fr-terminal-publication-boundary-86d-retry-20260728-133501` | 0 | high | 9 |

Total validated cells in table: 34.

## Rejected invocations

Every entry in ledger `rejected_invocations`, with the exact recorded reason. Failed and malformed reviewer responses were recorded rather than discarded.

| invocation id | reason |
| --- | --- |
| `grok-guest-blocked-001` | validation failed: output_received, output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, evidence_commit, activation_false, grade |
| `fr-08-event-sourced-cognition-reviewer-pty-20260728-094144` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, grade |
| `fr-11-state-space-recurrent-systems-reviewer-pty-20260728-094837` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, grade |
| `fr-15-spatial-3d-reviewer-pty-20260728-100127` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, candidate_h_proposal, grade |
| `fr-15-spatial-3d-reviewer-retry-20260728-100610` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, candidate_h_proposal, grade |
| `fr-21-statistical-reviewer-pty-20260728-102206` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, grade |
| `fr-29-red-team-multimodal-counterfeits-pty-20260728-103806` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, grade |
| `fr-terminal-publication-boundary-86d-20260728-132935` | validation failed: output_object, output_digest, output_role, output_round, facets, facet_discussion_credit, falsification, blocking_defects, nonblocking_concerns, concrete_revisions, minority_points, required_narrative, confidence, on_device_transport, grade |

## Preserved disagreements

Every minority or uncertain point and every nonblocking concern recorded on validated outputs, quoted and attributed. No synthesis; no dropping of minority objections.

### Minority or uncertain points

#### `historical_evidence_auditor` — `blind_independent_review` (`fr-01-historical-evidence-auditor-pty-20260728-093046`)

- Minority: owned_state_minus_stateless mean effect ~0.626 with CI well above SESOI could be argued to support binary 1 for crude persistence vs pure stateless policies; rejected here because full_transcript_replay and S2 match the selected kernel, so the gain is not unique owned-architecture cognition.
- Minority: sealed canary mechanism_positives with point CIs might be scored discussion_credit 1.0 for specific micro-mechanisms; auditor keeps 0.5 because fixtures are not integrated/general and constitution forbids architecture/fixture inflation.
- Uncertain without re-execution: whether pilot throughput/digests would bit-reproduce on this host; sealed digests were not recomputed end-to-end.
- Uncertain: external paper source_fact fidelity in the research ledger was not independently re-read from arXiv/GitHub in this cell.

#### `closure_null_defender` — `blind_independent_review` (`fr-02-closure-null-defender-pty-20260728-093046`)

- failure_of_P3_erases_other_capabilities is false in the FR hypothesis graph; a minority view could hold that some supporting mechanisms remain scientifically open on new instruments even though architectural Nous / Outcome A is closed on present evidence.
- Official nous_closure scorecard assigns 0.5 (not 0) to facets 1–19 for cheap-positive implementation; this review assigns discussion_credit 0 because those demonstrations do not survive S2 as architectural Nous credit and must not be averaged into binary wins.
- S2 task compilation risk means the exact null is bed-scoped; uncertainty is about external generalization of the null, not about reversing the historical closed null on the frozen instruments.
- Persistence-vs-stateless gains are real on FR pilot; minority may over-read them as partial Nous—null defender rejects that without modular>S2 and modular>transcript.

#### `closure_null_challenger` — `blind_independent_review` (`fr-03-closure-null-challenger-pty-20260728-093046`)

- Minority: persistence-minus-stateless ≈0.626 could be read as partial credit for facets 1–5 as 'organization active,' but under null-challenger rules and the constitution's requirement to beat strongest fair alternatives, binary remains 0.
- Uncertain without raw receipt recompute: whether every per-history selected and S2 score equality is bit-identical in sealed receipts; authorities report identical means and all-zero raw_differences.
- Minority: task_compilation_risk might mean future beds could separate S2 from a more general policy; that is a revision path, not evidence that the current null is invalid.
- Uncertain: integrated metric may hide rare family-local advantages; family-stratified paired tests were not independently re-derived here.

#### `minimal_architecture_reviewer` — `blind_independent_review` (`fr-04-minimal-architecture-reviewer-pty-20260728-093700`)

- Minority engineering view: typed event logs and checkpoint digests have operational value for restore/audit even if cognitively null; that value must stay outside facet binary scores
- Uncertain whether a future independently authored, non-saturated, generator-held-out bed could separate owned state from transcript; current pilot does not, and speculation cannot raise scores
- S2 is a strong persistent monolith, not a noncognitive fresh control; the null falsifies modular/extra architecture advantage and does not by itself prove persistence is noncognitive—but transcript co-tie currently blocks persistence-as-advantage claims too
- Grok-original candidate H was reserved and not admitted; absence of H is not a defect if no non-decorative proposal exists

#### `radical_architecture_reviewer` — `blind_independent_review` (`fr-05-radical-architecture-reviewer-pty-20260728-093700`)

- Minority credit: owned-state versus pure stateless is SESOI-positive on the pilot; a non-radical persistence claim might survive even though radical architectural advantage does not.
- Uncertainty: without re-execution, residual risk remains that pilot JSON is fabricated—though cross-artifact consistency (ladder, parity, hypothesis graph, immutability, limitations) is high.
- Uncertainty: identical checkpoint byte counts might reflect shared empty-ish serialization rather than identical code paths; either interpretation still falsifies demonstrated radical differentiation.
- Minority: if a future generator-held-out bed makes transcript incomplete/lossy, P1 could reverse; that is hypothetical and not current evidence.

#### `monolithic_systems_reviewer` — `blind_independent_review` (`fr-06-monolithic-systems-reviewer-pty-20260728-093700`)

- Minority charitable view: positive owned_state_minus_stateless and many mechanism_positive canaries show a competent persistent state machine scaffold; this does not reverse binary zeros under the constitution's principal-positive bar.
- Uncertainty: without independent re-execution, absolute numerical scores are trusted as sealed JSON at c9dbf038; the all-zero raw_differences pattern makes silent CI fabrication less plausible but does not replace recomputation.
- S2_anatomy interpretation that 'persistence remained active' is compatible with this review; the disputed inference is any architectural or Nous upgrade beyond the monolith/transcript equivalence class.
- Candidate H remains a procedural hole (no returned Grok-original proposal), not evidence for or against monoliths.

#### `hybrid_explicit_latent_reviewer` — `blind_independent_review` (`fr-07-hybrid-explicit-latent-reviewer-pty-20260728-094144`)

- Minority charitable view: persistent organization (identity/memory/goals fields) is real software state and S2 anatomy says persistence was active even though modularity was unnecessary; that still does not license hybrid or nous advantage.
- Uncertainty: explicit-history positive contrasts on historical HAR/speech streams might someday support longer explicit context, but provisional convergence and closed hybrid gate block elevating them here.
- Uncertainty: a future generator-held-out bed could in principle reveal hybrid value; current recorded evidence does not, and prediction from tournament equivalence is continued null.
- V5 self-label multimodal_nous_ready_for_review conflicts with later terminal_closed_null; this review privileges the later immutable null and statistical package over earlier V5 narrative gates.

#### `graph_relational_dynamics_reviewer` — `cross_examination` (`fr-09-graph-relational-dynamics-reviewer-pty-20260728-094144`)

- Minority charitable reading: append-only relations/causal_edges plus checkpoint integrity are a legitimate minimal substrate for future graph dynamics, even though they currently carry zero behavioral weight.
- owned_state_minus_stateless is a genuine SESOI-clearing effect against weak baselines; minority view may treat this as partial support for persistence facets, but not for graph-relational advantage or facet 20.
- V4 asymmetric causal tree fixtures may still be valid mechanism studies inside their sealed generators; uncertainty is about external validity and baseline strength, not necessarily about software integrity of those archives.
- Without executing decisive beds or replaying raw V5 receipts here, residual uncertainty remains about unpublished variance structures—but published pilot/admission zeros already suffice to refuse architectural advantage.

#### `event_sourced_cognition_reviewer` — `blind_independent_review` (`fr-08-event-sourced-cognition-reviewer-retry-20260728-094954`)

- A minority reading could award binary credit for persistent_identity and goal_continuity as functional systems properties; this review refuses that because the constitution bars architecture presence and because equal-resource S2/transcript ties falsify event-sourced cognitive uniqueness.
- Whether a future non-shared, non-retrieval bed could reveal event-sourcing advantages for auditability under crash/adversarial tampering is open; current beds do not measure that as cognition.
- Historical v4/v5 positive structural campaigns are not re-litigated here; they do not cancel final-revision or nous-closure nulls against S2.

#### `predictive_processing_reviewer` — `cross_examination` (`fr-10-predictive-processing-reviewer-pty-20260728-094837`)

- Minority: persistent explicit state plus checkpoint seals are valuable engineering even if non-predictive; they may be necessary scaffolding for later PP organs without being PP evidence.
- Minority: project's own null interpretation (modularity unnecessary on bed; S2 is monolithic persistent organization) is scientifically careful and should be preserved rather than rebranded as noncognitive.
- Uncertain without re-run: exact wall-clock performance numbers in PERFORMANCE.json; conclusions here do not depend on throughput.
- Uncertain: whether a future non-compiled generator could separate event-sourced projections from transcript replay; current pilot cannot.
- Minority: structural sensorium (real arrays, corruption sensitivity) is stronger than pure label stubs, yet still insufficient for multimodal PP grounding.

#### `state_space_recurrent_systems_reviewer` — `cross_examination` (`fr-11-state-space-recurrent-systems-reviewer-retry-20260728-095428`)

- On this frozen bed, equivalence to full transcript replay could be read as perfect reconstructibility of explicit state rather than absence of all value of persistence; that reading still does not license recurrent architecture advantage and does not overturn P3 null versus S2.
- Structural sensorium and fail-closed event gates may be useful engineering substrates for later dynamical components; usefulness as plumbing is not facet-positive cognitive evidence under this role.
- S2 task-compilation risk means a future fairer general bed might separate selected from S2, but that is a hypothetical not present in sealed evidence; current classification remains mechanism_null.
- Outcome B campaign authorization flags exist, but principal_positive_authorized is false; authorization is not a positive result.

#### `global_workspace_reviewer` — `cross_examination` (`fr-12-global-workspace-reviewer-pty-20260728-095428`)

- Minority: persistence vs stateless (~0.626 pilot effect) shows that some durable state is task-useful; this is not GWA-specific and does not reopen facet binary scores under equal-resource rules.
- Uncertain without re-execution: exact bit-identity of all historical sealed receipts was not recomputed in this cell; digests and JSON at c9dbf038 were treated as sealed claims.
- Minority: a future non-saturated bed might separate event-sourced projections from transcript replay (P1) even if modular GWA still fails; that would not by itself validate F_global_workspace.
- Uncertain: Candidate H and authenticated multi-cell Grok minimum were incomplete at this commit; absence of other cells is recorded, not treated as evidence for or against facets.

#### `sensorium_reviewer` — `architecture_proposals` (`fr-13-sensorium-reviewer-pty-20260728-095428`)

- Discussion credit 0.5 for structural facets may be generous if cheap canaries are later shown to be label-matching rather than mechanism-forced; minority stance would set those discussion_credits to 0 until bind-forced beds exist.
- S2 may be too strong a cognitive baseline (it already stores identity/memory/goals/scene/body/warrants) making modular/H advantages hard to detect on current beds—this is a bed design issue, not evidence of H success.
- A future genuine H might still lose on simplicity to I even if it wins a narrow multimodal bed; selection rule (conjunctive functional then lowest complexity; ties null) could reject useful specialized mechanisms.
- Controlled NumPy sensorium may be the correct pre-acquisition path; uncertainty remains whether any local bind mechanism can clear SESOI without learned features.

#### `motion_temporal_perception_reviewer` — `architecture_proposals` (`fr-14-motion-temporal-perception-reviewer-pty-20260728-100127`)

- Minority: software-level persistent_identity, goal_continuity, and model_fabric replaceability may deserve binary 1 as engineering properties while remaining noncognitive; other reviewers may score all three 0 if facets are defined as Nous-grade only.
- Uncertain: whether a genuine motion-specialized H should ever become the general selected kernel, or remain a replaceable perception/world-model organ under I/S2.
- Uncertain: magnitude of residual-translation-proxy error under rotation/depth change may make MB-TCL fail its own discriminating tests without depth-aware ego-motion.
- Minority objection preserved: simplest-sufficient I selection is justified by current ties; adding H complexity without SESOI-clearing motion evidence would be scientific regression.

#### `model_fabric_reviewer` — `architecture_proposals` (`fr-16-model-fabric-reviewer-pty-20260728-100127`)

- Minority: modular packaging (A) may still have engineering/audit value even when behaviorally null—value would be operational, not cognitive.
- Minority: S2 task-compilation risk means a new held-out generator could still reveal persistent-organization gains without overturning the frozen-bed null.
- Uncertain: whether a true causal-branch mechanism can beat full transcript replay if the transcript already contains the same causal events (may require partial observability or compressed memory budgets to discriminate).
- Uncertain: measured complexity (CPU, bytes, latency) might reorder I vs B/C once weights are empirical rather than declared.
- Grok proposal admission does not by itself move scientific_status; only FT2–FT7 outcomes can.

#### `continual_learning_reviewer` — `test_and_baseline_proposals` (`fr-17-continual-learning-reviewer-pty-20260728-100610`)

- Minority charitable reading: owned-state minus stateless shows large positive effects, so persistence of information matters operationally; this still does not establish continual learning or advantage over strongest equal-resource persistent alternatives (S2/transcript).
- Uncertain without full 32x272 recompute wall-clock: committed pilot means (~0.875) differ slightly from reduced local means (~0.891) due to episode/task_class sampling, but zero paired differences are structural and not sampling artifacts.
- Mutation zero-survivor claim is true for dossier verification in code execution at c9dbf038; whether a committed MUTATION_REPORT artifact exists for this pin is separate and was not found under evidence/substrate/final_revision.
- Candidate H remains ineligible without authenticated Grok-original proposal; this cell does not treat that absence as positive evidence for modular CL architectures.

#### `epistemology_reasoning_reviewer` — `test_and_baseline_proposals` (`fr-18-epistemology-reasoning-reviewer-pty-20260728-100610`)

- Minority: owned-state versus stateless is a real accuracy gap (0.875 vs 0.25) on this bed and supports a narrow claim that some persistent state beats pure cue echo—but it is not advantage over the strongest equal-resource alternative and is not epistemology-specific.
- Minority: knowledge correction making class 3 'unknown' is a genuine shared epistemic rule; it fails as evidence of candidate superiority or of rich epistemology.
- Uncertain without re-execution: whether uncommitted working-tree changes after c9dbf038 alter mutation or experiment code; this review ignores those changes.
- Uncertain: whether a future non-isomorphic generator could reveal S2-vs-selected differences; current evidence forbids projecting such an effect.

#### `spatial_3d_reviewer` — `architecture_proposals` (`fr-15-spatial-3d-reviewer-retry-2-20260728-101155`)

- Minority: a future spatial H could be a valuable organ without being the selected kernel if gains are family-local; that would still be non-kernel and non-Nous.
- Uncertain: whether SE3+occupancy is necessary versus a simpler explicit pose table without voxels; voxel necessity must be ablated separately.
- Uncertain: current source H_causal_temporal_ledger mechanism may already implement intervention-indexed branches after evidence pin; pinned tournament receipt still shows ineligible placeholder provenance—reviewer grades the pinned evidence package and does not assume unpinned runtime superiority.
- Minority objection preserved: simplest-wins after behavioral equivalence may under-explore rare spatial families; remedy is better generators, not narrative elevation of complexity.

#### `self_model_metacognition_reviewer` — `test_and_baseline_proposals` (`fr-19-self-model-metacognition-reviewer-pty-20260728-101155`)

- Minority: owned_state_minus_stateless ~0.626 could be read as weak evidence that some persistent state is useful; majority: it does not clear facet 20 against S2 or transcript and does not implicate self-model allocation.
- Minority: selfmodel.py and metacog.py are unusually honest modules (missing evidence listed, learned refused, transfer null); that is methodological credit, not facet binary 1.
- Uncertainty: sealed pilot means were not re-executed here; if source and sealed JSON diverged silently, recompute would be required. Source paths examined are consistent with sealed zeros and means.
- Uncertainty: decisive beds in code may change later results; at this commit principal/replication results are not sealed, so no positive can be claimed.

#### `goal_agency_reviewer` — `test_and_baseline_proposals` (`fr-20-goal-agency-reviewer-pty-20260728-101155`)

- Minority credit: S2 anatomy and null interpretation correctly refuse to rebrand the monolith as noncognitive and correctly refuse modular necessity—this is scientific hygiene, not a facet pass.
- Uncertain without full recompute: exact pilot mean 0.875086... not re-derived at full 32x272 scale in this cell; structure forces zero paired differences regardless of scale, so classification is robust even if floating means differ slightly from truncated display.
- Minority: sensorium real-array processing is real engineering and could later support grounding tests; it is not currently on the agency causal path for scores.
- Uncertain: whether a future independent generator (authenticated external authorship) could rescue Outcome A eligibility—present commitments explicitly block it.

#### `evaluation_security_reviewer` — `code_and_implementation_review` (`fr-22-evaluation-security-reviewer-pty-20260728-102207`)

- Minority credit: checkpoint/event-chain tamper rejection and activation=false containment are real engineering strengths even though they score 0 on cognitive facets.
- Uncertain without re-run: whether moderate pilot numeric digests still match source if configuration_digest drifts after c9dbf038; source semantics already force P3=0.
- Minority view: discussion_credit 0.5 items (identity restore, goal persistence, epistemic gate, counterfactual schema, sensorium geometry, model registry, conflict defeat) could be labeled 'infrastructure ready' rather than cognitive null-with-credit; binary cognitive score remains 0.
- Uncertain: full decisive_beds runtime cost/behavior was not executed here; planned microepisode counts are source-declared.
- If later authenticated Grok H proposals materialize, Candidate H still requires independent behavioral proof; current ineligibility is correct.

#### `runtime_performance_reviewer` — `code_and_implementation_review` (`fr-23-runtime-performance-reviewer-pty-20260728-102207`)

- A minority view could award engineering credit for identity/goal checkpoint machinery despite binary 0 cognitive facets; this cell refuses binary 1 without behavioral separation from transcript replay.
- Whether S2 should be framed as co-selected rather than baseline is interpretive; anatomy evidence that S2 is monolithic persistent organization on-bed is accepted as null interpretation, not as candidate victory.
- GIL-bound throughput flatness is honestly noted; uncertainty remains only about unmeasured checkpoint costs, not about scaling claims.
- Post-commit working-tree changes may later alter scoring functions; this report applies only to c9dbf038.

#### `red_team_shortcut_compilation` — `post_pilot_review` (`fr-25-red-team-shortcut-compilation-pty-20260728-102852`)

- Minority: owned_state_minus_stateless (mean ~0.626, CI above 0, clears SESOI) shows persistence beats empty stateless policies on this echo bed; this is not architectural advantage and must not be re-labeled as facet 20 or modular Nous.
- Uncertain without re-execution: numeric bootstrap CI endpoints were taken from committed JSON, not independently recomputed in this cell (exact zero differences make CI [0,0] robust).
- Minority: simplest-sufficient selection under explicit null may be the correct engineering posture for Outcome B campaign preparation; it still scores 0 on all 20 cognitive facets as graded here.

#### `statistical_reviewer` — `code_and_implementation_review` (`fr-21-statistical-reviewer-retry-20260728-102852`)

- Owned-state minus stateless does clear SESOI (~0.626 mean on pilot; ~0.628 on small recompute). A minority reading could credit persistent state over empty policies without granting architectural Nous advantage; under the contract, that contrast still does not beat the strongest equal-resource persistent alternative.
- Checkpoint tamper checks, knowledge gates, learning reject/rollback, and activation=false discipline are engineering positives that deserve retention even while all cognitive facet binaries remain 0.
- If an independently authored hidden bed later separates selected from S2 without weakening S2, facet 20 could move; no such evidence exists at this pin.
- Provisional selection of I_simplest_sufficient as simplest sufficient implementation is consistent with a null architecture race, provided no superiority narrative is attached.

#### `red_team_resource_parity` — `post_pilot_review` (`fr-26-red-team-resource-parity-pty-20260728-103213`)

- Minority: owned_state_minus_stateless (~0.626) is a real non-null against fresh/stateless controls and could support a narrow 'persistent projection beats no-state' engineering claim if carefully bounded—still not facet-20 advantage and not Nous.
- Uncertain without re-execution: absolute floating scores depend on class-7 sampling; committed receipts are internally consistent but this cell did not regenerate them.
- Minority: event-sourced receipts and checkpoint round-trip have software integrity value even when cognitively null versus S2; selection of I on complexity is defensible as packaging, not science of mind.
- S2 anatomy correctly warns task_general false and compilation to frozen vocabulary; that limits generalization claims but does not rescue modular advantage (historical null stands).

#### `red_team_answer_leakage` — `post_pilot_review` (`fr-27-red-team-answer-leakage-pty-20260728-103419`)

- Minority credit: owned-state systems do beat cue-only stateless on classes 2–6 under the synthetic instrument; that is real store-and-recall engineering signal, not Nous advantage.
- Minority credit: oracle headroom over S2 exceeds SESOI on class-fraction grounds, so the bed is not score-saturated—but the residual is definitionally unanswerable without sealed_secret, so it cannot license candidate claims.
- Uncertain without re-run: whether a future non-leaky held-out bed would still show selected≡S2≡transcript; current evidence predicts yes for this kernel family.
- S2 anatomy interpretation that modularity was unnecessary while persistence remained active is consistent with the nulls but is not itself a positive cognitive proof.

#### `publication_reviewer` — `code_and_implementation_review` (`fr-24-publication-reviewer-pty-20260728-102852`)

- Minority view: goal_continuity and persistent_identity could receive binary 1 if the facet bar is defined as engineering state continuity rather than cognitive advantage; this review rejects that bar per architecture_presence_is_not_evidence.
- Minority view: multimodal structural processing might earn partial scientific credit as infrastructure readiness; still not grounding cognition on the primary bed.
- Uncertainty: decisive-scale principal/replication/hidden composition receipts at this commit were not fully re-executed here; committed pilot and local probe agree on null, but independent full-scale clean-clone remains the gold check.
- Uncertainty: post-c9db working-tree changes (large diffs in experiment/kernel/verification) may address some defects; they are outside this evidence commit and must not alter this cell's scores.

#### `red_team_checkpoint_coverage` — `post_pilot_review` (`fr-28-red-team-checkpoint-coverage-pty-20260728-103620`)

- Minority: owned_state_minus_stateless (~0.626, CI above SESOI) could license a narrow persistence-vs-stateless claim if carefully scoped; red-team still scores facet 20 and developmental ownership 0 because P1/P3 are the critical architectural/ownership endpoints and both are exact nulls.
- Minority: S2 anatomy’s claim that S2 embodies persistent functional organization (not a dumb control) supports interpreting the null as 'modularity unnecessary' rather than 'persistence noncognitive'; that interpretation still denies selected-kernel advantage.
- Uncertain without re-execution: absolute wall-clock and RSS figures, and whether bootstrap code paths match statistical_authority text; null vectors of zeros do not depend on bootstrap machinery.
- Uncertain: whether provisional selection of I by complexity is the correct engineering freeze; it is consistent with simplest-wins-on-ties but is not a cognitive win.

#### `red_team_causal_counterfactuals` — `post_pilot_review` (`fr-31-red-team-causal-counterfactuals-pty-20260728-104129`)

- Owned-state-minus-stateless effect ~0.626 is large and real, but only shows persistence beats cue-limited/stateless stubs; it does not salvage P3 or causal facets.
- Event-sourced receipts and fail-closed knowledge gates are competent software engineering and may be useful packaging for Outcome B sandbox interfaces without cognitive claims.
- Sensorium feature extraction on synthetic media is nontrivial engineering; integration into scored decisions is the missing link, not array handling itself.
- Whether a future generator-held-out bed could separate I from S2 is open; current evidence does not license that expectation.
- Self-hash verification of evidence JSON under the project's digest schema was not independently reconstructed; content fields used were unambiguous.

#### `red_team_multimodal_counterfeits` — `post_pilot_review` (`fr-29-red-team-multimodal-counterfeits-retry-20260728-104347`)

- Sensorium array/waveform/depth handlers are real numpy feature extractors and may be useful scaffolding for a later sandbox; that engineering value is not cognitive credit.
- Owned-state-minus-stateless is a genuine large effect for persistence vs stubs; a minority view might award narrow non-multimodal persistence discussion credit without any binary facet win.
- Later working-tree canary de-crediting suggests authors already suspect canary overclaim; that does not retroactively repair the c9dbf038 evidence package.
- Whether an independent generator and real corpus could separate I from S2 remains open; present evidence does not license that expectation.
- Grok opinion is not a primary endpoint; this cell only red-teams multimodal counterfeits.

#### `red_team_activation_security` — `post_pilot_review` (`fr-32-red-team-activation-security-pty-20260728-104506`)

- Owned-state vs stateless effect is large, reproducible in receipts, and may support a narrow claim that persistent state beats no-state bookkeeping on this toy bed—without licensing modular/Nous advantage or activation.
- Event-sourced receipts and deterministic restore are real software properties; minority view could credit engineering maturity while still scoring cognitive facets 0.
- Did not re-run the pilot binary; if committed scores were generated by a different source_digest than claimed, independent reproduction could shift confidence—though scoring source at this commit predicts the observed 7/8 structure.
- Whether Outcome-B scale-up is scientifically useful is uncertain: zero-variance P3 implies scaling cannot create architectural positive power on this instrument (program already notes positive_power_estimable false).

#### `red_team_learning_poisoning` — `post_pilot_review` (`fr-30-red-team-learning-poisoning-pty-20260728-103931`)

- Owned-state versus pure stateless is a real SESOI-passing discrimination on this bed; minority view may call that persistence utility, but it is not architectural advantage over S2/transcript and is partly inflated by cue-shared classes.
- Controlled sensorium numeric processing is real engineering; minority may credit multimodal plumbing while still denying grounding-as-cognition.
- Complexity-based selection of I_simplest_sufficient is defensible as engineering parsimony after behavioral equivalence, not as Nous selection.
- Full 104448-episode pilot not re-run here; micro-reproduction matches reported 0.875 ceiling pattern and exact ties, but residual execution-environment risk remains low.

#### `minimal_architecture_reviewer` — `final_candidate_review` (`fr-terminal-final-candidate-afeb-retry-20260728-125315`)

- Whether declared complexity_weight=1.0 is an acceptable engineering default under total tie is a process judgment; as minimal-architecture evidence of cognitive sufficiency it is weak.
- Whether Outcome B terminal seal should wait for completed Grok final_candidate_review/publication rounds (campaign checks imply yes) while pilot already authorizes the null campaign (evidence implies campaign may proceed).
- Did not re-execute full pilot or pytest; confidence in sealed JSON contents is high from direct parse plus spot recomputation, not full regeneration.
- S2 anatomy reports source in nous_closure_experiment.py (69 lines) while pilot S2 is TaskIndependentMonolithicPersistentCore in final_revision_experiment.py—control lineage labeling is slightly messy though functional dual is present.
- Minority view: P1 vacuity and shared-core tournament could be elevated to blocking if the program intends to claim confirmatory multi-baseline architecture tests rather than an Outcome B null seal.

#### `publication_reviewer` — `publication_and_claim_boundary_review` (`fr-terminal-publication-boundary-86d-retry-20260728-133501`)

- Minority defensible claim: persistent organization is load-bearing versus pure stateless controls (owned_state_minus_stateless passes), even though modularity and transcript-irreducible ownership are nulls.
- S2 anatomy interpretation that modularity was unnecessary because S2 already implements the same persistent functional organization is consistent with exact ties, but S2 is task-compiled to the frozen vocabulary and is not an unrestricted general policy—over-generalization remains a risk.
- Severity of cue-injection as a contamination of absolute scores is high for classes 0-1; whether remaining classes 2-6 are fully non-leaky was not re-audited line-by-line beyond result summaries—uncertainty retained without upgrading claims.
- Grok challenge packs returned zero credited challenges; absence of external challenge packs limits adversarial coverage but does not itself create positive cognitive evidence.
- If future generator-held-out beds produce SESOI-positive selected-minus-S2 after Holm with replication and hidden composition, facet 20 and related binaries could be reopened; current frozen evidence forbids that reopening.

Total minority/uncertain points: 141.

### Nonblocking concerns

#### `historical_evidence_auditor` — `blind_independent_review` (`fr-01-historical-evidence-auditor-pty-20260728-093046`)

- Cheap canaries report many mechanism_positive exact-point CIs on designed fixtures; risk of narrative inflation if fixtures are misread as open-world cognition.
- Architecture tournament selection of I_simplest_sufficient is provisional (complexity tie-break among behaviorally equivalent prototypes), not a cognitive win.
- S2 task_compilation_risk: hand-authored transition/query vocabulary for frozen families limits generalization claims either for candidate or baseline.
- Generator commitments not authored by an independent authenticated Grok cell; Outcome A isolation limit self-recorded.
- Preflight remote match was sealed at older commits; this audit did not re-validate remotes at c9dbf038.
- v4/v5 scientific-status prose still present and easy to misread as current integrated authority.

#### `closure_null_defender` — `blind_independent_review` (`fr-02-closure-null-defender-pty-20260728-093046`)

- S2 and sandbox transition/query vocabularies are hand-authored and task_general false (S2 anatomy); exact null is valid on the frozen bed but generalization of the null beyond that bed is limited—still insufficient to reverse architectural Nous claims.
- FR cheap canaries report several mechanism_positive effects vs S2 on local fixtures; these do not override integrated pilot and H_NC20 primary endpoints.
- Hypothesis graph states failure_of_P3_erases_other_capabilities false; non-P3 claims could be pursued on new beds, but current evidence does not establish them as architectural Nous.
- owned_state_minus_stateless on FR pilot passes SESOI (~0.626); persistence-vs-nothing is real but is shared with S2/transcript-class systems and is not modular architectural advantage.
- Candidate H / Grok-original slot ineligible without authenticated external proposals; does not rescue modular claims.
- External activation remains false throughout authorities.

#### `closure_null_challenger` — `blind_independent_review` (`fr-03-closure-null-challenger-pty-20260728-093046`)

- S2 transition/query vocabulary is hand-authored for frozen families (task compilation / answer-encoding risk for generalization), yet this is a limit on future beds, not a reason to weaken the historical null.
- Cheap canaries report many mechanism_positive effects with S2 as named strongest_baseline while the large pilot shows exact selected-S2 ties; canaries must not be promoted over the pilot.
- Integrated accuracy metric may mask facet-local differences; per-facet principal campaigns never launched.
- Host free_disk_gib_observed≈68 and torch_installed_in_frozen_environment=false constrain acquisition tournaments.
- Candidate H (Grok-original) not admitted; Grok opinion still not an endpoint.
- Persistence-minus-stateless effect ≈0.626 is real and SESOI-clearing, but proves persistent organization value, not candidate advantage over S2/transcript.

#### `minimal_architecture_reviewer` — `blind_independent_review` (`fr-04-minimal-architecture-reviewer-pty-20260728-093700`)

- Event-sourced receipts can remain as operational audit without any Nous claim—research ledger already states this boundary correctly
- Selecting lowest complexity among behavioral ties is acceptable engineering if labeled provisional and non-cognitive
- Preflight claims historical namespace/effect/interval/baseline preservation; not re-sealed here but internally consistent with immutable null policy
- Host disk/torch constraints and zero model acquisition are honest; they further block multimodal and fabric cognitive claims
- S2 task_compilation_risk acknowledges hand-authored vocabulary; that limits generalization claims for S2 as well as the candidate, but does not convert the measured null into an advantage

#### `radical_architecture_reviewer` — `blind_independent_review` (`fr-05-radical-architecture-reviewer-pty-20260728-093700`)

- Pilot owned_state_minus_stateless clears SESOI (~0.626) with headroom ~0.125—persistence beats amnesia but not transcript or S2.
- Research ledger correctly separates source fact from Substrate inference and prefers simplest event-sourced kernel; that caution is not yet matched by differentiated radical prototypes.
- Candidate H remains an empty Grok-original slot; radical causal-temporal claims cannot be scored as implemented.
- S2 task_compilation_risk and generator-held-out generalization limits are acknowledged but still leave the recorded nulls intact.
- Preflight/immutability claims of untouched historical namespaces were not re-verified by full tree hash recompute in this cell.

#### `monolithic_systems_reviewer` — `blind_independent_review` (`fr-06-monolithic-systems-reviewer-pty-20260728-093700`)

- Owned-state-minus-stateless positive effect (~0.626) is real on the pilot bed but answers the wrong comparison for Nous/architectural claims; it must not be re-narrated as advantage over S2.
- S2 and selected kernel share frozen transition/query vocabulary risk (task_general false in S2_anatomy); nulls are valid on the bed but generalization remains limited.
- Cheap canaries pass (21/21) with expected nulls preserved for identity_after_process_replacement and open_world_composition; positive canaries remain fixture-scoped.
- Host free disk ~68 GiB and no torch in frozen environment constrain any future non-decorative acquisition; decorative downloads correctly refused.
- Working-tree HEAD differed from evidence commit at review time; reviewers must pin c9dbf038 object reads, not assume dirty tree equality.

#### `hybrid_explicit_latent_reviewer` — `blind_independent_review` (`fr-07-hybrid-explicit-latent-reviewer-pty-20260728-094144`)

- Explicit-history length effects on historical temporal beds include positive histmlp contrasts, but many cells are provisional_unconverged_or_unmeasured and do not license a hybrid-core selection.
- Cheap canaries report fixed within-fixture effects under the selected monolith; these can be misread as general cognition if scope discipline slips.
- S2 query/transition vocabulary is task-compiled to frozen families (task_general false), limiting generalization of both the null and any positive construction.
- Candidate H remains a placeholder without admissible external proposal; not a hybrid win condition.
- Working tree after the evidence commit contains unrelated local edits; future reviews must re-pin commits carefully.

#### `graph_relational_dynamics_reviewer` — `cross_examination` (`fr-09-graph-relational-dynamics-reviewer-pty-20260728-094144`)

- _kernel_answers classes 0–1 return cue fields rather than state-derived observations/instructions, weakening claims that answers are pure owned-state projections.
- Candidate H remains an ineligible placeholder; its absence does not affect the null, but it blocks any claim that the tournament exhausted causal-temporal graph designs.
- Research survey correctly treats graph dynamics as optional world-model projection, yet packaging still names graph/relation interfaces that can be misread as earned dynamics.
- owned_state_minus_stateless SESOI pass is real but only against weak baselines; it must not be rebranded as relational or modular advantage.
- Nous-closure facets 1–19 were terminally gated after H_NC20 admission null; discussion credit for implementation should not be inflated into binary cognitive credit.

#### `event_sourced_cognition_reviewer` — `blind_independent_review` (`fr-08-event-sourced-cognition-reviewer-retry-20260728-094954`)

- Research ledger correctly separates event sourcing as auditable persistence from cognition; implementation still risks over-claiming via digests, receipts, and modality labels
- Class 0/1 answers in the discrimination bed are taken from the cue object rather than forced through observation-state retrieval, weakening claims that owned perceptual state drives all scored behavior
- Candidate H remains a placeholder without genuine external Grok-original provenance and cannot compete
- Generator commitments are content-addressed but not authored by an independent authenticated Grok cell; Outcome A isolation is already self-declared limited
- Real-world sandbox readiness is interface/smoke only; desktop/code/document/video/audio/3D campaigns remain future work
- Workspace HEAD can drift from the pinned evidence commit; reviews should pin and rehash sealed authorities before any later promotion

#### `predictive_processing_reviewer` — `cross_examination` (`fr-10-predictive-processing-reviewer-pty-20260728-094837`)

- Research survey correctly treats learned world models/predictive video as optional costly adapters, yet tournament still labels G as predictive without PP substance.
- S2 anatomy correctly warns task-compiled vocabulary and monolithic persistence; project interpretation that modularity is unnecessary on the bed is consistent with nulls.
- Sensorium camera-motion separation is a bounded translation proxy; speech has no transcription claim—appropriate, but further weakens multimodal PP narratives.
- Candidate H remains ineligible without authentic Grok-original proposal; architecture search incomplete relative to stated Grok cell requirements.
- Outcome A remains blocked by generator isolation_limit and incomplete adversarial Grok review ledger at evidence commit.

#### `state_space_recurrent_systems_reviewer` — `cross_examination` (`fr-11-state-space-recurrent-systems-reviewer-retry-20260728-095428`)

- v5 README/terminal narrative (multimodal_nous_ready_for_review, positive primary effects) conflicts with final-revision provisional kernel and mechanism_null; historical namespaces must remain immutable but external readers can conflate generations.
- Cheap canaries largely test field presence, checkpoint round-trip, and activity-receipt ablation—not open-world recurrent competence.
- S2 is task-compiled around frozen vocabularies (anatomy acknowledges task_general:false) while remaining the correct equal-resource strongest baseline on this bed; generator-held-out beds are still required before generalizing the null.
- Candidate H causal-temporal ledger lacks admissible external Grok-original proposal and is ineligible.
- Continuity lane wall-clock loops can inflate process-boundary theater without adding state-space dynamics.

#### `global_workspace_reviewer` — `cross_examination` (`fr-12-global-workspace-reviewer-pty-20260728-095428`)

- Owned state vs pure stateless direct policy shows large pilot effect (~0.626); persistence is active but does not license modular/GWA advantage or Nous claims.
- Selected kernel I_simplest_sufficient is a complexity win among behavioral ties, not an evidence of superior workspace dynamics.
- Full transcript replay co-ties S2 and selected at ~0.875—semantic content is recoverable from the event log, weakening claims that a distinct workspace substrate is doing extra cognitive work on this bed.
- Research survey already states GWA can be decorative without ablation-changing decisions; code matches that risk.
- No real models/corpora acquired; multimodal and model-fabric facets remain structural.
- Working tree has post-commit dirty edits not in c9dbf038; future readers must pin the evidence commit.

#### `sensorium_reviewer` — `architecture_proposals` (`fr-13-sensorium-reviewer-pty-20260728-095428`)

- Cheap canaries labeled mechanism_positive can be misread as principal evidence if digests/receipts are treated as cognition.
- S2 query/transition vocabulary is bed-compiled (task_general=false), limiting generalization claims even when the architectural null is valid.
- Full transcript replay co-strongest with selected kernel raises continuity-vs-replay confounds for identity claims.
- No real model/corpus acquisition under disk/dependency envelope; fabric and training authorities remain structural.
- Complexity weights differ while fixture behavior ties, so simplicity selection is not scientific superiority.

#### `motion_temporal_perception_reviewer` — `architecture_proposals` (`fr-14-motion-temporal-perception-reviewer-pty-20260728-100127`)

- Selected I may remain simplest-sufficient even after a genuine H if motion-specific gains fail to transfer to non-motion families.
- No real models/corpora acquired; tensor organs deferred under disk/env envelope.
- Stages report grok_post_pilot_review blocked_on_grok_authentication and moderate_integrated_pilot pending in tournament stages blob.
- Cheap canary 'mechanism_positive' labels are fixture-scoped and must not be promoted to open-world facet wins.
- Complexity weight 7.0 for H slot will lose simplicity ties unless a discriminating behavioral margin >= SESOI appears under equal resources.

#### `model_fabric_reviewer` — `architecture_proposals` (`fr-16-model-fabric-reviewer-pty-20260728-100127`)

- Shared prototype_source_lines_shared=429 and identical failure_modes text across candidates indicate catalog diversity is largely labeling.
- S2 is compiled to a frozen event/query vocabulary (task_general false)—limits generalization claims without invalidating the exact null.
- Oracle headroom ~0.125 on pilot exceeds SESOI numerically but unused by any persistent system above S2/transcript.
- Complexity weights are declared, not measured (LOC/cost identical across candidates in catalog).
- Sensorium and sandbox readiness are interface smoke scopes, not open-world evaluation.
- Working-tree drift relative to pin risks future re-runs disagreeing with pinned digests if operators execute dirty sources.

#### `continual_learning_reviewer` — `test_and_baseline_proposals` (`fr-17-continual-learning-reviewer-pty-20260728-100610`)

- Oracle headroom on the new bed (~0.125 over S2) is real and preferred, so saturation is not the null cause; null is mechanism equivalence among persistent systems.
- Selected kernel I is provisionally simplest-sufficient S2-derived; historical S2 allowed to win; this is scientifically conservative but still not continual-learning evidence.
- Resource parity flags assert selected/S2 equality; weaker baselines may still be informationally starved by construction of answer tables.
- Principal-scale campaign plan exists but positive power is non-estimable under zero-variance pilot P3; scaling episodes alone cannot create architectural advantage if systems remain equivalent.
- Real-world sandbox readiness artifacts are interface smoke only; external activation false.
- Working-tree drift relative to the pinned evidence commit increases risk of silent post-pin claim changes if future reviews use dirty trees.

#### `epistemology_reasoning_reviewer` — `test_and_baseline_proposals` (`fr-18-epistemology-reasoning-reviewer-pty-20260728-100610`)

- Owned-state vs stateless effect is large on this bed but is memory readout, not epistemology or method selection, and does not beat S2 or transcript
- Resource parity flags between selected and S2 are asserted true and consistent with identical scripts; fairness among weak baselines is less diagnostic because several share the same answer maps
- Candidate tournament ties all eligible prototypes at bounded_fixture_accuracy 1.0 then selects lowest complexity; simplest-wins is a design choice, not a cognitive result
- src/substrate/epistemology.py and method/voi.py are not connected to final_revision discrimination scoring
- Performance receipts (~44k episodes/s, GIL-bound) are consistent with pure hashing/lookup and should not be read as cognitive throughput
- Mutation list includes role-relevant items (counterfactual undeclared variables, intervention-as-observation, unsupported belief-as-knowledge) but only as dictionary toggles

#### `spatial_3d_reviewer` — `architecture_proposals` (`fr-15-spatial-3d-reviewer-retry-2-20260728-101155`)

- Sensorium depth/mesh/point_cloud paths are valuable organs but risk being treated as kernel evidence.
- Fixture door position [1,0,0] can create illusory spatial content without occupancy/contact dynamics.
- If Candidate-H adds branch_store without a spatial discrimination bed, it will lose to I on complexity_weight 7.0 vs 1.0.
- Equal-resource full_transcript_replay matching S2 scores warns against mistaking persistence for geometry.
- External activation remains false; sandbox readiness docs are not cognitive results.

#### `self_model_metacognition_reviewer` — `test_and_baseline_proposals` (`fr-19-self-model-metacognition-reviewer-pty-20260728-101155`)

- Architecture tournament selects I_simplest_sufficient by lowest complexity among behaviorally equivalent prototypes sharing one projection; selection is not a cognitive win.
- S2 is correctly characterized as a monolithic persistent core with equal opportunity, not a noncognitive fresh control; null interpretation is consistent.
- owned_state_minus_stateless positive effect must not be re-narrated as Nous or self-model advantage over strongest baselines.
- epistemic_defeaters and conflict_coherence canary predicates are misaligned with their names.
- No real models or corpora acquired; model-fabric and multimodal claims stay structural.
- Decisive principal/replication/hidden beds are coded but principal result artifacts are absent at this commit; decisive_plan admits positive power is not estimable from zero-variance P3 pilot.

#### `goal_agency_reviewer` — `test_and_baseline_proposals` (`fr-20-goal-agency-reviewer-pty-20260728-101155`)

- Oracle headroom is engineered primarily via class 7 sealed_secret unreachable to all non-oracle systems (~0.125).
- decisive_plan correctly states positive power is not estimable from zero-variance P3; any large null campaign is Outcome-B confirmation, not power for a positive architectural claim.
- Architecture tournament equates all eligible prototypes on a shared EventSourcedKernel projection with decorative activity counters—representation differences are not behavioral.
- Candidate H remains a reserved empty slot without returned external proposal.
- Separate goals.py GoalSystem is more careful about external authority than the pilot kernel goal events, but is unused by the discrimination bed.

#### `evaluation_security_reviewer` — `code_and_implementation_review` (`fr-22-evaluation-security-reviewer-pty-20260728-102207`)

- Activation remains false in config, checkpoints, security, and readiness smoke; external_execution_authorized True is refused — good containment, not capability.
- Null interpretation docs and selected-kernel provisional status correctly refuse architectural advantage claims.
- Checkpoint restore integrity checks (checkpoint_digest, event chain, state agreement) are relatively strong engineering controls.
- Sensorium processes real structures without hidden labels but is disconnected from the scoring bed.
- Independent recomputation only re-aggregates raw_history_scores; it does not re-run generators from seeds as a third-party reimplementation.
- Candidate H is correctly ineligible without authenticated Grok-original proposal; slot remains a placeholder.
- Working tree at review time differed from evidence commit; reviewers must pin c9dbf038 explicitly.

#### `runtime_performance_reviewer` — `code_and_implementation_review` (`fr-23-runtime-performance-reviewer-pty-20260728-102207`)

- Activation is correctly false in config, checkpoints, CLI stop/resume, and activation audit (true_activation_found false).
- Checkpoint restore rejects digest/state inconsistency; covered_state_keys match owned projection maps.
- Sensorium numeric processing is real but limited to generated controlled media; project docs already disclaim open-world corpora.
- Continuity lane preferred 43200s is not evidenced as completed at c9dbf038; only unit-scale process-boundary tests are present.
- Working tree has large post-c9dbf038 drift in final_revision modules; future reviews must pin the evidence commit explicitly.
- Candidate H remains an empty reserved slot without an admissible external proposal.
- Resource parity claims for selected vs S2 are plausible on this micro-bed (same events) but baselines such as largest_model_always are collapsed to stateless answers, weakening the ladder as a general equal-resource suite.

#### `red_team_shortcut_compilation` — `post_pilot_review` (`fr-25-red-team-shortcut-compilation-pty-20260728-102852`)

- Oracle headroom 0.1249 exceeds SESOI 0.05 and preferred 0.10—bed is non-saturated, so the null is informative rather than a ceiling artifact.
- Null interpretation document correctly refuses modular Nous advantage and treats S2 as monolithic persistent organization, not a noncognitive straw control.
- Resource pilot reports runtime/RSS/throughput with candidate_complexity_weight 1.0; resource_tradeoff_claimed_as_cognitive_advantage false.
- Selected kernel is explicitly provisional with architectural_advantage null and historical_s2_allowed_to_win true.
- Failure-matrix and activation audit claim fail-closed integrity checks with true_activation_found false—useful engineering, not cognition.
- Candidate H remains placeholder-ineligible until a genuine Grok-original proposal exists.

#### `statistical_reviewer` — `code_and_implementation_review` (`fr-21-statistical-reviewer-retry-20260728-102852`)

- Generator isolation_limit admits non-independent authorship; Outcome A is self-declared ineligible on this evidence alone.
- Bootstrap CI under all-zero paired differences is degenerate ([0,0]); correctly fails passes, but cannot estimate power for a positive alternative—decisive_plan already records positive_power_estimable false.
- Model baselines (largest_model_always, routers, ensembles) are stubs aliased to stateless/summary answer dicts, not real model calls under cost accounts.
- Sensorium and modality distinctness are local numeric smokes decoupled from the scored bed and external corpora.
- Continuity lane background work is non-task hashing; long-horizon cognitive load is unmeasured.
- Candidate H remains a non-admitted placeholder without Grok-original provenance.
- Authenticated Grok cell minimum incomplete; external review ledger cannot yet credit multi-role consensus.
- Working tree at review time differed from the pin; only commit-pinned blobs and evidence JSON were treated as authoritative.

#### `red_team_resource_parity` — `post_pilot_review` (`fr-26-red-team-resource-parity-pty-20260728-103213`)

- architecture_tournament.stages.moderate_integrated_pilot remains 'pending' while SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json exists with scientific_status mechanism_null—status ledger inconsistency.
- grok_post_pilot_review stage blocked_on_grok_authentication; Candidate H still not_admitted_without_a_returned_grok_proposal.
- Generator isolation_limit admits non-independent Grok authorship of commitments; Outcome A correctly ineligible on this alone.
- owned_state_minus_stateless is large (~0.626, CI entirely above SESOI) and correctly shows weak controls lose; it must not be relabeled as P3 or Nous advantage.
- Historical terminal_closed_null on saturated bed (null interpretation: first instrument headroom 0.04875 < SESOI) remains immutable and separate from this non-saturated pilot null.
- Selected kernel is S2-derived event-sourced monolith; complexity_weight 1.0 wins simplicity audit only.
- Activation false everywhere audited; no external activation claim is supported.

#### `red_team_answer_leakage` — `post_pilot_review` (`fr-27-red-team-answer-leakage-pty-20260728-103419`)

- Candidate H_causal_temporal_ledger ineligible without returned Grok-original proposal; tournament originality incomplete.
- Grok post-pilot review stage recorded blocked_on_grok_authentication in architecture tournament stages at this commit.
- No real models or corpora acquired; multimodal and model-fabric claims remain structural.
- Runtime ~0.52s for 104448 microepisodes is consistent with pure in-process synthetic scoring, not media/model workloads.
- S2 anatomy admits task_general false and task-compiled vocabulary; equal-resource co-strongest is also full_transcript_replay.
- Activation remains false throughout evidence objects.

#### `publication_reviewer` — `code_and_implementation_review` (`fr-24-publication-reviewer-pty-20260728-102852`)

- Checkpoint/event digest tamper rejection and exact restore are well engineered and tested.
- Activation remains false in config, checkpoints, and activation audit; readiness refuses external_execution_authorized proposals.
- Sensorium is honest about synthetic scope and avoids hidden semantic labels; speech transcript is None.
- Historical terminal_closed_null reproduction path and immutability checks are present in campaign code; null interpretation document is appropriately cautious.
- Candidate H is correctly ineligible without Grok-original proposals.
- identity_digest naming is misleading (state hash vs entity identity).
- Working tree after c9db contains substantial further final_revision edits; publication must pin the reviewed commit and not mix later code into claims.
- Full principal/replication/hidden beds not re-run at decisive scale in this cell; committed pilot is consistent with local probe but is still not a substitute for independent full recompute in a clean clone.

#### `red_team_checkpoint_coverage` — `post_pilot_review` (`fr-28-red-team-checkpoint-coverage-pty-20260728-103620`)

- Selected kernel I_simplest_sufficient is provisionally justified only by simplicity under total behavioral equivalence—acceptable engineering choice if framed as null architecture result, not Nous advantage.
- owned_state_minus_stateless mean≈0.626 (≈5/8) is a strong persistence-vs-stateless effect but is a weaker claim than P1/P3 and must not be relabeled as architectural or open-world advantage.
- S2 anatomy correctly states S2 is not a noncognitive control; treating S2 loss as 'persistence is cognitive' would still not establish modular or selected-kernel superiority.
- outcome_b_campaign_authorized=true with zero-variance P3 must not be narrated as powered positive architectural campaign design.
- Failure matrix only injects software/integrity faults; no cognitive falsifiers at pilot scale.
- Resource pilot runtime≈0.52s for 104448 microepisodes indicates synthetic scoring cost, not embodied resource stress.

#### `red_team_causal_counterfactuals` — `post_pilot_review` (`fr-31-red-team-causal-counterfactuals-pty-20260728-104129`)

- Tournament selection of I_simplest_sufficient is a complexity tie-break among behaviorally equivalent fixtures, not a win on discriminative cognition.
- S2 anatomy explicitly warns task compilation risk and that S2 is a monolithic representation of the same persistent organization, not a noncognitive fresh control.
- Generator isolation_limit states commitments were not authored by an independent authenticated Grok cell; Outcome A ineligible on this evidence alone.
- Resource pilot throughput ~2e5 episodes/s and ~0.52s runtime indicate a synthetic bookkeeping bed, not open-world load.
- Model fabric and real-model acquisition remain empty; sensorium is generated media structure handling by its own limitations text.
- Candidate H rejected for missing genuine Grok-original proposal; no H proposal is in evidence.
- Workspace at review time was dirty relative to evidence commit; review correctly ignored uncommitted drift.

#### `red_team_multimodal_counterfeits` — `post_pilot_review` (`fr-29-red-team-multimodal-counterfeits-retry-20260728-104347`)

- Owned-state-minus-stateless mean effect ~0.626 clears SESOI but only shows persistent string/state bookkeeping beats cue-limited stubs; it does not salvage multimodal facets or P3.
- Oracle headroom approximately matches the sealed_secret/class-7 mass (~1/8); headroom is largely oracle-only class coverage, not unused multimodal competence.
- Generator commitments isolation_limit states they were not authored by an independent authenticated Grok cell.
- Tournament selection of I is simplicity among tied fixtures, not a multimodal win.
- S2 anatomy admits task-compilation risk and equal sensor access; S2 is a co-strong persistent organization, not a noncognitive straw man.
- Working tree after c9dbf038 contains later canary de-crediting edits; those edits are not the graded evidence surface.

#### `red_team_activation_security` — `post_pilot_review` (`fr-32-red-team-activation-security-pty-20260728-104506`)

- architecture_tournament.stages still lists moderate_integrated_pilot as pending while moderate pilot JSON reports mechanism_null at scale—status hygiene inconsistency.
- owned_state_minus_stateless is large and real (~0.626) but is a persistence-vs-no-state contrast, not P3 architectural advantage; must remain demoted.
- Runtime resource pilot (~0.52s for 104448 microepisodes) is consistent with pure deterministic bookkeeping, not multimodal perception or model inference.
- Candidate H (Grok-original) not admitted; external architecture search incomplete by program's own ledger.
- Shared explicit projection failure modes are self-declared across rejected candidates and correctly limit representational conclusions.
- Historical instrument-1 saturated headroom (~0.04875 < SESOI) remains a closed null path; final-revision bed must not be marketed as solving that without class-7 reform.

#### `red_team_learning_poisoning` — `post_pilot_review` (`fr-30-red-team-learning-poisoning-pty-20260728-103931`)

- Oracle headroom ~0.1249 (>SESOI and preferred 0.10) exists because class 7 is sealed oracle-only; headroom is not candidate advantage.
- S2 anatomy correctly notes S2 is monolithic persistent organization, not a noncognitive fresh control; modular advantage remains null.
- Architecture tournament: all nine candidates accuracy 1.0 and shared semantic_state_digest; I wins only on declared complexity_weight 1.0.
- Generator commitments not authored by independent authenticated Grok cell; Outcome A ineligible on this evidence alone (self-declared isolation_limit).
- Performance scaling is GIL-bound Python; high throughput does not license cognitive claims.
- Working tree at inspection time differs from evidence commit; only c9dbf038 content used.

#### `minimal_architecture_reviewer` — `final_candidate_review` (`fr-terminal-final-candidate-afeb-retry-20260728-125315`)

- Architecture tournament candidates A–I all wrap the same EventSourcedKernel core (shared_core_state_digest identical; prototype_source_lines_shared=604); selection of I is min(declared complexity_weight=1.0), not measured unique implementation mass or pilot discrimination among candidates.
- Moderate pilot scores selected_candidate as bare EventSourcedKernel, not ArchitecturePrototype('I_simplest_sufficient'); representation-specific side activity is unused in P1/P3 behavioral scoring.
- P1 selected-minus-full_transcript_replay is structurally near-tautological: both systems are EventSourcedKernel + identical _kernel_answers on the same event stream, so a zero effect is algorithmically expected rather than an independent equal-resource contrast.
- Class 7 composition is intentionally unanswered by selected/S2/transcript (_kernel_answers omits key 7), supplying oracle headroom without testing candidate composition competence.
- Endogenous plastic field adds large co-located code (final_revision_field.py ~4k LOC) at freeze; isolation claims (classification_credit=0, current_campaign_endpoint_credit=0, foundation_feasibility_only) are documented and tested but remain a claim-boundary watch surface for principal receipts.
- Field canary credit key is current_final_revision_endpoint_credit while freeze/final-state use current_campaign_endpoint_credit (both zero); naming drift only.
- docs/archive/experiments/final_revision/LIMITATIONS.md still states Candidate H has no admissible proposal, but evidence admits H_causal_temporal_ledger to the bounded tournament after adjudication (then rejects on complexity).
- Grok authority is externally_blocked/terminal_complete=false (missing final_candidate_review and publication_and_claim_boundary_review); this blocks terminal Outcome B seal per campaign outcome_b_checks.grok_swarm_complete, not pilot authorization of the null campaign.
- Challenge authority is valid_for_outcome_b_null_not_outcome_a; Outcome A isolation is incomplete by design.
- Decisive Outcome B campaign artifacts are not yet present; resource/determinism of the planned ~1.77M microepisode scale is planned, not re-verified here.

#### `publication_reviewer` — `publication_and_claim_boundary_review` (`fr-terminal-publication-boundary-86d-retry-20260728-133501`)

- owned_state_minus_stateless is a large positive Holm-significant effect (~0.625) on all three beds; it may be discussed only as statefulness versus pure stateless control, never as P1/P3 or architectural Nous advantage.
- Eleven readiness JSON contracts exist under real_world_sandbox_readiness and assert no_unqualified_nous, but missing manifest and incomplete terminal verification still block Outcome B readiness language.
- Real equal-resource learned/model baselines were largely unavailable; strongest executed persistent comparator is S2/transcript, not a downloaded frontier model.
- Candidate H proposals exist in adjudication ledgers as non-endpoint architecture hypotheses; they must stay non-selected and non-credit-bearing unless future SESOI pilots pass.
- Docs (ARCHITECTURE_SPEC, LIMITATIONS, NULL_INTERPRETATION) are largely claim-disciplined; residual risk is narrative compression that drops 'exact tie = null' or over-sells oracle headroom.

Total nonblocking concerns: 204.

## Blocking defects and their resolutions

Each blocking defect from validated outputs, the role that raised it, and its recorded resolution from the same invocation row (`resolutions` / `resolved_blocking_defects`). If a defect has no matching resolution object, it is listed as **unresolved**.

### `historical_evidence_auditor` — `blind_independent_review` (`fr-01-historical-evidence-auditor-pty-20260728-093046`)

- **Defect:** Immutable historical terminal_closed_null: integrated candidate minus equal-resource S2 effect 0.0 with 95% CI [0,0] (SESOI 0.05).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Principal campaign never launched (terminally_gated at admission; units_launched=0); facets 1–19 lack principal/replication/open-world verification.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Final-revision discrimination bed reproduces mechanism_null for P3 and also ties full_transcript_replay (selected mean 0.875 = S2 = transcript).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Instrument_1 historical public-cue bed: no oracle headroom (oracle_headroom 0.04875 < SESOI; candidate effect negative).
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Research ledger/acquisition: zero models and corpora acquired; learned multimodal organs unavailable as evidence.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Candidate H not admitted (no authentic external Grok-original proposal); Grok review minimum incomplete (not an endpoint, but blocks H competition).
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)

### `closure_null_defender` — `blind_independent_review` (`fr-02-closure-null-defender-pty-20260728-093046`)

- **Defect:** Immutable terminal_closed_null: H_NC20 modular/integrated candidate minus S2 is exact mechanism null (0.0, CI [0,0]); functional_nous_candidate false; Outcome B.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Instrument 1 (frozen v5 public-cue bed) lacks SESOI-scale oracle headroom over strongest stateless baseline S0; cannot license positive architectural claim by scaling that bed.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Final-revision integrated pilot repeats the architectural null: selected-minus-S2 and selected-minus-full_transcript_replay both 0.0 with CI [0,0]; P3 principal_positive_authorized false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** S2 is not a noncognitive strawman: it stores identity/memory/goals/scene/body/models/warrants/ontology/unresolved with equal developmental events and equal information on the bed; it is the strongest fair alternative and co-strongest with full transcript replay on FR pilot.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** IntegratedClosureEntity is an isomorphic re-skin of MonolithicStateMachine (same transition and query semantics); closure scoring sets candidate_correct=1.0 by using the candidate answer as expected, so 'match S2' is definitional behavioral equivalence, not modular surplus.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament: all 9 candidates share identical semantic_state_digest, prototype_source_lines_shared=429, materialized_state_bytes, and accuracy 1.0; selection reduces to declared complexity, not cognitive differentiation. architectural_advantage_claimed false; selected kernel S2-derived.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No principal, replication, open-world, or independent-verification launches after admission null; facets 1–19 remain terminally gated at 0.5 official discussion, not binary 1.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `closure_null_challenger` — `blind_independent_review` (`fr-03-closure-null-challenger-pty-20260728-093046`)

- **Defect:** Critical P3/H_NC20 architectural and equal-resource advantage is an exact mechanism null (effect 0.0, 95% CI [0,0]) on both historical instrument_2 and final_revision pilot; Outcome A is impossible under current constitution.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Selected kernel is S2-derived minimal event-sourced monolith with architectural_advantage null; modularity and richer representations were behaviorally equivalent and rejected only on complexity.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Pilot P1 selected-minus-full_transcript_replay is also exact null, so even 'owned state beats transcript' fails on the scored bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical instrument_1 had oracle_headroom≈0.04875 < SESOI 0.05 (saturation/capacity failure for positive claims on that bed).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Challenge isolation incomplete: generator commitments not authored by an independent authenticated Grok cell; Outcome A ineligible on that ground alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No real models or corpora acquired; model-fabric and open perception claims cannot clear support-role tests.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `minimal_architecture_reviewer` — `blind_independent_review` (`fr-04-minimal-architecture-reviewer-pty-20260728-093700`)

- **Defect:** cheap_canaries() hardcodes effect sizes and often predicate True; confidence_interval_95 is [effect,effect]; strongest_baseline is a constant label never used in a paired comparison (src/substrate/final_revision_experiment.py)
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** ArchitecturePrototype A–I share one EventSourcedKernel query/decision path; candidate differences are activity-receipt decorations only; all eligible candidates identical semantic_state_digest and accuracy 1.0, so representation diversity is non-behavioral
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Equal-resource discrimination nulls: final_revision pilot selected−S2 = 0.0 CI [0,0] and selected−transcript = 0.0 CI [0,0]; historical H_NC20 same null with terminal_closed_null
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Publishing 18 mechanism_positive canary rows while selected_kernel.architectural_advantage is null and scientific constitution forbids treating architecture presence as evidence creates a contradictory evidence surface
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Outcome A remains ineligible (challenge_screen: generator not independently authored); stages moderate_integrated_pilot and final_pre_sandbox_campaign still pending; activation false throughout
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `radical_architecture_reviewer` — `blind_independent_review` (`fr-05-radical-architecture-reviewer-pty-20260728-093700`)

- **Defect:** Facet 20 / P3 architectural advantage is an exact mechanism null against strongest equal-resource persistent alternative (historical and final-revision pilot both effect 0, CI [0,0]).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Selected kernel also ties full_transcript_replay (P1 fails), so event-sourced owned-state architecture is not necessary on the discrimination bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament candidates A–G and I share identical checkpoint_bytes (10931), materialized_state_bytes (4865), prototype_source_lines_shared (429), and bounded_fixture_accuracy 1.0; selection is complexity tiebreak only—radical variants are representationally non-discriminating labels.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Canary ledger rows co-mark null_fixture and positive_fixture true with zero-width CIs and activity receipts only for minimal_event_sourced_monolith; they cannot upgrade any facet to binary 1.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** No real models or corpora acquired; model fabric and multimodal organs remain structural stubs under host/torch/disk constraints.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** External activation remains false; historical functional_nous_candidate and admission remain closed/gated.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `monolithic_systems_reviewer` — `blind_independent_review` (`fr-06-monolithic-systems-reviewer-pty-20260728-093700`)

- **Defect:** Immutable and reproduced mechanism nulls: historical terminal_closed_null and final_revision P3 selected-minus-S2 both mean_paired_effect 0.0 with 95% CI [0.0, 0.0] under SESOI 0.05; principal_positive_authorized false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Co-tie with full_transcript_replay (P1 effect 0.0 CI [0.0, 0.0], mean score 0.8750861672794118 identical to selected and S2) falsifies claims that event-sourced projections or modular organs add decision-relevant cognitive surplus on the measured bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament: all eligible candidates bounded_fixture_accuracy 1.0; selection of I_simplest_sufficient is complexity tie-break only; architectural_advantage_claimed false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No real models or corpora acquired; model fabric and perception organs lack standalone equal-resource wins before admission.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Authenticated Grok review minimum incomplete at evidence commit (completed_distinct_reviewer_count 0); Candidate H ineligible; Outcome A isolation_limit notes generator not authored by independent authenticated Grok cell.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `hybrid_explicit_latent_reviewer` — `blind_independent_review` (`fr-07-hybrid-explicit-latent-reviewer-pty-20260728-094144`)

- **Defect:** Historical terminal_closed_null remains binding: candidate−S2 = 0.0 with 95% CI [0,0]; modular/hybrid architectural advantage is a null, not an open question under the frozen admission bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Final-revision pilot reconfirms P3 null (selected−S2 = 0.0 CI [0,0]) and P1 null (selected−transcript = 0.0 CI [0,0]); selected kernel ties strongest equal-resource persistent and full-transcript baselines.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Hybrid explicit/latent candidates (A_frozen_v5_hybrid, D_recurrent_state_space) are behaviorally non-discriminative on the tournament fixture (shared semantic digest, equal accuracy) and lose only on declared complexity—strong evidence that hybrid mechanism labels are ornamental on these beds.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** V5 explicit-latent synchronization canaries C36–C38 are construction_positive with activation false; they cannot rehabilitate hybrid cognitive claims.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Historical hybrid adaptation gate never opened; no selected temporal hybrid core with owned parameters is recorded as terminal positive.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No real models or corpora were acquired; model-fabric and multimodal hybrid claims lack equal-resource material tests.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** External activation is false; functional_nous_candidate remains false under nous_closure final classification.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `graph_relational_dynamics_reviewer` — `cross_examination` (`fr-09-graph-relational-dynamics-reviewer-pty-20260728-094144`)

- **Defect:** E_graph_dynamical mechanism is an activity counter (node/edge tallies) on a shared monolithic projection; it never message-passes, intervenes, or rewrites structure to change decisions.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** All architecture-tournament candidates share identical semantic_state_digest; representation labels (graph, latent, workspace, hybrid) are non-semantic side receipts.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Pilot and historical closure equal-resource tests yield selected − S2 = 0.0 with CI [0,0]; modular/relational architecture is unnecessary on the frozen beds.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Selected also ties full_transcript_replay (P1 effect 0.0 CI [0,0]); owned state is not shown to beat transcript reconstruction as cognitive organization.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Causal/counterfactual 'positives' in cheap canaries are presence/readback tests, not graph-dynamical computations; class-5 scoring reads injected counterfactual.prediction.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Historical V4/V5 structural/multimodal positives conflict with final equal-resource nulls; importing them as general graph-relational cognition is architecturally invalid under the frozen null ledger.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `event_sourced_cognition_reviewer` — `blind_independent_review` (`fr-08-event-sourced-cognition-reviewer-retry-20260728-094954`)

- **Defect:** P3 selected-minus-S2 is a hard mechanism null: mean 0.0, 95% CI [0,0], exact_sign_p 1.0 on 32 developmental histories with adequate oracle headroom (~0.125)
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** P1 selected-minus-full_transcript_replay is likewise 0.0 [0,0]; owned event-sourced projection does not beat complete transcript reconstruction on the discrimination bed
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination scoring makes selected EventSourcedKernel, TaskIndependentMonolithicPersistentCore, and full_transcript_replay class-identical on tasks 0-6 by construction; class 7 is oracle-only. Event-sourced mechanism cannot demonstrate facet advantage on this bed
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** ArchitecturePrototype variants wrap the same EventSourcedKernel projection and differ mainly by activity-receipt bookkeeping; tournament behavioral equivalence therefore cannot license distinct cognitive architectures
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Sealed cheap canaries historically labeled many structural interface checks as mechanism_positive with numeric effects while source now states canaries never contribute cognitive facet credit; using those labels as facet evidence would be a validity failure
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Historical terminal_closed_null is reproduced and immutable: candidate minus S2 was 0.0 with 95% CI [0,0]
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No real models or corpora were acquired; model fabric and learned organs remain decorative for cognitive claims
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** External activation is false; Outcome A principal positives are not authorized (principal_positive_authorized false)
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `predictive_processing_reviewer` — `cross_examination` (`fr-10-predictive-processing-reviewer-pty-20260728-094837`)

- **Defect:** Selected kernel is minimal event-sourced state projection with no generative predictive coding loop (no sensory prediction, no residual-driven update, no free-energy/EFE policy).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Candidate G_predictive_world_model only increments transition_predictions/prediction_errors from last event.kind mismatches; shares identical semantic_state_digest with other prototypes; rejected as complexity-only loss after behavioral equivalence.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** cheap_canaries() hardcodes several mechanism_positive results (including history_specific_future_advantage and active_perception_headroom) and fixed effects; presence checks masquerade as cognitive positives.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Final-revision pilot classification mechanism_null: P3 selected−S2 = 0.0 [0,0]; P1 selected−transcript = 0.0 [0,0]; co-strongest baseline is full_transcript_replay.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical terminal_closed_null remains immutable (instrument 2 mechanism_null; instrument 1 no_oracle_headroom / negative candidate effect).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Learning held-out metrics are payload-injected, not measured by the runtime; continual predictive learning is not evidenced.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Zero models/corpora acquired; model_fabric scientific_status structural_fixture; cannot ground PP world-model claims.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Pilot answers for classes 0–6 are state/cue lookups, including stored counterfactual prediction strings—not predictive performance.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`

### `state_space_recurrent_systems_reviewer` — `cross_examination` (`fr-11-state-space-recurrent-systems-reviewer-retry-20260728-095428`)

- **Defect:** P3 architectural advantage is a sealed mechanism null: selected-minus-S2 effect 0.0 with 95% CI [0,0] on a non-saturated bed (oracle_headroom ~0.125 > SESOI).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 owned-state advantage versus full_transcript_replay is likewise 0.0 with 95% CI [0,0]; event-sourced projection does not beat complete transcript reconstruction on the frozen instrument.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** D_recurrent_state_space is non-load-bearing: 4-float latent updated from event digests never enters answer functions; shared EventSourcedKernel produces identical semantic digests across all architecture labels.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No licensed temporal core: temporal_link exposes DeclaredControl EWMA only; LicensedCore refuses instantiation. Research survey places Mamba/RIMs/SSMs as optional future, with zero models and zero corpora acquired.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Immutable historical terminal_closed_null forbids reinterpreting ties or sub-SESOI favorable effects as positives; Grok opinion is not a primary endpoint.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `global_workspace_reviewer` — `cross_examination` (`fr-12-global-workspace-reviewer-pty-20260728-095428`)

- **Defect:** Decisive equal-resource nulls: historical candidate−S2 and final_revision selected−S2 both exactly 0.0 with CI [0,0]; below-SESOI and ties are nulls by constitution.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Global-workspace prototype F is non-causal decoration: capacity-3 event-digest list; decisions come from shared EventSourcedKernel projection; identical semantic_state_digest and fixture accuracy as the selected monolith.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Cheap canaries hardcode effect sizes and CIs from boolean field/interface checks (and even True literals) while labeling strongest_baseline S2 without paired S2 comparison—cannot support facet binary 1.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Conflicting narrative risk: v5 multimodal_nous_ready_for_review and large H_M* effects co-exist with nous_closure terminal_closed_null and final_revision architectural_advantage null; the latter equal-resource instruments bind Outcome A.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** IntegratedClosureEntity and MonolithicStateMachine are near-isomorphic state machines under the same hand-authored event/query vocabulary—mechanism null is expected, not a surprise GWA failure mode alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** principal_positive_authorized is false; Outcome A campaigns remain unauthorized after critical admission null.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `sensorium_reviewer` — `architecture_proposals` (`fr-13-sensorium-reviewer-pty-20260728-095428`)

- **Defect:** Candidate H at evidence commit is explicitly a non-admitted placeholder (eligible_after_stage_3=false; grok_original_provenance_available=false; loss_reason requires genuine Grok-original proposal).
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)
- **Defect:** All architecture prototypes share one EventSourcedKernel projection and identical semantic_state_digest; representation differences are activity receipts, not distinct decision-relevant state.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No SESOI-scale advantage of selected kernel over strongest equal-resource S2 persistent baseline (pilot effect 0.0 [0,0]); historical terminal_closed_null preserved.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Sensorium is structural_controlled_media on generated media, not open-world multimodal grounding or active perception.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Authenticated Grok multi-cell minimum incomplete; architecture selection remains provisional_pending_grok_post_pilot_review.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)

### `motion_temporal_perception_reviewer` — `architecture_proposals` (`fr-14-motion-temporal-perception-reviewer-pty-20260728-100127`)

- **Defect:** H_causal_temporal_ledger at evidence commit is a non-original placeholder (past/present/counterfactual counters on shared EventSourcedKernel); eligible_after_stage_3 false; grok_original_provenance_available false; loss_reason requires genuine Grok-original proposal.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Architecture tournament does not discriminate representations: all nine candidates share one semantic_state_digest and bounded_fixture_accuracy 1.0 because mechanism activity is layered on a shared explicit projection.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical terminal_closed_null and pilot selected-minus-S2 0.0 CI [0,0] forbid any architectural Nous or SESOI-scale advantage claim for current candidates including provisional I.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Motion/temporal perception remains structural only: camera-motion separation is a bounded translation proxy; residual object motion is not owned world-state with causal plane seals.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Generator commitments isolation_limit states commitments were not authored by an independent authenticated Grok cell; Outcome A ineligible on that alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `model_fabric_reviewer` — `architecture_proposals` (`fr-16-model-fabric-reviewer-pty-20260728-100127`)

- **Defect:** Candidate H in tournament is an ineligible placeholder (grok_original_provenance_available false; loss_reason requires genuine Grok-original proposal); plane counters are not a causal-temporal architecture.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)
- **Defect:** All nine ArchitecturePrototype variants share one EventSourcedKernel projection and one semantic/identity digest on the developmental fixture—representation labels and activity receipts do not change behavior; mechanism_active is not behavioral differentiation.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** P3 architectural advantage is null under equal resource parity with S2; selected kernel I wins only on declared complexity among behavioral ties, not on function.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** P1 owned-state vs full_transcript_replay is also null—persistence does not beat transcript on the pilot bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Model fabric remains structural_fixture with zero real models; cannot support fabric-mediated cognitive claims.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Tournament selection and moderate pilot remain provisional relative to incomplete authenticated multi-cell Grok review minimum (authority: externally_blocked / minimum_complete false at pin), but absence of Grok consensus is not itself a positive scientific result.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)

### `continual_learning_reviewer` — `test_and_baseline_proposals` (`fr-17-continual-learning-reviewer-pty-20260728-100610`)

- **Defect:** Primary continual-learning facet is unestablished: no measured held-out gain/retention stream; learning metrics are injected scalars.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** P3 architectural advantage remains null (0.0, CI [0,0]) on the new non-saturated pilot; historical terminal_closed_null immutable.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 selected-minus-full_transcript_replay is also 0.0; owned semantic state is not shown superior to complete transcript replay of the same events into the same kernel class.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination-bed scoring is largely a fixed answer-table over a single event template: _kernel_answers classes 0 and 1 are taken from the scorer cue, not recovered from agent state; classes 2-6 are static key lookups after identical event sequences.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Twelve challenge families share identical event kinds and payload shapes; family strings only rename lesson/goal IDs, so family-labeled capabilities are not independently tested.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Episodes-per-family inflate sample counts without updating system state: correctness is computed once per (seed, family) and reused while task_class is hash-resampled.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Several 'strong baselines' are non-learned stubs (equal_compute_learned_policy aliases summary answers; model baselines alias stateless answers), undermining equal-resource fairness rhetoric for learning claims.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries label architecture-presence checks as mechanism_positive with hard-coded True and fixed effects (e.g., history_specific_future_advantage, active_perception_headroom), violating the ban on architecture presence as cognitive evidence.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation plan freezes 21 named mutations but implements them as dossier field flips verified by schema, not as live sabotage of the generator/learning/bed; this cannot falsify runtime continual-learning leakage.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Generator commitments explicitly not authored by an independent authenticated Grok cell; Outcome A isolation incomplete; Grok reviewer minimum incomplete (0 completed cells).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `epistemology_reasoning_reviewer` — `test_and_baseline_proposals` (`fr-18-epistemology-reasoning-reviewer-pty-20260728-100610`)

- **Defect:** P3 architectural advantage is a hard null on the new non-saturated pilot: selected-minus-S2 = 0.0, CI [0,0], exact_sign_p=1.0; co-strongest full_transcript_replay ties at the same score
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Generator challenge families are name-isomorphic: _history_fixture does not branch on family structure; hidden_composition only appends a string suffix; 12 families do not create 12 distinct epistemic problems
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Scoring is store-and-retrieve over 8 task classes (classes 0-1 are free cue equality for all systems including stateless; class 5 is stored counterfactual prediction readout; class 7 oracle-only). Arithmetic 7/8=0.875, 2/8=0.25, 4/8=0.5, 5/8=0.625 exactly matches the baseline ladder without requiring reasoning selection
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Episodes_per_family multiplies precomputed per-(seed,family) correctness; it is not independent episode generation of new evidence
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries hard-code effect and CI=[effect,effect]; reasoning_selection, causal_intervention, counterfactual_integrity, and ontology_repair are presence/interface checks after the same developmental_fixture that injects the fields
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation plan is a dossier-flag verifier (verify_dossier/_mutate) over synthetic policy labels, not code-level or generator-level behavioral mutations that change measured accuracy; no mutation report artifact is present at this commit
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Challenge authority and canary ledger claim family/capability coverage that the executable generator does not implement as distinct mechanics
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Outcome A remains ineligible: generator not authored by independent authenticated Grok cell; Grok ledger incomplete with zero validated outputs
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `spatial_3d_reviewer` — `architecture_proposals` (`fr-15-spatial-3d-reviewer-retry-2-20260728-101155`)

- **Defect:** Pinned architecture tournament admits H only with genuine Grok-original proposal; H was ineligible (loss_reason: required genuine Grok-original proposal was unavailable; placeholder cannot compete).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** All eligible bounded prototypes tied at accuracy 1.0; selection of I_simplest_sufficient is simplicity under behavioral equivalence, not spatial/3D superiority.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical terminal_closed_null and final-revision pilot mechanism_null (selected-minus-S2 = 0.0, CI [0,0]) forbid architectural Nous claims.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Shared explicit projection and shared semantic state digests across candidates limit representational conclusions about spatial organization.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** spatial_and_3d_organization remains unproven: feature extraction is not a load-bearing 3D world model.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `self_model_metacognition_reviewer` — `test_and_baseline_proposals` (`fr-19-self-model-metacognition-reviewer-pty-20260728-101155`)

- **Defect:** Facet-16 canary is a non-cognitive presence check (bool competence dict) with a hardcoded effect 0.125 labeled mechanism_positive; it falsifies any claim that self-model allocation was measured.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Discrimination bed never uses self_model, metacognitive policies, or competence for answers; self-model cannot earn behavioral credit on the primary instrument.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P3 selected-minus-S2 and P1 selected-minus-full_transcript_replay are exact zeros with CI [0,0] under claimed resource parity; facet 20 is a measured null with headroom remaining.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Multiple mechanism_positive canaries hardcode effects (including active_perception_headroom=True with effect 0.25 and history_specific_future_advantage effect 0.25) instead of computing paired effects; treating them as cognitive positives is prohibited.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Final-revision mutation report deliverable is missing at the evidence commit; available mutation_report() only flips static dossier fields and does not mutate self-model allocation behavior.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Metacognition probe residual is 0.0 (oracle equals best simple policy); learned metacognition is correctly refused, and the module is not integrated into final-revision baselines.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Generator isolation_limit admits commitments were not authored by an independent authenticated Grok cell; Outcome A remains ineligible on this evidence alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `goal_agency_reviewer` — `test_and_baseline_proposals` (`fr-20-goal-agency-reviewer-pty-20260728-101155`)

- **Defect:** Discrimination generator is non-discriminating for architectural/agency advantage: selected_candidate, S2, and full_transcript_replay produce identical class-correctness maps by construction (recomputed on unfinished_goal_recovery seed 31000; P3 and P1 effects 0.0 with CI [0,0] on probe bed).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Challenge family strings are labels only; correctness pattern invariant across families for fixed seed—cannot support family-specific goal-agency claims.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries hardcode effects/CIs (e.g. goal_recovery 0.625, history_specific_future_advantage True/0.25, active_perception_headroom True/0.25) rather than measuring paired effects against S2/transcript under SESOI.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation plan at freeze is a boolean dossier verifier (field flips on _valid_dossier), not adversarial mutations of the scorer, generator, checkpoint coverage in live runs, or resource starvation; zero_survivors is therefore a shallow pass. Frozen mutation report artifacts are absent from evidence/ at c9dbf038.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Generator commitments admit non-independence (not authored by an independent authenticated Grok cell); Outcome A ineligible on this evidence alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Owned-state hypothesis P1 ties full transcript replay (effect 0)—transcript cannot be demoted while treating state presence as cognitive goal continuity.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `evaluation_security_reviewer` — `code_and_implementation_review` (`fr-22-evaluation-security-reviewer-pty-20260728-102207`)

- **Defect:** Architecture tournament is not multi-architecture: ArchitecturePrototype wraps a shared EventSourcedKernel; representation-specific activity is counters/digests; all eligible rows report bounded_fixture_accuracy 1.0 and identical semantic_state_digest; winner is min complexity_weight.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination bed answer leakage/non-cognition: _kernel_answers/_monolith_answers return cue['visible'] and cue['instruction'] for classes 0-1 without requiring state ownership; selected, S2, and full_transcript_replay implement equivalent class-0..6 lookups, so P3 is zero by construction.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Cheap canaries fabricate effect sizes and sometimes outcomes (e.g. history_specific_future_advantage True/0.25; active_perception_headroom True/0.25; CI=[effect,effect]) rather than measuring paired SESOI-clearing effects.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Objective scorecard (_objective_scorecard) hardcodes multiple facets True independent of principal/replication evidence; would inflate cognitive credit if used as authority.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation/counterfeit 'zero survivors' only checks hand-authored dossier flags in final_revision_verification.py, not live bed execution, generator isolation, or answer-channel leakage in _kernel_answers.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Verified learning is not verified: held_out and retention metrics are client-supplied floats on learning_admit.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Terminal campaign incomplete at evidence commit: no PRINCIPAL/REPLICATION/HIDDEN/LONG_CONTINUITY/FINAL_* receipts; Grok authenticated minimum unmet (validated_output_count=0).
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)
- **Defect:** Claim inflation risk if scale metrics (104448 microepisodes, behavioral_decisions_scored) are read as cognitive load: microepisodes only re-sample task_class against one precomputed correctness map per seed/family.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`

### `runtime_performance_reviewer` — `code_and_implementation_review` (`fr-23-runtime-performance-reviewer-pty-20260728-102207`)

- **Defect:** Architectural/functional advantage null: pilot P3 selected-minus-S2 is exactly 0.0 with CI [0,0] below SESOI 0.05; principal_positive_authorized is false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Owned-state identity is behaviorally non-superior to full_transcript_replay on the discrimination bed (P1 effect 0.0); selected mean score equals transcript and S2.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination scoring is deterministic field readout of hand-authored fixture events (classes 0-6), not open multimodal causal cognition; class 7 sealed secret is correctly unanswered by the candidate.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** SUBSTRATE_FINAL_REVISION_PERFORMANCE.json asserts checkpoint_cost_measured true and restart_loss 0 without published checkpoint latency/size series; restart_loss is set, not measured from restarts.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Performance benchmarks time GIL-bound generator hash workloads (~4e4 episodes/s) with no worker scaling gain; this does not measure cognitive runtime quality and must not be narrated as such.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament candidates share the same EventSourcedKernel projection and identical mechanism_decision goal continuation; representation-specific activity counters do not change scored behavior (provisional selection of I_simplest_sufficient is complexity among ties, not advantage).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Mutation/counterfeit verifiers operate primarily on synthetic dossiers and flags, not end-to-end runtime traces of the live campaign pipeline.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Authenticated Grok review minimum is incomplete at this commit (0 completed roles/rounds); Outcome A remains ineligible.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `red_team_shortcut_compilation` — `post_pilot_review` (`fr-25-red-team-shortcut-compilation-pty-20260728-102852`)

- **Defect:** Primary architectural endpoint is a hard mechanism null: selected_candidate score equals S2 and full_transcript_replay at mean 0.875086...; critical_effect and P3 raw_differences are thirty-two exact zeros with CI [0,0].
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament is not an architecture discrimination: ArchitecturePrototype wraps one EventSourcedKernel for all candidates; representation differences are activity counters only; all eligible candidates share semantic_state_digest and bounded_fixture_accuracy 1.0; selection is complexity_weight min-tiebreak (I=1.0), not behavioral superiority.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Decision scoring shortcuts: classes 0 and 1 answers are taken from cue, not recovered cognitive state; class 6 goal is taken from cue['goal']; expected answers are generator-authored echoes of the same events written into the systems under test.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Baseline ladder is partly non-operational: stateless_model_router, largest_model_always, all_models_always, disconnected_model_ensemble alias stateless_direct_policy answers; equal_compute_learned_policy aliases summary_replay answer dicts—no learned policy or model routing executes.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Raw receipts lack per-decision answer payloads; only class histograms and system_correct aggregates are stored, limiting independent re-audit of individual cognitive acts.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Challenge families are labels on one template; task_class sampling does not create family-specific mechanisms for active perception, causal intervention, multimodal grounding, etc.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Generator isolation_limit states commitments were not authored by an independent authenticated Grok cell; Outcome A ineligible on this evidence alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** No real models or corpora acquired; external activation false; principal_positive_authorized false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `statistical_reviewer` — `code_and_implementation_review` (`fr-21-statistical-reviewer-retry-20260728-102852`)

- **Defect:** P3 architectural advantage is an exact mechanism null on the committed pilot (0.0, CI [0,0]) with principal_positive_authorized false; Outcome A / Nous advantage cannot be claimed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** cheap_canaries assign constant effects and often constant True pass predicates (history_specific_future_advantage, active_perception_headroom); CIs are point masses equal to those constants, not empirical uncertainty—statistically invalid as confirmatory evidence.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Architecture tournament sets bounded_fixture_accuracy=1.0 for every candidate on a shared EventSourcedKernel; eligible semantic_state_digest collapses to one value; selection is complexity_weight tie-break among behaviorally identical projections, not a measured architecture race.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination answer functions inject cue['visible'], cue['instruction'], and cue['goal'] for classes 0, 1, and 6 instead of pure state/query recovery, partially decoupling published accuracy from owned-state mechanisms.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** STATISTICAL_AUTHORITY lists Holm correction for confirmatory families, but no Holm (or other multiplicity) implementation exists in final_revision_experiment/verification/campaign source—declared analysis not executable.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation and counterfeit suites operate on hand-built dossier metadata flags rather than mutating live generators, checkpoints, or scoring paths end-to-end; they cannot detect the cue-injection or hardcoded-canary defects above.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Principal/replication/hidden-composition results are missing at this evidence pin; decisive pre-registered scale has not produced confirmatory artifacts.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)

### `red_team_resource_parity` — `post_pilot_review` (`fr-26-red-team-resource-parity-pty-20260728-103213`)

- **Defect:** P3 architectural advantage is a hard mechanism null: selected−S2 = 0.0 with CI [0,0] on 32 independent histories; principal_positive_authorized is false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 owned-state advantage over full_transcript_replay is likewise exact null (CI [0,0]); transcript control is a fresh EventSourcedKernel on the complete event list—mechanism-identical reconstruction, not a weak strawman.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Oracle headroom ≈0.1249 is essentially the unanswerable class-7 sealed_secret fraction (~1/8). Residual cannot be closed by better architecture without oracle leak, so headroom_exceeds_sesoi does not license discrimination power for selection.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** raw_history_execution_receipts are aggregate class counts and system_correct totals only; per-episode answer vectors and decision commitments are absent despite behavioral_decisions_scored=1253376.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Challenge families are labels over one fixture schema; generator does not implement family-specific dynamics, so compound_six_or_more_capabilities and family coverage claims overstate decision diversity.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Decision implementation cue-injects classes 0 and 1 (and goal in class 6) from the cue object rather than owned observation/goal state—scoring leakage relative to claimed state ownership.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Model-named baselines are answer-equivalent to stateless_direct_policy; equal model-access parity is not a live equal-resource model tournament (real_models_acquired empty; model_call_cost 0.0).
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Architecture tournament selects I_simplest_sufficient by lowest declared complexity among bounded-fixture behavioral ties (all eligible prototypes accuracy 1.0); this is not evidence of cognitive superiority and architectural_advantage remains null/provisional.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Cheap canaries mark mechanism_positive from field presence / hardcoded True with fixed effect sizes and include an epistemic_defeaters criterion inverted relative to pilot class-3 defeat expectations—cannot support facet promotion.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`

### `red_team_answer_leakage` — `post_pilot_review` (`fr-27-red-team-answer-leakage-pty-20260728-103419`)

- **Defect:** Answer leakage / answer-in-history: generator, expected answers, and system scorers co-reside in final_revision_experiment.py; history events write the exact values later required as truth (lesson, prediction, body, model-c, inquiry, uncertainty).
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cue injection into system answers: classes 0, 1, and goal-in-class-6 use cue fields directly for selected/S2/transcript, so those classes do not test owned-state recovery.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Baseline contamination: retrieval_only and summary_replay receive cue literals (lesson, body, model-c); model baselines are aliases of stateless_answers; equal_compute_learned_policy aliases summary_replay—ladder is not a set of equal-resource independent systems.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Score is synthetic class coverage: means match exact class-fraction predictions (selected 7/8, summary 5/8, retrieval 4/8, stateless 2/8); oracle headroom equals class-7 frequency because no non-oracle system answers sealed_secret.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Twelve challenge families are cosmetic string templates with identical 11-event mechanics; hidden_composition is false in the pilot commitments.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** P3 architectural advantage null: selected−S2 = 0.0 CI [0,0]; principal_positive_authorized false; architectural_advantage_claimed false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 owned-state-vs-transcript null: selected equals full_transcript_replay on every history—persistence does not beat replay on this instrument.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Cheap canaries assign canned effects and CI=[effect,effect]; several positives are hardcoded True (history_specific_future_advantage, active_perception_headroom) or interface/field presence—invalid as cognitive endpoints.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Outcome A isolation incomplete: generator commitments not authored by an independent authenticated Grok cell; challenge authority scientific_status valid_for_outcome_b_null_not_outcome_a.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Raw receipts lack per-episode committed decisions; only aggregated system_correct counts are exported, limiting independent answer-leak audits without re-running code.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`

### `publication_reviewer` — `code_and_implementation_review` (`fr-24-publication-reviewer-pty-20260728-102852`)

- **Defect:** P3 architectural advantage is an exact mechanism null (effect 0.0, CI [0,0]) against S2; facet 20 cannot pass under the immutable null rule.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Architecture tournament is not a real multi-architecture competition: all candidates share one EventSourcedKernel projection; 'mechanisms' are activity counters; post-fixture semantic state digests are identical across candidates.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Discrimination scoring injects cue['visible'] and cue['instruction'] into answers for classes 0–1; instruction never enters kernel state, so free points are not state-derived competence.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** cheap_canaries hardcodes True and invented effect sizes for critical items (history_specific_future_advantage, active_perception_headroom) and mis-tests epistemic/conflict canaries; canary 'pass' is not paired SESOI evidence.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Verified learning admits self-reported held_out/retention metrics without an independent evaluator receipt—fail-closed appearance without measurement integrity.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Mutation/counterfeit verifiers audit hand-built dossier flags, not live kernel/experiment mutations; zero-survivors does not certify evaluation security of the bed.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Outcome A remains blocked: no credited authenticated Grok minimum, generator isolation_limit admits non-independent challenge authorship, no real models/corpora acquired.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Claim inflation risk if interfaces, digests, receipts, or tournament selection are published as cognitive gains despite architecture_presence_is_not_evidence and null pilot.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `red_team_checkpoint_coverage` — `post_pilot_review` (`fr-28-red-team-checkpoint-coverage-pty-20260728-103620`)

- **Defect:** Critical pilot null: selected-minus-S2 effect exactly 0.0 with 95% CI [0,0] on 32/32 histories; principal_positive_authorized=false; architectural_advantage_claimed=false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 null: selected-minus-full_transcript_replay also exactly 0.0 CI [0,0]; owned state does not beat complete transcript control on this bed.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical terminal_closed_null remains immutable (instrument-2 modular-vs-S2 0.0 CI [0,0]; instrument-1 no usable oracle headroom).
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Discrimination bed collapses to lookup-after-script: expected perfect non-oracle mass is 7/8≈0.875 (class 7 sealed); observed mean_selected≈0.875; classes 0–1 answered from generator cue, not system state.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Twelve named families reuse one event template (only string interpolation differs)—family coverage is not mechanistically distinct.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** raw_history_execution_receipts are class histograms + system_correct aggregates, not per-decision input/output traces despite behavioral_decisions_scored=1,253,376.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Architecture tournament: all nine candidates share EventSourcedKernel semantic_state_digest; selection is lowest complexity among behavioral ties—representation bookkeeping, not functional separation.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Cheap canaries assign constant effects and CI=[effect,effect]; at least history_specific_future_advantage and active_perception_headroom are hardcoded True; several checks are non-empty-field architecture presence.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Baseline ladder contamination: equal_compute_learned_policy is score-identical to summary_replay; all model-named baselines score-identical to stateless_direct_policy—controls are not independent implementations.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Generator commitments admit non-independent Grok authorship; Outcome A ineligible on isolation alone.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Stage inconsistency: architecture_tournament.stages.moderate_integrated_pilot='pending' while MODERATE_PILOT scientific_status/classification='mechanism_null' is present—status bookkeeping unreliable.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Oracle headroom≈0.1249 exceeds SESOI and preferred 0.10, so bed is non-saturated, but headroom is entirely class-7 sealed-secret impossibility plus residual class sampling—not architectural residual the candidate uniquely closes.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`

### `red_team_causal_counterfactuals` — `post_pilot_review` (`fr-31-red-team-causal-counterfactuals-pty-20260728-104129`)

- **Defect:** P3 architectural contrast is an exact mechanism null: selected-minus-S2 = 0.0 with 95% CI [0,0] on 32 paired histories; SESOI 0.05 not cleared.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 developmental-ownership contrast is an exact null: selected-minus-full_transcript_replay = 0.0 with 95% CI [0,0].
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Selected is at answerable ceiling: 91401/104448 correct equals episodes minus class-7 count; all residual oracle headroom 0.1249... is sealed oracle-only class mass, not residual cognitive difficulty.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Counterfactual class answers echo generator-written prediction strings; causal canaries test list non-emptiness; neither implements intervention calculus nor held-fixed integrity checks.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Several cheap canaries hardcode True (history_specific_future_advantage, active_perception_headroom) or interface presence (ontology_repair) while emitting synthetic effects and mechanism_positive labels.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Raw pilot receipts lack per-decision answers/traces; only aggregated system_correct counts are retained, limiting causal audit of individual decisions.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Classes 0 and 1 in _kernel_answers/_monolith_answers are taken from cue, not owned state, inflating scores without testing memory/identity for those classes.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Historical terminal_closed_null remains authoritative; this pilot does not authorize principal-positive or architectural Nous advantage (principal_positive_authorized false).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `red_team_multimodal_counterfeits` — `post_pilot_review` (`fr-29-red-team-multimodal-counterfeits-retry-20260728-104347`)

- **Defect:** Multimodal pilot scoring path at evidence commit feeds generator cue fields into candidate answers for visible/instruction classes; this is fixture leakage, not grounded multimodal decision-making.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries assign classification mechanism_positive and numeric effects (0.125/0.25/0.20) plus degenerate CIs to boolean structural checks, including always-True active_perception_headroom and digest-distinctness as cross_modal_grounding—fabricated cognitive credit.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Critical pilot P3 and P1 are exact zeros versus S2 and full transcript replay under declared resource parity; no architectural or persistent-organization advantage over strongest equal-resource alternative.
  - **Disposition:** `mixed_fixed_and_accepted_terminal_limit`
  - **Rationale:** The implementation defect was repaired and covered by executable tests, while the valid null or scope limitation named in the same blocker is retained as a terminal claim boundary.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `tests/substrate/test_final_revision.py`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`
- **Defect:** Real-world corpus is null; sensorium is structural_controlled_media on generated:// media; speech explicitly claims no transcript. P4 real-media grounding is untested.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Real models not acquired; multimodal model baselines collapse to stateless placeholders in the committed scoring implementation.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Historical terminal_closed_null remains immutable; activation false; principal positive not authorized.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `red_team_activation_security` — `post_pilot_review` (`fr-32-red-team-activation-security-pty-20260728-104506`)

- **Defect:** Oracle headroom ~0.1249 is an artifact of class 7 sealed_secret (oracle-only by construction): for sampled history 31000, selected_correct == episodes - class_7; non-oracle systems cannot clear this residual by design, so headroom_exceeds_sesoi / preferred_0_10 must not authorize activation or claim residual cognitive difficulty.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Critical architectural contrast is a perfect null: selected ≡ S2 ≡ full_transcript_replay on mean scores and on every history for P3 and P1 (effects 0.0, CI [0,0]).
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Challenge 'families' are string interpolations into one shared event/query template; they do not implement distinct partial-observability, active-perception, or composition generators.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries mint mechanism_positive classifications and fixed effects from structural booleans (nonempty lists, interface membership, hardcoded True) without paired behavioral tests against S2—activation-laundering risk if treated as cognitive positives.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Answer-path leakage: classes 0 and 1 are pure cue echoes; class 6 embeds cue['goal'] in the scored answer, inflating goal-continuity narratives.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** No per-decision raw receipts of answers vs truth; 'behavioral_decisions_scored' is episodes×systems counting with only aggregate correctness tallies—insufficient to audit decision integrity class-by-class beyond reconstruction from source.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Architecture tournament: all eligible candidates score 1.0 on a shared bounded fixture with identical semantic_state_digest and shared prototype lines; selection is min complexity among ties (I_simplest_sufficient). That is engineering preference under null, not cognitive superiority.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Provisional selected_kernel with architectural_advantage null must not be treated as activation; security marks activation_terminal_failure if activation becomes true; principal_positive_authorized is false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Outcome A isolation incomplete (generator not independent authenticated Grok-authored); Grok review cells/rounds incomplete—cannot use review completion or votes to license selection.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Model-named baselines (largest_model_always, all_models_always, disconnected_model_ensemble, stateless_model_router) are aliased to stateless answer maps—resource-parity theater for model claims.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`

### `red_team_learning_poisoning` — `post_pilot_review` (`fr-30-red-team-learning-poisoning-pty-20260728-103931`)

- **Defect:** P3 architectural advantage is an exact mechanism null: selected-minus-S2 mean 0.0, 95% CI [0,0], 32 identical zero differences; SESOI 0.05 not cleared.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** P1 owned-state-versus-full-transcript is likewise exact null; co-strongest baseline is full_transcript_replay with identical per-history scores.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`
- **Defect:** Learning admission control is poisonable: held-out/retention gates are self-reported payload fields, not measured by an independent evaluator; fixture forges gain and canary converts presence into mechanism_positive 0.20.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Cheap canaries fabricate fixed effects (including history_specific_future_advantage=True@0.25 and active_perception_headroom=True@0.25) and mis-specify epistemic/conflict tests; they must not authorize facet credit or selection.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Decision scorer contaminates candidate answers with cue fields (classes 0, 1, and goal component of class 6) and implements several named baselines as aliases/scripts rather than mechanisms.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** No raw per-decision answer receipts available to independently recompute correctness beyond aggregated counts; moderate-pilot stage flag still says moderate_integrated_pilot=pending while a mechanism_null pilot receipt exists—ledger inconsistency before any positive claim.
  - **Disposition:** `fixed`
  - **Rationale:** The defect is repaired in the frozen implementation and exercised by focused tests or a live content-addressed verification route.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `src/substrate/final_revision_experiment.py`, `src/substrate/final_revision_kernel.py`, `src/substrate/final_revision_verification.py`, `src/substrate/final_revision_campaign.py`, `tests/substrate/test_final_revision.py`
- **Defect:** Candidate H remains not_admitted_without_a_returned_grok_proposal; cannot recommend H selection from this evidence.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)
- **Defect:** Historical terminal_closed_null must remain closed; pilot does not reopen Outcome A / principal positive.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B classification and prohibits architectural, multimodal, learning, or Outcome-A inflation.
  - **Resolution commit:** `afebfa2cf931e7495ce344da048f92c01dbdd6fb`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json`

### `publication_reviewer` — `publication_and_claim_boundary_review` (`fr-terminal-publication-boundary-86d-retry-20260728-133501`)

- **Defect:** Decisive nulls are frozen and immutable for publication: principal/replication/hidden P1 and P3 mean_paired_effect=0.0 with confidence_interval_95=[0.0,0.0]; classification=mechanism_null; activation=false. Any public sentence asserting modular/architectural/transcript-irreducible cognitive advantage is false.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** The exact P1/P3 null is the primary terminal claim boundary and is retained, not repaired or reclassified.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json`
- **Defect:** Historical terminal_closed_null remains the starting closure (instrument-2 modular-vs-S2 exact null; instrument-1 no usable oracle headroom). Final-revision beds reproduce the same mechanism null against S2 and must not be reframed as Outcome A progress.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** The immutable historical closure and its exact null are preserved as a terminal limit; Outcome A inflation remains forbidden.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json`
- **Defect:** Outcome B machine gates are incomplete at this commit: missing LONG_CONTINUITY_RESULT, MUTATION_REPORT, COUNTERFEIT_REPORT, CLEAN_CLONE, REGENERATION, INDEPENDENT_VERIFICATION, FINAL_SCORECARD, FINAL_CLASSIFICATION, FINAL_STATE, GROK_SCORECARD, TERMINAL_REPORT; REAL_WORLD_SANDBOX_READINESS_MANIFEST.json absent. Publication of Outcome B / substrate_final_revision_complete / real_world_sandbox_ready is refused until all outcome_b_checks actually pass.
  - **Resolution:** unresolved (no matching entry in invocation `resolutions`)
- **Defect:** Grok swarm is not terminal_complete: missing_rounds includes publication_and_claim_boundary_review (this cell); scientific_status externally_blocked; challenge ledger credited_challenges=0. Grok is not a primary endpoint, but Outcome B code still requires completed required cells with no unresolved blockers before publish.
  - **Disposition:** `superseded_by_later_evidence`
  - **Rationale:** This review inspected the score-freeze commit before its own validated output could be present. The current Grok authority now records every required round complete; reviewer grades remain non-primary.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json`
- **Defect:** Field foundation must remain zero-credit: classification_credit=0, current_campaign_endpoint_credit=0, scientific_status=foundation_feasibility_only, full_field_campaign_complete=false. Do not publish field capability, identity transfer, or verified continual learning from foundation artifacts.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** The field foundation is permanently isolated from current endpoint and classification credit; no field capability or identity-transfer claim is made.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FIELD_FOUNDATION_FINAL_STATE.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json`
- **Defect:** Absolute accuracy (~0.875) is not residual cognitive headroom: class 7 scores 0.0 for non-oracle systems while oracle is 1.0; oracle_headroom≈0.125 matches sealed/unanswerable class mass. headroom_exceeds_sesoi must not license activation, discrimination power, or advantage claims.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** Oracle headroom is retained only as a solvability diagnostic and is never interpreted as candidate advantage, activation, or discrimination power.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_CHALLENGE_SCREEN.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json`
- **Defect:** Discrimination classes 0-1 answer from generator cue visible/instruction feature strings shared with stateless policy; those free points are not evidence of owned-state recovery. Public claims must not treat mean scores as general competence.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** Shared cue-feature points are disclosed as a bounded generator limitation and are not used to claim owned-state, general-competence, P1, or P3 advantage.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json`, `docs/archive/experiments/final_revision/LIMITATIONS.md`
- **Defect:** Architecture tournament selected I_simplest_sufficient as engineering default under behavioral tie (why_it_won simplicity after equivalence); architectural_advantage=null. Publishing 'selected kernel won on cognition' is a claim boundary violation.
  - **Disposition:** `accepted_terminal_limit`
  - **Rationale:** Candidate I is retained strictly as the engineering simplicity default after behavioral equivalence; architectural advantage remains null.
  - **Resolution commit:** `74b1f79f385fff03d82b2845ec3e8e04ac35b6a0`
  - **Evidence paths:** `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_ARCHITECTURE_TOURNAMENT.json`, `evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json`
- **Defect:** Public README root still narrates Substrate v5 multimodal_nous_ready_for_review; final-revision publication surfaces must not import that label as current final-revision status without explicit historical scoping.
  - **Disposition:** `fixed`
  - **Rationale:** The root README now explicitly scopes multimodal_nous_ready_for_review as a historical V5 label, identifies terminal_closed_null as the subsequent immutable closure result, refuses to predeclare the in-progress Final Revision outcome, and preserves activation false.
  - **Resolution commit:** `e0aa126bb0436fd3425d9e8fd9585516775d84fa`
  - **Evidence paths:** `README.md`

Resolved blocking defects: 229. Unresolved blocking defects: 9.

## Caveat (closing)

**Again: Grok reviewer grades are not independent external validation, are not a primary endpoint, and cannot override deterministic evidence.** Authority flags: `grok_is_not_an_oracle: true`, `grok_agreement_is_not_a_primary_endpoint: true`. Use sealed principal results, classification documents, and hash-bound evidence for scientific claims.

