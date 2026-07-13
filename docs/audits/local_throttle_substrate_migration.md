# Local throttle substrate migration contract

Status: applied and sealed on 2026-07-12.

The migration occurred only after `p5fresh_challenge_cpu` and `p5verify_cpu` had terminal sealed governor
receipts and the active-lane registry was empty. The immutable bridge marker
`c5bbafe78b8149cad38af9395d0328782b62b113f90589a8838c9fc4d94b91c1` binds old policy/governor
`d2d113bf…`/`a1a8d4e3…` to new policy/governor `bdea49bb…`/`0101d5d5…`; campaign
`mac-studio-substrate-policy-transition-v1` adopted that exact transition and terminated complete.

The old policy is frozen in three exact legacy-governor baselines:
`proof/LOCAL_THROTTLE_POLICY_BASELINE_V0.json` binds the completed P5 smoke governor `73ffca97…`,
`proof/LOCAL_THROTTLE_POLICY_BASELINE_V1.json` binds the grid/pilot/fresh governor `bd7dd790…`, and
`proof/LOCAL_THROTTLE_POLICY_BASELINE_V2.json` binds the verifier hotfix governor `a1a8d4e3…`. Each binds
policy `d2d113bf…`, the complete admission safety contract, monotone marker sets, and exact task-scoped
authorities for every P5/P6 producer. These are compatibility evidence, not permission to weaken a task or
safety gate; an old receipt must map to exactly one baseline.

The P5 verifier boundary is now terminal. Its first governed attempt exposed an implementation-only
singleton-classification mismatch: producers correctly kept `n < 2` fresh-subrun contrasts
`undetermined`, while the verifier entry point classified their degenerate intervals directionally.
The correction lives only in `scripts/verify_p5_context_capability.py`, so the already-sealed challenge
source `src/mop/studies/p5_context_verify.py` remains byte-exact; the new verifier artifact binds the
correction script SHA directly. `proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json` has file SHA
`743ce07180f0728f3074b2ac9c78a9aa12ff23f33aeebeffd45bebecacb5f077`, passes every control and mutation,
and returns a terminal scientific null. It is a current prerequisite/control authority, not positive P5
evidence.

The additive task set is frozen in `configs/campaign/substrate_task_overlay.yaml`; its rendered policy
preview is `proof/LOCAL_THROTTLE_SUBSTRATE_POLICY_MIGRATION_PREVIEW.json`. It adds two task markers plus
exact path fragments for the observed Hawking quantization/audit workers, Hugging Face writers, and
Computexchange release builds; four exclusive single-core tasks (EDCM producer/verifier and X0
producer/verifier); and one execution order rooted at the existing P5 verifier. The external markers
only tighten admission and dynamic-pressure handling; they never grant ownership or signal authority.
Both new producers require the exact self-hashed
`proof/ESCS_SUBSTRATE_PREFLIGHT.json`: all nine declared authorities must join live files, all 31 slots
must remain quiescent, and activation/promotion must remain false. It changes no profile, limit,
threshold, admission monitor, existing task, existing prerequisite, command, or P5/P6 order.

The applied post-P5 governor migration made these changes atomically with adversarial tests:

1. recognize `--out`, `--output`, and `--verification-out` as mutually exclusive output-authority flags;
2. require completion provenance for `edcm1_*` and `escs_x0_*` as well as P5/P6;
3. embed a self-hashed task-policy authority at admission and copy the admission-time policy and governor
   bindings into completion authority instead of rehashing mutable files at completion;
4. validate an embedded task authority against the current task and safety contract, or join an old receipt
   to exactly one reviewed legacy baseline; additive marker growth is the only policy relaxation;
5. retain the old governor hash in the explicit compatible set and reject every unknown historic hash;
6. teach producer lookup and prerequisite provenance reports the EDCM/X0 schemas and exact output paths;
7. add receipt mutation tests for task, command, output, policy, implementation, safety, marker removal,
   baseline splice, unknown legacy authority, and ambiguous producer mapping.

The fresh bridge process loaded and verified the authorized hashes before adopting the transition. No
result, seed, horizon, control, or threshold changed during migration. The detached null-safe router now
owns the EDCM/X0/P6 sequence; its governor will wait rather than overlap any exact external-heavy marker.
