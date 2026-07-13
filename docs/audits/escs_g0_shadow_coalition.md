# G0 counterfactual shadow-coalition executor audit

Status: activation-disabled mechanics prototype; no scientific or freeze authority

`src/mop/studies/escs_g0_shadow_coalition.py` composes the existing single-genotype G0 reference
evaluator into a finite multi-actor counterfactual episode. It is deliberately outside `src/mop/escs` so
it does not silently change the sealed ESCS mechanics surface while the detached campaign is active.

## Implemented boundary

- Every construction actor has one explicit input port binding its genotype, sole root node, schema,
  payload form, and byte cap.
- Round zero contains only explicit seeds. One FIFO delivery authorizes one actor evaluation; emitted
  messages become eligible only in the next round.
- Recipient, schema, form, payload, and port-byte checks occur before an outbound batch is enqueued.
  State staging and the complete outbound batch commit atomically inside the counterfactual trace.
- Actor state is immutable during an evaluation and becomes visible only to that actor's next activation.
- Round, activation, queue, message, declared actor-work, routed-payload, retained-state, and repeated
  actor/input/state caps are explicit. Evaluation refusal conservatively charges the actor's full declared
  operation envelope through the existing G0 refusal receipt.
- Episode, delivery, activation, and trace records are self-hashed. Verification independently replays
  successful evaluator receipts and the complete episode from the exact construction, grammar, registry,
  and executor-contract authorities.
- Source, effective, and rollback construction digests are identical. `counterfactual_only` is true;
  activation, shadow-execution authority, factual effects, factual mutation, and scientific promotion are
  hard-false at every exposed level.

Accounting reports evaluator work, logical indexing/dispatch checks, emitted-message work, routed payload
bytes, message-envelope bytes, declared retained-state bytes, and retained byte-rounds separately. These are
deterministic logical counters, not measured FLOPs, latency, energy, or proof that Python interpreter work is
fully captured.

## Deliberate limitations

- V1 requires exactly one root per actor and one port per construction actor. It does not invent fan-in,
  message aggregation, broadcast, implicit defaults, or partial participation semantics.
- It executes a supplied episode; it does not generate mutations, search genotypes, rank candidates, learn
  dispatch, infer utility, or establish complementarity.
- Messages are evaluated one at a time. There is no simultaneous-round arbitration or cross-message atomic
  transaction beyond one actor's complete emitted batch.
- Cycle detection uses a bounded actor/input/state signature. It is a safety halt, not a semantic proof of a
  global fixed point.
- The trace has an in-memory typed verifier but no standalone runner, persisted parser, config, proof
  artifact, or implementation authority. Adding any of those requires a later reviewed revision.
- Nothing here satisfies `G0_FREEZE`, enables X3, installs a genotype into a live substrate, or supports an
  intelligence, efficiency, emergence, ecosystem, or structural-adaptation claim.

Focused tests cover deterministic two-actor delayed routing, next-activation state visibility, atomic route
and byte-cap refusal, conservative evaluation refusal, repeated-state and round/work caps, unchanged
source/effective/rollback identity, replay mismatch, self-hash tampering, and absence of live or campaign
control imports.
