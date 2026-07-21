# Historical experiment defect ledger

15 defect classes discovered in the Fast State Plasticity Forge and its predecessors.
Every one is rebuilt as a live mutation in `mop.method.acceptance` and as a permanent regression test.

| id | defect | declared | actual | detector | caught at |
|---|---|---|---|---|---|
| D1 | order free control consumed temporal order | order free temporal control | the control contained a Conv1d with kernel 5 and therefore still consumed temporal order | `mop.method.controls.order_free` | control_semantic_proof |
| D2 | replay buffer stopped admitting items | continual replay | the buffer stopped admitting items after filling, so lstm and lstm_gdumb resolved to one policy | `mop.method.controls.replay_active` | mechanism_activity_proof |
| D3 | within domain runs never crossed a context boundary | continual learning across contexts | the within domain runs never crossed a context boundary, so replay had nothing to replay | `mop.method.controls.replay_active` | mechanism_activity_proof |
| D4 | arm aliasing | separate experimental arms | multiple arms shared implementations, defaults or behaviour | `mop.method.arms.distinctness` | arm_distinctness_proof |
| D5 | causal variable with no implementation path | memory_state and H.norm causal effects | one had no causal implementation path and one was a phantom alias | `mop.method.graph.validate` | causal_graph_validation |
| D6 | analytic quantity reported as measured | measured zero forgetting for domain local groups | the zero was true by construction from parameter partitioning and was never measured | `mop.method.contracts.Quantity` | causal_graph_validation |
| D7 | report read a nonexistent key | an answer to Q13 | the code read a key that did not exist and returned None | `mop.method.report.resolve` | report_integrity |
| D8 | baseline identity mismatch | effect versus LSTM plus GDumb | the effect was computed against a different baseline | `mop.method.baseline.comparison` | report_integrity |
| D9 | verdict softening in prose | a summary interpretation | the summary softened the sealed invalid_no_temporal_headroom verdict to the word marginal | `mop.method.report.wording_check` | report_integrity |
| D10 | adversarial panel refuted genuine defects | the panel refuted all attacks | several refuted attacks were genuine reproducible defects | `mop.method.defects.adjudicate` | adjudication |
| D11 | coverage shortfall left implicit | test coverage targets of 92 and 82 percent | statement coverage 68.9 percent and branch coverage 56.0 percent | `mop.method.gate.coverage_gate` | acceptance |
| D12 | ignored treatment flag | a treatment arm controlled by a configuration flag | the flag was read into a configuration object and never reached the implementation | `mop.method.arms.config_sensitivity` | arm_distinctness_proof |
| D13 | future information reached a decision time mechanism | a decision made from information available at the time | a statistic computed over the whole sequence entered a per step decision | `mop.method.graph.validate` | causal_graph_validation |
| D14 | headroom authority from two seeds | measured oracle headroom | the headroom rested on two seeds, inside its own noise | `mop.method.contracts.OracleContract` | oracle_headroom |
| D15 | unconverged baseline produced a verdict | a comparison against a strong baseline | the baseline had not plateaued when the comparison was taken | `mop.method.baseline.comparison` | baseline_convergence |

## The veto rule

a reproduced defect is confirmed regardless of reviewer votes. Consensus is evidence, not proof. Votes alone can never refute an attack; only a failed reproduction can.

A confirmed defect requires all six followups: freeze the original result, add a permanent regression test,
open a bounded repair authority, produce the repaired result, write the consequence analysis, revalidate
every dependent artifact.
