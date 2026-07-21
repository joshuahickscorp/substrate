# Method reformation ledger

## What this program changed

The previous method validated instrumentation after expensive scientific execution. This one inverts the
order: 12 stages run before any principal training compute, and an experiment that
fails one of them cannot spend any.

## Defect ledger

18 classes, every one rebuilt as a live mutation and as a permanent regression test.
20 mutations, all rejected: True.

| id | defect | detector | discovered |
|---|---|---|---|
| D1 | order free control consumed temporal order | `mop.method.controls.order_free` | inherited |
| D2 | replay buffer stopped admitting items | `mop.method.controls.replay_active` | inherited |
| D3 | within domain runs never crossed a context boundary | `mop.method.controls.replay_active` | inherited |
| D4 | arm aliasing | `mop.method.arms.distinctness` | inherited |
| D5 | causal variable with no implementation path | `mop.method.graph.validate` | inherited |
| D6 | analytic quantity reported as measured | `mop.method.contracts.Quantity` | inherited |
| D7 | report read a nonexistent key | `mop.method.report.resolve` | inherited |
| D8 | baseline identity mismatch | `mop.method.baseline.comparison` | inherited |
| D9 | verdict softening in prose | `mop.method.report.wording_check` | inherited |
| D10 | adversarial panel refuted genuine defects | `mop.method.defects.adjudicate` | inherited |
| D11 | coverage shortfall left implicit | `mop.method.gate.coverage_gate` | inherited |
| D12 | ignored treatment flag | `mop.method.arms.config_sensitivity` | inherited |
| D13 | future information reached a decision time mechanism | `mop.method.graph.validate` | inherited |
| D14 | headroom authority from two seeds | `mop.method.contracts.OracleContract` | inherited |
| D15 | unconverged baseline produced a verdict | `mop.method.baseline.comparison` | inherited |
| D16 | a context split that crosses no boundary | `mop.method.bed.context_boundary` | this program |
| D17 | a brittle plateau criterion reported a flat curve as still improving | `mop.method.baseline.plateau` | this program |
| D18 | a label permutation control scored against zero difference instead of the majority class rate | `mop.method.runs.mutations.e1_mutations` | this program |

## Fast State Forge reaudit

Findings: ['R1', 'R2', 'R3']. Load bearing: ['R1', 'R2'].
Inherited receipts modified: 0.

within domain, no arm beat the strongest baseline on har and speech, beds sealed as invalid_no_temporal_headroom. No within domain measurement exists on a bed that requires temporal dynamics, so the within domain question is open on har_stream and speech_stream

## Experiment selection

Queue selected ['E1', 'E4'] and refused ['E3', 'E5'].

## E1 core by readout factorial

| bed | contrast | mean | lower 95 cb | verdict |
|---|---|---|---|---|
| har_stream | core_effect_at_linear | +0.4550 | +0.4474 | positive |
| har_stream | core_effect_at_mlp | +0.4552 | +0.4421 | positive |
| har_stream | readout_effect_at_pooled | -0.0006 | -0.0096 | wrong_direction_failure |
| har_stream | readout_effect_at_fast | -0.0004 | -0.0083 | wrong_direction_failure |
| har_stream | long_range_state_at_linear | +0.1660 | +0.1555 | positive |
| har_stream | long_range_state_at_mlp | +0.1632 | +0.1530 | positive |
| har_stream | oracle_segmentation_value_at_linear | +0.0064 | -0.0017 | null_futile |
| har_stream | oracle_segmentation_value_at_mlp | -0.0002 | -0.0091 | wrong_direction_failure |
| har_stream | best_cell_over_external_baseline | +0.0084 | -0.0016 | null_futile |
| speech_stream | core_effect_at_linear | +0.4647 | +0.4420 | positive |
| speech_stream | core_effect_at_mlp | +0.4908 | +0.4818 | positive |
| speech_stream | readout_effect_at_pooled | -0.0111 | -0.0214 | wrong_direction_failure |
| speech_stream | readout_effect_at_fast | +0.0149 | -0.0046 | null |
| speech_stream | long_range_state_at_linear | +0.5895 | +0.5716 | positive |
| speech_stream | long_range_state_at_mlp | +0.6056 | +0.5874 | positive |
| speech_stream | oracle_segmentation_value_at_linear | +0.0334 | +0.0184 | null |
| speech_stream | oracle_segmentation_value_at_mlp | +0.0123 | -0.0036 | null |
| speech_stream | best_cell_over_external_baseline | +0.0682 | +0.0502 | positive |

## E4 adaptation locus

| bed | contrast | mean | lower 95 cb | verdict |
|---|---|---|---|---|
| speech_stream | acquisition state_only_vs_no_adapt | +0.0886 | +0.0624 | positive |
| speech_stream | acquisition state_only_vs_state_noise | +0.0932 | +0.0666 | positive |
| speech_stream | acquisition head_only_vs_no_adapt | +0.1374 | +0.0991 | positive |
| speech_stream | acquisition adapter_only_vs_no_adapt | +0.1357 | +0.1052 | positive |
| speech_stream | acquisition core_only_vs_no_adapt | +0.2709 | +0.2141 | positive |
| speech_stream | acquisition full_vs_no_adapt | +0.2709 | +0.2149 | positive |
| speech_stream | acquisition state_only_vs_full | -0.1823 | -0.2219 | harm |
| speech_stream | acquisition state_only_vs_head_only | -0.0488 | -0.0722 | wrong_direction_failure |
| har_stream | acquisition state_only_vs_no_adapt | +0.1313 | +0.0864 | positive |
| har_stream | acquisition state_only_vs_state_noise | +0.1338 | +0.0892 | positive |
| har_stream | acquisition head_only_vs_no_adapt | +0.2745 | +0.2103 | positive |
| har_stream | acquisition adapter_only_vs_no_adapt | +0.2740 | +0.2158 | positive |
| har_stream | acquisition core_only_vs_no_adapt | +0.3687 | +0.3060 | positive |
| har_stream | acquisition full_vs_no_adapt | +0.3726 | +0.3071 | positive |
| har_stream | acquisition state_only_vs_full | -0.2414 | -0.2971 | harm |
| har_stream | acquisition state_only_vs_head_only | -0.1432 | -0.1917 | harm |

## Positive mutations

All rejected: True.

## Coverage

Kernel statement 98.9 percent against a target of 92, branch
94.4 percent against a target of 82. Program stages are measured separately
at 0.0 percent and are listed rather than hidden.

## Activation

False. Nothing here licenses an architecture, and no claim in this ledger extends beyond its measured path.
