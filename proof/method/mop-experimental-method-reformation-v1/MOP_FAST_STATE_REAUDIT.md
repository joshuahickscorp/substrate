# Fast State Forge reaudit

The science was not rerun. The instruments were, and every terminal result was reclassified by the kernel.

## Instrument reproofs

The repaired order free control passes every invariance in the semantic proof
(repaired_and_proven), and the recurrent architecture fails the same proof, which is what makes the control
discriminating rather than merely permissive.

Arm distinctness over the 11 sealed arms: reproduced, aliased pairs
[].

## Terminal results

| result | sealed verdict | kernel classification | reaudit class |
|---|---|---|---|
| cross_domain_matrix | cross_domain_null | invalid_bed | requires_append_only_correction |
| within_domain_battery | within_domain_null | invalid_bed | requires_append_only_correction |
| secondary_matrix | secondary_null | mechanism_null | fully_valid |
| functional_reorganization | functional_reorganization_null | mechanism_null | fully_valid |
| task_free_context | task_free_context_null | mechanism_null | fully_valid |
| plasticity_policy | simple_partition_policy_sufficient | mechanism_null | fully_valid |
| improvement_rounds | improvement_round_null | mechanism_null | fully_valid |
| third_domain_preflight | invalid_no_temporal_headroom | mechanism_null | fully_valid |

## Findings

### R1 principal results were measured on beds the same program sealed as invalid

- severity: load_bearing
- path: `proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_DOMAIN_VALIDITY.json`
- condition: valid_domains is [har_stream, speech_stream] and principal_domains is empty, while the principal cross domain matrix and the within domain battery ran on har and speech
- expected: a claim about shared temporal dynamics is measured on a bed that requires them
- actual: measured on beds sealed invalid_no_temporal_headroom
- consequence: the nulls stand as nulls about transfer, but not as nulls about temporal dynamics. The dynamics claim is carried by the secondary matrix on har_stream and speech_stream, which is also null, so the scientific conclusion survives with a corrected claim ceiling
- repair: append only claim ceiling correction, no rerun required

### R2 human readable prose broadened a sealed invalid verdict

- severity: load_bearing
- path: `proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_DOMAIN_VALIDITY.json`
- condition: the sealed gate verdict is invalid_no_temporal_headroom and the prose calls the bed marginal
- expected: prose narrows or restates the sealed verdict
- actual: prose asserts the stronger class via ['marginal', 'nearly']
- consequence: a reader of the summary receives a weaker statement of invalidity than the machine sealed
- repair: append only wording correction in this program's synthesis, inherited text untouched

### R3 two registered builders construct the same architecture

- severity: non_load_bearing
- path: `fastforge/arch.py BUILDERS`
- condition: gru and shared_heads both build Conventional(dom, core=gru, share=True)
- expected: every registered builder is distinct or declared as an alias
- actual: an undeclared alias exists in the registry
- consequence: none in the sealed evidence, because shared_heads was never used by a principal run
- repair: declare the alias or delete the unused builder


## Immutability

0 inherited receipts were modified. Every correction is append only and
lives under this program's proof root.
