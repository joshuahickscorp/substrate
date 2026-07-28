# Substrate v4 terminal report

- Classification: `functional_proto_nous_candidate`
- Ready commit: `3eec1611e4f887d3955c6f41604dcceeccbe0f81`
- Raw receipts: `2136` of `2136`
- Principal histories: `48`
- Workload families: `8`
- Episodes: `181560`
- Models induced across unit summaries: `3129`
- Models revised: `792`
- Interventions: `18720`
- Counterfactuals: `3900`
- Mappings inferred: `71355`
- Inquiry actions: `3900`
- Principal wall time: `65.03358070796821` seconds
- Peak worker RSS: `50.09` MiB
- Worker count: `4`
- Checkpoint and body continuity: `pass`
- Independent verification: `pass`
- Mutations: `18/18` detected, `0` survivors
- Clean clone, clean install, full v4/runtime tests, lint, and double regeneration: `pass`
- Review package: `complete`
- Independent replication effect: `0.0310` (95% CI `0.0231` to `0.0398`; SESOI `0.05`; `null`)
- Generator-held-out open-world effect: `0.1945` (95% CI `0.1857` to `0.2033`; SESOI `0.05`; `positive`)
- Strongest missing condition: `independent replication effect must clear the preregistered 0.05 SESOI`
- Hawking coexistence: observation only; no signals or controller changes
- Activation: `false`
- Claim boundary: functional engineering and scientific classification only; no consciousness, sentience, personhood, life, moral status, or unqualified Nous claim

## Independently recomputed primary effects

| Hypothesis | Effect | 95% bootstrap CI | SESOI | Result |
|---|---:|---:|---:|---|
| H_S1 | 0.7767 | 0.7496 to 0.8031 | 0.05 | positive |
| H_S2 | 0.5947 | 0.5461 to 0.6433 | 0.05 | positive |
| H_S3 | 0.5433 | 0.5044 to 0.5822 | 0.05 | positive |
| H_S4 | 0.6333 | 0.6333 to 0.6333 | 0.05 | positive |
| H_S5 | 1.0000 | 1.0000 to 1.0000 | 0.05 | positive |
| H_S6 | 0.9600 | 0.9600 to 0.9600 | 0.05 | positive |
| H_S7 | 0.4392 | 0.4044 to 0.4739 | 0.05 | positive |
| H_S8 | 0.2974 | 0.2737 to 0.3210 | 0.05 | positive |
| H_S9 | 0.6031 | 0.5586 to 0.6475 | 0.05 | positive |
| H_S10 | 1.0000 | 1.0000 to 1.0000 | 0.05 | positive |

Replication and generator-held-out open-world results are classified independently against the frozen SESOI. The complete raw receipt archive, controls, null ledger, defect ledger, mutations, reproduction instructions, and known limitations are under `artifacts/substrate/v4/review/`.
