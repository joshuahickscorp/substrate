# MOP Scientific Frontier: Ancestry

Frontier branch `agent/mop-scientific-frontier` off base commit `094bdd9` (all scientific evidence). Current `main` `a1d6be3`. Source branch `agent/mop-evidence-salvage`.

## Included evidence commits

- `fae3eef7e` (stopped-run forensic closure): Salvage Phase 1 forensic closure + durable ledger
- `b88462b77` (categorized-wave recovery): Salvage Phase 2 incremental verifier recovers categorized wave
- `f66d1f234` (repaired runtime): Salvage Phase 3 work-conserving scheduler + reliability
- `d1c105372` (Phase 4): Salvage Phase 4 admission batteries prune all three lanes
- `16dc1c6b0` (Phase 4B): Salvage Phase 4B repaired calibrated battery admits all three
- `9ac24c641` (MNIST canaries): Campaign 2 launch N1 canary positive
- `b1d7c97fe` (MNIST canaries): Campaign 2 canaries terminal N1+P1R positive U1 pruned
- `ffb647f84` (CIFAR confirmations): Campaign 2 confirmation authorities N1 CIFAR-10 P1R CIFAR-100
- `d7f752bcd` (CIFAR confirmations): Campaign 2 confirmation terminal P1R confirms N1 nulls
- `094bdd934` (KMNIST R1 and P1R results): Cluster B R1 null KMNIST P1R third-source positive terminated

## Excluded unrelated

- `agent/mop-accretion-collapse` (PR #31 codebase collapse): must not be modified, rebased, merged, or disrupted.
- `agent/mop-extreme-condensation`: separate condensation baseline.
- Telegram notifier / orchestrator / General Run infrastructure: historical, immutable, not scientific evidence.

## Artifact hashes

- `campaign2/reports/MOP_CAMPAIGN2_CANARY_SUMMARY.json`: `f3982fdc415410ef`
- `campaign2/reports/MOP_CAMPAIGN2_CONFIRMATION_SUMMARY.json`: `dd1d7b708485ef31`
- `campaign2/reports/MOP_CAMPAIGN2_LAUNCH_RECEIPT.json`: `305358c5f273067c`
- `campaign2/reports/MOP_CLUSTERB_TERMINAL.json`: `c93b73878f945186`
- `campaign2/reports/MOP_G1_N1_CANARY_PREREG.json`: `b2532bf8666e49fa`
- `campaign2/reports/MOP_G1_N1_CANARY_RESULT.json`: `fb04af067eaf8e06`
- `campaign2/reports/MOP_G1_N1_CONFIRM_PREREG.json`: `6249b0ae492b5048`
- `campaign2/reports/MOP_G1_N1_CONFIRM_RESULT.json`: `f34131fba009ffd9`
- `campaign2/reports/MOP_G1_P1R_CANARY_RESULT.json`: `ae8fdb7bcc94cc03`
- `campaign2/reports/MOP_G1_P1R_CONFIRM_PREREG.json`: `908a686bc4b4e881`
- `campaign2/reports/MOP_G1_P1R_CONFIRM_RESULT.json`: `040d39c2630be2bb`
- `campaign2/reports/MOP_G1_P1R_THIRDSOURCE_RESULT.json`: `21f2db185be677de`
- `campaign2/reports/MOP_G1_R1_ADMISSION_PREREG.json`: `99ff5788d9e52e64`
- `campaign2/reports/MOP_G1_R1_ADMISSION_RESULT.json`: `481b4d4047993648`
- `campaign2/reports/MOP_G1_U1_CANARY_RESULT.json`: `93e9b85aa75fe6a1`
- `salvage/MOP_EVIDENCE_CAMPAIGN_STATE.json`: `a3c55ebcb4374ffc`

## Rollback
```
git -C /Users/scammermike/Downloads/mop worktree remove /Users/scammermike/Downloads/mop-scientific-frontier
git -C /Users/scammermike/Downloads/mop branch -D agent/mop-scientific-frontier
# evidence remains intact on agent/mop-evidence-salvage at 094bdd9
```

No model or assistant attribution in commits.
