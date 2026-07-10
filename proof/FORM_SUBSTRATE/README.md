# Form-substrate durable evidence

This directory is the durable evidence surface for F1-F20. Raw harness directories under `runs/`
are working state and are not evidence by themselves.

The campaign contract is `campaign/form_substrate_campaign.yaml`. It requires exact equality among
the experiment registry, runnable class, default config, saved run contract, and saved config for:
experiment id, ordered metric list, null hypothesis, and tier. A one-character difference blocks
promotion.

Artifacts are deliberately separated:

- `NULL_CARDS/`: registry-backed nulls and controls. Cards reconstructed during the audit say so and
  do not pretend historical runs were preregistered.
- `RECEIPTS/`: self-contained copies of canonical-candidate run evidence, hashes, metrics, density,
  and component-level OA inputs. A receipt with `all_ok: false` is a durable gap, not a result.
- `PREFLIGHT/`: non-scientific mechanics and fail-closed receipts for evidence-gated lanes.
- `VERIFIERS/`: independent adversarial verifier receipts for positive results.
- `VERDICT_GATES/`: the only authority for promoting a result to a scientific ledger.
- `BOUNDARY_EVIDENCE/`: measured local-resource limits for work claimed to require a Studio.
- `CONTRACT_AUDIT.json`, `OA_INPUT.json`, `DENSITY_INPUT.json`, and `SCORECARD.json`: aggregate inputs
  and campaign status. OA remains component-wise; there is no composite awareness score.
- `PRE_STUDIO_BOUNDARY.json`: separates unfinished local work, measured Studio hardware walls,
  external environment/license blockers, and work beyond the proposed Studio.

Run the pipeline with:

```bash
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py audit
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py preregister
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py preflight
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py run-local
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py collect
PYTHONPATH=src .venv/bin/python scripts/form_substrate_verifiers.py --refresh
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py gate
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py scorecard
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py boundary
PYTHONPATH=src .venv/bin/python scripts/form_substrate_campaign.py bundle
```

`bundle` is expected to fail until every required proof file exists and is tracked (or copied into a
durable bundle). `PRE_STUDIO_BOUNDARY.json` can claim a Studio-only hardware wall only after every
local obligation is complete, every Studio-scale claim has a measured boundary receipt, and no
campaign leg remains classified beyond Studio.
