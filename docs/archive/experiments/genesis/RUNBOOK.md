# Cognitive Material Genesis — runbook

External activation is `false` and no command here changes that.

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

## Status and gates

```bash
substrate genesis status
substrate genesis preflight
substrate genesis constitution
```

`preflight` refuses to proceed unless the inherited Final Revision Outcome B
classification is byte-identical, all eight preserved tags resolve, and this
branch descends from the Final Revision terminal commit.

## Module self-checks

Each module carries a runnable check that fails if its logic breaks.

```bash
python -m substrate.genesis_statistics
python -m substrate.genesis_harness
python -m substrate.genesis_history
python -m substrate.genesis_claims
python -m substrate.genesis_mutations
python -m substrate.genesis_verification
python -m substrate.genesis_publication
python -m substrate.genesis_continuity
python -m substrate.genesis_tournament
```

## Instruments

Two arms exist so a null can be interpreted. Run them before trusting any
result.

```bash
python - <<'PY'
from substrate import genesis_controls, genesis_reference  # noqa
from substrate.genesis_reference import solvability_report
print(solvability_report()["solvable_count"], "of 14 families solvable from observations")
PY
```

- `record_store_null` must score at or below the 0.125 chance level on every
  family. If it does not, the measure is testing storage, not development.
- `reference_learner` must score near 1.0 on every family. If it does not, that
  family is unanswerable from experience and any failure on it is an artefact.

## Mechanism canaries

```bash
python -c "from substrate import genesis_canaries as c; r=c.run_all(); print(r['all_pass'])"
```

Twelve canaries, each with a paired negative test. A canary that cannot fail is
worthless, so the negative tests are the load-bearing half.

## Tournament, freeze, campaign

The tournament runs every registered arm over every family and history. It is
CPU-bound and parallel; expect roughly fifteen minutes on a 28-core machine.

Freeze must follow the tournament and the canaries. It derives the principal,
replication and hidden-composition seed namespaces from the digest of the
freeze document itself, so no principal instance can be generated before the
freeze exists. The derivation is public:

```python
seed_namespace(freeze_commitment, "principal") == sha256([freeze_commitment, "principal"])
```

The continuity lane runs 12 to 24 hours of real wall-clock time and cannot be
shortened; that duration is what it measures. Principal, replication and
hidden-composition splits run in parallel with it.

## Verification

```bash
python -c "from substrate import genesis_mutations as m; r=m.run(); print(r['injected_count'], r['survivors'])"
```

Recomputation re-derives the decisive effect from raw published rows with plain
arithmetic and no use of the analysis module, so agreement means two
independent paths agreed.

## Stop switch

```bash
substrate genesis stop
substrate genesis resume
```

Every stage checks the stop switch before doing work.
