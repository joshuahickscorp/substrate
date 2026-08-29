# Roadmap

Where the project stands and what is likely next. The last campaign ended in a measured
null, so the near-term work is review, cleanup and documentation rather than new claims.
No dates are promised.

## Near term

- Get the Nous Closure package read by real outside reviewers. The reviewers so far were
  internal simulations, and the package already ships a nine-question bank for outsiders.
- Make `substrate v5 verify` work from a fresh clone. Today it stops with "raw principal,
  replication, and open-world receipts are incomplete" until the ~1.1 GiB run tree is
  rebuilt from the compressed raw-receipt archive.
- Get `make lint` and `make types` passing. `ruff format --check` wants 35 files
  reformatted and `mypy` reports 313 errors across 50 files. CI currently only runs
  `ruff check`, which does pass.
- Bring the docs up to date. `docs/ARCHITECTURE.md`, `RUNBOOK.md`, `SCIENTIFIC_STATUS.md`,
  `DEVELOPMENT.md` and `LONG_RUN_PLAN.md` still describe v4, and there is no docs page for
  the Nous Closure campaign at all.

## Later

- Build a harder test bed: a held-out, non-saturated task family where the modular design
  beats an equal-resource monolithic state machine by at least 0.05 with a 95% lower bound
  above zero. Uncertain — the repo says this needs a whole new preregistered program and
  gives no timeline.
- Admitting a real downloaded model stays possible only behind licence, hash, strict-load,
  resource and parity gates. Uncertain — none has ever been admitted and no plan to admit
  one is recorded.

## Not planned

- Nothing downstream runs under the current program. The 12-hour continuity lane, the
  principal campaign, replication and the open-world campaign stay unlaunched after the
  null, and their gated receipts are already published.
- Finished results stay frozen. Changing any premise, threshold, generator, split, seed or
  control requires a separately preregistered new campaign, not a refactor.
- Activation stays `false` and the launch boundary stays closed. `substrate run` is not to
  be crossed without a deliberate operator decision.
