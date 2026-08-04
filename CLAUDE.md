# UFCPolyML contributor guide

## Project scope

V1 is a batch applied-ML system:

`DATA → FEATURES → XGBoost + EVAL → mismatch report / paper-signal log`

The model is strictly odds-free. Market probability is a report comparator, never
a feature or training label. No deep learning, LLM enrichment, automated trading,
Kelly sizing, RL/OPE, CLOB simulation, or online learning belongs in v1.

## Non-negotiable correctness rules

- A feature for fight `X` may use only information available strictly before `X`.
- Training, calibration, and evaluation use event-grouped temporal splits; never
  random row splits.
- Calibration data must be disjoint from model-training data and precede test data.
- Features are emitted before a fight updates any accumulator.
- Validation failures are quarantined with reason codes; source rows are never
  silently dropped or auto-corrected.
- XGBoost consumes no historical or live market-derived feature.
- A served matchup probability must be symmetric: reversing fighter order returns
  exactly `1 - p`.

## Engineering conventions

- Python 3.12 and `uv`; use `uv run`, never global `pip` installs.
- DuckDB is the single datastore. Hydra holds configuration; no magic constants.
- Public functions are typed. Use frozen Pydantic models at external boundaries.
- Prefer small, named functions with one responsibility. Format with `ruff`.
- Run fixture-based tests only; scraper tests must never require the live site.
- Never commit secrets, DuckDB files, or raw data blobs.

## Documentation rules

- `docs/REQUIREMENTS.md` is the testable system contract; `docs/DESIGN.md` defines
  its implementation shape; `docs/TASKS.md` is the build order.
- `docs/DECISIONS.md` records active, non-obvious technical choices and rationale.
- `docs/FEATURES.md` is the human-owned registry. Do not invent feature ideas;
  mark unclear definitions `TODO(human)`.

## Current status

| Area | Status |
|---|---|
| Data ingestion, storage, validation | Built |
| Polymarket market capture | Built; production cron is P0 |
| Feature replay engine | Designed, not built |
| XGBoost, evaluation, calibration | Designed; implementation pending |
| Mismatch report and paper-signal log | Design pending |

## Common commands

```bash
make setup
make validate
make test
make lint
make format
make scrape
make capture
make backup
```
