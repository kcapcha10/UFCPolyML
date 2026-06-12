# CLAUDE.md — ufc-edge

Read this before touching anything. It's the rules of the house.

## What this project is

A systematic prediction-market trading system for **Polymarket UFC markets**. The
edge thesis is **not** "predict fight winners" — it's **find where Polymarket prices
are wrong**. A calibrated XGBoost model (~5,000 fights, ~80 temporally-aware
features) produces probabilities; a separate strategy layer compares them to market
prices, sizes positions with fractional Kelly, and models execution realistically
using self-captured order-book snapshots. Right now only the **data layer** is
active: a ufcstats.com scraper, a Kaggle cross-check, and a 5-minute Polymarket
order-book capture cron.

## Layer separation contract (architectural invariant)

- **Model layer** (`src/ufc_edge/model/`): predictive features only. Inputs that
  estimate P(fighter wins).
- **Strategy layer** (`src/ufc_edge/strategy/`): everything behavioral and
  execution-related — market behavior, sizing, slippage, correlation.
- **Behavioral features (entertainment-seeking, gameplan adherence, impulsivity)
  are excluded from XGBoost by design.** They belong to the post-model strategy
  layer. Do not move them into the model. Each layer documents its own assumptions
  and limitations.

## Temporal integrity (the most important rule in this repo)

Every feature must be computed with a **strict as-of timestamp**: no information
from a fight may leak into features for that fight or any earlier fight. This is
where most public UFC models silently leak and report fake accuracy. No-leakage
will be **pytest-enforced** once the feature layer exists; until then, treat any
computation that touches future data as a defect, not a style issue. Watch
especially for retroactive signals (e.g. opponent-trajectory features) — they are
deliberately leakage-shaped and need careful cutoff handling.

## Documentation-first workflow

- `DECISIONS.md` — every non-obvious choice (library, schema shape, rate limit)
  gets a one-line rationale and, where relevant, a revisit condition. Rejected
  approaches go here too.
- `FEATURES.md` — the **canonical feature registry**. Registry, not brainstorm.
- If you make a judgment call while coding, log it before moving on.

## Feature-ideation boundary

**The human owns feature ideation.** Claude Code's role is implementation,
infrastructure, and narrative/context — not inventing predictive features. Never
add features to `FEATURES.md` that the human hasn't specified. If a definition is
ambiguous, leave a `TODO(human):` marker; don't guess.

## Code style (CS106B philosophy, Python-first)

- **Decomposition is the top priority.** One logical task per function, named for
  that task. ~15–30 lines; >50 is a smell. Entry points (a `main`, a spider's
  `parse`, a cron tick) read like an outline of well-named calls.
- **No redundant code.** Repeated logic becomes a helper. Copy-paste with edits is
  a defect.
- **Names reveal intent.** `significant_strikes_landed`, never `ssl`. Python:
  `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- **Comments are contracts, not narration.** Docstrings say what/params/returns/
  pre-post-conditions. Module header states responsibility. Comment the tricky
  parts (rate-limit choices, leakage subtleties, CLOB quirks) — never the obvious.
- **No magic numbers.** A bare `300` in the cron is a defect;
  `CAPTURE_INTERVAL_SECONDS = 300` is correct.
- **Restraint.** Clear over clever. No global mutable state. Immutable by default
  (frozen Pydantic models). Right data structure for the job.
- **Types.** Hint all public functions. Pydantic for any structured external data;
  validate at the boundary, fail loudly.
- Formatting is **ruff format**'s job (configured in `pyproject.toml`) — don't
  fight it.

## Scope map

| Area | Status |
|---|---|
| Data layer (scraper, Kaggle cross-check, DuckDB) | **Active now** |
| Polymarket capture cron | **Active now** (P0 — must stay running) |
| Features (`src/ufc_edge/features/`) | Scaffolded, empty — do not implement |
| Model (XGBoost, calibration) | Scaffolded, empty — do not implement |
| Strategy (Kelly, backtest, Sharpe) | Scaffolded, empty — do not implement |
| LLM enrichment (Llama 3.1 70B, offline/cached) | Not started — do not implement |
| Deployment (FastAPI, paper trading) | Only the capture cron is deployed |

## Environment facts

- **uv** manages everything; Python 3.12. `uv sync --extra dev` then
  `uv run <cmd>`. Never pip-install into a global env.
- **DuckDB** is the single datastore: `data/ufc_edge.duckdb` (override with
  `DUCKDB_PATH`). Connection + idempotent upserts live in
  `src/ufc_edge/data/storage.py`; all rows carry `scraped_at`/`captured_at`.
- **Secrets** come from `.env` (gitignored). `.env.example` lists key names.
  Never commit a secret; detect-secrets runs in pre-commit.
- **Configs** are a Hydra tree under `configs/` (`data/`, `scrape/`, `capture/`,
  plus `mlflow.yaml`). Rate limits and intervals are config values, not literals.
- **DVC** versions `data/`; **MLflow** is wired to local sqlite
  (`mlruns/mlflow.db`) so `mlflow.start_run()` works with zero setup. No
  experiments exist yet.
- **Capture cron** deploys to Fly.io from `deploy/`; secrets via `fly secrets`.
  Its history is irreplaceable — Polymarket's historical order-book endpoint is
  dead, so every gap in our capture is permanent. Be careful around it.
- **Compute:** Stanford FarmShare NVIDIA L40 GPUs are available for later training.
  Not used today.

## Commands

| Command | What it does |
|---|---|
| `make setup` | uv sync + pre-commit install + dvc init guard |
| `make lint` | ruff check |
| `make format` | ruff format |
| `make test` | pytest (parsers run against fixtures, never the live site) |
| `make scrape` | run the ufcstats Scrapy spider (resumable via JOBDIR) |
| `make capture` | run one Polymarket capture tick locally for verification |
