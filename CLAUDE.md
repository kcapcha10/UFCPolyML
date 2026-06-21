# CLAUDE.md — ufc-edge

Read this before touching anything. It's the rules of the house.

## What this project is

A systematic prediction-market trading system for **Polymarket UFC markets**. The
edge thesis is **not** "predict fight winners" — it's **find where Polymarket prices
are wrong**. A learning pipeline produces calibrated `P(win)` probabilities; a
separate strategy layer compares them to market prices, sizes positions with
fractional Kelly, and models execution realistically using self-captured order-book
snapshots. Right now only the **data layer** is active: a ufcstats.com scraper and a
5-minute Polymarket order-book capture cron.

The full system is a **six-layer architecture** (high-level locked, low-level pending
— see [DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md)):
DATA → FEATURES → REPS → PREDICT → STRATEGY, with SIM (CLOB replay) and ONLINE
(feedback loop) as additional layers. The predictor (PREDICT) is model-agnostic;
XGBoost is the baseline and control, not the definition.

## Layer separation contract (architectural invariant)

Six layers; the separation contract must hold across all of them. Full detail and
the five build invariants are in
[DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md).

- **DATA** (`src/ufc_edge/data/`, `src/ufc_edge/market/`): scrapers, capture cron,
  offline LLM enrichment (cached structured columns). Single datastore: DuckDB.
- **FEATURES** (`src/ufc_edge/features/`): tabular as-of features — stats,
  short-notice/injury enrichment. Strict `event_date < fight_date` cutoff.
- **REPS** (`src/ufc_edge/reps/`, *not yet scaffolded*): learned representations —
  sequence encoder (GRU/Transformer over career history) + fight-graph GNN. Both
  emit as-of embeddings and are **two new leakage surfaces**, each pytest-guarded.
- **PREDICT** (`src/ufc_edge/model/`): model-agnostic interface — tabular ⊕
  embeddings → raw score → calibrate (Platt → isotonic) → `P(win)`. XGBoost is
  the baseline/control; never the only option.
- **STRATEGY** (`src/ufc_edge/strategy/`): sequential decision problem — Kelly/
  shrinkage → bandit → offline RL. Market-derived features (§16) live here, never
  in PREDICT. Behavioral features live here too.
- **SIM** (`src/ufc_edge/sim/`, *not yet scaffolded*): CLOB replay simulator.
  Rebuilds the order book at any decision time; a policy may only observe state
  that existed then — no future-price lookahead.
- **ONLINE** (`src/ufc_edge/online/`, *not yet scaffolded*): feedback on fight
  resolution — neural warm-start (Adam + replay buffer), GBT retrain cadence,
  rolling recalibration, drift monitoring.

**The model is structurally odds-free.** No odds-derived signal — historical or
live — enters FEATURES, REPS, or PREDICT. All market signal lives in STRATEGY.
This is what makes the separation contract the defense against circularity, not a
rule layered on top of it.

## Temporal integrity (the most important rule in this repo)

Every feature must be computed with a **strict as-of timestamp**: no information
from a fight may leak into features for that fight or any earlier fight. This is
where most public UFC models silently leak and report fake accuracy. No-leakage
will be **pytest-enforced** once the feature layer exists; until then, treat any
computation that touches future data as a defect, not a style issue.

The as-of rule extends through the full pipeline:
- **FEATURES:** `event_date < fight_date`. Opponent-trajectory features (FEATURES.md
  §9d) are the primary trap — they are deliberately leakage-shaped.
- **REPS:** both the sequence encoder (reads fights strictly before T) and the
  fight-graph GNN (edges with `event_date < fight_date`) are new leakage surfaces,
  each with their own pytest guard.
- **Offline LLM enrichment:** `source_published_at < fight_date`. Being offline buys
  reproducibility, not leakage protection — an offline job can still read a
  post-fight article. The guard is on the *source*.
- **SIM:** a policy may only observe order-book state that existed at decision time.

## Documentation-first workflow

- `DECISIONS.md` — every non-obvious choice gets a one-line rationale and revisit
  condition. Rejected approaches go here too.
- `DECISIONS_architecture_redesign.md` — the **locked high-level architecture**
  (six layers, five invariants, D1–D7). Read before touching anything structural.
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

Implementation stops after the capture cron. Everything below FEATURES is either
an empty scaffold (`__init__.py` only) or does not exist yet.

| Layer / Area | Package | Status |
|---|---|---|
| DATA — ufcstats scraper | `src/ufc_edge/data/ufcstats/` | **Built** |
| DATA — DuckDB schemas + storage | `src/ufc_edge/data/` | **Built** |
| DATA — Polymarket capture cron | `src/ufc_edge/market/` | **Built** (P0 — must stay running) |
| DATA — Kaggle cross-check | `src/ufc_edge/data/kaggle.py` | **Missing** — referenced in `dvc.yaml` but not implemented |
| DATA — offline LLM enrichment (D7) | `src/ufc_edge/data/enrichment/` | **Not started** — do not implement |
| FEATURES | `src/ufc_edge/features/` | **Scaffolded, empty** — do not implement |
| REPS (sequence encoder + GNN) | `src/ufc_edge/reps/` | **Not scaffolded** — do not implement |
| PREDICT (model-agnostic; XGBoost baseline) | `src/ufc_edge/model/` | **Scaffolded, empty** — do not implement |
| STRATEGY (Kelly → bandit → RL) | `src/ufc_edge/strategy/` | **Scaffolded, empty** — do not implement |
| SIM (CLOB replay) | `src/ufc_edge/sim/` | **Not scaffolded** — do not implement |
| ONLINE (feedback loop) | `src/ufc_edge/online/` | **Not scaffolded** — do not implement |
| Deployment | `deploy/` | Only the capture cron is deployed (Fly.io) |

## Environment facts

- **uv** manages everything; Python 3.12. `uv sync --extra dev` then
  `uv run <cmd>`. Never pip-install into a global env.
- **DuckDB** is the single datastore: `data/ufc_edge.duckdb` (override with
  `DUCKDB_PATH`). The connection manager lives in `src/ufc_edge/data/storage.py`;
  ufcstats upserts in `src/ufc_edge/data/ufcstats/storage.py`; Polymarket upserts
  in `src/ufc_edge/data/polymarket/storage.py`. All rows carry
  `scraped_at`/`captured_at`.
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
