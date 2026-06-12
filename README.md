# UFCPolyML

Systematic UFC outcome prediction and Polymarket trading — from raw fight stats to
live Kelly-sized bets.

Current phase: **data layer only**. A ufcstats.com scraper, a Kaggle cross-check, and
a 5-minute Polymarket order-book capture cron. The model, strategy, and LLM-enrichment
layers are scaffolded but intentionally empty — see [CLAUDE.md](CLAUDE.md) for the
rules of the house and [FEATURES.md](FEATURES.md) for the feature registry.

## Quickstart

```bash
make setup            # uv sync + pre-commit install + dvc init
cp .env.example .env  # fill in secrets (never committed)
make capture          # one Polymarket capture tick (verifies API access)
make scrape           # start the ufcstats crawl (resumable)
make test             # parser tests against fixtures — never hits the live site
```

All data lands in `data/ufc_edge.duckdb` (override with `DUCKDB_PATH`).

## Deploying the capture cron (Fly.io)

The capture cron snapshots every active UFC order book on Polymarket every 5
minutes. Its history is irreplaceable (Polymarket's historical order-book endpoint
stopped updating in Feb 2026), so this must stay running.

First-time setup:

```bash
curl -L https://fly.io/install.sh | sh   # install flyctl
flyctl auth login                        # browser login
flyctl apps create ufc-edge-capture
flyctl volumes create capture_data --region sjc --size 1 --app ufc-edge-capture
```

Deploy (and redeploy after any code change), always from the repo root:

```bash
flyctl deploy --config deploy/fly.toml --dockerfile deploy/Dockerfile
```

Verify it's ticking:

```bash
flyctl logs --app ufc-edge-capture            # look for "tick complete ..." lines
flyctl ssh console --app ufc-edge-capture -C \
  "python -c \"import duckdb; print(duckdb.connect('/data/ufc_edge.duckdb').execute('SELECT COUNT(*), MAX(captured_at) FROM order_book_snapshots').fetchall())\""
```

Pull the captured history down for analysis:

```bash
flyctl ssh sftp get /data/ufc_edge.duckdb data/raw/capture_remote.duckdb --app ufc-edge-capture
```

Secrets (when Polymarket credentials become necessary — public endpoints need none):

```bash
flyctl secrets set POLYMARKET_API_KEY=... --app ufc-edge-capture
```

## Repo map

| Path | What lives there |
|---|---|
| `src/ufc_edge/data/` | ufcstats scraper, Kaggle cross-check, Pydantic schemas, DuckDB I/O |
| `src/ufc_edge/market/` | Polymarket capture cron |
| `src/ufc_edge/{features,model,strategy}/` | scaffolded, empty by design |
| `configs/` | Hydra config tree (scrape politeness, capture cadence, MLflow) |
| `deploy/` | Dockerfile + fly.toml for the capture cron |
| `data/` | DVC-tracked DuckDB + raw artifacts (gitignored) |
| `tests/` | fixture-based parser tests |
