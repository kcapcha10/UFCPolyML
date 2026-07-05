# UFCPolyML

Systematic UFC outcome prediction and Polymarket trading — from raw fight stats to
live Kelly-sized bets.

Current phase: **data layer only**. A ufcstats.com scraper and a 5-minute Polymarket
order-book capture cron. The model, strategy, and LLM-enrichment
layers are scaffolded but intentionally empty — see [CLAUDE.md](CLAUDE.md) for the
rules of the house and [docs/FEATURES.md](docs/FEATURES.md) for the feature registry.

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

## DVC remote (Google Drive) — one-time setup

The DVC remote is Google Drive (see docs/DECISIONS.md). Google requires a self-created
OAuth client for dvc-gdrive, so setup is a one-time browser dance:

1. Create a folder in Google Drive; copy the folder ID from its URL
   (`https://drive.google.com/drive/folders/<FOLDER_ID>`).
2. In [console.cloud.google.com](https://console.cloud.google.com): create a
   project → enable the **Google Drive API** → configure the OAuth consent screen
   (External; add yourself as a test user) → Credentials → **Create OAuth client
   ID** (type: Desktop app).
3. Wire it up (the secret goes in `.dvc/config.local`, which is gitignored):

```bash
uv run dvc remote add -d gdrive gdrive://<FOLDER_ID>
uv run dvc remote modify gdrive gdrive_client_id '<CLIENT_ID>'
uv run dvc remote modify --local gdrive gdrive_client_secret '<CLIENT_SECRET>'
```

4. The first `uv run dvc push` opens a browser for Google sign-in; after that,
   pushes are non-interactive.

Then back up the irreplaceable capture history (run at least after every event
weekend — gaps are permanent):

```bash
make backup   # SFTP the Fly DuckDB down → dvc add → dvc push
```

## Repo map

| Path | What lives there |
|---|---|
| `src/ufc_edge/data/` | ufcstats scraper, Pydantic schemas, DuckDB I/O |
| `src/ufc_edge/data/polymarket/` | Polymarket capture cron |
| `src/ufc_edge/{features,reps,model,strategy,sim,online}/` | scaffolded, empty by design |
| `docs/` | decision log, locked architecture, feature registry |
| `.claude/spec/` | low-level design spec: requirements, design, tasks, key design decisions |
| `configs/` | Hydra config tree (scrape politeness, capture cadence, MLflow) |
| `deploy/` | Dockerfile + fly.toml for the capture cron |
| `data/` | DVC-tracked DuckDB + raw artifacts (gitignored) |
| `tests/` | fixture-based parser tests |

## Documentation map

Start with [CLAUDE.md](CLAUDE.md) (rules of the house), then:

| Doc | What it answers |
|---|---|
| [docs/DECISIONS_architecture_redesign.md](docs/DECISIONS_architecture_redesign.md) | The locked six-layer architecture, five invariants, decisions D1–D7 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Every non-obvious engineering choice, with rationale and revisit conditions |
| [docs/FEATURES.md](docs/FEATURES.md) | The canonical (human-owned) feature registry with as-of cutoffs |
| [.claude/spec/Key Design Decisions.md](.claude/spec/Key%20Design%20Decisions.md) | Plain-language "why" record for key low-level design calls (D8+) |
| [.claude/spec/](.claude/spec/) | The living LLD spec: `requirements.md`, `design.md`, `tasks.md` |
