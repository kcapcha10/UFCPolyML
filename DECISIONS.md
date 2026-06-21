# DECISIONS.md

Engineering decisions log. Format per entry: **Context → Decision → Rationale →
Revisit when**. Rejected approaches and abandoned features are recorded here too, so
future sessions don't re-litigate settled questions. Newest sections grouped by area.

---

## Architecture

### Four-layer design → six, learning-capable (2026-06-21)
- **Context:** The original four-layer design (data · features · model · strategy)
  had only data + capture built; features/model/strategy were empty scaffolds. The
  human decided to grow the model and strategy layers into something that learns.
- **Decision:** **Accepted** a six-layer, learning-capable redesign — keep the
  separation-contract spine, evolve the organs. Adds a **representation** layer
  (sequence encoder + fight-graph GNN), a **simulator** layer (CLOB replay), and a
  cross-cutting **online feedback** layer; PREDICT becomes model-agnostic (XGBoost
  is baseline + control); STRATEGY becomes a sequential decision problem evaluated
  OPE-first. Full record (architecture diagram, five invariants, decisions D1–D7) in
  [DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md).
- **Status:** High-level design **locked**; low-level design (schemas, interfaces,
  per-layer tradeoffs) is **human-owned and pending** — do not implement the new
  layers yet. This supersedes the four-layer framing wherever the two conflict.
- **Rationale:** The separation contract is the repo's strongest asset and the
  structural defense against circularity and leakage; the redesign extends it rather
  than replacing it. Notable downstream consequences recorded below and in
  FEATURES.md: market features (old §16) relocate to strategy, §7 is dropped (model
  is now structurally odds-free), and short-notice/injury enrichment (D7) enters the
  model gated on `source_published_at < fight_date`.
- **Revisit when:** Low-level design lands, or any of the five invariants proves
  unworkable in implementation.

---

## Tooling & environment

### uv for environment and dependency management
- **Context:** Need reproducible Python env + dependency pinning.
- **Decision:** uv, with deps pinned in `pyproject.toml` and locked in `uv.lock`
  (committed).
- **Rationale:** Fast, single-tool resolver + venv; lockfile gives reproducibility;
  the Dockerfile reuses the same lock with `uv sync --frozen`.
- **Revisit when:** A dependency cannot resolve under uv, or the team standardizes on
  another manager.

### Python 3.12 (not the system 3.13)
- **Context:** Host has 3.13; brief targets 3.12.
- **Decision:** Pin `requires-python = ">=3.12"`; uv provisions a 3.12 interpreter.
- **Rationale:** Match the brief and the eventual GPU/training stack; avoid 3.13-only
  surprises in ML libs.
- **Revisit when:** All downstream deps (xgboost, mlflow, etc.) certify 3.13.

### httpx over requests
- **Context:** HTTP client for Gamma + CLOB APIs.
- **Decision:** `httpx`.
- **Rationale:** Modern API, timeouts first-class, sync now with an async path
  available if the capture tick ever needs concurrency.
- **Revisit when:** Async capture is required and a different client fits better.

### DuckDB as the single datastore
- **Context:** Need storage for scraped fight data and captured order books.
- **Decision:** One DuckDB file (`data/ufc_edge.duckdb`, override via `DUCKDB_PATH`);
  all I/O through `src/ufc_edge/data/storage.py`.
- **Rationale:** Zero-server, columnar, great for analytical feature work later;
  trivially file-portable (we SFTP the Fly copy down for analysis).
- **Revisit when:** Concurrent multi-writer access is needed (DuckDB is single-writer
  — see the capture/scrape lock note below), or data outgrows a single-file workflow.

### Frozen Pydantic models, validate at the boundary
- **Context:** Structured external data (scraped rows, API responses, snapshots).
- **Decision:** Pydantic v2 models, `frozen=True`, validated on ingest; fail loudly.
- **Rationale:** Immutable-by-default per the style guide; schema drift surfaces as a
  loud error instead of silent corruption.
- **Revisit when:** A model genuinely needs mutation (prefer constructing a new
  instance first).

### MLflow local sqlite backend
- **Context:** Wire MLflow so a future session can `mlflow.start_run()` with zero
  setup; no experiments today.
- **Decision:** `configs/mlflow.yaml` points tracking at `sqlite:///mlruns/mlflow.db`
  with artifacts in `mlruns/artifacts`; `mlruns/` is gitignored.
- **Rationale:** No server to run; sqlite is enough until training starts.
- **Revisit when:** Multi-user tracking or a remote MLflow server is needed.

### DVC initialized; remote is a stub
- **Context:** Brief wants `data/` versioned, raw blobs not committed.
- **Decision:** `dvc init`; `dvc.yaml` declares scrape + cross-check stages. **No real
  remote configured yet.**
- **Rationale:** Local versioning works immediately; choosing a remote (S3/GDrive/etc.)
  is a separate cost/credentials decision.
- **Revisit when:** Data needs to be shared off this machine — **TODO: pick and
  configure a DVC remote.**

### Pre-commit hook set (no branch protection)
- **Context:** Brief requires ruff lint+format, end-of-file-fixer,
  trailing-whitespace, check-added-large-files, and a secrets-detection hook.
- **Decision:** Exactly that set + `check-merge-conflict`/`check-yaml`/`check-toml`.
  A `no-commit-to-branch --branch main` hook was trialed and **removed**.
- **Rationale:** This is a solo repo developed directly on `main`; blocking commits
  to `main` would break the day-to-day workflow with no PR process in place.
- **Revisit when:** A multi-contributor PR workflow is adopted — re-add branch
  protection then.

### Secrets handling
- **Context:** API keys/tokens must never be committed.
- **Decision:** `.env` (gitignored) + `.env.example` (key names only); detect-secrets
  pre-commit hook with a `.secrets.baseline`.
- **Rationale:** Standard, auditable; the hook is a backstop against accidental
  commits.
- **Revisit when:** Moving to a managed secrets store.

### .gitignore covers data blobs
- **Context:** A 3.3MB `ufc_edge.duckdb` was untracked but not ignored — `git add .`
  would have committed it.
- **Decision:** Ignore `*.duckdb`, `data/raw|interim|processed/`, `mlruns/`,
  `.scrapy/`.
- **Rationale:** DuckDB data is DVC-tracked, never a git blob; keeps history clean.
- **Revisit when:** Small fixture DBs need to be committed intentionally (use an
  explicit unignore for those paths).

---

## ufcstats scraper

### Politeness: download delay, concurrency, AutoThrottle, user-agent
- **Context:** ufcstats.com is a small site; we must not hammer it.
- **Decision (in `configs/scrape/default.yaml`):** `download_delay = 2.0s`,
  `concurrent_requests = 1`, AutoThrottle enabled with target concurrency `1.0`,
  descriptive user-agent identifying the bot as academic research.
- **Rationale:** Conservative, respectful crawl; AutoThrottle backs off further under
  latency. The crawl is a one-time backfill then incremental, so speed is not a
  priority.
- **Revisit when:** A full backfill is impractically slow *and* the site tolerates
  more — only loosen with evidence (HTTP 200s, no rate-limit responses).

### Resumable crawl via JOBDIR
- **Context:** A full crawl shouldn't restart from zero if interrupted.
- **Decision:** `make scrape` passes `JOBDIR=.scrapy/jobs/ufcstats`.
- **Rationale:** Scrapy persists request queue + dedupe state; restarts resume.
- **Revisit when:** Switching to incremental-only updates.

### ufcstats serves a JavaScript proof-of-work challenge
- **Context:** As of June 2026, ufcstats.com gates pages behind a "Checking your
  browser…" interstitial: a JS proof-of-work (find `n` where
  `sha256(nonce:n)` has K leading hex zeros), POST to `/__c`, then the server issues
  a `_fmc` clearance cookie. A plain Scrapy/BS4 GET returns the challenge, not data.
- **Decision:** Reproduce the handshake in Python (`ufcstats/challenge.py`): same
  computation the browser runs. A downloader middleware
  (`ChallengeSolverMiddleware`) detects the interstitial, solves it, caches the
  cookie, and retries the request.
- **Rationale:** It's a lightweight, legitimate PoW for publicly available data —
  no auth bypass. Difficulty is tiny (K=2 zeros, ~hundreds of hashes).
- **Revisit when:** ufcstats raises difficulty, changes the challenge, or the cookie
  TTL shortens — if solving becomes expensive, switch to a headless browser or lean
  on the Kaggle dataset for backfill.

### Challenge solve is synchronous inside the middleware
- **Context:** The solve does a blocking httpx GET+POST; Scrapy runs on Twisted.
- **Decision:** Accept the brief reactor block.
- **Rationale:** Crawl runs at concurrency 1 with a 2s delay and the challenge is
  rare (one cookie reused until expiry), so the block is negligible.
- **Revisit when:** Concurrency is raised, or challenges become frequent — move the
  solve into an async Request chain.

### sig_strike_breakdowns: round=0 sentinel for the full-fight aggregate
- **Context:** The breakdown table has both a full-fight row and per-round rows.
  `round` is part of the primary key, and DuckDB forbids NULL in a PK, so the
  natural `round=None` for "whole fight" cannot be stored.
- **Decision:** Use `round = 0` for the full-fight aggregate; `1..5` per round.
- **Rationale:** Keeps a clean composite PK `(fight_url, fighter_url, round)` and
  idempotent upserts; 0 is an unambiguous sentinel.
- **Revisit when:** Never expected to change; documented in the schema docstring.

### Scrapy 2.16 entry point: async start()
- **Context:** Scrapy 2.13+ replaced the generator `start_requests()` with
  `async def start()`. On 2.16, `start_requests` is silently not invoked.
- **Decision:** Use `async def start(self)` in the spider.
- **Revisit when:** N/A unless Scrapy changes the entry point again.

### Idempotent upserts keyed on stable URLs
- **Context:** Re-running the crawl must not duplicate rows.
- **Decision:** Primary keys are ufcstats URLs (`event_url`, `fight_url`,
  `fighter_url`); writes use `INSERT ... ON CONFLICT DO UPDATE`. Every row carries
  `scraped_at`.
- **Rationale:** URLs are stable IDs; upsert makes re-crawls safe and refreshes data.
- **Revisit when:** ufcstats changes its URL scheme.

---

## Polymarket capture cron

### 5-minute capture interval
- **Context:** Order-book history must be self-captured (Polymarket's historical
  `/orderbook-history` stopped updating ~Feb 2026); P0 time-critical.
- **Decision:** `CAPTURE_INTERVAL_SECONDS = 300` (`configs/capture/default.yaml`).
- **Rationale:** Dense enough to track line movement around fights; light enough to
  stay well within free-tier limits and finish a ~960-market tick (~90s) comfortably.
- **Revisit when:** We need finer resolution near fight time (consider a faster
  cadence in the final hours before an event).

### Market enumeration: Gamma `/events?tag_slug=ufc`, JSON-string fields
- **Context:** Need to find active UFC markets and their token IDs.
- **Decision:** Page Gamma `/events` filtered by `tag_slug=ufc`, `closed=false`; per
  market, `json.loads` the `clobTokenIds` and `outcomes` fields (Gamma returns them as
  JSON-encoded strings inside the JSON payload) and emit one row per token.
- **Rationale:** Tag filter scopes to UFC; the string-decoding is a documented Gamma
  quirk that silently yields zero tokens if not handled.
- **Revisit when:** Gamma changes its schema, or `tag_slug=ufc` misses relevant
  markets (observed: it currently also returns UFC-adjacent prop markets, which is
  acceptable — they are still UFC-tagged).

### Order-book depth = 10 levels per side
- **Context:** How much of each book to store.
- **Decision:** `orderbook_depth = 10`; keep best 10 bids (highest) and asks (lowest);
  compute `mid_price` and `spread` from the best level each side.
- **Rationale:** Enough for slippage modeling later without bloating storage.
- **Revisit when:** Slippage modeling needs full-depth books.

### Robustness: retries, per-market isolation, idempotent snapshot writes
- **Context:** Network is flaky; one bad market must not kill a tick.
- **Decision:** `tenacity` exponential backoff (`MAX_REQUEST_ATTEMPTS = 3`); each
  market wrapped in try/except inside the tick; snapshot writes use `ON CONFLICT
  (token_id, captured_at) DO NOTHING`; every tick logs `markets_seen /
  snapshots_written / errors`.
- **Rationale:** Self-healing cron; safe to restart; observable per tick.
- **Revisit when:** Error rates rise (would indicate an API change worth handling
  explicitly).

### Per-snapshot `captured_at` (no shared tick id) — known limitation
- **Context:** During verification, grouping by `captured_at` produced one row per
  snapshot, not per tick.
- **Decision (current):** Each snapshot is stamped with `datetime.now(UTC)` at the
  moment its book is fetched, so one tick's ~960 rows spread across ~90 seconds.
  Accepted as-is for now; "group by tick" is done by time-bucketing.
- **Rationale:** Per-market timestamps are strictly more precise; no data lost.
- **Revisit when:** Tick-level analysis becomes common — **TODO: add a `tick_id`
  (one UUID/timestamp per tick) column** so a tick is groupable without bucketing.

### Host: Fly.io + persistent volume + always-restart
- **Context:** Need a free always-on host; cron must survive restarts.
- **Decision:** Fly.io app `ufc-edge-capture` (region sjc, shared-cpu-1x/512MB, single
  machine), DuckDB on a 1GB persistent volume mounted at `/data`, `restart = always`.
- **Rationale:** Simple container deploy; volume persists snapshots across reboots
  (verified: data survived a machine restart). Oracle Cloud free tier was the
  documented fallback, not needed.
- **Revisit when:** Snapshot volume approaches 1GB, or we need multi-region capture.

### Fly.io trial requires a card to run >5 min
- **Context:** First deploy ran two clean ticks, then Fly force-stopped the machine:
  *"Trial machine stopping. To run for longer than 5m0s, add a credit card."*
- **Decision:** Added a payment card to the Fly account (free allowances still apply
  to this tiny worker; card is anti-abuse identity).
- **Rationale:** Least-effort path to genuine always-on; app was already deployed.
  Verified afterward: ~8 hours of unbroken 5-minute ticks, ~93k snapshots.
- **Revisit when:** Fly billing changes, or cost monitoring shows spend beyond the
  free allowance.

---

## Feature decisions (registry detail in FEATURES.md)

### Layer separation: behavioral & market features excluded from XGBoost
- **Decision:** Behavioral features (Section 13) and market-derived features
  (Section 16) are **strategy-layer only**, never model inputs.
- **Rationale:** Behavioral coverage is sparse/qualitative and would overfit or
  flatline in CV; using the market line to predict the market line is circular.
- **Revisit when:** Behavioral coverage becomes dense and survives CV as a model
  feature (would require a deliberate human decision to move it).
- **Extended by the 2026-06-21 redesign (D6/D6b):** §16 market features are now
  formally relocated to the strategy layer (convergence-trade logic + slippage
  model), and §7 (Performance vs Expectation) is **dropped entirely** — it fed
  *historical* closing odds into the model, which passes the temporal-leakage test
  but fails the economic-circularity test. The model is now structurally odds-free.
  See [DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md) D6/D6b.

### Elo configuration
- **Decision:** Variable K (method bonus + recency); decay toward mean (1500) during
  inactivity; injury stoppages Elo-neutral (K=0); DQ outcomes K × 0.1.
- **Rationale:** Reflects that not all wins/losses carry equal information and that
  layoffs erode rating confidence.
- **Revisit when:** Backtests show miscalibrated Elo movement; tune K schedule.

### Behavioral discount architecture
- **Decision:** Behavioral signals adjust the model's output probability in the
  strategy layer, gated on `profile_confidence ≥ 0.65`.
- **Rationale:** Keeps qualitative, low-coverage signals out of the gradient path
  while still letting them inform sizing/decisions.
- **Revisit when:** Behavioral eval coverage/quality changes.

### LLM provider conflict — RESOLVED (2026-06-21, redesign D7)
- **Context:** The project brief specified a local Llama 3.1 70B offline pipeline; the
  human's FEATURES.md draft specified a Gemini 2.5 Flash grounded-search pipeline.
  Previously flagged `TODO(human)`, unresolved.
- **Decision:** **Resolved on the architecture axis.** LLM extraction is an
  **offline, cached enrichment step** that emits structured columns ahead of time and
  is **never in the inference path** — i.e. the *offline-cached* approach wins and the
  *grounded-search-at-inference* approach is rejected. Leakage is handled separately
  by a `source_published_at < fight_date` gate, not by being offline. See
  [DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md) D7.
- **Rationale:** Moving the LLM offline gives reproducibility/robustness (no flaky
  model in the prediction path); grounded-search-at-inference would reintroduce both.
  The redesign approves an offline LLM role without re-litigating it.
- **Still open (human-owned, not blocking):** the *specific* offline model (e.g. the
  brief's local Llama 3.1 70B vs. another offline/batch model) is a low-level detail
  left to Step 2 — the architecture only locks "offline + cached + structured columns,
  not grounded-search."
- **Revisit when:** The human picks the concrete offline model, or eval shows the
  offline enrichment can't meet the FEATURES.md §15 quality thresholds.

---

## Abandoned features (mirrored in FEATURES.md §17)

| Feature | Reason |
|---|---|
| Retirement signal | LLM extraction too unreliable; fighters retire and return; false-positive rate unacceptable. |
| Judges' scorecards pre-fight | Not available as-of fight date. |
| Post-fight medical suspensions | Available only after the outcome — hard leakage. |
| Raw win % without context | Superseded by Elo and opponent-adjusted versions. |
| Height differential as standalone | Reach captures it better and more specifically. |
| Fighter nationality as standalone | Captured via `home_advantage_delta` + `regional_circuit_quality_tier`. |
| Social-media follower count | Correlated with name recognition already in odds; no independent signal. |
| Performance vs Expectation (old §7) | Dropped 2026-06-21 (redesign D6b): fed historical closing odds into the model; fails the economic-circularity test and muddies model-edge vs execution-edge attribution. Model is now structurally odds-free. |
