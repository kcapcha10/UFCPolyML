.PHONY: setup lint format test scrape capture backup validate

# ── Bootstrap ────────────────────────────────────────────────────────────────────

setup:
	uv sync --extra dev
	uv run pre-commit install
	@if [ ! -f .dvc/.gitignore ]; then uv run dvc init; fi
	@echo "✓ Environment ready. Copy .env.example → .env and fill in secrets."

# ── Code quality ─────────────────────────────────────────────────────────────────

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

# ── Tests ────────────────────────────────────────────────────────────────────────

test:
	uv run pytest

# ── Data pipeline ────────────────────────────────────────────────────────────────

scrape:
	uv run scrapy crawl ufcstats -s LOG_LEVEL=INFO \
	    --set JOBDIR=.scrapy/jobs/ufcstats

validate:
	uv run python -m ufc_edge.data.validation.runner

# ── Market capture ───────────────────────────────────────────────────────────────

capture:
	uv run python -m ufc_edge.data.polymarket.capture --once

# ── Backup ───────────────────────────────────────────────────────────────────────
# Pull the irreplaceable capture history off the Fly volume and push it to the
# DVC remote. Run after every event weekend at minimum — capture gaps are permanent.

backup:
	flyctl ssh sftp get /data/ufc_edge.duckdb data/raw/capture_remote.duckdb --app ufc-edge-capture
	uv run dvc add data/raw/capture_remote.duckdb
	uv run dvc push
