.PHONY: setup lint format test scrape capture

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

# ── Market capture ───────────────────────────────────────────────────────────────

capture:
	uv run python -m ufc_edge.market.capture --once
