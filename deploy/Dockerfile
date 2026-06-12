# Capture-cron image: runs the Polymarket order-book capture loop.
# Build context is the repo root (see README "Deploying the capture cron").
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first so they cache independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DUCKDB_PATH=/data/ufc_edge.duckdb

CMD ["python", "-m", "ufc_edge.market.capture"]
