# UFCPolyML

An end-to-end applied machine-learning project for UFC prediction markets.

The system builds point-in-time UFC features, trains and calibrates an XGBoost
win-probability model, evaluates it on future events, and produces a timestamped
paper-signal report that compares model probability with the current Polymarket
implied probability. It never places orders or claims trading returns.

## Status

The data foundation is built: UFCStats scraping, DuckDB storage, validation and
quarantine, and a five-minute Polymarket capture cron. The feature, evaluation,
and model layers are next; see [docs/TASKS.md](docs/TASKS.md).

## Quickstart

```bash
make setup
cp .env.example .env
make validate
make test
make capture
```

Data is stored in `data/ufc_edge.duckdb` by default. Override with `DUCKDB_PATH`.

## Documentation

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — testable product and correctness contract.
- [docs/DESIGN.md](docs/DESIGN.md) — implementation architecture and interfaces.
- [docs/TASKS.md](docs/TASKS.md) — ordered build plan and current status.
- [docs/DECISIONS.md](docs/DECISIONS.md) — key decisions and their rationale.
- [docs/FEATURES.md](docs/FEATURES.md) — human-owned, in-scope feature registry.
- [CLAUDE.md](CLAUDE.md) — contributor rules and repository conventions.

## Repository map

| Path | Purpose |
|---|---|
| `src/ufc_edge/data/` | UFCStats ingestion, DuckDB storage, validation, Polymarket capture |
| `src/ufc_edge/features/` | Point-in-time tabular feature engine |
| `src/ufc_edge/model/` | XGBoost training and calibrated prediction |
| `src/ufc_edge/eval/` | Temporal evaluation, calibration, metrics, MLflow provenance |
| `docs/` | Project plan, decisions, and feature registry |
| `configs/` | Hydra configuration |
| `deploy/` | Fly capture-cron deployment |

## Capture backup

The capture cron collects market snapshots that Polymarket does not provide
historically. Keep it running and configure the DVC Google Drive remote before
relying on it as the only copy. `make backup` pulls the Fly DuckDB and pushes the
versioned snapshot to DVC.
