# UFCPolyML

An end-to-end applied machine-learning system that builds point-in-time UFC
features, trains and calibrates a strictly odds-free XGBoost win-probability
model, and evaluates it against Polymarket prediction-market prices. The
deliverable is a timestamped paper-signal report and accumulating evidence log,
never live trading or profitability claims.

## Methodology highlights

- **Temporal integrity.** Event-grouped chronological splits with a 4-layer
  anti-leakage architecture: features emitted before outcome application, disjoint
  calibration slice, held-out 2026 evaluation window scored once, and property-based
  leakage tests over the full replay.
- **Calibrator selection.** Per-fold empirical selection among Platt scaling,
  isotonic regression, and beta calibration (3-parameter generalization of Platt),
  chosen by ECE on the fold's evaluation slice. Calibration quality reported
  stratified by probability bucket with binomial confidence intervals.
- **Honest evaluation.** Headline metric is Brier skill score vs. the market
  (`1 - Brier_model / Brier_market`) with paired permutation testing and a computed
  minimum detectable effect. The project reports what its evaluation can and cannot
  conclude at its sample size.
- **Feature-family ablation ladder.** Nested models (naive floor, record-only,
  +physical, +schedule-strength, +domain interactions) testing whether domain
  expertise adds measurable incremental value, with confidence intervals per rung.
- **LLM as evaluated due-diligence.** A search-augmented LLM runs a fixed pre-fight
  checklist on flagged signals only, returning a structured CONFIRM/QUALIFY/VETO
  verdict. Annotate-only: never in the gradient path, never auto-suppressing
  signals. Carries its own precision/recall eval harness.

## Architecture

```
UFCStats scrape ──┐                                    ┌── Mismatch report
                  ▼                                    │   (magnitude gate +
              DuckDB ──► Validation / ──► Point-in-time│    bucket transparency)
                  ▲      Quarantine       Feature      │
Polymarket    ────┘                       Replay       ├── Paper-signal log
capture cron                                │          │   (bid/ask/depth +
(5-min, Fly)                                ▼          │    post-signal snapshots)
                                     XGBoost + per-fold│
                                     calibration       ├── LLM due-diligence
                                            │          │   (annotate-only)
                                            ▼          │
                                     Evaluation harness┘
                                     (Brier skill, MDE,
                                      permutation test,
                                      ablation ladder)
```

## Status

| Layer | Status |
|---|---|
| Data ingestion, DuckDB storage, validation | Built |
| Polymarket capture cron (5-min, Fly) | Built and deployed |
| Feature engine | Fully specified; implementation starting (Wave 0) |
| Model training, calibration, evaluation | Fully specified; implementation starting (Wave 0) |
| Mismatch report and paper-signal log | Fully specified; implementation starting (Wave 0) |

Specifications live under `.kiro/specs/` and are being built via spec-driven
parallel execution.

## Quickstart

```bash
make setup
cp .env.example .env
make validate
make test
make capture
```

Data is stored in `data/ufc_edge.duckdb` by default. Override with `DUCKDB_PATH`.

## Repository map

| Path | Purpose |
|---|---|
| `src/ufc_edge/data/` | UFCStats ingestion, DuckDB storage, validation, Polymarket capture |
| `src/ufc_edge/features/` | Point-in-time tabular feature engine |
| `src/ufc_edge/model/` | XGBoost training and calibrated prediction |
| `src/ufc_edge/eval/` | Temporal evaluation, calibration, metrics, MLflow provenance |
| `src/ufc_edge/report/` | Mismatch report, paper-signal log, LLM due-diligence |
| `.kiro/specs/` | Requirements, design, task specs, decisions, feature registry |
| `.kiro/skills/` | Code conventions and agent skill definitions |
| `configs/` | Hydra configuration (data, model, eval, report, capture) |
| `deploy/` | Fly capture-cron deployment |
| `tests/` | Fixture-based test suites (no live-site dependencies) |

## Documentation

- [.kiro/specs/DECISIONS.md](.kiro/specs/DECISIONS.md) — key decisions and rationale
- [.kiro/specs/FEATURES.md](.kiro/specs/FEATURES.md) — feature registry (human-owned)
- [.kiro/specs/feature-engine/](.kiro/specs/feature-engine/) — feature replay engine spec
- [.kiro/specs/model-and-eval/](.kiro/specs/model-and-eval/) — XGBoost training and evaluation spec
- [.kiro/specs/mismatch-report/](.kiro/specs/mismatch-report/) — mismatch report and paper-signal log spec
- [AGENTS.md](AGENTS.md) — contributor and coding-agent rules

## Capture backup

The capture cron collects market snapshots that Polymarket does not provide
historically. Keep it running and configure the DVC Google Drive remote before
relying on it as the only copy. `make backup` pulls the Fly DuckDB and pushes the
versioned snapshot to DVC.
