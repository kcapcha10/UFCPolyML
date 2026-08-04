# V1 requirements

## Product goal

Produce calibrated UFC win probabilities from pre-fight UFCStats data and compare
them with a timestamped Polymarket implied probability. The deliverable is a batch
report and paper-signal log, not a betting or execution system.

## Data and feature requirements

- Training, calibration, and evaluation rows use fights on or after the configured
  `label_start_date`; feature state may use all earlier available UFC history.
- Validation must quarantine impossible or malformed rows with a reason code. It
  must fail for modern-era or undated violations and must never mutate source data.
- For each fight, feature values must be emitted before that fight updates any
  accumulator. Replaying after removing the fight and later data must reproduce its
  feature row exactly.
- Features are documented in [FEATURES.md](FEATURES.md), use no market data, and
  preserve missing values as `NULL`/`NaN`.

## Modeling and evaluation requirements

- Use XGBoost as the only v1 predictive model. Build mirrored fighter orientations
  for training; a reversed inference request must return exactly `1 - p`.
- Split chronologically by event, never randomly by fight. All fights from an event
  stay in one split, and calibration data is disjoint from fit and test data.
- Report per-fold and pooled Brier score, log loss, expected calibration error,
  reliability curves, and event-level bootstrap confidence intervals.
- Reserve every event from January through August 2026 as the final untouched
  holdout. Evaluate it once after August has completed; use no result from it to
  select features, hyperparameters, or calibration choices.
- Every reported run records its data revision, feature version, resolved config,
  random seeds, package versions, and XGBoost model/calibrator artifacts in MLflow.
- Retrain once after each completed UFC event, only after ingestion and validation
  have completed successfully.
- Provide one structured health command reporting data freshness, validation and
  quarantine status, market-capture freshness, latest MLflow run, and active
  model/feature versions.

## Report requirements

- Read scheduled bouts from a versioned upcoming-fights input and match them to
  Polymarket deterministically. Unmatched or ambiguous cases must be visible rather
  than silently dropped or guessed.
- Select the latest capture at or before a supplied `as_of_timestamp` and persist
  model probability, market implied probability, signed mismatch, identifiers, and
  provenance in the report.
- Paper signals are observational records. The system must not place orders, size
  positions, simulate fills, or report investment returns.

## Acceptance bar

`make validate`, `make lint`, and `make test` must pass. The final run must be
repeatable from versioned data and configuration, and its held-out metrics must be
reported without using the final test period for model selection.
