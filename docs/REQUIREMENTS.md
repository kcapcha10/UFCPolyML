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
- Every reported run records its data revision, feature version, resolved config,
  random seeds, package versions, and model artifact.

## Report requirements

- Match a scheduled UFC bout to a Polymarket market deterministically; unmatched or
  ambiguous cases must be visible rather than silently dropped.
- Select a documented as-of snapshot and persist model probability, market implied
  probability, signed mismatch, identifiers, and provenance in the report.
- Paper signals are observational records. The system must not place orders, size
  positions, simulate fills, or report investment returns.

## Acceptance bar

`make validate`, `make lint`, and `make test` must pass. The final run must be
repeatable from versioned data and configuration, and its held-out metrics must be
reported without using the final test period for model selection.
