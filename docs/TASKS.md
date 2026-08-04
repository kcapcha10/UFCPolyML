# V1 build plan

The data foundation is complete. Build the remaining vertical slice in this order;
each step has a usable testable result before the next begins.

## Completed

- [x] UFCStats scraper, Pydantic ingest models, DuckDB storage, and fixture tests
- [x] Data validation, quarantine, era-aware failure rule, and `make validate`
- [x] Five-minute Polymarket capture cron with persistent Fly storage
- [x] Scope and repository reset to the XGBoost batch workflow

## Next: evaluation and features

- [ ] Implement deterministic event-grouped walk-forward folds and property tests.
- [ ] Implement Brier, log loss, ECE, reliability curves, event-bootstrap intervals,
  and MLflow provenance logging.
- [ ] Implement chronological replay, component registry, feature table, source-hash
  version guard, and the deletion-oracle/determinism tests.
- [ ] Materialize the initial feature families from [FEATURES.md](FEATURES.md) only.

## Then: XGBoost model

- [ ] Add the direct XGBoost and scikit-learn dependencies with a locked environment.
- [ ] Build mirrored matrix assembly, fixed-round XGBoost training, in-fold Platt or
  isotonic calibration, and order-symmetric inference tests.
- [ ] Define the lean development-tuning protocol and freeze a final out-of-time
  event block before comparing candidate configurations.
- [ ] Produce the first reproducible held-out evaluation report.

## Finish: report product and polish

- [ ] Define market-to-fight matching and as-of snapshot selection.
- [ ] Write a versioned mismatch report and paper-signal log with ambiguity states.
- [ ] Add batch train, predict, and report commands plus data/model health summaries.
- [ ] Run the end-to-end workflow, publish held-out metrics, and document the result.

## Definition of done

A new machine can install the locked environment, reproduce a held-out evaluation,
and generate a timestamped paper report from versioned data. The README describes
only implemented capabilities; no result is presented as a trading return.
