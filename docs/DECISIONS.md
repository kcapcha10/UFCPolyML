# Active technical decisions

This is the single decision record for v1. It holds the choices worth being able
to explain in an interview: what was chosen, why, and the constraint it creates.

## Scope — D23

Build a batch XGBoost MLE system: validated UFC data → point-in-time features →
calibrated probabilities → timestamped market-mismatch report and paper-signal log.
This makes feature engineering, temporal validation, calibration, reproducibility,
and a useful output the portfolio story. Keep modest production polish—config-driven
commands, MLflow provenance, versioned artifacts, and health reporting.

Deep learning, LLM enrichment, automated trading, sequential strategy research,
CLOB simulation, and online learning are deliberately outside v1. They each add a
new evaluation surface without improving the core XGBoost project.

## Data integrity — D8, D11, D12

- Kaggle is not a source of truth. Internal consistency checks and quarantine are.
- Feature history uses all available UFC history; labels/train/eval default to fights
  on or after `label_start_date` (`2010-01-01`).
- Impossible rows are quarantined with reason codes. Modern-era or undated
  violations fail validation; older violations remain visible but non-blocking.

## Feature computation — D13, D14, D15

- Replay fights chronologically. Emit every feature before applying that fight's
  outcome to any state component.
- State is split by feature family; all emitters read frozen pre-fight state.
- Output is a versioned, wide `features_v{N}` table keyed by fight and fighter.
  A source-hash guard makes stale feature code loud. `NULL → NaN` is the explicit
  model missingness contract.

## Prediction and evaluation — D10, D16, D19–D22

- Split by event in chronological order; never split a card across train,
  calibration, or test. Report Brier, log loss, ECE, reliability curves, and
  event-level bootstrap intervals.
- Build both fighter orientations for training. At inference, average raw
  orientations, calibrate a canonical fighter ordering once, and return `1 - p`
  for the reversed order.
- XGBoost uses fixed boosting rounds per candidate—no early-stopping double use.
  If random search remains in the lean protocol, its active eight-axis surface is
  `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
  `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
- Reserve a trailing event-complete calibration slice with at least
  `max(250, ceil(20% × N))` unique fights. Use Platt below 1,000 unique calibration
  fights and isotonic at or above that threshold.

## Lean tuning and final evaluation — D25

- The final untouched holdout is every UFC event from **January through August
  2026**. It is evaluated once only after August is complete.
- Development uses three to four earlier chronological, event-grouped folds and a
  small fixed set of roughly 12 XGBoost configurations. Choose by mean calibrated
  Brier score; inspect log loss and ECE as supporting checks.
- Once the configuration is locked, do not make further modeling choices from the
  2026 holdout. Report its metrics with event-level bootstrap intervals.

## Market comparison — D24

The market is an external comparator, never a training input. V1 uses a small,
versioned upcoming-fights input (event date, fighter names, weight class) and
automatically matches only an unambiguous normalized Polymarket UFC market.
Unmatched or ambiguous cases are visible for manual review; the system never guesses.

A report uses the most recent capture at or before a supplied `as_of_timestamp` and
retains the snapshot timestamp, matched identities, model probability, market implied
probability, signed difference, and model/data/config versions. A paper signal is a
record of that report, not a wager or a performance claim.

## Retraining cadence — D26

Run one batch retrain after every completed UFC event, only after ingestion and
validation complete cleanly. This keeps the model current while remaining a simple,
auditable batch workflow rather than an online-learning system.

## Model artifacts — D27

MLflow is the sole v1 artifact store. Every training run logs its XGBoost model,
calibrator, feature schema and version, resolved config, metrics, data revision, and
run ID there. The project does not maintain a parallel on-disk model-bundle format.

## Health summary — D28

A single CLI health command emits a structured summary: newest UFC event and data
timestamp, validation result and quarantine census, latest market-capture timestamp,
last successful MLflow run, and active model/feature versions. No dashboard belongs
in v1.

## Operational choices

- DuckDB is the single datastore; Hydra configures runs; DVC versions data; MLflow
  records experiment provenance.
- The capture cron runs every five minutes on Fly with a persistent volume. Its
  history is irreplaceable; back it up through DVC/Google Drive.
