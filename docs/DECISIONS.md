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

## Market comparison — D24 (open implementation details)

The market is an external comparator, never a training input. A report must retain
the exact snapshot timestamp, matched fighter identities, model probability, market
implied probability, signed difference, and model/data/config versions. A paper
signal is a record of that report, not a wager or a performance claim.

## Open decisions

1. Define the lean development-tuning versus final holdout protocol.
2. Define market-to-fight matching, snapshot selection, and paper-signal persistence.
3. Define batch retraining cadence, model-artifact contract, and health summary.

## Operational choices

- DuckDB is the single datastore; Hydra configures runs; DVC versions data; MLflow
  records experiment provenance.
- The capture cron runs every five minutes on Fly with a persistent volume. Its
  history is irreplaceable; back it up through DVC/Google Drive.
