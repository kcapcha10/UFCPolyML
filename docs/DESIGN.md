# V1 design

## Architecture

```text
UFCStats + Polymarket snapshots
             ↓
DuckDB → validation/quarantine → point-in-time feature replay
                                      ↓
                         XGBoost → calibration → temporal evaluation
                                                        ↓
                                  mismatch report + paper-signal log
```

DuckDB is the system of record. DVC versions data, Hydra resolves configuration,
and MLflow records experiments. The deployed Fly process only captures market
snapshots; training and reporting run as explicit batch commands.

## Data contract

The feature replay reads full UFC history in date order, skips quarantined rows,
and has two phases for every fight: (1) emit both fighters' frozen pre-fight state;
(2) update state from the outcome. Feature families are isolated state components;
the registry rejects duplicate feature names. The output is a versioned wide feature
table keyed by fight and fighter, with labels joined downstream.

## Model contract

The matrix assembler creates both fighter orientations. XGBoost receives tabular,
odds-free inputs and produces raw scores. A calibrator is fitted only on a trailing,
event-complete calibration slice inside each training window. Inference canonicalizes
fighter order, calibrates once, and complements the probability for the reverse order.

## Evaluation contract

`WalkForwardSplitter` is the sole split implementation. It creates deterministic,
expanding event-grouped folds. Metrics are calculated per fold and pooled; confidence
intervals resample events, not individual fights. Development tuning cannot inspect
the final held-out event block.

## Report contract

Market matching uses normalized fighter/event identity plus explicit ambiguity
statuses. A report selects a configured as-of snapshot and writes one durable record
per matched bout containing the two probabilities, mismatch, timestamps, and full
model/data/config provenance. The report is an analysis artifact only.

## Package layout

| Package | Responsibility |
|---|---|
| `data/` | UFCStats ingestion, DuckDB storage, validation, Polymarket capture |
| `features/` | chronological state replay and feature materialization |
| `model/` | matrix assembly, XGBoost training, calibration, inference artifacts |
| `eval/` | temporal folds, metrics, bootstrap intervals, MLflow logging |
| `report/` | market matching, snapshot selection, mismatch and paper-signal records |

`eval/` and `report/` are created when their implementation tasks begin. There are
no `reps`, `strategy`, `sim`, `online`, or enrichment packages in v1.
