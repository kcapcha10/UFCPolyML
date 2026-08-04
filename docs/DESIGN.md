# UFCPolyML v1 — low-level design

**Status:** approved for implementation
**Audience:** engineers implementing or reviewing the v1 batch workflow
**Scope:** validated UFC data → point-in-time features → XGBoost probabilities →
market-mismatch report and paper-signal log

This document defines implementation contracts. Product requirements live in
[REQUIREMENTS.md](REQUIREMENTS.md); rationale and locked choices live in
[DECISIONS.md](DECISIONS.md); the allowed feature families live in
[FEATURES.md](FEATURES.md).

## 1. System boundaries

```text
UFCStats                     Polymarket
   │                              │
   ├── scrape → DuckDB             ├── five-minute capture → DuckDB
   │             │                 │
   │             └── validation/quarantine
   │                         │
   └──────────────────────► feature replay
                                      │
                                XGBoost + calibration
                                      │
                         temporal evaluation / MLflow
                                      │
                    report + persistent paper-signal record
```

### In scope

- Batch commands for feature generation, training, evaluation, prediction,
  reporting, and health checks.
- DuckDB as the operational datastore; DVC as the data-versioning mechanism;
  Hydra for resolved configuration; MLflow for experiment and model artifacts.
- The existing UFCStats scraper, validation/quarantine suite, and Polymarket
  capture cron.

### Explicitly out of scope

- Deep learning, embeddings, LLM enrichment, alternate data sources, automated
  trading, position sizing, order execution, CLOB replay, online learning, APIs,
  and dashboards.
- Any odds-derived feature or target. Polymarket data is a post-model comparator.

## 2. Package and command boundaries

The existing `data/` package remains the only owner of external ingestion and
DuckDB DDL. New packages are created only when their first task begins.

| Package | Public responsibility | Internal modules to add |
|---|---|---|
| `data/` | scrape, capture, storage, validation | existing packages only |
| `features/` | replay historical state and materialize fighter rows | `contracts`, `registry`, `replay`, `components`, `storage`, `versioning` |
| `eval/` | create legal folds and calculate evaluation results | `schemas`, `splits`, `calibration`, `metrics`, `provenance` |
| `model/` | assemble matrices, train XGBoost, load/predict a run | `schemas`, `matrix`, `train`, `inference` |
| `report/` | validate upcoming bouts, match markets, select snapshots, persist reports | `schemas`, `upcoming`, `matching`, `snapshots`, `storage`, `runner` |
| `ops/` | report pipeline health | `health` |

Each package exposes typed domain objects, not raw DuckDB rows. SQL is isolated in
that package's `storage` module. No package may query another package's table without
using its documented read function.

The Makefile will wrap the eventual module entry points. The intended command surface
is shown below; it is an implementation target, not a promise that commands exist
today.

```text
make validate                         # existing: validate and refresh quarantine
make features                         # materialize the configured feature version
make train                            # training + development evaluation + MLflow run
make evaluate                         # evaluate a named MLflow run on a legal split
make report AS_OF=<ISO-8601 timestamp> RUN_ID=<mlflow-run-id>
make health                           # structured JSON health summary
```

## 3. Source data and ownership

### 3.1 Existing DuckDB tables

| Table | Owner | Key | Use in v1 |
|---|---|---|---|
| `events` | `data.ufcstats` | `event_url` | event date, name, location |
| `fighters` | `data.ufcstats` | `fighter_url` | physical profile and identity |
| `fights` | `data.ufcstats` | `fight_url` | participants, result, method, class |
| `fight_totals` | `data.ufcstats` | `(fight_url, fighter_url)` | historical aggregate performance |
| `round_stats` | `data.ufcstats` | `(fight_url, fighter_url, round)` | optional historical round aggregates |
| `sig_strike_breakdowns` | `data.ufcstats` | `(fight_url, fighter_url, round)` | historical location/distance aggregates |
| `validation_quarantine` | `data.validation` | `(table_name, row_key, reason_code)` | exclusions and health reporting |
| `order_book_snapshots` | `data.polymarket` | `(token_id, captured_at)` | captured market price and matching metadata |

All source rows are immutable inputs to features. The validation runner refreshes
`validation_quarantine`; it never corrects or deletes source data.

### 3.2 New persisted tables

The following tables are created idempotently by their owning package. Exact feature
columns are derived from the registry; all other columns are stable contracts.

| Table | Owner | Required columns | Key / write behavior |
|---|---|---|---|
| `features_v{N}` | `features.storage` | `fight_url`, `event_url`, `event_date`, `fighter_url`, `opponent_url`, `weight_class`, `feature_version`, feature columns, `generated_at` | `(fight_url, fighter_url)`; replace atomically for one feature version |
| `report_runs` | `report.storage` | `report_run_id`, `as_of_timestamp`, `mlflow_run_id`, data and feature versions, `created_at` | immutable run record |
| `paper_signals` | `report.storage` | `signal_id`, `report_run_id`, bout identity, `market_id`, `token_id`, `snapshot_timestamp`, model and market probabilities, mismatch, match status | immutable per report/bout record |

`features_v{N}` contains no label. The matrix assembler joins `fights.winner_url`
only after feature materialization. A training row is eligible only if the outcome is
binary and the winner is one of the recorded participants; draws and no-contests are
excluded with a counted reason.

### 3.3 Provenance fields

Every materialized feature, evaluation, model, and report run carries:

- `data_revision`: DVC data revision or an equivalent immutable source snapshot ID;
- `feature_version` and the feature-package source hash;
- resolved Hydra configuration and random seed(s);
- package versions; and
- the MLflow run ID when a model is involved.

The value must be captured at run start and written with the result. A later query of
the live database is not valid provenance for a past result.

## 4. Feature replay design

### 4.1 Read model and ordering

The replay input is a typed `HistoricalFight` view assembled from the UFCStats tables.
It includes fight, event, both fighters, outcome, and only historical statistics.
Rows in `validation_quarantine` are excluded through an anti-join before replay.

Feature state uses all available completed UFC history. The `label_start_date` only
filters rows later used for train/calibration/evaluation; it never truncates state.

The replay unit is an **event**, ordered by `(event_date, event_url)`. Treating every
event as an atomic tick prevents a fight later on the same card from reading an
earlier result whose time is unavailable in the source data.

For event `E`:

1. Load all fights in `E` and sort them stably by `fight_url` for deterministic output.
2. Freeze the state view for every fighter and global component.
3. Emit rows for every fight and both fighter orientations using only that frozen state.
4. Apply every resolved outcome in `E` to state components.

No emitter may mutate state. No component may read a fight in the current or a later
event while emitting features.

### 4.2 Component and registry contracts

```python
class StateComponent(Protocol):
    def update(self, fight: FightOutcomeView) -> None: ...

class FeatureEmitter(Protocol):
    name: str
    def emit(self, context: EmitContext) -> dict[str, float | str | None]: ...
```

`EmitContext` exposes a read-only view of the focal fighter, opponent, event metadata,
and registered global state. It does not expose labels, market data, raw DuckDB
connections, or mutable component state.

The registry owns the output schema and fails process startup if:

- an emitter name or feature column is duplicated;
- an emitter declares a value outside the supported scalar types; or
- a configured feature family is absent from the registry.

Feature families are implemented only from [FEATURES.md](FEATURES.md). A new feature
requires a registry entry, deterministic definition, version bump, source-hash update,
and tests; it is never added as an ad hoc model-column transform.

### 4.3 Feature versioning and writes

`FEATURE_VERSION` is a human-readable, incremented identifier. A committed manifest
maps it to a hash of the feature package. A test fails when implementation source and
manifest disagree.

Feature generation writes to a staging table, validates row count and uniqueness, then
swaps the target `features_v{N}` table in one transaction. A failed generation leaves
the previously successful table unchanged. Missing source values remain DuckDB `NULL`
and become `NaN` only at matrix assembly.

### 4.4 Required feature tests

- **Deletion oracle:** deleting fight `X` and all later source data leaves `X`'s
  feature row byte-identical to its full-replay row.
- **Same-card isolation:** no row for an event changes when another fight on that
  event is removed or its result changes.
- **Determinism:** identical input and config generate identical rows and schema.
- **Registry safety:** duplicate columns, unknown families, and stale version hashes
  fail loudly.

## 5. Dataset, model, and calibration design

### 5.1 Matrix assembly

`model.matrix` transforms per-fighter rows into fight examples. It joins the two rows
for a fight, constructs only the matchup deltas/interactions declared in the feature
registry, and creates two training orientations:

```text
(fighter A features − fighter B features, label = A won)
(fighter B features − fighter A features, label = B won)
```

The assembler filters to the label universe and records exclusion counts for missing
rows, invalid results, and schema mismatches. It rejects an input matrix containing
market-derived columns or a feature schema different from the requested version.

### 5.2 XGBoost training

`model.train` is the only owner of XGBoost construction. Each candidate uses a fixed
number of boosting rounds; early stopping is not used. The allowed parameter surface
is `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
`colsample_bytree`, `reg_alpha`, and `reg_lambda`.

Development evaluates a small fixed set of about 12 candidates. The selected candidate
minimizes mean **calibrated Brier score** across development folds. Log loss and ECE
are reported as checks; they do not silently override the selection rule.

### 5.3 Symmetric inference

For a fight `(A, B)`, inference creates both legal orientations. Let `p_ab` be the
raw XGBoost probability for A over B and `p_ba` the raw probability for B over A.
The canonical raw probability is:

```text
p_raw = 0.5 × (p_ab + (1 − p_ba))
```

Canonical order is the lexicographically smaller canonical fighter URL first. The
calibrator is applied once to `p_raw`; if the requested order is reversed, return
`1 − p_calibrated`. This guarantees exact order symmetry and avoids fitting separate
calibrators for each orientation.

### 5.4 Calibration

For every fold, reserve a trailing event-complete calibration partition inside the
training window with at least `max(250, ceil(20% × N))` unique fights. Fit XGBoost on
the earlier training partition only. Create one canonical, symmetric raw prediction
per calibration fight and fit:

- Platt scaling when the calibration set has fewer than 1,000 unique fights;
- isotonic regression otherwise.

Neither the final test block nor its labels may reach model fitting, candidate
selection, or calibrator fitting.

## 6. Temporal evaluation design

### 6.1 Event index and folds

`eval.splits` owns the sole legal splitter. Its input is an `EventIndex` containing
`event_url` and event date for eligible fights; its output is immutable `Fold` objects
with train, calibration, and test event-ID sets.

```python
@dataclass(frozen=True)
class Fold:
    train_event_ids: frozenset[str]
    calibration_event_ids: frozenset[str]
    test_event_ids: frozenset[str]
```

Development folds are deterministic expanding windows over events before 2026. The
default configuration uses four folds; three is permitted when the requested minimum
training and test sizes cannot support four. Event URLs break same-date ties. A fold
is invalid if an event appears in more than one partition or if any fit/calibration
event is dated on or after the earliest test event.

### 6.2 Final holdout

Every UFC event dated from 2026-01-01 through 2026-08-31 is the final holdout. It is
not read during feature selection, candidate selection, or calibration-policy choice.
After August is complete, evaluate the locked configuration once using the same
expanding, event-grouped procedure; report metrics and confidence intervals only then.

### 6.3 Metrics and report shape

Metrics operate on one canonical probability per resolved fight, never on mirrored
training rows. Each evaluation emits a typed `EvaluationReport` containing:

- per-fold and pooled Brier score, log loss, and ECE;
- reliability-curve bins and counts;
- event-level cluster-bootstrap confidence intervals; and
- fold dates, sample counts, exclusions, calibration method, candidate configuration,
  and provenance.

The bootstrap resamples events with replacement and includes all eligible fights from
each sampled event. Fight-level resampling is prohibited.

### 6.4 MLflow artifact contract

MLflow is the only v1 model-artifact store. A successful training/evaluation run logs:

- parameters, resolved configuration, seeds, package versions, and data revision;
- feature schema/version and feature source hash;
- fold assignments, evaluation report, reliability data, and candidate comparison;
- the selected XGBoost model and fitted calibrator; and
- a `run_manifest.json` identifying every artifact and its provenance.

A report command accepts an MLflow run ID. It must load the model, calibrator, and
feature schema from that run; it may not substitute a local “latest model.”

## 7. Market report and paper-signal design

### 7.1 Upcoming-fights input

The report consumes a DVC-versioned upcoming-fights input with one record per bout:

```text
event_date, fighter_a_name, fighter_b_name, weight_class
```

On load, each name must resolve to exactly one existing UFCStats `fighter_url`.
Resolution failures are explicit input errors. The resolved URLs become the canonical
bout identity; display names are not used as model keys.

### 7.2 Strict market matching

`report.matching` normalizes case, whitespace, punctuation, and diacritics in the two
fighter names. It considers only active UFC-tagged captured markets and requires one
market whose question contains both normalized names. It returns one of:

```text
MATCHED | NO_CANDIDATE | MULTIPLE_CANDIDATES | INVALID_INPUT | MISSING_SNAPSHOT
```

Only `MATCHED` is eligible for a probability comparison. Other statuses are written
to the report with their candidate count and reason; they never receive an implied
probability by guesswork.

### 7.3 Snapshot selection and implied probability

The caller supplies a timezone-aware `as_of_timestamp`. For a matched market, select
the latest capture tick at or before that timestamp which contains the needed outcome
token. Prefer `tick_id` so a report uses a coherent capture cross-section; records
without a usable tick ID are not used for new v1 reports.

The market implied probability is the non-null `mid_price` for the token corresponding
to the report's canonical fighter. Missing prices produce `MISSING_SNAPSHOT`; there is
no bid/ask fallback in v1. The signed mismatch is:

```text
mismatch = model_probability − market_implied_probability
```

### 7.4 Report and paper-signal writes

One `report_runs` row is written before processing bouts. Each input bout produces one
`paper_signals` row, including failures. A successful row includes canonical fighter
URLs and names, event date, market/token IDs, snapshot timestamp and tick ID, both
probabilities, mismatch, match status, MLflow run ID, data revision, feature version,
and configuration hash.

Writes are append-only. Re-running the same report command creates a new report run;
it does not overwrite historical paper signals. This makes later performance analysis
possible without claiming that the system traded or beat the market.

## 8. Orchestration and operations

### 8.1 Post-event batch sequence

After a completed UFC event:

1. Run the scraper and allow idempotent source upserts to finish.
2. Run validation. If it fails, stop; do not build features or retrain.
3. Materialize the configured feature version.
4. Train/evaluate the locked XGBoost workflow and log the run to MLflow.
5. Mark the successful MLflow run as the explicit input for the next report command.

Retraining is event-triggered, not online. It must not overwrite an existing MLflow
run or change previous report records.

### 8.2 Health command

`ops.health` emits one JSON document suitable for a terminal, scheduler, or future UI.
It contains:

```text
generated_at
latest_ufc_event_date
latest_ufc_scraped_at
validation_status
quarantine_counts_by_reason
latest_market_capture_at
latest_market_tick_id
latest_successful_mlflow_run_id
latest_model_feature_version
```

Validation failure is a blocking health condition. Other fields are reported as facts
and ages; freshness thresholds are configuration, not hard-coded business logic.

## 9. Failure handling and observability

| Condition | Required behavior |
|---|---|
| Source schema drift or invalid ingest payload | fail the ingest operation; preserve prior rows |
| Validation failure in modern era or undated row | refresh quarantine, return nonzero, block downstream batch steps |
| Pre-cutoff validation violation | quarantine and report it; do not block the run |
| Feature registry/schema/version error | fail before target-table swap |
| Illegal fold or calibration leakage | fail evaluation before fitting a candidate |
| MLflow artifact write failure | mark the run failed; do not expose it to reporting |
| Missing/ambiguous market or price | persist a non-success paper-signal status; continue other bouts |
| Re-run of scrape/capture/report | retain idempotent source writes; report runs stay append-only |

All batch commands log a run ID, resolved configuration, source cutoff or data
revision, row counts, exclusion counts, and elapsed time. Logs must never contain
environment secrets.

## 10. Test gates

The following are release gates for their owning module:

| Area | Required tests |
|---|---|
| data | existing parser fixtures, schema validation, invariants, quarantine refresh |
| features | deletion oracle, same-card isolation, determinism, registry/version guard |
| eval | event grouping, temporal ordering, calibration isolation, deterministic folds, event bootstrap |
| model | mirrored assembly, no market columns, fixed-round configuration, exact reverse-order complement |
| report | name normalization, unique/zero/multiple candidate statuses, as-of tick selection, append-only writes |
| ops | health output for clean, stale, quarantined, and no-model states |

`make validate`, `make lint`, and `make test` must pass before any reported result is
accepted. The 2026 final holdout is a process gate as well as a metric gate: no model
choice may be changed after it is evaluated.
