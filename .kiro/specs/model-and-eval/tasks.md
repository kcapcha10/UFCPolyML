# Implementation Plan: UFCPolyML — Model and Evaluation

## Overview

This plan builds the model training, calibration, symmetric inference, temporal
evaluation, market-relative comparison, ablation ladder, and MLflow provenance layer.
It is organized into waves safe for parallel execution by independent subagents.

Key constraint: **ME Waves 0–2 build against FIXTURE feature tables** (synthetic
DuckDB files with known rows) so implementation parallelizes with the feature-engine
spec. Real `features_v{N}` integration happens only in ME Wave 4, which blocks on
FE completion. Market-relative evaluation blocks on `market_fight_links` from the
mismatch-report spec (MR Wave 1+).

Architecture reference: design.md Components 1–12, schemas, configs.

## Tasks

---

### Wave 0 — Schemas, Configuration, and Module Scaffolding (can start immediately)

- [ ] ME-0.1 Create eval/ package structure and schemas
  - Create `src/ufc_edge/eval/__init__.py`, `schemas.py`
  - Implement frozen Pydantic models: `Fold`, `EvaluationReport`, `FoldMetrics`, `ReliabilityBucket`, `AblationRungResult`, `PowerResult`, `PermutationResult`, `CumulativePoint`
  - All models use `ConfigDict(frozen=True)`, typed fields, field descriptions
  - Depends on: none
  - Files: `src/ufc_edge/eval/__init__.py`, `src/ufc_edge/eval/schemas.py`
  - _Requirements: ME-4, ME-7, ME-8, ME-9, ME-14_

- [ ] ME-0.2 Create model/ schemas and configuration
  - Implement `src/ufc_edge/model/schemas.py`: `AssemblyManifest`, `CandidateConfig`, `TrainResult`, `AblationRung` (StrEnum)
  - Create `configs/model/default.yaml` with ~12 candidate configurations per design.md
  - Create `configs/eval/default.yaml` with fold/calibration/holdout parameters
  - Depends on: none
  - Files: `src/ufc_edge/model/schemas.py`, `configs/model/default.yaml`, `configs/eval/default.yaml`
  - _Requirements: ME-3.1, ME-3.5_

- [ ] ME-0.3 Write schema and configuration tests
  - Test Pydantic frozen immutability, required fields, validation constraints
  - Test `CandidateConfig` rejects out-of-range values (negative learning_rate, etc.)
  - Test `AblationRung` enum ordering matches ladder sequence
  - Test Hydra config loads correctly via OmegaConf
  - Depends on: ME-0.1, ME-0.2
  - Files: `tests/model/test_schemas.py`, `tests/eval/test_schemas.py`
  - _Requirements: ME-3.5_

- [ ] ME-0.4 Create fixture feature table for testing
  - Build a pytest fixture (`conftest.py`) producing a DuckDB with 50 fights, 10 events, synthetic feature columns matching the expected schema
  - Include: known outcomes, multiple weight classes, at least 2 fights with draws/NC (for exclusion testing), NULL values in some columns
  - Include: a `MARKET_COLUMNS` entry to test rejection
  - Depends on: none
  - Files: `tests/conftest.py` (additions), `tests/fixtures/fixture_features.py`
  - _Requirements: ME-1, ME-2_

---

### Wave 1 — Core Model Components (can start immediately, parallel to Wave 0 after ME-0.2)

Within a wave, tasks have no dependency edges and may launch simultaneously.

- [ ] ME-1.1 Implement matrix assembler (TDD)
  - Write failing tests first:
    - Test: two orientations created per fight, label correctness
    - Test: market-column rejection raises `MarketLeakageError`
    - Test: schema version mismatch raises `SchemaMismatchError`
    - Test: NULL → NaN conversion (no Python None in output array)
    - Test: draw/NC exclusion with correct counts
    - Test: ablation rung subsetting (each rung ⊂ next rung's columns)
    - Test: manifest contains correct row count, exclusion counts, columns
  - Implement until green:
    - `src/ufc_edge/model/matrix.py`: `assemble_matrix()`, `MARKET_COLUMNS` frozen set, NULL→NaN conversion, orientation mirroring, exclusion counting, schema-version check
    - Ablation rung mapping: column-family → feature-name registry lookup
    - Rejects any column in `MARKET_COLUMNS`; raises `MarketLeakageError`
  - Depends on: ME-0.2, ME-0.4
  - Files: `src/ufc_edge/model/matrix.py`, `tests/model/test_matrix.py`
  - _Requirements: ME-1.1, ME-1.2, ME-1.3, ME-1.4, ME-1.5, ME-1.6, ME-1.7, ME-2_

- [ ] ME-1.2 Implement fold generator (TDD)
  - Write failing tests first:
    - Test: all fights from same event stay in one partition
    - Test: temporal ordering (train < calibration < test by date)
    - Test: expanding window grows with each fold
    - Test: holdout exclusion (2026 events raise `HoldoutLeakageError` if breached)
    - Test: calibration sizing `max(250, ceil(20% × N))`
    - Test: fallback to 3 folds when 4 not supportable
    - Test: determinism (same input → same folds)
    - Test: same-date tie-breaking by event_url
  - Implement until green:
    - `src/ufc_edge/eval/splits.py`: `generate_folds()`, holdout guard, expanding-window logic, calibration sizing rule, same-date tie-breaking by event_url
    - Raises `HoldoutLeakageError` if any 2026 event leaks
    - Configurable `n_folds`, `min_train_fights`, `min_test_fights`, `calibration_min`, `calibration_ratio`
  - Depends on: ME-0.1, ME-0.4
  - Files: `src/ufc_edge/eval/splits.py`, `tests/eval/test_splits.py`
  - _Requirements: ME-4.1, ME-4.2, ME-4.3, ME-4.4, ME-4.5, ME-4.6, ME-4.7_

- [ ] ME-1.3 Implement metrics module (TDD)
  - Write failing tests first:
    - Test: Brier score on known distribution (e.g., perfect predictor = 0.0, random = 0.25)
    - Test: log loss on known distribution
    - Test: ECE on perfectly calibrated vs miscalibrated predictions
    - Test: bootstrap CI width shrinks with more events
    - Test: bootstrap respects event grouping (never resamples individual fights)
  - Implement until green:
    - `src/ufc_edge/eval/metrics.py`: `brier_score()`, `log_loss_score()`, `expected_calibration_error()`, `event_bootstrap_ci()`
    - Pure functions, no side effects, no DuckDB, no MLflow
    - Bootstrap resamples events with replacement, includes all fights per sampled event
  - Depends on: ME-0.1
  - Files: `src/ufc_edge/eval/metrics.py`, `tests/eval/test_metrics.py`
  - _Requirements: ME-7, ME-8.7_

---

### Wave 2 — Calibration, Inference, and Evaluation Components (blocks on Wave 1)

- [ ] ME-2.1 Implement calibration module
  - `src/ufc_edge/eval/calibration.py`: `fit_calibrators()`, `select_calibrator()`, Platt (LogisticRegression on logit), isotonic (IsotonicRegression), beta (scipy minimize)
  - Selection: lowest ECE on eval slice; Brier tiebreak
  - Sizing guard: raises `CalibrationSizingError` if below threshold
  - Leakage guard: raises if 2026 holdout events detected in inputs
  - Depends on: ME-1.3, ME-0.1
  - Files: `src/ufc_edge/eval/calibration.py`
  - _Requirements: ME-5.1, ME-5.2, ME-5.3, ME-5.4, ME-5.5, ME-5.6_

- [ ] ME-2.2 Write calibration tests
  - Test: Platt correctness (known logistic transform recoverable)
  - Test: isotonic correctness (monotone non-decreasing output)
  - Test: beta calibration on synthetic data (verify 3-parameter fit)
  - Test: selection logic picks lowest ECE; tiebreak by Brier
  - Test: sizing guard fires below threshold
  - Test: leakage guard fires if holdout events present
  - Test: all three calibrators logged with metrics
  - Depends on: ME-2.1
  - Files: `tests/eval/test_calibration.py`
  - _Requirements: ME-5_

- [ ] ME-2.3 Implement symmetric inference
  - `src/ufc_edge/model/inference.py`: `predict_symmetric()` → `SymmetricPrediction`
  - Canonical order: lexicographically smaller fighter URL first
  - Raw averaging: `p_raw = 0.5 × (p_ab + (1 − p_ba))`
  - Single calibration pass on canonical raw; reversed order returns `1 − p_calibrated`
  - Raises `MissingDataError` if either fighter's features are missing
  - Depends on: ME-2.1, ME-1.1
  - Files: `src/ufc_edge/model/inference.py`
  - _Requirements: ME-6.1, ME-6.2, ME-6.3, ME-6.4, ME-6.5, ME-6.6_

- [ ] ME-2.4 Write symmetric inference tests
  - **Property test:** for 100 random (A,B) pairs, `p(A,B) + p(B,A) = 1.0` exactly
  - Test: canonical ordering is lexicographic on URL
  - Test: raw averaging formula verified against manual computation
  - Test: missing features raise `MissingDataError`
  - Test: SymmetricPrediction fields populated correctly
  - Depends on: ME-2.3, ME-0.4
  - Files: `tests/model/test_inference.py`
  - _Requirements: ME-6_

- [ ] ME-2.5 Implement XGBoost trainer
  - `src/ufc_edge/model/train.py`: `train_candidate()`, `select_best_candidate()`
  - Fixed `num_boost_round = config.n_estimators`; never passes `early_stopping_rounds`
  - Selection by mean calibrated Brier; tiebreak log loss, then ECE
  - Validates candidate count is 8–20
  - Depends on: ME-1.1, ME-0.2
  - Files: `src/ufc_edge/model/train.py`
  - _Requirements: ME-3.1, ME-3.2, ME-3.3, ME-3.4, ME-3.5, ME-3.6_

- [ ] ME-2.6 Write trainer tests
  - Test: model has exactly `n_estimators` trees (fixed-round invariant)
  - Test: no `early_stopping_rounds` in XGBoost params
  - Test: selection picks lowest mean calibrated Brier
  - Test: tiebreak logic (Brier tied → lower log loss wins)
  - Test: candidate count guard (rejects <8 or >20)
  - Test: seed, version, elapsed time recorded per run
  - Depends on: ME-2.5, ME-0.4
  - Files: `tests/model/test_train.py`
  - _Requirements: ME-3_

- [ ] ME-2.7 Implement reliability module
  - `src/ufc_edge/eval/reliability.py`: `stratified_reliability()`, `FIXED_BUCKETS`, Clopper-Pearson binomial CI
  - Four buckets: [0.1–0.3], [0.3–0.5], [0.5–0.7], [0.7–0.9]
  - `LOW_SUPPORT` flag when bucket has <10 fights
  - Exposes typed read function for D34 mismatch-report consumption
  - Depends on: ME-0.1
  - Files: `src/ufc_edge/eval/reliability.py`
  - _Requirements: ME-14.1, ME-14.2, ME-14.3, ME-14.4, ME-14.5_

- [ ] ME-2.8 Write reliability tests
  - Test: four buckets produced with correct boundaries
  - Test: Clopper-Pearson CI matches scipy reference for known n, k
  - Test: LOW_SUPPORT flag fires when n < 10
  - Test: calibration error = |mean_predicted − observed_win_rate|
  - Test: typed read function returns correct structure for downstream consumption
  - Depends on: ME-2.7
  - Files: `tests/eval/test_reliability.py`
  - _Requirements: ME-14_

---

### Wave 3 — Evaluation Orchestration, Power, Ablation (blocks on Wave 2)

- [ ] ME-3.1 Implement power analysis and permutation test
  - `src/ufc_edge/eval/power.py`: `minimum_detectable_effect()`, `paired_permutation_test()`
  - MDE: `(z_alpha + z_beta) × σ / √n` using scipy.stats.norm
  - Permutation test: 10,000 permutations, two-sided, seeded for reproducibility
  - Depends on: ME-1.3
  - Files: `src/ufc_edge/eval/power.py`
  - _Requirements: ME-8.1, ME-8.2_

- [ ] ME-3.2 Write power analysis and permutation tests
  - Test: MDE against scipy.stats.norm.ppf reference for known n, sigma
  - Test: permutation test rejects known non-null (synthetic large-effect data)
  - Test: permutation test does not reject known null (shuffled identical arrays)
  - Test: reproducibility with same seed
  - Depends on: ME-3.1
  - Files: `tests/eval/test_power.py`
  - _Requirements: ME-8.1, ME-8.2_

- [ ] ME-3.3 Implement market-relative evaluation
  - `src/ufc_edge/eval/market_relative.py`: `compute_brier_skill()`, join on `market_fight_links`
  - Brier_skill = 1 − (Brier_model / Brier_market)
  - Excludes fights without matched market probability; reports exclusion count
  - Verifies market data never used as training feature (read-only join)
  - Depends on: ME-1.3
  - Files: `src/ufc_edge/eval/market_relative.py`
  - _Requirements: ME-7.1, ME-7.2, ME-7.3, ME-7.4, ME-7.5_

- [ ] ME-3.4 Write market-relative evaluation tests
  - Test: Brier_skill formula correctness (model Brier 0.2, market 0.25 → skill = 0.2)
  - Test: exclusion count correct when fights have no market match
  - Test: both model and market scored on identical fight set
  - Test: result includes both absolute metrics and relative metrics
  - Depends on: ME-3.3
  - Files: `tests/eval/test_market_relative.py`
  - _Requirements: ME-7_

- [ ] ME-3.5 Implement ablation ladder
  - `src/ufc_edge/eval/ablation.py`: `run_ablation_ladder()`
  - Trains all ~12 candidates at each of 5 rungs on identical folds
  - Computes ΔBrier per rung with event-bootstrap CIs
  - Flags non-significant improvements (CI includes zero)
  - Depends on: ME-2.5, ME-2.1, ME-1.1, ME-1.2, ME-1.3
  - Files: `src/ufc_edge/eval/ablation.py`
  - _Requirements: ME-9.1, ME-9.2, ME-9.3, ME-9.4, ME-9.5_

- [ ] ME-3.6 Write ablation ladder tests
  - Test: 5 rungs produced in correct order (naive → record → physical → schedule_strength → domain_interactions)
  - Test: column sets are nested (rung k ⊃ rung k-1)
  - Test: ΔBrier correctly computed as difference between successive rungs
  - Test: significance flag fires when CI includes zero
  - Test: identical folds used across all rungs
  - Test: naive rung produces constant 0.5 (Brier = 0.25)
  - Depends on: ME-3.5, ME-0.4
  - Files: `tests/eval/test_ablation.py`
  - _Requirements: ME-9_

- [ ] ME-3.7 Implement history-depth-stratified metrics
  - Add to `src/ufc_edge/eval/metrics.py`: `stratify_by_history_depth()` function
  - Computes `min_prior_ufc_fights` per fight, tags `SPARSE_HISTORY` when ≤ threshold (default 3, from Hydra config)
  - Reports Brier, ECE, reliability separately for sparse vs non-sparse
  - Depends on: ME-1.3, ME-2.7
  - Files: `src/ufc_edge/eval/metrics.py` (additions)
  - _Requirements: ME-10.1, ME-10.2, ME-10.3, ME-10.4_

- [ ] ME-3.8 Write history-depth-stratified tests
  - Test: fights with ≤3 prior UFC fights tagged SPARSE_HISTORY
  - Test: threshold configurable from Hydra
  - Test: separate Brier/ECE reported for each stratum
  - Test: min_prior_ufc_fights correctly computed as lesser of both fighters
  - Depends on: ME-3.7
  - Files: `tests/eval/test_metrics.py` (additions)
  - _Requirements: ME-10_

- [ ] ME-3.9 Implement cumulative evidence tracker
  - `src/ufc_edge/eval/cumulative.py`: `update_cumulative_evidence()`
  - Maintains running Brier-skill time series with event-bootstrap CI bands
  - Appends new events; recomputes running statistic
  - Depends on: ME-3.3, ME-1.3
  - Files: `src/ufc_edge/eval/cumulative.py`
  - _Requirements: ME-8.5_

- [ ] ME-3.10 Write cumulative evidence tests
  - Test: series grows when new events added
  - Test: CI bands narrow with more data
  - Test: consistent with point-in-time Brier_skill computation
  - Depends on: ME-3.9
  - Files: `tests/eval/test_cumulative.py`
  - _Requirements: ME-8.5_

---

### Wave 4 — Provenance, Retrain Mode, Integration (blocks on Wave 3; real-data integration blocks on FE completion)

- [ ] ME-4.1 Implement MLflow provenance module
  - `src/ufc_edge/eval/provenance.py`: `log_training_run()` → returns `run_id`
  - Logs: resolved config, seeds, package version, data revision, feature schema/version/hash, fold assignments, eval report, reliability artifact, candidate comparison, ablation artifact, XGBoost model, calibrator, `run_manifest.json`
  - Marks run failed if any artifact write fails; does not expose failed runs
  - Depends on: ME-2.5, ME-2.1, ME-2.7, ME-3.5
  - Files: `src/ufc_edge/eval/provenance.py`
  - _Requirements: ME-11.1, ME-11.2, ME-11.3, ME-11.4, ME-11.5_

- [ ] ME-4.2 Write provenance tests
  - Test: successful run creates MLflow run with all expected artifacts
  - Test: `run_manifest.json` lists every artifact path and type
  - Test: failed artifact write marks run as failed
  - Test: no half-logged runs exposed (transactional behavior)
  - Test: resolved config, seed, feature version all present in logged params
  - Depends on: ME-4.1
  - Files: `tests/eval/test_provenance.py`
  - _Requirements: ME-11_

- [ ] ME-4.3 Implement retrain-after-event mode
  - Add retrain orchestration to `src/ufc_edge/model/train.py` or new `src/ufc_edge/model/retrain.py`
  - Triggers after ingestion+validation clean completion
  - Uses locked candidate config; expands training window to include new event
  - Never overwrites existing MLflow runs
  - Updates active model pointer (latest successful run ID)
  - Blocks if ingestion or validation failed
  - Depends on: ME-4.1, ME-2.5
  - Files: `src/ufc_edge/model/retrain.py`
  - _Requirements: ME-12.1, ME-12.2, ME-12.3, ME-12.4, ME-12.5_

- [ ] ME-4.4 Write retrain mode tests
  - Test: retrain creates new MLflow run (not overwrite)
  - Test: training window includes new event
  - Test: blocked when validation failed (raises, does not proceed)
  - Test: active model pointer updated on success
  - Test: uses locked config (not re-selecting candidates)
  - Depends on: ME-4.3
  - Files: `tests/model/test_retrain.py`
  - _Requirements: ME-12_

- [ ] ME-4.5 Implement holdout evaluation with lock
  - Add holdout evaluation to `src/ufc_edge/eval/` (extend provenance or new `holdout.py`)
  - Writes `holdout_evaluated_at` timestamp on completion
  - Refuses subsequent model/calibrator/hyperparameter changes (`PostHoldoutLockError`)
  - Reports: Brier, Brier_skill, log loss, ECE, reliability, MDE, paired test, CIs, history-depth stratification
  - Depends on: ME-4.1, ME-3.1, ME-3.3, ME-3.7, ME-2.7
  - Files: `src/ufc_edge/eval/holdout.py`
  - _Requirements: ME-13.1, ME-13.2, ME-13.3, ME-13.4, ME-13.5_

- [ ] ME-4.6 Write holdout evaluation tests
  - Test: holdout uses locked config and selected calibrator method
  - Test: `holdout_evaluated_at` written on completion
  - Test: `PostHoldoutLockError` raised on subsequent model changes
  - Test: all required metrics present in output (Brier, skill, MDE, p-value, CIs, stratified)
  - Test: history-depth stratification included
  - Test: reliability artifact with per-bucket CIs included
  - Depends on: ME-4.5
  - Files: `tests/eval/test_holdout.py`
  - _Requirements: ME-13_

- [ ] ME-4.7 End-to-end integration test on fixture data
  - Full pipeline: fixture features → matrix assembly → fold generation → train candidates → calibrate → evaluate → reliability → ablation → provenance → MLflow logged
  - Verify: determinism (same seed → same results)
  - Verify: no temporal leakage in any fold
  - Verify: symmetry holds for all predictions
  - Verify: all artifacts logged to MLflow
  - Depends on: ME-4.1, ME-4.5, all Wave 2–3 components
  - Files: `tests/integration/test_model_eval_pipeline.py`
  - _Requirements: ME-1 through ME-14_

---

### Wave 5 — Real-Data Integration and Framing (blocks on FE completion + MR market_fight_links)

- [ ] ME-5.1 Integration with real `features_v{N}`
  - Verify matrix assembler works against FE-produced `features_v1` table
  - Run fold generation on real event index
  - Confirm schema version match, NULL→NaN, no market columns
  - Depends on: FE completion (cross-spec), ME-1.1, ME-1.2
  - Files: no new files; verification task
  - _Requirements: ME-1.3_

- [ ] ME-5.2 Integration with `market_fight_links`
  - Verify market-relative evaluation joins correctly against MR-produced link table
  - Confirm exclusion handling for unmatched fights
  - Run Brier_skill computation on real resolved fights
  - Depends on: MR Wave 1+ (cross-spec), ME-3.3
  - Files: no new files; verification task
  - _Requirements: ME-7.1, ME-7.3_

- [ ] ME-5.3 Framing rule enforcement verification
  - Write a test that verifies: when permutation test p > 0.05, report text uses only "point estimate with intervals and MDE context" language
  - Verify: when p ≤ 0.05, claim language is permitted with effect size and CI
  - Depends on: ME-3.1, ME-4.5
  - Files: `tests/eval/test_framing.py`
  - _Requirements: ME-8.3, ME-8.4_

- [ ] ME-5.4 Run `make lint` and `make test` verification
  - Confirm all new code passes ruff lint with project configuration
  - Confirm all tests pass with `pytest -v`
  - Fix any issues discovered
  - Depends on: all prior tasks
  - Files: no new files; verification task
  - _Requirements: all_

---

## Module Ownership Table

| Module | Files | Owner (spec) |
|---|---|---|
| `src/ufc_edge/model/schemas.py` | schemas | ME |
| `src/ufc_edge/model/matrix.py` | assembler | ME |
| `src/ufc_edge/model/train.py` | trainer | ME |
| `src/ufc_edge/model/inference.py` | symmetric inference | ME |
| `src/ufc_edge/model/retrain.py` | retrain mode | ME |
| `src/ufc_edge/eval/__init__.py` | package init | ME |
| `src/ufc_edge/eval/schemas.py` | all eval schemas | ME |
| `src/ufc_edge/eval/splits.py` | fold generation | ME |
| `src/ufc_edge/eval/calibration.py` | calibrators + selection | ME |
| `src/ufc_edge/eval/metrics.py` | scoring functions | ME |
| `src/ufc_edge/eval/reliability.py` | stratified artifact | ME |
| `src/ufc_edge/eval/ablation.py` | ladder evaluation | ME |
| `src/ufc_edge/eval/market_relative.py` | Brier_skill | ME |
| `src/ufc_edge/eval/power.py` | MDE + permutation | ME |
| `src/ufc_edge/eval/cumulative.py` | evidence tracker | ME |
| `src/ufc_edge/eval/provenance.py` | MLflow logging | ME |
| `src/ufc_edge/eval/holdout.py` | final holdout + lock | ME |
| `configs/model/default.yaml` | candidate grid | ME |
| `configs/eval/default.yaml` | fold/eval params | ME |

## Notes

- All tests are fixture-based. No live UFCStats, no network calls.
- `MARKET_COLUMNS` frozen set is maintained alongside the feature registry and updated when Section 16 changes.
- The eval/ module does not exist today — every file listed above is created by this spec.
- Calibrator implementations use sklearn (Platt, isotonic) and scipy (beta). No additional ML dependencies.
- The reliability module's typed read function is the contract consumed by the mismatch-report spec for D34 signal gating.

## LOE Summary

6 waves, 36 tasks. Waves 0–2 (15 tasks) are buildable against fixture data with no
cross-spec dependency and represent the bulk of the implementation. Wave 3 (10 tasks)
adds orchestration components that still use fixtures. Wave 4 (7 tasks) adds
provenance, retrain mode, and integration testing. Wave 5 (4 tasks) is real-data
integration verification that blocks on feature-engine and mismatch-report. Estimated
total: 10–14 subagent sessions at current task granularity.

## Timeline Summary

| Wave | Tasks | Runs in parallel with | Cross-spec joins |
|---|---|---|---|
| Wave 0 | ME-0.1 – ME-0.4 | FE Wave 0, MR Wave 0 | None — can start immediately |
| Wave 1 | ME-1.1 – ME-1.3 | FE Wave 1–2, MR Wave 0–1 | None — uses fixture feature table |
| Wave 2 | ME-2.1 – ME-2.8 | FE Wave 2–3, MR Wave 1 | None — uses fixture feature table |
| Wave 3 | ME-3.1 – ME-3.10 | FE Wave 3–4, MR Wave 2 | None — uses fixture data; market_relative tested with synthetic market_fight_links fixture |
| Wave 4 | ME-4.1 – ME-4.7 | FE Wave 4+ | None — MLflow integration on fixtures |
| Wave 5 | ME-5.1 – ME-5.4 | — | **Blocks on FE completion** (ME-5.1: real features_v1); **Blocks on MR Wave 1+** (ME-5.2: market_fight_links) |

## Task Dependency Graph

```text
Wave 0:  ME-0.1 ─┬─ ME-0.3
         ME-0.2 ─┘
         ME-0.4 (independent)

Wave 1:  ME-0.1 → ME-1.2
         ME-0.1 → ME-1.3
         ME-0.2 → ME-1.1

Wave 2:  ME-1.3 → ME-2.1 → ME-2.2
         ME-1.1 + ME-2.1 → ME-2.3 → ME-2.4
         ME-1.1 → ME-2.5 → ME-2.6
         ME-0.1 → ME-2.7 → ME-2.8

Wave 3:  ME-1.3 → ME-3.1 → ME-3.2
         ME-1.3 → ME-3.3 → ME-3.4
         ME-2.5 + ME-2.1 + ME-1.1 + ME-1.2 + ME-1.3 → ME-3.5 → ME-3.6
         ME-1.3 + ME-2.7 → ME-3.7 → ME-3.8
         ME-3.3 + ME-1.3 → ME-3.9 → ME-3.10

Wave 4:  ME-2.5 + ME-2.1 + ME-2.7 + ME-3.5 → ME-4.1 → ME-4.2
         ME-4.1 + ME-2.5 → ME-4.3 → ME-4.4
         ME-4.1 + ME-3.1 + ME-3.3 + ME-3.7 + ME-2.7 → ME-4.5 → ME-4.6
         All Wave 2–4 → ME-4.7

Wave 5:  FE done + ME-1.1 + ME-1.2 → ME-5.1
         MR W1 + ME-3.3 → ME-5.2
         ME-3.1 + ME-4.5 → ME-5.3
         All → ME-5.4
```
