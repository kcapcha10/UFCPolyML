# Requirements Document — Model and Evaluation

## Introduction

This spec defines the model training, calibration, symmetric inference, temporal
evaluation, and market-relative comparison layer for UFCPolyML v1. It consumes the
versioned `features_v{N}` table produced by the feature-engine spec and the
`market_fight_links` table produced by the mismatch-report spec, and owns the XGBoost
lifecycle from matrix assembly through MLflow-logged calibrated probabilities. All
decisions D10, D16, D19–D22, D25–D29, D33, D35, D37 are implemented here; D28 and
D38 are consumed as inputs.

## Glossary

- **Matrix_Assembler**: module that transforms per-fighter feature rows into fight-level training examples with mirrored orientations
- **Candidate_Configuration**: one fixed-hyperparameter XGBoost specification from the ~12-member search set
- **Expanding_Window_Fold**: a chronological, event-grouped train/calibration/test partition where the training set grows with each subsequent fold
- **Calibrator**: a post-hoc probability map (Platt, isotonic, or beta) fitted on the calibration slice
- **Symmetric_Inference**: inference protocol averaging both fighter orientations and calibrating once in canonical order
- **Brier_Skill**: `1 − (Brier_model / Brier_market)`, the headline metric comparing model and market
- **Ablation_Ladder**: nested sequence of feature-family subsets trained on identical folds to measure incremental value
- **Reliability_Artifact**: stratified calibration report with per-bucket sample counts, ECE, and binomial CIs
- **Power_Analysis**: pre-evaluation computation of minimum detectable effect at α=0.05, power=0.8
- **Paired_Permutation_Test**: non-parametric test on per-fight Brier differences between model and market
- **Cumulative_Evidence_Tracker**: rolling Brier-skill plot with confidence bands accruing after holdout events resolve
- **Development_Fold**: any fold used before the 2026 holdout for candidate selection and calibrator choice
- **Final_Holdout**: every UFC event from January through August 2026; evaluated once only
- **Feature_Family**: a named group of related features (record, physical, schedule-strength, domain-interactions)
- **Calibration_Slice**: trailing event-complete partition inside the training window with at least `max(250, ceil(20% × N))` unique fights

## Requirements

---

**Bucket A — Matrix Assembly** (Requirements ME-1–ME-2)

### Requirement ME-1: Feature Matrix Construction

**User Story:** As a model trainer, I want the assembler to build fight-level examples from the per-fighter feature table, so that XGBoost receives correctly oriented, market-free input.

**Requirement:** The Matrix_Assembler SHALL consume `features_v{N}`, join both fighters for each fight, construct matchup deltas declared in the feature registry, create two training orientations per fight, and reject any matrix containing market-derived columns or a schema mismatch with the requested feature version.

#### Acceptance Criteria

1. WHEN the Matrix_Assembler receives a `features_v{N}` table, THE Matrix_Assembler SHALL construct one training example per fighter orientation: `(A_features − B_features, label = A won)` and `(B_features − A_features, label = B won)`.
2. WHEN a column in the input feature table is listed in the market-derived column blocklist, THE Matrix_Assembler SHALL reject the entire matrix with a named error before any model fitting.
3. WHEN the input feature schema version does not match the requested version, THE Matrix_Assembler SHALL raise a schema mismatch error and refuse to assemble.
4. WHEN a source value in `features_v{N}` is `NULL`, THE Matrix_Assembler SHALL convert it to `NaN` for XGBoost's native missing-value handling.
5. WHEN a fight has a non-binary outcome (draw, no-contest) or the winner is not a recorded participant, THE Matrix_Assembler SHALL exclude the fight and increment a named exclusion counter.
6. WHEN the assembler completes, THE Matrix_Assembler SHALL emit a typed assembly manifest containing: row count, exclusion counts by reason, feature schema version, source hash, and column list.
7. IF a feature-family subset is specified (for ablation), THE Matrix_Assembler SHALL include only columns belonging to the named families and their declared interactions.

### Requirement ME-2: Ablation-Ladder Feature Subsets

**User Story:** As an evaluator, I want to train models on progressively richer feature subsets, so that I can measure each family's incremental predictive value.

**Requirement:** The Matrix_Assembler SHALL support parameterized feature-family subsets corresponding to the ablation ladder: naive (no features, 50/50 floor), record, record+physical, record+physical+schedule-strength, and record+physical+schedule-strength+domain-interactions.

#### Acceptance Criteria

1. WHEN the ablation rung is `naive`, THE Matrix_Assembler SHALL produce a constant 0.5 prediction floor without assembling features.
2. WHEN the ablation rung is `record`, THE Matrix_Assembler SHALL include only features from Section 3 (win/loss record) and Section 3a (ufcstats-computable debut context).
3. WHEN the ablation rung is `physical`, THE Matrix_Assembler SHALL add Section 1 (physical profile) and Section 2 (activity) features to the prior rung.
4. WHEN the ablation rung is `schedule_strength`, THE Matrix_Assembler SHALL add Section 9a (Elo), Section 9a-bis (Glicko-2), Section 9b (PageRank), and Section 9c (common opponents) to the prior rung.
5. WHEN the ablation rung is `domain_interactions`, THE Matrix_Assembler SHALL add Sections 4/4a (finishing), 5/5a (output), 6 (experience), 8a/8b (weight class), 10/10a/10b (matchup/style), and 11 (rematch) to the prior rung.
6. WHEN an ablation rung is specified, THE Matrix_Assembler SHALL include only matchup deltas and interactions whose constituent features are all within the rung's column set.

---

**Bucket B — Training and Candidate Selection** (Requirements ME-3–ME-4)

### Requirement ME-3: XGBoost Training Protocol

**User Story:** As a model trainer, I want a reproducible, fixed-round training protocol with a constrained hyperparameter surface, so that results are auditable and overfit risk is controlled.

**Requirement:** The training module SHALL fit XGBoost binary classifiers using fixed boosting rounds, the 8-axis hyperparameter surface, and a small fixed candidate set of approximately 12 configurations selected by mean calibrated Brier across development folds.

#### Acceptance Criteria

1. WHEN training a candidate, THE Training_Module SHALL use a fixed `n_estimators` (no early stopping) and the 8-axis surface: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
2. WHEN evaluating candidates, THE Training_Module SHALL score each on mean calibrated Brier score across all development folds.
3. WHEN two candidates have equal mean calibrated Brier, THE Training_Module SHALL break ties by lower mean log loss, then lower mean ECE.
4. WHEN the selected candidate is locked, THE Training_Module SHALL log its configuration, selection rationale, all candidate scores, and the winning fold-level metrics to MLflow.
5. IF the candidate set contains fewer than 8 or more than 20 configurations, THE Training_Module SHALL raise a configuration error.
6. WHEN training completes for a candidate-fold combination, THE Training_Module SHALL record the random seed, feature version, data revision, and elapsed time.

### Requirement ME-4: Expanding-Window Event-Grouped Folds

**User Story:** As an evaluator, I want temporal, event-grouped expanding-window folds, so that evaluation respects real-world information flow.

**Requirement:** The fold generator SHALL produce 3–4 expanding-window, event-grouped development folds where all fights from one event stay together, train precedes calibration precedes test, and the 2026 holdout is never included in development folds.

#### Acceptance Criteria

1. WHEN generating folds, THE Fold_Generator SHALL assign every fight in an event to the same partition.
2. WHEN generating folds, THE Fold_Generator SHALL ensure every event in the training set has a date strictly before the earliest event in the calibration set, and every calibration event strictly before the earliest test event.
3. WHEN generating folds, THE Fold_Generator SHALL expand the training window in each successive fold to include all events from prior folds' training and test sets.
4. WHEN the event count cannot support 4 folds with the configured minimum training and test sizes, THE Fold_Generator SHALL produce 3 folds.
5. WHEN an event date falls within January–August 2026, THE Fold_Generator SHALL exclude it from all development folds.
6. WHEN generating folds, THE Fold_Generator SHALL reserve a trailing, event-complete calibration slice with at least `max(250, ceil(20% × N))` unique fights within each fold's training window.
7. WHEN same-date events exist, THE Fold_Generator SHALL break ties deterministically by `event_url`.

---

**Bucket C — Calibration and Symmetric Inference** (Requirements ME-5–ME-6)

### Requirement ME-5: Per-Fold Calibrator Model Selection

**User Story:** As an evaluator, I want the calibrator chosen empirically per fold from Platt/isotonic/beta, so that the mapping method is evidence-based, not assumed.

**Requirement:** On each development fold, the calibration module SHALL fit Platt scaling, isotonic regression, and beta calibration on the calibration slice, select by ECE with Brier tiebreak on the fold's evaluation slice, and log all three fits and the winner to MLflow.

#### Acceptance Criteria

1. WHEN fitting calibrators on a fold, THE Calibration_Module SHALL fit all three methods: Platt (2-parameter logistic on logit of raw score), isotonic (monotonic step function), and beta (3-parameter `sigmoid(a·ln(s) − b·ln(1−s) + c)`).
2. WHEN selecting the winner, THE Calibration_Module SHALL choose the calibrator with lowest ECE on the fold's evaluation slice; IF ECE is tied, THE Calibration_Module SHALL select by lower Brier score.
3. WHEN selection completes, THE Calibration_Module SHALL log all three fitted calibrators, their evaluation-slice metrics (ECE, Brier, log loss), and the winner identity to MLflow.
4. WHEN the calibration slice contains fewer than `max(250, ceil(20% × N))` unique fights, THE Calibration_Module SHALL raise a sizing error before fitting.
5. WHEN the final production calibrator is fitted for inference, THE Calibration_Module SHALL use the method that won across the majority of development folds (plurality rule).
6. IF the 2026 holdout data is referenced during calibrator selection or fitting, THE Calibration_Module SHALL raise a leakage error.

### Requirement ME-6: Symmetric Inference Protocol

**User Story:** As a consumer of predictions, I want exactly order-symmetric probabilities, so that reversing fighter order returns precisely `1 − p`.

**Requirement:** Inference SHALL create both orientations, average the raw probabilities to form a canonical raw score, calibrate once in canonical fighter order, and return `1 − p_calibrated` for the reversed order.

#### Acceptance Criteria

1. WHEN predicting fight (A, B), THE Inference_Module SHALL compute `p_raw = 0.5 × (p_ab + (1 − p_ba))` where `p_ab` is XGBoost's raw probability for A-over-B and `p_ba` for B-over-A.
2. WHEN determining canonical order, THE Inference_Module SHALL place the lexicographically smaller canonical fighter URL first.
3. WHEN calibrating, THE Inference_Module SHALL apply the fitted calibrator exactly once to `p_raw` in canonical order.
4. WHEN the requested order is reversed from canonical, THE Inference_Module SHALL return `1 − p_calibrated`.
5. WHEN predicting the same fight in both orders, THE Inference_Module SHALL produce probabilities that sum to exactly 1.0.
6. WHEN a fight's feature row is missing for either fighter, THE Inference_Module SHALL raise a missing-data error rather than imputing or defaulting.

---

**Bucket D — Evaluation, Market Comparison, and Statistical Testing** (Requirements ME-7–ME-10)

### Requirement ME-7: Market-Relative Evaluation

**User Story:** As a researcher, I want to evaluate my model against the market as a competing forecaster, so that I can answer whether the model prices fights better than Polymarket.

**Requirement:** The evaluation harness SHALL compute the market's Brier score and log loss on the same resolved fights as the model, report `Brier_skill = 1 − (Brier_model / Brier_market)` as the headline metric, and consume historical market-implied probabilities from the `market_fight_links` table.

#### Acceptance Criteria

1. WHEN evaluating a fold or holdout, THE Evaluation_Module SHALL join model predictions with market-implied probabilities from `market_fight_links` on the same resolved fights.
2. WHEN computing Brier_skill, THE Evaluation_Module SHALL use the formula `1 − (Brier_model / Brier_market)` where both Brier scores are computed on the identical fight set.
3. WHEN a fight in the evaluation set has no matched market probability in `market_fight_links`, THE Evaluation_Module SHALL exclude it from market-relative metrics and report the exclusion count.
4. WHEN reporting results, THE Evaluation_Module SHALL present Brier_skill as the headline metric alongside absolute Brier, log loss, and ECE for both model and market.
5. WHEN market-implied probability is used, THE Evaluation_Module SHALL verify it was never used as a training feature or label (market data flows only into the comparator path).

### Requirement ME-8: Power Analysis and Statistical Testing

**User Story:** As a researcher, I want to know the smallest effect my evaluation can detect before scoring the holdout, so that I report results with honest statistical context.

**Requirement:** The evaluation harness SHALL compute minimum detectable effect at α=0.05/power=0.8, apply a paired permutation test on per-fight Brier differences, maintain cumulative evidence tracking, and apply the framing rule.

#### Acceptance Criteria

1. WHEN preparing to score a holdout or fold, THE Power_Module SHALL compute the MDE (minimum detectable Brier-skill difference) given the fight count at α=0.05 and power=0.8.
2. WHEN comparing model vs market, THE Statistical_Test_Module SHALL compute a paired permutation test on per-fight Brier differences (model Brier_i − market Brier_i) with at least 10,000 permutations.
3. WHEN the paired test p-value does not clear α=0.05, THE Evaluation_Module SHALL frame results as point estimates with intervals and MDE context, and SHALL NOT claim the model beats the market.
4. WHEN the paired test p-value clears α=0.05, THE Evaluation_Module SHALL report the claim with the p-value, effect size, and confidence interval.
5. WHEN new events resolve after the holdout evaluation, THE Cumulative_Tracker SHALL update the running Brier-skill-vs-market plot with confidence bands.
6. WHEN reporting any metric, THE Evaluation_Module SHALL include the MDE alongside the point estimate so that the reader knows what the evaluation cannot detect.
7. WHEN computing confidence intervals, THE Evaluation_Module SHALL use event-level cluster bootstrap (resampling events with replacement, including all fights from each sampled event).

### Requirement ME-9: Ablation Evaluation with Incremental Brier and CIs

**User Story:** As a researcher, I want to see each feature family's incremental contribution with confidence intervals, so that the alpha hypothesis is testable.

**Requirement:** The ablation evaluator SHALL train and evaluate the nested model sequence on identical folds, reporting incremental Brier improvement per rung with event-bootstrap confidence intervals.

#### Acceptance Criteria

1. WHEN running the ablation ladder, THE Ablation_Module SHALL train each candidate configuration at every rung (naive → record → +physical → +schedule-strength → +domain-interactions) on identical folds.
2. WHEN computing incremental improvement, THE Ablation_Module SHALL report `ΔBrier = Brier_rung_k − Brier_rung_(k-1)` for each successive rung.
3. WHEN reporting incremental improvement, THE Ablation_Module SHALL include event-level cluster-bootstrap confidence intervals on each ΔBrier.
4. WHEN a rung's CI for ΔBrier includes zero, THE Ablation_Module SHALL flag it as "not significantly different from prior rung" in the report.
5. WHEN the ablation completes, THE Ablation_Module SHALL log all rung metrics, intervals, and the full ladder artifact to MLflow.

### Requirement ME-10: History-Depth-Stratified Metrics

**User Story:** As a researcher, I want model performance stratified by fighter history depth, so that reliability on sparse-history fighters is measured, not assumed.

**Requirement:** The evaluation harness SHALL report calibration and Brier stratified by history depth: fights involving a ≤3-fight fighter versus the rest.

#### Acceptance Criteria

1. WHEN evaluating a fold or holdout, THE Evaluation_Module SHALL compute `min_prior_ufc_fights` for each fight (lesser of both fighters' UFC fight counts at the time of the fight).
2. WHEN `min_prior_ufc_fights ≤ 3`, THE Evaluation_Module SHALL tag the fight as `SPARSE_HISTORY`.
3. WHEN reporting metrics, THE Evaluation_Module SHALL report Brier, ECE, and reliability curves separately for `SPARSE_HISTORY` fights and the remainder.
4. WHEN the `SPARSE_HISTORY` threshold is changed from the default (3), THE Evaluation_Module SHALL read it from Hydra configuration.

---

**Bucket E — Provenance, Operations, and Holdout** (Requirements ME-11–ME-13)

### Requirement ME-11: MLflow Provenance

**User Story:** As a researcher, I want every training and evaluation run fully logged to MLflow, so that any result is reproducible from its artifact.

**Requirement:** Every successful training/evaluation run SHALL log parameters, configuration, seeds, feature schema/version/hash, fold assignments, evaluation report, reliability data, candidate comparison, the XGBoost model, fitted calibrator, and a `run_manifest.json` to MLflow.

#### Acceptance Criteria

1. WHEN a training run completes successfully, THE Provenance_Module SHALL log to MLflow: resolved Hydra configuration, all random seeds, package version, data revision, feature schema version, feature source hash, and elapsed time.
2. WHEN a training run completes, THE Provenance_Module SHALL log the XGBoost model artifact, the fitted calibrator artifact, and the fold assignments.
3. WHEN evaluation completes, THE Provenance_Module SHALL log the typed `EvaluationReport`, per-fold and pooled metrics, reliability data, candidate comparison table, and the ablation ladder artifact.
4. WHEN any artifact write to MLflow fails, THE Provenance_Module SHALL mark the run as failed and SHALL NOT expose it to reporting.
5. WHEN the run completes, THE Provenance_Module SHALL write a `run_manifest.json` listing every artifact path, its type, and provenance fields.

### Requirement ME-12: Retrain-After-Each-Event Operational Mode

**User Story:** As an operator, I want the system to retrain after each completed UFC event, so that the model stays current without manual intervention.

**Requirement:** The training pipeline SHALL support a retrain-after-event mode that triggers training only after ingestion and validation complete cleanly, never overwrites existing MLflow runs, and updates the active model pointer.

#### Acceptance Criteria

1. WHEN a new UFC event's ingestion and validation complete cleanly, THE Retrain_Module SHALL trigger a full train/evaluate cycle using the locked candidate configuration.
2. WHEN retraining, THE Retrain_Module SHALL include all events up to and including the newly completed event in its training window.
3. WHEN a retrain completes, THE Retrain_Module SHALL NOT overwrite or mutate any existing MLflow run.
4. WHEN a retrain completes successfully, THE Retrain_Module SHALL update the active model pointer (latest successful MLflow run ID) for downstream report consumption.
5. IF ingestion or validation fails, THE Retrain_Module SHALL NOT proceed with retraining and SHALL log the blocking condition.

### Requirement ME-13: Final Holdout Evaluation

**User Story:** As a researcher, I want the 2026 January–August holdout scored exactly once after August completes, so that the headline result has maximum integrity.

**Requirement:** The 2026 holdout (every UFC event from 2026-01-01 through 2026-08-31) SHALL be evaluated once only after August is complete, using the locked configuration, and no modeling choice may be changed afterward.

#### Acceptance Criteria

1. WHEN evaluating the final holdout, THE Evaluation_Module SHALL use the locked candidate configuration and the calibrator method selected on development folds.
2. WHEN evaluating the final holdout, THE Evaluation_Module SHALL report Brier, Brier_skill, log loss, ECE, reliability curves, MDE, paired permutation test p-value, and event-bootstrap CIs.
3. WHEN the final holdout has been evaluated, THE Evaluation_Module SHALL write a `holdout_evaluated_at` timestamp and SHALL refuse subsequent model-selection or calibrator changes.
4. IF any development fold, feature selection, hyperparameter, or calibrator choice is attempted after holdout evaluation, THE Evaluation_Module SHALL raise a post-holdout-lock error.
5. WHEN reporting holdout results, THE Evaluation_Module SHALL include history-depth-stratified metrics (ME-10) and the reliability artifact with per-bucket binomial CIs (ME-5).

---

**Bucket F — Reliability Artifact** (Requirement ME-14)

### Requirement ME-14: Stratified Reliability Report with Binomial CIs

**User Story:** As a downstream consumer, I want per-bucket calibration quality with confidence intervals, so that signal-relevant probability ranges are individually trustworthy.

**Requirement:** Every evaluation run SHALL produce a stratified reliability artifact with probability buckets [0.1–0.3], [0.3–0.5], [0.5–0.7], [0.7–0.9], per-bucket sample counts, measured calibration error, and binomial confidence intervals.

#### Acceptance Criteria

1. WHEN evaluation completes, THE Reliability_Module SHALL produce a reliability artifact with the four fixed probability buckets: [0.1–0.3], [0.3–0.5], [0.5–0.7], [0.7–0.9].
2. WHEN computing per-bucket metrics, THE Reliability_Module SHALL report: bucket boundaries, sample count, mean predicted probability, observed win rate, absolute calibration error, and 95% binomial CI on the observed win rate.
3. WHEN a bucket contains fewer than 10 fights, THE Reliability_Module SHALL flag it as `LOW_SUPPORT` and include the flag in the artifact.
4. WHEN the reliability artifact is produced, THE Reliability_Module SHALL log it to MLflow as a standard artifact of every training run.
5. WHEN the mismatch-report layer queries per-bucket calibration error for gating (D34), THE Reliability_Module SHALL expose bucket boundaries, sample counts, calibration errors, and CIs via a typed read function.

## Standing Decisions with Named Fallbacks

1. **Fixed ~12-candidate random search.** If development fold variance exceeds 15% relative Brier between folds, evaluate whether doubling candidate count (to ~24) reduces selection noise before adding hyperparameters.

2. **Beta calibration as expected winner.** If beta consistently overfits (worse Brier on eval slice than Platt across ≥3 folds), investigate whether the 3-parameter fit needs regularization before removing it as a candidate.

3. **Paired permutation test for market comparison.** If the holdout fight count exceeds 1,000, evaluate whether a bootstrapped t-test provides tighter intervals before switching from the permutation approach.

4. **Four development folds.** If the event history before 2026 cannot support 4 folds with ≥150 test fights each, fall back to 3 folds rather than relaxing the minimum.
