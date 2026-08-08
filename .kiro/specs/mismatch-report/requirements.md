# Requirements Document — Mismatch Report and Paper-Signal Log

## Introduction

This specification defines the market-mismatch report, paper-signal persistence, and
LLM due-diligence annotation layer for UFCPolyML v1. It covers entity resolution
between Polymarket markets and UFC fights, capture schema extensions for bid/ask/depth
and post-signal snapshots, the magnitude-gated mismatch computation with
bucket-transparency columns, sparse-history annotation, and the search-augmented LLM
due-diligence service operating in annotate-only mode. The system produces
timestamped, append-only paper signals — it never places orders, sizes positions, or
claims trading returns.

## Glossary

- **Market_Fight_Link:** A persisted entity-resolution record joining a Polymarket
  token to a UFC fight, with match status and optional human confirmation.
- **Paper_Signal:** An immutable record of one bout's model vs. market comparison at
  a point in time, including provenance, match status, and optional annotations.
- **Report_Run:** An append-only metadata record identifying one execution of the
  report pipeline with its configuration and provenance.
- **Magnitude_Gate:** A statistical filter that flags a mismatch row when the
  absolute model–market divergence exceeds `k × bucket_calibration_error`.
- **Bucket_Transparency:** Per-row columns reporting the calibration bucket's sample
  count, measured error, and binomial confidence interval.
- **Sparse_History_Tag:** A flag applied when the lesser of both fighters' prior UFC
  fight counts falls below `min_prior_ufc_fights` (Hydra default: 3).
- **Due_Diligence_Verdict:** A frozen Pydantic schema (CONFIRM / QUALIFY / VETO) with
  evidence URLs and confidence, produced by the LLM due-diligence runner.
- **Post_Signal_Snapshot:** A scheduled market-state re-capture at approximately
  T+1h, T+4h, T+24h, and fight time following a flagged signal.
- **Top_Of_Book:** Best bid, best ask, and depth (total size) at the top level of the
  order book for a given token.
- **Verdict_Scoreboard:** A running tally comparing resolved fight outcomes against
  the due-diligence service's verdicts.

## Requirements

---

**Bucket A — Entity Resolution and Market Linkage** (Requirements 1–3)

### Requirement 1: Persisted Market↔Fight Linkage Table

**User Story:** As a report consumer, I want market-to-fight mappings materialized
and stable, so that evaluation joins are reproducible and matching logic changes do
not silently alter past results.

**Requirement:** The system SHALL maintain a `market_fight_links` table that
materializes the entity-resolution result between Polymarket tokens and UFC fights.
Once a link is confirmed, it SHALL NOT change.

#### Acceptance Criteria

1. WHEN the Link_Resolver processes a new fight-market pair, THE Link_Resolver SHALL
   write one `market_fight_links` row with columns `fight_url`, `token_id`,
   `match_status`, `match_method`, `matched_at`, and `reviewed_by`.
2. WHEN a link row has `match_status = MATCHED`, THE Link_Resolver SHALL NOT overwrite
   or delete that row in any subsequent run.
3. WHEN the Link_Resolver encounters a confirmed link for a fight-token pair, THE
   Link_Resolver SHALL reuse the existing link without re-running matching logic.
4. WHEN the Link_Resolver produces a match, THE Link_Resolver SHALL record
   `match_method` as one of `AUTO_NAME`, `HUMAN_CONFIRMED`, or `MANUAL_OVERRIDE`.
5. WHEN any downstream module requires a fight-to-market join, THAT module SHALL read
   exclusively from `market_fight_links` and SHALL NOT invoke name-matching logic
   directly.

### Requirement 2: Normalized Name Matching

**User Story:** As a system operator, I want name matching to handle common
variations automatically, so that unambiguous cases require no human intervention.

**Requirement:** The system SHALL normalize fighter names by lowercasing, stripping
diacritics, collapsing whitespace, and removing punctuation before attempting match.
It SHALL match only active UFC-tagged markets containing both normalized names.

#### Acceptance Criteria

1. WHEN the Name_Matcher normalizes a name, THE Name_Matcher SHALL apply lowercase,
   NFD decomposition with combining-mark removal, whitespace collapse, and
   punctuation stripping, in that order.
2. WHEN exactly one active UFC-tagged market question contains both normalized
   fighter names, THE Name_Matcher SHALL assign `match_status = MATCHED`.
3. WHEN zero active markets match, THE Name_Matcher SHALL assign
   `match_status = NO_CANDIDATE`.
4. WHEN two or more active markets match, THE Name_Matcher SHALL assign
   `match_status = MULTIPLE_CANDIDATES` and record the candidate count.
5. WHEN a fighter name fails to resolve to a valid `fighter_url`, THE Name_Matcher
   SHALL assign `match_status = INVALID_INPUT`.
6. WHEN a matched market has no capture snapshot at or before `as_of_timestamp`, THE
   Name_Matcher SHALL assign `match_status = MISSING_SNAPSHOT`.
7. WHEN the Name_Matcher assigns any status other than `MATCHED`, THE Name_Matcher
   SHALL persist the row with the non-success status and SHALL NOT infer a match.

### Requirement 3: One-Time Human Confirmation Flow

**User Story:** As a system operator, I want ambiguous matches surfaced for manual
review exactly once, so that resolution is fast and repeatable.

**Requirement:** The system SHALL surface `NO_CANDIDATE` and `MULTIPLE_CANDIDATES`
rows for human confirmation via a CLI command. Once confirmed, the row SHALL be
updated to `MATCHED` with `match_method = HUMAN_CONFIRMED` and `reviewed_by` set.

#### Acceptance Criteria

1. WHEN a human runs the confirmation CLI, THE Confirmation_CLI SHALL display all
   unresolved rows grouped by event date.
2. WHEN a human selects a token for a previously unresolved row, THE Confirmation_CLI
   SHALL update `match_status` to `MATCHED`, set `match_method` to
   `HUMAN_CONFIRMED`, and record the reviewer alias in `reviewed_by`.
3. WHEN a human skips a row, THE Confirmation_CLI SHALL leave the row status unchanged
   and SHALL NOT block subsequent report runs.
4. WHEN a row has been confirmed, THE Confirmation_CLI SHALL NOT present it again in
   future confirmation sessions.
5. WHEN the human provides a `token_id` that does not correspond to an active market,
   THE Confirmation_CLI SHALL reject the input with an error message.

---

**Bucket B — Capture Schema Extension and Post-Signal Snapshots** (Requirements 4–5)

### Requirement 4: Bid/Ask/Depth Capture Extension

**User Story:** As a future analyst, I want top-of-book bid, ask, and depth persisted
with every capture, so that convergence studies are possible in v2 without
retroactive data collection.

**Requirement:** The capture pipeline and paper-signal schema SHALL store best bid,
best ask, and top-of-book depth alongside the existing mid price for every captured
token.

#### Acceptance Criteria

1. WHEN the Capture_Pipeline writes an `order_book_snapshots` row, THE
   Capture_Pipeline SHALL persist `best_bid`, `best_ask`, and `best_bid_size`,
   `best_ask_size` extracted from the first level of the order book.
2. WHEN either side of the book is empty, THE Capture_Pipeline SHALL write `NULL` for
   the missing side's price and size.
3. WHEN a `paper_signals` row is created for a matched bout, THE Report_Pipeline SHALL
   denormalize `best_bid`, `best_ask`, `best_bid_size`, `best_ask_size` from the
   selected snapshot into the signal row.
4. WHEN the capture schema is extended, THE Capture_Pipeline SHALL remain
   backward-compatible with existing snapshot rows (NULL for new columns in legacy
   data).

### Requirement 5: Post-Signal Price Snapshots

**User Story:** As a future analyst, I want market state re-captured after a signal is
flagged, so that hold-to-resolution evaluation has the raw data it needs.

**Requirement:** For every flagged paper signal, the system SHALL schedule and persist
post-signal snapshots at approximately T+1h, T+4h, T+24h, and fight-time. These
snapshots are observational collection — v1 evaluates only hold-to-resolution.

#### Acceptance Criteria

1. WHEN a paper signal is flagged (passes the magnitude gate), THE Snapshot_Scheduler
   SHALL enqueue post-signal captures at approximately +1h, +4h, +24h, and at the
   event start time for the associated fight.
2. WHEN a scheduled snapshot fires, THE Snapshot_Scheduler SHALL write the capture to
   `post_signal_snapshots` with `signal_id`, `scheduled_offset`, `actual_captured_at`,
   and full top-of-book fields.
3. IF a scheduled snapshot fails (market closed, API error), THEN THE
   Snapshot_Scheduler SHALL record a `MISSED` status with the failure reason and SHALL
   NOT retry indefinitely.
4. WHEN evaluation consumes post-signal data, THE Evaluation_Module SHALL use only the
   fight-time snapshot for hold-to-resolution Brier; intermediate snapshots are
   retained for future v2 convergence study only.
5. WHEN the event start time is unavailable or in the past at scheduling time, THE
   Snapshot_Scheduler SHALL skip the fight-time snapshot and record `SKIPPED` with
   reason.

---

**Bucket C — Mismatch Computation and Signal Gating** (Requirements 6–8)

### Requirement 6: Mismatch Computation with Magnitude Gate

**User Story:** As a report consumer, I want mismatch signals filtered by a
statistically grounded threshold, so that rows within model noise are clearly
distinguished from actionable divergences.

**Requirement:** The mismatch report SHALL compute `mismatch = p_model − p_market_mid`
for every matched bout and SHALL flag rows where `|mismatch| > k ×
bucket_calibration_error`, where `k` is a Hydra-configurable multiplier (default: 2).

#### Acceptance Criteria

1. WHEN the Mismatch_Computer processes a matched bout, THE Mismatch_Computer SHALL
   compute `mismatch = p_model − p_market_mid` using the calibrated model probability
   and the market mid price.
2. WHEN `|mismatch|` exceeds `k × bucket_calibration_error` for the bucket containing
   `p_model`, THE Mismatch_Computer SHALL label the row `FLAGGED`.
3. WHEN `|mismatch|` does not exceed the threshold, THE Mismatch_Computer SHALL label
   the row `WITHIN_NOISE`.
4. WHEN the gate multiplier `k` is changed in Hydra config, THE Mismatch_Computer
   SHALL apply the new value without code changes.
5. WHEN a bout's `p_model` falls in a calibration bucket with zero sample count, THE
   Mismatch_Computer SHALL label the row `NO_BUCKET_DATA` and SHALL NOT flag it.

### Requirement 7: Bucket Transparency Columns

**User Story:** As a report consumer, I want each signal row to show the calibration
evidence behind the gate, so that I can discount signals backed by thin buckets.

**Requirement:** Every `paper_signals` row SHALL include the gating bucket's identity,
sample count, measured calibration error, and binomial confidence interval.

#### Acceptance Criteria

1. WHEN a `paper_signals` row is written, THE Report_Pipeline SHALL populate
   `bucket_id`, `bucket_n`, `bucket_calibration_error`, `bucket_ci_lower`, and
   `bucket_ci_upper` from the stratified reliability artifact.
2. WHEN the stratified reliability artifact is unavailable for the active MLflow run,
   THE Report_Pipeline SHALL fail the report run with a clear error rather than
   writing rows without bucket data.
3. WHEN the model's calibrated probability falls on a bucket boundary, THE
   Report_Pipeline SHALL assign it to the lower bucket consistently.
4. WHEN a consumer reads a `FLAGGED` row, THE Paper_Signal schema SHALL make the
   bucket transparency columns non-nullable so the evidence is always present.

### Requirement 8: Sparse-History Annotation

**User Story:** As a report consumer, I want fights involving low-experience fighters
tagged, so that I can weigh them against measured model reliability on such fighters.

**Requirement:** Every report row SHALL carry `min_prior_ufc_fights` and a
`SPARSE_HISTORY` boolean tag when that count is below `min_prior_ufc_fights_threshold`
(Hydra default: 3).

#### Acceptance Criteria

1. WHEN the Report_Pipeline computes a row, THE Report_Pipeline SHALL count completed
   UFC fights strictly before the current fight for each fighter and store the lesser
   count as `min_prior_ufc_fights`.
2. WHEN `min_prior_ufc_fights` is below the configured threshold, THE Report_Pipeline
   SHALL set `sparse_history = true`.
3. WHEN the threshold is changed in Hydra config, THE Report_Pipeline SHALL apply the
   new value without code changes.
4. WHEN `sparse_history = true`, THE Report_Pipeline SHALL NOT suppress or exclude the
   row from the report.
5. WHEN evaluation metrics are computed, THE Evaluation_Module SHALL report Brier and
   calibration stratified by history depth (sparse vs. non-sparse).

---

**Bucket D — LLM Due-Diligence Service** (Requirements 9–10)

### Requirement 9: Due-Diligence Verdict Production

**User Story:** As a report consumer, I want flagged signals annotated with structured
LLM research on injury, camp changes, and material news, so that I have additional
context without the LLM making suppression decisions.

**Requirement:** For flagged signals only, the system SHALL invoke a search-augmented
LLM that returns a frozen Pydantic verdict (CONFIRM / QUALIFY / VETO) with evidence
URLs and confidence. Verdicts are annotate-only; they SHALL NOT suppress signals.

#### Acceptance Criteria

1. WHEN a paper signal is `FLAGGED`, THE Due_Diligence_Runner SHALL invoke the LLM
   service for that (fight, report_run) pair.
2. WHEN the Due_Diligence_Runner produces a verdict, THE Due_Diligence_Runner SHALL
   validate it against the frozen `DueDiligenceVerdict` Pydantic schema before
   persistence.
3. WHEN a verdict already exists for a given `(fight_url, report_run_id)`, THE
   Due_Diligence_Runner SHALL return the existing verdict without re-invoking the LLM
   (idempotency guarantee).
4. WHEN the Due_Diligence_Runner writes a verdict, THE Due_Diligence_Runner SHALL log
   prompt version, model name, model version, and invocation timestamp alongside the
   verdict.
5. WHEN a verdict is VETO, THE Report_Pipeline SHALL attach the verdict to the signal
   row and SHALL NOT suppress or remove the signal from the report.
6. WHEN the LLM invocation fails (timeout, rate-limit, malformed response), THE
   Due_Diligence_Runner SHALL record `verdict = null` with `error_reason` and SHALL
   NOT block the report run.
7. WHEN the system writes a verdict, THE Due_Diligence_Runner SHALL include at least
   one evidence URL per material claim in the verdict.

### Requirement 10: Due-Diligence Eval Harness and Scoreboard

**User Story:** As a system operator, I want the LLM's annotation quality measured
against labeled data and tracked over time, so that I can detect degradation and
decide whether to promote it in v2.

**Requirement:** The due-diligence service SHALL carry an eval harness operating on
30–50 hand-labeled past fights, gating at precision ≥ 0.80 and recall ≥ 0.60. A
running verdict scoreboard SHALL track resolved outcomes of CONFIRM'd vs. VETO'd
signals.

#### Acceptance Criteria

1. WHEN the eval harness runs, THE Eval_Harness SHALL invoke the due-diligence runner
   on every fight in the labeled test set and compare verdicts to ground-truth labels.
2. WHEN precision falls below 0.80 or recall falls below 0.60 on the labeled set, THE
   Eval_Harness SHALL fail with a clear diagnostic and the measured metrics.
3. WHEN a new labeled fight is added to the test set, THE Eval_Harness SHALL require
   no code changes — only the fixture file update.
4. WHEN a fight referenced by a CONFIRM'd or VETO'd verdict resolves, THE
   Verdict_Scoreboard SHALL update the running tally of outcomes (correct direction,
   incorrect direction, push).
5. WHEN a user queries the scoreboard, THE Verdict_Scoreboard SHALL report per-verdict
   type: count, win rate, and mean absolute mismatch at signal time.
6. WHEN the prompt or model version changes, THE Eval_Harness SHALL be re-run before
   the new version is used in production report runs.
7. WHEN the eval harness runs, THE Eval_Harness SHALL use the same frozen Pydantic
   schema and runner code as production, with no test-specific overrides to the
   prompt or parsing logic.

---

**Bucket E — Persistence and Health** (Requirement 11)

### Requirement 11: Append-Only Persistence and Staleness Reporting

**User Story:** As a system operator, I want report runs and signals immutable and
health reporting surfaced, so that I can audit history and detect pipeline staleness.

**Requirement:** `report_runs` and `paper_signals` tables SHALL be append-only. The
health command SHALL report capture freshness, link coverage, and report-run recency.

#### Acceptance Criteria

1. WHEN a report run completes, THE Report_Storage module SHALL write one `report_runs`
   row and SHALL NOT update or delete any prior row in `report_runs` or
   `paper_signals`.
2. WHEN the same report parameters are run twice, THE Report_Storage module SHALL
   create a second `report_runs` row with a new `report_run_id`.
3. WHEN the health command executes, THE Health_Reporter SHALL emit
   `latest_report_run_at`, `total_paper_signals`, `flagged_signal_count`,
   `unresolved_link_count`, and `latest_capture_at`.
4. WHEN `latest_capture_at` exceeds the configured staleness threshold, THE
   Health_Reporter SHALL emit a `STALE_CAPTURE` warning.
5. WHEN unresolved links exist for upcoming fights, THE Health_Reporter SHALL emit an
   `UNRESOLVED_LINKS` warning with the count.
6. WHEN a `paper_signals` write is attempted with a `report_run_id` not present in
   `report_runs`, THE Report_Storage module SHALL reject the write.

## Standing Decisions with Named Fallbacks

1. **Magnitude gate multiplier `k = 2`.** If fewer than 5% of signals are flagged
   across three consecutive UFC events, evaluate whether the bucket errors have
   decreased and consider reducing `k` to 1.5 before changing model or features.

2. **LLM annotate-only mode.** If the verdict scoreboard shows precision ≥ 0.85 and
   recall ≥ 0.70 after 100+ resolved signals with VETO verdicts, evaluate the v2
   residual-overlay promotion path before granting suppression authority.

3. **Sparse-history threshold = 3 fights.** If the history-depth-stratified metrics
   (D37) show no meaningful calibration gap between sparse and non-sparse fighters
   after two full UFC events of data, evaluate raising the threshold to 5.

4. **Post-signal snapshot cadence (T+1h/4h/24h/fight).** If more than 30% of T+1h
   snapshots are MISSED due to market closure, evaluate removing the +1h tier before
   adding more frequent captures.
