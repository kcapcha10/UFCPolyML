# Implementation Plan: Mismatch Report — Paper-Signal Pipeline (v1 Scope)

## Overview

This plan implements the mismatch report pipeline in wave-based order. Early waves
(0–2) run against STUBBED model probabilities and a FIXTURE bucket artifact, so they
parallelize with the feature-engine and model-and-eval specs. Real-data integration
(model inference → live bucket table) happens in Wave 3, after the ME spec delivers
calibrated probabilities and the stratified reliability artifact.

Each task is sized for one subagent session. Test tasks use TDD: test file is created
first, implementation makes it pass. All tasks respect the MR- requirement namespace
and cross-reference ME- and FE- dependencies where applicable.

The `report/` package does not exist yet. Wave 0 creates the package structure.

## Tasks

---

### Wave 0 — Scaffolding and Schemas (no external dependencies)

- [ ] MR-0. Wave 0: Package Structure and Data Models (LOE: 1 day)
  **Phase LOE: 1 day**

  - [ ] MR-0.1 Create `src/ufc_edge/report/` package with `__init__.py`, `schemas.py`, `storage.py`
    - Define frozen Pydantic models: `MarketFightLink`, `PaperSignal`, `ReportRun`, `DueDiligenceVerdict`, `ChecklistFindings`, `Finding`, `PostSignalSnapshot`, `GateResult`, `SparseHistoryResult`, `ScoreboardEntry`
    - All models use `_FrozenModel` base (existing pattern in `data/schemas.py`)
    - `DueDiligenceVerdict` enforces `len(evidence_urls) >= 1` via `@field_validator`
    - `GateResult.verdict` is a `StrEnum`: `FLAGGED`, `WITHIN_NOISE`, `NO_BUCKET_DATA`
    - `match_status` is a `StrEnum`: `MATCHED`, `NO_CANDIDATE`, `MULTIPLE_CANDIDATES`, `INVALID_INPUT`, `MISSING_SNAPSHOT`
    - Files: `src/ufc_edge/report/__init__.py`, `src/ufc_edge/report/schemas.py`
    - _Requirements: MR-1.1, MR-7.4, MR-9.2_

  - [ ] MR-0.2 Implement `report.storage` DDL and write functions
    - `REPORT_DDL`: all 7 CREATE TABLE statements (market_fight_links, report_runs, paper_signals, post_signal_snapshots, due_diligence_verdicts, due_diligence_runs, verdict_scoreboard)
    - Write functions: `write_link`, `write_report_run`, `write_paper_signal`, `write_verdict`, `write_post_signal_snapshot`, `write_scoreboard_entry`
    - All writes are append-only; `write_link` raises on attempt to overwrite a MATCHED row
    - Register DDL in `data/storage.py` aggregation (same pattern as existing packages)
    - Files: `src/ufc_edge/report/storage.py`, modify `src/ufc_edge/data/storage.py` (DDL import)
    - _Requirements: MR-11.1, MR-11.2, MR-11.6_

  - [ ] MR-0.3 Write unit tests for schemas and storage
    - Schema validation: required fields, frozen immutability, enum constraints, evidence URL length validator
    - Storage: write round-trip, append-only enforcement, MATCHED-row immutability guard, FK constraint on paper_signals → report_runs
    - Files: `tests/report/test_schemas.py`, `tests/report/test_storage.py`, `tests/report/__init__.py`
    - _Requirements: MR-1.2, MR-9.2, MR-11.1, MR-11.6_

  - [ ] MR-0.4 Create Hydra configuration `configs/report/default.yaml`
    - Gate config: `k: 2.0`, bucket definitions
    - Sparse-history: `min_prior_ufc_fights_threshold: 3`
    - Due-diligence: model name, prompt version, search provider, timeout
    - Post-signal: offsets, fight-time flag, retries
    - Staleness: thresholds
    - Files: `configs/report/default.yaml`
    - _Requirements: MR-6.4, MR-8.3_

---

### Wave 1 — Entity Resolution (no external dependencies)

- [ ] MR-1. Wave 1: Name Matching and Link Resolution (LOE: 2 days)
  **Phase LOE: 2 days**

  - [ ] MR-1.1 Write tests for name normalization and link resolution
    - Test `normalize_name`: diacritics (Jiří → jiri), hyphens, apostrophes, suffixes (Jr., III), double spaces, case
    - Test `resolve_links`: exact-one-match → MATCHED, zero-match → NO_CANDIDATE, multi-match → MULTIPLE_CANDIDATES, invalid fighter_url → INVALID_INPUT, no snapshot → MISSING_SNAPSHOT
    - Test idempotency: re-running with same inputs produces no new rows
    - Test immutability: attempting to overwrite MATCHED raises
    - Fixtures: fake DuckDB with order_book_snapshots rows and fighters rows
    - Files: `tests/report/test_matching.py`
    - _Requirements: MR-2.1, MR-2.2, MR-2.3, MR-2.4, MR-2.5, MR-2.6, MR-2.7, MR-1.2, MR-1.3_

  - [ ] MR-1.2 Implement `report.matching` module
    - `normalize_name(raw: str) -> str`: unicodedata.normalize('NFD') + combining-mark regex + lower + punctuation strip + whitespace collapse
    - `resolve_links(fights, as_of, conn)`: check existing links, run name matching for missing, write results
    - `get_unresolved(conn)`: SELECT where match_status NOT IN ('MATCHED') AND reviewed_by IS NULL
    - Name matching queries `order_book_snapshots` for active UFC markets (join with MarketInfo or question text)
    - Files: `src/ufc_edge/report/matching.py`
    - _Requirements: MR-1.1, MR-1.4, MR-1.5, MR-2.1, MR-2.2, MR-2.3, MR-2.4, MR-2.5, MR-2.6, MR-2.7_

  - [ ] MR-1.3 Write tests and implement human confirmation CLI
    - Test: display groups by event date, confirm updates status, skip leaves unchanged, invalid token rejected
    - Implement `report.confirm_cli`: `display_unresolved`, `confirm_link`, CLI entry point with argparse
    - Files: `tests/report/test_confirm_cli.py`, `src/ufc_edge/report/confirm_cli.py`
    - _Requirements: MR-3.1, MR-3.2, MR-3.3, MR-3.4, MR-3.5_

  - [ ] MR-1.4 Implement capture schema extension (bid/ask/depth columns)
    - Add `best_bid`, `best_ask`, `best_bid_size`, `best_ask_size` fields to `OrderBookSnapshot` Pydantic model
    - Extend `POLYMARKET_DDL` ALTER TABLE or recreate with new columns (NULL default for existing rows)
    - Update `capture.py` to extract `bids[0].price`, `bids[0].size`, `asks[0].price`, `asks[0].size`
    - Write unit test: snapshot with populated book extracts correctly; empty book → NULL
    - Files: modify `src/ufc_edge/data/polymarket/schemas.py`, `storage.py`, `capture.py`; `tests/data/test_capture_extension.py`
    - _Requirements: MR-4.1, MR-4.2, MR-4.4_

---

### Wave 2 — Mismatch Computation, LLM Service, Scheduler (stubbed model input)

- [ ] MR-2. Wave 2: Mismatch Logic and Due-Diligence Service (LOE: 3 days)
  **Phase LOE: 3 days**

  Depends on: Wave 0 (schemas, storage), Wave 1 (link table for integration tests)

  - [ ] MR-2.1 Write tests for mismatch computation and gate logic
    - Test `compute_mismatch`: pure arithmetic, positive and negative mismatches
    - Test `apply_gate`: FLAGGED when |mismatch| > k × error; WITHIN_NOISE below; NO_BUCKET_DATA for empty bucket
    - Test bucket boundary assignment (probability on boundary → lower bucket)
    - Test edge cases: p_model outside [0.1, 0.9] → nearest bucket
    - Fixture: `BucketArtifact` with known values
    - Files: `tests/report/test_mismatch.py`
    - _Requirements: MR-6.1, MR-6.2, MR-6.3, MR-6.5, MR-7.1, MR-7.3_

  - [ ] MR-2.2 Implement `report.mismatch` module
    - `compute_mismatch(p_model, p_market_mid) -> float`
    - `apply_gate(mismatch, p_model, bucket_artifact, k) -> GateResult`
    - `tag_sparse_history(fighter_a_url, fighter_b_url, fight_url, conn, threshold) -> SparseHistoryResult`
    - `assign_bucket(p_model, buckets) -> BucketEntry | None`
    - Bucket logic: boundaries follow `[lower, upper)` except final bucket `[0.7, 0.9]`
    - Files: `src/ufc_edge/report/mismatch.py`
    - _Requirements: MR-6.1, MR-6.2, MR-6.3, MR-6.4, MR-6.5, MR-7.1, MR-7.2, MR-7.3, MR-7.4_

  - [ ] MR-2.3 Write tests for sparse-history tagging
    - Test: fighter with 2 prior fights + fighter with 5 → min=2, sparse=True (threshold 3)
    - Test: both fighters with 4+ → sparse=False
    - Test: count excludes current fight and same-card fights
    - Fixture: DuckDB with known fight history
    - Files: `tests/report/test_sparse_history.py`
    - _Requirements: MR-8.1, MR-8.2, MR-8.4, MR-8.5_

  - [ ] MR-2.4 Write tests and implement due-diligence runner (with mocked LLM)
    - Test idempotency: second call returns persisted verdict without LLM invocation
    - Test schema validation: malformed LLM response raises, null verdict recorded with error
    - Test logging: prompt_version, model_name, model_version, invoked_at all persisted
    - Test evidence_urls: at least one required per verdict
    - Mock: LLM client returns fixture response; search returns fixture URLs
    - Implement `report.due_diligence.py`: `run_due_diligence`, search integration, schema validation, idempotency check
    - Files: `tests/report/test_due_diligence.py`, `src/ufc_edge/report/due_diligence.py`
    - _Requirements: MR-9.1, MR-9.2, MR-9.3, MR-9.4, MR-9.5, MR-9.6, MR-9.7_

  - [ ] MR-2.5 Write tests and implement due-diligence eval harness
    - Fixture: `tests/fixtures/due_diligence_labels.json` with 30–50 entries (initially stubbed at 5 for development; full set is a human-provided deliverable)
    - Test: harness computes precision/recall correctly against known labels
    - Test: gate_check fails when P < 0.80 or R < 0.60
    - Test: harness uses production runner code path (no overrides)
    - Implement `report.due_diligence_eval.py`: `run_eval`, `gate_check`
    - Files: `tests/report/test_due_diligence_eval.py`, `src/ufc_edge/report/due_diligence_eval.py`, `tests/fixtures/due_diligence_labels.json`
    - _Requirements: MR-10.1, MR-10.2, MR-10.3, MR-10.6, MR-10.7_

  - [ ] MR-2.6 Write tests and implement post-signal snapshot scheduler
    - Test: FLAGGED signal enqueues 4 snapshots (1h, 4h, 24h, fight-time)
    - Test: missing event_start_time → fight-time SKIPPED
    - Test: failed capture → MISSED with reason
    - Test: successful capture writes full top-of-book fields
    - Implement `report.snapshots.py`: `schedule_snapshots`, `execute_pending_snapshots`
    - Files: `tests/report/test_snapshots.py`, `src/ufc_edge/report/snapshots.py`
    - _Requirements: MR-5.1, MR-5.2, MR-5.3, MR-5.4, MR-5.5_

  - [ ] MR-2.7 Write tests and implement verdict scoreboard
    - Test: CONFIRM verdict + fight resolves correctly → outcome_correct = True
    - Test: VETO verdict + fight resolves in signal direction → outcome_correct = False
    - Test: query returns per-verdict counts, win rates, mean mismatch
    - Implement `report.scoreboard.py`: `update_scoreboard`, `query_scoreboard`
    - Files: `tests/report/test_scoreboard.py`, `src/ufc_edge/report/scoreboard.py`
    - _Requirements: MR-10.4, MR-10.5_

---

### Wave 3 — Report Runner Integration (blocks on ME calibrated model + bucket artifact)

- [ ] MR-3. Wave 3: End-to-End Report Pipeline (LOE: 2 days)
  **Phase LOE: 2 days**

  Depends on: Wave 2 (all components); ME spec delivers calibrated model + `BucketArtifact` via MLflow

  - [ ] MR-3.1 Write integration tests for report runner with stubbed model
    - Test: full pipeline produces report_run + paper_signals rows for fixture input
    - Test: matched bout gets full columns; unmatched bout gets null probability fields
    - Test: FLAGGED signal triggers due-diligence invocation
    - Test: re-run produces new report_run_id (append-only)
    - Test: missing bucket artifact fails the run before writing
    - Fixture: stubbed MLflow run with known probabilities and bucket artifact
    - Files: `tests/report/test_runner_integration.py`
    - _Requirements: MR-6.1, MR-7.2, MR-9.1, MR-11.1, MR-11.2_

  - [ ] MR-3.2 Implement `report.runner` orchestrator
    - `run_report(as_of, mlflow_run_id, config, conn)`: create report_run, iterate bouts, resolve links, load model, compute mismatch, apply gate, tag sparse history, invoke due-diligence for FLAGGED, write paper_signals, schedule post-signal snapshots
    - Load model + calibrator + bucket artifact from MLflow (cross-spec join: ME inference + ME reliability artifact)
    - Symmetric inference call: pass (fighter_a_url, fighter_b_url), receive calibrated p_model
    - Files: `src/ufc_edge/report/runner.py`
    - _Requirements: MR-1.5, MR-4.3, MR-6.1, MR-6.2, MR-7.1, MR-8.1, MR-9.1, MR-11.1_

  - [ ] MR-3.3 Wire Makefile targets
    - `make report AS_OF=... RUN_ID=...` → `python -m ufc_edge.report.runner`
    - `make match-review` → `python -m ufc_edge.report.confirm_cli`
    - `make due-diligence-eval` → `python -m ufc_edge.report.due_diligence_eval`
    - `make verdict-scoreboard` → `python -m ufc_edge.report.scoreboard`
    - Files: modify `Makefile`
    - _Requirements: MR-11.3_

  - [ ] MR-3.4 Implement health extensions for report pipeline
    - Extend `ops.health` to include: `latest_report_run_at`, `total_paper_signals`, `flagged_signal_count`, `unresolved_link_count`, staleness warnings
    - Files: modify `src/ufc_edge/ops/health.py` (or create if ops/ doesn't exist yet), `tests/ops/test_health_report.py`
    - _Requirements: MR-11.3, MR-11.4, MR-11.5_

---

### Wave 4 — Verification and Polish

- [ ] MR-4. Wave 4: Final Verification (LOE: 1 day)
  **Phase LOE: 1 day**

  Depends on: Wave 3 (full pipeline)

  - [ ] MR-4.1 Determinism regression test
    - Run report pipeline twice with identical seeded UUIDs and fixed inputs
    - Assert paper_signals rows are byte-identical (except created_at, which uses frozen time)
    - Files: `tests/report/test_determinism.py`
    - _Requirements: MR-7.3, MR-11.2_

  - [ ] MR-4.2 Leakage test: no market features reach model
    - Assert the model inference path receives only fighter_url pairs and returns calibrated probabilities
    - Assert no column in the model's feature schema contains "bid", "ask", "market", "price", "spread"
    - Files: `tests/report/test_leakage.py`
    - _Requirements: MR-6.1 (model−market separation per D24)_

  - [ ] MR-4.3 Run `make lint` and `make test` — full green
    - All new tests pass
    - ruff clean
    - No type errors
    - _Requirements: all MR-*_

  - [ ] MR-4.4 Append-only stress test
    - Run 3 consecutive report executions; assert 3 report_runs rows, 3× bout_count paper_signals
    - Assert no row from run 1 is modified by run 2 or 3
    - Files: `tests/report/test_append_only.py`
    - _Requirements: MR-11.1, MR-11.2_

---

## Module Ownership Table

| Module | Owner | Creates/Modifies |
|---|---|---|
| `report/__init__.py` | MR spec | new file |
| `report/schemas.py` | MR spec | new file |
| `report/storage.py` | MR spec | new file |
| `report/matching.py` | MR spec | new file |
| `report/confirm_cli.py` | MR spec | new file |
| `report/mismatch.py` | MR spec | new file |
| `report/due_diligence.py` | MR spec | new file |
| `report/due_diligence_eval.py` | MR spec | new file |
| `report/snapshots.py` | MR spec | new file |
| `report/scoreboard.py` | MR spec | new file |
| `report/runner.py` | MR spec | new file |
| `data/storage.py` | MR spec (additive) | register REPORT_DDL |
| `data/polymarket/schemas.py` | MR spec (additive) | add bid/ask/depth fields |
| `data/polymarket/storage.py` | MR spec (additive) | extend DDL + INSERT |
| `data/polymarket/capture.py` | MR spec (additive) | extract top-of-book |
| `configs/report/default.yaml` | MR spec | new file |
| `Makefile` | MR spec (additive) | add report/match-review/eval/scoreboard targets |
| `tests/report/*` | MR spec | new directory + files |
| `tests/fixtures/due_diligence_labels.json` | MR spec | new fixture (stub; human fills) |

## Notes

- **Stubbed model probabilities:** Waves 0–2 use hardcoded `p_model` values in test
  fixtures (e.g., 0.65, 0.72). The ME spec's calibrated inference function is mocked
  at the integration boundary. This allows full MR development in parallel.
- **Bucket artifact fixture:** A `BucketArtifact` fixture with realistic bucket
  entries (n=50–300, error=0.03–0.08) is used in Waves 0–2. The real artifact comes
  from ME's stratified reliability report after training completes.
- **Due-diligence label set:** The initial fixture contains 5 stub entries for
  development. The full 30–50 hand-labeled set is a human deliverable before the eval
  harness gates production use. The eval harness code works on any fixture length ≥ 1.
- **LLM in tests:** All LLM-touching tests mock the client. No test makes a real API
  call. The eval harness in CI also uses mocked responses; the real eval is a manual
  `make due-diligence-eval` gate.
- **Cross-spec joins:**
  - MR consumes `BucketArtifact` from ME (MLflow logged artifact)
  - MR consumes calibrated `p_model` from ME inference function
  - MR consumes `market_fight_links` (self-produced) for ME's market Brier eval (D28)
- **No convergence-exit evaluation in v1** (D36): post-signal snapshots are collected
  for future v2 study; v1 evaluates hold-to-resolution only.

## LOE Summary

Total estimated effort: **9 days** across 5 waves. Waves 0–2 (6 days) run in
parallel with FE and ME specs. Wave 3 (2 days) blocks on ME delivering a calibrated
model and bucket artifact via MLflow. Wave 4 (1 day) is final verification.

## Timeline Summary

| Wave | Tasks | Parallel with | Cross-spec dependency |
|---|---|---|---|
| Wave 0 | MR-0.1–0.4: schemas, storage, DDL, config | FE Wave 0, ME Wave 0 | None |
| Wave 1 | MR-1.1–1.4: matching, confirmation, capture extension | FE Wave 1, ME Wave 1 | None |
| Wave 2 | MR-2.1–2.7: mismatch, gate, sparse, LLM, scheduler, scoreboard | FE Waves 2–3, ME Waves 1–2 | Stubbed bucket artifact (ME delivers real artifact in ME Wave 3+) |
| Wave 3 | MR-3.1–3.4: runner integration, Makefile, health | — | Blocks on ME calibrated model + BucketArtifact (ME training complete) |
| Wave 4 | MR-4.1–4.4: determinism, leakage, lint, append-only | — | Blocks on Wave 3 |

## Task Dependency Graph

```text
Wave 0 ─────────────────────────────────────────────────────
  MR-0.1 schemas ──┬── MR-0.2 storage ──┬── MR-0.3 tests
                   │                     │
                   └── MR-0.4 config ────┘

Wave 1 (depends on Wave 0) ────────────────────────────────
  MR-1.1 match tests ── MR-1.2 matching ── MR-1.3 confirm CLI
                                                │
  MR-1.4 capture extension ────────────────────┘ (independent)

Wave 2 (depends on Waves 0+1) ─────────────────────────────
  MR-2.1 mismatch tests ── MR-2.2 mismatch impl
  MR-2.3 sparse tests ─────┘ (shares mismatch module)
  MR-2.4 DD tests ── MR-2.4 DD impl  (independent track)
  MR-2.5 DD eval harness             (depends on MR-2.4)
  MR-2.6 snapshot scheduler          (independent track)
  MR-2.7 scoreboard                  (independent track)

Wave 3 (depends on Wave 2 + ME calibrated model) ──────────
  MR-3.1 integration tests ── MR-3.2 runner ── MR-3.3 Makefile
                                                     │
  MR-3.4 health extensions ─────────────────────────┘

Wave 4 (depends on Wave 3) ────────────────────────────────
  MR-4.1 determinism ── MR-4.2 leakage ── MR-4.3 lint+test ── MR-4.4 append-only
```
