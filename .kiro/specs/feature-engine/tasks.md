# Implementation Plan: Feature Engine — V1 (Point-in-Time Replay)

## Overview

The feature engine is built in six waves. Waves are designed for parallel execution
by independent subagents: no two tasks within a wave share files or have dependency
edges. Each task pairs implementation with test-first development; leakage and
property tests are first-class verification tasks, not afterthoughts.

The dependency spine: Wave 0 (protocols, configs, registry scaffolding) has no
prerequisites. Wave 1a (state components without cross-dependencies) depends on
Wave 0 protocols. Wave 1b (CommonOpponentIndex) depends on Wave 1a tasks FE-1.1
and FE-1.3. Wave 2 (emitters) depends on Wave 0 contracts + Waves 1a/1b components.
Wave 3 (replay engine + storage) depends on Waves 0–2. Wave 4 (leakage suite +
property tests) depends on Wave 3 (full replay available). Wave 5 (scraper extension
for §5a + Kaggle protocol) is independent of Waves 1–4 on the code side but §5a
emitter activation depends on Wave 5 completion.

Cross-spec dependency: model-and-eval Wave 2+ blocks on FE Wave 3 (materialized
`features_v1` table). ME Waves 0–1 (assembler interface, metrics, calibrators) can
run parallel to all FE waves using a fixture feature table.

## Tasks

---

### Wave 0 — Protocols, Registry, Configuration

- [ ] FE-0.1 Define protocol contracts and data models
  - Create `src/ufc_edge/features/contracts.py`: `StateComponent` protocol,
    `FeatureEmitter` protocol, `EmitContext` frozen dataclass, `FightOutcomeView`
    frozen dataclass, `FrozenState` base, `HistoricalFight`, `EventTick`, `FeatureRow`
  - Create `src/ufc_edge/features/__init__.py` with public exports
  - Test: verify EmitContext is frozen (assignment raises), FightOutcomeView excludes
    market fields, protocol signatures match spec
  - _Requirements: FE-2.1, FE-2.2, FE-2.3, FE-2.4, FE-2.8_
  - Depends on: none
  - Files: `src/ufc_edge/features/contracts.py`, `src/ufc_edge/features/__init__.py`,
    `tests/features/test_contracts.py`

- [ ] FE-0.2 Implement Feature Registry
  - Create `src/ufc_edge/features/registry.py`: family declarations, column→type map,
    `schema()` method, startup validation (duplicates, types, families), ordered family
    iteration
  - Test: duplicate columns raise RegistryError, missing family raises, invalid type
    raises, `schema()` returns expected dict
  - _Requirements: FE-11.1, FE-11.2, FE-11.3, FE-11.5, FE-11.7_
  - Depends on: none
  - Files: `src/ufc_edge/features/registry.py`, `tests/features/test_registry.py`

- [ ] FE-0.3 Implement feature versioning and source-hash guard
  - Create `src/ufc_edge/features/versioning.py`: `compute_source_hash()` over
    `src/ufc_edge/features/**/*.py`, `check_version_integrity()`,
    `features_version_manifest.json` read/write, `FEATURE_VERSION` constant
  - Create initial `features_version_manifest.json` in project root
  - Test: tampered source fails integrity check, manifest format is stable, hash is
    deterministic across runs
  - _Requirements: FE-3.1, FE-3.2, FE-3.4, FE-3.7, FE-3.8_
  - Depends on: none
  - Files: `src/ufc_edge/features/versioning.py`, `features_version_manifest.json`,
    `tests/features/test_versioning.py`

- [ ] FE-0.4 Create graph configuration
  - Create `configs/graph.yaml` with all Elo, Glicko-2, PageRank, and common-opponent
    parameter slots per design; values marked `TODO(human)` where unspecified
  - Test: OmegaConf loads without error, all expected keys present
  - _Requirements: FE-5.8, FE-6.6, FE-7.7, FE-7.8_
  - Depends on: none
  - Files: `configs/graph.yaml`, `tests/features/test_config.py`

---

### Wave 1a — State Components (independent)

- [ ] FE-1.1 Implement EloTracker StateComponent
  - Create `src/ufc_edge/features/components/elo.py`: variable K-factor, method bonus,
    recency weight, inactivity decay toward 1500, K=0 for injury stoppage, K×0.1 for
    DQ, debut at 1500, `freeze()` returns frozen `EloRecord` per fighter, `history`
    tracks last N ratings for trajectory
  - Test: debut = 1500, KO win increases more than decision, injury = no change,
    inactivity decays, freeze is immutable, DQ applies 0.1 multiplier
  - _Requirements: FE-5.1, FE-5.2, FE-5.3, FE-5.4, FE-5.5, FE-5.6, FE-5.7_
  - Depends on: FE-0.1, FE-0.4
  - Files: `src/ufc_edge/features/components/__init__.py`,
    `src/ufc_edge/features/components/elo.py`, `tests/features/test_elo.py`

- [ ] FE-1.2 Implement Glicko2Tracker StateComponent
  - Create `src/ufc_edge/features/components/glicko2.py`: standard Glicko-2 update,
    RD growth with inactivity, injury-stoppage neutral, configurable τ and period
  - Test: debut RD=350, win reduces RD, inactivity increases RD, injury no update,
    freeze is immutable
  - _Requirements: FE-6.1, FE-6.2, FE-6.3, FE-6.5, FE-6.7, FE-6.8_
  - Depends on: FE-0.1, FE-0.4
  - Files: `src/ufc_edge/features/components/glicko2.py`,
    `tests/features/test_glicko2.py`

- [ ] FE-1.3 Implement PageRankGraph StateComponent
  - Create `src/ufc_edge/features/components/pagerank.py`: directed graph loser→winner,
    edge weights (finish bonus, recency decay, early finish), networkx PageRank with
    α=0.85, isolated node = global minimum
  - Test: single fight creates edge, PageRank converges, isolated node gets minimum,
    recency decay reduces old edge weight, freeze is immutable
  - _Requirements: FE-7.1, FE-7.2, FE-7.3, FE-7.7_
  - Depends on: FE-0.1, FE-0.4
  - Files: `src/ufc_edge/features/components/pagerank.py`,
    `tests/features/test_pagerank.py`

- [ ] FE-1.5 Implement CareerAccumulator StateComponent
  - Create `src/ufc_edge/features/components/career.py`: win/loss counts, finish
    counts by type, streak tracking, per-window fight counts (12mo, 3yr, 5yr),
    last fight date and method, debut tracking, weight-class history
  - Test: counts accumulate correctly, streaks flip on loss, window counts respect
    dates, freeze is immutable
  - _Requirements: FE-8.2, FE-8.3_
  - Depends on: FE-0.1
  - Files: `src/ufc_edge/features/components/career.py`,
    `tests/features/test_career.py`

- [ ] FE-1.6 Implement RollingStatsAccumulator StateComponent
  - Create `src/ufc_edge/features/components/rolling_stats.py`: per-fighter deque of
    `FightStats` (from fight_totals), configurable window size, rolling averages for
    all §5 output metrics
  - Test: rolling average with full window, partial window, empty window (None),
    variance computation, freeze is immutable
  - _Requirements: FE-8.4, FE-8.5_
  - Depends on: FE-0.1
  - Files: `src/ufc_edge/features/components/rolling_stats.py`,
    `tests/features/test_rolling_stats.py`

- [ ] FE-1.7 Implement WeightClassTracker StateComponent
  - Create `src/ufc_edge/features/components/weight_class.py`: per-fighter weight
    class history, tracks migrations, computes "first class" (natural, UFCStats-only),
    top-quartile calculations for weight bully
  - Test: detects class change, tracks fight count per class, identifies large-for-class,
    freeze is immutable
  - _Requirements: FE-8.6 (implicit from §8a/8b features)_
  - Depends on: FE-0.1
  - Files: `src/ufc_edge/features/components/weight_class.py`,
    `tests/features/test_weight_class.py`

---

### Wave 1b — State Components (depends on Wave 1a)

- [ ] FE-1.4 Implement CommonOpponentIndex StateComponent
  - Create `src/ufc_edge/features/components/common_opponents.py`: per-fighter fight
    history with timestamps, 3-year lookback windowing, intersection computation,
    quality weighting by Elo and PageRank at fight time, recency decay
  - Test: identifies shared opponents within window, excludes outside window, weights
    by quality, returns empty when no common, freeze is immutable
  - _Requirements: FE-7.4, FE-7.5, FE-7.6, FE-7.8_
  - Depends on: FE-0.1, FE-0.4, FE-1.1 (needs Elo for quality weights), FE-1.3
    (needs PageRank for quality weights)
  - Files: `src/ufc_edge/features/components/common_opponents.py`,
    `tests/features/test_common_opponents.py`

---

### Wave 2 — Feature Emitters

- [ ] FE-2.1 Implement PhysicalEmitter
  - Create `src/ufc_edge/features/emitters/physical.py`: height_cm, reach_cm,
    reach_to_height_ratio, stance, age_at_fight (from dob + event_date), weight_class
  - Test: known profile → exact values, missing dob → None for age, missing reach →
    None for ratio
  - _Requirements: FE-8.1_
  - Depends on: FE-0.1, FE-0.2
  - Files: `src/ufc_edge/features/emitters/__init__.py`,
    `src/ufc_edge/features/emitters/physical.py`, `tests/features/test_emitter_physical.py`

- [ ] FE-2.2 Implement ActivityEmitter
  - Create `src/ufc_edge/features/emitters/activity.py`: days_since_last_fight,
    fights_last_12mo/3yr/5yr, total_ufc_fights, last_fight_injury_stoppage,
    age_x_inactivity, inactivity_tier
  - Test: tier boundaries (179d=0, 180d=1, 365d=1, 366d=2, 730d=2, 731d=3), debut
    fighter → None for days_since_last
  - _Requirements: FE-8.2_
  - Depends on: FE-0.1, FE-0.2, FE-1.5
  - Files: `src/ufc_edge/features/emitters/activity.py`,
    `tests/features/test_emitter_activity.py`

- [ ] FE-2.3 Implement RecordEmitter
  - Create `src/ufc_edge/features/emitters/record.py`: win_pct_all, win_pct_last3/5,
    current_streak, finish/decision pcts, ufc_win_pct, is_ufc_debut,
    debut_opponent_ufc_experience/win_pct, contender_series_win (from event name)
  - Test: exact pct calculations, streak sign convention, debut flag logic, 0 fights
    → None for pcts
  - _Requirements: FE-8.3_
  - Depends on: FE-0.1, FE-0.2, FE-1.5
  - Files: `src/ufc_edge/features/emitters/record.py`,
    `tests/features/test_emitter_record.py`

- [ ] FE-2.4 Implement FinishingEmitter
  - Create `src/ufc_edge/features/emitters/finishing.py`: finish_rate, ko_rate,
    submission_rate, early_finish_rate, avg/var fight duration, has_ever_been_finished,
    times_finished_by_ko/sub, has_been_finished_r1, never_been_finished,
    never_been_finished_x_opp_finish_rate
  - Test: zero wins → 0.0 rates, interaction term computation, variance with <3
    fights → None
  - _Requirements: FE-8.4 (partial)_
  - Depends on: FE-0.1, FE-0.2, FE-1.5
  - Files: `src/ufc_edge/features/emitters/finishing.py`,
    `tests/features/test_emitter_finishing.py`

- [ ] FE-2.5 Implement OutputEmitter
  - Create `src/ufc_edge/features/emitters/output.py`: all §5 rolling stats from
    RollingStatsAccumulator, damage_ratio, grappling_dominance, control_time_per_fight,
    distance/clinch/ground strike pcts, target pcts, knockdown_rate
  - Test: rolling window correctness, damage_ratio formula, zero denominator → None
  - _Requirements: FE-8.4_
  - Depends on: FE-0.1, FE-0.2, FE-1.6
  - Files: `src/ufc_edge/features/emitters/output.py`,
    `tests/features/test_emitter_output.py`

- [ ] FE-2.6 Implement CardPositionEmitter (gated)
  - Create `src/ufc_edge/features/emitters/card_position.py`: §5a features, checks
    bout_order availability; emits None + logs warning when unavailable; implements
    main-card heuristic (top-5 bouts) when available; requires 3+ UFC fights for
    variance features
  - Test: bout_order=None → all None with warning, bout_order present → correct
    split, <3 fights → None for variance
  - _Requirements: FE-8.6, FE-9.6, FE-9.7_
  - Depends on: FE-0.1, FE-0.2, FE-1.6
  - Files: `src/ufc_edge/features/emitters/card_position.py`,
    `tests/features/test_emitter_card_position.py`

- [ ] FE-2.7 Implement ExperienceEmitter
  - Create `src/ufc_edge/features/emitters/experience.py`: title_fight_experience,
    has_been_champion, days_as_champion, main_event_experience, five_round_experience,
    five_round_win_pct
  - Test: championship detection from outcome method, five-round from time_format,
    zero experience → 0
  - _Requirements: FE-8.4 (partial)_
  - Depends on: FE-0.1, FE-0.2, FE-1.5
  - Files: `src/ufc_edge/features/emitters/experience.py`,
    `tests/features/test_emitter_experience.py`

- [ ] FE-2.8 Implement WeightDominanceEmitter
  - Create `src/ufc_edge/features/emitters/weight.py`: §8a migration features
    (is_weight_class_change, direction, fights/win_pct at current/prior class), §8b
    weight bully (is_large_for_class, grappling_utilization_rate, weight_bully_score
    as product)
  - Test: class change detection, direction sign, top-quartile logic, product term
  - _Requirements: FE-8.4 (partial)_
  - Depends on: FE-0.1, FE-0.2, FE-1.7
  - Files: `src/ufc_edge/features/emitters/weight.py`,
    `tests/features/test_emitter_weight.py`

- [ ] FE-2.9 Implement GraphEmitter
  - Create `src/ufc_edge/features/emitters/graph.py`: elo_rating, elo_trajectory_last5,
    elo_peak, elo_current_vs_peak, glicko2_rating, glicko2_rd, pagerank_score,
    n_common_opponents, common_opp_score_a/b/delta, common_opp_win_rates
  - Test: reads frozen component state correctly, trajectory slope calculation,
    None when <5 for trajectory
  - _Requirements: FE-5.6, FE-6.4, FE-7.5_
  - Depends on: FE-0.1, FE-0.2, FE-1.1, FE-1.2, FE-1.3, FE-1.4
  - Files: `src/ufc_edge/features/emitters/graph.py`,
    `tests/features/test_emitter_graph.py`

- [ ] FE-2.10 Implement MatchupEmitter
  - Create `src/ufc_edge/features/emitters/matchup.py`: all §10 deltas (A−B convention),
    §10a wrestler_score/submission_score/deltas/grappling_type_mismatch, §10b style
    interactions (striker_vs_grappler, pressure_vs_counter, pace_mismatch_score,
    southpaw_orthodox_history)
  - Test: delta signs flip with orientation, known inputs → exact delta values,
    type mismatch detection
  - _Requirements: FE-8.7, FE-8.8_
  - Depends on: FE-0.1, FE-0.2, FE-1.5, FE-1.6
  - Files: `src/ufc_edge/features/emitters/matchup.py`,
    `tests/features/test_emitter_matchup.py`

- [ ] FE-2.11 Implement RematchEmitter
  - Create `src/ufc_edge/features/emitters/rematch.py`: is_rematch, fights_since,
    result_of_first_meeting, first_meeting_method, first_meeting_competitive
    (decision or split/majority), first_meeting_score_delta
  - Test: non-rematch → all None/False, known rematch → correct fields,
    competitive = decision-only
  - _Requirements: FE-8.4 (partial)_
  - Depends on: FE-0.1, FE-0.2, FE-1.5
  - Files: `src/ufc_edge/features/emitters/rematch.py`,
    `tests/features/test_emitter_rematch.py`

- [ ] FE-2.12 Implement WeightCutEmitter
  - Create `src/ufc_edge/features/emitters/weight_cut.py`: missed_weight_last_3,
    missed_weight_career, moving_down_in_weight, short-notice/full_camp where
    derivable from structured data (event-name-based DWCS detection, weight class
    change timing)
  - Test: missed weight detection from structured data, moving-down from class
    history, None when not derivable
  - _Requirements: FE-8.4 (partial)_
  - Depends on: FE-0.1, FE-0.2, FE-1.5, FE-1.7
  - Files: `src/ufc_edge/features/emitters/weight_cut.py`,
    `tests/features/test_emitter_weight_cut.py`

---

### Wave 3 — Replay Engine, Storage, Integration

- [ ] FE-3.1 Implement HistoricalFightLoader
  - Create `src/ufc_edge/features/loader.py`: SQL query joining events + fights +
    fighters + fight_totals + round_stats + sig_strike_breakdowns, anti-join on
    validation_quarantine, returns sorted `list[HistoricalFight]`
  - Test: fixture DB → correct joins, quarantined rows excluded, all fields populated
  - _Requirements: FE-1.6, FE-1.7_
  - Depends on: FE-0.1
  - Files: `src/ufc_edge/features/loader.py`, `tests/features/test_loader.py`

- [ ] FE-3.2 Implement Replay Engine orchestration
  - Create `src/ufc_edge/features/replay.py`: EventTicker groups by event, main
    `replay()` function: load → tick → freeze → emit → update loop, validates emitter
    outputs against registry, returns list of FeatureRow
  - Test: 3-event fixture → correct tick count, emit-before-update ordering verified
    by mock component that records call order
  - _Requirements: FE-1.1, FE-1.2, FE-1.3, FE-1.4, FE-1.5, FE-1.8_
  - Depends on: FE-0.1, FE-0.2, Waves 1a/1b (all components), Wave 2 (all emitters)
  - Files: `src/ufc_edge/features/replay.py`, `tests/features/test_replay.py`

- [ ] FE-3.3 Implement Feature Storage (staging, validation, swap)
  - Create `src/ufc_edge/features/storage.py`: DDL for `features_v{N}`, staging table
    write from FeatureRow list, PK uniqueness + row count validation, atomic swap via
    `ALTER TABLE RENAME`, provenance columns (feature_version, generated_at)
  - Test: staging with duplicates fails, swap replaces atomically, prior table
    unchanged on failure, schema matches registry
  - _Requirements: FE-4.1, FE-4.2, FE-4.3, FE-4.4, FE-4.7, FE-4.8_
  - Depends on: FE-0.1, FE-0.2, FE-0.3
  - Files: `src/ufc_edge/features/storage.py`, `tests/features/test_storage.py`

- [ ] FE-3.4 Implement CLI entry point and Makefile target
  - Create `src/ufc_edge/features/__main__.py`: loads config via OmegaConf, checks
    version integrity, runs replay, writes to storage; exit codes for each failure mode
  - Add `make features` target to Makefile
  - Test: CLI runs on fixture DB end-to-end, version mismatch exits nonzero
  - _Requirements: FE-3.6, FE-3.8_
  - Depends on: FE-3.1, FE-3.2, FE-3.3, FE-0.3
  - Files: `src/ufc_edge/features/__main__.py`, `Makefile`

---

### Wave 4 — Leakage Suite, Property Tests, Verification

- [ ] FE-4.1 Implement deletion oracle test
  - Create `tests/features/test_leakage.py::test_deletion_oracle`: for each fight in
    fixture, remove it + later data, replay, compare row byte-for-byte with full-replay
  - **Property 1: Temporal isolation** — feature row for fight X depends only on
    strictly-prior data
  - Tag: `Feature: feature-engine, Property 1: Temporal isolation`
  - **Validates: Requirements FE-1.1, FE-1.2, FE-10.1**
  - Depends on: FE-3.2
  - Files: `tests/features/test_leakage.py`

- [ ] FE-4.2 Implement same-card isolation test
  - `tests/features/test_leakage.py::test_same_card_isolation`: alter/remove one fight
    on an event, verify no other fight's row changes
  - **Property 2: Same-card isolation** — intra-event fights are independent
  - Tag: `Feature: feature-engine, Property 2: Same-card isolation`
  - **Validates: Requirements FE-1.1, FE-10.2**
  - Depends on: FE-3.2
  - Files: `tests/features/test_leakage.py`

- [ ] FE-4.3 Implement determinism test
  - `tests/features/test_leakage.py::test_determinism`: run replay twice with identical
    inputs, assert byte-identical output
  - **Property 3: Determinism** — same input → same output
  - Tag: `Feature: feature-engine, Property 3: Determinism`
  - **Validates: Requirements FE-1.3, FE-10.3**
  - Depends on: FE-3.2
  - Files: `tests/features/test_leakage.py`

- [ ] FE-4.4 Implement symmetry-input consistency test
  - `tests/features/test_leakage.py::test_symmetry_consistency`: for fight (A,B),
    verify delta signs flip between orientations, absolute features match fighter
  - **Property 4: Symmetry-input consistency** — orientations are coherent
  - Tag: `Feature: feature-engine, Property 4: Symmetry-input consistency`
  - **Validates: Requirements FE-1.8, FE-10.4**
  - Depends on: FE-3.2
  - Files: `tests/features/test_leakage.py`

- [ ] FE-4.5 Implement registry and import firewall tests
  - `tests/features/test_safety.py`: duplicate column, bad type, missing family,
    stale hash all raise RegistryError; AST/grep check that `features/` never imports
    from `polymarket` or reads `winner_url`
  - **Property 9: No market contamination + Property 10: Registry guards**
  - **Validates: Requirements FE-4.5, FE-4.6, FE-11.1, FE-11.2, FE-11.3, FE-11.8**
  - Depends on: FE-0.2, FE-0.3
  - Files: `tests/features/test_safety.py`

- [ ] FE-4.6 Full integration verification
  - Run `make test`, `make lint` on complete feature engine; verify all tests pass,
    no ruff violations, feature table materializes on fixture data
  - _Requirements: FE-10.6, FE-10.8_
  - Depends on: FE-4.1, FE-4.2, FE-4.3, FE-4.4, FE-4.5
  - Files: (verification only, no new files)

---

### Wave 5 — Scraper Extension and Kaggle Protocol (parallel to Waves 1–4)

- [ ] FE-5.1 Extend scraper to persist bout_order
  - Modify `src/ufc_edge/data/ufcstats/schemas.py`: add `bout_order: int | None` to
    `Fight` model
  - Modify `src/ufc_edge/data/ufcstats/storage.py`: add `bout_order INTEGER` column
    to `fights` DDL and upsert
  - Modify `src/ufc_edge/data/ufcstats/parsers.py`: persist the index from
    `parse_event()`'s URL list as `bout_order` (0 = main event, N = opener)
  - Test: fixture event → correct bout_order assignment, existing fights backfilled
    on re-scrape
  - _Requirements: FE-9.6, FE-9.7_
  - Depends on: none (data/ package only)
  - Files: `src/ufc_edge/data/ufcstats/schemas.py`,
    `src/ufc_edge/data/ufcstats/storage.py`,
    `src/ufc_edge/data/ufcstats/parsers.py`,
    `tests/data/test_parsers.py` (extend existing)

- [ ] FE-5.2 Implement Kaggle per-field validation protocol
  - Create `src/ufc_edge/features/kaggle_validator.py`: cross-check function (sample
    match rate), provenance audit schema, leakage test (correlation with post-fight
    columns), pass/fail report writer to `data/interim/kaggle_validation/`
  - Test: known-clean field passes, known-leaky field fails, low-match-rate field
    fails
  - _Requirements: FE-9.1, FE-9.2, FE-9.3, FE-9.4, FE-9.5, FE-9.8_
  - Depends on: none
  - Files: `src/ufc_edge/features/kaggle_validator.py`,
    `tests/features/test_kaggle_validator.py`

---

### Wave 6 — Documentation and Handoff

- [ ] FE-6.1 Regenerate .kiro/specs/FEATURES.md
  - Write `.kiro/specs/FEATURES.md` from legacy registry + D39 triage: verbatim definitions,
    exclusions marked with decision numbers, nothing invented, Glicko-2 addition noted
  - _Requirements: (documentation only)_
  - Depends on: FE-0.2 (registry finalized)
  - Files: `.kiro/specs/FEATURES.md`

---

## Timeline Summary

| Wave | Tasks | Parallel execution | Cross-spec joins |
|---|---|---|---|
| Wave 0 | FE-0.1, FE-0.2, FE-0.3, FE-0.4 | All 4 tasks run concurrently | None — no prerequisites |
| Wave 1a | FE-1.1, FE-1.2, FE-1.3, FE-1.5, FE-1.6, FE-1.7 | 6 tasks concurrent | None |
| Wave 1b | FE-1.4 | 1 task (depends on FE-1.1, FE-1.3 from Wave 1a) | None |
| Wave 2 | FE-2.1 through FE-2.12 | 12 tasks concurrent (each depends only on its component) | None |
| Wave 3 | FE-3.1 through FE-3.4 | FE-3.1 parallel with others; FE-3.2 waits on all W1+W2; FE-3.3 waits on FE-0.x; FE-3.4 waits on FE-3.1–3.3 | None |
| Wave 4 | FE-4.1 through FE-4.6 | FE-4.1–4.4 concurrent; FE-4.5 parallel; FE-4.6 waits on all | ME Wave 2+ blocks on FE Wave 3 (`features_v1` materialized) |
| Wave 5 | FE-5.1, FE-5.2 | Both run concurrent, parallel to Waves 1–4 | §5a emitter activation depends on FE-5.1 |
| Wave 6 | FE-6.1 | Single task | None |

## Module Ownership Table

| File | Owner task | Creates or modifies |
|---|---|---|
| `src/ufc_edge/features/contracts.py` | FE-0.1 | Creates |
| `src/ufc_edge/features/registry.py` | FE-0.2 | Creates |
| `src/ufc_edge/features/versioning.py` | FE-0.3 | Creates |
| `configs/graph.yaml` | FE-0.4 | Creates |
| `src/ufc_edge/features/components/*.py` | FE-1.1–1.7 | Creates |
| `src/ufc_edge/features/emitters/*.py` | FE-2.1–2.12 | Creates |
| `src/ufc_edge/features/loader.py` | FE-3.1 | Creates |
| `src/ufc_edge/features/replay.py` | FE-3.2 | Creates |
| `src/ufc_edge/features/storage.py` | FE-3.3 | Creates |
| `src/ufc_edge/features/__main__.py` | FE-3.4 | Creates |
| `src/ufc_edge/data/ufcstats/schemas.py` | FE-5.1 | Modifies |
| `src/ufc_edge/data/ufcstats/storage.py` | FE-5.1 | Modifies |
| `src/ufc_edge/data/ufcstats/parsers.py` | FE-5.1 | Modifies |
| `src/ufc_edge/features/kaggle_validator.py` | FE-5.2 | Creates |
| `.kiro/specs/FEATURES.md` | FE-6.1 | Creates |

## Notes

- All tests use fixture data and in-memory DuckDB; no live-site access per AGENTS.md.
- Each Wave's tasks produce no overlapping files — safe for concurrent execution.
- FE-1.4 (CommonOpponentIndex) is isolated in Wave 1b because it depends on
  FE-1.1 (Elo) and FE-1.3 (PageRank) from Wave 1a for quality weighting.
- The `TODO(human)` config values in `configs/graph.yaml` must be set before graph
  features produce meaningful outputs; the system runs with placeholder values but
  results are not production-valid.
- §9d (Opponent Trajectory) is explicitly excluded from v1 (D39); no task exists.

## Handoff Completion Definition

**The feature engine is complete and handoff-ready when:**
- `make features` produces a `features_v1` table on the full production DuckDB
- All leakage tests (deletion oracle, same-card, determinism, symmetry) pass
- `make test` and `make lint` pass with zero failures
- Source-hash guard activates correctly on any source modification
- Registry rejects all invalid configurations at startup
- `.kiro/specs/FEATURES.md` is regenerated and matches the registry
