# Feature Engine — Requirements

## Introduction

The feature engine materializes point-in-time tabular features for every UFC fight
and fighter orientation. It replays historical state event-by-event, enforces strict
temporal isolation (emit-before-update), produces a versioned wide DuckDB table keyed
by `(fight_url, fighter_url)`, and validates feature correctness through deletion
oracles, same-card isolation, determinism, and symmetry-input consistency tests. The
engine is the sole owner of graph rating systems (Elo, Glicko-2, PageRank, common
opponents), rolling performance accumulators, and the `features_v{N}` table contract
consumed by downstream model assembly.

## Glossary

- **Replay_Engine** — the orchestrator that processes events in temporal order, freezes
  state, invokes emitters, then applies outcomes to state components.
- **StateComponent** — a protocol-implementing class that accumulates historical
  outcomes (e.g., Elo tracker, career record counter) and exposes read-only state.
- **FeatureEmitter** — a protocol-implementing class that reads frozen state via an
  EmitContext and returns a dict of scalar feature values.
- **EmitContext** — a frozen, read-only view of the focal fighter, opponent, event
  metadata, and all registered state components; excludes labels, market data, and
  mutable state.
- **Event tick** — one atomic replay step encompassing all fights on a single UFC event;
  no fight within a tick may observe another fight's outcome from that same tick.
- **Feature_Registry** — the module that declares valid emitter names, output columns,
  and schema; enforces uniqueness and version consistency.
- **FEATURE_VERSION** — a human-incremented identifier mapped to a source-hash of the
  feature package; mismatches fail loudly.
- **Deletion_Oracle** — a correctness test proving that a fight's feature row depends
  only on strictly-prior information.
- **Kaggle_Validator** — the per-field validation protocol that gates admission of
  external Kaggle data (D30).
- **Graph_System** — collective term for Elo, Glicko-2, PageRank, and common-opponent
  analysis, all recomputed as-of each event tick.

## Requirements

---

**Bucket A — Replay Engine and Temporal Isolation** (Requirements 1–2)

### Requirement 1: Event-Atomic Replay Ordering

**User Story:** As a model engineer, I want features computed from a strict temporal
replay, so that no future information leaks into any feature row.

**Requirement:** The Replay_Engine SHALL process events in chronological order by
`(event_date, event_url)`, treating each event as an atomic tick. Within a tick it
SHALL freeze all state, emit features for every fight, and only then apply outcomes.

#### Acceptance Criteria

1. WHEN the Replay_Engine begins processing event `E`, THE Replay_Engine SHALL freeze
   a read-only snapshot of every registered StateComponent before emitting any row.
2. WHEN the Replay_Engine emits features for fight `F` in event `E`, THE FeatureEmitter
   SHALL receive only the frozen pre-event state via EmitContext and SHALL NOT observe
   outcomes from `E` or any later event.
3. WHEN multiple fights exist within a single event tick, THE Replay_Engine SHALL sort
   them by `fight_url` for deterministic output ordering.
4. WHEN all rows for event `E` are emitted, THE Replay_Engine SHALL apply every resolved
   outcome in `E` to the relevant StateComponents in a single batch.
5. IF an event has no resolved outcomes (all fights cancelled), THEN THE Replay_Engine
   SHALL emit feature rows with current frozen state and SHALL NOT update any component.
6. WHEN the Replay_Engine encounters a fight present in `validation_quarantine`, THE
   Replay_Engine SHALL exclude that fight via anti-join before replay begins.
7. WHEN feature state is initialized, THE Replay_Engine SHALL include all available UFC
   history regardless of `label_start_date`; the label universe filter applies only
   downstream at matrix assembly.
8. THE Replay_Engine SHALL emit exactly two rows per fight: one per fighter orientation
   (fighter as focal, opponent as counterpart), producing a symmetric pair.

### Requirement 2: EmitContext Isolation and Protocol Contracts

**User Story:** As a feature developer, I want compile-time-clear contracts between
state components and emitters, so that accidental leakage through mutable references
is structurally impossible.

**Requirement:** StateComponent and FeatureEmitter SHALL follow typed protocols.
EmitContext SHALL expose only the focal fighter, opponent, event metadata, and
registered state snapshots. It SHALL NOT expose labels, market data, raw DuckDB
handles, or mutable component internals.

#### Acceptance Criteria

1. WHEN a FeatureEmitter's `emit()` method is invoked, THE EmitContext it receives
   SHALL be a frozen (immutable) object with no setter methods or mutable references.
2. THE EmitContext SHALL expose: focal fighter URL/profile, opponent URL/profile, event
   date, event URL, weight class, and a mapping of component names to their frozen
   read-only views.
3. THE EmitContext SHALL NOT expose: `winner_url`, fight outcome, market prices, order
   book data, `validation_quarantine` contents, or any DuckDB connection.
4. WHEN a StateComponent's `update()` method is called, THE StateComponent SHALL
   accept a `FightOutcomeView` containing only the fight URL, event date, both fighter
   URLs, winner URL, method, ending round, ending time, and weight class.
5. IF a FeatureEmitter attempts to mutate state (write to a component or database),
   THEN THE system SHALL raise a runtime error before the mutation takes effect.
6. WHEN the Feature_Registry initializes, THE Feature_Registry SHALL fail startup IF
   any emitter name or output column is duplicated, IF an emitter declares a value
   outside `float | str | None`, or IF a configured family is absent from the registry.
7. WHEN a new FeatureEmitter is added, THE Feature_Registry SHALL require a registry
   entry, deterministic definition, version bump, source-hash update, and tests before
   the emitter is active.
8. THE FeatureEmitter protocol SHALL declare a `name: str` attribute and an
   `emit(context: EmitContext) -> dict[str, float | str | None]` method signature.

---

**Bucket B — Feature Versioning and Storage** (Requirements 3–4)

### Requirement 3: Feature Version Guard and Source-Hash Integrity

**User Story:** As a model engineer, I want stale feature code to fail loudly, so
that I never accidentally train on features produced by an outdated implementation.

**Requirement:** The system SHALL maintain a FEATURE_VERSION identifier mapped to a
cryptographic hash of the feature package source. Any mismatch between committed
manifest and implementation SHALL fail before writing to the target table.

#### Acceptance Criteria

1. WHEN the feature generation command starts, THE system SHALL compute a hash of all
   Python source files in `src/ufc_edge/features/` and compare it to the committed
   manifest entry for the current FEATURE_VERSION.
2. IF the computed source hash differs from the manifest hash, THEN THE system SHALL
   fail with a clear error naming the expected vs actual hash and SHALL NOT write to
   `features_v{N}`.
3. WHEN a developer bumps FEATURE_VERSION, THE manifest file SHALL record the new
   version string, its source hash, a human-readable changelog entry, and a timestamp.
4. THE FEATURE_VERSION identifier SHALL be a monotonically incrementing integer
   prefixed with `v` (e.g., `v1`, `v2`).
5. WHEN feature generation succeeds, THE system SHALL write the active FEATURE_VERSION
   into every row of the output table in the `feature_version` column.
6. IF a downstream consumer (model assembler) requests a feature version that does not
   match the materialized table's version, THEN THE system SHALL reject the request.
7. THE source-hash computation SHALL be deterministic: same source files in same order
   produce the same hash regardless of platform or filesystem metadata.
8. WHEN a feature version manifest exists, THE system SHALL expose a CLI or make
   target (`make features-check`) that verifies hash consistency without materializing.

### Requirement 4: Feature Table Schema and Write Semantics

**User Story:** As a downstream model assembler, I want a stable, typed wide table
with predictable schema and atomic writes, so that I can trust the feature matrix.

**Requirement:** Feature generation SHALL write to `features_v{N}` as a wide table
keyed by `(fight_url, fighter_url)`. Writes SHALL be atomic per version: staging,
validation, then swap. Missing values SHALL be `NULL` in DuckDB.

#### Acceptance Criteria

1. THE `features_v{N}` table SHALL have primary key `(fight_url, fighter_url)` and
   metadata columns: `event_url`, `event_date`, `opponent_url`, `weight_class`,
   `feature_version`, `generated_at`, followed by all registered feature columns.
2. WHEN feature generation runs, THE system SHALL write to a staging table, validate
   row count and primary-key uniqueness, then atomically swap staging into the target.
3. IF staging validation fails (duplicate keys, unexpected row count), THEN THE system
   SHALL leave the prior successful table unchanged and report the failure.
4. WHEN a feature value cannot be computed (insufficient history, missing source data),
   THE emitter SHALL return `None` which THE storage layer SHALL persist as DuckDB `NULL`.
5. THE table SHALL contain no label columns (`winner_url`, outcome) — labels are joined
   only at matrix assembly by the model package.
6. THE table SHALL contain no market-derived columns — market data is consumed only by
   the report package.
7. WHEN the Replay_Engine processes `N` events containing `M` total fights, THE output
   table SHALL contain exactly `2 × M` rows (two orientations per fight).
8. WHEN a prior `features_v{N}` table exists for the same version, THE system SHALL
   replace it atomically; it SHALL NOT append or merge incrementally.

---

**Bucket C — Graph-Derived Feature Systems** (Requirements 5–7)

### Requirement 5: Elo Rating System

**User Story:** As a feature engineer, I want an Elo rating that encodes fight-method
quality and handles inactivity, so that opponent-adjusted skill is captured.

**Requirement:** The Elo StateComponent SHALL implement variable K-factor with method
bonus and recency weighting, decay toward 1500 during inactivity, K=0 for injury
stoppages, K×0.1 for DQ outcomes, and debut initialization at 1500. All parameters
SHALL be sourced from `configs/graph.yaml`.

#### Acceptance Criteria

1. WHEN a fighter has no prior fights, THE Elo StateComponent SHALL initialize their
   rating at 1500.
2. WHEN a fight outcome is applied, THE Elo StateComponent SHALL compute the K-factor
   using method bonus and recency parameters from `configs/graph.yaml`.
3. IF the fight ended by injury stoppage (doctor stoppage or NC-injury), THEN THE Elo
   StateComponent SHALL apply K=0 (Elo-neutral).
4. IF the fight ended by disqualification, THEN THE Elo StateComponent SHALL multiply
   K by 0.1.
5. WHEN a fighter has been inactive, THE Elo StateComponent SHALL decay their rating
   toward 1500 at a configurable rate per inactivity period from `configs/graph.yaml`.
6. THE FeatureEmitter for Elo SHALL emit: `elo_rating` (pre-fight), `elo_trajectory_last5`
   (linear-regression slope of last 5 ratings), `elo_peak`, `elo_current_vs_peak`.
7. WHEN fewer than 5 rated fights exist for a fighter, THE `elo_trajectory_last5`
   feature SHALL be `None`.
8. THE K-factor parameters (`k_base`, `method_bonus_map`, `recency_weight_halflife`,
   `inactivity_decay_rate`) SHALL be defined as config slots in `configs/graph.yaml`
   with values marked `TODO(human)` until owner specifies.

### Requirement 6: Glicko-2 Rating System

**User Story:** As a feature engineer, I want Glicko-2's explicit uncertainty
(rating deviation) that grows with inactivity, so that the model knows when a
rating is unreliable.

**Requirement:** The Glicko-2 StateComponent SHALL track rating (μ), rating deviation
(RD), and volatility (σ) per fighter, updating after each fight and growing RD during
inactivity periods. Parameters SHALL be in `configs/graph.yaml`.

#### Acceptance Criteria

1. WHEN a fighter has no prior fights, THE Glicko-2 StateComponent SHALL initialize
   with the standard Glicko-2 defaults: μ=1500, RD=350, σ=τ (configurable).
2. WHEN a fight outcome is applied, THE Glicko-2 StateComponent SHALL update μ, RD,
   and σ for both fighters per the Glicko-2 algorithm.
3. WHEN a fighter has been inactive between events, THE Glicko-2 StateComponent SHALL
   increase RD proportional to the elapsed rating periods (configurable period length).
4. THE FeatureEmitter for Glicko-2 SHALL emit: `glicko2_rating`, `glicko2_rd`.
5. IF the fight ended by injury stoppage or NC-injury, THEN THE Glicko-2 StateComponent
   SHALL NOT update either fighter's rating (consistent with Elo-neutral treatment).
6. THE Glicko-2 parameters (`tau`, `initial_rd`, `initial_volatility`,
   `rating_period_days`) SHALL be defined as config slots in `configs/graph.yaml`
   with values marked `TODO(human)` until owner specifies.
7. WHEN Glicko-2 RD exceeds a configurable `high_uncertainty_threshold`, THE system
   SHALL still emit the value (not suppress it); downstream consumers decide handling.
8. THE Glicko-2 implementation SHALL use fight outcomes only (win/loss binary); draws
   and no-contests are excluded from rating updates.

### Requirement 7: PageRank and Common Opponents

**User Story:** As a feature engineer, I want graph-structural features (PageRank
and common-opponent analysis), so that indirect opponent quality is captured.

**Requirement:** PageRank SHALL operate on a directed win graph (loser→winner edges)
with configurable edge weights and damping. Common opponents SHALL use a 3-year
lookback with Elo/PageRank-weighted quality scoring. Parameters SHALL be in
`configs/graph.yaml`.

#### Acceptance Criteria

1. WHEN the PageRank StateComponent updates, THE directed graph SHALL add an edge from
   loser to winner with weight encoding: finish-type bonus, exponential recency decay,
   and early-finish bonus per `configs/graph.yaml`.
2. THE PageRank computation SHALL use damping factor α=0.85 (configurable) and converge
   to a stable score per fighter at each event tick.
3. WHEN a fighter has no edges in the graph (isolated node), THE PageRank emitter SHALL
   emit the global minimum PageRank score.
4. THE Common_Opponents StateComponent SHALL identify fighters who faced both the focal
   fighter and opponent within a 3-year lookback window as of the current event date.
5. WHEN common opponents exist, THE emitter SHALL emit: `n_common_opponents`,
   `common_opp_score_a`, `common_opp_score_b`, `common_opp_score_delta`,
   `common_opp_a_win_rate`, `common_opp_b_win_rate` — quality-weighted by Elo and
   PageRank at time of each respective fight, with recency decay.
6. WHEN no common opponents exist within the lookback window, THE emitter SHALL emit
   `None` for all common-opponent features.
7. THE PageRank parameters (`damping`, `finish_type_bonus_map`, `recency_decay_lambda`,
   `early_finish_bonus`, `convergence_tolerance`) SHALL be defined as config slots in
   `configs/graph.yaml`.
8. THE common-opponent lookback window (default 3 years) and quality-weight parameters
   SHALL be configurable in `configs/graph.yaml`.

---

**Bucket D — Feature Families and Emission Rules** (Requirements 8–9)

### Requirement 8: V1 Feature Families — Physical, Activity, Record, Output, Experience

**User Story:** As a model engineer, I want all in-scope feature families materialized
from UFC data only, so that the model has a complete, validated input matrix.

**Requirement:** The feature engine SHALL implement emitters for all v1-in-scope
families: §1 Physical Profile, §2 Activity & Inactivity, §3 Win/Loss Record (incl.
3a UFC-computable debut fields), §4 Finishing Profile (incl. 4a), §5 Output &
Efficiency, §5a Output Variance by Card Position (gated on bout_order availability),
§6 Experience & Championship, §8a/8b Weight Class & Physical Dominance, §10/10a/10b
Matchup-Level, §11 Rematch (minus LLM field), and deterministic §15 weight-cut fields.

#### Acceptance Criteria

1. WHEN the §1 Physical Profile emitter runs, THE emitter SHALL emit: `height_cm`,
   `reach_cm`, `reach_to_height_ratio`, `stance`, `age_at_fight`, `weight_class`.
2. WHEN the §2 Activity emitter runs, THE emitter SHALL emit: `days_since_last_fight`,
   `fights_last_12mo`, `fights_last_3yr`, `fights_last_5yr`, `total_ufc_fights`,
   `last_fight_injury_stoppage`, `age_x_inactivity`, `inactivity_tier` (bucketed:
   0=<6mo, 1=6–12mo, 2=1–2yr, 3=2yr+).
3. WHEN the §3 Record emitter runs, THE emitter SHALL emit: `win_pct_all`,
   `win_pct_last3`, `win_pct_last5`, `current_streak`, `win_pct_by_finish`,
   `win_pct_by_decision`, `loss_pct_by_finish`, `loss_pct_by_decision`,
   `ufc_win_pct`, `ufc_record_fights_count`, `is_ufc_debut`,
   `debut_opponent_ufc_experience`, `debut_opponent_ufc_win_pct`.
4. WHEN the §5 Output emitter runs, THE emitter SHALL compute rolling averages over
   prior fights (configurable window) for: `sig_strikes_per_min`,
   `sig_strikes_absorbed_per_min`, `striking_accuracy_pct`, `striking_defense_pct`,
   `td_per_15min`, `td_accuracy_pct`, `td_defense_pct`, `sub_attempts_per_15min`,
   `knockdown_rate`, `damage_ratio`, `grappling_dominance`, `control_time_per_fight`,
   `distance_strike_pct`, `clinch_strike_pct`, `ground_strike_pct`,
   `head_target_pct`, `body_target_pct`, `leg_target_pct`.
5. WHEN a fighter has fewer than 3 UFC fights, THE variance-based features (§5a,
   `fight_duration_variance`) SHALL be `None`.
6. WHEN the §5a Card Position emitter runs and `bout_order` data IS available in the
   `fights` table, THE emitter SHALL emit: `sig_strikes_main_card_avg`,
   `sig_strikes_prelim_avg`, `td_rate_main_card_avg`, `td_rate_prelim_avg`,
   `grappling_abandonment_delta`, `output_variance_by_position`.
7. WHEN the §10 Matchup emitter runs, THE emitter SHALL compute deltas as
   `fighter_A_value − fighter_B_value` (positive favors the focal fighter) for all
   matchup features specified in the registry.
8. WHEN the §10b Style Interaction emitter runs, THE emitter SHALL emit only
   deterministic fields: `striker_vs_grappler`, `pressure_vs_counter`,
   `pace_mismatch_score`, `southpaw_orthodox_history`.

### Requirement 9: Kaggle-Gated and Deferred Features

**User Story:** As a project owner, I want Kaggle-sourced fields admitted only after
per-field validation, so that leakage and data-quality risks are controlled.

**Requirement:** Kaggle-gated features (pre-UFC records, natural weight class,
camp/gym, training-base/travel) SHALL NOT enter the feature matrix until each field
passes the Kaggle_Validator protocol (D30). The §5a card-position features SHALL be
deferred until a scraper extension persists `bout_order`.

#### Acceptance Criteria

1. WHEN a Kaggle candidate field is proposed, THE Kaggle_Validator SHALL cross-check a
   sample against UFCStats where fields overlap and report match rate.
2. THE Kaggle_Validator SHALL audit provenance: dataset version, update cadence, known
   author, license, and documented methodology.
3. THE Kaggle_Validator SHALL test the candidate field for temporal leakage by checking
   correlation with post-fight or odds-derived columns in the source dataset.
4. IF any Kaggle field fails cross-check (match rate below threshold), provenance audit,
   OR leakage test, THEN THE system SHALL reject that field with a documented reason.
5. WHEN a Kaggle field passes all checks, THE system SHALL record: field name, source
   dataset, validation date, match rate, and provenance note in FEATURES.md.
6. UNTIL the `fights` table contains a `bout_order` column, THE §5a Output Variance
   by Card Position features SHALL NOT be materialized; the emitter SHALL emit `None`
   for all §5a columns and log a warning.
7. WHEN a scraper extension adds `bout_order` to the `fights` table, THE §5a emitter
   SHALL derive "main card" using the heuristic that top-5 bouts (by bout_order) on
   numbered events constitute the main card (configurable in Hydra).
8. THE Kaggle validation protocol results SHALL be stored in
   `data/interim/kaggle_validation/` with one JSON report per evaluated field.

---

**Bucket E — Correctness and Leakage Tests** (Requirements 10–11)

### Requirement 10: Leakage Test Suite

**User Story:** As a model engineer, I want automated leakage tests that prove no
future information contaminates any feature row, so that the temporal contract is
continuously verified.

**Requirement:** The feature engine SHALL ship a leakage test suite containing:
deletion oracle, same-card isolation, determinism, and symmetry-input consistency
tests. All tests SHALL use fixture data and require no live site access.

#### Acceptance Criteria

1. THE Deletion_Oracle test SHALL verify: removing fight `X` and all later source data,
   then replaying, produces a feature row for `X` that is byte-identical to the
   full-replay row.
2. THE Same-Card Isolation test SHALL verify: no row for event `E` changes when another
   fight on `E` is removed or its result is altered.
3. THE Determinism test SHALL verify: identical input data and configuration generate
   identical output rows, schema, and row ordering across multiple runs.
4. THE Symmetry-Input Consistency test SHALL verify: for fight `(A, B)`, the features
   emitted for fighter A with opponent B are consistent with features emitted for
   fighter B with opponent A (delta signs flip, absolute values match).
5. WHEN any leakage test fails, THE test suite SHALL report: the specific fight(s)
   that exhibit the violation, the columns that differ, and the nature of the
   difference (value, presence, ordering).
6. THE leakage test suite SHALL run as part of `make test` and SHALL use in-memory
   DuckDB with fixture data (not the production database).
7. IF a new StateComponent or FeatureEmitter is added, THEN THE leakage test suite
   SHALL cover it without requiring manual test additions (property-based test over
   the registry).
8. THE leakage tests SHALL complete within 60 seconds on fixture data containing at
   least 5 events and 30 fights.

### Requirement 11: Registry Safety and Schema Validation

**User Story:** As a feature developer, I want the system to fail loudly on schema
drift or configuration errors, so that broken features never reach the model.

**Requirement:** The Feature_Registry SHALL enforce column uniqueness, type safety,
family completeness, and version-hash agreement at startup and before any write.

#### Acceptance Criteria

1. WHEN the Feature_Registry discovers a duplicate column name across emitters, THE
   registry SHALL raise a `RegistryError` at startup with both emitter names.
2. WHEN an emitter declares a return type outside `float | str | None`, THE registry
   SHALL reject registration with the offending type name.
3. WHEN a feature family listed in configuration has no registered emitter, THE
   registry SHALL fail startup with the missing family name.
4. WHEN the source-hash check fails (per Requirement 3), THE registry SHALL prevent
   feature generation from proceeding.
5. THE registry SHALL expose a `schema() -> dict[str, type]` method returning the
   complete expected column set, used by the storage layer to validate the staging table.
6. WHEN the staging table's columns do not match the registry schema exactly (extra
   columns, missing columns, or type mismatches), THE storage layer SHALL abort the
   swap and report discrepancies.
7. THE registry SHALL maintain an ordered list of feature families, and generation
   SHALL process families in that order for deterministic output.
8. WHEN an emitter returns a key not in its declared output columns, THE replay engine
   SHALL raise an error for that row and abort generation.

## Standing Decisions with Named Fallbacks

1. **Event-atomic ticks (D13/D14).** The replay engine treats each event as one atomic
   tick because sub-event ordering is not reliably available from UFCStats. If a future
   source provides verified within-event fight ordering with timestamps, evaluate
   fight-level ticks before any architecture change.

2. **Elo as primary schedule-strength signal.** Elo is the primary opponent-quality
   measure. If the D29 ablation ladder shows Glicko-2 subsumes Elo's incremental
   value, evaluate dropping Elo emitter before adding complexity.

3. **§5a gated on bout_order (card-position verdict: DERIVABLE).** Bout order is
   derivable from DOM position but not yet persisted. A scraper extension task is
   required before §5a features activate. If scraper extension proves infeasible,
   mark all §5a features permanently deferred.

4. **Kaggle admission protocol (D30).** No Kaggle field enters until it passes
   per-field validation. If Kaggle dataset update cadence degrades below annual or
   provenance becomes untraceable, evaluate dropping Kaggle-gated features entirely.
