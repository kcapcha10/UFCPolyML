# requirements.md — ufc-edge low-level design

> **Living document.** Requirements are added per bucket as the corresponding
> design decisions lock (see [Key Design Decisions.md](Key%20Design%20Decisions.md)).
> Format: each requirement = User Story + EARS acceptance criteria
> (WHEN / IF / WHILE / THE / SHALL).

## Introduction

`TBD — one paragraph: what the system does, for whom, and the correctness bar
(no leakage, calibrated probabilities, honest backtest).`

## Glossary

`TBD — as-of cutoff, walk-forward, calibration (Platt/isotonic), ECE/Brier,
tick/tick_id, CLOB, order-book cross-section, model-edge vs execution-edge,
OPE, fractional Kelly, …`

## Requirements

### Bucket A — "One corrupted row poisons every accumulator downstream" (Data Integrity & the Two Universes)

*Locked by D11 + D12 (2026-07-05); validation suite implemented in
`src/ufc_edge/data/validation/`. The failures this bucket prevents are silent:
a parser regression ingesting quietly for weeks, or a boundary-truncated career
feeding fictional features to the earliest training folds.*

#### A1 — Two data universes (D11)
**User story:** As the researcher, I want training restricted to the modern era
without truncating career accumulators, so old fights stop being training rows
while features stay honest.

- WHEN training, calibration, or evaluation rows are selected, THE system SHALL
  include only fights with `event_date >= label_start_date` (config; default
  2010-01-01).
- WHEN any feature or accumulator (career stats, Elo, fight graph) is computed,
  THE system SHALL draw on the full scraped history with no lower date bound.
- THE label boundary SHALL be a single config value; no layer may hard-code it.

#### A2 — Impossibility invariants (D8/D12)
**User story:** As the researcher, I want impossible data detected the day it
appears, so parser bugs and source corruption never reach features.

- WHEN the validation suite runs, THE system SHALL check: landed ≤ attempted,
  non-negative counts, round-sums = totals (only where round rows exist),
  breakdown partitions agree, referential integrity, no self-fights, winner is
  a participant, ending round within schedule, plausible measurements,
  plausible event dates, and control time ≤ fight duration.
- IF a check cannot be evaluated for a row (expected historical absence), THEN
  the row SHALL be skipped, not flagged — incomplete is not corrupt.

#### A3 — Quarantine, never silent drop, never auto-fix (D12)
**User story:** As the researcher, I want every excluded row visible with a
reason, so data exclusion can never quietly bias the training distribution.

- WHEN a row violates an invariant, THE system SHALL record it in
  `validation_quarantine` with a stable reason code; source rows SHALL never be
  mutated or deleted.
- WHEN feature computation runs, THE system SHALL exclude quarantined rows via
  anti-join on `(table_name, row_key)`.
- WHEN the suite re-runs, THE quarantine SHALL be fully refreshed — a violation
  fixed by re-scrape disappears rather than lingering.
- WHEN any evaluation run executes, THE quarantine census (counts by reason)
  SHALL appear in its report and MLflow log.

#### A4 — Era-scoped alarm (D12)
**User story:** As the researcher, I want modern-era violations to fail the
build immediately, while historical quirks stay visible but non-blocking.

- IF any violation is dated on/after `label_start_date`, or cannot be dated at
  all, THEN the suite SHALL fail (nonzero exit for CI/make gating).
- WHILE all violations are dated strictly before `label_start_date`, THE suite
  SHALL pass with the violations reported and quarantined.
- WHEN a fighter-table violation is found, THE system SHALL date it by the
  fighter's most recent fight, so regressions on active fighters trip the
  modern alarm.

### Bucket B — "The engine that cannot read the future" (As-of Feature Computation)

*Locked by D13/D14/D15 (2026-07-05). The failure this bucket prevents: a
feature that quietly encodes how a career ended, or a cached table that no
longer matches the code that claims to have produced it.*

#### B1 — Replay ordering and the two-phase tick
**User story:** As the researcher, I want feature emission structurally unable
to see a fight's outcome or any later fight, so leakage is prevented by
construction rather than reviewer vigilance.

- WHEN fights are replayed, THE engine SHALL process them in strict event-date
  order over the full feature-history universe (D11), excluding quarantined
  rows (D12).
- WHEN processing fight X, THE engine SHALL emit all of X's features against
  frozen pre-X state before any component applies X's outcome.
- WHEN emitters read across components or fighters, THE result SHALL be
  independent of component registration order.

#### B2 — Component isolation and the registry
**User story:** As the researcher, I want each FEATURES.md family isolated in
its own component, so adding a family cannot corrupt an existing one.

- WHEN a new state component is added, THE change SHALL not require modifying
  any existing component.
- IF two emitters declare the same feature name, THEN the registry SHALL fail
  at startup.

#### B3 — Leakage and determinism guards (P1)
**User story:** As the researcher, I want an oracle that catches even
one-fight clairvoyance, so a plausible-but-fake metric gain cannot survive CI.

- WHEN the P1 test truncates the database to `event_date < X` and replays, THE
  emitted features for fight X SHALL be bit-identical to the full-replay row —
  including cross-family features.
- WHEN the engine replays the same input twice, THE output tables SHALL be
  identical.

#### B4 — Feature versioning with a hash guard
**User story:** As the researcher, I want any feature-code change without a
version bump to fail loudly, so cached tables can never silently mix code
generations.

- WHEN feature source code changes and `FEATURE_VERSION` does not, THEN the
  hash-guard pytest SHALL fail.
- WHEN any training or eval run executes, THE feature version SHALL be logged
  to MLflow and joined into the run's provenance.

#### B5 — Output contract
- THE engine SHALL write one wide row per `(fight_url, fighter_url)` to
  `features_v{N}`; missing values are NULL (→ NaN downstream, D9 contract);
  labels SHALL never be stored in the features table.

### Bucket C — Learned Representations (REPS)
`TBD`

### Bucket D — Prediction & Calibration
`TBD`

### Bucket E — "A plausible backtest number that is quietly wrong" (Evaluation & Reproducibility Spine)

*Locked by D10 (2026-07-05). The failure this bucket exists to prevent is silent:
an evaluation that runs clean and reports metrics better than reality.*

#### E1 — Walk-forward split generation
**User story:** As the researcher, I want folds generated by one canonical
expanding-window, event-grouped splitter, so that no evaluation anywhere in the
project can benefit from future information or same-card leakage.

- WHEN folds are generated, THE splitter SHALL assign every fight of a given
  event to the same side of every split (the split unit is the event, never the
  fight).
- WHEN fold *k* is generated, THE splitter SHALL include in its training set only
  events with `event_date` strictly before the earliest `event_date` of test
  block *k*.
- WHEN invoked twice with the same dataset snapshot and config, THE splitter
  SHALL emit identical folds (deterministic; no RNG in splitting).
- IF any generated fold violates the temporal or event-grouping property, THEN
  the P3 pytest guard SHALL fail the test suite.

#### E2 — In-fold calibration
**User story:** As the researcher, I want the calibrator fit strictly inside each
fold, so that reported calibration (ECE, reliability) reflects what live
deployment would have seen.

- WHEN a calibrator is fit for fold *k*, THE system SHALL fit it only on a
  trailing validation slice of fold *k*'s training window.
- IF any row used for calibration fitting has `event_date` on or after fold *k*'s
  first test-event date, THEN the P3 guard SHALL fail.
- WHILE the calibration slice holds fewer than `ISOTONIC_MIN_SAMPLES` rows
  (config; default 1000), THE system SHALL use Platt scaling; otherwise isotonic.
- THE strategy layer SHALL receive only calibrated `P(win)` (architecture
  invariant #3 restated as an acceptance criterion).

#### E3 — Metrics and uncertainty
**User story:** As the researcher, I want per-fold and pooled Brier, log-loss,
ECE, and reliability curves with honest uncertainty, so model comparisons (and
future REPS ablations) are decided by evidence, not noise.

- WHEN a walk-forward run completes, THE system SHALL report Brier, log-loss,
  ECE, and a reliability curve, per fold and pooled.
- WHEN confidence intervals are computed, THE system SHALL use cluster bootstrap
  resampling at the event level, never the fight level.

#### E4 — Era-drift ablation (one-time experiment)
**User story:** As the researcher, I want pooled Brier as a function of
training-window length, so that expanding-vs-rolling remains a data-driven
choice (D10 rider).

- WHEN the drift ablation runs, THE system SHALL evaluate the same model under
  multiple training-window lengths using the E1 splitter and report the curve.

#### E5 — Reproducibility of every run
**User story:** As the researcher, I want any reported number to be regenerable
from its logged provenance, so no result in this project is unexplainable later.

- WHEN any evaluation run executes, THE system SHALL log to MLflow: the DVC data
  snapshot revision, feature version, resolved Hydra config, random seeds, and
  library versions.
- WHEN a run is repeated with identical logged inputs, THE system SHALL
  reproduce the reported metrics (bit-identical where the underlying libraries
  allow; otherwise within a documented tolerance).

### Bucket F — Strategy & Execution
`TBD`

### Bucket G — Simulation (CLOB replay)
`TBD`

### Bucket H — Online Feedback & Operations
`TBD`

> Bucket names are provisional scaffolding; they get problem-framed titles
> (e.g. "Bucket A — a leaked feature looks like alpha") as they are written.
