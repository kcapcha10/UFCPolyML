# design.md — ufc-edge low-level design

> **Living document.** Sections are filled in as decisions lock (recorded in
> [Key Design Decisions.md](Key%20Design%20Decisions.md), numbering D8+). The
> high-level architecture this refines is locked in
> [docs/DECISIONS_architecture_redesign.md](../../docs/DECISIONS_architecture_redesign.md).
> Scope: the **entire six-layer system** (no milestones); build order lives in
> [tasks.md](tasks.md), sequenced per D9 (XGBoost-first end-to-end).

## Overview

`TBD — drafted after the first cluster of LLD decisions locks.`

## Architecture

`TBD — mermaid diagram of the six layers with the concrete module/interface
boundaries this LLD defines (refining the locked HLD block diagram).`

## Components & Interfaces

`TBD — one subsection per layer: DATA (incl. validation suite, D8), FEATURES,
REPS, PREDICT, STRATEGY, SIM, ONLINE. EVAL spine designed below (D10).`

### DATA — validation suite (locked — D8/D11/D12; implemented)

`src/ufc_edge/data/validation/`: `invariants.py` (one function per invariant
family, SQL over DuckDB, all conditional checks skip what cannot be evaluated),
`quarantine.py` (DDL + full-refresh semantics), `runner.py` (orchestration, JSON
report, nonzero exit on failure → `make validate` is a CI gate), `schemas.py`
(`Violation`, `ValidationReport`). Era scoping: every violation carries an
`event_date` (fighter rows are dated by their latest fight); the alarm fires on
label-universe or undated violations only. The D11 boundary lives in
`configs/data/default.yaml: label_start_date` — the single source of truth all
layers must read.

### FEATURES — as-of replay engine (design locked — D13/D14/D15; build gated on tasks.md)

Chronological state replay (D13): fights stream in event-date order; per-fighter
and global state components (D14) hold every accumulator; the **two-phase tick**
is the load-bearing rule — phase 1 emits all of fight X's features against
frozen pre-X state (any emitter may read any component, either fighter, or the
graph), phase 2 applies X's outcome to all components. One engine serves
training and upcoming-card prediction; replay input excludes quarantined rows
(D12) and reads the full feature-history universe (D11 — no lower date bound).

**Protocol (D15):**

```python
class StateComponent(Protocol):          # phase 2
    def update(self, fight: FightOutcomeView) -> None: ...

@emitter(section="2", features=["days_since_last_fight", ...])
def emit_activity(ctx: EmitContext) -> dict[str, float | None]: ...   # phase 1
```

`EmitContext` = frozen view: `ctx.a` / `ctx.b` (each fighter's full component
set), `ctx.graph` (global), `ctx.fight_info` (as-of bout metadata: date, weight
class, scheduled rounds). The **registry** collects emitters, fails loudly at
startup on duplicate feature names, and defines the output schema. `None` →
NULL → NaN per the D9 missingness contract.

**Component map (FEATURES.md → code):**

| Component | FEATURES.md | Notes |
|---|---|---|
| `PhysicalState` | §1 | first-observed UFC class (D8 resolution); age at emit from dob |
| `ActivityState` | §2 | fight dates, injury-stoppage flag |
| `RecordState` | §3, §3a | streaks, win/loss splits; debut context reads *opponent's* RecordState (cross-family) |
| `FinishingState` | §4, §4a | finish/finished rates, durations |
| `OutputState` | §5 | rolling per-min stats; **§5a deferred — card position not captured by the scraper (data gap, logged)** |
| `ChampionshipState` | §6 | title/main-event counts from bout metadata |
| `WeightClassState` | §8 | class history, per-class records; §8b interactions at emit |
| `EloState` | §9a | variable-K config per docs/DECISIONS.md |
| `FightGraph` (global) | §9b, §9c, §11 | dated who-beat-whom edges; PageRank as-of; common-opponent queries; rematch = pair-edge lookup |
| `GeoState` | §12 | event country; home-country/training-base fields deferred to enrichment (D7) |
| matchup emitters (stateless) | §10, §10a, §10b | pure deltas/interactions over both fighters' components |
| — excluded from engine v1 | §9d (leakage trap, TODO(human) semantics open), §13–§15 (strategy-side / enrichment-gated) | |

**Versioning (D15):** `FEATURE_VERSION` manual constant; committed lockfile maps
version → source hash of the features package; a pytest fails when they diverge
(forgetting to bump is impossible but loud). MLflow logs the version; walk-
forward windows lock to it (docs/FEATURES.md Versioning).

**Output (D15):** wide DuckDB table `features_v{N}`, one row per
`(fight_url, fighter_url)`; labels never stored here (join `fights.winner_url`
downstream — features and labels stay separate surfaces).

**P1 test plan (same build wave, non-negotiable):** (a) deletion oracle — for
sampled fights X, truncate the DB to `event_date < X`, replay, and require X's
emitted row to be bit-identical to the full-replay row; sample must include
cross-family features (common-opponent, interaction terms), where phase-mixing
bugs hide; (b) determinism — two full replays produce identical tables;
(c) hypothesis-generated synthetic careers exercising debut/rematch/layoff edge
cases.

### EVAL spine (locked — D10)

Lives in `src/ufc_edge/eval/` (package to be created in the build wave). Three
components; everything downstream consumes these, nothing reimplements them.

**`WalkForwardSplitter`** — the only legal way to produce train/test splits.

```python
@dataclass(frozen=True)
class Fold:
    """Event-ID sets; fights are resolved to events upstream, so a card can
    never straddle a boundary by construction."""
    train_event_ids: frozenset[str]
    calibration_event_ids: frozenset[str]   # trailing slice of train window
    test_event_ids: frozenset[str]

def generate_folds(events: EventIndex, config: SplitConfig) -> list[Fold]: ...
```

Expanding window; test blocks advance chronologically; deterministic (no RNG).
`SplitConfig` (Hydra) holds test-block length and calibration-slice sizing.

**Calibration stage** — separate from any model (D9 contract lock #1).

```python
def fit_calibrator(raw_scores: np.ndarray, outcomes: np.ndarray) -> Calibrator:
    """Platt if len < ISOTONIC_MIN_SAMPLES else isotonic (D10)."""
```

**Metrics module** — Brier, log-loss, ECE, reliability curve; per-fold + pooled;
cluster bootstrap CIs resampled at event level. Emits an `EvalReport` (frozen
Pydantic) that the MLflow logging wrapper persists with full provenance
(DVC rev, feature version, config, seeds, library versions).

## Data Models

`TBD — DuckDB tables, Pydantic schemas, feature-matrix contract (incl. the D9
missingness-semantics lock).`

## Error Handling

`Partially locked; remaining layers TBD.`

- **Ingest boundary:** frozen Pydantic models validate on ingest; schema drift
  fails loudly (existing DATA behavior).
- **Validation (D12):** impossible rows → `validation_quarantine` with reason
  codes; source rows never mutated or deleted; feature computation anti-joins
  the quarantine. Alarm is era-scoped: label-universe or undated violations
  fail the suite (exit 1); pre-cutoff violations report only. Quarantine census
  surfaces in every eval report — quarantine without a visible report is
  silent data loss.
- **Capture cron:** per-market isolation, retries, idempotent writes (existing
  behavior; see docs/DECISIONS.md).

## Correctness Properties

Property-based invariants — each is a pytest that generates or enumerates cases,
not a single-example unit test. P3 is locked (D10); the rest are drafted and
formalize when their layer's design locks.

**P3 (locked, D10) — split hygiene.** For every fold emitted by
`generate_folds`, over any event index and any valid `SplitConfig`:

1. `max(event_date over train ∪ calibration) < min(event_date over test)` —
   nothing used to fit the model *or the calibrator* postdates the test block;
2. `train ∩ test = ∅`, `calibration ⊆ train`, and every fight of any event maps
   to exactly one fold side — no card ever straddles a boundary;
3. two invocations with identical inputs yield identical folds.

Test strategy: hypothesis-generated synthetic event calendars (varying density,
duplicate dates, single-event eras) + the real event index once scraped.

**Drafted (formalize with their layers):**
- **P1** — no feature value for fight X changes if all data with
  `event_date ≥ fight_date(X)` is deleted before computation (the deletion test
  is the strongest leakage oracle: recompute-under-truncation must be a no-op).
- **P2** — no enrichment fact feeding fight T cites `source_published_at ≥ T`.
- **P4** — SIM state at decision time t is identical whether or not any
  snapshot with tick timestamp > t exists in the store.
- **P5** — the strategy layer's inputs are calibrated `P(win)` only; raw scores
  are unreachable from strategy code (enforced by module API, tested by import
  contract).

## Testing Strategy

`TBD — pytest leakage guards per layer, fixture-based parser tests (existing),
property-based tests for the invariants above, eval-suite gates for LLM
enrichment (docs/FEATURES.md §15 thresholds).`
