# UFC feature registry

This is a human-owned registry, not a feature brainstorm. V1 implements only the
tabular, pre-fight features below. Every input must be available strictly before the
fight being predicted; no odds-derived signal reaches XGBoost.

## Common contract

- Compute fighter state from prior fights only, then emit the upcoming fight row.
- A feature unavailable for a fighter is `NULL` in DuckDB and `NaN` in XGBoost.
- Per-fighter values are materialized first. Matchup deltas are assembled only in
  the PREDICT layer so fighter-order symmetry is enforceable.
- Definitions with unresolved data semantics are excluded, not guessed.

## V1 feature families

| Family | Core fields |
|---|---|
| Physical profile | height, reach, reach/height ratio, stance, age at fight, current and first observed UFC weight class |
| Activity | days since last fight, fight counts over 12 months/3 years/5 years, UFC and career fight counts, inactivity tier, injury-stoppage flag |
| Record | overall/UFC/recent win rates, current streak, finish/decision win and loss rates, UFC-debut flag and opponent UFC experience |
| Finishing and durability | KO, submission, early-finish, and overall finish rates; duration mean/variance; prior stoppage history; never-finished × opponent-finish-rate interaction |
| Fight output | striking pace, accuracy and defense; takedown pace, accuracy and defense; submission attempts; knockdowns; damage ratio; control time; strike-location shares |
| Experience and class | weight-class changes, class-specific record, scheduled-round context, and size × grappling-utilization interaction |
| Strength of schedule | pre-fight Elo, Elo trajectory/peak, PageRank on the dated win graph, and common-opponent summaries over the prior three years |
| Matchup | physical, rating, finishing, output, duration, experience, wrestling, submission, pace, and stance deltas/interactions |
| Rematch | prior-meeting result/method/competitiveness and fights since first meeting |

## Explicitly deferred or excluded

| Item | Reason |
|---|---|
| Pre-UFC record and regional-circuit features | Require a separate source and fighter identity matching |
| Card-position features | Required card-position data is not captured |
| Title, champion, home-country, and travel features | Require data that the current UFCStats pipeline does not capture |
| Opponent post-fight trajectory | Leakage-shaped; semantics are unresolved |
| Camp, injury, behavioral, and LLM features | Retired from v1 scope |
| Market-derived features | Used only as a post-model report comparator, never a model input |

## Versioning

`FEATURE_VERSION` is human-readable. A committed source-hash lock must match the
feature package; changing feature logic without a version update fails tests.
