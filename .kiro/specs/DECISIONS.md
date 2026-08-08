# Active technical decisions

This is the single decision record for v1. It holds the choices worth being able
to explain in an interview: what was chosen, why, and the constraint it creates.

## Scope — D23

Build a batch XGBoost MLE system: validated UFC data → point-in-time features →
calibrated probabilities → timestamped market-mismatch report and paper-signal log.
This makes feature engineering, temporal validation, calibration, reproducibility,
and a useful output the portfolio story. Keep modest production polish—config-driven
commands, MLflow provenance, versioned artifacts, and health reporting.

Deep learning, LLM enrichment, automated trading, sequential strategy research,
CLOB simulation, and online learning are deliberately outside v1. They each add a
new evaluation surface without improving the core XGBoost project.

## Data integrity — D8, D11, D12

- Kaggle is not a source of truth. Internal consistency checks and quarantine are.
- Feature history uses all available UFC history; labels/train/eval default to fights
  on or after `label_start_date` (`2010-01-01`).
- Impossible rows are quarantined with reason codes. Modern-era or undated
  violations fail validation; older violations remain visible but non-blocking.

## Feature computation — D13, D14, D15

- Replay fights chronologically. Emit every feature before applying that fight's
  outcome to any state component.
- State is split by feature family; all emitters read frozen pre-fight state.
- Output is a versioned, wide `features_v{N}` table keyed by fight and fighter.
  A source-hash guard makes stale feature code loud. `NULL → NaN` is the explicit
  model missingness contract.

## Prediction and evaluation — D10, D16, D19–D22

- Split by event in chronological order; never split a card across train,
  calibration, or test. Report Brier, log loss, ECE, reliability curves, and
  event-level bootstrap intervals.
- Build both fighter orientations for training. At inference, average raw
  orientations, calibrate a canonical fighter ordering once, and return `1 - p`
  for the reversed order.
- XGBoost uses fixed boosting rounds per candidate—no early-stopping double use.
  If random search remains in the lean protocol, its active eight-axis surface is
  `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
  `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
- Reserve a trailing event-complete calibration slice with at least
  `max(250, ceil(20% × N))` unique fights. Use Platt below 1,000 unique calibration
  fights and isotonic at or above that threshold.

## Lean tuning and final evaluation — D25

- The final untouched holdout is every UFC event from **January through August
  2026**. It is evaluated once only after August is complete.
- Development uses three to four earlier chronological, event-grouped folds and a
  small fixed set of roughly 12 XGBoost configurations. Choose by mean calibrated
  Brier score; inspect log loss and ECE as supporting checks.
- Once the configuration is locked, do not make further modeling choices from the
  2026 holdout. Report its metrics with event-level bootstrap intervals.

## Market comparison — D24

The market is an external comparator, never a training input. V1 uses a small,
versioned upcoming-fights input (event date, fighter names, weight class) and
automatically matches only an unambiguous normalized Polymarket UFC market.
Unmatched or ambiguous cases are visible for manual review; the system never guesses.

A report uses the most recent capture at or before a supplied `as_of_timestamp` and
retains the snapshot timestamp, matched identities, model probability, market implied
probability, signed difference, and model/data/config versions. A paper signal is a
record of that report, not a wager or a performance claim.

## Retraining cadence — D26

Run one batch retrain after every completed UFC event, only after ingestion and
validation complete cleanly. This keeps the model current while remaining a simple,
auditable batch workflow rather than an online-learning system.

## Model artifacts — D27

MLflow is the sole v1 artifact store. Every training run logs its XGBoost model,
calibrator, feature schema and version, resolved config, metrics, data revision, and
run ID there. The project does not maintain a parallel on-disk model-bundle format.

## Health summary — D40

A single CLI health command emits a structured summary: newest UFC event and data
timestamp, validation result and quarantine census, latest market-capture timestamp,
last successful MLflow run, and active model/feature versions. No dashboard belongs
in v1.

(Renumbered from D28 on 2026-08-08 to resolve a numbering collision; predates decisions D28–D39.)

## Operational choices

- DuckDB is the single datastore; Hydra configures runs; DVC versions data; MLflow
  records experiment provenance.
- The capture cron runs every five minutes on Fly with a persistent volume. Its
  history is irreplaceable; back it up through DVC/Google Drive.

## Market-relative evaluation — D28

Evaluate the market itself as a forecaster: compute the market's Brier score and
log loss on the same fights the model is scored on, and report
`Brier_skill = 1 − (Brier_model / Brier_market)` as the headline metric.

This does not violate the odds-free constraint (D24): market prices are used only
to *evaluate* predictions after the fact, never as a feature or label. Without
this comparison the project cannot answer its own question — "does the model
price fights better than Polymarket?" — because absolute Brier alone says nothing
about relative skill against the market.

Constraint created: the evaluation layer needs historical market-implied
probabilities joined to resolved fights, which requires a persisted
market-to-fight linkage.

## Feature-family ablation ladder — D29

Train and evaluate a sequence of nested models on identical folds, adding one
feature family per rung: (1) naive 50/50 floor, (2) record-only, (3) + physical,
(4) + schedule strength (Elo, PageRank, common opponents), (5) + domain
interactions. Report incremental Brier improvement per family with confidence
intervals. SHAP-only inspection was considered and rejected as the primary tool:
importance measures what the model uses, not what adds incremental predictive
value, and correlated features split credit unpredictably.

Owner rationale (recorded 2026-08-08): this project is about diligently finding
alpha, not rushing. The premise — that domain-expert features carry signal
ordinary bettors miss — is a hypothesis to test, not an assumption. The ladder is
the test: it can prove the domain features add value or honestly show they do
not, and inspection of that answer is the point of the project.

Constraint created: the evaluation harness must parameterize feature subsets, and
each candidate configuration is trained once per rung (~5× training compute,
still minutes at this data scale).

## Kaggle as a secondary source — D30 (conditional)

Kaggle UFC datasets are admitted conditionally for fields UFCStats cannot provide
(pre-UFC records, natural weight class, camp/gym history, training base), pending a
per-field validation protocol: cross-check a sample against UFCStats where they
overlap, audit provenance and update cadence, and test each candidate field for
temporal leakage (many Kaggle UFC datasets embed post-fight or odds-derived
columns). No Kaggle field enters the feature matrix until it passes.

Owner rationale (recorded 2026-08-08): worth checking whether the data quality
supports the debut/pre-UFC features (Section 3a of the registry) before discarding
them, since debut mispricing is part of the alpha thesis; but known leakage in
these datasets means admission must be field-by-field, not wholesale.

Constraint created: the feature-engine spec gains an explicit Kaggle-evaluation
task with pass/fail criteria per field, and every admitted field carries a
documented provenance note.

## Glicko-2 alongside Elo — D31

Add Glicko-2 as a second rating system next to Elo in the graph-derived family
(`glicko2_rating`, `glicko2_rd`, and the matchup deltas). The motivating feature
is the rating deviation (RD): Glicko-2 carries an explicit per-fighter
uncertainty that grows with inactivity and shrinks with fights against known
opposition. High RD marks fighters whose true skill the data does not yet pin
down — disproportionately prelim fighters, which is exactly the regime where the
owner hunts mispricing. Elo alone encodes a point estimate with no uncertainty.

Owner rationale (recorded 2026-08-08): the latent-form/state-space idea is the
right quant instinct but full particle filtering is out of proportion for this
project's horizon; Glicko-2 RD is the cheap, classical version of "uncertainty as
a feature" and is one library away.

Constraint created: rating-system parameters (initial rating/RD, volatility τ,
inactivity handling) join the Elo configuration in `configs/graph.yaml`, and the
D29 ablation ladder's schedule-strength rung includes both rating systems.

## LLM due-diligence layer, annotate-only — D32

The LLM enters the pipeline as a signal-time due-diligence service, never as a
model feature in v1. For flagged mismatch signals only (a handful per card), a
search-augmented LLM runs a fixed pre-fight checklist (injury news, weight-cut
concern, short-notice replacement, camp change, other material news) and returns
a structured verdict — CONFIRM / QUALIFY / VETO — with evidence links and
confidence, validated against a frozen Pydantic schema. Prompt version, model
version, and timestamp are logged with every verdict; runs are idempotent per
(fight, report run). The component carries its own eval harness: a hand-labeled
set of past fights gates the extractor at precision ≥ 0.80 / recall ≥ 0.60, and a
running scoreboard tracks outcomes of CONFIRM'd vs VETO'd signals.

The verdict is **annotate-only**: it ships attached to the signal, and the human
makes the final call. Owner rationale (recorded 2026-08-08): the LLM should not
make the call — there are no evals yet and its live performance is unknown;
suppressing signals automatically would also hide exactly the outcome data needed
to measure whether its vetoes are any good.

Rejected for v1: LLM-extracted training features (no point-in-time historical
news corpus exists; training on retroactively extracted coverage is leakage
theater) and deep-learning supplements (data-starved at ~7,500 fights). The
documented v2 promotion path is forward-collection of timestamped extractions,
per-signal event studies, then a residual overlay trained with
`base_margin = logit(p_base)` — gated on accrued sample size and eval thresholds.

Constraint created: the mismatch-report spec gains the due-diligence service,
its eval harness, and the verdict scoreboard; the design doc records the v2
overlay path as designed-not-built.

## Calibrator selected empirically per fold — D33 (supersedes D22's method rule)

The calibration map is chosen by model selection, not by a fixed rule. On each
development fold, fit Platt scaling (2-parameter logistic on the logit of the raw
score), isotonic regression (non-parametric monotonic step function), and beta
calibration (3-parameter map `sigmoid(a·ln(s) − b·ln(1−s) + c)`, a strict
generalization of Platt) on the calibration slice; select by ECE with Brier as
tiebreaker on the fold's evaluation slice; log all three fits and the winner to
MLflow. Selection happens on development folds only — the 2026 holdout is never
consulted. Beta is the expected winner at this data scale (isotonic overfits
below roughly a thousand points), but the choice is measured, not assumed.

D22's size-based switch (Platt below 1,000 calibration fights, isotonic at or
above) is superseded; the D22 calibration-slice sizing rule
(`max(250, ceil(20% × N))`, trailing and event-complete) still stands.

Calibration quality is additionally reported stratified: reliability diagrams and
calibration error per probability bucket ([0.1–0.3], [0.3–0.5], [0.5–0.7],
[0.7–0.9]) with binomial confidence intervals, because signal-relevant buckets
must be individually trustworthy, and a global ECE can hide tail miscalibration.

Owner rationale (recorded 2026-08-08): this is the same idea as cross-validation —
don't argue from priors about which small model is best; evaluate the candidates
under the same protocol as everything else and let held-out data decide.

Constraint created: the eval harness parameterizes the calibrator, records the
per-fold selection, and the stratified reliability report becomes a standard
artifact of every training run.

## Signal credibility: magnitude gate with bucket transparency — D34

A mismatch row in the report is labeled by a single statistical gate: it passes
when `|p_model − p_market|` exceeds `k ×` the model's measured calibration error
in the probability bucket containing `p_model` (buckets and per-bucket errors
come from the stratified reliability report, D33; `k` defaults to 2 and lives in
Hydra config). Rows below the threshold are labeled as within model noise.

There is no hard support gate. Instead, every row reports the gating bucket's
evidence alongside the verdict: bucket sample count, measured calibration error,
and a binomial confidence interval on that error. A pass earned against a
12-fight bucket is therefore visibly weaker than one earned against a 300-fight
bucket, and the human discounts it accordingly.

Owner rationale (recorded 2026-08-08): same principle as the annotate-only LLM
verdict (D32) — gate on the magnitude that can be measured, but do not
auto-suppress on thin support; report the bucket data alongside the verdict and
keep the final judgment human. Suppression would also hide the outcomes needed
to evaluate the gate itself.

Constraint created: the mismatch report schema carries per-row gate verdict,
bucket id, bucket n, bucket calibration error, and its confidence interval; the
gate multiplier k is config, not code.

## Statistical power disclosure — D35

The evaluation harness ships the full honesty package:

- **Minimum detectable effect.** Before the holdout is scored, compute and report
  the smallest Brier skill the evaluation could detect at α = 0.05, power = 0.8,
  given the holdout's fight count. Results are always presented next to this
  number.
- **Paired permutation test vs the market.** Model and market are scored on the
  same fights, so the comparison is paired: permutation test on per-fight Brier
  differences (no distributional assumptions). This extracts the maximum power
  available from a small sample.
- **Cumulative evidence tracking.** A running Brier-skill-vs-market plot with
  confidence bands that accrues after the holdout as new events resolve, framed
  as accumulating evidence rather than a one-shot verdict.
- **Framing rule.** The project never claims to beat the market unless the paired
  test clears significance; below that bar, results are reported as point
  estimates with intervals and their MDE context.

Owner rationale (recorded 2026-08-08): the holdout (~300–400 fights) cannot
detect a realistic 1–3% edge, and the signal-relevant tail buckets are smaller
still. Reporting what the evaluation can and cannot conclude is what makes the
headline artifact defensible under hostile questioning; a computed MDE and an
honest null beat a suspiciously good number on 300 samples.

Constraint created: `Brier_skill = 1 − Brier_model/Brier_market` becomes the
headline metric (per D28) and every reported result carries MDE, paired-test
p-value, and interval; the eval module gains power-analysis and permutation-test
functions (~50 lines of scipy, fixture-tested).

## Paper evaluation is hold-to-resolution; capture is convergence-ready — D36

v1 evaluates exactly one claim: the model's probabilities are better than the
market's, measured at fight resolution (Brier skill, paired test — D28/D35). The
owner's convergence-exit idea (enter at a large divergence, exit if the price
converges partway before fight night) remains discretionary live behavior and is
explicitly NOT evaluated by v1: it is a claim about price dynamics and fill
realism (path dependency, thin prelim order books, bid/ask spread) that would
require execution simulation, which D23 scopes out.

The capture and paper-signal schemas are nevertheless extended NOW, because this
data cannot be collected retroactively: each signal stores best bid, best ask,
top-of-book depth, and the mid; and post-signal price snapshots are captured at
approximately T+1h, T+4h, T+24h, and fight time. Data collected ≠ claims made;
the extension is what keeps an honest v2 convergence study possible.

Owner rationale (recorded 2026-08-08): option (a) keeps the v1 deliverable
ML-shaped — calibrated probabilities under proper scoring with honest tests —
rather than drifting into trading-strategy research, while preserving the raw
material for later.

Constraint created: the polymarket capture schema gains bid/ask/depth columns
(already partially present in raw order-book JSON — the paper-signal log must
denormalize them per signal), and the report pipeline schedules post-signal
snapshot reads.

## Sparse-history fighters: annotate and measure, never block — D37

Signals involving low-history fighters are never excluded from the report. Every
report row carries `min_prior_ufc_fights` (the lesser of the two fighters' UFC
fight counts) and a `SPARSE_HISTORY` tag when that count falls below a threshold
(default 3, Hydra config). To make the tag evidence-based rather than
decorative, the eval harness reports calibration and Brier stratified by history
depth — fights involving a ≤3-fight fighter versus the rest — so the model's
reliability on sparse fighters is measured, not assumed in either direction.

This composes with existing machinery: sparse fighters carry a high Glicko-2
rating deviation (D31) and typically land in thin calibration buckets whose
support is already surfaced per row (D34).

Owner rationale (recorded 2026-08-08): consistent with the project's reporting
philosophy (D32, D34, D36) — do not block a report on the basis of assumptions
about how the model will behave, because we do not yet know how it acts on these
fighters; annotate, measure, and let the evidence accumulate. Exclusion would
also amputate the debut-mispricing part of the alpha thesis (registry Section 3a)
and delete the data needed to ever evaluate it.

Constraint created: report schema gains `min_prior_ufc_fights` and the
`SPARSE_HISTORY` tag; the eval module gains history-depth-stratified metrics as a
standard artifact.

## Persisted market↔fight linkage — D38

Entity resolution between Polymarket markets and UFC fights is materialized in a
`market_fight_links` table (`fight_url`, `token_id`, `match_status`,
`match_method`, `matched_at`, `reviewed_by`) instead of being recomputed at
report time. Name-matching writes candidate links once; ambiguous outcomes
(`MULTIPLE_CANDIDATES`, `NO_CANDIDATE`) surface for one-time human confirmation.
All evaluation and reporting joins go through the table.

This is a prerequisite for reproducibility of the headline metric: scoring the
market as a forecaster (D28) requires market probabilities joined to resolved
historical fights deterministically — if matching logic changes, past
evaluations must not silently change. Post-signal price paths (D36) equally need
a durable fight↔token mapping.

Owner rationale (recorded 2026-08-08): materializing an entity-resolution result
instead of recomputing it is standard data engineering — the same pattern you'd
apply in any data-heavy SWE project.

## v1 feature scope triage — D39

The v1 feature registry is the owner's original registry filtered to what is
computable, temporally safe, and in D23 scope:

- **In:** physical profile (§1), activity (§2), record (§3) with the
  ufcstats-computable subset of debut context (§3a), finishing profile (§4/4a),
  output and efficiency (§5), output variance by card position (§5a — gated on
  verifying the scraper captures card position), experience (§6), weight-class
  migration and weight-bully interaction (§8a/8b), graph features (§9a Elo,
  §9b PageRank, §9c common opponents) plus Glicko-2 (D31), matchup deltas and
  style interactions (§10/10a/10b), rematch minus the LLM field (§11), and the
  deterministic weight-cut/short-notice fields from §15 where sourceable.
- **Kaggle-gated (D30):** pre-UFC record fields, natural weight class, camp/gym
  history (§14 deterministic fields), training-base/travel fields (§12) — each
  enters only after passing the per-field validation protocol.
- **Excluded from v1 — §9d opponent trajectory.** The registry's own leakage
  trap: it consumes opponents' post-fight results, its as-of semantics carry an
  unresolved TODO(human), and it requires dedicated leakage tests. Owner call
  (recorded 2026-08-08): the cost/risk ratio is unacceptable while the core
  pipeline does not yet exist; revisit in v2 with the dedicated test design.
- **Excluded from the model — LLM-extracted features (§13/§15 qualitative, and
  §3a/§11/§14 LLM fields):** superseded by the annotate-only due-diligence layer
  (D32) and its v2 promotion path.
- **Strategy-side only — §16 market-derived fields:** never model inputs; the
  underlying data is preserved by the D36 capture extension.

Constraint created: `.kiro/specs/FEATURES.md` is regenerated from the owner's legacy
registry plus this triage — definitions carried verbatim where still applicable,
nothing invented beyond owner-approved additions (Glicko-2), exclusions marked
with their decision numbers.
